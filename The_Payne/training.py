'''
This file is used to train the neural networks that predict the spectrum
given any set of stellar labels (stellar parameters + elemental abundances).

Note that, the approach here is slightly different from Ting+18. Instead of
training individual small networks for each pixel separately, we train a single
large network for all pixels simultaneously.

The advantage of doing so is that individual pixels could exploit information
from the adjacent pixel. This usually leads to more precise interpolations of
spectral models.

However to train a large network, GPUs are needed, and this code will
only run with GPUs. But even for a simple, inexpensive, GPU (GTX 1060), this code
should be pretty efficient -- for any grid of spectral models with 1000-10000
training spectra and > 10 labels, it should not take more than a day to train

The default training set are synthetic spectra the Kurucz models and have been
convolved to the appropriate R (~22500 for APOGEE) with the APOGEE LSF.
'''

from __future__ import absolute_import, division, print_function # python2 compatibility
import numpy as np
import sys
import os
import torch
import time
from torch.autograd import Variable
from . import radam
import gc


#===================================================================================================
# define container
class Payne_model(torch.nn.Module):
    def __init__(self, dim_in, num_neurons, num_features, mask_size):
        super(Payne_model, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Linear(dim_in, num_neurons),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(num_neurons, num_neurons),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(num_neurons, num_features),
        )

        self.deconv1 = torch.nn.ConvTranspose1d(8, 32, mask_size, stride=2)
        self.deconv2 = torch.nn.ConvTranspose1d(32, 32, mask_size, stride=2)
        self.deconv3 = torch.nn.ConvTranspose1d(32, 64, mask_size, stride=2, padding=1)
        self.deconv4 = torch.nn.ConvTranspose1d(64, 64, mask_size, stride=2, padding=1)
        self.deconv5 = torch.nn.ConvTranspose1d(64, 32, mask_size, stride=2)
        self.deconv6 = torch.nn.ConvTranspose1d(32, 32, mask_size, stride=2, output_padding=1)
        self.deconv7 = torch.nn.ConvTranspose1d(32, 1, mask_size, stride=2)

        self.batch_norm1 = torch.nn.Sequential(
                            torch.nn.BatchNorm1d(32),
                            torch.nn.LeakyReLU()
        )
        self.batch_norm2 = torch.nn.Sequential(
                            torch.nn.BatchNorm1d(32),
                            torch.nn.LeakyReLU()
        )
        self.batch_norm3 = torch.nn.Sequential(
                            torch.nn.BatchNorm1d(64),
                            torch.nn.LeakyReLU()
        )
        self.batch_norm4 = torch.nn.Sequential(
                            torch.nn.BatchNorm1d(64),
                            torch.nn.LeakyReLU()
        )
        self.batch_norm5 = torch.nn.Sequential(
                            torch.nn.BatchNorm1d(32),
                            torch.nn.LeakyReLU()
        )
        self.batch_norm6 = torch.nn.Sequential(
                            torch.nn.BatchNorm1d(32),
                            torch.nn.LeakyReLU()
        )
        self.batch_norm7 = torch.nn.Sequential(
                            torch.nn.BatchNorm1d(1),
                            torch.nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.features(x)[:,None,:]
        x = x.view(x.shape[0], 8, 52)
        x = self.deconv1(x)
        x = self.batch_norm1(x)
        x = self.deconv2(x)
        x = self.batch_norm2(x)
        x = self.deconv3(x)
        x = self.batch_norm3(x)
        x = self.deconv4(x)
        x = self.batch_norm4(x)
        x = self.deconv5(x)
        x = self.batch_norm5(x)
        x = self.deconv6(x)
        x = self.batch_norm6(x)
        x = self.deconv7(x)
        x = self.batch_norm7(x)[:,0,:]
        return x


#===================================================================================================
# train neural networks
def neural_net(training_labels, training_spectra, validation_labels, validation_spectra,\
             num_neurons = 300, num_steps=1e4, learning_rate=1e-4, batch_size=512,\
             num_features = 280, mask_size=11):

    '''
    Training neural networks to emulate spectral models

    training_labels has the dimension of [# training spectra, # stellar labels]
    training_spectra has the dimension of [# training spectra, # wavelength pixels]

    The validation set is used to independently evaluate how well the neural networks
    are emulating the spectra. If the networks overfit the spectral variation, while
    the loss function will continue to improve for the training set, but the validation
    set should show a worsen loss function.

    The training is designed in a way that it always returns the best neural networks
    before the networks start to overfit (gauged by the validation set).

    num_neurons = number of neurons per hidden layer in the neural networks.
    We assume a 2 hidden-layer neural network.
    Increasing this number increases the complexity of the network. This could be helpful
    in capturing a more subtle variation of flux as a function of stellar labels, but
    increasing the complexity could also lead to overfitting. It is also slower
    to train with a larger network.

    num_steps = how many steps to train until convergence.
    1e5 is good for the specific NN architecture and learning I used by default,
    but bigger networks take more steps, and decreasing the learning rate will
    also change this. You can get a sense of how many steps are needed for a new
    NN architecture by plotting the loss function evaluated on both the training set
    and a validation set as a function of step number. It should plateau once the NN
    has converged.

    learning_rate = step size to take for gradient descent
    This is also tunable, but 0.001 seems to work well for most use cases. Again,
    diagnose with a validation set if you change this.

    batch_size = the batch size for training the neural networks during the stochastic
    gradient descent. A larger batch_size reduces the stochasticity, but it might also
    risk to stuck in a local minimum

    returns:
        training loss and validation loss (per 1000 steps)
        the codes also outputs a numpy saved array ""NN_normalized_spectra.npz"
        which can be imported and substitutes the default neural networks (see tutorial)
    '''

    # run on cuda
    dtype = torch.cuda.FloatTensor
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # scale the labels, optimizing neural networks is easier if the labels are more normalized
    x_max = np.max(training_labels, axis = 0)
    x_min = np.min(training_labels, axis = 0)
    x = (training_labels - x_min)/(x_max - x_min) - 0.5
    x_valid = (validation_labels-x_min)/(x_max-x_min) - 0.5

    # save scaling relation
    np.savez("NN_scaling.npz", x_min=x_min, x_max=x_max)

    # dimension of the input
    dim_in = x.shape[1]

    # dimension of the output
    #num_pixel = training_spectra.shape[1]

#--------------------------------------------------------------------------------------------
    # restore the NMF components
    #temp = np.load("nmf_components.npz")
    #nmf_components = temp["nmf_components"]
    #mu_nmf = temp["mu_Y"]
    #std_nmf = temp["std_Y"]
    #num_pixel = nmf_components.shape[0]

#--------------------------------------------------------------------------------------------
    # assume L2 loss
    loss_fn = torch.nn.L1Loss(reduction = 'mean')

    # make pytorch variables
    x = Variable(torch.from_numpy(x)).type(dtype)
    y = Variable(torch.from_numpy(training_spectra), requires_grad=False).type(dtype)
    x_valid = Variable(torch.from_numpy(x_valid)).type(dtype)
    y_valid = Variable(torch.from_numpy(validation_spectra), requires_grad=False).type(dtype)

    # initiate Payne and optimizer
    model = Payne_model(dim_in, num_neurons, num_features, mask_size)
    model.cuda()
    model.train()

    #optimizer = radam.RAdam(model.parameters(), lr=learning_rate, weight_decay = 0)
    optimizer = radam.RAdam([p for p in model.parameters() if p.requires_grad==True], lr=learning_rate)

#--------------------------------------------------------------------------------------------
    # make NMF components pytorch variables as well
    #nmf_components = Variable(torch.from_numpy(nmf_components), requires_grad=False).type(dtype)
    #mu_nmf = Variable(torch.from_numpy(mu_nmf), requires_grad=False).type(dtype)
    #std_nmf = Variable(torch.from_numpy(std_nmf), requires_grad=False).type(dtype)

#--------------------------------------------------------------------------------------------
    # break into batches
    nsamples = x.shape[0]
    nbatches = nsamples // batch_size

    # initiate counter
    current_loss = np.inf
    training_loss =[]
    validation_loss = []

#-------------------------------------------------------------------------------------------------------
    # train the network
    for e in range(int(num_steps)):

        # randomly permute the data
        perm = torch.randperm(nsamples)
        perm = perm.cuda()

        # For each batch, calculate the gradient with respect to the loss and take
        # one step.
        for i in range(nbatches):
            idx = perm[i * batch_size : (i+1) * batch_size]
            y_pred = model(x[idx])

            # adopt the nmf representation
            #y_nmf = model(x[idx])
            #y_nmf = (y_nmf*std_nmf) + mu_nmf
            #y_pred = torch.mm(y_nmf, nmf_components)

            loss = loss_fn(y_pred, y[idx])*1e4
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        # garbage collect
        gc.collect()

        # the average loss.
        if e % 100 == 0:
            y_pred_valid = model(x_valid)

            # adopt the nmf representation
            #y_nmf_valid = model(x_valid)
            #y_nmf_valid = (y_nmf_valid*std_nmf) + mu_nmf
            #y_pred_valid = torch.mm(y_nmf_valid, nmf_components)

            loss_valid = loss_fn(y_pred_valid, y_valid)*1e4
            print('iter %s:' % e, 'training loss = %.3f' % loss,\
                 'validation loss = %.3f' % loss_valid)

            loss_data = loss.data.item()
            loss_valid_data = loss_valid.data.item()
            training_loss.append(loss_data)
            validation_loss.append(loss_valid_data)

            # record the weights and biases if the validation loss improves
            if loss_valid_data < current_loss:
                current_loss = loss_valid
                torch.save(model, 'NN_normalized_spectra.pt')

                #model_numpy = []
                #for param in model.parameters():
                #    model_numpy.append(param.data.cpu().numpy())

#--------------------------------------------------------------------------------------------
    # extract the weights and biases
    #w_array_0 = model_numpy[0]
    #b_array_0 = model_numpy[1]
    #w_array_1 = model_numpy[2]
    #b_array_1 = model_numpy[3]
    #w_array_2 = model_numpy[4]
    #b_array_2 = model_numpy[5]

    # save parameters and remember how we scaled the labels
    #np.savez("NN_normalized_spectra.npz",\
    #         w_array_0 = w_array_0,\
    #         w_array_1 = w_array_1,\
    #         w_array_2 = w_array_2,\
    #         b_array_0 = b_array_0,\
    #         b_array_1 = b_array_1,\
    #         b_array_2 = b_array_2,\
    #         x_max=x_max,\
    #         x_min=x_min,\
    #         training_loss = training_loss,\
    #         validation_loss = validation_loss)

    return training_loss, validation_loss
