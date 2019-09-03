'''
This file is used to train the neural networks that predict the spectrum
given any set of stellar labels (stellar parameters + elemental abundances).

Note that, the approach here is different from Ting+19. Instead of
training individual small MLP networks for each pixel separately, we train a single
large ResNet network for all pixels simultaneously.

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

        self.deconv1 = torch.nn.ConvTranspose1d(8, 64, mask_size, stride=2)
        self.deconv2 = torch.nn.ConvTranspose1d(64, 64, mask_size, stride=2)

        self.deconv3 = torch.nn.ConvTranspose1d(64, 64, mask_size, stride=2, padding=1)
        self.deconv4 = torch.nn.ConvTranspose1d(64, 64, mask_size, stride=2, padding=1)
        self.deconv5 = torch.nn.ConvTranspose1d(64, 64, mask_size, stride=2)
        self.deconv6 = torch.nn.ConvTranspose1d(64, 64, mask_size, stride=2, output_padding=1)
        self.deconv7 = torch.nn.ConvTranspose1d(64, 1, mask_size, stride=2)

        self.deconv2b = torch.nn.Sequential(
                            torch.nn.ConvTranspose1d(64, 64, 1, stride=2, output_padding=10),\
                            torch.nn.BatchNorm1d(64)
                        )
        self.deconv3b = torch.nn.Sequential(
                            torch.nn.ConvTranspose1d(64, 64, 1, stride=2, padding=1, output_padding=10),\
                            torch.nn.BatchNorm1d(64)
                        )
        self.deconv4b = torch.nn.Sequential(
                            torch.nn.ConvTranspose1d(64, 64, 1, stride=2, padding=1, output_padding=10),\
                            torch.nn.BatchNorm1d(64)
                        )
        self.deconv5b = torch.nn.Sequential(
                            torch.nn.ConvTranspose1d(64, 64, 1, stride=2, output_padding=10),\
                            torch.nn.BatchNorm1d(64)
                        )
        self.deconv6b = torch.nn.Sequential(
                            torch.nn.ConvTranspose1d(64, 64, 1, stride=2, output_padding=11),\
                            torch.nn.BatchNorm1d(64)
                        )

        self.batch_norm1 = torch.nn.Sequential(
                            torch.nn.BatchNorm1d(64),
                            torch.nn.LeakyReLU()
        )
        self.batch_norm2 = torch.nn.Sequential(
                            torch.nn.BatchNorm1d(64),
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
                            torch.nn.BatchNorm1d(64),
                            torch.nn.LeakyReLU()
        )
        self.batch_norm6 = torch.nn.Sequential(
                            torch.nn.BatchNorm1d(64),
                            torch.nn.LeakyReLU()
        )
        self.batch_norm7 = torch.nn.Sequential(
                            torch.nn.BatchNorm1d(1),
                            torch.nn.LeakyReLU()
        )

        self.relu2 = torch.nn.LeakyReLU()
        self.relu3 = torch.nn.LeakyReLU()
        self.relu4 = torch.nn.LeakyReLU()
        self.relu5 = torch.nn.LeakyReLU()
        self.relu6 = torch.nn.LeakyReLU()

    def forward(self, x):
        x = self.features(x)[:,None,:]
        x = x.view(x.shape[0], 8, 52)
        x = self.deconv1(x)
        x1 = self.batch_norm1(x)

        x2 = self.deconv2(x1)
        x2 = self.batch_norm2(x2)
        x2 += self.deconv2b(x1)
        x2 = self.relu2(x2)

        x3 = self.deconv3(x2)
        x3 = self.batch_norm3(x3)
        x3 += self.deconv3b(x2)
        x3 = self.relu2(x3)

        x4 = self.deconv4(x3)
        x4 = self.batch_norm4(x4)
        x4 += self.deconv4b(x3)
        x4 = self.relu2(x4)

        x5 = self.deconv5(x4)
        x5 = self.batch_norm5(x5)
        x5 += self.deconv5b(x4)
        x5 = self.relu2(x5)

        x6 = self.deconv6(x5)
        x6 = self.batch_norm6(x6)
        x6 += self.deconv6b(x5)
        x6 = self.relu2(x6)

        x7 = self.deconv7(x6)
        x7 = self.batch_norm7(x7)[:,0,:]
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
    
    Here we consider a multilayer ResNet. [more detail soon]

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
    np.savez("NN_scaling_2.npz", x_min=x_min, x_max=x_max)

    # dimension of the input
    dim_in = x.shape[1]

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

    nsamples_valid = x_valid.shape[0]
    nbatches_valid = nsamples_valid // batch_size

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
            loss.backward(retain_graph=False)
            optimizer.step()

        # the average loss.
        if e % 100 == 0:

            # randomly permute the data
            perm_valid = torch.randperm(nsamples_valid)
            perm_valid = perm_valid.cuda()
            loss_valid = 0

            for j in range(nbatches_valid):
                idx = perm_valid[j * batch_size : (j+1) * batch_size]
                y_pred_valid = model(x_valid[idx])
                loss_valid += loss_fn(y_pred_valid, y_valid[idx])*1e4
            loss_valid /= nbatches_valid

            # adopt the nmf representation
            #y_nmf_valid = model(x_valid)
            #y_nmf_valid = (y_nmf_valid*std_nmf) + mu_nmf
            #y_pred_valid = torch.mm(y_nmf_valid, nmf_components)

            print('iter %s:' % e, 'training loss = %.3f' % loss,\
                 'validation loss = %.3f' % loss_valid)

            loss_data = loss.detach().data.item()
            loss_valid_data = loss_valid.detach().data.item()
            training_loss.append(loss_data)
            validation_loss.append(loss_valid_data)

            # record the weights and biases if the validation loss improves
            if loss_valid_data < current_loss:
                current_loss = loss_valid_data

                state_dict =  model.state_dict()
                for k, v in state_dict.items():
                    state_dict[k] = v.cpu()
                torch.save(state_dict, 'NN_normalized_spectra_2.pt')

                np.savez("training_loss_2.npz",\
                         training_loss = training_loss,\
                         validation_loss = validation_loss)

            # clear cache to save memory
            torch.cuda.empty_cache()

            # check allocated memory
            print('current memory allocated: {}'.format(torch.cuda.memory_allocated() / 1024 ** 2))
            print('max memory allocated: {}'.format(torch.cuda.max_memory_allocated() / 1024 ** 2))
            print('cached memory: {}'.format(torch.cuda.memory_cached() / 1024 ** 2))

#--------------------------------------------------------------------------------------------
    # save the final training loss
    np.savez("training_loss_2.npz",\
             training_loss = training_loss,\
             validation_loss = validation_loss)

    return
