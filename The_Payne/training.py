'''
This file is used to train the neural net that predicts the spectrum
given any set of stellar labels (stellar parameters + elemental abundances).

Note that, the approach here is slightly different from Ting+19. Instead of
training individual small networks for each pixel separately, here we train a single
large network for all pixels simultaneously.

The advantage of doing so is that individual pixels will exploit information
from adjacent pixels. This usually leads to more precise interpolations.

However to train a large network, GPU is needed. This code will
only run with GPU. But even with an inexpensive GPU, this code
should be pretty efficient -- training with a grid of 10,000 training spectra,
with > 10 labels, should not take more than a few hours

The default training set are the Kurucz synthetic spectral models and have been
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
# simple multi-layer perceptron model
# class Payne_model(torch.nn.Module):
#     def __init__(self, dim_in, num_neurons, num_features, mask_size, num_pixel):
#         super(Payne_model, self).__init__()
#         self.features = torch.nn.Sequential(
#             torch.nn.Linear(dim_in, num_neurons),
#             torch.nn.LeakyReLU(),
#             torch.nn.Linear(num_neurons, num_neurons),
#             torch.nn.LeakyReLU(),
#             torch.nn.Linear(num_neurons, num_pixel),
#         )
#
#     def forward(self, x):
#         return self.features(x)

#---------------------------------------------------------------------------------------------------
# resnet models
# class Payne_model(torch.nn.Module):
#     def __init__(self, dim_in, num_neurons, num_features, mask_size, num_pixel):
#         super(Payne_model, self).__init__()
#         self.features = torch.nn.Sequential(
#             torch.nn.Linear(dim_in, num_neurons),
#             torch.nn.LeakyReLU(),
#             torch.nn.Linear(num_neurons, num_neurons),
#             torch.nn.LeakyReLU(),
#             torch.nn.Linear(num_neurons, num_features),
#         )
#
#         self.deconv1 = torch.nn.ConvTranspose1d(64, 64, mask_size, stride=3, padding=5)
#         self.deconv2 = torch.nn.ConvTranspose1d(64, 64, mask_size, stride=3, padding=5)
#         self.deconv3 = torch.nn.ConvTranspose1d(64, 64, mask_size, stride=3, padding=5)
#         self.deconv4 = torch.nn.ConvTranspose1d(64, 64, mask_size, stride=3, padding=5)
#         self.deconv5 = torch.nn.ConvTranspose1d(64, 64, mask_size, stride=3, padding=5)
#         self.deconv6 = torch.nn.ConvTranspose1d(64, 32, mask_size, stride=3, padding=5)
#         self.deconv7 = torch.nn.ConvTranspose1d(32, 1, mask_size, stride=3, padding=5)
#
#         self.deconv2b = torch.nn.ConvTranspose1d(64, 64, 1, stride=3)
#         self.deconv3b = torch.nn.ConvTranspose1d(64, 64, 1, stride=3)
#         self.deconv4b = torch.nn.ConvTranspose1d(64, 64, 1, stride=3)
#         self.deconv5b = torch.nn.ConvTranspose1d(64, 64, 1, stride=3)
#         self.deconv6b = torch.nn.ConvTranspose1d(64, 32, 1, stride=3)
#
#         self.relu2 = torch.nn.LeakyReLU()
#         self.relu3 = torch.nn.LeakyReLU()
#         self.relu4 = torch.nn.LeakyReLU()
#         self.relu5 = torch.nn.LeakyReLU()
#         self.relu6 = torch.nn.LeakyReLU()
#
#         self.num_pixel = num_pixel
#
#     def forward(self, x):
#         x = self.features(x)[:,None,:]
#         x = x.view(x.shape[0], 64, 3)
#         x1 = self.deconv1(x)
#
#         x2 = self.deconv2(x1)
#         x2 += self.deconv2b(x1)
#         x2 = self.relu2(x2)
#
#         x3 = self.deconv3(x2)
#         x3 += self.deconv3b(x2)
#         x3 = self.relu2(x3)
#
#         x4 = self.deconv4(x3)
#         x4 += self.deconv4b(x3)
#         x4 = self.relu2(x4)
#
#         x5 = self.deconv5(x4)
#         x5 += self.deconv5b(x4)
#         x5 = self.relu2(x5)
#
#         x6 = self.deconv6(x5)
#         x6 += self.deconv6b(x5)
#         x6 = self.relu2(x6)
#
#         x7 = self.deconv7(x6)[:,0,:self.num_pixel]
#         return x7


#---------------------------------------------------------------------------------------------------
# define network
class Payne_model(torch.nn.Module):
    def __init__(self, dim_in, num_neurons, num_features, mask_size, num_pixel):
        super(Payne_model, self).__init__()
        layers = []
        channel = 256

        for i in range(6):
            for j in range(2):

                if i == 0 and j == 0:
                    layers.append(torch.nn.ConvTranspose1d(dim_in, channel, 7, stride=1))
                    layers.append(torch.nn.BatchNorm1d(channel, momentum=0.001, affine=False))
                    layers.append(torch.nn.LeakyReLU(0.2, inplace=True))
                else:
                    layers.append(torch.nn.Conv1d(channel, channel, 5, stride=1, padding=2))
                    layers.append(torch.nn.BatchNorm1d(channel, momentum=0.001, affine=False))
                    layers.append(torch.nn.LeakyReLU(0.2, inplace=True))

            if i < 5:
                layers.append(torch.nn.Upsample(scale_factor=4, mode='linear', align_corners = False))
            else:
                layers.append(torch.nn.Conv1d(channel, 1, 6, stride=1))
                layers.append(torch.nn.LeakyReLU())

        self.model = torch.nn.Sequential(*layers)
        self.add_module("model", self.model)

    def forward(self, z):
        return self.model(z[:,:,None])[:,0,:6097]


#===================================================================================================
# train neural networks
def neural_net(training_labels, training_spectra, validation_labels, validation_spectra,\
             num_neurons = 300, num_steps=1e4, learning_rate=1e-4, batch_size=512,\
             num_features = 64*3, mask_size=11, num_pixel=4375):

    '''
    Training a neural net to emulate spectral models

    training_labels has the dimension of [# training spectra, # stellar labels]
    training_spectra has the dimension of [# training spectra, # wavelength pixels]

    The validation set is used to independently evaluate how well the neural net
    is emulating the spectra. If the neural network overfits the spectral variation, while
    the loss will continue to improve for the training set, but the validation
    set should show a worsen loss.

    The training is designed in a way that it always returns the best neural net
    before the network starts to overfit (gauged by the validation set).

    num_steps = how many steps to train until convergence.
    1e4 is good for the specific NN architecture and learning I used by default.
    Bigger networks will take more steps to converge, and decreasing the learning rate
    will also change this. You can get a sense of how many steps are needed for a new
    NN architecture by plotting the loss evaluated on both the training set and
    a validation set as a function of step number. It should plateau once the NN
    has converged.

    learning_rate = step size to take for gradient descent
    This is also tunable, but 1e-4 seems to work well for most use cases. Again,
    diagnose with a validation set if you change this.

    num_features is the number of features before the deconvolutional layers; it only
    applies if ResNet is used. For the simple multi-layer perceptron model, this parameter
    is not used. We truncate the predicted model if the output number of pixels is
    larger than what is needed. In the current default model, the output is ~8500 pixels
    in the case where the number of pixels is > 8500, increase the number of features, and
    tweak the ResNet model accordingly

    batch_size = the batch size for training the neural networks during the stochastic
    gradient descent. A larger batch_size reduces stochasticity, but it might also
    risk of stucking in local minima

    '''

    # run on cuda
    dtype = torch.cuda.FloatTensor
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # scale the labels, optimizing neural networks is easier if the labels are more normalized
    x_max = np.max(training_labels, axis = 0)
    x_min = np.min(training_labels, axis = 0)
    x = (training_labels - x_min)/(x_max - x_min) - 0.5
    x_valid = (validation_labels-x_min)/(x_max-x_min) - 0.5

    # dimension of the input
    dim_in = x.shape[1]

#--------------------------------------------------------------------------------------------
    # assume L1 loss
    loss_fn = torch.nn.L1Loss(reduction = 'mean')

    # make pytorch variables
    x = Variable(torch.from_numpy(x)).type(dtype)
    y = Variable(torch.from_numpy(training_spectra), requires_grad=False).type(dtype)
    x_valid = Variable(torch.from_numpy(x_valid)).type(dtype)
    y_valid = Variable(torch.from_numpy(validation_spectra), requires_grad=False).type(dtype)

    # initiate Payne and optimizer
    model = Payne_model(dim_in, num_neurons, num_features, mask_size, num_pixel)
    model.cuda()
    model.train()

    # we adopt rectified Adam for the optimization
    optimizer = radam.RAdam([p for p in model.parameters() if p.requires_grad==True], lr=learning_rate)

#--------------------------------------------------------------------------------------------
    # train in batches
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

        # for each batch, calculate the gradient with respect to the loss
        for i in range(nbatches):
            idx = perm[i * batch_size : (i+1) * batch_size]
            y_pred = model(x[idx])

            loss = loss_fn(y_pred, y[idx])*1e4
            optimizer.zero_grad()
            loss.backward(retain_graph=False)
            optimizer.step()

#-------------------------------------------------------------------------------------------------------
        # evaluate validation loss
        if e % 100 == 0:

            # here we also break into batches because when training ResNet
            # evaluating the whole validation set could go beyond the GPU memory
            # if needed, this part can be simplified to reduce overhead
            perm_valid = torch.randperm(nsamples_valid)
            perm_valid = perm_valid.cuda()
            loss_valid = 0

            for j in range(nbatches_valid):
                idx = perm_valid[j * batch_size : (j+1) * batch_size]
                y_pred_valid = model(x_valid[idx])
                loss_valid += loss_fn(y_pred_valid, y_valid[idx])*1e4
            loss_valid /= nbatches_valid

            print('iter %s:' % e, 'training loss = %.3f' % loss,\
                 'validation loss = %.3f' % loss_valid)

            loss_data = loss.detach().data.item()
            loss_valid_data = loss_valid.detach().data.item()
            training_loss.append(loss_data)
            validation_loss.append(loss_valid_data)

#--------------------------------------------------------------------------------------------
            # record the weights and biases if the validation loss improves
            if loss_valid_data < current_loss:
                current_loss = loss_valid_data

                # model_numpy = []
                # for param in model.parameters():
                #     model_numpy.append(param.data.cpu().numpy())

                ### for resnet ###
                state_dict =  model.state_dict()
                for k, v in state_dict.items():
                    state_dict[k] = v.cpu()
                torch.save(state_dict, 'NN_normalized_spectra.pt')

                np.savez("training_loss.npz",\
                        training_loss = training_loss,\
                        validation_loss = validation_loss)

#--------------------------------------------------------------------------------------------
    # # extract the weights and biases
    # w_array_0 = model_numpy[0]
    # b_array_0 = model_numpy[1]
    # w_array_1 = model_numpy[2]
    # b_array_1 = model_numpy[3]
    # w_array_2 = model_numpy[4]
    # b_array_2 = model_numpy[5]
    #
    # # save parameters and remember how we scaled the labels
    # np.savez("NN_normalized_spectra.npz",\
    #          w_array_0 = w_array_0,\
    #          w_array_1 = w_array_1,\
    #          w_array_2 = w_array_2,\
    #          b_array_0 = b_array_0,\
    #          b_array_1 = b_array_1,\
    #          b_array_2 = b_array_2,\
    #          x_max=x_max,\
    #          x_min=x_min,)

    # save the final training loss
    np.savez("training_loss.npz",\
             training_loss = training_loss,\
             validation_loss = validation_loss)

    return
