'''
This file is used to train the neural networks that predict the spectrum 
given any set of stellar labels (stellar parameters + elemental abundances). 

Note that, the approach here is slightly different from Ting+18. Instead of 
training individual small networks for each pixel separately, we train a single
 large network for all the pixels simultaneously. 

The advantage of doing so is that the information in the adjacent pixels could 
cross talks, and this leads to more precise interpolation of spectral models.

However to train a large network as such, GPUs are needed, and this code will 
only run with GPU. But even for a simple inexpensive GPU (GTX 1060), this code 
should be pretty fast, and for any normal grid of spectral models with 
1000-10000 training spectra, with > 10 labels, it should not take more than 
a day to train

The default training set are synthetic spectra that have been convolved to the 
appropriate R (~22500 for APOGEE) with the APOGEE LSF.
'''

from __future__ import absolute_import, division, print_function # python2 compatibility
import numpy as np
import sys
import os
import torch
from torch.autograd import Variable

# run on cuda
dtype = torch.cuda.FloatTensor
torch.set_default_tensor_type('torch.cuda.FloatTensor')

# path on cluster to training set. 
training_set_path = 'kurucz_training_spectra.npz'

# size of training set. Anything over a few 100 should be OK for a small network
# (see YST's paper), but it can't hurt to go larger if the training set is available. 
n_train = 2000

# how long to train each pixel until convergence. 1e5 is good for the specific NN
# architecture and learning I used by default, but bigger networks take more steps,
# and decreasing the learning rate will also change this. You can get a sense of 
# how many steps are needed for a new NN architecture by
# plotting the loss function evaluated on both the training set and a validation
# set as a function of step number. It should plateau once the NN is converged. 
num_steps_to_converge = 1e5

# this is also tunable, but 0.001 seems to work well for most use cases. Again, diagnose
# with a validation set if you change this. 
learning_rate = 0.001

# restore training spectra
temp = np.load(training_set_path)
x = (temp["labels"].T)[:n_train,:]
y = temp["spectra"][:n_train,:]

# and validation spectra
x_valid = (temp["labels"].T)[n_train:,:]
y_valid = temp["spectra"][n_train:,:]
temp.close()

x_max = np.max(x, axis = 0)
x_min = np.min(x, axis = 0)
x = (x - x_min)/(x_max - x_min) - 0.5
x_valid = (x_valid-x_min)/(x_max-x_min) - 0.5

# dimension of the input
dim_in = x.shape[1]

# dimension of the output
num_pixel = y.shape[1]

# define neural network
model = torch.nn.Sequential(
        torch.nn.Linear(dim_in, 300),
        torch.nn.Sigmoid(),
        torch.nn.Linear(300, 300),
        torch.nn.Sigmoid(),
        torch.nn.Linear(300,num_pixel)
)
model.cuda()

# assume L2 loss, can also switch to L1Loss 
loss_fn = torch.nn.MSELoss(reduction = 'mean')
#loss_fn = torch.nn.L1Loss(reduction = 'mean')
        
# make pytorch variables
x = Variable(torch.from_numpy(x)).type(dtype)
y = Variable(torch.from_numpy(y), requires_grad=False).type(dtype)
x_valid = Variable(torch.from_numpy(x_valid)).type(dtype)
y_valid = Variable(torch.from_numpy(y_valid),\
                   requires_grad=False).type(dtype)

# weight_decay is for regularization. Not required, but one can play with it. 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 0)

# train the neural network
t = 0
current_loss = np.inf
loss_array =[]
loss_valid_array = []
while t < num_steps_to_converge:
    y_pred = model(x)
    loss = loss_fn(y_pred, y)*1e4
    y_pred_valid = model(x_valid)
    loss_valid = loss_fn(y_pred_valid, y_valid)*1e4
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    t += 1

    # print loss function to monitor
    if t % 1000 == 0:
        loss_data = loss.data.item()
        loss_valid_data = loss_valid.data.item()
        loss_array.append(loss_data)
        loss_valid_array.append(loss_valid_data)
        print('Step ' + str(t) \
        + ': Training set loss = ' + str(int(loss_data*1000.)/1000.) \
        + ' / Validation set loss = ' + str(int(loss_valid_data*1000.)/1000.))
 
        # record the weights and biases if the validation set loss improves
        if loss_valid_data < current_loss:
            current_loss = loss_valid
            model_numpy = []
            for param in model.parameters():
                model_numpy.append(param.data.cpu().numpy())
 
# extract the weights and biases
w_array_0 = model_numpy[0]
b_array_0 = model_numpy[1]
w_array_1 = model_numpy[2]
b_array_1 = model_numpy[3]
w_array_2 = model_numpy[4]
b_array_2 = model_numpy[5]

# save parameters and remember how we scaled the labels
np.savez("NN_normalized_spectra.npz",\
         w_array_0 = w_array_0,\
         w_array_1 = w_array_1,\
         w_array_2 = w_array_2,\
         b_array_0 = b_array_0,\
         b_array_1 = b_array_1,\
         b_array_2 = b_array_2,\
         x_max=x_max,\
         x_min=x_min,\
         loss_valid_array = loss_valid_array,\
         loss_array = loss_array)
