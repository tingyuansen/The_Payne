'''
This file contains code to train the neural network that predicts Teff2 and logg2 from
Teff1, logg1, and [Fe/H]. In the default version, the training set only consists of
dwarf-dwarf binaries. 

The training set was constructed by drawing random pairs of stars with the same age 
and metallicity but different mass from a grid of MIST isochrones spanning the range
of Teff, logg, and metallicity where the spectral model works. 

If you train a different version on different isochrones (e.g. to work on giants), it's
critical check with cross-validation that it behaves as expected. Also important to 
check that the training set contains mock binaries throughout the regions of label space
where the model will be used. 

runs in ~15 minutes on my laptop. 
'''
from __future__ import absolute_import, division, print_function # python2 compatibility
import numpy as np
import sys
import os
import torch
from torch.autograd import Variable

# take 15000 of the 20000 as the training set. The other 5000 can be used for 
# cross validation. 
n_train = 15000
learning_rate = 0.001 # can play with this, but 0.001 is reasonable for this NN
num_steps_to_converge = 150000 # can play with this too, but this is reasonable


path = '../other_data/labels_for_20000_mock_dwarf_dwarf_binaries.npz'
tmp = np.load(path)
mock_labels = tmp['mock_labels']
tmp.close()

# mock_labels = [Teff_1, logg_1, Fe/H, alpha/Fe, mass_1, age, mass_2, Teff_2, logg_2]
# sorted such that mass_1 > mass_2, always
all_Teff1 = mock_labels[:, 0]
all_logg1 = mock_labels[:, 1]
all_feh = mock_labels[:, 2]
all_Teff2 = mock_labels[:, 7]
all_logg2 = mock_labels[:, 8]
all_q = mock_labels[:, 6]/mock_labels[:, 4]

# randomize 
np.random.seed(0)
ind = np.arange(len(all_Teff1))
np.random.shuffle(ind)
all_Teff1, all_logg1, all_Teff2, all_logg2, all_q, all_feh = all_Teff1[ind], \
    all_logg1[ind], all_Teff2[ind], all_logg2[ind], all_q[ind], all_feh[ind]

# input and output labels 
x_all = np.vstack([all_Teff1, all_logg1, all_feh, all_q]).T
y_all = np.vstack([all_Teff2/1000, all_logg2]).T # scale Teff2 by 1000

# training set. Copy for later, to check loss on validation vs training set. 
x_train = x_all[:n_train, :].copy()
y_train = y_all[:n_train, :].copy()

# validation set. Not used further here, but useful for cross validation alter. 
x_test = x_all[n_train:, :].copy()
y_test = y_all[n_train:].copy()

# scale thelabels 
x_max = np.max(x_all, axis = 0)
x_min = np.min(x_all, axis = 0)
x_all = (x_all - x_min)/(x_max - x_min) - 0.5

x = x_all[:n_train, :]
y = y_all[:n_train]

dim_in = x.shape[1]

# Pytorch variables. 
x = Variable(torch.from_numpy(x)).type(torch.FloatTensor)
y = Variable(torch.from_numpy(y), requires_grad = False).type(torch.FloatTensor)

# define neural network. One hidden layer with 20 neurons. 
model = torch.nn.Sequential(
        torch.nn.Linear(dim_in, 20),
        torch.nn.Sigmoid(),
        torch.nn.Linear(20, 2))
loss_fn = torch.nn.L1Loss(size_average = True)

# define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#-----------------------------------------------------------------------------
# convergence counter
current_loss = 1000.
count = 0
t = 0

# train the neural network
for i in range(num_steps_to_converge):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # print loss function to monitor. It should have plateaued by the end.  
    if i % 1000 == 0:
        print(i, loss.data[0])
    
# write the new model 
model_array = []
for param in model.parameters():
    model_array.append(param.data.numpy())
    
w_array_0 = model_array[0]
b_array_0 = model_array[1]
w_array_1 = model_array[2]
b_array_1 = model_array[3]

# save parameters and remember how we scale the labels
np.savez('NN_Teff2_logg2.npz', w_array_0 = w_array_0, w_array_1 = w_array_1,
         b_array_0 = b_array_0, b_array_1 = b_array_1, x_max = x_max, x_min = x_min)