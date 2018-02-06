'''
Code to train a neural network to predict stelar radius as a function of Teff, logg, and [Fe/H]
Very similar setup to the train_NN_Teff2_logg.py file.
The training set was put together for MIST isochrones of main-sequence stars only;
a new training set would be needed to properly model giants.

This implementation uses a fairly large NN, with 2 hidden layers and 100 neurons in 
each. After it was already implemented, I experimented more and found that such
a large network is not necessary -- we could do just as well with a small network 
like the one in train_NN_Teff2_logg.py. This isn't a big problem, since the network
trained here does fine in cross-validation, but a small network is faster to train 
and less susceptible to overfitting, so if I were to rewrite this, I'd use a smaller
network. This would require changing spectral_model.get_radius_NN()

As it is, this runs in a few hours on my laptop. With a smaller network, it should only
take ~15 minutes.

As always, if you retrain this with a different training set, it's important to do cross
validation and ensure it's behaving as expected.
'''
from __future__ import absolute_import, division, print_function # python2 compatibility
import numpy as np
import sys
import os
import torch
from torch.autograd import Variable

n_train = 20000
learning_rate = 0.01 # larger learning rate for bigger network. 
num_steps_for_convergence = 300000 

path = 'other_data/MIST_logR_Teff_logg_feh_single_stars.npz'
data = np.load(path)
all_logRs = data['logR']
all_Teffs = data['Teff']
all_loggs = data['logg']
all_fehs = data['feh']
mask = (all_Teffs < 8000) & (all_loggs > 4)
all_logRs, all_Teffs, all_loggs, all_fehs = all_logRs[mask], all_Teffs[mask], all_loggs[mask], all_fehs[mask]

# randomize 
np.random.seed(0)
ind = np.arange(len(all_logRs))
np.random.shuffle(ind)
all_logRs, all_Teffs, all_loggs, all_fehs = all_logRs[ind], all_Teffs[ind], all_loggs[ind], all_fehs[ind]
all_Rs = 10**all_logRs

x_all = np.vstack([all_Teffs, all_loggs, all_fehs]).T
y_all = all_Rs.T

# scale labels
x_max = np.max(x_all, axis = 0)
x_min = np.min(x_all, axis = 0)
x_all = (x_all - x_min)/(x_max - x_min) - 0.5

x = x_all[:n_train, :]
y = y_all[:n_train]

x_test = x_all[n_train:, :]
y_test = y_all[n_train:]

dim_in = x.shape[1]

x = Variable(torch.from_numpy(x)).type(torch.FloatTensor)
y = Variable(torch.from_numpy(y), requires_grad=False).type(torch.FloatTensor)

# define neural network
model = torch.nn.Sequential(
        torch.nn.Linear(dim_in, 100),
        torch.nn.Sigmoid(),
        torch.nn.Linear(100, 100),
        torch.nn.Sigmoid(),
        torch.nn.Linear(100, 1),
)
loss_fn = torch.nn.L1Loss(size_average = True)

# define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# train the neural network
for i in range(num_steps_for_convergence):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # print cost to monitor
    if i % 1000 == 0:
        print(i, loss.data[0])
    
model_array = []
for param in model.parameters():
    model_array.append(param.data.numpy())
    
w_array_0 = model_array[0]
b_array_0 = model_array[1]
w_array_1 = model_array[2]
b_array_1 = model_array[3]
w_array_2 = model_array[4]
b_array_2 = model_array[5]

# save parameters and remember how we scale the labels
np.savez('NN_R_from_Teff_and_logg_and_feh.npz',
         w_array_0 = w_array_0,
         w_array_1 = w_array_1,
         w_array_2 = w_array_2,
         b_array_0 = b_array_0,
         b_array_1 = b_array_1,
         b_array_2 = b_array_2,
         x_max = x_max,
         x_min = x_min)