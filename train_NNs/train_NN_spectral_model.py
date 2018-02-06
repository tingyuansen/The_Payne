#!/usr/bin/env python

#SBATCH --job-name=train_NN_spectral_model
#SBATCH --partition=productionQ
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1    ## MPI tasks per node
#SBATCH --cpus-per-task=24    ## OpenMP threads per MPI task
#SBATCH --time=12:00:00
#SBATCH --output=train_NN_%j.txt

'''
This file is used to train the neural networks that predict the spectrum at
each wavelength pixel. Because we need to train a separate network for each 
pixel (7000+ networks in total), it's most efficient to run this on a cluster
and train many networks simultaneously. It's pretty fast, though. For the 
default network here, everything can be run a single 24-core node in something
like 8 hours. 

The SBATCH commands above work on the specific cluster I use, but they'll likely 
have to be adjusted a bit for other clusters/scheduling protocols. 

Because this trains pixels in batches of size pixel_batch_size, it will produce
a bunch of files NN_apogee_data_driven_norm_i.npz, where i = num_start ... num_end. 
Afterward, you'll combine these into a single set of NN weights and biases that
describe the whole spectrum. 

Note that we separately train one (data-driven) NN for normalized spectra, and 
    a second one for unnormalized spectra. This needs to be run for each of them.

The spectra in the training set are either normal APOGEE combined spectra, normalized
using utils.get_apogee_continuum(), or synthetic spectra that have been convolved
to the appropriate R (~22500 for APOGEE). In the former case, we need to mask bad 
pixels (and pixels with low S/N) in the training set spectra. This is done by setting
their values to np.nan; such pixels are ignored. 
'''
from __future__ import absolute_import, division, print_function # python2 compatibility
import numpy as np
import sys
import os
import torch
from torch.autograd import Variable
import multiprocessing
from multiprocessing import Pool

num_CPU = multiprocessing.cpu_count()
pixel_batch_size = 200

# choose a testing batch
num_start, num_end = 0, 36

# path on cluster to training set. 
training_set_path = '/home/some_directory/training_set.npz'

# if you're training on synthetic spectra for which the typical normalization 
# very different from one, set is_flux = True and choose mult_factor such that
# mult_factor*f_lambda ~ 1 for an average spectrum. Note that if you change this,
# you'll have to change spectral_model.get_spectrum_from_neural_net()
is_flux = False
if is_flux:
    mult_factor = 1e6
else:
    mult_factor = 1

# size of training set. Anything over a few 100 should be OK for a small network
# (see YST's paper), but it can't hurt to go larger if the training set is available. 
n_train = 2000

# how long to train each pixel until convergence. 5e4 is good for the specific NN
# architecture and learning I used by default, but bigger networks take more steps,
# and decreasing the learning rate will also change this. You can get a sense of 
# how many steps are needed for a new NN architecture by choosing a pixel at random
# and plotting the loss function evaluated on both the training set and a validation
# set as a function of step number. It should plateau once the NN is converged. 
num_steps_to_converge = 5e4

# this is also tunable, but 0.001 seems to work well for most use cases. Again, diagnose
# with a validation set if you change this. 
learning_rate = 0.001

# set number of threads per CPU
os.environ['OMP_NUM_THREADS']='{:d}'.format(1)

for num_go in range(num_start, num_end + 1):    
    print('starting batch %d' % num_go)
    #==============================================================================
    # restore training spectra
    temp = np.load(training_set_path)
    x = (temp["labels"].T)[:n_train,:]
    y = temp["spectra"][:n_train, num_go*pixel_batch_size:(num_go+1)*pixel_batch_size]
    temp.close()
    
    x_max = np.max(x, axis = 0)
    x_min = np.min(x, axis = 0)
    x = (x - x_min)/(x_max - x_min) - 0.5

    # if you've reached the last pixel in the spectrum
    if not len(y):
        continue
        
    y *= mult_factor
    num_pix = len(y[0])
    
    # loop over all pixels
    def train_pixel(pixel_no):
        '''
        to be fed to multiprocessing
        '''
        import sys # just so you can print from multiprocessing
        this_y = y[:, pixel_no]
        
        # mask bad pixels (which you have privously set to nan
        msk = np.isfinite(this_y) 
        this_x, this_y = x[msk], this_y[msk]

        # dimension of the input
        dim_in = this_x.shape[1]
        
        # define neural network
        model = torch.nn.Sequential(
            torch.nn.Linear(in_features = dim_in, out_features = 5),
            torch.nn.Sigmoid(),
            torch.nn.Linear(in_features = 5, out_features = 1))
            
        loss_fn = torch.nn.L1Loss(size_average = True)
        
        # if all spectra in the training set have this pixel masked, this 
        # will stop it from crashing. Obviously, the trained NN for this 
        # particular pixel would be rubbish.
        if not len(this_y):
            this_y = np.array([1, 1])
            this_x = np.array([[0.1, 0.1, 0.1, 0.1], [0, 0, 0, 0]])
            
        # make pytorch variables
        xx = Variable(torch.from_numpy(this_x)).type(torch.FloatTensor)
        yy = Variable(torch.from_numpy(this_y), requires_grad = False).type(torch.FloatTensor)

        # weight_decay is for regularization. Not required, but one can play with it. 
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                weight_decay = 0)

        # train the neural network
        t = 0
        while t < num_steps_to_converge:
            y_pred = model(xx)
            loss = loss_fn(y_pred, yy)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t += 1

        # return best-fit weights and biases
        model_array = []
        for param in model.parameters():
            model_array.append(param.data.numpy())
        print('pixel %d done!' % pixel_no)
        sys.stdout.flush() 
        return model_array    

    # train in parallel
    pool = Pool(num_CPU)
    net_array = pool.map(train_pixel, range(num_pix))

    # extract the weights and biases
    w_array_0 = np.array([net_array[i][0] for i in range(len(net_array))])
    b_array_0 = np.array([net_array[i][1] for i in range(len(net_array))])
    w_array_1 = np.array([net_array[i][2][0] for i in range(len(net_array))])
    b_array_1 = np.array([net_array[i][3][0] for i in range(len(net_array))])

    # save parameters and remember how we scaled the labels
    np.savez("NN_apogee_data_driven_norm_i" + str(num_go) + ".npz",
             w_array_0 = w_array_0,
             w_array_1 = w_array_1,
             b_array_0 = b_array_0,
             b_array_1 = b_array_1,
             x_max = x_max,
             x_min = x_min)
    pool.close()
