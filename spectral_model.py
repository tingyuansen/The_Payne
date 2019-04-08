# code for predicting the spectrum of a single star in normalized space. 
from __future__ import absolute_import, division, print_function # python2 compatibility
import numpy as np
import utils

# read in the default wavelength array and the list of pixels used for fitting
wavelength = utils.load_wavelength_array()
cont_pixels = utils.load_cannon_contpixels()

def sigmoid(z):
    '''
    This is the activation function used by default in all our neural networks. 
    You can experiment with using an ReLU instead, but I got worse results in 
    some simple tests. 
    '''
    return 1.0/(1.0 + np.exp(-z))
    
def get_spectrum_from_neural_net(labels, NN_coeffs):
    '''
    Predict the rest-frame spectrum (normalized) of a single star. 
    We input the scaled stellar labels (not in the original unit). Each label ranges from -0.5 to 0.5
    '''
    
    # assuming your NN has two hidden layers. 
    w_array_0, w_array_1, w_array_2, b_array_0, b_array_1, b_array_2, x_min, x_max = NN_coeffs
    
    # this is just efficient matrix multiplication. quite a bit faster than np.dot()
    inside = np.einsum('ijk,k->ij', w_array_0, labels) + b_array_0
    outside = np.einsum('ik,ik->i', w_array_1, sigmoid(inside)) + b_array_1
    spectrum = w_array_2*sigmoid(outside) + b_array_2
    return spectrum
    
