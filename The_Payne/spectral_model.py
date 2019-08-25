# code for predicting the spectrum of a single star in normalized space.
from __future__ import absolute_import, division, print_function # python2 compatibility
import numpy as np
from . import utils


def leaky_relu(z):
    '''
    We adopt leaky_relu as the activation function in all our neural networks.
    You can experiment with using other activation function instead
    '''
    return np.maximum(0,z) + 0.01*np.minimum(0,z)

def get_spectrum_from_neural_net(scaled_labels, NN_coeffs):
    '''
    Predict the rest-frame spectrum (normalized) of a single star.
    We input the scaled stellar labels (not in the original unit). Each label ranges from -0.5 to 0.5
    '''

    # assuming your NN has two hidden layers.
    w_array_0, w_array_1, w_array_2, b_array_0, b_array_1, b_array_2, x_min, x_max = NN_coeffs

    # the neural network architecture adopted in Ting+ 18, individual networks for individual pixels
    #inside = np.einsum('ijk,k->ij', w_array_0, scaled_labels) + b_array_0
    #outside = np.einsum('ik,ik->i', w_array_1, sigmoid(inside)) + b_array_1
    #spectrum = w_array_2*sigmoid(outside) + b_array_2

    # having a single large network seems for all pixels seems to work better
    # as it exploits the information between adjacent pixels
    inside = np.einsum('ij,j->i', w_array_0, scaled_labels) + b_array_0
    outside = np.einsum('ij,j->i', w_array_1, leaky_relu(inside)) + b_array_1
    spectrum = np.einsum('ij,j->i', w_array_2, leaky_relu(outside)) + b_array_2
    return spectrum
