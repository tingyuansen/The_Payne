# a few low-level functions that are used throughout
from __future__ import absolute_import, division, print_function # python2 compatibility
import numpy as np
import os

def read_in_neural_network():
    '''
    read in the weights and biases parameterizing a particular neural network. 
    You can read in existing networks from the neural_nets/ directory, or you
    can train your own networks and edit this function to read them in. 
    '''

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'neural_nets/NN_normalized_spectra.npz')
    tmp = np.load(path)
    w_array_0 = tmp["w_array_0"]
    w_array_1 = tmp["w_array_1"]
    w_array_2 = tmp["w_array_2"]
    b_array_0 = tmp["b_array_0"]
    b_array_1 = tmp["b_array_1"]
    b_array_2 = tmp["b_array_2"]
    x_min = tmp["x_min"]
    x_max = tmp["x_max"]
    NN_coeffs = (w_array_0, w_array_1, w_array_2, b_array_0, b_array_1, b_array_2, x_min, x_max)
    tmp.close()
    return NN_coeffs

def load_wavelength_array():
    '''
    read in the default wavelength grid onto which we interpolate all spectra
    '''
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'other_data/apogee_wavelength.npz')
    tmp = np.load(path)
    wavelength = tmp['wavelength']
    tmp.close()
    return wavelength

def load_apogee_mask():
    '''
    read in the pixel mask with which we will omit bad pixels during spectral fitting
    The mask is made by comparing the tuned Kurucz models to the observed spectra from Arcturus and the Sun from APOGEE. We mask out pixels that show more than 2% of deviations.
    '''
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'other_data/apogee_mask.npz')
    tmp = np.load(path)
    mask = tmp['apogee_mask']
    tmp.close()
    return mask

def load_cannon_contpixels():
    '''
    read in the default list of APOGEE pixels to use for continuum fitting. 
    These are taken from Melissa Ness' work with the Cannon
    '''
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'other_data/cannon_cont_pixels_apogee.npz')
    tmp = np.load(path)
    pixels_cannon = tmp['pixels_cannon']
    tmp.close()
    return pixels_cannon

def load_training_data():
    '''
    read in the default Kurucz training spectra for APOGEE

    Here we only consider 1000 training spectra and 500 validation spectra
    for the tutorial (due to the GitHub upload limit); in practice, more
    training spectra will be better. The default neural networks included were
    trained using 10000 training spectra.
    '''
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'other_data/kurucz_training_spectra.npz')
    tmp = np.load(path)
    training_labels = (temp["labels"].T)[:1000,:]
    training_spectra = temp["spectra"][:1000,:]
    validation_labels = (temp["labels"].T)[1000:,:]
    validation_spectra = temp["spectra"][1000:,:]
    tmp.close()
    return training_labels. training_spectra, validation_labels, validation_spectra

def doppler_shift(wavelength, flux, dv):
    '''
    dv is in km/s
    We use the convention where a positive dv means the object is moving away.
    
    This linear interpolation is actually not that accurate, but is fine if you 
    only care about accuracy to the level of a few tenths of a km/s. If you care
    about better accuracy, you can do better with spline interpolation. 
    '''
    c = 2.99792458e5 # km/s
    doppler_factor = np.sqrt((1 - dv/c)/(1 + dv/c)) 
    new_wavelength = wavelength * doppler_factor
    new_flux = np.interp(new_wavelength, wavelength, flux)
    return new_flux

def get_apogee_continuum(spec, spec_err = None, cont_pixels = None):
    '''
    this is designed to give the same result as the normalization function from 
    Jo Bovy's APOGEE package, but it's much faster. 
    pixels with large uncertainty are weighted less in the fit. 
    '''
    if cont_pixels is None:
        cont_pixels = load_cannon_contpixels()
    cont = np.empty_like(spec)

    wavelength = load_wavelength_array()
    
    deg = 4
    
    # if we haven't given any uncertainties, just assume they're the same everywhere. 
    if spec_err is None:
        spec_err = np.zeros(spec.shape[0]) + 0.0001
    
    # Rescale wavelengths
    bluewav = 2*np.arange(2920)/2919 - 1
    greenwav = 2*np.arange(2400)/2399 - 1
    redwav = 2*np.arange(1894)/1893 - 1
    
    blue_pixels= cont_pixels[:2920]
    green_pixels= cont_pixels[2920:5320]
    red_pixels= cont_pixels[5320:]
    
    # blue
    cont[:2920]= _fit_cannonpixels(bluewav, spec[:2920], spec_err[:2920],
                        deg, blue_pixels)
    # green 
    cont[2920:5320]= _fit_cannonpixels(greenwav, spec[2920:5320], spec_err[2920:5320],
                        deg, green_pixels)
    # red
    cont[5320:]= _fit_cannonpixels(redwav, spec[5320:], spec_err[5320:], deg, red_pixels)
    return cont

def _fit_cannonpixels(wav, spec, specerr, deg, cont_pixels):
    '''
    Fit the continuum to a set of continuum pixels
    helper function for get_apogee_continuum()
    '''
    chpoly = np.polynomial.Chebyshev.fit(wav[cont_pixels], spec[cont_pixels],
                deg, w=1./specerr[cont_pixels])
    return chpoly(wav)
