# a few low-level functions that are used throughout
from __future__ import absolute_import, division, print_function # python2 compatibility
import numpy as np

def read_in_neural_network(name = 'normalized_spectra'):
    '''
    read in the weights and biases parameterizing a particular neural network. 
    You can read in existing networks from the neural_nets/ directory, or you
    can train your own networks and edit this function to read them in. 
    '''
    if name == 'normalized_spectra':
        path = 'neural_nets/NN_normalized_spectra.npz'
    elif name == 'unnormalized_spectra':
        path = 'neural_nets/NN_unnormalized_spectra.npz'
    elif name == 'radius':
        path = 'neural_nets/NN_radius.npz'
    elif name == 'Teff2_logg2':
        path = 'neural_nets/NN_Teff2_logg2.npz'
    tmp = np.load(path)
    
    # some of the networks we train have one hidden layer; others have two. 
    # assume the one we're looking for has two; if it doesn't, we won't find 
    # w_array_2 and b_array_2. 
    try:
        w_array_0 = tmp["w_array_0"]
        w_array_1 = tmp["w_array_1"]
        w_array_2 = tmp["w_array_2"]
        b_array_0 = tmp["b_array_0"]
        b_array_1 = tmp["b_array_1"]
        b_array_2 = tmp["b_array_2"]
        x_min = tmp["x_min"]
        x_max = tmp["x_max"]
        NN_coeffs = (w_array_0, w_array_1, w_array_2, b_array_0, b_array_1, b_array_2, x_min, x_max)
    except KeyError:
        w_array_0 = tmp["w_array_0"]
        w_array_1 = tmp["w_array_1"]
        b_array_0 = tmp["b_array_0"]
        b_array_1 = tmp["b_array_1"]
        x_min = tmp["x_min"]
        x_max = tmp["x_max"]
        NN_coeffs = (w_array_0, w_array_1, b_array_0, b_array_1, x_min, x_max)
    tmp.close()
    return NN_coeffs

def load_wavelength_array():
    '''
    read in the default wavelength grid onto which we interpolate all spectra
    '''
    path = 'other_data/apogee_wavelength.npz'
    tmp = np.load(path)
    wavelength = tmp['wavelength']
    tmp.close()
    return wavelength
    
def load_cannon_contpixels():
    '''
    read in the default list of APOGEE pixels to use for continuum fitting. 
    These are taken from Melissa Ness' work with the Cannon
    '''
    path = 'other_data/cannon_cont_pixels_apogee.npz'
    tmp = np.load(path)
    pixels_cannon = tmp['pixels_cannon']
    tmp.close()
    return pixels_cannon
    
def load_visit_wavelength():
    '''
    Read in the normal wavelength grid (12288 pixels) for visit spectra. Note that
    this is different from the normal wavelength grid for combined spectra, which 
    can be read in with spectral_model.load_wavelength_array()
    '''
    tmp = np.load('other_data/apogee_visit_wavelength.npz')
    wave = tmp['wave']
    tmp.close()
    return wave

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

def get_apogee_continuum(wavelength, spec, spec_err = None, cont_pixels = None):
    '''
    this is designed to give the same result as the normalization function from 
    Jo Bovy's APOGEE package, but it's much faster. 
    pixels with large uncertainty are weighted less in the fit. 
    '''
    if cont_pixels is None:
        cont_pixels = load_cannon_contpixels()
    cont = np.empty_like(spec)
    
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

def unstitch_model_spectra(model_spec, wavelength):
    '''
    for undoing concatenation of spectra from different visits
    '''
    all_model_specs = []
    npix = int(len(wavelength))
    assert(len(model_spec) % npix == 0)
    nspec = int(len(model_spec)/npix)
    for j in range(nspec):
        all_model_specs.append(model_spec[j*npix:(j+1)*npix])
    return all_model_specs

def get_chi2_difference(norm_spec, spec_err, norm_model_A, norm_model_B):
    '''
    for model comparison. Returns chi2_modelA - chi2_modelB.
    norm_model_A & B are normalized spectra predicted by two different models. 
    So e.g., if model A is more simple than model B (say, a single-star 
        vs a binary model), one would expect this to be positive. 
    '''
    chi2_A = np.sum((norm_spec - norm_model_A)**2/spec_err**2)
    chi2_B = np.sum((norm_spec - norm_model_B)**2/spec_err**2)
    return chi2_A - chi2_B