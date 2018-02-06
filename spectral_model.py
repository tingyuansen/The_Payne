# code for predicting the spectrum of a single star, or a binary, in normalized or unnormalized space. 
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
    
def get_spectrum_from_neural_net(labels, NN_coeffs, normalized = False):
    '''
    Predict the rest-frame spectrum (normalized or unnormalized) of a single star. 
    Note that the NN_coeffs (and potentially, labels) can be different depending on 
    whether we're getting a normalized or unnormalized spectrum. 
    '''
    # the unnormalized spectra are predicted as 1e6*f_lambda and then divided 
    # by 1e6, because the NN performs better when the quantity being predicted is 
    # of order unity. 
    if normalized:
        norm = 1
    else:
        norm = 1e6
    
    # assuming your NN was only a single hidden layer. 
    w_array_0, w_array_1, b_array_0, b_array_1, x_min, x_max = NN_coeffs
    scaled_labels = (labels - x_min)/(x_max - x_min) - 0.5
    
    # this is just efficient matrix multiplication. quite a bit faster than np.dot()
    inside = np.einsum('ijk,k->ij', w_array_0, scaled_labels) + b_array_0
    outside = np.einsum('ij,ij->i', w_array_1, sigmoid(inside)) + b_array_1
    spectrum = outside/norm
    return spectrum
    
def get_unnormalized_spectrum_single_star(labels, NN_coeffs_norm, NN_coeffs_flux, 
    NN_coeffs_R):
    '''
    (a) predict the unnormalized spectrum in surface flux units. 
    (b) scale by (R/d)^2 and by 1e-17 so the units are more convenient. 
    
    labels: [Teff, logg, feh, alpha, vmacro, dv] (dv is the velocity offset)
    NN_coeffs_i: the neural network coefficients for predicting normalized/unnormalized spectra,
        and the stellar radii
    
    The units of the returned spectrum are 1e-17 erg/s/cm^2/A, assuming the star is viewed 
        at a distance of d = 1 kpc
    '''
    R_sun_cm = 6.957e10
    pc_cm = 3.086e18 
    dist_pc = 1000
    R = get_radius_NN(input_labels = list(labels[:-3]), NN_coeffs_R = NN_coeffs_R) # in Rsun
    
    # in erg s^-1 cm^-2 AA^-1
    f_lambda = get_surface_flux_spectrum_single_star(labels = labels, 
        NN_coeffs_norm = NN_coeffs_norm, NN_coeffs_flux = NN_coeffs_flux)
    
    L_lambda = 4*np.pi*(R*R_sun_cm)**2*f_lambda # in erg s^-1 AA^-1 
    f_lambda = L_lambda/(4*np.pi*(dist_pc*pc_cm)**2) # erg/s/cm^2/AA
    f_scaled = f_lambda/1e-17 # so the values aren't too small
    
    return f_scaled

def get_surface_flux_spectrum_single_star(labels, NN_coeffs_norm, NN_coeffs_flux):
    '''
    (a) predict the normalized spectrum, presumably using a data-driven model, 
    (b) predict the unnormalized spectrum, using e.g. Kurucz models
    (c) get the continuum from the unnormalized spectrum
    (d) multiply the normalized spectrum by said continuum.
    (e) redshift as necessary
    The result is an unnormalized spectrum which has the high-frequency features 
        (e.g. lines) from the data-driven model and low-frequency features (continuum)
        from the synthetic model. 
    
    This has units of surface flux; i.e., it is not scaled by the radius of the star
    labels: [Teff, logg, feh, alpha, vmacro, dv] (dv is the velocity offset)
    '''
    c_aa_s = 2.998e18 # speed of light in angstroms/sec
    dv = labels[-1]
    
    # first, get f_nu in surface flux units. 
    # labels for the synthetic spectra are [Teff, logg, feh, alpha]
    # and for the data driven spectra, [Teff, logg, feh, alpha, vmacro]
    fnu = get_spectrum_from_neural_net(labels[:-2], NN_coeffs = NN_coeffs_flux, 
        normalized = False)
    cont = utils.get_apogee_continuum(wavelength = wavelength, spec = fnu, 
        spec_err = None, cont_pixels = cont_pixels)
    fnu_norm = get_spectrum_from_neural_net(labels[:-1], NN_coeffs = NN_coeffs_norm, 
        normalized = True)
    fnu_good = fnu_norm * cont # in erg cm^-2 s^-1 Hz^-1 Sr^-1
    
    # change from fnu to f_lambda. The factor of 4*pi gets rid of the Sr^-1. 
    f_lambda = fnu_good * (c_aa_s/wavelength**2) * 4*np.pi # in erg cm^-2 s^-1 AA^-1 
    f_shifted = utils.doppler_shift(wavelength = wavelength, flux = f_lambda, dv = dv)
    
    return f_shifted
    
def get_normalized_spectrum_binary(labels, NN_coeffs_norm, NN_coeffs_flux, 
    NN_coeffs_Teff2_logg2, NN_coeffs_R, spec_err = None):
    '''
    Determine Teff and logg of the secondary, predict the normalized spectra of the primary
    and the secondary, add them together, and normalize. 
    
    labels = [Teff1, logg1, [Fe/H], [Mg/Fe], q, v_macro1, v_macro2, dv1, dv2]
    spec_err is an array of uncertainties that is used in normalization only. If you just
        want to predict the theoretical spectrum, it isn't needed. But if you're fitting
        an observed spectrum, this should be the uncertainties of the observed spectrum,
        so that the observed and model spectra are normalized self-consistently. 
    '''
    Teff1, logg1, feh, alphafe, q, vmacro1, vmacro2, dv1, dv2 = labels
    Teff2, logg2 = get_Teff2_logg2_NN(labels = [Teff1, logg1, feh, q], 
        NN_coeffs_Teff2_logg2 = NN_coeffs_Teff2_logg2)
    labels1 = [Teff1, logg1, feh, alphafe, vmacro1, dv1]
    labels2 = [Teff2, logg2, feh, alphafe, vmacro2, dv2]
    
    f_lambda1 = get_unnormalized_spectrum_single_star(labels = labels1, 
        NN_coeffs_norm = NN_coeffs_norm, NN_coeffs_flux = NN_coeffs_flux, 
        NN_coeffs_R = NN_coeffs_R)
    f_lambda2 = get_unnormalized_spectrum_single_star(labels = labels2, 
        NN_coeffs_norm = NN_coeffs_norm, NN_coeffs_flux = NN_coeffs_flux, 
        NN_coeffs_R = NN_coeffs_R)
    f_lambda_binary = f_lambda1 + f_lambda2
    cont = utils.get_apogee_continuum(wavelength = wavelength, spec = f_lambda_binary, 
        spec_err = spec_err, cont_pixels = cont_pixels)
    f_lambda_binary_norm = f_lambda_binary/cont
    return f_lambda_binary_norm
    
def get_normalized_spectrum_single_star(labels, NN_coeffs_norm, NN_coeffs_flux, 
    spec_err = None):
    '''
    Predict a spectrum in unnormalized space. Then normalize it using the observed
    flux uncertainties, so that it can be compared to the observed spectrum 
    self-consistently. 
    
    labels = [Teff, logg, [Fe/H], [Mg/Fe], v_macro, dv]
    '''
    f_lambda = get_surface_flux_spectrum_single_star(labels = labels, 
        NN_coeffs_norm = NN_coeffs_norm, NN_coeffs_flux = NN_coeffs_flux)
    cont = utils.get_apogee_continuum(wavelength = wavelength, spec = f_lambda, 
        spec_err = spec_err, cont_pixels = cont_pixels)
    f_lambda_norm = f_lambda/cont
    return f_lambda_norm

def get_radius_NN(input_labels, NN_coeffs_R):
    '''
    Predict the radius, in units of Rsun, of a single star with particular [Teff, logg, feh]
    I trained a network with 2 hidden layers. 
    Note, the training set only consisted of MS stars; this will not give the right answer
        for giants, or for MS stars that are too hot or too far from an isochrone. 
    input_labels = [Teff, logg, feh]
    '''    
    # unpack the NN
    w_array_0, w_array_1, w_array_2, b_array_0, b_array_1, b_array_2, x_min, x_max = NN_coeffs_R
    
    # rescale labels the same way as we trained the neural network
    scaled_labels =  (input_labels - x_min)/(x_max - x_min) - 0.5
    
    predict_output = np.sum(w_array_2 * sigmoid(np.sum(w_array_1 * (sigmoid(np.dot(w_array_0,
        scaled_labels) + b_array_0)), axis = 1) + b_array_1).T, axis = 1) + b_array_2
    return predict_output
    
def get_Teff2_logg2_NN(labels, NN_coeffs_Teff2_logg2, force_lower_Teff = True):
    '''
    Use a neural network to predict Teff and logg of the secondary from Teff1, logg1,
    feh, and q. The NN was trained on isochrones of MS stars where the two stars have 
    the same age and composition. 
    
    labels = [Teff1, logg1, feh, q]
    outputs [Teff2, logg2]
    '''
    # if the mass ratio is 1, the two stars should be identical. The NN should predict something
    # close to identical anyway, but we force it to gaurantee that the normalized spectrum of a 
    # binary with no velocity offset and q=1 (and same vmacro for both stars) is identical to that
    # of a single star. 
    if np.isclose(labels[-1], 1):
        return labels[:2]
        
    # unpack the NN
    w_array_0, w_array_1, b_array_0, b_array_1, x_min, x_max = NN_coeffs_Teff2_logg2
    
    # rescale labels the same way as we trained the neural network
    scaled_labels =  (labels - x_min)/(x_max - x_min) - 0.5
    inside = np.dot(w_array_0, scaled_labels) + b_array_0
    outside = np.dot(w_array_1, sigmoid(inside)) + b_array_1
    outside[0] *= 1000 # because the NN was trained to predict Teff/1000
    
    # For equal-age, equal-composition stars on the MS, the secondary should always 
    # be cooler than the primary and have higher logg. This should happen anyway, if the
    # NN is functioning properly, but one can force it for increased stability. Note
    # that forcing this would not make sense for giants-dwarf binaries, where the more 
    # massive star could be cooler. 
    if force_lower_Teff:
        if outside[0] > labels[0]: 
            outside[0] = labels[0]
        if outside[1] < labels[1]:
            outside[1] = labels[1]
            
    return outside
    
def single_star_model_visit_spectra(labels, spec_errs, NN_coeffs_norm, NN_coeffs_flux):
    '''
    get the spectra predicted by the model for several visits. 
    Because this is the simplest single-star model, require the velocity 
        and other labels to be the same at each visit. 
    The only difference between the spectra predicted for different visits
        is that a different error array is used in normalizing each one. 
    
    labels = (Teff, logg, [Fe/h], [Mg/Fe], vmacro, dv)
    spec_errs is a list of error arrays, one for each visit. 
    
    Returns the model spectra for each visit, stitched together end-to-end 
        for convenience in fitting. 
    '''
    all_norm_specs = []
    for i, spec_err in enumerate(spec_errs):
        this_spec = get_normalized_spectrum_single_star(labels = labels, 
            NN_coeffs_norm = NN_coeffs_norm, NN_coeffs_flux = NN_coeffs_flux, 
            spec_err = spec_err)
        all_norm_specs.append(this_spec)
    stitched_norm_spec = np.concatenate(all_norm_specs)
    return stitched_norm_spec
    
def sb1_model_visit_spectra(labels, spec_errs, NN_coeffs_norm, NN_coeffs_flux):
    '''
    get the spectra predicted by the model for several visits. 
    require the structural labels of the star to be the same at each visit,
        but allow the heliocentric velocity to vary between visits.
        
    labels = (Teff, logg, [Fe/h], [Mg/Fe], vmacro, dv_i), where i = 0...N_visits
    spec_errs is a list of error arrays, one for each visit. 
    
    Returns the model spectra for each visit, stitched together end-to-end 
        for convenience in fitting. 
    '''
    Teff, logg, feh, alpha, vmacro = labels[:5]
    dv_i = labels[5:]
    
    all_norm_specs = []
    for i, spec_err in enumerate(spec_errs):
        this_label = [Teff, logg, feh, alpha, vmacro, dv_i[i]]
        this_spec = get_normalized_spectrum_single_star(labels = this_label, 
            NN_coeffs_norm = NN_coeffs_norm, NN_coeffs_flux = NN_coeffs_flux, 
            spec_err = spec_err)
        all_norm_specs.append(this_spec)
    stitched_norm_spec = np.concatenate(all_norm_specs)
    return stitched_norm_spec
    
def sb2_model_visit_spectra(labels, spec_errs, NN_coeffs_norm, NN_coeffs_flux,
    NN_coeffs_R, NN_coeffs_Teff2_logg2):
    '''
    get the spectra predicted by the model for several visits. 
    This model requires that the velocities of the two stars fall on a single
        line with negative slope, as is demanded by conservation of momentum
        for a true, isolated binary.
    
    for labels = (Teff, logg, feh, alpha, q, vmacro1, vmacro2, q_dyn, gamma, dv1_i),
        where i = 0...N_visits.
    '''
    Teff, logg, feh, alpha, q, vmacro1, vmacro2, q_dyn, gamma = labels[:9] 
    dv1_i = labels[9:] 
    
    all_norm_specs = []
    for i, spec_err in enumerate(spec_errs):
        dv_1 = dv1_i[i]
        dv_2 = (gamma*(q_dyn + 1) - dv_1)/q_dyn 
        this_label = [Teff, logg, feh, alpha, q, vmacro1, vmacro2, dv_1, dv_2]
        this_spec = get_normalized_spectrum_binary(labels = this_label, 
            NN_coeffs_norm = NN_coeffs_norm, NN_coeffs_flux = NN_coeffs_flux, 
            NN_coeffs_Teff2_logg2 = NN_coeffs_Teff2_logg2, NN_coeffs_R = NN_coeffs_R, 
            spec_err = spec_err)
        all_norm_specs.append(this_spec)
    stitched_norm_spec = np.concatenate(all_norm_specs)
    return stitched_norm_spec