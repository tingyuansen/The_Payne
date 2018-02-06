# code for fitting spectra, using the models in spectral_model.py  
from __future__ import absolute_import, division, print_function # python2 compatibility
import numpy as np
import spectral_model
import utils
    
# read in the default wavelength array and the list of pixels used for fitting
wavelength = utils.load_wavelength_array()
cont_pixels = utils.load_cannon_contpixels()

def fit_normalized_spectrum_single_star_model(norm_spec, spec_err,
    NN_coeffs_norm, NN_coeffs_flux, p0 = None, num_p0 = 1):
    '''
    fit a single-star model to a single combined spectrum
    
    p0 is an initial guess for where to initialize the optimizer. Because 
        this is a simple model, having a good initial guess is usually not
        important. 
    
    if num_p0 is set to a number greater than 1, this will initialize a bunch
        of different walkers at different points in parameter space. If they 
        converge on different solutions, it will pick the one with the lowest
        chi2. 
    labels = [Teff, logg, [Fe/H], [Mg/Fe], vmacro, dv]
    
    returns:
        popt: the best-fit labels
        pcov: the covariance matrix, from which you can get formal fitting uncertainties
        model_spec: the model spectrum corresponding to popt 
    '''
    tol = 5e-4 # tolerance for when the optimizer should stop optimizing.
    def fit_func(dummy_variable, *labels):
        norm_spec = spectral_model.get_normalized_spectrum_single_star(labels = labels, 
            NN_coeffs_norm = NN_coeffs_norm, NN_coeffs_flux = NN_coeffs_flux, 
            spec_err = spec_err)
        return norm_spec
    
    # if no initial guess is supplied, start with labels for the Sun. 
    if p0 is None: 
        p0 = [5777, 4.44, 0, 0, 5, 0]
        
    # don't allow the minimimizer outside Teff = [4200, 7000], etc. 
    bounds = [[4200, 4.0, -1, -0.3, 0, -100], [7000, 5.0, 0.5, 0.5, 45, 100]]
    
    # if we want to initialize many walkers in different parts of parameter space, do so now. 
    all_x0 = generate_starting_guesses_to_initialze_optimizers(p0 = p0, bounds = bounds, 
        num_p0 = num_p0, vrange = 10, model = 'single_star')
        
    # run the optimizer
    popt, pcov, model_spec = fit_all_p0s(fit_func = fit_func, norm_spec = norm_spec, 
        spec_err = spec_err, all_x0 = all_x0, bounds = bounds, tol = tol)
    return popt, pcov, model_spec

def fit_normalized_spectrum_binary_model(norm_spec, spec_err,
    NN_coeffs_norm, NN_coeffs_flux, NN_coeffs_Teff2_logg2, NN_coeffs_R,
    p0_single, num_p0 = 10):
    '''
    fit a binary model to a single combined spectrum
    
    p0_single is an initial guess for the labels *of a single star* used to 
        initialize the optimizer. A reasonable approach is to first fit a 
        single-star model, and then use the best-fit parameters from that 
        as a starting guess for the binary model.
    
    if num_p0 is set to a number greater than 1, this will initialize a bunch
        of different walkers at different points in parameter space. If they 
        converge on different solutions, it will pick the one with the lowest
        chi2. Most of the time, the code *does* find the best-fit model with a
        single walker, but the posterior generally *is* bimodal in q, so setting
        num_p0 > 1 can help. 
    
    labels = [Teff, logg, [Fe/H], [Mg/Fe], q, vmacro1, vmacro2, dv1, dv2]
    
    returns:
        popt: the best-fit labels
        pcov: the covariance matrix, from which you can get formal fitting uncertainties
        model_spec: the model spectrum corresponding to popt 
    '''
    tol = 5e-4 # tolerance for when the optimizer should stop optimizing.
    
    def fit_func(dummy_variable, *labels):
        norm_spec = spectral_model.get_normalized_spectrum_binary(labels = labels, 
            NN_coeffs_norm = NN_coeffs_norm, NN_coeffs_flux = NN_coeffs_flux, 
            NN_coeffs_Teff2_logg2 = NN_coeffs_Teff2_logg2, NN_coeffs_R = NN_coeffs_R, 
            spec_err = spec_err)
        return norm_spec
        
    teff1, logg1, feh, alphafe, vmacro1, dv1 = p0_single
    
    # a binary in which both stars are identical (should have a spectrum identical to 
    # that of the primary only)
    p0 = [teff1, logg1, feh, alphafe, 1, vmacro1, vmacro1, dv1, dv1]
    
    min_q = get_minimum_q_for_this_teff(Teff1 = teff1, logg1 = logg1, feh = feh, 
        NN_coeffs_Teff2_logg2 = NN_coeffs_Teff2_logg2)
        
    lower = [4200, 4.0, -1, -0.3, min_q, 0, 0, -100, -100]
    upper = [7000, 5.0, 0.5, 0.5, 1, 45, 45, 100, 100]
    bounds = [lower, upper]
    
    # if we want to initialize many walkers in different parts of parameter space, do so now. 
    all_x0 = generate_starting_guesses_to_initialze_optimizers(p0 = p0, bounds = bounds, 
        num_p0 = num_p0, vrange = 15, model = 'binary')
    
    # run the optimizer
    popt, pcov, model_spec = fit_all_p0s(fit_func = fit_func, norm_spec = norm_spec, 
        spec_err = spec_err, all_x0 = all_x0, bounds = bounds, tol = tol)

    # make sure a better fit can't be obtained with a single-star model
    single_model = fit_func([], *p0)
    chi2_single = np.sum((single_model - norm_spec)**2/(spec_err)**2) 
    chi2_binary = np.sum((model_spec - norm_spec)**2/(spec_err)**2)
    
    if chi2_single < chi2_binary:
        popt, model_spec = p0, single_model
        
    return popt, pcov, model_spec    

def generate_starting_guesses_to_initialze_optimizers(p0, bounds, num_p0, vrange = 10,
    model = 'single_star'):
    '''
    if we want to initialize many walkers in different parts of parameter space. 
    p0 is the initial guess around which to cluster our other guesses
    bounds is the region of which parameter space the optimizer is allowed to explore.
    num_p0 is how many walkers we want. 
    vrange is half the range in velocity that the starting guesses should be spread over.
        If you're fitting a close binary with very large velocity offset, it can 
        occur that the walkers are initialized too far from the best-fit velocity
        to find it. For such cases, it can be useful to increase vrange to ~50. 
    '''
    all_x0 = [p0]
    lower, upper = bounds
    
    if num_p0 > 1:
        if model == 'single_star':
            for i in range(num_p0-1):
                teff = np.random.uniform(max(lower[0], p0[0] - 300), min(upper[0], p0[0] + 500))
                logg = np.random.uniform(max(lower[1], p0[1] - 0.2), min(upper[1], p0[1] + 0.2))
                feh = np.random.uniform(max(lower[2], p0[2] - 0.2), min(upper[2], p0[2] + 0.2))
                alpha = np.random.uniform(max(lower[3], p0[3] - 0.05), min(upper[3], p0[3] + 0.05))
                vmac = np.random.uniform(max(lower[4], p0[4] - 10), min(upper[4], p0[4] + 10))
                dv = np.random.uniform(max(lower[5], p0[5] - vrange), min(upper[5], p0[5] + vrange))
                this_p0 = np.array([teff, logg, feh, alpha, vmac, dv])
                all_x0.append(this_p0)
                
        elif model == 'binary': # for combined spectrum
            dq = (1 - lower[4])/(num_p0 - 1)
            all_q0 = np.arange(lower[4] + 1e-5, upper[4], dq)
            for i, q in enumerate(all_q0):
                teff = np.random.uniform(max(lower[0], p0[0] - 300), min(upper[0], p0[0] + 500))
                logg = np.random.uniform(max(lower[1], p0[1] - 0.2), min(upper[1], p0[1] + 0.2))
                feh = np.random.uniform(max(lower[2], p0[2] - 0.2), min(upper[2], p0[2] + 0.2))
                alpha = np.random.uniform(max(lower[3], p0[3] - 0.05), min(upper[3], p0[3] + 0.05))
                vmac1 = np.random.uniform(max(lower[4], p0[4] - 10), min(upper[4], p0[4] + 10))
                vmac2 = np.random.uniform(max(lower[4], p0[4] - 10), min(upper[4], p0[4] + 10))
                dv1 = np.random.uniform(max(lower[5], p0[5] - vrange), min(upper[5], p0[5] + vrange))
                dv2 = np.random.uniform(max(lower[5], p0[5] - vrange), min(upper[5], p0[5] + vrange))
                this_p0 = np.array([teff, logg, feh, alpha, q, vmac1, vmac1, dv1, dv2])
                all_x0.append(this_p0)  
        
        elif model == 'sb1':
            for i in range(num_p0-1):
                p0_dv_i = p0[5:]
                teff = np.random.uniform(max(lower[0], p0[0] - 300), min(upper[0], p0[0] + 500))
                logg = np.random.uniform(max(lower[1], p0[1] - 0.2), min(upper[1], p0[1] + 0.2))
                feh = np.random.uniform(max(lower[2], p0[2] - 0.2), min(upper[2], p0[2] + 0.2))
                alpha = np.random.uniform(max(lower[3], p0[3] - 0.05), min(upper[3], p0[3] + 0.05))
                vmac = np.random.uniform(max(lower[4], p0[4] - 10), min(upper[4], p0[4] + 10))
                dv_change = np.random.uniform(-vrange/2, vrange/2, size = len(p0_dv_i))
                dv_i = p0_dv_i + dv_change
                this_p0 = np.concatenate([[teff, logg, feh, alpha, vmac], dv_i])                
                all_x0.append(this_p0)
        
        elif model == 'sb2':
            for i in range(num_p0-1):
                p0_dv_i = p0[9:]
                teff = np.random.uniform(max(lower[0], p0[0] - 300), min(upper[0], p0[0] + 500))
                logg = np.random.uniform(max(lower[1], p0[1] - 0.2), min(upper[1], p0[1] + 0.2))
                feh = np.random.uniform(max(lower[2], p0[2] - 0.2), min(upper[2], p0[2] + 0.2))
                alpha = np.random.uniform(max(lower[3], p0[3] - 0.05), min(upper[3], p0[3] + 0.05))
                q = np.random.uniform(max(lower[4], p0[4] - 0.1), min(upper[4], p0[4] + 0.1))
                vmac1 = np.random.uniform(max(lower[5], p0[5] - 5), min(upper[5], p0[5] + 5))
                vmac2 = np.random.uniform(max(lower[6], p0[6] - 5), min(upper[6], p0[6] + 5))
                q_dyn = np.random.uniform(max(lower[7], p0[7] - 0.1), min(upper[7], p0[7] + 0.1))
                gamma = np.random.uniform(max(lower[8], p0[8] - vrange/2), min(upper[8], p0[8] + vrange/2))
                dv_change = np.random.uniform(-vrange/2, vrange/2, size = len(p0_dv_i))
                dv_i = p0_dv_i + dv_change
                this_p0 = np.concatenate([[teff, logg, feh, alpha, q, vmac1, vmac2, q_dyn, gamma], dv_i])
                all_x0.append(this_p0)
                
    # make sure none of these walkers got initialized outside the allowed regions of label space. 
    # should not happen unless p0 was bad
    for j, p0 in enumerate(all_x0):
        for i, p in enumerate(p0):
            if (p < bounds[0][i]):
                all_x0[j][i] = bounds[0][i] + 1e-5
            if (p > bounds[1][i]):
                all_x0[j][i] = bounds[1][i] - 1e-5
    return all_x0

def get_minimum_q_for_this_teff(Teff1, logg1, feh, NN_coeffs_Teff2_logg2, min_teff = 4200):
    '''
    Because the spectra model is only reliable down to some minimum temperature 
        (4200 K in the paper), there's a minimum mass ratio q that can be modeled
        for any given primary. This estimates what that minimum q is. 
    '''
    qs = np.linspace(0.2, 1)
    all_teff2 = [spectral_model.get_Teff2_logg2_NN(labels = [Teff1, logg1, feh, q], 
        NN_coeffs_Teff2_logg2 = NN_coeffs_Teff2_logg2)[0] for q in qs] 
    min_q = np.interp(min_teff - 50, all_teff2, qs)
    return min_q

def fit_visit_spectra_single_star_model(norm_spectra, spec_errs, 
    NN_coeffs_norm, NN_coeffs_flux, v_helios, p0, num_p0 = 1):
    '''
    fit a set of visit spectra for a single object simultaneously with a single-star 
    model, with the restriction that the velocity be the same at each epoch. 
    
    v_helios is a guess of the heliocentric velocity at each visit, taken from 
        the allStar catalog.
    p0 is a starting guess for the optimizer, typically obtained from fitting
        the combined spectrum
    '''
    tol = 5e-4 # tolerance for when the optimizer should stop optimizing.
    stitched_spec, stitched_errs = np.concatenate(norm_spectra), np.concatenate(spec_errs)
    
    def fit_func(dummy_variable, *labels):
        stitched_model = spectral_model.single_star_model_visit_spectra(labels = labels,
            spec_errs = spec_errs, NN_coeffs_norm = NN_coeffs_norm, 
            NN_coeffs_flux = NN_coeffs_flux)
        return stitched_model
        
    lower = np.array([4200, 4.0, -1, -0.3, 0, -100+np.median(v_helios)])
    upper = np.array([7000, 5.0, 0.5, 0.5, 45, 100+np.median(v_helios)])
    bounds = [lower, upper]
    
    # coming from the combined fit, the velocity label in p0 will be relative the
    # rest frame assumed in making the combined spectrum. 
    p00 = np.copy(p0)
    p00[5] += np.median(v_helios)
    
    # if we want to initialize many walkers in different parts of parameter space, do so now. 
    all_x0 = generate_starting_guesses_to_initialze_optimizers(p0 = p00, bounds = bounds, 
        num_p0 = num_p0, vrange = (np.max(v_helios) - np.min(v_helios))/2, model = 'single_star')
        
    # run the optimizer
    popt, pcov, model_spec = fit_all_p0s(fit_func = fit_func, norm_spec = stitched_spec, 
        spec_err = stitched_errs, all_x0 = all_x0, bounds = bounds, tol = tol)
    model_specs = utils.unstitch_model_spectra(model_spec = model_spec, wavelength = wavelength)
    return popt, pcov, model_specs

def fit_visit_spectra_sb1_model(norm_spectra, spec_errs, NN_coeffs_norm, NN_coeffs_flux, 
    v_helios, p0, num_p0 = 5):
    '''
    fit a set of visit spectra for a single object simultaneously with an sb1
    model. I.e., the same single-star spectral model makes the spectrum at 
    each visit, but allow the heliocentric velocity to vary between visits.
    
    v_helios is a guess of the heliocentric velocity at each visit, taken from 
        the allStar catalog.
    p0 is a starting guess for the optimizer, typically obtained from fitting
        the combined spectrum
    '''
    tol = 5e-4 # tolerance for when the optimizer should stop optimizing.
    stitched_spec, stitched_errs = np.concatenate(norm_spectra), np.concatenate(spec_errs)
    
    def fit_func(dummy_variable, *labels):
        stitched_model = spectral_model.sb1_model_visit_spectra(labels = labels,
            spec_errs = spec_errs, NN_coeffs_norm = NN_coeffs_norm, 
            NN_coeffs_flux = NN_coeffs_flux)
        return stitched_model
        
    v_min, v_max = np.min(v_helios), np.max(v_helios)
    lower = np.concatenate([[4200, 4.0, -1, -0.3, 0], len(v_helios) * [-100 + v_min]])
    upper = np.concatenate([[7000, 5.0, 0.5, 0.5, 45], len(v_helios) * [100 + v_max]])
    bounds = [lower, upper]
    
    p00 = np.concatenate([p0[:5], v_helios])
    all_x0 = generate_starting_guesses_to_initialze_optimizers(p0 = p00, bounds = bounds, 
        num_p0 = num_p0, vrange = (v_max - v_min)/2, model = 'sb1')
        
    popt, pcov, model_spec = fit_all_p0s(fit_func = fit_func, norm_spec = stitched_spec, 
        spec_err = stitched_errs, all_x0 = all_x0, bounds = bounds, tol = tol)
    model_specs = utils.unstitch_model_spectra(model_spec = model_spec, wavelength = wavelength)
    return popt, pcov, model_specs
    
def fit_visit_spectra_sb2_model(norm_spectra, spec_errs, NN_coeffs_norm, NN_coeffs_flux, 
    NN_coeffs_R, NN_coeffs_Teff2_logg2, v_helios, p0_combined, num_p0 = 5):
    '''
    fit a set of visit spectra for a single target simultaneously with an sb2
        model. This requires the velocities of the two stars to follow conservation
        of momentum; i.e., to fall on a line with negative slope when plotted 
        against each other. 
    
    v_helios is a guess of the heliocentric velocity at each visit, taken from 
        the allStar catalog.
    p0_combined is a starting guess for the optimizer. It's expected to be in the format
        returned from fitting a binary model to a *combined* spectrum, i.e.,
        p0_combined = [Teff1, logg1, [Fe/H], [Mg/Fe], q, vmacro1, vmacro2, dv1, dv2]
    '''
    tol = 5e-4 # tolerance for when the optimizer should stop optimizing.
    stitched_spec, stitched_errs = np.concatenate(norm_spectra), np.concatenate(spec_errs)
    def fit_func(dummy_variable, *labels):
        stitched_model = spectral_model.sb2_model_visit_spectra(labels = labels, 
            spec_errs = spec_errs, NN_coeffs_norm = NN_coeffs_norm, 
            NN_coeffs_flux = NN_coeffs_flux, NN_coeffs_R = NN_coeffs_R, 
            NN_coeffs_Teff2_logg2 = NN_coeffs_Teff2_logg2)
        return stitched_model
    v_med, v_min, v_max = np.median(v_helios), np.min(v_helios), np.max(v_helios)
    sb2_p0 = np.concatenate([p0_combined[:7], [p0_combined[4]], [v_med], v_helios])
    
    teff1, logg1, feh, alphafe = p0_combined[:4]
    min_q = get_minimum_q_for_this_teff(Teff1 = teff1, logg1 = logg1, feh = feh, 
        NN_coeffs_Teff2_logg2 = NN_coeffs_Teff2_logg2)
    
    lower = np.concatenate([[4200, 4.0, -1, -0.3, min_q, 0, 0, 0.2, -100 + v_med], 
        len(v_helios) * [-100 + v_min] ])
    upper = np.concatenate([[7000, 5.0, 0.5, 0.5, 1, 45, 45, 1.5, 100 + v_med],
        len(v_helios) * [100 + v_max]])
    bounds = [lower, upper]
    
    all_x0 = generate_starting_guesses_to_initialze_optimizers(p0 = sb2_p0, bounds = bounds, 
        num_p0 = num_p0, vrange = (v_max - v_min)/2, model = 'sb2')
    
    popt, pcov, model_spec = fit_all_p0s(fit_func = fit_func, norm_spec = stitched_spec, 
        spec_err = stitched_errs, all_x0 = all_x0, bounds = bounds, tol = tol)
    model_specs = utils.unstitch_model_spectra(model_spec = model_spec, wavelength = wavelength)
    return popt, pcov, model_specs
    
def fit_all_p0s(fit_func, norm_spec, spec_err, all_x0, bounds, tol = 5e-4):
    '''
    loop through all the points to initialize the optimizer, if there are more than one
    run the optimizer at each point sequentially
    choose the best model as the one that minimizes chi2
    fit_func is the function to predict the spectrum for a given model 
    '''
    from scipy.optimize import curve_fit
    all_popt, all_chi2, all_model_specs, all_pcov = [], [], [], []
    for i, x0 in enumerate(all_x0):
        try:
            popt, pcov = curve_fit(fit_func, xdata=[], ydata = norm_spec, sigma = spec_err, p0 = x0,
                bounds = bounds, ftol = tol, xtol = tol, absolute_sigma = True, method = 'trf')
            model_spec = fit_func([], *popt)
            chi2 = np.sum((model_spec - norm_spec)**2/spec_err**2)
        except RuntimeError: # failed to converge (should not happen for a simple model)
            popt, pcov = x0, np.zeros((len(x0), len(x0)))
            model_spec = np.copy(norm_spec)
            chi2 = np.inf
        all_popt.append(popt)
        all_chi2.append(chi2)
        all_model_specs.append(model_spec)
        all_pcov.append(pcov)
    all_popt, all_chi2, all_model_specs, all_pcov = np.array(all_popt), np.array(all_chi2), \
        np.array(all_model_specs), np.array(all_pcov)
        
    best = np.argmin(all_chi2)
    popt, pcov, model_spec = all_popt[best], all_pcov[best], all_model_specs[best]
    return popt, pcov, model_spec