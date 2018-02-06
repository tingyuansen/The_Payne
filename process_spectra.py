'''
Code for reading in combined spectra and visit spectra.
This uses Jo Bovy's APOGEE package to read in spectra: https://github.com/jobovy/apogee
This code adapted only slightly from that provided by Yuan-Sen Ting. 
Any way that you can get your hands on the spectra should be fine, as long as you 
(a) set the uncertainties high in bad pixels, (b) normalize them using the 
spectral_model.get_apogee_continuum() function, and (c) set a max S/N of 200

I've only used it for spectra from DR12/13, so it's possible that some  
changes could be needed for DR14. 
'''
from __future__ import absolute_import, division, print_function # python2 compatibility
import numpy as np
import sys
import os
import utils
import spectral_model

import apogee.tools.read as apread
from apogee.tools import bitmask
from apogee.spec import continuum

# read in the default wavelength array and the list of pixels used for fitting
wavelength = utils.load_wavelength_array()
cont_pixels = utils.load_cannon_contpixels()

def read_apogee_catalog():
    '''
    read in the catalog of info for all stars in a data release. 
    '''
    all_star_catalog = apread.allStar(rmcommissioning = False, rmdups = False, 
        main = False, raw = True)
    
    # and match to allVisit for the fibers that each star was observed in
    allvdata = apread.allVisit(raw = True)
    fibers = np.zeros((len(all_star_catalog), 
        np.nanmax(all_star_catalog['NVISITS'])), dtype='int') - 1
    
    for ii in range(len(all_star_catalog)):
        for jj in range(all_star_catalog['NVISITS'][ii]):
            fibers[ii, jj] = allvdata[all_star_catalog['VISIT_PK'][ii][jj]]['FIBERID']

    return all_star_catalog, fibers

def read_batch_of_spectra(batch_count, batch_size = 10000):
    '''
    Download a bunch of *combined* spectra in one go. Set the uncertainties to a large
    value in bad pixels, normalize, and save the batch locally. 
    '''
    # read in the catalog catalog   
    catalog, fibers = read_apogee_catalog()
    catalog = catalog[batch_count*batch_size:(batch_count+1)*batch_size]
    fibers = fibers[batch_count*batch_size:(batch_count+1)*batch_size]
    _COMBINED_INDEX = 1
    
    nspec = len(catalog)
    spec = np.zeros((nspec, 7214))
    specerr = np.zeros((nspec, 7214))
    
    # Set up bad pixel mask
    badcombpixmask = bitmask.badpixmask() + 2**bitmask.apogee_pixmask_int("SIG_SKYLINE")
    
    # loop through the individual targets
    for ii in range(nspec):
        field = catalog['FIELD'][ii].decode()
        ap_id = catalog['APOGEE_ID'][ii].decode()
        loc_id = catalog['LOCATION_ID'][ii]
        print('processing target %d with id %s' % (ii, ap_id))
        
        try:
            if loc_id == 1:
                temp1 = apread.apStar(field, ap_id, ext = 1, header = False, aspcapWavegrid = True)
                temp2 = apread.apStar(field, ap_id, ext = 2, header = False, aspcapWavegrid = True)
                temp3 = apread.apStar(field, ap_id, ext = 3, header = False, aspcapWavegrid = True)
            else:
                temp1 = apread.apStar(loc_id, ap_id, ext = 1, header = False, aspcapWavegrid = True)
                temp2 = apread.apStar(loc_id, ap_id, ext = 2, header = False, aspcapWavegrid = True)
                temp3 = apread.apStar(loc_id, ap_id, ext = 3, header = False, aspcapWavegrid = True)

            if temp1.shape[0] > 6000:
                spec[ii] = temp1
                specerr[ii] = temp2
                mask = temp3
            else:
                spec[ii] = temp1[_COMBINED_INDEX]
                specerr[ii]= temp2[_COMBINED_INDEX]
                mask = temp3[_COMBINED_INDEX]
    
            # Inflate uncertainties for bad pixels        
            specerr[ii, (mask & (badcombpixmask)) != 0] += \
                100. * np.mean(spec[ii, np.isfinite(spec[ii])])
    
            # Inflate pixels with high SNR to 0.5
            highsnr = spec[ii]/specerr[ii] > 200.
            specerr[ii, highsnr] = 0.005*np.fabs(spec[ii, highsnr])

            # Continuum-normalize
            cont = utils.get_apogee_continuum(wavelength = wavelength, spec = spec[ii], 
                spec_err = specerr[ii], cont_pixels = cont_pixels)
            spec[ii] /= cont
            specerr[ii] /= cont
            specerr[ii, highsnr] = 0.005 
        except OSError:
            print('target could not be found!')
            continue

    # save spectra
    np.savez('spectra/apogee_all_spectra_' + str(batch_count) + '.npz',
             wavelength = wavelength,
             spectra = spec,
             spec_err = specerr,
             apogee_id = np.array(catalog["APOGEE_ID"]),
             apogee_fiber_id = fibers)

def get_combined_spectrum_single_object(apogee_id, catalog = None, save_local = False):
    '''
    apogee_id should be a byte-like object; i.e b'2M13012770+5754582'
    This downloads a single combined spectrum and the associated error array,
        and it normalizes both. 
    '''
    # read in the allStar catalog if you haven't already
    if catalog is None:
        catalog, fibers = read_apogee_catalog()
        
    # Set up bad pixel mask
    badcombpixmask = bitmask.badpixmask() + 2**bitmask.apogee_pixmask_int("SIG_SKYLINE")
    _COMBINED_INDEX = 1
    
    msk = np.where(catalog['APOGEE_ID'] == apogee_id)[0]
    if not len(msk):
        raise ValueError('the desired Apogee ID was not found in the allStar catalog.')
    
    field = catalog['FIELD'][msk[0]].decode()
    ap_id = apogee_id.decode()
    loc_id = catalog['LOCATION_ID'][msk[0]]
        
    if loc_id == 1:
        temp1 = apread.apStar(field, ap_id, ext = 1, header = False, aspcapWavegrid = True)
        temp2 = apread.apStar(field, ap_id, ext = 2, header = False, aspcapWavegrid = True)
        temp3 = apread.apStar(field, ap_id, ext = 3, header = False, aspcapWavegrid = True)
    else:
        temp1 = apread.apStar(loc_id, ap_id, ext = 1, header = False, aspcapWavegrid = True)
        temp2 = apread.apStar(loc_id, ap_id, ext = 2, header = False, aspcapWavegrid = True)
        temp3 = apread.apStar(loc_id, ap_id, ext = 3, header = False, aspcapWavegrid = True)

    if temp1.shape[0] > 6000:
        spec = temp1
        specerr = temp2
        mask = temp3
    else:
        spec = temp1[_COMBINED_INDEX]
        specerr = temp2[_COMBINED_INDEX]
        mask = temp3[_COMBINED_INDEX]

    # Inflate uncertainties for bad pixels    
    specerr[(mask & (badcombpixmask)) != 0] += 100*np.mean(spec[np.isfinite(spec)])

    # Inflate pixels with high SNR to 0.5
    highsnr = spec/specerr > 200.
    specerr[highsnr] = 0.005*np.fabs(spec[highsnr])

    # Continuum-normalize
    cont = utils.get_apogee_continuum(wavelength = wavelength, spec = spec, 
        spec_err = specerr, cont_pixels = cont_pixels)
    spec /= cont
    specerr /= cont
    specerr[highsnr] = 0.005 
    
    if save_local:
        np.savez('spectra/combined/spectrum_ap_id_' + str(apogee_id.decode()) + '_.npz',
                 spectrum = spec, spec_err = specerr)
    return spec, specerr
    
def get_visit_spectra_individual_object(apogee_id, allvisit_cat = None, save_local = False):
    '''
    Download the visit spectra for an individual object. 
    Get the v_helios from the allStar catalog, which are more accurate than 
        the values reported in the visit spectra fits files. 
    Use the barycentric correction to shift spectra to the heliocentric frame.
    Do a preliminary normalization using Bovy's visit spectrum normalization tool. 
        It's critical that the spectra be renormalized prior to fitting using the 
        spectral_model.get_apogee_continuum() function for self-consistency. 
    apogee_id = byte-like object, i.e. b'2M06133561+2433362'
    
    ##### NOTE #####
    I had to make some minor changes to Bovy's APOGEE package to get it to 
    properly read in the visit spectra. In particular, after the 3rd line of 
    the function apVisit() in that package's /apogee/tools/read.py file, I added
    the following lines:
    
    if ext == 0:
        from astropy.io import fits
        hdulist = fits.open(filePath)
        header = hdulist[0].header 
        hdulist.close()
        return header
    
    I think this is the only change that was necessary, but it's possible I'm 
    forgetting another change. 
    '''
    import apogee.tools.read as apread
    from apogee.tools import bitmask
    from apogee.spec import continuum
    import astropy

    if allvisit_cat is None:
        allvisit_cat = apread.allVisit(rmcommissioning = False, ak = False)
    where_visits = np.where(allvisit_cat['APOGEE_ID'] == apogee_id)[0]
    
    plate_ids = np.array([int(i) for i in allvisit_cat[where_visits]['PLATE']])
    fiberids = allvisit_cat[where_visits]['FIBERID']
    mjds = allvisit_cat[where_visits]['MJD']
    JDs = allvisit_cat[where_visits]['JD']
    vhelios_accurate = allvisit_cat[where_visits]['VHELIO']
    vhelios_synth = allvisit_cat[where_visits]['SYNTHVHELIO']
    snrs = allvisit_cat[where_visits]['SNR']
    BCs = allvisit_cat[where_visits]['BC']
    
    badcombpixmask = bitmask.badpixmask() + 2**bitmask.apogee_pixmask_int("SIG_SKYLINE")
    all_spec, all_err, all_snr, all_hjd, all_vhelio = [], [], [], [], []
    
    for i, pid in enumerate(plate_ids):
        try:
            spec = apread.apVisit(pid, mjds[i], fiberids[i], ext=1, header=False)
            specerr = apread.apVisit(pid, mjds[i], fiberids[i], ext=2, header=False)
            wave = apread.apVisit(pid, mjds[i], fiberids[i], ext=4, header=False)
            mask = apread.apVisit(pid, mjds[i], fiberids[i], ext=3, header=False)
            masterheader = apread.apVisit(pid, mjds[i], fiberids[i], ext=0, header=True)
            
            badpix = (mask & (badcombpixmask)) != 0
            if np.sum(badpix)/len(badpix) > 0.5:
                print('too many bad pixels!')
                continue # if 50% or more of the pixels are bad, don't bother.
            specerr[badpix] = 100*np.median(spec)

            # a small fraction of the visit spectra are on a different wavelength 
            # grid than normal (maybe commissioning?). In any case, interpolate them 
            # to the wavelength grid expected by Bovy's visit normalization routine. 
            if len(wave) != 12288:
                print('fixing wavelength...')
                standard_grid = utils.load_visit_wavelength()
                spec = np.interp(standard_grid, wave, spec)
                specerr = np.interp(standard_grid, wave, specerr)
                wave = np.copy(standard_grid)
            
            # preliminary normalization using Bovy's visit normalization routine. 
            cont = continuum.fitApvisit(spec, specerr, wave)
            specnorm, errnorm = spec/cont, specerr/cont
        
            # correct for Earth's orbital motion. 
            spec_shift = utils.doppler_shift(wavelength = wave, flux = specnorm, dv = BCs[i])
            spec_err_shift = utils.doppler_shift(wavelength = wave, flux = errnorm, dv = BCs[i]) 
        
            # interpolate to the standard wavelength grid we use for combined spectra.
            interp_spec = np.interp(wavelength, wave, spec_shift)
            interp_err = np.interp(wavelength, wave, spec_err_shift)
            
            # truncate SNR at 200
            interp_err[interp_err < 0.005] = 0.005
            
            all_spec.append(interp_spec)
            all_err.append(interp_err)
            
            # Ideally, get the Julian date of the observations in the heliocentric frame. 
            # Sometimes this isn't available; in that case, get the Julian date in Earth's
            # frame. These differ by at most 8 minutes, so not a big deal. 
            try:
                all_hjd.append(masterheader['HJD'])
            except KeyError:
                all_hjd.append(JDs[i])
            
            # There are a few cases where the v_helios from the allvisit catalog are clearly wrong. 
            if np.abs(vhelios_accurate[i] > 1000) and np.abs(vhelios_synth[i] < 1000):
                vhel = vhelios_synth[i]
            else: vhel = vhelios_accurate[i]
    
            all_snr.append(snrs[i])
            all_vhelio.append(vhel)
        except astropy.io.fits.verify.VerifyError:
            print('there was a verification error')
            continue
    all_spec, all_err, all_snr, all_hjd, all_vhelio = np.array(all_spec), \
        np.array(all_err), np.array(all_snr), np.array(all_hjd), np.array(all_vhelio)
    msk = np.argsort(all_hjd)
    all_spec, all_err, all_snr, all_hjd, all_vhelio = all_spec[msk], all_err[msk], \
        all_snr[msk], all_hjd[msk], all_vhelio[msk]
    
    if save_local:
        np.savez('spectra/visit/visit_spectra_ap_id_' + str(apogee_id.decode()) + '_.npz',
                 spectra = all_spec, spec_errs = all_err, snrs = all_snr, 
                 hjds = all_hjd, vhelios = all_vhelio)
                 
    return all_spec, all_err, all_snr, all_hjd, all_vhelio

def renormalize_visit_spectrum(norm_spec, spec_err, label_guess, NN_coeffs_norm,
    NN_coeffs_flux, v_helio):
    '''
    Because visit spectra are initially normalized using a different routine than 
        is implemented in the main spectral modle, then need to be normalized again.
    
    This first obtains the continuum for a synthetic single-star model with parameters
        given by label_guess, multiplies the spectrum by this continuum, and then 
        normalizes that "unnormalized" spectrum using the default normalization routine. 
        It isn't critical that label_guess be vary accurate, since it only supplies a 
        smooth continuum that is divided out again anyway, but it can help a bit. Normally,
        label_guess is obtained by fitting a single-star model to the combined spectrum.
    '''
    star_labels = label_guess[:5]
    labels = np.concatenate([star_labels, [v_helio]]) 
    flux_spec_synth = spectral_model.get_surface_flux_spectrum_single_star(labels = labels, 
        NN_coeffs_norm = NN_coeffs_norm, NN_coeffs_flux = NN_coeffs_flux)
    cont_synth = utils.get_apogee_continuum(wavelength = wavelength, spec = flux_spec_synth, 
        spec_err = None, cont_pixels = cont_pixels)
    flux_spec_data = cont_synth*norm_spec
    cont_data = utils.get_apogee_continuum(wavelength = wavelength, spec = flux_spec_data, 
        spec_err = spec_err, cont_pixels = cont_pixels)
    renormalized_spec = flux_spec_data/cont_data
    return renormalized_spec

def download_visit_spectra_single_object_and_renormalize(apogee_id, p0_single_combined, 
    NN_coeffs_norm, NN_coeffs_flux, allvisit_cat = None, snr_min = 30):
    '''
    Download the visit spectra for one object. Keep the visits with sufficiently high
    SNR. Normalize them in a way consistent with our model.
    '''
    all_spec, all_err, all_snr, all_hjd, all_vhelio = get_visit_spectra_individual_object(
        apogee_id = apogee_id, allvisit_cat = allvisit_cat, save_local = False)
    msk = all_snr > snr_min
    all_spec, all_err, all_snr, all_hjd, all_vhelio = all_spec[msk], all_err[msk], \
        all_snr[msk], all_hjd[msk], all_vhelio[msk]
        
    renorm_specs = []
    for i, spec in enumerate(all_spec):
        renorm_spec = renormalize_visit_spectrum(norm_spec = spec, spec_err = all_err[i],
            label_guess = p0_single_combined, NN_coeffs_norm = NN_coeffs_norm,
            NN_coeffs_flux = NN_coeffs_flux, v_helio = all_vhelio[i])
        renorm_specs.append(renorm_spec)
    renorm_specs = np.array(renorm_specs)
    return renorm_specs, all_err, all_snr, all_hjd, all_vhelio 