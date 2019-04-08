'''
Code for reading in combined spectra.
This uses Jo Bovy's APOGEE package to read in spectra: https://github.com/jobovy/apogee
Any way that you can get your hands on the spectra should be fine, as long as you 
(a) set the uncertainties high in bad pixels, (b) normalize them using the 
spectral_model.get_apogee_continuum() function, and (c) set a max S/N of 200

I've only used it for spectra from DR14, DR16
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
    

