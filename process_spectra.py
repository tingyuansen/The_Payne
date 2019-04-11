'''
Code for reading in combined spectra.
This uses Jo Bovy's APOGEE package to read in spectra: https://github.com/jobovy/apogee
Any way that you can get your hands on the spectra should be fine, as long as you 

Here we adopt APOGEE DR14. Edit os.environs below for a later version of APOGEE data release.
Since our spectral model training set was normalized using the DR12 wavelength definition, 
even thought the spectra are from DR14, we will resample them into DR12 wavelength format.
'''

from __future__ import absolute_import, division, print_function # python2 compatibility
import numpy as np
import sys
import os
import utils
import spectral_model
import time

os.environ["SDSS_LOCAL_SAS_MIRROR"] = "data.sdss3.org/sas/"
os.environ["RESULTS_VERS"] = "l31c.2" #v603 for DR12, l30e.2 for DR13, l31c.2 for DR14
os.environ["APOGEE_APOKASC_REDUX"] = "v6.2a"

from apogee.tools import toAspcapGrid
import apogee.tools.read as apread


# read in the default wavelength array and the list of pixels used for fitting
wavelength = utils.load_wavelength_array()
cont_pixels = utils.load_cannon_contpixels()

def read_apogee_catalog():
    '''
    read in the catalog of info for all stars in a data release. 
    '''
    all_star_catalog = apread.allStar(rmcommissioning = False, rmdups = False, 
        main = False, raw = True)
    catalog_id = all_star_catalog['APOGEE_ID'].astype("str")
    return all_star_catalog, catalog_id

def get_combined_spectrum_single_object(apogee_id, catalog = None, save_local = False):
    '''
    apogee_id should be a byte-like object; i.e b'2M13012770+5754582'
    This downloads a single combined spectrum and the associated error array,
        and it normalizes both. 
    '''
    
    # read in the allStar catalog if you haven't already
    if catalog is None:
        catalog, catalog_id = read_apogee_catalog()
    
    _COMBINED_INDEX = 1
    
    msk = np.where(catalog_id == apogee_id)[0]
    if not len(msk):
        raise ValueError('the desired Apogee ID was not found in the allStar catalog.')

    field = catalog['FIELD'][msk[0]].decode()
    loc_id = catalog['LOCATION_ID'][msk[0]]
        
    if loc_id == 1:
        temp1 = apread.apStar(field, apogee_id, ext = 1, header = False, aspcapWavegrid = False)
        temp2 = apread.apStar(field, apogee_id, ext = 2, header = False, aspcapWavegrid = False)
        temp3 = apread.apStar(field, apogee_id, ext = 3, header = False, aspcapWavegrid = False)
    else:
        temp1 = apread.apStar(loc_id, apogee_id, ext = 1, header = False, aspcapWavegrid = False)
        temp2 = apread.apStar(loc_id, apogee_id, ext = 2, header = False, aspcapWavegrid = False)
        temp3 = apread.apStar(loc_id, apogee_id, ext = 3, header = False, aspcapWavegrid = False)

    if temp1.shape[0] > 6000:
        spec = temp1
        specerr = temp2
        mask = temp3
    else:
        spec = temp1[_COMBINED_INDEX]
        specerr = temp2[_COMBINED_INDEX]
        mask = temp3[_COMBINED_INDEX]

    spec = toAspcapGrid(spec, dr='12') # dr12 wavelength format
    specerr = toAspcapGrid(specerr, dr='12')
    
    # cull dead pixels
    choose = spec <= 0
    spec[choose] = 0.01
    specerr[choose] = np.max(np.abs(spec))*999.
        
    # continuum-normalize
    cont = utils.get_apogee_continuum(wavelength = wavelength, spec = spec, 
        spec_err = specerr, cont_pixels = cont_pixels)
    spec /= cont
    specerr /= cont
    
    if save_local:
        np.savez('spectra/combined/spectrum_ap_id_' + str(apogee_id.decode()) + '_.npz',
                 spectrum = spec, spec_err = specerr)
    return spec, specerr
    

