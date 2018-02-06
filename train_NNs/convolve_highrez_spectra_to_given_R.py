'''
This code reads in a batch of very high resolution spectra and degrades them
to a lower resolution, assuming a Gaussian line-spread function. The main use
case is that we produce synthetic spectra (from an updated version of the Kurucz 
line list by default) at R~300,000 and need to convolve them the resolution of 
APOGEE before training the NN spectral model. Note, the high-res spectra are not
normalized.

If we were doing ab-initio fitting, it would be important to use the correct
(non-Gaussian) LSF from APOGEE, but since we just use the synthetic spectral
model to predict the continuum, this isn't important. 

For a few hundred ab-initio spectra, this runs on my laptop in a minute or two. 

I have not included the high-res model spectra that this operates on, as they're 
big files, but they're available up request if an example is needed. 
'''
from __future__ import absolute_import, division, print_function # python2 compatibility
import numpy as np
from scipy import sparse
from scipy.stats import norm
from scipy import interpolate
from multiprocessing import Pool
import utils
wavelength_template = utils.load_wavelength_array()

# restore spectra
temp = np.load('/path/to/highres_spectra.npz')
wavelength = temp['wavelength'] # this is the R~300,000 wavelength, *not* our default grid.
spectra = temp['spectra']
labels = temp['labels']
temp.close()

def sparsify_matrix(lsf):
    '''
    This just speeds up the computation, since the convolution matrix gets very large. 
    '''
    nx = lsf.shape[1]
    diagonals = []
    offsets = []
    for ii in range(nx):
        offset = nx//2 - ii
        offsets.append(offset)
        if offset < 0:
            diagonals.append(lsf[:offset, ii])
        else:
            diagonals.append(lsf[offset:, ii])
    return sparse.diags(diagonals, offsets)

R_res = 22500 # for apogee
start_wavelength = 15001
end_wavelength = 16999

# interpolation parameters. The interpolation resolution just needs to be 
# significantly better than the final resolution of the wavelength grid
inv_wl_res = 100
wl_res = 1./inv_wl_res
wl_range = end_wavelength - start_wavelength

# make convolution grid
wavelength_run = wl_res*np.arange(wl_range/wl_res + 1)+ start_wavelength

# determines where we can cut off the convolution kernel. 
template_width = np.median(np.diff(wavelength_template))

# how many kernel bin to keep
R_range = int(template_width/wl_res + 0.5)*5

# pad wavelength with zeros for convolution
wavelength_tmp = np.concatenate([np.zeros(R_range), wavelength_run, 
    np.zeros(R_range)])

# create convolution matrix. Each column is a Gaussian kernel with 
# different FWHM, with the FWMH equal to  FWHM = lambda/R_res
conv_matrix = np.zeros((len(wavelength_run), 2*R_range+1))
for i in range(len(wavelength_run)):
    this_wl = wavelength_tmp[i:(i + 2*R_range + 1)] - wavelength_tmp[i + R_range]
    this_scale = wavelength_tmp[i+R_range]/(R_res*2.355) # convert from FWHM to sigma. 
    this_kernel = norm.pdf(this_wl, scale = this_scale)*wl_res
    conv_matrix[i, :] = this_kernel
conv_sparse = sparsify(conv_matrix)

def convolve_spectrum(c1):
    '''
    convolve a single spectrum. Pass this to multiprocessing. 
    '''
    # interpolate spectra into the convolution unit
    f_flux_spec = interpolate.interp1d(wavelength, spectra[c1,:])
    full_spec = f_flux_spec(wavelength_run)
    
    # convolve spectrum
    convolved_flux = conv_sparse.dot(full_spec)
    f_flux_1D = interpolate.interp1d(wavelength_run, convolved_flux)

    # return convolved spectrum
    print('convolved spectrum number %d', c1)
    return f_flux_1D(wavelength_template)

# convolve multiple spectra in parallel
pool = Pool(multiprocessing.cpu_count())
spectra = pool.map(convolve_spectrum, range(spectra.shape[0]))

# save the convolved spectra and their labels 
np.savez('/path/to/convolved_synthetic_spectra.npz',
    labels = labels, spectra = spectra, wavelength = wavelength_template)