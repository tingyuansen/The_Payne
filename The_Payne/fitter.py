"""
PayneFitter class for fitting observed spectra to determine stellar parameters.
Provides a clean object-oriented interface for spectral fitting.
"""

from __future__ import absolute_import, division, print_function
import numpy as np
from . import fitting
from . import utils
from .predictor import PaynePredictor


class PayneFitter:
    """
    Class for fitting observed spectra to determine stellar parameters.
    
    This class wraps The Payne fitting functionality and provides methods
    for fitting single spectra and batch fitting.
    
    Attributes:
        predictor (PaynePredictor): Predictor instance for spectrum generation
        mask (np.ndarray): Pixel mask for bad pixels
        wavelength (np.ndarray): Wavelength array
        NN_coeffs (tuple): Neural network coefficients
    
    Example:
        >>> fitter = PayneFitter()
        >>> labels, uncertainties, model = fitter.fit_spectrum(observed_spec, spec_err)
        >>> print(f"Teff = {labels[0]:.0f} ± {uncertainties[0]:.0f} K")
    """
    
    def __init__(self, use_mask=True, nn_path=None):
        """
        Initialize the PayneFitter.
        
        Parameters:
            use_mask (bool): Whether to use the default APOGEE mask (default: True)
            nn_path (str, optional): Path to custom neural network file. If None, uses default.
        """
        # Initialize predictor
        self.predictor = PaynePredictor(nn_path=nn_path)
        
        # Load mask and wavelength
        if use_mask:
            self.mask = utils.load_apogee_mask()
        else:
            self.mask = np.zeros(self.predictor.num_pixels, dtype=bool)
        
        self.wavelength = self.predictor.wavelength
        self.NN_coeffs = self.predictor.NN_coeffs
        
    def fit_spectrum(self, spectrum, spec_err, p0=None, return_scaled=False):
        """
        Fit a normalized spectrum to determine stellar parameters.
        
        Parameters:
            spectrum (np.ndarray): Normalized observed spectrum
            spec_err (np.ndarray): Uncertainty array for spectrum
            p0 (np.ndarray, optional): Initial guess for parameters.
                If None, starts at midpoint of label space.
            return_scaled (bool): If True, also return scaled labels (default: False)
            
        Returns:
            labels (np.ndarray): Best-fit stellar labels in physical units
            uncertainties (np.ndarray): Formal 1-sigma uncertainties
            model_spectrum (np.ndarray): Best-fit model spectrum
            (scaled_labels) (np.ndarray, optional): Scaled labels if return_scaled=True
        """
        # Fit the spectrum
        popt, pstd, model_spec = fitting.fit_normalized_spectrum_single_star_model(
            spectrum, spec_err, self.NN_coeffs, self.wavelength, self.mask, p0=p0
        )
        
        # Separate labels and RV
        labels = popt[:-1]
        rv = popt[-1]
        uncertainties = pstd[:-1]
        rv_err = pstd[-1]
        
        # Add RV to uncertainties for consistent output
        labels_with_rv = np.append(labels, rv)
        uncertainties_with_rv = np.append(uncertainties, rv_err)
        
        if return_scaled:
            # Convert to scaled space
            scaled_labels = (labels - self.predictor.x_min) / \
                           (self.predictor.x_max - self.predictor.x_min) - 0.5
            return labels_with_rv, uncertainties_with_rv, model_spec, scaled_labels
        else:
            return labels_with_rv, uncertainties_with_rv, model_spec
    
    def fit_spectrum_with_initial_guess(self, spectrum, spec_err, initial_labels, initial_rv=0.0):
        """
        Fit spectrum with a specific initial guess for labels.
        
        Parameters:
            spectrum (np.ndarray): Normalized observed spectrum
            spec_err (np.ndarray): Uncertainty array
            initial_labels (np.ndarray): Initial guess for labels (physical units)
            initial_rv (float): Initial guess for radial velocity (km/s)
            
        Returns:
            labels (np.ndarray): Best-fit stellar labels (including RV)
            uncertainties (np.ndarray): Formal uncertainties
            model_spectrum (np.ndarray): Best-fit model spectrum
        """
        # Convert initial guess to scaled space
        scaled_initial = self.predictor.scale_labels(initial_labels)
        p0 = np.append(scaled_initial, initial_rv)
        
        return self.fit_spectrum(spectrum, spec_err, p0=p0)
    
    def compute_chi2(self, spectrum, spec_err, labels, rv=0.0):
        """
        Compute chi-squared for given parameters.
        
        Parameters:
            spectrum (np.ndarray): Observed spectrum
            spec_err (np.ndarray): Uncertainty array
            labels (np.ndarray): Stellar labels in physical units
            rv (float): Radial velocity in km/s
            
        Returns:
            float: Chi-squared value
            float: Reduced chi-squared
            int: Number of degrees of freedom
        """
        # Predict model spectrum
        model_spec = self.predictor.predict_spectrum(labels, rv=rv)
        
        # Compute chi-squared on unmasked pixels
        good_pixels = ~self.mask
        residuals = (spectrum[good_pixels] - model_spec[good_pixels]) / spec_err[good_pixels]
        chi2 = np.sum(residuals**2)
        
        # Compute reduced chi-squared
        n_params = len(labels) + 1  # +1 for RV
        dof = np.sum(good_pixels) - n_params
        chi2_reduced = chi2 / dof if dof > 0 else np.inf
        
        return chi2, chi2_reduced, dof
    
    def get_residuals(self, spectrum, model_spectrum):
        """
        Compute residuals between observed and model spectrum.
        
        Parameters:
            spectrum (np.ndarray): Observed spectrum
            model_spectrum (np.ndarray): Model spectrum
            
        Returns:
            np.ndarray: Residuals (observed - model)
        """
        return spectrum - model_spectrum
    
    def get_label_names(self):
        """
        Get label names including RV.
        
        Returns:
            list: List of label names
        """
        names = self.predictor.get_label_names()
        return names + ['RV']
    
    def set_mask(self, mask):
        """
        Set a custom pixel mask.
        
        Parameters:
            mask (np.ndarray): Boolean array where True = masked (bad) pixel
        """
        if len(mask) != self.predictor.num_pixels:
            raise ValueError(f"Mask length {len(mask)} doesn't match spectrum length {self.predictor.num_pixels}")
        self.mask = mask
    
    def add_mask_region(self, wave_min, wave_max):
        """
        Add a wavelength region to the mask.
        
        Parameters:
            wave_min (float): Minimum wavelength to mask (Angstroms)
            wave_max (float): Maximum wavelength to mask (Angstroms)
        """
        mask_region = (self.wavelength >= wave_min) & (self.wavelength <= wave_max)
        self.mask = self.mask | mask_region
    
    def reset_mask(self):
        """Reset mask to default APOGEE mask."""
        self.mask = utils.load_apogee_mask()
    
    def __repr__(self):
        masked_frac = 100 * np.sum(self.mask) / len(self.mask)
        return (f"PayneFitter(num_labels={self.predictor.num_labels}, "
                f"masked_pixels={masked_frac:.1f}%)")

