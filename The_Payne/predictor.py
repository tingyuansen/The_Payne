"""
PaynePredictor class for spectral prediction using pre-trained neural networks.
Provides a clean object-oriented interface for spectrum prediction.
"""

from __future__ import absolute_import, division, print_function
import numpy as np
from . import spectral_model
from . import utils


class PaynePredictor:
    """
    Class for predicting stellar spectra from stellar labels using The Payne.
    
    This class wraps the neural network prediction functionality and provides
    an easy-to-use interface for spectrum interpolation.
    
    Attributes:
        NN_coeffs (tuple): Neural network weights, biases, and scaling parameters
        wavelength (np.ndarray): Wavelength array in Angstroms
        x_min (np.ndarray): Minimum values for label scaling
        x_max (np.ndarray): Maximum values for label scaling
        num_labels (int): Number of stellar labels
        num_pixels (int): Number of wavelength pixels
    
    Example:
        >>> predictor = PaynePredictor()
        >>> labels = predictor.get_default_labels()
        >>> labels[0] = 5500  # Teff = 5500 K
        >>> spectrum = predictor.predict_spectrum(labels)
    """
    
    def __init__(self, nn_path=None, wavelength_path=None):
        """
        Initialize the PaynePredictor.
        
        Parameters:
            nn_path (str, optional): Path to neural network file. If None, uses default.
            wavelength_path (str, optional): Path to wavelength file. If None, uses default.
        """
        # Load neural network
        if nn_path is not None:
            tmp = np.load(nn_path)
            w_array_0 = tmp["w_array_0"]
            w_array_1 = tmp["w_array_1"]
            w_array_2 = tmp["w_array_2"]
            b_array_0 = tmp["b_array_0"]
            b_array_1 = tmp["b_array_1"]
            b_array_2 = tmp["b_array_2"]
            x_min = tmp["x_min"]
            x_max = tmp["x_max"]
            tmp.close()
            self.NN_coeffs = (w_array_0, w_array_1, w_array_2, b_array_0, b_array_1, b_array_2, x_min, x_max)
        else:
            self.NN_coeffs = utils.read_in_neural_network()
        
        self.w_array_0, self.w_array_1, self.w_array_2, \
            self.b_array_0, self.b_array_1, self.b_array_2, \
            self.x_min, self.x_max = self.NN_coeffs
        
        # Load wavelength array
        if wavelength_path is not None:
            tmp = np.load(wavelength_path)
            self.wavelength = tmp['wavelength']
            tmp.close()
        else:
            self.wavelength = utils.load_wavelength_array()
        
        # Store dimensions
        self.num_labels = self.w_array_0.shape[1]
        self.num_pixels = len(self.wavelength)
        
    def scale_labels(self, labels):
        """
        Scale labels from physical units to neural network input space [-0.5, 0.5].
        
        Parameters:
            labels (np.ndarray): Array of stellar labels in physical units
            
        Returns:
            np.ndarray: Scaled labels in range [-0.5, 0.5]
        """
        scaled = (labels - self.x_min) / (self.x_max - self.x_min) - 0.5
        return scaled
    
    def unscale_labels(self, scaled_labels):
        """
        Convert scaled labels back to physical units.
        
        Parameters:
            scaled_labels (np.ndarray): Scaled labels in range [-0.5, 0.5]
            
        Returns:
            np.ndarray: Labels in physical units
        """
        labels = (scaled_labels + 0.5) * (self.x_max - self.x_min) + self.x_min
        return labels
    
    def predict_spectrum(self, labels, rv=0.0):
        """
        Predict a normalized spectrum from stellar labels.
        
        Parameters:
            labels (np.ndarray): Stellar labels in physical units
                [Teff, Logg, Vturb, [X/H], ..., C12/C13, Vmacro]
            rv (float): Radial velocity in km/s (default: 0.0)
            
        Returns:
            np.ndarray: Predicted normalized spectrum
        """
        # Scale labels
        scaled_labels = self.scale_labels(labels)
        
        # Predict spectrum
        spectrum = spectral_model.get_spectrum_from_neural_net(scaled_labels, self.NN_coeffs)
        
        # Apply radial velocity shift if needed
        if abs(rv) > 1e-6:
            spectrum = utils.doppler_shift(self.wavelength, spectrum, rv)
        
        return spectrum
    
    def predict_spectrum_scaled(self, scaled_labels, rv=0.0):
        """
        Predict spectrum from already-scaled labels.
        
        Parameters:
            scaled_labels (np.ndarray): Scaled labels in range [-0.5, 0.5]
            rv (float): Radial velocity in km/s (default: 0.0)
            
        Returns:
            np.ndarray: Predicted normalized spectrum
        """
        spectrum = spectral_model.get_spectrum_from_neural_net(scaled_labels, self.NN_coeffs)
        
        if abs(rv) > 1e-6:
            spectrum = utils.doppler_shift(self.wavelength, spectrum, rv)
        
        return spectrum
    
    def get_default_labels(self):
        """
        Get default (mid-range) stellar labels as starting point.
        
        Returns:
            np.ndarray: Default labels at midpoint of training range
        """
        return (self.x_min + self.x_max) / 2.0
    
    def get_label_names(self):
        """
        Get standard label names for APOGEE model.
        
        Returns:
            list: List of label names
        """
        names = ['Teff', 'logg', 'Vturb',
                 '[C/H]', '[N/H]', '[O/H]', '[Na/H]', '[Mg/H]',
                 '[Al/H]', '[Si/H]', '[P/H]', '[S/H]', '[K/H]',
                 '[Ca/H]', '[Ti/H]', '[V/H]', '[Cr/H]', '[Mn/H]',
                 '[Fe/H]', '[Co/H]', '[Ni/H]', '[Cu/H]', '[Ge/H]',
                 'C12/C13', 'Vmacro']
        return names[:self.num_labels]
    
    def get_wavelength(self):
        """
        Get the wavelength array.
        
        Returns:
            np.ndarray: Wavelength array in Angstroms
        """
        return self.wavelength
    
    def __repr__(self):
        return (f"PaynePredictor(num_labels={self.num_labels}, "
                f"num_pixels={self.num_pixels}, "
                f"wavelength_range=[{self.wavelength[0]:.2f}, {self.wavelength[-1]:.2f}] Å)")

