"""
The Payne - Neural network interpolation for stellar spectra

Main classes:
    PaynePredictor - For predicting spectra from stellar labels
    PayneFitter - For fitting observed spectra to derive stellar parameters
    PayneTrainer - For training new neural networks

Legacy modules (functional interface):
    spectral_model - Low-level spectrum prediction
    fitting - Low-level spectrum fitting
    training - Low-level neural network training
    utils - Utility functions
"""

from .predictor import PaynePredictor
from .fitter import PayneFitter
from .trainer import PayneTrainer

__all__ = ['PaynePredictor', 'PayneFitter', 'PayneTrainer']

