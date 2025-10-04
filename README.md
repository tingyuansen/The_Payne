# The Payne
Tools for interpolating spectral models with neural networks. This package uses neural networks to predict stellar spectra from stellar labels (effective temperature, surface gravity, chemical abundances, etc.) and fit observed spectra to determine stellar parameters.

## Table of Contents
- [Installation](#installation)
- [Key Features](#key-features)
- [Quick Start](#quick-start)
  - [Use Case 1: Predicting Stellar Spectra](#use-case-1-predicting-stellar-spectra)
  - [Use Case 2: Fitting Observed Spectra](#use-case-2-fitting-observed-spectra)
  - [Use Case 3: Training a New Neural Network](#use-case-3-training-a-new-neural-network)
- [Testing](#testing)
- [Stellar Labels](#stellar-labels)
- [Project Overview](#project-overview)

## Installation
Clone this repository and run code from the base directory:
```bash
python setup.py install
```

The [tutorial.ipynb](tutorial.ipynb) shows simple use cases for fitting stellar spectra.

## Dependencies
* **Spectral model and fitting**: Numpy and Scipy
* **Training neural networks**: [PyTorch](http://pytorch.org/) (GPU required)
* All dependencies will be automatically installed with this package
* Developed in Python 3.7+ using Anaconda

## Project Overview

### Core Modules

**`spectral_model.py`** - Neural network prediction
- `get_spectrum_from_neural_net()`: Predicts normalized stellar spectra from scaled stellar labels using a trained neural network
- `leaky_relu()`: Activation function used in the neural networks

**`fitting.py`** - Spectral fitting routines
- `fit_normalized_spectrum_single_star_model()`: Fits a single-star model to an observed spectrum using optimization to determine best-fit stellar parameters (Teff, logg, abundances, radial velocity, etc.)

**`training.py`** - Neural network training (requires GPU)
- `neural_net()`: Trains a neural network on a grid of synthetic spectra to learn the mapping from stellar labels to spectra
- `Payne_model`: PyTorch neural network architecture

**`utils.py`** - Utility functions
- `read_in_neural_network()`: Loads pre-trained neural network weights and biases
- `load_wavelength_array()`: Loads the APOGEE wavelength grid
- `load_apogee_mask()`: Loads pixel mask for bad pixels during fitting
- `doppler_shift()`: Applies Doppler shift to spectra for radial velocity
- `get_apogee_continuum()`: Continuum normalizes APOGEE spectra
- `load_training_data()`: Loads Kurucz synthetic training spectra

**`process_spectra.py`** - APOGEE data handling
- `read_apogee_catalog()`: Downloads and reads APOGEE allStar catalog
- `get_combined_spectrum_single_object()`: Downloads and processes a single APOGEE combined spectrum
- `toAspcapGrid()`: Converts apStar wavelength grid to ASPCAP grid

**`radam.py`** - Optimization algorithms
- `RAdam`: Rectified Adam optimizer for neural network training
- `PlainRAdam`, `AdamW`: Alternative optimizers

### Data Files

- `neural_nets/`: Pre-trained neural network coefficients
- `other_data/`: APOGEE wavelength grids, masks, and Kurucz training spectra


## Class-Based Interface
- **PaynePredictor**: Easy spectrum prediction from stellar parameters
- **PayneFitter**: Automated fitting with uncertainty estimation
- **PayneTrainer**: Simplified neural network training with progress tracking



## Quick Start

### Use Case 1: Predicting Stellar Spectra

Use **PaynePredictor** to generate synthetic spectra from stellar parameters:

```python
from The_Payne import PaynePredictor

# Initialize the predictor
predictor = PaynePredictor()
print(predictor)  # Shows model info: labels, pixels, wavelength range

# Get default stellar parameters (mid-range values)
labels = predictor.get_default_labels()

# Customize stellar parameters
labels[0] = 5500   # Teff = 5500 K
labels[1] = 4.5    # logg = 4.5
labels[18] = 0.0   # [Fe/H] = 0.0 (solar metallicity)

# Predict spectrum with radial velocity
spectrum = predictor.predict_spectrum(labels, rv=25.0)  # rv in km/s
wavelength = predictor.get_wavelength()

# View label names
label_names = predictor.get_label_names()
print(label_names)  # ['Teff', 'logg', 'Vturb', '[C/H]', ...]

# Scale labels for neural network input (if needed)
scaled_labels = predictor.scale_labels(labels)
```

### Use Case 2: Fitting Observed Spectra

Use **PayneFitter** to determine stellar parameters from observed spectra:

```python
from The_Payne import PayneFitter
import numpy as np

# Initialize the fitter
fitter = PayneFitter(use_mask=True)  # Uses default APOGEE mask
print(fitter)  # Shows model info and masked pixel percentage

# Fit your observed spectrum
# observed_spec: normalized flux array
# spec_err: uncertainty array (same length as spectrum)
fitted_labels, uncertainties, model_spectrum = fitter.fit_spectrum(
    observed_spec, spec_err
)

# Display results
print(f"Teff:  {fitted_labels[0]:7.1f} ± {uncertainties[0]:5.1f} K")
print(f"logg:  {fitted_labels[1]:7.2f} ± {uncertainties[1]:5.2f}")
print(f"[Fe/H]: {fitted_labels[18]:6.2f} ± {uncertainties[18]:5.2f}")
print(f"RV:    {fitted_labels[-1]:7.2f} ± {uncertainties[-1]:5.2f} km/s")

# Evaluate fit quality
chi2, chi2_reduced, dof = fitter.compute_chi2(
    observed_spec, spec_err, fitted_labels[:-1], fitted_labels[-1]
)
print(f"Reduced χ²: {chi2_reduced:.3f}")

# Get residuals
residuals = fitter.get_residuals(observed_spec, model_spectrum)

# Customize masking (e.g., mask a telluric region)
fitter.add_mask_region(15890, 15920)  # wavelength in Angstroms
fitter.reset_mask()  # Reset to default if needed
```

### Use Case 3: Training a New Neural Network

Use **PayneTrainer** to train your own models on custom spectral grids:

```python
from The_Payne import PayneTrainer
from The_Payne import utils

# Load training data (or use your own)
train_labels, train_spectra, valid_labels, valid_spectra = utils.load_training_data()

# Initialize trainer
trainer = PayneTrainer(
    training_labels=train_labels,
    training_spectra=train_spectra,
    validation_labels=valid_labels,
    validation_spectra=valid_spectra,
    num_neurons=300,      # neurons per hidden layer
    use_cuda=True         # use GPU (highly recommended)
)

print(trainer)  # Shows training configuration

# Train the network
history = trainer.train(
    num_steps=10000,           # training iterations
    learning_rate=1e-4,        # learning rate
    batch_size=512,            # batch size
    save_path="my_network.npz", # where to save best model
    verbose=True               # print progress
)

# Training automatically saves the best model based on validation loss
# You can also manually save
trainer.save_network("final_network.npz")
trainer.save_training_history("training_history.npz")

# View training progress
import matplotlib.pyplot as plt
plt.plot(history['training_loss'], label='Training')
plt.plot(history['validation_loss'], label='Validation')
plt.xlabel('Step (x100)')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

### Additional Examples

For complete working examples, see:
- `examples/quick_start.py` - Three complete examples with synthetic data
- `tutorial.ipynb` - Jupyter notebook with detailed workflows

## Typical Workflow

1. **Using pre-trained models**: Load neural network → predict spectra for given stellar parameters or fit observed spectra to derive parameters
2. **Training new models**: Prepare training spectra → train neural network on GPU → save network coefficients → use for predictions/fitting

## Stellar Labels

The default model predicts 25 labels:
- Physical parameters: Teff, log(g), microturbulent velocity
- Chemical abundances: [C/H], [N/H], [O/H], [Na/H], [Mg/H], [Al/H], [Si/H], [P/H], [S/H], [K/H], [Ca/H], [Ti/H], [V/H], [Cr/H], [Mn/H], [Fe/H], [Co/H], [Ni/H], [Cu/H], [Ge/H]
- Other: C12/C13, macroturbulent velocity, radial velocity

## Citing this code
Please cite [Ting et al. 2019](https://ui.adsabs.harvard.edu/abs/2019ApJ...879...69T/abstract) when using this code. The paper describes the method and its application to APOGEE spectra.

## Authors
* [Yuan-Sen Ting](http://www.ysting.space) -- ting dot 74 at osu dot edu

## License
Copyright 2018 by Yuan-Sen Ting.

This software is governed by the MIT License: In brief, you can use, distribute, and change this package as you please.
