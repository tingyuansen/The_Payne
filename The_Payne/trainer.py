"""
PayneTrainer class for training new neural networks on spectral grids.
Provides a clean object-oriented interface for neural network training.
"""

from __future__ import absolute_import, division, print_function
import numpy as np
import os
import torch
from torch.autograd import Variable
from . import radam
from . import training


class PayneTrainer:
    """
    Class for training neural networks to emulate stellar spectra.
    
    This class provides an object-oriented interface for training The Payne
    neural networks on grids of synthetic or observed spectra.
    
    Attributes:
        training_labels (np.ndarray): Training set labels
        training_spectra (np.ndarray): Training set spectra
        validation_labels (np.ndarray): Validation set labels
        validation_spectra (np.ndarray): Validation set spectra
        model (torch.nn.Module): Neural network model
        num_labels (int): Number of stellar labels
        num_pixels (int): Number of wavelength pixels
    
    Example:
        >>> trainer = PayneTrainer(train_labels, train_spectra, 
        ...                        valid_labels, valid_spectra)
        >>> trainer.train(num_steps=1000, learning_rate=1e-4)
        >>> trainer.save_network("my_network.npz")
    """
    
    def __init__(self, training_labels, training_spectra, 
                 validation_labels, validation_spectra,
                 num_neurons=300, use_cuda=True):
        """
        Initialize the PayneTrainer.
        
        Parameters:
            training_labels (np.ndarray): Training labels [n_train, n_labels]
            training_spectra (np.ndarray): Training spectra [n_train, n_pixels]
            validation_labels (np.ndarray): Validation labels [n_valid, n_labels]
            validation_spectra (np.ndarray): Validation spectra [n_valid, n_pixels]
            num_neurons (int): Number of neurons in hidden layers (default: 300)
            use_cuda (bool): Whether to use GPU (default: True)
        """
        self.training_labels = training_labels
        self.training_spectra = training_spectra
        self.validation_labels = validation_labels
        self.validation_spectra = validation_spectra
        
        self.num_labels = training_labels.shape[1]
        self.num_pixels = training_spectra.shape[1]
        self.num_neurons = num_neurons
        
        # Check CUDA availability
        self.use_cuda = use_cuda and torch.cuda.is_available()
        if use_cuda and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Using CPU.")
            self.use_cuda = False
        
        # Set device
        if self.use_cuda:
            self.dtype = torch.cuda.FloatTensor
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.dtype = torch.FloatTensor
            torch.set_default_tensor_type('torch.FloatTensor')
        
        # Scale the labels
        self.x_min = np.min(training_labels, axis=0)
        self.x_max = np.max(training_labels, axis=0)
        self.x_train = (training_labels - self.x_min) / (self.x_max - self.x_min) - 0.5
        self.x_valid = (validation_labels - self.x_min) / (self.x_max - self.x_min) - 0.5
        
        # Initialize model
        self.model = None
        self.optimizer = None
        self.training_loss_history = []
        self.validation_loss_history = []
        self.best_model_state = None
        
    def initialize_model(self, num_features=320, mask_size=11):
        """
        Initialize the neural network model.
        
        Parameters:
            num_features (int): Number of features (for ResNet, not used in MLP)
            mask_size (int): Mask size (for ResNet, not used in MLP)
        """
        self.model = training.Payne_model(
            self.num_labels, self.num_neurons, 
            num_features, mask_size, self.num_pixels
        )
        
        if self.use_cuda:
            self.model.cuda()
        
        self.model.train()
        
    def train(self, num_steps=10000, learning_rate=1e-4, batch_size=512,
              save_path="NN_normalized_spectra.npz", verbose=True):
        """
        Train the neural network.
        
        Parameters:
            num_steps (int): Number of training steps (default: 10000)
            learning_rate (float): Learning rate (default: 1e-4)
            batch_size (int): Batch size for training (default: 512)
            save_path (str): Path to save best model (default: "NN_normalized_spectra.npz")
            verbose (bool): Whether to print progress (default: True)
            
        Returns:
            dict: Training history with losses
        """
        if self.model is None:
            if verbose:
                print("Initializing model...")
            self.initialize_model()
        
        # Initialize optimizer
        self.optimizer = radam.RAdam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=learning_rate
        )
        
        # Loss function
        loss_fn = torch.nn.L1Loss(reduction='mean')
        
        # Convert to PyTorch tensors
        x_train = Variable(torch.from_numpy(self.x_train)).type(self.dtype)
        y_train = Variable(torch.from_numpy(self.training_spectra), 
                          requires_grad=False).type(self.dtype)
        x_valid = Variable(torch.from_numpy(self.x_valid)).type(self.dtype)
        y_valid = Variable(torch.from_numpy(self.validation_spectra), 
                          requires_grad=False).type(self.dtype)
        
        # Training setup
        nsamples = x_train.shape[0]
        nbatches = nsamples // batch_size
        nsamples_valid = x_valid.shape[0]
        nbatches_valid = nsamples_valid // batch_size
        
        current_loss = np.inf
        self.training_loss_history = []
        self.validation_loss_history = []
        
        if verbose:
            print(f"Training on {nsamples} spectra, validating on {nsamples_valid} spectra")
            print(f"Batch size: {batch_size}, Number of batches: {nbatches}")
            print(f"Using device: {'GPU' if self.use_cuda else 'CPU'}")
            print("-" * 70)
        
        # Training loop
        for step in range(int(num_steps)):
            # Randomly permute data
            perm = torch.randperm(nsamples)
            if self.use_cuda:
                perm = perm.cuda()
            
            # Train on batches
            for i in range(nbatches):
                idx = perm[i * batch_size : (i+1) * batch_size]
                y_pred = self.model(x_train[idx])
                
                loss = loss_fn(y_pred, y_train[idx]) * 1e4
                self.optimizer.zero_grad()
                loss.backward(retain_graph=False)
                self.optimizer.step()
            
            # Evaluate validation loss
            if step % 100 == 0:
                perm_valid = torch.randperm(nsamples_valid)
                if self.use_cuda:
                    perm_valid = perm_valid.cuda()
                
                loss_valid = 0
                for j in range(nbatches_valid):
                    idx = perm_valid[j * batch_size : (j+1) * batch_size]
                    y_pred_valid = self.model(x_valid[idx])
                    loss_valid += loss_fn(y_pred_valid, y_valid[idx]) * 1e4
                loss_valid /= nbatches_valid
                
                if verbose:
                    print(f'Step {step:5d}: train_loss = {loss:.3f}, '
                          f'valid_loss = {loss_valid:.3f}')
                
                loss_data = loss.detach().data.item()
                loss_valid_data = loss_valid.detach().data.item()
                self.training_loss_history.append(loss_data)
                self.validation_loss_history.append(loss_valid_data)
                
                # Save best model
                if loss_valid_data < current_loss:
                    current_loss = loss_valid_data
                    self.save_network(save_path)
                    if verbose and step > 0:
                        print(f"  → New best model saved (valid_loss = {loss_valid_data:.3f})")
        
        if verbose:
            print("-" * 70)
            print(f"Training complete! Best validation loss: {current_loss:.3f}")
            print(f"Model saved to: {save_path}")
        
        return {
            'training_loss': self.training_loss_history,
            'validation_loss': self.validation_loss_history,
            'best_loss': current_loss
        }
    
    def save_network(self, filepath="NN_normalized_spectra.npz"):
        """
        Save the neural network weights and biases.
        
        Parameters:
            filepath (str): Path to save the network
        """
        model_numpy = []
        for param in self.model.parameters():
            model_numpy.append(param.data.cpu().numpy())
        
        w_array_0 = model_numpy[0]
        b_array_0 = model_numpy[1]
        w_array_1 = model_numpy[2]
        b_array_1 = model_numpy[3]
        w_array_2 = model_numpy[4]
        b_array_2 = model_numpy[5]
        
        np.savez(filepath,
                 w_array_0=w_array_0, w_array_1=w_array_1, w_array_2=w_array_2,
                 b_array_0=b_array_0, b_array_1=b_array_1, b_array_2=b_array_2,
                 x_min=self.x_min, x_max=self.x_max)
    
    def save_training_history(self, filepath="training_loss.npz"):
        """
        Save training history.
        
        Parameters:
            filepath (str): Path to save training history
        """
        np.savez(filepath,
                 training_loss=self.training_loss_history,
                 validation_loss=self.validation_loss_history)
    
    def __repr__(self):
        return (f"PayneTrainer(num_labels={self.num_labels}, "
                f"num_pixels={self.num_pixels}, "
                f"num_neurons={self.num_neurons}, "
                f"device={'GPU' if self.use_cuda else 'CPU'})")

