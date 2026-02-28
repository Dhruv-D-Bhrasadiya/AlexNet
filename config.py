"""
Configuration file for AlexNet training

This file contains hyperparameters and training settings as described in:
"ImageNet Classification with Deep Convolutional Neural Networks"
by Krizhevsky, Sutskever, and Hinton (2012)

All hyperparameters are linked to their corresponding paper sections.
"""

import torch


class Config:
    """
    Configuration class containing all hyperparameters from the paper
    """
    
    # Dataset settings
    # Paper Section 2: ImageNet 2012 training set
    DATA_DIR = "/path/to/imagenet"  # Update this path to your ImageNet directory
    NUM_CLASSES = 1000  # Paper Section 2: 1000 classes
    
    # Training hyperparameters
    # Paper Section 4: Training details
    
    # Batch size
    # Paper Section 4: "We trained our networks using stochastic gradient descent
    # with a batch size of 128 examples"
    BATCH_SIZE = 128
    
    # Learning rate
    # Paper Section 4: "We initialized the weights in each layer from a zero-mean
    # Gaussian distribution with standard deviation 0.01. We initialized the neuron
    # biases in the second, fourth, and fifth convolutional layers, as well as in the
    # fully-connected hidden layers, with the constant 1. This initialization
    # accelerated the early stages of learning by providing the ReLUs with positive inputs.
    # We initialized the neuron biases in the remaining layers with the constant 0."
    # Paper Section 4: "We used an equal learning rate for all layers, which we adjusted
    # manually throughout training. The heuristic which we followed was to divide the
    # learning rate by 10 when the validation error rate stopped improving with the
    # current learning rate. The learning rate was initialized at 0.01 and reduced
    # three times prior to termination."
    INITIAL_LEARNING_RATE = 0.01
    LEARNING_RATE_DECAY_FACTOR = 0.1  # Divide by 10
    LEARNING_RATE_DECAY_PATIENCE = 3  # Reduce when validation error stops improving
    
    # Momentum
    # Paper Section 4: "We used a momentum of 0.9"
    MOMENTUM = 0.9
    
    # Weight decay
    # Paper Section 4: "We used a weight decay of 0.0005"
    WEIGHT_DECAY = 0.0005
    
    # Number of epochs
    # Paper Section 4: "We trained the network for roughly 90 cycles through the training set"
    NUM_EPOCHS = 90
    
    # Dropout
    # Paper Section 4.2: "We use dropout in the first two fully-connected layers
    # with a dropout ratio of 0.5"
    DROPOUT = 0.5
    
    # Data loading
    NUM_WORKERS = 4  # Number of data loading workers
    PIN_MEMORY = True  # Pin memory for faster GPU transfer
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model saving
    SAVE_DIR = "./checkpoints"
    SAVE_FREQUENCY = 10  # Save checkpoint every N epochs
    
    # Logging
    LOG_FREQUENCY = 100  # Print training info every N batches
    
    # Validation
    VAL_FREQUENCY = 1  # Validate every N epochs
    
    # Random seed for reproducibility
    RANDOM_SEED = 42


# Create a global config instance
config = Config()
