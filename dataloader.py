"""
Data Loading Utilities for ImageNet

This module provides data loading functionality for the ImageNet dataset
as used in the AlexNet paper.

Paper: "ImageNet Classification with Deep Convolutional Neural Networks"
by Krizhevsky, Sutskever, and Hinton (2012)

Dataset: ImageNet ILSVRC-2012
- Training set: ~1.2 million images
- Validation set: 50,000 images
- Test set: 100,000 images
- 1000 classes

Note: This module does NOT download the dataset. The dataset should be
downloaded separately and organized in the standard ImageNet directory structure:
    imagenet/
    ├── train/
    │   ├── n01440764/
    │   ├── n01443537/
    │   └── ...
    └── val/
        ├── n01440764/
        ├── n01443537/
        └── ...
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from data_augmentation import get_train_transforms, get_val_transforms


def get_imagenet_dataloaders(data_dir, batch_size=128, num_workers=4, pin_memory=True):
    """
    Create ImageNet data loaders for training and validation
    
    Paper Section 2: "We trained our networks on the ImageNet 2012 training set
    (1.2 million images, 1000 classes)"
    
    Args:
        data_dir: Path to ImageNet dataset root directory
                  Expected structure: data_dir/train/ and data_dir/val/
        batch_size: Batch size for training (Paper Section 4: batch size of 128)
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
    
    Returns:
        train_loader: DataLoader for training set
        val_loader: DataLoader for validation set
    """
    
    # Training dataset with augmentation
    # Paper Section 4.1: Data augmentation is applied during training
    train_dataset = datasets.ImageFolder(
        root=f"{data_dir}/train",
        transform=get_train_transforms()
    )
    
    # Validation dataset (no augmentation)
    val_dataset = datasets.ImageFolder(
        root=f"{data_dir}/val",
        transform=get_val_transforms()
    )
    
    # Training data loader
    # Paper Section 4: "We trained our networks using stochastic gradient descent
    # with a batch size of 128 examples"
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop last incomplete batch for consistent batch size
    )
    
    # Validation data loader
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation data
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader


def get_imagenet_info():
    """
    Get information about the ImageNet dataset
    
    Returns:
        Dictionary with dataset information
    """
    return {
        "name": "ImageNet ILSVRC-2012",
        "training_images": 1281167,  # ~1.2 million
        "validation_images": 50000,
        "test_images": 100000,
        "num_classes": 1000,
        "image_size": (224, 224),  # Input size for AlexNet
        "reference": "Paper Section 2: 'We trained our networks on the ImageNet 2012 training set'"
    }


if __name__ == "__main__":
    # Example usage (will fail if dataset not present)
    print("ImageNet Dataset Information:")
    info = get_imagenet_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("\nNote: To use the dataloader, you need to:")
    print("1. Download ImageNet dataset from http://www.image-net.org/")
    print("2. Organize it in the following structure:")
    print("   imagenet/")
    print("   ├── train/")
    print("   │   ├── n01440764/")
    print("   │   └── ...")
    print("   └── val/")
    print("       ├── n01440764/")
    print("       └── ...")
    print("3. Pass the path to get_imagenet_dataloaders()")
