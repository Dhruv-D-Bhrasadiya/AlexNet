"""
Data Augmentation for AlexNet

This module implements data augmentation techniques as described in:
"ImageNet Classification with Deep Convolutional Neural Networks"
by Krizhevsky, Sutskever, and Hinton (2012)

Paper Section 4.1: "The easiest and most common method to reduce overfitting
on image data is to artificially enlarge the dataset using label-preserving
transformations."

Key augmentations from the paper:
1. Image cropping and horizontal reflection (Section 4.1)
2. Intensity variations (Section 4.1)
3. PCA-based color augmentation (Section 4.1)
"""

import torch
import torchvision.transforms as transforms
import numpy as np


class PCAColorAugmentation:
    """
    PCA-based Color Augmentation
    
    Implements the color augmentation scheme described in Section 4.1 of the paper:
    "We perform PCA on the set of RGB pixel values throughout the ImageNet training set.
    To each training image, we add multiples of the found principal components, with
    magnitudes proportional to the corresponding eigenvalues times a random variable
    drawn from a Gaussian with mean zero and standard deviation 0.1."
    
    The paper states: "This scheme approximately captures an important property of
    natural images, namely, that object identity is invariant to changes in the
    intensity and color of the illumination."
    """
    
    def __init__(self, alpha_std=0.1):
        """
        Args:
            alpha_std: Standard deviation for the Gaussian random variable (default: 0.1)
        """
        self.alpha_std = alpha_std
        
        # PCA components and eigenvalues for ImageNet
        # These are pre-computed from ImageNet training set statistics
        # Paper Section 4.1: "We perform PCA on the set of RGB pixel values"
        # These values are standard ImageNet PCA statistics
        self.eigenvalues = torch.tensor([0.2175, 0.0188, 0.0045])
        self.eigenvectors = torch.tensor([
            [-0.5675, 0.7192, 0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948, 0.4203]
        ])
    
    def __call__(self, img):
        """
        Apply PCA-based color augmentation to an image
        
        Args:
            img: PIL Image or tensor of shape (C, H, W)
        
        Returns:
            Augmented image
        """
        # Convert to tensor if PIL Image
        if not isinstance(img, torch.Tensor):
            to_tensor = transforms.ToTensor()
            img = to_tensor(img)
        
        # Get random alphas from Gaussian distribution
        # Paper: "random variable drawn from a Gaussian with mean zero and standard deviation 0.1"
        alpha = torch.randn(3) * self.alpha_std
        
        # Compute color shift
        # Paper: "multiples of the found principal components, with magnitudes
        # proportional to the corresponding eigenvalues times a random variable"
        rgb_shift = torch.sum(
            self.eigenvectors * alpha.unsqueeze(1) * self.eigenvalues.unsqueeze(0),
            dim=0
        )
        
        # Apply shift to each channel
        # Reshape for broadcasting: (3,) -> (3, 1, 1)
        rgb_shift = rgb_shift.view(3, 1, 1)
        augmented_img = img + rgb_shift
        
        # Clamp values to [0, 1] range
        augmented_img = torch.clamp(augmented_img, 0.0, 1.0)
        
        return augmented_img


def get_train_transforms():
    """
    Get training data augmentation transforms
    
    Implements the augmentation strategy from Section 4.1 of the paper:
    1. Random cropping: "We extract random 224×224 patches (and their horizontal
       reflections) from the 256×256 images"
    2. Horizontal reflection: "We extract random 224×224 patches (and their horizontal
       reflections)"
    3. PCA color augmentation: "We perform PCA on the set of RGB pixel values"
    
    Returns:
        Composition of transforms for training data
    """
    return transforms.Compose([
        # Resize to 256x256 (larger than input size for cropping)
        # Paper Section 4.1: "We extract random 224×224 patches from the 256×256 images"
        transforms.Resize(256),
        
        # Random crop to 224x224
        # Paper Section 4.1: "We extract random 224×224 patches"
        transforms.RandomCrop(224),
        
        # Random horizontal flip
        # Paper Section 4.1: "and their horizontal reflections"
        transforms.RandomHorizontalFlip(p=0.5),
        
        # Convert to tensor
        transforms.ToTensor(),
        
        # PCA-based color augmentation
        # Paper Section 4.1: PCA color augmentation
        PCAColorAugmentation(alpha_std=0.1),
        
        # Normalize using ImageNet statistics
        # Note: The paper doesn't explicitly mention normalization, but it's standard practice
        # and helps with training stability
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_val_transforms():
    """
    Get validation/test data transforms
    
    For validation, we use center cropping instead of random cropping
    and no augmentation (Section 4.1: augmentation is only for training)
    
    Returns:
        Composition of transforms for validation data
    """
    return transforms.Compose([
        # Resize to 256x256
        transforms.Resize(256),
        
        # Center crop to 224x224
        # Paper: For validation, we use center crop (standard practice)
        transforms.CenterCrop(224),
        
        # Convert to tensor
        transforms.ToTensor(),
        
        # Normalize using ImageNet statistics
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


if __name__ == "__main__":
    # Test the augmentation transforms
    from PIL import Image
    import numpy as np
    
    # Create a dummy image
    dummy_img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    
    # Test training transforms
    train_transform = get_train_transforms()
    augmented = train_transform(dummy_img)
    print(f"Training transform output shape: {augmented.shape}")
    print(f"Training transform output range: [{augmented.min():.3f}, {augmented.max():.3f}]")
    
    # Test validation transforms
    val_transform = get_val_transforms()
    val_augmented = val_transform(dummy_img)
    print(f"Validation transform output shape: {val_augmented.shape}")
    print(f"Validation transform output range: [{val_augmented.min():.3f}, {val_augmented.max():.3f}]")
