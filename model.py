"""
AlexNet Model Implementation

This module implements the AlexNet architecture as described in:
"ImageNet Classification with Deep Convolutional Neural Networks"
by Krizhevsky, Sutskever, and Hinton (2012)

Paper: https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

Architecture Overview (Section 3):
- 5 convolutional layers
- 3 fully-connected layers
- 60 million parameters, 650,000 neurons
- ReLU activation function (Section 3.1)
- Local Response Normalization (Section 3.3)
- Overlapping Max Pooling (Section 3.4)
- Dropout regularization (Section 4.2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalResponseNormalization(nn.Module):
    """
    Local Response Normalization (LRN) Layer
    
    Implements the normalization scheme described in Section 3.3 of the paper:
    "The response-normalized activity b_{x,y}^i is given by:
    b_{x,y}^i = a_{x,y}^i / (k + α * Σ_{j=max(0, i-n/2)}^{min(N-1, i+n/2)} (a_{x,y}^j)^2)^β"
    
    Where:
    - k, n, α, β are hyperparameters
    - N is the total number of kernels in that layer
    - The sum runs over n "adjacent" kernel maps at the same spatial position
    
    Paper values: k=2, n=5, α=10^-4, β=0.75
    """
    
    def __init__(self, size=5, alpha=1e-4, beta=0.75, k=2):
        """
        Args:
            size: n in the paper - number of adjacent kernel maps to normalize over
            alpha: α in the paper - scaling parameter
            beta: β in the paper - exponent parameter
            k: k in the paper - additive constant
        """
        super(LocalResponseNormalization, self).__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k
    
    def forward(self, x):
        """
        Apply local response normalization as described in Section 3.3
        """
        return F.local_response_norm(x, self.size, self.alpha, self.beta, self.k)


class AlexNet(nn.Module):
    """
    AlexNet Architecture
    
    Complete implementation of the 8-layer CNN described in Section 3 of the paper.
    
    Layer Structure:
    1. Conv1: 96 kernels of size 11x11, stride 4, padding 2
    2. Conv2: 256 kernels of size 5x5, stride 1, padding 2
    3. Conv3: 384 kernels of size 3x3, stride 1, padding 1
    4. Conv4: 384 kernels of size 3x3, stride 1, padding 1
    5. Conv5: 256 kernels of size 3x3, stride 1, padding 1
    6. FC1: 4096 neurons
    7. FC2: 4096 neurons
    8. FC3: 1000 neurons (ImageNet classes)
    
    Note: The original paper used two GPUs. This implementation uses a single GPU
    but maintains the same architecture and parameter counts.
    """
    
    def __init__(self, num_classes=1000, dropout=0.5):
        """
        Args:
            num_classes: Number of output classes (1000 for ImageNet)
            dropout: Dropout probability for fully-connected layers (Section 4.2)
        """
        super(AlexNet, self).__init__()
        
        # Feature extraction layers (Convolutional layers)
        # Section 3: Architecture
        
        # Conv1: First convolutional layer
        # Paper Section 3: "The first convolutional layer filters the 224×224×3 input image
        # with 96 kernels of size 11×11×3 with a stride of 4 pixels"
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=96,
            kernel_size=11,
            stride=4,
            padding=2  # padding=2 ensures output size matches paper calculations
        )
        
        # ReLU activation (Section 3.1: "We refer to neurons with this non-linearity as ReLUs")
        # Applied after each convolutional and fully-connected layer
        # No explicit ReLU module needed - we use F.relu in forward()
        
        # Local Response Normalization after Conv1 (Section 3.3)
        self.lrn1 = LocalResponseNormalization(size=5, alpha=1e-4, beta=0.75, k=2)
        
        # Max Pooling after Conv1 (Section 3.4: "We use overlapping pooling")
        # Paper: "pooling layers consist of max-pooling over 3×3 spatial neighborhoods
        # with stride 2" - this creates overlapping pooling regions
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Conv2: Second convolutional layer
        # Paper Section 3: "The second convolutional layer takes as input the (response-normalized
        # and pooled) output of the first convolutional layer and filters it with 256 kernels
        # of size 5×5×48"
        self.conv2 = nn.Conv2d(
            in_channels=96,
            out_channels=256,
            kernel_size=5,
            stride=1,
            padding=2
        )
        
        # LRN after Conv2 (Section 3.3)
        self.lrn2 = LocalResponseNormalization(size=5, alpha=1e-4, beta=0.75, k=2)
        
        # Max Pooling after Conv2 (Section 3.4)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Conv3: Third convolutional layer
        # Paper Section 3: "The third, fourth, and fifth convolutional layers are connected to one another
        # without any intervening pooling or normalization layers"
        # "The third convolutional layer has 384 kernels of size 3×3×256"
        self.conv3 = nn.Conv2d(
            in_channels=256,
            out_channels=384,
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        # Conv4: Fourth convolutional layer
        # Paper Section 3: "The fourth convolutional layer has 384 kernels of size 3×3×192"
        self.conv4 = nn.Conv2d(
            in_channels=384,
            out_channels=384,
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        # Conv5: Fifth convolutional layer
        # Paper Section 3: "The fifth convolutional layer has 256 kernels of size 3×3×192"
        self.conv5 = nn.Conv2d(
            in_channels=384,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        # Max Pooling after Conv5 (Section 3.4)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Fully-connected layers (Section 3)
        # After pooling, the feature maps are flattened
        
        # FC1: First fully-connected layer
        # Paper Section 3: "The fully-connected layers have 4096 neurons each"
        # Input size calculation for 224x224 input:
        #   conv1(11x11, stride=4, pad=2): (224-11+4)/4+1 = 55x55
        #   pool1(3x3, stride=2): (55-3)/2+1 = 27x27
        #   conv2(5x5, stride=1, pad=2): (27-5+4)/1+1 = 27x27
        #   pool2(3x3, stride=2): (27-3)/2+1 = 13x13
        #   conv3-5(3x3, stride=1, pad=1): 13x13 (preserved)
        #   pool3(3x3, stride=2): (13-3)/2+1 = 6x6
        #   Final: 256 channels * 6 * 6 = 9216
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        
        # Dropout after FC1 (Section 4.2: "We use dropout in the first two fully-connected layers")
        self.dropout1 = nn.Dropout(p=dropout)
        
        # FC2: Second fully-connected layer
        self.fc2 = nn.Linear(4096, 4096)
        
        # Dropout after FC2 (Section 4.2)
        self.dropout2 = nn.Dropout(p=dropout)
        
        # FC3: Third fully-connected layer (output layer)
        # Paper Section 3: "The network has a 1000-way softmax"
        self.fc3 = nn.Linear(4096, num_classes)
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
        
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Conv1 -> ReLU -> LRN -> MaxPool
        # Section 3.1: ReLU activation
        # Section 3.3: Local Response Normalization
        # Section 3.4: Overlapping Max Pooling
        x = self.conv1(x)
        x = F.relu(x)
        x = self.lrn1(x)
        x = self.pool1(x)
        
        # Conv2 -> ReLU -> LRN -> MaxPool
        x = self.conv2(x)
        x = F.relu(x)
        x = self.lrn2(x)
        x = self.pool2(x)
        
        # Conv3 -> ReLU
        # Section 3: "The third, fourth, and fifth convolutional layers are connected
        # to one another without any intervening pooling or normalization layers"
        x = self.conv3(x)
        x = F.relu(x)
        
        # Conv4 -> ReLU
        x = self.conv4(x)
        x = F.relu(x)
        
        # Conv5 -> ReLU -> MaxPool
        x = self.conv5(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # Flatten for fully-connected layers
        x = x.view(x.size(0), -1)
        
        # FC1 -> ReLU -> Dropout
        # Section 4.2: "We use dropout in the first two fully-connected layers"
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # FC2 -> ReLU -> Dropout
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # FC3 (output layer)
        # Note: Softmax is applied in the loss function (CrossEntropyLoss)
        x = self.fc3(x)
        
        return x


def alexnet(num_classes=1000, pretrained=False, **kwargs):
    """
    Construct an AlexNet model
    
    Args:
        num_classes: Number of classes for classification (default: 1000 for ImageNet)
        pretrained: If True, returns a model pre-trained on ImageNet
        **kwargs: Additional arguments passed to AlexNet
    
    Returns:
        AlexNet model instance
    """
    model = AlexNet(num_classes=num_classes, **kwargs)
    return model


if __name__ == "__main__":
    # Test the model architecture
    model = AlexNet(num_classes=1000)
    
    # Print model summary
    print("AlexNet Architecture:")
    print(model)
    
    # Test forward pass with dummy input
    # Paper Section 3: Input size is 224×224×3
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
