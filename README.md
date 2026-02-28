# AlexNet Paper Replication

This repository contains a complete implementation of the AlexNet architecture as described in the paper:

**"ImageNet Classification with Deep Convolutional Neural Networks"**  
by Krizhevsky, Sutskever, and Hinton (2012)  
[Paper Link](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Code-to-Paper Mapping](#code-to-paper-mapping)
- [Key Features](#key-features)
- [Documentation](#documentation)
- [References](#references)

## Overview

This implementation faithfully replicates the AlexNet architecture and training procedure from the original paper. All code includes detailed comments linking each component to specific sections of the paper.

**Note:** This repository is configured to be ready for training in Google Colab. The training script is disabled by default to prevent accidental execution on local machines without proper resources.

## Architecture

AlexNet is an 8-layer deep convolutional neural network:

- **5 Convolutional Layers** (Section 3)
- **3 Fully-Connected Layers** (Section 3)
- **60 million parameters, 650,000 neurons** (Section 3)

### Key Components

1. **ReLU Activation** (Section 3.1): Non-saturating neurons for faster training
2. **Local Response Normalization** (Section 3.3): Normalization scheme for improved generalization
3. **Overlapping Max Pooling** (Section 3.4): 3×3 pooling with stride 2
4. **Dropout Regularization** (Section 4.2): 0.5 dropout in fully-connected layers
5. **Data Augmentation** (Section 4.1): Random cropping, horizontal reflection, and PCA color augmentation

## Repository Structure

```
AlexNet/
├── model.py                 # AlexNet architecture implementation
├── data_augmentation.py     # Data augmentation (Section 4.1)
├── dataloader.py           # ImageNet data loading utilities
├── train.py                # Training script (Section 4)
├── config.py               # Hyperparameters from paper
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── AlexNet_1.pdf           # Original paper
```

## Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd AlexNet
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download ImageNet Dataset

The ImageNet ILSVRC-2012 dataset is required for training. You need to:

1. Download the dataset from [ImageNet](http://www.image-net.org/)
2. Organize it in the following structure:
   ```
   imagenet/
   ├── train/
   │   ├── n01440764/
   │   ├── n01443537/
   │   └── ...
   └── val/
       ├── n01440764/
       ├── n01443537/
       └── ...
   ```

3. Update the `DATA_DIR` path in `config.py`

## Usage

### For Google Colab

1. **Clone the repository in Colab:**
   ```python
   !git clone <repository-url>
   %cd AlexNet
   ```

2. **Install dependencies:**
   ```python
   !pip install -r requirements.txt
   ```

3. **Download ImageNet dataset** (or mount from Google Drive)

4. **Update config.py** with the correct `DATA_DIR` path

5. **Enable training** in `train.py`:
   ```python
   RUN_TRAINING = True
   ```

6. **Run training:**
   ```python
   !python train.py
   ```

### Testing the Model

You can test the model architecture without training:

```python
from model import AlexNet

# Create model
model = AlexNet(num_classes=1000)

# Test with dummy input
import torch
dummy_input = torch.randn(1, 3, 224, 224)
output = model(dummy_input)
print(f"Output shape: {output.shape}")  # Should be (1, 1000)
```

## Code-to-Paper Mapping

### Model Architecture (`model.py`)

| Code Component | Paper Section | Description |
|---------------|---------------|-------------|
| `Conv1` | Section 3 | First conv layer: 96 kernels, 11×11, stride 4 |
| `Conv2` | Section 3 | Second conv layer: 256 kernels, 5×5 |
| `Conv3-5` | Section 3 | Third-fifth conv layers: 384, 384, 256 kernels, 3×3 |
| `LocalResponseNormalization` | Section 3.3 | LRN with k=2, n=5, α=10⁻⁴, β=0.75 |
| `MaxPool2d` | Section 3.4 | Overlapping pooling: 3×3 kernel, stride 2 |
| `ReLU` | Section 3.1 | Non-saturating activation function |
| `Dropout` | Section 4.2 | 0.5 dropout in FC layers |
| `FC1-3` | Section 3 | Fully-connected layers: 4096, 4096, 1000 |

### Data Augmentation (`data_augmentation.py`)

| Code Component | Paper Section | Description |
|---------------|---------------|-------------|
| `RandomCrop(224)` | Section 4.1 | Extract 224×224 patches from 256×256 images |
| `RandomHorizontalFlip` | Section 4.1 | Horizontal reflection |
| `PCAColorAugmentation` | Section 4.1 | PCA-based color augmentation |

### Training (`train.py`)

| Hyperparameter | Paper Section | Value |
|----------------|---------------|-------|
| Batch Size | Section 4 | 128 |
| Learning Rate | Section 4 | 0.01 (initial) |
| Momentum | Section 4 | 0.9 |
| Weight Decay | Section 4 | 0.0005 |
| Epochs | Section 4 | ~90 |
| Dropout | Section 4.2 | 0.5 |
| Optimizer | Section 4 | SGD |

### Configuration (`config.py`)

All hyperparameters are documented with their corresponding paper sections and values.

## Key Features

✅ **Complete Architecture**: All 8 layers implemented exactly as described  
✅ **Paper References**: Every component linked to specific paper sections  
✅ **Data Augmentation**: Full implementation of Section 4.1 augmentations  
✅ **Training Script**: Complete training loop with paper hyperparameters  
✅ **Google Colab Ready**: Configured for easy Colab deployment  
✅ **Well Documented**: Extensive comments explaining design choices

## Documentation

This repository includes comprehensive documentation:

- **[README.md](README.md)**: Main repository documentation
- **[PAPER_MAPPING.md](PAPER_MAPPING.md)**: Detailed section-by-section mapping between paper and code
- **[COLAB_SETUP.md](COLAB_SETUP.md)**: Step-by-step guide for setting up training in Google Colab
- **Code Comments**: Every file contains inline comments linking to paper sections  

## Training Details (From Paper)

### Initialization (Section 4)
- Weights: Zero-mean Gaussian, std=0.01
- Biases: 1 for conv2, conv4, conv5, and FC layers; 0 elsewhere

### Learning Rate Schedule (Section 4)
- Initial: 0.01
- Reduced by factor of 10 when validation error stops improving
- Reduced 3 times before termination

### Results (Section 5)
- Top-1 error: 37.5% (validation), 40.7% (test)
- Top-5 error: 17.0% (validation), 18.2% (test)

## References

1. **Original Paper**: Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. *Advances in neural information processing systems*, 25.

2. **ImageNet Dataset**: Deng, J., Dong, W., Socher, R., Li, L. J., Li, K., & Fei-Fei, L. (2009). Imagenet: A large-scale hierarchical image database. *2009 IEEE conference on computer vision and pattern recognition*.

## License

This implementation is for educational and research purposes. Please refer to the original paper and ImageNet dataset for their respective licenses.

## Contributing

This is a replication project. If you find any discrepancies with the paper, please open an issue.

## Acknowledgments

- Original AlexNet paper authors: Krizhevsky, Sutskever, and Hinton
- ImageNet dataset creators
- PyTorch team for the excellent deep learning framework
