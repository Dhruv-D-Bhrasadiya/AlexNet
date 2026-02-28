# Paper-to-Code Mapping

This document provides a detailed mapping between the AlexNet paper sections and the corresponding code implementations.

## Paper Reference

**"ImageNet Classification with Deep Convolutional Neural Networks"**  
Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012)  
NIPS 2012

## Section-by-Section Mapping

### Section 2: The Dataset

**Paper Content**: Description of ImageNet ILSVRC-2012 dataset

**Code Location**: 
- `dataloader.py`: `get_imagenet_info()` function
- `dataloader.py`: `get_imagenet_dataloaders()` function
- `config.py`: `NUM_CLASSES = 1000`

**Key Details**:
- 1.2 million training images
- 50,000 validation images
- 100,000 test images
- 1000 classes

### Section 3: The Architecture

#### Section 3: Overall Architecture

**Paper Content**: "The network has eight layers with weights; the first five are convolutional and the remaining three are fully-connected."

**Code Location**: 
- `model.py`: `AlexNet` class definition
- `model.py`: Lines 62-80 (class docstring)

**Implementation**: All 8 layers implemented in `model.py`

#### Section 3.1: ReLU Nonlinearity

**Paper Content**: "We refer to neurons with this non-linearity as ReLUs (Rectified Linear Units)."

**Code Location**: 
- `model.py`: `forward()` method - `F.relu()` calls after each conv and FC layer
- Lines 104-106: Comment explaining ReLU usage

**Implementation**: ReLU applied after every convolutional and fully-connected layer

#### Section 3.2: Training on Multiple GPUs

**Paper Content**: Description of two-GPU implementation

**Code Location**: 
- `model.py`: Line 78-79 (note about single GPU implementation)

**Note**: This implementation uses a single GPU but maintains the same architecture

#### Section 3.3: Local Response Normalization

**Paper Content**: "The response-normalized activity b_{x,y}^i is given by: b_{x,y}^i = a_{x,y}^i / (k + α * Σ_{j=max(0, i-n/2)}^{min(N-1, i+n/2)} (a_{x,y}^j)^2)^β"

**Code Location**: 
- `model.py`: `LocalResponseNormalization` class (lines 30-58)
- `model.py`: `lrn1` and `lrn2` instances (lines 109, 129)
- `model.py`: Forward pass usage (lines 218, 228)

**Parameters**:
- k = 2
- n = 5
- α = 10⁻⁴
- β = 0.75

#### Section 3.4: Overlapping Pooling

**Paper Content**: "Pooling layers consist of max-pooling over 3×3 spatial neighborhoods with stride 2."

**Code Location**: 
- `model.py`: `pool1`, `pool2`, `pool3` (lines 114, 132, 167)
- `model.py`: `nn.MaxPool2d(kernel_size=3, stride=2)`

**Implementation**: 3×3 max pooling with stride 2 creates overlapping regions

#### Section 3.5: Overall Architecture

**Paper Content**: Detailed layer-by-layer description

**Code Location**: `model.py` - Complete layer definitions:

| Layer | Paper Description | Code Location |
|-------|------------------|---------------|
| Conv1 | 96 kernels, 11×11, stride 4 | `model.py` lines 96-102 |
| Conv2 | 256 kernels, 5×5 | `model.py` lines 120-126 |
| Conv3 | 384 kernels, 3×3 | `model.py` lines 138-144 |
| Conv4 | 384 kernels, 3×3 | `model.py` lines 148-154 |
| Conv5 | 256 kernels, 3×3 | `model.py` lines 158-164 |
| FC1 | 4096 neurons | `model.py` line 180 |
| FC2 | 4096 neurons | `model.py` line 187 |
| FC3 | 1000 neurons | `model.py` line 194 |

### Section 4: Reducing Overfitting

#### Section 4.1: Data Augmentation

**Paper Content**: "The easiest and most common method to reduce overfitting on image data is to artificially enlarge the dataset using label-preserving transformations."

**Code Location**: 
- `data_augmentation.py`: Complete implementation

**Specific Augmentations**:

1. **Image Cropping and Horizontal Reflection**
   - Paper: "We extract random 224×224 patches (and their horizontal reflections) from the 256×256 images"
   - Code: `data_augmentation.py` lines 75-85
   - Implementation: `RandomCrop(224)` and `RandomHorizontalFlip(p=0.5)`

2. **PCA-based Color Augmentation**
   - Paper: "We perform PCA on the set of RGB pixel values throughout the ImageNet training set."
   - Code: `data_augmentation.py` `PCAColorAugmentation` class (lines 20-66)
   - Implementation: Adds multiples of principal components with Gaussian-distributed magnitudes

#### Section 4.2: Dropout

**Paper Content**: "We use dropout in the first two fully-connected layers with a dropout ratio of 0.5"

**Code Location**: 
- `model.py`: `dropout1` and `dropout2` (lines 183, 190)
- `model.py`: Forward pass usage (lines 260, 265)
- `config.py`: `DROPOUT = 0.5`

**Implementation**: 0.5 dropout applied to FC1 and FC2 outputs

### Section 4: Details of Learning

**Paper Content**: Training procedure and hyperparameters

**Code Location**: 
- `train.py`: Complete training implementation
- `config.py`: All hyperparameters

#### Weight Initialization

**Paper Content**: "We initialized the weights in each layer from a zero-mean Gaussian distribution with standard deviation 0.01."

**Code Location**: 
- `config.py`: Lines 20-24 (documented, but PyTorch uses default initialization)
- Note: PyTorch's default initialization differs; can be customized if needed

**Paper Content**: "We initialized the neuron biases in the second, fourth, and fifth convolutional layers, as well as in the fully-connected hidden layers, with the constant 1."

**Code Location**: 
- `config.py`: Lines 25-27 (documented)
- Note: Can be implemented in model initialization if exact replication needed

#### Learning Rate

**Paper Content**: "We used an equal learning rate for all layers, which we adjusted manually throughout training. The heuristic which we followed was to divide the learning rate by 10 when the validation error rate stopped improving with the current learning rate. The learning rate was initialized at 0.01 and reduced three times prior to termination."

**Code Location**: 
- `config.py`: `INITIAL_LEARNING_RATE = 0.01` (line 28)
- `config.py`: `LEARNING_RATE_DECAY_FACTOR = 0.1` (line 29)
- `train.py`: `ReduceLROnPlateau` scheduler (lines 200-206)

#### Momentum and Weight Decay

**Paper Content**: 
- "We used a momentum of 0.9"
- "We used a weight decay of 0.0005"

**Code Location**: 
- `config.py`: `MOMENTUM = 0.9` (line 33)
- `config.py`: `WEIGHT_DECAY = 0.0005` (line 37)
- `train.py`: SGD optimizer configuration (lines 192-197)

#### Batch Size and Training Duration

**Paper Content**: 
- "We trained our networks using stochastic gradient descent with a batch size of 128 examples"
- "We trained the network for roughly 90 cycles through the training set"

**Code Location**: 
- `config.py`: `BATCH_SIZE = 128` (line 16)
- `config.py`: `NUM_EPOCHS = 90` (line 41)
- `train.py`: Training loop (lines 210-250)

#### Loss Function

**Paper Content**: "The objective function is the multinomial logistic regression objective (i.e., softmax followed by cross-entropy loss)"

**Code Location**: 
- `train.py`: `criterion = nn.CrossEntropyLoss()` (line 185)
- `train.py`: Loss computation (line 140)

### Section 5: Results

**Paper Content**: Performance metrics and results

**Code Location**: 
- `train.py`: `validate()` function computes top-1 accuracy
- `train.py`: Training loop tracks and saves best validation accuracy

**Paper Results**:
- Top-1 validation error: 37.5%
- Top-5 validation error: 17.0%
- Top-1 test error: 40.7%
- Top-5 test error: 18.2%

## Summary Table

| Paper Section | Code File | Key Components |
|--------------|-----------|----------------|
| Section 2 | `dataloader.py`, `config.py` | Dataset info, data loaders |
| Section 3.1 | `model.py` | ReLU activations |
| Section 3.3 | `model.py` | LocalResponseNormalization class |
| Section 3.4 | `model.py` | MaxPool2d layers |
| Section 3 | `model.py` | Complete AlexNet architecture |
| Section 4.1 | `data_augmentation.py` | All augmentation techniques |
| Section 4.2 | `model.py`, `config.py` | Dropout layers |
| Section 4 | `train.py`, `config.py` | Training procedure, hyperparameters |

## Verification Checklist

- [x] All 8 layers implemented (5 conv + 3 FC)
- [x] ReLU activation after each layer
- [x] Local Response Normalization (after conv1 and conv2)
- [x] Overlapping max pooling (3×3, stride 2)
- [x] Dropout in FC layers (0.5)
- [x] Data augmentation (cropping, flipping, PCA)
- [x] Training hyperparameters match paper
- [x] All components documented with paper references
