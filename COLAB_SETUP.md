# Google Colab Setup Guide

This guide will help you set up and train AlexNet in Google Colab.

## Step 1: Clone the Repository

```python
# Clone the repository
!git clone <your-repository-url>
%cd AlexNet
```

## Step 2: Install Dependencies

```python
# Install required packages
!pip install -r requirements.txt
```

## Step 3: Download ImageNet Dataset

You have several options:

### Option A: Download from ImageNet (Official)

1. Register at [ImageNet](http://www.image-net.org/)
2. Download ILSVRC2012 dataset
3. Upload to Google Drive
4. Mount Google Drive in Colab:

```python
from google.colab import drive
drive.mount('/content/drive')
```

5. Extract the dataset if needed:

```python
# If you have a tar file
!tar -xzf /content/drive/MyDrive/ILSVRC2012_img_train.tar -C /content/
!tar -xzf /content/drive/MyDrive/ILSVRC2012_img_val.tar -C /content/
```

### Option B: Use Pre-processed Dataset

If you have a pre-processed ImageNet dataset, mount it from Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')

# Update the path in config.py
import sys
sys.path.append('/content/AlexNet')
```

## Step 4: Organize Dataset Structure

Ensure your dataset is organized as:

```
imagenet/
├── train/
│   ├── n01440764/
│   │   ├── n01440764_18.JPEG
│   │   └── ...
│   ├── n01443537/
│   └── ...
└── val/
    ├── n01440764/
    ├── n01443537/
    └── ...
```

## Step 5: Update Configuration

Edit `config.py` to set the correct data path:

```python
# In config.py, update:
DATA_DIR = "/content/imagenet"  # or your path
```

Or modify it directly in Colab:

```python
import config
config.config.DATA_DIR = "/content/imagenet"
```

## Step 6: Enable Training

Edit `train.py` to enable training:

```python
# In train.py, change:
RUN_TRAINING = True
```

Or modify it directly:

```python
import train
train.RUN_TRAINING = True
```

## Step 7: Start Training

```python
# Run training
!python train.py
```

Or run in the background:

```python
# Run in background (optional)
import subprocess
process = subprocess.Popen(['python', 'train.py'], 
                          stdout=subprocess.PIPE, 
                          stderr=subprocess.PIPE)
```

## Step 8: Monitor Training

Check the output for:
- Training loss and accuracy
- Validation loss and accuracy
- Learning rate adjustments
- Checkpoint saves

## Step 9: Save Checkpoints

Checkpoints are saved in the `checkpoints/` directory. To download:

```python
from google.colab import files

# Download best model
files.download('checkpoints/best_model.pth')

# Or download entire checkpoints folder
!zip -r checkpoints.zip checkpoints/
files.download('checkpoints.zip')
```

## Tips for Colab

1. **GPU Runtime**: Make sure to enable GPU in Colab:
   - Runtime → Change runtime type → GPU

2. **Memory Management**: If you run out of memory:
   - Reduce batch size in `config.py`
   - Use gradient accumulation

3. **Resume Training**: To resume from a checkpoint:
   ```python
   checkpoint = torch.load('checkpoints/checkpoint_epoch_10.pth')
   model.load_state_dict(checkpoint['model_state_dict'])
   optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
   ```

4. **Time Limits**: Colab has time limits for free tier. Consider:
   - Saving checkpoints frequently
   - Using Colab Pro for longer sessions

## Troubleshooting

### Out of Memory Error
- Reduce `BATCH_SIZE` in `config.py`
- Reduce `NUM_WORKERS` in `config.py`

### Dataset Not Found
- Check `DATA_DIR` path in `config.py`
- Verify dataset structure matches expected format

### Import Errors
- Make sure you're in the correct directory: `%cd AlexNet`
- Install dependencies: `!pip install -r requirements.txt`

## Expected Training Time

- Full training (90 epochs) on ImageNet: ~5-7 days on a single GPU
- Per epoch: ~1-2 hours depending on GPU
- Validation: ~10-15 minutes per epoch

## Quick Test

Before full training, test the model:

```python
from model import AlexNet
import torch

model = AlexNet(num_classes=1000)
dummy_input = torch.randn(1, 3, 224, 224)
output = model(dummy_input)
print(f"Output shape: {output.shape}")  # Should be (1, 1000)
```
