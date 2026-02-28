"""
Training script for AlexNet

This script implements the training procedure described in:
"ImageNet Classification with Deep Convolutional Neural Networks"
by Krizhevsky, Sutskever, and Hinton (2012)

Paper Section 4: Training details

IMPORTANT: This script is configured to NOT run automatically.
Set RUN_TRAINING = True to enable training, but make sure you have:
1. Downloaded the ImageNet dataset
2. Updated the DATA_DIR path in config.py
3. Have sufficient GPU memory and compute resources

For Google Colab usage:
1. Clone this repository
2. Download ImageNet dataset
3. Update config.py with the correct DATA_DIR
4. Run this script
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.backends.cudnn as cudnn

from model import AlexNet
from dataloader import get_imagenet_dataloaders
from config import config

# Set this to True to enable training
# Set to False to prevent accidental execution
RUN_TRAINING = False


def set_seed(seed):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    Train the model for one epoch
    
    Paper Section 4: "We trained our networks using stochastic gradient descent
    with a batch size of 128 examples"
    
    Args:
        model: AlexNet model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Computing device (CPU/GPU)
        epoch: Current epoch number
    
    Returns:
        Average training loss for the epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Compute loss
        # Paper Section 4: "The objective function is the multinomial logistic regression
        # objective (i.e., softmax followed by cross-entropy loss)"
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Print progress
        if (batch_idx + 1) % config.LOG_FREQUENCY == 0:
            print(f'Epoch [{epoch}/{config.NUM_EPOCHS}], '
                  f'Batch [{batch_idx + 1}/{len(train_loader)}], '
                  f'Loss: {loss.item():.4f}, '
                  f'Acc: {100.*correct/total:.2f}%')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """
    Validate the model
    
    Paper Section 4: Validation error rate is used to determine when to reduce learning rate
    
    Args:
        model: AlexNet model
        val_loader: Validation data loader
        criterion: Loss function
        device: Computing device (CPU/GPU)
    
    Returns:
        Average validation loss and accuracy
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute loss
            loss = criterion(outputs, targets)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def save_checkpoint(model, optimizer, epoch, loss, acc, filepath):
    """
    Save model checkpoint
    
    Args:
        model: AlexNet model
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss
        acc: Current accuracy
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': acc,
    }
    torch.save(checkpoint, filepath)
    print(f'Checkpoint saved to {filepath}')


def main():
    """
    Main training function
    
    Implements the training procedure from Paper Section 4
    """
    
    # Check if training is enabled
    if not RUN_TRAINING:
        print("=" * 60)
        print("TRAINING IS DISABLED")
        print("=" * 60)
        print("To enable training:")
        print("1. Set RUN_TRAINING = True in train.py")
        print("2. Make sure ImageNet dataset is downloaded")
        print("3. Update DATA_DIR in config.py")
        print("=" * 60)
        return
    
    # Set random seed
    set_seed(config.RANDOM_SEED)
    
    # Create save directory
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    
    # Print configuration
    print("=" * 60)
    print("AlexNet Training Configuration")
    print("=" * 60)
    print(f"Device: {config.DEVICE}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Initial learning rate: {config.INITIAL_LEARNING_RATE}")
    print(f"Momentum: {config.MOMENTUM}")
    print(f"Weight decay: {config.WEIGHT_DECAY}")
    print(f"Number of epochs: {config.NUM_EPOCHS}")
    print(f"Dropout: {config.DROPOUT}")
    print(f"Data directory: {config.DATA_DIR}")
    print("=" * 60)
    
    # Load data
    print("Loading ImageNet dataset...")
    train_loader, val_loader = get_imagenet_dataloaders(
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    # Paper Section 3: Architecture description
    print("Creating AlexNet model...")
    model = AlexNet(num_classes=config.NUM_CLASSES, dropout=config.DROPOUT)
    model = model.to(config.DEVICE)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    # Paper Section 3: "The network has 60 million parameters and 650,000 neurons"
    
    # Loss function
    # Paper Section 4: "The objective function is the multinomial logistic regression
    # objective (i.e., softmax followed by cross-entropy loss)"
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    # Paper Section 4: "We trained our networks using stochastic gradient descent
    # with a batch size of 128 examples"
    # Paper Section 4: "We used a momentum of 0.9"
    # Paper Section 4: "We used a weight decay of 0.0005"
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.INITIAL_LEARNING_RATE,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    # Paper Section 4: "We used an equal learning rate for all layers, which we adjusted
    # manually throughout training. The heuristic which we followed was to divide the
    # learning rate by 10 when the validation error rate stopped improving with the
    # current learning rate."
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.LEARNING_RATE_DECAY_FACTOR,
        patience=config.LEARNING_RATE_DECAY_PATIENCE,
        verbose=True
    )
    
    # Training loop
    # Paper Section 4: "We trained the network for roughly 90 cycles through the training set"
    print("\nStarting training...")
    print("=" * 60)
    
    best_val_acc = 0.0
    start_time = time.time()
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE, epoch
        )
        
        # Validate
        if epoch % config.VAL_FREQUENCY == 0:
            val_loss, val_acc = validate(model, val_loader, criterion, config.DEVICE)
            
            # Update learning rate based on validation loss
            scheduler.step(val_loss)
            
            # Print epoch results
            epoch_time = time.time() - epoch_start
            print(f'\nEpoch [{epoch}/{config.NUM_EPOCHS}] Summary:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'  Time: {epoch_time:.2f}s, LR: {optimizer.param_groups[0]["lr"]:.6f}')
            print("-" * 60)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_checkpoint(
                    model, optimizer, epoch, val_loss, val_acc,
                    os.path.join(config.SAVE_DIR, 'best_model.pth')
                )
        
        # Save checkpoint periodically
        if epoch % config.SAVE_FREQUENCY == 0:
            save_checkpoint(
                model, optimizer, epoch, train_loss, train_acc,
                os.path.join(config.SAVE_DIR, f'checkpoint_epoch_{epoch}.pth')
            )
    
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
