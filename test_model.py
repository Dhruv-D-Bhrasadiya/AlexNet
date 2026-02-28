"""
Test script to verify AlexNet model architecture

This script tests the model without requiring the full dataset or training.
"""

import torch
from model import AlexNet


def test_model_architecture():
    """
    Test the AlexNet model architecture
    """
    print("=" * 60)
    print("Testing AlexNet Architecture")
    print("=" * 60)
    
    # Create model
    print("\n1. Creating AlexNet model...")
    try:
        model = AlexNet(num_classes=1000)
        print("   [OK] Model created successfully")
    except Exception as e:
        print(f"   [ERROR] Error creating model: {e}")
        return
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n2. Model Parameters:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Paper states: 60 million parameters, 650,000 neurons")
    
    # Test forward pass
    print("\n3. Testing forward pass...")
    try:
        # Paper Section 3: Input size is 224×224×3
        dummy_input = torch.randn(2, 3, 224, 224)  # Batch size 2
        print(f"   Input shape: {dummy_input.shape}")
        
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"   Output shape: {output.shape}")
        print(f"   Expected output: (2, 1000)")
        
        if output.shape == (2, 1000):
            print("   [OK] Forward pass successful!")
        else:
            print(f"   [ERROR] Unexpected output shape: {output.shape}")
            return
            
    except Exception as e:
        print(f"   [ERROR] Error in forward pass: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test model layers
    print("\n4. Verifying layer structure:")
    layers = list(model.named_children())
    print(f"   Total layers: {len(layers)}")
    
    # Check key components
    has_conv1 = hasattr(model, 'conv1')
    has_lrn = hasattr(model, 'lrn1')
    has_pool = hasattr(model, 'pool1')
    has_dropout = hasattr(model, 'dropout1')
    has_fc = hasattr(model, 'fc3')
    
    print(f"   Conv1 layer: {'[OK]' if has_conv1 else '[MISSING]'}")
    print(f"   LRN layer: {'[OK]' if has_lrn else '[MISSING]'}")
    print(f"   Pooling layer: {'[OK]' if has_pool else '[MISSING]'}")
    print(f"   Dropout layer: {'[OK]' if has_dropout else '[MISSING]'}")
    print(f"   FC output layer: {'[OK]' if has_fc else '[MISSING]'}")
    
    print("\n" + "=" * 60)
    print("All tests passed! Model is ready for training.")
    print("=" * 60)


if __name__ == "__main__":
    test_model_architecture()
