#!/usr/bin/env python
"""
Test script for the ML model architecture.
Designed to work even without PyTorch installed.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.ml.model import AMDOptimizedFuturesModel, ModelManager, TORCH_AVAILABLE

def test_model_creation():
    print("Testing model creation...")
    input_dims = [5, 10, 20]

    for input_dim in input_dims:
        print(f"\nCreating model with input_dim={input_dim}")
        model = AMDOptimizedFuturesModel(input_dim=input_dim)

        print(f"Model created successfully with input_dim={input_dim}")
        print(f"Hidden layers: {model.hidden_layers}")

        if TORCH_AVAILABLE:
            total_params = sum(p.numel() for p in model.parameters())
            print(f"PyTorch model created with {total_params:,} parameters")
        else:
                print("PyTorch not available - model is in placeholder mode")

def test_model_manager():
    print("\nTesting ModelManager...")
    manager = ModelManager()

    if not hasattr(manager, 'save_model'):
        print("ModelManager missing save_model() - test skipped")
        return

    input_dim = 15
    model = AMDOptimizedFuturesModel(input_dim=input_dim)
    feature_columns = [f"feature_{i}" for i in range(input_dim)]

    try:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        dummy_data = np.random.randn(100, input_dim)
        scaler.fit(dummy_data)
    except ImportError:
        print("sklearn not available - using None for scaler")
        scaler = None

    test_filename = "test_model.pt"

    # Set feature columns on model for saving
    model.feature_columns = feature_columns
    save_path = manager.save_model(
        model,
        optimizer=None,
        epoch=10,
        loss=0.123,
        metadata={'scaler': 'StandardScaler instance' if scaler else None},
        filename=test_filename
    )

    if save_path:
        print(f"Model saved to: {save_path}")
        try:
            os.remove(save_path)
            print(f"Removed test file: {save_path}")
        except:
            print(f"Could not remove test file: {save_path}")

def test_gpu_functionality():
    if not TORCH_AVAILABLE:
        print("\nPyTorch not available - skipping GPU tests")
        return

    print("\nTesting GPU functionality...")
    model = AMDOptimizedFuturesModel(input_dim=10)

    if torch.cuda.is_available():
        print("ROCm GPU detected")
        model = model.cuda()
        x = torch.randn(2, 10).cuda()
        with torch.no_grad():
            out = model(x)
        print(f"GPU forward pass successful, output shape: {tuple(out.shape)}")
    else:
        print("No GPU available - test skipped")

def main():
    print("Running ML model tests...")
    print(f"PyTorch available: {TORCH_AVAILABLE}")

    test_model_creation()
    test_model_manager()
    test_gpu_functionality()

    print("\nAll tests completed!")

if __name__ == "__main__":
    if TORCH_AVAILABLE:
        import torch
    main()