#!/usr/bin/env python
"""
Test script for the ML model architecture.
Designed to work even without PyTorch installed.
"""

import os
import sys
import numpy as np

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.ml.model import AMDOptimizedFuturesModel, ModelManager, TORCH_AVAILABLE

def test_model_creation():
    """Test model creation functionality"""
    print("Testing model creation...")
    
    # Test with different input dimensions
    input_dims = [5, 10, 20]
    
    for input_dim in input_dims:
        print(f"\nCreating model with input_dim={input_dim}")
        
        # Create model
        model = AMDOptimizedFuturesModel(input_dim=input_dim)
        
        print(f"Model created successfully with input_dim={input_dim}")
        print(f"Hidden layers: {model.hidden_layers}")
        
        if TORCH_AVAILABLE and model.model is not None:
            # Print model parameters if PyTorch is available
            total_params = sum(p.numel() for p in model.model.parameters())
            print(f"PyTorch model created with {total_params:,} parameters")
        else:
            print("PyTorch not available - model is in placeholder mode")

def test_model_manager():
    """Test model manager functionality if PyTorch is available"""
    print("\nTesting ModelManager...")
    
    # Create model manager
    manager = ModelManager()
    
    # Create a test model
    input_dim = 15
    model = manager.create_model(input_dim=input_dim)
    
    print(f"Model created with input_dim={input_dim}")
    
    if not TORCH_AVAILABLE:
        print("PyTorch not available - skipping model save/load tests")
        return
    
    # Create dummy feature names and scaler
    feature_columns = [f"feature_{i}" for i in range(input_dim)]
    
    # Create a scaler with some dummy data
    try:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        dummy_data = np.random.randn(100, input_dim)
        scaler.fit(dummy_data)
    except ImportError:
        print("sklearn not available - using None for scaler")
        scaler = None
    
    # Add some extra data
    extra_data = {
        'training_accuracy': 0.85,
        'validation_accuracy': 0.78,
        'epochs': 50
    }
    
    # Test saving model
    test_filename = "test_model.pt"
    save_path = manager.save_model(
        model, 
        feature_columns=feature_columns,
        scaler=scaler,
        extra_data=extra_data,
        filename=test_filename
    )
    
    if save_path:
        print(f"Model saved to: {save_path}")
        
        # Test loading model
        loaded_model, loaded_features, loaded_scaler, loaded_extra = manager.load_model(save_path)
        
        # Verify loaded model
        if loaded_model is not None:
            print("\nSuccessfully loaded model:")
            print(f"Loaded feature columns: {loaded_features[:3]}... (total: {len(loaded_features)})")
            
            # Check if scaler was loaded correctly
            if loaded_scaler is not None:
                print("Scaler loaded successfully")
            
            # Check extra data
            if loaded_extra:
                print("\nLoaded extra data:")
                for key, value in loaded_extra.items():
                    print(f"  {key}: {value}")
            
            # Test forward pass on loaded model if PyTorch is available
            if TORCH_AVAILABLE:
                import torch
                x = torch.randn(1, input_dim)
                loaded_model.eval()
                with torch.no_grad():
                    out = loaded_model(x)
                print(f"\nLoaded model forward pass successful, output shape: {tuple(out.shape)}")
        
        # Clean up test file
        try:
            os.remove(save_path)
            print(f"\nRemoved test file: {save_path}")
        except:
            print(f"Could not remove test file: {save_path}")

def test_gpu_functionality():
    """Test GPU functionality if PyTorch and CUDA/ROCm are available"""
    if not TORCH_AVAILABLE:
        print("\nPyTorch not available - skipping GPU tests")
        return
        
    print("\nTesting GPU functionality...")
    
    # Create model
    model = AMDOptimizedFuturesModel(input_dim=10)
    
    # Try to move to GPU
    gpu_available = model.to_gpu()
    
    if gpu_available:
        print("Successfully moved model to GPU")
        
        # Test forward pass on GPU
        import torch
        x = torch.randn(2, 10).cuda()
        model.eval()
        with torch.no_grad():
            out = model(x)
        print(f"GPU forward pass successful, output shape: {tuple(out.shape)}")
    else:
        print("GPU not available - model remains on CPU")

def main():
    """Main test function"""
    print("Running ML model tests...")
    print(f"PyTorch available: {TORCH_AVAILABLE}")
    
    # Test model creation
    test_model_creation()
    
    # Test model manager
    test_model_manager()
    
    # Test GPU functionality
    test_gpu_functionality()
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main()