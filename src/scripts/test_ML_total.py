"""
Simple validation script to verify the refactored components work together.
"""
import sys
import os
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.data import FuturesDataManager
from src.ml.feature_engineering import prepare_features_and_labels
from src.ml.model import AMDOptimizedFuturesModel
from src.ml.distributed import AMDFuturesTensorParallel

def main():
    """Validate refactored components work together."""
    print("Loading data...")
    data_manager = FuturesDataManager()
    data = data_manager.load_futures_data()
    
    print("Preparing features...")
    X, y, feature_cols, scaler = prepare_features_and_labels(
        data, use_feature_selection=True, max_features=10
    )
    
    print(f"Created features with shape: {X.shape}")
    print(f"Selected {len(feature_cols)} features")
    
    print("Creating model...")
    model = AMDOptimizedFuturesModel(input_dim=len(feature_cols))
    
    print("Testing forward pass...")
    # Convert a small sample to tensor for forward pass test
    x_sample = torch.tensor(X[:5], dtype=torch.float32)
    with torch.no_grad():
        output = model(x_sample)
    print(f"Model output shape: {output.shape}")
    
    print("Creating trainer...")
    trainer = AMDFuturesTensorParallel()
    
    print("All components work together successfully!")

if __name__ == "__main__":
    main()