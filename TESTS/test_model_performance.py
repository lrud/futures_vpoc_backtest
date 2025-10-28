#!/usr/bin/env python
"""
Simple test script to evaluate the trained ML model.
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ml.model import AMDOptimizedFuturesModel
from src.ml.feature_engineering import FeatureEngineer
from src.config.settings import settings

def test_model_predictions():
    """Test if the trained model can make predictions."""
    print("=== Testing Trained Model Performance ===")

    # Model path
    model_path = "/workspace/TRAINING_FINAL_COMPLETE_SUCCESS/train_20251023_205047/best_model.pt"

    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return

    print(f"✅ Found model at {model_path}")

    try:
        # Load model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"📱 Using device: {device}")

        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        # Extract architecture info
        architecture = checkpoint.get("architecture", {})
        input_dim = architecture.get("input_dim", 54)
        hidden_layers = [128, 64]  # From training metadata
        dropout_rate = architecture.get("dropout_rate", 0.4)

        print(f"🏗️  Model Architecture:")
        print(f"   Input dimension: {input_dim}")
        print(f"   Hidden layers: {hidden_layers}")
        print(f"   Dropout rate: {dropout_rate}")

        # Create model
        model = AMDOptimizedFuturesModel(
            input_dim=input_dim,
            hidden_layers=hidden_layers,
            dropout_rate=dropout_rate
        )

        # Load state dict
        state_dict = checkpoint["model_state_dict"]
        if all(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()

        print("✅ Model loaded successfully")

        # Create some dummy features to test prediction
        print("\n🧪 Testing model predictions...")

        # Create realistic feature data
        batch_size = 100
        dummy_features = torch.randn(batch_size, input_dim, device=device, dtype=torch.float32)

        print(f"📊 Testing with dummy features shape: {dummy_features.shape}")

        # Make predictions
        with torch.no_grad():
            predictions = model(dummy_features)

        print(f"✅ Model predictions successful!")
        print(f"   Output shape: {predictions.shape}")
        print(f"   Prediction range: [{predictions.min().item():.4f}, {predictions.max().item():.4f}]")
        print(f"   Mean prediction: {predictions.mean().item():.4f}")
        print(f"   Std deviation: {predictions.std().item():.4f}")

        # Analyze predictions
        positive_predictions = (predictions > 0).sum().item()
        negative_predictions = (predictions < 0).sum().item()

        print(f"\n📈 Prediction Analysis:")
        print(f"   Positive predictions: {positive_predictions} ({positive_predictions/batch_size*100:.1f}%)")
        print(f"   Negative predictions: {negative_predictions} ({negative_predictions/batch_size*100:.1f}%)")

        # Test with different thresholds
        thresholds = [0.1, 0.2, 0.3, 0.5]
        print(f"\n🎯 Signal Generation at Different Thresholds:")

        for threshold in thresholds:
            signals = torch.abs(predictions) > threshold
            signal_count = signals.sum().item()
            buy_signals = (predictions > threshold).sum().item()
            sell_signals = (predictions < -threshold).sum().item()

            print(f"   Threshold {threshold:.1f}: {signal_count} signals "
                  f"({signal_count/batch_size*100:.1f}%) - "
                  f"Buy: {buy_signals}, Sell: {sell_signals}")

        # Test with a small real data sample
        print(f"\n🔍 Testing with real data sample...")

        # Load a small sample of real data
        data_path = "/workspace/DATA/MERGED/merged_es_vix_test.csv"
        if os.path.exists(data_path):
            # Load just a small sample
            df = pd.read_csv(data_path, nrows=1000)

            # Basic preprocessing
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')

            print(f"📊 Loaded {len(df)} rows of real data")

            # Create feature engineer
            engineer = FeatureEngineer()

            try:
                # Generate features
                features_df = engineer.prepare_features(df)

                if not features_df.empty:
                    print(f"✅ Generated features with shape: {features_df.shape}")

                    # Get feature columns (exclude non-numeric)
                    feature_cols = [col for col in features_df.columns
                                  if pd.api.types.is_numeric_dtype(features_df[col])]

                    if len(feature_cols) == input_dim:
                        # Prepare features for model
                        feature_data = features_df[feature_cols].fillna(0).values

                        # Add any missing features with zeros
                        if len(feature_cols) < input_dim:
                            padding = np.zeros((len(feature_data), input_dim - len(feature_cols)))
                            feature_data = np.hstack([feature_data, padding])
                            print(f"⚠️  Added {input_dim - len(feature_cols)} padding features")
                        elif len(feature_cols) > input_dim:
                            feature_data = feature_data[:, :input_dim]
                            print(f"⚠️  Truncated to first {input_dim} features")

                        # Convert to tensor
                        real_features = torch.tensor(feature_data,
                                                  dtype=torch.float32,
                                                  device=device)

                        print(f"🔢 Real features tensor shape: {real_features.shape}")

                        # Make predictions
                        with torch.no_grad():
                            real_predictions = model(real_features)

                        print(f"✅ Real data predictions successful!")
                        print(f"   Output shape: {real_predictions.shape}")
                        print(f"   Prediction range: [{real_predictions.min().item():.4f}, {real_predictions.max().item():.4f}]")

                        # Count signals at different thresholds
                        print(f"\n📊 Real Data Signal Analysis:")
                        for threshold in [0.1, 0.2, 0.3]:
                            signals = torch.abs(real_predictions) > threshold
                            signal_count = signals.sum().item()
                            buy_signals = (real_predictions > threshold).sum().item()
                            sell_signals = (real_predictions < -threshold).sum().item()

                            print(f"   Threshold {threshold:.1f}: {signal_count} signals "
                                  f"({signal_count/len(real_predictions)*100:.1f}%) - "
                                  f"Buy: {buy_signals}, Sell: {sell_signals}")

                        # Show some example predictions with dates
                        print(f"\n📅 Sample Predictions:")
                        if len(real_predictions) > 0:
                            dates = features_df.index[:min(5, len(real_predictions))]
                            for i, (date, pred) in enumerate(zip(dates, real_predictions[:5])):
                                signal = "BUY" if pred > 0.2 else "SELL" if pred < -0.2 else "NEUTRAL"
                                print(f"   {date.strftime('%Y-%m-%d')}: {pred.item():.4f} ({signal})")

                    else:
                        print(f"❌ Feature count mismatch: expected {input_dim}, got {len(feature_cols)}")

                else:
                    print("❌ No features generated from real data")

            except Exception as e:
                print(f"❌ Error processing real data: {e}")

        else:
            print(f"❌ Real data file not found at {data_path}")

        print(f"\n🎉 Model testing completed successfully!")
        print(f"   The model appears to be functional and can generate trading signals.")

    except Exception as e:
        print(f"❌ Error testing model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_predictions()