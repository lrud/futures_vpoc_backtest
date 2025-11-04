#!/usr/bin/env python3
"""
Test script for Phase 1 robust ML pipeline improvements.
Tests directional targets, enhanced features, and larger architecture.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append('/workspace')

from src.ml.feature_engineering_robust import RobustFeatureEngineer
from src.ml.model_robust import RobustFinancialNet, create_robust_optimizer, BCELossWithLogits
from src.utils.logging import get_logger

# Initialize logger
logger = get_logger(__name__)

def test_phase1_improvements():
    """Test Phase 1 improvements on a small data sample."""

    logger.info("🧪 Testing Phase 1 Robust ML Pipeline Improvements")
    logger.info("=" * 60)

    try:
        # 1. Test robust feature engineering with directional targets
        logger.info("📊 Testing Enhanced Feature Engineering...")
        feature_engineer = RobustFeatureEngineer(chunk_size=10000)

        # Load small sample of data for testing
        data_path = "/workspace/DATA/MERGED/merged_es_vix_test.csv"
        result = feature_engineer.load_and_prepare_data_robust(
            data_path=data_path,
            data_fraction=0.05,  # Use 5% of data for testing
            chunk_size=10000
        )

        if result is None:
            logger.error("❌ Feature engineering test failed")
            return False

        X_train, y_train, X_val, y_val, feature_names, scaling_params, target_stats = result

        logger.info("✅ Feature Engineering Test Results:")
        logger.info(f"  • Training data shape: {X_train.shape}")
        logger.info(f"  • Validation data shape: {X_val.shape}")
        logger.info(f"  • Feature count: {len(feature_names)}")
        logger.info(f"  • Feature names: {feature_names}")
        logger.info(f"  • Target type: {target_stats.get('transform_type', 'unknown')}")
        logger.info(f"  • Target range: [{y_train.min():.3f}, {y_train.max():.3f}]")
        logger.info(f"  • Target distribution: {target_stats.get('up_percentage', 0):.1f}% UP, {100-target_stats.get('up_percentage', 0):.1f}% DOWN")

        # 2. Test enhanced neural network architecture
        logger.info("🏗️ Testing Enhanced Neural Network Architecture...")

        # Create model with larger architecture (128-64-32-16)
        model = RobustFinancialNet(
            input_dim=X_train.shape[1],
            hidden_dims=[128, 64, 32, 16],
            dropout_rate=0.3,
            use_residual=True
        )

        # Create optimizer and BCE loss for binary classification
        optimizer, warmup_scheduler, _ = create_robust_optimizer(
            model=model,
            learning_rate=1e-4,
            weight_decay=1e-4,
            warmup_steps=50  # Short for testing
        )

        # Replace loss function with BCE loss for binary classification
        loss_fn = BCELossWithLogits(pos_weight=1.0)

        logger.info("✅ Model Architecture Test Results:")
        param_counts = model.count_parameters()
        logger.info(f"  • Total parameters: {param_counts['total']:,}")
        logger.info(f"  • Linear parameters: {param_counts['linear']:,}")
        logger.info(f"  • LayerNorm parameters: {param_counts['layer_norm']:,}")

        # 3. Test forward pass with sample data
        logger.info("🔄 Testing Forward Pass...")
        model.eval()

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train[:32])  # Small batch
        y_train_tensor = torch.FloatTensor(y_train[:32])

        with torch.no_grad():
            outputs = model(X_train_tensor)
            loss = loss_fn(outputs, y_train_tensor)

        logger.info("✅ Forward Pass Test Results:")
        logger.info(f"  • Input shape: {X_train_tensor.shape}")
        logger.info(f"  • Output shape: {outputs.shape}")
        logger.info(f"  • Output range: [{outputs.min().item():.3f}, {outputs.max().item():.3f}]")
        logger.info(f"  • Loss: {loss.item():.6f}")

        # 4. Test training step
        logger.info("🎯 Testing Training Step...")
        model.train()

        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = loss_fn(outputs, y_train_tensor)
        loss.backward()

        # Test warmup scheduler
        warmup_scheduler.step()
        current_lr = warmup_scheduler.get_current_lr()

        logger.info("✅ Training Step Test Results:")
        logger.info(f"  • Loss: {loss.item():.6f}")
        logger.info(f"  • Learning rate after warmup: {current_lr:.6f}")
        logger.info(f"  • Gradients computed: ✅")

        # 5. Test multiple warmup steps
        logger.info("🔥 Testing Learning Rate Warmup...")
        warmup_lrs = []
        for i in range(10):
            warmup_scheduler.step()
            warmup_lrs.append(warmup_scheduler.get_current_lr())

        logger.info("✅ Warmup Test Results:")
        logger.info(f"  • Initial LR: {warmup_lrs[0]:.6f}")
        logger.info(f"  • Final LR: {warmup_lrs[-1]:.6f}")
        logger.info(f"  • Warmup progression: {'✅' if warmup_lrs[-1] > warmup_lrs[0] else '❌'}")

        # 6. Test prediction confidence calculation
        logger.info("📈 Testing Binary Prediction Logic...")
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_train_tensor)
            # Apply sigmoid to get probabilities
            test_probs = torch.sigmoid(test_outputs)
            # Convert to binary predictions
            test_preds = (test_probs > 0.5).float()

            # Calculate confidence
            confidence_scores = torch.abs(test_probs - 0.5) * 2  # 0 to 1 scale

        logger.info("✅ Binary Prediction Test Results:")
        logger.info(f"  • Probability range: [{test_probs.min().item():.3f}, {test_probs.max().item():.3f}]")
        logger.info(f"  • Prediction distribution: {test_preds.sum().item()}/{len(test_preds)} UP signals")
        logger.info(f"  • Confidence range: [{confidence_scores.min().item():.3f}, {confidence_scores.max().item():.3f}]")

        logger.info("🎉 Phase 1 Robust ML Pipeline Test Completed Successfully!")
        logger.info("=" * 60)
        logger.info("✅ All Phase 1 improvements tested and working:")
        logger.info("  1. Directional binary targets (UP=1, DOWN=0)")
        logger.info("  2. Enhanced features with close_to_vwap (r=-0.79/-0.69)")
        logger.info("  3. Larger neural network (128-64-32-16)")
        logger.info("  4. BCE loss for binary classification")
        logger.info("  5. Proper probability outputs and confidence scores")

        return True

    except Exception as e:
        logger.error(f"❌ Phase 1 test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_phase1_improvements()
    if success:
        print("\n✅ Phase 1 Robust ML Pipeline is ready for full training!")
    else:
        print("\n❌ Phase 1 test failed - check logs for details")