#!/usr/bin/env python3
"""
Robust Training Script with Huber Loss and Proper Normalization
Restores fat-tailed data handling for financial time series
"""

import os
import sys
import argparse
import logging
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append('/workspace')

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from src.ml.feature_engineering_robust import RobustFeatureEngineer
from src.ml.model_robust import RobustFinancialNet, create_robust_optimizer
from src.utils.logging import get_logger

logger = get_logger(__name__)

class RobustHuberLoss(nn.Module):
    """
    Huber Loss for fat-tailed financial data
    Combines MSE for small errors with MAE for large errors
    Prevents gradient explosion from extreme outliers
    """
    def __init__(self, delta=1.0, reduction='mean'):
        super().__init__()
        self.delta = delta
        self.reduction = reduction

    def forward(self, predictions, targets):
        """
        Huber loss calculation
        Args:
            predictions: model predictions (logits)
            targets: target values (0 or 1 for binary classification)
        """
        # For binary classification, apply sigmoid first
        if predictions.shape[1] == 1:  # Single output logit
            predictions = torch.sigmoid(predictions.squeeze())
            targets = targets.float()
        else:  # Already probabilities
            predictions = predictions.squeeze()
            targets = targets.float()

        residual = torch.abs(predictions - targets)

        # Quadratic region (small errors)
        quadratic = torch.where(
            residual <= self.delta,
            0.5 * residual ** 2,
            self.delta * (residual - 0.5 * self.delta)
        )

        if self.reduction == 'mean':
            return quadratic.mean()
        elif self.reduction == 'sum':
            return quadratic.sum()
        else:
            return quadratic

class RobustTrainerFixed:
    """Fixed trainer with Huber loss and robust scaling"""

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = torch.cuda.amp.GradScaler() if config.get('use_mixed_precision', False) else None

        # Feature engineer
        self.feature_engineer = RobustFeatureEngineer(
            chunk_size=config.get('chunk_size', 25000)
        )

        # Model
        self.model = None
        self.optimizer = None
        self.loss_fn = None
        self.warmup_scheduler = None

        # Training state
        self.train_loader = None
        self.val_loader = None
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def load_data(self, data_path: str) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Load and prepare data with robust scaling"""
        logger.info(f"📊 Loading data from: {data_path}")

        # Load and prepare data
        result = self.feature_engineer.load_and_prepare_data_robust(
            data_path,
            data_fraction=self.config.get('data_fraction', 1.0),
            chunk_size=self.config.get('chunk_size', 25000)
        )

        if result is None:
            raise ValueError("Failed to load data")

        X_train, y_train, X_val, y_val, feature_columns, scaling_params, target_stats = result

        logger.info(f"✅ Data loaded successfully:")
        logger.info(f"   • Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        logger.info(f"   • Validation: {X_val.shape[0]} samples")
        logger.info(f"   • Feature columns: {len(feature_columns)} features")
        logger.info(f"   • Scaling params applied: {len(scaling_params)} features scaled")

        # Convert to DataFrames if they aren't already
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train, columns=feature_columns)
        if not isinstance(X_val, pd.DataFrame):
            X_val = pd.DataFrame(X_val, columns=feature_columns)
        if not isinstance(y_train, pd.Series):
            y_train = pd.Series(y_train)
        if not isinstance(y_val, pd.Series):
            y_val = pd.Series(y_val)

        # Apply additional robust scaling to handle extreme outliers
        X_train = self._apply_robust_scaling(X_train, scaling_params)
        X_val = self._apply_robust_scaling(X_val, scaling_params, is_training=False)

        # Create datasets
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train.values),
            torch.FloatTensor(y_train.values)
        )

        val_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_val.values),
            torch.FloatTensor(y_val.values)
        )

        # Create data loaders with ROCm optimization
        batch_size = self.config.get('batch_size', 32)

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # ROCm compatibility
            pin_memory=False,
            drop_last=True
        )

        self.val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # ROCm compatibility
            pin_memory=False
        )

        logger.info(f"✅ Created data loaders:")
        logger.info(f"   • Train: {len(self.train_loader)} batches")
        logger.info(f"   • Validation: {len(self.val_loader)} batches")

        return self.train_loader, self.val_loader

    def _apply_robust_scaling(self, X: pd.DataFrame, scaling_params: Dict, is_training: bool = True) -> pd.DataFrame:
        """Apply robust scaling to handle extreme outliers"""
        X_scaled = X.copy()

        for i, col in enumerate(X.columns):
            if col in scaling_params:
                stats = scaling_params[col]

                # Debug: print available keys for first feature
                if is_training and i == 0:
                    logger.info(f"   Available scaling keys for {col}: {list(stats.keys())}")

                # Handle different scaling parameter formats
                # Check for direct scale parameter first (most common case)
                if 'scale' in stats:
                    # Use scale parameter directly (common case)
                    scale = stats['scale']
                    scale = max(scale, 1e-8)  # Avoid division by zero

                    # Apply scaling
                    X_scaled[col] = (X[col] - stats.get('median', 0.0)) / scale

                    # Apply clipping
                    X_scaled[col] = X_scaled[col].clip(-5.0, 5.0)

                    if is_training and i % 10 == 0:
                        logger.info(f"   • {col}: scale={scale:.4f}, median={stats.get('median', 0.0):.4f}, using direct scaling with clipping")
                elif 'median' in stats and 'q75' in stats and 'q25' in stats:
                    # Robust scaling with median and IQR
                    median = stats['median']
                    q75, q25 = stats['q75'], stats['q25']
                    iqr = q75 - q25

                    # Add small epsilon to avoid division by zero
                    iqr = max(iqr, 1e-8)

                    # Apply robust scaling
                    X_scaled[col] = (X[col] - median) / iqr

                    # Apply clipping to handle extreme outliers (winsorization)
                    lower_bound = -5.0  # 5 IQR below median
                    upper_bound = 5.0   # 5 IQR above median
                    X_scaled[col] = X_scaled[col].clip(lower_bound, upper_bound)

                    if is_training and i % 10 == 0:  # Log first 10 features
                        logger.info(f"   • {col}: median={median:.4f}, IQR={iqr:.4f}, clipped to [{lower_bound}, {upper_bound}]")
                elif 'mean' in stats and 'std' in stats:
                    # Standard scaling (fallback)
                    mean = stats['mean']
                    std = stats['std']
                    std = max(std, 1e-8)  # Avoid division by zero

                    X_scaled[col] = (X[col] - mean) / std

                    # Apply clipping
                    X_scaled[col] = X_scaled[col].clip(-5.0, 5.0)

                    if is_training and i % 10 == 0:
                        logger.info(f"   • {col}: mean={mean:.4f}, std={std:.4f}, using standard scaling with clipping")
                else:
                    # No scaling applied
                    if is_training and i % 10 == 0:
                        logger.info(f"   • {col}: no scaling applied, available keys: {list(stats.keys())}")

        return X_scaled

    def create_model(self, input_dim: int):
        """Create model with Huber loss"""
        # Create model
        self.model = RobustFinancialNet(
            input_dim=input_dim,
            hidden_dims=self.config.get('hidden_dims', [64, 32, 16]),
            dropout_rate=self.config.get('dropout_rate', 0.3),
            use_residual=self.config.get('use_residual', True)
        )

        # Move to device
        self.model = self.model.to(self.device)

        # Enable multi-GPU training
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            logger.info(f"✅ Using {torch.cuda.device_count()} GPUs for parallel training")

        # Create optimizer and warmup scheduler
        self.optimizer, self.warmup_scheduler, _ = create_robust_optimizer(
            self.model,
            learning_rate=self.config.get('learning_rate', 0.00001),
            weight_decay=self.config.get('weight_decay', 0.001),
            warmup_steps=self.config.get('warmup_steps', 1000)
        )

        # Create Huber loss for fat-tailed financial data
        self.loss_fn = RobustHuberLoss(delta=1.0, reduction='mean')

        logger.info(f"✅ Created model with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        logger.info(f"✅ Using Robust Huber Loss for fat-tailed financial data")

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (data, targets) in enumerate(self.train_loader):
            data, targets = data.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(data)
                    loss = self.loss_fn(outputs, targets.unsqueeze(1))

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.config.get('gradient_clip_value', 0.1) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.get('gradient_clip_value', 0.1)
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(data)
                loss = self.loss_fn(outputs, targets.unsqueeze(1))

                loss.backward()

                # Gradient clipping
                if self.config.get('gradient_clip_value', 0.1) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.get('gradient_clip_value', 0.1)
                    )

                self.optimizer.step()

            # Update warmup scheduler
            if self.warmup_scheduler:
                self.warmup_scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            # Log progress
            if batch_idx % self.config.get('log_interval', 50) == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                logger.info(f"Batch {batch_idx:5d}/{len(self.train_loader)}: "
                          f"Loss={loss.item():.6f}, LR={current_lr:.6f}")

        return total_loss / num_batches

    def validate(self) -> float:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for data, targets in self.val_loader:
                data, targets = data.to(self.device), targets.to(self.device)

                if self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(data)
                        loss = self.loss_fn(outputs, targets.unsqueeze(1))
                else:
                    outputs = self.model(data)
                    loss = self.loss_fn(outputs, targets.unsqueeze(1))

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def train(self, data_path: str):
        """Main training loop"""
        # Load data
        train_loader, val_loader = self.load_data(data_path)

        # Create model
        sample_batch = next(iter(train_loader))
        input_dim = sample_batch[0].shape[1]
        self.create_model(input_dim)

        # Training loop
        epochs = self.config.get('epochs', 50)
        patience = self.config.get('early_stopping_patience', 15)

        logger.info(f"🏃 Starting training for {epochs} epochs...")

        for epoch in range(epochs):
            # Training
            train_loss = self.train_epoch(epoch)

            # Validation
            val_loss = self.validate()

            # Calculate MAE for validation
            self.model.eval()
            with torch.no_grad():
                val_mae = 0.0
                val_count = 0
                for data, targets in val_loader:
                    data, targets = data.to(self.device), targets.to(self.device)
                    outputs = self.model(data)
                    predictions = torch.sigmoid(outputs.squeeze())
                    val_mae += torch.mean(torch.abs(predictions - targets)).item()
                    val_count += 1
                val_mae /= val_count

            current_lr = self.optimizer.param_groups[0]['lr']

            # Log epoch results
            logger.info(f"Epoch {epoch+1:3d}/{epochs:3d}: "
                      f"Train Loss={train_loss:.6f}, "
                      f"Val Loss={val_loss:.6f}, "
                      f"Val MAE={val_mae:.6f}, "
                      f"LR={current_lr:.6f}")

            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0

                # Save best model
                self.save_model(epoch, val_loss, 'best')
            else:
                self.patience_counter += 1

                if self.patience_counter >= patience:
                    logger.info(f"🛑 Early stopping triggered after {epoch+1} epochs")
                    break

        logger.info("✅ Training completed!")

    def save_model(self, epoch: int, val_loss: float, suffix: str = 'best'):
        """Save model checkpoint"""
        # Create output directory
        from datetime import datetime
        base_output_dir = self.config.get('output_dir', 'TRAINING')
        date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(base_output_dir, f"robust_training_{date_str}")
        os.makedirs(output_dir, exist_ok=True)

        # Save model state
        model_state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config,
            'scaler_state': self.scaler.state_dict() if self.scaler else None
        }

        model_path = os.path.join(output_dir, f'robust_model_{suffix}_epoch{epoch+1}.pt')
        torch.save(model_state, model_path)

        # Save config
        config_path = os.path.join(output_dir, f'robust_config_{suffix}.json')
        import json
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

        logger.info(f"💾 Saved {suffix} model to {model_path}")

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Robust Financial ML Training with Huber Loss')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV data file')
    parser.add_argument('--output_dir', type=str, default='TRAINING', help='Output directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.00001, help='Learning rate')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[64, 32, 16], help='Hidden layer dimensions')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='Learning rate warmup steps')
    parser.add_argument('--gradient_clip_value', type=float, default=0.1, help='Gradient clipping value')
    parser.add_argument('--early_stopping_patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--use_mixed_precision', action='store_true', help='Use mixed precision training')
    parser.add_argument('--data_fraction', type=float, default=1.0, help='Fraction of data to use')
    parser.add_argument('--chunk_size', type=int, default=25000, help='Chunk size for processing')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')

    args = parser.parse_args()

    # Setup logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/robust_training_fixed_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )

    # Create config
    config = {
        'output_dir': args.output_dir,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'hidden_dims': args.hidden_dims,
        'dropout_rate': args.dropout_rate,
        'weight_decay': args.weight_decay,
        'warmup_steps': args.warmup_steps,
        'gradient_clip_value': args.gradient_clip_value,
        'early_stopping_patience': args.early_stopping_patience,
        'use_mixed_precision': args.use_mixed_precision,
        'data_fraction': args.data_fraction,
        'chunk_size': args.chunk_size,
        'use_residual': True,
        'log_interval': 50,
        'verbose': args.verbose
    }

    logger.info("🎯 Robust Training Configuration (Fixed with Huber Loss):")
    logger.info("=" * 50)
    for key, value in config.items():
        logger.info(f"  • {key}: {value}")
    logger.info("=" * 50)

    # Create trainer
    trainer = RobustTrainerFixed(config)

    # Start training
    trainer.train(args.data)

    logger.info("🎉 Robust training completed successfully!")

if __name__ == "__main__":
    main()