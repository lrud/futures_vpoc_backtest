#!/usr/bin/env python3
"""
Enhanced Robust Training Script with Comprehensive Gradient Explosion Mitigation
Addresses all identified issues for stable large model training
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

class EnhancedRobustHuberLoss(nn.Module):
    """
    Enhanced Huber Loss with adaptive delta and numerical stability
    """
    def __init__(self, delta=1.0, reduction='mean', adaptive_delta=True):
        super().__init__()
        self.delta = delta
        self.reduction = reduction
        self.adaptive_delta = adaptive_delta
        self.training_losses = []

    def forward(self, predictions, targets):
        """
        Enhanced Huber loss with numerical stability checks
        """
        # Ensure inputs are clean
        predictions = torch.clamp(predictions, min=-10, max=10)
        targets = torch.clamp(targets, min=0, max=1)

        # For binary classification, apply sigmoid first
        if predictions.shape[1] == 1:  # Single output logit
            predictions = torch.sigmoid(predictions.squeeze())
            targets = targets.float()
        else:  # Already probabilities
            predictions = predictions.squeeze()
            targets = targets.float()

        residual = torch.abs(predictions - targets)

        # Adaptive delta based on residual distribution
        if self.adaptive_delta and self.training:
            residual_median = torch.median(residual).item()
            adaptive_delta = max(self.delta, residual_median * 2.0)
        else:
            adaptive_delta = self.delta

        # Quadratic region (small errors)
        quadratic = torch.where(
            residual <= adaptive_delta,
            0.5 * residual ** 2,
            adaptive_delta * (residual - 0.5 * adaptive_delta)
        )

        # Ensure numerical stability
        quadratic = torch.clamp(quadratic, min=0, max=100)

        if self.reduction == 'mean':
            return quadratic.mean()
        elif self.reduction == 'sum':
            return quadratic.sum()
        else:
            return quadratic

class EnhancedRobustTrainer:
    """Enhanced trainer with comprehensive gradient explosion mitigation"""

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = torch.cuda.amp.GradScaler() if config.get('use_mixed_precision', False) else None

        # Feature engineer
        self.feature_engineer = RobustFeatureEngineer(
            chunk_size=config.get('chunk_size', 25000)
        )

        # Model components
        self.model = None
        self.optimizer = None
        self.loss_fn = None
        self.warmup_scheduler = None

        # Enhanced gradient tracking
        self.gradient_norms = []
        self.loss_history = []
        self.nan_count = 0
        self.total_batches = 0

        # Training state
        self.train_loader = None
        self.val_loader = None
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def load_data(self, data_path: str) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Load and prepare data with enhanced validation"""
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

        # Convert to DataFrames if they aren't already
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train, columns=feature_columns)
        if not isinstance(X_val, pd.DataFrame):
            X_val = pd.DataFrame(X_val, columns=feature_columns)
        if not isinstance(y_train, pd.Series):
            y_train = pd.Series(y_train)
        if not isinstance(y_val, pd.Series):
            y_val = pd.Series(y_val)

        # Apply enhanced robust scaling
        X_train = self._apply_enhanced_robust_scaling(X_train, scaling_params)
        X_val = self._apply_enhanced_robust_scaling(X_val, scaling_params, is_training=False)

        # Enhanced data validation
        self._validate_scaled_data(X_train, "Training")
        self._validate_scaled_data(X_val, "Validation")

        # Create datasets
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train.values),
            torch.FloatTensor(y_train.values)
        )

        val_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_val.values),
            torch.FloatTensor(y_val.values)
        )

        # Create data loaders with enhanced settings
        batch_size = self.config.get('batch_size', 32)

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # ROCm compatibility
            pin_memory=False,
            drop_last=True  # Ensure consistent batch sizes
        )

        self.val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False
        )

        logger.info(f"✅ Created data loaders:")
        logger.info(f"   • Train: {len(self.train_loader)} batches")
        logger.info(f"   • Validation: {len(self.val_loader)} batches")

        return self.train_loader, self.val_loader

    def _apply_enhanced_robust_scaling(self, X: pd.DataFrame, scaling_params: Dict, is_training: bool = True) -> pd.DataFrame:
        """Enhanced robust scaling with comprehensive outlier handling"""
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

                    # Enhanced clipping with tighter bounds for stability
                    X_scaled[col] = X_scaled[col].clip(-3.0, 3.0)

                    if is_training and i % 10 == 0:
                        logger.info(f"   • {col}: scale={scale:.4f}, median={stats.get('median', 0.0):.4f}, using direct scaling with tight clipping [-3, 3]")
                elif 'median' in stats and 'q75' in stats and 'q25' in stats:
                    # Robust scaling with median and IQR
                    median = stats['median']
                    q75, q25 = stats['q75'], stats['q25']
                    iqr = q75 - q25

                    # Add small epsilon to avoid division by zero
                    iqr = max(iqr, 1e-8)

                    # Apply robust scaling
                    X_scaled[col] = (X[col] - median) / iqr

                    # Enhanced clipping for financial data stability
                    lower_bound = -3.0  # 3 IQR below median
                    upper_bound = 3.0   # 3 IQR above median
                    X_scaled[col] = X_scaled[col].clip(lower_bound, upper_bound)

                    if is_training and i % 10 == 0:  # Log first 10 features
                        logger.info(f"   • {col}: median={median:.4f}, IQR={iqr:.4f}, clipped to [{lower_bound}, {upper_bound}]")
                elif 'mean' in stats and 'std' in stats:
                    # Standard scaling (fallback)
                    mean = stats['mean']
                    std = stats['std']
                    std = max(std, 1e-8)  # Avoid division by zero

                    X_scaled[col] = (X[col] - mean) / std

                    # Enhanced clipping
                    X_scaled[col] = X_scaled[col].clip(-3.0, 3.0)

                    if is_training and i % 10 == 0:
                        logger.info(f"   • {col}: mean={mean:.4f}, std={std:.4f}, using standard scaling with tight clipping")
                else:
                    # No scaling applied - apply safe clipping anyway
                    X_scaled[col] = X_scaled[col].clip(-3.0, 3.0)
                    if is_training and i % 10 == 0:
                        logger.info(f"   • {col}: no scaling applied, applied safe clipping [-3, 3]")

        return X_scaled

    def _validate_scaled_data(self, X: pd.DataFrame, dataset_name: str):
        """Validate scaled data for numerical issues"""
        logger.info(f"🔍 Validating {dataset_name} data:")

        # Check for NaN, Inf, and extreme values
        nan_count = X.isnull().sum().sum()
        inf_count = np.isinf(X.values).sum()

        logger.info(f"   • NaN values: {nan_count}")
        logger.info(f"   • Infinite values: {inf_count}")

        # Check value ranges
        min_val = X.min().min()
        max_val = X.max().max()
        mean_val = X.mean().mean()
        std_val = X.std().mean()

        logger.info(f"   • Value range: [{min_val:.4f}, {max_val:.4f}]")
        logger.info(f"   • Mean: {mean_val:.4f}, Std: {std_val:.4f}")

        # Warnings for potential issues
        if nan_count > 0:
            logger.warning(f"⚠️  Found {nan_count} NaN values in {dataset_name} data")
        if inf_count > 0:
            logger.warning(f"⚠️  Found {inf_count} infinite values in {dataset_name} data")
        if abs(max_val) > 10 or abs(min_val) > 10:
            logger.warning(f"⚠️  Extreme values detected in {dataset_name} data: [{min_val:.4f}, {max_val:.4f}]")

    def create_model(self, input_dim: int):
        """Create model with enhanced gradient stability"""
        # Create model
        self.model = RobustFinancialNet(
            input_dim=input_dim,
            hidden_dims=self.config.get('hidden_dims', [64, 32, 16]),
            dropout_rate=self.config.get('dropout_rate', 0.3),
            use_residual=self.config.get('use_residual', True)
        )

        # Move to device
        self.model = self.model.to(self.device)

        # Enhanced model initialization for stability
        self._enhanced_model_init()

        # Enable multi-GPU training
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            logger.info(f"✅ Using {torch.cuda.device_count()} GPUs for parallel training")

        # Create optimizer with enhanced settings
        self.optimizer, self.warmup_scheduler, _ = create_robust_optimizer(
            self.model,
            learning_rate=self.config.get('learning_rate', 0.0001),
            weight_decay=self.config.get('weight_decay', 0.001),
            warmup_steps=self.config.get('warmup_steps', 1000)
        )

        # Enhanced Huber loss
        self.loss_fn = EnhancedRobustHuberLoss(
            delta=1.0,
            reduction='mean',
            adaptive_delta=self.config.get('adaptive_loss', True)
        )

        logger.info(f"✅ Created enhanced model with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        logger.info(f"✅ Using Enhanced Robust Huber Loss with adaptive delta")

    def _enhanced_model_init(self):
        """Enhanced model initialization for training stability"""
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                if len(param.shape) > 1:  # Linear layer weights
                    nn.init.xavier_uniform_(param, gain=0.5)  # Conservative initialization
                else:  # Batch norm weights
                    nn.init.ones_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        logger.info("✅ Applied enhanced model initialization for stability")

    def train_epoch(self, epoch: int) -> float:
        """Enhanced training with comprehensive gradient explosion mitigation"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        valid_batches = 0

        for batch_idx, (data, targets) in enumerate(self.train_loader):
            data, targets = data.to(self.device), targets.to(self.device)

            # Enhanced data validation
            if torch.isnan(data).any() or torch.isinf(data).any():
                logger.warning(f"⚠️  Skipping batch {batch_idx} due to invalid data")
                continue

            if torch.isnan(targets).any() or torch.isinf(targets).any():
                logger.warning(f"⚠️  Skipping batch {batch_idx} due to invalid targets")
                continue

            self.optimizer.zero_grad()

            # Forward pass with enhanced error handling
            try:
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(data)
                        loss = self.loss_fn(outputs, targets.unsqueeze(1))

                        # Enhanced loss validation
                        if torch.isnan(loss) or torch.isinf(loss):
                            self.nan_count += 1
                            continue

                        loss = torch.clamp(loss, min=0, max=10)  # Clip loss for stability

                    # Backward pass with enhanced gradient scaling
                    self.scaler.scale(loss).backward()

                    # Enhanced gradient clipping before optimizer step
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.get('gradient_clip_value', 1.0)
                    )

                    # Track gradient norms
                    self.gradient_norms.append(grad_norm.item())

                    # Check for gradient explosion
                    if grad_norm > 10.0:
                        logger.warning(f"⚠️  Large gradient norm: {grad_norm:.4f} in batch {batch_idx}")
                        # Skip this batch update
                        self.optimizer.zero_grad()
                        continue

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(data)
                    loss = self.loss_fn(outputs, targets.unsqueeze(1))

                    # Enhanced loss validation
                    if torch.isnan(loss) or torch.isinf(loss):
                        self.nan_count += 1
                        continue

                    loss = torch.clamp(loss, min=0, max=10)

                    loss.backward()

                    # Enhanced gradient clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.get('gradient_clip_value', 1.0)
                    )

                    # Track gradient norms
                    self.gradient_norms.append(grad_norm.item())

                    # Check for gradient explosion
                    if grad_norm > 10.0:
                        logger.warning(f"⚠️  Large gradient norm: {grad_norm:.4f} in batch {batch_idx}")
                        self.optimizer.zero_grad()
                        continue

                    self.optimizer.step()

                # Update warmup scheduler
                if self.warmup_scheduler:
                    self.warmup_scheduler.step()

                total_loss += loss.item()
                valid_batches += 1

            except RuntimeError as e:
                logger.error(f"❌ RuntimeError in batch {batch_idx}: {str(e)}")
                continue

            self.total_batches += 1

            # Log progress with enhanced metrics
            if batch_idx % self.config.get('log_interval', 50) == 0 and valid_batches > 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                avg_grad_norm = np.mean(self.gradient_norms[-50:]) if len(self.gradient_norms) >= 50 else np.mean(self.gradient_norms)
                logger.info(f"Batch {batch_idx:5d}/{len(self.train_loader)}: "
                          f"Loss={loss.item():.6f}, LR={current_lr:.6f}, "
                          f"GradNorm={avg_grad_norm:.4f}, ValidBatches={valid_batches}/{batch_idx+1}")

        if valid_batches > 0:
            avg_loss = total_loss / valid_batches
            self.loss_history.append(avg_loss)

            # Log epoch statistics
            nan_rate = self.nan_count / max(self.total_batches, 1) * 100
            avg_grad_norm = np.mean(self.gradient_norms[-100:]) if len(self.gradient_norms) >= 100 else np.mean(self.gradient_norms)

            logger.info(f"📊 Epoch {epoch+1} Statistics:")
            logger.info(f"   • Average Loss: {avg_loss:.6f}")
            logger.info(f"   • Valid Batches: {valid_batches}/{len(self.train_loader)} ({valid_batches/len(self.train_loader)*100:.1f}%)")
            logger.info(f"   • NaN Rate: {nan_rate:.2f}%")
            logger.info(f"   • Average Gradient Norm: {avg_grad_norm:.4f}")

            return avg_loss
        else:
            logger.error(f"❌ No valid batches in epoch {epoch+1}")
            return float('inf')

    def validate(self) -> float:
        """Enhanced validation with comprehensive metrics"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        valid_batches = 0

        # Enhanced metrics
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for data, targets in self.val_loader:
                data, targets = data.to(self.device), targets.to(self.device)

                # Enhanced data validation
                if torch.isnan(data).any() or torch.isinf(data).any():
                    continue
                if torch.isnan(targets).any() or torch.isinf(targets).any():
                    continue

                try:
                    if self.scaler:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(data)
                            loss = self.loss_fn(outputs, targets.unsqueeze(1))
                    else:
                        outputs = self.model(data)
                        loss = self.loss_fn(outputs, targets.unsqueeze(1))

                    # Enhanced loss validation
                    if torch.isnan(loss) or torch.isinf(loss):
                        continue

                    loss = torch.clamp(loss, min=0, max=10)
                    total_loss += loss.item()
                    valid_batches += 1

                    # Collect predictions for metrics
                    predictions = torch.sigmoid(outputs.squeeze())
                    all_predictions.extend(predictions.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())

                except RuntimeError as e:
                    logger.error(f"❌ RuntimeError in validation: {str(e)}")
                    continue

                num_batches += 1

        if valid_batches > 0:
            avg_loss = total_loss / valid_batches

            # Calculate enhanced metrics
            if all_predictions and all_targets:
                predictions_array = np.array(all_predictions)
                targets_array = np.array(all_targets)

                # MAE
                mae = np.mean(np.abs(predictions_array - targets_array))

                # Binary accuracy (threshold 0.5)
                pred_binary = (predictions_array > 0.5).astype(int)
                accuracy = np.mean(pred_binary == targets_array)

                logger.info(f"📊 Enhanced Validation Metrics:")
                logger.info(f"   • Valid Batches: {valid_batches}/{len(self.val_loader)} ({valid_batches/len(self.val_loader)*100:.1f}%)")
                logger.info(f"   • Average Loss: {avg_loss:.6f}")
                logger.info(f"   • MAE: {mae:.6f}")
                logger.info(f"   • Accuracy: {accuracy:.4f}")

            return avg_loss
        else:
            logger.error(f"❌ No valid batches in validation")
            return float('inf')

    def train(self, data_path: str):
        """Enhanced training loop with comprehensive monitoring"""
        # Load data
        train_loader, val_loader = self.load_data(data_path)

        # Create model
        sample_batch = next(iter(train_loader))
        input_dim = sample_batch[0].shape[1]
        self.create_model(input_dim)

        # Enhanced training loop
        epochs = self.config.get('epochs', 50)
        patience = self.config.get('early_stopping_patience', 15)

        logger.info(f"🏃 Starting enhanced training for {epochs} epochs...")

        for epoch in range(epochs):
            logger.info(f"\n🚀 Epoch {epoch+1}/{epochs}")

            # Training
            train_loss = self.train_epoch(epoch)

            # Validation
            val_loss = self.validate()

            current_lr = self.optimizer.param_groups[0]['lr']

            # Enhanced epoch results
            logger.info(f"📈 Epoch {epoch+1:3d}/{epochs:3d} Results:")
            logger.info(f"   • Train Loss: {train_loss:.6f}")
            logger.info(f"   • Val Loss: {val_loss:.6f}")
            logger.info(f"   • Learning Rate: {current_lr:.6f}")
            logger.info(f"   • Total NaN Count: {self.nan_count}")
            logger.info(f"   • Total Batches Processed: {self.total_batches}")

            # Enhanced early stopping check
            if val_loss < self.best_val_loss and not np.isinf(val_loss):
                self.best_val_loss = val_loss
                self.patience_counter = 0
                logger.info(f"✅ New best validation loss: {val_loss:.6f}")
                # Save best model
                self.save_model(epoch, val_loss, 'best')
            else:
                self.patience_counter += 1
                logger.info(f"⏳ Patience counter: {self.patience_counter}/{patience}")

                if self.patience_counter >= patience:
                    logger.info(f"🛑 Enhanced early stopping triggered after {epoch+1} epochs")
                    break

        logger.info("✅ Enhanced training completed!")
        logger.info(f"📊 Final Statistics:")
        logger.info(f"   • Total NaN Rate: {self.nan_count/max(self.total_batches, 1)*100:.2f}%")
        logger.info(f"   • Average Gradient Norm (last 100): {np.mean(self.gradient_norms[-100:]):.4f}")

    def save_model(self, epoch: int, val_loss: float, suffix: str = 'best'):
        """Save enhanced model checkpoint"""
        # Create output directory
        from datetime import datetime
        base_output_dir = self.config.get('output_dir', 'TRAINING')
        date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(base_output_dir, f"enhanced_robust_training_{date_str}")
        os.makedirs(output_dir, exist_ok=True)

        # Save model state with enhanced information
        model_state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config,
            'scaler_state': self.scaler.state_dict() if self.scaler else None,
            'gradient_norms': self.gradient_norms[-1000:],  # Save recent gradient norms
            'loss_history': self.loss_history[-100:],  # Save recent loss history
            'nan_count': self.nan_count,
            'total_batches': self.total_batches
        }

        model_path = os.path.join(output_dir, f'enhanced_robust_model_{suffix}_epoch{epoch+1}.pt')
        torch.save(model_state, model_path)

        # Save enhanced config
        config_path = os.path.join(output_dir, f'enhanced_robust_config_{suffix}.json')
        import json
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

        # Save training statistics
        stats_path = os.path.join(output_dir, f'training_stats_{suffix}.json')
        training_stats = {
            'final_gradient_norm': np.mean(self.gradient_norms[-100:]) if len(self.gradient_norms) >= 100 else np.mean(self.gradient_norms),
            'final_loss': self.loss_history[-1] if self.loss_history else None,
            'nan_rate': self.nan_count / max(self.total_batches, 1) * 100,
            'total_batches': self.total_batches
        }
        with open(stats_path, 'w') as f:
            json.dump(training_stats, f, indent=2)

        logger.info(f"💾 Saved enhanced {suffix} model to {model_path}")

def main():
    """Main enhanced training function"""
    parser = argparse.ArgumentParser(description='Enhanced Robust Financial ML Training')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV data file')
    parser.add_argument('--output_dir', type=str, default='TRAINING', help='Output directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[64, 32, 16], help='Hidden layer dimensions')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='Learning rate warmup steps')
    parser.add_argument('--gradient_clip_value', type=float, default=1.0, help='Gradient clipping value')
    parser.add_argument('--early_stopping_patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--use_mixed_precision', action='store_true', help='Use mixed precision training')
    parser.add_argument('--data_fraction', type=float, default=1.0, help='Fraction of data to use')
    parser.add_argument('--chunk_size', type=int, default=25000, help='Chunk size for processing')
    parser.add_argument('--adaptive_loss', action='store_true', default=True, help='Use adaptive loss delta')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')

    args = parser.parse_args()

    # Setup logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/enhanced_robust_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )

    # Create enhanced config
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
        'adaptive_loss': args.adaptive_loss,
        'log_interval': 50,
        'verbose': args.verbose
    }

    logger.info("🎯 Enhanced Robust Training Configuration:")
    logger.info("=" * 60)
    for key, value in config.items():
        logger.info(f"  • {key}: {value}")
    logger.info("=" * 60)

    # Create enhanced trainer
    trainer = EnhancedRobustTrainer(config)

    # Start enhanced training
    trainer.train(args.data)

    logger.info("🎉 Enhanced robust training completed successfully!")

if __name__ == "__main__":
    main()