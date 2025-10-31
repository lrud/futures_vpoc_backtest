#!/usr/bin/env python3
"""
Robust Neural Network Training for Financial Time Series
Based on research-backed solutions for stable training

This script implements the complete robust training pipeline with:
1. Huber Loss (robust to outliers)
2. Rank-based target transformation (eliminates extreme outliers)
3. Learning Rate Warmup (prevents early gradient explosion)
4. Layer Normalization (stabilizes hidden activations)
5. Residual Connections (prevents vanishing/exploding gradients)
6. ROCm 7 consumer GPU optimization
7. Top 5 most important features from statistical analysis

Usage:
    python src/ml/train_robust.py --data DATA/MERGED/merged_es_vix_test.csv --epochs 50 --batch_size 32
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append('/workspace')

from src.ml.feature_engineering_robust import RobustFeatureEngineer
from src.ml.model_robust import create_robust_model, create_robust_optimizer, HuberLoss
from src.utils.logging import get_logger
from src.config.settings import settings

# Initialize logger
logger = get_logger(__name__)

class RobustTrainer:
    """
    Robust trainer implementing all research-backed solutions for stable financial ML training.
    """

    def __init__(self, config: Dict):
        """
        Initialize robust trainer.

        Args:
            config: Configuration dictionary with training parameters
        """
        self.config = config
        self.setup_device()
        self.setup_logging()

        # Training components
        self.model = None
        self.optimizer = None
        self.loss_fn = None
        self.warmup_scheduler = None
        self.scaler = None

        # Data components
        self.feature_engineer = RobustFeatureEngineer(
            device_ids=getattr(self, 'device_ids', [0]) if torch.cuda.is_available() else None,
            chunk_size=config.get('chunk_size', 15000)
        )

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []

        logger.info("🚀 Robust Trainer initialized")

    def setup_device(self):
        """Setup device with ROCm optimization and dual GPU support."""
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()

            if gpu_count >= 2:
                # Use both GPUs for distributed training
                self.device = torch.device('cuda:0')
                self.use_cuda = True
                self.device_ids = [0, 1]  # Use both RX 7900 XT GPUs
                torch.cuda.set_device(0)

                logger.info(f"🚀 Using {gpu_count} GPUs for distributed training")
                logger.info(f"  • GPU 0: {torch.cuda.get_device_name(0)}")
                logger.info(f"  • GPU 1: {torch.cuda.get_device_name(1)}")
                logger.info(f"  • GPU 0 Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
                logger.info(f"  • GPU 1 Memory: {torch.cuda.get_device_properties(1).total_memory / 1024**3:.1f}GB")
            else:
                # Single GPU fallback
                self.device = torch.device('cuda:0')
                self.use_cuda = True
                self.device_ids = [0]
                torch.cuda.set_device(0)

                logger.info(f"✅ Using single GPU: {torch.cuda.get_device_name(0)}")
                logger.info(f"  • GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

            # ROCm 7 consumer GPU optimization
            os.environ['HIP_VISIBLE_DEVICES'] = '0,1'
            os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'
            os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'

            # Enable mixed precision
            if self.config.get('use_mixed_precision', True):
                self.use_amp = True
                logger.info("✅ ROCm 7 with mixed precision enabled")
            else:
                self.use_amp = False
                logger.info("✅ ROCm 7 without mixed precision")

            # Memory optimization
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.9)

        else:
            self.device = torch.device('cpu')
            self.use_cuda = False
            self.use_amp = False
            self.device_ids = []
            logger.error("❌ No CUDA GPUs available - this pipeline requires GPU support!")
            raise RuntimeError("Robust pipeline requires GPU support for financial ML training")

    def setup_logging(self):
        """Setup logging configuration."""
        import logging
        import os
        from datetime import datetime

        # Create logs directory
        os.makedirs('logs', exist_ok=True)

        # Setup file logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/robust_training_{timestamp}.log"

        # Configure file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)

        # Add file handler to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)

        logger.info(f"📝 Logging to file: {log_file}")

    def load_data(self, data_path: str) -> Optional[Tuple]:
        """
        Load and prepare data using robust feature engineering.

        Args:
            data_path: Path to data file

        Returns:
            Tuple of (X_train, y_train, X_val, y_val, feature_names) or None if failed
        """
        logger.info(f"📊 Loading data from: {data_path}")

        result = self.feature_engineer.load_and_prepare_data_robust(
            data_path=data_path,
            device_ids=getattr(self, 'device_ids', [0]) if torch.cuda.is_available() else None,
            data_fraction=self.config.get('data_fraction', 1.0),
            chunk_size=self.config.get('chunk_size', 15000)
        )

        if result is None:
            logger.error("❌ Failed to load data")
            return None

        X_train, y_train, X_val, y_val, feature_names, scaling_params, target_stats = result

        # Convert to tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)

        # Store for inverse transformation
        self.scaling_params = scaling_params
        self.target_stats = target_stats

        logger.info(f"✅ Data loaded successfully:")
        logger.info(f"  • Train: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
        logger.info(f"  • Validation: {X_val.shape[0]:,} samples")
        logger.info(f"  • Features: {', '.join(feature_names)}")
        logger.info(f"  • Target range: [{y_train.min().item():.3f}, {y_train.max().item():.3f}]")

        return X_train, y_train, X_val, y_val, feature_names

    def create_model(self, input_dim: int):
        """Create robust model with research-backed architecture and dual GPU support."""
        logger.info("🏗️ Creating robust model...")

        self.model = create_robust_model(
            input_dim=input_dim,
            hidden_dims=self.config.get('hidden_dims', [64, 32]),
            dropout_rate=self.config.get('dropout_rate', 0.1),
            use_residual=self.config.get('use_residual', True)
        )

        self.model.to(self.device)

        # Temporarily disable DataParallel to debug matrix dimension issue
        if self.use_cuda:
            logger.info(f"✅ Using single GPU: {self.device} (DataParallel temporarily disabled for debugging)")
        else:
            logger.error("❌ GPU-only training required for robust pipeline")
            raise RuntimeError("Robust pipeline requires GPU support")

        # Debug: Log actual model architecture
        logger.info(f"🔍 Model architecture debug:")
        logger.info(f"  • Input dim expected: {input_dim}")
        logger.info(f"  • First layer should be: Linear({input_dim}, {self.config.get('hidden_dims', [64, 32])[0]})")

        # Create optimizer, warmup scheduler, and loss function
        self.optimizer, self.warmup_scheduler, self.loss_fn = create_robust_optimizer(
            model=self.model,
            learning_rate=self.config.get('learning_rate', 1e-4),
            weight_decay=self.config.get('weight_decay', 1e-4),
            warmup_steps=self.config.get('warmup_steps', 1000)
        )

        # Create gradient scaler for mixed precision
        if self.use_amp:
            self.scaler = GradScaler()
            logger.info("✅ Mixed precision training enabled")
        else:
            self.scaler = None
            logger.info("✅ Full precision training")

        # Model parameter count (handle DataParallel wrapper)
        if isinstance(self.model, nn.DataParallel):
            param_counts = self.model.module.count_parameters()
        else:
            param_counts = self.model.count_parameters()
        logger.info(f"✅ Model created with {param_counts['total']:,} parameters")

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch with robust techniques."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    output = self.model(data)
                    loss = self.loss_fn(output, target)

                # Backward pass with gradient scaling
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()

                # Gradient clipping for stability
                if self.config.get('gradient_clip_value', 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.get('gradient_clip_value', 1.0)
                    )

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Full precision training
                output = self.model(data)
                loss = self.loss_fn(output, target)

                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                if self.config.get('gradient_clip_value', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.get('gradient_clip_value', 1.0)
                    )

                self.optimizer.step()

            # Learning rate warmup
            self.warmup_scheduler.step()

            total_loss += loss.item()

            # Logging
            if batch_idx % self.config.get('log_interval', 50) == 0:
                current_lr = self.warmup_scheduler.get_current_lr()
                logger.info(f"Batch {batch_idx:4d}/{num_batches:4d}: "
                           f"Loss={loss.item():.6f}, LR={current_lr:.6f}")

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate model and return loss and MAE."""
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        num_samples = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)

                # Forward pass
                if self.use_amp:
                    with autocast():
                        output = self.model(data)
                        loss = self.loss_fn(output, target)
                else:
                    output = self.model(data)
                    loss = self.loss_fn(output, target)

                total_loss += loss.item() * data.size(0)
                total_mae += torch.mean(torch.abs(output - target)).item() * data.size(0)
                num_samples += data.size(0)

        avg_loss = total_loss / num_samples
        avg_mae = total_mae / num_samples

        return avg_loss, avg_mae

    def train(self, data_path: str):
        """Main training loop with robust techniques."""
        logger.info("🚀 Starting robust training pipeline...")
        logger.info(f"Configuration: {self.config}")

        # Load data
        data_result = self.load_data(data_path)
        if data_result is None:
            logger.error("❌ Training failed: Could not load data")
            return False

        X_train, y_train, X_val, y_val, feature_names = data_result

        # Create model
        self.create_model(input_dim=X_train.shape[1])

        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)

        # For ROCm/CUDA, use num_workers=0 to avoid multiprocessing issues
        if self.use_cuda:
            num_workers_train = 0
            num_workers_val = 0
            pin_memory = False  # Cannot pin CUDA tensors, only CPU tensors
            logger.info("🔧 Using CUDA-compatible DataLoader settings (num_workers=0, pin_memory=False)")
        else:
            num_workers_train = min(4, os.cpu_count())
            num_workers_val = min(2, os.cpu_count())
            pin_memory = True  # Pin memory for CPU tensors to speed up GPU transfer
            logger.info("🔧 Using CPU DataLoader settings")

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=True,
            num_workers=num_workers_train,
            pin_memory=pin_memory,
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=False,
            num_workers=num_workers_val,
            pin_memory=pin_memory
        )

        logger.info(f"✅ Created data loaders:")
        logger.info(f"  • Train: {len(train_loader)} batches")
        logger.info(f"  • Validation: {len(val_loader)} batches")
        logger.info(f"  • CUDA workers: {num_workers_train}")
        logger.info(f"  • Pin memory: {pin_memory}")

        # Training loop
        num_epochs = self.config.get('epochs', 50)
        patience = self.config.get('early_stopping_patience', 15)
        patience_counter = 0

        logger.info(f"🏃 Starting training for {num_epochs} epochs...")
        start_time = time.time()

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            # Training
            train_loss = self.train_epoch(train_loader)

            # Validation
            val_loss, val_mae = self.validate(val_loader)

            # Store losses
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            total_time = time.time() - start_time

            # Logging
            logger.info(f"Epoch {epoch+1:3d}/{num_epochs:3d} ({epoch_time:.1f}s, {total_time/60:.1f}m total):")
            logger.info(f"  • Train Loss: {train_loss:.6f}")
            logger.info(f"  • Val Loss:   {val_loss:.6f}")
            logger.info(f"  • Val MAE:    {val_mae:.6f}")
            logger.info(f"  • LR:        {self.warmup_scheduler.get_current_lr():.6f}")

            # Learning rate scheduling (after warmup)
            if epoch >= self.config.get('warmup_steps', 1000) // len(train_loader):
                # Simple cosine decay after warmup
                if hasattr(self, 'lr_scheduler'):
                    self.lr_scheduler.step()

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0

                # Save best model
                self.save_model('best')
                logger.info(f"  ✅ New best model saved (Val Loss: {val_loss:.6f})")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"🛑 Early stopping triggered (patience: {patience})")
                break

        # Final evaluation
        logger.info("🎯 Final model evaluation...")
        final_val_loss, final_val_mae = self.validate(val_loader)
        logger.info(f"  • Final Val Loss: {final_val_loss:.6f}")
        logger.info(f"  • Final Val MAE:  {final_val_mae:.6f}")
        logger.info(f"  • Best Val Loss: {self.best_val_loss:.6f}")

        logger.info("✅ Robust training completed successfully!")
        return True

    def save_model(self, suffix: str = 'final'):
        """Save model with robust configuration."""
        import os
        import json
        from datetime import datetime

        # Create output directory
        output_dir = self.config.get('output_dir', 'TRAINING_ROBUST')
        os.makedirs(output_dir, exist_ok=True)

        # Save model state
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(output_dir, f"robust_model_{suffix}_{timestamp}.pt")

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'feature_names': self.config.get('feature_names', []),
            'target_stats': self.target_stats,
            'scaling_params': self.scaling_params,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }, model_path)

        # Save configuration
        config_path = os.path.join(output_dir, f"robust_config_{suffix}_{timestamp}.json")
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

        logger.info(f"💾 Model saved: {model_path}")
        logger.info(f"📝 Config saved: {config_path}")

def get_base_parser(description=None):
    """Get argument parser with common robust training arguments."""
    parser = argparse.ArgumentParser(
        description=description or "Robust Neural Network Training for Financial Time Series",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data arguments
    parser.add_argument('--data', type=str, default='DATA/MERGED/merged_es_vix_test.csv',
                       help='Path to training data file')
    parser.add_argument('--output_dir', type=str, default='TRAINING_ROBUST',
                       help='Output directory for models and logs')
    parser.add_argument('--data_fraction', type=float, default=1.0,
                       help='Fraction of data to use (1.0 = full dataset)')

    # Model arguments
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[64, 32],
                       help='Hidden layer dimensions')
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                       help='Dropout rate for regularization')
    parser.add_argument('--use_residual', action='store_true', default=True,
                       help='Use residual connections')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay for regularization')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                       help='Number of learning rate warmup steps')
    parser.add_argument('--gradient_clip_value', type=float, default=1.0,
                       help='Gradient clipping value (0 to disable)')
    parser.add_argument('--early_stopping_patience', type=int, default=15,
                       help='Early stopping patience')

    # Optimization arguments
    parser.add_argument('--use_mixed_precision', action='store_true', default=True,
                       help='Use mixed precision training')
    parser.add_argument('--chunk_size', type=int, default=15000,
                       help='Chunk size for memory-efficient processing')

    # Logging arguments
    parser.add_argument('--log_interval', type=int, default=50,
                       help='Logging interval (batches)')
    parser.add_argument('--verbose', action='store_true', default=False,
                       help='Enable verbose logging')

    return parser

def main():
    """Main training function."""
    parser = get_base_parser()
    args = parser.parse_args()

    # Create configuration
    config = {
        # Data configuration
        'data_fraction': args.data_fraction,
        'chunk_size': args.chunk_size,

        # Model configuration
        'hidden_dims': args.hidden_dims,
        'dropout_rate': args.dropout_rate,
        'use_residual': args.use_residual,

        # Training configuration
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'warmup_steps': args.warmup_steps,
        'gradient_clip_value': args.gradient_clip_value,
        'early_stopping_patience': args.early_stopping_patience,

        # Optimization configuration
        'use_mixed_precision': args.use_mixed_precision,

        # Logging configuration
        'log_interval': args.log_interval,
        'verbose': args.verbose,
        'output_dir': args.output_dir
    }

    # Set logging level
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)

    # Print configuration
    logger.info("🎯 Robust Training Configuration:")
    logger.info("=" * 50)
    for key, value in config.items():
        logger.info(f"  • {key}: {value}")
    logger.info("=" * 50)

    # Create trainer
    trainer = RobustTrainer(config)

    # Start training
    success = trainer.train(args.data)

    if success:
        logger.info("🎉 Robust training completed successfully!")
        logger.info(f"📁 Models saved to: {args.output_dir}")
    else:
        logger.error("💥 Robust training failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()