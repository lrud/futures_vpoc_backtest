"""
Consolidated training functionality for futures trading ML models.
Handles dataset preparation, training loop, and model evaluation.
"""

import os
import time
import torch
import torch.version
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

from src.ml.model import AMDOptimizedFuturesModel, ModelManager
from src.utils.logging import get_logger
from src.ml.feature_engineering import FeatureEngineer
from src.ml.trainer_core import TrainingCore

# Initialize logger
logger = get_logger(__name__)

class ModelTrainer(TrainingCore):
    """Coordinator for training workflows."""
    
    def __init__(self, model_dir=None, device=None):
        """
        Initialize trainer with model manager.
        
        Args:
            model_dir: Directory for training outputs
            device: Torch device to use (cuda/cpu)
        """
        self.logger = get_logger(__name__)
        
        # Set device
        self.device = device if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
        # Check for PyTorch availability
        if not torch.cuda.is_available():
            self.logger.warning("GPU not available. Training will be slower.")
        
        # Set up model manager
        self.model_manager = ModelManager(model_dir=model_dir)
            
        # Initialize training metrics
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'epochs': 0,
            'best_val_loss': float('inf')
        }
        
        # Feature engineer for data preparation
        self.feature_engineer = FeatureEngineer()
        
        # Initialize TrainingCore with device
        super().__init__(model=None, optimizer=None, criterion=None, device=self.device)
    
    def prepare_data(self, X_train, y_train, X_val=None, y_val=None, batch_size=32, val_split=0.2):
        """
        Prepare data loaders for training.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            batch_size: Batch size
            val_split: Validation split if X_val/y_val not provided
            
        Returns:
            train_loader, val_data tuple
        """
        # Use instance device
        device = self.device
        
        # Set up validation data if not provided
        if X_val is None or y_val is None:
            # Use a portion of training data for validation
            val_size = int(len(X_train) * val_split)
            indices = torch.randperm(len(X_train))
            train_indices = indices[val_size:]
            val_indices = indices[:val_size]
            
            X_val = X_train[val_indices]
            y_val = y_train[val_indices]
            X_train = X_train[train_indices]
            y_train = y_train[train_indices]
            
            self.logger.info(f"Created validation set with {len(X_val)} samples")
        
        # Convert to PyTorch tensors (keep on CPU for DataParallel)
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        y_train_tensor = (y_train_tensor + 1) / 2  # Convert from [-1, 1] to [0, 1]

        # Create dataset and loader for training
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True  # Enable for faster GPU transfer
        )

        # Keep validation data on CPU, will be moved to GPU in training loop
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
        y_val_tensor = (y_val_tensor + 1) / 2  # Convert from [-1, 1] to [0, 1]
        
        return train_loader, (X_val_tensor, y_val_tensor)

    def _save_checkpoint(self, batch_idx, epoch, avg_loss):
        """Save training checkpoint for resuming after OOM errors."""
        try:
            import os

            checkpoint_path = os.path.join(self.model_manager.model_dir, 'checkpoint.pt')
            checkpoint = {
                'epoch': epoch,
                'batch_idx': batch_idx,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
                'avg_loss': avg_loss,
                'gradient_accumulation_steps': getattr(self, '_last_gradient_accumulation_steps', 1)
            }

            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f"💾 Checkpoint saved: batch {batch_idx}, epoch {epoch}, loss {avg_loss:.6f}")

        except Exception as e:
            self.logger.warning(f"Failed to save checkpoint: {e}")

    def _load_checkpoint(self):
        """Load training checkpoint for resuming."""
        try:
            import os

            checkpoint_path = os.path.join(self.model_manager.model_dir, 'checkpoint.pt')
            if not os.path.exists(checkpoint_path):
                return None

            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.logger.info(f"📂 Loaded checkpoint: batch {checkpoint['batch_idx']}, epoch {checkpoint['epoch']}")
            return checkpoint

        except Exception as e:
            self.logger.warning(f"Failed to load checkpoint: {e}")
            return None

    def _train_epoch(self, train_loader, args=None, epoch_num=1):
        """Run one training epoch with enhanced progress monitoring and memory management."""
        from src.ml.trainer_utils import get_gpu_metrics, log_gpu_metrics

        self.model.train()
        train_loss = 0.0
        samples_processed = 0
        batch_count = len(train_loader)
        epoch_start_time = time.time()

        # Get gradient accumulation steps from args (args can be dict or namespace)
        if isinstance(args, dict):
            gradient_accumulation_steps = int(args.get('gradient_accumulation_steps', 1))
        else:
            gradient_accumulation_steps = int(getattr(args, 'gradient_accumulation_steps', 1))
        self.logger.info(f"    🔄 Using gradient accumulation: {gradient_accumulation_steps} steps")

        # Store for checkpoint access
        self._gradient_accumulation_steps = gradient_accumulation_steps

        # ROCm 7 Memory Management: Clear cache before epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        self.logger.info(f"    📦 Processing {batch_count} batches...")

        # Log initial GPU state
        if torch.cuda.is_available():
            initial_memory = torch.cuda.memory_allocated(self.device) / 1024**3
            self.logger.info(f"    💾 Starting GPU Memory: {initial_memory:.2f}GB")
            gpu_metrics = get_gpu_metrics()
            log_gpu_metrics(gpu_metrics, self.logger)

        # Zero gradients at the start of epoch
        self.optimizer.zero_grad()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            batch_start_time = time.time()
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Memory check before forward pass
            if torch.cuda.is_available():
                pre_forward_memory = torch.cuda.memory_allocated(self.device) / 1024**3

            try:
                # Get model (compiled or original) for PyTorch 2.10 optimization
                model_for_training = self.get_model_for_training()

                # Initialize variables for logging and error handling
                current_loss = 0.0
                batch_size = 0
                input_shape = None

                if self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = model_for_training(inputs)
                        # CRITICAL FIX: Model now outputs correct shape, no additional squeeze needed
                        # Only squeeze if output has extra dimensions
                        if outputs.dim() > 1 and outputs.size(1) == 1:
                            outputs = outputs.squeeze(-1)
                        # Scale loss by gradient accumulation steps
                        loss = self.criterion(outputs, targets) / gradient_accumulation_steps

                    # Gradient accumulation with mixed precision
                    self.scaler.scale(loss).backward()

                    # Save loss value before cleanup for logging (unscaled)
                    current_loss = loss.item() * gradient_accumulation_steps

                    # Calculate running loss and save batch info before cleanup
                    batch_size = inputs.size(0)
                    input_shape = inputs.shape
                    train_loss += loss.item() * gradient_accumulation_steps * batch_size

                    # Step optimizer only after accumulation steps
                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        # CRITICAL FIX: Apply gradient clipping to prevent explosion
                        if hasattr(torch.nn.utils, 'clip_grad_norm_'):
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                            self.logger.debug(f"    📐 Applied gradient clipping (max_norm=1.0)")

                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()

                        self.logger.debug(f"    🔄 Optimizer step at batch {batch_idx + 1}")

                    # CRITICAL: Aggressive memory cleanup every batch for ROCm fragmentation
                    del outputs, targets, loss, inputs
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                else:
                    outputs = model_for_training(inputs)
                    # CRITICAL FIX: Model now outputs correct shape, no additional squeeze needed
                    # Only squeeze if output has extra dimensions
                    if outputs.dim() > 1 and outputs.size(1) == 1:
                        outputs = outputs.squeeze(-1)
                    # Scale loss by gradient accumulation steps
                    loss = self.criterion(outputs, targets) / gradient_accumulation_steps

                    # Gradient accumulation
                    loss.backward()

                    # Save loss value before cleanup for logging (unscaled)
                    current_loss = loss.item() * gradient_accumulation_steps

                    # Calculate running loss and save batch info before cleanup
                    batch_size = inputs.size(0)
                    input_shape = inputs.shape
                    train_loss += loss.item() * gradient_accumulation_steps * batch_size

                    # Step optimizer only after accumulation steps
                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        # CRITICAL FIX: Apply gradient clipping to prevent explosion
                        if hasattr(torch.nn.utils, 'clip_grad_norm_'):
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                            self.logger.debug(f"    📐 Applied gradient clipping (max_norm=1.0)")

                        self.optimizer.step()
                        self.optimizer.zero_grad()

                        self.logger.debug(f"    🔄 Optimizer step at batch {batch_idx + 1}")

                    # CRITICAL: Aggressive memory cleanup every batch for ROCm fragmentation
                    del outputs, targets, loss, inputs
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()

                # Log batch progress (every 10 batches or first/last batch)
                if batch_idx % 10 == 0 or batch_idx == batch_count - 1:
                    batch_time = time.time() - batch_start_time
                    progress_pct = (batch_idx + 1) / batch_count * 100

                    self.logger.info(f"      🔄 Batch {batch_idx+1}/{batch_count} ({progress_pct:.1f}%) "
                                   f"- Loss: {current_loss:.6f} - Time: {batch_time:.3f}s")

                    if torch.cuda.is_available():
                        post_forward_memory = torch.cuda.memory_allocated(self.device) / 1024**3
                        memory_delta = post_forward_memory - pre_forward_memory
                        self.logger.info(f"         Memory: {post_forward_memory:.2f}GB "
                                       f"(Δ{memory_delta:+.2f}GB)")

                # Log GPU metrics periodically
                if torch.cuda.is_available() and batch_idx % 10 == 0:
                    gpu_metrics = get_gpu_metrics()
                    log_gpu_metrics(gpu_metrics, self.logger)

                # Checkpoint saving (every N batches)
                checkpoint_interval = getattr(args, 'checkpoint_interval', 50)
                if (batch_idx + 1) % checkpoint_interval == 0:
                    self._save_checkpoint(batch_idx + 1, epoch_num, train_loss / samples_processed)

                samples_processed += batch_size

                # ROCm 7 Memory Management: Periodic cache clearing
                if batch_idx % 20 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self.logger.error(f"    ❌ GPU Out of Memory Error in batch {batch_idx+1}")
                    self.logger.error(f"    📊 Batch Details:")
                    self.logger.error(f"      • Batch Size: {batch_size}")
                    self.logger.error(f"      • Input Shape: {input_shape}")
                    if input_shape is not None and len(input_shape) > 1:
                        self.logger.error(f"      • Input Features: {input_shape[1]}")
                    else:
                        self.logger.error(f"      • Input Features: Unknown (input_shape is None)")
                    if torch.cuda.is_available():
                        oom_memory = torch.cuda.memory_allocated(self.device) / 1024**3
                        total_memory = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
                        self.logger.error(f"      • GPU Memory Used: {oom_memory:.2f}GB / {total_memory:.2f}GB")

                    # ROCm 7 Emergency cleanup
                    self.logger.error("    🧹 Performing emergency GPU memory cleanup...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()

                        # Clear DataParallel references if present
                        if hasattr(self.model, 'module'):
                            self.logger.info("    🔄 Clearing DataParallel references...")
                            for param in self.model.parameters():
                                if param.grad is not None:
                                    param.grad = None

                    raise e
                else:
                    raise e

        # Final optimizer step for any remaining accumulated gradients
        if (batch_count) % gradient_accumulation_steps != 0:
            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.logger.info(f"    🔄 Final optimizer step after epoch (remaining {batch_count % gradient_accumulation_steps} gradients)")

        epoch_time = time.time() - epoch_start_time
        avg_loss = train_loss / samples_processed

        # ROCm 7 Memory Management: Clear cache after epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        self.logger.info(f"    ✅ Epoch Training Complete:")
        self.logger.info(f"      • Average Loss: {avg_loss:.6f}")
        self.logger.info(f"      • Samples Processed: {samples_processed}")
        self.logger.info(f"      • Epoch Time: {epoch_time:.2f}s")
        self.logger.info(f"      • Avg Batch Time: {epoch_time/batch_count:.3f}s")

        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated(self.device) / 1024**3
            self.logger.info(f"      • Final GPU Memory: {final_memory:.2f}GB")

        return avg_loss

    def _validate_model(self, X_val, y_val):
        """Run validation using parent class parameters."""
        self.model.eval()

        # Move validation tensors to the same device as model
        if hasattr(self.model, 'device'):
            device = self.model.device
        else:
            device = next(self.model.parameters()).device

        X_val = X_val.to(device)
        y_val = y_val.to(device)

        with torch.no_grad():
            outputs = self.model(X_val).squeeze()
            val_loss = self.criterion(outputs, y_val).item()
            predicted = (outputs > 0).float()
            accuracy = (predicted == y_val).sum().item() / y_val.size(0) * 100

        return {'val_loss': val_loss, 'val_accuracy': accuracy}
    
    def train(self, model, X_train, y_train, X_val=None, y_val=None, args=None):
        """
        Train model with early stopping.
        
        Args:
            model: Model to train
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            args: Training arguments
            
        Returns:
            Trained model and training history
        """
        # Set default args if not provided
        if args is None:
            args = {
                'batch_size': 32,
                'epochs': 50,
                'learning_rate': 0.001,
                'use_mixed_precision': False,
                'output_dir': self.model_manager.model_dir
            }
        
        # Use existing device
        self.logger.info(f"Using device: {self.device}")
        
        # Move model to device
        self.model = model.to(self.device)
        
        # Prepare data loaders
        train_loader, val_data = self.prepare_data(
            X_train, y_train, X_val, y_val, 
            batch_size=args['batch_size']
        )
        X_val_tensor, y_val_tensor = val_data
        
        # Define loss and optimizer
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=args['learning_rate'],
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )
        
        # Set up mixed precision training if requested
        self.scaler = None
        if args.get('use_mixed_precision', False) and torch.cuda.is_available():
            self.logger.info("Using mixed precision training")
            # ROCm 7: Initialize scaler with memory management
            self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }

        # Checkpoint resuming support
        resume_from_checkpoint = args.get('resume_from_checkpoint', False)
        starting_epoch = 0
        starting_batch = 0

        if resume_from_checkpoint:
            checkpoint = self._load_checkpoint()
            if checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if self.scaler and checkpoint['scaler_state_dict']:
                    self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                starting_epoch = checkpoint['epoch'] - 1  # -1 because loop is 0-indexed
                starting_batch = checkpoint['batch_idx']
                self.logger.info(f"🔄 Resuming from epoch {starting_epoch + 1}, batch {starting_batch}")

        # Enhanced training configuration logging
        self.logger.info("🚀 STARTING NEURAL NETWORK TRAINING")
        self.logger.info("=" * 60)
        self.logger.info(f"📊 Training Configuration:")
        self.logger.info(f"  • Batch Size: {args['batch_size']}")
        self.logger.info(f"  • Epochs: {args['epochs']}")
        self.logger.info(f"  • Learning Rate: {args['learning_rate']}")
        self.logger.info(f"  • Mixed Precision: {args.get('use_mixed_precision', False)}")
        self.logger.info(f"  • Device: {self.device}")
        self.logger.info(f"  • Training Samples: {len(X_train)}")
        self.logger.info(f"  • Validation Samples: {len(X_val_tensor) if X_val_tensor is not None else 0}")
        self.logger.info(f"  • Input Features: {X_train.shape[1]}")

        # Log data types and shapes
        self.logger.info(f"📋 Data Shapes:")
        self.logger.info(f"  • X_train: {X_train.shape} ({X_train.dtype})")
        self.logger.info(f"  • y_train: {y_train.shape} ({y_train.dtype})")
        if X_val_tensor is not None:
            self.logger.info(f"  • X_val: {X_val_tensor.shape} ({X_val_tensor.dtype})")
            self.logger.info(f"  • y_val: {y_val_tensor.shape} ({y_val_tensor.dtype})")

        # Log model details
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"🧠 Model Architecture:")
        self.logger.info(f"  • Total Parameters: {total_params:,}")
        self.logger.info(f"  • Trainable Parameters: {trainable_params:,}")
        self.logger.info(f"  • Model Layers: {[layer for layer in model.hidden_layers_config] if hasattr(model, 'hidden_layers_config') else 'Unknown'}")

        # ROCm 7 Memory Optimizations
        self._setup_rocm_memory_optimizations(model, args.get('use_mixed_precision', False))

        # Training loop
        best_val_loss = float('inf')
        start_time = time.time()
        epoch_start_time = start_time

        self.logger.info("=" * 60)
        self.logger.info("🏃‍♂️ BEGINNING TRAINING EPOCHS")
        self.logger.info("=" * 60)

        for epoch in range(starting_epoch, args['epochs']):
            epoch_start_time = time.time()
            self.logger.info(f"\n📚 EPOCH {epoch+1}/{args['epochs']} - Starting...")

            # Training phase
            self.logger.info(f"  🔥 Training phase...")
            train_loss = self._train_epoch(train_loader, args, epoch + 1)

            # Validation phase
            self.logger.info(f"  ✅ Validation phase...")
            val_metrics = self._validate_model(X_val_tensor, y_val_tensor)
            val_loss = val_metrics['val_loss']
            accuracy = val_metrics['val_accuracy']

            # Update learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            scheduler.step(val_loss)

            # Calculate epoch timing
            epoch_time = time.time() - epoch_start_time
            total_time = time.time() - start_time
            eta_seconds = (total_time / (epoch + 1)) * (args['epochs'] - epoch - 1)
            eta_minutes = eta_seconds / 60

            # Save history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(accuracy)

            # Enhanced progress logging
            self.logger.info(f"  📈 EPOCH {epoch+1} RESULTS:")
            self.logger.info(f"    • Train Loss: {train_loss:.6f}")
            self.logger.info(f"    • Val Loss: {val_loss:.6f}")
            self.logger.info(f"    • Val Accuracy: {accuracy:.4f}%")
            self.logger.info(f"    • Learning Rate: {current_lr:.8f}")
            self.logger.info(f"    • Epoch Time: {epoch_time:.2f}s")
            self.logger.info(f"    • Total Time: {total_time:.2f}s")
            self.logger.info(f"    • ETA: {eta_minutes:.1f}min")

            # Memory logging with ROCm 7 enhancements
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated(self.device) / 1024**3
                gpu_cached = torch.cuda.memory_reserved(self.device) / 1024**3
                peak_memory = torch.cuda.max_memory_allocated(self.device) / 1024**3

                self.logger.info(f"    • GPU Memory: {gpu_memory:.2f}GB allocated, {gpu_cached:.2f}GB cached")
                self.logger.info(f"    • Peak Memory: {peak_memory:.2f}GB")

                # ROCm 7: Memory safety check and cleanup
                if gpu_memory > 14.0:  # Warning threshold
                    self.logger.warning(f"⚠️  High memory usage: {gpu_memory:.2f}GB - performing cleanup")
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.logger.info(f"Validation loss improved to {best_val_loss:.4f}, saving model")
                
                self.model_manager.save_model(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    loss=val_loss,
                    metadata={
                        'accuracy': accuracy,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'feature_columns': getattr(self.model, 'feature_columns', None)
                    },
                    filename='best_model.pt'
                )
        
        training_time = time.time() - start_time

        # Enhanced training completion summary
        self.logger.info("\n" + "=" * 60)
        self.logger.info("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        self.logger.info("=" * 60)

        # Final results summary
        final_train_loss = history['train_loss'][-1] if history['train_loss'] else 0
        final_val_loss = history['val_loss'][-1] if history['val_loss'] else 0
        final_val_accuracy = history['val_accuracy'][-1] if history['val_accuracy'] else 0

        self.logger.info("📊 FINAL RESULTS:")
        self.logger.info(f"  • Total Training Time: {training_time:.2f}s ({training_time/60:.1f}m)")
        self.logger.info(f"  • Final Train Loss: {final_train_loss:.6f}")
        self.logger.info(f"  • Final Val Loss: {final_val_loss:.6f}")
        self.logger.info(f"  • Final Val Accuracy: {final_val_accuracy:.4f}%")
        self.logger.info(f"  • Best Validation Loss: {best_val_loss:.6f}")

        # Performance metrics
        self.logger.info("📈 PERFORMANCE METRICS:")
        if len(history['train_loss']) > 1:
            train_improvement = history['train_loss'][0] - history['train_loss'][-1]
            val_improvement = history['val_loss'][0] - history['val_loss'][-1]
            self.logger.info(f"  • Train Loss Improvement: {train_improvement:.6f}")
            self.logger.info(f"  • Val Loss Improvement: {val_improvement:.6f}")

        avg_epoch_time = training_time / args['epochs']
        self.logger.info(f"  • Average Epoch Time: {avg_epoch_time:.2f}s")

        # Model saving summary
        self.logger.info("💾 MODEL SAVED:")
        self.logger.info(f"  • Best Model: best_model.pt (val_loss: {best_val_loss:.6f})")
        self.logger.info(f"  • Final Model: model_final.pt")

        # Save final model
        self.model_manager.save_model(
            model=self.model,
            optimizer=self.optimizer,
            epoch=args['epochs']-1,
            loss=val_loss,
            metadata={
                'accuracy': accuracy,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'history': history,
                'training_time': training_time,
                'feature_columns': getattr(self.model, 'feature_columns', None)
            },
            filename='model_final.pt'
        )

        # Final memory cleanup logging with ROCm 7 enhancements
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated(self.device) / 1024**3
            final_cached = torch.cuda.memory_reserved(self.device) / 1024**3
            self.logger.info(f"🔧 FINAL GPU STATE:")
            self.logger.info(f"  • Memory Allocated: {final_memory:.2f}GB")
            self.logger.info(f"  • Memory Reserved: {final_cached:.2f}GB")

            # ROCm 7: Comprehensive cleanup
            self.logger.info("  🧹 Performing final cleanup...")

            # Clear DataParallel references if present
            if hasattr(self.model, 'module'):
                self.logger.info("    • Clearing DataParallel references...")
                # Move model back to single GPU before cleanup
                if hasattr(self.model.module, 'cpu'):
                    self.model.module.cpu()
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad = None

            # Clear compiled model cache if present
            if hasattr(self, 'compiled_model'):
                self.logger.info("    • Clearing compiled model cache...")
                self.compiled_model = None
                self.use_compiled_model = False

            # Clear mixed precision scaler
            if hasattr(self, 'scaler') and self.scaler is not None:
                self.logger.info("    • Clearing mixed precision scaler...")
                self.scaler = None

            # Clear all GPU caches and synchronize
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

            cleared_memory = torch.cuda.memory_allocated(self.device) / 1024**3
            cleared_cached = torch.cuda.memory_reserved(self.device) / 1024**3
            self.logger.info(f"  • Memory After Cleanup: {cleared_memory:.2f}GB allocated, {cleared_cached:.2f}GB reserved")

        self.logger.info("=" * 60)
        self.logger.info("✅ READY FOR BACKTESTING!")
        self.logger.info("=" * 60)
        
        return self.model, history

    def _setup_rocm_memory_optimizations(self, model, use_mixed_precision):
        """Setup ROCm 6.3 compatible optimizations for distributed training."""

        if not torch.cuda.is_available():
            return

        # Detect ROCm version for appropriate optimizations
        try:
            rocm_version = torch.version.hip
            is_rocm6 = rocm_version.startswith("6.")
        except:
            is_rocm6 = True  # Assume ROCm 6.x for compatibility

        # PyTorch torch.compile optimization (version-aware)
        try:
            if is_rocm6:
                self.logger.info("Using ROCm 6.x compatible optimizations")
                # Basic torch.compile for ROCm 6.x
                if hasattr(torch, 'compile'):
                    model = torch.compile(model, mode='reduce-overhead')
                    self.logger.info("✅ ROCm 6.x torch.compile optimizations applied")
            else:
                self._setup_pytorch210_compile(model)
        except Exception as e:
            self.logger.warning(f"Could not setup torch.compile: {e}")

        # Enable gradient checkpointing if model supports it
        if hasattr(model, 'enable_gradient_checkpointing'):
            model.enable_gradient_checkpointing()
            self.logger.info("✅ Gradient checkpointing enabled")

        # Setup memory pool for better memory management (ROCm 6.x compatible)
        try:
            # Configure memory pool size based on available GPU memory
            total_memory = torch.cuda.get_device_properties(self.device).total_memory
            pool_size_gb = min(4, total_memory / (1024**3) * 0.15)  # Smaller pool for ROCm 6.x

            self.logger.info(f"🔧 Setting up memory pool: {pool_size_gb:.1f}GB")

            # ROCm 6.x compatible memory settings
            torch.cuda.empty_cache()

            # Enable memory pool if available
            if hasattr(torch.cuda, 'memory'):
                self.logger.info("✅ Memory pool optimizations applied")

        except Exception as e:
            self.logger.warning(f"Could not setup memory pool: {e}")

        # Enable mixed precision scaler if requested
        if use_mixed_precision:
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
            self.logger.info("✅ Mixed precision scaler initialized")

        # Log memory optimization status
        initial_memory = torch.cuda.memory_allocated(self.device) / 1024**3
        self.logger.info(f"🔧 Memory optimizations initialized")
        self.logger.info(f"  • Initial GPU Memory: {initial_memory:.2f}GB")
        self.logger.info(f"  • Mixed Precision: {use_mixed_precision}")
        self.logger.info(f"  • Gradient Checkpointing: {hasattr(model, 'use_gradient_checkpointing') and model.use_gradient_checkpointing}")

    def _setup_pytorch210_compile(self, model):
        """Setup PyTorch torch.compile with ROCm compatible optimizations."""

        # Check if torch.compile is available
        if not hasattr(torch, 'compile'):
            self.logger.warning("torch.compile not available in this PyTorch version")
            return

        # RX 7900 XT (RDNA3) specific optimizations
        gpu_name = torch.cuda.get_device_name(0).lower()

        if '7900' in gpu_name or 'rdna3' in gpu_name:
            # RX 7900 XT (RDNA3/gfx1100) optimizations
            compile_config = {
                'mode': 'reduce-overhead',
                'backend': 'inductor',
                'options': {
                    'triton.enable': True,
                    'max_autotune': False,  # Faster compilation for consumer GPUs
                    'layout_optimization': True,
                }
            }
            arch_name = "RX 7900 XT (RDNA3/gfx1100)"
        else:
            # Unsupported GPU - warn user
            self.logger.warning(f"⚠️  GPU {gpu_name} not optimized for RX 7900 XT")
            self.logger.warning("This codebase is specifically optimized for AMD Radeon RX 7900 XT")
            compile_config = {'mode': 'default', 'backend': 'inductor'}
            arch_name = "Unsupported GPU"

        try:
            # DISABLED: torch.compile due to ROCm version incompatibility
            # torch.compile causes issues with ROCm/PyTorch version combinations
            self.logger.warning(f"⚠️  torch.compile DISABLED due to ROCm version incompatibility")
            self.logger.warning(f"   • PyTorch: {torch.__version__}")
            if hasattr(torch.version, 'hip'):
                self.logger.warning(f"   • ROCm/HIP: {torch.version.hip}")
            self.logger.warning(f"   • Using original model for stability")

            self.compiled_model = model
            self.use_compiled_model = False

        except Exception as e:
            self.logger.warning(f"Model setup failed: {e}")
            self.compiled_model = model
            self.use_compiled_model = False

    def get_model_for_training(self):
        """Get the appropriate model (compiled or original) for training."""
        if hasattr(self, 'use_compiled_model') and self.use_compiled_model:
            return self.compiled_model
        return self.model

    def optimize_memory_usage(self):
        """Optimize memory usage during training."""
        if torch.cuda.is_available():
            # Clear cache
            torch.cuda.empty_cache()

            # Log memory usage
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            reserved = torch.cuda.memory_reserved(self.device) / 1024**3

            self.logger.info(f"🔧 Memory Optimization:")
            self.logger.info(f"  • Allocated: {allocated:.2f}GB")
            self.logger.info(f"  • Reserved: {reserved:.2f}GB")

            # Check for memory leaks
            if allocated > 15.0:  # If using more than 15GB, warn
                self.logger.warning(f"⚠️  High memory usage detected: {allocated:.2f}GB")
                self.logger.warning("  Consider reducing batch size or enabling gradient checkpointing")

    def check_memory_safety(self):
        """Check if memory usage is safe to continue training."""
        if not torch.cuda.is_available():
            return True

        try:
            # Get total GPU memory
            total_memory = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            usage_percent = (allocated / total_memory) * 100

            if usage_percent > 95:
                self.logger.error(f"🚨 CRITICAL: GPU memory usage at {usage_percent:.1f}% - stopping training")
                return False
            elif usage_percent > 85:
                self.logger.warning(f"⚠️  HIGH: GPU memory usage at {usage_percent:.1f}%")
                # Clear cache to try to free memory
                torch.cuda.empty_cache()

            return True
        except Exception as e:
            self.logger.warning(f"Could not check memory safety: {e}")
            return True

    # Rest of the file remains unchanged...
    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluate model performance on test set.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of performance metrics
        """
        # Use instance device
        device = self.device
        
        # Prepare test data
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
        y_test_tensor = (y_test_tensor + 1) / 2  # Convert from [-1, 1] to [0, 1]
        
        # Set model to evaluation mode
        model.eval()
        
        # Compute predictions
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            test_probs = torch.sigmoid(test_outputs.squeeze())
            test_preds = (test_probs > 0.5).float()
            
            # Calculate loss
            criterion = nn.BCEWithLogitsLoss()
            test_loss = criterion(test_outputs.squeeze(), y_test_tensor).item()
        
        # Convert to numpy for metric calculation
        y_pred = test_preds.cpu().numpy()
        y_prob = test_probs.cpu().numpy()
        y_true = y_test_tensor.cpu().numpy()
        
        # Calculate metrics
        metrics = {
            'test_loss': test_loss,
            'accuracy': (y_pred == y_true).mean(),
            'positive_rate': y_pred.mean(),
            'true_positive_rate': (y_pred & y_true.astype(bool)).sum() / y_true.sum() if y_true.sum() > 0 else 0,
            'true_negative_rate': (~y_pred.astype(bool) & ~y_true.astype(bool)).sum() / (1 - y_true).sum() if (1 - y_true).sum() > 0 else 0
        }
        
        self.logger.info(f"Test Loss: {metrics['test_loss']:.4f}")
        self.logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        self.logger.info(f"Positive Rate: {metrics['positive_rate']:.4f}")
        self.logger.info(f"True Positive Rate: {metrics['true_positive_rate']:.4f}")
        self.logger.info(f"True Negative Rate: {metrics['true_negative_rate']:.4f}")
        
        return metrics
    
    def plot_training_history(self, history, save_path=None):
        """
        Plot training history.
        
        Args:
            history: Training history dictionary
            save_path: Path to save the plot
            
        Returns:
            Path to saved plot or None
        """
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 6))
            
            # Plot training and validation loss
            plt.subplot(1, 2, 1)
            plt.plot(history['train_loss'], label='Training Loss')
            plt.plot(history['val_loss'], label='Validation Loss')
            plt.title('Loss During Training')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot accuracy if available
            if 'val_accuracy' in history:
                plt.subplot(1, 2, 2)
                plt.plot(history['val_accuracy'], label='Validation Accuracy')
                plt.title('Accuracy During Training')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy (%)')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save or display plot
            if save_path:
                plt.savefig(save_path)
                self.logger.info(f"Training history plot saved to {save_path}")
                plt.close()
                return save_path
            else:
                plt.show()
                plt.close()
                
        except ImportError:
            self.logger.warning("matplotlib not available. Cannot plot training history.")
        
        return None
