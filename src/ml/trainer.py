"""
Consolidated training functionality for futures trading ML models.
Handles dataset preparation, training loop, and model evaluation.
"""

import os
import time
import torch
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
        
        # Convert to PyTorch tensors and move to GPU immediately
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
        y_train_tensor = (y_train_tensor + 1) / 2  # Convert from [-1, 1] to [0, 1]
        
        # Create dataset and loader for training
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            pin_memory=False  # Disable since data is already on GPU
        )
        
        # Move validation data to GPU
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)
        y_val_tensor = (y_val_tensor + 1) / 2  # Convert from [-1, 1] to [0, 1]
        
        return train_loader, (X_val_tensor, y_val_tensor)
    
    def _train_epoch(self, train_loader):
        """Run one training epoch using parent class parameters."""
        from src.ml.trainer_utils import get_gpu_metrics, log_gpu_metrics
        
        self.model.train()
        train_loss = 0.0
        samples_processed = 0
        
        # Log initial GPU state
        if torch.cuda.is_available():
            gpu_metrics = get_gpu_metrics()
            log_gpu_metrics(gpu_metrics, self.logger)
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            
            # Log GPU metrics periodically
            if torch.cuda.is_available() and batch_idx % 10 == 0:
                gpu_metrics = get_gpu_metrics()
                log_gpu_metrics(gpu_metrics, self.logger)
            
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs).squeeze()
                    loss = self.criterion(outputs, targets)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs).squeeze()
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            samples_processed += inputs.size(0)
            
        return train_loss / samples_processed

    def _validate_model(self, X_val, y_val):
        """Run validation using parent class parameters."""
        self.model.eval()
        
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
            patience=10,
            verbose=True
        )
        
        # Set up mixed precision training if requested
        self.scaler = None
        if args.get('use_mixed_precision', False) and torch.cuda.is_available():
            self.logger.info("Using mixed precision training")
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        # Training loop
        best_val_loss = float('inf')
        start_time = time.time()
        
        for epoch in range(args['epochs']):
            # Training phase
            train_loss = self._train_epoch(train_loader)
            
            # Validation phase
            val_metrics = self._validate_model(X_val_tensor, y_val_tensor)
            val_loss = val_metrics['val_loss']
            accuracy = val_metrics['val_accuracy']
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Save history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(accuracy)
            
            # Log progress
            self.logger.info(f"Epoch {epoch+1}/{args['epochs']} - "
                           f"Train Loss: {train_loss:.4f} - "
                           f"Val Loss: {val_loss:.4f} - "
                           f"Val Accuracy: {accuracy:.2f}%")
            
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
        
        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        
        return self.model, history
    
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
