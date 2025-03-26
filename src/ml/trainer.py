"""
Model training functionality for futures VPOC trading strategy.
Handles dataset creation, training loop, and model evaluation.
"""

import os
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime

from src.utils.logging import get_logger
from src.ml.model import AMDOptimizedFuturesModel, ModelManager, TORCH_AVAILABLE

# Check for required ML libraries
SKLEARN_AVAILABLE = False
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    SKLEARN_AVAILABLE = True
except ImportError:
    pass

# Check if PyTorch is available
if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    import torch.optim as optim


class FuturesModelTrainer:
    """
    Training manager for AMD-optimized futures trading models.
    Handles data preparation, training, and evaluation.
    """
    
    def __init__(self, model_manager=None, training_dir=None):
        """
        Initialize trainer with model manager.
        
        Args:
            model_manager: Optional ModelManager instance
            training_dir: Directory for training outputs
        """
        self.logger = get_logger(__name__)
        
        # Check for PyTorch availability
        if not TORCH_AVAILABLE:
            self.logger.warning("PyTorch not available. Training functionality will be limited.")
        
        # Set up model manager
        if model_manager is None:
            self.model_manager = ModelManager(model_dir=training_dir)
        else:
            self.model_manager = model_manager
            
        # Initialize training metrics
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'epochs': 0,
            'best_val_loss': float('inf')
        }
    
    def prepare_features(self, features_df, target_column, feature_columns=None, 
                         test_size=0.2, time_series_split=True):
        """
        Prepare features for training with time series awareness.
        
        Args:
            features_df: DataFrame with features and target
            target_column: Name of target column
            feature_columns: List of feature columns (uses all numeric columns if None)
            test_size: Proportion of data for test set
            time_series_split: Whether to use time series splitting
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, feature_columns, scaler)
        """
        if not SKLEARN_AVAILABLE:
            self.logger.error("sklearn not available. Cannot prepare features.")
            return None, None, None, None, None, None
            
        self.logger.info(f"Preparing features with target column: {target_column}")
        
        # Select feature columns if not specified
        if feature_columns is None:
            # Exclude target and non-numeric columns
            exclude_cols = [target_column]
            if 'date' in features_df.columns:
                exclude_cols.append('date')
                
            # Find numeric columns
            feature_columns = [col for col in features_df.columns 
                              if col not in exclude_cols and
                              features_df[col].dtype in ['float64', 'float32', 'int64', 'int32']]
            
        self.logger.info(f"Using {len(feature_columns)} features")
        
        # Extract features and target
        X = features_df[feature_columns].values
        y = features_df[target_column].values
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Handle time series split
        if time_series_split and 'date' in features_df.columns:
            self.logger.info("Using time series split")
            
            # Sort by date
            sorted_indices = features_df['date'].argsort()
            X_scaled = X_scaled[sorted_indices]
            y = y[sorted_indices]
            
            # Split based on time series proportion
            split_idx = int(len(X_scaled) * (1 - test_size))
            X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            self.logger.info(f"Split data into {len(X_train)} training and {len(X_test)} testing samples")
        else:
            # Random split
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=42)
                
            self.logger.info(f"Random split: {len(X_train)} training and {len(X_test)} testing samples")
        
        return X_train, X_test, y_train, y_test, feature_columns, scaler
    
    def train(self, model, X_train, y_train, X_val=None, y_val=None, 
              epochs=50, batch_size=32, learning_rate=0.001, 
              weight_decay=1e-4, patience=10):
        """
        Train model with early stopping.
        
        Args:
            model: Model to train
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Max number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
            weight_decay: L2 regularization
            patience: Early stopping patience
            
        Returns:
            Training history
        """
        if not TORCH_AVAILABLE:
            self.logger.error("PyTorch not available. Cannot train model.")
            return None
            
        # Set up validation data
        if X_val is None or y_val is None:
            # Use a portion of training data for validation
            val_size = int(len(X_train) * 0.2)
            indices = torch.randperm(len(X_train))
            train_indices = indices[val_size:]
            val_indices = indices[:val_size]
            
            X_val = X_train[val_indices]
            y_val = y_train[val_indices]
            X_train = X_train[train_indices]
            y_train = y_train[train_indices]
            
            self.logger.info(f"Created validation set with {len(X_val)} samples")
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)
        
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        
        # Loss function and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(
            model.model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Initialize tracking variables
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        start_time = time.time()
        
        # Check if GPU is available
        if torch.cuda.is_available():
            self.logger.info(f"Moving model to GPU: {torch.cuda.get_device_name(0)}")
            model.to_gpu()
            X_train_tensor = X_train_tensor.cuda()
            y_train_tensor = y_train_tensor.cuda()
            X_val_tensor = X_val_tensor.cuda()
            y_val_tensor = y_val_tensor.cuda()
        
        # Training loop
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            
            # Mini-batch training
            for inputs, targets in train_loader:
                # Forward pass
                outputs = model.model(inputs)
                loss = criterion(outputs, targets)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            # Compute average training loss
            train_loss = running_loss / len(train_loader)
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model.model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()
                
                # Calculate validation metrics (accuracy etc.)
                val_preds = torch.sigmoid(val_outputs) > 0.5
                accuracy = (val_preds == y_val_tensor).float().mean().item()
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Log progress
            self.logger.info(f"Epoch {epoch+1}/{epochs}, "
                            f"Train Loss: {train_loss:.4f}, "
                            f"Val Loss: {val_loss:.4f}, "
                            f"Accuracy: {accuracy:.4f}")
            
            # Update history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['epochs'] = epoch + 1
            
            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.training_history['best_val_loss'] = val_loss
                epochs_without_improvement = 0
                
                # Save best model checkpoint
                checkpoint_path = os.path.join(
                    self.model_manager.model_dir, 
                    f"best_model_epoch_{epoch+1}.pt"
                )
                
                self.logger.info(f"New best model! Saving checkpoint to {checkpoint_path}")
                
                # Don't save during training - will save at the end
            else:
                epochs_without_improvement += 1
                
            # Early stopping
            if epochs_without_improvement >= patience:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Calculate total training time
        training_time = time.time() - start_time
        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Update final statistics
        self.training_history['training_time'] = training_time
        self.training_history['final_train_loss'] = train_loss
        self.training_history['final_val_loss'] = val_loss
        self.training_history['final_accuracy'] = accuracy
        
        return self.training_history
    
    def save_trained_model(self, model, feature_columns, scaler):
        """
        Save the trained model with metadata.
        
        Args:
            model: Trained model
            feature_columns: Feature column names
            scaler: Feature scaler
            
        Returns:
            Path to saved model
        """
        if not TORCH_AVAILABLE:
            self.logger.error("PyTorch not available. Cannot save model.")
            return None
            
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trained_model_{timestamp}.pt"
        
        # Add training history to extra data
        extra_data = self.training_history.copy()
        
        # Save model with metadata
        save_path = self.model_manager.save_model(
            model, 
            feature_columns=feature_columns,
            scaler=scaler,
            extra_data=extra_data,
            filename=filename
        )
        
        if save_path:
            self.logger.info(f"Trained model saved to {save_path}")
        
        return save_path

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
        if not TORCH_AVAILABLE:
            self.logger.error("PyTorch not available. Cannot evaluate model.")
            return None
            
        self.logger.info("Evaluating model on test set")
        
        # Convert to PyTorch tensors
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            X_test_tensor = X_test_tensor.cuda()
            y_test_tensor = y_test_tensor.cuda()
            model.to_gpu()
        
        # Set model to evaluation mode
        model.eval()
        
        # Compute predictions
        with torch.no_grad():
            test_outputs = model.model(X_test_tensor)
            test_probs = torch.sigmoid(test_outputs)
            test_preds = test_probs > 0.5
            
            # Calculate loss
            criterion = nn.BCEWithLogitsLoss()
            test_loss = criterion(test_outputs, y_test_tensor).item()
        
        # Convert to numpy for metric calculation
        y_pred = test_preds.cpu().numpy()
        y_prob = test_probs.cpu().numpy()
        y_true = y_test_tensor.cpu().numpy()
        
        # Calculate metrics
        metrics = {
            'test_loss': test_loss,
            'accuracy': (y_pred == y_true).mean(),
            'positive_rate': y_pred.mean(),
            'true_positive_rate': (y_pred & y_true).sum() / y_true.sum() if y_true.sum() > 0 else 0,
            'true_negative_rate': (~y_pred & ~y_true).sum() / (1 - y_true).sum() if (1 - y_true).sum() > 0 else 0
        }
        
        self.logger.info(f"Test Loss: {metrics['test_loss']:.4f}")
        self.logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        self.logger.info(f"Positive Rate: {metrics['positive_rate']:.4f}")
        self.logger.info(f"True Positive Rate: {metrics['true_positive_rate']:.4f}")
        self.logger.info(f"True Negative Rate: {metrics['true_negative_rate']:.4f}")
        
        return metrics
    
    def plot_training_history(self, save_path=None):
        """
        Plot training history.
        
        Args:
            save_path: Path to save the plot
            
        Returns:
            Path to saved plot or None
        """
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 6))
            
            # Plot training and validation loss
            plt.subplot(1, 2, 1)
            plt.plot(self.training_history['train_loss'], label='Training Loss')
            plt.plot(self.training_history['val_loss'], label='Validation Loss')
            plt.title('Loss During Training')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot learning progress
            plt.subplot(1, 2, 2)
            epochs = range(1, self.training_history['epochs'] + 1)
            best_epoch = self.training_history['val_loss'].index(
                min(self.training_history['val_loss'])) + 1
                
            plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch: {best_epoch}')
            plt.annotate(f'Best Val Loss: {min(self.training_history["val_loss"]):.4f}',
                        xy=(best_epoch, min(self.training_history["val_loss"])),
                        xytext=(best_epoch + 1, min(self.training_history["val_loss"])),
                        arrowprops=dict(facecolor='black', arrowstyle='->'),
                        fontsize=10)
            
            plt.title('Model Training Progress')
            plt.xlabel('Epoch')
            plt.ylabel('Metrics')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot if path provided
            if save_path:
                plt.savefig(save_path)
                self.logger.info(f"Training history plot saved to {save_path}")
                plt.close()
                return save_path
            else:
                plt.show()
                plt.close()
                return None
                
        except ImportError:
            self.logger.warning("matplotlib not available. Cannot plot training history.")
            return None