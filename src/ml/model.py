"""
Machine learning model architecture for futures VPOC trading.
Optimized for AMD GPUs with ROCm support.
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

from src.utils.logging import get_logger

# Check if PyTorch is available
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    pass


class AMDOptimizedFuturesModel:
    """
    Neural Network model optimized for AMD GPUs with ROCm support.
    Implements a model that can run without PyTorch if not available.
    """
    
    def __init__(self, input_dim, hidden_layers=[64, 32], dropout_rate=0.4):
        """
        Initialize model architecture.
        
        Args:
            input_dim: Number of input features
            hidden_layers: List of hidden layer sizes
            dropout_rate: Dropout probability for regularization
        """
        self.logger = get_logger(__name__)
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        
        self.model = None
        
        # Check if PyTorch is available
        if TORCH_AVAILABLE:
            self.logger.info(f"Creating PyTorch model with input_dim={input_dim}")
            self._create_pytorch_model()
        else:
            self.logger.warning("PyTorch not available. Model will be in placeholder mode only.")
    
    def _create_pytorch_model(self):
        """Create PyTorch model with AMDGpu optimizations"""
        if not TORCH_AVAILABLE:
            return
            
        # Set ROCm environment variables if not already set
        if 'ROCM_HOME' not in os.environ:
            os.environ['ROCM_HOME'] = '/opt/rocm'
        if 'HIP_VISIBLE_DEVICES' not in os.environ:
            os.environ['HIP_VISIBLE_DEVICES'] = '0,1'
        if 'PYTORCH_ROCM_ARCH' not in os.environ:
            os.environ['PYTORCH_ROCM_ARCH'] = 'gfx1100'
        
        class AMDPytorchModel(nn.Module):
            """PyTorch model implementation"""
            def __init__(self, input_dim, hidden_layers, dropout_rate):
                super().__init__()
                
                # Input layer
                self.input_layer = nn.Linear(input_dim, hidden_layers[0])
                if hasattr(nn.init, 'kaiming_normal_'):
                    nn.init.kaiming_normal_(self.input_layer.weight)
                
                # Dynamic hidden layers
                layers = []
                prev_dim = hidden_layers[0]
                
                for i, hidden_dim in enumerate(hidden_layers[1:]):
                    # Gradually increase dropout
                    layer_dropout = dropout_rate * (1 + i * 0.2)
                    
                    layer_block = nn.Sequential(
                        nn.Linear(prev_dim, hidden_dim),
                        nn.GroupNorm(min(8, hidden_dim), hidden_dim),
                        nn.SiLU(),
                        nn.Dropout(layer_dropout)
                    )
                    layers.append(layer_block)
                    prev_dim = hidden_dim
                
                self.hidden_layers = nn.ModuleList(layers)
                
                # Output layer
                self.output_layer = nn.Linear(hidden_layers[-1], 1)
                nn.init.zeros_(self.output_layer.bias)
                
                # Normalization before output
                self.final_norm = nn.LayerNorm(hidden_layers[-1])
            
            def forward(self, x):
                # Input layer
                x = self.input_layer(x)
                x = F.silu(x)
                
                # Hidden layers
                for layer in self.hidden_layers:
                    x = layer(x)
                
                # Final normalization and output
                x = self.final_norm(x)
                return self.output_layer(x)
        
        # Create the model
        self.model = AMDPytorchModel(self.input_dim, self.hidden_layers, self.dropout_rate)
        self.logger.info("PyTorch model created successfully")
    
    def to_gpu(self):
        """Move model to GPU if available"""
        if TORCH_AVAILABLE and self.model is not None:
            # Check if CUDA/ROCm is available
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                self.logger.info(f"Model moved to GPU: {torch.cuda.get_device_name(0)}")
                return True
            else:
                self.logger.warning("CUDA/ROCm not available. Model remains on CPU.")
                return False
        return False
    
    def load_state_dict(self, state_dict):
        """Load model weights"""
        if TORCH_AVAILABLE and self.model is not None:
            self.model.load_state_dict(state_dict)
            self.logger.info("Model weights loaded successfully")
    
    def state_dict(self):
        """Get model state dictionary"""
        if TORCH_AVAILABLE and self.model is not None:
            return self.model.state_dict()
        return {}
    
    def eval(self):
        """Set model to evaluation mode"""
        if TORCH_AVAILABLE and self.model is not None:
            self.model.eval()
    
    def train(self):
        """Set model to training mode"""
        if TORCH_AVAILABLE and self.model is not None:
            self.model.train()
    
    def forward(self, x):
        """Forward pass (for compatibility)"""
        if TORCH_AVAILABLE and self.model is not None:
            return self.model(x)
        return None
    
    def __call__(self, x):
        """Call forward method"""
        return self.forward(x)


class ModelManager:
    """
    Manager for ML models with saving and loading capabilities.
    Compatible with systems that may not have PyTorch installed.
    """
    
    def __init__(self, model_dir=None):
        """
        Initialize model manager.
        
        Args:
            model_dir: Directory for saving models
        """
        self.logger = get_logger(__name__)
        
        # Get model directory from settings or use default
        if model_dir is None:
            try:
                from src.config.settings import settings
                self.model_dir = getattr(settings, 'TRAINING_DIR', 
                                         os.path.join(os.getcwd(), 'TRAINING'))
            except (ImportError, AttributeError):
                self.model_dir = os.path.join(os.getcwd(), 'TRAINING')
        else:
            self.model_dir = model_dir
            
        # Create directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.logger.info(f"Initialized ModelManager with model_dir: {self.model_dir}")
    
    def create_model(self, input_dim, hidden_layers=[64, 32], dropout_rate=0.4):
        """
        Create and initialize a new model.
        
        Args:
            input_dim: Number of input features
            hidden_layers: List of hidden layer sizes
            dropout_rate: Dropout probability
            
        Returns:
            Initialized model
        """
        model = AMDOptimizedFuturesModel(input_dim, hidden_layers, dropout_rate)
        self.logger.info(f"Created model with input_dim={input_dim}, hidden_layers={hidden_layers}")
        return model
    
    def save_model(self, model, feature_columns=None, scaler=None, extra_data=None, filename=None):
        """
        Save model to file with metadata.
        
        Args:
            model: Model to save
            feature_columns: List of feature names
            scaler: Feature scaler
            extra_data: Additional data to save
            filename: Custom filename
            
        Returns:
            Path to saved model
        """
        if not TORCH_AVAILABLE:
            self.logger.error("PyTorch not available. Cannot save model.")
            return None
            
        if filename is None:
            # Generate default filename with timestamp
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"futures_model_{timestamp}.pt"
            
        # Create full path
        model_path = os.path.join(self.model_dir, filename)
        
        try:
            # Prepare checkpoint data
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'feature_columns': feature_columns,
                'scaler': scaler
            }
            
            # Add extra data if provided
            if extra_data:
                checkpoint.update(extra_data)
                
            # Save to file
            torch.save(checkpoint, model_path)
            self.logger.info(f"Model saved to {model_path}")
            
            return model_path
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            return None
    
    def load_model(self, filepath):
        """
        Load model from file.
        
        Args:
            filepath: Path to saved model
            
        Returns:
            Tuple of (model, feature_columns, scaler, extra_data)
        """
        if not TORCH_AVAILABLE:
            self.logger.error("PyTorch not available. Cannot load model.")
            return None, None, None, None
            
        self.logger.info(f"Loading model from {filepath}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
            
            # Extract model state dict and metadata
            model_state = checkpoint.get('model_state_dict')
            feature_columns = checkpoint.get('feature_columns')
            
            if model_state is None:
                self.logger.error("Model state not found in checkpoint")
                return None, None, None, None
                
            # Determine input dimension from feature columns or first layer
            if feature_columns:
                input_dim = len(feature_columns)
            else:
                # Try to infer from model weights
                # Find first layer weight shape from state dict
                first_layer_key = next(key for key in model_state.keys() if 'input_layer.weight' in key)
                input_dim = model_state[first_layer_key].shape[1]
                self.logger.warning(f"No feature columns found, inferred input_dim={input_dim}")
            
            # Create model with determined architecture
            model = self.create_model(input_dim)
            model.load_state_dict(model_state)
            model.eval()  # Set to evaluation mode
            
            # Get scaler if available
            scaler = checkpoint.get('scaler')
            
            # Get extra data
            extra_data = {k: v for k, v in checkpoint.items() 
                          if k not in ['model_state_dict', 'feature_columns', 'scaler']}
                          
            self.logger.info(f"Successfully loaded model with input_dim={input_dim}")
            
            return model, feature_columns, scaler, extra_data
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return None, None, None, None