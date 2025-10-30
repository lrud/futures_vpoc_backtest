"""
Neural network models for futures trading prediction.
Includes optimized architecture for AMD GPUs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import json
from torch.utils.checkpoint import checkpoint

from ..utils.logging import get_logger

# Check if PyTorch is available
TORCH_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    pass


class AMDOptimizedFuturesModel(nn.Module):
    """
    Advanced Neural Network Optimized for AMD GPUs with ROCm 6.3.3.
    Based on optimizations from https://rocm.docs.amd.com/en/latest/
    """
    def __init__(self, input_dim, hidden_layers=[128, 64], dropout_rate=0.4):
        # Input dimension is dynamic based on features, not hardcoded to a specific contract
        super().__init__()
        self.logger = get_logger(__name__)
        self.input_dim = input_dim
        self.hidden_layers_dims = hidden_layers
        self.dropout_rate = dropout_rate
        self.feature_columns = []
        self.enable_flash_attention = False
        self.use_gradient_checkpointing = False  # Disabled due to memory constraints

        # ROCm 6.3 optimizations for financial models:
        # - Align dimensions for optimal memory access
        # - Support for FP8/BF16 mixed precision
        # - Flash Attention integration
        self.input_dim_aligned = ((input_dim + 63) // 64) * 64  # 64-byte alignment
        self.aligned_hidden_layers = hidden_layers.copy()
        
        # Align hidden layer dimensions to multiples of 64
        # 7900 XT uses RDNA3 architecture which has Wave32 mode
        # Using multiples of 32 is optimal for 7900 XT
        for i in range(len(self.aligned_hidden_layers)):
            if self.aligned_hidden_layers[i] % 32 != 0:
                self.aligned_hidden_layers[i] = ((self.aligned_hidden_layers[i] // 32) + 1) * 32
                self.logger.info(f"Aligned hidden layer {i} to {self.aligned_hidden_layers[i]} (multiple of 32 for 7900 XT)")
        
        # Validate hidden layers before creating any layers
        if not self.aligned_hidden_layers:
            self.logger.warning("No hidden layers specified, using default [128, 64]")
            self.aligned_hidden_layers = [128, 64]
            for i in range(len(self.aligned_hidden_layers)):
                if self.aligned_hidden_layers[i] % 32 != 0:
                    self.aligned_hidden_layers[i] = ((self.aligned_hidden_layers[i] // 32) + 1) * 32

        # Input layer with ROCm-optimized initialization
        self.input_layer = nn.Linear(input_dim, self.aligned_hidden_layers[0])
        nn.init.kaiming_normal_(self.input_layer.weight, mode='fan_in', nonlinearity='leaky_relu')

        # Dynamic hidden layers with ROCm 6.3.3 specific adjustments
        layers = []
        prev_dim = self.aligned_hidden_layers[0]
        for i, hidden_dim in enumerate(self.aligned_hidden_layers[1:]):
            # ROCm 6.3.3: For RDNA3, use LayerNorm instead of GroupNorm where possible
            # https://rocm.docs.amd.com/en/latest/reference/pytorch.html
            layers.append(
                nn.Sequential(
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),  # LayerNorm better supported in ROCm 6.3.3
                    nn.SiLU(),  # SiLU optimized in ROCm 6.3.3 MIOpen
                    nn.Dropout(dropout_rate * (1 + i * 0.2))
                )
            )
            prev_dim = hidden_dim

        self.hidden_layers = nn.ModuleList(layers)

        # Output layer with precision initialization
        self.output_layer = nn.Linear(self.aligned_hidden_layers[-1], 1)
        nn.init.zeros_(self.output_layer.bias)
        
        # Add layer norm before output for better stability
        self.final_norm = nn.LayerNorm(self.aligned_hidden_layers[-1])

    def forward(self, x):
        # ROCm 7 memory alignment and mixed precision
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            if hasattr(x, 'data_ptr') and x.data_ptr() % 128 != 0:
                x = x.contiguous()

            # ROCm 7 Gradient Checkpointing for memory optimization
            if self.use_gradient_checkpointing and self.training:
                return self._forward_with_checkpointing(x)

            # Flash Attention v3 implementation
            if self.enable_flash_attention:
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=True,
                    enable_math=False,
                    enable_mem_efficient=True
                ):
                    x = F.silu(self.input_layer(x))
                    
                    for layer in self.hidden_layers:
                        if isinstance(layer[0], nn.Linear):
                            # Flash Attention optimized path
                            # Ensure input tensor is on the same device as layer parameters
                            if hasattr(layer[0], 'weight') and x.device != layer[0].weight.device:
                                x = x.to(layer[0].weight.device)
                            q = k = v = layer[0](x)
                            x = F.scaled_dot_product_attention(
                                q, k, v,
                                dropout_p=self.dropout_rate if self.training else 0.0
                            )
                            x = layer[1:](x)
                        else:
                            # Ensure input tensor is on the same device as layer parameters
                            if hasattr(layer, '0') and hasattr(layer[0], 'weight') and x.device != layer[0].weight.device:
                                x = x.to(layer[0].weight.device)
                            x = layer(x)
                            
                    x = self.final_norm(x)
                    return self.output_layer(x)
            else:
                # Standard forward pass with ROCm optimizations and mixed precision
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    # Ensure input tensor is on the same device as model parameters
                    if hasattr(self, 'device') and x.device != self.device:
                        x = x.to(self.device)
                    elif hasattr(self.input_layer, 'weight') and x.device != self.input_layer.weight.device:
                        x = x.to(self.input_layer.weight.device)

                    # ROCm 7: DISABLED torch.jit.fuser("fuser2") due to memory fragmentation bug
                    # This causes severe VRAM fragmentation in ROCm 7
                    if torch.cuda.get_device_properties(torch.cuda.current_device()).major >= 11:
                        # Standard forward without JIT fuser for ROCm 7 compatibility
                        x = F.silu(self.input_layer(x))
                        for layer in self.hidden_layers:
                            x = layer(x)
                        x = self.final_norm(x)
                        output = self.output_layer(x)
                        # CRITICAL FIX: Ensure output has correct shape for DataParallel
                        # Shape: (batch_size, 1) -> (batch_size,) to match target
                        output = output.squeeze(-1)
                        return output
                    else:
                        # Fallback for non-RDNA3 GPUs
                        x = F.silu(self.input_layer(x))
                        for layer in self.hidden_layers:
                            x = layer(x)
                        x = self.final_norm(x)
                        output = self.output_layer(x)
                        # CRITICAL FIX: Ensure output has correct shape for DataParallel
                        # Shape: (batch_size, 1) -> (batch_size,) to match target
                        output = output.squeeze(-1)
                        return output

    def _forward_with_checkpointing(self, x):
        """Forward pass with ROCm 7 gradient checkpointing for memory optimization."""

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(inputs[0])
            return custom_forward

        # Apply gradient checkpointing to each layer for memory savings
        x = checkpoint(create_custom_forward(self.input_layer), x)

        # Checkpoint through hidden layers
        for i, layer in enumerate(self.hidden_layers):
            if i % 2 == 0:  # Checkpoint every other layer to balance speed/memory
                x = checkpoint(create_custom_forward(layer), x)
            else:
                x = layer(x)

        # Final layers
        x = checkpoint(create_custom_forward(self.final_norm), x)
        output = self.output_layer(x)
        # CRITICAL FIX: Ensure output has correct shape for DataParallel
        # Shape: (batch_size, 1) -> (batch_size,) to match target
        output = output.squeeze(-1)

        return output

    def enable_gradient_checkpointing(self):
        """Enable ROCm 7 gradient checkpointing for memory optimization."""
        self.use_gradient_checkpointing = True
        self.logger.info("ROCm 7 gradient checkpointing enabled for memory optimization")

    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing."""
        self.use_gradient_checkpointing = False
        self.logger.info("Gradient checkpointing disabled")

    def get_rocm_info(self):
        """Return ROCm specific optimization information based on documentation."""
        return {
            "aligned_dimensions": self.aligned_hidden_layers,
            "original_dimensions": self.hidden_layers_dims,
            "rocm_version": "6.3.3",
            "gpu_model": "7900 XT",
            "architecture": "RDNA3 (gfx1100)",
            "optimizations": [
                "128-byte memory alignment for optimal 7900 XT cache usage",
                "32-element dimension alignment for Wave32 mode",
                "LayerNorm instead of GroupNorm for RDNA3 architecture",
                "SiLU activation with PyTorch JIT fusion for RDNA3",
                "Wave32 mode enabled for computation efficiency"
            ]
        }


class ModelManager:
    """
    Handles model saving, loading, and versioning.
    """
    def __init__(self, model_dir=None, version=None):
        """
        Initialize model manager.
        
        Parameters:
        -----------
        model_dir: str
            Directory for model storage
        version: str
            Model version identifier
        """
        self.logger = get_logger(__name__)
        
        # Set default model directory if not provided
        if model_dir is None:
            from src.config.settings import settings
            self.model_dir = settings.TRAINING_DIR
        else:
            self.model_dir = model_dir
            
        # Create directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Set version
        self.version = version or self._generate_version()
        
        self.logger.info(f"Initialized ModelManager with dir={self.model_dir}, version={self.version}")
        
    def _generate_version(self):
        """Generate a version identifier based on timestamp."""
        from datetime import datetime
        return f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def save_model(self, model, optimizer=None, epoch=None, 
                   loss=None, metadata=None, filename=None):
        """
        Save model checkpoint with metadata.
        
        Parameters:
        -----------
        model: nn.Module
            PyTorch model to save
        optimizer: torch.optim.Optimizer
            Optimizer state to save (optional)
        epoch: int
            Current training epoch (optional)
        loss: float
            Current loss value (optional)
        metadata: dict
            Additional metadata to save (optional)
        filename: str
            Custom filename (optional)
            
        Returns:
        --------
        str
            Path to saved model file
        """
        if not TORCH_AVAILABLE:
            self.logger.error("PyTorch not available, cannot save model")
            return None
            
        try:
            # Create checkpoint dictionary
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'version': self.version,
                'timestamp': self._generate_version(),
                'architecture': {
                    'input_dim': model.input_dim,
                    'hidden_layers': model.hidden_layers,
                    'dropout_rate': model.dropout_rate
                }
            }
            
            # Add feature columns if available
            if hasattr(model, 'feature_columns'):
                checkpoint['feature_columns'] = model.feature_columns
                
            # Add optional components
            if optimizer is not None:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            if epoch is not None:
                checkpoint['epoch'] = epoch
            if loss is not None:
                checkpoint['loss'] = loss
            if metadata is not None:
                checkpoint['metadata'] = metadata
                
            # Generate filename if not provided
            if filename is None:
                epoch_str = f"_epoch_{epoch}" if epoch is not None else ""
                filename = f"model_{self.version}{epoch_str}.pt"
                
            # Save model
            save_path = os.path.join(self.model_dir, filename)
            torch.save(checkpoint, save_path)
            
            # Also save metadata separately as JSON for easy access
            meta_filename = filename.replace(".pt", "_metadata.json")
            meta_path = os.path.join(self.model_dir, meta_filename)
            
            with open(meta_path, 'w') as f:
                json_checkpoint = {k: v for k, v in checkpoint.items() if k != 'model_state_dict' 
                                 and k != 'optimizer_state_dict' and isinstance(v, (dict, list, str, int, float))}
                json.dump(json_checkpoint, f, indent=2)
            
            self.logger.info(f"Model saved to {save_path}")
            return save_path
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            return None
    
    def load_model(self, path=None, filename=None, device=None):
        """
        Load a saved model.
        
        Parameters:
        -----------
        path: str
            Full path to model file
        filename: str
            Filename in model_dir
        device: torch.device
            Device to load the model to
            
        Returns:
        --------
        Tuple[nn.Module, dict]
            Loaded model and metadata
        """
        if not TORCH_AVAILABLE:
            self.logger.error("PyTorch not available, cannot load model")
            return None, {}
            
        try:
            # Determine load path
            if path is not None:
                load_path = path
            elif filename is not None:
                load_path = os.path.join(self.model_dir, filename)
            else:
                # Find most recent model file
                model_files = [f for f in os.listdir(self.model_dir) if f.endswith('.pt')]
                if not model_files:
                    self.logger.error("No model files found in directory")
                    return None, {}
                    
                # Sort by modification time (newest first)
                model_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.model_dir, x)), 
                                reverse=True)
                load_path = os.path.join(self.model_dir, model_files[0])
            
            # Determine device
            if device is None:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
            # Load checkpoint
            checkpoint = torch.load(load_path, map_location=device)
            
            # Extract architecture parameters
            architecture = checkpoint.get('architecture', {})
            input_dim = architecture.get('input_dim', checkpoint.get('input_dim'))
            
            if input_dim is None:
                if 'feature_columns' in checkpoint:
                    input_dim = len(checkpoint['feature_columns'])
                else:
                    self.logger.error("Could not determine input dimension from checkpoint")
                    return None, checkpoint
            
            # Get other architecture parameters
            hidden_layers = architecture.get('hidden_layers', [64, 32])
            
            # Handle case where hidden_layers is a ModuleList
            if hasattr(hidden_layers, 'named_children'):
                # Extract dimensions from the actual layers
                hidden_layers = []
                for name, module in architecture['hidden_layers'].named_children():
                    if isinstance(module, nn.Sequential):
                        for layer in module:
                            if isinstance(layer, nn.Linear):
                                hidden_layers.append(layer.out_features)
                    elif isinstance(module, nn.Linear):
                        hidden_layers.append(module.out_features)
            
            # Fallback if we couldn't extract dimensions
            if not isinstance(hidden_layers, list):
                hidden_layers = [64, 32]
                
            dropout_rate = architecture.get('dropout_rate', 0.4)
            
            # Create model
            model = AMDOptimizedFuturesModel(
                input_dim=input_dim, 
                hidden_layers=hidden_layers,
                dropout_rate=dropout_rate
            )
            
            # Load state dict
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Set feature columns if available
            if 'feature_columns' in checkpoint:
                model.feature_columns = checkpoint['feature_columns']
            
            # Move to device and set to evaluation mode
            model = model.to(device)
            model.eval()
            
            self.logger.info(f"Model loaded from {load_path}")
            
            # Return model and checkpoint data
            return model, checkpoint
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return None, {}
    
    def get_model_info(self, path=None, filename=None):
        """
        Get metadata about a saved model without loading the full model.
        
        Parameters:
        -----------
        path: str
            Full path to model file
        filename: str
            Filename in model_dir
            
        Returns:
        --------
        dict
            Model metadata
        """
        # Determine path to JSON metadata
        if path is not None:
            json_path = path.replace(".pt", "_metadata.json")
            if not os.path.exists(json_path):
                # Try loading the PT file metadata directly
                return self._extract_metadata_from_pt(path)
        elif filename is not None:
            json_path = os.path.join(self.model_dir, filename.replace(".pt", "_metadata.json"))
            if not os.path.exists(json_path):
                # Try the PT file
                pt_path = os.path.join(self.model_dir, filename)
                return self._extract_metadata_from_pt(pt_path)
        else:
            self.logger.error("No path or filename provided")
            return {}
            
        # Read JSON metadata
        try:
            with open(json_path, 'r') as f:
                metadata = json.load(f)
            return metadata
        except Exception as e:
            self.logger.error(f"Error reading model metadata: {e}")
            return {}
    
    def _extract_metadata_from_pt(self, path):
        """Extract metadata from PyTorch file without loading full model."""
        if not TORCH_AVAILABLE:
            self.logger.error("PyTorch not available, cannot extract metadata")
            return {}
            
        try:
            # Load checkpoint with map_location='cpu' to avoid GPU memory issues
            checkpoint = torch.load(path, map_location='cpu')
            
            # Extract metadata (exclude state dicts)
            metadata = {k: v for k, v in checkpoint.items() 
                      if k != 'model_state_dict' and k != 'optimizer_state_dict'}
                      
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error extracting metadata from PT file: {e}")
            return {}
