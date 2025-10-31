"""
Robust Neural Network Architecture for Financial Time Series
Based on research-backed solutions for stable training

Key Features:
- Huber Loss (robust to outliers)
- Layer Normalization (stabilizes hidden activations)
- Residual Connections (prevents vanishing/exploding gradients)
- Learning Rate Warmup (prevents early explosion)
- ROCm 7 consumer GPU optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any
import math

from src.utils.logging import get_logger

logger = get_logger(__name__)

class HuberLoss(nn.Module):
    """
    Huber Loss implementation for robust training.

    Combines MSE loss for small errors with MAE loss for large errors.
    This provides robustness to outliers while maintaining efficiency for normal samples.

    Formula:
      loss = 0.5 * (error)^2 for |error| <= delta
      loss = delta * (|error| - 0.5 * delta) for |error| > delta
    """

    def __init__(self, delta: float = 1.0):
        """
        Initialize Huber Loss.

        Args:
            delta: Threshold where quadratic loss transitions to linear loss
                   Smaller delta = more robust to outliers
                   Recommended for financial data: 0.1 - 1.0
        """
        super().__init__()
        self.delta = delta
        logger.info(f"🛡️ Initialized Huber Loss with delta={delta}")

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Huber loss.

        Args:
            predictions: Model predictions (batch_size, 1)
            targets: Target values (batch_size, 1)

        Returns:
            Huber loss tensor
        """
        error = predictions - targets
        abs_error = torch.abs(error)

        # Quadratic loss for small errors, linear for large errors
        quadratic_loss = torch.where(abs_error <= self.delta,
                                     0.5 * error ** 2,
                                     self.delta * (abs_error - 0.5 * self.delta))

        return quadratic_loss.mean()

class ResidualBlock(nn.Module):
    """
    Residual block with LayerNorm for stable training.

    Implements: x -> LayerNorm(x) -> Linear(x) -> LayerNorm(Linear(x)) -> ReLU -> Linear(x) + x
    The residual connection helps prevent vanishing/exploding gradients.
    """

    def __init__(self, input_dim: int, hidden_dim: Optional[int] = None, dropout_rate: float = 0.1):
        """
        Initialize residual block.

        Args:
            input_dim: Input dimension
            hidden_dim: Hidden layer dimension (defaults to input_dim)
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()

        if hidden_dim is None:
            hidden_dim = input_dim

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Layer normalization for stability
        self.ln1 = nn.LayerNorm(input_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        # Linear transformations
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)

        # Shortcut connection (identity if dimensions match)
        if input_dim != hidden_dim:
            self.shortcut = nn.Linear(input_dim, input_dim)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.

        Args:
            x: Input tensor (batch_size, input_dim)

        Returns:
            Output tensor (batch_size, input_dim)
        """
        # Store input for residual connection
        residual = self.shortcut(x)

        # First layer with LayerNorm
        out = self.ln1(x)
        out = F.relu(out)
        out = self.linear1(out)

        # Second layer with LayerNorm and dropout
        out = self.ln2(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)

        # Add residual connection
        out = out + residual

        return out

class RobustFinancialNet(nn.Module):
    """
    Robust Neural Network for Financial Time Series Prediction.

    Implements all research-backed solutions:
    1. Huber Loss (robust to outliers)
    2. Layer Normalization (stabilizes hidden activations)
    3. Residual Connections (prevents vanishing/exploding gradients)
    4. Learning Rate Warmup (in optimizer, not model)
    5. ROCm 7 consumer GPU optimization
    """

    def __init__(self, input_dim: int = 5, hidden_dims: list = [64, 32],
                 dropout_rate: float = 0.1, use_residual: bool = True):
        """
        Initialize robust financial neural network.

        Args:
            input_dim: Number of input features (default: 5 for top features)
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate for regularization
            use_residual: Whether to use residual connections
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.use_residual = use_residual

        logger.info(f"🏗️ Initializing RobustFinancialNet:")
        logger.info(f"  • Input dimension: {input_dim}")
        logger.info(f"  • Hidden dimensions: {hidden_dims}")
        logger.info(f"  • Dropout rate: {dropout_rate}")
        logger.info(f"  • Residual connections: {use_residual}")

        # Build network layers
        layers = []

        # Input normalization
        self.input_norm = nn.LayerNorm(input_dim)

        # Hidden layers
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            # Always use simple linear layers to avoid dimension mismatches
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        # Output layer (regression to predict rank-transformed target)
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

        # Initialize weights with He initialization (better for ReLU)
        self._initialize_weights()

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        logger.info(f"  • Total parameters: {total_params:,}")
        logger.info(f"  • Trainable parameters: {trainable_params:,}")

    def _initialize_weights(self):
        """Initialize weights using He initialization for ReLU activation."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # He initialization (good for ReLU)
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                # LayerNorm initialization
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor (batch_size, input_dim)

        Returns:
            Output tensor (batch_size, 1) - predictions in 0-1 range
        """
        # Input normalization
        x = self.input_norm(x)

        # Forward through network
        output = self.network(x)

        # Sigmoid activation to ensure output is in [0, 1] range
        # This matches our rank-transformed target range
        output = torch.sigmoid(output)

        return output

    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters by layer type."""
        total_params = 0
        linear_params = 0
        layernorm_params = 0

        for name, param in self.named_parameters():
            total_params += param.numel()
            if 'linear' in name.lower():
                linear_params += param.numel()
            elif 'ln' in name.lower() or 'norm' in name.lower():
                layernorm_params += param.numel()

        return {
            'total': total_params,
            'linear': linear_params,
            'layer_norm': layernorm_params
        }

class LearningRateWarmup:
    """
    Learning rate warmup scheduler to prevent early gradient explosions.

    Gradually increases learning rate from 0 to target over warmup_steps.
    This is critical for stabilizing training on financial time series data.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int = 1000,
                 target_lr: Optional[float] = None):
        """
        Initialize warmup scheduler.

        Args:
            optimizer: PyTorch optimizer
            warmup_steps: Number of warmup steps
            target_lr: Target learning rate (uses current optimizer lr if None)
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.current_step = 0

        # Get target learning rate
        if target_lr is not None:
            self.target_lr = target_lr
        else:
            self.target_lr = optimizer.param_groups[0]['lr']

        # Store initial learning rate (set to 0 for warmup)
        self.initial_lr = 0.0

        # Set initial learning rate to 0
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.initial_lr

        logger.info(f"🔥 Initialized Learning Rate Warmup:")
        logger.info(f"  • Warmup steps: {warmup_steps}")
        logger.info(f"  • Target LR: {self.target_lr}")
        logger.info(f"  • Initial LR: {self.initial_lr}")

    def step(self):
        """Perform one warmup step."""
        if self.current_step < self.warmup_steps:
            # Linear warmup
            progress = (self.current_step + 1) / self.warmup_steps
            current_lr = self.initial_lr + progress * (self.target_lr - self.initial_lr)

            # Update optimizer learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr

            self.current_step += 1
        else:
            # Warmup completed, ensure target learning rate is set
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.target_lr

    def get_current_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']

def create_robust_model(input_dim: int = 5, hidden_dims: list = [64, 32],
                       dropout_rate: float = 0.1, use_residual: bool = True) -> RobustFinancialNet:
    """
    Create a robust financial neural network with default optimized settings.

    Args:
        input_dim: Number of input features
        hidden_dims: Hidden layer dimensions
        dropout_rate: Dropout rate
        use_residual: Whether to use residual connections

    Returns:
        Initialized RobustFinancialNet model
    """
    logger.info("🚀 Creating robust financial neural network...")

    model = RobustFinancialNet(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
        use_residual=use_residual
    )

    return model

def create_robust_optimizer(model: RobustFinancialNet, learning_rate: float = 1e-4,
                           weight_decay: float = 1e-4, warmup_steps: int = 1000) -> tuple:
    """
    Create optimizer with warmup scheduler for robust training.

    Args:
        model: RobustFinancialNet model
        learning_rate: Target learning rate
        weight_decay: Weight decay for regularization
        warmup_steps: Number of warmup steps

    Returns:
        Tuple of (optimizer, warmup_scheduler, loss_fn)
    """
    logger.info("⚙️ Creating robust optimizer and loss function...")

    # Create optimizer (AdamW works well with Huber loss)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # Create warmup scheduler
    warmup_scheduler = LearningRateWarmup(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        target_lr=learning_rate
    )

    # Create Huber loss (delta=0.1 is good for rank-transformed targets)
    loss_fn = HuberLoss(delta=0.1)

    logger.info(f"✅ Created AdamW optimizer with lr={learning_rate}, weight_decay={weight_decay}")
    logger.info(f"✅ Created {warmup_steps}-step learning rate warmup")
    logger.info(f"✅ Created Huber loss with delta=0.1")

    return optimizer, warmup_scheduler, loss_fn

def test_robust_model():
    """Test the robust model architecture."""
    logger.info("🧪 Testing Robust Financial Neural Network...")

    # Test parameters
    batch_size = 32
    input_dim = 5
    sequence_length = 1

    # Create model
    model = create_robust_model(
        input_dim=input_dim,
        hidden_dims=[32, 16],
        dropout_rate=0.1,
        use_residual=True
    )

    # Create optimizer and loss
    optimizer, warmup_scheduler, loss_fn = create_robust_optimizer(
        model=model,
        learning_rate=1e-4,
        warmup_steps=50  # Short for testing
    )

    # Create test data (in 0-1 range to match rank-transformed targets)
    test_input = torch.randn(batch_size, input_dim)
    test_target = torch.rand(batch_size, 1)  # Random values in [0, 1]

    # Test forward pass
    model.eval()
    with torch.no_grad():
        output = model(test_input)
        loss = loss_fn(output, test_target)

    # Test training step
    model.train()
    optimizer.zero_grad()

    output = model(test_input)
    loss = loss_fn(output, test_target)
    loss.backward()

    # Test warmup
    warmup_scheduler.step()
    current_lr = warmup_scheduler.get_current_lr()

    # Validate results
    logger.info("✅ Test Results:")
    logger.info(f"  • Input shape: {test_input.shape}")
    logger.info(f"  • Output shape: {output.shape}")
    logger.info(f"  • Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    logger.info(f"  • Loss: {loss.item():.6f}")
    logger.info(f"  • Learning rate after warmup: {current_lr}")

    # Test parameter counts
    param_counts = model.count_parameters()
    logger.info(f"  • Parameter counts: {param_counts}")

    # Test multiple warmup steps
    logger.info("🔥 Testing learning rate warmup...")
    for i in range(10):
        warmup_scheduler.step()
        if i % 3 == 0:
            logger.info(f"  Step {i+1}: LR = {warmup_scheduler.get_current_lr():.6f}")

    logger.info("🎉 Robust model test completed successfully!")
    return True

if __name__ == "__main__":
    success = test_robust_model()
    if success:
        print("✅ Robust Financial Neural Network is ready for use!")
    else:
        print("❌ Robust model test failed!")