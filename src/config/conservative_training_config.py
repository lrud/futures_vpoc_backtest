"""
Conservative training configuration for stable ML model training.
Addresses gradient explosions, nan/inf losses, and convergence issues.
"""

from typing import Dict, Any

# CONSERVATIVE TRAINING CONFIGURATION
# Designed to prevent training instabilities while maintaining performance

CONSERVATIVE_TRAINING_CONFIG: Dict[str, Any] = {
    # Model Architecture (Conservative)
    "model": {
        "hidden_layers": [64, 32],          # Smaller network to prevent overfitting
        "dropout_rate": 0.3,                 # Moderate dropout
        "use_residual": True,                # Enable for gradient flow
        "use_attention": True,               # Enable attention with low weight
        "attention_weight": 0.1,             # Conservative attention contribution
        "residual_weight": 0.1,              # Conservative residual contribution
    },

    # Training Parameters (Conservative)
    "training": {
        "batch_size": 16,                    # Smaller batches for stability
        "epochs": 100,                       # More epochs with smaller learning rate
        "learning_rate": 0.0005,             # Half the default learning rate
        "weight_decay": 0.02,                # Higher regularization
        "gradient_accumulation_steps": 2,    # Accumulate gradients for effective larger batch
        "gradient_clip_value": 0.5,          # More conservative clipping
        "gradient_clip_type": "adaptive",    # Adaptive clipping based on statistics
    },

    # Learning Rate Scheduling
    "scheduler": {
        "use_cosine_schedule": True,         # Cosine annealing with warmup
        "warmup_steps": 200,                 # Gradual warmup
        "warmup_epochs": 10,                 # Warmup for first 10 epochs
        "min_lr": 1e-6,                      # Minimum learning rate
        "reduce_lr_factor": 0.7,             # Conservative reduction factor
        "reduce_lr_patience": 15,            # More patience before reduction
    },

    # Noise Injection for Regularization
    "noise_injection": {
        "enabled": True,
        "noise_type": "gaussian",             # Gaussian noise for inputs
        "noise_std": 0.005,                  # Very small noise (0.5% std)
        "noise_probability": 0.05,           # 5% chance of noise injection
        "weight_noise_enabled": True,        # Small weight noise for regularization
        "weight_noise_std": 0.0001,          # Tiny weight noise
    },

    # Advanced Optimizer Settings
    "optimizer": {
        "type": "adamw",                     # AdamW with decoupled weight decay
        "betas": (0.9, 0.999),               # Default betas for stability
        "eps": 1e-8,                         # Small epsilon for numerical stability
        "amsgrad": False,                    # Disabled for memory efficiency
    },

    # Mixed Precision Training
    "mixed_precision": {
        "enabled": False,                     # Disabled initially for stability
        "dtype": "bfloat16",                 # BF16 if enabled later
        "loss_scaling": "dynamic",           # Dynamic loss scaling
    },

    # Early Stopping
    "early_stopping": {
        "enabled": True,
        "patience": 25,                      # More patience (25 epochs)
        "min_delta": 1e-6,                  # Small minimum improvement
        "restore_best_weights": True,        # Restore best model weights
        "monitor": "val_loss",               # Monitor validation loss
    },

    # Feature Engineering (Conservative)
    "features": {
        "scaling_method": "robust",          # Robust scaling to handle outliers
        "target_transformation": "vol_adjusted",  # Volatility-adjusted targets
        "feature_selection": True,           # Enable feature selection
        "max_features": 20,                  # Limit features to prevent overfitting
        "outlier_clipping": True,            # Clip extreme feature values
        "outlier_threshold": 8.0,            # Clip at ±8 standard deviations
    },

    # Data Handling
    "data": {
        "train_val_split": 0.8,              # 80/20 split
        "shuffle_training": True,            # Shuffle training data
        "data_fraction": 1.0,                # Use all data initially
        "remove_outliers": True,             # Remove extreme outliers
        "outlier_percentile": 0.99,          # Remove top 1% outliers
    },

    # Monitoring and Debugging
    "monitoring": {
        "log_frequency": 20,                 # Log every 20 batches
        "gradient_monitoring": True,         # Monitor gradient norms
        "loss_monitoring": True,             # Monitor for explosion/nan
        "memory_monitoring": True,           # Monitor GPU memory usage
        "health_checks": True,               # Enable training health checks
    },

    # Memory Management (ROCm 7)
    "memory": {
        "gradient_checkpointing": False,     # Disabled initially for speed
        "cache_clearing_frequency": 10,      # Clear cache every 10 batches
        "memory_limit_gb": 14,               # Limit memory usage to 14GB
        "enable_memory_efficient": True,     # Enable memory optimizations
    },

    # Validation and Testing
    "validation": {
        "frequency": 1,                      # Validate every epoch
        "early_exit_on_nan": True,           # Stop if NaN/inf detected
        "check_convergence": True,           # Check for convergence
        "convergence_threshold": 1e-6,       # Convergence threshold
    },

    # Recovery and Checkpointing
    "checkpointing": {
        "enabled": True,
        "frequency": 10,                     # Save every 10 epochs
        "save_best_only": True,              # Only save best models
        "keep_last_n": 3,                    # Keep last 3 checkpoints
        "resume_on_failure": True,           # Resume from checkpoint on failure
    },

    # ROCm-specific Optimizations
    "rocm_optimizations": {
        "enable_torch_compile": False,       # Disabled due to instability
        "memory_alignment": True,            # Enable memory alignment
        "use_layer_norm": True,              # LayerNorm instead of BatchNorm
        "mixed_precision_backend": "native",  # Native mixed precision
    }
}

# CONSERVATIVE FEATURE ENGINEERING CONFIG
CONSERVATIVE_FEATURE_CONFIG = {
    "target_transformations": [
        "vol_adjusted",     # Volatility-adjusted returns (most stable)
        "rank",            # Rank-based transformation
        "quantile",        # Quantile transformation
        "log"              # Log transformation (fallback)
    ],

    "scaling_methods": [
        "robust",          # Robust scaling (median/MAD)
        "quantile",        # Quantile transformation
        "power"            # Power transformation (Yeo-Johnson)
    ],

    "advanced_features": {
        "garch": True,                     # Enhanced GARCH features
        "wavelet": False,                  # Disabled initially (complexity)
        "cointegration": True,             # Mean reversion features
        "max_lookback_periods": [5, 10, 20],  # Conservative lookback periods
    },

    "feature_selection": {
        "method": "select_k_best",         # SelectKBest for simplicity
        "k": 15,                          # Limit to 15 best features
        "score_func": "f_regression",     # F-regression for feature selection
    }
}

# TRAINING STABILITY CONFIG
STABILITY_CONFIG = {
    "gradient_checks": {
        "max_grad_norm": 2.0,             # Maximum allowed gradient norm
        "min_grad_norm": 1e-7,            # Minimum gradient norm (vanishing check)
        "check_frequency": 10,            # Check every 10 batches
        "clip_on_explosion": True,        # Auto-clip on explosion
    },

    "loss_checks": {
        "max_loss": 100.0,                # Maximum allowed loss
        "min_loss": 1e-8,                 # Minimum allowed loss
        "check_nan_inf": True,            # Check for NaN/inf values
        "restart_on_nan": True,           # Restart from checkpoint on NaN
    },

    "convergence_checks": {
        "min_improvement": 1e-6,          # Minimum improvement threshold
        "patience_epochs": 20,            # Patience for convergence
        "early_stop_patience": 30,        # Early stopping patience
    }
}

# SUCCESS METRICS CONFIG
SUCCESS_METRICS = {
    "training_stability": {
        "no_gradient_explosions": True,   # No gradient explosions
        "no_nan_inf_losses": True,        # No NaN/inf losses
        "stable_convergence": True,        # Stable and converging training
        "reasonable_loss_values": True,    # Loss values in reasonable range
    },

    "performance_targets": {
        "min_val_accuracy": 55.0,         # Minimum validation accuracy (%)
        "max_val_loss": 0.7,              # Maximum validation loss
        "min_improvement_rate": 0.01,     # Minimum improvement rate per epoch
        "training_efficiency": 0.8,       # Training efficiency metric
    },

    "robustness_metrics": {
        "consistency_across_runs": True,   # Consistent performance across runs
        "low_variance": True,              # Low variance in predictions
        "stable_predictions": True,        # Stable prediction distributions
    }
}


def get_conservative_config() -> Dict[str, Any]:
    """
    Get the complete conservative training configuration.

    Returns:
        Dictionary containing all conservative training parameters
    """
    config = CONSERVATIVE_TRAINING_CONFIG.copy()
    config.update({
        "feature_engineering": CONSERVATIVE_FEATURE_CONFIG,
        "stability_checks": STABILITY_CONFIG,
        "success_metrics": SUCCESS_METRICS
    })
    return config


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate that the configuration is conservative enough for stable training.

    Args:
        config: Configuration dictionary to validate

    Returns:
        True if configuration is valid, False otherwise
    """
    # Check learning rate is conservative
    if config.get("training", {}).get("learning_rate", 0) > 0.001:
        return False

    # Check batch size is reasonable
    if config.get("training", {}).get("batch_size", 0) > 32:
        return False

    # Check gradient clipping is enabled
    if config.get("training", {}).get("gradient_clip_value", 0) <= 0:
        return False

    # Check early stopping is enabled
    if not config.get("early_stopping", {}).get("enabled", False):
        return False

    return True


if __name__ == "__main__":
    # Example usage
    config = get_conservative_config()
    print("Conservative Training Configuration:")
    print(f"Learning Rate: {config['training']['learning_rate']}")
    print(f"Batch Size: {config['training']['batch_size']}")
    print(f"Gradient Clip Value: {config['training']['gradient_clip_value']}")
    print(f"Configuration Valid: {validate_config(config)}")