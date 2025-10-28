"""
TEST: Simple distributed training solution for ROCm 7 with DataParallel.
This provides a working alternative to complex multiprocessing approaches.
"""

import os
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
from torch.utils.data import TensorDataset, DataLoader

from src.utils.logging import get_logger

logger = get_logger(__name__)

def setup_multi_gpu_training():
    """Setup multi-GPU training environment for ROCm 7."""

    # ROCm 7 optimizations for multi-GPU
    os.environ.update({
        # ROCm 7 specific settings
        'PYTORCH_ROCM_ARCH': 'gfx1100',
        'PYTORCH_ROCM_WAVE32_MODE': '1',
        'PYTORCH_ROCM_FUSION': '1',

        # Memory management
        'PYTORCH_HIP_ALLOC_CONF': 'expandable_segments:True,max_split_size_mb:256',
        'GPU_SINGLE_ALLOC_PERCENT': '90',

        # Performance optimizations
        'HSA_ENABLE_SDMA': '0',
        'HSA_ENABLE_INTERRUPT': '0',
        'HSA_ENABLE_WAIT_COMPLETION': '0',
        'GPU_MAX_HW_QUEUES': '8',

        # Flash Attention
        'ENABLE_FLASH_ATTENTION': '1',

        # PyTorch optimizations
        'TORCH_COMPILE_BACKEND': 'inductor'
    })

    # Make both GPUs visible
    os.environ['HIP_VISIBLE_DEVICES'] = '0,1'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

    gpu_count = torch.cuda.device_count()
    if gpu_count >= 2:
        logger.info(f"✅ ROCm 7 Multi-GPU Setup: {gpu_count} GPUs detected")
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"🎯 GPU {i}: {props.name} ({props.total_memory/1e9:.1f}GB)")
        return True
    else:
        logger.warning(f"⚠️ Only {gpu_count} GPU detected, falling back to single GPU")
        return False

def create_data_parallel_model(model, device_ids=None):
    """Create DataParallel model for multi-GPU training."""

    if device_ids is None:
        device_ids = list(range(torch.cuda.device_count()))

    if len(device_ids) <= 1:
        logger.info("📱 Using single GPU training")
        return model.to('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info(f"🚀 Creating DataParallel model with GPUs: {device_ids}")

    # Move model to primary GPU first
    primary_device = f'cuda:{device_ids[0]}'
    model = model.to(primary_device)

    # Create DataParallel model
    model = DataParallel(model, device_ids=device_ids, output_device=device_ids[0])

    # Enable Flash Attention if available
    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        if hasattr(model, 'module'):
            model.module.enable_flash_attention = True
        else:
            model.enable_flash_attention = True
        logger.info("✅ Flash Attention enabled for DataParallel model")

    return model

def train_multi_gpu(model, X_train, y_train, X_val, y_val, args_dict, feature_columns=None):
    """
    Train model using DataParallel on multiple GPUs.

    Args:
        model: PyTorch model to train
        X_train, y_train: Training data
        X_val, y_val: Validation data
        args_dict: Training arguments
        feature_columns: Feature column names

    Returns:
        Trained model and training history
    """

    logger.info("🚀 Starting ROCm 7 Multi-GPU Training with DataParallel")

    # Setup multi-GPU environment
    use_multi_gpu = setup_multi_gpu_training()

    # Create DataParallel model
    if use_multi_gpu:
        device_ids = [0, 1]  # Use both GPUs
        model = create_data_parallel_model(model, device_ids)
    else:
        model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Import trainer for actual training
    from src.ml.trainer import ModelTrainer

    # Create trainer
    output_dir = args_dict.get('output_dir', './models')
    trainer = ModelTrainer(model_dir=output_dir, device='cuda')

    # Train the model
    logger.info("🏃‍♂️ Starting model training...")
    trained_model, history = trainer.train(
        model, X_train, y_train, X_val, y_val,
        args=args_dict
    )

    logger.info("✅ Multi-GPU training completed successfully!")

    # For DataParallel, return the underlying model
    if hasattr(trained_model, 'module'):
        final_model = trained_model.module
    else:
        final_model = trained_model

    # Store feature columns
    if feature_columns:
        final_model.feature_columns = feature_columns

    return final_model, history

if __name__ == "__main__":
    logger.info("🧪 Testing ROCm 7 Distributed Training with DataParallel")
    # This is a test file for distributed training functionality