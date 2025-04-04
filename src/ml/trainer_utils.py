"""
Utility functions for model training.
"""

import os
import sys
import numpy as np
import torch
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union

from src.core.data import FuturesDataManager
from src.ml.feature_engineering import FeatureEngineer
from src.utils.logging import get_logger
from src.config.settings import settings

# Initialize logger
logger = get_logger(__name__)

def get_base_parser(description=None):
    """Returns ArgumentParser with common ML arguments shared across scripts."""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("-o", "--output_dir", default=settings.TRAINING_DIR, help="Output directory")
    parser.add_argument("-d", "--data_path", default=settings.DATA_DIR, help="Input data directory") 
    return parser

def set_random_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Additional settings for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_gpu_metrics(device_ids=None):
    """Get memory and utilization metrics for specified GPUs.
    
    Args:
        device_ids: List of GPU IDs to check (None for all)
        
    Returns:
        Dict of {gpu_id: {'memory': (used, total), 'utilization': percent}}
    """
    metrics = {}
    device_ids = device_ids or range(torch.cuda.device_count())
    
    for i in device_ids:
        mem = torch.cuda.memory_stats(i)
        metrics[i] = {
            'memory': (
                mem['allocated_bytes.all.current'] / 1024**2,
                torch.cuda.get_device_properties(i).total_memory / 1024**2
            ),
            'utilization': torch.cuda.utilization(i)
        }
    return metrics

def log_gpu_metrics(metrics, logger=None):
    """Log GPU metrics in human-readable format.
    
    Args:
        metrics: Dict from get_gpu_metrics()
        logger: Logger instance (defaults to module logger)
    """
    logger = logger or globals().get('logger')
    for gpu_id, data in metrics.items():
        used, total = data['memory']
        logger.info(
            f"GPU {gpu_id}: {used:.1f}/{total:.1f} MB ({used/total*100:.1f}%) "
            f"utilization: {data['utilization']}%"
        )

def setup_rocm_environment():
    """Set up ROCm environment for optimal performance with dual 7900 XT GPUs."""
    if not torch.cuda.is_available():
        logger.warning("ROCm setup called but no AMD GPUs detected!")
        return False

    # Verify ROCm version and GPU count
    gpu_count = torch.cuda.device_count()
    logger.info(f"Detected {gpu_count} AMD GPU(s)")
    
    # Multi-GPU specific optimizations
    os.environ['HSA_ENABLE_SDMA'] = '0'
    os.environ['GPU_MAX_HW_QUEUES'] = '8'
    os.environ['HSA_MAX_QUEUES'] = '36'
    os.environ['HIP_HIDDEN_FREE_MEM'] = '256'
    
    # Enhanced PyTorch ROCm optimizations
    os.environ['PYTORCH_ROCM_FUSION'] = '1'
    os.environ['PYTORCH_JIT_USE_NNC_NOT_NVFUSER'] = '1'
    os.environ['PYTORCH_ROCM_WAVE32_MODE'] = '1'
    os.environ['GPU_SINGLE_ALLOC_PERCENT'] = '90'  # Lower for multi-GPU
    
    # Multi-GPU communication optimizations
    os.environ['HSA_ENABLE_INTERRUPT'] = '0'
    os.environ['HSA_ENABLE_WAIT_COMPLETION'] = '0'
    os.environ['NCCL_DEBUG'] = 'WARN'
    os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
    os.environ['NCCL_NSOCKS_PERTHREAD'] = '8'
    
    # Memory and thread optimization
    os.environ['OMP_NUM_THREADS'] = str(os.cpu_count() // 4)  # More conservative
    os.environ['ROCM_LAZY_MEM_ALLOC'] = '1'
    
    # Cache optimization
    cache_dir = os.path.join(os.path.expanduser("~"), ".rocm_shader_cache")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ['ROCM_SHADER_CACHE_PATH'] = cache_dir
    
    # Verify ROCm devices
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        logger.info(f"GPU {i}: {props.name}, {props.total_memory/1024**3:.2f}GB VRAM")
    
    logger.info(f"ROCm environment optimized for {gpu_count} 7900 XT GPUs")
    return True

def prepare_data(args, contract_filter=None, data_path=None, device_ids=None):
    """
    Load and prepare data for training using FeatureEngineer.
    
    Parameters:
    -----------
    args: argparse.Namespace
        Command line arguments
    contract_filter: Optional[str]
        Optional contract filter (ES, VIX, or None for all)
    data_path: Optional[str]
        Explicit data path to use (overrides args.data_path)
    device_ids: Optional[List[int]]
        List of GPU device IDs for distributed data loading
        
    Returns:
    --------
    tuple or None
        (X_train, y_train, X_val, y_val, feature_columns) or None if failed
    """
    feature_engineer = FeatureEngineer()
    try:
        # Use explicit data_path if provided, otherwise try args.data_path, fall back to settings.DATA_DIR
        path = data_path if data_path else (
            str(args.data_path) if hasattr(args, 'data_path') else str(settings.DATA_DIR)
        )
        return feature_engineer.load_and_prepare_data(
            data_path=path,
            contract_filter=contract_filter
        )
    except Exception as e:
        logger.error(f"Data preparation failed: {e}")
        return None

def preprocess_contract_data(args, contract_types=None):
    """
    Process data for multiple contract types.
    
    Parameters:
    -----------
    args: argparse.Namespace
        Command line arguments
    contract_types: List[str]
        List of contract types to process
        
    Returns:
    --------
    Dict or None
        Dictionary with data for each contract type or None if failed
    """
    if contract_types is None:
        contract_types = ["ES", "VIX"]
    
    results = {}
    
    for contract in contract_types:
        logger.info(f"Processing data for {contract}")
        result = prepare_data(args, contract_filter=contract)
        
        if result is None:
            logger.error(f"Failed to process {contract} data")
            continue
            
        results[contract] = result
    
    if not results:
        logger.error("No data processed successfully")
        return None
        
    return results
