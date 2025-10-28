"""
Distributed training interface module.
This module provides the main interface classes for distributed training.
"""

from .distributed_trainer import AMDFuturesTensorParallel, train_distributed

__all__ = ['AMDFuturesTensorParallel', 'train_distributed']