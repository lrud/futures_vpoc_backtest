#!/usr/bin/env python3
"""
Memory Management Integration for Training Scripts
Integrates GPU memory clearing functionality with existing training infrastructure.

Use this to add memory management to existing training scripts without major modifications.
"""

import os
import sys
import torch
import gc
from typing import Optional, Dict, Any

# Add workspace root to path for imports
sys.path.append('/workspace')
from src.ml.gpu_memory_clear import (
    aggressive_vram_cleanup,
    emergency_memory_cleanup,
    print_gpu_memory_stats,
    initialize_rocm_memory_environment,
    test_memory_allocation
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

class MemoryManager:
    """
    Memory management wrapper for training scripts.

    Use this class to add memory management to existing training code:

    ```python
    from src.ml.memory_integration import MemoryManager

    # Initialize at start of training script
    memory_manager = MemoryManager()

    # Before training
    memory_manager.pre_training_cleanup()

    # During training (every N batches or epochs)
    memory_manager.periodic_cleanup()

    # After training
    memory_manager.post_training_cleanup()
    ```
    """

    def __init__(self,
                 cleanup_interval_batches: int = 20,
                 cleanup_interval_epochs: int = 1,
                 emergency_cleanup_threshold: float = 0.95,
                 enable_auto_cleanup: bool = True):
        """
        Initialize memory manager.

        Args:
            cleanup_interval_batches: Cleanup every N batches
            cleanup_interval_epochs: Cleanup every N epochs
            emergency_cleanup_threshold: Trigger cleanup at this memory usage (0.0-1.0)
            enable_auto_cleanup: Enable automatic cleanup during training
        """
        self.cleanup_interval_batches = cleanup_interval_batches
        self.cleanup_interval_epochs = cleanup_interval_epochs
        self.emergency_cleanup_threshold = emergency_cleanup_threshold
        self.enable_auto_cleanup = enable_auto_cleanup

        # Training state tracking
        self.batch_count = 0
        self.epoch_count = 0
        self.last_cleanup_batch = 0
        self.last_cleanup_epoch = 0

        # Initialize ROCm environment
        initialize_rocm_memory_environment()

        logger.info("🔧 Memory Manager initialized")
        logger.info(f"  • Batch cleanup interval: {cleanup_interval_batches}")
        logger.info(f"  • Epoch cleanup interval: {cleanup_interval_epochs}")
        logger.info(f"  • Emergency cleanup threshold: {emergency_cleanup_threshold*100:.1f}%")
        logger.info(f"  • Auto cleanup enabled: {enable_auto_cleanup}")

    def pre_training_cleanup(self):
        """
        Perform cleanup before training starts.
        Call this before creating the model or data loaders.
        """
        logger.info("🧹 Pre-training memory cleanup...")

        # Show initial state
        print_gpu_memory_stats()

        # Perform aggressive cleanup
        success = aggressive_vram_cleanup()

        if success:
            logger.info("✅ Pre-training cleanup completed")
        else:
            logger.warning("⚠️  Pre-training cleanup had issues")

        # Show final state
        print_gpu_memory_stats()

        return success

    def post_training_cleanup(self):
        """
        Perform cleanup after training completes.
        Call this after saving the final model.
        """
        logger.info("🧹 Post-training memory cleanup...")

        # Show initial state
        print_gpu_memory_stats()

        # Perform aggressive cleanup
        success = aggressive_vram_cleanup()

        # Additional cleanup for training artifacts
        if torch.cuda.is_available():
            logger.debug("  Performing additional post-training cleanup...")

            # Clear all GPU caches
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Reset memory statistics
            for device_id in [0, 1]:
                if device_id < torch.cuda.device_count():
                    torch.cuda.set_device(device_id)
                    torch.cuda.reset_peak_memory_stats()

            # Force garbage collection
            gc.collect()

        if success:
            logger.info("✅ Post-training cleanup completed")
        else:
            logger.warning("⚠️  Post-training cleanup had issues")

        # Show final state
        print_gpu_memory_stats()

        return success

    def periodic_cleanup(self, force: bool = False):
        """
        Perform periodic cleanup during training.

        Args:
            force: Force cleanup regardless of interval settings
        """
        if not self.enable_auto_cleanup and not force:
            return

        self.batch_count += 1

        # Check if cleanup is needed
        cleanup_needed = force
        cleanup_reason = ""

        if not cleanup_needed and self.batch_count - self.last_cleanup_batch >= self.cleanup_interval_batches:
            cleanup_needed = True
            cleanup_reason = f"batch interval ({self.cleanup_interval_batches})"

        if not cleanup_needed and self._check_memory_usage():
            cleanup_needed = True
            cleanup_reason = "high memory usage"

        if cleanup_needed:
            logger.info(f"🧹 Periodic cleanup (triggered by {cleanup_reason})")

            success = aggressive_vram_cleanup()

            if success:
                self.last_cleanup_batch = self.batch_count
                logger.info("✅ Periodic cleanup completed")
            else:
                logger.warning("⚠️  Periodic cleanup had issues")

    def epoch_cleanup(self, epoch: int):
        """
        Perform cleanup at epoch boundaries.

        Args:
            epoch: Current epoch number
        """
        self.epoch_count = epoch

        if epoch - self.last_cleanup_epoch >= self.cleanup_interval_epochs:
            logger.info(f"🧹 Epoch {epoch} cleanup...")

            success = aggressive_vram_cleanup()

            if success:
                self.last_cleanup_epoch = epoch
                logger.info("✅ Epoch cleanup completed")
            else:
                logger.warning("⚠️  Epoch cleanup had issues")

    def emergency_cleanup(self):
        """
        Perform emergency cleanup when standard methods fail.
        """
        logger.warning("🚨 Emergency cleanup triggered!")

        success = emergency_memory_cleanup()

        if success:
            logger.info("✅ Emergency cleanup completed")
        else:
            logger.error("❌ Emergency cleanup failed")

        return success

    def _check_memory_usage(self) -> bool:
        """
        Check if memory usage exceeds emergency threshold.

        Returns:
            bool: True if cleanup is needed
        """
        if not torch.cuda.is_available():
            return False

        try:
            for device_id in [0, 1]:
                if device_id < torch.cuda.device_count():
                    torch.cuda.set_device(device_id)
                    free, total = torch.cuda.mem_get_info(device_id)
                    used = total - free
                    usage_percent = used / total

                    if usage_percent > self.emergency_cleanup_threshold:
                        logger.warning(f"⚠️  GPU {device_id} memory usage: {usage_percent*100:.1f}% (threshold: {self.emergency_cleanup_threshold*100:.1f}%)")
                        return True

            return False

        except Exception as e:
            logger.warning(f"Could not check memory usage: {e}")
            return False

    def memory_check(self) -> Dict[str, Any]:
        """
        Get current memory status.

        Returns:
            Dict with memory information
        """
        memory_info = {
            'gpu_available': torch.cuda.is_available(),
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'gpus': []
        }

        if torch.cuda.is_available():
            for device_id in range(min(2, torch.cuda.device_count())):
                try:
                    torch.cuda.set_device(device_id)
                    free, total = torch.cuda.mem_get_info(device_id)
                    used = total - free
                    usage_percent = (used / total) * 100

                    memory_info['gpus'].append({
                        'device_id': device_id,
                        'used_gb': used / 1024**3,
                        'total_gb': total / 1024**3,
                        'free_gb': free / 1024**3,
                        'usage_percent': usage_percent
                    })

                except Exception as e:
                    memory_info['gpus'].append({
                        'device_id': device_id,
                        'error': str(e)
                    })

        return memory_info

    def test_allocation(self, size_mb: int = 100) -> bool:
        """
        Test memory allocation capability.

        Args:
            size_mb: Size of test allocation in MB

        Returns:
            bool: True if allocation successful
        """
        return test_memory_allocation(size_mb)

def patch_training_script():
    """
    Instructions for patching existing training scripts with memory management.

    Add these lines to your training script:

    ```python
    # At the top of your main() function
    from src.ml.memory_integration import MemoryManager
    memory_manager = MemoryManager()

    # Before model creation and data loading
    memory_manager.pre_training_cleanup()

    # In your training loop (every batch or every N batches):
    memory_manager.periodic_cleanup()

    # At epoch boundaries:
    memory_manager.epoch_cleanup(epoch)

    # After training completes:
    memory_manager.post_training_cleanup()
    ```
    """

    instructions = """
    To integrate memory management into your training script:

    1. Add this import at the top of your script:
       from src.ml.memory_integration import MemoryManager

    2. Initialize the memory manager at the start of main():
       memory_manager = MemoryManager(
           cleanup_interval_batches=20,  # Cleanup every 20 batches
           cleanup_interval_epochs=1,    # Cleanup every epoch
           emergency_cleanup_threshold=0.95,  # 95% memory usage trigger
           enable_auto_cleanup=True
       )

    3. Add pre-training cleanup before creating model/data:
       memory_manager.pre_training_cleanup()

    4. Add periodic cleanup in your training loop:
       for batch_idx, (data, target) in enumerate(train_loader):
           # ... your training code ...
           memory_manager.periodic_cleanup()

    5. Add epoch cleanup:
       for epoch in range(epochs):
           # ... your epoch code ...
           memory_manager.epoch_cleanup(epoch)

    6. Add post-training cleanup:
       memory_manager.post_training_cleanup()
    """

    print(instructions)

def main():
    """Test the memory manager integration."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Memory Manager Integration")
    parser.add_argument("--test-size", type=int, default=500,
                       help="Test allocation size in MB")
    parser.add_argument("--pre-cleanup", action="store_true",
                       help="Test pre-training cleanup")
    parser.add_argument("--periodic", action="store_true",
                       help="Test periodic cleanup")

    args = parser.parse_args()

    print("=" * 60)
    print("Memory Manager Integration Test")
    print("=" * 60)

    # Initialize memory manager
    memory_manager = MemoryManager(
        cleanup_interval_batches=5,
        cleanup_interval_epochs=1,
        emergency_cleanup_threshold=0.90
    )

    # Test pre-training cleanup
    if args.pre_cleanup:
        print("\n1. Testing pre-training cleanup...")
        memory_manager.pre_training_cleanup()

    # Test periodic cleanup
    if args.periodic:
        print("\n2. Testing periodic cleanup...")
        for i in range(10):
            print(f"   Batch {i+1}/10")
            memory_manager.periodic_cleanup()

    # Test allocation
    print(f"\n3. Testing {args.test_size}MB allocation...")
    success = memory_manager.test_allocation(args.test_size)

    if success:
        print("✅ Memory manager integration test successful!")
    else:
        print("❌ Memory manager integration test failed!")
        print("Trying emergency cleanup...")
        memory_manager.emergency_cleanup()
        success = memory_manager.test_allocation(args.test_size)
        if success:
            print("✅ Emergency cleanup resolved the issue!")
        else:
            print("❌ Even emergency cleanup failed!")

    # Show final memory status
    print(f"\n4. Final memory status:")
    memory_info = memory_manager.memory_check()
    for gpu in memory_info['gpus']:
        if 'error' not in gpu:
            print(f"   GPU {gpu['device_id']}: {gpu['used_gb']:.2f}GB / {gpu['total_gb']:.2f}GB ({gpu['usage_percent']:.1f}%)")

if __name__ == "__main__":
    main()