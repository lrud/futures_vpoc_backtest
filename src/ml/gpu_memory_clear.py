#!/usr/bin/env python3
"""
GPU Memory Clearing Utility for ROCm 6.3 on AMD RX 7900 XT
Addresses severe VRAM fragmentation issues on dual GPU setup.

Based on primary solution from GPU_Memory_Training_Solutions.md
"""

import os
import sys
import gc
import torch
import time
from typing import List, Optional

# Add workspace root to path for imports
sys.path.append('/workspace')
from src.utils.logging import get_logger

# Initialize logger
logger = get_logger(__name__)

# CRITICAL: Set environment variables BEFORE importing torch for ROCm memory management
os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'max_split_size_mb:256,garbage_collection_threshold:0.95'

def print_gpu_memory_stats():
    """Print current GPU memory usage for monitoring."""
    if not torch.cuda.is_available():
        logger.warning("CUDA not available - cannot get GPU memory stats")
        return

    logger.info("🔍 Current GPU Memory Status:")
    for device_id in [0, 1]:
        if device_id < torch.cuda.device_count():
            try:
                torch.cuda.set_device(device_id)
                free, total = torch.cuda.mem_get_info(device_id)
                used = total - free
                percent = (used / total) * 100
                logger.info(f"  GPU {device_id}: {used/1024**3:.2f}GB / {total/1024**3:.2f}GB ({percent:.1f}%)")
            except Exception as e:
                logger.warning(f"  GPU {device_id}: Could not get memory info - {e}")

def aggressive_vram_cleanup():
    """
    Defragment GPU memory WITHOUT killing the process.
    This is the primary solution from GPU_Memory_Training_Solutions.md

    Call this to clear VRAM fragmentation on ROCm 6.3 with RX 7900 XT GPUs.
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available - cannot perform VRAM cleanup")
        return False

    logger.info("🔧 Starting aggressive VRAM cleanup...")
    start_time = time.time()

    try:
        # Step 1: Synchronize all GPU operations
        logger.debug("  Synchronizing GPU operations...")
        for device_id in [0, 1]:
            if device_id < torch.cuda.device_count():
                torch.cuda.set_device(device_id)
                torch.cuda.synchronize()

        # Step 2: Clear PyTorch GPU cache
        logger.debug("  Clearing PyTorch GPU cache...")
        torch.cuda.empty_cache()

        # Step 3: Force Python garbage collection (3 levels)
        logger.debug("  Running Python garbage collection...")
        gc.collect(0)  # Collect young objects
        gc.collect(1)  # Collect intermediate objects
        gc.collect(2)  # Collect old objects

        # Step 4: Clear GPU cache again after GC
        logger.debug("  Clearing cache after garbage collection...")
        torch.cuda.empty_cache()

        # Step 5: Reset memory statistics (for monitoring)
        logger.debug("  Resetting memory statistics...")
        for device_id in [0, 1]:
            if device_id < torch.cuda.device_count():
                torch.cuda.set_device(device_id)
                torch.cuda.reset_accumulated_memory_stats()

        # Step 6: Final synchronization
        logger.debug("  Final synchronization...")
        for device_id in [0, 1]:
            if device_id < torch.cuda.device_count():
                torch.cuda.set_device(device_id)
                torch.cuda.synchronize()

        cleanup_time = time.time() - start_time
        logger.info(f"✅ VRAM cleanup completed in {cleanup_time:.2f} seconds")
        return True

    except Exception as e:
        logger.error(f"❌ VRAM cleanup failed: {e}")
        return False

def emergency_memory_cleanup():
    """
    Emergency cleanup for when GPU is completely fragmented.
    More aggressive version with additional steps.
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available - cannot perform emergency cleanup")
        return False

    logger.warning("🚨 Starting EMERGENCY memory cleanup...")

    try:
        # Set aggressive memory management
        os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'max_split_size_mb:128,garbage_collection_threshold:0.85'

        # Perform standard cleanup first
        aggressive_vram_cleanup()

        # Additional aggressive steps
        logger.debug("  Performing additional emergency cleanup steps...")

        # Clear any remaining cached memory
        for device_id in [0, 1]:
            if device_id < torch.cuda.device_count():
                torch.cuda.set_device(device_id)

                # Try to reset peak memory stats
                try:
                    torch.cuda.reset_peak_memory_stats()
                except:
                    pass

                # Additional synchronization
                try:
                    torch.cuda.synchronize()
                except:
                    pass

        # Force multiple rounds of garbage collection
        for _ in range(3):
            gc.collect()

        # Final cache clear
        torch.cuda.empty_cache()

        logger.warning("✅ Emergency cleanup completed")
        return True

    except Exception as e:
        logger.error(f"❌ Emergency cleanup failed: {e}")
        return False

def test_memory_allocation(test_size_mb: int = 100):
    """
    Test if GPU memory allocation works after cleanup.

    Args:
        test_size_mb: Size of test tensor in MB

    Returns:
        bool: True if allocation successful, False otherwise
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available - cannot test memory allocation")
        return False

    logger.info(f"🧪 Testing {test_size_mb}MB memory allocation...")

    try:
        # Try to allocate on GPU 0 first
        device_id = 0
        if device_id < torch.cuda.device_count():
            torch.cuda.set_device(device_id)

            # Calculate tensor size
            elements = int(test_size_mb * 1024**2 / 4)  # 4 bytes per float32

            # Allocate test tensor
            test_tensor = torch.randn(elements, dtype=torch.float32, device=device_id)

            # Immediate cleanup
            del test_tensor
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            logger.info(f"✅ {test_size_mb}MB allocation test successful on GPU {device_id}")
            return True

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error(f"❌ {test_size_mb}MB allocation test failed - still out of memory")
            return False
        else:
            logger.error(f"❌ Allocation test failed with error: {e}")
            return False
    except Exception as e:
        logger.error(f"❌ Allocation test failed: {e}")
        return False

def clear_and_verify(test_size_mb: int = 100, max_attempts: int = 3):
    """
    Clear GPU memory and verify it works with test allocation.

    Args:
        test_size_mb: Size of test allocation in MB
        max_attempts: Maximum cleanup attempts

    Returns:
        bool: True if successful, False otherwise
    """
    logger.info(f"🧹 Starting memory clear and verify process (max {max_attempts} attempts)...")

    # Show initial state
    print_gpu_memory_stats()

    for attempt in range(max_attempts):
        logger.info(f"\n--- Attempt {attempt + 1}/{max_attempts} ---")

        # Perform cleanup
        if attempt == 0:
            # Standard cleanup on first attempt
            success = aggressive_vram_cleanup()
        else:
            # Emergency cleanup on subsequent attempts
            success = emergency_memory_cleanup()

        if not success:
            logger.error(f"Cleanup failed on attempt {attempt + 1}")
            continue

        # Test allocation
        if test_memory_allocation(test_size_mb):
            logger.info(f"✅ Memory clear and verify successful on attempt {attempt + 1}")

            # Show final state
            print_gpu_memory_stats()
            return True
        else:
            logger.warning(f"⚠️  Memory test failed on attempt {attempt + 1}")

    logger.error(f"❌ Memory clear and verify failed after {max_attempts} attempts")
    return False

def initialize_rocm_memory_environment():
    """
    Initialize ROCm environment variables for optimal memory management.
    Call this once at program startup.
    """
    logger.info("🔧 Initializing ROCm memory environment...")

    # ROCm 6.3 memory management settings
    os.environ.update({
        'PYTORCH_HIP_ALLOC_CONF': 'max_split_size_mb:256,garbage_collection_threshold:0.95',
        'CUDA_LAUNCH_BLOCKING': '1',  # Better error handling
        'HSA_ENABLE_SDMA': '0',  # ROCm optimization
        'HSA_ENABLE_INTERRUPT': '0',  # ROCm optimization
        'HSA_UNALIGNED_ACCESS_MODE': '1',  # ROCm optimization
        'HIP_VISIBLE_DEVICES': '0,1',  # Dual GPU setup
        'CUDA_VISIBLE_DEVICES': '0,1',  # Fallback for PyTorch
    })

    logger.info("✅ ROCm memory environment initialized")

def main():
    """Main function for standalone memory clearing."""
    import argparse

    parser = argparse.ArgumentParser(description="GPU Memory Clearing Utility for ROCm")
    parser.add_argument("--test-size", type=int, default=100,
                       help="Test allocation size in MB (default: 100)")
    parser.add_argument("--max-attempts", type=int, default=3,
                       help="Maximum cleanup attempts (default: 3)")
    parser.add_argument("--emergency", action="store_true",
                       help="Use emergency cleanup mode")
    parser.add_argument("--test-only", action="store_true",
                       help="Only test memory allocation, no cleanup")

    args = parser.parse_args()

    print("=" * 60)
    print("GPU Memory Clearing Utility for ROCm 6.3")
    print("AMD RX 7900 XT - Dual GPU Setup")
    print("=" * 60)

    # Initialize environment
    initialize_rocm_memory_environment()

    # Show initial state
    print_gpu_memory_stats()

    if args.test_only:
        # Only test allocation
        success = test_memory_allocation(args.test_size)
    elif args.emergency:
        # Emergency cleanup only
        success = emergency_memory_cleanup()
        if success:
            success = test_memory_allocation(args.test_size)
    else:
        # Full clear and verify
        success = clear_and_verify(args.test_size, args.max_attempts)

    if success:
        print("\n✅ GPU memory clearing completed successfully!")
        return 0
    else:
        print("\n❌ GPU memory clearing failed!")
        print("You may need to restart the Python process or reboot the system.")
        return 1

if __name__ == "__main__":
    exit(main())