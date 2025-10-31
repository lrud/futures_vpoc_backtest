#!/usr/bin/env python3
"""
Realistic GPU test to verify kernel-level parameters are working
Tests GPU compute capability, memory management, and tensor operations
"""

import torch
import gc
import time
import os

def print_system_info():
    """Print system and GPU information."""
    print("=" * 60)
    print("GPU System Information")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"HIP version: {torch.version.hip if hasattr(torch.version, 'hip') else 'N/A'}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {props.name}")
        print(f"  Compute capability: {props.major}.{props.minor}")
        print(f"  Total memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"  Multiprocessors: {props.multi_processor_count}")

def test_gpu_memory_basics():
    """Test basic GPU memory allocation and reporting."""
    print("\n" + "=" * 60)
    print("Testing GPU Memory Basics")
    print("=" * 60)

    try:
        for device_id in range(torch.cuda.device_count()):
            torch.cuda.set_device(device_id)

            # Get initial memory state
            free, total = torch.cuda.mem_get_info(device_id)
            print(f"\nGPU {device_id} initial state:")
            print(f"  Free: {free / 1024**3:.2f} GB")
            print(f"  Total: {total / 1024**3:.2f} GB")
            print(f"  Used: {(total - free) / 1024**3:.2f} GB ({(total - free) / total * 100:.1f}%)")

            # Test small allocation
            print(f"  Testing small tensor allocation...")
            small_tensor = torch.randn(100, 100, device=device_id)
            print(f"  ✅ Small tensor (100x100) created successfully")

            # Check memory after small allocation
            free_after, _ = torch.cuda.mem_get_info(device_id)
            print(f"  Memory after small tensor: {free_after / 1024**3:.2f} GB free")

            # Clean up
            del small_tensor
            torch.cuda.empty_cache()

        return True

    except Exception as e:
        print(f"  ❌ Memory test failed: {e}")
        return False

def test_tensor_operations():
    """Test progressively larger tensor operations."""
    print("\n" + "=" * 60)
    print("Testing Tensor Operations")
    print("=" * 60)

    # Test sizes - start small and gradually increase
    test_sizes = [
        (100, 100),      # 10K elements
        (500, 500),      # 250K elements
        (1000, 1000),    # 1M elements
        (2000, 2000),    # 4M elements
    ]

    device_id = 0  # Test on first GPU

    try:
        torch.cuda.set_device(device_id)

        for size in test_sizes:
            print(f"\nTesting size {size[0]}x{size[1]} ({size[0] * size[1]:,} elements)...")

            # Create tensor
            start_time = time.time()
            x = torch.randn(size[0], size[1], device=device_id)
            create_time = time.time() - start_time

            # Matrix multiplication
            start_time = time.time()
            y = torch.matmul(x, x.t())  # Use transpose for square result
            matmul_time = time.time() - start_time

            print(f"  ✅ Create: {create_time:.3f}s, MatMul: {matmul_time:.3f}s")
            print(f"  Result shape: {y.shape}")

            # Memory usage
            allocated = torch.cuda.memory_allocated(device_id)
            cached = torch.cuda.memory_reserved(device_id)
            print(f"  Allocated: {allocated / 1024**2:.1f} MB")
            print(f"  Cached: {cached / 1024**2:.1f} MB")

            # Clean up
            del x, y
            torch.cuda.empty_cache()
            gc.collect()

            # Small delay between tests
            time.sleep(0.1)

        return True

    except Exception as e:
        print(f"  ❌ Tensor operations failed: {e}")
        return False

def test_multi_gpu():
    """Test operations across multiple GPUs."""
    print("\n" + "=" * 60)
    print("Testing Multi-GPU Operations")
    print("=" * 60)

    if torch.cuda.device_count() < 2:
        print("Only one GPU available - skipping multi-GPU test")
        return True

    try:
        # Create tensors on different GPUs
        print("Creating tensors on GPU 0 and GPU 1...")
        x0 = torch.randn(500, 500, device=0)
        x1 = torch.randn(500, 500, device=1)

        print("✅ Multi-GPU tensor creation successful")

        # Test memory info on both
        for i in [0, 1]:
            free, total = torch.cuda.mem_get_info(i)
            print(f"GPU {i}: {(total - free) / 1024**2:.1f} MB used")

        # Clean up
        del x0, x1
        torch.cuda.empty_cache()

        return True

    except Exception as e:
        print(f"❌ Multi-GPU test failed: {e}")
        return False

def test_kernel_parameters():
    """Test operations that stress kernel-level parameters."""
    print("\n" + "=" * 60)
    print("Testing Kernel-Level Parameters")
    print("=" * 60)

    try:
        device_id = 0
        torch.cuda.set_device(device_id)

        # Test 1: Many small operations (tests scheduler)
        print("Test 1: Many small operations...")
        start_time = time.time()
        tensors = []
        for i in range(100):
            t = torch.randn(50, 50, device=device_id)
            tensors.append(t)

        # Synchronize all operations
        torch.cuda.synchronize(device_id)
        many_ops_time = time.time() - start_time
        print(f"  ✅ 100 small tensors in {many_ops_time:.3f}s")

        # Clean up
        del tensors
        torch.cuda.empty_cache()

        # Test 2: Large matrix operation (tests memory bandwidth)
        print("\nTest 2: Large matrix operation...")
        start_time = time.time()
        large_x = torch.randn(2000, 2000, device=device_id)
        large_result = torch.matmul(large_x, large_x.t())
        torch.cuda.synchronize(device_id)
        large_ops_time = time.time() - start_time
        print(f"  ✅ Large matrix op in {large_ops_time:.3f}s")

        # Test 3: Memory transfer speed
        print("\nTest 3: CPU-GPU memory transfer...")
        cpu_tensor = torch.randn(1000, 1000)

        start_time = time.time()
        gpu_tensor = cpu_tensor.to(device_id)
        torch.cuda.synchronize(device_id)
        cpu_to_gpu_time = time.time() - start_time

        start_time = time.time()
        back_to_cpu = gpu_tensor.cpu()
        gpu_to_cpu_time = time.time() - start_time

        print(f"  ✅ CPU→GPU: {cpu_to_gpu_time:.3f}s, GPU→CPU: {gpu_to_cpu_time:.3f}s")

        # Final memory state
        allocated = torch.cuda.memory_allocated(device_id)
        cached = torch.cuda.memory_reserved(device_id)
        free, total = torch.cuda.mem_get_info(device_id)

        print(f"\nFinal memory state:")
        print(f"  Allocated: {allocated / 1024**2:.1f} MB")
        print(f"  Cached: {cached / 1024**2:.1f} MB")
        print(f"  Free: {free / 1024**3:.2f} GB")

        # Clean up everything
        del large_x, large_result, gpu_tensor, back_to_cpu, cpu_tensor
        torch.cuda.empty_cache()
        gc.collect()

        return True

    except Exception as e:
        print(f"❌ Kernel parameter test failed: {e}")
        return False

def main():
    """Run all GPU tests."""
    print("Starting comprehensive GPU test...")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Run tests
    tests = [
        ("System Info", print_system_info),
        ("Memory Basics", test_gpu_memory_basics),
        ("Tensor Operations", test_tensor_operations),
        ("Multi-GPU", test_multi_gpu),
        ("Kernel Parameters", test_kernel_parameters),
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            print(f"\n{'='*60}")
            print(f"Running: {test_name}")
            print(f"{'='*60}")

            result = test_func()
            results[test_name] = result

            if result:
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")

        except Exception as e:
            print(f"❌ {test_name} CRASHED: {e}")
            results[test_name] = False

    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All GPU tests passed! Kernel parameters are working correctly.")
        return True
    else:
        print("⚠️  Some GPU tests failed. Check kernel parameters and ROCm installation.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)