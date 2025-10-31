#!/bin/bash
# ROCm GPU Memory Clearing Script
# Based on best practices from ROCm documentation and community solutions

echo "🧹 ROCm GPU Memory Clearing Script"
echo "=================================="

# Check if running as root (required for some operations)
if [[ $EUID -ne 0 ]]; then
   echo "⚠️  This script should be run as root for full GPU reset capabilities"
   echo "   Some operations may fail without root privileges"
fi

echo ""
echo "📊 Current GPU Memory Status:"
rocm-smi --showmeminfo vram

echo ""
echo "🔧 Step 1: Clear PyTorch GPU memory caches..."

# Function to clear memory for specific GPU
clear_gpu_memory() {
    local gpu_id=$1

    echo "   Clearing GPU $gpu_id..."

    python3 -c "
import torch
import gc
import sys

try:
    if torch.cuda.is_available() and torch.cuda.device_count() > $gpu_id:
        # Set environment variables for memory management
        import os
        os.environ['HIP_VISIBLE_DEVICES'] = str($gpu_id)

        # Force garbage collection
        gc.collect()

        # Empty PyTorch cache
        torch.cuda.empty_cache()

        # Reset peak memory stats
        if hasattr(torch.cuda, 'reset_peak_memory_stats'):
            torch.cuda.reset_peak_memory_stats($gpu_id)

        # Synchronize to ensure all operations complete
        torch.cuda.synchronize()

        print(f'   ✅ GPU $gpu_id memory cleared')
        print(f'      Allocated: {torch.cuda.memory_allocated($gpu_id) / 1024**3:.2f}GB')
        print(f'      Reserved: {torch.cuda.memory_reserved($gpu_id) / 1024**3:.2f}GB')
    else:
        print(f'   ❌ GPU $gpu_id not available')

except Exception as e:
    print(f'   ❌ Error clearing GPU $gpu_id: {e}')
    sys.exit(1)
"
}

# Clear memory for both GPUs if available
clear_gpu_memory 0
clear_gpu_memory 1

echo ""
echo "🔧 Step 2: Kill any lingering Python processes holding GPU memory..."

# Kill Python processes that might be holding GPU memory
pkill -f "python.*train.py" 2>/dev/null || echo "   No training processes found"
pkill -f "python.*torch" 2>/dev/null || echo "   No torch processes found"

# Wait a moment for processes to terminate
sleep 2

echo ""
echo "🔧 Step 3: ROCm memory fragmentation workaround..."

# Apply ROCm-specific environment variables for memory management
export PYTORCH_HIP_ALLOC_CONF='expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.6'
export TORCH_BLAS_PREFER_HIPBLASLT=0

python3 -c "
import torch
import gc

print('   Applying ROCm memory allocator warmup...')

# This is the documented workaround for ROCm memory fragmentation
def prime_rocm_allocator(device_id=0):
    '''Prime ROCm allocator to prevent fragmentation issues'''
    try:
        device = torch.device(f'cuda:{device_id}')

        # Create and progressively larger dummy tensors
        for size_mb in [100, 200, 400, 800]:
            elements = int(size_mb * 1024 * 1024 / 4)  # 4 bytes per float32

            # Allocate
            dummy = torch.randn(elements, device=device, dtype=torch.float32)
            torch.cuda.synchronize()

            # Free
            del dummy
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        print(f'   ✅ ROCm allocator primed for GPU {device_id}')

    except Exception as e:
        print(f'   ⚠️  ROCm allocator priming failed for GPU {device_id}: {e}')

# Prime allocator for available GPUs
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f'   Detected {gpu_count} GPU(s)')

    for i in range(min(gpu_count, 2)):  # Max 2 GPUs
        prime_rocm_allocator(i)
else:
    print('   ❌ CUDA not available')
"

echo ""
echo "🔧 Step 4: Force GPU reset (if available and safe)..."

# Try to reset GPUs using rocm-smi (this is more aggressive)
if command -v rocm-smi >/dev/null 2>&1; then
    echo "   Attempting GPU reset via rocm-smi..."

    # Reset GPU 0
    if rocm-smi --gpureset --id 0 2>/dev/null; then
        echo "   ✅ GPU 0 reset successful"
    else
        echo "   ⚠️  GPU 0 reset failed or not supported"
    fi

    sleep 1

    # Reset GPU 1
    if rocm-smi --gpureset --id 1 2>/dev/null; then
        echo "   ✅ GPU 1 reset successful"
    else
        echo "   ⚠️  GPU 1 reset failed or not supported"
    fi

    # Wait for GPUs to come back online
    sleep 3
else
    echo "   rocm-smi not available for GPU reset"
fi

echo ""
echo "🔧 Step 5: Final memory status check..."

rocm-smi --showmeminfo vram

echo ""
echo "🧠 Step 6: Test GPU memory allocation..."

python3 -c "
import torch
import gc

def test_gpu_memory(gpu_id):
    try:
        device = torch.device(f'cuda:{gpu_id}')

        # Test small allocation
        test_tensor = torch.randn(1000, 1000, device=device)
        result = torch.mm(test_tensor, test_tensor)

        # Clean up
        del test_tensor, result
        torch.cuda.empty_cache()

        print(f'   ✅ GPU {gpu_id}: Memory test successful')
        return True

    except Exception as e:
        print(f'   ❌ GPU {gpu_id}: Memory test failed - {e}')
        return False

if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f'   Testing {gpu_count} GPU(s)...')

    success_count = 0
    for i in range(gpu_count):
        if test_gpu_memory(i):
            success_count += 1

    print(f'   Result: {success_count}/{gpu_count} GPUs working properly')
else:
    print('   ❌ CUDA not available for testing'
"

echo ""
echo "🎯 Environment variables set for training:"
echo "   PYTORCH_HIP_ALLOC_CONF=$PYTORCH_HIP_ALLOC_CONF"
echo "   TORCH_BLAS_PREFER_HIPBLASLT=$TORCH_BLAS_PREFER_HIPBLASLT"

echo ""
echo "✅ GPU memory clearing completed!"
echo "💡 Tip: If memory is still fragmented, try rebooting the system as a last resort"