#!/bin/bash
# ROCm 6.3 RX 7900 XT Specific Test Script

echo "🔧 Testing ROCm 6.3 with RX 7900 XT specific settings..."

# RX 7900 XT (RDNA3/gfx1100) specific environment variables
export HIP_VISIBLE_DEVICES=0
export PYTORCH_ROCM_ARCH=gfx1100
export TORCH_BLAS_PREFER_HIPBLASLT=0  # Critical for RDNA3 stability

# RX 7900 XT memory management (16GB/24GB GDDR6)
export PYTORCH_HIP_ALLOC_CONF='expandable_segments:True,max_split_size_mb:128'
export GPU_SINGLE_ALLOC_PERCENT=90
export GPU_MAX_ALLOC_PERCENT=100
export HIP_HIDDEN_FREE_MEM=256

# RX 7900 XT performance optimizations (RDNA3 specific)
export HSA_ENABLE_SDMA=0
export HSA_ENABLE_INTERRUPT=0
export HSA_UNALIGNED_ACCESS_MODE=1

# RX 7900 XT PyTorch settings
export TORCH_COMPILE_BACKEND=inductor
export CUDA_LAUNCH_BLOCKING=1

# Communication settings (dual RX 7900 XT)
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=lo
export NCCL_NSOCKS_PERTHREAD=4

echo "✅ RX 7900 XT ROCm 6.3 environment configured"
echo "🎯 Testing GPU detection..."

python3 -c "
import torch
import os

print('=== RX 7900 XT ROCm 6.3 Test ===')
print(f'PyTorch version: {torch.__version__}')
print(f'ROCm/HIP version: {torch.version.hip}')

if torch.cuda.is_available():
    print(f'✅ CUDA available: {torch.cuda.device_count()} GPU(s)')

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'🎯 GPU {i}: {props.name}')
        print(f'   Architecture: {props.gcnArchName}')
        print(f'   Memory: {props.total_memory/1e9:.1f}GB')
        print(f'   Compute Units: {props.multi_processor_count}')

        # Verify it's RX 7900 XT (gfx1100)
        if '7900' in props.name or 'gfx1100' in props.gcnArchName:
            print('   ✅ RX 7900 XT detected!')
        else:
            print('   ⚠️  Warning: Non-RX 7900 XT GPU detected')
else:
    print('❌ CUDA not available')
    exit(1)

# Test basic tensor operations
print('\\n🧪 Testing RX 7900 XT tensor operations...')
try:
    device = torch.device('cuda:0')
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)
    z = torch.mm(x, y)
    print('✅ Matrix multiplication successful')

    # Test mixed precision
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        z_fp16 = torch.mm(x.half(), y.half())
    print('✅ Mixed precision (BF16) successful')

    print('✅ All RX 7900 XT tests passed!')
except Exception as e:
    print(f'❌ Tensor operation failed: {e}')
    exit(1)
"

echo "🚀 RX 7900 XT ROCm 6.3 test completed!"