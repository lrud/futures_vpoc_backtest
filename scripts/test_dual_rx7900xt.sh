#!/bin/bash
# Dual RX 7900 XT ROCm 6.3 Test Script

echo "🚀 Testing Dual RX 7900 XT Configuration..."

# Dual RX 7900 XT environment variables
export HIP_VISIBLE_DEVICES=0,1
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_ROCM_ARCH=gfx1100
export TORCH_BLAS_PREFER_HIPBLASLT=0

# Memory management for dual RX 7900 XT
export PYTORCH_HIP_ALLOC_CONF='expandable_segments:True,max_split_size_mb:128'
export GPU_SINGLE_ALLOC_PERCENT=90
export GPU_MAX_ALLOC_PERCENT=100
export HIP_HIDDEN_FREE_MEM=256

# Dual GPU communication
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=lo
export NCCL_NSOCKS_PERTHREAD=4

python3 -c "
import torch
import torch.nn as nn
import torch.optim as optim

print('=== Dual RX 7900 XT Test ===')
print(f'PyTorch version: {torch.__version__}')
print(f'ROCm/HIP version: {torch.version.hip}')

if not torch.cuda.is_available():
    print('❌ CUDA not available')
    exit(1)

print(f'✅ CUDA available: {torch.cuda.device_count()} GPU(s)')

# Test both GPUs
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'🎯 GPU {i}: {props.name}')
    print(f'   Architecture: {props.gcnArchName}')
    print(f'   Memory: {props.total_memory/1e9:.1f}GB')
    print(f'   Compute Units: {props.multi_processor_count}')

# Create a simple model for dual GPU testing
class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.layers(x)

print('\\n🧪 Testing dual GPU operations...')

try:
    # Test tensor operations on both GPUs
    device0 = torch.device('cuda:0')
    device1 = torch.device('cuda:1')

    # Create tensors on both GPUs
    x0 = torch.randn(1000, 100, device=device0)
    x1 = torch.randn(1000, 100, device=device1)

    print('✅ Tensors created on both GPUs')

    # Test model on both GPUs
    model0 = TestModel().to(device0)
    model1 = TestModel().to(device1)

    y0 = model0(x0)
    y1 = model1(x1)

    print('✅ Models running on both GPUs')

    # Test DataParallel
    if torch.cuda.device_count() >= 2:
        print('\\n🔄 Testing DataParallel...')
        model_dp = TestModel()
        model_dp = nn.DataParallel(model_dp, device_ids=[0,1])
        model_dp = model_dp.to(device0)

        # Test with batch that spans both GPUs
        batch = torch.randn(2000, 100, device=device0)
        output_dp = model_dp(batch)

        print('✅ DataParallel working on both GPUs')
        print(f'   Output shape: {output_dp.shape}')

        # Check GPU utilization
        for i in range(2):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            cached = torch.cuda.memory_reserved(i) / 1e9
            print(f'   GPU {i}: {allocated:.2f}GB allocated, {cached:.2f}GB cached')

    print('\\n✅ All dual RX 7900 XT tests passed!')

except Exception as e:
    print(f'❌ Dual GPU test failed: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
"

echo "🚀 Dual RX 7900 XT test completed!"