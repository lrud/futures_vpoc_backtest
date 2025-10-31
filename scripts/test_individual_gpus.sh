#!/bin/bash
# Individual GPU testing script to isolate GPU issues

echo "🔧 Testing individual GPU functionality..."
echo "=========================================="

# Test GPU 0
echo ""
echo "🎯 Testing GPU 0 only..."
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ROCM_ARCH=gfx1100
export TORCH_BLAS_PREFER_HIPBLASLT=0

python3 -c "
import torch
import torch.nn as nn
import numpy as np

print('=== GPU 0 Test ===')
print(f'PyTorch version: {torch.__version__}')
print(f'ROCm/HIP version: {torch.version.hip}')

try:
    if not torch.cuda.is_available():
        print('❌ CUDA not available')
        exit(1)

    print(f'✅ CUDA available')
    print(f'GPU count: {torch.cuda.device_count()}')

    # Test GPU 0 specifically
    device = torch.device('cuda:0')
    print(f'✅ Using device: {device}')

    # Get GPU properties
    props = torch.cuda.get_device_properties(0)
    print(f'GPU 0: {props.name}')
    print(f'Architecture: {props.gcnArchName}')
    print(f'Memory: {props.total_memory/1e9:.1f}GB')
    print(f'Compute Units: {props.multi_processor_count}')

    # Test basic tensor operations
    print('\\n🧪 Testing tensor operations...')
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)

    # Matrix multiplication
    z = torch.mm(x, y)
    print('✅ Matrix multiplication successful')

    # Test mixed precision
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        z_fp16 = torch.mm(x.half(), y.half())
    print('✅ Mixed precision (BF16) successful')

    # Test neural network
    print('\\n🧠 Testing neural network...')
    model = nn.Sequential(
        nn.Linear(1000, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 1)
    ).to(device)

    # Forward pass
    input_tensor = torch.randn(32, 1000, device=device)
    output = model(input_tensor)
    print(f'✅ Neural network forward pass successful. Output shape: {output.shape}')

    # Test backward pass
    loss = output.sum()
    loss.backward()
    print('✅ Backward pass successful')

    # Memory test
    print('\\n💾 Testing memory allocation...')
    try:
        big_tensor = torch.randn(5000, 5000, device=device)  # ~100MB
        del big_tensor
        torch.cuda.empty_cache()
        print('✅ Large memory allocation successful')
    except Exception as e:
        print(f'❌ Memory allocation failed: {e}')

    print('\\n✅ GPU 0 ALL TESTS PASSED')

except Exception as e:
    print(f'❌ GPU 0 test failed: {e}')
    import traceback
    traceback.print_exc()
"

echo ""
echo "=========================================="
echo ""

# Test GPU 1
echo "🎯 Testing GPU 1 only..."
export HIP_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES=1

python3 -c "
import torch
import torch.nn as nn
import numpy as np

print('=== GPU 1 Test ===')
print(f'PyTorch version: {torch.__version__}')
print(f'ROCm/HIP version: {torch.version.hip}')

try:
    if not torch.cuda.is_available():
        print('❌ CUDA not available')
        exit(1)

    print(f'✅ CUDA available')
    print(f'GPU count: {torch.cuda.device_count()}')

    # Test GPU 1 specifically - map to device 0 since it's the only visible one
    device = torch.device('cuda:0')  # This is actually physical GPU 1
    print(f'✅ Using device: {device}')

    # Get GPU properties
    props = torch.cuda.get_device_properties(0)
    print(f'GPU 1: {props.name}')
    print(f'Architecture: {props.gcnArchName}')
    print(f'Memory: {props.total_memory/1e9:.1f}GB')
    print(f'Compute Units: {props.multi_processor_count}')

    # Test basic tensor operations
    print('\\n🧪 Testing tensor operations...')
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)

    # Matrix multiplication
    z = torch.mm(x, y)
    print('✅ Matrix multiplication successful')

    # Test mixed precision
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        z_fp16 = torch.mm(x.half(), y.half())
    print('✅ Mixed precision (BF16) successful')

    # Test neural network
    print('\\n🧠 Testing neural network...')
    model = nn.Sequential(
        nn.Linear(1000, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 1)
    ).to(device)

    # Forward pass
    input_tensor = torch.randn(32, 1000, device=device)
    output = model(input_tensor)
    print(f'✅ Neural network forward pass successful. Output shape: {output.shape}')

    # Test backward pass
    loss = output.sum()
    loss.backward()
    print('✅ Backward pass successful')

    # Memory test
    print('\\n💾 Testing memory allocation...')
    try:
        big_tensor = torch.randn(5000, 5000, device=device)  # ~100MB
        del big_tensor
        torch.cuda.empty_cache()
        print('✅ Large memory allocation successful')
    except Exception as e:
        print(f'❌ Memory allocation failed: {e}')

    print('\\n✅ GPU 1 ALL TESTS PASSED')

except Exception as e:
    print(f'❌ GPU 1 test failed: {e}')
    import traceback
    traceback.print_exc()
"

echo ""
echo "=========================================="
echo "🎯 Dual GPU comparison test..."
export HIP_VISIBLE_DEVICES=0,1
export CUDA_VISIBLE_DEVICES=0,1

python3 -c "
import torch

print('=== Dual GPU Test ===')

try:
    if not torch.cuda.is_available():
        print('❌ CUDA not available')
        exit(1)

    gpu_count = torch.cuda.device_count()
    print(f'Detected {gpu_count} GPUs')

    for i in range(gpu_count):
        try:
            props = torch.cuda.get_device_properties(i)
            print(f'GPU {i}: {props.name} ({props.total_memory/1e9:.1f}GB)')

            # Quick tensor test on each GPU
            x = torch.randn(100, 100, device=f'cuda:{i}')
            y = torch.mm(x, x)
            print(f'  ✅ GPU {i} basic operations successful')

        except Exception as e:
            print(f'  ❌ GPU {i} failed: {e}')

    print('\\n✅ Dual GPU test completed')

except Exception as e:
    print(f'❌ Dual GPU test failed: {e}')
    import traceback
    traceback.print_exc()
"

echo ""
echo "=========================================="
echo "📊 GPU Memory Status..."
rocm-smi --showmeminfo vram --showtemp --showpower --showperflevel

echo ""
echo "=========================================="
echo "✅ Individual GPU testing completed!"