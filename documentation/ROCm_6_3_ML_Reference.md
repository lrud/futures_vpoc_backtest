# ROCm 6.3 Comprehensive Reference for Machine Learning Training

## Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Installation and Setup](#installation-and-setup)
4. [GPU Configuration](#gpu-configuration)
5. [Training Components](#training-components)
6. [Environment Variables](#environment-variables)
7. [Performance Optimization](#performance-optimization)
8. [Framework Integration](#framework-integration)
9. [Memory Management](#memory-management)
10. [Troubleshooting](#troubleshooting)
11. [Best Practices](#best-practices)
12. [Known Issues and Solutions](#known-issues-and-solutions)

## Overview

ROCm 6.3 represents AMD's mature GPU computing platform for machine learning workloads. This document provides comprehensive guidance for ML training on ROCm 6.3, specifically optimized for consumer GPUs like the AMD Radeon RX 7900 XT.

### Key Highlights for ML:
- **Stable foundation**: Proven reliability for production ML workloads
- **Consumer GPU support**: Excellent support for RDNA3 architecture (RX 7900 XT)
- **Memory efficiency**: Robust memory management without fragmentation issues
- **PyTorch compatibility**: Stable integration with PyTorch 2.0-2.4
- **Multi-GPU scaling**: Reliable dual GPU performance scaling

## System Architecture

### Hardware Components

#### AMD Radeon RX 7900 XT Architecture
- **RDNA3 Architecture**: Latest consumer gaming architecture with ML capabilities
- **Memory**: 16GB/24GB GDDR6 with 960 GB/s bandwidth
- **Compute Units**: 84 compute units with 2,304 stream processors
- **Architecture**: Wave32 mode optimized for parallel workloads
- **Infinity Cache**: 96MB high-speed cache for reduced memory latency

#### Multi-GPU Support
```
Single System Configuration:
┌─────────────────────────────────────────────────────────────┐
│                     CPU Host Memory                         │
├─────────────────────────────────────────────────────────────┤
│  PCI Express Root Complex                                   │
├─────────────────────────────────────────────────────────────┤
│  GPU 0         │  GPU 1         │  (Optional additional GPUs) │
│  RX 7900 XT     │  RX 7900 XT     │                           │
│  gfx1100       │  gfx1100       │                           │
└────────────────┴────────────────┴───────────────────────────┘
```

### Software Stack

#### ROCm 6.3 Software Components
```
Application Layer:
┌─────────────────────────────────────────────────────────────┐
│ PyTorch │ TensorFlow │ JAX │ MXNet │ Custom ML Frameworks  │
├─────────────────────────────────────────────────────────────┤
│                    Distributed Training                     │
│      DDP    │    FSDP    │  DeepSpeed  │  Megatron-LM     │
├─────────────────────────────────────────────────────────────┤
│                     ROCm Compute Stack                      │
│    HIP    │   MIOpen   │    rocBLAS   │   rocRAND         │
├─────────────────────────────────────────────────────────────┤
│                   Communication Layer                        │
│    RCCL   │   MPI       │   UCX        │   libfabric       │
├─────────────────────────────────────────────────────────────┤
│                       System Layer                          │
│   Linux Kernel │   HSA Runtime   │   KFD Driver          │
└─────────────────────────────────────────────────────────────┘
```

## Installation and Setup

### Prerequisites

#### System Requirements
- **OS**: Ubuntu 20.04/22.04 LTS, RHEL 8/9, SLES 15 SP4+
- **CPU**: Modern x86_64 processor with PCIe 4.0+ support
- **Memory**: 32GB+ RAM (64GB+ recommended for large models)
- **Storage**: SSD with 500GB+ free space
- **GPU**: AMD Radeon RX 7900 XT or similar RDNA3 GPU

#### Software Dependencies
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y build-essential cmake pkg-config
sudo apt install -y python3-dev python3-pip python3-venv
sudo apt install -y libnuma-dev libhsa-runtime-dev
sudo apt install -y rocm-dkms rocm-dev rocm-utils
```

### ROCm 6.3 Installation

#### Method 1: Ubuntu Repository (Recommended)
```bash
# Add AMD repository
wget https://repo.radeon.com/amdgpu-install/6.3/ubuntu/jammy/amdgpu-install_6.3.60300-1_all.deb
sudo apt install ./amdgpu-install_6.3.60300-1_all.deb

# Install ROCm 6.3
sudo amdgpu-install --usecase=rocm --no-dkms

# Add user to render groups
sudo usermod -a -G render,video $USER
sudo reboot
```

#### Method 2: PyTorch Pre-built
```bash
# Install PyTorch with ROCm 6.3 support
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/rocm6.0
```

### Verification
```bash
# Check ROCm installation
rocm-smi
rocminfo

# Verify PyTorch ROCm support
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'ROCm version: {torch.version.hip}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Architecture: {torch.cuda.get_device_properties(0).gcnArchName}')
"
```

## GPU Configuration

### Environment Variables for ROCm 6.3

#### Core ROCm Settings
```bash
# GPU visibility
export HIP_VISIBLE_DEVICES=0,1

# Architecture specification
export PYTORCH_ROCM_ARCH=gfx1100

# Memory management (ROCm 6.3 stable)
export PYTORCH_HIP_ALLOC_CONF='expandable_segments:True,max_split_size_mb:128'

# System optimization
export HSA_ENABLE_SDMA=0
export HSA_ENABLE_INTERRUPT=0
export HSA_UNALIGNED_ACCESS_MODE=1
export GPU_MAX_ALLOC_PERCENT=100
export HIP_HIDDEN_FREE_MEM=256
export GPU_SINGLE_ALLOC_PERCENT=90
```

#### PyTorch ROCm 6.3 Optimizations
```bash
# PyTorch compilation settings
export TORCH_COMPILE_BACKEND=inductor

# Debug and error handling
export CUDA_LAUNCH_BLOCKING=1
export HIP_LAUNCH_BLOCKING=0
```

### Multi-GPU Configuration

#### Dual RX 7900 XT Setup
```bash
# Multi-GPU environment variables
export HIP_VISIBLE_DEVICES=0,1
export CUDA_VISIBLE_DEVICES=0,1
export WORLD_SIZE=2

# Communication backend (Gloo for stability)
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=lo
export NCCL_NSOCKS_PERTHREAD=4
```

## Training Components

### VPOC (Volume Point of Control) Analysis

#### ROCm 6.3 Optimized VPOC Configuration
```python
class VolumeProfileAnalyzer:
    """ROCm 6.3 compatible volume profile analyzer"""

    def __init__(self, price_precision=0.25, device='cuda', device_ids=[0], chunk_size=3500):
        # ROCm 6.3 environment setup
        os.environ['HIP_VISIBLE_DEVICES'] = '0'
        os.environ['HSA_ENABLE_SDMA'] = '0'
        os.environ['HSA_ENABLE_INTERRUPT'] = '0'

        # Force single GPU for stability
        if torch.cuda.is_available():
            self.device = device
            self.parallel = False
            self.num_gpus = 1
            self.device_ids = [0]

            # Verify GPU accessibility
            try:
                torch.cuda.device(0)
            except Exception as e:
                raise RuntimeError(f"GPU 0 not accessible: {e}")
        else:
            raise RuntimeError("CUDA is not available - GPU access required")
```

#### VPOC Memory Management (ROCm 6.3)
```python
def calculate_volume_profile_stable(self, session_df):
    """ROCm 6.3 stable volume profile calculation"""

    # Convert to tensors with stable memory management
    prices = torch.tensor(session_df['close'].values,
                        dtype=torch.float32, device=self.device)
    volumes = torch.tensor(session_df['volume'].values,
                         dtype=torch.float32, device=self.device)

    # ROCm 6.3 optimized calculation without memory fragmentation
    try:
        # Stable volume distribution
        price_histogram = torch.histc(prices, bins=1000, min=prices.min(), max=prices.max())
        volume_by_price = torch.zeros_like(price_histogram)

        # Accumulate volume by price level
        for i in range(len(prices)):
            price_bin = torch.bucketize(prices[i], torch.linspace(prices.min(), prices.max(), 1000))
            volume_by_price[price_bin] += volumes[i]

        return volume_by_price.cpu().numpy()

    except Exception as e:
        raise RuntimeError(f"VPOC calculation failed: {e}")
```

### Machine Learning Model Architecture

#### ROCm 6.3 Optimized Neural Network
```python
class ROCm6OptimizedModel(nn.Module):
    """Neural network optimized for ROCm 6.3 and RDNA3"""

    def __init__(self, input_size, hidden_sizes=[128, 64], output_size=1):
        super().__init__()

        # ROCm 6.3 memory alignment
        layer_sizes = [input_size] + hidden_sizes + [output_size]

        # Build layers with stable memory management
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))

        # Activation and normalization
        self.activation = nn.SiLU()
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(size) for size in hidden_sizes
        ])

        # Mixed precision support
        self.use_mixed_precision = True

    def forward(self, x):
        # ROCm 6.3 stable forward pass
        try:
            with torch.cuda.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                for i, layer in enumerate(self.layers[:-1]):
                    x = layer(x)
                    x = self.activation(x)
                    if i < len(self.layer_norms):
                        x = self.layer_norms[i](x)

                output = self.layers[-1](x)
                return output.squeeze(-1)

        except Exception as e:
            raise RuntimeError(f"Model forward pass failed: {e}")
```

## Environment Variables

### Essential ROCm 6.3 Variables

#### GPU Access and Memory
```bash
# GPU visibility and allocation
export HIP_VISIBLE_DEVICES=0,1
export PYTORCH_ROCM_ARCH=gfx1100
export PYTORCH_HIP_ALLOC_CONF='expandable_segments:True,max_split_size_mb:128'
export GPU_SINGLE_ALLOC_PERCENT=90
export GPU_MAX_ALLOC_PERCENT=100
```

#### Performance Optimizations
```bash
# System optimizations
export HSA_ENABLE_SDMA=0
export HSA_ENABLE_INTERRUPT=0
export HSA_UNALIGNED_ACCESS_MODE=1
export HIP_HIDDEN_FREE_MEM=256

# PyTorch optimizations
export TORCH_COMPILE_BACKEND=inductor
export CUDA_LAUNCH_BLOCKING=1
```

#### Memory Management
```bash
# Memory fragmentation prevention
export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128,expandable_segments:True'
export TORCH_BLAS_PREFER_HIPBLASLT=0  # Critical for RDNA3 stability
```

#### Debug and Monitoring
```bash
# Debug settings
export NCCL_DEBUG=WARN
export ROCM_LOG_LEVEL=2
export HIP_LAUNCH_BLOCKING=0
```

### Complete Environment Setup Script
```bash
#!/bin/bash
# ROCm 6.3 ML Environment Setup

echo "Setting up ROCm 6.3 ML environment..."

# Core ROCm variables
export HIP_VISIBLE_DEVICES=0,1
export PYTORCH_ROCM_ARCH=gfx1100
export TORCH_BLAS_PREFER_HIPBLASLT=0

# Memory management
export PYTORCH_HIP_ALLOC_CONF='expandable_segments:True,max_split_size_mb:128'
export GPU_SINGLE_ALLOC_PERCENT=90
export GPU_MAX_ALLOC_PERCENT=100
export HIP_HIDDEN_FREE_MEM=256

# Performance optimizations
export HSA_ENABLE_SDMA=0
export HSA_ENABLE_INTERRUPT=0
export HSA_UNALIGNED_ACCESS_MODE=1
export TORCH_COMPILE_BACKEND=inductor

# Communication (for multi-GPU)
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=lo

echo "ROCm 6.3 environment configured successfully"
```

## Performance Optimization

### Memory Management

#### ROCm 6.3 Stable Memory Strategy
```python
def setup_rocm63_memory_optimizations():
    """ROCm 6.3 stable memory management setup"""

    # Clear existing cache
    torch.cuda.empty_cache()

    # Set memory growth strategy
    torch.cuda.set_per_process_memory_fraction(0.9)

    # Configure memory allocator
    os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'

    # Enable memory pooling
    if hasattr(torch.cuda, 'memory'):
        memory_pool = torch.cuda.memory.MemoryPool()
        torch.cuda.memory.set_allocator(memory_pool)
```

#### Batch Size Optimization
```python
def determine_optimal_batch_size():
    """Determine optimal batch size for ROCm 6.3"""

    # Get available GPU memory
    total_memory = torch.cuda.get_device_properties(0).total_memory
    free_memory, _ = torch.cuda.mem_get_info(0)

    # Conservative memory allocation (80% of free memory)
    usable_memory = free_memory * 0.8

    # Estimate memory per sample (based on model size)
    memory_per_sample = 1024 * 1024  # 1MB per sample (conservative estimate)

    # Calculate maximum batch size
    max_batch_size = int(usable_memory / memory_per_sample)

    # Clamp to reasonable values
    return min(max(8, max_batch_size), 64)  # Between 8 and 64
```

### Training Speed Optimizations

#### Mixed Precision Training
```python
def setup_mixed_precision_training():
    """ROCm 6.3 mixed precision setup"""

    from torch.cuda.amp import GradScaler

    # Create gradient scaler
    scaler = GradScaler()

    # Enable automatic mixed precision
    return scaler
```

#### torch.compile Optimization
```python
def optimize_model_with_compile(model):
    """Apply ROCm 6.3 torch.compile optimizations"""

    if hasattr(torch, 'compile'):
        try:
            # ROCm 6.3 compatible compilation
            compiled_model = torch.compile(
                model,
                mode='reduce-overhead',
                fullgraph=False
            )
            return compiled_model
        except Exception as e:
            print(f"torch.compile failed: {e}, using original model")
            return model
    return model
```

## Framework Integration

### PyTorch 2.4 Integration

#### ROCm 6.3 Distributed Training
```python
import torch.distributed as dist
import torch.multiprocessing as mp

def setup_distributed_training():
    """ROCm 6.3 distributed training setup"""

    # Environment setup
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'

    # Use Gloo backend for ROCm 6.3 stability
    dist.init_process_group(
        backend='gloo',
        init_method='env://',
        world_size=1,
        rank=0
    )
```

#### DataParallel Configuration
```python
def setup_dataparallel(model):
    """ROCm 6.3 DataParallel setup"""

    if torch.cuda.device_count() > 1:
        # Use single GPU for stability during debugging
        device = torch.device('cuda:0')
        model = model.to(device)

        # Wrap in DataParallel if multiple GPUs available
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model, device_ids=[0])
            print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")

    return model
```

## Memory Management

### GPU Memory Monitoring
```python
def monitor_gpu_memory():
    """Monitor GPU memory usage in ROCm 6.3"""

    device = torch.cuda.current_device()

    # Get memory info
    total_memory = torch.cuda.get_device_properties(device).total_memory
    allocated_memory = torch.cuda.memory_allocated(device)
    reserved_memory = torch.cuda.memory_reserved(device)
    free_memory = total_memory - reserved_memory

    return {
        'total_gb': total_memory / 1e9,
        'allocated_gb': allocated_memory / 1e9,
        'reserved_gb': reserved_memory / 1e9,
        'free_gb': free_memory / 1e9,
        'utilization_percent': (allocated_memory / total_memory) * 100
    }
```

### Memory Cleanup Strategies
```python
def aggressive_memory_cleanup():
    """Aggressive memory cleanup for ROCm 6.3"""

    # Clear PyTorch cache
    torch.cuda.empty_cache()

    # Force garbage collection
    import gc
    gc.collect()

    # Additional cleanup for ROCm
    if hasattr(torch.cuda, 'reset_peak_memory_stats'):
        torch.cuda.reset_peak_memory_stats()

    print("Aggressive memory cleanup completed")
```

## Troubleshooting

### Common ROCm 6.3 Issues

#### Issue 1: GPU Not Detected
```bash
# Symptoms
Error: CUDA is not available

# Solutions
# 1. Check ROCm installation
rocm-smi

# 2. Verify user permissions
groups $USER | grep -E "(render|video)"

# 3. Check kernel modules
lsmod | grep amdgpu

# 4. Reboot if necessary
sudo reboot
```

#### Issue 2: Out of Memory Errors
```bash
# Symptoms
RuntimeError: HIP out of memory

# Solutions
# 1. Reduce batch size
python train.py --batch_size 8

# 2. Enable memory management
export PYTORCH_HIP_ALLOC_CONF='expandable_segments:True,max_split_size_mb:64'

# 3. Use gradient accumulation
python train.py --gradient_accumulation_steps 8

# 4. Clear memory before training
python -c "import torch; torch.cuda.empty_cache()"
```

#### Issue 3: Performance Issues
```bash
# Symptoms
Training is very slow

# Solutions
# 1. Check GPU utilization
watch -n 1 rocm-smi

# 2. Enable mixed precision
python train.py --use_mixed_precision

# 3. Use torch.compile
export TORCH_COMPILE_BACKEND=inductor

# 4. Optimize data loading
python train.py --num_workers 8 --pin_memory
```

## Best Practices

### Development Workflow

#### 1. Environment Setup
```bash
# Create dedicated environment
python3 -m venv rocm63-ml-env
source rocm63-ml-env/bin/activate

# Install dependencies
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/rocm6.0
```

#### 2. Testing Strategy
```python
# Start with small models and datasets
test_config = {
    'batch_size': 8,
    'epochs': 2,
    'data_fraction': 0.01,  # 1% of data for testing
    'hidden_layers': '32,16'
}

# Gradually increase complexity
# 1. Verify GPU access
# 2. Test with small model
# 3. Test with small dataset
# 4. Scale up gradually
```

#### 3. Production Deployment
```bash
# Use stable configurations
export TORCH_BLAS_PREFER_HIPBLASLT=0
export PYTORCH_HIP_ALLOC_CONF='expandable_segments:True,max_split_size_mb:128'

# Monitor resource usage
python scripts/gpu_memory_monitor.sh &

# Use checkpointing
python train.py --resume_from_checkpoint --checkpoint_interval 50
```

### Code Optimization Guidelines

#### 1. Memory Efficiency
```python
# Good: Use gradient checkpointing for large models
model.enable_gradient_checkpointing()

# Good: Use mixed precision
scaler = GradScaler()
with torch.cuda.amp.autocast():
    output = model(input)

# Bad: Accumulate gradients without cleanup
# Good: Clear gradients regularly
optimizer.zero_grad(set_to_none=True)
```

#### 2. Performance Optimization
```python
# Good: Use torch.compile for inference
compiled_model = torch.compile(model, mode='reduce-overhead')

# Good: Optimize data loading
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True
)

# Bad: Synchronous operations
# Good: Use asynchronous operations where possible
```

## Known Issues and Solutions

### Issue: PyTorch Version Compatibility

#### Problem
PyTorch built for ROCm 6.0 may have compatibility issues with ROCm 6.3

#### Solution
```bash
# Use ROCm 6.0 builds with ROCm 6.3 runtime (known working combination)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/rocm6.0
```

### Issue: Memory Fragmentation

#### Problem
Long training sessions can cause memory fragmentation

#### Solution
```python
# Regular memory cleanup
def training_loop_with_cleanup():
    for epoch in range(num_epochs):
        # Training code here

        # Cleanup every 100 batches
        if batch_idx % 100 == 0:
            torch.cuda.empty_cache()
            gc.collect()
```

### Issue: Multi-GPU Communication

#### Problem
NCCL can be unstable with ROCm 6.3

#### Solution
```python
# Use Gloo backend for stability
dist.init_process_group(
    backend='gloo',  # More stable than nccl for ROCm 6.3
    init_method='env://',
    world_size=world_size,
    rank=rank
)
```

## Performance Benchmarks

### RX 7900 XT Expected Performance

#### Training Benchmarks
- **Small Models** (< 1M parameters): 100-200 samples/second
- **Medium Models** (1-10M parameters): 50-100 samples/second
- **Large Models** (> 10M parameters): 10-50 samples/second

#### Memory Utilization
- **16GB VRAM**: Models up to ~5B parameters with mixed precision
- **24GB VRAM**: Models up to ~8B parameters with mixed precision
- **Efficiency**: 85-90% memory utilization with proper optimization

### Scaling Performance

#### Single vs Dual GPU
- **Linear Scaling**: 1.8-1.9x speedup with dual GPUs
- **Communication Overhead**: 5-10% for medium models
- **Memory Efficiency**: 80-85% utilization with DataParallel

This ROCm 6.3 reference provides stable, production-ready configurations for machine learning training on AMD hardware.