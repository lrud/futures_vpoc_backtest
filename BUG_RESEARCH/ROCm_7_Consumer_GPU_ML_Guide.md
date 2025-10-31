# ROCm 7.0 Comprehensive Reference for Distributed Machine Learning Training
## With Consumer GPU (Radeon RX 7900 XT) Best Practices

## Table of Contents
1. [Overview](#overview)
2. [Consumer GPU Specifics](#consumer-gpu-specifics)
3. [System Architecture](#system-architecture)
4. [Installation and Setup](#installation-and-setup)
5. [Multi-GPU Configuration](#multi-gpu-configuration)
6. [Distributed Training Components](#distributed-training-components)
7. [Communication Libraries](#communication-libraries)
8. [Performance Optimization for Consumer GPUs](#performance-optimization-for-consumer-gpus)
9. [Framework Integration](#framework-integration)
10. [Consumer GPU Best Practices](#consumer-gpu-best-practices)
11. [Monitoring and Debugging](#monitoring-and-debugging)
12. [Troubleshooting](#troubleshooting)
13. [Benchmarking](#benchmarking)

## Overview

ROCm 7.0 represents a major advancement in AMD's GPU computing platform, optimized for distributed machine learning training. While primarily designed for datacenter Instinct GPUs (MI300X, MI350), ROCm 7.0 now includes production-grade support for **consumer Radeon GPUs (7000 and 9000 series)** on Linux through Radeon Software for Linux integration.

### Key Highlights for Consumer GPU Distributed ML:
- **Native support for Radeon RX 7900 XT/XTX** (RDNA3, 20-24GB VRAM)
- **Native support for Radeon RX 9070/9060** (RDNA4, 16-48GB VRAM)
- **20-30% performance improvement** over ROCm 6.4.x for consumer GPUs
- **Improved multi-GPU scaling** for data parallel training
- **Enhanced memory efficiency** for large model training
- **Mixed precision training** support with torch.cuda.amp
- **Gradient accumulation** for effective large batch sizes
- **DDP (Distributed Data Parallel)** support across dual/multiple consumer GPUs

### Performance Context:
- RX 7900 XTX achieves ~75% of RTX 3090 Ti performance for LLM fine-tuning
- RX 7900 XT achieves ~70% of RTX 3090 performance
- **Dual RX 7900 XTs provide near-linear scaling for DDP training** (1.9x speedup vs single GPU for smaller models)
- Memory bandwidth: 576 GB/s (RX 7900 XT) vs 1TB/s (MI300X)

## Consumer GPU Specifics

### Radeon RX 7900 XT/XTX Specifications
```
Hardware Configuration:
├── GPU Architecture: RDNA3 (GFX1100)
├── Compute Units: 84 (7900 XT) / 96 (7900 XTX)
├── Stream Processors: 5,376 / 6,144
├── Memory: 20GB / 24GB GDDR6 (or GDDR6X for XTX)
├── Memory Bandwidth: 576 GB/s (9GHz) / 624 GB/s (XTX at 9.5GHz)
├── Max Power: 250W / 290W
├── PCIe: Gen 4 x16 (dual GPU setup: shared x8 each)
├── Float32 Performance: 56 TFLOPS / 64 TFLOPS
├── Tensor Operations: Limited compared to datacenter GPUs
└── Infinity Fabric: Not available (consumer SKU limitation)

Multi-GPU Connection:
├── Physical: PCIe Gen 4 (8GB/s theoretical per direction)
├── Practical Throughput: ~6.5 GB/s per direction
├── Latency: ~0.5-1.0 μs per hop
└── Note: Consumer SKUs lack direct GPU interconnect; all traffic goes through CPU/PCIe root complex
```

### Why Dual RX 7900 XT Setup?
For consumer systems, dual GPUs connected via PCIe provide:

1. **Parallelism Benefits**:
   - Run multiple small experiments simultaneously (2x throughput for ablations)
   - Alternative: Sequential training jobs with near-zero latency switching

2. **Memory Aggregation**:
   - Combined 40GB-48GB VRAM enables training larger models
   - Data Parallel (DDP) mode can hold larger batches

3. **Throughput for Specific Workloads**:
   - Near-linear scaling (1.9-2.0x) for DDP on:
     - Smaller models (<7B parameters)
     - Moderate batch sizes per GPU
     - Low communication overhead workloads

4. **Throughput Trade-offs**:
   - **NOT recommended for** large model training requiring frequent gradient synchronization
   - Communication overhead (PCIe bandwidth limitation) becomes significant for:
     - Large models with many layers
     - High frequency communication patterns
     - Models exceeding single GPU capacity

### Memory Management for Consumer GPUs

**RX 7900 XT (20GB) Recommended Settings:**

```python
# Recommended configurations by use case

# Use Case 1: Fine-tuning 7B LLM
config_7b_finetune = {
    'batch_size_per_gpu': 8,          # Per-GPU batch size
    'gradient_accumulation_steps': 4,  # Effective batch: 8 * 2 GPUs * 4 = 64
    'max_sequence_length': 2048,
    'use_mixed_precision': True,       # BF16/FP16 - essential for consumer GPUs
    'use_deepspeed': False,            # Not recommended for dual consumer GPUs
    'use_flash_attention': True,       # Huge memory saver
    'memory_efficient_attention': True,
    'estimated_vram_per_gpu': 18.5,   # GB, leaves 1.5GB headroom
}

# Use Case 2: Fine-tuning 3B LLM (smaller model)
config_3b_finetune = {
    'batch_size_per_gpu': 16,
    'gradient_accumulation_steps': 2,
    'max_sequence_length': 4096,
    'use_mixed_precision': True,
    'use_lora': True,                  # Low-Rank Adaptation - cuts memory 4-10x
    'lora_rank': 16,
    'lora_alpha': 32,
    'estimated_vram_per_gpu': 12.0,    # Much lower with LoRA
}

# Use Case 3: Multi-GPU DDP training (both GPUs)
config_ddp_training = {
    'batch_size_per_gpu': 6,           # Small due to PCIe communication overhead
    'gradient_accumulation_steps': 8,  # Effective batch: 6 * 2 * 8 = 96
    'max_sequence_length': 2048,
    'use_mixed_precision': True,
    'ddp_find_unused_parameters': False,
    'ddp_bucket_cap_mb': 100,          # Reduce for consumer GPUs
    'num_gpus': 2,
    'expected_scaling_efficiency': 0.85,  # Not perfect due to PCIe overhead
}
```

## System Architecture

### Consumer GPU Multi-GPU Topology

```
Consumer System with Dual RX 7900 XT:
┌─────────────────────────────────────────────────────────┐
│                    CPU Host System                       │
│              (Ryzen 7000/9000 recommended)              │
├─────────────────────────────────────────────────────────┤
│  CPU-Attached PCIe Root Complex (Gen 4)                 │
├──────────────────────┬──────────────────────────────────┤
│                      │                                   │
│   PCIe Bus 1 (x8)    │   PCIe Bus 2 (x8)                │
│   (9GB/s per dir)    │   (9GB/s per dir)                │
│                      │                                   │
├──────────┐           │            ┌──────────┐           │
│ GPU0     │           │            │ GPU1     │           │
│ RX 7900XT│◄──────────┤────────────►RX 7900XT│           │
│ 20GB     │           │            │ 20GB    │           │
│          │           │            │         │           │
└──────────┘           │            └─────────┘           │
                       │
              CPU Memory Bus (128 GB/s)

Key Points:
- NO direct GPU-to-GPU interconnect (unlike datacenter MI300X with Infinity Fabric)
- ALL GPU communication flows through CPU/PCIe
- Each GPU gets x8 lanes (down from x16 in single-GPU systems)
- Theoretical PCIe Gen 4 x8 = 8 GB/s, practical ~6.5 GB/s
- This is the primary performance bottleneck for multi-GPU training
```

### Software Stack for Consumer GPUs

```
Application Layer (User Code):
├── PyTorch 2.4+ (with ROCm 7.0 wheels)
├── HuggingFace Transformers
├── vLLM / Text Generation WebUI
└── Custom Training Scripts

ROCm Runtime & Libraries:
├── HIP Runtime (GPU programming interface)
├── rocBLAS (Linear algebra, up to ~55% of MI300X speed)
├── MIOpen (Deep learning primitives)
├── rocFFT, rocSPARSE, rocRAND
└── HIP Memory Management

Communication Layer:
├── RCCL (for multi-GPU coordination, PCIe-based)
├── UCX (optional, higher overhead on consumer PCIe)
└── gloo (optional, CPU-based fallback)

Driver & Kernel Layer:
├── AMD amdgpu Driver (v6.14.14 tested, v6.16+ recommended)
├── KFD (Kernel Fusion Driver)
├── HIP Runtime Management
└── PCIe DMA Engine Control
```

## Installation and Setup

### Prerequisites for Consumer GPU Setup

```
System Requirements:
├── OS: Ubuntu 22.04/24.04 LTS recommended (best tested)
├── Kernel: 6.8+ recommended (6.14+ for best compatibility)
├── CPU: Ryzen 7000/9000 series or similar x86_64 with PCIe Gen 4
├── RAM: 32GB minimum, 64GB recommended
├── PCIe: Motherboard with dual PCIe x16 slots (running at x8/x8 mode)
├── Power Supply: 1000W+ 80+ Bronze minimum for dual RX 7900 XT
└── Cooling: Dual 120mm+ fans per GPU, case with good airflow
```

### Step 1: Install ROCm 7.0 for Consumer GPUs

```bash
# For Ubuntu 22.04/24.04

# 1. Add AMD ROCm repository
wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -

# For Ubuntu 22.04:
echo "deb [arch=amd64] https://repo.radeon.com/rocm/apt/7.0 jammy main" | \
  sudo tee /etc/apt/sources.list.d/rocm.list

# For Ubuntu 24.04 (Noble):
echo "deb [arch=amd64] https://repo.radeon.com/rocm/apt/7.0 noble main" | \
  sudo tee /etc/apt/sources.list.d/rocm.list

# 2. Update and install ROCm
sudo apt update
sudo apt install -y \
  rocm-core \
  rocm-dkms \
  rocm-dev \
  rocm-libs \
  miopen-hip \
  rocblas \
  hipcub \
  hipfft \
  hipsparse \
  rocrand \
  rccl \
  rocm-utils \
  rocm-hip-runtime \
  rocm-hip-runtime-dev

# 3. Install Radeon Software for Linux (driver support for consumer GPUs)
# This is critical for consumer GPU support in ROCm 7.0
wget https://repo.radeon.com/amdgpu-install/7.1/ubuntu/noble/amdgpu-install_7.1.70100-1_all.deb
sudo apt install ./amdgpu-install_7.1.70100-1_all.deb
sudo amdgpu-install -y --usecase=graphics,rocm --rocmrelease=7.1

# 4. Add user to groups
sudo usermod -a -G render,video,kvm $USER

# 5. Verify installation
amd-smi
# Should show both RX 7900 XT GPUs with full capabilities
```

### Step 2: Configure System for Multi-GPU Training

```bash
# 1. Verify PCIe configuration (should show x8/x8 or x16)
lspci | grep VGA
lspci -vv | grep -A5 "RX 7900"

# 2. Set PCIe Gen 4 mode if needed (check BIOS)
# Most modern boards auto-negotiate correctly, but verify:
lspci -vv | grep "LnkCap:\|LnkSta:"
# Should show: LnkSta: Speed 16GT/s, Width x8 (or similar)

# 3. Configure kernel modules for GPU access
echo 'options amdgpu ppfeaturemask=0xffffffff' | \
  sudo tee /etc/modprobe.d/amdgpu.conf

# 4. Configure for ROCm HIP
echo 'export ROCM_PATH=/opt/rocm' >> ~/.bashrc
echo 'export HIP_PATH=/opt/rocm/hip' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export PATH=$ROCM_PATH/bin:$PATH' >> ~/.bashrc

# 5. Consumer GPU specific variables
echo 'export HSA_OVERRIDE_GFX_VERSION=11.0.0' >> ~/.bashrc  # Critical for RX 7900 XT
echo 'export HIP_VISIBLE_DEVICES=0,1' >> ~/.bashrc

source ~/.bashrc

# 6. Verify GPU detection
rocminfo | grep "Name:"  # Should show both GPUs
amd-smi                   # Should detect both RX 7900 XTs
```

### Step 3: Install PyTorch with ROCm 7.0 Support

```bash
# Create virtual environment
python3 -m venv ~/rocm_ml_env
source ~/rocm_ml_env/bin/activate

# Install PyTorch with ROCm 7.0 support (consumer GPU compatible)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
# Note: As of late 2025, use rocm5.7 wheels; ROCm 7.0 specific wheels may follow

# Alternative: Install nightly builds with latest ROCm 7.0 support
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm5.8

# Install distributed training frameworks
pip install transformers datasets accelerate bitsandbytes
pip install deepspeed  # Optional, use selectively for consumer GPUs
pip install wandb tensorboard  # Monitoring

# Verify PyTorch ROCm support
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
# Should output: True, 2
```

### Step 4: Optimize System for Sustained Training

```bash
# 1. Set GPU performance governors
echo "performance" | sudo tee /sys/class/drm/card0/device/power_dpm_force_performance_level
echo "performance" | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level

# 2. Disable C-states for stability (optional, increases power/heat)
echo 1 | sudo tee /sys/module/amd_pstate/parameters/shared_memory

# 3. Monitor temperatures
# Create thermal monitoring script
cat > ~/monitor_temps.sh << 'EOF'
#!/bin/bash
while true; do
    amd-smi metric | grep -E "GPU|Temp"
    sleep 5
done
EOF
chmod +x ~/monitor_temps.sh

# 4. Set thermal throttle limits (optional)
# Keep these conservative: 85-90°C max for RX 7900 XT
# Most cards thermally throttle around 95-110°C depending on BIOS

# 5. Create systemd service to apply settings on boot
sudo tee /etc/systemd/system/rocm-performance-mode.service > /dev/null << 'EOF'
[Unit]
Description=Set ROCm GPUs to Performance Mode
After=network.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/bin/bash -c 'echo performance | tee /sys/class/drm/card0/device/power_dpm_force_performance_level; echo performance | tee /sys/class/drm/card1/device/power_dpm_force_performance_level'

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable rocm-performance-mode
```

## Multi-GPU Configuration

### GPU Topology Discovery for Consumer Systems

```bash
# 1. Check GPU topology
rocm-smi --showtopo

# Example output for dual RX 7900 XT:
# GPU topology
# GPU0    GPU1    CPU Affinity    NUMA Affinity
# GPU0    X       PIX             0-15            0
# GPU1    PIX     X               16-31           0
# 
# Legend:
# X   = Self
# PIX = Connection through PCIe root complex (consumer GPU typical)
# SYS = Further NUMA distance

# 2. Check memory bandwidth between GPUs
# This script measures actual PCIe throughput
cat > ~/test_gpu_interconnect.py << 'EOF'
import torch
import time

# Test GPU interconnect bandwidth
device0 = torch.device('cuda:0')
device1 = torch.device('cuda:1')

# Create large tensor
size_mb = 256
tensor_size = size_mb * 256 * 1024  # MB to elements (4 bytes per float32)
tensor = torch.randn(tensor_size, device=device0)

# Warm up
for _ in range(5):
    tensor_copy = tensor.to(device1)

# Time transfer
torch.cuda.synchronize()
start = time.time()
for _ in range(10):
    tensor_copy = tensor.to(device1)
    torch.cuda.synchronize()
elapsed = time.time() - start

bandwidth_gb_s = (size_mb * 10) / elapsed / 1000
print(f"GPU0 -> GPU1 Bandwidth: {bandwidth_gb_s:.2f} GB/s")
print(f"Expected theoretical max: ~6.5 GB/s (PCIe Gen 4 x8)")
print(f"Efficiency: {bandwidth_gb_s/6.5*100:.1f}%")
EOF

python ~/test_gpu_interconnect.py
# Expected: 5.5-6.5 GB/s for PCIe Gen 4 x8
```

### Memory Management for Dual Consumer GPUs

```python
#!/usr/bin/env python3
"""
Memory optimization specifically for RX 7900 XT consumer GPUs
"""

import torch
import os

def setup_consumer_gpu_memory():
    """Setup memory management for consumer GPUs"""
    
    # Get GPU info
    print(f"Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  Total Memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"  SM Count: {props.multi_processor_count}")
    
    # Set memory management strategy
    # For RX 7900 XT: use async memory management to handle PCIe transfers better
    torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of available
    
    # Enable memory growth (PyTorch default for consumer GPUs)
    torch.cuda.empty_cache()
    
    return True

def estimate_batch_size(model_params_millions, per_gpu_memory_gb=18.0):
    """
    Estimate optimal batch size for consumer GPU
    
    Formula (approximate):
    Memory = (params * 2 + activations * batch_size * seq_len * layers)
    
    For consumer GPUs with PCIe constraints:
    - Reduce batch size to 60-70% of what datacenter GPUs would use
    - Keep batch_size per GPU at 4-16 to balance communication overhead
    """
    
    # Conservative estimate: ~2.5 bytes per parameter (gradients + optimizer state + activations)
    param_memory_gb = model_params_millions * 2.5 / 1024
    
    # Reserve headroom for activations and temporary buffers
    available_for_batch = per_gpu_memory_gb - param_memory_gb - 2.0  # 2GB headroom
    
    # Typical consumer GPU training: 3-5KB per sequence token per layer
    if model_params_millions <= 3000:      # 3B model
        recommended_batch = 16
    elif model_params_millions <= 7000:    # 7B model
        recommended_batch = 8
    elif model_params_millions <= 13000:   # 13B model
        recommended_batch = 4
    else:
        recommended_batch = 2
    
    return recommended_batch

# Example usage
setup_consumer_gpu_memory()
batch_size = estimate_batch_size(model_params_millions=7000)
print(f"Recommended batch size per RX 7900 XT for 7B model: {batch_size}")
```

## Distributed Training Components

### Data Parallel (DDP) for Dual Consumer GPUs

```python
#!/usr/bin/env python3
"""
Distributed Data Parallel (DDP) training optimized for consumer RX 7900 XT GPUs

Key differences from datacenter training:
1. Lower per-GPU batch size (4-8 vs 32-64 for MI300X)
2. More frequent gradient synchronization due to PCIe bandwidth limits
3. Higher relative overhead of gradient communication
4. No direct GPU-GPU interconnect
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import os

class ConsumerGPUDDPTrainer:
    """DDP trainer optimized for consumer GPUs (RX 7900 XT)"""
    
    def __init__(self, model, config=None):
        self.model = model
        self.config = config or {}
        self.setup_distributed()
        
    def setup_distributed(self):
        """Setup distributed training for consumer GPUs"""
        
        # Get distributed rank info
        self.rank = int(os.environ.get('RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        if self.world_size > 1:
            # For consumer GPUs, use gloo backend (more stable than nccl on PCIe)
            # or hccl with conservative settings
            dist.init_process_group(
                backend='hccl',  # ROCm's RCCL backend
                rank=self.rank,
                world_size=self.world_size,
                timeout=torch.distributed.Timeout(seconds=3600)
            )
        
        # Set device
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device(f'cuda:{self.local_rank}')
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Wrap with DDP
        if self.world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
                
                # Consumer GPU specific settings:
                bucket_cap_mb=100,  # Reduce bucket size for more frequent sync
                gradient_as_bucket_view=True,  # Memory efficient
                
                # Disable features that don't work well with PCIe:
                # - No NVLink equivalent on consumer GPUs
                # - Broadcast reduction optimization may not help
            )
            
            print(f"Rank {self.rank}: DDP initialized with {self.world_size} GPUs")
        else:
            print("Running on single GPU (or no distributed setup)")
        
        self.world_size = world_size if self.world_size > 1 else 1
    
    def create_dataloader(self, dataset, batch_size=4, num_workers=4):
        """Create dataloader optimized for consumer GPUs"""
        
        # Use small batch size for consumer GPUs
        batch_size = self.config.get('batch_size_per_gpu', batch_size)
        
        if self.world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=True
            )
            shuffle = False
        else:
            sampler = None
            shuffle = True
        
        # Reduce num_workers for consumer GPUs to minimize CPU overhead
        num_workers = min(num_workers, 4)
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=2  # Smaller prefetch for consumer GPUs
        )
        
        return dataloader
    
    def train_step(self, batch, optimizer, loss_fn, use_amp=True):
        """Single training step optimized for consumer GPUs"""
        
        data, target = batch
        data = data.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Mixed precision for memory efficiency (critical for consumer GPUs)
        if use_amp:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                output = self.model(data)
                loss = loss_fn(output, target)
            
            # Use gradient scaler for mixed precision
            # (initialized outside, used here)
            loss = loss / 4  # Gradient accumulation
            loss.backward()
        else:
            output = self.model(data)
            loss = loss_fn(output, target)
            (loss / 4).backward()
        
        return loss.detach()

def run_ddp_training_consumer_gpu():
    """Example DDP training loop for dual RX 7900 XT"""
    
    # Configuration for consumer GPUs
    config = {
        'batch_size_per_gpu': 4,           # Small due to PCIe overhead
        'num_epochs': 10,
        'learning_rate': 1e-4,
        'gradient_accumulation_steps': 4,  # Effective batch = 4 * 2 GPUs * 4 = 32
        'use_mixed_precision': True,
    }
    
    # Create simple model
    model = nn.Sequential(
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    
    # Setup trainer
    trainer = ConsumerGPUDDPTrainer(model, config)
    
    # Create dummy dataset
    dataset = torch.utils.data.TensorDataset(
        torch.randn(1000, 784),
        torch.randint(0, 10, (1000,))
    )
    
    # Create dataloader
    dataloader = trainer.create_dataloader(
        dataset,
        batch_size=config['batch_size_per_gpu']
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=config['learning_rate'])
    loss_fn = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(config['num_epochs']):
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            loss = trainer.train_step(batch, optimizer, loss_fn, use_amp=config['use_mixed_precision'])
            
            # Optimizer step every gradient_accumulation_steps
            if (batch_idx + 1) % config['gradient_accumulation_steps'] == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item()
        
        if trainer.rank == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")

if __name__ == "__main__":
    # Run with: torchrun --nproc_per_node=2 script.py
    run_ddp_training_consumer_gpu()
```

## Performance Optimization for Consumer GPUs

### Memory Optimization Strategies

**Strategy 1: Mixed Precision Training (Recommended, Essential)**

```python
from torch.cuda.amp import autocast, GradScaler

# Setup mixed precision
scaler = GradScaler()

# Training loop with mixed precision
for batch_idx, (data, target) in enumerate(dataloader):
    data, target = data.to(device), target.to(device)
    
    optimizer.zero_grad()
    
    # Forward pass in FP16
    with autocast(dtype=torch.float16):
        output = model(data)
        loss = criterion(output, target)
    
    # Backward and optimize
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)  # Unscale gradients before clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

# Memory savings: ~40-50% reduction
# Speed improvement: 1.2-1.5x faster due to reduced memory bandwidth
# Trade-off: Slight accuracy loss (usually negligible with proper scaling)
```

**Strategy 2: Gradient Accumulation**

```python
# Simulate larger batch size without larger memory footprint
accumulation_steps = 4
effective_batch = batch_size * num_gpus * accumulation_steps

optimizer.zero_grad()
for step in range(accumulation_steps):
    batch = next(dataloader)
    output = model(batch)
    loss = criterion(output, target) / accumulation_steps
    loss.backward()  # Gradient accumulates

optimizer.step()
```

**Strategy 3: Gradient Checkpointing**

```python
from torch.utils.checkpoint import checkpoint

class GradientCheckpointedModel(nn.Module):
    def forward(self, x):
        # Only store activations for last layer, recompute others
        x = checkpoint(self.layer1, x)
        x = checkpoint(self.layer2, x)
        x = checkpoint(self.layer3, x)
        return x

# Memory savings: 30-40%
# Speed trade-off: 10-20% slower due to recomputation
```

**Strategy 4: Flash Attention (if supported)**

```python
# Use optimized attention implementation
# Reduces memory from O(N²) to O(N)
try:
    from flash_attn import flash_attn_func
    # Use flash_attn in your model
except ImportError:
    # Fallback to standard attention
    pass

# Memory savings: 5-10x for long sequences
# Speed improvement: 2-4x faster
```

### Gradient Accumulation for Effective Large Batches

```python
"""
For RX 7900 XT with 20GB memory:
- Typical batch per GPU: 4-8
- Gradient accumulation steps: 4-8
- Effective batch: 32-128

This simulates training with large batches without the memory cost.
"""

def train_with_accumulation(model, dataloader, num_accumulation_steps=4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data = data.to(device)
        target = target.to(device)
        
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Scale loss by accumulation steps
        scaled_loss = loss / num_accumulation_steps
        scaled_loss.backward()
        
        # Update only after accumulating gradients
        if (batch_idx + 1) % num_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)
```

### Communication Overhead Mitigation

```python
"""
PCIe bandwidth is the bottleneck for consumer GPU DDP.
Strategies to minimize communication:

1. Larger per-GPU batch sizes (less frequent sync)
2. Fewer communication operations
3. Communication-computation overlap
4. Reduced precision for communication (FP16/BF16)
"""

def setup_rccl_for_consumer_gpu():
    """Configure RCCL for consumer GPU PCIe communication"""
    
    import os
    
    # RCCL settings optimized for PCIe
    os.environ['RCCL_DEBUG'] = 'INFO'
    os.environ['RCCL_TREE_THRESHOLD'] = '0'  # Use tree algorithm for all sizes
    os.environ['RCCL_BUFFSIZE'] = '8388608'  # 8MB buffers
    os.environ['RCCL_MAX_NCHANNELS'] = '4'   # Reduce channels for PCIe
    os.environ['RCCL_NCHANNELS'] = '4'
    
    # For HCCL (AMD's RCCL)
    os.environ['HCCL_DEBUG'] = 'INFO'
```

## Framework Integration

### PyTorch Best Practices for Consumer GPUs

```python
#!/usr/bin/env python3
"""
Complete training example for RX 7900 XT optimized for consumer GPUs
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset
import time

class RX7900XTTrainer:
    """Trainer class optimized for RX 7900 XT consumer GPU"""
    
    def __init__(self, model, lr=1e-4, use_amp=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        
        self.use_amp = use_amp
        if use_amp:
            self.scaler = GradScaler()
        else:
            self.scaler = None
        
        self.device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        print(f"Using device: {self.device_name}")
        print(f"Total memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    def train_epoch(self, dataloader, accumulation_steps=4):
        """Train one epoch with gradient accumulation"""
        self.model.train()
        total_loss = 0
        num_batches = len(dataloader)
        
        self.optimizer.zero_grad()
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast(dtype=torch.float16):
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                # Scale loss by accumulation steps
                scaled_loss = loss / accumulation_steps
                self.scaler.scale(scaled_loss).backward()
                
                # Update weights
                if (batch_idx + 1) % accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                output = self.model(data)
                loss = self.criterion(output, target)
                (loss / accumulation_steps).backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 100 == 0:
                print(f"Batch {batch_idx + 1}/{num_batches}, Loss: {loss.item():.4f}")
        
        return total_loss / num_batches
    
    def validate(self, dataloader):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in dataloader:
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                
                with autocast(dtype=torch.float16) if self.use_amp else torch.no_grad():
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(dataloader)
        
        return avg_loss, accuracy

# Example training script
def main():
    # Create simple model
    model = nn.Sequential(
        nn.Linear(784, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, 10)
    )
    
    # Create trainer
    trainer = RX7900XTTrainer(model, lr=1e-4, use_amp=True)
    
    # Create dummy dataset
    X_train = torch.randn(10000, 784)
    y_train = torch.randint(0, 10, (10000,))
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    
    # Training loop
    num_epochs = 5
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        avg_loss = trainer.train_epoch(train_loader, accumulation_steps=4)
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}, Time = {epoch_time:.1f}s")
    
    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.1f}s")
    print(f"Time per epoch: {total_time / num_epochs:.1f}s")
    
    # Report memory usage
    print(f"\nMemory usage: {torch.cuda.memory_allocated() / 1024**3:.1f}GB / "
          f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

if __name__ == "__main__":
    main()
```

## Consumer GPU Best Practices

### Do's and Don'ts for RX 7900 XT Training

**DO:**
- ✓ Use mixed precision (BF16/FP16) - essential for memory efficiency
- ✓ Use gradient accumulation to simulate larger batches
- ✓ Enable flash attention if available (huge memory/speed win)
- ✓ Use LoRA or QLoRA for large model fine-tuning (10x memory reduction)
- ✓ Profile your training to identify bottlenecks
- ✓ Monitor GPU temperatures (throttle starts ~85°C, force limit ~95°C)
- ✓ Keep batch sizes moderate (4-16 per GPU)
- ✓ Use torch.compile for 10-20% speed improvements
- ✓ Consider data parallel for multiple GPUs
- ✓ Pin memory in dataloaders (pin_memory=True)

**DON'T:**
- ✗ Don't use full precision (FP32) - wastes memory and slower
- ✗ Don't use DeepSpeed Stage 3 - communication overhead too high for PCIe
- ✗ Don't expect perfect scaling with 2 GPUs (expect ~1.8-1.9x vs theoretical 2x)
- ✗ Don't run sustained loads above 90°C - thermal throttling reduces performance 30%+
- ✗ Don't use find_unused_parameters=True in DDP - high overhead
- ✗ Don't train very large models (>30B) on single RX 7900 XT
- ✗ Don't ignore communication bottlenecks - they're ~30% of training time on dual GPU setup
- ✗ Don't use too many workers in DataLoader (4-8 workers max)
- ✗ Don't expect DeepSpeed Zero-3 to work well without high-speed interconnect
- ✗ Don't forget to set HSA_OVERRIDE_GFX_VERSION=11.0.0 environment variable

### Recommended Model Sizes and Settings

```python
CONSUMER_GPU_CONFIG = {
    # 3B Parameter Model (e.g., Mistral 3B, Phi 3)
    "3B": {
        "batch_size_per_gpu": 16,
        "gradient_accumulation": 1,
        "use_lora": True,  # Optional, further memory reduction
        "max_seq_len": 4096,
        "max_vram_per_gpu": 12,  # GB
        "training_throughput": "50-60 tokens/sec/GPU"
    },
    
    # 7B Parameter Model (e.g., Llama 7B, Mistral 7B)
    "7B": {
        "batch_size_per_gpu": 8,
        "gradient_accumulation": 4,
        "use_lora": True,  # Recommended for consumer GPU
        "use_flash_attention": True,
        "max_seq_len": 2048,
        "max_vram_per_gpu": 18.5,  # GB, near max for RX 7900 XT
        "training_throughput": "30-40 tokens/sec/GPU",
        "suggested_strategy": "LoRA fine-tuning on consumer GPU"
    },
    
    # 13B Parameter Model
    "13B": {
        "batch_size_per_gpu": 4,
        "gradient_accumulation": 8,
        "use_lora": True,  # Essential
        "use_flash_attention": True,
        "quantization": "8-bit",  # Consider bitsandbytes
        "max_seq_len": 1024,
        "max_vram_per_gpu": 19.5,
        "training_throughput": "15-20 tokens/sec/GPU",
        "suggested_strategy": "LoRA + QLoRA (quantization)"
    },
    
    # Dual RX 7900 XT Setup (40GB total)
    "DDP_7B": {
        "batch_size_per_gpu": 6,  # Smaller due to communication overhead
        "gradient_accumulation": 2,
        "use_flash_attention": True,
        "num_gpus": 2,
        "expected_scaling_efficiency": 0.85,  # Not perfect due to PCIe
        "total_throughput": "85-110 tokens/sec",
        "training_throughput_per_gpu": "42-55 tokens/sec/GPU",
        "suggested_use_case": "Faster training of standard models, multi-task training"
    }
}
```

### Thermal Management

```bash
# Monitor GPU temperatures and frequencies during training
watch -n 1 'amd-smi metric | grep -E "GPU|Temp|Power|SCLK|MCLK"'

# Typical thermal profile for RX 7900 XT during sustained training:
# Idle: 25-35°C
# Light Load: 40-50°C
# Full Training Load: 65-85°C (normal)
# Thermal Throttle: 95°C (power reduced ~15-30%)
# Forced Shutdown: ~110°C

# If temperatures exceed 85°C, you can:
# 1. Reduce batch size (less compute, less heat)
# 2. Increase fan speed (BIOS setting or manual control)
# 3. Improve case airflow
# 4. Reduce clock speeds (amdgpu driver settings)
# 5. Take breaks between training sessions
```

## Troubleshooting

### Common Issues and Solutions

**Issue 1: "GPU out of memory" despite reported available VRAM**

```python
# Solution: Memory fragmentation - clear and reset
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# Check actual usage:
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
print(f"Cached: {torch.cuda.memory_reserved() / 1024**3:.2f}GB")
```

**Issue 2: Slow multi-GPU training (expected 2x, getting 1.5x speedup)**

```python
# This is normal for consumer GPUs due to PCIe limitations.
# Expected scaling:
# - 1 GPU: Baseline
# - 2 GPUs via PCIe Gen 4 x8/x8: 1.8-1.9x (10-20% overhead)
# - 2 GPUs via Infinity Fabric (datacenter): 1.95-2.0x (near-perfect)

# To optimize:
# 1. Reduce communication frequency: increase batch size per GPU
# 2. Use communication/computation overlap
# 3. Verify PCIe is running at x8 (not x4): lspci -vv | grep "LnkSta"
```

**Issue 3: Model not detected / HSA_OVERRIDE_GFX_VERSION errors**

```bash
# Solution: Ensure proper environment setup
export HSA_OVERRIDE_GFX_VERSION=11.0.0  # Critical for RX 7900 XT
export ROCM_PATH=/opt/rocm
export HIP_VISIBLE_DEVICES=0  # or 0,1 for both

# Verify:
rocminfo | grep "Name:"
amd-smi
```

**Issue 4: PyTorch ROCm wheels not working**

```bash
# Solution: Verify PyTorch ROCm support
python -c "import torch; print(torch.cuda.is_available()); print(torch.__version__)"

# If not working, reinstall with correct wheel:
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7

# Or for latest nightly:
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm5.8
```

## Conclusion

ROCm 7.0 brings production-grade support for consumer Radeon GPUs (RX 7900 XT/XTX), enabling cost-effective distributed ML training on consumer hardware. While performance scales differently than datacenter setups due to PCIe interconnects, modern optimization techniques (mixed precision, gradient accumulation, LoRA) make training large models practical on consumer systems.

**Key Takeaways for RX 7900 XT Training:**
1. Always use mixed precision (BF16/FP16) - non-negotiable for 20GB cards
2. Gradient accumulation is your friend for effective larger batches
3. Dual GPU setups achieve ~1.8-1.9x speedup (not perfect 2x due to PCIe)
4. Set HSA_OVERRIDE_GFX_VERSION=11.0.0 for proper GPU detection
5. Monitor temperatures - throttling starts at 85°C
6. Use LoRA for large model fine-tuning (~10x memory reduction)
7. Profile your training - find and optimize bottlenecks
8. Expected throughput: 30-60 tokens/sec for 7B models on single GPU

For more details, refer to official AMD ROCm documentation:
- https://rocm.docs.amd.com/
- https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/
- https://github.com/RadeonOpenCompute/ROCm/
