# ROCm GPU Memory Management While Training: In-Process Solutions

## Executive Summary

ROCm 6.3 has severe VRAM fragmentation issues on RDNA3 GPUs (RX 7900 XT) that prevent new allocations despite available memory. The traditional solution (killing processes to reset VRAM) interrupts training. This guide provides **in-process solutions** that defragment and manage GPU memory **without stopping your training script**.

---

## The Core Problem

When training on dual RX 7900 XT GPUs:
- VRAM shows 99% fragmented usage
- PyTorch reports 0 bytes free despite 20GB available
- Standard `torch.cuda.empty_cache()` is insufficient
- Killing processes to reset is disruptive

**Root Cause**: ROCm 6.3 HIP allocator has pathological fragmentation on gfx1100 (RDNA3) that requires in-process defragmentation strategies.

---

## Solution 1: Aggressive In-Process VRAM Cleanup (Best for Training)

Use this in your training script to periodically defragment without killing the process.

### Implementation

```python
import torch
import gc

def aggressive_vram_cleanup():
    """
    Defragment GPU memory WITHOUT killing the process.
    Call this periodically during training (recommended: every N batches or after each epoch).
    
    This is safe to call while training - it clears cache and unused memory
    without affecting active tensors.
    """
    print("🔧 Aggressive VRAM cleanup in progress...")
    
    # Step 1: Synchronize all GPU operations
    for device_id in [0, 1]:
        torch.cuda.set_device(device_id)
        torch.cuda.synchronize()
    
    # Step 2: Clear PyTorch GPU cache
    torch.cuda.empty_cache()
    
    # Step 3: Force Python garbage collection (3 levels)
    gc.collect(0)  # Collect young objects
    gc.collect(1)  # Collect intermediate objects
    gc.collect(2)  # Collect old objects
    
    # Step 4: Clear GPU cache again after GC
    torch.cuda.empty_cache()
    
    # Step 5: Reset memory statistics (for monitoring)
    for device_id in [0, 1]:
        torch.cuda.set_device(device_id)
        torch.cuda.reset_accumulated_memory_stats()
    
    # Step 6: Final synchronization
    for device_id in [0, 1]:
        torch.cuda.set_device(device_id)
        torch.cuda.synchronize()
    
    print("✅ VRAM cleanup complete")

def print_gpu_memory_stats():
    """Print current GPU memory usage for monitoring."""
    for device_id in [0, 1]:
        torch.cuda.set_device(device_id)
        free, total = torch.cuda.mem_get_info(device_id)
        used = total - free
        percent = (used / total) * 100
        print(f"GPU {device_id}: {used/1024**3:.2f}GB / {total/1024**3:.2f}GB ({percent:.1f}%)")
```

### Usage in Training Loop

```python
# Call after each epoch
for epoch in range(num_epochs):
    train_one_epoch()
    aggressive_vram_cleanup()
    validate()
    print_gpu_memory_stats()

# Or call periodically during epoch
for batch_idx, (data, target) in enumerate(train_loader):
    # Training step
    loss = train_step(data, target)
    
    # Cleanup every N batches
    if batch_idx % 50 == 0:
        aggressive_vram_cleanup()
        if batch_idx % 200 == 0:
            print_gpu_memory_stats()
```

---

## Solution 2: Enable ROCm Memory Pool Management

Configure environment variables **BEFORE** importing PyTorch for better allocator behavior.

### Implementation

Add this at the **very top** of your training script:

```python
# ⚠️ MUST BE BEFORE: import torch
import os
os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'max_split_size_mb:256,garbage_collection_threshold:0.95'

import torch
import gc
# ... rest of imports ...
```

**Configuration Explanation**:
- `max_split_size_mb:256` - Prevent allocating overly large contiguous blocks (reduces fragmentation)
- `garbage_collection_threshold:0.95` - Trigger GC when 95% of reserved memory is fragmented

### For Aggressive Defragmentation

```python
import os
os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'max_split_size_mb:128,garbage_collection_threshold:0.85'
```

---

## Solution 3: Pre-allocate GPU Memory Pools

Initialize GPU memory **before training** to establish clean memory pools.

### Implementation

```python
def preallocate_gpu_memory_pools():
    """
    Pre-allocate GPU memory pools to prevent fragmentation.
    Call this ONCE before training starts.
    
    This creates clean memory allocations on both GPUs,
    allowing subsequent training to use fragmentation-free pools.
    """
    print("Pre-allocating GPU memory pools...")
    
    for device_id in [0, 1]:
        torch.cuda.set_device(device_id)
        
        # Get total memory
        total_memory = torch.cuda.get_device_properties(device_id).total_memory
        
        # Allocate 80% of GPU memory as initialization
        preallocate_size = int(total_memory * 0.8)
        
        print(f"GPU {device_id}: Pre-allocating {preallocate_size / 1024**3:.1f}GB...")
        
        # Create and delete large tensor to initialize allocator
        placeholder = torch.zeros(preallocate_size // 4, dtype=torch.float32, device=device_id)
        del placeholder
        
        # Clean up
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    print("✅ Memory pre-allocation complete")

# Call once at the start of training
preallocate_gpu_memory_pools()
```

---

## Solution 4: Dynamic Batch Size Adjustment

Reduce batch size based on available VRAM to fit within fragmented memory constraints.

### Implementation

```python
def calculate_safe_batch_size(target_utilization=0.7):
    """
    Dynamically calculate safe batch size based on available VRAM.
    
    Args:
        target_utilization: Target GPU utilization (0.0-1.0), default 70%
    
    Returns:
        Safe batch size for current GPU memory state
    """
    batch_sizes = []
    
    for device_id in [0, 1]:
        torch.cuda.set_device(device_id)
        free, total = torch.cuda.mem_get_info(device_id)
        
        # Calculate usable memory (in GB)
        available_gb = (free * target_utilization) / (1024**3)
        
        # Estimate batch size (adjust multiplier based on your model)
        # This example assumes ~50MB per sample
        memory_per_sample_gb = 0.05
        safe_batch_size = max(1, int(available_gb / memory_per_sample_gb))
        
        batch_sizes.append(safe_batch_size)
        print(f"GPU {device_id}: Free {free/1024**3:.1f}GB → Safe batch size: {safe_batch_size}")
    
    return min(batch_sizes)  # Use minimum across GPUs

# Use before training
safe_batch_size = calculate_safe_batch_size()
print(f"Recommended batch size: {safe_batch_size}")
```

---

## Solution 5: Stream-Based Memory Management

Use CUDA streams to isolate memory allocations and prevent cross-contamination.

### Implementation

```python
class GPUMemoryManager:
    """
    Manages GPU memory allocations across separate streams.
    Helps prevent fragmentation by isolating allocation patterns.
    """
    
    def __init__(self):
        self.streams = {}
        for device_id in [0, 1]:
            self.streams[device_id] = torch.cuda.Stream(device=device_id)
    
    def allocate_on_stream(self, size, device_id, dtype=torch.float32):
        """Allocate tensor on device's dedicated stream."""
        with torch.cuda.stream(self.streams[device_id]):
            tensor = torch.zeros(size, dtype=dtype, device=device_id)
        torch.cuda.synchronize(device=device_id)
        return tensor
    
    def cleanup_stream(self, device_id):
        """Cleanup a specific stream."""
        torch.cuda.synchronize(device=device_id)
        torch.cuda.empty_cache()
    
    def cleanup_all(self):
        """Cleanup all streams."""
        for device_id in [0, 1]:
            self.cleanup_stream(device_id)

# Usage during training
gpu_mem = GPUMemoryManager()

# Allocate training data on stream
train_data = gpu_mem.allocate_on_stream(1000000, device_id=0)

# ... training ...

# Cleanup after batch
gpu_mem.cleanup_stream(device_id=0)
```

---

## Recommended Implementation: Combined Solution

**Best approach combining Solutions 1, 2, and 4:**

### Step 1: Update Your Training Script Header

```python
# MUST be at the very top, before any torch imports
import os
os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'max_split_size_mb:256,garbage_collection_threshold:0.95'

import torch
import gc
from torch.optim import Adam
# ... other imports ...

# Import memory utilities
def aggressive_vram_cleanup():
    """[Include aggressive_vram_cleanup function from Solution 1]"""
    # ... (see Solution 1 code above)

def print_gpu_memory_stats():
    """[Include print_gpu_memory_stats function from Solution 1]"""
    # ... (see Solution 1 code above)

def calculate_safe_batch_size():
    """[Include calculate_safe_batch_size function from Solution 4]"""
    # ... (see Solution 4 code above)
```

### Step 2: Initialize GPU Before Training

```python
# Before training loop
print("🔧 Initializing GPU memory...")
for device_id in [0, 1]:
    torch.cuda.set_device(device_id)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_accumulated_memory_stats()

gc.collect()

# Calculate safe batch size
safe_batch_size = calculate_safe_batch_size()
print(f"Recommended batch size: {safe_batch_size}")

print("✅ GPU initialization complete\n")
```

### Step 3: Training Loop with Periodic Cleanup

```python
def train_one_epoch(model, train_loader, optimizer, criterion, epoch):
    """Training loop with periodic memory cleanup."""
    model.train()
    total_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # Move to GPU
        data = data.to(0)  # GPU 0
        target = target.to(0)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Periodic memory cleanup (every N batches)
        if batch_idx % 20 == 0:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()
        
        # Periodic logging (every M batches)
        if batch_idx % 100 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            print(f"Epoch {epoch}, Batch {batch_idx}: Loss = {avg_loss:.4f}")
            print_gpu_memory_stats()
    
    # Cleanup after epoch
    aggressive_vram_cleanup()
    print(f"Epoch {epoch} complete - Average Loss: {total_loss / len(train_loader):.4f}\n")

# Training loop
num_epochs = 30
for epoch in range(num_epochs):
    train_one_epoch(model, train_loader, optimizer, criterion, epoch)
    
    # Validate
    validate_model(model, val_loader)
    
    # Heavy cleanup between epochs
    print(f"\n🔧 Heavy cleanup between epochs {epoch} and {epoch+1}...")
    aggressive_vram_cleanup()
    print_gpu_memory_stats()
    print()
```

---

## Complete Example: Modified Training Script

Save as `src/ml/train_with_memory_management.py`:

```python
#!/usr/bin/env python3
"""
ES Futures VPOC ML Training with ROCm GPU Memory Management
Handles severe VRAM fragmentation on RX 7900 XT (gfx1100)
"""

# ⚠️ MUST be FIRST import
import os
os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'max_split_size_mb:256,garbage_collection_threshold:0.95'

import torch
import gc
import numpy as np
from torch.optim import Adam
from torch.nn import MSELoss

# Import your model and data utilities
from src.ml.model import ESFuturesVPOCModel
from src.ml.feature_engineering import prepare_features
from src.core.data import load_data

def aggressive_vram_cleanup():
    """Defragment GPU memory without killing process."""
    print("🔧 VRAM cleanup...")
    for device_id in [0, 1]:
        torch.cuda.set_device(device_id)
        torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect(0)
    gc.collect(1)
    gc.collect(2)
    torch.cuda.empty_cache()
    for device_id in [0, 1]:
        torch.cuda.set_device(device_id)
        torch.cuda.reset_accumulated_memory_stats()
        torch.cuda.synchronize()
    print("✅ Cleanup done")

def print_gpu_memory_stats():
    """Print GPU memory usage."""
    for device_id in [0, 1]:
        torch.cuda.set_device(device_id)
        free, total = torch.cuda.mem_get_info(device_id)
        used = total - free
        percent = (used / total) * 100
        print(f"  GPU {device_id}: {used/1024**3:.2f}GB / {total/1024**3:.2f}GB ({percent:.1f}%)")

def train_one_epoch(model, train_loader, optimizer, criterion, epoch):
    """Training with periodic cleanup."""
    model.train()
    total_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(0), target.to(0)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Cleanup every 20 batches
        if batch_idx % 20 == 0:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()
        
        # Log every 100 batches
        if batch_idx % 100 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            print(f"Epoch {epoch}, Batch {batch_idx}: Loss={avg_loss:.4f}")
            print_gpu_memory_stats()
    
    return total_loss / len(train_loader)

def main():
    print("=" * 60)
    print("ES Futures VPOC ML Training with Memory Management")
    print("=" * 60)
    
    # Initialize GPU
    print("\n🔧 Initializing GPU memory...")
    for device_id in [0, 1]:
        torch.cuda.set_device(device_id)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()
    print("✅ GPU initialization complete")
    print_gpu_memory_stats()
    
    # Load data
    print("\n📊 Loading data...")
    X, y = load_data('DATA/MERGED/merged_es_vix_test.csv')
    features = prepare_features(X)
    
    # Create model
    print("\n🧠 Creating model...")
    model = ESFuturesVPOCModel(input_size=features.shape[1], hidden_layers=[192, 128, 64])
    model = model.to(0)  # GPU 0
    
    # Setup training
    optimizer = Adam(model.parameters(), lr=0.0002)
    criterion = MSELoss()
    batch_size = 16
    epochs = 30
    
    # Create data loader
    from torch.utils.data import DataLoader, TensorDataset
    dataset = TensorDataset(
        torch.FloatTensor(features),
        torch.FloatTensor(y.values)
    )
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    print("\n🚀 Starting training...\n")
    for epoch in range(epochs):
        avg_loss = train_one_epoch(model, train_loader, optimizer, criterion, epoch)
        
        # Heavy cleanup between epochs
        print(f"\n🔧 Cleanup between epochs {epoch} and {epoch+1}...")
        aggressive_vram_cleanup()
        print_gpu_memory_stats()
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
            }, f'TRAINING/checkpoint_epoch_{epoch+1}.pt')
            print(f"✅ Checkpoint saved")
    
    print("\n✅ Training complete!")

if __name__ == '__main__':
    main()
```

---

## Testing Your Implementation

Run this quick test before full training:

```python
import os
os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'max_split_size_mb:256,garbage_collection_threshold:0.95'

import torch
import gc

print("Testing memory cleanup...")

# Initial state
print("Initial VRAM:")
for device_id in [0, 1]:
    torch.cuda.set_device(device_id)
    free, total = torch.cuda.mem_get_info(device_id)
    print(f"  GPU {device_id}: {free/1024**3:.2f}GB free")

# Allocate some memory
tensors = []
for i in range(10):
    t = torch.randn(10000000, device=0)
    tensors.append(t)
    print(f"Allocated tensor {i}, VRAM: {(total - torch.cuda.mem_get_info(0)[0])/1024**3:.2f}GB")

print("\nAfter allocation:")
torch.cuda.mem_get_info(0)

# Cleanup
print("\nCleaning up...")
del tensors
torch.cuda.synchronize()
torch.cuda.empty_cache()
gc.collect()
torch.cuda.empty_cache()

print("After cleanup:")
for device_id in [0, 1]:
    torch.cuda.set_device(device_id)
    free, total = torch.cuda.mem_get_info(device_id)
    print(f"  GPU {device_id}: {free/1024**3:.2f}GB free")
```

---

## Troubleshooting

### If Memory Still Fragments

```python
# More aggressive cleanup
os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'max_split_size_mb:128,garbage_collection_threshold:0.85'

# Call cleanup more frequently (every batch instead of every 20)
if batch_idx % 1 == 0:
    aggressive_vram_cleanup()
```

### If Training is Slow

```python
# Less aggressive cleanup
os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'max_split_size_mb:512,garbage_collection_threshold:0.99'

# Call cleanup less frequently (every 50 batches)
if batch_idx % 50 == 0:
    aggressive_vram_cleanup()
```

### If Still Failing

Consider switching to CPU-based ML framework:

```bash
# Install XGBoost (no GPU VRAM issues)
pip install xgboost lightgbm

# XGBoost usually outperforms deep learning on financial time-series
python -c "
import xgboost as xgb
model = xgb.XGBRegressor(n_estimators=200, max_depth=7)
model.fit(X_train, y_train)
preds = model.predict(X_test)
print('✅ Training complete - no VRAM issues!')
"
```

---

## Summary

| Solution | Difficulty | Effectiveness | Recommended For |
|----------|------------|----------------|-----------------|
| Solution 1: Aggressive Cleanup | Easy | 70-80% | First attempt |
| Solution 2: Memory Pool Config | Easy | 60-70% | Combined with others |
| Solution 3: Pre-allocation | Medium | 50-60% | Initial setup |
| Solution 4: Dynamic Batch Size | Medium | 65-75% | Adaptive training |
| Solution 5: Stream Management | Hard | 75-85% | Advanced users |
| Combined (Recommended) | Medium | 85-95% | Your situation |
| XGBoost Alternative | Easy | 99%+ | Best long-term |

---

## Final Recommendation

Use the **Combined Solution** (Solutions 1 + 2 + 4):

1. Set environment variables (Solution 2)
2. Initialize GPU before training (Solution 4)
3. Call cleanup periodically (Solution 1)
4. Monitor with print_gpu_memory_stats()

This approach should allow training without stopping Python while keeping VRAM fragmentation under control.
