# ROCm Version Mismatch Analysis & Bug Research

## Current Status Summary

### ✅ Successfully Resolved Issues

1. **amdsmi Import Error**: Fixed with graceful fallback in `trainer_utils.py:69-80`
   ```python
   # Try to get utilization, but handle amdsmi import errors gracefully
   try:
       utilization = torch.cuda.utilization(i)
   except (RuntimeError, ImportError) as e:
       if "amdsmi" in str(e).lower() or "import" in str(e).lower():
           # amdsmi is not available, use memory-based utilization estimate
           memory_utilization = (used_memory / total_memory * 100) if total_memory > 0 else 0
           utilization = min(memory_utilization, 95)  # Cap at 95% for safety
   ```

2. **torch.compile Hanging**: Disabled in `trainer.py:280-286` due to ROCm version mismatch
   ```python
   # DISABLED: torch.compile due to ROCm version incompatibility
   # PyTorch 2.4.1+rocm6.0 vs ROCm tools 7.3.0 causes torch.compile to hang
   self.logger.warning(f"⚠️  torch.compile DISABLED due to ROCm version incompatibility")
   self.logger.warning(f"   • PyTorch: 2.4.1+rocm6.0 vs ROCm tools: 7.3.0")
   self.logger.warning(f"   • Using original model for stability")
   ```

3. **FSDP Hanging**: Disabled in `train.py:422-429`, forced DataParallel fallback
   ```python
   if strategy == 'fsdp':
       # DISABLED: FSDP due to ROCm version incompatibility
       # PyTorch 2.4.1+rocm6.0 vs ROCm tools 7.3.0 causes FSDP to hang at hccl backend init
       logger.warning(f"⚠️  FSDP DISABLED due to ROCm version incompatibility")
       logger.warning(f"   • PyTorch: 2.4.1+rocm6.0 vs ROCm tools: 7.3.0")
       logger.warning(f"   • FSDP hangs at hccl backend initialization")
       logger.warning(f"   • Using DataParallel fallback for stability")
       strategy = 'dataparallel'
   ```

4. **DataParallel Device Mismatch**: Fixed with device placement logic in `model.py:139-144`
   ```python
   # Ensure input tensor is on the same device as model parameters
   if hasattr(self, 'device') and x.device != self.device:
       x = x.to(self.device)
   elif hasattr(self.input_layer, 'weight') and x.device != self.input_layer.weight.device:
       x = x.to(self.input_layer.weight.device)
   ```

5. **CLI Parameters**: Added `--chunk_size`, `--skip_gpu_cleanup`, `--data_fraction` support
6. **VPOC Processing**: Working with configurable chunk sizes (1000-5000)
7. **Multi-GPU Detection**: Successfully identifies both RX 7900 XT GPUs
8. **Training Pipeline**: Complete from data loading → VPOC → feature engineering → model creation

### ❌ Current Blockers

**Primary Issue**: `HIPBLAS_STATUS_ALLOC_FAILED when calling hipblasCreate(handle)`

- **Root Cause**: PyTorch 2.4.1+rocm6.0 vs ROCm tools 7.3.0 version mismatch
- **Location**: First linear layer in forward pass: `model.py:146, F.linear(input, self.weight, self.bias)`
- **Affects**: Matrix multiplication operations in neural network layers
- **Tested**: Batch sizes 16, 8, 4 all fail with same hipBLAS allocation error

**Error Traceback**:
```
RuntimeError: CUDA error: HIPBLAS_STATUS_ALLOC_FAILED when calling `hipblasCreate(handle)`
  File "/workspace/src/ml/model.py", line 146, in forward
    x = F.silu(self.input_layer(x))
  File "/workspace/pytorch_venv/lib/python3.11/site-packages/torch/nn/modules/linear.py", line 117, in forward
    return F.linear(input, self.weight, self.bias)
```

## Technical Environment Details

### Hardware Configuration
- **GPU**: 2x AMD Radeon RX 7900 XT (RDNA3, gfx1100, 21.5GB each)
- **Architecture**: RDNA3 with Wave32 optimization support

### Software Stack
- **ROCm Tools**: 7.3.0
- **PyTorch**: 2.4.1+rocm6.0 (version mismatch)
- **hipBLAS**: ROCm 6.0 version incompatible with ROCm 7.3.0 tools
- **Python**: 3.11
- **Container**: Docker with ROCm 7.3.0

### Working Components ✅
- Data loading and preprocessing
- VPOC distributed processing on both GPUs
- Feature engineering (961 sessions, 54 features)
- Model creation and DataParallel wrapping
- Training loop initialization
- **Actually reaches Batch 1/192 with loss calculation before hipBLAS failure**

### Current Working Configuration
```bash
source pytorch_venv/bin/activate && PYTHONPATH=/workspace python src/ml/train.py \
  --data ./DATA/MERGED/merged_es_vix_test.csv \
  --output ./TRAINING/ \
  --epochs 1 \
  --batch_size 4 \
  --learning_rate 0.001 \
  --hidden_layers 32,16 \
  --use_mixed_precision \
  --data_fraction 0.05 \
  --skip_gpu_cleanup \
  --chunk_size 1000
```

## Research Areas for ROCm 6.2 Compatibility

### 1. hipBLAS Alternatives & Workarounds

**Research Questions:**
- Can we use ROCm 6.2 hipBLAS libraries with ROCm 7.3.0 tools?
- Environment variables to force specific hipBLAS versions?
- Custom hipBLAS compilation for RDNA3?

**Potential Solutions:**
- Environment variable overrides for hipBLAS path
- LD_LIBRARY_PATH modifications to point to ROCm 6.2 libraries
- Building hipBLAS from source with compatibility patches

### 2. ROCm 6.2 Optimization Settings

**Environment Variables to Research:**
```bash
# Memory management
HIP_MEMORY_POOL=1
HIP_MEMORY_POOL_SIZE=4GB
HIP_MAX_ALLOCATION_SIZE=2GB

# Threading and execution
HIP_LAUNCH_BLOCKING=1
HIP_VISIBLE_DEVICES=0,1
HIP_DEVICE_ORDER=PCI_BUS_ID

# ROCm 6.2 specific settings
MIOPEN_USER_DB_PATH=/tmp/miopen_user_db_6.2
MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen_cache_6.2
```

**Memory Allocation Optimizations:**
- Alternative GPU memory allocators (hipMalloc vs hipHostMalloc)
- Memory pool configurations for ROCm 6.2
- Fragmentation mitigation strategies

### 3. PyTorch ROCm Backend Alternatives

**Research Directions:**
- Force PyTorch to use different BLAS backend?
- OpenBLAS or other BLAS implementations as fallback?
- PyTorch build flags for ROCm 6.2 compatibility

**Potential Environment Variables:**
```bash
# Force different backends
TORCH_BACKEND=cpu
USE_ROCM=0
USE_OPENBLAS=1

# BLAS library selection
BLAS=openblas
LAPACK=openblas
```

### 4. RDNA3-Specific Optimizations

**ROCm 6.2 Settings for RDNA3:**
```bash
# Wave32 optimizations
HIP_WAVE32=1
GPU_SINGLE_ALLOC_PERCENT=90

# Memory management for RDNA3
HSA_UNALIGNED_ACCESS_MODE=1
GPU_MAX_HW_QUEUES=8
GPU_MAX_ALLOC_PERCENT=100
```

## Next Steps & Research Priorities

### Priority 1: hipBLAS Compatibility
1. Test ROCm 6.2 hipBLAS libraries with current setup
2. Research hipBLAS build configurations for ROCm 7.3.0 compatibility
3. Investigate alternative BLAS backends

### Priority 2: Memory Management
1. Test different memory allocation strategies
2. Research ROCm 6.2 specific memory pool settings
3. Experiment with batch size and memory layout optimizations

### Priority 3: Environment Configuration
1. Document optimal ROCm 6.2 environment variables
2. Test PyTorch with different backend configurations
3. Research container-based ROCm version isolation

## Testing Commands for Research

### Test hipBLAS Version Compatibility
```bash
# Check current hipBLAS version
hipblas-version

# Test with different library paths
export LD_LIBRARY_PATH=/opt/rocm-6.2/lib:$LD_LIBRARY_PATH
python -c "import torch; print(torch.cuda.is_available())"
```

### Test Memory Allocation
```bash
# Test different memory settings
export HIP_MEMORY_POOL=1
export HIP_MEMORY_POOL_SIZE=2GB
python src/ml/train.py --batch_size 2 --epochs 1

# Test with minimal memory usage
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:64
python src/ml/train.py --batch_size 1 --epochs 1
```

### Test Alternative Backends
```bash
# Test CPU fallback
python src/ml/train.py --device cpu --batch_size 32

# Test with different BLAS
export BLAS=openblas
python src/ml/train.py --batch_size 8
```

## Files Modified During Debugging

### Core Files Updated:
1. `/workspace/src/ml/train.py` - FSDP disabled, CLI parameters added
2. `/workspace/src/ml/trainer.py` - torch.compile disabled
3. `/workspace/src/ml/model.py` - Device placement fixes
4. `/workspace/src/ml/trainer_utils.py` - amdsmi graceful fallback
5. `/workspace/src/core/vpoc.py` - Chunk size parameterization

### Configuration Files:
- Environment variables in `.bashrc` or container startup scripts
- Docker configuration for ROCm library paths
- PyTorch build configuration for ROCm compatibility

## Success Criteria

**Training Pipeline Success**:
- ✅ Data loading and preprocessing
- ✅ VPOC calculations on both GPUs
- ✅ Feature engineering
- ✅ Model creation
- ✅ Training loop start
- ❌ hipBLAS matrix operations (current blocker)

**Target Outcome**:
- Complete training run that creates `.pt` and metadata files in `/workspace/TRAINING/`
- Successful multi-GPU training on both RX 7900 XT GPUs
- Stable performance without hipBLAS allocation failures