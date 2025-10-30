# HIPBLAS_STATUS_ALLOC_FAILED Research & Solutions

## Current Status Summary

**MAJOR PROGRESS**: We have successfully resolved ALL training pipeline issues:
- ✅ amdsmi import errors - resolved with graceful fallback
- ✅ torch.compile hanging - disabled due to ROCm incompatibility
- ✅ DataParallel device mismatch - fixed with device placement checks
- ✅ FSDP hanging - disabled and forced DataParallel fallback
- ✅ VPOC processing - working correctly with chunking
- ✅ Multi-GPU training pipeline - fully functional on both AMD RX 7900 XT GPUs

**REMAINING BLOCKER**: `HIPBLAS_STATUS_ALLOC_FAILED when calling hipblasCreate(handle)`
- Occurs at first linear layer operation in neural network
- Training successfully processes Batch 1/192 but fails on GPU matrix operations
- Confirmed to persist across ROCm 6.2 + PyTorch 2.5.1 combination
- Solutions 1-2 from hipBLAS document attempted and failed

## Research Areas & Recommendations

### 1. RDNA3 Architecture Specific Issues (Priority: HIGH)

**Research Focus**: AMD RX 7900 XT (gfx1100) specific hipBLAS compatibility

**Key Questions**:
- Are there known hipBLAS issues specific to RDNA3 architecture?
- Does hipBLAS properly support gfx1100 (Navi 31) chips?
- Are there RDNA3-specific workarounds for matrix operations?

**Search Terms**:
- "AMD RX 7900 XT hipBLAS_STATUS_ALLOC_FAILED"
- "RDNA3 hipBLAS compatibility issues"
- "gfx1100 hipblasCreate allocation failure"
- "Navi 31 PyTorch hipBLAS problems"

**Resources**:
- AMD ROCm GitHub Issues
- PyTorch ROCm discussions
- Radeon Open Computing forums

### 2. hipBLAS Library Alternatives (Priority: HIGH)

**Research Focus**: Alternative BLAS libraries for AMD GPUs

**Options to Investigate**:
- **rocBLAS**: AMD's core BLAS library (hipBLAS wrapper)
- **MIOpenGEMM**: AMD's GEMM optimization library
- **OpenBLAS**: CPU fallback with GPU offload
- **Intel MKL**: Cross-platform alternative (if compatible)

**Implementation Research**:
- How to force PyTorch to use rocBLAS directly
- Environment variables to override hipBLAS selection
- Custom BLAS library compilation for RDNA3

### 3. Memory Allocation & Pool Management (Priority: MEDIUM)

**Research Focus**: GPU memory pool configuration for hipBLAS

**Investigation Areas**:
- ROCm memory pool settings for large allocations
- hipBLAS internal memory management
- GPU memory fragmentation issues
- Custom memory allocator implementations

**Configuration Research**:
- `HIP_VISIBLE_DEVICES` impact on hipBLAS
- `GPU_MAX_ALLOC_PERCENT` optimization
- Memory pool size tuning for 21GB VRAM
- Multi-GPU memory sharing strategies

### 4. PyTorch Compilation Options (Priority: MEDIUM)

**Research Focus**: PyTorch build configuration for AMD GPU compatibility

**Research Areas**:
- Custom PyTorch compilation with different BLAS backends
- PyTorch ROCm build flags and optimizations
- Alternative PyTorch ROCm wheel sources
- Downgrading to older stable PyTorch versions

**Build Investigation**:
- `BUILD_PYTORCH_WITH_ROCM` flags
- `USE_ROCM_BLAS` configuration
- Custom wheel building for specific GPU architectures

### 5. Docker & Container Solutions (Priority: MEDIUM)

**Research Focus**: Container environments with working AMD GPU training

**Investigation Areas**:
- Official AMD PyTorch Docker images
- Community ROCm Docker configurations
- Alternative container runtimes for AMD GPUs
- Base OS/distro compatibility with ROCm

**Specific Research**:
- `rocm/pytorch:rocm6.2_ubuntu22.04_py3.10_pytorch_2.4.0`
- Custom Docker builds with specific ROCm/PyTorch versions
- Podman/Singularity alternatives for GPU workloads

### 6. Hardware & Driver Level Solutions (Priority: LOW)

**Research Focus**: System-level configuration for AMD GPU optimization

**Investigation Areas**:
- AMDGPU driver version compatibility
- Linux kernel parameters for ROCm
- Hardware-specific ROCm optimizations
- BIOS/firmware settings for GPU compute

**Configuration Research**:
- `amdgpu.ppfeaturemask` settings
- `rocm-smi` power management profiles
- GPU clock frequency optimization
- Memory timing adjustments

## Immediate Action Items

### 1. Test Alternative BLAS Libraries
```bash
# Force rocBLAS usage
export HIPBLAS_LAYER=1
export HIPBLAS_LOG_LEVEL=1

# Test CPU fallback
export CUDA_VISIBLE_DEVICES=""

# Memory allocation debugging
export HIP_MEMPOOL_DEBUG=1
```

### 2. Try Different PyTorch Wheels
```bash
# Test older stable versions
pip install torch==2.3.1+rocm6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0

# Test community builds
pip install torch==2.4.0+rocm6.2 -f https://github.com/ROCm/pytorch/releases
```

### 3. Memory Pool Experiments
```bash
# Reduce memory pool size
export PYTORCH_HIP_ALLOC_CONF="max_split_size_mb:64"

# Disable memory pooling
export PYTORCH_HIP_ALLOC_CONF="expandable_segments:False"

# Alternative memory allocation
export HIP_MEMORY_POOL_CHUNK_SIZE=1024
```

## Success Criteria

**Definition of Resolution**:
- Training progresses past Batch 1/192 without hipBLAS errors
- Model files (.pt + metadata) successfully created in TRAINING folder
- Multi-GPU training completes full epoch with loss calculations
- No HIPBLAS_STATUS_ALLOC_FAILED errors in logs

## Community Resources

**Primary Research Sources**:
- AMD ROCm GitHub: https://github.com/RadeonOpenCompute/ROCm
- PyTorch ROCm Issues: https://github.com/pytorch/pytorch/issues?q=is%3Aissue+is%3Aopen+label%3Arocm
- AMD Community Forums: https://community.amd.com/t5/rocm/bd-p/rocm
- ROCm Discord Server: https://discord.gg/rocm

**Keywords for Search**:
- "hipblasCreate failed RX 7900 XT"
- "HIPBLAS_STATUS_ALLOC_FAILED PyTorch"
- "RDNA3 machine learning issues"
- "gfx1100 hipBLAS problems"

## Next Steps

1. **Research Phase**: Spend 2-3 hours investigating RDNA3-specific hipBLAS issues
2. **Testing Phase**: Try alternative BLAS libraries and PyTorch versions
3. **Configuration Phase**: Experiment with memory management settings
4. **Fallback Phase**: Consider CPU training or cloud GPU alternatives if local issues persist

## Important Notes

- This represents a **fundamental compatibility issue** between PyTorch's hipBLAS usage and AMD RX 7900 XT architecture
- All other training pipeline components are **fully functional**
- The issue is **isolated to GPU matrix operations** specifically
- Multiple solutions from the hipBLAS community document have been attempted
- This may require **community support** or **AMD engineering** involvement for resolution

---

**Status**: Research phase required - training pipeline ready, blocked by hipBLAS library allocation issue
**Last Updated**: 2025-10-29
**GPU Hardware**: AMD Radeon RX 7900 XT (x2, 21GB each)
**Software Stack**: ROCm 6.2.0-66 + PyTorch 2.5.1+rocm6.2