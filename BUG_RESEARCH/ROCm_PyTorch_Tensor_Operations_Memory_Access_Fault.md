# ROCm PyTorch Tensor Operations Memory Access Fault Issue

## Issue Summary

**Date**: 2025-10-30
**Severity**: Critical - Complete ML Pipeline Failure
**Status**: Investigation Complete - Container Rebuild Recommended

## Problem Description

The entire ML training pipeline is failing due to fundamental PyTorch tensor operations on ROCm GPUs causing memory access faults. This prevents both VPOC calculations and model training from functioning.

## Error Details

### Primary Error Message
```
Memory access fault by GPU node-1 (Agent handle: 0x...) on address 0x...
Reason: Page not present or supervisor privilege.
```

### When Error Occurs
- **Basic PyTorch GPU operations**: `torch.tensor(..., device='cuda:0')` or `x.to('cuda:0')`
- **VPOC initialization**: During `VolumeProfileAnalyzer.__init__()` when creating GPU tensors
- **Both GPUs**: Error occurs on GPU 0 and GPU 1 identically

### What Works (Doesn't Crash)
- ✅ GPU Detection: `torch.cuda.is_available()` returns `True`
- ✅ GPU Count: `torch.cuda.device_count()` returns `2`
- ✅ GPU Properties: `torch.cuda.get_device_properties()` works
- ✅ Basic queries: All non-tensor operations function correctly

### What Fails (Crashes)
- ❌ Tensor Creation: `torch.randn(10, 10).cuda()`
- ❌ Tensor Movement: `cpu_tensor.to('cuda:0')`
- ❌ VPOC Calculations: Any GPU-based volume profile operations
- ❌ ML Training: Complete pipeline failure

## Environment Analysis

### Hardware
- **GPU 0**: Radeon RX 7900 XT (Navi 31, gfx1100)
- **GPU 1**: Radeon RX 7900 XT (Navi 31, gfx1100)
- **VRAM Status**: 0-1% utilization (GPUs idle)
- **GPU Clocks**: 0Mhz SCLK, 96Mhz MCLK (idle state)

### Software Versions
- **ROCm Version**: 6.3.2-66 (NOT ROCm 7)
- **PyTorch Version**: 2.4.0a0+git2a26858 (development build)
- **HIP Version**: 6.3.42134-a9a80e791
- **CUDA Version**: None (expected for ROCm)

### Environment Variables
```
HIP_VISIBLE_DEVICES=0,1
PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:256,garbage_collection_threshold:0.95
PYTORCH_ROCM_ARCH=gfx1100
TORCH_BLAS_PREFER_HIPBLASLT=0
PYTORCH_NO_HIP_MEMORY_CACHING=1
```

## Root Cause Analysis

### Eliminated Causes
1. **VRAM Fragmentation**: VRAM is at 0% utilization
2. **Memory Leaks**: No prior GPU usage or memory allocation
3. **Over-utilization**: GPUs are in idle state
4. **ROCm 7 Issues**: Container uses ROCm 6.3.2
5. **Hardware Failure**: Both GPUs fail identically

### Likely Root Causes
1. **Corrupted ROCm Installation**: ROCm 6.3.2 installation may be corrupted
2. **PyTorch Build Issues**: Development build `2.4.0a0+git2a26858` appears unstable
3. **Driver/GPU Compatibility**: Fundamental incompatibility between PyTorch and ROCm drivers
4. **Installation Order**: Components may have been installed in incorrect sequence

## Debug Steps Performed

1. **Basic GPU Detection Tests**: ✅ Passed
2. **Simple Tensor Operations**: ❌ Failed with memory access fault
3. **Environment Variable Cleanup**: ❌ Issue persisted
4. **Multi-GPU Testing**: ❌ Both GPUs fail identically
5. **Timeout Tests**: ❌ All GPU operations hang then crash

## Impact Assessment

### Current Impact
- **Complete ML Pipeline Failure**: No training possible
- **VPOC Calculations Broken**: Volume profiling unavailable
- **Development Blocked**: All ML development halted

### Business Impact
- **Strategy Testing**: ML-enhanced strategy testing impossible
- **Model Development**: New model development blocked
- **Production Readiness**: System not functional for ML workflows

## Recommended Solution

### Primary Solution: Container Rebuild
1. **Rebuild Container**: Fresh installation of all components
2. **Verify Component Versions**: Use stable, tested versions
3. **Test Incrementally**:
   - Basic GPU detection
   - Simple tensor operations
   - VPOC calculations
   - Full training pipeline

### Alternative Solutions (if rebuild fails)
1. **ROCm Version Change**: Try different ROCm 6.3.x versions
2. **PyTorch Version Change**: Use stable PyTorch release instead of dev build
3. **Driver Update**: Update AMD GPU drivers
4. **Hardware Verification**: Test GPUs on different system

## Technical Details

### Command That Fails
```bash
python -c "import torch; x = torch.randn(10, 10); y = x.to('cuda:0')"
```

### Error Location
The error occurs at PyTorch level when attempting to allocate GPU memory, specifically in the tensor-to-GPU transfer operation.

### System State at Failure
- **GPU Utilization**: 0%
- **VRAM Usage**: 0-1%
- **GPU Clocks**: Idle
- **System Load**: Minimal

## Related Issues

- Similar to `ROCm7_VRAM_Fragmentation_Issue.md` but with different root cause
- Related to `gpu_memory_fragmentation_severe_case.md`
- Follows pattern of PyTorch+ROCm compatibility issues in project

## Next Steps

1. **Immediate**: Rebuild container with fresh component installation
2. **Verification**: Test incremental functionality after rebuild
3. **Documentation**: Update with rebuild results
4. **Prevention**: Implement container health checks for future deployments

---

**Investigation By**: Claude AI Assistant
**Date Created**: 2025-10-30
**Last Updated**: 2025-10-30