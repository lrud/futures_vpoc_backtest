# Severe ROCm GPU Memory Fragmentation Case Study

## Executive Summary

**Issue**: Critical ROCm GPU memory fragmentation preventing any ML training despite functional hardware
**Severity**: BLOCKING - Complete training failure on otherwise perfectly working hardware
**Status**: CONFIRMED - Both GPUs functional, but 99% VRAM fragmentation makes training impossible

---

## Hardware Configuration (Verified Working)

### GPU Specifications
- **GPU 0**: AMD Radeon RX 7900 XT (21.5GB VRAM, gfx1100, 42 Compute Units)
- **GPU 1**: AMD Radeon RX 7900 XT (21.5GB VRAM, gfx1100, 42 Compute Units)
- **Architecture**: RDNA3 (consumer gaming GPUs, not server Instinct models)
- **Total GPU Memory**: 43GB combined

### Individual GPU Test Results ✅
```bash
# Both GPUs PASSED all individual tests:
✅ Matrix multiplication successful
✅ Mixed precision (BF16) successful
✅ Neural network forward/backward pass successful
✅ Basic tensor operations working
✅ CUDA/ROCm detection working
```

**Conclusion**: Hardware is PERFECTLY functional. No hardware issues detected.

---

## Current Problem State

### Memory Fragmentation Severity
```bash
# Both GPUs show severe fragmentation:
GPU 0: 21.46GB used / 21.46GB total (99.9% used, ~166MB free)
GPU 1: 21.21GB used / 21.46GB total (98.8% used, ~250MB free)
```

### Symptoms
1. **Individual GPU tests**: Work perfectly when tested in isolation
2. **ML Training**: Fails immediately with VRAM allocation errors
3. **Memory allocation**: "HIP out of memory" even for small allocations (96MB)
4. **PyTorch error**: `HIPBLAS_STATUS_ALLOC_FAILED when calling hipblasCreate(handle)`
5. **Fragmentation**: rocm-smi shows 99% usage but PyTorch reports minimal allocation

### Error Messages Encountered
```
1. HIP out of memory. Tried to allocate 96.00 MiB. GPU has 19.98 GiB total of which 0 bytes is free.
2. HIPBLAS_STATUS_ALLOC_FAILED when calling hipblasCreate(handle)
3. device >= 0 && device < num_gpus INTERNAL ASSERT FAILED
4. expandable_segments not supported on this platform
```

---

## Troubleshooting Attempts (All Failed)

### 1. Standard Memory Clearing Methods ❌
```bash
# PyTorch methods (ineffective):
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
gc.collect()

# Environment variables (ineffective):
export PYTORCH_HIP_ALLOC_CONF='expandable_segments:True,max_split_size_mb:128'
export TORCH_BLAS_PREFER_HIPBLASLT=0
```

### 2. ROCm Allocator Priming ❌
```python
# Documented ROCm workaround (failed):
def prime_rocm_allocator(device_id=0):
    for size_mb in [100, 200, 400, 800]:
        dummy = torch.randn(elements, device=device)
        del dummy
        torch.cuda.empty_cache()
# Result: HIP out of memory even for 100MB allocation
```

### 3. GPU Reset Methods ❌
```bash
# ROCm reset commands:
rocm-smi --gpureset -d 0
# Result: "reset_gpu, Not supported on the given system"

# Driver module reload:
# Not attempted due to container environment risk
```

### 4. Process Cleanup ❌
```bash
# Kill lingering processes:
pkill -f "python.*train.py"
pkill -f "python.*torch"
# Result: No processes found, memory still fragmented
```

### 5. Alternative Environment Variables ❌
```bash
# Various memory management settings tested:
export PYTORCH_NO_HIP_MEMORY_CACHING=1
export HSA_ENABLE_SDMA=0
export HSA_ENABLE_INTERRUPT=0
# Result: No improvement in memory availability
```

---

## System Configuration

### Software Stack
- **PyTorch**: 2.4.0a0+git2a26858
- **ROCm/HIP**: 6.3.42134-a9a80e791
- **System ROCm**: 6.3.2 (container)
- **GPU Architecture**: gfx1100 (RDNA3)

### Version Mismatch Identified
```
PyTorch: Built for ROCm 6.0 (2.4.1+rocm6.0 era)
System: Running ROCm 6.3.2
Result: Potential version incompatibility contributing to fragmentation
```

### Environment Variables Currently Set
```bash
HIP_VISIBLE_DEVICES=0,1
PYTORCH_ROCM_ARCH=gfx1100
TORCH_BLAS_PREFER_HIPBLASLT=0
PYTORCH_HIP_ALLOC_CONF='expandable_segments:True,max_split_size_mb:128'
HSA_ENABLE_SDMA=0
HSA_ENABLE_INTERRUPT=0
```

---

## Root Cause Analysis

### Primary Issue: ROCm Memory Allocator Bug
- **Problem**: ROCm 6.x memory allocator has known fragmentation issues
- **Behavior**: Memory gets allocated but never properly freed, leading to 99% utilization
- **Impact**: Prevents any new allocations despite theoretical availability

### Secondary Issue: Version Mismatch
- **Problem**: PyTorch built for ROCm 6.0 running on ROCm 6.3.2
- **Impact**: May exacerbate memory management issues

### Tertiary Issue: Consumer GPU Limitations
- **Problem**: RX 7900 XT (RDNA3) not optimized for heavy ML workloads
- **Impact**: More susceptible to memory fragmentation vs server GPUs

---

## Failed Solutions Summary

| Approach | Status | Reason for Failure |
|----------|--------|-------------------|
| `torch.cuda.empty_cache()` | ❌ | Fragmentation too severe |
| Environment variables | ❌ | expandable_segments not supported |
| ROCm allocator priming | ❌ | Can't allocate even small tensors |
| GPU reset via rocm-smi | ❌ | Not supported on this system |
| Process cleanup | ❌ | No lingering processes found |
| Driver module reload | ⚠️ | Not attempted (container risk) |

---

## Research Needed: Advanced Memory Recovery

### Areas for Investigation

1. **System-Level GPU Memory Reset**
   - Investigate kernel-level GPU reset commands
   - Research PCI device reset procedures
   - Look into container-level GPU reset methods

2. **ROCm Memory Allocator Deep Dive**
   - Research ROCm 6.3.2 specific allocator bugs
   - Find community workarounds for severe fragmentation
   - Investigate alternative ROCm memory management libraries

3. **PyTorch/ROCm Version Alignment**
   - Research PyTorch builds specifically for ROCm 6.3
   - Investigate custom PyTorch compilation for this ROCm version
   - Look into version compatibility matrices

4. **Container GPU Reset Strategies**
   - Research Docker container GPU reset approaches
   - Investigate container restart vs system restart options
   - Look into GPU virtualization layer reset methods

5. **Alternative Memory Management**
   - Research non-standard ROCm memory clearing techniques
   - Investigate third-party GPU memory management tools
   - Look into low-level HIP memory reset functions

### Key Questions for Research

1. **How do other users reset ROCm GPU memory when standard methods fail?**
2. **Are there system-level commands to forcibly clear GPU VRAM?**
3. **Can ROCm memory fragmentation be resolved without full system reboot?**
4. **Are there PyTorch builds that better handle ROCm 6.3 memory management?**
5. **What are the advanced techniques for clearing GPU memory in containerized environments?**

---

## Current Status: BLOCKING

**Hardware**: ✅ Perfectly functional
**Software**: ❌ Completely blocked by memory fragmentation
**Training**: ❌ Impossible to start any ML training
**Workarounds**: ❌ All standard methods ineffective

**Next Steps**: Research advanced GPU memory reset techniques or consider system reboot as last resort.

---

## Files Referenced
- `/workspace/scripts/test_individual_gpus.sh` - Individual GPU test results
- `/workspace/scripts/clear_gpu_memory.sh` - Memory clearing attempts
- `/workspace/BUG_RESEARCH/hipblas-fix-solutions.md` - Related ROCm issues
- `/workspace/BUG_RESEARCH/pytorch27-oom-shape-fix.md` - Memory fragmentation research

## Timestamp
2025-10-30 14:53:00 UTC - Case documented for research and troubleshooting