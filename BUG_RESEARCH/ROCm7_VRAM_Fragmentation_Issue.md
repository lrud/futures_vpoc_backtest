# ROCm 6.2 Training Issue - ES Futures VPOC Training

## **Issue Summary**
The ES Futures VPOC algorithmic trading repository cannot successfully train merged ES/VIX data due to a critical issue during the GPU cleanup phase. All training attempts hang at "🧹 Performing pre-training GPU cleanup..." and never reach actual neural network training.

## **Technical Details**

### **Hardware Configuration**
- **GPU**: 2x AMD Radeon RX 7900 XT (RDNA3, gfx1100)
- **Architecture**: Wave32 mode optimized for 7900 XT
- **ROCm Version**: 6.2.0 with PyTorch integration (NOT ROCm 7)
- **VRAM**: 21.46GB per GPU

### **Bug Manifestation**
- **Symptom**: `rocm-smi` shows 99% VRAM usage while PyTorch reports minimal allocation (203KB)
- **Failure Point**: Processes hang at "🧹 Performing pre-training GPU cleanup..."
- **Error**: Training freezes indefinitely during neural network initialization
- **Impact**: Prevents both distributed and single-GPU training from reaching actual model training

### **Current Status: FAILED - ROCm 6.2 GPU Cleanup Issue Persists**

### **✅ RESOLVED: Primary VRAM Fragmentation Issue**
- **VPOC Calculations**: Successfully complete with chunked processing
- **Faster VPOC Processing**: Improved from 200-bar to 500-bar chunks
- **Memory Management**: Chunked VPOC calculation prevents VRAM fragmentation during volume profile analysis

### **❌ PERSISTING ISSUE: ROCm 6.2 GPU Cleanup Hang**
Training consistently hangs at "🧹 Performing pre-training GPU cleanup..." despite attempted fixes. This is **NOT** a ROCm 7 issue - we are running ROCm 6.2.0.

#### **Failed Attempts (2025-10-29)**:
1. **500-bar chunks** - Improved VPOC speed but didn't fix cleanup hang
2. **hipBLASLt disable** (`TORCH_BLAS_PREFER_HIPBLASLT=0`) - No effect on cleanup issue
3. **Enhanced memory variables** - Still hangs at same point
4. **ROCm 6.2.0 downgrade** - Successfully downgraded from ROCm 7, but still has the same GPU cleanup hang issue
5. **Latest attempt (20:04)** - Training run fbdcc8 successfully completed VPOC processing (8+ minutes) and reached GPU cleanup phase at 19:58:54, but still hangs at the same critical point

#### **Current Failure Point**:
```
2025-10-29 19:58:54,228 - __main__ - INFO - Multi-GPU training detected with 2 GPUs
2025-10-29 19:58:54,228 - __main__ - INFO - 🧹 Performing pre-training GPU cleanup...
[HANGS HERE INDEFINITELY - Confirmed with ROCm 6.2.0 + 500-bar chunks]
```

### **❌ ROCm 6.2 Training Incompatibility**
The issue persists in ROCm 6.2.0, indicating this is not a ROCm 7-specific bug but a broader PyTorch + AMD GPU training initialization issue.

#### **1. HIPBLAS_STATUS_INTERNAL_ERROR**
```
RuntimeError: HIP error: hipblasStatusInternalError
HIP error: hipblasStatusInternalError when calling [someFunction]
```
- **Cause**: ROCm 7 HIPBLAS library incompatibility with PyTorch matrix operations
- **Impact**: Prevents neural network training during forward/backward passes
- **Status**: Occurs with both mixed precision and standard precision

#### **2. Kernel Image Unavailable Error**
```
RuntimeError: HIP error: no kernel image is available for execution on the device
```
- **Cause**: ROCm 7 missing or incompatible GPU kernels for specific operations
- **Suggested Fix**: Error message suggests using `AMD_SERIALIZE_KERNEL=3`
- **Impact**: Prevents mixed precision training and certain GPU optimizations

#### **3. Multiple ROCm 7 Library Incompatibilities**
- **Flash Attention**: Disabled (causes fragmentation)
- **JIT Fuser**: Disabled (causes memory leaks)
- **Mixed Precision**: Problematic due to kernel availability
- **HIPBLAS**: Internal errors during matrix operations

### **Root Cause Analysis**
While the VPOC VRAM fragmentation issue was resolved through chunked calculations, ROCm 7 itself appears to have significant compatibility issues with PyTorch that prevent successful neural network training. The problems have shifted from VPOC-specific fragmentation to general ROCm 7 + PyTorch incompatibility.

## **Commands That Fail**
```bash
# Distributed training (hangs at VPOC init)
PYTHONPATH=/workspace HIP_VISIBLE_DEVICES=0,1 PYTORCH_ROCM_ARCH=gfx1100 \
PYTORCH_HIP_ALLOC_CONF='expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.6' \
python src/ml/train.py --data DATA/MERGED/merged_es_vix_test.csv \
--output TRAINING/ --epochs 3 --batch_size 16 --learning_rate 0.0002 \
--hidden_layers 64,32 --use_mixed_precision --distributed_strategy fsdp

# Single GPU training (hangs at VPOC init)
HIP_VISIBLE_DEVICES=0 PYTHONPATH=/workspace python src/ml/train.py \
--data DATA/MERGED/merged_es_vix_test.csv --output TRAINING/ \
--epochs 3 --batch_size 8 --learning_rate 0.0002 --hidden_layers 32,16 \
--no_distributed --device_ids 0
```

## **Successful Progress Before Current Failures**
All training attempts now successfully complete:
- ✅ Data loading and preprocessing
- ✅ VPOC VolumeProfileAnalyzer initialization: "🚀 Initialized ROCm 7 VolumeProfileAnalyzer with precision 0.25"
- ✅ **VPOC Volume Profile Calculations** (RESOLVED with chunked processing)
- ✅ Feature engineering for 1014 sessions with 54 features
- ✅ Target variable creation with 1013 non-null values
- ✅ FSDP distributed strategy initialization: "🎯 Using distributed strategy: fsdp"

**Current Failure Point**: Training fails during neural network matrix operations with ROCm 7 compatibility issues.

## **Repository Context**
- **Purpose**: ES Futures VPOC (Volume Point of Control) algorithmic trading strategy
- **Goal**: "A means to an end to train my data alone" - not intended as general ML tool
- **Data**: Merged ES/VIX futures data for enhanced trading signals
- **Architecture**: Optimized for AMD GPUs with ROCm 7

## **Environment Variables Tested**
```bash
# Current configuration (still fails)
export HIP_VISIBLE_DEVICES=0,1
export PYTORCH_ROCM_ARCH=gfx1100
export PYTORCH_HIP_ALLOC_CONF='expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.6'
export PYTHONPATH=/workspace
```

## **Key Files Modified**
- `/workspace/src/ml/model.py` - Fixed torch.jit.fuser() memory leak
- `/workspace/src/ml/train.py` - Fixed FSDP timeout error
- `/workspace/src/ml/distributed_trainer.py` - Disabled Flash Attention
- `/workspace/src/ml/trainer.py` - Enhanced memory management
- `/workspace/README.md` - Updated with VRAM bug warnings
- `/workspace/documentation/ML_Training_Guide.md` - Added ROCm 7 optimization notes

## **Research Needed**
Looking for solutions to:
1. **ROCm 7 GPU cleanup hang during neural network initialization**
2. PyTorch `torch.cuda.empty_cache()` alternatives that work with ROCm 7
3. AMD GPU memory management strategies for pre-training phase
4. Community fixes for ROCm 7 + PyTorch training initialization issues
5. Alternative approaches to bypass GPU cleanup phase

## **Success Criteria**
Training is considered successful when:
- Training files (`.pt` and `_metadata.json`) appear in `TRAINING/` folder
- Training progresses past VPOC calculations to actual neural network training
- No "HIP out of memory" errors during execution
- Process completes all requested epochs

## **Contact Research Focus**
Please research solutions specifically for:
- ✅ **RESOLVED**: ROCm 7 VPOC memory fragmentation (chunked calculations)
- ❌ **CRITICAL ISSUE**: ROCm 7 GPU cleanup hang during neural network initialization
- PyTorch GPU memory cleanup alternatives for ROCm 7
- Community fixes for "Performing pre-training GPU cleanup" hang
- ROCm 7 vs ROCm 6.2 comparison for training initialization issues
- Alternative approaches to bypass GPU cleanup phase entirely

## **Internet Research Required: ROCm 7 Compatibility Issues**

### **Critical Research Questions**
1. **GPU Cleanup Hang**: "ROCm 7 PyTorch pre-training GPU cleanup hang fix"
2. **Memory Management**: "PyTorch torch.cuda.empty_cache ROCm 7 alternatives"
3. **Initialization Issues**: "ROCm 7 neural network initialization freeze workaround"
4. **Training Pipeline**: "PyTorch training GPU cleanup phase bypass ROCm 7"
5. **ROCm Comparison**: "ROCm 6.2 vs 7 PyTorch training stability comparison"

### **Potential Solutions to Research**
- **GPU cleanup bypass**: Disable or skip pre-training GPU cleanup phase
- **ROCm 6.2 downgrade**: Complete downgrade procedures for stable training
- **Alternative memory management**: PyTorch alternatives to `torch.cuda.empty_cache()`
- **Training pipeline modification**: Reorder initialization to avoid cleanup hang
- **Debugging approaches**: Tools and techniques to identify the exact cleanup blockage