# 🚀 Safe Training Guide

## 🛡️ **Preventing GPU Memory Issues**

### **Before Training:**

1. **Check GPU Memory Status:**
```bash
rocm-smi --showmeminfo vram
```

2. **Run GPU Memory Monitor:**
```bash
/workspace/scripts/gpu_memory_monitor.sh check
```

3. **If memory usage > 20%, cleanup:**
```bash
/workspace/scripts/gpu_memory_monitor.sh cleanup
```

### **Safe Training Methods:**

#### **Method 1: Use Safe Training Wrapper (Recommended)**
```bash
/workspace/scripts/safe_train.sh \
    --hidden_layers 192,128,64 \
    --learning_rate 0.0002 \
    --epochs 50 \
    --batch_size 32 \
    --use_mixed_precision \
    --device_ids 0,1 \
    --contract ES \
    --data DATA/MERGED/merged_es_vix_test.csv \
    --dropout_rate 0.3 \
    --train_split 0.85 \
    --output TRAINING \
    --verbose \
    --distributed_strategy auto
```

#### **Method 2: Manual Training with Pre-checks**
```bash
# 1. Check memory first
/workspace/scripts/gpu_memory_monitor.sh check

# 2. Start monitoring in background
/workspace/scripts/gpu_memory_monitor.sh monitor &

# 3. Run training
PYTHONPATH=/workspace python src/ml/train.py [args...]

# 4. Stop monitoring
kill %1
```

### **If Training Crashes:**

1. **Immediate Cleanup:**
```bash
/workspace/scripts/gpu_memory_monitor.sh cleanup
```

2. **Check for Orphaned Processes:**
```bash
rocm-smi --showall | grep "PID.*UNKNOWN"
```

3. **If issues persist, check logs:**
```bash
tail -50 /workspace/logs/gpu_monitor.log
tail -50 /workspace/logs/safe_train.log
```

### **Memory Safety Settings:**

- **Max safe memory usage:** 15GB per GPU
- **Warning threshold:** 85% usage
- **Critical threshold:** 95% usage (auto-stop)
- **Cleanup attempts:** 3 automatic attempts

### **Recommended Training Parameters:**

- **Conservative:** `--batch_size 16`, `--hidden_layers 128,64`
- **Balanced:** `--batch_size 24`, `--hidden_layers 192,128,64`
- **Aggressive:** `--batch_size 32`, `--hidden_layers 256,192,128,64`

### **Emergency Procedures:**

If GPU memory remains stuck > 20% after cleanup:

1. **Reboot system (only if necessary)**
2. **Check for hardware issues**
3. **Verify ROCm driver integrity**

## 📊 **Monitoring Logs:**

- **GPU Monitor:** `/workspace/logs/gpu_monitor.log`
- **Safe Training:** `/workspace/logs/safe_train.log`
- **Training Logs:** `/workspace/logs/futures_vpoc_*.log`

## 🎯 **Success Indicators:**

✅ Pre-training memory check passes (< 20% usage)
✅ Training starts without memory warnings
✅ Memory usage stays < 85% during training
✅ Training completes successfully
✅ Post-training cleanup completes
✅ Model file saved to `TRAINING/` directory