# ES Futures VPOC Strategy - Quick Start Guide

## Overview

This guide will get you up and running with the ES Futures VPOC trading strategy quickly. Whether you're a quantitative trader, researcher, or developer, this guide will walk you through the essential steps to set up, configure, and run the system.

## What You'll Need

### Prerequisites
- **Hardware**: AMD GPU (RX 7900 XT recommended) with ROCm 7.0 support
- **Operating System**: Ubuntu 22.04 LTS (recommended) or RHEL 8/9
- **Python**: Version 3.10 or higher
- **RAM**: Minimum 32GB, 64GB+ recommended
- **Storage**: At least 500GB SSD for data and models

### Software Requirements
- ROCm 7.0 or later
- PyTorch 2.10 with ROCm support
- Git
- Basic understanding of Python and algorithmic trading concepts

## Quick Installation (15 Minutes)

### Step 1: Install ROCm 7.0
```bash
# Add AMD repository and install ROCm
wget https://repo.radeon.com/amdgpu-install/6.0.2/ubuntu/jammy/amdgpu-install_6.0.2.60002-1_all.deb
sudo apt-get install ./amdgpu-install_6.0.2.60002-1_all.deb
sudo amdgpu-install --usecase=rocm

# Add user to required groups
sudo usermod -a -G render,video $LOGNAME
sudo reboot
```

### Step 2: Verify ROCm Installation
```bash
# Check GPU detection
rocm-smi

# Expected output should show your AMD GPU
```

### Step 3: Set Up Python Environment
```bash
# Create virtual environment
python3 -m venv futures-vpoc-env
source futures-vpoc-env/bin/activate

# Install PyTorch 2.10 with ROCm 7.0
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.0

# Install project dependencies
pip install numpy pandas matplotlib seaborn scikit-learn
pip install transformers>=4.35.0 wandb tensorboard
pip install flash-attn --no-build-isolation
```

### Step 4: Clone and Set Up Project
```bash
# Navigate to workspace
cd /workspace

# Set up Python path
export PYTHONPATH=/workspace:$PYTHONPATH

# Create necessary directories
mkdir -p DATA CLEANED RESULTS STRATEGY BACKTEST TRAINING LOGS
```

### Step 5: Verify Installation
```bash
# Test GPU and PyTorch setup
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'ROCm available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"
```

## Your First Backtest (10 Minutes)

### Step 1: Prepare Sample Data
Create a sample data file at `DATA/ES/sample_5min.csv`:

```csv
Date,Time,Open,High,Low,Close,Volume
2024-01-02,09:30,4500.25,4502.50,4498.75,4501.00,1500000
2024-01-02,09:31,4501.00,4503.25,4499.50,4502.75,1200000
2024-01-02,09:32,4502.75,4504.00,4500.50,4503.50,980000
... (add more rows)
```

### Step 2: Run Quick Test
```bash
# Test VPOC calculation
python src/scripts/test_vpoc.py

# Test ML components
python src/scripts/test_model.py

# Run basic backtest
python src/scripts/test_backtest.py
```

### Step 3: Configure Strategy
Create `config/quick_config.json`:
```json
{
  "trading": {
    "initial_capital": 100000,
    "risk_per_trade": 0.01,
    "commission_per_trade": 10,
    "slippage": 0.25
  },
  "ml": {
    "use_ml_filter": true,
    "confidence_threshold": 60,
    "prediction_threshold": 0.5
  },
  "system": {
    "device_ids": [0],
    "num_workers": 4
  }
}
```

### Step 4: Run Your First Backtest
```bash
# Create a simple backtest script
cat > quick_backtest.py << 'EOF'
import sys
sys.path.append('/workspace')

from src.analysis.backtest import BacktestEngine
from src.config.settings import settings

# Quick configuration
config = {
    'initial_capital': 100000,
    'risk_per_trade': 0.01,
    'commission_per_trade': 10,
    'slippage': 0.25,
    'use_ml_filter': False,  # Start without ML
    'confidence_threshold': 60
}

print("Running quick backtest...")
print("This will test the basic VPOC strategy without ML enhancement.")

# Note: You'll need actual ES data for this to work properly
# The system is designed for 1-minute or 5-minute ES futures data

EOF

python quick_backtest.py
```

## Training Your First ML Model (30 Minutes)

### Step 1: Prepare Training Data
```bash
# Ensure you have sufficient historical data
# Minimum recommended: 2 years of 5-minute ES data
# Place in: DATA/ES/5min/

# Validate data format
python -c "
import pandas as pd
df = pd.read_csv('DATA/ES/5min/your_data.csv')
print(f'Data shape: {df.shape}')
print(f'Columns: {list(df.columns)}')
print(f'Date range: {df.iloc[0][\"date\"]} to {df.iloc[-1][\"date\"]}')
"
```

### Step 2: Run Feature Engineering
```bash
python src/scripts/test_feature_engineering.py
```

### Step 3: Train Model
```bash
# Single GPU training
python src/ml/train.py \
    --data_path DATA/ES/5min/ \
    --model_name vpoc_enhanced_v1 \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 1e-4

# Multi-GPU training (if available)
python src/ml/distributed_trainer.py \
    --data_path DATA/ES/5min/ \
    --world_size 2 \
    --model_name vpoc_enhanced_v1_multi \
    --epochs 100
```

### Step 4: Evaluate Model
```bash
python src/ml/evaluate_models.py \
    --model_path TRAINING/vpoc_enhanced_v1/best_model.pt \
    --test_data_path DATA/ES/5min/
```

## Running Full Backtest with ML Enhancement

### Step 1: Update Configuration
Edit your config to enable ML:
```json
{
  "ml": {
    "use_ml_filter": true,
    "ml_model_path": "TRAINING/vpoc_enhanced_v1/best_model.pt",
    "confidence_threshold": 60,
    "prediction_threshold": 0.5
  }
}
```

### Step 2: Run Comprehensive Backtest
```bash
# This will use the enhanced strategy
python src/analysis/run_ml_backtest.py \
    --config config/quick_config.json \
    --data_path DATA/ES/5min/ \
    --output_dir BACKTEST_RESULTS/
```

### Step 3: Analyze Results
The backtest will generate:
- **Performance metrics** (return, Sharpe, win rate, etc.)
- **Trade list** with detailed P&L information
- **Equity curve** showing capital over time
- **Risk analysis** including drawdown metrics

## Common Usage Patterns

### 1. Strategy Development Workflow
```bash
# 1. Test basic VPOC signals
python src/scripts/test_vpoc.py

# 2. Run quick backtest without ML
python src/scripts/test_backtest.py

# 3. Train ML model on historical data
python src/ml/train.py --data_path DATA/ES/5min/

# 4. Run enhanced backtest
python src/analysis/run_ml_backtest.py

# 5. Analyze results and optimize parameters
python src/analysis/math_utils.py  # For mathematical validation
```

### 2. Parameter Optimization
```bash
# Test different risk levels
python src/analysis/run_ml_backtest.py --risk_per_trade 0.005
python src/analysis/run_ml_backtest.py --risk_per_trade 0.015
python src/analysis/run_ml_backtest.py --risk_per_trade 0.02

# Test different confidence thresholds
python src/analysis/run_ml_backtest.py --confidence_threshold 50
python src/analysis/run_ml_backtest.py --confidence_threshold 70
python src/analysis/run_ml_backtest.py --confidence_threshold 80
```

### 3. Multi-GPU Training
```bash
# Check available GPUs
python -c "import torch; print(f'Available GPUs: {torch.cuda.device_count()}')"

# Run distributed training
export WORLD_SIZE=2
export RANK=0
python src/ml/distributed_trainer.py &
export RANK=1
python src/ml/distributed_trainer.py &
wait
```

## Monitoring and Debugging

### Check GPU Utilization
```bash
# Monitor GPU usage during training/backtesting
watch -n 1 rocm-smi

# Check memory usage
rocm-smi --showmeminfo vram
```

### Check System Logs
```bash
# View training logs
tail -f LOGS/training.log

# View backtest logs
tail -f LOGS/backtest.log

# Check error logs
tail -f LOGS/error.log
```

### Common Issues and Solutions

**Issue 1: Out of Memory**
```bash
# Reduce batch size
python src/ml/train.py --batch_size 16

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"
```

**Issue 2: Slow Training**
```bash
# Enable mixed precision
python src/ml/train.py --mixed_precision

# Use more workers for data loading
python src/ml/train.py --num_workers 8
```

**Issue 3: Data Loading Errors**
```bash
# Validate data format
python -c "
import pandas as pd
df = pd.read_csv('DATA/ES/5min/your_data.csv')
print('Required columns: date, time, open, high, low, close, volume')
print(f'Your columns: {list(df.columns)}')
"
```

## Next Steps

### 1. Data Preparation
- Obtain high-quality ES futures historical data
- Ensure data covers at least 2 years
- Validate data quality and completeness

### 2. Model Training
- Experiment with different model architectures
- Optimize hyperparameters using grid search or Bayesian optimization
- Validate model performance on out-of-sample data

### 3. Strategy Optimization
- Test different risk parameters
- Optimize entry/exit thresholds
- Analyze performance across different market regimes

### 4. Advanced Features
- Implement walk-forward analysis
- Run Monte Carlo simulations
- Add additional technical indicators

### 5. Production Considerations
- Set up automated data feeds
- Implement real-time signal generation
- Add risk monitoring and alerts

## Getting Help

### Documentation References
- [Project Architecture Guide](Project_Architecture_Guide.md)
- [Installation Guide](Installation_Setup_Guide.md)
- [ML Training Guide](ML_Training_Guide.md)
- [Backtesting Guide](Backtesting_Strategy_Guide.md)
- [ROCm 7 Reference](ROCm_7_Distributed_ML_Reference.md)
- [PyTorch 2.10 Reference](PyTorch_2_4_Distributed_ML_Reference.md)

### Troubleshooting
1. Check logs in the `LOGS/` directory
2. Verify GPU availability with `rocm-smi`
3. Ensure data format is correct
4. Check configuration files for syntax errors

### Community and Support
- Review the comprehensive documentation for detailed explanations
- Check the test scripts for usage examples
- Monitor GPU performance during intensive operations

## Quick Reference Commands

```bash
# Environment Setup
source futures-vpoc-env/bin/activate
export PYTHONPATH=/workspace:$PYTHONPATH

# GPU Status
rocm-smi
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"

# Training
python src/ml/train.py --data_path DATA/ES/5min/ --epochs 50

# Backtesting
python src/analysis/run_ml_backtest.py --config config/quick_config.json

# Testing
python src/scripts/test_vpoc.py
python src/scripts/test_model.py
python src/scripts/test_backtest.py

# Distributed Training
export WORLD_SIZE=2
python src/ml/distributed_trainer.py
```

This quick start guide should get you up and running with the ES Futures VPOC strategy in under an hour. For more detailed information, refer to the comprehensive documentation guides.