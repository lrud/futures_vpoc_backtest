# ES Futures VPOC Strategy - Installation and Setup Guide

## System Requirements

### Hardware Requirements
**Minimum Configuration**:
- CPU: 8-core processor (AMD Ryzen 7 or Intel i7 recommended)
- GPU: AMD Radeon RX 7900 XT or higher (ROCm 7.0 compatible)
- RAM: 32GB DDR4/DDR5
- Storage: 1TB NVMe SSD (for data and model storage)
- Network: Stable internet connection for market data

**Recommended Configuration**:
- CPU: AMD Ryzen 9 or Threadripper
- GPU: AMD Instinct MI300X (for production) or dual RX 7900 XT
- RAM: 64GB+ DDR5
- Storage: 2TB NVMe SSD + additional storage for historical data
- Network: High-speed connection with low latency

### Software Requirements
**Operating System**:
- Ubuntu 22.04 LTS (recommended)
- RHEL 8/9 (for production)
- Windows 11 with WSL2 (development only)

**Required Software**:
- ROCm 7.0 or later
- Python 3.10+
- PyTorch 2.10 with ROCm support
- Git
- Docker (optional, for containerized deployment)

## Installation Steps

### 1. Install ROCm 7.0

#### Ubuntu 22.04 LTS Installation
```bash
# Add AMD GPU repository
sudo apt-get update
sudo apt-get install "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)"
wget https://repo.radeon.com/amdgpu-install/6.0.2/ubuntu/jammy/amdgpu-install_6.0.2.60002-1_all.deb
sudo apt-get install ./amdgpu-install_6.0.2.60002-1_all.deb

# Install ROCm 7.0
sudo amdgpu-install --usecase=rocm --no-dkms

# Add user to render and video groups
sudo usermod -a -G render,video $LOGNAME

# Reboot to load kernel modules
sudo reboot
```

#### Verify ROCm Installation
```bash
# Check GPU detection
rocm-smi

# Expected output (example for RX 7900 XT):
# ================================== ROCm System Management Interface ==================================
# GPU  Temp   PwrJls  SCLK      MCLK    Fan     Perf  PwrCap  VRAM%  GPU%
# 0    45.0c  300.0W  2500Mhz   2500Mhz  25.0%   auto  355W    25%    15%
```

### 2. Install Python Environment

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install python3-dev python3-pip python3-venv

# Create virtual environment
python3 -m venv futures-vpoc-env
source futures-vpoc-env/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### 3. Install PyTorch 2.10 with ROCm Support

```bash
# Install PyTorch 2.10 with ROCm 7.0
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.0

# Alternative: Install nightly build for latest features
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm7.0

# Verify PyTorch installation
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'ROCm version: {torch.version.hip}')"
```

### 4. Install Project Dependencies

```bash
# Clone the repository (if not already done)
git clone <your-repository-url>
cd es-futures-vpoc

# Install core dependencies
pip install numpy pandas matplotlib seaborn
pip install scikit-learn scipy
pip install tqdm

# Install ML/Deep Learning dependencies
pip install transformers>=4.35.0
pip install datasets
pip install wandb
pip install tensorboard

# Install ROCm-specific packages
pip install flash-attn --no-build-isolation
pip install triton==2.10.0

# Install project-specific dependencies
pip install -r requirements.txt  # Create this file if it doesn't exist
```

### 5. Project Setup

```bash
# Navigate to project directory
cd /workspace

# Set up Python path
export PYTHONPATH=/workspace:$PYTHONPATH

# Create necessary directories
mkdir -p DATA CLEANED RESULTS STRATEGY BACKTEST TRAINING LOGS

# Set permissions
chmod +x src/scripts/*.py
```

## Configuration Setup

### 1. Environment Configuration

Create `.env` file:
```bash
# ROCm Configuration
HIP_VISIBLE_DEVICES=0,1
HSA_OVERRIDE_GFX_VERSION=9.4.2
MIOPEN_USER_DB_PATH=/tmp/miopen_user_db
MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen_cache

# GPU Optimization
GPU_MAX_HW_QUEUES=8
GPU_SINGLE_ALLOC_PERCENT=90
PYTORCH_ROCM_ARCH=gfx1100

# Project Configuration
DATA_DIR=/workspace/DATA
RESULTS_DIR=/workspace/RESULTS
TRAINING_DIR=/workspace/TRAINING
LOG_LEVEL=INFO

# Trading Configuration
INITIAL_CAPITAL=100000
RISK_PER_TRADE=0.01
COMMISSION_PER_TRADE=10
```

### 2. Configuration File Setup

Create `config/user_config.json`:
```json
{
  "trading": {
    "initial_capital": 100000,
    "risk_per_trade": 0.01,
    "commission_per_trade": 10,
    "slippage": 0.25,
    "max_position_size": 10
  },
  "ml": {
    "lookback_periods": [5, 10, 20, 50],
    "prediction_threshold": 0.5,
    "confidence_threshold": 70,
    "batch_size": 32,
    "learning_rate": 0.001
  },
  "system": {
    "device_ids": [0, 1],
    "num_workers": 8,
    "pin_memory": true,
    "mixed_precision": true
  }
}
```

## Verification and Testing

### 1. GPU and PyTorch Verification

```python
# Create test script: test_setup.py
import torch
import torch.nn as nn

def test_gpu_setup():
    print("=== GPU Setup Verification ===")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"ROCm Version: {torch.version.hip}")
    print(f"CUDA Available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")

        # Test tensor operations
        print("\n=== Testing Tensor Operations ===")
        device = torch.device('cuda:0')
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        z = torch.matmul(x, y)
        print(f"Matrix multiplication: ✓")

        # Test torch.compile
        if hasattr(torch, 'compile'):
            def simple_model(x):
                return torch.nn.functional.linear(x, torch.randn(1000, 1000, device=device))

            compiled = torch.compile(simple_model)
            result = compiled(x)
            print(f"torch.compile: ✓")
        else:
            print("torch.compile: ✗ (not available)")

if __name__ == "__main__":
    test_gpu_setup()
```

Run the test:
```bash
python test_setup.py
```

### 2. Project Components Verification

```bash
# Test VPOC calculation
python src/scripts/test_vpoc.py

# Test feature engineering
python src/scripts/test_feature_engineering.py

# Test ML model
python src/scripts/test_model.py

# Test distributed training
python src/scripts/test_distributed.py
```

## Data Setup

### 1. Historical Data Acquisition

The system requires ES futures historical data with the following format:
- Date, Time, Open, High, Low, Close, Volume
- 1-minute or 5-minute intervals
- Regular Trading Hours (RTH) data

### 2. Data Directory Structure

```
/workspace/DATA/
├── ES/                    # E-mini S&P 500 data
│   ├── 1min/             # 1-minute bars
│   ├── 5min/             # 5-minute bars
│   └── daily/            # Daily bars
├── CLEANED/              # Processed data
└── external/             # External data sources
```

### 3. Data Validation

```python
# Create data validation script
import pandas as pd
from pathlib import Path

def validate_data_format(data_path):
    """Validate data format and completeness"""
    expected_columns = ['date', 'time', 'open', 'high', 'low', 'close', 'volume']

    df = pd.read_csv(data_path)

    # Check columns
    if not all(col in df.columns for col in expected_columns):
        raise ValueError(f"Missing required columns. Expected: {expected_columns}")

    # Check data types
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column {col} should be numeric")

    # Check for missing values
    if df.isnull().any().any():
        print("Warning: Data contains missing values")

    # Check data ranges
    if (df['high'] < df['low']).any():
        raise ValueError("High prices cannot be lower than low prices")

    if (df['high'] < df['open']) | (df['high'] < df['close']) | \
       (df['low'] > df['open']) | (df['low'] > df['close']):
        print("Warning: OHLC data may have inconsistencies")

    print(f"Data validation passed: {len(df)} records")
    return True
```

## Common Installation Issues

### 1. ROCm Installation Problems

**Issue**: ROCm not detecting GPU
```bash
# Solution: Check kernel modules
lsmod | grep amdgpu
sudo modprobe amdgpu

# Reinstall if necessary
sudo apt-get remove --purge amdgpu-install
sudo apt-get install ./amdgpu-install_6.0.2.60002-1_all.deb
sudo amdgpu-install --usecase=rocm --no-dkms
```

**Issue**: Permission denied errors
```bash
# Solution: Add user to correct groups
sudo usermod -a -G render,video $LOGNAME
newgrp render
newgrp video
```

### 2. PyTorch ROCm Issues

**Issue**: Import errors with torch
```bash
# Solution: Verify ROCm-PyTorch compatibility
pip uninstall torch torchvision torchaudio
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.0
```

**Issue**: CUDA out of memory
```python
# Solution: Adjust batch size and memory management
import torch

# Set memory fraction
torch.cuda.set_per_process_memory_fraction(0.8)

# Clear cache
torch.cuda.empty_cache()
```

### 3. Project-Specific Issues

**Issue**: Module import errors
```bash
# Solution: Set PYTHONPATH
export PYTHONPATH=/workspace:$PYTHONPATH

# Or add to ~/.bashrc
echo 'export PYTHONPATH=/workspace:$PYTHONPATH' >> ~/.bashrc
```

**Issue**: Data loading errors
```bash
# Solution: Check file permissions and paths
chmod -R 755 /workspace/DATA
ls -la /workspace/DATA/
```

## Performance Optimization

### 1. GPU Memory Optimization

```python
# Optimize GPU memory usage
import torch

# Set memory pool
memory_pool = torch.cuda.memory.MemoryPool(torch.device('cuda'))
torch.cuda.set_memory_pool(memory_pool)

# Enable memory mapping
torch.multiprocessing.set_sharing_strategy('file_system')
```

### 2. Multi-GPU Setup

```bash
# Configure for dual GPU setup
export HIP_VISIBLE_DEVICES=0,1
export CUDA_VISIBLE_DEVICES=0,1
export WORLD_SIZE=2
export RANK=0  # For first process
```

### 3. Compilation Optimizations

```python
# Enable PyTorch compilation
model = torch.compile(model, mode='max-autotune')
```

## Monitoring and Maintenance

### 1. System Monitoring

```bash
# Monitor GPU usage
watch -n 1 rocm-smi

# Monitor system resources
htop
iostat -x 1
```

### 2. Log Management

```bash
# Set up log rotation
sudo nano /etc/logrotate.d/futures-vpoc

# Content:
/workspace/LOGS/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 user user
}
```

## Security Considerations

### 1. Environment Security

```bash
# Set appropriate file permissions
chmod 600 config/user_config.json
chmod 700 src/ml/checkpoints/
chmod 700 DATA/
```

### 2. API Keys and Credentials

Use environment variables for sensitive information:
```bash
export DATA_PROVIDER_API_KEY="your_api_key_here"
export TRADING_API_SECRET="your_secret_here"
```

## Next Steps

After successful installation:

1. **Load Historical Data**: Populate the DATA directory with ES futures data
2. **Run Initial Tests**: Execute all test scripts to verify functionality
3. **Train ML Model**: Use the provided training scripts
4. **Run Backtests**: Test the strategy on historical data
5. **Paper Trading**: Implement paper trading before live deployment

For detailed usage instructions, refer to the [User Guide](User_Guide.md) and [Architecture Guide](Project_Architecture_Guide.md).