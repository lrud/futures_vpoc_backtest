# ES Futures VPOC Strategy - Project Architecture Guide

## Overview

This document provides a comprehensive overview of the ES Futures VPOC (Volume Point of Control) algorithmic trading system architecture. The system combines traditional technical analysis with advanced machine learning techniques, specifically optimized for AMD GPU hardware using ROCm.

## System Architecture

```
ES Futures VPOC Trading System
├── Data Layer
│   ├── Market Data Ingestion
│   ├── Data Cleaning & Validation
│   └── Historical Data Storage
├── Analysis Layer
│   ├── Volume Profile Analysis (VPOC)
│   ├── Mathematical Validation
│   └── Signal Generation
├── Machine Learning Layer
│   ├── Feature Engineering
│   ├── Model Training (ROCm Optimized)
│   └── Model Inference
├── Backtesting Layer
│   ├── Strategy Simulation
│   ├── Performance Analytics
│   └── Risk Management
└── Execution Layer
    ├── Order Management
    ├── Position Management
    └── Real-time Monitoring
```

## Core Components

### 1. Volume Profile Analysis (`src/core/vpoc.py`)
**Purpose**: GPU-accelerated volume profile calculations for identifying market value areas

**Key Features**:
- Multi-GPU VPOC calculations using PyTorch tensors
- ROCm 7.0 optimizations for AMD 7900 XT and MI300X
- Value Area High/Low (VAH/VAL) calculations
- VPOC migration tracking and analysis

**GPU Optimizations**:
```python
# ROCm 7.0 specific optimizations
os.environ['PYTORCH_ROCM_WAVE32_MODE'] = '1'
os.environ['HSA_ENABLE_SDMA'] = '0'
os.environ['GPU_MAX_HW_QUEUES'] = '8'
```

### 2. Mathematical Validation (`src/analysis/math_utils.py`)
**Purpose**: Statistical validation of VPOC signals to ensure high-quality trade entries

**Key Functions**:
- `validate_trend()`: Linear regression analysis on VPOC trends
- Bayesian probability calculations for directional confirmation
- Runs test for randomness validation
- R² and slope analysis for trend quality

**Validation Criteria**:
- Trend slope > 0 (for uptrends)
- R² > 0.6 (trend quality)
- Bayesian probability > 53%
- Runs test p-value < 0.05

### 3. Signal Generation (`src/core/signals.py`)
**Purpose**: Integration of VPOC analysis with mathematical validation to generate trading signals

**Signal Requirements**:
1. VPOC trend analysis confirmation
2. High-quality trend validation (R² > 0.6)
3. Market condition filter (volatility or Bayesian)
4. Confidence threshold > 60%

**Trade Logic**:
- **Entry**: Value Area Low (VAL) for buys, Value Area High (VAH) for sells
- **Target**: VAH for buys, VAL for sells (2:1 risk-reward)
- **Stop Loss**: Dynamic based on daily volatility
- **Position Sizing**: Volatility-adjusted with 1% maximum risk

### 4. Machine Learning Enhancement (`src/ml/`)
**Purpose**: Neural network filtering to improve signal quality and selectivity

#### Model Architecture (`src/ml/model.py`)
```python
class AMDOptimizedFuturesModel(nn.Module):
    """ROCm 7.0 optimized neural network for futures trading"""

    def __init__(self, input_size, hidden_sizes, output_size):
        # 64-byte memory alignment for AMD GPU optimization
        # Wave32 mode optimization for RDNA3 architecture
        # Mixed precision BF16 training support
```

**AMD GPU Optimizations**:
- Memory alignment: 64-byte boundaries
- Wave32 mode for RDNA3 efficiency
- Flash Attention v3 support
- Mixed precision training (BF16)

#### Feature Engineering (`src/ml/feature_engineering.py`)
**Feature Categories**:
- **Price Features**: Multi-timeframe momentum calculations
- **Volatility Features**: GARCH-style volatility modeling
- **Volume Features**: VPOC migration patterns
- **Technical Indicators**: RSI, MACD, Bollinger Bands
- **Statistical Features**: Robust scaling, log transforms

#### Distributed Training (`src/ml/distributed_trainer.py`)
**ROCm 7.0 Multi-GPU Setup**:
```python
# Dual 7900 XT optimization
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx1100'
os.environ['HIP_VISIBLE_DEVICES'] = str(rank)
os.environ['GPU_SINGLE_ALLOC_PERCENT'] = '90'
```

### 5. Backtesting Engine (`src/analysis/backtest.py`)
**Purpose**: Realistic simulation of trading strategy performance

**Simulation Parameters**:
- Initial Capital: $100,000
- Commission: $10 per trade
- Slippage: 0.25 points (1 tick)
- Risk per Trade: 1% maximum
- Margin Requirements: 10% day, 15% overnight

**Performance Metrics**:
- Win Rate, Profit Factor, Sharpe Ratio
- Maximum Drawdown analysis
- Trade distribution statistics
- Risk-adjusted returns

## Data Flow Architecture

```
Market Data → Data Cleaning → VPOC Analysis → Math Validation → Signal Generation
                    ↓                                              ↓
            Feature Engineering ← ML Model Training ← Historical Performance
                    ↓                                              ↓
         ML-Enhanced Signals → Backtesting → Strategy Optimization → Live Trading
```

## Configuration Management

### Settings (`src/config/settings.py`)
**Global Configuration**:
```python
class Settings:
    # Trading Parameters
    INITIAL_CAPITAL = 100000
    RISK_PER_TRADE = 0.01
    COMMISSION_PER_TRADE = 10
    SLIPPAGE = 0.25

    # ML Parameters
    LOOKBACK_PERIODS = [5, 10, 20, 50]
    PREDICTION_THRESHOLD = 0.5
    CONFIDENCE_THRESHOLD = 70

    # System Paths
    DATA_DIR = BASE_DIR / 'DATA'
    RESULTS_DIR = BASE_DIR / 'RESULTS'
    TRAINING_DIR = BASE_DIR / 'TRAINING'
```

## GPU Computing Architecture

### ROCm 7.0 Integration
The system is specifically optimized for AMD GPU hardware:

**Supported Hardware**:
- AMD Radeon RX 7900 XT (RDNA3)
- AMD Instinct MI300X (CDNA3)
- Multi-GPU configurations

**Performance Optimizations**:
- PyTorch 2.10 ROCm backend
- HCCL for multi-GPU communication
- Memory pooling and management
- Kernel fusion and optimization

### Distributed Training Setup
```bash
# Multi-GPU training configuration
export HIP_VISIBLE_DEVICES=0,1
export PYTORCH_ROCM_ARCH=gfx1100
export ROCM_HCCL_DEBUG=WARN
export TORCH_COMPILE_BACKEND=inductor
```

## Risk Management Framework

### Position Sizing Algorithm
```python
def calculate_position_size(account_balance, volatility, risk_per_trade=0.01):
    """Calculate position size based on volatility and risk tolerance"""
    atr_multiplier = 2.0  # Stop loss at 2x ATR
    position_risk = account_balance * risk_per_trade
    position_size = position_risk / (atr_multiplier * volatility)
    return min(position_size, MAX_POSITION_SIZE)
```

### Risk Controls
- Maximum 1% risk per trade
- Volatility-adjusted position sizing
- Dynamic stop-loss based on ATR
- Maximum position size limits
- Margin requirement monitoring

## Performance Monitoring

### Key Metrics Dashboard
1. **Trading Performance**:
   - Win Rate, Profit Factor, Sharpe Ratio
   - Average Trade P&L
   - Maximum Drawdown

2. **System Performance**:
   - GPU utilization and memory usage
   - Training throughput (samples/second)
   - Inference latency

3. **Model Performance**:
   - Prediction accuracy
   - Confidence score distribution
   - Feature importance analysis

## Deployment Architecture

### Development Environment
```bash
# ROCm 7.0 + PyTorch 2.10 setup
conda create -n futures-vpoc python=3.10
conda activate futures-vpoc
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.0
```

### Production Considerations
- GPU resource monitoring
- Model versioning and rollback
- Real-time data feed integration
- Order execution system integration
- Compliance and audit logging

## Extensibility

### Adding New Features
1. **New Technical Indicators**: Add to `src/analysis/`
2. **ML Models**: Extend `src/ml/model.py`
3. **Data Sources**: Modify `src/core/data.py`
4. **Risk Rules**: Update `src/config/settings.py`

### Configuration Management
- Environment-specific settings
- Runtime parameter adjustment
- Model hyperparameter optimization
- Strategy variant testing

## Troubleshooting Guide

### Common Issues
1. **GPU Memory**: Adjust batch sizes and model architecture
2. **ROCm Compatibility**: Verify driver versions and PyTorch builds
3. **Data Quality**: Implement validation checks
4. **Performance**: Profile GPU utilization and memory usage

### Debugging Tools
- ROCm profiling tools (rocm-smi, rocprof)
- PyTorch profiler integration
- Custom logging and monitoring
- Performance benchmarking scripts

This architecture guide serves as the foundation for understanding, extending, and maintaining the ES Futures VPOC trading system.