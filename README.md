# ES Futures VPOC Strategy Backtester

## Overview
Advanced algorithmic trading strategy for E-mini S&P 500 (ES) futures that combines Volume Point of Control (VPOC) analysis with statistical validation. The strategy identifies high-probability trading opportunities by analyzing volume distribution patterns, value area migrations, and market microstructure.

## Strategy Performance (March 10, 2025)

### Original VPOC Strategy
- **Total Trades**: 1,649
- **Win Rate**: 73.38%
- **Total Profit**: $192,624.83
- **Profit Factor**: 3.04
- **Sharpe Ratio**: 5.31
- **Max Drawdown**: -1.00%

### ML-Enhanced Strategy
- **Total Trades**: 143
- **Win Rate**: 56.64%
- **Total Profit**: $44,116.45
- **Average Profit Per Trade**: $308.51 (vs $116.81 for original)
- **Profit Factor**: 2.34
- **Sharpe Ratio**: 4.52
- **Max Drawdown**: -3.60%

![Strategy Comparison](strategy_comparison.png)

## Strategy Components

### Volume Profile Analysis
- Calculates price-volume distributions for each trading session
- Identifies VPOC (price level with highest trading volume)
- Determines Value Area (70% of volume) boundaries
- Tracks VPOC migrations to identify institutional price acceptance

### Machine Learning Enhancement
- **Neural Network Architecture**: Optimized for AMD GPUs with SiLU activation
- **Feature Engineering**: Comprehensive feature set including:
  - Price momentum (10, 20, 50-day windows)
  - Volatility metrics
  - Volume trends
  - VPOC migrations
  - Range evolution
- **Signal Generation**: ML-filtered signals with confidence thresholds
- **Performance**: Higher per-trade profitability with more selective entry criteria
- **Distributed Training**: Supports multi-GPU training with AMD ROCm optimization

### Trade Setup Requirements
- **Long Entries**:
  - Price testing Value Area Low (VAL)
  - Confirmed upward VPOC migration (slope >2.47)
  - Strong statistical validation (R² >0.69)
  - Higher timeframe momentum aligned (Bayesian prob >53%)
  - Volume profile showing accumulation pattern
  - ML confidence score above threshold (for ML strategy)

- **Short Entries**:
  - Price testing Value Area High (VAH)
  - Mirror conditions of long entries
  - Additional validation through volatility windows
  - Institutional selling pressure confirmed
  - ML confidence score above threshold (for ML strategy)

### Risk Management
- Position sizing based on account volatility
- Dynamic stops using ATR and value area boundaries
- Maximum exposure limits per trade
- Multiple timeframe validation
- Capital preservation rules

### Mathematical Validation
- **Trend Analysis**: Slope 2.47, R² 0.69
- **Volatility Windows**:
  - 10-day: 71.97
  - 20-day: 57.73
  - 50-day: 76.86
- **Bayesian Probabilities**:
  - Upward: 53.47%
  - Downward: 46.53%

## Project Structure
```text
futures_vpoc_backtest/
├── src/                    # Refactored, modular implementation
│   ├── analysis/           # Analysis components
│   │   ├── `__init__.py`
│   │   ├── backtest.py     # Backtesting functionality
│   │   ├── math_utils.py   # Mathematical utilities
│   │   └── run_ml_backtest.py # Script to run ML-enhanced backtest
│   ├── config/             # Configuration settings
│   │   ├── `__init__.py`
│   │   └── settings.py     # Global settings and constants
│   ├── core/               # Core functionality
│   │   ├── `__init__.py`
│   │   ├── data.py         # Data management utilities
│   │   ├── signals.py      # Trading signal generation
│   │   └── vpoc.py         # VPOC calculation utilities
│   ├── ml/                 # Machine learning components
│   │   ├── `__init__.py`
│   │   ├── arguments.py            # Argument parsing for ML scripts
│   │   ├── backtest_integration.py # Integration with backtesting engine
│   │   ├── cmd_runner.py           # Command running utilities
│   │   ├── distributed_trainer.py  # Distributed training functionality (replaces distributed.py)
│   │   ├── evaluate_models.py      # Model evaluation scripts
│   │   ├── feature_engineering.py  # Feature extraction and selection
│   │   ├── model.py                # PyTorch model architecture
│   │   ├── train.py                # Main training script
│   │   ├── trainer_core.py         # Core training loop logic
│   │   └── trainer_utils.py        # Utility functions for training
│   ├── scripts/            # Scripts and tests
│   │   ├── test_backtest.py        # Backtest unit tests
│   │   ├── test_data_loader.py     # Data loader unit tests
│   │   ├── test_distributed.py     # Distributed training unit tests (may need update/removal)
│   │   ├── test_feature_engineering.py  # Feature engineering unit tests
│   │   ├── test_ML_total.py        # Integration test
│   │   ├── test_model.py           # Model architecture unit tests
│   │   ├── test_signal.py          # Signal generation tests
│   │   ├── test_vix_features.py    # VIX feature tests
│   │   └── test_vpoc.py            # VPOC calculation tests
│   ├── tests/              # Additional tests (e.g., integration)
│   │   └── test_ml_backtest.py     # ML backtest specific tests
│   └── utils/              # Utility functions
├── NOTEBOOKS/              # Original implementation
│   ├── VPOC.py            # Volume profile analysis & calculations
│   ├── STRATEGY.py        # Trading signal generation
│   ├── BACKTEST.py        # Performance testing & risk management
│   ├── MATH.py            # Statistical validation tools
│   ├── DATA_LOADER.py     # Data preprocessing utilities
│   ├── ML_TEST.py         # Original ML model architecture and training
│   └── ML_BACKTEST.py     # ML-enhanced backtesting framework
├── DATA/                  # Data directory (not included in repo)
├── TRAINING/              # Model training outputs (not included in repo)
└── .gitignore             # Git ignore rules
```
## Installation & Usage

### 1. Setup Environment
```bash
# Clone repository and install dependencies
git clone https://github.com/lrud/futures_vpoc_backtest.git
cd futures_vpoc_backtest
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Legacy Analysis Pipeline
```bash
cd NOTEBOOKS
python VPOC.py      # Volume profile calculations
python MATH.py      # Statistical validation
python DATA_LOADER.py  # Data preprocessing
python STRATEGY.py   # Signal generation
python BACKTEST.py   # Performance backtesting
```

### 3. Use Refactored ML Components
```python
from src.core.data import FuturesDataManager
from src.ml.train import main as train_main
import sys

# Example: Train ML model
# Run training with default parameters
sys.argv = ['train.py', '--epochs', '10', '--batch_size', '32']
train_main()

# Or load data and prepare features
data_manager = FuturesDataManager()
data = data_manager.load_futures_data()
print(f"Loaded {len(data)} records")
```

### 4. Run Tests
```bash
# Run individual test files
python src/scripts/test_backtest.py
python src/scripts/test_model.py
python src/scripts/test_vpoc.py
python src/tests/test_ml_backtest.py

# Run integration test
python src/scripts/test_ML_total.py
```

### Key Scripts
| Script | Purpose | Location |
|--------|---------|----------|
| `VPOC.py` | Volume profile analysis | `NOTEBOOKS/` |
| `train.py` | Main ML model training script | `src/ml/` |
| `run_ml_backtest.py` | Run ML-enhanced backtest | `src/analysis/` |
| `test_ML_total.py` | End-to-end validation | `src/scripts/` |

## Dependencies

### Key Dependencies:
- **`torch`**: Used in `src/ml/model.py` and ML training pipeline for neural networks
- **`pandas`**: Core data manipulation throughout the project
- **`numpy`**: Numerical computations and array operations
- **`pandas-ta`**: Technical analysis indicators in `NOTEBOOKS/STRATEGY.py`
- **`scikit-learn`**: Feature selection and ML preprocessing utilities
- **`matplotlib/seaborn`**: Visualization for backtest results and volume profiles

To install:
```bash
pip install -r requirements.txt
```

## ML Refactoring Notes

The ML components have been refactored from the original monolithic ML_TEST.py script into modular components:

![ML Pipeline Workflow Diagram](flow.png)

- **Feature Engineering**: (`src/ml/feature_engineering.py`) Extracts and selects features.
- **Model Architecture**: (`src/ml/model.py`) PyTorch neural network optimized for AMD GPUs.
- **Distributed Training**: (`src/ml/distributed_trainer.py`) Multi-GPU training using PyTorch's DDP.
- **Training Orchestration**: (`src/ml/train.py`, `src/ml/trainer_core.py`) Manages the training process.
- **Backtest Integration**: (`src/ml/backtest_integration.py`) Connects ML predictions to the backtester.
- **Model Management**: (`src/ml/trainer_utils.py`, `src/ml/evaluate_models.py`) Utilities for saving, loading, and evaluating models.

This refactoring improves:
- **Maintainability**: Easier to understand and modify individual components
- **Testability**: Each component has dedicated unit tests
- **Extensibility**: New features can be added without modifying existing code

## License
MIT License - See LICENSE file for details
