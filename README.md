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
```
futures_vpoc_backtest/
├── NOTEBOOKS/             # Core implementation
│   ├── VPOC.py            # Volume profile analysis & calculations
│   ├── STRATEGY.py        # Trading signal generation
│   ├── BACKTEST.py        # Performance testing & risk management
│   ├── MATH.py            # Statistical validation tools
│   ├── DATA_LOADER.py     # Data preprocessing utilities
│   ├── ML_TEST.py         # ML model architecture and training
│   └── ML_BACKTEST.py     # ML-enhanced backtesting framework
└── .gitignore             # Git ignore rules

Required Data Structure (not included):
- Minute-level ES futures data
- Columns: timestamp, open, high, low, close, volume
- Format: CSV with headers
- Date range: 2021-12-05 to 2025-02-27
```

## Installation & Usage

1. **Setup Environment**
```bash
git clone https://github.com/lrud/futures_vpoc_backtest.git
cd futures_vpoc_backtest
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. **Run Analysis Pipeline**
```bash
cd NOTEBOOKS

# Generate VPOC and mathematical analysis
python VPOC.py      # Calculate volume profiles and value areas
python MATH.py      # Perform statistical validation

# Process and validate data
python DATA_LOADER.py  # Clean and prepare market data

# Generate signals and evaluate performance
python STRATEGY.py   # Generate trading signals with confidence scores
python BACKTEST.py   # Run performance analysis with risk management
```

Each script performs specific tasks:
- VPOC.py: Generates volume profiles, VPOCs, and value areas
- MATH.py: Calculates trend slopes, R-squared values, and Bayesian probabilities
- DATA_LOADER.py: Preprocesses market data and validates data integrity
- STRATEGY.py: Combines analysis to generate high-probability trade signals
- BACKTEST.py: Tests strategy with realistic commission and slippage

## Dependencies
```txt
pandas==2.0.0
numpy==1.24.0
scipy==1.10.0
matplotlib==3.7.0
seaborn==0.12.2
pandas-ta==0.3.14b
statsmodels==0.14.0
```

## License
MIT License - See LICENSE file for details