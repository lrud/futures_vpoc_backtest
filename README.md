# Futures Trading Strategy Backtesting Project

## Overview

This project implements a systematic backtesting framework for a discretionary futures trading strategy focused on the Micro E-mini S&P 500 (MES) futures contract. The strategy identifies trading opportunities based on Volume Point of Control (VPOC) migration patterns, value area analysis, and market structure.

## Core Strategy Mechanics

The strategy combines multiple technical approaches:

1. **VPOC Migration Analysis**: Tracks the movement of volume distribution centers across consecutive sessions to identify higher timeframe trends.
2. **Value Area Identification**: Locates price zones containing approximately 70% of the session's volume.
3. **Reversion-to-Value Opportunities**: Seeks entry points when price revisits value areas aligned with the prevailing higher timeframe trend.
4. **Trend Confirmation**: Uses a minimum of 3 consecutive migrations in the same direction to establish trend robustness.

## Key Features

- **Volume Profile Calculation**: Implements market volume distribution analysis using price-volume histograms.
- **VPOC Migration Detection**: Identifies significant shifts in trading activity focus.
- **Value Area Analysis**: Calculates and visualizes the boundaries of high-volume price zones.
- **Trend Identification**: Detects sustained directional biases through consecutive VPOC migrations.
- **Trade Setup Detection**: Finds potential entry points based on value area tests.
- **Performance Analytics**: Generates comprehensive metrics on strategy effectiveness.

## Data Requirements

- **Time Series Data**: Minute bar data for the MES futures contract
- **Required Columns**: timestamp, open, high, low, close, volume, session
- **Session Types**: Regular Trading Hours (RTH) and Extended Trading Hours (ETH)

## Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/futures-backtesting.git
   cd futures-backtesting
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure data paths**
   - Update `DATA_FILE` in `simple-vpoc-calculator.py` to point to your data file
   - Update `OUTPUT_DIR` to your preferred results location

## Usage

### Basic VPOC Analysis

```bash
python simple-vpoc-calculator.py
```

This will:
- Load the futures data
- Calculate volume profiles for each session
- Identify VPOCs and value areas
- Detect VPOC migrations and trends
- Generate visualizations
- Output results to CSV files

### Interpreting Results

The analysis produces several key outputs:

1. **VPOC Data CSV**: Contains session-by-session VPOC and value area information
2. **VPOC Trends CSV**: Lists significant migration trends with direction and magnitude
3. **Volume Profile Visualizations**: Shows volume distribution, VPOC, and value areas
4. **VPOC Migration Chart**: Displays VPOC movement over time with trend indicators

## Example Output

```
Summary of VPOC Migration Trends:
=================================
Trend #1:
  Direction: DOWN
  Duration: 3 sessions
  Date Range: 2021-12-10 to 2021-12-15
  VPOC Change: -78.00 points

Trend #2:
  Direction: UP
  Duration: 5 sessions
  Date Range: 2021-12-20 to 2021-12-28
  VPOC Change: 238.00 points
```

## Future Development

Planned enhancements include:

1. **Trade Simulation**: Implementing the complete trading logic with entry/exit rules
2. **Performance Metrics**: Calculating profitability, risk-reward ratios, and drawdowns
3. **Optimization Framework**: Parameter tuning to maximize strategy performance
4. **Monte Carlo Simulation**: Stress testing the strategy under various market conditions
5. **Live Trading Integration**: Connecting to broker APIs for automated execution

## Dependencies

- Python 3.8+
- pandas
- numpy
- matplotlib
- scipy

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by auction market theory and market profile analysis
- Based on principles of volume-driven price action
- Developed for educational and research purposes
