# ES Futures VPOC Strategy - Comprehensive Backtesting Guide

## Overview

This guide provides detailed instructions for backtesting the ES Futures VPOC (Volume Point of Control) trading strategy. The backtesting system incorporates realistic market conditions, sophisticated risk management, and comprehensive performance analytics.

## Backtesting Architecture

### Core Components

```
Backtesting System
├── Data Management
│   ├── Historical Data Loading
│   ├── Data Validation & Cleaning
│   └── Market Session Filtering
├── Strategy Engine
│   ├── VPOC Calculation
│   ├── Signal Generation
│   ├── ML Enhancement
│   └── Risk Management
├── Simulation Engine
│   ├── Order Execution
│   ├── Position Management
│   ├── Commission & Slippage
│   └── Margin Calculations
└── Analytics
    ├── Performance Metrics
    ├── Risk Analysis
    ├── Trade Statistics
    └── Visualization
```

### System Requirements for Backtesting

**Minimum Requirements**:
- CPU: 8-core processor
- RAM: 32GB (for large datasets)
- Storage: 500GB SSD (for historical data)
- GPU: AMD RX 7900 XT or higher (for VPOC calculations)

**Recommended Requirements**:
- CPU: AMD Ryzen 9 or Threadripper
- RAM: 64GB+
- Storage: 1TB+ NVMe SSD
- GPU: Dual AMD GPUs for parallel processing

## Data Preparation

### 1. Historical Data Requirements

**Required Data Format**:
```python
# CSV format expected
Date,Time,Open,High,Low,Close,Volume
2024-01-02,09:30,4500.25,4502.50,4498.75,4501.00,1500000
2024-01-02,09:31,4501.00,4503.25,4499.50,4502.75,1200000
```

**Data Specifications**:
- **Instrument**: E-mini S&P 500 (ES) futures
- **Timeframe**: 1-minute or 5-minute bars
- **Session**: Regular Trading Hours (RTH) 9:30 AM - 4:00 PM ET
- **History**: Minimum 2 years for robust backtesting
- **Fields**: Date, Time, Open, High, Low, Close, Volume

### 2. Data Validation Script

```python
# data_validator.py
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, time

class DataValidator:
    def __init__(self, data_path):
        self.data_path = Path(data_path)

    def validate_data_integrity(self, df):
        """Validate data integrity and completeness"""

        # Check required columns
        required_columns = ['date', 'time', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Check data types
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Check for OHLC consistency
        invalid_ohlc = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        )

        if invalid_ohlc.any():
            print(f"Warning: Found {invalid_ohlc.sum()} bars with invalid OHLC data")

        # Check for missing values
        missing_data = df.isnull().sum()
        if missing_data.any():
            print(f"Missing data: {missing_data[missing_data > 0]}")
            # Forward fill missing prices
            df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].ffill()
            df['volume'] = df['volume'].fillna(0)

        # Remove duplicate timestamps
        df = df.drop_duplicates(subset=['date', 'time'], keep='first')

        # Sort by date and time
        df = df.sort_values(['date', 'time']).reset_index(drop=True)

        return df

    def filter_trading_hours(self, df):
        """Filter data to regular trading hours (RTH)"""

        # Convert to datetime
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])

        # Filter for RTH (9:30 AM - 4:00 PM ET)
        start_time = time(9, 30)
        end_time = time(16, 0)

        # Extract time component
        df['time_only'] = df['datetime'].dt.time

        # Filter by trading hours
        rth_mask = (df['time_only'] >= start_time) & (df['time_only'] <= end_time)
        df_rth = df[rth_mask].copy()

        print(f"Filtered {len(df)} records to {len(df_rth)} RTH records")

        return df_rth

    def detect_data_gaps(self, df):
        """Detect gaps in the data"""

        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        df = df.sort_values('datetime')

        # Expected 1-minute intervals
        expected_interval = pd.Timedelta(minutes=1)

        # Calculate actual intervals
        actual_intervals = df['datetime'].diff()

        # Find gaps larger than expected
        gaps = actual_intervals[actual_intervals > expected_interval * 2]

        if len(gaps) > 0:
            print(f"Found {len(gaps)} data gaps:")
            for idx, gap in gaps.items():
                print(f"  Gap at {df.loc[idx-1, 'datetime']}: {gap}")

        return gaps

    def validate_volume_distribution(self, df):
        """Validate volume distribution for anomalies"""

        volume_stats = {
            'mean': df['volume'].mean(),
            'median': df['volume'].median(),
            'std': df['volume'].std(),
            'min': df['volume'].min(),
            'max': df['volume'].max()
        }

        # Check for zero volume bars
        zero_volume_bars = (df['volume'] == 0).sum()
        if zero_volume_bars > 0:
            print(f"Warning: Found {zero_volume_bars} bars with zero volume")

        # Check for volume anomalies (3+ standard deviations)
        volume_threshold = volume_stats['mean'] + 3 * volume_stats['std']
        high_volume_bars = df[df['volume'] > volume_threshold]

        if len(high_volume_bars) > 0:
            print(f"Found {len(high_volume_bars)} high-volume bars (>{volume_threshold:.0f})")

        return volume_stats

    def generate_data_report(self, df):
        """Generate comprehensive data quality report"""

        print("=== Data Quality Report ===")
        print(f"Total records: {len(df)}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")

        # Trading days analysis
        trading_days = df['date'].nunique()
        print(f"Trading days: {trading_days}")

        # Price range analysis
        print(f"Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")

        # Volume analysis
        volume_stats = self.validate_volume_distribution(df)
        print(f"Average volume: {volume_stats['mean']:,.0f}")
        print(f"Volume std: {volume_stats['std']:,.0f}")

        # Check for gaps
        gaps = self.detect_data_gaps(df)

        return {
            'total_records': len(df),
            'trading_days': trading_days,
            'price_range': (df['low'].min(), df['high'].max()),
            'volume_stats': volume_stats,
            'data_gaps': len(gaps)
        }

# Usage example
def prepare_es_data(data_path):
    """Prepare ES futures data for backtesting"""

    validator = DataValidator(data_path)

    # Load data
    df = pd.read_csv(data_path)

    # Validate and clean
    df = validator.validate_data_integrity(df)

    # Filter trading hours
    df = validator.filter_trading_hours(df)

    # Generate report
    report = validator.generate_data_report(df)

    # Save cleaned data
    output_path = data_path.replace('.csv', '_cleaned.csv')
    df.to_csv(output_path, index=False)

    print(f"Cleaned data saved to: {output_path}")

    return df, report
```

## Backtesting Configuration

### 1. Strategy Parameters

```python
# backtest_config.py
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class BacktestConfig:
    # Trading Parameters
    initial_capital: float = 100000.0
    commission_per_trade: float = 10.0
    slippage_per_contract: float = 0.25  # 1 tick for ES futures
    risk_per_trade: float = 0.01  # 1% of capital

    # Margin Requirements
    day_margin_requirement: float = 0.10  # 10% for day trading
    overnight_margin_requirement: float = 0.15  # 15% overnight

    # Position Management
    max_position_size: float = 10.0  # Maximum contracts
    min_capital_buffer: float = 0.2  # 20% capital buffer

    # VPOC Parameters
    lookback_periods: List[int] = None
    vpoc_sensitivity: float = 0.02  # 2% price sensitivity
    value_area_percentage: float = 0.70  # 70% value area

    # Signal Parameters
    prediction_threshold: float = 0.5
    confidence_threshold: float = 60.0  # 60% confidence required
    trend_r2_threshold: float = 0.6
    bayesian_probability_threshold: float = 0.53

    # Risk Management
    max_drawdown_limit: float = 0.20  # 20% max drawdown
    consecutive_loss_limit: int = 5  # Stop after 5 consecutive losses

    # ML Enhancement
    use_ml_filter: bool = True
    ml_model_path: str = "TRAINING/best_model.pt"

    def __post_init__(self):
        if self.lookback_periods is None:
            self.lookback_periods = [5, 10, 20, 50]

@dataclass
class SimulationConfig:
    # Data Configuration
    data_path: str = "DATA/ES/5min/"
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    # Simulation Settings
    enable_realistic_fills: bool = True
    enable_partial_fills: bool = True
    fill_probability: float = 0.95  # 95% fill probability

    # Performance Tracking
    save_trades: bool = True
    save_equity_curve: bool = True
    generate_reports: bool = True

    # Output Configuration
    output_dir: str = "BACKTEST_RESULTS/"
    save_frequency: int = 1000  # Save every 1000 trades
```

### 2. Backtesting Engine

```python
# backtesting_engine.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import torch

from src.core.vpoc import VolumeProfileAnalyzer
from src.core.signals import SignalGenerator
from src.core.data import DataManager
from src.utils.logging import get_logger

class BacktestingEngine:
    def __init__(self, config: BacktestConfig, sim_config: SimulationConfig):
        self.config = config
        self.sim_config = sim_config
        self.logger = get_logger(__name__)

        # Initialize components
        self.vpoc_analyzer = VolumeProfileAnalyzer(
            price_precision=0.25,
            device_ids=[0, 1] if torch.cuda.device_count() > 1 else [0]
        )

        self.signal_generator = SignalGenerator(
            vpoc_analyzer=self.vpoc_analyzer,
            config=config
        )

        self.data_manager = DataManager()

        # Initialize state variables
        self.reset_state()

    def reset_state(self):
        """Reset backtesting state"""

        # Account state
        self.current_capital = self.config.initial_capital
        self.available_capital = self.config.initial_capital
        self.allocated_capital = 0.0

        # Position state
        self.current_position = 0.0  # Number of contracts
        self.entry_price = 0.0
        self.entry_date = None
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0

        # Trade tracking
        self.trades = []
        self.equity_curve = []
        self.daily_returns = []

        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.peak_capital = self.config.initial_capital

        # Risk management
        self.consecutive_losses = 0
        self.last_trade_date = None

    def calculate_position_size(self, price: float, volatility: float) -> float:
        """Calculate position size based on risk management rules"""

        # Calculate risk amount
        risk_amount = self.current_capital * self.config.risk_per_trade

        # Calculate stop loss distance (2x ATR as proxy)
        stop_distance = volatility * 2.0

        # Calculate position size
        position_value = risk_amount / stop_distance
        position_contracts = position_value / price

        # Apply limits
        max_contracts_by_capital = self.available_capital / (price * self.config.day_margin_requirement)
        max_contracts_by_rule = self.config.max_position_size

        position_contracts = min(
            position_contracts,
            max_contracts_by_capital,
            max_contracts_by_rule
        )

        # Round to nearest whole contract
        position_contracts = max(1, int(position_contracts))

        return position_contracts

    def calculate_margin_requirement(self, position: float, price: float) -> float:
        """Calculate margin requirement for position"""

        if abs(position) <= 0:
            return 0.0

        # Check if position is held overnight (simplified)
        is_overnight = False  # This would need actual time logic

        margin_rate = (self.config.overnight_margin_requirement if is_overnight
                      else self.config.day_margin_requirement)

        return abs(position) * price * margin_rate

    def execute_trade(self, signal: Dict, current_bar: pd.Series) -> Optional[Dict]:
        """Execute trade based on signal"""

        action = signal['action']  # 'BUY', 'SELL', or 'HOLD'
        price = current_bar['close']
        volume = current_bar['volume']

        # Skip if insufficient capital or risk limits exceeded
        if self.consecutive_losses >= self.config.consecutive_loss_limit:
            self.logger.warning(f"Skipping trade due to consecutive loss limit: {self.consecutive_losses}")
            return None

        # Calculate position size
        volatility = signal.get('volatility', price * 0.01)  # 1% default volatility
        position_size = self.calculate_position_size(price, volatility)

        trade = None

        if action == 'BUY' and self.current_position <= 0:
            # Close existing short position
            if self.current_position < 0:
                self._close_position(price, current_bar['datetime'])

            # Open long position
            if self.available_capital > price * position_size * self.config.day_margin_requirement:
                trade = self._open_long_position(position_size, price, current_bar, signal)

        elif action == 'SELL' and self.current_position >= 0:
            # Close existing long position
            if self.current_position > 0:
                self._close_position(price, current_bar['datetime'])

            # Open short position
            if self.available_capital > price * position_size * self.config.day_margin_requirement:
                trade = self._open_short_position(position_size, price, current_bar, signal)

        return trade

    def _open_long_position(self, size: float, price: float, bar: pd.Series, signal: Dict) -> Dict:
        """Open long position"""

        # Apply slippage
        entry_price = price + self.config.slippage_per_contract
        commission = self.config.commission_per_trade

        # Calculate margin requirement
        margin_required = size * entry_price * self.config.day_margin_requirement

        # Update state
        self.current_position = size
        self.entry_price = entry_price
        self.entry_date = bar['datetime']
        self.allocated_capital = margin_required
        self.available_capital -= margin_required + commission

        # Create trade record
        trade = {
            'datetime': bar['datetime'],
            'action': 'BUY',
            'contracts': size,
            'price': entry_price,
            'commission': commission,
            'margin': margin_required,
            'signal_strength': signal.get('strength', 0.0),
            'confidence': signal.get('confidence', 0.0),
            'vpoc_trend': signal.get('vpoc_trend', 0.0),
        }

        self.trades.append(trade)
        self.total_trades += 1

        self.logger.info(f"Opened LONG position: {size} contracts @ ${entry_price:.2f}")

        return trade

    def _open_short_position(self, size: float, price: float, bar: pd.Series, signal: Dict) -> Dict:
        """Open short position"""

        # Apply slippage
        entry_price = price - self.config.slippage_per_contract
        commission = self.config.commission_per_trade

        # Calculate margin requirement
        margin_required = size * entry_price * self.config.day_margin_requirement

        # Update state
        self.current_position = -size
        self.entry_price = entry_price
        self.entry_date = bar['datetime']
        self.allocated_capital = margin_required
        self.available_capital -= margin_required + commission

        # Create trade record
        trade = {
            'datetime': bar['datetime'],
            'action': 'SELL',
            'contracts': size,
            'price': entry_price,
            'commission': commission,
            'margin': margin_required,
            'signal_strength': signal.get('strength', 0.0),
            'confidence': signal.get('confidence', 0.0),
            'vpoc_trend': signal.get('vpoc_trend', 0.0),
        }

        self.trades.append(trade)
        self.total_trades += 1

        self.logger.info(f"Opened SHORT position: {size} contracts @ ${entry_price:.2f}")

        return trade

    def _close_position(self, exit_price: float, exit_time: datetime) -> Dict:
        """Close current position"""

        if self.current_position == 0:
            return None

        # Apply slippage
        if self.current_position > 0:
            exit_price = exit_price - self.config.slippage_per_contract
        else:
            exit_price = exit_price + self.config.slippage_per_contract

        # Calculate P&L
        price_change = exit_price - self.entry_price
        gross_pnl = self.current_position * price_change
        commission = self.config.commission_per_trade
        net_pnl = gross_pnl - commission

        # Update capital
        self.available_capital += self.allocated_capital + net_pnl
        self.allocated_capital = 0.0

        # Update realized P&L
        self.realized_pnl += net_pnl

        # Update consecutive loss tracking
        if net_pnl < 0:
            self.consecutive_losses += 1
            self.losing_trades += 1
        else:
            self.consecutive_losses = 0
            self.winning_trades += 1

        # Create trade record
        trade = {
            'datetime': exit_time,
            'action': 'CLOSE',
            'contracts': abs(self.current_position),
            'entry_price': self.entry_price,
            'exit_price': exit_price,
            'gross_pnl': gross_pnl,
            'commission': commission,
            'net_pnl': net_pnl,
            'entry_date': self.entry_date,
            'exit_date': exit_time,
            'hold_time': exit_time - self.entry_date,
        }

        # Update position state
        position_size = self.current_position
        self.current_position = 0
        self.entry_price = 0.0
        self.entry_date = None

        self.trades.append(trade)
        self.total_trades += 1

        trade_type = "LONG" if position_size > 0 else "SHORT"
        self.logger.info(f"Closed {trade_type} position: {abs(position_size)} contracts, "
                        f"P&L: ${net_pnl:.2f}")

        return trade

    def update_unrealized_pnl(self, current_price: float):
        """Update unrealized P&L for open position"""

        if self.current_position == 0:
            self.unrealized_pnl = 0.0
        else:
            price_change = current_price - self.entry_price
            self.unrealized_pnl = self.current_position * price_change

    def update_equity_curve(self, bar: pd.Series):
        """Update equity curve and performance metrics"""

        total_capital = self.available_capital + self.allocated_capital + self.realized_pnl + self.unrealized_pnl
        self.current_capital = total_capital

        # Update peak capital and drawdown
        if total_capital > self.peak_capital:
            self.peak_capital = total_capital
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.peak_capital - total_capital) / self.peak_capital
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)

        # Record equity point
        equity_point = {
            'datetime': bar['datetime'],
            'close': bar['close'],
            'total_capital': total_capital,
            'available_capital': self.available_capital,
            'allocated_capital': self.allocated_capital,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'position': self.current_position,
            'drawdown': self.current_drawdown,
        }

        self.equity_curve.append(equity_point)

    def run_backtest(self, data: pd.DataFrame) -> Dict:
        """Run complete backtest on provided data"""

        self.logger.info(f"Starting backtest with {len(data)} bars")
        self.logger.info(f"Initial capital: ${self.config.initial_capital:,.2f}")

        # Process each bar
        for i, (idx, bar) in enumerate(data.iterrows()):
            try:
                # Generate signals
                signals = self.signal_generator.generate_signals(
                    data.iloc[:i+1],  # All data up to current bar
                    bar,
                    use_ml=self.config.use_ml_filter
                )

                # Execute trades based on signals
                if signals:
                    for signal in signals:
                        trade = self.execute_trade(signal, bar)

                # Update unrealized P&L
                self.update_unrealized_pnl(bar['close'])

                # Update equity curve
                self.update_equity_curve(bar)

                # Log progress
                if i % 1000 == 0:
                    self.logger.info(f"Processed {i}/{len(data)} bars, "
                                   f"Capital: ${self.current_capital:,.2f}, "
                                   f"Position: {self.current_position}")

                # Check for maximum drawdown
                if self.current_drawdown > self.config.max_drawdown_limit:
                    self.logger.warning(f"Maximum drawdown limit exceeded: {self.current_drawdown:.2%}")
                    break

            except Exception as e:
                self.logger.error(f"Error processing bar {i}: {e}")
                continue

        # Close any open position at the end
        if self.current_position != 0:
            last_bar = data.iloc[-1]
            self._close_position(last_bar['close'], last_bar['datetime'])

        # Calculate final metrics
        results = self.calculate_performance_metrics()

        self.logger.info("Backtest completed successfully")

        return results

    def calculate_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""

        if not self.equity_curve:
            return {}

        # Basic metrics
        total_return = (self.current_capital - self.config.initial_capital) / self.config.initial_capital

        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0

        # Calculate average trade metrics
        if self.trades:
            trade_pnls = [t['net_pnl'] for t in self.trades if 'net_pnl' in t]
            avg_trade = np.mean(trade_pnls) if trade_pnls else 0
            avg_winning_trade = np.mean([p for p in trade_pnls if p > 0]) if any(p > 0 for p in trade_pnls) else 0
            avg_losing_trade = np.mean([p for p in trade_pnls if p < 0]) if any(p < 0 for p in trade_pnls) else 0
        else:
            avg_trade = avg_winning_trade = avg_losing_trade = 0

        # Calculate daily returns for Sharpe ratio
        if len(self.equity_curve) > 1:
            equity_df = pd.DataFrame(self.equity_curve)
            equity_df['datetime'] = pd.to_datetime(equity_df['datetime'])
            equity_df.set_index('datetime', inplace=True)

            # Resample to daily
            daily_equity = equity_df['total_capital'].resample('D').last()
            daily_returns = daily_equity.pct_change().dropna()

            if len(daily_returns) > 1:
                sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0

        # Calculate profit factor
        if self.trades:
            trade_pnls = [t['net_pnl'] for t in self.trades if 'net_pnl' in t]
            gross_profits = sum(p for p in trade_pnls if p > 0)
            gross_losses = abs(sum(p for p in trade_pnls if p < 0))
            profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')
        else:
            profit_factor = 0

        metrics = {
            # Capital metrics
            'initial_capital': self.config.initial_capital,
            'final_capital': self.current_capital,
            'total_return': total_return,
            'realized_pnl': self.realized_pnl,
            'max_drawdown': self.max_drawdown,

            # Trade metrics
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'avg_trade': avg_trade,
            'avg_winning_trade': avg_winning_trade,
            'avg_losing_trade': avg_losing_trade,
            'profit_factor': profit_factor,

            # Risk metrics
            'sharpe_ratio': sharpe_ratio,
            'max_consecutive_losses': max(getattr(self, 'consecutive_losses', 0), 0),

            # Performance summary
            'total_commission': sum(t.get('commission', 0) for t in self.trades),
            'total_slippage': sum(abs(t.get('slippage', 0)) for t in self.trades),
        }

        return metrics
```

## Advanced Backtesting Features

### 1. Monte Carlo Simulation

```python
# monte_carlo.py
import numpy as np
import pandas as pd
from typing import List, Dict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

class MonteCarloSimulator:
    def __init__(self, base_config: BacktestConfig, num_simulations: int = 1000):
        self.base_config = base_config
        self.num_simulations = num_simulations

    def create_variated_config(self, variation_factor: float = 0.1) -> BacktestConfig:
        """Create a configuration with randomized parameters"""

        # Randomize key parameters within variation_factor
        config_dict = self.base_config.__dict__.copy()

        # Vary risk parameters
        config_dict['risk_per_trade'] = np.clip(
            np.random.normal(self.base_config.risk_per_trade,
                           self.base_config.risk_per_trade * variation_factor),
            0.005, 0.02
        )

        # Vary commission
        config_dict['commission_per_trade'] = np.clip(
            np.random.normal(self.base_config.commission_per_trade,
                           self.base_config.commission_per_trade * variation_factor),
            5, 20
        )

        # Vary slippage
        config_dict['slippage_per_contract'] = np.clip(
            np.random.normal(self.base_config.slippage_per_contract,
                           self.base_config.slippage_per_contract * variation_factor),
            0.1, 0.5
        )

        return BacktestConfig(**config_dict)

    def run_single_simulation(self, data: pd.DataFrame, sim_id: int) -> Dict:
        """Run a single backtest simulation"""

        # Create varied configuration
        config = self.create_variated_config()

        # Initialize backtesting engine
        sim_config = SimulationConfig(
            save_trades=False,  # Don't save individual trades for MC
            save_equity_curve=True,
            generate_reports=False,
        )

        engine = BacktestingEngine(config, sim_config)

        # Run backtest
        results = engine.run_backtest(data)

        # Add simulation ID
        results['simulation_id'] = sim_id
        results['config'] = config

        return results

    def run_monte_carlo(self, data: pd.DataFrame) -> List[Dict]:
        """Run Monte Carlo simulation"""

        print(f"Running {self.num_simulations} Monte Carlo simulations...")

        # Use parallel processing for speed
        with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
            futures = [
                executor.submit(self.run_single_simulation, data, i)
                for i in range(self.num_simulations)
            ]

            results = []
            for i, future in enumerate(futures):
                try:
                    result = future.result()
                    results.append(result)

                    if (i + 1) % 100 == 0:
                        print(f"Completed {i + 1}/{self.num_simulations} simulations")

                except Exception as e:
                    print(f"Simulation {i} failed: {e}")

        print(f"Monte Carlo simulation completed. {len(results)} successful runs.")

        return results

    def analyze_results(self, results: List[Dict]) -> Dict:
        """Analyze Monte Carlo simulation results"""

        if not results:
            return {}

        # Extract key metrics
        returns = [r['total_return'] for r in results]
        sharpe_ratios = [r['sharpe_ratio'] for r in results]
        max_drawdowns = [r['max_drawdown'] for r in results]
        win_rates = [r['win_rate'] for r in results]

        analysis = {
            'num_simulations': len(results),
            'total_return': {
                'mean': np.mean(returns),
                'std': np.std(returns),
                'min': np.min(returns),
                'max': np.max(returns),
                'percentile_5': np.percentile(returns, 5),
                'percentile_95': np.percentile(returns, 95),
            },
            'sharpe_ratio': {
                'mean': np.mean(sharpe_ratios),
                'std': np.std(sharpe_ratios),
                'min': np.min(sharpe_ratios),
                'max': np.max(sharpe_ratios),
            },
            'max_drawdown': {
                'mean': np.mean(max_drawdowns),
                'std': np.std(max_drawdowns),
                'min': np.min(max_drawdowns),
                'max': np.max(max_drawdowns),
            },
            'win_rate': {
                'mean': np.mean(win_rates),
                'std': np.std(win_rates),
                'min': np.min(win_rates),
                'max': np.max(win_rates),
            },
        }

        # Calculate probability of positive return
        positive_returns = [r for r in returns if r > 0]
        analysis['probability_positive_return'] = len(positive_returns) / len(returns)

        # Calculate probability of meeting target return (e.g., 10% annual)
        target_return = 0.10
        target_returns = [r for r in returns if r >= target_return]
        analysis['probability_target_return'] = len(target_returns) / len(returns)

        return analysis
```

### 2. Walk-Forward Analysis

```python
# walk_forward.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Dict

class WalkForwardAnalyzer:
    def __init__(self, base_config: BacktestConfig):
        self.base_config = base_config

    def create_walk_forward_periods(self, data: pd.DataFrame,
                                 train_period_months: int = 12,
                                 test_period_months: int = 3,
                                 step_months: int = 3) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Create walk-forward analysis periods"""

        periods = []
        data['datetime'] = pd.to_datetime(data['date'] + ' ' + data['time'])
        data = data.sort_values('datetime')

        start_date = data['datetime'].min()
        end_date = data['datetime'].max()

        current_start = start_date
        period_id = 0

        while True:
            # Calculate training period
            train_end = current_start + timedelta(days=train_period_months * 30)

            # Calculate test period
            test_start = train_end
            test_end = test_start + timedelta(days=test_period_months * 30)

            # Check if we have enough data
            if test_end > end_date:
                break

            # Extract periods
            train_data = data[(data['datetime'] >= current_start) & (data['datetime'] < train_end)]
            test_data = data[(data['datetime'] >= test_start) & (data['datetime'] < test_end)]

            if len(train_data) > 1000 and len(test_data) > 100:  # Minimum data requirements
                periods.append((train_data, test_data, period_id))
                print(f"Period {period_id}: Train {len(train_data)} bars, Test {len(test_data)} bars")

            # Move to next period
            current_start = current_start + timedelta(days=step_months * 30)
            period_id += 1

        return periods

    def optimize_parameters(self, train_data: pd.DataFrame) -> Dict:
        """Optimize strategy parameters on training data"""

        # Define parameter grid
        param_grid = {
            'risk_per_trade': [0.005, 0.01, 0.015, 0.02],
            'confidence_threshold': [50, 60, 70, 80],
            'vpoc_sensitivity': [0.01, 0.02, 0.03],
        }

        best_params = None
        best_score = -np.inf

        # Grid search (simplified - could use more sophisticated optimization)
        for risk in param_grid['risk_per_trade']:
            for confidence in param_grid['confidence_threshold']:
                for sensitivity in param_grid['vpoc_sensitivity']:

                    # Create config with test parameters
                    config = BacktestConfig(**self.base_config.__dict__)
                    config.risk_per_trade = risk
                    config.confidence_threshold = confidence
                    config.vpoc_sensitivity = sensitivity

                    # Run backtest on training data
                    sim_config = SimulationConfig(save_trades=False, save_equity_curve=False)
                    engine = BacktestingEngine(config, sim_config)
                    results = engine.run_backtest(train_data)

                    # Calculate score (could be composite of multiple metrics)
                    score = results['sharpe_ratio'] * (1 - results['max_drawdown'])

                    if score > best_score:
                        best_score = score
                        best_params = {
                            'risk_per_trade': risk,
                            'confidence_threshold': confidence,
                            'vpoc_sensitivity': sensitivity,
                            'training_score': score,
                            'training_results': results,
                        }

        return best_params

    def run_walk_forward_analysis(self, data: pd.DataFrame) -> Dict:
        """Run complete walk-forward analysis"""

        print("Starting walk-forward analysis...")

        # Create periods
        periods = self.create_walk_forward_periods(data)

        if not periods:
            print("No valid periods found for walk-forward analysis")
            return {}

        # Store results
        all_results = []
        optimized_params = []

        for i, (train_data, test_data, period_id) in enumerate(periods):
            print(f"\nProcessing period {period_id + 1}/{len(periods)}")

            # Optimize parameters on training data
            best_params = self.optimize_parameters(train_data)
            optimized_params.append(best_params)

            # Apply optimized parameters to test data
            config = BacktestConfig(**self.base_config.__dict__)
            config.risk_per_trade = best_params['risk_per_trade']
            config.confidence_threshold = best_params['confidence_threshold']
            config.vpoc_sensitivity = best_params['vpoc_sensitivity']

            # Run backtest on test data
            sim_config = SimulationConfig()
            engine = BacktestingEngine(config, sim_config)
            test_results = engine.run_backtest(test_data)

            # Store results
            period_results = {
                'period_id': period_id,
                'optimized_params': best_params,
                'test_results': test_results,
                'train_performance': best_params['training_results'],
            }

            all_results.append(period_results)

            print(f"Period {period_id}: Train Sharpe: {best_params['training_results']['sharpe_ratio']:.2f}, "
                  f"Test Sharpe: {test_results['sharpe_ratio']:.2f}, "
                  f"Test Return: {test_results['total_return']:.2%}")

        # Analyze overall performance
        analysis = self.analyze_walk_forward_results(all_results)

        return {
            'periods': all_results,
            'analysis': analysis,
            'optimized_parameters': optimized_params,
        }

    def analyze_walk_forward_results(self, results: List[Dict]) -> Dict:
        """Analyze walk-forward analysis results"""

        if not results:
            return {}

        # Extract test results
        test_returns = [r['test_results']['total_return'] for r in results]
        test_sharpes = [r['test_results']['sharpe_ratio'] for r in results]
        test_drawdowns = [r['test_results']['max_drawdown'] for r in results]
        test_win_rates = [r['test_results']['win_rate'] for r in results]

        # Extract training results for comparison
        train_returns = [r['train_performance']['total_return'] for r in results]
        train_sharpes = [r['train_performance']['sharpe_ratio'] for r in results]

        analysis = {
            'num_periods': len(results),
            'out_of_sample_performance': {
                'avg_return': np.mean(test_returns),
                'avg_sharpe': np.mean(test_sharpes),
                'avg_drawdown': np.mean(test_drawdowns),
                'avg_win_rate': np.mean(test_win_rates),
                'return_std': np.std(test_returns),
                'sharpe_std': np.std(test_sharpes),
            },
            'in_sample_performance': {
                'avg_return': np.mean(train_returns),
                'avg_sharpe': np.mean(train_sharpes),
                'return_std': np.std(train_returns),
                'sharpe_std': np.std(train_sharpes),
            },
            'stability_analysis': {
                'positive_periods': sum(1 for r in test_returns if r > 0),
                'positive_period_ratio': sum(1 for r in test_returns if r > 0) / len(test_returns),
                'worst_period_return': np.min(test_returns),
                'best_period_return': np.max(test_returns),
                'return_range': np.max(test_returns) - np.min(test_returns),
            },
            'parameter_stability': {
                'risk_per_trade_std': np.std([r['optimized_params']['risk_per_trade'] for r in results]),
                'confidence_std': np.std([r['optimized_params']['confidence_threshold'] for r in results]),
                'sensitivity_std': np.std([r['optimized_params']['vpoc_sensitivity'] for r in results]),
            }
        }

        # Calculate stability metrics
        analysis['out_of_sample_stability'] = 1 - (analysis['out_of_sample_performance']['return_std'] /
                                                   abs(analysis['out_of_sample_performance']['avg_return'])
                                                   if analysis['out_of_sample_performance']['avg_return'] != 0 else 0)

        return analysis
```

## Backtesting Execution and Results

### 1. Main Backtesting Script

```python
# run_backtest.py
import pandas as pd
import numpy as np
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns

from src.config.settings import settings
from documentation.backtest_config import BacktestConfig, SimulationConfig
from documentation.backtesting_engine import BacktestingEngine
from documentation.monte_carlo import MonteCarloSimulator
from documentation.walk_forward import WalkForwardAnalyzer

def main():
    """Main backtesting execution"""

    print("=" * 60)
    print("ES Futures VPOC Strategy - Backtesting Engine")
    print("=" * 60)

    # Load data
    data_path = "DATA/ES/5min/ES_5min_cleaned.csv"
    print(f"Loading data from: {data_path}")

    data = pd.read_csv(data_path)
    print(f"Loaded {len(data)} records")

    # Configure backtest
    config = BacktestConfig(
        initial_capital=100000,
        risk_per_trade=0.01,
        commission_per_trade=10,
        slippage_per_contract=0.25,
        use_ml_filter=True,
        confidence_threshold=60.0,
        max_drawdown_limit=0.20,
    )

    sim_config = SimulationConfig(
        data_path=data_path,
        save_trades=True,
        save_equity_curve=True,
        generate_reports=True,
        output_dir="BACKTEST_RESULTS/"
    )

    # Run primary backtest
    print("\n1. Running Primary Backtest")
    print("-" * 40)

    engine = BacktestingEngine(config, sim_config)
    primary_results = engine.run_backtest(data)

    # Display primary results
    print("\nPrimary Backtest Results:")
    print(f"Total Return: {primary_results['total_return']:.2%}")
    print(f"Sharpe Ratio: {primary_results['sharpe_ratio']:.2f}")
    print(f"Win Rate: {primary_results['win_rate']:.2%}")
    print(f"Profit Factor: {primary_results['profit_factor']:.2f}")
    print(f"Max Drawdown: {primary_results['max_drawdown']:.2%}")
    print(f"Total Trades: {primary_results['total_trades']}")

    # Save primary results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    primary_output = f"BACKTEST_RESULTS/primary_backtest_{timestamp}.json"

    with open(primary_output, 'w') as f:
        json.dump(primary_results, f, indent=2, default=str)

    print(f"\nPrimary results saved to: {primary_output}")

    # Save equity curve
    if engine.equity_curve:
        equity_df = pd.DataFrame(engine.equity_curve)
        equity_output = f"BACKTEST_RESULTS/equity_curve_{timestamp}.csv"
        equity_df.to_csv(equity_output, index=False)
        print(f"Equity curve saved to: {equity_output}")

    # Save trades
    if engine.trades:
        trades_df = pd.DataFrame(engine.trades)
        trades_output = f"BACKTEST_RESULTS/trades_{timestamp}.csv"
        trades_df.to_csv(trades_output, index=False)
        print(f"Trades saved to: {trades_output}")

    # Run Monte Carlo simulation
    print("\n2. Running Monte Carlo Simulation (1000 runs)")
    print("-" * 40)

    mc_simulator = MonteCarloSimulator(config, num_simulations=1000)
    mc_results = mc_simulator.run_monte_carlo(data)
    mc_analysis = mc_simulator.analyze_results(mc_results)

    print("\nMonte Carlo Results:")
    print(f"Average Return: {mc_analysis['total_return']['mean']:.2%}")
    print(f"Return Std Dev: {mc_analysis['total_return']['std']:.2%}")
    print(f"5th Percentile Return: {mc_analysis['total_return']['percentile_5']:.2%}")
    print(f"95th Percentile Return: {mc_analysis['total_return']['percentile_95']:.2%}")
    print(f"Probability Positive Return: {mc_analysis['probability_positive_return']:.2%}")

    # Save Monte Carlo results
    mc_output = f"BACKTEST_RESULTS/monte_carlo_{timestamp}.json"
    with open(mc_output, 'w') as f:
        json.dump(mc_analysis, f, indent=2, default=str)

    print(f"Monte Carlo results saved to: {mc_output}")

    # Run walk-forward analysis
    print("\n3. Running Walk-Forward Analysis")
    print("-" * 40)

    wf_analyzer = WalkForwardAnalyzer(config)
    wf_results = wf_analyzer.run_walk_forward_analysis(data)

    if wf_results:
        print("\nWalk-Forward Results:")
        print(f"Out-of-Sample Avg Return: {wf_results['analysis']['out_of_sample_performance']['avg_return']:.2%}")
        print(f"Out-of-Sample Avg Sharpe: {wf_results['analysis']['out_of_sample_performance']['avg_sharpe']:.2f}")
        print(f"Positive Periods: {wf_results['analysis']['stability_analysis']['positive_period_ratio']:.2%}")
        print(f"Strategy Stability: {wf_results['analysis']['out_of_sample_stability']:.2%}")

        # Save walk-forward results
        wf_output = f"BACKTEST_RESULTS/walk_forward_{timestamp}.json"
        with open(wf_output, 'w') as f:
            json.dump(wf_results['analysis'], f, indent=2, default=str)

        print(f"Walk-forward results saved to: {wf_output}")

    # Generate comprehensive report
    print("\n4. Generating Comprehensive Report")
    print("-" * 40)

    generate_comprehensive_report(primary_results, mc_analysis, wf_results, timestamp)

    print("\n" + "=" * 60)
    print("Backtesting completed successfully!")
    print(f"All results saved to: BACKTEST_RESULTS/")
    print("=" * 60)

def generate_comprehensive_report(primary_results, mc_analysis, wf_results, timestamp):
    """Generate comprehensive HTML report"""

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ES Futures VPOC Strategy - Backtesting Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ text-align: center; color: #333; }}
            .section {{ margin: 30px 0; }}
            .metric {{ display: inline-block; margin: 10px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
            .metric-label {{ font-weight: bold; color: #666; }}
            .metric-value {{ font-size: 1.2em; color: #333; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>ES Futures VPOC Strategy - Backtesting Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <div class="section">
            <h2>Primary Backtest Results</h2>
            <div class="metric">
                <div class="metric-label">Total Return</div>
                <div class="metric-value">{primary_results['total_return']:.2%}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Sharpe Ratio</div>
                <div class="metric-value">{primary_results['sharpe_ratio']:.2f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Win Rate</div>
                <div class="metric-value">{primary_results['win_rate']:.2%}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Profit Factor</div>
                <div class="metric-value">{primary_results['profit_factor']:.2f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Max Drawdown</div>
                <div class="metric-value">{primary_results['max_drawdown']:.2%}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Total Trades</div>
                <div class="metric-value">{primary_results['total_trades']}</div>
            </div>
        </div>

        <div class="section">
            <h2>Monte Carlo Simulation (1000 runs)</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Average Return</td>
                    <td>{mc_analysis['total_return']['mean']:.2%}</td>
                </tr>
                <tr>
                    <td>Return Standard Deviation</td>
                    <td>{mc_analysis['total_return']['std']:.2%}</td>
                </tr>
                <tr>
                    <td>5th Percentile Return</td>
                    <td>{mc_analysis['total_return']['percentile_5']:.2%}</td>
                </tr>
                <tr>
                    <td>95th Percentile Return</td>
                    <td>{mc_analysis['total_return']['percentile_95']:.2%}</td>
                </tr>
                <tr>
                    <td>Probability of Positive Return</td>
                    <td>{mc_analysis['probability_positive_return']:.2%}</td>
                </tr>
            </table>
        </div>
    """

    if wf_results:
        html_content += f"""
        <div class="section">
            <h2>Walk-Forward Analysis</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>In-Sample</th>
                    <th>Out-of-Sample</th>
                </tr>
                <tr>
                    <td>Average Return</td>
                    <td>{wf_results['analysis']['in_sample_performance']['avg_return']:.2%}</td>
                    <td>{wf_results['analysis']['out_of_sample_performance']['avg_return']:.2%}</td>
                </tr>
                <tr>
                    <td>Average Sharpe Ratio</td>
                    <td>{wf_results['analysis']['in_sample_performance']['avg_sharpe']:.2f}</td>
                    <td>{wf_results['analysis']['out_of_sample_performance']['avg_sharpe']:.2f}</td>
                </tr>
                <tr>
                    <td>Positive Periods</td>
                    <td>-</td>
                    <td>{wf_results['analysis']['stability_analysis']['positive_period_ratio']:.2%}</td>
                </tr>
                <tr>
                    <td>Strategy Stability</td>
                    <td>-</td>
                    <td>{wf_results['analysis']['out_of_sample_stability']:.2%}</td>
                </tr>
            </table>
        </div>
        """

    html_content += """
    </body>
    </html>
    """

    report_path = f"BACKTEST_RESULTS/comprehensive_report_{timestamp}.html"
    with open(report_path, 'w') as f:
        f.write(html_content)

    print(f"Comprehensive report saved to: {report_path}")

if __name__ == "__main__":
    main()
```

This comprehensive backtesting guide provides all the necessary tools and methodologies to thoroughly test the ES Futures VPOC strategy, including advanced techniques like Monte Carlo simulation and walk-forward analysis to ensure robustness and reliability of the trading strategy.