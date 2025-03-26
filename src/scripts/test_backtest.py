#!/usr/bin/env python
"""
Test script for the backtesting engine.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.core.data import FuturesDataManager
from src.core.signals import SignalGenerator
from src.analysis.backtest import BacktestEngine

def main():
    """Test backtesting functionality"""
    print("Testing backtesting engine...")
    
    # Create synthetic signals for testing
    print("Creating synthetic trading signals...")
    
    # Date range for synthetic signals
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 2, 28)
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    
    # Create synthetic signals
    signals = []
    price = 5000.0  # Starting price
    
    for i, date in enumerate(date_range):
        # Alternate between BUY and SELL signals
        signal_type = 'BUY' if i % 2 == 0 else 'SELL'
        
        # Create random price movements
        price_change = np.random.normal(0, 50)
        price += price_change
        
        # Create signal
        signal = {
            'date': date,
            'signal': signal_type,
            'price': price,
            'stop_loss': price * (0.98 if signal_type == 'BUY' else 1.02),  # 2% stop loss
            'target': price * (1.04 if signal_type == 'BUY' else 0.96),  # 4% target
            'position_size': 1.0,
            'confidence': 70,
            'reason': "Synthetic Signal"
        }
        
        signals.append(signal)
    
    # Create DataFrame
    signals_df = pd.DataFrame(signals)
    print(f"Created {len(signals_df)} synthetic signals")
    
    # Initialize backtest engine
    backtest = BacktestEngine(
        initial_capital=100000,
        commission=10,
        slippage=0.25,
        risk_per_trade=0.01
    )
    
    # Run backtest
    print("\nRunning backtest...")
    performance = backtest.run_backtest(signals_df)
    
    # Print performance metrics
    print("\nBacktest Performance Metrics:")
    for metric, value in performance.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.2f}")
        else:
            print(f"{metric}: {value}")
    
    # Generate and print report
    print("\nGenerating performance report...")
    report = backtest.generate_report()
    print(report)
    
    # Plot performance
    print("\nGenerating performance plot...")
    output_path = os.path.join(os.getcwd(), "backtest_performance.png")
    backtest.plot_performance(save_path=output_path)
    print(f"Performance plot saved to {output_path}")
    
    # Save trades
    trades_path = os.path.join(os.getcwd(), "backtest_trades.csv")
    backtest.save_trades(output_path=trades_path)
    print(f"Trade details saved to {trades_path}")
    
    print("\nTest completed.")

if __name__ == "__main__":
    main()