"""
Backtesting engine for futures VPOC trading strategy.
Handles trade execution, position sizing, and performance metrics.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

from src.utils.logging import get_logger


class BacktestEngine:
    """
    Engine for backtesting trading strategies with realistic trade simulation.
    """
    
    def __init__(self, 
                initial_capital=100000, 
                commission=10, 
                slippage=0.25,
                risk_per_trade=0.01,
                settings=None):
        """
        Initialize the backtest engine with parameters.
        
        Args:
            initial_capital: Starting capital
            commission: Commission per trade
            slippage: Slippage in price points
            risk_per_trade: Percentage of capital to risk per trade
            settings: Configuration settings
        """
        self.logger = get_logger(__name__)
        
        # Get settings if provided
        if settings is None:
            try:
                from src.config.settings import settings
            except ImportError:
                settings = None
        
        # Set backtest parameters from settings or defaults
        self.initial_capital = getattr(settings, 'INITIAL_CAPITAL', initial_capital)
        self.commission = getattr(settings, 'COMMISSION_PER_TRADE', commission)
        self.slippage = getattr(settings, 'SLIPPAGE', slippage)
        self.risk_per_trade = getattr(settings, 'RISK_PER_TRADE', risk_per_trade)
        self.risk_free_rate = getattr(settings, 'RISK_FREE_RATE', 0.02)
        self.margin_requirement = getattr(settings, 'MARGIN_REQUIREMENT', 0.1)
        self.overnight_margin = getattr(settings, 'OVERNIGHT_MARGIN', 0.15)
        self.max_position_size = getattr(settings, 'MAX_POSITION_SIZE', 10)
        self.min_capital_buffer = getattr(settings, 'MIN_CAPITAL_BUFFER', 0.2)
        
        # Set output directory
        self.output_dir = getattr(settings, 'BACKTEST_DIR', 
                                os.path.join(os.getcwd(), 'BACKTEST'))
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize results storage
        self.trades = pd.DataFrame()
        self.equity_curve = []
        self.performance_metrics = {}
        
        self.logger.info(f"Initialized BacktestEngine with capital={self.initial_capital}")
    
    def run_backtest(self, signals_df):
        """
        Execute backtest with realistic trade simulation.
        
        Args:
            signals_df: DataFrame with trading signals
            
        Returns:
            Dict with performance metrics
        """
        self.logger.info(f"Running backtest on {len(signals_df)} signals")
        
        # Validate input data
        if signals_df.empty:
            self.logger.warning("No signals provided for backtest")
            return {}
            
        required_columns = ['date', 'signal', 'price', 'stop_loss', 'target']
        missing_columns = [col for col in required_columns if col not in signals_df.columns]
        
        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}")
            return {}
        
        # Sort signals by date
        signals_df = signals_df.sort_values('date')
        
        # Initialize tracking variables
        capital = self.initial_capital
        trade_log = []
        equity_curve = [capital]
        current_position = None
        
        # Process each signal
        for _, signal in signals_df.iterrows():
            date = signal['date']
            signal_type = signal['signal']
            entry_price = signal['price']
            stop_loss = signal['stop_loss']
            target = signal['target']
            
            # Close existing position if needed
            if current_position:
                if (current_position['type'] == 'LONG' and signal_type == 'SELL') or \
                   (current_position['type'] == 'SHORT' and signal_type == 'BUY'):
                    # Simulate realistic exit price
                    exit_price = self._simulate_exit_price(
                        entry_price,
                        self.slippage,
                        signal_type
                    )
                    
                    # Calculate profit
                    profit = self._calculate_trade_profit(current_position, exit_price)

                    # Ensure profit is a scalar to avoid pandas reindexing issues
                    if hasattr(profit, 'iloc'):
                        profit = profit.iloc[0] if len(profit) > 0 else profit
                    elif hasattr(profit, 'item'):
                        profit = profit.item()

                    profit = float(profit)  # Convert to regular Python float
                    capital += profit
                    
                    # Log trade
                    trade_log.append({
                        'entry_date': current_position['entry_date'],
                        'exit_date': date,
                        'entry_price': current_position['entry_price'],
                        'exit_price': exit_price,
                        'type': current_position['type'],
                        'profit': profit,
                        'position_size': current_position['position_size'],
                        'capital_after': capital
                    })
                    
                    # Update equity curve
                    equity_curve.append(capital)
                    
                    # Reset position
                    current_position = None
            
            # Open new position if none exists
            if not current_position:
                # Calculate position size
                position_size = self._calculate_position_size(entry_price, stop_loss, capital)

                # Validate position size (handle pandas Series)
                if hasattr(position_size, 'iloc'):  # It's a pandas Series
                    position_size = position_size.clip(upper=self.max_position_size)
                    # Extract scalar value if it's a Series with single element
                    if len(position_size) == 1:
                        position_size = position_size.iloc[0]
                    else:
                        # If multiple values, take the first one (shouldn't happen in normal operation)
                        position_size = position_size.iloc[0]
                else:
                    position_size = min(position_size, self.max_position_size)

                # Ensure position_size is a scalar integer
                if hasattr(position_size, 'item'):
                    position_size = position_size.item()
                position_size = int(position_size)

                # Check margin requirements
                if self._check_margin_requirements(position_size, entry_price, capital):
                    current_position = {
                        'type': 'LONG' if signal_type == 'BUY' else 'SHORT',
                        'entry_price': entry_price,
                        'entry_date': date,
                        'stop_loss': stop_loss,
                        'target': target,
                        'position_size': position_size
                    }
                    
                    self.logger.debug(f"Opened {current_position['type']} position with {position_size} contracts at {entry_price}")
        
        # Convert trade log to DataFrame
        self.trades = pd.DataFrame(trade_log)
        self.equity_curve = equity_curve
        
        # Calculate performance metrics if trades exist
        if not self.trades.empty:
            self._calculate_metrics()
            
        return self.performance_metrics
    
    def _calculate_position_size(self, entry_price, stop_loss, capital):
        """
        Calculate optimal position size using fixed fractional risk management.
        
        Args:
            entry_price: Entry price of the trade
            stop_loss: Stop loss price
            capital: Available trading capital
            
        Returns:
            Position size (number of contracts)
        """
        # Risk per trade based on configuration
        risk_amount = capital * self.risk_per_trade
        
        # Calculate risk per contract
        risk_per_contract = abs(entry_price - stop_loss)

        # Handle pandas Series for risk comparison
        if hasattr(risk_per_contract, 'iloc'):  # It's a pandas Series
            if (risk_per_contract <= 0).any():
                self.logger.warning("Invalid risk per contract (entry price = stop loss)")
                return 1
        else:
            if risk_per_contract <= 0:
                self.logger.warning("Invalid risk per contract (entry price = stop loss)")
                return 1
        
        # Calculate position size
        if hasattr(risk_per_contract, 'iloc'):  # It's a pandas Series
            # Handle division by zero and invalid values
            denominator = risk_per_contract + self.slippage + self.commission
            position_size = risk_amount / denominator

            # Replace invalid values (inf, -inf, nan) with 1
            position_size = position_size.fillna(1).replace([float('inf'), float('-inf')], 1)
            position_size = position_size.astype(int)

            # Ensure minimum 1 contract
            position_size = position_size.clip(lower=1)

            # If it's a Series with one element, extract scalar
            if len(position_size) == 1:
                return position_size.iloc[0]
            else:
                # Multiple values shouldn't happen in normal operation, but handle it
                return position_size.iloc[0]
        else:
            # Handle scalar calculation
            denominator = risk_per_contract + self.slippage + self.commission
            if denominator == 0:
                position_size = 1
            else:
                position_size = risk_amount / denominator

                # Handle invalid values
                if not np.isfinite(position_size) or position_size <= 0:
                    position_size = 1

            position_size = int(position_size)
            return max(1, position_size)  # Ensure at least 1 contract
    
    def _simulate_exit_price(self, base_price, slippage, signal_type):
        """
        Simulate realistic exit price with random slippage.
        
        Args:
            base_price: Base price for exit
            slippage: Slippage amount
            signal_type: Signal type (BUY/SELL)
            
        Returns:
            Simulated exit price
        """
        slippage_impact = np.random.uniform(-slippage, slippage * 2)
        return base_price + (slippage_impact if signal_type == 'BUY' else -slippage_impact)
    
    def _check_margin_requirements(self, position_size, price, capital):
        """
        Validate margin requirements for position.

        Args:
            position_size: Number of contracts
            price: Contract price
            capital: Available capital

        Returns:
            Boolean indicating if margin requirements are met
        """
        # Convert all inputs to scalars to handle pandas Series
        if hasattr(position_size, 'iloc'):
            position_size = position_size.iloc[0] if len(position_size) > 0 else position_size
        if hasattr(price, 'iloc'):
            price = price.iloc[0] if len(price) > 0 else price
        if hasattr(capital, 'iloc'):
            capital = capital.iloc[0] if len(capital) > 0 else capital

        # Handle numpy arrays
        import numpy as np
        if isinstance(position_size, np.ndarray):
            position_size = float(position_size[0] if position_size.size > 0 else position_size)
        if isinstance(price, np.ndarray):
            price = float(price[0] if price.size > 0 else price)
        if isinstance(capital, np.ndarray):
            capital = float(capital[0] if capital.size > 0 else capital)

        # Ensure all are scalars
        position_size = float(position_size)
        price = float(price)
        capital = float(capital)

        margin_required = position_size * price * self.margin_requirement
        buffer_capital = capital * (1 - self.min_capital_buffer)

        if margin_required > buffer_capital:
            self.logger.warning(f"Margin requirement ({margin_required:.2f}) exceeds available capital with buffer ({buffer_capital:.2f})")
            return False

        return True
    
    def _calculate_trade_profit(self, position, exit_price):
        """
        Calculate realistic profit including all costs.

        Args:
            position: Position dictionary
            exit_price: Exit price

        Returns:
            Net profit
        """
        # Extract scalar values from position to avoid pandas Series issues
        entry_price = position['entry_price']
        position_size = position['position_size']

        # Ensure exit_price is scalar
        if hasattr(exit_price, 'iloc'):
            exit_price = exit_price.iloc[0] if len(exit_price) > 0 else exit_price
        elif hasattr(exit_price, 'item'):
            exit_price = exit_price.item()

        # Ensure position values are scalars
        if hasattr(entry_price, 'iloc'):
            entry_price = entry_price.iloc[0] if len(entry_price) > 0 else entry_price
        elif hasattr(entry_price, 'item'):
            entry_price = entry_price.item()

        if hasattr(position_size, 'iloc'):
            position_size = position_size.iloc[0] if len(position_size) > 0 else position_size
        elif hasattr(position_size, 'item'):
            position_size = position_size.item()

        # Convert to regular Python floats
        exit_price = float(exit_price)
        entry_price = float(entry_price)
        position_size = float(position_size)

        # Calculate gross profit
        if position['type'] == 'LONG':
            gross_profit = (exit_price - entry_price) * position_size
        else:  # SHORT
            gross_profit = (entry_price - exit_price) * position_size

        # Enhanced cost calculation
        total_commission = self.commission * 2  # Entry and exit
        exchange_fees = 2.50 * position_size * 2  # $2.50 per contract per side
        clearing_fees = 0.50 * position_size * 2  # $0.50 per contract per side

        # Calculate net profit
        net_profit = gross_profit - (total_commission + exchange_fees + clearing_fees)

        return net_profit
    
    def _calculate_metrics(self):
        """
        Calculate comprehensive trading performance metrics.
        """
        if self.trades.empty:
            self.logger.warning("No trades to analyze")
            return
        
        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = self.trades[self.trades['profit'] > 0]
        losing_trades = self.trades[self.trades['profit'] <= 0]
        
        # Performance calculations
        self.performance_metrics = {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / total_trades * 100 if total_trades > 0 else 0,
            'total_profit': self.trades['profit'].sum(),
            'average_profit_per_trade': self.trades['profit'].mean() if total_trades > 0 else 0,
            'max_profit': self.trades['profit'].max() if total_trades > 0 else 0,
            'max_loss': self.trades['profit'].min() if total_trades > 0 else 0,
            'profit_factor': abs(winning_trades['profit'].sum() / losing_trades['profit'].sum()) if len(losing_trades) > 0 and losing_trades['profit'].sum() != 0 else float('inf')
        }
        
        # Advanced metrics
        final_capital = self.equity_curve[-1] if self.equity_curve else self.initial_capital
        returns = self.trades['profit'] / self.initial_capital
        
        # Calculate drawdown
        peaks = pd.Series(self.equity_curve).cummax()
        drawdowns = (pd.Series(self.equity_curve) - peaks) / peaks * 100
        
        # Portfolio metrics
        self.performance_metrics.update({
            'total_return': (final_capital - self.initial_capital) / self.initial_capital * 100,
            'annualized_return': self._calculate_annualized_return(),
            'max_drawdown': drawdowns.min() if not drawdowns.empty else 0,
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'sortino_ratio': self._calculate_sortino_ratio(returns)
        })
        
        self.logger.info(f"Calculated performance metrics: win_rate={self.performance_metrics['win_rate']:.1f}%, profit=${self.performance_metrics['total_profit']:.2f}")
    
    def _calculate_annualized_return(self):
        """
        Calculate annualized return.
        """
        if not self.trades.empty and 'entry_date' in self.trades.columns and 'exit_date' in self.trades.columns:
            start_date = self.trades['entry_date'].min()
            end_date = self.trades['exit_date'].max()
            
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
                
            trading_days = (end_date - start_date).days
            
            if trading_days > 0:
                # Convert to years assuming 252 trading days per year
                years = trading_days / 252
                
                # Calculate final return
                final_capital = self.equity_curve[-1] if self.equity_curve else self.initial_capital
                total_return = (final_capital - self.initial_capital) / self.initial_capital
                
                # Annualize
                annualized_return = (1 + total_return) ** (1 / max(years, 0.01)) - 1
                return annualized_return * 100  # Convert to percentage
        
        return 0
    
    def _calculate_sharpe_ratio(self, returns):
        """
        Calculate Sharpe Ratio of the trading strategy.
        
        Args:
            returns: Series of returns
            
        Returns:
            Sharpe Ratio
        """
        if returns.empty or returns.std() == 0:
            return 0
            
        # Annualize returns and standard deviation
        avg_return = returns.mean() * 252  # Assuming 252 trading days per year
        std_return = returns.std() * np.sqrt(252)
        
        # Calculate Sharpe Ratio
        sharpe_ratio = (avg_return - self.risk_free_rate) / std_return if std_return != 0 else 0
        
        return sharpe_ratio
    
    def _calculate_sortino_ratio(self, returns):
        """
        Calculate Sortino Ratio (using only downside risk).
        
        Args:
            returns: Series of returns
            
        Returns:
            Sortino Ratio
        """
        if returns.empty:
            return 0
            
        # Downside returns
        downside_returns = returns[returns < 0]
        
        if downside_returns.empty or downside_returns.std() == 0:
            return 0 if returns.mean() <= 0 else float('inf')
            
        # Annualize
        avg_return = returns.mean() * 252
        downside_std = downside_returns.std() * np.sqrt(252)
        
        # Calculate Sortino
        sortino_ratio = (avg_return - self.risk_free_rate) / downside_std if downside_std != 0 else 0
        
        return sortino_ratio
    
    def plot_performance(self, save_path=None):
        """
        Generate performance visualization.
        
        Args:
            save_path: Path to save the performance plot
        """
        if self.trades.empty:
            self.logger.warning("No trades to plot")
            return
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 15), gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # 1. Equity Curve
        ax1 = axes[0]
        ax1.plot(range(len(self.equity_curve)), self.equity_curve, 'b-', linewidth=2)
        ax1.set_title('Equity Curve', fontsize=14)
        ax1.set_xlabel('Trade Number')
        ax1.set_ylabel('Account Equity ($)')
        ax1.grid(True, alpha=0.3)
        
        # Add horizontal line for initial capital
        ax1.axhline(y=self.initial_capital, color='r', linestyle='--', alpha=0.7, 
                   label=f'Initial Capital (${self.initial_capital:,.0f})')
        ax1.legend()
        
        # 2. Trade Profit/Loss
        ax2 = axes[1]
        profits = self.trades['profit'].values
        colors = ['g' if p > 0 else 'r' for p in profits]
        ax2.bar(range(len(profits)), profits, color=colors, alpha=0.7)
        ax2.set_title('Trade Profit/Loss', fontsize=14)
        ax2.set_xlabel('Trade Number')
        ax2.set_ylabel('Profit/Loss ($)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Drawdown
        ax3 = axes[2]
        peaks = pd.Series(self.equity_curve).cummax()
        drawdowns = (pd.Series(self.equity_curve) - peaks) / peaks * 100
        ax3.fill_between(range(len(drawdowns)), 0, drawdowns, color='r', alpha=0.3)
        ax3.plot(range(len(drawdowns)), drawdowns, 'r-', linewidth=1)
        ax3.set_title('Drawdown (%)', fontsize=14)
        ax3.set_xlabel('Trade Number')
        ax3.set_ylabel('Drawdown (%)')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save or display
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Performance plot saved to {save_path}")
            plt.close(fig)
        else:
            plt.show()
    
    def generate_report(self, output_path=None):
        """
        Generate a comprehensive trading performance report.
        
        Args:
            output_path: Path to save the report
            
        Returns:
            Formatted performance report
        """
        report = "===== Trading Strategy Performance Report =====\n\n"
        
        for metric, value in self.performance_metrics.items():
            # Format metric names and values
            formatted_metric = ' '.join(word.capitalize() for word in metric.split('_'))
            if isinstance(value, float):
                report += f"{formatted_metric}: {value:.2f}\n"
            else:
                report += f"{formatted_metric}: {value}\n"
        
        # Add trade details
        report += "\n===== Individual Trade Performance =====\n"
        if not self.trades.empty:
            trade_summary = self.trades.describe()
            report += trade_summary.to_string()
        else:
            report += "No trades were executed during the backtest."
        
        # Save report if path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            self.logger.info(f"Performance report saved to {output_path}")
        
        return report
    
    def save_trades(self, output_path=None):
        """
        Save trade details to CSV file.
        
        Args:
            output_path: Path to save the trades
            
        Returns:
            Path to saved file
        """
        if self.trades.empty:
            self.logger.warning("No trades to save")
            return None
            
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f'backtest_trades_{timestamp}.csv')
            
        try:
            self.trades.to_csv(output_path, index=False)
            self.logger.info(f"Trades saved to {output_path}")
            return output_path
        except Exception as e:
            self.logger.error(f"Error saving trades: {str(e)}")
            return None