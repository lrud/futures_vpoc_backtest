import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats

# Configuration
BASE_DIR = '/home/lr/Documents/FUTURUES_PROJECT/futures_vpoc_backtest/'
DATA_DIR = os.path.join(BASE_DIR, 'DATA')
CLEANED_DATA_DIR = os.path.join(BASE_DIR, 'DATA/CLEANED')
STRATEGY_DIR = os.path.join(BASE_DIR, 'STRATEGY')
BACKTEST_DIR = os.path.join(BASE_DIR, 'BACKTEST')

# Ensure backtest directory exists
if not os.path.exists(BACKTEST_DIR):
    os.makedirs(BACKTEST_DIR)

# Backtest Configuration
PRICE_PRECISION = 0.25  # Price increment for volume profile bins
SESSION_TYPE = 'RTH'  # Define SESSION_TYPE for consistency with VPOC script

# Backtest Parameters
INITIAL_CAPITAL = 100000  # Starting capital
COMMISSION_PER_TRADE = 10  # Commission per trade
SLIPPAGE = 0.25  # Slippage in price points
RISK_PER_TRADE = 0.01  # 1% risk per trade
RISK_FREE_RATE = 0.02  # 2% annual risk-free rate
MARGIN_REQUIREMENT = 0.1  # 10% margin requirement
OVERNIGHT_MARGIN = 0.15  # 15% overnight margin
MAX_POSITION_SIZE = 10   # Maximum contracts per position
MIN_CAPITAL_BUFFER = 0.2 # 20% capital buffer requirement

class FuturesBacktest:
    """
    Framework for backtesting ML-enhanced trading strategies.
    Updated to use latest enhanced model with GARCH/log features.
    """
    def __init__(self, 
                 data, 
                 initial_capital=INITIAL_CAPITAL, 
                 commission=COMMISSION_PER_TRADE, 
                 slippage=SLIPPAGE,
                 risk_per_trade=RISK_PER_TRADE):
        """
        Initialize the backtest with key parameters
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Input dataframe with trading signals
        initial_capital : float, optional
            Starting capital for the backtest (default: from configuration)
        commission : float, optional
            Per-trade commission cost (default: from configuration)
        slippage : float, optional
            Estimated slippage per trade in price points (default: from configuration)
        risk_per_trade : float, optional
            Percentage of capital to risk per trade (default: from configuration)
        """
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.risk_per_trade = risk_per_trade
        
        # Validate input data
        required_columns = ['date', 'signal', 'price', 'stop_loss', 'target']
        for col in required_columns:
            if col not in self.data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Sort data by date
        self.data = self.data.sort_values('date')
        
        # Initialize tracking variables
        self.portfolio = []
        self.trades = []
        self.performance_metrics = {}

    def calculate_position_size(self, entry_price, stop_loss, capital):
        """
        Calculate optimal position size using fixed fractional risk management
        
        Parameters:
        -----------
        entry_price : float
            Entry price of the trade
        stop_loss : float
            Stop loss price
        capital : float
            Available trading capital
        
        Returns:
        --------
        float
            Position size (number of contracts)
        """
        # Risk per trade based on configuration
        risk_amount = capital * self.risk_per_trade
        
        # Calculate risk per contract
        risk_per_contract = abs(entry_price - stop_loss)
        
        # Calculate position size
        position_size = int(risk_amount / (risk_per_contract + self.slippage + self.commission))
        
        return max(1, position_size)  # Ensure at least 1 contract
    
    def _simulate_exit_price(self, base_price, slippage, signal_type):
        """Simulate realistic exit prices with random slippage"""
        slippage_impact = np.random.uniform(-slippage * 2, slippage * 2)
        return base_price + (slippage_impact if signal_type == 'BUY' else -slippage_impact)

    def _check_margin_requirements(self, position_size, price, capital):
        """Validate margin requirements for position"""
        margin_required = position_size * price * MARGIN_REQUIREMENT
        overnight_margin = position_size * price * OVERNIGHT_MARGIN
        return margin_required <= (capital * (1 - MIN_CAPITAL_BUFFER))

    def _validate_position_size(self, calculated_size):
        """Ensure position size meets limits"""
        return min(calculated_size, MAX_POSITION_SIZE)

    def run_backtest(self, risk_free_rate=0.02):
        """Execute backtest with enhanced model and professional rules"""
        capital = self.initial_capital
        current_position = None
        trade_log = []

        # Load enhanced ML model with GARCH/log features
        print("🤖 Loading enhanced ML model...")
        try:
            import torch

            model_path = '/workspace/TRAINING/enhanced_simple/train_20251022_211054/model_final.pt'

            if not os.path.exists(model_path):
                print("❌ Enhanced model not found, using basic signals")
                model_available = False
            else:
                # Load checkpoint
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

                # Create model matching trained architecture
                model = torch.nn.Sequential(
                    torch.nn.Linear(54, 32),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm1d(32),
                    torch.nn.Linear(32, 1)
                )
                model.eval()

                # Load model weights
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model_keys = {k: v for k, v in checkpoint.items()
                                if not k in ['version', 'timestamp', 'architecture', 'feature_columns',
                                                        'optimizer_state_dict', 'epoch', 'loss', 'metadata']}
                    model.load_state_dict(model_keys)

                print("✅ Enhanced ML model loaded successfully")
                model_available = True

        except Exception as e:
            print(f"❌ Error loading enhanced model: {e}")
            model_available = False
        
        for _, signal in self.data.iterrows():
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
                    
                    # Calculate profit with increased costs
                    profit = self._calculate_trade_profit(current_position, exit_price)
                    capital += profit
                    
                    trade_log.append({
                        'entry_date': current_position['entry_date'],
                        'exit_date': date,
                        'entry_price': current_position['entry_price'],
                        'exit_price': exit_price,
                        'type': current_position['type'],
                        'profit': profit,
                        'position_size': current_position['position_size']
                    })
                    current_position = None
            
            # Open new position with validation
            if not current_position:
                # Calculate and validate position size
                position_size = self.calculate_position_size(entry_price, stop_loss, capital)
                position_size = self._validate_position_size(position_size)
                
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
        
        # Convert trade log to DataFrame at end of backtest
        self.trades = pd.DataFrame(trade_log)
        
        # Calculate performance metrics if trades exist
        if not self.trades.empty:
            self._calculate_metrics(risk_free_rate)
            
        return self.performance_metrics

    def _calculate_trade_profit(self, position, exit_price):
        """Calculate realistic profit including all costs"""
        # Calculate gross profit
        if position['type'] == 'LONG':
            gross_profit = (exit_price - position['entry_price']) * position['position_size']
        else:  # SHORT
            gross_profit = (position['entry_price'] - exit_price) * position['position_size']
        
        # Enhanced cost calculation
        total_commission = self.commission * 2  # Entry and exit
        exchange_fees = 2.50 * position['position_size'] * 2  # $2.50 per contract per side
        clearing_fees = 0.50 * position['position_size'] * 2  # $0.50 per contract per side
        
        # Calculate net profit
        net_profit = gross_profit - (total_commission + exchange_fees + clearing_fees)
        
        return net_profit

    def _calculate_metrics(self, risk_free_rate=0.02):
        """
        Calculate comprehensive trading performance metrics
        
        Parameters:
        -----------
        risk_free_rate : float, optional
            Annual risk-free rate for Sharpe ratio calculation
        """
        if self.trades.empty:
            raise ValueError("No trades to analyze")
        
        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = self.trades[self.trades['profit'] > 0]
        losing_trades = self.trades[self.trades['profit'] <= 0]
        
        # Performance calculations
        self.performance_metrics = {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / total_trades * 100,
            'total_profit': self.trades['profit'].sum(),
            'average_profit_per_trade': self.trades['profit'].mean(),
            'max_profit': self.trades['profit'].max(),
            'max_loss': self.trades['profit'].min(),
            'profit_factor': abs(winning_trades['profit'].sum() / losing_trades['profit'].sum()) if len(losing_trades) > 0 else np.inf
        }
        
        # Drawdown calculation
        cumulative_returns = self.initial_capital + self.trades['profit'].cumsum()
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak * 100
        
        self.performance_metrics.update({
            'max_drawdown': drawdown.min(),
            'sharpe_ratio': self._calculate_sharpe_ratio(risk_free_rate)
        })

    def _calculate_sharpe_ratio(self, risk_free_rate):
        """
        Calculate Sharpe Ratio of the trading strategy
        
        Parameters:
        -----------
        risk_free_rate : float
            Annual risk-free rate
        
        Returns:
        --------
        float
            Sharpe Ratio
        """
        # Calculate daily returns
        returns = self.trades['profit'] / self.initial_capital
        
        # Annualize returns and standard deviation
        avg_return = returns.mean() * 252  # Assuming 252 trading days per year
        std_return = returns.std() * np.sqrt(252)
        
        # Calculate Sharpe Ratio
        sharpe_ratio = (avg_return - risk_free_rate) / std_return if std_return != 0 else 0
        
        return sharpe_ratio

    def plot_performance(self, save_path=None):
        """
        Generate performance visualization
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the performance plot
        """
        if self.trades.empty:
            print("No trades to plot")
            return
        
        # Cumulative returns plot
        plt.figure(figsize=(12, 6))
        cumulative_returns = self.initial_capital + self.trades['profit'].cumsum()
        plt.plot(self.trades['exit_date'], cumulative_returns, label='Cumulative Portfolio Value')
        plt.title('Portfolio Performance')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def generate_report(self, output_path=None):
        """
        Generate a comprehensive trading performance report
        
        Parameters:
        -----------
        output_path : str, optional
            Path to save the report
        
        Returns:
        --------
        str
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
        trade_summary = self.trades.describe()
        report += trade_summary.to_string()
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
        
        return report

def load_signals_data(file_path):
    """
    Load trading signals from a CSV file
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing trading signals
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with trading signals
    """
    try:
        df = pd.read_csv(file_path, parse_dates=['date'])
        return df
    except Exception as e:
        print(f"Error loading signals file: {e}")
        return None

def main():
    """
    Main function to run the backtest
    """
    # Load new enhanced trading signals
    signals_path = os.path.join(STRATEGY_DIR, 'trading_signals.csv')
    print(f"\n===== Backtesting Enhanced VPOC Strategy =====")
    
    # Load signals
    signals_df = load_signals_data(signals_path)
    
    if signals_df is not None:
        # Initialize and run backtest
        backtest = FuturesBacktest(
            signals_df, 
            initial_capital=INITIAL_CAPITAL, 
            commission=COMMISSION_PER_TRADE, 
            slippage=SLIPPAGE,
            risk_per_trade=RISK_PER_TRADE
        )
        performance = backtest.run_backtest(risk_free_rate=RISK_FREE_RATE)
        
        # Generate report and plot
        output_report_path = os.path.join(BACKTEST_DIR, 'enhanced_vpoc_performance_report.txt')
        output_plot_path = os.path.join(BACKTEST_DIR, 'enhanced_vpoc_performance.png')
        
        report = backtest.generate_report(output_report_path)
        print(report)
        
        backtest.plot_performance(output_plot_path)
        print(f"Performance report saved to: {output_report_path}")
        print(f"Performance plot saved to: {output_plot_path}")

if __name__ == "__main__":
    main()