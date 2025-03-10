#!/usr/bin/env python
"""
ML_BACKTEST.py - Backtesting Framework for ML-Enhanced Futures Strategy

This script implements a comprehensive backtesting framework for evaluating
ML-enhanced trading strategies on ES futures. It loads trained models,
generates predictions, creates trading signals, and evaluates performance
through the existing backtesting infrastructure.
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import argparse

# Dynamic path configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SCRIPT_DIR)

# Import project-specific modules
from DATA_LOADER import load_futures_data
from STRATEGY import calculate_volume_profile, find_vpoc, find_value_area
from BACKTEST import FuturesBacktest
from ML_TEST import AMDOptimizedFuturesModel

# Configuration
TRAINING_DIR = os.path.join(PROJECT_ROOT, "TRAINING")
BACKTEST_DIR = os.path.join(PROJECT_ROOT, "BACKTEST")
ML_BACKTEST_DIR = os.path.join(BACKTEST_DIR, "ML_RESULTS")

# Ensure output directories exist
for directory in [BACKTEST_DIR, ML_BACKTEST_DIR]:
    os.makedirs(directory, exist_ok=True)

# Backtest Parameters - same as in BACKTEST.py for consistency
INITIAL_CAPITAL = 100000
COMMISSION_PER_TRADE = 10
SLIPPAGE = 0.25
RISK_PER_TRADE = 0.01
RISK_FREE_RATE = 0.02

class MLBacktester:
    """
    Framework for backtesting ML-enhanced trading strategies.
    """
    def __init__(self, 
                model_path=None,
                data_path=None,
                session_type='RTH',
                contract_filter='ES',
                lookback_periods=[10, 20, 50],
                prediction_threshold=0.5,
                signal_confidence_threshold=70,
                output_dir=ML_BACKTEST_DIR):
        """
        Initialize the ML backtest framework.
        
        Parameters:
        -----------
        model_path : str
            Path to the trained model checkpoint
        data_path : str
            Path to the futures data directory
        session_type : str
            Trading session type (RTH or ETH)
        contract_filter : str
            Contract prefix to filter (e.g., 'ES')
        lookback_periods : list
            Periods for feature engineering
        prediction_threshold : float
            Threshold for converting model outputs to signals
        signal_confidence_threshold : float
            Minimum confidence required for a valid signal
        output_dir : str
            Directory for saving results
        """
        self.model_path = model_path or os.path.join(TRAINING_DIR, "es_futures_model_final.pt")
        self.data_path = data_path or os.path.join(PROJECT_ROOT, "DATA")
        self.session_type = session_type
        self.contract_filter = contract_filter
        self.lookback_periods = lookback_periods
        self.prediction_threshold = prediction_threshold
        self.signal_confidence_threshold = signal_confidence_threshold
        self.output_dir = output_dir
        
        # Load data
        print(f"Loading futures data from {self.data_path}...")
        self.raw_data = load_futures_data(self.data_path)
        if self.raw_data is None:
            raise ValueError("Failed to load futures data")
        
        # Filter data
        if session_type:
            self.raw_data = self.raw_data[self.raw_data['session'] == session_type]
            print(f"Filtered to {session_type} sessions: {len(self.raw_data)} rows")
        
        if contract_filter:
            self.raw_data = self.raw_data[self.raw_data['contract'].str.startswith(contract_filter)]
            print(f"Filtered to {contract_filter} contracts: {len(self.raw_data)} rows")
        
        # Load model
        self.model, self.feature_columns, self.scaler, self.scaler_fitted = self.load_ml_model()
        
        # Initialize results storage
        self.signals_df = None
        self.backtest_results = None
        self.comparison_results = None
        
    def load_ml_model(self):
        """
        Load trained model and associated metadata.
        
        Returns:
        --------
        tuple
            (model, feature_columns, scaler)
        """
        print(f"Loading model from {self.model_path}...")
        
        try:
            checkpoint = torch.load(self.model_path, map_location=torch.device('cpu'))
            
            # Extract model state dict and metadata
            model_state = checkpoint.get('model_state_dict')
            feature_columns = checkpoint.get('feature_columns')
            
            if model_state is None:
                raise ValueError("Model state not found in checkpoint")
            
            # If feature columns weren't saved, use default columns based on lookback periods
            if feature_columns is None:
                print("Feature columns not found in checkpoint, using default feature set")
                feature_columns = [
                    'vpoc', 'total_volume', 'price_range', 'range_pct', 'close_change_pct',
                    'session_high', 'session_low', 'session_open', 'session_close', 'vwap',
                    'close_to_vwap_pct'
                ]
                # Add derived features for each lookback period
                for period in self.lookback_periods:
                    period_features = [
                        f'price_mom_{period}d', f'volatility_{period}d', f'volume_trend_{period}d',
                        f'vpoc_change_{period}d', f'vpoc_pct_change_{period}d', f'range_change_{period}d'
                    ]
                    feature_columns.extend(period_features)
            
            # Determine input dimension from feature columns
            input_dim = len(feature_columns)
            
            # Recreate model architecture
            model = AMDOptimizedFuturesModel(input_dim=input_dim)
            model.load_state_dict(model_state)
            model.eval()  # Set to evaluation mode
            
            # Create a new StandardScaler
            scaler = StandardScaler()
            scaler_fitted = False
            
            # If a scaler was saved in the checkpoint, use its properties
            saved_scaler = checkpoint.get('scaler')
            if saved_scaler is not None:
                try:
                    # If the saved scaler has mean_ and scale_ attributes, use them
                    if hasattr(saved_scaler, 'mean_') and hasattr(saved_scaler, 'scale_'):
                        scaler.mean_ = saved_scaler.mean_
                        scaler.scale_ = saved_scaler.scale_
                        scaler.n_samples_seen_ = saved_scaler.n_samples_seen_
                        scaler_fitted = True
                        print("Successfully loaded scaler from checkpoint")
                except Exception as e:
                    print(f"Warning: Could not load saved scaler properties: {e}")
                    print("Will fit a new StandardScaler on the data")
            else:
                print("No scaler found in checkpoint. Will fit a new StandardScaler on the data")
            
            print(f"Successfully loaded model with {input_dim} input features")
            
            return model, feature_columns, scaler, scaler_fitted
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    
    def generate_features(self):
        """
        Generate features for ML model from raw data using pandas vectorized operations.
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with engineered features
        """
        print("\n===== FEATURE GENERATION DIAGNOSTICS =====")
        
        # Get all unique dates for session-based feature engineering
        self.raw_data['date'] = pd.to_datetime(self.raw_data['date'])
        all_dates = sorted(self.raw_data['date'].unique())
        
        features_list = []
        skipped_dates = []
        
        # Process each session to extract features
        for date in all_dates:
            try:
                # Get data for this session
                session_data = self.raw_data[self.raw_data['date'] == date]
                
                # Skip sessions with insufficient data
                if len(session_data) < 10:
                    print(f"Skipping {date}: Insufficient data ({len(session_data)} rows)")
                    skipped_dates.append(date)
                    continue
                
                # Calculate volume profile and derived metrics
                volume_profile = calculate_volume_profile(session_data)
                vpoc = find_vpoc(volume_profile)
                val, vah, va_volume_pct = find_value_area(volume_profile)
                
                # Calculate session statistics
                session_high = session_data['high'].max()
                session_low = session_data['low'].min()
                session_open = session_data['open'].iloc[0]
                session_close = session_data['close'].iloc[-1]
                session_volume = session_data['volume'].sum()
                
                # Calculate price change percentage
                close_change_pct = (session_close - session_open) / max(session_open, 0.0001) * 100
                
                # Calculate range metrics
                price_range = max(session_high - session_low, 0.0001)
                range_pct = price_range / max(session_open, 0.0001) * 100
                
                # Create feature dictionary for this session
                session_features = {
                    'date': date,
                    'vpoc': float(vpoc),
                    'total_volume': float(session_volume),
                    'price_range': float(price_range),
                    'range_pct': float(range_pct),
                    'close_change_pct': float(close_change_pct),
                    'session_high': float(session_high),
                    'session_low': float(session_low),
                    'session_open': float(session_open),
                    'session_close': float(session_close),
                    'value_area_low': float(val),
                    'value_area_high': float(vah),
                    'value_area_width': float(vah - val)
                }
                
                # Calculate VWAP
                vwap = (session_data['close'] * session_data['volume']).sum() / max(session_data['volume'].sum(), 0.0001)
                session_features['vwap'] = float(vwap)
                session_features['close_to_vwap_pct'] = float((session_close - vwap) / max(vwap, 0.0001) * 100)
                
                features_list.append(session_features)
                
            except Exception as e:
                print(f"Error processing session {date}: {e}")
                skipped_dates.append(date)
                continue
        
        # Convert list to DataFrame
        if not features_list:
            raise ValueError("No valid features could be generated. Check data and processing.")
        
        features_df = pd.DataFrame(features_list)
        features_df['date'] = pd.to_datetime(features_df['date'])
        
        # Sort by date to ensure correct order for time series features
        features_df = features_df.sort_values('date')
        
        # Add lagged features based on lookback periods
        max_lookback = max(self.lookback_periods)
        
        for period in self.lookback_periods:
            # Skip if not enough data
            if len(features_df) <= period:
                continue
            
            # Price momentum
            features_df[f'price_mom_{period}d'] = features_df['session_close'].pct_change(period) * 100
            
            # Volatility
            features_df[f'volatility_{period}d'] = features_df['close_change_pct'].rolling(period).std()
            
            # Volume trend
            features_df[f'volume_trend_{period}d'] = features_df['total_volume'].pct_change(period) * 100
            
            # VPOC migration
            features_df[f'vpoc_change_{period}d'] = features_df['vpoc'].diff(period)
            features_df[f'vpoc_pct_change_{period}d'] = features_df['vpoc'].pct_change(period) * 100
            
            # Range evolution
            features_df[f'range_change_{period}d'] = features_df['range_pct'].pct_change(period) * 100
        
        # Clean up data
        features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Drop rows with NaN values
        original_count = len(features_df)
        features_df.dropna(inplace=True)
        dropped_count = original_count - len(features_df)
        
        # Print comprehensive diagnostics
        print("\n===== FEATURE GENERATION SUMMARY =====")
        print(f"Total sessions processed: {len(features_df)}")
        print(f"Total sessions skipped: {len(skipped_dates)}")
        
        if len(skipped_dates) < 20:
            print(f"Skipped dates: {skipped_dates}")
        else:
            print(f"Skipped dates: {len(skipped_dates)} dates")
        
        print(f"Rows dropped due to NaN values: {dropped_count}")
        print(f"Date range: {features_df['date'].min()} to {features_df['date'].max()}")
        
        # Feature statistics
        print("\nFeature Statistics:")
        for col in features_df.columns:
            if col != 'date':
                print(f"\n{col}:")
                print(features_df[col].describe())
        
        return features_df
    
    def generate_ml_signals(self, features_df):
        """
        Generate trading signals based on ML model predictions with compatible format for FuturesBacktest.
        """
        print("\n===== ML SIGNAL GENERATION =====")
        
        # Validate feature columns
        available_features = [col for col in features_df.columns if col in self.feature_columns]
        
        # Check if features are available
        if not available_features:
            print(f"ERROR: No matching features found. Model expects: {self.feature_columns}")
            print(f"Available columns: {features_df.columns.tolist()}")
            return None
        
        # Prepare features for model
        X = features_df[available_features].values
        
        # Check if scaler is fitted, if not fit it now
        if not hasattr(self, 'scaler_fitted') or not self.scaler_fitted:
            print("Fitting StandardScaler on current data...")
            self.scaler.fit(X)
            self.scaler_fitted = True
        
        # Now transform the data
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        
        # Generate predictions
        with torch.no_grad():
            predictions = self.model(X_tensor).squeeze().cpu().numpy()
        
        # Add predictions and confidence scores
        features_df['ml_prediction'] = predictions
        features_df['confidence'] = np.abs(predictions) * 100  # Simplified confidence calculation
        
        # Generate signals - simplified thresholding
        features_df['signal_type'] = np.where(
            predictions > self.prediction_threshold, 'BUY',
            np.where(predictions < -self.prediction_threshold, 'SELL', 'NEUTRAL')
        )
        
        # Filter signals by confidence
        signals_df = features_df[
            (features_df['signal_type'] != 'NEUTRAL') & 
            (features_df['confidence'] >= self.signal_confidence_threshold)
        ].copy()
        
        # Signal values as TEXT for compatibility with FuturesBacktest class
        signals_df['signal'] = signals_df['signal_type']  # Use 'BUY' and 'SELL' strings
        
        # Set critical price levels
        signals_df['price'] = signals_df['vpoc']  # Use VPOC as entry price
        signals_df['stop_loss'] = np.where(
            signals_df['signal'] == 'BUY',
            signals_df['value_area_low'] * 0.99,  # Long stop below VA low
            signals_df['value_area_high'] * 1.01   # Short stop above VA high
        )
        
        # Add target based on risk:reward (e.g., 2:1)
        signals_df['target'] = np.where(
            signals_df['signal'] == 'BUY',
            signals_df['price'] + 2 * (signals_df['price'] - signals_df['stop_loss']),  # Long target
            signals_df['price'] - 2 * (signals_df['stop_loss'] - signals_df['price'])   # Short target
        )
        
        # Position sizing and metadata
        signals_df['position_size'] = 1.0
        signals_df['reason'] = 'ML_' + signals_df['signal_type']
        
        # Print signal statistics
        signal_counts = features_df['signal_type'].value_counts()
        print("SIGNAL TYPE DISTRIBUTION:")
        print(signal_counts[signal_counts.index != 'NEUTRAL'])
        print(f"Total signals before confidence filter: {len(features_df)}")
        print(f"Total signals after confidence filter: {len(signals_df)}")
        
        # Save detailed signal data for debugging
        signal_debug_file = os.path.join(self.output_dir, 'ml_signal_debug.csv')
        features_df.to_csv(signal_debug_file, index=False)
        print(f"Detailed signal debug saved to: {signal_debug_file}")
        
        # Final columns for backtest - ensure these match what FuturesBacktest expects
        backtest_cols = ['date', 'signal', 'price', 'stop_loss', 'target', 'position_size', 'confidence', 'reason']
        
        print(f"Final signals shape: {signals_df[backtest_cols].shape}")
        return signals_df[backtest_cols]


    def run_backtest(self, signals_df):
        """
        Run backtest on the generated signals with improved diagnostics and error handling.
        
        Parameters:
        -----------
        signals_df : DataFrame
            DataFrame containing trading signals
            
        Returns:
        --------
        dict
            Backtest results including performance metrics and trades
        """
        print("\nStage 3: Running Backtest")
        
        if signals_df is None or signals_df.empty:
            print("No valid trading signals found.")
            return None

        # Print detailed information about signals before backtesting
        print(f"Total signals to backtest: {len(signals_df)}")
        print(f"Signal date range: {signals_df['date'].min()} to {signals_df['date'].max()}")
        print(f"Signal distribution: {signals_df['signal'].value_counts().to_dict()}")
        
        # Validate columns required for backtesting
        required_columns = ['date', 'signal', 'price', 'stop_loss', 'target']
        missing_columns = [col for col in required_columns if col not in signals_df.columns]
        
        if missing_columns:
            print(f"ERROR: Missing required columns for backtest: {missing_columns}")
            return None
        
        # Sample signals for inspection
        print("\nSample signals for inspection (first 3):")
        print(signals_df.head(3).to_string())
        
        try:
            print("\nExecuting backtest...")
            
            # Create and run backtest
            backtest = FuturesBacktest(
                signals_df,
                initial_capital=INITIAL_CAPITAL,
                commission=COMMISSION_PER_TRADE,
                slippage=SLIPPAGE,
                risk_per_trade=RISK_PER_TRADE
            )
            
            # Run backtest and capture results
            results = backtest.run_backtest(risk_free_rate=RISK_FREE_RATE)
            
            # Store results for later use
            self.backtest_results = {
                'performance': results,
                'trades': backtest.trades
            }
            
            # Print performance metrics
            if results:
                print("\nBacktest Performance Metrics:")
                for metric, value in results.items():
                    if isinstance(value, float):
                        print(f"{metric}: {value:.4f}")
                    else:
                        print(f"{metric}: {value}")
            
            # Print trade summary
            if hasattr(backtest, 'trades') and not backtest.trades.empty:
                trades_df = backtest.trades
                print(f"\nTotal Trades: {len(trades_df)}")
                print(f"Winning Trades: {len(trades_df[trades_df['profit'] > 0])}")
                print(f"Losing Trades: {len(trades_df[trades_df['profit'] <= 0])}")
                print(f"Average Profit: {trades_df['profit'].mean():.2f}")
                print(f"Total P&L: {trades_df['profit'].sum():.2f}")
            else:
                print("\nWARNING: No trades were executed.")
                # Create empty trades DataFrame to avoid errors in comparison
                if hasattr(backtest, 'trades'):
                    empty_trades = backtest.trades
                else:
                    empty_trades = pd.DataFrame(columns=[
                        'entry_date', 'exit_date', 'type', 'entry_price', 'exit_price', 
                        'profit', 'position_size'
                    ])
                
                self.backtest_results = {
                    'performance': results or {},
                    'trades': empty_trades
                }
            
            return results
            
        except Exception as e:
            print(f"ERROR in backtest execution: {e}")
            import traceback
            traceback.print_exc()
            return None
        
    def compare_with_original_strategy(self):
        """Compare ML strategy with the original VPOC strategy."""
        print("Comparing ML strategy with original VPOC strategy...")
        
        # Initialize results if they don't exist
        if not hasattr(self, 'backtest_results') or self.backtest_results is None:
            self.backtest_results = {
                'performance': {},
                'trades': pd.DataFrame(),
                'signals': pd.DataFrame()
            }
        
        # Load original strategy signals
        original_signals_path = os.path.join(PROJECT_ROOT, "STRATEGY", "trading_signals.csv")
        
        if not os.path.exists(original_signals_path):
            print(f"Original strategy signals not found at {original_signals_path}")
            print("Running original strategy to generate signals...")
            
            # Import and run original strategy
            from STRATEGY import run_analysis
            run_analysis()
        
        if os.path.exists(original_signals_path):
            # Load original strategy signals
            original_signals = pd.read_csv(original_signals_path, parse_dates=['date'], 
                                dtype={'confidence': float, 'price': float, 'stop_loss': float, 'target': float})
            
            # Run backtest on original strategy
            original_backtest = FuturesBacktest(
                original_signals,
                initial_capital=INITIAL_CAPITAL,
                commission=COMMISSION_PER_TRADE,
                slippage=SLIPPAGE,
                risk_per_trade=RISK_PER_TRADE
            )
            
            original_performance = original_backtest.run_backtest(risk_free_rate=RISK_FREE_RATE)
            
            # Prepare comparison DataFrame
            comparison = []
            
            # Add metrics from both strategies
            for metric, value in self.backtest_results['performance'].items():
                if metric in original_performance:
                    comparison.append({
                        'Metric': metric,
                        'ML Strategy': value,
                        'Original Strategy': original_performance[metric],
                        'Difference': value - original_performance[metric],
                        'Improvement %': ((value - original_performance[metric]) / abs(original_performance[metric])) * 100 if original_performance[metric] != 0 else float('inf')
                    })
            
            comparison_df = pd.DataFrame(comparison)
            
            # Save comparison to CSV
            comparison_file = os.path.join(self.output_dir, 'strategy_comparison.csv')
            comparison_df.to_csv(comparison_file, index=False)
            print(f"Strategy comparison saved to {comparison_file}")
            
            # Create comparison plot
            self._plot_strategy_comparison(comparison_df, original_backtest.trades)
            
            # Store comparison results
            self.comparison_results = comparison_df
            
            return comparison_df
        else:
            print("Could not load original strategy signals for comparison")
            return None
    
    def _plot_strategy_comparison(self, comparison_df, original_trades):
        """Create visualization comparing the two strategies with robust error handling."""
        print("\n===== Strategy Comparison Visualization =====")
        
        try:
            # Prepare ML strategy trades
            ml_trades = self.backtest_results.get('trades', pd.DataFrame())
            if (ml_trades.empty):
                print("Warning: No ML strategy trades available for comparison")
                ml_trades = pd.DataFrame(columns=['entry_date', 'exit_date', 'profit'])
            
            # Prepare original trades DataFrame
            if original_trades is None or original_trades.empty:
                print("Warning: No original strategy trades available for comparison")
                original_trades = pd.DataFrame(columns=['entry_date', 'exit_date', 'profit'])
            
            # Ensure datetime columns for plotting
            for trades_df in [ml_trades, original_trades]:
                for col in ['entry_date', 'exit_date']:
                    if col in trades_df.columns:
                        trades_df[col] = pd.to_datetime(trades_df[col])
            
            # Create figure with multiple subplots
            fig, axs = plt.subplots(2, 2, figsize=(15, 12))
            plt.suptitle('Strategy Comparison', fontsize=16)
            
            # 1. Equity Curves
            ax = axs[0, 0]
            if not ml_trades.empty and 'exit_date' in ml_trades.columns and 'profit' in ml_trades.columns:
                ml_equity = INITIAL_CAPITAL + ml_trades['profit'].cumsum()
                ax.plot(ml_trades['exit_date'], ml_equity, label='ML Strategy', color='blue')
            
            if not original_trades.empty and 'exit_date' in original_trades.columns and 'profit' in original_trades.columns:
                original_equity = INITIAL_CAPITAL + original_trades['profit'].cumsum()
                ax.plot(original_trades['exit_date'], original_equity, label='Original Strategy', color='red')
            
            ax.set_title('Equity Curves')
            ax.set_xlabel('Date')
            ax.set_ylabel('Equity ($)')
            ax.legend()
            ax.grid(True)
            
            # Rest of the plotting code remains the same
            # 2. Performance Metrics Comparison
            ax = axs[0, 1]
            if not comparison_df.empty:
                metrics = comparison_df['Metric'].tolist()[:3]  # Take first 3 metrics
                ml_values = comparison_df['ML Strategy'].tolist()[:3]
                orig_values = comparison_df['Original Strategy'].tolist()[:3]
                
                x = np.arange(len(metrics))
                width = 0.35
                ax.bar(x - width/2, ml_values, width, label='ML Strategy', color='blue')
                ax.bar(x + width/2, orig_values, width, label='Original Strategy', color='red')
                
                ax.set_title('Key Performance Metrics')
                ax.set_xticks(x)
                ax.set_xticklabels(metrics, rotation=45)
                ax.legend()
            
            # 3. Trade Distribution
            ax = axs[1, 0]
            if 'profit' in ml_trades.columns:
                ax.hist(ml_trades['profit'], bins=20, alpha=0.5, label='ML Strategy', color='blue')
            if 'profit' in original_trades.columns:
                ax.hist(original_trades['profit'], bins=20, alpha=0.5, label='Original Strategy', color='red')
            ax.set_title('Trade Profit Distribution')
            ax.set_xlabel('Profit ($)')
            ax.set_ylabel('Frequency')
            ax.legend()
            
            # 4. Monthly Performance
            ax = axs[1, 1]
            for trades, label, color in [(ml_trades, 'ML Strategy', 'blue'),
                                       (original_trades, 'Original Strategy', 'red')]:
                if not trades.empty and 'exit_date' in trades.columns and 'profit' in trades.columns:
                    monthly = trades.set_index('exit_date')['profit'].resample('M').sum()
                    monthly.plot(kind='bar', ax=ax, label=label, color=color, alpha=0.5)
            
            ax.set_title('Monthly Performance')
            ax.set_xlabel('Month')
            ax.set_ylabel('Profit ($)')
            ax.legend()
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            plt.tight_layout()
            comparison_plot_path = os.path.join(self.output_dir, 'strategy_comparison.png')
            plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Strategy comparison plot saved to {comparison_plot_path}")
            
        except Exception as e:
            print(f"Error in strategy comparison plotting: {e}")
            import traceback
            traceback.print_exc()
    
    def run(self):
        """
        Run the complete ML backtesting pipeline with enhanced diagnostic capabilities.
        """
        print("\n===== ML BACKTESTING PIPELINE =====\n")
        
        try:
            # Stage 1: Feature Generation
            print("Stage 1: Generating Features")
            features_df = self.generate_features()
            
            if features_df is None or features_df.empty:
                print("ERROR: No features could be generated!")
                return
            
            print(f"Features generated: {len(features_df)} rows")
            print(f"Feature columns: {features_df.columns.tolist()}")
            
            # Stage 2: ML Signal Generation
            print("\nStage 2: Generating ML Signals")
            self.signals_df = self.generate_ml_signals(features_df)
            
            # Comprehensive signal validation
            if self.signals_df is None:
                print("ERROR: Signal generation returned None!")
                return
            
            if self.signals_df.empty:
                print("WARNING: No signals were generated after filtering!")
                return
            
            print(f"Signals generated: {len(self.signals_df)} rows")
            print("Signal columns:", self.signals_df.columns.tolist())
            
            # Stage 3: Backtest Execution
            print("\nStage 3: Running Backtest")
            backtest_results = self.run_backtest(self.signals_df)
            
            if backtest_results is None:
                print("WARNING: Backtest did not produce results!")
            
            # Stage 4: Strategy Comparison
            print("\nStage 4: Comparing Strategies")
            comparison_results = self.compare_with_original_strategy()
            
            print("\n===== ML BACKTESTING COMPLETE =====")
            
            return {
                'features': features_df,
                'signals': self.signals_df,
                'backtest': backtest_results,
                'comparison': comparison_results
            }
        
        except Exception as e:
            print(f"Critical Error in ML Backtesting Pipeline: {e}")
            import traceback
            traceback.print_exc()
            return None

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='ML Backtest for Futures Trading')
    
    parser.add_argument('--model', type=str, 
                        help='Path to trained model checkpoint')
    
    parser.add_argument('--data', type=str,
                        help='Path to futures data directory')
    
    parser.add_argument('--session', type=str, default='RTH',
                        choices=['RTH', 'ETH', ''],
                        help='Trading session type (RTH or ETH, empty for all)')
    
    parser.add_argument('--contract', type=str, default='ES',
                        help='Contract prefix to filter (e.g., ES)')
    
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Prediction threshold for signal generation')
    
    parser.add_argument('--confidence', type=float, default=70,
                        help='Minimum confidence threshold for signals')
    
    parser.add_argument('--output', type=str,
                        help='Directory for saving results')
    
    return parser.parse_args()

def main():
    """Main execution function."""
    args = parse_args()
    
    # Set session_type to None if empty string
    session_type = args.session if args.session else None
    
    # Initialize backtest with command line arguments
    backtest = MLBacktester(
        model_path=args.model,
        data_path=args.data,
        session_type=session_type,
        contract_filter=args.contract,
        prediction_threshold=args.threshold,
        signal_confidence_threshold=args.confidence,
        output_dir=args.output or ML_BACKTEST_DIR
    )
    
    # Run the backtesting pipeline
    backtest.run()

if __name__ == "__main__":
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Check for PyTorch
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
    except ImportError:
        print("PyTorch not available")
    
    main()
