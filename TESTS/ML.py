"""
Robust Machine Learning Framework for Futures Strategy
=====================================================
This module implements a comprehensive framework for enhancing trading strategy
robustness through advanced validation techniques, machine learning integration,
and forward testing capabilities.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
import joblib
import sqlite3
import os

class RobustStrategyFramework:
    """
    Framework for enhancing trading strategies with robust validation and ML integration.
    """
    
    def __init__(self, base_dir=None):
        """
        Initialize the framework with configuration settings.
        
        Parameters:
        -----------
        base_dir : str, optional
            Base directory for data storage
        """
        self.base_dir = base_dir or os.path.join(os.getcwd(), 'FUTURES_FRAMEWORK')
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Create subdirectories
        self.models_dir = os.path.join(self.base_dir, 'MODELS')
        self.validation_dir = os.path.join(self.base_dir, 'VALIDATION')
        self.forward_test_dir = os.path.join(self.base_dir, 'FORWARD_TEST')
        
        for directory in [self.models_dir, self.validation_dir, self.forward_test_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Configuration
        self.default_config = {
            'validation': {
                'cv_folds': 5,
                'train_size': 0.7,
                'time_series_gap': 20,
                'permutation_tests': 100
            },
            'ml': {
                'lookback_periods': [5, 10, 20, 50],
                'test_size': 0.3,
                'feature_selection_threshold': 0.01
            },
            'monte_carlo': {
                'iterations': 1000,
                'confidence_level': 0.95
            }
        }
        
        # Initialize database
        self.setup_database()
    
    def setup_database(self):
        """Setup SQLite database for metrics tracking."""
        db_path = os.path.join(self.base_dir, 'strategy_metrics.db')
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        
        # Create validation results table
        c.execute('''
        CREATE TABLE IF NOT EXISTS validation_results (
            id INTEGER PRIMARY KEY,
            run_date TEXT,
            validation_type TEXT,
            train_period_start TEXT,
            train_period_end TEXT,
            test_period_start TEXT,
            test_period_end TEXT,
            win_rate REAL,
            profit_factor REAL,
            sharpe_ratio REAL,
            max_drawdown REAL,
            parameters TEXT
        )
        ''')
        
        # Create feature importance table
        c.execute('''
        CREATE TABLE IF NOT EXISTS feature_importance (
            id INTEGER PRIMARY KEY,
            run_date TEXT,
            feature_name TEXT,
            importance_score REAL,
            model_type TEXT
        )
        ''')
        
        # Create forward test results table
        c.execute('''
        CREATE TABLE IF NOT EXISTS forward_test_results (
            id INTEGER PRIMARY KEY,
            date TEXT,
            equity REAL,
            daily_pnl REAL,
            trade_count INTEGER,
            win_rate REAL,
            model_version TEXT,
            parameters TEXT
        )
        ''')
        
        conn.commit()
        conn.close()
        
    def walk_forward_validation(self, data, strategy_fn, param_grid=None, window_size=180, step_size=30):
        """
        Perform walk-forward validation to test strategy robustness.
        
        Parameters:
        -----------
        data : DataFrame
            Historical data for testing
        strategy_fn : function
            Function that implements the strategy logic
        param_grid : dict, optional
            Grid of parameters to optimize
        window_size : int
            Size of sliding window in days
        step_size : int
            Step size for sliding window
            
        Returns:
        --------
        DataFrame
            Results of walk-forward validation
        """
        print(f"Performing walk-forward validation with {window_size}-day windows")
        
        # Ensure data is sorted by date
        data = data.sort_values('date')
        dates = data['date'].unique()
        results = []
        
        # Create windows
        for i in range(0, len(dates) - window_size, step_size):
            if i + window_size >= len(dates):
                break
                
            train_start_idx = i
            train_end_idx = i + int(window_size * 0.7)  # 70% for training
            test_start_idx = train_end_idx + 1
            test_end_idx = min(i + window_size, len(dates) - 1)
            
            train_start = dates[train_start_idx]
            train_end = dates[train_end_idx]
            test_start = dates[test_start_idx]
            test_end = dates[test_end_idx]
            
            print(f"\nWindow {i//step_size + 1}: {train_start} to {test_end}")
            print(f"  Training: {train_start} to {train_end}")
            print(f"  Testing:  {test_start} to {test_end}")
            
            # Split data
            train_data = data[(data['date'] >= train_start) & (data['date'] <= train_end)]
            test_data = data[(data['date'] >= test_start) & (data['date'] <= test_end)]
            
            # Find optimal parameters if param_grid provided
            if param_grid:
                optimal_params = self.optimize_parameters(train_data, strategy_fn, param_grid)
                print(f"  Optimal parameters: {optimal_params}")
            else:
                optimal_params = {}
            
            # Run strategy on test set
            test_results = strategy_fn(test_data, **optimal_params)
            
            # Calculate performance metrics
            performance = self.calculate_performance_metrics(test_results)
            
            # Store results
            window_result = {
                'window': i//step_size + 1,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'parameters': optimal_params,
                **performance
            }
            
            results.append(window_result)
            print(f"  Test performance: Win Rate={performance['win_rate']:.2f}%, "
                  f"Profit Factor={performance['profit_factor']:.2f}, "
                  f"Sharpe={performance['sharpe_ratio']:.2f}")
            
            # Save to database
            self.save_validation_results('walk_forward', window_result)
        
        return pd.DataFrame(results)
    
    def optimize_parameters(self, data, strategy_fn, param_grid):
        """
        Find optimal strategy parameters using grid search.
        
        Parameters:
        -----------
        data : DataFrame
            Training data
        strategy_fn : function
            Strategy function
        param_grid : dict
            Grid of parameter values to test
            
        Returns:
        --------
        dict
            Optimal parameter values
        """
        from itertools import product
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))
        
        best_sharpe = -np.inf
        best_params = {}
        
        for i, combo in enumerate(combinations):
            params = dict(zip(param_names, combo))
            
            # Run strategy with these parameters
            results = strategy_fn(data, **params)
            
            # Calculate Sharpe ratio
            if not results.empty:
                performance = self.calculate_performance_metrics(results)
                sharpe = performance['sharpe_ratio']
                
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = params
        
        return best_params
    
    def calculate_performance_metrics(self, results):
        """
        Calculate standard trading performance metrics.
        
        Parameters:
        -----------
        results : DataFrame
            Trading results
            
        Returns:
        --------
        dict
            Performance metrics
        """
        if results.empty:
            return {
                'win_rate': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'total_return': 0
            }
        
        # Calculate basic metrics
        wins = results[results['profit'] > 0]
        losses = results[results['profit'] <= 0]
        
        win_rate = len(wins) / len(results) * 100 if len(results) > 0 else 0
        
        profit_factor = abs(wins['profit'].sum() / losses['profit'].sum()) if len(losses) > 0 and losses['profit'].sum() != 0 else 0
        
        # Calculate Sharpe ratio
        returns = results['profit'] / results['entry_price'] / results['position_size']
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
        
        # Calculate drawdown
        cumulative_returns = results['profit'].cumsum()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / (running_max + 1e-10) * 100
        max_drawdown = drawdown.min()
        
        total_return = results['profit'].sum()
        
        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_return': total_return
        }
    
    def save_validation_results(self, validation_type, results):
        """Save validation results to database."""
        conn = sqlite3.connect(os.path.join(self.base_dir, 'strategy_metrics.db'))
        c = conn.cursor()
        
        # Convert parameters to string for storage
        parameters = str(results.get('parameters', {}))
        
        c.execute('''
        INSERT INTO validation_results 
        (run_date, validation_type, train_period_start, train_period_end, 
        test_period_start, test_period_end, win_rate, profit_factor, 
        sharpe_ratio, max_drawdown, parameters)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            validation_type,
            str(results.get('train_start')),
            str(results.get('train_end')),
            str(results.get('test_start')),
            str(results.get('test_end')),
            results.get('win_rate', 0),
            results.get('profit_factor', 0),
            results.get('sharpe_ratio', 0),
            results.get('max_drawdown', 0),
            parameters
        ))
        
        conn.commit()
        conn.close()
    
    def monte_carlo_simulation(self, trade_results, initial_capital=100000, iterations=1000):
        """
        Perform Monte Carlo simulation to assess strategy robustness.
        
        Parameters:
        -----------
        trade_results : DataFrame
            DataFrame of trade results
        initial_capital : float
            Starting capital
        iterations : int
            Number of Monte Carlo iterations
            
        Returns:
        --------
        dict
            Simulation results
        """
        print(f"Running Monte Carlo simulation with {iterations} iterations")
        
        if trade_results.empty:
            print("No trade results to analyze")
            return None
        
        # Extract trade returns as percentage
        returns = trade_results['profit'] / trade_results['entry_price'] / trade_results['position_size']
        
        # Initialize simulation results
        simulation_results = []
        
        for i in range(iterations):
            # Generate random sequence of trades (with replacement)
            sampled_returns = np.random.choice(returns, size=len(returns), replace=True)
            
            # Calculate equity curve
            equity = [initial_capital]
            for r in sampled_returns:
                equity.append(equity[-1] * (1 + r))
            
            # Calculate metrics for this run
            final_equity = equity[-1]
            pct_return = (final_equity - initial_capital) / initial_capital * 100
            peak = max(equity)
            drawdowns = [(equity[i] - max(equity[:i+1])) / max(equity[:i+1]) * 100 for i in range(len(equity))]
            max_drawdown = min(drawdowns)
            
            # Store results
            simulation_results.append({
                'final_equity': final_equity,
                'return_pct': pct_return,
                'max_drawdown': max_drawdown,
                'equity_curve': equity
            })
            
            if (i+1) % 100 == 0:
                print(f"  Completed {i+1} iterations")
        
        # Analyze results
        final_equities = [r['final_equity'] for r in simulation_results]
        returns_pct = [r['return_pct'] for r in simulation_results]
        drawdowns = [r['max_drawdown'] for r in simulation_results]
        
        # Calculate confidence intervals
        ci_level = 0.95
        lower_ci_idx = int((1 - ci_level) / 2 * iterations)
        upper_ci_idx = int((1 - (1 - ci_level) / 2) * iterations)
        
        sorted_returns = sorted(returns_pct)
        sorted_drawdowns = sorted(drawdowns)
        
        results = {
            'mean_return': np.mean(returns_pct),
            'median_return': np.median(returns_pct),
            'mean_drawdown': np.mean(drawdowns),
            'median_drawdown': np.median(drawdowns),
            'worst_drawdown': min(drawdowns),
            'best_return': max(returns_pct),
            'worst_return': min(returns_pct),
            'return_ci_lower': sorted_returns[lower_ci_idx],
            'return_ci_upper': sorted_returns[upper_ci_idx],
            'drawdown_ci_lower': sorted_drawdowns[upper_ci_idx],  # Note: Order is reversed for drawdowns
            'drawdown_ci_upper': sorted_drawdowns[lower_ci_idx],
            'probability_positive': sum(1 for r in returns_pct if r > 0) / iterations * 100
        }
        
        # Save visualization
        self._plot_monte_carlo_results(simulation_results, initial_capital, ci_level)
        
        return results, simulation_results
    
    def _plot_monte_carlo_results(self, simulation_results, initial_capital, ci_level=0.95):
        """Plot Monte Carlo simulation results."""
        equity_curves = [r['equity_curve'] for r in simulation_results]
        
        # Find min length (some might differ slightly)
        min_length = min(len(curve) for curve in equity_curves)
        equity_curves = [curve[:min_length] for curve in equity_curves]
        
        # Convert to DataFrame
        curves_df = pd.DataFrame(equity_curves).T  # Transpose to get trades as columns
        
        # Calculate percentiles at each point
        lower_ci = curves_df.quantile(0.5 - ci_level/2, axis=1)
        upper_ci = curves_df.quantile(0.5 + ci_level/2, axis=1)
        median_curve = curves_df.quantile(0.5, axis=1)
        
        plt.figure(figsize=(12, 8))
        
        # Plot sample of curves
        sample_size = min(50, len(equity_curves))
        for i in np.random.choice(curves_df.columns, sample_size, replace=False):
            plt.plot(curves_df[i], color='skyblue', alpha=0.1)
        
        # Plot confidence intervals and median
        plt.plot(median_curve, color='blue', linewidth=2, label='Median')
        plt.plot(lower_ci, color='red', linewidth=2, label=f'Lower {ci_level*100}% CI')
        plt.plot(upper_ci, color='green', linewidth=2, label=f'Upper {ci_level*100}% CI')
        
        # Add initial capital line
        plt.axhline(y=initial_capital, color='black', linestyle='--', label='Initial Capital')
        
        plt.title('Monte Carlo Simulation of Strategy Equity Curves')
        plt.xlabel('Trade Number')
        plt.ylabel('Account Equity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save figure
        output_path = os.path.join(self.validation_dir, f'monte_carlo_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Monte Carlo visualization saved to {output_path}")
    
    def create_ml_features(self, df, lookback_periods=None):
        """
        Create machine learning features from VPOC and price data.
        
        Parameters:
        -----------
        df : DataFrame
            Combined price and VPOC data
        lookback_periods : list, optional
            List of lookback periods for feature creation
            
        Returns:
        --------
        DataFrame
            DataFrame with ML features
        """
        lookback_periods = lookback_periods or self.default_config['ml']['lookback_periods']
        print(f"Creating ML features with lookback periods: {lookback_periods}")
        
        # Make a working copy of the dataframe
        features = df.copy()
        
        # Create basic VPOC features if they exist
        if 'vpoc' in features.columns:
            print("Creating VPOC-based features")
            for period in lookback_periods:
                # VPOC slope and momentum
                features[f'vpoc_slope_{period}d'] = features['vpoc'].diff(period) / period
                features[f'vpoc_std_{period}d'] = features['vpoc'].rolling(period).std()
                
                # Normalized distance features
                features[f'vpoc_zscore_{period}d'] = (
                    features['vpoc'] - features['vpoc'].rolling(period).mean()
                ) / features['vpoc'].rolling(period).std()
                
                # Value area features if available
                if all(col in features.columns for col in ['value_area_low', 'value_area_high']):
                    features[f'va_width_{period}d_ma'] = (
                        features['value_area_high'] - features['value_area_low']
                    ).rolling(period).mean()
                    
                    # VA expansion/contraction
                    current_width = features['value_area_high'] - features['value_area_low']
                    features[f'va_width_change_{period}d'] = (
                        current_width / features[f'va_width_{period}d_ma']
                    )
        
        # Create price-based features
        print("Creating price-based features")
        for period in lookback_periods:
            # Price momentum
            if 'close' in features.columns:
                features[f'price_mom_{period}d'] = (
                    features['close'] / features['close'].shift(period) - 1
                ) * 100
                
                # Volatility features
                if all(col in features.columns for col in ['high', 'low']):
                    # Volatility (ATR-like)
                    features[f'vol_{period}d'] = (
                        (features['high'] - features['low']) / features['close']
                    ).rolling(period).mean() * 100
                    
                    # Normalized volatility
                    features[f'vol_zscore_{period}d'] = (
                        features[f'vol_{period}d'] - features[f'vol_{period}d'].rolling(period*2).mean()
                    ) / features[f'vol_{period}d'].rolling(period*2).std()
        
        # VPOC and price relationship features
        if all(col in features.columns for col in ['close', 'vpoc']):
            print("Creating price-to-VPOC relationship features")
            for period in lookback_periods:
                # Price relative to VPOC
                features[f'price_to_vpoc_{period}d'] = (
                    (features['close'] - features['vpoc']) / features['vpoc']
                ) * 100
                
                # Smoothed price to VPOC
                features[f'price_to_vpoc_ma_{period}d'] = (
                    features[f'price_to_vpoc_{period}d'].rolling(period).mean()
                )
                
                # Divergence between price and VPOC
                features[f'price_vpoc_divergence_{period}d'] = (
                    features['close'].pct_change(period) - 
                    features['vpoc'].pct_change(period)
                ) * 100
        
        # Volume and value area features
        if 'volume' in features.columns:
            print("Creating volume-based features")
            for period in lookback_periods:
                # Relative volume
                features[f'rel_vol_{period}d'] = (
                    features['volume'] / features['volume'].rolling(period).mean()
                )
                
                # Volume trend
                features[f'vol_trend_{period}d'] = (
                    features['volume'].diff(period) / features['volume'].shift(period)
                ) * 100
        
        # Delta features if available
        if 'delta_15min_pctl' in features.columns:
            print("Creating delta-based features")
            for period in lookback_periods:
                features[f'delta_trend_{period}d'] = (
                    features['delta_15min_pctl'].rolling(period).mean()
                )
                
                # Oscillator
                features[f'delta_osc_{period}d'] = (
                    features['delta_15min_pctl'] - 
                    features['delta_15min_pctl'].rolling(period).mean()
                )
        
        # Fill or drop NaN values
        null_counts = features.isnull().sum()
        if null_counts.any():
            print(f"Features contain {null_counts.sum()} null values across {null_counts[null_counts > 0].shape[0]} columns")
            print("Dropping rows with NaN values")
            features = features.dropna()
        
        return features
    
    def train_ml_models(self, features, target_column, test_size=None, random_state=42):
        """
        Train machine learning models for strategy enhancement.
        
        Parameters:
        -----------
        features : DataFrame
            Feature DataFrame
        target_column : str
            Column name for prediction target
        test_size : float, optional
            Proportion of data to use for testing
        random_state : int
            Random seed for reproducibility
            
        Returns:
        --------
        tuple
            (trained_models, feature_importance, model_performance)
        """
        from sklearn.model_selection import train_test_split
        
        test_size = test_size or self.default_config['ml']['test_size']
        print(f"Training ML models with test_size={test_size}")
        
        if target_column not in features.columns:
            raise ValueError(f"Target column '{target_column}' not found in features")
        
        # Identify feature columns (exclude date and target)
        exclude_cols = [target_column]
        if 'date' in features.columns:
            exclude_cols.append('date')
        
        # Check for any non-numeric features
        non_numeric_cols = features.select_dtypes(exclude=['number']).columns.tolist()
        for col in non_numeric_cols:
            if col not in exclude_cols:
                exclude_cols.append(col)
                print(f"Excluding non-numeric column: {col}")
        
        # Select feature columns
        feature_cols = [col for col in features.columns if col not in exclude_cols]
        
        print(f"Using {len(feature_cols)} features for training")
        
        # Prepare features and target
        X = features[feature_cols].copy()
        y = features[target_column].copy()
        
        # Time-series aware train-test split
        if 'date' in features.columns:
            # Sort by date
            features_sorted = features.sort_values('date')
            X = features_sorted[feature_cols].copy()
            y = features_sorted[target_column].copy()
            
            # Use the last test_size portion for testing
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            train_dates = features_sorted['date'].iloc[:split_idx]
            test_dates = features_sorted['date'].iloc[split_idx:]
            
            print(f"Training period: {train_dates.min()} to {train_dates.max()}")
            print(f"Testing period: {test_dates.min()} to {test_dates.max()}")
        else:
            # Random split if no dates
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Save scaler for later use
        joblib.dump(scaler, os.path.join(self.models_dir, 'feature_scaler.joblib'))
        
        # Define models
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100, 
                max_depth=5, 
                random_state=random_state
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100, 
                max_depth=3, 
                random_state=random_state
            )
        }
        
        # Train and evaluate models
        trained_models = {}
        feature_importance = {}
        model_performance = {}
        
        for name, model in models.items():
            print(f"\nTraining {name} model...")
            
            # Train model
            model.fit(X_train_scaled, y_train)
            trained_models[name] = model
            
            # Save model
            joblib.dump(model, os.path.join(self.models_dir, f'{name}_model.joblib'))
            
            # Evaluate on test set
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            try:
                roc_auc = roc_auc_score(y_test, y_prob)
            except Exception as e:
                print(f"Error calculating ROC AUC: {e}")
                roc_auc = 0
            
            # Store performance 
            model_performance[name] = {
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'roc_auc': roc_auc
            }
            
            print(f"Test ROC AUC: {roc_auc:.4f}")
            print(classification_report(y_test, y_pred))
            
            # Extract feature importance
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_importance[name] = dict(zip(feature_cols, importances))
                
                # Save feature importances to database
                self._save_feature_importance(name, feature_importance[name])
                
                # Plot top features
                self._plot_feature_importance(name, feature_importance[name])
            
        return trained_models, feature_importance, model_performance
    
    def _save_feature_importance(self, model_name, feature_importance):
        """Save feature importance to database."""
        conn = sqlite3.connect(os.path.join(self.base_dir, 'strategy_metrics.db'))
        c = conn.cursor()
        
        run_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        for feature, importance in feature_importance.items():
            c.execute('''
            INSERT INTO feature_importance (run_date, feature_name, importance_score, model_type)
            VALUES (?, ?, ?, ?)
            ''', (run_date, feature, float(importance), model_name))
        
        conn.commit()
        conn.close()
    
    def _plot_feature_importance(self, model_name, feature_importance, top_n=20):
        """Plot feature importance."""
        # Sort features by importance
        sorted_features = dict(sorted(feature_importance.items(), 
                                      key=lambda x: x[1], reverse=True)[:top_n])
        
        plt.figure(figsize=(12, 8))
        plt.barh(list(sorted_features.keys()), list(sorted_features.values()))
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Top {top_n} Features - {model_name}')
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(self.models_dir, f'{model_name}_feature_importance.png')
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Feature importance plot saved to {output_path}")
    
    def enhance_signals_with_ml(self, signals, ml_models, features, confidence_threshold=0.65):
        """
        Enhance trading signals with ML predictions.
        
        Parameters:
        -----------
        signals : DataFrame
            Original strategy signals
        ml_models : dict
            Dictionary of trained ML models
        features : DataFrame
            Feature DataFrame
        confidence_threshold : float
            Minimum confidence level for signal acceptance
            
        Returns:
        --------
        DataFrame
            Enhanced signals
        """
        if signals.empty:
            print("No signals to enhance")
            return signals
        
        print(f"Enhancing {len(signals)} signals with ML predictions")
        
        # Load scaler if available
        scaler_path = os.path.join(self.models_dir, 'feature_scaler.joblib')
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
        else:
            print("Warning: Feature scaler not found. Using StandardScaler on current data.")
            scaler = StandardScaler()
            scaler.fit(features.select_dtypes(include=['number']).fillna(0))
        
        # Create a list to store enhanced signals
        enhanced_signals = []
        
        # Process each signal
        for _, signal in signals.iterrows():
            signal_date = signal['date']
            
            # Find features for this date
            date_features = features[features['date'] == signal_date]
            
            if not date_features.empty:
                # Extract feature values (excluding non-feature columns)
                exclude_cols = ['date']
                if 'signal_success' in date_features.columns:
                    exclude_cols.append('signal_success')
                
                feature_cols = [col for col in date_features.columns 
                               if col not in exclude_cols 
                               and date_features[col].dtype in ['float64', 'int64']]
                
                feature_values = date_features[feature_cols].values
                
                # Scale features
                scaled_features = scaler.transform(feature_values)
                
                # Get predictions from each model
                ml_scores = {}
                for name, model in ml_models.items():
                    try:
                        # Get prediction probability
                        prob = model.predict_proba(scaled_features)[0, 1]
                        ml_scores[f'{name}_score'] = prob
                    except Exception as e:
                        print(f"Error getting prediction from {name} model: {e}")
                        ml_scores[f'{name}_score'] = 0.5  # Default to neutral
                
                # Calculate ensemble score
                ensemble_score = sum(ml_scores.values()) / len(ml_scores) if ml_scores else 0.5
                
                # Apply ML filter based on signal direction
                signal_dict = signal.to_dict()
                
                # Determine if signal passes ML filter
                passes_filter = False
                if signal['signal'] == 'BUY':
                    passes_filter = ensemble_score >= confidence_threshold
                elif signal['signal'] == 'SELL':
                    passes_filter = ensemble_score <= (1 - confidence_threshold)
                
                # Add ML scores to signal
                signal_dict.update(ml_scores)
                signal_dict['ensemble_score'] = ensemble_score
                signal_dict['ml_validated'] = passes_filter
                
                # Only include signals that pass the filter
                if passes_filter:
                    enhanced_signals.append(signal_dict)
                    print(f"Signal for {signal_date} ({signal['signal']}) validated with score {ensemble_score:.4f}")
                else:
                    print(f"Signal for {signal_date} ({signal['signal']}) rejected with score {ensemble_score:.4f}")
            else:
                print(f"No features found for date {signal_date}")
        
        if enhanced_signals:
            return pd.DataFrame(enhanced_signals)
        else:
            print("No signals passed ML validation")
            return pd.DataFrame()
    
    def parameter_sensitivity(self, data, strategy_fn, baseline_params, param_ranges, metric='sharpe_ratio'):
        """
        Analyze strategy sensitivity to parameter changes.
        
        Parameters:
        -----------
        data : DataFrame
            Historical data for testing
        strategy_fn : function
            Strategy function
        baseline_params : dict
            Baseline strategy parameters
        param_ranges : dict
            Dictionary with parameter names and ranges to test
        metric : str
            Performance metric to track
            
        Returns:
        --------
        DataFrame
            Sensitivity analysis results
        """
        print(f"Performing parameter sensitivity analysis, tracking {metric}")
        
        # Get baseline performance
        baseline_results = strategy_fn(data, **baseline_params)
        baseline_performance = self.calculate_performance_metrics(baseline_results)
        
        print(f"Baseline {metric}: {baseline_performance[metric]:.4f}")
        
        sensitivity_results = []
        
        # Test each parameter individually
        for param_name, param_range in param_ranges.items():
            print(f"\nTesting sensitivity to {param_name}")
            
            for param_value in param_range:
                # Create modified parameters
                test_params = baseline_params.copy()
                test_params[param_name] = param_value
                
                # Run strategy with modified parameters
                test_results = strategy_fn(data, **test_params)
                
                # Calculate performance
                test_performance = self.calculate_performance_metrics(test_results)
                
                # Calculate sensitivity metrics
                performance_change = {
                    'win_rate_change': test_performance['win_rate'] - baseline_performance['win_rate'],
                    'profit_factor_change': test_performance['profit_factor'] - baseline_performance['profit_factor'],
                    'sharpe_change': test_performance['sharpe_ratio'] - baseline_performance['sharpe_ratio'],
                    'drawdown_change': test_performance['max_drawdown'] - baseline_performance['max_drawdown'],
                    'return_change': test_performance['total_return'] - baseline_performance['total_return']
                }
                
                # Store result
                sensitivity_results.append({
                    'parameter': param_name,
                    'baseline_value': baseline_params[param_name],
                    'test_value': param_value,
                    'absolute_change': test_performance[metric] - baseline_performance[metric],
                    'percent_change': (test_performance[metric] - baseline_performance[metric]) / 
                                   abs(baseline_performance[metric]) * 100 if baseline_performance[metric] != 0 else 0,
                    **performance_change
                })
                
                print(f"  {param_name}={param_value}: {metric}={test_performance[metric]:.4f} " +
                     f"(change: {performance_change[metric + '_change']:.4f})")
        
        # Create DataFrame and plot results
        sensitivity_df = pd.DataFrame(sensitivity_results)
        self._plot_sensitivity_analysis(sensitivity_df, metric)
        
        return sensitivity_df
    
    def _plot_sensitivity_analysis(self, sensitivity_df, metric):
        """Plot parameter sensitivity analysis results."""
        if sensitivity_df.empty:
            print("No sensitivity results to plot")
            return
        
        # Create one plot per parameter
        for param_name in sensitivity_df['parameter'].unique():
            param_data = sensitivity_df[sensitivity_df['parameter'] == param_name]
            
            plt.figure(figsize=(10, 6))
            plt.plot(param_data['test_value'], param_data[f'{metric}_change'], 'o-', linewidth=2)
            plt.axhline(y=0, color='red', linestyle='--')
            plt.title(f'Sensitivity of {metric} to {param_name}')
            plt.xlabel(param_name)
            plt.ylabel(f'Change in {metric}')
            plt.grid(True, alpha=0.3)
            
            # Save figure
            output_path = os.path.join(self.validation_dir, f'sensitivity_{param_name}_{metric}.png')
            plt.savefig(output_path, dpi=300)
            plt.close()
            print(f"Sensitivity plot for {param_name} saved to {output_path}")
    
    def simulate_market_conditions(self, base_data, strategy_fn, params=None):
        """
        Test strategy under various simulated market conditions.
        
        Parameters:
        -----------
        base_data : DataFrame
            Original market data
        strategy_fn : function
            Strategy function
        params : dict, optional
            Strategy parameters
            
        Returns:
        --------
        dict
            Test results under different conditions
        """
        params = params or {}
        print("Testing strategy under alternative market conditions")
        
        # Function to run backtest and get metrics
        def run_condition_test(data, condition_name):
            print(f"\nTesting {condition_name} conditions")
            results = strategy_fn(data, **params)
            if not results.empty:
                metrics = self.calculate_performance_metrics(results)
                metrics['trade_count'] = len(results)
                print(f"  {condition_name}: {len(results)} trades, " +
                     f"Win Rate={metrics['win_rate']:.2f}%, " +
                     f"Profit Factor={metrics['profit_factor']:.2f}, " +
                     f"Sharpe={metrics['sharpe_ratio']:.2f}")
                return metrics
            else:
                print(f"  {condition_name}: No trades generated")
                return None
        
        # Test base condition first
        conditions = {'base': base_data.copy()}
        
        # Create high volatility condition
        conditions['high_volatility'] = self._simulate_high_volatility(base_data.copy())
        
        # Create low volatility condition
        conditions['low_volatility'] = self._simulate_low_volatility(base_data.copy())
        
        # Create trending markets
        conditions['bull_market'] = self._simulate_bull_market(base_data.copy())
        conditions['bear_market'] = self._simulate_bear_market(base_data.copy())
        
        # Create choppy market
        conditions['choppy_market'] = self._simulate_choppy_market(base_data.copy())
        
        # Create flash crash
        conditions['flash_crash'] = self._simulate_flash_crash(base_data.copy())
        
        # Run tests on each condition
        results = {}
        for condition_name, data in conditions.items():
            results[condition_name] = run_condition_test(data, condition_name)
        
        # Plot comparative results
        self._plot_condition_comparison(results)
        
        return results
    
    def _simulate_high_volatility(self, data, factor=1.8):
        """
        Simulate high volatility market conditions.
        
        Parameters:
        -----------
        data : DataFrame
            Original market data
        factor : float
            Volatility increase factor
            
        Returns:
        --------
        DataFrame
            Modified data with high volatility
        """
        print(f"Creating high volatility scenario (factor={factor})")
        
        # Deep copy to avoid modifying original
        high_vol_data = data.copy()
        
        # Make sure we have required columns
        required_cols = ['high', 'low', 'open', 'close']
        if not all(col in high_vol_data.columns for col in required_cols):
            print("Warning: Missing required columns for volatility simulation")
            return data
        
        # Process daily price movements
        prev_close = None
        
        for i, (idx, row) in enumerate(high_vol_data.iterrows()):
            if i == 0:
                prev_close = row['close']
                continue
                
            # Calculate daily return
            daily_return = (row['close'] - prev_close) / prev_close
            
            # Exaggerate return to increase volatility
            exaggerated_return = daily_return * factor
            
            # New close price
            new_close = prev_close * (1 + exaggerated_return)
            
            # Adjust high and low proportionally
            original_range = row['high'] - row['low']
            new_range = original_range * factor
            midpoint = (row['high'] + row['low']) / 2
            
            high_vol_data.loc[idx, 'close'] = new_close
            high_vol_data.loc[idx, 'high'] = midpoint + new_range/2
            high_vol_data.loc[idx, 'low'] = midpoint - new_range/2
            
            # Store for next iteration
            prev_close = new_close
        
        return high_vol_data
    
    def _simulate_low_volatility(self, data, factor=0.5):
        """Simulate low volatility market conditions."""
        print(f"Creating low volatility scenario (factor={factor})")
        
        # Just use the high volatility function with a factor < 1
        return self._simulate_high_volatility(data, factor)
    
    def _simulate_bull_market(self, data, drift=0.0005):
        """Simulate bullish market conditions with upward drift."""
        print(f"Creating bull market scenario (drift={drift*100}% per day)")
        
        bull_data = data.copy()
        
        if not all(col in bull_data.columns for col in ['open', 'high', 'low', 'close']):
            print("Warning: Missing required columns for market simulation")
            return data
        
        # Apply compounding upward drift to prices
        for i in range(len(bull_data)):
            # Apply drift factor (compounding daily)
            drift_factor = (1 + drift) ** i
            
            # Apply to all price columns
            for col in ['open', 'high', 'low', 'close']:
                bull_data.iloc[i][col] *= drift_factor
        
        return bull_data
    
    def _simulate_bear_market(self, data, drift=-0.0005):
        """Simulate bearish market conditions with downward drift."""
        print(f"Creating bear market scenario (drift={drift*100}% per day)")
        
        # Use bull market function with negative drift
        return self._simulate_bull_market(data, drift)
    
    def _simulate_choppy_market(self, data, reversal_prob=0.6, magnitude=0.003):
        """Simulate choppy market with frequent reversals."""
        print(f"Creating choppy market scenario (reversal_prob={reversal_prob})")
        
        choppy_data = data.copy()
        
        if not all(col in choppy_data.columns for col in ['open', 'high', 'low', 'close']):
            print("Warning: Missing required columns for market simulation")
            return data
        
        prev_close = choppy_data['close'].iloc[0]
        trend_direction = 0  # 0 = no trend, 1 = up, -1 = down
        
        for i in range(1, len(choppy_data)):
            # Determine if we should reverse the previous day's movement
            prev_return = (choppy_data['close'].iloc[i-1] - choppy_data['close'].iloc[i-2]) / choppy_data['close'].iloc[i-2] if i > 1 else 0
            
            # Current return
            curr_return = (choppy_data['close'].iloc[i] - choppy_data['close'].iloc[i-1]) / choppy_data['close'].iloc[i-1]
            
            # Determine trend direction
            if prev_return > 0:
                trend_direction = 1
            elif prev_return < 0:
                trend_direction = -1
            
            # Check for reversal based on probability
            if np.random.random() < reversal_prob:
                # Reverse the movement
                new_return = -trend_direction * abs(np.random.normal(0, magnitude))
            else:
                # Continue the movement
                new_return = trend_direction * abs(np.random.normal(0, magnitude))
            
            # Apply the new return
            new_close = choppy_data['close'].iloc[i-1] * (1 + new_return)
            
            # Adjust high/low/open proportionally around the new close
            old_range = choppy_data['high'].iloc[i] - choppy_data['low'].iloc[i]
            mid_point = new_close
            
            choppy_data.loc[choppy_data.index[i], 'close'] = new_close
            choppy_data.loc[choppy_data.index[i], 'high'] = mid_point + old_range/2
            choppy_data.loc[choppy_data.index[i], 'low'] = mid_point - old_range/2
        
        return choppy_data
    
    def _simulate_flash_crash(self, data, crash_day_idx=None, crash_pct=-0.07):
        """Simulate a flash crash event."""
        print(f"Creating flash crash scenario (crash={crash_pct*100}%)")
        
        flash_crash_data = data.copy()
        
        if not all(col in flash_crash_data.columns for col in ['open', 'high', 'low', 'close']):
            print("Warning: Missing required columns for market simulation")
            return data
        
        # Pick a day for the flash crash if not specified
        if crash_day_idx is None:
            # Pick a day in the middle third of the data
            start_idx = len(data) // 3
            end_idx = start_idx * 2
            crash_day_idx = np.random.randint(start_idx, end_idx)
        
        # Apply the crash
        pre_crash_close = flash_crash_data['close'].iloc[crash_day_idx - 1]
        crash_day_close = pre_crash_close * (1 + crash_pct)
        
        # Update the crash day
        flash_crash_data.loc[flash_crash_data.index[crash_day_idx], 'close'] = crash_day_close
        flash_crash_data.loc[flash_crash_data.index[crash_day_idx], 'low'] = crash_day_close * 0.98  # Even lower intraday
        
        # Partial recovery over next few days
        recovery_days = 5
        recovery_pct = -crash_pct * 0.6  # Recover 60% of the crash
        
        for i in range(1, recovery_days + 1):
            if crash_day_idx + i < len(flash_crash_data):
                day_idx = crash_day_idx + i
                recovery_factor = (1 + recovery_pct / recovery_days) ** i
                
                # Apply recovery factor
                flash_crash_data.loc[flash_crash_data.index[day_idx], 'close'] *= recovery_factor
                flash_crash_data.loc[flash_crash_data.index[day_idx], 'high'] *= recovery_factor
                flash_crash_data.loc[flash_crash_data.index[day_idx], 'low'] *= recovery_factor
                flash_crash_data.loc[flash_crash_data.index[day_idx], 'open'] *= recovery_factor
        
        return flash_crash_data
    
    def _plot_condition_comparison(self, results):
        """Plot comparison of strategy performance under different market conditions."""
        if not results:
            print("No results to compare")
            return
        
        # Extract metrics for comparison
        metrics = ['win_rate', 'profit_factor', 'sharpe_ratio', 'max_drawdown', 'total_return']
        conditions = list(results.keys())
        
        comparison_data = {}
        for metric in metrics:
            metric_values = []
            for condition in conditions:
                if results[condition] is not None:
                    metric_values.append(results[condition].get(metric, 0))
                else:
                    metric_values.append(0)
            comparison_data[metric] = metric_values
        
        # Create a subplot for each metric
        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 15))
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            bars = ax.bar(conditions, comparison_data[metric])
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(comparison_data[metric]),
                       f'{height:.2f}', ha='center', va='bottom', rotation=0)
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(self.validation_dir, 'market_conditions_comparison.png')
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Market conditions comparison saved to {output_path}")
    
    def forward_test_simulation(self, data, strategy_fn, params=None, ml_models=None, period_days=20):
        """
        Simulate a forward testing environment.
        
        Parameters:
        -----------
        data : DataFrame
            Historical data for testing
        strategy_fn : function
            Strategy function
        params : dict, optional
            Strategy parameters
        ml_models : dict, optional
            ML models for signal enhancement
        period_days : int
            Number of days to include in each forward test period
            
        Returns:
        --------
        DataFrame
            Forward test results
        """
        params = params or {}
        print(f"Simulating forward testing with {period_days}-day periods")
        
        # Sort data by date
        if 'date' not in data.columns:
            print("Error: Data must contain a 'date' column for forward testing")
            return pd.DataFrame()
        
        data_sorted = data.sort_values('date')
        
        # Determine number of periods
        unique_dates = data_sorted['date'].unique()
        num_periods = len(unique_dates) // period_days
        
        print(f"Data spans {len(unique_dates)} days, running {num_periods} forward test periods")
        
        forward_results = []
        
        for period in range(num_periods):
            start_idx = period * period_days
            end_idx = start_idx + period_days - 1
            
            if end_idx >= len(unique_dates):
                break
                
            period_start = unique_dates[start_idx]
            period_end = unique_dates[end_idx]
            
            print(f"\nForward test period {period + 1}: {period_start} to {period_end}")
            
            # Get data for this period
            period_data = data_sorted[
                (data_sorted['date'] >= period_start) & 
                (data_sorted['date'] <= period_end)
            ]
            
            # Run strategy on this period
            signals = strategy_fn(period_data, **params)
            
            # Apply ML enhancement if available
            if ml_models and not signals.empty:
                # Create ML features
                ml_features = self.create_ml_features(period_data)
                signals = self.enhance_signals_with_ml(signals, ml_models, ml_features)
            
            # Calculate performance metrics
            if not signals.empty:
                metrics = self.calculate_performance_metrics(signals)
                metrics['period'] = period + 1
                metrics['start_date'] = period_start
                metrics['end_date'] = period_end
                metrics['trade_count'] = len(signals)
                
                forward_results.append(metrics)
                
                print(f"Period results: {len(signals)} trades, " +
                     f"Win Rate={metrics['win_rate']:.2f}%, " +
                     f"Profit Factor={metrics['profit_factor']:.2f}, " +
                     f"Return={metrics['total_return']:.2f}")
            else:
                print("No trades generated in this period")
        
        if forward_results:
            # Convert to DataFrame
            results_df = pd.DataFrame(forward_results)
            
            # Plot forward test results
            self._plot_forward_test_results(results_df)
            
            return results_df
        else:
            print("No forward test results generated")
            return pd.DataFrame()
    
    def _plot_forward_test_results(self, results):
        """Plot forward test results."""
        if results.empty:
            return
        
        # Equity curve
        equity = [100000]  # Starting with 100k
        returns = results['total_return'].values
        
        for r in returns:
            equity.append(equity[-1] + r)
        
        # Plot equity curve
        plt.figure(figsize=(12, 8))
        
        # Plot equity curve
        plt.subplot(3, 1, 1)
        plt.plot(range(len(equity)), equity, 'b-', linewidth=2)
        plt.title('Forward Test Equity Curve')
        plt.ylabel('Equity')
        plt.grid(True, alpha=0.3)
        
        # Plot win rate
        plt.subplot(3, 1, 2)
        plt.bar(results['period'], results['win_rate'], color='green', alpha=0.7)
        plt.title('Win Rate by Period')
        plt.ylabel('Win Rate (%)')
        plt.grid(True, alpha=0.3)
        
        # Plot trade count
        plt.subplot(3, 1, 3)
        plt.bar(results['period'], results['trade_count'], color='blue', alpha=0.7)
        plt.title('Trade Count by Period')
        plt.xlabel('Period')
        plt.ylabel('Number of Trades')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(self.forward_test_dir, 'forward_test_results.png')
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Forward test results saved to {output_path}")
    
    def automate_model_retraining(self, data, strategy_fn, retrain_interval=90, initial_train_days=180):
        """
        Simulate periodic model retraining for ML-enhanced strategy.
        
        Parameters:
        -----------
        data : DataFrame
            Historical data for testing
        strategy_fn : function
            Strategy function
        retrain_interval : int
            Number of days between model retraining
        initial_train_days : int
            Initial training period in days
            
        Returns:
        --------
        DataFrame
            Results with periodic model retraining
        """
        pass  # Implementation details would be added here

# Example usage functions (to be implemented in a separate module)
def example_futures_strategy(data, **params):
    """Example strategy implementation to demonstrate framework usage."""
    pass

def prepare_futures_data(data, **params):
    """Example data preparation function."""
    pass

def create_target_labels(data, **params):
    """Example function to create machine learning target labels."""
    pass