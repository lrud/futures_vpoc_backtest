"""
Unit tests for ML backtest integration.
"""

import unittest
import os
import pandas as pd
import numpy as np
import torch
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.ml.backtest_integration import MLBacktestIntegrator
from src.ml.model import AMDOptimizedFuturesModel

class TestMLBacktestIntegration(unittest.TestCase):
    """Test cases for ML backtest integration."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test outputs
        self.test_dir = os.path.join(os.path.dirname(__file__), "test_output")
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Create a mock model path
        self.mock_model_path = os.path.join(self.test_dir, "mock_model.pt")
        
        # Create a simple test model and save it
        input_dim = 10
        model = AMDOptimizedFuturesModel(input_dim=input_dim)
        model.feature_columns = [f'feature_{i}' for i in range(input_dim)]
        
        # Save model checkpoint
        torch.save({
            'model_state_dict': model.state_dict(),
            'feature_columns': model.feature_columns,
            'input_dim': input_dim
        }, self.mock_model_path)
        
        # Create test data
        self.create_test_data()
    
    def create_test_data(self):
        """Create test data for backtesting."""
        # Create simple OHLCV data
        dates = pd.date_range('2023-01-01', '2023-01-10', freq='H')
        self.test_data = pd.DataFrame({
            'open': np.random.normal(100, 5, len(dates)),
            'high': np.random.normal(105, 5, len(dates)),
            'low': np.random.normal(95, 5, len(dates)),
            'close': np.random.normal(100, 5, len(dates)),
            'volume': np.random.randint(1000, 5000, len(dates))
        }, index=dates)
        
        # Ensure high is actually highest
        self.test_data['high'] = self.test_data[['open', 'high', 'close']].max(axis=1) + 1
        
        # Ensure low is actually lowest
        self.test_data['low'] = self.test_data[['open', 'low', 'close']].min(axis=1) - 1
        
        # Add date column
        self.test_data['date'] = self.test_data.index.date
        
        # Add contract and session columns
        self.test_data['contract'] = 'ES'
        self.test_data['session'] = 'RTH'
    
    def tearDown(self):
        """Clean up after tests."""
        # Delete mock model file if it exists
        if os.path.exists(self.mock_model_path):
            os.remove(self.mock_model_path)
        
        # Option to remove test directory
        # import shutil
        # shutil.rmtree(self.test_dir, ignore_errors=True)
    
    @patch('src.ml.feature_engineering.FeatureEngineer.prepare_features')
    @patch('src.ml.backtest_integration.SignalGenerator.generate_ml_signals')
    @patch('src.ml.backtest_integration.BacktestEngine')
    def test_run_backtest_pipeline(self, mock_backtest, mock_signal_gen, mock_feature_eng):
        """Test the complete backtest pipeline."""
        # Setup mocks
        mock_features = pd.DataFrame({
            'date': pd.date_range('2023-01-01', '2023-01-10'),
            'vpoc': np.random.normal(100, 5, 10),
            'value_area_low': np.random.normal(95, 5, 10),
            'value_area_high': np.random.normal(105, 5, 10),
            'feature_0': np.random.normal(0, 1, 10)
        })
        mock_feature_eng.return_value = mock_features
        
        mock_signals = pd.DataFrame({
            'date': pd.date_range('2023-01-01', '2023-01-05'),
            'signal': ['BUY', 'SELL', 'BUY', 'SELL', 'BUY'],
            'price': np.random.normal(100, 5, 5),
            'stop_loss': np.random.normal(95, 5, 5),
            'target': np.random.normal(105, 5, 5),
            'position_size': [1.0] * 5,
            'confidence': [80] * 5,
            'reason': ['ML_BUY'] * 5
        })
        mock_signal_gen.return_value = mock_signals
        
        # Mock backtest engine
        mock_backtest_instance = mock_backtest.return_value
        mock_backtest_instance.performance_metrics = {
            'total_trades': 5,
            'win_rate': 60.0,
            'total_profit': 1000.0,
            'sharpe_ratio': 1.5,
            'max_drawdown': -5.0
        }
        mock_backtest_instance.equity_curve = [10000, 10200, 10400, 10300, 10500, 11000]
        
        # Create integrator and run backtest
        integrator = MLBacktestIntegrator(
            model_path=self.mock_model_path,
            output_dir=self.test_dir
        )
        
        results = integrator.run_backtest(self.test_data)
        
        # Verify results
        self.assertIn('features', results)
        self.assertIn('signals', results)
        self.assertIn('backtest', results)
        self.assertIn('performance', results)
        
        # Verify interactions with mocked components
        mock_feature_eng.assert_called_once()
        mock_signal_gen.assert_called_once()
        mock_backtest_instance.run_backtest.assert_called_once()
        
        # Verify performance metrics
        perf = results['performance']
        self.assertEqual(perf['total_trades'], 5)
        self.assertEqual(perf['win_rate'], 60.0)
        self.assertEqual(perf['total_profit'], 1000.0)
    
    def test_load_invalid_model(self):
        """Test loading an invalid model."""
        # Create invalid model path
        invalid_model_path = os.path.join(self.test_dir, "invalid_model.pt")
        
        # Save invalid model format
        torch.save({
            'wrong_key': 'wrong_value'
        }, invalid_model_path)
        
        # Create integrator with invalid model
        integrator = MLBacktestIntegrator(
            model_path=invalid_model_path,
            output_dir=self.test_dir
        )
        
        # Model should be None
        self.assertIsNone(integrator.model)
        
        # Clean up
        if os.path.exists(invalid_model_path):
            os.remove(invalid_model_path)
    
    @patch('src.ml.backtest_integration.MLBacktestIntegrator._save_backtest_results')
    @patch('src.ml.feature_engineering.FeatureEngineer.prepare_features')
    def test_empty_feature_generation(self, mock_feature_eng, mock_save):
        """Test handling empty feature generation."""
        # Mock empty feature DataFrame
        mock_feature_eng.return_value = pd.DataFrame()
        
        # Create integrator
        integrator = MLBacktestIntegrator(
            model_path=self.mock_model_path,
            output_dir=self.test_dir
        )
        
        # Run backtest with empty features
        results = integrator.run_backtest(self.test_data)
        
        # Check error handling
        self.assertIn('error', results)
        self.assertEqual(results['error'], 'Feature generation failed')
        
        # Verify no saving occurred
        mock_save.assert_not_called()
    
    @patch('src.ml.feature_engineering.FeatureEngineer.prepare_features')
    @patch('src.ml.backtest_integration.SignalGenerator.generate_ml_signals')
    def test_no_signals_generated(self, mock_signal_gen, mock_feature_eng):
        """Test handling no signals generated."""
        # Mock features but empty signals
        mock_features = pd.DataFrame({
            'date': pd.date_range('2023-01-01', '2023-01-10'),
            'vpoc': np.random.normal(100, 5, 10)
        })
        mock_feature_eng.return_value = mock_features
        mock_signal_gen.return_value = pd.DataFrame()  # Empty signals
        
        # Create integrator
        integrator = MLBacktestIntegrator(
            model_path=self.mock_model_path,
            output_dir=self.test_dir
        )
        
        # Run backtest with empty signals
        results = integrator.run_backtest(self.test_data)
        
        # Check warning handling
        self.assertIn('error', results)
        self.assertEqual(results['error'], 'No signals generated')
        self.assertIn('features', results)
        self.assertTrue(isinstance(results['features'], pd.DataFrame))
    
    @patch('src.ml.backtest_integration.plt')
    def test_strategy_comparison(self, mock_plt):
        """Test strategy comparison functionality."""
        # Create mock backtest engines
        mock_ml_backtest = MagicMock()
        mock_baseline_backtest = MagicMock()
        
        # Set up equity curves
        mock_ml_backtest.equity_curve = [10000, 10200, 10400, 10300, 10600]
        mock_baseline_backtest.equity_curve = [10000, 10100, 10200, 10150, 10300]
        
        # Create mock results
        ml_results = {
            "backtest": mock_ml_backtest,
            "performance": {
                "total_trades": 10,
                "win_rate": 70.0,
                "total_profit": 2000.0,
                "sharpe_ratio": 1.8,
                "max_drawdown": -3.0
            }
        }
        
        baseline_results = {
            "backtest": mock_baseline_backtest,
            "performance": {
                "total_trades": 20,
                "win_rate": 60.0,
                "total_profit": 1000.0,
                "sharpe_ratio": 1.2,
                "max_drawdown": -5.0
            }
        }
        
        # Create integrator
        integrator = MLBacktestIntegrator(
            model_path=self.mock_model_path,
            output_dir=self.test_dir
        )
        
        # Perform comparison
        comparison = integrator.compare_strategies(ml_results, baseline_results)
        
        # Verify comparison metrics
        self.assertEqual(comparison['ml_trade_count'], 10)
        self.assertEqual(comparison['baseline_trade_count'], 20)
        self.assertEqual(comparison['ml_win_rate'], 70.0)
        self.assertEqual(comparison['baseline_win_rate'], 60.0)
        self.assertAlmostEqual(comparison['win_rate_change'], (70.0/60.0 - 1) * 100, places=4)
        
        # Verify plot was created
        mock_plt.figure.assert_called_once()
        mock_plt.savefig.assert_called_once()


if __name__ == '__main__':
    unittest.main()
