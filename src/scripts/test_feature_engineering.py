"""
Unit tests for the feature engineering module.
"""

import unittest
import os
import sys
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from sklearn.preprocessing import StandardScaler

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.ml.feature_engineering import prepare_features_and_labels, _extract_session_features


class TestFeatureEngineering(unittest.TestCase):
    """Tests for feature engineering module."""
    
    def setUp(self):
        """Set up test data with all required columns."""
        # Create sample data spanning multiple days
        dates = pd.date_range(start='2023-01-01', end='2023-01-05')
        times = ['09:30:00', '10:00:00', '10:30:00', '11:00:00']
        
        rows = []
        for date in dates:
            for time in times:
                base_price = 4000 + np.random.normal(0, 20)
                row = {
                    'date': date.strftime('%Y-%m-%d'),
                    'time': time,
                    'open': base_price,
                    'high': base_price + abs(np.random.normal(0, 5)),
                    'low': base_price - abs(np.random.normal(0, 5)),
                    'close': base_price + np.random.normal(0, 2),
                    'volume': max(100, np.random.normal(500, 200))
                }
                rows.append(row)
                
        self.test_data = pd.DataFrame(rows)
    
    def test_extract_session_features(self):
        """Test extraction of session features."""
        features_df = _extract_session_features(self.test_data)
        
        # Verify the result
        self.assertIsInstance(features_df, pd.DataFrame)
        self.assertEqual(len(features_df), 5)  # Should have one row per day
        
        # Check required columns exist
        required_cols = ['date', 'session_open', 'session_close', 
                         'session_high', 'session_low', 'total_volume',
                         'close_change_pct', 'price_range']
        for col in required_cols:
            self.assertIn(col, features_df.columns)
    
    def test_prepare_features_and_labels(self):
        """Test preparation of features and labels."""
        X, y, feature_cols, scaler = prepare_features_and_labels(
            self.test_data, 
            use_feature_selection=False
        )
        
        # Verify output shapes
        self.assertEqual(X.shape[0], 4)  # 5 days - 1 due to target shift
        self.assertEqual(X.shape[1], len(feature_cols))
        self.assertEqual(y.shape[0], 4)
        
        # Verify scaler
        self.assertIsInstance(scaler, StandardScaler)
    
    def test_prepare_features_with_selection(self):
        """Test feature selection."""
        X, y, feature_cols, scaler = prepare_features_and_labels(
            self.test_data, 
            use_feature_selection=True,
            max_features=3
        )
        
        # Verify feature selection reduced the number of features
        self.assertLessEqual(len(feature_cols), 3)
        self.assertEqual(X.shape[1], len(feature_cols))


if __name__ == '__main__':
    unittest.main()