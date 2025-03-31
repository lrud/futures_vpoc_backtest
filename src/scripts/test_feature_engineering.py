"""
Unit tests for the feature engineering module.
"""

import unittest
import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.ml.feature_engineering import prepare_features_and_labels, _extract_session_features

class TestFeatureEngineering(unittest.TestCase):
    """Tests for feature engineering module."""

    def setUp(self):
        """Set up test data with all required columns."""
        # Create sample futures data with datetime dates
        dates = pd.date_range(start='2023-01-01', end='2023-01-05')
        times = ['09:30:00', '10:00:00', '10:30:00', '11:00:00']

        rows = []
        for date in dates:
            for time in times:
                base_price = 4000 + np.random.normal(0, 20)
                row = {
                    'date': date.date(),  # Convert to datetime.date
                    'time': time,
                    'open': base_price,
                    'high': base_price + abs(np.random.normal(0, 5)),
                    'low': base_price - abs(np.random.normal(0, 5)),
                    'close': base_price + np.random.normal(0, 2),
                    'volume': max(100, np.random.normal(500, 200))
                }
                rows.append(row)

        self.test_data = pd.DataFrame(rows)

        # Create mock VIX data with matching datetime.date format
        self.vix_data = pd.DataFrame({
            'date': [d.date() for d in dates],  # Ensure same type as futures data
            'vix_close': np.random.uniform(15, 30, size=len(dates)),
            'vix_1d_change': np.random.uniform(-5, 5, size=len(dates))
        })

    def test_extract_session_features(self):
        """Test extraction of session features."""
        features_df = _extract_session_features(self.test_data)
        self.assertIsInstance(features_df, pd.DataFrame)
        self.assertEqual(len(features_df), 5)
    def test_prepare_features_and_labels(self):
        """Test feature preparation without VIX."""
        X, y, feature_cols, scaler = prepare_features_and_labels(
            self.test_data,
            use_feature_selection=False
        )
        self.assertEqual(X.shape[0], 4)  # 5 days - 1 due to target shift
    def test_prepare_features_with_vix(self):
        """Test feature preparation with VIX data."""
        X, y, feature_cols, scaler = prepare_features_and_labels(
            self.test_data,
            vix_data=self.vix_data,
            use_feature_selection=False
        )
        self.assertIn('vix_close', feature_cols)
        self.assertIn('vix_1d_change', feature_cols)

    def test_prepare_features_with_vix_and_selection(self):
        """Test VIX features survive feature selection."""
        X, y, feature_cols, scaler = prepare_features_and_labels(
            self.test_data,
            vix_data=self.vix_data,
            use_feature_selection=True,
            max_features=5
        )
        self.assertTrue(any(col in feature_cols for col in ['vix_close', 'vix_1d_change']))
if __name__ == '__main__':
    unittest.main()