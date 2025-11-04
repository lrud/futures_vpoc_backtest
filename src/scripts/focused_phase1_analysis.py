#!/usr/bin/env python3
"""
Focused Phase 1 Feature Statistical Analysis

Quick analysis of Phase 1 features that were successfully created.
Tests RSI, MACD, Stochastic, ATR, Bollinger Bands, and Time features.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append('/workspace')

from src.utils.logging import get_logger

# Phase 1 Technical Analysis Libraries
import talib

logger = get_logger(__name__)

class FocusedPhase1Analyzer:
    """Quick analysis of Phase 1 features that were successfully created."""

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data = None
        self.target = None
        self.features = {}

    def load_and_prepare_data(self):
        """Load data with reduced memory footprint."""
        logger.info(f"📁 Loading data from {self.data_path}")

        # Load data
        self.data = pd.read_csv(self.data_path)

        # Sample for faster analysis (use 20% of data)
        if len(self.data) > 200000:
            logger.info(f"📊 Sampling 20% of data for faster analysis...")
            self.data = self.data.sample(n=200000, random_state=42)

        logger.info(f"✅ Loaded {len(self.data):,} rows for analysis")

        # Basic processing
        if 'VIX' in self.data.columns:
            self.data = self.data.rename(columns={'VIX': 'vix'})

        # Convert date column if exists
        if 'date' in self.data.columns:
            self.data['date'] = pd.to_datetime(self.data['date'])
            self.data = self.data.set_index('date')

        # Remove rows with missing essential data
        essential_cols = ['open', 'high', 'low', 'close', 'volume']
        initial_len = len(self.data)
        for col in essential_cols:
            if col in self.data.columns:
                self.data = self.data[self.data[col].notna()]

        removed_rows = initial_len - len(self.data)
        logger.info(f"📊 Removed {removed_rows:,} rows with missing essential data")

        # Forward fill missing values
        self.data = self.data.fillna(method='ffill').fillna(method='bfill')

        # Create target variable
        self.data['target'] = self.data['close'].pct_change().shift(-1)
        self.target = self.data['target'].dropna()

        # Align data with target
        self.data = self.data.iloc[:len(self.target)]

        logger.info(f"✅ Target created: {len(self.target):,} observations")
        logger.info(f"  • Target mean: {self.target.mean():.6f}")
        logger.info(f"  • Target std: {self.target.std():.6f}")

    def create_phase1_technical_indicators(self):
        """Create Phase 1 technical indicators that were successfully created."""
        logger.info("📈 Creating Phase 1 technical indicators...")
        try:
            # RSI (14)
            self.features['rsi_14'] = talib.RSI(self.data['close'].values, timeperiod=14)

            # MACD (12, 26, 9)
            macd, macd_signal, macd_hist = talib.MACD(
                self.data['close'].values, fastperiod=12, slowperiod=26, signalperiod=9
            )
            self.features['macd_line'] = macd
            self.features['macd_signal'] = macd_signal
            self.features['macd_histogram'] = macd_hist

            # Stochastic Oscillator (%K 14, %D 3)
            slowk, slowd = talib.STOCH(
                self.data['high'].values, self.data['low'].values, self.data['close'].values,
                fastk_period=14, slowk_period=3, slowd_period=3
            )
            self.features['stoch_k'] = slowk
            self.features['stoch_d'] = slowd

            # ATR (14)
            self.features['atr_14'] = talib.ATR(
                self.data['high'].values, self.data['low'].values, self.data['close'].values, timeperiod=14
            )

            # Bollinger Bands Position (Normalized within bands)
            upper, middle, lower = talib.BBANDS(
                self.data['close'].values, timeperiod=20, nbdevup=2, nbdevdn=2
            )
            band_width = upper - lower
            self.features['bb_position'] = np.where(
                band_width > 0, (self.data['close'].values - middle) / band_width, 0
            )

            logger.info(f"✅ Created 8 Phase 1 technical indicators")

        except Exception as e:
            logger.error(f"❌ Phase 1 technical indicators failed: {e}")

    def create_phase1_time_features(self):
        """Create Phase 1 time-based features."""
        logger.info("⏰ Creating Phase 1 time-based features...")
        try:
            # Convert timestamp to datetime if not already
            if hasattr(self.data.index, 'hour'):
                timestamp_series = self.data.index
            else:
                timestamp_series = pd.to_datetime(self.data['timestamp'])

            # Day of week (0=Monday, 4=Friday)
            self.features['day_of_week'] = timestamp_series.dayofweek

            # Time of day (normalized 0-1)
            self.features['time_of_day'] = (timestamp_series.hour / 24.0 +
                                          timestamp_series.minute / 1440.0)

            # Session indicator (if available)
            if 'session' in self.data.columns:
                self.features['session_indicator'] = self.data['session']
            else:
                self.features['session_indicator'] = 1

            logger.info(f"✅ Created 3 Phase 1 time features")

        except Exception as e:
            logger.error(f"❌ Phase 1 time features failed: {e}")

    def calculate_correlation(self, feature_name: str, feature_series: pd.Series):
        """Calculate statistical correlations."""
        # Align with target
        aligned_target = self.target.iloc[:len(feature_series)]
        aligned_feature = feature_series.dropna()

        # Further alignment
        min_len = min(len(aligned_target), len(aligned_feature))
        target_clean = aligned_target.iloc[:min_len]
        feature_clean = aligned_feature.iloc[:min_len]

        # Remove NaN values
        valid_mask = ~(target_clean.isna() | feature_clean.isna() |
                      np.isinf(target_clean) | np.isinf(feature_clean))
        target_clean = target_clean[valid_mask]
        feature_clean = feature_clean[valid_mask]

        if len(target_clean) < 100:
            return None

        results = {
            'feature_name': feature_name,
            'sample_size': len(target_clean),
            'feature_mean': feature_clean.mean(),
            'feature_std': feature_clean.std()
        }

        try:
            # Pearson correlation
            pearson_corr, pearson_p = pearsonr(feature_clean, target_clean)
            results['pearson_corr'] = pearson_corr
            results['pearson_p_value'] = pearson_p
            results['pearson_significant'] = pearson_p < 0.05
        except:
            results['pearson_corr'] = np.nan
            results['pearson_p_value'] = np.nan
            results['pearson_significant'] = False

        try:
            # Spearman correlation
            spearman_corr, spearman_p = spearmanr(feature_clean, target_clean)
            results['spearman_corr'] = spearman_corr
            results['spearman_p_value'] = spearman_p
            results['spearman_significant'] = spearman_p < 0.05
        except:
            results['spearman_corr'] = np.nan
            results['spearman_p_value'] = np.nan
            results['spearman_significant'] = False

        # Effect size
        abs_pearson = abs(results['pearson_corr']) if not np.isnan(results['pearson_corr']) else 0
        if abs_pearson >= 0.1:
            results['effect_size'] = 'medium' if abs_pearson >= 0.3 else 'small'
        else:
            results['effect_size'] = 'negligible'

        # Overall significance
        if results['pearson_significant'] and results['spearman_significant']:
            results['overall_significance'] = 'high'
        elif results['pearson_significant'] or results['spearman_significant']:
            results['overall_significance'] = 'moderate'
        else:
            results['overall_significance'] = 'low'

        return results

    def analyze_features(self):
        """Analyze all Phase 1 features."""
        logger.info("📈 Analyzing Phase 1 features...")

        results = []
        for feature_name, feature_series in self.features.items():
            logger.info(f"  Analyzing {feature_name}...")

            feature_results = self.calculate_correlation(feature_name, feature_series)
            if feature_results:
                results.append(feature_results)

        return results

    def print_results(self, results):
        """Print analysis results."""
        logger.info("\n" + "="*80)
        logger.info("📊 PHASE 1 FEATURE STATISTICAL ANALYSIS RESULTS")
        logger.info("="*80)

        if not results:
            logger.info("❌ No results to display")
            return

        # Sort by absolute Pearson correlation
        sorted_results = sorted(results,
                             key=lambda x: abs(x['pearson_corr']) if not np.isnan(x['pearson_corr']) else 0,
                             reverse=True)

        # Categorize features
        technical_features = [r for r in sorted_results if
                            any(ind in r['feature_name'] for ind in ['rsi', 'macd', 'stoch', 'atr', 'bb'])]
        time_features = [r for r in sorted_results if
                        any(time in r['feature_name'] for time in ['day_of_week', 'time_of_day', 'session'])]

        def print_category(title, features):
            logger.info(f"\n🎯 {title}:")
            if not features:
                logger.info("  ❌ No features in this category")
                return

            significant = [f for f in features if f['overall_significance'] == 'high']
            moderate = [f for f in features if f['overall_significance'] == 'moderate']

            logger.info(f"  • Total features: {len(features)}")
            logger.info(f"  • Highly significant: {len(significant)}")
            logger.info(f"  • Moderately significant: {len(moderate)}")

            logger.info(f"  • Top features by correlation:")
            for i, result in enumerate(features[:5]):
                if not np.isnan(result['pearson_corr']):
                    logger.info(f"    {i+1}. {result['feature_name']}: "
                              f"r={result['pearson_corr']:.4f} (p={result['pearson_p_value']:.4f})")

            # Assessment
            high_sig_pct = (len(significant) / len(features)) * 100 if features else 0
            if high_sig_pct >= 50:
                assessment = "🟢 HIGH RELEVANCE"
            elif high_sig_pct >= 25:
                assessment = "🟡 MODERATE RELEVANCE"
            else:
                assessment = "🔴 LOW RELEVANCE"
            logger.info(f"  • Assessment: {assessment}")

        # Print categories
        print_category("Phase 1 Technical Indicators", technical_features)
        print_category("Phase 1 Time Features", time_features)

        # Overall assessment
        total_high_sig = len([r for r in results if r['overall_significance'] == 'high'])
        total_features = len(results)

        logger.info(f"\n🚀 OVERALL PHASE 1 ASSESSMENT:")
        logger.info(f"  • Total highly significant features: {total_high_sig}/{total_features}")

        if total_high_sig >= 6:
            logger.info(f"  🎉 EXCELLENT: Phase 1 features show strong predictive power")
        elif total_high_sig >= 3:
            logger.info(f"  👍 GOOD: Phase 1 features show moderate predictive power")
        else:
            logger.info(f"  ⚠️  LIMITED: Phase 1 features show weak predictive power")

        logger.info("\n" + "="*80)

def main():
    """Main execution."""
    logger.info("🚀 Starting Focused Phase 1 Feature Analysis...")

    # Initialize analyzer
    analyzer = FocusedPhase1Analyzer('/workspace/DATA/MERGED/merged_es_vix_test.csv')

    # Load and prepare data
    analyzer.load_and_prepare_data()

    # Create Phase 1 features (only those that work)
    analyzer.create_phase1_technical_indicators()
    analyzer.create_phase1_time_features()

    # Analyze features
    results = analyzer.analyze_features()

    # Print results
    analyzer.print_results(results)

    logger.info("✅ Phase 1 analysis completed!")

if __name__ == "__main__":
    main()