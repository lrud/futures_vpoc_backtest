#!/usr/bin/env python3
"""
Optimized Statistical Relevance Analysis for Large Datasets

This script performs efficient statistical analysis on large minute-level datasets
by using optimized calculations and intelligent sampling strategies.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append('/workspace')

from src.utils.logging import get_logger

logger = get_logger(__name__)

class OptimizedFeatureAnalyzer:
    """Optimized analyzer for large-scale financial datasets."""

    def __init__(self, data_path: str, sample_size: int = 100000):
        self.data_path = data_path
        self.sample_size = sample_size
        self.data = None
        self.target = None
        self.features = {}

    def load_and_prepare_data(self):
        """Load and prepare data with intelligent sampling."""
        logger.info(f"📁 Loading data from {self.data_path}")

        # Load data
        data = pd.read_csv(self.data_path)

        # Basic data processing
        if 'VIX' in data.columns:
            data = data.rename(columns={'VIX': 'vix'})

        # Convert date column if exists
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
            data = data.set_index('date')

        # Remove rows with critical missing values
        essential_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in essential_cols:
            if col in data.columns:
                data = data[data[col].notna()]

        # Forward fill remaining missing values
        data = data.fillna(method='ffill').fillna(method='bfill')

        logger.info(f"✅ Loaded {len(data):,} rows of minute-level data")

        # Create target
        data['target'] = data['close'].pct_change().shift(-1)
        data = data.dropna(subset=['target'])

        # Intelligent sampling for statistical analysis
        if len(data) > self.sample_size:
            logger.info(f"📊 Sampling {self.sample_size:,} rows from {len(data):,} total rows")
            # Use systematic sampling to maintain temporal structure
            sample_step = len(data) // self.sample_size
            data = data.iloc[::sample_step][:self.sample_size]

        logger.info(f"✅ Analyzing {len(data):,} observations")
        logger.info(f"  • Target mean: {data['target'].mean():.6f}")
        logger.info(f"  • Target std: {data['target'].std():.6f}")

        self.data = data
        self.target = data['target']

    def create_efficient_features(self):
        """Create a focused set of efficient features."""
        logger.info("🚀 Creating efficient feature set...")

        # Basic price features (most predictive)
        close = self.data['close']
        self.features['price_change_1'] = close.pct_change(1)
        self.features['price_change_5'] = close.pct_change(5)
        self.features['price_change_15'] = close.pct_change(15)
        self.features['price_volatility_15'] = close.pct_change().rolling(15).std()

        # VIX features (if available)
        if 'vix' in self.data.columns:
            vix = self.data['vix'].fillna(method='ffill')
            self.features['vix_level'] = vix
            self.features['vix_change_1'] = vix.pct_change(1)
            self.features['vix_ma_60'] = vix.rolling(60).mean()
            self.features['vix_percentile_240'] = vix.rolling(240).rank(pct=True)
            logger.info("✅ Created 4 VIX features")

        # Volume features (if available)
        if 'volume' in self.data.columns:
            volume = self.data['volume']
            self.features['volume_level'] = volume
            self.features['volume_change_1'] = volume.pct_change(1)
            self.features['volume_ma_15'] = volume.rolling(15).mean()

            # Efficient VWAP calculation
            typical_price = (self.data['high'] + self.data['low'] + close) / 3
            self.features['vwap_15'] = (typical_price * volume).rolling(15).sum() / volume.rolling(15).sum()
            self.features['close_to_vwap'] = (close - self.features['vwap_15']) / self.features['vwap_15']
            logger.info("✅ Created 5 volume features")

        # Remove any NaN values
        for feature_name, feature_series in list(self.features.items()):
            if feature_series.isna().all():
                del self.features[feature_name]
            else:
                self.features[feature_name] = feature_series.fillna(0)

        logger.info(f"✅ Created {len(self.features)} total features")

    def calculate_efficient_correlations(self, feature_name: str, feature_series: pd.Series):
        """Calculate correlations efficiently."""
        # Align with target
        aligned_target = self.target.iloc[:len(feature_series)]
        aligned_feature = feature_series.iloc[:len(aligned_target)]

        # Remove any remaining NaN values
        valid_mask = ~(aligned_target.isna() | aligned_feature.isna() |
                      np.isinf(aligned_target) | np.isinf(aligned_feature))

        target_clean = aligned_target[valid_mask]
        feature_clean = aligned_feature[valid_mask]

        if len(target_clean) < 1000:
            return None

        results = {
            'sample_size': len(target_clean),
            'feature_mean': feature_clean.mean(),
            'feature_std': feature_clean.std()
        }

        try:
            # Pearson correlation (most important)
            pearson_corr, pearson_p = pearsonr(feature_clean, target_clean)
            results['pearson_corr'] = pearson_corr
            results['pearson_p_value'] = pearson_p
            results['pearson_significant'] = pearson_p < 0.05
            results['effect_size'] = 'large' if abs(pearson_corr) >= 0.3 else 'medium' if abs(pearson_corr) >= 0.1 else 'small'
        except:
            results['pearson_corr'] = 0
            results['pearson_p_value'] = 1.0
            results['pearson_significant'] = False
            results['effect_size'] = 'negligible'

        return results

    def run_efficient_analysis(self):
        """Run the complete efficient analysis."""
        logger.info("📈 Running efficient statistical analysis...")

        results = {}
        for feature_name, feature_series in self.features.items():
            logger.info(f"  Analyzing {feature_name}...")
            feature_results = self.calculate_efficient_correlations(feature_name, feature_series)
            if feature_results:
                results[feature_name] = feature_results

        return results

    def print_efficient_summary(self, results):
        """Print an efficient summary."""
        logger.info("\n" + "="*80)
        logger.info("📊 OPTIMIZED STATISTICAL FEATURE RELEVANCE SUMMARY")
        logger.info("="*80)

        # Sort by absolute correlation
        sorted_results = sorted(results.items(),
                              key=lambda x: abs(x[1]['pearson_corr']) if not np.isnan(x[1]['pearson_corr']) else 0,
                              reverse=True)

        logger.info(f"\n🎯 TOP 15 FEATURES BY CORRELATION STRENGTH:")

        for i, (name, stats) in enumerate(sorted_results[:15]):
            corr = stats['pearson_corr']
            p_val = stats['pearson_p_value']
            significant = "✅" if stats['pearson_significant'] else "❌"
            effect = stats['effect_size'].upper()

            logger.info(f"  {i+1:2d}. {name:<25} | r={corr:+7.4f} | p={p_val:.4f} | {significant} | {effect}")

        # Categorize results
        significant_features = [k for k, v in results.items() if v['pearson_significant']]
        high_correlation = [k for k, v in results.items() if abs(v['pearson_corr']) >= 0.1]
        medium_correlation = [k for k, v in results.items() if 0.05 <= abs(v['pearson_corr']) < 0.1]

        logger.info(f"\n📈 SUMMARY STATISTICS:")
        logger.info(f"  • Total features analyzed: {len(results)}")
        logger.info(f"  • Statistically significant: {len(significant_features)} ({len(significant_features)/len(results)*100:.1f}%)")
        logger.info(f"  • High correlation (|r|≥0.1): {len(high_correlation)} ({len(high_correlation)/len(results)*100:.1f}%)")
        logger.info(f"  • Medium correlation (0.05≤|r|<0.1): {len(medium_correlation)} ({len(medium_correlation)/len(results)*100:.1f}%)")

        # Feature recommendations
        logger.info(f"\n💡 FEATURE RECOMMENDATIONS:")

        if len(high_correlation) >= 5:
            logger.info(f"  ✅ EXCELLENT: {len(high_correlation)} features show strong correlations")
            logger.info("  → RECOMMENDATION: Include all high-correlation features in model")
        elif len(high_correlation) >= 2:
            logger.info(f"  🟡 GOOD: {len(high_correlation)} features show moderate correlations")
            logger.info("  → RECOMMENDATION: Include high-correlation features, consider medium ones")
        else:
            logger.info(f"  🔴 LIMITED: Only {len(high_correlation)} features show meaningful correlations")
            logger.info("  → RECOMMENDATION: Focus on feature engineering or different features")

        # VIX specific analysis
        vix_features = {k: v for k, v in results.items() if 'vix' in k}
        if vix_features:
            vix_significant = len([k for k, v in vix_features.items() if v['pearson_significant']])
            logger.info(f"\n🎯 VIX FEATURES: {vix_significant}/{len(vix_features)} significant")
            if vix_significant >= len(vix_features) * 0.5:
                logger.info("  → VIX is highly predictive for minute-level returns")
            else:
                logger.info("  → VIX has limited predictive power at this timeframe")

        # Volume specific analysis
        volume_features = {k: v for k, v in results.items() if any(word in k for word in ['volume', 'vwap'])}
        if volume_features:
            vol_significant = len([k for k, v in volume_features.items() if v['pearson_significant']])
            logger.info(f"📊 VOLUME FEATURES: {vol_significant}/{len(volume_features)} significant")
            if vol_significant >= len(volume_features) * 0.3:
                logger.info("  → Volume profile features are valuable")
            else:
                logger.info("  → Volume profile features need refinement")

        logger.info("\n" + "="*80)

def main():
    """Main execution function."""
    logger.info("🚀 Starting Optimized Statistical Feature Analysis...")

    # Initialize analyzer with sample size for efficiency
    analyzer = OptimizedFeatureAnalyzer(
        '/workspace/DATA/MERGED/merged_es_vix_test.csv',
        sample_size=100000  # 100K rows for efficient processing
    )

    # Load and prepare data
    analyzer.load_and_prepare_data()

    # Create focused feature set
    analyzer.create_efficient_features()

    # Run efficient analysis
    results = analyzer.run_efficient_analysis()

    # Print summary
    analyzer.print_efficient_summary(results)

    logger.info("✅ Optimized statistical analysis completed!")

if __name__ == "__main__":
    main()