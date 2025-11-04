#!/usr/bin/env python3
"""
Comprehensive Volume Profile Analysis for Full Dataset

This script analyzes the complete 1.18M observation dataset with:
- Full dataset statistical analysis (no sampling)
- Detailed volume profile creation and statistics
- Hourly volume profile characterization
- VPOC detection and value area analysis
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append('/workspace')

from src.utils.logging import get_logger

logger = get_logger(__name__)

class ComprehensiveVolumeAnalyzer:
    """Comprehensive analyzer for full dataset with detailed volume profiles."""

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data = None
        self.target = None
        self.features = {}
        self.volume_profiles = {}
        self.hourly_stats = {}

    def load_and_prepare_data(self):
        """Load and prepare complete dataset."""
        logger.info(f"📁 Loading complete dataset from {self.data_path}")

        # Load data
        data = pd.read_csv(self.data_path)

        # Basic data processing
        if 'VIX' in data.columns:
            data = data.rename(columns={'VIX': 'vix'})

        # Convert timestamp column for proper minute-level indexing
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data = data.set_index('timestamp')

        # Remove rows with critical missing values
        essential_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in essential_cols:
            if col in data.columns:
                data = data[data[col].notna()]

        # Forward fill remaining missing values
        data = data.fillna(method='ffill').fillna(method='bfill')

        logger.info(f"✅ Loaded complete dataset: {len(data):,} rows")

        # Create target
        data['target'] = data['close'].pct_change().shift(-1)
        data = data.dropna(subset=['target'])

        logger.info(f"✅ Target created: {len(data):,} observations")
        logger.info(f"  • Target mean: {data['target'].mean():.6f}")
        logger.info(f"  • Target std: {data['target'].std():.6f}")

        # Calculate dataset statistics
        unique_days = len(data.index.normalize().unique())
        hours_per_day = len(data) / unique_days
        logger.info(f"📅 Dataset spans {unique_days} trading days")
        logger.info(f"⏱️  Average minutes per day: {hours_per_day:.1f}")

        self.data = data
        self.target = data['target']

    def create_four_hour_volume_profiles(self):
        """Create detailed 4-hour volume profiles."""
        logger.info("📊 Creating detailed 4-hour volume profiles...")

        # Resample to 4-hour data
        four_hourly_data = self.data.resample('4H').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        logger.info(f"✅ Created {len(four_hourly_data)} 4-hour profiles")

        # Calculate volume profile statistics for each 4-hour period
        logger.info("🔍 Analyzing 4-hour volume profile characteristics...")

        four_hourly_stats = []
        for period_idx, (timestamp, period_data) in enumerate(four_hourly_data.iterrows()):
            # Get minute-level data for this 4-hour period
            period_start = timestamp
            period_end = timestamp + pd.Timedelta(hours=4)
            minute_data = self.data.loc[period_start:period_end]

            if len(minute_data) > 0:
                # Volume profile analysis
                volume_profile_stats = self.analyze_volume_profile(minute_data, period_data)
                volume_profile_stats['period'] = timestamp
                volume_profile_stats['minute_count'] = len(minute_data)
                four_hourly_stats.append(volume_profile_stats)

        self.four_hourly_stats = pd.DataFrame(four_hourly_stats)
        logger.info(f"✅ Analyzed {len(self.four_hourly_stats)} 4-hour volume profiles")

        # Log 4-hour statistics summary
        self.log_four_hour_statistics()

    def analyze_volume_profile(self, minute_data, hourly_summary):
        """Analyze volume profile for a single hour."""
        high = minute_data['high']
        low = minute_data['low']
        close = minute_data['close']
        volume = minute_data['volume']

        # Basic statistics
        stats = {
            'hourly_high': hourly_summary['high'],
            'hourly_low': hourly_summary['low'],
            'hourly_close': hourly_summary['close'],
            'hourly_volume': hourly_summary['volume'],
            'hourly_range': hourly_summary['high'] - hourly_summary['low'],
            'minute_count': len(minute_data)
        }

        # Volume distribution analysis
        price_bins = 50  # Number of price levels
        if len(minute_data) > 1 and not (high == low).all():
            price_levels = np.linspace(low.min(), high.max(), price_bins)
            volume_distribution = []

            for i in range(len(price_levels) - 1):
                price_lower = price_levels[i]
                price_upper = price_levels[i + 1]

                # Volume traded at this price level (approximation)
                mask = (close >= price_lower) & (close < price_upper)
                vol_at_level = volume[mask].sum()
                volume_distribution.append(vol_at_level)

            volume_distribution = np.array(volume_distribution)

            # VPOC (Volume Point of Control) - price with max volume
            vpoc_idx = np.argmax(volume_distribution)
            vpoc_price = (price_levels[vpoc_idx] + price_levels[vpoc_idx + 1]) / 2
            vpoc_volume = volume_distribution[vpoc_idx]

            # VWAP (Volume Weighted Average Price)
            vwap = (close * volume).sum() / volume.sum()

            # Value Area (70% of total volume)
            total_volume = volume_distribution.sum()
            cumulative_volume = np.cumsum(volume_distribution)
            value_area_threshold = total_volume * 0.7

            value_area_lower_idx = np.argmax(cumulative_volume >= value_area_threshold * 0.16)
            value_area_upper_idx = np.argmax(cumulative_volume >= value_area_threshold * 0.84)

            value_area_lower = (price_levels[value_area_lower_idx] + price_levels[value_area_lower_idx + 1]) / 2
            value_area_upper = (price_levels[value_area_upper_idx] + price_levels[value_area_upper_idx + 1]) / 2

            # Volume concentration metrics
            max_volume_concentration = vpoc_volume / total_volume
            volume_std = np.std(volume_distribution)

            stats.update({
                'vpoc_price': vpoc_price,
                'vpoc_volume': vpoc_volume,
                'vwap': vwap,
                'value_area_lower': value_area_lower,
                'value_area_upper': value_area_upper,
                'value_area_width': value_area_upper - value_area_lower,
                'max_volume_concentration': max_volume_concentration,
                'volume_std': volume_std,
                'volume_distribution_entropy': self.calculate_entropy(volume_distribution),
                'close_to_vpoc_pct': (hourly_summary['close'] - vpoc_price) / vpoc_price,
                'close_to_vwap_pct': (hourly_summary['close'] - vwap) / vwap,
                'vpoc_in_value_area': value_area_lower <= vpoc_price <= value_area_upper,
                'close_in_value_area': value_area_lower <= hourly_summary['close'] <= value_area_upper
            })

        return stats

    def calculate_entropy(self, distribution):
        """Calculate entropy of volume distribution."""
        # Normalize to create probability distribution
        if distribution.sum() == 0:
            return 0
        probs = distribution / distribution.sum()
        probs = probs[probs > 0]  # Remove zero probabilities
        return -np.sum(probs * np.log2(probs))

    def log_four_hour_statistics(self):
        """Log comprehensive 4-hour volume profile statistics."""
        logger.info("\n" + "="*80)
        logger.info("📊 4-HOUR VOLUME PROFILE STATISTICS SUMMARY")
        logger.info("="*80)

        # Basic statistics
        logger.info(f"\n📈 BASIC 4-HOUR STATISTICS:")
        logger.info(f"  • Total 4-hour periods analyzed: {len(self.four_hourly_stats)}")
        logger.info(f"  • Average minutes per 4-hour period: {self.four_hourly_stats['minute_count'].mean():.1f}")
        logger.info(f"  • Average 4-hour volume: {self.four_hourly_stats['hourly_volume'].mean():,.0f}")
        logger.info(f"  • Average 4-hour range: {self.four_hourly_stats['hourly_range'].mean():.2f}")

        # VPOC statistics
        vpoc_data = self.four_hourly_stats.dropna(subset=['vpoc_price'])
        if len(vpoc_data) > 0:
            logger.info(f"\n🎯 VPOC (Volume Point of Control) ANALYSIS:")
            logger.info(f"  • Average VPOC volume concentration: {vpoc_data['max_volume_concentration'].mean():.3f}")
            logger.info(f"  • VPOC in value area: {vpoc_data['vpoc_in_value_area'].sum()}/{len(vpoc_data)} ({vpoc_data['vpoc_in_value_area'].mean()*100:.1f}%)")
            logger.info(f"  • Average close-to-VPOC distance: {abs(vpoc_data['close_to_vpoc_pct']).mean():.3f}")

        # Value Area statistics
        value_area_data = self.four_hourly_stats.dropna(subset=['value_area_width'])
        if len(value_area_data) > 0:
            logger.info(f"\n📏 VALUE AREA ANALYSIS:")
            logger.info(f"  • Average value area width: {value_area_data['value_area_width'].mean():.2f}")
            logger.info(f"  • Average value area width as % of range: {(value_area_data['value_area_width'] / value_area_data['hourly_range']).mean()*100:.1f}%")
            logger.info(f"  • Close in value area: {value_area_data['close_in_value_area'].sum()}/{len(value_area_data)} ({value_area_data['close_in_value_area'].mean()*100:.1f}%)")

        # Volume distribution characteristics
        entropy_data = self.four_hourly_stats.dropna(subset=['volume_distribution_entropy'])
        if len(entropy_data) > 0:
            logger.info(f"\n🔢 VOLUME DISTRIBUTION ANALYSIS:")
            logger.info(f"  • Average entropy: {entropy_data['volume_distribution_entropy'].mean():.2f} bits")
            logger.info(f"  • Entropy range: {entropy_data['volume_distribution_entropy'].min():.2f} - {entropy_data['volume_distribution_entropy'].max():.2f} bits")

        # VWAP vs VPOC comparison
        vwap_comparison = self.four_hourly_stats.dropna(subset=['vpoc_price', 'vwap'])
        if len(vwap_comparison) > 0:
            vwap_vpoc_diff = abs(vwap_comparison['vwap'] - vwap_comparison['vpoc_price'])
            logger.info(f"\n⚖️  VWAP vs VPOC COMPARISON:")
            logger.info(f"  • Average VWAP-VPOC difference: {vwap_vpoc_diff.mean():.2f}")
            logger.info(f"  • VWAP-VPOC difference as % of price: {(vwap_vpoc_diff / vwap_comparison['vwap']).mean()*100:.2f}%")

        logger.info("\n" + "="*80)

    def create_efficient_features(self):
        """Create features for statistical analysis."""
        logger.info("🚀 Creating features for full dataset analysis...")

        # Price features
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

        # Enhanced Volume Profile features
        if 'volume' in self.data.columns:
            volume = self.data['volume']
            high = self.data['high']
            low = self.data['low']
            close = self.data['close']

            # Basic volume features
            self.features['volume_level'] = volume
            self.features['volume_change_1'] = volume.pct_change(1)
            self.features['volume_ma_15'] = volume.rolling(15).mean()

            # VWAP calculations
            typical_price = (high + low + close) / 3
            self.features['vwap_15'] = (typical_price * volume).rolling(15).sum() / volume.rolling(15).sum()
            self.features['vwap_60'] = (typical_price * volume).rolling(60).sum() / volume.rolling(60).sum()
            self.features['close_to_vwap_15'] = (close - self.features['vwap_15']) / self.features['vwap_15']
            self.features['close_to_vwap_60'] = (close - self.features['vwap_60']) / self.features['vwap_60']

            # Map hourly VPOC features
            self.map_hourly_vpoc_features()

            logger.info("✅ Created enhanced volume profile features")

        # Remove any NaN values
        for feature_name, feature_series in list(self.features.items()):
            if feature_series.isna().all():
                del self.features[feature_name]
            else:
                self.features[feature_name] = feature_series.fillna(0)

        logger.info(f"✅ Created {len(self.features)} total features")

    def map_hourly_vpoc_features(self):
        """Map hourly VPOC analysis back to minute-level data."""
        if len(self.hourly_stats) == 0:
            return

        # Create dictionaries for mapping
        vpoc_price_map = {}
        vwap_map = {}
        value_area_lower_map = {}
        value_area_upper_map = {}

        for _, row in self.hourly_stats.iterrows():
            hour_key = row['hour']
            vpoc_price_map[hour_key] = row.get('vpoc_price', np.nan)
            vwap_map[hour_key] = row.get('vwap', np.nan)
            value_area_lower_map[hour_key] = row.get('value_area_lower', np.nan)
            value_area_upper_map[hour_key] = row.get('value_area_upper', np.nan)

        # Map to minute-level data
        def map_hourly_feature(timestamp, feature_map):
            hour_key = timestamp.replace(minute=0, second=0, microsecond=0)
            return feature_map.get(hour_key, np.nan)

        self.features['hourly_vpoc_price'] = self.data.index.to_series().apply(
            lambda x: map_hourly_feature(x, vpoc_price_map)
        )
        self.features['hourly_vwap'] = self.data.index.to_series().apply(
            lambda x: map_hourly_feature(x, vwap_map)
        )
        self.features['hourly_value_area_lower'] = self.data.index.to_series().apply(
            lambda x: map_hourly_feature(x, value_area_lower_map)
        )
        self.features['hourly_value_area_upper'] = self.data.index.to_series().apply(
            lambda x: map_hourly_feature(x, value_area_upper_map)
        )

        # Forward-fill within each hour
        for feature in ['hourly_vpoc_price', 'hourly_vwap', 'hourly_value_area_lower', 'hourly_value_area_upper']:
            self.features[feature] = self.features[feature].ffill(limit=59)

        # Create derived features
        self.features['close_to_hourly_vpoc'] = (self.data['close'] - self.features['hourly_vpoc_price']) / self.features['hourly_vpoc_price']
        self.features['close_in_hourly_value_area'] = (
            (self.data['close'] >= self.features['hourly_value_area_lower']) &
            (self.data['close'] <= self.features['hourly_value_area_upper'])
        ).astype(int)

    def calculate_correlations(self, feature_name: str, feature_series: pd.Series):
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
            # Pearson correlation
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

    def run_full_analysis(self):
        """Run the complete analysis on full dataset."""
        logger.info("📈 Running full dataset statistical analysis...")

        results = {}
        feature_count = 0
        for feature_name, feature_series in self.features.items():
            feature_count += 1
            logger.info(f"  Analyzing feature {feature_count}/{len(self.features)}: {feature_name}")

            feature_results = self.calculate_correlations(feature_name, feature_series)
            if feature_results:
                results[feature_name] = feature_results

        return results

    def print_comprehensive_summary(self, results):
        """Print comprehensive analysis summary."""
        logger.info("\n" + "="*80)
        logger.info("📊 COMPREHENSIVE FULL DATASET STATISTICAL ANALYSIS")
        logger.info("="*80)

        # Sort by absolute correlation
        sorted_results = sorted(results.items(),
                              key=lambda x: abs(x[1]['pearson_corr']) if not np.isnan(x[1]['pearson_corr']) else 0,
                              reverse=True)

        logger.info(f"\n🎯 TOP 20 FEATURES BY CORRELATION STRENGTH:")

        for i, (name, stats) in enumerate(sorted_results[:20]):
            corr = stats['pearson_corr']
            p_val = stats['pearson_p_value']
            significant = "✅" if stats['pearson_significant'] else "❌"
            effect = stats['effect_size'].upper()

            logger.info(f"  {i+1:2d}. {name:<30} | r={corr:+7.4f} | p={p_val:.4f} | {significant} | {effect}")

        # Categorize results
        significant_features = [k for k, v in results.items() if v['pearson_significant']]
        high_correlation = [k for k, v in results.items() if abs(v['pearson_corr']) >= 0.1]
        medium_correlation = [k for k, v in results.items() if 0.05 <= abs(v['pearson_corr']) < 0.1]

        logger.info(f"\n📈 FULL DATASET STATISTICS:")
        logger.info(f"  • Total features analyzed: {len(results)}")
        logger.info(f"  • Total observations: {self.target.shape[0]:,}")
        logger.info(f"  • Statistically significant: {len(significant_features)} ({len(significant_features)/len(results)*100:.1f}%)")
        logger.info(f"  • High correlation (|r|≥0.1): {len(high_correlation)} ({len(high_correlation)/len(results)*100:.1f}%)")
        logger.info(f"  • Medium correlation (0.05≤|r|<0.1): {len(medium_correlation)} ({len(medium_correlation)/len(results)*100:.1f}%)")

        # Feature categories analysis
        price_features = [k for k, v in results.items() if 'price' in k]
        vix_features = [k for k, v in results.items() if 'vix' in k]
        vpoc_features = [k for k, v in results.items() if any(word in k for word in ['vpoc', 'value_area'])]
        vwap_features = [k for k, v in results.items() if 'vwap' in k]
        volume_features = [k for k, v in results.items() if 'volume' in k]

        logger.info(f"\n📊 FEATURE CATEGORY PERFORMANCE:")
        logger.info(f"  • Price features: {len([k for k in price_features if k in significant_features])}/{len(price_features)} significant")
        logger.info(f"  • VIX features: {len([k for k in vix_features if k in significant_features])}/{len(vix_features)} significant")
        logger.info(f"  • VPOC features: {len([k for k in vpoc_features if k in significant_features])}/{len(vpoc_features)} significant")
        logger.info(f"  • VWAP features: {len([k for k in vwap_features if k in significant_features])}/{len(vwap_features)} significant")
        logger.info(f"  • Volume features: {len([k for k in volume_features if k in significant_features])}/{len(volume_features)} significant")

        # Top correlations
        top_5 = sorted_results[:5]
        logger.info(f"\n🏆 TOP 5 PREDICTIVE FEATURES:")
        for i, (name, stats) in enumerate(top_5):
            logger.info(f"  {i+1}. {name}: r={stats['pearson_corr']:+.4f} (p={stats['pearson_p_value']:.4f})")

        logger.info("\n" + "="*80)

def main():
    """Main execution function."""
    logger.info("🚀 Starting Comprehensive Full Dataset Volume Analysis...")

    # Initialize analyzer
    analyzer = ComprehensiveVolumeAnalyzer('/workspace/DATA/MERGED/merged_es_vix_test.csv')

    # Load and prepare complete dataset
    analyzer.load_and_prepare_data()

    # Create hourly volume profiles
    analyzer.create_four_hour_volume_profiles()

    # Create features
    analyzer.create_efficient_features()

    # Run full analysis
    results = analyzer.run_full_analysis()

    # Print comprehensive summary
    analyzer.print_comprehensive_summary(results)

    logger.info("✅ Comprehensive analysis completed!")

if __name__ == "__main__":
    main()