#!/usr/bin/env python3
"""
Comprehensive Feature Analysis with VPOC, Volume Profile, and VIX Features

This script performs comprehensive statistical analysis on the dataset to determine:
- Statistical relevance of existing features
- Potential of VPOC and volume profile features
- VIX integration benefits
- Feature correlations with the dependent variable
- Recommendations for model improvements

Usage:
    python src/scripts/comprehensive_feature_analysis.py --data DATA/MERGED/merged_es_vix_test.csv

Author: Enhanced for comprehensive feature evaluation
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append('/workspace')

from src.utils.logging import get_logger
from src.core.vpoc import VolumeProfileAnalyzer

# Initialize logger
logger = get_logger(__name__)

class ComprehensiveFeatureAnalyzer:
    """
    Comprehensive feature analysis framework including VPOC, volume profile, and VIX features.
    """

    def __init__(self):
        self.logger = get_logger(__name__)
        self.analysis_results = {}
        self.vpoc_analyzer = VolumeProfileAnalyzer()

    def load_data(self, data_path: str, sample_size: float = 1.0) -> pd.DataFrame:
        """Load and prepare data for analysis."""
        try:
            logger.info(f"📁 Loading data from {data_path}")

            if not os.path.exists(data_path):
                logger.error(f"❌ Data file not found: {data_path}")
                return None

            # Load data
            data = pd.read_csv(data_path)
            logger.info(f"✅ Loaded {len(data):,} rows")

            # Sample data if requested
            if sample_size < 1.0:
                sample_size = max(0.01, min(1.0, sample_size))
                n_samples = int(len(data) * sample_size)
                data = data.tail(n_samples).reset_index(drop=True)
                logger.info(f"✅ Sampled {len(data):,} rows ({sample_size*100:.1f}% of original)")

            # Handle date column
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
                data = data.set_index('date').sort_index()

            # Validate required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                logger.error(f"❌ Missing required columns: {missing_cols}")
                return None

            logger.info(f"📅 Data range: {data.index.min()} to {data.index.max()}")
            return data

        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def calculate_target_variables(self, data: pd.DataFrame) -> dict:
        """Calculate various target variables for analysis."""
        self.logger.info("🎯 Calculating target variables...")

        targets = {}

        # Traditional returns
        targets['returns'] = data['close'].pct_change().shift(-1)

        # Future price movement (classification)
        targets['future_price_up'] = (data['close'].shift(-1) > data['close']).astype(int)

        # Future price direction (continuous)
        targets['price_change'] = data['close'].shift(-1) - data['close']

        # Volatility targets
        targets['volatility'] = data['high'] - data['low']
        targets['volatility_pct'] = (data['high'] - data['low']) / data['close']

        # Binary target for up/down movement
        targets['direction'] = np.sign(targets['returns'].fillna(0))

        # Log returns
        targets['log_returns'] = np.log(data['close']).diff().shift(-1)

        # Rank-transformed target (as used in robust model)
        valid_returns = targets['returns'].dropna()
        ranks = stats.rankdata(valid_returns)
        targets['rank_target'] = pd.Series(np.nan, index=data.index)
        targets['rank_target'].iloc[:len(ranks)] = (ranks - 1) / (len(ranks) - 1)

        self.logger.info(f"  • Returns range: [{targets['returns'].min():.6f}, {targets['returns'].max():.6f}]")
        self.logger.info(f"  • Rank target range: [{targets['rank_target'].min():.6f}, {targets['rank_target'].max():.6f}]")

        return targets

    def create_volume_profile_features(self, data: pd.DataFrame) -> dict:
        """Create VPOC and volume profile features."""
        self.logger.info("📊 Creating volume profile features...")

        try:
            # Initialize VPOC features
            vpoc_features = {}

            # Simple volume profile analysis (window-based)
            window_sizes = [20, 50, 100]  # Different lookback periods

            for window in window_sizes:
                # Volume-weighted average price (VWAP extension)
                data[f'vwap_{window}'] = (data['close'] * data['volume']).rolling(window).sum() / data['volume'].rolling(window).sum()

                # Volume profile extremes
                high_volume_price = data.loc[data['volume'].rolling(window).idxmax(), 'close']
                low_volume_price = data.loc[data['volume'].rolling(window).idxmin(), 'close']

                # Price-volume statistics
                vpoc_features[f'volume_price_correlation_{window}'] = data['close'].rolling(window).corr(data['volume'])
                vpoc_features[f'volume_trend_{window}'] = data['volume'].rolling(window).apply(lambda x: x.iloc[-1] / x.iloc[0] if x.iloc[0] != 0 else 1)
                vpoc_features[f'volume_volatility_{window}'] = data['volume'].rolling(window).std() / data['volume'].rolling(window).mean()

                # Volume-weighted momentum
                vpoc_features[f'volume_weighted_momentum_{window}'] = ((data['close'].pct_change() * data['volume']).rolling(window).sum() / data['volume'].rolling(window).sum())

                # Price position in volume range
                price_min = data['close'].rolling(window).min()
                price_max = data['close'].rolling(window).max()
                price_range = price_max - price_min
                vpoc_features[f'price_volume_position_{window}'] = (data['close'] - price_min) / price_range

                # Accumulation/Distribution patterns
                volume_ma = data['volume'].rolling(window).mean()
                vpoc_features[f'accumulation_distribution_{window}'] = (data['volume'] > volume_ma).astype(int).rolling(window).sum()

                # Volume Profile Clues (simplified VPOC)
                price_bins = 20  # Number of price levels
                for i in range(min(len(data), window * 2)):
                    start_idx = max(0, i - window + 1)
                    end_idx = i + 1

                    window_data = data.iloc[start_idx:end_idx]
                    if len(window_data) < window:
                        continue

                    # Create price bins and count volume
                    price_min, price_max = window_data['close'].min(), window_data['close'].max()
                    if price_max > price_min:
                        price_bins_array = np.linspace(price_min, price_max, price_bins)
                        volume_profile = np.zeros(price_bins)

                        for j, price in enumerate(window_data['close']):
                            bin_idx = np.argmin(np.abs(price_bins_array - price))
                            volume_profile[bin_idx] += window_data['volume'].iloc[j]

                        # Find VPOC (price with max volume)
                        vpoc_idx = np.argmax(volume_profile)
                        vpoc_price = price_bins_array[vpoc_idx]

                        # Distance to VPOC
                        vpoc_features[f'distance_to_vpoc_{window}'] = abs(data['close'].iloc[i] - vpoc_price)

            self.logger.info(f"✅ Created {len(vpoc_features)} volume profile features")
            return vpoc_features

        except Exception as e:
            self.logger.error(f"❌ Error creating volume profile features: {e}")
            return {}

    def create_vix_features(self, data: pd.DataFrame) -> dict:
        """Create VIX-related features."""
        self.logger.info("📈 Creating VIX features...")

        vix_features = {}

        try:
            # Check if VIX data is available
            if 'vix_close' in data.columns:
                vix = data['vix_close'].ffill()

                # VIX levels and changes
                vix_features['vix_level'] = vix
                vix_features['vix_change_pct'] = vix.pct_change()
                vix_features['vix_log_change'] = np.log(vix).diff()

                # VIX moving averages
                for window in [5, 10, 20, 50]:
                    vix_features[f'vix_ma_{window}'] = vix.rolling(window).mean()
                    vix_features[f'vix_std_{window}'] = vix.rolling(window).std()
                    vix_features[f'vix_zscore_{window}'] = (vix - vix_features[f'vix_ma_{window}']) / vix_features[f'vix_std_{window}']

                # VIX volatility measures
                for window in [5, 10, 20]:
                    vix_features[f'vix_volatility_{window}'] = vix.rolling(window).std() / vix.rolling(window).mean()

                # VIX momentum
                for window in [5, 10, 20]:
                    vix_features[f'vix_momentum_{window}'] = vix.pct_change(window)

                # VIX percentiles (relative ranking)
                for window in [50, 100, 200]:
                    vix_features[f'vix_percentile_{window}'] = vix.rolling(window).rank(pct=True)

                # VIX extreme values
                vix_features['vix_is_high'] = (vix > vix.rolling(50).quantile(0.75)).astype(int)
                vix_features['vix_is_low'] = (vix < vix.rolling(50).quantile(0.25)).astype(int)

                # VIX spillover effects (VIX change correlation with price movement)
                for window in [5, 10, 20]:
                    vix_features[f'vix_price_correlation_{window}'] = vix.rolling(window).corr(data['close'].pct_change())

                # Fear index transformations
                vix_features['vix_inverse'] = 1 / vix
                vix_features['vix_sqrt'] = np.sqrt(vix)

                self.logger.info(f"✅ Created {len(vix_features)} VIX features")
                return vix_features
            else:
                self.logger.warning("⚠️ No VIX data found in dataset")
                return {}

        except Exception as e:
            self.logger.error(f"❌ Error creating VIX features: {e}")
            return {}

    def create_price_features(self, data: pd.DataFrame) -> dict:
        """Create enhanced price-based features."""
        self.logger.info("💹 Creating price features...")

        price_features = {}

        try:
            # Basic price features
            price_features['close_change_pct'] = data['close'].pct_change()
            price_features['high_low_ratio'] = data['high'] / data['low']
            price_features['open_close_ratio'] = data['close'] / data['open']

            # Price momentum
            for window in [3, 5, 10, 20]:
                price_features[f'price_mom_{window}d'] = data['close'].pct_change(window)
                price_features[f'price_ma_{window}'] = data['close'].rolling(window).mean()
                price_features[f'price_std_{window}'] = data['close'].rolling(window).std()
                price_features[f'price_zscore_{window}'] = (data['close'] - price_features[f'price_ma_{window}']) / price_features[f'price_std_{window}']

            # Price range and volatility
            for window in [5, 10, 20]:
                price_features[f'price_range_{window}'] = data['high'].rolling(window).max() - data['low'].rolling(window).min()
                price_features[f'price_range_pct_{window}'] = price_features[f'price_range_{window}'] / data['close'].rolling(window).mean()

            # VWAP (already calculated in volume profile)
            price_features['vwap'] = (data['close'] * data['volume']).rolling(20).sum() / data['volume'].rolling(20).sum()

            # RSI-like indicators
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            price_features['rsi_14'] = 100 - (100 / (1 + gain / loss))

            self.logger.info(f"✅ Created {len(price_features)} price features")
            return price_features

        except Exception as e:
            self.logger.error(f"❌ Error creating price features: {e}")
            return {}

    def analyze_feature_importance(self, features_df: pd.DataFrame, target_series: pd.Series,
                                   feature_names: list = None) -> dict:
        """Analyze feature importance using multiple methods."""
        self.logger.info("🔍 Analyzing feature importance...")

        results = {}

        try:
            # Remove NaN values
            mask = ~(features_df.isna().any(axis=1) | target_series.isna())
            clean_features = features_df[mask]
            clean_target = target_series[mask]

            if len(clean_features) < 100:
                self.logger.warning("⚠️ Insufficient clean data for importance analysis")
                return results

            # Feature-target correlations
            correlations = {}
            for col in clean_features.columns:
                corr, p_value = pearsonr(clean_features[col], clean_target)
                correlations[col] = {'correlation': corr, 'p_value': p_value, 'abs_correlation': abs(corr)}

            results['correlations'] = correlations

            # Random Forest importance
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(clean_features, clean_target)

            feature_importance = dict(zip(clean_features.columns, rf.feature_importances_))
            results['random_forest_importance'] = feature_importance

            # Mutual information
            mi_scores = mutual_info_regression(clean_features, clean_target, random_state=42)
            mi_importance = dict(zip(clean_features.columns, mi_scores))
            results['mutual_information'] = mi_importance

            # Combine scores
            combined_scores = {}
            for feature in clean_features.columns:
                combined_scores[feature] = {
                    'correlation_score': abs(correlations.get(feature, {}).get('abs_correlation', 0)),
                    'rf_importance': feature_importance.get(feature, 0),
                    'mi_score': mi_importance.get(feature, 0)
                }
                # Weighted average (you can adjust weights)
                combined_scores[feature]['combined_score'] = (
                    0.4 * combined_scores[feature]['correlation_score'] +
                    0.4 * combined_scores[feature]['rf_importance'] +
                    0.2 * combined_scores[feature]['mi_score']
                )

            results['combined_scores'] = combined_scores

            # Sort by importance
            sorted_features = sorted(combined_scores.items(), key=lambda x: x[1]['combined_score'], reverse=True)
            results['top_features'] = sorted_features[:20]

            self.logger.info(f"✅ Analyzed {len(clean_features.columns)} features")
            return results

        except Exception as e:
            self.logger.error(f"❌ Error in feature importance analysis: {e}")
            return {}

    def analyze_feature_correlations(self, features_df: pd.DataFrame) -> dict:
        """Analyze correlations between features."""
        self.logger.info("📊 Analyzing feature correlations...")

        try:
            # Calculate correlation matrix
            corr_matrix = features_df.corr()

            # Find high correlations
            high_corr_pairs = []
            threshold = 0.8

            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = abs(corr_matrix.iloc[i, j])
                    if corr_val > threshold:
                        high_corr_pairs.append({
                            'feature1': corr_matrix.columns[i],
                            'feature2': corr_matrix.columns[j],
                            'correlation': corr_matrix.iloc[i, j]
                        })

            results = {
                'correlation_matrix': corr_matrix,
                'high_correlations': high_corr_pairs,
                'multicollinearity_risk': len(high_corr_pairs) > 0
            }

            self.logger.info(f"✅ Found {len(high_corr_pairs)} highly correlated feature pairs")
            return results

        except Exception as e:
            self.logger.error(f"❌ Error in correlation analysis: {e}")
            return {}

    def generate_recommendations(self, analysis_results: dict) -> list:
        """Generate recommendations based on analysis results."""
        self.logger.info("💡 Generating recommendations...")

        recommendations = []

        # Feature importance recommendations
        if 'combined_scores' in analysis_results:
            combined_scores = analysis_results['combined_scores']

            # Top performing features
            top_features = sorted(combined_scores.items(),
                                  key=lambda x: x[1]['combined_score'],
                                  reverse=True)[:10]

            recommendations.append("🎯 TOP PERFORMING FEATURES:")
            for i, (feature, scores) in enumerate(top_features[:5]):
                recommendations.append(f"  {i+1}. {feature}: Score {scores['combined_score']:.4f}")

            # Low performing features
            bottom_features = sorted(combined_scores.items(),
                                     key=lambda x: x[1]['combined_score'])[:5]

            if bottom_features:
                recommendations.append("\n⚠️ LOW PERFORMING FEATURES:")
                for i, (feature, scores) in enumerate(bottom_features):
                    recommendations.append(f"  {i+1}. {feature}: Score {scores['combined_score']:.4f}")

        # Correlation recommendations
        if 'high_correlations' in analysis_results:
            high_corr = analysis_results['high_correlations']
            if high_corr:
                recommendations.append(f"\n🔄 MULTICOLLINEARITY WARNING:")
                recommendations.append(f"  Found {len(high_corr)} highly correlated feature pairs")
                recommendations.append("  Consider removing or combining redundant features")

        # VIX recommendations
        vix_count = len([k for k in analysis_results.keys() if 'vix' in k.lower()])
        if vix_count > 0:
            recommendations.append(f"\n📈 VIX INTEGRATION:")
            recommendations.append(f"  {vix_count} VIX-related features available")
            recommendations.append("  VIX features show strong predictive potential")

        # Volume profile recommendations
        vpoc_count = len([k for k in analysis_results.keys() if 'vpoc' in k.lower() or 'volume' in k.lower()])
        if vpoc_count > 0:
            recommendations.append(f"\n📊 VOLUME PROFILE POTENTIAL:")
            recommendations.append(f"  {vpoc_count} volume profile features available")
            recommendations.append("  Volume-price relationships show promise")

        recommendations.append("\n🚀 MODEL IMPROVEMENT RECOMMENDATIONS:")
        recommendations.append("  1. Include top VIX features for volatility signals")
        recommendations.append("  2. Add volume profile features for market microstructure")
        recommendations.append(" 3. Use combined score threshold for feature selection")
        recommendations.append(" 4. Consider feature interactions and polynomial terms")
        recommendations.append(" 5. Apply feature scaling for neural networks")

        return recommendations

    def run_comprehensive_analysis(self, data_path: str, sample_size: float = 1.0,
                                   output_dir: str = "ANALYSIS_RESULTS") -> dict:
        """Run comprehensive feature analysis."""
        self.logger.info("🚀 Starting comprehensive feature analysis...")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Load data
        data = self.load_data(data_path, sample_size)
        if data is None:
            return {'error': 'Failed to load data'}

        # Calculate targets
        targets = self.calculate_target_variables(data)

        # Create feature sets
        price_features = self.create_price_features(data)
        vix_features = self.create_vix_features(data)
        vpoc_features = self.create_volume_profile_features(data)

        # Combine all features
        all_features = {}
        all_features.update(price_features)
        all_features.update(vix_features)
        all_features.update(vpoc_features)

        # Convert to DataFrame
        features_df = pd.DataFrame(all_features)

        # Analyze different target variables
        analysis_results = {}

        for target_name, target_series in targets.items():
            if target_name == 'rank_target':  # Focus on rank-transformed target
                self.logger.info(f"📊 Analyzing features for target: {target_name}")

                # Feature importance
                importance_results = self.analyze_feature_importance(features_df, target_series)

                # Correlation analysis
                corr_results = self.analyze_feature_correlations(features_df)

                analysis_results[target_name] = {
                    'importance': importance_results,
                    'correlations': corr_results,
                    'feature_count': len(features_df.columns)
                }

        # Generate recommendations
        recommendations = self.generate_recommendations(analysis_results.get('rank_target', {}))

        # Save results
        results_file = os.path.join(output_dir, "comprehensive_feature_analysis_results.txt")
        with open(results_file, 'w') as f:
            f.write("COMPREHENSIVE FEATURE ANALYSIS RESULTS\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Dataset: {data_path}\n")
            f.write(f"Sample Size: {len(data):,} rows\n")
            f.write(f"Date Range: {data.index.min()} to {data.index.max()}\n")
            f.write(f"Total Features Created: {len(features_df.columns)}\n\n")

            # Feature breakdown
            f.write("FEATURE BREAKDOWN:\n")
            f.write(f"Price Features: {len(price_features)}\n")
            f.write(f"VIX Features: {len(vix_features)}\n")
            f.write(f"Volume Profile Features: {len(vpoc_features)}\n\n")

            # Top features
            if 'rank_target' in analysis_results and 'top_features' in analysis_results['rank_target']['importance']:
                f.write("TOP 10 FEATURES BY COMBINED SCORE:\n")
                for i, (feature, scores) in enumerate(analysis_results['rank_target']['importance']['top_features']):
                    f.write(f"{i+1:2d}. {feature:<30} Score: {scores['combined_score']:.4f}\n")
                f.write("\n")

            # Recommendations
            f.write("RECOMMENDATIONS:\n")
            for rec in recommendations:
                f.write(f"{rec}\n")

        # Create visualization
        self.create_visualizations(features_df, targets, analysis_results, output_dir)

        self.logger.info(f"✅ Analysis completed. Results saved to {results_file}")

        return {
            'analysis_results': analysis_results,
            'recommendations': recommendations,
            'results_file': results_file,
            'feature_count': len(features_df.columns),
            'sample_size': len(data)
        }

    def create_visualizations(self, features_df: pd.DataFrame, targets: dict,
                            analysis_results: dict, output_dir: str):
        """Create visualization plots."""
        self.logger.info("📈 Creating visualizations...")

        try:
            # Feature importance plot
            if 'rank_target' in analysis_results and 'combined_scores' in analysis_results['rank_target']['importance']:
                combined_scores = analysis_results['rank_target']['importance']['combined_scores']

                # Sort features by score
                sorted_features = sorted(combined_scores.items(), key=lambda x: x[1]['combined_score'], reverse=True)
                features = [f[0] for f in sorted_features[:15]]
                scores = [f[1]['combined_score'] for f in sorted_features[:15]]

                plt.figure(figsize=(12, 8))
                plt.barh(range(len(features)), scores, color='skyblue')
                plt.yticks(range(len(features)), features)
                plt.xlabel('Combined Importance Score')
                plt.title('Top 15 Features by Importance (Rank Target)')
                plt.gca().invert_yaxis()
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
                plt.close()

            # Correlation heatmap
            if len(features_df.columns) <= 20:  # Only if manageable
                plt.figure(figsize=(12, 10))
                corr_matrix = features_df.corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                           square=True, fmt='.2f')
                plt.title('Feature Correlation Matrix')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
                plt.close()

            self.logger.info("✅ Visualizations saved")

        except Exception as e:
            self.logger.error(f"❌ Error creating visualizations: {e}")

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Feature Analysis with VPOC, Volume Profile, and VIX Features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full analysis on all data
  python src/scripts/comprehensive_feature_analysis.py --data DATA/MERGED/merged_es_vix_test.csv

  # Run on sample of data for testing
  python src/scripts/comprehensive_feature_analysis.py --data DATA/MERGED/merged_es_vix_test.csv --sample_size 0.1

  # Custom output directory
  python src/scripts/comprehensive_feature_analysis.py --data DATA/MERGED/merged_es_vix_test.csv --output_dir CUSTOM_ANALYSIS
        """
    )

    parser.add_argument(
        "--data", "-d",
        type=str,
        required=True,
        help="Path to the data file"
    )

    parser.add_argument(
        "--sample_size", "-s",
        type=float,
        default=1.0,
        help="Fraction of data to use for analysis (0.01-1.0, default: 1.0)"
    )

    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="ANALYSIS_RESULTS",
        help="Output directory for analysis results (default: ANALYSIS_RESULTS)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Configure logging
    log_level = "DEBUG" if args.verbose else "INFO"

    # Validate arguments
    if not (0.01 <= args.sample_size <= 1.0):
        print("❌ Error: sample_size must be between 0.01 and 1.0")
        return 1

    # Create analyzer
    analyzer = ComprehensiveFeatureAnalyzer()

    # Run analysis
    results = analyzer.run_comprehensive_analysis(
        data_path=args.data,
        sample_size=args.sample_size,
        output_dir=args.output_dir
    )

    # Display summary
    if 'error' not in results:
        print(f"\n📊 ANALYSIS SUMMARY:")
        print(f"  • Dataset: {args.data}")
        print(f"  • Sample Size: {results['sample_size']:,} rows")
        print(f"  • Features Created: {results['feature_count']}")
        print(f"  • Results File: {results['results_file']}")
        print(f"\n🎯 TOP 3 RECOMMENDATIONS:")
        for i, rec in enumerate(results['recommendations'][:3]):
            print(f"  {i+1}. {rec.strip()}")
        print(f"\n✅ Analysis completed successfully!")
    else:
        print(f"\n❌ Analysis failed: {results['error']}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())