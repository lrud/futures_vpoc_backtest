#!/usr/bin/env python3
"""
Efficient feature analysis using existing tools on available data.
Optimized for large datasets (1.14M rows) with chunked processing and GPU acceleration where beneficial.
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import gc
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append('/workspace')

from src.core.data import FuturesDataManager
from src.ml.deprecated.feature_analysis import quick_feature_analysis, FeatureAnalyzer
from src.ml.deprecated.target_analysis import analyze_target_for_training, TargetAnalyzer
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Try to import GPU-accelerated libraries
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    if GPU_AVAILABLE:
        logger.info(f"🚀 GPU acceleration available: {torch.cuda.device_count()} devices")
except ImportError:
    GPU_AVAILABLE = False

def create_enhanced_features_with_vix_vpoc(data: pd.DataFrame, chunk_size: int = 100000) -> pd.DataFrame:
    """Create enhanced features including VIX, VPOC, and comprehensive volume profile analysis."""
    logger.info(f"🔧 Creating enhanced features with VIX, VPOC, and volume profile (chunk_size={chunk_size})...")

    features = pd.DataFrame(index=data.index)

    # Process in chunks to manage memory for large datasets
    total_chunks = (len(data) + chunk_size - 1) // chunk_size

    for chunk_idx in range(total_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(data))
        chunk_data = data.iloc[start_idx:end_idx].copy()

        chunk_features = pd.DataFrame(index=chunk_data.index)

        # === BASIC PRICE FEATURES ===
        chunk_features['close_change_pct'] = chunk_data['close'].pct_change()
        chunk_features['price_range'] = (chunk_data['high'] - chunk_data['low']) / chunk_data['close']
        chunk_features['volatility_5d'] = chunk_features['close_change_pct'].rolling(5).std()
        chunk_features['volatility_10d'] = chunk_features['close_change_pct'].rolling(10).std()
        chunk_features['volatility_20d'] = chunk_features['close_change_pct'].rolling(20).std()

        # === MOMENTUM FEATURES ===
        chunk_features['price_mom_1d'] = chunk_data['close'].pct_change(1)
        chunk_features['price_mom_3d'] = chunk_data['close'].pct_change(3)
        chunk_features['price_mom_5d'] = chunk_data['close'].pct_change(5)
        chunk_features['price_mom_10d'] = chunk_data['close'].pct_change(10)

        # === VWAP FEATURES ===
        typical_price = (chunk_data['high'] + chunk_data['low'] + chunk_data['close']) / 3
        chunk_features['vwap'] = typical_price
        chunk_features['close_to_vwap_pct'] = (chunk_data['close'] - typical_price) / typical_price

        # === VOLUME PROFILE & VPOC FEATURES ===
        if 'volume' in chunk_data.columns:
            chunk_features['total_volume'] = chunk_data['volume']
            chunk_features['volume_trend_5d'] = chunk_features['total_volume'].pct_change(5)
            chunk_features['volume_ma_5d'] = chunk_features['total_volume'].rolling(5).mean()
            chunk_features['volume_ma_10d'] = chunk_features['total_volume'].rolling(10).mean()

            # Volume-Price Analysis
            chunk_features['volume_price_trend'] = (chunk_features['total_volume'] * chunk_features['close_change_pct']).rolling(5).sum()
            chunk_features['volume_weighted_price'] = (typical_price * chunk_features['total_volume']).rolling(10).sum() / chunk_features['total_volume'].rolling(10).sum()

            # VPOC (Volume Point of Control) - Approximation using volume-weighted price
            chunk_features['vpoc_approx'] = chunk_features['volume_weighted_price']
            chunk_features['close_to_vpoc_pct'] = (chunk_data['close'] - chunk_features['vpoc_approx']) / chunk_features['vpoc_approx']
            chunk_features['vpoc_distance'] = abs(chunk_data['close'] - chunk_features['vpoc_approx'])

            # Volume Profile Value Area Width
            price_range_vol = chunk_data['high'] - chunk_data['low']
            chunk_features['va_width_5d'] = price_range_vol.rolling(5).mean()
            chunk_features['va_width_10d'] = price_range_vol.rolling(10).mean()
            chunk_features['va_width_trend'] = (chunk_features['va_width_5d'] - chunk_features['va_width_10d']) / chunk_features['va_width_10d']

            # Volume Accumulation/Distribution
            chunk_features['vol_acc_dist'] = ((chunk_data['close'] - chunk_data['low']) - (chunk_data['high'] - chunk_data['close'])) / (chunk_data['high'] - chunk_data['low']) * chunk_features['total_volume']
            chunk_features['vol_acc_dist_5d_sum'] = chunk_features['vol_acc_dist'].rolling(5).sum()
            chunk_features['vol_acc_dist_10d_sum'] = chunk_features['vol_acc_dist'].rolling(10).sum()

            # Price-Volume Correlation Features
            chunk_features['price_volume_corr_5d'] = chunk_features['close_change_pct'].rolling(5).corr(chunk_features['total_volume'].pct_change())
            chunk_features['price_volume_corr_10d'] = chunk_features['close_change_pct'].rolling(10).corr(chunk_features['total_volume'].pct_change())

            # Volume Relative Strength
            chunk_features['volume_relative_strength'] = chunk_features['total_volume'] / chunk_features['total_volume'].rolling(20).mean()

            # On-Balance Volume (OBV)
            obv = np.where(chunk_features['close_change_pct'] > 0, chunk_features['total_volume'],
                          np.where(chunk_features['close_change_pct'] < 0, -chunk_features['total_volume'], 0))
            chunk_features['obv'] = pd.Series(obv, index=chunk_data.index).cumsum()
            chunk_features['obv_ma_5d'] = chunk_features['obv'].rolling(5).mean()
            chunk_features['obv_trend'] = (chunk_features['obv'] - chunk_features['obv_ma_5d']) / chunk_features['obv_ma_5d']
        else:
            logger.warning("Volume data not available - creating dummy volume features")
            chunk_features['total_volume'] = 1.0
            chunk_features['volume_trend_5d'] = 0.0
            chunk_features['vpoc_distance'] = 0.0

        # === VIX FEATURES ===
        if 'vix' in chunk_data.columns:
            chunk_features['vix_level'] = chunk_data['vix']
            chunk_features['vix_change_1d'] = chunk_data['vix'].pct_change(1)
            chunk_features['vix_change_3d'] = chunk_data['vix'].pct_change(3)
            chunk_features['vix_change_5d'] = chunk_data['vix'].pct_change(5)

            # VIX moving averages
            chunk_features['vix_ma_5d'] = chunk_data['vix'].rolling(5).mean()
            chunk_features['vix_ma_10d'] = chunk_data['vix'].rolling(10).mean()
            chunk_features['vix_ma_20d'] = chunk_data['vix'].rolling(20).mean()

            # VIX relative to moving averages
            chunk_features['vix_vs_ma5'] = (chunk_data['vix'] - chunk_features['vix_ma_5d']) / chunk_features['vix_ma_5d']
            chunk_features['vix_vs_ma10'] = (chunk_data['vix'] - chunk_features['vix_ma_10d']) / chunk_features['vix_ma_10d']

            # VIX volatility and z-score
            chunk_features['vix_volatility_5d'] = chunk_features['vix_change_1d'].rolling(5).std()
            chunk_features['vix_zscore_20d'] = (chunk_data['vix'] - chunk_features['vix_ma_20d']) / chunk_features['vix_ma_20d'].rolling(20).std()

            # VIX percentiles (fear/regime indicators)
            chunk_features['vix_percentile_20d'] = chunk_data['vix'].rolling(20).rank(pct=True)
            chunk_features['vix_percentile_60d'] = chunk_data['vix'].rolling(60).rank(pct=True)

            # VIX extreme values
            vix_20d_high = chunk_data['vix'].rolling(20).max()
            vix_20d_low = chunk_data['vix'].rolling(20).min()
            chunk_features['vix_extreme_high'] = (chunk_data['vix'] - vix_20d_high) / (vix_20d_high - vix_20d_low)
            chunk_features['vix_extreme_low'] = (chunk_data['vix'] - vix_20d_low) / (vix_20d_high - vix_20d_low)

            # VIX trend strength
            chunk_features['vix_trend_strength'] = abs(chunk_features['vix_vs_ma5'])

            # VIX-Price relationship
            chunk_features['vix_price_correlation_5d'] = chunk_features['close_change_pct'].rolling(5).corr(chunk_features['vix_change_1d'])
            chunk_features['vix_price_correlation_10d'] = chunk_features['close_change_pct'].rolling(10).corr(chunk_features['vix_change_1d'])

            # Fear/Greed Index approximation
            chunk_features['fear_greed_index'] = 1 - chunk_features['vix_percentile_20d']  # Inverted: high VIX = fear

        else:
            logger.warning("VIX data not available - creating dummy VIX features")
            chunk_features['vix_level'] = 20.0  # Typical VIX level
            chunk_features['vix_change_1d'] = 0.0

        # === TECHNICAL INDICATORS ===
        # RSI approximation
        gains = chunk_features['close_change_pct'].clip(lower=0)
        losses = -chunk_features['close_change_pct'].clip(upper=0)
        avg_gains = gains.rolling(14).mean()
        avg_losses = losses.rolling(14).mean()
        rs = avg_gains / avg_losses
        chunk_features['rsi_14'] = 100 - (100 / (1 + rs))

        # Bollinger Band position
        bb_ma = chunk_data['close'].rolling(20).mean()
        bb_std = chunk_data['close'].rolling(20).std()
        chunk_features['bb_position'] = (chunk_data['close'] - bb_ma) / (2 * bb_std)

        # Price position in recent range
        chunk_features['price_position_5d'] = (chunk_data['close'] - chunk_data['low'].rolling(5).min()) / (chunk_data['high'].rolling(5).max() - chunk_data['low'].rolling(5).min())
        chunk_features['price_position_20d'] = (chunk_data['close'] - chunk_data['low'].rolling(20).min()) / (chunk_data['high'].rolling(20).max() - chunk_data['low'].rolling(20).min())

        # Merge chunk features
        features = pd.concat([features, chunk_features], ignore_index=False)

        # Progress update
        if (chunk_idx + 1) % 5 == 0 or chunk_idx == total_chunks - 1:
            logger.info(f"  Processed chunk {chunk_idx + 1}/{total_chunks} ({end_idx:,}/{len(data):,} rows)")

        # Force garbage collection
        del chunk_data, chunk_features
        gc.collect()

    # Clean up infinite and NaN values
    logger.info("🧹 Cleaning up infinite and NaN values...")
    features = features.replace([np.inf, -np.inf], np.nan)

    # Drop columns with too many NaN values
    nan_threshold = 0.1  # Drop columns with >10% NaN values
    cols_to_drop = []
    for col in features.columns:
        nan_pct = features[col].isna().sum() / len(features)
        if nan_pct > nan_threshold:
            logger.info(f"  Dropping {col}: {nan_pct*100:.1f}% NaN values")
            cols_to_drop.append(col)

    if cols_to_drop:
        features = features.drop(columns=cols_to_drop)

    # Final NaN drop for remaining rows
    initial_rows = len(features)
    features = features.dropna()
    final_rows = len(features)

    logger.info(f"✅ Created {len(features.columns)} enhanced features")
    logger.info(f"📊 Feature categories:")

    feature_categories = {
        'Price momentum': [col for col in features.columns if 'mom' in col or 'change' in col],
        'Volatility': [col for col in features.columns if 'volatil' in col or 'std' in col],
        'Volume profile': [col for col in features.columns if 'vpoc' in col or 'va_' in col or 'volume' in col],
        'VIX features': [col for col in features.columns if 'vix' in col],
        'Technical indicators': [col for col in features.columns if any(x in col for x in ['rsi', 'bb_', 'position'])]
    }

    for category, cols in feature_categories.items():
        if cols:
            logger.info(f"  • {category}: {len(cols)} features")

    logger.info(f"✅ Data retention: {final_rows:,}/{initial_rows:,} ({(final_rows/initial_rows)*100:.1f}%)")

    return features

def efficient_correlation_analysis(X: np.ndarray, feature_names: List[str],
                                 sample_size: int = 50000) -> Dict:
    """Efficient correlation analysis using sampling."""
    logger.info(f"🔗 Computing correlations with {sample_size:,} sample size...")

    # Sample for efficiency
    if len(X) > sample_size:
        sample_indices = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[sample_indices]
    else:
        X_sample = X

    # Compute correlation matrix
    corr_matrix = np.corrcoef(X_sample.T)

    # Find high correlations
    high_corr_pairs = []
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            if abs(corr_matrix[i, j]) > 0.8:
                high_corr_pairs.append({
                    'feature1': feature_names[i],
                    'feature2': feature_names[j],
                    'correlation': corr_matrix[i, j]
                })

    logger.info(f"  Found {len(high_corr_pairs)} high correlation pairs (>0.8)")

    return {
        'correlation_matrix': corr_matrix,
        'high_correlation_pairs': high_corr_pairs,
        'sample_size': len(X_sample)
    }

def main():
    """Main analysis function with efficient processing for large datasets."""
    logger.info("🚀 Starting EFFICIENT Full Dataset Feature Analysis...")

    try:
        # Load data
        data_path = "/workspace/DATA/MERGED/merged_es_vix_test.csv"
        logger.info(f"Loading data from: {data_path}")

        data_manager = FuturesDataManager()
        data = data_manager.load_futures_data_file(data_path)

        if data is None or data.empty:
            logger.error("❌ No data loaded!")
            return

        logger.info(f"✅ Loaded data: {len(data):,} rows, {len(data.columns)} columns")

        # Dependent variable clarification
        logger.info("=" * 80)
        logger.info("🎯 DEPENDENT VARIABLE CLARIFICATION:")
        logger.info("  • We are modeling: NEXT PERIOD PRICE RETURNS")
        logger.info("  • Target definition: close[t+1] / close[t] - 1 (1-period forward return)")
        logger.info("  • This is a REGRESSION problem predicting magnitude and direction")
        logger.info("  • Using ENTIRE dataset for comprehensive statistical analysis")
        logger.info("=" * 80)

        # Create enhanced features including VIX, VPOC, and volume profile
        features_df = create_enhanced_features_with_vix_vpoc(data, chunk_size=100000)

        # Create target (next period returns) - our dependent variable
        logger.info("🎯 Creating target variable...")
        target_series = data['close'].pct_change().shift(-1).dropna()
        logger.info(f"✅ Target variable created: {len(target_series):,} observations")
        logger.info("  • Target type: Raw price returns (not log returns, not volatility-adjusted)")
        logger.info("  • Time horizon: 1 period forward prediction")

        # Align features and target
        common_index = features_df.index.intersection(target_series.index)
        features_df = features_df.loc[common_index]
        target_series = target_series.loc[common_index]

        logger.info(f"✅ Final analysis dataset:")
        logger.info(f"  • Feature matrix shape: {features_df.shape}")
        logger.info(f"  • Target vector shape: {target_series.shape}")
        logger.info(f"  • Features: {', '.join(features_df.columns)}")

        # Convert to numpy for efficiency
        X = features_df.values
        y = target_series.values
        feature_names = features_df.columns.tolist()

        # Memory-efficient analyses
        logger.info("📊 Starting EFFICIENT ANALYSIS PIPELINE...")

        # 1. Basic Statistics (memory efficient)
        logger.info("📈 Computing basic statistics...")
        basic_stats = {
            'features': {
                'means': np.mean(X, axis=0),
                'stds': np.std(X, axis=0),
                'mins': np.min(X, axis=0),
                'maxs': np.max(X, axis=0),
                'skews': stats.skew(X, axis=0),
                'kurtosis': stats.kurtosis(X, axis=0)
            },
            'target': {
                'mean': np.mean(y),
                'std': np.std(y),
                'min': np.min(y),
                'max': np.max(y),
                'skewness': stats.skew(y),
                'kurtosis': stats.kurtosis(y),
                'length': len(y)
            }
        }

        logger.info(f"  ✅ Computed statistics for {len(X):,} samples")

        # 2. Efficient Correlation Analysis
        correlation_results = efficient_correlation_analysis(X, feature_names, sample_size=50000)

        # 3. Target Distribution Analysis
        logger.info("🎯 Analyzing target distribution...")
        z_scores = np.abs(stats.zscore(y))
        z_outliers = np.sum(z_scores > 3)
        z_outlier_percentage = (z_outliers / len(y)) * 100

        Q1 = np.percentile(y, 25)
        Q3 = np.percentile(y, 75)
        IQR = Q3 - Q1
        iqr_outliers = np.sum((y < Q1 - 1.5*IQR) | (y > Q3 + 1.5*IQR))
        iqr_outlier_percentage = (iqr_outliers / len(y)) * 100

        outlier_analysis = {
            'zscore_method': {
                'outlier_count': int(z_outliers),
                'outlier_percentage': z_outlier_percentage,
                'threshold': 3.0
            },
            'iqr_method': {
                'outlier_count': int(iqr_outliers),
                'outlier_percentage': iqr_outlier_percentage,
                'q1': Q1,
                'q3': Q3,
                'iqr': IQR
            }
        }

        logger.info(f"  ✅ Z-score outliers: {z_outlier_percentage:.2f}%, IQR outliers: {iqr_outlier_percentage:.2f}%")

        # 4. Efficient Feature Importance (using smaller samples)
        logger.info("🎯 Computing feature importance...")

        # Use existing tools but with smaller samples for efficiency
        sample_size_for_ml = min(20000, len(X))
        sample_indices = np.random.choice(len(X), sample_size_for_ml, replace=False)

        X_sample = X[sample_indices]
        y_sample = y[sample_indices]
        features_sample_df = features_df.iloc[sample_indices]
        target_sample_series = target_series.iloc[sample_indices]

        logger.info(f"  Using {sample_size_for_ml:,} samples for ML-based analysis...")

        # Run existing tools on samples
        logger.info("  Running feature importance on sample...")
        feature_analyzer = FeatureAnalyzer()
        importance_results = feature_analyzer.analyze_feature_importance(
            X_sample, y_sample, feature_names, methods=['rf', 'mi']
        )

        logger.info("  Running target analysis on sample...")
        target_analyzer = TargetAnalyzer()
        target_results = analyze_target_for_training(target_sample_series, "efficient_returns")

        # Clean up memory
        del X_sample, y_sample, features_sample_df, target_sample_series
        gc.collect()

        # 5. Data Quality Assessment
        logger.info("✅ Assessing data quality...")
        missing_values = np.sum(np.isnan(X), axis=0)
        infinite_values = np.sum(np.isinf(X), axis=0)
        missing_target = np.sum(np.isnan(y))
        infinite_target = np.sum(np.isinf(y))

        data_quality = {
            'missing_values': missing_values.tolist(),
            'infinite_values': infinite_values.tolist(),
            'missing_target': int(missing_target),
            'infinite_target': int(infinite_target),
            'total_samples': len(X),
            'total_features': len(feature_names)
        }

        logger.info(f"  ✅ Data quality - Missing: {np.sum(missing_values)}, Infinite: {np.sum(infinite_values)}")

        # Generate comprehensive summary
        logger.info("=" * 80)
        logger.info("📋 EFFICIENT FULL DATASET ANALYSIS SUMMARY")
        logger.info("=" * 80)

        # Feature summary
        logger.info("🔧 DATASET OVERVIEW:")
        logger.info(f"  • Total samples analyzed: {len(X):,}")
        logger.info(f"  • Total features: {len(feature_names)}")
        logger.info(f"  • Feature matrix shape: {X.shape}")
        logger.info(f"  • Target vector shape: {y.shape}")
        logger.info(f"  • Features: {', '.join(feature_names)}")

        # Target statistics
        target_stats = basic_stats['target']
        logger.info("🎯 TARGET VARIABLE STATISTICS:")
        logger.info(f"  • Mean: {target_stats['mean']:.8f}")
        logger.info(f"  • Std Dev: {target_stats['std']:.8f}")
        logger.info(f"  • Range: [{target_stats['min']:.8f}, {target_stats['max']:.8f}]")
        logger.info(f"  • Skewness: {target_stats['skewness']:.4f}")
        logger.info(f"  • Kurtosis: {target_stats['kurtosis']:.4f}")

        # Feature statistics summary
        feature_stats = basic_stats['features']
        logger.info("🔧 FEATURE STATISTICS SUMMARY:")
        for i, name in enumerate(feature_names):
            logger.info(f"  • {name}: mean={feature_stats['means'][i]:.6f}, std={feature_stats['stds'][i]:.6f}")

        # Correlation analysis
        high_corr_pairs = correlation_results.get('high_correlation_pairs', [])
        logger.info("🔗 CORRELATION ANALYSIS:")
        logger.info(f"  • Sample size: {correlation_results['sample_size']:,}")
        logger.info(f"  • High correlation pairs (>0.8): {len(high_corr_pairs)}")
        if high_corr_pairs:
            logger.info("  • Top correlations:")
            for pair in high_corr_pairs[:5]:
                logger.info(f"    - {pair['feature1']} ↔ {pair['feature2']}: {pair['correlation']:.3f}")

        # Outlier analysis
        logger.info("🚨 OUTLIER ANALYSIS:")
        logger.info(f"  • Z-score outliers (>3σ): {outlier_analysis['zscore_method']['outlier_percentage']:.2f}%")
        logger.info(f"  • IQR outliers: {outlier_analysis['iqr_method']['outlier_percentage']:.2f}%")

        # Feature importance
        consensus_ranking = importance_results.get('consensus_ranking', [])
        if consensus_ranking:
            logger.info("🎯 FEATURE IMPORTANCE (from {sample_size:,} samples):".format(sample_size=sample_size_for_ml))
            logger.info("  • Top 10 most important features:")
            for i, (feature, score) in enumerate(consensus_ranking[:10], 1):
                logger.info(f"    {i}. {feature}: {score:.4f}")

        # Data quality
        logger.info("✅ DATA QUALITY:")
        logger.info(f"  • Missing values: {np.sum(data_quality['missing_values'])}")
        logger.info(f"  • Infinite values: {np.sum(data_quality['infinite_values'])}")
        logger.info(f"  • Target completeness: {len(y) - data_quality['missing_target'] - data_quality['infinite_target']}/{len(y)}")

        # Training recommendations
        logger.info("💡 TRAINING READINESS RECOMMENDATIONS:")

        recommendations = []

        # Correlation issues
        if len(high_corr_pairs) > 0:
            recommendations.append(f"⚠️ Address {len(high_corr_pairs)} highly correlated feature pairs")

        # Outlier issues
        if outlier_analysis['zscore_method']['outlier_percentage'] > 5:
            recommendations.append(f"⚠️ High outlier percentage ({outlier_analysis['zscore_method']['outlier_percentage']:.1f}%) - consider target clipping")

        # Target distribution
        if abs(target_stats['skewness']) > 1:
            recommendations.append(f"⚠️ Target is skewed ({target_stats['skewness']:.2f}) - consider transformation")

        # Stability recommendations based on our gradient explosion history
        recommendations.append("🔧 Use conservative learning rates (≤1e-5) due to gradient explosion history")
        recommendations.append("🔧 Implement strong gradient clipping (≤0.5)")
        recommendations.append("🔧 Consider batch normalization for training stability")
        recommendations.append("🔧 Start with Phase 1 simplified 11-feature approach")

        for i, rec in enumerate(recommendations, 1):
            logger.info(f"  {i}. {rec}")

        # Training readiness assessment
        logger.info("=" * 80)
        logger.info("🎯 TRAINING READINESS ASSESSMENT:")

        issues_found = 0

        if len(high_corr_pairs) > 0:
            logger.warning(f"  ⚠️ Correlation issues detected ({len(high_corr_pairs)} pairs)")
            issues_found += 1

        if outlier_analysis['zscore_method']['outlier_percentage'] > 5:
            logger.warning(f"  ⚠️ High outlier percentage ({outlier_analysis['zscore_method']['outlier_percentage']:.1f}%)")
            issues_found += 1

        if abs(target_stats['skewness']) > 1:
            logger.warning(f"  ⚠️ Target skewness detected ({target_stats['skewness']:.2f})")
            issues_found += 1

        if issues_found == 0:
            logger.info("  ✅ No major issues detected - READY FOR TRAINING")
            logger.info("  ✅ Dataset appears suitable for stable neural network training")
            logger.info("  ✅ Target distribution is reasonable")
        else:
            logger.warning(f"  ⚠️ Found {issues_found} potential issues - see recommendations above")
            logger.info("  📝 Address issues before optimal training, but can proceed cautiously")

        logger.info("=" * 80)
        logger.info("✅ EFFICIENT FULL DATASET ANALYSIS COMPLETED!")
        logger.info("=" * 80)

        return {
            'basic_stats': basic_stats,
            'correlation_analysis': correlation_results,
            'outlier_analysis': outlier_analysis,
            'importance_analysis': importance_results,
            'target_analysis': target_results,
            'data_quality': data_quality,
            'features_df': features_df,
            'target_series': target_series,
            'analysis_metadata': {
                'total_samples': len(X),
                'total_features': len(feature_names),
                'correlation_sample_size': correlation_results['sample_size'],
                'ml_sample_size': sample_size_for_ml,
                'chunk_size': 100000,
                'analysis_method': 'efficient_chunked_processing'
            }
        }

    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

if __name__ == "__main__":
    results = main()
    if results:
        logger.info("🎉 All analyses completed successfully!")
    else:
        logger.error("💥 Analysis failed!")