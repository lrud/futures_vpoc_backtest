#!/usr/bin/env python3
"""
Run comprehensive Phase 1 feature analysis using existing analysis tools.
Uses src/ml/feature_analysis.py and src/ml/target_analysis.py for statistical analysis.
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

# Add project root to path
sys.path.append('/workspace')

from src.core.data import FuturesDataManager
from src.ml.feature_engineering import FeatureEngineer
from src.ml.feature_analysis import quick_feature_analysis, FeatureAnalyzer
from src.ml.target_analysis import analyze_target_for_training, TargetAnalyzer
from src.utils.logging import get_logger

logger = get_logger(__name__)

def main():
    """Main analysis function using existing analysis tools."""
    logger.info("🚀 Starting Phase 1 Comprehensive Analysis using Existing Tools...")

    # Phase 1 core features (from TRAINING_REDESIGN_PLAN.md)
    PHASE1_CORE_FEATURES = [
        # Price action (5 features)
        'close_change_pct', 'price_range', 'close_to_vwap_pct',
        'volatility_5d', 'volume_trend_5d',

        # Basic momentum (3 features)
        'price_mom_3d', 'price_mom_5d', 'price_mom_10d',

        # Volume profile (4 features)
        'vpoc_change_3d', 'vpoc_zscore_5d', 'va_width_5d_ma', 'total_volume',

        # Market context (2 features)
        'session_close', 'vwap'
    ]

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

        # Create feature engineer in Phase 1 mode
        feature_engineer = FeatureEngineer(phase1_mode=True)

        # Process data with Phase 1 to get features and target
        logger.info("🔧 Processing data with Phase 1 feature engineering...")

        # Use a sample for initial analysis to manage memory
        sample_size = min(50000, len(data))  # 50k sample or full dataset if smaller
        sample_data = data.head(sample_size)

        logger.info(f"Using sample of {len(sample_data):,} rows for analysis")

        # Get Phase 1 features and targets
        try:
            X, y, feature_columns = feature_engineer._load_and_prepare_data_phase1(
                data_path=sample_data,
                data_fraction=1.0,
                target_type='raw',
                scaling_method='robust'
            )[:3]  # Get first 3 elements (X, y, feature_columns)

        except Exception as e:
            logger.error(f"Phase 1 feature processing failed: {e}")
            # Fallback: try to identify features from existing columns
            logger.info("🔄 Attempting fallback feature identification...")

            # Find available features that match our Phase 1 list
            available_features = []
            for feature in PHASE1_CORE_FEATURES:
                matching_cols = [col for col in data.columns if feature.lower() in col.lower()]
                if matching_cols:
                    available_features.extend(matching_cols)
                    logger.info(f"Found feature: {feature} → {matching_cols}")

            # Remove duplicates
            available_features = list(dict.fromkeys(available_features))

            if not available_features:
                logger.error("❌ No Phase 1 features found in data!")
                return

            logger.info(f"✅ Identified {len(available_features)} Phase 1 features")

            # Create feature matrix
            feature_data = data[available_features].dropna()
            X = feature_data.values
            feature_columns = available_features

            # Create target (next period returns)
            if 'close' in data.columns:
                data['target'] = data.groupby('symbol')['close'].pct_change().shift(-1)
                target_data = data['target'].dropna()
                y = target_data.values
                # Align features and target
                min_len = min(len(X), len(y))
                X = X[:min_len]
                y = y[:min_len]
            else:
                logger.error("❌ No 'close' column found for target creation!")
                return

        logger.info(f"✅ Feature matrix shape: {X.shape}")
        logger.info(f"✅ Target vector shape: {y.shape}")

        # Convert to DataFrames for analysis
        features_df = pd.DataFrame(X, columns=feature_columns)
        target_series = pd.Series(y)

        # Run existing comprehensive analysis
        logger.info("📊 Running Feature Analysis using existing tools...")

        # Feature analysis using existing tool
        feature_results = quick_feature_analysis(features_df, target_series)

        logger.info("🎯 Running Target Analysis using existing tools...")

        # Target analysis using existing tool
        target_results = analyze_target_for_training(target_series, "phase1_raw_returns")

        # Additional detailed analysis using the full analyzer classes
        logger.info("🔍 Running Detailed Feature Correlation Analysis...")
        feature_analyzer = FeatureAnalyzer()
        correlation_results = feature_analyzer.analyze_feature_correlations(features_df, target_series)

        logger.info("🎯 Running Detailed Feature Importance Analysis...")
        importance_results = feature_analyzer.analyze_feature_importance(
            X, y, feature_columns, methods=['rf', 'mi', 'permutation']
        )

        logger.info("📈 Running Detailed Target Distribution Analysis...")
        target_analyzer = TargetAnalyzer()
        detailed_target_results = target_analyzer.analyze_target_distribution(target_series, "phase1_target")

        # Generate comprehensive summary
        logger.info("=" * 80)
        logger.info("📋 PHASE 1 COMPREHENSIVE ANALYSIS SUMMARY")
        logger.info("=" * 80)

        # Feature summary
        logger.info("🔧 FEATURE ENGINEERING SUMMARY:")
        logger.info(f"  • Total features generated: {len(feature_columns)}")
        logger.info(f"  • Feature matrix shape: {X.shape}")
        logger.info(f"  • Target vector shape: {y.shape}")
        logger.info(f"  • Features: {', '.join(feature_columns[:10])}{'...' if len(feature_columns) > 10 else ''}")

        # Correlation summary
        high_corr_pairs = correlation_results.get('high_correlation_pairs', [])
        vif_issues = correlation_results.get('multicollinearity_issues', [])

        logger.info("🔗 CORRELATION ANALYSIS SUMMARY:")
        logger.info(f"  • High correlation pairs: {len(high_corr_pairs)}")
        logger.info(f"  • Multicollinearity issues (VIF > 10): {len([v for v in vif_issues if v['severity'] == 'high'])}")

        if high_corr_pairs:
            logger.info("  • Top correlations:")
            for pair in high_corr_pairs[:3]:
                logger.info(f"    - {pair['feature1']} ↔ {pair['feature2']}: {pair['correlation']:.3f}")

        # Importance summary
        consensus_ranking = importance_results.get('consensus_ranking', [])
        if consensus_ranking:
            logger.info("🎯 FEATURE IMPORTANCE SUMMARY:")
            logger.info("  • Top 5 most important features:")
            for i, (feature, score) in enumerate(consensus_ranking[:5], 1):
                logger.info(f"    {i}. {feature}: {score:.4f}")

        # Target summary
        target_basic_stats = detailed_target_results.get('basic_stats', {})
        if target_basic_stats:
            logger.info("🎯 TARGET DISTRIBUTION SUMMARY:")
            logger.info(f"  • Mean: {target_basic_stats['mean']:.6f}")
            logger.info(f"  • Std Dev: {target_basic_stats['std']:.6f}")
            logger.info(f"  • Range: [{target_basic_stats['min']:.6f}, {target_basic_stats['max']:.6f}]")
            logger.info(f"  • Skewness: {target_basic_stats['skewness']:.4f}")
            logger.info(f"  • Kurtosis: {target_basic_stats['kurtosis']:.4f}")

        # Outlier summary
        outlier_analysis = detailed_target_results.get('outlier_analysis', {})
        if outlier_analysis:
            z_outliers = outlier_analysis.get('zscore_method', {}).get('outlier_percentage', 0)
            iqr_outliers = outlier_analysis.get('iqr_method', {}).get('outlier_percentage', 0)
            logger.info("🚨 OUTLIER ANALYSIS SUMMARY:")
            logger.info(f"  • Z-score outliers (>3σ): {z_outliers:.1f}%")
            logger.info(f"  • IQR outliers: {iqr_outliers:.1f}%")

        # Recommendations from all analyses
        logger.info("💡 COMBINED RECOMMENDATIONS:")

        feature_recommendations = correlation_results.get('recommendations', [])
        target_recommendations = detailed_target_results.get('recommendations', [])

        all_recommendations = feature_recommendations + target_recommendations
        unique_recommendations = list(set(all_recommendations))  # Remove duplicates

        for i, rec in enumerate(unique_recommendations[:10], 1):  # Top 10 recommendations
            logger.info(f"  {i}. {rec}")

        # Final assessment
        logger.info("=" * 80)
        logger.info("🎯 PHASE 1 TRAINING READINESS ASSESSMENT:")

        issues_found = 0

        if len(high_corr_pairs) > len(feature_columns) * 0.2:
            logger.warning(f"  ⚠️ High multicollinearity detected ({len(high_corr_pairs)} pairs)")
            issues_found += 1

        if z_outliers > 10:
            logger.warning(f"  ⚠️ High outlier percentage in target ({z_outliers:.1f}%)")
            issues_found += 1

        if abs(target_basic_stats.get('skewness', 0)) > 2:
            logger.warning(f"  ⚠️ High target skewness ({target_basic_stats.get('skewness', 0):.2f})")
            issues_found += 1

        if issues_found == 0:
            logger.info("  ✅ No major issues detected - READY FOR PHASE 1 TRAINING")
            logger.info("  ✅ Features appear suitable for stable neural network training")
            logger.info("  ✅ Target distribution is reasonable")
        else:
            logger.warning(f"  ⚠️ Found {issues_found} potential issues - see recommendations above")
            logger.info("  📝 Address issues before training or proceed with caution")

        logger.info("=" * 80)
        logger.info("✅ Phase 1 Comprehensive Analysis Completed Successfully!")

        return {
            'feature_analysis': feature_results,
            'target_analysis': target_results,
            'correlation_analysis': correlation_results,
            'importance_analysis': importance_results,
            'detailed_target_analysis': detailed_target_results,
            'features_df': features_df,
            'target_series': target_series
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