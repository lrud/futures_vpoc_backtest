#!/usr/bin/env python3
"""
GPU-Only Phase 1 Feature Statistical Analysis using ROCm 7

Uses PyTorch tensors on GPU for all heavy computations to prevent CPU crashes.
Analyzes RSI, MACD, Stochastic, ATR, Bollinger Bands, and Time features.
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Set GPU environment variables
os.environ['HIP_VISIBLE_DEVICES'] = '0,1'
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx1100'

# Add project root to path
sys.path.append('/workspace')

from src.utils.logging import get_logger

# GPU Libraries
import torch
import talib

logger = get_logger(__name__)

class GPUPhase1Analyzer:
    """GPU-accelerated Phase 1 feature analysis using ROCm 7."""

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data = None
        self.target = None
        self.features = {}
        self.device = None
        self.use_gpu = False

        # Initialize GPU
        self._initialize_gpu()

    def _initialize_gpu(self):
        """Initialize GPU with memory management."""
        try:
            if torch.cuda.is_available():
                # Use first GPU
                self.device = torch.device('cuda:0')
                self.use_gpu = True

                # Clear GPU cache
                torch.cuda.empty_cache()

                # Get GPU memory info
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info(f"🚀 GPU Initialized: {torch.cuda.get_device_name(0)}")
                logger.info(f"🎮 GPU Memory: {gpu_memory:.1f} GB")
                logger.info(f"✅ Using GPU for statistical analysis")
            else:
                raise RuntimeError("GPU not available")
        except Exception as e:
            logger.error(f"❌ GPU initialization failed: {e}")
            raise RuntimeError("GPU is required for this analysis to prevent CPU crashes")

    def load_data_to_gpu(self):
        """Load data directly to GPU memory."""
        logger.info(f"📁 Loading data from {self.data_path}")

        # Load data in chunks to manage memory
        logger.info("📊 Loading data with chunked reading...")

        # Read CSV in chunks
        chunk_size = 100000
        chunks = []
        for chunk in pd.read_csv(self.data_path, chunksize=chunk_size):
            chunks.append(chunk)
            if len(chunks) >= 10:  # Limit to ~1M rows to manage GPU memory
                break

        self.data = pd.concat(chunks, ignore_index=True)
        logger.info(f"✅ Loaded {len(self.data):,} rows")

        # Basic processing on CPU (minimal)
        if 'VIX' in self.data.columns:
            self.data = self.data.rename(columns={'VIX': 'vix'})

        # Remove essential missing data
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

        # Move target to GPU
        self.target_gpu = torch.tensor(self.target.values, dtype=torch.float32, device=self.device)
        logger.info(f"✅ Target moved to GPU: {self.target_gpu.shape}")

    def create_phase1_technical_indicators(self):
        """Create Phase 1 technical indicators using GPU acceleration."""
        logger.info("📈 Creating Phase 1 technical indicators on GPU...")
        try:
            # Use numpy arrays first, then move to GPU
            close_np = self.data['close'].values
            high_np = self.data['high'].values
            low_np = self.data['low'].values

            # RSI (14)
            rsi_values = talib.RSI(close_np, timeperiod=14)
            self.features['rsi_14'] = torch.tensor(rsi_values, dtype=torch.float32, device=self.device)

            # MACD (12, 26, 9)
            macd, macd_signal, macd_hist = talib.MACD(close_np, fastperiod=12, slowperiod=26, signalperiod=9)
            self.features['macd_line'] = torch.tensor(macd, dtype=torch.float32, device=self.device)
            self.features['macd_signal'] = torch.tensor(macd_signal, dtype=torch.float32, device=self.device)
            self.features['macd_histogram'] = torch.tensor(macd_hist, dtype=torch.float32, device=self.device)

            # Stochastic Oscillator (%K 14, %D 3)
            slowk, slowd = talib.STOCH(high_np, low_np, close_np, fastk_period=14, slowk_period=3, slowd_period=3)
            self.features['stoch_k'] = torch.tensor(slowk, dtype=torch.float32, device=self.device)
            self.features['stoch_d'] = torch.tensor(slowd, dtype=torch.float32, device=self.device)

            # ATR (14)
            atr_values = talib.ATR(high_np, low_np, close_np, timeperiod=14)
            self.features['atr_14'] = torch.tensor(atr_values, dtype=torch.float32, device=self.device)

            # Bollinger Bands Position
            upper, middle, lower = talib.BBANDS(close_np, timeperiod=20, nbdevup=2, nbdevdn=2)
            band_width = upper - lower
            bb_position = np.where(band_width > 0, (close_np - middle) / band_width, 0)
            self.features['bb_position'] = torch.tensor(bb_position, dtype=torch.float32, device=self.device)

            logger.info(f"✅ Created 8 Phase 1 technical indicators on GPU")

        except Exception as e:
            logger.error(f"❌ Phase 1 technical indicators failed: {e}")
            raise

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
            day_of_week = timestamp_series.dayofweek.values
            self.features['day_of_week'] = torch.tensor(day_of_week, dtype=torch.float32, device=self.device)

            # Time of day (normalized 0-1)
            time_of_day = (timestamp_series.hour / 24.0 + timestamp_series.minute / 1440.0).values
            self.features['time_of_day'] = torch.tensor(time_of_day, dtype=torch.float32, device=self.device)

            # Session indicator (if available)
            if 'session' in self.data.columns:
                session = self.data['session'].values
            else:
                session = np.ones(len(self.data))
            self.features['session_indicator'] = torch.tensor(session, dtype=torch.float32, device=self.device)

            logger.info(f"✅ Created 3 Phase 1 time features on GPU")

        except Exception as e:
            logger.error(f"❌ Phase 1 time features failed: {e}")
            raise

    def calculate_gpu_correlation(self, feature_name: str, feature_gpu: torch.Tensor):
        """Calculate correlations using GPU tensors."""
        try:
            # Align feature with target
            min_len = min(len(self.target_gpu), len(feature_gpu))
            target_clean = self.target_gpu[:min_len]
            feature_clean = feature_gpu[:min_len]

            # Remove NaN values (convert to mask)
            nan_mask = ~(torch.isnan(target_clean) | torch.isnan(feature_clean) |
                         torch.isinf(target_clean) | torch.isinf(feature_clean))

            target_clean = target_clean[nan_mask]
            feature_clean = feature_clean[nan_mask]

            if len(target_clean) < 100:
                return None

            # Calculate Pearson correlation on GPU
            target_mean = torch.mean(target_clean)
            feature_mean = torch.mean(feature_clean)

            target_centered = target_clean - target_mean
            feature_centered = feature_clean - feature_mean

            numerator = torch.sum(target_centered * feature_centered)
            target_std = torch.sqrt(torch.sum(target_centered ** 2))
            feature_std = torch.sqrt(torch.sum(feature_centered ** 2))

            denominator = target_std * feature_std

            if denominator == 0:
                pearson_corr = 0.0
                pearson_p = 1.0
            else:
                pearson_corr = numerator / denominator
                # Approximate p-value calculation
                n = len(target_clean)
                if n > 2:
                    t_stat = pearson_corr * np.sqrt((n-2) / (1 - pearson_corr**2))
                    from scipy.stats import t as t_dist
                    pearson_p = 2 * (1 - t_dist.cdf(abs(t_stat), n-2))
                else:
                    pearson_p = 1.0

            # Calculate Spearman correlation (convert to ranks on GPU)
            target_ranks = torch.argsort(torch.argsort(target_clean)).float()
            feature_ranks = torch.argsort(torch.argsort(feature_clean)).float()

            target_ranks_mean = torch.mean(target_ranks)
            feature_ranks_mean = torch.mean(feature_ranks)

            target_ranks_centered = target_ranks - target_ranks_mean
            feature_ranks_centered = feature_ranks - feature_ranks_mean

            spearman_num = torch.sum(target_ranks_centered * feature_ranks_centered)
            spearman_den = torch.sqrt(torch.sum(target_ranks_centered ** 2)) * torch.sqrt(torch.sum(feature_ranks_centered ** 2))

            if spearman_den == 0:
                spearman_corr = 0.0
                spearman_p = 1.0
            else:
                spearman_corr = spearman_num / spearman_den
                n = len(target_clean)
                if n > 2:
                    t_stat = spearman_corr * np.sqrt((n-2) / (1 - spearman_corr**2))
                    spearman_p = 2 * (1 - t_dist.cdf(abs(t_stat), n-2))
                else:
                    spearman_p = 1.0

            # Convert to CPU for results
            results = {
                'feature_name': feature_name,
                'sample_size': int(len(target_clean)),
                'feature_mean': float(feature_mean.cpu()),
                'feature_std': float(torch.std(feature_clean).cpu()),
                'pearson_corr': float(pearson_corr.cpu()),
                'pearson_p_value': float(pearson_p),
                'pearson_significant': pearson_p < 0.05,
                'spearman_corr': float(spearman_corr.cpu()),
                'spearman_p_value': float(spearman_p),
                'spearman_significant': spearman_p < 0.05,
            }

            # Effect size
            abs_pearson = abs(results['pearson_corr'])
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

        except Exception as e:
            logger.warning(f"GPU correlation calculation failed for {feature_name}: {e}")
            return None

    def analyze_features_on_gpu(self):
        """Analyze all Phase 1 features on GPU."""
        logger.info("📈 Analyzing Phase 1 features on GPU...")

        results = []
        for feature_name, feature_gpu in self.features.items():
            logger.info(f"  Analyzing {feature_name} on GPU...")

            feature_results = self.calculate_gpu_correlation(feature_name, feature_gpu)
            if feature_results:
                results.append(feature_results)

        # Clear GPU cache after analysis
        torch.cuda.empty_cache()
        logger.info("🗑️ Cleared GPU cache")

        return results

    def print_results(self, results):
        """Print analysis results."""
        logger.info("\n" + "="*80)
        logger.info("📊 GPU-ACCELERATED PHASE 1 FEATURE ANALYSIS RESULTS")
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

    def cleanup_gpu(self):
        """Clean up GPU memory."""
        if self.use_gpu:
            torch.cuda.empty_cache()
            logger.info("🗑️ GPU memory cleaned up")

def main():
    """Main execution."""
    logger.info("🚀 Starting GPU-Accelerated Phase 1 Feature Analysis...")

    try:
        # Initialize analyzer
        analyzer = GPUPhase1Analyzer('/workspace/DATA/MERGED/merged_es_vix_test.csv')

        # Load data to GPU
        analyzer.load_data_to_gpu()

        # Create Phase 1 features on GPU
        analyzer.create_phase1_technical_indicators()
        analyzer.create_phase1_time_features()

        # Analyze features on GPU
        results = analyzer.analyze_features_on_gpu()

        # Print results
        analyzer.print_results(results)

        logger.info("✅ GPU Phase 1 analysis completed successfully!")

    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        raise
    finally:
        # Cleanup
        if 'analyzer' in locals():
            analyzer.cleanup_gpu()

if __name__ == "__main__":
    main()