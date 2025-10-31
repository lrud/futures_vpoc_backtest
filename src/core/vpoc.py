import os
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from src.utils.logging import get_logger
from src.config.settings import settings

logger = get_logger(__name__)

class VolumeProfileAnalyzer:
    """Handles volume profile analysis, VPOC, and value area calculations with ROCm 6.3 compatible multi-GPU optimization."""

    def __init__(self, price_precision=None, device='cuda', device_ids=None, chunk_size=3500):
        self.logger = get_logger(__name__)
        self.price_precision = price_precision or settings.PRICE_PRECISION
        self.chunk_size = chunk_size

        # ROCm 6.3 compatible optimizations for dual RX 7900 XT
        os.environ['HIP_VISIBLE_DEVICES'] = '0,1'
        os.environ['HSA_ENABLE_SDMA'] = '0'
        os.environ['HSA_ENABLE_INTERRUPT'] = '0'

        # Handle dual RX 7900 XT setup
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            self.logger.info(f"Detected {gpu_count} RX 7900 XT GPUs")

            if gpu_count >= 2 and device_ids and len(device_ids) >= 2:
                # Dual GPU configuration
                self.device_ids = device_ids[:2]  # Use first two GPUs
                self.parallel = True
                self.device = 'cuda'
                self.num_gpus = 2
                self.logger.info(f"🚀 Dual RX 7900 XT configuration: GPUs {self.device_ids}")

                # Verify both GPUs are accessible
                for gpu_id in self.device_ids:
                    try:
                        torch.cuda.device(gpu_id)
                        props = torch.cuda.get_device_properties(gpu_id)
                        self.logger.info(f"✅ GPU {gpu_id}: {props.name} ({props.total_memory/1e9:.1f}GB)")
                    except Exception as e:
                        self.logger.error(f"❌ GPU {gpu_id} not accessible: {e}")
                        raise RuntimeError(f"GPU {gpu_id} not accessible")
            else:
                # Single GPU fallback
                self.device = device
                self.parallel = False
                self.num_gpus = 1
                self.device_ids = [0]
                self.logger.info("🔧 Single GPU configuration")
        else:
            raise RuntimeError("CUDA is not available - GPU access required")

        self.logger.info(f"🚀 Initialized ROCm 6.3 compatible VolumeProfileAnalyzer with precision {self.price_precision} on {self.num_gpus} GPU(s): {self.device_ids}")

    def _setup_distributed_vpoc(self):
        """Setup distributed processing for VPOC calculations."""
        try:
            # Use ROCm's optimized scatter gather operations
            self.logger.info("Setting up ROCm 6.3 compatible distributed VPOC processing")

            # Removed pre-allocation that was causing memory leaks
            # Memory will be allocated dynamically as needed

        except Exception as e:
            self.logger.warning(f"Could not setup distributed VPOC optimizations: {e}")
            self.parallel = False
    
    def calculate_volume_profile(self, session_df):
        """
        Calculate volume profile for a single session with ROCm 7 multi-GPU optimization.

        Parameters:
        -----------
        session_df : pandas.DataFrame
            DataFrame containing session data with OHLCV columns

        Returns:
        --------
        pandas.DataFrame
            DataFrame with price levels and corresponding volumes
        """
        self.logger.debug(f"🚀 ROCm 7 Multi-GPU Volume Profile Calculation: {len(session_df)} bars on {self.num_gpus} GPUs")

        try:
            if self.parallel and self.num_gpus > 1:
                return self._calculate_volume_profile_distributed(session_df)
            else:
                return self._calculate_volume_profile_single_gpu(session_df)

        except Exception as e:
            self.logger.error(f"GPU volume profile calculation failed: {str(e)}")
            self.logger.error("GPU processing is required - no CPU fallback available")
            raise RuntimeError(f"VPOC GPU processing failed: {str(e)}")

    def _calculate_volume_profile_distributed(self, session_df):
        """Distributed VPOC calculation across multiple GPUs with ROCm 7 optimization."""
        num_bars = len(session_df)
        bars_per_gpu = num_bars // self.num_gpus
        remainder = num_bars % self.num_gpus

        # Split data across GPUs with proper validation
        gpu_data = []
        for gpu_id in range(self.num_gpus):
            start_idx = gpu_id * bars_per_gpu
            end_idx = start_idx + bars_per_gpu + (remainder if gpu_id == self.num_gpus - 1 else 0)

            if start_idx < num_bars and end_idx > start_idx:
                gpu_df = session_df.iloc[start_idx:end_idx].copy()
                if len(gpu_df) > 0:  # Only add non-empty slices
                    gpu_data.append(gpu_df)
                else:
                    self.logger.warning(f"Skipping GPU {gpu_id} due to empty data slice")
            else:
                self.logger.warning(f"Skipping GPU {gpu_id} due to invalid slice indices: start={start_idx}, end={end_idx}, num_bars={num_bars}")

        # If we have less data than GPUs, ensure at least one GPU gets data
        if not gpu_data and num_bars > 0:
            gpu_data = [session_df.copy()]  # Give all data to first GPU
            self.logger.warning(f"Insufficient data for {self.num_gpus} GPUs, using single GPU")

        # Process VPOC on each GPU in parallel
        results = []

        def process_gpu_data(gpu_id, local_df):
            """Process VPOC on a specific GPU."""
            # Validate input data
            if len(local_df) == 0:
                raise ValueError(f"Empty dataframe provided to GPU {gpu_id}")

            device = f'cuda:{gpu_id}'

            with torch.cuda.device(device):
                # Convert to tensors on this specific GPU
                open_prices = torch.tensor(local_df['open'].values, device=device, dtype=torch.float32)
                high_prices = torch.tensor(local_df['high'].values, device=device, dtype=torch.float32)
                low_prices = torch.tensor(local_df['low'].values, device=device, dtype=torch.float32)
                close_prices = torch.tensor(local_df['close'].values, device=device, dtype=torch.float32)
                volumes = torch.tensor(local_df['volume'].values, device=device, dtype=torch.float32)

                # Validate tensors are not empty
                if len(open_prices) == 0 or len(high_prices) == 0 or len(low_prices) == 0:
                    raise ValueError(f"Empty price tensors for GPU {gpu_id}")

                # ROCm 7 optimized calculation with proper min/max operations
                min_price = torch.min(torch.stack([
                    low_prices.min(),
                    open_prices.min(),
                    close_prices.min()
                ]))
                max_price = torch.max(torch.stack([
                    high_prices.max(),
                    open_prices.max(),
                    close_prices.max()
                ]))

                # ROCm 7 aligned memory operations
                min_price = torch.floor(min_price / self.price_precision) * self.price_precision
                max_price = torch.ceil(max_price / self.price_precision) * self.price_precision

                # Ensure price range is valid (handle edge case where all prices are the same)
                if max_price <= min_price:
                    # Expand the range to include at least one price level
                    price_range = max_price * 0.001  # 0.1% of price
                    min_price = max_price - price_range
                    max_price = max_price + price_range
                    self.logger.warning(f"Expanded price range for GPU {gpu_id}: {min_price:.2f} - {max_price:.2f}")

                price_bins = torch.arange(min_price, max_price + self.price_precision,
                                        self.price_precision, device=device, dtype=torch.float32)

                # Initialize volume profile
                volume_profile = torch.zeros_like(price_bins, dtype=torch.float32)

                # Vectorized bar processing for ROCm 7
                bar_mins = torch.minimum(torch.minimum(open_prices, close_prices), low_prices)
                bar_maxs = torch.maximum(torch.maximum(open_prices, close_prices), high_prices)

                # ROCm 7 optimized matrix operations
                for i in range(len(local_df)):
                    bar_min = bar_mins[i]
                    bar_max = bar_maxs[i]

                    # Vectorized mask calculation
                    mask = (price_bins >= bar_min) & (price_bins <= bar_max)
                    price_points = price_bins[mask]

                    if price_points.numel() > 0:  # Use numel() instead of len for tensors
                        # ROCm 7 proximity weighting with distance calculations
                        distances = torch.abs(price_points - close_prices[i])
                        proximity_weights = 1.0 / (1.0 + distances)
                        weights = proximity_weights + 1.0
                        weights = weights / weights.sum()

                        # Apply volume
                        volume_profile[mask] += weights * volumes[i]

                # ROCm 7 smoothing with convolution-like operations
                if volume_profile.numel() > 0:
                    kernel_size = 3
                    padding = kernel_size // 2
                    volume_profile_expanded = torch.nn.functional.pad(volume_profile, (padding, padding))
                    volume_profile_smooth = torch.nn.functional.avg_pool1d(
                        volume_profile_expanded.unsqueeze(0).unsqueeze(0),
                        kernel_size, stride=1, padding=0
                    ).squeeze()
                else:
                    volume_profile_smooth = volume_profile

                # Convert results to CPU and clean up GPU memory
                result = (price_bins.cpu().numpy(), volume_profile.cpu().numpy(), volume_profile_smooth.cpu().numpy())

                # Clean up GPU tensors
                del open_prices, high_prices, low_prices, close_prices, volumes
                del price_bins, volume_profile, volume_profile_smooth
                del bar_mins, bar_maxs
                if 'volume_profile_expanded' in locals():
                    del volume_profile_expanded

                return result

        # Launch parallel processing with proper error handling
        import concurrent.futures
        results = []

        if not gpu_data:
            raise ValueError("No valid data to process on any GPU")

        # Use actual number of GPUs with data for worker count
        actual_gpus = len(gpu_data)
        with concurrent.futures.ThreadPoolExecutor(max_workers=actual_gpus) as executor:
            futures = []
            for i, local_df in enumerate(gpu_data):
                # Use original GPU ID if available, otherwise use index
                gpu_id = self.device_ids[i] if i < len(self.device_ids) else i
                futures.append(executor.submit(process_gpu_data, gpu_id, local_df))

            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"GPU processing failed: {e}")
                    raise

        # Combine results from all GPUs
        all_price_bins = []
        all_volumes = []
        all_volumes_smooth = []

        for price_bins, volumes, volumes_smooth in results:
            all_price_bins.extend(price_bins)
            all_volumes.extend(volumes)
            all_volumes_smooth.extend(volumes_smooth)

        # Convert back to pandas DataFrame
        volume_profile_df = pd.DataFrame({
            'price_level': all_price_bins,
            'volume': all_volumes,
            'volume_smooth': all_volumes_smooth
        })

        # Clear GPU memory after distributed processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return volume_profile_df

    def _calculate_volume_profile_single_gpu(self, session_df):
        """Optimized single GPU VPOC calculation with ROCm 7."""
        device = self.device

        # Convert to tensors with ROCm 7 optimizations
        open_prices = torch.tensor(session_df['open'].values, device=device, dtype=torch.float32)
        high_prices = torch.tensor(session_df['high'].values, device=device, dtype=torch.float32)
        low_prices = torch.tensor(session_df['low'].values, device=device, dtype=torch.float32)
        close_prices = torch.tensor(session_df['close'].values, device=device, dtype=torch.float32)
        volumes = torch.tensor(session_df['volume'].values, device=device, dtype=torch.float32)

        # Find price range
        min_price = torch.min(torch.stack([low_prices.min(), open_prices.min(), close_prices.min()]))
        max_price = torch.max(torch.stack([high_prices.max(), open_prices.max(), close_prices.max()]))

        # ROCm 7 aligned memory
        min_price = torch.floor(min_price / self.price_precision) * self.price_precision
        max_price = torch.ceil(max_price / self.price_precision) * self.price_precision

        # Create price bins
        price_bins = torch.arange(min_price, max_price + self.price_precision,
                                self.price_precision, device=device, dtype=torch.float32)

        # Initialize volume profile
        volume_profile = torch.zeros_like(price_bins, dtype=torch.float32)

        # Vectorized processing
        bar_mins = torch.minimum(torch.minimum(open_prices, close_prices), low_prices)
        bar_maxs = torch.maximum(torch.maximum(open_prices, close_prices), high_prices)

        num_bars = len(session_df)
        for i in range(num_bars):
            bar_min = bar_mins[i]
            bar_max = bar_maxs[i]

            # Vectorized mask calculation
            mask = (price_bins >= bar_min) & (price_bins <= bar_max)
            price_points = price_bins[mask]

            if len(price_points) > 0:
                # ROCm 7 proximity weighting
                distances = torch.abs(price_points - close_prices[i])
                proximity_weights = 1.0 / (1.0 + distances)
                weights = proximity_weights + 1.0
                weights = weights / weights.sum()

                volume_profile[mask] += weights * volumes[i]

        # ROCm 7 optimized smoothing
        kernel_size = 3
        padding = kernel_size // 2
        volume_profile_expanded = torch.nn.functional.pad(volume_profile, (padding, padding))
        volume_profile_smooth = torch.nn.functional.avg_pool1d(
            volume_profile_expanded.unsqueeze(0).unsqueeze(0),
            kernel_size, stride=1, padding=0
        ).squeeze()

        # Convert back to pandas
        volume_profile_df = pd.DataFrame({
            'price_level': price_bins.cpu().numpy(),
            'volume': volume_profile.cpu().numpy(),
            'volume_smooth': volume_profile_smooth.cpu().numpy()
        })

        # Clear GPU memory after single GPU processing
        del open_prices, high_prices, low_prices, close_prices, volumes
        del price_bins, volume_profile, volume_profile_smooth
        del bar_mins, bar_maxs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return volume_profile_df
    
    def find_vpoc(self, volume_profile, use_smoothing=True):
        """
        Find the Volume Point of Control (VPOC) with improved detection using clustering.
        
        Parameters:
        -----------
        volume_profile : pandas.DataFrame
            DataFrame with price levels and volumes
        use_smoothing : bool, optional
            Whether to use smoothed volumes for calculation
            
        Returns:
        --------
        float
            VPOC price level
        """
        vol_column = 'volume_smooth' if use_smoothing and 'volume_smooth' in volume_profile.columns else 'volume'

        # Basic method: find the single highest volume price level
        vpoc_idx = volume_profile[vol_column].argmax()
        vpoc_simple = volume_profile.iloc[vpoc_idx]['price_level']

        # Advanced method: find cluster of high volume
        # Look at top 5% of volume levels
        threshold = volume_profile[vol_column].quantile(0.95)
        high_vol_levels = volume_profile[volume_profile[vol_column] >= threshold]

        if len(high_vol_levels) >= 3:  # Need enough points for clustering
            # Calculate weighted average of high volume cluster to find center
            vpoc_cluster = np.average(high_vol_levels['price_level'],
                                    weights=high_vol_levels[vol_column])

            # Find the actual price level closest to this weighted average
            vpoc_idx = (volume_profile['price_level'] - vpoc_cluster).abs().argmin()
            vpoc_cluster = volume_profile.iloc[vpoc_idx]['price_level']

            return vpoc_cluster
        else:
            # Fall back to simple method if not enough data points
            return vpoc_simple
    
    def find_value_area(self, volume_profile, value_area_pct=0.7, use_smoothing=True):
        """
        Calculate the Value Area ensuring price continuity.
        
        Parameters:
        -----------
        volume_profile : pandas.DataFrame
            DataFrame with price levels and volumes
        value_area_pct : float, optional
            Percentage of volume to include in value area
        use_smoothing : bool, optional
            Whether to use smoothed volumes
            
        Returns:
        --------
        tuple
            (value_area_low, value_area_high, value_area_volume_percentage)
        """
        vol_column = 'volume_smooth' if use_smoothing and 'volume_smooth' in volume_profile.columns else 'volume'

        # Find VPOC first
        vpoc_idx = volume_profile[vol_column].argmax()
        vpoc_level = volume_profile.iloc[vpoc_idx]['price_level']

        # Start with VPOC and expand outward
        total_volume = volume_profile[vol_column].sum()
        target_volume = total_volume * value_area_pct

        # Initialize with VPOC
        included_indices = [vpoc_idx]
        current_volume = volume_profile.iloc[vpoc_idx][vol_column]

        # Initialize upper and lower boundaries
        upper_idx = vpoc_idx
        lower_idx = vpoc_idx

        # Expand outward until we reach target volume
        while current_volume < target_volume and (upper_idx < len(volume_profile) - 1 or lower_idx > 0):
            # Check volume at next level up
            upper_vol = volume_profile.iloc[upper_idx + 1][vol_column] if upper_idx < len(volume_profile) - 1 else 0

            # Check volume at next level down
            lower_vol = volume_profile.iloc[lower_idx - 1][vol_column] if lower_idx > 0 else 0

            # Add the level with more volume
            if upper_vol >= lower_vol and upper_idx < len(volume_profile) - 1:
                upper_idx += 1
                included_indices.append(upper_idx)
                current_volume += upper_vol
            elif lower_idx > 0:
                lower_idx -= 1
                included_indices.append(lower_idx)
                current_volume += lower_vol
            else:
                # Reached boundary, can't expand further
                break

        # Get value area bounds
        value_area_levels = volume_profile.iloc[included_indices]['price_level'].values
        val = min(value_area_levels)
        vah = max(value_area_levels)

        # Calculate value area volume percentage
        va_volume_pct = (current_volume / total_volume) * 100

        return val, vah, va_volume_pct
    
    def _calculate_volume_profile_chunked(self, session_df):
        """
        ROCm 7 SOLUTION 2: Chunked VPOC calculation to prevent VRAM fragmentation.
        Processes large sessions in 3500-bar chunks with aggressive memory cleanup.
        """
        self.logger.info(f"ROCm 7: Processing {len(session_df)} bars in chunks to prevent VRAM fragmentation")

        chunk_size = self.chunk_size  # Use configured chunk size
        total_bars = len(session_df)
        num_chunks = (total_bars + chunk_size - 1) // chunk_size

        self.logger.info(f"ROCm 7: Splitting into {num_chunks} chunks of {chunk_size} bars each")

        # Process session in chunks and combine results
        all_volume_profiles = []

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, total_bars)

            chunk_df = session_df.iloc[start_idx:end_idx].copy()

            self.logger.debug(f"ROCm 7: Processing chunk {chunk_idx + 1}/{num_chunks}, bars {start_idx}-{end_idx}")

            try:
                # Calculate volume profile for this chunk using single GPU method
                chunk_profile_df = self._calculate_volume_profile_single_gpu(chunk_df)
                all_volume_profiles.append(chunk_profile_df)

                # ROCm 7: Aggressive memory cleanup after each chunk
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    # Force garbage collection
                    import gc
                    gc.collect()

                self.logger.debug(f"ROCm 7: Completed chunk {chunk_idx + 1}/{num_chunks}, memory cleaned up")

            except Exception as e:
                self.logger.error(f"Failed to process chunk {chunk_idx + 1}: {e}")
                # No CPU fallback - GPU processing is required
                raise RuntimeError(f"Chunk {chunk_idx + 1} GPU processing failed: {e}")

        # Combine all chunk results
        self.logger.info("ROCm 7: Combining volume profiles from all chunks")

        if not all_volume_profiles:
            raise ValueError("No volume profiles were generated from any chunks")

        # Combine all volume profiles by merging price levels and summing volumes
        combined_profile = pd.DataFrame(columns=['price_level', 'volume', 'volume_smooth'])

        for profile in all_volume_profiles:
            if combined_profile.empty:
                combined_profile = profile.copy()
            else:
                # Merge price levels and sum volumes
                for _, row in profile.iterrows():
                    price_level = row['price_level']
                    existing_idx = combined_profile[combined_profile['price_level'] == price_level].index

                    if len(existing_idx) > 0:
                        # Add to existing price level
                        idx = existing_idx[0]
                        combined_profile.loc[idx, 'volume'] += row['volume']
                        combined_profile.loc[idx, 'volume_smooth'] += row.get('volume_smooth', row['volume'])
                    else:
                        # Add new price level
                        new_row = pd.DataFrame([row])
                        combined_profile = pd.concat([combined_profile, new_row], ignore_index=True)

        # Sort by price level and recalculate smoothing
        combined_profile = combined_profile.sort_values('price_level').reset_index(drop=True)

        # Recalculate smoothed volume across combined profile
        if len(combined_profile) > 0:
            combined_profile['volume_smooth'] = combined_profile['volume'].rolling(window=3, center=True, min_periods=1).mean()
            combined_profile['volume_smooth'] = combined_profile['volume_smooth'].fillna(combined_profile['volume'])

        self.logger.info(f"ROCm 7: Combined volume profile created with {len(combined_profile)} price levels")

        # Final aggressive memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc
            gc.collect()

        # Convert to dictionary format for legacy compatibility
        return {row['price_level']: row['volume'] for _, row in combined_profile.iterrows()}

    def _calculate_volume_profile_cpu(self, session_df):
        """Fallback CPU implementation of volume profile calculation"""
        self.logger.debug("Using CPU implementation for volume profile")

        # Find min and max prices
        min_price = min(session_df['low'].min(), session_df['open'].min(), session_df['close'].min())
        max_price = max(session_df['high'].max(), session_df['open'].max(), session_df['close'].max())

        # Round to nearest tick
        min_price = np.floor(min_price / self.price_precision) * self.price_precision
        max_price = np.ceil(max_price / self.price_precision) * self.price_precision

        # Create price bins
        price_bins = np.arange(min_price, max_price + self.price_precision, self.price_precision)

        # Create empty volume profile
        volume_profile = pd.DataFrame({
            'price_level': price_bins,
            'volume': 0.0
        })

        # Improved volume distribution
        for _, row in session_df.iterrows():
            bar_min = min(row['low'], row['open'], row['close'])
            bar_max = max(row['high'], row['open'], row['close'])
            mask = (volume_profile['price_level'] >= bar_min) & (volume_profile['price_level'] <= bar_max)
            price_points = volume_profile.loc[mask, 'price_level'].values

            if len(price_points) > 0:
                weights = np.ones(len(price_points))
                for price in [row['open'], row['high'], row['low'], row['close']]:
                    distance = np.abs(price_points - price)
                    proximity_weight = 1.0 / (1.0 + distance)
                    weights += proximity_weight

                weights = weights / weights.sum()
                weighted_volume = weights * row['volume']
                for i, price_level in enumerate(price_points):
                    idx = volume_profile.index[volume_profile['price_level'] == price_level].tolist()
                    if idx:
                        volume_profile.loc[idx[0], 'volume'] += weighted_volume[i]

        # Apply smoothing
        volume_profile['volume_smooth'] = volume_profile['volume'].rolling(window=3, center=True).mean()
        volume_profile['volume_smooth'] = volume_profile['volume_smooth'].fillna(volume_profile['volume'])

        return volume_profile

    def analyze_session(self, session_df):
        """
        Analyze a complete trading session and return all relevant metrics.
        
        Parameters:
        -----------
        session_df : pandas.DataFrame
            DataFrame containing session data with OHLCV columns
            
        Returns:
        --------
        dict
            Dictionary with VPOC analysis results
        """
        self.logger.info(f"Analyzing session with {len(session_df)} bars")
        
        volume_profile = self.calculate_volume_profile(session_df)
        vpoc = self.find_vpoc(volume_profile)
        val, vah, va_volume_pct = self.find_value_area(volume_profile)
        
        return {
            'vpoc': vpoc,
            'value_area_low': val,
            'value_area_high': vah,
            'value_area_width': vah - val,
            'value_area_pct': va_volume_pct,
            'volume_profile': volume_profile,
            'total_volume': session_df['volume'].sum()
        }

# Function wrappers for backward compatibility and convenience
# Global analyzer cache to avoid recreating analyzers for each session
_analyzer_cache = {}

def calculate_volume_profile(session_data: pd.DataFrame, price_precision: float = 0.25, device_ids: List[int] = None, chunk_size: int = 3500) -> dict:
    """Wrapper for VolumeProfileAnalyzer.calculate_volume_profile for backward compatibility with ROCm 7 chunked processing"""
    # Create cache key based on price_precision, device_ids, and chunk_size
    cache_key = (price_precision, tuple(device_ids) if device_ids else None, chunk_size)

    # Reuse existing analyzer if available, otherwise create and cache
    if cache_key not in _analyzer_cache:
        _analyzer_cache[cache_key] = VolumeProfileAnalyzer(price_precision=price_precision, device_ids=device_ids, chunk_size=chunk_size)
        logger.info(f"Created new VolumeProfileAnalyzer cache entry for precision {price_precision}, device_ids {device_ids}, chunk_size {chunk_size}")

    analyzer = _analyzer_cache[cache_key]

    # ROCm 7 SOLUTION 2: Chunked VPOC calculations for memory fragmentation fix
    # For large sessions (>1000 bars), process in chunks to prevent VRAM fragmentation
    if len(session_data) > 1000:
        logger.info(f"ROCm 7: Using chunked VPOC calculation for {len(session_data)} bars")
        return analyzer._calculate_volume_profile_chunked(session_data)
    else:
        vol_profile_df = analyzer.calculate_volume_profile(session_data)
        # Convert DataFrame to dictionary format used by legacy code
        return {row['price_level']: row['volume'] for _, row in vol_profile_df.iterrows()}
    
def find_vpoc(volume_profile):
    """Wrapper for backward compatibility - accepts either DataFrame or dict format"""
    if isinstance(volume_profile, pd.DataFrame):
        # Use cached analyzer if available, otherwise create a simple one
        if (0.25, None) in _analyzer_cache:
            return _analyzer_cache[(0.25, None)].find_vpoc(volume_profile)
        else:
            analyzer = VolumeProfileAnalyzer()
            return analyzer.find_vpoc(volume_profile)
    elif isinstance(volume_profile, dict):
        # Handle dictionary format for legacy code
        if not volume_profile:
            logger.warning("Empty volume profile provided")
            return 0.0
        return max(volume_profile.items(), key=lambda x: x[1])[0]
    else:
        logger.error(f"Unsupported volume profile type: {type(volume_profile)}")
        return 0.0
    
def find_value_area(volume_profile, value_area_pct: float = 0.7):
    """Wrapper for backward compatibility - accepts either DataFrame or dict format"""
    if isinstance(volume_profile, pd.DataFrame):
        analyzer = VolumeProfileAnalyzer()
        return analyzer.find_value_area(volume_profile, value_area_pct)
    elif isinstance(volume_profile, dict):
        # Handle dictionary format for legacy code
        if not volume_profile:
            logger.warning("Empty volume profile provided")
            return 0.0, 0.0, 0.0
            
        # Sort price levels by volume (highest first)
        sorted_levels = sorted(volume_profile.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate total volume
        total_volume = sum(volume_profile.values())
        target_volume = total_volume * value_area_pct
        
        # Initialize
        cumulative_volume = 0
        included_prices = []
        
        # Add levels until we reach target volume
        for price, volume in sorted_levels:
            cumulative_volume += volume
            included_prices.append(price)
            
            if cumulative_volume >= target_volume:
                break
                
        # Find Value Area boundaries
        val = min(included_prices)
        vah = max(included_prices)
        actual_pct = cumulative_volume / total_volume
        
        return val, vah, actual_pct
    else:
        logger.error(f"Unsupported volume profile type: {type(volume_profile)}")
        return 0.0, 0.0, 0.0
