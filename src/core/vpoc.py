import numpy as np
import pandas as pd
import torch
from src.utils.logging import get_logger
from src.config.settings import settings

logger = get_logger(__name__)

class VolumeProfileAnalyzer:
    """Handles volume profile analysis, VPOC, and value area calculations."""
    
    def __init__(self, price_precision=None, device='cuda', device_ids=None):
        self.logger = get_logger(__name__)
        self.price_precision = price_precision or settings.PRICE_PRECISION
        
        # Handle multi-GPU setup
        if torch.cuda.is_available():
            if device_ids and len(device_ids) > 1:
                self.device = f'cuda:{device_ids[0]}'  # Primary device
                self.parallel = True
                self.device_ids = device_ids
            else:
                self.device = device
                self.parallel = False
        else:
            self.device = 'cpu'
            self.parallel = False
            
        self.logger.info(f"Initialized VolumeProfileAnalyzer with precision {self.price_precision} on {self.device}" + 
                        (f" using devices {device_ids}" if self.parallel else ""))
    
    def calculate_volume_profile(self, session_df):
        """
        Calculate volume profile for a single session with improved volume distribution.
        
        Parameters:
        -----------
        session_df : pandas.DataFrame
            DataFrame containing session data with OHLCV columns
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with price levels and corresponding volumes
        """
        self.logger.debug(f"Calculating volume profile for session with {len(session_df)} bars")
        
        # Convert to PyTorch tensors on GPU
        try:
            # Get OHLCV data as tensors
            if self.parallel:
                # Use DataParallel for multi-GPU
                open_prices = torch.tensor(session_df['open'].values).to(self.device)
                high_prices = torch.tensor(session_df['high'].values).to(self.device)
                low_prices = torch.tensor(session_df['low'].values).to(self.device)
                close_prices = torch.tensor(session_df['close'].values).to(self.device)
                volumes = torch.tensor(session_df['volume'].values).to(self.device)
                
                # Replicate tensors across GPUs
                open_prices = torch.nn.parallel.replicate(open_prices, self.device_ids)
                high_prices = torch.nn.parallel.replicate(high_prices, self.device_ids)
                low_prices = torch.nn.parallel.replicate(low_prices, self.device_ids)
                close_prices = torch.nn.parallel.replicate(close_prices, self.device_ids)
                volumes = torch.nn.parallel.replicate(volumes, self.device_ids)
            else:
                # Single GPU
                open_prices = torch.tensor(session_df['open'].values, device=self.device)
                high_prices = torch.tensor(session_df['high'].values, device=self.device)
                low_prices = torch.tensor(session_df['low'].values, device=self.device)
                close_prices = torch.tensor(session_df['close'].values, device=self.device)
                volumes = torch.tensor(session_df['volume'].values, device=self.device)

            # Find min and max prices
            if self.parallel:
                # Multi-GPU version
                min_price = min(
                    torch.min(torch.cat([t.min().unsqueeze(0) for t in low_prices])),
                    torch.min(torch.cat([t.min().unsqueeze(0) for t in open_prices])),
                    torch.min(torch.cat([t.min().unsqueeze(0) for t in close_prices]))
                )
                max_price = max(
                    torch.max(torch.cat([t.max().unsqueeze(0) for t in high_prices])),
                    torch.max(torch.cat([t.max().unsqueeze(0) for t in open_prices])),
                    torch.max(torch.cat([t.max().unsqueeze(0) for t in close_prices]))
                )
            else:
                # Single GPU version
                min_price = min(low_prices.min(), open_prices.min(), close_prices.min())
                max_price = max(high_prices.max(), open_prices.max(), close_prices.max())

            # Round to nearest tick
            min_price = torch.floor(min_price / self.price_precision) * self.price_precision
            max_price = torch.ceil(max_price / self.price_precision) * self.price_precision

            # Create price bins on GPU
            price_bins = torch.arange(min_price, max_price + self.price_precision, 
                                    self.price_precision, device=self.device)

            # Initialize volume profile on GPU
            volume_profile = torch.zeros_like(price_bins)

            def process_bar(i):
                """Process a single bar (helper function for parallel processing)"""
                if self.parallel:
                    bar_min = min(
                        low_prices[i % len(self.device_ids)][i],
                        open_prices[i % len(self.device_ids)][i],
                        close_prices[i % len(self.device_ids)][i]
                    )
                    bar_max = max(
                        high_prices[i % len(self.device_ids)][i],
                        open_prices[i % len(self.device_ids)][i],
                        close_prices[i % len(self.device_ids)][i]
                    )
                else:
                    bar_min = min(low_prices[i], open_prices[i], close_prices[i])
                    bar_max = max(high_prices[i], open_prices[i], close_prices[i])

                mask = (price_bins >= bar_min) & (price_bins <= bar_max)
                price_points = price_bins[mask]

                if len(price_points) > 0:
                    weights = torch.ones(len(price_points), device=self.device)
                    prices = [
                        open_prices[i % len(self.device_ids)][i] if self.parallel else open_prices[i],
                        high_prices[i % len(self.device_ids)][i] if self.parallel else high_prices[i],
                        low_prices[i % len(self.device_ids)][i] if self.parallel else low_prices[i],
                        close_prices[i % len(self.device_ids)][i] if self.parallel else close_prices[i]
                    ]
                    
                    for price in prices:
                        distance = torch.abs(price_points - price)
                        proximity_weight = 1.0 / (1.0 + distance)
                        weights += proximity_weight

                    weights = weights / weights.sum()
                    vol = volumes[i % len(self.device_ids)][i] if self.parallel else volumes[i]
                    return mask, weights * vol
                return None, None

            # Process all bars in parallel
            results = []
            if self.parallel:
                # Distribute work across GPUs
                inputs = [(i,) for i in range(len(session_df))]
                results = torch.nn.parallel.parallel_apply(
                    [process_bar]*len(inputs), 
                    inputs,
                    devices=self.device_ids
                )
            else:
                # Single GPU processing
                for i in range(len(session_df)):
                    mask, weighted_vol = process_bar(i)
                    if mask is not None:
                        volume_profile[mask] += weighted_vol

            # Combine results from multiple GPUs
            if self.parallel:
                for mask, weighted_vol in results:
                    if mask is not None:
                        volume_profile[mask] += weighted_vol.to(self.device)

            # Apply smoothing on GPU
            volume_profile_smooth = torch.zeros_like(volume_profile)
            for i in range(len(volume_profile)):
                start = max(0, i-1)
                end = min(len(volume_profile), i+2)
                volume_profile_smooth[i] = volume_profile[start:end].mean()

            # Convert back to pandas DataFrame
            volume_profile_df = pd.DataFrame({
                'price_level': price_bins.cpu().numpy(),
                'volume': volume_profile.cpu().numpy(),
                'volume_smooth': volume_profile_smooth.cpu().numpy()
            })

            return volume_profile_df

        except Exception as e:
            self.logger.error(f"GPU volume profile calculation failed: {str(e)}")
            self.logger.info("Falling back to CPU implementation")
            return self._calculate_volume_profile_cpu(session_df)
    
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
def calculate_volume_profile(session_data: pd.DataFrame, price_precision: float = 0.25) -> dict:
    """Wrapper for VolumeProfileAnalyzer.calculate_volume_profile for backward compatibility"""
    analyzer = VolumeProfileAnalyzer(price_precision=price_precision)
    vol_profile_df = analyzer.calculate_volume_profile(session_data)
    # Convert DataFrame to dictionary format used by legacy code
    return {row['price_level']: row['volume'] for _, row in vol_profile_df.iterrows()}
    
def find_vpoc(volume_profile):
    """Wrapper for backward compatibility - accepts either DataFrame or dict format"""
    if isinstance(volume_profile, pd.DataFrame):
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
