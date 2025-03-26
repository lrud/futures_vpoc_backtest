import numpy as np
import pandas as pd
from src.utils.logging import get_logger
from src.config.settings import settings

class VolumeProfileAnalyzer:
    """Handles volume profile analysis, VPOC, and value area calculations."""
    
    def __init__(self, price_precision=None):
        self.logger = get_logger(__name__)
        self.price_precision = price_precision or settings.PRICE_PRECISION
        self.logger.info(f"Initialized VolumeProfileAnalyzer with precision {self.price_precision}")
    
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
        
        # Find min and max prices
        min_price = min(session_df['low'].min(), session_df['open'].min(), session_df['close'].min())
        max_price = max(session_df['high'].max(), session_df['open'].max(), session_df['close'].max())

        # Round to nearest tick
        min_price = np.floor(min_price / self.price_precision) * self.price_precision
        max_price = np.ceil(max_price / self.price_precision) * self.price_precision

        # Create price bins
        price_bins = np.arange(min_price, max_price + self.price_precision, self.price_precision)

        # Create empty volume profile - using float64 instead of int for volume
        volume_profile = pd.DataFrame({
            'price_level': price_bins,
            'volume': 0.0  # Initialize with float instead of int
        })

        # Improved volume distribution across price levels within each bar
        for _, row in session_df.iterrows():
            # Calculate price range for the bar
            bar_min = min(row['low'], row['open'], row['close'])
            bar_max = max(row['high'], row['open'], row['close'])

            # Find bins that fall within this bar's range
            mask = (volume_profile['price_level'] >= bar_min) & (volume_profile['price_level'] <= bar_max)

            # Count how many bins are in this range
            bins_count = mask.sum()

            if bins_count > 0:
                # Get price points within the bar
                price_points = volume_profile.loc[mask, 'price_level'].values

                # Create a weighted distribution based on proximity to OHLC prices
                weights = np.ones(len(price_points))

                # Add more weight to levels near open, high, low, and close
                for price in [row['open'], row['high'], row['low'], row['close']]:
                    # Add extra weight inversely proportional to distance from price
                    distance = np.abs(price_points - price)
                    proximity_weight = 1.0 / (1.0 + distance)
                    weights += proximity_weight

                # Normalize weights to sum to 1
                weights = weights / weights.sum()

                # Distribute volume according to weights
                weighted_volume = weights * row['volume']

                # Add to volume profile
                for i, price_level in enumerate(price_points):
                    idx = volume_profile.index[volume_profile['price_level'] == price_level].tolist()
                    if idx:
                        volume_profile.loc[idx[0], 'volume'] += weighted_volume[i]

        # Apply smoothing to reduce noise
        volume_profile['volume_smooth'] = volume_profile['volume'].rolling(window=3, center=True).mean()
        volume_profile['volume_smooth'] = volume_profile['volume_smooth'].fillna(volume_profile['volume'])

        return volume_profile
    
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