# %%
# Cell 1: Imports and Configuration
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import ast
import glob
from scipy import stats
import warnings
from DATA_LOADER import load_futures_data  # Import load_futures_data from data_loader
warnings.filterwarnings('ignore')

# Configuration
BASE_DIR = '/home/lr/Documents/FUTURUES_PROJECT/futures_vpoc_backtest/'
DATA_DIR = os.path.join(BASE_DIR, 'DATA')
CLEANED_DATA_DIR = os.path.join(BASE_DIR, 'DATA/CLEANED')
OUTPUT_DIR = os.path.join(BASE_DIR, 'RESULTS')
STRATEGY_DIR = os.path.join(BASE_DIR, 'STRATEGY')
MATH_OUTPUT_DIR = os.path.join(BASE_DIR, 'MATH_ANALYSIS')
PRICE_PRECISION = 0.25  # Price increment for volume profile bins
SESSION_TYPE = 'RTH' # Define SESSION_TYPE for consistency with VPOC script

# Create strategy directory if it doesn't exist
if not os.path.exists(STRATEGY_DIR):
    os.makedirs(STRATEGY_DIR)
    print(f"Created directory: {STRATEGY_DIR}")

# %%
# ===== REMOVED: load_data() FUNCTION =====
# The load_data() function has been removed as data loading
# is now handled by the load_futures_data function from data_loader.py
# ==========================================

# %%
# Cell 3: Volume Profile Functions (ADVANCED VERSION from VPOC Script)
def calculate_volume_profile(session_df, price_precision=PRICE_PRECISION):
    """Calculate volume profile for a single session with improved volume distribution."""
    # Find min and max prices
    min_price = min(session_df['low'].min(), session_df['open'].min(), session_df['close'].min())
    max_price = max(session_df['high'].max(), session_df['open'].max(), session_df['close'].max())

    # Round to nearest tick
    min_price = np.floor(min_price / price_precision) * price_precision
    max_price = np.ceil(max_price / price_precision) * price_precision

    # Create price bins
    price_bins = np.arange(min_price, max_price + price_precision, price_precision)

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
            weights = np.ones(price_points.size)  # Initialize weights with correct size

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


def find_vpoc(volume_profile, use_smoothing=True):
    """Find the Volume Point of Control (VPOC) with improved detection using clustering."""
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


def find_value_area(volume_profile, value_area_pct=0.7, use_smoothing=True):
    """Calculate the Value Area ensuring price continuity."""
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

# %%
# Cell 4: LVN Detection Functions (No changes needed - keeping as is)
def identify_lvn_zones(volume_profile, threshold_pct=0.15):
    """
    Identify Low Volume Node zones below threshold % of average session volume.
    """
    avg_volume = volume_profile['volume'].mean()
    lvn_threshold = avg_volume * threshold_pct
    lvn_zones = volume_profile[volume_profile['volume'] < lvn_threshold].copy()
    lvn_zones['pct_of_avg'] = lvn_zones['volume'] / avg_volume * 100
    lvn_zones = lvn_zones.sort_values('price_level')
    if len(lvn_zones) > 0:
        lvn_zones['zone_group'] = (lvn_zones['price_level'].diff() > lvn_zones['price_level'].diff().mean() * 2).cumsum()
        grouped_zones = lvn_zones.groupby('zone_group').agg(
            zone_start=('price_level', 'min'),
            zone_end=('price_level', 'max'),
            avg_volume=('volume', 'mean'),
            pct_of_avg=('pct_of_avg', 'mean'),
            zone_width=('price_level', lambda x: max(x) - min(x))
        )
        return grouped_zones
    else:
        return pd.DataFrame(columns=['zone_start', 'zone_end', 'avg_volume', 'pct_of_avg', 'zone_width'])

def plot_lvn_zones(volume_profile, lvn_zones, vpoc, val, vah, session_title, output_file=None):
    """
    Plot volume profile with LVN zones highlighted.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    try: bar_height = volume_profile['price_level'].diff().iloc[1]
    except: bar_height = 0.25
    ax.barh(volume_profile['price_level'], volume_profile['volume'], height=bar_height, color='steelblue', alpha=0.7)
    vpoc_idx = (volume_profile['price_level'] - vpoc).abs().argmin()
    ax.barh(volume_profile.iloc[vpoc_idx]['price_level'], volume_profile.iloc[vpoc_idx]['volume'], height=bar_height, color='crimson', alpha=0.8)
    val_mask = (volume_profile['price_level'] >= val) & (volume_profile['price_level'] <= vah)
    ax.barh(volume_profile.loc[val_mask, 'price_level'], volume_profile.loc[val_mask, 'volume'], height=bar_height, color='cornflowerblue', alpha=0.5)
    for _, zone in lvn_zones.iterrows():
        ax.add_patch(plt.Rectangle((0, zone['zone_start']), volume_profile['volume'].max() * 1.05, zone['zone_end'] - zone['zone_start'], color='yellow', alpha=0.3))
    ax.axhline(y=vpoc, color='crimson', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.axhline(y=val, color='navy', linestyle=':', alpha=0.7, linewidth=1.5)
    ax.axhline(y=vah, color='navy', linestyle=':', alpha=0.7, linewidth=1.5)
    max_volume = volume_profile['volume'].max() * 1.05
    ax.text(max_volume, vpoc, f' VPOC: {vpoc:.2f}', verticalalignment='center', color='crimson', fontweight='bold')
    ax.text(max_volume, val, f' VAL: {val:.2f}', verticalalignment='center', color='navy', fontweight='bold')
    ax.text(max_volume, vah, f' VAH: {vah:.2f}', verticalalignment='center', color='navy', fontweight='bold')
    for i, zone in lvn_zones.iterrows():
        mid_point = (zone['zone_start'] + zone['zone_end']) / 2
        ax.text(max_volume/2, mid_point, f'LVN: {zone["pct_of_avg"]:.1f}%', horizontalalignment='center', verticalalignment='center', bbox=dict(facecolor='yellow', alpha=0.3))
    ax.set_xlabel('Volume')
    ax.set_ylabel('Price')
    ax.set_title(f'Volume Profile with LVN Zones - {session_title}', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.invert_yaxis()
    plt.tight_layout()
    if output_file: plt.savefig(output_file, dpi=300); plt.close(fig); return None
    else: return fig, ax

# Cell 5: Gap Identification Functions (UPDATED to use DataFrame Index for Timestamp - Readability Improved)
def identify_gaps(df, min_gap_pct=0.3):
    """
    Identify overnight gaps in the futures data.
    """
    print("\nIdentifying overnight gaps...") # Added print statement for clarity

    # 1. Make a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()

    # 2. Ensure Data is Sorted Properly
    if isinstance(df_copy.index, pd.DatetimeIndex):
        # If we already have a DatetimeIndex, sort by it
        df_copy = df_copy.sort_index()
    elif 'timestamp' in df_copy.columns:
        # Try to set timestamp as index if it exists
        df_copy = df_copy.set_index('timestamp').sort_index()
        print("Set DataFrame index to 'timestamp' column")
    else:
        # Otherwise just sort by existing index
        df_copy = df_copy.sort_index()
        print("Using existing index for sorting")

    # 3. Ensure 'session_date' Column Exists
    if 'session_date' not in df_copy.columns:
        if 'date' in df_copy.columns:
            df_copy['session_date'] = df_copy['date']
            print("Using 'date' column as 'session_date'")
        elif isinstance(df_copy.index, pd.DatetimeIndex):
            df_copy['session_date'] = df_copy.index.date
            print("Created 'session_date' from index")
        else:
            print("WARNING: Could not create 'session_date' column. Gap analysis may be incomplete.")
            return pd.DataFrame() # Return empty DataFrame if we can't process

    # 4. Initialize Data Storage
    gap_data = []
    session_dates = sorted(df_copy['session_date'].unique())
    
    # 5. Iterate Through Consecutive Sessions to Identify Gaps
    for i in range(1, len(session_dates)):
        prev_date = session_dates[i-1]
        curr_date = session_dates[i]
        try:
            # Get Previous Session Close and Current Session Open
            prev_session = df_copy[df_copy['session_date'] == prev_date]
            if 'session' in prev_session.columns and (prev_session['session'] == 'RTH').any():
                prev_close = prev_session[prev_session['session'] == 'RTH']['close'].iloc[-1] # RTH close if available
            else:
                prev_close = prev_session['close'].iloc[-1] # Otherwise, use last close of prev session
            curr_open = df_copy[df_copy['session_date'] == curr_date]['open'].iloc[0] # First open of current session

            # Calculate Gap Size and Percentage
            gap_points = curr_open - prev_close
            gap_pct = (gap_points / prev_close) * 100

            # 6. Check if Gap Exceeds Minimum Threshold
            if abs(gap_pct) >= min_gap_pct:
                gap_type = 'up' if gap_points > 0 else 'down'

                # 7. Calculate Gap Fill Status
                # Find the start of the current session using either timestamp index or another approach
                if isinstance(df_copy.index, pd.DatetimeIndex):
                    # Get first timestamp of current session
                    curr_session_start = df_copy[df_copy['session_date'] == curr_date].index.min()
                    # Calculate end of first hour
                    first_hour_end_time = curr_session_start + pd.Timedelta(hours=1)
                    # Filter data for first hour
                    first_hour = df_copy[(df_copy['session_date'] == curr_date) & 
                                        (df_copy.index <= first_hour_end_time)]
                else:
                    # If we don't have a timestamp index, just take the first 15-30 rows of the current session
                    # (adjust this based on your typical data frequency)
                    first_hour = df_copy[df_copy['session_date'] == curr_date].iloc[:30]

                if not first_hour.empty: # Check if there is data for the first hour
                    if gap_type == 'up': # Upward Gap
                        lowest_point = first_hour['low'].min()
                        gap_fill_pct = max(0, min(100, (curr_open - lowest_point) / gap_points * 100 if gap_points != 0 else 0))
                        filled = lowest_point <= prev_close # Gap filled if low reaches or goes below previous close
                    else: # Downward Gap
                        highest_point = first_hour['high'].max()
                        gap_fill_pct = max(0, min(100, (highest_point - curr_open) / abs(gap_points) * 100 if gap_points != 0 else 0))
                        filled = highest_point >= prev_close # Gap filled if high reaches or goes above previous close
                else: # No data in first hour (unlikely, but handle just in case)
                    gap_fill_pct = 0
                    filled = False

                # 8. Additional Contextual Metrics (Gap to Daily Range Ratio)
                daily_range = first_hour['high'].max() - first_hour['low'].min() if not first_hour.empty else 0
                gap_to_range_ratio = abs(gap_points) / daily_range if daily_range > 0 else float('inf') # Ratio of gap size to daily range

                # 9. Store Gap Data
                gap_data.append({ 
                    'date': curr_date, 'prev_close': prev_close, 'open': curr_open, 
                    'gap_points': gap_points, 'gap_pct': gap_pct, 'gap_type': gap_type, 
                    'gap_fill_pct_1h': gap_fill_pct, 'filled_1h': filled, 
                    'gap_zone_start': min(prev_close, curr_open), 'gap_zone_end': max(prev_close, curr_open), 
                    'gap_to_range_ratio': gap_to_range_ratio 
                })

        except (IndexError, KeyError) as e: # Handle potential errors during gap calculation
            print(f"Error processing gap between {prev_date} and {curr_date}: {str(e)}")
            continue # Continue to next session pair if error occurs

    # 10. Create Gaps DataFrame and Print Summary
    if gap_data:
        gaps_df = pd.DataFrame(gap_data)
        print(f"Found {len(gaps_df)} gaps ({len(gaps_df[gaps_df['gap_type'] == 'up'])} up, {len(gaps_df[gaps_df['gap_type'] == 'down'])} down)")
        print(f"Average gap size: {gaps_df['gap_pct'].abs().mean():.2f}%")
        print(f"Gap fill rate (1st hour): {(gaps_df['filled_1h'].sum() / len(gaps_df)) * 100:.1f}%")
        return gaps_df
    else: # No gaps found
        print("No gaps found with current criteria")
        return pd.DataFrame(columns=['date', 'prev_close', 'open', 'gap_points', 'gap_pct', 
                                    'gap_type', 'gap_fill_pct_1h', 'filled_1h', 
                                    'gap_zone_start', 'gap_zone_end', 'gap_to_range_ratio'])

# Fixed plot_gaps function
def plot_gaps(df, gaps_df, days_to_plot=5):
    """
    Plot price action with gap zones highlighted for recent gaps.

    Generates and saves plots for the most recent overnight gaps,
    showing price action and highlighting the gap zone.

    Parameters:
    -----------
    df : DataFrame
        DataFrame containing the full dataset with price and timestamp data.
    gaps_df : DataFrame
        DataFrame containing gap analysis results (output of identify_gaps).
    days_to_plot : int, optional
        Number of recent gap days to plot (default: 5).
    """
    print("\nPlotting gap charts...")  # Indicate plotting start

    if len(gaps_df) == 0:
        print("No gaps found to plot.")  # Inform if no gaps to plot
        return

    # 1. Ensure we have a working copy of the DataFrame
    df_copy = df.copy()

    # 2. Ensure 'session_date' Column Exists
    if 'session_date' not in df_copy.columns:
        if 'date' in df_copy.columns:
            df_copy['session_date'] = df_copy['date']
            print("Using 'date' column as 'session_date' for gap plotting")
        elif isinstance(df_copy.index, pd.DatetimeIndex):
            df_copy['session_date'] = df_copy.index.date
            print("Created 'session_date' from index for gap plotting")
        else:
            print("ERROR: Cannot create 'session_date' for gap plotting")
            return

    # 3. Select Most Recent Gaps to Plot
    recent_gaps = gaps_df.sort_values('date', ascending=False).head(days_to_plot)  # Get DataFrame of recent gaps

    # 4. Iterate Through Recent Gaps and Generate Plots
    for _, gap in recent_gaps.iterrows():  # Loop through each gap in recent_gaps DataFrame
        gap_date = gap['date']
        prev_date = gap_date - timedelta(days=1)  # Calculate previous date

        # 5. Filter Data for Plotting (Previous and Current Sessions)
        plot_data = df_copy[(df_copy['session_date'] >= prev_date) & (df_copy['session_date'] <= gap_date)]  # Filter DataFrame for date range

        if not plot_data.empty:  # Proceed only if there is data to plot
            # Check if we need to set a timestamp index for plotting
            if not isinstance(plot_data.index, pd.DatetimeIndex) and 'timestamp' in plot_data.columns:
                try:
                    # Try to set timestamp as index for plotting, but don't fail if it doesn't work
                    plot_data = plot_data.set_index('timestamp')
                    print(f"Set timestamp index for plotting gap on {gap_date}")
                except Exception as e:
                    print(f"Warning: Could not set timestamp index for gap plot: {str(e)}")

            # 6. Create Figure and Subplot
            fig, ax = plt.subplots(figsize=(14, 7))  # Initialize figure and axes for plot

            # 7. Plot Price Data (Close Price Line Chart)
            ax.plot(plot_data.index, plot_data['close'], color='black', label='Price')  # Plot close price line

            # 8. Highlight Gap Zone (Yellow for Up Gap, Red for Down Gap)
            ax.axhspan(gap['gap_zone_start'], gap['gap_zone_end'],  # Shade region between gap start and end prices
                      color='yellow' if gap['gap_type'] == 'up' else 'red',
                      alpha=0.3, label=f"{gap['gap_type'].title()} Gap Zone")  # Yellow for up gap, red for down gap

            # 9. Add Annotations for Previous Close and Current Open Lines
            ax.axhline(y=gap['prev_close'], color='blue', linestyle='--',
                      label=f"Previous Close: {gap['prev_close']:.2f}")  # Horizontal line for previous day's close
            ax.axhline(y=gap['open'], color='green', linestyle='--',
                      label=f"Open: {gap['open']:.2f}")  # Horizontal line for current day's open

            # 10. Set Plot Title and Labels
            gap_fill_status = "Filled" if gap['filled_1h'] else f"{gap['gap_fill_pct_1h']:.1f}% Filled"  # Determine gap fill status string
            ax.set_title(f"{gap_date} - {gap['gap_type'].upper()} Gap: {abs(gap['gap_pct']):.2f}% ({gap_fill_status} in 1st hour)",
                         fontsize=14, fontweight='bold')  # Set plot title with gap details
            ax.set_xlabel('Time', fontweight='bold')
            ax.set_ylabel('Price', fontweight='bold')

            # 11. Add Legend and Grid
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)

            # 12. Format x-axis with dates if using a datetime index
            if isinstance(plot_data.index, pd.DatetimeIndex):
                plt.gcf().autofmt_xdate()  # Auto-format the dates

            # 13. Save Plot to File
            output_file = os.path.join(STRATEGY_DIR, f"gap_{gap_date}_{gap['gap_type']}.png")  # Define output file path
            plt.savefig(output_file, dpi=300, bbox_inches='tight')  # Save plot as PNG
            plt.close(fig)  # Close figure to free memory
            print(f"Saved gap plot to {output_file}")  # Confirm plot saved

    print("Gap plotting complete.")  # Indicate plotting completion

def calculate_position_size(volatility_metrics):
    """
    Dynamic position sizing based on volatility analysis
    """
    # Using volatility metrics from math analysis:
    # Short-term (10 period): 71.97
    # Medium-term (20 period): 57.73
    # Long-term (50 period): 76.86
    
    base_size = 1.0
    vol_ratio = volatility_metrics['volatility_10'] / volatility_metrics['volatility_50']
    
    if vol_ratio < 0.8:  # Lower short-term volatility
        return base_size * 1.2  # Increase size by 20%
    elif vol_ratio > 1.2:  # Higher short-term volatility
        return base_size * 0.8  # Decrease size by 20%
    return base_size

def calculate_stop_loss(price, volatility_metrics):
    """
    Dynamic stop loss based on volatility metrics
    """
    # Use volatility metrics for adaptive stops
    vol_10 = volatility_metrics['volatility_10']
    vol_20 = volatility_metrics['volatility_20']
    
    # Base stop on shorter-term volatility with context from medium term
    if vol_10 < vol_20:  # Lower current volatility
        stop_distance = vol_10 * 1.5
    else:  # Higher current volatility
        stop_distance = vol_10 * 2.0
        
    return price * (1 - stop_distance/price)

def calculate_signal_confidence(linear_regression_params, bayesian_prob, volatility_metrics):
    """
    Calculate signal confidence score based on mathematical metrics
    """
    confidence = 0.0
    
    # Strong regression metrics from math analysis
    confidence += (linear_regression_params['r_squared'] > 0.69) * 30
    confidence += (linear_regression_params['trend_slope'] > 2.0) * 30
    
    # Bayesian probability contribution
    confidence += (bayesian_prob > 0.53) * 20
    
    # Volatility context
    vol_ratio = volatility_metrics['volatility_10'] / volatility_metrics['volatility_50']
    confidence += (vol_ratio < 1.0) * 20
    
    return min(confidence, 100)

# Fixed calculate_cumulative_delta function
def calculate_cumulative_delta(df, volume_split_method='hlc'):
    """
    Calculate buying vs selling pressure using cumulative delta.
    """
    # Make a copy of the DataFrame to avoid modifying the original
    result_df = df.copy()
    
    # Ensure session_date exists
    if 'session_date' not in result_df.columns:
        if 'date' in result_df.columns:
            result_df['session_date'] = result_df['date']
        else:
            # Try to create session_date from index if it's a datetime index
            try:
                if isinstance(result_df.index, pd.DatetimeIndex):
                    result_df['session_date'] = result_df.index.date
                    print("Created 'session_date' from index for cumulative delta calculation")
                else:
                    # If we can't create session_date, warn and use a placeholder
                    print("WARNING: No date column found for session grouping in delta calculation")
                    result_df['session_date'] = pd.to_datetime('today').date()
            except:
                print("WARNING: Could not create session_date from index")
                result_df['session_date'] = pd.to_datetime('today').date()
    
    # Ensure contract exists (needed for groupby)
    if 'contract' not in result_df.columns:
        result_df['contract'] = 'default'
        print("Using default contract name for grouping in delta calculation")
    
    # Initialize columns for volume calculations
    result_df['buy_volume'] = 0.0
    result_df['sell_volume'] = 0.0
    result_df['delta'] = 0.0
    
    # Process each session and contract group separately
    for (session_date, contract), group in result_df.groupby(['session_date', 'contract']):
        idx = group.index
        
        # Random volume split method (FIXED close_change definition)
        if volume_split_method == 'random':
            for i in range(1, len(group)):
                curr_idx = idx[i]
                prev_idx = idx[i-1]
                # Define close_change before using it (this was missing)
                close_change = result_df.loc[curr_idx, 'close'] - result_df.loc[prev_idx, 'close']
                price_up = result_df.loc[curr_idx, 'close'] >= result_df.loc[prev_idx, 'close']
                vol = result_df.loc[curr_idx, 'volume']
                
                if close_change > 0:
                    buy_pct = 0.5 + 0.5 * min(abs(close_change) / (result_df.loc[curr_idx, 'high'] - result_df.loc[curr_idx, 'low']), 1)
                elif close_change < 0:
                    buy_pct = 0.5 - 0.5 * min(abs(close_change) / (result_df.loc[curr_idx, 'high'] - result_df.loc[curr_idx, 'low']), 1)
                else:
                    buy_pct = 0.5
                    
                result_df.loc[curr_idx, 'buy_volume'] = vol * buy_pct
                result_df.loc[curr_idx, 'sell_volume'] = vol * (1 - buy_pct)
                result_df.loc[curr_idx, 'delta'] = result_df.loc[curr_idx, 'buy_volume'] - result_df.loc[curr_idx, 'sell_volume']
        
        # HLC volume split method (unchanged)
        elif volume_split_method == 'hlc':
            for i in range(len(group)):
                curr_idx = idx[i]
                high = result_df.loc[curr_idx, 'high']
                low = result_df.loc[curr_idx, 'low']
                open_price = result_df.loc[curr_idx, 'open']
                close = result_df.loc[curr_idx, 'close']
                vol = result_df.loc[curr_idx, 'volume']
                
                price_range = high - low
                if price_range == 0:
                    if i > 0:
                        prev_idx = idx[i-1]
                        prev_close = result_df.loc[prev_idx, 'close']
                        buy_pct = 0.5 + 0.5 * ((close - prev_close) / (abs(close - prev_close) if close != prev_close else 1))
                    else:
                        buy_pct = 0.5
                else:
                    close_position = (close - low) / price_range
                    open_position = (open_price - low) / price_range
                    if close > open_price:
                        high_portion = (high - close) / price_range
                        middle_portion = (close - open_price) / price_range
                        low_portion = (open_price - low) / price_range
                        buy_pct = 0.5 * low_portion + 0.8 * middle_portion + 0.65 * high_portion
                    else:
                        high_portion = (high - open_price) / price_range
                        middle_portion = (open_price - close) / price_range
                        low_portion = (close - low) / price_range
                        buy_pct = 0.65 * high_portion + 0.2 * middle_portion + 0.35 * low_portion
                        
                result_df.loc[curr_idx, 'buy_volume'] = vol * buy_pct
                result_df.loc[curr_idx, 'sell_volume'] = vol * (1 - buy_pct)
                result_df.loc[curr_idx, 'delta'] = result_df.loc[curr_idx, 'buy_volume'] - result_df.loc[curr_idx, 'sell_volume']
        
        # Close-based volume split method (unchanged)
        elif volume_split_method == 'close':
            for i in range(1, len(group)):
                curr_idx = idx[i]
                prev_idx = idx[i-1]
                close_change = result_df.loc[curr_idx, 'close'] - result_df.loc[prev_idx, 'close']
                vol = result_df.loc[curr_idx, 'volume']
                
                if close_change > 0:
                    buy_pct = 0.5 + 0.5 * min(abs(close_change) / (result_df.loc[curr_idx, 'high'] - result_df.loc[curr_idx, 'low']), 1)
                elif close_change < 0:
                    buy_pct = 0.5 - 0.5 * min(abs(close_change) / (result_df.loc[curr_idx, 'high'] - result_df.loc[curr_idx, 'low']), 1)
                else:
                    buy_pct = 0.5
                    
                result_df.loc[curr_idx, 'buy_volume'] = vol * buy_pct
                result_df.loc[curr_idx, 'sell_volume'] = vol * (1 - buy_pct)
                result_df.loc[curr_idx, 'delta'] = result_df.loc[curr_idx, 'buy_volume'] - result_df.loc[curr_idx, 'sell_volume']
        
        # Calculate cumulative delta for this group
        cum_delta = 0
        for i in range(len(group)):
            cum_delta += result_df.loc[idx[i], 'delta']
            result_df.loc[idx[i], 'cum_delta'] = cum_delta
        
        # Calculate rolling 15-minute delta
        result_df.loc[idx, 'delta_15min'] = result_df.loc[idx, 'delta'].rolling(window=15, min_periods=1).sum()
    
    # Calculate percentile rank of 15-minute delta within each session/contract
    result_df['delta_15min_pctl'] = result_df.groupby(['session_date', 'contract'])['delta_15min'].transform(
        lambda x: x.rank(pct=True) * 100
    )
    
    return result_df

# Fixed plot_cumulative_delta function
def plot_cumulative_delta(df, date, contract=None, output_file=None):
    """
    Plot cumulative delta alongside price.
    """
    # 1. Filter data for the specified date
    if 'session_date' in df.columns:
        date_df = df[df['session_date'] == date]
    elif 'date' in df.columns:
        date_df = df[df['date'] == date]
    else:
        print(f"Error: No date column found in DataFrame for filtering")
        return
    
    # 2. Apply contract filter if provided
    if contract and 'contract' in date_df.columns:
        date_df = date_df[date_df['contract'] == contract]
    
    if len(date_df) == 0:
        print(f"No data found for date {date}" + (f" and contract {contract}" if contract else ""))
        return
    
    # 3. Handle index for plotting
    plot_df = date_df.copy()
    
    # Check if we need to set a timestamp index
    if not isinstance(plot_df.index, pd.DatetimeIndex):
        if 'timestamp' in plot_df.columns:
            try:
                plot_df = plot_df.set_index('timestamp')
            except Exception as e:
                print(f"Warning: Could not set timestamp as index: {str(e)}")
                # Continue with the existing index
    
    # 4. Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # 5. Plot price on top subplot
    ax1.plot(plot_df.index, plot_df['close'], color='black', label='Price') 
    
    # 6. Plot cumulative delta on bottom subplot
    ax2.plot(plot_df.index, plot_df['cum_delta'], color='blue', label='Cum. Delta')
    
    # 7. Add zero line to delta subplot
    ax2.axhline(y=0, color='red', linestyle='-', linewidth=0.5)
    
    # 8. Add 15-min delta threshold indicators
    if 'delta_15min' in plot_df.columns:
        delta_threshold = plot_df['delta_15min'].quantile(0.67)
        plot_df['above_threshold'] = plot_df['delta_15min'] > delta_threshold
        threshold_changes = plot_df['above_threshold'].diff().fillna(0) != 0
        threshold_times = plot_df[threshold_changes].index
        
        # 9. Mark threshold crossings with annotations
        for idx in threshold_times:
            row = plot_df.loc[idx]
            if row['above_threshold']:
                ax2.annotate('', xy=(idx, row['cum_delta']), xytext=(idx, row['cum_delta'] - 500),
                             arrowprops=dict(facecolor='green', shrink=0.05)) # Green arrow for above threshold
            else:
                ax2.annotate('', xy=(idx, row['cum_delta']), xytext=(idx, row['cum_delta'] + 500),
                             arrowprops=dict(facecolor='red', shrink=0.05)) # Red arrow for below threshold
    
    # 10. Set labels and titles
    contract_str = f" - {contract}" if contract else ""
    ax1.set_title(f"Price and Cumulative Delta - {date}{contract_str}", fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price', fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    ax2.set_title('Cumulative Delta', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time', fontweight='bold')
    ax2.set_ylabel('Delta', fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    # Format x-axis for datetime
    if isinstance(plot_df.index, pd.DatetimeIndex):
        plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    
    # 11. Save or display plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return None
    else:
        return fig, (ax1, ax2)
    
# Cell 7: VPOC Trend Validation (ADVANCED VERSION from VPOC Script - UPDATED for Readability and Syntax Correction)
def validate_vpoc_trend(vpocs, dates, lookback=20):
    """
    Validate VPOC trend using improved statistical methods (linear regression and run test).

    This function assesses the statistical validity of a VPOC trend by:
    1. Checking for sufficient data points.
    2. Calculating price differences to determine trend direction.
    3. Counting consecutive up and down moves.
    4. Performing linear regression to measure trend strength (R-squared, slope, p-value).
    5. Conducting a Run Test to assess randomness (p-value for runs).
    6. Combining results to determine if the trend is statistically valid and calculate confidence.

    Parameters:
    -----------
    vpocs : list
        List of VPOC prices
    dates : list
        List of corresponding dates
    lookback : int
        Number of days to analyze for trend

    Returns:
    --------
    dict
        Dictionary with trend validation results, including:
        - valid_trend (bool): True if trend is considered statistically valid
        - p_value (float): p-value from linear regression
        - run_test_p (float): p-value from run test
        - direction (str): Overall trend direction ('up', 'down', or 'neutral')
        - consecutive_up (int): Maximum consecutive up moves
        - consecutive_down (int): Maximum consecutive down moves
        - confidence (float): Confidence level in the valid trend (percentage)
        - slope (float): Slope of the linear regression line
        - r_squared (float): R-squared value from linear regression
    """

    # --- 1. Check for Sufficient Data Points ---
    if len(vpocs) < 5:  # Reduced minimum requirement from lookback to 5
        return { 'valid_trend': False, 'p_value': None, 'direction': None, 'consecutive_count': 0, 'confidence': 0, 'slope': None, 'r_squared': None }
    
    # Use as many data points as available, up to lookback limit
    recent_vpocs = vpocs[-min(lookback, len(vpocs)):]
    recent_dates = dates[-min(lookback, len(vpocs)):]

    # --- 2. Calculate Price Differences and Trend Direction ---
    diffs = [recent_vpocs[i] - recent_vpocs[i-1] for i in range(1, len(recent_vpocs))]
    pos_moves = sum(1 for d in diffs if d > 0)
    neg_moves = sum(1 for d in diffs if d < 0)
    zero_moves = sum(1 for d in diffs if d == 0)
    
    # More lenient direction determination - just needs more moves in one direction
    direction = 'up' if pos_moves > neg_moves else 'down' if neg_moves > pos_moves else 'neutral'

    # --- 3. Find Consecutive Moves (Streaks) ---
    current_up_streak = 0
    current_down_streak = 0
    max_up_streak = 0
    max_down_streak = 0
    for diff in diffs:
        if diff > 0:
            current_up_streak += 1
            current_down_streak = 0
            max_up_streak = max(max_up_streak, current_up_streak) # Update max up streak
        elif diff < 0:
            current_down_streak += 1
            current_up_streak = 0
            max_down_streak = max(max_down_streak, current_down_streak) # Update max down streak
        else: # diff == 0 (No change)
            # Consider no change as continuing the current streak (instead of resetting)
            pass

    # --- 4. Linear Regression for Trend Strength ---
    x = np.arange(len(recent_vpocs))
    y = np.array(recent_vpocs)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    r_squared = r_value ** 2

    # --- 5. Run Test for Randomness ---
    median = np.median(recent_vpocs)
    runs = [1 if v > median else 0 for v in recent_vpocs]
    runs_count = 1 # Initialize run count to 1 (for the first run)
    for i in range(1, len(runs)):
        if runs[i] != runs[i-1]: # Check for change in run sequence
            runs_count += 1

    n1 = sum(runs)
    n2 = len(runs) - n1
    expected_runs = (2 * n1 * n2) / (n1 + n2) + 1
    std_runs = np.sqrt((2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2)**2 * (n1 + n2 - 1)))
    run_test_p = 2 * (1 - stats.norm.cdf(abs((runs_count - expected_runs) / std_runs))) if std_runs > 0 else 1.0

    # --- 6. Less Stringent Trend Validation Criteria ---
    # Original criteria:
    # is_valid_trend = (
    #     (p_value < 0.05) and         # Significant linear relationship (p < 0.05)
    #     (run_test_p < 0.05) and      # Non-random pattern (p < 0.05) 
    #     (r_squared > 0.3) and        # At least moderate linear fit (R-squared > 0.3)
    #     (                            # Check for minimum consecutive moves in the trend direction
    #         (direction == 'up' and max_up_streak >= 3) or
    #         (direction == 'down' and max_down_streak >= 3)
    #     )
    # )
    
    # Relaxed criteria for trend validation
    is_valid_trend = (
        (p_value < 0.1) and          # More lenient p-value threshold
        (r_squared > 0.2) and        # Lower R-squared requirement
        (                            # Reduced consecutive moves requirement
            (direction == 'up' and max_up_streak >= 2) or
            (direction == 'down' and max_down_streak >= 2)
        )
    )
    
    # Alternative validation if basic conditions are met but statistical tests fail
    if not is_valid_trend and ((max_up_streak >= 3) or (max_down_streak >= 3)):
        is_valid_trend = True  # Consider valid if we have at least 3 consecutive moves in same direction
    
    # If trend is not valid, but we have a clear directional bias, still set direction
    if not is_valid_trend and direction == 'neutral':
        # Use overall slope as direction indicator
        direction = 'up' if slope > 0 else 'down'
    
    # Calculate confidence - more lenient scale
    if is_valid_trend:
        trend_strength = min(r_squared * 100, 100)  # R-squared as percentage
        statistical_significance = (1 - min(p_value, 0.5) * 2) * 100  # Convert p-value to confidence, capped at 50%
        streak_factor = (max(max_up_streak, max_down_streak) / 5) * 100  # Streak factor (5 would be 100%)
        confidence = (trend_strength + statistical_significance + streak_factor) / 3  # Weighted average
    else:
        # Even without full validation, assign some confidence based on direction strength
        consecutive_factor = max(max_up_streak, max_down_streak) * 10  # 10% per consecutive move
        confidence = min(consecutive_factor, 50)  # Cap at 50% if not fully validated

    return {
        'valid_trend': is_valid_trend,
        'p_value': p_value,
        'run_test_p': run_test_p,
        'direction': direction,
        'consecutive_up': max_up_streak,
        'consecutive_down': max_down_streak,
        'confidence': confidence,
        'slope': slope,
        'r_squared': r_squared
    }# %%

# Cell 9: Session Analysis (UPDATED for Readability - Output Directory Handling)
def analyze_session(df, date, contract=None):
    """
    Perform complete analysis for a single session.
    """
    print(f"\n===== Analyzing Session: {date} =====")
    
    # Handle different date column names
    if 'date' in df.columns:
        session_df = df[df['date'] == date]
    elif 'session_date' in df.columns:
        session_df = df[df['session_date'] == date]
    else:
        print(f"Error: No date column found in DataFrame")
        return None
        
    # Apply contract filter if provided
    if contract and 'contract' in session_df.columns: 
        session_df = session_df[session_df['contract'] == contract]
        print(f"Filtered to contract {contract}: {len(session_df)} rows")
    
    if len(session_df) == 0: 
        print(f"No data found for session {date}")
        return None
    
    print(f"Session data: {len(session_df)} rows")

    # --- Volume Profile and Key Level Calculations ---
    volume_profile = calculate_volume_profile(session_df)
    vpoc = find_vpoc(volume_profile)
    val, vah, va_volume_pct = find_value_area(volume_profile)

    lvn_zones = identify_lvn_zones(volume_profile)
    
    # Ensure session_df has necessary date columns for cumulative delta
    if 'session_date' not in session_df.columns:
        session_df['session_date'] = date
    
    # Ensure contract column exists for cumulative delta
    if 'contract' not in session_df.columns:
        session_df['contract'] = contract if contract else 'default'
        
    session_with_delta = calculate_cumulative_delta(session_df)

    # --- Prepare Result Dictionary ---
    result = { 
        'date': date, 
        'contract': contract, 
        'vpoc': vpoc, 
        'value_area_low': val, 
        'value_area_high': vah, 
        'value_area_width': vah - val, 
        'total_volume': session_df['volume'].sum(), 
        'lvn_zones': lvn_zones, 
        'volume_profile': volume_profile, 
        'session_data': session_with_delta 
    }

    # --- Output Directory Handling (Corrected and Readability Improved) ---
    output_dir = STRATEGY_DIR  # Start with base strategy directory
    if contract:  # If a contract is specified
        output_dir = os.path.join(STRATEGY_DIR, contract)  # Create contract-specific subdirectory
    if not os.path.exists(output_dir):  # Check if output directory exists
        os.makedirs(output_dir)  # Create directory if it doesn't exist

    # --- Generate and Save Plots ---
    profile_file = os.path.join(output_dir, f"profile_lvn_{date}.png")
    plot_lvn_zones(volume_profile, lvn_zones, vpoc, val, vah, f"Session {date}" + (f" - {contract}" if contract else ""), profile_file)
    print(f"Saved volume profile with LVN zones to {profile_file}")

    delta_file = os.path.join(output_dir, f"delta_{date}.png")
    plot_cumulative_delta(session_with_delta, date, contract, delta_file)
    print(f"Saved cumulative delta plot to {delta_file}")

    return result


# %%
# Cell 10: Main Analysis Function (MODIFIED to use load_futures_data)
def run_analysis():
    """
    Main function to run all strategy components with enhanced mathematical insights.
    """
    print("===== ES Futures Strategy Components Analysis =====")

    # 1. Load cleaned data using data_loader module
    print("\n===== STEP 1: LOADING FUTURES DATA =====")
    data_dir = DATA_DIR  # Use DATA_DIR configuration
    df = load_futures_data(DATA_DIR) # Load data using data_loader

    if df is None:
        print("Error: Failed to load futures data using data_loader. Exiting Strategy Script.")
        return

    # 2. Load VPOC data if available (from previous analysis)
    vpoc_file = os.path.join(OUTPUT_DIR, 'RTH_vpoc_data.csv')
    if os.path.exists(vpoc_file):
        vpoc_df = pd.read_csv(vpoc_file)
        if 'date' in vpoc_df.columns:
            vpoc_df['date'] = pd.to_datetime(vpoc_df['date']).dt.date
            print(f"Loaded VPOC data with {len(vpoc_df)} sessions")

            # Validate VPOC trends with improved validation function
            vpoc_trend = validate_vpoc_trend(vpoc_df['vpoc'].tolist(), vpoc_df['date'].tolist())
    else:
        print(f"VPOC data file not found at {vpoc_file}")
        vpoc_df = None
        vpoc_trend = {'direction': 'unknown', 'confidence': 0, 'valid_trend': False}

    # 3. Identify overnight gaps
    gaps_df = identify_gaps(df)

    # 4. Load Mathematical Analysis Data
    print("\n===== LOADING MATHEMATICAL ANALYSIS DATA =====")
    import glob
    
    # Find the most recent momentum file
    momentum_files = glob.glob(os.path.join(MATH_OUTPUT_DIR, 'vpoc_analysis_*_momentum.csv'))
    momentum_data = None
    if momentum_files:
        latest_momentum_file = max(momentum_files, key=os.path.getctime)
        momentum_data = pd.read_csv(latest_momentum_file, parse_dates=['date'])
        print(f"Loaded momentum data from {latest_momentum_file}")

    # Find the most recent mathematical analysis file
    math_analysis_files = glob.glob(os.path.join(MATH_OUTPUT_DIR, 'vpoc_analysis_*.csv'))
    linear_regression_params = {'trend_slope': 0, 'r_squared': 0}
    
    if math_analysis_files:
        latest_math_file = max(math_analysis_files, key=os.path.getctime)
        try:
            math_analysis_df = pd.read_csv(latest_math_file)
            print(f"Columns in mathematical analysis file: {math_analysis_df.columns.tolist()}")
            
            if 'linear_regression_trend_slope' in math_analysis_df.columns and 'linear_regression_r_squared' in math_analysis_df.columns:
                linear_regression_params = {
                    'trend_slope': math_analysis_df.iloc[0]['linear_regression_trend_slope'],
                    'r_squared': math_analysis_df.iloc[0]['linear_regression_r_squared']
                }
                print(f"Loaded mathematical analysis from {latest_math_file}")
                print(f"Trend Slope: {linear_regression_params['trend_slope']}")
                print(f"R-squared: {linear_regression_params['r_squared']}")
        except Exception as e:
            print(f"Error loading mathematical analysis: {e}")

    # 5. Run signal failure analysis
    failure_analysis = analyze_signal_failures(vpoc_df, momentum_data, linear_regression_params)
    if not any(failure_analysis.values()):
        print("⚠️ All signal generation conditions are failing")
    
    # 6. Generate Enhanced Trading Signals
    print("\n===== STEP 5: GENERATING ENHANCED TRADING SIGNALS =====")
    
    if vpoc_df is not None:
        print("\n===== GENERATING COMBINED TRADING SIGNALS =====")
        
        # Debug momentum data
        print("\nMomentum Data Check:")
        if momentum_data is not None and not momentum_data.empty:
            print(f"Momentum data shape: {momentum_data.shape}")
            print(f"Momentum data columns: {momentum_data.columns.tolist()}")
            print(f"Average confidence: {momentum_data['window_confidence'].mean():.2f}")
        else:
            print("No momentum data available")

        # Enhanced signal generation with mathematical insights
        combined_signals = generate_combined_signals(
            df, 
            vpoc_df, 
            gaps_df, 
            momentum_data if momentum_data is not None else pd.DataFrame(),
            linear_regression_params,
            min_momentum_confidence=0.7,
            min_bayesian_up_prob=0.52
        )
        
        if len(combined_signals) > 0:
            signals_file = os.path.join(STRATEGY_DIR, 'trading_signals.csv')
            combined_signals.to_csv(signals_file, index=False)
            print(f"\nSaved {len(combined_signals)} trading signals to {signals_file}")
        else:
            print("\nNo trading signals generated")

    print("\n===== Strategy Components Analysis Complete =====")
    print(f"All results saved to {STRATEGY_DIR}")


# Function to generate trading signals based on custom trading rules
def generate_combined_signals(df, vpoc_data, gaps_df, momentum_data, linear_regression_params,
                            lookback_days=10, htf_lookback_weeks=3,
                            min_momentum_confidence=0.3,  # Lowered threshold based on math analysis
                            min_bayesian_up_prob=0.52,
                            math_analysis_df=None):
    """
    Enhanced signal generation incorporating mathematical insights
    """
    print("\nSignal Generation Debug:")
    print(f"Lookback days: {lookback_days}")
    
    # Unpack linear regression parameters with debug
    trend_slope = linear_regression_params.get('trend_slope', 0)
    trend_r_squared = linear_regression_params.get('r_squared', 0)
    print(f"Trend slope: {trend_slope:.2f}, R-squared: {trend_r_squared:.2f}")
    
    enhanced_signals = []

    # Get Bayesian probability from math analysis data
    if math_analysis_df is not None and 'bayesian_probabilities_probability_up' in math_analysis_df.columns:
        bayesian_prob_up = math_analysis_df.iloc[0]['bayesian_probabilities_probability_up']
    else:
        bayesian_prob_up = 0.5305  # Default value from math analysis

    # Process each day with enhanced validation
    for i in range(lookback_days, len(vpoc_data)):
        current_row = vpoc_data.iloc[i]
        current_date = current_row['date']
        
        print(f"\nAnalyzing session: {current_date}")
        
        # Get volatility metrics for the session
        volatility_metrics = {
            'volatility_10': current_row.get('volatility_10', 71.97),  # Default from math analysis
            'volatility_20': current_row.get('volatility_20', 57.73),
            'volatility_50': current_row.get('volatility_50', 76.86)
        }
        
        # Calculate signal confidence score
        signal_confidence = calculate_signal_confidence(
            linear_regression_params,
            bayesian_prob_up,
            volatility_metrics
        )
        
        # New dynamic conditions based on math analysis
        signal_conditions = (
            trend_slope > 0 and                    # Trend is positive
            trend_r_squared > 0.6 and              # Strong R-squared from math
            (
                (volatility_metrics['volatility_10'] < volatility_metrics['volatility_50']) or  # Lower short-term volatility
                (bayesian_prob_up > min_bayesian_up_prob)                                       # Strong Bayesian probability
            ) and
            signal_confidence > 60                  # Overall confidence above 60%
        )

        if signal_conditions:
            current_vpoc = current_row['vpoc']
            current_val = current_row['value_area_low']
            current_vah = current_row['value_area_high']
            
            # Calculate position size based on volatility
            position_size = calculate_position_size(volatility_metrics)
            
            # Calculate adaptive stop loss
            long_stop = calculate_stop_loss(current_val, volatility_metrics)
            short_stop = calculate_stop_loss(current_vah, volatility_metrics)
            
            # Long Signal Generation
            long_signal = {
                'date': current_date,
                'signal': 'BUY',
                'price': current_val,
                'stop_loss': long_stop,
                'target': current_vah,
                'position_size': position_size,
                'signal_type': 'Enhanced-VPOC',
                'confidence': signal_confidence,
                'reason': f"Trend:{trend_slope:.2f}, R²:{trend_r_squared:.2f}, Conf:{signal_confidence:.0f}%"
            }
            
            # Short Signal Generation
            short_signal = {
                'date': current_date,
                'signal': 'SELL',
                'price': current_vah,
                'stop_loss': short_stop,
                'target': current_val,
                'position_size': position_size,
                'signal_type': 'Enhanced-VPOC',
                'confidence': signal_confidence,
                'reason': f"Trend:{trend_slope:.2f}, R²:{trend_r_squared:.2f}, Conf:{signal_confidence:.0f}%"
            }
            
            enhanced_signals.extend([long_signal, short_signal])
            print(f"Generated signals for {current_date} - Confidence: {signal_confidence:.0f}%")
        else:
            print(f"No signals generated for {current_date} - conditions not met")

    # Convert to DataFrame and sort
    enhanced_signals_df = pd.DataFrame(enhanced_signals)
    if len(enhanced_signals_df) > 0:
        enhanced_signals_df = enhanced_signals_df.sort_values('date')
    
    print(f"\nTotal signals generated: {len(enhanced_signals_df)}")
    return enhanced_signals_df

# %%

def analyze_signal_failures(vpoc_data, momentum_data, linear_regression_params, min_momentum_confidence=0.7):
    """
    Analyze why signals aren't being generated
    """
    print("\nSignal Generation Failure Analysis:")
    
    # Check trend parameters
    trend_slope = linear_regression_params.get('trend_slope', 0)
    trend_r_squared = linear_regression_params.get('r_squared', 0)
    
    if trend_slope <= 0:
        print("❌ Trend slope is not positive")
    if trend_r_squared <= 0.6:
        print("❌ R-squared is below threshold")
        
    # Check momentum data
    if momentum_data is None or momentum_data.empty:
        print("❌ No momentum data available")
    else:
        avg_confidence = momentum_data['window_confidence'].mean()
        print(f"Average momentum confidence: {avg_confidence:.2f}")
        if avg_confidence < min_momentum_confidence:
            print(f"❌ Average momentum confidence ({avg_confidence:.2f}) below threshold ({min_momentum_confidence})")
    
    # Check VPOC data
    if vpoc_data is None or vpoc_data.empty:
        print("❌ No VPOC data available")
    else:
        print(f"VPOC sessions available: {len(vpoc_data)}")

    return {
        'trend_valid': trend_slope > 0 and trend_r_squared > 0.6,
        'momentum_valid': avg_confidence >= min_momentum_confidence if 'avg_confidence' in locals() else False,
        'data_available': all(x is not None and not x.empty for x in [vpoc_data, momentum_data])
    }

# %%
if __name__ == "__main__":
    # Run the main analysis function
    run_analysis()