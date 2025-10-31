# %%
import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
from scipy import stats
from DATA_LOADER import load_futures_data  # Import the load_futures_data function
import matplotlib.pyplot as plt

# Configuration
BASE_DIR = '/home/lr/Documents/FUTURUES_PROJECT/futures_vpoc_backtest'
DATA_DIR = os.path.join(BASE_DIR, 'DATA')
OUTPUT_DIR = os.path.join(BASE_DIR, 'RESULTS')
CLEANED_DATA_DIR = os.path.join(BASE_DIR, 'DATA/CLEANED')  # Directory for cleaned data
SESSION_TYPE = 'RTH'  # Regular Trading Hours
PRICE_PRECISION = 0.25  # Price increment for volume profile bins
# %%
# VPOC Analysis Functions
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

# %%
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

# %%
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

    # Add price levels until we reach the target volume
    for _, row in sorted_profile.iterrows():
        value_area_prices.append(row['price_level'])
        cum_volume += row['volume']

        if cum_volume >= target_volume:
            break

    # Return min and max of the value area
    return min(value_area_prices), max(value_area_prices)

# %%
def plot_volume_profile(volume_profile, vpoc, val, vah, session_title, output_file=None):
    """Plot volume profile with VPOC and value area."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Check if we have enough data points for plotting
    if len(volume_profile) < 2:
        print(f"Warning: Not enough price levels in profile for {session_title}. Skipping plot.")
        if output_file:
            plt.close(fig)
            return None
        return fig, ax

    # Calculate height for bars (use price precision if diff fails)
    try:
        bar_height = volume_profile['price_level'].diff().iloc[1]
        if pd.isna(bar_height) or bar_height <= 0:
            bar_height = PRICE_PRECISION  # Fall back to price precision
    except (IndexError, KeyError):
        bar_height = PRICE_PRECISION  # Fall back to price precision

    # Plot volume profile (horizontal bars)
    ax.barh(volume_profile['price_level'], volume_profile['volume'],
           height=bar_height,
           color='steelblue', alpha=0.7)

    # Highlight VPOC (with error handling)
    try:
        vpoc_idx = (volume_profile['price_level'] - vpoc).abs().argmin()
        ax.barh(volume_profile.iloc[vpoc_idx]['price_level'],
               volume_profile.iloc[vpoc_idx]['volume'],
               height=bar_height,
               color='crimson', alpha=0.8)
    except (IndexError, KeyError) as e:
        print(f"Warning: Could not highlight VPOC in {session_title}. Error: {e}")

    # Highlight value area (with error handling)
    try:
        val_mask = (volume_profile['price_level'] >= val) & (volume_profile['price_level'] <= vah)
        if val_mask.any():
            ax.barh(volume_profile.loc[val_mask, 'price_level'],
                   volume_profile.loc[val_mask, 'volume'],
                   height=bar_height,
                   color='cornflowerblue', alpha=0.5)
    except Exception as e:
        print(f"Warning: Could not highlight value area in {session_title}. Error: {e}")

    # Add annotations
    ax.axhline(y=vpoc, color='crimson', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.axhline(y=val, color='navy', linestyle=':', alpha=0.7, linewidth=1.5)
    ax.axhline(y=vah, color='navy', linestyle=':', alpha=0.7, linewidth=1.5)

    # Add text labels
    try:
        max_volume = volume_profile['volume'].max() * 1.05
        ax.text(max_volume, vpoc, f' VPOC: {vpoc:.2f}', verticalalignment='center',
               color='crimson', fontweight='bold')
        ax.text(max_volume, val, f' VAL: {val:.2f}', verticalalignment='center',
               color='navy', fontweight='bold')
        ax.text(max_volume, vah, f' VAH: {vah:.2f}', verticalalignment='center',
               color='navy', fontweight='bold')
    except Exception as e:
        print(f"Warning: Could not add text labels in {session_title}. Error: {e}")

    # Set labels and title
    ax.set_xlabel('Volume')
    ax.set_ylabel('Price')
    ax.set_title(f'Volume Profile - {session_title}', fontsize=14)

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.6)

    # Invert y-axis for natural price direction (higher prices at top)
    ax.invert_yaxis()

    plt.tight_layout()

    if output_file:
        try:
            plt.savefig(output_file, dpi=300)
        except Exception as e:
            print(f"Error saving plot to {output_file}: {e}")
        plt.close(fig)
        return None
    else:
        return fig, ax

def plot_vpoc_migrations(dates, vpocs, session_type, output_file=None):
    """Plot VPOC migrations over time."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Convert dates to datetime if they're strings
    if isinstance(dates[0], str):
        date_objects = [pd.to_datetime(date) for date in dates]
    else:
        date_objects = dates

    # Plot VPOC line
    ax.plot(date_objects, vpocs, '-o', linewidth=2, markersize=5, color='mediumblue')

    # Add markers for VPOC changes
    for i in range(1, len(dates)):
        prev_vpoc = vpocs[i-1]
        curr_vpoc = vpocs[i]

        # Determine color based on direction (green for up, red for down)
        if curr_vpoc > prev_vpoc:
            color = 'green'
            direction = 'up'
        elif curr_vpoc < prev_vpoc:
            color = 'red'
            direction = 'down'
        else:
            color = 'gray'
            direction = 'none'

        # Draw arrow for significant migrations
        if abs(curr_vpoc - prev_vpoc) > 0.5:  # Only show significant migrations
            ax.annotate('',
                       xy=(date_objects[i], curr_vpoc),
                       xytext=(date_objects[i-1], prev_vpoc),
                       arrowprops=dict(arrowstyle='->',
                                     color=color,
                                     lw=1.5,
                                     alpha=0.7))

    # Format x-axis for dates
    plt.gcf().autofmt_xdate()

    # Set labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('VPOC Price')
    ax.set_title(f'VPOC Migrations - {session_type} Sessions', fontsize=14)

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300)
        plt.close(fig)
        return None
    else:
        return fig, ax

# %%
# %%
def detect_vpoc_migration(previous_vpoc, current_vpoc, min_migration_pct=0.4, avg_price=None):
    """
    Detect if VPOC has migrated significantly, using adaptive thresholds.

    Parameters:
    -----------
    previous_vpoc : float
        The VPOC price from the previous session
    current_vpoc : float
        The VPOC price from the current session
    min_migration_pct : float
        Minimum migration as a percentage of average price
    avg_price : float
        Average price level for normalization (optional)

    Returns:
    --------
    tuple
        (migrated, direction, magnitude)
    """
    diff = current_vpoc - previous_vpoc

    # Use absolute threshold if avg_price not provided
    if avg_price is None:
        avg_price = (previous_vpoc + current_vpoc) / 2
        min_migration = min_migration_pct * avg_price / 100
    else:
        min_migration = min_migration_pct * avg_price / 100

    magnitude = abs(diff)

    if magnitude >= min_migration:
        direction = 'up' if diff > 0 else 'down'
        return True, direction, magnitude
    else:
        return False, 'none', magnitude

def find_migration_trends(dates, vpocs, min_consecutive=3, min_migration=1.0):
    """Find trends in VPOC migrations."""
    if len(dates) < min_consecutive + 1:
        return []

    trends = []
    current_direction = None
    consecutive_count = 0
    start_idx = 0

    for i in range(1, len(dates)):
        prev_vpoc = vpocs[i-1]
        curr_vpoc = vpocs[i]

        migrated, direction = detect_vpoc_migration(prev_vpoc, curr_vpoc, min_migration)

        if migrated and direction != 'none':
            if current_direction is None:
                # Start new trend
                current_direction = direction
                consecutive_count = 1
                start_idx = i - 1
            elif direction == current_direction:
                # Continue trend
                consecutive_count += 1
            else:
                # Direction changed, check if previous trend is significant
                if consecutive_count >= min_consecutive:
                    trend_start = dates[start_idx]
                    trend_end = dates[i-1]

                    trends.append({
                        'start_date': trend_start,
                        'end_date': trend_end,
                        'direction': current_direction,
                        'consecutive_count': consecutive_count,
                        'vpoc_start': vpocs[start_idx],
                        'vpoc_end': vpocs[i-1],
                        'vpoc_change': vpocs[i-1] - vpocs[start_idx]
                    })

                # Start new trend
                current_direction = direction
                consecutive_count = 1
                start_idx = i - 1
        else:
            # No migration, check if previous trend is significant
            if current_direction is not None and consecutive_count >= min_consecutive:
                trend_start = dates[start_idx]
                trend_end = dates[i-1]

                trends.append({
                    'start_date': trend_start,
                    'end_date': trend_end,
                    'direction': current_direction,
                    'consecutive_count': consecutive_count,
                    'vpoc_start': vpocs[start_idx],
                    'vpoc_end': vpocs[i-1],
                    'vpoc_change': vpocs[i-1] - vpocs[start_idx]
                })

            # Reset trend
            current_direction = None
            consecutive_count = 0

    # Check for trend at the end of the data
    if current_direction is not None and consecutive_count >= min_consecutive:
        trend_start = dates[start_idx]
        trend_end = dates[-1]

        trends.append({
            'start_date': trend_start,
            'end_date': trend_end,
            'direction': current_direction,
            'consecutive_count': consecutive_count,
            'vpoc_start': vpocs[start_idx],
            'vpoc_end': vpocs[-1],
            'vpoc_change': vpocs[-1] - vpocs[start_idx]
        })

    return trends

# %%
def validate_vpoc_trend(vpocs, dates, lookback=20):
    """
    Validate VPOC trend using improved statistical methods.

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
        Dictionary with trend validation results
    """
    # Use only the most recent data for analysis
    if len(vpocs) > lookback:
        recent_vpocs = vpocs[-lookback:]
        recent_dates = dates[-lookback:]
    else:
        recent_vpocs = vpocs
        recent_dates = dates

    if len(recent_vpocs) < 5:  # Need enough data points for meaningful analysis
        return {
            'valid_trend': False,
            'p_value': None,
            'direction': None,
            'consecutive_count': 0,
            'confidence': 0,
            'slope': None,
            'r_squared': None
        }

    # Calculate price differences
    diffs = [recent_vpocs[i] - recent_vpocs[i-1] for i in range(1, len(recent_vpocs))]

    # Count positive, negative, and zero moves
    pos_moves = sum(1 for d in diffs if d > 0)
    neg_moves = sum(1 for d in diffs if d < 0)
    zero_moves = sum(1 for d in diffs if d == 0)

    # Determine overall direction
    direction = 'up' if pos_moves > neg_moves else 'down' if neg_moves > pos_moves else 'neutral'

    # Find consecutive moves
    current_up_streak = 0
    current_down_streak = 0
    max_up_streak = 0
    max_down_streak = 0

    for diff in diffs:
        if diff > 0:
            current_up_streak += 1
            current_down_streak = 0
            max_up_streak = max(max_up_streak, current_up_streak)
        elif diff < 0:
            current_down_streak += 1
            current_up_streak = 0
            max_down_streak = max(max_down_streak, current_down_streak)
        else:
            # No change case
            current_up_streak = 0
            current_down_streak = 0

    # Linear regression to get trend strength
    x = np.arange(len(recent_vpocs))
    y = np.array(recent_vpocs)

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    r_squared = r_value ** 2

    # Calculate run test p-value
    # More robust test for randomness than chi-square
    median = np.median(recent_vpocs)
    runs = [1 if v > median else 0 for v in recent_vpocs]

    # Count runs
    runs_count = 1
    for i in range(1, len(runs)):
        if runs[i] != runs[i-1]:
            runs_count += 1

    n1 = sum(runs)
    n2 = len(runs) - n1

    # Expected runs and standard deviation under random hypothesis
    expected_runs = (2 * n1 * n2) / (n1 + n2) + 1
    std_runs = np.sqrt((2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2)**2 * (n1 + n2 - 1)))

    # Z-score
    if std_runs > 0:
        z = (runs_count - expected_runs) / std_runs
        run_test_p = 2 * (1 - stats.norm.cdf(abs(z)))  # Two-tailed test
    else:
        run_test_p = 1.0

    # Combined validation
    is_valid_trend = (
        (p_value < 0.05) and  # Significant linear relationship
        (run_test_p < 0.05) and  # Non-random pattern
        (r_squared > 0.3) and  # At least moderate fit
        (
            (direction == 'up' and max_up_streak >= 3) or
            (direction == 'down' and max_down_streak >= 3)
        )
    )

    # Calculate confidence (combined measure)
    if is_valid_trend:
        trend_strength = min(r_squared * 100, 100)  # R-squared as percentage
        statistical_significance = (1 - p_value) * 100  # Convert p-value to confidence
        confidence = (trend_strength + statistical_significance) / 2  # Average
    else:
        confidence = 0

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
    }

# %%
# %%
def analyze_vpoc(df, session_type=SESSION_TYPE, contract_filter=None):
    """Run improved VPOC analysis on the provided dataframe."""
    print(f"\n===== VPOC ANALYSIS ({session_type} Sessions) =====")

    # Filter to selected session type
    session_df = df[df['session'] == session_type]
    print(f"Filtered to {session_type} sessions: {len(session_df)} rows")

    # Apply contract filter if provided
    if contract_filter:
        session_df = session_df[session_df['contract'] == contract_filter]
        print(f"Filtered to contract {contract_filter}: {len(session_df)} rows")
        # Create contract-specific output directory
        contract_output_dir = os.path.join(OUTPUT_DIR, contract_filter)
        if not os.path.exists(contract_output_dir):
            os.makedirs(contract_output_dir)
            print(f"Created directory: {contract_output_dir}")
    else:
        contract_output_dir = OUTPUT_DIR

    # Group by date
    grouped = session_df.groupby('date')
    print(f"Found {len(grouped)} unique {session_type} sessions")

    # Process each session
    print(f"\nCalculating volume profiles and VPOCs...")

    results = []
    dates = []
    vpocs = []
    avg_price = None  # Will calculate this for adaptive migration detection

    for date, group in grouped:
        # Skip sessions with too few data points
        if len(group) < 5:  # Adjust this threshold as needed
            print(f"Skipping session {date} with only {len(group)} data points")
            continue

        try:
            # Calculate volume profile
            volume_profile = calculate_volume_profile(group)

            # Skip sessions with empty volume profiles
            if len(volume_profile) < 2:
                print(f"Skipping session {date} with insufficient price levels")
                continue

            # Find VPOC with improved method
            vpoc = find_vpoc(volume_profile, use_smoothing=True)

            # Calculate value area with improved method
            val, vah, va_volume_pct = find_value_area(volume_profile, value_area_pct=0.7, use_smoothing=True)

            # Calculate typical price for this session
            typical_price = group[['high', 'low', 'close']].mean(axis=1).mean()

            # Update running average price (for adaptive migration detection)
            if avg_price is None:
                avg_price = typical_price
            else:
                avg_price = 0.95 * avg_price + 0.05 * typical_price  # Exponential moving average

            # Store results
            results.append({
                'date': date,
                'vpoc': vpoc,
                'value_area_low': val,
                'value_area_high': vah,
                'value_area_width': vah - val,
                'value_area_pct': va_volume_pct,
                'volume_profile': volume_profile,
                'total_volume': group['volume'].sum(),
                'typical_price': typical_price
            })

            dates.append(date)
            vpocs.append(vpoc)
        except Exception as e:
            print(f"Error processing session {date}: {str(e)}")
            continue

    # Call the improved validation function
    trend_analysis = validate_vpoc_trend(vpocs, dates)
    print("\nVPOC Trend Analysis:")
    print(f"Direction: {trend_analysis['direction']}")
    print(f"Valid Trend: {trend_analysis['valid_trend']}")
    print(f"Confidence: {trend_analysis['confidence']:.2f}%")
    print(f"R-squared: {trend_analysis['r_squared']:.3f}")
    print(f"P-value: {trend_analysis['p_value']:.4f}")

    # Save the trend analysis
    trend_output_file = os.path.join(contract_output_dir, f'{contract_filter or "all"}_vpoc_trend_validation.csv')
    pd.DataFrame([trend_analysis]).to_csv(trend_output_file, index=False)

    # Rest of the function remains similar...
    # [Save VPOC data, plot migrations, etc.]
    # Save VPOC data to CSV

    vpoc_data = [{
        'date': r['date'],
        'vpoc': r['vpoc'],
        'value_area_low': r['value_area_low'],
        'value_area_high': r['value_area_high'],
        'value_area_width': r['value_area_width'],
        'total_volume': r['total_volume']
    } for r in results]

    contract_prefix = f"{contract_filter}_" if contract_filter else ""
    vpoc_df = pd.DataFrame(vpoc_data)
    vpoc_output_file = os.path.join(contract_output_dir, f'{contract_prefix}{session_type}_vpoc_data.csv')
    vpoc_df.to_csv(vpoc_output_file, index=False)
    print(f"Saved VPOC data to {vpoc_output_file}")

    return vpoc_df, trend_analysis

# %%
def main():
    """Main function to run the complete pipeline."""
    print("ES Futures Data Processing Pipeline")
    print("===================================")

    # ===== MODIFIED STEP 1: Load futures data using data_loader =====
    print("\n===== STEP 1: LOADING FUTURES DATA =====")
    data_dir = DATA_DIR  # Use DATA_DIR configuration
    merged_data = load_futures_data(data_dir) # Load data using data_loader

    if merged_data is None:
        print("Error: Failed to load futures data using data_loader. Exiting.")
        return
    print("Futures data loaded successfully.")
    # ===============================================================

    # Step 2: Run VPOC analysis on the whole dataset (all contracts combined)
    vpoc_data, trends = analyze_vpoc(merged_data)

    # Step 3: Run separate VPOC analysis for each contract
    print("\n===== INDIVIDUAL CONTRACT ANALYSIS =====")
    contracts = merged_data['contract'].unique()
    print(f"Found {len(contracts)} contracts to analyze individually")

    for contract in contracts:
        print(f"\nAnalyzing contract: {contract}")
        contract_vpoc_data, contract_trends = analyze_vpoc(merged_data, contract_filter=contract)

    print("\n===== PIPELINE EXECUTION COMPLETE =====")
    print(f"All data processed and results saved to {OUTPUT_DIR}")

# %%
# Run the main pipeline
main()