"""
Feature engineering utilities for futures ML models.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression

def prepare_features_and_labels(data, use_feature_selection=True, max_features=15):
    """
    Prepares features and labels for model training.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Raw futures data
    use_feature_selection : bool
        Whether to perform feature selection
    max_features : int
        Maximum number of features to select
        
    Returns:
    --------
    X : numpy.ndarray
        Feature matrix
    y : numpy.ndarray
        Target labels
    feature_columns : list
        Names of selected features
    scaler : StandardScaler
        Fitted scaler for feature normalization
    """
    # Process raw data into session features
    features_df = _extract_session_features(data)
    
    # Define target variable (next day's return)
    features_df['target'] = features_df['close_change_pct'].shift(-1)
    features_df = features_df.dropna()
    
    # Split features and target
    target_col = 'target'
    exclude_cols = ['date', target_col]
    feature_columns = [col for col in features_df.columns if col not in exclude_cols]
    
    X = features_df[feature_columns]
    y = features_df[target_col]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply feature selection if requested
    if use_feature_selection and X.shape[1] > max_features:
        selector = SelectKBest(f_regression, k=max_features)
        X_selected = selector.fit_transform(X_scaled, y)
        
        # Get selected feature names
        selected_mask = selector.get_support()
        feature_columns = [feature_columns[i] for i in range(len(feature_columns)) 
                           if selected_mask[i]]
        
        return X_selected, y.values, feature_columns, scaler
    
    return X_scaled, y.values, feature_columns, scaler

def _extract_session_features(data):
    """Extract session-level features from raw tick data."""
    features_list = []
    
    # Group by date
    for date, session_data in data.groupby('date'):
        if len(session_data) < 2:  # Skip short sessions
            continue
            
        # Extract basic session stats
        session_open = session_data['open'].iloc[0]
        session_high = session_data['high'].max()
        session_low = session_data['low'].min()
        session_close = session_data['close'].iloc[-1]
        session_volume = session_data['volume'].sum()
        
        # Calculate price changes
        close_change = session_close - session_open
        close_change_pct = (close_change / session_open) * 100 if session_open != 0 else 0
        
        # Calculate price range
        price_range = session_high - session_low
        range_pct = (price_range / session_open) * 100 if session_open != 0 else 0
        
        # Compile features
        session_features = {
            'date': date,
            'session_open': session_open,
            'session_high': session_high,
            'session_low': session_low,
            'session_close': session_close,
            'total_volume': session_volume,
            'close_change': close_change,
            'close_change_pct': close_change_pct,
            'price_range': price_range,
            'range_pct': range_pct,
        }
        
        features_list.append(session_features)
    
    # Create DataFrame and ensure date is properly formatted
    if not features_list:
        return pd.DataFrame()
        
    features_df = pd.DataFrame(features_list)
    
    # Sort by date
    features_df = features_df.sort_values(by='date').reset_index(drop=True)
    
    return features_df