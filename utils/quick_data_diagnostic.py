#!/usr/bin/env python3
"""
Quick diagnostic to identify NaN/infinity issues in training data
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('/workspace')

def main():
    print("🔍 Loading data for diagnostic...")
    df = pd.read_csv('/workspace/DATA/MERGED/merged_es_vix_test.csv')
    print(f"📊 Original data shape: {df.shape}")

    # Check basic columns
    print("\n📋 Original columns:", df.columns.tolist())

    # Check for immediate issues
    print(f"\n🚨 NaN values in original data:")
    print(df.isnull().sum())

    print(f"\n📈 Basic stats for key columns:")
    key_cols = ['close', 'volume', 'vwap', 'vix']
    for col in key_cols:
        if col in df.columns:
            print(f"  {col}: min={df[col].min():.6f}, max={df[col].max():.6f}, mean={df[col].mean():.6f}")

    # Add our target features manually to see what happens
    print(f"\n🔧 Creating features manually...")
    df['close_change_pct'] = df['close'].pct_change()
    df['vwap'] = (df['close'] * df['volume']).rolling(15, min_periods=1).mean() / df['volume'].rolling(15, min_periods=1).sum()
    df['price_range'] = (df['high'] - df['low']) / df['close']
    df['price_mom_3d'] = df['close'].pct_change(3)
    df['price_mom_5d'] = df['close'].pct_change(5)

    # VWAP features (these might be causing the NaN issues)
    df['vwap_15'] = (df['close'] * df['volume']).rolling(15, min_periods=1).mean() / df['volume'].rolling(15, min_periods=1).sum()
    df['vwap_60'] = (df['close'] * df['volume']).rolling(60, min_periods=1).mean() / df['volume'].rolling(60, min_periods=1).sum()
    df['close_to_vwap_15'] = (df['close'] - df['vwap_15']) / df['vwap_15']
    df['close_to_vwap_60'] = (df['close'] - df['vwap_60']) / df['vwap_60']
    df['volume_change_1'] = df['volume'].pct_change(1)

    # Check created features
    features = ['close_change_pct', 'vwap', 'price_range', 'price_mom_3d', 'price_mom_5d',
                'close_to_vwap_15', 'close_to_vwap_60', 'volume_change_1', 'vwap_15', 'vwap_60']

    print(f"\n🔍 Feature diagnostics:")
    for feature in features:
        if feature in df.columns:
            nan_count = df[feature].isnull().sum()
            inf_count = np.isinf(df[feature]).sum()
            finite_count = np.isfinite(df[feature]).sum()
            min_val = df[feature].min()
            max_val = df[feature].max()

            print(f"  {feature}:")
            print(f"    NaN: {nan_count}, Inf: {inf_count}, Finite: {finite_count}")
            print(f"    Range: [{min_val:.6f}, {max_val:.6f}]")

            if nan_count > 0 or inf_count > 0:
                print(f"    ⚠️  PROBLEM DETECTED in {feature}")

    # Check correlations to see the problematic features
    print(f"\n📊 Correlation matrix for features:")
    feature_df = df[features].dropna()
    if len(feature_df) > 0:
        corr_matrix = feature_df.corr()
        print("Correlations with close_to_vwap_15:")
        for col in corr_matrix.columns:
            if 'close_to_vwap' in col:
                for other_col in corr_matrix.columns:
                    if other_col != col:
                        corr_val = corr_matrix.loc[col, other_col]
                        if abs(corr_val) > 0.7:
                            print(f"  {col} vs {other_col}: {corr_val:.3f} (HIGH)")

if __name__ == "__main__":
    main()