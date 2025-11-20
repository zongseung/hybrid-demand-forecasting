"""
Data splitting utilities for time series
"""
import pandas as pd
from typing import Tuple


def split_train_val_test(
    df: pd.DataFrame,
    val_months: int = 1,
    test_months: int = 1
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split time series data into train, validation, and test sets
    
    Args:
        df: DataFrame with timestamp column
        val_months: Number of months for validation (default: 1)
        test_months: Number of months for test (default: 1)
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # Calculate split points
    val_hours = val_months * 30 * 24  # Approximate
    test_hours = test_months * 30 * 24
    
    total_hours = len(df)
    test_start_idx = total_hours - test_hours
    val_start_idx = test_start_idx - val_hours
    
    # Split
    train_df = df.iloc[:val_start_idx].copy()
    val_df = df.iloc[val_start_idx:test_start_idx].copy()
    test_df = df.iloc[test_start_idx:].copy()
    
    print(f"Data Split:")
    print(f"  Train: {len(train_df)} hours ({len(train_df)/24:.1f} days) - "
          f"{train_df['timestamp'].iloc[0]} to {train_df['timestamp'].iloc[-1]}")
    print(f"  Val:   {len(val_df)} hours ({len(val_df)/24:.1f} days) - "
          f"{val_df['timestamp'].iloc[0]} to {val_df['timestamp'].iloc[-1]}")
    print(f"  Test:  {len(test_df)} hours ({len(test_df)/24:.1f} days) - "
          f"{test_df['timestamp'].iloc[0]} to {test_df['timestamp'].iloc[-1]}")
    
    return train_df, val_df, test_df


def split_train_val(
    df: pd.DataFrame,
    val_months: int = 1
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and validation sets only
    
    Args:
        df: DataFrame with timestamp column
        val_months: Number of months for validation
        
    Returns:
        Tuple of (train_df, val_df)
    """
    val_hours = val_months * 30 * 24
    val_start_idx = len(df) - val_hours
    
    train_df = df.iloc[:val_start_idx].copy()
    val_df = df.iloc[val_start_idx:].copy()
    
    print(f"Data Split:")
    print(f"  Train: {len(train_df)} hours ({len(train_df)/24:.1f} days)")
    print(f"  Val:   {len(val_df)} hours ({len(val_df)/24:.1f} days)")
    
    return train_df, val_df



