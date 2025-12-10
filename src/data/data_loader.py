"""
Data loading utilities for wind turbine SCADA data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


def load_scada_data(data_path: str) -> pd.DataFrame:
    """
    Load SCADA data from CSV file.
    
    Args:
        data_path: Path to the CSV file
        
    Returns:
        DataFrame with SCADA data
    """
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Convert timestamp to datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    logger.info(f"Loaded {len(df)} records")
    logger.info(f"Columns: {list(df.columns)}")
    
    return df


def split_data_by_time(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    timestamp_col: str = 'timestamp'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data by time (chronological split).
    
    Args:
        df: Input DataFrame
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        timestamp_col: Name of timestamp column
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.01:
        raise ValueError("Ratios must sum to 1.0")
    
    # Sort by timestamp
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    logger.info(f"Time-based split:")
    logger.info(f"  Train: {len(train_df)} samples ({train_df[timestamp_col].min()} to {train_df[timestamp_col].max()})")
    logger.info(f"  Val: {len(val_df)} samples ({val_df[timestamp_col].min()} to {val_df[timestamp_col].max()})")
    logger.info(f"  Test: {len(test_df)} samples ({test_df[timestamp_col].min()} to {test_df[timestamp_col].max()})")
    
    return train_df, val_df, test_df


def split_data_by_turbine(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    turbine_col: str = 'turbine_id'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data by turbine (each turbine goes entirely to one split).
    
    Args:
        df: Input DataFrame
        train_ratio: Proportion of turbines for training
        val_ratio: Proportion of turbines for validation
        test_ratio: Proportion of turbines for testing
        turbine_col: Name of turbine ID column
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.01:
        raise ValueError("Ratios must sum to 1.0")
    
    unique_turbines = df[turbine_col].unique()
    np.random.shuffle(unique_turbines)
    
    n_turbines = len(unique_turbines)
    train_end = int(n_turbines * train_ratio)
    val_end = int(n_turbines * (train_ratio + val_ratio))
    
    train_turbines = unique_turbines[:train_end]
    val_turbines = unique_turbines[train_end:val_end]
    test_turbines = unique_turbines[val_end:]
    
    train_df = df[df[turbine_col].isin(train_turbines)].copy()
    val_df = df[df[turbine_col].isin(val_turbines)].copy()
    test_df = df[df[turbine_col].isin(test_turbines)].copy()
    
    logger.info(f"Turbine-based split:")
    logger.info(f"  Train: {len(train_df)} samples from {len(train_turbines)} turbines")
    logger.info(f"  Val: {len(val_df)} samples from {len(val_turbines)} turbines")
    logger.info(f"  Test: {len(test_df)} samples from {len(test_turbines)} turbines")
    
    return train_df, val_df, test_df


def prepare_features_and_labels(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    target_col: str = 'failure_within_horizon',
    exclude_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate features and target labels.
    
    Args:
        df: Input DataFrame
        feature_cols: List of feature column names. If None, auto-detect
        target_col: Name of target column
        exclude_cols: Columns to exclude from features
        
    Returns:
        Tuple of (features_df, target_series)
    """
    if exclude_cols is None:
        exclude_cols = ['timestamp', 'turbine_id', 'failure_within_horizon',
                        'failed_component', 'time_to_failure_hours', 'status_code']
    
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Remove any feature cols that don't exist
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy() if target_col in df.columns else None
    
    logger.info(f"Prepared {len(feature_cols)} features")
    if y is not None:
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
    
    return X, y

