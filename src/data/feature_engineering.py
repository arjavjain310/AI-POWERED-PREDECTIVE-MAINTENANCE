"""
Feature engineering for wind turbine SCADA data.

Creates lag features, rolling statistics, and derived features.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering for time-series SCADA data.
    """
    
    def __init__(
        self,
        lag_periods: List[int] = [1, 3, 6],
        rolling_windows_hours: List[float] = [1, 6, 24],
        rolling_stats: List[str] = ['mean', 'std', 'min', 'max'],
        interval_minutes: int = 10
    ):
        """
        Initialize feature engineer.
        
        Args:
            lag_periods: List of lag periods (number of timestamps)
            rolling_windows_hours: List of rolling window sizes in hours
            rolling_stats: List of statistics to compute ('mean', 'std', 'min', 'max', 'median')
            interval_minutes: Sampling interval in minutes
        """
        self.lag_periods = lag_periods
        self.rolling_windows_hours = rolling_windows_hours
        self.rolling_stats = rolling_stats
        self.interval_minutes = interval_minutes
    
    def _create_lag_features(
        self,
        df: pd.DataFrame,
        cols: List[str],
        group_by: Optional[str] = 'turbine_id'
    ) -> pd.DataFrame:
        """
        Create lag features for specified columns.
        
        Args:
            df: Input DataFrame
            cols: Columns to create lags for
            group_by: Column to group by (e.g., turbine_id)
            
        Returns:
            DataFrame with lag features added
        """
        df = df.copy()
        
        if group_by and group_by in df.columns:
            grouped = df.groupby(group_by)
        else:
            grouped = [('all', df)]
        
        lag_cols = []
        for col in cols:
            if col not in df.columns:
                continue
            
            for lag in self.lag_periods:
                lag_col_name = f"{col}_lag_{lag}"
                lag_cols.append(lag_col_name)
                
                if group_by and group_by in df.columns:
                    df[lag_col_name] = grouped[col].shift(lag)
                else:
                    df[lag_col_name] = df[col].shift(lag)
        
        logger.info(f"Created {len(lag_cols)} lag features")
        return df
    
    def _create_rolling_features(
        self,
        df: pd.DataFrame,
        cols: List[str],
        group_by: Optional[str] = 'turbine_id'
    ) -> pd.DataFrame:
        """
        Create rolling statistics for specified columns.
        
        Args:
            df: Input DataFrame
            cols: Columns to create rolling stats for
            group_by: Column to group by
            
        Returns:
            DataFrame with rolling features added
        """
        df = df.copy()
        
        # Convert hours to number of periods
        rolling_windows = [int(w * 60 / self.interval_minutes) for w in self.rolling_windows_hours]
        
        rolling_cols = []
        for col in cols:
            if col not in df.columns:
                continue
            
            for window in rolling_windows:
                if window < 2:
                    continue
                
                for stat in self.rolling_stats:
                    rolling_col_name = f"{col}_rolling_{window}_{stat}"
                    rolling_cols.append(rolling_col_name)
                    
                    if group_by and group_by in df.columns:
                        grouped = df.groupby(group_by)[col]
                    else:
                        grouped = df[col]
                    
                    if stat == 'mean':
                        df[rolling_col_name] = grouped.rolling(window=window, min_periods=1).mean().reset_index(0, drop=True)
                    elif stat == 'std':
                        df[rolling_col_name] = grouped.rolling(window=window, min_periods=1).std().reset_index(0, drop=True)
                    elif stat == 'min':
                        df[rolling_col_name] = grouped.rolling(window=window, min_periods=1).min().reset_index(0, drop=True)
                    elif stat == 'max':
                        df[rolling_col_name] = grouped.rolling(window=window, min_periods=1).max().reset_index(0, drop=True)
                    elif stat == 'median':
                        df[rolling_col_name] = grouped.rolling(window=window, min_periods=1).median().reset_index(0, drop=True)
        
        logger.info(f"Created {len(rolling_cols)} rolling features")
        return df
    
    def _create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features (e.g., power curve deviation, health index).
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with derived features added
        """
        df = df.copy()
        
        # Power curve deviation (expected vs actual power)
        if 'wind_speed' in df.columns and 'power_output' in df.columns:
            # Simplified expected power curve: P ≈ 0.3 * v^3 for v in [3, 12]
            def expected_power(v):
                if v < 3:
                    return 0
                elif v < 12:
                    return 0.3 * v ** 3
                elif v < 25:
                    return 0.3 * 12 ** 3
                else:
                    return 0
            
            df['expected_power'] = df['wind_speed'].apply(expected_power)
            df['power_curve_deviation'] = df['power_output'] - df['expected_power']
            df['power_curve_deviation_pct'] = (df['power_curve_deviation'] / (df['expected_power'] + 1e-6)) * 100
        
        # Temperature differentials
        if 'gearbox_oil_temperature' in df.columns and 'ambient_temperature' in df.columns:
            df['gearbox_temp_delta'] = df['gearbox_oil_temperature'] - df['ambient_temperature']
        
        if 'generator_temperature' in df.columns and 'ambient_temperature' in df.columns:
            df['generator_temp_delta'] = df['generator_temperature'] - df['ambient_temperature']
        
        # Vibration ratio
        if 'vibration_level_gearbox' in df.columns and 'vibration_level_generator' in df.columns:
            df['vibration_ratio'] = df['vibration_level_gearbox'] / (df['vibration_level_generator'] + 1e-6)
        
        # Efficiency metrics
        if 'power_output' in df.columns and 'wind_speed' in df.columns:
            df['power_per_wind'] = df['power_output'] / (df['wind_speed'] + 1e-6)
        
        # Simple health index (composite score)
        health_components = []
        if 'gearbox_oil_temperature' in df.columns:
            # Normalize temperature (assuming normal range 40-80°C)
            temp_norm = (df['gearbox_oil_temperature'] - 40) / 40
            health_components.append(1 - np.clip(temp_norm, 0, 1))
        
        if 'vibration_level_gearbox' in df.columns:
            # Normalize vibration (assuming normal < 2.0)
            vib_norm = df['vibration_level_gearbox'] / 2.0
            health_components.append(1 - np.clip(vib_norm, 0, 1))
        
        if 'power_curve_deviation_pct' in df.columns:
            # Normalize power deviation (assuming normal ±10%)
            dev_norm = np.abs(df['power_curve_deviation_pct']) / 10.0
            health_components.append(1 - np.clip(dev_norm, 0, 1))
        
        if health_components:
            df['health_index'] = np.mean(health_components, axis=0)
        
        logger.info("Created derived features")
        return df
    
    def create_features(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        group_by: Optional[str] = 'turbine_id',
        create_lags: bool = True,
        create_rolling: bool = True,
        create_derived: bool = True
    ) -> pd.DataFrame:
        """
        Create all engineered features.
        
        Args:
            df: Input DataFrame
            feature_cols: Columns to create features for. If None, auto-detect numerical cols
            group_by: Column to group by for time-series operations
            create_lags: Whether to create lag features
            create_rolling: Whether to create rolling features
            create_derived: Whether to create derived features
            
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        
        # Sort by timestamp and group if needed
        if 'timestamp' in df.columns:
            if group_by and group_by in df.columns:
                df = df.sort_values([group_by, 'timestamp']).reset_index(drop=True)
            else:
                df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Auto-detect feature columns if not provided
        if feature_cols is None:
            exclude = ['timestamp', 'turbine_id', 'failure_within_horizon',
                      'failed_component', 'time_to_failure_hours', 'status_code']
            feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns
                          if col not in exclude]
        
        # Key columns for feature engineering
        key_cols = ['wind_speed', 'power_output', 'gearbox_oil_temperature',
                   'generator_temperature', 'vibration_level_gearbox',
                   'vibration_level_generator', 'rotor_speed', 'generator_speed']
        key_cols = [col for col in key_cols if col in feature_cols]
        
        logger.info(f"Creating features for {len(key_cols)} key columns")
        
        # Create lag features
        if create_lags:
            df = self._create_lag_features(df, key_cols, group_by=group_by)
        
        # Create rolling features
        if create_rolling:
            df = self._create_rolling_features(df, key_cols, group_by=group_by)
        
        # Create derived features
        if create_derived:
            df = self._create_derived_features(df)
        
        # Drop rows with NaN from lag features
        if create_lags:
            max_lag = max(self.lag_periods) if self.lag_periods else 0
            if max_lag > 0:
                initial_n = len(df)
                df = df.iloc[max_lag:].reset_index(drop=True)
                logger.info(f"Dropped {initial_n - len(df)} rows due to lag features")
        
        return df

