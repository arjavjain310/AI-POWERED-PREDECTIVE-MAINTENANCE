"""
Tests for data pipeline (loading, preprocessing, feature engineering).
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_loader import load_scada_data, split_data_by_time
from src.data.preprocessing import DataPreprocessor
from src.data.feature_engineering import FeatureEngineer


@pytest.fixture
def sample_data():
    """Create sample SCADA data for testing."""
    dates = pd.date_range('2023-01-01', periods=1000, freq='10min')
    data = {
        'timestamp': dates,
        'turbine_id': np.random.randint(1, 4, 1000),
        'wind_speed': np.random.uniform(5, 15, 1000),
        'power_output': np.random.uniform(100, 1000, 1000),
        'gearbox_oil_temperature': np.random.uniform(40, 80, 1000),
        'vibration_level_gearbox': np.random.uniform(0.5, 2.0, 1000),
        'failure_within_horizon': np.random.choice([0, 1], 1000, p=[0.9, 0.1]),
        'time_to_failure_hours': np.random.uniform(10, 200, 1000)
    }
    return pd.DataFrame(data)


def test_data_loading(sample_data, tmp_path):
    """Test data loading functionality."""
    # Save sample data
    data_path = tmp_path / "test_data.csv"
    sample_data.to_csv(data_path, index=False)
    
    # Load data
    loaded_data = load_scada_data(str(data_path))
    
    assert len(loaded_data) == len(sample_data)
    assert 'timestamp' in loaded_data.columns
    assert isinstance(loaded_data['timestamp'].iloc[0], pd.Timestamp)


def test_data_splitting(sample_data):
    """Test data splitting."""
    train_df, val_df, test_df = split_data_by_time(
        sample_data,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    assert len(train_df) + len(val_df) + len(test_df) == len(sample_data)
    assert train_df['timestamp'].max() <= val_df['timestamp'].min()
    assert val_df['timestamp'].max() <= test_df['timestamp'].min()


def test_preprocessing(sample_data):
    """Test data preprocessing."""
    preprocessor = DataPreprocessor(scaling_method='standard')
    
    # Select numerical columns
    numerical_cols = ['wind_speed', 'power_output', 'gearbox_oil_temperature', 'vibration_level_gearbox']
    X = sample_data[numerical_cols]
    
    # Fit and transform
    X_processed = preprocessor.fit_transform(X)
    
    assert X_processed.shape == X.shape
    assert not X_processed.isna().any().any()


def test_feature_engineering(sample_data):
    """Test feature engineering."""
    feature_engineer = FeatureEngineer(
        lag_periods=[1, 3],
        rolling_windows_hours=[1, 6],
        interval_minutes=10
    )
    
    df_with_features = feature_engineer.create_features(
        sample_data,
        create_lags=True,
        create_rolling=True,
        create_derived=True
    )
    
    # Check that new features were created
    original_cols = set(sample_data.columns)
    new_cols = set(df_with_features.columns)
    
    assert len(new_cols) > len(original_cols)
    assert 'power_curve_deviation' in df_with_features.columns or 'health_index' in df_with_features.columns


def test_feature_engineering_lags(sample_data):
    """Test lag feature creation."""
    feature_engineer = FeatureEngineer(lag_periods=[1, 2], interval_minutes=10)
    
    df_with_lags = feature_engineer.create_features(
        sample_data,
        create_lags=True,
        create_rolling=False,
        create_derived=False
    )
    
    # Check lag features exist
    lag_cols = [col for col in df_with_lags.columns if '_lag_' in col]
    assert len(lag_cols) > 0

