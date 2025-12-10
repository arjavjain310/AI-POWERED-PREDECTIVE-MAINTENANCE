"""
Data preprocessing utilities for wind turbine SCADA data.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Preprocessor for wind turbine SCADA data.
    """
    
    def __init__(
        self,
        numerical_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        scaling_method: str = 'standard',  # 'standard', 'minmax', or None
        handle_outliers: bool = True,
        outlier_method: str = 'iqr'  # 'iqr' or 'zscore'
    ):
        """
        Initialize preprocessor.
        
        Args:
            numerical_cols: List of numerical column names
            categorical_cols: List of categorical column names
            scaling_method: Scaling method for numerical features
            handle_outliers: Whether to handle outliers
            outlier_method: Method for outlier detection
        """
        self.numerical_cols = numerical_cols or []
        self.categorical_cols = categorical_cols or []
        self.scaling_method = scaling_method
        self.handle_outliers = handle_outliers
        self.outlier_method = outlier_method
        
        self.scaler = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.outlier_bounds: Dict[str, Tuple[float, float]] = {}
    
    def _detect_numerical_cols(self, df: pd.DataFrame) -> List[str]:
        """Auto-detect numerical columns."""
        exclude = ['timestamp', 'turbine_id', 'failure_within_horizon',
                  'failed_component', 'time_to_failure_hours']
        numerical = df.select_dtypes(include=[np.number]).columns.tolist()
        return [col for col in numerical if col not in exclude]
    
    def _detect_categorical_cols(self, df: pd.DataFrame) -> List[str]:
        """Auto-detect categorical columns."""
        exclude = ['timestamp', 'turbine_id']
        categorical = df.select_dtypes(include=['object', 'category']).columns.tolist()
        return [col for col in categorical if col not in exclude]
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values."""
        df = df.copy()
        
        # Forward fill for time-series data
        for col in self.numerical_cols:
            if col in df.columns and df[col].isna().any():
                df[col] = df[col].ffill().bfill()
        
        # Fill remaining with median for numerical, mode for categorical
        for col in self.numerical_cols:
            if col in df.columns and df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
        
        for col in self.categorical_cols:
            if col in df.columns and df[col].isna().any():
                df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'unknown')
        
        return df
    
    def _detect_outliers_iqr(self, series: pd.Series, factor: float = 1.5) -> Tuple[float, float]:
        """Detect outliers using IQR method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        return lower_bound, upper_bound
    
    def _detect_outliers_zscore(self, series: pd.Series, threshold: float = 3.0) -> Tuple[float, float]:
        """Detect outliers using Z-score method."""
        mean = series.mean()
        std = series.std()
        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std
        return lower_bound, upper_bound
    
    def _handle_outliers(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Handle outliers in numerical columns."""
        df = df.copy()
        
        for col in self.numerical_cols:
            if col not in df.columns:
                continue
            
            if fit:
                if self.outlier_method == 'iqr':
                    lower, upper = self._detect_outliers_iqr(df[col])
                else:  # zscore
                    lower, upper = self._detect_outliers_zscore(df[col])
                
                self.outlier_bounds[col] = (lower, upper)
            else:
                lower, upper = self.outlier_bounds.get(col, (None, None))
            
            if lower is not None and upper is not None:
                # Clip outliers
                df[col] = df[col].clip(lower=lower, upper=upper)
        
        return df
    
    def _encode_categorical(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical variables."""
        df = df.copy()
        
        for col in self.categorical_cols:
            if col not in df.columns:
                continue
            
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders.get(col)
                if le is not None:
                    # Handle unseen categories
                    df[col] = df[col].astype(str)
                    known_classes = set(le.classes_)
                    df[col] = df[col].apply(lambda x: x if x in known_classes else le.classes_[0])
                    df[col] = le.transform(df[col])
        
        return df
    
    def _scale_numerical(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numerical features."""
        if self.scaling_method is None:
            return df
        
        df = df.copy()
        numerical_data = df[self.numerical_cols]
        
        if fit:
            if self.scaling_method == 'standard':
                self.scaler = StandardScaler()
            elif self.scaling_method == 'minmax':
                self.scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unknown scaling method: {self.scaling_method}")
            
            scaled_data = self.scaler.fit_transform(numerical_data)
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call fit() first.")
            scaled_data = self.scaler.transform(numerical_data)
        
        df[self.numerical_cols] = scaled_data
        return df
    
    def fit(self, df: pd.DataFrame) -> 'DataPreprocessor':
        """
        Fit preprocessor on training data.
        
        Args:
            df: Training DataFrame
            
        Returns:
            Self for method chaining
        """
        # Auto-detect columns if not provided
        if not self.numerical_cols:
            self.numerical_cols = self._detect_numerical_cols(df)
        if not self.categorical_cols:
            self.categorical_cols = self._detect_categorical_cols(df)
        
        logger.info(f"Fitting preprocessor:")
        logger.info(f"  Numerical cols: {len(self.numerical_cols)}")
        logger.info(f"  Categorical cols: {len(self.categorical_cols)}")
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Handle outliers
        if self.handle_outliers:
            df = self._handle_outliers(df, fit=True)
        
        # Encode categorical
        df = self._encode_categorical(df, fit=True)
        
        # Scale numerical
        if self.scaling_method:
            df = self._scale_numerical(df, fit=True)
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted preprocessor.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            Transformed DataFrame
        """
        df = df.copy()
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Handle outliers
        if self.handle_outliers:
            df = self._handle_outliers(df, fit=False)
        
        # Encode categorical
        df = self._encode_categorical(df, fit=False)
        
        # Scale numerical
        if self.scaling_method:
            df = self._scale_numerical(df, fit=False)
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)

