"""
Baseline machine learning models for failure prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib
import logging

# XGBoost is optional
XGBOOST_AVAILABLE = False
xgb = None
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except (ImportError, Exception) as e:
    # XGBoost may fail to load due to missing OpenMP or other issues
    pass

logger = logging.getLogger(__name__)


class BaselineModelTrainer:
    """
    Trainer for baseline ML models.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize trainer.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models: Dict[str, Any] = {}
        self.is_fitted = False
    
    def train_logistic_regression(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        **kwargs
    ) -> LogisticRegression:
        """
        Train logistic regression model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            **kwargs: Additional arguments for LogisticRegression
            
        Returns:
            Trained model
        """
        logger.info("Training Logistic Regression...")
        model = LogisticRegression(random_state=self.random_state, max_iter=1000, **kwargs)
        model.fit(X_train, y_train)
        self.models['logistic_regression'] = model
        self.is_fitted = True
        logger.info("Logistic Regression training complete")
        return model
    
    def train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_estimators: int = 100,
        max_depth: int = 20,
        **kwargs
    ) -> RandomForestClassifier:
        """
        Train random forest model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            **kwargs: Additional arguments for RandomForestClassifier
            
        Returns:
            Trained model
        """
        logger.info("Training Random Forest...")
        # Remove random_state from kwargs if present to avoid conflict
        kwargs.pop('random_state', None)
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=self.random_state,
            n_jobs=-1,
            **kwargs
        )
        model.fit(X_train, y_train)
        self.models['random_forest'] = model
        self.is_fitted = True
        logger.info("Random Forest training complete")
        return model
    
    def train_gradient_boosting(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        **kwargs
    ) -> GradientBoostingClassifier:
        """
        Train gradient boosting model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            n_estimators: Number of boosting stages
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            **kwargs: Additional arguments for GradientBoostingClassifier
            
        Returns:
            Trained model
        """
        logger.info("Training Gradient Boosting...")
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=self.random_state,
            **kwargs
        )
        model.fit(X_train, y_train)
        self.models['gradient_boosting'] = model
        self.is_fitted = True
        logger.info("Gradient Boosting training complete")
        return model
    
    def train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        **kwargs
    ) -> Any:
        """
        Train XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            **kwargs: Additional arguments for XGBClassifier
            
        Returns:
            Trained model
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not available. Please install it or use an alternative model.")
        
        logger.info("Training XGBoost...")
        # Remove random_state from kwargs if present to avoid conflict
        kwargs.pop('random_state', None)
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=self.random_state,
            n_jobs=-1,
            **kwargs
        )
        model.fit(X_train, y_train)
        self.models['xgboost'] = model
        self.is_fitted = True
        logger.info("XGBoost training complete")
        return model
    
    def predict(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with a trained model.
        
        Args:
            model_name: Name of the model
            X: Features
            
        Returns:
            Predictions
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        return self.models[model_name].predict(X)
    
    def predict_proba(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            model_name: Name of the model
            X: Features
            
        Returns:
            Prediction probabilities
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        return self.models[model_name].predict_proba(X)
    
    def get_feature_importance(self, model_name: str) -> Optional[Dict[str, float]]:
        """
        Get feature importance for tree-based models.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary of feature importances or None
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            return dict(zip(range(len(model.feature_importances_)), model.feature_importances_))
        
        return None
    
    def save_model(self, model_name: str, filepath: str):
        """
        Save a trained model to disk.
        
        Args:
            model_name: Name of the model
            filepath: Path to save the model
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        joblib.dump(self.models[model_name], filepath)
        logger.info(f"Model {model_name} saved to {filepath}")
    
    def load_model(self, model_name: str, filepath: str):
        """
        Load a model from disk.
        
        Args:
            model_name: Name for the model
            filepath: Path to load the model from
        """
        self.models[model_name] = joblib.load(filepath)
        self.is_fitted = True
        logger.info(f"Model {model_name} loaded from {filepath}")

