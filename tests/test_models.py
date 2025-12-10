"""
Tests for machine learning models.
"""

import pytest
import numpy as np
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.baseline_models import BaselineModelTrainer
from src.models.deep_learning_models import MLP, LSTM, get_device


@pytest.fixture
def sample_data():
    """Create sample training data."""
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    
    return X, y


def test_baseline_model_creation():
    """Test that baseline models can be created."""
    trainer = BaselineModelTrainer(random_state=42)
    assert trainer is not None


def test_baseline_model_training(sample_data):
    """Test baseline model training."""
    X, y = sample_data
    trainer = BaselineModelTrainer(random_state=42)
    
    # Train Random Forest
    model = trainer.train_random_forest(X, y, n_estimators=10, max_depth=5)
    
    assert model is not None
    assert 'random_forest' in trainer.models


def test_baseline_model_prediction(sample_data):
    """Test baseline model prediction."""
    X, y = sample_data
    trainer = BaselineModelTrainer(random_state=42)
    
    trainer.train_random_forest(X, y, n_estimators=10, max_depth=5)
    
    predictions = trainer.predict('random_forest', X)
    probabilities = trainer.predict_proba('random_forest', X)
    
    assert len(predictions) == len(X)
    assert probabilities.shape[0] == len(X)
    assert probabilities.shape[1] == 2  # Binary classification


def test_mlp_creation():
    """Test MLP model creation."""
    mlp = MLP(input_size=10, hidden_layers=[32, 16], num_classes=1)
    
    assert mlp is not None
    
    # Test forward pass
    x = torch.randn(5, 10)
    output = mlp(x)
    
    assert output.shape[0] == 5
    assert output.shape[1] == 1 or output.dim() == 1


def test_lstm_creation():
    """Test LSTM model creation."""
    lstm = LSTM(input_size=10, hidden_size=32, num_layers=1, num_classes=1)
    
    assert lstm is not None
    
    # Test forward pass
    x = torch.randn(5, 24, 10)  # (batch, sequence, features)
    output = lstm(x)
    
    assert output.shape[0] == 5
    assert output.shape[1] == 1 or output.dim() == 1


def test_model_device():
    """Test device selection."""
    device = get_device()
    assert device is not None
    assert isinstance(device, torch.device)

