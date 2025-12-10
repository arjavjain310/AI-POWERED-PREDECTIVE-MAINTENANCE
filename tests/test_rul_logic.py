"""
Tests for RUL estimation and maintenance logic.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.maintenance.maintenance_rules import MaintenanceRules
from src.maintenance.schedule_optimizer import MaintenanceScheduler
from src.models.rul_estimation import prepare_rul_sequences


def test_maintenance_rules():
    """Test maintenance rule evaluation."""
    rules = MaintenanceRules(
        failure_probability_threshold=0.7,
        rul_threshold_hours=48.0
    )
    
    # High failure probability
    decision = rules.evaluate_maintenance_need(failure_probability=0.8, rul_hours=100)
    assert decision['maintenance_needed'] == True
    assert decision['urgency'] == 'high'
    
    # Low RUL
    decision = rules.evaluate_maintenance_need(failure_probability=0.3, rul_hours=24)
    assert decision['maintenance_needed'] == True
    assert decision['urgency'] == 'high'
    
    # No maintenance needed
    decision = rules.evaluate_maintenance_need(failure_probability=0.3, rul_hours=100)
    assert decision['maintenance_needed'] == False


def test_maintenance_rules_turbines():
    """Test maintenance evaluation for multiple turbines."""
    rules = MaintenanceRules(failure_probability_threshold=0.7, rul_threshold_hours=48.0)
    
    turbine_data = pd.DataFrame({
        'turbine_id': [1, 2, 3],
        'failure_probability': [0.8, 0.3, 0.5],
        'rul_hours': [100, 24, 60]
    })
    
    results = rules.evaluate_turbines(turbine_data)
    
    assert len(results) == 3
    assert 'maintenance_needed' in results.columns
    assert 'urgency' in results.columns


def test_schedule_optimizer():
    """Test maintenance schedule creation."""
    scheduler = MaintenanceScheduler(
        maintenance_duration_hours=8.0,
        preventive_cost=1000.0,
        corrective_cost=5000.0
    )
    
    turbine_data = pd.DataFrame({
        'turbine_id': [1, 2, 3],
        'maintenance_needed': [True, True, False],
        'urgency': ['high', 'medium', 'low'],
        'rul_hours': [10, 30, 100],
        'failure_probability': [0.9, 0.6, 0.3]
    })
    
    schedule = scheduler.create_schedule(turbine_data, planning_horizon_days=7)
    
    # Should schedule turbines 1 and 2
    assert len(schedule) == 2
    assert schedule['turbine_id'].isin([1, 2]).all()


def test_rul_sequence_preparation():
    """Test RUL sequence preparation."""
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y_rul = np.random.uniform(10, 200, n_samples)
    
    sequence_length = 24
    X_seq, y_seq = prepare_rul_sequences(X, y_rul, sequence_length)
    
    assert len(X_seq) == len(y_seq)
    assert X_seq.shape[1] == sequence_length
    assert X_seq.shape[2] == n_features
    assert len(X_seq) == n_samples - sequence_length + 1

