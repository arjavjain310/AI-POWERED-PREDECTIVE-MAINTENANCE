"""
Remaining Useful Life (RUL) estimation utilities.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from typing import Tuple, Optional, Dict
import logging

from src.models.deep_learning_models import TimeSeriesDataset, get_device
from src.models.training import train_model, validate_epoch
from src.models.evaluation import evaluate_regression, plot_rul_prediction
import torch.nn as nn

logger = logging.getLogger(__name__)


def prepare_rul_sequences(
    X: np.ndarray,
    y_rul: np.ndarray,
    sequence_length: int = 24,
    max_rul: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare sequences for RUL prediction.
    
    Args:
        X: Feature array (n_samples, n_features)
        y_rul: RUL values (n_samples,)
        sequence_length: Length of input sequences
        max_rul: Maximum RUL to cap at (None for no capping)
        
    Returns:
        Tuple of (X_sequences, y_rul_targets)
    """
    # Cap RUL if specified
    if max_rul is not None:
        y_rul = np.clip(y_rul, 0, max_rul)
    
    # Create sequences
    X_sequences = []
    y_sequences = []
    
    for i in range(len(X) - sequence_length + 1):
        X_sequences.append(X[i:i+sequence_length])
        # Use RUL at the end of the sequence
        y_sequences.append(y_rul[i+sequence_length-1])
    
    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)
    
    # Filter out NaN RUL values
    valid_mask = ~np.isnan(y_sequences)
    X_sequences = X_sequences[valid_mask]
    y_sequences = y_sequences[valid_mask]
    
    logger.info(f"Prepared {len(X_sequences)} RUL sequences")
    logger.info(f"RUL range: {y_sequences.min():.2f} to {y_sequences.max():.2f} hours")
    
    return X_sequences, y_sequences


def train_rul_model(
    model: torch.nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    sequence_length: int = 24,
    batch_size: int = 32,
    epochs: int = 50,
    learning_rate: float = 0.001,
    device: Optional[torch.device] = None,
    early_stopping_patience: int = 10,
    save_path: Optional[str] = None
) -> Dict[str, list]:
    """
    Train a model for RUL prediction.
    
    Args:
        model: PyTorch model (should output single value for regression)
        X_train: Training features
        y_train: Training RUL values
        X_val: Validation features
        y_val: Validation RUL values
        sequence_length: Sequence length
        batch_size: Batch size
        epochs: Number of epochs
        learning_rate: Learning rate
        device: Device to run on
        early_stopping_patience: Early stopping patience
        save_path: Path to save best model
        
    Returns:
        Training history
    """
    if device is None:
        device = get_device()
    
    # Prepare sequences
    X_train_seq, y_train_seq = prepare_rul_sequences(X_train, y_train, sequence_length)
    X_val_seq, y_val_seq = prepare_rul_sequences(X_val, y_val, sequence_length)
    
    # Create datasets
    train_dataset = TimeSeriesDataset(X_train_seq, y_train_seq, sequence_length=sequence_length)
    val_dataset = TimeSeriesDataset(X_val_seq, y_val_seq, sequence_length=sequence_length)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Modify model output for regression (remove sigmoid if present)
    # This assumes the model has an output layer we can modify
    # For simplicity, we'll use a custom training loop
    
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    from src.models.training import EarlyStopping
    early_stopping = EarlyStopping(patience=early_stopping_patience, mode='min')
    
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    best_val_loss = float('inf')
    
    logger.info("Training RUL model...")
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            
            # Handle output shape
            if outputs.dim() > 1:
                outputs = outputs.squeeze(1)
            
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_x)
                if outputs.dim() > 1:
                    outputs = outputs.squeeze(1)
                
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        if early_stopping(val_loss):
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_path:
                import os
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)
        
        if (epoch + 1) % 5 == 0:
            logger.info(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    logger.info("RUL model training complete!")
    return history


def predict_rul(
    model: torch.nn.Module,
    X: np.ndarray,
    sequence_length: int = 24,
    batch_size: int = 32,
    device: Optional[torch.device] = None
) -> np.ndarray:
    """
    Predict RUL for given features.
    
    Args:
        model: Trained PyTorch model
        X: Feature array
        sequence_length: Sequence length
        batch_size: Batch size
        device: Device to run on
        
    Returns:
        Predicted RUL values
    """
    if device is None:
        device = get_device()
    
    model = model.to(device)
    model.eval()
    
    # Create sequences
    X_sequences = []
    for i in range(len(X) - sequence_length + 1):
        X_sequences.append(X[i:i+sequence_length])
    X_sequences = np.array(X_sequences)
    
    # Create dataset and loader
    dataset = TimeSeriesDataset(X_sequences, np.zeros(len(X_sequences)), sequence_length=sequence_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    predictions = []
    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            if outputs.dim() > 1:
                outputs = outputs.squeeze(1)
            predictions.append(outputs.cpu().numpy())
    
    predictions = np.concatenate(predictions)
    
    # Pad predictions to match original length
    padded_predictions = np.full(len(X), np.nan)
    padded_predictions[sequence_length-1:] = predictions
    
    return padded_predictions

