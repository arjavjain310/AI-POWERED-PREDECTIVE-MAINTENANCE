"""
Training utilities for deep learning models.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable, Tuple
from pathlib import Path
import logging

from src.models.deep_learning_models import get_device

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping utility to stop training when validation loss stops improving.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation score
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
        elif self._is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_better(self, current: float, best: float) -> bool:
        """Check if current score is better than best."""
        if self.mode == 'min':
            return current < best - self.min_delta
        else:
            return current > best + self.min_delta


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """
    Train model for one epoch.
    
    Args:
        model: PyTorch model
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run on
        
    Returns:
        Tuple of (average_loss, average_accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch_x)
        
        # Reshape if needed
        if outputs.dim() > 1 and outputs.size(1) == 1:
            outputs = outputs.squeeze(1)
        
        loss = criterion(outputs, batch_y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        predictions = (outputs > 0.5).float() if outputs.dim() == 1 else outputs.argmax(dim=1)
        correct += (predictions == batch_y).sum().item()
        total += batch_y.size(0)
    
    avg_loss = total_loss / len(dataloader)
    avg_acc = correct / total if total > 0 else 0.0
    
    return avg_loss, avg_acc


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Validate model for one epoch.
    
    Args:
        model: PyTorch model
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to run on
        
    Returns:
        Tuple of (average_loss, average_accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_x)
            
            # Reshape if needed
            if outputs.dim() > 1 and outputs.size(1) == 1:
                outputs = outputs.squeeze(1)
            
            loss = criterion(outputs, batch_y)
            
            total_loss += loss.item()
            predictions = (outputs > 0.5).float() if outputs.dim() == 1 else outputs.argmax(dim=1)
            correct += (predictions == batch_y).sum().item()
            total += batch_y.size(0)
    
    avg_loss = total_loss / len(dataloader)
    avg_acc = correct / total if total > 0 else 0.0
    
    return avg_loss, avg_acc


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    learning_rate: float = 0.001,
    device: Optional[torch.device] = None,
    early_stopping_patience: int = 10,
    save_path: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, list]:
    """
    Train a deep learning model.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of epochs
        learning_rate: Learning rate
        device: Device to run on
        early_stopping_patience: Patience for early stopping
        save_path: Path to save best model
        verbose: Whether to print progress
        
    Returns:
        Dictionary with training history
    """
    if device is None:
        device = get_device()
    
    model = model.to(device)
    
    criterion = nn.BCELoss() if model.output[0].out_features == 1 else nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    early_stopping = EarlyStopping(patience=early_stopping_patience, mode='min')
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    
    logger.info(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Early stopping
        if early_stopping(val_loss):
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), save_path)
                if verbose:
                    logger.info(f"Saved best model at epoch {epoch + 1} (val_loss: {val_loss:.4f})")
        
        if verbose and (epoch + 1) % 5 == 0:
            logger.info(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )
    
    logger.info("Training complete!")
    return history

