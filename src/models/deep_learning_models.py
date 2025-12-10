"""
Deep learning models for predictive maintenance using PyTorch.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class TimeSeriesDataset(Dataset):
    """
    Dataset for time-series sequences.
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray, sequence_length: int = 24):
        """
        Initialize dataset.
        
        Args:
            X: Feature array (n_samples, n_features)
            y: Target array (n_samples,)
            sequence_length: Length of sequences to create
        """
        self.sequence_length = sequence_length
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
        # Create sequences
        self.sequences = []
        self.labels = []
        
        for i in range(len(X) - sequence_length + 1):
            self.sequences.append(self.X[i:i+sequence_length])
            self.labels.append(self.y[i+sequence_length-1])
        
        self.sequences = torch.stack(self.sequences)
        self.labels = torch.stack(self.labels)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for tabular data.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_layers: list = [128, 64, 32],
        dropout: float = 0.3,
        activation: str = 'relu',
        num_classes: int = 1
    ):
        """
        Initialize MLP.
        
        Args:
            input_size: Number of input features
            hidden_layers: List of hidden layer sizes
            dropout: Dropout rate
            activation: Activation function ('relu', 'tanh', 'sigmoid')
            num_classes: Number of output classes (1 for binary, >1 for multi-class)
        """
        super(MLP, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Activation function
        if activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'tanh':
            act_fn = nn.Tanh()
        elif activation == 'sigmoid':
            act_fn = nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build layers
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(act_fn)
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        self.features = nn.Sequential(*layers)
        
        # Output layer
        if num_classes == 1:
            self.output = nn.Sequential(
                nn.Linear(prev_size, 1),
                nn.Sigmoid()
            )
        else:
            self.output = nn.Sequential(
                nn.Linear(prev_size, num_classes),
                nn.Softmax(dim=1)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, input_size)
            
        Returns:
            Output tensor
        """
        x = self.features(x)
        x = self.output(x)
        return x


class LSTM(nn.Module):
    """
    LSTM model for time-series prediction.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 1,
        bidirectional: bool = False
    ):
        """
        Initialize LSTM.
        
        Args:
            input_size: Number of input features per timestep
            hidden_size: Hidden state size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            num_classes: Number of output classes
            bidirectional: Whether to use bidirectional LSTM
        """
        super(LSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Output size accounting for bidirectional
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        self.dropout = nn.Dropout(dropout)
        
        if num_classes == 1:
            self.output = nn.Sequential(
                nn.Linear(lstm_output_size, 1),
                nn.Sigmoid()
            )
        else:
            self.output = nn.Sequential(
                nn.Linear(lstm_output_size, num_classes),
                nn.Softmax(dim=1)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the last timestep output
        last_output = lstm_out[:, -1, :]
        
        # Apply dropout
        last_output = self.dropout(last_output)
        
        # Output layer
        output = self.output(last_output)
        return output


class GRU(nn.Module):
    """
    GRU model for time-series prediction.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 1,
        bidirectional: bool = False
    ):
        """
        Initialize GRU.
        
        Args:
            input_size: Number of input features per timestep
            hidden_size: Hidden state size
            num_layers: Number of GRU layers
            dropout: Dropout rate
            num_classes: Number of output classes
            bidirectional: Whether to use bidirectional GRU
        """
        super(GRU, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Output size accounting for bidirectional
        gru_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        self.dropout = nn.Dropout(dropout)
        
        if num_classes == 1:
            self.output = nn.Sequential(
                nn.Linear(gru_output_size, 1),
                nn.Sigmoid()
            )
        else:
            self.output = nn.Sequential(
                nn.Linear(gru_output_size, num_classes),
                nn.Softmax(dim=1)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor
        """
        # GRU forward
        gru_out, h_n = self.gru(x)
        
        # Use the last timestep output
        last_output = gru_out[:, -1, :]
        
        # Apply dropout
        last_output = self.dropout(last_output)
        
        # Output layer
        output = self.output(last_output)
        return output


def get_device() -> torch.device:
    """
    Get the appropriate device (CPU or CUDA).
    
    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info("Using CUDA device")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU device")
    return device

