import torch
import torch.nn as nn
from typing import List, Optional

class ModelFactory:
    """Generates PyTorch models based on dynamic configuration."""
    
    @staticmethod
    def create_mlp(
        input_dim: int,
        hidden_layers: List[int],
        output_dim: int,
        activation: str = "ReLU",
        dropout_rate: float = 0.0,
        use_batchnorm: bool = False
    ) -> nn.Module:
        """
        Builds a Multi-Layer Perceptron.
        
        Args:
            input_dim: Number of input features.
            hidden_layers: List of neuron counts for hidden layers.
            output_dim: Number of output neurons.
            activation: Name of activation function ('ReLU', 'Tanh', 'Sigmoid').
            dropout_rate: Probability of dropout (0.0 to 1.0).
            use_batchnorm: Whether to add Batch Normalization layers.
        """
        layers = []
        current_dim = input_dim
        
        # Map string to class
        act_fn = getattr(nn, activation)() if hasattr(nn, activation) else nn.ReLU()
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(current_dim, hidden_dim))
            
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
                
            layers.append(act_fn)
            
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
                
            current_dim = hidden_dim
            
        # Output Layer
        layers.append(nn.Linear(current_dim, output_dim))
        
        # Note: No Softmax here if using CrossEntropyLoss (it includes LogSoftmax)
        # But for visualization or if user wants strict probabilities, might be needed.
        # Standard PyTorch practice for classification is raw logits + CrossEntropyLoss.
        
        return nn.Sequential(*layers)
