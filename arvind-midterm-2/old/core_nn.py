# =============================================================================
# Core Neural Network Model - Election Prediction Project
# =============================================================================
"""
Defines the core PyTorch neural network architecture (DynamicMLP) used for
election outcome prediction. This class allows for flexible creation of
multi-layer perceptrons.
"""

import torch
import torch.nn as nn
from typing import List, Type

# =============================================================================
# Dynamic MLP Class (Template for PyTorch MLP)
# =============================================================================
class DynamicMLP(nn.Module):
    """
    A flexible Multi-Layer Perceptron (MLP) PyTorch module.

    Constructs an MLP with specified hidden layers, sizes, activation, and dropout.
    The final layer applies Softmax for probability distributions.

    Attributes:
        network (nn.Sequential): Main layers (Linear, Activation, Dropout) pre-softmax.
        softmax (nn.Softmax): Final Softmax activation layer.
    """
    def __init__(self,
                 input_dim: int,
                 hidden_layers: List[int],
                 activation_fn: Type[nn.Module],
                 dropout_rate: float):
        """
        Initializes the DynamicMLP module.

        Args:
            input_dim (int): Number of input features.
            hidden_layers (List[int]): List of neuron counts for hidden layers.
            activation_fn (Type[nn.Module]): PyTorch activation function class.
            dropout_rate (float): Dropout probability (0.0 to 1.0).
        """
        super().__init__()

        layers = []
        current_dim = input_dim

        # Hidden Layers
        if hidden_layers:
            layers.append(nn.Linear(current_dim, hidden_layers[0]))
            layers.append(activation_fn())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_layers[0]
            for units in hidden_layers[1:]:
                layers.append(nn.Linear(current_dim, units))
                layers.append(activation_fn())
                layers.append(nn.Dropout(dropout_rate))
                current_dim = units

        # Final Linear Layer (Logits) -> 4 outputs assumed
        layers.append(nn.Linear(current_dim, 4))

        # Main network sequence (excluding final Softmax)
        self.network = nn.Sequential(*layers)

        # Final Softmax activation (applied across class dimension)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor [batch_size, input_dim].

        Returns:
            torch.Tensor: Output probabilities [batch_size, 4].
        """
        logits = self.network(x)
        probabilities = self.softmax(logits)
        return probabilities