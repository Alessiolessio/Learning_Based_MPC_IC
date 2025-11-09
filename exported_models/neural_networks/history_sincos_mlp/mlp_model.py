#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mlp_model.py

Defines a parameterizable MLP for dynamics modeling.
"""

import torch
import torch.nn as nn

class MLPDynamicsModel(nn.Module):
    """
    Simple configurable MLP for f(history_state_action) -> next_state (or components).
    """

    def __init__(self, input_dim, output_dim, hidden_layers=[64, 64], p_dropout: float = 0.0):
        """
        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output features.
            hidden_layers (List[int]): Sizes of hidden layers, e.g., [128, 128, 128].
            p_dropout (float): Dropout probability. 0.0 disables dropout.
        """
        super(MLPDynamicsModel, self).__init__()

        layers = []

        # If no hidden layers are provided, connect input directly to output
        if not hidden_layers:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            # First hidden layer
            layers.append(nn.Linear(input_dim, hidden_layers[0]))
            layers.append(nn.ReLU())
            if p_dropout > 0:
                layers.append(nn.Dropout(p=p_dropout))

            # Optional intermediate hidden layers
            for i in range(len(hidden_layers) - 1):
                layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
                layers.append(nn.ReLU())
                if p_dropout > 0:
                    layers.append(nn.Dropout(p=p_dropout))

            # Output layer
            layers.append(nn.Linear(hidden_layers[-1], output_dim))

        # Sequence container executes layers in order
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """Standard forward pass."""
        return self.model(x)
