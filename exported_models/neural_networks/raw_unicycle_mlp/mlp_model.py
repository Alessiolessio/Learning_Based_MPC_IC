#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mlp_model.py

Defines the MLP architecture used to approximate system dynamics.
The number/size of hidden layers and dropout are configurable.
"""

import torch
import torch.nn as nn


class MLPDynamicsModel(nn.Module):
    """Configurable MLP: f(state_action_history) -> next_state."""

    def __init__(self, input_dim, output_dim, hidden_layers=[64, 64], p_dropout: float = 0.0):
        """
        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output features.
            hidden_layers (List[int]): Hidden layer sizes.
            p_dropout (float): Dropout probability (0.0 disables).
        """
        super(MLPDynamicsModel, self).__init__()

        layers = []

        # -- If there are no hidden layers, go directly to output --
        if not hidden_layers:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            # -- Input -> first hidden --
            layers.append(nn.Linear(input_dim, hidden_layers[0]))
            layers.append(nn.ReLU())
            if p_dropout > 0:
                layers.append(nn.Dropout(p=p_dropout))

            # -- Intermediate hidden layers (if any) --
            for i in range(len(hidden_layers) - 1):
                layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
                layers.append(nn.ReLU())
                if p_dropout > 0:
                    layers.append(nn.Dropout(p=p_dropout))

            # -- Last hidden -> output --
            layers.append(nn.Linear(hidden_layers[-1], output_dim))

        # -- Sequential container executes the layers in order --
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """Standard forward pass."""
        return self.model(x)
