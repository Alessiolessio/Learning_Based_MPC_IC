#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mlp_model.py

Location: /home/nexus/VQ_PMCnmpc/VQ_PMC/exported_models/neural_networks/

Defines the MLP (Multilayer Perceptron) model architecture.
The architecture (number of neurons and layers) is parameterizable.
"""

import torch
import torch.nn as nn

class MLPDynamicsModel(nn.Module):
    """
    Defines the architecture of our Neural Network (MLP).
    'nn.Module' is the PyTorch base class for all models.
    """
    
    def __init__(self, input_dim, output_dim, hidden_layers=[64, 64], p_dropout: float = 0.0):
        """
        Class constructor. Defines the network layers.
        
        - input_dim: Input dimension (e.g., 5, or 15 if history_length=3)
        - output_dim: Output dimension (e.g., 3)
        - hidden_layers: A list of integers defining the size of each
                         hidden layer. Ex: [64, 64] for two
                         hidden layers of 64 neurons each.
        - p_dropout: Probabilidade de dropout (ex: 0.1). 0.0 desativa.
        """
        super(MLPDynamicsModel, self).__init__()
        
        layers = []
        
        # Input layer
        if not hidden_layers:
            # Simple case: direct from input to output
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            # Input layer to the first hidden layer
            layers.append(nn.Linear(input_dim, hidden_layers[0]))
            layers.append(nn.ReLU())
            # Adicionada camada de Dropout
            if p_dropout > 0:
                layers.append(nn.Dropout(p=p_dropout))
            
            # Intermediate hidden layers
            for i in range(len(hidden_layers) - 1):
                layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
                layers.append(nn.ReLU())
                # Adicionada camada de Dropout
                if p_dropout > 0:
                    layers.append(nn.Dropout(p=p_dropout))
            
            # Last hidden layer to the output layer
            layers.append(nn.Linear(hidden_layers[-1], output_dim))

        # 'nn.Sequential' is a container that executes layers in order.
        self.model = nn.Sequential(*layers) # The '*' unpacks the list

    def forward(self, x):
        """
        Defines the "forward pass".
        It is called automatically when you do model(input).
        """
        return self.model(x)