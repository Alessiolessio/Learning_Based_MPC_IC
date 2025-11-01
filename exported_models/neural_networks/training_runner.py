#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
training_runner.py

Location: /home/nexus/VQ_PMCnmpc/VQ_PMC/exported_models/neural_networks/

Contains the main logic for the training and validation loop.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def train_model(model, train_data, val_data, epochs, learning_rate, batch_size):
    """
    Executes the training and validation loop for the given model.
    
    Args:
        model (nn.Module): The MLP model instance to be trained.
        train_data (Dataset): Training data (already split).
        val_data (Dataset): Validation data (already split).
        epochs (int): Number of epochs to train.
        learning_rate (float): Learning rate for the optimizer.
        batch_size (int): Batch size.
        
    Returns:
        tuple: (train_losses, val_losses) - Lists containing the loss history.
    """
    
    # DataLoaders: Tools that shuffle and split data into batches
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    print(f"Starting training with {len(train_data)} train samples and {len(val_data)} validation samples.")
    
    # Optimizer: 'Adam' is robust and modern.
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Loss Function: 'MSELoss' (Mean Squared Error) - Perfect for regression.
    criterion = nn.MSELoss()

    # Lists to store loss history for plotting
    train_losses = []
    val_losses = []

    print(f"Training for {epochs} epochs...")
    
    # --- Training Loop ---
    for epoch in range(epochs):
        
        # --- Training Phase ---
        model.train() # Puts the model in "train" mode
        train_loss = 0.0
        
        for x_batch, y_batch in train_loader:
            # 1. Make the prediction
            pred = model(x_batch)
            
            # 2. Calculate the error (Loss)
            loss = criterion(pred, y_batch)
            
            # 3. Zero old gradients
            optimizer.zero_grad()
            
            # 4. Calculate new gradients (Backpropagation)
            loss.backward()
            
            # 5. Update the network weights
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # --- Validation Phase ---
        model.eval() # Puts the model in "evaluation" mode
        val_loss = 0.0
        
        with torch.no_grad(): # Disables gradient calculation (saves computation)
            for x_batch, y_batch in val_loader:
                # 1. Make the prediction
                pred = model(x_batch)
                # 2. Calculate the error (Loss)
                loss = criterion(pred, y_batch)
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Print progress
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}")

    return train_losses, val_losses