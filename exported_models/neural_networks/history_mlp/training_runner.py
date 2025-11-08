#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
training_runner.py

Location: /home/nexus/VQ_PMCnmpc/VQ_PMC/exported_models/neural_networks/

Contains the main logic for the training and validation loop.
Now also computes, **only for visualization**, an L1/MAE per target
component (x, y, yaw). The optimizer/loss for training remains MSE.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

TARGET_NAMES = ["x", "y", "yaw"]

def _init_component_logs():
    return {name: [] for name in TARGET_NAMES}

def train_model(model, train_data, val_data, epochs, learning_rate, batch_size):
    """
    Executes the training and validation loop for the given model.

    Returns:
        tuple: (train_losses, val_losses, train_l1_per_comp, val_l1_per_comp)
               where the last two are dicts with per-epoch MAE lists per component.
    """
    # DataLoaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    print(f"Starting training with {len(train_data)} train samples and {len(val_data)} validation samples.")

    # Optimizer / primary loss (unchanged)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Histories for main loss
    train_losses, val_losses = [], []

    # Histories for visualization-only per-component L1 (MAE)
    train_l1_per_comp = _init_component_logs()
    val_l1_per_comp = _init_component_logs()

    print(f"Training for {epochs} epochs...")

    for epoch in range(epochs):
        # ------------------- TRAIN -------------------
        model.train()
        train_loss = 0.0

        # Accumulators for MAE per component (sum of abs error, then / N)
        comp_sum_abs = torch.zeros(3)
        n_train = 0

        for x_batch, y_batch in train_loader:
            pred = model(x_batch)

            # Primary optimization loss (MSE) — unchanged
            loss = criterion(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # --- Secondary metric (visualization only): MAE per component ---
            # sum over batch, keep component dimension
            abs_err = (pred - y_batch).abs()  # shape [B, 3]
            comp_sum_abs += abs_err.sum(dim=0).detach()
            n_train += y_batch.shape[0]

        train_loss /= len(train_loader)
        if epoch > 0:
            train_losses.append(train_loss)

        # finalize epoch MAE per component for train
        if n_train > 0:
            comp_mae_train = (comp_sum_abs / n_train).tolist()
        else:
            comp_mae_train = [float('nan')]*3
        for i, name in enumerate(TARGET_NAMES):
            train_l1_per_comp[name].append(comp_mae_train[i])

        # ------------------- VAL -------------------
        model.eval()
        val_loss = 0.0
        comp_sum_abs_v = torch.zeros(3)
        n_val = 0

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                pred = model(x_batch)
                loss = criterion(pred, y_batch)
                val_loss += loss.item()

                abs_err = (pred - y_batch).abs()  # [B,3]
                comp_sum_abs_v += abs_err.sum(dim=0)
                n_val += y_batch.shape[0]

        val_loss /= len(val_loader)
        if epoch > 0:
            val_losses.append(val_loss)

        if n_val > 0:
            comp_mae_val = (comp_sum_abs_v / n_val).tolist()
        else:
            comp_mae_val = [float('nan')]*3
        for i, name in enumerate(TARGET_NAMES):
            val_l1_per_comp[name].append(comp_mae_val[i])

        # Console print: show both MSE (global) and per-component MAE
        mae_str_train = ", ".join([f"{k}: {train_l1_per_comp[k][-1]:.6f}" for k in TARGET_NAMES])
        mae_str_val = ", ".join([f"{k}: {val_l1_per_comp[k][-1]:.6f}" for k in TARGET_NAMES])
        print(f"Epoch {epoch+1}/{epochs} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f} | "
              f"Train MAE [{mae_str_train}] | Val MAE [{mae_str_val}]")

    return train_losses, val_losses, train_l1_per_comp, val_l1_per_comp