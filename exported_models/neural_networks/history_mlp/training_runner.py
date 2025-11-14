#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
training_runner.py

Runs the training/validation loop with Adam + MSE,
and logs per-component MAE (x, y, yaw) for visualization only.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Names for per-component MAE logging (kept as in the original)
TARGET_NAMES = ["x", "y", "yaw"]


def _init_component_logs():
    """Create a dict of component-name -> empty list for MAE tracking."""
    return {name: [] for name in TARGET_NAMES}


def train_model(model, train_data, val_data, epochs, learning_rate, batch_size, device):
    """
    Executes the training/validation loop.

    Returns:
        (train_losses, val_losses, train_l1_per_comp, val_l1_per_comp)
    """
    # -- DataLoaders (shuffle only train) --
    pin_mem = device.type == "cuda"
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=pin_mem)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, pin_memory=pin_mem)

    print(f"Starting training with {len(train_data)} train samples and {len(val_data)} validation samples.")

    # -- Optimizer and main loss (unchanged) --
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # -- Global loss histories (MSE) --
    train_losses, val_losses = [], []

    # -- Per-component MAE histories (visualization only) --
    train_l1_per_comp = _init_component_logs()
    val_l1_per_comp = _init_component_logs()

    print(f"Training for {epochs} epochs...")

    for epoch in range(epochs):
        # ------------------- TRAIN -------------------
        model.train()
        train_loss = 0.0

        # Accumulators for MAE per component (sum over batch, then / N)
        comp_sum_abs = torch.zeros(3, device=device)
        n_train = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device, non_blocking=pin_mem)
            y_batch = y_batch.to(device, non_blocking=pin_mem)

            # Forward pass
            pred = model(x_batch)

            # Primary training objective (MSE)
            loss = criterion(pred, y_batch)

            # Backpropagation step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track MSE for this batch
            train_loss += loss.item()

            # Secondary metric (visualization): MAE per component
            abs_err = (pred - y_batch).abs()  # [B, 3]
            comp_sum_abs += abs_err.sum(dim=0).detach()
            n_train += y_batch.shape[0]

        # Average train MSE across batches (skip epoch 1 alignment)
        train_loss /= len(train_loader)
        if epoch > 0:
            train_losses.append(train_loss)

        # Finalize MAE per component for the epoch
        comp_mae_train = (comp_sum_abs / max(1, n_train)).tolist()
        for i, name in enumerate(TARGET_NAMES):
            train_l1_per_comp[name].append(comp_mae_train[i])

        # ------------------- VAL -------------------
        model.eval()
        val_loss = 0.0
        comp_sum_abs_v = torch.zeros(3, device=device)
        n_val = 0

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device, non_blocking=pin_mem)
                y_batch = y_batch.to(device, non_blocking=pin_mem)

                pred = model(x_batch)
                loss = criterion(pred, y_batch)
                val_loss += loss.item()

                abs_err = (pred - y_batch).abs()  # [B, 3]
                comp_sum_abs_v += abs_err.sum(dim=0)
                n_val += y_batch.shape[0]

        # Average val MSE across batches (skip epoch 1 alignment)
        val_loss /= len(val_loader)
        if epoch > 0:
            val_losses.append(val_loss)

        # Finalize MAE per component for validation
        comp_mae_val = (comp_sum_abs_v / max(1, n_val)).tolist()
        for i, name in enumerate(TARGET_NAMES):
            val_l1_per_comp[name].append(comp_mae_val[i])

        # Console print summarizing MSE and per-component MAE
        mae_str_train = ", ".join([f"{k}: {train_l1_per_comp[k][-1]:.6f}" for k in TARGET_NAMES])
        mae_str_val = ", ".join([f"{k}: {val_l1_per_comp[k][-1]:.6f}" for k in TARGET_NAMES])
        print(
            f"Epoch {epoch+1}/{epochs} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f} | "
            f"Train MAE [{mae_str_train}] | Val MAE [{mae_str_val}]"
        )

    return train_losses, val_losses, train_l1_per_comp, val_l1_per_comp
