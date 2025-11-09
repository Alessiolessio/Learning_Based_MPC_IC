#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
training_runner.py

Runs the training/validation loop on CPU/GPU (device-aware),
optimizing MSE while logging per-component MAE (x, y, yaw) for visualization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Names for per-component MAE logging (visual inspection only)
TARGET_NAMES = ["x", "y", "yaw"]


def _init_component_logs():
    """Create a dict {component_name: []} to accumulate per-epoch MAE."""
    return {name: [] for name in TARGET_NAMES}


def train_model(model, train_data, val_data, epochs, learning_rate, batch_size, device="cuda"):
    """
    Execute the training/validation loop.

    Returns:
        (train_losses, val_losses, train_l1_per_comp, val_l1_per_comp)
    """
    # -- Move model to the desired device (GPU/CPU) --
    model = model.to(device)

    # -- DataLoaders (shuffle only the training set) --
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    print(
        f"Starting training with {len(train_data)} train samples and "
        f"{len(val_data)} validation samples on {device}."
    )

    # -- Optimizer and primary objective (MSE) --
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # -- Global loss histories --
    train_losses, val_losses = [], []

    # -- Visualization-only MAE histories per component --
    train_l1_per_comp = _init_component_logs()
    val_l1_per_comp = _init_component_logs()

    print(f"Training for {epochs} epochs...")

    for epoch in range(epochs):
        # ------------------- TRAIN -------------------
        model.train()
        train_loss = 0.0

        # Accumulates absolute error sums per component on the device
        comp_sum_abs = torch.zeros(3, device=device)
        n_train = 0

        for x_batch, y_batch in train_loader:
            # Move data to device
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # Forward pass
            pred = model(x_batch)

            # MSE optimization step
            loss = criterion(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track training loss (sum over batches)
            train_loss += loss.item()

            # Accumulate absolute error for per-component MAE
            abs_err = (pred - y_batch).abs()  # shape [B, 3]
            comp_sum_abs += abs_err.sum(dim=0).detach()
            n_train += y_batch.shape[0]

        # Average training MSE over all batches
        train_loss /= len(train_loader)
        if epoch > 0:  # keep alignment with plotting labels
            train_losses.append(train_loss)

        # Finalize per-component MAE for training
        comp_mae_train = (comp_sum_abs / max(1, n_train)).detach().cpu().tolist()
        for i, name in enumerate(TARGET_NAMES):
            train_l1_per_comp[name].append(comp_mae_train[i])

        # ------------------- VALIDATION -------------------
        model.eval()
        val_loss = 0.0
        comp_sum_abs_v = torch.zeros(3, device=device)
        n_val = 0

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                # Move data to device
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                # Forward pass and loss
                pred = model(x_batch)
                loss = criterion(pred, y_batch)
                val_loss += loss.item()

                # Accumulate absolute error for MAE
                abs_err = (pred - y_batch).abs()  # [B, 3]
                comp_sum_abs_v += abs_err.sum(dim=0).detach()
                n_val += y_batch.shape[0]

        # Average validation MSE
        val_loss /= len(val_loader)
        if epoch > 0:
            val_losses.append(val_loss)

        # Finalize per-component MAE for validation
        comp_mae_val = (comp_sum_abs_v / max(1, n_val)).detach().cpu().tolist()
        for i, name in enumerate(TARGET_NAMES):
            val_l1_per_comp[name].append(comp_mae_val[i])

        # Pretty console print with both MSE and MAE
        mae_tr = ", ".join([f"{k}: {train_l1_per_comp[k][-1]:.6f}" for k in TARGET_NAMES])
        mae_vl = ", ".join([f"{k}: {val_l1_per_comp[k][-1]:.6f}" for k in TARGET_NAMES])
        print(
            f"Epoch {epoch+1}/{epochs} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f} | "
            f"Train MAE [{mae_tr}] | Val MAE [{mae_vl}]"
        )

    return train_losses, val_losses, train_l1_per_comp, val_l1_per_comp
