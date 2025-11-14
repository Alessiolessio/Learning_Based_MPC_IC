#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
training_runner.py

Runs the train/validation loop with Adam + MSE.
Logs per-component MAE (x, y, yaw or yaw_sin/yaw_cos) for visualization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def _names_for_output_dim(dim: int):
    if dim == 3:
        return ["x", "y", "yaw"]
    if dim == 4:
        return ["x", "y", "yaw_sin", "yaw_cos"]
    return [f"t{i}" for i in range(dim)]

def _init_component_logs(names):
    return {name: [] for name in names}

def train_model(model, train_data, val_data, epochs, learning_rate, batch_size, device):
    pin_mem = device.type == "cuda"
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=pin_mem)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, pin_memory=pin_mem)

    print(f"Starting training with {len(train_data)} train samples and {len(val_data)} validation samples.")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []

    sample_y = next(iter(train_loader))[1]
    out_dim = sample_y.shape[1]
    target_names = _names_for_output_dim(out_dim)

    train_l1_per_comp = _init_component_logs(target_names)
    val_l1_per_comp = _init_component_logs(target_names)

    print(f"Training for {epochs} epochs... (output_dim={out_dim} -> {target_names})")

    for epoch in range(epochs):
        # ------------------- TRAIN -------------------
        model.train()
        train_loss = 0.0
        comp_sum_abs = torch.zeros(out_dim, device=device)
        n_train = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device, non_blocking=pin_mem)
            y_batch = y_batch.to(device, non_blocking=pin_mem)

            pred = model(x_batch)
            loss = criterion(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            abs_err = (pred - y_batch).abs()
            comp_sum_abs += abs_err.sum(dim=0).detach()
            n_train += y_batch.shape[0]

        train_loss /= len(train_loader)
        if epoch > 0:
            train_losses.append(train_loss)

        comp_mae_train = (comp_sum_abs / max(1, n_train)).tolist()
        for i, name in enumerate(target_names):
            train_l1_per_comp[name].append(comp_mae_train[i])

        # ------------------- VAL -------------------
        model.eval()
        val_loss = 0.0
        comp_sum_abs_v = torch.zeros(out_dim, device=device)
        n_val = 0

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device, non_blocking=pin_mem)
                y_batch = y_batch.to(device, non_blocking=pin_mem)

                pred = model(x_batch)
                loss = criterion(pred, y_batch)
                val_loss += loss.item()

                abs_err = (pred - y_batch).abs()
                comp_sum_abs_v += abs_err.sum(dim=0)
                n_val += y_batch.shape[0]

        val_loss /= len(val_loader)
        if epoch > 0:
            val_losses.append(val_loss)

        comp_mae_val = (comp_sum_abs_v / max(1, n_val)).tolist()
        for i, name in enumerate(target_names):
            val_l1_per_comp[name].append(comp_mae_val[i])

        mae_str_train = ", ".join([f"{k}: {train_l1_per_comp[k][-1]:.6f}" for k in target_names])
        mae_str_val = ", ".join([f"{k}: {val_l1_per_comp[k][-1]:.6f}" for k in target_names])
        print(
            f"Epoch {epoch+1}/{epochs} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f} | "
            f"Train MAE [{mae_str_train}] | Val MAE [{mae_str_val}]"
        )

    return train_losses, val_losses, train_l1_per_comp, val_l1_per_comp
