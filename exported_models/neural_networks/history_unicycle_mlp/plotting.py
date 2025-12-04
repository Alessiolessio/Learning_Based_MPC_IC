#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plotting.py

Helper functions to plot: (1) Train vs. Validation MSE loss,
and (2) Per-component MAE curves.
"""

import os
import matplotlib.pyplot as plt


def plot_losses(train_losses, val_losses, save_dir: str = "."):
    """Plot training vs. validation loss (MSE) and save the figure."""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("MLP: Train vs. Validation Loss per Epoch")
    plt.xlabel("Epoch (starting from 2)")
    plt.ylabel("Loss (MSE Loss)")
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(save_dir, "loss_plot.png")
    plt.savefig(save_path)
    print(f"Loss plot saved to: {save_path}")
    plt.close()


def plot_component_l1(train_l1_per_comp, val_l1_per_comp, save_dir: str = "."):
    """Plot per-component MAE curves (train and val) in the same figure."""
    plt.figure(figsize=(10, 6))
    for name, series in train_l1_per_comp.items():
        plt.plot(series, label=f"Train MAE {name}")
    for name, series in val_l1_per_comp.items():
        plt.plot(series, linestyle="--", label=f"Val MAE {name}")
    plt.title("Per-Component MAE (|error|) over Epochs")
    plt.xlabel("Epoch (starting from 2)")
    plt.ylabel("MAE (L1) in normalized units")
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(save_dir, "component_l1_plot.png")
    plt.savefig(save_path)
    print(f"Component L1 plot saved to: {save_path}")
    plt.close()
