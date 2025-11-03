#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plotting.py

Location: /home/nexus/VQ_PMCnmpc/VQ_PMC/exported_models/neural_networks/raw_mlp

Utility function for plotting loss graphs.
"""

import matplotlib.pyplot as plt
import os

def plot_losses(train_losses, val_losses, save_dir="."):
    """
    Plots the Training vs. Validation Loss graph.
    
    This helps us see if the network is learning (both falling)
    or memorizing (Train falling, Val rising - "overfitting").
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("MLP: Train vs. Validation Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE Loss)")
    plt.legend()
    plt.grid(True)
    
    # Saves the plot to a PNG file
    save_path = os.path.join(save_dir, "loss_plot.png")
    plt.savefig(save_path)
    print(f"Loss plot saved to: {save_path}")
    plt.close()