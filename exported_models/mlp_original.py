#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mlp.py

This script trains an MLP (Multilayer Perceptron) Neural Network to
learn the kinematic dynamics of a robot from a dataset.

Objective: (System Identification)
  The network learns to predict the *next state* (x, y, yaw)
  given the *current state* (x, y, yaw) and the *current action* (v, w).

Model Format:
  (x_{k+1}, y_{k+1}, yaw_{k+1}) = MLP(x_k, y_k, yaw_k, v_k, w_k)

How to run:
1. Save this code as 'mlp.py'
2. Ensure you have the dataset (e.g., 'dataset_random.csv') in the correct folder.
3. Install the libraries: pip3 install torch pandas matplotlib scikit-learn joblib
4. Run in the terminal:
   python3 mlp.py --csv_path /home/nexus/VQ_PMCnmpc/VQ_PMC/logs/datasets/dataset_random.csv
"""

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import argparse
import matplotlib.pyplot as plt
import os
import numpy as np # Used for data processing
from sklearn.preprocessing import StandardScaler
import joblib

# --- 1. Neural Network (MLP) Definition ---

class MLPDynamicsModel(nn.Module):
    """
    Defines the architecture of our Neural Network (MLP).
    
    'nn.Module' is the PyTorch base class for all models.
    """
    def __init__(self, input_dim, output_dim):
        """
        The class 'constructor'. Defines the network layers.
        
        - input_dim: How many numbers go into the network (e.g., 5)
        - output_dim: How many numbers come out of the network (e.g., 3)
        """
        # Calls the constructor of the parent class (nn.Module)
        super(MLPDynamicsModel, self).__init__()
        
        # 'nn.Sequential' is a container that runs layers in order.
        # Think of it as a pipeline for the data.
        self.model = nn.Sequential(
            # 1st Layer: Linear(inputs, neurons)
            # Takes 'input_dim' (5) and transforms it to 64.
            # This is the input layer.
            nn.Linear(input_dim, 64),
            
            # Activation Function: ReLU (Rectified Linear Unit)
            # Introduces non-linearity. f(x) = max(0, x).
            # Essential for the network to learn complex patterns.
            nn.ReLU(),
            
            # 2nd Layer: Linear(previous_layer_neurons, neurons)
            # Takes 64 and transforms it to 64. This is a "hidden layer".
            nn.Linear(64, 64),
            nn.ReLU(),
            
            # 3rd Layer: Linear(neurons, outputs)
            # Takes 64 and transforms it to 'output_dim' (3).
            # This is the output layer. Note there is no ReLU at the end,
            # as we want to predict continuous values (x, y, yaw).
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        """
        Defines the "forward pass".
        This is called automatically when you do model(input).
        
        It simply tells PyTorch to pass the input 'x' through
        the sequential model we defined.
        """
        return self.model(x)


# --- 2. Data Preparation ---

def prepare_data(csv_path, scalers_dir="trained_models"):
    """
    Reads the dataset.csv and transforms it into (input, target) pairs.
    
    It also normalizes the data (StandardScaler) and saves the
    'scaler' objects for future use (for prediction).
    """
    print(f"Reading and processing the dataset: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return None, None, None, None

    input_cols = ['x', 'y', 'yaw', 'v', 'w']
    target_cols = ['x', 'y', 'yaw']

    all_inputs = []
    all_targets = []

    print(f"Processing {len(df['episode'].unique())} episodes...")
    for episode_id in df['episode'].unique():
        df_episode = df[df['episode'] == episode_id].sort_values('step')
        inputs_df = df_episode[input_cols]
        targets_df = df_episode[target_cols].shift(-1)
        combined = pd.concat([inputs_df, targets_df.add_suffix('_next')], axis=1)
        combined = combined.dropna()

        if not combined.empty:
            all_inputs.append(combined[input_cols])
            all_targets.append(combined[[col + '_next' for col in target_cols]])

    if not all_inputs:
        print("Error: No valid data was generated. Check the CSV.")
        return None, None, None, None

    final_inputs_df = pd.concat(all_inputs)
    final_targets_df = pd.concat(all_targets)

    # --- Data Normalization ---
    print("Normalizing data (StandardScaler)...")
    # 1. Create the 'scalers'
    input_scaler = StandardScaler()
    target_scaler = StandardScaler()

    # 2. Fit the scalers (learn mean and std)
    input_scaler.fit(final_inputs_df)
    target_scaler.fit(final_targets_df)

    # 3. Transform the data
    inputs_scaled = input_scaler.transform(final_inputs_df)
    targets_scaled = target_scaler.transform(final_targets_df)

    # 4. Save the scalers (VERY IMPORTANT)
    # (We need them later to 'denormalize' the network's predictions)
    os.makedirs(scalers_dir, exist_ok=True)
    joblib.dump(input_scaler, os.path.join(scalers_dir, 'input_scaler.joblib'))
    joblib.dump(target_scaler, os.path.join(scalers_dir, 'target_scaler.joblib'))
    print(f"Scalers saved to: {scalers_dir}")

    # Converts the NORMALIZED data to Tensors
    inputs_tensor = torch.tensor(inputs_scaled, dtype=torch.float32)
    targets_tensor = torch.tensor(targets_scaled, dtype=torch.float32)

    print(f"Processing complete. Total of {len(inputs_tensor)} training samples generated.")

    # Returns the tensors AND the scalers
    return inputs_tensor, targets_tensor, input_scaler, target_scaler

# --- 3. Plotting Function ---

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
    # plt.show() # Uncomment if you want the plot to display on screen
    plt.close()

# --- 4. Training Loop ---

def main(args):
    """
    Main function that runs data preparation,
    training, and evaluation.
    """
    
    # --- Data Preparation ---
    inputs, targets, _, _ = prepare_data(args.csv_path, scalers_dir=os.path.dirname(args.model_path))
    if inputs is None:
        return # Exits if data preparation failed
        
    # TensorDataset is a container that joins inputs and targets
    dataset = TensorDataset(inputs, targets)

    # Split the dataset: 80% for Training, 20% for Validation
    # Validation: Data the network doesn't use for training, only to
    # check if it is learning ("generalizing") or just
    # memorizing ("overfitting").
    val_len = int(0.2 * len(dataset))
    train_len = len(dataset) - val_len
    train_data, val_data = random_split(dataset, [train_len, val_len])

    # DataLoaders: Tools that shuffle and split data into "batches"
    # batch_size: How many examples the network sees before updating its weights.
    # shuffle=True: Shuffles the training data every epoch (essential)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    print(f"Starting training with {train_len} train samples and {val_len} validation samples.")
    
    # --- Model Initialization ---
    
    # Input/Output Decision:
    # input_dim = 5 (x, y, yaw, v, w)
    # output_dim = 3 (x_next, y_next, yaw_next)
    input_dim = 5
    output_dim = 3
    model = MLPDynamicsModel(input_dim=input_dim, output_dim=output_dim)
    
    # Optimizer: 'Adam' is a modern and robust optimizer.
    # It is the "engine" that adjusts the network weights.
    # lr (learning_rate): The "step size" the optimizer takes.
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Loss Function: 'MSELoss' (Mean Squared Error)
    # Measures the error: (prediction - target)². Perfect for regression (predicting numbers).
    criterion = nn.MSELoss()

    # Lists to store loss history for plotting
    train_losses = []
    val_losses = []

    print(f"Training for {args.epochs} epochs...")
    # --- Training Loop ---
    for epoch in range(args.epochs):
        
        # --- Training Phase ---
        model.train() # Puts the model in "train" mode
        train_loss = 0.0
        
        # Iterates over the training data batches
        for x_batch, y_batch in train_loader:
            # x_batch: batch of inputs (state k, action k)
            # y_batch: batch of targets (state k+1)
            
            # 1. Make the prediction
            pred = model(x_batch)
            
            # 2. Calculate the error (Loss) between the prediction (pred) and the target (y_batch)
            loss = criterion(pred, y_batch)
            
            # 3. Zero out old gradients (important)
            optimizer.zero_grad()
            
            # 4. Calculate new gradients (Backpropagation)
            # (Calculates how each weight contributed to the error)
            loss.backward()
            
            # 5. Update the network weights
            optimizer.step()
            
            # Accumulates the batch loss
            train_loss += loss.item()
            
        # Calculates the average training loss for this epoch
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # --- Validation Phase ---
        model.eval() # Puts the model in "evaluation" mode (disables dropout, etc)
        val_loss = 0.0
        
        # 'with torch.no_grad()' disables gradient calculation
        # (saves memory and time, since we are not training)
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                # 1. Make the prediction
                pred = model(x_batch)
                # 2. Calculate the error (Loss)
                loss = criterion(pred, y_batch)
                val_loss += loss.item()
                
        # Calculates the average validation loss for this epoch
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Print progress
        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}")

    # --- End of Training ---
    print("Training complete.")

    # Save the trained model
    if args.model_path:
        # Create the directory (e.g., "trained_models") if it doesn't exist
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        # Saves the "state dictionary" (the trained weights)
        torch.save(model.state_dict(), args.model_path)
        print(f"Model saved to {args.model_path}")

    # Plot the loss graphs
    # Save the plot in the same folder as the model
    plot_dir = os.path.dirname(args.model_path) if args.model_path else "."
    plot_losses(train_losses, val_losses, save_dir=plot_dir)


# --- 5. Script Entry Point ---
if __name__ == "__main__":
    # Ex: python3 mlp.py --csv_path "my_dataset.csv" --epochs 50
    
    parser = argparse.ArgumentParser(description="Trains an MLP model to predict robot dynamics.")
    
    parser.add_argument("--csv_path", type=str, default="/home/nexus/VQ_PMCnmpc/VQ_PMC/logs/datasets/dataset_random.csv", 
                        help="Path to the dataset.csv file.")
    
    parser.add_argument("--epochs", type=int, default=100, 
                        help="Number of training epochs.")
    
    parser.add_argument("--batch_size", type=int, default=128, 
                        help="Batch size for training.")
    
    parser.add_argument("--learning_rate", type=float, default=1e-4, # (0.0001)
                        help="Learning rate for the optimizer.")
    
    parser.add_argument("--model_path", type=str, default="trained_models/mlp_dynamics.pth",
                        help="Path to save the trained model (.pth file).")
    
    args = parser.parse_args()
    
    main(args)