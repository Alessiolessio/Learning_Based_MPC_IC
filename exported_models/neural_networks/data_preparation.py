#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
data_preparation.py

Location: /home/nexus/VQ_PMCnmpc/VQ_PMC/exported_models/neural_networks/

Functions to read the .csv dataset, process it, normalize,
save the scalers (StandardScaler), and split into train/validation.
"""

import pandas as pd
import torch
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from torch.utils.data import TensorDataset, random_split

def prepare_data(csv_path, scalers_dir="trained_models", val_split_ratio=0.2):
    """
    Reads the dataset.csv, transforms, normalizes, and SPLITS it into train/validation.
    
    Saves the 'scaler' objects for future use.
    
    Args:
        csv_path (str): Path to the CSV file.
        scalers_dir (str): Directory to save the scaler .joblib files.
        val_split_ratio (float): Proportion of the dataset to use for validation (e.g., 0.2).
        
    Returns:
        tuple: (train_data, val_data, input_scaler, target_scaler)
               On failure, returns (None, None, None, None)
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

    print("Normalizing data (StandardScaler)...")
    input_scaler = StandardScaler()
    target_scaler = StandardScaler()
    input_scaler.fit(final_inputs_df)
    target_scaler.fit(final_targets_df)
    inputs_scaled = input_scaler.transform(final_inputs_df)
    targets_scaled = target_scaler.transform(final_targets_df)
    os.makedirs(scalers_dir, exist_ok=True)
    joblib.dump(input_scaler, os.path.join(scalers_dir, 'input_scaler.joblib'))
    joblib.dump(target_scaler, os.path.join(scalers_dir, 'target_scaler.joblib'))
    print(f"Scalers saved to: {scalers_dir}")

    # Convert the NORMALIZED data to Tensors
    inputs_tensor = torch.tensor(inputs_scaled, dtype=torch.float32)
    targets_tensor = torch.tensor(targets_scaled, dtype=torch.float32)

    print(f"Processing complete. Total of {len(inputs_tensor)} samples generated.")

    # --- Train/Validation Split ---
    print(f"Splitting the dataset (validation ratio: {val_split_ratio})...")
    
    # 1. Create the TensorDataset
    dataset = TensorDataset(inputs_tensor, targets_tensor)

    # 2. Perform the split
    val_len = int(val_split_ratio * len(dataset))
    train_len = len(dataset) - val_len
    train_data, val_data = random_split(dataset, [train_len, val_len])
    
    print(f"Split complete: {train_len} train samples, {val_len} validation samples.")

    # Return the already split data and the scalers
    return train_data, val_data, input_scaler, target_scaler