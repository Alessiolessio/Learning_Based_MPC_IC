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

# <--- MUDANÇA: Adicionado 'normalize_data=True' como parâmetro
def prepare_data(csv_path, scalers_dir="trained_models", val_split_ratio=0.2, normalize_data=True):
    """
    Reads the dataset.csv, transforms, (optionally) normalizes, 
    and SPLITS it into train/validation.
    
    Saves the 'scaler' objects for future use if normalization is enabled.
    
    Args:
        csv_path (str): Path to the CSV file.
        scalers_dir (str): Directory to save the scaler .joblib files.
        val_split_ratio (float): Proportion of the dataset to use for validation (e.g., 0.2).
        normalize_data (bool): If True, applies StandardScaler. If False, uses raw data.
        
    Returns:
        tuple: (train_data, val_data, input_scaler, target_scaler)
               (Scalers will be None if normalize_data is False)
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

    # <--- MUDANÇA: Bloco condicional para normalização ---
    if normalize_data:
        print("Normalizing data (StandardScaler)...")
        input_scaler = StandardScaler()
        target_scaler = StandardScaler()
        
        input_scaler.fit(final_inputs_df)
        target_scaler.fit(final_targets_df)
        
        # Usa os dados normalizados
        inputs_to_tensor = input_scaler.transform(final_inputs_df)
        targets_to_tensor = target_scaler.transform(final_targets_df)
        
        os.makedirs(scalers_dir, exist_ok=True)
        joblib.dump(input_scaler, os.path.join(scalers_dir, 'input_scaler.joblib'))
        joblib.dump(target_scaler, os.path.join(scalers_dir, 'target_scaler.joblib'))
        print(f"Scalers saved to: {scalers_dir}")
    
    else:
        print("Normalization skipped. Using raw data.")
        # Define scalers como None para consistência no retorno
        input_scaler = None
        target_scaler = None
        
        # Usa os dados brutos (convertidos para numpy array)
        inputs_to_tensor = final_inputs_df.values
        targets_to_tensor = final_targets_df.values
    # <--- FIM DA MUDANÇA ---

    # Converte os dados (normalizados ou brutos) para Tensores
    inputs_tensor = torch.tensor(inputs_to_tensor, dtype=torch.float32)
    targets_tensor = torch.tensor(targets_to_tensor, dtype=torch.float32)

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

    # Retorna os dados divididos e os scalers (que podem ser None)
    return train_data, val_data, input_scaler, target_scaler
