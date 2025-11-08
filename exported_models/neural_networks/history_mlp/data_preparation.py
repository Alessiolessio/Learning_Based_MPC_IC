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

# <--- MUDANÇA 1: Adicionado 'history_length=1' (default 1 para manter compatibilidade)
def prepare_data(csv_path, scalers_dir="trained_models", val_split_ratio=0.2, normalize_data=True, history_length=1):
    """
    Reads the dataset.csv, transforms, (optionally) normalizes, 
    and SPLITS it into train/validation.
    
    Saves the 'scaler' objects for future use if normalization is enabled.
    
    Args:
        csv_path (str): Path to the CSV file.
        scalers_dir (str): Directory to save the scaler .joblib files.
        val_split_ratio (float): Proportion of the dataset to use for validation (e.g., 0.2).
        normalize_data (bool): If True, applies StandardScaler. If False, uses raw data.
        history_length (int): Number of time steps to use as input sequence (e.g., 3).
        
    Returns:
        tuple: (train_data, val_data, input_scaler, target_scaler, input_dim, output_dim)
               (Scalers will be None if normalize_data is False)
               On failure, returns (None, None, None, None, None, None)
    """
    print(f"Reading and processing the dataset: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        # <--- MUDANÇA 3a: Atualiza retorno de falha
        return None, None, None, None, None, None

    input_cols = ['x', 'y', 'yaw', 'v', 'w']
    target_cols = ['x', 'y', 'yaw']

    all_inputs = []
    all_targets = []

    # <--- MUDANÇA 2: Lógica de processamento de episódio reescrita ---
    print(f"Processing {len(df['episode'].unique())} episodes with history_length={history_length}...")
    for episode_id in df['episode'].unique():
        df_episode = df[df['episode'] == episode_id].sort_values('step')
        
        # Pega os dataframes de input (estado+ação) e target (estado)
        state_action_df = df_episode[input_cols]
        target_state_df = df_episode[target_cols]

        dfs_to_concat = []
        input_feature_cols = []

        # 1. Constrói colunas de input histórico (t, t+1, ... t+history_length-1)
        for i in range(history_length):
            shifted_inputs = state_action_df.shift(-i)
            # Renomeia colunas para ex: 'x_h0', 'y_h0', ... 'x_h1', 'y_h1', ...
            current_cols = [f"{col}_h{i}" for col in input_cols]
            shifted_inputs.columns = current_cols
            
            dfs_to_concat.append(shifted_inputs)
            input_feature_cols.extend(current_cols)

        # 2. Constrói coluna do target (estado em t + history_length)
        target_df = target_state_df.shift(-history_length).add_suffix('_next')
        target_feature_cols = [col + '_next' for col in target_cols]
        
        dfs_to_concat.append(target_df)

        # 3. Combina tudo e remove NaNs (linhas no fim do episódio que não formam uma sequência completa)
        combined = pd.concat(dfs_to_concat, axis=1)
        combined = combined.dropna()

        if not combined.empty:
            all_inputs.append(combined[input_feature_cols])
            all_targets.append(combined[target_feature_cols])
    # <--- FIM DA MUDANÇA 2 ---

    if not all_inputs:
        print("Error: No valid data was generated. Check the CSV or history_length.")
        # <--- MUDANÇA 3b: Atualiza retorno de falha
        return None, None, None, None, None, None

    final_inputs_df = pd.concat(all_inputs)
    final_targets_df = pd.concat(all_targets)

    # <--- Bloco de normalização (sem mudanças) ---
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
    # <--- MUDANÇA 3c: Captura as dimensões dinamicamente
    input_dim = inputs_tensor.shape[1]
    output_dim = targets_tensor.shape[1]
    print(f"Detected dimensions: Input={input_dim}, Output={output_dim}")


    # --- Train/Validation Split ---
    print(f"Splitting the dataset (validation ratio: {val_split_ratio})...")
    
    # 1. Create the TensorDataset
    dataset = TensorDataset(inputs_tensor, targets_tensor)

    # 2. Perform the split
    if csv_path == "/home/nexus/VQ_PMCnmpc/VQ_PMC/logs/datasets/dataset_random.csv":
        SEED = 50   # SEED for dataset_random split
    elif csv_path == "/home/nexus/VQ_PMCnmpc/VQ_PMC/logs/datasets/dataset_nmpc.csv":
        SEED = 50   # SEED for dataset_nmpc split
    elif csv_path == "/home/nexus/VQ_PMCnmpc/VQ_PMC/logs/datasets/dataset_nmpc_better.csv":
        SEED = 51   # SEED for dataset_nmpc_better split
    else:
        SEED = 50   # SEED for other dataset split
    generator = torch.Generator()
    generator.manual_seed(SEED)
    val_len = int(val_split_ratio * len(dataset))
    train_len = len(dataset) - val_len
    train_data, val_data = random_split(dataset, [train_len, val_len], generator=generator)
    
    print(f"Split complete: {train_len} train samples, {val_len} validation samples.")

    # <--- MUDANÇA 3d: Adiciona dims ao retorno
    return train_data, val_data, input_scaler, target_scaler, input_dim, output_dim