#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
data_utils.py
(Substitui data_preparation.py)

Funções para carregar e processar o dataset.
Lê o CSV, normaliza e cria listas de episódios de treino e validação.
"""

import pandas as pd
import torch
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.model_selection import train_test_split

def load_and_prep_episodes(csv_path, scalers_dir, val_split_ratio=0.2, normalize_data=True):
    """
    Lê o dataset, separa em treino/validação POR EPISÓDIO,
    normaliza os dados e retorna listas de episódios processados.
    
    Retorna:
    - train_episodes (list): Lista de tuplas (inputs_tensor, targets_tensor)
    - val_episodes (list): Lista de tuplas (inputs_tensor, targets_tensor)
    - input_scaler: O scaler treinado
    - target_scaler: O scaler treinado
    """
    print(f"Lendo e processando o dataset: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Erro: Arquivo CSV não encontrado em {csv_path}")
        return None, None, None, None

    input_cols = ['x', 'y', 'yaw', 'v', 'w']
    target_cols = ['x', 'y', 'yaw']
    
    # Dicionários para guardar os dados brutos (Numpy)
    all_episode_inputs = {}
    all_episode_targets = {}
    
    episode_ids = df['episode'].unique()
    print(f"Processando {len(episode_ids)} episódios...")

    for episode_id in episode_ids:
        df_ep = df[df['episode'] == episode_id].sort_values('step')
        
        # Pega todos os inputs (x, y, yaw, v, w) do episódio
        inputs_np = df_ep[input_cols].values
        
        # Pega todos os targets (x, y, yaw) do episódio
        targets_np = df_ep[target_cols].values
        
        # Para a lógica (x1,x2,x3) -> y4, precisamos de pelo menos 4 steps
        if len(inputs_np) > 3: 
            all_episode_inputs[episode_id] = inputs_np
            all_episode_targets[episode_id] = targets_np

    if not all_episode_inputs:
        print("Erro: Nenhum episódio com dados suficientes foi encontrado.")
        return None, None, None, None

    print(f"Dados válidos de {len(all_episode_inputs)} episódios carregados.")

    # --- Divisão Treino/Validação (baseado nos IDs dos episódios) ---
    print(f"Separando episódios: {1-val_split_ratio:.0%} Treino, {val_split_ratio:.0%} Validação")
    train_ids, val_ids = train_test_split(list(all_episode_inputs.keys()), test_size=val_split_ratio, shuffle=True)

    # Concatena todos os steps APENAS dos episódios de TREINO para o scaler
    train_inputs_full = np.concatenate([all_episode_inputs[ep_id] for ep_id in train_ids], axis=0)
    train_targets_full = np.concatenate([all_episode_targets[ep_id] for ep_id in train_ids], axis=0)

    # --- Normalização ---
    input_scaler = None
    target_scaler = None
    
    if normalize_data:
        print("Normalizando dados (StandardScaler)...")
        input_scaler = StandardScaler()
        target_scaler = StandardScaler()
        
        # Fit (treina) o scaler APENAS nos dados de TREINO
        input_scaler.fit(train_inputs_full)
        target_scaler.fit(train_targets_full)
        
        os.makedirs(scalers_dir, exist_ok=True)
        joblib.dump(input_scaler, os.path.join(scalers_dir, 'input_scaler.joblib'))
        joblib.dump(target_scaler, os.path.join(scalers_dir, 'target_scaler.joblib'))
        print(f"Scalers salvos em: {scalers_dir}")
    else:
        print("Normalization skipped.")

    # --- Processa e Normaliza as listas de episódios ---
    train_episodes = []
    for ep_id in train_ids:
        inputs_raw = all_episode_inputs[ep_id]
        targets_raw = all_episode_targets[ep_id]
        if normalize_data:
            inputs_scaled = input_scaler.transform(inputs_raw)
            targets_scaled = target_scaler.transform(targets_raw)
            train_episodes.append((torch.tensor(inputs_scaled, dtype=torch.float32), 
                                   torch.tensor(targets_scaled, dtype=torch.float32)))
        else:
            train_episodes.append((torch.tensor(inputs_raw, dtype=torch.float32), 
                                   torch.tensor(targets_raw, dtype=torch.float32)))

    val_episodes = []
    for ep_id in val_ids:
        inputs_raw = all_episode_inputs[ep_id]
        targets_raw = all_episode_targets[ep_id]
        if normalize_data:
            inputs_scaled = input_scaler.transform(inputs_raw)
            targets_scaled = target_scaler.transform(targets_raw)
            val_episodes.append((torch.tensor(inputs_scaled, dtype=torch.float32), 
                                 torch.tensor(targets_scaled, dtype=torch.float32)))
        else:
            val_episodes.append((torch.tensor(inputs_raw, dtype=torch.float32), 
                                 torch.tensor(targets_raw, dtype=torch.float32)))

    print(f"Listas de episódios criadas: {len(train_episodes)} treino, {len(val_episodes)} validação.")
    
    return train_episodes, val_episodes, input_scaler, target_scaler