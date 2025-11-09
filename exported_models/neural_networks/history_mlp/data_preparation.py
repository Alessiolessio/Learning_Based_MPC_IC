#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
data_preparation.py

Functions to read the dataset CSV, build history windows,
(optionally) normalize with StandardScaler, persist the scalers,
and split into train/validation sets.

NOTE:
- Keeps the original behavior: random split (not by episode),
  target at (t + history_length), inputs from [t ... t+H-1] with shift(-i).
"""

import os
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, random_split


def prepare_data(
    csv_path: str,
    scalers_dir: str = "trained_models",
    val_split_ratio: float = 0.2,
    normalize_data: bool = True,
    history_length: int = 1,
):
    """
    Reads the dataset from CSV, constructs history windows, optionally normalizes,
    and returns (train_data, val_data, input_scaler, target_scaler, input_dim, output_dim).

    Returns (None, None, None, None, None, None) on failure to keep a stable signature.
    """
    print(f"Reading and processing the dataset: {csv_path}")

    # -- Load CSV (fail early if missing) --
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return None, None, None, None, None, None

    # -- Define base columns used for inputs and targets --
    input_cols = ["x", "y", "yaw", "v", "w"]
    target_cols = ["x", "y", "yaw"]

    all_inputs = []
    all_targets = []

    # -- Build history windows per-episode (keeps original shifting: future-aligned) --
    print(f"Processing {len(df['episode'].unique())} episodes with history_length={history_length}...")
    for episode_id in df["episode"].unique():
        # Sort by 'step' to ensure temporal order
        df_episode = df[df["episode"] == episode_id].sort_values("step")

        # Base tables for state-action (inputs) and state (targets)
        state_action_df = df_episode[input_cols]
        target_state_df = df_episode[target_cols]

        dfs_to_concat = []
        input_feature_cols = []

        # 1) Build input features for each history offset (t, t+1, ... t+H-1)
        for i in range(history_length):
            shifted_inputs = state_action_df.shift(-i)
            current_cols = [f"{col}_h{i}" for col in input_cols]  # rename columns with suffix
            shifted_inputs.columns = current_cols
            dfs_to_concat.append(shifted_inputs)
            input_feature_cols.extend(current_cols)

        # 2) Build the target at t + history_length
        target_df = target_state_df.shift(-history_length).add_suffix("_next")
        target_feature_cols = [f"{col}_next" for col in target_cols]

        # 3) Concatenate inputs and target, then drop rows with NaN (trailing steps)
        dfs_to_concat.append(target_df)
        combined = pd.concat(dfs_to_concat, axis=1).dropna()

        if not combined.empty:
            all_inputs.append(combined[input_feature_cols])
            all_targets.append(combined[target_feature_cols])

    # -- Check if we produced any data at all --
    if not all_inputs:
        print("Error: No valid data was generated. Check the CSV or history_length.")
        return None, None, None, None, None, None

    # -- Final input/target frames (concatenate all episodes) --
    final_inputs_df = pd.concat(all_inputs)
    final_targets_df = pd.concat(all_targets)

    # -- Normalize (unchanged behavior: fit on full data) --
    if normalize_data:
        print("Normalizing data (StandardScaler)...")
        input_scaler = StandardScaler()
        target_scaler = StandardScaler()
        input_scaler.fit(final_inputs_df)
        target_scaler.fit(final_targets_df)

        # Transform to normalized arrays
        inputs_to_tensor = input_scaler.transform(final_inputs_df)
        targets_to_tensor = target_scaler.transform(final_targets_df)

        # Persist scalers alongside the model artifacts
        os.makedirs(scalers_dir, exist_ok=True)
        joblib.dump(input_scaler, os.path.join(scalers_dir, "input_scaler.joblib"))
        joblib.dump(target_scaler, os.path.join(scalers_dir, "target_scaler.joblib"))
        print(f"Scalers saved to: {scalers_dir}")
    else:
        print("Normalization skipped. Using raw data.")
        input_scaler = None
        target_scaler = None
        inputs_to_tensor = final_inputs_df.values
        targets_to_tensor = final_targets_df.values

    # -- Convert to tensors and pack into TensorDatasets --
    inputs_tensor = torch.tensor(inputs_to_tensor, dtype=torch.float32)
    targets_tensor = torch.tensor(targets_to_tensor, dtype=torch.float32)

    print(f"Processing complete. Total of {len(inputs_tensor)} samples generated.")

    # -- Detect dynamic dimensions for model construction --
    input_dim = inputs_tensor.shape[1]
    output_dim = targets_tensor.shape[1]
    print(f"Detected dimensions: Input={input_dim}, Output={output_dim}")

    # -- Random split (original behavior preserved) --
    print(f"Splitting the dataset (validation ratio: {val_split_ratio})...")
    dataset = TensorDataset(inputs_tensor, targets_tensor)

    # Reuse original seed logic (stable splits across runs/files)
    if csv_path == "/home/nexus/VQ_PMCnmpc/VQ_PMC/logs/datasets/dataset_random.csv":
        SEED = 50
    elif csv_path == "/home/nexus/VQ_PMCnmpc/VQ_PMC/logs/datasets/dataset_nmpc.csv":
        SEED = 50
    elif csv_path == "/home/nexus/VQ_PMCnmpc/VQ_PMC/logs/datasets/dataset_nmpc_better.csv":
        SEED = 51
    else:
        SEED = 50

    generator = torch.Generator()
    generator.manual_seed(SEED)

    val_len = int(val_split_ratio * len(dataset))
    train_len = len(dataset) - val_len
    train_data, val_data = random_split(dataset, [train_len, val_len], generator=generator)

    print(f"Split complete: {train_len} train samples, {val_len} validation samples.")
    return train_data, val_data, input_scaler, target_scaler, input_dim, output_dim
