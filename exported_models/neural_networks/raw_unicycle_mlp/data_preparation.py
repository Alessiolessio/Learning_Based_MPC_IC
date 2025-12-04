#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
data_preparation.py

Reads the dataset CSV, builds single-step pairs (t -> t+1) per episode,
optionally applies StandardScaler, persists the scalers, and performs
a random train/validation split (seeded by CSV path).
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
):
    """
    Build (X_t, Y_{t+1}) pairs episode-by-episode and split into train/val.

    Returns:
        (train_data, val_data, input_scaler, target_scaler)
        or (None, None, None, None) on failure.
    """
    print(f"Reading and processing the dataset: {csv_path}")

    # -- Load CSV early and fail fast on missing file --
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return None, None, None, None

    # -- Columns used for inputs and targets --
    input_cols = ["x", "y", "yaw", "v", "w"]
    target_cols = ["x", "y", "yaw"]

    all_inputs = []
    all_targets = []

    # -- Process each episode independently to keep time order intact --
    print(f"Processing {len(df['episode'].unique())} episodes...")
    for episode_id in df["episode"].unique():
        # Sort by 'step' so that shifts align to temporal order
        df_ep = df[df["episode"] == episode_id].sort_values("step")

        # Build inputs at time t
        inputs_df = df_ep[input_cols]

        # Build targets at time t+1 (one-step ahead)
        targets_df = df_ep[target_cols].shift(-1)  # future alignment

        # Concatenate and drop incomplete tail rows
        combined = pd.concat([inputs_df, targets_df.add_suffix("_next")], axis=1).dropna()

        if not combined.empty:
            all_inputs.append(combined[input_cols])
            all_targets.append(combined[[f"{c}_next" for c in target_cols]])

    # -- Ensure some data was produced --
    if not all_inputs:
        print("Error: No valid data was generated. Check the CSV.")
        return None, None, None, None

    # -- Merge all episodes into final frames --
    final_inputs_df = pd.concat(all_inputs)
    final_targets_df = pd.concat(all_targets)

    # -- Optional normalization with StandardScaler (fit on full data) --
    if normalize_data:
        print("Normalizing data (StandardScaler)...")
        input_scaler = StandardScaler()
        target_scaler = StandardScaler()

        input_scaler.fit(final_inputs_df)
        target_scaler.fit(final_targets_df)

        inputs_np = input_scaler.transform(final_inputs_df)
        targets_np = target_scaler.transform(final_targets_df)

        # Persist scalers inside the run folder
        os.makedirs(scalers_dir, exist_ok=True)
        joblib.dump(input_scaler, os.path.join(scalers_dir, "input_scaler.joblib"))
        joblib.dump(target_scaler, os.path.join(scalers_dir, "target_scaler.joblib"))
        print(f"Scalers saved to: {scalers_dir}")
    else:
        print("Normalization skipped. Using raw data.")
        input_scaler = None
        target_scaler = None
        inputs_np = final_inputs_df.values
        targets_np = final_targets_df.values

    # -- Convert to tensors and make a TensorDataset --
    inputs_tensor = torch.tensor(inputs_np, dtype=torch.float32)
    targets_tensor = torch.tensor(targets_np, dtype=torch.float32)
    dataset = TensorDataset(inputs_tensor, targets_tensor)

    print(f"Processing complete. Total of {len(inputs_tensor)} samples generated.")
    print(f"Splitting the dataset (validation ratio: {val_split_ratio})...")

    # -- Deterministic split seed based on known CSV paths (as in original) --
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

    # -- Compute split sizes and perform the split --
    val_len = int(val_split_ratio * len(dataset))
    train_len = len(dataset) - val_len
    train_data, val_data = random_split(dataset, [train_len, val_len], generator=generator)

    print(f"Split complete: {train_len} train samples, {val_len} validation samples.")
    return train_data, val_data, input_scaler, target_scaler
