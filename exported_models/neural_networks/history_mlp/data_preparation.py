#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
data_preparation.py

Reads the dataset CSV, builds history windows per episode,
(optionally) normalizes with StandardScaler, persists the scalers,
and splits into train/validation sets with an **episode-based** split.

Notes:
- Preserves original windowing: inputs from [t ... t+H-1] using shift(-i),
  target at (t + history_length).
- Scalers are still fit on the **full** dataset (train+val), as before.
"""

import os
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset


def _seed_from_csv(csv_path: str) -> int:
    """Deterministic seed based on known CSV paths (keeps prior behavior)."""
    if csv_path == "/home/nexus/VQ_PMCnmpc/VQ_PMC/logs/datasets/dataset_random.csv":
        return 50
    if csv_path == "/home/nexus/VQ_PMCnmpc/VQ_PMC/logs/datasets/dataset_nmpc.csv":
        return 50
    if csv_path == "/home/nexus/VQ_PMCnmpc/VQ_PMC/logs/datasets/dataset_nmpc_better.csv":
        return 51
    return 50


def prepare_data(
    csv_path: str,
    scalers_dir: str = "trained_models",
    val_split_ratio: float = 0.2,
    normalize_data: bool = True,
    history_length: int = 1,
):
    """
    Reads the dataset, constructs history windows **per episode**, performs
    an **episode-based** split (no episode appears in both sets),
    and returns (train_data, val_data, input_scaler, target_scaler, input_dim, output_dim).

    Returns (None, None, None, None, None, None) on failure.
    """
    print(f"Reading and processing the dataset: {csv_path}")

    # -- Load CSV (fail early if missing) --
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return None, None, None, None, None, None

    # -- Columns used for inputs and targets --
    input_cols = ["x", "y", "yaw", "v", "w"]
    target_cols = ["x", "y", "yaw"]

    # We’ll keep per-episode frames to enable episode-wise split later
    inputs_by_ep = {}
    targets_by_ep = {}

    uniq_eps = df["episode"].unique()
    print(f"Processing {len(uniq_eps)} episodes with history_length={history_length}...")

    # --------------------- Build windows per episode ---------------------
    for ep in uniq_eps:
        # Sort by 'step' to ensure temporal order
        df_ep = df[df["episode"] == ep].sort_values("step")

        # Base tables for state-action (inputs) and state (targets)
        state_action_df = df_ep[input_cols]
        target_state_df = df_ep[target_cols]

        # Accumulators
        dfs_to_concat = []
        input_feature_cols = []

        # 1) Inputs for each history offset (t, t+1, ... t+H-1) using shift(-i)
        for i in range(history_length):
            shifted = state_action_df.shift(-i).copy()
            renamed = [f"{col}_h{i}" for col in input_cols]  # name with suffix
            shifted.columns = renamed
            dfs_to_concat.append(shifted)
            input_feature_cols.extend(renamed)

        # 2) Target at t + history_length (future-aligned target)
        target_df = target_state_df.shift(-history_length).add_suffix("_next")
        target_feature_cols = [f"{c}_next" for c in target_cols]

        # 3) Concatenate inputs and target, drop trailing NaNs within the episode
        dfs_to_concat.append(target_df)
        combined = pd.concat(dfs_to_concat, axis=1).dropna()

        # Store non-empty per-episode frames
        if not combined.empty:
            inputs_by_ep[ep] = combined[input_feature_cols]
            targets_by_ep[ep] = combined[target_feature_cols]

    if not inputs_by_ep:
        print("Error: No valid data was generated. Check the CSV or history_length.")
        return None, None, None, None, None, None

    # --------------------- Episode-based split (hold-out) ---------------------
    # Shuffle episodes deterministically (same idea as old random split seed)
    rng = np.random.RandomState(_seed_from_csv(csv_path))
    episodes = list(inputs_by_ep.keys())
    rng.shuffle(episodes)

    n_val = max(1, int(round(val_split_ratio * len(episodes))))  # at least 1 val episode
    val_eps = set(episodes[:n_val])
    train_eps = [e for e in episodes if e not in val_eps]

    def _concat_subset(dct, keys):
        frames = [dct[k] for k in keys if k in dct]
        return pd.concat(frames) if frames else pd.DataFrame()

    # Concatenate full episodes into train/val frames
    train_inputs_df = _concat_subset(inputs_by_ep, train_eps)
    train_targets_df = _concat_subset(targets_by_ep, train_eps)
    val_inputs_df = _concat_subset(inputs_by_ep, val_eps)
    val_targets_df = _concat_subset(targets_by_ep, val_eps)

    if train_inputs_df.empty or val_inputs_df.empty:
        print("Error: Episode-based split produced empty set(s). Try adjusting val_split_ratio.")
        return None, None, None, None, None, None

    # --------------------- Normalization (fit on ALL data) ---------------------
    # Preserves previous behavior: fit scalers on the full dataset (train+val)
    if normalize_data:
        print("Normalizing data (StandardScaler) on all samples (train+val)...")
        all_inputs_df = pd.concat([train_inputs_df, val_inputs_df])
        all_targets_df = pd.concat([train_targets_df, val_targets_df])

        input_scaler = StandardScaler()
        target_scaler = StandardScaler()
        input_scaler.fit(all_inputs_df)
        target_scaler.fit(all_targets_df)

        # Transform to normalized arrays
        train_inputs_np = input_scaler.transform(train_inputs_df)
        train_targets_np = target_scaler.transform(train_targets_df)
        val_inputs_np = input_scaler.transform(val_inputs_df)
        val_targets_np = target_scaler.transform(val_targets_df)

        # Persist scalers alongside the model artifacts (same as before)
        os.makedirs(scalers_dir, exist_ok=True)
        joblib.dump(input_scaler, os.path.join(scalers_dir, "input_scaler.joblib"))
        joblib.dump(target_scaler, os.path.join(scalers_dir, "target_scaler.joblib"))
        print(f"Scalers saved to: {scalers_dir}")
    else:
        print("Normalization skipped. Using raw values.")
        input_scaler = None
        target_scaler = None
        train_inputs_np = train_inputs_df.values
        train_targets_np = train_targets_df.values
        val_inputs_np = val_inputs_df.values
        val_targets_np = val_targets_df.values

    # --------------------- Tensors & Datasets ---------------------
    train_inputs_tensor = torch.tensor(train_inputs_np, dtype=torch.float32)
    train_targets_tensor = torch.tensor(train_targets_np, dtype=torch.float32)
    val_inputs_tensor = torch.tensor(val_inputs_np, dtype=torch.float32)
    val_targets_tensor = torch.tensor(val_targets_np, dtype=torch.float32)

    train_data = TensorDataset(train_inputs_tensor, train_targets_tensor)
    val_data = TensorDataset(val_inputs_tensor, val_targets_tensor)

    # Dynamic dimensions for the model
    input_dim = train_inputs_tensor.shape[1]
    output_dim = train_targets_tensor.shape[1]

    print(f"Detected dimensions: Input={input_dim}, Output={output_dim}")
    print(f"Episode split: {len(train_eps)} train ep(s), {len(val_eps)} val ep(s).")

    return train_data, val_data, input_scaler, target_scaler, input_dim, output_dim
