#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
data_preparation.py

Builds sequence windows per episode, applies feature scaling,
persists sklearn scalers, and returns PyTorch datasets for training/validation.

Notes:
- Episode-based split (no episode appears in both train and val).
- Yaw is represented as sin/cos both in inputs (per-history step) and target (at t+H).
"""

import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import joblib
from torch.utils.data import TensorDataset

# ----------------------------- Helpers -----------------------------

def _episode_seed_from_csv(csv_path: str) -> int:
    # Keep the same deterministic seeds used previously for reproducibility
    if csv_path == "/home/nexus/VQ_PMCnmpc/VQ_PMC/logs/datasets/dataset_random.csv":
        return 50
    elif csv_path == "/home/nexus/VQ_PMCnmpc/VQ_PMC/logs/datasets/dataset_nmpc.csv":
        return 50
    elif csv_path == "/home/nexus/VQ_PMCnmpc/VQ_PMC/logs/datasets/dataset_nmpc_better.csv":
        return 51
    return 50

def _wrap_to_pi(a: np.ndarray) -> np.ndarray:
    # Wraps angles to (-pi, pi]
    return (a + np.pi) % (2 * np.pi) - np.pi

# ----------------------------- Main API -----------------------------

def prepare_data(
    csv_path: str,
    scalers_dir: str = "trained_models",
    val_split_ratio: float = 0.2,
    normalize_data: bool = True,
    history_length: int = 1,
):
    """
    Reads CSV, builds history windows per-episode, splits episodes into train/val,
    scales inputs/targets (optionally), and returns TensorDatasets.

    Returns: (train_data, val_data, input_scaler, target_scaler, input_dim, output_dim)
    """
    print(f"Reading and processing the dataset: {csv_path}")

    # -- Load CSV (fail early if absent) --
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return None, None, None, None, None, None

    # -- Containers to aggregate episode-wise dataframes --
    inputs_by_ep = {}
    targets_by_ep = {}

    uniq_episodes = df["episode"].unique()
    print(f"Processing {len(uniq_episodes)} episodes with history_length={history_length}...")

    # ------------------------ Build windows (per episode) ------------------------
    for episode_id in uniq_episodes:
        # Sort steps to keep the temporal order
        df_ep = df[df["episode"] == episode_id].sort_values("step").copy()

        # Prepare yaw -> sin/cos for all rows in the current episode
        yaw = df_ep["yaw"].to_numpy(dtype=float)
        yaw = _wrap_to_pi(yaw)
        yaw_sin = np.sin(yaw)
        yaw_cos = np.cos(yaw)

        # State-action table used to create input windows (per-history step)
        state_action_df = pd.DataFrame({
            "x": df_ep["x"].to_numpy(dtype=float),
            "y": df_ep["y"].to_numpy(dtype=float),
            "yaw_sin": yaw_sin,
            "yaw_cos": yaw_cos,
            "v": df_ep["v"].to_numpy(dtype=float),
            "w": df_ep["w"].to_numpy(dtype=float),
        })

        # Target state table; will be shifted by history_length to build next-state target
        target_state_df = pd.DataFrame({
            "x": df_ep["x"].to_numpy(dtype=float),
            "y": df_ep["y"].to_numpy(dtype=float),
            "yaw": yaw,  # used to produce yaw_sin_next/yaw_cos_next
        })

        # Accumulators for feature construction
        dfs_to_concat = []
        input_feature_cols = []

        # Build time windows using the original (future-aligned) shift(-i)
        # h0 = most recent, h1 = one step ahead in the CSV order, etc.
        for i in range(history_length):
            shifted_inputs = state_action_df.shift(-i).copy()
            current_cols = [
                f"x_h{i}", f"y_h{i}", f"yaw_sin_h{i}", f"yaw_cos_h{i}", f"v_h{i}", f"w_h{i}"
            ]
            shifted_inputs.columns = current_cols
            dfs_to_concat.append(shifted_inputs)
            input_feature_cols.extend(current_cols)

        # Target at t+history_length: next position + next yaw (as sin/cos)
        x_next = target_state_df["x"].shift(-history_length)
        y_next = target_state_df["y"].shift(-history_length)
        yaw_next = target_state_df["yaw"].shift(-history_length)
        yaw_sin_next = np.sin(yaw_next)
        yaw_cos_next = np.cos(yaw_next)

        target_df = pd.DataFrame({
            "x_next": x_next,
            "y_next": y_next,
            "yaw_sin_next": yaw_sin_next,
            "yaw_cos_next": yaw_cos_next,
        })
        target_feature_cols = ["x_next", "y_next", "yaw_sin_next", "yaw_cos_next"]

        # Concatenate inputs and target; drop rows with NaNs (trailing steps)
        dfs_to_concat.append(target_df)
        combined = pd.concat(dfs_to_concat, axis=1).dropna()

        # Keep only non-empty episodes
        if not combined.empty:
            inputs_by_ep[episode_id] = combined[input_feature_cols]
            targets_by_ep[episode_id] = combined[target_feature_cols]

    if not inputs_by_ep:
        print("Error: No valid data was generated. Check the CSV or history_length.")
        return None, None, None, None, None, None

    # ------------------------ Episode-based split ------------------------
    rng = np.random.RandomState(_episode_seed_from_csv(csv_path))  # deterministic shuffle
    episodes = list(inputs_by_ep.keys())
    rng.shuffle(episodes)

    n_val = max(1, int(round(val_split_ratio * len(episodes))))
    val_episodes = set(episodes[:n_val])
    train_episodes = [e for e in episodes if e not in val_episodes]

    def _concat_by_keys(dct, keys):
        # Concatenate frames for the given set of episode keys
        frames = [dct[k] for k in keys if k in dct]
        return pd.concat(frames) if frames else pd.DataFrame()

    # Build train/val dataframes (no episode overlap)
    train_inputs_df = _concat_by_keys(inputs_by_ep, train_episodes)
    train_targets_df = _concat_by_keys(targets_by_ep, train_episodes)
    val_inputs_df = _concat_by_keys(inputs_by_ep, val_episodes)
    val_targets_df = _concat_by_keys(targets_by_ep, val_episodes)

    if train_inputs_df.empty or val_inputs_df.empty:
        print("Error: episode-based split returned empty set(s). Adjust val_split_ratio.")
        return None, None, None, None, None, None

    # ------------------------ Scaling (fit on full dataset, as before) ------------------------
    if normalize_data:
        print("Normalizing data with StandardScaler (fit on all data: train+val)...")
        all_inputs_df = pd.concat([train_inputs_df, val_inputs_df])
        all_targets_df = pd.concat([train_targets_df, val_targets_df])

        input_scaler = StandardScaler()
        target_scaler = StandardScaler()
        input_scaler.fit(all_inputs_df)
        target_scaler.fit(all_targets_df)

        os.makedirs(scalers_dir, exist_ok=True)
        joblib.dump(input_scaler, os.path.join(scalers_dir, 'input_scaler.joblib'))
        joblib.dump(target_scaler, os.path.join(scalers_dir, 'target_scaler.joblib'))
        print(f"Scalers saved to: {scalers_dir}")

        train_inputs_np = input_scaler.transform(train_inputs_df)
        val_inputs_np = input_scaler.transform(val_inputs_df)
        train_targets_np = target_scaler.transform(train_targets_df)
        val_targets_np = target_scaler.transform(val_targets_df)
    else:
        print("Skipping normalization. Using raw values.")
        input_scaler = None
        target_scaler = None
        train_inputs_np = train_inputs_df.values
        val_inputs_np = val_inputs_df.values
        train_targets_np = train_targets_df.values
        val_targets_np = val_targets_df.values

    # ------------------------ Convert to tensors/datasets ------------------------
    train_inputs_tensor = torch.tensor(train_inputs_np, dtype=torch.float32)
    train_targets_tensor = torch.tensor(train_targets_np, dtype=torch.float32)
    val_inputs_tensor = torch.tensor(val_inputs_np, dtype=torch.float32)
    val_targets_tensor = torch.tensor(val_targets_np, dtype=torch.float32)

    train_data = TensorDataset(train_inputs_tensor, train_targets_tensor)
    val_data = TensorDataset(val_inputs_tensor, val_targets_tensor)

    input_dim = train_inputs_tensor.shape[1]
    output_dim = train_targets_tensor.shape[1]

    print(f"Detected dimensions: Input={input_dim}, Output={output_dim}")
    print(f"Episode split: {len(train_episodes)} train ep(s), {len(val_episodes)} val ep(s).")

    return train_data, val_data, input_scaler, target_scaler, input_dim, output_dim
