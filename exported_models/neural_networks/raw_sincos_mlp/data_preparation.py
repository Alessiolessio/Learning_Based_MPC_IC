#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
data_preparation.py

Reads the dataset CSV, builds single-step pairs (t -> t+1) per episode,
applies PARTIAL StandardScaler (only x, y, v, w - NOT sin/cos),
persists the scalers, and performs a random train/validation split.

Normalization approach:
- Input: [x, y, yaw] -> [x, y, yaw_sin, yaw_cos] -> [gauss_x, gauss_y, yaw_sin, yaw_cos, gauss_v, gauss_w]
- Output: [x_next, y_next, yaw_next] -> [gauss_x_next, gauss_y_next, yaw_sin_next, yaw_cos_next]
"""

import os
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, random_split


def _wrap_to_pi(a: np.ndarray) -> np.ndarray:
    """Wraps angles to (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def prepare_data(
    csv_path: str,
    scalers_dir: str = "trained_models",
    val_split_ratio: float = 0.2,
    normalize_data: bool = True,
):
    """
    Build (X_t, Y_{t+1}) pairs episode-by-episode and split into train/val.
    
    Input features: [x, y, yaw_sin, yaw_cos, v, w] (6 features)
    Target features: [x_next, y_next, yaw_sin_next, yaw_cos_next] (4 features)
    
    Only x, y, v, w are normalized with StandardScaler.
    sin/cos values are kept raw (already in [-1, 1]).

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

    all_inputs = []
    all_targets = []

    # -- Process each episode independently to keep time order intact --
    print(f"Processing {len(df['episode'].unique())} episodes...")
    for episode_id in df["episode"].unique():
        # Sort by 'step' so that shifts align to temporal order
        df_ep = df[df["episode"] == episode_id].sort_values("step")

        # Convert yaw to sin/cos
        yaw = _wrap_to_pi(df_ep["yaw"].to_numpy(dtype=float))
        yaw_sin = np.sin(yaw)
        yaw_cos = np.cos(yaw)

        # Build inputs at time t: [x, y, yaw_sin, yaw_cos, v, w]
        inputs_df = pd.DataFrame({
            "x": df_ep["x"].to_numpy(dtype=float),
            "y": df_ep["y"].to_numpy(dtype=float),
            "yaw_sin": yaw_sin,
            "yaw_cos": yaw_cos,
            "v": df_ep["v"].to_numpy(dtype=float),
            "w": df_ep["w"].to_numpy(dtype=float),
        })

        # Build targets at time t+1: [x_next, y_next, yaw_sin_next, yaw_cos_next]
        x_next = df_ep["x"].shift(-1).to_numpy(dtype=float)
        y_next = df_ep["y"].shift(-1).to_numpy(dtype=float)
        yaw_next = _wrap_to_pi(df_ep["yaw"].shift(-1).to_numpy(dtype=float))
        yaw_sin_next = np.sin(yaw_next)
        yaw_cos_next = np.cos(yaw_next)

        targets_df = pd.DataFrame({
            "x_next": x_next,
            "y_next": y_next,
            "yaw_sin_next": yaw_sin_next,
            "yaw_cos_next": yaw_cos_next,
        })

        # Concatenate and drop incomplete tail rows (NaN from shift)
        combined = pd.concat([inputs_df, targets_df], axis=1).dropna()

        if not combined.empty:
            all_inputs.append(combined[["x", "y", "yaw_sin", "yaw_cos", "v", "w"]])
            all_targets.append(combined[["x_next", "y_next", "yaw_sin_next", "yaw_cos_next"]])

    # -- Ensure some data was produced --
    if not all_inputs:
        print("Error: No valid data was generated. Check the CSV.")
        return None, None, None, None

    # -- Merge all episodes into final frames --
    final_inputs_df = pd.concat(all_inputs)
    final_targets_df = pd.concat(all_targets)

    # -- Column definitions for partial normalization --
    input_cols_all = ["x", "y", "yaw_sin", "yaw_cos", "v", "w"]
    input_cols_to_normalize = ["x", "y", "v", "w"]  # Gaussian normalization
    input_cols_sincos = ["yaw_sin", "yaw_cos"]  # Keep raw

    target_cols_all = ["x_next", "y_next", "yaw_sin_next", "yaw_cos_next"]
    target_cols_to_normalize = ["x_next", "y_next"]  # Gaussian normalization
    target_cols_sincos = ["yaw_sin_next", "yaw_cos_next"]  # Keep raw

    # -- Optional PARTIAL normalization with StandardScaler --
    if normalize_data:
        print("Normalizing data with PARTIAL StandardScaler...")
        print(f"  Input columns to normalize (Gaussian): {input_cols_to_normalize}")
        print(f"  Input columns kept raw (sin/cos): {input_cols_sincos}")
        print(f"  Target columns to normalize (Gaussian): {target_cols_to_normalize}")
        print(f"  Target columns kept raw (sin/cos): {target_cols_sincos}")

        # Fit scalers ONLY on the columns that need normalization
        input_scaler = StandardScaler()
        target_scaler = StandardScaler()
        input_scaler.fit(final_inputs_df[input_cols_to_normalize])
        target_scaler.fit(final_targets_df[target_cols_to_normalize])

        # Apply partial normalization
        def _apply_partial_normalization(df, scaler, cols_to_norm, cols_sincos, all_cols):
            """Apply scaler only to specific columns, keep others raw."""
            result = np.zeros((len(df), len(all_cols)), dtype=np.float32)
            # Transform the columns that need normalization
            normalized_part = scaler.transform(df[cols_to_norm])
            # Get raw sin/cos columns
            raw_sincos_part = df[cols_sincos].values
            
            # Reassemble in original column order
            for i, col in enumerate(all_cols):
                if col in cols_to_norm:
                    idx_in_norm = cols_to_norm.index(col)
                    result[:, i] = normalized_part[:, idx_in_norm]
                else:  # sin/cos column
                    idx_in_sincos = cols_sincos.index(col)
                    result[:, i] = raw_sincos_part[:, idx_in_sincos]
            return result

        inputs_np = _apply_partial_normalization(
            final_inputs_df, input_scaler, input_cols_to_normalize, input_cols_sincos, input_cols_all
        )
        targets_np = _apply_partial_normalization(
            final_targets_df, target_scaler, target_cols_to_normalize, target_cols_sincos, target_cols_all
        )

        # Persist scalers inside the run folder
        os.makedirs(scalers_dir, exist_ok=True)
        joblib.dump(input_scaler, os.path.join(scalers_dir, "input_scaler.joblib"))
        joblib.dump(target_scaler, os.path.join(scalers_dir, "target_scaler.joblib"))
        
        # Also save column info for inference
        col_info = {
            'input_cols_all': input_cols_all,
            'input_cols_to_normalize': input_cols_to_normalize,
            'input_cols_sincos': input_cols_sincos,
            'target_cols_all': target_cols_all,
            'target_cols_to_normalize': target_cols_to_normalize,
            'target_cols_sincos': target_cols_sincos,
        }
        joblib.dump(col_info, os.path.join(scalers_dir, "column_info.joblib"))
        print(f"Scalers and column_info saved to: {scalers_dir}")
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
    print(f"  Input shape: {inputs_tensor.shape} (6 features: x, y, yaw_sin, yaw_cos, v, w)")
    print(f"  Target shape: {targets_tensor.shape} (4 features: x_next, y_next, yaw_sin_next, yaw_cos_next)")
    print(f"Splitting the dataset (validation ratio: {val_split_ratio})...")

    # -- Deterministic split seed based on known CSV paths --
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
