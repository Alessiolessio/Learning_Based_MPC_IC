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
    # NOTE: We only normalize x, y, v, w columns with StandardScaler.
    #       sin/cos columns remain as raw values (already in [-1, 1] range).
    if normalize_data:
        print("Normalizing data with StandardScaler (ONLY for x, y, v, w columns)...")
        print("sin/cos columns will NOT be normalized (kept as raw values).")
        
        all_inputs_df = pd.concat([train_inputs_df, val_inputs_df])
        all_targets_df = pd.concat([train_targets_df, val_targets_df])

        # Identify columns to normalize vs keep raw
        input_cols_all = all_inputs_df.columns.tolist()
        target_cols_all = all_targets_df.columns.tolist()
        
        # Columns containing sin/cos should NOT be normalized
        input_cols_to_normalize = [c for c in input_cols_all if 'sin' not in c and 'cos' not in c]
        input_cols_sincos = [c for c in input_cols_all if 'sin' in c or 'cos' in c]
        
        target_cols_to_normalize = [c for c in target_cols_all if 'sin' not in c and 'cos' not in c]
        target_cols_sincos = [c for c in target_cols_all if 'sin' in c or 'cos' in c]
        
        print(f"\nInput columns to normalize (Gaussian): {input_cols_to_normalize}")
        print(f"Input columns kept raw (sin/cos): {input_cols_sincos}")
        print(f"Target columns to normalize (Gaussian): {target_cols_to_normalize}")
        print(f"Target columns kept raw (sin/cos): {target_cols_sincos}")

        # Fit scalers ONLY on the columns that need normalization
        input_scaler = StandardScaler()
        target_scaler = StandardScaler()
        input_scaler.fit(all_inputs_df[input_cols_to_normalize])
        target_scaler.fit(all_targets_df[target_cols_to_normalize])

        os.makedirs(scalers_dir, exist_ok=True)
        joblib.dump(input_scaler, os.path.join(scalers_dir, 'input_scaler.joblib'))
        joblib.dump(target_scaler, os.path.join(scalers_dir, 'target_scaler.joblib'))
        
        # Also save the column order info for inference
        col_info = {
            'input_cols_all': input_cols_all,
            'input_cols_to_normalize': input_cols_to_normalize,
            'input_cols_sincos': input_cols_sincos,
            'target_cols_all': target_cols_all,
            'target_cols_to_normalize': target_cols_to_normalize,
            'target_cols_sincos': target_cols_sincos,
        }
        joblib.dump(col_info, os.path.join(scalers_dir, 'column_info.joblib'))
        print(f"Scalers and column_info saved to: {scalers_dir}")

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

        # ===================== DEBUG: BEFORE NORMALIZATION =====================
        print("\n" + "="*80)
        print("DEBUG: RAW DATA STATISTICS (BEFORE NORMALIZATION)")
        print("="*80)
        
        # Pick 5 samples from the middle of the dataset for detailed inspection
        n_samples = len(all_inputs_df)
        mid_start = max(0, n_samples // 2 - 2)
        mid_indices = list(range(mid_start, min(mid_start + 5, n_samples)))
        
        print(f"\n┌{'─'*78}┐")
        print(f"│{'INPUT FEATURES':^78}│")
        print(f"│{'Total samples: ' + str(n_samples):^78}│")
        print(f"└{'─'*78}┘")
        print(f"\nFeature names ({len(all_inputs_df.columns)} features):")
        # Print feature names in a grid format
        cols = all_inputs_df.columns.tolist()
        for i in range(0, len(cols), 4):
            row = cols[i:i+4]
            print("  " + "  |  ".join(f"{c:20s}" for c in row))
        
        print(f"\n{'─'*80}")
        print("Input Statistics (per feature):")
        print("─"*80)
        stats_df = all_inputs_df.describe().T
        stats_df = stats_df[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
        print(stats_df.to_string())
        
        print(f"\n{'─'*80}")
        print(f"5 SAMPLE INPUTS FROM MIDDLE (indices {mid_indices})")
        print("─"*80)
        sample_inputs = all_inputs_df.iloc[mid_indices].T  # Transpose for better readability
        print(sample_inputs.to_string())
        
        print(f"\n\n┌{'─'*78}┐")
        print(f"│{'TARGET FEATURES':^78}│")
        print(f"└{'─'*78}┘")
        print(f"\nFeature names: {all_targets_df.columns.tolist()}")
        
        print(f"\n{'─'*80}")
        print("Target Statistics (per feature):")
        print("─"*80)
        stats_tgt = all_targets_df.describe().T
        stats_tgt = stats_tgt[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
        print(stats_tgt.to_string())
        
        print(f"\n{'─'*80}")
        print(f"5 SAMPLE TARGETS FROM MIDDLE (indices {mid_indices})")
        print("─"*80)
        sample_targets = all_targets_df.iloc[mid_indices].T  # Transpose for better readability
        print(sample_targets.to_string())
        
        # ===================== DEBUG: SCALER PARAMETERS =====================
        print("\n\n" + "="*80)
        print("DEBUG: SCALER PARAMETERS (FITTED)")
        print("="*80)
        
        print(f"\n┌{'─'*78}┐")
        print(f"│{'INPUT SCALER (only x, y, v, w columns)':^78}│")
        print(f"└{'─'*78}┘")
        print(f"\nFeature names: {input_scaler.feature_names_in_.tolist()}")
        print(f"\n{'Feature':<25} {'Mean':>15} {'Std (Scale)':>15} {'Variance':>15}")
        print("─"*70)
        for i, feat in enumerate(input_scaler.feature_names_in_):
            print(f"{feat:<25} {input_scaler.mean_[i]:>15.6f} {input_scaler.scale_[i]:>15.6f} {input_scaler.var_[i]:>15.6f}")
        
        print(f"\n┌{'─'*78}┐")
        print(f"│{'TARGET SCALER (only x, y columns)':^78}│")
        print(f"└{'─'*78}┘")
        print(f"\nFeature names: {target_scaler.feature_names_in_.tolist()}")
        print(f"\n{'Feature':<25} {'Mean':>15} {'Std (Scale)':>15} {'Variance':>15}")
        print("─"*70)
        for i, feat in enumerate(target_scaler.feature_names_in_):
            print(f"{feat:<25} {target_scaler.mean_[i]:>15.6f} {target_scaler.scale_[i]:>15.6f} {target_scaler.var_[i]:>15.6f}")
        
        print(f"\n┌{'─'*78}┐")
        print(f"│{'SIN/COS COLUMNS (NOT normalized, kept raw)':^78}│")
        print(f"└{'─'*78}┘")
        print(f"Input sin/cos cols: {input_cols_sincos}")
        print(f"Target sin/cos cols: {target_cols_sincos}")

        # Apply partial normalization
        train_inputs_np = _apply_partial_normalization(
            train_inputs_df, input_scaler, input_cols_to_normalize, input_cols_sincos, input_cols_all)
        val_inputs_np = _apply_partial_normalization(
            val_inputs_df, input_scaler, input_cols_to_normalize, input_cols_sincos, input_cols_all)
        train_targets_np = _apply_partial_normalization(
            train_targets_df, target_scaler, target_cols_to_normalize, target_cols_sincos, target_cols_all)
        val_targets_np = _apply_partial_normalization(
            val_targets_df, target_scaler, target_cols_to_normalize, target_cols_sincos, target_cols_all)
        
        # ===================== DEBUG: AFTER NORMALIZATION =====================
        print("\n\n" + "="*80)
        print("DEBUG: NORMALIZED DATA STATISTICS (AFTER PARTIAL NORMALIZATION)")
        print("="*80)
        
        print(f"\n┌{'─'*78}┐")
        print(f"│{'NORMALIZED TRAIN INPUTS':^78}│")
        print(f"│{'Shape: ' + str(train_inputs_np.shape):^78}│")
        print(f"└{'─'*78}┘")
        
        print(f"\n{'Feature':<25} {'Min':>12} {'Max':>12} {'Mean':>12} {'Std':>12} {'Normalized?':>12}")
        print("─"*85)
        for i, feat in enumerate(input_cols_all):
            is_norm = "YES" if feat in input_cols_to_normalize else "NO (sin/cos)"
            print(f"{feat:<25} {train_inputs_np[:, i].min():>12.4f} {train_inputs_np[:, i].max():>12.4f} {train_inputs_np[:, i].mean():>12.4f} {train_inputs_np[:, i].std():>12.4f} {is_norm:>12}")
        
        print(f"\n┌{'─'*78}┐")
        print(f"│{'NORMALIZED TRAIN TARGETS':^78}│")
        print(f"│{'Shape: ' + str(train_targets_np.shape):^78}│")
        print(f"└{'─'*78}┘")
        
        print(f"\n{'Feature':<25} {'Min':>12} {'Max':>12} {'Mean':>12} {'Std':>12} {'Normalized?':>12}")
        print("─"*85)
        for i, feat in enumerate(target_cols_all):
            is_norm = "YES" if feat in target_cols_to_normalize else "NO (sin/cos)"
            print(f"{feat:<25} {train_targets_np[:, i].min():>12.4f} {train_targets_np[:, i].max():>12.4f} {train_targets_np[:, i].mean():>12.4f} {train_targets_np[:, i].std():>12.4f} {is_norm:>12}")
        
        # Show the same 5 samples after normalization
        all_inputs_np = _apply_partial_normalization(
            all_inputs_df, input_scaler, input_cols_to_normalize, input_cols_sincos, input_cols_all)
        all_targets_np = _apply_partial_normalization(
            all_targets_df, target_scaler, target_cols_to_normalize, target_cols_sincos, target_cols_all)
        
        print(f"\n{'─'*80}")
        print(f"5 SAMPLE INPUTS FROM MIDDLE - AFTER NORMALIZATION (indices {mid_indices})")
        print("─"*80)
        print(f"\n{'Feature':<20}", end="")
        for idx in mid_indices:
            print(f"{'Sample '+str(idx):>14}", end="")
        print()
        print("─"*90)
        for i, feat in enumerate(input_cols_all):
            print(f"{feat:<20}", end="")
            for idx in mid_indices:
                print(f"{all_inputs_np[idx, i]:>14.4f}", end="")
            print()
        
        print(f"\n{'─'*80}")
        print(f"5 SAMPLE TARGETS FROM MIDDLE - AFTER NORMALIZATION (indices {mid_indices})")
        print("─"*80)
        print(f"\n{'Feature':<20}", end="")
        for idx in mid_indices:
            print(f"{'Sample '+str(idx):>14}", end="")
        print()
        print("─"*90)
        for i, feat in enumerate(target_cols_all):
            print(f"{feat:<20}", end="")
            for idx in mid_indices:
                print(f"{all_targets_np[idx, i]:>14.4f}", end="")
            print()
        
        # ===================== DEBUG: SANITY CHECKS =====================
        print("\n\n" + "="*80)
        print("DEBUG: SANITY CHECKS")
        print("="*80)
        
        # Check sin/cos columns - sin^2 + cos^2 should be ~1 BEFORE normalization
        print(f"\n┌{'─'*78}┐")
        print(f"│{'Checking sin² + cos² for yaw (BEFORE normalization)':^78}│")
        print(f"│{'Expected: ≈ 1.0 for all':^78}│")
        print(f"└{'─'*78}┘")
        
        print(f"\n{'History Step':<20} {'Min':>15} {'Max':>15} {'Mean':>15}")
        print("─"*65)
        for i in range(history_length):
            sin_col = f"yaw_sin_h{i}"
            cos_col = f"yaw_cos_h{i}"
            if sin_col in all_inputs_df.columns and cos_col in all_inputs_df.columns:
                sin_vals = all_inputs_df[sin_col].values
                cos_vals = all_inputs_df[cos_col].values
                norm_check = sin_vals**2 + cos_vals**2
                print(f"h{i:<19} {norm_check.min():>15.6f} {norm_check.max():>15.6f} {norm_check.mean():>15.6f}")
        
        # Target sin/cos check
        sin_next = all_targets_df["yaw_sin_next"].values
        cos_next = all_targets_df["yaw_cos_next"].values
        norm_check_tgt = sin_next**2 + cos_next**2
        print(f"{'target':<20} {norm_check_tgt.min():>15.6f} {norm_check_tgt.max():>15.6f} {norm_check_tgt.mean():>15.6f}")
        
        print("\n" + "="*80 + "\n")
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
