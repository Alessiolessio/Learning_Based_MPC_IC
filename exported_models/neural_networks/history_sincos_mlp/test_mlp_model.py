#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_mlp_model.py

Loads a trained MLP (and its scalers/config) and compares rollouts:
real trajectory vs. unicycle vs. MLP with history buffer.

Notes:
- Reads config_snapshot.yaml from the model folder to reproduce exactly
  the history_length, hidden_layers, and dropout used during training.
- Model output is 4D: [x_next, y_next, yaw_sin_next, yaw_cos_next].
"""

import os
import math
import yaml
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
from collections import deque

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt

# ------------------------- USER-PROVIDED PATHS -------------------------
MODEL_BASE_PATH = "/home/nexus/VQ_PMCnmpc/VQ_PMC/exported_models/neural_networks/history_sincos_mlp/trained_models/model_hist_5_epoch_400_batch_64_lr_1e-05_vs_30_hl_512_512_512_512_512"
CSV_PATH = "/home/nexus/VQ_PMCnmpc/VQ_PMC/logs/datasets/dataset_nmpc_test.csv"
DT = 0.02  # integration step (s)
# ----------------------------------------------------------------------

MODEL_PATH = os.path.join(MODEL_BASE_PATH, "mlp_dynamics.pth")
INPUT_SCALER_PATH = os.path.join(MODEL_BASE_PATH, "input_scaler.joblib")
TARGET_SCALER_PATH = os.path.join(MODEL_BASE_PATH, "target_scaler.joblib")
COLUMN_INFO_PATH = os.path.join(MODEL_BASE_PATH, "column_info.joblib")
CONFIG_SNAPSHOT_YML = os.path.join(MODEL_BASE_PATH, "config_snapshot.yaml")
OUT_DIR = os.path.join(MODEL_BASE_PATH, "tests")
os.makedirs(OUT_DIR, exist_ok=True)

# Columns required from the CSV (consistency check)
REQ_COLS = [
    "env", "episode", "step", "timestamp", "vx", "vy", "wz", "x", "y",
    "qw", "qx", "qy", "qz", "yaw", "v", "w"
]

# ------------------------------- Model --------------------------------

class MLP(nn.Module):
    """Simple MLP to match the architecture used during training."""
    def __init__(self, input_dim, output_dim, hidden_layers, p_dropout: float = 0.0):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_layers:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            if p_dropout and p_dropout > 0:
                layers.append(nn.Dropout(p=p_dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Standard forward pass
        return self.model(x)

# ------------------------------ Utils ---------------------------------

def wrap_to_pi(a: float) -> float:
    # Wraps a single angle to (-pi, pi]
    return (a + np.pi) % (2 * np.pi) - np.pi

def rollout_unicycle(x0, y0, th0, v_arr, w_arr, dt):
    """Simple unicycle forward integration for comparison plots."""
    n = len(v_arr)
    x_pred, y_pred = np.empty(n), np.empty(n)
    x, y, th = float(x0), float(y0), float(th0)
    x_pred[0], y_pred[0] = x, y
    for k in range(n - 1):
        v, w = float(v_arr[k]), float(w_arr[k])
        x += v * math.cos(th) * dt
        y += v * math.sin(th) * dt
        th = wrap_to_pi(th + w * dt)
        x_pred[k + 1], y_pred[k + 1] = x, y
    return x_pred, y_pred

def rollout_mlp(
    x0, y0, th0,
    v_cmd, w_cmd,
    model, in_scaler, tgt_scaler, col_info, device,
    history_length, input_features,
    debug_steps=5  # Print debug info for N steps in the middle
):
    """
    Performs MLP rollout using a history buffer.
    Input (per step): [x, y, yaw_sin, yaw_cos, v, w]
    Output:          [x_next, y_next, yaw_sin_next, yaw_cos_next]
    
    NOTE: Only x, y, v, w are normalized with StandardScaler.
          sin/cos columns are kept as raw values.
    """
    n = len(v_cmd)
    x_pred, y_pred = np.empty(n), np.empty(n)
    x, y, th = float(x0), float(y0), float(th0)
    x_pred[0], y_pred[0] = x, y

    # Initialize history deque with repeated initial vector
    q = deque(maxlen=history_length)
    ysin0, ycos0 = np.sin(th), np.cos(th)
    s0 = np.array([x, y, ysin0, ycos0, v_cmd[0], w_cmd[0]], dtype=float)
    for _ in range(history_length):
        q.append(s0)

    # Extract column info for partial normalization
    input_cols_all = col_info['input_cols_all']
    input_cols_to_normalize = col_info['input_cols_to_normalize']
    input_cols_sincos = col_info['input_cols_sincos']
    target_cols_all = col_info['target_cols_all']
    target_cols_to_normalize = col_info['target_cols_to_normalize']
    target_cols_sincos = col_info['target_cols_sincos']

    # Debug: Print scaler info once
    print("\n" + "="*80)
    print("DEBUG: SCALER INFORMATION (PARTIAL NORMALIZATION)")
    print("="*80)
    print(f"Input scaler feature names (normalized): {in_scaler.feature_names_in_.tolist()}")
    print(f"Input scaler mean: {in_scaler.mean_}")
    print(f"Input scaler scale (std): {in_scaler.scale_}")
    print(f"Input columns kept RAW (sin/cos): {input_cols_sincos}")
    print(f"\nTarget scaler feature names (normalized): {tgt_scaler.feature_names_in_.tolist()}")
    print(f"Target scaler mean: {tgt_scaler.mean_}")
    print(f"Target scaler scale (std): {tgt_scaler.scale_}")
    print(f"Target columns kept RAW (sin/cos): {target_cols_sincos}")
    print("="*80 + "\n")

    # Calculate middle range for debug prints
    mid_start = max(0, (n - 1) // 2 - debug_steps // 2)
    mid_end = mid_start + debug_steps
    debug_range = set(range(mid_start, mid_end))
    print(f"[DEBUG] Will print detailed info for steps {mid_start} to {mid_end-1} (middle of trajectory)\n")

    def _apply_partial_norm_input(raw_input_df):
        """Apply scaler only to x, y, v, w columns; keep sin/cos raw."""
        result = np.zeros((1, len(input_cols_all)), dtype=np.float32)
        # Normalize only the columns that need it - use DataFrame to avoid sklearn warning
        norm_df = raw_input_df[input_cols_to_normalize]
        normalized_part = in_scaler.transform(norm_df)
        # Keep sin/cos raw
        raw_sincos = raw_input_df[input_cols_sincos].values
        
        # Reassemble in original column order
        for i, col in enumerate(input_cols_all):
            if col in input_cols_to_normalize:
                idx_in_norm = input_cols_to_normalize.index(col)
                result[0, i] = normalized_part[0, idx_in_norm]
            else:  # sin/cos column
                idx_in_sincos = input_cols_sincos.index(col)
                result[0, i] = raw_sincos[0, idx_in_sincos]
        return result

    def _apply_partial_denorm_output(model_output_np):
        """Denormalize only x, y columns; keep sin/cos raw from model output."""
        result = np.zeros(len(target_cols_all), dtype=np.float32)
        
        # Extract the parts that were normalized (x_next, y_next)
        norm_indices = [target_cols_all.index(c) for c in target_cols_to_normalize]
        sincos_indices = [target_cols_all.index(c) for c in target_cols_sincos]
        
        # Get normalized values and denormalize them - use DataFrame to avoid sklearn warning
        norm_values = model_output_np[0, norm_indices].reshape(1, -1)
        norm_df = pd.DataFrame(norm_values, columns=target_cols_to_normalize)
        denorm_values = tgt_scaler.inverse_transform(norm_df)[0]
        
        # Get sin/cos values directly (they were not normalized)
        sincos_values = model_output_np[0, sincos_indices]
        
        # Reassemble in original order
        for i, col in enumerate(target_cols_all):
            if col in target_cols_to_normalize:
                idx_in_norm = target_cols_to_normalize.index(col)
                result[i] = denorm_values[idx_in_norm]
            else:  # sin/cos column
                idx_in_sincos = target_cols_sincos.index(col)
                result[i] = sincos_values[idx_in_sincos]
        return result

    for k in range(n - 1):
        # Current state-action vector appended to history (most recent first)
        ysin, ycos = np.sin(th), np.cos(th)
        s = np.array([x, y, ysin, ycos, v_cmd[k], w_cmd[k]], dtype=float)
        q.append(s)

        # Flatten history as [h0, h1, ..., hH-1]
        flat = []
        for vec in reversed(q):
            flat.extend(vec)
        flat = np.array([flat], dtype=np.float32)

        # Use scaler (feature order controlled by scaler.feature_names_in_)
        df_in = pd.DataFrame(flat, columns=input_features)
        
        # Debug prints for middle steps
        if k in debug_range:
            print(f"\n{'='*80}")
            print(f"DEBUG STEP {k}: BEFORE NORMALIZATION (RAW INPUT)")
            print(f"{'='*80}")
            print(f"Current state: x={x:.6f}, y={y:.6f}, yaw={th:.6f} (sin={ysin:.6f}, cos={ycos:.6f})")
            print(f"Command: v={v_cmd[k]:.6f}, w={w_cmd[k]:.6f}")
            print(f"Flattened input shape: {flat.shape}")
            print(f"Flattened input (raw):\n{flat}")
            print(f"DataFrame columns: {df_in.columns.tolist()}")
        
        # Apply PARTIAL normalization (only x, y, v, w)
        x_scaled = _apply_partial_norm_input(df_in)
        
        if k in debug_range:
            print(f"\n{'='*80}")
            print(f"DEBUG STEP {k}: AFTER PARTIAL NORMALIZATION")
            print(f"{'='*80}")
            print(f"Scaled input shape: {x_scaled.shape}")
            print(f"Scaled input:\n{x_scaled}")
            print(f"Note: sin/cos columns should be in [-1, 1] range (NOT normalized)")
            print(f"Note: x, y, v, w columns should be ~0 mean, ~1 std (normalized)")

        # Predict scaled output, then inverse-transform
        with torch.no_grad():
            input_tensor = torch.tensor(x_scaled, dtype=torch.float32, device=device)
            y_scaled = model(input_tensor)
            
        if k in debug_range:
            print(f"\n{'='*80}")
            print(f"DEBUG STEP {k}: MODEL OUTPUT (PARTIALLY NORMALIZED)")
            print(f"{'='*80}")
            print(f"Model output shape: {y_scaled.shape}")
            print(f"Model output:\n{y_scaled.cpu().numpy()}")
            print(f"Note: x_next, y_next are normalized; sin/cos are raw")
            
        # Apply PARTIAL denormalization (only x, y)
        y_next = _apply_partial_denorm_output(y_scaled.cpu().numpy())
        
        if k in debug_range:
            print(f"\n{'='*80}")
            print(f"DEBUG STEP {k}: AFTER PARTIAL DENORMALIZATION (RAW OUTPUT)")
            print(f"{'='*80}")
            print(f"Denormalized output: {y_next}")
            print(f"  x_next={y_next[0]:.6f}, y_next={y_next[1]:.6f}")
            print(f"  yaw_sin_next={y_next[2]:.6f}, yaw_cos_next={y_next[3]:.6f}")
            # Sanity check: sin^2 + cos^2 should be ~1
            sin_cos_norm = y_next[2]**2 + y_next[3]**2
            print(f"  sin^2 + cos^2 = {sin_cos_norm:.6f} (should be ~1.0)")
            
        x, y, ysin_next, ycos_next = float(y_next[0]), float(y_next[1]), float(y_next[2]), float(y_next[3])
        th = wrap_to_pi(math.atan2(ysin_next, ycos_next))
        
        if k in debug_range:
            print(f"  Recovered yaw (atan2): {th:.6f} rad ({math.degrees(th):.2f} deg)")
            print(f"{'='*80}\n")

        x_pred[k + 1], y_pred[k + 1] = x, y

    return x_pred, y_pred

# ------------------------------- Main ---------------------------------

def main():
    # Basic path checks (fail early with a clear message)
    for p in (MODEL_BASE_PATH, MODEL_PATH, INPUT_SCALER_PATH, TARGET_SCALER_PATH, COLUMN_INFO_PATH, CONFIG_SNAPSHOT_YML, CSV_PATH):
        if not os.path.exists(p if p != MODEL_BASE_PATH else MODEL_BASE_PATH):
            raise FileNotFoundError(f"File/dir not found: {p}")

    # Read config snapshot to recover history_length/architecture/dropout
    with open(CONFIG_SNAPSHOT_YML, "r") as f:
        cfg = yaml.safe_load(f)
    hist_len = cfg["training_params"].get("history_length", 1)
    hid_layers = cfg["model_params"]["hidden_layers"]
    p_drop = cfg["model_params"].get("p_dropout", 0.0)

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] From snapshot: history_length={hist_len}, hidden_layers={hid_layers}, p_dropout={p_drop}")

    # Load scalers and column info (for partial normalization)
    in_scaler = joblib.load(INPUT_SCALER_PATH)
    tgt_scaler = joblib.load(TARGET_SCALER_PATH)
    col_info = joblib.load(COLUMN_INFO_PATH)
    
    input_features = col_info['input_cols_all']
    input_dim = len(input_features)
    output_dim = len(col_info['target_cols_all'])
    
    print(f"[INFO] input_dim={input_dim}, output_dim={output_dim}")
    print(f"[INFO] input_features={input_features}")
    print(f"[INFO] Columns normalized (Gaussian): inputs={col_info['input_cols_to_normalize']}, targets={col_info['target_cols_to_normalize']}")
    print(f"[INFO] Columns kept raw (sin/cos): inputs={col_info['input_cols_sincos']}, targets={col_info['target_cols_sincos']}")

    # Build model exactly as trained (to match state_dict)
    model = MLP(input_dim=input_dim, output_dim=output_dim, hidden_layers=hid_layers, p_dropout=p_drop).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # Load dataset and basic sanity checks
    df = pd.read_csv(CSV_PATH)
    miss = [c for c in REQ_COLS if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns in CSV: {miss}")
    df.dropna(subset=["x", "y", "yaw", "v", "w"], inplace=True)
    df = df.sort_values(["episode", "env", "step"]).reset_index(drop=True)

    episodes = df["episode"].unique()
    print(f"[INFO] Episodes: {len(episodes)}")

    for ep in episodes:  # Process all episodes
        df_ep = df[df["episode"] == ep].copy()
        
        # Debug: Print ground truth data for first episode
        print("\n" + "#"*80)
        print(f"DEBUG: GROUND TRUTH DATA FOR EPISODE {ep}")
        print("#"*80)
        print(f"Episode shape: {df_ep.shape}")
        print(f"Columns: {df_ep.columns.tolist()}")
        print(f"\nFirst 5 rows of relevant columns:")
        print(df_ep[["step", "x", "y", "yaw", "v", "w"]].head())
        print(f"\nData statistics:")
        print(df_ep[["x", "y", "yaw", "v", "w"]].describe())
        print("#"*80 + "\n")

        # Create figure with 3 panels: traj, per-step error, cumulative error
        fig, (ax_traj, ax_err, ax_acc) = plt.subplots(3, 1, figsize=(10, 22), gridspec_kw={'height_ratios': [3, 1, 1]})
        plotted = False

        for env_id, df_env in df_ep.groupby("env"):
            df_env = df_env.sort_values("step")
            if len(df_env) < 2:
                continue
            plotted = True

            steps = np.arange(len(df_env))
            x_real = df_env["x"].to_numpy(float)
            y_real = df_env["y"].to_numpy(float)
            v_cmd = df_env["v"].to_numpy(float)
            w_cmd = df_env["w"].to_numpy(float)
            x0, y0 = float(x_real[0]), float(y_real[0])
            yaw0 = float(df_env["yaw"].iloc[0])

            # Baseline (unicycle) rollout
            x_u, y_u = rollout_unicycle(x0, y0, yaw0, v_cmd, w_cmd, DT)

            # MLP rollout (history buffer + scalers)
            x_m, y_m = rollout_mlp(
                x0, y0, yaw0, v_cmd, w_cmd,
                model, in_scaler, tgt_scaler, col_info, device, hist_len, input_features
            )

            # Per-step Euclidean error and cumulative sums
            err_u = np.hypot(x_real - x_u, y_real - y_u)
            err_m = np.hypot(x_real - x_m, y_real - y_m)
            acc_u = np.cumsum(err_u)
            acc_m = np.cumsum(err_m)

            # Plots: trajectory + error panels
            ax_traj.plot(x_real, y_real, label=f"env {env_id} — Real", linewidth=2.5, alpha=0.85)
            ax_traj.plot(x_u, y_u, "--", label=f"env {env_id} — Unicycle", linewidth=2)
            ax_traj.plot(x_m, y_m, ":", label=f"env {env_id} — MLP", linewidth=2)

            ax_err.plot(steps, err_u, "--", label=f"env {env_id} — Unicycle Err", linewidth=2)
            ax_err.plot(steps, err_m, ":", label=f"env {env_id} — MLP Err", linewidth=2)

            ax_acc.plot(steps, acc_u, "--", label=f"env {env_id} — Unicycle Acc. Err", linewidth=2)
            ax_acc.plot(steps, acc_m, ":", label=f"env {env_id} — MLP Acc. Err", linewidth=2)

        # Finalize figure if something was plotted
        if plotted:
            ax_traj.set_aspect("equal")
            ax_traj.grid(True, alpha=0.3)
            ax_traj.set_xlabel("X (m)")
            ax_traj.set_ylabel("Y (m)")
            ax_traj.set_title(f"Trajectory Comparison (hist={hist_len})")
            ax_traj.legend(loc="best")

            ax_err.grid(True, alpha=0.3)
            ax_err.set_xlabel("Step (k)")
            ax_err.set_ylabel("Euclidean Error (m)")
            ax_err.set_title("Prediction Error (Real vs. Model)")
            ax_err.legend(loc="best")
            ax_err.set_xlim(left=0)
            ax_err.set_ylim(bottom=0)

            ax_acc.grid(True, alpha=0.3)
            ax_acc.set_xlabel("Step (k)")
            ax_acc.set_ylabel("Cumulative Error (m)")
            ax_acc.set_title("Cumulative Prediction Error")
            ax_acc.legend(loc="best")
            ax_acc.set_xlim(left=0)
            ax_acc.set_ylim(bottom=0)

            fig.suptitle(f"Episode {int(ep)} — dt={DT}s", fontsize=16)
            out_path = os.path.join(OUT_DIR, f"ep_{int(ep):05d}_compare.png")
            fig.tight_layout(rect=[0, 0.03, 1, 0.96])
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"[OK] Saved: {out_path}")
        else:
            print(f"[WARN] No valid data to plot for episode {ep}.")
        plt.close(fig)

    print("\n[INFO] Test finished.")

if __name__ == "__main__":
    main()
