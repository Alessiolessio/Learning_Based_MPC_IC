#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_mlp_model.py

Compares real trajectory with:
- Unicycle baseline rollout
- MLP rollout (using the exact architecture from config_snapshot.yaml)

Loads model/scalers from MODEL_BASE_PATH and produces per-episode plots.

Note: This MLP uses sin/cos representation for yaw with PARTIAL normalization:
- Input: [x, y, yaw_sin, yaw_cos, v, w] (6 features)
- Output: [x_next, y_next, yaw_sin_next, yaw_cos_next] (4 features)
- Only x, y, v, w are normalized with StandardScaler
- sin/cos values are kept raw (already in [-1, 1])
"""

import os
import math
import yaml
import joblib
import torch
import numpy as np
import pandas as pd
import torch.nn as nn

# Headless plotting backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------- USER CONFIG ---------------------
CSV_PATH = "/home/nexus/VQ_PMCnmpc/VQ_PMC/logs/datasets/datasets_raw/dataset_nmpc_test_raw.csv"
MODEL_BASE_PATH = "/home/nexus/VQ_PMCnmpc/VQ_PMC/exported_models/neural_networks/raw_sincos_mlp/trained_models/model_epoch_150_batch_128_lr_1e-05_vs_30_hl_64"
DT = 0.02  # time step for unicycle baseline
# Action scaling constants for unicycle model (must match dataset preprocessing)
KV = 0.2839
KW = 0.13
# -------------------------------------------------------

# Derived artifact paths
MODEL_PATH = os.path.join(MODEL_BASE_PATH, "mlp_dynamics.pth")
INPUT_SCALER_PATH = os.path.join(MODEL_BASE_PATH, "input_scaler.joblib")
TARGET_SCALER_PATH = os.path.join(MODEL_BASE_PATH, "target_scaler.joblib")
COLUMN_INFO_PATH = os.path.join(MODEL_BASE_PATH, "column_info.joblib")
CONFIG_SNAPSHOT_YML = os.path.join(MODEL_BASE_PATH, "config_snapshot.yaml")
OUT_DIR = os.path.join(MODEL_BASE_PATH, "tests")
os.makedirs(OUT_DIR, exist_ok=True)

# Required CSV columns
REQ_COLS = [
    "env", "episode", "step", "timestamp",
    "vx", "vy", "wz", "x", "y",
    "qw", "qx", "qy", "qz", "yaw", "v", "w",
]


class MLP(nn.Module):
    """Plain MLP matching the training-time architecture."""
    def __init__(self, input_dim, output_dim, hidden_layers, p_dropout=0.0):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_layers:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            if p_dropout and p_dropout > 0.0:
                layers.append(nn.Dropout(p=p_dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def wrap_to_pi(angle: float) -> float:
    """Wrap angle to (-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def rollout_unicycle(x0, y0, th0, v_arr, w_arr, dt):
    """Simple forward integration of the unicycle model."""
    n = len(v_arr)
    x_pred = np.empty(n, dtype=float)
    y_pred = np.empty(n, dtype=float)

    x, y, th = float(x0), float(y0), float(th0)
    x_pred[0], y_pred[0] = x, y

    for k in range(n - 1):
        v, w = float(v_arr[k]) * KV, float(w_arr[k]) * KW
        x = x + v * math.cos(th) * dt
        y = y + v * math.sin(th) * dt
        th = wrap_to_pi(th + w * dt)
        x_pred[k + 1], y_pred[k + 1] = x, y

    return x_pred, y_pred


def rollout_mlp(x0, y0, th0, v_cmd_arr, w_cmd_arr,
                model, input_scaler, target_scaler, col_info, device):
    """
    Perform one-step MLP predictions in sequence with PARTIAL normalization:
      input:  [x, y, yaw_sin, yaw_cos, v, w] (6 features)
      output: [x_next, y_next, yaw_sin_next, yaw_cos_next] (4 features)
    
    Only x, y, v, w are normalized with StandardScaler.
    sin/cos values are kept raw.
    """
    n = len(v_cmd_arr)
    x_pred = np.empty(n, dtype=float)
    y_pred = np.empty(n, dtype=float)

    x, y, th = float(x0), float(y0), float(th0)
    x_pred[0], y_pred[0] = x, y

    # Extract column info for partial normalization
    input_cols_all = col_info['input_cols_all']
    input_cols_to_normalize = col_info['input_cols_to_normalize']
    input_cols_sincos = col_info['input_cols_sincos']
    target_cols_all = col_info['target_cols_all']
    target_cols_to_normalize = col_info['target_cols_to_normalize']
    target_cols_sincos = col_info['target_cols_sincos']

    def _apply_partial_norm_input(raw_values):
        """Apply scaler only to x, y, v, w columns; keep sin/cos raw."""
        # raw_values: [x, y, yaw_sin, yaw_cos, v, w]
        result = np.zeros((1, len(input_cols_all)), dtype=np.float32)
        
        # Extract values to normalize: x, y, v, w
        norm_values = np.array([[raw_values[0], raw_values[1], raw_values[4], raw_values[5]]])
        norm_df = pd.DataFrame(norm_values, columns=input_cols_to_normalize)
        normalized_part = input_scaler.transform(norm_df)[0]
        
        # Keep sin/cos raw: yaw_sin, yaw_cos
        sincos_values = [raw_values[2], raw_values[3]]
        
        # Reassemble in original column order
        for i, col in enumerate(input_cols_all):
            if col in input_cols_to_normalize:
                idx_in_norm = input_cols_to_normalize.index(col)
                result[0, i] = normalized_part[idx_in_norm]
            else:  # sin/cos column
                idx_in_sincos = input_cols_sincos.index(col)
                result[0, i] = sincos_values[idx_in_sincos]
        return result

    def _apply_partial_denorm_output(model_output_np):
        """Denormalize only x, y columns; keep sin/cos raw from model output."""
        # model_output_np: [x_next, y_next, yaw_sin_next, yaw_cos_next]
        result = np.zeros(len(target_cols_all), dtype=np.float32)
        
        # Extract normalized values: x_next, y_next (indices 0, 1)
        norm_values = model_output_np[0, :2].reshape(1, -1)
        norm_df = pd.DataFrame(norm_values, columns=target_cols_to_normalize)
        denorm_values = target_scaler.inverse_transform(norm_df)[0]
        
        # Get sin/cos values directly (indices 2, 3)
        sincos_values = model_output_np[0, 2:4]
        
        # Reassemble in original order: [x_next, y_next, yaw_sin_next, yaw_cos_next]
        result[0] = denorm_values[0]  # x_next
        result[1] = denorm_values[1]  # y_next
        result[2] = sincos_values[0]  # yaw_sin_next
        result[3] = sincos_values[1]  # yaw_cos_next
        return result

    for k in range(n - 1):
        v_cmd, w_cmd = float(v_cmd_arr[k]), float(w_cmd_arr[k])
        yaw_sin, yaw_cos = np.sin(th), np.cos(th)

        # Build input: [x, y, yaw_sin, yaw_cos, v, w]
        raw_input = [x, y, yaw_sin, yaw_cos, v_cmd, w_cmd]

        # Apply PARTIAL normalization (only x, y, v, w)
        x_scaled = _apply_partial_norm_input(raw_input)
        input_tensor = torch.tensor(x_scaled, dtype=torch.float32, device=device)

        with torch.no_grad():
            scaled_output = model(input_tensor)

        # Apply PARTIAL denormalization (only x, y)
        pred_output = _apply_partial_denorm_output(scaled_output.cpu().numpy())
        
        x_next = float(pred_output[0])
        y_next = float(pred_output[1])
        yaw_sin_next = float(pred_output[2])
        yaw_cos_next = float(pred_output[3])
        
        # Convert sin/cos back to angle
        th_next = np.arctan2(yaw_sin_next, yaw_cos_next)

        # Update current state for the next step
        x_pred[k + 1], y_pred[k + 1] = x_next, y_next
        x, y, th = x_next, y_next, wrap_to_pi(th_next)

    return x_pred, y_pred


def main():
    # -- Sanity checks for required paths --
    for p in (CSV_PATH, MODEL_PATH, INPUT_SCALER_PATH, TARGET_SCALER_PATH, COLUMN_INFO_PATH, CONFIG_SNAPSHOT_YML):
        if not os.path.exists(p):
            raise FileNotFoundError(f"File not found: {p}")

    # -- Load snapshot to reconstruct the exact architecture --
    with open(CONFIG_SNAPSHOT_YML, "r") as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg["model_params"]
    input_dim = int(model_cfg["input_dim"])
    output_dim = int(model_cfg["output_dim"])
    hidden_layers = list(model_cfg["hidden_layers"])
    p_dropout = float(model_cfg.get("p_dropout", 0.0))
    print(f"[INFO] From snapshot: input_dim={input_dim}, output_dim={output_dim}, "
          f"hidden_layers={hidden_layers}, p_dropout={p_dropout}")

    # -- Device selection --
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # -- Build model and load weights --
    model = MLP(input_dim, output_dim, hidden_layers, p_dropout).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # -- Load scalers and column info --
    input_scaler = joblib.load(INPUT_SCALER_PATH)
    target_scaler = joblib.load(TARGET_SCALER_PATH)
    col_info = joblib.load(COLUMN_INFO_PATH)
    
    print(f"[INFO] Input columns (all): {col_info['input_cols_all']}")
    print(f"[INFO] Input columns (normalized): {col_info['input_cols_to_normalize']}")
    print(f"[INFO] Target columns (all): {col_info['target_cols_all']}")
    print(f"[INFO] Target columns (normalized): {col_info['target_cols_to_normalize']}")

    # -- Load test CSV and basic checks --
    print(f"[INFO] Loading CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")
    df.dropna(subset=["x", "y", "yaw", "v", "w"], inplace=True)
    if df.empty:
        print("[ERRO] Empty DataFrame after NaN removal. Aborting.")
        return
    df = df.sort_values(["episode", "env", "step"]).reset_index(drop=True)

    # -- Group by episode and produce comparison plots per env --
    episodes = df["episode"].unique()
    print(f"[INFO] Episodes found: {len(episodes)}")

    for ep in episodes:
        df_ep = df[df["episode"] == ep].copy()

        fig, (ax_traj, ax_err, ax_acc) = plt.subplots(
            3, 1, figsize=(10, 22), gridspec_kw={"height_ratios": [3, 1, 1]}
        )
        did_plot = False

        for env_id, df_env in df_ep.groupby("env"):
            df_env = df_env.sort_values("step")
            if len(df_env) < 2:
                continue
            did_plot = True

            steps = np.arange(len(df_env))
            x_real = df_env["x"].to_numpy(float)
            y_real = df_env["y"].to_numpy(float)
            v_cmd = df_env["v"].to_numpy(float)
            w_cmd = df_env["w"].to_numpy(float)
            x0, y0 = float(x_real[0]), float(y_real[0])
            yaw0 = float(df_env["yaw"].iloc[0])

            # Unicycle baseline
            x_u, y_u = rollout_unicycle(x0, y0, yaw0, v_cmd, w_cmd, DT)

            # MLP rollout
            x_m, y_m = rollout_mlp(
                x0, y0, yaw0, v_cmd, w_cmd,
                model, input_scaler, target_scaler, col_info, device
            )

            # Errors: per-step euclidean and cumulative sums
            err_u = np.hypot(x_real - x_u, y_real - y_u)
            err_m = np.hypot(x_real - x_m, y_real - y_m)
            acc_u = np.cumsum(err_u)
            acc_m = np.cumsum(err_m)

            # Plots for this env
            ax_traj.plot(x_real, y_real, label=f"env {env_id} — Real", linewidth=2.5, alpha=0.85)
            ax_traj.plot(x_u, y_u, "--", label=f"env {env_id} — Unicycle", linewidth=2)
            ax_traj.plot(x_m, y_m, ":", label=f"env {env_id} — MLP", linewidth=2)

            ax_err.plot(steps, err_u, "--", label=f"env {env_id} — Unicycle Err", linewidth=2)
            ax_err.plot(steps, err_m, ":", label=f"env {env_id} — MLP Err", linewidth=2)

            ax_acc.plot(steps, acc_u, "--", label=f"env {env_id} — Unicycle Acc. Err", linewidth=2)
            ax_acc.plot(steps, acc_m, ":", label=f"env {env_id} — MLP Acc. Err", linewidth=2)

        # Finalize and save figure for the episode
        if did_plot:
            ax_traj.set_aspect("equal")
            ax_traj.grid(True, alpha=0.3)
            ax_traj.set_xlabel("X (m)")
            ax_traj.set_ylabel("Y (m)")
            ax_traj.set_title("Comparativo de Trajetória (Real vs. Unicycle vs. MLP)")
            ax_traj.legend(loc="best")

            ax_err.grid(True, alpha=0.3)
            ax_err.set_xlabel("Step (k)")
            ax_err.set_ylabel("Erro Euclidiano (m)")
            ax_err.set_title("Erro de Predição (Distância Real - Modelo)")
            ax_err.legend(loc="best")
            ax_err.set_xlim(left=0)
            ax_err.set_ylim(bottom=0)

            ax_acc.grid(True, alpha=0.3)
            ax_acc.set_xlabel("Step (k)")
            ax_acc.set_ylabel("Erro Acumulado (m)")
            ax_acc.set_title("Erro de Predição Acumulado (Soma Cumulativa)")
            ax_acc.legend(loc="best")
            ax_acc.set_xlim(left=0)
            ax_acc.set_ylim(bottom=0)

            fig.suptitle(f"Episode {int(ep)} — dt={DT}s", fontsize=16)
            out_path = os.path.join(OUT_DIR, f"ep_{int(ep):05d}_compare.png")
            fig.tight_layout(rect=[0, 0.03, 1, 0.96])
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"[OK] Salvo: {out_path}")
        else:
            print(f"[WARN] Nenhum dado válido para plotar no episódio {ep}.")
        plt.close(fig)

    print("\n[INFO] Processamento concluído.")




if __name__ == "__main__":
    main()
