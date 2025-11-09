#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_mlp_model.py

Compares real trajectory with:
- Unicycle baseline rollout
- MLP rollout (using the exact architecture from config_snapshot.yaml)

Loads model/scalers from MODEL_BASE_PATH and produces per-episode plots.
"""

import os
import math
import yaml
import joblib
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from collections import deque  # kept (not used in current logic), harmless import

# Headless plotting backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------- USER CONFIG ---------------------
CSV_PATH = "/home/nexus/VQ_PMCnmpc/VQ_PMC/logs/datasets/dataset_nmpc_test.csv"
MODEL_BASE_PATH = "/home/nexus/VQ_PMCnmpc/VQ_PMC/exported_models/neural_networks/raw_mlp/trained_models/model_epoch_500_batch_64_lr_1e-05_vs_30_hl_256_256_256_256"
DT = 0.02  # time step for unicycle baseline
# -------------------------------------------------------

# Derived artifact paths
MODEL_PATH = os.path.join(MODEL_BASE_PATH, "mlp_dynamics.pth")
INPUT_SCALER_PATH = os.path.join(MODEL_BASE_PATH, "input_scaler.joblib")
TARGET_SCALER_PATH = os.path.join(MODEL_BASE_PATH, "target_scaler.joblib")
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
        v, w = float(v_arr[k]), float(w_arr[k])
        x = x + v * math.cos(th) * dt
        y = y + v * math.sin(th) * dt
        th = wrap_to_pi(th + w * dt)
        x_pred[k + 1], y_pred[k + 1] = x, y

    return x_pred, y_pred


def rollout_mlp(x0, y0, th0, v_cmd_arr, w_cmd_arr,
                model, input_scaler, target_scaler, device, input_features):
    """
    Perform one-step MLP predictions in sequence:
      input:  [x, y, yaw, v, w] at current step
      output: [x_next, y_next, yaw_next]
    """
    n = len(v_cmd_arr)
    x_pred = np.empty(n, dtype=float)
    y_pred = np.empty(n, dtype=float)

    x, y, th = float(x0), float(y0), float(th0)
    x_pred[0], y_pred[0] = x, y

    for k in range(n - 1):
        v_cmd, w_cmd = float(v_cmd_arr[k]), float(w_cmd_arr[k])

        # Build a single-step input row with feature names matching the scaler
        model_input_df = pd.DataFrame([[x, y, th, v_cmd, w_cmd]], columns=input_features)

        # Scale, infer, inverse-scale
        scaled_input = input_scaler.transform(model_input_df)
        input_tensor = torch.tensor(scaled_input, dtype=torch.float32, device=device)

        with torch.no_grad():
            scaled_output = model(input_tensor)

        pred_output = target_scaler.inverse_transform(scaled_output.cpu().numpy())[0]
        x_next, y_next, th_next = float(pred_output[0]), float(pred_output[1]), float(pred_output[2])

        # Update current state for the next step
        x_pred[k + 1], y_pred[k + 1] = x_next, y_next
        x, y, th = x_next, y_next, wrap_to_pi(th_next)

    return x_pred, y_pred


def main():
    # -- Sanity checks for required paths --
    for p in (CSV_PATH, MODEL_PATH, INPUT_SCALER_PATH, TARGET_SCALER_PATH, CONFIG_SNAPSHOT_YML):
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

    # -- Load scalers and feature names (defines input column order) --
    input_scaler = joblib.load(INPUT_SCALER_PATH)
    target_scaler = joblib.load(TARGET_SCALER_PATH)
    input_features = input_scaler.feature_names_in_.tolist()
    if len(input_features) != input_dim:
        raise ValueError(
            f"Input dim mismatch: scaler expects {len(input_features)} features, "
            f"but model input_dim={input_dim}."
        )

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
                model, input_scaler, target_scaler, device, input_features
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
