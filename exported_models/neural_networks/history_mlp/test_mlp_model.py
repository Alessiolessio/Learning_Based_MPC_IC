#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_mlp_model.py

Loads a trained MLP and its scalers/config to compare rollouts:
- Ground-truth trajectory
- Unicycle baseline
- MLP rollout with a history buffer

NOTE:
- Reads 'config_snapshot.yaml' from the model folder to reproduce
  the same history_length, hidden_layers and dropout used in training.
- Keeps the original target layout: [x_next, y_next, yaw_next].
"""

import os
import math
import yaml
import joblib
import torch
import numpy as np
import pandas as pd
from collections import deque

import matplotlib
matplotlib.use("Agg")  # headless backend for saving figures
import matplotlib.pyplot as plt

from mlp_model import MLPDynamicsModel  # ensure same architecture as training

# ---------------------- USER-CONFIGURABLE PATHS ----------------------
MODEL_BASE_PATH = "/home/nexus/VQ_PMCnmpc/VQ_PMC/exported_models/neural_networks/history_mlp/trained_models/model_hist_5_epoch_200_batch_64_lr_1e-05_vs_30_hl_256_256_256_256_256_256"
CSV_PATH = "/home/nexus/VQ_PMCnmpc/VQ_PMC/logs/datasets/dataset_nmpc_test.csv"
DT = 0.02  # integration step for unicycle baseline (seconds)
# ---------------------------------------------------------------------

# Derived artifact paths
MODEL_PATH = os.path.join(MODEL_BASE_PATH, "mlp_dynamics.pth")
INPUT_SCALER_PATH = os.path.join(MODEL_BASE_PATH, "input_scaler.joblib")
TARGET_SCALER_PATH = os.path.join(MODEL_BASE_PATH, "target_scaler.joblib")
CONFIG_SNAPSHOT_YML = os.path.join(MODEL_BASE_PATH, "config_snapshot.yaml")
OUT_DIR = os.path.join(MODEL_BASE_PATH, "tests")
os.makedirs(OUT_DIR, exist_ok=True)

# Required columns in the test CSV
REQ_COLS = [
    "env", "episode", "step", "timestamp",
    "vx", "vy", "wz", "x", "y",
    "qw", "qx", "qy", "qz", "yaw", "v", "w"
]


def wrap_to_pi(angle: float) -> float:
    """Wraps angle to (-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def rollout_unicycle(x0, y0, th0, v_arr, w_arr, dt):
    """Simple unicycle forward integration (for comparison)."""
    n = len(v_arr)
    x_pred = np.empty(n, dtype=float)
    y_pred = np.empty(n, dtype=float)

    x = float(x0)
    y = float(y0)
    th = float(th0)

    x_pred[0] = x
    y_pred[0] = y

    for k in range(n - 1):
        v = float(v_arr[k])
        w = float(w_arr[k])
        x = x + v * math.cos(th) * dt
        y = y + v * math.sin(th) * dt
        th = wrap_to_pi(th + w * dt)
        x_pred[k + 1] = x
        y_pred[k + 1] = y

    return x_pred, y_pred


def rollout_mlp(
    x0, y0, th0, v_cmd_arr, w_cmd_arr,
    model, input_scaler, target_scaler, device,
    history_length: int, input_features: list[str]
):
    """
    Rollout using the MLP model trained as:
      f([sa(k), sa(k-1), ..., sa(k-H+1)]) -> state(k+1),
    where sa(k) = [x, y, yaw, v, w].
    """
    n = len(v_cmd_arr)
    x_pred = np.empty(n, dtype=float)
    y_pred = np.empty(n, dtype=float)

    # Current predicted state (x, y, yaw)
    current_state = np.array([x0, y0, th0], dtype=float)

    # History buffer (oldest->newest internally)
    history_q = deque(maxlen=history_length)
    initial_sa = np.array([x0, y0, th0, v_cmd_arr[0], w_cmd_arr[0]], dtype=float)
    for _ in range(history_length):
        history_q.append(initial_sa)

    x_pred[0] = x0
    y_pred[0] = y0

    for k in range(n - 1):
        v_cmd = float(v_cmd_arr[k])
        w_cmd = float(w_cmd_arr[k])

        # Build current state-action vector and push into history
        current_sa = np.array([current_state[0], current_state[1], current_state[2],
                               v_cmd, w_cmd], dtype=float)
        history_q.append(current_sa)

        # Flatten window in the scaler's expected order: h0 (most recent), h1, ...
        flat_input = []
        for sa_vec in reversed(history_q):
            flat_input.extend(sa_vec)
        flat_input = np.array([flat_input], dtype=np.float32)

        # Keep feature names to match scaler order
        model_input_df = pd.DataFrame(flat_input, columns=input_features)

        # Scale → infer → inverse-scale
        scaled_in = input_scaler.transform(model_input_df)
        input_tensor = torch.tensor(scaled_in, dtype=torch.float32, device=device)

        with torch.no_grad():
            scaled_out = model(input_tensor)

        pred_state = target_scaler.inverse_transform(scaled_out.cpu().numpy())[0]
        x_next, y_next, th_next = float(pred_state[0]), float(pred_state[1]), float(pred_state[2])

        x_pred[k + 1] = x_next
        y_pred[k + 1] = y_next
        current_state[:] = [x_next, y_next, wrap_to_pi(th_next)]

    return x_pred, y_pred


def main():
    # -- Basic artifact checks --
    if not os.path.isdir(MODEL_BASE_PATH):
        raise FileNotFoundError(f"Model directory not found: {MODEL_BASE_PATH}")
    for p in (MODEL_PATH, INPUT_SCALER_PATH, TARGET_SCALER_PATH, CONFIG_SNAPSHOT_YML):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing artifact: {p}")
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    # -- Read config snapshot to reproduce training settings --
    with open(CONFIG_SNAPSHOT_YML, "r") as f:
        cfg = yaml.safe_load(f)

    history_length = cfg["training_params"].get("history_length", 1)
    hidden_layers = cfg["model_params"]["hidden_layers"]
    p_dropout = cfg["model_params"].get("p_dropout", 0.0)

    state_action_dim = 5  # [x, y, yaw, v, w]
    input_dim = state_action_dim * history_length
    output_dim = 3        # [x_next, y_next, yaw_next]

    print(f"[INFO] From config_snapshot.yaml: H={history_length}, hidden_layers={hidden_layers}, p_dropout={p_dropout}")
    print(f"[INFO] Dimensions: input_dim={input_dim}, output_dim={output_dim}")

    # -- Load dataset and basic checks --
    df = pd.read_csv(CSV_PATH)
    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")
    df.dropna(subset=["x", "y", "yaw", "v", "w"], inplace=True)
    if df.empty:
        raise ValueError("Empty DataFrame after removing critical NaNs.")
    df = df.sort_values(["episode", "env", "step"]).reset_index(drop=True)

    # -- Device selection --
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # -- Model and scalers (match training-time architecture exactly) --
    model = MLPDynamicsModel(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_layers=hidden_layers,
        p_dropout=p_dropout,
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    input_scaler = joblib.load(INPUT_SCALER_PATH)
    target_scaler = joblib.load(TARGET_SCALER_PATH)

    input_features = input_scaler.feature_names_in_.tolist()
    if len(input_features) != input_dim:
        raise ValueError(
            f"Input dim mismatch: scaler expects {len(input_features)} features, "
            f"but model was built with input_dim={input_dim} (H={history_length})."
        )
    print(f"[INFO] Input features ({len(input_features)}): {input_features}")

    # -- Process by episode/env and generate comparison plots --
    episodes = df["episode"].unique()
    print(f"[INFO] Episodes found: {len(episodes)}")

    for ep in episodes:
        df_ep = df[df["episode"] == ep].copy()

        fig, (ax_traj, ax_err, ax_acc) = plt.subplots(
            3, 1, figsize=(10, 22), gridspec_kw={"height_ratios": [3, 1, 1]}
        )
        plotted_any = False

        for env_id, df_env in df_ep.groupby("env"):
            df_env = df_env.sort_values("step")
            if len(df_env) < 2:
                continue
            plotted_any = True

            steps = np.arange(len(df_env))
            x_real = df_env["x"].to_numpy(float)
            y_real = df_env["y"].to_numpy(float)
            v_cmd = df_env["v"].to_numpy(float)
            w_cmd = df_env["w"].to_numpy(float)
            x0, y0 = float(x_real[0]), float(y_real[0])
            yaw0 = float(df_env["yaw"].iloc[0])

            # Baseline rollout (unicycle)
            x_u, y_u = rollout_unicycle(x0, y0, yaw0, v_cmd, w_cmd, DT)

            # MLP rollout (history buffer + scalers)
            x_m, y_m = rollout_mlp(
                x0, y0, yaw0, v_cmd, w_cmd,
                model, input_scaler, target_scaler, device,
                history_length, input_features
            )

            # Errors (per step and cumulative)
            err_u = np.hypot(x_real - x_u, y_real - y_u)
            err_m = np.hypot(x_real - x_m, y_real - y_m)
            acc_u = np.cumsum(err_u)
            acc_m = np.cumsum(err_m)

            # Plots
            ax_traj.plot(x_real, y_real, label=f"env {env_id} — Real", linewidth=2.5, alpha=0.85)
            ax_traj.plot(x_u, y_u, "--", label=f"env {env_id} — Unicycle", linewidth=2)
            ax_traj.plot(x_m, y_m, ":", label=f"env {env_id} — MLP", linewidth=2)

            ax_err.plot(steps, err_u, "--", label=f"env {env_id} — Unicycle Err", linewidth=2)
            ax_err.plot(steps, err_m, ":", label=f"env {env_id} — MLP Err", linewidth=2)

            ax_acc.plot(steps, acc_u, "--", label=f"env {env_id} — Unicycle Acc. Err", linewidth=2)
            ax_acc.plot(steps, acc_m, ":", label=f"env {env_id} — MLP Acc. Err", linewidth=2)

        # Figure cosmetics + save
        if plotted_any:
            ax_traj.set_aspect("equal")
            ax_traj.grid(True, alpha=0.3)
            ax_traj.set_xlabel("X (m)")
            ax_traj.set_ylabel("Y (m)")
            ax_traj.set_title(f"Trajectory Comparison (H={history_length})")
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

            fig.suptitle(f"Episode {int(ep)} — Real vs. Unicycle vs. MLP (H={history_length})", fontsize=16)
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
