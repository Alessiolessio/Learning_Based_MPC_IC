# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to run an environment with an MPPI controller (replacing the previous NMPC).

Key fixes:
- Ensure all cost matrices (Q, R, P) live on the same device as MPPI (cuda/cpu) and in float64 (double).
- Compute quadratic costs with a robust pattern: sum(z * (z @ M), dim=-1).
- Run MPPI on GPU when available; avoid unnecessary CPU<->GPU transfers.
"""

import argparse
from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# CLI: add Isaac/Sim arguments first (required by SimulationApp)
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="MPPI agent for Isaac Lab environments (TerraSentia).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# -----------------------------------------------------------------------------
# Launch SimulationApp BEFORE importing any Omniverse/Isaac modules
# -----------------------------------------------------------------------------
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# Standard imports (only after SimulationApp exists)
# -----------------------------------------------------------------------------
import gymnasium as gym
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import os
import datetime
import time
import numpy as np
import yaml
import joblib
from typing import Optional

# Project / Isaac Lab utilities
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plantation_utils import apply_overrides_train
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils import math as ilmath
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.sim as sim_utils

# MPPI (PyPI: pytorch-mppi==0.4.2)
from pytorch_mppi import MPPI

# =============================================================================
# MLP Dynamics - Simple self-contained inference function
# =============================================================================
# Path to trained MLP model (change this to use a different model)
HISTORY_MLP_MODEL_PATH = "/home/nexus/VQ_PMCnmpc/VQ_PMC/exported_models/neural_networks/history_mlp/trained_models/model_hist_5_epoch_200_batch_64_lr_1e-05_vs_30_hl_256_256_256_256_256_256"

# Global cache for MLP model and scalers (loaded once on first call)
_mlp_cache = {
    "model": None,
    "input_mean": None,
    "input_scale": None,
    "target_mean": None,
    "target_scale": None,
    "history_length": None,
    "device": None,
    "loaded": False,
}

# Path to trained raw MLP model (no history, single step: [x,y,yaw,v,w] -> [x_next,y_next,yaw_next])
RAW_MLP_MODEL_PATH = "/home/nexus/VQ_PMCnmpc/VQ_PMC/exported_models/neural_networks/raw_mlp/trained_models/model_epoch_200_batch_64_lr_1e-05_vs_30_hl_64_64"

# Global cache for raw MLP model (loaded once on first call)
_raw_mlp_cache = {
    "model": None,
    "input_mean": None,
    "input_scale": None, 
    "target_mean": None,
    "target_scale": None,
    "loaded": False
}

# Path to trained raw sincos MLP model (single step with sin/cos yaw representation)
# Input: [x, y, yaw_sin, yaw_cos, v, w] -> Output: [x_next, y_next, yaw_sin_next, yaw_cos_next]
RAW_SINCOS_MLP_MODEL_PATH = "/home/nexus/VQ_PMCnmpc/VQ_PMC/exported_models/neural_networks/raw_sincos_mlp/trained_models/model_epoch_300_batch_64_lr_1e-05_vs_30_hl_64"

# Global cache for raw sincos MLP model (loaded once on first call)
_raw_sincos_mlp_cache = {
    "model": None,
    "input_mean": None,  # Only for [x, y, v, w]
    "input_scale": None,
    "target_mean": None,  # Only for [x_next, y_next]
    "target_scale": None,
    "loaded": False
}

# Path to trained history sincos MLP model (with history buffer and sin/cos yaw representation)
# Input: [x, y, yaw_sin, yaw_cos, v, w] * history_length -> Output: [x_next, y_next, yaw_sin_next, yaw_cos_next]
HISTORY_SINCOS_MLP_MODEL_PATH = "/home/nexus/VQ_PMCnmpc/VQ_PMC/exported_models/neural_networks/history_sincos_mlp/trained_models/model_hist_5_epoch_200_batch_64_lr_1e-05_vs_30_hl_64_64"

# Global cache for history sincos MLP model (loaded once on first call)
_history_sincos_mlp_cache = {
    "model": None,
    "input_mean": None,  # Only for [x, y, v, w] repeated H times
    "input_scale": None,
    "target_mean": None,  # Only for [x_next, y_next]
    "target_scale": None,
    "history_length": None,
    "device": None,
    "loaded": False
}

# =============================================================================
# Plotting utilities
# =============================================================================
def save_plot(history, env_idx, log_dir, trim_last_n=1, waypoint_tolerance=0.1):
    """Save XY trajectory, orientation, linear speed, and angular speed plots for one env."""
    if len(history["pos_x"]) == 0:
        return

    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"trajectory_env{env_idx}_{timestamp}.png"
    filepath = os.path.join(log_dir, filename)

    n_points = len(history["pos_x"])
    end_idx = n_points - trim_last_n if n_points > trim_last_n else n_points

    pos_x = history["pos_x"][:end_idx]
    pos_y = history["pos_y"][:end_idx]
    theta = history["theta"][:end_idx]
    vel_x = history["vel_x"][:end_idx]
    vel_y = history["vel_y"][:end_idx]
    omega = history["omega"][:end_idx]

    if len(pos_x) == 0:
        return

    fig, ((ax_traj, ax_theta), (ax_speed, ax_omega)) = plt.subplots(2, 2, figsize=(14, 10))
    times = list(range(len(pos_x)))

    # XY trajectory with waypoints
    ax_traj.plot(pos_x, pos_y, 'b-', linewidth=2, label="Robot Path", alpha=0.7)
    ax_traj.plot(pos_x[0], pos_y[0], 'go', markersize=10, label="Start")
    ax_traj.plot(pos_x[-1], pos_y[-1], 'rs', markersize=10, label="End")
    if "waypoints" in history and len(history["waypoints"]) > 0:
        waypoints = np.array(history["waypoints"])
        ax_traj.plot(waypoints[:, 0], waypoints[:, 1], 'r*', markersize=15, label="Waypoints",
                     markeredgecolor='black', markeredgewidth=1)
        ax_traj.plot(waypoints[:, 0], waypoints[:, 1], 'r--', alpha=0.3, linewidth=1)
        for i, wp in enumerate(waypoints):
            ax_traj.annotate(f'{i}', xy=(wp[0], wp[1]), xytext=(5, 5), textcoords='offset points',
                             fontsize=8, color='red', weight='bold')
    ax_traj.set_xlabel("X Position (m)")
    ax_traj.set_ylabel("Y Position (m)")
    ax_traj.set_title("XY Trajectory")
    ax_traj.legend(loc='best')
    ax_traj.grid(True, alpha=0.3)
    ax_traj.axis('equal')

    # Orientation (θ) vs target θ (following current waypoint selection over time)
    ax_theta.plot(times, theta, label="Current θ", linewidth=2, color='green')
    if "waypoints" in history and len(history["waypoints"]) > 0:
        waypoints = np.array(history["waypoints"])
        target_theta_over_time = []
        current_waypoint_idx = 0
        for i in range(len(pos_x)):
            curr_x, curr_y = pos_x[i], pos_y[i]
            if current_waypoint_idx < len(waypoints):
                tx, ty = waypoints[current_waypoint_idx, 0], waypoints[current_waypoint_idx, 1]
                dist = np.hypot(curr_x - tx, curr_y - ty)
                if dist < waypoint_tolerance and current_waypoint_idx < len(waypoints) - 1:
                    current_waypoint_idx += 1
            target_theta_over_time.append(waypoints[current_waypoint_idx, 2])
        ax_theta.plot(times, target_theta_over_time, 'r--', linewidth=2, alpha=0.7, label="Target θ")
    ax_theta.set_xlabel("Step")
    ax_theta.set_ylabel("Orientation (rad)")
    ax_theta.set_title("Orientation over Time")
    ax_theta.legend(loc='best')
    ax_theta.grid(True, alpha=0.3)

    # Linear speed
    speed = [float(np.hypot(vel_x[i], vel_y[i])) for i in range(len(vel_x))]
    ax_speed.plot(times, speed, label="Linear Speed", linewidth=2, color='purple')
    ax_speed.set_xlabel("Step")
    ax_speed.set_ylabel("Speed (m/s)")
    ax_speed.set_title("Linear Speed over Time")
    ax_speed.legend(loc='best')
    ax_speed.grid(True, alpha=0.3)

    # Angular velocity
    ax_omega.plot(times, omega, label="Angular Velocity ω", linewidth=2, color='orange')
    ax_omega.set_xlabel("Step")
    ax_omega.set_ylabel("Angular Velocity (rad/s)")
    ax_omega.set_title("Angular Velocity over Time")
    ax_omega.legend(loc='best')
    ax_omega.grid(True, alpha=0.3)

    fig.suptitle(f"Environment {env_idx} — Trajectory Analysis — {len(history.get('waypoints', []))} Waypoints ({timestamp})",
                 fontsize=14, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[LOG] Saved trajectory plot for env {env_idx} -> {filepath}")

# =============================================================================
# Observation utilities
# =============================================================================
def current_euler_obs(policy_obs: torch.Tensor):
    """
    Convert obs['policy'] into [vx, vy, vz, wx, wy, wz, x, y, z, roll, pitch, yaw].
    This uses Isaac Lab's math helpers for quaternion->Euler.
    """
    vx = policy_obs[:, 0]
    vy = policy_obs[:, 1]
    vz = policy_obs[:, 2]
    wx = policy_obs[:, 3]
    wy = policy_obs[:, 4]
    wz = policy_obs[:, 5]
    x = policy_obs[:, 6]
    y = policy_obs[:, 7]
    z = policy_obs[:, 8]
    qw = policy_obs[:, 9]
    qx = policy_obs[:, 10]
    qy = policy_obs[:, 11]
    qz = policy_obs[:, 12]
    quat = torch.stack([qw, qx, qy, qz], dim=1)
    roll, pitch, yaw = ilmath.euler_xyz_from_quat(quat, wrap_to_2pi=False)
    return torch.stack([vx, vy, vz, wx, wy, wz, x, y, z, roll, pitch, yaw], dim=1)

# =============================================================================
# MPPI configuration (scalars; moved to device later)
# =============================================================================
DT_SIM = 0.1
NX = 3                           # [x, y, theta]
NU = 2                           # [v, w]
ACTION_LOW  = torch.tensor([-2.0, -2.0], dtype=torch.double)
ACTION_HIGH = torch.tensor([ 2.0,  2.0], dtype=torch.double)

N_SAMPLES = 2048
TIMESTEPS = 30
LAMBDA = 0.001
NOISE_SIGMA = torch.diag(torch.tensor([0.35, 0.35], dtype=torch.double))

Q_torch = torch.diag(torch.tensor([10.0, 10.0, 0.0], dtype=torch.double))
R_torch = torch.diag(torch.tensor([0.1, 0.00001], dtype=torch.double))
P_torch = torch.diag(torch.tensor([100.0, 100.0, 0.0], dtype=torch.double))

target_position_torch = None

# =============================================================================
# Dynamics functions
# =============================================================================
def unicycle_dynamics(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    V = u[..., 0]
    w = u[..., 1]
    theta = x[..., 2]
    x_next = torch.empty_like(x)
    x_next[..., 0] = x[..., 0] + DT_SIM * (V * torch.cos(theta))
    x_next[..., 1] = x[..., 1] + DT_SIM * (V * torch.sin(theta))
    x_next[..., 2] = x[..., 2] + DT_SIM * w
    x_next[..., 2] = torch.atan2(torch.sin(x_next[..., 2]), torch.cos(x_next[..., 2]))
    return x_next

def raw_mlp_dynamics(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """
    Raw MLP dynamics (no history). Input: [x, y, yaw, v, w] -> Output: [x_next, y_next, yaw_next]
    Loads model/scalers on first call, then performs batched GPU inference.
    """
    global _raw_mlp_cache
    
    if not _raw_mlp_cache["loaded"]:
        device = x.device
        
        # Load config
        config_path = os.path.join(RAW_MLP_MODEL_PATH, "config_snapshot.yaml")
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        
        hidden_layers = cfg["model_params"]["hidden_layers"]
        p_dropout = cfg["model_params"].get("p_dropout", 0.0)
        input_dim = cfg["model_params"]["input_dim"]  # 5: [x, y, yaw, v, w]
        output_dim = cfg["model_params"]["output_dim"]  # 3: [x_next, y_next, yaw_next]
        
        # Build model
        layers = []
        in_d = input_dim
        for h in hidden_layers:
            layers += [nn.Linear(in_d, h), nn.ReLU()]
            if p_dropout > 0:
                layers.append(nn.Dropout(p=p_dropout))
            in_d = h
        layers.append(nn.Linear(in_d, output_dim))
        model = nn.Sequential(*layers).to(device)
        
        # Load weights (strip "model." prefix)
        model_path = os.path.join(RAW_MLP_MODEL_PATH, "mlp_dynamics.pth")
        state_dict = torch.load(model_path, map_location=device)
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.eval()
        
        # Load scalers
        input_scaler = joblib.load(os.path.join(RAW_MLP_MODEL_PATH, "input_scaler.joblib"))
        target_scaler = joblib.load(os.path.join(RAW_MLP_MODEL_PATH, "target_scaler.joblib"))
        
        # Cache as torch tensors on GPU
        _raw_mlp_cache["model"] = model
        _raw_mlp_cache["input_mean"] = torch.tensor(input_scaler.mean_, dtype=torch.float32, device=device)
        _raw_mlp_cache["input_scale"] = torch.tensor(input_scaler.scale_, dtype=torch.float32, device=device)
        _raw_mlp_cache["target_mean"] = torch.tensor(target_scaler.mean_, dtype=torch.float32, device=device)
        _raw_mlp_cache["target_scale"] = torch.tensor(target_scaler.scale_, dtype=torch.float32, device=device)
        _raw_mlp_cache["loaded"] = True
        
        print(f"[RAW_MLP] Loaded from: {RAW_MLP_MODEL_PATH}")
        print(f"[RAW_MLP] Device: {device}, Hidden: {hidden_layers}")
    
    # Get cached values
    model = _raw_mlp_cache["model"]
    input_mean = _raw_mlp_cache["input_mean"]
    input_scale = _raw_mlp_cache["input_scale"]
    target_mean = _raw_mlp_cache["target_mean"]
    target_scale = _raw_mlp_cache["target_scale"]
    
    # Remember original shape
    original_shape = x.shape[:-1]
    
    # Action scaling constants (must match dataset preprocessing)
    KV = 1.0 # Scaling factor for linear velocity 0.2839
    KW = 1.0 # Scaling factor for angular velocity 0.13
    
    # Flatten and build input: [x, y, yaw, v, w]
    x_flat = x.reshape(-1, 3).float()  # [batch, 3] -> [x, y, theta]
    u_flat = u.reshape(-1, 2).float()  # [batch, 2] -> [v, w]
    
    # Scale actions to match training data
    u_scaled = u_flat.clone()
    u_scaled[:, 0] = u_flat[:, 0] * KV  # v * kv
    u_scaled[:, 1] = u_flat[:, 1] * KW  # w * kw
    
    mlp_input = torch.cat([x_flat, u_scaled], dim=-1)  # [batch, 5]
    
    # Normalize
    mlp_input = (mlp_input - input_mean) / input_scale
    
    # Inference
    with torch.no_grad():
        mlp_output = model(mlp_input)
    
    # Denormalize
    output = mlp_output * target_scale + target_mean
    
    # Wrap theta to (-pi, pi]
    output[..., 2] = torch.atan2(torch.sin(output[..., 2]), torch.cos(output[..., 2]))
    
    # Reshape and convert to double
    return output.reshape(*original_shape, 3).double()


def raw_sincos_mlp_dynamics(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """
    Raw sincos MLP dynamics (no history, sin/cos yaw representation).
    Input state: [x, y, yaw] -> converted to [x, y, yaw_sin, yaw_cos, v, w]
    Output: [x_next, y_next, yaw_sin_next, yaw_cos_next] -> converted back to [x_next, y_next, yaw_next]
    
    PARTIAL normalization: Only x, y, v, w are normalized with StandardScaler.
    sin/cos values are kept raw (already in [-1, 1]).
    """
    global _raw_sincos_mlp_cache
    
    if not _raw_sincos_mlp_cache["loaded"]:
        device = x.device
        
        # Load config
        config_path = os.path.join(RAW_SINCOS_MLP_MODEL_PATH, "config_snapshot.yaml")
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        
        hidden_layers = cfg["model_params"]["hidden_layers"]
        p_dropout = cfg["model_params"].get("p_dropout", 0.0)
        input_dim = cfg["model_params"]["input_dim"]  # 6: [x, y, yaw_sin, yaw_cos, v, w]
        output_dim = cfg["model_params"]["output_dim"]  # 4: [x_next, y_next, yaw_sin_next, yaw_cos_next]
        
        # Build model
        layers = []
        in_d = input_dim
        for h in hidden_layers:
            layers += [nn.Linear(in_d, h), nn.ReLU()]
            if p_dropout > 0:
                layers.append(nn.Dropout(p=p_dropout))
            in_d = h
        layers.append(nn.Linear(in_d, output_dim))
        model = nn.Sequential(*layers).to(device)
        
        # Load weights (strip "model." prefix)
        model_path = os.path.join(RAW_SINCOS_MLP_MODEL_PATH, "mlp_dynamics.pth")
        state_dict = torch.load(model_path, map_location=device)
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.eval()
        
        # Load scalers (these only normalize x, y, v, w - NOT sin/cos)
        input_scaler = joblib.load(os.path.join(RAW_SINCOS_MLP_MODEL_PATH, "input_scaler.joblib"))
        target_scaler = joblib.load(os.path.join(RAW_SINCOS_MLP_MODEL_PATH, "target_scaler.joblib"))
        
        # Cache as torch tensors on GPU
        # Input scaler: [x, y, v, w] (4 features)
        _raw_sincos_mlp_cache["model"] = model
        _raw_sincos_mlp_cache["input_mean"] = torch.tensor(input_scaler.mean_, dtype=torch.float32, device=device)
        _raw_sincos_mlp_cache["input_scale"] = torch.tensor(input_scaler.scale_, dtype=torch.float32, device=device)
        # Target scaler: [x_next, y_next] (2 features)
        _raw_sincos_mlp_cache["target_mean"] = torch.tensor(target_scaler.mean_, dtype=torch.float32, device=device)
        _raw_sincos_mlp_cache["target_scale"] = torch.tensor(target_scaler.scale_, dtype=torch.float32, device=device)
        _raw_sincos_mlp_cache["loaded"] = True
        
        print(f"[RAW_SINCOS_MLP] Loaded from: {RAW_SINCOS_MLP_MODEL_PATH}")
        print(f"[RAW_SINCOS_MLP] Device: {device}, Hidden: {hidden_layers}")
        print(f"[RAW_SINCOS_MLP] Input scaler (x,y,v,w): mean={input_scaler.mean_}, scale={input_scaler.scale_}")
        print(f"[RAW_SINCOS_MLP] Target scaler (x,y): mean={target_scaler.mean_}, scale={target_scaler.scale_}")
    
    # Get cached values
    model = _raw_sincos_mlp_cache["model"]
    input_mean = _raw_sincos_mlp_cache["input_mean"]  # [x, y, v, w]
    input_scale = _raw_sincos_mlp_cache["input_scale"]
    target_mean = _raw_sincos_mlp_cache["target_mean"]  # [x_next, y_next]
    target_scale = _raw_sincos_mlp_cache["target_scale"]
    
    # Remember original shape
    original_shape = x.shape[:-1]
    
    # Flatten state and action
    x_flat = x.reshape(-1, 3).float()  # [batch, 3] -> [x, y, theta]
    u_flat = u.reshape(-1, 2).float()  # [batch, 2] -> [v, w]
    
    # Extract components
    pos_x = x_flat[:, 0]
    pos_y = x_flat[:, 1]
    theta = x_flat[:, 2]
    v = u_flat[:, 0]
    w = u_flat[:, 1]
    
    # Convert yaw to sin/cos
    yaw_sin = torch.sin(theta)
    yaw_cos = torch.cos(theta)
    
    # Apply PARTIAL normalization:
    # - Normalize x, y, v, w using scaler
    # - Keep sin/cos raw
    
    # Stack values to normalize: [x, y, v, w]
    to_normalize = torch.stack([pos_x, pos_y, v, w], dim=-1)  # [batch, 4]
    normalized = (to_normalize - input_mean) / input_scale  # [batch, 4]
    
    # Build full input: [gauss_x, gauss_y, yaw_sin, yaw_cos, gauss_v, gauss_w]
    mlp_input = torch.stack([
        normalized[:, 0],  # gauss_x
        normalized[:, 1],  # gauss_y
        yaw_sin,           # raw sin
        yaw_cos,           # raw cos
        normalized[:, 2],  # gauss_v
        normalized[:, 3],  # gauss_w
    ], dim=-1)  # [batch, 6]
    
    # Inference
    with torch.no_grad():
        mlp_output = model(mlp_input)  # [batch, 4]: [gauss_x_next, gauss_y_next, yaw_sin_next, yaw_cos_next]
    
    # Apply PARTIAL denormalization:
    # - Denormalize x_next, y_next using scaler
    # - Keep sin/cos raw
    
    # Extract normalized position outputs
    pos_normalized = mlp_output[:, :2]  # [batch, 2]: [gauss_x_next, gauss_y_next]
    pos_denorm = pos_normalized * target_scale + target_mean  # [batch, 2]: [x_next, y_next]
    
    # Extract raw sin/cos outputs
    yaw_sin_next = mlp_output[:, 2]
    yaw_cos_next = mlp_output[:, 3]
    
    # Convert sin/cos back to angle
    theta_next = torch.atan2(yaw_sin_next, yaw_cos_next)
    
    # Build output: [x_next, y_next, theta_next]
    output = torch.stack([pos_denorm[:, 0], pos_denorm[:, 1], theta_next], dim=-1)
    
    # Reshape and convert to double
    return output.reshape(*original_shape, 3).double()


def history_mlp_dynamics(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """
    MLP-based dynamics function compatible with MPPI.
    Loads model/scalers on first call, then performs batched inference.
    
    Args:
        x: State tensor [..., 3] with [x, y, theta]
        u: Action tensor [..., 2] with [v, w]
        
    Returns:
        x_next: Next state tensor [..., 3] with [x_next, y_next, theta_next]
    """
    global _mlp_cache
    
    # Load model on first call
    if not _mlp_cache["loaded"]:
        device = x.device
        
        # Load config
        config_path = os.path.join(HISTORY_MLP_MODEL_PATH, "config_snapshot.yaml")
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        
        history_length = cfg["training_params"].get("history_length", 1)
        hidden_layers = cfg["model_params"]["hidden_layers"]
        p_dropout = cfg["model_params"].get("p_dropout", 0.0)
        
        # Build model (same architecture as training)
        input_dim = 5 * history_length  # [x, y, yaw, v, w] * H
        output_dim = 3  # [x_next, y_next, yaw_next]
        
        layers = []
        in_d = input_dim
        for h in hidden_layers:
            layers += [nn.Linear(in_d, h), nn.ReLU()]
            if p_dropout > 0:
                layers.append(nn.Dropout(p=p_dropout))
            in_d = h
        layers.append(nn.Linear(in_d, output_dim))
        model = nn.Sequential(*layers).to(device)
        
        # Load weights (strip "model." prefix from keys since original was wrapped in a class)
        model_path = os.path.join(HISTORY_MLP_MODEL_PATH, "mlp_dynamics.pth")
        state_dict = torch.load(model_path, map_location=device)
        # Remove "model." prefix from keys if present
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.eval()
        
        # Load scalers and convert to torch tensors
        input_scaler = joblib.load(os.path.join(HISTORY_MLP_MODEL_PATH, "input_scaler.joblib"))
        target_scaler = joblib.load(os.path.join(HISTORY_MLP_MODEL_PATH, "target_scaler.joblib"))
        
        # Cache everything
        _mlp_cache["model"] = model
        _mlp_cache["input_mean"] = torch.tensor(input_scaler.mean_, dtype=torch.float32, device=device)
        _mlp_cache["input_scale"] = torch.tensor(input_scaler.scale_, dtype=torch.float32, device=device)
        _mlp_cache["target_mean"] = torch.tensor(target_scaler.mean_, dtype=torch.float32, device=device)
        _mlp_cache["target_scale"] = torch.tensor(target_scaler.scale_, dtype=torch.float32, device=device)
        _mlp_cache["history_length"] = history_length
        _mlp_cache["device"] = device
        _mlp_cache["loaded"] = True
        
        print(f"[MLP] Loaded model from: {HISTORY_MLP_MODEL_PATH}")
        print(f"[MLP] Device: {device}, History: {history_length}, Hidden: {hidden_layers}")
    
    # Get cached values
    model = _mlp_cache["model"]
    input_mean = _mlp_cache["input_mean"]
    input_scale = _mlp_cache["input_scale"]
    target_mean = _mlp_cache["target_mean"]
    target_scale = _mlp_cache["target_scale"]
    H = _mlp_cache["history_length"]
    
    # Remember original shape
    original_shape = x.shape[:-1]
    
    # Flatten to [batch, 3] and [batch, 2]
    x_flat = x.reshape(-1, 3).float()
    u_flat = u.reshape(-1, 2).float()
    
    # Build input: [x, y, yaw, v, w] repeated H times
    state_action = torch.cat([x_flat, u_flat], dim=-1)  # [batch, 5]
    mlp_input = state_action.repeat(1, H)  # [batch, 5*H]
    
    # Normalize: (x - mean) / scale
    mlp_input = (mlp_input - input_mean) / input_scale
    
    # Inference
    with torch.no_grad():
        mlp_output = model(mlp_input)
    
    # Denormalize: y * scale + mean
    output = mlp_output * target_scale + target_mean
    
    # Wrap theta to (-pi, pi]
    output[..., 2] = torch.atan2(torch.sin(output[..., 2]), torch.cos(output[..., 2]))
    
    # Reshape and convert to double (MPPI dtype)
    return output.reshape(*original_shape, 3).double()

def history_sincos_mlp_dynamics(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """
    History-based MLP dynamics with sin/cos representation and partial normalization.
    Loads model/scalers on first call, then performs batched inference.
    
    Input per step: [x, y, yaw_sin, yaw_cos, v, w]
    Output: [x_next, y_next, yaw_sin_next, yaw_cos_next]
    
    Normalization: Only [x, y, v, w] are Gaussian-normalized; sin/cos stay raw.
    
    Args:
        x: State tensor [..., 3] with [x, y, theta]
        u: Action tensor [..., 2] with [v, w]
        
    Returns:
        x_next: Next state tensor [..., 3] with [x_next, y_next, theta_next]
    """
    global _history_sincos_mlp_cache
    
    # Load model on first call
    if not _history_sincos_mlp_cache["loaded"]:
        import yaml
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load config to get architecture
        config_path = os.path.join(HISTORY_SINCOS_MLP_MODEL_PATH, "config_snapshot.yaml")
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        
        history_length = cfg["training_params"]["history_length"]
        hidden_layers = cfg["model_params"]["hidden_layers"]
        p_dropout = cfg["model_params"].get("p_dropout", 0.0)
        
        # Input: [x, y, yaw_sin, yaw_cos, v, w] * history_length = 6 * H features
        input_dim = 6 * history_length
        # Output: [x_next, y_next, yaw_sin_next, yaw_cos_next]
        output_dim = 4
        
        # Build model architecture
        layers = []
        if not hidden_layers:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            layers.append(nn.Linear(input_dim, hidden_layers[0]))
            layers.append(nn.ReLU())
            if p_dropout > 0:
                layers.append(nn.Dropout(p=p_dropout))
            for i in range(len(hidden_layers) - 1):
                layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
                layers.append(nn.ReLU())
                if p_dropout > 0:
                    layers.append(nn.Dropout(p=p_dropout))
            layers.append(nn.Linear(hidden_layers[-1], output_dim))
        
        model = nn.Sequential(*layers).to(device)
        
        # Load weights
        state_dict = torch.load(
            os.path.join(HISTORY_SINCOS_MLP_MODEL_PATH, "mlp_dynamics.pth"),
            map_location=device
        )
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.eval()
        
        # Load scalers (only normalize x, y, v, w - not sin/cos)
        input_scaler = joblib.load(os.path.join(HISTORY_SINCOS_MLP_MODEL_PATH, "input_scaler.joblib"))
        target_scaler = joblib.load(os.path.join(HISTORY_SINCOS_MLP_MODEL_PATH, "target_scaler.joblib"))
        
        # Input scaler: normalizes [x, y, v, w] repeated H times -> shape (4*H,)
        # Target scaler: normalizes [x_next, y_next] -> shape (2,)
        
        # Cache everything
        _history_sincos_mlp_cache["model"] = model
        _history_sincos_mlp_cache["input_mean"] = torch.tensor(input_scaler.mean_, dtype=torch.float32, device=device)
        _history_sincos_mlp_cache["input_scale"] = torch.tensor(input_scaler.scale_, dtype=torch.float32, device=device)
        _history_sincos_mlp_cache["target_mean"] = torch.tensor(target_scaler.mean_, dtype=torch.float32, device=device)
        _history_sincos_mlp_cache["target_scale"] = torch.tensor(target_scaler.scale_, dtype=torch.float32, device=device)
        _history_sincos_mlp_cache["history_length"] = history_length
        _history_sincos_mlp_cache["device"] = device
        _history_sincos_mlp_cache["loaded"] = True
        
        print(f"[HISTORY_SINCOS_MLP] Loaded model from: {HISTORY_SINCOS_MLP_MODEL_PATH}")
        print(f"[HISTORY_SINCOS_MLP] Device: {device}, History: {history_length}, Hidden: {hidden_layers}")
    
    # Get cached values
    model = _history_sincos_mlp_cache["model"]
    input_mean = _history_sincos_mlp_cache["input_mean"]
    input_scale = _history_sincos_mlp_cache["input_scale"]
    target_mean = _history_sincos_mlp_cache["target_mean"]
    target_scale = _history_sincos_mlp_cache["target_scale"]
    H = _history_sincos_mlp_cache["history_length"]
    device = _history_sincos_mlp_cache["device"]
    
    # Remember original shape
    original_shape = x.shape[:-1]
    
    # Flatten to [batch, 3] and [batch, 2]
    x_flat = x.reshape(-1, 3).float()
    u_flat = u.reshape(-1, 2).float()
    
    # Extract state components
    pos_x = x_flat[:, 0]
    pos_y = x_flat[:, 1]
    theta = x_flat[:, 2]
    v = u_flat[:, 0]
    w = u_flat[:, 1]
    
    # Convert theta to sin/cos
    yaw_sin = torch.sin(theta)
    yaw_cos = torch.cos(theta)
    
    # Build per-step state-action: [x, y, yaw_sin, yaw_cos, v, w]
    state_action = torch.stack([pos_x, pos_y, yaw_sin, yaw_cos, v, w], dim=-1)  # [batch, 6]
    
    # Repeat H times to simulate history buffer (all same for single-step inference)
    state_action_history = state_action.repeat(1, H)  # [batch, 6*H]
    
    # Apply PARTIAL normalization: normalize only [x, y, v, w] components
    # Input scaler expects: [x_h0, y_h0, v_h0, w_h0, x_h1, y_h1, v_h1, w_h1, ...]
    # We need to extract and normalize the [x, y, v, w] values, keeping sin/cos raw
    
    # Build tensor to normalize: extract [x, y, v, w] from each history step
    to_normalize_list = []
    for h in range(H):
        # Each history step contributes [x, y, v, w] at positions [h*6+0, h*6+1, h*6+4, h*6+5]
        to_normalize_list.append(state_action_history[:, h*6 + 0])  # x
        to_normalize_list.append(state_action_history[:, h*6 + 1])  # y
        to_normalize_list.append(state_action_history[:, h*6 + 4])  # v
        to_normalize_list.append(state_action_history[:, h*6 + 5])  # w
    
    to_normalize = torch.stack(to_normalize_list, dim=-1)  # [batch, 4*H]
    normalized = (to_normalize - input_mean) / input_scale  # [batch, 4*H]
    
    # Build full MLP input: interleave normalized [x, y] with raw [sin, cos], then normalized [v, w]
    # Format: [gauss_x_h0, gauss_y_h0, sin_h0, cos_h0, gauss_v_h0, gauss_w_h0, ...]
    mlp_input_list = []
    for h in range(H):
        mlp_input_list.append(normalized[:, h*4 + 0])  # gauss_x
        mlp_input_list.append(normalized[:, h*4 + 1])  # gauss_y
        mlp_input_list.append(state_action_history[:, h*6 + 2])  # yaw_sin (raw)
        mlp_input_list.append(state_action_history[:, h*6 + 3])  # yaw_cos (raw)
        mlp_input_list.append(normalized[:, h*4 + 2])  # gauss_v
        mlp_input_list.append(normalized[:, h*4 + 3])  # gauss_w
    
    mlp_input = torch.stack(mlp_input_list, dim=-1)  # [batch, 6*H]
    
    # Inference
    with torch.no_grad():
        mlp_output = model(mlp_input)  # [batch, 4]: [x_next, y_next, yaw_sin_next, yaw_cos_next]
    
    # Apply PARTIAL denormalization: denormalize only [x_next, y_next]
    pos_normalized = mlp_output[:, :2]  # [batch, 2]
    pos_denorm = pos_normalized * target_scale + target_mean  # [batch, 2]
    
    # Extract raw sin/cos outputs
    yaw_sin_next = mlp_output[:, 2]
    yaw_cos_next = mlp_output[:, 3]
    
    # Convert sin/cos back to angle
    theta_next = torch.atan2(yaw_sin_next, yaw_cos_next)
    
    # Build output: [x_next, y_next, theta_next]
    output = torch.stack([pos_denorm[:, 0], pos_denorm[:, 1], theta_next], dim=-1)
    
    # Reshape and convert to double (MPPI dtype)
    return output.reshape(*original_shape, 3).double()

# =============================================================================
# Cost
# =============================================================================

def running_cost(x: torch.Tensor, u: torch.Tensor, *args) -> torch.Tensor:
    global target_position_torch, Q_torch, R_torch, P_torch
    t = args[0] if len(args) > 0 else None

    x_ref = target_position_torch
    while x_ref.ndim < x.ndim:
        x_ref = x_ref.unsqueeze(0)

    dx = x - x_ref
    state_cost = torch.sum(dx * dx.matmul(Q_torch), dim=-1)
    ctrl_cost  = torch.sum(u  *  u.matmul(R_torch), dim=-1)
    cost = state_cost + ctrl_cost

    if (t is not None) and (t == TIMESTEPS - 1):
        term_cost = torch.sum(dx * dx.matmul(P_torch), dim=-1)
        cost = cost + term_cost

    return cost

# =============================================================================
# Waypoint generator
# =============================================================================

# Cosine trajectory waypoints (9 waypoints forming a cosine wave)
# Sampled points along a cosine curve for trajectory following
COSINE_WAYPOINTS = np.array([
    [0.7180, 0.4705],
    [1.4360, 1.5279],
    [2.1540, 2.3760],
    [2.8720, 2.3764],
    [3.5900, 1.5287],
    [4.3080, 0.4712],
    [5.0260, 0.0000],
    [5.7440, 0.4698],
    [6.4620, 1.5271],
    [7.1800, 2.3752]
])

# Curve trajectory waypoints forming a smooth S-curve
# Fixed (x, y) waypoints for smooth curved path
CURVE_WAYPOINTS = np.array([
    [0.0, 0.0],      # Start
    [1.0, 0.0],      # Gentle rise
    [2.0, 0.0],      # Increasing curve
    [3.0, 0.0],      # Peak of first curve
    [4.0, 0.0],      # Maintain height
    [5.0, 0.0],      # Start descent
    [6.0, 0.0],      # Decreasing curve
    [7.0, 0.0],      # Almost flat
    [8.0, 0.0],      # Return to baseline
    [9.0, 0.0]       # End point
])

# U-turn trajectory waypoints forming a U shape
# Waypoints create a smooth U-turn maneuver
U_TURN_WAYPOINTS = np.array([
    [0.0, 0.0],      # Start
    [0.8, 0.0],      # Straight
    [1.6, 0.0],      # Continue straight
    [2.4, 0.2],      # Start curve
    [3.0, 0.6],      # Increasing curvature
    [3.3, 1.1],      # Pre-apex curve
    [3.5, 2.0],      # Apex (x = 3.5)
    [3.3, 2.6],      # Starting return
    [2.9, 3.1],      # Returning
    [2.3, 3.4],      # Continue return
    [1.6, 3.6],      # Straightening
    [0.8, 3.6],      # Final stretch
    [0.0, 3.4],      # Approaching end
    [0.0, 3.0],      # End point (requested)
])


def waypoint_generator(current_position, r_max, r_min, theta_max, theta_min, markers=None, all_waypoints=None, 
                       mode="random", cosine_index=None, curve_index=None, u_turn_index=None):
    """Generate a new waypoint and optionally visualize it with a 3D marker.
    
    Args:
        current_position: Current [x, y, theta] position
        r_max, r_min: Max and min radius for waypoint generation (random mode)
        theta_max, theta_min: Max and min angle for waypoint generation (random mode)
        markers: Optional VisualizationMarkers instance for visualization
        all_waypoints: Optional list to accumulate all waypoints for visualization
        mode: "random", "cosine", "curve", or "u_turn" for different trajectory types
        cosine_index: Current index in cosine trajectory (only used in cosine mode)
        curve_index: Current index in curve trajectory (only used in curve mode)
        u_turn_index: Current index in U-turn trajectory (only used in u_turn mode)
    
    Returns:
        tuple: (waypoint, next_index) where waypoint is np.array([x_goal, y_goal, theta_goal])
               and next_index is the next trajectory index (or None for random mode)
    """
    if mode == "cosine":
        # Use predefined cosine trajectory waypoints
        if cosine_index is None:
            cosine_index = 0
        
        # Get waypoint position
        x_goal, y_goal = COSINE_WAYPOINTS[cosine_index]
        
        # Calculate theta to face the next waypoint (or face forward if last waypoint)
        if cosine_index < len(COSINE_WAYPOINTS) - 1:
            next_idx = cosine_index + 1
            x_next, y_next = COSINE_WAYPOINTS[next_idx]
            theta_goal = np.arctan2(y_next - y_goal, x_next - x_goal)
        else:
            # Last waypoint - no more waypoints after this
            next_idx = None
            theta_goal = 0.0  # Face forward
        
        waypoint = np.array([x_goal, y_goal, theta_goal])
        next_index = next_idx
        
    elif mode == "curve":
        # Use predefined curve trajectory waypoints
        if curve_index is None:
            curve_index = 0
        
        # Get waypoint position
        x_goal, y_goal = CURVE_WAYPOINTS[curve_index]
        
        # Calculate theta to face the next waypoint (or face forward if last waypoint)
        if curve_index < len(CURVE_WAYPOINTS) - 1:
            next_idx = curve_index + 1
            x_next, y_next = CURVE_WAYPOINTS[next_idx]
            theta_goal = np.arctan2(y_next - y_goal, x_next - x_goal)
        else:
            # Last waypoint - no more waypoints after this
            next_idx = None
            theta_goal = 0.0  # Face forward
        
        waypoint = np.array([x_goal, y_goal, theta_goal])
        next_index = next_idx
    
    elif mode == "u_turn":
        # Use predefined U-turn trajectory waypoints
        if u_turn_index is None:
            u_turn_index = 0
        
        # Get waypoint position
        x_goal, y_goal = U_TURN_WAYPOINTS[u_turn_index]
        
        # Calculate theta to face the next waypoint (or face forward if last waypoint)
        if u_turn_index < len(U_TURN_WAYPOINTS) - 1:
            next_idx = u_turn_index + 1
            x_next, y_next = U_TURN_WAYPOINTS[next_idx]
            theta_goal = np.arctan2(y_next - y_goal, x_next - x_goal)
        else:
            # Last waypoint - no more waypoints after this
            next_idx = None
            theta_goal = 0.0  # Face forward
        
        waypoint = np.array([x_goal, y_goal, theta_goal])
        next_index = next_idx

    elif mode == "random":  # random mode
        # Seed for reproducibility (optional)
        # Smooth curve: seed(4)
        np.random.seed(None)

        x_curr, y_curr, theta_curr = current_position
        r = np.random.uniform(r_min, r_max)
        theta_rel = np.random.uniform(theta_min, theta_max)
        theta_goal = theta_curr + theta_rel
        x_goal = x_curr + r * np.cos(theta_goal)
        y_goal = y_curr + r * np.sin(theta_goal)
        theta_goal = np.arctan2(np.sin(theta_goal), np.cos(theta_goal))
        waypoint = np.array([x_goal, y_goal, theta_goal])
        next_index = None
    
    # Add waypoint to the list and update markers
    if all_waypoints is not None:
        all_waypoints.append(waypoint.copy())
    
    if markers is not None and all_waypoints is not None:
        # Update marker visualization with all waypoints
        # Extract [x, y, z] positions (z=0 for ground plane)
        waypoint_positions = np.array([[wp[0], wp[1], 0.05] for wp in all_waypoints])
        markers.visualize(translations=waypoint_positions)
    
    return waypoint, next_index

# =============================================================================
# Main control loop
# =============================================================================
def main():
    # 1) Build Isaac Lab env (device here is the physics/env device; MPPI device can differ)
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    num_envs = env.unwrapped.num_envs   # type: ignore
    device   = env.unwrapped.device     # type: ignore
    action_dim = 2

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space:      {env.action_space}")

    # 2) Force MPPI to use GPU if available (minimal change)
    device_torch = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO]: MPPI device: {device_torch}")

    # Keep all MPPI-related tensors on the same device and dtype
    global Q_torch, R_torch, P_torch
    Q_torch = Q_torch.to(device=device_torch, dtype=torch.double)
    R_torch = R_torch.to(device=device_torch, dtype=torch.double)
    P_torch = P_torch.to(device=device_torch, dtype=torch.double)

    ctrl = MPPI(
        dynamics=raw_sincos_mlp_dynamics,  # Change to history_mlp_dynamics or raw_sincos_mlp_dynamics as needed
        running_cost=running_cost,
        nx=NX,
        noise_sigma=NOISE_SIGMA.to(device=device_torch, dtype=torch.double),
        num_samples=N_SAMPLES,
        horizon=TIMESTEPS,
        lambda_=LAMBDA,
        device=device_torch,
        u_min=ACTION_LOW.to(device_torch),
        u_max=ACTION_HIGH.to(device_torch),
    )

    # 3) Reset env and prepare logging/waypoints
    env.reset()

    # Create waypoint markers for visualization
    waypoint_marker_cfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/WaypointMarkers",
        markers={
            "waypoint": sim_utils.SphereCfg(
                radius=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),  # Red spheres
            ),
        }
    )
    waypoint_markers = VisualizationMarkers(waypoint_marker_cfg)
    all_waypoints_list = []  # Track all waypoints for visualization

    delay_steps = 50
    histories = [{"pos_x": [], "pos_y": [], "theta": [], "vel_x": [], "vel_y": [], "omega": [], "waypoints": []}
                 for _ in range(num_envs)]
    LOG_DIR = "/home/nexus/VQ_PMCnmpc/VQ_PMC/logs/mppi_trajectories"
    step_counter = 0
    obs = None

    # Waypoint sampling parameters
    R_MIN, R_MAX = 1.0, 2.0
    THETA_MIN = - (np.pi * 7) / 18
    THETA_MAX =   (np.pi * 7) / 18
    WAYPOINT_TOLERANCE = 0.3  # Distance threshold to switch to next waypoint
    
    # Waypoint mode: "random", "cosine", "curve", or "u_turn"
    WAYPOINT_MODE = "curve"  # Change to "random", "cosine", "curve", or "u_turn" for different trajectory types
    cosine_indices = [0] * num_envs  # Track current index for each environment
    curve_indices = [0] * num_envs  # Track current curve trajectory index for each environment
    u_turn_indices = [0] * num_envs  # Track current U-turn trajectory index for each environment
    trajectory_completed = [False] * num_envs  # Track if trajectory is completed to avoid spam
    
    # Timer tracking for each environment
    timer_started = [False] * num_envs  # Track if timer has started for each environment
    start_times = [None] * num_envs  # Start time for each environment
    elapsed_times = [None] * num_envs  # Elapsed time for each environment
    
    # Trail visualization settings
    ENABLE_TRAIL = True  # Set to False to disable trail visualization
    TRAIL_UPDATE_FREQUENCY = 5  # Update trail every N steps (to avoid too many markers)
    
    # Create trail markers for robot path visualization
    trail_markers = None
    trail_positions_list = [[] for _ in range(num_envs)]  # Track trail positions per environment
    if ENABLE_TRAIL:
        trail_marker_cfg = VisualizationMarkersCfg(
            prim_path="/World/Visuals/TrailMarkers",
            markers={
                "trail": sim_utils.SphereCfg(
                    radius=0.05,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),  # Green spheres
                ),
            }
        )
        trail_markers = VisualizationMarkers(trail_marker_cfg)

    # Initial targets per env
    target_position_np = np.zeros((num_envs, 3))
    for i in range(num_envs):
        waypoint, next_idx = waypoint_generator(
            current_position=np.array([0.0, 0.0, 0.0]),
            r_max=R_MAX, r_min=R_MIN, theta_max=THETA_MAX, theta_min=THETA_MIN,
            markers=waypoint_markers,
            all_waypoints=all_waypoints_list,
            mode=WAYPOINT_MODE,
            cosine_index=cosine_indices[i],
            curve_index=curve_indices[i],
            u_turn_index=u_turn_indices[i]
        )
        target_position_np[i] = waypoint
        if WAYPOINT_MODE == "cosine" and next_idx is not None:
            cosine_indices[i] = next_idx
        elif WAYPOINT_MODE == "curve" and next_idx is not None:
            curve_indices[i] = next_idx
        elif WAYPOINT_MODE == "u_turn" and next_idx is not None:
            u_turn_indices[i] = next_idx
        histories[i]["waypoints"].append(target_position_np[i].copy())

    global target_position_torch
    target_position_torch = torch.from_numpy(target_position_np).to(dtype=torch.double, device=device_torch)

    # 4) Control loop
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = torch.zeros((num_envs, action_dim), device=device)

            if obs is not None and (step_counter >= delay_steps):
                # Start timer for environments that haven't started yet
                for i in range(num_envs):
                    if not timer_started[i] and not trajectory_completed[i]:
                        start_times[i] = time.time()
                        timer_started[i] = True
                        print(f"[ENV {i}] Timer started!")
                
                policy_obs = obs['policy']
                euler_obs = current_euler_obs(policy_obs)
                current_state_torch32 = euler_obs[:, [6, 7, 11]]  # (N,3): x, y, yaw

                # --- Use torch tensor directly on MPPI device (no CPU roundtrip) ---
                x_now = current_state_torch32.to(device=device_torch, dtype=torch.double)

                # For waypoint switching logic (distance), keep NumPy copy
                current_state_np = current_state_torch32.detach().cpu().numpy()

                # Waypoint switching
                for i in range(num_envs):
                    # Skip MPPI if trajectory is completed
                    if trajectory_completed[i]:
                        continue
                        
                    dist = np.hypot(current_state_np[i, 0] - target_position_np[i, 0],
                                    current_state_np[i, 1] - target_position_np[i, 1])
                    if dist < WAYPOINT_TOLERANCE:
                        # Check if we should generate a new waypoint
                        should_generate = True
                        if WAYPOINT_MODE == "cosine" and cosine_indices[i] >= len(COSINE_WAYPOINTS) - 1:
                            should_generate = False
                        elif WAYPOINT_MODE == "curve" and curve_indices[i] >= len(CURVE_WAYPOINTS) - 1:
                            should_generate = False
                        elif WAYPOINT_MODE == "u_turn" and u_turn_indices[i] >= len(U_TURN_WAYPOINTS) - 1:
                            should_generate = False
                        
                        if should_generate:
                            waypoint, next_idx = waypoint_generator(
                                current_position=current_state_np[i],
                                r_max=R_MAX, r_min=R_MIN, theta_max=THETA_MAX, theta_min=THETA_MIN,
                                markers=waypoint_markers,
                                all_waypoints=all_waypoints_list,
                                mode=WAYPOINT_MODE,
                                cosine_index=cosine_indices[i],
                                curve_index=curve_indices[i],
                                u_turn_index=u_turn_indices[i]
                            )
                            target_position_np[i] = waypoint
                            if WAYPOINT_MODE == "cosine" and next_idx is not None:
                                cosine_indices[i] = next_idx
                            elif WAYPOINT_MODE == "curve" and next_idx is not None:
                                curve_indices[i] = next_idx
                            elif WAYPOINT_MODE == "u_turn" and next_idx is not None:
                                u_turn_indices[i] = next_idx
                            histories[i]["waypoints"].append(target_position_np[i].copy())
                            waypoint_num = cosine_indices[i] if WAYPOINT_MODE == "cosine" else (curve_indices[i] if WAYPOINT_MODE == "curve" else (u_turn_indices[i] if WAYPOINT_MODE == "u_turn" else "N/A"))
                            print(f"[ENV {i}] Reached waypoint #{waypoint_num} -> New target: ({target_position_np[i, 0]:.2f}, "
                                  f"{target_position_np[i, 1]:.2f}, {target_position_np[i, 2]:.2f})")
                        else:
                            if not trajectory_completed[i]:
                                # Stop timer and calculate elapsed time
                                if timer_started[i] and start_times[i] is not None:
                                    elapsed_times[i] = time.time() - start_times[i]
                                    print(f"[ENV {i}] Reached final waypoint! Trajectory complete.")
                                    print(f"[ENV {i}] Total time: {elapsed_times[i]:.2f} seconds")
                                else:
                                    print(f"[ENV {i}] Reached final waypoint! Trajectory complete.")
                                trajectory_completed[i] = True

                # Update global target on MPPI device
                target_position_torch = torch.from_numpy(target_position_np).to(dtype=torch.double, device=device_torch)

                # MPPI on GPU
                u_cmd = ctrl.command(x_now)

                # Apply to Isaac (float32 on env device)
                actions[:, 0:2] = u_cmd.to(device=device, dtype=torch.float32)

            # Step simulation
            obs, reward, terminated, truncated, info = env.step(actions)
            step_counter += 1

            # Lightweight logging
            if obs is not None:
                try:
                    policy_obs = obs['policy']
                    euler_obs = current_euler_obs(policy_obs)
                    pos_np = euler_obs[:, [6, 7, 11]].detach().cpu().numpy()
                    vel_np = euler_obs[:, [0, 1, 5]].detach().cpu().numpy()
                    for i in range(num_envs):
                        histories[i]["pos_x"].append(float(pos_np[i, 0]))
                        histories[i]["pos_y"].append(float(pos_np[i, 1]))
                        histories[i]["theta"].append(float(pos_np[i, 2]))
                        histories[i]["vel_x"].append(float(vel_np[i, 0]))
                        histories[i]["vel_y"].append(float(vel_np[i, 1]))
                        histories[i]["omega"].append(float(vel_np[i, 2]))
                        
                        # Update trail visualization
                        if ENABLE_TRAIL and trail_markers is not None and step_counter % TRAIL_UPDATE_FREQUENCY == 0:
                            trail_positions_list[i].append([float(pos_np[i, 0]), float(pos_np[i, 1]), 0.03])
                    
                    # Visualize trail for all environments
                    if ENABLE_TRAIL and trail_markers is not None and step_counter % TRAIL_UPDATE_FREQUENCY == 0:
                        all_trail_positions = []
                        for trail in trail_positions_list:
                            all_trail_positions.extend(trail)
                        if all_trail_positions:
                            trail_markers.visualize(translations=np.array(all_trail_positions))
                except Exception:
                    pass

            # Reset on timeout/termination
            done_mask = (terminated | truncated)
            if torch.any(done_mask):
                done_mask_cpu = done_mask.detach().cpu().numpy()
                for idx, finished in enumerate(done_mask_cpu):
                    if finished:
                        save_plot(histories[idx], idx, LOG_DIR)
                        histories[idx] = {"pos_x": [], "pos_y": [], "theta": [],
                                          "vel_x": [], "vel_y": [], "omega": [], "waypoints": []}
                        # Clear trail for this environment
                        if ENABLE_TRAIL:
                            trail_positions_list[idx] = []
                        # Reset trajectory indices on environment reset
                        cosine_indices[idx] = 0
                        curve_indices[idx] = 0
                        u_turn_indices[idx] = 0
                        trajectory_completed[idx] = False  # Reset completion flag
                        
                        # Reset timer variables
                        timer_started[idx] = False
                        start_times[idx] = None
                        elapsed_times[idx] = None
                        waypoint, next_idx = waypoint_generator(
                            current_position=np.array([0.0, 0.0, 0.0]),
                            r_max=R_MAX, r_min=R_MIN, theta_max=THETA_MAX, theta_min=THETA_MIN,
                            markers=waypoint_markers,
                            all_waypoints=all_waypoints_list,
                            mode=WAYPOINT_MODE,
                            cosine_index=cosine_indices[idx],
                            curve_index=curve_indices[idx],
                            u_turn_index=u_turn_indices[idx]
                        )
                        target_position_np[idx] = waypoint
                        if WAYPOINT_MODE == "cosine" and next_idx is not None:
                            cosine_indices[idx] = next_idx
                        elif WAYPOINT_MODE == "curve" and next_idx is not None:
                            curve_indices[idx] = next_idx
                        elif WAYPOINT_MODE == "u_turn" and next_idx is not None:
                            u_turn_indices[idx] = next_idx
                        histories[idx]["waypoints"].append(target_position_np[idx].copy())

                # Clear all waypoints and reset markers
                all_waypoints_list.clear()
                # Re-add the reset waypoints
                for idx in range(num_envs):
                    if done_mask_cpu[idx]:
                        all_waypoints_list.append(target_position_np[idx].copy())
                # Update marker visualization with reset waypoints
                if all_waypoints_list:
                    waypoint_positions = np.array([[wp[0], wp[1], 0.05] for wp in all_waypoints_list])
                    waypoint_markers.visualize(translations=waypoint_positions)
                
                env.reset()
                step_counter = 0
                target_position_torch = torch.from_numpy(target_position_np).to(dtype=torch.double, device=device_torch)

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
