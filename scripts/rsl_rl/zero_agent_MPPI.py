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
import numpy as np
import yaml
import joblib
from typing import Optional

# Project / Isaac Lab utilities
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plantation_utils import apply_overrides_train
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils import math as ilmath

# MPPI (PyPI: pytorch-mppi==0.4.2)
from pytorch_mppi import MPPI

# =============================================================================
# MLP Dynamics - Simple self-contained inference function
# =============================================================================
# Path to trained MLP model (change this to use a different model)
MLP_MODEL_PATH = "/home/nexus/VQ_PMCnmpc/VQ_PMC/exported_models/neural_networks/history_mlp/trained_models/model_hist_5_epoch_200_batch_64_lr_1e-05_vs_30_hl_256_256_256_256_256_256"

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

N_SAMPLES = 1024
TIMESTEPS = 15
LAMBDA = 0.1
NOISE_SIGMA = torch.diag(torch.tensor([0.35, 0.35], dtype=torch.double))

Q_torch = torch.diag(torch.tensor([10.0, 10.0, 0.0], dtype=torch.double))
R_torch = torch.diag(torch.tensor([0.1, 0.00001], dtype=torch.double))
P_torch = torch.diag(torch.tensor([100.0, 100.0, 0.0], dtype=torch.double))

target_position_torch = None

# =============================================================================
# Dynamics and cost
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

def mlp_dynamics(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
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
        config_path = os.path.join(MLP_MODEL_PATH, "config_snapshot.yaml")
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
        model_path = os.path.join(MLP_MODEL_PATH, "mlp_dynamics.pth")
        state_dict = torch.load(model_path, map_location=device)
        # Remove "model." prefix from keys if present
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.eval()
        
        # Load scalers and convert to torch tensors
        input_scaler = joblib.load(os.path.join(MLP_MODEL_PATH, "input_scaler.joblib"))
        target_scaler = joblib.load(os.path.join(MLP_MODEL_PATH, "target_scaler.joblib"))
        
        # Cache everything
        _mlp_cache["model"] = model
        _mlp_cache["input_mean"] = torch.tensor(input_scaler.mean_, dtype=torch.float32, device=device)
        _mlp_cache["input_scale"] = torch.tensor(input_scaler.scale_, dtype=torch.float32, device=device)
        _mlp_cache["target_mean"] = torch.tensor(target_scaler.mean_, dtype=torch.float32, device=device)
        _mlp_cache["target_scale"] = torch.tensor(target_scaler.scale_, dtype=torch.float32, device=device)
        _mlp_cache["history_length"] = history_length
        _mlp_cache["device"] = device
        _mlp_cache["loaded"] = True
        
        print(f"[MLP] Loaded model from: {MLP_MODEL_PATH}")
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
    
    # Reshape and convert back to double (MPPI dtype)
    return output.reshape(*original_shape, 3).double()


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
def waypoint_generator(current_position, r_max, r_min, theta_max, theta_min):
    x_curr, y_curr, theta_curr = current_position
    r = np.random.uniform(r_min, r_max)
    theta_rel = np.random.uniform(theta_min, theta_max)
    theta_goal = theta_curr + theta_rel
    x_goal = x_curr + r * np.cos(theta_goal)
    y_goal = y_curr + r * np.sin(theta_goal)
    theta_goal = np.arctan2(np.sin(theta_goal), np.cos(theta_goal))
    return np.array([x_goal, y_goal, theta_goal])

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
        dynamics=mlp_dynamics,  # Change to mlp_dynamics for learned model
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

    delay_steps = 100
    histories = [{"pos_x": [], "pos_y": [], "theta": [], "vel_x": [], "vel_y": [], "omega": [], "waypoints": []}
                 for _ in range(num_envs)]
    LOG_DIR = "/home/nexus/VQ_PMCnmpc/VQ_PMC/logs/mppi_trajectories"
    step_counter = 0
    obs = None

    # Waypoint sampling parameters
    R_MIN, R_MAX = 1.0, 2.0
    THETA_MIN = - (np.pi * 7) / 18
    THETA_MAX =   (np.pi * 7) / 18
    WAYPOINT_TOLERANCE = 0.1

    # Initial targets per env
    target_position_np = np.zeros((num_envs, 3))
    for i in range(num_envs):
        target_position_np[i] = waypoint_generator(
            current_position=np.array([0.0, 0.0, 0.0]),
            r_max=R_MAX, r_min=R_MIN, theta_max=THETA_MAX, theta_min=THETA_MIN
        )
        histories[i]["waypoints"].append(target_position_np[i].copy())

    global target_position_torch
    target_position_torch = torch.from_numpy(target_position_np).to(dtype=torch.double, device=device_torch)

    # 4) Control loop
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = torch.zeros((num_envs, action_dim), device=device)

            if obs is not None and (step_counter >= delay_steps):
                policy_obs = obs['policy']
                euler_obs = current_euler_obs(policy_obs)
                current_state_torch32 = euler_obs[:, [6, 7, 11]]  # (N,3): x, y, yaw

                # --- Use torch tensor directly on MPPI device (no CPU roundtrip) ---
                x_now = current_state_torch32.to(device=device_torch, dtype=torch.double)

                # For waypoint switching logic (distance), keep NumPy copy
                current_state_np = current_state_torch32.detach().cpu().numpy()

                # Waypoint switching
                for i in range(num_envs):
                    dist = np.hypot(current_state_np[i, 0] - target_position_np[i, 0],
                                    current_state_np[i, 1] - target_position_np[i, 1])
                    if dist < WAYPOINT_TOLERANCE:
                        target_position_np[i] = waypoint_generator(
                            current_position=current_state_np[i],
                            r_max=R_MAX, r_min=R_MIN, theta_max=THETA_MAX, theta_min=THETA_MIN
                        )
                        histories[i]["waypoints"].append(target_position_np[i].copy())
                        print(f"[ENV {i}] Reached waypoint -> New target: ({target_position_np[i, 0]:.2f}, "
                              f"{target_position_np[i, 1]:.2f}, {target_position_np[i, 2]:.2f})")

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
                        target_position_np[idx] = waypoint_generator(
                            current_position=np.array([0.0, 0.0, 0.0]),
                            r_max=R_MAX, r_min=R_MIN, theta_max=THETA_MAX, theta_min=THETA_MIN
                        )
                        histories[idx]["waypoints"].append(target_position_np[idx].copy())

                env.reset()
                step_counter = 0
                target_position_torch = torch.from_numpy(target_position_np).to(dtype=torch.double, device=device_torch)

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
