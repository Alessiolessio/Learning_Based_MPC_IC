# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

import argparse
from isaaclab.app import AppLauncher

# --- CLI / App ---
parser = argparse.ArgumentParser(description="Random-actions agent for Isaac Lab environments.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Imports ---
import gymnasium as gym
import torch
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import os
import datetime
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plantation_utils import apply_overrides_train  # (ok manter import; não usamos)
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils import math as ilmath


# ----------------------------- PLOTS ---------------------------------
def save_plot(history, env_idx, log_dir, trim_last_n=1):
    """
    Plota 4 gráficos: trajetória XY, orientação theta, velocidade linear (norma) e velocidade angular.
    Agora sem qualquer informação de waypoints.
    """
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

    # Trajetória XY
    ax_traj.plot(pos_x, pos_y, 'b-', linewidth=2, label="Robot Path", alpha=0.7)
    ax_traj.plot(pos_x[0], pos_y[0], 'go', markersize=10, label="Start")
    ax_traj.plot(pos_x[-1], pos_y[-1], 'rs', markersize=10, label="End")
    ax_traj.set_xlabel("X Position (m)")
    ax_traj.set_ylabel("Y Position (m)")
    ax_traj.set_title("XY Trajectory")
    ax_traj.legend(loc='best')
    ax_traj.grid(True, alpha=0.3)
    ax_traj.axis('equal')

    # Theta
    ax_theta.plot(times, theta, label="θ (yaw)", linewidth=2, color='green')
    ax_theta.set_xlabel("Step")
    ax_theta.set_ylabel("Orientation (rad)")
    ax_theta.set_title("Orientation over Time")
    ax_theta.legend(loc='best')
    ax_theta.grid(True, alpha=0.3)

    # Velocidade linear (norma)
    speed = [float(np.hypot(vel_x[i], vel_y[i])) for i in range(len(vel_x))]
    ax_speed.plot(times, speed, label="Linear Speed", linewidth=2, color='purple')
    ax_speed.set_xlabel("Step")
    ax_speed.set_ylabel("Speed (m/s)")
    ax_speed.set_title("Linear Speed over Time")
    ax_speed.legend(loc='best')
    ax_speed.grid(True, alpha=0.3)

    # Velocidade angular
    ax_omega.plot(times, omega, label="Angular Velocity ω", linewidth=2, color='orange')
    ax_omega.set_xlabel("Step")
    ax_omega.set_ylabel("Angular Velocity (rad/s)")
    ax_omega.set_title("Angular Velocity over Time")
    ax_omega.legend(loc='best')
    ax_omega.grid(True, alpha=0.3)

    fig.suptitle(f"Environment {env_idx} — Random Actions ({timestamp})", fontsize=14, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[LOG] Saved trajectory plot for env {env_idx} -> {filepath}")


# ----------------------------- OBS UTILS ---------------------------------
def current_euler_obs(policy_obs: torch.Tensor):
    """
    Converte obs['policy'] para eixos euclidianos + Euler:
      policy: [vx, vy, vz, wx, wy, wz, x, y, z, qw, qx, qy, qz]
    Retorna euler_obs: [vx, vy, vz, wx, wy, wz, x, y, z, roll, pitch, yaw]
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
    quat = torch.stack([qw, qx, qy, qz], dim=1)  # (w,x,y,z)

    roll, pitch, yaw = ilmath.euler_xyz_from_quat(quat, wrap_to_2pi=False)
    euler_obs = torch.stack([vx, vy, vz, wx, wy, wz, x, y, z, roll, pitch, yaw], dim=1)
    return euler_obs


# ----------------------------- MAIN ---------------------------------
def main():
    # --- cria env ---
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    # apply_overrides_train(env_cfg)  # (se quiser ligar, mantenha aqui)
    env = gym.make(args_cli.task, cfg=env_cfg)
    num_envs = env.unwrapped.num_envs   # type: ignore
    device = env.unwrapped.device       # type: ignore

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space:      {env.action_space}")

    # --- parâmetros da simulação aleatória ---
    V_MIN, V_MAX = 0.0, 2.0     # v ~ U[0, 2]
    W_MIN, W_MAX = -2.0, 2.0    # w ~ U[-2, 2]
    HOLD_STEPS = 50             # manter (v,w) por este nº de steps antes de sortear de novo
                                # (ex.: se dt_sim ~ 0.1 s, 20 steps ≈ 2 s)

    DELAY_STEPS = 100           # aguarda o mundo estabilizar antes de movimentar
    LOG_DIR = "/home/nexus/VQ_PMCnmpc/VQ_PMC/logs/random_trajectories"
    step_counter = 0
    obs = None

    # histórico para gráficos (por env)
    histories = [{"pos_x": [], "pos_y": [], "theta": [], "vel_x": [], "vel_y": [], "omega": []}
                 for _ in range(num_envs)]

    # estado do gerador aleatório (por env)
    hold_counter = 0
    current_actions = torch.zeros((num_envs, 2), device=device)  # [v, w] por env

    # reset
    env.reset()

    while simulation_app.is_running():
        with torch.inference_mode():
            # zera ações por segurança
            actions = torch.zeros(env.action_space.shape, device=device)

            # quando passar o delay, começamos a aplicar ações aleatórias “em blocos”
            if obs is not None and (step_counter >= DELAY_STEPS):
                if hold_counter % HOLD_STEPS == 0:
                    # sorteia novas ações para todos os envs
                    v = torch.rand((num_envs,), device=device) * (V_MAX - V_MIN) + V_MIN
                    w = (torch.rand((num_envs,), device=device) * (W_MAX - W_MIN) + W_MIN)
                    current_actions[:, 0] = v
                    current_actions[:, 1] = w
                hold_counter += 1

                # aplica as ações atuais
                actions[:, 0:2] = current_actions

            # avança sim
            obs, reward, terminated, truncated, info = env.step(actions)
            step_counter += 1

            # log de estados/velocidades para os plots
            if obs is not None:
                try:
                    policy_obs = obs['policy']  # tensor (N, ...)
                    euler_obs = current_euler_obs(policy_obs)
                    # x,y,theta
                    pos_np = euler_obs[:, [6, 7, 11]].detach().cpu().numpy()
                    # vx,vy,omega(z)
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

            # fim de episódio? salvar gráficos e resetar buffers
            done_mask = (terminated | truncated)
            if torch.any(done_mask):
                done_mask_cpu = done_mask.detach().cpu().numpy()
                for idx, finished in enumerate(done_mask_cpu):
                    if finished:
                        save_plot(histories[idx], idx, LOG_DIR)
                        histories[idx] = {"pos_x": [], "pos_y": [], "theta": [], "vel_x": [], "vel_y": [], "omega": []}
                env.reset()
                step_counter = 0
                hold_counter = 0
                current_actions.zero_()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
