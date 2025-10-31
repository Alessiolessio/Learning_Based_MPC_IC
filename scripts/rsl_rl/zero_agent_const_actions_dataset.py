# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

import argparse
from isaaclab.app import AppLauncher

# --- CLI / App ---
parser = argparse.ArgumentParser(description="Two-phase constant-actions agent for Isaac Lab environments.")
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
import pandas as pd  # <-- (novo)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plantation_utils import apply_overrides_train  # (ok manter import; não usamos)
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils import math as ilmath


# ============================= USER CONFIG =============================
# Fase 2 (ações finais):
V_CONST = 0.0    # m/s
W_CONST = 2.0    # rad/s

# Fase 1 (aplicada primeiro, por um tempo):
PHASE1_V = 0.0   # m/s
PHASE1_W = 2.0   # rad/s
PHASE1_STEPS = 300  # nº de steps

# Delay inicial (mundo estabilizar)
DELAY_STEPS = 100

# Logging CSV
DT = 0.02
CSV_DIR = "/home/nexus/VQ_PMCnmpc/VQ_PMC/logs"
CSV_PATH = os.path.join(CSV_DIR, "dataset_const.csv")
CSV_COLUMNS = ["env", "episode", "step", "timestamp",
               "vx", "vy", "wz", "x", "y",
               "qw", "qx", "qy", "qz",
               "yaw", "v", "w"]
# ======================================================================


# ----------------------------- PLOTS ---------------------------------
def save_plot(history, env_idx, log_dir, trim_last_n=1,
              p1v=PHASE1_V, p1w=PHASE1_W, p1n=PHASE1_STEPS,
              v2=V_CONST, w2=W_CONST):
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

    ax_traj.plot(pos_x, pos_y, linewidth=2, label="Robot Path", alpha=0.7)
    ax_traj.plot(pos_x[0], pos_y[0], 'go', markersize=10, label="Start")
    ax_traj.plot(pos_x[-1], pos_y[-1], 'rs', markersize=10, label="End")
    ax_traj.set_xlabel("X Position (m)")
    ax_traj.set_ylabel("Y Position (m)")
    ax_traj.set_title("XY Trajectory")
    ax_traj.legend(loc='best')
    ax_traj.grid(True, alpha=0.3)
    ax_traj.axis('equal')

    ax_theta.plot(times, theta, label="θ (yaw)", linewidth=2)
    ax_theta.set_xlabel("Step")
    ax_theta.set_ylabel("Orientation (rad)")
    ax_theta.set_title("Orientation over Time")
    ax_theta.legend(loc='best')
    ax_theta.grid(True, alpha=0.3)

    speed = [float(np.hypot(vel_x[i], vel_y[i])) for i in range(len(vel_x))]
    ax_speed.plot(times, speed, label="Linear Speed", linewidth=2)
    ax_speed.set_xlabel("Step")
    ax_speed.set_ylabel("Speed (m/s)")
    ax_speed.set_title("Linear Speed over Time")
    ax_speed.legend(loc='best')
    ax_speed.grid(True, alpha=0.3)

    ax_omega.plot(times, omega, label="Angular Velocity ω", linewidth=2)
    ax_omega.set_xlabel("Step")
    ax_omega.set_ylabel("Angular Velocity (rad/s)")
    ax_omega.set_title("Angular Velocity over Time")
    ax_omega.legend(loc='best')
    ax_omega.grid(True, alpha=0.3)

    fig.suptitle(
        f"Env {env_idx} — Phase1(v={p1v:.2f}, w={p1w:.2f}, steps={p1n}) → "
        f"Phase2(v={v2:.2f}, w={w2:.2f}) — {timestamp}",
        fontsize=14, weight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[LOG] Saved trajectory plot for env {env_idx} -> {filepath}")


# ----------------------------- OBS UTILS ---------------------------------
def current_euler_obs(policy_obs: torch.Tensor):
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
    euler_obs = torch.stack([vx, vy, vz, wx, wy, wz, x, y, z, roll, pitch, yaw], dim=1)
    return euler_obs


# ----------------------------- MAIN ---------------------------------
def main():
    # prepara CSV
    os.makedirs(CSV_DIR, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        pd.DataFrame([], columns=CSV_COLUMNS).to_csv(CSV_PATH, index=False)
        print(f"[CSV] Created file with header: {CSV_PATH}")
    else:
        print(f"[CSV] Appending to existing file: {CSV_PATH}")

    # cria env
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    # apply_overrides_train(env_cfg)
    env = gym.make(args_cli.task, cfg=env_cfg)
    num_envs = env.unwrapped.num_envs   # type: ignore
    device = env.unwrapped.device       # type: ignore

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space:      {env.action_space}")
    print(f"[INFO]: Phase1 v={PHASE1_V}, w={PHASE1_W} for {PHASE1_STEPS} steps; then Phase2 v={V_CONST}, w={W_CONST}")

    LOG_DIR = "/home/nexus/VQ_PMCnmpc/VQ_PMC/logs/two_phase_actions"
    step_counter = 0
    current_episode = 0
    obs = None

    # histórico para gráficos (por env)
    histories = [{"pos_x": [], "pos_y": [], "theta": [], "vel_x": [], "vel_y": [], "omega": []}
                 for _ in range(num_envs)]

    # reset
    env.reset()

    # ações constantes (todas iguais para todos os envs)
    phase1_actions = torch.zeros(env.action_space.shape, device=device)
    phase1_actions[:, 0] = PHASE1_V
    phase1_actions[:, 1] = PHASE1_W

    phase2_actions = torch.zeros(env.action_space.shape, device=device)
    phase2_actions[:, 0] = V_CONST
    phase2_actions[:, 1] = W_CONST

    while simulation_app.is_running():
        with torch.inference_mode():
            # começa zerado por segurança
            actions = torch.zeros(env.action_space.shape, device=device)

            # aplica ações após o delay
            if obs is not None and (step_counter >= DELAY_STEPS):
                if step_counter < (DELAY_STEPS + PHASE1_STEPS):
                    actions[:, 0:2] = phase1_actions
                else:
                    actions[:, 0:2] = phase2_actions

            # avança sim
            obs, reward, terminated, truncated, info = env.step(actions)
            step_counter += 1

            # ----------- LOG CSV + PLOTS -----------
            if obs is not None:
                try:
                    policy_obs = obs['policy']
                    euler_obs = current_euler_obs(policy_obs)

                    # cols necessárias
                    vx = policy_obs[:, 0]; vy = policy_obs[:, 1]; wz = policy_obs[:, 5]
                    x  = policy_obs[:, 6]; y  = policy_obs[:, 7]
                    qw = policy_obs[:, 9]; qx = policy_obs[:,10]; qy = policy_obs[:,11]; qz = policy_obs[:,12]
                    yaw = euler_obs[:, 11]
                    act_v = actions[:, 0]; act_w = actions[:, 1]
                    timestamp = step_counter * DT

                    # grava CSV (um bloco por step para todos envs)
                    rows = []
                    for i in range(num_envs):
                        rows.append({
                            "env": int(i),
                            "episode": int(current_episode),
                            "step": int(step_counter),
                            "timestamp": float(timestamp),
                            "vx": float(vx[i].item()), "vy": float(vy[i].item()), "wz": float(wz[i].item()),
                            "x": float(x[i].item()), "y": float(y[i].item()),
                            "qw": float(qw[i].item()), "qx": float(qx[i].item()),
                            "qy": float(qy[i].item()), "qz": float(qz[i].item()),
                            "yaw": float(yaw[i].item()),
                            "v": float(act_v[i].item()), "w": float(act_w[i].item()),
                        })
                    pd.DataFrame(rows, columns=CSV_COLUMNS).to_csv(CSV_PATH, mode="a", header=False, index=False)

                    # atualiza histórico para os plots
                    pos_np = euler_obs[:, [6, 7, 11]].detach().cpu().numpy()
                    vel_np = euler_obs[:, [0, 1, 5]].detach().cpu().numpy()
                    for i in range(num_envs):
                        histories[i]["pos_x"].append(float(pos_np[i, 0]))
                        histories[i]["pos_y"].append(float(pos_np[i, 1]))
                        histories[i]["theta"].append(float(pos_np[i, 2]))
                        histories[i]["vel_x"].append(float(vel_np[i, 0]))
                        histories[i]["vel_y"].append(float(vel_np[i, 1]))
                        histories[i]["omega"].append(float(vel_np[i, 2]))
                except Exception as e:
                    print(f"[CSV] Falha ao registrar linha: {type(e).__name__}: {e}")
            # ----------- /LOG CSV + PLOTS -----------

            # fim de episódio? salvar gráficos e resetar buffers
            done_mask = (terminated | truncated)
            if torch.any(done_mask):
                # salva gráficos por env concluído
                done_mask_cpu = done_mask.detach().cpu().numpy()
                for idx, finished in enumerate(done_mask_cpu):
                    if finished:
                        save_plot(
                            histories[idx], idx, LOG_DIR,
                            p1v=PHASE1_V, p1w=PHASE1_W, p1n=PHASE1_STEPS,
                            v2=V_CONST, w2=W_CONST
                        )
                        histories[idx] = {"pos_x": [], "pos_y": [], "theta": [], "vel_x": [], "vel_y": [], "omega": []}

                # ——— IMPORTANTE: preparar próximo episódio ———
                current_episode += 1          # (1) sobe antes do reset
                step_counter = 0              # (2) zera antes do reset
                env.reset()                   # (3) agora sim, reset
                # ------------------------------------------------

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
