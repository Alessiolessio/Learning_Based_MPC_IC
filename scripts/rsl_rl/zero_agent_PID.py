# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import matplotlib.pyplot as plt
from isaaclab_tasks.utils import parse_env_cfg
import sys
from pathlib import Path
import os
import datetime
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plantation_utils import apply_overrides_train


class SimplePID:
    """PID controller for Vx (velocity on the x axis) control"""

    def __init__(self, kp, ki, kd, device, num_envs):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.device = device
        self.num_envs = num_envs
        self.prev_error = torch.zeros(num_envs, device=device)
        self.integral = torch.zeros(num_envs, device=device)

    def compute(self, error):
        dt = 0.5   # must be at least 0.2 to avoid critical oscillations
        # PID terms
        p_term = self.kp * error
        self.integral = self.integral + error * dt
        i_term = self.ki * self.integral
        d_term = self.kd * (error - self.prev_error) / dt
        self.prev_error = error.clone()
        return p_term + i_term + d_term

    def reset(self, device, num_envs):
        self.prev_error = torch.zeros(num_envs, device=device)
        self.integral = torch.zeros(num_envs, device=device)
# ===========================
# Example Usage
# ===========================
#
#     INITIALIZATION
#     pid = SimplePID(
#       kp=7.2,
#       ki=1.5,
#       kd=0.7,
#       device=device,    # type: ignore
#       num_envs=num_envs,    # type: ignore
#   )
#
#     COMPUTE ERROR AND APPLY ACTION
#     error = target_velocity - current_linear_vel_x
#     pid_output = pid.compute(error)
#     actions[:, 0] = pid_output   # Linear
#     actions[:, 1] = 0.0          # Angular
#
#     RESET:
#     pid.reset(device, num_envs)


def save_velocity_plot(history, target_velocity, env_idx, log_dir):
    """
    history: dict with keys 'vel' (list of floats), 'err' (list of floats)
    target_velocity: float
    env_idx: int
    log_dir: path string
    """
    if len(history["vel"]) == 0:
        return  # nothing to plot

    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"vel_plot_env{env_idx}_{timestamp}.png"
    filepath = os.path.join(log_dir, filename)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    times = list(range(len(history["vel"])))

    # Top: target and velocity
    ax1.plot(times, history["vel"], label="current_velocity")
    ax1.plot(times, [target_velocity] * len(times), linestyle="--", label="target_velocity")
    ax1.set_ylabel("Velocity (m/s)")
    ax1.legend()
    ax1.grid(True)

    # Bottom: error
    ax2.plot(times, history["err"], label="error (target - vel)")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Error (m/s)")
    ax2.legend()
    ax2.grid(True)

    fig.suptitle(f"Env {env_idx} — Velocity and Error ({timestamp})")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filepath)
    plt.close(fig)
    print(f"[LOG] Saved velocity plot for env {env_idx} -> {filepath}")


def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    apply_overrides_train(env_cfg)
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    num_envs = env.unwrapped.num_envs       # type: ignore
    device = env.unwrapped.device           # type: ignore
    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    pid = SimplePID(
        kp=7.2,
        ki=1.5,
        kd=0.7,
        device=device,    # type: ignore
        num_envs=num_envs,    # type: ignore
    )
    target_velocity = 0.5
    #   reset environment
    env.reset()
    # set up parameters
    delay_steps = 100  # number of steps to wait before moving
    histories = [{"vel": [], "err": []} for _ in range(num_envs)]     # clean history -> for the graphs
    LOG_DIR = "/home/nexus/VQ_PMC/logs"     # directory for saving the plots
    step_counter = 0    # number of steps of the simulation
    obs = None          # for completeness, we initialize the observations

    # simulate environment
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=device)    # reset the actions to avoid overrides

            # If we waited enough -> PERFORM CONTROL ACTION
            if obs is not None and (step_counter >= delay_steps):
                policy_obs = obs['policy']
                current_linear_vel_x = policy_obs[:, 0]     # shape (N,)
                error = target_velocity - current_linear_vel_x
                pid_output = pid.compute(error)
                actions[:, 0] = pid_output   # Linear
                actions[:, 1] = 0.0          # Angular

            # Step simulation
            obs, reward, terminated, truncated, info = env.step(actions)
            step_counter += 1

            # Observations
            if obs is not None:
                try:
                    policy_obs = obs['policy']          # tensor shape (N, ...)
                    current_linear_vel_x = policy_obs[:, 0]  # (N,)
                except Exception:
                    current_linear_vel_x = None

                if current_linear_vel_x is not None:
                    vel_np = policy_obs[:, 0].detach().cpu().numpy()
                    for i in range(num_envs):
                        histories[i]["vel"].append(float(vel_np[i]))
                        histories[i]["err"].append(float(target_velocity - vel_np[i]))

            # If one or all envs are terminated for any reason, WE RESET
            done_mask = (terminated | truncated)
            if torch.any(done_mask):
                # 1) Reset the history for the graphs
                done_mask_cpu = done_mask.detach().cpu().numpy()
                for idx, finished in enumerate(done_mask_cpu):
                    if finished:
                        save_velocity_plot(histories[idx], target_velocity, idx, LOG_DIR)
                        histories[idx] = {"vel": [], "err": []}     # clear history for this env
                # 2) Reset the env, the PID state, and the step counter for the next episode
                env.reset()
                pid.reset(device, num_envs)
                step_counter = 0

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
