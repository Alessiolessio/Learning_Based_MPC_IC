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


class NMPCController:
    """
    Nonlinear Model Predictive Controller for position control

    System dynamics (continuous):
        ẋ = V·cos(θ)
        ẏ = V·sin(θ)
        θ̇ = ω

    States: x = [x, y, θ]ᵀ
    Inputs: u = [V, ω]ᵀ
    Outputs: y = [x, y]ᵀ
    """

    def __init__(self, dt=0.2, N=3, Q=np.diag([1.0, 1.0, 1.0]), R=np.diag([1.0, 1.0]), P=np.diag([20.0, 20.0, 0.0]), v_max=0.5, w_max=0.5, method='pytorch', device='cuda', num_envs=4):
        """
        Args:
            dt: Sampling time (discretization step) [s]
            N: Prediction horizon (number of steps)
            Q: State matrix
            R: Input matrix
            P: Terminal matrix
            v_max: absolute value of the maximum allowed linear velocity
            w_max: absolute value of the maximum allowed angular velocity
            method: casadi or pytorch
            device: current device
            num_envs: Number of parallel environments
        """
        self.dt = dt
        self.N = N
        self.device = device
        self.num_envs = num_envs
        self.method = method

        # State and input dimensions
        self.nx = 3  # [x, y, θ]
        self.nu = 2  # [V, ω]
        self.ny = 2  # [x, y]

        # Input constraints: |V| <= 0.5, |ω| <= 0.5
        self.u_min = torch.tensor([-v_max, -w_max], device=device)
        self.u_max = torch.tensor([v_max, w_max], device=device)

        # Cost matrices
        self.Q = torch.diag(torch.tensor([Q[0, 0], Q[1, 1], Q[2, 2]], dtype=torch.float32, device=device))
        self.R = torch.diag(torch.tensor([R[0, 0], R[1, 1]], dtype=torch.float32, device=device))
        self.P = torch.diag(torch.tensor([P[0, 0], P[1, 1], P[2, 2]], dtype=torch.float32, device=device))

        # Initialize warm-start solution (previous optimal trajectory) -> zero vector
        self.u_prev = torch.zeros((num_envs, N, self.nu), device=device)

    def dynamics_continuous(self, x, u):
        """
        Continuous-time dynamics
            x: state [x, y, θ] - shape (num_envs, 3)
            u: input [V, ω] - shape (num_envs, 2)
        Returns:
            x_dot: state derivative - shape (num_envs, 3)
        """
        V = u[:, 0]
        omega = u[:, 1]
        theta = x[:, 2]
        x_dot = torch.stack([V * torch.cos(theta), V * torch.sin(theta), omega], dim=1)

        return x_dot

    def dynamics_discrete(self, x, u):
        """
        Discrete-time dynamics using Euler integration: x_{k+1} = f_d(x_k, u_k)
        """
        x_dot = self.dynamics_continuous(x, u)
        x_next = x + self.dt * x_dot
        # normalize theta to [-pi, pi]
        theta_normalized = torch.atan2(torch.sin(x_next[:, 2]), torch.cos(x_next[:, 2]))
        # Rebuild tensor instead of modifying in-place
        x_next = torch.cat([x_next[:, :2], theta_normalized.unsqueeze(1)], dim=1)
        return x_next

    def predict_trajectory(self, x0, u_sequence):
        """
        Predict state trajectory given initial state and control sequence.
            x0: initial state - shape (num_envs, 3)
            u_sequence: control sequence - shape (num_envs, N, 2)
            x_trajectory: predicted states - shape (num_envs, N+1, 3)
        """
        x_list = [x0]  # Lista invece di tensore pre-allocato

        for k in range(self.N):
            x_next = self.dynamics_discrete(x_list[-1], u_sequence[:, k, :])
            x_list.append(x_next)

        # Concatena alla fine
        x_traj = torch.stack(x_list, dim=1)

        return x_traj  # shape (num_envs, N+1, 3)

    def stage_cost(self, x, u, x_ref):
        """
        Stage cost: L(x, u) = (x - x_ref)ᵀ Q (x - x_ref) + uᵀ R u
            x: state - shape (num_envs, 3)
            u: input - shape (num_envs, 2)
            x_ref: reference state - shape (num_envs, 3)
            cost: scalar cost - shape (num_envs,)
        """
        x_error = x - x_ref
        state_cost = torch.sum(x_error * (x_error @ self.Q.T), dim=1)
        control_cost = torch.sum(u * (u @ self.R.T), dim=1)
        return state_cost + control_cost

    def terminal_cost(self, x, x_ref):
        """
        Terminal cost: V_f(x) = (x - x_ref)ᵀ P (x - x_ref)
            x: final state - shape (num_envs, 3)
            x_ref: reference state - shape (num_envs, 3)
            cost: scalar cost - shape (num_envs,)
        """
        x_error = x - x_ref
        return torch.sum(x_error * (x_error @ self.P.T), dim=1)

    def total_cost(self, x0, u_sequence, x_ref):
        """
        Total cost over horizon: J = Σ L(x_k, u_k) + V_f(x_N)
            x0: initial state - shape (num_envs, 3)
            u_sequence: control sequence - shape (num_envs, N, 2)
            x_ref: reference state (goal) - shape (num_envs, 3)
            cost: total cost - shape (num_envs,)
        """
        x_traj = self.predict_trajectory(x0, u_sequence)        # Predict trajectory
        total_cost = torch.zeros(self.num_envs, device=self.device)   # total stage cost
        for k in range(self.N):
            total_cost += self.stage_cost(x_traj[:, k, :], u_sequence[:, k, :], x_ref)

        total_cost += self.terminal_cost(x_traj[:, -1, :], x_ref)   # Add terminal cost

        return total_cost

    def apply_constraints(self, u):
        """
        Apply box constraints on inputs: u_min <= u <= u_max
            u: unconstrained input - shape (num_envs, N, 2)
            u_constrained: constrained input - shape (num_envs, N, 2)
        """
        return torch.clamp(u, self.u_min, self.u_max)

    def _solve_pytorch(self, x_current, x_goal):
        """
        Solve NMPC using PyTorch gradient descent with Adam optimizer.
            x_current: current state - shape (num_envs, 3)
            x_goal: goal state - shape (num_envs, 3)
            u_opt: optimal control input - shape (num_envs, 2)
        """
        # error checking -> it slows down the code SO MUCH if enabled, use only for debugging!!!
        # torch.autograd.set_detect_anomaly(True)
        # Optimization hyperparameters
        learning_rate = 0.4     # Step size for gradient descent
        max_iterations = 15   # Number of optimization steps

        # Initialize optimal control input tensor to zero for all envs
        u_opt = torch.zeros((self.num_envs, self.nu), device=self.device)

        # Solve for each environment sequentially -> slow!
        for env_idx in range(self.num_envs):
            # Initialize with warm-start (previous solution shifted)
            u_sequence = self.u_prev[env_idx].clone().detach()
            u_sequence.requires_grad = True

            # Setup Adam optimizer
            optimizer = torch.optim.Adam([u_sequence], lr=learning_rate)

            # Extract single environment data for all environments
            x0 = x_current[env_idx : env_idx + 1]  # shape (1, 3)
            x_g = x_goal[env_idx : env_idx + 1]     # shape (1, 3)

            # Optimization loop
            for iteration in range(max_iterations):
                optimizer.zero_grad()

                # Compute total cost
                u_seq = u_sequence.unsqueeze(0)  # shape (1, N, 2)
                cost = self.total_cost(x0, u_seq, x_g)

                # Backward pass
                cost.backward()
                optimizer.step()

                # Apply box constraints (project onto feasible set)
                u_sequence.data.clamp_(self.u_min, self.u_max)  # here we don't use the apply_constraints(self, u) function

            with torch.no_grad():
                self.u_prev[env_idx, :-1, :] = u_sequence[1:].clone()  # Store solution for warm-start (shift strategy)
                self.u_prev[env_idx, -1, :] = 0.0
                u_opt[env_idx] = u_sequence[0].clone()  # Extract first control action

        return u_opt  # shape (num_envs, 2)

    def compute_control(self, x_current, x_goal):
        """
        Solve the NMPC optimization problem using the selected method.
            x_current: current state - shape (num_envs, 3)
            x_goal: goal state - shape (num_envs, 3)
            u_opt: optimal control input for current step - shape (num_envs, 2)
        """
        if self.method == 'pytorch':
            return self._solve_pytorch(x_current, x_goal)
        # elif self.method == 'casadi':
            # TODO: return self._solve_casadi(x_current, x_goal)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def reset(self, num_envs=None):
        """Reset the controller (clear warm-start solution)."""
        if num_envs is None:
            num_envs = self.num_envs
        self.u_prev = torch.zeros((num_envs, self.N, self.nu), device=self.device)


# ===========================
# Example Usage
# ===========================
#
# # Initialize controller
# nmpc = NMPCController(dt=0.5, N=10, Q=np.diag([1.0, 1.0, 1.0]), R=np.diag([1.0, 1.0]), P=np.diag([20.0, 20.0, 0.0]), v_max=0.5, w_max=0.5, device=device, num_envs=num_envs)
#
# # Current state: [x, y, θ]
# x_current = torch.tensor([[0.0, 0.0, 0.0],
#                           [1.0, 1.0, 0.5],
#                           [2.0, -1.0, 1.57],
#                           [-1.0, 2.0, -0.5]], device='cuda')
#thod='pytorch
# # Goal state: [x_goal, y_goal, θ_goal]
# x_goal = torch.tensor([[5.0, 5.0, 0.0],
#                        [3.0, 4.0, 0.0],
#                        [0.0, 0.0, 0.0],
#                        [2.0, 2.0, 1.57]], device='cuda')
#
# # Compute optimal control
# u_opt = nmpc.compute_control(x_current, x_goal)
# print("Optimal control:", u_opt)

def save_plot(history, target_position, env_idx, log_dir):
    """
    Plot 4 subplots: x position, y position, theta orientation, and x velocity
        history: dict with keys 'pos_x', 'pos_y', 'theta', 'vel_x'
        target_position: array [x_goal, y_goal, theta_goal]
        env_idx: int
        log_dir: path string
    """
    if len(history["pos_x"]) == 0:
        return  # nothing to plot

    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"trajectory_env{env_idx}_{timestamp}.png"
    filepath = os.path.join(log_dir, filename)

    # Create figure with 4 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    times = list(range(len(history["pos_x"])))

    # Top-left: X position
    ax1.plot(times, history["pos_x"], label="Current X", linewidth=2)
    ax1.axhline(y=target_position[0], color='r', linestyle='--', label="Target X")
    ax1.set_ylabel("X Position (m)")
    ax1.legend()
    ax1.grid(True)
    ax1.set_title("X Position")

    # Top-right: Y position
    ax2.plot(times, history["pos_y"], label="Current Y", linewidth=2, color='orange')
    ax2.axhline(y=target_position[1], color='r', linestyle='--', label="Target Y")
    ax2.set_ylabel("Y Position (m)")
    ax2.legend()
    ax2.grid(True)
    ax2.set_title("Y Position")

    # Bottom-left: Theta orientation
    ax3.plot(times, history["theta"], label="Current θ", linewidth=2, color='green')
    ax3.axhline(y=target_position[2], color='r', linestyle='--', label="Target θ")
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Orientation (rad)")
    ax3.legend()
    ax3.grid(True)
    ax3.set_title("Orientation")

    # Bottom-right: X velocity
    ax4.plot(times, history["vel_x"], label="Linear Velocity", linewidth=2, color='purple')
    ax4.set_xlabel("Step")
    ax4.set_ylabel("Velocity (m/s)")
    ax4.legend()
    ax4.grid(True)
    ax4.set_title("Linear Velocity (X)")

    fig.suptitle(f"Environment {env_idx} — Trajectory Analysis ({timestamp})", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"[LOG] Saved trajectory plot for env {env_idx} -> {filepath}")


def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    # apply_overrides_train(env_cfg)
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    num_envs = env.unwrapped.num_envs       # type: ignore
    device = env.unwrapped.device           # type: ignore
    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    NMPC = NMPCController(dt=0.2,
                          N=3,
                          Q=np.diag([10.0, 20.0, 0.0]),
                          R=np.diag([0.1, 0.001]),
                          P=np.diag([200.0, 200.0, 0.0]),
                          v_max=0.5,
                          w_max=0.5,
                          device=device,
                          num_envs=num_envs)

    # set the target position TODO: change it later with a waypoint generator
    target_position = torch.zeros((num_envs, 3), device=device)
    target_position[:, 0] = 3  # x_goal = 3m
    target_position[:, 1] = -3  # y_goal = slightly to the right
    target_position[:, 2] = 0.0  # theta_goal = 0rad but we actually don't care about theta
    target_position_np = target_position.cpu().numpy()
    # reset environment
    env.reset()
    # set up parameters
    current_linear_pos = torch.zeros((num_envs, 3), device=device)
    delay_steps = 100  # number of steps to wait before moving
    histories = [{"pos_x": [], "pos_y": [], "theta": [], "vel_x": []} for _ in range(num_envs)]
    LOG_DIR = "/home/nexus/VQ_PMC/logs"     # directory for saving the plots
    step_counter = 0    # number of steps of the simulation
    obs = None          # for completeness, we initialize the observations

    # simulate environment
    while simulation_app.is_running():
        # we don't use torch.inference_mode() since we are going to compute gradients for NMPC
        actions = torch.zeros(env.action_space.shape, device=device)    # reset the actions to avoid overrides

        # If we waited enough -> PERFORM CONTROL ACTION
        if obs is not None and (step_counter >= delay_steps):
            policy_obs = obs['policy']
            current_linear_pos = policy_obs[:, [6, 7, 12]]    # shape (N, 3) -> for all envs x,y,theta
            u_opt = NMPC.compute_control(current_linear_pos, target_position)
            actions[:, 0:2] = u_opt

        # Step simulation
        obs, reward, terminated, truncated, info = env.step(actions)
        step_counter += 1

        # Observations
        if obs is not None:
            try:
                policy_obs = obs['policy']          # tensor shape (N, ...)
                current_linear_pos = policy_obs[:, [6, 7, 12]]  # (N,3) -> for all envs x,y,theta
            except Exception:
                current_linear_pos = None

            if current_linear_pos is not None:
                pos_np = policy_obs[:, [6, 7, 12]].detach().cpu().numpy()
                vel_np = policy_obs[:, 0].detach().cpu().numpy()
                for i in range(num_envs):
                    histories[i]["pos_x"].append(float(pos_np[i][0]))
                    histories[i]["pos_y"].append(float(pos_np[i][1]))
                    histories[i]["theta"].append(float(pos_np[i][2]))
                    histories[i]["vel_x"].append(float(vel_np[i]))
        # If one or all envs are terminated for any reason, WE RESET
        done_mask = (terminated | truncated)
        if torch.any(done_mask):
            # 1) Reset the history for the graphs
            done_mask_cpu = done_mask.detach().cpu().numpy()
            for idx, finished in enumerate(done_mask_cpu):
                if finished:
                    save_plot(histories[idx], target_position_np[idx], idx, LOG_DIR)
                    histories[idx] = {"pos_x": [], "pos_y": [], "theta": [], "vel_x": []}    # clear history for this env
            # 2) Reset the env, the PID state, and the step counter for the next episode
            env.reset()
            NMPC.reset()
            step_counter = 0

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
