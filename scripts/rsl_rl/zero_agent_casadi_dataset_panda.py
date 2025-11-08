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
import casadi as ca
import torch
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import os
import datetime
import numpy as np
import pandas as pd
from typing import Dict, Optional
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plantation_utils import apply_overrides_train
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils import math as ilmath


class NMPCControllerCasadi:
    """
    Nonlinear Model Predictive Controller using CasADi/IPOPT

    System dynamics (continuous):
        ẋ = V·cos(θ)
        ẏ = V·sin(θ)
        θ̇ = ω

    States: x = [x, y, θ]ᵀ
    Inputs: u = [V, ω]ᵀ
    """

    def __init__(
        self,
        dt: float = 0.5,
        N: int = 3,
        Q: np.ndarray = np.diag([1.0, 1.0, 1.0]),
        R: np.ndarray = np.diag([1.0, 1.0]),
        P: np.ndarray = np.diag([20.0, 20.0, 0.0]),
        v_max: float = 0.5,     # Max linear velocity (m/s²)
        w_max: float = 0.5,     # Max angular velocity (m/s²)
        a_v_max: float = np.inf,  # Max linear acceleration (m/s²)
        a_w_max: float = np.inf,  # Max angular acceleration (rad/s²)
        num_envs: int = 4,
        ipopt_options: Optional[Dict] = None,  # IPOPT solver options
        # Additional NMPC options
        warm_start: bool = True,
        integration_method: str = 'euler',  # 'euler', 'rk4'
        verbose: bool = False
    ):

        self.dt = dt
        self.N = N
        self.num_envs = num_envs
        self.warm_start_enabled = warm_start
        self.integration_method = integration_method
        self.verbose = verbose
        self.nx = 3  # state dimensions [x, y, θ]
        self.nu = 2  # input dimensions [V, ω]
        self.u_min = np.array([-v_max, -w_max])     # Constraints input
        self.u_max = np.array([v_max, w_max])
        self.a_v_max = a_v_max
        self.a_w_max = a_w_max
        self.Q = Q
        self.R = R
        self.P = P
        self.u_prev = [np.zeros((N, self.nu)) for _ in range(num_envs)]   # Store previous solutions for warm-start
        self.u_prev_step = [np.zeros(self.nu) for _ in range(num_envs)]   # Store only the last input
        self.x_prev = [None for _ in range(num_envs)]  # Store previous state trajectories

        # Warm start of multipliers -> not the MPC ones, but the IPOPT ones
        self.lam_g_prev = [None for _ in range(num_envs)]
        self.lam_x_prev = [None for _ in range(num_envs)]

        # Default IPOPT options (following best practices from Wächter & Biegler)
        self.default_ipopt_options = {
            # 1. Convergence tolerance - CRITICAL
            'ipopt.tol': 1e-6,
            # 1.5 Maximum iteration if the tolerance is reached
            'ipopt.acceptable_iter': 5,
            # 2. Maximum iterations - CRITICAL
            'ipopt.max_iter': 200,
            # 3. Print level
            'ipopt.print_level': 0 if not verbose else 5,
            'print_time': 0,
            # 4. Linear solver - CRITICAL
            'ipopt.linear_solver': 'mumps',
            # 5. Barrier parameter strategy - CRITICAL
            'ipopt.mu_strategy': 'adaptive',
            # 6. Warm start - CRITICAL for MPC
            'ipopt.warm_start_init_point': 'yes' if warm_start else 'no',
            'ipopt.warm_start_bound_push': 1e-9,
            'ipopt.warm_start_mult_bound_push': 1e-9,
            # 7. Hessian computation
            'ipopt.hessian_approximation': 'exact',
            # 8. Filter line-search
            'ipopt.accept_every_trial_step': 'no',
            # 9. NLP scaling
            'ipopt.nlp_scaling_method': 'gradient-based',
            # 10. Bound relaxation
            'ipopt.bound_relax_factor': 1e-5,
        }

        self.ipopt_options = self.default_ipopt_options.copy()
        if ipopt_options is not None:
            self.ipopt_options.update(ipopt_options)

        self._build_nlp()   # Build the NLP for each environment

        if verbose:
            print(f"[NMPC CasADi] Initialized with N={N}, dt={dt}, num_envs={num_envs}")
            print(f"[NMPC CasADi] Integration method: {integration_method}")
            print(f"[NMPC CasADi] IPOPT linear solver: {self.ipopt_options['ipopt.linear_solver']}")

    def _build_nlp(self):
        """Build the NLP problem using CasADi symbolic framework."""
        x = ca.MX.sym('x', self.nx, self.N + 1)  # Future states until N
        u = ca.MX.sym('u', self.nu, self.N)      # Future inputs until N-1
        u_prev = ca.MX.sym('u_prev', self.nu)  # Previous input (for acceleration constraints)
        x0 = ca.MX.sym('x0', self.nx)   # initial state
        x_ref = ca.MX.sym('x_ref', self.nx)     # reference state
        obj = 0     # Objective function

        # Stage costs
        for k in range(self.N):
            x_error = x[:, k] - x_ref
            obj += ca.mtimes([x_error.T, self.Q, x_error])
            obj += ca.mtimes([u[:, k].T, self.R, u[:, k]])

        # Terminal cost: (x_N - x_ref)ᵀ P (x_N - x_ref)
        x_error_final = x[:, self.N] - x_ref
        obj += ca.mtimes([x_error_final.T, self.P, x_error_final])

        # Constraints and bounds (constraints are dynamical models, bounds are constraints in the optimizatin literature sense)
        g = []  # Constraint expressions
        lbg = []  # Lower bounds on constraints
        ubg = []  # Upper bounds on constraints

        # Initial condition constraint: x_0 = x0
        g.append(x[:, 0] - x0)
        lbg.append(np.zeros(self.nx))
        ubg.append(np.zeros(self.nx))

        # Dynamics constraints
        for k in range(self.N):
            x_next = self._integrate(x[:, k], u[:, k])
            g.append(x[:, k + 1] - x_next)
            lbg.append(np.zeros(self.nx))
            ubg.append(np.zeros(self.nx))

        # We have N future actions, we need N constraints for acceleration limits
        for k in range(self.N):
            if k == 0:
                du = u[:, k] - u_prev
            else:
                du = u[:, k] - u[:, k - 1]
            accel = du / self.dt

            # Linear acceleration constraint
            if not np.isinf(self.a_v_max):
                g.append(accel[0])  # Linear acceleration
                lbg.append(np.array([-self.a_v_max]))
                ubg.append(np.array([self.a_v_max]))

            # Angular acceleration constraint
            if not np.isinf(self.a_w_max):
                g.append(accel[1])  # Angular acceleration
                lbg.append(np.array([-self.a_w_max]))
                ubg.append(np.array([self.a_w_max]))

        opt_variables = ca.vertcat(ca.reshape(x, -1, 1), ca.reshape(u, -1, 1))  # Flatten decision variables
        params = ca.vertcat(x0, x_ref, u_prev)  # parameters to the problem
        # Constraints
        g = ca.vertcat(*g)
        lbg = np.concatenate(lbg)
        ubg = np.concatenate(ubg)
        # Variable bounds
        lbx = []
        ubx = []
        # State bounds (no bounds on states for now)
        for k in range(self.N + 1):
            lbx.append([-np.inf, -np.inf, -np.inf])  # [x, y, θ] unbounded
            ubx.append([np.inf, np.inf, np.inf])
        # Input bounds
        for k in range(self.N):  # for all the prediction steps
            lbx.append(self.u_min)
            ubx.append(self.u_max)
        lbx = np.concatenate(lbx)
        ubx = np.concatenate(ubx)
        # Create NLP problem
        nlp = {
            'x': opt_variables,
            'f': obj,
            'g': g,
            'p': params
        }

        # Create solver of type ipopt with all pre-defined options
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, self.ipopt_options)
        # Store dimensions for warm start
        self.n_states = (self.N + 1) * self.nx
        self.n_inputs = self.N * self.nu
        self.n_vars = self.n_states + self.n_inputs
        # Store bounds
        self.lbx = lbx
        self.ubx = ubx
        self.lbg = lbg
        self.ubg = ubg

        if self.verbose:
            print(f"[NMPC CasADi] NLP built: {self.n_vars} variables, {len(lbg)} constraints")

    def _integrate(self, x, u):
        """
        Integrate dynamics using specified method
            x: Current state [x, y, θ]
            u: Control input [V, ω]
            x_next: Next state
        """
        if self.integration_method == 'euler':
            return self._euler_integration(x, u)
        elif self.integration_method == 'rk4':
            return self._rk4_integration(x, u)
        else:
            raise ValueError(f"Unknown integration method: {self.integration_method}")

    def _continuous_dynamics(self, x, u):
        """
        Continuous-time dynamics: ẋ = f(x, u)
            x: State [x, y, θ]
            u: Input [V, ω]
            x_dot: State derivative
        """
        V = u[0]
        omega = u[1]
        theta = x[2]
        x_dot = ca.vertcat(
            V * ca.cos(theta),
            V * ca.sin(theta),
            omega
        )
        return x_dot

    def _euler_integration(self, x, u):
        """Forward Euler integration."""
        x_dot = self._continuous_dynamics(x, u)
        x_next = x + self.dt * x_dot
        # Normalize theta to [-π, π]
        x_next_normalized = ca.vertcat(x_next[0], x_next[1], ca.atan2(ca.sin(x_next[2]), ca.cos(x_next[2])))
        return x_next_normalized

    def _rk4_integration(self, x, u):
        """4th-order Runge-Kutta integration."""
        k1 = self._continuous_dynamics(x, u)
        k2 = self._continuous_dynamics(x + self.dt / 2 * k1, u)
        k3 = self._continuous_dynamics(x + self.dt / 2 * k2, u)
        k4 = self._continuous_dynamics(x + self.dt * k3, u)
        x_next = x + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        # Normalize theta
        x_next_normalized = ca.vertcat(x_next[0], x_next[1], ca.atan2(ca.sin(x_next[2]), ca.cos(x_next[2])))
        return x_next_normalized

    def compute_control(self, x_current: np.ndarray, x_goal: np.ndarray) -> np.ndarray:
        """
        Solve NMPC problem for all environments.
            x_current: Current states, shape (num_envs, 3)
            x_goal: Goal states, shape (num_envs, 3)
            u_opt: Optimal controls, shape (num_envs, 2)
        """
        u_opt = np.zeros((self.num_envs, self.nu))
        # Here we solve each environment independently -> could be parallelized if needed
        for env_idx in range(self.num_envs):
            u_opt[env_idx] = self._solve_single(x_current[env_idx], x_goal[env_idx], env_idx)
        return u_opt

    def _solve_single(self, x0: np.ndarray, x_ref: np.ndarray, env_idx: int) -> np.ndarray:
        """
        Solve NMPC for a single environment.
            x0: Initial state (3,)
            x_ref: Reference state (3,)
            env_idx: Environment index
            u_opt: Optimal control (2,)
        """
        # Initial guess
        if self.warm_start_enabled and self.u_prev[env_idx] is not None:
            u_guess = np.vstack([self.u_prev[env_idx][1:], np.zeros((1, self.nu))])  # Shift input previous solution
            x_guess = np.zeros((self.N + 1, self.nx))  # Simple state trajectory guess by forward simulation
            x_guess[0] = x0
            for k in range(self.N):  # for all the prediction steps
                # Simple Euler step for initial guess
                V, omega = u_guess[k]
                theta = x_guess[k, 2]
                x_guess[k + 1, 0] = x_guess[k, 0] + self.dt * V * np.cos(theta)
                x_guess[k + 1, 1] = x_guess[k, 1] + self.dt * V * np.sin(theta)
                x_guess[k + 1, 2] = x_guess[k, 2] + self.dt * omega
        else:
            # Cold start with zeros
            x_guess = np.zeros((self.N + 1, self.nx))
            x_guess[0] = x0
            u_guess = np.zeros((self.N, self.nu))

        # Flatten initial guess
        x0_opt = np.concatenate([x_guess.flatten(), u_guess.flatten()])
        p = np.concatenate([x0, x_ref, self.u_prev_step[env_idx]])

        # Solve NLP
        try:
            # Build solver arguments
            solver_args = {
                'x0': x0_opt,
                'lbx': self.lbx,
                'ubx': self.ubx,
                'lbg': self.lbg,
                'ubg': self.ubg,
                'p': p
            }

            # Add IPOPT warm-start (dual variables from previous solve)
            if self.warm_start_enabled:
                if self.lam_g_prev[env_idx] is not None:
                    solver_args['lam_g0'] = self.lam_g_prev[env_idx]
                if self.lam_x_prev[env_idx] is not None:
                    solver_args['lam_x0'] = self.lam_x_prev[env_idx]

            # Solve
            sol = self.solver(**solver_args)
            x_sol = sol['x'].full().flatten()  # Extract solution
            x_opt = x_sol[:self.n_states].reshape(self.N + 1, self.nx)  # Extract optimal states
            u_opt = x_sol[self.n_states:].reshape(self.N, self.nu)  # Extract optima inputs
            self.u_prev[env_idx] = u_opt.copy()  # Store for warm start
            self.x_prev[env_idx] = x_opt.copy()  # Store for warm start

            # Store IPOPT warm-start (dual variables)
            if self.warm_start_enabled:
                self.lam_g_prev[env_idx] = sol['lam_g'].full().flatten()
                self.lam_x_prev[env_idx] = sol['lam_x'].full().flatten()

            # Update last input for acceleration constraints
            self.u_prev_step[env_idx] = u_opt[0].copy()

            # Return first control action
            return u_opt[0]

        except Exception as e:
            print(f"[NMPC CasADi] Solver failed for env {env_idx}: {e}")

            # Return zero control on failure
            return np.zeros(self.nu)

    def reset(self, num_envs: Optional[int] = None):
        """Reset the controller (clear warm-start solution)."""
        if num_envs is None:
            num_envs = self.num_envs
        else:
            self.num_envs = num_envs
        # Clear state-input history to not bias the next episode - NMPC warm-start
        self.u_prev = [np.zeros((self.N, self.nu)) for _ in range(num_envs)]
        self.x_prev = [None for _ in range(num_envs)]
        self.u_prev_step = [np.zeros(self.nu) for _ in range(num_envs)]
        # Clear IPOPT warm-start
        self.lam_g_prev = [None for _ in range(num_envs)]
        self.lam_x_prev = [None for _ in range(num_envs)]
        if self.verbose:
            print(f"[NMPC CasADi] Reset for {num_envs} environments")

    def set_ipopt_option(self, option_name: str, option_value):
        """
        Update a single IPOPT option and rebuild solver.
            option_name: IPOPT option name (e.g., 'ipopt.max_iter')
            option_value: New value for the option
        """
        self.ipopt_options[option_name] = option_value
        self._build_nlp()  # Rebuild with new options
        if self.verbose:
            print(f"[NMPC CasADi] Updated {option_name} = {option_value}")

    def get_solver_stats(self) -> Dict:
        """ Get statistics from the last solve """
        if hasattr(self, 'solver'):
            stats = self.solver.stats()
            return {
                'return_status': stats['return_status'],
                'iter_count': stats['iter_count'],
                'success': stats['success'],
                't_proc_total': stats['t_proc_total'],
                't_wall_total': stats['t_wall_total']
            }
        return {}
"""
What I figured out about the NMPC controller configuration---
- The weight on the angular velocity must be very small wrt the one on the linear velocity
- The prediction horizon must be long enough (N=15 seems good)
- The P matrix is a great introduction to improve convergence and stability - the wights must be high enough wrt the Q matrix
- The difference between Q and R must be high enough (e.g., 100x) but not too high (e.g., 10000x seems to create instability)
- The integration method does not seem to have a big impact, Euler is fine and faster
- The warm-start does not seems to have a big impact, but it is makes the algo faster
"""


def save_plot(history, env_idx, log_dir, trim_last_n=1, waypoint_tolerance=0.1):
    """
    Plot 4 subplots: XY trajectory, theta orientation, linear speed, and angular velocity
        history: dict with keys 'pos_x', 'pos_y', 'theta', 'vel_x', 'vel_y', 'omega', 'waypoints'
        env_idx: int
        log_dir: path string
        trim_last_n: number of last points to exclude (to avoid plotting teleportation)
        waypoint_tolerance: distance threshold to consider waypoint reached (m)
    """
    if len(history["pos_x"]) == 0:
        return  # nothing to plot

    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"trajectory_env{env_idx}_{timestamp}.png"
    filepath = os.path.join(log_dir, filename)

    # Trim last N points to avoid teleportation artifacts
    n_points = len(history["pos_x"])
    if n_points > trim_last_n:
        end_idx = n_points - trim_last_n
    else:
        end_idx = n_points
    
    # Slice all history arrays
    pos_x = history["pos_x"][:end_idx]
    pos_y = history["pos_y"][:end_idx]
    theta = history["theta"][:end_idx]
    vel_x = history["vel_x"][:end_idx]
    vel_y = history["vel_y"][:end_idx]
    omega = history["omega"][:end_idx]

    if len(pos_x) == 0:
        return  # nothing left to plot after trimming

    # Create figure with 4 subplots (2x2 grid)
    fig, ((ax_traj, ax_theta), (ax_speed, ax_omega)) = plt.subplots(2, 2, figsize=(14, 10))
    
    times = list(range(len(pos_x)))

    # --- XY Trajectory Plot (Top-Left) ---
    ax_traj.plot(pos_x, pos_y, 'b-', linewidth=2, label="Robot Path", alpha=0.7)
    ax_traj.plot(pos_x[0], pos_y[0], 'go', markersize=10, label="Start")
    ax_traj.plot(pos_x[-1], pos_y[-1], 'rs', markersize=10, label="End")

    # Plot waypoints
    if "waypoints" in history and len(history["waypoints"]) > 0:
        waypoints = np.array(history["waypoints"])
        ax_traj.plot(waypoints[:, 0], waypoints[:, 1], 'r*', markersize=15, 
                     label="Waypoints", markeredgecolor='black', markeredgewidth=1)
        ax_traj.plot(waypoints[:, 0], waypoints[:, 1], 'r--', alpha=0.3, linewidth=1)

        for i, wp in enumerate(waypoints):
            ax_traj.annotate(f'{i}', xy=(wp[0], wp[1]), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, color='red', weight='bold')

    ax_traj.set_xlabel("X Position (m)", fontsize=11)
    ax_traj.set_ylabel("Y Position (m)", fontsize=11)
    ax_traj.set_title("XY Trajectory", fontsize=12, weight='bold')
    ax_traj.legend(loc='best')
    ax_traj.grid(True, alpha=0.3)
    ax_traj.axis('equal')

    # --- Theta Orientation Plot (Top-Right) ---
    ax_theta.plot(times, theta, label="Current θ", linewidth=2, color='green')
    
    # Build step function for target theta based on waypoint switching
    if "waypoints" in history and len(history["waypoints"]) > 0:
        waypoints = np.array(history["waypoints"])
        target_theta_over_time = []

        current_waypoint_idx = 0  # Start with first waypoint as target
        
        for i in range(len(pos_x)):
            # Current position
            curr_x = pos_x[i]
            curr_y = pos_y[i]
            
            # Check if we reached the current target waypoint
            if current_waypoint_idx < len(waypoints):
                target_x = waypoints[current_waypoint_idx, 0]
                target_y = waypoints[current_waypoint_idx, 1]
                distance = np.sqrt((curr_x - target_x)**2 + (curr_y - target_y)**2)
                
                # If waypoint reached, switch to next waypoint
                if distance < waypoint_tolerance and current_waypoint_idx < len(waypoints) - 1:
                    current_waypoint_idx += 1
            
            # Append the current active target theta
            target_theta_over_time.append(waypoints[current_waypoint_idx, 2])
        
        # Plot the step function target theta
        ax_theta.plot(times, target_theta_over_time, 'r--', linewidth=2, 
                     alpha=0.7, label="Target θ (step function)")
    
    ax_theta.set_xlabel("Step", fontsize=11)
    ax_theta.set_ylabel("Orientation (rad)", fontsize=11)
    ax_theta.set_title("Orientation over Time", fontsize=12, weight='bold')
    ax_theta.legend(loc='best')
    ax_theta.grid(True, alpha=0.3)

    # --- Linear Speed Plot (Bottom-Left) ---
    speed = [np.sqrt(vel_x[i]**2 + vel_y[i]**2) for i in range(len(vel_x))]
    
    ax_speed.plot(times, speed, label="Linear Speed", linewidth=2, color='purple')
    ax_speed.set_xlabel("Step", fontsize=11)
    ax_speed.set_ylabel("Speed (m/s)", fontsize=11)
    ax_speed.set_title("Linear Speed over Time", fontsize=12, weight='bold')
    ax_speed.legend(loc='best')
    ax_speed.grid(True, alpha=0.3)

    # --- Angular Velocity Plot (Bottom-Right) ---
    ax_omega.plot(times, omega, label="Angular Velocity ω", linewidth=2, color='orange')
    ax_omega.set_xlabel("Step", fontsize=11)
    ax_omega.set_ylabel("Angular Velocity (rad/s)", fontsize=11)
    ax_omega.set_title("Angular Velocity over Time", fontsize=12, weight='bold')
    ax_omega.legend(loc='best')
    ax_omega.grid(True, alpha=0.3)

    # Overall title
    num_waypoints = len(history["waypoints"]) if "waypoints" in history else 0
    fig.suptitle(f"Environment {env_idx} — Trajectory Analysis — {num_waypoints} Waypoints ({timestamp})", 
                 fontsize=14, weight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[LOG] Saved trajectory plot for env {env_idx} (trimmed last {trim_last_n} points) -> {filepath}")

def current_euler_obs(policy_obs:torch.Tensor):
    """
    Convert the obs['policy'] to an euclidean space:
      - obs['policy']: [(vel), (pose)]
      - vel: [(vx, vy, vz), (wx, wy, wz)]
      - pose: [(x, y, z), (qw, qx, qy, qz)]
      - obs['policy']: [vx, vy, vz, wx, wy, wz, x, y, z, qw, qx, qy, qz]
    Returns: euler_obs: [vx, vy, vz, wx, wy, wz, x, y, z, raw, pitch, yaw]
    Exemple:    policy_obs = obs['policy']
                euler_obs = current_euler_obs(policy_obs)
                current_state_torch = euler_obs[:, [6, 7, 11]]
    """

    vx = policy_obs[:, 0]   # Linear velocity x
    vy = policy_obs[:, 1]   # Linear velocity y
    vz = policy_obs[:, 2]   # Linear velocity z
    wx = policy_obs[:, 3]   # Angular velocity x roll
    wy = policy_obs[:, 4]   # Angular velocity y pitch
    wz = policy_obs[:, 5]   # Angular velocity z yaw

    x = policy_obs[:, 6]    # Linear position x
    y = policy_obs[:, 7]    # Linear position y
    z = policy_obs[:, 8]    # Linear position z
    qw = policy_obs[:, 9]   # Angular quaternion w
    qx = policy_obs[:, 10]  # Angular quaternion x
    qy = policy_obs[:, 11]  # Angular quaternion y
    qz = policy_obs[:, 12]  # Angular quaternion z
    quat = torch.stack([qw, qx, qy, qz], dim=1)

    roll, pitch, yaw = ilmath.euler_xyz_from_quat(quat, wrap_to_2pi=False)

    # For NMPC we use x, y, theta = euler_obs[:, [6, 7, 11]]
    euler_obs = torch.stack([vx, vy, vz, wx, wy, wz, x, y, z, roll, pitch, yaw], dim=1)

    return euler_obs 


def waypoint_generator(current_position, r_max, r_min, theta_max, theta_min):
    """
    Generate a waypoint relative to the current robot position.
        current_position: np.array([x, y, theta]) - current robot pose
        r_max: maximum distance from current position (m)
        r_min: minimum distance from current position (m)
        theta_max: maximum angle relative to current heading (radians)
        theta_min: minimum angle relative to current heading (radians)
        np.array([x_goal, y_goal, theta_goal]) - next waypoint
    """
    x_curr, y_curr, theta_curr = current_position   # current position
    r = np.random.uniform(r_min, r_max)
    theta_rel = np.random.uniform(theta_min, theta_max)     # random r and theta

    theta_goal = theta_curr + theta_rel    # Goal orientation angle
    x_goal = x_curr + r * np.cos(theta_goal)    # Goal x coordinate
    y_goal = y_curr + r * np.sin(theta_goal)       # Goal y coordinate
    theta_goal = np.arctan2(np.sin(theta_goal), np.cos(theta_goal))  # Normalize theta_goal to [-π, π]

    return np.array([x_goal, y_goal, theta_goal])


def main():
    """Zero actions agent with Isaac Lab environment."""

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    #  apply_overrides_train(env_cfg)
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    num_envs = env.unwrapped.num_envs       # type: ignore
    device = env.unwrapped.device           # type: ignore
    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    # CSV simples
    DT = 0.02
    CSV_DIR = "/home/nexus/VQ_PMCnmpc/VQ_PMC/logs/datasets"
    os.makedirs(CSV_DIR, exist_ok=True)
    CSV_PATH = os.path.join(CSV_DIR, "dataset_nmpc_test.csv")
    CSV_COLUMNS = ["env", "episode", "step", "timestamp", "vx", "vy", "wz", "x", "y", "qw", "qx", "qy", "qz", "yaw", "v", "w"]
    if not os.path.exists(CSV_PATH):
        pd.DataFrame([], columns=CSV_COLUMNS).to_csv(CSV_PATH, index=False)
        print(f"[CSV] Created file with header: {CSV_PATH}")
    else:
        print(f"[CSV] Appending to existing file: {CSV_PATH}")

    NMPC = NMPCControllerCasadi(dt=0.1,
                                N=15,
                                Q=np.diag([10.0, 10.0, 0.0]),
                                R=np.diag([0.1, 0.00001]),      # v=0.1 / w=0.0001
                                P=np.diag([100.0, 100.0, 0.0]),
                                v_max=2,
                                w_max=2,
                                a_v_max=5,
                                a_w_max=5,
                                integration_method='euler',
                                warm_start=True,
                                num_envs=num_envs,
                                verbose=False)

    # reset environment
    env.reset()

    # set up parameters
    current_episode = 0
    current_linear_pos = torch.zeros((num_envs, 3), device=device)
    delay_steps = 50  # number of steps to wait before moving
    histories = [{"pos_x": [], "pos_y": [], "theta": [], "vel_x": [], "vel_y": [], "omega": [], "waypoints": []} for _ in range(num_envs)]
    LOG_DIR = "/home/nexus/VQ_PMC/logs/nmpc_trajectories"     # directory for saving the plots
    step_counter = 0    # number of steps of the simulation
    obs = None          # for completeness, we initialize the observations

    # Waypoint generator parameters
    R_MIN = 1  # minimum distance (m)
    R_MAX = 2    # maximum distance (m)
    THETA_MIN = -(np.pi) / 3   # -60 degrees -70 before
    THETA_MAX = (np.pi) / 3   # +60 degrees 70 before
    WAYPOINT_TOLERANCE = 0.2  # tolerance to consider waypoint reached (m)

    # generate initial target positions for all envs
    target_position_np = np.zeros((num_envs, 3))
    for i in range(num_envs):
        target_position_np[i] = waypoint_generator(
            current_position=np.array([0.0, 0.0, 0.0]),
            r_max=R_MAX,
            r_min=R_MIN,
            theta_max=THETA_MAX,
            theta_min=THETA_MIN
        )
        histories[i]["waypoints"].append(target_position_np[i].copy())

    # simulate environment
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=device)    # reset the actions to avoid overrides

            # If we waited enough -> PERFORM CONTROL ACTION
            if obs is not None and (step_counter >= delay_steps):
                policy_obs = obs['policy']
                euler_obs = current_euler_obs(policy_obs)
                current_state_torch = euler_obs[:, [6, 7, 11]]    # shape (N, 3) -> for all envs x,y,theta
                current_state_np = current_state_torch .cpu().numpy()  # to numpy for NMPC
                
                for i in range(num_envs):  # Check if any environment reached its waypoint
                    # Compute Euclidean distance to target (only x,y, ignore theta)
                    distance = np.sqrt((current_state_np[i, 0] - target_position_np[i, 0])**2 + (current_state_np[i, 1] - target_position_np[i, 1])**2)

                    if distance < WAYPOINT_TOLERANCE:  # Generate new waypoint from current position
                        target_position_np[i] = waypoint_generator(current_position=current_state_np[i], r_max=R_MAX, r_min=R_MIN, theta_max=THETA_MAX, theta_min=THETA_MIN)
                        histories[i]["waypoints"].append(target_position_np[i].copy())  # log the new waypoint
                        print(f"[ENV {i}] Reached waypoint! New target: "f"({target_position_np[i, 0]:.2f}, {target_position_np[i, 1]:.2f}, "f"{target_position_np[i, 2]:.2f})")

                u_opt_np = NMPC.compute_control(current_state_np, target_position_np)  # compute control for the new target position
                actions[:, 0:2] = torch.from_numpy(u_opt_np).to(device)

            # Step simulation
            obs, reward, terminated, truncated, info = env.step(actions)
            step_counter += 1

            # Observations
            if obs is not None:
                try:
                    policy_obs = obs['policy']                  # (N, 13)
                    euler_obs = current_euler_obs(policy_obs)   # (N, 12)
                    vx = policy_obs[:, 0]; vy = policy_obs[:, 1]; wz = policy_obs[:, 5]
                    x  = policy_obs[:, 6]; y  = policy_obs[:, 7]
                    qw = policy_obs[:, 9]; qx = policy_obs[:,10]; qy = policy_obs[:,11]; qz = policy_obs[:,12]
                    yaw = euler_obs[:, 11]
                    act_v = actions[:, 0]; act_w = actions[:, 1]
                    timestamp = step_counter * DT

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

                    # histórico p/ gráficos (opcional, mantido)
                    pos_np = euler_obs[:, [6, 7, 11]].detach().cpu().numpy()
                    vel_np = euler_obs[:, [0, 1, 5]].detach().cpu().numpy()
                    for i in range(num_envs):
                        histories[i]["pos_x"].append(float(pos_np[i][0]))
                        histories[i]["pos_y"].append(float(pos_np[i][1]))
                        histories[i]["theta"].append(float(pos_np[i][2]))
                        histories[i]["vel_x"].append(float(vel_np[i][0]))
                        histories[i]["vel_y"].append(float(vel_np[i][1]))
                        histories[i]["omega"].append(float(vel_np[i][2]))
                except Exception as e:
                    print(f"[CSV] Falha ao registrar linha: {type(e).__name__}: {e}")

            # If one or all envs are terminated for any reason, WE RESET
            done_mask = (terminated | truncated)
            if torch.any(done_mask):
                # 1) Reset the history for the graphs
                done_mask_cpu = done_mask.detach().cpu().numpy()
                for idx, finished in enumerate(done_mask_cpu):
                    if finished:
                        save_plot(histories[idx], idx, LOG_DIR)
                        histories[idx] = {"pos_x": [], "pos_y": [], "theta": [], "vel_x": [], "vel_y": [], "omega": [], "waypoints": []}    # clear history for this env
                        # 2) Reset the target positions and log the first waypoint
                        target_position_np[idx] = waypoint_generator(
                            current_position=np.array([0.0, 0.0, 0.0]),
                            r_max=R_MAX,
                            r_min=R_MIN,
                            theta_max=THETA_MAX,
                            theta_min=THETA_MIN
                        )
                        histories[idx]["waypoints"].append(target_position_np[idx].copy())
                # 3) Reset the env, the NMPC state, and the step counter for the next episode
                env.reset()
                NMPC.reset()
                step_counter = 0
                current_episode += 1
                print(f"[EPISODE] New Episode: {current_episode}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
