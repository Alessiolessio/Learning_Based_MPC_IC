# Copyright (c) 2022-2025
# SPDX-License-Identifier: BSD-3-Clause
"""
zero_agent_NMPC_casadi_multishoot.py
------------------------------------
Nonlinear MPC (unicycle) via CasADi + IPOPT (multiple shooting) for Isaac Lab.

- States:   x = [x, y, theta]
- Controls: u = [v, omega]
- Dynamics: x_dot = [v*cos(theta), v*sin(theta), omega]
"""

# =====[ Isaac App bootstrap ]==================================================
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="NMPC CasADi (multiple shooting) agent for Isaac Lab.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# =====[ Imports ]==============================================================
import os
import sys
import datetime
import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
from pathlib import Path

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import VQ_PMC.tasks  # noqa: F401
from VQ_PMC.tasks.manager_based.vq_pmc.mdp.scene import ENABLE_PLANTATION
from plantation_utils import apply_overrides_train  # noqa: F401


# =====[ Paths ]==============================================================
sys.path.insert(0, str(Path(__file__).resolve().parent))


# =====[ casADi ]==============================================================
try:
    import casadi as ca
except Exception as e:
    raise ImportError("CasADi is required. Install with `pip install casadi`.") from e


# =====[ NMPC Controller (CasADi + IPOPT) ]====================================
class NMPCMultipleShooting:
    """
    NMPC formulation with multiple shooting.
    - Free states (no bounds).
    - Bounded controls: v >= 0, |omega| <= w_max.
    """

    def __init__(
        self,
        T=0.2, N=3,
        v_max=0.5, v_min=0.0,
        w_max=0.5, w_min=None,
        Q=None, R=None, P=None,
        ipopt_max_iter=500, ipopt_print=False
    ):
        # Horizon/time parameters
        self.T = float(T)
        self.N = int(N)

        # Control Bounds
        self.v_max = float(v_max)
        self.v_min = float(0.0 if v_min is None else v_min)    # v >= 0  (no reverse)
        self.w_max = float(w_max)
        self.w_min = float(-w_max if w_min is None else w_min) # symmetric omega by default

        # Cost Matrices
        self.Q = np.diag([1.0, 1.0, 1.0])   if Q is None else np.asarray(Q, dtype=float)
        self.R = np.diag([1.0, 1.0])        if R is None else np.asarray(R, dtype=float)
        self.P = np.diag([20.0, 20.0, 0.0]) if P is None else np.asarray(P, dtype=float)

        # Build Solver
        self._build_solver(ipopt_max_iter=ipopt_max_iter, ipopt_print=ipopt_print)

        # Warm Start
        self.u0 = np.zeros((self.N, 2), dtype=float)  # (N, 2)
        self.X0 = None                                # (N+1, 3) or None

    # -------------------------------------------------------------------------
    # NLP construction with multiple shooting
    # -------------------------------------------------------------------------
    def _build_solver(self, ipopt_max_iter=500, ipopt_print=False):
        # State and control symbols
        x  = ca.SX.sym('x')
        y  = ca.SX.sym('y')
        th = ca.SX.sym('theta')
        states = ca.vertcat(x, y, th)
        n_states = 3

        v = ca.SX.sym('v')
        w = ca.SX.sym('omega')
        controls = ca.vertcat(v, w)
        n_controls = 2

        # Continuous unicycle dynamics
        rhs = ca.vertcat(
            v * ca.cos(th),
            v * ca.sin(th),
            w
        )
        f = ca.Function('f', [states, controls], [rhs])

        # Decision variables over the horizon
        U = ca.SX.sym('U', n_controls, self.N)     # controls (2 x N)
        X = ca.SX.sym('X', n_states,   self.N + 1) # states   (3 x (N+1))

        # Parameters: x0 (initial state) + x_ref (goal)
        P = ca.SX.sym('P', n_states + n_states)    # [x0(3); xref(3)]

        # Cost and constraints
        obj = 0
        g   = []

        Q = ca.SX(self.Q)
        R = ca.SX(self.R)
        Pterm = ca.SX(self.P)

        x0   = P[0:3]
        xref = P[3:6]

        # Initial condition: X[:,0] = x0
        g.append(X[:, 0] - x0)

        # Horizon loop: stage cost + dynamic constraints (Euler)
        for k in range(self.N):
            st  = X[:, k]
            con = U[:, k]

            err = st - xref
            obj = obj + ca.mtimes([err.T, Q, err]) + ca.mtimes([con.T, R, con])

            st_next = st + self.T * f(st, con)
            g.append(X[:, k + 1] - st_next)

        # Terminal cost
        errN = X[:, self.N] - xref
        obj  = obj + ca.mtimes([errN.T, Pterm, errN])

        # Pack decision variables into a single vector
        OPT_variables = ca.vertcat(
            ca.reshape(X, n_states * (self.N + 1), 1),
            ca.reshape(U, n_controls * self.N, 1)
        )

        # Define the problem
        nlp_prob = {
            'f': obj,
            'x': OPT_variables,
            'g': ca.vertcat(*g),
            'p': P
        }

        # IPOPT options
        opts = {
            'ipopt': {
                'max_iter': ipopt_max_iter,
                'print_level': (5 if ipopt_print else 0),
                'acceptable_tol': 1e-8,
                'acceptable_obj_change_tol': 1e-6
            },
            'print_time': ipopt_print
        }

        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

        # Equality constraints (g) -> dynamics and x0
        self.lbg = np.zeros(3 * (self.N + 1), dtype=float)
        self.ubg = np.zeros(3 * (self.N + 1), dtype=float)

        # --- BOUNDS ON DECISION VARIABLES ---
        # Partition: [X (states) | U (controls)]
        # -> STATES: unbounded (±inf)
        # -> CONTROLS: v in [v_min, v_max], w in [w_min, w_max]

        # States (3*(N+1)):
        lbx_states = [-np.inf, -np.inf, -np.inf] * (self.N + 1)
        ubx_states = [ np.inf,  np.inf,  np.inf] * (self.N + 1)

        # Controls (2*N):
        lbx_controls = []
        ubx_controls = []
        for _ in range(self.N):
            lbx_controls += [self.v_min, self.w_min]
            ubx_controls += [self.v_max, self.w_max]

        # Concatenate
        self.lbx = np.array(lbx_states + lbx_controls, dtype=float)
        self.ubx = np.array(ubx_states + ubx_controls, dtype=float)

        # Save dims
        self.n_states   = n_states
        self.n_controls = n_controls

    # -------------------------------------------------------------------------
    # Warm-start utilities (solver initialization)
    # -------------------------------------------------------------------------
    def _pack_initial_guess(self, x0_vec):
        """
        Build initial guess by concatenating X0 (state trajectory) and u0 (controls).
        If X0 is None, repeat x0 along the horizon.
        """
        if self.X0 is None:
            X0 = np.tile(x0_vec, (self.N + 1, 1))  # (N+1, 3)
        else:
            X0 = self.X0                           # (N+1, 3)

        x_part = X0.reshape(-1, 1)        # 3*(N+1) x 1
        u_part = self.u0.reshape(-1, 1)   # 2*N     x 1
        return np.vstack([x_part, u_part])

    def shift_warm_start(self, X_sol, U_sol):
        """
        Warm-start strategy: shift the optimal trajectory to the next iteration.
        """
        X_arr = np.array(X_sol).reshape(self.N + 1, self.n_states)   # (N+1, 3)
        U_arr = np.array(U_sol).reshape(self.N,     self.n_controls) # (N, 2)

        # Shift: drop k=0 and repeat the last one
        X_next = np.vstack([X_arr[1:], X_arr[-1:]])
        U_next = np.vstack([U_arr[1:], U_arr[-1:]])

        self.X0 = X_next
        self.u0 = U_next

    # -------------------------------------------------------------------------
    # Main solver call
    # -------------------------------------------------------------------------
    def solve(self, x0_vec, xref_vec):
        """
        Solve one NMPC step.
        Inputs:
            x0_vec   : (3,) initial state
            xref_vec : (3,) reference
        Outputs:
            u_first  : (2,) first optimal control
            X_sol    : (N+1, 3) optimal state trajectory
            U_sol    : (N, 2)   optimal control sequence
        """
        # Problem parameters
        P = np.concatenate([x0_vec, xref_vec], axis=0)

        # Initial guess for [X; U]
        x_init = self._pack_initial_guess(x0_vec)

        # Solve NLP
        try:
            sol = self.solver(
                x0=x_init,
                lbx=self.lbx, ubx=self.ubx,
                lbg=self.lbg, ubg=self.ubg,
                p=P
            )
        except Exception as e:
            # On failure, return zeros to keep the simulation running
            print(f"[WARN] NMPC solve failed: {e}")
            return np.array([0.0, 0.0]), None, None

        # Unpack solution: first X, then U
        x_opt = np.array(sol['x']).squeeze()
        nX    = self.n_states * (self.N + 1)

        X_sol = x_opt[:nX].reshape(self.N + 1, self.n_states)
        U_sol = x_opt[nX: ].reshape(self.N,     self.n_controls)

        # Update warm start (shift)
        self.shift_warm_start(X_sol, U_sol)

        # First control of the horizon
        u_first = U_sol[0]
        return u_first, X_sol, U_sol


# =====[ Helper functions for the loop ]========================================
def extract_xytheta_and_vel(policy_obs: torch.Tensor):
    """
    Extract from obs['policy'] the components we use:
      - x, y, theta -> indices [6, 7, 12]
      - v_x         -> index [0]
    """
    xytheta = policy_obs[:, [6, 7, 12]]
    velx    = policy_obs[:, 0]
    return xytheta, velx


# =====[ Save Plots ]==================================================
def save_plot(history, target_position, env_idx, log_dir, Q, R, P, N):
    """
    Draw 4 subplots: x, y, theta and v_x along the steps.
    """
    if len(history["pos_x"]) == 0:
        return
    
    def fmt_num(x: float) -> str:
        s = f"{float(x):.3f}".rstrip('0').rstrip('.')
        return s.replace('-', 'm').replace('.', 'p') or "0"

    def tag(name: str, M) -> str:
        import numpy as _np
        diag = _np.diag(_np.asarray(M, dtype=float))
        return f"{name}-" + "_".join(fmt_num(v) for v in diag)

    q_tag = tag("Q", Q)     # ex: Q-10_20_0
    r_tag = tag("R", R)     # ex: R-0p1_0p001
    p_tag = tag("P", P)     # ex: P-200_200_0

    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"trajectory_env{env_idx}_N-{N}_{q_tag}_{r_tag}_{p_tag}_{timestamp}.png"
    filepath = os.path.join(log_dir, filename)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    t = list(range(len(history["pos_x"])))

    ax1.plot(t, history["pos_x"], label="x", linewidth=2)
    ax1.axhline(y=target_position[0], color='r', linestyle='--', label="x_ref")
    ax1.set_title("X Position"); ax1.set_xlabel("Step"); ax1.set_ylabel("m")
    ax1.grid(True); ax1.legend()

    ax2.plot(t, history["pos_y"], label="y", linewidth=2, color='orange')
    ax2.axhline(y=target_position[1], color='r', linestyle='--', label="y_ref")
    ax2.set_title("Y Position"); ax2.set_xlabel("Step"); ax2.set_ylabel("m")
    ax2.grid(True); ax2.legend()

    ax3.plot(t, history["theta"], label="theta", linewidth=2, color='green')
    ax3.axhline(y=target_position[2], color='r', linestyle='--', label="theta_ref")
    ax3.set_title("Theta (rad)"); ax3.set_xlabel("Step"); ax3.set_ylabel("rad")
    ax3.grid(True); ax3.legend()

    ax4.plot(t, history["vel_x"], label="v_x", linewidth=2, color='purple')
    ax4.set_title("Linear Velocity X"); ax4.set_xlabel("Step"); ax4.set_ylabel("m/s")
    ax4.grid(True); ax4.legend()

    fig.suptitle(f"Env {env_idx} — Trajectory", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filepath, dpi=150)
    plt.close(fig)

    print(f"[LOG] Saved plot for env {env_idx} -> {filepath}")


# =====[ Main loop ]============================================================
def main():
    # Create env
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    if ENABLE_PLANTATION:
        apply_overrides_train(env_cfg)
        
    env = gym.make(args_cli.task, cfg=env_cfg)

    num_envs = env.unwrapped.num_envs   # type: ignore
    device   = env.unwrapped.device     # type: ignore

    print(f"[INFO] num_envs={num_envs}, device={device}")
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space:      {env.action_space}")

    # NMPC Parameters and Boundaries
    T = 0.1
    N = 7
    v_max, v_min = 1.0, 0.0    # v >= 0
    w_max, w_min = 1.0, -1.0

    nmpc = NMPCMultipleShooting(
        T=T, N=N,
        v_max=v_max, v_min=v_min,
        w_max=w_max, w_min=w_min,
        Q=np.diag([20.0, 20.0, 0.0]),
        R=np.diag([1, 1]),
        P=np.diag([20.0, 20.0, 0.0]),
        ipopt_max_iter=500,
        ipopt_print=False
    )

    # Target (can be replaced by a waypoint generator later)
    target_position = torch.zeros((num_envs, 3), device=device)
    target_position[:, 0] = 2.0     # x pose
    target_position[:, 1] = -2.0    # y pose
    target_position[:, 2] = 0.0     # theta pose (we do not care)
    target_np = target_position.detach().cpu().numpy()

    # Reset / History / Logs
    obs, _ = env.reset()
    LOG_DIR = "/home/nexus/VQ_PMCnmpc/VQ_PMC/logs"

    histories = [{"pos_x": [], "pos_y": [], "theta": [], "vel_x": []}
                 for _ in range(num_envs)]

    delay_steps = 50
    step_counter = 0

    # Simulation loop
    while simulation_app.is_running():
        with torch.inference_mode():
            # zero actions each step (avoid leftovers)
            actions = torch.zeros(env.action_space.shape, device=device)

            # run control only after a small "delay" (give time for static/visual to load)
            if obs is not None and (step_counter >= delay_steps):
                policy_obs = obs["policy"]
                xytheta, velx = extract_xytheta_and_vel(policy_obs)

                # solve NMPC per environment (independently)
                for i in range(num_envs):
                    x0 = xytheta[i].detach().cpu().numpy().astype(float)  # [x, y, theta]
                    xr = target_np[i]                                      # reference

                    u_first, X_sol, U_sol = nmpc.solve(x0, xr)

                    # apply final saturations (extra safety, already enforced in the NLP)
                    v_cmd = float(np.clip(u_first[0], v_min, v_max))
                    w_cmd = float(np.clip(u_first[1], w_min, w_max))

                    actions[i, 0] = v_cmd
                    actions[i, 1] = w_cmd

            # advance simulation
            obs, reward, terminated, truncated, info = env.step(actions)
            step_counter += 1

            # log states/velocities for plotting
            if obs is not None:
                policy_obs = obs["policy"]
                xytheta, velx = extract_xytheta_and_vel(policy_obs)

                xytheta_np = xytheta.detach().cpu().numpy()
                velx_np    = velx.detach().cpu().numpy()
                for i in range(num_envs):
                    histories[i]["pos_x"].append(float(xytheta_np[i, 0]))
                    histories[i]["pos_y"].append(float(xytheta_np[i, 1]))
                    histories[i]["theta"].append(float(xytheta_np[i, 2]))
                    histories[i]["vel_x"].append(float(velx_np[i]))

            # end of episode?
            done_mask = (terminated | truncated)
            if torch.any(done_mask):
                done_np = done_mask.detach().cpu().numpy().astype(bool)

                # save plots for finished envs
                for i, d in enumerate(done_np):
                    if d:
                        save_plot(histories[i], target_np[i], i, LOG_DIR, nmpc.Q, nmpc.R, nmpc.P, nmpc.N)
                        histories[i] = {"pos_x": [], "pos_y": [], "theta": [], "vel_x": []}

                # reset env and solver warm start
                obs, _ = env.reset()
                nmpc.X0 = None
                nmpc.u0[:] = 0.0
                step_counter = 0

    # close
    env.close()


# =====[ Entrypoint ]===========================================================
if __name__ == "__main__":
    main()
    simulation_app.close()
