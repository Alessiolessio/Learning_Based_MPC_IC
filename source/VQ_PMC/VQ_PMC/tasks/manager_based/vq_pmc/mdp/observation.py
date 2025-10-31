from collections.abc import Sequence
from dataclasses import field

import omni.log

from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg, gaussian_noise, UniformNoiseCfg
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.envs import ManagerBasedEnv

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup

from isaaclab.envs.mdp import base_lin_vel, base_ang_vel

import torch

from typing import List

# ---------------------------------------------------------------------------- #
#                               Custom ActionCfg                               #
# ---------------------------------------------------------------------------- #


@configclass
class UnicycleDriveActionCfg(ActionTermCfg):
    """Config for unicycle-drive-based velocity control (v, ω)."""
    class_type: type = field(default=None)
    wheel_joint_names: List[str] = None
    axle_length: float = 0.4318
    wheel_radius: float = 0.097
    scale: List[float] = None

    def __post_init__(self):
        if self.wheel_joint_names is None:
            self.wheel_joint_names = [
                "front_left_wheel_joint",
                "front_right_wheel_joint",
                "rear_left_wheel_joint",
                "rear_right_wheel_joint",
            ]
        if self.scale is None:
            self.scale = [0.5, 2.0]


# ---------------------------------------------------------------------------- #
#                           UnicycleDriveAction Term                           #
# ---------------------------------------------------------------------------- #

class UnicycleDriveAction(ActionTerm):
    """Maps 2D unicycle commands (v, ω) to 4-wheel velocity targets."""

    cfg: "UnicycleDriveActionCfg"
    _asset: Articulation

    def __init__(self, cfg: UnicycleDriveActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # Resolve wheel joints
        self._wheel_joint_ids, self._wheel_joint_names = self._asset.find_joints(self.cfg.wheel_joint_names)
        if len(self._wheel_joint_ids) != 4:
            raise ValueError(f"Expected 4 wheel joints, got {len(self._wheel_joint_ids)}")

        omni.log.info(f"[UnicycleDriveAction] Using wheels: {self._wheel_joint_names}")

        # Preallocate tensors
        self._raw_actions = torch.zeros(self.num_envs, 2, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._wheel_vel_targets = torch.zeros(self.num_envs, 4, device=self.device)

        self._scale = torch.tensor(self.cfg.scale, device=self.device).unsqueeze(0)
        self._axle_length = self.cfg.axle_length
        self._wheel_radius = self.cfg.wheel_radius
        # NEW: noise config for actions [v, ω]
        # self._act_noise_cfg = GaussianNoiseCfg(mean=0.0, std=0.10, operation="add") # ~0.10 m/s and 0.10 rad/s jitter
        # NEW: bias (constant throughout the episode)
        # self._act_bias = torch.zeros(self.num_envs, 2, device=self.device)
        # NEW: bias (constant throughout the episode)
        # self._act_bias_std = torch.tensor([0.05, 0.02], device=self.device)  # ~0.05 m/s and 0.02 rad/s


    @property
    def action_dim(self) -> int:
        return 2  # [v, ω]

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions = self._raw_actions * self._scale
        # NEW: apply bias + noise per step
        # self._processed_actions = self._processed_actions + self._act_bias
        # NEW: apply noise to already scaled actions
        # self._processed_actions = gaussian_noise(self._processed_actions, self._act_noise_cfg)

    def apply_actions(self):
        v = self._processed_actions[:, 0]
        # v *= 0.0
        # v += 1.0

        omega = self._processed_actions[:, 1]

        # Clip v to be >= 0 (no backward motion)
        v = torch.clamp(v, min=0.0)

        L = self._axle_length
        R = self._wheel_radius

        v_l = (v - omega * L / 2) / R
        v_r = (v + omega * L / 2) / R

        self._wheel_vel_targets[:, 0] = v_l  # front left
        self._wheel_vel_targets[:, 1] = v_r  # front right
        self._wheel_vel_targets[:, 2] = v_l  # rear left
        self._wheel_vel_targets[:, 3] = v_r  # rear right

        self._asset.set_joint_velocity_target(self._wheel_vel_targets, joint_ids=self._wheel_joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0
        # NEW: replace bias only for reseted envs
        # if env_ids is None:
        #     env_ids = slice(None)
        # eps = torch.randn((self.num_envs if isinstance(env_ids, slice) else len(env_ids), 2), device=self.device)
        # self._act_bias[env_ids] = eps * self._act_bias_std


# ---------------------------------------------------------------------------- #
#                             Actions Cfg                                      #
# ---------------------------------------------------------------------------- #

@configclass
class ActionsCfg:
    """2D unicycle (v, ω) action mapped to 4-wheel velocity control."""
    unicycle_drive = UnicycleDriveActionCfg(
        asset_name="robot",
        class_type=UnicycleDriveAction
    )


# ---------------------------------------------------------------------------- #
#                             Observations Cfg                                 #
# ---------------------------------------------------------------------------- #

# REMEMBER: N = number of envelopes
# here we use root instead of base (we consider only the robot's root) and the link reference instead of the center of mass

# EDIT: separate robot_pose into robot_pos and robot_quat to apply noise only in robot_pos (do not touch the quaternions)
def robot_pos(env: ManagerBasedEnv) -> torch.Tensor:
    """
    Returns the robot's pose in the world: position (x, y, z)
    """
    robot = env.scene["robot"]
    return robot.data.root_pos_w    # shape: (N, 3)

# EDIT: separate robot_pose into robot_pos and robot_quat to apply noise only in robot_pos (do not touch the quaternions)
def robot_quat(env: ManagerBasedEnv) -> torch.Tensor:
    """
    Returns the robot's quaternion in the world: orientation (w, x, y, z)
    """
    robot = env.scene["robot"]
    return robot.data.root_quat_w   # shape: (N, 4)


def robot_vel(env: ManagerBasedEnv) -> torch.Tensor:
    """
    Returns the robot velocity (linear and angular) in the world's frame
    """
    robot = env.scene["robot"]
    v = robot.data.root_link_lin_vel_w  # shape: (N,3)
    w = robot.data.root_link_ang_vel_w  # shape: (N,3)
    return torch.cat([v, w], dim=-1)    # shape: (N,6)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # NEW: reasonable Gaussian noise for IMU/odometry: linear vel ~ 0.02 m/s, ang vel ~ 0.02 rad/s
        # sim_velocity = ObsTerm(func=robot_vel, noise=GaussianNoiseCfg(mean=0.0, std=0.02, operation="add"))
        # OLD: no Gaussian noise
        sim_velocity = ObsTerm(func=robot_vel)

        # NEW: position noise ~0.02 m (additive per step)
        # pos = ObsTerm(func=robot_pos, noise=GaussianNoiseCfg(mean=0.0, std=0.02, operation="add"))
        # OLD: no Gaussian noise
        pos = ObsTerm(func=robot_pos)
        
        # NEW: quat separately cause we do not put noise in it
        quat = ObsTerm(func=robot_quat)

        def __post_init__(self):
            # EDIT: enable isaaclab built in corruption/noise
            self.enable_corruption = False   # True=noise / False=nonoise
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()