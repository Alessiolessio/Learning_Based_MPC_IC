# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from isaaclab.utils import configclass
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp import base_lin_vel


def reward_forward_velocity(env: ManagerBasedRLEnv, max_velocity: float) -> torch.Tensor:
    """Reward the robot for moving forward along the x-axis (local frame)."""
    v = base_lin_vel(env)[:, 0]
    return torch.clamp(v, max=max_velocity)

@configclass
class RewardsCfg:
    """Single reward term, combining task and penalty with an exponential transformation."""
     # -- task
    reward_forward_velocity: RewTerm = RewTerm(
        func=reward_forward_velocity,
        weight=1.0,
        params={
            "max_velocity": 1.0
        }
    )