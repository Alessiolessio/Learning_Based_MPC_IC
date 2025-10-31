from isaaclab.utils import configclass
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv

import torch


def reset_state_environment(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, float | tuple[float, float] | list[float]],
    reset_velocities: dict[str, list[float]],
    plantation_params: dict[str, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    # extract the robot asset and get the root states
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    root_states = asset.data.default_root_state[env_ids].clone()

    # custom robot position sampling
    x_values = torch.tensor(pose_range.get("x", [0.0, 0.0]), device=asset.device)
    x_samples = x_values[torch.randint(0, len(x_values), (len(env_ids),), device=asset.device)]
    keys_no_yaw = ["y", "z", "roll", "pitch"]
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in keys_no_yaw]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_no_yaw = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), len(keys_no_yaw)), device=asset.device)
    yaw_max = pose_range.get("yaw", 0.0)
    yaw_max = torch.tensor(yaw_max, device=asset.device)
    yaw_low = torch.where(x_samples == 0.0, torch.zeros_like(x_samples), -yaw_max)
    yaw_high = torch.where(x_samples == 0.0, yaw_max, torch.zeros_like(x_samples))
    yaw_samples = math_utils.sample_uniform(yaw_low, yaw_high, (len(env_ids),), device=asset.device)
    rand_samples = torch.cat([x_samples.unsqueeze(1), rand_no_yaw, yaw_samples.unsqueeze(1)], dim=1)
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)

    # set up the plantation curve
    frequency = plantation_params["frequency"]
    amplitude = plantation_params["amplitude"]
    y_curve = amplitude * torch.sin(frequency * x_samples)  # calculate y from x using the curve
    position_x = root_states[:, 0] + env.scene.env_origins[env_ids, 0] + x_samples
    position_y = root_states[:, 1] + env.scene.env_origins[env_ids, 1] + y_curve
    position_z = root_states[:, 2] + env.scene.env_origins[env_ids, 2] + rand_samples[:, 2]

    # Setup the new robots and plantation position at reset
    position_assets = torch.stack([position_x, position_y, position_z], dim=-1)
    asset.write_root_pose_to_sim(torch.cat([position_assets, orientations], dim=-1), env_ids=env_ids)

    # Root velocity: (vx, vy, vz, wx, wy, wz)
    root_vel = torch.tensor(reset_velocities["root_velocities"], device=asset.device)
    asset.write_root_velocity_to_sim(root_vel, env_ids=env_ids)

    # Joint velocities -> note that all velocities here are setup with default values
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=reset_state_environment,
        mode="reset",
        params={
            "pose_range": {
                "x": [0.0],
                "z": (0.1, 0.12),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": 0.0,
            },
            "reset_velocities": {
                "root_velocities": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # velocities
            },
            "plantation_params": {
                "frequency": 0.0,
                "amplitude": 0.0,
            },
        },
    )
