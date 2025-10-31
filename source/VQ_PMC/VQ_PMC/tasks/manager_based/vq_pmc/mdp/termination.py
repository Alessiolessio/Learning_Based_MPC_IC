import torch
from isaaclab.utils import configclass
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp import base_lin_vel
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

class VelocityRatioHistory:
    def __init__(self, num_envs: int, buffer_size: int = 5, device: torch.device = "cpu"):
        self.buffer_size = buffer_size
        self.device = device
        self.history = torch.zeros((num_envs, buffer_size), dtype=torch.float32, device=device)
        self.index = torch.zeros(num_envs, dtype=torch.long, device=device)

    def update(self, velocity_ratio: torch.Tensor):
        idx = self.index % self.buffer_size
        for i in range(self.history.shape[0]):
            self.history[i, idx[i]] = velocity_ratio[i]
        self.index += 1

    def average(self):
        return torch.mean(self.history, dim=1)

    def is_full(self):
        return self.index >= self.buffer_size

    def clear(self, mask: torch.Tensor):
        """
        Clears the history buffer for environments where mask is True.
        """
        self.history[mask] = 0.0
        self.index[mask] = 0


# Shared instance (will be initialized at first call)
velocity_history = None


def collision(env: ManagerBasedEnv, threshold: float = 0.1, buffer_size=15) -> torch.Tensor:
    global velocity_history

    if velocity_history is None or velocity_history.history.shape[0] != env.num_envs:
        velocity_history = VelocityRatioHistory(env.num_envs, buffer_size=buffer_size, device=env.device)

    v_cmd = env.action_manager.action[..., 0]
    v_actual = base_lin_vel(env)[..., 0]

    velocity_ratio = torch.abs(v_actual / (v_cmd + 1e-6))
    velocity_history.update(velocity_ratio)

    avg_ratio = velocity_history.average()
    is_moving = torch.abs(v_cmd) > 0.05
    full_mask = velocity_history.is_full().to(env.device)

    # Compute termination condition
    termination = (avg_ratio < threshold) & is_moving & full_mask

    # Clear buffer where termination occurred
    velocity_history.clear(termination)

    return termination

@configclass
class TerminationsCfg:
    """
    Configuration for termination conditions in the MDP.
    """
    time_out = DoneTerm(
        func=mdp.time_out,
        time_out=True
    )

    # REMOVED: removed this if you don't want to reset the env if there is a collision
    # collision = DoneTerm(
    #     func=collision,
    #     time_out=False
    # )