# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

from . import mdp

@configclass
class VqPmcEnvCfg(ManagerBasedRLEnvCfg):

    # Scene settings
    scene: mdp.MySceneCfg = mdp.MySceneCfg(num_envs=4, env_spacing=20)

    # Basic settings
    observations: mdp.ObservationsCfg = mdp.ObservationsCfg()
    actions: mdp.ActionsCfg = mdp.ActionsCfg()
    
    # MDP settings
    rewards: mdp.RewardsCfg = mdp.RewardsCfg()
    terminations: mdp.TerminationsCfg = mdp.TerminationsCfg()
    events: mdp.EventCfg = mdp.EventCfg()


    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        # self.scene.contact_forces.update_period = self.sim.dt
        
        
        
        
@configclass
class VqPmcEnvCfg_PLAY(VqPmcEnvCfg):
    """Environment config for PLAY mode (evaluation/demo)."""

    def __post_init__(self) -> None:
        # eredità da VqPmcEnvCfg
        super().__post_init__()

        # riduci numero di ambienti → più leggero da visualizzare
        self.scene.num_envs = 1
        self.scene.env_spacing = 5.0

        # disattiva eventuali corruzioni negli osservabili
        self.observations.policy.enable_corruption = False

        # modifica durata episodio se vuoi
        self.episode_length_s = 30.0

        # (opzionale) togli eventi disturbanti se ne avessi
        # es: self.events.some_random_push = None
        
        self.events.base_external_force_torque = None
        self.events.push_robot = None
