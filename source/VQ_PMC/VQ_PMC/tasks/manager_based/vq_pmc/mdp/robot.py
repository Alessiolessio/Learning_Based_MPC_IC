"""Configuration for the Terrasentia 4-wheel robot using DifferentialDriveAction."""
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import Articulation, ArticulationCfg

# Path to the USD file
current_dir = os.path.dirname(os.path.abspath(__file__))
TERRASENTIA_USD_PATH = os.path.join(current_dir, '../asset', 'robot', 'terrasentia', 'terrasentia.usd')


class TolerantArticulation(Articulation):
    """An articulation that ignores joint limit validation errors."""

    def _validate_cfg(self):
        """Override to skip the joint limit validation."""
        pass


##
# Configuration - Actuators.
##

TERRASENTIA_WHEEL_ACTUATOR_CFG = DCMotorCfg(
    joint_names_expr=[
        "front_left_wheel_joint",
        "front_right_wheel_joint",
        "rear_left_wheel_joint",
        "rear_right_wheel_joint",
    ],
    saturation_effort=15.0,
    effort_limit=10.0,
    velocity_limit=15.0,
    stiffness={".*": 0.0},
    damping={".*": 1.0},
)
"""DC motor configuration for the Terrasentia robot's wheel joints."""

##
# Configuration - Articulation.
##

TERRASENTIA_CFG = ArticulationCfg(
    class_type=TolerantArticulation,
    spawn=sim_utils.UsdFileCfg(
        usd_path=TERRASENTIA_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.1),
        joint_pos={
            "front_left_leg_joint": 0.520,
            "front_right_leg_joint": 0.520,
            "rear_left_leg_joint": -0.520,
            "rear_right_leg_joint": -0.520,
        },
    ),
    actuators={"wheels": TERRASENTIA_WHEEL_ACTUATOR_CFG},
    soft_joint_pos_limit_factor=1,
)
