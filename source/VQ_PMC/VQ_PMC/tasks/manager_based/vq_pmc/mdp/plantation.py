"""
Configuration for plant objects and plantation layout.
Provides utilities to generate initial states and full RigidObjectCollectionCfg.

If you want to disable plantation globally, prefer to NOT create the collection
at all in scene.py (see MySceneCfg). This file still supports `enable=False`,
but we no longer rely on returning an empty collection (that can crash PhysX).
"""

import math
import random
from typing import Dict, List, Tuple
from scipy.spatial.transform import Rotation as R

from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass


@configclass
class Plant:
    """Physical and visual configuration of a plant object."""

    def __init__(
        self,
        usd_path: str,
        scale: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        rigid_props: sim_utils.RigidBodyPropertiesCfg = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        mass_props: sim_utils.MassPropertiesCfg = sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props: sim_utils.CollisionPropertiesCfg = sim_utils.CollisionPropertiesCfg(
            collision_enabled=True, contact_offset=0.1, rest_offset=0.05
        )
    ):
        self.usd_path = usd_path
        self.scale = scale
        self.rigid_props = rigid_props
        self.mass_props = mass_props
        self.collision_props = collision_props

    def __repr__(self):
        return f"Plant(usd_path='{self.usd_path}', scale={self.scale})"


def generate_poses(
    segments: List[Dict[str, float]],
    num_rows: int,
    plant_spacing: float = 0.25,
    row_spacing: float = 0.7,
    z_height: float = 0.0,
    x_offset: float = 0.0,
    random_rotation: bool = True,
) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]]:
    """Generate position and quaternion rotation for each plant."""
    poses = []
    y_offset_center = -((num_rows - 1) * row_spacing) / 2

    for row in range(num_rows):
        y_base = row * row_spacing + y_offset_center
        x_current = x_offset

        for segment in segments:
            count = segment["count"]
            freq = segment["frequency"]
            amp = segment["amplitude"]

            for _ in range(count):  # type: ignore
                x = x_current
                y = y_base + amp * math.sin(freq * x)
                z = z_height

                z_rot_deg = random.uniform(0, 360) if random_rotation else 0.0
                quat = R.from_euler("z", z_rot_deg, degrees=True).as_quat()
                quat_wxyz = (quat[3], quat[0], quat[1], quat[2])

                poses.append(((x, y, z), quat_wxyz))
                x_current += plant_spacing

    return poses


def get_plant_initial_states(
    segments_list: List[str] = ["straight"],
    num_rows: int = 2,
    plant_spacing: float = 0.25,
    row_spacing: float = 0.7,
    z_height: float = 0.0,
    x_offset: float = 0.15,
    random_rotation: bool = True,
) -> Dict[int, RigidObjectCfg.InitialStateCfg]:
    """Generate initial states for plants based on selected patterns."""

    patterns = {
        "straight": {"frequency": 0.0, "amplitude": 0.0},
        "moderate_curve": {"frequency": 1.8, "amplitude": 0.20},
        "slight_curve": {"frequency": 2.7, "amplitude": 0.20},
        "tight_curve": {"frequency": 0.9, "amplitude": 0.30},
    }

    segments = []
    for i, name in enumerate(segments_list):
        if name not in patterns:
            raise ValueError(f"Unknown segment '{name}'. Available options: {list(patterns.keys())}")

        length_m = 12 ** (i + 1)  # geometric growth
        plant_count = int(length_m / plant_spacing)

        cfg = patterns[name]
        segments.append({
            "count": plant_count,
            "frequency": cfg["frequency"],
            "amplitude": cfg["amplitude"]
        })

    poses = generate_poses(
        segments=segments,
        num_rows=num_rows,
        plant_spacing=plant_spacing,
        row_spacing=row_spacing,
        z_height=z_height,
        x_offset=x_offset,
        random_rotation=random_rotation,
    )

    return {
        plant_id: RigidObjectCfg.InitialStateCfg(
            pos=[float(x) for x in pos],
            rot=[float(x) for x in rot],
        )
        for plant_id, (pos, rot) in enumerate(poses)
    }


def create_plant_collection(
    plant: Plant,
    pattern: str = "moderate_curve",
    num_rows: int = 2,
    plant_spacing: float = 0.25,
    row_spacing: float = 0.7,
    z_height: float = 0.0,
    x_offset: float = 0.15,
    random_rotation: bool = True,
    *,
    enable: bool = True,
) -> RigidObjectCollectionCfg:
    """
    Return a RigidObjectCollectionCfg containing all plants for the given pattern.
    Note: prefer to *not call this function* when you want plants disabled.
    Keeping `enable` for backward-compat, but we don't rely on empty collections anymore.
    """
    if not enable:
        # Kept for backward-compat. Avoid using this path (prefer not to create the collection).
        return RigidObjectCollectionCfg(rigid_objects={})

    initial_states = get_plant_initial_states(
        segments_list=[pattern],
        num_rows=num_rows,
        plant_spacing=plant_spacing,
        row_spacing=row_spacing,
        z_height=z_height,
        x_offset=x_offset,
        random_rotation=random_rotation,
    )

    return RigidObjectCollectionCfg(
        rigid_objects={
            f"plant_{plant_id}": RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/Plant_{plant_id}",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=plant.usd_path,
                    scale=plant.scale,
                    rigid_props=plant.rigid_props,
                    mass_props=plant.mass_props,
                    collision_props=plant.collision_props,
                ),
                init_state=initial_states[plant_id],
            )
            for plant_id in initial_states
        }
    )
