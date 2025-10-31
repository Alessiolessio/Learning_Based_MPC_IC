import os

import isaaclab.sim as sim_utils

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from .terrain import TERRAIN_PLANE_CFG, TERRAIN_ROUGH_CFG
from .plantation import Plant, create_plant_collection
from .robot import TERRASENTIA_CFG


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SORGHUM_PATH = os.path.join(CURRENT_DIR, "../asset/plant/sorghum/sorghum.usdc")

# Plantation
# Options: True / False
ENABLE_PLANTATION = False

# Single Plant descriptor used by the collection builder
plant = Plant(usd_path=SORGHUM_PATH)


@configclass
class MySceneCfg(InteractiveSceneCfg):

    # Terrain configuration
    # Options: TERRAIN_PLANE_CFG / TERRAIN_ROUGH_CFG
    terrain = TERRAIN_PLANE_CFG

    # Robot
    robot: ArticulationCfg = TERRASENTIA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Only create the plants collection when enabled.
    if ENABLE_PLANTATION:
        object_collection = create_plant_collection(
            plant=plant,
            pattern="moderate_curve",   # straight / slight_curve / moderate_curve / tight_curve
            num_rows=2,
            plant_spacing=0.25,
            row_spacing=0.7,
            z_height=0.0,
            x_offset=0.15,
            random_rotation=True,
            enable=True,
        )

    # lights -> TODO: there is a problem with those lights: the previous prims don't get deleted!!!
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=20.0,
            exposure=2.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
