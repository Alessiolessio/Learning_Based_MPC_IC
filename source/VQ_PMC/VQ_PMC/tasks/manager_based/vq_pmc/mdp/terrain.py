"""Configuration for custom terrains."""

import isaaclab.sim as sim_utils
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sim.spawners.materials.visual_materials_cfg import PreviewSurfaceCfg

# Rough terrain generator
ROUGH_TERRAINS_ONLY_CFG = TerrainGeneratorCfg(
    size=(40.0, 40.0),
    border_width=1.0,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.2,
    vertical_scale=0.002,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=False,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0,
            noise_range=(0.01, 0.02),
            noise_step=0.005,
            border_width=0.25,
        ),
    },
)


# Full TerrainImporter config for rough terrain
TERRAIN_ROUGH_CFG = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator=ROUGH_TERRAINS_ONLY_CFG,
    collision_group=-1,
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
    ),
    visual_material=PreviewSurfaceCfg(
        diffuse_color=(0.5, 0.3, 0.1),  # RGB
        roughness=0.9,
        metallic=0.0,
    ),
    debug_vis=True,
)


# Full TerrainImporter config for plane
TERRAIN_PLANE_CFG = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="plane",
    collision_group=-1,
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
    ),
    debug_vis=True,
)
