# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import sys
from pathlib import Path

# add the source folder to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source"))

# import the function
from VQ_PMC.tasks.manager_based.vq_pmc.mdp.plantation import get_plant_initial_states


def apply_overrides_play(env_cfg):

    initial_states = get_plant_initial_states()

    for plant_id, init_state in initial_states.items():
        key = f"plant_{plant_id}"
        if key in env_cfg.scene.object_collection.rigid_objects:
            env_cfg.scene.object_collection.rigid_objects[key].init_state = init_state
        else:
            print(f"[WARNING] Rigid object '{key}' not found in scene; skipping.")


def apply_overrides_train(env_cfg):

    initial_states = get_plant_initial_states()

    for plant_id, init_state in initial_states.items():
        key = f"plant_{plant_id}"
        if key in env_cfg.scene.object_collection.rigid_objects:
            env_cfg.scene.object_collection.rigid_objects[key].init_state = init_state
        else:
            print(f"[WARNING] Rigid object '{key}' not found in scene; skipping.")
