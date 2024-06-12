# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from omni.isaac.kit import SimulationApp

kit = SimulationApp({"renderer": "RayTracedLighting", "headless": False})


import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.cloner import GridCloner
from pxr import UsdGeom
import omni.isaac.lab.sim as sim_utils
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.articulations import Articulation

usd_path = "/home/octipus/Projects/FineOrbit/tycho_env-master/tycho_env/assets/hebi.usd"
env_zero_path = "/World/envs/env_0"
num_envs = 1

stage_utils.add_reference_to_stage(usd_path, prim_path=f"{env_zero_path}/hebi")


cloner = GridCloner(spacing=2)
cloner.define_base_env(env_zero_path)
UsdGeom.Xform.Define(stage_utils.get_current_stage(), env_zero_path)
cloner.clone(source_prim_path=env_zero_path, prim_paths=cloner.generate_paths("/World/envs/env", num_envs))

while True:
    kit.update()

