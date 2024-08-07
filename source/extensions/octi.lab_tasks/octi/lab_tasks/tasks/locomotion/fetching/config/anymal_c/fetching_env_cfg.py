# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass
from ... import fetching_env
##
# Pre-defined configs
##
import octi.lab_assets.anymal as anymal

@configclass
class ActionsCfg:
    actions = anymal.ANYMAL_C_JOINT_POSITION

@configclass
class AnymalCRoughPositionEnvCfg(fetching_env.LocomotionFetchingRoughEnvCfg):
    actions:ActionsCfg = ActionsCfg()
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # switch robot to anymal-c
        self.scene.robot = anymal.ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class AnymalCFlatPositionEnvCfg(AnymalCRoughPositionEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        self.rewards.flat_orientation_l2.weight = -5.0
        self.rewards.dof_torques_l2.weight = -2.5e-5
        self.rewards.feet_air_time.weight = 0.5
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None