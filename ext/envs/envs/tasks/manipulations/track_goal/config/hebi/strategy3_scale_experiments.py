# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from .Hebi_JointPos_GoalTracking_Env import Strategy3MotorHebi_JointPos_GoalTracking_Env


class Strategy3MotorHebi_JointPos_GoalTracking_pp0dot5_ep1_Env(Strategy3MotorHebi_JointPos_GoalTracking_Env):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.actuators['HEBI'].position_p_scale = 0.5  # type: ignore


class Strategy3MotorHebi_JointPos_GoalTracking_pp1dot5_ep1_Env(Strategy3MotorHebi_JointPos_GoalTracking_Env):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.actuators['HEBI'].position_p_scale = 1.5  # type: ignore


class Strategy3MotorHebi_JointPos_GoalTracking_pp2_ep1_Env(Strategy3MotorHebi_JointPos_GoalTracking_Env):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.actuators['HEBI'].position_p_scale = 2  # type: ignore


class Strategy3MotorHebi_JointPos_GoalTracking_pp5_ep1_Env(Strategy3MotorHebi_JointPos_GoalTracking_Env):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.actuators['HEBI'].position_p_scale = 5  # type: ignore


class Strategy3MotorHebi_JointPos_GoalTracking_pp10_ep1_Env(Strategy3MotorHebi_JointPos_GoalTracking_Env):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.actuators['HEBI'].position_p_scale = 10  # type: ignore


class Strategy3MotorHebi_JointPos_GoalTracking_pp1_ep0dot2_Env(Strategy3MotorHebi_JointPos_GoalTracking_Env):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.actuators['HEBI'].effort_p_scale = 0.2  # type: ignore


class Strategy3MotorHebi_JointPos_GoalTracking_pp1_ep0dot5_Env(Strategy3MotorHebi_JointPos_GoalTracking_Env):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.actuators['HEBI'].effort_p_scale = 0.5  # type: ignore


class Strategy3MotorHebi_JointPos_GoalTracking_pp1_ep1dot5_Env(Strategy3MotorHebi_JointPos_GoalTracking_Env):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.actuators['HEBI'].effort_p_scale = 1.5  # type: ignore


class Strategy3MotorHebi_JointPos_GoalTracking_pp1_ep2_Env(Strategy3MotorHebi_JointPos_GoalTracking_Env):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.actuators['HEBI'].effort_p_scale = 2  # type: ignore


class Strategy3MotorHebi_JointPos_GoalTracking_pp5_ep0dot5_Env(Strategy3MotorHebi_JointPos_GoalTracking_Env):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.actuators['HEBI'].position_p_scale = 5  # type: ignore
        self.scene.robot.actuators['HEBI'].effort_p_scale = 0.5  # type: ignore
