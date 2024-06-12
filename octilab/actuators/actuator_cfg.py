# Copyright (c) 2022-2024, The Tycho Project Developers

from __future__ import annotations

from dataclasses import MISSING

from omni.isaac.lab.utils import configclass

from . import actuator_pd
from omni.isaac.lab.actuators.actuator_cfg import ActuatorBaseCfg

'''
Orbit Implemented Explicit Actuators for Use

If error happens from the import below, orbit has been updated. This is a sign
for tycho developers to update the reference.
'''

@configclass
class HebiMotorCfg(ActuatorBaseCfg):
    """Configuration for MLP-based actuator model."""
    class_type: type = actuator_pd.HebiMotor

    '''
    We reference the stiffness(P) and damping(D)
    from gain_xml as how it done in tycho_master
    instead of setting them in stiffness and damping field below.
    '''
    stiffness = None
    damping = None

    gain_xml_path: str = MISSING

    only_position_control: bool = False

    actuator_biasprm: list = MISSING
