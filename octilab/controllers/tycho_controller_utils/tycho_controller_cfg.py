# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from omni.isaac.lab.utils import configclass

from ..TychoController import TychoController


@configclass
class TychoControllerCfg:
    """Configuration for differential inverse kinematics controller."""

    class_type: type = TychoController
    """The associated controller class."""

    gain_path: str = MISSING

    onlyPositionCtrl: bool = MISSING

    biasprm: list = []
