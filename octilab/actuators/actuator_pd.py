# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from omni.isaac.core.utils.types import ArticulationActions

from omni.isaac.lab.actuators.actuator_base import ActuatorBase

if TYPE_CHECKING:
    from .actuator_cfg import HebiMotorCfg

from octilab.controllers.TychoController import TychoController


class HebiMotor(ActuatorBase):
    r"""
    Direct control (DC) motor actuator model with velocity-based saturation model.

    It uses the same model as the :class:`IdealActuator` for computing the torques from input commands.
    However, it implements a saturation model defined by DC motor characteristics.

    A DC motor is a type of electric motor that is powered by direct current electricity. In most cases,
    the motor is connected to a constant source of voltage supply, and the current is controlled by a rheostat.
    Depending on various design factors such as windings and materials, the motor can draw a limited maximum power
    from the electronic source, which limits the produced motor torque and speed.

    A DC motor characteristics are defined by the following parameters:

    * Continuous-rated speed (:math:`\dot{q}_{motor, max}`) : The maximum-rated speed of the motor.
    * Continuous-stall torque (:math:`\tau_{motor, max}`): The maximum-rated torque produced at 0 speed.
    * Saturation torque (:math:`\tau_{motor, sat}`): The maximum torque that can be outputted for a short period.

    Based on these parameters, the instantaneous minimum and maximum torques are defined as follows:

    .. math::

        \tau_{j, max}(\dot{q}) & = clip \left (\tau_{j, sat} \times \left(1 -
            \frac{\dot{q}}{\dot{q}_{j, max}}\right), 0.0, \tau_{j, max} \right) \\
        \tau_{j, min}(\dot{q}) & = clip \left (\tau_{j, sat} \times \left( -1 -
            \frac{\dot{q}}{\dot{q}_{j, max}}\right), - \tau_{j, max}, 0.0 \right)

    where :math:`\gamma` is the gear ratio of the gear box connecting the motor and the actuated joint ends,
    :math:`\dot{q}_{j, max} = \gamma^{-1} \times  \dot{q}_{motor, max}`, :math:`\tau_{j, max} =
    \gamma \times \tau_{motor, max}` and :math:`\tau_{j, peak} = \gamma \times \tau_{motor, peak}`
    are the maximum joint velocity, maximum joint torque and peak torque, respectively. These parameters
    are read from the configuration instance passed to the class.

    Using these values, the computed torques are clipped to the minimum and maximum values based on the
    instantaneous joint velocity:

    .. math::

        \tau_{j, applied} = clip(\tau_{computed}, \tau_{j, min}(\dot{q}), \tau_{j, max}(\dot{q}))

    """

    cfg: HebiMotorCfg
    """The configuration for the actuator model."""

    def __init__(self, cfg: HebiMotorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        # parse configuration
        if self.cfg.gain_xml_path is None:
            raise ValueError("To use HebiMotor you must specify the gain file in HebiMotorCfg")

        if self.cfg.actuator_biasprm is not None:
            self.actuator_biasprm = torch.tensor(self.cfg.actuator_biasprm, device=self._device)
        else:
            raise ValueError("To use HebiMotor you must specify the actuator_biasprm in HebiMotorCfg")
        self.ctrl = TychoController(self.cfg.gain_xml_path, self.cfg.only_position_control)

    """
    Operations.
    """
    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        joint_effort: torch.Tensor
    ) -> ArticulationActions:
        if control_action.joint_positions is None:
            raise ValueError("Hebi pwm motor requires a desired(target) joint position to calculate the \
                            the joint force. You must use action that outputs joint position.")
        elif len(control_action.joint_positions) > 1:
            raise ValueError("Hebi pwm motor does not support vectorized environment processing, \
                             please set num_env to 1 if you want to use HebiMotor")
        else:
            pwm = self.ctrl.act(
                control_action.joint_positions[0].tolist(),
                None,
                joint_pos[0].cpu().numpy(),
                joint_vel[0].cpu().numpy(),
                joint_effort[0].cpu().numpy(),
                )
            pwm = torch.tensor(pwm, device=self._device)
            torque_des = self._pwm_to_torque(pwm, self.actuator_biasprm, joint_vel,)

            # setting below field passed to "data" for record purpose
            self.computed_effort = torque_des
            self.applied_effort = torque_des

            # control_action.joint_efforts is the field will be passed to simulator for physic stepping
            control_action.joint_efforts = self.applied_effort
            control_action.joint_positions = None
            control_action.joint_velocities = None
            return control_action

    def reset(self, env_ids: Sequence[int]):
        pass

    """
    Helper functions.
    """
    def _pwm_to_torque(self, pwm: torch.Tensor, biasprm: torch.Tensor, joint_vel, gravity=None):
        pwm = torch.clip(pwm, -1, 1)
        maxtorque = biasprm[0][0:7]
        speed_24v = biasprm[1][0:7]
        qvel = joint_vel
        ctrl = torch.multiply(pwm - torch.divide(torch.abs(qvel), speed_24v), maxtorque)
        if gravity is not None:
            ctrl += gravity
        ctrl = torch.clip(ctrl, -maxtorque, maxtorque)
        return ctrl
