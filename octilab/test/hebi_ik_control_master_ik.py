# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the differential inverse kinematics controller with the simulator.

The differential IK controller can be configured in different modes. It uses the Jacobians computed by
PhysX. This helps perform parallelized computation of the inverse kinematics.

.. code-block:: bash

    # Usage
    ./orbit.sh -p source/standalone/tutorials/05_controllers/ik_control.py

"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the differential IK controller.")
parser.add_argument("--robot", type=str, default="hebi", help="Name of the robot.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import traceback

import carb

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import AssetBaseCfg
from omni.isaac.lab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils.math import subtract_frame_transforms
from controllers.tycho_controller_utils.utils import construct_choppose, construct_command
from controllers.TychoController import TychoController

##
# Pre-defined configs
##
from omni.isaac.lab_assets import FRANKA_PANDA_HIGH_PD_CFG, UR10_CFG  # isort:skip
import numpy as np
def pwm_to_torque(pwm, qvel, gravity=None):
        pwm = np.clip(pwm, -1, 1)
        maxtorque = np.array([23.3, 44.7632, 23.3, 2.66, 2.66, 2.66, 2.66])
        speed_24v = np.array([4.4843, 2.3375, 4.4843, 14.12, 14.12, 14.12, 14.12])
        qvel = qvel
        ctrl = np.multiply(pwm - np.divide(np.abs(qvel), speed_24v), maxtorque)
        if gravity is not None:
            ctrl += gravity
        ctrl = np.clip(ctrl,-maxtorque,maxtorque)
        return ctrl

"""
Helper functions.
"""
def _compute_torques(des_dof_pos, dof_pos, dof_vel, dof_acc, device):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        # actions_scaled = actions * self.control_cfg.action_scale
        # control_type = self.control_cfg.control_type
        control_type = "P"

        # if self.domain_rand_cfg.randomize_gains:
        #     p_gains = self.randomized_p_gains
        #     d_gains = self.randomized_d_gains
        # else:
        #     p_gains = self.p_gains
        #     d_gains = self.d_gains

        p_gains = torch.tensor([7.5000, 17.000000, 15.00000, 30.000000, 15.00000, 18.00000, 20.00000], device=device)
        d_gains = torch.tensor([0.8, 0.38, 3.900, 0.9, 0.3000, 0.33000, 0.5000], device=device)


        if control_type=="P":
            desired_pos = des_dof_pos
            torques = p_gains*(desired_pos - dof_pos) - d_gains*dof_vel
  
        return torques
@configclass
class TableTopSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # mount
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", scale=(2.0, 2.0, 2.0)
        ),
    )

    # articulation
    if args_cli.robot == "franka_panda":
        robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    elif args_cli.robot == "ur10":
        robot = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    elif args_cli.robot == 'hebi':
        robot = HEBI_EXPLICIT_ACTUATOR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    robot = scene["robot"]

    # Create controller
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    ctrl = TychoController("assets/chopstick-gains-7D-all3.xml", False)

    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # Define goals for the arm
    ee_goals = [
        [-0.1, -0.3, 0.1, 0.0, 0.0, 1.0, 0.0],
        [-0.3, -0.1, 0.2, 0.0, 0.0, 1.0, 0.0],
        [-0.3, 0, 0.3, 0.0, 0.0, 1.0, 0.0],
    ]
    ee_goals = torch.tensor(ee_goals, device=sim.device)
    # Track the given command
    current_goal_idx = 0
    # Create buffers to store actions
    ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=robot.device)
    ik_commands[:] = ee_goals[current_goal_idx]

    # Specify robot-specific parameters
    if args_cli.robot == "franka_panda":
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    elif args_cli.robot == "ur10":
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*"], body_names=["ee_link"])
    elif args_cli.robot == "hebi":
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=["HEBI_base_X8_9",
                                                                "HEBI_shoulder_X8_16",
                                                                "HEBI_elbow_X8_9",
                                                                "HEBI_wrist1_X5_1",
                                                                "HEBI_wrist2_X5_1",
                                                                "HEBI_wrist3_X5_1",
                                                                "HEBI_chopstick_X5_1"], body_names=["end_effector"])
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10")
    # Resolving the scene entities
    robot_entity_cfg.resolve(scene)
    # Obtain the frame index of the end-effector
    # For a fixed base robot, the frame index is one less than the body index. This is because
    # the root body is not included in the returned Jacobians.
    if robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0]

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    while simulation_app.is_running():
        # reset
        if count % 450 == 0:
            # reset time
            count = 0
            # reset joint state
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()
            # reset actions
            ik_commands[:] = ee_goals[current_goal_idx]
            joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()
            # reset controller
            diff_ik_controller.reset()
            ee_pos_des, ee_quat_des= diff_ik_controller.set_command(ik_commands)
            # change goal
            current_goal_idx = (current_goal_idx + 1) % len(ee_goals)
        else:
            # obtain quantities from simulation
            jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
            ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
            root_pose_w = robot.data.root_state_w[:, 0:7]
            joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
            # compute frame in root frame
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            # compute the joint commands
            # joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

            chop_angle = torch.tensor([[-0.36]], device='cuda:0')  # Shape (1, 1) to match the existing row
            # Concatenate the new element with the combined tensor
            ee_quat_des_xyzw = ee_quat_des[:, [1, 2, 3, 0]]
            ee_pose_des = torch.cat((ee_pos_des, ee_quat_des_xyzw), dim=1)
            ee_pose_des = torch.cat((ee_pose_des, chop_angle), dim=1)
            joint_pos_des = construct_command(ctrl.arm, joint_pos[0].cpu().numpy(), target_vector=ee_pose_des[0].cpu().numpy())
            joint_pos_des = torch.tensor(joint_pos_des, device = sim.device)
            joint_pos_des = torch.clip(joint_pos_des, -6.25, 6.25)

        # apply actions
        # robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        
        joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
        joint_vel = robot.data.joint_vel[:, robot_entity_cfg.joint_ids]
        joint_effort = robot.data.joint_effort_target[:, robot_entity_cfg.joint_ids]

        joint_pos = joint_pos.cpu().numpy()[0]
        joint_vel = joint_vel.cpu().numpy()[0]
        joint_effort = joint_effort.cpu().numpy()[0]
        
        if isinstance(joint_pos_des, torch.Tensor):
            joint_pos_des = joint_pos_des.squeeze().cpu().tolist()

        
        pwm = ctrl.act(
                    joint_pos_des, 
                    None, 
                    joint_pos, 
                    joint_vel, 
                    joint_effort)
        joint_torque = pwm_to_torque(pwm,joint_vel,None)
            
        # joint_torque = _compute_torques(torch.tensor(joint_pos_des, device = scene.device), joint_pos, joint_vel, joint_effort, scene.device)

        if not isinstance(joint_torque, torch.Tensor):
            joint_torque = torch.tensor(joint_torque, device=scene.device)
        
        robot.set_joint_effort_target(joint_torque)
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        count += 1
        # update buffers
        scene.update(sim_dt)

        # obtain quantities from simulation
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        # update marker positions
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        goal_marker.visualize(ik_commands[:, 0:3] + scene.env_origins, ik_commands[:, 3:7])

def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    # Design scene
    scene_cfg = TableTopSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)

    

if __name__ == "__main__":
    try:
        # run the main execution
        main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        # close sim app
        simulation_app.close()
