import socket
import lz4.frame
import numpy as np
from omni.isaac.lab.devices import DeviceBase
from typing import Literal
import json
import torch
from collections.abc import Callable


class Rokoko_Glove(DeviceBase):
    def __init__(self,
                 pos_sensitivity: float = 0.4,
                 rot_sensitivity: float = 0.8,
                 which_hand: Literal["left", "right", "bimanual"] = "right",
                 device="cuda:0"):
        self.device = device
        self._additional_callbacks = dict()
        # Define the IP address and port to listen on
        UDP_IP = "0.0.0.0"  # Listen on all available network interfaces
        UDP_PORT = 14043     # Make sure this matches the port used in Rokoko Studio Live
        self.scale = 1.8
        # Create a UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8192)
        self.sock.bind((UDP_IP, UDP_PORT))

        print(f"Listening for UDP packets on {UDP_IP}:{UDP_PORT}")

        self.left_hand_joint_names = [
            'leftHand', 'leftThumbProximal', 'leftThumbMedial', 'leftThumbDistal', 'leftThumbTip',
            'leftIndexProximal', 'leftIndexMedial', 'leftIndexDistal', 'leftIndexTip',
            'leftMiddleProximal', 'leftMiddleMedial', 'leftMiddleDistal', 'leftMiddleTip',
            'leftRingProximal', 'leftRingMedial', 'leftRingDistal', 'leftRingTip',
            'leftLittleProximal', 'leftLittleMedial', 'leftLittleDistal', 'leftLittleTip']

        self.right_hand_joint_names = [
            'rightHand', 'rightThumbProximal', 'rightThumbMedial', 'rightThumbDistal', 'rightThumbTip',
            'rightIndexProximal', 'rightIndexMedial', 'rightIndexDistal', 'rightIndexTip',
            'rightMiddleProximal', 'rightMiddleMedial', 'rightMiddleDistal', 'rightMiddleTip',
            'rightRingProximal', 'rightRingMedial', 'rightRingDistal', 'rightRingTip',
            'rightLittleProximal', 'rightLittleMedial', 'rightLittleDistal', 'rightLittleTip']

        self.right_hand_joint_dict = {self.right_hand_joint_names[i] : i for i in range(len(self.right_hand_joint_names))}
        self.left_hand_joint_dict = {self.left_hand_joint_names[i] : i for i in range(len(self.left_hand_joint_names))}

        self.left_fingertip_names = ["leftHand", "leftThumbDistal", "leftIndexDistal", "leftMiddleDistal", "leftRingDistal", "leftLittleDistal"]
        self.right_fingertip_names = ["rightHand", "rightThumbDistal", "rightIndexDistal", "rightMiddleDistal", "rightRingDistal", "rightLittleDistal"]
        self.left_finger_dict = {self.left_fingertip_names[i] : i for i in range(len(self.left_fingertip_names))}
        self.right_finger_dict = {self.right_fingertip_names[i] : i for i in range(len(self.right_fingertip_names))}

        self.left_hand_positions = torch.zeros((21, 3), device=self.device)
        self.right_hand_positions = torch.zeros((21, 3), device=self.device)
        self.left_hand_orientations = torch.zeros((21, 4), device=self.device)
        self.right_hand_orientations = torch.zeros((21, 4), device=self.device)

        self.left_hand_positions = torch.zeros((21, 3), device=self.device)
        self.right_hand_positions = torch.zeros((21, 3), device=self.device)
        self.left_hand_orientations = torch.zeros((21, 4), device=self.device)
        self.right_hand_orientations = torch.zeros((21, 4), device=self.device)

        self.left_fingertip_poses = torch.zeros((len(self.left_fingertip_names), 7), device=self.device)
        self.right_fingertip_poses = torch.zeros((len(self.right_fingertip_names), 7), device=self.device)
        self.fingertip_poses = torch.zeros((len(self.left_fingertip_names) + len(self.right_fingertip_names), 7), device=self.device)

    def reset(self):
        pass

    def advance(self):
        data, addr = self.sock.recvfrom(8192)  # Buffer size is 1024 bytes
        decompressed_data = lz4.frame.decompress(data)
        received_json = json.loads(decompressed_data)
        body_data = received_json["scene"]["actors"][0]["body"]

        for joint_name in self.left_fingertip_names:
            joint_data = body_data[joint_name]
            joint_position = torch.tensor(list(joint_data["position"].values()))
            joint_rotation = torch.tensor(list(joint_data["rotation"].values()))
            self.left_fingertip_poses[self.left_finger_dict[joint_name]][:3] = joint_position
            self.left_fingertip_poses[self.left_finger_dict[joint_name]][3:] = joint_rotation

        for joint_name in self.right_fingertip_names:
            joint_data = body_data[joint_name]
            joint_position = torch.tensor(list(joint_data["position"].values()))
            joint_rotation = torch.tensor(list(joint_data["rotation"].values()))
            self.right_fingertip_poses[self.right_finger_dict[joint_name]][:3] = joint_position
            self.right_fingertip_poses[self.right_finger_dict[joint_name]][3:] = joint_rotation

        left_wrist_position = self.left_fingertip_poses[0][:3]
        right_wrist_position = self.right_fingertip_poses[0][:3]

        self.left_fingertip_poses[:, :3] = (self.left_fingertip_poses[:, :3] - left_wrist_position) * self.scale + left_wrist_position
        self.right_fingertip_poses[:, :3] = (self.right_fingertip_poses[:, :3] - right_wrist_position) * self.scale + right_wrist_position
        self.fingertip_poses[:len(self.left_fingertip_names)] = self.left_fingertip_poses
        self.fingertip_poses[len(self.left_fingertip_names):] = self.right_fingertip_poses

        return self.fingertip_poses, True  # True being a placeholder statisfy abstract method

    def add_callback(self, key: str, func: Callable):
        # check keys supported by callback
        if key not in ["L", "R"]:
            raise ValueError(f"Only left (L) and right (R) buttons supported. Provided: {key}.")
        # TODO: Improve this to allow multiple buttons on same key.
        self._additional_callbacks[key] = func

    def normalize_wrt_middle_proximal(self, hand_positions, is_left=True):
        middle_proximal_idx = self.left_hand_joint_names.index('leftMiddleProximal')
        if not is_left:
            middle_proximal_idx = self.right_hand_joint_names.index('rightMiddleProximal')

        wrist_position = hand_positions[0]
        middle_proximal_position = hand_positions[middle_proximal_idx]
        bone_length = np.linalg.norm(wrist_position - middle_proximal_position)
        normalized_hand_positions = (middle_proximal_position - hand_positions) / bone_length
        return normalized_hand_positions
