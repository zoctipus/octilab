import socket
import lz4.frame
import numpy as np
from omni.isaac.lab.devices import DeviceBase
import json
import torch
from collections.abc import Callable
from omni.isaac.lab.utils.math import quat_inv, quat_mul


class Rokoko_Glove(DeviceBase):
    def __init__(self, pos_sensitivity: float = 0.4, rot_sensitivity: float = 0.8, device="cuda:0"):
        self.device = device
        self.fingertip_poses = torch.zeros((12, 7), device=self.device)
        self._additional_callbacks = dict()
        # Define the IP address and port to listen on
        UDP_IP = "0.0.0.0"  # Listen on all available network interfaces
        UDP_PORT = 14043     # Make sure this matches the port used in Rokoko Studio Live
        self.scale = 1.8
        # Create a UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8192)
        # self.sock.setblocking(False)
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

        self.left_fingertip_names = ["leftThumbDistal", "leftIndexDistal", "leftMiddleDistal", "leftRingDistal", "leftLittleDistal"]
        self.right_fingertip_names = ["rightThumbDistal", "rightIndexDistal", "rightMiddleDistal", "rightRingDistal", "rightLittleDistal"]
        self.left_fingertip_index = [self.left_hand_joint_names.index(joint_name) for joint_name in self.left_fingertip_names]
        self.right_fingertip_index = [self.right_hand_joint_names.index(joint_name) for joint_name in self.right_fingertip_names]

    def reset(self):
        pass

    def advance(self):
        data, addr = self.sock.recvfrom(8192)  # Buffer size is 1024 bytes
        decompressed_data = lz4.frame.decompress(data)

        received_json = json.loads(decompressed_data)
        # Initialize arrays to store the positions
        left_hand_positions = torch.zeros((21, 3), device=self.device)
        right_hand_positions = torch.zeros((21, 3), device=self.device)

        left_hand_orientations = torch.zeros((21, 4), device=self.device)
        right_hand_orientations = torch.zeros((21, 4), device=self.device)

        # Iterate through the JSON data to extract hand joint positions
        for joint_name in self.left_hand_joint_names:
            joint_data = received_json["scene"]["actors"][0]["body"][joint_name]
            joint_position = torch.tensor(list(joint_data["position"].values()))
            joint_rotation = torch.tensor(list(joint_data["rotation"].values()))
            left_hand_positions[self.left_hand_joint_names.index(joint_name)] = joint_position
            left_hand_orientations[self.left_hand_joint_names.index(joint_name)] = joint_rotation

        for joint_name in self.right_hand_joint_names:
            joint_data = received_json["scene"]["actors"][0]["body"][joint_name]
            joint_position = torch.tensor(list(joint_data["position"].values()))
            joint_rotation = torch.tensor(list(joint_data["rotation"].values()))
            right_hand_positions[self.right_hand_joint_names.index(joint_name)] = joint_position
            right_hand_orientations[self.right_hand_joint_names.index(joint_name)] = joint_rotation

        # relative distance to middle proximal joint
        # normalize by bone distance (distance from wrist to middle proximal)
        # Define the indices of 'middleProximal' in your joint names

        # # Calculate bone length from 'wrist' to 'middleProximal' for both hands
        left_wrist_position = left_hand_positions[0]
        right_wrist_position = right_hand_positions[0]
        left_wrist_orientation = left_hand_orientations[0]
        right_wrist_orientation = right_hand_orientations[0]

        right_index_proximal_position = right_hand_positions[self.right_hand_joint_names.index("rightIndexProximal")]
        right_index_proximal_orientation = right_hand_orientations[self.right_hand_joint_names.index("rightIndexProximal")]

        # left_wrist_orientation_inv = quat_inv(left_wrist_orientation.unsqueeze(0))
        # right_wrist_orientation_inv = quat_inv(right_wrist_orientation.unsqueeze(0))
        for joint_name in self.left_fingertip_names:
            self.fingertip_poses[self.left_fingertip_names.index(joint_name), :3] = (left_hand_positions[self.left_hand_joint_names.index(joint_name)])
            # local_rotation = quat_mul(left_wrist_orientation_inv, left_hand_orientations[self.left_fingertip_names.index(joint_name)].view(1, -1))
            self.fingertip_poses[self.left_fingertip_names.index(joint_name), 3:] = left_hand_orientations[self.left_fingertip_names.index(joint_name)]

        for joint_name in self.right_fingertip_names:
            idx = 5 + self.right_fingertip_names.index(joint_name)
            self.fingertip_poses[idx, :3] = (right_hand_positions[self.right_hand_joint_names.index(joint_name)])
            self.fingertip_poses[idx, :3] = (self.fingertip_poses[idx, :3] - right_wrist_position) * self.scale + right_wrist_position
            self.fingertip_poses[idx, 3:] = right_hand_orientations[self.right_fingertip_names.index(joint_name)]

        self.fingertip_poses[10, :3] = left_wrist_position
        self.fingertip_poses[10, 3:] = left_wrist_orientation
        self.fingertip_poses[11, :3] = right_wrist_position
        self.fingertip_poses[11, 3:] = right_wrist_orientation

        return self.fingertip_poses

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

