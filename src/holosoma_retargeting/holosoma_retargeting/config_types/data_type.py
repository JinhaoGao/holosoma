# ruff: noqa: CPY001

"""Configuration types for motion data format."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from holosoma_retargeting.config_types.robot import (
    RobotDefaults,
    _default_robot_defaults,
    _validate_robot_type,
    robot_family,
)
from holosoma_retargeting.orientation_calibration import (
    ORIENTATION_ALIGNMENT_QUATERNIONS_WXYZ,
)

# Pre-defined constants for each data format
LAFAN_DEMO_JOINTS = [
    "Hips",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToeBase",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToeBase",
    "Spine",
    "Spine1",
    "Spine2",
    "Neck",
    "Head",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
]

# Noetix has an independent source format and loader. Its normalized output
# intentionally uses the same 22 target labels so the proven robot mapping can
# be reused without classifying company data as public LAFAN data.
NOETIX_MOCAP_DEMO_JOINTS = LAFAN_DEMO_JOINTS.copy()

SMPLH_DEMO_JOINTS = [
    "Pelvis",
    "L_Hip",
    "L_Knee",
    "L_Ankle",
    "L_Toe",
    "R_Hip",
    "R_Knee",
    "R_Ankle",
    "R_Toe",
    "Torso",
    "Spine",
    "Chest",
    "Neck",
    "Head",
    "L_Thorax",
    "L_Shoulder",
    "L_Elbow",
    "L_Wrist",
    "L_Index1",
    "L_Index2",
    "L_Index3",
    "L_Middle1",
    "L_Middle2",
    "L_Middle3",
    "L_Pinky1",
    "L_Pinky2",
    "L_Pinky3",
    "L_Ring1",
    "L_Ring2",
    "L_Ring3",
    "L_Thumb1",
    "L_Thumb2",
    "L_Thumb3",
    "R_Thorax",
    "R_Shoulder",
    "R_Elbow",
    "R_Wrist",
    "R_Index1",
    "R_Index2",
    "R_Index3",
    "R_Middle1",
    "R_Middle2",
    "R_Middle3",
    "R_Pinky1",
    "R_Pinky2",
    "R_Pinky3",
    "R_Ring1",
    "R_Ring2",
    "R_Ring3",
    "R_Thumb1",
    "R_Thumb2",
    "R_Thumb3",
]
OMOMO_DEMO_JOINTS = SMPLH_DEMO_JOINTS

MOCAP_DEMO_JOINTS = [
    "Hips",
    "Spine",
    "Spine1",
    "Neck",
    "Head",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "LeftHandThumb1",
    "LeftHandThumb2",
    "LeftHandThumb3",
    "LeftHandIndex1",
    "LeftHandIndex2",
    "LeftHandIndex3",
    "LeftHandMiddle1",
    "LeftHandMiddle2",
    "LeftHandMiddle3",
    "LeftHandRing1",
    "LeftHandRing2",
    "LeftHandRing3",
    "LeftHandPinky1",
    "LeftHandPinky2",
    "LeftHandPinky3",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "RightHandThumb1",
    "RightHandThumb2",
    "RightHandThumb3",
    "RightHandIndex1",
    "RightHandIndex2",
    "RightHandIndex3",
    "RightHandMiddle1",
    "RightHandMiddle2",
    "RightHandMiddle3",
    "RightHandRing1",
    "RightHandRing2",
    "RightHandRing3",
    "RightHandPinky1",
    "RightHandPinky2",
    "RightHandPinky3",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToeBase",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToeBase",
    "LeftFootMod",
    "RightFootMod",
]

# FBX motion is normalized to the same canonical joint labels and topology as
# mocap, but remains a separate format so its mappings and frame calibration
# cannot accidentally inherit the legacy mocap convention.
FBX_MOCAP_DEMO_JOINTS = MOCAP_DEMO_JOINTS.copy()

SMPLX_DEMO_JOINTS = [
    "Pelvis",
    "L_Hip",
    "R_Hip",
    "Spine1",
    "L_Knee",
    "R_Knee",
    "Spine2",
    "L_Ankle",
    "R_Ankle",
    "Spine3",
    "L_Foot",
    "R_Foot",
    "Neck",
    "L_Collar",
    "R_Collar",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
]

AMASS_DEMO_JOINTS = SMPLX_DEMO_JOINTS
GVHMR_DEMO_JOINTS = SMPLX_DEMO_JOINTS

# Joint mappings - organized by (data_format, robot_type)
JOINTS_MAPPINGS = {
    ("lafan", "g1"): {
        "Hips": "pelvis",
        "LeftUpLeg": "left_hip_yaw_link",
        "RightUpLeg": "right_hip_yaw_link",
        "LeftLeg": "left_knee_link",
        "RightLeg": "right_knee_link",
        "LeftArm": "left_shoulder_yaw_link",
        "RightArm": "right_shoulder_yaw_link",
        "LeftForeArm": "left_elbow_link",
        "RightForeArm": "right_elbow_link",
        "LeftFoot": "left_ankle_roll_link",
        "RightFoot": "right_ankle_roll_link",
        "LeftToeBase": "left_ankle_roll_sphere_5_link",
        "RightToeBase": "right_ankle_roll_sphere_5_link",
        "LeftHand": "left_rubber_hand_link",
        "RightHand": "right_rubber_hand_link",
    },
    ("lafan", "t1"): {
        "Spine1": "Trunk",
        "LeftUpLeg": "Hip_Pitch_Left",
        "RightUpLeg": "Hip_Pitch_Right",
        "LeftLeg": "Shank_Left",
        "RightLeg": "Shank_Right",
        "LeftArm": "AL1",
        "RightArm": "AR1",
        "LeftForeArm": "left_hand_link",
        "RightForeArm": "right_hand_link",
        "LeftFoot": "Ankle_Cross_Left",
        "RightFoot": "Ankle_Cross_Right",
        "LeftToeBase": "left_foot_sphere_5_link",
        "RightToeBase": "right_foot_sphere_5_link",
        "LeftHand": "left_hand_sphere_link",
        "RightHand": "right_hand_sphere_link",
    },
    ("lafan", "e1"): {
        "Hips": "base_link",
        "LeftUpLeg": "l_leg_hip_yaw_link",
        "RightUpLeg": "r_leg_hip_yaw_link",
        "LeftLeg": "l_leg_knee_link",
        "RightLeg": "r_leg_knee_link",
        "LeftArm": "l_arm_shoulder_yaw_link",
        "RightArm": "r_arm_shoulder_yaw_link",
        "LeftForeArm": "l_arm_elbow_pitch_link",
        "RightForeArm": "r_arm_elbow_pitch_link",
        "LeftFoot": "l_leg_ankle_roll_link",
        "RightFoot": "r_leg_ankle_roll_link",
        "LeftToeBase": "l_foot_sphere_5_link",
        "RightToeBase": "r_foot_sphere_5_link",
        "LeftHand": "l_hand_sphere_link",
        "RightHand": "r_hand_sphere_link",
    },
    ("lafan", "e2"): {
        "Hips": "base_link",
        "LeftUpLeg": "l_leg_hip_yaw_link",
        "RightUpLeg": "r_leg_hip_yaw_link",
        "LeftLeg": "l_leg_knee_link",
        "RightLeg": "r_leg_knee_link",
        "LeftArm": "l_arm_shoulder_roll_link",
        "RightArm": "r_arm_shoulder_roll_link",
        "LeftForeArm": "l_arm_elbow_link",
        "RightForeArm": "r_arm_elbow_link",
        "LeftFoot": "l_leg_ankle_roll_link",
        "RightFoot": "r_leg_ankle_roll_link",
        "LeftToeBase": "l_foot_sphere_5_link",
        "RightToeBase": "r_foot_sphere_5_link",
        "LeftHand": "l_hand_sphere_link",
        "RightHand": "r_hand_sphere_link",
    },
    ("noetix_mocap", "g1"): {
        "Hips": "pelvis",
        "LeftUpLeg": "left_hip_pitch_link",
        "RightUpLeg": "right_hip_pitch_link",
        "LeftLeg": "left_knee_link",
        "RightLeg": "right_knee_link",
        "LeftArm": "left_shoulder_yaw_link",
        "RightArm": "right_shoulder_yaw_link",
        "LeftForeArm": "left_elbow_link",
        "RightForeArm": "right_elbow_link",
        "LeftFoot": "left_ankle_roll_link",
        "RightFoot": "right_ankle_roll_link",
        "LeftToeBase": "left_ankle_roll_sphere_5_link",
        "RightToeBase": "right_ankle_roll_sphere_5_link",
        "LeftHand": "left_rubber_hand_link",
        "RightHand": "right_rubber_hand_link",
    },
    ("noetix_mocap", "e1"): {
        "Hips": "base_link",
        "LeftUpLeg": "l_leg_hip_pitch_link",
        "RightUpLeg": "r_leg_hip_pitch_link",
        "LeftLeg": "l_leg_knee_link",
        "RightLeg": "r_leg_knee_link",
        "LeftArm": "l_arm_shoulder_yaw_link",
        "RightArm": "r_arm_shoulder_yaw_link",
        "LeftForeArm": "l_arm_elbow_pitch_link",
        "RightForeArm": "r_arm_elbow_pitch_link",
        "LeftFoot": "l_leg_ankle_roll_link",
        "RightFoot": "r_leg_ankle_roll_link",
        "LeftToeBase": "l_foot_sphere_5_link",
        "RightToeBase": "r_foot_sphere_5_link",
        "LeftHand": "l_hand_sphere_link",
        "RightHand": "r_hand_sphere_link",
    },
    ("noetix_mocap", "e2"): {
        "Hips": "base_link",
        "LeftUpLeg": "l_leg_hip_pitch_link",
        "RightUpLeg": "r_leg_hip_pitch_link",
        "LeftLeg": "l_leg_knee_link",
        "RightLeg": "r_leg_knee_link",
        "LeftArm": "l_arm_shoulder_roll_link",
        "RightArm": "r_arm_shoulder_roll_link",
        "LeftForeArm": "l_arm_elbow_link",
        "RightForeArm": "r_arm_elbow_link",
        "LeftFoot": "l_leg_ankle_roll_link",
        "RightFoot": "r_leg_ankle_roll_link",
        "LeftToeBase": "l_foot_sphere_5_link",
        "RightToeBase": "r_foot_sphere_5_link",
        "LeftHand": "l_hand_sphere_link",
        "RightHand": "r_hand_sphere_link",
    },
    ("fbx_mocap", "g1"): {
        "Spine": "pelvis",
        "LeftUpLeg": "left_hip_pitch_link",
        "RightUpLeg": "right_hip_pitch_link",
        "LeftLeg": "left_knee_link",
        "RightLeg": "right_knee_link",
        "LeftArm": "left_shoulder_roll_link",
        "RightArm": "right_shoulder_roll_link",
        "LeftForeArm": "left_elbow_link",
        "RightForeArm": "right_elbow_link",
        "LeftFoot": "left_ankle_roll_link",
        "RightFoot": "right_ankle_roll_link",
        "LeftToeBase": "left_ankle_roll_sphere_5_link",
        "RightToeBase": "right_ankle_roll_sphere_5_link",
        "LeftHand": "left_rubber_hand_link",
        "RightHand": "right_rubber_hand_link",
    },
    ("fbx_mocap", "e1"): {
        "Spine": "base_link",
        "LeftUpLeg": "l_leg_hip_pitch_link",
        "RightUpLeg": "r_leg_hip_pitch_link",
        "LeftLeg": "l_leg_knee_link",
        "RightLeg": "r_leg_knee_link",
        "LeftArm": "l_arm_shoulder_roll_link",
        "RightArm": "r_arm_shoulder_roll_link",
        "LeftForeArm": "l_arm_elbow_pitch_link",
        "RightForeArm": "r_arm_elbow_pitch_link",
        "LeftFoot": "l_leg_ankle_roll_link",
        "RightFoot": "r_leg_ankle_roll_link",
        "LeftToeBase": "l_foot_sphere_5_link",
        "RightToeBase": "r_foot_sphere_5_link",
        "LeftHand": "l_hand_sphere_link",
        "RightHand": "r_hand_sphere_link",
    },
    ("fbx_mocap", "e2"): {
        "Spine": "base_link",
        "LeftUpLeg": "l_leg_hip_pitch_link",
        "RightUpLeg": "r_leg_hip_pitch_link",
        "LeftLeg": "l_leg_knee_link",
        "RightLeg": "r_leg_knee_link",
        "LeftArm": "l_arm_shoulder_roll_link",
        "RightArm": "r_arm_shoulder_roll_link",
        "LeftForeArm": "l_arm_elbow_link",
        "RightForeArm": "r_arm_elbow_link",
        "LeftFoot": "l_leg_ankle_roll_link",
        "RightFoot": "r_leg_ankle_roll_link",
        "LeftToeBase": "l_foot_sphere_5_link",
        "RightToeBase": "r_foot_sphere_5_link",
        "LeftHand": "l_hand_sphere_link",
        "RightHand": "r_hand_sphere_link",
    },
    ("omomo", "g1"): {
        "Pelvis": "pelvis",
        "L_Hip": "left_hip_pitch_link",
        "R_Hip": "right_hip_pitch_link",
        "L_Knee": "left_knee_link",
        "R_Knee": "right_knee_link",
        "L_Shoulder": "left_shoulder_yaw_link",
        "R_Shoulder": "right_shoulder_yaw_link",
        "L_Elbow": "left_elbow_link",
        "R_Elbow": "right_elbow_link",
        "L_Ankle": "left_ankle_roll_link",
        "R_Ankle": "right_ankle_roll_link",
        "L_Toe": "left_ankle_roll_sphere_5_link",
        "R_Toe": "right_ankle_roll_sphere_5_link",
        "L_Wrist": "left_rubber_hand_link",
        "R_Wrist": "right_rubber_hand_link",
    },
    ("omomo", "t1"): {
        "Pelvis": "Trunk",
        "L_Hip": "Hip_Pitch_Left",
        "R_Hip": "Hip_Pitch_Right",
        "L_Knee": "Shank_Left",
        "R_Knee": "Shank_Right",
        "L_Shoulder": "AL1",
        "R_Shoulder": "AR1",
        "L_Elbow": "left_hand_link",
        "R_Elbow": "right_hand_link",
        "L_Ankle": "Ankle_Cross_Left",
        "R_Ankle": "Ankle_Cross_Right",
        "L_Toe": "left_foot_sphere_5_link",
        "R_Toe": "right_foot_sphere_5_link",
        "L_Wrist": "left_hand_sphere_link",
        "R_Wrist": "right_hand_sphere_link",
    },
    ("omomo", "e1"): {
        "Pelvis": "base_link",
        "L_Hip": "l_leg_hip_pitch_link",
        "R_Hip": "r_leg_hip_pitch_link",
        "L_Knee": "l_leg_knee_link",
        "R_Knee": "r_leg_knee_link",
        "L_Shoulder": "l_arm_shoulder_yaw_link",
        "R_Shoulder": "r_arm_shoulder_yaw_link",
        "L_Elbow": "l_arm_elbow_pitch_link",
        "R_Elbow": "r_arm_elbow_pitch_link",
        "L_Ankle": "l_leg_ankle_roll_link",
        "R_Ankle": "r_leg_ankle_roll_link",
        "L_Toe": "l_foot_sphere_5_link",
        "R_Toe": "r_foot_sphere_5_link",
        "L_Wrist": "l_hand_sphere_link",
        "R_Wrist": "r_hand_sphere_link",
    },
    ("omomo", "e2"): {
        "Pelvis": "base_link",
        "L_Hip": "l_leg_hip_pitch_link",
        "R_Hip": "r_leg_hip_pitch_link",
        "L_Knee": "l_leg_knee_link",
        "R_Knee": "r_leg_knee_link",
        "L_Shoulder": "l_arm_shoulder_roll_link",
        "R_Shoulder": "r_arm_shoulder_roll_link",
        "L_Elbow": "l_arm_elbow_link",
        "R_Elbow": "r_arm_elbow_link",
        "L_Ankle": "l_leg_ankle_roll_link",
        "R_Ankle": "r_leg_ankle_roll_link",
        "L_Toe": "l_foot_sphere_5_link",
        "R_Toe": "r_foot_sphere_5_link",
        "L_Wrist": "l_hand_sphere_link",
        "R_Wrist": "r_hand_sphere_link",
    },
    ("amass", "g1"): {
        "Pelvis": "pelvis",
        "L_Hip": "left_hip_roll_link",
        "R_Hip": "right_hip_roll_link",
        "L_Knee": "left_knee_link",
        "R_Knee": "right_knee_link",
        "L_Shoulder": "left_shoulder_yaw_link",
        "R_Shoulder": "right_shoulder_yaw_link",
        "L_Elbow": "left_elbow_link",
        "R_Elbow": "right_elbow_link",
        "L_Ankle": "left_ankle_roll_link",
        "R_Ankle": "right_ankle_roll_link",
        "L_Foot": "left_ankle_roll_sphere_5_link",
        "R_Foot": "right_ankle_roll_sphere_5_link",
        "L_Wrist": "left_rubber_hand_link",
        "R_Wrist": "right_rubber_hand_link",
    },
    ("amass", "e1"): {
        "Pelvis": "base_link",
        "L_Hip": "l_leg_hip_roll_link",
        "R_Hip": "r_leg_hip_roll_link",
        "L_Knee": "l_leg_knee_link",
        "R_Knee": "r_leg_knee_link",
        "L_Shoulder": "l_arm_shoulder_yaw_link",
        "R_Shoulder": "r_arm_shoulder_yaw_link",
        "L_Elbow": "l_arm_elbow_pitch_link",
        "R_Elbow": "r_arm_elbow_pitch_link",
        "L_Ankle": "l_leg_ankle_roll_link",
        "R_Ankle": "r_leg_ankle_roll_link",
        "L_Foot": "l_foot_sphere_5_link",
        "R_Foot": "r_foot_sphere_5_link",
        "L_Wrist": "l_hand_sphere_link",
        "R_Wrist": "r_hand_sphere_link",
    },
    ("amass", "e2"): {
        "Pelvis": "base_link",
        "L_Hip": "l_leg_hip_roll_link",
        "R_Hip": "r_leg_hip_roll_link",
        "L_Knee": "l_leg_knee_link",
        "R_Knee": "r_leg_knee_link",
        "L_Shoulder": "l_arm_shoulder_roll_link",
        "R_Shoulder": "r_arm_shoulder_roll_link",
        "L_Elbow": "l_arm_elbow_link",
        "R_Elbow": "r_arm_elbow_link",
        "L_Ankle": "l_leg_ankle_roll_link",
        "R_Ankle": "r_leg_ankle_roll_link",
        "L_Foot": "l_foot_sphere_5_link",
        "R_Foot": "r_foot_sphere_5_link",
        "L_Wrist": "l_hand_sphere_link",
        "R_Wrist": "r_hand_sphere_link",
    },
    ("gvhmr", "g1"): {
        "Pelvis": "pelvis",
        "L_Hip": "left_hip_roll_link",
        "R_Hip": "right_hip_roll_link",
        "L_Knee": "left_knee_link",
        "R_Knee": "right_knee_link",
        "L_Shoulder": "left_shoulder_yaw_link",
        "R_Shoulder": "right_shoulder_yaw_link",
        "L_Elbow": "left_elbow_link",
        "R_Elbow": "right_elbow_link",
        "L_Ankle": "left_ankle_roll_link",
        "R_Ankle": "right_ankle_roll_link",
        "L_Foot": "left_ankle_roll_sphere_5_link",
        "R_Foot": "right_ankle_roll_sphere_5_link",
        "L_Wrist": "left_rubber_hand_link",
        "R_Wrist": "right_rubber_hand_link",
    },
    ("gvhmr", "e1"): {
        "Pelvis": "base_link",
        "L_Hip": "l_leg_hip_roll_link",
        "R_Hip": "r_leg_hip_roll_link",
        "L_Knee": "l_leg_knee_link",
        "R_Knee": "r_leg_knee_link",
        "L_Shoulder": "l_arm_shoulder_yaw_link",
        "R_Shoulder": "r_arm_shoulder_yaw_link",
        "L_Elbow": "l_arm_elbow_pitch_link",
        "R_Elbow": "r_arm_elbow_pitch_link",
        "L_Ankle": "l_leg_ankle_roll_link",
        "R_Ankle": "r_leg_ankle_roll_link",
        "L_Foot": "l_foot_sphere_5_link",
        "R_Foot": "r_foot_sphere_5_link",
        "L_Wrist": "l_hand_sphere_link",
        "R_Wrist": "r_hand_sphere_link",
    },
    ("gvhmr", "e2"): {
        "Pelvis": "base_link",
        "L_Hip": "l_leg_hip_roll_link",
        "R_Hip": "r_leg_hip_roll_link",
        "L_Knee": "l_leg_knee_link",
        "R_Knee": "r_leg_knee_link",
        "L_Shoulder": "l_arm_shoulder_roll_link",
        "R_Shoulder": "r_arm_shoulder_roll_link",
        "L_Elbow": "l_arm_elbow_link",
        "R_Elbow": "r_arm_elbow_link",
        "L_Ankle": "l_leg_ankle_roll_link",
        "R_Ankle": "r_leg_ankle_roll_link",
        "L_Foot": "l_foot_sphere_5_link",
        "R_Foot": "r_foot_sphere_5_link",
        "L_Wrist": "l_hand_sphere_link",
        "R_Wrist": "r_hand_sphere_link",
    },
    ("mocap", "g1"): {
        "Hips": "pelvis",
        "LeftUpLeg": "left_hip_pitch_link",
        "RightUpLeg": "right_hip_pitch_link",
        "LeftLeg": "left_knee_link",
        "RightLeg": "right_knee_link",
        "LeftFoot": "left_ankle_roll_link",
        "RightFoot": "right_ankle_roll_link",
        "LeftToeBase": "left_ankle_roll_sphere_5_link",
        "RightToeBase": "right_ankle_roll_sphere_5_link",
        "LeftArm": "left_shoulder_yaw_link",
        "RightArm": "right_shoulder_yaw_link",
        "LeftForeArm": "left_elbow_link",
        "RightForeArm": "right_elbow_link",
        "LeftHand": "left_rubber_hand_link",
        "RightHand": "right_rubber_hand_link",
    },
    ("mocap", "t1"): {
        "Spine1": "Trunk",
        "LeftUpLeg": "Hip_Pitch_Left",
        "LeftLeg": "Shank_Left",
        "LeftToeBase": "left_foot_sphere_5_link",
        "RightUpLeg": "Hip_Pitch_Right",
        "RightLeg": "Shank_Right",
        "RightToeBase": "right_foot_sphere_5_link",
        "LeftArm": "AL1",
        "LeftForeArm": "left_hand_link",
        "LeftHandMiddle3": "left_hand_sphere_link",
        "RightArm": "AR1",
        "RightForeArm": "right_hand_link",
        "RightHandMiddle3": "right_hand_sphere_link",
        "LeftFoot": "Ankle_Cross_Left",
        "RightFoot": "Ankle_Cross_Right",
    },
    ("mocap", "e1"): {
        "Hips": "base_link",
        "LeftUpLeg": "l_leg_hip_pitch_link",
        "RightUpLeg": "r_leg_hip_pitch_link",
        "LeftLeg": "l_leg_knee_link",
        "RightLeg": "r_leg_knee_link",
        "LeftFoot": "l_leg_ankle_roll_link",
        "RightFoot": "r_leg_ankle_roll_link",
        "LeftToeBase": "l_foot_sphere_5_link",
        "RightToeBase": "r_foot_sphere_5_link",
        "LeftArm": "l_arm_shoulder_yaw_link",
        "RightArm": "r_arm_shoulder_yaw_link",
        "LeftForeArm": "l_arm_elbow_pitch_link",
        "RightForeArm": "r_arm_elbow_pitch_link",
        "LeftHand": "l_hand_sphere_link",
        "RightHand": "r_hand_sphere_link",
    },
    ("mocap", "e2"): {
        "Hips": "base_link",
        "LeftUpLeg": "l_leg_hip_pitch_link",
        "RightUpLeg": "r_leg_hip_pitch_link",
        "LeftLeg": "l_leg_knee_link",
        "RightLeg": "r_leg_knee_link",
        "LeftFoot": "l_leg_ankle_roll_link",
        "RightFoot": "r_leg_ankle_roll_link",
        "LeftToeBase": "l_foot_sphere_5_link",
        "RightToeBase": "r_foot_sphere_5_link",
        "LeftArm": "l_arm_shoulder_roll_link",
        "RightArm": "r_arm_shoulder_roll_link",
        "LeftForeArm": "l_arm_elbow_link",
        "RightForeArm": "r_arm_elbow_link",
        "LeftHand": "l_hand_sphere_link",
        "RightHand": "r_hand_sphere_link",
    },
}

# GMR applies a fixed frame offset after each source joint orientation. Every
# source/robot pair is intentionally written as an independent table so one
# calibration can be changed and inspected without mutating another pair.
ORIENTATION_JOINTS_MAPPINGS: dict[tuple[str, str], dict[str, str]] = {
    ("fbx_mocap", "g1"): {
        "Hips": "pelvis",
        "LeftUpLeg": "left_hip_pitch_link",
        "RightUpLeg": "right_hip_pitch_link",
        "LeftLeg": "left_knee_link",
        "RightLeg": "right_knee_link",
        "LeftFoot": "left_ankle_roll_link",
        "RightFoot": "right_ankle_roll_link",
        "LeftToeBase": "left_ankle_roll_sphere_5_link",
        "RightToeBase": "right_ankle_roll_sphere_5_link",
        "LeftArm": "left_shoulder_yaw_link",
        "RightArm": "right_shoulder_yaw_link",
        "LeftForeArm": "left_elbow_link",
        "RightForeArm": "right_elbow_link",
        "LeftHand": "left_rubber_hand_link",
        "RightHand": "right_rubber_hand_link",
    },
    ("fbx_mocap", "e1"): {
        "Hips": "base_link",
        "LeftUpLeg": "l_leg_hip_pitch_link",
        "RightUpLeg": "r_leg_hip_pitch_link",
        "LeftLeg": "l_leg_knee_link",
        "RightLeg": "r_leg_knee_link",
        "LeftFoot": "l_leg_ankle_roll_link",
        "RightFoot": "r_leg_ankle_roll_link",
        "LeftToeBase": "l_foot_sphere_5_link",
        "RightToeBase": "r_foot_sphere_5_link",
        "LeftArm": "l_arm_shoulder_yaw_link",
        "RightArm": "r_arm_shoulder_yaw_link",
        "LeftForeArm": "l_arm_elbow_pitch_link",
        "RightForeArm": "r_arm_elbow_pitch_link",
        "LeftHand": "l_hand_sphere_link",
        "RightHand": "r_hand_sphere_link",
    },
    ("fbx_mocap", "e2"): {
        "Hips": "base_link",
        "LeftUpLeg": "l_leg_hip_pitch_link",
        "RightUpLeg": "r_leg_hip_pitch_link",
        "LeftLeg": "l_leg_knee_link",
        "RightLeg": "r_leg_knee_link",
        "LeftFoot": "l_leg_ankle_roll_link",
        "RightFoot": "r_leg_ankle_roll_link",
        "LeftToeBase": "l_foot_sphere_5_link",
        "RightToeBase": "r_foot_sphere_5_link",
        "LeftArm": "l_arm_shoulder_yaw_link",
        "RightArm": "r_arm_shoulder_yaw_link",
        "LeftForeArm": "l_arm_elbow_link",
        "RightForeArm": "r_arm_elbow_link",
        "LeftHand": "l_hand_sphere_link",
        "RightHand": "r_hand_sphere_link",
    },
    ("amass", "g1"): {
        "Pelvis": "pelvis",
        "L_Hip": "left_hip_roll_link",
        "R_Hip": "right_hip_roll_link",
        "L_Knee": "left_knee_link",
        "R_Knee": "right_knee_link",
        "L_Ankle": "left_ankle_roll_link",
        "R_Ankle": "right_ankle_roll_link",
        "L_Foot": "left_ankle_roll_sphere_5_link",
        "R_Foot": "right_ankle_roll_sphere_5_link",
        "L_Shoulder": "left_shoulder_yaw_link",
        "R_Shoulder": "right_shoulder_yaw_link",
        "L_Elbow": "left_elbow_link",
        "R_Elbow": "right_elbow_link",
        "L_Wrist": "left_rubber_hand_link",
        "R_Wrist": "right_rubber_hand_link",
    },
    ("amass", "e1"): {
        "Pelvis": "base_link",
        "L_Hip": "l_leg_hip_roll_link",
        "R_Hip": "r_leg_hip_roll_link",
        "L_Knee": "l_leg_knee_link",
        "R_Knee": "r_leg_knee_link",
        "L_Ankle": "l_leg_ankle_roll_link",
        "R_Ankle": "r_leg_ankle_roll_link",
        "L_Foot": "l_foot_sphere_5_link",
        "R_Foot": "r_foot_sphere_5_link",
        "L_Shoulder": "l_arm_shoulder_yaw_link",
        "R_Shoulder": "r_arm_shoulder_yaw_link",
        "L_Elbow": "l_arm_elbow_pitch_link",
        "R_Elbow": "r_arm_elbow_pitch_link",
        "L_Wrist": "l_hand_sphere_link",
        "R_Wrist": "r_hand_sphere_link",
    },
    ("amass", "e2"): {
        "Pelvis": "base_link",
        "L_Hip": "l_leg_hip_roll_link",
        "R_Hip": "r_leg_hip_roll_link",
        "L_Knee": "l_leg_knee_link",
        "R_Knee": "r_leg_knee_link",
        "L_Ankle": "l_leg_ankle_roll_link",
        "R_Ankle": "r_leg_ankle_roll_link",
        "L_Foot": "l_foot_sphere_5_link",
        "R_Foot": "r_foot_sphere_5_link",
        "L_Shoulder": "l_arm_shoulder_yaw_link",
        "R_Shoulder": "r_arm_shoulder_yaw_link",
        "L_Elbow": "l_arm_elbow_link",
        "R_Elbow": "r_arm_elbow_link",
        "L_Wrist": "l_hand_sphere_link",
        "R_Wrist": "r_hand_sphere_link",
    },
    ("gvhmr", "g1"): {
        "Pelvis": "pelvis",
        "L_Hip": "left_hip_roll_link",
        "R_Hip": "right_hip_roll_link",
        "L_Knee": "left_knee_link",
        "R_Knee": "right_knee_link",
        "L_Ankle": "left_ankle_roll_link",
        "R_Ankle": "right_ankle_roll_link",
        "L_Foot": "left_ankle_roll_sphere_5_link",
        "R_Foot": "right_ankle_roll_sphere_5_link",
        "L_Shoulder": "left_shoulder_yaw_link",
        "R_Shoulder": "right_shoulder_yaw_link",
        "L_Elbow": "left_elbow_link",
        "R_Elbow": "right_elbow_link",
        "L_Wrist": "left_rubber_hand_link",
        "R_Wrist": "right_rubber_hand_link",
    },
    ("gvhmr", "e1"): {
        "Pelvis": "base_link",
        "L_Hip": "l_leg_hip_roll_link",
        "R_Hip": "r_leg_hip_roll_link",
        "L_Knee": "l_leg_knee_link",
        "R_Knee": "r_leg_knee_link",
        "L_Ankle": "l_leg_ankle_roll_link",
        "R_Ankle": "r_leg_ankle_roll_link",
        "L_Foot": "l_foot_sphere_5_link",
        "R_Foot": "r_foot_sphere_5_link",
        "L_Shoulder": "l_arm_shoulder_yaw_link",
        "R_Shoulder": "r_arm_shoulder_yaw_link",
        "L_Elbow": "l_arm_elbow_pitch_link",
        "R_Elbow": "r_arm_elbow_pitch_link",
        "L_Wrist": "l_hand_sphere_link",
        "R_Wrist": "r_hand_sphere_link",
    },
    ("gvhmr", "e2"): {
        "Pelvis": "base_link",
        "L_Hip": "l_leg_hip_roll_link",
        "R_Hip": "r_leg_hip_roll_link",
        "L_Knee": "l_leg_knee_link",
        "R_Knee": "r_leg_knee_link",
        "L_Ankle": "l_leg_ankle_roll_link",
        "R_Ankle": "r_leg_ankle_roll_link",
        "L_Foot": "l_foot_sphere_5_link",
        "R_Foot": "r_foot_sphere_5_link",
        "L_Shoulder": "l_arm_shoulder_yaw_link",
        "R_Shoulder": "r_arm_shoulder_yaw_link",
        "L_Elbow": "l_arm_elbow_link",
        "R_Elbow": "r_arm_elbow_link",
        "L_Wrist": "l_hand_sphere_link",
        "R_Wrist": "r_hand_sphere_link",
    },
    ("lafan", "g1"): {
        "Hips": "pelvis",
        "LeftUpLeg": "left_hip_yaw_link",
        "RightUpLeg": "right_hip_yaw_link",
        "LeftLeg": "left_knee_link",
        "RightLeg": "right_knee_link",
        "LeftFoot": "left_ankle_roll_link",
        "RightFoot": "right_ankle_roll_link",
        "LeftToeBase": "left_ankle_roll_sphere_5_link",
        "RightToeBase": "right_ankle_roll_sphere_5_link",
        "LeftArm": "left_shoulder_yaw_link",
        "RightArm": "right_shoulder_yaw_link",
        "LeftForeArm": "left_elbow_link",
        "RightForeArm": "right_elbow_link",
        "LeftHand": "left_rubber_hand_link",
        "RightHand": "right_rubber_hand_link",
    },
    ("lafan", "e1"): {
        "Hips": "base_link",
        "LeftUpLeg": "l_leg_hip_yaw_link",
        "RightUpLeg": "r_leg_hip_yaw_link",
        "LeftLeg": "l_leg_knee_link",
        "RightLeg": "r_leg_knee_link",
        "LeftFoot": "l_leg_ankle_roll_link",
        "RightFoot": "r_leg_ankle_roll_link",
        "LeftToeBase": "l_foot_sphere_5_link",
        "RightToeBase": "r_foot_sphere_5_link",
        "LeftArm": "l_arm_shoulder_yaw_link",
        "RightArm": "r_arm_shoulder_yaw_link",
        "LeftForeArm": "l_arm_elbow_pitch_link",
        "RightForeArm": "r_arm_elbow_pitch_link",
        "LeftHand": "l_hand_sphere_link",
        "RightHand": "r_hand_sphere_link",
    },
    ("lafan", "e2"): {
        "Hips": "base_link",
        "LeftUpLeg": "l_leg_hip_yaw_link",
        "RightUpLeg": "r_leg_hip_yaw_link",
        "LeftLeg": "l_leg_knee_link",
        "RightLeg": "r_leg_knee_link",
        "LeftFoot": "l_leg_ankle_roll_link",
        "RightFoot": "r_leg_ankle_roll_link",
        "LeftToeBase": "l_foot_sphere_5_link",
        "RightToeBase": "r_foot_sphere_5_link",
        "LeftArm": "l_arm_shoulder_yaw_link",
        "RightArm": "r_arm_shoulder_yaw_link",
        "LeftForeArm": "l_arm_elbow_link",
        "RightForeArm": "r_arm_elbow_link",
        "LeftHand": "l_hand_sphere_link",
        "RightHand": "r_hand_sphere_link",
    },
    ("mocap", "g1"): {
        "Hips": "pelvis",
        "LeftUpLeg": "left_hip_pitch_link",
        "RightUpLeg": "right_hip_pitch_link",
        "LeftLeg": "left_knee_link",
        "RightLeg": "right_knee_link",
        "LeftFoot": "left_ankle_roll_link",
        "RightFoot": "right_ankle_roll_link",
        "LeftToeBase": "left_ankle_roll_sphere_5_link",
        "RightToeBase": "right_ankle_roll_sphere_5_link",
        "LeftArm": "left_shoulder_yaw_link",
        "RightArm": "right_shoulder_yaw_link",
        "LeftForeArm": "left_elbow_link",
        "RightForeArm": "right_elbow_link",
        "LeftHand": "left_rubber_hand_link",
        "RightHand": "right_rubber_hand_link",
    },
    ("mocap", "e1"): {
        "Hips": "base_link",
        "LeftUpLeg": "l_leg_hip_pitch_link",
        "RightUpLeg": "r_leg_hip_pitch_link",
        "LeftLeg": "l_leg_knee_link",
        "RightLeg": "r_leg_knee_link",
        "LeftFoot": "l_leg_ankle_roll_link",
        "RightFoot": "r_leg_ankle_roll_link",
        "LeftToeBase": "l_foot_sphere_5_link",
        "RightToeBase": "r_foot_sphere_5_link",
        "LeftArm": "l_arm_shoulder_yaw_link",
        "RightArm": "r_arm_shoulder_yaw_link",
        "LeftForeArm": "l_arm_elbow_pitch_link",
        "RightForeArm": "r_arm_elbow_pitch_link",
        "LeftHand": "l_hand_sphere_link",
        "RightHand": "r_hand_sphere_link",
    },
    ("mocap", "e2"): {
        "Hips": "base_link",
        "LeftUpLeg": "l_leg_hip_pitch_link",
        "RightUpLeg": "r_leg_hip_pitch_link",
        "LeftLeg": "l_leg_knee_link",
        "RightLeg": "r_leg_knee_link",
        "LeftFoot": "l_leg_ankle_roll_link",
        "RightFoot": "r_leg_ankle_roll_link",
        "LeftToeBase": "l_foot_sphere_5_link",
        "RightToeBase": "r_foot_sphere_5_link",
        "LeftArm": "l_arm_shoulder_yaw_link",
        "RightArm": "r_arm_shoulder_yaw_link",
        "LeftForeArm": "l_arm_elbow_link",
        "RightForeArm": "r_arm_elbow_link",
        "LeftHand": "l_hand_sphere_link",
        "RightHand": "r_hand_sphere_link",
    },
    ("noetix_mocap", "g1"): {
        "Hips": "pelvis",
        "LeftUpLeg": "left_hip_pitch_link",
        "RightUpLeg": "right_hip_pitch_link",
        "LeftLeg": "left_knee_link",
        "RightLeg": "right_knee_link",
        "LeftFoot": "left_ankle_roll_link",
        "RightFoot": "right_ankle_roll_link",
        "LeftToeBase": "left_ankle_roll_sphere_5_link",
        "RightToeBase": "right_ankle_roll_sphere_5_link",
        "LeftArm": "left_shoulder_yaw_link",
        "RightArm": "right_shoulder_yaw_link",
        "LeftForeArm": "left_elbow_link",
        "RightForeArm": "right_elbow_link",
        "LeftHand": "left_rubber_hand_link",
        "RightHand": "right_rubber_hand_link",
    },
    ("noetix_mocap", "e1"): {
        "Hips": "base_link",
        "LeftUpLeg": "l_leg_hip_pitch_link",
        "RightUpLeg": "r_leg_hip_pitch_link",
        "LeftLeg": "l_leg_knee_link",
        "RightLeg": "r_leg_knee_link",
        "LeftFoot": "l_leg_ankle_roll_link",
        "RightFoot": "r_leg_ankle_roll_link",
        "LeftToeBase": "l_foot_sphere_5_link",
        "RightToeBase": "r_foot_sphere_5_link",
        "LeftArm": "l_arm_shoulder_yaw_link",
        "RightArm": "r_arm_shoulder_yaw_link",
        "LeftForeArm": "l_arm_elbow_pitch_link",
        "RightForeArm": "r_arm_elbow_pitch_link",
        "LeftHand": "l_hand_sphere_link",
        "RightHand": "r_hand_sphere_link",
    },
    ("noetix_mocap", "e2"): {
        "Hips": "base_link",
        "LeftUpLeg": "l_leg_hip_pitch_link",
        "RightUpLeg": "r_leg_hip_pitch_link",
        "LeftLeg": "l_leg_knee_link",
        "RightLeg": "r_leg_knee_link",
        "LeftFoot": "l_leg_ankle_roll_link",
        "RightFoot": "r_leg_ankle_roll_link",
        "LeftToeBase": "l_foot_sphere_5_link",
        "RightToeBase": "r_foot_sphere_5_link",
        "LeftArm": "l_arm_shoulder_yaw_link",
        "RightArm": "r_arm_shoulder_yaw_link",
        "LeftForeArm": "l_arm_elbow_link",
        "RightForeArm": "r_arm_elbow_link",
        "LeftHand": "l_hand_sphere_link",
        "RightHand": "r_hand_sphere_link",
    },
    ("omomo", "g1"): {
        "Pelvis": "pelvis",
        "L_Hip": "left_hip_pitch_link",
        "R_Hip": "right_hip_pitch_link",
        "L_Knee": "left_knee_link",
        "R_Knee": "right_knee_link",
        "L_Ankle": "left_ankle_roll_link",
        "R_Ankle": "right_ankle_roll_link",
        "L_Toe": "left_ankle_roll_sphere_5_link",
        "R_Toe": "right_ankle_roll_sphere_5_link",
        "L_Shoulder": "left_shoulder_yaw_link",
        "R_Shoulder": "right_shoulder_yaw_link",
        "L_Elbow": "left_elbow_link",
        "R_Elbow": "right_elbow_link",
        "L_Wrist": "left_rubber_hand_link",
        "R_Wrist": "right_rubber_hand_link",
    },
    ("omomo", "e1"): {
        "Pelvis": "base_link",
        "L_Hip": "l_leg_hip_pitch_link",
        "R_Hip": "r_leg_hip_pitch_link",
        "L_Knee": "l_leg_knee_link",
        "R_Knee": "r_leg_knee_link",
        "L_Ankle": "l_leg_ankle_roll_link",
        "R_Ankle": "r_leg_ankle_roll_link",
        "L_Toe": "l_foot_sphere_5_link",
        "R_Toe": "r_foot_sphere_5_link",
        "L_Shoulder": "l_arm_shoulder_yaw_link",
        "R_Shoulder": "r_arm_shoulder_yaw_link",
        "L_Elbow": "l_arm_elbow_pitch_link",
        "R_Elbow": "r_arm_elbow_pitch_link",
        "L_Wrist": "l_hand_sphere_link",
        "R_Wrist": "r_hand_sphere_link",
    },
    ("omomo", "e2"): {
        "Pelvis": "base_link",
        "L_Hip": "l_leg_hip_pitch_link",
        "R_Hip": "r_leg_hip_pitch_link",
        "L_Knee": "l_leg_knee_link",
        "R_Knee": "r_leg_knee_link",
        "L_Ankle": "l_leg_ankle_roll_link",
        "R_Ankle": "r_leg_ankle_roll_link",
        "L_Toe": "l_foot_sphere_5_link",
        "R_Toe": "r_foot_sphere_5_link",
        "L_Shoulder": "l_arm_shoulder_yaw_link",
        "R_Shoulder": "r_arm_shoulder_yaw_link",
        "L_Elbow": "l_arm_elbow_link",
        "R_Elbow": "r_arm_elbow_link",
        "L_Wrist": "l_hand_sphere_link",
        "R_Wrist": "r_hand_sphere_link",
    },
}

_IDENTITY_WXYZ = (1.0, 0.0, 0.0, 0.0)
_SMPLX_Z_UP_T_POSE_WXYZ = (
    0.7071067811865476,
    0.7071067811865475,
    0.0,
    0.0,
)
ORIENTATION_T_POSE_HUMAN_QUATERNIONS_WXYZ: dict[
    tuple[str, str],
    dict[str, tuple[float, float, float, float]],
] = {
    ("amass", "g1"): dict.fromkeys(ORIENTATION_JOINTS_MAPPINGS[("amass", "g1")], _SMPLX_Z_UP_T_POSE_WXYZ),
    ("amass", "e1"): dict.fromkeys(ORIENTATION_JOINTS_MAPPINGS[("amass", "e1")], _SMPLX_Z_UP_T_POSE_WXYZ),
    ("amass", "e2"): dict.fromkeys(ORIENTATION_JOINTS_MAPPINGS[("amass", "e2")], _SMPLX_Z_UP_T_POSE_WXYZ),
    ("gvhmr", "g1"): dict.fromkeys(ORIENTATION_JOINTS_MAPPINGS[("gvhmr", "g1")], _SMPLX_Z_UP_T_POSE_WXYZ),
    ("gvhmr", "e1"): dict.fromkeys(ORIENTATION_JOINTS_MAPPINGS[("gvhmr", "e1")], _SMPLX_Z_UP_T_POSE_WXYZ),
    ("gvhmr", "e2"): dict.fromkeys(ORIENTATION_JOINTS_MAPPINGS[("gvhmr", "e2")], _SMPLX_Z_UP_T_POSE_WXYZ),
    ("lafan", "g1"): dict.fromkeys(ORIENTATION_JOINTS_MAPPINGS[("lafan", "g1")], _IDENTITY_WXYZ),
    ("lafan", "e1"): dict.fromkeys(ORIENTATION_JOINTS_MAPPINGS[("lafan", "e1")], _IDENTITY_WXYZ),
    ("lafan", "e2"): dict.fromkeys(ORIENTATION_JOINTS_MAPPINGS[("lafan", "e2")], _IDENTITY_WXYZ),
    ("mocap", "g1"): dict.fromkeys(ORIENTATION_JOINTS_MAPPINGS[("mocap", "g1")], _IDENTITY_WXYZ),
    ("mocap", "e1"): dict.fromkeys(ORIENTATION_JOINTS_MAPPINGS[("mocap", "e1")], _IDENTITY_WXYZ),
    ("mocap", "e2"): dict.fromkeys(ORIENTATION_JOINTS_MAPPINGS[("mocap", "e2")], _IDENTITY_WXYZ),
    ("noetix_mocap", "g1"): dict.fromkeys(
        ORIENTATION_JOINTS_MAPPINGS[("noetix_mocap", "g1")],
        _IDENTITY_WXYZ,
    ),
    ("noetix_mocap", "e1"): dict.fromkeys(
        ORIENTATION_JOINTS_MAPPINGS[("noetix_mocap", "e1")],
        _IDENTITY_WXYZ,
    ),
    ("noetix_mocap", "e2"): dict.fromkeys(
        ORIENTATION_JOINTS_MAPPINGS[("noetix_mocap", "e2")],
        _IDENTITY_WXYZ,
    ),
    ("omomo", "g1"): dict.fromkeys(ORIENTATION_JOINTS_MAPPINGS[("omomo", "g1")], _IDENTITY_WXYZ),
    ("omomo", "e1"): dict.fromkeys(ORIENTATION_JOINTS_MAPPINGS[("omomo", "e1")], _IDENTITY_WXYZ),
    ("omomo", "e2"): dict.fromkeys(ORIENTATION_JOINTS_MAPPINGS[("omomo", "e2")], _IDENTITY_WXYZ),
}

_Y_FACING_ROBOT_BASE_WXYZ = (
    0.7071067811865476,
    0.0,
    0.0,
    0.7071067811865475,
)
_NEGATIVE_Y_FACING_ROBOT_BASE_WXYZ = (
    0.7071067811865476,
    0.0,
    0.0,
    -0.7071067811865475,
)
ORIENTATION_T_POSE_ROBOT_BASE_QUATERNIONS_WXYZ: dict[
    tuple[str, str],
    tuple[float, float, float, float],
] = {
    ("fbx_mocap", "g1"): _NEGATIVE_Y_FACING_ROBOT_BASE_WXYZ,
    ("fbx_mocap", "e1"): _NEGATIVE_Y_FACING_ROBOT_BASE_WXYZ,
    ("fbx_mocap", "e2"): _NEGATIVE_Y_FACING_ROBOT_BASE_WXYZ,
    ("amass", "g1"): _NEGATIVE_Y_FACING_ROBOT_BASE_WXYZ,
    ("amass", "e1"): _NEGATIVE_Y_FACING_ROBOT_BASE_WXYZ,
    ("amass", "e2"): _NEGATIVE_Y_FACING_ROBOT_BASE_WXYZ,
    ("gvhmr", "g1"): _NEGATIVE_Y_FACING_ROBOT_BASE_WXYZ,
    ("gvhmr", "e1"): _NEGATIVE_Y_FACING_ROBOT_BASE_WXYZ,
    ("gvhmr", "e2"): _NEGATIVE_Y_FACING_ROBOT_BASE_WXYZ,
    ("lafan", "g1"): _Y_FACING_ROBOT_BASE_WXYZ,
    ("lafan", "e1"): _Y_FACING_ROBOT_BASE_WXYZ,
    ("lafan", "e2"): _Y_FACING_ROBOT_BASE_WXYZ,
    ("mocap", "g1"): _Y_FACING_ROBOT_BASE_WXYZ,
    ("mocap", "e1"): _Y_FACING_ROBOT_BASE_WXYZ,
    ("mocap", "e2"): _Y_FACING_ROBOT_BASE_WXYZ,
    ("noetix_mocap", "g1"): _Y_FACING_ROBOT_BASE_WXYZ,
    ("noetix_mocap", "e1"): _Y_FACING_ROBOT_BASE_WXYZ,
    ("noetix_mocap", "e2"): _Y_FACING_ROBOT_BASE_WXYZ,
    ("omomo", "g1"): _IDENTITY_WXYZ,
    ("omomo", "e1"): _IDENTITY_WXYZ,
    ("omomo", "e2"): _IDENTITY_WXYZ,
}

_ROBOT_T_POSE_JOINT_POSITIONS: dict[str, dict[str, float]] = {
    "g1": {
        "left_shoulder_pitch_joint": 0.2006477150487628,
        "left_shoulder_roll_joint": 1.476490678407796,
        "left_shoulder_yaw_joint": 0.8742948427729365,
        "left_elbow_joint": 1.414980653432781,
        "right_shoulder_pitch_joint": 0.2006477150487628,
        "right_shoulder_roll_joint": -1.476490678407796,
        "right_shoulder_yaw_joint": -0.8742948427729365,
        "right_elbow_joint": 1.414980653432781,
    },
    "e1": {
        "l_arm_shoulder_roll_joint": 1.5707963267948966,
        "r_arm_shoulder_roll_joint": -1.5707963267948966,
    },
    "e2": {
        "l_arm_shoulder_roll_joint": 1.5707963267948966,
        "l_arm_elbow_joint": 1.0025094781323536,
        "r_arm_shoulder_roll_joint": -1.5707963267948966,
        "r_arm_elbow_joint": 1.0043259754318,
    },
}
ORIENTATION_T_POSE_ROBOT_JOINT_POSITIONS: dict[
    tuple[str, str],
    dict[str, float],
] = {
    ("fbx_mocap", "g1"): _ROBOT_T_POSE_JOINT_POSITIONS["g1"].copy(),
    ("fbx_mocap", "e1"): _ROBOT_T_POSE_JOINT_POSITIONS["e1"].copy(),
    ("fbx_mocap", "e2"): _ROBOT_T_POSE_JOINT_POSITIONS["e2"].copy(),
    ("amass", "g1"): _ROBOT_T_POSE_JOINT_POSITIONS["g1"].copy(),
    ("amass", "e1"): _ROBOT_T_POSE_JOINT_POSITIONS["e1"].copy(),
    ("amass", "e2"): _ROBOT_T_POSE_JOINT_POSITIONS["e2"].copy(),
    ("gvhmr", "g1"): _ROBOT_T_POSE_JOINT_POSITIONS["g1"].copy(),
    ("gvhmr", "e1"): _ROBOT_T_POSE_JOINT_POSITIONS["e1"].copy(),
    ("gvhmr", "e2"): _ROBOT_T_POSE_JOINT_POSITIONS["e2"].copy(),
    ("lafan", "g1"): _ROBOT_T_POSE_JOINT_POSITIONS["g1"].copy(),
    ("lafan", "e1"): _ROBOT_T_POSE_JOINT_POSITIONS["e1"].copy(),
    ("lafan", "e2"): _ROBOT_T_POSE_JOINT_POSITIONS["e2"].copy(),
    ("mocap", "g1"): _ROBOT_T_POSE_JOINT_POSITIONS["g1"].copy(),
    ("mocap", "e1"): _ROBOT_T_POSE_JOINT_POSITIONS["e1"].copy(),
    ("mocap", "e2"): _ROBOT_T_POSE_JOINT_POSITIONS["e2"].copy(),
    ("noetix_mocap", "g1"): _ROBOT_T_POSE_JOINT_POSITIONS["g1"].copy(),
    ("noetix_mocap", "e1"): _ROBOT_T_POSE_JOINT_POSITIONS["e1"].copy(),
    ("noetix_mocap", "e2"): _ROBOT_T_POSE_JOINT_POSITIONS["e2"].copy(),
    ("omomo", "g1"): _ROBOT_T_POSE_JOINT_POSITIONS["g1"].copy(),
    ("omomo", "e1"): _ROBOT_T_POSE_JOINT_POSITIONS["e1"].copy(),
    ("omomo", "e2"): _ROBOT_T_POSE_JOINT_POSITIONS["e2"].copy(),
}

# Data format specific constants
TOE_NAMES_BY_FORMAT = {
    "fbx_mocap": ["LeftToeBase", "RightToeBase"],
    "lafan": ["LeftToeBase", "RightToeBase"],
    "noetix_mocap": ["LeftToeBase", "RightToeBase"],
    "omomo": ["L_Toe", "R_Toe"],
    "mocap": ["LeftToeBase", "RightToeBase"],
    "amass": ["L_Foot", "R_Foot"],
    "gvhmr": ["L_Foot", "R_Foot"],
}


# Skeleton registry. File/task contracts and loaders live in data_utils/motion_data.py.
DEMO_JOINTS_REGISTRY: dict[str, list[str]] = {
    "fbx_mocap": FBX_MOCAP_DEMO_JOINTS,
    "lafan": LAFAN_DEMO_JOINTS,
    "noetix_mocap": NOETIX_MOCAP_DEMO_JOINTS,
    "omomo": OMOMO_DEMO_JOINTS,
    "mocap": MOCAP_DEMO_JOINTS,
    "amass": AMASS_DEMO_JOINTS,
    "gvhmr": GVHMR_DEMO_JOINTS,
}

APPROVED_DIRECT_ORIENTATION_SOURCES: dict[str, frozenset[str]] = {
    "amass": frozenset({"direct_local_rotation_fk"}),
    "fbx_mocap": frozenset({"fbx_local_rotation_curves_fk"}),
    "gvhmr": frozenset({"direct_local_rotation_fk"}),
    "lafan": frozenset({"bvh_rotation_channels_fk"}),
    "mocap": frozenset({"bone_rotation_channels_fk"}),
    "noetix_mocap": frozenset({"bvh_rotation_channels_fk"}),
    "omomo": frozenset({"intermimic_global_orientation_tensor"}),
}


def _parent_indices(
    joint_names: list[str],
    parent_names: dict[str, str | None],
) -> tuple[int, ...]:
    """Resolve a named canonical skeleton into one stable parent-index order."""

    if set(parent_names) != set(joint_names):
        missing = sorted(set(joint_names).difference(parent_names))
        extra = sorted(set(parent_names).difference(joint_names))
        raise ValueError(
            f"Canonical skeleton topology does not match its joint registry: missing={missing}, extra={extra}"
        )
    index = {name: joint_index for joint_index, name in enumerate(joint_names)}
    resolved: list[int] = []
    for name in joint_names:
        parent_name = parent_names[name]
        resolved.append(-1 if parent_name is None else index[parent_name])
    return tuple(resolved)


_LAFAN_PARENT_NAMES: dict[str, str | None] = {
    "Hips": None,
    "RightUpLeg": "Hips",
    "RightLeg": "RightUpLeg",
    "RightFoot": "RightLeg",
    "RightToeBase": "RightFoot",
    "LeftUpLeg": "Hips",
    "LeftLeg": "LeftUpLeg",
    "LeftFoot": "LeftLeg",
    "LeftToeBase": "LeftFoot",
    "Spine": "Hips",
    "Spine1": "Spine",
    "Spine2": "Spine1",
    "Neck": "Spine2",
    "Head": "Neck",
    "RightShoulder": "Spine2",
    "RightArm": "RightShoulder",
    "RightForeArm": "RightArm",
    "RightHand": "RightForeArm",
    "LeftShoulder": "Spine2",
    "LeftArm": "LeftShoulder",
    "LeftForeArm": "LeftArm",
    "LeftHand": "LeftForeArm",
}

_SMPLX_PARENT_NAMES: dict[str, str | None] = {
    name: (None if parent_index == -1 else SMPLX_DEMO_JOINTS[parent_index])
    for name, parent_index in zip(
        SMPLX_DEMO_JOINTS,
        (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19),
        strict=True,
    )
}

_SMPLH_PARENT_NAMES: dict[str, str | None] = {
    "Pelvis": None,
    "L_Hip": "Pelvis",
    "L_Knee": "L_Hip",
    "L_Ankle": "L_Knee",
    "L_Toe": "L_Ankle",
    "R_Hip": "Pelvis",
    "R_Knee": "R_Hip",
    "R_Ankle": "R_Knee",
    "R_Toe": "R_Ankle",
    "Torso": "Pelvis",
    "Spine": "Torso",
    "Chest": "Spine",
    "Neck": "Chest",
    "Head": "Neck",
    "L_Thorax": "Chest",
    "L_Shoulder": "L_Thorax",
    "L_Elbow": "L_Shoulder",
    "L_Wrist": "L_Elbow",
    "R_Thorax": "Chest",
    "R_Shoulder": "R_Thorax",
    "R_Elbow": "R_Shoulder",
    "R_Wrist": "R_Elbow",
}
for _side in ("L", "R"):
    for _finger in ("Index", "Middle", "Pinky", "Ring", "Thumb"):
        _SMPLH_PARENT_NAMES[f"{_side}_{_finger}1"] = f"{_side}_Wrist"
        _SMPLH_PARENT_NAMES[f"{_side}_{_finger}2"] = f"{_side}_{_finger}1"
        _SMPLH_PARENT_NAMES[f"{_side}_{_finger}3"] = f"{_side}_{_finger}2"

_MOCAP_PARENT_NAMES: dict[str, str | None] = {
    "Hips": None,
    "Spine": "Hips",
    "Spine1": "Spine",
    "Neck": "Spine1",
    "Head": "Neck",
    "LeftShoulder": "Spine1",
    "LeftArm": "LeftShoulder",
    "LeftForeArm": "LeftArm",
    "LeftHand": "LeftForeArm",
    "RightShoulder": "Spine1",
    "RightArm": "RightShoulder",
    "RightForeArm": "RightArm",
    "RightHand": "RightForeArm",
    "LeftUpLeg": "Hips",
    "LeftLeg": "LeftUpLeg",
    "LeftFoot": "LeftLeg",
    "LeftToeBase": "LeftFoot",
    "RightUpLeg": "Hips",
    "RightLeg": "RightUpLeg",
    "RightFoot": "RightLeg",
    "RightToeBase": "RightFoot",
    "LeftFootMod": "LeftToeBase",
    "RightFootMod": "RightToeBase",
}
for _side in ("Left", "Right"):
    for _finger in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
        _MOCAP_PARENT_NAMES[f"{_side}Hand{_finger}1"] = f"{_side}Hand"
        _MOCAP_PARENT_NAMES[f"{_side}Hand{_finger}2"] = f"{_side}Hand{_finger}1"
        _MOCAP_PARENT_NAMES[f"{_side}Hand{_finger}3"] = f"{_side}Hand{_finger}2"

DEMO_JOINT_PARENT_INDICES: dict[str, tuple[int, ...]] = {
    "fbx_mocap": _parent_indices(FBX_MOCAP_DEMO_JOINTS, _MOCAP_PARENT_NAMES),
    "lafan": _parent_indices(LAFAN_DEMO_JOINTS, _LAFAN_PARENT_NAMES),
    "noetix_mocap": _parent_indices(
        NOETIX_MOCAP_DEMO_JOINTS,
        _LAFAN_PARENT_NAMES,
    ),
    "omomo": _parent_indices(OMOMO_DEMO_JOINTS, _SMPLH_PARENT_NAMES),
    "mocap": _parent_indices(MOCAP_DEMO_JOINTS, _MOCAP_PARENT_NAMES),
    "amass": _parent_indices(AMASS_DEMO_JOINTS, _SMPLX_PARENT_NAMES),
    "gvhmr": _parent_indices(GVHMR_DEMO_JOINTS, _SMPLX_PARENT_NAMES),
}

DATA_FORMAT_ALIASES = {
    "smplh": "omomo",
    "smplx": "amass",
    "noetix_lafan": "noetix_mocap",
    "noetix-mocap": "noetix_mocap",
}

DataFormat = str


def normalize_data_format(data_format: str) -> str:
    """Return the canonical user-facing name for a supported data format."""

    normalized = DATA_FORMAT_ALIASES.get(data_format.lower(), data_format.lower())
    if normalized not in DEMO_JOINTS_REGISTRY:
        available = ", ".join(sorted(DEMO_JOINTS_REGISTRY.keys()))
        raise ValueError(f"Invalid data_format: '{data_format}'. Available data formats: {available}.")
    return normalized


@dataclass(frozen=True)
class MotionDataConfig:
    data_format: str = "omomo"
    # Use str instead of Literal to allow dynamic robot types
    robot_type: str = "g1"
    robot_defaults: dict[str, RobotDefaults] = field(default_factory=_default_robot_defaults)
    human_height: float | None = None
    """Optional subject height in meters. Overrides the data-format default height."""

    def __post_init__(self) -> None:
        """Validate data_format and robot_type."""
        object.__setattr__(self, "data_format", normalize_data_format(self.data_format))
        _validate_robot_type(self.robot_type, self.robot_defaults)

    # Optional overrides - if None, will use defaults from data_format
    demo_joints: list[str] | None = None
    joint_parent_indices: tuple[int, ...] | None = None
    joints_mapping: dict[str, str] | None = None
    orientation_joints_mapping: dict[str, str] | None = None
    orientation_t_pose_human_quaternions_wxyz: (
        dict[
            str,
            tuple[float, float, float, float],
        ]
        | None
    ) = None
    orientation_t_pose_robot_base_quaternion_wxyz: tuple[float, float, float, float] | None = None
    orientation_t_pose_robot_joint_positions: dict[str, float] | None = None
    orientation_alignment_quaternions_wxyz: (
        dict[
            str,
            tuple[float, float, float, float],
        ]
        | None
    ) = None

    @property
    def resolved_demo_joints(self) -> list[str]:
        """Get demo joints - use override if provided, else use data_format default."""
        if self.demo_joints is not None:
            return self.demo_joints

        return DEMO_JOINTS_REGISTRY[self.data_format]

    @property
    def resolved_joint_parent_indices(self) -> tuple[int, ...]:
        """Return complete topology matching ``resolved_demo_joints``."""

        if self.joint_parent_indices is not None:
            parents = tuple(int(parent) for parent in self.joint_parent_indices)
            if len(parents) != len(self.resolved_demo_joints):
                raise ValueError("joint_parent_indices must have one entry per demo joint")
            joint_count = len(parents)
            if any(parent < -1 or parent >= joint_count for parent in parents):
                raise ValueError("joint_parent_indices must contain -1 or valid joint indices")
            if any(parent == index for index, parent in enumerate(parents)):
                raise ValueError("joint_parent_indices must not contain self-parent joints")
            roots = tuple(index for index, parent in enumerate(parents) if parent == -1)
            if len(roots) != 1:
                raise ValueError("joint_parent_indices must describe exactly one rooted tree")
            root = roots[0]
            for start in range(joint_count):
                current = start
                visited: set[int] = set()
                while current != -1:
                    if current in visited:
                        raise ValueError("joint_parent_indices must not contain cycles")
                    visited.add(current)
                    current = parents[current]
                if root not in visited:
                    raise ValueError("every joint_parent_indices entry must connect to the root")
            return parents
        if self.demo_joints is not None and self.demo_joints != DEMO_JOINTS_REGISTRY[self.data_format]:
            raise ValueError("Custom demo_joints require explicit joint_parent_indices")
        return DEMO_JOINT_PARENT_INDICES[self.data_format]

    @property
    def resolved_joints_mapping(self) -> dict[str, str]:
        """Get joints mapping - use override if provided, else lookup by (data_format, robot_type)."""
        if self.joints_mapping is not None:
            return self.joints_mapping

        key = (self.data_format, robot_family(self.robot_type))
        if key in JOINTS_MAPPINGS:
            return JOINTS_MAPPINGS[key]

        raise ValueError(f"No joint mapping found for data_format={self.data_format}, robot_type={self.robot_type}")

    @property
    def resolved_orientation_joints_mapping(self) -> dict[str, str]:
        """Get the human-joint to robot-body mapping used by orientation tracking."""
        if self.orientation_joints_mapping is not None:
            return self.orientation_joints_mapping
        return ORIENTATION_JOINTS_MAPPINGS.get(
            (self.data_format, robot_family(self.robot_type)),
            {},
        )

    @property
    def resolved_orientation_t_pose_human_quaternions_wxyz(
        self,
    ) -> dict[str, tuple[float, float, float, float]]:
        """Get source-human global joint frames in the canonical T-pose."""

        if self.orientation_t_pose_human_quaternions_wxyz is not None:
            return self.orientation_t_pose_human_quaternions_wxyz
        return ORIENTATION_T_POSE_HUMAN_QUATERNIONS_WXYZ.get(
            (self.data_format, robot_family(self.robot_type)),
            {},
        )

    @property
    def resolved_orientation_t_pose_robot_base_quaternion_wxyz(
        self,
    ) -> tuple[float, float, float, float] | None:
        """Get the robot root orientation sharing the human T-pose facing."""

        if self.orientation_t_pose_robot_base_quaternion_wxyz is not None:
            return self.orientation_t_pose_robot_base_quaternion_wxyz
        return ORIENTATION_T_POSE_ROBOT_BASE_QUATERNIONS_WXYZ.get(
            (self.data_format, robot_family(self.robot_type))
        )

    @property
    def resolved_orientation_t_pose_robot_joint_positions(
        self,
    ) -> dict[str, float]:
        """Get robot joint positions forming the geometric T-pose."""

        if self.orientation_t_pose_robot_joint_positions is not None:
            return self.orientation_t_pose_robot_joint_positions
        return ORIENTATION_T_POSE_ROBOT_JOINT_POSITIONS.get(
            (self.data_format, robot_family(self.robot_type)),
            {},
        )

    @property
    def resolved_orientation_alignment_quaternions_wxyz(
        self,
    ) -> dict[str, tuple[float, float, float, float]]:
        """Get the checked-in per-link calibration generated from T-pose FK."""

        if self.orientation_alignment_quaternions_wxyz is not None:
            return self.orientation_alignment_quaternions_wxyz
        return ORIENTATION_ALIGNMENT_QUATERNIONS_WXYZ.get(
            (self.data_format, robot_family(self.robot_type)),
            {},
        )

    @property
    def toe_names(self) -> list[str]:
        """Get toe joint names for this data format."""
        if self.data_format not in TOE_NAMES_BY_FORMAT:
            raise ValueError(
                f"Toe names not defined for data_format: {self.data_format}. "
                f"Add entry to TOE_NAMES_BY_FORMAT in config_types/data_type.py"
            )
        return TOE_NAMES_BY_FORMAT[self.data_format]

    def legacy_constants(self) -> dict[str, Any]:
        """Return uppercase legacy constants for backward compatibility."""
        return {
            "DEMO_JOINTS": self.resolved_demo_joints,
            "DEMO_JOINT_PARENT_INDICES": self.resolved_joint_parent_indices,
            "JOINTS_MAPPING": self.resolved_joints_mapping,
            "ORIENTATION_JOINTS_MAPPING": self.resolved_orientation_joints_mapping,
            "ORIENTATION_T_POSE_HUMAN_QUATERNIONS_WXYZ": (self.resolved_orientation_t_pose_human_quaternions_wxyz),
            "ORIENTATION_T_POSE_ROBOT_BASE_QUATERNION_WXYZ": (
                self.resolved_orientation_t_pose_robot_base_quaternion_wxyz
            ),
            "ORIENTATION_T_POSE_ROBOT_JOINT_POSITIONS": (self.resolved_orientation_t_pose_robot_joint_positions),
            "ORIENTATION_ALIGNMENT_QUATERNIONS_WXYZ": (self.resolved_orientation_alignment_quaternions_wxyz),
            "TOE_NAMES": self.toe_names,
        }
