"""Configuration types for motion data format."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from holosoma_retargeting.config_types.robot import (
    RobotDefaults,
    _default_robot_defaults,
    _validate_robot_type,
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
        "Spine1": "pelvis_contour_link",
        "LeftUpLeg": "left_hip_pitch_link",
        "RightUpLeg": "right_hip_pitch_link",
        "LeftLeg": "left_knee_link",
        "RightLeg": "right_knee_link",
        "LeftArm": "left_shoulder_roll_link",
        "RightArm": "right_shoulder_roll_link",
        "LeftForeArm": "left_elbow_link",
        "RightForeArm": "right_elbow_link",
        "LeftFoot": "left_ankle_intermediate_1_link",
        "RightFoot": "right_ankle_intermediate_1_link",
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
        "Spine1": "base_link",
        "LeftUpLeg": "l_leg_hip_pitch_link",
        "RightUpLeg": "r_leg_hip_pitch_link",
        "LeftLeg": "l_leg_knee_link",
        "RightLeg": "r_leg_knee_link",
        "LeftArm": "l_arm_shoulder_roll_link",
        "RightArm": "r_arm_shoulder_roll_link",
        "LeftForeArm": "l_arm_elbow_pitch_link",
        "RightForeArm": "r_arm_elbow_pitch_link",
        "LeftFoot": "l_leg_ankle_intermediate_1_link",
        "RightFoot": "r_leg_ankle_intermediate_1_link",
        "LeftToeBase": "l_foot_sphere_5_link",
        "RightToeBase": "r_foot_sphere_5_link",
        "LeftHand": "l_hand_sphere_link",
        "RightHand": "r_hand_sphere_link",
    },
    ("omomo", "g1"): {
        "Pelvis": "pelvis_contour_link",
        "L_Hip": "left_hip_pitch_link",
        "R_Hip": "right_hip_pitch_link",
        "L_Knee": "left_knee_link",
        "R_Knee": "right_knee_link",
        "L_Shoulder": "left_shoulder_roll_link",
        "R_Shoulder": "right_shoulder_roll_link",
        "L_Elbow": "left_elbow_link",
        "R_Elbow": "right_elbow_link",
        "L_Ankle": "left_ankle_intermediate_1_link",
        "R_Ankle": "right_ankle_intermediate_1_link",
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
        "L_Shoulder": "l_arm_shoulder_roll_link",
        "R_Shoulder": "r_arm_shoulder_roll_link",
        "L_Elbow": "l_arm_elbow_pitch_link",
        "R_Elbow": "r_arm_elbow_pitch_link",
        "L_Ankle": "l_leg_ankle_intermediate_1_link",
        "R_Ankle": "r_leg_ankle_intermediate_1_link",
        "L_Toe": "l_foot_sphere_5_link",
        "R_Toe": "r_foot_sphere_5_link",
        "L_Wrist": "l_hand_sphere_link",
        "R_Wrist": "r_hand_sphere_link",
    },
    ("amass", "g1"): {
        "Pelvis": "pelvis_contour_link",
        "L_Hip": "left_hip_pitch_link",
        "R_Hip": "right_hip_pitch_link",
        "L_Knee": "left_knee_link",
        "R_Knee": "right_knee_link",
        "L_Shoulder": "left_shoulder_roll_link",
        "R_Shoulder": "right_shoulder_roll_link",
        "L_Elbow": "left_elbow_link",
        "R_Elbow": "right_elbow_link",
        "L_Ankle": "left_ankle_intermediate_1_link",
        "R_Ankle": "right_ankle_intermediate_1_link",
        "L_Foot": "left_ankle_roll_sphere_5_link",
        "R_Foot": "right_ankle_roll_sphere_5_link",
        "L_Wrist": "left_rubber_hand_link",
        "R_Wrist": "right_rubber_hand_link",
    },
    ("amass", "e1"): {
        "Pelvis": "base_link",
        "L_Hip": "l_leg_hip_pitch_link",
        "R_Hip": "r_leg_hip_pitch_link",
        "L_Knee": "l_leg_knee_link",
        "R_Knee": "r_leg_knee_link",
        "L_Shoulder": "l_arm_shoulder_roll_link",
        "R_Shoulder": "r_arm_shoulder_roll_link",
        "L_Elbow": "l_arm_elbow_pitch_link",
        "R_Elbow": "r_arm_elbow_pitch_link",
        "L_Ankle": "l_leg_ankle_intermediate_1_link",
        "R_Ankle": "r_leg_ankle_intermediate_1_link",
        "L_Foot": "l_foot_sphere_5_link",
        "R_Foot": "r_foot_sphere_5_link",
        "L_Wrist": "l_hand_sphere_link",
        "R_Wrist": "r_hand_sphere_link",
    },
    ("mocap", "g1"): {
        "Spine1": "pelvis_contour_link",
        "LeftUpLeg": "left_hip_pitch_link",
        "LeftLeg": "left_knee_link",
        "LeftToeBase": "left_ankle_roll_sphere_5_link",
        "RightUpLeg": "right_hip_pitch_link",
        "RightLeg": "right_knee_link",
        "RightToeBase": "right_ankle_roll_sphere_5_link",
        "LeftArm": "left_shoulder_roll_link",
        "LeftForeArm": "left_elbow_link",
        "LeftHandMiddle3": "left_sphere_hand_link",
        "RightArm": "right_shoulder_roll_link",
        "RightForeArm": "right_elbow_link",
        "RightHandMiddle3": "right_sphere_hand_link",
        "LeftFoot": "left_ankle_intermediate_1_link",
        "RightFoot": "right_ankle_intermediate_1_link",
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
        "Spine1": "base_link",
        "LeftUpLeg": "l_leg_hip_pitch_link",
        "LeftLeg": "l_leg_knee_link",
        "LeftToeBase": "l_foot_sphere_5_link",
        "RightUpLeg": "r_leg_hip_pitch_link",
        "RightLeg": "r_leg_knee_link",
        "RightToeBase": "r_foot_sphere_5_link",
        "LeftArm": "l_arm_shoulder_roll_link",
        "LeftForeArm": "l_arm_elbow_pitch_link",
        "LeftHandMiddle3": "l_hand_sphere_link",
        "RightArm": "r_arm_shoulder_roll_link",
        "RightForeArm": "r_arm_elbow_pitch_link",
        "RightHandMiddle3": "r_hand_sphere_link",
        "LeftFoot": "l_leg_ankle_intermediate_1_link",
        "RightFoot": "r_leg_ankle_intermediate_1_link",
    },
}

ORIENTATION_JOINTS_MAPPINGS: dict[tuple[str, str], dict[str, str]] = {
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
}

# BVH joints have identity global frames when every rotation channel is zero.
# The Noetix converter changes coordinates by basis conjugation, which preserves
# that identity T-pose frame.  Keeping this reference table separate from the
# motion itself prevents the first animation frame from being mistaken for a
# calibration pose.
ORIENTATION_T_POSE_HUMAN_QUATERNIONS_WXYZ: dict[
    tuple[str, str],
    dict[str, tuple[float, float, float, float]],
] = {
    ("noetix_mocap", "e1"): dict.fromkeys(
        ORIENTATION_JOINTS_MAPPINGS[("noetix_mocap", "e1")],
        (1.0, 0.0, 0.0, 0.0),
    ),
}

ORIENTATION_T_POSE_ROBOT_BASE_QUATERNIONS_WXYZ: dict[
    tuple[str, str],
    tuple[float, float, float, float],
] = {
    # Canonical Noetix T-pose faces +Y; E1 qpos0 faces +X.
    ("noetix_mocap", "e1"): (
        0.7071067811865476,
        0.0,
        0.0,
        0.7071067811865475,
    ),
}

ORIENTATION_T_POSE_ROBOT_JOINT_POSITIONS: dict[
    tuple[str, str],
    dict[str, float],
] = {
    ("noetix_mocap", "e1"): {
        "l_arm_shoulder_roll_joint": 1.5707963267948966,
        "r_arm_shoulder_roll_joint": -1.5707963267948966,
    },
}

# Noetix is a separate company-collected dataset and has its own loader and
# converter. Its normalized skeleton reuses only the LAFAN joint-name topology
# and robot mapping. AMASS and GVHMR both use the first 22 SMPL-X body joints.
for _robot_type in ("g1", "e1"):
    JOINTS_MAPPINGS[("noetix_mocap", _robot_type)] = JOINTS_MAPPINGS[("lafan", _robot_type)].copy()
    JOINTS_MAPPINGS[("gvhmr", _robot_type)] = JOINTS_MAPPINGS[("amass", _robot_type)].copy()

# Data format specific constants
TOE_NAMES_BY_FORMAT = {
    "lafan": ["LeftToeBase", "RightToeBase"],
    "noetix_mocap": ["LeftToeBase", "RightToeBase"],
    "omomo": ["L_Toe", "R_Toe"],
    "mocap": ["LeftToeBase", "RightToeBase"],
    "amass": ["L_Foot", "R_Foot"],
    "gvhmr": ["L_Foot", "R_Foot"],
}


# Skeleton registry. File/task contracts and loaders live in data_utils/motion_data.py.
DEMO_JOINTS_REGISTRY: dict[str, list[str]] = {
    "lafan": LAFAN_DEMO_JOINTS,
    "noetix_mocap": NOETIX_MOCAP_DEMO_JOINTS,
    "omomo": OMOMO_DEMO_JOINTS,
    "mocap": MOCAP_DEMO_JOINTS,
    "amass": AMASS_DEMO_JOINTS,
    "gvhmr": GVHMR_DEMO_JOINTS,
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

    @property
    def resolved_demo_joints(self) -> list[str]:
        """Get demo joints - use override if provided, else use data_format default."""
        if self.demo_joints is not None:
            return self.demo_joints

        return DEMO_JOINTS_REGISTRY[self.data_format]

    @property
    def resolved_joints_mapping(self) -> dict[str, str]:
        """Get joints mapping - use override if provided, else lookup by (data_format, robot_type)."""
        if self.joints_mapping is not None:
            return self.joints_mapping

        key = (self.data_format, self.robot_type)
        if key in JOINTS_MAPPINGS:
            return JOINTS_MAPPINGS[key]

        raise ValueError(f"No joint mapping found for data_format={self.data_format}, robot_type={self.robot_type}")

    @property
    def resolved_orientation_joints_mapping(self) -> dict[str, str]:
        """Get the independent human-joint to robot-body orientation mapping."""
        if self.orientation_joints_mapping is not None:
            return self.orientation_joints_mapping
        return ORIENTATION_JOINTS_MAPPINGS.get(
            (self.data_format, self.robot_type),
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
            (self.data_format, self.robot_type),
            {},
        )

    @property
    def resolved_orientation_t_pose_robot_base_quaternion_wxyz(
        self,
    ) -> tuple[float, float, float, float] | None:
        """Get the robot root orientation sharing the human T-pose facing."""

        if self.orientation_t_pose_robot_base_quaternion_wxyz is not None:
            return self.orientation_t_pose_robot_base_quaternion_wxyz
        return ORIENTATION_T_POSE_ROBOT_BASE_QUATERNIONS_WXYZ.get((self.data_format, self.robot_type))

    @property
    def resolved_orientation_t_pose_robot_joint_positions(
        self,
    ) -> dict[str, float]:
        """Get robot joint positions forming the geometric T-pose."""

        if self.orientation_t_pose_robot_joint_positions is not None:
            return self.orientation_t_pose_robot_joint_positions
        return ORIENTATION_T_POSE_ROBOT_JOINT_POSITIONS.get(
            (self.data_format, self.robot_type),
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
            "JOINTS_MAPPING": self.resolved_joints_mapping,
            "ORIENTATION_JOINTS_MAPPING": self.resolved_orientation_joints_mapping,
            "ORIENTATION_T_POSE_HUMAN_QUATERNIONS_WXYZ": (self.resolved_orientation_t_pose_human_quaternions_wxyz),
            "ORIENTATION_T_POSE_ROBOT_BASE_QUATERNION_WXYZ": (
                self.resolved_orientation_t_pose_robot_base_quaternion_wxyz
            ),
            "ORIENTATION_T_POSE_ROBOT_JOINT_POSITIONS": (self.resolved_orientation_t_pose_robot_joint_positions),
            "TOE_NAMES": self.toe_names,
        }
