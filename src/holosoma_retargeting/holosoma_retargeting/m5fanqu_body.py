# ruff: noqa: CPY001
"""Source-direction-preserving body scaling for the M5 reverse-knee robot."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from holosoma_retargeting.body_proportions import _unit
from holosoma_retargeting.data_utils.motion_data import HumanMotion

M5FANQU_BODY_PROPORTIONS_VERSION = "source_directions_robot_segments_v1"


@dataclass(frozen=True)
class ScaledFanQuBody:
    """Source-directed leg segments sized from the robot's default pose."""

    joints: np.ndarray
    visualization_joints: np.ndarray | None
    metadata: dict


def scale_m5fanqu_body(
    motion: HumanMotion,
    model: mujoco.MjModel,
    *,
    scale: float,
    toe_names: list[str],
    leg_length_scale: float = 1.12,
) -> ScaledFanQuBody:
    """Scale every source bone to the established M5 fanqu target dimensions.

    Never rotate or reflect a source bone.

    The two dimensions come from world-Z differences in model.qpos0, not
    Euclidean link distances or a straightened robot pose. leg_length_scale
    adjusts both target lengths without changing their ratio. Ankle-to-sole
    height is measured separately so it is not counted twice in the shin.
    A common per-frame Z translation restores the original scaled lowest
    toe height, retaining source flight clearance after the longer legs.
    """
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Source scale must be finite and positive")
    if not np.isfinite(leg_length_scale) or leg_length_scale <= 0:
        raise ValueError("Leg length scale must be finite and positive")
    reference = mujoco.MjData(model)
    reference.qpos[:] = model.qpos0
    mujoco.mj_forward(model, reference)
    lengths = {}
    knee_to_sole = []
    ankle_to_sole = []
    for side, prefix in (("Left", "l"), ("Right", "r")):
        hip_z = reference.body(prefix + "_leg_hip_pitch_link").xpos[2]
        knee_z = reference.body(prefix + "_leg_knee_link").xpos[2]
        ankle_z = reference.body(prefix + "_leg_ankle_pitch_link").xpos[2]
        sole_z = min(reference.body(f"{prefix}_foot_sphere_{i}_link").xpos[2] for i in range(1, 6))
        lengths[side + "Leg"] = float(hip_z - knee_z)
        lengths[side + "Foot"] = float(knee_z - ankle_z)
        knee_to_sole.append(float(knee_z - sole_z))
        ankle_to_sole.append(float(ankle_z - sole_z))
    if min(lengths.values()) <= 0:
        raise ValueError("Default pose must have hips above knees and knees above ankles")
    reference_lengths = np.asarray(list(lengths.values()))
    lengths = {name: length * leg_length_scale for name, length in lengths.items()}

    leg_lengths = dict(lengths)
    segment_metadata = {}
    regional_scales = {}
    # All mapped edges receive an independent robot dimension. Extra head
    # and finger edges have no robot counterpart and inherit their region.
    names = motion.canonical_joint_names
    source = np.asarray(motion.joints, dtype=np.float64)
    parents = motion.joint_parent_indices
    robot_links = {"Hips": "base_link", "Spine": "waist_yaw_link", "Spine1": "torso_tracking_link"}
    for side, prefix in (("Left", "l"), ("Right", "r")):
        for suffix, link in (
            ("UpLeg", "leg_hip_pitch_link"),
            ("Leg", "leg_knee_link"),
            ("Foot", "leg_ankle_pitch_link"),
            ("ToeBase", "foot_sphere_5_link"),
            ("Shoulder", "arm_shoulder_pitch_link"),
            ("Arm", "arm_shoulder_roll_link"),
            ("ForeArm", "arm_elbow_link"),
            ("Hand", "hand_sphere_link"),
        ):
            robot_links[side + suffix] = prefix + "_" + link
    source_lengths = {}
    for i, name in enumerate(names):
        parent = int(parents[i])
        if parent < 0:
            continue
        parent_name = names[parent]
        source_lengths[name] = float(np.median(np.linalg.norm(source[:, i] - source[:, parent], axis=-1)))
        if name in robot_links and parent_name in robot_links and name not in lengths:
            lengths[name] = float(
                np.linalg.norm(reference.body(robot_links[name]).xpos - reference.body(robot_links[parent_name]).xpos)
            )
    torso_scale = (lengths["Spine"] + lengths["Spine1"]) / (source_lengths["Spine"] + source_lengths["Spine1"])
    # The robot has no neck or head chain. Reserve half a shoulder width
    # for these display-only edges rather than extending the robot torso.
    head_scale = (
        (lengths["LeftShoulder"] + lengths["RightShoulder"]) * 0.5 / (source_lengths["Neck"] + source_lengths["Head"])
    )
    for i, name in enumerate(names):
        if int(parents[i]) < 0:
            continue
        region_scale = head_scale if name in {"Neck", "Head"} else torso_scale
        for side in ("Left", "Right"):
            if name.startswith(side + "Hand"):
                region_scale = lengths[side + "Hand"] / source_lengths[side + "Hand"]
            elif name.startswith(side + "FootMod"):
                region_scale = lengths[side + "ToeBase"] / source_lengths[side + "ToeBase"]
        if name not in lengths:
            lengths[name] = source_lengths[name] * region_scale
        regional_scales[name] = lengths[name] / source_lengths[name] if source_lengths[name] > 1e-8 else region_scale
    segment_metadata = {
        "body_proportion_joint_names": np.asarray(list(source_lengths)),
        "body_proportion_source_lengths_m": np.asarray(list(source_lengths.values())),
        "body_proportion_target_lengths_m": np.asarray([lengths[n] for n in source_lengths]),
        "body_proportion_segment_scales": np.asarray([regional_scales[n] for n in source_lengths]),
        "root_translation_scale": scale,
        "body_proportion_reference_qpos": model.qpos0.copy(),
    }

    def resize(source, names, parents):
        source = np.asarray(source, dtype=np.float64)
        result = np.empty_like(source)
        for i, name in enumerate(names):
            parent = int(parents[i])
            if parent < 0:
                result[:, i] = source[:, i] * scale
                continue
            if parent >= i:
                raise ValueError("Leg adaptation requires a parent-first source skeleton")
            bone = source[:, i] - source[:, parent]
            if name in lengths and lengths[name] > 1e-8:
                bone = _unit(bone) * lengths[name]
            else:
                bone = bone * regional_scales.get(name, regional_scales.get(names[parent], scale))
            result[:, i] = result[:, parent] + bone
        return result

    names = motion.canonical_joint_names
    joints = resize(motion.joints, names, motion.joint_parent_indices)
    toe_indices = [names.index(name) for name in toe_names]
    source_toes_z = np.asarray(motion.joints)[:, toe_indices, 2]
    ground_height = float(np.min(source_toes_z))
    if ground_height >= 0.1:
        ground_height -= 0.1
    target_toe_height = (np.min(source_toes_z, axis=1).astype(np.float64) - ground_height) * scale
    height_shift = target_toe_height - np.min(joints[:, toe_indices, 2], axis=1)
    joints[..., 2] += height_shift[:, None]
    visualization = None
    if motion.source_skeleton_positions is not None:
        visualization = resize(
            motion.source_skeleton_positions,
            motion.source_skeleton_joint_names,
            motion.source_skeleton_parent_indices,
        )
        visualization[..., 2] += height_shift[:, None]
    return ScaledFanQuBody(
        joints=joints,
        visualization_joints=visualization,
        metadata={
            "human_position_preprocessing": M5FANQU_BODY_PROPORTIONS_VERSION,
            **segment_metadata,
            "source_trajectory_scale": scale,
            "leg_length_joint_names": np.asarray(list(leg_lengths)),
            "leg_length_targets_m": np.asarray(list(leg_lengths.values())),
            "leg_length_scale": leg_length_scale,
            "leg_default_vertical_lengths_m": reference_lengths,
            "leg_length_reference_qpos": model.qpos0.copy(),
            "leg_default_knee_to_sole_height_m": np.asarray(knee_to_sole),
            "leg_default_ankle_to_sole_height_m": np.asarray(ankle_to_sole),
            "leg_adaptation_height_shift_m": height_shift,
            "leg_adaptation_source_ground_height_m": ground_height,
            "leg_adaptation_target_toe_height_m": target_toe_height,
        },
    )
