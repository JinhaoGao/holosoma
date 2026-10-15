# ruff: noqa: CPY001

"""Rebuild FBX position targets with the actual M5 kinematic dimensions."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from holosoma_retargeting.data_utils.motion_data import HumanMotion
from holosoma_retargeting.foot_contact import FootContactPlan, source_floor_clearance

M5C_BODY_PROPORTIONS_VERSION = "m5c_bone_lengths_v2"
M5_BODY_PROPORTIONS_VERSIONS = {robot: f"{robot}_bone_lengths_v2" for robot in ("m4g", "m5c", "m5d", "m5e")}
M5_BODY_PROPORTIONS_VERSIONS["m5fanqu"] = "m5fanqu_reverse_knee_v1"


@dataclass(frozen=True)
class RebuiltBody:
    """Robot-sized targets, independent of the original contact detection."""

    joints: np.ndarray
    visualization_joints: np.ndarray | None
    sole_centers: np.ndarray
    metadata: dict


def _unit(vectors: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if np.any(lengths < 1e-8):
        raise ValueError("Cannot rebuild a zero-length source bone or torso basis")
    return vectors / lengths


def rebuild_m5_body(
    motion: HumanMotion,
    model: mujoco.MjModel,
    *,
    orientation_reference_qpos: np.ndarray,
    source_contact_plan: FootContactPlan | None = None,
    ground_clearance: float = 0.04,
    robot_type: str = "m5c",
) -> RebuiltBody:
    """Preserve limb directions, replace lengths, and align the actual soles.

    Root XY displacement uses the mean leg-chain ratio. Root Z is recovered
    from the reconstructed legs and soles, preserving scaled source flight
    clearance relative to support phases. Pelvis offsets follow the source base frame; shoulder width
    and torso height follow an orthonormal source torso frame. Foot landmarks
    use calibrated source foot rotations and the model's fixed sole offsets.
    Unmapped fingers/head are display-only and inherit regional scaling.
    """
    version = M5_BODY_PROPORTIONS_VERSIONS[robot_type]
    source = np.asarray(motion.joints, dtype=np.float64)
    names = motion.canonical_joint_names
    indices = {name: i for i, name in enumerate(names)}
    if motion.root_quaternions_wxyz is None or motion.root_frame_to_robot_base_quaternion_wxyz is None:
        raise ValueError("Body reconstruction requires calibrated root orientations")
    root_rotations = (
        Rotation.from_quat(motion.root_quaternions_wxyz[:, [1, 2, 3, 0]])
        * Rotation.from_quat(motion.root_frame_to_robot_base_quaternion_wxyz[[1, 2, 3, 0]])
    ).as_matrix()
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    data.qpos[:3] = 0
    data.qpos[3:7] = [1, 0, 0, 0]
    data.qpos[7:] = 0
    mujoco.mj_forward(model, data)

    def position(name):
        return data.body(name).xpos.copy()

    def source_vector(a, b):
        return source[:, indices[b]] - source[:, indices[a]]

    segments = []
    lengths = {}
    source_lengths = {}
    for side, prefix in (("Left", "l"), ("Right", "r")):
        for start, end, robot_start, robot_end in (
            ("UpLeg", "Leg", "leg_hip_pitch_link", "leg_knee_link"),
            ("Leg", "Foot", "leg_knee_link", "leg_ankle_pitch_link"),
            ("Arm", "ForeArm", "arm_shoulder_roll_link", "arm_elbow_link"),
            ("ForeArm", "Hand", "arm_elbow_link", "hand_sphere_link"),
        ):
            a, b = side + start, side + end
            key = a + ":" + b
            lengths[key] = float(
                np.linalg.norm(position(prefix + "_" + robot_end) - position(prefix + "_" + robot_start))
            )
            source_lengths[key] = float(np.median(np.linalg.norm(source_vector(a, b), axis=-1)))
            segments.append((a, b, lengths[key]))
    leg_keys = [key for key in lengths if "UpLeg" in key or key.endswith((":LeftFoot", ":RightFoot"))]
    leg_scale = sum(lengths[key] for key in leg_keys) / sum(source_lengths[key] for key in leg_keys)

    # The source torso supplies directions; its absolute dimensions are not
    # reused for the robot's pelvis and shoulder anchors.
    up = _unit(source_vector("Hips", "Spine1"))
    lateral = source_vector("RightArm", "LeftArm")
    lateral = _unit(lateral - np.sum(lateral * up, axis=-1, keepdims=True) * up)
    torso_rotations = np.stack([np.cross(lateral, up), lateral, up], axis=-1)
    root = source[:, indices["Hips"]] * leg_scale
    rebuilt = np.zeros_like(source)
    anchors = {"Hips": root}
    waist = position("waist_yaw_link")
    anchors["Spine"] = root + np.einsum("tij,j->ti", root_rotations, waist)
    shoulder_center = (position("l_arm_shoulder_pitch_link") + position("r_arm_shoulder_pitch_link")) / 2
    anchors["Spine1"] = anchors["Spine"] + np.einsum("tij,j->ti", torso_rotations, shoulder_center - waist)
    torso_scale = float(
        np.linalg.norm(shoulder_center) / np.median(np.linalg.norm(source_vector("Hips", "Spine1"), axis=-1))
    )
    region_scales = dict.fromkeys(names, torso_scale)
    for side, prefix in (("Left", "l"), ("Right", "r")):
        anchors[side + "UpLeg"] = root + np.einsum(
            "tij,j->ti", root_rotations, position(prefix + "_leg_hip_pitch_link")
        )
        for suffix, link in (("Shoulder", "arm_shoulder_pitch_link"), ("Arm", "arm_shoulder_roll_link")):
            anchors[side + suffix] = anchors["Spine"] + np.einsum(
                "tij,j->ti", torso_rotations, position(prefix + "_" + link) - waist
            )
        arm_keys = [side + "Arm:" + side + "ForeArm", side + "ForeArm:" + side + "Hand"]
        arm_scale = sum(lengths[key] for key in arm_keys) / sum(source_lengths[key] for key in arm_keys)
        for name in names:
            if name.startswith(side + "Hand"):
                region_scales[name] = arm_scale
    for a, b, length in segments:
        anchors[b] = anchors[a] + _unit(source_vector(a, b)) * length

    if robot_type == "m5fanqu":
        # Reflect the sagittal knee bend across the hip-to-ankle line. The
        # supplied reverse-knee mechanism bends behind the leg, while the
        # human knee bends in front. Both bone lengths, lateral displacement,
        # and the ankle trajectory remain unchanged.
        for side in ("Left", "Right"):
            hip = anchors[side + "UpLeg"]
            knee = anchors[side + "Leg"]
            ankle = anchors[side + "Foot"]
            bend_normal = _unit(np.cross(root_rotations[:, :, 1], _unit(ankle - hip)))
            bend = np.sum((knee - hip) * bend_normal, axis=-1, keepdims=True)
            anchors[side + "Leg"] = knee - 2.0 * bend * bend_normal

    if (
        motion.orientation_quaternions_wxyz is None
        or motion.t_pose_orientation_quaternions_wxyz is None
        or motion.orientation_joint_names is None
        or motion.t_pose_orientation_joint_names is None
    ):
        raise ValueError("Body reconstruction requires direct foot rotations and bind rotations")
    orientation_names = list(motion.orientation_joint_names)
    bind_names = list(motion.t_pose_orientation_joint_names)
    reference_data = mujoco.MjData(model)
    reference_data.qpos[:] = orientation_reference_qpos
    mujoco.mj_forward(model, reference_data)
    sole_sets = []
    for side, prefix in (("Left", "l"), ("Right", "r")):
        name = side + "Foot"
        rotation = Rotation.from_quat(
            motion.orientation_quaternions_wxyz[:, orientation_names.index(name)][:, [1, 2, 3, 0]]
        )
        bind = Rotation.from_quat(motion.t_pose_orientation_quaternions_wxyz[bind_names.index(name)][[1, 2, 3, 0]])
        robot_reference = Rotation.from_matrix(reference_data.body(prefix + "_leg_ankle_pitch_link").xmat.reshape(3, 3))
        foot_rotations = (rotation * bind.inv() * robot_reference).as_matrix()
        offsets = np.stack([model.body(prefix + f"_foot_sphere_{i}_link").pos for i in range(1, 6)])
        points = anchors[name][:, None] + np.einsum("tij,kj->tki", foot_rotations, offsets)
        sole_sets.append(points)
        anchors[side + "ToeBase"] = points[:, 4]

    for i, name in enumerate(names):
        if name in anchors:
            rebuilt[:, i] = anchors[name]
        else:
            parent = int(motion.joint_parent_indices[i])
            if parent < 0 or parent >= i:
                raise ValueError("Body reconstruction needs a parent-first canonical skeleton")
            rebuilt[:, i] = rebuilt[:, parent] + (source[:, i] - source[:, parent]) * region_scales[name]

    soles = np.stack(sole_sets, axis=1)
    raw_clearance = source_floor_clearance(
        source,
        names,
        motion.joint_parent_indices,
        ["LeftToeBase", "RightToeBase"],
        ground_clearance=ground_clearance,
    )
    support_frames = (
        np.flatnonzero(np.any(source_contact_plan.modes != "swing", axis=1))
        if source_contact_plan is not None
        else np.array([], dtype=int)
    )
    # Remove floor-estimation drift at source support frames. Interpolate the
    # support baseline across flight intervals, so toe-off/touchdown remain
    # continuous instead of abruptly snapping an airborne frame to ground.
    support_baseline = (
        np.interp(np.arange(len(source)), support_frames, raw_clearance[support_frames])
        if len(support_frames)
        else float(np.min(raw_clearance))
    )
    flight_clearance = np.maximum(raw_clearance - support_baseline, 0.0) * leg_scale
    height_shift = flight_clearance - np.min(soles[..., 2], axis=(1, 2))
    rebuilt[..., 2] += height_shift[:, None]
    soles[..., 2] += height_shift[:, None, None]

    visualization = None
    if motion.source_skeleton_positions is not None:
        if motion.source_skeleton_joint_names is None or motion.source_skeleton_parent_indices is None:
            raise ValueError("Body reconstruction requires complete source-skeleton names and parents")
        original = motion.source_skeleton_positions
        visualization = np.empty_like(original)
        for i, name in enumerate(motion.source_skeleton_joint_names):
            if name in indices:
                visualization[:, i] = rebuilt[:, indices[name]]
            else:
                parent = int(motion.source_skeleton_parent_indices[i])
                if parent < 0 or parent >= i:
                    raise ValueError("Cannot anchor an unknown visualization root")
                parent_name = motion.source_skeleton_joint_names[parent]
                scale = region_scales.get(parent_name, torso_scale)
                region_scales[name] = scale
                visualization[:, i] = visualization[:, parent] + (original[:, i] - original[:, parent]) * scale

    return RebuiltBody(
        joints=rebuilt,
        visualization_joints=visualization,
        sole_centers=np.mean(soles[:, :, :4], axis=2),
        metadata={
            "human_position_preprocessing": version,
            "body_proportion_segment_names": np.asarray(list(lengths)),
            "body_proportion_robot_lengths_m": np.asarray(list(lengths.values())),
            "body_proportion_source_lengths_m": np.asarray(list(source_lengths.values())),
            "root_translation_scale": leg_scale,
            "body_proportion_root_height_shift_m": height_shift,
            "body_proportion_target_sole_points": soles,
            "body_proportion_source_clearance_m": raw_clearance,
            "body_proportion_target_clearance_m": flight_clearance,
            "body_proportion_contact_phase_height_enabled": source_contact_plan is not None,
            "body_proportion_height_policy": (
                "heel_toe_clearance_relative_to_source_support"
                if source_contact_plan is not None
                else "heel_toe_clearance_without_contact_phases"
            ),
        },
    )


# Preserve the existing M5C API for callers and saved experiment scripts.
rebuild_m5c_body = rebuild_m5_body
