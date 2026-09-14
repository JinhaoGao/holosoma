# ruff: noqa: CPY001
"""Numerical invariants for morphology preprocessing, independent of optimizer weights."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from holosoma_retargeting.config_types.robot_profiles import load_robot_profile
from holosoma_retargeting.data_utils.npz_to_csv import convert_npz_to_csv
from holosoma_retargeting.data_utils.robot_motion import RobotKinematics, RobotMotionProfile, load_robot_motion
from holosoma_retargeting.data_utils.robot_proportions import rebuild_robot_positions
from scipy.spatial.transform import Rotation


def _kinematics(robot: str) -> RobotKinematics:
    return RobotKinematics(RobotMotionProfile.from_profile(load_robot_profile(robot=robot)))


def _motion(
    tmp_path: Path,
    source: RobotKinematics,
    target: RobotKinematics,
    *,
    flight: bool = False,
    articulate: bool = False,
):
    qpos = np.tile(source.reference_qpos, (90, 1))
    if articulate:
        phase = np.sin(np.linspace(0, 2 * np.pi, len(qpos)))
        for i, address in enumerate(source.joint_addresses.values()):
            qpos[:, address] += 0.15 * phase * (-1) ** i
        root_rotation = Rotation.from_rotvec(
            np.linspace(0, 0.7, len(qpos))[:, None] * np.array([0, 0, 1])
        ) * Rotation.from_quat(
            np.asarray(source.profile.reference_base)[[1, 2, 3, 0]],
        )
        qpos[:, 3:7] = root_rotation.as_quat()[:, [3, 0, 1, 2]]
    source.forward(qpos[0])
    qpos[:, 2] -= source.data.xpos[np.concatenate(list(source.sole_geometry_ids.values())), 2].min()
    qpos[:, 0] += np.linspace(0, 0.4, len(qpos))
    if flight:
        qpos[25:65, 2] += 0.2 * np.sin(np.linspace(0, np.pi, 40)) ** 2
    path = tmp_path / "source.npz"
    np.savez(path, qpos=qpos, robot_actuated_joint_names=list(source.joint_addresses))
    csv = convert_npz_to_csv(path, source.urdf)
    return load_robot_motion(csv, source=source, target=target.profile, fps=30)


def _rebuild(motion, source, target):
    return rebuild_robot_positions(
        motion,
        source=source,
        target=target,
        contact_config=load_robot_profile(robot=target.profile.robot.robot_type).planar_foot_contact,
    )


@pytest.mark.parametrize(("source_name", "target_name"), [("g1", "e2"), ("e2", "g1"), ("e1_23dof", "e2")])
def test_target_bone_lengths_directions_torso_width_and_soles(
    tmp_path: Path, source_name: str, target_name: str
) -> None:
    source, target = _kinematics(source_name), _kinematics(target_name)
    motion = _motion(tmp_path, source, target, articulate=True)
    result = _rebuild(motion, source, target)
    names = result.canonical_joint_names
    for edge, mode, length in zip(
        result.metadata["body_proportion_segment_names"],
        result.metadata["body_proportion_segment_modes"],
        result.metadata["body_proportion_target_lengths_m"],
        strict=True,
    ):
        start, end = (names.index(n) for n in edge.split(":"))
        vector = result.position_targets[:, end] - result.position_targets[:, start]
        if mode in {"bone", "foot", "root_anchor"}:
            np.testing.assert_allclose(np.linalg.norm(vector, axis=-1), length, atol=1e-12)
        if mode == "bone":
            original = motion.joints[:, end] - motion.joints[:, start]
            np.testing.assert_allclose(
                vector / np.linalg.norm(vector, axis=-1, keepdims=True),
                original / np.linalg.norm(original, axis=-1, keepdims=True),
                atol=1e-12,
            )
    # Torso dimensions are the target's geometry, not a scaled root-to-shoulder diagonal.
    left, right = (names.index(n) for n in target.profile.torso_anchor_names)
    target_names = tuple(target.profile.position_links)
    expected_width = np.linalg.norm(
        target.reference_positions[target_names.index(names[left])]
        - target.reference_positions[target_names.index(names[right])]
    )
    np.testing.assert_allclose(
        np.linalg.norm(result.position_targets[:, left] - result.position_targets[:, right], axis=-1),
        expected_width,
        atol=1e-12,
    )
    soles = result.metadata["body_proportion_target_sole_points"]
    np.testing.assert_allclose(soles[:, :, :4].mean(axis=2), result.foot_reference_centers)
    for side in range(2):
        distances = np.linalg.norm(soles[:, side, :, None] - soles[:, side, None, :], axis=-1)
        reference = target.reference_sole_points[side]
        expected = np.linalg.norm(reference[:, None] - reference[None, :], axis=-1)
        np.testing.assert_allclose(distances, np.broadcast_to(expected, distances.shape), atol=1e-12)
    # Contact detection and orientation tracking continue to see the unmodified source.
    np.testing.assert_array_equal(result.joints, motion.joints)
    np.testing.assert_array_equal(result.orientation_quaternions_wxyz, motion.orientation_quaternions_wxyz)
    np.testing.assert_array_equal(result.foot_polygon_positions, motion.foot_polygon_positions)


def test_same_robot_reference_pose_is_identity(tmp_path: Path) -> None:
    source = _kinematics("e2")
    motion = _motion(tmp_path, source, source)
    result = _rebuild(motion, source, source)
    assert result.metadata["root_translation_scale"] == pytest.approx(1)
    np.testing.assert_allclose(result.position_targets, motion.joints, atol=1e-12)


def test_root_motion_and_flight_clearance_use_leg_ratio(tmp_path: Path) -> None:
    source, target = _kinematics("g1"), _kinematics("e2")
    motion = _motion(tmp_path, source, target, flight=True)
    result = _rebuild(motion, source, target)
    ratio = result.metadata["root_translation_scale"]
    root = motion.canonical_joint_names.index(source.profile.root_joint)
    np.testing.assert_allclose(
        result.position_targets[:, root, :2] - result.position_targets[0, root, :2],
        (motion.joints[:, root, :2] - motion.joints[0, root, :2]) * ratio,
    )
    source_clearance = result.metadata["body_proportion_source_clearance_m"]
    target_clearance = result.metadata["body_proportion_target_clearance_m"]
    assert target_clearance.max() > 0.1
    np.testing.assert_allclose(target_clearance, source_clearance * ratio)
    actual_clearance = result.metadata["body_proportion_target_sole_points"][..., 2].min(axis=(1, 2))
    np.testing.assert_allclose(actual_clearance, target_clearance, atol=1e-12)
    assert np.max(np.abs(np.diff(result.position_targets[:, root, 2]))) < 0.03


def test_opposite_reference_facing_fails_before_solver(tmp_path: Path) -> None:
    source, target = _kinematics("g1"), _kinematics("e2")
    source = RobotKinematics(replace(source.profile, reference_base=(2**-0.5, 0, 0, -(2**-0.5))))
    motion = _motion(tmp_path, source, target)
    with pytest.raises(ValueError, match="face opposite"):
        _rebuild(motion, source, target)


def test_zero_length_source_bone_is_rejected(tmp_path: Path) -> None:
    source, target = _kinematics("g1"), _kinematics("e2")
    motion = _motion(tmp_path, source, target)
    positions = motion.joints.copy()
    names = motion.canonical_joint_names
    positions[:, names.index("LeftLeg")] = positions[:, names.index("LeftUpLeg")]
    with pytest.raises(ValueError, match="zero-length source segment"):
        _rebuild(replace(motion, joints=positions), source, target)
