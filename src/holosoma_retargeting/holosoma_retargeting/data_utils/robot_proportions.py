# ruff: noqa: CPY001
"""Build target-sized spatial references before the ordinary retargeting solve."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from holosoma_retargeting.config_types.retargeter import PlanarFootContactConfig
from holosoma_retargeting.data_utils.motion_data import SourceMotion
from holosoma_retargeting.data_utils.robot_motion import RobotKinematics
from holosoma_retargeting.foot_contact import build_planar_foot_contact_plan
from scipy.spatial.transform import Rotation

ROBOT_POSITION_SCALING_VERSION = "robot_segment_lengths_v1"


def _matrices(quaternions: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(np.asarray(quaternions)[..., [1, 2, 3, 0]]).as_matrix()


def _tree_order(parents: tuple[int, ...]) -> list[int]:
    """Traverse a validated tree without requiring parent-first storage."""
    order = []
    pending = set(range(len(parents)))
    while pending:
        ready = sorted(i for i in pending if parents[i] == -1 or parents[i] in order)
        if not ready:
            raise ValueError("Position scaling requires one acyclic rooted tree")
        order.extend(ready)
        pending.difference_update(ready)
    return order


def _leg_segments(parents: tuple[int, ...], toe_indices: list[int], root: int) -> list[int]:
    """Collect hip-to-ankle edges, excluding pelvis offsets and the foot itself."""
    segments = []
    for toe in toe_indices:
        node = parents[toe]
        chain = []
        while node != root and parents[node] != root:
            chain.append(node)
            node = parents[node]
        if not chain:
            raise ValueError("Segment scaling requires a leg chain between each root and ankle")
        segments.extend(chain)
    return segments


def rebuild_robot_positions(
    motion: SourceMotion,
    *,
    source: RobotKinematics,
    target: RobotKinematics,
    contact_config: PlanarFootContactConfig,
) -> SourceMotion:
    """Preserve source bone directions with target lengths, torso anchors, and soles.

    Source positions/orientations remain untouched for contact detection and
    orientation losses. Only the position reference passed to the common
    Interaction Mesh solver is rebuilt. Root XY displacements and airborne
    sole clearance use the model leg-chain ratio; root Z follows the rebuilt
    legs instead of the original robot's height.
    """
    if (
        motion.foot_polygon_positions is None
        or motion.orientation_quaternions_wxyz is None
        or motion.root_quaternions_wxyz is None
        or motion.orientation_joint_names is None
    ):
        raise ValueError("Segment scaling requires source FK orientations and sole polygons")
    names = motion.canonical_joint_names
    parents = motion.joint_parent_indices
    index = {name: i for i, name in enumerate(names)}
    target_names = tuple(target.profile.position_links)
    target_index = {name: i for i, name in enumerate(target_names)}
    root = index[source.profile.root_joint]
    toes = [index[n] for n in source.profile.toe_names]
    if source.profile.toe_names != target.profile.toe_names:
        raise ValueError("Position scaling requires matching left/right semantic toes")
    for name, i in target_index.items():
        parent = target.profile.parents[i]
        source_parent = parents[index[name]]
        if (None if parent == -1 else target_names[parent]) != (None if source_parent == -1 else names[source_parent]):
            raise ValueError(f"Source/target semantic parents differ for {name!r}")
    positions = np.asarray(motion.joints, dtype=np.float64)
    if not np.isfinite(positions).all():
        raise ValueError("Source FK positions must be finite")
    source_ref = source.reference_positions
    target_ref = np.empty_like(source_ref)
    for name, i in target_index.items():
        target_ref[index[name]] = target.reference_positions[i]
    # Reference facing is a geometric contract, not just quaternion algebra.
    hips = []
    for toe in toes:
        node = toe
        while parents[node] != root:
            node = parents[node]
        hips.append(node)
    if any(names[h] not in target_index for h in hips):
        raise ValueError("Target profile must map the source hip landmarks")
    source_lateral = source_ref[hips[0]] - source_ref[hips[1]]
    target_lateral = target_ref[hips[0]] - target_ref[hips[1]]
    if np.dot(source_lateral, target_lateral) < 0:
        raise ValueError(
            "Source and target reference poses face opposite directions; align their reference poses first"
        )
    source_lengths = np.zeros(len(names))
    target_lengths = np.zeros(len(names))
    for i in range(len(names)):
        if parents[i] != -1:
            source_lengths[i] = np.linalg.norm(source_ref[i] - source_ref[parents[i]])
            if names[i] in target_index:
                target_lengths[i] = np.linalg.norm(target_ref[i] - target_ref[parents[i]])
    leg_segments = _leg_segments(parents, toes, root)
    if any(names[i] not in target_index for i in leg_segments):
        raise ValueError("Target profile must map every source leg-chain landmark")
    source_leg_length = source_lengths[leg_segments].sum()
    target_leg_length = target_lengths[leg_segments].sum()
    if source_leg_length <= 1e-8 or target_leg_length <= 1e-8:
        raise ValueError("Source and target leg lengths must be positive")
    leg_scale = float(target_leg_length / source_leg_length)
    for name in names:
        if name not in target_index:
            target_lengths[index[name]] = source_lengths[index[name]] * leg_scale

    root_rotations = _matrices(motion.root_quaternions_wxyz)
    root_deltas = root_rotations @ _matrices(np.asarray(source.profile.reference_base)).T
    rebuilt = np.zeros_like(positions)
    rebuilt[:, root, :2] = positions[0, root, :2] + (positions[:, root, :2] - positions[0, root, :2]) * leg_scale
    target_root_ref = target_ref[root]
    torso_anchors = set(target.profile.torso_anchor_names)
    torso_origins = None
    torso_deltas = None
    if torso_anchors:
        if source.reference_torso_quaternion is None or "torso" not in motion.position_frame_quaternions_wxyz:
            raise ValueError("Torso anchors require a torso_link in both source and target position_scaling profiles")
        torso_deltas = (
            _matrices(motion.position_frame_quaternions_wxyz["torso"])
            @ _matrices(
                source.reference_torso_quaternion,
            ).T
        )
        torso_origins = rebuilt[:, root] + np.einsum(
            "tij,j->ti",
            root_deltas,
            target.reference_torso_position - target_root_ref,
        )

    orientation_names = tuple(motion.orientation_joint_names)
    sole_offsets = {}
    for side, toe in enumerate(toes):
        foot = parents[toe]
        foot_name = names[foot]
        if foot_name not in orientation_names:
            raise ValueError(f"Foot {foot_name!r} requires a source orientation for target sole reconstruction")
        orientation_index = orientation_names.index(foot_name)
        foot_deltas = (
            _matrices(motion.orientation_quaternions_wxyz[:, orientation_index])
            @ _matrices(
                source.reference_orientations[orientation_index],
            ).T
        )
        sole_offsets[toe] = np.einsum(
            "tij,kj->tki",
            foot_deltas,
            target.reference_sole_points[side] - target_ref[foot],
        )
    soles = np.empty_like(motion.foot_polygon_positions)
    modes = ["root"] * len(names)
    for i in _tree_order(parents):
        parent = parents[i]
        if i == root:
            continue
        if i in sole_offsets:
            side = toes.index(i)
            soles[:, side] = rebuilt[:, parent, None] + sole_offsets[i]
            rebuilt[:, i] = soles[:, side, 4]
            modes[i] = "foot"
        elif names[i] in torso_anchors:
            rebuilt[:, i] = torso_origins + np.einsum(
                "tij,j->ti",
                torso_deltas,
                target_ref[i] - target.reference_torso_position,
            )
            modes[i] = "torso_anchor"
        elif parent == root and names[i] in target_index:
            rebuilt[:, i] = rebuilt[:, root] + np.einsum("tij,j->ti", root_deltas, target_ref[i] - target_root_ref)
            modes[i] = "root_anchor"
        else:
            vector = positions[:, i] - positions[:, parent]
            lengths = np.linalg.norm(vector, axis=-1)
            if np.any(lengths < 1e-8) and target_lengths[i] > 1e-8:
                raise ValueError(
                    f"Cannot recover a direction for zero-length source segment {names[parent]}:{names[i]}"
                )
            rebuilt[:, i] = rebuilt[:, parent] + vector * (target_lengths[i] / np.maximum(lengths, 1e-8))[:, None]
            modes[i] = "bone"

    source_plan = build_planar_foot_contact_plan(
        positions,
        names,
        parents,
        source.profile.toe_names,
        fps=motion.fps,
        config=contact_config,
        source_heel_positions=motion.foot_heel_positions,
        source_toe_positions=motion.foot_toe_positions,
    )
    raw_floor = np.min(motion.foot_polygon_positions[..., 2], axis=(1, 2))
    support = np.flatnonzero(np.any(source_plan.modes != "swing", axis=1))
    baseline = (
        np.interp(np.arange(len(positions)), support, raw_floor[support])
        if len(support)
        else np.full(len(positions), np.percentile(raw_floor, 2.0))
    )
    clearance = np.maximum(raw_floor - baseline, 0.0) * leg_scale
    height_shift = clearance - np.min(soles[..., 2], axis=(1, 2))
    rebuilt[..., 2] += height_shift[:, None]
    soles[..., 2] += height_shift[:, None, None]
    if not np.isfinite(rebuilt).all():
        raise ValueError("Rebuilt robot positions must be finite")
    edges = [i for i in range(len(names)) if i != root]
    return replace(
        motion,
        position_targets=rebuilt,
        foot_reference_centers=soles[:, :, :4].mean(axis=2),
        metadata={
            **motion.metadata,
            "human_position_preprocessing": ROBOT_POSITION_SCALING_VERSION,
            "body_proportion_segment_names": np.asarray([f"{names[parents[i]]}:{names[i]}" for i in edges]),
            "body_proportion_segment_modes": np.asarray([modes[i] for i in edges]),
            "body_proportion_source_lengths_m": source_lengths[edges],
            "body_proportion_target_lengths_m": target_lengths[edges],
            "root_translation_scale": leg_scale,
            "body_proportion_root_height_shift_m": height_shift,
            "body_proportion_target_sole_points": soles,
            "body_proportion_source_support_baseline_m": baseline,
            "body_proportion_source_clearance_m": np.maximum(raw_floor - baseline, 0),
            "body_proportion_target_clearance_m": clearance,
            "body_proportion_source_contact_modes": source_plan.modes,
            "robot_motion_source_positions_world": positions,
        },
    )
