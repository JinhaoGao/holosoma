#!/usr/bin/env python3
"""Convert Noetix CSV box-climb captures into retargeting-ready scenes."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Literal

import numpy as np

src_root = Path(__file__).resolve().parents[2]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from holosoma_retargeting.config_types.data_type import MOCAP_DEMO_JOINTS  # noqa: E402


NOETIX_BONE_PARENTS: dict[str, str | None] = {
    "Root": None,
    "Hips": "Root",
    "LeftUpLeg": "Hips",
    "LeftLeg": "LeftUpLeg",
    "LeftFoot": "LeftLeg",
    "LeftToe": "LeftFoot",
    "RightUpLeg": "Hips",
    "RightLeg": "RightUpLeg",
    "RightFoot": "RightLeg",
    "RightToe": "RightFoot",
    "Spine1": "Hips",
    "Spine2": "Spine1",
    "Chest": "Spine2",
    "Neck": "Chest",
    "Head": "Neck",
    "LeftShoulder": "Chest",
    "LeftArm": "LeftShoulder",
    "LeftForeArm": "LeftArm",
    "LeftHand": "LeftForeArm",
    "RightShoulder": "Chest",
    "RightArm": "RightShoulder",
    "RightForeArm": "RightArm",
    "RightHand": "RightForeArm",
    "HeadEnd": "Head",
    "LToeEnd": "LeftToe",
    "RToeEnd": "RightToe",
}

for _side, _hand in (("Left", "LeftHand"), ("Right", "RightHand")):
    _prefix = f"FZ{_side}"
    for _finger in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
        NOETIX_BONE_PARENTS[f"{_prefix}{_finger}1"] = _hand
        NOETIX_BONE_PARENTS[f"{_prefix}{_finger}2"] = f"{_prefix}{_finger}1"
        NOETIX_BONE_PARENTS[f"{_prefix}{_finger}3"] = f"{_prefix}{_finger}2"


MOCAP_SOURCE_BONES: dict[str, str] = {
    "Hips": "Hips",
    "Spine": "Spine1",
    "Spine1": "Spine2",
    "Neck": "Neck",
    "Head": "Head",
    "LeftShoulder": "LeftShoulder",
    "LeftArm": "LeftArm",
    "LeftForeArm": "LeftForeArm",
    "LeftHand": "LeftHand",
    "LeftHandThumb1": "FZLeftThumb1",
    "LeftHandThumb2": "FZLeftThumb2",
    "LeftHandThumb3": "FZLeftThumb3",
    "LeftHandIndex1": "FZLeftIndex1",
    "LeftHandIndex2": "FZLeftIndex2",
    "LeftHandIndex3": "FZLeftIndex3",
    "LeftHandMiddle1": "FZLeftMiddle1",
    "LeftHandMiddle2": "FZLeftMiddle2",
    "LeftHandMiddle3": "FZLeftMiddle3",
    "LeftHandRing1": "FZLeftRing1",
    "LeftHandRing2": "FZLeftRing2",
    "LeftHandRing3": "FZLeftRing3",
    "LeftHandPinky1": "FZLeftPinky1",
    "LeftHandPinky2": "FZLeftPinky2",
    "LeftHandPinky3": "FZLeftPinky3",
    "RightShoulder": "RightShoulder",
    "RightArm": "RightArm",
    "RightForeArm": "RightForeArm",
    "RightHand": "RightHand",
    "RightHandThumb1": "FZRightThumb1",
    "RightHandThumb2": "FZRightThumb2",
    "RightHandThumb3": "FZRightThumb3",
    "RightHandIndex1": "FZRightIndex1",
    "RightHandIndex2": "FZRightIndex2",
    "RightHandIndex3": "FZRightIndex3",
    "RightHandMiddle1": "FZRightMiddle1",
    "RightHandMiddle2": "FZRightMiddle2",
    "RightHandMiddle3": "FZRightMiddle3",
    "RightHandRing1": "FZRightRing1",
    "RightHandRing2": "FZRightRing2",
    "RightHandRing3": "FZRightRing3",
    "RightHandPinky1": "FZRightPinky1",
    "RightHandPinky2": "FZRightPinky2",
    "RightHandPinky3": "FZRightPinky3",
    "LeftUpLeg": "LeftUpLeg",
    "LeftLeg": "LeftLeg",
    "LeftFoot": "LeftFoot",
    "LeftToeBase": "LeftToe",
    "RightUpLeg": "RightUpLeg",
    "RightLeg": "RightLeg",
    "RightFoot": "RightFoot",
    "RightToeBase": "RightToe",
    "LeftFootMod": "LToeEnd",
    "RightFootMod": "RToeEnd",
}


@dataclass
class ColumnIndex:
    ncols: int
    lookup: dict[tuple[str, str, str, str, str], int]
    entities_by_type: dict[str, list[tuple[str, str]]]


@dataclass
class NoetixCsvMotion:
    metadata: dict[str, str]
    fps: float
    frame_numbers: np.ndarray
    times: np.ndarray
    bone_names: list[str]
    parent_indices: np.ndarray
    bone_positions_z_up_m: np.ndarray
    bone_rotations_wxyz: np.ndarray
    skeleton_marker_names: list[str]
    skeleton_marker_positions_z_up_m: np.ndarray
    box_marker_names: list[str]
    box_marker_positions_z_up_m: np.ndarray
    box_position_z_up_m: np.ndarray | None
    box_quat_xyzw: np.ndarray | None
    scene_origin_xy_m: np.ndarray
    floor_z_m: float


@dataclass
class Config:
    csv_path: Path
    output_root: Path
    task_name: str | None
    export: Literal["mocap"] = "mocap"
    target_fps: float | None = None
    scene_template_dir: Path | None = None
    asset_scale: float = 1.32 / 1.70


def _strip_skeleton_prefix(name: str) -> str:
    return name.split(":", 1)[1] if ":" in name else name


def _entity_sort_key(entity: tuple[str, str]) -> tuple[int, str, str]:
    name, entity_id = entity
    if entity_id.isdigit():
        return (0, f"{int(entity_id):08d}", name)
    return (1, entity_id, name)


def _metadata_from_row(row: list[str]) -> dict[str, str]:
    return {
        row[i].strip(): row[i + 1].strip()
        for i in range(0, len(row) - 1, 2)
        if row[i].strip()
    }


def _build_column_index(rows: list[list[str]]) -> ColumnIndex:
    ncols = max(len(row) for row in rows[:7])
    header_rows = rows[2:7]
    lookup: dict[tuple[str, str, str, str, str], int] = {}
    entities: dict[str, set[tuple[str, str]]] = defaultdict(set)

    for col in range(ncols):
        values = []
        for row in header_rows:
            values.append(row[col].strip() if col < len(row) else "")
        entity_type, name, entity_id, prop, axis = values
        if not entity_type and not name and not prop:
            continue

        name = _strip_skeleton_prefix(name)
        key = (entity_type, name, entity_id, prop, axis)
        lookup[key] = col
        if entity_type and name and entity_id:
            entities[entity_type].add((name, entity_id))

    return ColumnIndex(
        ncols=ncols,
        lookup=lookup,
        entities_by_type={key: sorted(value, key=_entity_sort_key) for key, value in entities.items()},
    )


def _read_float(row: list[str], col: int, required: bool = True) -> float:
    if col >= len(row) or row[col] == "":
        if required:
            raise ValueError(f"Missing required numeric value at column {col}")
        return np.nan
    return float(row[col])


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return np.where(norm > 1e-12, q / norm, np.array([1.0, 0.0, 0.0, 0.0]))


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    out = np.asarray(q, dtype=np.float64).copy()
    out[..., 1:] *= -1.0
    return out


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = _quat_normalize(q)
    v = np.asarray(v, dtype=np.float64)
    v_quat = np.concatenate([np.zeros(v.shape[:-1] + (1,), dtype=np.float64), v], axis=-1)
    return _quat_multiply(_quat_multiply(q, v_quat), _quat_conjugate(q))[..., 1:]


def _read_bone_channels(
    rows: list[list[str]],
    col_index: ColumnIndex,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    bone_entities = [
        (name, entity_id)
        for name, entity_id in col_index.entities_by_type.get("Bone", [])
        if name in NOETIX_BONE_PARENTS
    ]
    if not bone_entities:
        raise ValueError("No supported Noetix Bone columns found in CSV")

    bone_names = [name for name, _ in bone_entities]
    missing_parents = [
        (name, NOETIX_BONE_PARENTS[name])
        for name in bone_names
        if NOETIX_BONE_PARENTS[name] is not None and NOETIX_BONE_PARENTS[name] not in bone_names
    ]
    if missing_parents:
        raise ValueError(f"Bone parents are missing from CSV: {missing_parents}")

    frame_count = len(rows) - 7
    offsets = np.zeros((frame_count, len(bone_names), 3), dtype=np.float64)
    rotations = np.zeros((frame_count, len(bone_names), 4), dtype=np.float64)

    for frame_idx, row in enumerate(rows[7:]):
        for bone_idx, (name, bone_id) in enumerate(bone_entities):
            quat_xyzw = [
                _read_float(row, col_index.lookup[("Bone", name, bone_id, "Rotation", axis)])
                for axis in ("X", "Y", "Z", "W")
            ]
            rotations[frame_idx, bone_idx] = _quat_normalize(
                np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)
            )
            offsets[frame_idx, bone_idx] = [
                0.001 * _read_float(row, col_index.lookup[("Bone", name, bone_id, "Offset", axis)])
                for axis in ("X", "Y", "Z")
            ]

    return bone_names, offsets, rotations


def _build_parent_indices(bone_names: list[str]) -> np.ndarray:
    name_to_idx = {name: idx for idx, name in enumerate(bone_names)}
    return np.asarray(
        [-1 if NOETIX_BONE_PARENTS[name] is None else name_to_idx[NOETIX_BONE_PARENTS[name]] for name in bone_names],
        dtype=np.int32,
    )


def _forward_kinematics(
    local_offsets: np.ndarray,
    local_rotations_wxyz: np.ndarray,
    parent_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    frame_count, joint_count, _ = local_offsets.shape
    global_positions = np.zeros((frame_count, joint_count, 3), dtype=np.float64)
    global_rotations = np.zeros((frame_count, joint_count, 4), dtype=np.float64)

    for frame_idx in range(frame_count):
        for joint_idx in range(joint_count):
            parent_idx = int(parent_indices[joint_idx])
            local_rot = local_rotations_wxyz[frame_idx, joint_idx]
            if parent_idx < 0:
                global_positions[frame_idx, joint_idx] = local_offsets[frame_idx, joint_idx]
                global_rotations[frame_idx, joint_idx] = local_rot
                continue

            parent_rot = global_rotations[frame_idx, parent_idx]
            global_positions[frame_idx, joint_idx] = (
                global_positions[frame_idx, parent_idx]
                + _quat_rotate(parent_rot, local_offsets[frame_idx, joint_idx])
            )
            global_rotations[frame_idx, joint_idx] = _quat_normalize(_quat_multiply(parent_rot, local_rot))

    return global_positions, global_rotations


def _y_up_m_to_scene_axes(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    converted = points[..., [0, 2, 1]].copy()
    converted[..., 1] *= -1.0
    return converted


def _read_positions(
    rows: list[list[str]],
    col_index: ColumnIndex,
    entity_type: str,
) -> tuple[list[str], np.ndarray]:
    entities = col_index.entities_by_type.get(entity_type, [])
    names = [name for name, _ in entities]
    positions = np.full((len(rows) - 7, len(entities), 3), np.nan, dtype=np.float64)
    for frame_idx, row in enumerate(rows[7:]):
        for entity_idx, (name, entity_id) in enumerate(entities):
            for axis_idx, axis in enumerate(("X", "Y", "Z")):
                key = (entity_type, name, entity_id, "Position", axis)
                if key in col_index.lookup:
                    positions[frame_idx, entity_idx, axis_idx] = 0.001 * _read_float(
                        row,
                        col_index.lookup[key],
                        required=False,
                    )

    return names, _y_up_m_to_scene_axes(positions)


def _read_box_pose(
    rows: list[list[str]],
    col_index: ColumnIndex,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    box_entities = [
        (name, entity_id)
        for name, entity_id in col_index.entities_by_type.get("Rigid Body", [])
        if name.lower() == "box"
    ]
    if not box_entities:
        return None, None

    name, entity_id = box_entities[0]
    position = np.full((len(rows) - 7, 3), np.nan, dtype=np.float64)
    quat_xyzw = np.full((len(rows) - 7, 4), np.nan, dtype=np.float64)
    for frame_idx, row in enumerate(rows[7:]):
        for axis_idx, axis in enumerate(("X", "Y", "Z")):
            key = ("Rigid Body", name, entity_id, "Position", axis)
            if key in col_index.lookup:
                position[frame_idx, axis_idx] = 0.001 * _read_float(row, col_index.lookup[key], required=False)
        for axis_idx, axis in enumerate(("X", "Y", "Z", "W")):
            key = ("Rigid Body", name, entity_id, "Rotation", axis)
            if key in col_index.lookup:
                quat_xyzw[frame_idx, axis_idx] = _read_float(row, col_index.lookup[key], required=False)

    return _y_up_m_to_scene_axes(position), quat_xyzw


def _box_origin_from_markers(box_markers_scene_axes: np.ndarray) -> tuple[np.ndarray, float]:
    valid_frames = np.where(~np.isnan(box_markers_scene_axes).any(axis=(1, 2)))[0]
    if len(valid_frames) == 0:
        raise ValueError("No valid box marker frame found; cannot define fixed box scene origin")

    top = box_markers_scene_axes[int(valid_frames[0])]
    min_xy = np.nanmin(top[:, :2], axis=0)
    max_xy = np.nanmax(top[:, :2], axis=0)
    center_xy = (min_xy + max_xy) * 0.5
    return center_xy.astype(np.float64), 0.0


def _select_box_markers_from_unlabeled_markers(
    marker_names: list[str],
    marker_positions_scene_axes: np.ndarray,
) -> tuple[list[str], np.ndarray]:
    """Recover box top-corner markers from generic Marker columns.

    Some Noetix exports keep the box rigid body but leave RigidBodyMarker
    samples empty. In those files the same physical box top markers can still
    appear as generic Marker columns. Select a roughly 1m square, near-coplanar
    quartet from the first frame with enough visible markers.
    """
    if marker_positions_scene_axes.size == 0:
        raise ValueError("No generic Marker columns found for box marker fallback")

    valid_counts = np.sum(~np.isnan(marker_positions_scene_axes).any(axis=2), axis=1)
    candidate_frames = np.where(valid_counts >= 4)[0]
    if len(candidate_frames) == 0:
        raise ValueError("No generic Marker frame has enough visible points for box marker fallback")

    best: tuple[float, int, tuple[int, int, int, int]] | None = None
    for frame_idx in candidate_frames[: min(len(candidate_frames), 30)]:
        frame = marker_positions_scene_axes[int(frame_idx)]
        valid_indices = np.where(~np.isnan(frame).any(axis=1))[0]
        for combo in combinations(valid_indices.tolist(), 4):
            points = frame[list(combo)]
            z_span = float(np.ptp(points[:, 2]))
            min_xy = points[:, :2].min(axis=0)
            max_xy = points[:, :2].max(axis=0)
            dims = max_xy - min_xy
            if z_span > 0.12:
                continue
            if not (0.75 <= dims[0] <= 1.25 and 0.75 <= dims[1] <= 1.25):
                continue

            corners = np.asarray(
                [
                    [min_xy[0], min_xy[1]],
                    [max_xy[0], min_xy[1]],
                    [max_xy[0], max_xy[1]],
                    [min_xy[0], max_xy[1]],
                ],
                dtype=np.float64,
            )
            distances = np.linalg.norm(points[:, None, :2] - corners[None, :, :], axis=2)
            corner_coverage = float(np.min(distances, axis=0).sum())
            score = abs(float(dims[0]) - 1.0) + abs(float(dims[1]) - 1.0) + 5.0 * z_span + corner_coverage
            if best is None or score < best[0]:
                best = (score, int(frame_idx), combo)

    if best is None:
        raise ValueError("Could not identify a square top-face marker quartet for box marker fallback")

    _, _, combo = best
    selected_names = [marker_names[idx] for idx in combo]
    selected_positions = marker_positions_scene_axes[:, list(combo), :]
    return selected_names, selected_positions


def _apply_scene_normalization(points: np.ndarray, origin_xy: np.ndarray, floor_z: float) -> np.ndarray:
    normalized = np.asarray(points, dtype=np.float64).copy()
    normalized[..., 0] -= origin_xy[0]
    normalized[..., 1] -= origin_xy[1]
    normalized[..., 2] -= floor_z
    return normalized


def load_noetix_csv(csv_path: Path) -> NoetixCsvMotion:
    with csv_path.open(newline="") as f:
        rows = list(csv.reader(f))

    if len(rows) < 8:
        raise ValueError(f"CSV is too short to contain Noetix motion data: {csv_path}")

    metadata = _metadata_from_row(rows[0])
    fps = float(metadata.get("FrameRate", "120"))
    col_index = _build_column_index(rows)

    frame_numbers = np.asarray([int(row[0]) for row in rows[7:]], dtype=np.int32)
    times = np.asarray([float(row[1]) for row in rows[7:]], dtype=np.float64)

    bone_names, local_offsets, local_rotations = _read_bone_channels(rows, col_index)
    parent_indices = _build_parent_indices(bone_names)
    bone_positions_y_up, bone_rotations_wxyz = _forward_kinematics(
        local_offsets,
        local_rotations,
        parent_indices,
    )
    bone_positions_scene_axes = _y_up_m_to_scene_axes(bone_positions_y_up)

    skeleton_marker_names, skeleton_markers_scene_axes = _read_positions(rows, col_index, "SkeletonMarker")
    box_marker_names, box_markers_scene_axes = _read_positions(rows, col_index, "RigidBodyMarker")
    generic_marker_names, generic_markers_scene_axes = _read_positions(rows, col_index, "Marker")
    box_position_scene_axes, box_quat_xyzw = _read_box_pose(rows, col_index)

    if len(np.where(~np.isnan(box_markers_scene_axes).any(axis=(1, 2)))[0]) == 0:
        box_marker_names, box_markers_scene_axes = _select_box_markers_from_unlabeled_markers(
            generic_marker_names,
            generic_markers_scene_axes,
        )

    origin_xy, floor_z = _box_origin_from_markers(box_markers_scene_axes)

    bone_positions_z_up_m = _apply_scene_normalization(bone_positions_scene_axes, origin_xy, floor_z)
    skeleton_markers_z_up_m = _apply_scene_normalization(skeleton_markers_scene_axes, origin_xy, floor_z)
    box_markers_z_up_m = _apply_scene_normalization(box_markers_scene_axes, origin_xy, floor_z)
    if box_position_scene_axes is not None:
        box_position_z_up_m = _apply_scene_normalization(box_position_scene_axes, origin_xy, floor_z)
    else:
        box_position_z_up_m = None

    return NoetixCsvMotion(
        metadata=metadata,
        fps=fps,
        frame_numbers=frame_numbers,
        times=times,
        bone_names=bone_names,
        parent_indices=parent_indices,
        bone_positions_z_up_m=bone_positions_z_up_m,
        bone_rotations_wxyz=bone_rotations_wxyz,
        skeleton_marker_names=skeleton_marker_names,
        skeleton_marker_positions_z_up_m=skeleton_markers_z_up_m,
        box_marker_names=box_marker_names,
        box_marker_positions_z_up_m=box_markers_z_up_m,
        box_position_z_up_m=box_position_z_up_m,
        box_quat_xyzw=box_quat_xyzw,
        scene_origin_xy_m=origin_xy,
        floor_z_m=floor_z,
    )


def _downsample(array: np.ndarray, source_fps: float, target_fps: float | None) -> tuple[np.ndarray, int, float]:
    if target_fps is None:
        return array, 1, float(source_fps)
    stride = max(1, int(round(float(source_fps) / float(target_fps))))
    return array[::stride], stride, float(source_fps) / stride


def map_to_mocap_joints(motion: NoetixCsvMotion) -> np.ndarray:
    bone_idx = {name: idx for idx, name in enumerate(motion.bone_names)}
    missing = [
        source_name
        for target_name in MOCAP_DEMO_JOINTS
        for source_name in [MOCAP_SOURCE_BONES[target_name]]
        if source_name not in bone_idx
    ]
    if missing:
        raise ValueError(f"Missing source bones for mocap conversion: {missing}")

    indices = [bone_idx[MOCAP_SOURCE_BONES[name]] for name in MOCAP_DEMO_JOINTS]
    return motion.bone_positions_z_up_m[:, indices, :].astype(np.float32)


def _box_vertices_from_markers(
    box_marker_positions: np.ndarray,
    box_position: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    valid_frames = np.where(~np.isnan(box_marker_positions).any(axis=(1, 2)))[0]
    if len(valid_frames) == 0:
        raise ValueError("No valid box markers found; cannot reconstruct platform mesh")

    top = box_marker_positions[int(valid_frames[0])]
    min_x, min_y = np.nanmin(top[:, :2], axis=0)
    max_x, max_y = np.nanmax(top[:, :2], axis=0)
    if box_position is not None and len(box_position) > int(valid_frames[0]) and not np.isnan(box_position[int(valid_frames[0]), 2]):
        top_z = float(box_position[int(valid_frames[0]), 2])
    else:
        top_z = float(np.nanmax(top[:, 2]))
    bottom_z = 0.0

    vertices = np.asarray(
        [
            [min_x, min_y, bottom_z],
            [max_x, min_y, bottom_z],
            [max_x, max_y, bottom_z],
            [min_x, max_y, bottom_z],
            [min_x, min_y, top_z],
            [max_x, min_y, top_z],
            [max_x, max_y, top_z],
            [min_x, max_y, top_z],
        ],
        dtype=np.float32,
    )
    faces = np.asarray(
        [
            [1, 2, 3],
            [1, 3, 4],
            [5, 7, 6],
            [5, 8, 7],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
            [3, 7, 8],
            [3, 8, 4],
            [4, 8, 5],
            [4, 5, 1],
        ],
        dtype=np.int32,
    )
    return vertices, faces


def _write_obj(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    source_csv: Path,
    origin_xy: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# Reconstructed from {source_csv.name} frame 1 box markers.\n")
        f.write("# Scene-local frame: Noetix Y-up mm -> MuJoCo Z-up m with right-handed [x, -z, y], then subtract box XY center.\n")
        f.write(f"# scene_origin_xy_m {origin_xy[0]:.9f} {origin_xy[1]:.9f}\n")
        f.write(f"# box_min_m {bbox_min[0]:.9f} {bbox_min[1]:.9f} {bbox_min[2]:.9f}\n")
        f.write(f"# box_max_m {bbox_max[0]:.9f} {bbox_max[1]:.9f} {bbox_max[2]:.9f}\n")
        f.write("o box1\n")
        for vertex in vertices:
            f.write(f"v {vertex[0]:.9f} {vertex[1]:.9f} {vertex[2]:.9f}\n")
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")


def _write_single_platform_mesh(
    output_dir: Path,
    motion: NoetixCsvMotion,
    source_csv: Path,
) -> tuple[np.ndarray, np.ndarray]:
    vertices, faces = _box_vertices_from_markers(motion.box_marker_positions_z_up_m, motion.box_position_z_up_m)
    _write_obj(output_dir / "multi_boxes.obj", vertices, faces, source_csv=source_csv, origin_xy=motion.scene_origin_xy_m)
    _write_obj(
        output_dir / "box_models" / "box1.obj",
        vertices,
        faces,
        source_csv=source_csv,
        origin_xy=motion.scene_origin_xy_m,
    )
    return vertices, faces


def _write_single_box_urdf(output_dir: Path, asset_scale: float) -> None:
    scale = f"{asset_scale:.16g} {asset_scale:.16g} {asset_scale:.16g}"
    (output_dir / "multi_boxes.urdf").write_text(
        f"""<?xml version="1.0"?>
<robot name="multi_boxes">
  <link name="world"/>

  <link name="multi_boxes_link">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="box_models/box1.obj" scale="{scale}"/>
      </geometry>
      <material name="box1_material">
        <color rgba="0.3 0.7 0.9 0.5"/>
      </material>
    </visual>

    <collision name="multi_boxes">
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="box_models/box1.obj" scale="{scale}"/>
      </geometry>
    </collision>

    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="33.33"/>
      <inertia ixx="33.33" ixy="0.0" ixz="0.0" iyy="33.33" iyz="0.0" izz="33.33"/>
    </inertial>
  </link>

  <joint name="world_to_multi_boxes" type="fixed">
    <parent link="world"/>
    <child link="multi_boxes_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>
</robot>
""",
        encoding="utf-8",
    )


def _mesh_path_for_box_assets(output_dir: Path) -> str:
    package_root = Path(__file__).resolve().parents[1]
    demo_data_root = package_root / "demo_data"
    try:
        seq_rel = output_dir.resolve().relative_to(demo_data_root.resolve())
        return f"../../../demo_data/{seq_rel.as_posix()}/box_models/box1.obj"
    except ValueError:
        return (output_dir / "box_models" / "box1.obj").resolve().as_posix()


def _write_mujoco_box_includes(output_dir: Path, asset_scale: float) -> None:
    mesh_path = _mesh_path_for_box_assets(output_dir)
    scale = f"{asset_scale:.16g} {asset_scale:.16g} {asset_scale:.16g}"
    (output_dir / "box_assets.xml").write_text(
        f"""<mujocoinclude>
    <mesh name="box1" file="{mesh_path}"
        scale="{scale}"/>
    <material name="box1_material" rgba="0.3 0.7 0.9 0.5"/>
</mujocoinclude>
""",
        encoding="utf-8",
    )
    (output_dir / "box_body.xml").write_text(
        """<mujocoinclude>
    <body name="multi_boxes_box1_link" pos="0 0 0" quat="1 0 0 0">
        <geom name="multi_boxes_link_1" type="mesh" mesh="box1" pos="0 0 0" quat="1 0 0 0" material="box1_material" contype="1" conaffinity="1"/>
    </body>
</mujocoinclude>
""",
        encoding="utf-8",
    )


def _copy_scene_templates(output_dir: Path, template_dir: Path) -> None:
    if not template_dir.exists():
        print(f"[convert_noetix_csv] template directory missing, skipped scene XML copy: {template_dir}")
        return

    package_root = Path(__file__).resolve().parents[1]
    try:
        output_dir.resolve().relative_to(package_root.resolve())
        model_root = os.path.relpath(package_root / "models", output_dir).replace(os.sep, "/")
    except ValueError:
        model_root = (package_root / "models").resolve().as_posix()
    for xml_path in sorted(template_dir.glob("*_w_multi_boxes.xml")):
        content = xml_path.read_text(encoding="utf-8")
        content = content.replace('meshdir="../../../models', f'meshdir="{model_root}')
        content = content.replace('meshdir="../../../../../models', f'meshdir="{model_root}')
        (output_dir / xml_path.name).write_text(content, encoding="utf-8")


def export_mocap_climb(
    motion: NoetixCsvMotion,
    output_root: Path,
    task_name: str,
    target_fps: float | None,
    scene_template_dir: Path | None,
    asset_scale: float,
    source_csv: Path,
) -> Path:
    seq_dir = output_root / task_name
    seq_dir.mkdir(parents=True, exist_ok=True)

    mocap_positions, stride, output_fps = _downsample(map_to_mocap_joints(motion), motion.fps, target_fps)
    npy_path = seq_dir / f"{task_name}_joint_positions.npy"
    np.save(npy_path, mocap_positions.astype(np.float32))

    box_vertices, box_faces = _write_single_platform_mesh(seq_dir, motion, source_csv)
    _write_single_box_urdf(seq_dir, asset_scale)
    _write_mujoco_box_includes(seq_dir, asset_scale)
    if scene_template_dir is not None:
        _copy_scene_templates(seq_dir, scene_template_dir)

    box_position = (
        motion.box_position_z_up_m.astype(np.float32)
        if motion.box_position_z_up_m is not None
        else np.empty((0, 3), dtype=np.float32)
    )
    box_quat_xyzw = (
        motion.box_quat_xyzw.astype(np.float32)
        if motion.box_quat_xyzw is not None
        else np.empty((0, 4), dtype=np.float32)
    )
    np.savez_compressed(
        seq_dir / "scene_reconstruction.npz",
        mocap_joint_positions=mocap_positions.astype(np.float32),
        mocap_joint_names=np.asarray(MOCAP_DEMO_JOINTS, dtype=str),
        bone_positions=motion.bone_positions_z_up_m.astype(np.float32),
        bone_names=np.asarray(motion.bone_names, dtype=str),
        skeleton_marker_positions=motion.skeleton_marker_positions_z_up_m.astype(np.float32),
        skeleton_marker_names=np.asarray(motion.skeleton_marker_names, dtype=str),
        box_marker_positions=motion.box_marker_positions_z_up_m.astype(np.float32),
        box_marker_names=np.asarray(motion.box_marker_names, dtype=str),
        box_mesh_vertices=box_vertices.astype(np.float32),
        box_mesh_faces=box_faces.astype(np.int32),
        box_position=box_position,
        box_quat_xyzw=box_quat_xyzw,
        source_fps=np.float32(motion.fps),
        fps=np.float32(output_fps),
        downsample_stride=np.int32(stride),
        scene_origin_xy_m=motion.scene_origin_xy_m.astype(np.float32),
        floor_z_m=np.float32(motion.floor_z_m),
    )

    print(
        f"[convert_noetix_csv] mocap climb: frames={mocap_positions.shape[0]}, "
        f"fps={output_fps:.2f}, joints={mocap_positions.shape[1]} -> {npy_path}"
    )
    return seq_dir


def parse_args() -> Config:
    package_root = Path(__file__).resolve().parents[1]
    default_template_dir = package_root / "demo_data" / "climb" / "mocap_climb_seq_0"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, help="Noetix CSV file to convert.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Directory where the task scene folder is written. Defaults to the CSV parent.",
    )
    parser.add_argument("--task-name", type=str, default=None, help="Output task name. Defaults to CSV stem.")
    parser.add_argument("--export", choices=("mocap",), default="mocap")
    parser.add_argument(
        "--target-fps",
        type=float,
        default=None,
        help="Target FPS for the .npy. Default keeps the source 120fps sequence.",
    )
    parser.add_argument(
        "--scene-template-dir",
        type=Path,
        default=default_template_dir,
        help="Directory containing *_w_multi_boxes.xml scene templates to copy.",
    )
    parser.add_argument(
        "--asset-scale",
        type=float,
        default=1.32 / 1.70,
        help="Initial box mesh scale written into XML/URDF. Retargeting rewrites it per robot height.",
    )
    args = parser.parse_args()
    return Config(
        csv_path=args.csv_path,
        output_root=args.output_root or args.csv_path.parent,
        task_name=args.task_name,
        export=args.export,
        target_fps=args.target_fps,
        scene_template_dir=args.scene_template_dir,
        asset_scale=args.asset_scale,
    )


def main(cfg: Config) -> None:
    csv_path = cfg.csv_path.expanduser().resolve()
    task_name = cfg.task_name or csv_path.stem
    output_root = cfg.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    motion = load_noetix_csv(csv_path)
    print(
        f"[convert_noetix_csv] loaded {csv_path.name}: frames={len(motion.frame_numbers)}, "
        f"fps={motion.fps:.2f}, bones={len(motion.bone_names)}, box_markers={len(motion.box_marker_names)}"
    )

    if cfg.export != "mocap":
        raise ValueError(f"Unsupported export mode: {cfg.export}")

    seq_dir = export_mocap_climb(
        motion=motion,
        output_root=output_root,
        task_name=task_name,
        target_fps=cfg.target_fps,
        scene_template_dir=cfg.scene_template_dir,
        asset_scale=cfg.asset_scale,
        source_csv=csv_path,
    )

    for stale_dir in ("generated", "__pycache__"):
        shutil.rmtree(seq_dir / stale_dir, ignore_errors=True)


if __name__ == "__main__":
    main(parse_args())
