# ruff: noqa: CPY001

"""Validation and atomic persistence for versioned retargeting results."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

import numpy as np

from holosoma_retargeting.config_types.data_type import (
    APPROVED_DIRECT_ORIENTATION_SOURCES,
)

RESULT_SCHEMA_VERSION = 2
QUATERNION_NORM_ATOL = 1e-3
POSITION_MATCH_ATOL = 1e-5
OBJECT_TRANSFORM_MATCH_ATOL = 1e-4
ORIENTATION_RELATION_ATOL_RAD = 1e-4
NONLINEAR_CONSTRAINT_VIOLATION_ATOL = 1e-6
COLLISION_INTERIOR_MARGIN_MIN_M = 1e-5
COLLISION_INTERIOR_MARGIN_MAX_M = 5e-4
FRAME_ZERO_GROUND_RETRY_POLICY = "robot_only_frame_zero_horizontal_ground_lift_v1"
QPOS_LAYOUT = "mujoco_free_root_xyz_wxyz_then_actuated_then_optional_object_free_joint"
OBJECT_POSE_LAYOUT = "xyz_wxyz"
QUATERNION_CONVENTION = "wxyz"
WORLD_COORDINATE_SYSTEM = "right_handed_z_up"
ABSENT_ORIENTATION_SOURCE = "absent"
_HUMAN_ORIENTATION_HASH_DOMAIN = b"holosoma-human-orientation-v1\0"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_OBJECT_ASSET_MANIFEST_VERSION = 1
_OBJECT_ASSET_MANIFEST_MAX_BYTES = 16 * 1024 * 1024

REQUIRED_RESULT_KEYS = frozenset(
    {
        "schema_version",
        "run_kind",
        "variant",
        "source_path",
        "source_sha256",
        "config_json",
        "config_sha256",
        "source_data_format",
        "robot_type",
        "task_type",
        "dataset_partition",
        "sequence_key",
        "experiment_name",
        "qpos",
        "qpos_layout",
        "human_joints",
        "human_joint_names",
        "human_joint_parent_indices",
        "mapped_human_joints",
        "mapped_human_joint_names",
        "mapped_robot_joints",
        "mapped_robot_link_names",
        "robot_link_positions",
        "robot_link_quaternions_wxyz",
        "robot_link_names",
        "robot_link_parent_indices",
        "robot_actuated_joint_names",
        "object_name",
        "object_urdf",
        "object_urdf_sha256",
        "object_asset_manifest_json",
        "object_asset_manifest_sha256",
        "contains_object_in_qpos",
        "object_poses_demo",
        "object_poses_target",
        "object_pose_layout",
        "quaternion_convention",
        "world_coordinate_system",
        "orientation_source",
        "source_human_height",
        "human_position_scale",
        "human_position_preprocessing",
        "foot_sticking_side_names",
        "foot_sticking_states",
        "foot_sticking_tolerance",
        "foot_sticking_fallback_tolerance",
        "foot_sticking_fallback_frames",
        "release_foot_sticking_on_infeasible",
        "foot_sticking_release_frames",
        "release_object_non_penetration_on_infeasible",
        "object_non_penetration_release_frames",
        "object_non_penetration_eligible_for_saved_trajectory",
        "foot_sticking_enabled_for_saved_trajectory",
        "foot_sticking_full_sequence_retry_frame",
        "frame_zero_ground_retry_policy",
        "frame_zero_ground_retry_eligible",
        "frame_zero_ground_retry_triggered",
        "frame_zero_ground_retry_initial_min_distance_m",
        "frame_zero_ground_retry_corrected_min_distance_m",
        "frame_zero_ground_retry_lift_m",
        "frame_zero_ground_retry_interior_margin_m",
        "frame_zero_ground_retry_initial_sqp_iterations",
        "frame_costs",
        "sqp_iteration_counts",
        "sqp_stop_reasons",
        "ground_non_penetration_violation",
        "object_non_penetration_violation",
        "foot_sticking_violation",
        "foot_lock_violation",
        "self_collision_violation",
        "joint_limits_violation",
        "constraint_mode_foot_sticking",
        "constraint_mode_object_non_penetration_released",
        "constraint_mode_trust_region_released",
        "cost",
        "orientation_tracking_enabled",
        "orientation_diagnostics_enabled",
        "orientation_human_joint_names",
        "orientation_robot_link_names",
        "orientation_weights",
        "orientation_alignment_mode",
        "orientation_alignment_quaternions_wxyz",
        "orientation_reference_human_quaternions_wxyz",
        "orientation_reference_robot_quaternions_wxyz",
        "orientation_reference_robot_qpos",
        "orientation_target_quaternions_wxyz",
        "orientation_robot_quaternions_wxyz",
        "orientation_errors_rad",
        "orientation_frame_costs",
        "fps",
        "interaction_source_vertices_w",
        "interaction_target_vertices_w",
        "interaction_tetrahedra",
        "interaction_tetrahedra_counts",
        "interaction_num_human_vertices",
        "interaction_num_object_vertices",
        "interaction_mesh_edges_default",
    }
)

HUMAN_ORIENTATION_KEYS = frozenset(
    {
        "human_orientation_joint_names",
        "human_orientation_quaternions_wxyz",
        "human_orientation_sha256",
    }
)

INTERACTION_MESH_KEYS = frozenset(
    {
        "interaction_source_vertices_w",
        "interaction_target_vertices_w",
        "interaction_tetrahedra",
        "interaction_tetrahedra_counts",
        "interaction_num_human_vertices",
        "interaction_num_object_vertices",
        "interaction_mesh_edges_default",
    }
)

OBJECT_POINT_KEYS = frozenset(
    {
        "object_points_demo_local",
        "object_points_target_local",
        "object_points_demo_world",
        "object_points_target_world",
    }
)

ALLOWED_RESULT_KEYS = REQUIRED_RESULT_KEYS | HUMAN_ORIENTATION_KEYS | OBJECT_POINT_KEYS

_LEGACY_HUMAN_ORIENTATION_KEYS = frozenset(
    {
        "global_joint_quaternions_wxyz",
        "human_joint_quaternions_wxyz",
    }
)
_NPZ_RESERVED_KEYS = frozenset({"allow_pickle", "file"})


class ResultArtifactValidationError(ValueError):
    """Raised when a result payload does not satisfy the current schema."""


def _fail(message: str) -> None:
    raise ResultArtifactValidationError(message)


def collision_interior_margin_m(penetration_tolerance: float) -> float:
    """Return the shared strict interior margin for collision constraints."""

    tolerance = float(penetration_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("penetration_tolerance must be finite and non-negative")
    return max(
        COLLISION_INTERIOR_MARGIN_MIN_M,
        min(
            COLLISION_INTERIOR_MARGIN_MAX_M,
            0.5 * tolerance,
        ),
    )


def _as_array(payload: Mapping[str, Any], key: str) -> np.ndarray:
    try:
        return np.asarray(payload[key])
    except (TypeError, ValueError) as exc:
        raise ResultArtifactValidationError(f"{key!r} cannot be represented as a NumPy array") from exc


def _require_no_object_arrays(payload: Mapping[str, Any]) -> None:
    for key, value in payload.items():
        if not isinstance(key, str) or not key:
            _fail("result artifact keys must be non-empty strings")
        try:
            array = np.asarray(value)
        except (TypeError, ValueError) as exc:
            raise ResultArtifactValidationError(f"{key!r} cannot be represented as a NumPy array") from exc
        if array.dtype.hasobject:
            _fail(f"{key!r} must not use object dtype because artifacts are loaded with allow_pickle=False")


def _decode_text(value: Any, *, key: str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ResultArtifactValidationError(f"{key!r} contains non-UTF-8 bytes") from exc
    raise ResultArtifactValidationError(f"{key!r} must contain strings")


def _require_scalar_text(
    payload: Mapping[str, Any],
    key: str,
    *,
    allow_empty: bool = False,
) -> str:
    array = _as_array(payload, key)
    if array.ndim != 0 or array.dtype.kind != "U":
        _fail(f"{key!r} must be a scalar NumPy Unicode string, got shape {array.shape} and dtype {array.dtype}")
    text = _decode_text(array.item(), key=key)
    if not allow_empty and not text.strip():
        _fail(f"{key!r} must be a non-empty string")
    if "\x00" in text:
        _fail(f"{key!r} must not contain NUL characters")
    return text


def _require_bool_scalar(payload: Mapping[str, Any], key: str) -> bool:
    array = _as_array(payload, key)
    if array.ndim != 0 or array.dtype.kind != "b":
        _fail(f"{key!r} must be a boolean scalar, got shape {array.shape} and dtype {array.dtype}")
    return bool(array.item())


def _require_string_vector(
    payload: Mapping[str, Any],
    key: str,
    *,
    expected_count: int,
    unique: bool,
) -> tuple[str, ...]:
    array = _as_array(payload, key)
    if array.ndim != 1 or array.shape != (expected_count,) or array.dtype.kind != "U":
        _fail(
            f"{key!r} must be a one-dimensional NumPy Unicode array with shape "
            f"({expected_count},), got {array.shape} and dtype {array.dtype}"
        )
    values = tuple(_decode_text(value, key=key) for value in array.tolist())
    if any(not value.strip() for value in values):
        _fail(f"{key!r} must not contain empty or whitespace-only strings")
    if any("\x00" in value for value in values):
        _fail(f"{key!r} must not contain NUL characters")
    if unique and len(set(values)) != len(values):
        _fail(f"{key!r} must contain unique strings")
    return values


def _require_names(
    payload: Mapping[str, Any],
    key: str,
    *,
    expected_count: int,
) -> tuple[str, ...]:
    return _require_string_vector(
        payload,
        key,
        expected_count=expected_count,
        unique=True,
    )


def _require_integer_scalar(
    payload: Mapping[str, Any],
    key: str,
    *,
    dtype: np.dtype[Any] | type[np.generic] = np.int32,
) -> int:
    array = _as_array(payload, key)
    expected_dtype = np.dtype(dtype)
    if array.ndim != 0 or array.dtype != expected_dtype:
        _fail(f"{key!r} must be a {expected_dtype} scalar, got shape {array.shape} and dtype {array.dtype}")
    return int(array.item())


def _require_frame_indices(
    payload: Mapping[str, Any],
    key: str,
    *,
    num_frames: int,
) -> np.ndarray:
    array = _as_array(payload, key)
    if array.ndim != 1 or array.dtype != np.dtype(np.int32):
        _fail(f"{key!r} must be a one-dimensional int32 array, got {array.shape} and dtype {array.dtype}")
    indices = array.astype(np.int64, copy=False)
    if np.any(indices < 0) or np.any(indices >= num_frames):
        _fail(f"{key!r} values must be valid frame indices in [0, {num_frames})")
    if indices.size and (len(np.unique(indices)) != len(indices) or np.any(indices[1:] <= indices[:-1])):
        _fail(f"{key!r} must contain unique frame indices in ascending order")
    return indices


def _require_bool_frame_array(
    payload: Mapping[str, Any],
    key: str,
    *,
    num_frames: int,
) -> np.ndarray:
    array = _as_array(payload, key)
    if array.shape != (num_frames,) or array.dtype.kind != "b":
        _fail(
            f"{key!r} must be a boolean array with shape ({num_frames},), got {array.shape} and dtype {array.dtype}",
        )
    return array


def _require_real_scalar(
    payload: Mapping[str, Any],
    key: str,
    *,
    dtype: np.dtype[Any] | type[np.generic] = np.float64,
) -> float:
    array = _as_array(payload, key)
    expected_dtype = np.dtype(dtype)
    if array.ndim != 0 or array.dtype != expected_dtype:
        _fail(f"{key!r} must be a {expected_dtype} scalar, got shape {array.shape} and dtype {array.dtype}")
    value = float(array.item())
    if not np.isfinite(value):
        _fail(f"{key!r} must be finite")
    return value


def _require_float_array(
    payload: Mapping[str, Any],
    key: str,
    *,
    shape: tuple[int, ...] | None = None,
    ndim: int | None = None,
    nonempty_axes: tuple[int, ...] = (),
    dtype: np.dtype[Any] | type[np.generic],
) -> np.ndarray:
    array = _as_array(payload, key)
    expected_dtype = np.dtype(dtype)
    if array.dtype != expected_dtype:
        _fail(f"{key!r} must use dtype {expected_dtype}, got {array.dtype}")
    if shape is not None and array.shape != shape:
        _fail(f"{key!r} must have shape {shape}, got {array.shape}")
    if ndim is not None and array.ndim != ndim:
        _fail(f"{key!r} must be {ndim}-D, got shape {array.shape}")
    for axis in nonempty_axes:
        if array.shape[axis] == 0:
            _fail(f"{key!r} axis {axis} must be non-empty, got shape {array.shape}")
    if not np.isfinite(array).all():
        _fail(f"{key!r} must contain only finite values")
    return array


def _validate_unit_quaternions(array: np.ndarray, *, key: str) -> None:
    norms = np.linalg.norm(array.astype(np.float64, copy=False), axis=-1)
    if not np.all(np.abs(norms - 1.0) <= QUATERNION_NORM_ATOL):
        max_error = float(np.max(np.abs(norms - 1.0)))
        _fail(
            f"{key!r} must contain unit wxyz quaternions within "
            f"{QUATERNION_NORM_ATOL:g}; maximum norm error is {max_error:g}"
        )


def compute_human_orientation_sha256(
    joint_names: tuple[str, ...] | list[str] | np.ndarray,
    quaternions_wxyz: np.ndarray,
) -> str:
    """Return the canonical digest for a direct human-orientation tensor.

    The digest domain-separates this contract, length-prefixes the compact
    UTF-8 JSON joint-name vector, includes the tensor shape as little-endian
    unsigned 64-bit integers, and finally includes C-contiguous little-endian
    float32 quaternion bytes. The validator separately enforces the canonical
    tensor dtype, shape, finiteness, and direct-source provenance.
    """

    names_array = np.asarray(joint_names)
    if names_array.ndim != 1 or names_array.dtype.kind not in {"U", "S"}:
        raise ValueError("joint_names must be a one-dimensional string array")
    names = [_decode_text(value, key="joint_names") for value in names_array.tolist()]
    tensor = np.asarray(quaternions_wxyz)
    if tensor.ndim != 3 or tensor.shape[-1] != 4:
        raise ValueError(f"quaternions_wxyz must have shape (frames, joints, 4), got {tensor.shape}")
    if tensor.shape[1] != len(names):
        raise ValueError(
            f"joint_names length must match the quaternion joint dimension; {len(names)} != {tensor.shape[1]}"
        )

    names_bytes = json.dumps(
        names,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    canonical_tensor = np.ascontiguousarray(tensor, dtype="<f4")
    shape_bytes = np.asarray(canonical_tensor.shape, dtype="<u8").tobytes(order="C")
    hasher = hashlib.sha256()
    hasher.update(_HUMAN_ORIENTATION_HASH_DOMAIN)
    hasher.update(len(names_bytes).to_bytes(8, byteorder="big"))
    hasher.update(names_bytes)
    hasher.update(shape_bytes)
    hasher.update(canonical_tensor.tobytes(order="C"))
    return hasher.hexdigest()


def _normalized_quaternions(array: np.ndarray) -> np.ndarray:
    values = array.astype(np.float64, copy=False)
    return values / np.linalg.norm(values, axis=-1, keepdims=True)


def _quaternion_multiply_wxyz(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    left_normalized = _normalized_quaternions(left)
    right_normalized = _normalized_quaternions(right)
    lw = left_normalized[..., :1]
    rw = right_normalized[..., :1]
    lv = left_normalized[..., 1:]
    rv = right_normalized[..., 1:]
    scalar = lw * rw - np.sum(lv * rv, axis=-1, keepdims=True)
    vector = lw * rv + rw * lv + np.cross(lv, rv)
    return np.concatenate((scalar, vector), axis=-1)


def _quaternion_inverse_wxyz(array: np.ndarray) -> np.ndarray:
    normalized = _normalized_quaternions(array)
    result = normalized.copy()
    result[..., 1:] *= -1.0
    return result


def _quaternion_geodesic_errors(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    left_normalized = _normalized_quaternions(left)
    right_normalized = _normalized_quaternions(right)
    absolute_dots = np.abs(np.sum(left_normalized * right_normalized, axis=-1))
    return 2.0 * np.arccos(np.clip(absolute_dots, 0.0, 1.0))


def _require_same_rotations(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    message: str,
) -> None:
    errors = _quaternion_geodesic_errors(actual, expected)
    if np.any(errors > ORIENTATION_RELATION_ATOL_RAD):
        _fail(f"{message}; maximum geodesic mismatch is {float(np.max(errors)):g} rad")


def _transform_local_points(
    local_points: np.ndarray,
    poses_xyz_wxyz: np.ndarray,
) -> np.ndarray:
    quaternions = _normalized_quaternions(poses_xyz_wxyz[:, 3:7])
    vectors = quaternions[:, None, 1:]
    scalar = quaternions[:, None, :1]
    points = local_points.astype(np.float64, copy=False)[None, :, :]
    twice_cross = 2.0 * np.cross(vectors, points)
    rotated = points + scalar * twice_cross + np.cross(vectors, twice_cross)
    return rotated + poses_xyz_wxyz[:, None, :3]


def _validate_parent_indices(
    payload: Mapping[str, Any],
    key: str,
    *,
    node_count: int,
) -> np.ndarray:
    parents = _as_array(payload, key)
    if parents.shape != (node_count,) or parents.dtype != np.dtype(np.int32):
        _fail(
            f"{key!r} must be an int32 array with shape ({node_count},), got {parents.shape} and dtype {parents.dtype}"
        )
    parents = parents.astype(np.int64, copy=False)
    if np.any(parents < -1) or np.any(parents >= node_count):
        _fail(f"{key!r} values must be -1 or valid link indices in [0, {node_count})")
    if np.any(parents == np.arange(node_count)):
        _fail(f"{key!r} must not contain self-parent links")
    roots = np.flatnonzero(parents == -1)
    if roots.size != 1:
        _fail(f"{key!r} must describe one rooted tree; found {roots.size} roots")
    root = int(roots[0])
    for start in range(node_count):
        current = start
        visited: set[int] = set()
        while current != -1:
            if current in visited:
                _fail(f"{key!r} contains a parent cycle involving link index {current}")
            visited.add(current)
            current = int(parents[current])
        if root not in visited:
            _fail(f"{key!r} link index {start} does not connect to root index {root}")
    return parents


def _validate_human_orientation_group(
    payload: Mapping[str, Any],
    *,
    num_frames: int,
    human_names: tuple[str, ...],
    source_data_format: str,
    orientation_source: str,
) -> tuple[tuple[str, ...], np.ndarray | None]:
    present = HUMAN_ORIENTATION_KEYS.intersection(payload)
    if not present:
        if orientation_source != ABSENT_ORIENTATION_SOURCE:
            _fail("'orientation_source' must be 'absent' when no source-human orientation tensor is saved")
        return (), None
    missing = HUMAN_ORIENTATION_KEYS.difference(payload)
    if missing:
        _fail(f"human source orientations must be stored as one complete group; missing {', '.join(sorted(missing))}")
    orientation_names_array = _as_array(payload, "human_orientation_joint_names")
    if orientation_names_array.ndim != 1:
        _fail(
            "'human_orientation_joint_names' must be a one-dimensional string array, "
            f"got {orientation_names_array.shape}"
        )
    orientation_count = int(orientation_names_array.shape[0])
    if not 1 <= orientation_count <= len(human_names):
        _fail(
            "'human_orientation_joint_names' must describe between 1 and "
            f"{len(human_names)} joints, got {orientation_count}"
        )
    orientation_names = _require_names(
        payload,
        "human_orientation_joint_names",
        expected_count=orientation_count,
    )
    unknown_names = sorted(set(orientation_names).difference(human_names))
    if unknown_names:
        _fail(
            f"'human_orientation_joint_names' must be a subset of 'human_joint_names'; unknown names: {unknown_names}"
        )
    quaternions = _require_float_array(
        payload,
        "human_orientation_quaternions_wxyz",
        shape=(num_frames, orientation_count, 4),
        dtype=np.float32,
    )
    _validate_unit_quaternions(
        quaternions,
        key="human_orientation_quaternions_wxyz",
    )
    orientation_sha256 = _require_scalar_text(
        payload,
        "human_orientation_sha256",
    )
    if _SHA256_PATTERN.fullmatch(orientation_sha256) is None:
        _fail("'human_orientation_sha256' must be a lowercase hexadecimal SHA-256 digest")
    expected_sha256 = compute_human_orientation_sha256(
        orientation_names,
        quaternions,
    )
    if orientation_sha256 != expected_sha256:
        _fail(
            "'human_orientation_sha256' must match the canonical joint names "
            "and float32 quaternion tensor bytes; "
            f"expected {expected_sha256}, got {orientation_sha256}"
        )
    approved_sources = APPROVED_DIRECT_ORIENTATION_SOURCES.get(
        source_data_format,
    )
    if approved_sources is None:
        _fail(f"'source_data_format' has no registered direct-orientation provenance contract: {source_data_format!r}")
    if orientation_source not in approved_sources:
        _fail(
            "'orientation_source' must explicitly identify an approved direct "
            f"source for {source_data_format!r}; expected one of "
            f"{sorted(approved_sources)}, got {orientation_source!r}"
        )
    return orientation_names, quaternions


def _validate_interaction_mesh_group(
    payload: Mapping[str, Any],
    *,
    num_frames: int,
    mapped_human_joints: np.ndarray,
    mapped_robot_joints: np.ndarray,
    object_points_demo_world: np.ndarray | None,
    object_points_target_world: np.ndarray | None,
) -> None:
    source_vertices = _require_float_array(
        payload,
        "interaction_source_vertices_w",
        ndim=3,
        nonempty_axes=(1,),
        dtype=np.float32,
    )
    target_vertices = _require_float_array(
        payload,
        "interaction_target_vertices_w",
        shape=source_vertices.shape,
        dtype=np.float32,
    )
    if source_vertices.shape[0] != num_frames or source_vertices.shape[2] != 3:
        _fail(
            "'interaction_source_vertices_w' and 'interaction_target_vertices_w' must "
            f"have shape ({num_frames}, vertices, 3), got {source_vertices.shape}"
        )
    num_vertices = int(source_vertices.shape[1])

    tetrahedra = _as_array(payload, "interaction_tetrahedra")
    if (
        tetrahedra.ndim != 3
        or tetrahedra.shape[0] != num_frames
        or tetrahedra.shape[2] != 4
        or tetrahedra.dtype != np.dtype(np.int32)
    ):
        _fail(
            "'interaction_tetrahedra' must be an int32 array with shape "
            f"({num_frames}, max_tetrahedra, 4), got {tetrahedra.shape} and dtype {tetrahedra.dtype}"
        )
    tetrahedra = tetrahedra.astype(np.int64, copy=False)
    max_tetrahedra = int(tetrahedra.shape[1])

    counts = _as_array(payload, "interaction_tetrahedra_counts")
    if counts.shape != (num_frames,) or counts.dtype != np.dtype(np.int32):
        _fail(
            "'interaction_tetrahedra_counts' must be an int32 array with shape "
            f"({num_frames},), got {counts.shape} and dtype {counts.dtype}"
        )
    counts = counts.astype(np.int64, copy=False)
    if np.any(counts < 0) or np.any(counts > max_tetrahedra):
        _fail(f"'interaction_tetrahedra_counts' values must be between 0 and {max_tetrahedra}, got {counts.tolist()}")
    expected_width = int(np.max(counts)) if counts.size else 0
    if max_tetrahedra != expected_width:
        _fail(
            "'interaction_tetrahedra' padded width must equal the maximum saved count; "
            f"got width {max_tetrahedra} and maximum count {expected_width}"
        )

    num_human_vertices = _require_integer_scalar(
        payload,
        "interaction_num_human_vertices",
        dtype=np.int32,
    )
    num_object_vertices = _require_integer_scalar(
        payload,
        "interaction_num_object_vertices",
        dtype=np.int32,
    )
    expected_human_vertices = int(mapped_human_joints.shape[1])
    if num_human_vertices != expected_human_vertices:
        _fail(
            "'interaction_num_human_vertices' must equal the mapped joint count "
            f"{expected_human_vertices}, got {num_human_vertices}"
        )
    if num_object_vertices < 0:
        _fail("'interaction_num_object_vertices' must be non-negative")
    if num_human_vertices + num_object_vertices != num_vertices:
        _fail(
            "Interaction Mesh vertex counts must sum to the saved vertex dimension; "
            f"{num_human_vertices} + {num_object_vertices} != {num_vertices}"
        )
    if not np.allclose(
        source_vertices[:, :num_human_vertices],
        mapped_human_joints,
        atol=POSITION_MATCH_ATOL,
        rtol=0.0,
    ):
        _fail("the human prefix of 'interaction_source_vertices_w' must match 'mapped_human_joints'")
    if not np.allclose(
        target_vertices[:, :num_human_vertices],
        mapped_robot_joints,
        atol=POSITION_MATCH_ATOL,
        rtol=0.0,
    ):
        _fail("the robot prefix of 'interaction_target_vertices_w' must match 'mapped_robot_joints'")

    for frame_index, count_value in enumerate(counts):
        count = int(count_value)
        active = tetrahedra[frame_index, :count]
        if active.size:
            if np.any(active < 0) or np.any(active >= num_vertices):
                _fail(f"'interaction_tetrahedra' contains an out-of-range active vertex index at frame {frame_index}")
            if np.any(np.diff(np.sort(active, axis=1), axis=1) == 0):
                _fail(f"'interaction_tetrahedra' contains a tetrahedron with repeated vertices at frame {frame_index}")
        padding = tetrahedra[frame_index, count:]
        if padding.size and np.any(padding != -1):
            _fail(
                "'interaction_tetrahedra' entries after each frame count must use -1 "
                f"padding; frame {frame_index} is invalid"
            )
        if active.shape[0] > 1:
            canonical_tetrahedra = np.sort(active, axis=1)
            if np.unique(canonical_tetrahedra, axis=0).shape[0] != active.shape[0]:
                _fail(f"'interaction_tetrahedra' contains duplicate active tetrahedra at frame {frame_index}")

    if object_points_demo_world is not None:
        if object_points_target_world is None:
            raise AssertionError("target object points must accompany demo points")
        if num_object_vertices != object_points_demo_world.shape[1]:
            _fail(
                "'interaction_num_object_vertices' must equal the dynamic "
                f"object-point count {object_points_demo_world.shape[1]}"
            )
        if not np.allclose(
            source_vertices[:, num_human_vertices:],
            object_points_demo_world,
            atol=POSITION_MATCH_ATOL,
            rtol=0.0,
        ):
            _fail("the object suffix of 'interaction_source_vertices_w' must match 'object_points_demo_world'")
        if not np.allclose(
            target_vertices[:, num_human_vertices:],
            object_points_target_world,
            atol=POSITION_MATCH_ATOL,
            rtol=0.0,
        ):
            _fail("the object suffix of 'interaction_target_vertices_w' must match 'object_points_target_world'")

    edge_mode = _require_scalar_text(
        payload,
        "interaction_mesh_edges_default",
    )
    if edge_mode not in {"all", "cross"}:
        _fail("'interaction_mesh_edges_default' must be either 'all' or 'cross'")


def _validate_hash_metadata(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    source_sha256 = _require_scalar_text(payload, "source_sha256")
    if _SHA256_PATTERN.fullmatch(source_sha256) is None:
        _fail("'source_sha256' must be a lowercase hexadecimal SHA-256 digest")

    config_json = _require_scalar_text(payload, "config_json")
    try:
        decoded_config = json.loads(config_json)
    except json.JSONDecodeError as exc:
        raise ResultArtifactValidationError("'config_json' must contain valid JSON") from exc
    if not isinstance(decoded_config, dict):
        _fail("'config_json' must encode a JSON object")

    config_sha256 = _require_scalar_text(payload, "config_sha256")
    expected_digest = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
    if config_sha256 != expected_digest:
        _fail(
            "'config_sha256' must equal SHA-256(config_json UTF-8 bytes); "
            f"expected {expected_digest}, got {config_sha256}"
        )
    return decoded_config


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _require_canonical_absolute_manifest_path(
    value: Any,
    *,
    key: str,
) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{key} must be a non-empty string")
    if "\x00" in value:
        _fail(f"{key} must not contain NUL characters")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ResultArtifactValidationError(
            f"{key} must be valid UTF-8 text",
        ) from exc
    path = Path(value)
    if not path.is_absolute():
        _fail(f"{key} must be an absolute path")
    if os.path.normpath(value) != value:
        _fail(f"{key} must be a lexically canonical absolute path")
    return value


def _validate_object_asset_file_entry(
    value: Any,
    *,
    key: str,
    include_kind: bool,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{key} must be a JSON object")
    expected_keys = {"path", "size", "sha256"}
    if include_kind:
        expected_keys.add("kind")
    if set(value) != expected_keys:
        _fail(
            f"{key} must contain exactly {sorted(expected_keys)}, got {sorted(value)}",
        )
    path = _require_canonical_absolute_manifest_path(
        value["path"],
        key=f"{key}.path",
    )
    size = value["size"]
    if type(size) is not int or size < 0:
        _fail(f"{key}.size must be a non-negative JSON integer")
    sha256 = value["sha256"]
    if not isinstance(sha256, str) or _SHA256_PATTERN.fullmatch(sha256) is None:
        _fail(f"{key}.sha256 must be a lowercase hexadecimal SHA-256 digest")
    normalized: dict[str, Any] = {
        "path": path,
        "sha256": sha256,
        "size": size,
    }
    if include_kind:
        kind = value["kind"]
        if kind not in {"mesh", "texture"}:
            _fail(f"{key}.kind must be either 'mesh' or 'texture'")
        normalized["kind"] = kind
    return normalized


def _validate_object_asset_manifest_metadata(
    *,
    object_urdf: str,
    object_urdf_sha256: str,
    manifest_json: str,
    manifest_sha256: str,
) -> dict[str, Any] | None:
    if not object_urdf:
        if object_urdf_sha256:
            _fail("'object_urdf_sha256' must be empty when 'object_urdf' is empty")
        if manifest_json:
            _fail("'object_asset_manifest_json' must be empty when 'object_urdf' is empty")
        if manifest_sha256:
            _fail("'object_asset_manifest_sha256' must be empty when 'object_urdf' is empty")
        return None

    if _SHA256_PATTERN.fullmatch(object_urdf_sha256) is None:
        _fail("'object_urdf_sha256' must be a lowercase hexadecimal SHA-256 digest")
    if not manifest_json:
        _fail("'object_asset_manifest_json' must be non-empty for a non-ground object asset")
    try:
        manifest_bytes = manifest_json.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ResultArtifactValidationError(
            "'object_asset_manifest_json' must contain valid UTF-8 text",
        ) from exc
    if len(manifest_bytes) > _OBJECT_ASSET_MANIFEST_MAX_BYTES:
        _fail(
            f"'object_asset_manifest_json' exceeds the canonical {_OBJECT_ASSET_MANIFEST_MAX_BYTES}-byte limit",
        )
    if _SHA256_PATTERN.fullmatch(manifest_sha256) is None:
        _fail("'object_asset_manifest_sha256' must be a lowercase hexadecimal SHA-256 digest")
    observed_manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if observed_manifest_sha256 != manifest_sha256:
        _fail(
            "'object_asset_manifest_sha256' must equal SHA-256 of the exact "
            "UTF-8 manifest JSON bytes; "
            f"expected {observed_manifest_sha256}, got {manifest_sha256}",
        )
    try:
        manifest = json.loads(manifest_json)
    except json.JSONDecodeError as exc:
        raise ResultArtifactValidationError(
            "'object_asset_manifest_json' must contain valid JSON",
        ) from exc
    if not isinstance(manifest, dict) or set(manifest) != {
        "dependencies",
        "urdf",
        "version",
    }:
        _fail(
            "'object_asset_manifest_json' must encode exactly version, urdf, and dependencies fields",
        )
    if type(manifest["version"]) is not int or manifest["version"] != _OBJECT_ASSET_MANIFEST_VERSION:
        _fail(
            f"'object_asset_manifest_json' version must equal {_OBJECT_ASSET_MANIFEST_VERSION}",
        )
    urdf_entry = _validate_object_asset_file_entry(
        manifest["urdf"],
        key="object_asset_manifest_json.urdf",
        include_kind=False,
    )
    if urdf_entry["path"] != object_urdf:
        _fail(
            "object_asset_manifest_json.urdf.path must exactly match "
            f"'object_urdf'; {urdf_entry['path']!r} != {object_urdf!r}",
        )
    if urdf_entry["sha256"] != object_urdf_sha256:
        _fail(
            "object_asset_manifest_json.urdf.sha256 must exactly match 'object_urdf_sha256'",
        )
    dependencies_value = manifest["dependencies"]
    if not isinstance(dependencies_value, list):
        _fail("object_asset_manifest_json.dependencies must be a JSON array")
    dependencies: list[dict[str, Any]] = []
    identities: list[tuple[str, str]] = []
    for index, dependency_value in enumerate(dependencies_value):
        dependency = _validate_object_asset_file_entry(
            dependency_value,
            key=f"object_asset_manifest_json.dependencies[{index}]",
            include_kind=True,
        )
        dependency_path = Path(dependency["path"])
        if dependency_path == Path(object_urdf):
            _fail("object_asset_manifest_json dependencies must not repeat the URDF itself")
        identity = (str(dependency["kind"]), str(dependency["path"]))
        identities.append(identity)
        dependencies.append(dependency)
    if identities != sorted(identities):
        _fail(
            "object_asset_manifest_json dependencies must be sorted by kind and absolute path",
        )
    if len(set(identities)) != len(identities):
        _fail("object_asset_manifest_json dependencies must not contain duplicate kind/path entries")
    normalized_manifest = {
        "dependencies": dependencies,
        "urdf": urdf_entry,
        "version": _OBJECT_ASSET_MANIFEST_VERSION,
    }
    if _canonical_json(normalized_manifest) != manifest_json:
        _fail(
            "'object_asset_manifest_json' must use canonical sorted-key, compact UTF-8 JSON encoding",
        )
    return normalized_manifest


def _validate_job_identity(
    payload: Mapping[str, Any],
    *,
    decoded_config: Mapping[str, Any],
    run_kind: str,
    variant: str,
    source_data_format: str,
    robot_type: str,
    task_type: str,
) -> tuple[str, str, str, str, str, bool]:
    dataset_partition = _require_scalar_text(
        payload,
        "dataset_partition",
    )
    sequence_key = _require_scalar_text(payload, "sequence_key")
    experiment_name = _require_scalar_text(
        payload,
        "experiment_name",
        allow_empty=True,
    )
    object_name = _require_scalar_text(payload, "object_name")
    object_urdf = _require_scalar_text(
        payload,
        "object_urdf",
        allow_empty=True,
    )
    object_urdf_sha256 = _require_scalar_text(
        payload,
        "object_urdf_sha256",
        allow_empty=True,
    )
    object_asset_manifest_json = _require_scalar_text(
        payload,
        "object_asset_manifest_json",
        allow_empty=True,
    )
    object_asset_manifest_sha256 = _require_scalar_text(
        payload,
        "object_asset_manifest_sha256",
        allow_empty=True,
    )

    if run_kind == "ablation" and not experiment_name:
        _fail("'experiment_name' must be non-empty for ablation results")
    if run_kind != "ablation" and experiment_name:
        _fail("'experiment_name' must be empty for single and augmentation results")
    if "/" in dataset_partition or "\\" in dataset_partition:
        _fail("'dataset_partition' must be one path component")
    if dataset_partition in {".", ".."}:
        _fail("'dataset_partition' must not be '.' or '..'")
    sequence_parts = sequence_key.split("/")
    if sequence_key.startswith("/") or any(part in {"", ".", ".."} for part in sequence_parts):
        _fail("'sequence_key' must be a non-empty relative POSIX path without empty, '.' or '..' components")

    expected_job_values: dict[str, Any] = {
        "run_kind": run_kind,
        "experiment_name": experiment_name or None,
        "dataset_partition": dataset_partition,
        "sequence_key": sequence_key,
    }
    for key, expected in expected_job_values.items():
        if decoded_config.get(key) != expected:
            _fail(
                f"{key!r} must match the same field in 'config_json'; "
                f"expected {decoded_config.get(key)!r}, got {expected!r}"
            )
    variant_config = decoded_config.get("variant")
    if not isinstance(variant_config, Mapping) or variant_config.get("name") != variant:
        _fail(
            "'variant' must match config_json['variant']['name']; "
            f"got artifact {variant!r} and config {variant_config!r}"
        )
    try:
        translation = np.asarray(variant_config["translation"], dtype=float)
        rotation = float(variant_config["rotation"])
        object_scale = np.asarray(variant_config["object_scale"], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ResultArtifactValidationError(
            "config_json['variant'] must contain numeric translation, rotation, and object_scale"
        ) from exc
    if (
        translation.shape != (3,)
        or object_scale.shape != (3,)
        or not np.isfinite(translation).all()
        or not np.isfinite(rotation)
        or not np.isfinite(object_scale).all()
        or np.any(object_scale <= 0.0)
    ):
        _fail("config_json['variant'] contains an invalid motion transformation")
    changes_motion = (
        not np.allclose(translation, 0.0) or not np.isclose(rotation, 0.0) or not np.allclose(object_scale, 1.0)
    )
    if run_kind == "single" and (variant != "identity" or changes_motion):
        _fail("single results require the exact identity variant")
    if run_kind == "augmentation" and not changes_motion:
        _fail("augmentation results require a non-identity motion transformation")
    if run_kind == "ablation" and changes_motion:
        _fail("ablation results must not transform motion")
    solver_config = decoded_config.get("config")
    if not isinstance(solver_config, Mapping):
        _fail("'config_json' must contain a 'config' JSON object")
    expected_solver_values = {
        "data_format": source_data_format,
        "task_type": task_type,
    }
    for key, expected in expected_solver_values.items():
        if solver_config.get(key) != expected:
            _fail(f"{key!r} must match config_json['config']; expected {solver_config.get(key)!r}, got {expected!r}")
    robot_config = solver_config.get("robot_config")
    if not isinstance(robot_config, Mapping) or robot_config.get("robot_type") != robot_type:
        _fail("'robot_type' must match config_json['config']['robot_config']['robot_type']")
    task_config = solver_config.get("task_config")
    if not isinstance(task_config, Mapping) or task_config.get("object_name") != object_name:
        _fail("'object_name' must match config_json['config']['task_config']['object_name']")
    retargeter_config = solver_config.get("retargeter")
    if not isinstance(retargeter_config, Mapping):
        _fail("'config_json' must contain config.retargeter")
    activate_object_non_penetration = retargeter_config.get(
        "activate_obj_non_penetration",
    )
    if not isinstance(activate_object_non_penetration, bool):
        _fail("config_json['config']['retargeter']['activate_obj_non_penetration'] must be boolean")
    expected_object_eligibility = task_type != "robot_only" and activate_object_non_penetration
    return (
        object_name,
        object_urdf,
        object_urdf_sha256,
        object_asset_manifest_json,
        object_asset_manifest_sha256,
        expected_object_eligibility,
    )


def _validate_task_object_contract(
    *,
    task_type: str,
    object_name: str,
    object_urdf: str,
    object_urdf_sha256: str,
    object_asset_manifest_json: str,
    object_asset_manifest_sha256: str,
    contains_object: bool,
    object_non_penetration_eligible: bool,
    expected_object_non_penetration_eligibility: bool,
) -> None:
    if object_non_penetration_eligible != expected_object_non_penetration_eligibility:
        _fail(
            "'object_non_penetration_eligible_for_saved_trajectory' must "
            "match the normalized retargeter/task configuration"
        )
    if task_type == "robot_only":
        if object_name != "ground":
            _fail("'object_name' must be 'ground' for robot_only results")
        if object_urdf:
            _fail("'object_urdf' must be empty for canonical ground results")
        if object_urdf_sha256:
            _fail("'object_urdf_sha256' must be empty for canonical ground results")
        if object_asset_manifest_json:
            _fail("'object_asset_manifest_json' must be empty for canonical ground results")
        if object_asset_manifest_sha256:
            _fail("'object_asset_manifest_sha256' must be empty for canonical ground results")
        if contains_object:
            _fail("'contains_object_in_qpos' must be false for robot_only results")
        if object_non_penetration_eligible:
            _fail("'object_non_penetration_eligible_for_saved_trajectory' must be false for canonical ground results")
        return
    if task_type == "object_interaction":
        if object_name == "ground":
            _fail("'object_name' must identify a non-ground asset for object_interaction results")
        if not object_urdf:
            _fail("'object_urdf' must identify the retargeting asset for object_interaction results")
        if not Path(object_urdf).is_absolute():
            _fail("'object_urdf' must be an absolute path for canonical non-ground results")
        if _SHA256_PATTERN.fullmatch(object_urdf_sha256) is None:
            _fail("'object_urdf_sha256' must be a lowercase hexadecimal SHA-256 digest")
        _validate_object_asset_manifest_metadata(
            object_urdf=object_urdf,
            object_urdf_sha256=object_urdf_sha256,
            manifest_json=object_asset_manifest_json,
            manifest_sha256=object_asset_manifest_sha256,
        )
        if not contains_object:
            _fail("'contains_object_in_qpos' must be true for object_interaction results")
        return
    if task_type == "climbing":
        if object_name != "multi_boxes":
            _fail("'object_name' must be 'multi_boxes' for climbing results")
        if not object_urdf:
            _fail("'object_urdf' must identify the terrain asset for climbing results")
        if not Path(object_urdf).is_absolute():
            _fail("'object_urdf' must be an absolute path for canonical non-ground results")
        if _SHA256_PATTERN.fullmatch(object_urdf_sha256) is None:
            _fail("'object_urdf_sha256' must be a lowercase hexadecimal SHA-256 digest")
        _validate_object_asset_manifest_metadata(
            object_urdf=object_urdf,
            object_urdf_sha256=object_urdf_sha256,
            manifest_json=object_asset_manifest_json,
            manifest_sha256=object_asset_manifest_sha256,
        )
        if contains_object:
            _fail("'contains_object_in_qpos' must be false for climbing results")
        return
    _fail("'task_type' must be 'robot_only', 'object_interaction', or 'climbing'")


def _validate_qpos_and_object_poses(
    payload: Mapping[str, Any],
    *,
    qpos: np.ndarray,
    num_frames: int,
    num_actuated_joints: int,
) -> tuple[bool, np.ndarray, np.ndarray]:
    qpos_layout = _require_scalar_text(payload, "qpos_layout")
    if qpos_layout != QPOS_LAYOUT:
        _fail(f"'qpos_layout' must be {QPOS_LAYOUT!r}, got {qpos_layout!r}")
    contains_object = _require_bool_scalar(
        payload,
        "contains_object_in_qpos",
    )
    expected_qpos_width = 7 + num_actuated_joints + (7 if contains_object else 0)
    if qpos.shape[1] != expected_qpos_width:
        _fail(
            "'qpos' width must match the free robot root, "
            "'robot_actuated_joint_names', and optional object free joint; "
            f"expected {expected_qpos_width}, got {qpos.shape[1]}"
        )
    _validate_unit_quaternions(qpos[:, 3:7], key="qpos robot root")

    object_pose_layout = _require_scalar_text(payload, "object_pose_layout")
    if object_pose_layout != OBJECT_POSE_LAYOUT:
        _fail(f"'object_pose_layout' must be {OBJECT_POSE_LAYOUT!r}, got {object_pose_layout!r}")
    object_poses_demo = _require_float_array(
        payload,
        "object_poses_demo",
        shape=(num_frames, 7),
        dtype=np.float32,
    )
    object_poses_target = _require_float_array(
        payload,
        "object_poses_target",
        shape=(num_frames, 7),
        dtype=np.float32,
    )
    _validate_unit_quaternions(
        object_poses_demo[:, 3:7],
        key="object_poses_demo",
    )
    _validate_unit_quaternions(
        object_poses_target[:, 3:7],
        key="object_poses_target",
    )
    if contains_object and not np.allclose(
        qpos[:, -7:],
        object_poses_target,
        atol=POSITION_MATCH_ATOL,
        rtol=0.0,
    ):
        _fail("the object free-joint suffix of 'qpos' must match 'object_poses_target'")
    return contains_object, object_poses_demo, object_poses_target


def _validate_object_point_group(
    payload: Mapping[str, Any],
    *,
    num_frames: int,
    contains_object: bool,
    object_poses_demo: np.ndarray,
    object_poses_target: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    present = OBJECT_POINT_KEYS.intersection(payload)
    if not present:
        if contains_object:
            _fail("dynamic-object results must save the complete object-point group")
        return None, None
    missing = OBJECT_POINT_KEYS.difference(payload)
    if missing:
        _fail(f"object-point fields must be stored as one complete group; missing {', '.join(sorted(missing))}")
    if not contains_object:
        _fail("object-point trajectories are only valid when 'contains_object_in_qpos' is true")

    demo_local = _require_float_array(
        payload,
        "object_points_demo_local",
        ndim=2,
        nonempty_axes=(0,),
        dtype=np.float32,
    )
    if demo_local.shape[1] != 3:
        _fail(f"'object_points_demo_local' must have shape (points, 3), got {demo_local.shape}")
    point_count = int(demo_local.shape[0])
    target_local = _require_float_array(
        payload,
        "object_points_target_local",
        shape=(point_count, 3),
        dtype=np.float32,
    )
    demo_world = _require_float_array(
        payload,
        "object_points_demo_world",
        shape=(num_frames, point_count, 3),
        dtype=np.float32,
    )
    target_world = _require_float_array(
        payload,
        "object_points_target_world",
        shape=(num_frames, point_count, 3),
        dtype=np.float32,
    )
    expected_demo_world = _transform_local_points(
        demo_local,
        object_poses_demo,
    )
    if not np.allclose(
        demo_world,
        expected_demo_world,
        atol=OBJECT_TRANSFORM_MATCH_ATOL,
        rtol=0.0,
    ):
        _fail("'object_points_demo_world' must equal object_poses_demo ∘ object_points_demo_local")
    expected_target_world = _transform_local_points(
        target_local,
        object_poses_target,
    )
    if not np.allclose(
        target_world,
        expected_target_world,
        atol=OBJECT_TRANSFORM_MATCH_ATOL,
        rtol=0.0,
    ):
        _fail("'object_points_target_world' must equal object_poses_target ∘ object_points_target_local")
    return demo_world, target_world


def _validate_preprocessing_metadata(payload: Mapping[str, Any]) -> None:
    source_height = _require_real_scalar(
        payload,
        "source_human_height",
        dtype=np.float64,
    )
    if source_height <= 0.0:
        _fail("'source_human_height' must be positive")
    position_scale = _require_real_scalar(
        payload,
        "human_position_scale",
        dtype=np.float64,
    )
    if position_scale <= 0.0:
        _fail("'human_position_scale' must be positive")
    _require_scalar_text(payload, "human_position_preprocessing")


def _validate_foot_diagnostics(
    payload: Mapping[str, Any],
    *,
    num_frames: int,
    object_non_penetration_eligible: bool,
) -> tuple[np.ndarray, np.ndarray]:
    side_names = _require_names(
        payload,
        "foot_sticking_side_names",
        expected_count=2,
    )
    if side_names != ("left", "right"):
        _fail("'foot_sticking_side_names' must use canonical order ('left', 'right')")
    states = _as_array(payload, "foot_sticking_states")
    if states.shape != (num_frames, 2) or states.dtype.kind != "b":
        _fail(
            "'foot_sticking_states' must be a boolean array with shape "
            f"({num_frames}, 2), got {states.shape} and dtype {states.dtype}"
        )
    tolerance = _require_real_scalar(
        payload,
        "foot_sticking_tolerance",
        dtype=np.float64,
    )
    if tolerance < 0.0:
        _fail("'foot_sticking_tolerance' must be non-negative")
    fallback_tolerance_array = _as_array(
        payload,
        "foot_sticking_fallback_tolerance",
    )
    if fallback_tolerance_array.ndim != 0 or fallback_tolerance_array.dtype != np.dtype(np.float64):
        _fail("'foot_sticking_fallback_tolerance' must be a float64 scalar")
    fallback_tolerance = float(fallback_tolerance_array.item())
    if not np.isnan(fallback_tolerance) and (not np.isfinite(fallback_tolerance) or fallback_tolerance < tolerance):
        _fail(
            "'foot_sticking_fallback_tolerance' must be NaN (disabled) or finite and at least 'foot_sticking_tolerance'"
        )

    fallback_frames = _require_frame_indices(
        payload,
        "foot_sticking_fallback_frames",
        num_frames=num_frames,
    )
    foot_release_enabled = _require_bool_scalar(
        payload,
        "release_foot_sticking_on_infeasible",
    )
    foot_release_frames = _require_frame_indices(
        payload,
        "foot_sticking_release_frames",
        num_frames=num_frames,
    )
    object_release_enabled = _require_bool_scalar(
        payload,
        "release_object_non_penetration_on_infeasible",
    )
    object_release_frames = _require_frame_indices(
        payload,
        "object_non_penetration_release_frames",
        num_frames=num_frames,
    )
    foot_sticking_enabled = _require_bool_scalar(
        payload,
        "foot_sticking_enabled_for_saved_trajectory",
    )
    retry_frame = _require_integer_scalar(
        payload,
        "foot_sticking_full_sequence_retry_frame",
    )
    if retry_frame < -1 or retry_frame >= num_frames:
        _fail(f"'foot_sticking_full_sequence_retry_frame' must be -1 or a valid frame index in [0, {num_frames})")
    foot_modes = np.asarray(
        _require_string_vector(
            payload,
            "constraint_mode_foot_sticking",
            expected_count=num_frames,
            unique=False,
        ),
    )
    allowed_foot_modes = {"inactive", "normal", "relaxed", "released"}
    unknown_foot_modes = sorted(set(foot_modes.tolist()).difference(allowed_foot_modes))
    if unknown_foot_modes:
        _fail(
            "'constraint_mode_foot_sticking' values must be inactive, normal, "
            f"relaxed, or released; unknown values: {unknown_foot_modes}",
        )
    object_released = _require_bool_frame_array(
        payload,
        "constraint_mode_object_non_penetration_released",
        num_frames=num_frames,
    )
    trust_region_released = _require_bool_frame_array(
        payload,
        "constraint_mode_trust_region_released",
        num_frames=num_frames,
    )

    if fallback_frames.size and np.isnan(fallback_tolerance):
        _fail("'foot_sticking_fallback_frames' must be empty when fallback tolerance is disabled")
    if foot_release_frames.size and not foot_release_enabled:
        _fail("'foot_sticking_release_frames' must be empty when 'release_foot_sticking_on_infeasible' is false")
    if object_release_frames.size and not object_release_enabled:
        _fail(
            "'object_non_penetration_release_frames' must be empty when "
            "'release_object_non_penetration_on_infeasible' is false"
        )
    if object_release_frames.size and not object_non_penetration_eligible:
        _fail(
            "'object_non_penetration_release_frames' must be empty when "
            "object non-penetration was not eligible for the saved trajectory"
        )
    if retry_frame >= 0 and foot_sticking_enabled:
        _fail(
            "'foot_sticking_enabled_for_saved_trajectory' must be false when "
            "'foot_sticking_full_sequence_retry_frame' records a complete-sequence retry",
        )

    detected_sticking = np.any(states, axis=1)
    expected_inactive = ~(foot_sticking_enabled & detected_sticking)
    observed_inactive = foot_modes == "inactive"
    if not np.array_equal(observed_inactive, expected_inactive):
        frames = np.flatnonzero(observed_inactive != expected_inactive).tolist()
        _fail(
            "'constraint_mode_foot_sticking' must be inactive exactly when "
            "foot sticking is disabled for the saved trajectory or neither foot "
            f"is detected as sticking; inconsistent frames: {frames}",
        )

    fallback_available = bool(
        np.isfinite(fallback_tolerance) and fallback_tolerance > tolerance,
    )
    if np.any(foot_modes == "relaxed") and not fallback_available:
        _fail(
            "'constraint_mode_foot_sticking' cannot contain relaxed when no "
            "strictly larger fallback tolerance is available",
        )
    expected_fallback = np.flatnonzero(
        (foot_modes == "relaxed") | ((foot_modes == "released") & fallback_available),
    )
    if not np.array_equal(fallback_frames, expected_fallback):
        _fail(
            "'foot_sticking_fallback_frames' must exactly match relaxed modes "
            "and released modes for which a relaxed fallback was available",
        )
    expected_foot_release = np.flatnonzero(foot_modes == "released")
    if not np.array_equal(foot_release_frames, expected_foot_release):
        _fail(
            "'foot_sticking_release_frames' must exactly match released entries in 'constraint_mode_foot_sticking'",
        )

    expected_object_release = np.flatnonzero(object_released)
    if not np.array_equal(object_release_frames, expected_object_release):
        _fail(
            "'object_non_penetration_release_frames' must exactly match true "
            "entries in 'constraint_mode_object_non_penetration_released'",
        )
    if np.any(object_released) and not object_non_penetration_eligible:
        _fail(
            "'constraint_mode_object_non_penetration_released' must be false "
            "when object non-penetration was not eligible for the saved trajectory",
        )
    if np.any(trust_region_released[1:]):
        _fail(
            "'constraint_mode_trust_region_released' may only be true on the "
            "initial frame, where the canonical solver permits that fallback",
        )
    return foot_modes, object_released


def _validate_nonlinear_constraint_diagnostics(
    payload: Mapping[str, Any],
    *,
    num_frames: int,
    foot_modes: np.ndarray,
    object_released: np.ndarray,
) -> None:
    """Validate every true-geometry residual against its accepted mode."""

    residuals = {
        key: _require_float_array(
            payload,
            key,
            shape=(num_frames,),
            dtype=np.float64,
        )
        for key in (
            "ground_non_penetration_violation",
            "object_non_penetration_violation",
            "foot_sticking_violation",
            "foot_lock_violation",
            "self_collision_violation",
            "joint_limits_violation",
        )
    }
    for key, values in residuals.items():
        if np.any(values < 0.0):
            _fail(f"{key!r} must be non-negative")

    always_hard = (
        "ground_non_penetration_violation",
        "foot_lock_violation",
        "self_collision_violation",
        "joint_limits_violation",
    )
    for key in always_hard:
        if np.any(residuals[key] > NONLINEAR_CONSTRAINT_VIOLATION_ATOL):
            frames = np.flatnonzero(
                residuals[key] > NONLINEAR_CONSTRAINT_VIOLATION_ATOL,
            ).tolist()
            _fail(
                f"{key!r} exceeds the hard nonlinear acceptance tolerance "
                f"{NONLINEAR_CONSTRAINT_VIOLATION_ATOL:g} on frames {frames}",
            )

    violating_unreleased_object = (
        residuals["object_non_penetration_violation"] > NONLINEAR_CONSTRAINT_VIOLATION_ATOL
    ) & ~object_released
    if np.any(violating_unreleased_object):
        frames = np.flatnonzero(violating_unreleased_object).tolist()
        _fail(
            "'object_non_penetration_violation' exceeds the hard nonlinear "
            "acceptance tolerance on frames whose accepted ConstraintMode did "
            f"not release object non-penetration: {frames}",
        )

    foot_constraint_active = np.isin(foot_modes, ("normal", "relaxed"))
    violating_active_foot = (
        residuals["foot_sticking_violation"] > NONLINEAR_CONSTRAINT_VIOLATION_ATOL
    ) & foot_constraint_active
    if np.any(violating_active_foot):
        frames = np.flatnonzero(violating_active_foot).tolist()
        _fail(
            "'foot_sticking_violation' exceeds the hard nonlinear acceptance "
            "tolerance on frames whose accepted foot-sticking mode remained "
            f"active: {frames}",
        )


def _validate_solver_diagnostics(
    payload: Mapping[str, Any],
    *,
    num_frames: int,
) -> None:
    frame_costs = _require_float_array(
        payload,
        "frame_costs",
        shape=(num_frames,),
        dtype=np.float64,
    )
    iteration_counts = _as_array(payload, "sqp_iteration_counts")
    if iteration_counts.shape != (num_frames,) or iteration_counts.dtype != np.dtype(np.int32):
        _fail(
            "'sqp_iteration_counts' must be an int32 array with shape "
            f"({num_frames},), got {iteration_counts.shape} and dtype "
            f"{iteration_counts.dtype}"
        )
    if np.any(iteration_counts <= 0):
        _fail("'sqp_iteration_counts' must contain positive iteration counts")
    _require_string_vector(
        payload,
        "sqp_stop_reasons",
        expected_count=num_frames,
        unique=False,
    )
    final_cost = _require_real_scalar(
        payload,
        "cost",
        dtype=np.float64,
    )
    if not np.isclose(
        final_cost,
        frame_costs[-1],
        atol=1e-9,
        rtol=1e-7,
    ):
        _fail("'cost' must match the final entry in 'frame_costs'")


def _validate_frame_zero_ground_retry(
    payload: Mapping[str, Any],
    *,
    decoded_config: Mapping[str, Any],
    run_kind: str,
    task_type: str,
    num_frames: int,
) -> None:
    """Validate the fail-closed robot-only frame-zero ground retry audit."""

    policy = _require_scalar_text(
        payload,
        "frame_zero_ground_retry_policy",
    )
    if policy != FRAME_ZERO_GROUND_RETRY_POLICY:
        _fail(
            f"'frame_zero_ground_retry_policy' must be {FRAME_ZERO_GROUND_RETRY_POLICY!r}",
        )
    eligible = _require_bool_scalar(
        payload,
        "frame_zero_ground_retry_eligible",
    )
    triggered = _require_bool_scalar(
        payload,
        "frame_zero_ground_retry_triggered",
    )
    initial_distance = _require_real_scalar(
        payload,
        "frame_zero_ground_retry_initial_min_distance_m",
        dtype=np.float64,
    )
    corrected_distance = _require_real_scalar(
        payload,
        "frame_zero_ground_retry_corrected_min_distance_m",
        dtype=np.float64,
    )
    lift = _require_real_scalar(
        payload,
        "frame_zero_ground_retry_lift_m",
        dtype=np.float64,
    )
    margin = _require_real_scalar(
        payload,
        "frame_zero_ground_retry_interior_margin_m",
        dtype=np.float64,
    )
    initial_iterations = _require_integer_scalar(
        payload,
        "frame_zero_ground_retry_initial_sqp_iterations",
        dtype=np.int32,
    )

    solver_config = decoded_config.get("config")
    if not isinstance(solver_config, Mapping):
        _fail("'config_json' must contain a 'config' JSON object")
    retargeter_config = solver_config.get("retargeter")
    if not isinstance(retargeter_config, Mapping):
        _fail("'config_json' must contain config.retargeter")
    retry_enabled = retargeter_config.get(
        "retry_frame_zero_ground_on_infeasible",
    )
    if not isinstance(retry_enabled, bool):
        _fail(
            "config_json['config']['retargeter']['retry_frame_zero_ground_on_infeasible'] must be boolean",
        )
    q_a_init_idx = retargeter_config.get("q_a_init_idx")
    if not isinstance(q_a_init_idx, int) or isinstance(q_a_init_idx, bool):
        _fail("config_json['config']['retargeter']['q_a_init_idx'] must be an integer")
    sqp_max_iterations = retargeter_config.get("sqp_max_iterations")
    if not isinstance(sqp_max_iterations, int) or isinstance(sqp_max_iterations, bool) or sqp_max_iterations <= 0:
        _fail(
            "config_json['config']['retargeter']['sqp_max_iterations'] must be a positive integer",
        )
    penetration_tolerance_value = retargeter_config.get(
        "penetration_tolerance",
    )
    if not isinstance(penetration_tolerance_value, (int, float)) or isinstance(penetration_tolerance_value, bool):
        _fail(
            "config_json['config']['retargeter']['penetration_tolerance'] must be numeric",
        )
    penetration_tolerance = float(penetration_tolerance_value)
    try:
        expected_margin = collision_interior_margin_m(
            penetration_tolerance,
        )
    except ValueError as exc:
        raise ResultArtifactValidationError(
            "config_json['config']['retargeter']['penetration_tolerance'] must be finite and non-negative",
        ) from exc

    expected_eligible = (
        retry_enabled and task_type == "robot_only" and run_kind in {"single", "ablation"} and 7 + q_a_init_idx <= 2
    )
    if eligible != expected_eligible:
        _fail(
            "'frame_zero_ground_retry_eligible' must match the normalized "
            "run kind, task type, root-z optimization, and retargeter policy",
        )
    stop_reasons = _require_string_vector(
        payload,
        "sqp_stop_reasons",
        expected_count=num_frames,
        unique=False,
    )
    retry_reason_frames = [
        frame_index for frame_index, reason in enumerate(stop_reasons) if reason.startswith("frame_zero_ground_retry:")
    ]

    if not triggered:
        if (
            any(
                value != 0.0
                for value in (
                    initial_distance,
                    corrected_distance,
                    lift,
                    margin,
                )
            )
            or initial_iterations != 0
        ):
            _fail(
                "Untriggered frame-zero ground retry diagnostics must use zero-valued numeric sentinels",
            )
        if retry_reason_frames:
            _fail(
                "'sqp_stop_reasons' must not record a frame-zero ground retry when the retry was not triggered",
            )
        return

    if not eligible:
        _fail(
            "'frame_zero_ground_retry_triggered' requires an eligible artifact",
        )
    if retry_reason_frames != [0]:
        _fail(
            "A triggered frame-zero ground retry must be recorded exactly on frame zero in 'sqp_stop_reasons'",
        )
    if initial_iterations <= 0:
        _fail(
            "'frame_zero_ground_retry_initial_sqp_iterations' must be positive when the retry was triggered",
        )
    if initial_iterations > sqp_max_iterations:
        _fail(
            "'frame_zero_ground_retry_initial_sqp_iterations' must not exceed config.retargeter.sqp_max_iterations",
        )
    foot_modes = _require_string_vector(
        payload,
        "constraint_mode_foot_sticking",
        expected_count=num_frames,
        unique=False,
    )
    if foot_modes[0] != "inactive":
        _fail(
            "A triggered frame-zero ground retry requires an inactive frame-zero foot-sticking mode",
        )
    if margin != expected_margin:
        _fail(
            "'frame_zero_ground_retry_interior_margin_m' must equal the shared collision interior margin policy",
        )
    if initial_distance >= -penetration_tolerance - NONLINEAR_CONSTRAINT_VIOLATION_ATOL:
        _fail(
            "'frame_zero_ground_retry_initial_min_distance_m' must describe a strict ground violation",
        )
    expected_lift = max(
        0.0,
        -initial_distance - penetration_tolerance + expected_margin,
    )
    if lift <= 0.0 or not np.isclose(
        lift,
        expected_lift,
        atol=1e-12,
        rtol=1e-12,
    ):
        _fail(
            "'frame_zero_ground_retry_lift_m' must be the one-step strict horizontal-ground correction",
        )
    if corrected_distance < -penetration_tolerance - NONLINEAR_CONSTRAINT_VIOLATION_ATOL:
        _fail(
            "'frame_zero_ground_retry_corrected_min_distance_m' remains outside the hard ground acceptance tolerance",
        )
    if not np.isclose(
        corrected_distance,
        initial_distance + lift,
        atol=1e-8,
        rtol=0.0,
    ):
        _fail(
            "'frame_zero_ground_retry_corrected_min_distance_m' must match the recorded horizontal lift",
        )


def _validate_orientation_diagnostics(
    payload: Mapping[str, Any],
    *,
    qpos_width: int,
    num_frames: int,
    human_names: tuple[str, ...],
    robot_names: tuple[str, ...],
    source_orientation_names: tuple[str, ...],
    source_orientation_quaternions: np.ndarray | None,
    robot_link_quaternions: np.ndarray,
    object_poses_demo: np.ndarray,
    object_poses_target: np.ndarray,
) -> None:
    human_names_array = _as_array(payload, "orientation_human_joint_names")
    if human_names_array.ndim != 1:
        _fail("'orientation_human_joint_names' must be one-dimensional")
    tracked_count = int(human_names_array.shape[0])
    tracked_human_names = _require_string_vector(
        payload,
        "orientation_human_joint_names",
        expected_count=tracked_count,
        unique=True,
    )
    tracked_robot_names = _require_string_vector(
        payload,
        "orientation_robot_link_names",
        expected_count=tracked_count,
        unique=True,
    )
    unknown_human = sorted(set(tracked_human_names).difference(human_names))
    if unknown_human:
        _fail(f"'orientation_human_joint_names' contains unknown source joints: {unknown_human}")
    unknown_robot = sorted(set(tracked_robot_names).difference(robot_names))
    if unknown_robot:
        _fail(f"'orientation_robot_link_names' contains unknown robot links: {unknown_robot}")
    unavailable_source = sorted(set(tracked_human_names).difference(source_orientation_names))
    if unavailable_source:
        _fail(
            "orientation diagnostics may only reference saved direct-source "
            f"human orientations; unavailable joints: {unavailable_source}"
        )
    if tracked_count and source_orientation_quaternions is None:
        _fail("orientation diagnostics require the saved direct-source human orientation tensor")

    weights = _require_float_array(
        payload,
        "orientation_weights",
        shape=(tracked_count,),
        dtype=np.float64,
    )
    if np.any(weights < 0.0):
        _fail("'orientation_weights' must be non-negative")
    diagnostics_enabled = _require_bool_scalar(
        payload,
        "orientation_diagnostics_enabled",
    )
    tracking_enabled = _require_bool_scalar(
        payload,
        "orientation_tracking_enabled",
    )
    if diagnostics_enabled != (tracked_count > 0):
        _fail("'orientation_diagnostics_enabled' must equal whether tracked orientation joints are present")
    if tracking_enabled != bool(np.any(weights > 0.0)):
        _fail("'orientation_tracking_enabled' must equal whether any orientation weight is positive")

    alignment_mode = _require_scalar_text(
        payload,
        "orientation_alignment_mode",
    )
    if alignment_mode not in {"t_pose", "first_frame", "explicit"}:
        _fail("'orientation_alignment_mode' must be 't_pose', 'first_frame', or 'explicit'")
    alignment = _require_float_array(
        payload,
        "orientation_alignment_quaternions_wxyz",
        shape=(tracked_count, 4),
        dtype=np.float32,
    )
    if tracked_count:
        _validate_unit_quaternions(
            alignment,
            key="orientation_alignment_quaternions_wxyz",
        )

    reference_count = tracked_count if tracked_count and alignment_mode in {"t_pose", "first_frame"} else 0
    reference_human = _require_float_array(
        payload,
        "orientation_reference_human_quaternions_wxyz",
        shape=(reference_count, 4),
        dtype=np.float32,
    )
    reference_robot = _require_float_array(
        payload,
        "orientation_reference_robot_quaternions_wxyz",
        shape=(reference_count, 4),
        dtype=np.float32,
    )
    if reference_count:
        _validate_unit_quaternions(
            reference_human,
            key="orientation_reference_human_quaternions_wxyz",
        )
        _validate_unit_quaternions(
            reference_robot,
            key="orientation_reference_robot_quaternions_wxyz",
        )
        expected_alignment = _quaternion_multiply_wxyz(
            _quaternion_inverse_wxyz(reference_human),
            reference_robot,
        )
        _require_same_rotations(
            alignment,
            expected_alignment,
            message=("'orientation_alignment_quaternions_wxyz' must equal inverse(reference_human) * reference_robot"),
        )
    reference_qpos = _require_float_array(
        payload,
        "orientation_reference_robot_qpos",
        shape=((qpos_width,) if reference_count else (0,)),
        dtype=np.float32,
    )
    if reference_count:
        _validate_unit_quaternions(
            reference_qpos[None, 3:7],
            key="orientation_reference_robot_qpos robot root",
        )

    target = _require_float_array(
        payload,
        "orientation_target_quaternions_wxyz",
        shape=(num_frames, tracked_count, 4),
        dtype=np.float32,
    )
    robot = _require_float_array(
        payload,
        "orientation_robot_quaternions_wxyz",
        shape=(num_frames, tracked_count, 4),
        dtype=np.float32,
    )
    if tracked_count:
        _validate_unit_quaternions(
            target,
            key="orientation_target_quaternions_wxyz",
        )
        _validate_unit_quaternions(
            robot,
            key="orientation_robot_quaternions_wxyz",
        )
        source_index = {name: index for index, name in enumerate(source_orientation_names)}
        selected_source = source_orientation_quaternions[
            :,
            [source_index[name] for name in tracked_human_names],
        ]
        world_rotation_deltas = _quaternion_multiply_wxyz(
            object_poses_target[:, 3:7],
            _quaternion_inverse_wxyz(object_poses_demo[:, 3:7]),
        )
        target_world_source = _quaternion_multiply_wxyz(
            world_rotation_deltas[:, None, :],
            selected_source,
        )
        expected_target = _quaternion_multiply_wxyz(
            target_world_source,
            alignment[None, :, :],
        )
        _require_same_rotations(
            target,
            expected_target,
            message=(
                "'orientation_target_quaternions_wxyz' must equal the "
                "demo-to-target object rotation * saved direct-source "
                "orientations * saved alignment"
            ),
        )
        if alignment_mode == "first_frame":
            _require_same_rotations(
                reference_human,
                target_world_source[0],
                message=(
                    "'orientation_reference_human_quaternions_wxyz' must "
                    "equal the target-world first source frame in first_frame mode"
                ),
            )
        robot_index = {name: index for index, name in enumerate(robot_names)}
        expected_robot = robot_link_quaternions[
            :,
            [robot_index[name] for name in tracked_robot_names],
        ]
        _require_same_rotations(
            robot,
            expected_robot,
            message=(
                "'orientation_robot_quaternions_wxyz' must equal the named subset of 'robot_link_quaternions_wxyz'"
            ),
        )
    errors = _require_float_array(
        payload,
        "orientation_errors_rad",
        shape=(num_frames, tracked_count),
        dtype=np.float32,
    )
    if np.any(errors < 0.0) or np.any(errors > np.pi + 1e-5):
        _fail("'orientation_errors_rad' must contain angles in [0, pi]")
    expected_errors = _quaternion_geodesic_errors(target, robot)
    if not np.allclose(
        errors,
        expected_errors,
        atol=ORIENTATION_RELATION_ATOL_RAD,
        rtol=1e-5,
    ):
        _fail(
            "'orientation_errors_rad' must equal the sign-invariant "
            "quaternion geodesic angle between target and robot orientations"
        )
    frame_costs = _require_float_array(
        payload,
        "orientation_frame_costs",
        shape=(num_frames,),
        dtype=np.float64,
    )
    if np.any(frame_costs < 0.0):
        _fail("'orientation_frame_costs' must be non-negative")
    expected_costs = np.sum(weights[None, :] * np.square(errors), axis=1)
    if not np.allclose(
        frame_costs,
        expected_costs,
        atol=1e-7,
        rtol=1e-6,
    ):
        _fail("'orientation_frame_costs' must equal the weighted squared orientation errors for each frame")


def validate_result_artifact(payload: Mapping[str, Any]) -> None:
    """Validate a complete payload against the current result schema.

    Validation never casts, normalizes, estimates, or inserts data. In
    particular, source human orientations remain optional and are never
    fabricated. Interaction Mesh data is mandatory in the canonical schema.
    """

    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")

    _require_no_object_arrays(payload)
    missing = REQUIRED_RESULT_KEYS.difference(payload)
    if missing:
        _fail(f"result artifact is missing required fields: {', '.join(sorted(missing))}")
    reserved = _NPZ_RESERVED_KEYS.intersection(payload)
    if reserved:
        _fail(f"result artifact uses NumPy archive reserved keys: {', '.join(sorted(reserved))}")
    unsupported = set(payload).difference(ALLOWED_RESULT_KEYS)
    if unsupported:
        _fail(
            f"result artifact contains fields outside schema v{RESULT_SCHEMA_VERSION}: {', '.join(sorted(unsupported))}"
        )
    legacy_orientation_keys = _LEGACY_HUMAN_ORIENTATION_KEYS.intersection(payload)
    if legacy_orientation_keys:
        _fail(
            "versioned result artifacts must use the explicit human orientation subset "
            "fields instead of legacy fields: "
            f"{', '.join(sorted(legacy_orientation_keys))}"
        )
    schema_version = _require_integer_scalar(
        payload,
        "schema_version",
        dtype=np.int32,
    )
    if schema_version != RESULT_SCHEMA_VERSION:
        _fail(f"unsupported schema_version {schema_version}; this writer supports version {RESULT_SCHEMA_VERSION}")
    run_kind = _require_scalar_text(payload, "run_kind")
    if run_kind not in {"single", "augmentation", "ablation"}:
        _fail("'run_kind' must be 'single', 'augmentation', or 'ablation'")
    variant = _require_scalar_text(payload, "variant")
    _require_scalar_text(payload, "source_path")
    robot_type = _require_scalar_text(payload, "robot_type")
    task_type = _require_scalar_text(payload, "task_type")
    source_data_format = _require_scalar_text(
        payload,
        "source_data_format",
    )
    if source_data_format not in APPROVED_DIRECT_ORIENTATION_SOURCES:
        _fail(f"'source_data_format' must be a registered canonical format, got {source_data_format!r}")
    orientation_source = _require_scalar_text(payload, "orientation_source")
    quaternion_convention = _require_scalar_text(
        payload,
        "quaternion_convention",
    )
    if quaternion_convention != QUATERNION_CONVENTION:
        _fail(f"'quaternion_convention' must be {QUATERNION_CONVENTION!r}, got {quaternion_convention!r}")
    world_coordinate_system = _require_scalar_text(
        payload,
        "world_coordinate_system",
    )
    if world_coordinate_system != WORLD_COORDINATE_SYSTEM:
        _fail(f"'world_coordinate_system' must be {WORLD_COORDINATE_SYSTEM!r}, got {world_coordinate_system!r}")
    decoded_config = _validate_hash_metadata(payload)
    (
        object_name,
        object_urdf,
        object_urdf_sha256,
        object_asset_manifest_json,
        object_asset_manifest_sha256,
        expected_object_non_penetration_eligibility,
    ) = _validate_job_identity(
        payload,
        decoded_config=decoded_config,
        run_kind=run_kind,
        variant=variant,
        source_data_format=source_data_format,
        robot_type=robot_type,
        task_type=task_type,
    )
    _validate_preprocessing_metadata(payload)

    qpos = _require_float_array(
        payload,
        "qpos",
        ndim=2,
        nonempty_axes=(0, 1),
        dtype=np.float64,
    )
    num_frames = int(qpos.shape[0])

    human_joints = _require_float_array(
        payload,
        "human_joints",
        ndim=3,
        nonempty_axes=(1,),
        dtype=np.float32,
    )
    if human_joints.shape[0] != num_frames or human_joints.shape[2] != 3:
        _fail(f"'human_joints' must have shape ({num_frames}, joints, 3), got {human_joints.shape}")
    human_names = _require_names(
        payload,
        "human_joint_names",
        expected_count=int(human_joints.shape[1]),
    )
    _validate_parent_indices(
        payload,
        "human_joint_parent_indices",
        node_count=len(human_names),
    )

    mapped_human_joints = _require_float_array(
        payload,
        "mapped_human_joints",
        ndim=3,
        nonempty_axes=(1,),
        dtype=np.float32,
    )
    if mapped_human_joints.shape[0] != num_frames or mapped_human_joints.shape[2] != 3:
        _fail(
            f"'mapped_human_joints' must have shape ({num_frames}, mapped_joints, 3), got {mapped_human_joints.shape}"
        )
    num_mapped_joints = int(mapped_human_joints.shape[1])
    mapped_human_names = _require_names(
        payload,
        "mapped_human_joint_names",
        expected_count=num_mapped_joints,
    )
    human_index = {name: index for index, name in enumerate(human_names)}
    unknown_human_names = [name for name in mapped_human_names if name not in human_index]
    if unknown_human_names:
        _fail(
            f"'mapped_human_joint_names' must be a subset of 'human_joint_names'; unknown names: {unknown_human_names}"
        )
    selected_human_joints = human_joints[
        :,
        [human_index[name] for name in mapped_human_names],
    ]
    if not np.allclose(
        mapped_human_joints,
        selected_human_joints,
        atol=POSITION_MATCH_ATOL,
        rtol=0.0,
    ):
        _fail("'mapped_human_joints' must match the corresponding entries in 'human_joints'")

    mapped_robot_joints = _require_float_array(
        payload,
        "mapped_robot_joints",
        shape=(num_frames, num_mapped_joints, 3),
        dtype=np.float32,
    )
    mapped_robot_names = _require_names(
        payload,
        "mapped_robot_link_names",
        expected_count=num_mapped_joints,
    )

    robot_link_positions = _require_float_array(
        payload,
        "robot_link_positions",
        ndim=3,
        nonempty_axes=(1,),
        dtype=np.float32,
    )
    if robot_link_positions.shape[0] != num_frames or robot_link_positions.shape[2] != 3:
        _fail(f"'robot_link_positions' must have shape ({num_frames}, links, 3), got {robot_link_positions.shape}")
    num_robot_links = int(robot_link_positions.shape[1])
    robot_link_quaternions = _require_float_array(
        payload,
        "robot_link_quaternions_wxyz",
        shape=(num_frames, num_robot_links, 4),
        dtype=np.float32,
    )
    _validate_unit_quaternions(
        robot_link_quaternions,
        key="robot_link_quaternions_wxyz",
    )
    robot_link_names = _require_names(
        payload,
        "robot_link_names",
        expected_count=num_robot_links,
    )
    _validate_parent_indices(
        payload,
        "robot_link_parent_indices",
        node_count=num_robot_links,
    )
    robot_actuated_array = _as_array(
        payload,
        "robot_actuated_joint_names",
    )
    if robot_actuated_array.ndim != 1:
        _fail("'robot_actuated_joint_names' must be one-dimensional")
    num_actuated_joints = int(robot_actuated_array.shape[0])
    _require_names(
        payload,
        "robot_actuated_joint_names",
        expected_count=num_actuated_joints,
    )
    (
        contains_object,
        object_poses_demo,
        object_poses_target,
    ) = _validate_qpos_and_object_poses(
        payload,
        qpos=qpos,
        num_frames=num_frames,
        num_actuated_joints=num_actuated_joints,
    )
    object_non_penetration_eligible = _require_bool_scalar(
        payload,
        "object_non_penetration_eligible_for_saved_trajectory",
    )
    _validate_task_object_contract(
        task_type=task_type,
        object_name=object_name,
        object_urdf=object_urdf,
        object_urdf_sha256=object_urdf_sha256,
        object_asset_manifest_json=object_asset_manifest_json,
        object_asset_manifest_sha256=object_asset_manifest_sha256,
        contains_object=contains_object,
        object_non_penetration_eligible=object_non_penetration_eligible,
        expected_object_non_penetration_eligibility=(expected_object_non_penetration_eligibility),
    )

    robot_index = {name: index for index, name in enumerate(robot_link_names)}
    unknown_robot_names = [name for name in mapped_robot_names if name not in robot_index]
    if unknown_robot_names:
        _fail(f"'mapped_robot_link_names' must be a subset of 'robot_link_names'; unknown names: {unknown_robot_names}")
    selected_robot_positions = robot_link_positions[
        :,
        [robot_index[name] for name in mapped_robot_names],
    ]
    if not np.allclose(
        mapped_robot_joints,
        selected_robot_positions,
        atol=POSITION_MATCH_ATOL,
        rtol=0.0,
    ):
        _fail("'mapped_robot_joints' must match the corresponding entries in 'robot_link_positions'")

    fps = _require_real_scalar(
        payload,
        "fps",
        dtype=np.float64,
    )
    if fps <= 0:
        _fail(f"'fps' must be positive, got {fps}")

    (
        source_orientation_names,
        source_orientation_quaternions,
    ) = _validate_human_orientation_group(
        payload,
        num_frames=num_frames,
        human_names=human_names,
        source_data_format=source_data_format,
        orientation_source=orientation_source,
    )
    foot_modes, object_released = _validate_foot_diagnostics(
        payload,
        num_frames=num_frames,
        object_non_penetration_eligible=object_non_penetration_eligible,
    )
    _validate_nonlinear_constraint_diagnostics(
        payload,
        num_frames=num_frames,
        foot_modes=foot_modes,
        object_released=object_released,
    )
    _validate_solver_diagnostics(payload, num_frames=num_frames)
    _validate_frame_zero_ground_retry(
        payload,
        decoded_config=decoded_config,
        run_kind=run_kind,
        task_type=task_type,
        num_frames=num_frames,
    )
    _validate_orientation_diagnostics(
        payload,
        qpos_width=int(qpos.shape[1]),
        num_frames=num_frames,
        human_names=human_names,
        robot_names=robot_link_names,
        source_orientation_names=source_orientation_names,
        source_orientation_quaternions=source_orientation_quaternions,
        robot_link_quaternions=robot_link_quaternions,
        object_poses_demo=object_poses_demo,
        object_poses_target=object_poses_target,
    )
    (
        object_points_demo_world,
        object_points_target_world,
    ) = _validate_object_point_group(
        payload,
        num_frames=num_frames,
        contains_object=contains_object,
        object_poses_demo=object_poses_demo,
        object_poses_target=object_poses_target,
    )
    _validate_interaction_mesh_group(
        payload,
        num_frames=num_frames,
        mapped_human_joints=mapped_human_joints,
        mapped_robot_joints=mapped_robot_joints,
        object_points_demo_world=object_points_demo_world,
        object_points_target_world=object_points_target_world,
    )


def _stat_signature(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _reject_symlink_components(path: Path) -> None:
    if not path.is_absolute():
        raise ValueError(f"External result asset path must be absolute: {path}")
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        try:
            metadata = current.lstat()
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"External result asset path component does not exist: {current}",
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(
                f"External result asset paths must not contain symlinks: {current}",
            )


def _read_stable_regular_file(
    path: Path,
    *,
    include_contents: bool,
) -> tuple[int, str, bytes | None]:
    _reject_symlink_components(path)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"External result asset is not a regular file: {path}")
    open_flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        open_flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        open_flags |= os.O_NOFOLLOW
    file_descriptor = os.open(path, open_flags)
    digest = hashlib.sha256()
    chunks: list[bytes] | None = [] if include_contents else None
    try:
        opened = os.fstat(file_descriptor)
        if not stat.S_ISREG(opened.st_mode) or _stat_signature(opened) != _stat_signature(before):
            raise RuntimeError(
                f"External result asset changed before hashing: {path}",
            )
        with os.fdopen(file_descriptor, "rb", closefd=False) as file:
            while chunk := file.read(1024 * 1024):
                digest.update(chunk)
                if chunks is not None:
                    chunks.append(chunk)
    finally:
        os.close(file_descriptor)
    after = path.lstat()
    _reject_symlink_components(path)
    if _stat_signature(after) != _stat_signature(before):
        raise RuntimeError(f"External result asset changed while hashing: {path}")
    contents = b"".join(chunks) if chunks is not None else None
    return before.st_size, digest.hexdigest(), contents


def _canonical_external_file_path(
    path: str | os.PathLike[str],
) -> Path:
    asset_path = Path(path).expanduser()
    if not asset_path.is_absolute():
        raise ValueError(f"External result asset path must be absolute: {asset_path}")
    _reject_symlink_components(asset_path)
    resolved = asset_path.resolve(strict=True)
    if resolved != asset_path:
        raise ValueError(
            f"External result asset path must be canonical and free of symlinks or traversal components: {asset_path}",
        )
    return resolved


def _resolve_urdf_dependency(
    urdf_path: Path,
    *,
    filename: str,
    kind: str,
) -> Path:
    if not filename or filename != filename.strip() or "\x00" in filename:
        raise ValueError(
            f"URDF {kind} filename must be non-empty canonical text: {filename!r}",
        )
    if "\\" in filename or "%" in filename:
        raise ValueError(
            f"URDF {kind} filename uses an unsupported URI/path encoding: {filename!r}",
        )
    parsed = urlsplit(filename)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError(
            f"URDF {kind} filename uses an unsupported URI: {filename!r}",
        )
    reference = PurePosixPath(parsed.path)
    if str(reference) != filename or any(part in {"", ".", ".."} for part in reference.parts):
        raise ValueError(
            f"URDF {kind} filename must be a canonical local path without traversal: {filename!r}",
        )
    dependency = Path(filename) if reference.is_absolute() else urdf_path.parent.joinpath(*reference.parts)
    return _canonical_external_file_path(dependency)


def build_object_asset_manifest(
    object_urdf: str | os.PathLike[str],
) -> tuple[str, str]:
    """Freeze one URDF and every direct mesh/texture file it references.

    Canonical absolute local filenames and canonical relative filenames are
    accepted. URI schemes, traversal, missing files, non-regular files, and
    symlinks are rejected so later visualization can validate the same closed
    dependency set without implicit package or network resolution.
    """

    urdf_path = _canonical_external_file_path(object_urdf)
    urdf_size, urdf_sha256, urdf_contents = _read_stable_regular_file(
        urdf_path,
        include_contents=True,
    )
    if urdf_contents is None:
        raise AssertionError("URDF contents must be materialized")
    try:
        urdf_text = urdf_contents.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"Object URDF must use UTF-8 XML encoding: {urdf_path}",
        ) from exc
    if "<!DOCTYPE" in urdf_text.upper() or "<!ENTITY" in urdf_text.upper():
        raise ValueError(
            f"Object URDF must not contain DTD or entity declarations: {urdf_path}",
        )
    try:
        root = ET.fromstring(urdf_text)  # noqa: S314
    except ET.ParseError as exc:
        raise ValueError(f"Object URDF is not valid XML: {urdf_path}: {exc}") from exc

    dependency_identities: set[tuple[str, Path]] = set()
    for element in root.iter():
        kind = element.tag.rsplit("}", maxsplit=1)[-1]
        if kind not in {"mesh", "texture"}:
            continue
        filename = element.attrib.get("filename")
        if filename is None:
            raise ValueError(
                f"Object URDF {kind} element is missing filename: {urdf_path}",
            )
        dependency_identities.add(
            (
                kind,
                _resolve_urdf_dependency(
                    urdf_path,
                    filename=filename,
                    kind=kind,
                ),
            )
        )

    dependencies: list[dict[str, Any]] = []
    for kind, dependency_path in sorted(
        dependency_identities,
        key=lambda item: (item[0], str(item[1])),
    ):
        size, sha256, _ = _read_stable_regular_file(
            dependency_path,
            include_contents=False,
        )
        dependencies.append(
            {
                "kind": kind,
                "path": str(dependency_path),
                "sha256": sha256,
                "size": size,
            }
        )

    # Verify the complete snapshot once more after every dependency was read.
    observed_urdf_size, observed_urdf_sha256, _ = _read_stable_regular_file(
        urdf_path,
        include_contents=False,
    )
    if (observed_urdf_size, observed_urdf_sha256) != (
        urdf_size,
        urdf_sha256,
    ):
        raise RuntimeError(
            f"Object URDF changed while its dependency closure was frozen: {urdf_path}",
        )
    for dependency in dependencies:
        observed_size, observed_sha256, _ = _read_stable_regular_file(
            Path(dependency["path"]),
            include_contents=False,
        )
        if (observed_size, observed_sha256) != (
            dependency["size"],
            dependency["sha256"],
        ):
            raise RuntimeError(
                f"Object asset dependency changed while its closure was frozen: {dependency['path']}",
            )

    manifest = {
        "dependencies": dependencies,
        "urdf": {
            "path": str(urdf_path),
            "sha256": urdf_sha256,
            "size": urdf_size,
        },
        "version": _OBJECT_ASSET_MANIFEST_VERSION,
    }
    manifest_json = _canonical_json(manifest)
    manifest_sha256 = hashlib.sha256(manifest_json.encode("utf-8")).hexdigest()
    return manifest_json, manifest_sha256


def compute_object_asset_manifest(
    object_urdf: str | os.PathLike[str],
) -> tuple[str, str, int, int]:
    """Return the canonical closure plus unique-file byte and file counts."""

    manifest_json, manifest_sha256 = build_object_asset_manifest(object_urdf)
    manifest = json.loads(manifest_json)
    entries = [manifest["urdf"], *manifest["dependencies"]]
    files: dict[str, tuple[int, str]] = {}
    for entry in entries:
        identity = (int(entry["size"]), str(entry["sha256"]))
        previous = files.setdefault(str(entry["path"]), identity)
        if previous != identity:
            raise RuntimeError(
                f"One object dependency path produced conflicting file identities: {entry['path']}",
            )
    return (
        manifest_json,
        manifest_sha256,
        sum(size for size, _sha256 in files.values()),
        len(files),
    )


def compute_file_sha256(path: str | os.PathLike[str]) -> str:
    """Hash one stable regular file and reject mutation during the read."""

    asset_path = Path(path)
    if not asset_path.is_file():
        raise FileNotFoundError(f"External result asset does not exist: {asset_path}")
    before = asset_path.stat()
    before_signature = _stat_signature(before)
    digest = hashlib.sha256()
    with asset_path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    after = asset_path.stat()
    if _stat_signature(after) != before_signature:
        raise RuntimeError(f"External result asset changed while hashing: {asset_path}")
    return digest.hexdigest()


def validate_result_external_assets(
    payload: Mapping[str, Any],
) -> tuple[Path | None, str]:
    """Verify the saved URDF and its closed visualization dependency set."""

    object_urdf = _require_scalar_text(
        payload,
        "object_urdf",
        allow_empty=True,
    )
    expected_sha256 = _require_scalar_text(
        payload,
        "object_urdf_sha256",
        allow_empty=True,
    )
    expected_manifest_json = _require_scalar_text(
        payload,
        "object_asset_manifest_json",
        allow_empty=True,
    )
    expected_manifest_sha256 = _require_scalar_text(
        payload,
        "object_asset_manifest_sha256",
        allow_empty=True,
    )
    _validate_object_asset_manifest_metadata(
        object_urdf=object_urdf,
        object_urdf_sha256=expected_sha256,
        manifest_json=expected_manifest_json,
        manifest_sha256=expected_manifest_sha256,
    )
    if not object_urdf:
        return None, ""
    object_urdf_path = Path(object_urdf)
    try:
        observed_manifest_json, observed_manifest_sha256 = build_object_asset_manifest(object_urdf_path)
    except (ET.ParseError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        raise ResultArtifactValidationError(
            f"Cannot verify external object asset dependency closure {object_urdf_path}: {exc}",
        ) from exc
    if observed_manifest_json != expected_manifest_json:
        _fail(
            f"External object asset dependency closure differs from the saved manifest for {object_urdf_path}",
        )
    if observed_manifest_sha256 != expected_manifest_sha256:
        _fail(
            "External object asset manifest SHA-256 mismatch for "
            f"{object_urdf_path}: expected {expected_manifest_sha256}, "
            f"observed {observed_manifest_sha256}",
        )
    return object_urdf_path, expected_manifest_sha256


def write_result_artifact(
    path: str | os.PathLike[str],
    payload: Mapping[str, Any],
) -> Path:
    """Validate all saved data and assets, then atomically publish an NPZ."""

    validate_result_artifact(payload)
    validate_result_external_assets(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp.npz",
        dir=destination.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        temporary_file = os.fdopen(file_descriptor, "wb")
        file_descriptor = -1
        with temporary_file:
            np.savez_compressed(temporary_file, **dict(payload))
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        temporary_path.replace(destination)
    except BaseException:
        if file_descriptor >= 0:
            try:
                os.close(file_descriptor)
            except OSError:
                pass
        temporary_path.unlink(missing_ok=True)
        raise
    return destination


def _open_npz(path: str | os.PathLike[str]) -> np.lib.npyio.NpzFile:
    loaded = np.load(Path(path), allow_pickle=False)
    if not isinstance(loaded, np.lib.npyio.NpzFile):
        close = getattr(loaded, "close", None)
        if close is not None:
            close()
        raise ValueError(f"{path!s} is not an NPZ archive")
    return loaded


def _schema_version_from_npz(data: np.lib.npyio.NpzFile) -> int | None:
    if "schema_version" not in data.files:
        return None
    array = np.asarray(data["schema_version"])
    if array.ndim != 0 or array.dtype != np.dtype(np.int32):
        raise ResultArtifactValidationError("'schema_version' in a saved artifact must be an int32 scalar")
    return int(array.item())


def read_result_schema_version(path: str | os.PathLike[str]) -> int | None:
    """Return the stored schema version, or ``None`` for an unversioned NPZ."""

    with _open_npz(path) as data:
        return _schema_version_from_npz(data)


def is_legacy_result_artifact(path: str | os.PathLike[str]) -> bool:
    """Lightly identify an unversioned retargeting result without adapting it."""

    with _open_npz(path) as data:
        version = _schema_version_from_npz(data)
        return version is None and "qpos" in data.files


__all__ = [
    "COLLISION_INTERIOR_MARGIN_MAX_M",
    "COLLISION_INTERIOR_MARGIN_MIN_M",
    "FRAME_ZERO_GROUND_RETRY_POLICY",
    "HUMAN_ORIENTATION_KEYS",
    "INTERACTION_MESH_KEYS",
    "NONLINEAR_CONSTRAINT_VIOLATION_ATOL",
    "OBJECT_POINT_KEYS",
    "POSITION_MATCH_ATOL",
    "QUATERNION_NORM_ATOL",
    "REQUIRED_RESULT_KEYS",
    "RESULT_SCHEMA_VERSION",
    "ResultArtifactValidationError",
    "build_object_asset_manifest",
    "collision_interior_margin_m",
    "compute_file_sha256",
    "compute_human_orientation_sha256",
    "compute_object_asset_manifest",
    "is_legacy_result_artifact",
    "read_result_schema_version",
    "validate_result_artifact",
    "validate_result_external_assets",
    "write_result_artifact",
]
