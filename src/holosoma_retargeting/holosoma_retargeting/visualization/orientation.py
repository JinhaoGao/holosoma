# ruff: noqa: CPY001
"""Shared loading and interpolation for saved orientation diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class OrientationDiagnostics:
    """Saved target and robot frame trajectories for one result."""

    human_joint_names: tuple[str, ...]
    robot_link_names: tuple[str, ...]
    weights: np.ndarray
    target_quaternions_wxyz: np.ndarray
    robot_quaternions_wxyz: np.ndarray
    errors_rad: np.ndarray


ORIENTATION_KEYS = (
    "orientation_human_joint_names",
    "orientation_robot_link_names",
    "orientation_weights",
    "orientation_target_quaternions_wxyz",
    "orientation_robot_quaternions_wxyz",
    "orientation_errors_rad",
)


def load_orientation_diagnostics(
    path: str | Path,
    expected_frames: int,
) -> OrientationDiagnostics | None:
    """Load optional orientation arrays while preserving legacy NPZ support."""

    result_path = Path(path)
    with np.load(result_path, allow_pickle=False) as data:
        present = tuple(key for key in ORIENTATION_KEYS if key in data.files)
        if not present:
            return None
        missing = tuple(key for key in ORIENTATION_KEYS if key not in data.files)
        if missing:
            raise ValueError(f"{result_path} has incomplete orientation diagnostics; missing {missing}")
        human_joint_names = tuple(str(name) for name in np.asarray(data["orientation_human_joint_names"]).tolist())
        robot_link_names = tuple(str(name) for name in np.asarray(data["orientation_robot_link_names"]).tolist())
        weights = np.asarray(data["orientation_weights"], dtype=float)
        target_quaternions = np.asarray(
            data["orientation_target_quaternions_wxyz"],
            dtype=float,
        )
        robot_quaternions = np.asarray(
            data["orientation_robot_quaternions_wxyz"],
            dtype=float,
        )
        errors = np.asarray(data["orientation_errors_rad"], dtype=float)

    joint_count = len(human_joint_names)
    if joint_count == 0:
        empty_shapes = (
            robot_link_names == ()
            and weights.shape == (0,)
            and target_quaternions.shape == (expected_frames, 0, 4)
            and robot_quaternions.shape == (expected_frames, 0, 4)
            and errors.shape == (expected_frames, 0)
        )
        if empty_shapes:
            return None
        raise ValueError(f"{result_path} has inconsistent empty orientation diagnostics")
    if not human_joint_names or len(set(human_joint_names)) != joint_count:
        raise ValueError(f"{result_path} orientation human joint names must be non-empty and unique")
    if len(robot_link_names) != joint_count:
        raise ValueError(
            f"{result_path} has {joint_count} human orientation joints but {len(robot_link_names)} robot links"
        )
    expected_quaternion_shape = (expected_frames, joint_count, 4)
    expected_error_shape = (expected_frames, joint_count)
    if weights.shape != (joint_count,):
        raise ValueError(f"{result_path} orientation_weights shape {weights.shape} != {(joint_count,)}")
    if target_quaternions.shape != expected_quaternion_shape:
        raise ValueError(
            f"{result_path} target quaternion shape {target_quaternions.shape} != {expected_quaternion_shape}"
        )
    if robot_quaternions.shape != expected_quaternion_shape:
        raise ValueError(
            f"{result_path} robot quaternion shape {robot_quaternions.shape} != {expected_quaternion_shape}"
        )
    if errors.shape != expected_error_shape:
        raise ValueError(f"{result_path} orientation error shape {errors.shape} != {expected_error_shape}")
    arrays = (weights, target_quaternions, robot_quaternions, errors)
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError(f"{result_path} orientation diagnostics contain non-finite values")
    if np.any(weights < 0.0) or np.any(errors < 0.0):
        raise ValueError(f"{result_path} orientation weights and errors must be non-negative")
    target_norms = np.linalg.norm(target_quaternions, axis=-1, keepdims=True)
    robot_norms = np.linalg.norm(robot_quaternions, axis=-1, keepdims=True)
    if np.any(target_norms < 1e-8) or np.any(robot_norms < 1e-8):
        raise ValueError(f"{result_path} orientation diagnostics contain zero quaternions")
    return OrientationDiagnostics(
        human_joint_names=human_joint_names,
        robot_link_names=robot_link_names,
        weights=weights,
        target_quaternions_wxyz=target_quaternions / target_norms,
        robot_quaternions_wxyz=robot_quaternions / robot_norms,
        errors_rad=errors,
    )


def orientation_joint_indices(
    diagnostics: OrientationDiagnostics,
    requested_joint_names: tuple[str, ...],
) -> np.ndarray:
    """Resolve a stable diagnostic subset from human joint names."""

    if len(set(requested_joint_names)) != len(requested_joint_names):
        raise ValueError("orientation_joints must not contain duplicate names")
    if not requested_joint_names:
        return np.arange(len(diagnostics.human_joint_names), dtype=np.int32)
    name_to_index = {name: index for index, name in enumerate(diagnostics.human_joint_names)}
    missing = tuple(name for name in requested_joint_names if name not in name_to_index)
    if missing:
        raise ValueError(f"Unknown orientation joints {missing}; available: {diagnostics.human_joint_names}")
    return np.asarray(
        [name_to_index[name] for name in requested_joint_names],
        dtype=np.int32,
    )


def orientation_skeleton_point_indices(
    diagnostics: OrientationDiagnostics,
    mapped_joint_names: tuple[str, ...],
    mapped_robot_link_names: tuple[str, ...],
) -> np.ndarray:
    """Match every diagnostic frame to its displayed mapped keypoint."""

    if len(mapped_robot_link_names) != len(mapped_joint_names):
        raise ValueError("Mapped human joints and robot links must have matching counts")
    human_index = {name: index for index, name in enumerate(mapped_joint_names)}
    robot_index = {name: index for index, name in enumerate(mapped_robot_link_names) if name}
    aliases = {
        "Hips": ("Spine1", "Pelvis"),
        "Pelvis": ("Hips", "Spine1"),
        "Spine1": ("Hips", "Pelvis"),
    }
    indices: list[int] = []
    unresolved: list[tuple[str, str]] = []
    for human_name, robot_link_name in zip(
        diagnostics.human_joint_names,
        diagnostics.robot_link_names,
        strict=True,
    ):
        index = human_index.get(human_name)
        if index is None:
            index = robot_index.get(robot_link_name)
        if index is None:
            index = next(
                (human_index[alias] for alias in aliases.get(human_name, ()) if alias in human_index),
                None,
            )
        if index is None:
            unresolved.append((human_name, robot_link_name))
        else:
            indices.append(index)
    if unresolved:
        raise ValueError(f"Orientation frames cannot be matched to displayed skeleton keypoints: {unresolved}")
    return np.asarray(indices, dtype=np.int32)


def _slerp(q0: np.ndarray, q1: np.ndarray, fraction: float) -> np.ndarray:
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    q0 /= np.linalg.norm(q0)
    q1 /= np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        result = q0 + fraction * (q1 - q0)
        return result / np.linalg.norm(result)
    theta = float(np.arccos(np.clip(dot, -1.0, 1.0)))
    sin_theta = float(np.sin(theta))
    return (np.sin((1.0 - fraction) * theta) * q0 + np.sin(fraction * theta) * q1) / sin_theta


def interpolate_orientation_quaternions(
    quaternions_wxyz: np.ndarray,
    frame_float: float,
    *,
    loop: bool = False,
) -> np.ndarray:
    """Interpolate a ``(frames, joints, 4)`` sequence with quaternion SLERP."""

    values = np.asarray(quaternions_wxyz, dtype=float)
    if values.ndim != 3 or values.shape[-1] != 4 or values.shape[0] == 0:
        raise ValueError("orientation quaternion sequence must have shape (frames, joints, 4)")
    if loop:
        sample_frame = float(frame_float) % values.shape[0]
        frame0 = int(np.floor(sample_frame))
        frame1 = (frame0 + 1) % values.shape[0]
    else:
        sample_frame = float(np.clip(frame_float, 0.0, values.shape[0] - 1))
        frame0 = int(np.floor(sample_frame))
        frame1 = min(frame0 + 1, values.shape[0] - 1)
    fraction = sample_frame - frame0
    if frame0 == frame1:
        return values[frame0].copy()
    return np.stack(
        [_slerp(values[frame0, index], values[frame1, index], fraction) for index in range(values.shape[1])],
        axis=0,
    )


def quaternion_matrices_wxyz(quaternions_wxyz: np.ndarray) -> np.ndarray:
    """Convert normalized or unnormalized wxyz quaternions to rotation matrices."""

    quaternions = np.asarray(quaternions_wxyz, dtype=float)
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise ValueError("quaternions must have shape (joints, 4)")
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("quaternions must be non-zero")
    w, x, y, z = (quaternions / norms).T
    matrices = np.empty((quaternions.shape[0], 3, 3), dtype=float)
    matrices[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrices[:, 0, 1] = 2.0 * (x * y - w * z)
    matrices[:, 0, 2] = 2.0 * (x * z + w * y)
    matrices[:, 1, 0] = 2.0 * (x * y + w * z)
    matrices[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrices[:, 1, 2] = 2.0 * (y * z - w * x)
    matrices[:, 2, 0] = 2.0 * (x * z - w * y)
    matrices[:, 2, 1] = 2.0 * (y * z + w * x)
    matrices[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return matrices


def orientation_axis_segments(
    origins: np.ndarray,
    quaternions_wxyz: np.ndarray,
    axis_length: float,
) -> np.ndarray:
    """Convert joint origins and frame rotations to RGB arrow segments."""

    origins_array = np.asarray(origins, dtype=float)
    if origins_array.ndim != 2 or origins_array.shape[1] != 3 or origins_array.shape[0] != len(quaternions_wxyz):
        raise ValueError("origins and quaternions must have matching joint counts")
    if not np.isfinite(axis_length) or axis_length <= 0.0:
        raise ValueError("orientation axis length must be finite and positive")
    matrices = quaternion_matrices_wxyz(quaternions_wxyz)
    world_axes = np.transpose(matrices, (0, 2, 1))
    endpoints = origins_array[:, None, :] + axis_length * world_axes
    segments = np.stack(
        (
            np.repeat(origins_array[:, None, :], 3, axis=1),
            endpoints,
        ),
        axis=2,
    )
    return segments.reshape(-1, 2, 3).astype(np.float32)
