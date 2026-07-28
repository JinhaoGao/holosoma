#!/usr/bin/env python3
"""Synchronously compare multiple retargeting result files in Viser."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import tyro
import viser  # type: ignore[import-not-found]
import yourdfpy  # type: ignore[import-untyped]
from viser.extras import ViserUrdf  # type: ignore[import-not-found]

from holosoma_retargeting.augmentation_viser_player import (
    VariantResult,
    _resolve_object_urdf,
    _resolve_robot_urdf,
    _resolve_robot_xml,
    _rgba,
    _robot_joints_for_viser,
    _slerp,
    interpolate_qpos,
    load_variant_result,
)
from holosoma_retargeting.src.viser_utils import (
    actuated_joint_names_from_mujoco_xml,
    build_joint_order_indices,
    create_motion_control_sliders,
)
from holosoma_retargeting.viser_player import _mapped_skeleton_edges


@dataclass(frozen=True)
class AblationViserConfig:
    """Configuration for synchronized comparison of arbitrary retargeting results."""

    qpos_npzs: tuple[Path, ...]
    """Result NPZ paths in display order."""

    labels: tuple[str, ...] = ()
    """Optional display labels. Defaults to each result filename."""

    robot_urdf: Path | None = None
    """Optional robot URDF override. Otherwise resolved from result metadata."""

    robot_mujoco_xml: Path | None = None
    """Optional MuJoCo XML override used to recover qpos joint order."""

    object_urdf: Path | None = None
    """Optional object URDF override for results that contain object poses."""

    fps: float | None = None
    """Playback FPS override. Otherwise uses the FPS saved in the results."""

    loop: bool = False
    """Loop playback."""

    visual_fps_multiplier: int = 2
    """Interpolation multiplier for smooth playback."""

    x_offset: float = 0.6
    """Spacing between the human reference and robot result groups."""

    robot_mesh_opacity: float = 0.22
    """Opacity of the color-coded robot meshes."""

    object_mesh_opacity: float = 0.30
    """Opacity of object meshes when object poses are present."""

    show_human_skeleton: bool = True
    """Show one shared source-human reference made from mapped keypoints."""

    human_skeleton_line_width: float = 2.0
    """Pixel width shared by the clean human and robot skeletons."""

    human_joint_point_size: float = 0.010
    """World-space size shared by human and robot keypoints."""

    grid_width: float = 8.0
    """Viewer grid width."""

    grid_height: float = 8.0
    """Viewer grid height."""

    orientation_joints: tuple[str, ...] = ()
    """Human joint names whose target and robot frames are shown. Empty means all."""

    show_target_orientation_axes: bool = True
    """Show the calibrated target link frames with short opaque RGB arrows."""

    show_robot_orientation_axes: bool = True
    """Show the actual robot link frames with opaque RGB arrows."""

    show_orientation_error_labels: bool = False
    """Show each selected joint's SO(3) geodesic error in degrees."""

    orientation_axis_length: float = 0.065
    """Length of each orientation axis in scene units."""

    orientation_axis_shaft_radius: float = 0.0025
    """Radius of each solid orientation arrow shaft."""

    orientation_axis_head_radius: float = 0.0055
    """Radius of each solid orientation arrow head."""

    orientation_axis_head_length: float = 0.011
    """Length of each solid orientation arrow head."""


@dataclass(frozen=True)
class OrientationDiagnostics:
    """Saved target/robot frame trajectories for one retargeting result."""

    human_joint_names: tuple[str, ...]
    robot_link_names: tuple[str, ...]
    weights: np.ndarray
    target_quaternions_wxyz: np.ndarray
    robot_quaternions_wxyz: np.ndarray
    errors_rad: np.ndarray


@dataclass(frozen=True)
class HumanSkeletonTrajectory:
    """Source-human trajectories required by the reference layer."""

    joint_names: tuple[str, ...]
    points: np.ndarray
    edges: tuple[tuple[int, int], ...]
    full_joint_names: tuple[str, ...]
    full_points: np.ndarray


@dataclass
class HumanReferenceOverlay:
    """Shared mapped human skeleton and target link-frame axes."""

    trajectory: HumanSkeletonTrajectory
    offset: np.ndarray
    joints_handle: object
    bones_handle: object
    diagnostics: OrientationDiagnostics | None
    joint_indices: np.ndarray
    full_point_indices: np.ndarray
    target_axes: object | None
    axis_length: float
    target_visible: bool
    enabled: bool = True

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self.joints_handle.visible = self.enabled
        self.bones_handle.visible = self.enabled
        if self.target_axes is not None:
            self.target_axes.visible = self.enabled and self.target_visible

    def update(self, frame_float: float) -> None:
        points = interpolate_human_points(self.trajectory.points, frame_float) + self.offset
        self.joints_handle.points = points.astype(np.float32)
        edge_indices = np.asarray(self.trajectory.edges, dtype=np.int32)
        self.bones_handle.points = points[edge_indices].astype(np.float32)
        if self.diagnostics is not None and self.target_axes is not None:
            full_points = interpolate_human_points(self.trajectory.full_points, frame_float)
            origins = full_points[self.full_point_indices] + self.offset
            target_quaternions = interpolate_orientation_quaternions(
                self.diagnostics.target_quaternions_wxyz[:, self.joint_indices],
                frame_float,
            )
            self.target_axes.points = orientation_axis_segments(
                origins,
                target_quaternions,
                self.axis_length,
            )


@dataclass
class RobotSkeletonOverlay:
    """Mapped robot keypoints and connecting segments for one result."""

    points: np.ndarray
    edges: tuple[tuple[int, int], ...]
    offset: np.ndarray
    joints_handle: object
    bones_handle: object
    enabled: bool = True

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self.joints_handle.visible = self.enabled
        self.bones_handle.visible = self.enabled

    def update(self, frame_float: float) -> None:
        points = interpolate_human_points(self.points, frame_float) + self.offset
        self.joints_handle.points = points.astype(np.float32)
        edge_indices = np.asarray(self.edges, dtype=np.int32)
        self.bones_handle.points = points[edge_indices].astype(np.float32)


@dataclass
class OrientationOverlay:
    """Actual robot link-frame axes and errors for one result."""

    diagnostics: OrientationDiagnostics
    joint_indices: np.ndarray
    model: mujoco.MjModel
    data: mujoco.MjData
    body_ids: np.ndarray
    offset: np.ndarray
    robot_axes: object
    error_labels: list[object]
    axis_length: float
    robot_visible: bool
    labels_visible: bool
    enabled: bool = True

    def refresh_visibility(self) -> None:
        self.robot_axes.visible = self.enabled and self.robot_visible
        for label in self.error_labels:
            label.visible = self.enabled and self.labels_visible

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self.refresh_visibility()

    def update(self, q: np.ndarray, frame_float: float) -> None:
        if q.shape[0] < self.model.nq:
            raise ValueError(f"qpos has {q.shape[0]} values but the orientation FK model requires {self.model.nq}")
        self.data.qpos[:] = q[: self.model.nq]
        mujoco.mj_forward(self.model, self.data)
        origins = np.asarray(self.data.xpos[self.body_ids]) + self.offset
        robot_quaternions = interpolate_orientation_quaternions(
            self.diagnostics.robot_quaternions_wxyz[:, self.joint_indices],
            frame_float,
        )
        self.robot_axes.points = orientation_axis_segments(
            origins,
            robot_quaternions,
            self.axis_length,
        )
        errors = interpolate_orientation_errors(
            self.diagnostics.errors_rad[:, self.joint_indices],
            frame_float,
        )
        for local_index, label in enumerate(self.error_labels):
            source_index = int(self.joint_indices[local_index])
            joint_name = self.diagnostics.human_joint_names[source_index]
            weight = float(self.diagnostics.weights[source_index])
            label.position = origins[local_index] + np.asarray([0.0, 0.0, 1.15 * self.axis_length])
            label.text = f"{joint_name}: {np.degrees(errors[local_index]):.1f}° (w={weight:g})"


@dataclass
class ComparisonScene:
    """Viser handles and display state for one comparison result."""

    label: str
    result: VariantResult
    color: tuple[int, int, int]
    offset: np.ndarray
    robot: ViserUrdf
    robot_root: object
    object_visual: ViserUrdf | None
    object_root: object | None
    robot_skeleton: RobotSkeletonOverlay
    orientation_overlay: OrientationOverlay | None
    enabled: bool = True

    def set_enabled(self, enabled: bool) -> None:
        """Toggle mesh, mapped skeleton, and actual link axes as one group."""

        self.enabled = bool(enabled)
        self.robot.show_visual = self.enabled
        if self.object_visual is not None:
            self.object_visual.show_visual = self.enabled
        self.robot_skeleton.set_enabled(self.enabled)
        if self.orientation_overlay is not None:
            self.orientation_overlay.set_enabled(self.enabled)


_ORIENTATION_KEYS = (
    "orientation_human_joint_names",
    "orientation_robot_link_names",
    "orientation_weights",
    "orientation_target_quaternions_wxyz",
    "orientation_robot_quaternions_wxyz",
    "orientation_errors_rad",
)

_AXIS_COLORS = np.asarray(
    ((255, 0, 0), (0, 255, 0), (0, 0, 255)),
    dtype=np.uint8,
)
_TARGET_AXIS_LENGTH_SCALE = 0.78
_HUMAN_SKELETON_COLOR = np.asarray((35, 38, 42), dtype=np.uint8)
_CLASSIC_GROUP_COLORS = (
    (0, 114, 178),
    (213, 94, 0),
    (0, 158, 115),
    (204, 121, 167),
    (230, 159, 0),
    (86, 180, 233),
)


def comparison_color(index: int) -> tuple[int, int, int]:
    """Return stable colorblind-safe group colors."""

    return _CLASSIC_GROUP_COLORS[index % len(_CLASSIC_GROUP_COLORS)]


def darker_color(color: tuple[int, int, int]) -> np.ndarray:
    """Produce a darker opaque color for keypoints and skeleton segments."""

    return np.asarray(
        tuple(max(0, round(channel * 0.72)) for channel in color),
        dtype=np.uint8,
    )


def pale_mesh_color(color: tuple[int, int, int]) -> tuple[int, int, int]:
    """Produce a pale low-saturation tint that does not obscure the skeleton."""

    return tuple(round(0.25 * channel + 0.75 * 235) for channel in color)


def visualization_skeleton_edges(
    joint_names: tuple[str, ...],
) -> tuple[tuple[int, int], ...]:
    """Remove redundant crossbars from the mapped solver-keypoint graph."""

    edges = _mapped_skeleton_edges(list(joint_names))
    redundant_pairs = {
        frozenset(("LeftArm", "RightArm")),
        frozenset(("LeftUpLeg", "RightUpLeg")),
        frozenset(("L_Shoulder", "R_Shoulder")),
        frozenset(("L_Hip", "R_Hip")),
    }
    return tuple(
        edge for edge in edges if frozenset((joint_names[edge[0]], joint_names[edge[1]])) not in redundant_pairs
    )


def load_human_skeleton(
    path: Path,
    expected_frames: int,
) -> HumanSkeletonTrajectory:
    """Load both mapped solver keypoints and full joints for axis origins."""

    with np.load(path, allow_pickle=False) as data:
        required = (
            "human_joints",
            "human_joint_names",
            "mapped_human_joint_names",
        )
        missing = tuple(key for key in required if key not in data)
        if missing:
            raise KeyError(f"{path} is missing human skeleton fields {missing}")
        full_points = np.asarray(data["human_joints"], dtype=np.float32)
        full_joint_names = tuple(str(name) for name in np.asarray(data["human_joint_names"]).tolist())
        joint_names = tuple(str(name) for name in np.asarray(data["mapped_human_joint_names"]).tolist())
    expected_shape = (expected_frames, len(full_joint_names), 3)
    if full_points.shape != expected_shape:
        raise ValueError(f"{path} human_joints shape {full_points.shape} != {expected_shape}")
    if not full_joint_names or len(set(full_joint_names)) != len(full_joint_names):
        raise ValueError(f"{path} human_joint_names must be non-empty and unique")
    if not joint_names or len(set(joint_names)) != len(joint_names):
        raise ValueError(f"{path} human_joint_names must be non-empty and unique")
    full_index = {name: index for index, name in enumerate(full_joint_names)}
    unavailable = tuple(name for name in joint_names if name not in full_index)
    if unavailable:
        raise ValueError(f"{path} mapped human joints are absent from full joints: {unavailable}")
    if not np.isfinite(full_points).all():
        raise ValueError(f"{path} human_joints contain non-finite values")
    points = full_points[
        :,
        [full_index[name] for name in joint_names],
    ]
    return HumanSkeletonTrajectory(
        joint_names=joint_names,
        points=points,
        edges=visualization_skeleton_edges(joint_names),
        full_joint_names=full_joint_names,
        full_points=full_points,
    )


def interpolate_human_points(
    points: np.ndarray,
    frame_float: float,
) -> np.ndarray:
    """Linearly interpolate a ``(frames, joints, 3)`` skeleton sequence."""

    values = np.asarray(points, dtype=float)
    if values.ndim != 3 or values.shape[-1] != 3 or values.shape[0] == 0:
        raise ValueError("human skeleton sequence must have shape (frames, joints, 3)")
    clipped_frame = float(np.clip(frame_float, 0.0, values.shape[0] - 1))
    frame0 = int(np.floor(clipped_frame))
    frame1 = min(frame0 + 1, values.shape[0] - 1)
    fraction = clipped_frame - frame0
    return (1.0 - fraction) * values[frame0] + fraction * values[frame1]


def _make_human_reference_overlay(
    *,
    server,
    namespace: str,
    trajectory: HumanSkeletonTrajectory,
    diagnostics: OrientationDiagnostics | None,
    joint_indices: np.ndarray,
    offset: np.ndarray,
    config: AblationViserConfig,
) -> HumanReferenceOverlay:
    points = trajectory.points[0] + offset
    point_colors = np.tile(
        _HUMAN_SKELETON_COLOR,
        (len(trajectory.joint_names), 1),
    )
    joints_handle = server.scene.add_point_cloud(
        f"{namespace}/human/joints",
        points=points,
        colors=point_colors,
        point_size=config.human_joint_point_size,
        point_shape="circle",
        visible=config.show_human_skeleton,
    )
    bones_handle = server.scene.add_line_segments(
        f"{namespace}/human/skeleton",
        points=points[np.asarray(trajectory.edges, dtype=np.int32)],
        colors=_HUMAN_SKELETON_COLOR,
        line_width=config.human_skeleton_line_width,
        visible=config.show_human_skeleton,
    )
    full_point_indices = np.empty((0,), dtype=np.int32)
    target_axes = None
    if diagnostics is not None:
        full_index = {name: index for index, name in enumerate(trajectory.full_joint_names)}
        selected_names = tuple(diagnostics.human_joint_names[int(index)] for index in joint_indices)
        unavailable = tuple(name for name in selected_names if name not in full_index)
        if unavailable:
            raise ValueError(f"Target orientation joints are absent from the source human trajectory: {unavailable}")
        full_point_indices = np.asarray(
            [full_index[name] for name in selected_names],
            dtype=np.int32,
        )
        origins = trajectory.full_points[0, full_point_indices] + offset
        target_axes = server.scene.add_arrows(
            f"{namespace}/orientation/target_axes",
            points=orientation_axis_segments(
                origins,
                diagnostics.target_quaternions_wxyz[0, joint_indices],
                config.orientation_axis_length * _TARGET_AXIS_LENGTH_SCALE,
            ),
            colors=np.tile(_AXIS_COLORS, (len(joint_indices), 1)),
            shaft_radius=config.orientation_axis_shaft_radius,
            head_radius=config.orientation_axis_head_radius,
            head_length=config.orientation_axis_head_length,
            visible=(config.show_human_skeleton and config.show_target_orientation_axes),
        )
    return HumanReferenceOverlay(
        trajectory=trajectory,
        offset=offset,
        joints_handle=joints_handle,
        bones_handle=bones_handle,
        diagnostics=diagnostics,
        joint_indices=joint_indices,
        full_point_indices=full_point_indices,
        target_axes=target_axes,
        axis_length=config.orientation_axis_length * _TARGET_AXIS_LENGTH_SCALE,
        target_visible=config.show_target_orientation_axes,
        enabled=config.show_human_skeleton,
    )


def _make_robot_skeleton_overlay(
    *,
    server,
    namespace: str,
    result: VariantResult,
    color: tuple[int, int, int],
    offset: np.ndarray,
    config: AblationViserConfig,
) -> RobotSkeletonOverlay:
    edges = visualization_skeleton_edges(result.mapped_joint_names)
    points = result.robot_points[0] + offset
    skeleton_color = darker_color(color)
    joints_handle = server.scene.add_point_cloud(
        f"{namespace}/skeleton/joints",
        points=points,
        colors=np.tile(skeleton_color, (len(result.mapped_joint_names), 1)),
        point_size=config.human_joint_point_size,
        point_shape="circle",
    )
    bones_handle = server.scene.add_line_segments(
        f"{namespace}/skeleton/links",
        points=points[np.asarray(edges, dtype=np.int32)],
        colors=skeleton_color,
        line_width=config.human_skeleton_line_width,
    )
    return RobotSkeletonOverlay(
        points=result.robot_points,
        edges=edges,
        offset=offset,
        joints_handle=joints_handle,
        bones_handle=bones_handle,
    )


def load_orientation_diagnostics(
    path: Path,
    expected_frames: int,
) -> OrientationDiagnostics | None:
    """Load optional orientation arrays while preserving legacy NPZ support."""

    with np.load(path, allow_pickle=False) as data:
        present = tuple(key for key in _ORIENTATION_KEYS if key in data.files)
        if not present:
            return None
        missing = tuple(key for key in _ORIENTATION_KEYS if key not in data.files)
        if missing:
            raise ValueError(f"{path} has incomplete orientation diagnostics; missing {missing}")
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
    if not human_joint_names or len(set(human_joint_names)) != joint_count:
        raise ValueError(f"{path} orientation human joint names must be non-empty and unique")
    if len(robot_link_names) != joint_count:
        raise ValueError(f"{path} has {joint_count} human orientation joints but {len(robot_link_names)} robot links")
    expected_quaternion_shape = (expected_frames, joint_count, 4)
    expected_error_shape = (expected_frames, joint_count)
    if weights.shape != (joint_count,):
        raise ValueError(f"{path} orientation_weights shape {weights.shape} != {(joint_count,)}")
    if target_quaternions.shape != expected_quaternion_shape:
        raise ValueError(f"{path} target quaternion shape {target_quaternions.shape} != {expected_quaternion_shape}")
    if robot_quaternions.shape != expected_quaternion_shape:
        raise ValueError(f"{path} robot quaternion shape {robot_quaternions.shape} != {expected_quaternion_shape}")
    if errors.shape != expected_error_shape:
        raise ValueError(f"{path} orientation error shape {errors.shape} != {expected_error_shape}")
    arrays = (weights, target_quaternions, robot_quaternions, errors)
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError(f"{path} orientation diagnostics contain non-finite values")
    if np.any(weights < 0.0) or np.any(errors < 0.0):
        raise ValueError(f"{path} orientation weights and errors must be non-negative")
    target_norms = np.linalg.norm(target_quaternions, axis=-1, keepdims=True)
    robot_norms = np.linalg.norm(robot_quaternions, axis=-1, keepdims=True)
    if np.any(target_norms < 1e-8) or np.any(robot_norms < 1e-8):
        raise ValueError(f"{path} orientation diagnostics contain zero quaternions")
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


def interpolate_orientation_quaternions(
    quaternions_wxyz: np.ndarray,
    frame_float: float,
) -> np.ndarray:
    """Interpolate a ``(frames, joints, 4)`` quaternion sequence with SLERP."""

    values = np.asarray(quaternions_wxyz, dtype=float)
    if values.ndim != 3 or values.shape[-1] != 4 or values.shape[0] == 0:
        raise ValueError("orientation quaternion sequence must have shape (frames, joints, 4)")
    clipped_frame = float(np.clip(frame_float, 0.0, values.shape[0] - 1))
    frame0 = int(np.floor(clipped_frame))
    frame1 = min(frame0 + 1, values.shape[0] - 1)
    fraction = clipped_frame - frame0
    if frame0 == frame1:
        return values[frame0].copy()
    return np.stack(
        [_slerp(values[frame0, index], values[frame1, index], fraction) for index in range(values.shape[1])],
        axis=0,
    )


def interpolate_orientation_errors(
    errors_rad: np.ndarray,
    frame_float: float,
) -> np.ndarray:
    """Linearly interpolate a ``(frames, joints)`` error sequence."""

    values = np.asarray(errors_rad, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("orientation error sequence must have shape (frames, joints)")
    clipped_frame = float(np.clip(frame_float, 0.0, values.shape[0] - 1))
    frame0 = int(np.floor(clipped_frame))
    frame1 = min(frame0 + 1, values.shape[0] - 1)
    fraction = clipped_frame - frame0
    return (1.0 - fraction) * values[frame0] + fraction * values[frame1]


def _quaternion_matrices_wxyz(quaternions_wxyz: np.ndarray) -> np.ndarray:
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
    """Convert joint origins and frame rotations to RGB arrow endpoints."""

    origins_array = np.asarray(origins, dtype=float)
    if origins_array.ndim != 2 or origins_array.shape[1] != 3 or origins_array.shape[0] != len(quaternions_wxyz):
        raise ValueError("origins and quaternions must have matching joint counts")
    if not np.isfinite(axis_length) or axis_length <= 0.0:
        raise ValueError("orientation axis length must be finite and positive")
    matrices = _quaternion_matrices_wxyz(quaternions_wxyz)
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


def _comparison_labels(config: AblationViserConfig) -> tuple[str, ...]:
    if len(config.qpos_npzs) < 2:
        raise ValueError("At least two result NPZ paths are required for comparison.")
    if config.labels and len(config.labels) != len(config.qpos_npzs):
        raise ValueError(f"labels has {len(config.labels)} entries for {len(config.qpos_npzs)} result paths")
    if len(set(config.qpos_npzs)) != len(config.qpos_npzs):
        raise ValueError("Comparison result paths must be unique.")
    return config.labels or tuple(path.stem for path in config.qpos_npzs)


def load_comparison_results(
    config: AblationViserConfig,
) -> tuple[tuple[str, ...], list[VariantResult]]:
    """Load arbitrary result paths and verify that their timelines are synchronized."""

    labels = _comparison_labels(config)
    results = [load_variant_result(label, path) for label, path in zip(labels, config.qpos_npzs, strict=True)]
    reference = results[0]
    incompatibilities: list[str] = []
    for result in results[1:]:
        if result.qpos.shape != reference.qpos.shape:
            incompatibilities.append(f"{result.path}: qpos shape {result.qpos.shape} != {reference.qpos.shape}")
        if not np.isclose(result.fps, reference.fps):
            incompatibilities.append(f"{result.path}: fps={result.fps} != {reference.fps}")
        if result.robot_type != reference.robot_type:
            incompatibilities.append(f"{result.path}: robot_type={result.robot_type!r} != {reference.robot_type!r}")
        if result.contains_object_in_qpos != reference.contains_object_in_qpos:
            incompatibilities.append(f"{result.path}: object qpos layout differs")
        if result.contains_object_in_qpos and result.object_name != reference.object_name:
            incompatibilities.append(f"{result.path}: object_name={result.object_name!r} != {reference.object_name!r}")
    if incompatibilities:
        formatted = "\n".join(f"  {message}" for message in incompatibilities)
        raise ValueError(f"Comparison results are not synchronized:\n{formatted}")
    return labels, results


def _make_orientation_overlay(
    *,
    server,
    namespace: str,
    diagnostics: OrientationDiagnostics,
    joint_indices: np.ndarray,
    model: mujoco.MjModel,
    initial_q: np.ndarray,
    offset: np.ndarray,
    config: AblationViserConfig,
) -> OrientationOverlay:
    data = mujoco.MjData(model)
    selected_link_names = tuple(diagnostics.robot_link_names[int(index)] for index in joint_indices)
    body_ids = np.asarray(
        [
            mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                link_name,
            )
            for link_name in selected_link_names
        ],
        dtype=np.int32,
    )
    missing_links = tuple(
        link_name
        for link_name, body_id in zip(
            selected_link_names,
            body_ids,
            strict=True,
        )
        if body_id < 0
    )
    if missing_links:
        raise ValueError(f"Orientation robot links are absent from the MuJoCo model: {missing_links}")
    if initial_q.shape[0] < model.nq:
        raise ValueError(f"qpos has {initial_q.shape[0]} values but the orientation FK model requires {model.nq}")
    data.qpos[:] = initial_q[: model.nq]
    mujoco.mj_forward(model, data)
    origins = np.asarray(data.xpos[body_ids]) + offset
    robot_quaternions = diagnostics.robot_quaternions_wxyz[
        0,
        joint_indices,
    ]
    robot_axes = server.scene.add_arrows(
        f"{namespace}/orientation/robot_axes",
        points=orientation_axis_segments(
            origins,
            robot_quaternions,
            config.orientation_axis_length,
        ),
        colors=np.tile(_AXIS_COLORS, (len(joint_indices), 1)),
        shaft_radius=config.orientation_axis_shaft_radius,
        head_radius=config.orientation_axis_head_radius,
        head_length=config.orientation_axis_head_length,
        visible=config.show_robot_orientation_axes,
    )
    error_labels = []
    for local_index, source_index_value in enumerate(joint_indices):
        source_index = int(source_index_value)
        joint_name = diagnostics.human_joint_names[source_index]
        error_degrees = np.degrees(diagnostics.errors_rad[0, source_index])
        weight = diagnostics.weights[source_index]
        error_labels.append(
            server.scene.add_label(
                (f"{namespace}/orientation/errors/{local_index:02d}_{joint_name}"),
                f"{joint_name}: {error_degrees:.1f}° (w={weight:g})",
                position=origins[local_index] + np.asarray([0.0, 0.0, 1.15 * config.orientation_axis_length]),
                visible=config.show_orientation_error_labels,
                font_screen_scale=0.55,
                depth_test=True,
                anchor="bottom-center",
            )
        )
    return OrientationOverlay(
        diagnostics=diagnostics,
        joint_indices=joint_indices,
        model=model,
        data=data,
        body_ids=body_ids,
        offset=offset,
        robot_axes=robot_axes,
        error_labels=error_labels,
        axis_length=config.orientation_axis_length,
        robot_visible=config.show_robot_orientation_axes,
        labels_visible=config.show_orientation_error_labels,
    )


def make_ablation_player(
    config: AblationViserConfig,
    labels: tuple[str, ...],
    results: list[VariantResult],
):
    """Build one shared-timeline Viser scene for all comparison results."""

    reference = results[0]
    robot_urdf = _resolve_robot_urdf(config, reference)
    robot_xml = _resolve_robot_xml(config, robot_urdf)
    contains_object = reference.contains_object_in_qpos
    object_urdf = _resolve_object_urdf(config, reference) if contains_object else None
    if not np.isfinite(config.orientation_axis_length) or config.orientation_axis_length <= 0.0:
        raise ValueError("orientation_axis_length must be finite and positive")
    arrow_dimensions = {
        "orientation_axis_shaft_radius": config.orientation_axis_shaft_radius,
        "orientation_axis_head_radius": config.orientation_axis_head_radius,
        "orientation_axis_head_length": config.orientation_axis_head_length,
    }
    invalid_arrow_dimensions = {
        name: value for name, value in arrow_dimensions.items() if not np.isfinite(value) or value <= 0.0
    }
    if invalid_arrow_dimensions:
        raise ValueError(f"Orientation arrow dimensions must be finite and positive: {invalid_arrow_dimensions}")
    if config.orientation_axis_head_length >= (config.orientation_axis_length * _TARGET_AXIS_LENGTH_SCALE):
        raise ValueError("orientation_axis_head_length must be shorter than the target arrows")
    orientation_diagnostics = [
        load_orientation_diagnostics(
            result.path,
            expected_frames=result.qpos.shape[0],
        )
        for result in results
    ]
    human_skeletons = [
        load_human_skeleton(
            result.path,
            expected_frames=result.qpos.shape[0],
        )
        for result in results
    ]
    reference_skeleton = human_skeletons[0]
    for result, skeleton in zip(
        results[1:],
        human_skeletons[1:],
        strict=True,
    ):
        if skeleton.joint_names != reference_skeleton.joint_names:
            raise ValueError(f"{result.path} human skeleton joint names differ from the first comparison result")
        if not np.allclose(
            skeleton.points,
            reference_skeleton.points,
            atol=1e-6,
        ):
            raise ValueError(f"{result.path} source human keypoints differ from the first comparison result")
    robot_fk_model = None
    if any(item is not None for item in orientation_diagnostics):
        if robot_xml is None:
            raise FileNotFoundError(
                "A MuJoCo robot XML is required to position orientation arrows; pass --robot-mujoco-xml."
            )
        robot_fk_model = mujoco.MjModel.from_xml_path(str(robot_xml))

    server = viser.ViserServer()
    server.scene.add_grid(
        "/grid",
        width=config.grid_width,
        height=config.grid_height,
        position=(0.0, 0.0, 0.0),
    )

    robot_urdf_model = yourdfpy.URDF.load(
        str(robot_urdf),
        load_meshes=True,
        build_scene_graph=True,
    )
    object_urdf_model = (
        yourdfpy.URDF.load(
            str(object_urdf),
            load_meshes=True,
            build_scene_graph=True,
        )
        if object_urdf is not None
        else None
    )

    scenes: list[ComparisonScene] = []
    robot_dof: int | None = None
    joint_order_indices: np.ndarray | None = None
    centered_indices = np.arange(len(results) + 1, dtype=float) - 0.5 * len(results)
    human_offset = np.asarray([centered_indices[0] * config.x_offset, 0.0, 0.0])
    reference_diagnostics = next(
        (diagnostics for diagnostics in orientation_diagnostics if diagnostics is not None),
        None,
    )
    reference_joint_indices = (
        orientation_joint_indices(
            reference_diagnostics,
            config.orientation_joints,
        )
        if reference_diagnostics is not None
        else np.empty((0,), dtype=np.int32)
    )
    human_reference = _make_human_reference_overlay(
        server=server,
        namespace="/reference_human",
        trajectory=reference_skeleton,
        diagnostics=reference_diagnostics,
        joint_indices=reference_joint_indices,
        offset=human_offset,
        config=config,
    )

    for index, (label, result, diagnostics) in enumerate(
        zip(
            labels,
            results,
            orientation_diagnostics,
            strict=True,
        )
    ):
        color = comparison_color(index)
        namespace = f"/groups/{index:02d}_{label}"
        offset = np.asarray([centered_indices[index + 1] * config.x_offset, 0.0, 0.0])
        robot_skeleton = _make_robot_skeleton_overlay(
            server=server,
            namespace=namespace,
            result=result,
            color=color,
            offset=offset,
            config=config,
        )
        robot_root = server.scene.add_frame(f"{namespace}/robot", show_axes=False)
        robot = ViserUrdf(
            server,
            urdf_or_path=robot_urdf_model,
            root_node_name=f"{namespace}/robot",
            mesh_color_override=_rgba(
                pale_mesh_color(color),
                config.robot_mesh_opacity,
            ),
        )

        object_root = None
        object_visual = None
        if object_urdf_model is not None:
            object_root = server.scene.add_frame(f"{namespace}/object", show_axes=False)
            object_visual = ViserUrdf(
                server,
                urdf_or_path=object_urdf_model,
                root_node_name=f"{namespace}/object",
                mesh_color_override=_rgba(color, config.object_mesh_opacity),
            )

        current_robot_dof = len(robot.get_actuated_joint_limits())
        if robot_dof is None:
            robot_dof = current_robot_dof
            if robot_xml is not None:
                qpos_joint_names = actuated_joint_names_from_mujoco_xml(robot_xml)
                viser_joint_names = list(robot.get_actuated_joint_limits().keys())
                if qpos_joint_names != viser_joint_names:
                    joint_order_indices = build_joint_order_indices(
                        qpos_joint_names,
                        viser_joint_names,
                    )
        elif current_robot_dof != robot_dof:
            raise ValueError("Loaded comparison robots expose inconsistent actuated joint counts.")

        orientation_overlay = None
        if diagnostics is not None:
            if robot_fk_model is None:
                raise RuntimeError("Orientation FK model was not initialized.")
            joint_indices = orientation_joint_indices(
                diagnostics,
                config.orientation_joints,
            )
            orientation_overlay = _make_orientation_overlay(
                server=server,
                namespace=namespace,
                diagnostics=diagnostics,
                joint_indices=joint_indices,
                model=robot_fk_model,
                initial_q=result.qpos[0],
                offset=offset,
                config=config,
            )

        scenes.append(
            ComparisonScene(
                label=label,
                result=result,
                color=color,
                offset=offset,
                robot=robot,
                robot_root=robot_root,
                object_visual=object_visual,
                object_root=object_root,
                robot_skeleton=robot_skeleton,
                orientation_overlay=orientation_overlay,
            )
        )

    if robot_dof is None:
        raise RuntimeError("No comparison results were loaded.")

    def _render_scene(
        scene: ComparisonScene,
        q: np.ndarray,
        frame_float: float,
    ) -> None:
        scene.robot_skeleton.update(frame_float)
        scene.robot.update_cfg(
            _robot_joints_for_viser(
                q,
                robot_dof,
                joint_order_indices,
            )
        )
        scene.robot_root.position = q[0:3] + scene.offset
        scene.robot_root.wxyz = q[3:7]
        if contains_object and scene.object_root is not None:
            scene.object_root.position = q[-7:-4] + scene.offset
            scene.object_root.wxyz = q[-4:]
        if scene.orientation_overlay is not None:
            scene.orientation_overlay.update(q, frame_float)

    def _render_comparison(driver_q: np.ndarray, frame_float: float) -> None:
        human_reference.update(frame_float)
        for index, scene in enumerate(scenes):
            q = (
                driver_q
                if index == 0
                else interpolate_qpos(
                    scene.result.qpos,
                    frame_float,
                    robot_dof,
                    contains_object=contains_object,
                )
            )
            _render_scene(scene, q, frame_float)

    with server.gui.add_folder("Display layers"):
        human_checkbox = server.gui.add_checkbox(
            "Original human: mapped skeleton + target link axes",
            initial_value=config.show_human_skeleton,
        )

        @human_checkbox.on_update
        def _(_event) -> None:
            human_reference.set_enabled(bool(human_checkbox.value))

        for scene in scenes:
            color_hex = "#" + "".join(f"{channel:02x}" for channel in scene.color)
            group_checkbox = server.gui.add_checkbox(
                (f"{scene.label}: robot mesh + mapped skeleton + actual link axes ({color_hex})"),
                initial_value=True,
            )

            def _register_group_callback(
                scene_ref: ComparisonScene,
                checkbox_ref,
            ) -> None:
                @checkbox_ref.on_update
                def _(_event) -> None:
                    scene_ref.set_enabled(bool(checkbox_ref.value))

            _register_group_callback(scene, group_checkbox)

    driver = scenes[0]
    create_motion_control_sliders(
        server=server,
        viser_robot=driver.robot,
        robot_base_frame=driver.robot_root,
        motion_sequence=reference.qpos,
        robot_dof=robot_dof,
        viser_object=driver.object_visual,
        object_base_frame=driver.object_root,
        contains_object_in_qpos=contains_object,
        initial_fps=round(config.fps or reference.fps),
        initial_interp_mult=config.visual_fps_multiplier,
        loop=config.loop,
        qpos_to_viser_joint_indices=joint_order_indices,
        on_frame=_render_comparison,
    )

    print(
        f"[ablation_viser_player] Loaded {len(results)} synchronized results, "
        f"frames={reference.qpos.shape[0]}, fps={config.fps or reference.fps:.3f}"
    )
    for scene in scenes:
        print(
            f"  robot group {scene.label}: translucent mesh + mapped skeleton "
            f"+ actual link axes, rgb={scene.color}, path={scene.result.path}"
        )
        if scene.orientation_overlay is not None:
            selected_names = tuple(
                scene.orientation_overlay.diagnostics.human_joint_names[int(index)]
                for index in scene.orientation_overlay.joint_indices
            )
            print(f"    orientation arrows: joints={selected_names}, robot=opaque RGB arrows")
        else:
            print("    orientation arrows: unavailable in this legacy result")
    print("  human reference: charcoal mapped-keypoint skeleton + target RGB link axes")
    print("Open the viewer URL printed above. Close the process (Ctrl+C) to exit.")
    return server


def main(config: AblationViserConfig) -> None:
    """Load the requested results and keep their synchronized player alive."""

    labels, results = load_comparison_results(config)
    make_ablation_player(config, labels, results)
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main(tyro.cli(AblationViserConfig))
