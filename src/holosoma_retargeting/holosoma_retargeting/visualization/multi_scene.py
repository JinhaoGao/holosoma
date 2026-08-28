# ruff: noqa: CPY001
"""Shared multi-motion loading and synchronized Viser scene."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import mujoco
import numpy as np
import viser  # type: ignore[import-not-found]
import yourdfpy  # type: ignore[import-untyped]
from viser.extras import ViserUrdf  # type: ignore[import-not-found]

from holosoma_retargeting.data_utils.hand_skeleton import build_hand_visualization_spec
from holosoma_retargeting.src.viser_utils import (
    create_motion_control_sliders,
    format_foot_sticking_status,
)
from holosoma_retargeting.viser_player import (
    InteractionMeshOverlay,
    ObjectKeypointOverlay,
    RetargetingPointCloudOverlay,
    SavedOrientationAxesOverlay,
    _mapped_skeleton_edges,
    _requested_saved_orientation_indices,
    _saved_foot_sticking_constraint_status,
)
from holosoma_retargeting.visualization.layers import (
    LayerController,
    LayerId,
    add_visualization_tabs,
)
from holosoma_retargeting.visualization.orientation import (
    OrientationDiagnostics,
    interpolate_orientation_quaternions,
    orientation_axis_segments,
    orientation_joint_indices,
    orientation_skeleton_point_indices,
)
from holosoma_retargeting.visualization.result_loader import (
    VariantResult,
    _resolve_object_urdf,
    _resolve_robot_urdf,
    _resolve_robot_xml,
    _rgba,
    _robot_joints_for_viser,
    interpolate_qpos,
    load_variant_result,
    resolve_qpos_to_viser_joint_indices,
    split_result_family,
)


@dataclass(frozen=True)
class MultiResultViserConfig:
    """Configuration for one or more synchronized retargeting results."""

    qpos_npzs: tuple[Path, ...]
    """Result NPZ paths in display order."""

    labels: tuple[str, ...] = ()
    """Optional unique display labels. Strict results default to saved variants."""

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

    robot_mesh_opacity: float = 0.5
    """Opacity of the color-coded robot meshes."""

    object_mesh_opacity: float = 0.30
    """Opacity of object meshes when object poses are present."""

    show_robot_mesh: bool = True
    """Show robot meshes for all motions."""

    show_object_mesh: bool = True
    """Show object or scene meshes for all motions."""

    show_human_skeleton: bool = True
    """Show one shared source-human reference made from mapped keypoints."""

    show_robot_skeleton: bool = True
    """Show mapped robot keypoints and skeletons for all motions."""

    show_human_hands: bool = True
    """Show complete source-human hand details when present."""

    show_source_orientation_axes: bool = True
    """Show directly observed source-human orientation frames when saved."""

    show_object_keypoints: bool = False
    """Show saved demonstration and target object samples for all motions."""

    show_point_clouds: bool = False
    """Show saved human, robot, terrain, and object point clouds."""

    show_interaction_mesh: bool = False
    """Show saved source and target interaction meshes for all motions."""

    interaction_mesh_mode: str = "both"
    """Which interaction mesh content to show: source, target, or both."""

    interaction_mesh_edges: str = "cross"
    """Which interaction mesh edges to show: cross or all."""

    interaction_mesh_line_width: float = 1.0
    """Line width for saved interaction mesh edges."""

    object_keypoint_radius: float = 0.02
    """World-space radius for saved object samples."""

    point_cloud_point_size: float = 0.012
    """Rendered size of saved retargeting point-cloud samples."""

    show_foot_sticking: bool = True
    """Show per-motion foot-sticking status when saved states are available."""

    human_skeleton_line_width: float = 3.5
    """Pixel width shared by the clean human and robot skeletons."""

    human_joint_point_size: float = 0.015
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

    orientation_axis_length: float = 0.065
    """Length of each orientation axis in scene units."""

    orientation_axis_shaft_radius: float = 0.005
    """Radius of each solid orientation arrow shaft."""

    orientation_axis_head_radius: float = 0.01
    """Radius of each solid orientation arrow head."""

    orientation_axis_head_length: float = 0.015
    """Length of each solid orientation arrow head."""


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
    axis_point_indices: np.ndarray
    target_axes: object | None
    axis_length: float
    target_visible: bool
    loop: bool
    enabled: bool = True

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self.joints_handle.visible = self.enabled
        self.bones_handle.visible = self.enabled

    def set_target_visible(self, visible: bool) -> None:
        self.target_visible = bool(visible)
        if self.target_axes is not None:
            self.target_axes.visible = self.target_visible

    def update(self, frame_float: float) -> None:
        points = (
            interpolate_human_points(
                self.trajectory.points,
                frame_float,
                loop=self.loop,
            )
            + self.offset
        )
        self.joints_handle.points = points.astype(np.float32)
        edge_indices = np.asarray(self.trajectory.edges, dtype=np.int32)
        self.bones_handle.points = points[edge_indices].astype(np.float32)
        if self.diagnostics is not None and self.target_axes is not None:
            origins = points[self.axis_point_indices]
            target_quaternions = interpolate_orientation_quaternions(
                self.diagnostics.target_quaternions_wxyz[:, self.joint_indices],
                frame_float,
                loop=self.loop,
            )
            self.target_axes.points = orientation_axis_segments(
                origins,
                target_quaternions,
                self.axis_length,
            )


@dataclass
class HumanHandsOverlay:
    """Visual-only full source-human finger keypoints and chains."""

    points: np.ndarray
    keypoint_indices: np.ndarray
    edge_indices: np.ndarray
    offset: np.ndarray
    joints_handle: object
    bones_handle: object
    loop: bool
    enabled: bool = True

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self.joints_handle.visible = self.enabled
        self.bones_handle.visible = self.enabled

    def update(self, frame_float: float) -> None:
        frame_points = (
            interpolate_human_points(
                self.points,
                frame_float,
                loop=self.loop,
            )
            + self.offset
        )
        self.joints_handle.points = frame_points[self.keypoint_indices].astype(np.float32)
        self.bones_handle.points = frame_points[self.edge_indices].astype(np.float32)


@dataclass
class RobotSkeletonOverlay:
    """Mapped robot keypoints and connecting segments for one result."""

    points: np.ndarray
    edges: tuple[tuple[int, int], ...]
    offset: np.ndarray
    joints_handle: object
    bones_handle: object
    loop: bool
    enabled: bool = True

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self.joints_handle.visible = self.enabled
        self.bones_handle.visible = self.enabled

    def update(self, frame_float: float) -> None:
        points = (
            interpolate_human_points(
                self.points,
                frame_float,
                loop=self.loop,
            )
            + self.offset
        )
        self.joints_handle.points = points.astype(np.float32)
        edge_indices = np.asarray(self.edges, dtype=np.int32)
        self.bones_handle.points = points[edge_indices].astype(np.float32)


@dataclass
class OrientationOverlay:
    """Actual robot link-frame axes and errors for one result."""

    diagnostics: OrientationDiagnostics
    joint_indices: np.ndarray
    model: mujoco.MjModel
    skeleton_points: np.ndarray
    axis_point_indices: np.ndarray
    offset: np.ndarray
    robot_axes: object
    axis_length: float
    robot_visible: bool
    loop: bool
    enabled: bool = True

    def refresh_visibility(self) -> None:
        self.robot_axes.visible = self.enabled and self.robot_visible

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self.refresh_visibility()

    def set_robot_visible(self, visible: bool) -> None:
        self.robot_visible = bool(visible)
        self.refresh_visibility()

    def update(self, q: np.ndarray, frame_float: float) -> None:
        if q.shape[0] < self.model.nq:
            raise ValueError(f"qpos has {q.shape[0]} values but the orientation FK model requires {self.model.nq}")
        skeleton_points = interpolate_human_points(
            self.skeleton_points,
            frame_float,
            loop=self.loop,
        )
        origins = skeleton_points[self.axis_point_indices] + self.offset
        robot_quaternions = interpolate_orientation_quaternions(
            self.diagnostics.robot_quaternions_wxyz[:, self.joint_indices],
            frame_float,
            loop=self.loop,
        )
        self.robot_axes.points = orientation_axis_segments(
            origins,
            robot_quaternions,
            self.axis_length,
        )


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
    contains_object: bool
    qpos_to_viser_joint_indices: np.ndarray | None
    robot_skeleton: RobotSkeletonOverlay
    orientation_overlay: OrientationOverlay | SavedOrientationAxesOverlay | None
    object_keypoints_overlay: ObjectKeypointOverlay | None = None
    point_cloud_overlay: RetargetingPointCloudOverlay | None = None
    interaction_mesh_overlay: InteractionMeshOverlay | None = None
    robot_mesh_enabled: bool = True
    object_mesh_enabled: bool = True
    skeleton_enabled: bool = True
    orientation_enabled: bool = True
    object_keypoints_enabled: bool = False
    point_clouds_enabled: bool = False
    interaction_mesh_enabled: bool = False
    motion_visible: bool = True

    @property
    def enabled(self) -> bool:
        """Whether any part of this comparison result is visible."""

        return self.motion_visible and (
            self.robot_mesh_enabled
            or self.object_mesh_enabled
            or self.skeleton_enabled
            or self.orientation_enabled
            or self.object_keypoints_enabled
            or self.point_clouds_enabled
            or self.interaction_mesh_enabled
        )

    def _refresh_visibility(self) -> None:
        self.robot.show_visual = self.motion_visible and self.robot_mesh_enabled
        if self.object_visual is not None:
            self.object_visual.show_visual = self.motion_visible and self.object_mesh_enabled
        self.robot_skeleton.set_enabled(self.motion_visible and self.skeleton_enabled)
        if self.orientation_overlay is not None:
            self.orientation_overlay.set_enabled(self.motion_visible and self.orientation_enabled)
        if self.object_keypoints_overlay is not None:
            self.object_keypoints_overlay.set_visible(self.motion_visible and self.object_keypoints_enabled)
        if self.point_cloud_overlay is not None:
            self.point_cloud_overlay.set_visible(self.motion_visible and self.point_clouds_enabled)
        if self.interaction_mesh_overlay is not None:
            self.interaction_mesh_overlay.set_visible(self.motion_visible and self.interaction_mesh_enabled)

    def set_mesh_enabled(self, enabled: bool) -> None:
        """Backward-compatible combined mesh toggle."""

        self.set_robot_mesh_enabled(enabled)
        self.set_object_mesh_enabled(enabled)

    def set_robot_mesh_enabled(self, enabled: bool) -> None:
        self.robot_mesh_enabled = bool(enabled)
        self.robot.show_visual = self.motion_visible and self.robot_mesh_enabled

    def set_object_mesh_enabled(self, enabled: bool) -> None:
        self.object_mesh_enabled = bool(enabled)
        if self.object_visual is not None:
            self.object_visual.show_visual = self.motion_visible and self.object_mesh_enabled

    def set_skeleton_enabled(self, enabled: bool) -> None:
        """Toggle mapped robot keypoints, bones, and their attached axes."""

        self.skeleton_enabled = bool(enabled)
        self.robot_skeleton.set_enabled(self.motion_visible and self.skeleton_enabled)
        if self.orientation_overlay is not None:
            self.orientation_enabled = self.skeleton_enabled
            self.orientation_overlay.set_enabled(self.motion_visible and self.orientation_enabled)

    def set_robot_orientation_enabled(self, enabled: bool) -> None:
        self.orientation_enabled = bool(enabled)
        if self.orientation_overlay is not None:
            self.orientation_overlay.set_enabled(self.motion_visible and self.orientation_enabled)

    def set_object_keypoints_enabled(self, enabled: bool) -> None:
        self.object_keypoints_enabled = bool(enabled)
        if self.object_keypoints_overlay is not None:
            self.object_keypoints_overlay.set_visible(self.motion_visible and self.object_keypoints_enabled)

    def set_point_clouds_enabled(self, enabled: bool) -> None:
        self.point_clouds_enabled = bool(enabled)
        if self.point_cloud_overlay is not None:
            self.point_cloud_overlay.set_visible(self.motion_visible and self.point_clouds_enabled)

    def set_interaction_mesh_enabled(self, enabled: bool) -> None:
        self.interaction_mesh_enabled = bool(enabled)
        if self.interaction_mesh_overlay is not None:
            self.interaction_mesh_overlay.set_visible(self.motion_visible and self.interaction_mesh_enabled)

    def set_enabled(self, enabled: bool) -> None:
        """Hide or restore a motion without losing its layer selections."""

        self.motion_visible = bool(enabled)
        self._refresh_visibility()


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


def compact_layer_label(label: str) -> str:
    """Make command-line comparison labels compact and readable in the GUI."""

    return " ".join(label.replace("_", " ").split())


def comparison_offsets(
    result_count: int,
    x_offset: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Overlay the human reference on the first robot and space only extra results."""

    if result_count < 1:
        raise ValueError("result_count must be positive")
    centered_x = (np.arange(result_count, dtype=float) - 0.5 * (result_count - 1)) * float(x_offset)
    robot_offsets = np.zeros((result_count, 3), dtype=float)
    robot_offsets[:, 0] = centered_x
    return robot_offsets[0].copy(), robot_offsets


def pale_mesh_color(color: tuple[int, int, int]) -> tuple[int, int, int]:
    """Produce a pale low-saturation tint that does not obscure the skeleton."""

    return (
        round(0.25 * color[0] + 0.75 * 235),
        round(0.25 * color[1] + 0.75 * 235),
        round(0.25 * color[2] + 0.75 * 235),
    )


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
    source: Path | VariantResult,
    expected_frames: int | None = None,
) -> HumanSkeletonTrajectory:
    """Build human skeleton trajectories from the shared result loader."""

    result = (
        source
        if isinstance(source, VariantResult)
        else load_variant_result(
            Path(source).stem,
            source,
        )
    )
    path = result.path
    if result.human_joints is None:
        raise KeyError(f"{path} is missing required field 'human_joints'")
    full_points = np.asarray(result.human_joints, dtype=np.float32)
    full_joint_names = result.human_joint_names
    joint_names = result.mapped_joint_names
    frame_count = result.qpos.shape[0] if expected_frames is None else expected_frames
    expected_shape = (frame_count, len(full_joint_names), 3)
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
    *,
    loop: bool = False,
) -> np.ndarray:
    """Linearly interpolate a ``(frames, joints, 3)`` skeleton sequence."""

    values = np.asarray(points, dtype=float)
    if values.ndim != 3 or values.shape[-1] != 3 or values.shape[0] == 0:
        raise ValueError("human skeleton sequence must have shape (frames, joints, 3)")
    if loop:
        sample_frame = float(frame_float) % values.shape[0]
        frame0 = int(np.floor(sample_frame))
        frame1 = (frame0 + 1) % values.shape[0]
    else:
        sample_frame = float(np.clip(frame_float, 0.0, values.shape[0] - 1))
        frame0 = int(np.floor(sample_frame))
        frame1 = min(frame0 + 1, values.shape[0] - 1)
    fraction = sample_frame - frame0
    return (1.0 - fraction) * values[frame0] + fraction * values[frame1]


def _make_human_reference_overlay(
    *,
    server,
    namespace: str,
    trajectory: HumanSkeletonTrajectory,
    diagnostics: OrientationDiagnostics | None,
    joint_indices: np.ndarray,
    axis_point_indices: np.ndarray,
    offset: np.ndarray,
    config: MultiResultViserConfig,
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
    target_axes = None
    if diagnostics is not None:
        origins = trajectory.points[0, axis_point_indices] + offset
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
            visible=config.show_target_orientation_axes,
        )
    return HumanReferenceOverlay(
        trajectory=trajectory,
        offset=offset,
        joints_handle=joints_handle,
        bones_handle=bones_handle,
        diagnostics=diagnostics,
        joint_indices=joint_indices,
        axis_point_indices=axis_point_indices,
        target_axes=target_axes,
        axis_length=config.orientation_axis_length * _TARGET_AXIS_LENGTH_SCALE,
        target_visible=config.show_target_orientation_axes,
        loop=config.loop,
        enabled=config.show_human_skeleton,
    )


def _make_human_hands_overlay(
    *,
    server,
    namespace: str,
    trajectory: HumanSkeletonTrajectory,
    offset: np.ndarray,
    config: MultiResultViserConfig,
) -> HumanHandsOverlay | None:
    spec = build_hand_visualization_spec(
        list(trajectory.full_joint_names),
        list(trajectory.joint_names),
    )
    if not spec.keypoint_indices or not spec.edge_indices:
        return None
    keypoint_indices = np.asarray(spec.keypoint_indices, dtype=np.int32)
    edge_indices = np.asarray(spec.edge_indices, dtype=np.int32)
    frame_points = trajectory.full_points[0] + offset
    visible = bool(config.show_human_hands)
    hand_color = np.asarray((75, 78, 88), dtype=np.uint8)
    joints_handle = server.scene.add_point_cloud(
        f"{namespace}/human/hands/joints",
        points=frame_points[keypoint_indices],
        colors=np.tile(hand_color, (len(keypoint_indices), 1)),
        point_size=config.human_joint_point_size * 0.7,
        point_shape="circle",
        visible=visible,
    )
    bones_handle = server.scene.add_line_segments(
        f"{namespace}/human/hands/skeleton",
        points=frame_points[edge_indices],
        colors=hand_color,
        line_width=config.human_skeleton_line_width * 0.7,
        visible=visible,
    )
    return HumanHandsOverlay(
        points=trajectory.full_points,
        keypoint_indices=keypoint_indices,
        edge_indices=edge_indices,
        offset=offset,
        joints_handle=joints_handle,
        bones_handle=bones_handle,
        loop=config.loop,
        enabled=visible,
    )


def _make_robot_skeleton_overlay(
    *,
    server,
    namespace: str,
    result: VariantResult,
    color: tuple[int, int, int],
    offset: np.ndarray,
    config: MultiResultViserConfig,
) -> RobotSkeletonOverlay:
    if (
        result.robot_link_positions is not None
        and result.robot_link_parent_indices is not None
        and result.robot_link_names
    ):
        trajectory = result.robot_link_positions
        display_names = result.robot_link_names
        edges = tuple(
            (int(parent), index) for index, parent in enumerate(result.robot_link_parent_indices) if parent >= 0
        )
    else:
        if result.robot_points is None:
            raise ValueError(f"{result.path} has neither full nor mapped robot skeleton positions")
        trajectory = result.robot_points
        display_names = result.mapped_robot_link_names
        edges = visualization_skeleton_edges(result.mapped_joint_names)
    points = trajectory[0] + offset
    skeleton_color = darker_color(color)
    joints_handle = server.scene.add_point_cloud(
        f"{namespace}/skeleton/joints",
        points=points,
        colors=np.tile(skeleton_color, (len(display_names), 1)),
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
        points=trajectory,
        edges=edges,
        offset=offset,
        joints_handle=joints_handle,
        bones_handle=bones_handle,
        loop=config.loop,
    )


def _comparison_paths(config: MultiResultViserConfig) -> tuple[Path, ...]:
    if not config.qpos_npzs:
        raise ValueError("At least one result NPZ path is required.")
    if config.labels and len(config.labels) != len(config.qpos_npzs):
        raise ValueError(f"labels has {len(config.labels)} entries for {len(config.qpos_npzs)} result paths")
    if config.labels and (
        len(set(config.labels)) != len(config.labels) or any(not label.strip() for label in config.labels)
    ):
        raise ValueError("Comparison labels must be non-empty and unique.")

    paths = tuple(Path(path).expanduser() for path in config.qpos_npzs)
    resolved_paths = tuple(path.resolve(strict=True) for path in paths)
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError(
            "Comparison result paths must refer to unique physical files; symlink aliases are duplicates.",
        )
    return paths


def _semantic_comparison_labels(
    config: MultiResultViserConfig,
    results: list[VariantResult],
) -> tuple[str, ...]:
    if config.labels:
        return config.labels

    base_labels = [result.variant or result.path.stem for result in results]
    base_counts = {label: base_labels.count(label) for label in set(base_labels)}
    labels: list[str] = []
    used: dict[str, int] = {}
    for base_label, result in zip(base_labels, results, strict=True):
        label = base_label
        if base_counts[base_label] > 1:
            context = result.experiment_name or result.run_kind or result.path.parent.name
            label = f"{context}:{base_label}"
        occurrence = used.get(label, 0) + 1
        used[label] = occurrence
        labels.append(label if occurrence == 1 else f"{label} [{occurrence}]")
    return tuple(labels)


def load_comparison_results(
    config: MultiResultViserConfig,
) -> tuple[tuple[str, ...], list[VariantResult]]:
    """Load arbitrary result paths and recover fields needed for rendering."""

    paths = _comparison_paths(config)
    results = [
        load_variant_result(
            split_result_family(path)[1] or "identity",
            path,
        )
        for path in paths
    ]
    labels = _semantic_comparison_labels(config, results)
    reference = results[0]
    if config.fps is None and any(not np.isclose(result.fps, reference.fps) for result in results[1:]):
        raise ValueError(
            "Comparison results use different saved FPS values; pass --fps to choose one shared playback rate.",
        )
    robot_types = {result.robot_type for result in results if result.robot_type}
    if len(robot_types) > 1:
        raise ValueError(
            f"One shared robot model cannot render multiple robot types: {sorted(robot_types)}",
        )
    needs_layout_inference = any(result.contains_object_in_qpos is None for result in results)
    needs_robot_fk = any(result.robot_points is None for result in results)
    if needs_layout_inference or needs_robot_fk:
        robot_urdf = _resolve_robot_urdf(config, reference)
        robot_xml = _resolve_robot_xml(config, robot_urdf)
        if robot_xml is None:
            missing_fields = []
            if needs_layout_inference:
                missing_fields.append("contains_object_in_qpos")
            if needs_robot_fk:
                missing_fields.append("mapped_robot_joints")
            raise FileNotFoundError(
                "A MuJoCo robot XML is required to recover legacy result fields "
                f"{tuple(missing_fields)}; pass --robot-mujoco-xml."
            )
        robot_model = mujoco.MjModel.from_xml_path(str(robot_xml))
        robot_data = mujoco.MjData(robot_model)
        normalized_results: list[VariantResult] = []
        for result in results:
            contains_object = result.contains_object_in_qpos
            if contains_object is None:
                qpos_width = result.qpos.shape[1]
                if qpos_width == robot_model.nq:
                    contains_object = False
                elif qpos_width == robot_model.nq + 7:
                    contains_object = True
                else:
                    raise ValueError(
                        f"{result.path} qpos width {qpos_width} cannot be matched to "
                        f"robot nq={robot_model.nq} with or without a 7-DoF object pose"
                    )
            robot_points = result.robot_points
            if robot_points is None:
                if result.qpos.shape[1] < robot_model.nq:
                    raise ValueError(
                        f"{result.path} qpos width {result.qpos.shape[1]} is smaller than robot nq={robot_model.nq}"
                    )
                body_ids = np.asarray(
                    [
                        mujoco.mj_name2id(
                            robot_model,
                            mujoco.mjtObj.mjOBJ_BODY,
                            link_name,
                        )
                        for link_name in result.mapped_robot_link_names
                    ],
                    dtype=np.int32,
                )
                missing_links = tuple(
                    link_name
                    for link_name, body_id in zip(
                        result.mapped_robot_link_names,
                        body_ids,
                        strict=True,
                    )
                    if body_id < 0
                )
                if missing_links:
                    raise ValueError(
                        f"{result.path} mapped robot links are absent from the MuJoCo model: {missing_links}"
                    )
                robot_points = np.empty(
                    (result.qpos.shape[0], len(body_ids), 3),
                    dtype=np.float32,
                )
                for frame_index, qpos_frame in enumerate(result.qpos):
                    robot_data.qpos[:] = qpos_frame[: robot_model.nq]
                    mujoco.mj_forward(robot_model, robot_data)
                    robot_points[frame_index] = robot_data.xpos[body_ids]
            normalized_results.append(
                replace(
                    result,
                    robot_points=robot_points,
                    contains_object_in_qpos=contains_object,
                )
            )
        results = normalized_results
    return labels, results


def resolve_comparison_object_urdfs(
    config: MultiResultViserConfig,
    results: list[VariantResult],
) -> tuple[Path | None, ...]:
    """Resolve the exact object or scene asset used by each result.

    Climbing scale variants intentionally point at distinct generated URDFs.
    Resolving only the reference result would therefore render every comparison
    with the reference scale even though the saved trajectories are different.
    An explicit viewer override remains shared by design.
    """

    resolved: list[Path | None] = []
    for result in results:
        has_asset = bool(
            result.contains_object_in_qpos or config.object_urdf is not None or result.object_urdf is not None
        )
        resolved.append(_resolve_object_urdf(config, result) if has_asset else None)
    return tuple(resolved)


def _make_orientation_overlay(
    *,
    server,
    namespace: str,
    diagnostics: OrientationDiagnostics,
    joint_indices: np.ndarray,
    model: mujoco.MjModel,
    initial_q: np.ndarray,
    skeleton_points: np.ndarray,
    axis_point_indices: np.ndarray,
    offset: np.ndarray,
    config: MultiResultViserConfig,
) -> OrientationOverlay:
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
    origins = skeleton_points[0, axis_point_indices] + offset
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
    return OrientationOverlay(
        diagnostics=diagnostics,
        joint_indices=joint_indices,
        model=model,
        skeleton_points=skeleton_points,
        axis_point_indices=axis_point_indices,
        offset=offset,
        robot_axes=robot_axes,
        axis_length=config.orientation_axis_length,
        robot_visible=config.show_robot_orientation_axes,
        loop=config.loop,
    )


def _make_full_robot_orientation_overlay(
    *,
    server,
    namespace: str,
    result: VariantResult,
    offset: np.ndarray,
    config: MultiResultViserConfig,
) -> SavedOrientationAxesOverlay | None:
    """Build robot axes directly from the complete saved link trajectories."""

    if result.robot_link_positions is None or result.robot_link_quaternions_wxyz is None or not result.robot_link_names:
        return None
    aliases = dict(
        zip(
            result.mapped_joint_names,
            result.mapped_robot_link_names,
            strict=True,
        )
    )
    selected = _requested_saved_orientation_indices(
        result.robot_link_names,
        config.orientation_joints,
        aliases=aliases,
    )
    if selected.size == 0:
        return None
    return SavedOrientationAxesOverlay(
        server=server,
        namespace=f"{namespace}/orientation/robot",
        names=tuple(result.robot_link_names[int(index)] for index in selected),
        positions=result.robot_link_positions[:, selected] + offset,
        quaternions_wxyz=result.robot_link_quaternions_wxyz[:, selected],
        axis_length=config.orientation_axis_length,
        shaft_radius=config.orientation_axis_shaft_radius,
        head_radius=config.orientation_axis_head_radius,
        head_length=config.orientation_axis_head_length,
        visible=config.show_robot_orientation_axes,
        loop=config.loop,
    )


def _make_source_orientation_overlay(
    *,
    server,
    result: VariantResult,
    offset: np.ndarray,
    config: MultiResultViserConfig,
) -> SavedOrientationAxesOverlay | None:
    """Build source axes only from explicitly saved human frames."""

    if (
        result.human_joints is None
        or result.human_orientation_quaternions_wxyz is None
        or not result.human_orientation_joint_names
    ):
        return None
    full_index = {name: index for index, name in enumerate(result.human_joint_names)}
    point_indices = np.asarray(
        [full_index[name] for name in result.human_orientation_joint_names],
        dtype=np.int32,
    )
    selected = _requested_saved_orientation_indices(
        result.human_orientation_joint_names,
        config.orientation_joints,
    )
    if selected.size == 0:
        return None
    return SavedOrientationAxesOverlay(
        server=server,
        namespace="/reference_human/orientation/source",
        names=tuple(result.human_orientation_joint_names[int(index)] for index in selected),
        positions=(result.human_joints[:, point_indices[selected]] + offset),
        quaternions_wxyz=(result.human_orientation_quaternions_wxyz[:, selected]),
        axis_length=config.orientation_axis_length,
        shaft_radius=config.orientation_axis_shaft_radius,
        head_radius=config.orientation_axis_head_radius,
        head_length=config.orientation_axis_head_length,
        visible=config.show_source_orientation_axes,
        loop=config.loop,
    )


def make_multi_result_player(
    config: MultiResultViserConfig,
    labels: tuple[str, ...],
    results: list[VariantResult],
):
    """Build one shared-timeline Viser scene for all comparison results."""

    reference = results[0]
    robot_urdf = _resolve_robot_urdf(config, reference)
    robot_xml = _resolve_robot_xml(config, robot_urdf)
    driver_contains_object = bool(reference.contains_object_in_qpos)
    object_urdfs = resolve_comparison_object_urdfs(config, results)
    if config.interaction_mesh_mode not in {"source", "target", "both"}:
        raise ValueError("interaction_mesh_mode must be one of source, target, or both")
    if config.interaction_mesh_edges not in {"cross", "all"}:
        raise ValueError("interaction_mesh_edges must be cross or all")
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
    orientation_diagnostics = [result.orientation_diagnostics for result in results]
    reference_skeleton = load_human_skeleton(reference)
    robot_fk_model = None
    needs_legacy_orientation_fk = any(
        diagnostics is not None and result.robot_link_quaternions_wxyz is None
        for result, diagnostics in zip(
            results,
            orientation_diagnostics,
            strict=True,
        )
    )
    if needs_legacy_orientation_fk:
        if robot_xml is None:
            raise FileNotFoundError(
                "A MuJoCo robot XML is required to position orientation arrows; pass --robot-mujoco-xml."
            )
        robot_fk_model = mujoco.MjModel.from_xml_path(str(robot_xml))

    server = viser.ViserServer()
    tabs = add_visualization_tabs(server.gui, include_motions=True)
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
    object_urdf_models: dict[Path, object] = {}

    scenes: list[ComparisonScene] = []
    robot_dof: int | None = None
    driver_joint_order_indices: np.ndarray | None = None
    human_offset, robot_offsets = comparison_offsets(
        len(results),
        config.x_offset,
    )
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
    reference_axis_point_indices = (
        orientation_skeleton_point_indices(
            reference_diagnostics,
            reference_skeleton.joint_names,
            reference.mapped_robot_link_names,
        )[reference_joint_indices]
        if reference_diagnostics is not None
        else np.empty((0,), dtype=np.int32)
    )
    human_reference = _make_human_reference_overlay(
        server=server,
        namespace="/reference_human",
        trajectory=reference_skeleton,
        diagnostics=reference_diagnostics,
        joint_indices=reference_joint_indices,
        axis_point_indices=reference_axis_point_indices,
        offset=human_offset,
        config=config,
    )
    source_orientation_overlay = _make_source_orientation_overlay(
        server=server,
        result=reference,
        offset=human_offset,
        config=config,
    )
    human_hands = _make_human_hands_overlay(
        server=server,
        namespace="/reference_human",
        trajectory=reference_skeleton,
        offset=human_offset,
        config=config,
    )

    for index, (label, result, diagnostics, object_urdf) in enumerate(
        zip(
            labels,
            results,
            orientation_diagnostics,
            object_urdfs,
            strict=True,
        )
    ):
        color = comparison_color(index)
        namespace = f"/groups/{index:02d}_{label}"
        offset = robot_offsets[index]
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
        if object_urdf is not None:
            object_urdf_model = object_urdf_models.get(object_urdf)
            if object_urdf_model is None:
                object_urdf_model = yourdfpy.URDF.load(
                    str(object_urdf),
                    load_meshes=True,
                    build_scene_graph=True,
                )
                object_urdf_models[object_urdf] = object_urdf_model
            object_root = server.scene.add_frame(f"{namespace}/object", show_axes=False)
            if not result.contains_object_in_qpos:
                object_root.position = offset
            object_visual = ViserUrdf(
                server,
                urdf_or_path=object_urdf_model,
                root_node_name=f"{namespace}/object",
                mesh_color_override=_rgba(color, config.object_mesh_opacity),
            )

        current_robot_dof = len(robot.get_actuated_joint_limits())
        if robot_dof is None:
            robot_dof = current_robot_dof
        elif current_robot_dof != robot_dof:
            raise ValueError("Loaded comparison robots expose inconsistent actuated joint counts.")
        qpos_to_viser_joint_indices = resolve_qpos_to_viser_joint_indices(
            result_path=result.path,
            saved_joint_names=result.robot_actuated_joint_names,
            viser_joint_names=tuple(robot.get_actuated_joint_limits().keys()),
            fallback_mujoco_xml=robot_xml,
        )
        if index == 0:
            driver_joint_order_indices = qpos_to_viser_joint_indices

        orientation_overlay = _make_full_robot_orientation_overlay(
            server=server,
            namespace=namespace,
            result=result,
            offset=offset,
            config=config,
        )
        if orientation_overlay is None and diagnostics is not None:
            if robot_fk_model is None:
                raise RuntimeError("Orientation FK model was not initialized.")
            joint_indices = orientation_joint_indices(
                diagnostics,
                config.orientation_joints,
            )
            axis_point_indices = orientation_skeleton_point_indices(
                diagnostics,
                result.mapped_joint_names,
                result.mapped_robot_link_names,
            )[joint_indices]
            orientation_overlay = _make_orientation_overlay(
                server=server,
                namespace=namespace,
                diagnostics=diagnostics,
                joint_indices=joint_indices,
                model=robot_fk_model,
                initial_q=result.qpos[0],
                skeleton_points=result.robot_points,
                axis_point_indices=axis_point_indices,
                offset=offset,
                config=config,
            )

        object_keypoints_overlay = (
            ObjectKeypointOverlay(
                server=server,
                keypoint_data=result.object_keypoints,
                point_radius=config.object_keypoint_radius,
                namespace=f"{namespace}/object_keypoints",
                offset=offset,
                loop=config.loop,
            )
            if result.object_keypoints is not None
            else None
        )
        if object_keypoints_overlay is not None:
            object_keypoints_overlay.set_visible(config.show_object_keypoints)
        point_cloud_trajectories: dict[str, np.ndarray] = {}
        for layer_name, points in (
            ("human", result.human_points_world),
            ("robot", result.robot_points_world),
            ("terrain", result.terrain_points_world),
        ):
            if points is not None:
                point_cloud_trajectories[layer_name] = (
                    np.asarray(
                        points,
                        dtype=np.float32,
                    )
                    + offset
                )
        if result.object_keypoints is not None:
            for layer_name, field_name in (
                ("object_demo", "demo_world"),
                ("object_target", "target_world"),
            ):
                points = result.object_keypoints.get(field_name)
                if points is not None:
                    point_cloud_trajectories[layer_name] = (
                        np.asarray(
                            points,
                            dtype=np.float32,
                        )
                        + offset
                    )
        point_cloud_overlay = (
            RetargetingPointCloudOverlay(
                server=server,
                trajectories=point_cloud_trajectories,
                point_size=config.point_cloud_point_size,
                namespace=f"{namespace}/retargeting_points",
                loop=config.loop,
            )
            if point_cloud_trajectories
            else None
        )
        if point_cloud_overlay is not None:
            point_cloud_overlay.set_visible(config.show_point_clouds)
        interaction_mesh_overlay = (
            InteractionMeshOverlay(
                server=server,
                mesh_data=result.interaction_mesh,
                mode=config.interaction_mesh_mode,
                edge_mode=config.interaction_mesh_edges,
                line_width=config.interaction_mesh_line_width,
                namespace=f"{namespace}/interaction_mesh",
                offset=offset,
                loop=config.loop,
            )
            if result.interaction_mesh is not None
            else None
        )
        if interaction_mesh_overlay is not None:
            interaction_mesh_overlay.set_visible(config.show_interaction_mesh)

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
                contains_object=bool(result.contains_object_in_qpos),
                qpos_to_viser_joint_indices=qpos_to_viser_joint_indices,
                robot_skeleton=robot_skeleton,
                orientation_overlay=orientation_overlay,
                object_keypoints_overlay=object_keypoints_overlay,
                point_cloud_overlay=point_cloud_overlay,
                interaction_mesh_overlay=interaction_mesh_overlay,
                robot_mesh_enabled=config.show_robot_mesh,
                object_mesh_enabled=config.show_object_mesh and object_visual is not None,
                skeleton_enabled=config.show_robot_skeleton,
                object_keypoints_enabled=config.show_object_keypoints and object_keypoints_overlay is not None,
                point_clouds_enabled=config.show_point_clouds and point_cloud_overlay is not None,
                interaction_mesh_enabled=config.show_interaction_mesh and interaction_mesh_overlay is not None,
            )
        )
        robot.show_visual = bool(config.show_robot_mesh)
        robot_skeleton.set_enabled(config.show_robot_skeleton)
        if object_visual is not None:
            object_visual.show_visual = bool(config.show_object_mesh)
        if orientation_overlay is not None:
            orientation_overlay.set_enabled(config.show_robot_orientation_axes)

    if robot_dof is None:
        raise RuntimeError("No comparison results were loaded.")

    def _render_scene(
        scene: ComparisonScene,
        q: np.ndarray,
        frame_float: float,
        *,
        update_articulation: bool,
    ) -> None:
        if scene.skeleton_enabled:
            scene.robot_skeleton.update(frame_float)
        if update_articulation:
            scene.robot.update_cfg(
                _robot_joints_for_viser(
                    q,
                    robot_dof,
                    scene.qpos_to_viser_joint_indices,
                )
            )
        scene.robot_root.position = q[0:3] + scene.offset
        if update_articulation:
            scene.robot_root.wxyz = q[3:7]
        if scene.contains_object and scene.object_root is not None:
            scene.object_root.position = q[-7:-4] + scene.offset
            if update_articulation:
                scene.object_root.wxyz = q[-4:]
        if scene.orientation_overlay is not None and scene.orientation_overlay.enabled:
            scene.orientation_overlay.update(q, frame_float)
        if scene.object_keypoints_overlay is not None and scene.object_keypoints_enabled:
            scene.object_keypoints_overlay.draw(frame_float)
        if scene.point_cloud_overlay is not None and scene.point_clouds_enabled:
            scene.point_cloud_overlay.draw(frame_float)
        if scene.interaction_mesh_overlay is not None and scene.interaction_mesh_enabled:
            scene.interaction_mesh_overlay.draw(frame_float)

    foot_sticking_status_handle = None
    last_frame = {"value": 0.0}

    def _foot_sticking_markdown(frame_float: float) -> str:
        sections: list[str] = []
        for scene in scenes:
            foot_sticking = scene.result.foot_sticking
            if foot_sticking is None:
                continue
            states = np.asarray(foot_sticking["states"], dtype=bool)
            if config.loop:
                frame_idx = round(float(frame_float)) % states.shape[0]
            else:
                frame_idx = int(np.clip(round(float(frame_float)), 0, states.shape[0] - 1))
            sections.append(
                f"#### {compact_layer_label(scene.label)}\n"
                + format_foot_sticking_status(
                    frame_idx,
                    states[frame_idx],
                    constraint_status=_saved_foot_sticking_constraint_status(
                        foot_sticking,
                        frame_idx,
                    ),
                    modes=(
                        np.asarray(foot_sticking["modes"], dtype=str)[frame_idx] if "modes" in foot_sticking else None
                    ),
                )
            )
        return "\n\n".join(sections)

    def _render_comparison(driver_q: np.ndarray, frame_float: float) -> None:
        last_frame["value"] = float(frame_float)
        if human_reference.enabled or human_reference.target_visible:
            human_reference.update(frame_float)
        if source_orientation_overlay is not None and source_orientation_overlay.enabled:
            source_orientation_overlay.draw(frame_float)
        if human_hands is not None and human_hands.enabled:
            human_hands.update(frame_float)
        if foot_sticking_status_handle is not None:
            foot_sticking_status_handle.content = _foot_sticking_markdown(frame_float)
        for index, scene in enumerate(scenes):
            q = (
                driver_q
                if index == 0
                else interpolate_qpos(
                    scene.result.qpos,
                    frame_float,
                    robot_dof,
                    contains_object=scene.contains_object,
                    loop=config.loop,
                )
            )
            # The shared motion controller already updates the first robot.
            # Rewriting it here doubled the continuous-playback message load.
            _render_scene(
                scene,
                q,
                frame_float,
                update_articulation=(index != 0),
            )

    layer_controller = LayerController()

    def _set_all_robot_skeletons(visible: bool) -> None:
        for scene in scenes:
            scene.set_skeleton_enabled(visible)
            if layer_controller.is_visible(LayerId.ROBOT_ORIENTATION):
                scene.set_robot_orientation_enabled(True)

    layer_controller.register(
        LayerId.ROBOT_MESH,
        available=True,
        visible=config.show_robot_mesh,
        callback=lambda visible: [scene.set_robot_mesh_enabled(visible) for scene in scenes],
    )
    layer_controller.register(
        LayerId.OBJECT_MESH,
        available=any(scene.object_visual is not None for scene in scenes),
        visible=config.show_object_mesh,
        callback=lambda visible: [scene.set_object_mesh_enabled(visible) for scene in scenes],
        unavailable_reason="No input contains a resolved object or scene asset.",
    )
    layer_controller.register(
        LayerId.HUMAN_SKELETON,
        available=True,
        visible=config.show_human_skeleton,
        callback=human_reference.set_enabled,
    )
    layer_controller.register(
        LayerId.ROBOT_SKELETON,
        available=True,
        visible=config.show_robot_skeleton,
        callback=_set_all_robot_skeletons,
    )
    layer_controller.register(
        LayerId.HUMAN_HANDS,
        available=human_hands is not None,
        visible=config.show_human_hands,
        callback=lambda visible: human_hands.set_enabled(visible) if human_hands is not None else None,
        unavailable_reason="The source joint set has no registered finger chains.",
    )
    layer_controller.register(
        LayerId.OBJECT_KEYPOINTS,
        available=any(scene.object_keypoints_overlay is not None for scene in scenes),
        visible=config.show_object_keypoints,
        callback=lambda visible: [scene.set_object_keypoints_enabled(visible) for scene in scenes],
        unavailable_reason="No input contains saved demonstration/target object samples.",
    )
    layer_controller.register(
        LayerId.RETARGETING_POINT_CLOUDS,
        available=any(scene.point_cloud_overlay is not None for scene in scenes),
        visible=config.show_point_clouds,
        callback=lambda visible: [scene.set_point_clouds_enabled(visible) for scene in scenes],
        unavailable_reason="No input contains saved retargeting point clouds.",
    )
    layer_controller.register(
        LayerId.INTERACTION_MESH,
        available=any(scene.interaction_mesh_overlay is not None for scene in scenes),
        visible=config.show_interaction_mesh,
        callback=lambda visible: [scene.set_interaction_mesh_enabled(visible) for scene in scenes],
        unavailable_reason="No input contains saved interaction vertices and tetrahedra.",
    )
    layer_controller.register(
        LayerId.FOOT_STICKING,
        available=any(scene.result.foot_sticking is not None for scene in scenes),
        visible=config.show_foot_sticking,
        callback=lambda visible: (
            setattr(foot_sticking_status_handle, "visible", bool(visible))
            if foot_sticking_status_handle is not None
            else None
        ),
        unavailable_reason="No input contains saved foot-sticking states.",
    )
    layer_controller.register(
        LayerId.TARGET_ORIENTATION,
        available=human_reference.target_axes is not None,
        visible=config.show_target_orientation_axes,
        callback=human_reference.set_target_visible,
        unavailable_reason="No input contains saved target orientation frames.",
    )
    layer_controller.register(
        LayerId.SOURCE_ORIENTATION,
        available=source_orientation_overlay is not None,
        visible=config.show_source_orientation_axes,
        callback=lambda visible: (
            source_orientation_overlay.set_visible(visible) if source_orientation_overlay is not None else None
        ),
        unavailable_reason=(
            "The reference result does not contain a directly observed source-human orientation subset."
        ),
    )
    layer_controller.register(
        LayerId.ROBOT_ORIENTATION,
        available=any(scene.orientation_overlay is not None for scene in scenes),
        visible=config.show_robot_orientation_axes,
        callback=lambda visible: [scene.set_robot_orientation_enabled(visible) for scene in scenes],
        unavailable_reason="No input contains saved robot orientation frames.",
    )
    for unavailable_layer, reason in (
        (LayerId.JOINT_LABELS, "Joint labels are available for raw-human inputs."),
        (LayerId.BODY_COM, "Body-center trajectories are available for converted motion inputs."),
        (LayerId.BODY_VELOCITY, "Body velocities are available for converted motion inputs."),
    ):
        layer_controller.register(
            unavailable_layer,
            available=False,
            visible=False,
            callback=lambda _visible: None,
            unavailable_reason=reason,
        )

    with tabs.layers:
        layer_controller.add_gui(server.gui)
        if layer_controller.is_available(LayerId.INTERACTION_MESH):
            with server.gui.add_folder("Interaction mesh settings", expand_by_default=False):
                interaction_mode = server.gui.add_dropdown(
                    "Content",
                    ("source", "target", "both"),
                    initial_value=config.interaction_mesh_mode,
                )
                interaction_edges = server.gui.add_dropdown(
                    "Edges",
                    ("cross", "all"),
                    initial_value=config.interaction_mesh_edges,
                )

                @interaction_mode.on_update
                def _(_event) -> None:
                    for scene in scenes:
                        if scene.interaction_mesh_overlay is not None:
                            scene.interaction_mesh_overlay.mode = str(interaction_mode.value)
                    if layer_controller.is_visible(LayerId.INTERACTION_MESH):
                        current_frame = last_frame["value"]
                        current_q = interpolate_qpos(
                            reference.qpos,
                            current_frame,
                            robot_dof,
                            contains_object=driver_contains_object,
                            loop=config.loop,
                        )
                        _render_comparison(current_q, current_frame)

                @interaction_edges.on_update
                def _(_event) -> None:
                    for scene in scenes:
                        if scene.interaction_mesh_overlay is not None:
                            scene.interaction_mesh_overlay.edge_mode = str(interaction_edges.value)
                    if layer_controller.is_visible(LayerId.INTERACTION_MESH):
                        current_frame = last_frame["value"]
                        current_q = interpolate_qpos(
                            reference.qpos,
                            current_frame,
                            robot_dof,
                            contains_object=driver_contains_object,
                            loop=config.loop,
                        )
                        _render_comparison(current_q, current_frame)

        if layer_controller.is_available(LayerId.FOOT_STICKING):
            with server.gui.add_folder("Foot sticking status", expand_by_default=True):
                foot_sticking_status_handle = server.gui.add_markdown(
                    _foot_sticking_markdown(0.0),
                    visible=config.show_foot_sticking,
                )

    motion_checkboxes: list[tuple[ComparisonScene, object, object, object]] = []
    if tabs.motions is None:
        raise RuntimeError("Multi-motion viewer did not create a Motions tab")
    with tabs.motions:
        for scene in scenes:
            color_hex = "#" + "".join(f"{channel:02x}" for channel in scene.color)
            with server.gui.add_folder(compact_layer_label(scene.label), expand_by_default=True):
                server.gui.add_markdown(f"Color `{color_hex}`  \n`{scene.result.path}`")
                visible_checkbox = server.gui.add_checkbox("Visible", initial_value=True)
                mesh_checkbox = server.gui.add_checkbox("Meshes", initial_value=config.show_robot_mesh)
                skeleton_checkbox = server.gui.add_checkbox(
                    "Robot skeleton",
                    initial_value=config.show_robot_skeleton,
                )

                def _register_motion_callbacks(
                    scene_ref: ComparisonScene,
                    visible_ref,
                    mesh_ref,
                    skeleton_ref,
                ) -> None:
                    @visible_ref.on_update
                    def _(_event) -> None:
                        if not bool(visible_ref.value):
                            scene_ref.set_enabled(False)
                            return
                        scene_ref.set_robot_mesh_enabled(layer_controller.is_visible(LayerId.ROBOT_MESH))
                        scene_ref.set_object_mesh_enabled(layer_controller.is_visible(LayerId.OBJECT_MESH))
                        scene_ref.set_skeleton_enabled(layer_controller.is_visible(LayerId.ROBOT_SKELETON))
                        scene_ref.set_robot_orientation_enabled(layer_controller.is_visible(LayerId.ROBOT_ORIENTATION))
                        scene_ref.set_object_keypoints_enabled(layer_controller.is_visible(LayerId.OBJECT_KEYPOINTS))
                        scene_ref.set_interaction_mesh_enabled(layer_controller.is_visible(LayerId.INTERACTION_MESH))

                    @mesh_ref.on_update
                    def _(_event) -> None:
                        scene_ref.set_mesh_enabled(bool(mesh_ref.value))

                    @skeleton_ref.on_update
                    def _(_event) -> None:
                        scene_ref.set_skeleton_enabled(bool(skeleton_ref.value))

                _register_motion_callbacks(
                    scene,
                    visible_checkbox,
                    mesh_checkbox,
                    skeleton_checkbox,
                )
                motion_checkboxes.append((scene, visible_checkbox, mesh_checkbox, skeleton_checkbox))

    with tabs.style:
        server.gui.add_number(
            "Motion spacing",
            initial_value=float(config.x_offset),
            min=0.0,
            max=5.0,
            step=0.05,
            disabled=True,
            hint="Set with --x-offset before creating the scene.",
        )
        server.gui.add_number(
            "Robot mesh opacity",
            initial_value=float(config.robot_mesh_opacity),
            min=0.0,
            max=1.0,
            step=0.05,
            disabled=True,
        )
        server.gui.add_number(
            "Object mesh opacity",
            initial_value=float(config.object_mesh_opacity),
            min=0.0,
            max=1.0,
            step=0.05,
            disabled=True,
        )

    layer_controller.register_shortcuts(server)
    server._holosoma_layer_controller = layer_controller

    driver = scenes[0]
    with tabs.playback:
        create_motion_control_sliders(
            server=server,
            viser_robot=driver.robot,
            robot_base_frame=driver.robot_root,
            motion_sequence=reference.qpos,
            robot_dof=robot_dof,
            viser_object=driver.object_visual,
            object_base_frame=driver.object_root,
            contains_object_in_qpos=driver_contains_object,
            initial_fps=round(config.fps or reference.fps),
            initial_interp_mult=config.visual_fps_multiplier,
            loop=config.loop,
            qpos_to_viser_joint_indices=driver_joint_order_indices,
            on_frame=_render_comparison,
        )

    print(
        f"[multi_viser_player] Loaded {len(results)} synchronized results, "
        f"frames={reference.qpos.shape[0]}, fps={config.fps or reference.fps:.3f}"
    )
    for scene in scenes:
        print(
            f"  robot group {scene.label}: independent mesh/skeleton controls, "
            f"overlaid mapped skeleton + anchored actual link axes, "
            f"rgb={scene.color}, path={scene.result.path}"
        )
        if scene.orientation_overlay is not None:
            if isinstance(
                scene.orientation_overlay,
                SavedOrientationAxesOverlay,
            ):
                selected_names = scene.orientation_overlay.names
            else:
                selected_names = tuple(
                    scene.orientation_overlay.diagnostics.human_joint_names[int(index)]
                    for index in scene.orientation_overlay.joint_indices
                )
            print(f"    orientation arrows: links={selected_names}, robot=opaque RGB arrows")
        else:
            print("    orientation arrows: unavailable in this result")
    print(
        "  human reference: overlaid on the first robot, "
        "charcoal mapped-keypoint skeleton + anchored target RGB link axes"
    )
    print("Open the viewer URL printed above. Close the process (Ctrl+C) to exit.")
    return server
