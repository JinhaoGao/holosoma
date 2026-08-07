# ruff: noqa: CPY001
"""Shared-world Viser playback for saved paired refinement results."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import viser  # type: ignore[import-not-found]
import yourdfpy  # type: ignore[import-untyped]
from viser.extras import ViserUrdf  # type: ignore[import-not-found]

from holosoma_retargeting.config_types.robot import RobotConfig
from holosoma_retargeting.src.viser_utils import (
    build_joint_order_indices,
    create_motion_control_sliders,
)
from holosoma_retargeting.visualization.layers import (
    LayerController,
    LayerId,
    add_visualization_tabs,
)

_ACTOR_COLORS = ((0, 114, 178), (213, 94, 0))
_ORIGINAL_HUMAN_COLORS = ((0, 0, 0), (0, 0, 0))


@dataclass(frozen=True)
class PairedViserConfig:
    """Configuration for one saved paired-result Viser scene."""

    qpos_npz: Path
    port: int = 8080
    fps: float | None = None
    loop: bool = False
    visual_fps_multiplier: int = 2
    show_robot_mesh: bool = True
    show_human_skeleton: bool = True
    show_original_human_skeleton: bool = True
    show_robot_skeleton: bool = True
    show_interaction_mesh: bool = True
    original_source_fbx: Path | None = None
    original_start_frame: int = 0
    interaction_mesh_mode: str = "both"
    robot_mesh_opacity: float = 0.55
    skeleton_point_size: float = 0.015
    skeleton_line_width: float = 2.5
    interaction_mesh_line_width: float = 1.0
    grid_width: float = 8.0
    grid_height: float = 8.0

    def __post_init__(self) -> None:
        if self.port <= 0:
            raise ValueError("Viser port must be positive")
        if self.fps is not None and self.fps <= 0.0:
            raise ValueError("Viser fps must be positive")
        if self.visual_fps_multiplier < 1:
            raise ValueError("visual_fps_multiplier must be positive")
        if self.original_start_frame < 0:
            raise ValueError("original_start_frame must be non-negative")
        if self.interaction_mesh_mode not in {"source", "target", "both"}:
            raise ValueError("interaction_mesh_mode must be source, target, or both")
        if not 0.0 <= self.robot_mesh_opacity <= 1.0:
            raise ValueError("robot_mesh_opacity must be between zero and one")


@dataclass(frozen=True)
class PairedVisualizationActor:
    """Saved actor arrays needed by the paired Viser renderer."""

    name: str
    robot_type: str
    qpos: np.ndarray
    robot_actuated_joint_names: tuple[str, ...]
    mapped_human_joint_names: tuple[str, ...]
    mapped_robot_link_names: tuple[str, ...]
    mapped_human_joints: np.ndarray
    mapped_robot_joints: np.ndarray
    human_joints: np.ndarray | None
    human_joint_names: tuple[str, ...]
    human_joint_parent_indices: np.ndarray | None
    source_result_path: Path | None


@dataclass(frozen=True)
class PairedVisualizationResult:
    """Validated paired artifact consumed by the Viser scene."""

    path: Path
    fps: float
    actor_a: PairedVisualizationActor
    actor_b: PairedVisualizationActor
    interaction_source_vertices: np.ndarray
    interaction_target_vertices: np.ndarray
    interaction_edges: np.ndarray
    interaction_edge_counts: np.ndarray
    source_alignment_mode: str
    source_relative_xy_m: np.ndarray
    source_scene_scale: float
    source_fbx: Path | None


@dataclass(frozen=True)
class OriginalHumanActor:
    """One complete source-FBX actor skeleton in the paired world."""

    name: str
    source_actor: str
    points: np.ndarray
    joint_names: tuple[str, ...]
    joint_parent_indices: np.ndarray


@dataclass(frozen=True)
class OriginalHumanReference:
    """Both raw FBX actors transformed only by one shared translation."""

    source_fbx: Path
    actor_a: OriginalHumanActor
    actor_b: OriginalHumanActor
    native_scale: float
    source_ground_z_m: float


def _scalar(data: np.lib.npyio.NpzFile, name: str):
    if name not in data:
        raise ValueError(f"Paired result is missing required field {name!r}")
    value = np.asarray(data[name])
    if value.ndim != 0:
        raise ValueError(f"Paired result field {name!r} must be scalar")
    return value.item()


def _names(data: np.lib.npyio.NpzFile, name: str) -> tuple[str, ...]:
    if name not in data:
        raise ValueError(f"Paired result is missing required field {name!r}")
    values = tuple(str(value) for value in np.asarray(data[name]).tolist())
    if not values or any(not value for value in values) or len(values) != len(set(values)):
        raise ValueError(f"Paired result field {name!r} must contain unique non-empty names")
    return values


def _optional_path(data: np.lib.npyio.NpzFile, name: str) -> Path | None:
    if name not in data:
        return None
    value = str(_scalar(data, name))
    return Path(value).expanduser().resolve() if value else None


def _actor_from_npz(
    data: np.lib.npyio.NpzFile,
    prefix: str,
    frame_count: int,
) -> PairedVisualizationActor:
    qpos = np.asarray(data[f"{prefix}_qpos"], dtype=np.float64)
    mapped_human_joint_names = _names(data, f"{prefix}_mapped_human_joint_names")
    mapped_robot_link_names = _names(data, f"{prefix}_mapped_robot_link_names")
    mapped_human_joints = np.asarray(data[f"{prefix}_mapped_human_joints"], dtype=np.float32)
    mapped_robot_joints = np.asarray(data[f"{prefix}_mapped_robot_joints"], dtype=np.float32)
    expected_mapped_shape = (frame_count, len(mapped_human_joint_names), 3)
    if qpos.ndim != 2 or qpos.shape[0] != frame_count or not np.all(np.isfinite(qpos)):
        raise ValueError(f"Paired result {prefix}_qpos must be finite with {frame_count} frames")
    if mapped_human_joints.shape != expected_mapped_shape or mapped_robot_joints.shape != expected_mapped_shape:
        raise ValueError(f"Paired result {prefix} mapped skeleton shapes must be {expected_mapped_shape}")
    if not np.all(np.isfinite(mapped_human_joints)) or not np.all(np.isfinite(mapped_robot_joints)):
        raise ValueError(f"Paired result {prefix} mapped skeletons must be finite")
    if len(mapped_robot_link_names) != len(mapped_human_joint_names):
        raise ValueError(f"Paired result {prefix} mapped human and robot name counts differ")

    robot_type = str(_scalar(data, f"{prefix}_robot_type"))
    actuated_joint_names = _names(data, f"{prefix}_robot_actuated_joint_names")
    robot_config = RobotConfig(robot_type=robot_type)
    if len(actuated_joint_names) != robot_config.ROBOT_DOF or qpos.shape[1] != 7 + robot_config.ROBOT_DOF:
        raise ValueError(f"Paired result {prefix} qpos and joint names do not match robot type {robot_type!r}")

    human_joints = None
    human_joint_names: tuple[str, ...] = ()
    human_joint_parent_indices = None
    human_key = f"{prefix}_human_joints"
    if human_key in data:
        human_joints = np.asarray(data[human_key], dtype=np.float32)
        human_joint_names = _names(data, f"{prefix}_human_joint_names")
        human_joint_parent_indices = np.asarray(
            data[f"{prefix}_human_joint_parent_indices"],
            dtype=np.int32,
        )
        if human_joints.shape != (frame_count, len(human_joint_names), 3):
            raise ValueError(f"Paired result {human_key} shape does not match its names")
        if not np.all(np.isfinite(human_joints)):
            raise ValueError(f"Paired result {human_key} must be finite")
        if human_joint_parent_indices.shape != (len(human_joint_names),):
            raise ValueError(f"Paired result {prefix} human parent shape does not match its names")
    return PairedVisualizationActor(
        name=str(_scalar(data, f"{prefix}_name")),
        robot_type=robot_type,
        qpos=qpos,
        robot_actuated_joint_names=actuated_joint_names,
        mapped_human_joint_names=mapped_human_joint_names,
        mapped_robot_link_names=mapped_robot_link_names,
        mapped_human_joints=mapped_human_joints,
        mapped_robot_joints=mapped_robot_joints,
        human_joints=human_joints,
        human_joint_names=human_joint_names,
        human_joint_parent_indices=human_joint_parent_indices,
        source_result_path=_optional_path(data, f"{prefix}_source_path"),
    )


def _validate_interaction_vertices(
    frame_count: int,
    source_vertices: np.ndarray,
    target_vertices: np.ndarray,
) -> None:
    expected_shape = (frame_count, source_vertices.shape[1], 3)
    if source_vertices.shape != expected_shape or target_vertices.shape != expected_shape:
        raise ValueError("Paired interaction source and target vertices must have matching frame shapes")
    if not np.all(np.isfinite(source_vertices)) or not np.all(np.isfinite(target_vertices)):
        raise ValueError("Paired interaction source and target vertices must be finite")


def _validate_interaction_edges(
    frame_count: int,
    vertex_count: int,
    edges: np.ndarray,
    edge_counts: np.ndarray,
) -> None:
    if edges.ndim != 3 or edges.shape[0] != frame_count or edges.shape[2] != 2:
        raise ValueError("Paired interaction edges must have shape (frames, edges, 2)")
    if edge_counts.shape != (frame_count,):
        raise ValueError("Paired interaction edge counts must contain one value per frame")
    if np.any(edge_counts < 0) or np.any(edge_counts > edges.shape[1]):
        raise ValueError("Paired interaction edge counts exceed the packed edge width")
    for frame_index, edge_count in enumerate(edge_counts):
        frame_edges = edges[frame_index, : int(edge_count)]
        if np.any(frame_edges < 0) or np.any(frame_edges >= vertex_count):
            raise ValueError(f"Paired interaction edges contain invalid vertex indices at frame {frame_index}")


def _validate_source_alignment(source_relative_xy_m: np.ndarray, source_scene_scale: float) -> None:
    if source_relative_xy_m.shape != (2,) or not np.all(np.isfinite(source_relative_xy_m)):
        raise ValueError("Paired source relative XY must contain two finite values")
    if not np.isfinite(source_scene_scale) or source_scene_scale <= 0.0:
        raise ValueError("Paired source scene scale must be positive and finite")


def load_paired_visualization_result(path: str | Path) -> PairedVisualizationResult:
    """Load the self-contained shared-world arrays from a paired NPZ."""
    result_path = Path(path).expanduser().resolve()
    if not result_path.is_file():
        raise FileNotFoundError(f"Paired result does not exist: {result_path}")
    with np.load(result_path, allow_pickle=False) as data:
        if str(_scalar(data, "run_kind")) != "paired_refinement":
            raise ValueError(f"Result is not a paired refinement artifact: {result_path}")
        fps = float(_scalar(data, "fps"))
        frame_count = int(np.asarray(data["actor_a_qpos"]).shape[0])
        actor_a = _actor_from_npz(data, "actor_a", frame_count)
        actor_b = _actor_from_npz(data, "actor_b", frame_count)
        source_vertices = np.asarray(data["interaction_source_vertices_w"], dtype=np.float32)
        target_vertices = np.asarray(data["interaction_target_vertices_w"], dtype=np.float32)
        edges = np.asarray(data["interaction_edges"], dtype=np.int32)
        edge_counts = np.asarray(data["interaction_edge_counts"], dtype=np.int32)
        source_relative_xy_m = np.asarray(data["source_relative_xy_m"], dtype=np.float64)
        source_scene_scale = float(_scalar(data, "source_scene_scale"))
        source_alignment_mode = str(_scalar(data, "source_alignment_mode"))
        source_fbx = _optional_path(data, "source_fbx")
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError("Paired result fps must be positive and finite")
    if actor_a.qpos.shape[0] != actor_b.qpos.shape[0]:
        raise ValueError("Paired visualization actors have different frame counts")
    _validate_interaction_vertices(frame_count, source_vertices, target_vertices)
    _validate_interaction_edges(frame_count, source_vertices.shape[1], edges, edge_counts)
    _validate_source_alignment(source_relative_xy_m, source_scene_scale)
    return PairedVisualizationResult(
        path=result_path,
        fps=fps,
        actor_a=actor_a,
        actor_b=actor_b,
        interaction_source_vertices=source_vertices,
        interaction_target_vertices=target_vertices,
        interaction_edges=edges,
        interaction_edge_counts=edge_counts,
        source_alignment_mode=source_alignment_mode,
        source_relative_xy_m=source_relative_xy_m,
        source_scene_scale=source_scene_scale,
        source_fbx=source_fbx,
    )


@dataclass(frozen=True)
class _RawOriginalActor:
    source_actor: str
    source_fbx: Path
    source_motion_path: Path
    points: np.ndarray
    joint_names: tuple[str, ...]
    joint_parent_indices: np.ndarray
    source_xy_origin_m: np.ndarray
    fps: float


def _load_raw_original_actor(actor: PairedVisualizationActor) -> _RawOriginalActor:
    result_path = actor.source_result_path
    if result_path is None or not result_path.is_file():
        raise FileNotFoundError(f"Actor {actor.name!r} does not reference an existing single-robot result")
    with np.load(result_path, allow_pickle=False) as result_data:
        source_motion_path = _optional_path(result_data, "source_path")
    if source_motion_path is None or not source_motion_path.is_file():
        raise FileNotFoundError(
            f"Actor {actor.name!r} single-robot result does not reference an existing converted FBX motion"
        )
    with np.load(source_motion_path, allow_pickle=False) as source_data:
        required = (
            "source_fbx",
            "source_actor",
            "source_xy_origin_m",
            "source_skeleton_positions",
            "source_skeleton_joint_names",
            "source_skeleton_parent_indices",
            "fps",
        )
        missing = [name for name in required if name not in source_data]
        if missing:
            raise ValueError(f"Converted FBX actor {source_motion_path} is missing fields: {missing}")
        source_fbx = _optional_path(source_data, "source_fbx")
        assert source_fbx is not None
        source_actor = str(_scalar(source_data, "source_actor"))
        points = np.asarray(source_data["source_skeleton_positions"], dtype=np.float64)
        joint_names = _names(source_data, "source_skeleton_joint_names")
        parent_indices = np.asarray(
            source_data["source_skeleton_parent_indices"],
            dtype=np.int32,
        ).copy()
        source_xy_origin_m = np.asarray(source_data["source_xy_origin_m"], dtype=np.float64).copy()
        fps = float(_scalar(source_data, "fps"))
    expected_shape = (points.shape[0], len(joint_names), 3)
    if points.ndim != 3 or points.shape != expected_shape or points.shape[0] == 0:
        raise ValueError(f"Converted FBX actor {source_motion_path} has an invalid skeleton shape")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"Converted FBX actor {source_motion_path} contains non-finite skeleton points")
    if parent_indices.shape != (len(joint_names),):
        raise ValueError(f"Converted FBX actor {source_motion_path} has invalid skeleton parents")
    if np.any(parent_indices < -1) or np.any(parent_indices >= len(joint_names)):
        raise ValueError(f"Converted FBX actor {source_motion_path} has out-of-range skeleton parents")
    if source_xy_origin_m.shape != (2,) or not np.all(np.isfinite(source_xy_origin_m)):
        raise ValueError(f"Converted FBX actor {source_motion_path} has an invalid source XY origin")
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"Converted FBX actor {source_motion_path} has an invalid fps")
    return _RawOriginalActor(
        source_actor=source_actor,
        source_fbx=source_fbx,
        source_motion_path=source_motion_path,
        points=points,
        joint_names=joint_names,
        joint_parent_indices=parent_indices,
        source_xy_origin_m=source_xy_origin_m,
        fps=fps,
    )


def load_original_human_reference(
    result: PairedVisualizationResult,
    config: PairedViserConfig,
) -> OriginalHumanReference | None:
    """Load both raw FBX skeletons and restore their common source scene."""
    expected_source_fbx = (
        config.original_source_fbx.expanduser().resolve()
        if config.original_source_fbx is not None
        else result.source_fbx
    )
    if expected_source_fbx is None:
        return None
    if not expected_source_fbx.is_file():
        if config.original_source_fbx is not None:
            raise FileNotFoundError(f"Original source FBX does not exist: {expected_source_fbx}")
        return None
    raw_a = _load_raw_original_actor(result.actor_a)
    raw_b = _load_raw_original_actor(result.actor_b)
    for raw_actor in (raw_a, raw_b):
        if raw_actor.source_fbx != expected_source_fbx:
            raise ValueError(
                f"Actor {raw_actor.source_actor!r} comes from {raw_actor.source_fbx}, "
                f"not requested FBX {expected_source_fbx}"
            )
        if not np.isclose(raw_actor.fps, result.fps, rtol=0.0, atol=1e-6):
            raise ValueError(
                f"Original actor {raw_actor.source_actor!r} fps {raw_actor.fps} "
                f"does not match paired result fps {result.fps}"
            )
    if raw_a.source_actor == raw_b.source_actor:
        raise ValueError("Original FBX reference requires two distinct source actors")
    frame_count = result.actor_a.qpos.shape[0]
    frame_start = config.original_start_frame
    frame_stop = frame_start + frame_count
    if frame_stop > raw_a.points.shape[0] or frame_stop > raw_b.points.shape[0]:
        raise ValueError(
            f"Original FBX skeletons do not cover the requested paired playback window [{frame_start}, {frame_stop})"
        )
    source_ground_z_m = float(min(np.min(raw_a.points[..., 2]), np.min(raw_b.points[..., 2])))
    anchor = np.asarray(
        (raw_a.source_xy_origin_m[0], raw_a.source_xy_origin_m[1], source_ground_z_m),
        dtype=np.float64,
    )

    def _shared_world_points(raw_actor: _RawOriginalActor) -> np.ndarray:
        points = raw_actor.points[frame_start:frame_stop].copy()
        points[..., :2] += raw_actor.source_xy_origin_m
        return points - anchor

    return OriginalHumanReference(
        source_fbx=expected_source_fbx,
        actor_a=OriginalHumanActor(
            name=result.actor_a.name,
            source_actor=raw_a.source_actor,
            points=_shared_world_points(raw_a),
            joint_names=raw_a.joint_names,
            joint_parent_indices=raw_a.joint_parent_indices,
        ),
        actor_b=OriginalHumanActor(
            name=result.actor_b.name,
            source_actor=raw_b.source_actor,
            points=_shared_world_points(raw_b),
            joint_names=raw_b.joint_names,
            joint_parent_indices=raw_b.joint_parent_indices,
        ),
        native_scale=1.0,
        source_ground_z_m=source_ground_z_m,
    )


def _interpolate_points(points: np.ndarray, frame_float: float, *, loop: bool) -> np.ndarray:
    sample = frame_float % points.shape[0] if loop else np.clip(frame_float, 0.0, points.shape[0] - 1)
    frame_a = int(np.floor(sample))
    frame_b = (frame_a + 1) % points.shape[0] if loop else min(frame_a + 1, points.shape[0] - 1)
    fraction = float(sample - frame_a)
    return (1.0 - fraction) * points[frame_a] + fraction * points[frame_b]


def _slerp(quaternion_a: np.ndarray, quaternion_b: np.ndarray, fraction: float) -> np.ndarray:
    q_a = quaternion_a / max(float(np.linalg.norm(quaternion_a)), 1e-12)
    q_b = quaternion_b / max(float(np.linalg.norm(quaternion_b)), 1e-12)
    dot = float(np.dot(q_a, q_b))
    if dot < 0.0:
        q_b = -q_b
        dot = -dot
    if dot > 0.9995:
        result = q_a + fraction * (q_b - q_a)
        return result / max(float(np.linalg.norm(result)), 1e-12)
    theta = float(np.arccos(np.clip(dot, -1.0, 1.0)))
    return (np.sin((1.0 - fraction) * theta) * q_a + np.sin(fraction * theta) * q_b) / np.sin(theta)


def _interpolate_qpos(qpos: np.ndarray, frame_float: float, *, loop: bool) -> np.ndarray:
    sample = frame_float % qpos.shape[0] if loop else np.clip(frame_float, 0.0, qpos.shape[0] - 1)
    frame_a = int(np.floor(sample))
    frame_b = (frame_a + 1) % qpos.shape[0] if loop else min(frame_a + 1, qpos.shape[0] - 1)
    fraction = float(sample - frame_a)
    result = (1.0 - fraction) * qpos[frame_a] + fraction * qpos[frame_b]
    result[3:7] = _slerp(qpos[frame_a, 3:7], qpos[frame_b, 3:7], fraction)
    return result


def _parent_edges(parent_indices: np.ndarray) -> np.ndarray:
    return np.asarray(
        [(int(parent), child) for child, parent in enumerate(parent_indices) if parent >= 0],
        dtype=np.int32,
    ).reshape(-1, 2)


def _full_skeleton_edges(actor: PairedVisualizationActor) -> np.ndarray:
    if actor.human_joint_parent_indices is None:
        return np.asarray(
            [(0, index) for index in range(1, len(actor.mapped_human_joint_names))],
            dtype=np.int32,
        ).reshape(-1, 2)
    return _parent_edges(actor.human_joint_parent_indices)


def _mapped_skeleton_edges(actor: PairedVisualizationActor) -> np.ndarray:
    if actor.human_joint_parent_indices is None or not actor.human_joint_names:
        return np.asarray(
            [(0, index) for index in range(1, len(actor.mapped_human_joint_names))],
            dtype=np.int32,
        ).reshape(-1, 2)
    full_index = {name: index for index, name in enumerate(actor.human_joint_names)}
    missing_names = [name for name in actor.mapped_human_joint_names if name not in full_index]
    if missing_names:
        raise ValueError(f"Actor {actor.name!r} mapped joints are absent from its full skeleton: {missing_names}")
    mapped_full_indices = {full_index[name]: index for index, name in enumerate(actor.mapped_human_joint_names)}
    edges: list[tuple[int, int]] = []
    for full_child, mapped_child in mapped_full_indices.items():
        parent = int(actor.human_joint_parent_indices[full_child])
        while parent >= 0 and parent not in mapped_full_indices:
            parent = int(actor.human_joint_parent_indices[parent])
        if parent in mapped_full_indices:
            edges.append((mapped_full_indices[parent], mapped_child))
    return np.asarray(edges, dtype=np.int32).reshape(-1, 2)


@dataclass
class _SkeletonOverlay:
    points: np.ndarray
    edges: np.ndarray
    joints_handle: object
    bones_handle: object
    loop: bool

    def set_visible(self, visible: bool) -> None:
        self.joints_handle.visible = bool(visible)
        self.bones_handle.visible = bool(visible)

    def update(self, frame_float: float) -> None:
        frame_points = _interpolate_points(self.points, frame_float, loop=self.loop)
        self.joints_handle.points = np.asarray(frame_points, dtype=np.float32)
        self.bones_handle.points = np.asarray(frame_points[self.edges], dtype=np.float32)


class _InteractionOverlay:
    def __init__(self, server, result: PairedVisualizationResult, config: PairedViserConfig) -> None:
        self.result = result
        self.mode = config.interaction_mesh_mode
        self.loop = config.loop
        first_edges = result.interaction_edges[0, : result.interaction_edge_counts[0]]
        self.source_handle = server.scene.add_line_segments(
            "/paired/interaction/source",
            points=result.interaction_source_vertices[0, first_edges],
            colors=np.asarray((255, 140, 0), dtype=np.uint8),
            line_width=config.interaction_mesh_line_width,
            visible=config.show_interaction_mesh and self.mode in {"source", "both"},
        )
        self.target_handle = server.scene.add_line_segments(
            "/paired/interaction/target",
            points=result.interaction_target_vertices[0, first_edges],
            colors=np.asarray((0, 220, 255), dtype=np.uint8),
            line_width=config.interaction_mesh_line_width,
            visible=config.show_interaction_mesh and self.mode in {"target", "both"},
        )

    def set_visible(self, visible: bool) -> None:
        self.source_handle.visible = bool(visible) and self.mode in {"source", "both"}
        self.target_handle.visible = bool(visible) and self.mode in {"target", "both"}

    def update(self, frame_float: float) -> None:
        if self.loop:
            frame_index = round(float(frame_float)) % self.result.interaction_edges.shape[0]
        else:
            frame_index = int(np.clip(round(float(frame_float)), 0, self.result.interaction_edges.shape[0] - 1))
        edge_count = int(self.result.interaction_edge_counts[frame_index])
        edges = self.result.interaction_edges[frame_index, :edge_count]
        self.source_handle.points = self.result.interaction_source_vertices[frame_index, edges]
        self.target_handle.points = self.result.interaction_target_vertices[frame_index, edges]


@dataclass
class _ActorViserScene:
    actor: PairedVisualizationActor
    robot: ViserUrdf
    robot_root: object
    joint_order_indices: np.ndarray
    human_skeleton: _SkeletonOverlay
    robot_skeleton: _SkeletonOverlay


@dataclass(frozen=True)
class _PairedSceneContent:
    actors: tuple[_ActorViserScene, _ActorViserScene]
    original_skeletons: tuple[_SkeletonOverlay, ...]
    original_reference: OriginalHumanReference | None
    original_unavailable_reason: str
    interaction: _InteractionOverlay


def _mesh_color(color: tuple[int, int, int], opacity: float) -> tuple[float, float, float, float]:
    pale = tuple(round(0.25 * channel + 0.75 * 235) for channel in color)
    return pale[0] / 255.0, pale[1] / 255.0, pale[2] / 255.0, opacity


def _make_skeleton_overlay(
    server,
    *,
    namespace: str,
    points: np.ndarray,
    edges: np.ndarray,
    color: tuple[int, int, int],
    config: PairedViserConfig,
    visible: bool,
) -> _SkeletonOverlay:
    joints = server.scene.add_point_cloud(
        f"{namespace}/joints",
        points=points[0],
        colors=np.tile(np.asarray(color, dtype=np.uint8), (points.shape[1], 1)),
        point_size=config.skeleton_point_size,
        point_shape="circle",
        visible=visible,
    )
    bones = server.scene.add_line_segments(
        f"{namespace}/bones",
        points=points[0, edges],
        colors=np.asarray(color, dtype=np.uint8),
        line_width=config.skeleton_line_width,
        visible=visible,
    )
    return _SkeletonOverlay(points=points, edges=edges, joints_handle=joints, bones_handle=bones, loop=config.loop)


def _make_actor_scene(
    server,
    actor: PairedVisualizationActor,
    actor_index: int,
    config: PairedViserConfig,
) -> _ActorViserScene:
    color = _ACTOR_COLORS[actor_index]
    namespace = f"/paired/{actor_index:02d}_{actor.name}"
    robot_config = RobotConfig(robot_type=actor.robot_type)
    robot_urdf_path = Path(__file__).resolve().parents[1] / robot_config.ROBOT_URDF_FILE
    robot_urdf = yourdfpy.URDF.load(str(robot_urdf_path), load_meshes=True, build_scene_graph=True)
    robot_root = server.scene.add_frame(f"{namespace}/robot", show_axes=False)
    robot = ViserUrdf(
        server,
        urdf_or_path=robot_urdf,
        root_node_name=f"{namespace}/robot",
        mesh_color_override=_mesh_color(color, config.robot_mesh_opacity),
    )
    viser_joint_names = tuple(robot.get_actuated_joint_limits().keys())
    joint_order_indices = build_joint_order_indices(actor.robot_actuated_joint_names, viser_joint_names)
    if actor.qpos.shape[1] != 7 + len(actor.robot_actuated_joint_names):
        raise ValueError(f"Actor {actor.name!r} qpos width does not match its saved joint count")

    if actor.human_joints is not None:
        human_points = actor.human_joints
        human_edges = _full_skeleton_edges(actor)
    else:
        human_points = actor.mapped_human_joints
        human_edges = _mapped_skeleton_edges(actor)
    human_skeleton = _make_skeleton_overlay(
        server,
        namespace=f"{namespace}/human",
        points=human_points,
        edges=human_edges,
        color=tuple(round(0.55 * channel + 0.45 * 255) for channel in color),
        config=config,
        visible=config.show_human_skeleton,
    )
    robot_skeleton = _make_skeleton_overlay(
        server,
        namespace=f"{namespace}/robot_skeleton",
        points=actor.mapped_robot_joints,
        edges=_mapped_skeleton_edges(actor),
        color=color,
        config=config,
        visible=config.show_robot_skeleton,
    )
    robot.show_visual = config.show_robot_mesh
    return _ActorViserScene(
        actor=actor,
        robot=robot,
        robot_root=robot_root,
        joint_order_indices=joint_order_indices,
        human_skeleton=human_skeleton,
        robot_skeleton=robot_skeleton,
    )


def _robot_joint_values(scene: _ActorViserScene, qpos: np.ndarray) -> np.ndarray:
    source_values = qpos[7 : 7 + len(scene.actor.robot_actuated_joint_names)]
    return source_values[scene.joint_order_indices]


def _optional_original_reference(
    result: PairedVisualizationResult,
    config: PairedViserConfig,
) -> tuple[OriginalHumanReference | None, str]:
    unavailable_reason = "The paired artifact does not reference a recoverable source FBX."
    try:
        return load_original_human_reference(result, config), unavailable_reason
    except FileNotFoundError as exc:
        if config.original_source_fbx is not None:
            raise
        return None, str(exc)


def _make_original_skeletons(
    server,
    original_reference: OriginalHumanReference | None,
    config: PairedViserConfig,
) -> tuple[_SkeletonOverlay, ...]:
    if original_reference is None:
        return ()
    actors = (original_reference.actor_a, original_reference.actor_b)
    return tuple(
        _make_skeleton_overlay(
            server,
            namespace=f"/paired/original_fbx/{actor_index:02d}_{actor.name}",
            points=actor.points,
            edges=_parent_edges(actor.joint_parent_indices),
            color=_ORIGINAL_HUMAN_COLORS[actor_index],
            config=config,
            visible=config.show_original_human_skeleton,
        )
        for actor_index, actor in enumerate(actors)
    )


def _configure_layers(
    server,
    tabs,
    content: _PairedSceneContent,
    config: PairedViserConfig,
) -> None:
    layer_controller = LayerController()
    layer_controller.register(
        LayerId.ROBOT_MESH,
        available=True,
        visible=config.show_robot_mesh,
        callback=lambda visible: [setattr(scene.robot, "show_visual", bool(visible)) for scene in content.actors],
    )
    layer_controller.register(
        LayerId.HUMAN_SKELETON,
        available=True,
        visible=config.show_human_skeleton,
        callback=lambda visible: [scene.human_skeleton.set_visible(visible) for scene in content.actors],
    )
    layer_controller.register(
        LayerId.ORIGINAL_HUMAN_SKELETON,
        available=content.original_reference is not None,
        visible=config.show_original_human_skeleton,
        callback=lambda visible: [skeleton.set_visible(visible) for skeleton in content.original_skeletons],
        unavailable_reason=content.original_unavailable_reason,
    )
    layer_controller.register(
        LayerId.ROBOT_SKELETON,
        available=True,
        visible=config.show_robot_skeleton,
        callback=lambda visible: [scene.robot_skeleton.set_visible(visible) for scene in content.actors],
    )
    layer_controller.register(
        LayerId.INTERACTION_MESH,
        available=True,
        visible=config.show_interaction_mesh,
        callback=content.interaction.set_visible,
    )
    with tabs.layers:
        layer_controller.add_gui(server.gui)
    layer_controller.register_shortcuts(server)
    server._holosoma_layer_controller = layer_controller


def _populate_pair_tabs(
    server,
    tabs,
    content: _PairedSceneContent,
    result: PairedVisualizationResult,
    config: PairedViserConfig,
) -> None:
    if tabs.motions is None:
        raise RuntimeError("Paired Viser scene did not create a Motions tab")
    with tabs.motions:
        for scene, color in zip(content.actors, _ACTOR_COLORS, strict=True):
            with server.gui.add_folder(scene.actor.name, expand_by_default=True):
                server.gui.add_markdown(
                    f"Robot `{scene.actor.robot_type}`  \nColor `#{color[0]:02x}{color[1]:02x}{color[2]:02x}`"
                )
        server.gui.add_markdown(
            f"Source alignment `{result.source_alignment_mode}`  \n"
            f"Source relative XY `{result.source_relative_xy_m.tolist()}` m  \n"
            f"Scene scale `{result.source_scene_scale:.6f}`"
        )
        if content.original_reference is not None:
            original_reference = content.original_reference
            server.gui.add_markdown(
                f"Original FBX `{original_reference.source_fbx}`  \n"
                f"Actors `{original_reference.actor_a.source_actor}` and "
                f"`{original_reference.actor_b.source_actor}`  \n"
                f"Original scale `{original_reference.native_scale:.1f}` (unscaled metres)  \n"
                f"Shared source ground `{original_reference.source_ground_z_m:.6f}` m"
            )
    with tabs.style:
        server.gui.add_number(
            "Robot mesh opacity",
            initial_value=config.robot_mesh_opacity,
            min=0.0,
            max=1.0,
            step=0.05,
            disabled=True,
        )


def make_paired_result_player(config: PairedViserConfig):
    """Build a synchronized shared-world Viser scene for a paired result."""
    result = load_paired_visualization_result(config.qpos_npz)
    original_reference, original_unavailable_reason = _optional_original_reference(result, config)
    server = viser.ViserServer(port=config.port)
    tabs = add_visualization_tabs(server.gui, include_motions=True)
    server.scene.add_grid(
        "/grid",
        width=config.grid_width,
        height=config.grid_height,
        position=(0.0, 0.0, 0.0),
    )
    scenes = (
        _make_actor_scene(server, result.actor_a, 0, config),
        _make_actor_scene(server, result.actor_b, 1, config),
    )
    original_skeletons = _make_original_skeletons(server, original_reference, config)
    interaction = _InteractionOverlay(server, result, config)
    content = _PairedSceneContent(
        actors=scenes,
        original_skeletons=original_skeletons,
        original_reference=original_reference,
        original_unavailable_reason=original_unavailable_reason,
        interaction=interaction,
    )
    _configure_layers(server, tabs, content, config)
    _populate_pair_tabs(server, tabs, content, result, config)

    def _render_pair(_driver_qpos: np.ndarray, frame_float: float) -> None:
        for actor_index, scene in enumerate(scenes):
            qpos = (
                _driver_qpos if actor_index == 0 else _interpolate_qpos(scene.actor.qpos, frame_float, loop=config.loop)
            )
            if actor_index != 0:
                scene.robot.update_cfg(_robot_joint_values(scene, qpos))
            scene.robot_root.position = qpos[:3]
            scene.robot_root.wxyz = qpos[3:7]
            scene.human_skeleton.update(frame_float)
            scene.robot_skeleton.update(frame_float)
        for skeleton in original_skeletons:
            skeleton.update(frame_float)
        interaction.update(frame_float)

    driver = scenes[0]
    with tabs.playback:
        create_motion_control_sliders(
            server=server,
            viser_robot=driver.robot,
            robot_base_frame=driver.robot_root,
            motion_sequence=result.actor_a.qpos,
            robot_dof=len(driver.actor.robot_actuated_joint_names),
            contains_object_in_qpos=False,
            initial_fps=round(config.fps or result.fps),
            initial_interp_mult=config.visual_fps_multiplier,
            loop=config.loop,
            qpos_to_viser_joint_indices=driver.joint_order_indices,
            on_frame=_render_pair,
        )
    print(
        f"[paired_viser_player] Loaded {result.path}, frames={result.actor_a.qpos.shape[0]}, "
        f"fps={config.fps or result.fps:.3f}, alignment={result.source_alignment_mode}"
    )
    print("Open the Viser URL printed above. Close the process with Ctrl+C to exit.")
    return server


def run_paired_result_player(config: PairedViserConfig) -> None:
    """Build the paired Viser scene and keep its server alive."""
    make_paired_result_player(config)
    while True:
        time.sleep(1.0)
