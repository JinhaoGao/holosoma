# ruff: noqa: CPY001
"""Load and save the stable NPZ contract for paired refinement."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from holosoma_retargeting.config_types.robot import RobotConfig
from holosoma_retargeting.paired_retargeting.config import PairedRefinementConfig

_EXPECTED_QPOS_LAYOUT = "mujoco_free_root_xyz_wxyz_then_actuated_then_optional_object_free_joint"
_EXPECTED_QUATERNION_CONVENTION = "wxyz"
_EXPECTED_WORLD_COORDINATE_SYSTEM = "right_handed_z_up"


def _scalar(data: Mapping[str, np.ndarray], name: str) -> Any:
    if name not in data:
        raise ValueError(f"Retargeting result is missing required field {name!r}")
    value = np.asarray(data[name])
    if value.ndim != 0:
        raise ValueError(f"Retargeting result field {name!r} must be scalar")
    return value.item()


def _string_tuple(data: Mapping[str, np.ndarray], name: str) -> tuple[str, ...]:
    if name not in data:
        raise ValueError(f"Retargeting result is missing required field {name!r}")
    value = np.asarray(data[name])
    if value.ndim != 1:
        raise ValueError(f"Retargeting result field {name!r} must be one-dimensional")
    names = tuple(str(item) for item in value.tolist())
    if not names or any(not item for item in names):
        raise ValueError(f"Retargeting result field {name!r} must contain non-empty names")
    if len(names) != len(set(names)):
        raise ValueError(f"Retargeting result field {name!r} must contain unique names")
    return names


def _finite_array(data: Mapping[str, np.ndarray], name: str, ndim: int) -> np.ndarray:
    if name not in data:
        raise ValueError(f"Retargeting result is missing required field {name!r}")
    value = np.asarray(data[name], dtype=np.float64)
    if value.ndim != ndim:
        raise ValueError(f"Retargeting result field {name!r} must have {ndim} dimensions")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"Retargeting result field {name!r} contains non-finite values")
    return value.copy()


@dataclass(frozen=True)
class _HumanSkeleton:
    joints: np.ndarray | None
    joint_names: tuple[str, ...]
    parent_indices: np.ndarray | None


@dataclass(frozen=True)
class _SourceSceneMetadata:
    human_position_scale: float | None
    source_motion_path: Path | None
    source_fbx: Path | None
    source_actor: str | None
    source_xy_origin_m: np.ndarray | None


def _validated_qpos(
    data: Mapping[str, np.ndarray],
    robot_type: str,
) -> tuple[np.ndarray, RobotConfig]:
    robot_config = RobotConfig(robot_type=robot_type)
    qpos = _finite_array(data, "qpos", 2)
    expected_width = 7 + robot_config.ROBOT_DOF
    if qpos.shape[1] != expected_width:
        raise ValueError(f"qpos width for {robot_type!r} must be {expected_width}, got {qpos.shape[1]}")
    if qpos.shape[0] == 0:
        raise ValueError("Retargeting result must contain at least one frame")
    root_quaternion_norms = np.linalg.norm(qpos[:, 3:7], axis=1)
    invalid_norm = (root_quaternion_norms <= 1e-8) | (np.abs(root_quaternion_norms - 1.0) > 1e-3)
    if np.any(invalid_norm):
        raise ValueError("Retargeting result root quaternions must be non-zero and unit length")
    return qpos, robot_config


def _validated_mapping(
    data: Mapping[str, np.ndarray],
    qpos: np.ndarray,
    robot_dof: int,
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    mapped_human_joints = _finite_array(data, "mapped_human_joints", 3)
    mapped_human_names = _string_tuple(data, "mapped_human_joint_names")
    mapped_robot_names = _string_tuple(data, "mapped_robot_link_names")
    actuated_joint_names = _string_tuple(data, "robot_actuated_joint_names")
    if mapped_human_joints.shape != (qpos.shape[0], len(mapped_human_names), 3):
        raise ValueError("mapped_human_joints shape does not match qpos and mapped joint names")
    if len(mapped_robot_names) != len(mapped_human_names):
        raise ValueError("Mapped human joint and robot link counts must match")
    if len(actuated_joint_names) != robot_dof:
        raise ValueError("robot_actuated_joint_names count does not match the configured robot DOF")
    return mapped_human_joints, mapped_human_names, mapped_robot_names, actuated_joint_names


def _validated_motion_contract(data: Mapping[str, np.ndarray]) -> tuple[float, str, str, str]:
    fps = float(_scalar(data, "fps"))
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"fps must be positive and finite, got {fps!r}")
    qpos_layout = str(_scalar(data, "qpos_layout"))
    quaternion_convention = str(_scalar(data, "quaternion_convention"))
    world_coordinate_system = str(_scalar(data, "world_coordinate_system"))
    expected = (
        (qpos_layout, _EXPECTED_QPOS_LAYOUT, "qpos layout"),
        (quaternion_convention, _EXPECTED_QUATERNION_CONVENTION, "quaternion convention"),
        (world_coordinate_system, _EXPECTED_WORLD_COORDINATE_SYSTEM, "world coordinate system"),
    )
    for actual, required, label in expected:
        if actual != required:
            raise ValueError(f"Unsupported {label}: {actual!r}")
    return fps, qpos_layout, quaternion_convention, world_coordinate_system


def _optional_human_skeleton(data: Mapping[str, np.ndarray], frame_count: int) -> _HumanSkeleton:
    if "human_joints" not in data:
        return _HumanSkeleton(None, (), None)
    joints = _finite_array(data, "human_joints", 3)
    joint_names = _string_tuple(data, "human_joint_names")
    if joints.shape != (frame_count, len(joint_names), 3):
        raise ValueError("human_joints shape does not match qpos and human_joint_names")
    if "human_joint_parent_indices" not in data:
        raise ValueError("Retargeting result with human_joints must contain human_joint_parent_indices")
    parent_indices = np.asarray(data["human_joint_parent_indices"], dtype=np.int32).copy()
    if parent_indices.shape != (len(joint_names),):
        raise ValueError("human_joint_parent_indices shape does not match human_joint_names")
    if np.any(parent_indices < -1) or np.any(parent_indices >= len(joint_names)):
        raise ValueError("human_joint_parent_indices contains out-of-range indices")
    return _HumanSkeleton(joints, joint_names, parent_indices)


def _optional_path_scalar(data: Mapping[str, np.ndarray], name: str) -> Path | None:
    return Path(str(_scalar(data, name))).expanduser() if name in data else None


def _source_scene_metadata(
    data: Mapping[str, np.ndarray],
    source_path: Path,
    source_data_format: str,
) -> _SourceSceneMetadata:
    human_position_scale = float(_scalar(data, "human_position_scale")) if "human_position_scale" in data else None
    if human_position_scale is not None and (not np.isfinite(human_position_scale) or human_position_scale <= 0.0):
        raise ValueError("human_position_scale must be positive and finite")
    source_motion_path = _optional_path_scalar(data, "source_path")
    if source_motion_path is not None:
        source_motion_path = source_motion_path.resolve()
    source_fbx = _optional_path_scalar(data, "source_fbx")
    source_actor = str(_scalar(data, "source_actor")) if "source_actor" in data else None
    source_xy_origin_m = (
        np.asarray(data["source_xy_origin_m"], dtype=np.float64).copy() if "source_xy_origin_m" in data else None
    )
    if source_data_format == "fbx_mocap" and source_xy_origin_m is None:
        source_fbx, source_actor, source_xy_origin_m = _recover_fbx_source_metadata(
            source_motion_path,
            source_path,
        )
    if source_xy_origin_m is not None and (
        source_xy_origin_m.shape != (2,) or not np.all(np.isfinite(source_xy_origin_m))
    ):
        raise ValueError("source_xy_origin_m must contain two finite values")
    return _SourceSceneMetadata(
        human_position_scale=human_position_scale,
        source_motion_path=source_motion_path,
        source_fbx=source_fbx,
        source_actor=source_actor,
        source_xy_origin_m=source_xy_origin_m,
    )


def _recover_fbx_source_metadata(
    source_motion_path: Path | None,
    result_path: Path,
) -> tuple[Path, str, np.ndarray]:
    if source_motion_path is None or not source_motion_path.is_file():
        raise ValueError(
            f"FBX result {result_path} must contain source scene metadata or reference an existing source_path"
        )
    with np.load(source_motion_path, allow_pickle=False) as source_data:
        required = ("source_fbx", "source_actor", "source_xy_origin_m")
        missing = [name for name in required if name not in source_data]
        if missing:
            raise ValueError(f"FBX source motion is missing paired-scene fields: {missing}")
        return (
            Path(str(np.asarray(source_data["source_fbx"]).item())).expanduser(),
            str(np.asarray(source_data["source_actor"]).item()),
            np.asarray(source_data["source_xy_origin_m"], dtype=np.float64).copy(),
        )


@dataclass(frozen=True)
class PairedActorTrajectory:
    """Validated single-actor result consumed by the paired solver."""

    source_path: Path
    qpos: np.ndarray
    fps: float
    robot_type: str
    source_data_format: str
    mapped_human_joints: np.ndarray
    mapped_human_joint_names: tuple[str, ...]
    mapped_robot_link_names: tuple[str, ...]
    robot_actuated_joint_names: tuple[str, ...]
    qpos_layout: str
    quaternion_convention: str
    world_coordinate_system: str
    human_joints: np.ndarray | None = None
    human_joint_names: tuple[str, ...] = ()
    human_joint_parent_indices: np.ndarray | None = None
    human_position_scale: float | None = None
    source_motion_path: Path | None = None
    source_fbx: Path | None = None
    source_actor: str | None = None
    source_xy_origin_m: np.ndarray | None = None

    @property
    def frame_count(self) -> int:
        """Number of synchronized frames in the trajectory."""
        return self.qpos.shape[0]


@dataclass(frozen=True)
class PairedRefinementResult:
    """Refined trajectories and compact optimization diagnostics."""

    actor_a_name: str
    actor_b_name: str
    actor_a: PairedActorTrajectory
    actor_b: PairedActorTrajectory
    actor_a_qpos: np.ndarray
    actor_b_qpos: np.ndarray
    frame_costs: np.ndarray
    sqp_iteration_counts: np.ndarray
    interaction_errors: np.ndarray
    root_position_errors: np.ndarray
    root_yaw_errors: np.ndarray
    minimum_inter_actor_distances: np.ndarray
    contact_errors: np.ndarray
    interaction_source_vertices: np.ndarray
    interaction_target_vertices: np.ndarray
    interaction_tetrahedra: tuple[np.ndarray, ...]
    interaction_edges: tuple[np.ndarray, ...]
    actor_a_source_translation: np.ndarray
    actor_b_source_translation: np.ndarray
    source_scene_scale: float
    source_relative_xy_m: np.ndarray
    source_fbx: Path | None
    source_alignment_mode: str
    actor_a_human_joints: np.ndarray | None
    actor_b_human_joints: np.ndarray | None
    config: PairedRefinementConfig


def load_actor_trajectory(path: str | Path) -> PairedActorTrajectory:
    """Load and validate one robot-only retargeting result."""
    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Retargeting result does not exist: {source_path}")

    with np.load(source_path, allow_pickle=False) as data:
        task_type = str(_scalar(data, "task_type"))
        if task_type != "robot_only":
            raise ValueError(f"Paired refinement requires task_type='robot_only', got {task_type!r}")
        if bool(_scalar(data, "contains_object_in_qpos")):
            raise ValueError("Paired refinement does not accept object coordinates in actor qpos")

        robot_type = str(_scalar(data, "robot_type"))
        source_data_format = str(_scalar(data, "source_data_format"))
        qpos, robot_config = _validated_qpos(data, robot_type)
        (
            mapped_human_joints,
            mapped_human_joint_names,
            mapped_robot_link_names,
            robot_actuated_joint_names,
        ) = _validated_mapping(data, qpos, robot_config.ROBOT_DOF)
        fps, qpos_layout, quaternion_convention, world_coordinate_system = _validated_motion_contract(data)
        human = _optional_human_skeleton(data, qpos.shape[0])
        source = _source_scene_metadata(data, source_path, source_data_format)

        return PairedActorTrajectory(
            source_path=source_path,
            qpos=qpos,
            fps=fps,
            robot_type=robot_type,
            source_data_format=source_data_format,
            mapped_human_joints=mapped_human_joints,
            mapped_human_joint_names=mapped_human_joint_names,
            mapped_robot_link_names=mapped_robot_link_names,
            robot_actuated_joint_names=robot_actuated_joint_names,
            qpos_layout=qpos_layout,
            quaternion_convention=quaternion_convention,
            world_coordinate_system=world_coordinate_system,
            human_joints=human.joints,
            human_joint_names=human.joint_names,
            human_joint_parent_indices=human.parent_indices,
            human_position_scale=source.human_position_scale,
            source_motion_path=source.source_motion_path,
            source_fbx=source.source_fbx,
            source_actor=source.source_actor,
            source_xy_origin_m=source.source_xy_origin_m,
        )


def load_paired_trajectories(
    actor_a_path: str | Path,
    actor_b_path: str | Path,
) -> tuple[PairedActorTrajectory, PairedActorTrajectory]:
    """Load two actor results and verify their synchronization contract."""
    actor_a = load_actor_trajectory(actor_a_path)
    actor_b = load_actor_trajectory(actor_b_path)
    if actor_a.frame_count != actor_b.frame_count:
        raise ValueError(
            f"Paired trajectories must have equal frame counts, got {actor_a.frame_count} and {actor_b.frame_count}"
        )
    if not np.isclose(actor_a.fps, actor_b.fps, rtol=0.0, atol=1e-9):
        raise ValueError(f"Paired trajectories must have equal fps, got {actor_a.fps} and {actor_b.fps}")
    if actor_a.quaternion_convention != actor_b.quaternion_convention:
        raise ValueError("Paired trajectories use different quaternion conventions")
    if actor_a.world_coordinate_system != actor_b.world_coordinate_system:
        raise ValueError("Paired trajectories use different world coordinate systems")
    if actor_a.source_data_format != actor_b.source_data_format:
        raise ValueError("Paired trajectories use different source data formats")
    if actor_a.source_data_format == "fbx_mocap":
        if actor_a.source_actor == actor_b.source_actor:
            raise ValueError("Paired FBX trajectories must represent distinct source actors")
        if actor_a.source_fbx is None or actor_b.source_fbx is None:
            raise ValueError("Paired FBX trajectories are missing source_fbx metadata")
        if actor_a.source_fbx.resolve() != actor_b.source_fbx.resolve():
            raise ValueError("Paired FBX trajectories must come from the same source container")
    return actor_a, actor_b


def save_paired_result(
    result: PairedRefinementResult,
    destination: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Save a paired refinement result without embedding Python objects."""
    output_path = Path(destination).expanduser().resolve()
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tetrahedra, tetrahedra_counts = _pack_ragged_indices(result.interaction_tetrahedra, 4)
    edges, edge_counts = _pack_ragged_indices(result.interaction_edges, 2)
    payload = {
        "actor_a_name": np.asarray(result.actor_a_name),
        "actor_b_name": np.asarray(result.actor_b_name),
        "actor_a_qpos": np.asarray(result.actor_a_qpos, dtype=np.float64),
        "actor_b_qpos": np.asarray(result.actor_b_qpos, dtype=np.float64),
        "actor_a_robot_type": np.asarray(result.actor_a.robot_type),
        "actor_b_robot_type": np.asarray(result.actor_b.robot_type),
        "actor_a_source_data_format": np.asarray(result.actor_a.source_data_format),
        "actor_b_source_data_format": np.asarray(result.actor_b.source_data_format),
        "actor_a_source_path": np.asarray(str(result.actor_a.source_path)),
        "actor_b_source_path": np.asarray(str(result.actor_b.source_path)),
        "actor_a_robot_actuated_joint_names": np.asarray(result.actor_a.robot_actuated_joint_names),
        "actor_b_robot_actuated_joint_names": np.asarray(result.actor_b.robot_actuated_joint_names),
        "actor_a_mapped_human_joint_names": np.asarray(result.actor_a.mapped_human_joint_names),
        "actor_b_mapped_human_joint_names": np.asarray(result.actor_b.mapped_human_joint_names),
        "actor_a_mapped_robot_link_names": np.asarray(result.actor_a.mapped_robot_link_names),
        "actor_b_mapped_robot_link_names": np.asarray(result.actor_b.mapped_robot_link_names),
        "actor_a_mapped_human_joints": np.asarray(
            result.interaction_source_vertices[:, : result.actor_a.mapped_human_joints.shape[1]],
            dtype=np.float32,
        ),
        "actor_b_mapped_human_joints": np.asarray(
            result.interaction_source_vertices[:, result.actor_a.mapped_human_joints.shape[1] :],
            dtype=np.float32,
        ),
        "actor_a_mapped_robot_joints": np.asarray(
            result.interaction_target_vertices[:, : result.actor_a.mapped_human_joints.shape[1]],
            dtype=np.float32,
        ),
        "actor_b_mapped_robot_joints": np.asarray(
            result.interaction_target_vertices[:, result.actor_a.mapped_human_joints.shape[1] :],
            dtype=np.float32,
        ),
        "actor_a_source_translation": np.asarray(result.actor_a_source_translation, dtype=np.float64),
        "actor_b_source_translation": np.asarray(result.actor_b_source_translation, dtype=np.float64),
        "source_scene_scale": np.asarray(result.source_scene_scale, dtype=np.float64),
        "source_relative_xy_m": np.asarray(result.source_relative_xy_m, dtype=np.float64),
        "source_fbx": np.asarray("" if result.source_fbx is None else str(result.source_fbx)),
        "source_alignment_mode": np.asarray(result.source_alignment_mode),
        "fps": np.asarray(result.actor_a.fps, dtype=np.float64),
        "qpos_layout": np.asarray(_EXPECTED_QPOS_LAYOUT),
        "quaternion_convention": np.asarray(_EXPECTED_QUATERNION_CONVENTION),
        "world_coordinate_system": np.asarray(_EXPECTED_WORLD_COORDINATE_SYSTEM),
        "frame_costs": np.asarray(result.frame_costs, dtype=np.float64),
        "sqp_iteration_counts": np.asarray(result.sqp_iteration_counts, dtype=np.int32),
        "interaction_errors": np.asarray(result.interaction_errors, dtype=np.float64),
        "root_position_errors": np.asarray(result.root_position_errors, dtype=np.float64),
        "root_yaw_errors": np.asarray(result.root_yaw_errors, dtype=np.float64),
        "minimum_inter_actor_distances": np.asarray(result.minimum_inter_actor_distances, dtype=np.float64),
        "contact_errors": np.asarray(result.contact_errors, dtype=np.float64),
        "interaction_source_vertices_w": np.asarray(result.interaction_source_vertices, dtype=np.float32),
        "interaction_target_vertices_w": np.asarray(result.interaction_target_vertices, dtype=np.float32),
        "interaction_tetrahedra": tetrahedra,
        "interaction_tetrahedra_counts": tetrahedra_counts,
        "interaction_edges": edges,
        "interaction_edge_counts": edge_counts,
        "interaction_num_actor_a_vertices": np.asarray(
            result.actor_a.mapped_human_joints.shape[1],
            dtype=np.int32,
        ),
        "refinement_config_json": np.asarray(json.dumps(asdict(result.config), sort_keys=True)),
        "run_kind": np.asarray("paired_refinement"),
    }
    if result.actor_a_human_joints is not None:
        payload.update(
            {
                "actor_a_human_joints": np.asarray(result.actor_a_human_joints, dtype=np.float32),
                "actor_a_human_joint_names": np.asarray(result.actor_a.human_joint_names),
                "actor_a_human_joint_parent_indices": np.asarray(
                    result.actor_a.human_joint_parent_indices,
                    dtype=np.int32,
                ),
            }
        )
    if result.actor_b_human_joints is not None:
        payload.update(
            {
                "actor_b_human_joints": np.asarray(result.actor_b_human_joints, dtype=np.float32),
                "actor_b_human_joint_names": np.asarray(result.actor_b.human_joint_names),
                "actor_b_human_joint_parent_indices": np.asarray(
                    result.actor_b.human_joint_parent_indices,
                    dtype=np.int32,
                ),
            }
        )
    np.savez_compressed(output_path, **payload)
    return output_path


def _pack_ragged_indices(values: tuple[np.ndarray, ...], width: int) -> tuple[np.ndarray, np.ndarray]:
    """Pack per-frame topology arrays using negative-one padding."""
    counts = np.asarray([len(value) for value in values], dtype=np.int32)
    maximum = int(counts.max(initial=0))
    packed = np.full((len(values), maximum, width), -1, dtype=np.int32)
    for frame_index, value in enumerate(values):
        indices = np.asarray(value, dtype=np.int32)
        if indices.shape != (counts[frame_index], width):
            raise ValueError(f"Topology entries must have width {width}, got {indices.shape}")
        packed[frame_index, : counts[frame_index]] = indices
    return packed, counts
