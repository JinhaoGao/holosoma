# ruff: noqa: CPY001
"""Shared result loading, asset resolution, and interpolation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from holosoma_retargeting.config_types.robot import RobotConfig
from holosoma_retargeting.data_utils.object_assets import get_omomo_object_asset
from holosoma_retargeting.data_utils.omomo import resolve_omomo_result_object_name
from holosoma_retargeting.src.viser_utils import (
    actuated_joint_names_from_mujoco_xml,
    build_joint_order_indices,
    infer_mujoco_xml_path,
)
from holosoma_retargeting.visualization.orientation import (
    OrientationDiagnostics,
    OrientationPreview,
    load_orientation_diagnostics,
    load_orientation_preview,
)

DEFAULT_VARIANTS: tuple[str, ...] = (
    "identity",
    "trans_0",
    "trans_1",
    "trans_2",
    "rot_0",
    "rot_1",
)

_RESULT_SUFFIX_PATTERN = re.compile(
    r"^(?P<sequence>.+)_(?P<variant>augmented|trans_[0-9]+|rot_[0-9]+|z_scale_[0-9]+(?:p|\.)[0-9]+)$"
)


@dataclass(frozen=True)
class AugmentationViserConfig:
    """Deprecated compatibility config for augmentation-family loading.

    New callers should use ``MultiViserConfig`` from ``multi_viser_player``.
    """

    qpos_npz: Path
    """Any result in the flat ``motion.npz`` augmentation family."""

    variants: tuple[str, ...] = DEFAULT_VARIANTS
    """Variant suffixes to load and display."""

    robot_urdf: Path | None = None
    """Optional robot URDF override. Otherwise resolved from result metadata."""

    robot_mujoco_xml: Path | None = None
    """Optional MuJoCo XML override used to recover qpos joint order."""

    object_urdf: Path | None = None
    """Optional object URDF override. Otherwise resolved from result metadata."""

    fps: float | None = None
    """Playback FPS override. Otherwise uses the FPS saved in the results."""

    loop: bool = False
    """Loop playback."""

    visual_fps_multiplier: int = 2
    """Interpolation multiplier for smooth playback."""

    robot_mesh_opacity: float = 0.45
    """Opacity shared by the color-coded robot meshes."""

    object_mesh_opacity: float = 0.65
    """Opacity shared by the color-coded object meshes."""

    skeleton_line_width: float = 2.0
    """Line width for human and robot skeletons."""

    grid_width: float = 8.0
    """Viewer grid width."""

    grid_height: float = 8.0
    """Viewer grid height."""


@dataclass(frozen=True)
class VariantResult:
    """Arrays and metadata required to render one augmentation variant."""

    variant: str
    path: Path
    config_json: str | None
    dataset_partition: str | None
    sequence_key: str | None
    experiment_name: str | None
    source_data_format: str
    run_kind: str | None
    task_type: str
    qpos: np.ndarray
    cost: float | None
    fps: float
    source_human_height: float | None
    human_position_scale: float | None
    human_joints: np.ndarray | None
    human_joint_names: tuple[str, ...]
    human_joint_parent_indices: np.ndarray | None
    human_points: np.ndarray | None
    robot_points: np.ndarray | None
    human_points_world: np.ndarray | None
    robot_points_world: np.ndarray | None
    terrain_points_world: np.ndarray | None
    mapped_joint_names: tuple[str, ...]
    mapped_robot_link_names: tuple[str, ...]
    human_orientation_joint_names: tuple[str, ...]
    human_orientation_quaternions_wxyz: np.ndarray | None
    robot_link_positions: np.ndarray | None
    robot_link_quaternions_wxyz: np.ndarray | None
    robot_link_names: tuple[str, ...]
    robot_link_parent_indices: np.ndarray | None
    robot_actuated_joint_names: tuple[str, ...]
    orientation_source: str | None
    orientation_diagnostics: OrientationDiagnostics | None
    orientation_preview: OrientationPreview | None
    robot_type: str
    object_name: str
    object_urdf: str | None
    object_poses_demo: np.ndarray | None
    object_poses_target: np.ndarray | None
    object_pose_layout: str | None
    contains_object_in_qpos: bool | None
    object_keypoints: dict[str, np.ndarray] | None
    interaction_mesh: dict[str, np.ndarray | int] | None
    interaction_mesh_edges_default: str | None
    foot_sticking: dict[str, object] | None


def split_result_family(path: str | Path) -> tuple[str, str | None]:
    """Return the shared sequence stem and optional augmentation suffix."""

    result_path = Path(path)
    stem = result_path.stem
    match = _RESULT_SUFFIX_PATTERN.fullmatch(stem)
    if match is None:
        return stem, None
    return match.group("sequence"), match.group("variant")


def _canonical_variant_sort_key(variant: str) -> tuple[int, float, str]:
    """Return a stable semantic order for automatically discovered variants."""

    if variant == "identity":
        return (0, 0.0, variant)
    translation_match = re.fullmatch(r"trans_([0-9]+)", variant)
    if translation_match is not None:
        return (1, float(translation_match.group(1)), variant)
    rotation_match = re.fullmatch(r"rot_([0-9]+)", variant)
    if rotation_match is not None:
        return (2, float(rotation_match.group(1)), variant)
    scale_match = re.fullmatch(r"z_scale_([0-9]+)(?:p|\.)([0-9]+)", variant)
    if scale_match is not None:
        scale = float(f"{scale_match.group(1)}.{scale_match.group(2)}")
        return (3, scale, variant)
    return (4, 0.0, variant)


def _result_path_for_variant(directory: Path, sequence: str, variant: str) -> Path:
    suffix = "" if variant in {"identity", "original"} else f"_{variant}"
    return directory / f"{sequence}{suffix}.npz"


def _discover_variant_paths(reference: Path) -> dict[str, Path]:
    """Discover a flat ``motion.npz`` augmentation family."""

    sequence, _ = split_result_family(reference)
    identity_path = _result_path_for_variant(reference.parent, sequence, "identity")
    if not identity_path.is_file():
        raise FileNotFoundError(f"No identity result was found for {reference}: {identity_path}")
    discovered = {"identity": identity_path}
    for path in sorted(reference.parent.glob("*.npz")):
        sibling_sequence, variant = split_result_family(path)
        if sibling_sequence == sequence and variant is not None:
            discovered[variant] = path
    return dict(
        sorted(
            discovered.items(),
            key=lambda item: _canonical_variant_sort_key(item[0]),
        )
    )


def discover_variant_paths(
    reference_path: str | Path,
    variants: tuple[str, ...] | None = DEFAULT_VARIANTS,
) -> dict[str, Path]:
    """Resolve one flat ``motion.npz`` augmentation family."""

    reference = Path(reference_path).expanduser()
    if variants is not None and not variants:
        raise ValueError("At least one augmentation variant must be requested.")
    if variants is not None and len(set(variants)) != len(variants):
        raise ValueError(f"Duplicate augmentation variants are not allowed: {variants}")

    if reference.is_dir():
        identities = [
            path for path in sorted(reference.glob("*.npz")) if path.is_file() and split_result_family(path)[1] is None
        ]
        if len(identities) != 1:
            raise ValueError(
                "A family directory must contain exactly one motion identity "
                f"NPZ, found {len(identities)} in {reference}"
            )
        reference = identities[0]
    sequence, _ = split_result_family(reference)
    if variants is None:
        return _discover_variant_paths(reference)
    paths = {variant: _result_path_for_variant(reference.parent, sequence, variant) for variant in variants}
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(f"Missing augmentation result files:\n{formatted}")
    return paths


def _npz_scalar(data: np.lib.npyio.NpzFile, key: str, default=None):
    if key not in data:
        return default
    return np.asarray(data[key]).item()


def _optional_npz_array(
    data: np.lib.npyio.NpzFile,
    key: str,
    *,
    dtype: type[np.generic | str],
) -> np.ndarray | None:
    if key not in data:
        return None
    return np.asarray(data[key], dtype=dtype)


def _optional_npz_text(
    data: np.lib.npyio.NpzFile,
    key: str,
) -> str | None:
    value = _npz_scalar(data, key)
    if value in {None, ""}:
        return None
    return str(value)


def _npz_string_list(data: np.lib.npyio.NpzFile, key: str) -> tuple[str, ...]:
    if key not in data:
        raise KeyError(f"Result is missing required field {key!r}")
    value = np.asarray(data[key])
    if value.ndim == 0:
        return (str(value.item()),)
    return tuple(str(item) for item in value.tolist())


def _optional_npz_string_list(
    data: np.lib.npyio.NpzFile,
    key: str,
) -> tuple[str, ...]:
    return _npz_string_list(data, key) if key in data else ()


def _load_foot_sticking_npz(
    data: np.lib.npyio.NpzFile,
    *,
    num_frames: int,
) -> dict[str, object] | None:
    if "foot_sticking_states" not in data:
        return None
    states = np.asarray(data["foot_sticking_states"], dtype=bool)
    if states.shape != (num_frames, 2):
        raise ValueError(f"foot_sticking_states must have shape ({num_frames}, 2), got {states.shape}")
    side_names = (
        _npz_string_list(data, "foot_sticking_side_names") if "foot_sticking_side_names" in data else ("left", "right")
    )
    if side_names != ("left", "right"):
        raise ValueError(f"foot_sticking_side_names must use canonical order ('left', 'right'), got {side_names}")
    metadata = {
        "states": states,
        "enabled": bool(_npz_scalar(data, "foot_sticking_enabled_for_saved_trajectory", False)),
        "tolerance": _npz_scalar(data, "foot_sticking_tolerance"),
    }
    if "foot_contact_modes" not in data:
        return metadata

    required_contact_fields = (
        "foot_contact_phase_ids",
        "foot_contact_confidences",
        "foot_contact_pivot_regions",
        "foot_contact_pivot_uv",
        "foot_contact_reference_displacements_xy",
    )
    missing_contact_fields = [name for name in required_contact_fields if name not in data]
    if missing_contact_fields:
        raise ValueError(
            "Saved foot-contact plan is incomplete; missing fields: " + ", ".join(missing_contact_fields),
        )

    modes = np.asarray(data["foot_contact_modes"], dtype=str)
    phase_ids = np.asarray(data["foot_contact_phase_ids"], dtype=np.int32)
    confidences = np.asarray(data["foot_contact_confidences"], dtype=np.float32)
    pivot_regions = np.asarray(data["foot_contact_pivot_regions"], dtype=str)
    pivot_uv = np.asarray(data["foot_contact_pivot_uv"], dtype=np.float32)
    reference_displacements = np.asarray(
        data["foot_contact_reference_displacements_xy"],
        dtype=np.float32,
    )
    expected_pair_shape = (num_frames, 2)
    for name, value in {
        "foot_contact_modes": modes,
        "foot_contact_phase_ids": phase_ids,
        "foot_contact_confidences": confidences,
        "foot_contact_pivot_regions": pivot_regions,
    }.items():
        if value.shape != expected_pair_shape:
            raise ValueError(f"{name} must have shape {expected_pair_shape}, got {value.shape}")
    expected_vector_shape = (num_frames, 2, 2)
    if pivot_uv.shape != expected_vector_shape:
        raise ValueError(f"foot_contact_pivot_uv must have shape {expected_vector_shape}, got {pivot_uv.shape}")
    if reference_displacements.shape != expected_vector_shape:
        raise ValueError(
            "foot_contact_reference_displacements_xy must have shape "
            f"{expected_vector_shape}, got {reference_displacements.shape}",
        )
    metadata.update(
        {
            "modes": modes,
            "phase_ids": phase_ids,
            "confidences": confidences,
            "pivot_regions": pivot_regions,
            "pivot_uv": pivot_uv,
            "reference_displacements_xy": reference_displacements,
            "plan_version": int(_npz_scalar(data, "foot_contact_plan_version", 1)),
        },
    )
    return metadata


def _load_interaction_mesh_npz(
    data: np.lib.npyio.NpzFile,
) -> dict[str, np.ndarray | int] | None:
    legacy_required = (
        "interaction_source_vertices_w",
        "interaction_target_vertices_w",
        "interaction_tetrahedra",
        "interaction_tetrahedra_counts",
        "interaction_num_human_vertices",
    )
    if not all(key in data for key in legacy_required):
        return None
    source_vertices = np.asarray(
        data["interaction_source_vertices_w"],
        dtype=np.float32,
    )
    num_human_vertices = int(np.asarray(data["interaction_num_human_vertices"]).item())
    num_object_vertices = int(
        _npz_scalar(
            data,
            "interaction_num_object_vertices",
            source_vertices.shape[1] - num_human_vertices if source_vertices.ndim == 3 else 0,
        )
    )
    return {
        "source_vertices": source_vertices,
        "target_vertices": np.asarray(
            data["interaction_target_vertices_w"],
            dtype=np.float32,
        ),
        "tetrahedra": np.asarray(
            data["interaction_tetrahedra"],
            dtype=np.int32,
        ),
        "tetrahedra_counts": np.asarray(
            data["interaction_tetrahedra_counts"],
            dtype=np.int32,
        ),
        "num_human_vertices": num_human_vertices,
        "num_object_vertices": num_object_vertices,
    }


def _load_object_keypoints_npz(
    data: np.lib.npyio.NpzFile,
) -> dict[str, np.ndarray] | None:
    required_keys = (
        "object_points_demo_world",
        "object_points_target_world",
    )
    if all(key in data for key in required_keys):
        result = {
            "demo_world": np.asarray(
                data["object_points_demo_world"],
                dtype=np.float32,
            ),
            "target_world": np.asarray(
                data["object_points_target_world"],
                dtype=np.float32,
            ),
        }
        if "object_points_demo_local" in data:
            result["demo_local"] = np.asarray(
                data["object_points_demo_local"],
                dtype=np.float32,
            )
        if "object_points_target_local" in data:
            result["target_local"] = np.asarray(
                data["object_points_target_local"],
                dtype=np.float32,
            )
        return result

    legacy_keys = (
        "interaction_source_vertices_w",
        "interaction_target_vertices_w",
        "interaction_num_human_vertices",
    )
    if not all(key in data for key in legacy_keys):
        return None
    num_human_vertices = int(np.asarray(data["interaction_num_human_vertices"]).item())
    return {
        "demo_world": np.asarray(
            data["interaction_source_vertices_w"][:, num_human_vertices:],
            dtype=np.float32,
        ),
        "target_world": np.asarray(
            data["interaction_target_vertices_w"][:, num_human_vertices:],
            dtype=np.float32,
        ),
    }


def _validate_saved_quaternions(
    values: np.ndarray,
    *,
    path: Path,
    field_name: str,
) -> np.ndarray:
    quaternions = np.asarray(values, dtype=np.float32)
    if not np.isfinite(quaternions).all():
        raise ValueError(f"{path} {field_name} contains non-finite values")
    norms = np.linalg.norm(quaternions.astype(np.float64), axis=-1)
    if np.any(np.abs(norms - 1.0) > 1e-3):
        raise ValueError(f"{path} {field_name} must contain unit wxyz quaternions")
    return quaternions


def _load_human_orientations(
    data: np.lib.npyio.NpzFile,
    *,
    result_path: Path,
    num_frames: int,
    human_joint_names: tuple[str, ...],
) -> tuple[tuple[str, ...], np.ndarray | None]:
    canonical_keys = (
        "human_orientation_joint_names",
        "human_orientation_quaternions_wxyz",
    )
    present = tuple(key for key in canonical_keys if key in data)
    if present:
        missing = tuple(key for key in canonical_keys if key not in data)
        if missing:
            raise ValueError(f"{result_path} has incomplete source-human orientation fields; missing {missing}")
        names = _npz_string_list(data, canonical_keys[0])
        quaternions = _validate_saved_quaternions(
            np.asarray(data[canonical_keys[1]]),
            path=result_path,
            field_name=canonical_keys[1],
        )
        if quaternions.shape != (num_frames, len(names), 4):
            raise ValueError(
                f"{result_path} {canonical_keys[1]} shape {quaternions.shape} != {(num_frames, len(names), 4)}"
            )
        if len(set(names)) != len(names) or not set(names).issubset(human_joint_names):
            raise ValueError(
                f"{result_path} source-human orientation names must be a unique subset of human_joint_names"
            )
        return names, quaternions

    legacy_pair_keys = (
        "orientation_joint_names",
        "orientation_quaternions_wxyz",
    )
    legacy_pair_present = tuple(key for key in legacy_pair_keys if key in data)
    if legacy_pair_present:
        missing = tuple(key for key in legacy_pair_keys if key not in data)
        if missing:
            raise ValueError(f"{result_path} has incomplete legacy source-human orientation fields; missing {missing}")
        names = _npz_string_list(data, legacy_pair_keys[0])
        quaternions = _validate_saved_quaternions(
            np.asarray(data[legacy_pair_keys[1]]),
            path=result_path,
            field_name=legacy_pair_keys[1],
        )
        if quaternions.shape != (num_frames, len(names), 4):
            raise ValueError(
                f"{result_path} {legacy_pair_keys[1]} shape {quaternions.shape} != {(num_frames, len(names), 4)}"
            )
        if len(set(names)) != len(names) or not set(names).issubset(human_joint_names):
            raise ValueError(
                f"{result_path} legacy source-human orientation names must be a unique subset of human_joint_names"
            )
        return names, quaternions

    direct_orientation_provenance = {
        "direct_source",
        "direct_source_bones",
        "direct_source_bvh_fk",
    }
    orientation_provenance = str(_npz_scalar(data, "orientation_provenance", ""))
    if (
        "global_joint_quaternions_wxyz" in data
        and human_joint_names
        and orientation_provenance in direct_orientation_provenance
    ):
        quaternions = _validate_saved_quaternions(
            np.asarray(data["global_joint_quaternions_wxyz"]),
            path=result_path,
            field_name="global_joint_quaternions_wxyz",
        )
        if quaternions.shape != (num_frames, len(human_joint_names), 4):
            raise ValueError(
                f"{result_path} legacy global_joint_quaternions_wxyz shape "
                f"{quaternions.shape} != {(num_frames, len(human_joint_names), 4)}"
            )
        return human_joint_names, quaternions
    return (), None


def load_variant_result(
    variant: str,
    path: str | Path,
) -> VariantResult:
    """Load one result directly from fields used by downstream consumers."""

    result_path = Path(path)
    with np.load(result_path, allow_pickle=False) as data:
        required_arrays = ("qpos",)
        missing = [key for key in required_arrays if key not in data]
        if missing:
            raise KeyError(f"{result_path} is missing required fields: {', '.join(missing)}")

        qpos = np.asarray(data["qpos"], dtype=np.float64)
        human_joints = np.asarray(data["human_joints"], dtype=np.float32) if "human_joints" in data else None
        robot_points = (
            np.asarray(data["mapped_robot_joints"], dtype=np.float32) if "mapped_robot_joints" in data else None
        )
        human_points_world = _optional_npz_array(
            data,
            "human_points_world",
            dtype=np.float32,
        )
        robot_points_world = _optional_npz_array(
            data,
            "robot_points_world",
            dtype=np.float32,
        )
        terrain_points_world = _optional_npz_array(
            data,
            "terrain_points_world",
            dtype=np.float32,
        )
        human_joint_names = _optional_npz_string_list(data, "human_joint_names")
        human_joint_parent_indices = (
            np.asarray(data["human_joint_parent_indices"], dtype=np.int32)
            if "human_joint_parent_indices" in data
            else None
        )
        mapped_joint_names = _optional_npz_string_list(
            data,
            "mapped_human_joint_names",
        )
        mapped_robot_link_names = (
            _npz_string_list(data, "mapped_robot_link_names")
            if "mapped_robot_link_names" in data
            else tuple("" for _ in mapped_joint_names)
        )
        human_orientation_joint_names, human_orientation_quaternions = _load_human_orientations(
            data,
            result_path=result_path,
            num_frames=int(qpos.shape[0]),
            human_joint_names=human_joint_names,
        )
        robot_link_positions = (
            np.asarray(data["robot_link_positions"], dtype=np.float32) if "robot_link_positions" in data else None
        )
        robot_link_quaternions = (
            np.asarray(data["robot_link_quaternions_wxyz"], dtype=np.float32)
            if "robot_link_quaternions_wxyz" in data
            else None
        )
        robot_link_names = _optional_npz_string_list(data, "robot_link_names")
        robot_link_parent_indices = (
            np.asarray(data["robot_link_parent_indices"], dtype=np.int32)
            if "robot_link_parent_indices" in data
            else None
        )
        robot_actuated_joint_names = _optional_npz_string_list(
            data,
            "robot_actuated_joint_names",
        )
        fps = float(_npz_scalar(data, "fps", 30.0))
        robot_type = str(_npz_scalar(data, "robot_type", ""))
        config_json = _optional_npz_text(data, "config_json")
        dataset_partition = _optional_npz_text(data, "dataset_partition")
        sequence_key = _optional_npz_text(data, "sequence_key")
        experiment_name = _optional_npz_text(data, "experiment_name")
        source_data_format = str(_npz_scalar(data, "source_data_format", ""))
        run_kind = _optional_npz_text(data, "run_kind")
        task_type = str(_npz_scalar(data, "task_type", ""))
        saved_cost = _npz_scalar(data, "cost")
        cost = float(saved_cost) if saved_cost is not None else None
        saved_source_human_height = _npz_scalar(
            data,
            "source_human_height",
        )
        source_human_height = float(saved_source_human_height) if saved_source_human_height is not None else None
        saved_human_position_scale = _npz_scalar(
            data,
            "human_position_scale",
        )
        human_position_scale = float(saved_human_position_scale) if saved_human_position_scale is not None else None
        orientation_source = _optional_npz_text(data, "orientation_source")
        object_name = str(_npz_scalar(data, "object_name", ""))
        object_urdf_value = str(_npz_scalar(data, "object_urdf", ""))
        object_poses_demo = _optional_npz_array(
            data,
            "object_poses_demo",
            dtype=np.float32,
        )
        object_poses_target = _optional_npz_array(
            data,
            "object_poses_target",
            dtype=np.float32,
        )
        object_pose_layout = _optional_npz_text(data, "object_pose_layout")
        saved_contains_object = _npz_scalar(data, "contains_object_in_qpos")
        contains_object = bool(saved_contains_object) if saved_contains_object is not None else None
        object_keypoints = _load_object_keypoints_npz(data)
        interaction_mesh = _load_interaction_mesh_npz(data)
        interaction_mesh_edges_default = _optional_npz_text(
            data,
            "interaction_mesh_edges_default",
        )
        foot_sticking = _load_foot_sticking_npz(
            data,
            num_frames=int(qpos.shape[0]),
        )

    if qpos.ndim != 2 or qpos.shape[0] == 0 or not np.isfinite(qpos).all():
        raise ValueError(f"{result_path} qpos must be a non-empty finite 2-D array, got {qpos.shape}")
    if human_joints is not None and (
        human_joints.ndim != 3
        or human_joints.shape[0] != qpos.shape[0]
        or human_joints.shape[-1] != 3
        or not np.isfinite(human_joints).all()
    ):
        raise ValueError(
            f"{result_path} human_joints must have shape (frames, joints, 3) matching qpos; got {human_joints.shape}"
        )
    if robot_points is not None and robot_points.shape != (qpos.shape[0], len(mapped_joint_names), 3):
        raise ValueError(
            f"{result_path} mapped_robot_joints must have shape "
            f"({qpos.shape[0]}, {len(mapped_joint_names)}, 3), got {robot_points.shape}"
        )
    for field_name, points in (
        ("human_points_world", human_points_world),
        ("robot_points_world", robot_points_world),
        ("terrain_points_world", terrain_points_world),
    ):
        if points is not None and (
            points.ndim != 3
            or points.shape[0] != qpos.shape[0]
            or points.shape[-1] != 3
            or not np.isfinite(points).all()
        ):
            raise ValueError(
                f"{result_path} {field_name} must have shape (frames, points, 3) matching qpos, got {points.shape}"
            )
    if len(mapped_robot_link_names) != len(mapped_joint_names):
        raise ValueError(
            f"{result_path} mapped_robot_link_names has {len(mapped_robot_link_names)} entries "
            f"for {len(mapped_joint_names)} mapped joints"
        )
    if human_joints is not None and len(human_joint_names) != human_joints.shape[1]:
        raise ValueError(
            f"{result_path} human_joint_names has {len(human_joint_names)} entries for {human_joints.shape[1]} joints"
        )
    if human_joint_parent_indices is not None:
        expected_parent_shape = (len(human_joint_names),)
        if human_joint_parent_indices.shape != expected_parent_shape:
            raise ValueError(
                f"{result_path} human_joint_parent_indices shape "
                f"{human_joint_parent_indices.shape} != {expected_parent_shape}"
            )
        joint_indices = np.arange(len(human_joint_names), dtype=np.int32)
        invalid_parents = (
            (human_joint_parent_indices < -1)
            | (human_joint_parent_indices >= len(human_joint_names))
            | (human_joint_parent_indices == joint_indices)
        )
        if np.any(invalid_parents):
            raise ValueError(
                f"{result_path} human_joint_parent_indices must contain -1 "
                "for roots or a different valid joint index for every child"
            )
    if not object_name:
        try:
            object_name = resolve_omomo_result_object_name(result_path)
        except ValueError:
            pass
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"{result_path} has invalid fps={fps}")

    human_points = None
    if human_joints is not None and mapped_joint_names:
        human_index = {name: index for index, name in enumerate(human_joint_names)}
        unavailable = [name for name in mapped_joint_names if name not in human_index]
        if not unavailable:
            human_points = human_joints[
                :,
                [human_index[name] for name in mapped_joint_names],
            ]

    full_robot_group_present = any(
        (
            robot_link_positions is not None,
            robot_link_quaternions is not None,
            bool(robot_link_names),
            robot_link_parent_indices is not None,
        )
    )
    if full_robot_group_present and (
        robot_link_positions is None
        or robot_link_quaternions is None
        or not robot_link_names
        or robot_link_parent_indices is None
    ):
        robot_link_positions = None
        robot_link_quaternions = None
        robot_link_names = ()
        robot_link_parent_indices = None

    orientation_diagnostics = load_orientation_diagnostics(
        result_path,
        expected_frames=qpos.shape[0],
    )
    orientation_preview = load_orientation_preview(
        result_path,
        expected_frames=qpos.shape[0],
    )

    return VariantResult(
        variant="identity" if variant == "original" else variant,
        path=result_path,
        config_json=config_json,
        dataset_partition=dataset_partition,
        sequence_key=sequence_key,
        experiment_name=experiment_name,
        source_data_format=source_data_format,
        run_kind=run_kind,
        task_type=task_type,
        qpos=qpos,
        cost=cost,
        fps=fps,
        source_human_height=source_human_height,
        human_position_scale=human_position_scale,
        human_joints=human_joints,
        human_joint_names=human_joint_names,
        human_joint_parent_indices=human_joint_parent_indices,
        human_points=human_points,
        robot_points=robot_points,
        human_points_world=human_points_world,
        robot_points_world=robot_points_world,
        terrain_points_world=terrain_points_world,
        mapped_joint_names=mapped_joint_names,
        mapped_robot_link_names=mapped_robot_link_names,
        human_orientation_joint_names=human_orientation_joint_names,
        human_orientation_quaternions_wxyz=human_orientation_quaternions,
        robot_link_positions=robot_link_positions,
        robot_link_quaternions_wxyz=robot_link_quaternions,
        robot_link_names=robot_link_names,
        robot_link_parent_indices=robot_link_parent_indices,
        robot_actuated_joint_names=robot_actuated_joint_names,
        orientation_source=orientation_source,
        orientation_diagnostics=orientation_diagnostics,
        orientation_preview=orientation_preview,
        robot_type=robot_type,
        object_name=object_name,
        object_urdf=object_urdf_value or None,
        object_poses_demo=object_poses_demo,
        object_poses_target=object_poses_target,
        object_pose_layout=object_pose_layout,
        contains_object_in_qpos=contains_object,
        object_keypoints=object_keypoints,
        interaction_mesh=interaction_mesh,
        interaction_mesh_edges_default=interaction_mesh_edges_default,
        foot_sticking=foot_sticking,
    )


def load_result_family(config: AugmentationViserConfig) -> list[VariantResult]:
    """Load every requested augmentation result."""

    paths = discover_variant_paths(config.qpos_npz, config.variants)
    return [load_variant_result(variant, path) for variant, path in paths.items()]


def variant_result_metadata(result: VariantResult) -> dict[str, object]:
    """Expose one loader result through the legacy single-viewer metadata API."""

    return {
        "variant": result.variant,
        "config_json": result.config_json,
        "dataset_partition": result.dataset_partition,
        "sequence_key": result.sequence_key,
        "experiment_name": result.experiment_name,
        "source_data_format": result.source_data_format or None,
        "run_kind": result.run_kind,
        "task_type": result.task_type or None,
        "cost": result.cost,
        "source_human_height": result.source_human_height,
        "human_position_scale": result.human_position_scale,
        "human_joint_names": list(result.human_joint_names) or None,
        "human_joint_parent_indices": result.human_joint_parent_indices,
        "mapped_human_joint_names": list(result.mapped_joint_names) or None,
        "mapped_robot_joints": result.robot_points,
        "mapped_robot_link_names": list(result.mapped_robot_link_names) or None,
        "human_points_world": result.human_points_world,
        "robot_points_world": result.robot_points_world,
        "terrain_points_world": result.terrain_points_world,
        "human_orientation_joint_names": list(result.human_orientation_joint_names) or None,
        "human_orientation_quaternions_wxyz": (result.human_orientation_quaternions_wxyz),
        "robot_link_positions": result.robot_link_positions,
        "robot_link_quaternions_wxyz": result.robot_link_quaternions_wxyz,
        "robot_link_names": list(result.robot_link_names) or None,
        "robot_link_parent_indices": result.robot_link_parent_indices,
        "robot_actuated_joint_names": list(result.robot_actuated_joint_names) or None,
        "orientation_source": result.orientation_source,
        "orientation_diagnostics": result.orientation_diagnostics,
        "orientation_preview": result.orientation_preview,
        "robot_type": result.robot_type or None,
        "object_name": result.object_name or None,
        "object_urdf": result.object_urdf,
        "object_poses_demo": result.object_poses_demo,
        "object_poses_target": result.object_poses_target,
        "object_pose_layout": result.object_pose_layout,
        "contains_object_in_qpos": result.contains_object_in_qpos,
        "object_keypoints": result.object_keypoints,
        "interaction_mesh_edges_default": (result.interaction_mesh_edges_default),
        "foot_sticking": result.foot_sticking,
    }


def _resolve_asset_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    package_root = Path(__file__).resolve().parents[1]
    package_candidate = package_root / path
    return package_candidate if package_candidate.exists() else path


def _resolve_robot_urdf(config: AugmentationViserConfig, result: VariantResult) -> Path:
    value = config.robot_urdf
    if value is None:
        if not result.robot_type:
            raise ValueError(f"{result.path} has no robot_type metadata; pass --robot-urdf explicitly.")
        value = RobotConfig(robot_type=result.robot_type).ROBOT_URDF_FILE
    path = _resolve_asset_path(value)
    if not path.is_file():
        raise FileNotFoundError(f"Robot URDF not found: {path}")
    return path


def _resolve_object_urdf(config: AugmentationViserConfig, result: VariantResult) -> Path:
    value = config.object_urdf or result.object_urdf
    if value:
        path = _resolve_asset_path(value)
    else:
        if not result.object_name:
            raise ValueError(f"{result.path} has no object metadata; pass --object-urdf explicitly.")
        path = get_omomo_object_asset(result.object_name).urdf_path
    if not path.is_file():
        raise FileNotFoundError(f"Object URDF not found: {path}")
    return path


def _resolve_robot_xml(config: AugmentationViserConfig, robot_urdf: Path) -> Path | None:
    if config.robot_mujoco_xml is not None:
        path = _resolve_asset_path(config.robot_mujoco_xml)
        if not path.is_file():
            raise FileNotFoundError(f"Robot MuJoCo XML not found: {path}")
        return path
    return infer_mujoco_xml_path(robot_urdf)


def resolve_qpos_to_viser_joint_indices(
    *,
    result_path: str | Path,
    saved_joint_names: tuple[str, ...],
    viser_joint_names: tuple[str, ...],
    fallback_mujoco_xml: Path | None,
) -> np.ndarray | None:
    """Map qpos articulation using saved names or a MuJoCo XML fallback."""

    path = Path(result_path)
    if saved_joint_names:
        source_joint_names = saved_joint_names
        source_description = "saved robot_actuated_joint_names"
    else:
        if fallback_mujoco_xml is None:
            raise FileNotFoundError(
                f"{path} has no robot_actuated_joint_names and no MuJoCo XML is available to recover qpos joint order."
            )
        if not fallback_mujoco_xml.is_file():
            raise FileNotFoundError(f"{path} qpos-order XML does not exist: {fallback_mujoco_xml}")
        source_joint_names = tuple(actuated_joint_names_from_mujoco_xml(fallback_mujoco_xml))
        source_description = f"MuJoCo XML {fallback_mujoco_xml}"

    if len(set(viser_joint_names)) != len(viser_joint_names):
        raise ValueError(f"The current Viser URDF exposes duplicate actuated joint names: {viser_joint_names}")
    if len(source_joint_names) != len(viser_joint_names) or set(source_joint_names) != set(viser_joint_names):
        missing_from_result = tuple(name for name in viser_joint_names if name not in source_joint_names)
        missing_from_urdf = tuple(name for name in source_joint_names if name not in viser_joint_names)
        raise ValueError(
            f"{path} {source_description} is inconsistent with the current "
            "Viser URDF actuated joints: "
            f"missing_from_result={missing_from_result}, "
            f"missing_from_urdf={missing_from_urdf}, "
            f"saved_count={len(source_joint_names)}, "
            f"urdf_count={len(viser_joint_names)}"
        )
    if source_joint_names == viser_joint_names:
        return None
    return build_joint_order_indices(
        source_joint_names,
        viser_joint_names,
    )


def _rgba(color: tuple[int, int, int], opacity: float) -> tuple[float, float, float, float]:
    alpha = float(np.clip(opacity, 0.0, 1.0))
    return color[0] / 255.0, color[1] / 255.0, color[2] / 255.0, alpha


def _slerp(q0: np.ndarray, q1: np.ndarray, fraction: float) -> np.ndarray:
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    q0 /= max(float(np.linalg.norm(q0)), 1e-12)
    q1 /= max(float(np.linalg.norm(q1)), 1e-12)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        result = q0 + fraction * (q1 - q0)
        return result / max(float(np.linalg.norm(result)), 1e-12)
    theta = float(np.arccos(np.clip(dot, -1.0, 1.0)))
    return (np.sin((1.0 - fraction) * theta) * q0 + np.sin(fraction * theta) * q1) / np.sin(theta)


def interpolate_qpos(
    qpos: np.ndarray,
    frame_float: float,
    robot_dof: int,
    *,
    contains_object: bool,
    loop: bool = False,
) -> np.ndarray:
    """Interpolate one MuJoCo-order qpos frame with quaternion SLERP."""

    if loop:
        sample_frame = float(frame_float) % qpos.shape[0]
        i0 = int(np.floor(sample_frame))
        i1 = (i0 + 1) % qpos.shape[0]
    else:
        sample_frame = float(np.clip(frame_float, 0.0, qpos.shape[0] - 1))
        i0 = int(np.floor(sample_frame))
        i1 = min(i0 + 1, qpos.shape[0] - 1)
    fraction = sample_frame - i0
    if i0 == i1:
        return qpos[i0].copy()

    q0 = qpos[i0]
    q1 = qpos[i1]
    result = (1.0 - fraction) * q0 + fraction * q1
    result[3:7] = _slerp(q0[3:7], q1[3:7], fraction)
    if contains_object:
        result[-4:] = _slerp(q0[-4:], q1[-4:], fraction)
    expected_minimum = 7 + robot_dof + (7 if contains_object else 0)
    if result.shape[0] < expected_minimum:
        raise ValueError(
            f"qpos has {result.shape[0]} values, expected at least {expected_minimum} for robot_dof={robot_dof}"
        )
    return result


def _robot_joints_for_viser(
    q: np.ndarray,
    robot_dof: int,
    joint_order_indices: np.ndarray | None,
) -> np.ndarray:
    if joint_order_indices is None:
        return np.asarray(q[7 : 7 + robot_dof])
    required_length = int(joint_order_indices.max()) + 1
    raw_joints = np.asarray(q[7 : 7 + required_length])
    if raw_joints.shape[0] < required_length:
        raise ValueError(f"qpos only has {raw_joints.shape[0]} robot joints; {required_length} are required")
    return raw_joints[joint_order_indices]
