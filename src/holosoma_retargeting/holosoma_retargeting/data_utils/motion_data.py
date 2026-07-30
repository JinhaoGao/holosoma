# ruff: noqa: CPY001

"""Unified loading and discovery for supported human-motion datasets."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Sequence

import numpy as np
import torch
from holosoma_retargeting.config_types.data_type import (
    APPROVED_DIRECT_ORIENTATION_SOURCES,
    DEMO_JOINT_PARENT_INDICES,
    DEMO_JOINTS_REGISTRY,
    normalize_data_format,
)
from holosoma_retargeting.data_utils.omomo import parse_omomo_sequence_name

OrientationMode = Literal["bvh", "smpl", "mocap"]


@dataclass(frozen=True)
class DirectOrientationNPZSpec:
    """Declared metadata contract for direct orientations stored in NPZ."""

    source_formats: frozenset[str]
    coordinate_systems: frozenset[str]
    require_orientation_coordinate_system: bool = False


@dataclass(frozen=True)
class MotionFormatSpec:
    """File and initialization contract for one canonical data format."""

    name: str
    suffixes: tuple[str, ...]
    task_types: frozenset[str]
    root_joint: str
    orientation_mode: OrientationMode
    default_fps: float
    default_human_height: float | None = None
    nested_files: bool = False
    recursive_files: bool = False
    excluded_filenames: frozenset[str] = frozenset()
    direct_orientation_npz: DirectOrientationNPZSpec | None = None


@dataclass(frozen=True)
class HumanMotion:
    """Validated source motion consumed by the retargeting pipeline.

    Orientations contain only frames that were present directly in the source
    representation. ``orientation_joint_names`` therefore need not cover the
    complete canonical skeleton.
    """

    joints: np.ndarray
    fps: float
    human_height: float
    source_path: Path
    object_poses_wxyz_xyz: np.ndarray | None = None
    orientation_joint_names: tuple[str, ...] | None = None
    orientation_quaternions_wxyz: np.ndarray | None = None
    orientation_source: str | None = None
    joint_parent_indices: tuple[int, ...] = ()
    _canonical_joint_names: tuple[str, ...] = field(default=(), repr=False)
    _root_joint_name: str | None = field(default=None, repr=False)

    @property
    def canonical_joint_names(self) -> tuple[str, ...]:
        """Return the exact joint names/order associated with ``joints``."""

        return self._canonical_joint_names

    @property
    def root_quaternions_wxyz(self) -> np.ndarray | None:
        """Return the directly observed canonical root orientation, if present."""
        if (
            self.orientation_joint_names is None
            or self.orientation_quaternions_wxyz is None
            or self._root_joint_name not in self.orientation_joint_names
        ):
            return None
        root_index = self.orientation_joint_names.index(self._root_joint_name)
        return self.orientation_quaternions_wxyz[:, root_index]

    @property
    def global_joint_quaternions_wxyz(self) -> np.ndarray | None:
        """Return legacy dense orientations only when every joint is observed."""
        if (
            self.orientation_joint_names is None
            or self.orientation_quaternions_wxyz is None
            or len(self.orientation_joint_names) != len(self._canonical_joint_names)
            or set(self.orientation_joint_names) != set(self._canonical_joint_names)
        ):
            return None
        if self.orientation_joint_names == self._canonical_joint_names:
            return self.orientation_quaternions_wxyz
        indices = [self.orientation_joint_names.index(name) for name in self._canonical_joint_names]
        return self.orientation_quaternions_wxyz[:, indices]


def validate_motion_skeleton_contract(
    motion: HumanMotion,
    configured_joint_names: Sequence[str],
    configured_parent_indices: Sequence[int],
    *,
    data_format: str,
) -> None:
    """Require config names/topology to describe the loaded adapter tensor.

    Registered adapters currently emit a fixed canonical joint axis. A config
    override must therefore match that axis exactly; silently relabeling or
    reordering the tensor would corrupt all downstream mappings and metadata.
    """

    adapter_names = motion.canonical_joint_names
    config_names = tuple(str(name) for name in configured_joint_names)
    if config_names != adapter_names:
        mismatch_index = next(
            (
                index
                for index, (configured, adapter) in enumerate(
                    zip(config_names, adapter_names, strict=False),
                )
                if configured != adapter
            ),
            min(len(config_names), len(adapter_names)),
        )
        configured_value = config_names[mismatch_index] if mismatch_index < len(config_names) else "<missing>"
        adapter_value = adapter_names[mismatch_index] if mismatch_index < len(adapter_names) else "<missing>"
        raise ValueError(
            "MotionDataConfig human skeleton contract is incompatible with "
            f"the loaded {data_format!r} adapter: joint names/order must "
            "match exactly; first mismatch at index "
            f"{mismatch_index}: configured={configured_value!r}, "
            f"adapter={adapter_value!r}; counts configured={len(config_names)}, "
            f"adapter={len(adapter_names)}",
        )

    adapter_parents = tuple(int(parent) for parent in motion.joint_parent_indices)
    config_parents = tuple(int(parent) for parent in configured_parent_indices)
    if config_parents != adapter_parents:
        mismatch_index = next(
            (
                index
                for index, (configured, adapter) in enumerate(
                    zip(config_parents, adapter_parents, strict=False),
                )
                if configured != adapter
            ),
            min(len(config_parents), len(adapter_parents)),
        )
        configured_value = config_parents[mismatch_index] if mismatch_index < len(config_parents) else "<missing>"
        adapter_value = adapter_parents[mismatch_index] if mismatch_index < len(adapter_parents) else "<missing>"
        raise ValueError(
            "MotionDataConfig human skeleton contract is incompatible with "
            f"the loaded {data_format!r} adapter: parent topology must match "
            "exactly; first mismatch at index "
            f"{mismatch_index}: configured={configured_value!r}, "
            f"adapter={adapter_value!r}; counts configured={len(config_parents)}, "
            f"adapter={len(adapter_parents)}",
        )


MOTION_FORMATS: dict[str, MotionFormatSpec] = {
    "amass": MotionFormatSpec(
        name="amass",
        suffixes=(".npz",),
        task_types=frozenset({"robot_only"}),
        root_joint="Pelvis",
        orientation_mode="smpl",
        default_fps=30.0,
        direct_orientation_npz=DirectOrientationNPZSpec(
            source_formats=frozenset({"amass"}),
            coordinate_systems=frozenset({"right_handed_z_up"}),
            require_orientation_coordinate_system=True,
        ),
    ),
    "gvhmr": MotionFormatSpec(
        name="gvhmr",
        suffixes=(".npz",),
        task_types=frozenset({"robot_only"}),
        root_joint="Pelvis",
        orientation_mode="smpl",
        default_fps=30.0,
        direct_orientation_npz=DirectOrientationNPZSpec(
            source_formats=frozenset({"gvhmr"}),
            coordinate_systems=frozenset({"right_handed_z_up"}),
            require_orientation_coordinate_system=True,
        ),
    ),
    "lafan": MotionFormatSpec(
        name="lafan",
        suffixes=(".npz", ".npy"),
        task_types=frozenset({"robot_only"}),
        root_joint="Spine1",
        orientation_mode="bvh",
        default_fps=30.0,
        default_human_height=1.7,
        direct_orientation_npz=DirectOrientationNPZSpec(
            source_formats=frozenset({"lafan"}),
            coordinate_systems=frozenset({"z_up"}),
        ),
    ),
    "mocap": MotionFormatSpec(
        name="mocap",
        suffixes=(".npz", ".npy"),
        task_types=frozenset({"robot_only", "climbing"}),
        root_joint="Spine1",
        orientation_mode="mocap",
        default_fps=30.0,
        default_human_height=1.78,
        nested_files=True,
        recursive_files=True,
        excluded_filenames=frozenset({"scene_reconstruction.npz"}),
        direct_orientation_npz=DirectOrientationNPZSpec(
            source_formats=frozenset({"mocap", "noetix_csv_mocap"}),
            coordinate_systems=frozenset({"z_up"}),
        ),
    ),
    "noetix_mocap": MotionFormatSpec(
        name="noetix_mocap",
        suffixes=(".npz",),
        task_types=frozenset({"robot_only"}),
        root_joint="Spine1",
        orientation_mode="bvh",
        default_fps=30.0,
        recursive_files=True,
        excluded_filenames=frozenset({"scene_reconstruction.npz"}),
        direct_orientation_npz=DirectOrientationNPZSpec(
            source_formats=frozenset({"noetix_mocap"}),
            coordinate_systems=frozenset({"z_up"}),
        ),
    ),
    "omomo": MotionFormatSpec(
        name="omomo",
        suffixes=(".pt",),
        task_types=frozenset({"robot_only", "object_interaction"}),
        root_joint="Pelvis",
        orientation_mode="smpl",
        default_fps=30.0,
    ),
}


def get_motion_format_spec(data_format: str) -> MotionFormatSpec:
    """Resolve aliases and return the format's explicit loading contract."""

    canonical = normalize_data_format(data_format)
    try:
        return MOTION_FORMATS[canonical]
    except KeyError as exc:
        raise ValueError(f"No motion loader registered for data_format={canonical!r}") from exc


def validate_motion_task(data_format: str, task_type: str) -> str:
    """Validate a format/task combination and return the canonical format."""

    spec = get_motion_format_spec(data_format)
    if task_type not in spec.task_types:
        supported = ", ".join(sorted(spec.task_types))
        raise ValueError(f"data_format={spec.name!r} supports task types: {supported}")
    return spec.name


def _npz_matches_format(path: Path, spec: MotionFormatSpec) -> bool:
    """Disambiguate formats that intentionally share the NPZ suffix."""
    if path.suffix.lower() != ".npz":
        return True
    if spec.name not in {"lafan", "mocap", "noetix_mocap"}:
        return True
    with np.load(path, allow_pickle=False) as data:
        source_format = str(np.asarray(data["source_format"]).item()).lower() if "source_format" in data else None
        if spec.name == "lafan":
            return source_format == "lafan"
        if spec.name == "mocap":
            if source_format is not None:
                return source_format in {"mocap", "noetix_csv_mocap"}
            if "joint_names" not in data:
                return False
            return [str(name) for name in np.asarray(data["joint_names"]).tolist()] == DEMO_JOINTS_REGISTRY["mocap"]
        if source_format is not None:
            return source_format in {
                "noetix_mocap",
                "noetix-mocap",
                "noetix_lafan",
            }
        if {"raw_joint_names", "source_bvh"}.intersection(data.files):
            return True
        if "joint_names" not in data:
            return False
        return [str(name) for name in np.asarray(data["joint_names"]).tolist()] == DEMO_JOINTS_REGISTRY["noetix_mocap"]


def _is_motion_file(path: Path, spec: MotionFormatSpec) -> bool:
    return path.is_file() and path.name not in spec.excluded_filenames and _npz_matches_format(path, spec)


def _is_path_candidate(path: Path, spec: MotionFormatSpec) -> bool:
    if not path.is_file() or path.name in spec.excluded_filenames:
        return False
    if path.suffix.lower() != ".npz" or spec.name not in {"lafan", "mocap", "noetix_mocap"}:
        return True
    with np.load(path, allow_pickle=False) as data:
        if "source_format" not in data:
            return True
        source_format = str(np.asarray(data["source_format"]).item()).lower()
    expected = {
        "lafan": {"lafan"},
        "mocap": {"mocap", "noetix_csv_mocap"},
        "noetix_mocap": {
            "noetix_mocap",
            "noetix-mocap",
            "noetix_lafan",
        },
    }
    return source_format in expected[spec.name]


def resolve_motion_path(data_dir: Path | str, task_name: str, data_format: str) -> Path:
    """Resolve one sequence path without guessing another format."""

    directory = Path(data_dir).expanduser()
    spec = get_motion_format_spec(data_format)
    if spec.nested_files:
        task_directory = directory / task_name
        for suffix in spec.suffixes:
            task_files = [path for path in sorted(task_directory.glob(f"*{suffix}")) if _is_path_candidate(path, spec)]
            if len(task_files) == 1:
                return task_files[0]
            if len(task_files) > 1:
                raise ValueError(
                    f"{spec.name} task directory contains multiple {suffix} motion files: {task_directory}"
                )
    for suffix in spec.suffixes:
        candidate = directory / f"{task_name}{suffix}"
        if _is_path_candidate(candidate, spec):
            return candidate
    if spec.recursive_files:
        for suffix in spec.suffixes:
            matches = [
                path for path in sorted(directory.rglob(f"{task_name}{suffix}")) if _is_path_candidate(path, spec)
            ]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                paths = ", ".join(str(path) for path in matches)
                raise ValueError(f"{spec.name} sequence {task_name!r} is ambiguous: {paths}")
    expected = ", ".join(str(directory / f"{task_name}{suffix}") for suffix in spec.suffixes)
    raise FileNotFoundError(f"{spec.name} motion file not found; expected one of: {expected}")


def discover_motion_files(
    data_dir: Path | str,
    data_format: str,
    object_name: str | None = None,
) -> list[Path]:
    """Discover source files using the same format registry as single-sequence loading."""

    directory = Path(data_dir).expanduser()
    if not directory.is_dir():
        raise FileNotFoundError(f"Motion data directory not found: {directory}")

    spec = get_motion_format_spec(data_format)
    selected_by_sequence: dict[str, Path] = {}
    for suffix in spec.suffixes:
        iterator = directory.rglob(f"*{suffix}") if spec.recursive_files else directory.glob(f"*{suffix}")
        candidates_by_sequence: dict[str, list[Path]] = {}
        for path in sorted(iterator):
            if not _is_motion_file(path, spec):
                continue
            if spec.nested_files:
                relative_parent = path.parent.relative_to(directory)
                sequence_id = path.stem if relative_parent == Path() else relative_parent.as_posix()
            elif spec.recursive_files:
                sequence_id = path.relative_to(directory).with_suffix("").as_posix()
            else:
                sequence_id = path.stem
            candidates_by_sequence.setdefault(sequence_id, []).append(path)
        for sequence_id, candidates in candidates_by_sequence.items():
            if sequence_id in selected_by_sequence:
                continue
            if len(candidates) > 1:
                paths = ", ".join(str(path) for path in candidates)
                raise ValueError(f"{spec.name} sequence {sequence_id!r} is not unique: {paths}")
            selected_by_sequence[sequence_id] = candidates[0]
    files = list(selected_by_sequence.values())
    if spec.name == "omomo" and object_name:
        files = [
            path
            for path in files
            if parse_omomo_sequence_name(path, require_known_object=False).object_name == object_name
        ]
    return sorted(files)


def _validate_motion(
    *,
    joints: np.ndarray,
    spec: MotionFormatSpec,
    source_path: Path,
    human_height: float | None,
    fps: float,
    object_poses: np.ndarray | None = None,
    orientation_joint_names: tuple[str, ...] | list[str] | np.ndarray | None = None,
    orientation_quaternions: np.ndarray | None = None,
    orientation_source: str | None = None,
) -> HumanMotion:
    canonical_joint_names = tuple(DEMO_JOINTS_REGISTRY[spec.name])
    expected_joints = len(canonical_joint_names)
    joints = np.asarray(joints, dtype=np.float32)
    if joints.ndim != 3 or joints.shape[1:] != (expected_joints, 3):
        raise ValueError(f"{spec.name} joints must have shape (T, {expected_joints}, 3), got {joints.shape}")
    if joints.shape[0] == 0 or not np.isfinite(joints).all():
        raise ValueError(f"{spec.name} joints must contain at least one finite frame")
    if human_height is None or not np.isfinite(human_height) or human_height <= 0:
        raise ValueError(f"{spec.name} human height must be positive and finite, got {human_height}")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"{spec.name} fps must be positive and finite, got {fps}")

    validated_object_poses = None
    if object_poses is not None:
        validated_object_poses = np.asarray(object_poses, dtype=np.float32)
        if validated_object_poses.shape != (joints.shape[0], 7):
            raise ValueError(
                f"{spec.name} object poses must have shape ({joints.shape[0]}, 7), got {validated_object_poses.shape}"
            )
        if not np.isfinite(validated_object_poses).all():
            raise ValueError(f"{spec.name} object poses contain NaN or Inf")

    orientation_fields_present = (
        orientation_joint_names is not None,
        orientation_quaternions is not None,
        orientation_source is not None,
    )
    if any(orientation_fields_present) and not all(orientation_fields_present):
        raise ValueError(
            f"{spec.name} orientation names, quaternions, and direct-source "
            "provenance must either all be present or all be absent"
        )

    validated_orientation_names: tuple[str, ...] | None = None
    validated_orientation_quaternions: np.ndarray | None = None
    validated_orientation_source: str | None = None
    if orientation_joint_names is not None and orientation_quaternions is not None and orientation_source is not None:
        validated_orientation_source = str(orientation_source).strip()
        approved_sources = APPROVED_DIRECT_ORIENTATION_SOURCES[spec.name]
        if validated_orientation_source not in approved_sources:
            raise ValueError(
                f"{spec.name} orientation_source must be one of the approved "
                f"direct-source values {sorted(approved_sources)}, got "
                f"{validated_orientation_source!r}"
            )
        orientation_names_array = np.asarray(orientation_joint_names)
        if orientation_names_array.ndim != 1:
            raise ValueError(f"{spec.name} orientation joint names must be one-dimensional")
        validated_orientation_names = tuple(str(name) for name in orientation_names_array.tolist())
        if not validated_orientation_names:
            raise ValueError(f"{spec.name} orientation joint subset must not be empty")
        if len(set(validated_orientation_names)) != len(validated_orientation_names):
            raise ValueError(f"{spec.name} orientation joint names must be unique")
        unknown_names = set(validated_orientation_names).difference(canonical_joint_names)
        if unknown_names:
            names = ", ".join(sorted(unknown_names))
            raise ValueError(f"{spec.name} orientation joints are not canonical: {names}")

        validated_orientation_quaternions = np.ascontiguousarray(
            orientation_quaternions,
            dtype=np.float32,
        ).copy()
        expected_shape = (
            joints.shape[0],
            len(validated_orientation_names),
            4,
        )
        if validated_orientation_quaternions.shape != expected_shape:
            raise ValueError(
                f"{spec.name} orientation quaternions must have shape "
                f"{expected_shape}, got {validated_orientation_quaternions.shape}"
            )
        norms = np.linalg.norm(
            validated_orientation_quaternions.astype(np.float64),
            axis=-1,
        )
        if not np.isfinite(validated_orientation_quaternions).all() or np.any(norms <= 1e-8):
            raise ValueError(f"{spec.name} orientation quaternions must be finite and non-zero")
        if np.any(np.abs(norms - 1.0) > 1e-3):
            raise ValueError(
                f"{spec.name} direct orientation quaternions must already be unit length",
            )

    return HumanMotion(
        joints=joints,
        fps=float(fps),
        human_height=float(human_height),
        source_path=source_path,
        object_poses_wxyz_xyz=validated_object_poses,
        orientation_joint_names=validated_orientation_names,
        orientation_quaternions_wxyz=validated_orientation_quaternions,
        orientation_source=validated_orientation_source,
        joint_parent_indices=DEMO_JOINT_PARENT_INDICES[spec.name],
        _canonical_joint_names=canonical_joint_names,
        _root_joint_name=spec.root_joint,
    )


def _read_scalar(data: np.lib.npyio.NpzFile, key: str) -> float:
    if key not in data:
        raise KeyError(f"NPZ file is missing required field {key!r}")
    return float(np.asarray(data[key]).item())


def _validate_joint_names(data: np.lib.npyio.NpzFile, spec: MotionFormatSpec) -> None:
    if "joint_names" not in data:
        raise KeyError(f"{spec.name} NPZ file is missing required field 'joint_names'")
    actual = [str(name) for name in np.asarray(data["joint_names"]).tolist()]
    expected = DEMO_JOINTS_REGISTRY[spec.name]
    if actual != expected:
        raise ValueError(f"{spec.name} joint_names do not match the registered joint order")


def _read_orientations(
    data: np.lib.npyio.NpzFile,
    spec: MotionFormatSpec,
) -> tuple[tuple[str, ...] | None, np.ndarray | None, str | None]:
    """Read only the canonical, explicitly proven direct-source field group."""
    orientation_source = _read_optional_string(data, "orientation_source")
    has_names = "orientation_joint_names" in data
    has_quaternions = "orientation_quaternions_wxyz" in data
    if has_names or has_quaternions or orientation_source is not None:
        if not (has_names and has_quaternions and orientation_source is not None):
            raise KeyError(
                f"{spec.name} NPZ must contain 'orientation_joint_names', "
                "'orientation_quaternions_wxyz', and approved "
                "'orientation_source' together"
            )
        _validate_direct_orientation_npz_metadata(data, spec)
        names_array = np.asarray(data["orientation_joint_names"])
        if names_array.ndim != 1:
            raise ValueError(f"{spec.name} orientation joint names must be one-dimensional")
        names = tuple(str(name) for name in names_array.tolist())
        return names, data["orientation_quaternions_wxyz"], orientation_source
    return None, None, None


def _validate_direct_orientation_npz_metadata(
    data: np.lib.npyio.NpzFile,
    spec: MotionFormatSpec,
) -> None:
    """Reject ambiguous conventions rather than converting or guessing them."""

    metadata_spec = spec.direct_orientation_npz
    if metadata_spec is None:
        raise ValueError(f"{spec.name} does not register direct-orientation NPZ metadata")

    source_format = _read_required_string(data, "source_format", spec)
    if source_format not in metadata_spec.source_formats:
        raise ValueError(
            f"{spec.name} direct-orientation NPZ source_format must be one of "
            f"{sorted(metadata_spec.source_formats)}, got {source_format!r}"
        )

    quaternion_convention = _read_required_string(
        data,
        "quaternion_convention",
        spec,
    )
    if quaternion_convention != "wxyz":
        raise ValueError(
            f"{spec.name} direct-orientation NPZ quaternion_convention must be 'wxyz', got {quaternion_convention!r}"
        )

    coordinate_system = _read_required_string(data, "coordinate_system", spec)
    if coordinate_system not in metadata_spec.coordinate_systems:
        raise ValueError(
            f"{spec.name} direct-orientation NPZ coordinate_system must be one "
            f"of {sorted(metadata_spec.coordinate_systems)}, got "
            f"{coordinate_system!r}"
        )

    orientation_coordinate_system = _read_optional_string(
        data,
        "orientation_coordinate_system",
    )
    if metadata_spec.require_orientation_coordinate_system and orientation_coordinate_system is None:
        raise KeyError(f"{spec.name} direct-orientation NPZ is missing required field 'orientation_coordinate_system'")
    if (
        orientation_coordinate_system is not None
        and orientation_coordinate_system not in metadata_spec.coordinate_systems
    ):
        raise ValueError(
            f"{spec.name} direct-orientation NPZ orientation_coordinate_system "
            f"must be one of {sorted(metadata_spec.coordinate_systems)}, got "
            f"{orientation_coordinate_system!r}"
        )


def _read_required_string(
    data: np.lib.npyio.NpzFile,
    key: str,
    spec: MotionFormatSpec,
) -> str:
    value = _read_optional_string(data, key)
    if value is None:
        raise KeyError(f"{spec.name} direct-orientation NPZ is missing required field {key!r}")
    return value


def _read_optional_string(
    data: np.lib.npyio.NpzFile,
    key: str,
) -> str | None:
    if key not in data:
        return None
    value = np.asarray(data[key])
    if value.ndim != 0:
        raise ValueError(f"NPZ field {key!r} must be a scalar string")
    result = str(value.item()).strip()
    if not result:
        raise ValueError(f"NPZ field {key!r} must not be empty")
    return result


def _load_amass(path: Path, spec: MotionFormatSpec, human_height: float | None) -> HumanMotion:
    with np.load(path, allow_pickle=False) as data:
        if "joint_names" in data:
            _validate_joint_names(data, spec)
        height = human_height if human_height is not None else _read_scalar(data, "height")
        fps = float(np.asarray(data["fps"]).item()) if "fps" in data else spec.default_fps
        orientation_names, orientations, orientation_source = _read_orientations(data, spec)
        return _validate_motion(
            joints=data["global_joint_positions"],
            spec=spec,
            source_path=path,
            human_height=height,
            fps=fps,
            orientation_joint_names=orientation_names,
            orientation_quaternions=orientations,
            orientation_source=orientation_source,
        )


def _load_gvhmr(path: Path, spec: MotionFormatSpec, human_height: float | None) -> HumanMotion:
    with np.load(path, allow_pickle=False) as data:
        _validate_joint_names(data, spec)
        height = human_height if human_height is not None else _read_scalar(data, "height")
        orientation_names, orientations, orientation_source = _read_orientations(data, spec)
        return _validate_motion(
            joints=data["global_joint_positions"],
            spec=spec,
            source_path=path,
            human_height=height,
            fps=_read_scalar(data, "fps"),
            orientation_joint_names=orientation_names,
            orientation_quaternions=orientations,
            orientation_source=orientation_source,
        )


def _load_lafan(path: Path, spec: MotionFormatSpec, human_height: float | None) -> HumanMotion:
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            _validate_joint_names(data, spec)
            orientation_names, orientations, orientation_source = _read_orientations(data, spec)
            height = (
                human_height
                if human_height is not None
                else (_read_scalar(data, "height") if "height" in data else spec.default_human_height)
            )
            fps = _read_scalar(data, "fps") if "fps" in data else spec.default_fps
            return _validate_motion(
                joints=data["global_joint_positions"],
                spec=spec,
                source_path=path,
                human_height=height,
                fps=fps,
                orientation_joint_names=orientation_names,
                orientation_quaternions=orientations,
                orientation_source=orientation_source,
            )

    joints_y_up = np.load(path, allow_pickle=False)
    joints_z_up = np.asarray(joints_y_up)[..., [0, 2, 1]]
    return _validate_motion(
        joints=joints_z_up,
        spec=spec,
        source_path=path,
        human_height=human_height if human_height is not None else spec.default_human_height,
        fps=spec.default_fps,
    )


def _load_mocap(path: Path, spec: MotionFormatSpec, human_height: float | None) -> HumanMotion:
    orientation_names = None
    orientations = None
    orientation_source = None
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            _validate_joint_names(data, spec)
            joints = np.asarray(data["global_joint_positions"])
            source_fps = _read_scalar(data, "fps")
            orientation_names, orientations, orientation_source = _read_orientations(data, spec)
            height = (
                human_height
                if human_height is not None
                else (_read_scalar(data, "height") if "height" in data else spec.default_human_height)
            )
    else:
        source_fps = 120.0
        joints = np.load(path, allow_pickle=False)
        height = human_height if human_height is not None else spec.default_human_height
    stride = max(1, round(source_fps / spec.default_fps))
    joints = joints[::stride]
    if orientations is not None:
        orientations = orientations[::stride]
    return _validate_motion(
        joints=joints,
        spec=spec,
        source_path=path,
        human_height=height,
        fps=source_fps / stride,
        orientation_joint_names=orientation_names,
        orientation_quaternions=orientations,
        orientation_source=orientation_source,
    )


def _load_noetix(path: Path, spec: MotionFormatSpec, human_height: float | None) -> HumanMotion:
    with np.load(path, allow_pickle=False) as data:
        _validate_joint_names(data, spec)
        height = human_height if human_height is not None else _read_scalar(data, "height")
        orientation_names, orientations, orientation_source = _read_orientations(data, spec)
        return _validate_motion(
            joints=data["global_joint_positions"],
            spec=spec,
            source_path=path,
            human_height=height,
            fps=_read_scalar(data, "fps"),
            orientation_joint_names=orientation_names,
            orientation_quaternions=orientations,
            orientation_source=orientation_source,
        )


def _load_omomo(path: Path, spec: MotionFormatSpec, human_height: float | None) -> HumanMotion:
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    if not torch.is_tensor(tensor) or tensor.ndim != 2 or tensor.shape[1] < 325:
        shape = tuple(tensor.shape) if torch.is_tensor(tensor) else None
        raise ValueError(f"OMOMO file must contain a tensor with shape (T, >=325), got {shape}")
    values = tensor.detach().to(device="cpu", dtype=torch.float32).numpy()
    joints = values[:, 162 : 162 + 52 * 3].reshape(-1, 52, 3)
    object_poses = values[:, 318:325][:, [6, 3, 4, 5, 0, 1, 2]]
    orientation_names = None
    orientations = None
    orientation_source = None
    if values.shape[1] >= 591:
        orientations_xyzw = values[:, 383:591].reshape(-1, 52, 4)
        orientations = orientations_xyzw[..., [3, 0, 1, 2]]
        orientation_names = tuple(DEMO_JOINTS_REGISTRY[spec.name])
        orientation_source = "intermimic_global_orientation_tensor"

    if human_height is None:
        height_file = path.parent.parent / "height_dict.pkl"
        if not height_file.is_file():
            raise FileNotFoundError(
                f"OMOMO subject heights not found: {height_file}. "
                "Pass --motion-data-config.human-height or place height_dict.pkl beside the dataset directory."
            )
        with height_file.open("rb") as file:
            height_by_subject = pickle.load(file)
        subject = path.stem.split("_", 1)[0]
        if not isinstance(height_by_subject, dict) or subject not in height_by_subject:
            raise KeyError(f"OMOMO height_dict.pkl has no entry for subject {subject!r}")
        human_height = float(height_by_subject[subject])

    return _validate_motion(
        joints=joints,
        spec=spec,
        source_path=path,
        human_height=human_height,
        fps=spec.default_fps,
        object_poses=object_poses,
        orientation_joint_names=orientation_names,
        orientation_quaternions=orientations,
        orientation_source=orientation_source,
    )


MotionLoader = Callable[[Path, MotionFormatSpec, float | None], HumanMotion]
_LOADERS: dict[str, MotionLoader] = {
    "amass": _load_amass,
    "gvhmr": _load_gvhmr,
    "lafan": _load_lafan,
    "mocap": _load_mocap,
    "noetix_mocap": _load_noetix,
    "omomo": _load_omomo,
}


def load_human_motion(
    data_format: str,
    data_dir: Path | str,
    task_name: str,
    human_height: float | None = None,
) -> HumanMotion:
    """Load one supported sequence through its registered adapter."""

    spec = get_motion_format_spec(data_format)
    path = resolve_motion_path(data_dir, task_name, spec.name)
    return _LOADERS[spec.name](path, spec, human_height)
