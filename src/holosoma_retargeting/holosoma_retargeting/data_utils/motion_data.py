"""Unified loading and discovery for supported human-motion datasets."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import torch
from holosoma_retargeting.config_types.data_type import (
    DEMO_JOINTS_REGISTRY,
    normalize_data_format,
)
from holosoma_retargeting.data_utils.omomo import parse_omomo_sequence_name

OrientationMode = Literal["bvh", "smpl", "mocap"]


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


@dataclass(frozen=True)
class HumanMotion:
    """Validated source motion consumed by the retargeting pipeline."""

    joints: np.ndarray
    fps: float
    human_height: float
    source_path: Path
    object_poses_wxyz_xyz: np.ndarray | None = None
    root_quaternions_wxyz: np.ndarray | None = None
    global_joint_quaternions_wxyz: np.ndarray | None = None


MOTION_FORMATS: dict[str, MotionFormatSpec] = {
    "amass": MotionFormatSpec(
        name="amass",
        suffixes=(".npz",),
        task_types=frozenset({"robot_only"}),
        root_joint="Pelvis",
        orientation_mode="smpl",
        default_fps=30.0,
    ),
    "gvhmr": MotionFormatSpec(
        name="gvhmr",
        suffixes=(".npz",),
        task_types=frozenset({"robot_only"}),
        root_joint="Pelvis",
        orientation_mode="smpl",
        default_fps=30.0,
    ),
    "lafan": MotionFormatSpec(
        name="lafan",
        suffixes=(".npy",),
        task_types=frozenset({"robot_only"}),
        root_joint="Spine1",
        orientation_mode="bvh",
        default_fps=30.0,
        default_human_height=1.7,
    ),
    "mocap": MotionFormatSpec(
        name="mocap",
        suffixes=(".npy",),
        task_types=frozenset({"robot_only", "climbing"}),
        root_joint="Spine1",
        orientation_mode="mocap",
        default_fps=30.0,
        default_human_height=1.78,
        nested_files=True,
    ),
    "noetix_mocap": MotionFormatSpec(
        name="noetix_mocap",
        suffixes=(".npz",),
        task_types=frozenset({"robot_only"}),
        root_joint="Spine1",
        orientation_mode="bvh",
        default_fps=30.0,
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


def resolve_motion_path(data_dir: Path | str, task_name: str, data_format: str) -> Path:
    """Resolve one sequence path without guessing another format."""

    directory = Path(data_dir).expanduser()
    spec = get_motion_format_spec(data_format)
    if spec.nested_files:
        task_files = [
            path
            for suffix in spec.suffixes
            for path in sorted((directory / task_name).glob(f"*{suffix}"))
            if path.is_file()
        ]
        if len(task_files) == 1:
            return task_files[0]
        if len(task_files) > 1:
            raise ValueError(
                f"{spec.name} task directory must contain exactly one motion file: {directory / task_name}"
            )
    for suffix in spec.suffixes:
        candidate = directory / f"{task_name}{suffix}"
        if candidate.is_file():
            return candidate
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
    pattern_prefix = "*/*" if spec.nested_files else "*"
    selected_by_sequence: dict[str, Path] = {}
    for suffix in spec.suffixes:
        for path in sorted(directory.glob(f"{pattern_prefix}{suffix}")):
            if path.is_file():
                selected_by_sequence.setdefault(path.stem, path)
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
    root_quaternions: np.ndarray | None = None,
    global_joint_quaternions: np.ndarray | None = None,
) -> HumanMotion:
    expected_joints = len(DEMO_JOINTS_REGISTRY[spec.name])
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

    validated_root_quaternions = None
    if root_quaternions is not None:
        validated_root_quaternions = np.asarray(root_quaternions, dtype=np.float64)
        if validated_root_quaternions.shape != (joints.shape[0], 4):
            raise ValueError(
                f"{spec.name} root quaternions must have shape ({joints.shape[0]}, 4), "
                f"got {validated_root_quaternions.shape}"
            )
        norms = np.linalg.norm(validated_root_quaternions, axis=1)
        if not np.isfinite(validated_root_quaternions).all() or np.any(norms <= 1e-8):
            raise ValueError(f"{spec.name} root quaternions must be finite and non-zero")
        validated_root_quaternions = validated_root_quaternions / norms[:, None]

    validated_global_joint_quaternions = None
    if global_joint_quaternions is not None:
        validated_global_joint_quaternions = np.asarray(
            global_joint_quaternions,
            dtype=np.float64,
        )
        expected_shape = (joints.shape[0], expected_joints, 4)
        if validated_global_joint_quaternions.shape != expected_shape:
            raise ValueError(
                f"{spec.name} global joint quaternions must have shape "
                f"{expected_shape}, got {validated_global_joint_quaternions.shape}"
            )
        norms = np.linalg.norm(validated_global_joint_quaternions, axis=-1)
        if (
            not np.isfinite(validated_global_joint_quaternions).all()
            or np.any(norms <= 1e-8)
        ):
            raise ValueError(
                f"{spec.name} global joint quaternions must be finite and non-zero"
            )
        validated_global_joint_quaternions = (
            validated_global_joint_quaternions / norms[..., None]
        )

    return HumanMotion(
        joints=joints,
        fps=float(fps),
        human_height=float(human_height),
        source_path=source_path,
        object_poses_wxyz_xyz=validated_object_poses,
        root_quaternions_wxyz=validated_root_quaternions,
        global_joint_quaternions_wxyz=validated_global_joint_quaternions,
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


def _load_amass(path: Path, spec: MotionFormatSpec, human_height: float | None) -> HumanMotion:
    with np.load(path, allow_pickle=False) as data:
        if "joint_names" in data:
            _validate_joint_names(data, spec)
        height = human_height if human_height is not None else _read_scalar(data, "height")
        fps = float(np.asarray(data["fps"]).item()) if "fps" in data else spec.default_fps
        return _validate_motion(
            joints=data["global_joint_positions"],
            spec=spec,
            source_path=path,
            human_height=height,
            fps=fps,
            root_quaternions=data.get("root_quaternions_wxyz"),
        )


def _load_gvhmr(path: Path, spec: MotionFormatSpec, human_height: float | None) -> HumanMotion:
    with np.load(path, allow_pickle=False) as data:
        _validate_joint_names(data, spec)
        height = human_height if human_height is not None else _read_scalar(data, "height")
        return _validate_motion(
            joints=data["global_joint_positions"],
            spec=spec,
            source_path=path,
            human_height=height,
            fps=_read_scalar(data, "fps"),
            root_quaternions=data["root_quaternions_wxyz"],
        )


def _load_lafan(path: Path, spec: MotionFormatSpec, human_height: float | None) -> HumanMotion:
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
    source_fps = 120.0
    stride = round(source_fps / spec.default_fps)
    joints = np.load(path, allow_pickle=False)[::stride]
    return _validate_motion(
        joints=joints,
        spec=spec,
        source_path=path,
        human_height=human_height if human_height is not None else spec.default_human_height,
        fps=source_fps / stride,
    )


def _load_noetix(path: Path, spec: MotionFormatSpec, human_height: float | None) -> HumanMotion:
    with np.load(path, allow_pickle=False) as data:
        _validate_joint_names(data, spec)
        height = human_height if human_height is not None else _read_scalar(data, "height")
        return _validate_motion(
            joints=data["global_joint_positions"],
            spec=spec,
            source_path=path,
            human_height=height,
            fps=_read_scalar(data, "fps"),
            global_joint_quaternions=data.get(
                "global_joint_quaternions_wxyz"
            ),
        )


def _load_omomo(path: Path, spec: MotionFormatSpec, human_height: float | None) -> HumanMotion:
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    if not torch.is_tensor(tensor) or tensor.ndim != 2 or tensor.shape[1] < 325:
        shape = tuple(tensor.shape) if torch.is_tensor(tensor) else None
        raise ValueError(f"OMOMO file must contain a tensor with shape (T, >=325), got {shape}")
    values = tensor.detach().to(device="cpu", dtype=torch.float32).numpy()
    joints = values[:, 162 : 162 + 52 * 3].reshape(-1, 52, 3)
    object_poses = values[:, 318:325][:, [6, 3, 4, 5, 0, 1, 2]]

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
