# ruff: noqa: CPY001

"""Configuration types for retargeting (top-level config)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from holosoma_retargeting.config_types.data_type import MotionDataConfig
from holosoma_retargeting.config_types.retargeter import RetargeterConfig
from holosoma_retargeting.config_types.robot import RobotConfig
from holosoma_retargeting.config_types.task import TaskConfig

TaskType = Literal["robot_only", "object_interaction", "climbing"]
DatasetName = Literal[
    "climbing",
    "gvhmr",
    "lafan",
    "noetix_csv_climb",
    "noetix_mocap",
    "OMOMO_new",
]

DATASET_DATA_FORMATS: dict[str, str] = {
    "climbing": "mocap",
    "gvhmr": "gvhmr",
    "lafan": "lafan",
    "noetix_csv_climb": "mocap",
    "noetix_mocap": "noetix_mocap",
    "OMOMO_new": "omomo",
}

_DEMO_DATA_ROOT = Path(__file__).resolve().parents[1] / "demo_data"
DATASET_DEFAULT_PATHS: dict[str, Path] = {
    "climbing": _DEMO_DATA_ROOT / "climb",
    "gvhmr": _DEMO_DATA_ROOT / "gvhmr",
    "lafan": _DEMO_DATA_ROOT / "lafan",
    "noetix_csv_climb": _DEMO_DATA_ROOT / "noetix_csv_climb",
    "noetix_mocap": _DEMO_DATA_ROOT / "noetix_mocap",
    "OMOMO_new": _DEMO_DATA_ROOT / "OMOMO_new",
}

SUPPORTED_TASK_DATASETS: dict[TaskType, frozenset[str]] = {
    "robot_only": frozenset(DATASET_DATA_FORMATS),
    "object_interaction": frozenset({"OMOMO_new"}),
    "climbing": frozenset({"climbing", "noetix_csv_climb"}),
}


def validate_production_task(
    *,
    task: str,
    robot: str,
    dataset: str,
) -> None:
    """Reject task/robot/dataset combinations outside the production matrix."""

    if task not in SUPPORTED_TASK_DATASETS:
        choices = ", ".join(SUPPORTED_TASK_DATASETS)
        raise ValueError(f"Unknown task {task!r}; choose one of: {choices}")
    if dataset not in SUPPORTED_TASK_DATASETS[task]:
        supported = ", ".join(sorted(SUPPORTED_TASK_DATASETS[task]))
        raise ValueError(f"task={task!r} supports dataset presets: {supported}")
    if task != "robot_only" and robot != "g1":
        raise ValueError(f"task={task!r} is supported only on robot='g1'")


@dataclass(frozen=True)
class RetargetingCommand:
    """Small user-facing command shared by both retargeting entry points."""

    task: TaskType = "object_interaction"
    """Retargeting task."""

    robot: Literal["g1", "e1", "e2"] = "g1"
    """Target robot."""

    dataset: DatasetName = "OMOMO_new"
    """Human motion dataset preset."""

    motion: str = "sub3_largebox_003"
    """One motion name, or its path relative to the dataset directory."""

    data_path: Path | None = None
    """Optional dataset directory override."""

    save_dir: Path | None = None
    """Optional result-root override."""

    overwrite: bool = False
    """Replace an existing result for the exact same motion and configuration."""

    orientation: bool = False
    """Enable T-pose-calibrated link orientation tracking with equal weights."""

    orientation_config: Path | None = None
    """Optional JSON file containing per-human-keypoint or per-robot-link weights."""


def _orientation_weights_from_command(
    command: RetargetingCommand,
    *,
    data_format: str,
) -> dict[str, float]:
    """Resolve the compact orientation switch/profile into solver weights."""

    motion = MotionDataConfig(
        data_format=data_format,
        robot_type=command.robot,
    )
    mapping = motion.resolved_orientation_joints_mapping
    if not mapping:
        raise ValueError(
            f"dataset={command.dataset!r} has no orientation mapping for robot={command.robot!r}",
        )

    profile_enabled = False
    configured_weights: object = None
    if command.orientation_config is not None:
        profile_path = command.orientation_config.expanduser()
        if profile_path.suffix.lower() != ".json":
            raise ValueError("orientation_config must be a JSON file")
        try:
            with profile_path.open(encoding="utf-8") as profile_file:
                profile = json.load(profile_file)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read orientation_config {profile_path}: {exc}") from exc
        if not isinstance(profile, dict):
            raise ValueError("orientation_config must contain one JSON object")
        unknown_fields = sorted(set(profile).difference({"enabled", "weights"}))
        if unknown_fields:
            raise ValueError(f"orientation_config contains unknown fields: {unknown_fields}")
        profile_enabled = profile.get("enabled", True)
        if not isinstance(profile_enabled, bool):
            raise ValueError("orientation_config field 'enabled' must be a boolean")
        configured_weights = profile.get("weights")

    enabled = command.orientation or profile_enabled
    if not enabled:
        return {}
    if configured_weights is None:
        return dict.fromkeys(mapping, 1.0)
    if not isinstance(configured_weights, dict):
        raise ValueError("orientation_config field 'weights' must be an object")

    link_to_human = {robot_link: human_joint for human_joint, robot_link in mapping.items()}
    weights: dict[str, float] = {}
    for configured_name, raw_weight in configured_weights.items():
        if not isinstance(configured_name, str):
            raise ValueError("orientation_config weight names must be strings")
        if configured_name in mapping:
            human_joint = configured_name
        elif configured_name in link_to_human:
            human_joint = link_to_human[configured_name]
        else:
            raise ValueError(
                f"orientation_config weight {configured_name!r} is neither a mapped human keypoint nor robot link",
            )
        if human_joint in weights:
            raise ValueError(
                f"orientation_config defines {human_joint!r} more than once through keypoint/link aliases",
            )
        if isinstance(raw_weight, bool) or not isinstance(raw_weight, (int, float)):
            raise ValueError(f"orientation weight for {configured_name!r} must be a number")
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(
                f"orientation weight for {configured_name!r} must be finite and non-negative",
            )
        weights[human_joint] = weight
    if not weights or not any(weight > 0.0 for weight in weights.values()):
        raise ValueError("enabled orientation tracking requires at least one positive weight")
    return weights


def internal_config_from_command(command: RetargetingCommand) -> RetargetingConfig:
    """Resolve the compact public command into the internal solver config."""

    dataset = str(command.dataset)
    try:
        data_format = DATASET_DATA_FORMATS[dataset]
        default_data_path = DATASET_DEFAULT_PATHS[dataset]
    except KeyError as exc:
        choices = ", ".join(DATASET_DATA_FORMATS)
        raise ValueError(f"Unknown dataset preset {dataset!r}; choose one of: {choices}") from exc
    validate_production_task(
        task=command.task,
        robot=command.robot,
        dataset=dataset,
    )
    return RetargetingConfig(
        task_type=command.task,
        robot=command.robot,
        dataset=dataset,
        data_format=data_format,
        task_name=command.motion,
        data_path=command.data_path or default_data_path,
        save_dir=command.save_dir,
        overwrite_existing=command.overwrite,
        retargeter=RetargeterConfig(
            orientation_weights=_orientation_weights_from_command(
                command,
                data_format=data_format,
            ),
        ),
    )


@dataclass
class RetargetingConfig:
    """Top-level retargeting configuration used by the Tyro CLI.

    This combines all configuration types needed for retargeting.
    """

    # --- Task type selection ---
    task_type: TaskType = "object_interaction"
    """Type of retargeting task."""

    # --- top-level run knobs ---
    robot: str = "g1"
    """Robot type. Use str to allow dynamic robot types via _ROBOT_DEFAULTS."""

    dataset: str | None = None
    """User-facing dataset preset when invoked through a production command."""

    data_format: str | None = None
    """Motion data format. Auto-determined by task_type if None.
    Can be any format registered in DEMO_JOINTS_REGISTRY
    (amass, lafan, omomo, noetix_mocap, gvhmr, or mocap)."""

    task_name: str = "sub3_largebox_003"
    """Name of the task/sequence."""

    data_path: Path = Path("demo_data/OMOMO_new")
    """Path to data directory."""

    save_dir: Path | None = None
    """Directory to save results. Auto-determined if None."""

    augmentation: bool = False
    """Whether to use augmentation."""

    overwrite_existing: bool = False
    """Regenerate an existing canonical path. Without this explicit opt-in,
    exact artifacts resume and mismatched or invalid artifacts are rejected."""

    # --- Nested configs ---
    robot_config: RobotConfig = field(default_factory=lambda: RobotConfig(robot_type="g1"))
    """Robot configuration (nested - can override robot_urdf_file, robot_dof, etc.
    via --robot-config.robot-urdf-file)."""

    motion_data_config: MotionDataConfig = field(
        default_factory=lambda: MotionDataConfig(data_format="omomo", robot_type="g1")
    )
    """Motion data configuration (nested - can override demo_joints, joints_mapping, etc.
    via --motion-data-config.demo-joints).
    Note: data_format default will be set based on task_type in main()."""

    task_config: TaskConfig = field(default_factory=TaskConfig)
    """Task-specific configuration (nested - can override ground_size, surface_weight_threshold, etc.
    via --task-config.ground-size)."""

    retargeter: RetargeterConfig = field(default_factory=RetargeterConfig)
    """Retargeter configuration (nested - can override q_a_init_idx, activate_joint_limits, etc.
    via --retargeter.q-a-init-idx)."""
