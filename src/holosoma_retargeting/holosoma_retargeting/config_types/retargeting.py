# ruff: noqa: CPY001

"""Configuration types for retargeting (top-level config)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal, cast

import tyro

from holosoma_retargeting.config_types.data_type import MotionDataConfig
from holosoma_retargeting.config_types.retargeter import (
    RetargeterConfig,
    RootStabilityConfig,
    ShoulderDirectionConfig,
)
from holosoma_retargeting.config_types.robot import RobotConfig
from holosoma_retargeting.config_types.task import TaskConfig
from holosoma_retargeting.config_types.weight_profiles import (
    resolve_natural_pose,
    resolve_orientation_weights,
)

TaskType = Literal["robot_only", "object_interaction", "climbing"]
DatasetName = Literal[
    "climbing",
    "fbx_mocap",
    "gvhmr",
    "lafan",
    "noetix_csv_climb",
    "noetix_mocap",
    "OMOMO_new",
]

DATASET_DATA_FORMATS: dict[str, str] = {
    "climbing": "mocap",
    "fbx_mocap": "fbx_mocap",
    "gvhmr": "gvhmr",
    "lafan": "lafan",
    "noetix_csv_climb": "mocap",
    "noetix_mocap": "noetix_mocap",
    "OMOMO_new": "omomo",
}
_DEMO_DATA_ROOT = Path(__file__).resolve().parents[1] / "demo_data"
DATASET_DEFAULT_PATHS: dict[str, Path] = {
    "climbing": _DEMO_DATA_ROOT / "climb",
    "fbx_mocap": _DEMO_DATA_ROOT / "fbx_mocap",
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
    task_type = cast("TaskType", task)
    if dataset not in SUPPORTED_TASK_DATASETS[task_type]:
        supported = ", ".join(sorted(SUPPORTED_TASK_DATASETS[task_type]))
        raise ValueError(f"task={task!r} supports dataset presets: {supported}")
    if task != "robot_only" and robot != "g1":
        raise ValueError(f"task={task!r} is supported only on robot='g1'")


@dataclass(frozen=True)
class RetargeterRuntimeOptions:
    """Small public option group for live retargeting inspection."""

    visualize: bool = True
    """Show the live Viser viewer. Use --retargeter.no-visualize for headless runs."""

    debug: bool = False
    """Show mapped human/robot keypoints, hands, skeletons, and object points."""


@dataclass(frozen=True)
class RetargetingCommand:
    """Small user-facing command shared by both retargeting entry points."""

    task: TaskType = "object_interaction"
    """Retargeting task."""

    robot: Literal["g1", "e1", "e1_23dof", "e1_24dof", "e2"] = "g1"
    """Target robot."""

    dataset: DatasetName = "OMOMO_new"
    """Human motion dataset preset."""

    motion: str = "sub3_largebox_003"
    """One motion name, or its path relative to the dataset directory."""

    data_path: Path | None = None
    """Optional dataset directory override."""

    save_dir: Path | None = None
    """Optional result-root override."""

    output_name: str | None = None
    """Optional output filename. The .npz suffix may be omitted."""

    overwrite: bool = False
    """Replace an existing result for the exact same motion and configuration."""

    foot_sticking: Literal[True, False] = True
    """Enable or completely disable the foot-sticking XY hard constraints."""

    dynamic_ground_window: bool = True
    """Move the robot-only ground sampling window with the robot root."""

    interaction_mesh_weight: float = 10.0
    """Global Interaction Mesh deformation-energy weight."""

    arm_interaction_mesh_weight_scale: float = 1.0
    """Upper-limb mesh-anchor multiplier relative to the global weight."""

    root_position_weight: float = 0.0
    """Track the source root's aligned world translation when positive."""

    root_orientation_weight: float = 0.0
    """Track the source torso's aligned world orientation when positive."""

    retargeter: RetargeterRuntimeOptions = field(
        default_factory=RetargeterRuntimeOptions,
    )
    """Live visualization options."""

    orientation_weights: Annotated[
        float | None,
        tyro.conf.arg(aliases=("--orientation_weights",)),
    ] = None
    """Uniform non-negative weight for every mapped link. None keeps the
    orientation objective disabled."""

    orientation_config: Annotated[
        Path | None,
        tyro.conf.arg(aliases=("--orientation_config",)),
    ] = None
    """Robot-specific JSON file or directory with per-link orientation weights."""

    orientation_preview: bool = False
    """Save per-link frame-alignment overlays without enabling orientation costs."""

    shoulder_direction_tracking: bool = False
    """Use sequence-aware upper-arm directions instead of upper-limb SO(3)."""

    nature_weights: Annotated[
        float | None,
        tyro.conf.arg(aliases=("--nature_weights",)),
    ] = None
    """Uniform non-negative natural-pose weight for every actuated joint.
    None keeps the natural-pose objective disabled."""

    nature_config: Annotated[
        Path | None,
        tyro.conf.arg(aliases=("--nature_config",)),
    ] = None
    """Robot-specific JSON file or directory with natural references and
    per-joint weights."""


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
    orientation_weights = resolve_orientation_weights(
        robot=command.robot,
        dataset=dataset,
        data_format=data_format,
        uniform_weight=command.orientation_weights,
        config_path=command.orientation_config,
    )
    natural_pose = resolve_natural_pose(
        robot=command.robot,
        uniform_weight=command.nature_weights,
        config_path=command.nature_config,
    )
    return RetargetingConfig(
        task_type=command.task,
        robot=command.robot,
        dataset=dataset,
        data_format=data_format,
        task_name=command.motion,
        data_path=command.data_path or default_data_path,
        save_dir=command.save_dir,
        output_name=command.output_name,
        overwrite_existing=command.overwrite,
        retargeter=RetargeterConfig(
            visualize=command.retargeter.visualize,
            debug=command.retargeter.debug,
            dynamic_ground_window=command.dynamic_ground_window,
            interaction_mesh_weight=command.interaction_mesh_weight,
            arm_interaction_mesh_weight_scale=(
                command.arm_interaction_mesh_weight_scale
            ),
            root_stability=RootStabilityConfig(
                position_weight=command.root_position_weight,
                orientation_weight=command.root_orientation_weight,
            ),
            activate_foot_sticking=command.foot_sticking,
            orientation_weights=orientation_weights,
            orientation_preview=command.orientation_preview,
            shoulder_direction=ShoulderDirectionConfig(
                enable=command.shoulder_direction_tracking,
            ),
            natural_pose_joint_positions=natural_pose.references,
            natural_pose_weights=natural_pose.weights,
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
    (amass, fbx_mocap, lafan, omomo, noetix_mocap, gvhmr, or mocap)."""

    task_name: str = "sub3_largebox_003"
    """Name of the task/sequence."""

    data_path: Path = Path("demo_data/OMOMO_new")
    """Path to data directory."""

    save_dir: Path | None = None
    """Directory to save results. Auto-determined if None."""

    output_name: str | None = None
    """Optional filename override for the saved NPZ artifact."""

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
