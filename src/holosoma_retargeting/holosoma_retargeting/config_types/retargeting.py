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
from holosoma_retargeting.config_types.robot_profiles import (
    load_robot_profile,
    resolve_natural_pose,
    resolve_orientation_weights,
)
from holosoma_retargeting.config_types.task import TaskConfig

TaskType = Literal["robot_only", "object_interaction", "climbing"]
RobotName = Literal["g1", "e1", "e1_23dof", "e1_24dof", "e2"]
DatasetName = Literal[
    "amass",
    "climbing",
    "fbx_mocap",
    "gvhmr",
    "lafan",
    "noetix_csv_climb",
    "noetix_mocap",
    "OMOMO_new",
]

DATASET_DATA_FORMATS: dict[str, str] = {
    "amass": "amass",
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
    "amass": _DEMO_DATA_ROOT / "amass_smplx_processed",
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

    robot: RobotName = "g1"
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

    robot_profile: Annotated[
        Path | None,
        tyro.conf.arg(aliases=("--robot_profile",)),
    ] = None
    """Optional robot-profile JSON file or directory override."""

    robot_dof: int | None = None
    """Override the robot-profile actuated degree-of-freedom count."""

    robot_height: float | None = None
    """Override the robot-profile height in meters."""

    robot_urdf_file: Path | None = None
    """Override the robot-profile URDF path."""

    foot_sticking: Literal[True, False] | None = None
    """Override contact-aware planar foot constraints."""

    dynamic_ground_window: bool | None = None
    """Override whether the ground window follows the robot root."""

    interaction_mesh_weight: float | None = None
    """Override the global Interaction Mesh deformation-energy weight."""

    arm_interaction_mesh_weight_scale: float | None = None
    """Override the upper-limb Interaction Mesh row multiplier."""

    root_position_weight: float | None = None
    """Track the source root's aligned world translation when positive."""

    root_orientation_weight: float | None = None
    """Track the source root's aligned world orientation when positive."""

    q_a_init_idx: int | None = None
    """Override the first optimized qpos offset."""

    activate_joint_limits: bool | None = None
    """Override joint-limit enforcement."""

    activate_obj_non_penetration: bool | None = None
    """Override object non-penetration enforcement."""

    penetration_tolerance: float | None = None
    """Override non-penetration tolerance in meters."""

    foot_sticking_tolerance: float | None = None
    """Override planted-foot XY tolerance in meters."""

    step_size: float | None = None
    """Override the SQP trust-region radius."""

    w_nominal_tracking_init: float | None = None
    """Override the initial nominal-pose tracking weight."""

    nominal_tracking_tau: float | None = None
    """Override the nominal-pose tracking decay constant."""

    retargeter: RetargeterRuntimeOptions = field(
        default_factory=RetargeterRuntimeOptions,
    )
    """Live visualization options."""

    orientation_weights: Annotated[
        float | None,
        tyro.conf.arg(aliases=("--orientation_weights",)),
    ] = None
    """Override all mapped orientation weights with one value."""

    orientation_tracking: bool | None = None
    """Override the robot-profile orientation objective switch."""

    orientation_preview: bool | None = None
    """Override saving of orientation alignment diagnostics."""

    shoulder_direction_tracking: bool | None = None
    """Override direct upper-arm direction tracking."""

    shoulder_direction_weight: float | None = None
    """Override the direct upper-arm direction residual weight."""

    shoulder_wrist_axis_weight_scale: float | None = None
    """Override the retained wrist-axis orientation weight scale."""

    natural_pose_tracking: bool | None = None
    """Override the robot-profile natural-pose objective switch."""

    nature_weights: Annotated[
        float | None,
        tyro.conf.arg(aliases=("--nature_weights",)),
    ] = None
    """Override every natural-pose joint weight with one value."""


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
    profile = load_robot_profile(
        robot=command.robot,
        config_path=command.robot_profile,
    )
    defaults = profile.retargeting
    orientation_enabled = (
        profile.orientation_enabled if command.orientation_tracking is None else command.orientation_tracking
    )
    orientation_weights = resolve_orientation_weights(
        profile=profile,
        dataset=dataset,
        data_format=data_format,
        enabled=orientation_enabled,
        uniform_weight=command.orientation_weights,
        robot=command.robot,
    )
    natural_pose_enabled = (
        profile.natural_pose_enabled if command.natural_pose_tracking is None else command.natural_pose_tracking
    )
    natural_pose = resolve_natural_pose(
        profile=profile,
        enabled=natural_pose_enabled,
        uniform_weight=command.nature_weights,
    )
    robot_dof = profile.robot_dof if command.robot_dof is None else command.robot_dof
    robot_height = profile.robot_height if command.robot_height is None else command.robot_height
    if robot_dof <= 0:
        raise ValueError("robot_dof must be positive")
    if robot_height <= 0.0:
        raise ValueError("robot_height must be positive")
    robot_urdf_file = (
        profile.robot_urdf_file if command.robot_urdf_file is None else str(command.robot_urdf_file.expanduser())
    )
    shoulder_enabled = (
        profile.shoulder_direction.enable
        if command.shoulder_direction_tracking is None
        else command.shoulder_direction_tracking
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
        robot_config=RobotConfig(
            robot_type=command.robot,
            robot_dof=robot_dof,
            robot_height=robot_height,
            robot_urdf_file=robot_urdf_file,
        ),
        retargeter=RetargeterConfig(
            visualize=command.retargeter.visualize,
            debug=command.retargeter.debug,
            dynamic_ground_window=(
                defaults.dynamic_ground_window
                if command.dynamic_ground_window is None
                else command.dynamic_ground_window
            ),
            interaction_mesh_weight=(
                defaults.interaction_mesh_weight
                if command.interaction_mesh_weight is None
                else command.interaction_mesh_weight
            ),
            arm_interaction_mesh_weight_scale=(
                defaults.arm_interaction_mesh_weight_scale
                if command.arm_interaction_mesh_weight_scale is None
                else command.arm_interaction_mesh_weight_scale
            ),
            root_stability=RootStabilityConfig(
                position_weight=(
                    defaults.root_position_weight
                    if command.root_position_weight is None
                    else command.root_position_weight
                ),
                orientation_weight=(
                    defaults.root_orientation_weight
                    if command.root_orientation_weight is None
                    else command.root_orientation_weight
                ),
            ),
            q_a_init_idx=(defaults.q_a_init_idx if command.q_a_init_idx is None else command.q_a_init_idx),
            activate_joint_limits=(
                defaults.activate_joint_limits
                if command.activate_joint_limits is None
                else command.activate_joint_limits
            ),
            activate_obj_non_penetration=(
                defaults.activate_obj_non_penetration
                if command.activate_obj_non_penetration is None
                else command.activate_obj_non_penetration
            ),
            activate_foot_sticking=(defaults.foot_sticking if command.foot_sticking is None else command.foot_sticking),
            penetration_tolerance=(
                defaults.penetration_tolerance
                if command.penetration_tolerance is None
                else command.penetration_tolerance
            ),
            foot_sticking_tolerance=(
                defaults.foot_sticking_tolerance
                if command.foot_sticking_tolerance is None
                else command.foot_sticking_tolerance
            ),
            planar_foot_contact=profile.planar_foot_contact,
            step_size=(defaults.step_size if command.step_size is None else command.step_size),
            w_nominal_tracking_init=(
                defaults.w_nominal_tracking_init
                if command.w_nominal_tracking_init is None
                else command.w_nominal_tracking_init
            ),
            nominal_tracking_tau=(
                defaults.nominal_tracking_tau if command.nominal_tracking_tau is None else command.nominal_tracking_tau
            ),
            orientation_weights=orientation_weights,
            orientation_preview=(
                profile.orientation_preview if command.orientation_preview is None else command.orientation_preview
            ),
            shoulder_direction=ShoulderDirectionConfig(
                enable=shoulder_enabled,
                direction_weight=(
                    profile.shoulder_direction.direction_weight
                    if command.shoulder_direction_weight is None
                    else command.shoulder_direction_weight
                ),
                wrist_axis_weight_scale=(
                    profile.shoulder_direction.wrist_axis_weight_scale
                    if command.shoulder_wrist_axis_weight_scale is None
                    else command.shoulder_wrist_axis_weight_scale
                ),
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
