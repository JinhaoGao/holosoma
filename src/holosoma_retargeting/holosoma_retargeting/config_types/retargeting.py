"""Configuration types for retargeting (top-level config)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from holosoma_retargeting.config_types.data_type import MotionDataConfig
from holosoma_retargeting.config_types.retargeter import RetargeterConfig
from holosoma_retargeting.config_types.robot import RobotConfig
from holosoma_retargeting.config_types.task import TaskConfig


@dataclass
class RetargetingConfig:
    """Top-level retargeting configuration used by the Tyro CLI.

    This combines all configuration types needed for retargeting.
    """

    # --- Task type selection ---
    task_type: Literal["robot_only", "object_interaction", "climbing"] = "object_interaction"
    """Type of retargeting task."""

    # --- top-level run knobs ---
    robot: str = "g1"
    """Robot type. Use str to allow dynamic robot types via _ROBOT_DEFAULTS."""

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


@dataclass
class ParallelRetargetingConfig(RetargetingConfig):
    """Extended retargeting config for parallel processing.

    Adds parallel-specific fields while inheriting all retargeting config fields.
    This config is used for processing multiple files in parallel.
    """

    # Parallel processing specific fields
    data_dir: Path = Path("demo_data/OMOMO_new")
    """Directory containing input data files for parallel processing.
    This overrides data_path from RetargetingConfig when processing multiple files."""

    max_workers: int | None = None
    """Maximum number of parallel workers. Defaults to a safe bound of four."""

    object_names: tuple[str, ...] | None = None
    """Optional OMOMO object categories to process. None processes every category."""

    preflight: bool = True
    """Validate OMOMO motion data and all object assets before launching workers."""

    validate_input_tensors: bool = True
    """Load and validate every selected OMOMO tensor during preflight."""

    dry_run: bool = False
    """Write the batch manifest and report without running retargeting."""

    overwrite_existing: bool = False
    """Regenerate outputs that already exist instead of resuming around them."""

    report_path: Path | None = None
    """Optional JSON report path. Defaults to <save_dir>/batch_report.json."""
