# ruff: noqa: CPY001

"""Stage, validate, and explicitly promote the complete demo result matrix."""

from __future__ import annotations

import ctypes
import errno
import fcntl
import hashlib
import json
import os
import re
import stat
import tempfile
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import mujoco
import numpy as np
import tyro

from holosoma_retargeting.config_types.data_type import MotionDataConfig
from holosoma_retargeting.config_types.retargeter import RetargeterConfig
from holosoma_retargeting.config_types.retargeting import (
    ParallelRetargetingConfig,
    RetargetingConfig,
)
from holosoma_retargeting.config_types.robot import RobotConfig
from holosoma_retargeting.config_types.task import TaskConfig
from holosoma_retargeting.data_utils.motion_data import (
    discover_motion_files,
    load_human_motion,
    validate_motion_skeleton_contract,
)
from holosoma_retargeting.examples import parallel_robot_retarget
from holosoma_retargeting.result_artifact import (
    HUMAN_ORIENTATION_KEYS,
    INTERACTION_MESH_KEYS,
    ResultArtifactValidationError,
    compute_human_orientation_sha256,
    compute_object_asset_manifest,
    validate_result_artifact,
    validate_result_external_assets,
)
from holosoma_retargeting.retargeting_pipeline import (
    build_retarget_job,
    create_task_constants,
    planned_variants,
    resolve_task_object_name,
    results_state_lock_path,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PACKAGE_ROOT / "demo_data"
DEFAULT_RESULTS_ROOT = PACKAGE_ROOT / "demo_results"
DEFAULT_RUN_ID = "demo-rebuild-v1"
SUPPORTED_ROBOTS = ("g1", "e1")
HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
MAX_REPORTED_ISSUES = 200
PROMOTION_JOURNAL_SCHEMA_VERSION = 1
PROMOTION_JOURNAL_FILE_NAME = "promotion.json"
REBUILD_CONTRACT_SCHEMA_VERSION = 2
STRUCTURAL_E1_FAILURE_TYPES = frozenset(
    {
        "SQPNonlinearFeasibilityError",
    }
)
E1_DEFAULT_OMISSION_TASK_TYPES = frozenset(
    {
        "object_interaction",
        "climbing",
    }
)
FORMAL_QUALITY_LIMITS = {
    "max_fallback_frame_fraction_per_artifact": 0.10,
    "max_release_union_frame_fraction_per_artifact": 0.10,
    "max_full_sequence_retry_fraction_per_artifact": 1.0,
    "max_fallback_artifact_fraction_per_group": 0.10,
    "max_foot_release_artifact_fraction_per_group": 0.10,
    "max_object_release_artifact_fraction_per_group": 0.10,
    "max_release_union_artifact_fraction_per_group": 0.10,
    "max_full_sequence_retry_artifact_fraction_per_group": 0.02,
}


@dataclass(frozen=True)
class DatasetSpec:
    """One stable dataset/task row in the canonical rebuild matrix."""

    dataset_id: str
    relative_data_dir: str | None
    task_type: str
    data_format: str
    augmentation: bool
    optional: bool = False


DATASET_SPECS = (
    DatasetSpec(
        dataset_id="omomo_robot_only",
        relative_data_dir="OMOMO_new",
        task_type="robot_only",
        data_format="omomo",
        augmentation=False,
    ),
    DatasetSpec(
        dataset_id="omomo_object_interaction",
        relative_data_dir="OMOMO_new",
        task_type="object_interaction",
        data_format="omomo",
        augmentation=True,
    ),
    DatasetSpec(
        dataset_id="amass",
        relative_data_dir="amass_smplx_processed",
        task_type="robot_only",
        data_format="amass",
        augmentation=False,
    ),
    DatasetSpec(
        dataset_id="gvhmr",
        relative_data_dir="gvhmr",
        task_type="robot_only",
        data_format="gvhmr",
        augmentation=False,
    ),
    DatasetSpec(
        dataset_id="lafan",
        relative_data_dir="lafan",
        task_type="robot_only",
        data_format="lafan",
        augmentation=False,
    ),
    DatasetSpec(
        dataset_id="noetix_bvh",
        relative_data_dir="noetix_mocap",
        task_type="robot_only",
        data_format="noetix_mocap",
        augmentation=False,
    ),
    DatasetSpec(
        dataset_id="generic_climb",
        relative_data_dir="climb",
        task_type="climbing",
        data_format="mocap",
        augmentation=True,
    ),
    DatasetSpec(
        dataset_id="noetix_csv_climb",
        relative_data_dir="noetix_csv_climb",
        task_type="climbing",
        data_format="mocap",
        augmentation=True,
    ),
)
DATASET_SPEC_BY_ID = {spec.dataset_id: spec for spec in DATASET_SPECS}
CORE_DATASET_IDS = frozenset(spec.dataset_id for spec in DATASET_SPECS if not spec.optional)


@dataclass
class RebuildDemoResultsConfig:
    """Configuration for the complete staged demo rebuild."""

    data_root: Path = DEFAULT_DATA_ROOT
    """Root containing the canonical converted demo datasets."""

    results_root: Path = DEFAULT_RESULTS_ROOT
    """Root containing staging, versioned results, archives, and run reports."""

    noetix_csv_root: Path | None = None
    """Override for converted Noetix CSV climbing data; defaults to data_root/noetix_csv_climb."""

    run_id: str = DEFAULT_RUN_ID
    """Stable staging and report identifier. Reuse it to resume the same rebuild."""

    robots: tuple[str, ...] = SUPPORTED_ROBOTS
    """Robot subset. Supported values are g1 and e1."""

    datasets: tuple[str, ...] = ()
    """Dataset IDs to run. Empty selects the complete matrix."""

    include_augmentation_variants: bool = True
    """Include canonical augmentation variants for augmentation-enabled datasets.

    Disable this to rebuild exactly one identity artifact per source while
    retaining the same dataset/task matrix and formal validation policy.
    """

    max_workers: int = 4
    """Worker count passed to each parallel batch."""

    overwrite: bool = False
    """Regenerate valid matching artifacts instead of resuming them."""

    dry_run: bool = False
    """Write batch manifests and the aggregate plan without running solvers."""

    validate_only: bool = False
    """Skip solvers and validate the existing staging tree."""

    promote: bool = False
    """After successful validation, archive the current v1 and promote staging."""

    omomo_preflight: bool = True
    """Run the existing OMOMO data and object-asset preflight."""

    validate_input_tensors: bool = True
    """Load every OMOMO tensor during the existing preflight."""

    release_object_non_penetration_on_infeasible: bool = True
    """Allow the final per-frame object-collision fallback for G1.

    E1 always runs the strict constraint so a structural infeasibility becomes
    auditable failure evidence instead of a released low-quality artifact.
    Every G1 released frame remains explicit in artifact and quality statistics.
    """

    allow_e1_best_effort_omissions: bool = True
    """Accept explicitly reported E1 structural failures as audited omissions.

    This only applies to object-interaction and climbing tasks by default. Every
    source must still appear in a complete, identity-checked batch report, and
    every artifact that exists is validated normally.
    """

    allow_e1_robot_only_omissions: bool = False
    """Also allow explicitly reported E1 robot-only failures as temporary omissions.

    This is deliberately separate and disabled by default because E1 robot-only
    coverage is expected to be substantially stronger than object interaction
    and climbing coverage.
    """

    max_fallback_frame_fraction_per_artifact: float = FORMAL_QUALITY_LIMITS["max_fallback_frame_fraction_per_artifact"]
    """Maximum foot-sticking fallback-frame fraction in one eligible artifact."""

    max_release_union_frame_fraction_per_artifact: float = FORMAL_QUALITY_LIMITS[
        "max_release_union_frame_fraction_per_artifact"
    ]
    """Maximum unique foot/object release-frame fraction in one eligible artifact."""

    max_full_sequence_retry_fraction_per_artifact: float = FORMAL_QUALITY_LIMITS[
        "max_full_sequence_retry_fraction_per_artifact"
    ]
    """Maximum full-sequence retry fraction in one eligible artifact.

    A complete-sequence retry affects the whole saved artifact, so its observed
    fraction is either zero or one. Set this below one to reject every retry.
    """

    max_fallback_artifact_fraction_per_group: float = FORMAL_QUALITY_LIMITS["max_fallback_artifact_fraction_per_group"]
    """Maximum fallback-affected fraction among foot-eligible artifacts per group."""

    max_foot_release_artifact_fraction_per_group: float = FORMAL_QUALITY_LIMITS[
        "max_foot_release_artifact_fraction_per_group"
    ]
    """Maximum foot-release fraction among foot-eligible artifacts per group."""

    max_object_release_artifact_fraction_per_group: float = FORMAL_QUALITY_LIMITS[
        "max_object_release_artifact_fraction_per_group"
    ]
    """Maximum object-release fraction among truly object-eligible artifacts per group."""

    max_release_union_artifact_fraction_per_group: float = FORMAL_QUALITY_LIMITS[
        "max_release_union_artifact_fraction_per_group"
    ]
    """Maximum foot/object release-affected artifact fraction per eligible group."""

    max_full_sequence_retry_artifact_fraction_per_group: float = FORMAL_QUALITY_LIMITS[
        "max_full_sequence_retry_artifact_fraction_per_group"
    ]
    """Maximum complete-sequence retry fraction among foot-eligible artifacts per group."""


@dataclass(frozen=True)
class BatchPlan:
    """One robot/dataset invocation of the existing parallel entry point."""

    robot: str
    spec: DatasetSpec
    data_dir: Path | None
    skip_reason: str | None = None

    @property
    def batch_id(self) -> str:
        return f"{self.robot}-{self.spec.dataset_id}"

    @property
    def available(self) -> bool:
        return self.data_dir is not None and self.data_dir.is_dir()


@dataclass(frozen=True)
class ExpectedArtifact:
    """Exact persisted identity of one planned output."""

    output_path: Path
    source_path: Path
    source_sha256: str
    config_sha256: str
    robot: str
    task_type: str
    data_format: str
    variant: str
    run_kind: str
    dataset_id: str
    frame_count: int
    human_joint_names: tuple[str, ...]
    human_joint_parent_indices: tuple[int, ...]
    human_orientation_joint_names: tuple[str, ...] | None
    human_orientation_source: str
    human_orientation_sha256: str | None
    human_orientation_tensor_shape: tuple[int, int, int] | None
    human_orientation_tensor_bytes: bytes | None = field(
        repr=False,
        compare=False,
    )
    robot_link_names: tuple[str, ...]
    robot_link_parent_indices: tuple[int, ...]


@dataclass(frozen=True)
class _SourceContract:
    """Adapter-derived fields that every variant of one source must retain."""

    frame_count: int
    human_joint_names: tuple[str, ...]
    human_joint_parent_indices: tuple[int, ...]
    human_orientation_joint_names: tuple[str, ...] | None
    human_orientation_source: str
    human_orientation_sha256: str | None
    human_orientation_tensor_shape: tuple[int, int, int] | None
    human_orientation_tensor_bytes: bytes | None = field(
        repr=False,
        compare=False,
    )


@dataclass(frozen=True)
class _RobotContract:
    """MuJoCo-derived complete robot body topology."""

    link_names: tuple[str, ...]
    parent_indices: tuple[int, ...]


@dataclass(frozen=True)
class ValidationIssue:
    """One bounded validation diagnostic."""

    code: str
    path: str
    message: str


@dataclass(frozen=True)
class ValidationSummary:
    """Auditable aggregate validation outcome."""

    ok: bool
    root: str
    artifact_count: int
    expected_artifact_count: int
    expected_source_count: int
    unexpected_artifact_count: int
    tree_sha256: str
    tree_file_count: int
    tree_byte_count: int
    external_asset_sha256: str
    external_asset_file_count: int
    external_asset_byte_count: int
    issue_count: int
    issues: tuple[ValidationIssue, ...]
    statistics: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["issues"] = [asdict(issue) for issue in self.issues]
        return payload


@dataclass(frozen=True)
class PromotionResult:
    """Paths affected by one successful atomic promotion."""

    promoted_root: Path
    archived_root: Path | None
    journal_path: Path


@dataclass(frozen=True)
class _DirectoryFingerprint:
    """Exact immutable identity of every regular file in a result tree."""

    sha256: str
    file_count: int
    byte_count: int


@dataclass(frozen=True)
class _BatchInvocation:
    """Freshness and exception evidence from one batch invocation."""

    report_refreshed: bool
    error: str | None


@dataclass(frozen=True)
class _ExecutionAudit:
    """Trusted source attempts and the exact missing artifacts they justify."""

    ok: bool
    omitted_artifact_paths: frozenset[Path]
    blocking_failures: tuple[Mapping[str, str], ...]
    report: Mapping[str, Any]


class DemoRebuildError(RuntimeError):
    """Raised after an aggregate report records a failed rebuild."""


class ArtifactPlanCollisionError(ValueError):
    """Raised when distinct planned jobs resolve to the same artifact path."""

    def __init__(
        self,
        collision_paths: Iterable[Path],
        *,
        expected_artifact_count: int,
        expected_source_count: int,
    ) -> None:
        self.collision_paths = tuple(Path(path) for path in collision_paths)
        self.expected_artifact_count = expected_artifact_count
        self.expected_source_count = expected_source_count
        paths = ", ".join(str(path) for path in self.collision_paths[:5])
        super().__init__(f"Rebuild matrix produces {len(self.collision_paths)} duplicate output path(s): {paths}")


class _IssueCollector:
    def __init__(self) -> None:
        self.count = 0
        self.samples: list[ValidationIssue] = []

    def add(self, code: str, path: Path | str, message: str) -> None:
        self.count += 1
        if len(self.samples) < MAX_REPORTED_ISSUES:
            self.samples.append(ValidationIssue(code=code, path=str(path), message=message))


def staging_root(results_root: Path, run_id: str) -> Path:
    """Return the stable version root used by a resumable rebuild."""

    return Path(results_root).expanduser().resolve() / ".staging" / run_id / "v1"


def run_report_root(results_root: Path, run_id: str) -> Path:
    """Return the persistent report directory outside the staged result tree."""

    return Path(results_root).expanduser().resolve() / "runs" / run_id


def promotion_journal_path(results_root: Path, run_id: str) -> Path:
    """Return the durable publication transaction journal path."""

    return run_report_root(results_root, run_id) / PROMOTION_JOURNAL_FILE_NAME


def _validate_config(cfg: RebuildDemoResultsConfig) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", cfg.run_id):
        raise ValueError(
            "run_id must start with an alphanumeric character and contain only letters, digits, '.', '_', or '-'"
        )
    if cfg.max_workers <= 0:
        raise ValueError("max_workers must be greater than zero")
    if cfg.dry_run and cfg.validate_only:
        raise ValueError("dry_run and validate_only are mutually exclusive")
    if cfg.dry_run and cfg.promote:
        raise ValueError("dry_run cannot promote results")
    if cfg.allow_e1_robot_only_omissions and not cfg.allow_e1_best_effort_omissions:
        raise ValueError("allow_e1_robot_only_omissions requires allow_e1_best_effort_omissions")
    unknown_robots = sorted(set(cfg.robots).difference(SUPPORTED_ROBOTS))
    if unknown_robots:
        raise ValueError(f"Unsupported robot filters: {', '.join(unknown_robots)}")
    if not cfg.robots:
        raise ValueError("robots must select at least one robot")
    quality_fractions = {
        "max_fallback_frame_fraction_per_artifact": (cfg.max_fallback_frame_fraction_per_artifact),
        "max_release_union_frame_fraction_per_artifact": (cfg.max_release_union_frame_fraction_per_artifact),
        "max_full_sequence_retry_fraction_per_artifact": (cfg.max_full_sequence_retry_fraction_per_artifact),
        "max_fallback_artifact_fraction_per_group": (cfg.max_fallback_artifact_fraction_per_group),
        "max_foot_release_artifact_fraction_per_group": (cfg.max_foot_release_artifact_fraction_per_group),
        "max_object_release_artifact_fraction_per_group": (cfg.max_object_release_artifact_fraction_per_group),
        "max_release_union_artifact_fraction_per_group": (cfg.max_release_union_artifact_fraction_per_group),
        "max_full_sequence_retry_artifact_fraction_per_group": (
            cfg.max_full_sequence_retry_artifact_fraction_per_group
        ),
    }
    invalid_quality_fractions = {
        name: value for name, value in quality_fractions.items() if not np.isfinite(value) or not 0.0 <= value <= 1.0
    }
    if invalid_quality_fractions:
        raise ValueError(f"Rebuild quality fractions must be finite values in [0, 1]: {invalid_quality_fractions}")
    unknown_datasets = sorted(set(cfg.datasets).difference(DATASET_SPEC_BY_ID))
    if unknown_datasets:
        raise ValueError(f"Unknown dataset filters: {', '.join(unknown_datasets)}")
    results_root = cfg.results_root.expanduser().resolve()
    if results_root == Path(results_root.anchor):
        raise ValueError("results_root must not be a filesystem root")


def selected_dataset_ids(cfg: RebuildDemoResultsConfig) -> tuple[str, ...]:
    """Return a stable, duplicate-free dataset selection."""

    requested = set(cfg.datasets) if cfg.datasets else set(DATASET_SPEC_BY_ID)
    return tuple(spec.dataset_id for spec in DATASET_SPECS if spec.dataset_id in requested)


def build_rebuild_matrix(
    cfg: RebuildDemoResultsConfig,
) -> tuple[BatchPlan, ...]:
    """Build the explicit G1/E1 x dataset/task execution matrix."""

    _validate_config(cfg)
    data_root = cfg.data_root.expanduser().resolve()
    noetix_csv_root = (
        cfg.noetix_csv_root.expanduser().resolve()
        if cfg.noetix_csv_root is not None
        else data_root / "noetix_csv_climb"
    )
    selected = set(selected_dataset_ids(cfg))
    plans: list[BatchPlan] = []
    for robot in dict.fromkeys(cfg.robots):
        for spec in DATASET_SPECS:
            if spec.dataset_id not in selected:
                continue
            if spec.dataset_id == "noetix_csv_climb":
                data_dir = noetix_csv_root
                skip_reason = (
                    None if data_dir.is_dir() else f"Required converted Noetix CSV root does not exist: {data_dir}"
                )
            else:
                data_dir = data_root / str(spec.relative_data_dir)
                skip_reason = None if data_dir.is_dir() else f"Required dataset directory does not exist: {data_dir}"
            plans.append(
                BatchPlan(
                    robot=robot,
                    spec=spec,
                    data_dir=data_dir,
                    skip_reason=skip_reason,
                )
            )

    available_climb_roots = {
        plan.data_dir.name
        for plan in plans
        if plan.available
        and plan.spec.task_type == "climbing"
        and plan.spec.data_format == "mocap"
        and plan.data_dir is not None
    }
    available_climb_plans = [
        plan
        for plan in plans
        if plan.available and plan.spec.task_type == "climbing" and plan.spec.data_format == "mocap"
    ]
    if len(available_climb_roots) < len({plan.spec.dataset_id for plan in available_climb_plans}):
        raise ValueError(
            "Generic and Noetix CSV climbing roots must have distinct directory "
            "names because the canonical path uses the directory name as its partition"
        )
    return tuple(plans)


def _release_object_non_penetration_for_plan(
    cfg: RebuildDemoResultsConfig,
    plan: BatchPlan,
) -> bool:
    """Return the one formal object-release policy used by planning and execution."""

    return plan.robot == "g1" and cfg.release_object_non_penetration_on_infeasible


def _effective_augmentation(
    cfg: RebuildDemoResultsConfig,
    plan: BatchPlan,
) -> bool:
    """Return the single augmentation decision shared by planning and execution."""

    return plan.spec.augmentation and cfg.include_augmentation_variants


def _planned_variants_for_plan(
    cfg: RebuildDemoResultsConfig,
    plan: BatchPlan,
):
    """Return the exact variant family selected for one rebuild batch."""

    return planned_variants(
        plan.spec.task_type,
        augmentation=_effective_augmentation(cfg, plan),
    )


def _parallel_config(
    cfg: RebuildDemoResultsConfig,
    plan: BatchPlan,
    *,
    stage: Path,
    batch_report_path: Path,
) -> ParallelRetargetingConfig:
    if plan.data_dir is None:
        raise ValueError(f"{plan.batch_id} has no data directory")
    robot_config = RobotConfig(robot_type=plan.robot)
    motion_config = MotionDataConfig(
        data_format=plan.spec.data_format,
        robot_type=plan.robot,
    )
    return ParallelRetargetingConfig(
        task_type=plan.spec.task_type,
        robot=plan.robot,
        data_format=plan.spec.data_format,
        data_path=plan.data_dir,
        save_dir=stage,
        augmentation=_effective_augmentation(cfg, plan),
        robot_config=robot_config,
        motion_data_config=motion_config,
        task_config=TaskConfig(),
        retargeter=RetargeterConfig(
            release_object_non_penetration_on_infeasible=(
                _release_object_non_penetration_for_plan(
                    cfg,
                    plan,
                )
            ),
        ),
        data_dir=plan.data_dir,
        max_workers=cfg.max_workers,
        preflight=cfg.omomo_preflight,
        validate_input_tensors=cfg.validate_input_tensors,
        dry_run=cfg.dry_run,
        overwrite_existing=cfg.overwrite,
        report_path=batch_report_path,
        run_id=plan.batch_id,
    )


def _source_task_identity(
    source_path: Path,
    *,
    data_root: Path,
    task_type: str,
) -> tuple[str, Path, str, Path | None]:
    source_path = source_path.resolve()
    data_root = data_root.resolve()
    if task_type == "climbing":
        task_dir = source_path.parent
        sequence_key = task_dir.relative_to(data_root).as_posix()
        if sequence_key in {"", "."}:
            raise ValueError(f"Climbing source must be nested in one sequence directory: {source_path}")
        return task_dir.name, task_dir.parent, sequence_key, task_dir
    return (
        source_path.stem,
        source_path.parent,
        source_path.relative_to(data_root).with_suffix("").as_posix(),
        None,
    )


def _canonical_direct_orientation_tensor(
    values: np.ndarray,
) -> np.ndarray:
    """Return the exact float32 C-order tensor persisted by the solver."""

    tensor = np.asarray(values, dtype=np.float32)
    if tensor.ndim != 3 or tensor.shape[-1] != 4:
        raise ValueError(f"Direct orientation tensor must have shape (frames, joints, 4), got {tensor.shape}")
    return np.ascontiguousarray(tensor)


def _direct_orientation_tensor_sha256(
    joint_names: tuple[str, ...],
    tensor: np.ndarray,
) -> str:
    """Return the result schema's digest for the named float32 wxyz tensor."""

    canonical_tensor = _canonical_direct_orientation_tensor(tensor)
    return compute_human_orientation_sha256(
        joint_names,
        canonical_tensor,
    )


def _source_contract_from_adapter(
    source_path: Path,
    *,
    data_format: str,
    data_path: Path,
    task_name: str,
    human_height: float | None,
    human_joint_names: tuple[str, ...],
    human_joint_parent_indices: tuple[int, ...],
) -> _SourceContract:
    """Load one source through the production adapter and freeze its output contract."""

    motion = load_human_motion(
        data_format,
        data_path,
        task_name,
        human_height=human_height,
    )
    resolved_source = source_path.resolve()
    if motion.source_path.resolve() != resolved_source:
        raise ValueError(
            "Motion adapter resolved a different source while planning: "
            f"{motion.source_path.resolve()} != {resolved_source}"
        )
    validate_motion_skeleton_contract(
        motion,
        human_joint_names,
        human_joint_parent_indices,
        data_format=data_format,
    )
    canonical_names = motion.canonical_joint_names
    if not canonical_names or len(canonical_names) != motion.joints.shape[1]:
        raise ValueError(f"Motion adapter did not expose a complete canonical joint order for {resolved_source}")
    parent_indices = tuple(int(parent) for parent in motion.joint_parent_indices)
    if len(parent_indices) != len(canonical_names):
        raise ValueError(
            f"Motion configuration did not expose one parent index per canonical joint for {resolved_source}"
        )
    orientation_names = motion.orientation_joint_names
    if motion.orientation_quaternions_wxyz is None:
        if orientation_names is not None or motion.orientation_source is not None:
            raise ValueError(f"Motion adapter returned incomplete absent-orientation metadata for {resolved_source}")
        orientation_source = "absent"
        orientation_tensor_sha256 = None
        orientation_tensor_shape = None
        orientation_tensor_bytes = None
    else:
        if orientation_names is None:
            raise ValueError(f"Motion adapter returned orientations without names for {resolved_source}")
        orientation_names = tuple(orientation_names)
        orientation_source = motion.orientation_source or "explicit_source_orientation_fields"
        orientation_tensor = _canonical_direct_orientation_tensor(motion.orientation_quaternions_wxyz)
        if orientation_tensor.shape[:2] != (
            int(motion.joints.shape[0]),
            len(orientation_names),
        ):
            raise ValueError(
                "Motion adapter returned a direct orientation tensor whose "
                f"shape does not match its frames/names for {resolved_source}"
            )
        orientation_tensor_sha256 = _direct_orientation_tensor_sha256(
            orientation_names,
            orientation_tensor,
        )
        orientation_tensor_shape = tuple(int(size) for size in orientation_tensor.shape)
        orientation_tensor_bytes = orientation_tensor.tobytes(order="C")
    return _SourceContract(
        frame_count=int(motion.joints.shape[0]),
        human_joint_names=canonical_names,
        human_joint_parent_indices=parent_indices,
        human_orientation_joint_names=orientation_names,
        human_orientation_source=orientation_source,
        human_orientation_sha256=orientation_tensor_sha256,
        human_orientation_tensor_shape=orientation_tensor_shape,
        human_orientation_tensor_bytes=orientation_tensor_bytes,
    )


def _robot_contract_from_model(robot: str) -> _RobotContract:
    """Load the production MuJoCo model and reproduce the saved robot subtree."""

    constants = create_task_constants(
        robot_config=RobotConfig(robot_type=robot),
        motion_data_config=MotionDataConfig(
            data_format="amass",
            robot_type=robot,
        ),
        task_config=TaskConfig(),
        task_type="robot_only",
    )
    model_path = Path(constants.ROBOT_URDF_FILE).with_suffix(".xml")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    root_joint_candidates = [joint_id for joint_id in range(model.njnt) if int(model.jnt_qposadr[joint_id]) == 0]
    if len(root_joint_candidates) != 1:
        raise ValueError(f"{model_path} must contain exactly one robot root joint at qpos address 0")
    root_body_id = int(model.jnt_bodyid[root_joint_candidates[0]])
    body_ids: list[int] = []
    for body_id in range(1, model.nbody):
        ancestor = body_id
        while ancestor not in {0, root_body_id}:
            ancestor = int(model.body_parentid[ancestor])
        if ancestor == root_body_id:
            body_ids.append(body_id)
    if not body_ids or body_ids[0] != root_body_id:
        raise ValueError(f"Could not resolve the robot body subtree in {model_path}")
    link_names = tuple(
        str(name)
        for body_id in body_ids
        for name in [
            mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                body_id,
            )
        ]
        if name is not None
    )
    if len(link_names) != len(body_ids) or len(set(link_names)) != len(link_names):
        raise ValueError(f"Robot bodies in {model_path} must have unique names")
    local_index = {body_id: index for index, body_id in enumerate(body_ids)}
    parent_indices = tuple(
        -1 if body_id == root_body_id else local_index[int(model.body_parentid[body_id])] for body_id in body_ids
    )
    return _RobotContract(
        link_names=link_names,
        parent_indices=parent_indices,
    )


def plan_expected_artifacts(
    plans: Iterable[BatchPlan],
    *,
    stage: Path,
    cfg: RebuildDemoResultsConfig | None = None,
) -> tuple[ExpectedArtifact, ...]:
    """Resolve every expected output through the shared job constructor."""

    cfg = cfg or RebuildDemoResultsConfig()
    stage = stage.expanduser().resolve()
    source_hashes: dict[Path, str] = {}
    source_contracts: dict[tuple[Path, str], _SourceContract] = {}
    robot_contracts: dict[str, _RobotContract] = {}
    expected: list[ExpectedArtifact] = []
    seen_outputs: set[Path] = set()
    collision_paths: list[Path] = []
    planned_artifact_count = 0
    planned_sources: set[Path] = set()
    for plan in plans:
        if not plan.available or plan.data_dir is None:
            continue
        data_dir = plan.data_dir.resolve()
        sources = discover_motion_files(data_dir, plan.spec.data_format)
        if not sources:
            raise FileNotFoundError(f"No {plan.spec.data_format} motion sources found in {data_dir}")
        robot_config = RobotConfig(robot_type=plan.robot)
        motion_config = MotionDataConfig(
            data_format=plan.spec.data_format,
            robot_type=plan.robot,
        )
        robot_contract = robot_contracts.get(plan.robot)
        if robot_contract is None:
            robot_contract = _robot_contract_from_model(plan.robot)
            robot_contracts[plan.robot] = robot_contract
        for source_path in sources:
            task_name, data_path, sequence_key, task_dir = _source_task_identity(
                source_path,
                data_root=data_dir,
                task_type=plan.spec.task_type,
            )
            object_name = resolve_task_object_name(
                plan.spec.task_type,
                plan.spec.data_format,
                task_name,
                None,
            )
            task_config = TaskConfig(
                object_name=object_name,
                object_dir=task_dir,
            )
            base_config = RetargetingConfig(
                task_type=plan.spec.task_type,
                robot=plan.robot,
                data_format=plan.spec.data_format,
                task_name=task_name,
                data_path=data_path,
                save_dir=None,
                augmentation=False,
                robot_config=robot_config,
                motion_data_config=motion_config,
                task_config=task_config,
                retargeter=RetargeterConfig(
                    release_object_non_penetration_on_infeasible=(
                        _release_object_non_penetration_for_plan(
                            cfg,
                            plan,
                        )
                    ),
                ),
            )
            resolved_source = source_path.resolve()
            planned_sources.add(resolved_source)
            source_contract_key = (
                resolved_source,
                plan.spec.data_format,
            )
            source_contract = source_contracts.get(source_contract_key)
            if source_contract is None:
                source_contract = _source_contract_from_adapter(
                    resolved_source,
                    data_format=plan.spec.data_format,
                    data_path=data_path,
                    task_name=task_name,
                    human_height=motion_config.human_height,
                    human_joint_names=tuple(motion_config.resolved_demo_joints),
                    human_joint_parent_indices=(motion_config.resolved_joint_parent_indices),
                )
                source_contracts[source_contract_key] = source_contract
            source_sha256 = source_hashes.get(resolved_source)
            for variant in _planned_variants_for_plan(cfg, plan):
                planned_artifact_count += 1
                run_kind = "single" if variant.is_identity else "augmentation"
                job = build_retarget_job(
                    base_config,
                    variant=variant,
                    run_kind=run_kind,
                    results_root=stage,
                    dataset_partition=data_dir.name,
                    sequence_key=sequence_key,
                    source_path=resolved_source,
                    source_sha256=source_sha256,
                )
                source_sha256 = job.source_sha256
                source_hashes[resolved_source] = source_sha256
                output_path = job.output_path.resolve()
                if output_path in seen_outputs:
                    collision_paths.append(output_path)
                    continue
                seen_outputs.add(output_path)
                expected.append(
                    ExpectedArtifact(
                        output_path=output_path,
                        source_path=resolved_source,
                        source_sha256=job.source_sha256,
                        config_sha256=job.config_sha256,
                        robot=plan.robot,
                        task_type=plan.spec.task_type,
                        data_format=plan.spec.data_format,
                        variant=variant.name,
                        run_kind=run_kind,
                        dataset_id=plan.spec.dataset_id,
                        frame_count=source_contract.frame_count,
                        human_joint_names=source_contract.human_joint_names,
                        human_joint_parent_indices=(source_contract.human_joint_parent_indices),
                        human_orientation_joint_names=(source_contract.human_orientation_joint_names),
                        human_orientation_source=(source_contract.human_orientation_source),
                        human_orientation_sha256=(source_contract.human_orientation_sha256),
                        human_orientation_tensor_shape=(source_contract.human_orientation_tensor_shape),
                        human_orientation_tensor_bytes=(source_contract.human_orientation_tensor_bytes),
                        robot_link_names=robot_contract.link_names,
                        robot_link_parent_indices=(robot_contract.parent_indices),
                    )
                )
    if collision_paths:
        raise ArtifactPlanCollisionError(
            collision_paths,
            expected_artifact_count=planned_artifact_count,
            expected_source_count=len(planned_sources),
        )
    return tuple(sorted(expected, key=lambda item: str(item.output_path)))


def _scalar_text(payload: Mapping[str, Any], key: str) -> str:
    value = np.asarray(payload[key])
    if value.ndim != 0:
        raise ValueError(f"{key} must be scalar")
    item = value.item()
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def _text_tuple(payload: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = np.asarray(payload[key])
    if value.ndim != 1:
        raise ValueError(f"{key} must be one-dimensional")
    names: list[str] = []
    for item in value.tolist():
        if isinstance(item, bytes):
            names.append(item.decode("utf-8"))
        else:
            names.append(str(item))
    return tuple(names)


def _artifact_identity_from_path(
    artifact_path: Path,
    *,
    root: Path,
) -> tuple[str, str, str, str, str]:
    relative = artifact_path.relative_to(root)
    parts = relative.parts
    if len(parts) < 7 or parts[0] != "canonical":
        raise ValueError("expected canonical/<robot>/<task>/<format>/<partition>/<sequence>/<variant>.npz")
    return parts[1], parts[2], parts[3], parts[-2], artifact_path.stem


def _expected_variant_names(
    cfg: RebuildDemoResultsConfig,
    plan: BatchPlan,
) -> tuple[str, ...]:
    """Return exact expected variants under the configured rebuild mode."""

    return tuple(variant.name for variant in _planned_variants_for_plan(cfg, plan))


def _canonical_variant_names(task_type: str) -> tuple[str, ...]:
    """Return every schema-valid canonical variant for one task type."""

    return tuple(
        variant.name
        for variant in planned_variants(
            task_type,
            augmentation=True,
        )
    )


QUALITY_METRICS = (
    "foot_sticking_fallback",
    "foot_sticking_release",
    "object_non_penetration_release",
    "release_frame_union",
    "foot_sticking_full_sequence_retry",
)


@dataclass
class _QualityMetricAccumulator:
    """Numerators and denominators for one eligibility-aware quality metric."""

    eligible_artifact_count: int = 0
    eligible_frame_count: int = 0
    affected_artifact_count: int = 0
    affected_frame_count: int = 0

    def add(
        self,
        *,
        eligible: bool,
        frame_count: int,
        affected_frame_count: int,
    ) -> None:
        if not eligible:
            return
        self.eligible_artifact_count += 1
        self.eligible_frame_count += frame_count
        if affected_frame_count:
            self.affected_artifact_count += 1
            self.affected_frame_count += affected_frame_count

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "affected_artifact_numerator": self.affected_artifact_count,
            "eligible_artifact_denominator": self.eligible_artifact_count,
            "affected_artifact_fraction": (
                self.affected_artifact_count / self.eligible_artifact_count if self.eligible_artifact_count else None
            ),
            "affected_frame_numerator": self.affected_frame_count,
            "eligible_frame_denominator": self.eligible_frame_count,
            "affected_frame_fraction": (
                self.affected_frame_count / self.eligible_frame_count if self.eligible_frame_count else None
            ),
        }


def _quality_group_label(
    level: str,
    key: tuple[str, ...],
) -> dict[str, str]:
    if level == "robot_dataset_task":
        robot, dataset_id, task_type = key
        return {
            "robot": robot,
            "dataset_id": dataset_id,
            "task_type": task_type,
        }
    if level == "robot_dataset_task_variant":
        robot, dataset_id, task_type, variant = key
        return {
            "robot": robot,
            "dataset_id": dataset_id,
            "task_type": task_type,
            "variant": variant,
        }
    raise ValueError(f"Unknown quality group level: {level}")


def _scalar_bool(payload: Mapping[str, Any], key: str) -> bool:
    value = np.asarray(payload[key])
    if value.ndim != 0:
        raise ValueError(f"{key} must be scalar")
    return bool(value.item())


def _directory_inventory(
    root: Path,
) -> tuple[tuple[str, int, int, int, int], ...]:
    if root.is_symlink():
        raise ValueError(f"Result tree root must not be a symlink: {root}")
    entries: list[tuple[str, int, int, int, int]] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Result tree must not contain symlinks: {path}")
        if not path.is_file():
            continue
        stat = path.stat()
        entries.append(
            (
                path.relative_to(root).as_posix(),
                stat.st_size,
                stat.st_mtime_ns,
                stat.st_ctime_ns,
                stat.st_ino,
            )
        )
    return tuple(sorted(entries))


def _update_length_prefixed(
    digest: Any,
    value: bytes,
) -> None:
    digest.update(len(value).to_bytes(8, byteorder="big", signed=False))
    digest.update(value)


def _fingerprint_directory(root: Path) -> _DirectoryFingerprint:
    """Hash path, size, and bytes while rejecting a concurrently changing tree."""

    root = root.resolve(strict=True)
    before = _directory_inventory(root)
    digest = hashlib.sha256()
    digest.update(b"holosoma-demo-results-tree-v1\0")
    byte_count = 0
    for relative_path, size, _mtime_ns, _ctime_ns, _inode in before:
        path = root / relative_path
        _update_length_prefixed(
            digest,
            relative_path.encode("utf-8"),
        )
        digest.update(size.to_bytes(8, byteorder="big", signed=False))
        with path.open("rb") as file:
            while chunk := file.read(1024 * 1024):
                digest.update(chunk)
        byte_count += size
    after = _directory_inventory(root)
    if after != before:
        raise RuntimeError(f"Result tree changed while fingerprinting: {root}")
    return _DirectoryFingerprint(
        sha256=digest.hexdigest(),
        file_count=len(before),
        byte_count=byte_count,
    )


def _fingerprint_external_assets(
    assets: Mapping[Path, str],
) -> _DirectoryFingerprint:
    """Verify and bind each unique external object dependency closure."""

    digest = hashlib.sha256()
    digest.update(b"holosoma-demo-results-external-assets-v1\0")
    files: dict[str, tuple[int, str]] = {}
    for raw_path, expected_closure_sha256 in sorted(
        assets.items(),
        key=lambda item: str(item[0]),
    ):
        path = Path(raw_path)
        (
            manifest_json,
            observed_manifest_sha256,
            _closure_byte_count,
            _closure_file_count,
        ) = compute_object_asset_manifest(path)
        if observed_manifest_sha256 != expected_closure_sha256:
            raise ValueError(
                "External object asset closure digest changed: "
                f"{path}; expected {expected_closure_sha256}, observed "
                f"{observed_manifest_sha256}",
            )
        _update_length_prefixed(digest, str(path).encode("utf-8"))
        _update_length_prefixed(
            digest,
            expected_closure_sha256.encode("ascii"),
        )
        manifest = json.loads(manifest_json)
        for entry in [manifest["urdf"], *manifest["dependencies"]]:
            entry_path = str(entry["path"])
            identity = (int(entry["size"]), str(entry["sha256"]))
            previous = files.setdefault(entry_path, identity)
            if previous != identity:
                raise ValueError(
                    f"External object asset path has conflicting identities: {entry_path}",
                )
    return _DirectoryFingerprint(
        sha256=digest.hexdigest(),
        file_count=len(files),
        byte_count=sum(size for size, _sha256 in files.values()),
    )


def _external_assets_from_artifacts(
    artifacts: Iterable[Path],
) -> dict[Path, str]:
    """Return conflict-free URDF paths and dependency-closure digests."""

    assets: dict[Path, str] = {}
    for artifact_path in sorted(Path(path) for path in artifacts):
        with np.load(artifact_path, allow_pickle=False) as payload:
            validate_result_artifact(payload)
            object_urdf_path, object_asset_closure_sha256 = validate_result_external_assets(payload)
        if object_urdf_path is None:
            continue
        previous = assets.setdefault(
            object_urdf_path,
            object_asset_closure_sha256,
        )
        if previous != object_asset_closure_sha256:
            raise ValueError(
                "Artifacts assign conflicting dependency-closure identities "
                f"to external object URDF {object_urdf_path}",
            )
    return assets


def validate_rebuild_directory(
    root: Path,
    expected_artifacts: Iterable[ExpectedArtifact],
    *,
    omitted_artifact_paths: Iterable[Path] = (),
    max_fallback_frame_fraction_per_artifact: float | None = None,
    max_release_union_frame_fraction_per_artifact: float | None = None,
    max_full_sequence_retry_fraction_per_artifact: float | None = None,
    max_fallback_artifact_fraction_per_group: float | None = None,
    max_foot_release_artifact_fraction_per_group: float | None = None,
    max_object_release_artifact_fraction_per_group: float | None = None,
    max_release_union_artifact_fraction_per_group: float | None = None,
    max_full_sequence_retry_artifact_fraction_per_group: float | None = None,
) -> ValidationSummary:
    """Validate every artifact plus audited completeness against the plan.

    ``omitted_artifact_paths`` may only contain planned paths that are absent.
    Callers are responsible for deriving that set from trusted batch failure
    evidence. Existing artifacts are never exempted from schema, identity,
    residual, or quality validation.
    """

    root = root.expanduser().resolve()
    expected = {item.output_path.expanduser().resolve(): item for item in expected_artifacts}
    omitted_paths = {Path(path).expanduser().resolve() for path in omitted_artifact_paths}
    expected_sources = {item.source_path for item in expected.values()}
    artifacts = sorted(path.resolve() for path in root.rglob("*.npz")) if root.is_dir() else []
    issues = _IssueCollector()
    for omitted_path in sorted(omitted_paths.difference(expected)):
        issues.add(
            "unplanned_omission",
            omitted_path,
            "Omission evidence refers to an artifact outside the exact plan",
        )
    omitted_paths.intersection_update(expected)
    for omitted_path in sorted(omitted_paths):
        if _lexists(omitted_path):
            issues.add(
                "existing_artifact_omitted",
                omitted_path,
                "An existing artifact cannot be exempted by source failure evidence",
            )
    tree_fingerprint = _DirectoryFingerprint(
        sha256="",
        file_count=0,
        byte_count=0,
    )
    if not root.is_dir():
        issues.add("missing_root", root, "Staging result directory does not exist")
    elif not artifacts:
        issues.add("empty_root", root, "Staging result directory contains no NPZ artifacts")
    if root.is_dir():
        try:
            tree_fingerprint = _fingerprint_directory(root)
            expected_regular_files = {path for path in expected}
            observed_regular_files = {
                (root / relative_path).resolve()
                for (
                    relative_path,
                    _size,
                    _mtime_ns,
                    _ctime_ns,
                    _inode,
                ) in _directory_inventory(root)
            }
            for unexpected_file in sorted(observed_regular_files.difference(expected_regular_files)):
                if unexpected_file.suffix == ".npz":
                    continue
                issues.add(
                    "unexpected_file",
                    unexpected_file,
                    "Staging tree contains a non-artifact file",
                )
        except (OSError, RuntimeError, ValueError) as exc:
            issues.add(
                "unsafe_result_tree",
                root,
                f"{type(exc).__name__}: {exc}",
            )

    expected_paths = set(expected)
    observed_paths = set(artifacts)
    for missing_path in sorted(expected_paths.difference(observed_paths).difference(omitted_paths)):
        issues.add(
            "missing_artifact",
            missing_path,
            "Planned source variant is missing",
        )
    for unexpected_path in sorted(observed_paths.difference(expected_paths)):
        issues.add(
            "unexpected_artifact",
            unexpected_path,
            "Artifact is not part of the exact planned source/variant matrix",
        )

    orientation_by_format: dict[str, Counter[str]] = defaultdict(Counter)
    robot_link_counts: Counter[str] = Counter()
    mesh_presence = Counter()
    variants = Counter()
    constraint_relaxations: dict[str, Counter[str]] = {
        key: Counter()
        for key in (
            "foot_sticking_fallback_frames",
            "foot_sticking_release_frames",
            "object_non_penetration_release_frames",
            "foot_sticking_full_sequence_retry_frame",
        )
    }
    valid_artifacts = 0
    group_variants: dict[tuple[str, str, str, str, str], set[str]] = defaultdict(set)
    expected_group_variants: dict[
        tuple[str, str, str, str, str],
        set[str],
    ] = defaultdict(set)
    for expected_path, expected_item in expected.items():
        if expected_path in omitted_paths:
            continue
        expected_group_key = (
            expected_item.robot,
            expected_item.task_type,
            expected_item.data_format,
            expected_path.parent.parent.relative_to(root).as_posix(),
            expected_item.source_path.as_posix(),
        )
        expected_group_variants[expected_group_key].add(
            expected_item.variant,
        )
    quality_groups: dict[
        str,
        dict[
            tuple[str, ...],
            dict[str, _QualityMetricAccumulator],
        ],
    ] = {
        "robot_dataset_task": {},
        "robot_dataset_task_variant": {},
    }
    per_artifact_quality_checked = Counter()
    per_artifact_quality_exceeded = Counter()
    external_assets: dict[Path, str] = {}

    for artifact_path in artifacts:
        try:
            with np.load(artifact_path, allow_pickle=False) as payload:
                validate_result_artifact(payload)
                object_urdf_path, object_asset_closure_sha256 = validate_result_external_assets(payload)
                if object_urdf_path is not None:
                    previous_asset_closure_sha256 = external_assets.setdefault(
                        object_urdf_path,
                        object_asset_closure_sha256,
                    )
                    if previous_asset_closure_sha256 != object_asset_closure_sha256:
                        raise ValueError(
                            "Artifacts assign conflicting dependency-closure "
                            "identities to external object URDF "
                            f"{object_urdf_path}",
                        )
                robot = _scalar_text(payload, "robot_type")
                task_type = _scalar_text(payload, "task_type")
                data_format = _scalar_text(payload, "source_data_format")
                variant = _scalar_text(payload, "variant")
                run_kind = _scalar_text(payload, "run_kind")
                source_path = Path(_scalar_text(payload, "source_path")).resolve()
                path_robot, path_task, path_format, _sequence_leaf, path_variant = _artifact_identity_from_path(
                    artifact_path, root=root
                )
                if (robot, task_type, data_format, variant) != (
                    path_robot,
                    path_task,
                    path_format,
                    path_variant,
                ):
                    raise ValueError("path robot/task/format/variant does not match artifact fields")
                allowed_variants = _canonical_variant_names(task_type)
                if variant not in allowed_variants:
                    raise ValueError(f"variant {variant!r} is not canonical for task {task_type!r}")
                expected_run_kind = "single" if variant == "identity" else "augmentation"
                if run_kind != expected_run_kind:
                    raise ValueError(f"run_kind {run_kind!r} does not match variant {variant!r}")

                for hash_key in ("source_sha256", "config_sha256"):
                    hash_value = _scalar_text(payload, hash_key)
                    if HASH_PATTERN.fullmatch(hash_value) is None:
                        raise ValueError(f"{hash_key} must be a lowercase SHA-256 digest")
                config_json = _scalar_text(payload, "config_json")
                config_digest = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
                if config_digest != _scalar_text(payload, "config_sha256"):
                    raise ValueError("config_sha256 does not match the saved config_json")

                orientation_present = HUMAN_ORIENTATION_KEYS.issubset(payload.files)
                frame_count = int(np.asarray(payload["human_joints"]).shape[0])
                expected_item = expected.get(artifact_path)
                if expected_item is not None:
                    exact_values = {
                        "source_path": (source_path, expected_item.source_path),
                        "source_sha256": (
                            _scalar_text(payload, "source_sha256"),
                            expected_item.source_sha256,
                        ),
                        "config_sha256": (
                            _scalar_text(payload, "config_sha256"),
                            expected_item.config_sha256,
                        ),
                        "robot_type": (robot, expected_item.robot),
                        "task_type": (task_type, expected_item.task_type),
                        "source_data_format": (
                            data_format,
                            expected_item.data_format,
                        ),
                        "variant": (variant, expected_item.variant),
                        "run_kind": (run_kind, expected_item.run_kind),
                    }
                    mismatches = [key for key, (observed, planned) in exact_values.items() if observed != planned]
                    if mismatches:
                        raise ValueError(f"artifact does not match the planned job fields: {', '.join(mismatches)}")

                    if frame_count != expected_item.frame_count:
                        raise ValueError(
                            "human_joints frame count does not match the "
                            f"source adapter: {frame_count} != "
                            f"{expected_item.frame_count}"
                        )
                    human_joint_names = _text_tuple(
                        payload,
                        "human_joint_names",
                    )
                    if human_joint_names != expected_item.human_joint_names:
                        raise ValueError("human_joint_names do not match the complete canonical adapter joint order")
                    human_joint_parent_indices = tuple(
                        int(index) for index in np.asarray(payload["human_joint_parent_indices"]).tolist()
                    )
                    if human_joint_parent_indices != expected_item.human_joint_parent_indices:
                        raise ValueError("human_joint_parent_indices do not match the exact canonical adapter topology")

                    expected_orientation_names = expected_item.human_orientation_joint_names
                    observed_orientation_source = _scalar_text(
                        payload,
                        "orientation_source",
                    )
                    if expected_orientation_names is None:
                        if any(
                            value is not None
                            for value in (
                                expected_item.human_orientation_sha256,
                                expected_item.human_orientation_tensor_shape,
                                expected_item.human_orientation_tensor_bytes,
                            )
                        ):
                            raise ValueError(
                                "planned absent-orientation contract contains an unexpected tensor identity"
                            )
                        if set(payload.files).intersection(HUMAN_ORIENTATION_KEYS):
                            raise ValueError(
                                "source adapter has no direct orientations, "
                                "but artifact contains human orientation fields"
                            )
                    else:
                        if (
                            expected_item.human_orientation_sha256 is None
                            or expected_item.human_orientation_tensor_shape is None
                            or expected_item.human_orientation_tensor_bytes is None
                        ):
                            raise ValueError(
                                "planned direct-orientation contract is missing its frozen tensor identity"
                            )
                        if not orientation_present:
                            raise ValueError(
                                "source adapter has direct orientations, but artifact omits human orientation fields"
                            )
                        observed_orientation_names = _text_tuple(
                            payload,
                            "human_orientation_joint_names",
                        )
                        if observed_orientation_names != expected_orientation_names:
                            raise ValueError(
                                "human_orientation_joint_names do not match the exact direct adapter subset"
                            )
                        observed_orientation_tensor = np.asarray(payload["human_orientation_quaternions_wxyz"])
                        if observed_orientation_tensor.dtype != np.dtype(np.float32):
                            raise ValueError(
                                "human_orientation_quaternions_wxyz is not the adapter's persisted float32 dtype"
                            )
                        if tuple(observed_orientation_tensor.shape) != expected_item.human_orientation_tensor_shape:
                            raise ValueError(
                                "human_orientation_quaternions_wxyz shape does not match the direct adapter tensor"
                            )
                        expected_orientation_tensor = np.frombuffer(
                            expected_item.human_orientation_tensor_bytes,
                            dtype=np.float32,
                        ).reshape(expected_item.human_orientation_tensor_shape)
                        observed_saved_orientation_sha256 = _scalar_text(
                            payload,
                            "human_orientation_sha256",
                        )
                        if observed_saved_orientation_sha256 != expected_item.human_orientation_sha256:
                            raise ValueError(
                                "human_orientation_sha256 does not match the "
                                "production adapter's frozen named tensor digest"
                            )
                        observed_orientation_sha256 = _direct_orientation_tensor_sha256(
                            observed_orientation_names,
                            observed_orientation_tensor,
                        )
                        if observed_orientation_sha256 != expected_item.human_orientation_sha256:
                            raise ValueError(
                                "human_orientation_quaternions_wxyz SHA-256 "
                                "does not match the named direct adapter tensor"
                            )
                        if not np.array_equal(
                            observed_orientation_tensor,
                            expected_orientation_tensor,
                        ):
                            raise ValueError(
                                "human_orientation_quaternions_wxyz values do "
                                "not exactly match the direct adapter tensor"
                            )
                        if (
                            observed_orientation_tensor.tobytes(order="C")
                            != expected_item.human_orientation_tensor_bytes
                        ):
                            raise ValueError(
                                "human_orientation_quaternions_wxyz bytes do "
                                "not exactly match the float32 direct adapter tensor"
                            )
                    if observed_orientation_source != expected_item.human_orientation_source:
                        raise ValueError(
                            "orientation_source does not match adapter "
                            f"provenance: {observed_orientation_source!r} != "
                            f"{expected_item.human_orientation_source!r}"
                        )

                    robot_link_names = _text_tuple(
                        payload,
                        "robot_link_names",
                    )
                    if robot_link_names != expected_item.robot_link_names:
                        raise ValueError(
                            f"robot_link_names do not match the complete {expected_item.robot.upper()} MuJoCo subtree"
                        )
                    robot_link_parent_indices = tuple(
                        int(index) for index in np.asarray(payload["robot_link_parent_indices"]).tolist()
                    )
                    if robot_link_parent_indices != expected_item.robot_link_parent_indices:
                        raise ValueError(
                            "robot_link_parent_indices do not match the exact "
                            f"{expected_item.robot.upper()} MuJoCo parent tree"
                        )

                orientation_by_format[data_format]["present" if orientation_present else "absent"] += 1
                mesh_present = INTERACTION_MESH_KEYS.issubset(payload.files)
                mesh_presence["present" if mesh_present else "absent"] += 1
                if not mesh_present:
                    raise ValueError("Interaction Mesh fields are required for rebuilt demo results")
                robot_link_count = int(np.asarray(payload["robot_link_positions"]).shape[1])
                robot_link_counts[str(robot_link_count)] += 1
                variants[variant] += 1
                group_key = (
                    robot,
                    task_type,
                    data_format,
                    artifact_path.parent.parent.relative_to(root).as_posix(),
                    source_path.as_posix(),
                )
                group_variants[group_key].add(variant)
                affected_frames: dict[str, set[int]] = {}
                for field in (
                    "foot_sticking_fallback_frames",
                    "foot_sticking_release_frames",
                    "object_non_penetration_release_frames",
                ):
                    frame_set = {int(frame) for frame in np.asarray(payload[field]).tolist()}
                    affected_frames[field] = frame_set
                    affected_frame_count = len(frame_set)
                    if affected_frame_count:
                        constraint_relaxations[field]["artifact_count"] += 1
                        constraint_relaxations[field]["frame_count"] += affected_frame_count
                retry_frame = int(np.asarray(payload["foot_sticking_full_sequence_retry_frame"]).item())
                if retry_frame >= 0:
                    retry_statistics = constraint_relaxations["foot_sticking_full_sequence_retry_frame"]
                    retry_statistics["artifact_count"] += 1
                    retry_statistics["frame_count"] += frame_count

                if expected_item is not None:
                    foot_saved_trajectory_eligible = _scalar_bool(
                        payload,
                        "foot_sticking_enabled_for_saved_trajectory",
                    )
                    foot_retry_eligible = foot_saved_trajectory_eligible or retry_frame >= 0
                    object_eligible = _scalar_bool(
                        payload,
                        "object_non_penetration_eligible_for_saved_trajectory",
                    ) and _scalar_bool(
                        payload,
                        "release_object_non_penetration_on_infeasible",
                    )
                    fallback_frames = affected_frames["foot_sticking_fallback_frames"]
                    foot_release_frames = affected_frames["foot_sticking_release_frames"]
                    object_release_frames = affected_frames["object_non_penetration_release_frames"]
                    release_union_frames = foot_release_frames | object_release_frames
                    if not foot_saved_trajectory_eligible and (fallback_frames or foot_release_frames):
                        issues.add(
                            "quality_ineligible_event",
                            artifact_path,
                            "foot fallback/release frames are present although "
                            "foot sticking was not enabled for this trajectory",
                        )
                    if not object_eligible and object_release_frames:
                        issues.add(
                            "quality_ineligible_event",
                            artifact_path,
                            "object release frames are present although this "
                            "artifact is not object non-penetration release eligible",
                        )

                    quality_observations = {
                        "foot_sticking_fallback": (
                            foot_saved_trajectory_eligible,
                            len(fallback_frames),
                        ),
                        "foot_sticking_release": (
                            foot_saved_trajectory_eligible,
                            len(foot_release_frames),
                        ),
                        "object_non_penetration_release": (
                            object_eligible,
                            len(object_release_frames),
                        ),
                        "release_frame_union": (
                            foot_saved_trajectory_eligible or object_eligible,
                            len(release_union_frames),
                        ),
                        "foot_sticking_full_sequence_retry": (
                            foot_retry_eligible,
                            frame_count if retry_frame >= 0 else 0,
                        ),
                    }
                    per_artifact_limits = {
                        "foot_sticking_fallback": (max_fallback_frame_fraction_per_artifact),
                        "release_frame_union": (max_release_union_frame_fraction_per_artifact),
                        "foot_sticking_full_sequence_retry": (max_full_sequence_retry_fraction_per_artifact),
                    }
                    for metric, limit in per_artifact_limits.items():
                        eligible, affected_frame_count = quality_observations[metric]
                        if not eligible:
                            continue
                        per_artifact_quality_checked[metric] += 1
                        observed_fraction = affected_frame_count / frame_count
                        if limit is not None and observed_fraction > limit:
                            per_artifact_quality_exceeded[metric] += 1
                            issues.add(
                                "quality_per_artifact_fraction",
                                artifact_path,
                                f"{metric}={affected_frame_count}/{frame_count}"
                                f"={observed_fraction:.6f} exceeds "
                                f"{limit:.6f}",
                            )

                    group_keys = {
                        "robot_dataset_task": (
                            robot,
                            expected_item.dataset_id,
                            task_type,
                        ),
                        "robot_dataset_task_variant": (
                            robot,
                            expected_item.dataset_id,
                            task_type,
                            variant,
                        ),
                    }
                    for level, quality_group_key in group_keys.items():
                        metrics = quality_groups[level].setdefault(
                            quality_group_key,
                            {metric: _QualityMetricAccumulator() for metric in QUALITY_METRICS},
                        )
                        for metric, (
                            eligible,
                            affected_frame_count,
                        ) in quality_observations.items():
                            metrics[metric].add(
                                eligible=eligible,
                                frame_count=frame_count,
                                affected_frame_count=affected_frame_count,
                            )
                valid_artifacts += 1
        except (
            KeyError,
            OSError,
            UnicodeDecodeError,
            ResultArtifactValidationError,
            ValueError,
        ) as exc:
            issues.add(
                "invalid_artifact",
                artifact_path,
                f"{type(exc).__name__}: {exc}",
            )

    for group_key in sorted(set(expected_group_variants).union(group_variants)):
        observed_variants = group_variants.get(group_key, set())
        expected_variants = expected_group_variants.get(group_key, set())
        if observed_variants != expected_variants:
            missing = sorted(expected_variants.difference(observed_variants))
            extra = sorted(observed_variants.difference(expected_variants))
            issues.add(
                "incomplete_variant_family",
                group_key[3],
                f"missing={missing}, extra={extra}",
            )

    per_artifact_quality_limits = {
        "foot_sticking_fallback_frame_fraction": (max_fallback_frame_fraction_per_artifact),
        "release_frame_union_fraction": (max_release_union_frame_fraction_per_artifact),
        "foot_sticking_full_sequence_retry_fraction": (max_full_sequence_retry_fraction_per_artifact),
    }
    grouped_quality_limits = {
        "foot_sticking_fallback": (max_fallback_artifact_fraction_per_group),
        "foot_sticking_release": (max_foot_release_artifact_fraction_per_group),
        "object_non_penetration_release": (max_object_release_artifact_fraction_per_group),
        "release_frame_union": (max_release_union_artifact_fraction_per_group),
        "foot_sticking_full_sequence_retry": (max_full_sequence_retry_artifact_fraction_per_group),
    }
    quality_group_reports: dict[str, list[dict[str, Any]]] = {}
    exceeded_groups: list[dict[str, Any]] = []
    for level, groups in quality_groups.items():
        reports: list[dict[str, Any]] = []
        for quality_group_key, metrics in sorted(groups.items()):
            label = _quality_group_label(level, quality_group_key)
            metric_payload = {metric: accumulator.to_dict() for metric, accumulator in metrics.items()}
            group_exceeded: list[dict[str, Any]] = []
            for metric, limit in grouped_quality_limits.items():
                accumulator = metrics[metric]
                if limit is None or not accumulator.eligible_artifact_count:
                    continue
                observed_fraction = accumulator.affected_artifact_count / accumulator.eligible_artifact_count
                if observed_fraction <= limit:
                    continue
                exceeded = {
                    "level": level,
                    "group": label,
                    "metric": metric,
                    "affected_artifact_numerator": (accumulator.affected_artifact_count),
                    "eligible_artifact_denominator": (accumulator.eligible_artifact_count),
                    "observed_fraction": observed_fraction,
                    "limit": limit,
                }
                group_exceeded.append(exceeded)
                exceeded_groups.append(exceeded)
                issues.add(
                    "quality_group_fraction",
                    root,
                    f"{level} {label}: {metric}="
                    f"{accumulator.affected_artifact_count}/"
                    f"{accumulator.eligible_artifact_count}="
                    f"{observed_fraction:.6f} exceeds {limit:.6f}",
                )
            reports.append(
                {
                    "group": label,
                    "metrics": metric_payload,
                    "exceeded_gates": group_exceeded,
                }
            )
        quality_group_reports[level] = reports

    statistics = {
        "valid_artifacts": valid_artifacts,
        "audited_omissions": {
            "artifact_count": len(omitted_paths),
            "artifact_paths": [str(path) for path in sorted(omitted_paths)],
        },
        "human_orientation": {
            data_format: dict(sorted(counts.items())) for data_format, counts in sorted(orientation_by_format.items())
        },
        "interaction_mesh": dict(sorted(mesh_presence.items())),
        "robot_link_counts": dict(sorted(robot_link_counts.items(), key=lambda item: int(item[0]))),
        "variants": dict(sorted(variants.items())),
        "constraint_relaxations": {
            field: {
                "artifact_count": counts["artifact_count"],
                "frame_count": counts["frame_count"],
            }
            for field, counts in constraint_relaxations.items()
        },
        "quality_gates": {
            "limits": {
                "per_artifact": per_artifact_quality_limits,
                "grouped_affected_artifact_fraction": (grouped_quality_limits),
            },
            "per_artifact": {
                "checked_artifact_count": {
                    metric: per_artifact_quality_checked[metric]
                    for metric in (
                        "foot_sticking_fallback",
                        "release_frame_union",
                        "foot_sticking_full_sequence_retry",
                    )
                },
                "exceeded_artifact_count": {
                    metric: per_artifact_quality_exceeded[metric]
                    for metric in (
                        "foot_sticking_fallback",
                        "release_frame_union",
                        "foot_sticking_full_sequence_retry",
                    )
                },
            },
            "groups": quality_group_reports,
            "exceeded_groups": exceeded_groups,
        },
    }
    external_asset_fingerprint = _DirectoryFingerprint(
        sha256="",
        file_count=0,
        byte_count=0,
    )
    try:
        external_asset_fingerprint = _fingerprint_external_assets(
            external_assets,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        issues.add(
            "invalid_external_asset",
            root,
            f"{type(exc).__name__}: {exc}",
        )
    if root.is_dir() and tree_fingerprint.sha256:
        try:
            final_tree_fingerprint = _fingerprint_directory(root)
            if final_tree_fingerprint != tree_fingerprint:
                issues.add(
                    "result_tree_changed",
                    root,
                    "Staging tree changed while artifacts were being validated",
                )
        except (OSError, RuntimeError, ValueError) as exc:
            issues.add(
                "unsafe_result_tree",
                root,
                f"{type(exc).__name__}: {exc}",
            )
    return ValidationSummary(
        ok=issues.count == 0,
        root=str(root),
        artifact_count=len(artifacts),
        expected_artifact_count=len(expected),
        expected_source_count=len(expected_sources),
        unexpected_artifact_count=len(observed_paths.difference(expected_paths)),
        tree_sha256=tree_fingerprint.sha256,
        tree_file_count=tree_fingerprint.file_count,
        tree_byte_count=tree_fingerprint.byte_count,
        external_asset_sha256=external_asset_fingerprint.sha256,
        external_asset_file_count=external_asset_fingerprint.file_count,
        external_asset_byte_count=external_asset_fingerprint.byte_count,
        issue_count=issues.count,
        issues=tuple(issues.samples),
        statistics=statistics,
    )


def _archive_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def _unique_archive_root(results_root: Path) -> Path:
    archive_parent = results_root / "archive" / _archive_timestamp()
    candidate = archive_parent / "v1"
    suffix = 1
    while candidate.parent.exists() or candidate.exists():
        candidate = results_root / "archive" / f"{archive_parent.name}-{suffix}" / "v1"
        suffix += 1
    return candidate


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _assert_real_directory(path: Path, *, label: str) -> None:
    if not _lexists(path):
        raise DemoRebuildError(f"{label} does not exist: {path}")
    path_stat = path.lstat()
    if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISDIR(path_stat.st_mode):
        raise DemoRebuildError(f"{label} must be a real non-symlink directory: {path}")


def _ensure_real_directory(path: Path, *, parent: Path, label: str) -> None:
    """Create one child directory without following a substituted entry."""

    if path.parent != parent:
        raise DemoRebuildError(f"{label} escaped its fixed parent: {path}")
    if _lexists(path):
        _assert_real_directory(path, label=label)
        return
    path.mkdir()
    _assert_real_directory(path, label=label)
    _fsync_directory(parent)


def _fingerprint_to_dict(
    fingerprint: _DirectoryFingerprint,
) -> dict[str, int | str]:
    return {
        "sha256": fingerprint.sha256,
        "file_count": fingerprint.file_count,
        "byte_count": fingerprint.byte_count,
    }


def _fingerprint_from_mapping(
    payload: Mapping[str, Any],
    *,
    label: str,
) -> _DirectoryFingerprint:
    sha256 = payload.get("sha256")
    file_count = payload.get("file_count")
    byte_count = payload.get("byte_count")
    if not isinstance(sha256, str) or HASH_PATTERN.fullmatch(sha256) is None:
        raise DemoRebuildError(f"{label}.sha256 is invalid")
    if (
        isinstance(file_count, bool)
        or not isinstance(file_count, int)
        or file_count < 0
        or isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count < 0
    ):
        raise DemoRebuildError(f"{label} file_count/byte_count must be non-negative integers")
    return _DirectoryFingerprint(
        sha256=sha256,
        file_count=file_count,
        byte_count=byte_count,
    )


def _optional_directory_fingerprint(
    path: Path,
    *,
    label: str,
) -> _DirectoryFingerprint | None:
    if not _lexists(path):
        return None
    _assert_real_directory(path, label=label)
    return _fingerprint_directory(path)


def _promotion_journal_payload(
    *,
    results_root: Path,
    run_id: str,
    stage: Path,
    formal: Path,
    archived: Path | None,
    new_tree: _DirectoryFingerprint,
    old_tree: _DirectoryFingerprint | None,
    external_assets: _DirectoryFingerprint,
    final_report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "schema_version": PROMOTION_JOURNAL_SCHEMA_VERSION,
        "status": "prepared",
        "run_id": run_id,
        "results_root": str(results_root),
        "staging_root": str(stage),
        "formal_root": str(formal),
        "archived_root": str(archived) if archived is not None else None,
        "had_formal": old_tree is not None,
        "new_tree": _fingerprint_to_dict(new_tree),
        "old_tree": (_fingerprint_to_dict(old_tree) if old_tree is not None else None),
        "external_assets": _fingerprint_to_dict(external_assets),
        "final_report": dict(final_report) if final_report is not None else None,
        "created_at_utc": now,
        "updated_at_utc": now,
    }


def _load_promotion_journal(path: Path) -> dict[str, Any]:
    if not _lexists(path):
        raise FileNotFoundError(f"Promotion journal not found: {path}")
    path_stat = path.lstat()
    if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISREG(path_stat.st_mode):
        raise DemoRebuildError(f"Promotion journal must be a real regular file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DemoRebuildError(f"Promotion journal is not readable valid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DemoRebuildError(f"Promotion journal must contain a JSON object: {path}")
    return payload


def _validate_promotion_journal(
    payload: Mapping[str, Any],
    *,
    results_root: Path,
    run_id: str,
) -> tuple[
    Path,
    Path,
    Path | None,
    _DirectoryFingerprint,
    _DirectoryFingerprint | None,
    _DirectoryFingerprint,
]:
    if payload.get("schema_version") != PROMOTION_JOURNAL_SCHEMA_VERSION:
        raise DemoRebuildError("Unsupported promotion journal schema")
    if payload.get("run_id") != run_id:
        raise DemoRebuildError("Promotion journal run_id does not match")
    if payload.get("status") not in {
        "prepared",
        "exchanged",
        "filesystem_promoted",
        "completed",
    }:
        raise DemoRebuildError("Promotion journal has an invalid status")
    if payload.get("results_root") != str(results_root):
        raise DemoRebuildError("Promotion journal results_root does not match")
    stage = staging_root(results_root, run_id)
    formal = results_root / "v1"
    if payload.get("staging_root") != str(stage):
        raise DemoRebuildError("Promotion journal staging_root does not match")
    if payload.get("formal_root") != str(formal):
        raise DemoRebuildError("Promotion journal formal_root does not match")
    archived_value = payload.get("archived_root")
    if archived_value is not None and not isinstance(archived_value, str):
        raise DemoRebuildError("Promotion journal archived_root is invalid")
    archived = Path(archived_value) if isinstance(archived_value, str) else None
    if archived is not None:
        archive_root = results_root / "archive"
        if not archived.is_absolute() or archived.parent.parent != archive_root or archived.name != "v1":
            raise DemoRebuildError(
                "Promotion journal archived_root is not a fixed results_root/archive/<transaction>/v1 path"
            )
    new_tree_payload = payload.get("new_tree")
    old_tree_payload = payload.get("old_tree")
    external_payload = payload.get("external_assets")
    if not isinstance(new_tree_payload, Mapping) or not isinstance(
        external_payload,
        Mapping,
    ):
        raise DemoRebuildError("Promotion journal fingerprints are incomplete")
    new_tree = _fingerprint_from_mapping(new_tree_payload, label="new_tree")
    external_assets = _fingerprint_from_mapping(
        external_payload,
        label="external_assets",
    )
    if old_tree_payload is None:
        old_tree = None
    elif isinstance(old_tree_payload, Mapping):
        old_tree = _fingerprint_from_mapping(old_tree_payload, label="old_tree")
    else:
        raise DemoRebuildError("Promotion journal old_tree is invalid")
    if bool(payload.get("had_formal")) != (old_tree is not None):
        raise DemoRebuildError("Promotion journal had_formal is inconsistent")
    if (archived is not None) != (old_tree is not None):
        raise DemoRebuildError("Promotion journal archive/old-tree identity is inconsistent")
    return (
        stage,
        formal,
        archived,
        new_tree,
        old_tree,
        external_assets,
    )


def _write_promotion_journal(
    path: Path,
    payload: Mapping[str, Any],
    *,
    status: str,
) -> dict[str, Any]:
    updated = dict(payload)
    updated["status"] = status
    updated["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_write_json(path, updated)
    return updated


@contextmanager
def _exclusive_results_state_lock(results_root: Path) -> Iterator[None]:
    lock_path = results_state_lock_path(results_root)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        lock_path,
        os.O_CREAT | os.O_RDWR,
        0o600,
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


@contextmanager
def _exclusive_rebuild_run_lock(
    results_root: Path,
    run_id: str,
) -> Iterator[None]:
    """Reject concurrent orchestrators sharing one staging/report identity."""

    lock_path = Path(results_root).expanduser().resolve() / ".locks" / f"rebuild-{run_id}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        lock_path,
        os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        try:
            fcntl.flock(
                descriptor,
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as exc:
            raise DemoRebuildError(
                "Another rebuild process already owns run_id "
                f"{run_id!r}; choose a different run_id or wait for it to finish",
            ) from exc
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _exchange_directories(left: Path, right: Path) -> bool:
    """Atomically exchange two paths on Linux when renameat2 is available."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        return False
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(left),
        -100,
        os.fsencode(right),
        2,
    )
    if result == 0:
        return True
    error_number = ctypes.get_errno()
    if error_number in {
        errno.EINVAL,
        errno.ENOSYS,
        errno.EOPNOTSUPP,
        errno.EXDEV,
    }:
        return False
    raise OSError(
        error_number,
        os.strerror(error_number),
        f"{left} <-> {right}",
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _final_promotion_report(
    report_payload: Mapping[str, Any],
    *,
    formal: Path,
    archived: Path | None,
    journal_path: Path,
) -> dict[str, Any]:
    final_report = dict(report_payload)
    final_report["status"] = "promoted"
    final_report["batch_failures"] = []
    final_report["promotion"] = {
        "promoted_root": str(formal),
        "archived_root": str(archived) if archived is not None else None,
        "journal_path": str(journal_path),
    }
    return final_report


def _prepare_promotion_journal(
    *,
    results_root: Path,
    run_id: str,
    validation: ValidationSummary,
    report_payload: Mapping[str, Any] | None,
) -> tuple[Path, dict[str, Any]]:
    journal_path = promotion_journal_path(results_root, run_id)
    if _lexists(journal_path):
        journal = _load_promotion_journal(journal_path)
        (
            _stage,
            formal,
            archived,
            new_tree,
            _old_tree,
            external_assets,
        ) = _validate_promotion_journal(
            journal,
            results_root=results_root,
            run_id=run_id,
        )
        expected_tree = _DirectoryFingerprint(
            sha256=validation.tree_sha256,
            file_count=validation.tree_file_count,
            byte_count=validation.tree_byte_count,
        )
        expected_external_assets = _DirectoryFingerprint(
            sha256=validation.external_asset_sha256,
            file_count=validation.external_asset_file_count,
            byte_count=validation.external_asset_byte_count,
        )
        if new_tree != expected_tree or external_assets != expected_external_assets:
            raise DemoRebuildError("Promotion journal does not match the supplied validation")
        if report_payload is not None:
            expected_final_report = _final_promotion_report(
                report_payload,
                formal=formal,
                archived=archived,
                journal_path=journal_path,
            )
            stored_final_report = journal.get("final_report")
            if stored_final_report is None:
                journal = dict(journal)
                journal["final_report"] = expected_final_report
                journal = _write_promotion_journal(
                    journal_path,
                    journal,
                    status=str(journal["status"]),
                )
            elif stored_final_report != expected_final_report:
                raise DemoRebuildError("Promotion journal final report identity changed")
        return journal_path, journal

    stage = staging_root(results_root, run_id)
    formal = results_root / "v1"
    if Path(validation.root).resolve() != stage:
        raise ValueError("Validation root does not match the selected staging tree")
    _assert_real_directory(stage, label="staging result directory")
    current_fingerprint = _fingerprint_directory(stage)
    expected_fingerprint = _DirectoryFingerprint(
        sha256=validation.tree_sha256,
        file_count=validation.tree_file_count,
        byte_count=validation.tree_byte_count,
    )
    if current_fingerprint != expected_fingerprint:
        raise DemoRebuildError("Staging result tree changed after validation; refusing promotion")
    try:
        current_external_asset_fingerprint = _fingerprint_external_assets(
            _external_assets_from_artifacts(stage.rglob("*.npz")),
        )
    except (
        FileNotFoundError,
        OSError,
        ResultArtifactValidationError,
        RuntimeError,
        ValueError,
    ) as exc:
        raise DemoRebuildError(
            "External object assets failed integrity validation; refusing promotion",
        ) from exc
    expected_external_asset_fingerprint = _DirectoryFingerprint(
        sha256=validation.external_asset_sha256,
        file_count=validation.external_asset_file_count,
        byte_count=validation.external_asset_byte_count,
    )
    if current_external_asset_fingerprint != expected_external_asset_fingerprint:
        raise DemoRebuildError(
            "External object assets changed after validation; refusing promotion",
        )

    if _lexists(formal):
        _assert_real_directory(formal, label="formal result directory")
        old_tree = _fingerprint_directory(formal)
        archive_root = results_root / "archive"
        _ensure_real_directory(
            archive_root,
            parent=results_root,
            label="result archive directory",
        )
        archived = _unique_archive_root(results_root)
    else:
        old_tree = None
        archived = None
    report_root = run_report_root(results_root, run_id)
    runs_root = results_root / "runs"
    _ensure_real_directory(
        runs_root,
        parent=results_root,
        label="rebuild report directory",
    )
    _ensure_real_directory(
        report_root,
        parent=runs_root,
        label="rebuild run report directory",
    )
    final_report = (
        _final_promotion_report(
            report_payload,
            formal=formal,
            archived=archived,
            journal_path=journal_path,
        )
        if report_payload is not None
        else None
    )
    journal = _promotion_journal_payload(
        results_root=results_root,
        run_id=run_id,
        stage=stage,
        formal=formal,
        archived=archived,
        new_tree=current_fingerprint,
        old_tree=old_tree,
        external_assets=current_external_asset_fingerprint,
        final_report=final_report,
    )
    _atomic_write_json(journal_path, journal)
    return journal_path, journal


def _ensure_promotion_archive_parent(
    results_root: Path,
    archived: Path,
) -> None:
    archive_root = results_root / "archive"
    _assert_real_directory(archive_root, label="result archive directory")
    _ensure_real_directory(
        archived.parent,
        parent=archive_root,
        label="promotion archive transaction directory",
    )
    unexpected_entries = {path.name for path in archived.parent.iterdir() if path != archived}
    if unexpected_entries:
        raise DemoRebuildError(
            f"Promotion archive transaction contains unexpected entries: {sorted(unexpected_entries)}"
        )


def _commit_promotion_journal(
    *,
    results_root: Path,
    run_id: str,
    journal_path: Path,
    journal: Mapping[str, Any],
    report_path: Path | None,
) -> tuple[PromotionResult, dict[str, Any]]:
    (
        stage,
        formal,
        archived,
        new_tree,
        old_tree,
        expected_external_assets,
    ) = _validate_promotion_journal(
        journal,
        results_root=results_root,
        run_id=run_id,
    )
    stage_fingerprint = _optional_directory_fingerprint(
        stage,
        label="staging result directory",
    )
    formal_fingerprint = _optional_directory_fingerprint(
        formal,
        label="formal result directory",
    )
    archived_fingerprint = (
        _optional_directory_fingerprint(
            archived,
            label="archived formal result directory",
        )
        if archived is not None
        else None
    )
    journal_status = str(journal["status"])
    prepared_state = stage_fingerprint == new_tree and formal_fingerprint == old_tree and archived_fingerprint is None
    exchanged_state = (
        old_tree is not None
        and new_tree != old_tree
        and stage_fingerprint == old_tree
        and formal_fingerprint == new_tree
        and archived_fingerprint is None
    )
    final_state = stage_fingerprint is None and formal_fingerprint == new_tree and archived_fingerprint == old_tree
    if not (prepared_state or exchanged_state or final_state):
        raise DemoRebuildError(
            "Promotion transaction is ambiguous: staging/formal/archive fingerprints do not match any journaled state"
        )
    if journal_status == "exchanged" and not (exchanged_state or final_state):
        raise DemoRebuildError("Promotion journal status is ahead of its filesystem state")
    if journal_status in {"filesystem_promoted", "completed"} and not final_state:
        raise DemoRebuildError(
            "Promotion journal reports a completed filesystem transition but the journaled tree layout regressed"
        )

    updated_journal = dict(journal)
    if old_tree is None:
        if stage_fingerprint == new_tree and formal_fingerprint is None:
            stage.replace(formal)
            _fsync_directory(results_root)
            _fsync_directory(stage.parent)
        elif stage_fingerprint is None and formal_fingerprint == new_tree:
            pass
        else:
            raise DemoRebuildError(
                "Promotion transaction is ambiguous: expected the new tree at exactly one of staging or formal"
            )
    elif new_tree == old_tree:
        if (
            stage_fingerprint == new_tree
            and formal_fingerprint == new_tree
            and archived_fingerprint is None
            and archived is not None
        ):
            _ensure_promotion_archive_parent(results_root, archived)
            stage.replace(archived)
            _fsync_directory(stage.parent)
            _fsync_directory(archived.parent)
        elif stage_fingerprint is None and formal_fingerprint == new_tree and archived_fingerprint == old_tree:
            pass
        else:
            raise DemoRebuildError("Promotion transaction is ambiguous for equivalent old/new trees")
    else:
        if (
            stage_fingerprint == new_tree
            and formal_fingerprint == old_tree
            and archived_fingerprint is None
            and archived is not None
        ):
            _ensure_promotion_archive_parent(results_root, archived)
            if not _exchange_directories(stage, formal):
                raise DemoRebuildError(
                    "This filesystem cannot atomically exchange staging and "
                    "formal result trees; refusing a promotion that would make "
                    "v1 temporarily disappear",
                )
            _fsync_directory(results_root)
            _fsync_directory(stage.parent)
            updated_journal = _write_promotion_journal(
                journal_path,
                updated_journal,
                status="exchanged",
            )
            stage_fingerprint = old_tree
            formal_fingerprint = new_tree
        if (
            stage_fingerprint == old_tree
            and formal_fingerprint == new_tree
            and archived_fingerprint is None
            and archived is not None
        ):
            _ensure_promotion_archive_parent(results_root, archived)
            stage.replace(archived)
            _fsync_directory(stage.parent)
            _fsync_directory(archived.parent)
        elif not (stage_fingerprint is None and formal_fingerprint == new_tree and archived_fingerprint == old_tree):
            raise DemoRebuildError(
                "Promotion transaction is ambiguous: staging/formal/archive "
                "do not match a recoverable prepared, exchanged, or completed state"
            )

    final_formal_fingerprint = _optional_directory_fingerprint(
        formal,
        label="promoted formal result directory",
    )
    final_stage_fingerprint = _optional_directory_fingerprint(
        stage,
        label="post-promotion staging result directory",
    )
    final_archived_fingerprint = (
        _optional_directory_fingerprint(
            archived,
            label="post-promotion archived result directory",
        )
        if archived is not None
        else None
    )
    if (
        final_formal_fingerprint != new_tree
        or final_stage_fingerprint is not None
        or final_archived_fingerprint != old_tree
    ):
        raise DemoRebuildError("Promotion transaction did not converge to its journaled tree identities")
    try:
        current_external_assets = _fingerprint_external_assets(
            _external_assets_from_artifacts(formal.rglob("*.npz")),
        )
    except (
        FileNotFoundError,
        OSError,
        ResultArtifactValidationError,
        RuntimeError,
        ValueError,
    ) as exc:
        raise DemoRebuildError("Promoted external object assets failed integrity validation") from exc
    if current_external_assets != expected_external_assets:
        raise DemoRebuildError("Promoted external object assets no longer match the validation journal")
    if journal_status not in {"filesystem_promoted", "completed"}:
        updated_journal = _write_promotion_journal(
            journal_path,
            updated_journal,
            status="filesystem_promoted",
        )
    final_report = updated_journal.get("final_report")
    if final_report is not None:
        if not isinstance(final_report, Mapping):
            raise DemoRebuildError("Promotion journal final_report is invalid")
        expected_report_path = run_report_root(results_root, run_id) / "report.json"
        if report_path is None:
            report_path = expected_report_path
        if report_path.expanduser().resolve() != expected_report_path:
            raise DemoRebuildError("Promotion report path escaped the fixed rebuild run directory")
        _atomic_write_json(report_path, final_report)
        updated_journal = _write_promotion_journal(
            journal_path,
            updated_journal,
            status="completed",
        )
    return (
        PromotionResult(
            promoted_root=formal,
            archived_root=archived,
            journal_path=journal_path,
        ),
        updated_journal,
    )


def promote_staging(
    results_root: Path,
    run_id: str,
    validation: ValidationSummary,
    *,
    report_path: Path | None = None,
    report_payload: Mapping[str, Any] | None = None,
) -> PromotionResult:
    """Durably journal, reconcile, and publish one validated staging tree."""

    if not validation.ok:
        raise DemoRebuildError("Refusing to promote a failed validation")
    if (report_path is None) != (report_payload is None):
        raise ValueError("report_path and report_payload must be provided together")
    results_root = results_root.expanduser().resolve()
    if results_root == Path(results_root.anchor):
        raise ValueError("results_root must not be a filesystem root")
    results_root.mkdir(parents=True, exist_ok=True)
    _assert_real_directory(results_root, label="results root")
    with _exclusive_results_state_lock(results_root):
        journal_path, journal = _prepare_promotion_journal(
            results_root=results_root,
            run_id=run_id,
            validation=validation,
            report_payload=report_payload,
        )
        promotion, _updated_journal = _commit_promotion_journal(
            results_root=results_root,
            run_id=run_id,
            journal_path=journal_path,
            journal=journal,
            report_path=report_path,
        )
    return promotion


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        temporary_path.replace(path)
        _fsync_directory(path.parent)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _read_batch_report(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as file:
        report = json.load(file)
    return report if isinstance(report, dict) else None


def _batch_report_file_identity(
    path: Path,
) -> tuple[int, int, int, int] | None:
    """Return enough metadata to prove that one invocation refreshed a report."""

    if not _lexists(path):
        return None
    path_stat = path.lstat()
    return (
        path_stat.st_ino,
        path_stat.st_size,
        path_stat.st_mtime_ns,
        path_stat.st_ctime_ns,
    )


def _batch_plan_digest(
    expected_items: Iterable[ExpectedArtifact],
    *,
    include_augmentation_variants: bool,
) -> str:
    """Bind one batch attempt to exact source and output job identities."""

    identity = {
        "include_augmentation_variants": include_augmentation_variants,
        "items": sorted(
            (
                str(item.source_path.resolve()),
                item.source_sha256,
                str(item.output_path.resolve()),
                item.config_sha256,
                item.variant,
                item.run_kind,
            )
            for item in expected_items
        )
    }
    return hashlib.sha256(
        json.dumps(
            identity,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _batch_rebuild_contract(
    *,
    outer_run_id: str,
    plan: BatchPlan,
    expected_items: Iterable[ExpectedArtifact],
    include_augmentation_variants: bool,
) -> dict[str, Any]:
    """Return the durable final-plan identity attached to a batch report."""

    items = tuple(expected_items)
    effective_augmentation = plan.spec.augmentation and include_augmentation_variants
    return {
        "schema_version": REBUILD_CONTRACT_SCHEMA_VERSION,
        "outer_run_id": outer_run_id,
        "batch_id": plan.batch_id,
        "include_augmentation_variants": include_augmentation_variants,
        "effective_augmentation": effective_augmentation,
        "plan_sha256": _batch_plan_digest(
            items,
            include_augmentation_variants=include_augmentation_variants,
        ),
        "planned_source_count": len({item.source_path.resolve() for item in items}),
        "planned_artifact_count": len(items),
    }


def _attach_batch_rebuild_contract(
    path: Path,
    *,
    outer_run_id: str,
    plan: BatchPlan,
    expected_items: Iterable[ExpectedArtifact],
    include_augmentation_variants: bool,
) -> None:
    """Atomically enrich a freshly written worker report with its final plan."""

    report, _report_sha256 = _load_trusted_batch_report(path)
    report["rebuild_contract"] = _batch_rebuild_contract(
        outer_run_id=outer_run_id,
        plan=plan,
        expected_items=expected_items,
        include_augmentation_variants=include_augmentation_variants,
    )
    _atomic_write_json(path, report)


def _validate_batch_rebuild_contract(
    report: Mapping[str, Any],
    *,
    outer_run_id: str,
    plan: BatchPlan,
    expected_items: Iterable[ExpectedArtifact],
    include_augmentation_variants: bool,
) -> None:
    """Fail closed unless persisted attempt evidence matches the final plan."""

    observed = report.get("rebuild_contract")
    if not isinstance(observed, Mapping):
        raise ValueError("Batch report has no rebuild_contract")
    required = _batch_rebuild_contract(
        outer_run_id=outer_run_id,
        plan=plan,
        expected_items=expected_items,
        include_augmentation_variants=include_augmentation_variants,
    )
    if dict(observed) != required:
        raise ValueError("Batch report rebuild_contract does not match the final source/config plan")


def _load_trusted_batch_report(
    path: Path,
) -> tuple[dict[str, Any], str]:
    """Load one real regular JSON report and bind its exact persisted bytes."""

    if not _lexists(path):
        raise FileNotFoundError(f"Batch report does not exist: {path}")
    path_stat = path.lstat()
    if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISREG(path_stat.st_mode):
        raise ValueError(f"Batch report must be a real regular file: {path}")
    report_bytes = path.read_bytes()
    try:
        report = json.loads(report_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Batch report is not valid UTF-8 JSON: {path}: {exc}") from exc
    if not isinstance(report, dict):
        raise ValueError(f"Batch report must contain a JSON object: {path}")
    return report, hashlib.sha256(report_bytes).hexdigest()


def _report_nonnegative_int(
    report: Mapping[str, Any],
    key: str,
) -> int:
    value = report.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Batch report {key} must be a non-negative integer")
    return value


def _report_absolute_path(
    value: Any,
    *,
    label: str,
) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Batch report {label} must be a non-empty string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"Batch report {label} must be absolute: {value!r}")
    return path.resolve()


def _expected_task_name(
    source_path: Path,
    *,
    task_type: str,
) -> str:
    return source_path.parent.name if task_type == "climbing" else source_path.stem


def _audit_one_batch_report(
    *,
    outer_run_id: str,
    plan: BatchPlan,
    stage: Path,
    report_path: Path,
    expected_items: tuple[ExpectedArtifact, ...],
    include_augmentation_variants: bool,
) -> tuple[
    dict[str, Any],
    str,
    tuple[Path, ...],
    tuple[Mapping[str, str], ...],
]:
    """Validate report identity and return successful and failed sources."""

    if plan.data_dir is None:
        raise ValueError(f"{plan.batch_id} has no data directory")
    report, report_sha256 = _load_trusted_batch_report(report_path)
    _validate_batch_rebuild_contract(
        report,
        outer_run_id=outer_run_id,
        plan=plan,
        expected_items=expected_items,
        include_augmentation_variants=include_augmentation_variants,
    )
    exact_identity = {
        "robot": (report.get("robot"), plan.robot),
        "task_type": (report.get("task_type"), plan.spec.task_type),
        "data_format": (
            report.get("data_format"),
            plan.spec.data_format,
        ),
    }
    mismatches = [key for key, (observed, required) in exact_identity.items() if observed != required]
    if mismatches:
        raise ValueError("Batch report identity does not match its plan: " + ", ".join(mismatches))
    if (
        _report_absolute_path(
            report.get("data_dir"),
            label="data_dir",
        )
        != plan.data_dir.resolve()
    ):
        raise ValueError("Batch report data_dir does not match its plan")
    if (
        _report_absolute_path(
            report.get("save_dir"),
            label="save_dir",
        )
        != stage
    ):
        raise ValueError("Batch report save_dir does not match staging")

    by_source: dict[Path, tuple[ExpectedArtifact, ...]] = {}
    grouped: dict[Path, list[ExpectedArtifact]] = defaultdict(list)
    for item in expected_items:
        grouped[item.source_path.resolve()].append(item)
    for source_path, source_items in grouped.items():
        by_source[source_path] = tuple(sorted(source_items, key=lambda item: item.variant))
    expected_sources = set(by_source)
    if _report_nonnegative_int(report, "total_files") != len(expected_sources):
        raise ValueError("Batch report total_files does not match the planned source count")

    manifest = report.get("manifest")
    if not isinstance(manifest, list):
        raise ValueError("Batch report manifest must be a list")
    manifest_sources: set[Path] = set()
    for entry in manifest:
        if not isinstance(entry, Mapping):
            raise ValueError("Batch report manifest entries must be objects")
        source_path = _report_absolute_path(
            entry.get("source_path"),
            label="manifest source_path",
        )
        if source_path in manifest_sources:
            raise ValueError(f"Batch report manifest duplicates source {source_path}")
        if source_path not in expected_sources:
            raise ValueError(f"Batch report manifest contains an unplanned source: {source_path}")
        required_task_name = _expected_task_name(
            source_path,
            task_type=plan.spec.task_type,
        )
        if entry.get("task_name") != required_task_name:
            raise ValueError(f"Batch report manifest task_name does not match source {source_path}")
        manifest_sources.add(source_path)
    if manifest_sources != expected_sources:
        missing = sorted(expected_sources.difference(manifest_sources))
        extra = sorted(manifest_sources.difference(expected_sources))
        raise ValueError(f"Batch report manifest is not the exact planned source set: missing={missing}, extra={extra}")

    results = report.get("results")
    failures = report.get("failures")
    if not isinstance(results, list) or not isinstance(failures, list):
        raise ValueError("Batch report results and failures must both be lists")
    output_owner = {
        item.output_path.resolve(): source_path
        for source_path, source_items in by_source.items()
        for item in source_items
    }
    successful_sources: list[Path] = []
    successful_statuses: list[str] = []
    outcome_sources: set[Path] = set()
    for result in results:
        if not isinstance(result, Mapping):
            raise ValueError("Batch report result entries must be objects")
        generated = result.get("generated_files")
        skipped = result.get("skipped_files")
        if (
            not isinstance(generated, list)
            or not all(isinstance(path, str) for path in generated)
            or not isinstance(skipped, list)
            or not all(isinstance(path, str) for path in skipped)
        ):
            raise ValueError("Batch report generated_files/skipped_files must be string lists")
        generated_paths = {
            _report_absolute_path(
                path,
                label="generated artifact path",
            )
            for path in generated
        }
        skipped_paths = {
            _report_absolute_path(
                path,
                label="skipped artifact path",
            )
            for path in skipped
        }
        if len(generated_paths) != len(generated) or len(skipped_paths) != len(skipped):
            raise ValueError("Batch report result contains duplicate artifact paths")
        if generated_paths.intersection(skipped_paths):
            raise ValueError("Batch report marks one artifact as both generated and skipped")
        result_paths = generated_paths.union(skipped_paths)
        if not result_paths:
            raise ValueError("Batch report result does not identify any planned artifacts")
        unplanned_paths = result_paths.difference(output_owner)
        if unplanned_paths:
            raise ValueError(f"Batch report result contains unplanned artifact paths: {sorted(unplanned_paths)}")
        owner_sources = {output_owner[path] for path in result_paths}
        if len(owner_sources) != 1:
            raise ValueError("Batch report result combines artifacts from multiple sources")
        source_path = owner_sources.pop()
        required_paths = {item.output_path.resolve() for item in by_source[source_path]}
        if result_paths != required_paths:
            raise ValueError(f"Batch report successful result does not cover every planned variant for {source_path}")
        if source_path in outcome_sources:
            raise ValueError(f"Batch report repeats source outcome {source_path}")
        required_task_name = _expected_task_name(
            source_path,
            task_type=plan.spec.task_type,
        )
        if result.get("task_name") != required_task_name:
            raise ValueError(f"Batch report result task_name does not match source {source_path}")
        required_status = "completed" if generated_paths else "skipped"
        if result.get("status") != required_status:
            raise ValueError("Batch report result status does not match generated artifacts")
        outcome_sources.add(source_path)
        successful_sources.append(source_path)
        successful_statuses.append(required_status)

    failed_sources: list[Mapping[str, str]] = []
    for failure in failures:
        if not isinstance(failure, Mapping):
            raise ValueError("Batch report failure entries must be objects")
        source_path = _report_absolute_path(
            failure.get("source_path"),
            label="failure source_path",
        )
        if source_path not in expected_sources:
            raise ValueError(f"Batch report failure contains an unplanned source: {source_path}")
        if source_path in outcome_sources:
            raise ValueError(f"Batch report repeats source outcome {source_path}")
        required_task_name = _expected_task_name(
            source_path,
            task_type=plan.spec.task_type,
        )
        if failure.get("task_name") != required_task_name:
            raise ValueError(f"Batch report failure task_name does not match source {source_path}")
        error = failure.get("error")
        if not isinstance(error, str) or not error.strip():
            raise ValueError("Batch report failure must contain a non-empty error")
        outcome_sources.add(source_path)
        failed_sources.append(
            {
                "source_path": str(source_path),
                "task_name": required_task_name,
                "error": error,
            }
        )
    if outcome_sources != expected_sources:
        missing = sorted(expected_sources.difference(outcome_sources))
        raise ValueError(f"Batch report does not record an outcome for every attempted source: {missing}")

    completed_tasks = successful_statuses.count("completed")
    skipped_tasks = successful_statuses.count("skipped")
    exact_counts = {
        "completed_tasks": (
            _report_nonnegative_int(report, "completed_tasks"),
            completed_tasks,
        ),
        "skipped_tasks": (
            _report_nonnegative_int(report, "skipped_tasks"),
            skipped_tasks,
        ),
        "failed_tasks": (
            _report_nonnegative_int(report, "failed_tasks"),
            len(failed_sources),
        ),
    }
    count_mismatches = [key for key, (observed, required) in exact_counts.items() if observed != required]
    if count_mismatches:
        raise ValueError("Batch report outcome counts are inconsistent: " + ", ".join(count_mismatches))
    required_report_status = "completed_with_failures" if failed_sources else "completed"
    if report.get("status") != required_report_status:
        raise ValueError("Batch report status does not match its failure set")
    return (
        report,
        report_sha256,
        tuple(sorted(successful_sources)),
        tuple(
            sorted(
                failed_sources,
                key=lambda item: item["source_path"],
            )
        ),
    )


def _empty_attempt_counts() -> dict[str, int]:
    return {
        "planned_source_count": 0,
        "attempted_source_count": 0,
        "succeeded_source_count": 0,
        "omitted_source_count": 0,
        "failed_source_count": 0,
        "unverified_source_count": 0,
        "planned_artifact_count": 0,
        "present_artifact_count": 0,
        "omitted_artifact_count": 0,
        "missing_required_artifact_count": 0,
    }


def _add_attempt_count(
    counts: dict[str, int],
    key: str,
    amount: int = 1,
) -> None:
    counts[key] += amount


def _e1_source_omission_allowed(
    cfg: RebuildDemoResultsConfig,
    *,
    task_type: str,
) -> bool:
    if not cfg.allow_e1_best_effort_omissions:
        return False
    if task_type in E1_DEFAULT_OMISSION_TASK_TYPES:
        return True
    return task_type == "robot_only" and cfg.allow_e1_robot_only_omissions


def _structural_failure_type(error: str) -> str | None:
    """Return the allowlisted solver failure type encoded by a worker report."""

    failure_type, separator, _message = error.partition(":")
    if not separator or failure_type not in STRUCTURAL_E1_FAILURE_TYPES:
        return None
    return failure_type


def _audit_persisted_batch_reports(
    cfg: RebuildDemoResultsConfig,
    plans: Iterable[BatchPlan],
    expected_artifacts: Iterable[ExpectedArtifact],
    *,
    stage: Path,
    report_root: Path,
    invocations: Mapping[str, _BatchInvocation] | None = None,
) -> _ExecutionAudit:
    """Rebuild the exact E1 omission set from durable batch reports."""

    stage = stage.resolve()
    expected = tuple(expected_artifacts)
    expected_by_batch: dict[str, tuple[ExpectedArtifact, ...]] = {}
    for plan in plans:
        expected_by_batch[plan.batch_id] = tuple(
            item for item in expected if item.robot == plan.robot and item.dataset_id == plan.spec.dataset_id
        )
    aggregate_counts = _empty_attempt_counts()
    strata: dict[tuple[str, str], dict[str, int]] = {}
    batch_summaries: list[dict[str, Any]] = []
    source_summaries: list[dict[str, Any]] = []
    omissions: list[dict[str, Any]] = []
    omitted_paths: set[Path] = set()
    blocking_failures: list[dict[str, str]] = []

    def update_strata(
        robot: str,
        task_type: str,
        key: str,
        amount: int = 1,
    ) -> None:
        _add_attempt_count(aggregate_counts, key, amount)
        counts = strata.setdefault(
            (robot, task_type),
            _empty_attempt_counts(),
        )
        _add_attempt_count(counts, key, amount)

    for item in expected:
        present = _lexists(item.output_path)
        update_strata(
            item.robot,
            item.task_type,
            "planned_artifact_count",
        )
        if present:
            update_strata(
                item.robot,
                item.task_type,
                "present_artifact_count",
            )

    for plan in plans:
        if not plan.available:
            continue
        batch_expected = expected_by_batch[plan.batch_id]
        grouped: dict[Path, tuple[ExpectedArtifact, ...]] = {}
        mutable_grouped: dict[Path, list[ExpectedArtifact]] = defaultdict(list)
        for item in batch_expected:
            mutable_grouped[item.source_path.resolve()].append(item)
        for source_path, source_items in mutable_grouped.items():
            grouped[source_path] = tuple(source_items)
        for _source_path in grouped:
            update_strata(
                plan.robot,
                plan.spec.task_type,
                "planned_source_count",
            )

        batch_report_path = report_root / "batches" / f"{plan.batch_id}.json"
        invocation = invocations.get(plan.batch_id) if invocations else None
        invalid_reason: str | None = None
        if invocation is not None and not invocation.report_refreshed:
            invalid_reason = "Batch invocation did not create or refresh its persistent report"
        try:
            (
                _batch_report,
                report_sha256,
                successful_sources,
                failed_sources,
            ) = _audit_one_batch_report(
                outer_run_id=cfg.run_id,
                plan=plan,
                stage=stage,
                report_path=batch_report_path,
                expected_items=batch_expected,
                include_augmentation_variants=cfg.include_augmentation_variants,
            )
        except (
            FileNotFoundError,
            OSError,
            UnicodeDecodeError,
            ValueError,
        ) as exc:
            if invalid_reason is None:
                invalid_reason = f"{type(exc).__name__}: {exc}"
            report_sha256 = ""
            successful_sources = ()
            failed_sources = ()

        if invalid_reason is not None:
            invocation_suffix = (
                f"; invocation error: {invocation.error}"
                if invocation is not None and invocation.error is not None
                else ""
            )
            error = invalid_reason + invocation_suffix
            blocking_failures.append(
                {
                    "batch_id": plan.batch_id,
                    "error": error,
                }
            )
            for source_path in sorted(grouped):
                update_strata(
                    plan.robot,
                    plan.spec.task_type,
                    "unverified_source_count",
                )
                source_summaries.append(
                    {
                        "batch_id": plan.batch_id,
                        "robot": plan.robot,
                        "dataset": plan.spec.dataset_id,
                        "task_type": plan.spec.task_type,
                        "data_format": plan.spec.data_format,
                        "source_path": str(source_path),
                        "task_name": _expected_task_name(
                            source_path,
                            task_type=plan.spec.task_type,
                        ),
                        "status": "unverified",
                        "error": error,
                        "omitted_variants": [],
                    }
                )
            batch_summaries.append(
                {
                    "batch_id": plan.batch_id,
                    "robot": plan.robot,
                    "dataset": plan.spec.dataset_id,
                    "task_type": plan.spec.task_type,
                    "data_format": plan.spec.data_format,
                    "effective_augmentation": _effective_augmentation(cfg, plan),
                    "status": "invalid_report",
                    "report_path": str(batch_report_path),
                    "report_sha256": report_sha256,
                    "attempted_source_count": 0,
                    "succeeded_source_count": 0,
                    "omitted_source_count": 0,
                    "failed_source_count": 0,
                    "unverified_source_count": len(grouped),
                    "invocation_error": (invocation.error if invocation is not None else None),
                    "error": error,
                }
            )
            continue

        for _source_path in grouped:
            update_strata(
                plan.robot,
                plan.spec.task_type,
                "attempted_source_count",
            )
        successful_set = set(successful_sources)
        failed_by_source = {Path(failure["source_path"]).resolve(): failure for failure in failed_sources}
        for source_path in sorted(successful_set):
            update_strata(
                plan.robot,
                plan.spec.task_type,
                "succeeded_source_count",
            )
            source_summaries.append(
                {
                    "batch_id": plan.batch_id,
                    "robot": plan.robot,
                    "dataset": plan.spec.dataset_id,
                    "task_type": plan.spec.task_type,
                    "data_format": plan.spec.data_format,
                    "source_path": str(source_path),
                    "task_name": _expected_task_name(
                        source_path,
                        task_type=plan.spec.task_type,
                    ),
                    "status": "succeeded",
                    "error": None,
                    "omitted_variants": [],
                }
            )

        batch_omitted_sources = 0
        batch_failed_sources = 0
        all_failures_accepted = bool(failed_by_source)
        for source_path, failure in sorted(failed_by_source.items()):
            source_items = grouped[source_path]
            missing_items = tuple(item for item in source_items if not _lexists(item.output_path))
            policy_allowed = plan.robot == "e1" and _e1_source_omission_allowed(
                cfg,
                task_type=plan.spec.task_type,
            )
            structural_failure_type = _structural_failure_type(
                failure["error"],
            )
            accepted = policy_allowed and structural_failure_type is not None and bool(missing_items)
            if accepted:
                omitted_variants = sorted(item.variant for item in missing_items)
                omitted_artifact_paths = sorted(str(item.output_path.resolve()) for item in missing_items)
                evidence = {
                    "batch_id": plan.batch_id,
                    "batch_report_path": str(batch_report_path),
                    "batch_report_sha256": report_sha256,
                    "robot": plan.robot,
                    "dataset": plan.spec.dataset_id,
                    "task_type": plan.spec.task_type,
                    "data_format": plan.spec.data_format,
                    "source_path": str(source_path),
                    "task_name": failure["task_name"],
                    "error": failure["error"],
                    "failure_type": structural_failure_type,
                    "omitted_variants": omitted_variants,
                    "omitted_artifact_paths": omitted_artifact_paths,
                }
                omissions.append(evidence)
                omitted_paths.update(item.output_path.resolve() for item in missing_items)
                update_strata(
                    plan.robot,
                    plan.spec.task_type,
                    "omitted_source_count",
                )
                update_strata(
                    plan.robot,
                    plan.spec.task_type,
                    "omitted_artifact_count",
                    len(missing_items),
                )
                batch_omitted_sources += 1
                source_status = "omitted"
            else:
                all_failures_accepted = False
                omitted_variants = []
                if policy_allowed and not missing_items:
                    policy_reason = "failure report has a complete artifact family and cannot justify an omission"
                elif policy_allowed and structural_failure_type is None:
                    policy_reason = (
                        "only allowlisted SQPNonlinearFeasibilityError failures can justify an E1 structural omission"
                    )
                elif plan.robot == "g1":
                    policy_reason = "G1 source failures are never eligible for omission"
                elif plan.robot != "e1":
                    policy_reason = f"robot {plan.robot!r} is not eligible for E1 omissions"
                elif plan.spec.task_type == "robot_only" and not cfg.allow_e1_robot_only_omissions:
                    policy_reason = "E1 robot-only omissions require the explicit allow_e1_robot_only_omissions option"
                else:
                    policy_reason = "E1 best-effort omissions are disabled for this task"
                blocking_failures.append(
                    {
                        "batch_id": plan.batch_id,
                        "source_path": str(source_path),
                        "error": (f"{failure['error']} ({policy_reason})"),
                    }
                )
                update_strata(
                    plan.robot,
                    plan.spec.task_type,
                    "failed_source_count",
                )
                batch_failed_sources += 1
                source_status = "failed"
            source_summaries.append(
                {
                    "batch_id": plan.batch_id,
                    "robot": plan.robot,
                    "dataset": plan.spec.dataset_id,
                    "task_type": plan.spec.task_type,
                    "data_format": plan.spec.data_format,
                    "source_path": str(source_path),
                    "task_name": failure["task_name"],
                    "status": source_status,
                    "error": failure["error"],
                    "omitted_variants": omitted_variants,
                }
            )

        if (
            invocation is not None
            and invocation.error is not None
            and not all_failures_accepted
            and not failed_by_source
        ):
            blocking_failures.append(
                {
                    "batch_id": plan.batch_id,
                    "error": (f"Batch invocation raised despite a report with no failed sources: {invocation.error}"),
                }
            )
        if batch_failed_sources:
            batch_status = "failed"
        elif batch_omitted_sources:
            batch_status = "completed_with_omissions"
        else:
            batch_status = "completed"
        batch_summaries.append(
            {
                "batch_id": plan.batch_id,
                "robot": plan.robot,
                "dataset": plan.spec.dataset_id,
                "task_type": plan.spec.task_type,
                "data_format": plan.spec.data_format,
                "effective_augmentation": _effective_augmentation(cfg, plan),
                "status": batch_status,
                "report_path": str(batch_report_path),
                "report_sha256": report_sha256,
                "attempted_source_count": len(grouped),
                "succeeded_source_count": len(successful_sources),
                "omitted_source_count": batch_omitted_sources,
                "failed_source_count": batch_failed_sources,
                "unverified_source_count": 0,
                "invocation_error": (invocation.error if invocation is not None else None),
                "error": None,
            }
        )

    aggregate_counts["missing_required_artifact_count"] = (
        aggregate_counts["planned_artifact_count"]
        - aggregate_counts["present_artifact_count"]
        - aggregate_counts["omitted_artifact_count"]
    )
    for counts in strata.values():
        counts["missing_required_artifact_count"] = (
            counts["planned_artifact_count"] - counts["present_artifact_count"] - counts["omitted_artifact_count"]
        )
    by_robot: dict[str, dict[str, Any]] = {}
    for robot in dict.fromkeys(cfg.robots):
        robot_counts = _empty_attempt_counts()
        by_task: dict[str, dict[str, int]] = {}
        for (stratum_robot, task_type), counts in sorted(strata.items()):
            if stratum_robot != robot:
                continue
            by_task[task_type] = dict(counts)
            for key, value in counts.items():
                robot_counts[key] += value
        by_robot[robot] = {
            **robot_counts,
            "by_task": by_task,
        }

    omission_identity = [
        {
            "batch_id": item["batch_id"],
            "batch_report_sha256": item["batch_report_sha256"],
            "source_path": item["source_path"],
            "error": item["error"],
            "omitted_artifact_paths": item["omitted_artifact_paths"],
        }
        for item in sorted(
            omissions,
            key=lambda entry: (
                entry["batch_id"],
                entry["source_path"],
            ),
        )
    ]
    omission_set_sha256 = hashlib.sha256(
        json.dumps(
            omission_identity,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    report = {
        "ok": not blocking_failures,
        "policy": {
            "include_augmentation_variants": (cfg.include_augmentation_variants),
            "allow_e1_best_effort_omissions": (cfg.allow_e1_best_effort_omissions),
            "default_e1_task_types": sorted(E1_DEFAULT_OMISSION_TASK_TYPES),
            "allow_e1_robot_only_omissions": (cfg.allow_e1_robot_only_omissions),
            "g1_requires_complete_success": True,
            "existing_artifacts_are_always_validated": True,
        },
        "source_job_counts": aggregate_counts,
        "by_robot": by_robot,
        "batches": sorted(
            batch_summaries,
            key=lambda item: item["batch_id"],
        ),
        "sources": sorted(
            source_summaries,
            key=lambda item: (
                item["batch_id"],
                item["source_path"],
            ),
        ),
        "omissions": sorted(
            omissions,
            key=lambda item: (
                item["batch_id"],
                item["source_path"],
            ),
        ),
        "omission_set_sha256": omission_set_sha256,
        "issues": list(blocking_failures),
    }
    return _ExecutionAudit(
        ok=not blocking_failures,
        omitted_artifact_paths=frozenset(omitted_paths),
        blocking_failures=tuple(blocking_failures),
        report=report,
    )


def _full_matrix_selected(cfg: RebuildDemoResultsConfig) -> bool:
    selected_robots = set(cfg.robots)
    selected_datasets = set(selected_dataset_ids(cfg))
    return selected_robots == set(SUPPORTED_ROBOTS) and CORE_DATASET_IDS.issubset(selected_datasets)


def _formal_quality_profile_relaxations(
    cfg: RebuildDemoResultsConfig,
) -> dict[str, dict[str, float]]:
    """Return quality limits that are wider than the formal publication profile."""

    return {
        name: {
            "configured": float(getattr(cfg, name)),
            "formal_maximum": formal_maximum,
        }
        for name, formal_maximum in FORMAL_QUALITY_LIMITS.items()
        if float(getattr(cfg, name)) > formal_maximum
    }


def _augmentation_plan_identity(
    cfg: RebuildDemoResultsConfig,
    plans: Iterable[BatchPlan],
) -> dict[str, Any]:
    """Return the explicit global and per-batch augmentation plan identity."""

    return {
        "include_augmentation_variants": cfg.include_augmentation_variants,
        "effective_augmentation_by_batch": {
            plan.batch_id: _effective_augmentation(cfg, plan)
            for plan in plans
        },
    }


def _planning_summary(
    expected: Iterable[ExpectedArtifact],
    *,
    cfg: RebuildDemoResultsConfig,
    plans: Iterable[BatchPlan],
) -> dict[str, Any]:
    items = tuple(expected)
    by_batch: dict[str, dict[str, int]] = {}
    batch_groups: dict[str, list[ExpectedArtifact]] = defaultdict(list)
    for item in items:
        batch_groups[f"{item.robot}-{item.dataset_id}"].append(item)
    for batch_id, batch_items in sorted(batch_groups.items()):
        by_batch[batch_id] = {
            "expected_artifact_count": len(batch_items),
            "expected_source_count": len({item.source_path for item in batch_items}),
        }
    return {
        "ok": True,
        **_augmentation_plan_identity(cfg, plans),
        "expected_artifact_count": len(items),
        "expected_source_count": len({item.source_path for item in items}),
        "expected_source_job_count": len({(item.robot, item.dataset_id, item.source_path) for item in items}),
        "output_collision_count": 0,
        "collision_paths": [],
        "by_batch": by_batch,
    }


def _planning_failure_summary(
    exc: Exception,
    *,
    cfg: RebuildDemoResultsConfig,
    plans: Iterable[BatchPlan],
) -> dict[str, Any]:
    if isinstance(exc, ArtifactPlanCollisionError):
        expected_artifact_count = exc.expected_artifact_count
        expected_source_count = exc.expected_source_count
        collision_paths = [str(path) for path in exc.collision_paths[:MAX_REPORTED_ISSUES]]
        collision_count = len(exc.collision_paths)
    else:
        expected_artifact_count = 0
        expected_source_count = 0
        collision_paths = []
        collision_count = 0
    return {
        "ok": False,
        **_augmentation_plan_identity(cfg, plans),
        "expected_artifact_count": expected_artifact_count,
        "expected_source_count": expected_source_count,
        "expected_source_job_count": 0,
        "output_collision_count": collision_count,
        "collision_paths": collision_paths,
        "by_batch": {},
        "error": f"{type(exc).__name__}: {exc}",
    }


def _recover_promotion_if_present(
    *,
    results_root: Path,
    run_id: str,
    report_path: Path,
    include_augmentation_variants: bool,
) -> dict[str, Any] | None:
    journal_path = promotion_journal_path(results_root, run_id)
    if not _lexists(journal_path):
        return None
    with _exclusive_results_state_lock(results_root):
        journal = _load_promotion_journal(journal_path)
        journal_report = journal.get("final_report")
        observed_augmentation_mode = (
            journal_report.get("include_augmentation_variants")
            if isinstance(journal_report, Mapping)
            else None
        )
        if (
            not isinstance(journal_report, Mapping)
            or not isinstance(observed_augmentation_mode, bool)
            or observed_augmentation_mode != include_augmentation_variants
        ):
            raise DemoRebuildError(
                "Promotion journal augmentation mode does not match the requested rebuild",
            )
        _promotion, updated_journal = _commit_promotion_journal(
            results_root=results_root,
            run_id=run_id,
            journal_path=journal_path,
            journal=journal,
            report_path=report_path,
        )
    final_report = updated_journal.get("final_report")
    if not isinstance(final_report, dict):
        raise DemoRebuildError("Promotion journal has no aggregate report to recover")
    return final_report


def _run_rebuild_locked(
    cfg: RebuildDemoResultsConfig,
) -> dict[str, Any]:
    """Execute one rebuild while its run-level lock is held."""

    _validate_config(cfg)
    results_root = cfg.results_root.expanduser().resolve()
    stage = staging_root(results_root, cfg.run_id)
    report_root = run_report_root(results_root, cfg.run_id)
    report_path = report_root / "report.json"
    if _lexists(promotion_journal_path(results_root, cfg.run_id)):
        if not cfg.promote:
            raise DemoRebuildError(
                "This run_id already has a promotion transaction; rerun with "
                "--promote to reconcile it instead of starting another rebuild"
            )
        recovered_report = _recover_promotion_if_present(
            results_root=results_root,
            run_id=cfg.run_id,
            report_path=report_path,
            include_augmentation_variants=(cfg.include_augmentation_variants),
        )
        if recovered_report is None:
            raise DemoRebuildError("Promotion journal disappeared during recovery")
        return recovered_report
    plans = build_rebuild_matrix(cfg)
    batch_results: list[dict[str, Any]] = []
    batch_failures: list[dict[str, str]] = []

    for plan in plans:
        if plan.available:
            continue
        batch_report_path = report_root / "batches" / f"{plan.batch_id}.json"
        outcome = {
            "batch_id": plan.batch_id,
            "robot": plan.robot,
            "dataset": plan.spec.dataset_id,
            "effective_augmentation": _effective_augmentation(cfg, plan),
            "status": "skipped" if plan.spec.optional else "failed",
            "reason": plan.skip_reason,
            "report_path": str(batch_report_path),
        }
        batch_results.append(outcome)
        if not plan.spec.optional:
            batch_failures.append(
                {
                    "batch_id": plan.batch_id,
                    "error": str(plan.skip_reason),
                }
            )

    expected: tuple[ExpectedArtifact, ...] = ()
    planning_error: Exception | None = None
    if batch_failures:
        planning_error = FileNotFoundError("Required rebuild dataset roots are unavailable")
    else:
        try:
            expected = plan_expected_artifacts(
                plans,
                stage=stage,
                cfg=cfg,
            )
        except Exception as exc:
            planning_error = exc
            batch_failures.append(
                {
                    "batch_id": "expected-artifact-plan",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    planning = (
        _planning_summary(
            expected,
            cfg=cfg,
            plans=plans,
        )
        if planning_error is None
        else _planning_failure_summary(
            planning_error,
            cfg=cfg,
            plans=plans,
        )
    )

    report: dict[str, Any] = {
        "run_id": cfg.run_id,
        "status": "planning",
        "data_root": str(cfg.data_root.expanduser().resolve()),
        "results_root": str(results_root),
        "staging_root": str(stage),
        "robots": list(dict.fromkeys(cfg.robots)),
        "datasets": list(selected_dataset_ids(cfg)),
        "include_augmentation_variants": (cfg.include_augmentation_variants),
        "max_workers": cfg.max_workers,
        "overwrite": cfg.overwrite,
        "dry_run": cfg.dry_run,
        "validate_only": cfg.validate_only,
        "promote_requested": cfg.promote,
        "release_object_non_penetration_on_infeasible_by_robot": {
            robot: _release_object_non_penetration_for_plan(
                cfg,
                next(plan for plan in plans if plan.robot == robot),
            )
            for robot in dict.fromkeys(cfg.robots)
        },
        "allow_e1_best_effort_omissions": (cfg.allow_e1_best_effort_omissions),
        "allow_e1_robot_only_omissions": (cfg.allow_e1_robot_only_omissions),
        "quality_limits": {
            "fallback_frame_fraction_per_artifact": (cfg.max_fallback_frame_fraction_per_artifact),
            "release_union_frame_fraction_per_artifact": (cfg.max_release_union_frame_fraction_per_artifact),
            "full_sequence_retry_fraction_per_artifact": (cfg.max_full_sequence_retry_fraction_per_artifact),
            "fallback_artifact_fraction_per_group": (cfg.max_fallback_artifact_fraction_per_group),
            "foot_release_artifact_fraction_per_group": (cfg.max_foot_release_artifact_fraction_per_group),
            "object_release_artifact_fraction_per_group": (cfg.max_object_release_artifact_fraction_per_group),
            "release_union_artifact_fraction_per_group": (cfg.max_release_union_artifact_fraction_per_group),
            "full_sequence_retry_artifact_fraction_per_group": (
                cfg.max_full_sequence_retry_artifact_fraction_per_group
            ),
        },
        "planning": planning,
        "batches": batch_results,
        "batch_failures": batch_failures,
        "execution_audit": None,
        "validation": None,
        "promotion": None,
    }

    if cfg.dry_run:
        batch_results.extend(
            {
                "batch_id": plan.batch_id,
                "robot": plan.robot,
                "dataset": plan.spec.dataset_id,
                "effective_augmentation": _effective_augmentation(cfg, plan),
                "status": ("planned" if planning_error is None else "planning_failed"),
                "report_path": str(report_root / "batches" / f"{plan.batch_id}.json"),
            }
            for plan in plans
            if plan.available
        )
        report["status"] = "dry_run" if planning_error is None and not batch_failures else "dry_run_failed"
        _atomic_write_json(report_path, report)
        if report["status"] != "dry_run":
            raise DemoRebuildError(f"Demo rebuild planning failed; see {report_path}")
        return report

    if planning_error is not None or batch_failures:
        batch_results.extend(
            {
                "batch_id": plan.batch_id,
                "robot": plan.robot,
                "dataset": plan.spec.dataset_id,
                "effective_augmentation": _effective_augmentation(cfg, plan),
                "status": "planning_failed",
                "report_path": str(report_root / "batches" / f"{plan.batch_id}.json"),
            }
            for plan in plans
            if plan.available
        )
        report["status"] = "planning_failed"
        _atomic_write_json(report_path, report)
        raise DemoRebuildError(f"Demo rebuild planning failed; see {report_path}")

    invocations: dict[str, _BatchInvocation] | None = None if cfg.validate_only else {}
    if not cfg.validate_only:
        stage.mkdir(parents=True, exist_ok=True)
        expected_by_batch = {
            plan.batch_id: tuple(
                item for item in expected if item.robot == plan.robot and item.dataset_id == plan.spec.dataset_id
            )
            for plan in plans
            if plan.available
        }
        for plan in plans:
            if not plan.available:
                continue
            batch_report_path = report_root / "batches" / f"{plan.batch_id}.json"
            parallel_cfg = _parallel_config(
                cfg,
                plan,
                stage=stage,
                batch_report_path=batch_report_path,
            )
            previous_report_identity = _batch_report_file_identity(batch_report_path)
            invocation_error: str | None = None
            try:
                parallel_robot_retarget.main(parallel_cfg)
            except Exception as exc:
                invocation_error = f"{type(exc).__name__}: {exc}"
            current_report_identity = _batch_report_file_identity(batch_report_path)
            report_refreshed = (
                current_report_identity is not None and current_report_identity != previous_report_identity
            )
            if report_refreshed:
                try:
                    _attach_batch_rebuild_contract(
                        batch_report_path,
                        outer_run_id=cfg.run_id,
                        plan=plan,
                        expected_items=expected_by_batch[plan.batch_id],
                        include_augmentation_variants=(cfg.include_augmentation_variants),
                    )
                except Exception as exc:
                    contract_error = f"Could not attach final rebuild contract: {type(exc).__name__}: {exc}"
                    invocation_error = (
                        f"{invocation_error}; {contract_error}" if invocation_error is not None else contract_error
                    )
                    report_refreshed = False
            assert invocations is not None
            invocations[plan.batch_id] = _BatchInvocation(
                report_refreshed=report_refreshed,
                error=invocation_error,
            )

    execution_audit = _audit_persisted_batch_reports(
        cfg,
        plans,
        expected,
        stage=stage,
        report_root=report_root,
        invocations=invocations,
    )
    batch_results.extend(execution_audit.report["batches"])
    batch_failures.extend(dict(failure) for failure in execution_audit.blocking_failures)
    report["execution_audit"] = dict(execution_audit.report)

    validation = validate_rebuild_directory(
        stage,
        expected,
        omitted_artifact_paths=(execution_audit.omitted_artifact_paths),
        max_fallback_frame_fraction_per_artifact=(cfg.max_fallback_frame_fraction_per_artifact),
        max_release_union_frame_fraction_per_artifact=(cfg.max_release_union_frame_fraction_per_artifact),
        max_full_sequence_retry_fraction_per_artifact=(cfg.max_full_sequence_retry_fraction_per_artifact),
        max_fallback_artifact_fraction_per_group=(cfg.max_fallback_artifact_fraction_per_group),
        max_foot_release_artifact_fraction_per_group=(cfg.max_foot_release_artifact_fraction_per_group),
        max_object_release_artifact_fraction_per_group=(cfg.max_object_release_artifact_fraction_per_group),
        max_release_union_artifact_fraction_per_group=(cfg.max_release_union_artifact_fraction_per_group),
        max_full_sequence_retry_artifact_fraction_per_group=(cfg.max_full_sequence_retry_artifact_fraction_per_group),
    )
    report["validation"] = validation.to_dict()
    execution_report = dict(report["execution_audit"])
    execution_counts = dict(execution_report["source_job_counts"])
    execution_counts["validated_artifact_count"] = int(validation.statistics["valid_artifacts"])
    execution_report["source_job_counts"] = execution_counts
    report["execution_audit"] = execution_report

    validation_ok = validation.ok and execution_audit.ok
    if cfg.promote:
        if not _full_matrix_selected(cfg):
            batch_failures.append(
                {
                    "batch_id": "promotion",
                    "error": (
                        "Promotion requires both G1 and E1 plus every required dataset, including noetix_csv_climb"
                    ),
                }
            )
        elif quality_relaxations := _formal_quality_profile_relaxations(cfg):
            batch_failures.append(
                {
                    "batch_id": "promotion",
                    "error": (f"Promotion quality limits are wider than the formal profile: {quality_relaxations}"),
                }
            )
        elif batch_failures or not validation_ok:
            batch_failures.append(
                {
                    "batch_id": "promotion",
                    "error": "Promotion skipped because execution or validation failed",
                }
            )
        else:
            try:
                promotion = promote_staging(
                    results_root,
                    cfg.run_id,
                    validation,
                    report_path=report_path,
                    report_payload=report,
                )
                persisted_report = _read_batch_report(report_path)
                if persisted_report is None:
                    raise DemoRebuildError("Promotion completed without its aggregate report")
                report = persisted_report
                report["promotion"] = {
                    "promoted_root": str(promotion.promoted_root),
                    "archived_root": (str(promotion.archived_root) if promotion.archived_root is not None else None),
                    "journal_path": str(promotion.journal_path),
                }
            except Exception as exc:
                batch_failures.append(
                    {
                        "batch_id": "promotion",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    report["batch_failures"] = batch_failures
    failed = bool(batch_failures) or not validation_ok
    if failed:
        report["status"] = "validation_failed"
    elif report["promotion"] is not None:
        report["status"] = "promoted"
    else:
        report["status"] = "validated"
    _atomic_write_json(report_path, report)
    if failed:
        raise DemoRebuildError(f"Demo rebuild failed; see {report_path}")
    return report


def run_rebuild(cfg: RebuildDemoResultsConfig) -> dict[str, Any]:
    """Execute or validate one resumable, single-writer rebuild."""

    _validate_config(cfg)
    results_root = cfg.results_root.expanduser().resolve()
    with _exclusive_rebuild_run_lock(results_root, cfg.run_id):
        return _run_rebuild_locked(cfg)


def main(cfg: RebuildDemoResultsConfig) -> None:
    """CLI entry point."""

    report = run_rebuild(cfg)
    print(
        f"Demo rebuild status={report['status']}; "
        f"report={run_report_root(cfg.results_root, cfg.run_id) / 'report.json'}"
    )


if __name__ == "__main__":
    main(tyro.cli(RebuildDemoResultsConfig))
