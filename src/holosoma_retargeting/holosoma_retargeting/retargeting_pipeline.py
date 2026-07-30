# ruff: noqa: CPY001, PLR0917

"""
Shared robot-retargeting pipeline for all task types:
- robot_only: Robot-only retargeting with ground interaction
- object_interaction: Object manipulation retargeting (InterMimic)
- climbing: Climbing retargeting with dynamic terrain
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import logging
import os
import re
import tempfile
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Literal
from urllib.parse import quote

import numpy as np

from holosoma_retargeting.config_types.data_type import MotionDataConfig, normalize_data_format
from holosoma_retargeting.config_types.retargeter import RetargeterConfig
from holosoma_retargeting.config_types.retargeting import (
    RetargetingConfig,
    validate_production_task,
)
from holosoma_retargeting.config_types.robot import RobotConfig
from holosoma_retargeting.config_types.task import TaskConfig
from holosoma_retargeting.data_utils.motion_data import (
    get_motion_format_spec,
    load_human_motion,
    resolve_motion_path,
    validate_motion_skeleton_contract,
    validate_motion_task,
)
from holosoma_retargeting.data_utils.object_assets import (
    create_omomo_object_scene,
    create_scaled_omomo_object_urdf,
    get_omomo_object_asset,
)
from holosoma_retargeting.data_utils.omomo import parse_omomo_sequence_name
from holosoma_retargeting.result_artifact import (
    RESULT_SCHEMA_VERSION,
    ResultArtifactValidationError,
    validate_result_artifact,
    validate_result_external_assets,
)
from holosoma_retargeting.retargeting_fingerprint import (
    solver_dependency_fingerprint,
    solver_implementation_fingerprint,
)
from holosoma_retargeting.src.interaction_mesh_retargeter import (
    InteractionMeshRetargeter,  # type: ignore[import-not-found]
)
from holosoma_retargeting.src.utils import (
    augment_object_poses,
    create_new_scene_xml_file,
    create_scaled_multi_boxes_urdf,
    create_scaled_multi_boxes_xml,
    estimate_human_orientation,
    estimate_mocap_foot_orientation,
    estimate_smpl_orientation,
    extract_foot_sticking_sequence_velocity,
    extract_object_first_moving_frame,
    load_object_data,
    preprocess_motion_data,
    transform_from_human_to_world,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ----------------------------- Constants -----------------------------

# Task-specific defaults
DEFAULT_DATA_FORMATS = {
    "robot_only": "omomo",
    "object_interaction": "omomo",
    "climbing": "mocap",
}

# Constants for numpy arrays (not in dataclass to avoid tyro parsing issues)
_OBJECT_SCALE_AUGMENTED = np.array([1.0, 1.0, 1.2])
_OBJECT_SCALE_NORMAL = np.array([1.0, 1.0, 1.0])
_AUGMENTATION_TRANSLATION = np.array([0.2, 0.0, 0.0])


# Type aliases
TaskType = Literal["robot_only", "object_interaction", "climbing"]
RunKind = Literal["single", "augmentation", "ablation"]


@dataclass(frozen=True)
class RetargetVariant:
    """One semantic motion variant, independent of serial/parallel execution."""

    name: str = "identity"
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: float = 0.0
    object_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.name) is None:
            raise ValueError("Variant names must contain only letters, digits, dot, underscore, and hyphen")
        translation = np.asarray(self.translation, dtype=float)
        scale = np.asarray(self.object_scale, dtype=float)
        if translation.shape != (3,) or not np.isfinite(translation).all():
            raise ValueError("Variant translation must contain three finite values")
        if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("Variant object_scale must contain three positive finite values")
        if not np.isfinite(self.rotation):
            raise ValueError("Variant rotation must be finite")
        if self.name == "identity" and (
            not np.allclose(translation, 0.0) or not np.isclose(self.rotation, 0.0) or not np.allclose(scale, 1.0)
        ):
            raise ValueError("The reserved 'identity' variant must not transform motion")

    @property
    def changes_motion(self) -> bool:
        return (
            not np.allclose(self.translation, 0.0)
            or not np.isclose(self.rotation, 0.0)
            or not np.allclose(self.object_scale, 1.0)
        )

    @property
    def is_identity(self) -> bool:
        return self.name == "identity" and not self.changes_motion


IDENTITY_VARIANT = RetargetVariant()


@dataclass(frozen=True)
class RetargetJob:
    """Fully normalized description of one solver invocation."""

    config: RetargetingConfig
    source_path: Path
    output_path: Path
    baseline_path: Path
    state_lock_path: Path
    output_lock_path: Path
    baseline_lock_path: Path
    generated_assets_dir: Path
    sequence_key: str
    dataset_partition: str
    run_kind: RunKind
    variant: RetargetVariant
    experiment_name: str | None
    source_sha256: str
    source_size: int
    source_mtime_ns: int
    config_json: str
    config_sha256: str
    baseline_source_sha256: str
    baseline_config_json: str
    baseline_config_sha256: str


@dataclass(frozen=True)
class RetargetJobResult:
    """Minimal execution result shared by single, batch, and experiments."""

    output_path: Path
    source_path: Path
    sequence_key: str
    variant: str
    resumed: bool = False


@dataclass(frozen=True)
class RetargetFamilyResult:
    """Results from one single-action invocation, ordered by variant family."""

    results: tuple[RetargetJobResult, ...]

    def __post_init__(self) -> None:
        if not self.results:
            raise ValueError("A retargeting family must contain at least one result")
        if self.results[0].variant != IDENTITY_VARIANT.name:
            raise ValueError("A retargeting family must begin with the identity result")
        variants = tuple(result.variant for result in self.results)
        if len(set(variants)) != len(variants):
            raise ValueError("A retargeting family must contain unique variants")

    @property
    def identity(self) -> RetargetJobResult:
        """Return the identity result, which is always first."""

        return self.results[0]


def planned_variants(
    task_type: TaskType,
    *,
    augmentation: bool,
) -> tuple[RetargetVariant, ...]:
    """Return the canonical variant family for one task type."""

    variants = [IDENTITY_VARIANT]
    if not augmentation or task_type == "robot_only":
        return tuple(variants)
    if task_type == "object_interaction":
        variants.extend(
            (
                RetargetVariant(name="trans_0", translation=(0.2, 0.0, 0.0)),
                RetargetVariant(name="trans_1", translation=(0.0, 0.2, 0.0)),
                RetargetVariant(name="trans_2", translation=(0.0, -0.2, 0.0)),
                RetargetVariant(
                    name="rot_0",
                    translation=(0.0, 0.2, 0.0),
                    rotation=float(np.pi / 4.0),
                ),
                RetargetVariant(
                    name="rot_1",
                    translation=(0.0, -0.2, 0.0),
                    rotation=float(-np.pi / 4.0),
                ),
            )
        )
        return tuple(variants)
    if task_type == "climbing":
        variants.extend(
            RetargetVariant(
                name=f"z_scale_{str(scale).replace('.', 'p')}",
                object_scale=(1.0, 1.0, scale),
            )
            for scale in (0.8, 0.9, 1.1, 1.2)
        )
        return tuple(variants)
    raise ValueError(f"Unknown task type: {task_type}")


def _json_ready(value):
    if isinstance(value, Path):
        return str(value.expanduser().resolve())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _solver_dependency_paths(
    normalized: RetargetingConfig,
) -> dict[str, Path]:
    """Resolve static inputs that can change a solve without changing config."""

    robot_urdf = Path(normalized.robot_config.ROBOT_URDF_FILE).expanduser()
    if not robot_urdf.is_absolute():
        robot_urdf = Path(__file__).resolve().parent / robot_urdf
    robot_urdf = robot_urdf.resolve(strict=True)
    dependencies = {
        "robot_model_tree": robot_urdf.parent,
    }

    if normalized.data_format == "omomo" and normalized.motion_data_config.human_height is None:
        logical_data_path = Path(normalized.data_path).expanduser().absolute()
        dependencies["omomo_height_table"] = logical_data_path.parent / "height_dict.pkl"

    if normalized.task_type == "object_interaction":
        object_name = normalized.task_config.object_name
        if object_name is None:
            raise ValueError("Normalized object-interaction jobs require an object name")
        object_asset = get_omomo_object_asset(object_name)
        dependencies["object_model_tree"] = object_asset.mesh_path.parent
    elif normalized.task_type == "climbing":
        object_dir = normalized.task_config.object_dir
        if object_dir is None:
            raise ValueError("Normalized climbing jobs require object_dir")
        dependencies["climbing_scene_tree"] = object_dir

    return dependencies


def _solver_identity(
    normalized: RetargetingConfig,
) -> dict[str, object]:
    """Bind a job to solver code, runtime, and static dependency bytes."""

    return {
        **solver_implementation_fingerprint(),
        "dependencies": solver_dependency_fingerprint(_solver_dependency_paths(normalized)),
    }


def _normalized_job_payload(
    normalized: RetargetingConfig,
    *,
    run_kind: RunKind,
    variant: RetargetVariant,
    experiment_name: str | None,
    dataset_partition: str,
    sequence_key: str,
    solver_identity: dict[str, object],
) -> tuple[str, str]:
    solver_config = asdict(normalized)
    solver_config["augmentation"] = False
    solver_config["overwrite_existing"] = False
    solver_config["save_dir"] = None
    payload = {
        "config": solver_config,
        "run_kind": run_kind,
        "variant": asdict(variant),
        "experiment_name": experiment_name,
        "dataset_partition": dataset_partition,
        "sequence_key": sequence_key,
        "solver_identity": solver_identity,
    }
    config_json = json.dumps(
        _json_ready(payload),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return config_json, hashlib.sha256(config_json.encode("utf-8")).hexdigest()


def _artifact_scalar_text(data: np.lib.npyio.NpzFile, key: str) -> str:
    value = np.asarray(data[key])
    if value.ndim != 0 or value.dtype.kind not in {"S", "U"}:
        raise ValueError(f"{key!r} must be a scalar string")
    item = value.item()
    return item.decode("utf-8") if isinstance(item, bytes) else str(item)


def result_artifact_matches_job(
    path: Path,
    job: RetargetJob,
    *,
    identity_baseline: bool = False,
) -> bool:
    """Return whether an artifact exactly matches a planned job.

    Schema validity alone is insufficient for safe resume. This contract also
    binds the result to the resolved source path and bytes, normalized config,
    semantic variant, and run kind. ``identity_baseline`` applies the identity
    metadata stored on an augmented job before its qpos can be reused.
    """

    artifact_path = Path(path)
    if not artifact_path.is_file():
        return False
    expected_variant = IDENTITY_VARIANT.name if identity_baseline else job.variant.name
    expected_run_kind = "single" if identity_baseline else job.run_kind
    expected_source_sha256 = job.baseline_source_sha256 if identity_baseline else job.source_sha256
    expected_config_json = job.baseline_config_json if identity_baseline else job.config_json
    expected_config_sha256 = job.baseline_config_sha256 if identity_baseline else job.config_sha256
    try:
        with np.load(artifact_path, allow_pickle=False) as data:
            validate_result_artifact(data)
            matches = (
                int(np.asarray(data["schema_version"]).item()) == RESULT_SCHEMA_VERSION
                and _artifact_scalar_text(data, "source_path") == str(job.source_path)
                and _artifact_scalar_text(data, "source_sha256") == expected_source_sha256
                and _artifact_scalar_text(data, "config_json") == expected_config_json
                and _artifact_scalar_text(data, "config_sha256") == expected_config_sha256
                and _artifact_scalar_text(data, "variant") == expected_variant
                and _artifact_scalar_text(data, "run_kind") == expected_run_kind
            )
            if not matches:
                return False
            validate_result_external_assets(data)
            return True
    except (
        KeyError,
        OSError,
        ResultArtifactValidationError,
        TypeError,
        UnicodeDecodeError,
        ValueError,
    ):
        return False


def _encode_path_component(value: str, field_name: str) -> str:
    """Encode one source-derived value as a portable result-path component."""

    if not isinstance(value, str) or not value or value in {".", ".."} or "\0" in value:
        raise ValueError(f"{field_name} must be a non-empty path component")
    if "/" in value:
        raise ValueError(f"{field_name} must contain exactly one path component")
    encoded = quote(value, safe="._-")
    if len(encoded.encode("ascii")) > 240:
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
        encoded = f"sha256-{digest}"
    return encoded


def normalize_retargeting_config(cfg: RetargetingConfig) -> RetargetingConfig:
    """Copy and normalize every top-level selection into nested configs."""

    normalized = copy.deepcopy(cfg)
    validate_config(normalized)
    data_format = normalize_data_format(normalized.data_format or DEFAULT_DATA_FORMATS[normalized.task_type])
    normalized.data_format = data_format
    if normalized.robot_config.robot_type != normalized.robot:
        normalized.robot_config = replace(
            normalized.robot_config,
            robot_type=normalized.robot,
        )
    if (
        normalized.motion_data_config.robot_type != normalized.robot
        or normalized.motion_data_config.data_format != data_format
    ):
        normalized.motion_data_config = replace(
            normalized.motion_data_config,
            data_format=data_format,
            robot_type=normalized.robot,
        )

    resolved_object_name = resolve_task_object_name(
        normalized.task_type,
        data_format,
        normalized.task_name,
        normalized.task_config.object_name,
    )
    normalized.task_config = replace(
        normalized.task_config,
        object_name=resolved_object_name,
    )
    if normalized.task_type == "climbing" and normalized.task_config.object_dir is None:
        normalized.task_config = replace(
            normalized.task_config,
            object_dir=normalized.data_path / normalized.task_name,
        )
    return normalized


def _canonical_result_path(
    *,
    results_root: Path,
    robot: str,
    task_type: str,
    data_format: str,
    dataset_partition: str,
    sequence_key: str,
    variant: RetargetVariant,
    run_kind: RunKind,
    experiment_name: str | None,
) -> Path:
    sequence_path = Path(sequence_key)
    if sequence_path.is_absolute():
        raise ValueError(f"Invalid sequence_key: {sequence_key!r}")
    raw_sequence_parts = sequence_path.parts
    if not raw_sequence_parts or any(part in {"", ".", ".."} for part in raw_sequence_parts):
        raise ValueError(f"Invalid sequence_key: {sequence_key!r}")
    sequence_parts = tuple(_encode_path_component(part, "sequence_key") for part in raw_sequence_parts)
    common = (
        Path(robot)
        / task_type
        / data_format
        / _encode_path_component(dataset_partition, "dataset_partition")
        / Path(*sequence_parts)
    )
    if run_kind == "ablation":
        if not experiment_name:
            raise ValueError("Ablation jobs require experiment_name")
        return (
            results_root
            / "ablations"
            / _encode_path_component(experiment_name, "experiment_name")
            / variant.name
            / common
            / "identity.npz"
        )
    return results_root / "canonical" / common / f"{variant.name}.npz"


def _persistent_results_state_root(results_root: Path) -> Path:
    resolved_root = Path(results_root).expanduser().resolve()
    parts = resolved_root.parts
    if ".staging" in parts:
        staging_index = parts.index(".staging")
        return Path(*parts[:staging_index])
    if re.fullmatch(r"v[0-9]+", resolved_root.name):
        return resolved_root.parent
    return resolved_root


def generated_assets_root(results_root: Path) -> Path:
    """Resolve a promotion-safe generated-asset root for a result tree."""

    return _persistent_results_state_root(results_root) / ".generated-assets" / "jobs"


def results_state_lock_path(results_root: Path) -> Path:
    """Return the promotion/job lifecycle lock shared by staging and formal roots."""

    return _persistent_results_state_root(results_root) / ".locks" / "results-state.lock"


def _job_lock_path(
    results_root: Path,
    logical_output_path: Path,
) -> Path:
    logical_name = logical_output_path.as_posix()
    digest = hashlib.sha256(logical_name.encode("utf-8")).hexdigest()
    return _persistent_results_state_root(results_root) / ".locks" / "retargeting" / digest[:2] / f"{digest}.lock"


def _generated_assets_dir_for_job(
    results_root: Path,
    *,
    config_sha256: str,
    source_sha256: str,
) -> Path:
    """Return the deterministic cache directory for one exact job identity."""

    fingerprint = hashlib.sha256(f"{config_sha256}:{source_sha256}".encode("ascii")).hexdigest()
    return generated_assets_root(results_root) / fingerprint[:2] / fingerprint


def _default_sequence_key(
    normalized: RetargetingConfig,
    source_path: Path,
) -> str:
    """Derive the canonical identity used by recursive batch discovery."""

    spec = get_motion_format_spec(str(normalized.data_format))
    if not spec.recursive_files:
        return normalized.task_name

    data_root = Path(normalized.data_path).expanduser().resolve()
    try:
        relative_source = source_path.relative_to(data_root)
    except ValueError as exc:
        raise ValueError(
            "A recursive source outside data_path requires an explicit "
            f"sequence_key: source={source_path}, data_path={data_root}",
        ) from exc

    relative_identity = (
        relative_source.parent if normalized.task_type == "climbing" else relative_source.with_suffix("")
    )
    sequence_key = relative_identity.as_posix()
    if sequence_key in {"", "."}:
        raise ValueError(
            "A recursive source must have a non-empty identity relative to "
            f"data_path: source={source_path}, data_path={data_root}",
        )
    return sequence_key


def build_retarget_job(
    cfg: RetargetingConfig,
    *,
    variant: RetargetVariant = IDENTITY_VARIANT,
    run_kind: RunKind = "single",
    experiment_name: str | None = None,
    results_root: Path | None = None,
    dataset_partition: str | None = None,
    sequence_key: str | None = None,
    source_path: Path | None = None,
    source_sha256: str | None = None,
    overwrite_existing: bool | None = None,
) -> RetargetJob:
    """Normalize a config and resolve one stable source/output contract."""

    _validate_job_semantics(
        variant=variant,
        run_kind=run_kind,
        experiment_name=experiment_name,
    )
    normalized = normalize_retargeting_config(cfg)
    # Augmentation is an orchestration request. The variant is the complete,
    # stable solver input, so this CLI-only flag must not split otherwise
    # identical identity artifacts into separate resume identities.
    normalized.augmentation = False
    if overwrite_existing is not None:
        normalized.overwrite_existing = bool(overwrite_existing)
    canonical_format = str(normalized.data_format)
    resolved_source_path = (
        Path(source_path).expanduser().resolve()
        if source_path is not None
        else resolve_motion_path(
            normalized.data_path,
            normalized.task_name,
            canonical_format,
        ).resolve()
    )
    if not resolved_source_path.is_file():
        raise FileNotFoundError(f"Retargeting source file not found: {resolved_source_path}")

    partition = dataset_partition or normalized.data_path.name
    key = sequence_key if sequence_key is not None else _default_sequence_key(normalized, resolved_source_path)
    root = (
        Path(results_root).expanduser().resolve()
        if results_root is not None
        else (
            normalized.save_dir.expanduser().resolve()
            if normalized.save_dir is not None
            else Path(__file__).resolve().parent / "demo_results" / "v1"
        )
    )
    output_path = _canonical_result_path(
        results_root=root,
        robot=normalized.robot,
        task_type=normalized.task_type,
        data_format=canonical_format,
        dataset_partition=partition,
        sequence_key=key,
        variant=variant,
        run_kind=run_kind,
        experiment_name=experiment_name,
    )
    baseline_path = _canonical_result_path(
        results_root=root,
        robot=normalized.robot,
        task_type=normalized.task_type,
        data_format=canonical_format,
        dataset_partition=partition,
        sequence_key=key,
        variant=IDENTITY_VARIANT,
        run_kind="single",
        experiment_name=None,
    )
    output_lock_path = _job_lock_path(
        root,
        output_path.relative_to(root),
    )
    baseline_lock_path = _job_lock_path(
        root,
        baseline_path.relative_to(root),
    )
    state_lock_path = results_state_lock_path(root)
    source_stat = resolved_source_path.stat()
    resolved_source_sha256 = source_sha256 or _sha256_file(resolved_source_path)
    solver_identity = _solver_identity(normalized)
    config_json, config_sha256 = _normalized_job_payload(
        normalized,
        run_kind=run_kind,
        variant=variant,
        experiment_name=experiment_name,
        dataset_partition=partition,
        sequence_key=key,
        solver_identity=solver_identity,
    )
    baseline_config_json, baseline_config_sha256 = _normalized_job_payload(
        normalized,
        run_kind="single",
        variant=IDENTITY_VARIANT,
        experiment_name=None,
        dataset_partition=partition,
        sequence_key=key,
        solver_identity=solver_identity,
    )
    generated_assets_dir = _generated_assets_dir_for_job(
        root,
        config_sha256=config_sha256,
        source_sha256=resolved_source_sha256,
    )
    return RetargetJob(
        config=normalized,
        source_path=resolved_source_path,
        output_path=output_path,
        baseline_path=baseline_path,
        state_lock_path=state_lock_path,
        output_lock_path=output_lock_path,
        baseline_lock_path=baseline_lock_path,
        generated_assets_dir=generated_assets_dir,
        sequence_key=key,
        dataset_partition=partition,
        run_kind=run_kind,
        variant=variant,
        experiment_name=experiment_name,
        source_sha256=resolved_source_sha256,
        source_size=source_stat.st_size,
        source_mtime_ns=source_stat.st_mtime_ns,
        config_json=config_json,
        config_sha256=config_sha256,
        baseline_source_sha256=resolved_source_sha256,
        baseline_config_json=baseline_config_json,
        baseline_config_sha256=baseline_config_sha256,
    )


# ----------------------------- Helper Functions -----------------------------


def _validate_job_semantics(
    *,
    variant: RetargetVariant,
    run_kind: RunKind,
    experiment_name: str | None,
) -> None:
    """Reject ambiguous combinations before they can share an output path."""

    if run_kind not in {"single", "augmentation", "ablation"}:
        raise ValueError(
            "run_kind must be 'single', 'augmentation', or 'ablation'",
        )
    if run_kind == "single":
        if not variant.is_identity:
            raise ValueError("single jobs require the exact identity variant")
        if experiment_name is not None:
            raise ValueError("single jobs must not set experiment_name")
        return
    if run_kind == "augmentation":
        if not variant.changes_motion:
            raise ValueError(
                "augmentation jobs require a non-identity motion transformation",
            )
        if experiment_name is not None:
            raise ValueError("augmentation jobs must not set experiment_name")
        return
    if variant.changes_motion:
        raise ValueError("ablation jobs must not transform motion")
    if not experiment_name:
        raise ValueError("ablation jobs require experiment_name")


def resolve_task_object_name(
    task_type: str,
    data_format: str,
    task_name: str,
    explicit_object_name: str | None,
) -> str:
    """Resolve the object category and reject sequence/config mismatches."""

    if task_type == "robot_only":
        if explicit_object_name not in {None, "ground"}:
            raise ValueError(
                "robot_only tasks require object_name='ground'",
            )
        return "ground"
    if task_type == "climbing":
        if explicit_object_name not in {None, "multi_boxes"}:
            raise ValueError(
                "climbing tasks require object_name='multi_boxes'",
            )
        return "multi_boxes"
    if task_type != "object_interaction":
        raise ValueError(f"Unknown task type: {task_type}")
    if data_format != "omomo":
        if explicit_object_name is None:
            raise ValueError("Object-interaction tasks require an explicit object name for non-OMOMO data")
        return explicit_object_name

    sequence_object = parse_omomo_sequence_name(task_name).object_name
    if explicit_object_name is not None and explicit_object_name != sequence_object:
        raise ValueError(
            f"OMOMO task {task_name!r} contains object {sequence_object!r}, "
            f"but task_config.object_name is {explicit_object_name!r}"
        )
    return sequence_object


def create_task_constants(
    robot_config: RobotConfig,
    motion_data_config: MotionDataConfig,
    task_config: TaskConfig,
    task_type: str,
) -> SimpleNamespace:
    """Create combined task constants from robot and motion data configs.

    Args:
        robot_config: Robot configuration
        motion_data_config: Motion data format configuration
        task_config: Task-specific configuration
        task_type: Type of task ("robot_only", "object_interaction", "climbing")

    Returns:
        SimpleNamespace with all task constants
    """
    task_constants = SimpleNamespace()

    # Copy all attributes from robot_config
    for attr in dir(robot_config):
        if attr.isupper() and not attr.startswith("_"):
            setattr(task_constants, attr, getattr(robot_config, attr))
    robot_urdf_path = Path(task_constants.ROBOT_URDF_FILE).expanduser()
    if not robot_urdf_path.is_absolute():
        robot_urdf_path = Path(__file__).resolve().parent / robot_urdf_path
    task_constants.ROBOT_URDF_FILE = str(robot_urdf_path.resolve())

    # Copy legacy motion data constants (upper-case for compatibility)
    for attr, value in motion_data_config.legacy_constants().items():
        setattr(task_constants, attr, value)
    task_constants.ROBOT_TYPE = robot_config.robot_type
    task_constants.TASK_TYPE = task_type
    task_constants.SOURCE_DATA_FORMAT = motion_data_config.data_format
    task_constants.SOURCE_FPS = 30.0
    task_constants.SOURCE_ROOT_QUATERNIONS = None
    format_spec = get_motion_format_spec(motion_data_config.data_format)
    task_constants.SOURCE_ROOT_JOINT = format_spec.root_joint
    task_constants.SOURCE_ORIENTATION_MODE = format_spec.orientation_mode

    # Task-specific object setup
    if task_type == "robot_only":
        obj_name = task_config.object_name or "ground"
        task_constants.OBJECT_NAME = obj_name
        task_constants.OBJECT_URDF_FILE = None
        task_constants.OBJECT_MESH_FILE = None
    elif task_type == "object_interaction":
        if task_config.object_name is None:
            raise ValueError("object_interaction constants require a resolved object name")
        obj_name = task_config.object_name
        object_asset = get_omomo_object_asset(obj_name)
        task_constants.OBJECT_NAME = obj_name
        task_constants.OBJECT_URDF_FILE = str(object_asset.urdf_path)
        task_constants.OBJECT_MESH_FILE = str(object_asset.mesh_path)
        task_constants.OBJECT_URDF_TEMPLATE = str(
            object_asset.mesh_path.parent.parent / "templates" / "omomo_object.urdf.jinja"
        )
        task_constants.SCENE_XML_FILE = ""
    elif task_type == "climbing":
        obj_name = task_config.object_name or "multi_boxes"
        task_constants.OBJECT_NAME = obj_name
        object_dir = task_config.object_dir
        task_constants.OBJECT_DIR = str(object_dir) if object_dir else ""
        task_constants.OBJECT_URDF_FILE = str(object_dir / f"{obj_name}.urdf") if object_dir else f"{obj_name}.urdf"
        task_constants.OBJECT_MESH_FILE = str(object_dir / f"{obj_name}.obj") if object_dir else f"{obj_name}.obj"
        task_constants.SCENE_XML_FILE = ""  # Will be set later

    return task_constants


def validate_config(cfg: RetargetingConfig) -> None:
    """Validate configuration consistency.

    Args:
        cfg: Configuration arguments

    Raises:
        ValueError: If configuration is invalid
    """
    if cfg.dataset is not None:
        validate_production_task(
            task=cfg.task_type,
            robot=cfg.robot,
            dataset=cfg.dataset,
        )
    data_format = cfg.data_format or DEFAULT_DATA_FORMATS[cfg.task_type]
    validate_motion_task(data_format, cfg.task_type)
    if not cfg.retargeter.save_interaction_mesh:
        raise ValueError(
            "Canonical retargeting artifacts require Interaction Mesh data; save_interaction_mesh must remain enabled"
        )


def create_ground_points(x_range: tuple[float, float], y_range: tuple[float, float], size: int) -> np.ndarray:
    """Create ground point meshgrid.

    Args:
        x_range: (min, max) x-coordinate range
        y_range: (min, max) y-coordinate range
        size: Number of points per dimension

    Returns:
        (N, 3) array of ground points
    """
    x = np.linspace(x_range[0], x_range[1], size)
    y = np.linspace(y_range[0], y_range[1], size)
    X, Y = np.meshgrid(x, y)
    return np.stack([X.flatten(), Y.flatten(), np.zeros_like(X.flatten())], axis=1)


def _prepare_generated_assets_dir(
    generated_assets_dir: Path,
    *,
    protected_source_dir: Path,
) -> Path:
    """Create a generated directory while refusing to place it in source data."""

    destination = Path(generated_assets_dir).expanduser().resolve()
    protected = Path(protected_source_dir).expanduser().resolve()
    if destination == protected or destination.is_relative_to(protected):
        raise ValueError(f"generated_assets_dir must be outside source data: {destination} is inside {protected}")
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def _resolve_climbing_scene_xml(
    object_dir: Path,
    constants: SimpleNamespace,
) -> Path:
    """Resolve the one unscaled robot-terrain scene shipped with a sequence."""

    object_name = str(constants.OBJECT_NAME)
    preferred_name = Path(constants.ROBOT_URDF_FILE).name.replace(
        ".urdf",
        f"_w_{object_name}.xml",
    )
    preferred = object_dir / preferred_name
    if preferred.is_file():
        return preferred
    robot_type = str(constants.ROBOT_TYPE)
    candidates = tuple(sorted(object_dir.glob(f"{robot_type}*_w_{object_name}.xml")))
    if len(candidates) != 1:
        raise FileNotFoundError(
            "Expected exactly one unscaled robot-terrain scene for "
            f"{robot_type}/{object_name} in {object_dir}, found "
            f"{[path.name for path in candidates]}"
        )
    return candidates[0]


def setup_object_data(
    task_type: TaskType,
    constants: SimpleNamespace,
    object_dir: Path | None,
    smpl_scale: float,
    task_config: TaskConfig,
    augmentation: bool,
    object_scale_augmented: np.ndarray | None = None,
    *,
    generated_assets_dir: Path,
) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    """Setup object-specific data (ground, object mesh, climbing terrain).
    Args:
        task_type: Type of task
        constants: Task constants
        object_dir: Object directory path (for climbing)
        smpl_scale: SMPL scaling factor
        task_config: Task configuration
        augmentation: Whether augmentation is enabled
        object_scale_augmented: Scale factor for augmented objects (default: [1.0, 1.0, 1.2])
        generated_assets_dir: Job-scoped generated-asset cache outside source data
    Returns:
        Tuple of (object_local_pts, object_local_pts_demo, object_urdf_path)
    """
    object_scale_normal = np.array([1.0, 1.0, 1.0])
    if object_scale_augmented is None:
        object_scale_augmented = np.array([1.0, 1.0, 1.2])  # For climbing task augmentation
    logger.info("Setting up object data for task: %s", task_type)

    if task_type == "robot_only":
        # Create ground points meshgrid
        ground_pts = create_ground_points(task_config.ground_range, task_config.ground_range, task_config.ground_size)
        return ground_pts, ground_pts, None

    if task_type == "object_interaction":
        # Load object data
        if constants.OBJECT_MESH_FILE is None:
            raise ValueError("OBJECT_MESH_FILE not set for object_interaction task")

        object_local_pts, object_local_pts_demo = load_object_data(
            constants.OBJECT_MESH_FILE, smpl_scale=smpl_scale, sample_count=100
        )
        object_scale = np.asarray(task_config.object_scale, dtype=float)
        if object_scale.shape != (3,) or np.any(object_scale <= 0):
            raise ValueError(f"object_scale must contain three positive values, got {task_config.object_scale}")

        object_local_pts = object_local_pts * object_scale
        scale_factors = tuple(float(value) for value in object_scale)
        models_root = Path(constants.OBJECT_MESH_FILE).parent.parent
        assets_dir = _prepare_generated_assets_dir(
            generated_assets_dir,
            protected_source_dir=models_root,
        )
        object_urdf_file = create_scaled_omomo_object_urdf(
            constants.OBJECT_NAME,
            scale_factors,
            models_root=models_root,
            output_dir=assets_dir,
        )
        robot_xml_file = Path(constants.ROBOT_URDF_FILE).with_suffix(".xml")
        if not robot_xml_file.is_absolute():
            robot_xml_file = Path(__file__).resolve().parent / robot_xml_file
        constants.SCENE_XML_FILE = str(
            create_omomo_object_scene(
                robot_xml_file,
                constants.OBJECT_NAME,
                scale=scale_factors,
                models_root=models_root,
                output_dir=assets_dir,
            )
        )

        return object_local_pts, object_local_pts_demo, str(object_urdf_file)

    if task_type == "climbing":
        if object_dir is None:
            raise ValueError("object_dir must be provided for climbing task")

        object_dir = Path(object_dir).expanduser().resolve()
        assets_dir = _prepare_generated_assets_dir(
            generated_assets_dir,
            protected_source_dir=object_dir,
        )
        # Setup climbing-specific object
        constants.OBJECT_URDF_FILE = str(object_dir / f"{constants.OBJECT_NAME}.urdf")
        constants.OBJECT_MESH_FILE = str(object_dir / f"{constants.OBJECT_NAME}.obj")
        box_asset_xml = object_dir / "box_assets.xml"
        scene_xml_file = _resolve_climbing_scene_xml(object_dir, constants)
        # Set SCENE_XML_FILE in constants BEFORE creating retargeter (needed for temp_retargeter)
        constants.SCENE_XML_FILE = str(scene_xml_file)

        np.random.seed(0)
        print("object mesh file: ", constants.OBJECT_MESH_FILE)
        object_local_pts, object_local_pts_demo_original = load_object_data(
            constants.OBJECT_MESH_FILE,
            smpl_scale=smpl_scale,
            surface_weights=lambda p: (
                task_config.surface_weight_high
                if p[2] > task_config.surface_weight_threshold
                else task_config.surface_weight_low
            ),
            sample_count=100,
        )

        if augmentation:
            ground_pts = create_ground_points(
                task_config.climbing_ground_range, task_config.climbing_ground_range, task_config.climbing_ground_size
            )
            object_local_pts_demo = np.concatenate([object_local_pts_demo_original, ground_pts], axis=0)
            object_scale = object_scale_augmented
            object_local_pts = object_scale * object_local_pts_demo
        else:
            object_scale = object_scale_normal
            object_local_pts_demo = object_local_pts_demo_original
            object_local_pts = object_local_pts_demo

        # Create scaled URDF and XML files
        scale_factors = tuple(float(value) for value in (object_scale * smpl_scale))
        object_urdf_file = create_scaled_multi_boxes_urdf(
            constants.OBJECT_URDF_FILE,
            scale_factors,
            output_dir=assets_dir,
        )
        object_asset_xml_path = create_scaled_multi_boxes_xml(
            str(box_asset_xml),
            scale_factors,
            output_dir=assets_dir,
        )
        new_scene_xml_path = create_new_scene_xml_file(
            str(scene_xml_file),
            scale_factors,
            object_asset_xml_path,
            output_dir=assets_dir,
        )
        constants.SCENE_XML_FILE = new_scene_xml_path

        return object_local_pts, object_local_pts_demo, object_urdf_file

    raise ValueError(f"Unknown task type: {task_type}")


def _compute_q_init_base(
    task_type: TaskType,
    human_joints: np.ndarray,
    object_poses: np.ndarray,
    constants: SimpleNamespace,
    retargeter: InteractionMeshRetargeter | None = None,
) -> np.ndarray:
    """Compute base robot pose initialization (q_init_base).
    This is a shared helper function used by both single and parallel processing.
    Args:
        task_type: Type of task
        human_joints: Human joint positions
        object_poses: Object poses in format [qw, qx, qy, qz, x, y, z]
        constants: Task constants
        retargeter: Optional retargeter instance (needed for climbing)
    Returns:
        q_init_base in MuJoCo order: [0:3] position, [3:7] quaternion, [7:] joints
    """
    if task_type == "robot_only":
        base_joint_idx = constants.DEMO_JOINTS.index(constants.SOURCE_ROOT_JOINT)
        if constants.SOURCE_ROOT_QUATERNIONS is not None:
            human_quat_init = constants.SOURCE_ROOT_QUATERNIONS[0]
        elif constants.SOURCE_ORIENTATION_MODE == "bvh":
            human_quat_init = estimate_human_orientation(human_joints, constants.DEMO_JOINTS)
        elif constants.SOURCE_ORIENTATION_MODE == "smpl":
            human_quat_init = estimate_smpl_orientation(human_joints, constants.DEMO_JOINTS)
        elif constants.SOURCE_ORIENTATION_MODE == "mocap":
            human_quat_init = estimate_mocap_foot_orientation(human_joints, constants.DEMO_JOINTS)
        else:
            raise ValueError(f"Unknown source orientation mode: {constants.SOURCE_ORIENTATION_MODE}")
        q_init_base = np.concatenate(
            [human_joints[0, base_joint_idx, :3], human_quat_init, np.zeros(constants.ROBOT_DOF)]
        )
    elif task_type == "object_interaction":
        _, human_quat_init = transform_from_human_to_world(
            human_joints[0, 0, :], object_poses[0], np.array([0.0, 0.0, 0.0])
        )
        # MuJoCo order: pos first, then quat
        q_init_base = np.concatenate([human_joints[0, 0, :3], human_quat_init, np.zeros(constants.ROBOT_DOF)])
    elif task_type == "climbing":
        if retargeter is None:
            raise ValueError("retargeter is required for climbing task")
        human_quat_init = estimate_mocap_foot_orientation(human_joints, retargeter.demo_joints)
        root_joint_idx = retargeter.demo_joints.index(constants.SOURCE_ROOT_JOINT)
        q_init_base = np.concatenate(
            [
                human_joints[0, root_joint_idx],
                human_quat_init,
                np.zeros(constants.ROBOT_DOF),
            ]
        )
    else:
        raise ValueError(f"Invalid task type: {task_type}")

    return q_init_base


def convert_object_poses_to_mujoco_order(object_poses: np.ndarray) -> np.ndarray:
    """Convert object poses from [qw, qx, qy, qz, x, y, z] to MuJoCo order [x, y, z, qw, qx, qy, qz].
    Args:
        object_poses: Object poses array of shape (T, 7) in format [qw, qx, qy, qz, x, y, z]
    Returns:
        Object poses array in MuJoCo order [x, y, z, qw, qx, qy, qz]
    """
    return object_poses[:, [4, 5, 6, 0, 1, 2, 3]]


def _normalize_quaternions_wxyz(quaternions: np.ndarray, *, label: str) -> np.ndarray:
    """Return finite, unit-length WXYZ quaternions without mutating the input."""

    quaternions = np.asarray(quaternions, dtype=np.float64)
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise ValueError(f"{label} must have shape (T, 4), got {quaternions.shape}")
    if not np.isfinite(quaternions).all():
        raise ValueError(f"{label} must contain only finite values")
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if np.any(norms <= np.finfo(np.float64).eps):
        raise ValueError(f"{label} must not contain zero-length quaternions")
    return quaternions / norms


def _multiply_quaternions_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Multiply equally shaped batches of WXYZ quaternions."""

    left_w = left[:, :1]
    right_w = right[:, :1]
    left_xyz = left[:, 1:]
    right_xyz = right[:, 1:]
    return np.concatenate(
        (
            left_w * right_w - np.sum(left_xyz * right_xyz, axis=1, keepdims=True),
            left_w * right_xyz + right_w * left_xyz + np.cross(left_xyz, right_xyz),
        ),
        axis=1,
    )


def _rotate_vectors_by_quaternions_wxyz(vectors: np.ndarray, quaternions: np.ndarray) -> np.ndarray:
    """Rotate one vector per unit WXYZ quaternion."""

    quaternion_xyz = quaternions[:, 1:]
    twice_cross = 2.0 * np.cross(quaternion_xyz, vectors)
    return vectors + quaternions[:, :1] * twice_cross + np.cross(quaternion_xyz, twice_cross)


def _object_pose_rotation_deltas_wxyz(
    demo_object_poses_xyz_wxyz: np.ndarray,
    target_object_poses_xyz_wxyz: np.ndarray,
) -> np.ndarray:
    """Return per-frame demo-to-target world rotations from saved pose layouts."""

    demo = np.asarray(demo_object_poses_xyz_wxyz, dtype=np.float64)
    target = np.asarray(target_object_poses_xyz_wxyz, dtype=np.float64)
    if demo.ndim != 2 or demo.shape[1] != 7:
        raise ValueError(f"demo_object_poses_xyz_wxyz must have shape (T, 7), got {demo.shape}")
    if target.ndim != 2 or target.shape[1] != 7:
        raise ValueError(f"target_object_poses_xyz_wxyz must have shape (T, 7), got {target.shape}")
    if len(demo) != len(target):
        raise ValueError(
            "Demo and target object pose trajectories must have the same frame count, "
            f"got {len(demo)} and {len(target)}"
        )
    if len(demo) == 0:
        raise ValueError("Object pose trajectories must contain at least one frame")

    demo_quaternions = _normalize_quaternions_wxyz(
        demo[:, 3:7],
        label="demo object quaternions",
    )
    target_quaternions = _normalize_quaternions_wxyz(
        target[:, 3:7],
        label="target object quaternions",
    )
    demo_inverse = demo_quaternions.copy()
    demo_inverse[:, 1:] *= -1.0
    return _normalize_quaternions_wxyz(
        _multiply_quaternions_wxyz(target_quaternions, demo_inverse),
        label="object pose rotation deltas",
    )


def _transform_nominal_robot_root_with_object_delta(
    q_nominal: np.ndarray,
    demo_object_poses: np.ndarray,
    target_object_poses: np.ndarray,
) -> np.ndarray:
    """Apply each object's demo-to-target rigid transform to the nominal robot root.

    Object poses use source order ``[qw, qx, qy, qz, x, y, z]`` while the
    baseline robot free root uses MuJoCo order ``[x, y, z, qw, qx, qy, qz]``.
    Only that robot free root is transformed; actuated joints and any appended
    object free joint remain unchanged for the dynamic-object writer.
    """

    nominal = np.asarray(q_nominal, dtype=np.float64)
    demo = np.asarray(demo_object_poses, dtype=np.float64)
    target = np.asarray(target_object_poses, dtype=np.float64)
    if nominal.ndim != 2 or nominal.shape[1] < 7:
        raise ValueError(f"q_nominal must have shape (T, >=7), got {nominal.shape}")
    if demo.ndim != 2 or demo.shape[1] != 7:
        raise ValueError(f"demo_object_poses must have shape (T, 7), got {demo.shape}")
    if target.ndim != 2 or target.shape[1] != 7:
        raise ValueError(f"target_object_poses must have shape (T, 7), got {target.shape}")
    if not (len(nominal) == len(demo) == len(target)):
        raise ValueError(
            "q_nominal, demo_object_poses, and target_object_poses must have the same frame count, "
            f"got {len(nominal)}, {len(demo)}, and {len(target)}"
        )
    if len(nominal) == 0:
        raise ValueError("Nominal and object pose trajectories must contain at least one frame")
    if not np.isfinite(nominal).all() or not np.isfinite(demo[:, 4:]).all() or not np.isfinite(target[:, 4:]).all():
        raise ValueError("Nominal qpos and object positions must contain only finite values")

    demo_quaternions = _normalize_quaternions_wxyz(demo[:, :4], label="demo object quaternions")
    target_quaternions = _normalize_quaternions_wxyz(target[:, :4], label="target object quaternions")
    nominal_root_quaternions = _normalize_quaternions_wxyz(
        nominal[:, 3:7],
        label="nominal robot root quaternions",
    )
    demo_inverse = demo_quaternions.copy()
    demo_inverse[:, 1:] *= -1.0
    delta_quaternions = _normalize_quaternions_wxyz(
        _multiply_quaternions_wxyz(target_quaternions, demo_inverse),
        label="object pose delta quaternions",
    )

    transformed = nominal.copy()
    transformed[:, :3] = target[:, 4:] + _rotate_vectors_by_quaternions_wxyz(
        nominal[:, :3] - demo[:, 4:],
        delta_quaternions,
    )
    transformed[:, 3:7] = _normalize_quaternions_wxyz(
        _multiply_quaternions_wxyz(delta_quaternions, nominal_root_quaternions),
        label="transformed nominal robot root quaternions",
    )
    return transformed


def build_retargeter_kwargs_from_config(
    retargeter_config: RetargeterConfig,
    constants: SimpleNamespace,
    object_urdf_path: str | None,
    task_type: str,
) -> dict:
    """Build kwargs for InteractionMeshRetargeter from a RetargeterConfig.
    This is a convenience function that allows building kwargs directly from
    a RetargeterConfig without needing a full RetargetingConfig.
    Args:
        retargeter_config: Retargeter configuration
        constants: Task constants
        object_urdf_path: Path to object URDF file
        task_type: Type of task
    Returns:
        Dictionary of kwargs for InteractionMeshRetargeter
    """
    return {
        "task_constants": constants,
        "object_urdf_path": object_urdf_path,
        "q_a_init_idx": retargeter_config.q_a_init_idx,
        "activate_joint_limits": retargeter_config.activate_joint_limits,
        "activate_obj_non_penetration": retargeter_config.activate_obj_non_penetration,
        "activate_foot_sticking": retargeter_config.activate_foot_sticking,
        "foot_lock": retargeter_config.foot_lock,
        "penetration_tolerance": retargeter_config.penetration_tolerance,
        "foot_sticking_tolerance": retargeter_config.foot_sticking_tolerance,
        "foot_sticking_fallback_tolerance": retargeter_config.foot_sticking_fallback_tolerance,
        "release_foot_sticking_on_infeasible": retargeter_config.release_foot_sticking_on_infeasible,
        "release_object_non_penetration_on_infeasible": (
            retargeter_config.release_object_non_penetration_on_infeasible
        ),
        "retry_without_foot_sticking_on_infeasible": (retargeter_config.retry_without_foot_sticking_on_infeasible),
        "retry_frame_zero_ground_on_infeasible": (
            retargeter_config.retry_frame_zero_ground_on_infeasible and task_type == "robot_only"
        ),
        "self_collision": retargeter_config.self_collision,
        "step_size": retargeter_config.step_size,
        "sqp_max_iterations": retargeter_config.sqp_max_iterations,
        "sqp_min_iterations": retargeter_config.sqp_min_iterations,
        "sqp_convergence_patience": retargeter_config.sqp_convergence_patience,
        "visualize": retargeter_config.visualize,
        "mesh_opacity": retargeter_config.mesh_opacity,
        "debug": retargeter_config.debug,
        "show_interaction_mesh": retargeter_config.show_interaction_mesh,
        "save_interaction_mesh": retargeter_config.save_interaction_mesh,
        "interaction_mesh_mode": retargeter_config.interaction_mesh_mode,
        "interaction_mesh_edges": retargeter_config.interaction_mesh_edges,
        "interaction_mesh_line_width": retargeter_config.interaction_mesh_line_width,
        "w_nominal_tracking_init": retargeter_config.w_nominal_tracking_init,
        "nominal_tracking_tau": retargeter_config.nominal_tracking_tau,
        "orientation_joints_mapping": constants.ORIENTATION_JOINTS_MAPPING,
        "orientation_weights": retargeter_config.orientation_weights,
        "orientation_alignment_mode": retargeter_config.orientation_alignment_mode,
        "orientation_t_pose_human_quaternions_wxyz": (constants.ORIENTATION_T_POSE_HUMAN_QUATERNIONS_WXYZ),
        "orientation_t_pose_robot_base_quaternion_wxyz": (constants.ORIENTATION_T_POSE_ROBOT_BASE_QUATERNION_WXYZ),
        "orientation_t_pose_robot_joint_positions": (constants.ORIENTATION_T_POSE_ROBOT_JOINT_POSITIONS),
        "orientation_alignment_quaternions_wxyz": (retargeter_config.orientation_alignment_quaternions_wxyz),
    }


def initialize_robot_pose(
    task_type: TaskType,
    human_joints: np.ndarray,
    object_poses: np.ndarray,
    constants: SimpleNamespace,
    retargeter: InteractionMeshRetargeter,
    task_config: TaskConfig,
    augmentation: bool,
    save_dir: Path,
    task_name: str,
    augmentation_translation: np.ndarray | None = None,
    augmentation_rotation: float | None = 0.0,
    baseline_path: Path | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray, np.ndarray, np.ndarray]:
    """Initialize robot pose (q_init, q_nominal) based on task.
    Returns qpos in MuJoCo order: [0:3] position, [3:7] quaternion, [7:] joints.
    Object poses are returned in MuJoCo order: [0:3] position, [3:7] quaternion.
    Args:
        task_type: Type of task
        human_joints: Human joint positions
        object_poses: Object poses (assumed to be in format: [quat, pos] or [pos, quat])
        constants: Task constants
        retargeter: Retargeter instance
        task_config: Task configuration
        augmentation: Whether augmentation is enabled
        save_dir: Save directory path
        task_name: Task name
        augmentation_translation: Translation vector for augmentation (default: [0.2, 0.0, 0.0])
    Returns:
        Tuple of (q_init, q_nominal, object_poses_augmented, human_joints_modified, object_poses_modified)
        where qpos is in MuJoCo order and object_poses are in MuJoCo order
    """
    # Use default if not provided
    if augmentation_translation is None:
        augmentation_translation = _AUGMENTATION_TRANSLATION
    logger.info("Initializing robot pose")

    if task_type == "robot_only":
        q_init = _compute_q_init_base(task_type, human_joints, object_poses, constants)
        object_poses = convert_object_poses_to_mujoco_order(object_poses)
        return q_init, None, object_poses, human_joints, object_poses

    if task_type == "object_interaction":
        if augmentation:
            object_moving_frame_idx = extract_object_first_moving_frame(object_poses)
            object_poses_augmented = augment_object_poses(
                object_poses,
                object_moving_frame_idx,
                human_joints[0, 0, :],
                augmentation_translation,
                augmentation_rotation,
            )

            original_path = baseline_path if baseline_path is not None else save_dir / f"{task_name}_original.npz"
            if not original_path.exists():
                raise FileNotFoundError(f"Original file not found: {original_path}. Run without --augmentation first.")

            with np.load(original_path, allow_pickle=False) as data:
                q_nominal = np.asarray(data["qpos"], dtype=np.float64)
            q_nominal = _transform_nominal_robot_root_with_object_delta(
                q_nominal,
                object_poses,
                object_poses_augmented,
            )
            # Convert object poses only after applying the source-order rigid transform.
            object_poses_augmented = convert_object_poses_to_mujoco_order(object_poses_augmented)
            object_poses = convert_object_poses_to_mujoco_order(object_poses)
            return q_nominal[0], q_nominal, object_poses_augmented, human_joints, object_poses
        object_poses_augmented = object_poses.copy()
        q_init = _compute_q_init_base(task_type, human_joints, object_poses, constants)
        # Convert object_poses to MuJoCo order
        object_poses = convert_object_poses_to_mujoco_order(object_poses)
        object_poses_augmented = convert_object_poses_to_mujoco_order(object_poses_augmented)
        return q_init, None, object_poses_augmented, human_joints, object_poses

    if task_type == "climbing":
        if augmentation:
            original_path = baseline_path if baseline_path is not None else save_dir / f"{task_name}_original.npz"
            if not original_path.exists():
                raise FileNotFoundError(f"Original file not found: {original_path}. Run without --augmentation first.")

            with np.load(original_path, allow_pickle=False) as data:
                q_nominal = np.asarray(data["qpos"], dtype=np.float64)
            # Convert object_poses to MuJoCo order
            object_poses = convert_object_poses_to_mujoco_order(object_poses)
            return q_nominal[0], q_nominal, object_poses, human_joints, object_poses
        q_init = _compute_q_init_base(task_type, human_joints, object_poses, constants, retargeter)
        # Convert object_poses to MuJoCo order
        object_poses = convert_object_poses_to_mujoco_order(object_poses)
        return q_init, None, object_poses, human_joints, object_poses

    raise ValueError(f"Unknown task type: {task_type}")


def _direct_motion_orientations(
    motion,
) -> tuple[np.ndarray | None, tuple[str, ...], str]:
    """Read the adapter's explicit direct-source orientation subset."""

    direct_quaternions = motion.orientation_quaternions_wxyz
    direct_names = motion.orientation_joint_names
    if direct_quaternions is None:
        if direct_names is not None or motion.orientation_source is not None:
            raise ValueError("Motion adapter returned incomplete source orientation metadata")
        return None, (), "absent"
    if direct_names is None:
        raise ValueError("Motion adapter returned source orientations without joint names")
    return (
        np.ascontiguousarray(direct_quaternions, dtype=np.float32),
        tuple(direct_names),
        motion.orientation_source or "explicit_source_orientation_fields",
    )


def _source_stat_signature(
    path: Path,
) -> tuple[int, int, int, int, int]:
    source_stat = path.stat()
    return (
        source_stat.st_dev,
        source_stat.st_ino,
        source_stat.st_size,
        source_stat.st_mtime_ns,
        source_stat.st_ctime_ns,
    )


def _verify_job_source(
    job: RetargetJob,
) -> tuple[int, int, int, int, int]:
    before = _source_stat_signature(job.source_path)
    observed_sha256 = _sha256_file(job.source_path)
    after = _source_stat_signature(job.source_path)
    if before != after:
        raise RuntimeError(f"Retargeting source changed while hashing: {job.source_path}")
    if observed_sha256 != job.source_sha256:
        raise RuntimeError(
            "Retargeting source bytes changed after job planning: "
            f"{job.source_path}; expected {job.source_sha256}, "
            f"observed {observed_sha256}"
        )
    return after


def _verify_job_solver_identity(job: RetargetJob) -> None:
    """Reject config, code, runtime, or static assets changed after planning."""

    try:
        planned_payload = json.loads(job.config_json)
        planned_solver_identity = planned_payload["solver_identity"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Retargeting job contains an invalid planned solver identity") from exc
    observed_solver_identity = _solver_identity(job.config)
    if observed_solver_identity != planned_solver_identity:
        raise RuntimeError(
            "Retargeting solver implementation, runtime, or static dependency bytes changed after job planning",
        )
    observed_config_json, observed_config_sha256 = _normalized_job_payload(
        job.config,
        run_kind=job.run_kind,
        variant=job.variant,
        experiment_name=job.experiment_name,
        dataset_partition=job.dataset_partition,
        sequence_key=job.sequence_key,
        solver_identity=observed_solver_identity,
    )
    if observed_config_json != job.config_json or observed_config_sha256 != job.config_sha256:
        raise RuntimeError(
            "Retargeting normalized configuration changed after job planning",
        )


def _revalidate_mutable_job_inputs(
    job: RetargetJob,
    *,
    source_signature: tuple[int, int, int, int, int],
) -> None:
    """Recheck every path-based job input at one commit boundary.

    The full source hash and the complete planned solver identity are both
    recomputed. Calling this immediately after a potentially long validation
    closes that validation's mutation window. It deliberately does not claim
    filesystem snapshot isolation: without making private input snapshots, a
    non-cooperating writer could still swap a path after this check or
    transiently replace and restore the original inode between checks.
    """

    if _verify_job_source(job) != source_signature:
        raise RuntimeError(
            f"Retargeting source changed during job validation: {job.source_path}",
        )
    _verify_job_solver_identity(job)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _cleanup_stale_solve_candidates(output_path: Path) -> None:
    """Remove candidates orphaned by an uncatchable prior process exit.

    The caller must hold this output's exclusive job lock, so no cooperating
    writer can own either the outer solve candidate or the nested atomic NPZ
    temporary file at the same time.
    """

    parent = output_path.parent
    if not parent.is_dir():
        return
    patterns = (
        f".{output_path.name}.solve.*.tmp.npz",
        f"..{output_path.name}.solve.*.tmp.npz.*.tmp.npz",
    )
    removed = False
    candidates = {
        candidate
        for pattern in patterns
        for candidate in parent.glob(pattern)
    }
    for candidate in sorted(candidates):
        candidate.unlink(missing_ok=True)
        removed = True
    if removed:
        _fsync_directory(parent)


def _publish_solved_artifact(
    job: RetargetJob,
    temporary_output_path: Path,
    *,
    source_signature: tuple[int, int, int, int, int],
) -> None:
    """Atomically publish after double-checking each path-based mutable input.

    Candidate validation can be expensive, so the source bytes and solver
    identity are checked both before and after it. The final check is the
    strongest path-based guarantee available without private immutable input
    snapshots; it is not advertised as protection from a non-cooperating
    writer in the final instructions between revalidation and ``replace``.
    """

    _revalidate_mutable_job_inputs(
        job,
        source_signature=source_signature,
    )
    if not result_artifact_matches_job(
        temporary_output_path,
        job,
    ):
        raise RuntimeError(
            "Solver output failed the exact schema, source, config, or external-asset contract before publication",
        )
    _revalidate_mutable_job_inputs(
        job,
        source_signature=source_signature,
    )
    temporary_output_path.replace(job.output_path)
    _fsync_directory(job.output_path.parent)


@contextmanager
def _job_file_lock(
    path: Path,
    *,
    shared: bool,
):
    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        lock_path,
        os.O_CREAT | os.O_RDWR,
        0o600,
    )
    try:
        fcntl.flock(
            descriptor,
            fcntl.LOCK_SH if shared else fcntl.LOCK_EX,
        )
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _resumed_job_result(job: RetargetJob) -> RetargetJobResult:
    return RetargetJobResult(
        output_path=job.output_path,
        source_path=job.source_path,
        sequence_key=job.sequence_key,
        variant=job.variant.name,
        resumed=True,
    )


def run_retargeting_job(job: RetargetJob) -> RetargetJobResult:
    """Execute or exactly resume one job under baseline and output locks.

    Source bytes and solver identity are fully rechecked after an existing
    artifact is validated and immediately before it is reported as resumed.
    These are path-based checks rather than filesystem snapshots, so
    non-cooperating writers must still avoid transient replace-and-restore
    operations during the final instruction window.
    """

    _validate_job_semantics(
        variant=job.variant,
        run_kind=job.run_kind,
        experiment_name=job.experiment_name,
    )
    is_augmented = job.run_kind == "augmentation"
    with ExitStack() as locks:
        locks.enter_context(
            _job_file_lock(
                job.state_lock_path,
                shared=True,
            )
        )
        if is_augmented:
            locks.enter_context(
                _job_file_lock(
                    job.baseline_lock_path,
                    shared=True,
                )
            )
        locks.enter_context(
            _job_file_lock(
                job.output_lock_path,
                shared=False,
            )
        )
        _cleanup_stale_solve_candidates(job.output_path)
        source_signature = _verify_job_source(job)
        _verify_job_solver_identity(job)
        if job.output_path.exists():
            if result_artifact_matches_job(job.output_path, job):
                if not job.config.overwrite_existing:
                    _revalidate_mutable_job_inputs(
                        job,
                        source_signature=source_signature,
                    )
                    return _resumed_job_result(job)
            elif not job.config.overwrite_existing:
                raise FileExistsError(
                    "Canonical result path already exists but does not match "
                    "this exact source/config job. Pass --overwrite-existing "
                    f"to replace it explicitly: {job.output_path}"
                )
            _revalidate_mutable_job_inputs(
                job,
                source_signature=source_signature,
            )
        return _run_retargeting_job_unlocked(
            job,
            source_signature=source_signature,
        )


def _run_retargeting_job_unlocked(
    job: RetargetJob,
    *,
    source_signature: tuple[int, int, int, int, int],
) -> RetargetJobResult:
    """Execute the sole load/preprocess/solve/save lifecycle."""

    _validate_job_semantics(
        variant=job.variant,
        run_kind=job.run_kind,
        experiment_name=job.experiment_name,
    )
    is_augmented = job.run_kind == "augmentation"
    if is_augmented and not result_artifact_matches_job(
        job.baseline_path,
        job,
        identity_baseline=True,
    ):
        raise ValueError(
            "Augmentation requires an exact identity baseline for the same "
            "source bytes and normalized identity configuration: "
            f"{job.baseline_path}"
        )
    if is_augmented:
        _revalidate_mutable_job_inputs(
            job,
            source_signature=source_signature,
        )

    cfg = job.config
    task_name = cfg.task_name
    task_type = cfg.task_type
    data_format = str(cfg.data_format)
    motion = load_human_motion(
        data_format,
        cfg.data_path,
        task_name,
        human_height=cfg.motion_data_config.human_height,
    )
    if _verify_job_source(job) != source_signature:
        raise RuntimeError(f"Retargeting source changed while loading: {job.source_path}")
    if motion.source_path.resolve() != job.source_path:
        raise ValueError(
            "Normalized job source does not match the motion adapter: "
            f"{job.source_path} != {motion.source_path.resolve()}"
        )
    validate_motion_skeleton_contract(
        motion,
        cfg.motion_data_config.resolved_demo_joints,
        cfg.motion_data_config.resolved_joint_parent_indices,
        data_format=data_format,
    )

    constants = create_task_constants(
        robot_config=cfg.robot_config,
        motion_data_config=cfg.motion_data_config,
        task_config=cfg.task_config,
        task_type=task_type,
    )
    constants.SOURCE_FPS = motion.fps
    constants.SOURCE_ROOT_QUATERNIONS = motion.root_quaternions_wxyz
    human_joints = motion.joints.copy()
    if data_format in {"lafan", "noetix_mocap"}:
        spine_joint_idx = constants.DEMO_JOINTS.index("Spine1")
        human_joints[:, spine_joint_idx, 2] -= 0.06

    (
        human_joint_quaternions,
        human_orientation_joint_names,
        orientation_source,
    ) = _direct_motion_orientations(motion)
    constants.SOURCE_ROOT_QUATERNIONS = motion.root_quaternions_wxyz
    if (
        constants.SOURCE_ROOT_QUATERNIONS is None
        and human_joint_quaternions is not None
        and constants.SOURCE_ROOT_JOINT in human_orientation_joint_names
    ):
        root_orientation_idx = human_orientation_joint_names.index(constants.SOURCE_ROOT_JOINT)
        constants.SOURCE_ROOT_QUATERNIONS = human_joint_quaternions[
            :,
            root_orientation_idx,
        ]
    if task_type == "object_interaction":
        if motion.object_poses_wxyz_xyz is None:
            raise ValueError(f"data_format={data_format!r} does not provide object poses")
        object_poses = motion.object_poses_wxyz_xyz.copy()
    else:
        object_poses = np.tile(
            np.asarray(
                [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                dtype=np.float32,
            ),
            (human_joints.shape[0], 1),
        )
    smpl_scale = constants.ROBOT_HEIGHT / motion.human_height

    job.output_path.parent.mkdir(parents=True, exist_ok=True)
    object_local_pts, object_local_pts_demo, object_urdf_path = setup_object_data(
        task_type,
        constants,
        cfg.task_config.object_dir,
        smpl_scale,
        cfg.task_config,
        is_augmented,
        object_scale_augmented=np.asarray(
            job.variant.object_scale,
            dtype=float,
        ),
        generated_assets_dir=job.generated_assets_dir,
    )
    retargeter = InteractionMeshRetargeter(
        **build_retargeter_kwargs_from_config(
            cfg.retargeter,
            constants,
            object_urdf_path,
            task_type,
        )
    )

    toe_names = cfg.motion_data_config.toe_names
    if task_type == "robot_only":
        human_joints = preprocess_motion_data(
            human_joints,
            retargeter,
            toe_names,
            smpl_scale,
        )
    else:
        human_joints, object_poses, _ = preprocess_motion_data(
            human_joints,
            retargeter,
            toe_names,
            scale=smpl_scale,
            object_poses=object_poses,
        )

    (
        q_init,
        q_nominal,
        object_poses_augmented,
        human_joints,
        object_poses,
    ) = initialize_robot_pose(
        task_type,
        human_joints,
        object_poses,
        constants,
        retargeter,
        cfg.task_config,
        is_augmented,
        job.output_path.parent,
        task_name,
        augmentation_translation=np.asarray(
            job.variant.translation,
            dtype=float,
        ),
        augmentation_rotation=job.variant.rotation,
        baseline_path=job.baseline_path,
    )
    orientation_target_world_rotation_deltas_wxyz = _object_pose_rotation_deltas_wxyz(
        object_poses,
        object_poses_augmented,
    )
    foot_sticking_sequences = extract_foot_sticking_sequence_velocity(
        human_joints,
        retargeter.demo_joints,
        toe_names,
    )
    if task_type == "object_interaction":
        foot_sticking_sequences[0][toe_names[0]] = False
        foot_sticking_sequences[0][toe_names[1]] = False

    result_metadata = {
        "run_kind": job.run_kind,
        "variant": job.variant.name,
        "experiment_name": job.experiment_name or "",
        "dataset_partition": job.dataset_partition,
        "sequence_key": job.sequence_key,
        "source_path": str(job.source_path),
        "source_sha256": job.source_sha256,
        "config_json": job.config_json,
        "config_sha256": job.config_sha256,
        "orientation_source": orientation_source,
        "source_human_height": float(motion.human_height),
        "human_position_scale": float(smpl_scale),
        "human_position_preprocessing": (
            "spine1_z_minus_0p06_then_ground_align_and_uniform_scale"
            if data_format in {"lafan", "noetix_mocap"}
            else "ground_align_and_uniform_scale"
        ),
    }
    logger.info(
        "Retargeting %s/%s for %s -> %s",
        job.run_kind,
        job.variant.name,
        task_name,
        job.output_path,
    )
    temporary_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{job.output_path.name}.solve.",
        suffix=".tmp.npz",
        dir=job.output_path.parent,
    )
    os.close(temporary_descriptor)
    temporary_output_path = Path(temporary_name)
    try:
        retargeter.retarget_motion(
            human_joint_motions=human_joints,
            human_joint_quaternions_wxyz=human_joint_quaternions,
            human_orientation_joint_names=human_orientation_joint_names,
            orientation_target_world_rotation_deltas_wxyz=(orientation_target_world_rotation_deltas_wxyz),
            object_poses=object_poses,
            object_poses_augmented=object_poses_augmented,
            object_points_local_demo=object_local_pts_demo,
            object_points_local=object_local_pts,
            foot_sticking_sequences=foot_sticking_sequences,
            q_a_init=q_init,
            q_nominal_list=q_nominal,
            original=not is_augmented,
            dest_res_path=str(temporary_output_path),
            fps=constants.SOURCE_FPS,
            result_metadata=result_metadata,
        )
        _publish_solved_artifact(
            job,
            temporary_output_path,
            source_signature=source_signature,
        )
    finally:
        temporary_output_path.unlink(missing_ok=True)
    if cfg.retargeter.debug:
        input("Press Enter to exit ...")
    return RetargetJobResult(
        output_path=job.output_path,
        source_path=job.source_path,
        sequence_key=job.sequence_key,
        variant=job.variant.name,
    )


def main(cfg: RetargetingConfig) -> RetargetFamilyResult:
    """Run one action through the same identity-first canonical variant family."""

    results: list[RetargetJobResult] = []
    source_sha256: str | None = None
    for variant in planned_variants(
        cfg.task_type,
        augmentation=cfg.augmentation,
    ):
        job = build_retarget_job(
            cfg,
            variant=variant,
            run_kind="single" if variant.is_identity else "augmentation",
            source_sha256=source_sha256,
        )
        source_sha256 = job.source_sha256
        result = run_retargeting_job(job)
        results.append(result)
    return RetargetFamilyResult(results=tuple(results))
