# ruff: noqa: CPY001

"""Run reproducible SO(3) tracking ablations through the shared pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

import numpy as np
import tyro

from holosoma_retargeting.config_types.data_type import (
    MotionDataConfig,
    normalize_data_format,
)
from holosoma_retargeting.config_types.retargeter import RetargeterConfig
from holosoma_retargeting.config_types.retargeting import RetargetingConfig
from holosoma_retargeting.data_utils.motion_data import (
    HumanMotion,
    load_human_motion,
    resolve_motion_path,
    validate_motion_task,
)
from holosoma_retargeting.retargeting_pipeline import (
    RetargetJob,
    RetargetVariant,
    build_retarget_job,
    normalize_retargeting_config,
    run_retargeting_job,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASK_NAME = "breaking+hippop.bvh_Skeleton1"
EXPERIMENT_NAME = "orientation_ablation"
INPUT_SUBSET_SCHEMA_VERSION = 3

ORIENTATION_ROLES = (
    "root",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_toe",
    "right_toe",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)

_BVH_ROLE_ALIASES = {
    "root": "Hips",
    "left_hip": "LeftUpLeg",
    "right_hip": "RightUpLeg",
    "left_knee": "LeftLeg",
    "right_knee": "RightLeg",
    "left_ankle": "LeftFoot",
    "right_ankle": "RightFoot",
    "left_toe": "LeftToeBase",
    "right_toe": "RightToeBase",
    "left_shoulder": "LeftArm",
    "right_shoulder": "RightArm",
    "left_elbow": "LeftForeArm",
    "right_elbow": "RightForeArm",
    "left_wrist": "LeftHand",
    "right_wrist": "RightHand",
}
_SMPL_ROLE_ALIASES = {
    "root": "Pelvis",
    "left_hip": "L_Hip",
    "right_hip": "R_Hip",
    "left_knee": "L_Knee",
    "right_knee": "R_Knee",
    "left_ankle": "L_Ankle",
    "right_ankle": "R_Ankle",
    "left_toe": "L_Foot",
    "right_toe": "R_Foot",
    "left_shoulder": "L_Shoulder",
    "right_shoulder": "R_Shoulder",
    "left_elbow": "L_Elbow",
    "right_elbow": "R_Elbow",
    "left_wrist": "L_Wrist",
    "right_wrist": "R_Wrist",
}
_SMPLH_ROLE_ALIASES = {
    **_SMPL_ROLE_ALIASES,
    "left_toe": "L_Toe",
    "right_toe": "R_Toe",
}
ORIENTATION_ROLE_ALIASES: dict[str, dict[str, str]] = {
    "noetix_mocap": _BVH_ROLE_ALIASES,
    "lafan": _BVH_ROLE_ALIASES,
    "mocap": _BVH_ROLE_ALIASES,
    "amass": _SMPL_ROLE_ALIASES,
    "gvhmr": _SMPL_ROLE_ALIASES,
    "omomo": _SMPLH_ROLE_ALIASES,
}

# Compatibility name vector used by the existing Noetix search/comparison
# helpers. Profiles below are role-based and are expanded to source names only
# after the data format has been normalized.
ORIENTATION_JOINTS = tuple(ORIENTATION_ROLE_ALIASES["noetix_mocap"][role] for role in ORIENTATION_ROLES)

# Relative per-role coefficients. The CLI's --weight-scales values multiply
# them to produce the actual source-joint weights passed into the SQP objective.
PROFILE_WEIGHTS: dict[str, dict[str, float]] = {
    "baseline": {},
    "root": {
        "root": 1.0,
    },
    "feet": {
        "left_ankle": 2.0,
        "right_ankle": 2.0,
    },
    "gmr_legs": {
        "left_hip": 1.0,
        "right_hip": 1.0,
        "left_knee": 1.0,
        "right_knee": 1.0,
        "left_ankle": 1.0,
        "right_ankle": 1.0,
    },
    "shoulders": {
        "left_shoulder": 1.0,
        "right_shoulder": 1.0,
    },
    "shoulders_feet": {
        "left_shoulder": 1.0,
        "right_shoulder": 1.0,
        "left_ankle": 1.0,
        "right_ankle": 1.0,
    },
    "upper": {
        "left_shoulder": 1.0,
        "right_shoulder": 1.0,
        "left_elbow": 1.5,
        "right_elbow": 1.5,
        "left_wrist": 1.0,
        "right_wrist": 1.0,
    },
    "full": {
        "root": 1.0,
        "left_hip": 0.5,
        "right_hip": 0.5,
        "left_knee": 0.75,
        "right_knee": 0.75,
        "left_ankle": 2.0,
        "right_ankle": 2.0,
        "left_toe": 2.0,
        "right_toe": 2.0,
        "left_shoulder": 1.0,
        "right_shoulder": 1.0,
        "left_elbow": 1.5,
        "right_elbow": 1.5,
        "left_wrist": 1.0,
        "right_wrist": 1.0,
    },
    "full_equal": dict.fromkeys(ORIENTATION_ROLES, 1.0),
    "balanced_optimal": dict.fromkeys(ORIENTATION_ROLES, 0.085),
}

_FRAME_SUBSET_FORMATS = frozenset(
    {
        "amass",
        "gvhmr",
        "lafan",
        "noetix_mocap",
    }
)
_FRAME_ALIGNED_NPZ_FIELDS = frozenset(
    {
        "global_joint_positions",
        "global_joint_quaternions_wxyz",
        "orientation_quaternions_wxyz",
        "root_quaternions_wxyz",
    }
)
_KNOWN_VECTOR_METADATA_FIELDS = frozenset(
    {
        "joint_names",
        "joint_parents",
        "joint_parent_indices",
        "orientation_joint_names",
        "raw_joint_names",
    }
)


@dataclass
class Config:
    robot: Literal["g1", "e1"] = "e1"
    task_type: Literal["robot_only", "object_interaction", "climbing"] = "robot_only"
    data_format: str = "noetix_mocap"
    data_path: Path = PACKAGE_ROOT / "demo_data" / "noetix_mocap" / "0724_BEITI"
    task_name: str = DEFAULT_TASK_NAME
    output_root: Path = PACKAGE_ROOT / "demo_results" / "v1"
    orientation_alignment_mode: Literal["t_pose", "first_frame"] = "t_pose"
    variants: tuple[str, ...] = (
        "baseline",
        "root",
        "feet",
        "gmr_legs",
        "shoulders",
        "upper",
        "full",
    )
    weight_scales: tuple[float, ...] = (1.0,)
    frame_start: int = 0
    frame_count: int | None = None
    overwrite: bool = False
    dry_run: bool = False
    fail_fast: bool = False


def orientation_weights_for_variant(
    variant: str,
    weight_scale: float,
    *,
    data_format: str = "noetix_mocap",
) -> dict[str, float]:
    """Expand one semantic profile into source-format joint weights."""

    if variant not in PROFILE_WEIGHTS:
        raise ValueError(f"Unknown orientation ablation variant {variant!r}; available: {sorted(PROFILE_WEIGHTS)}")
    if not np.isfinite(weight_scale) or weight_scale < 0.0:
        raise ValueError("weight_scale must be finite and non-negative")
    if variant != "baseline" and weight_scale == 0.0:
        raise ValueError(
            "Only the named baseline variant may have zero orientation "
            "weights; non-baseline weight_scale must be positive"
        )
    canonical_format = normalize_data_format(data_format)
    try:
        aliases = ORIENTATION_ROLE_ALIASES[canonical_format]
    except KeyError as exc:
        supported = ", ".join(sorted(ORIENTATION_ROLE_ALIASES))
        raise ValueError(
            "Orientation ablation profiles have no source-joint role aliases "
            f"for data_format={canonical_format!r}; supported formats: {supported}"
        ) from exc
    active = PROFILE_WEIGHTS[variant]
    return {aliases[role]: float(active.get(role, 0.0) * weight_scale) for role in ORIENTATION_ROLES}


def ablation_run_specs(
    variants: tuple[str, ...],
    weight_scales: tuple[float, ...],
) -> tuple[tuple[str, str, float], ...]:
    """Expand profiles into unique run names, keeping one zero-weight baseline."""
    specs: list[tuple[str, str, float]] = []
    seen_names: set[str] = set()
    for variant in variants:
        scales = (1.0,) if variant == "baseline" else weight_scales
        for weight_scale in scales:
            run_name = variant if weight_scale == 1.0 else f"{variant}_x{weight_scale:g}"
            if run_name in seen_names:
                raise ValueError(f"Duplicate ablation run name: {run_name}")
            orientation_weights_for_variant(variant, weight_scale)
            seen_names.add(run_name)
            specs.append((run_name, variant, float(weight_scale)))
    if not specs:
        raise ValueError("Orientation ablation must plan at least one variant/weight run")
    return tuple(specs)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _update_digest_component(
    digest: Any,
    value: str | bytes,
) -> None:
    encoded = value if isinstance(value, bytes) else value.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
    digest.update(encoded)


def preparation_payload_sha256(
    payload: dict[str, np.ndarray],
    *,
    excluded_fields: frozenset[str] = frozenset({"preparation_payload_sha256"}),
) -> str:
    """Hash exact named array values without trusting serialized metadata."""

    digest = hashlib.sha256()
    for key in sorted(payload):
        if key in excluded_fields:
            continue
        value = np.asarray(payload[key])
        if value.dtype.hasobject:
            raise TypeError(f"Preparation payload field {key!r} must not use object dtype")
        contiguous = np.ascontiguousarray(value)
        _update_digest_component(digest, key)
        _update_digest_component(digest, value.dtype.str)
        _update_digest_component(
            digest,
            json.dumps(
                list(value.shape),
                separators=(",", ":"),
            ),
        )
        _update_digest_component(digest, contiguous.tobytes(order="C"))
    return digest.hexdigest()


def preparation_payload_matches(
    path: Path,
    expected_payload: dict[str, np.ndarray],
) -> bool:
    """Compare every cached field to source-derived expected bytes."""

    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != set(expected_payload):
                return False
            cached_payload = {key: np.asarray(data[key]) for key in data.files}
        cached_digest_array = np.asarray(cached_payload["preparation_payload_sha256"])
        if cached_digest_array.ndim != 0 or str(cached_digest_array.item()) != preparation_payload_sha256(
            cached_payload
        ):
            return False
        for key, expected in expected_payload.items():
            actual = np.asarray(cached_payload[key])
            expected_array = np.asarray(expected)
            if (
                actual.dtype != expected_array.dtype
                or actual.shape != expected_array.shape
                or np.ascontiguousarray(actual).tobytes(order="C")
                != np.ascontiguousarray(expected_array).tobytes(order="C")
            ):
                return False
    except (KeyError, OSError, TypeError, ValueError):
        return False
    return True


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as temporary_file:
            temporary_file.write(f"{json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)}\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        temporary_path.replace(path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _atomic_write_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as temporary_file:
            np.savez_compressed(temporary_file, **payload)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        temporary_path.replace(path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _subset_identity(
    *,
    data_format: str,
    source_sha256: str,
    frame_start: int,
    frame_end: int,
) -> tuple[str, str]:
    implementation_sha256 = _sha256(Path(__file__).resolve())
    digest = hashlib.sha256()
    for label, value in (
        ("schema_version", str(INPUT_SUBSET_SCHEMA_VERSION)),
        ("data_format", data_format),
        ("source_sha256", source_sha256),
        ("frame_start", str(frame_start)),
        ("frame_end", str(frame_end)),
        ("implementation_sha256", implementation_sha256),
    ):
        for component in (label, value):
            encoded = component.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
            digest.update(encoded)
    return digest.hexdigest(), implementation_sha256


def _valid_cached_subset(
    path: Path,
    *,
    expected_payload: dict[str, np.ndarray],
) -> bool:
    """Accept only the exact subset derived from the current formal source."""

    return preparation_payload_matches(path, expected_payload)


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PACKAGE_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _require_direct_orientation(
    motion: HumanMotion,
    *,
    data_format: str,
    source_path: Path,
) -> None:
    if (
        motion.orientation_joint_names is None
        or motion.orientation_quaternions_wxyz is None
        or motion.orientation_source is None
    ):
        raise ValueError(
            "Orientation ablation requires adapter-proven direct source "
            f"orientations; {data_format} source {source_path} has none. "
            "Positions, bone directions, and toe directions are never used "
            "to estimate missing orientations."
        )
    expected_names = set(ORIENTATION_ROLE_ALIASES[data_format].values())
    missing_names = sorted(expected_names.difference(motion.orientation_joint_names))
    if missing_names:
        raise ValueError(
            f"{data_format} source {source_path} direct orientation tensor "
            "does not cover every ablation role; missing source joints: "
            f"{missing_names}"
        )


def _validate_prepared_subset(
    *,
    data_format: str,
    subset_dir: Path,
    task_name: str,
    source_motion: HumanMotion,
    frame_start: int,
    frame_end: int,
) -> None:
    subset_motion = load_human_motion(
        data_format,
        subset_dir,
        task_name,
    )
    _require_direct_orientation(
        subset_motion,
        data_format=data_format,
        source_path=subset_motion.source_path,
    )
    expected_joints = source_motion.joints[frame_start:frame_end]
    if not np.array_equal(subset_motion.joints, expected_joints):
        raise ValueError("Prepared frame subset changed the adapter's human joint tensor")
    if (
        subset_motion.orientation_joint_names != source_motion.orientation_joint_names
        or subset_motion.orientation_source != source_motion.orientation_source
        or not np.array_equal(
            subset_motion.orientation_quaternions_wxyz,
            source_motion.orientation_quaternions_wxyz[frame_start:frame_end],
        )
    ):
        raise ValueError(
            "Prepared frame subset did not preserve the direct orientation names, provenance, and quaternion bytes"
        )


def _prepare_input(cfg: Config) -> tuple[Path, Path]:
    data_format = validate_motion_task(cfg.data_format, cfg.task_type)
    if data_format not in ORIENTATION_ROLE_ALIASES:
        supported = ", ".join(sorted(ORIENTATION_ROLE_ALIASES))
        raise ValueError(
            f"Orientation ablation does not define semantic roles for "
            f"data_format={data_format!r}; supported formats: {supported}"
        )
    source_path = resolve_motion_path(
        cfg.data_path,
        cfg.task_name,
        data_format,
    ).resolve()
    wants_subset = cfg.frame_start != 0 or cfg.frame_count is not None
    if wants_subset and (data_format not in _FRAME_SUBSET_FORMATS or source_path.suffix.lower() != ".npz"):
        supported = ", ".join(sorted(_FRAME_SUBSET_FORMATS))
        raise ValueError(
            "Frame subsetting is only supported for metadata-preserving NPZ "
            f"adapters ({supported}); data_format={data_format!r} resolved "
            f"to {source_path.suffix or '<no extension>'!r}. Run the complete "
            "source instead."
        )

    source_motion = load_human_motion(
        data_format,
        cfg.data_path,
        cfg.task_name,
    )
    if source_motion.source_path.resolve() != source_path:
        raise ValueError(
            "Motion registry resolution changed between source discovery and "
            f"loading: {source_path} != {source_motion.source_path}"
        )
    _require_direct_orientation(
        source_motion,
        data_format=data_format,
        source_path=source_path,
    )
    if cfg.frame_start == 0 and cfg.frame_count is None:
        return cfg.data_path.expanduser().resolve(), source_path

    source_sha256 = _sha256(source_path)
    with np.load(source_path, allow_pickle=False) as data:
        source_frames = int(source_motion.joints.shape[0])
        raw_frames = int(np.asarray(data["global_joint_positions"]).shape[0])
        if raw_frames != source_frames:
            raise ValueError(
                f"{data_format} frame subset would not preserve adapter "
                f"semantics: raw frames={raw_frames}, loaded frames={source_frames}"
            )
        start = int(cfg.frame_start)
        end = source_frames if cfg.frame_count is None else start + int(cfg.frame_count)
        if start < 0 or start >= source_frames or end <= start or end > source_frames:
            raise ValueError(f"Invalid frame interval [{start}, {end}) for {source_frames} frames")
        identity_sha256, implementation_sha256 = _subset_identity(
            data_format=data_format,
            source_sha256=source_sha256,
            frame_start=start,
            frame_end=end,
        )
        subset_dir = (
            cfg.output_root.expanduser().resolve() / "ablations" / EXPERIMENT_NAME / "_inputs" / identity_sha256
        )
        subset_path = subset_dir / source_path.name
        payload: dict[str, np.ndarray] = {}
        for key in data.files:
            value = np.asarray(data[key])
            if key in _FRAME_ALIGNED_NPZ_FIELDS:
                if value.ndim == 0 or value.shape[0] != source_frames:
                    raise ValueError(
                        f"{data_format} frame-aligned field {key!r} has "
                        f"shape {value.shape}, expected first dimension "
                        f"{source_frames}"
                    )
                payload[key] = value[start:end]
            elif value.ndim > 0 and value.shape[0] == source_frames and key not in _KNOWN_VECTOR_METADATA_FIELDS:
                raise ValueError(
                    f"Cannot safely frame-subset {source_path}: unknown "
                    f"frame-aligned field {key!r} must be classified explicitly"
                )
            else:
                payload[key] = value
    lineage = {
        "schema_version": INPUT_SUBSET_SCHEMA_VERSION,
        "data_format": data_format,
        "source_path": str(source_path),
        "source_sha256": source_sha256,
        "frame_start": start,
        "frame_end": end,
        "implementation_sha256": implementation_sha256,
        "identity_sha256": identity_sha256,
        "transformation": "explicit_frame_fields_only",
    }
    lineage_json = json.dumps(
        lineage,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    lineage_sha256 = hashlib.sha256(lineage_json.encode("utf-8")).hexdigest()
    payload.update(
        {
            "preparation_schema_version": np.asarray(
                INPUT_SUBSET_SCHEMA_VERSION,
                dtype=np.int64,
            ),
            "preparation_data_format": np.asarray(data_format),
            "preparation_source_path": np.asarray(str(source_path)),
            "preparation_source_sha256": np.asarray(source_sha256),
            "preparation_frame_start": np.asarray(start, dtype=np.int64),
            "preparation_frame_end": np.asarray(end, dtype=np.int64),
            "preparation_identity_sha256": np.asarray(identity_sha256),
            "preparation_implementation_sha256": np.asarray(implementation_sha256),
            "preparation_lineage_json": np.asarray(lineage_json),
            "preparation_lineage_sha256": np.asarray(lineage_sha256),
        }
    )
    payload["preparation_payload_sha256"] = np.asarray(preparation_payload_sha256(payload))
    if _sha256(source_path) != source_sha256:
        raise RuntimeError(f"Source input changed while preparing subset: {source_path}")
    if _valid_cached_subset(
        subset_path,
        expected_payload=payload,
    ):
        _validate_prepared_subset(
            data_format=data_format,
            subset_dir=subset_dir,
            task_name=cfg.task_name,
            source_motion=source_motion,
            frame_start=start,
            frame_end=end,
        )
        return subset_dir, subset_path
    _atomic_write_npz(subset_path, payload)
    if not _valid_cached_subset(
        subset_path,
        expected_payload=payload,
    ):
        raise ValueError(f"Failed to prepare a valid frame subset: {subset_path}")
    _validate_prepared_subset(
        data_format=data_format,
        subset_dir=subset_dir,
        task_name=cfg.task_name,
        source_motion=source_motion,
        frame_start=start,
        frame_end=end,
    )
    return subset_dir, subset_path


def _safe_path_component(value: str) -> str:
    """Legacy lossy helper retained only for callers being migrated."""

    component = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    if not component:
        raise ValueError(f"Cannot derive a safe path component from {value!r}")
    return component


def dataset_partition_identity(data_path: Path) -> str:
    """Return a readable partition bound to the complete canonical root path."""

    resolved = Path(data_path).expanduser().resolve()
    label = resolved.name or "root"
    digest = hashlib.sha256(resolved.as_posix().encode("utf-8")).hexdigest()
    return f"{label}--path-sha256-{digest}"


def _ablation_sequence_key(cfg: Config) -> str:
    if cfg.frame_start == 0 and cfg.frame_count is None:
        return cfg.task_name
    frame_end = "end" if cfg.frame_count is None else f"{cfg.frame_start + cfg.frame_count:06d}"
    return f"{cfg.task_name}/frames_{cfg.frame_start:06d}_{frame_end}"


def _encode_summary_component(value: str) -> str:
    """Mirror the shared pipeline's reversible portable component encoding."""

    if not value or value in {".", ".."} or "\0" in value or "/" in value:
        raise ValueError("Summary identity values must contain exactly one path component")
    encoded = quote(value, safe="._-")
    if len(encoded.encode("ascii")) > 240:
        encoded = f"sha256-{hashlib.sha256(value.encode('utf-8')).hexdigest()}"
    return encoded


def _encoded_summary_sequence(sequence_key: str) -> Path:
    sequence_path = Path(sequence_key)
    if sequence_path.is_absolute():
        raise ValueError(f"Invalid sequence_key: {sequence_key!r}")
    parts = sequence_path.parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"Invalid sequence_key: {sequence_key!r}")
    return Path(*(_encode_summary_component(part) for part in parts))


def _summary_path(cfg: Config) -> Path:
    """Return the invocation summary path without cross-job collisions."""

    data_format = validate_motion_task(cfg.data_format, cfg.task_type)
    return (
        cfg.output_root.expanduser().resolve()
        / "ablations"
        / EXPERIMENT_NAME
        / "_summaries"
        / cfg.robot
        / cfg.task_type
        / data_format
        / _encode_summary_component(dataset_partition_identity(cfg.data_path))
        / _encoded_summary_sequence(_ablation_sequence_key(cfg))
        / "summary.json"
    )


def _build_ablation_job(
    *,
    cfg: Config,
    input_dir: Path,
    input_path: Path,
    run_name: str,
    orientation_weights: dict[str, float],
) -> RetargetJob:
    retargeting_cfg = _retargeting_config(
        input_dir=input_dir,
        output_dir=cfg.output_root,
        task_name=cfg.task_name,
        orientation_weights=orientation_weights,
        robot=cfg.robot,
        task_type=cfg.task_type,
        data_format=cfg.data_format,
        orientation_alignment_mode=cfg.orientation_alignment_mode,
    )
    return build_retarget_job(
        retargeting_cfg,
        variant=RetargetVariant(name=run_name),
        run_kind="ablation",
        experiment_name=EXPERIMENT_NAME,
        results_root=cfg.output_root,
        dataset_partition=dataset_partition_identity(cfg.data_path),
        sequence_key=_ablation_sequence_key(cfg),
        source_path=input_path,
        overwrite_existing=cfg.overwrite,
    )


def _result_summary(result_path: Path, elapsed_seconds: float) -> dict[str, Any]:
    with np.load(result_path, allow_pickle=False) as data:
        qpos = np.asarray(data["qpos"])
        human_mapped = np.asarray(data["mapped_human_joints"])
        robot_mapped = np.asarray(data["mapped_robot_joints"])
        position_errors = np.linalg.norm(robot_mapped - human_mapped, axis=-1)
        orientation_errors = np.asarray(data["orientation_errors_rad"])
        orientation_names = [str(name) for name in np.asarray(data["orientation_human_joint_names"]).tolist()]
        per_link_orientation = {
            name: {
                "mean_rad": float(np.mean(orientation_errors[:, link_idx])),
                "median_rad": float(np.median(orientation_errors[:, link_idx])),
                "p95_rad": float(np.percentile(orientation_errors[:, link_idx], 95)),
            }
            for link_idx, name in enumerate(orientation_names)
        }
        mapped_names = [str(name) for name in np.asarray(data["mapped_human_joint_names"]).tolist()]
        per_link_position = {
            name: {
                "mean_m": float(np.mean(position_errors[:, link_idx])),
                "p95_m": float(np.percentile(position_errors[:, link_idx], 95)),
            }
            for link_idx, name in enumerate(mapped_names)
        }
        orientation_values = orientation_errors.reshape(-1)
        sqp_iterations = np.asarray(data["sqp_iteration_counts"])
        actuated_qpos = qpos[:, 7:]
        joint_steps = np.linalg.norm(
            np.diff(actuated_qpos, axis=0),
            axis=1,
        )
        joint_accelerations = np.linalg.norm(
            np.diff(actuated_qpos, n=2, axis=0),
            axis=1,
        )
        return {
            "result_path": str(result_path),
            "frames": int(qpos.shape[0]),
            "qpos_shape": list(qpos.shape),
            "all_finite": bool(np.isfinite(qpos).all()),
            "elapsed_seconds": float(elapsed_seconds),
            "orientation_tracking_enabled": bool(np.asarray(data["orientation_tracking_enabled"]).item()),
            "orientation_alignment_mode": str(np.asarray(data["orientation_alignment_mode"]).item()),
            "orientation_alignment_quaternions_wxyz": np.asarray(
                data["orientation_alignment_quaternions_wxyz"]
            ).tolist(),
            "orientation_weights": np.asarray(data["orientation_weights"]).tolist(),
            "orientation_overall": {
                "mean_rad": (float(np.mean(orientation_values)) if orientation_values.size else None),
                "median_rad": (float(np.median(orientation_values)) if orientation_values.size else None),
                "p95_rad": (float(np.percentile(orientation_values, 95)) if orientation_values.size else None),
            },
            "orientation_per_link": per_link_orientation,
            "mapped_position_error": {
                "mean_m": float(np.mean(position_errors)),
                "median_m": float(np.median(position_errors)),
                "p95_m": float(np.percentile(position_errors, 95)),
            },
            "mapped_position_error_per_link": per_link_position,
            "actuated_motion_smoothness": {
                "mean_step_rad": (float(np.mean(joint_steps)) if joint_steps.size else 0.0),
                "p95_step_rad": (float(np.percentile(joint_steps, 95)) if joint_steps.size else 0.0),
                "mean_second_difference_rad": (
                    float(np.mean(joint_accelerations)) if joint_accelerations.size else 0.0
                ),
                "p95_second_difference_rad": (
                    float(np.percentile(joint_accelerations, 95)) if joint_accelerations.size else 0.0
                ),
            },
            "sqp_iterations": {
                "mean": float(np.mean(sqp_iterations)),
                "p95": float(np.percentile(sqp_iterations, 95)),
                "max": int(np.max(sqp_iterations)),
            },
            "foot_sticking_fallback_frames": int(np.asarray(data["foot_sticking_fallback_frames"]).size),
            "foot_sticking_release_frames": int(np.asarray(data["foot_sticking_release_frames"]).size),
            "object_non_penetration_release_frames": int(
                np.asarray(data["object_non_penetration_release_frames"]).size
            ),
        }


def _comparisons_to_baseline(
    summaries: dict[str, Any],
) -> dict[str, dict[str, float]]:
    baseline = summaries.get("baseline")
    if not isinstance(baseline, dict) or "orientation_overall" not in baseline:
        return {}
    baseline_orientation = baseline["orientation_overall"]["mean_rad"]
    baseline_position = baseline["mapped_position_error"]["mean_m"]
    baseline_smoothness = baseline["actuated_motion_smoothness"]["mean_second_difference_rad"]
    comparisons: dict[str, dict[str, float]] = {}
    for run_name, summary in summaries.items():
        if run_name == "baseline" or "orientation_overall" not in summary:
            continue
        orientation = summary["orientation_overall"]["mean_rad"]
        position = summary["mapped_position_error"]["mean_m"]
        smoothness = summary["actuated_motion_smoothness"]["mean_second_difference_rad"]
        comparisons[run_name] = {
            "orientation_mean_delta_rad": float(orientation - baseline_orientation),
            "orientation_mean_change_percent": float(100.0 * (orientation / baseline_orientation - 1.0)),
            "position_mean_delta_m": float(position - baseline_position),
            "position_mean_change_percent": float(100.0 * (position / baseline_position - 1.0)),
            "joint_second_difference_delta_rad": float(smoothness - baseline_smoothness),
        }
    return comparisons


def _derived_orientation_mapping(
    motion_config: MotionDataConfig,
    *,
    orientation_weights: dict[str, float],
) -> dict[str, str]:
    """Bind semantic source joints to links selected by shared mappings."""

    required_names = tuple(orientation_weights)
    dedicated_mapping = motion_config.resolved_orientation_joints_mapping
    if set(required_names).issubset(dedicated_mapping):
        return {name: dedicated_mapping[name] for name in required_names}

    aliases = ORIENTATION_ROLE_ALIASES[motion_config.data_format]
    position_mapping = motion_config.resolved_joints_mapping
    position_source_overrides = {
        "lafan": {
            "root": "Spine1",
        },
        "mocap": {
            "root": "Spine1",
            "left_wrist": "LeftHandMiddle3",
            "right_wrist": "RightHandMiddle3",
        },
    }.get(motion_config.data_format, {})
    mapping: dict[str, str] = {}
    missing_roles: list[str] = []
    for role in ORIENTATION_ROLES:
        source_name = aliases[role]
        position_source_name = position_source_overrides.get(
            role,
            source_name,
        )
        robot_link = position_mapping.get(position_source_name)
        if robot_link is None:
            missing_roles.append(f"{role} ({source_name} via {position_source_name})")
        else:
            mapping[source_name] = robot_link
    if missing_roles:
        raise ValueError(
            "Shared position mapping cannot supply every orientation-ablation "
            f"role for data_format={motion_config.data_format!r}, "
            f"robot={motion_config.robot_type!r}: {missing_roles}"
        )
    return mapping


def _has_complete_t_pose_calibration(
    motion_config: MotionDataConfig,
    orientation_mapping: dict[str, str],
) -> bool:
    human_frames = motion_config.resolved_orientation_t_pose_human_quaternions_wxyz
    return set(orientation_mapping).issubset(human_frames) and (
        motion_config.resolved_orientation_t_pose_robot_base_quaternion_wxyz is not None
    )


def _retargeting_config(
    *,
    input_dir: Path,
    output_dir: Path,
    task_name: str,
    orientation_weights: dict[str, float],
    robot: str = "e1",
    task_type: str = "robot_only",
    data_format: str = "noetix_mocap",
    orientation_alignment_mode: Literal[
        "t_pose",
        "first_frame",
    ] = "t_pose",
) -> RetargetingConfig:
    """Build and normalize one shared-pipeline ablation configuration."""

    normalized = normalize_retargeting_config(
        RetargetingConfig(
            task_type=task_type,
            robot=robot,
            data_format=data_format,
            task_name=task_name,
            data_path=input_dir,
            save_dir=output_dir,
            retargeter=RetargeterConfig(
                orientation_weights=orientation_weights,
                orientation_alignment_mode=orientation_alignment_mode,
            ),
        )
    )
    orientation_mapping = _derived_orientation_mapping(
        normalized.motion_data_config,
        orientation_weights=orientation_weights,
    )
    if orientation_alignment_mode == "t_pose" and not _has_complete_t_pose_calibration(
        normalized.motion_data_config,
        orientation_mapping,
    ):
        enabled = any(weight > 0.0 for weight in orientation_weights.values())
        purpose = "non-zero orientation tracking" if enabled else "zero-weight baseline orientation diagnostics"
        raise ValueError(
            f"data_format={normalized.data_format!r}, robot={robot!r} has no "
            f"complete shared T-pose calibration for {purpose}. Pass "
            "--orientation-alignment-mode first_frame explicitly; first_frame "
            "uses only adapter-proven direct source orientations and robot FK, "
            "never positions or inferred directions."
        )
    if orientation_mapping != (normalized.motion_data_config.resolved_orientation_joints_mapping):
        normalized.motion_data_config = replace(
            normalized.motion_data_config,
            orientation_joints_mapping=orientation_mapping,
        )
    return normalized


def main(cfg: Config) -> None:
    input_dir, input_path = _prepare_input(cfg)
    summary_path = _summary_path(cfg)
    planned_jobs: list[tuple[str, str, float, dict[str, float], RetargetJob]] = []
    for run_name, variant, weight_scale in ablation_run_specs(
        cfg.variants,
        cfg.weight_scales,
    ):
        weights = orientation_weights_for_variant(
            variant,
            weight_scale,
            data_format=cfg.data_format,
        )
        job = _build_ablation_job(
            cfg=cfg,
            input_dir=input_dir,
            input_path=input_path,
            run_name=run_name,
            orientation_weights=weights,
        )
        planned_jobs.append((run_name, variant, weight_scale, weights, job))

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Any] = {}
    failures: list[str] = []
    first_failure: Exception | None = None
    attempted_runs: list[str] = []

    for run_name, variant, weight_scale, weights, job in planned_jobs:
        attempted_runs.append(run_name)
        result_path = job.output_path
        result_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "run_name": run_name,
            "variant": job.variant.name,
            "profile": variant,
            "weight_scale": weight_scale,
            "orientation_weights": weights,
            "input_path": str(job.source_path),
            "input_sha256": job.source_sha256,
            "git_commit": _git_commit(),
            "frame_start": cfg.frame_start,
            "frame_count": cfg.frame_count,
            "robot": job.config.robot,
            "task_type": job.config.task_type,
            "data_format": job.config.data_format,
            "orientation_alignment_mode": (job.config.retargeter.orientation_alignment_mode),
            "retargeting_config": _jsonable(asdict(job.config)),
            "run_kind": job.run_kind,
            "experiment_name": job.experiment_name,
            "config_sha256": job.config_sha256,
            "canonical_baseline_path": str(job.baseline_path),
            "result_path": str(result_path),
            "status": "planned" if cfg.dry_run else "running",
        }
        manifest_path = result_path.parent / "manifest.json"
        _atomic_write_json(manifest_path, manifest)
        if cfg.dry_run:
            summaries[run_name] = manifest
            continue
        start_time = time.monotonic()
        try:
            completed_job = run_retargeting_job(job)
            elapsed = time.monotonic() - start_time
            result_path = completed_job.output_path
            manifest["result_path"] = str(result_path)
            manifest["status"] = "reused" if completed_job.resumed else "completed"
            manifest["elapsed_seconds"] = 0.0 if completed_job.resumed else elapsed
            summary = _result_summary(
                result_path,
                elapsed_seconds=manifest["elapsed_seconds"],
            )
            summary["canonical_baseline_path"] = str(job.baseline_path)
            summaries[run_name] = summary
        except Exception as exc:
            manifest["status"] = "failed"
            manifest["error"] = f"{type(exc).__name__}: {exc}"
            failures.append(run_name)
            summaries[run_name] = {
                "status": "failed",
                "error": manifest["error"],
                "result_path": str(job.output_path),
                "config_sha256": job.config_sha256,
                "variant": job.variant.name,
                "profile": variant,
            }
            if first_failure is None:
                first_failure = exc
        finally:
            _atomic_write_json(manifest_path, manifest)
        if cfg.fail_fast and first_failure is not None:
            break

    attempted_set = set(attempted_runs)
    not_run = [run_name for run_name, _, _, _, _ in planned_jobs if run_name not in attempted_set]
    for run_name, variant, weight_scale, weights, job in planned_jobs:
        if run_name in attempted_set:
            continue
        summaries[run_name] = {
            "status": "not_run",
            "reason": "fail_fast",
            "result_path": str(job.output_path),
            "config_sha256": job.config_sha256,
            "variant": job.variant.name,
            "profile": variant,
            "weight_scale": weight_scale,
            "orientation_weights": weights,
        }

    summary_payload = {
        "status": ("planned" if cfg.dry_run else ("failed" if failures else "completed")),
        "experiment_name": EXPERIMENT_NAME,
        "results_root": str(cfg.output_root.expanduser().resolve()),
        "robot": cfg.robot,
        "task_type": cfg.task_type,
        "data_format": validate_motion_task(
            cfg.data_format,
            cfg.task_type,
        ),
        "task_name": cfg.task_name,
        "input_path": str(planned_jobs[0][4].source_path),
        "input_sha256": planned_jobs[0][4].source_sha256,
        "dataset_partition": planned_jobs[0][4].dataset_partition,
        "sequence_key": planned_jobs[0][4].sequence_key,
        "orientation_alignment_mode": cfg.orientation_alignment_mode,
        "runs": summaries,
        "comparisons_to_baseline": _comparisons_to_baseline(summaries),
        "failures": failures,
        "not_run": not_run,
    }
    _atomic_write_json(summary_path, summary_payload)
    if failures:
        error = RuntimeError(f"Orientation ablations failed: {failures}; see {summary_path}")
        if first_failure is not None:
            raise error from first_failure
        raise error


if __name__ == "__main__":
    main(tyro.cli(Config))
