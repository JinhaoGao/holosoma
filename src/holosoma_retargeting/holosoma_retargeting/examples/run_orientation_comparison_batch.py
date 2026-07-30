# ruff: noqa: CPY001

"""Build a full Noetix baseline-versus-orientation comparison dataset."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
import shlex
import subprocess
import tempfile
import time
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tyro

from holosoma_retargeting.config_types import data_type as data_type_module
from holosoma_retargeting.config_types.data_type import (
    APPROVED_DIRECT_ORIENTATION_SOURCES,
    DEMO_JOINTS_REGISTRY,
)
from holosoma_retargeting.data_utils import convert_noetix_bvh
from holosoma_retargeting.data_utils.convert_noetix_bvh import convert_file
from holosoma_retargeting.examples.run_orientation_ablation import (
    ORIENTATION_JOINTS,
    PACKAGE_ROOT,
    _retargeting_config,
    dataset_partition_identity,
    orientation_weights_for_variant,
    preparation_payload_matches,
    preparation_payload_sha256,
)
from holosoma_retargeting.retargeting_fingerprint import (
    NUMERICAL_RUNTIME_ENVIRONMENT,
)
from holosoma_retargeting.retargeting_pipeline import (
    RetargetJob,
    RetargetVariant,
    build_retarget_job,
    run_retargeting_job,
)

DEFAULT_DATA_PATH = PACKAGE_ROOT / "demo_data" / "noetix_mocap" / "0724_BEITI"
DEFAULT_BVH_PATH = PACKAGE_ROOT / "demo_data" / "noetix_ori" / "0724_BEITI"
DEFAULT_OUTPUT_ROOT = PACKAGE_ROOT / "demo_results" / "v1"
EXPERIMENT_NAME = "orientation_comparison"
INPUT_PREPARATION_SCHEMA_VERSION = 2
INPUT_PREPARATION_TARGET_FPS = 30.0
INPUT_PREPARATION_DROP_JUMP_THRESHOLD_M = 2.0
INPUT_PREPARATION_POSITION_ABS_TOLERANCE_M = 1e-4
INPUT_PREPARATION_FRAME_TIME_TOLERANCE_S = 1e-3

_ORIENTATION_RESULT_KEYS = (
    "orientation_tracking_enabled",
    "orientation_diagnostics_enabled",
    "orientation_human_joint_names",
    "orientation_robot_link_names",
    "orientation_weights",
    "orientation_alignment_mode",
    "orientation_alignment_quaternions_wxyz",
    "orientation_reference_human_quaternions_wxyz",
    "orientation_reference_robot_quaternions_wxyz",
    "orientation_reference_robot_qpos",
    "orientation_target_quaternions_wxyz",
    "orientation_robot_quaternions_wxyz",
    "orientation_errors_rad",
    "orientation_frame_costs",
)


@dataclass(frozen=True)
class Config:
    """Configuration for the complete baseline/orientation comparison batch."""

    data_path: Path = DEFAULT_DATA_PATH
    """Directory containing all Noetix input NPZ files."""

    bvh_path: Path = DEFAULT_BVH_PATH
    """Matching raw BVH directory used to restore missing source rotations."""

    output_root: Path = DEFAULT_OUTPUT_ROOT
    """Unified result root containing versioned ablation artifacts."""

    max_workers: int = 8
    """Maximum number of independent motion sequences solved in parallel."""

    orientation_weight: float = 0.085
    """Uniform weight applied to all 15 tracked link orientations."""

    overwrite_optimal: bool = False
    """Recompute valid balanced_optimal result files."""

    overwrite_baseline: bool = False
    """Rebuild valid diagnostic-enriched baseline result files."""

    dry_run: bool = False
    """Audit and write a planned batch report without running optimization."""

    fail_fast: bool = False
    """Stop submission after the first failed sequence."""


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
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
        os.replace(temporary_path, path)  # noqa: PTH105
        _fsync_directory(path.parent)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _write_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
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
        os.replace(temporary_path, path)  # noqa: PTH105
        _fsync_directory(path.parent)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


@contextlib.contextmanager
def _exclusive_preparation_lock(
    prepared_root: Path,
    preparation_identity: str,
) -> Iterator[None]:
    """Serialize all cache work for one content-addressed preparation."""

    if len(preparation_identity) != 64 or any(
        character not in "0123456789abcdef" for character in preparation_identity
    ):
        raise ValueError(f"Invalid preparation identity: {preparation_identity!r}")
    lock_root = prepared_root / ".locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    lock_path = lock_root / f"{preparation_identity}.lock"
    flags = os.O_CREAT | os.O_RDWR | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    lock_fd = os.open(lock_path, flags, 0o600)
    try:
        _fsync_directory(lock_root)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_preparation_sources(
    *,
    source_path: Path,
    source_sha256: str,
    bvh_path: Path,
    bvh_sha256: str,
) -> None:
    if _sha256(source_path) != source_sha256:
        raise RuntimeError(f"Source input changed while preparing orientations: {source_path}")
    if bvh_sha256 and _sha256(bvh_path) != bvh_sha256:
        raise RuntimeError(f"Source BVH changed while preparing orientations: {bvh_path}")


def _update_fingerprint(digest: Any, label: str, value: str) -> None:
    for component in (label, value):
        encoded = component.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)


def _orientation_preparation_implementation_sha256() -> str:
    """Fingerprint code and runtimes that determine prepared orientation inputs."""

    module_paths = {
        "comparison_batch": Path(__file__).resolve(),
        "convert_noetix_bvh": Path(convert_noetix_bvh.__file__).resolve(),
        "data_type": Path(data_type_module.__file__).resolve(),
    }
    lafan_root = Path(convert_noetix_bvh.__file__).resolve().parent / "lafan1"
    for source_path in sorted(lafan_root.rglob("*.py")):
        module_paths[f"lafan1/{source_path.relative_to(lafan_root).as_posix()}"] = source_path
    digest = hashlib.sha256()
    for label, source_path in sorted(module_paths.items()):
        _update_fingerprint(digest, label, _sha256(source_path))
    for distribution in ("numpy", "scipy"):
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            version = "not-installed"
        _update_fingerprint(digest, distribution, version)
    return digest.hexdigest()


def _preparation_identity(
    *,
    source_sha256: str,
    bvh_sha256: str,
    implementation_sha256: str,
    orientation_source: str,
) -> str:
    digest = hashlib.sha256()
    for label, value in (
        ("schema_version", str(INPUT_PREPARATION_SCHEMA_VERSION)),
        ("source_sha256", source_sha256),
        ("bvh_sha256", bvh_sha256),
        ("implementation_sha256", implementation_sha256),
        ("orientation_source", orientation_source),
        ("target_fps", f"{INPUT_PREPARATION_TARGET_FPS:.17g}"),
        (
            "drop_jump_threshold_m",
            f"{INPUT_PREPARATION_DROP_JUMP_THRESHOLD_M:.17g}",
        ),
        (
            "position_abs_tolerance_m",
            f"{INPUT_PREPARATION_POSITION_ABS_TOLERANCE_M:.17g}",
        ),
        (
            "frame_time_tolerance_s",
            f"{INPUT_PREPARATION_FRAME_TIME_TOLERANCE_S:.17g}",
        ),
    ):
        _update_fingerprint(digest, label, value)
    return digest.hexdigest()


def _prepared_input_matches(
    path: Path,
    *,
    expected_payload: dict[str, np.ndarray],
) -> bool:
    """Accept only a cache equal to the freshly source-derived payload."""

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


def _input_frame_count(path: Path) -> int:
    with np.load(path, allow_pickle=False) as data:
        if "global_joint_positions" not in data.files:
            raise KeyError(f"{path} is missing global_joint_positions")
        positions = np.asarray(data["global_joint_positions"])
    if positions.ndim != 3 or positions.shape[-1] != 3 or positions.shape[0] == 0:
        raise ValueError(f"{path} has invalid global_joint_positions shape {positions.shape}")
    if not np.isfinite(positions).all():
        raise ValueError(f"{path} contains non-finite global joint positions")
    return int(positions.shape[0])


def _has_valid_global_orientations(path: Path, expected_frames: int) -> bool:
    """Accept only the canonical direct-orientation group read by production."""

    required_fields = {
        "orientation_joint_names",
        "orientation_quaternions_wxyz",
        "orientation_source",
    }
    try:
        with np.load(path, allow_pickle=False) as data:
            if not required_fields.issubset(data.files):
                return False
            source_array = np.asarray(data["orientation_source"])
            names_array = np.asarray(data["orientation_joint_names"])
            if source_array.ndim != 0 or names_array.ndim != 1:
                return False
            orientation_source = str(source_array.item()).strip()
            if orientation_source not in APPROVED_DIRECT_ORIENTATION_SOURCES["noetix_mocap"]:
                return False
            names = tuple(str(name) for name in names_array.tolist())
            canonical_names = set(DEMO_JOINTS_REGISTRY["noetix_mocap"])
            if (
                not names
                or len(set(names)) != len(names)
                or not set(names).issubset(canonical_names)
                or not set(ORIENTATION_JOINTS).issubset(names)
            ):
                return False
            quaternions = np.asarray(
                data["orientation_quaternions_wxyz"],
                dtype=np.float64,
            )
    except (KeyError, OSError, TypeError, ValueError):
        return False
    expected_shape = (expected_frames, len(names), 4)
    if quaternions.shape != expected_shape or not np.isfinite(quaternions).all():
        return False
    norms = np.linalg.norm(quaternions, axis=-1)
    return bool(np.all(norms > 1e-8))


def _required_joint_names(
    data: np.lib.npyio.NpzFile,
    *,
    path: Path,
) -> tuple[str, ...]:
    if "joint_names" not in data.files:
        raise KeyError(f"{path} is missing joint_names required for safe BVH orientation merging")
    names_array = np.asarray(data["joint_names"])
    if names_array.ndim != 1:
        raise ValueError(f"{path} joint_names must be one-dimensional, got {names_array.shape}")
    names = tuple(str(name) for name in names_array.tolist())
    if len(set(names)) != len(names):
        raise ValueError(f"{path} joint_names must be unique")
    canonical_names = tuple(DEMO_JOINTS_REGISTRY["noetix_mocap"])
    if names != canonical_names:
        raise ValueError(
            f"{path} joint_names/order do not match the canonical Noetix skeleton: {names} versus {canonical_names}"
        )
    return names


def _required_fps(data: np.lib.npyio.NpzFile, *, path: Path) -> float:
    if "fps" not in data.files:
        raise KeyError(f"{path} is missing fps required for safe BVH orientation merging")
    fps_array = np.asarray(data["fps"])
    if fps_array.ndim != 0:
        raise ValueError(f"{path} fps must be scalar, got {fps_array.shape}")
    fps = float(fps_array.item())
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"{path} fps must be finite and positive, got {fps}")
    return fps


def _validate_bvh_merge_compatibility(
    *,
    source_path: Path,
    bvh_path: Path,
    source_data: np.lib.npyio.NpzFile,
    converted_data: np.lib.npyio.NpzFile,
) -> tuple[float, float]:
    """Prove that converted rotations describe the exact source trajectory."""

    source_positions = np.asarray(
        source_data["global_joint_positions"],
        dtype=np.float64,
    )
    converted_positions = np.asarray(
        converted_data["global_joint_positions"],
        dtype=np.float64,
    )
    if converted_positions.shape != source_positions.shape:
        raise ValueError(
            f"{source_path} and {bvh_path} position shapes differ: "
            f"{source_positions.shape} versus {converted_positions.shape}"
        )
    if not np.isfinite(source_positions).all() or not np.isfinite(converted_positions).all():
        raise ValueError(f"{source_path} or {bvh_path} contains non-finite positions")
    source_names = _required_joint_names(source_data, path=source_path)
    converted_names = _required_joint_names(converted_data, path=bvh_path)
    if source_names != converted_names:
        raise ValueError(
            f"{source_path} and {bvh_path} joint_names/order differ: {source_names} versus {converted_names}"
        )

    source_fps = _required_fps(source_data, path=source_path)
    converted_fps = _required_fps(converted_data, path=bvh_path)
    frame_indices = np.arange(source_positions.shape[0], dtype=np.float64)
    frame_time_difference = np.abs(frame_indices / source_fps - frame_indices / converted_fps)
    frame_time_mean_s = float(np.mean(frame_time_difference))
    frame_time_max_s = float(np.max(frame_time_difference))
    if frame_time_max_s > INPUT_PREPARATION_FRAME_TIME_TOLERANCE_S:
        raise ValueError(
            f"{source_path} and {bvh_path} frame times differ: "
            f"mean={frame_time_mean_s:.9g}s, max={frame_time_max_s:.9g}s, "
            f"hard tolerance={INPUT_PREPARATION_FRAME_TIME_TOLERANCE_S:.9g}s "
            f"(fps {source_fps:.9g} versus {converted_fps:.9g})"
        )

    position_difference = np.abs(converted_positions - source_positions)
    position_mean_difference_m = float(np.mean(position_difference))
    position_max_difference_m = float(np.max(position_difference))
    if position_max_difference_m > INPUT_PREPARATION_POSITION_ABS_TOLERANCE_M:
        raise ValueError(
            f"{source_path} and {bvh_path} positions differ elementwise: "
            f"mean={position_mean_difference_m:.9g}m, "
            f"max={position_max_difference_m:.9g}m, "
            f"hard tolerance={INPUT_PREPARATION_POSITION_ABS_TOLERANCE_M:.9g}m"
        )
    return position_mean_difference_m, position_max_difference_m


_BVH_DERIVED_ORIENTATION_FIELDS = (
    "orientation_joint_names",
    "orientation_quaternions_wxyz",
    "orientation_source",
    "orientation_provenance",
    "quaternion_convention",
    "orientation_coordinate_transform",
)


def _build_expected_prepared_payload(
    *,
    source_path: Path,
    bvh_path: Path,
    converted_root: Path,
    expected_frames: int,
    source_sha256: str,
    bvh_sha256: str,
    implementation_sha256: str,
    preparation_identity: str,
    orientation_source: str,
) -> tuple[dict[str, np.ndarray], float, float]:
    """Derive every prepared field again from the current formal inputs."""

    with np.load(source_path, allow_pickle=False) as source_data:
        source_payload = {key: np.asarray(source_data[key]) for key in source_data.files}

    removed_source_fields: list[str] = []
    derived_fields: list[str] = []
    if orientation_source == "source_npz":
        payload = dict(source_payload)
        position_mean_difference_m = 0.0
        position_max_difference_m = 0.0
    elif orientation_source == "converted_bvh":
        converted_dir = converted_root / preparation_identity
        converted_path = converted_dir / source_path.name
        convert_file(
            bvh_path=bvh_path,
            output_dir=converted_dir,
            target_fps=INPUT_PREPARATION_TARGET_FPS,
            drop_jump_threshold_m=INPUT_PREPARATION_DROP_JUMP_THRESHOLD_M,
            overwrite=True,
        )
        if not _has_valid_global_orientations(
            converted_path,
            expected_frames,
        ):
            raise ValueError(
                "BVH conversion did not produce the canonical "
                "direct-orientation group covering "
                f"{ORIENTATION_JOINTS}: {converted_path}"
            )
        with contextlib.ExitStack() as stack:
            source_data = stack.enter_context(np.load(source_path, allow_pickle=False))
            converted_data = stack.enter_context(np.load(converted_path, allow_pickle=False))
            payload = {key: np.asarray(source_data[key]) for key in source_data.files}
            if "global_joint_quaternions_wxyz" in payload:
                payload.pop("global_joint_quaternions_wxyz")
                removed_source_fields.append("global_joint_quaternions_wxyz")
            (
                position_mean_difference_m,
                position_max_difference_m,
            ) = _validate_bvh_merge_compatibility(
                source_path=source_path,
                bvh_path=bvh_path,
                source_data=source_data,
                converted_data=converted_data,
            )
            for key in _BVH_DERIVED_ORIENTATION_FIELDS:
                if key in converted_data.files:
                    payload[key] = np.asarray(converted_data[key])
                    derived_fields.append(key)
        payload["orientation_source_bvh"] = np.asarray(str(bvh_path))
        derived_fields.append("orientation_source_bvh")
    else:
        raise ValueError(f"Unknown orientation preparation mode: {orientation_source!r}")

    payload["orientation_positions_preserved_from"] = np.asarray(str(source_path))
    lineage = {
        "schema_version": INPUT_PREPARATION_SCHEMA_VERSION,
        "source_path": str(source_path),
        "source_sha256": source_sha256,
        "source_bvh": (str(bvh_path) if orientation_source == "converted_bvh" else ""),
        "source_bvh_sha256": bvh_sha256,
        "implementation_sha256": implementation_sha256,
        "identity_sha256": preparation_identity,
        "orientation_preparation_mode": orientation_source,
        "target_fps": INPUT_PREPARATION_TARGET_FPS,
        "drop_jump_threshold_m": (INPUT_PREPARATION_DROP_JUMP_THRESHOLD_M),
        "position_abs_tolerance_m": (INPUT_PREPARATION_POSITION_ABS_TOLERANCE_M),
        "frame_time_tolerance_s": (INPUT_PREPARATION_FRAME_TIME_TOLERANCE_S),
        "preserved_source_fields": sorted(set(source_payload).difference(removed_source_fields)),
        "removed_source_fields": sorted(removed_source_fields),
        "derived_fields": sorted(derived_fields),
    }
    lineage_json = json.dumps(
        lineage,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    payload.update(
        {
            "preparation_schema_version": np.asarray(
                INPUT_PREPARATION_SCHEMA_VERSION,
                dtype=np.int64,
            ),
            "preparation_source_path": np.asarray(str(source_path)),
            "preparation_source_sha256": np.asarray(source_sha256),
            "preparation_bvh_path": np.asarray(str(bvh_path) if orientation_source == "converted_bvh" else ""),
            "preparation_bvh_sha256": np.asarray(bvh_sha256),
            "preparation_implementation_sha256": np.asarray(implementation_sha256),
            "preparation_identity_sha256": np.asarray(preparation_identity),
            "preparation_orientation_source": np.asarray(orientation_source),
            "preparation_target_fps": np.asarray(
                INPUT_PREPARATION_TARGET_FPS,
                dtype=np.float64,
            ),
            "preparation_drop_jump_threshold_m": np.asarray(
                INPUT_PREPARATION_DROP_JUMP_THRESHOLD_M,
                dtype=np.float64,
            ),
            "preparation_position_abs_tolerance_m": np.asarray(
                INPUT_PREPARATION_POSITION_ABS_TOLERANCE_M,
                dtype=np.float64,
            ),
            "preparation_frame_time_tolerance_s": np.asarray(
                INPUT_PREPARATION_FRAME_TIME_TOLERANCE_S,
                dtype=np.float64,
            ),
            "preparation_lineage_json": np.asarray(lineage_json),
            "preparation_lineage_sha256": np.asarray(hashlib.sha256(lineage_json.encode("utf-8")).hexdigest()),
        }
    )
    payload["preparation_payload_sha256"] = np.asarray(preparation_payload_sha256(payload))
    return (
        payload,
        position_mean_difference_m,
        position_max_difference_m,
    )


def _prepare_orientation_inputs(
    *,
    source_paths: list[Path],
    bvh_root: Path,
    output_root: Path,
) -> tuple[list[Path], list[dict[str, Any]]]:
    """Preserve source positions and restore missing rotations from matching BVHs."""

    prepared_root = output_root / "_inputs"
    converted_root = output_root / "_converted_bvh"
    prepared_root.mkdir(parents=True, exist_ok=True)
    converted_root.mkdir(parents=True, exist_ok=True)
    prepared_paths: list[Path] = []
    records: list[dict[str, Any]] = []
    implementation_sha256 = _orientation_preparation_implementation_sha256()
    for raw_source_path in source_paths:
        source_path = raw_source_path.resolve(strict=True)
        source_sha256 = _sha256(source_path)
        expected_frames = _input_frame_count(source_path)
        bvh_path = (bvh_root / f"{source_path.stem}.bvh").resolve()
        source_has_orientations = _has_valid_global_orientations(
            source_path,
            expected_frames,
        )
        orientation_source = "source_npz" if source_has_orientations else "converted_bvh"
        if source_has_orientations:
            bvh_sha256 = ""
        else:
            if not bvh_path.is_file():
                raise FileNotFoundError(f"Source NPZ lacks orientations and matching BVH is absent: {bvh_path}")
            bvh_sha256 = _sha256(bvh_path)
        preparation_identity = _preparation_identity(
            source_sha256=source_sha256,
            bvh_sha256=bvh_sha256,
            implementation_sha256=implementation_sha256,
            orientation_source=orientation_source,
        )
        preparation_root = prepared_root / preparation_identity
        prepared_path = preparation_root / source_path.name
        record: dict[str, Any] = {
            "task_name": source_path.stem,
            "source_path": str(source_path),
            "prepared_path": str(prepared_path),
            "source_bvh": str(bvh_path),
            "source_sha256": source_sha256,
            "source_bvh_sha256": bvh_sha256,
            "preparation_implementation_sha256": implementation_sha256,
            "preparation_identity_sha256": preparation_identity,
            "orientation_source": orientation_source,
        }
        with _exclusive_preparation_lock(
            prepared_root,
            preparation_identity,
        ):
            _verify_preparation_sources(
                source_path=source_path,
                source_sha256=source_sha256,
                bvh_path=bvh_path,
                bvh_sha256=bvh_sha256,
            )
            (
                expected_payload,
                position_mean_difference_m,
                position_max_difference_m,
            ) = _build_expected_prepared_payload(
                source_path=source_path,
                bvh_path=bvh_path,
                converted_root=converted_root,
                expected_frames=expected_frames,
                source_sha256=source_sha256,
                bvh_sha256=bvh_sha256,
                implementation_sha256=implementation_sha256,
                preparation_identity=preparation_identity,
                orientation_source=orientation_source,
            )
            _verify_preparation_sources(
                source_path=source_path,
                source_sha256=source_sha256,
                bvh_path=bvh_path,
                bvh_sha256=bvh_sha256,
            )
            if _prepared_input_matches(
                prepared_path,
                expected_payload=expected_payload,
            ):
                record["status"] = "reused"
            else:
                _write_npz(prepared_path, expected_payload)
                if not _prepared_input_matches(
                    prepared_path,
                    expected_payload=expected_payload,
                ):
                    raise ValueError(f"Failed to prepare an exact source-derived orientation input: {prepared_path}")
                record["status"] = "prepared"
            _verify_preparation_sources(
                source_path=source_path,
                source_sha256=source_sha256,
                bvh_path=bvh_path,
                bvh_sha256=bvh_sha256,
            )
            record.update(
                {
                    "prepared_payload_sha256": str(np.asarray(expected_payload["preparation_payload_sha256"]).item()),
                    "converted_position_mean_difference_m": (position_mean_difference_m),
                    "converted_position_max_difference_m": (position_max_difference_m),
                }
            )
        prepared_paths.append(prepared_path)
        records.append(record)
    return prepared_paths, records


def _validate_current_result(
    path: Path,
    *,
    expected_frames: int,
    expected_weight: float,
    tracking_enabled: bool,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        required = (
            "qpos",
            "human_joints",
            "mapped_human_joints",
            "mapped_robot_joints",
            *_ORIENTATION_RESULT_KEYS,
        )
        missing = tuple(key for key in required if key not in data.files)
        if missing:
            raise KeyError(f"{path} is missing current result fields {missing}")
        qpos = np.asarray(data["qpos"])
        names = tuple(str(name) for name in np.asarray(data["orientation_human_joint_names"]).tolist())
        weights = np.asarray(data["orientation_weights"], dtype=float)
        target_quaternions = np.asarray(
            data["orientation_target_quaternions_wxyz"],
            dtype=float,
        )
        robot_quaternions = np.asarray(
            data["orientation_robot_quaternions_wxyz"],
            dtype=float,
        )
        errors = np.asarray(data["orientation_errors_rad"], dtype=float)
        actual_tracking = bool(np.asarray(data["orientation_tracking_enabled"]).item())
        diagnostics_enabled = bool(np.asarray(data["orientation_diagnostics_enabled"]).item())
        alignment_mode = str(np.asarray(data["orientation_alignment_mode"]).item())
    if qpos.shape[0] != expected_frames:
        raise ValueError(f"{path} has {qpos.shape[0]} frames; expected {expected_frames}")
    if names != ORIENTATION_JOINTS:
        raise ValueError(f"{path} has orientation joints {names}; expected {ORIENTATION_JOINTS}")
    if weights.shape != (len(ORIENTATION_JOINTS),) or not np.allclose(
        weights,
        expected_weight,
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError(f"{path} has orientation weights {weights}; expected {expected_weight}")
    expected_quaternion_shape = (
        expected_frames,
        len(ORIENTATION_JOINTS),
        4,
    )
    if target_quaternions.shape != expected_quaternion_shape:
        raise ValueError(
            f"{path} target orientation shape {target_quaternions.shape}; expected {expected_quaternion_shape}"
        )
    if robot_quaternions.shape != expected_quaternion_shape:
        raise ValueError(
            f"{path} robot orientation shape {robot_quaternions.shape}; expected {expected_quaternion_shape}"
        )
    if errors.shape != (expected_frames, len(ORIENTATION_JOINTS)):
        raise ValueError(f"{path} orientation error shape {errors.shape} is invalid")
    arrays = (qpos, weights, target_quaternions, robot_quaternions, errors)
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError(f"{path} contains non-finite result values")
    if actual_tracking != tracking_enabled:
        raise ValueError(f"{path} orientation_tracking_enabled={actual_tracking}; expected {tracking_enabled}")
    if not diagnostics_enabled:
        raise ValueError(f"{path} has orientation diagnostics disabled")
    if alignment_mode != "t_pose":
        raise ValueError(f"{path} uses orientation alignment {alignment_mode!r}")


def _result_metrics(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        qpos = np.asarray(data["qpos"], dtype=np.float64)
        position_errors = np.linalg.norm(
            np.asarray(data["mapped_robot_joints"], dtype=np.float64)
            - np.asarray(data["mapped_human_joints"], dtype=np.float64),
            axis=-1,
        )
        orientation_errors = np.asarray(
            data["orientation_errors_rad"],
            dtype=np.float64,
        )
        joint_second_differences = np.linalg.norm(
            np.diff(qpos[:, 7:], n=2, axis=0),
            axis=1,
        )
    return {
        "frames": int(qpos.shape[0]),
        "all_finite": bool(np.isfinite(qpos).all()),
        "position_mean_m": float(np.mean(position_errors)),
        "position_p95_m": float(np.percentile(position_errors, 95)),
        "orientation_mean_rad": float(np.mean(orientation_errors)),
        "orientation_mean_deg": float(np.degrees(np.mean(orientation_errors))),
        "orientation_p95_rad": float(np.percentile(orientation_errors, 95)),
        "orientation_p95_deg": float(np.degrees(np.percentile(orientation_errors, 95))),
        "joint_second_difference_mean_rad": (
            float(np.mean(joint_second_differences)) if joint_second_differences.size else 0.0
        ),
    }


def _comparison_metrics(
    baseline: dict[str, Any],
    optimal: dict[str, Any],
) -> dict[str, float]:
    comparisons: dict[str, float] = {}
    for metric_name in (
        "position_mean_m",
        "position_p95_m",
        "orientation_mean_rad",
        "orientation_p95_rad",
        "joint_second_difference_mean_rad",
    ):
        if metric_name not in baseline or metric_name not in optimal:
            continue
        baseline_value = float(baseline[metric_name])
        optimal_value = float(optimal[metric_name])
        comparisons[f"{metric_name}_delta"] = optimal_value - baseline_value
        comparisons[f"{metric_name}_change_percent"] = (
            100.0 * (optimal_value / baseline_value - 1.0) if baseline_value != 0.0 else float("nan")
        )
    return comparisons


def _build_comparison_job(
    *,
    input_path: Path,
    results_root: Path,
    dataset_partition: str,
    sequence_key: str,
    profile_name: str,
    orientation_weights: dict[str, float],
    overwrite_existing: bool,
) -> RetargetJob:
    config = _retargeting_config(
        input_dir=input_path.parent,
        output_dir=results_root,
        task_name=input_path.stem,
        orientation_weights=orientation_weights,
    )
    return build_retarget_job(
        config,
        variant=RetargetVariant(name=profile_name),
        run_kind="ablation",
        experiment_name=EXPERIMENT_NAME,
        results_root=results_root,
        dataset_partition=dataset_partition,
        sequence_key=sequence_key,
        source_path=input_path,
        overwrite_existing=overwrite_existing,
    )


def _run_task(task: dict[str, Any]) -> dict[str, Any]:
    task_name = str(task["task_name"])
    expected_frames = int(task["expected_frames"])
    baseline_job = task["baseline_job"]
    optimal_job = task["optimal_job"]
    if not isinstance(baseline_job, RetargetJob) or not isinstance(
        optimal_job,
        RetargetJob,
    ):
        raise TypeError("Comparison task requires RetargetJob instances")
    input_path = baseline_job.source_path
    baseline_path = baseline_job.output_path
    optimal_path = optimal_job.output_path
    log_path = Path(task["log_path"])
    orientation_weight = float(task["orientation_weight"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    optimal_status = "reused"
    baseline_status = "reused"
    with log_path.open("a", encoding="utf-8") as log_file:  # noqa: SIM117
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            print(f"[comparison-batch] starting {task_name}", flush=True)
            completed_baseline = run_retargeting_job(baseline_job)
            baseline_path = completed_baseline.output_path
            _validate_current_result(
                baseline_path,
                expected_frames=expected_frames,
                expected_weight=0.0,
                tracking_enabled=False,
            )
            baseline_status = "reused" if completed_baseline.resumed else "computed"

            completed_optimal = run_retargeting_job(optimal_job)
            optimal_path = completed_optimal.output_path
            _validate_current_result(
                optimal_path,
                expected_frames=expected_frames,
                expected_weight=orientation_weight,
                tracking_enabled=True,
            )
            optimal_status = "reused" if completed_optimal.resumed else "computed"
            print(f"[comparison-batch] finished {task_name}", flush=True)

    baseline_metrics = _result_metrics(baseline_path)
    optimal_metrics = _result_metrics(optimal_path)
    return {
        "task_name": task_name,
        "input_path": str(input_path),
        "input_sha256": task["input_sha256"],
        "canonical_baseline_path": str(baseline_job.baseline_path),
        "baseline_path": str(baseline_path),
        "balanced_optimal_path": str(optimal_path),
        "baseline_status": baseline_status,
        "balanced_optimal_status": optimal_status,
        "elapsed_seconds": time.monotonic() - started,
        "baseline": baseline_metrics,
        "balanced_optimal": optimal_metrics,
        "comparison": _comparison_metrics(baseline_metrics, optimal_metrics),
        "log_path": str(log_path),
    }


def _aggregate_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_position_errors: list[np.ndarray] = []
    baseline_orientation_errors: list[np.ndarray] = []
    baseline_smoothness: list[np.ndarray] = []
    optimal_position_errors: list[np.ndarray] = []
    optimal_orientation_errors: list[np.ndarray] = []
    optimal_smoothness: list[np.ndarray] = []
    for record in records:
        for (
            group_name,
            position_collection,
            orientation_collection,
            smoothness_collection,
        ) in (
            (
                "baseline_path",
                baseline_position_errors,
                baseline_orientation_errors,
                baseline_smoothness,
            ),
            (
                "balanced_optimal_path",
                optimal_position_errors,
                optimal_orientation_errors,
                optimal_smoothness,
            ),
        ):
            with np.load(Path(record[group_name]), allow_pickle=False) as data:
                position_collection.append(
                    np.linalg.norm(
                        np.asarray(data["mapped_robot_joints"], dtype=np.float64)
                        - np.asarray(data["mapped_human_joints"], dtype=np.float64),
                        axis=-1,
                    ).reshape(-1)
                )
                orientation_collection.append(np.asarray(data["orientation_errors_rad"], dtype=np.float64))
                smoothness_collection.append(
                    np.linalg.norm(
                        np.diff(
                            np.asarray(data["qpos"], dtype=np.float64)[:, 7:],
                            n=2,
                            axis=0,
                        ),
                        axis=1,
                    )
                )

    def summarize(
        position_arrays: list[np.ndarray],
        orientation_arrays: list[np.ndarray],
        smoothness_arrays: list[np.ndarray],
    ) -> dict[str, Any]:
        position = np.concatenate(position_arrays)
        orientation_by_link = np.concatenate(orientation_arrays, axis=0)
        orientation = orientation_by_link.reshape(-1)
        smoothness = np.concatenate(smoothness_arrays)
        return {
            "position_mean_m": float(np.mean(position)),
            "position_p95_m": float(np.percentile(position, 95)),
            "orientation_mean_rad": float(np.mean(orientation)),
            "orientation_mean_deg": float(np.degrees(np.mean(orientation))),
            "orientation_p95_rad": float(np.percentile(orientation, 95)),
            "orientation_p95_deg": float(np.degrees(np.percentile(orientation, 95))),
            "orientation_per_link": {
                joint_name: {
                    "mean_rad": float(np.mean(orientation_by_link[:, joint_index])),
                    "mean_deg": float(np.degrees(np.mean(orientation_by_link[:, joint_index]))),
                    "p95_rad": float(np.percentile(orientation_by_link[:, joint_index], 95)),
                    "p95_deg": float(
                        np.degrees(
                            np.percentile(
                                orientation_by_link[:, joint_index],
                                95,
                            )
                        )
                    ),
                }
                for joint_index, joint_name in enumerate(ORIENTATION_JOINTS)
            },
            "joint_second_difference_mean_rad": float(np.mean(smoothness)),
            "joint_second_difference_p95_rad": float(np.percentile(smoothness, 95)),
        }

    baseline = summarize(
        baseline_position_errors,
        baseline_orientation_errors,
        baseline_smoothness,
    )
    optimal = summarize(
        optimal_position_errors,
        optimal_orientation_errors,
        optimal_smoothness,
    )
    return {
        "baseline": baseline,
        "balanced_optimal": optimal,
        "comparison": _comparison_metrics(baseline, optimal),
    }


def _write_visualization_assets(
    output_root: Path,
    records: list[dict[str, Any]],
) -> None:
    task_names_path = output_root / "task_names.txt"
    task_names_path.write_text(
        "".join(f"{record['task_name']}\n" for record in records),
        encoding="utf-8",
    )
    python_path = PACKAGE_ROOT.parent / ".venv" / "bin" / "python"
    viewer_path = PACKAGE_ROOT / "multi_viser_player.py"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
    ]
    for record in records:
        command = (
            f"{shlex.quote(str(python_path))} {shlex.quote(str(viewer_path))} "
            f"--qpos-npzs {shlex.quote(record['baseline_path'])} "
            f"{shlex.quote(record['balanced_optimal_path'])} "
            "--labels baseline balanced_optimal --x-offset 0.65 --loop"
        )
        lines.extend(
            (
                f"# {record['task_name']}",
                command,
                "",
            )
        )
    commands_path = output_root / "visualize_commands.sh"
    commands_path.write_text("\n".join(lines), encoding="utf-8")
    commands_path.chmod(0o755)


def main(config: Config) -> None:
    for environment_name in NUMERICAL_RUNTIME_ENVIRONMENT:
        os.environ[environment_name] = "1"

    if config.max_workers <= 0:
        raise ValueError("max_workers must be positive")
    if not np.isfinite(config.orientation_weight) or config.orientation_weight <= 0.0:
        raise ValueError("orientation_weight must be finite and positive")
    expected_profile = orientation_weights_for_variant("balanced_optimal", 1.0)
    if not all(
        np.isclose(value, config.orientation_weight, atol=1e-12, rtol=0.0) for value in expected_profile.values()
    ):
        raise ValueError(
            "orientation_weight differs from the checked-in balanced_optimal profile: "
            f"{config.orientation_weight} versus {sorted(set(expected_profile.values()))}"
        )

    source_paths = sorted(config.data_path.glob("*.npz"))
    if not source_paths:
        raise FileNotFoundError(f"No input NPZ files found in {config.data_path}")
    experiment_root = config.output_root / "ablations" / EXPERIMENT_NAME
    experiment_root.mkdir(parents=True, exist_ok=True)
    input_paths, input_preparation = _prepare_orientation_inputs(
        source_paths=source_paths,
        bvh_root=config.bvh_path,
        output_root=experiment_root,
    )
    tasks: list[dict[str, Any]] = []
    dataset_partition = dataset_partition_identity(config.data_path)
    dataset_path_sha256 = hashlib.sha256(config.data_path.expanduser().resolve().as_posix().encode("utf-8")).hexdigest()
    for input_path in input_paths:
        task_name = input_path.stem
        expected_frames = _input_frame_count(input_path)
        sequence_key = task_name
        log_identity = hashlib.sha256(task_name.encode("utf-8")).hexdigest()
        baseline_job = _build_comparison_job(
            input_path=input_path,
            results_root=config.output_root,
            dataset_partition=dataset_partition,
            sequence_key=sequence_key,
            profile_name="baseline",
            orientation_weights=dict.fromkeys(ORIENTATION_JOINTS, 0.0),
            overwrite_existing=config.overwrite_baseline,
        )
        optimal_job = _build_comparison_job(
            input_path=input_path,
            results_root=config.output_root,
            dataset_partition=dataset_partition,
            sequence_key=sequence_key,
            profile_name="balanced_optimal",
            orientation_weights=dict.fromkeys(
                ORIENTATION_JOINTS,
                config.orientation_weight,
            ),
            overwrite_existing=config.overwrite_optimal,
        )
        if (
            baseline_job.source_path != optimal_job.source_path
            or baseline_job.source_sha256 != optimal_job.source_sha256
        ):
            raise RuntimeError(f"Comparison profiles were not planned from one exact prepared source: {task_name}")
        tasks.append(
            {
                "task_name": task_name,
                "expected_frames": expected_frames,
                "input_path": str(input_path),
                "input_sha256": baseline_job.source_sha256,
                "baseline_path": str(baseline_job.output_path),
                "optimal_path": str(optimal_job.output_path),
                "canonical_baseline_path": str(baseline_job.baseline_path),
                "baseline_config_sha256": baseline_job.config_sha256,
                "optimal_config_sha256": optimal_job.config_sha256,
                "baseline_variant": baseline_job.variant.name,
                "optimal_variant": optimal_job.variant.name,
                "log_path": str(
                    experiment_root / "_logs" / f"dataset-path-sha256-{dataset_path_sha256}" / f"{log_identity}.log"
                ),
                "orientation_weight": config.orientation_weight,
                "overwrite_optimal": config.overwrite_optimal,
                "overwrite_baseline": config.overwrite_baseline,
                "baseline_job": baseline_job,
                "optimal_job": optimal_job,
            }
        )

    planned_tasks = [
        {key: value for key, value in task.items() if key not in {"baseline_job", "optimal_job"}} for task in tasks
    ]
    report_scope = experiment_root / "_summaries" / f"dataset-path-sha256-{dataset_path_sha256}"
    report_path = report_scope / "batch_summary.json"
    report_base: dict[str, Any] = {
        "status": "planned" if config.dry_run else "running",
        "git_commit": _git_commit(),
        "data_path": str(config.data_path),
        "bvh_path": str(config.bvh_path),
        "prepared_data_path": str(experiment_root / "_inputs"),
        "dataset_partition": dataset_partition,
        "input_preparation": input_preparation,
        "output_root": str(config.output_root),
        "experiment_name": EXPERIMENT_NAME,
        "orientation_profile": "balanced_optimal",
        "orientation_weight": config.orientation_weight,
        "orientation_joints": list(ORIENTATION_JOINTS),
        "max_workers": config.max_workers,
        "total_tasks": len(tasks),
        "total_frames": sum(int(task["expected_frames"]) for task in tasks),
        "planned_tasks": planned_tasks,
    }
    _write_json(report_path, report_base)
    if config.dry_run:
        print(
            f"[comparison-batch] dry run: tasks={len(tasks)}, "
            f"frames={report_base['total_frames']}, report={report_path}",
            flush=True,
        )
        return

    print(
        f"[comparison-batch] tasks={len(tasks)}, "
        f"frames={report_base['total_frames']}, "
        f"workers={min(config.max_workers, len(tasks))}, "
        f"weight={config.orientation_weight:g}",
        flush=True,
    )
    started = time.monotonic()
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    cancelled_tasks: list[str] = []
    submitted_tasks: list[str] = []
    first_failure: Exception | None = None
    worker_count = min(config.max_workers, len(tasks))
    context = multiprocessing.get_context("spawn")
    task_iterator = iter(tasks)
    stop_submission = False
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=context,
    ) as executor:
        futures: dict[Any, dict[str, Any]] = {}

        def submit_next() -> bool:
            try:
                next_task = next(task_iterator)
            except StopIteration:
                return False
            future = executor.submit(_run_task, next_task)
            futures[future] = next_task
            submitted_tasks.append(str(next_task["task_name"]))
            return True

        for _ in range(worker_count):
            if not submit_next():
                break

        while futures:
            future = next(as_completed(tuple(futures)))
            task = futures.pop(future)
            task_name = str(task["task_name"])
            if future.cancelled():
                cancelled_tasks.append(task_name)
                continue
            try:
                record = future.result()
                records.append(record)
                print(
                    f"[comparison-batch] {len(records)}/{len(tasks)} completed: "
                    f"{task_name} "
                    f"({record['elapsed_seconds']:.1f}s)",
                    flush=True,
                )
            except Exception as exc:
                failure = {
                    "task_name": task_name,
                    "error": f"{type(exc).__name__}: {exc}",
                    "log_path": str(task["log_path"]),
                }
                failures.append(failure)
                print(
                    f"[comparison-batch] failed: {failure['task_name']}: {failure['error']}",
                    flush=True,
                )
                if first_failure is None:
                    first_failure = exc
                if config.fail_fast and not stop_submission:
                    stop_submission = True
                    for pending_future, pending_task in tuple(futures.items()):
                        if pending_future.cancel():
                            cancelled_tasks.append(str(pending_task["task_name"]))
                            futures.pop(pending_future)
            if not stop_submission:
                submit_next()

    records.sort(key=lambda record: str(record["task_name"]))
    failures.sort(key=lambda failure: failure["task_name"])
    cancelled_tasks = sorted(set(cancelled_tasks))
    submitted_set = set(submitted_tasks)
    not_submitted_tasks = sorted(
        str(task["task_name"]) for task in tasks if str(task["task_name"]) not in submitted_set
    )
    aggregate = _aggregate_metrics(records) if records else {}
    report = {
        **report_base,
        "status": (
            "failed" if failures and config.fail_fast else ("completed_with_failures" if failures else "completed")
        ),
        "elapsed_seconds": time.monotonic() - started,
        "completed_tasks": len(records),
        "failed_tasks": len(failures),
        "submitted_tasks": submitted_tasks,
        "cancelled_tasks": cancelled_tasks,
        "not_submitted_tasks": not_submitted_tasks,
        "records": records,
        "aggregate": aggregate,
        "failures": failures,
    }
    _write_json(report_path, report)
    if records:
        _write_visualization_assets(report_scope, records)
    print(
        f"[comparison-batch] finished: completed={len(records)}, failed={len(failures)}, report={report_path}",
        flush=True,
    )
    if failures:
        error = RuntimeError(f"{len(failures)} of {len(tasks)} tasks failed; see {report_path}")
        if first_failure is not None:
            raise error from first_failure
        raise error


if __name__ == "__main__":
    main(tyro.cli(Config))
