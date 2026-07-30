# ruff: noqa: CPY001

"""Parallel search for balanced Noetix E1 position/orientation weights."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import multiprocessing
import os
import re
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tyro

from holosoma_retargeting.examples.run_orientation_ablation import (
    _FRAME_ALIGNED_NPZ_FIELDS,
    _KNOWN_VECTOR_METADATA_FIELDS,
    DEFAULT_TASK_NAME,
    ORIENTATION_JOINTS,
    PACKAGE_ROOT,
    _atomic_write_json,
    _atomic_write_npz,
    _result_summary,
    _retargeting_config,
)
from holosoma_retargeting.retargeting_pipeline import (
    RetargetVariant,
    build_retarget_job,
    run_retargeting_job,
)

ORIENTATION_GROUPS: dict[str, tuple[str, ...]] = {
    "root": ("Hips",),
    "hips": ("LeftUpLeg", "RightUpLeg"),
    "knees": ("LeftLeg", "RightLeg"),
    "ankles": ("LeftFoot", "RightFoot"),
    "toes": ("LeftToeBase", "RightToeBase"),
    "shoulders": ("LeftArm", "RightArm"),
    "forearms": ("LeftForeArm", "RightForeArm"),
    "hands": ("LeftHand", "RightHand"),
}

_RUN_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
EXPERIMENT_NAME = "orientation_weight_search"
WINDOW_INPUT_SCHEMA_VERSION = 2
_WINDOW_METADATA_KEYS = frozenset(
    {
        "_window_source_sha256",
        "_window_frame_start",
        "_window_frame_count",
        "_window_schema_version",
        "_window_implementation_sha256",
    }
)


@dataclass(frozen=True)
class Config:
    candidate_file: Path
    """JSON mapping candidate names to group or per-joint weights."""

    data_path: Path = PACKAGE_ROOT / "demo_data" / "noetix_mocap" / "0724_BEITI"
    task_name: str = DEFAULT_TASK_NAME
    output_root: Path = PACKAGE_ROOT / "demo_results" / "v1"
    frame_starts: tuple[int, ...] = (0, 900, 1800, 2450)
    frame_count: int = 120
    max_workers: int = 8
    overwrite: bool = False
    fail_fast: bool = False


def expand_candidate_weights(specification: dict[str, float]) -> dict[str, float]:
    """Expand all/group/joint keys into the complete 15-link weight table."""

    allowed_keys = {"all", *ORIENTATION_GROUPS, *ORIENTATION_JOINTS}
    unknown = sorted(set(specification) - allowed_keys)
    if unknown:
        raise ValueError(f"Unknown orientation weight keys: {unknown}")

    numeric_specification = {name: float(value) for name, value in specification.items()}
    invalid = {name: value for name, value in numeric_specification.items() if not np.isfinite(value) or value < 0.0}
    if invalid:
        raise ValueError(f"Orientation weights must be finite and non-negative: {invalid}")

    weights = dict.fromkeys(
        ORIENTATION_JOINTS,
        numeric_specification.get("all", 0.0),
    )
    for group_name, joint_names in ORIENTATION_GROUPS.items():
        if group_name not in numeric_specification:
            continue
        for joint_name in joint_names:
            weights[joint_name] = numeric_specification[group_name]
    for joint_name in ORIENTATION_JOINTS:
        if joint_name in numeric_specification:
            weights[joint_name] = numeric_specification[joint_name]
    return {name: float(value) for name, value in weights.items()}


def _load_candidates(path: Path) -> dict[str, dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError("Candidate JSON must be a non-empty object")

    candidates: dict[str, dict[str, float]] = {}
    for run_name, specification in payload.items():
        if not isinstance(run_name, str) or not _RUN_NAME_PATTERN.fullmatch(run_name):
            raise ValueError(f"Invalid candidate name: {run_name!r}")
        if not isinstance(specification, dict):
            raise ValueError(f"Candidate {run_name!r} must map weight keys to numbers")
        candidates[run_name] = expand_candidate_weights(specification)
    if "baseline" not in candidates:
        raise ValueError("Candidate JSON must include a zero-weight 'baseline'")
    if any(candidates["baseline"].values()):
        raise ValueError("The baseline candidate must have all orientation weights equal to zero")
    return candidates


def _source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PACKAGE_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _window_payload(
    payload: dict[str, np.ndarray],
    *,
    source_frames: int,
    frame_start: int,
    frame_end: int,
) -> dict[str, np.ndarray]:
    """Slice only explicitly classified frame-aligned fields."""

    window: dict[str, np.ndarray] = {}
    for key, value in payload.items():
        if key in _FRAME_ALIGNED_NPZ_FIELDS:
            if value.ndim == 0 or value.shape[0] != source_frames:
                raise ValueError(
                    f"Frame-aligned field {key!r} has shape {value.shape}; expected first dimension {source_frames}",
                )
            window[key] = value[frame_start:frame_end]
        elif value.ndim > 0 and value.shape[0] == source_frames and key not in _KNOWN_VECTOR_METADATA_FIELDS:
            raise ValueError(
                f"Cannot safely create an orientation-search window: unknown "
                f"frame-aligned field {key!r} must be classified explicitly",
            )
        else:
            window[key] = value
    return window


def _cached_window_matches(
    cached: np.lib.npyio.NpzFile,
    *,
    expected_payload: dict[str, np.ndarray],
    source_hash: str,
    frame_start: int,
    frame_count: int,
    implementation_hash: str,
) -> bool:
    """Verify cache identity and every preserved source/orientation byte."""

    if set(cached.files) != set(expected_payload) | _WINDOW_METADATA_KEYS:
        return False
    try:
        metadata_matches = (
            str(np.asarray(cached["_window_source_sha256"]).item()) == source_hash
            and int(np.asarray(cached["_window_frame_start"]).item()) == frame_start
            and int(np.asarray(cached["_window_frame_count"]).item()) == frame_count
            and int(np.asarray(cached["_window_schema_version"]).item()) == WINDOW_INPUT_SCHEMA_VERSION
            and str(np.asarray(cached["_window_implementation_sha256"]).item()) == implementation_hash
        )
    except (KeyError, TypeError, ValueError):
        return False
    return metadata_matches and all(
        np.array_equal(np.asarray(cached[key]), expected) for key, expected in expected_payload.items()
    )


def _prepare_window_inputs(
    *,
    source_path: Path,
    destination_root: Path,
    frame_starts: tuple[int, ...],
    frame_count: int,
    overwrite: bool,
) -> dict[int, Path]:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if len(set(frame_starts)) != len(frame_starts):
        raise ValueError("frame_starts must be unique")

    source_path = source_path.resolve(strict=True)
    source_hash = _source_sha256(source_path)
    implementation_hash = _source_sha256(Path(__file__).resolve())
    with np.load(source_path, allow_pickle=False) as data:
        source_frames = int(np.asarray(data["global_joint_positions"]).shape[0])
        payload = {key: np.asarray(data[key]) for key in data.files}
    orientation_pair = {
        "orientation_joint_names",
        "orientation_quaternions_wxyz",
    }
    if orientation_pair.intersection(payload) and not orientation_pair.issubset(payload):
        missing = sorted(orientation_pair.difference(payload))
        raise ValueError(
            f"Source input has incomplete direct orientation fields: {missing}",
        )
    if _source_sha256(source_path) != source_hash:
        raise RuntimeError(f"Source input changed while preparing search windows: {source_path}")

    inputs: dict[int, Path] = {}
    for frame_start in frame_starts:
        frame_end = frame_start + frame_count
        if frame_start < 0 or frame_end > source_frames:
            raise ValueError(f"Invalid window [{frame_start}, {frame_end}) for source with {source_frames} frames")
        window_dir = (
            destination_root
            / f"source_{source_hash}"
            / f"implementation_{implementation_hash}"
            / f"frames_{frame_start:06d}_{frame_count:06d}"
        )
        window_dir.mkdir(parents=True, exist_ok=True)
        window_path = window_dir / source_path.name
        expected_payload = _window_payload(
            payload,
            source_frames=source_frames,
            frame_start=frame_start,
            frame_end=frame_end,
        )
        cache_matches = False
        if window_path.is_file() and not overwrite:
            try:
                with np.load(window_path, allow_pickle=False) as cached:
                    cache_matches = _cached_window_matches(
                        cached,
                        expected_payload=expected_payload,
                        source_hash=source_hash,
                        frame_start=frame_start,
                        frame_count=frame_count,
                        implementation_hash=implementation_hash,
                    )
            except (KeyError, OSError, TypeError, ValueError):
                cache_matches = False
        if not cache_matches:
            window_payload = dict(expected_payload)
            window_payload.update(
                {
                    "_window_source_sha256": np.asarray(source_hash),
                    "_window_frame_start": np.asarray(frame_start, dtype=np.int64),
                    "_window_frame_count": np.asarray(frame_count, dtype=np.int64),
                    "_window_schema_version": np.asarray(
                        WINDOW_INPUT_SCHEMA_VERSION,
                        dtype=np.int64,
                    ),
                    "_window_implementation_sha256": np.asarray(implementation_hash),
                }
            )
            _atomic_write_npz(window_path, window_payload)
            with np.load(window_path, allow_pickle=False) as cached:
                if not _cached_window_matches(
                    cached,
                    expected_payload=expected_payload,
                    source_hash=source_hash,
                    frame_start=frame_start,
                    frame_count=frame_count,
                    implementation_hash=implementation_hash,
                ):
                    raise ValueError(
                        f"Failed to prepare an exact orientation-search window: {window_path}",
                    )
        inputs[frame_start] = window_path
    return inputs


def _run_candidate_window(
    *,
    run_name: str,
    orientation_weights: dict[str, float],
    frame_start: int,
    frame_count: int,
    input_path: Path,
    output_root: Path,
    task_name: str,
    source_hash: str,
    git_commit: str,
    overwrite: bool,
) -> dict[str, Any]:
    retargeting_config = _retargeting_config(
        input_dir=input_path.parent,
        output_dir=output_root,
        task_name=task_name,
        orientation_weights=orientation_weights,
    )
    job = build_retarget_job(
        retargeting_config,
        variant=RetargetVariant(name=run_name),
        run_kind="ablation",
        experiment_name=EXPERIMENT_NAME,
        results_root=output_root,
        dataset_partition="windowed",
        sequence_key=(f"{task_name}/frames_{frame_start:06d}_{frame_count:06d}/source_{source_hash}"),
        source_path=input_path,
        overwrite_existing=overwrite,
    )
    result_path = job.output_path
    result_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = result_path.parent / "manifest.json"
    log_path = result_path.parent / "run.log"
    manifest: dict[str, Any] = {
        "run_name": run_name,
        "frame_start": frame_start,
        "frame_count": frame_count,
        "orientation_weights": orientation_weights,
        "input_path": str(input_path),
        "source_sha256": source_hash,
        "artifact_source_sha256": job.source_sha256,
        "git_commit": git_commit,
        "run_kind": job.run_kind,
        "experiment_name": job.experiment_name,
        "config_sha256": job.config_sha256,
        "canonical_baseline_path": str(job.baseline_path),
        "result_path": str(result_path),
        "status": "running",
    }
    _atomic_write_json(manifest_path, manifest)

    start_time = time.monotonic()
    try:
        with contextlib.ExitStack() as stack:
            log_file = stack.enter_context(log_path.open("w", encoding="utf-8"))
            stack.enter_context(contextlib.redirect_stdout(log_file))
            stack.enter_context(contextlib.redirect_stderr(log_file))
            stream_handlers = [
                handler for handler in logging.getLogger().handlers if isinstance(handler, logging.StreamHandler)
            ]
            original_streams = [handler.stream for handler in stream_handlers]
            try:
                for handler in stream_handlers:
                    handler.setStream(log_file)
                completed_job = run_retargeting_job(job)
            finally:
                for handler, original_stream in zip(
                    stream_handlers,
                    original_streams,
                    strict=True,
                ):
                    handler.setStream(original_stream)
        elapsed_seconds = time.monotonic() - start_time
        result_path = completed_job.output_path
        manifest["result_path"] = str(result_path)
        manifest["status"] = "reused" if completed_job.resumed else "completed"
        manifest["elapsed_seconds"] = 0.0 if completed_job.resumed else elapsed_seconds
        summary = _result_summary(
            result_path,
            elapsed_seconds=manifest["elapsed_seconds"],
        )
        return {
            "run_name": run_name,
            "frame_start": frame_start,
            "frame_count": frame_count,
            "result_path": str(result_path),
            "canonical_baseline_path": str(job.baseline_path),
            "summary": summary,
        }
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        _atomic_write_json(manifest_path, manifest)


def _candidate_metrics(result_paths: list[Path]) -> dict[str, Any]:
    orientation_errors: list[np.ndarray] = []
    position_errors: list[np.ndarray] = []
    joint_accelerations: list[np.ndarray] = []
    sqp_iterations: list[np.ndarray] = []
    sqp_stop_reasons: list[np.ndarray] = []
    orientation_names: tuple[str, ...] | None = None
    position_names: tuple[str, ...] | None = None
    orientation_weights: np.ndarray | None = None
    release_count = 0
    frame_count = 0

    for result_path in result_paths:
        with np.load(result_path, allow_pickle=False) as data:
            current_orientation_names = tuple(
                str(name) for name in np.asarray(data["orientation_human_joint_names"]).tolist()
            )
            current_position_names = tuple(str(name) for name in np.asarray(data["mapped_human_joint_names"]).tolist())
            current_weights = np.asarray(data["orientation_weights"], dtype=np.float64)
            if orientation_names is None:
                orientation_names = current_orientation_names
                position_names = current_position_names
                orientation_weights = current_weights
            elif (
                current_orientation_names != orientation_names
                or current_position_names != position_names
                or not np.array_equal(current_weights, orientation_weights)
            ):
                raise ValueError(f"Inconsistent result schema for candidate: {result_path}")

            qpos = np.asarray(data["qpos"], dtype=np.float64)
            human_positions = np.asarray(data["mapped_human_joints"], dtype=np.float64)
            robot_positions = np.asarray(data["mapped_robot_joints"], dtype=np.float64)
            orientation_errors.append(np.asarray(data["orientation_errors_rad"], dtype=np.float64))
            position_errors.append(np.linalg.norm(robot_positions - human_positions, axis=-1))
            if qpos.shape[0] >= 3:
                joint_accelerations.append(np.linalg.norm(np.diff(qpos[:, 7:], n=2, axis=0), axis=1))
            sqp_iterations.append(np.asarray(data["sqp_iteration_counts"], dtype=np.float64))
            sqp_stop_reasons.append(np.asarray(data["sqp_stop_reasons"], dtype=str))
            release_count += sum(
                int(np.asarray(data[key]).size)
                for key in (
                    "foot_sticking_release_frames",
                    "object_non_penetration_release_frames",
                )
            )
            frame_count += qpos.shape[0]

    assert orientation_names is not None
    assert position_names is not None
    assert orientation_weights is not None
    orientation_array = np.concatenate(orientation_errors, axis=0)
    position_array = np.concatenate(position_errors, axis=0)
    acceleration_array = (
        np.concatenate(joint_accelerations) if joint_accelerations else np.empty((0,), dtype=np.float64)
    )
    sqp_array = np.concatenate(sqp_iterations)
    stop_reason_array = np.concatenate(sqp_stop_reasons)

    return {
        "frames": frame_count,
        "orientation_joint_names": list(orientation_names),
        "position_joint_names": list(position_names),
        "orientation_weights": orientation_weights.tolist(),
        "orientation_mean_rad": float(np.mean(orientation_array)),
        "orientation_p95_rad": float(np.percentile(orientation_array, 95)),
        "position_mean_m": float(np.mean(position_array)),
        "position_p95_m": float(np.percentile(position_array, 95)),
        "joint_second_difference_mean_rad": (float(np.mean(acceleration_array)) if acceleration_array.size else 0.0),
        "sqp_iterations_mean": float(np.mean(sqp_array)),
        "sqp_iterations_p95": float(np.percentile(sqp_array, 95)),
        "sqp_max_iteration_fraction": float(np.mean(stop_reason_array == "max_iterations")),
        "constraint_release_count": release_count,
        "orientation_per_link_mean_rad": {
            name: float(np.mean(orientation_array[:, index])) for index, name in enumerate(orientation_names)
        },
        "orientation_per_link_p95_rad": {
            name: float(np.percentile(orientation_array[:, index], 95)) for index, name in enumerate(orientation_names)
        },
        "position_per_link_mean_m": {
            name: float(np.mean(position_array[:, index])) for index, name in enumerate(position_names)
        },
        "position_per_link_p95_m": {
            name: float(np.percentile(position_array[:, index], 95)) for index, name in enumerate(position_names)
        },
    }


def balanced_score(
    metrics: dict[str, Any],
    baseline: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    """Return an equal mean/tail position-orientation score plus stability penalty."""

    ratios = {
        "position_mean": metrics["position_mean_m"] / baseline["position_mean_m"],
        "position_p95": metrics["position_p95_m"] / baseline["position_p95_m"],
        "orientation_mean": metrics["orientation_mean_rad"] / baseline["orientation_mean_rad"],
        "orientation_p95": metrics["orientation_p95_rad"] / baseline["orientation_p95_rad"],
        "smoothness": (
            metrics["joint_second_difference_mean_rad"] / baseline["joint_second_difference_mean_rad"]
            if baseline["joint_second_difference_mean_rad"] > 0.0
            else 1.0
        ),
    }
    core_score = (
        0.35 * ratios["position_mean"]
        + 0.15 * ratios["position_p95"]
        + 0.35 * ratios["orientation_mean"]
        + 0.15 * ratios["orientation_p95"]
    )
    stability_penalty = (
        0.10 * max(0.0, ratios["smoothness"] - 1.0)
        + 0.25 * metrics["sqp_max_iteration_fraction"]
        + float(metrics["constraint_release_count"] > 0)
    )
    return float(core_score + stability_penalty), ratios


def pareto_front(candidate_metrics: dict[str, dict[str, Any]]) -> list[str]:
    """Return candidates not dominated across mean/tail tracking and smoothness."""

    metric_names = (
        "position_mean_m",
        "position_p95_m",
        "orientation_mean_rad",
        "orientation_p95_rad",
        "joint_second_difference_mean_rad",
    )
    front: list[str] = []
    for candidate_name, metrics in candidate_metrics.items():
        values = np.asarray([metrics[name] for name in metric_names])
        dominated = False
        for other_name, other_metrics in candidate_metrics.items():
            if other_name == candidate_name:
                continue
            other_values = np.asarray([other_metrics[name] for name in metric_names])
            if np.all(other_values <= values) and np.any(other_values < values):
                dominated = True
                break
        if not dominated:
            front.append(candidate_name)
    return sorted(front)


def main(config: Config) -> None:
    if config.max_workers <= 0:
        raise ValueError("max_workers must be positive")
    source_path = config.data_path / f"{config.task_name}.npz"
    if not source_path.is_file():
        raise FileNotFoundError(f"Input motion not found: {source_path}")

    candidates = _load_candidates(config.candidate_file)
    experiment_root = config.output_root / "ablations" / EXPERIMENT_NAME
    experiment_root.mkdir(parents=True, exist_ok=True)
    window_inputs = _prepare_window_inputs(
        source_path=source_path,
        destination_root=experiment_root / "_inputs",
        frame_starts=config.frame_starts,
        frame_count=config.frame_count,
        overwrite=config.overwrite,
    )
    source_hash = _source_sha256(source_path)
    git_commit = _git_commit()
    task_records: dict[str, list[dict[str, Any]]] = {name: [] for name in candidates}
    failures: dict[str, str] = {}

    for environment_name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[environment_name] = "1"

    tasks = [
        {
            "run_name": run_name,
            "orientation_weights": weights,
            "frame_start": frame_start,
            "frame_count": config.frame_count,
            "input_path": input_path,
            "output_root": config.output_root,
            "task_name": config.task_name,
            "source_hash": source_hash,
            "git_commit": git_commit,
            "overwrite": config.overwrite,
        }
        for run_name, weights in candidates.items()
        for frame_start, input_path in window_inputs.items()
    ]
    print(
        f"[orientation-search] candidates={len(candidates)}, "
        f"windows={len(window_inputs)}, tasks={len(tasks)}, workers={config.max_workers}",
        flush=True,
    )
    completed_count = 0
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(config.max_workers, len(tasks)),
        mp_context=context,
    ) as executor:
        futures = {executor.submit(_run_candidate_window, **task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            task_key = f"{task['run_name']}/window_{task['frame_start']:06d}"
            try:
                record = future.result()
                task_records[task["run_name"]].append(record)
                completed_count += 1
                print(
                    f"[orientation-search] {completed_count}/{len(tasks)} completed: {task_key}",
                    flush=True,
                )
            except Exception as exc:
                failures[task_key] = f"{type(exc).__name__}: {exc}"
                print(
                    f"[orientation-search] failed: {task_key}: {failures[task_key]}",
                    flush=True,
                )
                if config.fail_fast:
                    for pending_future in futures:
                        pending_future.cancel()
                    raise

    metrics: dict[str, dict[str, Any]] = {}
    for run_name, records in task_records.items():
        if len(records) != len(window_inputs):
            continue
        ordered_records = sorted(
            records,
            key=lambda record: int(record["frame_start"]),
        )
        metrics[run_name] = _candidate_metrics([Path(record["result_path"]) for record in ordered_records])

    if "baseline" not in metrics:
        raise RuntimeError("Baseline did not complete; balanced scores cannot be computed")
    baseline = metrics["baseline"]
    scored_candidates: dict[str, Any] = {}
    for run_name, candidate_metrics in metrics.items():
        score, ratios = balanced_score(candidate_metrics, baseline)
        scored_candidates[run_name] = {
            "score": score,
            "normalized_ratios": ratios,
            "weights": candidates[run_name],
            "metrics": candidate_metrics,
        }
    ranking = sorted(
        scored_candidates,
        key=lambda name: scored_candidates[name]["score"],
    )
    payload = {
        "experiment_name": EXPERIMENT_NAME,
        "results_root": str(config.output_root),
        "task_name": config.task_name,
        "source_path": str(source_path),
        "source_sha256": source_hash,
        "git_commit": git_commit,
        "frame_starts": list(config.frame_starts),
        "frame_count": config.frame_count,
        "score_definition": {
            "core": (
                "0.35*position_mean_ratio + 0.15*position_p95_ratio + "
                "0.35*orientation_mean_ratio + 0.15*orientation_p95_ratio"
            ),
            "stability_penalty": (
                "0.10*max(0,smoothness_ratio-1) + 0.25*max_iteration_fraction + any_constraint_release"
            ),
        },
        "ranking": ranking,
        "best_candidate": ranking[0],
        "pareto_front": pareto_front(metrics),
        "candidates": scored_candidates,
        "records": {
            run_name: sorted(
                records,
                key=lambda record: int(record["frame_start"]),
            )
            for run_name, records in task_records.items()
        },
        "failures": failures,
    }
    summary_path = experiment_root / "search_summary.json"
    _atomic_write_json(summary_path, payload)
    print(
        f"[orientation-search] best={ranking[0]}, "
        f"score={scored_candidates[ranking[0]]['score']:.6f}, "
        f"pareto={payload['pareto_front']}, summary={summary_path}",
        flush=True,
    )
    if failures:
        raise RuntimeError(f"Some orientation-search tasks failed: {failures}")


if __name__ == "__main__":
    main(tyro.cli(Config))
