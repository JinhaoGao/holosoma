# ruff: noqa: CPY001

"""Build a full Noetix baseline-versus-orientation comparison dataset."""

from __future__ import annotations

import contextlib
import hashlib
import json
import multiprocessing
import os
import shlex
import subprocess
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import tyro

from holosoma_retargeting.data_utils.convert_noetix_bvh import convert_file
from holosoma_retargeting.examples.robot_retarget import main as run_retargeting
from holosoma_retargeting.examples.run_orientation_ablation import (
    ORIENTATION_JOINTS,
    PACKAGE_ROOT,
    _retargeting_config,
    orientation_weights_for_variant,
)

DEFAULT_DATA_PATH = PACKAGE_ROOT / "demo_data" / "noetix_mocap" / "0724_BEITI"
DEFAULT_BVH_PATH = PACKAGE_ROOT / "demo_data" / "noetix_ori" / "0724_BEITI"
DEFAULT_LEGACY_BASELINE_ROOT = PACKAGE_ROOT / "demo_results" / "e1" / "robot_only" / "noetix" / "0724_BEITI"
DEFAULT_OUTPUT_ROOT = (
    PACKAGE_ROOT / "demo_results_orientation" / "e1" / "robot_only" / "0724_BEITI" / "balanced_optimal_comparison"
)
DEFAULT_ROBOT_XML = PACKAGE_ROOT / "models" / "e1" / "e1_23dof.xml"

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

    legacy_baseline_root: Path = DEFAULT_LEGACY_BASELINE_ROOT
    """Existing zero-orientation result directory to preserve and enrich."""

    output_root: Path = DEFAULT_OUTPUT_ROOT
    """Destination containing baseline and balanced_optimal result groups."""

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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        f"{json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _sha256(path: Path) -> str:
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
    with np.load(path, allow_pickle=False) as data:
        if "global_joint_quaternions_wxyz" not in data.files:
            return False
        quaternions = np.asarray(
            data["global_joint_quaternions_wxyz"],
            dtype=np.float64,
        )
        positions = np.asarray(data["global_joint_positions"])
    expected_shape = (expected_frames, positions.shape[1], 4)
    if quaternions.shape != expected_shape or not np.isfinite(quaternions).all():
        return False
    norms = np.linalg.norm(quaternions, axis=-1)
    return bool(np.all(norms > 1e-8))


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
    for source_path in source_paths:
        expected_frames = _input_frame_count(source_path)
        prepared_path = prepared_root / source_path.name
        bvh_path = bvh_root / f"{source_path.stem}.bvh"
        source_has_orientations = _has_valid_global_orientations(
            source_path,
            expected_frames,
        )
        if (
            prepared_path.exists()
            and _input_frame_count(prepared_path) == expected_frames
            and _has_valid_global_orientations(prepared_path, expected_frames)
        ):
            prepared_paths.append(prepared_path)
            records.append(
                {
                    "task_name": source_path.stem,
                    "source_path": str(source_path),
                    "prepared_path": str(prepared_path),
                    "source_bvh": str(bvh_path),
                    "orientation_source": ("source_npz" if source_has_orientations else "converted_bvh"),
                    "status": "reused",
                }
            )
            continue

        if source_has_orientations:
            with np.load(source_path, allow_pickle=False) as source_data:
                payload = {key: np.asarray(source_data[key]) for key in source_data.files}
            orientation_source = "source_npz"
            position_mean_difference_m = 0.0
            position_max_difference_m = 0.0
        else:
            if not bvh_path.is_file():
                raise FileNotFoundError(f"Source NPZ lacks orientations and matching BVH is absent: {bvh_path}")
            converted_path = converted_root / source_path.name
            convert_file(
                bvh_path=bvh_path,
                output_dir=converted_root,
                target_fps=30.0,
                drop_jump_threshold_m=2.0,
                overwrite=True,
            )
            with np.load(source_path, allow_pickle=False) as source_data:  # noqa: SIM117
                with np.load(converted_path, allow_pickle=False) as converted_data:
                    payload = {key: np.asarray(source_data[key]) for key in source_data.files}
                    converted_positions = np.asarray(
                        converted_data["global_joint_positions"],
                        dtype=np.float64,
                    )
                    source_positions = np.asarray(
                        source_data["global_joint_positions"],
                        dtype=np.float64,
                    )
                    if converted_positions.shape != source_positions.shape:
                        raise ValueError(
                            f"{source_path} and {bvh_path} position shapes differ: "
                            f"{source_positions.shape} versus {converted_positions.shape}"
                        )
                    position_difference = np.linalg.norm(
                        converted_positions - source_positions,
                        axis=-1,
                    )
                    position_mean_difference_m = float(np.mean(position_difference))
                    position_max_difference_m = float(np.max(position_difference))
                    for key in (
                        "global_joint_quaternions_wxyz",
                        "quaternion_convention",
                        "orientation_coordinate_transform",
                    ):
                        payload[key] = np.asarray(converted_data[key])
                    payload["orientation_source_bvh"] = np.asarray(str(bvh_path))
            orientation_source = "converted_bvh"

        payload["orientation_positions_preserved_from"] = np.asarray(str(source_path))
        temporary_path = prepared_path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary_path, **payload)
        temporary_path.replace(prepared_path)
        if not _has_valid_global_orientations(prepared_path, expected_frames):
            raise ValueError(f"Failed to prepare valid global orientations: {prepared_path}")
        prepared_paths.append(prepared_path)
        records.append(
            {
                "task_name": source_path.stem,
                "source_path": str(source_path),
                "prepared_path": str(prepared_path),
                "source_bvh": str(bvh_path),
                "orientation_source": orientation_source,
                "status": "prepared",
                "converted_position_mean_difference_m": position_mean_difference_m,
                "converted_position_max_difference_m": position_max_difference_m,
            }
        )
    return prepared_paths, records


def _validate_legacy_baseline(path: Path, expected_frames: int) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Existing baseline result not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        required = (
            "qpos",
            "human_joints",
            "human_joint_names",
            "mapped_human_joints",
            "mapped_human_joint_names",
            "mapped_robot_joints",
            "mapped_robot_link_names",
            "fps",
        )
        missing = tuple(key for key in required if key not in data.files)
        if missing:
            raise KeyError(f"{path} is missing baseline fields {missing}")
        qpos = np.asarray(data["qpos"])
        human_joints = np.asarray(data["human_joints"])
        mapped_human = np.asarray(data["mapped_human_joints"])
        mapped_robot = np.asarray(data["mapped_robot_joints"])
    frame_arrays = {
        "qpos": qpos,
        "human_joints": human_joints,
        "mapped_human_joints": mapped_human,
        "mapped_robot_joints": mapped_robot,
    }
    for name, array in frame_arrays.items():
        if array.shape[0] != expected_frames:
            raise ValueError(f"{path} {name} has {array.shape[0]} frames; expected {expected_frames}")
        if not np.isfinite(array).all():
            raise ValueError(f"{path} {name} contains non-finite values")


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


def _robot_link_quaternions(
    qpos: np.ndarray,
    link_names: tuple[str, ...],
    robot_xml: Path,
) -> np.ndarray:
    model = mujoco.MjModel.from_xml_path(str(robot_xml))
    data = mujoco.MjData(model)
    if qpos.shape[1] < model.nq:
        raise ValueError(f"Baseline qpos width {qpos.shape[1]} is smaller than robot nq={model.nq}")
    body_ids = np.asarray(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link_name) for link_name in link_names],
        dtype=np.int32,
    )
    missing_links = tuple(link_name for link_name, body_id in zip(link_names, body_ids, strict=True) if body_id < 0)
    if missing_links:
        raise ValueError(f"Robot model is missing orientation links {missing_links}")
    quaternions = np.empty((qpos.shape[0], len(link_names), 4), dtype=np.float32)
    for frame_index, q in enumerate(qpos):
        data.qpos[:] = q[: model.nq]
        mujoco.mj_forward(model, data)
        quaternions[frame_index] = np.asarray(data.xquat[body_ids], dtype=np.float32)
    return quaternions


def _quaternion_geodesic_errors(
    target_quaternions_wxyz: np.ndarray,
    robot_quaternions_wxyz: np.ndarray,
) -> np.ndarray:
    target = np.asarray(target_quaternions_wxyz, dtype=np.float64)
    robot = np.asarray(robot_quaternions_wxyz, dtype=np.float64)
    if target.shape != robot.shape or target.shape[-1] != 4:
        raise ValueError(
            f"Target and robot quaternion shapes must match (..., 4), got {target.shape} and {robot.shape}"
        )
    target /= np.linalg.norm(target, axis=-1, keepdims=True)
    robot /= np.linalg.norm(robot, axis=-1, keepdims=True)
    absolute_dot = np.abs(np.sum(target * robot, axis=-1))
    return (2.0 * np.arccos(np.clip(absolute_dot, 0.0, 1.0))).astype(np.float32)


def enrich_legacy_baseline(
    *,
    legacy_path: Path,
    optimal_path: Path,
    destination_path: Path,
    robot_xml: Path = DEFAULT_ROBOT_XML,
) -> None:
    """Preserve legacy qpos while adding current 15-link orientation diagnostics."""

    with np.load(legacy_path, allow_pickle=False) as baseline_data:
        payload = {
            key: np.asarray(baseline_data[key]) for key in baseline_data.files if key not in _ORIENTATION_RESULT_KEYS
        }
    with np.load(optimal_path, allow_pickle=False) as optimal_data:
        target_quaternions = np.asarray(
            optimal_data["orientation_target_quaternions_wxyz"],
            dtype=np.float32,
        )
        link_names = tuple(str(name) for name in np.asarray(optimal_data["orientation_robot_link_names"]).tolist())
        orientation_metadata = {
            key: np.asarray(optimal_data[key])
            for key in (
                "orientation_human_joint_names",
                "orientation_robot_link_names",
                "orientation_alignment_mode",
                "orientation_alignment_quaternions_wxyz",
                "orientation_reference_human_quaternions_wxyz",
                "orientation_reference_robot_quaternions_wxyz",
                "orientation_reference_robot_qpos",
            )
        }
    qpos = np.asarray(payload["qpos"], dtype=np.float64)
    if qpos.shape[0] != target_quaternions.shape[0]:
        raise ValueError(f"Baseline and optimal frame counts differ: {qpos.shape[0]} and {target_quaternions.shape[0]}")
    robot_quaternions = _robot_link_quaternions(qpos, link_names, robot_xml)
    orientation_errors = _quaternion_geodesic_errors(
        target_quaternions,
        robot_quaternions,
    )
    payload.update(orientation_metadata)
    payload.update(
        {
            "orientation_tracking_enabled": np.asarray(False),
            "orientation_diagnostics_enabled": np.asarray(True),
            "orientation_weights": np.zeros(len(link_names), dtype=np.float64),
            "orientation_target_quaternions_wxyz": target_quaternions,
            "orientation_robot_quaternions_wxyz": robot_quaternions,
            "orientation_errors_rad": orientation_errors,
            "orientation_frame_costs": np.zeros(qpos.shape[0], dtype=np.float64),
        }
    )
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination_path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary_path, **payload)
    temporary_path.replace(destination_path)


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


def _run_task(task: dict[str, Any]) -> dict[str, Any]:
    task_name = str(task["task_name"])
    expected_frames = int(task["expected_frames"])
    input_path = Path(task["input_path"])
    legacy_path = Path(task["legacy_path"])
    baseline_path = Path(task["baseline_path"])
    optimal_path = Path(task["optimal_path"])
    log_path = Path(task["log_path"])
    orientation_weight = float(task["orientation_weight"])
    overwrite_optimal = bool(task["overwrite_optimal"])
    overwrite_baseline = bool(task["overwrite_baseline"])

    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    optimal_status = "reused"
    baseline_status = "reused"
    with log_path.open("a", encoding="utf-8") as log_file:  # noqa: SIM117
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            print(f"[comparison-batch] starting {task_name}", flush=True)
            optimal_valid = False
            if optimal_path.exists() and not overwrite_optimal:
                try:
                    _validate_current_result(
                        optimal_path,
                        expected_frames=expected_frames,
                        expected_weight=orientation_weight,
                        tracking_enabled=True,
                    )
                    optimal_valid = True
                except (KeyError, ValueError):
                    traceback.print_exc()
                    print("[comparison-batch] existing optimal is invalid; recomputing", flush=True)
            if not optimal_valid:
                optimal_path.parent.mkdir(parents=True, exist_ok=True)
                config = _retargeting_config(
                    input_dir=input_path.parent,
                    output_dir=optimal_path.parent,
                    task_name=task_name,
                    orientation_weights=dict.fromkeys(
                        ORIENTATION_JOINTS,
                        orientation_weight,
                    ),
                )
                run_retargeting(config)
                _validate_current_result(
                    optimal_path,
                    expected_frames=expected_frames,
                    expected_weight=orientation_weight,
                    tracking_enabled=True,
                )
                optimal_status = "computed"

            baseline_valid = False
            if baseline_path.exists() and not overwrite_baseline:
                try:
                    _validate_current_result(
                        baseline_path,
                        expected_frames=expected_frames,
                        expected_weight=0.0,
                        tracking_enabled=False,
                    )
                    baseline_valid = True
                except (KeyError, ValueError):
                    traceback.print_exc()
                    print("[comparison-batch] existing baseline is invalid; rebuilding", flush=True)
            if not baseline_valid:
                enrich_legacy_baseline(
                    legacy_path=legacy_path,
                    optimal_path=optimal_path,
                    destination_path=baseline_path,
                )
                _validate_current_result(
                    baseline_path,
                    expected_frames=expected_frames,
                    expected_weight=0.0,
                    tracking_enabled=False,
                )
                baseline_status = "enriched_from_legacy"
            print(f"[comparison-batch] finished {task_name}", flush=True)

    baseline_metrics = _result_metrics(baseline_path)
    optimal_metrics = _result_metrics(optimal_path)
    return {
        "task_name": task_name,
        "input_path": str(input_path),
        "input_sha256": task["input_sha256"],
        "legacy_baseline_path": str(legacy_path),
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
    viewer_path = PACKAGE_ROOT / "examples" / "ablation_viser_player.py"
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
    config.output_root.mkdir(parents=True, exist_ok=True)
    input_paths, input_preparation = _prepare_orientation_inputs(
        source_paths=source_paths,
        bvh_root=config.bvh_path,
        output_root=config.output_root,
    )
    tasks: list[dict[str, Any]] = []
    for input_path in input_paths:
        task_name = input_path.stem
        expected_frames = _input_frame_count(input_path)
        legacy_path = config.legacy_baseline_root / input_path.name
        _validate_legacy_baseline(legacy_path, expected_frames)
        tasks.append(
            {
                "task_name": task_name,
                "expected_frames": expected_frames,
                "input_path": str(input_path),
                "input_sha256": _sha256(input_path),
                "legacy_path": str(legacy_path),
                "baseline_path": str(config.output_root / "baseline" / input_path.name),
                "optimal_path": str(config.output_root / "balanced_optimal" / input_path.name),
                "log_path": str(config.output_root / "logs" / f"{task_name}.log"),
                "orientation_weight": config.orientation_weight,
                "overwrite_optimal": config.overwrite_optimal,
                "overwrite_baseline": config.overwrite_baseline,
            }
        )

    report_path = config.output_root / "batch_summary.json"
    report_base: dict[str, Any] = {
        "status": "planned" if config.dry_run else "running",
        "git_commit": _git_commit(),
        "data_path": str(config.data_path),
        "bvh_path": str(config.bvh_path),
        "prepared_data_path": str(config.output_root / "_inputs"),
        "input_preparation": input_preparation,
        "legacy_baseline_root": str(config.legacy_baseline_root),
        "output_root": str(config.output_root),
        "orientation_profile": "balanced_optimal",
        "orientation_weight": config.orientation_weight,
        "orientation_joints": list(ORIENTATION_JOINTS),
        "max_workers": config.max_workers,
        "total_tasks": len(tasks),
        "total_frames": sum(int(task["expected_frames"]) for task in tasks),
        "planned_tasks": tasks,
    }
    _write_json(report_path, report_base)
    if config.dry_run:
        print(
            f"[comparison-batch] dry run: tasks={len(tasks)}, "
            f"frames={report_base['total_frames']}, report={report_path}",
            flush=True,
        )
        return

    for environment_name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[environment_name] = "1"

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
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(config.max_workers, len(tasks)),
        mp_context=context,
    ) as executor:
        futures = {executor.submit(_run_task, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                record = future.result()
                records.append(record)
                print(
                    f"[comparison-batch] {len(records)}/{len(tasks)} completed: "
                    f"{task['task_name']} "
                    f"({record['elapsed_seconds']:.1f}s)",
                    flush=True,
                )
            except Exception as exc:
                failure = {
                    "task_name": str(task["task_name"]),
                    "error": f"{type(exc).__name__}: {exc}",
                    "log_path": str(task["log_path"]),
                }
                failures.append(failure)
                print(
                    f"[comparison-batch] failed: {failure['task_name']}: {failure['error']}",
                    flush=True,
                )
                if config.fail_fast:
                    for pending_future in futures:
                        pending_future.cancel()
                    raise

    records.sort(key=lambda record: str(record["task_name"]))
    failures.sort(key=lambda failure: failure["task_name"])
    aggregate = _aggregate_metrics(records) if records else {}
    report = {
        **report_base,
        "status": "completed_with_failures" if failures else "completed",
        "elapsed_seconds": time.monotonic() - started,
        "completed_tasks": len(records),
        "failed_tasks": len(failures),
        "records": records,
        "aggregate": aggregate,
        "failures": failures,
    }
    _write_json(report_path, report)
    if records:
        _write_visualization_assets(config.output_root, records)
    print(
        f"[comparison-batch] finished: completed={len(records)}, failed={len(failures)}, report={report_path}",
        flush=True,
    )
    if failures:
        raise RuntimeError(f"{len(failures)} of {len(tasks)} tasks failed; see {report_path}")


if __name__ == "__main__":
    main(tyro.cli(Config))
