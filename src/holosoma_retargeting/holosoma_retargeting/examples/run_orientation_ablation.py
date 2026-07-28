"""Run reproducible SO(3) tracking ablations for the Noetix E1 experiment."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tyro

from holosoma_retargeting.config_types.data_type import MotionDataConfig
from holosoma_retargeting.config_types.retargeter import RetargeterConfig
from holosoma_retargeting.config_types.retargeting import RetargetingConfig
from holosoma_retargeting.config_types.robot import RobotConfig
from holosoma_retargeting.config_types.task import TaskConfig
from holosoma_retargeting.examples.robot_retarget import main as run_retargeting

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASK_NAME = "breaking+hippop.bvh_Skeleton1"

ORIENTATION_JOINTS = (
    "Hips",
    "LeftUpLeg",
    "RightUpLeg",
    "LeftLeg",
    "RightLeg",
    "LeftFoot",
    "RightFoot",
    "LeftArm",
    "RightArm",
    "LeftForeArm",
    "RightForeArm",
    "LeftHand",
    "RightHand",
)

PROFILE_WEIGHTS: dict[str, dict[str, float]] = {
    "baseline": {},
    "root": {
        "Hips": 1.0,
    },
    "feet": {
        "LeftFoot": 2.0,
        "RightFoot": 2.0,
    },
    "gmr_legs": {
        "LeftUpLeg": 1.0,
        "RightUpLeg": 1.0,
        "LeftLeg": 1.0,
        "RightLeg": 1.0,
        "LeftFoot": 1.0,
        "RightFoot": 1.0,
    },
    "upper": {
        "LeftArm": 1.0,
        "RightArm": 1.0,
        "LeftForeArm": 1.5,
        "RightForeArm": 1.5,
        "LeftHand": 1.0,
        "RightHand": 1.0,
    },
    "full": {
        "Hips": 1.0,
        "LeftUpLeg": 0.5,
        "RightUpLeg": 0.5,
        "LeftLeg": 0.75,
        "RightLeg": 0.75,
        "LeftFoot": 2.0,
        "RightFoot": 2.0,
        "LeftArm": 1.0,
        "RightArm": 1.0,
        "LeftForeArm": 1.5,
        "RightForeArm": 1.5,
        "LeftHand": 1.0,
        "RightHand": 1.0,
    },
}


@dataclass
class Config:
    data_path: Path = (
        PACKAGE_ROOT
        / "demo_data"
        / "noetix_mocap"
        / "0724_BEITI"
    )
    task_name: str = DEFAULT_TASK_NAME
    output_root: Path = (
        PACKAGE_ROOT
        / "demo_results_orientation"
        / "e1"
        / "robot_only"
        / "0724_BEITI"
        / DEFAULT_TASK_NAME
    )
    variants: tuple[str, ...] = (
        "baseline",
        "root",
        "feet",
        "gmr_legs",
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
) -> dict[str, float]:
    if variant not in PROFILE_WEIGHTS:
        raise ValueError(
            f"Unknown orientation ablation variant {variant!r}; "
            f"available: {sorted(PROFILE_WEIGHTS)}"
        )
    if not np.isfinite(weight_scale) or weight_scale < 0.0:
        raise ValueError("weight_scale must be finite and non-negative")
    active = PROFILE_WEIGHTS[variant]
    return {
        joint_name: float(active.get(joint_name, 0.0) * weight_scale)
        for joint_name in ORIENTATION_JOINTS
    }


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
            run_name = (
                variant
                if weight_scale == 1.0
                else f"{variant}_x{weight_scale:g}"
            )
            if run_name in seen_names:
                raise ValueError(f"Duplicate ablation run name: {run_name}")
            orientation_weights_for_variant(variant, weight_scale)
            seen_names.add(run_name)
            specs.append((run_name, variant, float(weight_scale)))
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


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PACKAGE_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _prepare_input(cfg: Config) -> tuple[Path, Path]:
    source_path = cfg.data_path / f"{cfg.task_name}.npz"
    if not source_path.is_file():
        raise FileNotFoundError(f"Noetix input not found: {source_path}")
    if cfg.frame_start == 0 and cfg.frame_count is None:
        return cfg.data_path, source_path

    subset_dir = cfg.output_root / "_input"
    subset_dir.mkdir(parents=True, exist_ok=True)
    subset_path = subset_dir / source_path.name
    with np.load(source_path, allow_pickle=False) as data:
        source_frames = int(data["global_joint_positions"].shape[0])
        start = int(cfg.frame_start)
        end = (
            source_frames
            if cfg.frame_count is None
            else start + int(cfg.frame_count)
        )
        if start < 0 or start >= source_frames or end <= start or end > source_frames:
            raise ValueError(
                f"Invalid frame interval [{start}, {end}) for {source_frames} frames"
            )
        payload: dict[str, np.ndarray] = {}
        for key in data.files:
            value = np.asarray(data[key])
            payload[key] = (
                value[start:end]
                if value.ndim > 0 and value.shape[0] == source_frames
                else value
            )
    np.savez_compressed(subset_path, **payload)
    return subset_dir, subset_path


def _result_summary(result_path: Path, elapsed_seconds: float) -> dict[str, Any]:
    with np.load(result_path, allow_pickle=False) as data:
        qpos = np.asarray(data["qpos"])
        human_mapped = np.asarray(data["mapped_human_joints"])
        robot_mapped = np.asarray(data["mapped_robot_joints"])
        position_errors = np.linalg.norm(robot_mapped - human_mapped, axis=-1)
        orientation_errors = np.asarray(data["orientation_errors_rad"])
        orientation_names = [
            str(name)
            for name in np.asarray(
                data["orientation_human_joint_names"]
            ).tolist()
        ]
        per_link_orientation = {
            name: {
                "mean_rad": float(np.mean(orientation_errors[:, link_idx])),
                "median_rad": float(
                    np.median(orientation_errors[:, link_idx])
                ),
                "p95_rad": float(
                    np.percentile(orientation_errors[:, link_idx], 95)
                ),
            }
            for link_idx, name in enumerate(orientation_names)
        }
        mapped_names = [
            str(name)
            for name in np.asarray(data["mapped_human_joint_names"]).tolist()
        ]
        per_link_position = {
            name: {
                "mean_m": float(np.mean(position_errors[:, link_idx])),
                "p95_m": float(
                    np.percentile(position_errors[:, link_idx], 95)
                ),
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
            "orientation_tracking_enabled": bool(
                np.asarray(data["orientation_tracking_enabled"]).item()
            ),
            "orientation_weights": np.asarray(
                data["orientation_weights"]
            ).tolist(),
            "orientation_overall": {
                "mean_rad": (
                    float(np.mean(orientation_values))
                    if orientation_values.size
                    else None
                ),
                "median_rad": (
                    float(np.median(orientation_values))
                    if orientation_values.size
                    else None
                ),
                "p95_rad": (
                    float(np.percentile(orientation_values, 95))
                    if orientation_values.size
                    else None
                ),
            },
            "orientation_per_link": per_link_orientation,
            "mapped_position_error": {
                "mean_m": float(np.mean(position_errors)),
                "median_m": float(np.median(position_errors)),
                "p95_m": float(np.percentile(position_errors, 95)),
            },
            "mapped_position_error_per_link": per_link_position,
            "actuated_motion_smoothness": {
                "mean_step_rad": (
                    float(np.mean(joint_steps))
                    if joint_steps.size
                    else 0.0
                ),
                "p95_step_rad": (
                    float(np.percentile(joint_steps, 95))
                    if joint_steps.size
                    else 0.0
                ),
                "mean_second_difference_rad": (
                    float(np.mean(joint_accelerations))
                    if joint_accelerations.size
                    else 0.0
                ),
                "p95_second_difference_rad": (
                    float(np.percentile(joint_accelerations, 95))
                    if joint_accelerations.size
                    else 0.0
                ),
            },
            "sqp_iterations": {
                "mean": float(np.mean(sqp_iterations)),
                "p95": float(np.percentile(sqp_iterations, 95)),
                "max": int(np.max(sqp_iterations)),
            },
            "foot_sticking_fallback_frames": int(
                np.asarray(data["foot_sticking_fallback_frames"]).size
            ),
            "foot_sticking_release_frames": int(
                np.asarray(data["foot_sticking_release_frames"]).size
            ),
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
    baseline_smoothness = baseline["actuated_motion_smoothness"][
        "mean_second_difference_rad"
    ]
    comparisons: dict[str, dict[str, float]] = {}
    for run_name, summary in summaries.items():
        if run_name == "baseline" or "orientation_overall" not in summary:
            continue
        orientation = summary["orientation_overall"]["mean_rad"]
        position = summary["mapped_position_error"]["mean_m"]
        smoothness = summary["actuated_motion_smoothness"][
            "mean_second_difference_rad"
        ]
        comparisons[run_name] = {
            "orientation_mean_delta_rad": float(
                orientation - baseline_orientation
            ),
            "orientation_mean_change_percent": float(
                100.0 * (orientation / baseline_orientation - 1.0)
            ),
            "position_mean_delta_m": float(position - baseline_position),
            "position_mean_change_percent": float(
                100.0 * (position / baseline_position - 1.0)
            ),
            "joint_second_difference_delta_rad": float(
                smoothness - baseline_smoothness
            ),
        }
    return comparisons


def _retargeting_config(
    *,
    input_dir: Path,
    output_dir: Path,
    task_name: str,
    orientation_weights: dict[str, float],
) -> RetargetingConfig:
    robot_urdf = PACKAGE_ROOT / "models" / "e1" / "e1_23dof.urdf"
    return RetargetingConfig(
        task_type="robot_only",
        robot="e1",
        data_format="noetix_mocap",
        task_name=task_name,
        data_path=input_dir,
        save_dir=output_dir,
        robot_config=RobotConfig(
            robot_type="e1",
            robot_urdf_file=str(robot_urdf),
        ),
        motion_data_config=MotionDataConfig(
            data_format="noetix_mocap",
            robot_type="e1",
        ),
        task_config=TaskConfig(object_name="ground"),
        retargeter=RetargeterConfig(
            orientation_weights=orientation_weights,
        ),
    )


def main(cfg: Config) -> None:
    input_dir, input_path = _prepare_input(cfg)
    cfg.output_root.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Any] = {}
    failures: list[str] = []

    for run_name, variant, weight_scale in ablation_run_specs(
        cfg.variants,
        cfg.weight_scales,
    ):
        output_dir = cfg.output_root / run_name
        output_dir.mkdir(parents=True, exist_ok=True)
        result_path = output_dir / f"{cfg.task_name}.npz"
        weights = orientation_weights_for_variant(
            variant,
            weight_scale,
        )
        retargeting_cfg = _retargeting_config(
            input_dir=input_dir,
            output_dir=output_dir,
            task_name=cfg.task_name,
            orientation_weights=weights,
        )
        manifest = {
            "variant": variant,
            "weight_scale": weight_scale,
            "orientation_weights": weights,
            "input_path": str(input_path),
            "input_sha256": _sha256(input_path),
            "git_commit": _git_commit(),
            "frame_start": cfg.frame_start,
            "frame_count": cfg.frame_count,
            "retargeting_config": _jsonable(asdict(retargeting_cfg)),
            "result_path": str(result_path),
            "status": "planned" if cfg.dry_run else "running",
        }
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if cfg.dry_run:
            summaries[run_name] = manifest
            continue
        if result_path.exists() and not cfg.overwrite:
            summaries[run_name] = _result_summary(
                result_path,
                elapsed_seconds=0.0,
            )
            manifest["status"] = "reused"
            manifest["elapsed_seconds"] = 0.0
            manifest_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            continue

        start_time = time.monotonic()
        try:
            run_retargeting(retargeting_cfg)
            elapsed = time.monotonic() - start_time
            manifest["status"] = "completed"
            manifest["elapsed_seconds"] = elapsed
            summaries[run_name] = _result_summary(
                result_path,
                elapsed_seconds=elapsed,
            )
        except Exception as exc:
            manifest["status"] = "failed"
            manifest["error"] = f"{type(exc).__name__}: {exc}"
            failures.append(run_name)
            if cfg.fail_fast:
                raise
        finally:
            manifest_path = output_dir / "manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    summary_payload = {
        "task_name": cfg.task_name,
        "input_path": str(input_path),
        "runs": summaries,
        "comparisons_to_baseline": _comparisons_to_baseline(summaries),
        "failures": failures,
    }
    summary_path = cfg.output_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if failures:
        raise RuntimeError(
            f"Orientation ablations failed: {failures}; see {summary_path}"
        )


if __name__ == "__main__":
    main(tyro.cli(Config))
