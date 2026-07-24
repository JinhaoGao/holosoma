"""Run a short real-optimizer acceptance matrix for OMOMO objects and robots."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from holosoma_retargeting.config_types.data_type import MotionDataConfig
from holosoma_retargeting.config_types.retargeting import RetargetingConfig
from holosoma_retargeting.config_types.robot import RobotConfig
from holosoma_retargeting.data_utils.object_assets import default_models_root
from holosoma_retargeting.data_utils.omomo import (
    OMOMO_OBJECT_NAMES,
    parse_omomo_sequence_name,
    preflight_omomo_dataset,
    select_omomo_files,
)
from holosoma_retargeting.examples.robot_retarget import main as run_retargeting


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, help="Directory containing OMOMO .pt files")
    parser.add_argument(
        "--robots",
        nargs="+",
        default=["g1", "e1"],
        help="Robot types to validate",
    )
    parser.add_argument(
        "--object-names",
        nargs="+",
        choices=OMOMO_OBJECT_NAMES,
        help="Optional object subset; defaults to all thirteen categories",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=2,
        help="Number of real source frames to optimize per robot/object pair",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Keep validation inputs, results, and validation_report.json here",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip the full dataset and asset preflight",
    )
    return parser.parse_args()


def _representative_sources(
    data_dir: Path,
    object_names: tuple[str, ...],
) -> dict[str, Path]:
    selected: dict[str, Path] = {}
    for path in select_omomo_files(data_dir, object_names):
        object_name = parse_omomo_sequence_name(path).object_name
        selected.setdefault(object_name, path)
    missing = set(object_names).difference(selected)
    if missing:
        raise FileNotFoundError(
            f"No representative OMOMO sequence for: {', '.join(sorted(missing))}"
        )
    return selected


def _prepare_validation_inputs(
    data_dir: Path,
    validation_root: Path,
    object_names: tuple[str, ...],
    frame_count: int,
) -> dict[str, str]:
    input_dir = validation_root / "OMOMO_new"
    input_dir.mkdir(parents=True, exist_ok=True)
    height_file = data_dir.parent / "height_dict.pkl"
    if not height_file.is_file():
        raise FileNotFoundError(f"OMOMO subject heights not found: {height_file}")
    shutil.copy2(height_file, validation_root / "height_dict.pkl")

    task_names: dict[str, str] = {}
    for object_name, source_path in _representative_sources(
        data_dir,
        object_names,
    ).items():
        tensor = torch.load(source_path, map_location="cpu", weights_only=True)
        if tensor.shape[0] < frame_count:
            raise ValueError(
                f"{source_path} has {tensor.shape[0]} frames, fewer than requested "
                f"acceptance length {frame_count}"
            )
        torch.save(tensor[:frame_count], input_dir / source_path.name)
        task_names[object_name] = source_path.stem
    return task_names


def _validate_result(
    result_path: Path,
    *,
    robot_name: str,
    robot_dof: int,
    object_name: str,
    frame_count: int,
) -> tuple[int, int]:
    with np.load(result_path, allow_pickle=False) as result:
        qpos = np.asarray(result["qpos"])
        if qpos.shape != (frame_count, 7 + robot_dof + 7):
            raise ValueError(
                f"Unexpected qpos shape {qpos.shape}; expected "
                f"{(frame_count, 7 + robot_dof + 7)}"
            )
        if not np.isfinite(qpos).all():
            raise ValueError("qpos contains NaN or Inf")
        if str(np.asarray(result["robot_type"]).item()) != robot_name:
            raise ValueError("Saved robot_type metadata does not match the run")
        if str(np.asarray(result["object_name"]).item()) != object_name:
            raise ValueError("Saved object_name metadata does not match the run")
        expected_object_shapes = {
            "object_points_demo_local": (100, 3),
            "object_points_target_local": (100, 3),
            "object_points_demo_world": (frame_count, 100, 3),
            "object_points_target_world": (frame_count, 100, 3),
        }
        for key, expected_shape in expected_object_shapes.items():
            points = np.asarray(result[key])
            if points.shape != expected_shape:
                raise ValueError(
                    f"Unexpected {key} shape {points.shape}; expected {expected_shape}"
                )
            if not np.isfinite(points).all():
                raise ValueError(f"{key} contains NaN or Inf")
    return qpos.shape


def run_acceptance(
    data_dir: Path,
    validation_root: Path,
    *,
    robots: tuple[str, ...],
    object_names: tuple[str, ...],
    frame_count: int,
    run_preflight: bool,
) -> dict:
    """Run the requested real-optimizer matrix and return a JSON-compatible report."""

    if frame_count < 1:
        raise ValueError("frames must be greater than zero")
    data_dir = data_dir.expanduser().resolve()
    validation_root.mkdir(parents=True, exist_ok=True)

    preflight = None
    if run_preflight:
        preflight = preflight_omomo_dataset(
            data_dir,
            asset_root=default_models_root(),
            validate_tensors=True,
        )
        if not preflight.ok:
            raise ValueError(
                f"OMOMO preflight reported {len(preflight.issues)} issue(s)"
            )

    task_names = _prepare_validation_inputs(
        data_dir,
        validation_root,
        object_names,
        frame_count,
    )
    input_dir = validation_root / "OMOMO_new"
    results: list[dict] = []
    failures: list[dict] = []

    for robot_name in robots:
        robot_config = RobotConfig(robot_type=robot_name)
        motion_config = MotionDataConfig(
            data_format="omomo",
            robot_type=robot_name,
        )
        for object_name in object_names:
            task_name = task_names[object_name]
            result_path = (
                validation_root
                / "results"
                / robot_name
                / f"{task_name}_original.npz"
            )
            try:
                run_retargeting(
                    RetargetingConfig(
                        task_type="object_interaction",
                        robot=robot_name,
                        data_format="omomo",
                        task_name=task_name,
                        data_path=input_dir,
                        save_dir=result_path.parent,
                        robot_config=robot_config,
                        motion_data_config=motion_config,
                    )
                )
                qpos_shape = _validate_result(
                    result_path,
                    robot_name=robot_name,
                    robot_dof=robot_config.ROBOT_DOF,
                    object_name=object_name,
                    frame_count=frame_count,
                )
            except Exception as exc:
                failures.append(
                    {
                        "robot": robot_name,
                        "object_name": object_name,
                        "task_name": task_name,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            else:
                results.append(
                    {
                        "robot": robot_name,
                        "object_name": object_name,
                        "task_name": task_name,
                        "result_path": str(result_path),
                        "qpos_shape": list(qpos_shape),
                    }
                )

    report = {
        "status": "passed" if not failures else "failed",
        "data_dir": str(data_dir),
        "validation_root": str(validation_root),
        "robots": list(robots),
        "object_names": list(object_names),
        "frames_per_case": frame_count,
        "expected_cases": len(robots) * len(object_names),
        "passed_cases": len(results),
        "failed_cases": len(failures),
        "preflight": preflight.to_dict() if preflight is not None else None,
        "results": results,
        "failures": failures,
    }
    report_path = validation_root / "validation_report.json"
    report_path.write_text(
        f"{json.dumps(report, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    """Run the acceptance matrix and exit non-zero if any case fails."""

    args = _parse_args()
    object_names = tuple(args.object_names or OMOMO_OBJECT_NAMES)
    robots = tuple(dict.fromkeys(args.robots))
    if args.output_dir is not None:
        report = run_acceptance(
            args.data_dir,
            args.output_dir.expanduser().resolve(),
            robots=robots,
            object_names=object_names,
            frame_count=args.frames,
            run_preflight=not args.skip_preflight,
        )
    else:
        with tempfile.TemporaryDirectory(prefix="holosoma_omomo_acceptance_") as tmpdir:
            report = run_acceptance(
                args.data_dir,
                Path(tmpdir),
                robots=robots,
                object_names=object_names,
                frame_count=args.frames,
                run_preflight=not args.skip_preflight,
            )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["failed_cases"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
