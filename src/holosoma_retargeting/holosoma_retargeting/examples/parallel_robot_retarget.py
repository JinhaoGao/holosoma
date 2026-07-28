"""Parallel entry point for all registered motion formats and task types."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys

# Add src to path for direct execution
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np
import tyro

src_root = Path(__file__).resolve().parents[2]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from holosoma_retargeting.config_types.data_type import normalize_data_format  # noqa: E402
from holosoma_retargeting.config_types.retargeting import ParallelRetargetingConfig  # noqa: E402
from holosoma_retargeting.config_types.robot import RobotConfig  # noqa: E402
from holosoma_retargeting.data_utils.motion_data import discover_motion_files  # noqa: E402
from holosoma_retargeting.data_utils.object_assets import default_models_root  # noqa: E402
from holosoma_retargeting.data_utils.omomo import (  # noqa: E402
    OMOMO_OBJECT_NAMES,
    parse_omomo_sequence_name,
    preflight_omomo_dataset,
)

# Import reusable functions from robot_retarget.py
from holosoma_retargeting.examples.robot_retarget import (  # type: ignore[import-not-found]  # noqa: E402
    DEFAULT_DATA_FORMATS,
    build_retargeter_kwargs_from_config,
    create_task_constants,
    initialize_robot_pose,
    load_motion_data,
    resolve_task_object_name,
    setup_object_data,
    validate_config,
)

# Import after path modification
from holosoma_retargeting.src.interaction_mesh_retargeter import (  # noqa: E402
    InteractionMeshRetargeter,  # type: ignore[import-not-found]
)
from holosoma_retargeting.src.utils import (  # type: ignore[import-not-found]  # noqa: E402
    extract_foot_sticking_sequence_velocity,
    preprocess_motion_data,
)

# ----------------------------- Constants -----------------------------

# Override save directories for parallel processing (use demo_results_parallel instead of demo_results)
PARALLEL_SAVE_DIRS = {
    "robot_only": "demo_results_parallel/{robot}/robot_only/{data_format}",
    "object_interaction": "demo_results_parallel/{robot}/object_interaction/{data_format}",
    "climbing": "demo_results_parallel/{robot}/climbing/{data_format}",
}


@dataclass(frozen=True)
class TaskProcessResult:
    """Serializable result returned by one worker."""

    task_name: str
    object_name: str
    generated_files: tuple[str, ...]
    skipped_files: tuple[str, ...]
    elapsed_seconds: float

    @property
    def status(self) -> str:
        """Summarize whether this task generated or only skipped outputs."""

        return "completed" if self.generated_files else "skipped"


def resolve_batch_object_names(
    task_type: str,
    data_format: str,
    task_object_name: str | None,
    object_names: Iterable[str] | None,
) -> tuple[str, ...] | None:
    """Normalize multi-object filters and reject conflicting selections."""

    requested = tuple(dict.fromkeys(object_names or ()))
    if task_type != "object_interaction" or data_format != "omomo":
        if requested:
            raise ValueError("object_names is only supported for OMOMO object-interaction batches")
        return (task_object_name,) if task_object_name else None

    unknown = set(requested).difference(OMOMO_OBJECT_NAMES)
    if unknown:
        raise ValueError(f"Unknown OMOMO object filters: {', '.join(sorted(unknown))}")
    if task_object_name is not None:
        if requested and requested != (task_object_name,):
            raise ValueError("task_config.object_name conflicts with object_names")
        return (task_object_name,)
    return requested or None


def find_files(
    data_dir: Path,
    data_format: str,
    object_names: Iterable[str] | str | None = None,
) -> list[str]:
    """Discover files through the shared single/batch format registry."""

    if isinstance(object_names, str):
        selected_objects = (object_names,)
    else:
        selected_objects = tuple(object_names) if object_names is not None else None
    files = discover_motion_files(data_dir, data_format)
    if data_format == "omomo" and selected_objects is not None:
        selected = set(selected_objects)
        files = [path for path in files if parse_omomo_sequence_name(path).object_name in selected]
    return [str(path) for path in files]


def _output_path(save_dir: Path, task_type: str, task_name: str, augmentation_name: str) -> Path:
    if task_type == "robot_only":
        return save_dir / f"{task_name}.npz"
    return save_dir / f"{task_name}_{augmentation_name}.npz"


def _write_json_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def generate_augmentation_configs(task_type: str, augmentation: bool = True):
    """Generate augmentation configurations based on task type."""
    if task_type == "robot_only":
        # No augmentation for robot_only
        return [{"name": "original"}]

    if task_type == "object_interaction":
        """Generate different augmentation configurations for object interaction."""
        augmentations = []
        augmentations.append({"name": "original", "translation": np.array([0.0, 0.0, 0.0]), "rotation": 0.0})

        if augmentation:
            # Translation augmentations
            translations = [
                [0.2, 0.0, 0.0],  # forward
                [0.0, 0.2, 0.0],  # left
                [0.0, -0.2, 0.0],  # right
            ]
            for i, trans in enumerate(translations):
                augmentations.append({"name": f"trans_{i}", "translation": np.array(trans), "rotation": 0.0})

            # Rotation augmentations
            rotations = [np.pi / 4, -np.pi / 4]
            for i, rot in enumerate(rotations):
                augmentations.append(
                    {
                        "name": f"rot_{i}",
                        "translation": np.array([0.0, 0.2 * (-1) ** i, 0.0]),
                        "rotation": rot,
                    }
                )

        return augmentations

    if task_type == "climbing":
        """Generate augmentation configurations for climbing (object scaling)."""
        configs = [{"name": "original", "scale": np.array([1, 1, 1])}]
        if augmentation:
            configs.extend(
                {"name": f"z_scale_{z_scale}", "scale": np.array([1, 1, z_scale])} for z_scale in [0.8, 0.9, 1.1, 1.2]
            )
        return configs

    raise ValueError(f"Invalid task type: {task_type}")


def extract_task_name(file_path):
    """Extract task name from file path."""
    return Path(file_path).stem


def process_single_task(args):
    """Process a single task with all augmentations.

    This function follows the same structure as main() in robot_retarget.py,
    but handles multiple augmentations in a loop for parallel processing.
    """
    (
        file_path,
        save_dir,
        task_type,
        data_format,
        robot_config,
        motion_data_config,
        task_config,
        retargeter_config,
        augmentation,
        overwrite_existing,
    ) = args

    task_start = time.monotonic()
    os.makedirs(save_dir, exist_ok=True)
    save_dir = Path(save_dir)
    source_path = Path(file_path)
    if task_type == "climbing":
        task_dir = source_path.parent
        task_name = task_dir.name
        data_path = task_dir.parent
    else:
        task_name = extract_task_name(source_path)
        data_path = source_path.parent
    print(f"Processing: {task_name}")

    resolved_object_name = resolve_task_object_name(
        task_type,
        data_format,
        task_name,
        task_config.object_name,
    )
    task_config = replace(task_config, object_name=resolved_object_name)

    # Task-specific object setup: set default object_dir for climbing if not provided
    if task_type == "climbing" and task_config.object_dir is None:
        task_config = replace(task_config, object_dir=task_dir)

    augmentations = generate_augmentation_configs(task_type, augmentation)
    expected_outputs = [
        _output_path(save_dir, task_type, task_name, config["name"])
        for config in augmentations
    ]
    if not overwrite_existing and all(path.exists() for path in expected_outputs):
        return TaskProcessResult(
            task_name=task_name,
            object_name=resolved_object_name,
            generated_files=(),
            skipped_files=tuple(str(path) for path in expected_outputs),
            elapsed_seconds=time.monotonic() - task_start,
        )

    constants = create_task_constants(robot_config, motion_data_config, task_config, task_type)

    # Load motion data
    (
        human_joints,
        object_poses,
        human_joint_quaternions,
        smpl_scale,
    ) = load_motion_data(
        task_type, data_format, data_path, task_name, constants, motion_data_config
    )

    # Preserve original data (preprocess_motion_data modifies them in place)
    human_joints_original = human_joints.copy()
    object_poses_original = object_poses.copy()
    human_joint_quaternions_original = (
        None
        if human_joint_quaternions is None
        else human_joint_quaternions.copy()
    )

    # Get toe names from motion data config (depends only on data_format)
    toe_names = motion_data_config.toe_names

    # Process all augmentations
    print("The number of augmentations: ", len(augmentations))
    generated_files: list[str] = []
    skipped_files: list[str] = []

    for k, aug_config in enumerate(augmentations):
        # Use fresh copies for each iteration
        human_joints = human_joints_original.copy()
        object_poses = object_poses_original.copy()
        human_joint_quaternions = (
            None
            if human_joint_quaternions_original is None
            else human_joint_quaternions_original.copy()
        )
        aug_name = aug_config["name"]
        file_name = str(_output_path(save_dir, task_type, task_name, aug_name))

        print(f"  Processing augmentation: {aug_name}")
        if Path(file_name).exists() and not overwrite_existing:
            print(f"  Skipping existing result: {file_name}")
            skipped_files.append(file_name)
            continue

        # Setup object data
        if task_type == "climbing":
            print("object_dir: ", task_config.object_dir)
            object_local_pts, object_local_pts_demo, object_urdf_path = setup_object_data(
                task_type,
                constants,
                task_config.object_dir,
                smpl_scale,
                task_config,
                augmentation=(k > 0),
                object_scale_augmented=aug_config["scale"],
            )
        else:
            object_local_pts, object_local_pts_demo, object_urdf_path = setup_object_data(
                task_type,
                constants,
                task_config.object_dir,
                smpl_scale,
                task_config,
                augmentation=(k > 0),
            )

        # Create retargeter
        retargeter_kwargs = build_retargeter_kwargs_from_config(
            retargeter_config, constants, object_urdf_path, task_type
        )
        retargeter = InteractionMeshRetargeter(**retargeter_kwargs)

        # Preprocess motion data
        if task_type == "robot_only":
            human_joints = preprocess_motion_data(human_joints, retargeter, toe_names, smpl_scale)
        elif task_type in {"object_interaction", "climbing"}:
            human_joints, object_poses, _object_moving_frame_idx = preprocess_motion_data(
                human_joints, retargeter, toe_names, scale=smpl_scale, object_poses=object_poses
            )

        # Extract foot sticking sequences
        foot_sticking_sequences = extract_foot_sticking_sequence_velocity(
            human_joints, retargeter.demo_joints, toe_names
        )

        # Task-specific foot sticking adjustments
        if task_type == "object_interaction":
            # Disable initial sticking
            foot_sticking_sequences[0][toe_names[0]] = False
            foot_sticking_sequences[0][toe_names[1]] = False

        # Determine if this is an augmentation run (k > 0 means we're augmenting)
        is_augmentation_run = k > 0

        if task_type == "object_interaction":
            # Initialize robot pose
            q_init, q_nominal, object_poses_augmented, human_joints, object_poses = initialize_robot_pose(
                task_type,
                human_joints,
                object_poses,
                constants,
                retargeter,
                task_config,
                is_augmentation_run,
                save_dir,
                task_name,
                augmentation_translation=aug_config["translation"],
                augmentation_rotation=aug_config["rotation"],
            )
        else:
            # Initialize robot pose
            q_init, q_nominal, object_poses_augmented, human_joints, object_poses = initialize_robot_pose(
                task_type,
                human_joints,
                object_poses,
                constants,
                retargeter,
                task_config,
                is_augmentation_run,
                save_dir,
                task_name,
            )

        # Retarget motion
        retargeter.retarget_motion(
            human_joint_motions=human_joints,
            human_joint_quaternions_wxyz=human_joint_quaternions,
            object_poses=object_poses,
            object_poses_augmented=object_poses_augmented,
            object_points_local_demo=object_local_pts_demo,
            object_points_local=object_local_pts,
            foot_sticking_sequences=foot_sticking_sequences,
            q_a_init=q_init,
            q_nominal_list=q_nominal,
            original=(k == 0),
            dest_res_path=file_name,
            fps=constants.SOURCE_FPS,
        )
        generated_files.append(file_name)

    return TaskProcessResult(
        task_name=task_name,
        object_name=resolved_object_name,
        generated_files=tuple(generated_files),
        skipped_files=tuple(skipped_files),
        elapsed_seconds=time.monotonic() - task_start,
    )


def main(cfg: ParallelRetargetingConfig) -> None:
    """Main parallel retargeting pipeline.

    Args:
        cfg: Configuration arguments
    """
    validate_config(cfg)
    robot = cfg.robot
    task_type = cfg.task_type

    # Set defaults based on task type
    data_format = normalize_data_format(cfg.data_format or DEFAULT_DATA_FORMATS[task_type])
    save_dir = (
        cfg.save_dir
        if cfg.save_dir is not None
        else Path(PARALLEL_SAVE_DIRS[task_type].format(robot=robot, data_format=data_format))
    )
    data_dir = cfg.data_dir

    save_dir = Path(save_dir)
    data_dir = Path(data_dir)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Task type: {task_type}, Format: {data_format}")
    print(f"Data dir: {data_dir}, Save dir: {save_dir}")

    # Ensure configs match top-level selections
    if cfg.robot_config.robot_type != robot:
        cfg.robot_config = RobotConfig(robot_type=robot)

    if cfg.motion_data_config.robot_type != robot or cfg.motion_data_config.data_format != data_format:
        cfg.motion_data_config = replace(cfg.motion_data_config, data_format=data_format, robot_type=robot)

    requested_objects = resolve_batch_object_names(
        task_type,
        data_format,
        cfg.task_config.object_name,
        cfg.object_names,
    )
    preflight_report = None
    if task_type == "object_interaction" and data_format == "omomo" and cfg.preflight:
        print("Running OMOMO dataset and object-asset preflight")
        preflight_report = preflight_omomo_dataset(
            data_dir,
            asset_root=default_models_root(),
            validate_tensors=cfg.validate_input_tensors,
        )
        if not preflight_report.ok:
            issue_summary = "; ".join(
                f"{issue.code}: {issue.path} ({issue.message})"
                for issue in preflight_report.issues[:10]
            )
            remaining = len(preflight_report.issues) - 10
            if remaining > 0:
                issue_summary += f"; and {remaining} more issue(s)"
            raise ValueError(f"OMOMO preflight failed: {issue_summary}")

    files = find_files(data_dir, data_format, requested_objects)
    print(f"Found {len(files)} files for task type: {task_type}")
    if not files:
        raise FileNotFoundError(f"No {data_format} motion files found in {data_dir}")

    manifest: list[dict[str, str | None]] = []
    for file_path in files:
        source_path = Path(file_path)
        task_name = source_path.parent.name if task_type == "climbing" else source_path.stem
        if task_type == "object_interaction" and data_format == "omomo":
            object_name = parse_omomo_sequence_name(task_name).object_name
        else:
            object_name = cfg.task_config.object_name
        manifest.append(
            {
                "source_path": str(source_path),
                "task_name": task_name,
                "object_name": object_name,
            }
        )

    object_counts = Counter(
        entry["object_name"] for entry in manifest if entry["object_name"] is not None
    )
    report_path = cfg.report_path or save_dir / "batch_report.json"
    report_base = {
        "task_type": task_type,
        "robot": robot,
        "data_format": data_format,
        "data_dir": str(data_dir),
        "save_dir": str(save_dir),
        "object_names": list(requested_objects) if requested_objects is not None else None,
        "total_files": len(files),
        "per_object": dict(sorted(object_counts.items())),
        "manifest": manifest,
        "preflight": preflight_report.to_dict() if preflight_report is not None else None,
    }
    if cfg.dry_run:
        report = {
            **report_base,
            "status": "dry_run",
            "completed_tasks": 0,
            "skipped_tasks": 0,
            "failed_tasks": 0,
            "results": [],
            "failures": [],
        }
        _write_json_report(Path(report_path), report)
        print(f"Dry-run manifest written to: {report_path}")
        return

    # Pass configs to worker processes
    process_args = [
        (
            file_path,
            save_dir,
            task_type,
            data_format,
            cfg.robot_config,
            cfg.motion_data_config,
            cfg.task_config,
            cfg.retargeter,
            cfg.augmentation,
            cfg.overwrite_existing,
        )
        for file_path in files
    ]

    # Set up parallel processing
    max_workers = cfg.max_workers if cfg.max_workers is not None else min(4, mp.cpu_count())
    if max_workers <= 0:
        raise ValueError("max_workers must be greater than zero")
    print(f"Using {max_workers} parallel workers")

    start_time = time.monotonic()
    results: list[dict] = []
    failures: list[dict[str, str | None]] = []

    # Process files in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(process_single_task, arg): arg[0] for arg in process_args}

        # Process completed tasks
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                task_result = future.result()
                result = asdict(task_result)
                result["status"] = task_result.status
                results.append(result)
                print(f"{task_result.status.capitalize()}: {file_path}")
            except Exception as e:
                print(f"Failed {file_path}: {e}")
                traceback.print_exc()
                source_path = Path(file_path)
                task_name = source_path.parent.name if task_type == "climbing" else source_path.stem
                if task_type == "object_interaction" and data_format == "omomo":
                    object_name = parse_omomo_sequence_name(task_name).object_name
                else:
                    object_name = cfg.task_config.object_name
                failures.append(
                    {
                        "source_path": str(source_path),
                        "task_name": task_name,
                        "object_name": object_name,
                        "error": f"{type(e).__name__}: {e}",
                    }
                )

    elapsed_seconds = time.monotonic() - start_time
    results.sort(key=lambda result: result["task_name"])
    failures.sort(key=lambda failure: str(failure["task_name"]))
    completed_tasks = sum(result["status"] == "completed" for result in results)
    skipped_tasks = sum(result["status"] == "skipped" for result in results)
    report = {
        **report_base,
        "status": "completed_with_failures" if failures else "completed",
        "completed_tasks": completed_tasks,
        "skipped_tasks": skipped_tasks,
        "failed_tasks": len(failures),
        "elapsed_seconds": elapsed_seconds,
        "results": results,
        "failures": failures,
    }
    _write_json_report(Path(report_path), report)

    print("\n=== Processing Summary ===")
    print(f"Task type: {task_type}")
    print(f"Total files: {len(files)}")
    print(f"Completed: {completed_tasks}")
    print(f"Skipped: {skipped_tasks}")
    print(f"Failed: {len(failures)}")
    print(f"Total time: {elapsed_seconds:.2f} seconds")
    if len(files) > 0:
        print(f"Average time per file: {elapsed_seconds / len(files):.2f} seconds")
    print(f"Results saved to: {save_dir}")
    print(f"Report written to: {report_path}")
    if failures:
        raise RuntimeError(
            f"{len(failures)} of {len(files)} retargeting tasks failed; "
            f"see {report_path} for details"
        )


if __name__ == "__main__":
    cfg = tyro.cli(ParallelRetargetingConfig)
    main(cfg)
