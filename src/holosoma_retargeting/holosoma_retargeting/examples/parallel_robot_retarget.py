# ruff: noqa: CPY001

"""Parallel entry point for all registered motion formats and task types."""

from __future__ import annotations

import fcntl
import json
import multiprocessing as mp
import os
import sys
import tempfile
import time
import traceback
import uuid
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import tyro

src_root = Path(__file__).resolve().parents[2]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from holosoma_retargeting.config_types.data_type import normalize_data_format  # noqa: E402
from holosoma_retargeting.config_types.retargeting import (  # noqa: E402
    ParallelRetargetingConfig,
    RetargetingConfig,
)
from holosoma_retargeting.data_utils.motion_data import discover_motion_files  # noqa: E402
from holosoma_retargeting.data_utils.object_assets import default_models_root  # noqa: E402
from holosoma_retargeting.data_utils.omomo import (  # noqa: E402
    OMOMO_OBJECT_NAMES,
    parse_omomo_sequence_name,
    preflight_omomo_dataset,
)
from holosoma_retargeting.retargeting_pipeline import (  # noqa: E402
    DEFAULT_DATA_FORMATS,
    RetargetJob,
    build_retarget_job,
    planned_variants,
    resolve_task_object_name,
    run_retargeting_job,
    validate_config,
)

# ----------------------------- Constants -----------------------------

DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parents[1] / "demo_results" / "v1"
_INHERITED_DEFAULT_DATA_PATH = Path("demo_data/OMOMO_new")
_INHERITED_DEFAULT_TASK_NAME = "sub3_largebox_003"


class BatchReportInUseError(RuntimeError):
    """Raised when another invocation owns the exact batch report path."""


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
        return None

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


def resolve_batch_data_dir(cfg: ParallelRetargetingConfig) -> Path:
    """Resolve the batch input root while keeping ``data_path`` as an alias.

    ``ParallelRetargetingConfig`` inherits the single-action ``data_path``
    field. Treat a non-default value as an explicit alias for ``data_dir`` and
    reject two different explicit roots instead of silently ignoring one.
    """

    data_dir = Path(cfg.data_dir)
    data_path = Path(cfg.data_path)
    data_dir_is_explicit = data_dir != _INHERITED_DEFAULT_DATA_PATH
    data_path_is_explicit = data_path != _INHERITED_DEFAULT_DATA_PATH
    if data_dir_is_explicit and data_path_is_explicit:
        if data_dir.expanduser().resolve() != data_path.expanduser().resolve():
            raise ValueError(
                "Parallel retargeting received conflicting --data-dir and "
                "--data-path values. Use one batch input root.",
            )
        return data_dir
    return data_path if data_path_is_explicit else data_dir


def _write_json_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as temporary_file:
            temporary_file.write(f"{json.dumps(payload, indent=2, sort_keys=True)}\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        temporary_path.replace(path)
        directory_descriptor = os.open(
            path.parent,
            os.O_RDONLY | os.O_DIRECTORY,
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _cleanup_stale_report_temps(path: Path) -> None:
    """Remove atomic report temporaries left by an uncatchable prior exit."""

    removed = False
    for candidate in sorted(
        path.parent.glob(f".{path.name}.*.tmp"),
    ):
        candidate.unlink(missing_ok=True)
        removed = True
    if removed:
        directory_descriptor = os.open(
            path.parent,
            os.O_RDONLY | os.O_DIRECTORY,
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)


@contextmanager
def _exclusive_report_lock(path: Path):
    """Fail fast when another invocation owns this report transaction."""

    lock_path = path.parent / f".{path.name}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        try:
            fcntl.flock(
                descriptor,
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as exc:
            raise BatchReportInUseError(
                "Another batch invocation already owns report path "
                f"{path}; choose a distinct --run-id/--report-path or wait "
                "for that invocation to finish.",
            ) from exc
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class _BatchReportTransaction:
    """One fail-closed report lifecycle held under an outermost file lock."""

    path: Path
    invocation_id: str
    started_at: str
    context: dict
    terminal_written: bool = False

    def _with_lifecycle(self, payload: dict, *, terminal: bool) -> dict:
        return {
            **payload,
            "invocation_id": self.invocation_id,
            "started_at": self.started_at,
            "finished_at": _utc_timestamp() if terminal else None,
        }

    def write_running(self) -> None:
        _write_json_report(
            self.path,
            self._with_lifecycle(self.context, terminal=False),
        )

    def update_context(self, payload: dict) -> None:
        self.context = {
            **self.context,
            **payload,
        }

    def write_terminal(self, payload: dict) -> None:
        if self.terminal_written:
            raise RuntimeError(
                f"Batch report transaction is already terminal: {self.path}",
            )
        terminal_payload = dict(payload)
        terminal_payload.setdefault("fatal_error", None)
        _write_json_report(
            self.path,
            self._with_lifecycle(terminal_payload, terminal=True),
        )
        self.context = terminal_payload
        self.terminal_written = True

    def write_failure(self, error: BaseException) -> None:
        failure_payload = {
            **self.context,
            "status": "failed",
            "completed_tasks": int(self.context.get("completed_tasks", 0)),
            "skipped_tasks": int(self.context.get("skipped_tasks", 0)),
            "failed_tasks": int(self.context.get("failed_tasks", 0)),
            "results": list(self.context.get("results", [])),
            "failures": list(self.context.get("failures", [])),
            "fatal_error": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }
        _write_json_report(
            self.path,
            self._with_lifecycle(failure_payload, terminal=True),
        )
        self.context = failure_payload
        self.terminal_written = True


def _provisional_data_format(cfg: ParallelRetargetingConfig) -> str:
    selected = cfg.data_format or DEFAULT_DATA_FORMATS.get(
        str(cfg.task_type),
        "unknown",
    )
    try:
        return normalize_data_format(selected)
    except (KeyError, TypeError, ValueError):
        return str(selected)


def _resolve_batch_report_path(cfg: ParallelRetargetingConfig) -> Path:
    save_dir = Path(cfg.save_dir) if cfg.save_dir is not None else DEFAULT_RESULTS_ROOT
    data_format = _provisional_data_format(cfg)
    run_id = cfg.run_id or f"{cfg.robot}-{cfg.task_type}-{data_format}"
    report_path = cfg.report_path or save_dir / "runs" / run_id / "report.json"
    return Path(report_path).expanduser().resolve()


def _initial_report_context(
    cfg: ParallelRetargetingConfig,
) -> dict:
    return {
        "status": "running",
        "task_type": str(cfg.task_type),
        "robot": str(cfg.robot),
        "data_format": _provisional_data_format(cfg),
        "data_dir": str(cfg.data_dir),
        "data_path": str(cfg.data_path),
        "save_dir": str(
            Path(cfg.save_dir) if cfg.save_dir is not None else DEFAULT_RESULTS_ROOT,
        ),
        "object_names": (
            list(cfg.object_names)
            if cfg.object_names is not None
            else None
        ),
        "total_files": None,
        "completed_tasks": 0,
        "skipped_tasks": 0,
        "failed_tasks": 0,
        "results": [],
        "failures": [],
        "fatal_error": None,
    }


@contextmanager
def _batch_report_transaction(
    path: Path,
    cfg: ParallelRetargetingConfig,
):
    """Own one report path and replace stale terminal state before work starts."""

    with _exclusive_report_lock(path):
        transaction = _BatchReportTransaction(
            path=path,
            invocation_id=uuid.uuid4().hex,
            started_at=_utc_timestamp(),
            context=_initial_report_context(cfg),
        )
        transaction.write_running()
        try:
            _cleanup_stale_report_temps(path)
            yield transaction
        except BaseException as error:
            if not transaction.terminal_written:
                try:
                    transaction.write_failure(error)
                except BaseException as reporting_error:
                    error.add_note(
                        "Failed to persist the terminal batch failure report: "
                        f"{type(reporting_error).__name__}: {reporting_error}",
                    )
            raise
        if not transaction.terminal_written:
            error = RuntimeError(
                "Batch report transaction exited without a terminal report",
            )
            transaction.write_failure(error)
            raise error


def extract_task_name(file_path):
    """Extract task name from file path."""
    return Path(file_path).stem


def process_single_task(args):
    """Plan and execute one source family through the shared job runner."""

    (
        file_path,
        results_root,
        data_root,
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
    results_root = Path(results_root)
    data_root = Path(data_root).expanduser().absolute()
    source_path = Path(file_path).expanduser().absolute()
    if task_type == "climbing":
        task_dir = source_path.parent
        task_name = task_dir.name
        data_path = task_dir.parent
        sequence_key = task_dir.relative_to(data_root).as_posix()
    else:
        task_name = extract_task_name(source_path)
        data_path = source_path.parent
        sequence_key = source_path.relative_to(data_root).with_suffix("").as_posix()
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

    base_config = RetargetingConfig(
        task_type=task_type,
        robot=robot_config.robot_type,
        data_format=data_format,
        task_name=task_name,
        data_path=data_path,
        save_dir=None,
        augmentation=False,
        robot_config=robot_config,
        motion_data_config=motion_data_config,
        task_config=task_config,
        retargeter=retargeter_config,
    )
    variants = planned_variants(task_type, augmentation=augmentation)
    jobs: list[RetargetJob] = []
    source_sha256: str | None = None
    for variant in variants:
        run_kind = "single" if variant.is_identity else "augmentation"
        job = build_retarget_job(
            base_config,
            variant=variant,
            run_kind=run_kind,
            results_root=results_root,
            dataset_partition=data_root.name,
            sequence_key=sequence_key,
            source_path=source_path,
            source_sha256=source_sha256,
            overwrite_existing=overwrite_existing,
        )
        source_sha256 = job.source_sha256
        jobs.append(job)

    generated_files: list[str] = []
    skipped_files: list[str] = []

    for job in jobs:
        completed_job = run_retargeting_job(job)
        if completed_job.resumed:
            print(f"  Resuming valid result: {job.output_path}")
            skipped_files.append(str(job.output_path))
        else:
            generated_files.append(str(job.output_path))

    return TaskProcessResult(
        task_name=task_name,
        object_name=resolved_object_name,
        generated_files=tuple(generated_files),
        skipped_files=tuple(skipped_files),
        elapsed_seconds=time.monotonic() - task_start,
    )


def _run_batch_locked(
    cfg: ParallelRetargetingConfig,
    report_transaction: _BatchReportTransaction,
) -> None:
    """Run one batch while its report transaction lock is held.

    Args:
        cfg: Configuration arguments
        report_transaction: Outermost report lifecycle for this invocation
    """
    validate_config(cfg)
    if cfg.task_name != _INHERITED_DEFAULT_TASK_NAME:
        raise ValueError(
            "Parallel retargeting derives each task name from discovered "
            "source files; inherited --task-name is not supported.",
        )
    robot = cfg.robot
    task_type = cfg.task_type

    # Set defaults based on task type
    data_format = normalize_data_format(cfg.data_format or DEFAULT_DATA_FORMATS[task_type])
    save_dir = cfg.save_dir if cfg.save_dir is not None else DEFAULT_RESULTS_ROOT
    data_dir = resolve_batch_data_dir(cfg)

    save_dir = Path(save_dir)
    data_dir = Path(data_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Task type: {task_type}, Format: {data_format}")
    print(f"Data dir: {data_dir}, Save dir: {save_dir}")

    # Ensure configs match top-level selections
    if cfg.robot_config.robot_type != robot:
        cfg.robot_config = replace(cfg.robot_config, robot_type=robot)

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
                f"{issue.code}: {issue.path} ({issue.message})" for issue in preflight_report.issues[:10]
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

    object_counts = Counter(entry["object_name"] for entry in manifest if entry["object_name"] is not None)
    report_path = report_transaction.path
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
    report_transaction.update_context(report_base)
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
        report_transaction.write_terminal(report)
        print(f"Dry-run manifest written to: {report_path}")
        return

    # Pass configs to worker processes
    process_args = [
        (
            file_path,
            save_dir,
            data_dir,
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
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=context,
    ) as executor:
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
    report_transaction.write_terminal(report)

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
        raise RuntimeError(f"{len(failures)} of {len(files)} retargeting tasks failed; see {report_path} for details")


def main(cfg: ParallelRetargetingConfig) -> None:
    """Run one auditable batch under an exclusive report transaction.

    The report path is locked before validation or preflight. A fresh
    ``running`` tombstone atomically replaces any older terminal report before
    work starts; ordinary exceptions, ``KeyboardInterrupt``, and ``SystemExit``
    are then recorded as terminal failures. An uncatchable process or machine
    termination intentionally leaves ``running`` rather than a stale
    ``completed`` claim.
    """

    report_path = _resolve_batch_report_path(cfg)
    with _batch_report_transaction(
        report_path,
        cfg,
    ) as report_transaction:
        _run_batch_locked(
            cfg,
            report_transaction,
        )


if __name__ == "__main__":
    cfg = tyro.cli(ParallelRetargetingConfig)
    main(cfg)
