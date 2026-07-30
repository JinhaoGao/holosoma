# ruff: noqa: CPY001

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from holosoma_retargeting.config_types.data_type import MotionDataConfig
from holosoma_retargeting.config_types.retargeting import RetargetingConfig
from holosoma_retargeting.config_types.robot import RobotConfig
from holosoma_retargeting.retargeting_pipeline import (
    IDENTITY_VARIANT,
    RetargetVariant,
    _job_file_lock,
    _validate_job_semantics,
    _verify_job_config,
    _verify_job_source,
    build_retarget_job,
)


def _robot_only_config(source: Path) -> RetargetingConfig:
    return RetargetingConfig(
        task_type="robot_only",
        robot="g1",
        dataset="lafan",
        data_format="lafan",
        task_name=source.stem,
        data_path=source.parent,
        robot_config=RobotConfig(robot_type="g1"),
        motion_data_config=MotionDataConfig(
            data_format="lafan",
            robot_type="g1",
        ),
    )


def test_job_contract_is_compact_and_colocated(tmp_path: Path) -> None:
    source = tmp_path / "walk.npy"
    np.save(source, np.zeros((2, 22, 3), dtype=np.float32))
    results_root = tmp_path / "results"

    job = build_retarget_job(
        _robot_only_config(source),
        results_root=results_root,
    )
    decoded = json.loads(job.config_json)

    assert job.variant == IDENTITY_VARIANT
    assert job.output_path == results_root / "g1" / "robot_only" / "lafan" / "walk" / "identity.npz"
    assert job.generated_assets_dir == job.output_path.parent / ".assets" / "identity"
    assert job.output_lock_path == job.output_path.parent / ".locks" / "identity.lock"
    assert job.baseline_lock_path == job.output_lock_path
    assert decoded["experiment_name"] is None
    assert "solver_identity" not in decoded


def test_source_and_config_mutation_are_detected(tmp_path: Path) -> None:
    source = tmp_path / "walk.npy"
    np.save(source, np.zeros((2, 22, 3), dtype=np.float32))
    job = build_retarget_job(
        _robot_only_config(source),
        results_root=tmp_path / "results",
    )

    _verify_job_source(job)
    _verify_job_config(job)

    source.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="source bytes changed"):
        _verify_job_source(job)

    np.save(source, np.zeros((2, 22, 3), dtype=np.float32))
    rebuilt = build_retarget_job(
        _robot_only_config(source),
        results_root=tmp_path / "results",
    )
    rebuilt.config.task_name = "mutated"
    with pytest.raises(RuntimeError, match="configuration changed"):
        _verify_job_config(rebuilt)


def test_only_production_run_kinds_are_accepted() -> None:
    _validate_job_semantics(
        variant=IDENTITY_VARIANT,
        run_kind="single",
    )
    _validate_job_semantics(
        variant=RetargetVariant(
            name="translated",
            translation=(0.2, 0.0, 0.0),
        ),
        run_kind="augmentation",
    )
    with pytest.raises(ValueError, match=r"single.*augmentation"):
        _validate_job_semantics(
            variant=IDENTITY_VARIANT,
            run_kind="ablation",
        )
    with pytest.raises(ValueError, match="exact identity"):
        _validate_job_semantics(
            variant=RetargetVariant(name="named-zero"),
            run_kind="single",
        )
    with pytest.raises(ValueError, match="non-identity"):
        _validate_job_semantics(
            variant=RetargetVariant(name="named-zero"),
            run_kind="augmentation",
        )


def test_output_lock_serializes_concurrent_writers(tmp_path: Path) -> None:
    lock_path = tmp_path / ".locks" / "identity.lock"
    start = threading.Barrier(2)
    state_guard = threading.Lock()
    active = 0
    max_active = 0

    def hold_lock() -> None:
        nonlocal active, max_active
        start.wait()
        with _job_file_lock(lock_path, shared=False):
            with state_guard:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.03)
            with state_guard:
                active -= 1

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(hold_lock) for _ in range(2)]
        for future in futures:
            future.result()

    assert max_active == 1
