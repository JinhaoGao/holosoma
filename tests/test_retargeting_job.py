# ruff: noqa: CPY001

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from holosoma_retargeting.config_types.data_type import MotionDataConfig
from holosoma_retargeting.config_types.retargeter import RetargeterConfig
from holosoma_retargeting.config_types.retargeting import RetargetingConfig
from holosoma_retargeting.config_types.robot import RobotConfig
from holosoma_retargeting.data_utils.batch_npz_to_csv import convert_npz_directory
from holosoma_retargeting.data_utils.npz_to_csv import convert_npz_to_csv
from holosoma_retargeting.retargeting_pipeline import (
    IDENTITY_VARIANT,
    RetargetVariant,
    _validate_job_semantics,
    build_retarget_job,
    run_retargeting_job,
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


def _write_two_joint_urdf(path: Path) -> None:
    path.write_text(
        """<robot name="test">
  <link name="base"/>
  <link name="one"/>
  <link name="two"/>
  <joint name="joint_b" type="revolute"><parent link="base"/><child link="one"/></joint>
  <joint name="fixed_joint" type="fixed"><parent link="one"/><child link="two"/></joint>
  <joint name="joint_a" type="revolute"><parent link="one"/><child link="two"/></joint>
</robot>""",
        encoding="utf-8",
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
    assert job.output_path == results_root / "g1" / "robot_only" / "lafan" / "walk.npz"
    assert job.generated_assets_dir == job.output_path.parent / ".assets" / "walk"
    assert decoded["experiment_name"] is None
    assert "solver_identity" not in decoded


def test_npz_to_csv_reorders_joints_and_writes_only_robot_numeric_data(tmp_path: Path) -> None:
    urdf = tmp_path / "robot.urdf"
    _write_two_joint_urdf(urdf)
    source = tmp_path / "motion.npz"
    qpos = np.arange(32, dtype=np.float64).reshape(2, 16)
    np.savez(
        source,
        qpos=qpos,
        robot_actuated_joint_names=np.asarray(("joint_a", "joint_b")),
    )

    destination = convert_npz_to_csv(source, urdf)

    assert destination == tmp_path / "motion.csv"
    np.testing.assert_allclose(
        np.loadtxt(destination, delimiter=","),
        np.concatenate((qpos[:, (0, 1, 2, 4, 5, 6, 3)], qpos[:, (8, 7)]), axis=1),
    )
    assert all(character not in destination.read_text() for character in "abcdefghijklmnopqrstuvwxyz")


def test_batch_npz_to_csv_preserves_relative_paths(tmp_path: Path) -> None:
    urdf = tmp_path / "robot.urdf"
    _write_two_joint_urdf(urdf)
    source = tmp_path / "input" / "nested" / "motion.npz"
    source.parent.mkdir(parents=True)
    np.savez(
        source,
        qpos=np.zeros((1, 9), dtype=np.float64),
        robot_actuated_joint_names=np.asarray(("joint_a", "joint_b")),
    )

    outputs = convert_npz_directory(tmp_path / "input", urdf, tmp_path / "output")

    assert outputs == (tmp_path / "output" / "nested" / "motion.csv",)


def test_live_display_options_do_not_change_result_identity(tmp_path: Path) -> None:
    source = tmp_path / "walk.npy"
    np.save(source, np.zeros((2, 22, 3), dtype=np.float32))
    results_root = tmp_path / "results"
    headless_config = _robot_only_config(source)
    headless_config.retargeter = RetargeterConfig(
        visualize=False,
        debug=False,
    )
    visual_config = _robot_only_config(source)
    visual_config.retargeter = RetargeterConfig(
        visualize=True,
        debug=True,
    )

    headless_job = build_retarget_job(
        headless_config,
        results_root=results_root,
    )
    visual_job = build_retarget_job(
        visual_config,
        results_root=results_root,
    )
    decoded = json.loads(visual_job.config_json)

    assert visual_job.config_json == headless_job.config_json
    assert "visualize" not in decoded["config"]["retargeter"]
    assert "debug" not in decoded["config"]["retargeter"]


def test_foot_sticking_changes_result_identity(tmp_path: Path) -> None:
    source = tmp_path / "walk.npy"
    np.save(source, np.zeros((2, 22, 3), dtype=np.float32))
    results_root = tmp_path / "results"
    enabled_config = _robot_only_config(source)
    enabled_config.retargeter = RetargeterConfig(
        activate_foot_sticking=True,
    )
    disabled_config = _robot_only_config(source)
    disabled_config.retargeter = RetargeterConfig(
        activate_foot_sticking=False,
    )

    enabled_job = build_retarget_job(
        enabled_config,
        results_root=results_root,
    )
    disabled_job = build_retarget_job(
        disabled_config,
        results_root=results_root,
    )
    enabled_payload = json.loads(enabled_job.config_json)
    disabled_payload = json.loads(disabled_job.config_json)

    assert enabled_job.output_path == disabled_job.output_path
    assert enabled_job.config_json != disabled_job.config_json
    assert enabled_payload["config"]["retargeter"]["activate_foot_sticking"] is True
    assert disabled_payload["config"]["retargeter"]["activate_foot_sticking"] is False


def test_existing_result_resumes_only_for_the_exact_saved_config(tmp_path: Path) -> None:
    source = tmp_path / "walk.npy"
    np.save(source, np.zeros((2, 22, 3), dtype=np.float32))
    job = build_retarget_job(
        _robot_only_config(source),
        results_root=tmp_path / "results",
    )
    job.output_path.parent.mkdir(parents=True)
    np.savez(
        job.output_path,
        qpos=np.zeros((2, 36), dtype=np.float32),
        config_json=np.asarray(job.config_json),
    )

    result = run_retargeting_job(job)

    assert result.resumed is True


def test_existing_result_rejects_a_stale_solver_config(tmp_path: Path) -> None:
    source = tmp_path / "walk.npy"
    np.save(source, np.zeros((2, 22, 3), dtype=np.float32))
    job = build_retarget_job(
        _robot_only_config(source),
        results_root=tmp_path / "results",
    )
    job.output_path.parent.mkdir(parents=True)
    np.savez(
        job.output_path,
        qpos=np.zeros((2, 36), dtype=np.float32),
        config_json=np.asarray('{"stale":true}'),
    )

    with pytest.raises(ValueError, match=r"different solver configuration.*--overwrite"):
        run_retargeting_job(job)


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
