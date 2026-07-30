# ruff: noqa: CPY001, E402, SIM117

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "holosoma_retargeting"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from holosoma_retargeting.config_types.data_type import MotionDataConfig
from holosoma_retargeting.config_types.retargeter import RetargeterConfig
from holosoma_retargeting.config_types.retargeting import RetargetingConfig
from holosoma_retargeting.config_types.robot import RobotConfig
from holosoma_retargeting.retargeting_fingerprint import (
    _implementation_source_paths,
    fingerprint_tree,
)
from holosoma_retargeting.retargeting_pipeline import (
    RetargetJobResult,
    RetargetVariant,
    _publish_solved_artifact,
    _verify_job_source,
    build_retarget_job,
    normalize_retargeting_config,
    resolve_task_object_name,
    run_retargeting_job,
    validate_config,
)


def _robot_only_config(
    *,
    data_format: str,
    source: Path,
    robot_urdf: Path,
    human_height: float | None = 1.75,
) -> RetargetingConfig:
    return RetargetingConfig(
        task_type="robot_only",
        robot="g1",
        data_format=data_format,
        task_name=source.stem,
        data_path=source.parent,
        robot_config=RobotConfig(
            robot_type="g1",
            robot_urdf_file=str(robot_urdf),
        ),
        motion_data_config=MotionDataConfig(
            data_format=data_format,
            robot_type="g1",
            human_height=human_height,
        ),
    )


def test_robot_type_normalization_preserves_nested_robot_overrides(
    tmp_path: Path,
) -> None:
    robot_config = RobotConfig(
        robot_type="g1",
        robot_dof=17,
        robot_height=9.25,
        robot_name="custom-platform",
        robot_urdf_file="/custom/platform.urdf",
        foot_sticking_links=["left_custom", "right_custom"],
        manual_lb={"7": -0.25},
        manual_ub={"7": 0.25},
        manual_cost={"7": 2.0},
        nominal_tracking_indices=np.asarray([0, 3, 5]),
    )
    config = RetargetingConfig(
        task_type="robot_only",
        robot="e1",
        data_format="amass",
        task_name="motion",
        data_path=tmp_path,
        robot_config=robot_config,
        motion_data_config=MotionDataConfig(
            data_format="amass",
            robot_type="g1",
        ),
    )

    normalized = normalize_retargeting_config(config)

    assert normalized.robot_config.robot_type == "e1"
    assert normalized.robot_config.robot_dof == robot_config.robot_dof
    assert normalized.robot_config.robot_height == robot_config.robot_height
    assert normalized.robot_config.robot_name == robot_config.robot_name
    assert normalized.robot_config.robot_urdf_file == robot_config.robot_urdf_file
    assert normalized.robot_config.foot_sticking_links == robot_config.foot_sticking_links
    assert normalized.robot_config.manual_lb == robot_config.manual_lb
    assert normalized.robot_config.manual_ub == robot_config.manual_ub
    assert normalized.robot_config.manual_cost == robot_config.manual_cost
    np.testing.assert_array_equal(
        normalized.robot_config.nominal_tracking_indices,
        robot_config.nominal_tracking_indices,
    )
    assert normalized.robot_config.robot_defaults == robot_config.robot_defaults


def test_recursive_single_action_identity_matches_explicit_batch_identity(
    tmp_path: Path,
) -> None:
    robot_dir = tmp_path / "robot"
    robot_dir.mkdir()
    robot_urdf = robot_dir / "robot.urdf"
    robot_urdf.write_text("<robot name='test'/>", encoding="utf-8")
    data_root = tmp_path / "noetix"
    sequence_dir = data_root / "session"
    sequence_dir.mkdir(parents=True)
    source = sequence_dir / "clip.npz"
    np.savez(
        source,
        source_format=np.asarray("noetix_mocap"),
    )
    config = _robot_only_config(
        data_format="noetix_mocap",
        source=source,
        robot_urdf=robot_urdf,
    )
    config.data_path = data_root
    explicit_key = source.relative_to(data_root).with_suffix("").as_posix()

    single_job = build_retarget_job(
        config,
        results_root=tmp_path / "results",
    )
    batch_job = build_retarget_job(
        config,
        results_root=tmp_path / "results",
        source_path=source,
        sequence_key=explicit_key,
    )

    assert single_job.sequence_key == "session/clip"
    assert single_job.sequence_key == batch_job.sequence_key
    assert single_job.output_path == batch_job.output_path
    assert single_job.config_json == batch_job.config_json
    assert single_job.config_sha256 == batch_job.config_sha256


def test_climbing_single_action_keeps_batch_directory_identity(
    tmp_path: Path,
) -> None:
    robot_dir = tmp_path / "robot"
    robot_dir.mkdir()
    robot_urdf = robot_dir / "robot.urdf"
    robot_urdf.write_text("<robot name='test'/>", encoding="utf-8")
    data_root = tmp_path / "climbing"
    sequence_dir = data_root / "wall_route"
    sequence_dir.mkdir(parents=True)
    source = sequence_dir / "global_positions.npy"
    np.save(source, np.zeros((1, 1), dtype=np.float64))
    config = RetargetingConfig(
        task_type="climbing",
        robot="g1",
        data_format="mocap",
        task_name=sequence_dir.name,
        data_path=data_root,
        robot_config=RobotConfig(
            robot_type="g1",
            robot_urdf_file=str(robot_urdf),
        ),
        motion_data_config=MotionDataConfig(
            data_format="mocap",
            robot_type="g1",
            human_height=1.75,
        ),
    )
    explicit_key = source.parent.relative_to(data_root).as_posix()

    single_job = build_retarget_job(
        config,
        results_root=tmp_path / "results",
    )
    batch_job = build_retarget_job(
        config,
        results_root=tmp_path / "results",
        source_path=source,
        sequence_key=explicit_key,
    )

    assert single_job.sequence_key == "wall_route"
    assert single_job.sequence_key == batch_job.sequence_key
    assert single_job.output_path == batch_job.output_path
    assert single_job.config_json == batch_job.config_json
    assert single_job.config_sha256 == batch_job.config_sha256


def test_implementation_manifest_contains_only_explicit_python_sources() -> None:
    paths = _implementation_source_paths()

    assert paths
    assert all(path.suffix == ".py" for path in paths.values())
    assert not any("__pycache__" in path.parts for path in paths.values())
    assert "retargeting_pipeline.py" in paths
    assert "src/mujoco_utils.py" in paths


def test_tree_fingerprint_rehashes_changed_bytes_in_one_process(
    tmp_path: Path,
) -> None:
    dependency = tmp_path / "asset.bin"
    dependency.write_bytes(b"first")
    first = fingerprint_tree(str(dependency))

    dependency.write_bytes(b"second-version")
    second = fingerprint_tree(str(dependency))

    assert first.sha256 != second.sha256


def test_job_identity_binds_robot_asset_tree_and_runtime(
    tmp_path: Path,
) -> None:
    robot_dir = tmp_path / "robot"
    robot_dir.mkdir()
    robot_urdf = robot_dir / "robot.urdf"
    robot_urdf.write_text("<robot name='test'/>", encoding="utf-8")
    mesh = robot_dir / "mesh.obj"
    mesh.write_text("v 0 0 0\n", encoding="utf-8")
    source = tmp_path / "motion.npz"
    source.touch()
    config = _robot_only_config(
        data_format="amass",
        source=source,
        robot_urdf=robot_urdf,
    )

    first = build_retarget_job(
        config,
        results_root=tmp_path / "results",
        source_path=source,
    )
    first_identity = json.loads(first.config_json)["solver_identity"]
    mesh.write_text("v 1 0 0\nv 0 1 0\n", encoding="utf-8")
    second = build_retarget_job(
        config,
        results_root=tmp_path / "results",
        source_path=source,
    )
    second_identity = json.loads(second.config_json)["solver_identity"]

    assert first.config_sha256 != second.config_sha256
    assert first_identity["dependencies"]["sha256"] != second_identity["dependencies"]["sha256"]
    assert first_identity["implementation_sha256"]
    assert first_identity["runtime_sha256"]
    assert first_identity["runtime_versions"]["mujoco"]


def test_job_identity_binds_omomo_height_table(
    tmp_path: Path,
) -> None:
    robot_dir = tmp_path / "robot"
    robot_dir.mkdir()
    robot_urdf = robot_dir / "robot.urdf"
    robot_urdf.write_text("<robot name='test'/>", encoding="utf-8")
    data_dir = tmp_path / "OMOMO_new"
    data_dir.mkdir()
    source = data_dir / "sub1_tripod_001.pt"
    source.touch()
    height_table = tmp_path / "height_dict.pkl"
    height_table.write_bytes(b"height-version-one")
    config = _robot_only_config(
        data_format="omomo",
        source=source,
        robot_urdf=robot_urdf,
        human_height=None,
    )

    first = build_retarget_job(
        config,
        results_root=tmp_path / "results",
        source_path=source,
    )
    height_table.write_bytes(b"height-version-two")
    second = build_retarget_job(
        config,
        results_root=tmp_path / "results",
        source_path=source,
    )

    assert first.config_sha256 != second.config_sha256
    entries = json.loads(second.config_json)["solver_identity"]["dependencies"]["entries"]
    assert entries["omomo_height_table"]["path"] == str(height_table)


def test_run_rejects_source_bytes_changed_after_planning(
    tmp_path: Path,
) -> None:
    robot_dir = tmp_path / "robot"
    robot_dir.mkdir()
    robot_urdf = robot_dir / "robot.urdf"
    robot_urdf.write_text("<robot name='test'/>", encoding="utf-8")
    source = tmp_path / "motion.npz"
    source.write_bytes(b"planned")
    job = build_retarget_job(
        _robot_only_config(
            data_format="amass",
            source=source,
            robot_urdf=robot_urdf,
        ),
        results_root=tmp_path / "results",
        source_path=source,
    )
    source.write_bytes(b"changed")

    with mock.patch("holosoma_retargeting.retargeting_pipeline._run_retargeting_job_unlocked") as runner:
        with pytest.raises(
            RuntimeError,
            match="source bytes changed",
        ):
            run_retargeting_job(job)

    runner.assert_not_called()


def test_run_rejects_source_mutation_during_adapter_load(
    tmp_path: Path,
) -> None:
    robot_dir = tmp_path / "robot"
    robot_dir.mkdir()
    robot_urdf = robot_dir / "robot.urdf"
    robot_urdf.write_text("<robot name='test'/>", encoding="utf-8")
    source = tmp_path / "motion.npz"
    source.write_bytes(b"planned")
    job = build_retarget_job(
        _robot_only_config(
            data_format="amass",
            source=source,
            robot_urdf=robot_urdf,
        ),
        results_root=tmp_path / "results",
        source_path=source,
    )

    def mutate_source(*args, **kwargs):
        del args, kwargs
        source.write_bytes(b"changed")
        return mock.Mock()

    with mock.patch(
        "holosoma_retargeting.retargeting_pipeline.load_human_motion",
        side_effect=mutate_source,
    ), mock.patch(
        "holosoma_retargeting.retargeting_pipeline.create_task_constants",
    ) as create_constants:
        with pytest.raises(RuntimeError, match="source bytes changed"):
            run_retargeting_job(job)

    create_constants.assert_not_called()


def test_staging_and_formal_jobs_share_persistent_lock_identity(
    tmp_path: Path,
) -> None:
    robot_dir = tmp_path / "robot"
    robot_dir.mkdir()
    robot_urdf = robot_dir / "robot.urdf"
    robot_urdf.write_text("<robot name='test'/>", encoding="utf-8")
    source = tmp_path / "motion.npz"
    source.touch()
    config = _robot_only_config(
        data_format="amass",
        source=source,
        robot_urdf=robot_urdf,
    )

    formal_job = build_retarget_job(
        config,
        results_root=tmp_path / "results" / "v1",
        source_path=source,
    )
    staging_job = build_retarget_job(
        config,
        results_root=(tmp_path / "results" / ".staging" / "demo-rebuild-v1" / "v1"),
        source_path=source,
    )

    assert staging_job.output_lock_path == formal_job.output_lock_path
    assert staging_job.baseline_lock_path == formal_job.baseline_lock_path
    assert staging_job.state_lock_path == formal_job.state_lock_path


def test_identity_variant_name_is_reserved_for_no_transform() -> None:
    with pytest.raises(ValueError, match="reserved 'identity'"):
        RetargetVariant(
            name="identity",
            translation=(0.1, 0.0, 0.0),
        )


@pytest.mark.parametrize(
    ("task_type", "object_name", "message"),
    [
        ("robot_only", "tripod", "ground"),
        ("climbing", "tripod", "multi_boxes"),
    ],
)
def test_task_object_contract_is_rejected_during_normalization(
    task_type: str,
    object_name: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_task_object_name(
            task_type,
            "amass",
            "sequence",
            object_name,
        )


@pytest.mark.parametrize(
    ("run_kind", "variant", "experiment_name", "message"),
    [
        (
            "single",
            RetargetVariant(name="named-zero"),
            None,
            "exact identity",
        ),
        (
            "augmentation",
            RetargetVariant(name="named-zero"),
            None,
            "non-identity",
        ),
        (
            "augmentation",
            RetargetVariant(
                name="translated",
                translation=(0.1, 0.0, 0.0),
            ),
            "unexpected",
            "must not set experiment_name",
        ),
        (
            "ablation",
            RetargetVariant(
                name="translated",
                translation=(0.1, 0.0, 0.0),
            ),
            "orientation",
            "must not transform",
        ),
        (
            "ablation",
            RetargetVariant(name="root-only"),
            None,
            "require experiment_name",
        ),
    ],
)
def test_build_rejects_ambiguous_run_kind_variant_combinations(
    tmp_path: Path,
    run_kind: str,
    variant: RetargetVariant,
    experiment_name: str | None,
    message: str,
) -> None:
    robot_urdf = tmp_path / "robot.urdf"
    robot_urdf.write_text("<robot name='test'/>", encoding="utf-8")
    source = tmp_path / "motion.npz"
    source.touch()
    with pytest.raises(ValueError, match=message):
        build_retarget_job(
            _robot_only_config(
                data_format="amass",
                source=source,
                robot_urdf=robot_urdf,
            ),
            results_root=tmp_path / "results",
            source_path=source,
            run_kind=run_kind,
            variant=variant,
            experiment_name=experiment_name,
        )


def test_run_rejects_static_dependency_same_size_with_restored_mtime(
    tmp_path: Path,
) -> None:
    robot_dir = tmp_path / "robot"
    robot_dir.mkdir()
    robot_urdf = robot_dir / "robot.urdf"
    robot_urdf.write_text("<robot name='test'/>", encoding="utf-8")
    mesh = robot_dir / "mesh.obj"
    mesh.write_bytes(b"first-bytes")
    original_stat = mesh.stat()
    source = tmp_path / "motion.npz"
    source.touch()
    job = build_retarget_job(
        _robot_only_config(
            data_format="amass",
            source=source,
            robot_urdf=robot_urdf,
        ),
        results_root=tmp_path / "results",
        source_path=source,
    )

    mesh.write_bytes(b"other-bytes")
    os.utime(
        mesh,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    with mock.patch(
        "holosoma_retargeting.retargeting_pipeline._run_retargeting_job_unlocked",
    ) as runner, pytest.raises(RuntimeError, match="static dependency"):
        run_retargeting_job(job)
    runner.assert_not_called()


def test_run_rejects_numerical_runtime_environment_changed_after_planning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robot_urdf = tmp_path / "robot.urdf"
    robot_urdf.write_text("<robot name='test'/>", encoding="utf-8")
    source = tmp_path / "motion.npz"
    source.touch()
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    job = build_retarget_job(
        _robot_only_config(
            data_format="amass",
            source=source,
            robot_urdf=robot_urdf,
        ),
        results_root=tmp_path / "results",
        source_path=source,
    )
    monkeypatch.setenv("OMP_NUM_THREADS", "2")

    with mock.patch(
        "holosoma_retargeting.retargeting_pipeline._run_retargeting_job_unlocked",
    ) as runner, pytest.raises(RuntimeError, match="runtime"):
        run_retargeting_job(job)
    runner.assert_not_called()


def test_transactional_publication_preserves_existing_output_on_postcheck_failure(
    tmp_path: Path,
) -> None:
    robot_dir = tmp_path / "robot"
    robot_dir.mkdir()
    robot_urdf = robot_dir / "robot.urdf"
    robot_urdf.write_text("<robot name='test'/>", encoding="utf-8")
    source = tmp_path / "motion.npz"
    source.write_bytes(b"planned")
    job = build_retarget_job(
        _robot_only_config(
            data_format="amass",
            source=source,
            robot_urdf=robot_urdf,
        ),
        results_root=tmp_path / "results",
        source_path=source,
    )
    job.output_path.parent.mkdir(parents=True)
    job.output_path.write_bytes(b"known-good-existing-result")
    temporary_output = job.output_path.parent / ".candidate.npz"
    temporary_output.write_bytes(b"candidate")
    signature = (
        source.stat().st_dev,
        source.stat().st_ino,
        source.stat().st_size,
        source.stat().st_mtime_ns,
        source.stat().st_ctime_ns,
    )
    source.write_bytes(b"mutated")

    with pytest.raises(RuntimeError, match="source bytes changed"):
        _publish_solved_artifact(
            job,
            temporary_output,
            source_signature=signature,
        )

    assert job.output_path.read_bytes() == b"known-good-existing-result"
    assert temporary_output.read_bytes() == b"candidate"


def test_publication_rechecks_source_after_candidate_validation(
    tmp_path: Path,
) -> None:
    robot_dir = tmp_path / "robot"
    robot_dir.mkdir()
    robot_urdf = robot_dir / "robot.urdf"
    robot_urdf.write_text("<robot name='test'/>", encoding="utf-8")
    source = tmp_path / "motion.npz"
    source.write_bytes(b"planned")
    job = build_retarget_job(
        _robot_only_config(
            data_format="amass",
            source=source,
            robot_urdf=robot_urdf,
        ),
        results_root=tmp_path / "results",
        source_path=source,
    )
    job.output_path.parent.mkdir(parents=True)
    candidate = job.output_path.parent / ".candidate.npz"
    candidate.write_bytes(b"candidate")
    source_signature = _verify_job_source(job)

    def mutate_source_during_validation(*_args, **_kwargs) -> bool:
        source.write_bytes(b"changed")
        return True

    with mock.patch(
        "holosoma_retargeting.retargeting_pipeline.result_artifact_matches_job",
        side_effect=mutate_source_during_validation,
    ), pytest.raises(RuntimeError, match="source bytes changed"):
        _publish_solved_artifact(
            job,
            candidate,
            source_signature=source_signature,
        )

    assert not job.output_path.exists()
    assert candidate.read_bytes() == b"candidate"


def test_publication_rechecks_solver_identity_after_candidate_validation(
    tmp_path: Path,
) -> None:
    robot_dir = tmp_path / "robot"
    robot_dir.mkdir()
    robot_urdf = robot_dir / "robot.urdf"
    robot_urdf.write_text("<robot name='planned'/>", encoding="utf-8")
    source = tmp_path / "motion.npz"
    source.write_bytes(b"planned")
    job = build_retarget_job(
        _robot_only_config(
            data_format="amass",
            source=source,
            robot_urdf=robot_urdf,
        ),
        results_root=tmp_path / "results",
        source_path=source,
    )
    job.output_path.parent.mkdir(parents=True)
    candidate = job.output_path.parent / ".candidate.npz"
    candidate.write_bytes(b"candidate")
    source_signature = _verify_job_source(job)

    def mutate_solver_dependency(*_args, **_kwargs) -> bool:
        robot_urdf.write_text("<robot name='changed'/>", encoding="utf-8")
        return True

    with mock.patch(
        "holosoma_retargeting.retargeting_pipeline.result_artifact_matches_job",
        side_effect=mutate_solver_dependency,
    ), pytest.raises(RuntimeError, match="static dependency"):
        _publish_solved_artifact(
            job,
            candidate,
            source_signature=source_signature,
        )

    assert not job.output_path.exists()
    assert candidate.read_bytes() == b"candidate"


def test_resume_rechecks_source_after_artifact_validation(
    tmp_path: Path,
) -> None:
    robot_dir = tmp_path / "robot"
    robot_dir.mkdir()
    robot_urdf = robot_dir / "robot.urdf"
    robot_urdf.write_text("<robot name='test'/>", encoding="utf-8")
    source = tmp_path / "motion.npz"
    source.write_bytes(b"planned")
    job = build_retarget_job(
        _robot_only_config(
            data_format="amass",
            source=source,
            robot_urdf=robot_urdf,
        ),
        results_root=tmp_path / "results",
        source_path=source,
    )
    job.output_path.parent.mkdir(parents=True)
    job.output_path.touch()

    def mutate_source_during_validation(*_args, **_kwargs) -> bool:
        source.write_bytes(b"changed")
        return True

    with mock.patch(
        "holosoma_retargeting.retargeting_pipeline.result_artifact_matches_job",
        side_effect=mutate_source_during_validation,
    ), mock.patch(
        "holosoma_retargeting.retargeting_pipeline._run_retargeting_job_unlocked",
    ) as runner, pytest.raises(RuntimeError, match="source bytes changed"):
        run_retargeting_job(job)

    runner.assert_not_called()


def test_resume_rechecks_solver_identity_after_artifact_validation(
    tmp_path: Path,
) -> None:
    robot_dir = tmp_path / "robot"
    robot_dir.mkdir()
    robot_urdf = robot_dir / "robot.urdf"
    robot_urdf.write_text("<robot name='planned'/>", encoding="utf-8")
    source = tmp_path / "motion.npz"
    source.write_bytes(b"planned")
    job = build_retarget_job(
        _robot_only_config(
            data_format="amass",
            source=source,
            robot_urdf=robot_urdf,
        ),
        results_root=tmp_path / "results",
        source_path=source,
    )
    job.output_path.parent.mkdir(parents=True)
    job.output_path.touch()

    def mutate_solver_dependency(*_args, **_kwargs) -> bool:
        robot_urdf.write_text("<robot name='changed'/>", encoding="utf-8")
        return True

    with mock.patch(
        "holosoma_retargeting.retargeting_pipeline.result_artifact_matches_job",
        side_effect=mutate_solver_dependency,
    ), mock.patch(
        "holosoma_retargeting.retargeting_pipeline._run_retargeting_job_unlocked",
    ) as runner, pytest.raises(RuntimeError, match="static dependency"):
        run_retargeting_job(job)

    runner.assert_not_called()


def test_canonical_job_rejects_disabling_interaction_mesh(
    tmp_path: Path,
) -> None:
    source = tmp_path / "motion.npz"
    source.touch()
    config = _robot_only_config(
        data_format="amass",
        source=source,
        robot_urdf=tmp_path / "robot.urdf",
    )
    config.retargeter = RetargeterConfig(save_interaction_mesh=False)

    with pytest.raises(ValueError, match="Interaction Mesh"):
        validate_config(config)


def test_existing_mismatched_artifact_requires_explicit_overwrite(
    tmp_path: Path,
) -> None:
    robot_dir = tmp_path / "robot"
    robot_dir.mkdir()
    robot_urdf = robot_dir / "robot.urdf"
    robot_urdf.write_text("<robot name='test'/>", encoding="utf-8")
    source = tmp_path / "motion.npz"
    source.touch()
    config = _robot_only_config(
        data_format="amass",
        source=source,
        robot_urdf=robot_urdf,
    )
    job = build_retarget_job(
        config,
        results_root=tmp_path / "results",
        source_path=source,
    )
    job.output_path.parent.mkdir(parents=True)
    job.output_path.write_bytes(b"not-an-artifact")

    with mock.patch("holosoma_retargeting.retargeting_pipeline._run_retargeting_job_unlocked") as runner:
        with pytest.raises(
            FileExistsError,
            match="--overwrite-existing",
        ):
            run_retargeting_job(job)
    runner.assert_not_called()

    overwrite_job = build_retarget_job(
        config,
        results_root=tmp_path / "results",
        source_path=source,
        overwrite_existing=True,
    )
    expected = RetargetJobResult(
        output_path=overwrite_job.output_path,
        source_path=source,
        sequence_key=source.stem,
        variant="identity",
    )
    with mock.patch(
        "holosoma_retargeting.retargeting_pipeline._run_retargeting_job_unlocked",
        return_value=expected,
    ) as runner:
        assert run_retargeting_job(overwrite_job) == expected
    runner.assert_called_once()
    assert overwrite_job.config_sha256 == job.config_sha256


def test_concurrent_identical_jobs_solve_once_under_output_lock(
    tmp_path: Path,
) -> None:
    robot_dir = tmp_path / "robot"
    robot_dir.mkdir()
    robot_urdf = robot_dir / "robot.urdf"
    robot_urdf.write_text("<robot name='test'/>", encoding="utf-8")
    source = tmp_path / "motion.npz"
    source.touch()
    job = build_retarget_job(
        _robot_only_config(
            data_format="amass",
            source=source,
            robot_urdf=robot_urdf,
        ),
        results_root=tmp_path / "results",
        source_path=source,
    )
    solve_count = 0

    def fake_solve(planned_job, *, source_signature):
        nonlocal solve_count
        del source_signature
        solve_count += 1
        time.sleep(0.1)
        planned_job.output_path.parent.mkdir(parents=True, exist_ok=True)
        planned_job.output_path.touch()
        return RetargetJobResult(
            output_path=planned_job.output_path,
            source_path=planned_job.source_path,
            sequence_key=planned_job.sequence_key,
            variant=planned_job.variant.name,
        )

    def fake_match(path, planned_job, *, identity_baseline=False):
        del planned_job, identity_baseline
        return Path(path).is_file()

    with mock.patch(
        "holosoma_retargeting.retargeting_pipeline._run_retargeting_job_unlocked",
        side_effect=fake_solve,
    ), mock.patch(
        "holosoma_retargeting.retargeting_pipeline.result_artifact_matches_job",
        side_effect=fake_match,
    ):
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = tuple(
                executor.map(
                    run_retargeting_job,
                    (job, job),
                )
            )

    assert solve_count == 1
    assert sum(result.resumed for result in results) == 1


def test_output_lock_owner_cleans_candidates_orphaned_by_process_exit(
    tmp_path: Path,
) -> None:
    robot_dir = tmp_path / "robot"
    robot_dir.mkdir()
    robot_urdf = robot_dir / "robot.urdf"
    robot_urdf.write_text("<robot name='test'/>", encoding="utf-8")
    source = tmp_path / "motion.npz"
    source.write_bytes(b"planned")
    job = build_retarget_job(
        _robot_only_config(
            data_format="amass",
            source=source,
            robot_urdf=robot_urdf,
        ),
        results_root=tmp_path / "results",
        source_path=source,
    )
    job.output_path.parent.mkdir(parents=True)
    job.output_path.touch()
    outer_candidate = (
        job.output_path.parent
        / f".{job.output_path.name}.solve.orphan.tmp.npz"
    )
    nested_candidate = (
        job.output_path.parent
        / f"..{job.output_path.name}.solve.orphan.tmp.npz.inner.tmp.npz"
    )
    unrelated = job.output_path.parent / ".unrelated.tmp.npz"
    outer_candidate.write_bytes(b"outer")
    nested_candidate.write_bytes(b"nested")
    unrelated.write_bytes(b"unrelated")

    with mock.patch(
        "holosoma_retargeting.retargeting_pipeline.result_artifact_matches_job",
        return_value=True,
    ):
        result = run_retargeting_job(job)

    assert result.resumed
    assert not outer_candidate.exists()
    assert not nested_candidate.exists()
    assert unrelated.read_bytes() == b"unrelated"
