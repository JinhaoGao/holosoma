# ruff: noqa: CPY001, PT009

from __future__ import annotations

import contextlib
import json
import multiprocessing
import os
import queue
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from holosoma_retargeting.config_types.data_type import DEMO_JOINTS_REGISTRY
from holosoma_retargeting.examples.run_orientation_ablation import (
    ORIENTATION_JOINTS,
    dataset_partition_identity,
    preparation_payload_sha256,
)
from holosoma_retargeting.examples.run_orientation_comparison_batch import (
    EXPERIMENT_NAME,
    INPUT_PREPARATION_FRAME_TIME_TOLERANCE_S,
    INPUT_PREPARATION_POSITION_ABS_TOLERANCE_M,
    Config,
    _build_comparison_job,
    _comparison_metrics,
    _exclusive_preparation_lock,
    _has_valid_global_orientations,
    _orientation_preparation_implementation_sha256,
    _preparation_identity,
    _prepare_orientation_inputs,
    _run_task,
    _sha256,
    _validate_bvh_merge_compatibility,
    _validate_current_result,
)
from holosoma_retargeting.examples.run_orientation_comparison_batch import (
    main as comparison_batch_main,
)
from holosoma_retargeting.retargeting_fingerprint import (
    NUMERICAL_RUNTIME_ENVIRONMENT,
)
from holosoma_retargeting.retargeting_pipeline import RetargetJobResult

NOETIX_JOINTS = tuple(DEMO_JOINTS_REGISTRY["noetix_mocap"])


def _identity_quaternions(frames: int, names: tuple[str, ...]) -> np.ndarray:
    quaternions = np.zeros((frames, len(names), 4), dtype=np.float32)
    quaternions[..., 0] = 1.0
    return quaternions


def _write_noetix_npz(
    path: Path,
    *,
    positions: np.ndarray,
    fps: float = 30.0,
    joint_names: tuple[str, ...] = NOETIX_JOINTS,
    orientation_names: tuple[str, ...] | None = None,
    orientation_source: str | None = None,
    legacy_quaternions: bool = False,
) -> None:
    payload: dict[str, np.ndarray] = {
        "global_joint_positions": np.asarray(positions, dtype=np.float32),
        "joint_names": np.asarray(joint_names, dtype=str),
        "fps": np.asarray(fps, dtype=np.float64),
        "height": np.asarray(1.78, dtype=np.float32),
        "source_format": np.asarray("noetix_mocap"),
    }
    if orientation_names is not None:
        payload["orientation_joint_names"] = np.asarray(
            orientation_names,
            dtype=str,
        )
        payload["orientation_quaternions_wxyz"] = _identity_quaternions(
            positions.shape[0],
            orientation_names,
        )
        payload["quaternion_convention"] = np.asarray("wxyz")
        payload["coordinate_system"] = np.asarray("z_up")
    if orientation_source is not None:
        payload["orientation_source"] = np.asarray(orientation_source)
    if legacy_quaternions:
        payload["global_joint_quaternions_wxyz"] = _identity_quaternions(
            positions.shape[0],
            joint_names,
        )
        payload["orientation_provenance"] = np.asarray("direct_source_bones")
    np.savez_compressed(path, **payload)


def _fake_bvh_converter(
    *,
    positions: np.ndarray,
    fps: float = 30.0,
    joint_names: tuple[str, ...] = NOETIX_JOINTS,
):
    def convert(
        *,
        bvh_path: Path,
        output_dir: Path,
        target_fps: float,
        drop_jump_threshold_m: float,
        overwrite: bool,
    ) -> None:
        del target_fps, drop_jump_threshold_m, overwrite
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_noetix_npz(
            output_dir / f"{bvh_path.stem}.npz",
            positions=positions,
            fps=fps,
            joint_names=joint_names,
            orientation_names=NOETIX_JOINTS,
            orientation_source="bvh_rotation_channels_fk",
        )

    return convert


def _validate_npz_pair(
    source_path: Path,
    converted_path: Path,
    bvh_path: Path,
) -> tuple[float, float]:
    with contextlib.ExitStack() as stack:
        source_data = stack.enter_context(np.load(source_path, allow_pickle=False))
        converted_data = stack.enter_context(np.load(converted_path, allow_pickle=False))
        return _validate_bvh_merge_compatibility(
            source_path=source_path,
            bvh_path=bvh_path,
            source_data=source_data,
            converted_data=converted_data,
        )


def _concurrent_prepare_worker(
    paths: tuple[str, str, str],
    coordination,
) -> None:
    source_path, bvh_root, output_root = paths
    start_event, ready_queue, result_queue = coordination
    ready_queue.put(True)
    if not start_event.wait(timeout=15.0):
        result_queue.put(("error", "start timeout"))
        return
    try:
        prepared_paths, records = _prepare_orientation_inputs(
            source_paths=[Path(source_path)],
            bvh_root=Path(bvh_root),
            output_root=Path(output_root),
        )
        with np.load(prepared_paths[0], allow_pickle=False) as prepared:
            result_queue.put(
                (
                    "ok",
                    records[0]["status"],
                    str(prepared_paths[0]),
                    tuple(prepared["global_joint_positions"].shape),
                    str(np.asarray(prepared["preparation_identity_sha256"]).item()),
                )
            )
    except Exception as error:
        result_queue.put(("error", f"{type(error).__name__}: {error}"))


class OrientationComparisonBatchTest(unittest.TestCase):
    def test_main_pins_worker_environment_before_building_jobs(self) -> None:
        frames = 2
        positions = np.zeros(
            (frames, len(NOETIX_JOINTS), 3),
            dtype=np.float32,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_path = root / "data"
            data_path.mkdir()
            _write_noetix_npz(
                data_path / "sequence.npz",
                positions=positions,
                orientation_names=NOETIX_JOINTS,
                orientation_source="bvh_rotation_channels_fk",
            )
            observed_environments: list[dict[str, str | None]] = []

            def recording_builder(**kwargs):
                observed_environments.append({name: os.environ.get(name) for name in NUMERICAL_RUNTIME_ENVIRONMENT})
                return _build_comparison_job(**kwargs)

            with patch.dict(
                os.environ,
                dict.fromkeys(NUMERICAL_RUNTIME_ENVIRONMENT, "7"),
            ):
                with patch(
                    "holosoma_retargeting.examples.run_orientation_comparison_batch._build_comparison_job",
                    side_effect=recording_builder,
                ):
                    comparison_batch_main(
                        Config(
                            data_path=data_path,
                            bvh_path=root / "bvh",
                            output_root=root / "results",
                            max_workers=1,
                            dry_run=True,
                        )
                    )
                self.assertEqual(
                    {name: os.environ.get(name) for name in NUMERICAL_RUNTIME_ENVIRONMENT},
                    dict.fromkeys(NUMERICAL_RUNTIME_ENVIRONMENT, "1"),
                )

        self.assertEqual(len(observed_environments), 2)
        self.assertTrue(
            all(
                environment == dict.fromkeys(NUMERICAL_RUNTIME_ENVIRONMENT, "1")
                for environment in observed_environments
            )
        )

    def test_batch_paths_preserve_special_names_and_dataset_identity(
        self,
    ) -> None:
        frames = 2
        positions = np.zeros(
            (frames, len(NOETIX_JOINTS), 3),
            dtype=np.float32,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_root = root / "results"
            data_roots = []
            for parent, task_name in (
                ("source_a", "walk+a"),
                ("source_b", "walk a"),
            ):
                data_root = root / parent / "motions"
                data_root.mkdir(parents=True)
                data_roots.append(data_root)
                _write_noetix_npz(
                    data_root / f"{task_name}.npz",
                    positions=positions,
                    orientation_names=NOETIX_JOINTS,
                    orientation_source=("bvh_rotation_channels_fk"),
                )
                comparison_batch_main(
                    Config(
                        data_path=data_root,
                        bvh_path=root / "bvh",
                        output_root=output_root,
                        max_workers=1,
                        dry_run=True,
                    )
                )

            self.assertNotEqual(
                dataset_partition_identity(data_roots[0]),
                dataset_partition_identity(data_roots[1]),
            )
            report_paths = sorted(output_root.rglob("batch_summary.json"))
            self.assertEqual(len(report_paths), 2)
            reports = [json.loads(path.read_text(encoding="utf-8")) for path in report_paths]
            baseline_paths = {Path(report["planned_tasks"][0]["baseline_path"]) for report in reports}
            self.assertEqual(len(baseline_paths), 2)
            encoded_parts = {part for path in baseline_paths for part in path.parts}
            self.assertIn("walk%2Ba", encoded_parts)
            self.assertIn("walk%20a", encoded_parts)

    def test_fail_fast_bounds_submission_and_finalizes_report(
        self,
    ) -> None:
        frames = 2
        positions = np.zeros(
            (frames, len(NOETIX_JOINTS), 3),
            dtype=np.float32,
        )

        class FakeFuture:
            def __init__(self, error: Exception | None = None):
                self.error = error
                self.was_cancelled = False

            def result(self):
                if self.error is not None:
                    raise self.error
                raise AssertionError("A cancelled fake future must not run")

            def cancel(self):
                self.was_cancelled = True
                return True

            def cancelled(self):
                return self.was_cancelled

        class FakeExecutor:
            submitted: list[dict] = []

            def __init__(self, *args, **kwargs):
                del args, kwargs

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                del exc_type, exc, traceback
                return False

            def submit(self, function, task):
                del function
                self.submitted.append(task)
                error = RuntimeError("first task failed") if len(self.submitted) == 1 else None
                return FakeFuture(error)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_path = root / "data"
            data_path.mkdir()
            for index in range(5):
                _write_noetix_npz(
                    data_path / f"sequence_{index}.npz",
                    positions=positions,
                    orientation_names=NOETIX_JOINTS,
                    orientation_source=("bvh_rotation_channels_fk"),
                )
            FakeExecutor.submitted = []

            def preserve_inputs(*, source_paths, **kwargs):
                del kwargs
                return list(source_paths), []

            with patch(
                "holosoma_retargeting.examples.run_orientation_comparison_batch._prepare_orientation_inputs",
                side_effect=preserve_inputs,
            ), patch(
                "holosoma_retargeting.examples.run_orientation_comparison_batch.ProcessPoolExecutor",
                FakeExecutor,
            ), patch(
                "holosoma_retargeting.examples.run_orientation_comparison_batch.as_completed",
                side_effect=lambda futures: iter(tuple(futures)),
            ), pytest.raises(
                RuntimeError,
                match="1 of 5 tasks failed",
            ):
                comparison_batch_main(
                    Config(
                        data_path=data_path,
                        bvh_path=root / "bvh",
                        output_root=root / "results",
                        max_workers=2,
                        fail_fast=True,
                    )
                )

            self.assertEqual(len(FakeExecutor.submitted), 2)
            report_paths = list((root / "results").rglob("batch_summary.json"))
            self.assertEqual(len(report_paths), 1)
            report = json.loads(report_paths[0].read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["failed_tasks"], 1)
            self.assertEqual(
                report["submitted_tasks"],
                ["sequence_0", "sequence_1"],
            )
            self.assertEqual(
                report["cancelled_tasks"],
                ["sequence_1"],
            )
            self.assertEqual(
                report["not_submitted_tasks"],
                ["sequence_2", "sequence_3", "sequence_4"],
            )
            self.assertEqual(
                report["failures"][0]["task_name"],
                "sequence_0",
            )

    def test_comparison_metrics_reports_delta_and_percent(self) -> None:
        baseline = {
            "position_mean_m": 2.0,
            "position_p95_m": 4.0,
            "orientation_mean_rad": 1.0,
            "orientation_p95_rad": 2.0,
            "joint_second_difference_mean_rad": 0.5,
        }
        optimal = {
            "position_mean_m": 1.0,
            "position_p95_m": 5.0,
            "orientation_mean_rad": 0.5,
            "orientation_p95_rad": 1.0,
            "joint_second_difference_mean_rad": 0.75,
        }

        comparison = _comparison_metrics(baseline, optimal)

        self.assertEqual(comparison["position_mean_m_delta"], -1.0)
        self.assertEqual(comparison["position_mean_m_change_percent"], -50.0)
        self.assertEqual(
            comparison["joint_second_difference_mean_rad_change_percent"],
            50.0,
        )

    def test_direct_orientation_readiness_matches_production_contract(self) -> None:
        frames = 3
        positions = np.zeros((frames, len(NOETIX_JOINTS), 3), dtype=np.float32)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            valid_path = root / "valid.npz"
            _write_noetix_npz(
                valid_path,
                positions=positions,
                orientation_names=NOETIX_JOINTS,
                orientation_source="bvh_rotation_channels_fk",
            )
            self.assertTrue(_has_valid_global_orientations(valid_path, frames))

            invalid_cases = {
                "missing_source": {
                    "orientation_names": NOETIX_JOINTS,
                },
                "unapproved_source": {
                    "orientation_names": NOETIX_JOINTS,
                    "orientation_source": "estimated_from_positions",
                },
                "missing_required_joint": {
                    "orientation_names": tuple(name for name in NOETIX_JOINTS if name != "LeftHand"),
                    "orientation_source": "bvh_rotation_channels_fk",
                },
                "legacy_only": {
                    "legacy_quaternions": True,
                },
            }
            for case_name, options in invalid_cases.items():
                with self.subTest(case_name=case_name):
                    case_path = root / f"{case_name}.npz"
                    _write_noetix_npz(
                        case_path,
                        positions=positions,
                        **options,
                    )
                    self.assertFalse(_has_valid_global_orientations(case_path, frames))

    def test_legacy_orientation_only_is_replaced_from_matching_bvh(self) -> None:
        frames = 4
        positions = np.zeros((frames, len(NOETIX_JOINTS), 3), dtype=np.float32)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_path = root / "source" / "sequence.npz"
            source_path.parent.mkdir()
            _write_noetix_npz(
                source_path,
                positions=positions,
                legacy_quaternions=True,
            )
            bvh_root = root / "bvh"
            bvh_root.mkdir()
            (bvh_root / "sequence.bvh").write_text(
                "matching source placeholder",
                encoding="utf-8",
            )
            fake_converter = _fake_bvh_converter(positions=positions)

            with patch(
                "holosoma_retargeting.examples.run_orientation_comparison_batch.convert_file",
                side_effect=fake_converter,
            ) as convert_mock:
                prepared_paths, records = _prepare_orientation_inputs(
                    source_paths=[source_path],
                    bvh_root=bvh_root,
                    output_root=root / "prepared",
                )

            convert_mock.assert_called_once()
            self.assertEqual(records[0]["orientation_source"], "converted_bvh")
            self.assertTrue(_has_valid_global_orientations(prepared_paths[0], frames))
            with np.load(prepared_paths[0], allow_pickle=False) as prepared:
                self.assertNotIn(
                    "global_joint_quaternions_wxyz",
                    prepared.files,
                )
                self.assertEqual(
                    str(np.asarray(prepared["orientation_source"]).item()),
                    "bvh_rotation_channels_fk",
                )

    def test_bvh_merge_rejects_elementwise_position_mismatch(self) -> None:
        frames = 4
        source_positions = np.zeros(
            (frames, len(NOETIX_JOINTS), 3),
            dtype=np.float32,
        )
        converted_positions = source_positions.copy()
        converted_positions[-1, -1, -1] = INPUT_PREPARATION_POSITION_ABS_TOLERANCE_M * 2.0
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_path = root / "sequence.npz"
            converted_path = root / "sequence.converted.npz"
            bvh_path = root / "sequence.bvh"
            _write_noetix_npz(source_path, positions=source_positions)
            _write_noetix_npz(converted_path, positions=converted_positions)

            with pytest.raises(
                ValueError,
                match=r"positions differ elementwise: mean=.*max=.*hard tolerance=",
            ):
                _validate_npz_pair(source_path, converted_path, bvh_path)

    def test_bvh_merge_rejects_joint_order_and_frame_time_mismatch(self) -> None:
        frames = 120
        positions = np.zeros(
            (frames, len(NOETIX_JOINTS), 3),
            dtype=np.float32,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_path = root / "sequence.npz"
            converted_path = root / "sequence.converted.npz"
            bvh_path = root / "sequence.bvh"
            _write_noetix_npz(source_path, positions=positions)
            swapped_names = (
                NOETIX_JOINTS[1],
                NOETIX_JOINTS[0],
                *NOETIX_JOINTS[2:],
            )
            _write_noetix_npz(
                converted_path,
                positions=positions,
                joint_names=swapped_names,
            )
            with pytest.raises(ValueError, match="joint_names/order"):
                _validate_npz_pair(source_path, converted_path, bvh_path)

            time_drift_fps = 30.0 - (2.0 * INPUT_PREPARATION_FRAME_TIME_TOLERANCE_S * 30.0**2 / (frames - 1))
            _write_noetix_npz(
                converted_path,
                positions=positions,
                fps=time_drift_fps,
            )
            with pytest.raises(
                ValueError,
                match=r"frame times differ: mean=.*max=.*hard tolerance=",
            ):
                _validate_npz_pair(source_path, converted_path, bvh_path)

    def test_current_result_validation_accepts_complete_schema(self) -> None:
        frames = 2
        joints = len(ORIENTATION_JOINTS)
        with tempfile.TemporaryDirectory() as temporary_directory:
            result_path = Path(temporary_directory) / "result.npz"
            np.savez(
                result_path,
                qpos=np.zeros((frames, 30)),
                human_joints=np.zeros((frames, 25, 3)),
                mapped_human_joints=np.zeros((frames, 15, 3)),
                mapped_robot_joints=np.zeros((frames, 15, 3)),
                orientation_tracking_enabled=np.asarray(True),
                orientation_diagnostics_enabled=np.asarray(True),
                orientation_human_joint_names=np.asarray(
                    ORIENTATION_JOINTS,
                    dtype=str,
                ),
                orientation_robot_link_names=np.asarray(
                    [f"link_{index}" for index in range(joints)],
                    dtype=str,
                ),
                orientation_weights=np.full(joints, 0.085),
                orientation_alignment_mode=np.asarray("t_pose"),
                orientation_alignment_quaternions_wxyz=np.tile(
                    [1.0, 0.0, 0.0, 0.0],
                    (joints, 1),
                ),
                orientation_reference_human_quaternions_wxyz=np.tile(
                    [1.0, 0.0, 0.0, 0.0],
                    (joints, 1),
                ),
                orientation_reference_robot_quaternions_wxyz=np.tile(
                    [1.0, 0.0, 0.0, 0.0],
                    (joints, 1),
                ),
                orientation_reference_robot_qpos=np.zeros(30),
                orientation_target_quaternions_wxyz=np.tile(
                    [1.0, 0.0, 0.0, 0.0],
                    (frames, joints, 1),
                ),
                orientation_robot_quaternions_wxyz=np.tile(
                    [1.0, 0.0, 0.0, 0.0],
                    (frames, joints, 1),
                ),
                orientation_errors_rad=np.zeros((frames, joints)),
                orientation_frame_costs=np.zeros(frames),
            )

            _validate_current_result(
                result_path,
                expected_frames=frames,
                expected_weight=0.085,
                tracking_enabled=True,
            )

    def test_comparison_worker_uses_shared_runner_for_both_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_dir = root / "inputs"
            input_dir.mkdir()
            input_path = input_dir / "sequence.npz"
            np.savez_compressed(
                input_path,
                global_joint_positions=np.zeros((2, 22, 3), dtype=np.float32),
            )
            results_root = root / "results"
            baseline_job = _build_comparison_job(
                input_path=input_path,
                results_root=results_root,
                dataset_partition="dataset",
                sequence_key="sequence",
                profile_name="baseline",
                orientation_weights=dict.fromkeys(ORIENTATION_JOINTS, 0.0),
                overwrite_existing=False,
            )
            optimal_job = _build_comparison_job(
                input_path=input_path,
                results_root=results_root,
                dataset_partition="dataset",
                sequence_key="sequence",
                profile_name="balanced_optimal",
                orientation_weights=dict.fromkeys(ORIENTATION_JOINTS, 0.085),
                overwrite_existing=False,
            )
            jobs = []

            def fake_runner(job):
                jobs.append(job)
                return RetargetJobResult(
                    output_path=job.output_path,
                    source_path=job.source_path,
                    sequence_key=job.sequence_key,
                    variant=job.variant.name,
                )

            metrics = {
                "frames": 2,
                "position_mean_m": 1.0,
                "position_p95_m": 1.0,
                "orientation_mean_rad": 1.0,
                "orientation_p95_rad": 1.0,
                "joint_second_difference_mean_rad": 1.0,
            }
            with patch(
                "holosoma_retargeting.examples.run_orientation_comparison_batch.run_retargeting_job",
                side_effect=fake_runner,
            ), patch("holosoma_retargeting.examples.run_orientation_comparison_batch._validate_current_result"), patch(
                "holosoma_retargeting.examples.run_orientation_comparison_batch._result_metrics",
                return_value=metrics,
            ):
                record = _run_task(
                    {
                        "task_name": "sequence",
                        "expected_frames": 2,
                        "input_sha256": "source-hash",
                        "log_path": str(root / "logs" / "sequence.log"),
                        "orientation_weight": 0.085,
                        "overwrite_optimal": False,
                        "overwrite_baseline": False,
                        "baseline_job": baseline_job,
                        "optimal_job": optimal_job,
                    }
                )

            self.assertEqual(jobs, [baseline_job, optimal_job])
            for job, profile in zip(
                jobs,
                ("baseline", "balanced_optimal"),
                strict=True,
            ):
                self.assertEqual(job.run_kind, "ablation")
                self.assertEqual(job.experiment_name, EXPERIMENT_NAME)
                self.assertEqual(job.variant.name, profile)
                self.assertIn("ablations", job.output_path.parts)
                self.assertIn("canonical", job.baseline_path.parts)
            self.assertEqual(
                record["baseline_path"],
                str(baseline_job.output_path),
            )
            self.assertEqual(
                record["balanced_optimal_path"],
                str(optimal_job.output_path),
            )
            self.assertEqual(
                record["canonical_baseline_path"],
                str(baseline_job.baseline_path),
            )

    def test_prepared_orientation_cache_is_bound_to_source_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_root = root / "source"
            source_root.mkdir()
            source_path = source_root / "sequence.npz"

            def write_source(position_value: float) -> None:
                _write_noetix_npz(
                    source_path,
                    positions=np.full(
                        (3, len(NOETIX_JOINTS), 3),
                        position_value,
                        dtype=np.float32,
                    ),
                    orientation_names=NOETIX_JOINTS,
                    orientation_source="bvh_rotation_channels_fk",
                )

            write_source(0.0)
            first_paths, first_records = _prepare_orientation_inputs(
                source_paths=[source_path],
                bvh_root=root / "bvh",
                output_root=root / "prepared",
            )
            second_paths, second_records = _prepare_orientation_inputs(
                source_paths=[source_path],
                bvh_root=root / "bvh",
                output_root=root / "prepared",
            )

            self.assertEqual(first_paths, second_paths)
            self.assertEqual(first_records[0]["status"], "prepared")
            self.assertEqual(second_records[0]["status"], "reused")

            with np.load(first_paths[0], allow_pickle=False) as prepared:
                tampered = {key: np.asarray(prepared[key]) for key in prepared.files}
            tampered["global_joint_positions"] = np.full_like(
                tampered["global_joint_positions"],
                7.0,
            )
            tampered["fps"] = np.asarray(1.0, dtype=np.float64)
            tampered["preparation_payload_sha256"] = np.asarray(preparation_payload_sha256(tampered))
            np.savez_compressed(first_paths[0], **tampered)
            repaired_paths, repaired_records = _prepare_orientation_inputs(
                source_paths=[source_path],
                bvh_root=root / "bvh",
                output_root=root / "prepared",
            )
            self.assertEqual(repaired_paths, first_paths)
            self.assertEqual(
                repaired_records[0]["status"],
                "prepared",
            )
            with np.load(
                repaired_paths[0],
                allow_pickle=False,
            ) as repaired:
                np.testing.assert_array_equal(
                    repaired["global_joint_positions"],
                    np.zeros(
                        (3, len(NOETIX_JOINTS), 3),
                        dtype=np.float32,
                    ),
                )
                self.assertEqual(float(repaired["fps"]), 30.0)
                self.assertIn(
                    str(source_path.resolve()),
                    str(repaired["preparation_lineage_json"]),
                )

            write_source(1.0)
            third_paths, third_records = _prepare_orientation_inputs(
                source_paths=[source_path],
                bvh_root=root / "bvh",
                output_root=root / "prepared",
            )

            self.assertNotEqual(third_paths, first_paths)
            self.assertEqual(third_records[0]["status"], "prepared")
            self.assertNotEqual(
                third_records[0]["source_sha256"],
                first_records[0]["source_sha256"],
            )
            with np.load(third_paths[0], allow_pickle=False) as prepared:
                np.testing.assert_array_equal(
                    prepared["global_joint_positions"],
                    np.ones(
                        (3, len(NOETIX_JOINTS), 3),
                        dtype=np.float32,
                    ),
                )

    def test_same_preparation_identity_is_serialized_across_processes(self) -> None:
        frames = 4
        positions = np.zeros(
            (frames, len(NOETIX_JOINTS), 3),
            dtype=np.float32,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_path = root / "source" / "sequence.npz"
            source_path.parent.mkdir()
            _write_noetix_npz(source_path, positions=positions)
            bvh_root = root / "bvh"
            bvh_root.mkdir()
            bvh_path = bvh_root / "sequence.bvh"
            bvh_path.write_text("matching source placeholder", encoding="utf-8")
            output_root = root / "prepared"
            implementation_sha256 = _orientation_preparation_implementation_sha256()
            preparation_identity = _preparation_identity(
                source_sha256=_sha256(source_path),
                bvh_sha256=_sha256(bvh_path),
                implementation_sha256=implementation_sha256,
                orientation_source="converted_bvh",
            )
            process_context = multiprocessing.get_context("fork")
            start_event = process_context.Event()
            ready_queue = process_context.Queue()
            result_queue = process_context.Queue()
            processes = [
                process_context.Process(
                    target=_concurrent_prepare_worker,
                    args=(
                        (
                            str(source_path),
                            str(bvh_root),
                            str(output_root),
                        ),
                        (start_event, ready_queue, result_queue),
                    ),
                )
                for _ in range(2)
            ]
            fake_converter = _fake_bvh_converter(positions=positions)
            try:
                with patch(
                    "holosoma_retargeting.examples.run_orientation_comparison_batch.convert_file",
                    side_effect=fake_converter,
                ):
                    for process in processes:
                        process.start()
                    for _ in processes:
                        self.assertTrue(ready_queue.get(timeout=15.0))
                    with _exclusive_preparation_lock(
                        output_root / "_inputs",
                        preparation_identity,
                    ):
                        start_event.set()
                        with pytest.raises(queue.Empty):
                            result_queue.get(timeout=1.0)
                    results = [result_queue.get(timeout=15.0) for _ in processes]
                    for process in processes:
                        process.join(timeout=15.0)
            finally:
                start_event.set()
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=5.0)

            self.assertEqual([process.exitcode for process in processes], [0, 0])
            self.assertTrue(all(result[0] == "ok" for result in results), results)
            self.assertCountEqual(
                [result[1] for result in results],
                ["prepared", "reused"],
            )
            self.assertEqual(
                {result[2] for result in results},
                {str(output_root / "_inputs" / preparation_identity / source_path.name)},
            )
            self.assertEqual(
                {result[3] for result in results},
                {(frames, len(NOETIX_JOINTS), 3)},
            )
            self.assertEqual(
                {result[4] for result in results},
                {preparation_identity},
            )
            prepared_path = Path(results[0][2])
            with np.load(prepared_path, allow_pickle=False) as prepared:
                self.assertEqual(
                    str(np.asarray(prepared["preparation_identity_sha256"]).item()),
                    preparation_identity,
                )
                np.testing.assert_array_equal(
                    prepared["global_joint_positions"],
                    positions,
                )
            self.assertEqual(
                list(prepared_path.parent.glob(f".{prepared_path.name}.*.tmp")),
                [],
            )


if __name__ == "__main__":
    unittest.main()
