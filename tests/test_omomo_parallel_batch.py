# ruff: noqa: CPY001, PT009, PT027

from __future__ import annotations

import json
import pickle
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "holosoma_retargeting"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from holosoma_retargeting.config_types.data_type import MotionDataConfig  # noqa: E402
from holosoma_retargeting.config_types.retargeter import RetargeterConfig  # noqa: E402
from holosoma_retargeting.config_types.retargeting import (  # noqa: E402
    ParallelRetargetingConfig,
    RetargetingConfig,
)
from holosoma_retargeting.config_types.robot import RobotConfig  # noqa: E402
from holosoma_retargeting.config_types.task import TaskConfig  # noqa: E402
from holosoma_retargeting.examples.parallel_robot_retarget import (  # noqa: E402
    BatchReportInUseError,
    _write_json_report,
    find_files,
    main,
    process_single_task,
    resolve_batch_data_dir,
    resolve_batch_object_names,
)
from holosoma_retargeting.result_artifact import (  # noqa: E402
    RESULT_SCHEMA_VERSION,
    build_object_asset_manifest,
    compute_file_sha256,
    write_result_artifact,
)
from holosoma_retargeting.retargeting_fingerprint import fingerprint_tree  # noqa: E402
from holosoma_retargeting.retargeting_pipeline import (  # noqa: E402
    IDENTITY_VARIANT,
    RetargetJobResult,
    RetargetVariant,
    build_retarget_job,
    result_artifact_matches_job,
    run_retargeting_job,
    validate_config,
)
from test_result_artifact import _dynamic_object_payload  # noqa: E402


def _write_job_artifact(job) -> RetargetJobResult:
    payload = _dynamic_object_payload()
    object_urdf = job.generated_assets_dir / "fixture-object.urdf"
    object_urdf.parent.mkdir(parents=True, exist_ok=True)
    object_urdf.write_text("<robot name='fixture'/>\n", encoding="utf-8")
    object_asset_manifest_json, object_asset_manifest_sha256 = build_object_asset_manifest(object_urdf)
    payload.update(
        {
            "run_kind": np.asarray(job.run_kind),
            "variant": np.asarray(job.variant.name),
            "dataset_partition": np.asarray(job.dataset_partition),
            "sequence_key": np.asarray(job.sequence_key),
            "experiment_name": np.asarray(job.experiment_name or ""),
            "source_path": np.asarray(str(job.source_path)),
            "source_data_format": np.asarray(job.config.data_format),
            "robot_type": np.asarray(job.config.robot),
            "task_type": np.asarray(job.config.task_type),
            "object_name": np.asarray(job.config.task_config.object_name),
            "object_urdf": np.asarray(str(object_urdf.resolve())),
            "object_urdf_sha256": np.asarray(
                compute_file_sha256(object_urdf),
            ),
            "object_asset_manifest_json": np.asarray(
                object_asset_manifest_json,
            ),
            "object_asset_manifest_sha256": np.asarray(
                object_asset_manifest_sha256,
            ),
            "source_sha256": np.asarray(job.source_sha256),
            "config_json": np.asarray(job.config_json),
            "config_sha256": np.asarray(job.config_sha256),
        }
    )
    write_result_artifact(job.output_path, payload)
    return RetargetJobResult(
        output_path=job.output_path,
        source_path=job.source_path,
        sequence_key=job.sequence_key,
        variant=job.variant.name,
    )


def _worker_args(source: Path, results_root: Path, data_root: Path) -> tuple:
    return (
        source,
        results_root,
        data_root,
        "object_interaction",
        "omomo",
        RobotConfig(robot_type="g1"),
        MotionDataConfig(
            data_format="omomo",
            robot_type="g1",
            human_height=1.75,
        ),
        TaskConfig(),
        RetargeterConfig(),
        False,
        False,
    )


def _object_job(
    source: Path,
    results_root: Path,
    *,
    variant=IDENTITY_VARIANT,
):
    task_name = source.stem
    config = RetargetingConfig(
        task_type="object_interaction",
        robot="g1",
        data_format="omomo",
        task_name=task_name,
        data_path=source.parent,
        robot_config=RobotConfig(robot_type="g1"),
        motion_data_config=MotionDataConfig(
            data_format="omomo",
            robot_type="g1",
            human_height=1.75,
        ),
        task_config=TaskConfig(),
        retargeter=RetargeterConfig(),
    )
    return build_retarget_job(
        config,
        variant=variant,
        run_kind="single" if variant.is_identity else "augmentation",
        results_root=results_root,
        dataset_partition=source.parent.name,
        sequence_key=task_name,
        source_path=source,
    )


class OmomoBatchSelectionTests(unittest.TestCase):
    def test_data_path_is_an_explicit_alias_and_conflicts_fail(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            alias = root / "alias"
            primary = root / "primary"
            primary.mkdir()
            physical_alias = root / "primary_alias"
            physical_alias.symlink_to(primary, target_is_directory=True)
            alias_cfg = ParallelRetargetingConfig(data_path=alias)
            primary_cfg = ParallelRetargetingConfig(data_dir=primary)
            same_cfg = ParallelRetargetingConfig(
                data_dir=primary,
                data_path=primary,
            )
            conflict_cfg = ParallelRetargetingConfig(
                data_dir=primary,
                data_path=alias,
            )
            physical_alias_cfg = ParallelRetargetingConfig(
                data_dir=primary,
                data_path=physical_alias,
            )

            self.assertEqual(resolve_batch_data_dir(alias_cfg), alias)
            self.assertEqual(resolve_batch_data_dir(primary_cfg), primary)
            self.assertEqual(resolve_batch_data_dir(same_cfg), primary)
            self.assertEqual(
                resolve_batch_data_dir(physical_alias_cfg),
                primary,
            )
            with self.assertRaisesRegex(ValueError, "conflicting --data-dir and --data-path"):
                resolve_batch_data_dir(conflict_cfg)

    def test_exact_multi_object_filtering(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            for task_name in (
                "sub1_largebox_001",
                "sub1_smallbox_001",
                "sub2_tripod_001",
            ):
                (data_dir / f"{task_name}.pt").touch()

            selected = find_files(
                data_dir,
                "omomo",
                ("largebox", "tripod"),
            )

        self.assertEqual(
            [Path(path).stem for path in selected],
            ["sub1_largebox_001", "sub2_tripod_001"],
        )

    def test_normalizes_filters_and_rejects_invalid_selections(self):
        self.assertEqual(
            resolve_batch_object_names(
                "object_interaction",
                "omomo",
                None,
                ("tripod", "tripod", "smallbox"),
            ),
            ("tripod", "smallbox"),
        )
        self.assertIsNone(resolve_batch_object_names("object_interaction", "omomo", None, None))
        self.assertIsNone(
            resolve_batch_object_names(
                "robot_only",
                "omomo",
                "ground",
                None,
            )
        )
        with self.assertRaisesRegex(ValueError, "Unknown OMOMO object"):
            resolve_batch_object_names(
                "object_interaction",
                "omomo",
                None,
                ("not_an_object",),
            )
        with self.assertRaisesRegex(ValueError, "conflicts"):
            resolve_batch_object_names(
                "object_interaction",
                "omomo",
                "tripod",
                ("smallbox",),
            )


class OmomoBatchExecutionTests(unittest.TestCase):
    def test_stale_completed_report_is_replaced_before_preflight_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "OMOMO_new"
            data_dir.mkdir()
            save_dir = root / "results"
            report_path = (
                save_dir
                / "runs"
                / "g1-object_interaction-omomo"
                / "report.json"
            )
            report_path.parent.mkdir(parents=True)
            report_path.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "invocation_id": "stale-invocation",
                    },
                ),
                encoding="utf-8",
            )
            orphaned_temp = (
                report_path.parent
                / ".report.json.orphaned.tmp"
            )
            orphaned_temp.write_bytes(b"partial")

            def fail_preflight(*_args, **_kwargs):
                running = json.loads(report_path.read_text(encoding="utf-8"))
                self.assertEqual(running["status"], "running")
                self.assertNotEqual(
                    running["invocation_id"],
                    "stale-invocation",
                )
                self.assertIsNone(running["finished_at"])
                raise RuntimeError("injected preflight failure")

            with mock.patch(
                "holosoma_retargeting.examples.parallel_robot_retarget.preflight_omomo_dataset",
                side_effect=fail_preflight,
            ), self.assertRaisesRegex(
                RuntimeError,
                "injected preflight failure",
            ):
                main(
                    ParallelRetargetingConfig(
                        task_type="object_interaction",
                        robot="g1",
                        data_format="omomo",
                        data_dir=data_dir,
                        save_dir=save_dir,
                    ),
                )

            failed = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertFalse(orphaned_temp.exists())

        self.assertEqual(failed["status"], "failed")
        self.assertNotEqual(failed["invocation_id"], "stale-invocation")
        self.assertEqual(failed["fatal_error"]["type"], "RuntimeError")
        self.assertEqual(
            failed["fatal_error"]["message"],
            "injected preflight failure",
        )
        self.assertIsNotNone(failed["finished_at"])

    def test_base_exceptions_are_recorded_as_terminal_failures(self):
        for injected_error in (KeyboardInterrupt(), SystemExit(7)):
            with self.subTest(
                error_type=type(injected_error).__name__,
            ), tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                save_dir = root / "results"
                report_path = (
                    save_dir
                    / "runs"
                    / "g1-object_interaction-omomo"
                    / "report.json"
                )

                def interrupt_after_tombstone(
                    _cfg,
                    *,
                    report_path=report_path,
                    injected_error=injected_error,
                ):
                    running = json.loads(
                        report_path.read_text(encoding="utf-8"),
                    )
                    self.assertEqual(running["status"], "running")
                    raise injected_error

                with mock.patch(
                    "holosoma_retargeting.examples.parallel_robot_retarget.validate_config",
                    side_effect=interrupt_after_tombstone,
                ), self.assertRaises(type(injected_error)):
                    main(
                        ParallelRetargetingConfig(
                            save_dir=save_dir,
                        ),
                    )

                failed = json.loads(
                    report_path.read_text(encoding="utf-8"),
                )
                self.assertEqual(failed["status"], "failed")
                self.assertEqual(
                    failed["fatal_error"]["type"],
                    type(injected_error).__name__,
                )
                self.assertIsNotNone(failed["finished_at"])

    def test_same_report_path_rejects_a_concurrent_invocation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "OMOMO_new"
            data_dir.mkdir()
            (data_dir / "sub1_tripod_001.pt").touch()
            save_dir = root / "results"
            report_path = (
                save_dir
                / "runs"
                / "g1-object_interaction-omomo"
                / "report.json"
            )
            entered = threading.Event()
            release = threading.Event()
            original_validate_config = validate_config

            def blocking_validate(config):
                original_validate_config(config)
                entered.set()
                if not release.wait(timeout=10):
                    raise TimeoutError("test did not release report owner")

            def config() -> ParallelRetargetingConfig:
                return ParallelRetargetingConfig(
                    task_type="object_interaction",
                    robot="g1",
                    data_format="omomo",
                    data_dir=data_dir,
                    save_dir=save_dir,
                    object_names=("tripod",),
                    preflight=False,
                    dry_run=True,
                )

            with mock.patch(
                "holosoma_retargeting.examples.parallel_robot_retarget.validate_config",
                side_effect=blocking_validate,
            ), ThreadPoolExecutor(max_workers=1) as executor:
                owner = executor.submit(main, config())
                self.assertTrue(entered.wait(timeout=10))
                running = json.loads(
                    report_path.read_text(encoding="utf-8"),
                )
                self.assertEqual(running["status"], "running")
                owner_invocation_id = running["invocation_id"]
                with self.assertRaisesRegex(
                    BatchReportInUseError,
                    "already owns report path",
                ):
                    main(config())
                unchanged = json.loads(
                    report_path.read_text(encoding="utf-8"),
                )
                self.assertEqual(
                    unchanged["invocation_id"],
                    owner_invocation_id,
                )
                self.assertEqual(unchanged["status"], "running")
                release.set()
                owner.result(timeout=10)

            final_report = json.loads(
                report_path.read_text(encoding="utf-8"),
            )

        self.assertEqual(final_report["status"], "dry_run")
        self.assertEqual(
            final_report["invocation_id"],
            owner_invocation_id,
        )

    def test_atomic_report_failure_preserves_old_report_and_cleans_temp(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            report_path = root / "report.json"
            report_path.write_text("old-report\n", encoding="utf-8")
            with mock.patch.object(
                Path,
                "replace",
                side_effect=RuntimeError("injected replace failure"),
            ), self.assertRaisesRegex(
                RuntimeError,
                "injected replace failure",
            ):
                _write_json_report(
                    report_path,
                    {"status": "running"},
                )

            self.assertEqual(
                report_path.read_text(encoding="utf-8"),
                "old-report\n",
            )
            self.assertFalse(tuple(root.glob(".report.json.*.tmp")))

    def test_spawned_worker_failure_keeps_structured_terminal_batch_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "OMOMO_new"
            data_dir.mkdir()
            (data_dir / "sub1_tripod_001.pt").touch()
            save_dir = root / "results"
            with self.assertRaisesRegex(
                RuntimeError,
                "1 of 1 retargeting tasks failed",
            ):
                main(
                    ParallelRetargetingConfig(
                        task_type="object_interaction",
                        robot="g1",
                        data_format="omomo",
                        data_dir=data_dir,
                        save_dir=save_dir,
                        object_names=("tripod",),
                        motion_data_config=MotionDataConfig(
                            data_format="omomo",
                            robot_type="g1",
                            human_height=1.75,
                        ),
                        preflight=False,
                        max_workers=1,
                    ),
                )
            report_path = (
                save_dir
                / "runs"
                / "g1-object_interaction-omomo"
                / "report.json"
            )
            report = json.loads(
                report_path.read_text(encoding="utf-8"),
            )

        self.assertEqual(report["status"], "completed_with_failures")
        self.assertEqual(report["completed_tasks"], 0)
        self.assertEqual(report["skipped_tasks"], 0)
        self.assertEqual(report["failed_tasks"], 1)
        self.assertIsNone(report["fatal_error"])
        self.assertEqual(len(report["failures"]), 1)
        self.assertIn(
            "EOFError",
            report["failures"][0]["error"],
        )

    def test_main_preserves_nested_overrides_when_binding_selectors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "motions"
            data_dir.mkdir()
            np.savez(
                data_dir / "walk.npz",
                source_format=np.asarray("noetix_mocap"),
            )
            cfg = ParallelRetargetingConfig(
                task_type="robot_only",
                robot="e1",
                data_format="noetix_mocap",
                data_dir=data_dir,
                save_dir=root / "results",
                robot_config=RobotConfig(
                    robot_type="g1",
                    robot_height=1.91,
                ),
                motion_data_config=MotionDataConfig(
                    data_format="omomo",
                    robot_type="g1",
                    human_height=1.83,
                ),
                dry_run=True,
            )

            main(cfg)

        self.assertEqual(cfg.robot_config.robot_type, "e1")
        self.assertEqual(cfg.robot_config.robot_height, 1.91)
        self.assertEqual(cfg.motion_data_config.robot_type, "e1")
        self.assertEqual(
            cfg.motion_data_config.data_format,
            "noetix_mocap",
        )
        self.assertEqual(cfg.motion_data_config.human_height, 1.83)

    def test_main_rejects_inherited_single_task_name_override(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = ParallelRetargetingConfig(
                data_dir=Path(tmpdir),
                task_name="walk",
            )
            with self.assertRaisesRegex(
                ValueError,
                "inherited --task-name is not supported",
            ):
                main(cfg)

    def test_exact_resume_contract_rejects_every_stale_identity_field(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_root = root / "OMOMO_new"
            data_root.mkdir()
            source = data_root / "sub1_tripod_001.pt"
            source.touch()
            job = _object_job(source, root / "results")
            _write_job_artifact(job)
            self.assertTrue(result_artifact_matches_job(job.output_path, job))

            with np.load(job.output_path, allow_pickle=False) as artifact:
                original_payload = {key: np.asarray(artifact[key]) for key in artifact.files}
            mismatches = {
                "schema_version": np.int32(RESULT_SCHEMA_VERSION + 1),
                "source_path": np.asarray("/different/source.pt"),
                "source_sha256": np.asarray("different-source-sha"),
                "config_json": np.asarray("{}"),
                "config_sha256": np.asarray("different-config-sha"),
                "variant": np.asarray("trans_0"),
                "run_kind": np.asarray("augmentation"),
            }
            for field_name, stale_value in mismatches.items():
                with self.subTest(field_name=field_name):
                    stale_payload = dict(original_payload)
                    stale_payload[field_name] = stale_value
                    np.savez_compressed(job.output_path, **stale_payload)
                    self.assertFalse(result_artifact_matches_job(job.output_path, job))

    def test_augmented_job_rejects_stale_identity_baseline_before_loading(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_root = root / "OMOMO_new"
            data_root.mkdir()
            source = data_root / "sub1_tripod_001.pt"
            source.touch()
            identity_job = _object_job(source, root / "results")
            augmented_job = _object_job(
                source,
                root / "results",
                variant=RetargetVariant(
                    name="trans_0",
                    translation=(0.2, 0.0, 0.0),
                ),
            )
            _write_job_artifact(identity_job)
            self.assertTrue(
                result_artifact_matches_job(
                    identity_job.output_path,
                    augmented_job,
                    identity_baseline=True,
                )
            )

            with np.load(identity_job.output_path, allow_pickle=False) as artifact:
                stale_payload = {key: np.asarray(artifact[key]) for key in artifact.files}
            stale_payload["config_sha256"] = np.asarray("stale")
            np.savez_compressed(identity_job.output_path, **stale_payload)

            with mock.patch(
                "holosoma_retargeting.retargeting_pipeline.load_human_motion",
                side_effect=AssertionError("stale baseline must fail first"),
            ), self.assertRaisesRegex(
                ValueError,
                "exact identity baseline",
            ):
                run_retargeting_job(augmented_job)

    def test_worker_skips_complete_task_without_loading_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_root = root / "OMOMO_new"
            data_root.mkdir()
            source = data_root / "sub1_tripod_001.pt"
            source.touch()
            save_dir = root / "results"
            args = _worker_args(source, save_dir, data_root)
            with mock.patch(
                "holosoma_retargeting.examples.parallel_robot_retarget.run_retargeting_job",
                side_effect=_write_job_artifact,
            ):
                generated = process_single_task(args)
            expected_output = Path(generated.generated_files[0])
            result = process_single_task(args)

        self.assertEqual(result.status, "skipped")
        self.assertEqual(result.object_name, "tripod")
        self.assertEqual(result.generated_files, ())
        self.assertEqual(result.skipped_files, (str(expected_output),))

    def test_worker_preserves_logical_identity_for_symlinked_batch_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            real_data_root = root / "real" / "OMOMO_new"
            real_data_root.mkdir(parents=True)
            real_source = real_data_root / "sub1_tripod_001.pt"
            real_source.write_bytes(b"fixture")
            real_height_table = real_data_root.parent / "height_dict.pkl"
            with real_height_table.open("wb") as file:
                pickle.dump({"sub1": 1.11}, file)
            batch_root = root / "subset" / "OMOMO_new"
            batch_root.mkdir(parents=True)
            linked_source = batch_root / real_source.name
            linked_source.symlink_to(real_source)
            logical_height_table = batch_root.parent / "height_dict.pkl"
            with logical_height_table.open("wb") as file:
                pickle.dump({"sub1": 1.91}, file)
            save_dir = root / "results"
            args = list(_worker_args(linked_source, save_dir, batch_root))
            args[6] = MotionDataConfig(
                data_format="omomo",
                robot_type="g1",
                human_height=None,
            )

            with mock.patch(
                "holosoma_retargeting.examples.parallel_robot_retarget.run_retargeting_job",
                side_effect=_write_job_artifact,
            ):
                result = process_single_task(
                    tuple(args),
                )

            self.assertEqual(result.status, "completed")
            (output_path,) = result.generated_files
            with np.load(output_path, allow_pickle=False) as artifact:
                self.assertEqual(
                    str(np.asarray(artifact["source_path"]).item()),
                    str(real_source.resolve()),
                )
                self.assertEqual(
                    str(np.asarray(artifact["sequence_key"]).item()),
                    "sub1_tripod_001",
                )
                self.assertEqual(
                    str(np.asarray(artifact["dataset_partition"]).item()),
                    "OMOMO_new",
                )
                config = json.loads(
                    str(np.asarray(artifact["config_json"]).item()),
                )
                height_dependency = config["solver_identity"]["dependencies"]["entries"]["omomo_height_table"]
                self.assertEqual(
                    height_dependency["path"],
                    str(logical_height_table.resolve()),
                )
                self.assertEqual(
                    height_dependency["sha256"],
                    fingerprint_tree(str(logical_height_table)).sha256,
                )
                self.assertNotEqual(
                    height_dependency["sha256"],
                    fingerprint_tree(str(real_height_table)).sha256,
                )

    def test_dry_run_preflights_and_writes_multi_object_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "OMOMO_new"
            data_dir.mkdir()
            torch.save(torch.zeros((2, 325)), data_dir / "sub1_tripod_001.pt")
            torch.save(torch.zeros((2, 325)), data_dir / "sub1_smallbox_002.pt")
            with (root / "height_dict.pkl").open("wb") as file:
                pickle.dump({"sub1": 1.75}, file)

            save_dir = root / "results"
            main(
                ParallelRetargetingConfig(
                    task_type="object_interaction",
                    robot="e1",
                    data_format="omomo",
                    data_dir=data_dir,
                    save_dir=save_dir,
                    object_names=("tripod", "smallbox"),
                    dry_run=True,
                )
            )
            report = json.loads((save_dir / "runs" / "e1-object_interaction-omomo" / "report.json").read_text())

        self.assertEqual(report["status"], "dry_run")
        self.assertEqual(report["robot"], "e1")
        self.assertEqual(report["total_files"], 2)
        self.assertEqual(report["per_object"], {"smallbox": 1, "tripod": 1})
        self.assertEqual(report["object_names"], ["tripod", "smallbox"])
        self.assertTrue(report["preflight"]["ok"])
        self.assertEqual(
            {entry["object_name"] for entry in report["manifest"]},
            {"tripod", "smallbox"},
        )

    def test_process_pool_reports_resumed_task(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "OMOMO_new"
            data_dir.mkdir()
            source = data_dir / "sub1_tripod_001.pt"
            source.touch()
            save_dir = root / "results"
            with mock.patch(
                "holosoma_retargeting.examples.parallel_robot_retarget.run_retargeting_job",
                side_effect=_write_job_artifact,
            ):
                seeded = process_single_task(_worker_args(source, save_dir, data_dir))
            self.assertEqual(seeded.status, "completed")

            main(
                ParallelRetargetingConfig(
                    task_type="object_interaction",
                    robot="g1",
                    data_format="omomo",
                    data_dir=data_dir,
                    save_dir=save_dir,
                    object_names=("tripod",),
                    motion_data_config=MotionDataConfig(
                        data_format="omomo",
                        robot_type="g1",
                        human_height=1.75,
                    ),
                    preflight=False,
                    max_workers=1,
                )
            )
            report = json.loads((save_dir / "runs" / "g1-object_interaction-omomo" / "report.json").read_text())

        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["completed_tasks"], 0)
        self.assertEqual(report["skipped_tasks"], 1)
        self.assertEqual(report["failed_tasks"], 0)
        self.assertEqual(report["results"][0]["status"], "skipped")


if __name__ == "__main__":
    unittest.main()
