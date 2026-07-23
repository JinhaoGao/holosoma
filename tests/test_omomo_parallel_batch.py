# ruff: noqa: PT009, PT027

from __future__ import annotations

import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "holosoma_retargeting"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from holosoma_retargeting.config_types.data_type import MotionDataConfig  # noqa: E402
from holosoma_retargeting.config_types.retargeter import RetargeterConfig  # noqa: E402
from holosoma_retargeting.config_types.retargeting import ParallelRetargetingConfig  # noqa: E402
from holosoma_retargeting.config_types.robot import RobotConfig  # noqa: E402
from holosoma_retargeting.config_types.task import TaskConfig  # noqa: E402
from holosoma_retargeting.examples.parallel_robot_retarget import (  # noqa: E402
    find_files,
    main,
    process_single_task,
    resolve_batch_object_names,
)


class OmomoBatchSelectionTests(unittest.TestCase):
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
        self.assertIsNone(
            resolve_batch_object_names("object_interaction", "omomo", None, None)
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
    def test_worker_skips_complete_task_without_loading_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "sub1_tripod_001.pt"
            source.touch()
            save_dir = root / "results"
            save_dir.mkdir()
            expected_output = save_dir / "sub1_tripod_001_original.npz"
            expected_output.touch()

            result = process_single_task(
                (
                    source,
                    save_dir,
                    "object_interaction",
                    "omomo",
                    RobotConfig(robot_type="g1"),
                    MotionDataConfig(data_format="omomo", robot_type="g1"),
                    TaskConfig(),
                    RetargeterConfig(),
                    False,
                    False,
                )
            )

        self.assertEqual(result.status, "skipped")
        self.assertEqual(result.object_name, "tripod")
        self.assertEqual(result.generated_files, ())
        self.assertEqual(result.skipped_files, (str(expected_output),))

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
            report = json.loads((save_dir / "batch_report.json").read_text())

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
            (data_dir / "sub1_tripod_001.pt").touch()
            save_dir = root / "results"
            save_dir.mkdir()
            (save_dir / "sub1_tripod_001_original.npz").touch()

            main(
                ParallelRetargetingConfig(
                    task_type="object_interaction",
                    robot="g1",
                    data_format="omomo",
                    data_dir=data_dir,
                    save_dir=save_dir,
                    object_names=("tripod",),
                    preflight=False,
                    max_workers=1,
                )
            )
            report = json.loads((save_dir / "batch_report.json").read_text())

        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["completed_tasks"], 0)
        self.assertEqual(report["skipped_tasks"], 1)
        self.assertEqual(report["failed_tasks"], 0)
        self.assertEqual(report["results"][0]["status"], "skipped")


if __name__ == "__main__":
    unittest.main()
