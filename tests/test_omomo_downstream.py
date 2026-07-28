# ruff: noqa: PT009, PT027

from __future__ import annotations

import pickle
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "holosoma_retargeting"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from holosoma_retargeting.config_types.data_type import MotionDataConfig  # noqa: E402
from holosoma_retargeting.config_types.robot import RobotConfig  # noqa: E402
from holosoma_retargeting.data_conversion.convert_data_format_mj import (  # noqa: E402
    create_task_constants as create_conversion_constants,
)
from holosoma_retargeting.data_conversion.convert_data_format_mj import (  # noqa: E402
    resolve_conversion_object_name,
)
from holosoma_retargeting.data_utils.omomo import OMOMO_OBJECT_NAMES  # noqa: E402
from holosoma_retargeting.data_utils.validate_omomo_retargeting import (  # noqa: E402
    run_acceptance,
)
from holosoma_retargeting.evaluation.eval_retargeting import (  # noqa: E402
    _evaluate_single_task,
)
from holosoma_retargeting.evaluation.eval_retargeting import (  # noqa: E402
    create_task_constants as create_evaluation_constants,
)


class OmomoDownstreamSceneTests(unittest.TestCase):
    def test_evaluation_and_conversion_use_every_g1_e1_object_scene(self):
        for robot_name, robot_dof in (("g1", 29), ("e1", 23)):
            robot_config = RobotConfig(robot_type=robot_name)
            motion_config = MotionDataConfig(
                data_format="omomo",
                robot_type=robot_name,
            )
            for object_name in OMOMO_OBJECT_NAMES:
                with self.subTest(robot=robot_name, object_name=object_name):
                    evaluation = create_evaluation_constants(
                        robot_config,
                        motion_config,
                        object_name=object_name,
                    )
                    conversion = create_conversion_constants(
                        robot_config,
                        motion_config,
                        object_name=object_name,
                    )

                    self.assertEqual(evaluation.OBJECT_NAME, object_name)
                    self.assertEqual(conversion.OBJECT_NAME, object_name)
                    self.assertEqual(
                        evaluation.SCENE_XML_FILE,
                        conversion.SCENE_XML_FILE,
                    )
                    self.assertTrue(Path(evaluation.OBJECT_URDF_FILE).is_file())
                    model = mujoco.MjModel.from_xml_path(
                        evaluation.SCENE_XML_FILE
                    )
                    self.assertEqual(model.nq, 7 + robot_dof + 7)


class OmomoDownstreamInferenceTests(unittest.TestCase):
    def test_evaluation_worker_infers_object_per_result(self):
        class FakeEvaluator:
            def __init__(self, *, object_name, constants, **_kwargs):
                self.object_name = object_name
                self.scene_path = constants.SCENE_XML_FILE

            def evaluate_trajectory(self, _task_name, _data_path, _input_data_dir):
                return {
                    "object_name": self.object_name,
                    "scene_path": self.scene_path,
                }

        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = Path(tmpdir) / "sub1_suitcase_001_original.npz"
            np.savez(result_path, object_name=np.asarray("suitcase"))
            robot_config = RobotConfig(robot_type="e1")
            motion_config = MotionDataConfig(
                data_format="omomo",
                robot_type="e1",
            )
            with patch(
                "holosoma_retargeting.evaluation.eval_retargeting.RetargetingEvaluator",
                FakeEvaluator,
            ):
                task_name, result = _evaluate_single_task(
                    "sub1_suitcase_001",
                    str(result_path),
                    tmpdir,
                    asdict(robot_config),
                    asdict(motion_config),
                    None,
                    "robot_object",
                )

        self.assertEqual(task_name, "sub1_suitcase_001")
        self.assertEqual(result["object_name"], "suitcase")
        self.assertTrue(result["scene_path"].endswith("e1_23dof_w_suitcase.xml"))

    def test_conversion_resolves_metadata_and_rejects_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = Path(tmpdir) / "renamed_result.npz"
            np.savez(result_path, object_name=np.asarray("monitor"))

            self.assertEqual(
                resolve_conversion_object_name(
                    result_path,
                    "omomo",
                    True,
                    None,
                ),
                "monitor",
            )
            with self.assertRaisesRegex(ValueError, "explicit override"):
                resolve_conversion_object_name(
                    result_path,
                    "omomo",
                    True,
                    "tripod",
                )


class OmomoAcceptanceHarnessTests(unittest.TestCase):
    def test_acceptance_matrix_validates_saved_qpos_and_metadata(self):
        def fake_retargeting(config):
            object_name = config.task_name.split("_")[1]
            output_path = (
                config.save_dir / f"{config.task_name}_original.npz"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                output_path,
                qpos=np.zeros(
                    (2, 7 + config.robot_config.ROBOT_DOF + 7),
                    dtype=np.float32,
                ),
                robot_type=np.asarray(config.robot),
                object_name=np.asarray(object_name),
                object_points_demo_local=np.zeros((100, 3), dtype=np.float32),
                object_points_target_local=np.zeros((100, 3), dtype=np.float32),
                object_points_demo_world=np.zeros((2, 100, 3), dtype=np.float32),
                object_points_target_world=np.zeros((2, 100, 3), dtype=np.float32),
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "source" / "OMOMO_new"
            data_dir.mkdir(parents=True)
            torch.save(
                torch.zeros((3, 325)),
                data_dir / "sub1_tripod_001.pt",
            )
            with (data_dir.parent / "height_dict.pkl").open("wb") as file:
                pickle.dump({"sub1": 1.75}, file)

            with patch(
                "holosoma_retargeting.data_utils.validate_omomo_retargeting.run_retargeting",
                fake_retargeting,
            ):
                report = run_acceptance(
                    data_dir,
                    root / "acceptance",
                    robots=("g1", "e1"),
                    object_names=("tripod",),
                    frame_count=2,
                    run_preflight=False,
                )

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["expected_cases"], 2)
        self.assertEqual(report["passed_cases"], 2)
        self.assertEqual(report["failed_cases"], 0)


if __name__ == "__main__":
    unittest.main()
