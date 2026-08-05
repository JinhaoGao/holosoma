# ruff: noqa: CPY001, PT009, PT027

from __future__ import annotations

import json
import pickle
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
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
    Args as EvaluationArgs,
)
from holosoma_retargeting.evaluation.eval_retargeting import (  # noqa: E402
    RetargetingEvaluator,
    _evaluate_single_task,
    _load_evaluation_result,
    get_task_names,
)
from holosoma_retargeting.evaluation.eval_retargeting import (  # noqa: E402
    create_task_constants as create_evaluation_constants,
)
from holosoma_retargeting.evaluation.eval_retargeting import (  # noqa: E402
    main as evaluation_main,
)
from holosoma_retargeting.retargeting_pipeline import build_retarget_job  # noqa: E402


def _write_strict_evaluation_result(
    path: Path,
    *,
    sequence_key: str,
    dataset_partition: str = "noetix_mocap",
    task_type: str = "robot_only",
    run_kind: str = "single",
    variant: str = "identity",
    robot_type: str = "e1",
    data_format: str = "noetix_mocap",
    object_name: str | None = None,
) -> None:
    resolved_object_name = object_name or (
        "largebox" if task_type == "object_interaction" else "multi_boxes" if task_type == "climbing" else "ground"
    )
    config = {
        "run_kind": run_kind,
        "experiment_name": None,
        "variant": {"name": variant},
        "dataset_partition": dataset_partition,
        "sequence_key": sequence_key,
        "config": {
            "task_type": task_type,
            "data_format": data_format,
            "robot_config": {"robot_type": robot_type},
            "task_config": {"object_name": resolved_object_name},
        },
    }
    frames = 2
    human_joints = np.zeros((frames, 1, 3), dtype=np.float32)
    qpos = np.zeros((frames, 30), dtype=np.float64)
    qpos[:, 3] = 1.0
    payload: dict[str, object] = {
        "qpos": qpos,
        "human_joints": human_joints,
        "human_joint_names": np.asarray(("Hips",)),
        "human_joint_parent_indices": np.asarray((-1,), dtype=np.int32),
        "run_kind": np.asarray(run_kind),
        "variant": np.asarray(variant),
        "task_type": np.asarray(task_type),
        "dataset_partition": np.asarray(dataset_partition),
        "sequence_key": np.asarray(sequence_key),
        "robot_type": np.asarray(robot_type),
        "source_data_format": np.asarray(data_format),
        "experiment_name": np.asarray(""),
        "object_name": np.asarray(resolved_object_name),
        "object_urdf": np.asarray(""),
        "object_poses_demo": np.zeros((frames, 7), dtype=np.float32),
        "object_poses_target": np.zeros((frames, 7), dtype=np.float32),
        "foot_sticking_side_names": np.asarray(("left", "right")),
        "foot_sticking_states": np.zeros((frames, 2), dtype=bool),
        "foot_sticking_enabled_for_saved_trajectory": np.asarray(False),
        "cost": np.asarray(0.0),
        "fps": np.asarray(30.0),
        "human_position_scale": np.asarray(1.0),
        "config_json": np.asarray(json.dumps(config, separators=(",", ":"), sort_keys=True)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    if task_type in {"object_interaction", "climbing"}:
        object_urdf = path.parent / f"{resolved_object_name}.urdf"
        object_urdf.write_text(
            '<robot name="test_object"><link name="base"/></robot>',
            encoding="utf-8",
        )
        payload["object_urdf"] = np.asarray(str(object_urdf))
    np.savez_compressed(path, **payload)


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
                    model = mujoco.MjModel.from_xml_path(evaluation.SCENE_XML_FILE)
                    self.assertEqual(model.nq, 7 + robot_dof + 7)


class OmomoDownstreamInferenceTests(unittest.TestCase):
    def test_evaluation_worker_infers_object_per_result(self):
        class FakeEvaluator:
            def __init__(self, *, object_name, constants, **_kwargs):
                self.object_name = object_name
                self.scene_path = constants.SCENE_XML_FILE

            def evaluate_trajectory(self, result):
                return {
                    "object_name": self.object_name,
                    "scene_path": self.scene_path,
                    "scene_exists": Path(self.scene_path).is_file(),
                    "saved_frames": result.qpos.shape[0],
                }

        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = Path(tmpdir) / "sub1_suitcase_001" / "identity.npz"
            _write_strict_evaluation_result(
                result_path,
                sequence_key="sub1_suitcase_001",
                dataset_partition="OMOMO_new",
                task_type="object_interaction",
                data_format="omomo",
                object_name="suitcase",
            )
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
                    asdict(robot_config),
                    asdict(motion_config),
                    None,
                    "robot_object",
                )

        self.assertEqual(task_name, "sub1_suitcase_001")
        self.assertEqual(result["object_name"], "suitcase")
        self.assertEqual(result["saved_frames"], 2)
        self.assertTrue(result["scene_exists"])

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


class EvaluationTaskDiscoveryTests(unittest.TestCase):
    @staticmethod
    def _write_canonical_result(
        path,
        *,
        sequence_key,
        dataset_partition="OMOMO_new",
        task_type="object_interaction",
        run_kind="single",
        variant="identity",
    ):
        _write_strict_evaluation_result(
            path,
            sequence_key=sequence_key,
            dataset_partition=dataset_partition,
            task_type=task_type,
            run_kind=run_kind,
            variant=variant,
        )

    def test_recursive_canonical_identity_takes_precedence_over_other_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sequence_dir = (
                root / "canonical" / "e1" / "robot_object" / "omomo" / "OMOMO_new" / "nested" / "sub1_suitcase_001"
            )
            sequence_dir.mkdir(parents=True)
            identity_path = sequence_dir / "identity.npz"
            self._write_canonical_result(
                identity_path,
                sequence_key="nested/sub1_suitcase_001",
            )
            np.savez(
                sequence_dir / "trans_0.npz",
                sequence_key=np.asarray("nested/sub1_suitcase_001"),
            )
            self._write_canonical_result(
                root / "canonical" / "g1" / "robot_only" / "sequence" / "identity.npz",
                sequence_key="must_not_include_other_task",
                task_type="robot_only",
            )
            np.savez(root / "sub1_suitcase_001_original.npz", qpos=np.zeros((1, 1)))
            np.savez(root / "sub1_suitcase_001_no_orientation.npz", qpos=np.zeros((1, 1)))

            task_names, files = get_task_names(root, "robot_object")

        self.assertEqual(task_names, ["nested/sub1_suitcase_001"])
        self.assertEqual(files, [str(identity_path)])

    def test_legacy_original_is_rejected_without_canonical_identity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            np.savez(root / "sub2_tripod_019_original.npz", qpos=np.zeros((1, 1)))
            np.savez(root / "sub2_tripod_019_trans_0.npz", qpos=np.zeros((1, 1)))
            np.savez(root / "sub2_tripod_019_ablation.npz", qpos=np.zeros((1, 1)))

            with self.assertRaisesRegex(ValueError, "required by the evaluator"):
                get_task_names(root, "robot_only")

    def test_unrecognized_metadata_does_not_gate_evaluation_discovery(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            identity_path = root / "canonical" / "sequence" / "identity.npz"
            _write_strict_evaluation_result(
                identity_path,
                sequence_key="sequence",
                task_type="object_interaction",
            )
            with np.load(identity_path, allow_pickle=False) as saved:
                payload = {key: np.asarray(saved[key]) for key in saved.files}
            payload["unrecognized_metadata"] = np.asarray("ignored")
            np.savez_compressed(identity_path, **payload)

            task_names, files = get_task_names(root, "robot_object")

        self.assertEqual(task_names, ["sequence"])
        self.assertEqual(files, [str(identity_path)])

    def test_duplicate_sequence_keys_across_partitions_require_a_narrower_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_canonical_result(
                root / "partition_a" / "sequence" / "identity.npz",
                sequence_key="sequence",
                dataset_partition="partition_a",
            )
            self._write_canonical_result(
                root / "partition_b" / "sequence" / "identity.npz",
                sequence_key="sequence",
                dataset_partition="partition_b",
            )

            with self.assertRaisesRegex(ValueError, "Select a narrower --res-dir"):
                get_task_names(root, "robot_object")

    def test_evaluation_result_is_bound_to_requested_robot_and_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = Path(tmpdir) / "walk" / "identity.npz"
            _write_strict_evaluation_result(
                result_path,
                sequence_key="walk",
            )

            with self.assertRaisesRegex(ValueError, "robot_type='e1'.*expected 'g1'"):
                _load_evaluation_result(
                    result_path,
                    data_type="robot_only",
                    robot_type="g1",
                )
            with self.assertRaisesRegex(
                ValueError,
                "source_data_format='noetix_mocap'.*expected 'amass'",
            ):
                _load_evaluation_result(
                    result_path,
                    data_type="robot_only",
                    data_format="amass",
                )

    def test_incomplete_v2_identity_is_not_treated_as_discoverable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            identity_path = Path(tmpdir) / "walk" / "identity.npz"
            identity_path.parent.mkdir(parents=True)
            np.savez(
                identity_path,
                run_kind=np.asarray("single"),
                variant=np.asarray("identity"),
            )

            with self.assertRaisesRegex(
                ValueError,
                "Cannot load evaluation result",
            ):
                get_task_names(tmpdir, "robot_only")

    def test_legacy_terrain_result_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            np.savez(
                root / "climb_joint_positions_original.npz",
                qpos=np.zeros((1, 1)),
            )

            with self.assertRaisesRegex(ValueError, "required by the evaluator"):
                get_task_names(root, "robot_terrain")


class EvaluationSavedTrajectoryMetricTests(unittest.TestCase):
    def test_evaluation_main_preserves_nested_selector_overrides(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = EvaluationArgs(
                res_dir=Path(tmpdir),
                data_type="robot_only",
                robot="e1",
                data_format="noetix_mocap",
                robot_config=RobotConfig(
                    robot_type="g1",
                    robot_height=1.91,
                ),
                motion_data_config=MotionDataConfig(
                    data_format="omomo",
                    robot_type="g1",
                    human_height=1.83,
                ),
            )

            evaluation_main(cfg)

        self.assertEqual(cfg.robot_config.robot_type, "e1")
        self.assertEqual(cfg.robot_config.robot_height, 1.91)
        self.assertEqual(cfg.motion_data_config.robot_type, "e1")
        self.assertEqual(
            cfg.motion_data_config.data_format,
            "noetix_mocap",
        )
        self.assertEqual(cfg.motion_data_config.human_height, 1.83)

    def test_foot_sliding_uses_saved_fps_and_zero_contact_is_safe(self):
        evaluator = RetargetingEvaluator.__new__(RetargetingEvaluator)
        evaluator.joints_mapping = {"left": "left_link", "right": "right_link"}
        evaluator.sliding_threshold = 0.01

        def fake_positions(q, _link_names):
            return np.asarray(
                [[float(q[0]) * 0.01, 0.0, 0.0], [0.0, 0.0, 0.0]],
                dtype=np.float64,
            )

        evaluator._get_robot_link_positions = fake_positions
        qpos = np.asarray([[0.0], [1.0], [2.0]], dtype=np.float64)
        contact = np.asarray([[True, False]] * 3)

        fraction, velocities = evaluator.detect_foot_sliding(
            qpos,
            contact,
            ("left", "right"),
            fps=10.0,
        )
        self.assertAlmostEqual(fraction, 2.0 / 3.0)
        np.testing.assert_allclose(velocities, [0.1, 0.1])

        fraction, velocities = evaluator.detect_foot_sliding(
            qpos,
            np.zeros((3, 2), dtype=bool),
            ("left", "right"),
            fps=10.0,
        )
        self.assertEqual(fraction, 0.0)
        self.assertEqual(velocities.shape, (0,))

    def test_terrain_contact_forwards_each_saved_qpos_frame(self):
        evaluator = RetargetingEvaluator.__new__(RetargetingEvaluator)
        evaluator._obj_VW = np.ones((1, 3), dtype=np.float64)
        evaluator.object_name = "multi_boxes"
        evaluator.robot_model = SimpleNamespace(
            ngeom=2,
            geom_bodyid=np.asarray([0, 1], dtype=np.int32),
        )
        evaluator.robot_data = SimpleNamespace(qpos=np.zeros(2, dtype=np.float64))
        evaluator.joints_mapping = {"joint": "robot_body"}
        evaluator.collision_detection_threshold = 0.1
        evaluator.contact_threshold = 0.1
        evaluator.detect_demo_contact = lambda *_args, **_kwargs: {
            "joint": np.zeros(3),
        }
        qpos = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        forwarded: list[np.ndarray] = []

        def fake_geom_name(_model, _object_type, geom_id):
            return "robot_geom" if geom_id == 0 else "multi_boxes_collision_0"

        with patch(
            "holosoma_retargeting.evaluation.eval_retargeting.mujoco.mj_id2name",
            side_effect=fake_geom_name,
        ), patch(
            "holosoma_retargeting.evaluation.eval_retargeting.mujoco.mj_name2id",
            return_value=0,
        ), patch(
            "holosoma_retargeting.evaluation.eval_retargeting.mujoco.mj_geomDistance",
            return_value=0.0,
        ), patch(
            "holosoma_retargeting.evaluation.eval_retargeting.mujoco.mj_forward",
            side_effect=lambda _model, data: forwarded.append(data.qpos.copy()),
        ):
            score = evaluator.evaluate_terrain_contact_precision(
                np.zeros((2, 1, 3), dtype=np.float64),
                qpos,
                joint_names=("joint",),
            )

        self.assertEqual(score, 1.0)
        np.testing.assert_array_equal(np.asarray(forwarded), qpos)


class OmomoAcceptanceHarnessTests(unittest.TestCase):
    def test_acceptance_matrix_validates_saved_qpos_and_metadata(self):
        def fake_retargeting(config):
            object_name = config.task_name.split("_")[1]
            output_path = build_retarget_job(config).output_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                output_path,
                qpos=np.zeros(
                    (2, 7 + config.robot_config.ROBOT_DOF + 7),
                    dtype=np.float64,
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
            qpos_dtypes = []
            for result in report["results"]:
                with np.load(result["result_path"], allow_pickle=False) as saved:
                    qpos_dtypes.append(saved["qpos"].dtype)

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["expected_cases"], 2)
        self.assertEqual(report["passed_cases"], 2)
        self.assertEqual(report["failed_cases"], 0)
        self.assertEqual(qpos_dtypes, [np.dtype(np.float64), np.dtype(np.float64)])
        self.assertTrue(
            all(
                "/canonical/" not in result["result_path"] and result["result_path"].endswith("/identity.npz")
                for result in report["results"]
            ),
        )


if __name__ == "__main__":
    unittest.main()
