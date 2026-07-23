# ruff: noqa: PT009, PT027

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "holosoma_retargeting"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from holosoma_retargeting.config_types.data_type import MotionDataConfig  # noqa: E402
from holosoma_retargeting.config_types.retargeter import RetargeterConfig  # noqa: E402
from holosoma_retargeting.config_types.robot import RobotConfig  # noqa: E402
from holosoma_retargeting.config_types.task import TaskConfig  # noqa: E402
from holosoma_retargeting.data_utils.omomo import OMOMO_OBJECT_NAMES  # noqa: E402
from holosoma_retargeting.examples.robot_retarget import (  # noqa: E402
    build_retargeter_kwargs_from_config,
    create_task_constants,
    resolve_task_object_name,
    setup_object_data,
)
from holosoma_retargeting.src.interaction_mesh_retargeter import InteractionMeshRetargeter  # noqa: E402


class OmomoObjectResolutionTests(unittest.TestCase):
    def test_infers_object_from_sequence(self):
        self.assertEqual(
            resolve_task_object_name("object_interaction", "omomo", "sub3_tripod_012", None),
            "tripod",
        )

    def test_rejects_explicit_object_mismatch(self):
        with self.assertRaisesRegex(ValueError, "contains object 'tripod'"):
            resolve_task_object_name(
                "object_interaction",
                "omomo",
                "sub3_tripod_012",
                "largebox",
            )

    def test_preserves_non_interaction_defaults(self):
        self.assertEqual(resolve_task_object_name("robot_only", "omomo", "sub3_tripod_012", None), "ground")
        self.assertEqual(resolve_task_object_name("climbing", "mocap", "wall", None), "multi_boxes")


class OmomoRetargetingObjectSetupTests(unittest.TestCase):
    ROBOTS = {"g1": 29, "e1": 23}

    def test_object_constants_require_resolved_category(self):
        with self.assertRaisesRegex(ValueError, "resolved object name"):
            create_task_constants(
                RobotConfig(robot_type="g1"),
                MotionDataConfig(data_format="omomo", robot_type="g1"),
                TaskConfig(),
                "object_interaction",
            )

    def test_all_objects_setup_for_g1_and_e1(self):
        for robot_name, robot_dof in self.ROBOTS.items():
            robot_config = RobotConfig(robot_type=robot_name)
            motion_config = MotionDataConfig(data_format="omomo", robot_type=robot_name)
            for object_name in OMOMO_OBJECT_NAMES:
                with self.subTest(robot=robot_name, object_name=object_name):
                    task_config = TaskConfig(object_name=object_name)
                    constants = create_task_constants(
                        robot_config,
                        motion_config,
                        task_config,
                        "object_interaction",
                    )
                    object_points, demo_points, object_urdf = setup_object_data(
                        "object_interaction",
                        constants,
                        None,
                        0.75,
                        task_config,
                        False,
                    )

                    self.assertEqual(constants.OBJECT_NAME, object_name)
                    self.assertTrue(Path(constants.OBJECT_MESH_FILE).is_file())
                    self.assertTrue(Path(object_urdf).is_file())
                    self.assertTrue(Path(constants.SCENE_XML_FILE).is_file())
                    self.assertEqual(object_points.shape, (100, 3))
                    self.assertEqual(demo_points.shape, (100, 3))
                    np.testing.assert_allclose(demo_points, object_points * 0.75)

                    model = mujoco.MjModel.from_xml_path(constants.SCENE_XML_FILE)
                    self.assertEqual(model.nq, 7 + robot_dof + 7)
                    self.assertGreaterEqual(
                        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, object_name),
                        0,
                    )
                    retargeter = InteractionMeshRetargeter(
                        **build_retargeter_kwargs_from_config(
                            RetargeterConfig(),
                            constants,
                            object_urdf,
                            "object_interaction",
                        )
                    )
                    self.assertTrue(retargeter.has_dynamic_object)
                    self.assertEqual(retargeter.nq, 7 + robot_dof + 7)

    def test_scaled_object_setup_updates_target_only(self):
        robot_config = RobotConfig(robot_type="g1")
        motion_config = MotionDataConfig(data_format="omomo", robot_type="g1")
        task_config = TaskConfig(object_name="smallbox", object_scale=(0.5, 1.0, 1.5))
        constants = create_task_constants(
            robot_config,
            motion_config,
            task_config,
            "object_interaction",
        )
        target_points, demo_points, object_urdf = setup_object_data(
            "object_interaction",
            constants,
            None,
            0.8,
            task_config,
            False,
        )

        unscaled = target_points / np.array([0.5, 1.0, 1.5])
        np.testing.assert_allclose(demo_points, unscaled * 0.8)
        self.assertIn("scaled_0.5_1_1.5", Path(object_urdf).stem)
        self.assertIn("scaled_0.5_1_1.5", Path(constants.SCENE_XML_FILE).stem)


if __name__ == "__main__":
    unittest.main()
