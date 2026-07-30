# ruff: noqa: CPY001, PT009

from __future__ import annotations

import unittest
from pathlib import Path

import mujoco
import numpy as np
from holosoma_retargeting.config_types.data_conversion import DataConversionConfig
from holosoma_retargeting.config_types.data_type import MotionDataConfig
from holosoma_retargeting.config_types.retargeting import RetargetingConfig
from holosoma_retargeting.config_types.robot import RobotConfig
from holosoma_retargeting.retargeting_pipeline import normalize_retargeting_config


class E2RetargetingConfigTest(unittest.TestCase):
    DATA_FORMATS = (
        "gvhmr",
        "lafan",
        "mocap",
        "noetix_mocap",
        "omomo",
    )
    DATA_FORMAT_ALIASES = {
        "smplh": "omomo",
        "noetix_lafan": "noetix_mocap",
        "noetix-mocap": "noetix_mocap",
    }

    @staticmethod
    def _model() -> mujoco.MjModel:
        package_root = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "holosoma_retargeting"
            / "holosoma_retargeting"
        )
        return mujoco.MjModel.from_xml_path(
            str(package_root / "models" / "e2" / "e2_23dof.xml"),
        )

    def test_all_human_formats_have_complete_e2_mappings(self) -> None:
        model = self._model()
        body_names = {
            str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id))
            for body_id in range(1, model.nbody)
        }

        for data_format in self.DATA_FORMATS:
            with self.subTest(data_format=data_format):
                motion = MotionDataConfig(
                    data_format=data_format,
                    robot_type="e2",
                )
                mapping = motion.resolved_joints_mapping
                self.assertEqual(len(mapping), 15)
                self.assertTrue(set(mapping).issubset(motion.resolved_demo_joints))
                self.assertTrue(set(mapping.values()).issubset(body_names))
                self.assertEqual(len(set(mapping.values())), len(mapping))
                self.assertEqual(len(motion.toe_names), 2)

    def test_human_format_aliases_resolve_to_the_same_e2_mappings(self) -> None:
        for alias, canonical in self.DATA_FORMAT_ALIASES.items():
            with self.subTest(alias=alias):
                aliased = MotionDataConfig(data_format=alias, robot_type="e2")
                canonical_config = MotionDataConfig(
                    data_format=canonical,
                    robot_type="e2",
                )
                self.assertEqual(aliased.data_format, canonical)
                self.assertEqual(
                    aliased.resolved_joints_mapping,
                    canonical_config.resolved_joints_mapping,
                )

    def test_top_level_e2_selection_normalizes_nested_configs_for_every_format(self) -> None:
        for data_format in self.DATA_FORMATS:
            with self.subTest(data_format=data_format):
                config = normalize_retargeting_config(
                    RetargetingConfig(
                        task_type="robot_only",
                        robot="e2",
                        data_format=data_format,
                        task_name="representative",
                        data_path=Path("demo_data") / data_format,
                    ),
                )
                self.assertEqual(config.robot_config.robot_type, "e2")
                self.assertEqual(config.robot_config.ROBOT_HEIGHT, 1.55)
                self.assertEqual(config.motion_data_config.robot_type, "e2")
                self.assertEqual(
                    config.motion_data_config.data_format,
                    data_format,
                )
                self.assertEqual(
                    len(config.motion_data_config.resolved_joints_mapping),
                    15,
                )

    def test_noetix_mapping_and_t_pose_are_complete(self) -> None:
        robot = RobotConfig(robot_type="e2")
        motion = MotionDataConfig(
            data_format="noetix_mocap",
            robot_type="e2",
        )

        self.assertEqual(robot.ROBOT_DOF, 23)
        self.assertEqual(robot.ROBOT_HEIGHT, 1.55)
        self.assertEqual(robot.ROBOT_URDF_FILE, "models/e2/e2_23dof.urdf")
        self.assertEqual(len(robot.FOOT_STICKING_LINKS), 10)
        self.assertEqual(robot.NOMINAL_TRACKING_INDICES.tolist(), list(range(22)))
        self.assertEqual(len(motion.resolved_joints_mapping), 15)
        self.assertEqual(len(motion.resolved_orientation_joints_mapping), 15)
        self.assertEqual(
            set(motion.resolved_orientation_joints_mapping),
            set(motion.resolved_orientation_t_pose_human_quaternions_wxyz),
        )
        self.assertIsNotNone(
            motion.resolved_orientation_t_pose_robot_base_quaternion_wxyz,
        )
        self.assertEqual(
            set(motion.resolved_orientation_t_pose_robot_joint_positions),
            {
                "l_arm_shoulder_roll_joint",
                "l_arm_elbow_joint",
                "r_arm_shoulder_roll_joint",
                "r_arm_elbow_joint",
            },
        )

    def test_noetix_t_pose_is_within_limits_and_geometrically_horizontal(self) -> None:
        model = self._model()
        data = mujoco.MjData(model)
        motion = MotionDataConfig(
            data_format="noetix_mocap",
            robot_type="e2",
        )
        qpos = model.qpos0.copy()
        qpos[3:7] = motion.resolved_orientation_t_pose_robot_base_quaternion_wxyz
        for joint_name, value in motion.resolved_orientation_t_pose_robot_joint_positions.items():
            joint_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )
            self.assertGreaterEqual(value, model.jnt_range[joint_id, 0])
            self.assertLessEqual(value, model.jnt_range[joint_id, 1])
            qpos[int(model.jnt_qposadr[joint_id])] = value

        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        for side, direction in (("l", -1.0), ("r", 1.0)):
            positions = np.asarray(
                [
                    data.xpos[
                        mujoco.mj_name2id(
                            model,
                            mujoco.mjtObj.mjOBJ_BODY,
                            f"{side}_arm_shoulder_yaw_link",
                        )
                    ],
                    data.xpos[
                        mujoco.mj_name2id(
                            model,
                            mujoco.mjtObj.mjOBJ_BODY,
                            f"{side}_arm_elbow_link",
                        )
                    ],
                    data.xpos[
                        mujoco.mj_name2id(
                            model,
                            mujoco.mjtObj.mjOBJ_BODY,
                            f"{side}_hand_sphere_link",
                        )
                    ],
                ],
            )
            with self.subTest(side=side):
                self.assertLess(np.ptp(positions[:, 2]), 0.01)
                self.assertLess(np.ptp(positions[:, 1]), 0.005)
                self.assertTrue(np.all(direction * np.diff(positions[:, 0]) > 0.0))

    def test_joint_conversion_order_matches_mujoco(self) -> None:
        model = self._model()
        configured_names = DataConversionConfig(
            input_file="unused.npz",
            robot="e2",
        ).JOINT_NAMES
        model_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            for joint_id in range(1, model.njnt)
        ]

        self.assertEqual(model.nq, 30)
        self.assertEqual(model.nv, 29)
        self.assertEqual(model.nbody - 1, 38)
        self.assertEqual(model_names, configured_names)
        self.assertTrue(np.allclose(model.qpos0[:3], (0.0, 0.0, 0.7595)))


if __name__ == "__main__":
    unittest.main()
