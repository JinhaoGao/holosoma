# ruff: noqa: CPY001, PT009

from __future__ import annotations

import unittest
from pathlib import Path

import mujoco
import numpy as np
from holosoma_retargeting.config_types.data_type import (
    APPROVED_DIRECT_ORIENTATION_SOURCES,
    MotionDataConfig,
)
from holosoma_retargeting.config_types.robot import RobotConfig
from scipy.spatial.transform import Rotation


class OrientationMappingMatrixTest(unittest.TestCase):
    DATA_FORMATS = (
        "gvhmr",
        "lafan",
        "mocap",
        "noetix_mocap",
        "omomo",
    )
    ROBOTS = ("g1", "e1", "e2")
    PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "holosoma_retargeting" / "holosoma_retargeting"

    @classmethod
    def setUpClass(cls) -> None:
        cls.models = {
            robot: mujoco.MjModel.from_xml_path(
                str(
                    cls.PACKAGE_ROOT
                    / Path(RobotConfig(robot_type=robot).ROBOT_URDF_FILE).with_suffix(
                        ".xml",
                    )
                ),
            )
            for robot in cls.ROBOTS
        }

    def test_every_direct_orientation_format_has_three_robot_mappings(self) -> None:
        self.assertEqual(
            set(self.DATA_FORMATS),
            set(APPROVED_DIRECT_ORIENTATION_SOURCES).difference({"amass"}),
        )
        for data_format in self.DATA_FORMATS:
            for robot in self.ROBOTS:
                with self.subTest(data_format=data_format, robot=robot):
                    motion = MotionDataConfig(
                        data_format=data_format,
                        robot_type=robot,
                    )
                    mapping = motion.resolved_orientation_joints_mapping
                    human_reference = motion.resolved_orientation_t_pose_human_quaternions_wxyz
                    robot_base = motion.resolved_orientation_t_pose_robot_base_quaternion_wxyz

                    self.assertEqual(len(mapping), 15)
                    self.assertEqual(set(mapping), set(human_reference))
                    self.assertTrue(set(mapping).issubset(motion.resolved_demo_joints))
                    self.assertIsNotNone(robot_base)
                    self.assertTrue(
                        set(mapping.values()).issubset(self._body_names(self.models[robot])),
                    )
                    self.assertEqual(len(set(mapping.values())), len(mapping))
                    self._assert_unit_quaternions(
                        (*human_reference.values(), robot_base),
                    )

    def test_orientation_links_are_independent_from_position_links(self) -> None:
        expected_shoulder_links = {
            "g1": "left_shoulder_yaw_link",
            "e1": "l_arm_shoulder_yaw_link",
            "e2": "l_arm_shoulder_yaw_link",
        }
        for data_format, human_shoulder in (
            ("gvhmr", "L_Shoulder"),
            ("lafan", "LeftArm"),
            ("mocap", "LeftArm"),
            ("noetix_mocap", "LeftArm"),
            ("omomo", "L_Shoulder"),
        ):
            for robot, expected_link in expected_shoulder_links.items():
                with self.subTest(data_format=data_format, robot=robot):
                    motion = MotionDataConfig(
                        data_format=data_format,
                        robot_type=robot,
                    )
                    self.assertEqual(
                        motion.resolved_orientation_joints_mapping[human_shoulder],
                        expected_link,
                    )
                    self.assertNotEqual(
                        motion.resolved_joints_mapping[human_shoulder],
                        expected_link,
                    )

    def test_t_pose_offsets_align_every_source_frame_with_robot_fk(self) -> None:
        for data_format in self.DATA_FORMATS:
            for robot in self.ROBOTS:
                with self.subTest(data_format=data_format, robot=robot):
                    model = self.models[robot]
                    motion = MotionDataConfig(
                        data_format=data_format,
                        robot_type=robot,
                    )
                    mapping = motion.resolved_orientation_joints_mapping
                    human_reference = motion.resolved_orientation_t_pose_human_quaternions_wxyz
                    robot_matrices = self._robot_reference_matrices(model, motion)
                    human_matrices = Rotation.from_quat(
                        np.asarray([human_reference[name] for name in mapping]),
                        scalar_first=True,
                    ).as_matrix()
                    offsets = np.swapaxes(human_matrices, -1, -2) @ robot_matrices

                    np.testing.assert_allclose(
                        human_matrices @ offsets,
                        robot_matrices,
                        atol=1e-12,
                    )
                    np.testing.assert_allclose(
                        np.linalg.det(offsets),
                        np.ones(len(mapping)),
                        atol=1e-12,
                    )

    @staticmethod
    def _body_names(model: mujoco.MjModel) -> set[str]:
        return {str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)) for body_id in range(1, model.nbody)}

    @staticmethod
    def _assert_unit_quaternions(
        quaternions: tuple[tuple[float, float, float, float], ...],
    ) -> None:
        values = np.asarray(quaternions, dtype=np.float64)
        np.testing.assert_allclose(
            np.linalg.norm(values, axis=1),
            np.ones(values.shape[0]),
            atol=1e-12,
        )

    @staticmethod
    def _robot_reference_matrices(
        model: mujoco.MjModel,
        motion: MotionDataConfig,
    ) -> np.ndarray:
        qpos = model.qpos0.copy()
        qpos[3:7] = motion.resolved_orientation_t_pose_robot_base_quaternion_wxyz
        for joint_name, value in motion.resolved_orientation_t_pose_robot_joint_positions.items():
            joint_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )
            if joint_id < 0:
                raise AssertionError(f"Unknown T-pose joint: {joint_name}")
            joint_range = model.jnt_range[joint_id]
            if not joint_range[0] <= value <= joint_range[1]:
                raise AssertionError(
                    f"T-pose value {value} is outside {joint_name} limits {joint_range}",
                )
            qpos[int(model.jnt_qposadr[joint_id])] = value

        data = mujoco.MjData(model)
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        return np.asarray(
            [
                data.xmat[
                    mujoco.mj_name2id(
                        model,
                        mujoco.mjtObj.mjOBJ_BODY,
                        link_name,
                    )
                ].reshape(3, 3)
                for link_name in motion.resolved_orientation_joints_mapping.values()
            ],
        )


if __name__ == "__main__":
    unittest.main()
