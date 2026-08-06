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
from holosoma_retargeting.data_utils.generate_orientation_calibration import (
    compute_orientation_alignment_quaternions,
    robot_t_pose_qpos,
)
from holosoma_retargeting.orientation_calibration import (
    ORIENTATION_ALIGNMENT_QUATERNIONS_WXYZ,
)
from scipy.spatial.transform import Rotation


class OrientationMappingMatrixTest(unittest.TestCase):
    DATA_FORMATS = (
        "amass",
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
            set(self.DATA_FORMATS) | {"fbx_mocap"},
            set(APPROVED_DIRECT_ORIENTATION_SOURCES),
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

    def test_calibrated_sources_use_source_specific_hip_links(self) -> None:
        for data_format, human_hip, link_component in (
            ("amass", "L_Hip", "hip_roll"),
            ("gvhmr", "L_Hip", "hip_roll"),
            ("lafan", "LeftUpLeg", "hip_yaw"),
        ):
            for robot in self.ROBOTS:
                with self.subTest(data_format=data_format, robot=robot):
                    link = MotionDataConfig(
                        data_format=data_format,
                        robot_type=robot,
                    ).resolved_orientation_joints_mapping[human_hip]
                    self.assertIn(link_component, link)

    def test_e2_position_and_orientation_use_different_shoulder_links(self) -> None:
        for data_format, human_shoulder in (
            ("amass", "L_Shoulder"),
            ("gvhmr", "L_Shoulder"),
            ("lafan", "LeftArm"),
            ("fbx_mocap", "LeftArm"),
            ("mocap", "LeftArm"),
            ("noetix_mocap", "LeftArm"),
            ("omomo", "L_Shoulder"),
        ):
            with self.subTest(data_format=data_format):
                motion = MotionDataConfig(
                    data_format=data_format,
                    robot_type="e2",
                )
                self.assertEqual(
                    motion.resolved_joints_mapping[human_shoulder],
                    "l_arm_shoulder_roll_link",
                )
                self.assertEqual(
                    motion.resolved_orientation_joints_mapping[human_shoulder],
                    "l_arm_shoulder_yaw_link",
                )

    def test_position_and_orientation_mapping_differences_are_explicit(self) -> None:
        for data_format in self.DATA_FORMATS:
            for robot in self.ROBOTS:
                with self.subTest(data_format=data_format, robot=robot):
                    motion = MotionDataConfig(
                        data_format=data_format,
                        robot_type=robot,
                    )
                    position_mapping = motion.resolved_joints_mapping
                    orientation_mapping = motion.resolved_orientation_joints_mapping
                    self.assertIsNot(position_mapping, orientation_mapping)
                    position_only = set(position_mapping).difference(orientation_mapping)
                    orientation_only = set(orientation_mapping).difference(position_mapping)
                    shared_differences = {
                        name
                        for name in set(position_mapping).intersection(orientation_mapping)
                        if position_mapping[name] != orientation_mapping[name]
                    }
                    expected_position_only = {"Spine"} if data_format == "fbx_mocap" else set()
                    expected_orientation_only = {"Hips"} if data_format == "fbx_mocap" else set()
                    self.assertEqual(position_only, expected_position_only)
                    self.assertEqual(orientation_only, expected_orientation_only)
                    if robot == "e2":
                        shoulder_names = (
                            {"L_Shoulder", "R_Shoulder"}
                            if data_format in {"amass", "gvhmr", "omomo"}
                            else {"LeftArm", "RightArm"}
                        )
                        self.assertEqual(shared_differences, shoulder_names)
                    else:
                        self.assertEqual(shared_differences, set())

    def test_fbx_uses_roll_position_shoulders_and_dynamic_source_bind_frames(self) -> None:
        for robot in self.ROBOTS:
            with self.subTest(robot=robot):
                fbx = MotionDataConfig(data_format="fbx_mocap", robot_type=robot)
                noetix = MotionDataConfig(
                    data_format="noetix_mocap",
                    robot_type=robot,
                )
                self.assertEqual(
                    fbx.resolved_orientation_joints_mapping,
                    noetix.resolved_orientation_joints_mapping,
                )
                self.assertIsNot(
                    fbx.resolved_orientation_joints_mapping,
                    noetix.resolved_orientation_joints_mapping,
                )
                fbx_only = set(fbx.resolved_joints_mapping).difference(
                    noetix.resolved_joints_mapping,
                )
                noetix_only = set(noetix.resolved_joints_mapping).difference(
                    fbx.resolved_joints_mapping,
                )
                differing_shared_joints = {
                    name
                    for name in set(fbx.resolved_joints_mapping).intersection(
                        noetix.resolved_joints_mapping,
                    )
                    if fbx.resolved_joints_mapping[name] != noetix.resolved_joints_mapping[name]
                }
                expected_differences = (
                    {"LeftArm", "RightArm"}
                    if robot in {"g1", "e1"}
                    else set()
                )
                self.assertEqual(
                    differing_shared_joints,
                    expected_differences,
                )
                self.assertEqual(fbx_only, {"Spine"})
                self.assertEqual(noetix_only, {"Hips"})
                self.assertEqual(
                    fbx.resolved_joints_mapping["Spine"],
                    noetix.resolved_joints_mapping["Hips"],
                )
                self.assertIn(
                    "shoulder_roll_link",
                    fbx.resolved_joints_mapping["LeftArm"],
                )
                self.assertEqual(fbx.resolved_orientation_t_pose_human_quaternions_wxyz, {})
                self.assertIsNotNone(
                    fbx.resolved_orientation_t_pose_robot_base_quaternion_wxyz,
                )
                self.assertTrue(
                    fbx.resolved_orientation_t_pose_robot_joint_positions,
                )
                self.assertEqual(fbx.resolved_orientation_alignment_quaternions_wxyz, {})

    def test_position_override_does_not_implicitly_override_orientation(self) -> None:
        motion = MotionDataConfig(
            data_format="omomo",
            robot_type="e2",
            joints_mapping={"Pelvis": "base_link"},
        )

        self.assertEqual(motion.resolved_joints_mapping, {"Pelvis": "base_link"})
        self.assertEqual(
            motion.resolved_orientation_joints_mapping["L_Shoulder"],
            "l_arm_shoulder_yaw_link",
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

    def test_saved_calibrations_match_generated_robot_fk(self) -> None:
        for data_format in ("amass", "lafan", "gvhmr", "mocap"):
            for robot in self.ROBOTS:
                with self.subTest(data_format=data_format, robot=robot):
                    generated = compute_orientation_alignment_quaternions(
                        data_format,
                        robot,
                    )
                    saved = ORIENTATION_ALIGNMENT_QUATERNIONS_WXYZ[(data_format, robot)]
                    self.assertEqual(set(generated), set(saved))
                    for human_joint in generated:
                        generated_matrix = Rotation.from_quat(
                            generated[human_joint],
                            scalar_first=True,
                        ).as_matrix()
                        saved_matrix = Rotation.from_quat(
                            saved[human_joint],
                            scalar_first=True,
                        ).as_matrix()
                        np.testing.assert_allclose(
                            generated_matrix,
                            saved_matrix,
                            atol=2e-12,
                        )

    def test_robot_reference_configurations_are_geometric_t_poses(self) -> None:
        for robot in self.ROBOTS:
            with self.subTest(robot=robot):
                model = self.models[robot]
                motion = MotionDataConfig(data_format="gvhmr", robot_type=robot)
                data = mujoco.MjData(model)
                data.qpos[:] = robot_t_pose_qpos(model, motion)
                mujoco.mj_forward(model, data)
                mapping = motion.resolved_orientation_joints_mapping
                for side in ("L", "R"):
                    body_positions = np.asarray(
                        [
                            data.xpos[
                                mujoco.mj_name2id(
                                    model,
                                    mujoco.mjtObj.mjOBJ_BODY,
                                    mapping[human_joint],
                                )
                            ]
                            for human_joint in (
                                f"{side}_Shoulder",
                                f"{side}_Elbow",
                                f"{side}_Wrist",
                            )
                        ]
                    )
                    self.assertLess(np.ptp(body_positions[:, 2]), 0.01)
                    self.assertGreater(
                        np.linalg.norm(body_positions[-1, :2] - body_positions[0, :2]),
                        0.25,
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
