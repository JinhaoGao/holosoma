# ruff: noqa: PT009

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "holosoma_retargeting"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from holosoma_retargeting.config_types.data_type import MotionDataConfig  # noqa: E402
from holosoma_retargeting.config_types.retargeter import RetargeterConfig  # noqa: E402
from holosoma_retargeting.config_types.robot import RobotConfig  # noqa: E402
from holosoma_retargeting.config_types.task import TaskConfig  # noqa: E402
from holosoma_retargeting.examples.robot_retarget import (  # noqa: E402
    build_retargeter_kwargs_from_config,
    create_task_constants,
)
from holosoma_retargeting.src.interaction_mesh_retargeter import (  # noqa: E402
    InteractionMeshRetargeter,
)


def _build_e1_retargeter(
    orientation_weights: dict[str, float],
    *,
    q_a_init_idx: int = 0,
    orientation_alignment_mode: str = "t_pose",
) -> InteractionMeshRetargeter:
    robot_urdf = PACKAGE_ROOT / "holosoma_retargeting" / "models" / "e1" / "e1_23dof.urdf"
    constants = create_task_constants(
        RobotConfig(
            robot_type="e1",
            robot_urdf_file=str(robot_urdf),
        ),
        MotionDataConfig(data_format="noetix_mocap", robot_type="e1"),
        TaskConfig(object_name="ground"),
        "robot_only",
    )
    config = RetargeterConfig(
        q_a_init_idx=q_a_init_idx,
        activate_foot_sticking=False,
        orientation_weights=orientation_weights,
        orientation_alignment_mode=orientation_alignment_mode,
    )
    return InteractionMeshRetargeter(
        **build_retargeter_kwargs_from_config(
            config,
            constants,
            None,
            "robot_only",
        )
    )


class OrientationObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retargeter = _build_e1_retargeter({"LeftArm": 1.0})
        cls.q = cls.retargeter.robot_model.qpos0.copy()

    def test_t_pose_alignment_matches_robot_reference_for_identity_human_pose(
        self,
    ):
        all_weights = dict.fromkeys(
            self.retargeter.orientation_joints_mapping,
            0.0,
        )
        retargeter = _build_e1_retargeter(all_weights)
        quaternions = np.zeros(
            (2, len(retargeter.demo_joints), 4),
            dtype=np.float64,
        )
        quaternions[..., 0] = 1.0

        targets, alignments = retargeter._prepare_orientation_targets(
            quaternions,
            retargeter.robot_model.qpos0.copy(),
            num_frames=2,
        )
        reference_robot_q = retargeter._orientation_t_pose_robot_qpos()
        robot_matrices, _ = retargeter._get_robot_link_orientation_data(
            reference_robot_q,
            retargeter.orientation_robot_link_names,
            with_jacobians=False,
        )

        self.assertEqual(alignments.shape, (13, 3, 3))
        np.testing.assert_allclose(alignments, robot_matrices, atol=1e-12)
        np.testing.assert_allclose(targets[0], robot_matrices, atol=1e-12)
        np.testing.assert_allclose(targets[1], robot_matrices, atol=1e-12)
        np.testing.assert_allclose(
            retargeter.orientation_t_pose_robot_base_quaternion_wxyz,
            np.sqrt(0.5) * np.asarray([1.0, 0.0, 0.0, 1.0]),
            atol=1e-12,
        )

    def test_e1_reference_pose_is_a_geometric_t_pose(self):
        retargeter = _build_e1_retargeter({"LeftArm": 0.0, "RightArm": 0.0})
        reference_q = retargeter._orientation_t_pose_robot_qpos()
        expected_joint_positions = {
            "l_arm_shoulder_roll_joint": np.pi / 2.0,
            "r_arm_shoulder_roll_joint": -np.pi / 2.0,
        }
        for joint_name, expected_position in expected_joint_positions.items():
            joint_id = mujoco.mj_name2id(
                retargeter.robot_model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )
            qpos_address = int(retargeter.robot_model.jnt_qposadr[joint_id])
            self.assertAlmostEqual(reference_q[qpos_address], expected_position)

        data = mujoco.MjData(retargeter.robot_model)
        data.qpos[:] = reference_q
        mujoco.mj_forward(retargeter.robot_model, data)
        shoulder_to_hand_vectors = []
        for shoulder_link, hand_link in (
            ("l_arm_shoulder_yaw_link", "l_hand_sphere_link"),
            ("r_arm_shoulder_yaw_link", "r_hand_sphere_link"),
        ):
            shoulder_id = mujoco.mj_name2id(
                retargeter.robot_model,
                mujoco.mjtObj.mjOBJ_BODY,
                shoulder_link,
            )
            hand_id = mujoco.mj_name2id(
                retargeter.robot_model,
                mujoco.mjtObj.mjOBJ_BODY,
                hand_link,
            )
            shoulder_to_hand_vectors.append(data.xpos[hand_id] - data.xpos[shoulder_id])

        left_vector, right_vector = shoulder_to_hand_vectors
        self.assertLess(abs(left_vector[2]), 1e-3)
        self.assertLess(abs(right_vector[2]), 1e-3)
        self.assertLess(left_vector[0], -0.25)
        self.assertGreater(right_vector[0], 0.25)

    def test_t_pose_alignment_does_not_bake_in_first_motion_frame(self):
        retargeter = _build_e1_retargeter({"LeftArm": 1.0})
        quaternions = np.zeros(
            (2, len(retargeter.demo_joints), 4),
            dtype=np.float64,
        )
        quaternions[..., 0] = 1.0
        shoulder_quaternion_xyzw = Rotation.from_euler(
            "x",
            40.0,
            degrees=True,
        ).as_quat()
        shoulder_index = retargeter.demo_joints.index("LeftArm")
        quaternions[:, shoulder_index] = shoulder_quaternion_xyzw[[3, 0, 1, 2]]

        targets, _ = retargeter._prepare_orientation_targets(
            quaternions,
            retargeter.robot_model.qpos0.copy(),
            num_frames=2,
        )
        reference_robot_q = retargeter._orientation_t_pose_robot_qpos()
        robot_matrices, _ = retargeter._get_robot_link_orientation_data(
            reference_robot_q,
            retargeter.orientation_robot_link_names,
            with_jacobians=False,
        )
        error_angle = np.linalg.norm(
            retargeter._so3_error_vectors(
                targets[0],
                robot_matrices,
            )[0]
        )

        self.assertAlmostEqual(np.degrees(error_angle), 40.0, places=8)

    def test_legacy_first_frame_mode_still_matches_initial_robot_frame(self):
        retargeter = _build_e1_retargeter(
            {"LeftArm": 1.0},
            orientation_alignment_mode="first_frame",
        )
        quaternions = np.zeros(
            (2, len(retargeter.demo_joints), 4),
            dtype=np.float64,
        )
        quaternions[..., 0] = 1.0
        shoulder_quaternion_xyzw = Rotation.from_euler(
            "x",
            40.0,
            degrees=True,
        ).as_quat()
        quaternions[:, retargeter.demo_joints.index("LeftArm")] = shoulder_quaternion_xyzw[[3, 0, 1, 2]]

        targets, _ = retargeter._prepare_orientation_targets(
            quaternions,
            retargeter.robot_model.qpos0.copy(),
            num_frames=2,
        )
        robot_matrices, _ = retargeter._get_robot_link_orientation_data(
            retargeter.robot_model.qpos0.copy(),
            retargeter.orientation_robot_link_names,
            with_jacobians=False,
        )

        np.testing.assert_allclose(targets[0], robot_matrices, atol=1e-12)

    def test_robot_only_object_pose_does_not_overwrite_arm_joints(self):
        q_locked_list = np.arange(
            2 * self.retargeter.robot_model.nq,
            dtype=np.float64,
        ).reshape(2, self.retargeter.robot_model.nq)
        expected = q_locked_list.copy()
        object_poses = np.zeros((2, 7), dtype=np.float64)
        object_poses[:, 3] = 1.0

        self.retargeter._apply_dynamic_object_poses(
            q_locked_list,
            object_poses,
        )

        np.testing.assert_array_equal(q_locked_list, expected)

    def test_world_angular_jacobian_matches_finite_difference(self):
        matrices, jacobians = self.retargeter._get_robot_link_orientation_data(
            self.q,
            self.retargeter.orientation_robot_link_names,
            with_jacobians=True,
        )
        self.assertIsNotNone(jacobians)
        joint_id = mujoco.mj_name2id(
            self.retargeter.robot_model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "l_arm_shoulder_yaw_joint",
        )
        qpos_address = int(self.retargeter.robot_model.jnt_qposadr[joint_id])
        reduced_index = int(np.flatnonzero(self.retargeter.q_a_indices == qpos_address)[0])
        epsilon = 1e-7
        perturbed_q = self.q.copy()
        perturbed_q[qpos_address] += epsilon
        perturbed_matrices, _ = self.retargeter._get_robot_link_orientation_data(
            perturbed_q,
            self.retargeter.orientation_robot_link_names,
            with_jacobians=False,
        )
        finite_difference = (
            self.retargeter._so3_error_vectors(
                perturbed_matrices,
                matrices,
            )[0]
            / epsilon
        )

        np.testing.assert_allclose(
            finite_difference,
            jacobians[0, :, reduced_index],
            atol=2e-6,
            rtol=2e-6,
        )

    def test_free_base_angular_jacobian_matches_normalized_qpos_step(self):
        retargeter = _build_e1_retargeter(
            {"Hips": 1.0},
            q_a_init_idx=-7,
        )
        q = retargeter.robot_model.qpos0.copy()
        matrices, jacobians = retargeter._get_robot_link_orientation_data(
            q,
            retargeter.orientation_robot_link_names,
            with_jacobians=True,
        )
        qx_address = 4
        reduced_index = int(np.flatnonzero(retargeter.q_a_indices == qx_address)[0])
        epsilon = 1e-7
        perturbed_q = q.copy()
        perturbed_q[qx_address] += epsilon
        perturbed_q[3:7] /= np.linalg.norm(perturbed_q[3:7])
        perturbed_matrices, _ = retargeter._get_robot_link_orientation_data(
            perturbed_q,
            retargeter.orientation_robot_link_names,
            with_jacobians=False,
        )
        finite_difference = (
            retargeter._so3_error_vectors(
                perturbed_matrices,
                matrices,
            )[0]
            / epsilon
        )

        np.testing.assert_allclose(
            finite_difference,
            jacobians[0, :, reduced_index],
            atol=2e-6,
            rtol=2e-6,
        )

    def test_jacobian_step_reduces_geodesic_error(self):
        current, jacobians = self.retargeter._get_robot_link_orientation_data(
            self.q,
            self.retargeter.orientation_robot_link_names,
            with_jacobians=True,
        )
        joint_id = mujoco.mj_name2id(
            self.retargeter.robot_model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "l_arm_shoulder_yaw_joint",
        )
        qpos_address = int(self.retargeter.robot_model.jnt_qposadr[joint_id])
        reduced_index = int(np.flatnonzero(self.retargeter.q_a_indices == qpos_address)[0])
        axis = jacobians[0, :, reduced_index]
        axis /= np.linalg.norm(axis)
        angle = 0.02
        target = (Rotation.from_rotvec(axis * angle).as_matrix() @ current[0]).reshape(1, 3, 3)
        error_before = np.linalg.norm(self.retargeter._so3_error_vectors(target, current))

        updated_q = self.q.copy()
        updated_q[qpos_address] += angle
        updated, _ = self.retargeter._get_robot_link_orientation_data(
            updated_q,
            self.retargeter.orientation_robot_link_names,
            with_jacobians=False,
        )
        error_after = np.linalg.norm(self.retargeter._so3_error_vectors(target, updated))

        self.assertLess(error_after, 0.05 * error_before)

    def test_quaternion_sign_does_not_change_orientation_matrix(self):
        quaternion = np.array([0.5, 0.5, -0.5, 0.5])
        matrices = self.retargeter._wxyz_to_matrices(np.stack([quaternion, -quaternion]))
        np.testing.assert_allclose(matrices[0], matrices[1], atol=1e-12)

    def test_positive_weights_require_source_orientations(self):
        with self.assertRaisesRegex(ValueError, "require source"):
            self.retargeter._prepare_orientation_targets(
                None,
                self.q,
                num_frames=2,
            )

    def test_zero_weight_preserves_disabled_path(self):
        retargeter = _build_e1_retargeter({"LeftArm": 0.0})
        quaternions = np.zeros(
            (3, len(retargeter.demo_joints), 4),
            dtype=np.float64,
        )
        quaternions[..., 0] = 1.0
        targets, alignments = retargeter._prepare_orientation_targets(
            quaternions,
            retargeter.robot_model.qpos0.copy(),
            num_frames=3,
        )
        self.assertFalse(retargeter.orientation_tracking_enabled)
        self.assertTrue(retargeter.orientation_diagnostics_enabled)
        self.assertEqual(targets.shape, (3, 1, 3, 3))
        self.assertEqual(alignments.shape, (1, 3, 3))

    def test_invalid_orientation_weights_fail_fast(self):
        for weights, message in (
            ({"Unknown": 1.0}, "outside the orientation mapping"),
            ({"LeftArm": -1.0}, "non-negative"),
        ):
            with self.subTest(
                weights=weights,
            ), self.assertRaisesRegex(ValueError, message):
                _build_e1_retargeter(weights)

    def test_mujoco_named_accessor_body_id_is_scalarized(self):
        self.assertEqual(
            self.retargeter._as_scalar_id(np.array([7]), "geom.bodyid"),
            7,
        )
        with self.assertRaisesRegex(ValueError, "exactly one ID"):
            self.retargeter._as_scalar_id(
                np.array([7, 8]),
                "geom.bodyid",
            )
