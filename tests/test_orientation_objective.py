# ruff: noqa: CPY001, PT009, PT027

from __future__ import annotations

import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import cvxpy as cp
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "holosoma_retargeting"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from holosoma_retargeting.config_types.data_type import MotionDataConfig  # noqa: E402
from holosoma_retargeting.config_types.retargeter import (  # noqa: E402
    FootLockConfig,
    RetargeterConfig,
)
from holosoma_retargeting.config_types.robot import RobotConfig  # noqa: E402
from holosoma_retargeting.config_types.task import TaskConfig  # noqa: E402
from holosoma_retargeting.examples.robot_retarget import (  # noqa: E402
    build_retargeter_kwargs_from_config,
    create_task_constants,
)
from holosoma_retargeting.src.interaction_mesh_retargeter import (  # noqa: E402
    ConstraintMode,
    InteractionMeshRetargeter,
    NonlinearConstraintResiduals,
    SQPIterationResult,
)


def _build_e1_retargeter(
    orientation_weights: dict[str, float],
    *,
    q_a_init_idx: int = 0,
    orientation_alignment_mode: str = "t_pose",
    activate_foot_sticking: bool = False,
    foot_lock: FootLockConfig | None = None,
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
        activate_foot_sticking=activate_foot_sticking,
        foot_lock=foot_lock or FootLockConfig(),
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


def _build_g1_retargeter(
    orientation_weights: dict[str, float],
    *,
    q_a_init_idx: int = -7,
) -> InteractionMeshRetargeter:
    robot_urdf = PACKAGE_ROOT / "holosoma_retargeting" / "models" / "g1" / "g1_29dof.urdf"
    constants = create_task_constants(
        RobotConfig(
            robot_type="g1",
            robot_urdf_file=str(robot_urdf),
        ),
        MotionDataConfig(data_format="noetix_mocap", robot_type="g1"),
        TaskConfig(object_name="ground"),
        "robot_only",
    )
    config = RetargeterConfig(
        q_a_init_idx=q_a_init_idx,
        activate_foot_sticking=False,
        orientation_weights=orientation_weights,
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

        self.assertEqual(alignments.shape, (15, 3, 3))
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

    def test_g1_t_pose_alignment_and_reference_pose_are_geometric(self):
        retargeter = _build_g1_retargeter(
            dict.fromkeys(
                MotionDataConfig(
                    data_format="noetix_mocap",
                    robot_type="g1",
                ).resolved_orientation_joints_mapping,
                0.0,
            )
        )
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
        reference_q = retargeter._orientation_t_pose_robot_qpos()
        robot_matrices, _ = retargeter._get_robot_link_orientation_data(
            reference_q,
            retargeter.orientation_robot_link_names,
            with_jacobians=False,
        )

        self.assertEqual(alignments.shape, (15, 3, 3))
        np.testing.assert_allclose(targets[0], robot_matrices, atol=1e-12)
        np.testing.assert_allclose(targets[1], robot_matrices, atol=1e-12)
        np.testing.assert_allclose(
            alignments,
            np.swapaxes(
                retargeter.orientation_reference_human_matrices,
                -1,
                -2,
            )
            @ robot_matrices,
            atol=1e-12,
        )

        data = mujoco.MjData(retargeter.robot_model)
        data.qpos[:] = reference_q
        mujoco.mj_forward(retargeter.robot_model, data)
        arm_segment_vectors = []
        for shoulder_link, elbow_link, hand_link in (
            (
                "left_shoulder_roll_link",
                "left_elbow_link",
                "left_rubber_hand_link",
            ),
            (
                "right_shoulder_roll_link",
                "right_elbow_link",
                "right_rubber_hand_link",
            ),
        ):
            shoulder_id = mujoco.mj_name2id(
                retargeter.robot_model,
                mujoco.mjtObj.mjOBJ_BODY,
                shoulder_link,
            )
            elbow_id = mujoco.mj_name2id(
                retargeter.robot_model,
                mujoco.mjtObj.mjOBJ_BODY,
                elbow_link,
            )
            hand_id = mujoco.mj_name2id(
                retargeter.robot_model,
                mujoco.mjtObj.mjOBJ_BODY,
                hand_link,
            )
            arm_segment_vectors.append(
                (
                    data.xpos[elbow_id] - data.xpos[shoulder_id],
                    data.xpos[hand_id] - data.xpos[elbow_id],
                )
            )

        left_segments, right_segments = arm_segment_vectors
        for segment in (*left_segments, *right_segments):
            self.assertLess(abs(segment[1]), 1e-10)
            self.assertLess(abs(segment[2]), 1e-10)
        for segment in left_segments:
            self.assertLess(segment[0], -0.15)
        for segment in right_segments:
            self.assertGreater(segment[0], 0.15)

    def test_g1_motion_rotation_is_applied_before_per_link_t_pose_offset(self):
        retargeter = _build_g1_retargeter({"LeftArm": 0.0})
        quaternions = np.zeros(
            (2, len(retargeter.demo_joints), 4),
            dtype=np.float64,
        )
        quaternions[..., 0] = 1.0
        human_motion = Rotation.from_euler(
            "xy",
            [31.0, -23.0],
            degrees=True,
        ).as_matrix()
        human_motion_xyzw = Rotation.from_matrix(human_motion).as_quat()
        quaternions[
            1,
            retargeter.demo_joints.index("LeftArm"),
        ] = human_motion_xyzw[[3, 0, 1, 2]]

        targets, fixed_offsets = retargeter._prepare_orientation_targets(
            quaternions,
            retargeter.robot_model.qpos0.copy(),
            num_frames=2,
        )

        expected = human_motion @ fixed_offsets[0]
        wrong_order = fixed_offsets[0] @ human_motion
        np.testing.assert_allclose(targets[1, 0], expected, atol=1e-12)
        self.assertGreater(np.linalg.norm(expected - wrong_order), 0.1)
        self.assertGreater(
            np.linalg.norm(fixed_offsets[0] - np.eye(3)),
            0.1,
        )

    def test_target_world_delta_precedes_source_and_alignment(self):
        retargeter = _build_g1_retargeter({"LeftArm": 0.0})
        quaternions = np.zeros(
            (2, len(retargeter.demo_joints), 4),
            dtype=np.float64,
        )
        quaternions[..., 0] = 1.0
        source_matrix = Rotation.from_euler(
            "x",
            31.0,
            degrees=True,
        ).as_matrix()
        source_xyzw = Rotation.from_matrix(source_matrix).as_quat()
        quaternions[
            1,
            retargeter.demo_joints.index("LeftArm"),
        ] = source_xyzw[[3, 0, 1, 2]]
        delta_matrix = Rotation.from_euler(
            "z",
            -47.0,
            degrees=True,
        ).as_matrix()
        delta_xyzw = Rotation.from_matrix(delta_matrix).as_quat()
        deltas = np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                delta_xyzw[[3, 0, 1, 2]],
            ],
        )

        targets, fixed_offsets = retargeter._prepare_orientation_targets(
            quaternions,
            retargeter.robot_model.qpos0.copy(),
            num_frames=2,
            orientation_target_world_rotation_deltas_wxyz=deltas,
        )

        expected = delta_matrix @ source_matrix @ fixed_offsets[0]
        wrong_order = source_matrix @ delta_matrix @ fixed_offsets[0]
        np.testing.assert_allclose(targets[1, 0], expected, atol=1e-12)
        self.assertGreater(np.linalg.norm(expected - wrong_order), 0.1)

    def test_first_frame_alignment_uses_target_world_source_reference(self):
        retargeter = _build_e1_retargeter(
            {"LeftArm": 1.0},
            orientation_alignment_mode="first_frame",
        )
        quaternions = np.zeros(
            (2, len(retargeter.demo_joints), 4),
            dtype=np.float64,
        )
        quaternions[..., 0] = 1.0
        delta_matrix = Rotation.from_euler(
            "z",
            35.0,
            degrees=True,
        ).as_matrix()
        delta_xyzw = Rotation.from_matrix(delta_matrix).as_quat()
        delta_wxyz = delta_xyzw[[3, 0, 1, 2]]
        deltas = np.repeat(delta_wxyz[None, :], 2, axis=0)

        targets, _ = retargeter._prepare_orientation_targets(
            quaternions,
            retargeter.robot_model.qpos0.copy(),
            num_frames=2,
            orientation_target_world_rotation_deltas_wxyz=deltas,
        )
        robot_matrices, _ = retargeter._get_robot_link_orientation_data(
            retargeter.robot_model.qpos0.copy(),
            retargeter.orientation_robot_link_names,
            with_jacobians=False,
        )

        np.testing.assert_allclose(
            retargeter.orientation_reference_human_matrices[0],
            delta_matrix,
            atol=1e-12,
        )
        np.testing.assert_allclose(targets[0], robot_matrices, atol=1e-12)

    def test_source_orientation_evidence_is_not_normalized_or_aliased(self):
        input_quaternions = np.zeros(
            (2, len(self.retargeter.demo_joints), 4),
            dtype=np.float32,
        )
        input_quaternions[..., 0] = np.float32(0.9995)
        original_bytes = input_quaternions.tobytes(order="C")

        saved, normalized_math, names = self.retargeter._validate_source_orientations(
            input_quaternions,
            tuple(self.retargeter.demo_joints),
            num_frames=2,
        )

        self.assertEqual(names, tuple(self.retargeter.demo_joints))
        self.assertEqual(saved.dtype, np.dtype(np.float32))
        self.assertEqual(saved.tobytes(order="C"), original_bytes)
        np.testing.assert_allclose(
            np.linalg.norm(normalized_math, axis=-1),
            1.0,
            atol=1e-12,
        )
        input_quaternions[0, 0, 0] = 0.5
        self.assertEqual(saved.tobytes(order="C"), original_bytes)

    def test_target_world_delta_rejects_invalid_inputs(self):
        quaternions = np.zeros(
            (2, len(self.retargeter.demo_joints), 4),
            dtype=np.float64,
        )
        quaternions[..., 0] = 1.0
        invalid_inputs = {
            "shape": np.zeros((1, 4)),
            "zero": np.zeros((2, 4)),
            "nonfinite": np.asarray(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [np.nan, 0.0, 0.0, 0.0],
                ]
            ),
        }

        for case, deltas in invalid_inputs.items():
            with self.subTest(case=case), self.assertRaises(ValueError):
                self.retargeter._prepare_orientation_targets(
                    quaternions,
                    self.q,
                    num_frames=2,
                    orientation_target_world_rotation_deltas_wxyz=deltas,
                )

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

    def test_complete_qpos_initialization_is_sliced_for_actuated_only_mode(self):
        retargeter = _build_e1_retargeter(
            {"LeftArm": 0.0},
            q_a_init_idx=0,
        )
        complete_robot_qpos = np.arange(
            7 + retargeter.task_constants.ROBOT_DOF,
            dtype=np.float64,
        )

        reduced = retargeter._initial_optimization_values(
            complete_robot_qpos,
        )

        np.testing.assert_array_equal(
            reduced,
            complete_robot_qpos[retargeter.q_a_indices],
        )
        np.testing.assert_array_equal(
            retargeter._initial_optimization_values(reduced),
            reduced,
        )
        locked = retargeter._initial_locked_qpos(complete_robot_qpos)
        np.testing.assert_array_equal(
            locked[: 7 + retargeter.task_constants.ROBOT_DOF],
            complete_robot_qpos,
        )
        with self.assertRaisesRegex(ValueError, "q_a_init"):
            retargeter._initial_optimization_values(np.zeros(2))

    def test_waist_only_mode_maps_full_qpos_metadata_to_reduced_indices(self):
        retargeter = _build_g1_retargeter(
            {},
            q_a_init_idx=12,
        )

        self.assertEqual(retargeter.q_a_indices[0], 19)
        np.testing.assert_allclose(retargeter.Q_diag[:2], [0.2, 0.2])
        self.assertTrue(np.all(retargeter.track_nominal_indices < len(retargeter.q_a_indices)))

    def test_reduced_manipulator_jacobians_support_active_qpos_suffixes(self):
        residuals = NonlinearConstraintResiduals(
            ground_non_penetration=0.0,
            object_non_penetration=0.0,
            foot_sticking=0.0,
            foot_lock=0.0,
            self_collision=0.0,
            joint_limits=0.0,
        )
        for q_a_init_idx in (0, 12):
            foot_constraints_active = q_a_init_idx == 0
            retargeter = _build_e1_retargeter(
                {},
                q_a_init_idx=q_a_init_idx,
                activate_foot_sticking=foot_constraints_active,
                foot_lock=FootLockConfig(
                    enable=foot_constraints_active,
                    windows={
                        "left": [(0, 0)],
                        "right": [(0, 0)],
                    },
                ),
            )
            q = retargeter.robot_model.qpos0.copy()
            jacobians, _, _ = retargeter._calc_manipulator_jacobians(
                q,
                links=retargeter.laplacian_match_links,
                obj_frame=False,
            )
            vertex_count = len(retargeter.laplacian_match_links)

            with self.subTest(
                q_a_init_idx=q_a_init_idx,
            ):
                self.assertTrue(all(jacobian.shape == (3, retargeter.nq_a) for jacobian in jacobians.values()))
                with ExitStack() as stack:
                    stack.enter_context(
                        mock.patch.object(
                            retargeter,
                            "_update_jacobians_and_phis_from_q",
                            return_value=({}, {}),
                        ),
                    )
                    stack.enter_context(
                        mock.patch.object(
                            retargeter,
                            "_compute_self_collision_constraints",
                            return_value=({}, {}),
                        ),
                    )
                    stack.enter_context(
                        mock.patch.object(
                            retargeter,
                            "_evaluate_nonlinear_constraint_residuals",
                            return_value=residuals,
                        ),
                    )
                    result = retargeter.solve_single_iteration(
                        q_locked=q,
                        q_a_n_last=q[retargeter.q_a_indices],
                        q_t_last=q,
                        target_laplacian=np.zeros(
                            (vertex_count, 3),
                            dtype=np.float64,
                        ),
                        adj_list=[[] for _ in range(vertex_count)],
                        obj_pts_local=np.empty((0, 3), dtype=np.float64),
                        foot_sticking={
                            "left": foot_constraints_active,
                            "right": foot_constraints_active,
                        },
                        frame_idx=0,
                        requested_mode=ConstraintMode(
                            "normal" if foot_constraints_active else "inactive",
                        ),
                    )

                self.assertEqual(result.q.shape, q.shape)

    def test_default_manipulator_jacobian_contract_is_unchanged(self):
        retargeter = _build_e1_retargeter(
            {},
            q_a_init_idx=-7,
        )
        q = retargeter.robot_model.qpos0.copy()
        jacobians, _, _ = retargeter._calc_manipulator_jacobians(
            q,
            links=retargeter.laplacian_match_links,
            obj_frame=False,
        )

        np.testing.assert_array_equal(
            retargeter.q_a_indices,
            np.arange(retargeter.nq_a),
        )
        for jacobian in jacobians.values():
            np.testing.assert_array_equal(
                jacobian,
                jacobian[:, retargeter.q_a_indices],
            )

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

    @staticmethod
    def _candidate(
        q: np.ndarray,
        *,
        cost: float,
        ground_violation: float,
        mode: ConstraintMode | None = None,
    ) -> SQPIterationResult:
        return SQPIterationResult(
            q=q,
            linearized_cost=cost,
            constraint_mode=mode or ConstraintMode("inactive"),
            residuals=NonlinearConstraintResiduals(
                ground_non_penetration=ground_violation,
                object_non_penetration=0.0,
                foot_sticking=0.0,
                foot_lock=0.0,
                self_collision=0.0,
                joint_limits=0.0,
            ),
        )

    def test_lower_local_cost_with_seven_mm_penetration_cannot_beat_feasible_candidate(
        self,
    ):
        q = self.retargeter.robot_model.qpos0.copy()
        feasible = self._candidate(
            q,
            cost=10.0,
            ground_violation=0.0,
        )
        penetrating = self._candidate(
            q + 1e-4,
            cost=0.0,
            ground_violation=0.007,
        )

        selected = self.retargeter._select_feasible_candidate(
            [feasible, penetrating],
        )

        self.assertIs(selected, feasible)

    def test_feasible_candidate_selection_prefers_stricter_constraint_mode(
        self,
    ):
        q = self.retargeter.robot_model.qpos0.copy()
        strict = self._candidate(
            q,
            cost=10.0,
            ground_violation=0.0,
            mode=ConstraintMode("normal"),
        )
        released = self._candidate(
            q,
            cost=0.0,
            ground_violation=0.0,
            mode=ConstraintMode(
                "released",
                object_non_penetration_released=True,
                trust_region_released=True,
            ),
        )

        selected = self.retargeter._select_feasible_candidate(
            [strict, released],
        )

        self.assertIs(selected, strict)

    def test_released_foot_mode_still_measures_diagnostic_violation(self):
        retargeter = _build_e1_retargeter(
            {},
            q_a_init_idx=-7,
        )
        retargeter.activate_foot_sticking = True
        previous = retargeter.robot_model.qpos0.copy()
        candidate = previous.copy()
        candidate[0] += 0.05

        residuals = retargeter._evaluate_nonlinear_constraint_residuals(
            candidate,
            q_t_last=previous,
            foot_sticking={"left": True, "right": False},
            frame_idx=0,
            mode=ConstraintMode("released"),
        )

        self.assertAlmostEqual(
            residuals.foot_sticking,
            0.05 - retargeter.foot_sticking_fallback_tolerance,
            places=6,
        )
        self.assertAlmostEqual(
            residuals.hard_max_violation(ConstraintMode("released")),
            max(
                residuals.ground_non_penetration,
                residuals.object_non_penetration,
                residuals.foot_lock,
                residuals.self_collision,
                residuals.joint_limits,
            ),
        )

    def test_iterate_fails_explicitly_when_no_nonlinear_feasible_candidate_exists(
        self,
    ):
        q = self.retargeter.robot_model.qpos0.copy()
        penetrating = self._candidate(
            q,
            cost=0.0,
            ground_violation=0.007,
        )
        with mock.patch.object(
            self.retargeter,
            "solve_single_iteration",
            return_value=penetrating,
        ), self.assertRaisesRegex(
            RuntimeError,
            "no true-geometry feasible candidate",
        ):
            self.retargeter.iterate(
                q_locked=q,
                q_n=q.copy(),
                q_t_last=q,
                target_laplacian=np.empty((0, 3)),
                adj_list=[],
                obj_pts_local=np.empty((0, 3)),
                foot_sticking={"left": False, "right": False},
                n_iter=2,
            )

    def test_fallback_solver_returns_explicit_constraint_mode(self):
        variable = cp.Variable(name="fallback_test")
        objective = cp.Minimize(cp.square(variable))
        normal = [variable >= 1.0, variable <= -1.0]
        relaxed = [variable == 0.0]

        normal_problem, normal_mode = self.retargeter._solve_with_foot_sticking_fallback(
            objective=objective,
            base_constraints=[],
            object_non_penetration_constraints=[],
            foot_sticking_constraints=[variable == 0.0],
            foot_sticking_fallback_constraints=[],
            solver_kwargs={"verbose": False},
            remove_soc_on_failure=False,
            release_on_failure=True,
            release_object_non_penetration_on_failure=False,
        )
        self.assertIn(
            normal_problem.status,
            (cp.OPTIMAL, cp.OPTIMAL_INACCURATE),
        )
        self.assertEqual(normal_mode.foot_sticking, "normal")

        problem, mode = self.retargeter._solve_with_foot_sticking_fallback(
            objective=objective,
            base_constraints=[],
            object_non_penetration_constraints=[],
            foot_sticking_constraints=normal,
            foot_sticking_fallback_constraints=relaxed,
            solver_kwargs={"verbose": False},
            remove_soc_on_failure=False,
            release_on_failure=True,
            release_object_non_penetration_on_failure=False,
        )

        self.assertIn(problem.status, (cp.OPTIMAL, cp.OPTIMAL_INACCURATE))
        self.assertEqual(mode.foot_sticking, "relaxed")
        self.assertFalse(mode.object_non_penetration_released)

        released_problem, released_mode = self.retargeter._solve_with_foot_sticking_fallback(
            objective=objective,
            base_constraints=[],
            object_non_penetration_constraints=[],
            foot_sticking_constraints=normal,
            foot_sticking_fallback_constraints=[],
            solver_kwargs={"verbose": False},
            remove_soc_on_failure=False,
            release_on_failure=True,
            release_object_non_penetration_on_failure=False,
        )
        self.assertIn(
            released_problem.status,
            (cp.OPTIMAL, cp.OPTIMAL_INACCURATE),
        )
        self.assertEqual(released_mode.foot_sticking, "released")

        object_problem, object_mode = self.retargeter._solve_with_foot_sticking_fallback(
            objective=objective,
            base_constraints=[variable >= 0.0],
            object_non_penetration_constraints=[variable <= -1.0],
            foot_sticking_constraints=[],
            foot_sticking_fallback_constraints=[],
            solver_kwargs={"verbose": False},
            remove_soc_on_failure=False,
            release_on_failure=False,
            release_object_non_penetration_on_failure=True,
        )
        self.assertIn(
            object_problem.status,
            (cp.OPTIMAL, cp.OPTIMAL_INACCURATE),
        )
        self.assertEqual(object_mode.foot_sticking, "inactive")
        self.assertTrue(object_mode.object_non_penetration_released)
