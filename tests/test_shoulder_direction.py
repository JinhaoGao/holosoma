# ruff: noqa: CPY001, PT009

from __future__ import annotations

import sys
import unittest
from contextlib import ExitStack
from dataclasses import fields
from inspect import signature
from pathlib import Path
from unittest import mock

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "holosoma_retargeting"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from holosoma_retargeting.config_types.data_type import MotionDataConfig  # noqa: E402
from holosoma_retargeting.config_types.retargeter import (  # noqa: E402
    RetargeterConfig,
    ShoulderDirectionConfig,
)
from holosoma_retargeting.config_types.robot import RobotConfig  # noqa: E402
from holosoma_retargeting.config_types.task import TaskConfig  # noqa: E402
from holosoma_retargeting.retargeting_pipeline import (  # noqa: E402
    build_retargeter_kwargs_from_config,
    create_task_constants,
)
from holosoma_retargeting.shoulder_direction import (  # noqa: E402
    unit_direction_jacobian,
)
from holosoma_retargeting.src.interaction_mesh_retargeter import (  # noqa: E402
    InteractionMeshRetargeter,
)


def _build_retargeter(
    robot: str,
    *,
    data_format: str = "noetix_mocap",
) -> InteractionMeshRetargeter:
    robot_file = {
        "g1": "models/g1/g1_29dof.urdf",
        "e1_23dof": "models/e1/e1_23dof.urdf",
        "e1_24dof": "models/e1/e1_24dof.urdf",
        "e2": "models/e2/e2_23dof.urdf",
    }[robot]
    constants = create_task_constants(
        RobotConfig(
            robot_type=robot,
            robot_urdf_file=str(
                PACKAGE_ROOT / "holosoma_retargeting" / robot_file,
            ),
        ),
        MotionDataConfig(data_format=data_format, robot_type=robot),
        TaskConfig(object_name="ground"),
        "robot_only",
    )
    config = RetargeterConfig(
        q_a_init_idx=0,
        activate_foot_sticking=False,
        shoulder_direction=ShoulderDirectionConfig(enable=True),
    )
    return InteractionMeshRetargeter(
        **build_retargeter_kwargs_from_config(
            config,
            constants,
            None,
            "robot_only",
        ),
    )


def _synthetic_upper_body(retargeter: InteractionMeshRetargeter, frames: int) -> np.ndarray:
    positions = np.zeros((frames, len(retargeter.demo_joints), 3), dtype=np.float64)
    indices = {name: retargeter.demo_joints.index(name) for name in retargeter.demo_joints}
    for frame in range(frames):
        phase = 0.15 * frame
        positions[frame, indices[retargeter.shoulder_human_torso_origin_name]] = [
            0.0,
            0.0,
            0.0,
        ]
        for side_index, spec in enumerate(retargeter.shoulder_side_specs):
            lateral = 0.2 if side_index == 0 else -0.2
            positions[frame, indices[spec.human_arm_name]] = [0.0, lateral, 0.5]
            upper = np.asarray([0.1 * np.sin(phase), 0.0, -0.25])
            positions[frame, indices[spec.human_forearm_name]] = positions[frame, indices[spec.human_arm_name]] + upper
            forearm = np.asarray([0.08 * np.sin(phase), 0.0, -0.22])
            positions[frame, indices[spec.human_hand_name]] = (
                positions[frame, indices[spec.human_forearm_name]] + forearm
            )
    return positions


class ShoulderDirectionMathTests(unittest.TestCase):
    def test_unit_direction_jacobian_matches_finite_difference(self):
        segment = np.asarray([0.4, -0.2, 0.7], dtype=np.float64)
        segment_jacobian = np.asarray(
            [[0.3, -0.1], [0.2, 0.6], [-0.4, 0.2]],
            dtype=np.float64,
        )
        analytic = unit_direction_jacobian(segment, segment_jacobian)
        epsilon = 1e-7
        finite = np.empty_like(analytic)
        for column in range(segment_jacobian.shape[1]):
            plus = segment + epsilon * segment_jacobian[:, column]
            minus = segment - epsilon * segment_jacobian[:, column]
            finite[:, column] = (plus / np.linalg.norm(plus) - minus / np.linalg.norm(minus)) / (2.0 * epsilon)
        np.testing.assert_allclose(analytic, finite, atol=1e-8)

    def test_public_config_and_solver_have_no_branch_planning_interface(self):
        self.assertEqual(
            tuple(field.name for field in fields(ShoulderDirectionConfig)),
            ("enable", "direction_weight", "wrist_axis_weight_scale"),
        )
        self.assertNotIn(
            "shoulder_reference_joint_positions",
            signature(InteractionMeshRetargeter.solve_single_iteration).parameters,
        )
        self.assertFalse(
            hasattr(InteractionMeshRetargeter, "_plan_shoulder_direction_references"),
        )


class ShoulderDirectionRetargeterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g1 = _build_retargeter("g1")
        cls.e1_23 = _build_retargeter("e1_23dof")
        cls.e1 = _build_retargeter("e1_24dof")
        cls.e2 = _build_retargeter("e2")

    def test_supported_chains_and_wrist_capabilities_are_discovered(self):
        for retargeter in (self.g1, self.e1_23, self.e1, self.e2):
            self.assertTrue(retargeter.shoulder_direction_enabled)
            self.assertEqual(retargeter.shoulder_joint_qpos_addresses.shape, (2, 3))
            for spec in retargeter.shoulder_side_specs:
                self.assertTrue(spec.torso_basis_link_name.endswith("shoulder_pitch_link"))
                self.assertTrue(spec.shoulder_anchor_link_name.endswith("shoulder_roll_link"))
        for retargeter in (self.e1_23, self.e1):
            self.assertTrue(
                all(len(spec.wrist_joint_names) == 1 for spec in retargeter.shoulder_side_specs),
            )
        self.assertTrue(
            all(not spec.wrist_joint_names for spec in self.e2.shoulder_side_specs),
        )
        self.assertTrue(
            all(len(spec.wrist_joint_names) == 3 for spec in self.g1.shoulder_side_specs),
        )
        np.testing.assert_array_equal(self.g1.shoulder_wrist_dof_counts, [3, 3])
        np.testing.assert_array_equal(self.e1.shoulder_wrist_dof_counts, [1, 1])
        np.testing.assert_array_equal(self.e2.shoulder_wrist_dof_counts, [0, 0])

    def test_smplx_human_schema_is_supported_for_e1_23dof_and_e2(self):
        for data_format in ("amass", "gvhmr"):
            for robot in ("e1_23dof", "e2"):
                with self.subTest(data_format=data_format, robot=robot):
                    retargeter = _build_retargeter(
                        robot,
                        data_format=data_format,
                    )
                    self.assertEqual(
                        retargeter.shoulder_human_torso_origin_name,
                        "Pelvis",
                    )
                    self.assertEqual(
                        retargeter.shoulder_human_torso_side_names,
                        ("L_Shoulder", "R_Shoulder"),
                    )
                    motion = _synthetic_upper_body(retargeter, 3)
                    upper = retargeter._prepare_shoulder_direction_targets(motion)
                    self.assertEqual(upper.shape, (3, 2, 3))
                    np.testing.assert_allclose(
                        np.linalg.norm(upper, axis=-1),
                        1.0,
                    )

    def test_robot_direction_jacobian_matches_shoulder_finite_difference(self):
        for retargeter in (self.g1, self.e2):
            q = retargeter.robot_model.qpos0.copy()
            directions, jacobians = retargeter._get_shoulder_direction_data(
                q,
                with_jacobians=True,
            )
            self.assertIsNotNone(jacobians)
            epsilon = 1e-7
            for side in range(2):
                for local_column, qpos_address in enumerate(
                    retargeter.shoulder_joint_qpos_addresses[side],
                ):
                    plus = q.copy()
                    minus = q.copy()
                    plus[qpos_address] += epsilon
                    minus[qpos_address] -= epsilon
                    plus_direction, _ = retargeter._get_shoulder_direction_data(
                        plus,
                        with_jacobians=False,
                    )
                    minus_direction, _ = retargeter._get_shoulder_direction_data(
                        minus,
                        with_jacobians=False,
                    )
                    finite = (plus_direction[side] - minus_direction[side]) / (2.0 * epsilon)
                    reduced = retargeter.shoulder_joint_reduced_indices[
                        side,
                        local_column,
                    ]
                    np.testing.assert_allclose(
                        jacobians[side, :, reduced],
                        finite,
                        atol=1e-6,
                    )
            self.assertEqual(directions.shape, (2, 3))

    def test_direction_only_objective_solves_without_a_joint_branch_reference(self):
        retargeter = self.e2
        qpos = retargeter.robot_model.qpos0.copy()
        targets, _ = retargeter._get_shoulder_direction_data(
            qpos,
            with_jacobians=False,
        )
        robot_keys = list(retargeter.laplacian_match_links)
        zero_jacobians = {key: np.zeros((3, retargeter.nq_a), dtype=np.float64) for key in robot_keys}
        zero_positions = {key: np.zeros(3, dtype=np.float64) for key in robot_keys}
        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(
                    retargeter,
                    "_calc_manipulator_jacobians",
                    return_value=(zero_jacobians, zero_positions, None),
                ),
            )
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
            candidate, cost = retargeter.solve_single_iteration(
                q_locked=qpos,
                q_a_n_last=qpos[retargeter.q_a_indices],
                q_t_last=qpos,
                target_laplacian=np.zeros(
                    (len(robot_keys), 3),
                    dtype=np.float64,
                ),
                adj_list=[[] for _ in robot_keys],
                obj_pts_local=np.empty((0, 3), dtype=np.float64),
                foot_sticking={"left": False, "right": False},
                init_t=True,
                frame_idx=0,
                shoulder_direction_targets=targets,
            )

        self.assertTrue(np.isfinite(cost))
        self.assertEqual(candidate.shape, qpos.shape)

    def test_e1_wrist_joint_is_a_scalar_hinge(self):
        for spec in self.e1.shoulder_side_specs:
            joint_id = mujoco.mj_name2id(
                self.e1.robot_model,
                mujoco.mjtObj.mjOBJ_JOINT,
                spec.wrist_joint_names[0],
            )
            self.assertEqual(
                int(self.e1.robot_model.jnt_type[joint_id]),
                int(mujoco.mjtJoint.mjJNT_HINGE),
            )

    def test_g1_wrist_orientation_variables_are_exactly_the_three_wrist_hinges(self):
        for side_index, spec in enumerate(self.g1.shoulder_side_specs):
            expected_reduced_indices = []
            for joint_name in spec.wrist_joint_names:
                joint_id = mujoco.mj_name2id(
                    self.g1.robot_model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    joint_name,
                )
                self.assertEqual(
                    int(self.g1.robot_model.jnt_type[joint_id]),
                    int(mujoco.mjtJoint.mjJNT_HINGE),
                )
                qpos_address = int(self.g1.robot_model.jnt_qposadr[joint_id])
                expected_reduced_indices.append(
                    int(np.flatnonzero(self.g1.q_a_indices == qpos_address)[0]),
                )
            np.testing.assert_array_equal(
                self.g1.shoulder_wrist_reduced_indices[side_index],
                expected_reduced_indices,
            )


if __name__ == "__main__":
    unittest.main()
