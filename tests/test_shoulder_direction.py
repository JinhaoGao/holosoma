# ruff: noqa: CPY001, PT009

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
    second_order_candidate_path,
    unit_direction_jacobian,
)
from holosoma_retargeting.src.interaction_mesh_retargeter import (  # noqa: E402
    InteractionMeshRetargeter,
)


def _build_retargeter(robot: str, *, candidate_count: int = 4) -> InteractionMeshRetargeter:
    robot_file = {
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
        MotionDataConfig(data_format="noetix_mocap", robot_type=robot),
        TaskConfig(object_name="ground"),
        "robot_only",
    )
    config = RetargeterConfig(
        q_a_init_idx=0,
        activate_foot_sticking=False,
        shoulder_direction=ShoulderDirectionConfig(
            enable=True,
            candidate_count=candidate_count,
            seed_count=max(8, candidate_count),
            refresh_candidate_count=min(2, candidate_count - 1),
            max_nfev=20,
        ),
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
        positions[frame, indices["Hips"]] = [0.0, 0.0, 0.0]
        positions[frame, indices["LeftArm"]] = [0.0, 0.2, 0.5]
        positions[frame, indices["RightArm"]] = [0.0, -0.2, 0.5]
        upper = np.asarray([0.1 * np.sin(phase), 0.0, -0.25])
        positions[frame, indices["LeftForeArm"]] = positions[frame, indices["LeftArm"]] + upper
        positions[frame, indices["RightForeArm"]] = positions[frame, indices["RightArm"]] + upper
        forearm = np.asarray([0.08 * np.sin(phase), 0.0, -0.22])
        positions[frame, indices["LeftHand"]] = positions[frame, indices["LeftForeArm"]] + forearm
        positions[frame, indices["RightHand"]] = positions[frame, indices["RightForeArm"]] + forearm
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

    def test_second_order_path_rejects_a_low_node_cost_branch_flip(self):
        states = np.asarray(
            [
                [[0.0], [1.0]],
                [[0.1], [0.9]],
                [[0.2], [0.8]],
                [[0.3], [0.7]],
            ],
            dtype=np.float64,
        )
        node_costs = np.asarray(
            [[0.0, 0.2], [0.0, 0.2], [0.3, 0.0], [0.3, 0.0]],
            dtype=np.float64,
        )
        selected = second_order_candidate_path(
            states,
            node_costs,
            np.ones_like(node_costs, dtype=bool),
            joint_ranges=np.ones(1),
            velocity_weight=2.0,
            acceleration_weight=8.0,
        )
        self.assertTrue(
            np.array_equal(selected, np.zeros(4, dtype=np.int32))
            or np.array_equal(selected, np.ones(4, dtype=np.int32)),
        )

    def test_second_order_path_enforces_joint_step_edges(self):
        states = np.asarray(
            [
                [[0.0], [1.0]],
                [[0.1], [0.9]],
                [[0.2], [0.8]],
            ],
            dtype=np.float64,
        )
        node_costs = np.asarray(
            [[0.0, 6.0], [0.0, 6.0], [10.0, 0.0]],
            dtype=np.float64,
        )
        selected = second_order_candidate_path(
            states,
            node_costs,
            np.ones_like(node_costs, dtype=bool),
            joint_ranges=np.ones(1),
            velocity_weight=0.0,
            acceleration_weight=0.0,
            max_joint_step=np.asarray([0.25]),
        )
        np.testing.assert_array_equal(selected, np.zeros(3, dtype=np.int32))

    def test_candidate_reduction_preserves_continuation_samples(self):
        states = np.arange(18, dtype=np.float64).reshape(6, 3)
        selected = InteractionMeshRetargeter._diverse_candidate_indices(
            states,
            np.arange(6, dtype=np.float64),
            np.ones(3, dtype=np.float64),
            4,
            required_indices=np.asarray([2, 4], dtype=np.int32),
        )
        self.assertTrue({2, 4}.issubset(set(selected)))


class ShoulderDirectionRetargeterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.e1_23 = _build_retargeter("e1_23dof")
        cls.e1 = _build_retargeter("e1_24dof")
        cls.e2 = _build_retargeter("e2")

    def test_supported_chains_and_e1_wrist_capability_are_discovered(self):
        for retargeter in (self.e1_23, self.e1, self.e2):
            self.assertTrue(retargeter.shoulder_direction_enabled)
            self.assertEqual(retargeter.shoulder_joint_qpos_addresses.shape, (2, 3))
            for spec in retargeter.shoulder_side_specs:
                self.assertTrue(spec.torso_basis_link_name.endswith("shoulder_pitch_link"))
                self.assertTrue(spec.shoulder_anchor_link_name.endswith("shoulder_roll_link"))
        for retargeter in (self.e1_23, self.e1):
            self.assertTrue(
                all(spec.wrist_joint_name is not None for spec in retargeter.shoulder_side_specs),
            )
        self.assertTrue(
            all(spec.wrist_joint_name is None for spec in self.e2.shoulder_side_specs),
        )

    def test_robot_direction_jacobian_matches_shoulder_finite_difference(self):
        retargeter = self.e2
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
                reduced = retargeter.shoulder_joint_reduced_indices[side, local_column]
                np.testing.assert_allclose(
                    jacobians[side, :, reduced],
                    finite,
                    atol=1e-6,
                )
        self.assertEqual(directions.shape, (2, 3))

    def test_short_sequence_planner_returns_valid_continuous_references(self):
        retargeter = self.e2
        motion = _synthetic_upper_body(retargeter, 4)
        upper, forearm = retargeter._prepare_shoulder_direction_targets(motion)
        plan = retargeter._plan_shoulder_direction_references(
            retargeter.robot_model.qpos0.copy(),
            upper,
            forearm,
        )
        self.assertEqual(plan["reference_joint_positions"].shape, (4, 2, 3))
        self.assertEqual(plan["raw_reference_joint_positions"].shape, (4, 2, 3))
        self.assertEqual(plan["selected_candidate_indices"].shape, (4, 2))
        self.assertTrue(np.isfinite(plan["reference_joint_positions"]).all())
        self.assertTrue(
            np.all(np.sum(plan["candidate_valid"], axis=-1) >= 1),
        )
        self.assertLess(
            float(np.nanmax(plan["candidate_direction_errors_rad"])),
            np.pi,
        )

    def test_reference_smoothing_spreads_a_discrete_branch_change(self):
        retargeter = self.e2
        raw = np.zeros((12, 3), dtype=np.float64)
        raw[6:, 2] = 2.4
        smoothed = retargeter._smooth_shoulder_reference(raw, side_index=0)
        steps = np.abs(np.diff(smoothed, axis=0))
        self.assertLessEqual(
            float(np.max(steps)),
            retargeter.shoulder_direction_config.max_frame_step_rad + 1e-7,
        )
        self.assertGreater(float(smoothed[5, 2]), 0.0)
        self.assertLess(float(smoothed[6, 2]), 2.4)

    def test_continuation_projection_respects_joint_step_limit(self):
        retargeter = self.e2
        q = retargeter.robot_model.qpos0.copy()
        seed = q[retargeter.shoulder_joint_qpos_addresses[0]].copy()
        projected, _ = retargeter._project_shoulder_candidate(
            q,
            side_index=0,
            target_direction=np.asarray([1.0, 0.0, 0.0]),
            seed=seed,
            max_step_rad=0.1,
        )
        self.assertTrue(np.all(np.abs(projected - seed) <= 0.099 + 1e-9))

    def test_e1_wrist_joint_is_a_scalar_hinge(self):
        for spec in self.e1.shoulder_side_specs:
            joint_id = mujoco.mj_name2id(
                self.e1.robot_model,
                mujoco.mjtObj.mjOBJ_JOINT,
                spec.wrist_joint_name,
            )
            self.assertEqual(
                int(self.e1.robot_model.jnt_type[joint_id]),
                int(mujoco.mjtJoint.mjJNT_HINGE),
            )


if __name__ == "__main__":
    unittest.main()
