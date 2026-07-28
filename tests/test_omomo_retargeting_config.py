# ruff: noqa: PT009, PT027

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cvxpy as cp
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
                        mujoco.mj_name2id(
                            model,
                            mujoco.mjtObj.mjOBJ_GEOM,
                            f"{object_name}_visual",
                        ),
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


class ObjectNonPenetrationToggleTests(unittest.TestCase):
    def test_toggle_controls_object_pairs_but_not_ground_pairs(self):
        retargeter = InteractionMeshRetargeter.__new__(InteractionMeshRetargeter)
        retargeter.object_name = "clothesstand"

        retargeter.activate_obj_non_penetration = True
        self.assertTrue(
            retargeter._environment_collision_pair_is_active(
                "left_hand_collision",
                "clothesstand_collision_000",
            )
        )

        retargeter.activate_obj_non_penetration = False
        self.assertFalse(
            retargeter._environment_collision_pair_is_active(
                "left_hand_collision",
                "clothesstand_collision_000",
            )
        )
        self.assertTrue(
            retargeter._environment_collision_pair_is_active(
                "left_foot_collision",
                "ground",
            )
        )
        self.assertFalse(
            retargeter._environment_collision_pair_is_active(
                "clothesstand_collision_000",
                "ground",
            )
        )


class ObjectPointPayloadTests(unittest.TestCase):
    def test_robot_only_payload_omits_static_ground_point_trajectories(self):
        payload = InteractionMeshRetargeter._object_points_save_payload(
            has_dynamic_object=False,
            object_points_local_demo=np.zeros((225, 3)),
            object_points_local=np.zeros((225, 3)),
            obj_pts_demo_list=[np.zeros((225, 3))],
            obj_pts_list=[np.zeros((225, 3))],
        )

        self.assertEqual(payload, {})

    def test_dynamic_object_payload_keeps_local_and_world_points(self):
        payload = InteractionMeshRetargeter._object_points_save_payload(
            has_dynamic_object=True,
            object_points_local_demo=np.zeros((4, 3)),
            object_points_local=np.ones((4, 3)),
            obj_pts_demo_list=[np.full((4, 3), 2.0)],
            obj_pts_list=[np.full((4, 3), 3.0)],
        )

        self.assertEqual(
            set(payload),
            {
                "object_points_demo_local",
                "object_points_target_local",
                "object_points_demo_world",
                "object_points_target_world",
            },
        )
        self.assertEqual(payload["object_points_demo_world"].shape, (1, 4, 3))


class FootStickingFallbackTests(unittest.TestCase):
    def test_config_passes_fallback_tolerance_to_retargeter(self):
        constants = create_task_constants(
            RobotConfig(robot_type="g1"),
            MotionDataConfig(data_format="omomo", robot_type="g1"),
            TaskConfig(object_name="largebox"),
            "object_interaction",
        )
        kwargs = build_retargeter_kwargs_from_config(
            RetargeterConfig(foot_sticking_fallback_tolerance=0.025),
            constants,
            "unused-object.urdf",
            "object_interaction",
        )

        self.assertEqual(kwargs["foot_sticking_fallback_tolerance"], 0.025)
        self.assertTrue(kwargs["release_foot_sticking_on_infeasible"])
        self.assertFalse(kwargs["release_object_non_penetration_on_infeasible"])
        self.assertTrue(kwargs["retry_without_foot_sticking_on_infeasible"])

    def test_solver_replaces_only_infeasible_foot_constraints(self):
        x = cp.Variable()
        objective = cp.Minimize(cp.square(x))
        base_constraints = [x <= 0.5]
        normal_foot_constraints = [x >= 1.0]
        fallback_foot_constraints = [x >= -0.1]

        (
            problem,
            resolution,
            object_constraints_released,
        ) = InteractionMeshRetargeter._solve_with_foot_sticking_fallback(
            objective=objective,
            base_constraints=base_constraints,
            object_non_penetration_constraints=[],
            foot_sticking_constraints=normal_foot_constraints,
            foot_sticking_fallback_constraints=fallback_foot_constraints,
            solver_kwargs={"verbose": False},
            remove_soc_on_failure=False,
            release_on_failure=True,
            release_object_non_penetration_on_failure=True,
        )

        self.assertEqual(resolution, "relaxed")
        self.assertFalse(object_constraints_released)
        self.assertIn(problem.status, (cp.OPTIMAL, cp.OPTIMAL_INACCURATE))
        self.assertLessEqual(float(x.value), 0.5 + 1e-8)
        self.assertGreaterEqual(float(x.value), -0.1 - 1e-8)

    def test_solver_can_release_only_foot_constraints_as_last_resort(self):
        x = cp.Variable()
        objective = cp.Minimize(cp.square(x))
        base_constraints = [x <= 0.5]

        (
            problem,
            resolution,
            object_constraints_released,
        ) = InteractionMeshRetargeter._solve_with_foot_sticking_fallback(
            objective=objective,
            base_constraints=base_constraints,
            object_non_penetration_constraints=[],
            foot_sticking_constraints=[x >= 1.0],
            foot_sticking_fallback_constraints=[x >= 0.8],
            solver_kwargs={"verbose": False},
            remove_soc_on_failure=False,
            release_on_failure=True,
            release_object_non_penetration_on_failure=True,
        )

        self.assertEqual(resolution, "released")
        self.assertFalse(object_constraints_released)
        self.assertIn(problem.status, (cp.OPTIMAL, cp.OPTIMAL_INACCURATE))
        self.assertLessEqual(float(x.value), 0.5 + 1e-8)

    def test_solver_can_release_only_object_constraints_as_last_resort(self):
        x = cp.Variable()
        objective = cp.Minimize(cp.square(x))
        base_constraints = [x <= 0.5]

        (
            problem,
            resolution,
            object_constraints_released,
        ) = InteractionMeshRetargeter._solve_with_foot_sticking_fallback(
            objective=objective,
            base_constraints=base_constraints,
            object_non_penetration_constraints=[x >= 1.0],
            foot_sticking_constraints=[],
            foot_sticking_fallback_constraints=[],
            solver_kwargs={"verbose": False},
            remove_soc_on_failure=False,
            release_on_failure=True,
            release_object_non_penetration_on_failure=True,
        )

        self.assertIsNone(resolution)
        self.assertTrue(object_constraints_released)
        self.assertIn(problem.status, (cp.OPTIMAL, cp.OPTIMAL_INACCURATE))
        self.assertLessEqual(float(x.value), 0.5 + 1e-8)


class SqpConvergenceTests(unittest.TestCase):
    @staticmethod
    def _retargeter_with_costs(
        costs: list[float],
        *,
        max_iterations: int = 20,
        patience: int = 3,
    ) -> tuple[InteractionMeshRetargeter, list[int]]:
        retargeter = InteractionMeshRetargeter.__new__(InteractionMeshRetargeter)
        retargeter.q_a_indices = np.asarray([0], dtype=int)
        retargeter.sqp_max_iterations = max_iterations
        retargeter.sqp_min_iterations = 1
        retargeter.sqp_convergence_patience = patience
        retargeter.sqp_abs_cost_tolerance = 1e-9
        retargeter.sqp_rel_cost_tolerance = 1e-7
        retargeter.last_sqp_iteration_count = 0
        retargeter.last_sqp_stop_reason = "not_started"

        calls: list[int] = []
        cost_iterator = iter(costs)

        def fake_single_iteration(**_kwargs):
            calls.append(len(calls) + 1)
            return np.asarray([float(calls[-1])]), next(cost_iterator)

        retargeter.solve_single_iteration = fake_single_iteration
        return retargeter, calls

    @staticmethod
    def _iterate(retargeter: InteractionMeshRetargeter):
        return retargeter.iterate(
            q_locked=np.zeros(1),
            q_n=np.zeros(1),
            q_t_last=np.zeros(1),
            target_laplacian=np.zeros((1, 3)),
            adj_list=[],
            obj_pts_local=np.zeros((0, 3)),
            foot_sticking=(False, False),
        )

    def test_stops_only_after_patience_is_exhausted(self):
        retargeter, calls = self._retargeter_with_costs(
            [10.0, 9.0, 8.0, 8.1, 8.2, 8.3, 7.0],
            patience=3,
        )

        q, cost = self._iterate(retargeter)

        self.assertEqual(len(calls), 6)
        self.assertEqual(retargeter.last_sqp_iteration_count, 6)
        self.assertEqual(retargeter.last_sqp_stop_reason, "cost_stalled")
        self.assertEqual(cost, 8.0)
        np.testing.assert_array_equal(q, np.asarray([3.0]))

    def test_significant_decrease_resets_patience(self):
        retargeter, calls = self._retargeter_with_costs(
            [10.0, 9.0, 9.0, 8.0, 8.0, 8.0, 8.0],
            patience=3,
        )

        q, cost = self._iterate(retargeter)

        self.assertEqual(len(calls), 7)
        self.assertEqual(retargeter.last_sqp_stop_reason, "cost_stalled")
        self.assertEqual(cost, 8.0)
        np.testing.assert_array_equal(q, np.asarray([4.0]))

    def test_safety_cap_returns_best_iteration_instead_of_last(self):
        retargeter, calls = self._retargeter_with_costs(
            [10.0, 9.0, 10.0, 11.0],
            max_iterations=4,
            patience=10,
        )

        q, cost = self._iterate(retargeter)

        self.assertEqual(len(calls), 4)
        self.assertEqual(retargeter.last_sqp_stop_reason, "max_iterations")
        self.assertEqual(cost, 9.0)
        np.testing.assert_array_equal(q, np.asarray([2.0]))


if __name__ == "__main__":
    unittest.main()
