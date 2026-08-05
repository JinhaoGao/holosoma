# ruff: noqa: CPY001, PT009, PT027

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
from holosoma_retargeting.src.interaction_mesh_retargeter import (  # noqa: E402
    InteractionMeshRetargeter,
)


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


class RetargetMotionNominalQposTests(unittest.TestCase):
    def setUp(self):
        self.retargeter = InteractionMeshRetargeter.__new__(
            InteractionMeshRetargeter,
        )
        self.retargeter.nq = 9

    def test_retarget_entry_promotes_float32_nominal_to_owned_float64(self):
        nominal = np.arange(18, dtype=np.float32).reshape(2, 9)
        original_nominal = nominal.copy()
        captured = {}

        self.retargeter._validate_source_orientations = lambda *_args: (
            None,
            None,
            (),
        )
        self.retargeter._foot_sticking_states_array = lambda *_args: np.zeros(
            (2, 2),
            dtype=bool,
        )

        def capture_locked_qpos(q_locked_list, _object_poses):
            captured["q_locked_list"] = q_locked_list
            raise RuntimeError("captured owned nominal qpos")

        self.retargeter._apply_dynamic_object_poses = capture_locked_qpos

        with self.assertRaisesRegex(RuntimeError, "captured owned nominal qpos"):
            self.retargeter.retarget_motion(
                human_joint_motions=np.zeros((2, 1, 3), dtype=np.float32),
                object_poses=None,
                object_poses_augmented=None,
                object_points_local_demo=None,
                object_points_local=None,
                foot_sticking_sequences=(),
                q_nominal_list=nominal,
            )

        owned = captured["q_locked_list"]
        self.assertEqual(owned.dtype, np.dtype(np.float64))
        self.assertTrue(owned.flags.owndata)
        self.assertTrue(owned.flags.c_contiguous)
        self.assertFalse(np.shares_memory(owned, nominal))
        np.testing.assert_array_equal(owned, original_nominal)

        nominal[0, 0] = np.float32(-123.0)
        self.assertEqual(owned[0, 0], original_nominal[0, 0])
        precision_sentinel = np.nextafter(
            np.float64(1.0),
            np.float64(2.0),
        )
        owned[0, 0] = precision_sentinel
        self.assertEqual(owned[0, 0], precision_sentinel)
        self.assertNotEqual(
            owned[0, 0],
            np.float64(np.float32(precision_sentinel)),
        )

    def test_nominal_qpos_validation_rejects_invalid_contracts(self):
        nonfinite = np.zeros((2, 9), dtype=np.float64)
        nonfinite[1, 8] = np.nan
        invalid_cases = (
            (
                np.zeros(9, dtype=np.float64),
                "two-dimensional",
            ),
            (
                np.zeros((1, 9), dtype=np.float64),
                "exactly 2 frames",
            ),
            (
                np.zeros((2, 8), dtype=np.float64),
                "complete model qpos vectors with width 9",
            ),
            (
                nonfinite,
                "only finite values",
            ),
        )

        for nominal, message in invalid_cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                ValueError,
                message,
            ):
                self.retargeter._owned_nominal_qpos(
                    nominal,
                    num_frames=2,
                )


class OmomoRetargetingObjectSetupTests(unittest.TestCase):
    ROBOTS = {"g1": 29}

    def setUp(self):
        self._asset_temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._asset_temp.cleanup)
        self.generated_assets_dir = Path(self._asset_temp.name)

    def test_object_constants_require_resolved_category(self):
        with self.assertRaisesRegex(ValueError, "resolved object name"):
            create_task_constants(
                RobotConfig(robot_type="g1"),
                MotionDataConfig(data_format="omomo", robot_type="g1"),
                TaskConfig(),
                "object_interaction",
            )

    def test_all_objects_setup_for_supported_robots(self):
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
                        generated_assets_dir=(self.generated_assets_dir / robot_name / object_name),
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
            generated_assets_dir=self.generated_assets_dir / "scaled",
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


class RetargeterRuntimeTests(unittest.TestCase):
    def test_sticking_states_are_packed_in_canonical_side_order(self):
        states = InteractionMeshRetargeter._foot_sticking_states_array(
            [
                {"R_Toe": False, "L_Toe": True},
                {"RightToeBase": True, "LeftToeBase": False},
            ],
            num_frames=2,
        )

        np.testing.assert_array_equal(
            states,
            np.asarray(
                [
                    [True, False],
                    [False, True],
                ],
                dtype=bool,
            ),
        )

    def test_sticking_state_packing_rejects_a_missing_side(self):
        with self.assertRaisesRegex(ValueError, "missing side"):
            InteractionMeshRetargeter._foot_sticking_states_array(
                [{"L_Toe": True}],
                num_frames=1,
            )

    def test_config_passes_live_visualization_options_to_retargeter(self):
        constants = create_task_constants(
            RobotConfig(robot_type="g1"),
            MotionDataConfig(data_format="omomo", robot_type="g1"),
            TaskConfig(object_name="largebox"),
            "object_interaction",
        )
        kwargs = build_retargeter_kwargs_from_config(
            RetargeterConfig(
                visualize=True,
                debug=True,
            ),
            constants,
            "unused-object.urdf",
            "object_interaction",
        )

        self.assertTrue(kwargs["visualize"])
        self.assertTrue(kwargs["debug"])

    def test_live_frame_updates_robot_without_debug_overlays(self):
        retargeter = InteractionMeshRetargeter.__new__(
            InteractionMeshRetargeter,
        )
        retargeter.visualize = True
        retargeter.debug = False
        retargeter.show_interaction_mesh = False
        q = np.arange(7, dtype=np.float64)
        foot_sticking_state = np.asarray([True, False])

        with patch.object(
            retargeter,
            "_update_foot_sticking_status",
        ) as update_status, patch.object(retargeter, "draw_q") as draw_q:
            retargeter._update_live_visualization_frame(
                q,
                frame_idx=3,
                foot_sticking_state=foot_sticking_state,
            )

        update_status.assert_called_once_with(
            3,
            foot_sticking_state,
        )
        draw_q.assert_called_once_with(q)

    def test_config_always_passes_custom_nominal_tracking_tau(self):
        constants = create_task_constants(
            RobotConfig(robot_type="g1"),
            MotionDataConfig(data_format="omomo", robot_type="g1"),
            TaskConfig(object_name="largebox"),
            "object_interaction",
        )
        config = RetargeterConfig(nominal_tracking_tau=37.25)

        for task_type in (
            "robot_only",
            "object_interaction",
            "climbing",
        ):
            with self.subTest(task_type=task_type):
                kwargs = build_retargeter_kwargs_from_config(
                    config,
                    constants,
                    None,
                    task_type,
                )
                self.assertEqual(kwargs["nominal_tracking_tau"], 37.25)


class MainStyleSqpIterationTests(unittest.TestCase):
    @staticmethod
    def _retargeter_with_candidates(
        candidates: list[tuple[np.ndarray, float]],
        *,
        max_iterations: int = 20,
    ) -> tuple[InteractionMeshRetargeter, list[dict], int]:
        retargeter = InteractionMeshRetargeter.__new__(InteractionMeshRetargeter)
        retargeter.q_a_indices = np.asarray([0], dtype=int)
        retargeter.last_sqp_iteration_count = 0
        retargeter.last_sqp_stop_reason = "not_started"

        calls: list[dict] = []
        candidate_iterator = iter(candidates)

        def fake_single_iteration(**kwargs):
            calls.append(kwargs)
            return next(candidate_iterator)

        retargeter.solve_single_iteration = fake_single_iteration
        return retargeter, calls, max_iterations

    @staticmethod
    def _candidate(q: float, *, cost: float) -> tuple[np.ndarray, float]:
        return np.asarray([q]), cost

    @staticmethod
    def _iterate(
        retargeter: InteractionMeshRetargeter,
        *,
        max_iterations: int,
    ):
        return retargeter.iterate(
            q_locked=np.zeros(1),
            q_n=np.zeros(1),
            q_t_last=np.zeros(1),
            target_laplacian=np.zeros((1, 3)),
            adj_list=[],
            obj_pts_local=np.zeros((0, 3)),
            foot_sticking={
                "left": False,
                "right": False,
            },
            n_iter=max_iterations,
        )

    def test_cost_stability_stops_and_keeps_latest_qp_step(self):
        retargeter, calls, max_iterations = self._retargeter_with_candidates(
            [
                self._candidate(1.0, cost=10.0),
                self._candidate(2.0, cost=9.0),
                self._candidate(3.0, cost=9.0),
                self._candidate(4.0, cost=8.0),
            ],
        )

        q, cost = self._iterate(
            retargeter,
            max_iterations=max_iterations,
        )

        self.assertEqual(len(calls), 3)
        self.assertEqual(retargeter.last_sqp_iteration_count, 3)
        self.assertEqual(retargeter.last_sqp_stop_reason, "cost_stable")
        self.assertEqual(cost, 9.0)
        np.testing.assert_array_equal(q, np.asarray([3.0]))

    def test_iteration_cap_returns_latest_qp_step(self):
        retargeter, calls, max_iterations = self._retargeter_with_candidates(
            [
                self._candidate(1.0, cost=1.0),
                self._candidate(2.0, cost=100.0),
                self._candidate(3.0, cost=10_000.0),
            ],
            max_iterations=3,
        )

        q, cost = self._iterate(
            retargeter,
            max_iterations=max_iterations,
        )

        self.assertEqual(len(calls), 3)
        self.assertEqual(retargeter.last_sqp_stop_reason, "max_iterations")
        self.assertEqual(cost, 10_000.0)
        np.testing.assert_array_equal(q, np.asarray([3.0]))


if __name__ == "__main__":
    unittest.main()
