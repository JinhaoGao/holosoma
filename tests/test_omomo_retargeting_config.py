# ruff: noqa: CPY001, PT009, PT027

from __future__ import annotations

import hashlib
import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
from holosoma_retargeting.result_artifact import validate_result_artifact  # noqa: E402
from holosoma_retargeting.src.interaction_mesh_retargeter import (  # noqa: E402
    ConstraintMode,
    InteractionMeshRetargeter,
    NonlinearConstraintResiduals,
    SQPIterationResult,
    SQPNonlinearFeasibilityError,
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
    def test_linearized_non_penetration_uses_bounded_interior_margin(self):
        rhs = InteractionMeshRetargeter._non_penetration_linearization_rhs(
            signed_distance=-0.002,
            penetration_tolerance=0.001,
        )
        self.assertAlmostEqual(rhs, 0.0015)

        zero_tolerance_rhs = InteractionMeshRetargeter._non_penetration_linearization_rhs(
            signed_distance=0.0,
            penetration_tolerance=0.0,
        )
        self.assertAlmostEqual(
            zero_tolerance_rhs,
            1e-5,
        )

        capped_rhs = InteractionMeshRetargeter._non_penetration_linearization_rhs(
            signed_distance=0.0,
            penetration_tolerance=0.1,
        )
        self.assertAlmostEqual(capped_rhs, -0.0995)

    def test_linearized_non_penetration_rejects_invalid_inputs(self):
        for signed_distance, tolerance in (
            (np.nan, 0.001),
            (0.0, np.inf),
            (0.0, -0.001),
        ):
            with self.subTest(
                signed_distance=signed_distance,
                tolerance=tolerance,
            ), self.assertRaisesRegex(ValueError, "finite|non-negative"):
                InteractionMeshRetargeter._non_penetration_linearization_rhs(
                    signed_distance,
                    tolerance,
                )

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
        self.assertFalse(kwargs["retry_frame_zero_ground_on_infeasible"])

        ground_constants = create_task_constants(
            RobotConfig(robot_type="g1"),
            MotionDataConfig(data_format="omomo", robot_type="g1"),
            TaskConfig(object_name="ground"),
            "robot_only",
        )
        ground_kwargs = build_retargeter_kwargs_from_config(
            RetargeterConfig(),
            ground_constants,
            None,
            "robot_only",
        )
        self.assertTrue(
            ground_kwargs["retry_frame_zero_ground_on_infeasible"],
        )

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

    def test_frame_zero_retry_requires_structured_unreleased_pure_ground_failure(
        self,
    ):
        pure_ground = NonlinearConstraintResiduals(
            ground_non_penetration=0.01,
            object_non_penetration=0.0,
            foot_sticking=0.0,
            foot_lock=0.0,
            self_collision=0.0,
            joint_limits=0.0,
        )
        exception = SQPNonlinearFeasibilityError(
            "pure ground",
            foot_related_failure=False,
            frame_idx=0,
            total_iterations=50,
            closest_residuals=pure_ground,
            closest_constraint_mode=ConstraintMode("inactive"),
            minimum_hard_constraint_violation=0.01,
        )
        self.assertTrue(
            InteractionMeshRetargeter._is_pure_frame_zero_ground_failure(
                exception,
            )
        )

        released_mode = SQPNonlinearFeasibilityError(
            "released",
            foot_related_failure=False,
            frame_idx=0,
            total_iterations=50,
            closest_residuals=pure_ground,
            closest_constraint_mode=ConstraintMode(
                "inactive",
                object_non_penetration_released=True,
            ),
        )
        self.assertFalse(
            InteractionMeshRetargeter._is_pure_frame_zero_ground_failure(
                released_mode,
            )
        )

        mixed_residuals = NonlinearConstraintResiduals(
            ground_non_penetration=0.01,
            object_non_penetration=0.0,
            foot_sticking=0.002,
            foot_lock=0.0,
            self_collision=0.0,
            joint_limits=0.0,
        )
        mixed = SQPNonlinearFeasibilityError(
            "mixed",
            foot_related_failure=True,
            frame_idx=0,
            total_iterations=50,
            closest_residuals=mixed_residuals,
            closest_constraint_mode=ConstraintMode("inactive"),
        )
        self.assertFalse(
            InteractionMeshRetargeter._is_pure_frame_zero_ground_failure(
                mixed,
            )
        )
        self.assertFalse(
            InteractionMeshRetargeter._is_pure_frame_zero_ground_failure(
                RuntimeError("unstructured"),
            )
        )

    def test_structured_feasibility_error_round_trips_through_worker_pickle(
        self,
    ):
        residuals = NonlinearConstraintResiduals(
            ground_non_penetration=0.01,
            object_non_penetration=0.002,
            foot_sticking=0.003,
            foot_lock=0.004,
            self_collision=0.005,
            joint_limits=0.006,
        )
        mode = ConstraintMode(
            "relaxed",
            object_non_penetration_released=True,
            trust_region_released=True,
        )
        original = SQPNonlinearFeasibilityError(
            "structured worker failure",
            foot_related_failure=True,
            frame_idx=17,
            total_iterations=123,
            closest_residuals=residuals,
            closest_constraint_mode=mode,
            minimum_hard_constraint_violation=0.006,
        )

        restored = pickle.loads(pickle.dumps(original))

        self.assertIsInstance(restored, SQPNonlinearFeasibilityError)
        self.assertEqual(str(restored), str(original))
        self.assertTrue(restored.foot_related_failure)
        self.assertEqual(restored.frame_idx, 17)
        self.assertEqual(restored.total_iterations, 123)
        self.assertEqual(restored.closest_residuals, residuals)
        self.assertEqual(restored.closest_constraint_mode, mode)
        self.assertEqual(
            restored.minimum_hard_constraint_violation,
            0.006,
        )

    def test_frame_zero_retry_stop_reason_survives_full_sequence_retry(self):
        stop_reason = "feasible_step_stable"
        prefixed = InteractionMeshRetargeter._frame_zero_ground_retry_stop_reason(
            stop_reason,
            triggered=True,
        )
        self.assertEqual(
            prefixed,
            "frame_zero_ground_retry:feasible_step_stable",
        )
        self.assertEqual(
            InteractionMeshRetargeter._frame_zero_ground_retry_stop_reason(
                prefixed,
                triggered=True,
            ),
            prefixed,
        )
        self.assertEqual(
            InteractionMeshRetargeter._frame_zero_ground_retry_stop_reason(
                stop_reason,
                triggered=False,
            ),
            stop_reason,
        )

    def test_frame_zero_retry_provenance_survives_saved_full_sequence_retry(
        self,
    ):
        robot_urdf = PACKAGE_ROOT / "holosoma_retargeting" / "models" / "e1" / "e1_23dof.urdf"
        constants = create_task_constants(
            RobotConfig(
                robot_type="e1",
                robot_urdf_file=str(robot_urdf),
            ),
            MotionDataConfig(
                data_format="noetix_mocap",
                robot_type="e1",
            ),
            TaskConfig(object_name="ground"),
            "robot_only",
        )
        config = RetargeterConfig()
        retargeter = InteractionMeshRetargeter(
            **build_retargeter_kwargs_from_config(
                config,
                constants,
                None,
                "robot_only",
            )
        )
        initial_q = retargeter.robot_model.qpos0.copy()
        initial_root_z = float(initial_q[2])
        zero_residuals = NonlinearConstraintResiduals(
            ground_non_penetration=0.0,
            object_non_penetration=0.0,
            foot_sticking=0.0,
            foot_lock=0.0,
            self_collision=0.0,
            joint_limits=0.0,
        )
        pure_ground_residuals = NonlinearConstraintResiduals(
            ground_non_penetration=0.001,
            object_non_penetration=0.0,
            foot_sticking=0.0,
            foot_lock=0.0,
            self_collision=0.0,
            joint_limits=0.0,
        )
        iterate_calls: list[tuple[int, bool]] = []

        def fake_iterate(**kwargs):
            frame_idx = int(kwargs["frame_idx"])
            iterate_calls.append(
                (
                    frame_idx,
                    retargeter.activate_foot_sticking,
                )
            )
            call_index = len(iterate_calls)
            if call_index == 1:
                raise SQPNonlinearFeasibilityError(
                    "mock frame-zero pure-ground failure",
                    foot_related_failure=False,
                    frame_idx=0,
                    total_iterations=7,
                    closest_residuals=pure_ground_residuals,
                    closest_constraint_mode=ConstraintMode("inactive"),
                    minimum_hard_constraint_violation=0.001,
                )
            if call_index == 3:
                raise SQPNonlinearFeasibilityError(
                    "mock later-frame foot failure",
                    foot_related_failure=True,
                    frame_idx=1,
                    total_iterations=5,
                    closest_residuals=NonlinearConstraintResiduals(
                        ground_non_penetration=0.0,
                        object_non_penetration=0.0,
                        foot_sticking=0.01,
                        foot_lock=0.0,
                        self_collision=0.0,
                        joint_limits=0.0,
                    ),
                    closest_constraint_mode=ConstraintMode("normal"),
                    minimum_hard_constraint_violation=0.01,
                )
            retargeter.last_sqp_iteration_count = 2
            retargeter.last_sqp_stop_reason = "feasible_step_stable"
            retargeter.last_nonlinear_constraint_residuals = zero_residuals
            retargeter.last_constraint_mode = ConstraintMode("inactive")
            return np.array(
                kwargs["q_n"],
                dtype=np.float64,
                copy=True,
            ), float(frame_idx + 1)

        def fake_true_geometry_residuals(
            q,
            **_kwargs,
        ):
            if np.isclose(float(q[2]), initial_root_z):
                return pure_ground_residuals
            return zero_residuals

        retargeter.iterate = fake_iterate
        retargeter._evaluate_nonlinear_constraint_residuals = fake_true_geometry_residuals
        retargeter._minimum_active_horizontal_ground_distance = lambda q: -0.002 + float(q[2]) - initial_root_z

        frames = 2
        joint_axis = np.arange(
            len(retargeter.demo_joints),
            dtype=np.float32,
        )
        human_joints = np.stack(
            (
                np.sin(0.7 * joint_axis),
                np.cos(0.4 * joint_axis),
                0.05 * joint_axis + 0.1 * np.sin(joint_axis),
            ),
            axis=-1,
        )
        human_joints = np.stack(
            (
                human_joints,
                human_joints + np.asarray([0.01, 0.0, 0.0], dtype=np.float32),
            ),
        ).astype(np.float32)
        object_poses = np.zeros((frames, 7), dtype=np.float32)
        object_poses[:, 3] = 1.0
        object_points = np.asarray(
            [
                [-0.5, -0.5, 0.0],
                [0.5, -0.5, 0.0],
                [0.0, 0.5, 0.0],
            ],
            dtype=np.float32,
        )
        foot_sticking = (
            {"LeftFoot": False, "RightFoot": False},
            {"LeftFoot": True, "RightFoot": False},
        )
        config_json = json.dumps(
            {
                "config": {
                    "data_format": "noetix_mocap",
                    "robot_config": {"robot_type": "e1"},
                    "task_config": {"object_name": "ground"},
                    "task_type": "robot_only",
                    "retargeter": {
                        "activate_obj_non_penetration": (config.activate_obj_non_penetration),
                        "penetration_tolerance": config.penetration_tolerance,
                        "q_a_init_idx": config.q_a_init_idx,
                        "retry_frame_zero_ground_on_infeasible": (config.retry_frame_zero_ground_on_infeasible),
                        "sqp_max_iterations": config.sqp_max_iterations,
                    },
                },
                "dataset_partition": "noetix_mocap",
                "experiment_name": None,
                "run_kind": "single",
                "sequence_key": "mock_recursive_retry",
                "variant": {
                    "name": "identity",
                    "object_scale": [1.0, 1.0, 1.0],
                    "rotation": 0.0,
                    "translation": [0.0, 0.0, 0.0],
                },
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        result_metadata = {
            "run_kind": "single",
            "variant": "identity",
            "dataset_partition": "noetix_mocap",
            "sequence_key": "mock_recursive_retry",
            "experiment_name": "",
            "source_path": "demo_data/noetix_mocap/mock_recursive_retry.npz",
            "source_sha256": "a" * 64,
            "config_json": config_json,
            "config_sha256": hashlib.sha256(
                config_json.encode("utf-8"),
            ).hexdigest(),
            "orientation_source": "absent",
            "source_human_height": np.float64(1.7),
            "human_position_scale": np.float64(1.0),
            "human_position_preprocessing": "mock_direct_positions",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = Path(temp_dir) / "identity.npz"
            retargeter.retarget_motion(
                human_joint_motions=human_joints,
                object_poses=object_poses,
                object_poses_augmented=object_poses.copy(),
                object_points_local_demo=object_points,
                object_points_local=object_points.copy(),
                foot_sticking_sequences=foot_sticking,
                q_a_init=initial_q,
                dest_res_path=result_path,
                result_metadata=result_metadata,
            )
            with np.load(result_path, allow_pickle=False) as archive:
                payload = {key: archive[key] for key in archive.files}

        validate_result_artifact(payload)
        stop_reason = str(payload["sqp_stop_reasons"][0])
        self.assertEqual(
            stop_reason,
            "frame_zero_ground_retry:feasible_step_stable",
        )
        self.assertEqual(
            stop_reason.count("frame_zero_ground_retry:"),
            1,
        )
        self.assertTrue(
            bool(payload["frame_zero_ground_retry_triggered"]),
        )
        self.assertEqual(
            int(payload["foot_sticking_full_sequence_retry_frame"]),
            1,
        )
        self.assertEqual(
            iterate_calls,
            [
                (0, True),
                (0, True),
                (1, True),
                (0, False),
                (1, False),
            ],
        )

    def test_only_frames_with_an_active_foot_state_are_retry_eligible(self):
        retargeter = InteractionMeshRetargeter.__new__(InteractionMeshRetargeter)
        retargeter.activate_foot_sticking = True
        retargeter.q_a_init_idx = -7
        retargeter.retry_without_foot_sticking_on_infeasible = True

        self.assertTrue(
            retargeter._has_active_foot_sticking_constraints(
                {"left": True, "right": False},
            )
        )
        self.assertTrue(
            retargeter._should_retry_without_foot_sticking(
                exception=SQPNonlinearFeasibilityError(
                    "SQP nonlinear feasibility failed at frame 3",
                    foot_related_failure=True,
                ),
                foot_sticking={"left": True, "right": False},
                already_retrying=False,
            )
        )
        self.assertFalse(
            retargeter._has_active_foot_sticking_constraints(
                {"left": False, "right": False},
            )
        )
        self.assertFalse(
            retargeter._should_retry_without_foot_sticking(
                exception=SQPNonlinearFeasibilityError(
                    "SQP nonlinear feasibility failed at frame 3",
                    foot_related_failure=True,
                ),
                foot_sticking={"left": False, "right": False},
                already_retrying=False,
            )
        )
        self.assertFalse(
            retargeter._should_retry_without_foot_sticking(
                exception=SQPNonlinearFeasibilityError(
                    "SQP nonlinear feasibility failed at frame 3",
                    foot_related_failure=True,
                ),
                foot_sticking={"left": True, "right": False},
                already_retrying=True,
            )
        )
        self.assertFalse(
            retargeter._should_retry_without_foot_sticking(
                exception=RuntimeError("unrelated failure"),
                foot_sticking={"left": True, "right": False},
                already_retrying=False,
            )
        )
        self.assertFalse(
            retargeter._should_retry_without_foot_sticking(
                exception=SQPNonlinearFeasibilityError(
                    "SQP nonlinear feasibility failed at frame 3",
                    foot_related_failure=False,
                ),
                foot_sticking={"left": True, "right": False},
                already_retrying=False,
            )
        )

        retargeter.q_a_init_idx = 12
        self.assertFalse(
            retargeter._has_active_foot_sticking_constraints(
                {"left": True, "right": True},
            )
        )
        retargeter.q_a_init_idx = -7
        retargeter.activate_foot_sticking = False
        self.assertFalse(
            retargeter._has_active_foot_sticking_constraints(
                {"left": True, "right": True},
            )
        )

    def test_full_sqp_constraint_modes_are_scheduled_in_priority_order(self):
        retargeter = InteractionMeshRetargeter.__new__(InteractionMeshRetargeter)
        retargeter.activate_foot_sticking = True
        retargeter.q_a_init_idx = -7
        retargeter.foot_sticking_tolerance = 0.001
        retargeter.foot_sticking_fallback_tolerance = 0.02
        retargeter.release_foot_sticking_on_infeasible = True
        retargeter.release_object_non_penetration_on_infeasible = True
        retargeter.activate_obj_non_penetration = True
        retargeter.object_name = "clothesstand"
        retargeter.object_model_path = "clothesstand.urdf"

        self.assertEqual(
            retargeter._constraint_mode_schedule(
                {"left": True, "right": False},
            ),
            (
                ConstraintMode("normal"),
                ConstraintMode("relaxed"),
                ConstraintMode("released"),
                ConstraintMode(
                    "normal",
                    object_non_penetration_released=True,
                ),
                ConstraintMode(
                    "relaxed",
                    object_non_penetration_released=True,
                ),
                ConstraintMode(
                    "released",
                    object_non_penetration_released=True,
                ),
            ),
        )

    def test_solver_replaces_only_infeasible_foot_constraints(self):
        x = cp.Variable()
        objective = cp.Minimize(cp.square(x))
        base_constraints = [x <= 0.5]
        normal_foot_constraints = [x >= 1.0]
        fallback_foot_constraints = [x >= -0.1]

        problem, mode = InteractionMeshRetargeter._solve_with_foot_sticking_fallback(
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

        self.assertEqual(mode.foot_sticking, "relaxed")
        self.assertFalse(mode.object_non_penetration_released)
        self.assertIn(problem.status, (cp.OPTIMAL, cp.OPTIMAL_INACCURATE))
        self.assertLessEqual(float(x.value), 0.5 + 1e-8)
        self.assertGreaterEqual(float(x.value), -0.1 - 1e-8)

    def test_solver_can_release_only_foot_constraints_as_last_resort(self):
        x = cp.Variable()
        objective = cp.Minimize(cp.square(x))
        base_constraints = [x <= 0.5]

        problem, mode = InteractionMeshRetargeter._solve_with_foot_sticking_fallback(
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

        self.assertEqual(mode.foot_sticking, "released")
        self.assertFalse(mode.object_non_penetration_released)
        self.assertIn(problem.status, (cp.OPTIMAL, cp.OPTIMAL_INACCURATE))
        self.assertLessEqual(float(x.value), 0.5 + 1e-8)

    def test_solver_can_release_only_object_constraints_as_last_resort(self):
        x = cp.Variable()
        objective = cp.Minimize(cp.square(x))
        base_constraints = [x <= 0.5]

        problem, mode = InteractionMeshRetargeter._solve_with_foot_sticking_fallback(
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

        self.assertEqual(mode.foot_sticking, "inactive")
        self.assertTrue(mode.object_non_penetration_released)
        self.assertIn(problem.status, (cp.OPTIMAL, cp.OPTIMAL_INACCURATE))
        self.assertLessEqual(float(x.value), 0.5 + 1e-8)

    def test_solver_retries_normal_foot_constraints_after_releasing_object_constraints(self):
        x = cp.Variable()
        objective = cp.Minimize(cp.square(x))

        problem, mode = InteractionMeshRetargeter._solve_with_foot_sticking_fallback(
            objective=objective,
            base_constraints=[x <= 0.5],
            object_non_penetration_constraints=[x >= 1.0],
            foot_sticking_constraints=[x >= 0.2],
            foot_sticking_fallback_constraints=[x >= 0.1],
            solver_kwargs={"verbose": False},
            remove_soc_on_failure=False,
            release_on_failure=True,
            release_object_non_penetration_on_failure=True,
        )

        self.assertEqual(mode.foot_sticking, "normal")
        self.assertTrue(mode.object_non_penetration_released)
        self.assertIn(problem.status, (cp.OPTIMAL, cp.OPTIMAL_INACCURATE))
        self.assertGreaterEqual(float(x.value), 0.2 - 1e-8)

    def test_solver_object_release_reports_the_actual_foot_mode(self):
        cases = {
            "normal": (
                [
                    cp.INFEASIBLE,
                    cp.INFEASIBLE,
                    cp.INFEASIBLE,
                    cp.OPTIMAL,
                ],
                True,
                "normal",
                True,
            ),
            "relaxed": (
                [
                    cp.INFEASIBLE,
                    cp.INFEASIBLE,
                    cp.INFEASIBLE,
                    cp.INFEASIBLE,
                    cp.OPTIMAL,
                ],
                True,
                "relaxed",
                True,
            ),
            "released": (
                [
                    cp.INFEASIBLE,
                    cp.INFEASIBLE,
                    cp.INFEASIBLE,
                    cp.INFEASIBLE,
                    cp.INFEASIBLE,
                    cp.OPTIMAL,
                ],
                True,
                "released",
                True,
            ),
            "foot-release-disabled": (
                [
                    cp.INFEASIBLE,
                    cp.INFEASIBLE,
                    cp.INFEASIBLE,
                    cp.INFEASIBLE,
                ],
                False,
                "normal",
                False,
            ),
        }

        for case_name, (
            statuses,
            release_on_failure,
            expected_resolution,
            expected_object_release,
        ) in cases.items():
            with self.subTest(case_name=case_name):
                pending_statuses = list(statuses)

                def _set_mock_status(
                    problem,
                    *args,
                    _pending_statuses=pending_statuses,
                    **kwargs,
                ):
                    del args, kwargs
                    problem._status = _pending_statuses.pop(0)

                x = cp.Variable()
                with patch.object(
                    cp.Problem,
                    "solve",
                    autospec=True,
                    side_effect=_set_mock_status,
                ) as solve:
                    problem, mode = InteractionMeshRetargeter._solve_with_foot_sticking_fallback(
                        objective=cp.Minimize(cp.square(x)),
                        base_constraints=[],
                        object_non_penetration_constraints=[x >= 1.0],
                        foot_sticking_constraints=[x >= 0.5],
                        foot_sticking_fallback_constraints=[x >= 0.25],
                        solver_kwargs={"verbose": False},
                        remove_soc_on_failure=False,
                        release_on_failure=release_on_failure,
                        release_object_non_penetration_on_failure=True,
                    )

                self.assertEqual(mode.foot_sticking, expected_resolution)
                self.assertEqual(
                    mode.object_non_penetration_released,
                    expected_object_release,
                )
                self.assertEqual(problem.status, statuses[-1])
                self.assertEqual(solve.call_count, len(statuses))
                self.assertEqual(pending_statuses, [])

    def test_constraint_fallback_metadata_matches_the_successful_mode(self):
        retargeter = InteractionMeshRetargeter.__new__(InteractionMeshRetargeter)
        retargeter.foot_sticking_tolerance = 0.001
        retargeter.foot_sticking_fallback_tolerance = 0.02
        retargeter.foot_sticking_fallback_frames = set()
        retargeter.foot_sticking_release_frames = set()
        retargeter.object_non_penetration_release_frames = set()

        with patch("builtins.print"):
            retargeter._record_constraint_fallbacks(
                frame_idx=1,
                foot_sticking_resolution=None,
                foot_sticking_fallback_available=True,
                object_non_penetration_released=True,
            )
            retargeter._record_constraint_fallbacks(
                frame_idx=2,
                foot_sticking_resolution="relaxed",
                foot_sticking_fallback_available=True,
                object_non_penetration_released=False,
            )
            retargeter._record_constraint_fallbacks(
                frame_idx=3,
                foot_sticking_resolution="released",
                foot_sticking_fallback_available=True,
                object_non_penetration_released=True,
            )
            retargeter._record_constraint_fallbacks(
                frame_idx=4,
                foot_sticking_resolution="released",
                foot_sticking_fallback_available=False,
                object_non_penetration_released=False,
            )

        self.assertEqual(retargeter.foot_sticking_fallback_frames, {2, 3})
        self.assertEqual(retargeter.foot_sticking_release_frames, {3, 4})
        self.assertEqual(retargeter.object_non_penetration_release_frames, {1, 3})


class NonlinearConstraintDiagnosticsTests(unittest.TestCase):
    @staticmethod
    def _valid_residual_kwargs() -> dict[str, float]:
        return {
            "ground_non_penetration": 0.0,
            "object_non_penetration": 0.0,
            "foot_sticking": 0.0,
            "foot_lock": 0.0,
            "self_collision": 0.0,
            "joint_limits": 0.0,
        }

    def test_residuals_require_finite_non_negative_values(self):
        for field_name in self._valid_residual_kwargs():
            for invalid_value in (-1e-9, np.nan, np.inf):
                with self.subTest(
                    field_name=field_name,
                    invalid_value=invalid_value,
                ), self.assertRaisesRegex(
                    ValueError,
                    "finite and non-negative",
                ):
                    kwargs = self._valid_residual_kwargs()
                    kwargs[field_name] = invalid_value
                    NonlinearConstraintResiduals(**kwargs)

    def test_constraint_mode_release_flags_require_booleans(self):
        self.assertEqual(
            ConstraintMode(
                "normal",
                object_non_penetration_released=np.bool_(True),
            ).object_non_penetration_released,
            np.bool_(True),
        )
        with self.assertRaisesRegex(TypeError, "must be boolean"):
            ConstraintMode(
                "normal",
                object_non_penetration_released=1,
            )
        with self.assertRaisesRegex(TypeError, "must be boolean"):
            ConstraintMode(
                "normal",
                trust_region_released="yes",
            )


class SqpBacktrackingTests(unittest.TestCase):
    @staticmethod
    def _candidate(
        alpha: float,
        *,
        feasible_through: float,
    ) -> SQPIterationResult:
        ground_violation = 0.0 if alpha <= feasible_through else 0.01
        mode = ConstraintMode(
            "inactive",
            object_non_penetration_released=True,
        )
        return SQPIterationResult(
            q=np.asarray([alpha], dtype=np.float64),
            linearized_cost=7.0 + alpha**2,
            constraint_mode=mode,
            residuals=NonlinearConstraintResiduals(
                ground_non_penetration=ground_violation,
                object_non_penetration=0.1 * alpha,
                foot_sticking=0.0,
                foot_lock=0.0,
                self_collision=0.0,
                joint_limits=0.0,
            ),
        )

    def test_feasible_incumbent_survives_when_only_zero_step_is_feasible(self):
        result = InteractionMeshRetargeter._accept_or_backtrack_candidate(
            lambda alpha: self._candidate(
                alpha,
                feasible_through=0.0,
            ),
            bisection_iterations=12,
        )

        np.testing.assert_array_equal(result.q, np.asarray([0.0]))
        self.assertEqual(result.linearized_cost, 7.0)
        self.assertTrue(result.residuals.is_feasible(result.constraint_mode))

    def test_largest_nonzero_feasible_backtrack_keeps_one_point_diagnostics(self):
        feasible_through = 0.625
        result = InteractionMeshRetargeter._accept_or_backtrack_candidate(
            lambda alpha: self._candidate(
                alpha,
                feasible_through=feasible_through,
            ),
            bisection_iterations=16,
        )

        accepted_alpha = float(result.q[0])
        self.assertGreater(accepted_alpha, 0.0)
        self.assertLessEqual(accepted_alpha, feasible_through)
        self.assertAlmostEqual(
            accepted_alpha,
            feasible_through,
            delta=2.0**-16,
        )
        self.assertAlmostEqual(
            result.linearized_cost,
            7.0 + accepted_alpha**2,
        )
        self.assertAlmostEqual(
            result.residuals.object_non_penetration,
            0.1 * accepted_alpha,
        )
        self.assertTrue(result.residuals.is_feasible(result.constraint_mode))
        self.assertTrue(result.constraint_mode.object_non_penetration_released)


class SqpConvergenceTests(unittest.TestCase):
    @staticmethod
    def _retargeter_with_candidates(
        candidates: list[SQPIterationResult],
        *,
        max_iterations: int = 20,
        patience: int = 2,
    ) -> tuple[InteractionMeshRetargeter, list[ConstraintMode]]:
        retargeter = InteractionMeshRetargeter.__new__(InteractionMeshRetargeter)
        retargeter.q_a_indices = np.asarray([0], dtype=int)
        retargeter.q_a_init_idx = -7
        retargeter.step_size = 0.1
        retargeter.sqp_max_iterations = max_iterations
        retargeter.sqp_min_iterations = 1
        retargeter.sqp_convergence_patience = patience
        retargeter.last_sqp_iteration_count = 0
        retargeter.last_sqp_stop_reason = "not_started"
        retargeter.foot_sticking_tolerance = 0.001
        retargeter.foot_sticking_fallback_tolerance = None
        retargeter.foot_sticking_fallback_frames = set()
        retargeter.foot_sticking_release_frames = set()
        retargeter.object_non_penetration_release_frames = set()
        retargeter.activate_foot_sticking = False
        retargeter.release_foot_sticking_on_infeasible = True
        retargeter.release_object_non_penetration_on_infeasible = False
        retargeter.activate_obj_non_penetration = True
        retargeter.object_name = "ground"
        retargeter.object_model_path = None

        calls: list[ConstraintMode] = []
        candidate_iterator = iter(candidates)

        def fake_single_iteration(**kwargs):
            calls.append(kwargs["requested_mode"])
            return next(candidate_iterator)

        retargeter.solve_single_iteration = fake_single_iteration
        return retargeter, calls

    @staticmethod
    def _candidate(
        q: float,
        *,
        cost: float,
        ground_violation: float = 0.0,
        mode: ConstraintMode | None = None,
    ) -> SQPIterationResult:
        return SQPIterationResult(
            q=np.asarray([q]),
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

    @staticmethod
    def _iterate(
        retargeter: InteractionMeshRetargeter,
        *,
        foot_sticking: dict[str, bool] | None = None,
    ):
        return retargeter.iterate(
            q_locked=np.zeros(1),
            q_n=np.zeros(1),
            q_t_last=np.zeros(1),
            target_laplacian=np.zeros((1, 3)),
            adj_list=[],
            obj_pts_local=np.zeros((0, 3)),
            foot_sticking=foot_sticking
            or {
                "left": False,
                "right": False,
            },
        )

    def test_strict_mode_gets_a_complete_sqp_run_before_any_fallback(self):
        strict_mode = ConstraintMode("normal")
        retargeter, calls = self._retargeter_with_candidates(
            [
                self._candidate(
                    0.1,
                    cost=3.0,
                    ground_violation=0.01,
                    mode=strict_mode,
                ),
                self._candidate(
                    0.2,
                    cost=2.0,
                    ground_violation=0.001,
                    mode=strict_mode,
                ),
                self._candidate(
                    0.3,
                    cost=1.0,
                    mode=strict_mode,
                ),
            ],
            max_iterations=3,
            patience=10,
        )
        retargeter.activate_foot_sticking = True
        retargeter.foot_sticking_fallback_tolerance = 0.02
        retargeter.release_object_non_penetration_on_infeasible = True
        retargeter.object_name = "clothesstand"
        retargeter.object_model_path = "clothesstand.urdf"

        q, cost = self._iterate(
            retargeter,
            foot_sticking={"left": True, "right": False},
        )

        self.assertEqual(calls, [strict_mode, strict_mode, strict_mode])
        self.assertEqual(retargeter.last_sqp_iteration_count, 3)
        self.assertEqual(retargeter.last_constraint_mode, strict_mode)
        self.assertEqual(cost, 1.0)
        np.testing.assert_array_equal(q, np.asarray([0.3]))

    def test_infeasible_iteration_breaks_feasible_stability_streak(self):
        retargeter, calls = self._retargeter_with_candidates(
            [
                self._candidate(1.0, cost=4.0),
                self._candidate(
                    2.0,
                    cost=3.0,
                    ground_violation=0.007,
                ),
                self._candidate(1.0, cost=2.0),
                self._candidate(1.0, cost=1.0),
            ],
            max_iterations=4,
            patience=1,
        )

        q, cost = self._iterate(retargeter)

        self.assertEqual(len(calls), 4)
        self.assertEqual(retargeter.last_sqp_iteration_count, 4)
        self.assertEqual(retargeter.last_sqp_stop_reason, "feasible_step_stable")
        self.assertEqual(cost, 1.0)
        np.testing.assert_array_equal(q, np.asarray([1.0]))

    def test_stops_after_feasible_step_is_stable_for_patience_window(self):
        retargeter, calls = self._retargeter_with_candidates(
            [
                self._candidate(1.0, cost=10.0),
                self._candidate(2.0, cost=9.0),
                self._candidate(2.0, cost=100.0),
                self._candidate(2.0, cost=1_000.0),
                self._candidate(9.0, cost=0.0),
            ],
            patience=2,
        )

        q, cost = self._iterate(retargeter)

        self.assertEqual(len(calls), 4)
        self.assertEqual(retargeter.last_sqp_iteration_count, 4)
        self.assertEqual(retargeter.last_sqp_stop_reason, "feasible_step_stable")
        self.assertEqual(cost, 1_000.0)
        np.testing.assert_array_equal(q, np.asarray([2.0]))

    def test_safety_cap_returns_latest_feasible_same_mode_candidate(self):
        retargeter, calls = self._retargeter_with_candidates(
            [
                self._candidate(1.0, cost=1.0),
                self._candidate(2.0, cost=100.0),
                self._candidate(3.0, cost=10_000.0),
            ],
            max_iterations=3,
            patience=10,
        )

        q, cost = self._iterate(retargeter)

        self.assertEqual(len(calls), 3)
        self.assertEqual(retargeter.last_sqp_stop_reason, "max_iterations")
        self.assertEqual(cost, 10_000.0)
        np.testing.assert_array_equal(q, np.asarray([3.0]))

    def test_infeasible_lower_cost_candidate_cannot_replace_feasible_state(self):
        retargeter, calls = self._retargeter_with_candidates(
            [
                self._candidate(1.0, cost=100.0),
                self._candidate(2.0, cost=0.0, ground_violation=0.007),
            ],
            max_iterations=2,
            patience=10,
        )

        q, cost = self._iterate(retargeter)

        self.assertEqual(len(calls), 2)
        self.assertEqual(cost, 100.0)
        np.testing.assert_array_equal(q, np.asarray([1.0]))


if __name__ == "__main__":
    unittest.main()
