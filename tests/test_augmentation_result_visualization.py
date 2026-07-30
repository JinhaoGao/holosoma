# ruff: noqa: CPY001, E402, PT009, PT027
from __future__ import annotations

import sys
import tempfile
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "holosoma_retargeting"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from holosoma_retargeting.config_types.data_type import MotionDataConfig
from holosoma_retargeting.config_types.retargeting import RetargetingConfig
from holosoma_retargeting.config_types.robot import RobotConfig
from holosoma_retargeting.config_types.task import TaskConfig
from holosoma_retargeting.multi_viser_player import (
    MultiViserConfig,
    resolve_multi_config,
)
from holosoma_retargeting.retargeting_pipeline import (
    RetargetJobResult,
    _object_pose_rotation_deltas_wxyz,
    _transform_nominal_robot_root_with_object_delta,
    initialize_robot_pose,
    planned_variants,
)
from holosoma_retargeting.retargeting_pipeline import (
    main as run_single_action,
)
from holosoma_retargeting.src.utils import augment_object_poses
from holosoma_retargeting.src.viser_utils import (
    create_motion_control_sliders,
)
from holosoma_retargeting.visualization.multi_scene import (
    ComparisonScene,
    MultiResultViserConfig,
    _make_orientation_overlay,
    compact_layer_label,
    comparison_color,
    comparison_offsets,
    interpolate_human_points,
    load_comparison_results,
    load_human_skeleton,
)
from holosoma_retargeting.visualization.orientation import (
    OrientationDiagnostics,
    interpolate_orientation_quaternions,
    load_orientation_diagnostics,
    orientation_axis_segments,
    orientation_joint_indices,
    orientation_skeleton_point_indices,
)
from holosoma_retargeting.visualization.result_loader import (
    DEFAULT_VARIANTS,
    AugmentationViserConfig,
    discover_variant_paths,
    interpolate_qpos,
    load_result_family,
    split_result_family,
)


class AugmentationResultVisualizationTests(unittest.TestCase):
    def test_multi_entry_accepts_explicit_paths_or_family_but_not_both(self):
        explicit = resolve_multi_config(MultiViserConfig(qpos_npzs=(Path("a.npz"), Path("b.npz"))))
        self.assertEqual(explicit.qpos_npzs, (Path("a.npz"), Path("b.npz")))

        with self.assertRaisesRegex(ValueError, "either --qpos-npzs or --family"):
            resolve_multi_config(MultiViserConfig())
        with self.assertRaisesRegex(ValueError, "either --qpos-npzs or --family"):
            resolve_multi_config(
                MultiViserConfig(
                    qpos_npzs=(Path("a.npz"),),
                    family=Path("a.npz"),
                )
            )

    def _write_result(
        self,
        path: Path,
        *,
        n_frames: int = 3,
        robot_type: str = "g1",
    ) -> None:
        qpos = np.zeros((n_frames, 43), dtype=np.float32)
        qpos[:, 3] = 1.0
        qpos[:, -4] = 1.0
        np.savez(
            path,
            qpos=qpos,
            fps=np.float32(30.0),
            human_joints=np.zeros((n_frames, 3, 3), dtype=np.float32),
            human_joint_names=np.asarray(["Pelvis", "L_Hip", "R_Hip"]),
            mapped_human_joint_names=np.asarray(["Pelvis", "L_Hip", "R_Hip"]),
            mapped_robot_joints=np.zeros((n_frames, 3, 3), dtype=np.float32),
            mapped_robot_link_names=np.asarray(["pelvis_contour_link", "left_hip_pitch_link", "right_hip_pitch_link"]),
            robot_type=np.asarray(robot_type),
            object_name=np.asarray("largebox"),
            object_urdf=np.asarray("models/largebox/largebox.urdf"),
            contains_object_in_qpos=np.asarray(True),
            interaction_source_vertices_w=np.full((1,), np.nan, dtype=np.float32),
            interaction_target_vertices_w=np.full((1,), np.nan, dtype=np.float32),
            interaction_tetrahedra=np.asarray([[-1, -1, -1, -1]], dtype=np.int32),
        )

    def test_official_object_interaction_augmentations_are_stable(self):
        augmentations = planned_variants(
            "object_interaction",
            augmentation=True,
        )

        self.assertEqual(
            [item.name for item in augmentations],
            list(DEFAULT_VARIANTS),
        )
        np.testing.assert_allclose(
            [item.translation for item in augmentations],
            [
                [0.0, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [0.0, 0.2, 0.0],
                [0.0, -0.2, 0.0],
                [0.0, 0.2, 0.0],
                [0.0, -0.2, 0.0],
            ],
        )
        np.testing.assert_allclose(
            [item.rotation for item in augmentations],
            [0.0, 0.0, 0.0, 0.0, np.pi / 4, -np.pi / 4],
        )

    def test_single_action_augmentation_uses_identity_first_family(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "sub1_tripod_001.pt"
            source.touch()
            config = RetargetingConfig(
                task_type="object_interaction",
                robot="g1",
                data_format="omomo",
                task_name=source.stem,
                data_path=root,
                save_dir=root / "results",
                augmentation=True,
                robot_config=RobotConfig(robot_type="g1"),
                motion_data_config=MotionDataConfig(
                    data_format="omomo",
                    robot_type="g1",
                    human_height=1.75,
                ),
                task_config=TaskConfig(),
            )
            jobs = []

            def fake_runner(job):
                jobs.append(job)
                return RetargetJobResult(
                    output_path=job.output_path,
                    source_path=job.source_path,
                    sequence_key=job.sequence_key,
                    variant=job.variant.name,
                )

            with mock.patch(
                "holosoma_retargeting.retargeting_pipeline.run_retargeting_job",
                side_effect=fake_runner,
            ):
                family = run_single_action(config)

        self.assertEqual(
            tuple(result.variant for result in family.results),
            tuple(
                item.name
                for item in planned_variants(
                    "object_interaction",
                    augmentation=True,
                )
            ),
        )
        self.assertTrue(jobs[0].variant.is_identity)
        self.assertEqual(jobs[0].run_kind, "single")
        self.assertTrue(all(job.run_kind == "augmentation" for job in jobs[1:]))
        self.assertTrue(all(job.baseline_config_sha256 == jobs[0].config_sha256 for job in jobs[1:]))

    def test_robot_only_family_never_adds_augmentation_variants(self):
        variants = planned_variants(
            "robot_only",
            augmentation=True,
        )

        self.assertEqual(len(variants), 1)
        self.assertTrue(variants[0].is_identity)

    def test_batched_single_axis_rotation_supports_current_scipy(self):
        object_poses = np.zeros((4, 7), dtype=np.float64)
        object_poses[:, 0] = 1.0

        augmented = augment_object_poses(
            object_poses,
            object_moving_frame_idx=4,
            human_initial_root=np.zeros(3),
            rotation_initial=np.pi / 2,
        )

        np.testing.assert_allclose(
            augmented[:, :4],
            np.tile([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)], (4, 1)),
        )
        np.testing.assert_allclose(augmented[:, 4:], object_poses[:, 4:])

    def test_augmentation_promotes_float32_pose_math_to_float64(self):
        object_poses = np.asarray(
            [
                [
                    0.87591344,
                    0.14888385,
                    -0.0992559,
                    0.4477979,
                    -0.21354938,
                    -0.13631772,
                    0.4019234,
                ],
            ],
            dtype=np.float32,
        )
        human_root = np.asarray(
            [-0.5571562, 0.11420971, 0.921315],
            dtype=np.float32,
        )
        target = augment_object_poses(
            object_poses,
            object_moving_frame_idx=0,
            human_initial_root=human_root,
            local_translation=np.asarray([0.0, -0.2, 0.0]),
            rotation_initial=-np.pi / 4.0,
        )
        q_nominal = np.asarray(
            [
                [
                    -0.71975777,
                    0.29768803,
                    0.7769866,
                    0.75232123,
                    0.02255589,
                    0.02619849,
                    -0.65788878,
                ],
            ],
            dtype=np.float64,
        )
        transformed = _transform_nominal_robot_root_with_object_delta(
            q_nominal,
            object_poses,
            target,
        )

        self.assertEqual(target.dtype, np.dtype(np.float64))
        self.assertLess(
            abs(float(transformed[0, 2] - q_nominal[0, 2])),
            1e-12,
        )

    def test_object_translation_moves_nominal_robot_root_only(self):
        demo_object_poses = np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0],
                [1.0, 0.0, 0.0, 0.0, -1.0, 0.5, 0.0],
            ]
        )
        translation = np.asarray([0.2, -0.3, 0.7])
        target_object_poses = demo_object_poses.copy()
        target_object_poses[:, 4:] += translation
        q_nominal = np.asarray(
            [
                [2.0, 4.0, 6.0, 1.0, 0.0, 0.0, 0.0, 0.25, -0.5, 9.0],
                [3.0, -2.0, 1.0, 1.0, 0.0, 0.0, 0.0, -0.75, 0.8, 8.0],
            ]
        )
        original_nominal = q_nominal.copy()

        transformed = _transform_nominal_robot_root_with_object_delta(
            q_nominal,
            demo_object_poses,
            target_object_poses,
        )

        np.testing.assert_allclose(transformed[:, :3], q_nominal[:, :3] + translation)
        np.testing.assert_allclose(transformed[:, 3:7], q_nominal[:, 3:7])
        np.testing.assert_array_equal(transformed[:, 7:], q_nominal[:, 7:])
        np.testing.assert_array_equal(q_nominal, original_nominal)

    def test_object_pose_rotation_delta_ignores_translation(self):
        demo = np.asarray(
            [
                [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0],
                [-2.0, 0.5, 4.0, 1.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        target = demo.copy()
        target[:, :3] += np.asarray([0.2, -0.3, 0.7])

        deltas = _object_pose_rotation_deltas_wxyz(demo, target)

        np.testing.assert_array_equal(
            deltas,
            np.asarray(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                ]
            ),
        )

    def test_object_pose_rotation_delta_is_target_times_demo_inverse(self):
        demo_angle = np.deg2rad(25.0)
        delta_angle = np.deg2rad(-70.0)
        target_angle = demo_angle + delta_angle
        demo = np.asarray(
            [
                [
                    0.0,
                    0.0,
                    0.0,
                    np.cos(demo_angle / 2.0),
                    0.0,
                    0.0,
                    np.sin(demo_angle / 2.0),
                ],
            ]
        )
        target = demo.copy()
        target[0, 3] = np.cos(target_angle / 2.0)
        target[0, 6] = np.sin(target_angle / 2.0)

        deltas = _object_pose_rotation_deltas_wxyz(demo, target)

        np.testing.assert_allclose(
            deltas[0],
            [
                np.cos(delta_angle / 2.0),
                0.0,
                0.0,
                np.sin(delta_angle / 2.0),
            ],
            atol=1e-12,
        )

    def test_object_pose_rotation_delta_rejects_invalid_input(self):
        poses = np.zeros((2, 7), dtype=np.float64)
        poses[:, 3] = 1.0

        with self.assertRaisesRegex(ValueError, "same frame count"):
            _object_pose_rotation_deltas_wxyz(poses, poses[:1])
        with self.assertRaisesRegex(ValueError, "shape"):
            _object_pose_rotation_deltas_wxyz(poses[:, :6], poses)
        with self.assertRaisesRegex(ValueError, "zero-length"):
            invalid = poses.copy()
            invalid[0, 3:7] = 0.0
            _object_pose_rotation_deltas_wxyz(invalid, poses)

    def test_object_rotation_rotates_nominal_robot_root_about_object(self):
        half_angle = np.pi / 4.0
        target_quaternion = np.asarray([np.cos(half_angle), 0.0, 0.0, np.sin(half_angle)])
        demo_object_poses = np.asarray([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
        target_object_poses = demo_object_poses.copy()
        target_object_poses[0, :4] = target_quaternion
        q_nominal = np.asarray([[2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.3, -0.4]])

        transformed = _transform_nominal_robot_root_with_object_delta(
            q_nominal,
            demo_object_poses,
            target_object_poses,
        )

        np.testing.assert_allclose(transformed[0, :3], [1.0, 1.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(transformed[0, 3:7], target_quaternion, atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(transformed[:, 3:7], axis=1), 1.0, atol=1e-12)
        np.testing.assert_array_equal(transformed[:, 7:], q_nominal[:, 7:])

    def test_object_delta_tracks_each_decaying_frame(self):
        angles = np.asarray([np.pi / 2.0, np.pi / 4.0, np.pi / 8.0, 0.0])
        translations = np.asarray([1.0, 0.5, 0.25, 0.0])
        demo_object_poses = np.zeros((len(angles), 7), dtype=np.float64)
        demo_object_poses[:, 0] = 1.0
        target_object_poses = demo_object_poses.copy()
        target_object_poses[:, 0] = np.cos(angles / 2.0)
        target_object_poses[:, 3] = np.sin(angles / 2.0)
        target_object_poses[:, 4] = translations
        q_nominal = np.zeros((len(angles), 9), dtype=np.float64)
        q_nominal[:, 0] = 1.0
        q_nominal[:, 3] = 1.0
        q_nominal[:, 7:] = [0.2, -0.6]

        transformed = _transform_nominal_robot_root_with_object_delta(
            q_nominal,
            demo_object_poses,
            target_object_poses,
        )

        expected_positions = np.column_stack(
            (
                translations + np.cos(angles),
                np.sin(angles),
                np.zeros(len(angles)),
            )
        )
        expected_quaternions = np.column_stack(
            (
                np.cos(angles / 2.0),
                np.zeros(len(angles)),
                np.zeros(len(angles)),
                np.sin(angles / 2.0),
            )
        )
        np.testing.assert_allclose(transformed[:, :3], expected_positions, atol=1e-12)
        np.testing.assert_allclose(transformed[:, 3:7], expected_quaternions, atol=1e-12)
        np.testing.assert_array_equal(transformed[:, 7:], q_nominal[:, 7:])

    def test_identity_object_delta_preserves_nominal_qpos_without_aliasing(self):
        object_poses = np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0, 1.0, -2.0, 0.5],
                [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5), -3.0, 4.0, 2.0],
            ]
        )
        q_nominal = np.asarray(
            [
                [0.5, 1.0, -0.5, 1.0, 0.0, 0.0, 0.0, 0.7, -0.2],
                [2.0, -1.0, 3.0, np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0, -0.4, 0.9],
            ]
        )

        transformed = _transform_nominal_robot_root_with_object_delta(
            q_nominal,
            object_poses,
            object_poses,
        )

        np.testing.assert_allclose(transformed, q_nominal, atol=1e-12)
        transformed[0, 0] = 123.0
        self.assertEqual(q_nominal[0, 0], 0.5)

    def test_object_delta_rejects_incompatible_shapes_and_frame_counts(self):
        q_nominal = np.zeros((2, 9), dtype=np.float64)
        q_nominal[:, 3] = 1.0
        object_poses = np.zeros((2, 7), dtype=np.float64)
        object_poses[:, 0] = 1.0

        with self.assertRaisesRegex(ValueError, "same frame count"):
            _transform_nominal_robot_root_with_object_delta(
                q_nominal,
                object_poses[:1],
                object_poses,
            )
        with self.assertRaisesRegex(ValueError, "shape"):
            _transform_nominal_robot_root_with_object_delta(
                q_nominal[:, :6],
                object_poses,
                object_poses,
            )
        with self.assertRaisesRegex(ValueError, "at least one frame"):
            _transform_nominal_robot_root_with_object_delta(
                q_nominal[:0],
                object_poses[:0],
                object_poses[:0],
            )
        invalid_object_poses = object_poses.copy()
        invalid_object_poses[0, :4] = 0.0
        with self.assertRaisesRegex(ValueError, "zero-length"):
            _transform_nominal_robot_root_with_object_delta(
                q_nominal,
                invalid_object_poses,
                object_poses,
            )

    def test_object_interaction_initialization_uses_transformed_baseline_root(self):
        demo_object_poses = np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
            ]
        )
        target_object_poses = demo_object_poses.copy()
        target_object_poses[:, 5] += [0.4, 0.2]
        q_baseline = np.zeros((2, 14), dtype=np.float64)
        q_baseline[:, :3] = [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
        q_baseline[:, 3] = 1.0
        q_baseline[:, 7:] = np.arange(14, dtype=np.float64).reshape(2, 7)
        expected = _transform_nominal_robot_root_with_object_delta(
            q_baseline,
            demo_object_poses,
            target_object_poses,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_path = Path(tmpdir) / "identity.npz"
            np.savez(baseline_path, qpos=q_baseline)
            with mock.patch(
                "holosoma_retargeting.retargeting_pipeline.augment_object_poses",
                return_value=target_object_poses,
            ):
                q_init, q_nominal, augmented_objects, _, demo_objects = initialize_robot_pose(
                    "object_interaction",
                    np.zeros((2, 1, 3), dtype=np.float64),
                    demo_object_poses,
                    SimpleNamespace(),
                    None,
                    TaskConfig(),
                    True,
                    Path(tmpdir),
                    "unused",
                    baseline_path=baseline_path,
                )

        np.testing.assert_allclose(q_init, expected[0])
        np.testing.assert_allclose(q_nominal, expected)
        np.testing.assert_array_equal(q_nominal[:, 7:], q_baseline[:, 7:])
        np.testing.assert_allclose(
            augmented_objects,
            target_object_poses[:, [4, 5, 6, 0, 1, 2, 3]],
        )
        np.testing.assert_allclose(
            demo_objects,
            demo_object_poses[:, [4, 5, 6, 0, 1, 2, 3]],
        )

    def test_discovers_family_from_any_variant(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            for variant in DEFAULT_VARIANTS:
                self._write_result(directory / f"sub3_largebox_003_{variant}.npz")

            paths = discover_variant_paths(
                directory / "sub3_largebox_003_rot_1.npz",
            )

        self.assertEqual(tuple(paths), DEFAULT_VARIANTS)
        self.assertEqual(paths["identity"].name, "sub3_largebox_003_identity.npz")
        self.assertEqual(
            split_result_family("sub3_largebox_003_trans_0.npz"),
            ("sub3_largebox_003", "trans_0"),
        )

    def test_load_family_requires_synchronized_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            self._write_result(directory / "sub3_largebox_003_original.npz")
            self._write_result(
                directory / "sub3_largebox_003_trans_0.npz",
                robot_type="e1",
            )
            config = AugmentationViserConfig(
                qpos_npz=directory / "sub3_largebox_003_original.npz",
                variants=("original", "trans_0"),
            )

            with self.assertRaisesRegex(ValueError, "robot_type"):
                load_result_family(config)

    def test_qpos_interpolation_uses_shortest_quaternion_arc(self):
        qpos = np.zeros((2, 43), dtype=np.float32)
        qpos[:, 3] = (1.0, -1.0)
        qpos[:, -4] = (1.0, -1.0)
        qpos[1, 0] = 2.0
        qpos[1, -7] = 4.0

        middle = interpolate_qpos(
            qpos,
            0.5,
            robot_dof=29,
            contains_object=True,
        )

        self.assertAlmostEqual(float(middle[0]), 1.0)
        self.assertAlmostEqual(float(middle[-7]), 2.0)
        np.testing.assert_allclose(middle[3:7], [1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(middle[-4:], [1.0, 0.0, 0.0, 0.0])

    def test_loop_boundary_interpolates_qpos_skeleton_and_axes_to_first_frame(self):
        qpos = np.zeros((2, 9), dtype=np.float64)
        qpos[:, 3] = 1.0
        qpos[1, 0] = 10.0
        qpos_middle = interpolate_qpos(
            qpos,
            1.5,
            robot_dof=2,
            contains_object=False,
            loop=True,
        )

        points = np.asarray(
            [
                [[0.0, 0.0, 0.0]],
                [[10.0, 0.0, 0.0]],
            ]
        )
        point_middle = interpolate_human_points(
            points,
            1.5,
            loop=True,
        )
        quaternions = np.asarray(
            [
                [[1.0, 0.0, 0.0, 0.0]],
                [[np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]],
            ]
        )
        quaternion_middle = interpolate_orientation_quaternions(
            quaternions,
            1.5,
            loop=True,
        )

        self.assertAlmostEqual(qpos_middle[0], 5.0)
        self.assertAlmostEqual(point_middle[0, 0], 5.0)
        np.testing.assert_allclose(
            quaternion_middle[0],
            [
                np.cos(np.pi / 8.0),
                0.0,
                0.0,
                np.sin(np.pi / 8.0),
            ],
            atol=1e-7,
        )

    def test_motion_controller_submits_each_complete_frame_atomically(self):
        class Handle:
            def __init__(self, value=None):
                self.value = value
                self.click_callback = None

            def on_click(self, callback):
                self.click_callback = callback
                return callback

            def on_update(self, callback):
                return callback

            def click(self):
                if self.click_callback is None:
                    raise AssertionError("button has no click callback")
                self.click_callback(None)

        class Gui:
            def __init__(self):
                self.buttons = {}

            @contextmanager
            def add_folder(self, _name):
                yield

            def add_slider(self, _name, **kwargs):
                return Handle(kwargs["initial_value"])

            def add_button(self, _name):
                handle = Handle()
                self.buttons[_name] = handle
                return handle

            def add_number(self, _name, **kwargs):
                return Handle(kwargs["initial_value"])

        class Server:
            def __init__(self):
                self.gui = Gui()
                self.atomic_depth = 0
                self.atomic_entries = 0

            @contextmanager
            def atomic(self):
                self.atomic_entries += 1
                self.atomic_depth += 1
                try:
                    yield
                finally:
                    self.atomic_depth -= 1

        class Robot:
            def __init__(self, server):
                self.server = server
                self.frames = []

            def update_cfg(self, joints):
                if self.server.atomic_depth != 1:
                    raise AssertionError("robot joints were updated outside one atomic frame")
                self.frames.append(np.asarray(joints).copy())

        class Frame:
            def __init__(self, server):
                self.server = server
                self._position = None
                self._wxyz = None

            @property
            def position(self):
                return self._position

            @position.setter
            def position(self, value):
                if self.server.atomic_depth != 1:
                    raise AssertionError("base position was updated outside one atomic frame")
                self._position = np.asarray(value).copy()

            @property
            def wxyz(self):
                return self._wxyz

            @wxyz.setter
            def wxyz(self, value):
                if self.server.atomic_depth != 1:
                    raise AssertionError("base orientation was updated outside one atomic frame")
                self._wxyz = np.asarray(value).copy()

        server = Server()
        robot = Robot(server)
        frame = Frame(server)
        qpos = np.zeros((3, 9), dtype=np.float64)
        qpos[:, 3] = 1.0
        qpos[:, 0] = [0.0, 1.0, 2.0]
        callback_frames = []

        def on_frame(q, frame_float):
            if server.atomic_depth != 1:
                raise AssertionError("overlay callback was updated outside one atomic frame")
            callback_frames.append((np.asarray(q).copy(), frame_float))

        create_motion_control_sliders(
            server=server,
            viser_robot=robot,
            robot_base_frame=frame,
            motion_sequence=qpos,
            robot_dof=2,
            contains_object_in_qpos=False,
            initial_fps=120,
            initial_interp_mult=2,
            on_frame=on_frame,
        )

        self.assertEqual(server.atomic_entries, 1)
        self.assertEqual(len(robot.frames), 1)
        self.assertEqual(len(callback_frames), 1)
        server.gui.buttons["Play / Pause"].click()
        deadline = time.monotonic() + 0.5
        while len(robot.frames) < 4 and time.monotonic() < deadline:
            time.sleep(0.005)
        server.gui.buttons["Play / Pause"].click()

        self.assertGreaterEqual(server.atomic_entries, 4)
        self.assertGreaterEqual(len(robot.frames), 4)
        self.assertEqual(len(callback_frames), len(robot.frames))

    def test_missing_variant_reports_exact_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reference = Path(tmpdir) / "sub3_largebox_003_original.npz"
            self._write_result(reference)

            with self.assertRaisesRegex(FileNotFoundError, "trans_0"):
                discover_variant_paths(reference, ("original", "trans_0"))

    def test_ablation_player_accepts_arbitrary_robot_only_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            paths = (
                directory / "before" / "motion.npz",
                directory / "after" / "motion.npz",
            )
            for path in paths:
                path.parent.mkdir()
                qpos = np.zeros((3, 36), dtype=np.float32)
                qpos[:, 3] = 1.0
                np.savez(
                    path,
                    qpos=qpos,
                    fps=np.float32(30.0),
                    human_joints=np.zeros((3, 3, 3), dtype=np.float32),
                    human_joint_names=np.asarray(["Pelvis", "L_Hip", "R_Hip"]),
                    mapped_human_joint_names=np.asarray(["Pelvis", "L_Hip", "R_Hip"]),
                    mapped_robot_joints=np.zeros((3, 3, 3), dtype=np.float32),
                    mapped_robot_link_names=np.asarray(
                        ["pelvis_contour_link", "left_hip_pitch_link", "right_hip_pitch_link"]
                    ),
                    robot_type=np.asarray("g1"),
                    object_name=np.asarray("ground"),
                    object_urdf=np.asarray(""),
                    contains_object_in_qpos=np.asarray(False),
                )

            config = MultiResultViserConfig(
                qpos_npzs=paths,
                labels=("before", "after"),
            )
            labels, results = load_comparison_results(config)

        self.assertEqual(labels, ("before", "after"))
        self.assertEqual(len(results), 2)
        self.assertFalse(results[0].contains_object_in_qpos)
        self.assertEqual(results[0].qpos.shape, (3, 36))
        self.assertEqual(
            results[0].mapped_robot_link_names,
            ("pelvis_contour_link", "left_hip_pitch_link", "right_hip_pitch_link"),
        )

    def test_ablation_player_accepts_one_result_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "motion.npz"
            self._write_result(path)

            labels, results = load_comparison_results(MultiResultViserConfig(qpos_npzs=(path,), labels=("E1",)))

        self.assertEqual(labels, ("E1",))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].path, path)

    def test_multi_loader_recovers_legacy_layout_and_robot_skeleton_with_fk(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sub3_largebox_003_original.npz"
            qpos = np.zeros((2, 43), dtype=np.float32)
            qpos[:, 3] = 1.0
            qpos[:, -4] = 1.0
            np.savez(
                path,
                qpos=qpos,
                fps=np.float32(30.0),
                human_joints=np.zeros((2, 3, 3), dtype=np.float32),
                human_joint_names=np.asarray(["Pelvis", "L_Hip", "R_Hip"]),
                mapped_human_joint_names=np.asarray(["Pelvis", "L_Hip", "R_Hip"]),
                mapped_robot_link_names=np.asarray(
                    [
                        "pelvis_contour_link",
                        "left_hip_pitch_link",
                        "right_hip_pitch_link",
                    ]
                ),
            )
            robot_urdf = PACKAGE_ROOT / "holosoma_retargeting" / "models" / "g1" / "g1_29dof.urdf"

            _, results = load_comparison_results(
                MultiResultViserConfig(
                    qpos_npzs=(path,),
                    robot_urdf=robot_urdf,
                )
            )

        self.assertTrue(results[0].contains_object_in_qpos)
        self.assertEqual(results[0].object_name, "largebox")
        self.assertEqual(results[0].robot_points.shape, (2, 3, 3))
        self.assertTrue(np.isfinite(results[0].robot_points).all())

    def test_legacy_result_has_no_orientation_overlay(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "legacy.npz"
            self._write_result(path)

            diagnostics = load_orientation_diagnostics(
                path,
                expected_frames=3,
            )

        self.assertIsNone(diagnostics)

    def test_loads_and_subsets_orientation_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "orientation.npz"
            quaternions = np.zeros((3, 2, 4), dtype=np.float32)
            quaternions[..., 0] = 2.0
            np.savez(
                path,
                orientation_human_joint_names=np.asarray(["LeftArm", "RightArm"]),
                orientation_robot_link_names=np.asarray(
                    [
                        "l_arm_shoulder_yaw_link",
                        "r_arm_shoulder_yaw_link",
                    ]
                ),
                orientation_weights=np.asarray([0.025, 0.0]),
                orientation_target_quaternions_wxyz=quaternions,
                orientation_robot_quaternions_wxyz=quaternions,
                orientation_errors_rad=np.zeros((3, 2)),
            )

            diagnostics = load_orientation_diagnostics(
                path,
                expected_frames=3,
            )

        self.assertIsNotNone(diagnostics)
        assert diagnostics is not None
        np.testing.assert_allclose(
            diagnostics.target_quaternions_wxyz[..., 0],
            1.0,
        )
        np.testing.assert_array_equal(
            orientation_joint_indices(diagnostics, ("RightArm",)),
            [1],
        )
        with self.assertRaisesRegex(ValueError, "Unknown orientation joints"):
            orientation_joint_indices(diagnostics, ("LeftFoot",))

    def test_orientation_axes_follow_wxyz_frames(self):
        origins = np.asarray([[1.0, 2.0, 3.0]])
        identity = np.asarray([[1.0, 0.0, 0.0, 0.0]])
        identity_segments = orientation_axis_segments(
            origins,
            identity,
            0.5,
        )
        np.testing.assert_allclose(
            identity_segments[:, 0],
            np.repeat(origins, 3, axis=0),
        )
        np.testing.assert_allclose(
            identity_segments[:, 1],
            [
                [1.5, 2.0, 3.0],
                [1.0, 2.5, 3.0],
                [1.0, 2.0, 3.5],
            ],
        )

        half_turn_z = np.asarray([[[1.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 1.0]]])
        middle = interpolate_orientation_quaternions(
            half_turn_z,
            0.5,
        )
        middle_segments = orientation_axis_segments(
            np.zeros((1, 3)),
            middle,
            1.0,
        )
        np.testing.assert_allclose(
            middle_segments[0, 1],
            [0.0, 1.0, 0.0],
            atol=1e-6,
        )

    def test_robot_mesh_and_skeleton_toggles_are_independent(self):
        robot = SimpleNamespace(show_visual=True)
        object_visual = SimpleNamespace(show_visual=True)
        orientation_overlay = SimpleNamespace(enabled=True)
        orientation_overlay.set_enabled = lambda enabled: setattr(
            orientation_overlay,
            "enabled",
            enabled,
        )
        robot_skeleton = SimpleNamespace(enabled=True)
        robot_skeleton.set_enabled = lambda enabled: setattr(
            robot_skeleton,
            "enabled",
            enabled,
        )
        scene = ComparisonScene(
            label="shoulders",
            result=object(),
            color=(255, 255, 255),
            offset=np.zeros(3),
            robot=robot,
            robot_root=object(),
            object_visual=object_visual,
            object_root=object(),
            robot_skeleton=robot_skeleton,
            orientation_overlay=orientation_overlay,
        )

        scene.set_mesh_enabled(False)

        self.assertFalse(robot.show_visual)
        self.assertFalse(object_visual.show_visual)
        self.assertTrue(robot_skeleton.enabled)
        self.assertTrue(orientation_overlay.enabled)
        self.assertTrue(scene.enabled)

        scene.set_skeleton_enabled(False)

        self.assertFalse(robot_skeleton.enabled)
        self.assertFalse(orientation_overlay.enabled)
        self.assertFalse(scene.enabled)

        scene.set_robot_mesh_enabled(True)
        scene.set_enabled(False)

        self.assertFalse(robot.show_visual)
        self.assertTrue(scene.robot_mesh_enabled)
        self.assertFalse(scene.enabled)

        scene.set_enabled(True)

        self.assertTrue(robot.show_visual)
        self.assertTrue(scene.enabled)

    def test_orientation_frames_anchor_to_displayed_skeleton_keypoints(self):
        diagnostics = SimpleNamespace(
            human_joint_names=("Hips", "LeftArm", "LeftFoot"),
            robot_link_names=(
                "base_link",
                "l_arm_shoulder_yaw_link",
                "l_leg_ankle_roll_link",
            ),
        )

        indices = orientation_skeleton_point_indices(
            diagnostics,
            ("Spine1", "LeftArm", "LeftFoot"),
            (
                "base_link",
                "l_arm_shoulder_roll_link",
                "l_leg_ankle_intermediate_1_link",
            ),
        )

        np.testing.assert_array_equal(indices, [0, 1, 2])

    def test_orientation_overlay_never_creates_per_link_text_labels(self):
        class Scene:
            def add_arrows(self, _name, **kwargs):
                return SimpleNamespace(
                    points=np.asarray(kwargs["points"]),
                    visible=bool(kwargs["visible"]),
                )

            def add_label(self, *_args, **_kwargs):
                raise AssertionError("per-link labels must not be created")

        model = mujoco.MjModel.from_xml_path(
            str(PACKAGE_ROOT / "holosoma_retargeting" / "models" / "g1" / "g1_29dof.xml")
        )
        diagnostics = OrientationDiagnostics(
            human_joint_names=("Hips",),
            robot_link_names=("pelvis",),
            weights=np.asarray([0.85]),
            target_quaternions_wxyz=np.asarray([[[1.0, 0.0, 0.0, 0.0]]]),
            robot_quaternions_wxyz=np.asarray([[[1.0, 0.0, 0.0, 0.0]]]),
            errors_rad=np.asarray([[0.0]]),
        )

        overlay = _make_orientation_overlay(
            server=SimpleNamespace(scene=Scene()),
            namespace="/test",
            diagnostics=diagnostics,
            joint_indices=np.asarray([0], dtype=np.int32),
            model=model,
            initial_q=model.qpos0.copy(),
            skeleton_points=np.zeros((1, 1, 3), dtype=np.float64),
            axis_point_indices=np.asarray([0], dtype=np.int32),
            offset=np.zeros(3),
            config=MultiResultViserConfig(
                qpos_npzs=(Path("unused.npz"),),
            ),
        )

        self.assertFalse(hasattr(overlay, "error_labels"))

    def test_display_layer_labels_are_compact(self):
        self.assertEqual(compact_layer_label("balanced_optimal"), "balanced optimal")
        self.assertEqual(compact_layer_label("  shoulder__focus  "), "shoulder focus")

    def test_single_result_overlays_human_and_robot_without_offset(self):
        human_offset, robot_offsets = comparison_offsets(1, 0.6)

        np.testing.assert_array_equal(human_offset, np.zeros(3))
        np.testing.assert_array_equal(robot_offsets, np.zeros((1, 3)))

    def test_comparison_human_overlays_first_robot(self):
        human_offset, robot_offsets = comparison_offsets(3, 0.6)

        np.testing.assert_array_equal(human_offset, robot_offsets[0])
        np.testing.assert_allclose(np.diff(robot_offsets[:, 0]), 0.6)

    def test_robot_groups_use_colorblind_safe_palette_order(self):
        self.assertEqual(comparison_color(0), (0, 114, 178))
        self.assertEqual(comparison_color(1), (213, 94, 0))
        self.assertEqual(comparison_color(2), (0, 158, 115))

    def test_loads_and_interpolates_complete_noetix_human_skeleton(self):
        joint_names = (
            "Hips",
            "RightUpLeg",
            "RightLeg",
            "RightFoot",
            "RightToeBase",
            "LeftUpLeg",
            "LeftLeg",
            "LeftFoot",
            "LeftToeBase",
            "Spine",
            "Spine1",
            "Spine2",
            "Neck",
            "Head",
            "RightShoulder",
            "RightArm",
            "RightForeArm",
            "RightHand",
            "LeftShoulder",
            "LeftArm",
            "LeftForeArm",
            "LeftHand",
        )
        points = np.zeros((2, len(joint_names), 3), dtype=np.float32)
        points[1, :, 0] = 2.0
        mapped_joint_names = ("Spine1", "LeftArm", "RightArm")
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "result.npz"
            np.savez(
                path,
                qpos=np.zeros((2, 7), dtype=np.float32),
                fps=np.float32(30.0),
                human_joints=points,
                human_joint_names=np.asarray(joint_names),
                mapped_human_joint_names=np.asarray(mapped_joint_names),
                mapped_robot_joints=np.zeros((2, 3, 3), dtype=np.float32),
                mapped_robot_link_names=np.asarray(
                    ("torso_link", "left_shoulder_pitch_link", "right_shoulder_pitch_link")
                ),
            )

            skeleton = load_human_skeleton(path, expected_frames=2)

        self.assertEqual(skeleton.joint_names, mapped_joint_names)
        self.assertEqual(skeleton.points.shape, (2, 3, 3))
        self.assertEqual(skeleton.full_points.shape, (2, 22, 3))
        self.assertIn((0, 1), skeleton.edges)
        self.assertIn((0, 2), skeleton.edges)
        self.assertNotIn((1, 2), skeleton.edges)
        np.testing.assert_allclose(
            interpolate_human_points(skeleton.points, 0.5)[:, 0],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
