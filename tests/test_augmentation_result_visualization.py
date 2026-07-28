# ruff: noqa: PT009, PT027

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "holosoma_retargeting"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from holosoma_retargeting.augmentation_viser_player import (  # noqa: E402
    DEFAULT_VARIANTS,
    AugmentationViserConfig,
    discover_variant_paths,
    interpolate_qpos,
    load_result_family,
    split_result_family,
)
from holosoma_retargeting.examples.ablation_viser_player import (  # noqa: E402
    AblationViserConfig,
    ComparisonScene,
    comparison_color,
    interpolate_human_points,
    interpolate_orientation_quaternions,
    load_comparison_results,
    load_human_skeleton,
    load_orientation_diagnostics,
    orientation_axis_segments,
    orientation_joint_indices,
)
from holosoma_retargeting.examples.parallel_robot_retarget import (  # noqa: E402
    generate_augmentation_configs,
)
from holosoma_retargeting.src.utils import augment_object_poses  # noqa: E402


class AugmentationResultVisualizationTests(unittest.TestCase):
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
        augmentations = generate_augmentation_configs(
            "object_interaction",
            augmentation=True,
        )

        self.assertEqual(
            [item["name"] for item in augmentations],
            list(DEFAULT_VARIANTS),
        )
        np.testing.assert_allclose(
            [item["translation"] for item in augmentations],
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
            [item["rotation"] for item in augmentations],
            [0.0, 0.0, 0.0, 0.0, np.pi / 4, -np.pi / 4],
        )

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

    def test_discovers_family_from_any_variant(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            for variant in DEFAULT_VARIANTS:
                self._write_result(directory / f"sub3_largebox_003_{variant}.npz")

            paths = discover_variant_paths(
                directory / "sub3_largebox_003_rot_1.npz",
            )

        self.assertEqual(tuple(paths), DEFAULT_VARIANTS)
        self.assertEqual(paths["original"].name, "sub3_largebox_003_original.npz")
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

            config = AblationViserConfig(
                qpos_npzs=paths,
                labels=("before", "after"),
            )
            labels, results = load_comparison_results(config)

        self.assertEqual(labels, ("before", "after"))
        self.assertEqual(len(results), 2)
        self.assertFalse(results[0].contains_object_in_qpos)
        self.assertEqual(results[0].qpos.shape, (3, 36))

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

    def test_robot_group_toggle_controls_mesh_skeleton_and_axes(self):
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

        scene.set_enabled(False)

        self.assertFalse(scene.enabled)
        self.assertFalse(robot.show_visual)
        self.assertFalse(object_visual.show_visual)
        self.assertFalse(robot_skeleton.enabled)
        self.assertFalse(orientation_overlay.enabled)

    def test_robot_groups_use_classic_red_green_blue_order(self):
        self.assertEqual(comparison_color(0), (220, 53, 69))
        self.assertEqual(comparison_color(1), (25, 135, 84))
        self.assertEqual(comparison_color(2), (13, 110, 253))

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
                human_joints=points,
                human_joint_names=np.asarray(joint_names),
                mapped_human_joint_names=np.asarray(mapped_joint_names),
            )

            skeleton = load_human_skeleton(path, expected_frames=2)

        self.assertEqual(skeleton.joint_names, mapped_joint_names)
        self.assertEqual(skeleton.points.shape, (2, 3, 3))
        self.assertEqual(skeleton.full_points.shape, (2, 22, 3))
        self.assertIn((0, 1), skeleton.edges)
        self.assertIn((0, 2), skeleton.edges)
        np.testing.assert_allclose(
            interpolate_human_points(skeleton.points, 0.5)[:, 0],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
