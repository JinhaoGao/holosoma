# ruff: noqa: CPY001, PT009, PT027
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "holosoma_retargeting"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from holosoma_retargeting.config_types.data_type import (  # noqa: E402
    MOCAP_DEMO_JOINTS,
)
from holosoma_retargeting.data_utils.convert_noetix_csv import (  # noqa: E402
    MOCAP_SOURCE_BONES,
    Y_UP_TO_SCENE_BASIS,
    NoetixCsvMotion,
    export_mocap_climb,
    mocap_parent_indices,
    transform_global_rotations_y_up_to_scene_wxyz,
)
from holosoma_retargeting.data_utils.motion_data import (  # noqa: E402
    discover_motion_files,
    load_human_motion,
)


class NoetixCsvOrientationTests(unittest.TestCase):
    def test_right_handed_basis_conjugation_preserves_rotated_vectors(self):
        source_rotation = Rotation.from_euler(
            "zyx",
            [23.0, -41.0, 12.0],
            degrees=True,
        )
        source_xyzw = source_rotation.as_quat()
        source_wxyz = source_xyzw[[3, 0, 1, 2]].reshape(1, 1, 4)

        target_wxyz = transform_global_rotations_y_up_to_scene_wxyz(source_wxyz)[0, 0]
        target_rotation = Rotation.from_quat(target_wxyz[[1, 2, 3, 0]])
        vector = np.asarray([0.2, -0.6, 1.4])

        expected = Y_UP_TO_SCENE_BASIS @ source_rotation.apply(vector)
        actual = target_rotation.apply(Y_UP_TO_SCENE_BASIS @ vector)

        np.testing.assert_allclose(actual, expected, atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(Y_UP_TO_SCENE_BASIS), 1.0)

    def test_zero_source_rotation_is_rejected_instead_of_fabricated(self):
        with self.assertRaisesRegex(ValueError, "finite and non-zero"):
            transform_global_rotations_y_up_to_scene_wxyz(np.zeros((1, 1, 4)))

    def test_export_keeps_legacy_assets_and_adds_canonical_npz(self):
        frame_count = 8
        bone_names = sorted(set(MOCAP_SOURCE_BONES.values()))
        bone_count = len(bone_names)
        positions = np.zeros((frame_count, bone_count, 3), dtype=np.float64)
        positions[..., 2] = np.linspace(0.0, 1.0, bone_count)
        orientations = np.zeros(
            (frame_count, bone_count, 4),
            dtype=np.float64,
        )
        orientations[..., 0] = 1.0
        box_markers = np.tile(
            np.asarray(
                [
                    [-0.5, -0.5, 1.0],
                    [0.5, -0.5, 1.0],
                    [0.5, 0.5, 1.0],
                    [-0.5, 0.5, 1.0],
                ]
            )[None],
            (frame_count, 1, 1),
        )
        motion = NoetixCsvMotion(
            metadata={},
            fps=120.0,
            frame_numbers=np.arange(frame_count, dtype=np.int32),
            times=np.arange(frame_count, dtype=np.float64) / 120.0,
            bone_names=bone_names,
            parent_indices=np.full(bone_count, -1, dtype=np.int32),
            bone_positions_z_up_m=positions,
            bone_rotations_wxyz=orientations,
            skeleton_marker_names=[],
            skeleton_marker_positions_z_up_m=np.empty((frame_count, 0, 3)),
            box_marker_names=["a", "b", "c", "d"],
            box_marker_positions_z_up_m=box_markers,
            box_position_z_up_m=np.tile(
                np.asarray([[0.0, 0.0, 1.0]]),
                (frame_count, 1),
            ),
            box_quat_xyzw=None,
            scene_origin_xy_m=np.zeros(2),
            floor_z_m=0.0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            sequence_dir = export_mocap_climb(
                motion=motion,
                output_root=output_root,
                task_name="climb",
                target_fps=30.0,
                scene_template_dir=None,
                asset_scale=1.0,
                source_csv=Path("source.csv"),
            )

            self.assertTrue((sequence_dir / "climb_joint_positions.npy").is_file())
            self.assertTrue((sequence_dir / "scene_reconstruction.npz").is_file())
            self.assertTrue((sequence_dir / "multi_boxes.urdf").is_file())
            with np.load(
                sequence_dir / "climb.npz",
                allow_pickle=False,
            ) as data:
                self.assertEqual(
                    data["joint_names"].tolist(),
                    MOCAP_DEMO_JOINTS,
                )
                self.assertEqual(
                    data["orientation_joint_names"].tolist(),
                    MOCAP_DEMO_JOINTS,
                )
                self.assertEqual(
                    data["orientation_quaternions_wxyz"].shape,
                    (2, len(MOCAP_DEMO_JOINTS), 4),
                )
                np.testing.assert_array_equal(
                    data["joint_parents"],
                    mocap_parent_indices(),
                )
                self.assertEqual(
                    str(data["orientation_source"]),
                    "bone_rotation_channels_fk",
                )
                self.assertAlmostEqual(float(data["height"]), 1.78)

            loaded = load_human_motion(
                "mocap",
                output_root,
                "climb",
            )
            self.assertEqual(
                discover_motion_files(output_root, "mocap"),
                [sequence_dir / "climb.npz"],
            )

        self.assertEqual(
            loaded.orientation_joint_names,
            tuple(MOCAP_DEMO_JOINTS),
        )
        self.assertEqual(
            loaded.global_joint_quaternions_wxyz.shape,
            (2, len(MOCAP_DEMO_JOINTS), 4),
        )
        self.assertEqual(
            loaded.orientation_source,
            "bone_rotation_channels_fk",
        )
        self.assertTrue(loaded.source_path.name == "climb.npz")


if __name__ == "__main__":
    unittest.main()
