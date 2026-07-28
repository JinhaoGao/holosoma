# ruff: noqa: PT009

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
    NOETIX_MOCAP_DEMO_JOINTS,
)
from holosoma_retargeting.data_utils.convert_noetix_bvh import (  # noqa: E402
    Y_UP_TO_Z_UP_BASIS,
    convert_file,
    enforce_quaternion_continuity_wxyz,
    transform_orientations_y_up_to_z_up,
)

EXPERIMENT_BVH = (
    PACKAGE_ROOT
    / "holosoma_retargeting"
    / "demo_data"
    / "noetix_ori"
    / "0724_BEITI"
    / "breaking+hippop.bvh_Skeleton1.bvh"
)


class NoetixOrientationTransformTests(unittest.TestCase):
    def test_reflected_basis_conjugation_preserves_rotated_vectors(self):
        rotation_y_up = Rotation.from_euler("xyz", [35.0, -20.0, 80.0], degrees=True)
        xyzw = rotation_y_up.as_quat()
        wxyz = xyzw[[3, 0, 1, 2]].reshape(1, 1, 4)

        transformed_wxyz = transform_orientations_y_up_to_z_up(wxyz)[0, 0]
        transformed_matrix = Rotation.from_quat(
            transformed_wxyz[[1, 2, 3, 0]]
        ).as_matrix()
        vector_y_up = np.array([0.3, -0.7, 1.2])

        expected_z_up = Y_UP_TO_Z_UP_BASIS @ rotation_y_up.apply(vector_y_up)
        actual_z_up = transformed_matrix @ (
            Y_UP_TO_Z_UP_BASIS @ vector_y_up
        )

        np.testing.assert_allclose(actual_z_up, expected_z_up, atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(transformed_matrix), 1.0, places=12)

    def test_quaternion_signs_are_continuous_per_joint(self):
        quaternions = np.array(
            [
                [[1.0, 0.0, 0.0, 0.0]],
                [[-1.0, 0.0, 0.0, 0.0]],
                [[2.0, 0.0, 0.0, 0.0]],
            ]
        )

        continuous = enforce_quaternion_continuity_wxyz(quaternions)

        np.testing.assert_allclose(np.linalg.norm(continuous, axis=-1), 1.0)
        self.assertTrue(
            np.all(np.sum(continuous[:-1] * continuous[1:], axis=-1) >= 0.0)
        )


@unittest.skipUnless(EXPERIMENT_BVH.is_file(), "local Noetix experiment BVH is unavailable")
class NoetixExperimentConversionTests(unittest.TestCase):
    def test_breaking_hippop_contains_aligned_global_orientations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            convert_file(
                EXPERIMENT_BVH,
                output_dir,
                target_fps=30.0,
                drop_jump_threshold_m=2.0,
            )
            output_path = output_dir / f"{EXPERIMENT_BVH.stem}.npz"

            with np.load(output_path, allow_pickle=False) as data:
                positions = data["global_joint_positions"]
                quaternions = data["global_joint_quaternions_wxyz"]

                self.assertEqual(positions.shape, (2660, 22, 3))
                self.assertEqual(quaternions.shape, (2660, 22, 4))
                self.assertEqual(
                    data["joint_names"].tolist(),
                    NOETIX_MOCAP_DEMO_JOINTS,
                )
                self.assertEqual(str(data["source_type"]), "fullbody57_chest_zyx")
                self.assertEqual(str(data["quaternion_convention"]), "wxyz")
                self.assertEqual(
                    str(data["orientation_coordinate_transform"]),
                    "basis_conjugation_xzy_reflection",
                )
                np.testing.assert_allclose(
                    np.linalg.norm(quaternions, axis=-1),
                    1.0,
                    atol=1e-6,
                )
                self.assertTrue(
                    np.all(
                        np.sum(
                            quaternions[:-1] * quaternions[1:],
                            axis=-1,
                        )
                        >= -1e-6
                    )
                )
