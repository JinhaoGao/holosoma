# ruff: noqa: CPY001, PT009
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
    LAFAN_DEMO_JOINTS,
)
from holosoma_retargeting.data_utils.extract_global_positions import (  # noqa: E402
    LAFAN_CANONICAL_TO_SOURCE,
    Y_UP_TO_Z_UP_BASIS,
    convert_file,
    transform_orientations_y_up_to_z_up,
)

LAFAN_BVH = PACKAGE_ROOT / "holosoma_retargeting" / "demo_data" / "lafan_ori" / "fallAndGetUp3_subject1.bvh"
LEGACY_LAFAN_NPY = PACKAGE_ROOT / "holosoma_retargeting" / "demo_data" / "lafan" / "fallAndGetUp3_subject1.npy"
SOURCE_JOINT_NAMES = (
    "Hips",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToe",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToe",
    "Spine",
    "Spine1",
    "Spine2",
    "Neck",
    "Head",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
)


class LafanOrientationTransformTests(unittest.TestCase):
    def test_reflected_basis_conjugation_preserves_rotated_vectors(self):
        source_rotation = Rotation.from_euler(
            "xyz",
            [17.0, -31.0, 64.0],
            degrees=True,
        )
        source_xyzw = source_rotation.as_quat()
        source_wxyz = source_xyzw[[3, 0, 1, 2]].reshape(1, 1, 4)

        target_wxyz = transform_orientations_y_up_to_z_up(source_wxyz)[0, 0]
        target_rotation = Rotation.from_quat(target_wxyz[[1, 2, 3, 0]])
        source_vector = np.asarray([0.4, -0.8, 1.3])

        expected = Y_UP_TO_Z_UP_BASIS @ source_rotation.apply(source_vector)
        actual = target_rotation.apply(Y_UP_TO_Z_UP_BASIS @ source_vector)

        np.testing.assert_allclose(actual, expected, atol=1e-12)


@unittest.skipUnless(
    LAFAN_BVH.is_file() and LEGACY_LAFAN_NPY.is_file(),
    "local LAFAN source and legacy conversion are unavailable",
)
class LafanConversionIntegrationTests(unittest.TestCase):
    def test_converter_writes_canonical_positions_and_direct_orientations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = convert_file(
                LAFAN_BVH,
                Path(tmpdir),
            )
            with np.load(output_path, allow_pickle=False) as data:
                positions = data["global_joint_positions"]
                orientations = data["orientation_quaternions_wxyz"]
                names = data["joint_names"].tolist()
                orientation_names = data["orientation_joint_names"].tolist()
                parents = data["joint_parents"]
                orientation_source = str(data["orientation_source"])

        legacy_y_up = np.load(LEGACY_LAFAN_NPY, allow_pickle=False)
        source_index = {name: index for index, name in enumerate(SOURCE_JOINT_NAMES)}
        canonical_indices = [source_index[LAFAN_CANONICAL_TO_SOURCE[name]] for name in LAFAN_DEMO_JOINTS]
        expected_positions = legacy_y_up[:, canonical_indices][..., [0, 2, 1]]

        self.assertEqual(names, LAFAN_DEMO_JOINTS)
        self.assertEqual(orientation_names, LAFAN_DEMO_JOINTS)
        self.assertEqual(positions.shape, orientations.shape[:-1] + (3,))
        self.assertEqual(parents.shape, (len(LAFAN_DEMO_JOINTS),))
        self.assertEqual(int(parents[0]), -1)
        self.assertEqual(orientation_source, "bvh_rotation_channels_fk")
        np.testing.assert_allclose(positions, expected_positions, atol=1e-6)
        np.testing.assert_allclose(
            np.linalg.norm(orientations, axis=-1),
            1.0,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
