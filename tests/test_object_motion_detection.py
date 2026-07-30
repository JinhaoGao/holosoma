# ruff: noqa: CPY001, E402, PT009
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "holosoma_retargeting"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from holosoma_retargeting.src.utils import extract_object_first_moving_frame


class ObjectMotionDetectionTests(unittest.TestCase):
    def test_sub11_floorlamp_052_sign_flip_does_not_precede_real_motion(self):
        """Reproduce the false frame-2 trigger observed in this OMOMO sample."""

        observed_frame_2 = np.asarray(
            [
                -0.5649685264,
                0.5651276112,
                0.3086365759,
                0.5159309506,
                -0.0398836806,
                -0.3458606303,
                0.9076257944,
            ],
            dtype=np.float32,
        )
        observed_frame_3 = np.asarray(
            [
                0.5651149154,
                -0.5650638342,
                -0.3087747693,
                -0.5157577991,
                -0.0398105383,
                -0.3457886279,
                0.9075508118,
            ],
            dtype=np.float32,
        )
        poses = np.repeat(observed_frame_3[None, :], 39, axis=0)
        poses[:3] = observed_frame_2
        poses[38, 4] += np.float32(0.003)

        raw_pose_delta = np.linalg.norm(np.diff(poses, axis=0), axis=1)
        self.assertEqual(int(np.argmax(raw_pose_delta > 0.0025)), 2)
        self.assertEqual(
            int(extract_object_first_moving_frame(poses)),
            37,
        )

    def test_real_quaternion_rotation_remains_motion(self):
        poses = np.zeros((4, 7), dtype=np.float64)
        poses[:, 0] = 1.0
        angle = 0.01
        poses[2:, 0] = np.cos(angle / 2.0)
        poses[2:, 3] = np.sin(angle / 2.0)

        self.assertEqual(
            int(extract_object_first_moving_frame(poses)),
            1,
        )

    def test_translation_threshold_and_no_motion_contract_are_preserved(self):
        poses = np.zeros((4, 7), dtype=np.float64)
        poses[:, 0] = 1.0

        self.assertEqual(
            int(extract_object_first_moving_frame(poses)),
            0,
        )

        poses[3:, 4] = 0.003
        self.assertEqual(
            int(extract_object_first_moving_frame(poses)),
            2,
        )

    def test_single_frame_and_invalid_inputs_are_handled_explicitly(self):
        self.assertEqual(
            extract_object_first_moving_frame(
                np.asarray([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
            ),
            0,
        )
        for poses, message in (
            (np.zeros((2, 6)), "shape"),
            (np.full((2, 7), np.nan), "finite"),
            (np.zeros((2, 7)), "non-zero quaternions"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(
                ValueError,
                message,
            ):
                extract_object_first_moving_frame(poses)


if __name__ == "__main__":
    unittest.main()
