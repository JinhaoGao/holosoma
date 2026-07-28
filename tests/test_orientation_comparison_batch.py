# ruff: noqa: CPY001, PT009

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from holosoma_retargeting.examples.run_orientation_ablation import (
    ORIENTATION_JOINTS,
)
from holosoma_retargeting.examples.run_orientation_comparison_batch import (
    _comparison_metrics,
    _quaternion_geodesic_errors,
    _validate_current_result,
)


class OrientationComparisonBatchTest(unittest.TestCase):
    def test_quaternion_geodesic_error_is_sign_invariant(self) -> None:
        identity = np.asarray([[[1.0, 0.0, 0.0, 0.0]]])
        negative_identity = -identity
        quarter_turn = np.asarray([[[np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]]])

        self.assertAlmostEqual(
            float(_quaternion_geodesic_errors(identity, negative_identity)[0, 0]),
            0.0,
        )
        self.assertAlmostEqual(
            float(_quaternion_geodesic_errors(identity, quarter_turn)[0, 0]),
            np.pi / 2.0,
            places=6,
        )

    def test_comparison_metrics_reports_delta_and_percent(self) -> None:
        baseline = {
            "position_mean_m": 2.0,
            "position_p95_m": 4.0,
            "orientation_mean_rad": 1.0,
            "orientation_p95_rad": 2.0,
            "joint_second_difference_mean_rad": 0.5,
        }
        optimal = {
            "position_mean_m": 1.0,
            "position_p95_m": 5.0,
            "orientation_mean_rad": 0.5,
            "orientation_p95_rad": 1.0,
            "joint_second_difference_mean_rad": 0.75,
        }

        comparison = _comparison_metrics(baseline, optimal)

        self.assertEqual(comparison["position_mean_m_delta"], -1.0)
        self.assertEqual(comparison["position_mean_m_change_percent"], -50.0)
        self.assertEqual(
            comparison["joint_second_difference_mean_rad_change_percent"],
            50.0,
        )

    def test_current_result_validation_accepts_complete_schema(self) -> None:
        frames = 2
        joints = len(ORIENTATION_JOINTS)
        with tempfile.TemporaryDirectory() as temporary_directory:
            result_path = Path(temporary_directory) / "result.npz"
            np.savez(
                result_path,
                qpos=np.zeros((frames, 30)),
                human_joints=np.zeros((frames, 25, 3)),
                mapped_human_joints=np.zeros((frames, 15, 3)),
                mapped_robot_joints=np.zeros((frames, 15, 3)),
                orientation_tracking_enabled=np.asarray(True),
                orientation_diagnostics_enabled=np.asarray(True),
                orientation_human_joint_names=np.asarray(
                    ORIENTATION_JOINTS,
                    dtype=str,
                ),
                orientation_robot_link_names=np.asarray(
                    [f"link_{index}" for index in range(joints)],
                    dtype=str,
                ),
                orientation_weights=np.full(joints, 0.085),
                orientation_alignment_mode=np.asarray("t_pose"),
                orientation_alignment_quaternions_wxyz=np.tile(
                    [1.0, 0.0, 0.0, 0.0],
                    (joints, 1),
                ),
                orientation_reference_human_quaternions_wxyz=np.tile(
                    [1.0, 0.0, 0.0, 0.0],
                    (joints, 1),
                ),
                orientation_reference_robot_quaternions_wxyz=np.tile(
                    [1.0, 0.0, 0.0, 0.0],
                    (joints, 1),
                ),
                orientation_reference_robot_qpos=np.zeros(30),
                orientation_target_quaternions_wxyz=np.tile(
                    [1.0, 0.0, 0.0, 0.0],
                    (frames, joints, 1),
                ),
                orientation_robot_quaternions_wxyz=np.tile(
                    [1.0, 0.0, 0.0, 0.0],
                    (frames, joints, 1),
                ),
                orientation_errors_rad=np.zeros((frames, joints)),
                orientation_frame_costs=np.zeros(frames),
            )

            _validate_current_result(
                result_path,
                expected_frames=frames,
                expected_weight=0.085,
                tracking_enabled=True,
            )


if __name__ == "__main__":
    unittest.main()
