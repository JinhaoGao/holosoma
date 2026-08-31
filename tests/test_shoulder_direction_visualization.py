# ruff: noqa: CPY001

from pathlib import Path

import numpy as np
from holosoma_retargeting.visualization.shoulder_direction import (
    load_shoulder_direction_diagnostics,
    save_shoulder_direction_diagnostics,
)


def _write_diagnostics(path: Path) -> None:
    frames = 5
    sides = 2
    target = np.zeros((frames, sides, 3), dtype=np.float32)
    target[..., 0] = 1.0
    actual = target.copy()
    actual_q = np.zeros((frames, sides, 3), dtype=np.float32)
    actual_q[:, :, 0] = np.linspace(0.0, 0.2, frames)[:, None]
    np.savez_compressed(
        path,
        shoulder_direction_tracking_enabled=np.asarray(True),
        shoulder_direction_human_arm_names=np.asarray(["LeftArm", "RightArm"]),
        shoulder_direction_joint_names=np.asarray(
            [["l_pitch_joint", "l_roll_joint", "l_yaw_joint"], ["r_pitch_joint", "r_roll_joint", "r_yaw_joint"]],
        ),
        shoulder_direction_target_vectors=target,
        shoulder_direction_actual_vectors=actual,
        shoulder_direction_errors_rad=np.zeros((frames, sides), dtype=np.float32),
        shoulder_direction_actual_joint_positions=actual_q,
        shoulder_direction_singular_values=np.ones((frames, sides, 3), dtype=np.float32),
    )


def test_load_and_render_shoulder_direction_diagnostics(tmp_path: Path) -> None:
    result_path = tmp_path / "result.npz"
    output_path = tmp_path / "diagnostics.png"
    _write_diagnostics(result_path)

    diagnostics = load_shoulder_direction_diagnostics(result_path)
    rendered = save_shoulder_direction_diagnostics(result_path, output_path)

    assert diagnostics.side_names == ("LeftArm", "RightArm")
    assert diagnostics.actual_joint_positions.shape == (5, 2, 3)
    assert rendered == output_path
    assert output_path.stat().st_size > 0
