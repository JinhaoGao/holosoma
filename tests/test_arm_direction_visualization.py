# ruff: noqa: CPY001

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "holosoma_retargeting"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from holosoma_retargeting.viser_player import ArmDirectionOverlay  # noqa: E402
from holosoma_retargeting.visualization.arm_direction import (  # noqa: E402
    ArmDirectionDiagnostics,
    build_arm_direction_scene_data,
    load_optional_arm_direction_diagnostics,
)
from holosoma_retargeting.visualization.orientation import (  # noqa: E402
    OrientationDiagnostics,
)


class _Scene:
    def __init__(self) -> None:
        self.arrows: dict[str, SimpleNamespace] = {}

    def add_arrows(self, name: str, **kwargs) -> SimpleNamespace:
        handle = SimpleNamespace(**kwargs)
        self.arrows[name] = handle
        return handle


def _identity_quaternions(frames: int, count: int) -> np.ndarray:
    values = np.zeros((frames, count, 4), dtype=np.float32)
    values[..., 0] = 1.0
    return values


def _scene_fixture() -> tuple[ArmDirectionDiagnostics, dict[str, Any]]:
    frames = 2
    human_names = (
        "Hips",
        "LeftArm",
        "LeftForeArm",
        "LeftHand",
        "RightArm",
        "RightForeArm",
        "RightHand",
    )
    human = np.asarray(
        [
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.5, 1.0],
                [0.5, 0.5, 1.0],
                [1.0, 0.5, 1.0],
                [0.0, -0.5, 1.0],
                [0.5, -0.5, 1.0],
                [1.0, -0.5, 1.0],
            ],
        ]
        * frames,
        dtype=np.float32,
    )
    robot_names = (
        "torso",
        "l_pitch_link",
        "r_pitch_link",
        "l_roll_link",
        "l_elbow_link",
        "l_hand_link",
        "r_roll_link",
        "r_elbow_link",
        "r_hand_link",
    )
    robot = np.asarray(
        [
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.5, 1.0],
                [0.0, -0.5, 1.0],
                [0.0, 0.5, 1.0],
                [0.5, 0.5, 1.0],
                [1.0, 0.5, 1.0],
                [0.0, -0.5, 1.0],
                [0.5, -0.5, 1.0],
                [1.0, -0.5, 1.0],
            ],
        ]
        * frames,
        dtype=np.float32,
    )
    diagnostics = ArmDirectionDiagnostics(
        side_names=("LeftArm", "RightArm"),
        shoulder_joint_names=(
            ("l_pitch_joint", "l_roll_joint", "l_yaw_joint"),
            ("r_pitch_joint", "r_roll_joint", "r_yaw_joint"),
        ),
        target_directions_local=np.tile(np.asarray([1.0, 0.0, 0.0]), (frames, 2, 1)),
        actual_directions_local=np.tile(np.asarray([1.0, 0.0, 0.0]), (frames, 2, 1)),
        errors_rad=np.zeros((frames, 2), dtype=np.float32),
        direction_weight=100.0,
        wrist_orientation_weights=np.zeros(2, dtype=np.float32),
        wrist_dof_counts=np.zeros(2, dtype=np.int32),
        human_chain_names=(),
        robot_chain_link_names=(
            ("l_roll_link", "l_elbow_link", "l_hand_link"),
            ("r_roll_link", "r_elbow_link", "r_hand_link"),
        ),
        robot_basis_link_names=("l_pitch_link", "r_pitch_link"),
        robot_torso_link_name="torso",
    )
    inputs: dict[str, Any] = {
        "human_joints": human,
        "human_joint_names": human_names,
        "robot_link_positions": robot,
        "robot_link_names": robot_names,
        "robot_parent_indices": np.asarray((-1, 0, 0, 1, 3, 4, 2, 6, 7), dtype=np.int32),
        "mapped_human_joint_names": human_names[1:],
        "mapped_robot_link_names": robot_names[3:],
        "orientation_diagnostics": OrientationDiagnostics(
            human_joint_names=human_names[1:],
            robot_link_names=robot_names[3:],
            weights=np.ones(6, dtype=np.float32),
            target_quaternions_wxyz=_identity_quaternions(frames, 6),
            robot_quaternions_wxyz=_identity_quaternions(frames, 6),
            errors_rad=np.full((frames, 6), 0.25, dtype=np.float32),
        ),
    }
    return diagnostics, inputs


def test_builds_exact_upper_and_diagnostic_lower_arm_arrows() -> None:
    diagnostics, inputs = _scene_fixture()
    data = build_arm_direction_scene_data(diagnostics=diagnostics, **inputs)

    assert data.source_segments.shape == (2, 2, 2, 2, 3)
    np.testing.assert_allclose(data.robot_target_segments, data.robot_actual_segments)
    np.testing.assert_allclose(data.upper_errors_rad, 0.0)
    np.testing.assert_allclose(data.lower_errors_rad, 0.0)
    assert data.orientation_modes[0] == (
        "replaced by upper direction",
        "diagnostic only",
        "diagnostic only",
    )

    scene = _Scene()
    overlay = ArmDirectionOverlay(
        server=SimpleNamespace(scene=scene),
        data=data,
        shaft_radius=0.01,
        head_radius=0.02,
        head_length=0.03,
        visible=True,
        loop=False,
    )
    assert len(scene.arrows) == 3
    assert "Lower-arm direction has no solver objective" in overlay.status_markdown(0.0)
    overlay.set_visible(False)
    assert all(not handle.visible for handle in scene.arrows.values())


def test_optional_loader_supports_legacy_and_explicit_chain_metadata(tmp_path: Path) -> None:
    path = tmp_path / "result.npz"
    target = np.tile(np.asarray([1.0, 0.0, 0.0]), (2, 2, 1))
    np.savez(
        path,
        shoulder_direction_tracking_enabled=np.asarray(True),
        shoulder_direction_human_arm_names=np.asarray(("LeftArm", "RightArm")),
        shoulder_direction_joint_names=np.asarray(
            (("l_pitch_joint", "l_roll_joint", "l_yaw_joint"), ("r_pitch_joint", "r_roll_joint", "r_yaw_joint")),
        ),
        shoulder_direction_target_vectors=target,
        shoulder_direction_actual_vectors=target,
        shoulder_direction_errors_rad=np.zeros((2, 2)),
        shoulder_direction_weight=np.asarray(100.0),
        shoulder_wrist_axis_orientation_weights=np.zeros(2),
        shoulder_wrist_dof_counts=np.zeros(2, dtype=np.int32),
        shoulder_direction_human_chain_names=np.asarray(
            (("LeftArm", "LeftForeArm", "LeftHand"), ("RightArm", "RightForeArm", "RightHand")),
        ),
        shoulder_direction_robot_chain_link_names=np.asarray(
            (("l_roll", "l_elbow", "l_hand"), ("r_roll", "r_elbow", "r_hand")),
        ),
        shoulder_direction_robot_basis_link_names=np.asarray(("l_pitch", "r_pitch")),
        shoulder_direction_robot_torso_link_name=np.asarray("torso"),
    )

    loaded = load_optional_arm_direction_diagnostics(path, expected_frames=2)

    assert loaded is not None
    assert loaded.human_chain_names[0] == ("LeftArm", "LeftForeArm", "LeftHand")
    assert loaded.robot_chain_link_names[1] == ("r_roll", "r_elbow", "r_hand")
    assert loaded.robot_torso_link_name == "torso"
