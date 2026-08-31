# ruff: noqa: CPY001

from __future__ import annotations

from collections import Counter
from typing import Any

import cvxpy as cp
import numpy as np

from holosoma_retargeting.config_types.retargeter import PlanarFootContactConfig
from holosoma_retargeting.config_types.robot import RobotConfig
from holosoma_retargeting.foot_contact import (
    FootContactState,
    build_planar_foot_contact_plan,
)
from holosoma_retargeting.src.interaction_mesh_retargeter import (
    InteractionMeshRetargeter,
    _ResolvedPlanarFootTarget,
)
from holosoma_retargeting.visualization.result_loader import (
    _load_foot_sticking_npz,
)

_JOINT_NAMES = ("root", "LeftFoot", "LeftToe", "RightFoot", "RightToe")
_PARENT_INDICES = np.asarray((-1, 0, 1, 0, 3), dtype=np.int32)
_TOE_NAMES = ("LeftToe", "RightToe")
_SEMANTIC_SOLE_REGIONS = {
    "heel_positive_lateral",
    "heel_negative_lateral",
    "forefoot_positive_lateral",
    "forefoot_negative_lateral",
    "toe",
}


def _empty_motion(frame_count: int = 90) -> np.ndarray:
    joints = np.zeros((frame_count, len(_JOINT_NAMES), 3), dtype=np.float64)
    joints[:, 0, 2] = 1.0
    return joints


def test_all_supported_robot_families_expose_semantic_sole_points() -> None:
    for robot_type in ("g1", "t1", "e1", "e1_23dof", "e1_24dof", "e2"):
        layout = RobotConfig(robot_type=robot_type).FOOT_CONTACT_LINKS
        assert set(layout) == {"left", "right"}
        assert all(set(side_layout) == _SEMANTIC_SOLE_REGIONS for side_layout in layout.values())


def _set_flat_foot(
    joints: np.ndarray,
    *,
    ankle_index: int,
    toe_index: int,
    xy: tuple[float, float],
    height: float = 0.0,
) -> None:
    joints[:, ankle_index] = (xy[0], xy[1], height + 0.1)
    joints[:, toe_index] = (xy[0] + 0.2, xy[1], height)


def test_flat_feet_share_one_persistent_contact_phase() -> None:
    joints = _empty_motion()
    _set_flat_foot(joints, ankle_index=1, toe_index=2, xy=(0.0, 0.1))
    _set_flat_foot(joints, ankle_index=3, toe_index=4, xy=(0.0, -0.1))

    plan = build_planar_foot_contact_plan(
        joints,
        _JOINT_NAMES,
        _PARENT_INDICES,
        _TOE_NAMES,
        fps=30.0,
    )

    assert tuple(plan.modes[0]) == ("swing", "swing")
    assert set(plan.modes[1:, 0]) == {"flat"}
    assert set(plan.modes[1:, 1]) == {"flat"}
    assert set(plan.phase_ids[1:, 0]) == {0}
    assert set(plan.phase_ids[1:, 1]) == {0}


def test_toe_pivot_keeps_only_the_toe_region_planted() -> None:
    joints = _empty_motion()
    angle = np.linspace(0.0, 0.6, len(joints))
    joints[:, 2] = (0.2, 0.1, 0.0)
    joints[:, 1, 0] = 0.2 - 0.2 * np.cos(angle)
    joints[:, 1, 1] = 0.1 - 0.2 * np.sin(angle)
    joints[:, 1, 2] = 0.1
    _set_flat_foot(joints, ankle_index=3, toe_index=4, xy=(0.0, -0.1))

    plan = build_planar_foot_contact_plan(
        joints,
        _JOINT_NAMES,
        _PARENT_INDICES,
        _TOE_NAMES,
        fps=30.0,
    )

    assert Counter(plan.modes[1:, 0]) == {"pivot": len(joints) - 1}
    assert set(plan.pivot_regions[1:, 0]) == {"toe"}
    np.testing.assert_allclose(
        plan.pivot_uv[1:, 0],
        np.tile((1.0, 0.0), (len(joints) - 1, 1)),
    )
    assert set(plan.phase_ids[1:, 0]) == {0}


def _platform_triangles(*, height: float) -> np.ndarray:
    return np.asarray(
        (
            ((-0.2, 0.0, height), (0.3, 0.0, height), (0.3, 0.2, height)),
            ((-0.2, 0.0, height), (0.3, 0.2, height), (-0.2, 0.2, height)),
        ),
        dtype=np.float64,
    )


def test_elevated_static_foot_without_surface_evidence_remains_in_swing() -> None:
    joints = _empty_motion()
    _set_flat_foot(
        joints,
        ankle_index=1,
        toe_index=2,
        xy=(0.0, 0.1),
        height=0.35,
    )
    _set_flat_foot(joints, ankle_index=3, toe_index=4, xy=(0.0, -0.1))

    plan = build_planar_foot_contact_plan(
        joints,
        _JOINT_NAMES,
        _PARENT_INDICES,
        _TOE_NAMES,
        fps=30.0,
    )

    assert set(plan.modes[:, 0]) == {"swing"}


def test_elevated_static_foot_is_accepted_on_explicit_platform() -> None:
    joints = _empty_motion()
    _set_flat_foot(
        joints,
        ankle_index=1,
        toe_index=2,
        xy=(0.0, 0.1),
        height=0.35,
    )
    _set_flat_foot(joints, ankle_index=3, toe_index=4, xy=(0.0, -0.1))

    plan = build_planar_foot_contact_plan(
        joints,
        _JOINT_NAMES,
        _PARENT_INDICES,
        _TOE_NAMES,
        fps=30.0,
        elevated_support_triangles=_platform_triangles(height=0.35),
    )

    assert set(plan.modes[1:, 0]) == {"flat"}


def test_intentional_slide_tracks_source_relative_displacement() -> None:
    joints = _empty_motion()
    travel = np.linspace(0.0, 0.3, len(joints))
    joints[:, 1, 0] = travel
    joints[:, 1, 1] = 0.1
    joints[:, 1, 2] = 0.1
    joints[:, 2, 0] = travel + 0.2
    joints[:, 2, 1] = 0.1
    _set_flat_foot(joints, ankle_index=3, toe_index=4, xy=(0.0, -0.1))

    plan = build_planar_foot_contact_plan(
        joints,
        _JOINT_NAMES,
        _PARENT_INDICES,
        _TOE_NAMES,
        fps=30.0,
    )

    assert set(plan.modes[1:, 0]) == {"slide"}
    assert not plan.sticking_states[1:, 0].any()
    np.testing.assert_allclose(
        plan.reference_displacements_xy[-1, 0],
        (travel[-1] - travel[1], 0.0),
        atol=2e-3,
    )


def test_target_scale_changes_slide_displacement_but_not_detection() -> None:
    joints = _empty_motion()
    travel = np.linspace(0.0, 0.3, len(joints))
    joints[:, 1, 0] = travel
    joints[:, 1, 1] = 0.1
    joints[:, 1, 2] = 0.1
    joints[:, 2, 0] = travel + 0.2
    joints[:, 2, 1] = 0.1
    _set_flat_foot(joints, ankle_index=3, toe_index=4, xy=(0.0, -0.1))
    scale = 1.37

    source_plan = build_planar_foot_contact_plan(
        joints,
        _JOINT_NAMES,
        _PARENT_INDICES,
        _TOE_NAMES,
        fps=30.0,
    )
    scaled_plan = build_planar_foot_contact_plan(
        joints,
        _JOINT_NAMES,
        _PARENT_INDICES,
        _TOE_NAMES,
        fps=30.0,
        reference_position_scale=scale,
    )

    np.testing.assert_array_equal(scaled_plan.modes, source_plan.modes)
    np.testing.assert_array_equal(
        scaled_plan.pivot_regions,
        source_plan.pivot_regions,
    )
    np.testing.assert_allclose(scaled_plan.pivot_uv, source_plan.pivot_uv)
    np.testing.assert_allclose(
        scaled_plan.reference_displacements_xy,
        source_plan.reference_displacements_xy * scale,
        atol=1e-7,
    )


def test_target_rigid_transform_rotates_and_translates_slide_reference() -> None:
    joints = _empty_motion()
    travel = np.linspace(0.0, 0.3, len(joints))
    joints[:, 1, 0] = travel
    joints[:, 1, 1] = 0.1
    joints[:, 1, 2] = 0.1
    joints[:, 2, 0] = travel + 0.2
    joints[:, 2, 1] = 0.1
    _set_flat_foot(joints, ankle_index=3, toe_index=4, xy=(0.0, -0.1))
    angle = np.pi / 4.0
    rotation = np.asarray(
        (
            (np.cos(angle), -np.sin(angle), 0.0),
            (np.sin(angle), np.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        ),
    )
    rotations = np.tile(rotation, (len(joints), 1, 1))
    translations = np.zeros((len(joints), 3), dtype=np.float64)
    translations[:, 1] = np.linspace(0.0, 0.1, len(joints))

    plan = build_planar_foot_contact_plan(
        joints,
        _JOINT_NAMES,
        _PARENT_INDICES,
        _TOE_NAMES,
        fps=30.0,
        reference_rotation_matrices=rotations,
        reference_translations=translations,
    )

    source_displacement = np.asarray((travel[-1] - travel[1], 0.0, 0.0))
    expected = (rotation @ source_displacement)[:2]
    expected += translations[-1, :2] - translations[1, :2]
    np.testing.assert_allclose(
        plan.reference_displacements_xy[-1, 0],
        expected,
        atol=2e-3,
    )


def _constraint_test_retargeter() -> InteractionMeshRetargeter:
    retargeter = InteractionMeshRetargeter.__new__(InteractionMeshRetargeter)
    retargeter.foot_sticking_tolerance = 1e-3
    retargeter.planar_foot_contact = PlanarFootContactConfig()
    retargeter.foot_contact_link_layout = {
        side: {
            "heel_positive_lateral": f"{side}_h_pos",
            "heel_negative_lateral": f"{side}_h_neg",
            "forefoot_positive_lateral": f"{side}_f_pos",
            "forefoot_negative_lateral": f"{side}_f_neg",
            "toe": f"{side}_toe",
        }
        for side in ("left", "right")
    }
    return retargeter


def _sole_kinematics(retargeter: InteractionMeshRetargeter) -> tuple[dict, dict]:
    jacobians = {}
    positions = {}
    for side, y_sign in (("left", 1.0), ("right", -1.0)):
        layout = retargeter.foot_contact_link_layout[side]
        point_values = {
            "heel_positive_lateral": (-0.05, 0.03 * y_sign),
            "heel_negative_lateral": (-0.05, -0.03 * y_sign),
            "forefoot_positive_lateral": (0.11, 0.03 * y_sign),
            "forefoot_negative_lateral": (0.11, -0.03 * y_sign),
            "toe": (0.14, 0.0),
        }
        for region, link in layout.items():
            positions[link] = np.asarray((*point_values[region], 0.0), dtype=np.float64)
            jacobians[link] = np.eye(3, dtype=np.float64)
    return jacobians, positions


def test_pivot_adds_two_planar_bounds_while_flat_adds_center_and_heading() -> None:
    retargeter = _constraint_test_retargeter()
    jacobians, positions = _sole_kinematics(retargeter)
    variable = cp.Variable(3)

    retargeter._active_planar_foot_targets = {
        "left": _ResolvedPlanarFootTarget(
            side="left",
            mode="pivot",
            phase_id=0,
            pivot_uv=(1.0, 0.0),
            anchor_xy=np.asarray((0.14, 0.0)),
        ),
    }
    pivot_constraints: list[Any] = []
    retargeter._append_contact_plan_constraints(
        pivot_constraints,
        variable,
        jacobians,
        positions,
    )

    retargeter._active_planar_foot_targets = {
        "left": _ResolvedPlanarFootTarget(
            side="left",
            mode="flat",
            phase_id=1,
            pivot_uv=(0.5, 0.0),
            anchor_xy=np.asarray((0.03, 0.0)),
            heading_anchor_xy=np.asarray((0.16, 0.0)),
        ),
    }
    flat_constraints: list[Any] = []
    retargeter._append_contact_plan_constraints(
        flat_constraints,
        variable,
        jacobians,
        positions,
    )

    assert len(pivot_constraints) == 2
    assert pivot_constraints[0].shape == (2,)
    assert len(flat_constraints) == 4
    assert flat_constraints[-1].shape == ()


def test_phase_anchor_is_reused_instead_of_following_the_previous_frame() -> None:
    retargeter = _constraint_test_retargeter()
    retargeter._foot_contact_phase_anchors = {}
    retargeter.foot_links = {
        link: link for side_layout in retargeter.foot_contact_link_layout.values() for link in side_layout.values()
    }
    jacobians, positions = _sole_kinematics(retargeter)

    def kinematics(q, **_kwargs):
        shifted = {name: point + np.asarray((q[0], q[1], 0.0)) for name, point in positions.items()}
        return jacobians, shifted, None

    retargeter._calc_manipulator_jacobians = kinematics
    state = {
        "left": FootContactState(
            mode="pivot",
            phase_id=3,
            pivot_region="toe",
            pivot_uv=(1.0, 0.0),
        ),
        "right": FootContactState(mode="swing"),
    }
    retargeter._prepare_planar_foot_targets(np.zeros(3), state)
    first_anchor = retargeter._active_planar_foot_targets["left"].anchor_xy.copy()
    retargeter._prepare_planar_foot_targets(np.asarray((0.02, 0.01, 0.0)), state)

    np.testing.assert_allclose(
        retargeter._active_planar_foot_targets["left"].anchor_xy,
        first_anchor,
    )


def test_saved_contact_plan_metadata_round_trips(tmp_path) -> None:
    path = tmp_path / "contact_plan.npz"
    modes = np.asarray((("swing", "swing"), ("pivot", "flat")))
    phase_ids = np.asarray(((-1, -1), (0, 0)), dtype=np.int32)
    pivot_uv = np.zeros((2, 2, 2), dtype=np.float32)
    reference_displacements = np.zeros((2, 2, 2), dtype=np.float32)
    np.savez(
        path,
        foot_sticking_states=np.asarray(((False, False), (True, True))),
        foot_sticking_side_names=np.asarray(("left", "right")),
        foot_sticking_enabled_for_saved_trajectory=True,
        foot_sticking_tolerance=1e-3,
        foot_contact_modes=modes,
        foot_contact_phase_ids=phase_ids,
        foot_contact_confidences=np.ones((2, 2), dtype=np.float32),
        foot_contact_pivot_regions=np.asarray((("sole", "sole"), ("toe", "sole"))),
        foot_contact_pivot_uv=pivot_uv,
        foot_contact_reference_displacements_xy=reference_displacements,
        foot_contact_plan_version=1,
    )

    with np.load(path) as data:
        metadata = _load_foot_sticking_npz(data, num_frames=2)

    assert metadata is not None
    np.testing.assert_array_equal(metadata["modes"], modes)
    np.testing.assert_array_equal(metadata["phase_ids"], phase_ids)
    np.testing.assert_array_equal(metadata["pivot_uv"], pivot_uv)
    assert metadata["plan_version"] == 1
