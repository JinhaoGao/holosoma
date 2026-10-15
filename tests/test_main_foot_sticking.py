# ruff: noqa: CPY001

"""Regression checks for main's displacement detector and previous-frame XY bounds."""

import cvxpy as cp
import numpy as np
import pytest
from holosoma_retargeting.src.utils import extract_foot_sticking_sequence_velocity
from test_mapped_position_tracking import build


@pytest.mark.parametrize("names", [("L_Toe", "R_Toe"), ("LeftToeBase", "RightToeBase")])
def test_detector_matches_main_per_frame_xy_displacement(names):
    rng = np.random.default_rng(42)
    joints = np.cumsum(rng.normal(scale=0.008, size=(40, 2, 3)), axis=0)
    # Vertical motion and height do not gate main's contact detector.
    joints[:, :, 2] = np.arange(40)[:, None] * 0.5 + 10
    actual = extract_foot_sticking_sequence_velocity(joints, list(names), names)
    expected = np.vstack(
        [np.zeros((1, 2), dtype=bool), np.linalg.norm(np.diff(joints[:, :, :2], axis=0), axis=2) <= 0.01]
    )
    np.testing.assert_array_equal([[row[n] for n in names] for row in actual], expected)


def test_detector_keeps_single_frame_contacts_and_inclusive_threshold():
    joints = np.zeros((5, 2, 3))
    joints[:, 0, 0] = [0, 0.01, 0.03, 0.03, 0.05]
    states = extract_foot_sticking_sequence_velocity(joints, ["LeftToe", "RightToe"], ["LeftToe", "RightToe"])
    assert [s["LeftToe"] for s in states] == [False, True, False, True, False]
    assert [s["RightToe"] for s in states] == [False, True, True, True, True]


@pytest.mark.parametrize(
    ("enabled", "left", "right", "points"),
    [
        (True, True, True, 10),
        (True, True, False, 5),
        (True, False, True, 5),
        (True, False, False, 0),
        (False, True, True, 0),
    ],
)
def test_solver_constrains_all_active_sole_points_against_previous_frame(monkeypatch, enabled, left, right, points):
    solver = build({})
    solver.activate_foot_sticking = enabled
    monkeypatch.setattr(solver, "_update_jacobians_and_phis_from_q", lambda _q: ({}, {}))
    q = solver.robot_model.qpos0.copy()
    captured = []

    class CapturedError(Exception):
        pass

    def capture_problem(objective, constraints):
        captured.extend(c for c in constraints if c.shape == (2,))
        raise CapturedError

    monkeypatch.setattr(cp, "Problem", capture_problem)
    for previous_x in (0.02, 0.04):
        captured.clear()
        previous = q.copy()
        previous[0] += previous_x
        with pytest.raises(CapturedError):
            solver.solve_single_iteration(
                q_locked=q,
                q_a_n_last=q[solver.q_a_indices],
                q_t_last=previous,
                target_laplacian=np.zeros((len(solver.laplacian_match_links), 3)),
                adj_list=[[] for _ in solver.laplacian_match_links],
                obj_pts_local=np.empty((0, 3)),
                foot_sticking={"LeftToeBase": left, "RightToeBase": right},
                init_t=True,
            )
        assert len(captured) == 2 * points
        for dx, valid in ((0, True), (0.0009, True), (-0.0009, True), (0.0011, False), (-0.0011, False)):
            for constraint in captured:
                step = np.zeros(solver.nq_a)
                step[0] = previous_x + dx
                step[2] = 0.25  # The sticking bounds must not pin Z.
                constraint.variables()[0].value = step
            if captured:
                assert all(np.max(c.violation()) < 1e-8 for c in captured) == valid
