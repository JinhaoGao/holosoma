# ruff: noqa: CPY001
"""Convergence must not confuse stable local cost with a changing pose."""

import numpy as np
import pytest
from holosoma_retargeting.config_types.retargeter import SQPConfig
from holosoma_retargeting.src.interaction_mesh_retargeter import InteractionMeshRetargeter


@pytest.mark.parametrize(
    "kwargs",
    [
        {"iterations": 0},
        {"first_frame_iterations": -1},
        {"iterations": True},
        {"cost_rtol": float("nan")},
        {"cost_atol": -1.0},
        {"step_tolerance": -1.0},
    ],
)
def test_invalid_solver_budgets_are_rejected(kwargs):
    with pytest.raises(ValueError, match="SQP"):
        SQPConfig(**kwargs)


def test_stable_cost_keeps_iterating_until_pose_step_is_small():
    solver = object.__new__(InteractionMeshRetargeter)
    solver.sqp = SQPConfig(step_tolerance=1e-4)
    solver.q_a_indices = np.array([0])
    candidates = iter([(np.array([1.0]), 10.0), (np.array([2.0]), 10.0), (np.array([2.00001]), 10.0)])
    solver.solve_single_iteration = lambda **_: next(candidates)
    q, cost = solver.iterate(
        q_locked=np.zeros(1),
        q_n=np.zeros(1),
        q_t_last=np.zeros(1),
        target_laplacian=np.zeros((1, 3)),
        adj_list=[],
        obj_pts_local=np.empty((0, 3)),
        foot_sticking={},
        n_iter=5,
    )
    assert solver.last_sqp_iteration_count == 3
    assert solver.last_sqp_stop_reason == "cost_and_step_stable"
    np.testing.assert_allclose(q, [2.00001])
    assert cost == 10.0
