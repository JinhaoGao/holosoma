# ruff: noqa: CPY001

from dataclasses import replace

import mujoco
import numpy as np
import pytest

from holosoma_retargeting.config_types.retargeter import RootStabilityConfig
from holosoma_retargeting.config_types.retargeting import RetargetingCommand, internal_config_from_command
from holosoma_retargeting.retargeting_pipeline import build_retargeter_kwargs_from_config, create_task_constants
from holosoma_retargeting.src.interaction_mesh_retargeter import InteractionMeshRetargeter
from holosoma_retargeting.src.utils import calculate_laplacian_coordinates, create_interaction_mesh, get_adjacency_list


def build(weights):
    cfg = internal_config_from_command(RetargetingCommand(task="robot_only", robot="m5fanqu", dataset="fbx_mocap"))
    constants = create_task_constants(cfg.robot_config, cfg.motion_data_config, cfg.task_config, "robot_only")
    retargeter = replace(
        cfg.retargeter,
        visualize=False,
        position_weights=weights,
        interaction_mesh_weight=0.0,
        root_stability=RootStabilityConfig(),
        orientation_weights={},
        natural_pose_weights={},
        natural_pose_joint_positions={},
    )
    return InteractionMeshRetargeter(**build_retargeter_kwargs_from_config(retargeter, constants, None, "robot_only"))


def test_position_objective_moves_chest_toward_reachable_absolute_target():
    solver = build({"Spine1": 1000.0})
    q = solver.robot_model.qpos0.copy()
    _, current, _ = solver._calc_manipulator_jacobians(q, links=solver.laplacian_match_links, obj_frame=False)
    targets = np.asarray(list(current.values()))
    chest = list(current).index("Spine1")
    targets[chest] += [0.04, 0.0, 0.0]
    floor = np.asarray([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]])
    vertices, tetrahedra = create_interaction_mesh(np.vstack([targets, floor]))
    adjacent = get_adjacency_list(tetrahedra, len(vertices))
    fitted, _ = solver.iterate(
        q_locked=q,
        q_n=q,
        q_t_last=q,
        target_laplacian=calculate_laplacian_coordinates(vertices, adjacent),
        adj_list=adjacent,
        obj_pts_local=floor,
        foot_sticking={},
        mapped_position_targets=targets,
        n_iter=10,
    )
    data = mujoco.MjData(solver.robot_model)
    data.qpos[:] = fitted
    mujoco.mj_forward(solver.robot_model, data)
    assert np.linalg.norm(data.body("torso_tracking_link").xpos - targets[chest]) < 0.001


@pytest.mark.parametrize("weights", [{"LeftLeg": 1.0}, {"Spine1": -1.0}, {"Spine1": float("nan")}])
def test_position_weights_reject_excluded_joints_and_invalid_weights(weights):
    with pytest.raises(ValueError, match="Position weights"):
        build(weights)
