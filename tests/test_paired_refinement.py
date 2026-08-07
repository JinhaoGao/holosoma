# ruff: noqa: CPY001
"""Regression coverage for two-actor nominal trajectory refinement."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = REPO_ROOT / "src" / "holosoma_retargeting"
PACKAGE_ROOT = PACKAGE_PARENT / "holosoma_retargeting"
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from holosoma_retargeting.config_types.robot import RobotConfig  # noqa: E402
from holosoma_retargeting.paired_retargeting.config import (  # noqa: E402
    InterActorCollisionConfig,
    PairedActorConfig,
    PairedContactConfig,
    PairedFrameWindow,
    PairedRefinementCommand,
    PairedRefinementConfig,
    PairedSourceAlignmentConfig,
)
from holosoma_retargeting.paired_retargeting.result import (  # noqa: E402
    PairedActorTrajectory,
    load_paired_trajectories,
    save_paired_result,
)
from holosoma_retargeting.paired_retargeting.retargeter import PairedTrajectoryRefiner  # noqa: E402
from holosoma_retargeting.paired_retargeting.robot_refine import run_config  # noqa: E402
from holosoma_retargeting.paired_retargeting.scene import PairedRobotScene  # noqa: E402
from holosoma_retargeting.paired_retargeting.visualization import (  # noqa: E402
    PairedViserConfig,
    load_original_human_reference,
    load_paired_visualization_result,
)

_QPOS_LAYOUT = "mujoco_free_root_xyz_wxyz_then_actuated_then_optional_object_free_joint"
_MAPPED_LINKS = (
    "base_link",
    "l_hand_sphere_link",
    "r_hand_sphere_link",
    "l_leg_ankle_roll_link",
    "r_leg_ankle_roll_link",
)


def _trajectory(
    robot_type: str,
    root_x: float,
    *,
    frame_count: int = 1,
    source_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> PairedActorTrajectory:
    robot_config = RobotConfig(robot_type=robot_type)
    model_path = (PACKAGE_ROOT / robot_config.ROBOT_URDF_FILE).with_suffix(".xml")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    qpos = np.repeat(model.qpos0[None, :], frame_count, axis=0)
    qpos[:, 0] = root_x + np.arange(frame_count) * 0.01
    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in _MAPPED_LINKS]
    mapped_human_joints = np.empty((frame_count, len(_MAPPED_LINKS), 3), dtype=np.float64)
    for frame_index in range(frame_count):
        data.qpos[:] = qpos[frame_index]
        mujoco.mj_forward(model, data)
        mapped_human_joints[frame_index] = data.xpos[body_ids] + np.asarray(source_offset)
    actuated = []
    for joint_id in range(model.njnt):
        qpos_address = int(model.jnt_qposadr[joint_id])
        if qpos_address < 7:
            continue
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        actuated.append((qpos_address, str(joint_name)))
    actuated_joint_names = tuple(name for _, name in sorted(actuated))
    return PairedActorTrajectory(
        source_path=Path(f"/{robot_type}_{root_x}.npz"),
        qpos=qpos,
        fps=30.0,
        robot_type=robot_type,
        source_data_format="synthetic",
        mapped_human_joints=mapped_human_joints,
        mapped_human_joint_names=("root", "left_hand", "right_hand", "left_ankle", "right_ankle"),
        mapped_robot_link_names=_MAPPED_LINKS,
        robot_actuated_joint_names=actuated_joint_names,
        qpos_layout=_QPOS_LAYOUT,
        quaternion_convention="wxyz",
        world_coordinate_system="right_handed_z_up",
    )


def _save_single_result(path: Path, trajectory: PairedActorTrajectory, *, fps: float | None = None) -> None:
    payload = {
        "qpos": trajectory.qpos,
        "fps": np.asarray(trajectory.fps if fps is None else fps),
        "robot_type": np.asarray(trajectory.robot_type),
        "task_type": np.asarray("robot_only"),
        "contains_object_in_qpos": np.asarray(False),
        "source_data_format": np.asarray(trajectory.source_data_format),
        "mapped_human_joints": trajectory.mapped_human_joints,
        "mapped_human_joint_names": np.asarray(trajectory.mapped_human_joint_names),
        "mapped_robot_link_names": np.asarray(trajectory.mapped_robot_link_names),
        "robot_actuated_joint_names": np.asarray(trajectory.robot_actuated_joint_names),
        "qpos_layout": np.asarray(trajectory.qpos_layout),
        "quaternion_convention": np.asarray(trajectory.quaternion_convention),
        "world_coordinate_system": np.asarray(trajectory.world_coordinate_system),
    }
    if trajectory.human_position_scale is not None:
        payload["human_position_scale"] = np.asarray(trajectory.human_position_scale)
    if trajectory.source_motion_path is not None:
        payload["source_path"] = np.asarray(str(trajectory.source_motion_path))
    if trajectory.source_fbx is not None:
        payload["source_fbx"] = np.asarray(str(trajectory.source_fbx))
    if trajectory.source_actor is not None:
        payload["source_actor"] = np.asarray(trajectory.source_actor)
    if trajectory.source_xy_origin_m is not None:
        payload["source_xy_origin_m"] = np.asarray(trajectory.source_xy_origin_m)
    np.savez_compressed(path, **payload)


def _identity_config() -> PairedRefinementConfig:
    return PairedRefinementConfig(
        interaction_weight=0.0,
        root_position_weight=0.0,
        root_yaw_weight=0.0,
        collision=InterActorCollisionConfig(enabled=False),
    )


def test_loader_enforces_synchronized_single_result_contract(tmp_path: Path) -> None:
    actor_a = _trajectory("e1_23dof", 0.0, frame_count=2)
    actor_b = _trajectory("e1_24dof", 1.0, frame_count=2)
    actor_a_path = tmp_path / "actor_a.npz"
    actor_b_path = tmp_path / "actor_b.npz"
    _save_single_result(actor_a_path, actor_a)
    _save_single_result(actor_b_path, actor_b, fps=60.0)

    with pytest.raises(ValueError, match="equal fps"):
        load_paired_trajectories(actor_a_path, actor_b_path)

    _save_single_result(actor_b_path, actor_b)
    loaded_a, loaded_b = load_paired_trajectories(actor_a_path, actor_b_path)
    assert loaded_a.qpos.shape == (2, 30)
    assert loaded_b.qpos.shape == (2, 31)


def test_loader_recovers_fbx_scene_origins_from_source_motion(tmp_path: Path) -> None:
    source_fbx = tmp_path / "take.fbx"
    source_paths = (tmp_path / "Skeleton0.npz", tmp_path / "Skeleton1.npz")
    for source_path, actor, origin in zip(
        source_paths,
        ("Skeleton0", "Skeleton1"),
        ((3.0, 4.0), (4.5, 3.5)),
        strict=True,
    ):
        np.savez_compressed(
            source_path,
            source_fbx=np.asarray(str(source_fbx)),
            source_actor=np.asarray(actor),
            source_xy_origin_m=np.asarray(origin),
        )
    actor_a = replace(
        _trajectory("e1_23dof", 0.0),
        source_data_format="fbx_mocap",
        human_position_scale=0.8,
        source_motion_path=source_paths[0],
    )
    actor_b = replace(
        _trajectory("e1_23dof", 0.0),
        source_data_format="fbx_mocap",
        human_position_scale=0.9,
        source_motion_path=source_paths[1],
    )
    result_paths = (tmp_path / "result_a.npz", tmp_path / "result_b.npz")
    _save_single_result(result_paths[0], actor_a)
    _save_single_result(result_paths[1], actor_b)

    loaded_a, loaded_b = load_paired_trajectories(*result_paths)

    np.testing.assert_allclose(loaded_a.source_xy_origin_m, (3.0, 4.0))
    np.testing.assert_allclose(loaded_b.source_xy_origin_m, (4.5, 3.5))
    assert loaded_a.source_actor == "Skeleton0"
    assert loaded_b.source_actor == "Skeleton1"


def test_scene_namespaces_mixed_robot_variants_without_xml_rewrites() -> None:
    actor_a = _trajectory("e1_23dof", 0.0)
    actor_b = _trajectory("e1_24dof", 1.0)
    scene = PairedRobotScene("leader", actor_a, "follower", actor_b)

    assert scene.model.nq == 61
    assert scene.actor_a.qpos_slice == slice(0, 30)
    assert scene.actor_b.qpos_slice == slice(30, 61)
    assert mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, "leader__base_link") > 0
    assert mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, "follower__base_link") > 0


def test_zero_coupling_preserves_both_nominal_trajectories_exactly() -> None:
    actor_a = _trajectory("e1_23dof", 0.0, frame_count=2)
    actor_b = _trajectory("e1_23dof", 1.0, frame_count=2)
    result = PairedTrajectoryRefiner("actor_a", actor_a, "actor_b", actor_b, _identity_config()).refine()

    assert np.array_equal(result.actor_a_qpos, actor_a.qpos)
    assert np.array_equal(result.actor_b_qpos, actor_b.qpos)
    assert np.array_equal(result.sqp_iteration_counts, np.zeros(2, dtype=np.int32))


def test_waist_roll_torso_anchor_does_not_require_mapped_root_body() -> None:
    actor_a = _trajectory("e1_24dof", 0.0)
    actor_b = _trajectory("e1_24dof", 1.0)
    waist_mapped_links = ("waist_roll_link", *actor_a.mapped_robot_link_names[1:])
    torso_source_names = ("Spine", *actor_a.mapped_human_joint_names[1:])
    actor_a = replace(
        actor_a,
        mapped_robot_link_names=waist_mapped_links,
        mapped_human_joint_names=torso_source_names,
    )
    actor_b = replace(
        actor_b,
        mapped_robot_link_names=waist_mapped_links,
        mapped_human_joint_names=torso_source_names,
    )

    refiner = PairedTrajectoryRefiner("actor_a", actor_a, "actor_b", actor_b, _identity_config())
    result = refiner.refine()

    assert refiner.actor_a_source_anchor_index == 0
    assert refiner.actor_b_source_anchor_index == 0
    assert np.array_equal(result.actor_a_qpos, actor_a.qpos)
    assert np.array_equal(result.actor_b_qpos, actor_b.qpos)


def test_fbx_alignment_restores_saved_source_relative_origin() -> None:
    shared_fbx = Path("/recordings/two_actor_take.fbx")
    actor_a = replace(
        _trajectory("e1_23dof", 0.0),
        source_data_format="fbx_mocap",
        human_position_scale=0.8,
        source_fbx=shared_fbx,
        source_actor="Skeleton0",
        source_xy_origin_m=np.asarray((5.0, -4.0)),
    )
    actor_b = replace(
        _trajectory("e1_23dof", 0.0),
        source_data_format="fbx_mocap",
        human_position_scale=1.0,
        source_fbx=shared_fbx,
        source_actor="Skeleton1",
        source_xy_origin_m=np.asarray((6.2, -4.5)),
    )
    result = PairedTrajectoryRefiner("actor_a", actor_a, "actor_b", actor_b, _identity_config()).refine()
    expected_translation = np.asarray((1.08, -0.45, 0.0))

    np.testing.assert_allclose(result.actor_a_source_translation, np.zeros(3))
    np.testing.assert_allclose(result.actor_b_source_translation, expected_translation)
    np.testing.assert_allclose(result.actor_b_qpos[0, :3] - actor_b.qpos[0, :3], expected_translation)
    np.testing.assert_allclose(
        result.interaction_source_vertices[0, len(actor_a.mapped_human_joint_names)]
        - result.interaction_source_vertices[0, 0],
        expected_translation,
    )
    assert result.source_scene_scale == pytest.approx(0.9)
    assert result.source_alignment_mode == "fbx_source_xy_origin"


def test_explicit_source_scene_scale_overrides_the_mean_actor_scale() -> None:
    shared_fbx = Path("/recordings/two_actor_take.fbx")
    actor_a = replace(
        _trajectory("e1_23dof", 0.0),
        source_data_format="fbx_mocap",
        human_position_scale=0.8,
        source_fbx=shared_fbx,
        source_actor="Skeleton0",
        source_xy_origin_m=np.asarray((0.0, 0.0)),
    )
    actor_b = replace(
        _trajectory("e1_23dof", 0.0),
        source_data_format="fbx_mocap",
        human_position_scale=1.0,
        source_fbx=shared_fbx,
        source_actor="Skeleton1",
        source_xy_origin_m=np.asarray((2.0, 1.0)),
    )
    config = replace(
        _identity_config(),
        source_alignment=PairedSourceAlignmentConfig(scene_scale=0.5),
    )
    result = PairedTrajectoryRefiner("actor_a", actor_a, "actor_b", actor_b, config).refine()

    np.testing.assert_allclose(result.actor_b_source_translation, (1.0, 0.5, 0.0))


def test_root_position_objective_tracks_source_actor_relationship() -> None:
    actor_a = _trajectory("e1_23dof", 0.0)
    actor_b = _trajectory("e1_23dof", 1.0, source_offset=(-0.2, 0.1, 0.0))
    source_relative = actor_b.mapped_human_joints[0, 0] - actor_a.mapped_human_joints[0, 0]
    nominal_relative = actor_b.qpos[0, :3] - actor_a.qpos[0, :3]
    config = PairedRefinementConfig(
        nominal_weight=1.0,
        smoothness_weight=0.0,
        interaction_weight=0.0,
        root_position_weight=1000.0,
        root_yaw_weight=0.0,
        max_sqp_iterations=2,
    )
    result = PairedTrajectoryRefiner("actor_a", actor_a, "actor_b", actor_b, config).refine()
    refined_relative = result.actor_b_qpos[0, :3] - result.actor_a_qpos[0, :3]

    assert np.linalg.norm(refined_relative - source_relative) < np.linalg.norm(nominal_relative - source_relative)


def test_interaction_mesh_reduces_cross_actor_source_error() -> None:
    actor_a = _trajectory("e1_23dof", 0.0)
    actor_b = _trajectory("e1_23dof", 1.0, source_offset=(-0.15, 0.04, 0.0))
    identity = PairedTrajectoryRefiner("actor_a", actor_a, "actor_b", actor_b, _identity_config()).refine()
    coupled_config = PairedRefinementConfig(
        nominal_weight=1.0,
        smoothness_weight=0.0,
        interaction_weight=1000.0,
        root_position_weight=0.0,
        root_yaw_weight=0.0,
        max_sqp_iterations=2,
    )
    coupled = PairedTrajectoryRefiner("actor_a", actor_a, "actor_b", actor_b, coupled_config).refine()

    assert coupled.interaction_errors[0] < identity.interaction_errors[0]
    assert not np.array_equal(coupled.actor_a_qpos, actor_a.qpos)
    assert not np.array_equal(coupled.actor_b_qpos, actor_b.qpos)


def test_soft_link_contact_moves_toward_requested_relative_offset() -> None:
    actor_a = _trajectory("e1_23dof", 0.0)
    actor_b = _trajectory("e1_23dof", 0.8)
    scene = PairedRobotScene("actor_a", actor_a, "actor_b", actor_b)
    scene.forward(scene.combine_qpos(actor_a.qpos[0], actor_b.qpos[0]))
    initial_offset, _ = scene.relative_link_position_and_jacobian(
        "l_hand_sphere_link",
        "r_hand_sphere_link",
    )
    target_offset = initial_offset + np.asarray((0.02, -0.01, 0.0))
    contact = PairedContactConfig(
        actor_a_link="l_hand_sphere_link",
        actor_b_link="r_hand_sphere_link",
        target_offset=tuple(target_offset),
        weight=1000.0,
    )
    config = replace(
        _identity_config(),
        nominal_weight=1.0,
        contacts=(contact,),
        max_sqp_iterations=2,
    )
    result = PairedTrajectoryRefiner("actor_a", actor_a, "actor_b", actor_b, config).refine()

    assert result.contact_errors[0, 0] < np.linalg.norm(initial_offset - target_offset)


def test_contact_window_changes_only_active_frames() -> None:
    actor_a = _trajectory("e1_23dof", 0.0, frame_count=2)
    actor_b = _trajectory("e1_23dof", 0.8, frame_count=2)
    scene = PairedRobotScene("actor_a", actor_a, "actor_b", actor_b)
    scene.forward(scene.combine_qpos(actor_a.qpos[1], actor_b.qpos[1]))
    initial_offset, _ = scene.relative_link_position_and_jacobian(
        "l_hand_sphere_link",
        "r_hand_sphere_link",
    )
    target_offset = initial_offset + np.asarray((0.02, -0.01, 0.0))
    contact = PairedContactConfig(
        actor_a_link="l_hand_sphere_link",
        actor_b_link="r_hand_sphere_link",
        target_offset=tuple(target_offset),
        weight=1000.0,
        windows=(PairedFrameWindow(start=1, stop=2),),
    )
    config = replace(
        _identity_config(),
        nominal_weight=1.0,
        contacts=(contact,),
        max_sqp_iterations=2,
    )

    result = PairedTrajectoryRefiner("actor_a", actor_a, "actor_b", actor_b, config).refine()

    assert np.array_equal(result.actor_a_qpos[0], actor_a.qpos[0])
    assert np.array_equal(result.actor_b_qpos[0], actor_b.qpos[0])
    assert not np.array_equal(result.actor_a_qpos[1], actor_a.qpos[1])
    assert not np.array_equal(result.actor_b_qpos[1], actor_b.qpos[1])
    assert np.isnan(result.contact_errors[0, 0])
    assert result.contact_errors[1, 0] < np.linalg.norm(initial_offset - target_offset)


def test_cross_actor_collision_builds_signed_distance_constraints() -> None:
    actor_a = _trajectory("e1_23dof", 0.0)
    actor_b = _trajectory("e1_23dof", 0.01)
    scene = PairedRobotScene("actor_a", actor_a, "actor_b", actor_b)
    scene.forward(scene.combine_qpos(actor_a.qpos[0], actor_b.qpos[0]))
    collision = InterActorCollisionConfig(
        enabled=True,
        minimum_distance=0.01,
        activation_distance=0.1,
        body_pairs=(("base_link", "base_link"),),
    )
    linearizations, minimum_distance = scene.collision_linearizations(collision)

    assert linearizations
    assert minimum_distance < collision.minimum_distance
    assert all(item.jacobian.shape == (scene.model.nq,) for item in linearizations)


def test_cross_actor_collision_solver_separates_overlapping_bases() -> None:
    actor_a = _trajectory("e1_23dof", 0.0)
    actor_b = _trajectory("e1_23dof", 0.01)
    collision = InterActorCollisionConfig(
        enabled=True,
        minimum_distance=0.005,
        activation_distance=0.1,
        body_pairs=(("base_link", "base_link"),),
    )
    scene = PairedRobotScene("actor_a", actor_a, "actor_b", actor_b)
    scene.forward(scene.combine_qpos(actor_a.qpos[0], actor_b.qpos[0]))
    _, initial_distance = scene.collision_linearizations(collision)
    config = replace(
        _identity_config(),
        nominal_weight=0.1,
        collision=collision,
        max_sqp_iterations=5,
        root_translation_step_limit=0.06,
    )

    result = PairedTrajectoryRefiner("actor_a", actor_a, "actor_b", actor_b, config).refine()

    assert result.minimum_inter_actor_distances[0] > initial_distance
    assert result.minimum_inter_actor_distances[0] >= collision.minimum_distance - 2e-4
    assert not np.array_equal(result.actor_a_qpos, actor_a.qpos)
    assert not np.array_equal(result.actor_b_qpos, actor_b.qpos)


def test_nominal_delta_smoothness_limits_abrupt_pair_corrections() -> None:
    actor_a = _trajectory("e1_23dof", 0.0, frame_count=2)
    actor_b = _trajectory("e1_23dof", 1.0, frame_count=2)
    source_points = actor_b.mapped_human_joints.copy()
    source_points[1, :, 0] -= 0.2
    actor_b = replace(actor_b, mapped_human_joints=source_points)

    def _solve(smoothness_weight: float):
        config = PairedRefinementConfig(
            nominal_weight=1.0,
            smoothness_weight=smoothness_weight,
            interaction_weight=0.0,
            root_position_weight=1000.0,
            root_yaw_weight=0.0,
            max_sqp_iterations=3,
        )
        return PairedTrajectoryRefiner("actor_a", actor_a, "actor_b", actor_b, config).refine()

    unsmoothed = _solve(0.0)
    smoothed = _solve(1000.0)

    def _relative_root_corrections(result) -> np.ndarray:
        actor_a_correction = result.actor_a_qpos[:, :3] - actor_a.qpos[:, :3]
        actor_b_correction = result.actor_b_qpos[:, :3] - actor_b.qpos[:, :3]
        return actor_b_correction - actor_a_correction

    unsmoothed_jump = np.linalg.norm(np.diff(_relative_root_corrections(unsmoothed), axis=0))
    smoothed_jump = np.linalg.norm(np.diff(_relative_root_corrections(smoothed), axis=0))
    assert smoothed_jump < unsmoothed_jump


def test_paired_result_saves_replay_and_diagnostic_contract(tmp_path: Path) -> None:
    actor_a = _trajectory("e1_23dof", 0.0)
    actor_b = _trajectory("e1_23dof", 1.0)
    result = PairedTrajectoryRefiner("actor_a", actor_a, "actor_b", actor_b, _identity_config()).refine()
    destination = save_paired_result(result, tmp_path / "paired.npz")

    with np.load(destination, allow_pickle=False) as data:
        assert data["run_kind"].item() == "paired_refinement"
        assert data["actor_a_qpos"].shape == (1, 30)
        assert data["actor_b_qpos"].shape == (1, 30)
        assert data["interaction_source_vertices_w"].shape == (1, 10, 3)
        assert data["interaction_tetrahedra_counts"].shape == (1,)
        assert data["refinement_config_json"].ndim == 0

    visualization = load_paired_visualization_result(destination)
    assert visualization.actor_a.name == "actor_a"
    assert visualization.actor_b.name == "actor_b"
    assert visualization.actor_a.mapped_robot_joints.shape == (1, 5, 3)
    assert visualization.interaction_edges.shape[0] == 1


def test_original_fbx_skeletons_restore_one_shared_scene_for_visualization(tmp_path: Path) -> None:
    source_fbx = tmp_path / "take.fbx"
    source_fbx.write_bytes(b"fbx")
    converted_paths = (tmp_path / "actor_a_source.npz", tmp_path / "actor_b_source.npz")
    source_result_paths = (tmp_path / "actor_a_result.npz", tmp_path / "actor_b_result.npz")
    origins = ((4.0, 2.0), (5.5, 1.5))
    actor_offsets = ((0.0, 0.0, 0.0), (0.1, 0.2, 0.1))
    for index, (converted_path, source_result_path, origin, offset) in enumerate(
        zip(converted_paths, source_result_paths, origins, actor_offsets, strict=True)
    ):
        skeleton = np.asarray(
            [
                [offset, (offset[0], offset[1], offset[2] + 0.5), (offset[0], offset[1], offset[2] + 1.0)],
                [offset, (offset[0] + 0.1, offset[1], offset[2] + 0.5), (offset[0], offset[1], offset[2] + 1.0)],
            ],
            dtype=np.float64,
        )
        np.savez_compressed(
            converted_path,
            source_fbx=np.asarray(str(source_fbx)),
            source_actor=np.asarray(f"Skeleton{index}"),
            source_xy_origin_m=np.asarray(origin),
            source_skeleton_positions=skeleton,
            source_skeleton_joint_names=np.asarray(("Hips", "Spine", "Head")),
            source_skeleton_parent_indices=np.asarray((-1, 0, 1), dtype=np.int32),
            fps=np.asarray(30.0),
        )
        np.savez_compressed(source_result_path, source_path=np.asarray(str(converted_path)))

    actor_a = replace(
        _trajectory("e1_23dof", 0.0, frame_count=2),
        source_path=source_result_paths[0],
        source_data_format="fbx_mocap",
        human_position_scale=0.8,
        source_fbx=source_fbx,
        source_actor="Skeleton0",
        source_xy_origin_m=np.asarray(origins[0]),
    )
    actor_b = replace(
        _trajectory("e1_23dof", 0.0, frame_count=2),
        source_path=source_result_paths[1],
        source_data_format="fbx_mocap",
        human_position_scale=1.0,
        source_fbx=source_fbx,
        source_actor="Skeleton1",
        source_xy_origin_m=np.asarray(origins[1]),
    )
    paired = PairedTrajectoryRefiner("actor_a", actor_a, "actor_b", actor_b, _identity_config()).refine()
    destination = save_paired_result(paired, tmp_path / "paired.npz")
    visualization = load_paired_visualization_result(destination)

    original = load_original_human_reference(
        visualization,
        PairedViserConfig(qpos_npz=destination, original_source_fbx=source_fbx),
    )

    assert original is not None
    assert original.actor_a.points.shape == (2, 3, 3)
    assert original.actor_b.points.shape == (2, 3, 3)
    expected_hips_relative = np.asarray((1.6, -0.3, 0.1))
    np.testing.assert_allclose(
        original.actor_b.points[0, 0] - original.actor_a.points[0, 0],
        expected_hips_relative,
    )
    assert original.native_scale == 1.0
    assert original.actor_a.joint_parent_indices.tolist() == [-1, 0, 1]


def test_standalone_command_composes_load_refine_and_save(tmp_path: Path) -> None:
    actor_a = _trajectory("e1_23dof", 0.0)
    actor_b = _trajectory("e1_23dof", 1.0)
    actor_a_path = tmp_path / "actor_a.npz"
    actor_b_path = tmp_path / "actor_b.npz"
    output_path = tmp_path / "paired.npz"
    _save_single_result(actor_a_path, actor_a)
    _save_single_result(actor_b_path, actor_b)
    command = PairedRefinementCommand(
        actor_a=PairedActorConfig(name="actor_a", result_path=actor_a_path),
        actor_b=PairedActorConfig(name="actor_b", result_path=actor_b_path),
        output_path=output_path,
        refinement=_identity_config(),
    )

    result = run_config(command)

    assert output_path.is_file()
    assert np.array_equal(result.actor_a_qpos, actor_a.qpos)
    assert np.array_equal(result.actor_b_qpos, actor_b.qpos)
