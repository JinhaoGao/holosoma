# ruff: noqa: CPY001, S314

"""Mechanical and objective checks for the supplied 17-DoF M5C model."""

import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from holosoma_retargeting.config_types.data_type import MotionDataConfig
from holosoma_retargeting.config_types.retargeting import RetargetingCommand, internal_config_from_command
from holosoma_retargeting.config_types.task import TaskConfig
from holosoma_retargeting.data_utils.convert_fbx import _SOURCE_JOINT_BY_CANONICAL, _discover_actors
from holosoma_retargeting.retargeting_pipeline import build_retargeter_kwargs_from_config, create_task_constants
from holosoma_retargeting.src.interaction_mesh_retargeter import InteractionMeshRetargeter

PACKAGE = Path(__file__).resolve().parents[1] / "src/holosoma_retargeting/holosoma_retargeting"


@pytest.mark.parametrize("robot", ["m4g", "m5c", "m5d", "m5e", "m5fanqu"])
def test_m5c_source_urdf_fk_and_limits_are_preserved(robot):
    package = "__00-mia-v4_2-G" if robot == "m4g" else f"__00-mia-v5-0{robot[-1].upper()}"
    if robot == "m5fanqu":
        package = "mia-v5-fanqu"
    model_dir = f"models/{robot}" if robot == "m4g" else f"models/m5/{robot}"
    source = ET.parse(PACKAGE / f"{model_dir}/{package}/urdf/update_{package}.urdf").getroot()
    generated = ET.parse(PACKAGE / f"{model_dir}/{robot}_17dof.xml").getroot()
    model = mujoco.MjModel.from_xml_path(str(PACKAGE / f"{model_dir}/{robot}_17dof.xml"))
    data = mujoco.MjData(model)
    assert model.nq == 24 and model.nv == 23 and model.nu == 17
    for link in source.findall("link"):
        inertial = link.find("inertial")
        body = model.body(link.get("name"))
        np.testing.assert_allclose(body.mass, float(inertial.find("mass").get("value")), atol=1e-12)
        np.testing.assert_allclose(body.ipos, np.fromstring(inertial.find("origin").get("xyz"), sep=" "), atol=1e-12)
        inertia = {key: float(value) for key, value in inertial.find("inertia").attrib.items()}
        serialized = generated.find(f".//body[@name='{link.get('name')}']/inertial")
        np.testing.assert_array_equal(
            np.fromstring(serialized.get("fullinertia"), sep=" "),
            [inertia[key] for key in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")],
        )
        expected = np.array(
            [
                [inertia["ixx"], inertia["ixy"], inertia["ixz"]],
                [inertia["ixy"], inertia["iyy"], inertia["iyz"]],
                [inertia["ixz"], inertia["iyz"], inertia["izz"]],
            ]
        )
        rotation = Rotation.from_quat(body.iquat[[1, 2, 3, 0]]).as_matrix()
        # Compilation diagonalizes fullinertia numerically; the serialized
        # tensor above must preserve the source exactly.
        np.testing.assert_allclose(rotation @ np.diag(body.inertia) @ rotation.T, expected, atol=1e-7)
    for side in ("l", "r"):
        for index in range(1, 6):
            geom = model.geom(f"{side}_foot_sphere_{index}_link_geom")
            assert model.geom_contype[geom.id] == 1
            # Each supplementary contact sphere touches the measured sole
            # plane while its marker body remains exactly on that plane.
            np.testing.assert_allclose(model.geom_pos[geom.id, 2] - model.geom_size[geom.id, 0], 0, atol=1e-12)
    for phase in (0.0, 0.37, 0.83):
        transforms = {"base_link": np.eye(4)}
        data.qpos[:] = model.qpos0
        data.qpos[:3] = 0
        for joint in source.findall("joint"):
            index = model.joint(joint.get("name")).id
            limit = joint.find("limit")
            limits = np.array([float(limit.get("lower")), float(limit.get("upper"))])
            np.testing.assert_allclose(model.jnt_range[index], limits, atol=1e-12)
            angle = limits[0] * phase + limits[1] * phase * 0.2
            data.qpos[model.jnt_qposadr[index]] = angle
            origin = joint.find("origin")
            matrix = np.eye(4)
            matrix[:3, 3] = np.fromstring(origin.get("xyz"), sep=" ")
            axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
            matrix[:3, :3] = (
                Rotation.from_euler("xyz", np.fromstring(origin.get("rpy"), sep=" ")).as_matrix()
                @ Rotation.from_rotvec(axis * angle).as_matrix()
            )
            transforms[joint.find("child").get("link")] = transforms[joint.find("parent").get("link")] @ matrix
        mujoco.mj_forward(model, data)
        for name, transform in transforms.items():
            np.testing.assert_allclose(data.body(name).xpos, transform[:3, 3], atol=1e-10)
            np.testing.assert_allclose(data.body(name).xmat.reshape(3, 3), transform[:3, :3], atol=1e-10)


@pytest.mark.parametrize("robot", ["m4g", "m5c", "m5d", "m5e", "m5fanqu"])
def test_m5c_two_axis_shoulder_jacobian_and_t_pose(robot):
    config = internal_config_from_command(
        RetargetingCommand(robot=robot, task="robot_only", dataset="noetix_mocap", shoulder_direction_tracking=True)
    )
    config.retargeter = replace(config.retargeter, visualize=False)
    motion = MotionDataConfig(robot_type=robot, data_format="noetix_mocap")
    constants = create_task_constants(config.robot_config, motion, TaskConfig(object_name="ground"), "robot_only")
    solver = InteractionMeshRetargeter(
        **build_retargeter_kwargs_from_config(config.retargeter, constants, None, "robot_only")
    )
    assert solver.shoulder_joint_qpos_addresses.shape == (2, 2)
    q = solver.robot_model.qpos0.copy()
    directions, jac = solver._get_shoulder_direction_data(q, with_jacobians=True)
    assert directions.shape == (2, 3)
    for side in range(2):
        for address, reduced in zip(
            solver.shoulder_joint_qpos_addresses[side], solver.shoulder_joint_reduced_indices[side]
        ):
            plus, minus = q.copy(), q.copy()
            plus[address] += 1e-6
            minus[address] -= 1e-6
            fd = (
                solver._get_shoulder_direction_data(plus, with_jacobians=False)[0]
                - solver._get_shoulder_direction_data(minus, with_jacobians=False)[0]
            ) / 2e-6
            np.testing.assert_allclose(jac[side, :, reduced], fd[side], atol=1e-6)
    q = solver._orientation_t_pose_robot_qpos()
    directions, _ = solver._get_shoulder_direction_data(q, with_jacobians=False)
    if robot == "m4g":
        # Right T reference respects the 1.17 rad limit (about 0.66 degrees
        # below horizontal due to asymmetric mechanical elbow offsets).
        np.testing.assert_allclose(directions[0], [0, 1, 0], atol=1e-6)
        assert np.dot(directions[1], [0, -1, 0]) > 0.9999
    elif robot == "m5fanqu":
        # Supplied shoulder limits stop 0.046 degrees short of horizontal.
        assert np.dot(directions[0], [0, 1, 0]) > 0.999999
        assert np.dot(directions[1], [0, -1, 0]) > 0.999999
    else:
        np.testing.assert_allclose(directions, [[0, 1, 0], [0, -1, 0]], atol=1e-6)
    # At the source reference pose every calibrated target equals M5C FK;
    # a known global rotation must left-multiply each robot target frame.
    names = tuple(solver.orientation_human_joint_names)
    reference = np.array([motion.resolved_orientation_t_pose_human_quaternions_wxyz[name] for name in names])
    known_rotation = Rotation.from_rotvec([0.3, 0, 0])
    rotated = (known_rotation * Rotation.from_quat(reference[:, [1, 2, 3, 0]])).as_quat()[:, [3, 0, 1, 2]]
    targets, _ = solver._prepare_orientation_targets(np.stack([reference, rotated]), q, 2, names)
    expected, _ = solver._get_robot_link_orientation_data(q, solver.orientation_robot_link_names, with_jacobians=False)
    np.testing.assert_allclose(targets[0], expected, atol=1e-8)
    np.testing.assert_allclose(targets[1], known_rotation.as_matrix() @ expected, atol=1e-8)
    for joint in motion.resolved_orientation_joints_mapping.values():
        assert solver.robot_model.body(joint).id >= 0
    assert not {"Spine", "LeftUpLeg", "RightUpLeg"}.intersection(motion.resolved_joints_mapping)
    assert solver.orientation_tracking_enabled and solver.natural_pose_tracking_enabled


def test_unprefixed_fbx_skeleton_is_scoped_to_hips_tree():
    names = ["Hips"] + sorted(set(_SOURCE_JOINT_BY_CANONICAL.values()) - {"Hips"})
    nodes = [{"name": name} for name in names]
    nodes[0]["children"] = list(range(1, len(nodes)))
    nodes.append({"name": "UnrelatedMesh"})
    actors = _discover_actors(nodes)
    assert set(actors) == {"Skeleton"}
    assert set(actors["Skeleton"]) == set(names)
