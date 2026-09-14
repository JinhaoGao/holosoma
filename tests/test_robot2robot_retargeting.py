# ruff: noqa: CPY001, S314
"""Exercise CSV/FK semantics and the real shared robot-to-robot optimizer."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import tyro
from holosoma_retargeting.config_types.retargeter import SqpConfig
from holosoma_retargeting.config_types.retargeting import RetargeterRuntimeOptions
from holosoma_retargeting.config_types.robot_profiles import load_robot_profile
from holosoma_retargeting.config_types.viser import ViserConfig
from holosoma_retargeting.data_utils.npz_to_csv import convert_npz_to_csv
from holosoma_retargeting.data_utils.robot_motion import RobotKinematics, RobotMotionProfile, load_robot_motion
from holosoma_retargeting.examples.robot2robot_retarget import Robot2RobotCommand, main
from holosoma_retargeting.viser_player import _resolve_runtime_config
from holosoma_retargeting.visualization.result_loader import load_variant_result, variant_result_metadata
from scipy.spatial.transform import Rotation


def kinematics(name: str) -> RobotKinematics:
    return RobotKinematics(RobotMotionProfile.from_profile(load_robot_profile(robot=name)))


def write_motion(tmp_path: Path, name: str, frames: int = 4) -> tuple[Path, np.ndarray]:
    profile = load_robot_profile(robot=name)
    robot = kinematics(name)
    qpos = np.tile(robot.model.qpos0, (frames, 1))
    qpos[:, 3:7] = [1, 0, 0, 0]
    for joint, value in profile.natural_pose.references.items():
        qpos[:, robot.joint_addresses[joint]] = value
    robot.forward(qpos[0])
    qpos[:, 2] -= robot.data.xpos[np.concatenate(list(robot.sole_ids.values())), 2].min()
    shoulder = next(j for j in robot.joint_addresses if "shoulder_pitch" in j)
    qpos[:, robot.joint_addresses[shoulder]] += np.linspace(0, -0.06, frames)
    # Deliberately permute saved order: the exporter and importer must honor names.
    names = list(robot.joint_addresses)[::-1]
    saved = np.concatenate([qpos[:, :7], qpos[:, [robot.joint_addresses[n] for n in names]]], axis=1)
    path = tmp_path / f"{name}_motion.npz"
    np.savez(path, qpos=saved, robot_actuated_joint_names=names)
    return convert_npz_to_csv(path, robot.urdf), qpos


@pytest.mark.parametrize("name", ["g1", "e1_23dof", "e1_24dof", "e2"])
def test_export_csv_import_fk_roundtrip(tmp_path: Path, name: str) -> None:
    csv, expected = write_motion(tmp_path, name)
    robot = kinematics(name)
    np.testing.assert_allclose(robot.read_csv(csv), expected, atol=1e-14)
    motion = load_robot_motion(csv, source=robot, target=robot.profile, fps=50)
    robot.forward(expected[-1])
    np.testing.assert_allclose(motion.joints[-1], robot.data.xpos[robot.position_ids])
    np.testing.assert_allclose(motion.orientation_quaternions_wxyz[-1], robot.data.xquat[robot.orientation_ids])
    left_sole = robot.data.xpos[robot.sole_ids["left"]]
    np.testing.assert_allclose(motion.foot_heel_positions[-1, 0], left_sole[:2].mean(axis=0))
    np.testing.assert_allclose(motion.foot_toe_positions[-1, 0], left_sole[2])


def test_root_rotation_is_measured_fk_and_not_euler_reinterpretation(tmp_path: Path) -> None:
    csv, _ = write_motion(tmp_path, "g1", frames=1)
    values = np.loadtxt(csv, delimiter=",", ndmin=2)
    rotation = Rotation.from_euler("z", 90, degrees=True)
    values[:, 3:7] = rotation.as_quat()
    np.savetxt(csv, values, delimiter=",")
    robot = kinematics("g1")
    motion = load_robot_motion(csv, source=robot, target=robot.profile, fps=50)
    np.testing.assert_allclose(motion.root_quaternions_wxyz[0], rotation.as_quat()[[3, 0, 1, 2]], atol=1e-12)


@pytest.mark.parametrize("error", ["columns", "nan", "zero_quaternion", "empty"])
def test_invalid_csv_fails_before_solver(tmp_path: Path, error: str) -> None:
    csv, _ = write_motion(tmp_path, "g1")
    values = np.loadtxt(csv, delimiter=",")
    if error == "columns":
        values = values[:, :-1]
    elif error == "nan":
        values[0, 0] = np.nan
    elif error == "zero_quaternion":
        values[0, 3:7] = 0
    else:
        values = values[:0]
    np.savetxt(csv, values, delimiter=",")
    with pytest.raises(ValueError, match="Robot CSV"):
        kinematics("g1").read_csv(csv)


def command_for(csv: Path, source: str, target: str, tmp_path: Path) -> Robot2RobotCommand:
    return Robot2RobotCommand(
        source_robot=source,
        robot=target,
        input_path=csv,
        fps=50,
        save_dir=tmp_path / "results",
        orientation_tracking=True,
        retargeter=RetargeterRuntimeOptions(visualize=False),
    )


@pytest.mark.parametrize(("source", "target"), [("g1", "e2"), ("e2", "g1"), ("e1_23dof", "e2")])
def test_shared_solver_and_both_outputs(tmp_path: Path, source: str, target: str) -> None:
    csv, _ = write_motion(tmp_path, source)
    command = command_for(csv, source, target, tmp_path)
    result = main(command)
    target_robot = kinematics(target)
    with np.load(result.job.output_path, allow_pickle=False) as data:
        qpos = data["qpos"].copy()
        assert qpos.shape == (4, target_robot.model.nq)
        assert np.isfinite(qpos).all()
        assert data["fps"] == 50
        assert data["source_data_format"] == "robot"
        assert data["source_robot_type"] == source
        assert data["orientation_source"] == "robot_fk"
        assert data["orientation_tracking_enabled"]
        assert data["human_position_preprocessing"] == "robot_segment_lengths_v1"
        assert "human_position_scale" not in data
        np.testing.assert_array_equal(data["body_proportion_source_contact_modes"], data["foot_contact_modes"])
        assert data["interaction_mesh_weight"] > 0
        assert data["orientation_frame_costs"].shape == (4,)
        assert np.isfinite(data["frame_costs"]).all()
        np.testing.assert_array_equal(data["sqp_converged"], data["sqp_stop_reasons"] == "converged")
        np.testing.assert_array_equal(
            data["sqp_iteration_limits"],
            [command.sqp.initial_max_iterations] + [command.sqp.max_iterations] * 3,
        )
        assert (data["sqp_iteration_counts"] <= data["sqp_iteration_limits"]).all()
        assert data["foot_contact_modes"].shape == (4, 2)
        assert data["human_points_world"].shape == data["robot_points_world"].shape
        assert any("mesh" in name for name in data.files)
        source_reference = Rotation.from_quat(data["orientation_reference_human_quaternions_wxyz"][:, [1, 2, 3, 0]])
        alignment = Rotation.from_quat(data["orientation_alignment_quaternions_wxyz"][:, [1, 2, 3, 0]])
        target_reference = Rotation.from_quat(data["orientation_reference_robot_quaternions_wxyz"][:, [1, 2, 3, 0]])
        np.testing.assert_allclose((source_reference * alignment).as_matrix(), target_reference.as_matrix(), atol=2e-7)
    np.testing.assert_allclose(target_robot.read_csv(result.csv_path), qpos, atol=1e-10)
    loaded = load_variant_result("identity", result.job.output_path)
    assert loaded.robot_urdf == str(target_robot.urdf)
    assert loaded.interaction_mesh is not None
    assert main(command).job.resumed
    result.csv_path.unlink()
    assert main(command).job.resumed
    assert result.csv_path.is_file()
    with pytest.raises(ValueError, match="different solver configuration"):
        main(replace(command, fps=30))
    with pytest.raises(ValueError, match="different solver configuration"):
        main(replace(command, position_scaling="uniform"))
    with pytest.raises(ValueError, match="different solver configuration"):
        main(replace(command, sqp=SqpConfig(max_iterations=200)))


def test_uniform_fallback_and_hard_cap_are_recorded(tmp_path: Path) -> None:
    csv, _ = write_motion(tmp_path, "g1")
    command = replace(
        command_for(csv, "g1", "e2", tmp_path),
        position_scaling="uniform",
        sqp=SqpConfig(max_iterations=1, initial_max_iterations=1),
    )
    with pytest.warns(RuntimeWarning, match="4/4 frames"):
        result = main(command)
    with np.load(result.job.output_path, allow_pickle=False) as data:
        assert data["human_position_preprocessing"] == "ground_align_and_uniform_scale"
        assert "human_position_scale" in data
        assert "body_proportion_target_lengths_m" not in data
        np.testing.assert_array_equal(data["sqp_iteration_counts"], np.ones(4, dtype=int))
        assert not data["sqp_converged"].any()
        assert (data["sqp_stop_reasons"] == "max_iterations").all()


def test_custom_robot_name_uses_profiles_without_pair_registration(tmp_path: Path) -> None:
    csv, _ = write_motion(tmp_path, "g1")
    profile = load_robot_profile(robot="e2")
    custom = json.loads(profile.path.read_text())
    custom["robot"] = "my_robot"
    custom["orientation"]["datasets"] = {}
    # Rename every body and actuated joint to prove the solver uses profile semantics,
    # including its optional shoulder chains, rather than known robot joint names.
    original = kinematics("e2")
    urdf_tree = ET.parse(original.urdf)
    link_names = {link.attrib["name"] for link in urdf_tree.getroot().findall("link")}
    renamed_links = {name: f"custom_{name}" for name in link_names}
    renamed_joints = {name: f"custom_{name}" for name in original.joint_addresses}
    for node in urdf_tree.iter():
        if node.tag == "link":
            node.set("name", renamed_links[node.attrib["name"]])
        elif node.tag in {"parent", "child"}:
            node.set("link", renamed_links[node.attrib["link"]])
        elif node.tag == "joint" and node.attrib["name"] in renamed_joints:
            node.set("name", renamed_joints[node.attrib["name"]])
        elif node.tag == "mesh":
            node.set("filename", str((original.urdf.parent / node.attrib["filename"]).resolve()))
    custom_urdf = tmp_path / "custom.urdf"
    urdf_tree.write(custom_urdf)
    xml_tree = ET.parse(original.urdf.with_suffix(".xml"))
    for node in xml_tree.iter():
        if node.tag == "compiler":
            node.set("meshdir", str((original.urdf.parent / node.get("meshdir", ".")).resolve()))
        if node.tag == "body" and node.get("name") in renamed_links:
            node.set("name", renamed_links[node.attrib["name"]])
        if node.tag == "joint" and node.get("name") in renamed_joints:
            node.set("name", renamed_joints[node.attrib["name"]])
        if node.get("joint") in renamed_joints:
            node.set("joint", renamed_joints[node.attrib["joint"]])
    xml_tree.write(custom_urdf.with_suffix(".xml"))
    custom["robot_urdf_file"] = str(custom_urdf)
    motion_profile = custom["robot_motion"]
    motion_profile["position_scaling"]["torso_link"] = renamed_links[motion_profile["position_scaling"]["torso_link"]]
    for key in ("position_links", "orientation_links"):
        motion_profile[key] = {name: renamed_links[link] for name, link in motion_profile[key].items()}
    motion_profile["reference_joint_positions"] = {
        renamed_joints[name]: value for name, value in motion_profile["reference_joint_positions"].items()
    }
    for key in ("reference_overrides", "weight_overrides"):
        custom["natural_pose"][key] = {
            renamed_joints[name]: value for name, value in custom["natural_pose"][key].items()
        }
    custom["natural_pose"]["joint_names"] = [renamed_joints[n] for n in custom["natural_pose"]["joint_names"]]
    robot_config = motion_profile["robot_config"]
    robot_config["foot_sticking_links"] = [renamed_links[n] for n in robot_config["foot_sticking_links"]]
    robot_config["foot_contact_links"] = {
        side: {region: renamed_links[link] for region, link in regions.items()}
        for side, regions in robot_config["foot_contact_links"].items()
    }
    robot_config["shoulder_direction_chains"] = [
        {
            "shoulder_joint_names": [
                renamed_joints[f"{side}_arm_shoulder_{axis}_joint"] for axis in ("pitch", "roll", "yaw")
            ],
            "torso_basis_link_name": renamed_links[f"{side}_arm_shoulder_pitch_link"],
            "shoulder_anchor_link_name": renamed_links[f"{side}_arm_shoulder_roll_link"],
            "elbow_link_name": renamed_links[f"{side}_arm_elbow_link"],
            "hand_link_name": renamed_links[f"{side}_hand_sphere_link"],
            "elbow_joint_name": renamed_joints[f"{side}_arm_elbow_joint"],
            "wrist_joint_names": [],
        }
        for side in ("l", "r")
    ]
    path = tmp_path / "my_robot.json"
    path.write_text(json.dumps(custom))
    result = main(
        replace(command_for(csv, "g1", "my_robot", tmp_path), robot_profile=path, shoulder_direction_tracking=True)
    )
    loaded = load_variant_result("identity", result.job.output_path)
    metadata = variant_result_metadata(loaded)
    runtime = _resolve_runtime_config(ViserConfig(qpos_npz=result.job.output_path), metadata)
    assert runtime.robot_type == "my_robot"
    assert Path(runtime.robot_urdf).is_file()
    assert "my_robot" in result.job.output_path.parts
    # The same custom definition can be the source of a subsequent run.
    reverse = main(
        Robot2RobotCommand(
            source_robot="my_robot",
            source_robot_profile=path,
            robot="e2",
            input_path=result.csv_path,
            fps=50,
            save_dir=tmp_path / "reverse",
            retargeter=RetargeterRuntimeOptions(visualize=False),
        )
    )
    assert reverse.csv_path.is_file()


def test_cli_requires_explicit_source_target_and_timing() -> None:
    with pytest.raises(SystemExit):
        tyro.cli(Robot2RobotCommand, args=["--source-robot", "g1", "--robot", "e2", "--input-path", "test.csv"])
    command = tyro.cli(
        Robot2RobotCommand,
        args=[
            "--source-robot",
            "g1",
            "--robot",
            "e2",
            "--input-path",
            "test.csv",
            "--fps",
            "50",
            "--foot-sticking",
            "False",
            "--orientation-tracking",
            "True",
            "--retargeter.no-visualize",
        ],
    )
    assert command.foot_sticking is False
    assert command.orientation_tracking is True
    assert not command.retargeter.visualize
