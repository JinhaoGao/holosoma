# ruff: noqa: CPY001, S314

"""Independent dimension, direction, and world-frame checks for M5C targets."""

import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest
from holosoma_retargeting.body_proportions import rebuild_m5_body
from holosoma_retargeting.config_types.data_type import MotionDataConfig
from holosoma_retargeting.data_utils.motion_data import HumanMotion
from scipy.spatial.transform import Rotation

PACKAGE = Path(__file__).resolve().parents[1] / "src/holosoma_retargeting/holosoma_retargeting"


@pytest.fixture(params=["m4g", "m5c", "m5d", "m5e"])
def robot(request):
    return request.param


@pytest.fixture
def model(robot):
    directory = f"models/{robot}" if robot == "m4g" else f"models/m5/{robot}"
    return mujoco.MjModel.from_xml_path(str(PACKAGE / f"{directory}/{robot}_17dof.xml"))


@pytest.fixture
def motion(robot):
    config = MotionDataConfig(robot_type=robot, data_format="fbx_mocap")
    names = tuple(config.resolved_demo_joints)
    parents = tuple(config.resolved_joint_parent_indices)
    points = {"Hips": np.array([0.0, 0, 1.0]), "Spine": np.array([0.0, 0, 1.1]), "Spine1": np.array([0.0, 0, 1.3])}
    for side, sign in (("Left", 1), ("Right", -1)):
        points[side + "UpLeg"] = points["Hips"] + [0, sign * 0.1, -0.05]
        points[side + "Leg"] = points[side + "UpLeg"] + [0.1, sign * 0.02, -0.4]
        points[side + "Foot"] = points[side + "Leg"] + [-0.1, 0, -0.35]
        points[side + "ToeBase"] = points[side + "Foot"] + [0.13, 0, -0.05]
        points[side + "Shoulder"] = points["Spine1"] + [0, sign * 0.1, 0]
        points[side + "Arm"] = points["Spine1"] + [0, sign * 0.16, 0]
        points[side + "ForeArm"] = points[side + "Arm"] + [0, sign * 0.1, -0.25]
        points[side + "Hand"] = points[side + "ForeArm"] + [0.05, 0, -0.20]
    joints = np.empty((5, len(names), 3))
    for i, name in enumerate(names):
        joints[:, i] = points[name] if name in points else joints[:, parents[i]] + [0.01, 0, 0.01]
    joints[..., 0] += np.linspace(0, 0.4, 5)[:, None]
    joints[..., 2] += np.array([0, 0.02, 0.1, 0.02, 0])[:, None]
    orientation_names = ("Hips", "LeftFoot", "RightFoot")
    return HumanMotion(
        joints=joints,
        fps=30,
        human_height=1.5,
        source_path=Path("synthetic.npz"),
        orientation_joint_names=orientation_names,
        orientation_quaternions_wxyz=np.tile([1.0, 0, 0, 0], (5, 3, 1)),
        t_pose_orientation_joint_names=orientation_names,
        t_pose_orientation_quaternions_wxyz=np.tile([1.0, 0, 0, 0], (3, 1)),
        root_frame_to_robot_base_quaternion_wxyz=np.array([1.0, 0, 0, 0]),
        joint_parent_indices=parents,
        _canonical_joint_names=names,
        _root_joint_name="Hips",
    )


def test_bone_lengths_match_original_urdf_and_preserve_source_directions(model, motion, robot):
    body = rebuild_m5_body(motion, model, robot_type=robot, orientation_reference_qpos=model.qpos0)
    package = "__00-mia-v4_2-G" if robot == "m4g" else f"__00-mia-v5-0{robot[-1].upper()}"
    directory = f"models/{robot}" if robot == "m4g" else f"models/m5/{robot}"
    urdf = ET.parse(PACKAGE / f"{directory}/{package}/urdf/update_{package}.urdf")
    origins = {
        joint.get("name"): np.fromstring(joint.find("origin").get("xyz"), sep=" ") for joint in urdf.findall("joint")
    }
    assert body.metadata["human_position_preprocessing"] == f"{robot}_bone_lengths_v2"
    names = motion.canonical_joint_names
    for side, prefix in (("Left", "l"), ("Right", "r")):
        for start, end, expected in (
            ("UpLeg", "Leg", np.linalg.norm(origins[prefix + "_leg_knee_joint"])),
            ("Leg", "Foot", np.linalg.norm(origins[prefix + "_leg_ankle_pitch_joint"])),
            ("Arm", "ForeArm", np.linalg.norm(origins[prefix + "_arm_elbow_joint"])),
            ("ForeArm", "Hand", np.linalg.norm(model.body(prefix + "_hand_sphere_link").pos)),
        ):
            a, b = names.index(side + start), names.index(side + end)
            before, after = motion.joints[:, b] - motion.joints[:, a], body.joints[:, b] - body.joints[:, a]
            np.testing.assert_allclose(np.linalg.norm(after, axis=-1), expected, atol=1e-12)
            np.testing.assert_allclose(
                after / expected, before / np.linalg.norm(before, axis=-1, keepdims=True), atol=1e-12
            )
    pelvis_width = np.linalg.norm(origins["l_leg_hip_yaw_joint"] - origins["r_leg_hip_yaw_joint"])
    shoulder_width = np.linalg.norm(
        origins["l_arm_shoulder_pitch_joint"]
        + origins["l_arm_shoulder_roll_joint"]
        - origins["r_arm_shoulder_pitch_joint"]
        - origins["r_arm_shoulder_roll_joint"]
    )
    for suffix, expected in (("UpLeg", pelvis_width), ("Arm", shoulder_width)):
        difference = body.joints[:, names.index("Left" + suffix)] - body.joints[:, names.index("Right" + suffix)]
        np.testing.assert_allclose(np.linalg.norm(difference, axis=-1), expected, atol=1e-12)
    scale = body.metadata["root_translation_scale"]
    np.testing.assert_allclose(np.diff(body.joints[:, 0, :2], axis=0), np.diff(motion.joints[:, 0, :2], axis=0) * scale)
    # Actual sole geometry, rather than the ankle or only the toe, determines
    # floor clearance. The middle airborne frame must remain airborne.
    np.testing.assert_allclose(
        body.metadata["body_proportion_target_sole_points"][..., 2].min(axis=(1, 2)),
        np.array([0, 0.02, 0.1, 0.02, 0]) * scale,
        atol=1e-12,
    )


def test_reconstruction_is_equivariant_under_world_yaw_and_translation(model, motion, robot):
    baseline = rebuild_m5_body(motion, model, robot_type=robot, orientation_reference_qpos=model.qpos0)
    rotation = Rotation.from_euler("z", 1.2)
    translation = np.array([2.0, -3.0, 0.0])
    quats = (
        (rotation * Rotation.from_quat(motion.orientation_quaternions_wxyz.reshape(-1, 4)[:, [1, 2, 3, 0]]))
        .as_quat()[:, [3, 0, 1, 2]]
        .reshape(5, 3, 4)
    )
    rotated = replace(
        motion, joints=motion.joints @ rotation.as_matrix().T + translation, orientation_quaternions_wxyz=quats
    )
    actual = rebuild_m5_body(rotated, model, robot_type=robot, orientation_reference_qpos=model.qpos0)
    np.testing.assert_allclose(
        actual.joints,
        baseline.joints @ rotation.as_matrix().T + translation * baseline.metadata["root_translation_scale"],
        atol=1e-12,
    )


def test_raised_toes_do_not_raise_a_planted_heel(model, motion, robot):
    joints = np.tile(motion.joints[:1], (5, 1, 1))
    for side in ("Left", "Right"):
        joints[:, motion.canonical_joint_names.index(side + "ToeBase"), 2] += [0, 0.02, 0.10, 0.02, 0]
    plan = np.ones((5, 2), dtype=bool)
    body = rebuild_m5_body(
        replace(motion, joints=joints),
        model,
        robot_type=robot,
        orientation_reference_qpos=model.qpos0,
        source_sticking_states=plan,
    )
    np.testing.assert_allclose(body.metadata["body_proportion_target_clearance_m"], 0, atol=1e-12)


def test_support_drift_is_removed_without_erasing_an_airborne_arch(model, motion, robot):
    joints = motion.joints.copy()
    joints[..., 2] += np.linspace(0, 0.08, len(joints))[:, None]
    plan = np.zeros((5, 2), dtype=bool)
    plan[[0, 4]] = True
    body = rebuild_m5_body(
        replace(motion, joints=joints),
        model,
        robot_type=robot,
        orientation_reference_qpos=model.qpos0,
        source_sticking_states=plan,
    )
    clearance = body.metadata["body_proportion_target_clearance_m"]
    np.testing.assert_allclose(clearance[[0, 4]], 0, atol=1e-12)
    assert clearance[2] > 0.08 * body.metadata["root_translation_scale"]
    without_contact = rebuild_m5_body(
        motion=replace(motion, joints=joints), model=model, robot_type=robot, orientation_reference_qpos=model.qpos0
    )
    assert not without_contact.metadata["body_proportion_contact_phase_height_enabled"]
    assert without_contact.metadata["body_proportion_target_clearance_m"][-1] > 0.05
    np.testing.assert_allclose(body.joints[..., :2], without_contact.joints[..., :2])


def test_zero_length_source_bone_is_rejected(model, motion, robot):
    joints = motion.joints.copy()
    names = motion.canonical_joint_names
    joints[:, names.index("LeftLeg")] = joints[:, names.index("LeftUpLeg")]
    with pytest.raises(ValueError, match="zero-length"):
        rebuild_m5_body(replace(motion, joints=joints), model, robot_type=robot, orientation_reference_qpos=model.qpos0)


def test_sole_geometry_uses_the_same_robot_reference_frame_as_orientation_targets(model, motion, robot):
    reference = model.qpos0.copy()
    reference[3:7] = Rotation.from_euler("z", -np.pi / 2).as_quat()[[3, 0, 1, 2]]
    body = rebuild_m5_body(motion, model, robot_type=robot, orientation_reference_qpos=reference)
    data = mujoco.MjData(model)
    data.qpos[:] = reference
    mujoco.mj_forward(model, data)
    for side_index, (side, prefix) in enumerate((("Left", "l"), ("Right", "r"))):
        ankle = body.joints[:, motion.canonical_joint_names.index(side + "Foot")]
        for point_index in range(5):
            offset = (
                data.body(prefix + f"_foot_sphere_{point_index + 1}_link").xpos
                - data.body(prefix + "_leg_ankle_pitch_link").xpos
            )
            np.testing.assert_allclose(
                body.metadata["body_proportion_target_sole_points"][:, side_index, point_index],
                ankle + offset,
                atol=1e-12,
            )


@pytest.mark.parametrize("robot", ["m5fanqu"])
def test_reverse_knee_preserves_endpoints_lengths_and_world_equivariance(model, motion, robot):
    reference = rebuild_m5_body(motion, model, robot_type="m5c", orientation_reference_qpos=model.qpos0)
    reverse = rebuild_m5_body(motion, model, robot_type=robot, orientation_reference_qpos=model.qpos0)
    assert reverse.metadata["human_position_preprocessing"] == "m5fanqu_reverse_knee_v1"
    names = motion.canonical_joint_names
    for side in ("Left", "Right"):
        hip, knee, ankle = [names.index(side + suffix) for suffix in ("UpLeg", "Leg", "Foot")]
        np.testing.assert_allclose(reverse.joints[:, [hip, ankle]], reference.joints[:, [hip, ankle]], atol=1e-12)
        for a, b in ((hip, knee), (knee, ankle)):
            np.testing.assert_allclose(
                np.linalg.norm(reverse.joints[:, b] - reverse.joints[:, a], axis=-1),
                np.linalg.norm(reference.joints[:, b] - reference.joints[:, a], axis=-1),
                atol=1e-12,
            )
        chord = reference.joints[:, ankle] - reference.joints[:, hip]
        forward = np.cross(np.array([0.0, 1.0, 0.0]), chord)
        before = np.sum((reference.joints[:, knee] - reference.joints[:, hip]) * forward, axis=-1)
        after = np.sum((reverse.joints[:, knee] - reverse.joints[:, hip]) * forward, axis=-1)
        assert np.all(np.abs(before) > 0.01)
        np.testing.assert_allclose(after, -before, atol=1e-12)
    np.testing.assert_allclose(reverse.sole_centers, reference.sole_centers, atol=1e-12)
    rotation = Rotation.from_euler("z", 1.2)
    translation = np.array([2.0, -3.0, 0.0])
    quaternions = (
        (rotation * Rotation.from_quat(motion.orientation_quaternions_wxyz.reshape(-1, 4)[:, [1, 2, 3, 0]]))
        .as_quat()[:, [3, 0, 1, 2]]
        .reshape(motion.orientation_quaternions_wxyz.shape)
    )
    rotated = replace(
        motion,
        joints=motion.joints @ rotation.as_matrix().T + translation,
        orientation_quaternions_wxyz=quaternions,
    )
    result = rebuild_m5_body(rotated, model, robot_type=robot, orientation_reference_qpos=model.qpos0)
    scale = reverse.metadata["root_translation_scale"]
    np.testing.assert_allclose(result.joints, reverse.joints @ rotation.as_matrix().T + translation * scale, atol=1e-12)
