# ruff: noqa: CPY001

"""Contract checks for the source-preserving reverse-knee experiment."""

from pathlib import Path

import mujoco
import numpy as np
import pytest

from holosoma_retargeting.config_types.retargeting import RetargetingCommand, internal_config_from_command
from holosoma_retargeting.retargeting_pipeline import normalize_retargeting_config


def test_experiment_keeps_other_targets_and_removes_both_knee_objectives():
    baseline = internal_config_from_command(
        RetargetingCommand(task="robot_only", robot="m5fanqu", dataset="fbx_mocap", preserve_human_knees=False),
    )
    experiment = normalize_retargeting_config(
        internal_config_from_command(
            RetargetingCommand(task="robot_only", robot="m5fanqu", dataset="fbx_mocap"),
        ),
    )
    knees = {"LeftLeg", "RightLeg"}
    original = baseline.motion_data_config.resolved_joints_mapping
    positions = experiment.motion_data_config.resolved_joints_mapping
    orientations = experiment.motion_data_config.resolved_orientation_joints_mapping
    assert not knees.intersection(positions)
    assert not knees.intersection(orientations)
    assert positions == {
        **{name: link for name, link in original.items() if name not in knees},
        "Spine1": "torso_tracking_link",
    }
    assert all(experiment.retargeter.orientation_weights[name] > 0 for name in positions)
    assert experiment.retargeter.orientation_weights["Spine1"] > experiment.retargeter.orientation_weights["LeftFoot"]
    assert not experiment.retargeter.shoulder_direction.enable
    assert experiment.retargeter.position_weights == {"Spine1": 1000.0}
    assert not experiment.retargeter.activate_foot_sticking
    assert not experiment.retargeter.foot_lock.enable
    assert experiment.preserve_human_knees
    assert not baseline.preserve_human_knees
    assert knees.issubset(original)


@pytest.mark.parametrize(("robot", "dataset"), [("e2", "fbx_mocap"), ("m5fanqu", "lafan")])
def test_experiment_rejects_unsupported_models_and_sources(robot, dataset):
    with pytest.raises(ValueError, match="requires robot_only/m5fanqu/fbx_mocap"):
        internal_config_from_command(
            RetargetingCommand(task="robot_only", robot=robot, dataset=dataset, preserve_human_knees=True),
        )


def test_virtual_chest_anchor_follows_shoulder_midpoint_without_new_dof():
    path = (
        Path(__file__).resolve().parents[1]
        / "src/holosoma_retargeting/holosoma_retargeting/models/m5/m5fanqu/m5fanqu_17dof.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    assert model.nq == 24 and model.nu == 17
    marker = model.body("torso_tracking_link")
    assert marker.mass == 0
    assert marker.jntnum == 0
    for waist_yaw in (0.0, 0.6, -0.9):
        data.qpos[:] = model.qpos0
        data.qpos[model.joint("waist_yaw_joint").qposadr] = waist_yaw
        data.qpos[:3] += [0.4, -0.3, 0.1]
        mujoco.mj_forward(model, data)
        center = (data.body("l_arm_shoulder_pitch_link").xpos + data.body("r_arm_shoulder_pitch_link").xpos) / 2
        np.testing.assert_allclose(data.body("torso_tracking_link").xpos, center, atol=1e-12)
