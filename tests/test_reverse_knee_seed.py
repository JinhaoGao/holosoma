# ruff: noqa: CPY001

"""Check the reverse-knee seed against independently generated FK targets."""

from pathlib import Path

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from holosoma_retargeting.models.m5.m5fanqu.kinematic_seed import ROBOT_LINK_NAMES, seed_reverse_knee_pose

MODEL = (
    Path(__file__).resolve().parents[1]
    / "src/holosoma_retargeting/holosoma_retargeting/models/m5/m5fanqu/m5fanqu_17dof.xml"
)


@pytest.mark.parametrize("yaw", [0.0, 1.4])
def test_seed_recovers_reachable_leg_targets_and_preserves_other_joints(yaw):
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    target_qpos = model.qpos0.copy()
    target_qpos[:3] = [2.0, -3.0, 1.2]
    rotation = Rotation.from_euler("xyz", [0.02, 0.2, yaw])
    target_qpos[3:7] = rotation.as_quat()[[3, 0, 1, 2]]
    target_qpos[7:17] = [0.1, 0.08, -0.35, -0.9, 0.2, -0.1, -0.08, 0.5, -1.1, 0.3]
    data.qpos[:] = target_qpos
    mujoco.mj_forward(model, data)
    targets = np.asarray([data.body(name).xpos.copy() for name in ROBOT_LINK_NAMES])
    initial = model.qpos0.copy()
    initial[18] = 0.4
    original = initial.copy()
    seeded = seed_reverse_knee_pose(
        model, initial, targets, root_position=target_qpos[:3], root_matrix=rotation.as_matrix()
    )
    np.testing.assert_array_equal(initial, original)
    np.testing.assert_array_equal(seeded[17:], initial[17:])
    np.testing.assert_allclose(seeded[:7], target_qpos[:7], atol=1e-12)
    assert np.all(seeded[7:17] >= model.jnt_range[1:11, 0])
    assert np.all(seeded[7:17] <= model.jnt_range[1:11, 1])
    data.qpos[:] = seeded
    mujoco.mj_forward(model, data)
    actual = np.asarray([data.body(name).xpos for name in ROBOT_LINK_NAMES])
    np.testing.assert_allclose(actual, targets, atol=2e-5)


def test_seed_rejects_nonfinite_targets():
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    with pytest.raises(ValueError, match="finite"):
        seed_reverse_knee_pose(model, model.qpos0, np.full((6, 3), np.nan))


def test_seed_preserves_root_quaternion_hemisphere():
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    targets = np.asarray([data.body(name).xpos.copy() for name in ROBOT_LINK_NAMES])
    initial = model.qpos0.copy()
    initial[3:7] *= -1
    result = seed_reverse_knee_pose(model, initial, targets, root_matrix=np.eye(3))
    np.testing.assert_array_equal(result[3:7], initial[3:7])
