# ruff: noqa: CPY001

"""Bounded leg initialization for the reverse-knee MIA V5 mechanism."""

from __future__ import annotations

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

HUMAN_JOINT_NAMES = ("LeftLeg", "LeftFoot", "LeftToeBase", "RightLeg", "RightFoot", "RightToeBase")
ROBOT_LINK_NAMES = tuple(
    f"{side}_{link}" for side in ("l", "r") for link in ("leg_knee_link", "leg_ankle_pitch_link", "foot_sphere_5_link")
)


def seed_reverse_knee_pose(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    targets: np.ndarray,
    *,
    root_position: np.ndarray | None = None,
    root_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """Fit knee, ankle and toe positions before the constrained SQP solve.

    Fast swing transitions can trap the previous-frame initialization in a
    folded leg configuration. This bounded seed follows the current root and
    reconstructed leg targets; the ordinary SQP still enforces ground contact,
    joint limits, orientation objectives and temporal regularization.
    """
    targets = np.asarray(targets, dtype=np.float64)
    if targets.shape != (6, 3) or not np.isfinite(targets).all():
        raise ValueError("Reverse-knee targets must be a finite (6, 3) array")
    result = np.asarray(qpos, dtype=np.float64).copy()
    if result.shape != (model.nq,) or not np.isfinite(result).all():
        raise ValueError("Reverse-knee initialization requires a finite model-sized qpos")
    if root_position is not None:
        result[:3] = root_position
    if root_matrix is not None:
        quaternion = Rotation.from_matrix(root_matrix).as_quat()[[3, 0, 1, 2]]
        if np.dot(quaternion, result[3:7]) < 0.0:
            quaternion = -quaternion
        result[3:7] = quaternion
    joint_ids = np.asarray(
        [
            model.joint(f"{side}_leg_{joint}_joint").id
            for side in ("l", "r")
            for joint in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch")
        ]
    )
    addresses = model.jnt_qposadr[joint_ids]
    lower, upper = model.jnt_range[joint_ids].T
    body_ids = np.asarray([model.body(name).id for name in ROBOT_LINK_NAMES])
    data = mujoco.MjData(model)
    data.qpos[:] = result

    def residual(angles: np.ndarray) -> np.ndarray:
        data.qpos[addresses] = angles
        mujoco.mj_forward(model, data)
        return (data.xpos[body_ids] - targets).ravel()

    fit = least_squares(
        residual,
        np.clip(result[addresses], lower + 1e-8, upper - 1e-8),
        bounds=(lower, upper),
        max_nfev=80,
        ftol=1e-6,
        xtol=1e-6,
        gtol=1e-6,
    )
    result[addresses] = fit.x
    return result
