#!/usr/bin/env python3
# ruff: noqa: CPY001

"""Recompute and verify canonical orientation calibration tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from holosoma_retargeting.config_types.data_type import MotionDataConfig
from holosoma_retargeting.config_types.robot import RobotConfig
from holosoma_retargeting.orientation_calibration import (
    ORIENTATION_ALIGNMENT_QUATERNIONS_WXYZ,
)
from scipy.spatial.transform import Rotation

CALIBRATED_DATA_FORMATS = ("amass", "lafan", "gvhmr", "mocap")
CALIBRATED_ROBOTS = ("g1", "e1", "e2")
PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _robot_model_path(robot_type: str) -> Path:
    relative_path = Path(RobotConfig(robot_type=robot_type).ROBOT_URDF_FILE).with_suffix(".xml")
    return PACKAGE_ROOT / relative_path


def _canonical_wxyz(matrix: np.ndarray) -> tuple[float, float, float, float]:
    quaternion = Rotation.from_matrix(matrix).as_quat(scalar_first=True)
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    quaternion[np.abs(quaternion) < 5e-15] = 0.0
    return (
        float(quaternion[0]),
        float(quaternion[1]),
        float(quaternion[2]),
        float(quaternion[3]),
    )


def robot_t_pose_qpos(model: mujoco.MjModel, motion: MotionDataConfig) -> np.ndarray:
    """Build the configured physical robot T-pose in MuJoCo qpos order."""

    qpos = model.qpos0.copy()
    robot_base = motion.resolved_orientation_t_pose_robot_base_quaternion_wxyz
    if robot_base is None:
        raise ValueError(f"Missing robot T-pose base for {motion.data_format}/{motion.robot_type}")
    qpos[3:7] = robot_base
    for joint_name, joint_position in motion.resolved_orientation_t_pose_robot_joint_positions.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Unknown robot T-pose joint {joint_name!r}")
        if model.jnt_limited[joint_id]:
            lower, upper = model.jnt_range[joint_id]
            if joint_position < lower or joint_position > upper:
                raise ValueError(f"Robot T-pose joint {joint_name!r}={joint_position} is outside [{lower}, {upper}]")
        qpos[int(model.jnt_qposadr[joint_id])] = joint_position
    return qpos


def compute_orientation_alignment_quaternions(
    data_format: str,
    robot_type: str,
) -> dict[str, tuple[float, float, float, float]]:
    """Compute ``inverse(R_human_ref) * R_robot_ref`` for one mapping."""

    motion = MotionDataConfig(data_format=data_format, robot_type=robot_type)
    mapping = motion.resolved_orientation_joints_mapping
    human_reference = motion.resolved_orientation_t_pose_human_quaternions_wxyz
    if set(mapping) != set(human_reference):
        raise ValueError(f"Human T-pose frames do not cover {data_format}/{robot_type}")

    model = mujoco.MjModel.from_xml_path(str(_robot_model_path(robot_type)))
    data = mujoco.MjData(model)
    data.qpos[:] = robot_t_pose_qpos(model, motion)
    mujoco.mj_forward(model, data)

    generated: dict[str, tuple[float, float, float, float]] = {}
    for human_joint, robot_link in mapping.items():
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, robot_link)
        if body_id < 0:
            raise ValueError(f"Unknown robot calibration link {robot_link!r}")
        human_matrix = Rotation.from_quat(
            human_reference[human_joint],
            scalar_first=True,
        ).as_matrix()
        robot_matrix = data.xmat[body_id].reshape(3, 3)
        generated[human_joint] = _canonical_wxyz(human_matrix.T @ robot_matrix)
    return generated


def maximum_saved_calibration_error_degrees(
    data_format: str,
    robot_type: str,
) -> float:
    """Return the maximum SO(3) error between generated and saved offsets."""

    key = (data_format, robot_type)
    generated = compute_orientation_alignment_quaternions(*key)
    saved = ORIENTATION_ALIGNMENT_QUATERNIONS_WXYZ[key]
    if set(generated) != set(saved):
        raise ValueError(f"Saved calibration joint set is stale for {data_format}/{robot_type}")
    generated_quaternions = np.asarray([generated[name] for name in generated])
    saved_quaternions = np.asarray([saved[name] for name in generated])
    generated_quaternions /= np.linalg.norm(generated_quaternions, axis=1, keepdims=True)
    saved_quaternions /= np.linalg.norm(saved_quaternions, axis=1, keepdims=True)
    dots = np.clip(np.abs(np.sum(generated_quaternions * saved_quaternions, axis=1)), 0.0, 1.0)
    return float(np.max(np.rad2deg(2.0 * np.arccos(dots))))


def _serializable_calibrations() -> dict[str, dict[str, tuple[float, float, float, float]]]:
    return {
        f"{data_format}:{robot_type}": compute_orientation_alignment_quaternions(
            data_format,
            robot_type,
        )
        for data_format in CALIBRATED_DATA_FORMATS
        for robot_type in CALIBRATED_ROBOTS
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify that the checked-in calibration tables match MuJoCo FK",
    )
    args = parser.parse_args()
    if not args.check:
        print(json.dumps(_serializable_calibrations(), indent=2))
        return

    errors = {
        f"{data_format}:{robot_type}": maximum_saved_calibration_error_degrees(
            data_format,
            robot_type,
        )
        for data_format in CALIBRATED_DATA_FORMATS
        for robot_type in CALIBRATED_ROBOTS
    }
    maximum_error = max(errors.values())
    print(json.dumps({"maximum_error_degrees": maximum_error, "pairs": errors}, indent=2))
    if maximum_error > 1e-5:
        raise SystemExit("Saved orientation calibrations do not match current robot FK")


if __name__ == "__main__":
    main()
