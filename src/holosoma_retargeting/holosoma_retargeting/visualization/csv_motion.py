# ruff: noqa: CPY001
"""Read robot trajectories exported by data_utils.npz_to_csv."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from holosoma_retargeting.config_types.viser import ViserConfig
from holosoma_retargeting.data_utils.npz_to_csv import actuated_joint_names_from_urdf
from holosoma_retargeting.visualization.result_loader import resolve_robot_asset_paths


def load_csv_motion(config: ViserConfig) -> tuple[np.ndarray, float, None, dict[str, object], None]:
    """Return xyz+wxyz and named URDF-order joints, without human references."""
    source = config.input_path or config.qpos_npz
    if source is None:
        raise ValueError("CSV visualization requires --input-path.")
    if config.robot_urdf is None and config.robot_type is None:
        raise ValueError("CSV has no robot metadata; pass --robot-type or --robot-urdf.")
    urdf, xml = resolve_robot_asset_paths(
        robot_type=config.robot_type,
        robot_urdf=config.robot_urdf,
        robot_mujoco_xml=config.robot_mujoco_xml,
    )
    names = actuated_joint_names_from_urdf(urdf)
    if len(set(names)) != len(names):
        raise ValueError(f"URDF contains duplicate joint names: {urdf}")
    path = Path(source).expanduser()
    if not path.read_text().strip():
        raise ValueError(f"CSV motion is empty: {path}")
    try:
        values = np.loadtxt(path, delimiter=",", ndmin=2, comments=None)
    except ValueError as exc:
        raise ValueError(f"{path}: expected a headerless numeric CSV: {exc}") from exc
    expected_columns = 7 + len(names)
    if values.shape[0] == 0 or values.shape[1] != expected_columns:
        raise ValueError(
            f"{path}: expected {expected_columns} columns (xyz, xyzw, {len(names)} URDF joints), got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: CSV motion contains non-finite values")
    norms = np.linalg.norm(values[:, 3:7], axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3, rtol=0):
        raise ValueError(f"{path}: root xyzw quaternions must have unit length")
    qpos = values.copy()
    qpos[:, 3:7] = values[:, (6, 3, 4, 5)]
    metadata: dict[str, object] = {
        "robot_type": config.robot_type,
        "robot_urdf": str(urdf),
        "robot_mujoco_xml": str(xml) if xml else None,
        "robot_actuated_joint_names": names,
        "contains_object_in_qpos": False,
    }
    return qpos, config.fps if config.fps is not None else 30.0, None, metadata, None
