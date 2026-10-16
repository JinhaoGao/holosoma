# ruff: noqa: CPY001
"""Validation shared by the public visualization entry points."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np
import yourdfpy

from holosoma_retargeting.visualization.result_loader import resolve_qpos_to_viser_joint_indices


def validate_playback_options(port: int, fps: float | None, interpolation: int = 1) -> None:
    """Reject invalid playback options before starting a server."""
    if not 1 <= port <= 65535:
        raise ValueError("Viser port must be between 1 and 65535")
    if fps is not None and (not np.isfinite(fps) or fps <= 0):
        raise ValueError("Playback FPS must be positive and finite")
    if interpolation < 1:
        raise ValueError("visual_fps_multiplier must be positive")


@lru_cache(maxsize=32)
def _model_joint_names(urdf: Path, xml: Path | None) -> tuple[str, ...]:
    robot = yourdfpy.URDF.load(str(urdf), load_meshes=True, build_scene_graph=True)
    if xml is not None:
        mujoco.MjModel.from_xml_path(str(xml))
    return tuple(robot.actuated_joint_names)


def validate_robot_motion(
    path: str | Path,
    qpos: np.ndarray,
    saved_joint_names: tuple[str, ...],
    urdf: Path,
    xml: Path | None,
    *,
    contains_object: bool | None = None,
) -> None:
    """Check model assets, joint correspondence and the robot/object pose layout."""
    names = _model_joint_names(urdf, xml)
    resolve_qpos_to_viser_joint_indices(
        result_path=path,
        saved_joint_names=saved_joint_names,
        viser_joint_names=names,
        fallback_mujoco_xml=xml,
    )
    widths = (7 + len(names), 14 + len(names))
    allowed = widths if contains_object is None else (widths[int(contains_object)],)
    if qpos.ndim != 2 or not len(qpos) or qpos.shape[1] not in allowed or not np.isfinite(qpos).all():
        raise ValueError(f"{path}: expected finite robot motion with columns {allowed}, got {qpos.shape}")
    norms = np.linalg.norm(qpos[:, 3:7], axis=1)
    if np.any(norms < 1e-8):
        raise ValueError(f"{path}: root quaternion must be nonzero")
