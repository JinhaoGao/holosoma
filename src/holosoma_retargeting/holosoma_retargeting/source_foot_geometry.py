# ruff: noqa: CPY001

"""Source foot geometry used for body proportion reconstruction."""

from __future__ import annotations

import numpy as np


def _foot_proxy_trajectories(
    joints: np.ndarray,
    *,
    toe_index: int,
    heel_joint_index: int,
    floor_height: float,
    ground_clearance: float,
) -> tuple[np.ndarray, np.ndarray]:
    toe = joints[:, toe_index].astype(np.float64, copy=True)
    ankle = joints[:, heel_joint_index].astype(np.float64, copy=True)
    near_floor = toe[:, 2] <= floor_height + ground_clearance
    vertical_offsets = ankle[near_floor, 2] - toe[near_floor, 2]
    vertical_offsets = vertical_offsets[np.isfinite(vertical_offsets) & (vertical_offsets > 0.0)]
    if len(vertical_offsets):
        sole_drop = float(np.median(vertical_offsets))
    else:
        segment_lengths = np.linalg.norm(toe - ankle, axis=1)
        sole_drop = 0.35 * float(np.median(segment_lengths))
    sole_drop = float(np.clip(sole_drop, 0.015, 0.12))

    heel = ankle
    heel[:, 2] -= sole_drop
    return heel, toe


def source_floor_clearance(
    joints: np.ndarray,
    joint_names: tuple[str, ...] | list[str],
    joint_parent_indices: tuple[int, ...] | np.ndarray,
    toe_names: tuple[str, str] | list[str],
    *,
    ground_clearance: float,
) -> np.ndarray:
    """Estimate whole-foot clearance using ankle-derived heel proxies.

    A raised toe with a planted heel is not a flight phase. Return the lowest
    heel/toe height above the source floor, clipped only after combining both
    feet; the caller can then remove support-phase height drift.
    """
    indices = [joint_names.index(name) for name in toe_names]
    floor = float(np.percentile(np.min(joints[:, indices, 2], axis=1), 2.0))
    points = []
    for index in indices:
        points.extend(
            _foot_proxy_trajectories(
                joints,
                toe_index=index,
                heel_joint_index=int(joint_parent_indices[index]),
                floor_height=floor,
                ground_clearance=ground_clearance,
            )
        )
    return np.maximum(np.min(np.stack(points, axis=1)[..., 2], axis=1) - floor, 0.0)
