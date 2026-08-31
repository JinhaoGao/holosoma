# ruff: noqa: CPY001

"""Pure helpers for direct upper-arm direction tracking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def normalize_vector(vector: np.ndarray, *, label: str) -> np.ndarray:
    """Return one finite unit vector and reject degenerate bone segments."""

    value = np.asarray(vector, dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError(f"{label} must be one finite three-vector")
    norm = float(np.linalg.norm(value))
    if norm <= 1e-8:
        raise ValueError(f"{label} must have non-zero length")
    return value / norm


def unit_direction_jacobian(
    segment: np.ndarray,
    segment_jacobian: np.ndarray,
) -> np.ndarray:
    """Differentiate ``segment / ||segment||`` analytically."""

    direction = normalize_vector(segment, label="segment")
    jacobian = np.asarray(segment_jacobian, dtype=np.float64)
    if jacobian.ndim != 2 or jacobian.shape[0] != 3:
        raise ValueError("segment_jacobian must have shape (3, n)")
    if not np.isfinite(jacobian).all():
        raise ValueError("segment_jacobian must be finite")
    projector = np.eye(3, dtype=np.float64) - np.outer(direction, direction)
    return projector @ jacobian / float(np.linalg.norm(segment))


def direction_error_angle(first: np.ndarray, second: np.ndarray) -> float:
    """Return the unsigned geodesic angle between two directions."""

    left = normalize_vector(first, label="first direction")
    right = normalize_vector(second, label="second direction")
    return float(np.arccos(np.clip(np.dot(left, right), -1.0, 1.0)))


@dataclass(frozen=True)
class ShoulderSideSpec:
    """Names and indices defining one supported robot upper-limb chain."""

    human_arm_name: str
    human_forearm_name: str
    human_hand_name: str
    shoulder_joint_names: tuple[str, str, str]
    torso_basis_link_name: str
    shoulder_anchor_link_name: str
    elbow_link_name: str
    hand_link_name: str
    elbow_joint_name: str
    wrist_joint_names: tuple[str, ...]
