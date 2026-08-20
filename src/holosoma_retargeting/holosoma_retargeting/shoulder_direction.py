# ruff: noqa: CPY001

"""Pure helpers for sequence-aware shoulder direction branch selection."""

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


def deterministic_seed_fractions(count: int, dimensions: int) -> np.ndarray:
    """Generate deterministic low-discrepancy fractions without RNG state."""

    if count <= 0 or dimensions <= 0:
        raise ValueError("count and dimensions must be positive")
    irrational_steps = np.asarray(
        [0.6180339887498949, 0.4142135623730951, 0.7320508075688772],
        dtype=np.float64,
    )
    if dimensions > len(irrational_steps):
        raise ValueError("deterministic seeds support at most three dimensions")
    indices = np.arange(1, count + 1, dtype=np.float64)[:, None]
    return np.mod(indices * irrational_steps[None, :dimensions], 1.0)


def normalized_limit_barrier(
    values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Return a bounded scale-independent penalty near joint limits."""

    q = np.asarray(values, dtype=np.float64)
    lb = np.asarray(lower, dtype=np.float64)
    ub = np.asarray(upper, dtype=np.float64)
    if q.shape != lb.shape or q.shape != ub.shape:
        raise ValueError("values and limits must have matching shapes")
    span = ub - lb
    if np.any(span <= 0.0) or not np.isfinite(span).all():
        raise ValueError("joint limits must be finite and ordered")
    margin = np.minimum((q - lb) / span, (ub - q) / span)
    margin = np.clip(margin, 0.0, 0.5)
    return float(np.mean(np.square(0.1 / (margin + 0.05))))


def second_order_candidate_path(
    states: np.ndarray,
    node_costs: np.ndarray,
    valid: np.ndarray,
    *,
    joint_ranges: np.ndarray,
    velocity_weight: float,
    acceleration_weight: float,
    max_joint_step: np.ndarray | None = None,
) -> np.ndarray:
    """Select a globally continuous candidate path with second-order DP."""

    q = np.asarray(states, dtype=np.float64)
    costs = np.asarray(node_costs, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    ranges = np.asarray(joint_ranges, dtype=np.float64)
    if q.ndim != 3:
        raise ValueError("states must have shape (frames, candidates, joints)")
    frame_count, candidate_count, joint_count = q.shape
    if costs.shape != (frame_count, candidate_count) or mask.shape != costs.shape:
        raise ValueError("node_costs and valid must match the first two state dimensions")
    if ranges.shape != (joint_count,) or np.any(ranges <= 0.0):
        raise ValueError("joint_ranges must be positive with one value per joint")
    if frame_count == 0 or candidate_count == 0:
        raise ValueError("candidate graph must be non-empty")
    if velocity_weight < 0.0 or acceleration_weight < 0.0:
        raise ValueError("temporal weights must be non-negative")
    step_limit = None
    if max_joint_step is not None:
        step_limit = np.asarray(max_joint_step, dtype=np.float64)
        if step_limit.shape != (joint_count,) or np.any(step_limit <= 0.0):
            raise ValueError("max_joint_step must be positive with one value per joint")
    if np.any(np.sum(mask, axis=1) == 0):
        raise ValueError("every frame must contain a valid candidate")

    normalized = q / ranges[None, None, :]
    infinite = np.inf
    if frame_count == 1:
        first_costs = np.where(mask[0], costs[0], infinite)
        return np.asarray([int(np.argmin(first_costs))], dtype=np.int32)

    pair_cost = np.full((candidate_count, candidate_count), infinite, dtype=np.float64)
    delta01 = normalized[1, :, None, :] - normalized[0, None, :, :]
    velocity01 = np.sum(np.square(delta01), axis=-1)
    pair_cost = costs[0, None, :] + costs[1, :, None] + velocity_weight * velocity01
    pair_cost = pair_cost.T
    pair_valid = mask[0, :, None] & mask[1, None, :]
    if step_limit is not None:
        raw_delta01 = q[1, :, None, :] - q[0, None, :, :]
        pair_valid &= np.all(np.abs(raw_delta01) <= step_limit, axis=-1).T
    pair_cost = np.where(pair_valid, pair_cost, infinite)
    backpointers: list[np.ndarray] = []

    for frame in range(2, frame_count):
        next_pair = np.full_like(pair_cost, infinite)
        backpointer = np.full(
            (candidate_count, candidate_count),
            -1,
            dtype=np.int32,
        )
        for previous in range(candidate_count):
            if not mask[frame - 1, previous]:
                continue
            previous_delta = normalized[frame - 1, previous][None, :] - normalized[frame - 2]
            current_delta = normalized[frame] - normalized[frame - 1, previous]
            velocity_cost = velocity_weight * np.sum(
                np.square(current_delta),
                axis=-1,
            )
            acceleration_cost = acceleration_weight * np.sum(
                np.square(
                    current_delta[None, :, :] - previous_delta[:, None, :],
                ),
                axis=-1,
            )
            transition = (
                pair_cost[:, previous, None] + acceleration_cost + velocity_cost[None, :] + costs[frame][None, :]
            )
            transition[:, ~mask[frame]] = infinite
            if step_limit is not None:
                raw_current_delta = q[frame] - q[frame - 1, previous]
                transition[:, np.any(np.abs(raw_current_delta) > step_limit, axis=-1)] = infinite
            best_earlier = np.argmin(transition, axis=0)
            best_cost = transition[best_earlier, np.arange(candidate_count)]
            next_pair[previous] = best_cost
            backpointer[previous] = best_earlier.astype(np.int32)
        pair_cost = next_pair
        backpointers.append(backpointer)

    final_previous, final_current = np.unravel_index(
        int(np.argmin(pair_cost)),
        pair_cost.shape,
    )
    if not np.isfinite(pair_cost[final_previous, final_current]):
        raise RuntimeError(
            "Shoulder candidate graph has no path within the configured joint-step limit",
        )
    selected = np.full(frame_count, -1, dtype=np.int32)
    selected[-2] = int(final_previous)
    selected[-1] = int(final_current)
    for frame in range(frame_count - 1, 1, -1):
        pointer = backpointers[frame - 2]
        selected[frame - 2] = pointer[selected[frame - 1], selected[frame]]
    if np.any(selected < 0):
        raise RuntimeError("second-order candidate path reconstruction failed")
    return selected


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
    wrist_joint_name: str | None
