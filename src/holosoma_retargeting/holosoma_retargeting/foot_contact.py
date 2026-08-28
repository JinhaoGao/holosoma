# ruff: noqa: CPY001

"""Contact-aware planar foot-state extraction for retargeting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np

from holosoma_retargeting.config_types.retargeter import PlanarFootContactConfig

FootContactMode = Literal["swing", "flat", "heel", "toe", "pivot", "slide"]
FOOT_CONTACT_SIDE_NAMES = ("left", "right")
_STICKING_MODES = frozenset({"flat", "heel", "toe", "pivot"})


@dataclass(frozen=True)
class FootContactState:
    """One foot's planar contact state at one source frame."""

    mode: FootContactMode
    phase_id: int = -1
    pivot_region: str = ""
    pivot_uv: tuple[float, float] = (0.5, 0.0)
    confidence: float = 0.0
    reference_displacement_xy: tuple[float, float] = (0.0, 0.0)

    @property
    def is_sticking(self) -> bool:
        """Whether the state represents a stationary planar contact."""
        return self.mode in _STICKING_MODES


@dataclass(frozen=True)
class FootContactPlan:
    """Per-frame contact states in canonical left/right order."""

    frames: tuple[dict[str, FootContactState], ...]

    def __post_init__(self) -> None:
        for frame_idx, frame in enumerate(self.frames):
            missing = [side for side in FOOT_CONTACT_SIDE_NAMES if side not in frame]
            if missing:
                raise ValueError(
                    f"Foot contact frame {frame_idx} is missing side(s): {', '.join(missing)}",
                )

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def modes(self) -> np.ndarray:
        """Return contact modes with shape ``[frames, left/right]``."""
        return np.asarray(
            [[frame[side].mode for side in FOOT_CONTACT_SIDE_NAMES] for frame in self.frames],
            dtype=str,
        )

    @property
    def phase_ids(self) -> np.ndarray:
        """Return contact phase identifiers with shape ``[frames, left/right]``."""
        return np.asarray(
            [[frame[side].phase_id for side in FOOT_CONTACT_SIDE_NAMES] for frame in self.frames],
            dtype=np.int32,
        )

    @property
    def confidences(self) -> np.ndarray:
        """Return detection confidence with shape ``[frames, left/right]``."""
        return np.asarray(
            [[frame[side].confidence for side in FOOT_CONTACT_SIDE_NAMES] for frame in self.frames],
            dtype=np.float32,
        )

    @property
    def pivot_regions(self) -> np.ndarray:
        """Return the active pivot region for each foot and frame."""
        return np.asarray(
            [[frame[side].pivot_region for side in FOOT_CONTACT_SIDE_NAMES] for frame in self.frames],
            dtype=str,
        )

    @property
    def pivot_uv(self) -> np.ndarray:
        """Return normalized sole coordinates for point contacts."""
        return np.asarray(
            [[frame[side].pivot_uv for side in FOOT_CONTACT_SIDE_NAMES] for frame in self.frames],
            dtype=np.float32,
        )

    @property
    def reference_displacements_xy(self) -> np.ndarray:
        """Return phase-relative target-world XY motion for intentional slides."""
        return np.asarray(
            [[frame[side].reference_displacement_xy for side in FOOT_CONTACT_SIDE_NAMES] for frame in self.frames],
            dtype=np.float32,
        )

    @property
    def sticking_states(self) -> np.ndarray:
        """Return the legacy boolean stationary-contact representation."""
        return np.asarray(
            [[frame[side].is_sticking for side in FOOT_CONTACT_SIDE_NAMES] for frame in self.frames],
            dtype=bool,
        )

    def legacy_sequences(self, toe_names: list[str] | tuple[str, str]) -> list[dict[str, bool]]:
        """Convert the plan to the historical named-toe boolean dictionaries."""
        if len(toe_names) != 2:
            raise ValueError("toe_names must contain exactly one left and one right name")
        return [
            {
                str(toe_names[0]): frame["left"].is_sticking,
                str(toe_names[1]): frame["right"].is_sticking,
            }
            for frame in self.frames
        ]

    @classmethod
    def from_sticking_sequences(
        cls,
        foot_sticking_sequences,
        *,
        num_frames: int,
    ) -> FootContactPlan:
        """Promote legacy per-frame booleans to flat-foot contact phases."""
        if len(foot_sticking_sequences) != num_frames:
            raise ValueError(
                "foot_sticking_sequences must contain one entry per motion frame; "
                f"got {len(foot_sticking_sequences)} entries for {num_frames} frames",
            )

        phase_counters = dict.fromkeys(FOOT_CONTACT_SIDE_NAMES, -1)
        previous = dict.fromkeys(FOOT_CONTACT_SIDE_NAMES, False)
        frames: list[dict[str, FootContactState]] = []
        for frame_idx, source_state in enumerate(foot_sticking_sequences):
            side_values: dict[str, bool] = {}
            for name, value in source_state.items():
                side = _name_side(str(name))
                if side is not None:
                    side_values[side] = bool(value)
            missing = [side for side in FOOT_CONTACT_SIDE_NAMES if side not in side_values]
            if missing:
                raise ValueError(
                    f"foot_sticking_sequences[{frame_idx}] is missing side(s): " + ", ".join(missing),
                )

            frame: dict[str, FootContactState] = {}
            for side in FOOT_CONTACT_SIDE_NAMES:
                active = side_values[side]
                if active and not previous[side]:
                    phase_counters[side] += 1
                frame[side] = FootContactState(
                    mode="flat" if active else "swing",
                    phase_id=phase_counters[side] if active else -1,
                    pivot_region="flat" if active else "",
                    confidence=1.0 if active else 0.0,
                )
                previous[side] = active
            frames.append(frame)
        return cls(tuple(frames))


def _name_side(name: str) -> str | None:
    normalized = name.lower().replace("-", "_")
    if normalized.startswith(("left", "l_")) or "_left" in normalized:
        return "left"
    if normalized.startswith(("right", "r_")) or "_right" in normalized:
        return "right"
    return None


def _validate_config(config: PlanarFootContactConfig) -> None:
    values = {
        "smoothing_window_seconds": config.smoothing_window_seconds,
        "static_speed": config.static_speed,
        "release_speed": config.release_speed,
        "normal_speed": config.normal_speed,
        "pivot_angular_speed": config.pivot_angular_speed,
        "slide_speed": config.slide_speed,
        "ground_clearance": config.ground_clearance,
        "minimum_phase_seconds": config.minimum_phase_seconds,
        "elevated_support_seconds": config.elevated_support_seconds,
        "heading_tolerance": config.heading_tolerance,
        "slide_tracking_tolerance": config.slide_tracking_tolerance,
    }
    invalid = {name: value for name, value in values.items() if not np.isfinite(value) or value < 0.0}
    if invalid:
        raise ValueError(f"Planar foot-contact parameters must be finite and non-negative: {invalid}")
    if config.release_speed < config.static_speed:
        raise ValueError("release_speed must be greater than or equal to static_speed")
    if config.slide_speed <= config.static_speed:
        raise ValueError("slide_speed must be greater than static_speed")


def _centered_smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) <= 2:
        return values.astype(np.float64, copy=True)
    window = min(window, len(values) if len(values) % 2 == 1 else len(values) - 1)
    window = max(window, 1)
    if window % 2 == 0:
        window -= 1
    if window <= 1:
        return values.astype(np.float64, copy=True)
    radius = window // 2
    padded = np.pad(values, ((radius, radius), (0, 0)), mode="edge")
    cumulative = np.vstack([np.zeros((1, values.shape[1])), np.cumsum(padded, axis=0)])
    return (cumulative[window:] - cumulative[:-window]) / float(window)


def _velocity(values: np.ndarray, fps: float) -> np.ndarray:
    if len(values) <= 1:
        return np.zeros_like(values, dtype=np.float64)
    return np.gradient(values, 1.0 / fps, axis=0, edge_order=1)


def _hysteresis_below(values: np.ndarray, enter: float, release: float) -> np.ndarray:
    active = False
    states = np.zeros(len(values), dtype=bool)
    for index, value in enumerate(values):
        if active:
            active = bool(value < release)
        else:
            active = bool(value <= enter)
        states[index] = active
    return states


def _retain_long_runs(mask: np.ndarray, minimum_frames: int) -> np.ndarray:
    retained = np.zeros_like(mask, dtype=bool)
    start = 0
    while start < len(mask):
        if not mask[start]:
            start += 1
            continue
        end = start + 1
        while end < len(mask) and mask[end]:
            end += 1
        if end - start >= minimum_frames:
            retained[start:end] = True
        start = end
    return retained


def _merge_short_mode_runs(modes: np.ndarray, minimum_frames: int) -> np.ndarray:
    merged = modes.astype(object, copy=True)
    if minimum_frames <= 1:
        return merged
    changed = True
    while changed:
        changed = False
        start = 0
        while start < len(merged):
            end = start + 1
            while end < len(merged) and merged[end] == merged[start]:
                end += 1
            if end - start < minimum_frames:
                left = merged[start - 1] if start > 0 else None
                right = merged[end] if end < len(merged) else None
                replacement = left if left == right else left or right
                if replacement is not None and replacement != merged[start]:
                    merged[start:end] = replacement
                    changed = True
                    break
            start = end
    return merged


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


def _points_on_elevated_support_surfaces(
    points: np.ndarray,
    support_triangles: np.ndarray | None,
    *,
    floor_height: float,
    clearance: float,
) -> np.ndarray:
    """Return where points lie on explicit upward-facing elevated surfaces."""
    supported = np.zeros(len(points), dtype=bool)
    if support_triangles is None:
        return supported

    triangles = np.asarray(support_triangles, dtype=np.float64)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise ValueError(
            f"elevated_support_triangles must have shape [triangles, 3, 3], got {triangles.shape}",
        )
    if not np.isfinite(triangles).all():
        raise ValueError("elevated_support_triangles must contain only finite vertices")
    if len(triangles) == 0:
        return supported

    edge_1 = triangles[:, 1] - triangles[:, 0]
    edge_2 = triangles[:, 2] - triangles[:, 0]
    normals = np.cross(edge_1, edge_2)
    normal_norms = np.linalg.norm(normals, axis=1)
    valid_normals = normal_norms > 1e-12
    normals[valid_normals] /= normal_norms[valid_normals, None]
    candidate_triangles = triangles[
        valid_normals & (normals[:, 2] >= 0.9) & (np.min(triangles[:, :, 2], axis=1) > floor_height + clearance)
    ]

    xy_tolerance = 1e-8
    for triangle in candidate_triangles:
        origin = triangle[0, :2]
        axis_u = triangle[1, :2] - origin
        axis_v = triangle[2, :2] - origin
        determinant = axis_u[0] * axis_v[1] - axis_u[1] * axis_v[0]
        if abs(determinant) <= 1e-12:
            continue
        relative = points[:, :2] - origin
        weight_u = (relative[:, 0] * axis_v[1] - relative[:, 1] * axis_v[0]) / determinant
        weight_v = (axis_u[0] * relative[:, 1] - axis_u[1] * relative[:, 0]) / determinant
        weight_origin = 1.0 - weight_u - weight_v
        inside = (weight_u >= -xy_tolerance) & (weight_v >= -xy_tolerance) & (weight_origin >= -xy_tolerance)
        surface_height = weight_origin * triangle[0, 2] + weight_u * triangle[1, 2] + weight_v * triangle[2, 2]
        supported |= inside & (np.abs(points[:, 2] - surface_height) <= clearance)
    return supported


def _classify_one_foot(
    heel: np.ndarray,
    toe: np.ndarray,
    *,
    floor_height: float,
    fps: float,
    config: PlanarFootContactConfig,
    elevated_support_triangles: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame_count = len(toe)
    smoothing_frames = max(1, round(config.smoothing_window_seconds * fps))
    if smoothing_frames % 2 == 0:
        smoothing_frames += 1
    heel_smooth = _centered_smooth(heel, smoothing_frames)
    toe_smooth = _centered_smooth(toe, smoothing_frames)
    heel_velocity = _velocity(heel_smooth, fps)
    toe_velocity = _velocity(toe_smooth, fps)
    heel_speed = np.linalg.norm(heel_velocity[:, :2], axis=1)
    toe_speed = np.linalg.norm(toe_velocity[:, :2], axis=1)

    foot_vector_xy = toe_smooth[:, :2] - heel_smooth[:, :2]
    foot_length = np.linalg.norm(foot_vector_xy, axis=1)
    valid_length = np.maximum(foot_length, 1e-6)
    forward = foot_vector_xy / valid_length[:, None]
    lateral = np.column_stack([-forward[:, 1], forward[:, 0]])
    heading = np.unwrap(np.arctan2(foot_vector_xy[:, 1], foot_vector_xy[:, 0]))
    angular_speed = _velocity(heading[:, None], fps)[:, 0]

    heel_quiet = _hysteresis_below(heel_speed, config.static_speed, config.release_speed)
    toe_quiet = _hysteresis_below(toe_speed, config.static_speed, config.release_speed)
    heel_normal = np.abs(heel_velocity[:, 2]) <= config.normal_speed
    toe_normal = np.abs(toe_velocity[:, 2]) <= config.normal_speed
    minimum_frames = max(2, round(config.minimum_phase_seconds * fps))
    elevated_frames = max(minimum_frames, round(config.elevated_support_seconds * fps))

    heel_candidate = heel_quiet & heel_normal
    toe_candidate = toe_quiet & toe_normal
    heel_near_floor = heel_smooth[:, 2] <= floor_height + config.ground_clearance
    toe_near_floor = toe_smooth[:, 2] <= floor_height + config.ground_clearance
    heel_on_elevated_support = _points_on_elevated_support_surfaces(
        heel_smooth,
        elevated_support_triangles,
        floor_height=floor_height,
        clearance=config.ground_clearance,
    )
    toe_on_elevated_support = _points_on_elevated_support_surfaces(
        toe_smooth,
        elevated_support_triangles,
        floor_height=floor_height,
        clearance=config.ground_clearance,
    )
    heel_supported = _retain_long_runs(
        heel_candidate & heel_near_floor,
        minimum_frames,
    ) | _retain_long_runs(
        heel_candidate & heel_on_elevated_support,
        elevated_frames,
    )
    toe_supported = _retain_long_runs(
        toe_candidate & toe_near_floor,
        minimum_frames,
    ) | _retain_long_runs(
        toe_candidate & toe_on_elevated_support,
        elevated_frames,
    )

    center = 0.5 * (heel_smooth + toe_smooth)
    center_velocity = 0.5 * (heel_velocity[:, :2] + toe_velocity[:, :2])
    pivot_uv = np.column_stack([np.full(frame_count, 0.5), np.zeros(frame_count)])
    valid_omega = np.abs(angular_speed) >= config.pivot_angular_speed
    icr = center[:, :2].copy()
    icr[valid_omega, 0] += -center_velocity[valid_omega, 1] / angular_speed[valid_omega]
    icr[valid_omega, 1] += center_velocity[valid_omega, 0] / angular_speed[valid_omega]
    heel_to_icr = icr - heel_smooth[:, :2]
    pivot_uv[:, 0] = np.sum(heel_to_icr * forward, axis=1) / valid_length
    assumed_half_width = np.maximum(0.18 * valid_length, 0.02)
    longitudinal_projection = pivot_uv[:, 0, None] * foot_vector_xy
    pivot_uv[:, 1] = np.sum((heel_to_icr - longitudinal_projection) * lateral, axis=1) / assumed_half_width
    icr_inside_sole = (pivot_uv[:, 0] >= -0.2) & (pivot_uv[:, 0] <= 1.2) & (np.abs(pivot_uv[:, 1]) <= 1.5)

    modes = np.full(frame_count, "swing", dtype=object)
    pivot_regions = np.full(frame_count, "", dtype=object)
    confidences = np.zeros(frame_count, dtype=np.float64)
    heel_near_support = heel_near_floor | heel_on_elevated_support
    toe_near_support = toe_near_floor | toe_on_elevated_support
    near_support = heel_near_support | toe_near_support | heel_supported | toe_supported
    for frame_idx in range(frame_count):
        heel_contact = bool(heel_supported[frame_idx])
        toe_contact = bool(toe_supported[frame_idx])
        rotating = bool(valid_omega[frame_idx])
        if heel_contact and toe_contact:
            speed_difference = heel_speed[frame_idx] - toe_speed[frame_idx]
            if rotating and (icr_inside_sole[frame_idx] or abs(speed_difference) > 0.005):
                modes[frame_idx] = "pivot"
                if toe_speed[frame_idx] + 0.005 < heel_speed[frame_idx]:
                    pivot_regions[frame_idx] = "toe"
                    pivot_uv[frame_idx] = (1.0, 0.0)
                elif heel_speed[frame_idx] + 0.005 < toe_speed[frame_idx]:
                    pivot_regions[frame_idx] = "heel"
                    pivot_uv[frame_idx] = (0.0, 0.0)
                else:
                    pivot_regions[frame_idx] = "sole"
                confidences[frame_idx] = 0.85
            else:
                modes[frame_idx] = "flat"
                pivot_regions[frame_idx] = "flat"
                confidences[frame_idx] = 1.0 - min(
                    max(heel_speed[frame_idx], toe_speed[frame_idx]) / max(config.release_speed, 1e-6),
                    0.5,
                )
        elif toe_contact:
            modes[frame_idx] = "pivot" if rotating else "toe"
            pivot_regions[frame_idx] = "toe"
            pivot_uv[frame_idx] = (1.0, 0.0)
            confidences[frame_idx] = 1.0 - min(toe_speed[frame_idx] / max(config.release_speed, 1e-6), 0.5)
        elif heel_contact:
            modes[frame_idx] = "pivot" if rotating else "heel"
            pivot_regions[frame_idx] = "heel"
            pivot_uv[frame_idx] = (0.0, 0.0)
            confidences[frame_idx] = 1.0 - min(heel_speed[frame_idx] / max(config.release_speed, 1e-6), 0.5)
        elif rotating and icr_inside_sole[frame_idx] and near_support[frame_idx]:
            modes[frame_idx] = "pivot"
            pivot_regions[frame_idx] = "sole"
            confidences[frame_idx] = 0.7
        else:
            common_speed = np.linalg.norm(center_velocity[frame_idx])
            relative_speed = np.linalg.norm(toe_velocity[frame_idx, :2] - heel_velocity[frame_idx, :2])
            normal_stable = max(abs(toe_velocity[frame_idx, 2]), abs(heel_velocity[frame_idx, 2]))
            if (
                (heel_near_support[frame_idx] or toe_near_support[frame_idx])
                and normal_stable <= config.normal_speed
                and common_speed >= config.slide_speed
                and relative_speed <= config.release_speed
            ):
                modes[frame_idx] = "slide"
                pivot_regions[frame_idx] = "sole"
                confidences[frame_idx] = 0.7

    modes[0] = "swing"
    pivot_regions[0] = ""
    confidences[0] = 0.0
    modes = _merge_short_mode_runs(modes, minimum_frames)
    modes[0] = "swing"
    pivot_regions[0] = ""
    confidences[0] = 0.0
    pivot_regions[modes == "swing"] = ""
    pivot_regions[modes == "flat"] = "flat"
    pivot_regions[modes == "heel"] = "heel"
    pivot_regions[modes == "toe"] = "toe"
    pivot_regions[modes == "slide"] = "sole"
    invalid_pivot_region = (modes == "pivot") & ~np.isin(
        pivot_regions,
        ("heel", "toe", "sole"),
    )
    pivot_regions[invalid_pivot_region] = "sole"
    segment_start = 0
    while segment_start < frame_count:
        segment_end = segment_start + 1
        segment_key = (modes[segment_start], pivot_regions[segment_start])
        while segment_end < frame_count and (modes[segment_end], pivot_regions[segment_end]) == segment_key:
            segment_end += 1
        if segment_key[0] == "pivot":
            pivot_uv[segment_start:segment_end] = np.median(
                pivot_uv[segment_start:segment_end],
                axis=0,
            )
        segment_start = segment_end
    return modes, pivot_regions, pivot_uv, confidences, center


def build_planar_foot_contact_plan(
    human_joints: np.ndarray,
    joint_names: list[str] | tuple[str, ...],
    joint_parent_indices: np.ndarray | list[int] | tuple[int, ...],
    toe_names: list[str] | tuple[str, str],
    *,
    fps: float,
    config: PlanarFootContactConfig | None = None,
    reference_position_scale: float = 1.0,
    reference_rotation_matrices: np.ndarray | None = None,
    reference_translations: np.ndarray | None = None,
    elevated_support_triangles: np.ndarray | None = None,
) -> FootContactPlan:
    """Infer contact modes and phases from a complete source motion sequence.

    Detection always runs in the source trajectory's meter scale so its modes
    do not depend on the target robot. ``reference_position_scale`` maps only
    intentional-slide displacements into the robot-preprocessed position
    scale used by the solver. Optional per-frame rigid transforms then map the
    scaled source sole path into the target world frame. Elevated contacts are
    accepted only when an explicit upward-facing support triangle confirms
    the source sole height and XY location.
    """
    config = config or PlanarFootContactConfig()
    _validate_config(config)
    if not np.isfinite(reference_position_scale) or reference_position_scale <= 0.0:
        raise ValueError("reference_position_scale must be finite and positive")
    joints = np.asarray(human_joints, dtype=np.float64)
    if joints.ndim != 3 or joints.shape[2] != 3:
        raise ValueError(f"human_joints must have shape [frames, joints, 3], got {joints.shape}")
    if not np.isfinite(joints).all():
        raise ValueError("human_joints must contain only finite positions")
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError("fps must be finite and positive")
    names = tuple(str(name) for name in joint_names)
    parents = np.asarray(joint_parent_indices, dtype=np.int32)
    if len(names) != joints.shape[1] or parents.shape != (joints.shape[1],):
        raise ValueError("joint names and parent indices must match the source joint dimension")
    if len(toe_names) != 2:
        raise ValueError("toe_names must contain exactly the left and right source toes")

    frame_count = joints.shape[0]
    if frame_count == 0:
        raise ValueError("human_joints must contain at least one frame")
    if reference_rotation_matrices is None:
        reference_rotations = np.tile(np.eye(3, dtype=np.float64), (frame_count, 1, 1))
    else:
        reference_rotations = np.asarray(reference_rotation_matrices, dtype=np.float64)
    if reference_rotations.shape != (frame_count, 3, 3):
        raise ValueError(
            f"reference_rotation_matrices must have shape {(frame_count, 3, 3)}, got {reference_rotations.shape}",
        )
    if reference_translations is None:
        target_translations = np.zeros((frame_count, 3), dtype=np.float64)
    else:
        target_translations = np.asarray(reference_translations, dtype=np.float64)
    if target_translations.shape != (frame_count, 3):
        raise ValueError(
            f"reference_translations must have shape {(frame_count, 3)}, got {target_translations.shape}",
        )
    if not np.isfinite(reference_rotations).all() or not np.isfinite(target_translations).all():
        raise ValueError("Reference transforms must contain only finite values")

    toe_indices = [names.index(str(name)) for name in toe_names]
    heel_indices = [int(parents[index]) for index in toe_indices]
    if any(index < 0 for index in heel_indices):
        raise ValueError("Each source toe must have a foot or ankle parent")
    floor_height = float(np.percentile(np.min(joints[:, toe_indices, 2], axis=1), 2.0))

    per_side = {}
    for side_index, side in enumerate(FOOT_CONTACT_SIDE_NAMES):
        toe_index = toe_indices[side_index]
        heel_index = heel_indices[side_index]
        heel, toe = _foot_proxy_trajectories(
            joints,
            toe_index=toe_index,
            heel_joint_index=heel_index,
            floor_height=floor_height,
            ground_clearance=config.ground_clearance,
        )
        per_side[side] = _classify_one_foot(
            heel,
            toe,
            floor_height=floor_height,
            fps=float(fps),
            config=config,
            elevated_support_triangles=elevated_support_triangles,
        )

    reference_centers_by_side = {
        side: (
            np.einsum(
                "tij,tj->ti",
                reference_rotations,
                per_side[side][4] * reference_position_scale,
            )
            + target_translations
        )
        for side in FOOT_CONTACT_SIDE_NAMES
    }
    phase_counter = dict.fromkeys(FOOT_CONTACT_SIDE_NAMES, -1)
    previous_key: dict[str, tuple[str, str] | None] = dict.fromkeys(FOOT_CONTACT_SIDE_NAMES)
    phase_start_center: dict[str, np.ndarray | None] = dict.fromkeys(FOOT_CONTACT_SIDE_NAMES)
    frames: list[dict[str, FootContactState]] = []
    for frame_idx in range(frame_count):
        frame: dict[str, FootContactState] = {}
        for side in FOOT_CONTACT_SIDE_NAMES:
            modes, pivot_regions, pivot_uv, confidences, _centers = per_side[side]
            reference_centers = reference_centers_by_side[side]
            mode = cast("FootContactMode", str(modes[frame_idx]))
            pivot_region = str(pivot_regions[frame_idx])
            key = None if mode == "swing" else (mode, pivot_region)
            if key is not None and key != previous_key[side]:
                phase_counter[side] += 1
                phase_start_center[side] = reference_centers[frame_idx, :2].copy()
            if key is None:
                phase_id = -1
                displacement = np.zeros(2, dtype=np.float64)
                phase_start_center[side] = None
            else:
                phase_id = phase_counter[side]
                start_center = phase_start_center[side]
                displacement = (
                    reference_centers[frame_idx, :2] - start_center
                    if mode == "slide" and start_center is not None
                    else np.zeros(2, dtype=np.float64)
                )
            frame[side] = FootContactState(
                mode=mode,
                phase_id=phase_id,
                pivot_region=pivot_region,
                pivot_uv=(float(pivot_uv[frame_idx, 0]), float(pivot_uv[frame_idx, 1])),
                confidence=float(np.clip(confidences[frame_idx], 0.0, 1.0)),
                reference_displacement_xy=(float(displacement[0]), float(displacement[1])),
            )
            previous_key[side] = key
        frames.append(frame)
    return FootContactPlan(tuple(frames))
