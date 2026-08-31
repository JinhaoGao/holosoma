# ruff: noqa: CPY001

"""Configuration types for retargeter settings."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class PlanarFootContactConfig:
    """Configuration for contact-aware planar foot constraints."""

    smoothing_window_seconds: float = 0.1
    """Centered smoothing window used before differentiating source feet."""

    static_speed: float = 0.025
    """Tangential speed in m/s below which a source contact point may plant."""

    release_speed: float = 0.06
    """Tangential speed in m/s above which a planted source point is released."""

    normal_speed: float = 0.08
    """Maximum absolute normal speed in m/s for a contact candidate."""

    pivot_angular_speed: float = 0.12
    """Minimum planar foot angular speed in rad/s for pivot classification."""

    slide_speed: float = 0.08
    """Minimum coherent planar speed in m/s for intentional sliding."""

    ground_clearance: float = 0.04
    """Distance in meters from the lowest support level treated as near-ground."""

    minimum_phase_seconds: float = 0.1
    """Minimum duration retained for a detected contact-point phase."""

    elevated_support_seconds: float = 0.2
    """Stable duration required after an elevated support surface is confirmed."""

    heading_tolerance: float = 2e-3
    """Linearized tangential tolerance used to hold flat-foot yaw."""

    slide_tracking_tolerance: float = 5e-3
    """XY tolerance for target-world intentional slide tracking."""


@dataclass(frozen=True)
class FootLockConfig:
    """Configuration for explicit frame-range based foot locking constraints."""

    enable: bool = False
    """Whether to enforce explicit frame-range based foot locking constraints."""

    windows: dict[str, list[tuple[int, int]]] | None = None
    """Per-foot inclusive frame windows for locking.
    Example: {"L_Toe": [(30, 60)], "R_Toe": [(10, 20), (80, 95)]}"""

    z_floor: float = 0.0
    """Floor height used by Z pinning constraints."""

    tolerance: float = 5e-3
    """Tolerance for Z floor pinning constraints."""


@dataclass(frozen=True)
class SelfCollisionConfig:
    """Configuration for self-collision avoidance constraints."""

    enable: bool = False
    """Whether to enforce self-collision constraints."""

    pairs: list[tuple[str, str]] = field(default_factory=list)
    """Body name pairs to check for self-collision.
    Example: [("left_elbow_link", "left_knee_link"), ("left_wrist_yaw_link", "left_knee_link")]"""

    windows: list[tuple[int, int]] | None = None
    """Inclusive frame windows during which self-collision is enforced.
    If None, enforced on all frames.
    Example: [(50, 120)] means only enforce on frames 50..120."""

    tolerance: float = 0.02
    """Minimum distance (meters) to maintain between body pairs."""


@dataclass(frozen=True)
class ShoulderDirectionConfig:
    """Direct upper-arm direction tracking for serial shoulders."""

    enable: bool = False
    """Replace upper-limb SO(3) costs with upper-arm direction tasks."""

    direction_weight: float = 100.0
    """SQP weight for each upper-arm unit-direction residual."""

    wrist_axis_weight_scale: float = 1.0
    """Scale applied to E1 Hand weights after projection onto elbow yaw."""


@dataclass(frozen=True)
class RootStabilityConfig:
    """Track source-root translation and orientation with the robot root."""

    position_weight: float = 0.0
    """SQP weight for the aligned source-root world-position target."""

    orientation_weight: float = 0.0
    """SQP weight for the aligned source-root world-orientation target."""


@dataclass(frozen=True)
class RetargeterConfig:
    """Configuration for retargeter parameters.

    These parameters control the retargeting optimization process.
    """

    visualize: bool = False
    """Whether to start the live Viser retargeting viewer."""

    debug: bool = False
    """Whether to draw human, robot, hand, and object diagnostic overlays."""

    dynamic_ground_window: bool = True
    """Whether robot-only ground points follow the robot root each frame."""

    interaction_mesh_weight: float = 10.0
    """Global weight for every Interaction Mesh Laplacian residual."""

    arm_interaction_mesh_weight_scale: float = 1.0
    """Multiplier applied to upper-limb anchor rows in the mesh objective."""

    root_stability: RootStabilityConfig = field(
        default_factory=RootStabilityConfig,
    )
    """Optional source-aligned floating-root stability objective."""

    q_a_init_idx: int = -7
    """Offset used to select the first optimized qpos address as ``7 + offset``.
    For example, -7 starts at qpos[0] (the full floating base), -3 starts at
    qpos[4] (the final three floating-base quaternion components), and 0 starts
    at qpos[7] (the actuated DOFs). Positive offsets select later actuated
    joints according to the configured robot topology."""

    activate_joint_limits: bool = True
    """Whether to enforce joint limits during retargeting."""

    activate_obj_non_penetration: bool = True
    """Whether to enforce object non-penetration constraints.
    Ground non-penetration remains active when this is disabled."""

    activate_foot_sticking: bool = True
    """Whether to enforce contact-aware planar foot constraints."""

    penetration_tolerance: float = 0.001
    """Tolerance for penetration when enforcing non-penetration constraints."""

    foot_sticking_tolerance: float = 1e-3
    """XY tolerance for planted sole-region constraints."""

    planar_foot_contact: PlanarFootContactConfig = field(
        default_factory=PlanarFootContactConfig,
    )
    """Detection and constraint parameters for contact-aware planar locking."""

    foot_lock: FootLockConfig = field(default_factory=FootLockConfig)
    """Configuration for explicit frame-range based foot locking."""

    step_size: float = 0.2
    """Trust region for each SQP iteration."""

    self_collision: SelfCollisionConfig = field(default_factory=SelfCollisionConfig)
    """Configuration for self-collision avoidance."""

    w_nominal_tracking_init: float = 5.0
    """Initial weight for nominal tracking cost."""

    nominal_tracking_tau: float = 1e6
    """Time constant for the nominal tracking cost."""

    orientation_weights: dict[str, float] = field(default_factory=dict)
    """Per-human-joint SO(3) tracking weights. Empty or all-zero disables
    orientation tracking and preserves the position-only solver path."""

    orientation_preview: bool = False
    """Save independently calibrated human-target and robot-link frames for
    visualization without adding an SO(3) objective or computing errors."""

    orientation_alignment_mode: Literal[
        "t_pose",
        "first_frame",
        "explicit",
    ] = "t_pose"
    """How fixed human-to-robot link-frame offsets are obtained. T-pose
    calibration is the safe default; first_frame is retained only for legacy
    reproducibility."""

    orientation_alignment_quaternions_wxyz: (
        dict[
            str,
            tuple[float, float, float, float],
        ]
        | None
    ) = None
    """Explicit per-human-joint alignment quaternions used when
    orientation_alignment_mode is 'explicit'."""

    shoulder_direction: ShoulderDirectionConfig = field(
        default_factory=ShoulderDirectionConfig,
    )
    """Optional direct upper-arm direction tracking."""

    natural_pose_joint_positions: dict[str, float] = field(
        default_factory=dict,
    )
    """Fixed natural-pose joint angles in radians, keyed by scalar robot
    joint name. A configured profile must cover every actuated joint."""

    natural_pose_weights: dict[str, float] = field(default_factory=dict)
    """Non-negative natural-pose cost weights keyed by the same joint names
    as natural_pose_joint_positions. Zero-weight joints stay documented
    in the profile but do not contribute to the objective."""
