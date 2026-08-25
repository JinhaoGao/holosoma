# ruff: noqa: CPY001

"""Configuration types for retargeter settings."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


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
    """Sequence-aware upper-arm direction tracking for serial shoulders."""

    enable: bool = False
    """Replace upper-limb SO(3) costs with direction and branch tasks."""

    direction_weight: float = 100.0
    """SQP weight for each upper-arm unit-direction residual."""

    branch_weight: float = 50.0
    """SQP weight that keeps the solution on the planned null-space branch."""

    candidate_count: int = 12
    """Maximum shoulder-manifold samples retained per side and frame."""

    seed_count: int = 32
    """Deterministic multi-start seeds used to discover the first-frame branches."""

    global_seed_stride: int = 30
    """Frame stride for refreshing the manifold with global multi-start seeds."""

    refresh_candidate_count: int = 4
    """Candidate slots reserved for non-continuation branch recovery samples."""

    max_nfev: int = 30
    """Maximum bounded least-squares evaluations for one candidate projection."""

    direction_tolerance_rad: float = 0.12
    """Direction-error scale used by the candidate node cost."""

    forearm_selection_weight: float = 0.25
    """Node-cost weight used to select an elbow-plane-compatible shoulder branch."""

    joint_limit_weight: float = 0.02
    """Node-cost weight for keeping planned upper-limb joints away from limits."""

    velocity_weight: float = 10.0
    """Temporal graph weight for normalized joint velocity."""

    acceleration_weight: float = 100.0
    """Temporal graph weight for normalized joint acceleration."""

    reference_tracking_weight: float = 20.0
    """Tracking weight used when smoothing the selected discrete branch path."""

    joint_reference_weight: float = 10.0
    """Low SQP weight for traversing smoothed off-manifold branch transitions."""

    max_candidate_step_rad: float = 0.75
    """Largest step used to propagate one candidate branch to the next frame."""

    max_frame_step_rad: float = 0.45
    """Largest per-frame shoulder joint change allowed in the final SQP."""

    wrist_axis_weight_scale: float = 1.0
    """Scale applied to E1 Hand weights after projection onto elbow yaw."""


@dataclass(frozen=True)
class RootStabilityConfig:
    """Track source-root translation and torso orientation with the robot root."""

    position_weight: float = 0.0
    """SQP weight for the aligned source-root world-position target."""

    orientation_weight: float = 0.0
    """SQP weight for the aligned source-torso world-orientation target."""


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
    """Whether to enforce foot sticking constraints."""

    penetration_tolerance: float = 0.001
    """Tolerance for penetration when enforcing non-penetration constraints."""

    foot_sticking_tolerance: float = 1e-3
    """Tolerance for foot sticking constraints in x, y."""

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
    """Optional sequence-aware shoulder direction and branch tracking."""

    natural_pose_joint_positions: dict[str, float] = field(
        default_factory=dict,
    )
    """Fixed natural-pose joint angles in radians, keyed by scalar robot
    joint name. A configured profile must cover every actuated joint."""

    natural_pose_weights: dict[str, float] = field(default_factory=dict)
    """Non-negative natural-pose cost weights keyed by the same joint names
    as natural_pose_joint_positions. Zero-weight joints stay documented
    in the profile but do not contribute to the objective."""
