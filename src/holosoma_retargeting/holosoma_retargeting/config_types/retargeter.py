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
class RetargeterConfig:
    """Configuration for retargeter parameters.

    These parameters control the retargeting optimization process.
    """

    q_a_init_idx: int = -7
    """Index in robot's configuration where optimization variables start.
    -7: starts from floating base, -3: starts from translation of floating base,
    0: starts from actuated DOF, 12: starts from waist, 15: starts from left shoulder"""

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

    foot_sticking_fallback_tolerance: float | None = 0.02
    """Fallback XY tolerance used only when the normal foot-sticking problem is infeasible.
    Set to None to disable the fallback."""

    release_foot_sticking_on_infeasible: bool = True
    """Whether to release only the current frame's foot-sticking constraints if
    both the normal and relaxed problems remain infeasible."""

    release_object_non_penetration_on_infeasible: bool = False
    """Opt in to releasing only the current frame's robot-object non-penetration
    constraints as a last local fallback. Ground constraints remain enabled.
    Disabled by default because such frames no longer satisfy the collision
    acceptance criterion."""

    retry_without_foot_sticking_on_infeasible: bool = True
    """Whether to retry an entire sequence without foot sticking when a local
    release still cannot recover a feasible trajectory."""

    foot_lock: FootLockConfig = field(default_factory=FootLockConfig)
    """Configuration for explicit frame-range based foot locking."""

    step_size: float = 0.2
    """Trust region for each SQP iteration."""

    sqp_max_iterations: int = 50
    """Safety cap for SQP iterations per frame."""

    sqp_min_iterations: int = 4
    """Minimum SQP iterations before convergence-based stopping is allowed."""

    sqp_convergence_patience: int = 3
    """Stop after this many consecutive iterations without a significant cost decrease."""

    sqp_abs_cost_tolerance: float = 1e-9
    """Absolute cost decrease below which an SQP iteration counts as stalled."""

    sqp_rel_cost_tolerance: float = 1e-7
    """Relative cost decrease below which an SQP iteration counts as stalled."""

    visualize: bool = False
    """Whether to visualize the retargeting process."""

    mesh_opacity: float = 1.0
    """Opacity for robot/object meshes in the retargeting viewer.
    Values below 1.0 use a neutral translucent mesh color."""

    debug: bool = False
    """Whether to enable debug mode."""

    show_interaction_mesh: bool = False
    """Whether to show the solver interaction mesh in the retargeting viewer."""

    save_interaction_mesh: bool = False
    """Whether to save interaction mesh vertices and tetrahedra into the output .npz."""

    interaction_mesh_mode: Literal["source", "target", "both"] = "both"
    """Which interaction mesh to visualize: source human-object, target robot-object, or both."""

    interaction_mesh_edges: Literal["all", "cross"] = "cross"
    """Which interaction mesh edges to draw. 'cross' keeps only human/robot-to-object edges."""

    interaction_mesh_line_width: float = 1.0
    """Line width for interaction mesh edges."""

    self_collision: SelfCollisionConfig = field(default_factory=SelfCollisionConfig)
    """Configuration for self-collision avoidance."""

    w_nominal_tracking_init: float = 5.0
    """Initial weight for nominal tracking cost."""

    nominal_tracking_tau: float = 1e6
    """Time constant for the nominal tracking cost."""

    orientation_weights: dict[str, float] = field(default_factory=dict)
    """Per-human-joint SO(3) tracking weights. Empty or all-zero disables
    orientation tracking and preserves the position-only solver path."""

    orientation_alignment_mode: Literal["first_frame", "explicit"] = "first_frame"
    """How fixed human-to-robot link-frame offsets are obtained."""

    orientation_alignment_quaternions_wxyz: dict[
        str,
        tuple[float, float, float, float],
    ] | None = None
    """Explicit per-human-joint alignment quaternions used when
    orientation_alignment_mode is 'explicit'."""
