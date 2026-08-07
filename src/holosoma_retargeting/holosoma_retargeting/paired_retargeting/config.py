# ruff: noqa: CPY001
"""Configuration types for paired robot trajectory refinement."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PairedActorConfig:
    """One independently retargeted actor used by paired refinement."""

    name: str
    result_path: Path

    def __post_init__(self) -> None:
        if not self.name or not self.name.isidentifier():
            raise ValueError(f"Actor name must be a non-empty identifier, got {self.name!r}")


@dataclass(frozen=True)
class PairedFrameWindow:
    """Half-open frame interval in which a paired constraint is active."""

    start: int = 0
    stop: int | None = None

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("Frame window start must be non-negative")
        if self.stop is not None and self.stop <= self.start:
            raise ValueError("Frame window stop must be greater than start")

    def contains(self, frame_index: int) -> bool:
        """Return whether a frame belongs to this half-open interval."""
        return frame_index >= self.start and (self.stop is None or frame_index < self.stop)


@dataclass(frozen=True)
class PairedContactConfig:
    """Desired actor-B minus actor-A link position in world coordinates."""

    actor_a_link: str
    actor_b_link: str
    target_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    weight: float = 20.0
    tolerance: float | None = None
    windows: tuple[PairedFrameWindow, ...] = (PairedFrameWindow(),)

    def __post_init__(self) -> None:
        if not self.actor_a_link or not self.actor_b_link:
            raise ValueError("Contact link names must be non-empty")
        if len(self.target_offset) != 3:
            raise ValueError("Contact target_offset must contain exactly three values")
        if self.weight < 0.0:
            raise ValueError("Contact weight must be non-negative")
        if self.tolerance is not None and self.tolerance < 0.0:
            raise ValueError("Contact tolerance must be non-negative")
        if not self.windows:
            raise ValueError("Contact must contain at least one frame window")

    def is_active(self, frame_index: int) -> bool:
        """Return whether the contact is active in a frame."""
        return any(window.contains(frame_index) for window in self.windows)


@dataclass(frozen=True)
class InterActorCollisionConfig:
    """Cross-actor collision constraints for paired refinement."""

    enabled: bool = False
    minimum_distance: float = 0.01
    activation_distance: float = 0.05
    body_pairs: tuple[tuple[str, str], ...] = ()
    windows: tuple[PairedFrameWindow, ...] = (PairedFrameWindow(),)

    def __post_init__(self) -> None:
        if self.minimum_distance < 0.0:
            raise ValueError("Collision minimum_distance must be non-negative")
        if self.activation_distance < self.minimum_distance:
            raise ValueError("Collision activation_distance must not be below minimum_distance")
        if any(not actor_a or not actor_b for actor_a, actor_b in self.body_pairs):
            raise ValueError("Collision body-pair names must be non-empty")
        if not self.windows:
            raise ValueError("Collision must contain at least one frame window")

    def is_active(self, frame_index: int) -> bool:
        """Return whether collision constraints are enabled in a frame."""
        return self.enabled and any(window.contains(frame_index) for window in self.windows)


@dataclass(frozen=True)
class PairedSourceAlignmentConfig:
    """Restore a shared source-scene frame before paired optimization."""

    enabled: bool = True
    scene_scale: float | None = None

    def __post_init__(self) -> None:
        if self.scene_scale is not None and self.scene_scale <= 0.0:
            raise ValueError("Source-alignment scene_scale must be positive")


@dataclass(frozen=True)
class PairedRefinementConfig:
    """Weights and numerical settings for joint paired refinement."""

    nominal_weight: float = 100.0
    smoothness_weight: float = 5.0
    interaction_weight: float = 2.0
    root_position_weight: float = 10.0
    root_yaw_weight: float = 2.0
    source_alignment: PairedSourceAlignmentConfig = field(default_factory=PairedSourceAlignmentConfig)
    contacts: tuple[PairedContactConfig, ...] = ()
    collision: InterActorCollisionConfig = field(default_factory=InterActorCollisionConfig)
    max_sqp_iterations: int = 3
    convergence_tolerance: float = 1e-5
    root_translation_step_limit: float = 0.03
    root_quaternion_step_limit: float = 0.05
    joint_step_limit: float = 0.15

    def __post_init__(self) -> None:
        weights = (
            self.nominal_weight,
            self.smoothness_weight,
            self.interaction_weight,
            self.root_position_weight,
            self.root_yaw_weight,
        )
        if any(weight < 0.0 for weight in weights):
            raise ValueError("Paired refinement weights must be non-negative")
        if self.max_sqp_iterations < 1:
            raise ValueError("max_sqp_iterations must be positive")
        if self.convergence_tolerance <= 0.0:
            raise ValueError("convergence_tolerance must be positive")
        step_limits = (
            self.root_translation_step_limit,
            self.root_quaternion_step_limit,
            self.joint_step_limit,
        )
        if any(limit <= 0.0 for limit in step_limits):
            raise ValueError("Paired refinement step limits must be positive")


@dataclass(frozen=True)
class PairedRefinementCommand:
    """Standalone command input for paired trajectory refinement."""

    actor_a: PairedActorConfig
    actor_b: PairedActorConfig
    output_path: Path
    refinement: PairedRefinementConfig = field(default_factory=PairedRefinementConfig)
    contacts_path: Path | None = None
    overwrite: bool = False
    visualize: bool = False
    loop_visualization: bool = False

    def __post_init__(self) -> None:
        if self.actor_a.name == self.actor_b.name:
            raise ValueError("Paired actor names must be unique")
        if self.actor_a.result_path == self.actor_b.result_path:
            raise ValueError("Paired actors must use distinct result paths")
        if self.output_path.suffix.lower() != ".npz":
            raise ValueError("Paired refinement output_path must use the .npz suffix")
