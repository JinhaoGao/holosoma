# ruff: noqa: CPY001
"""Recover a shared source-scene frame for independently solved actors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from holosoma_retargeting.paired_retargeting.config import PairedSourceAlignmentConfig
from holosoma_retargeting.paired_retargeting.result import PairedActorTrajectory


@dataclass(frozen=True)
class PairedSourceAlignment:
    """Static actor translations that restore source-relative placement."""

    actor_a_translation: np.ndarray
    actor_b_translation: np.ndarray
    scene_scale: float
    source_relative_xy_m: np.ndarray
    source_fbx: Path | None
    mode: str

    def translate_actor_a_points(self, points: np.ndarray) -> np.ndarray:
        """Place actor-A points in the paired world."""
        return np.asarray(points, dtype=np.float64) + self.actor_a_translation

    def translate_actor_b_points(self, points: np.ndarray) -> np.ndarray:
        """Place actor-B points in the paired world."""
        return np.asarray(points, dtype=np.float64) + self.actor_b_translation


def resolve_source_alignment(
    actor_a: PairedActorTrajectory,
    actor_b: PairedActorTrajectory,
    config: PairedSourceAlignmentConfig,
) -> PairedSourceAlignment:
    """Resolve an actor-A-anchored shared frame from source FBX metadata."""
    zero = np.zeros(3, dtype=np.float64)
    if not config.enabled:
        return PairedSourceAlignment(
            actor_a_translation=zero.copy(),
            actor_b_translation=zero.copy(),
            scene_scale=1.0,
            source_relative_xy_m=np.zeros(2, dtype=np.float64),
            source_fbx=None,
            mode="disabled",
        )
    if actor_a.source_data_format != "fbx_mocap":
        return PairedSourceAlignment(
            actor_a_translation=zero.copy(),
            actor_b_translation=zero.copy(),
            scene_scale=1.0,
            source_relative_xy_m=np.zeros(2, dtype=np.float64),
            source_fbx=None,
            mode="identity_non_fbx",
        )
    metadata = (
        actor_a.source_xy_origin_m,
        actor_b.source_xy_origin_m,
        actor_a.human_position_scale,
        actor_b.human_position_scale,
        actor_a.source_fbx,
        actor_b.source_fbx,
    )
    if any(value is None for value in metadata):
        raise ValueError("FBX paired alignment requires source origins, scales, and source_fbx metadata")
    assert actor_a.source_xy_origin_m is not None
    assert actor_b.source_xy_origin_m is not None
    assert actor_a.human_position_scale is not None
    assert actor_b.human_position_scale is not None
    assert actor_a.source_fbx is not None
    assert actor_b.source_fbx is not None
    if actor_a.source_fbx.resolve() != actor_b.source_fbx.resolve():
        raise ValueError(
            f"FBX paired actors must come from the same source container: {actor_a.source_fbx} != {actor_b.source_fbx}"
        )
    scene_scale = (
        float(config.scene_scale)
        if config.scene_scale is not None
        else 0.5 * (actor_a.human_position_scale + actor_b.human_position_scale)
    )
    source_relative_xy_m = actor_b.source_xy_origin_m - actor_a.source_xy_origin_m
    actor_b_translation = np.asarray(
        (
            scene_scale * source_relative_xy_m[0],
            scene_scale * source_relative_xy_m[1],
            0.0,
        ),
        dtype=np.float64,
    )
    return PairedSourceAlignment(
        actor_a_translation=zero.copy(),
        actor_b_translation=actor_b_translation,
        scene_scale=scene_scale,
        source_relative_xy_m=source_relative_xy_m.copy(),
        source_fbx=actor_a.source_fbx,
        mode="fbx_source_xy_origin",
    )
