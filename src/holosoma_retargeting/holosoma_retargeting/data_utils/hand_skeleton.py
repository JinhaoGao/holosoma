"""Visual-only hand skeleton helpers for SMPL-H motion."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HandVisualizationSpec:
    """Indices needed to draw fingers without changing retargeting anchors."""

    keypoint_indices: tuple[int, ...]
    edge_indices: tuple[tuple[int, int], ...]


def build_hand_visualization_spec(
    joint_names: list[str] | tuple[str, ...],
    mapped_joint_names: list[str] | tuple[str, ...],
) -> HandVisualizationSpec:
    """Return available SMPL-H finger points and edges in full-joint indexing."""

    joint_index = {name: index for index, name in enumerate(joint_names)}
    mapped_names = set(mapped_joint_names)
    keypoint_indices: list[int] = []
    edge_indices: list[tuple[int, int]] = []
    seen_keypoints: set[int] = set()

    for side in ("L", "R"):
        wrist_name = f"{side}_Wrist"
        if wrist_name not in joint_index:
            continue
        for finger_name in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
            chain_names = [
                wrist_name,
                *(f"{side}_{finger_name}{segment}" for segment in range(1, 4)),
            ]
            if not all(name in joint_index for name in chain_names):
                continue
            chain_indices = [joint_index[name] for name in chain_names]
            edge_indices.extend(zip(chain_indices[:-1], chain_indices[1:]))
            for name, index in zip(chain_names[1:], chain_indices[1:]):
                if name not in mapped_names and index not in seen_keypoints:
                    keypoint_indices.append(index)
                    seen_keypoints.add(index)

    return HandVisualizationSpec(
        keypoint_indices=tuple(keypoint_indices),
        edge_indices=tuple(edge_indices),
    )
