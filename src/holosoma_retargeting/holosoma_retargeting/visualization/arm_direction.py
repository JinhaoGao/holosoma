# ruff: noqa: CPY001
"""Scene data for comparing human and robot upper-limb directions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from holosoma_retargeting.visualization.orientation import OrientationDiagnostics


@dataclass(frozen=True)
class ArmDirectionDiagnostics:
    """Direction-only objective data saved by the retargeter."""

    side_names: tuple[str, ...]
    shoulder_joint_names: tuple[tuple[str, ...], ...]
    target_directions_local: np.ndarray
    actual_directions_local: np.ndarray
    errors_rad: np.ndarray
    direction_weight: float
    wrist_orientation_weights: np.ndarray
    wrist_dof_counts: np.ndarray
    human_chain_names: tuple[tuple[str, ...], ...]
    robot_chain_link_names: tuple[tuple[str, ...], ...]
    robot_basis_link_names: tuple[str, ...]
    robot_torso_link_name: str


@dataclass(frozen=True)
class ArmDirectionSceneData:
    """World-space arrows and per-frame errors used by the Viser overlay."""

    side_names: tuple[str, ...]
    source_segments: np.ndarray
    robot_target_segments: np.ndarray
    robot_actual_segments: np.ndarray
    upper_errors_rad: np.ndarray
    lower_errors_rad: np.ndarray
    orientation_errors_rad: np.ndarray
    orientation_modes: tuple[tuple[str, ...], ...]
    direction_weight: float
    wrist_orientation_weights: np.ndarray
    wrist_dof_counts: np.ndarray


_FALLBACK_HUMAN_CHAINS = {
    "LeftArm": ("LeftArm", "LeftForeArm", "LeftHand"),
    "RightArm": ("RightArm", "RightForeArm", "RightHand"),
    "L_Shoulder": ("L_Shoulder", "L_Elbow", "L_Wrist"),
    "R_Shoulder": ("R_Shoulder", "R_Elbow", "R_Wrist"),
}


def _string_rows(array: np.ndarray, width: int) -> tuple[tuple[str, ...], ...]:
    values = np.asarray(array, dtype=str)
    if values.ndim != 2 or values.shape[1] != width:
        raise ValueError(f"Expected a string array with shape (sides, {width}), got {values.shape}")
    return tuple(tuple(str(value) for value in row) for row in values)


def load_optional_arm_direction_diagnostics(
    result_path: str | Path,
    *,
    expected_frames: int,
) -> ArmDirectionDiagnostics | None:
    """Load optional direction-objective arrays while supporting older results."""

    path = Path(result_path)
    required = {
        "shoulder_direction_tracking_enabled",
        "shoulder_direction_human_arm_names",
        "shoulder_direction_joint_names",
        "shoulder_direction_target_vectors",
        "shoulder_direction_actual_vectors",
        "shoulder_direction_errors_rad",
    }
    with np.load(path, allow_pickle=False) as result:
        present = {key for key in result.files if key.startswith("shoulder_direction_")}
        if not present:
            return None
        missing = sorted(required.difference(result.files))
        if missing:
            raise ValueError(f"{path} has incomplete arm-direction diagnostics; missing {missing}")
        if not bool(np.asarray(result["shoulder_direction_tracking_enabled"]).item()):
            return None

        side_names = tuple(str(value) for value in result["shoulder_direction_human_arm_names"])
        joint_rows = np.asarray(result["shoulder_direction_joint_names"])
        if joint_rows.ndim != 2 or joint_rows.shape[1] not in (2, 3):
            raise ValueError(f"{path} must save two or three shoulder joint names per side")
        shoulder_joint_names = _string_rows(joint_rows, joint_rows.shape[1])
        target = np.asarray(result["shoulder_direction_target_vectors"], dtype=np.float64)
        actual = np.asarray(result["shoulder_direction_actual_vectors"], dtype=np.float64)
        errors = np.asarray(result["shoulder_direction_errors_rad"], dtype=np.float64)
        direction_weight = float(
            np.asarray(result["shoulder_direction_weight"]).item() if "shoulder_direction_weight" in result else 0.0
        )
        wrist_weights = np.asarray(
            result["shoulder_wrist_axis_orientation_weights"]
            if "shoulder_wrist_axis_orientation_weights" in result
            else np.zeros(len(side_names)),
            dtype=np.float64,
        )
        wrist_dof_counts = np.asarray(
            result["shoulder_wrist_dof_counts"] if "shoulder_wrist_dof_counts" in result else np.zeros(len(side_names)),
            dtype=np.int32,
        )
        human_chain_names = (
            _string_rows(result["shoulder_direction_human_chain_names"], 3)
            if "shoulder_direction_human_chain_names" in result
            else ()
        )
        robot_chain_link_names = (
            _string_rows(result["shoulder_direction_robot_chain_link_names"], 3)
            if "shoulder_direction_robot_chain_link_names" in result
            else ()
        )
        robot_basis_link_names = (
            tuple(str(value) for value in result["shoulder_direction_robot_basis_link_names"])
            if "shoulder_direction_robot_basis_link_names" in result
            else ()
        )
        robot_torso_link_name = (
            str(np.asarray(result["shoulder_direction_robot_torso_link_name"]).item())
            if "shoulder_direction_robot_torso_link_name" in result
            else ""
        )

    side_count = len(side_names)
    expected_vector_shape = (expected_frames, side_count, 3)
    if side_count != 2 or target.shape != expected_vector_shape or actual.shape != expected_vector_shape:
        raise ValueError(f"{path} arm-direction vectors have inconsistent dimensions")
    if errors.shape != (expected_frames, side_count):
        raise ValueError(f"{path} arm-direction errors have inconsistent dimensions")
    if len(shoulder_joint_names) != side_count:
        raise ValueError(f"{path} must save shoulder joint names for each side")
    if wrist_weights.shape != (side_count,) or wrist_dof_counts.shape != (side_count,):
        raise ValueError(f"{path} wrist diagnostics have inconsistent dimensions")
    if not all(np.isfinite(value).all() for value in (target, actual, errors, wrist_weights)):
        raise ValueError(f"{path} arm-direction diagnostics contain non-finite values")
    if not np.isfinite(direction_weight) or direction_weight < 0.0:
        raise ValueError(f"{path} has an invalid shoulder direction weight")
    if human_chain_names and len(human_chain_names) != side_count:
        raise ValueError(f"{path} human arm-chain names have inconsistent dimensions")
    if robot_chain_link_names and len(robot_chain_link_names) != side_count:
        raise ValueError(f"{path} robot arm-chain names have inconsistent dimensions")
    if robot_basis_link_names and len(robot_basis_link_names) != side_count:
        raise ValueError(f"{path} robot torso-basis links have inconsistent dimensions")

    return ArmDirectionDiagnostics(
        side_names=side_names,
        shoulder_joint_names=shoulder_joint_names,
        target_directions_local=target,
        actual_directions_local=actual,
        errors_rad=errors,
        direction_weight=direction_weight,
        wrist_orientation_weights=wrist_weights,
        wrist_dof_counts=wrist_dof_counts,
        human_chain_names=human_chain_names,
        robot_chain_link_names=robot_chain_link_names,
        robot_basis_link_names=robot_basis_link_names,
        robot_torso_link_name=robot_torso_link_name,
    )


def _normalize_rows(vectors: np.ndarray, *, label: str) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(values, axis=-1)
    if not np.isfinite(values).all() or np.any(norms <= 1e-8):
        raise ValueError(f"{label} contains a non-finite or zero-length segment")
    return values / norms[..., None], norms


def _anatomical_bases(
    left_shoulders: np.ndarray,
    right_shoulders: np.ndarray,
    torso_origins: np.ndarray,
) -> np.ndarray:
    lateral, _ = _normalize_rows(left_shoulders - right_shoulders, label="shoulder lateral axis")
    shoulder_midpoints = 0.5 * (left_shoulders + right_shoulders)
    up_hint, _ = _normalize_rows(shoulder_midpoints - torso_origins, label="torso up axis")
    forward, _ = _normalize_rows(np.cross(lateral, up_hint), label="torso forward axis")
    up, _ = _normalize_rows(np.cross(forward, lateral), label="orthogonal torso up axis")
    return np.stack((forward, lateral, up), axis=-1)


def _robot_parent_name(
    link_name: str,
    robot_link_names: tuple[str, ...],
    robot_parent_indices: np.ndarray,
) -> str:
    index = robot_link_names.index(link_name)
    parent_index = int(robot_parent_indices[index])
    if parent_index < 0:
        raise ValueError(f"Robot torso-basis link {link_name!r} has no parent")
    return robot_link_names[parent_index]


def _orientation_modes(
    chains: tuple[tuple[str, ...], ...],
    diagnostics: ArmDirectionDiagnostics,
    orientation_diagnostics: OrientationDiagnostics | None,
) -> tuple[tuple[str, ...], ...]:
    if orientation_diagnostics is not None and orientation_diagnostics.objective_modes:
        modes_by_name = dict(
            zip(
                orientation_diagnostics.human_joint_names,
                orientation_diagnostics.objective_modes,
                strict=True,
            ),
        )
        labels = {
            "full_so3": "active full SO(3)",
            "upper_arm_direction": "replaced by upper direction",
            "wrist_axis": "active wrist axis",
            "wrist_so3": "active wrist SO(3)",
            "root_stability": "replaced by root stability",
            "disabled": "disabled by weight",
            "diagnostic_only": "diagnostic only",
        }
        return tuple(
            tuple(
                labels.get(
                    modes_by_name.get(name, "diagnostic_only"),
                    modes_by_name.get(name, "diagnostic only"),
                )
                for name in chain
            )
            for chain in chains
        )
    modes: list[tuple[str, ...]] = []
    for side_index, _chain in enumerate(chains):
        upper_mode = "replaced by upper direction" if diagnostics.direction_weight > 0.0 else "diagnostic only"
        wrist_active = (
            diagnostics.wrist_dof_counts[side_index] > 0 and diagnostics.wrist_orientation_weights[side_index] > 0.0
        )
        wrist_mode = "wrist SO(3)/axis" if wrist_active else "diagnostic only"
        modes.append((upper_mode, "diagnostic only", wrist_mode))
    return tuple(modes)


def build_arm_direction_scene_data(
    *,
    diagnostics: ArmDirectionDiagnostics,
    human_joints: np.ndarray,
    human_joint_names: tuple[str, ...],
    robot_link_positions: np.ndarray,
    robot_link_names: tuple[str, ...],
    robot_parent_indices: np.ndarray,
    mapped_human_joint_names: tuple[str, ...],
    mapped_robot_link_names: tuple[str, ...],
    orientation_diagnostics: OrientationDiagnostics | None,
) -> ArmDirectionSceneData:
    """Construct exact objective arrows plus lower-arm and SO(3) diagnostics."""

    human_positions = np.asarray(human_joints, dtype=np.float64)
    robot_positions = np.asarray(robot_link_positions, dtype=np.float64)
    frame_count = diagnostics.target_directions_local.shape[0]
    if human_positions.shape[0] != frame_count or robot_positions.shape[0] != frame_count:
        raise ValueError("Arm-direction trajectories must use the same frame count")
    if len(robot_link_names) != robot_positions.shape[1]:
        raise ValueError("Robot link names do not match saved robot positions")
    if np.asarray(robot_parent_indices).shape != (len(robot_link_names),):
        raise ValueError("Robot parent indices do not match saved robot links")

    chains = diagnostics.human_chain_names or tuple(
        _FALLBACK_HUMAN_CHAINS.get(name, ()) for name in diagnostics.side_names
    )
    if any(len(chain) != 3 for chain in chains):
        raise ValueError(f"Cannot infer human arm chains from {diagnostics.side_names}")
    human_index = {name: index for index, name in enumerate(human_joint_names)}
    missing_human = sorted({name for chain in chains for name in chain if name not in human_index})
    torso_name = "Hips" if "Hips" in human_index else "Pelvis" if "Pelvis" in human_index else ""
    if missing_human or not torso_name:
        raise ValueError(f"Human arm-direction overlay is missing joints {missing_human or ['Hips/Pelvis']}")

    mapped_links = dict(zip(mapped_human_joint_names, mapped_robot_link_names, strict=True))
    robot_chains = diagnostics.robot_chain_link_names or tuple(
        tuple(mapped_links.get(name, "") for name in chain) for chain in chains
    )
    missing_robot = sorted(
        {name for chain in robot_chains for name in chain if not name or name not in robot_link_names}
    )
    if missing_robot:
        raise ValueError(f"Robot arm-direction overlay is missing links {missing_robot}")

    basis_link_names = diagnostics.robot_basis_link_names or tuple(
        names[0].removesuffix("_joint") + "_link" for names in diagnostics.shoulder_joint_names
    )
    missing_basis = [name for name in basis_link_names if name not in robot_link_names]
    if missing_basis:
        raise ValueError(f"Robot arm-direction overlay is missing torso-basis links {missing_basis}")
    robot_torso_name = diagnostics.robot_torso_link_name
    if not robot_torso_name:
        parent_names = tuple(
            _robot_parent_name(name, robot_link_names, robot_parent_indices) for name in basis_link_names
        )
        if len(set(parent_names)) != 1:
            raise ValueError(f"Left/right robot shoulder bases do not share a torso parent: {parent_names}")
        robot_torso_name = parent_names[0]
    if robot_torso_name not in robot_link_names:
        raise ValueError(f"Robot arm-direction overlay is missing torso link {robot_torso_name!r}")

    human_basis = _anatomical_bases(
        human_positions[:, human_index[chains[0][0]]],
        human_positions[:, human_index[chains[1][0]]],
        human_positions[:, human_index[torso_name]],
    )
    robot_index = {name: index for index, name in enumerate(robot_link_names)}
    robot_basis = _anatomical_bases(
        robot_positions[:, robot_index[basis_link_names[0]]],
        robot_positions[:, robot_index[basis_link_names[1]]],
        robot_positions[:, robot_index[robot_torso_name]],
    )

    source_segments = np.empty((frame_count, 2, 2, 2, 3), dtype=np.float64)
    robot_actual_segments = np.empty_like(source_segments)
    robot_target_segments = np.empty_like(source_segments)
    lower_errors = np.empty((frame_count, 2), dtype=np.float64)
    for side_index, (human_chain, robot_chain) in enumerate(zip(chains, robot_chains, strict=True)):
        human_arm, human_forearm, human_hand = (human_positions[:, human_index[name]] for name in human_chain)
        robot_arm, robot_elbow, robot_hand = (robot_positions[:, robot_index[name]] for name in robot_chain)
        source_segments[:, side_index, 0] = np.stack((human_arm, human_forearm), axis=1)
        source_segments[:, side_index, 1] = np.stack((human_forearm, human_hand), axis=1)
        robot_actual_segments[:, side_index, 0] = np.stack((robot_arm, robot_elbow), axis=1)
        robot_actual_segments[:, side_index, 1] = np.stack((robot_elbow, robot_hand), axis=1)

        _, upper_lengths = _normalize_rows(
            robot_elbow - robot_arm,
            label=f"{diagnostics.side_names[side_index]} robot upper arm",
        )
        lower_actual_world, lower_lengths = _normalize_rows(
            robot_hand - robot_elbow,
            label=f"{diagnostics.side_names[side_index]} robot lower arm",
        )
        human_lower_world, _ = _normalize_rows(
            human_hand - human_forearm,
            label=f"{diagnostics.side_names[side_index]} human lower arm",
        )
        human_lower_local = np.einsum("fji,fj->fi", human_basis, human_lower_world)
        upper_target_world = np.einsum(
            "fij,fj->fi",
            robot_basis,
            diagnostics.target_directions_local[:, side_index],
        )
        lower_target_world = np.einsum("fij,fj->fi", robot_basis, human_lower_local)
        robot_target_segments[:, side_index, 0] = np.stack(
            (robot_arm, robot_arm + upper_lengths[:, None] * upper_target_world),
            axis=1,
        )
        robot_target_segments[:, side_index, 1] = np.stack(
            (robot_elbow, robot_elbow + lower_lengths[:, None] * lower_target_world),
            axis=1,
        )
        lower_errors[:, side_index] = np.arccos(
            np.clip(np.sum(lower_actual_world * lower_target_world, axis=-1), -1.0, 1.0),
        )

    orientation_errors = np.full((frame_count, 2, 3), np.nan, dtype=np.float64)
    if orientation_diagnostics is not None:
        orientation_index = {name: index for index, name in enumerate(orientation_diagnostics.human_joint_names)}
        for side_index, chain in enumerate(chains):
            for segment_index, name in enumerate(chain):
                index = orientation_index.get(name)
                if index is not None:
                    orientation_errors[:, side_index, segment_index] = orientation_diagnostics.errors_rad[:, index]

    return ArmDirectionSceneData(
        side_names=diagnostics.side_names,
        source_segments=source_segments.astype(np.float32),
        robot_target_segments=robot_target_segments.astype(np.float32),
        robot_actual_segments=robot_actual_segments.astype(np.float32),
        upper_errors_rad=diagnostics.errors_rad.astype(np.float32),
        lower_errors_rad=lower_errors.astype(np.float32),
        orientation_errors_rad=orientation_errors.astype(np.float32),
        orientation_modes=_orientation_modes(
            chains,
            diagnostics,
            orientation_diagnostics,
        ),
        direction_weight=diagnostics.direction_weight,
        wrist_orientation_weights=diagnostics.wrist_orientation_weights.astype(np.float32),
        wrist_dof_counts=diagnostics.wrist_dof_counts.astype(np.int32),
    )
