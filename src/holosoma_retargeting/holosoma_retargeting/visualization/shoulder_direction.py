# ruff: noqa: CPY001

"""Static diagnostics for sequence-aware shoulder direction retargeting."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
from matplotlib import pyplot as plt


@dataclass(frozen=True)
class ShoulderDirectionDiagnostics:
    """Saved shoulder targets, solutions, references, and manifold samples."""

    side_names: tuple[str, ...]
    joint_names: tuple[tuple[str, ...], ...]
    target_directions: np.ndarray
    actual_directions: np.ndarray
    direction_errors: np.ndarray
    actual_joint_positions: np.ndarray
    reference_joint_positions: np.ndarray
    candidate_joint_positions: np.ndarray
    candidate_node_costs: np.ndarray
    candidate_valid: np.ndarray
    selected_candidate_indices: np.ndarray
    singular_values: np.ndarray


def load_shoulder_direction_diagnostics(
    result_path: str | Path,
) -> ShoulderDirectionDiagnostics:
    """Load and validate the diagnostic arrays saved by the retargeter."""

    path = Path(result_path)
    required = {
        "shoulder_direction_tracking_enabled",
        "shoulder_direction_human_arm_names",
        "shoulder_direction_joint_names",
        "shoulder_direction_target_vectors",
        "shoulder_direction_actual_vectors",
        "shoulder_direction_errors_rad",
        "shoulder_direction_actual_joint_positions",
        "shoulder_direction_reference_joint_positions",
        "shoulder_direction_candidate_joint_positions",
        "shoulder_direction_candidate_node_costs",
        "shoulder_direction_candidate_valid",
        "shoulder_direction_selected_candidate_indices",
        "shoulder_direction_singular_values",
    }
    with np.load(path, allow_pickle=False) as result:
        missing = sorted(required.difference(result.files))
        if missing:
            raise ValueError(
                f"{path} has incomplete shoulder direction diagnostics; missing {missing}",
            )
        if not bool(result["shoulder_direction_tracking_enabled"]):
            raise ValueError(f"{path} does not contain enabled shoulder tracking")
        side_names = tuple(str(value) for value in result["shoulder_direction_human_arm_names"])
        joint_names = tuple(tuple(str(value) for value in row) for row in result["shoulder_direction_joint_names"])
        target = np.asarray(result["shoulder_direction_target_vectors"], dtype=np.float64)
        actual = np.asarray(result["shoulder_direction_actual_vectors"], dtype=np.float64)
        errors = np.asarray(result["shoulder_direction_errors_rad"], dtype=np.float64)
        actual_q = np.asarray(
            result["shoulder_direction_actual_joint_positions"],
            dtype=np.float64,
        )
        reference_q = np.asarray(
            result["shoulder_direction_reference_joint_positions"],
            dtype=np.float64,
        )
        candidates = np.asarray(
            result["shoulder_direction_candidate_joint_positions"],
            dtype=np.float64,
        )
        node_costs = np.asarray(
            result["shoulder_direction_candidate_node_costs"],
            dtype=np.float64,
        )
        valid = np.asarray(result["shoulder_direction_candidate_valid"], dtype=bool)
        selected = np.asarray(
            result["shoulder_direction_selected_candidate_indices"],
            dtype=np.int32,
        )
        singular_values = np.asarray(
            result["shoulder_direction_singular_values"],
            dtype=np.float64,
        )

    frame_count = target.shape[0]
    side_count = len(side_names)
    expected_vector_shape = (frame_count, side_count, 3)
    if side_count == 0 or target.shape != expected_vector_shape:
        raise ValueError("Shoulder target vectors have inconsistent dimensions")
    if actual.shape != expected_vector_shape or actual_q.shape != expected_vector_shape:
        raise ValueError("Shoulder actual arrays have inconsistent dimensions")
    if reference_q.shape != expected_vector_shape or singular_values.shape != expected_vector_shape:
        raise ValueError("Shoulder reference arrays have inconsistent dimensions")
    if errors.shape != (frame_count, side_count):
        raise ValueError("Shoulder direction errors have inconsistent dimensions")
    if candidates.ndim != 4 or candidates.shape[:2] != (frame_count, side_count):
        raise ValueError("Shoulder candidate positions have inconsistent dimensions")
    if candidates.shape[-1] != 3:
        raise ValueError("Shoulder candidates must contain three joint angles")
    candidate_shape = candidates.shape[:3]
    if node_costs.shape != candidate_shape or valid.shape != candidate_shape:
        raise ValueError("Shoulder candidate metadata have inconsistent dimensions")
    if selected.shape != (frame_count, side_count):
        raise ValueError("Shoulder selected candidate indices have inconsistent dimensions")
    arrays = (target, actual, errors, actual_q, reference_q, singular_values)
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("Shoulder direction diagnostics contain non-finite values")
    for side_index in range(side_count):
        if len(joint_names[side_index]) != 3:
            raise ValueError("Every shoulder side must contain three joint names")
        for frame in range(frame_count):
            index = int(selected[frame, side_index])
            if index < 0 or index >= candidates.shape[2] or not valid[frame, side_index, index]:
                raise ValueError("Selected shoulder candidate is invalid")

    return ShoulderDirectionDiagnostics(
        side_names=side_names,
        joint_names=joint_names,
        target_directions=target,
        actual_directions=actual,
        direction_errors=errors,
        actual_joint_positions=actual_q,
        reference_joint_positions=reference_q,
        candidate_joint_positions=candidates,
        candidate_node_costs=node_costs,
        candidate_valid=valid,
        selected_candidate_indices=selected,
        singular_values=singular_values,
    )


def _direction_steps(directions: np.ndarray) -> np.ndarray:
    dots = np.sum(directions[1:] * directions[:-1], axis=-1)
    return np.arccos(np.clip(dots, -1.0, 1.0))


def _worst_transition(diagnostics: ShoulderDirectionDiagnostics, side: int) -> int:
    actual_steps = np.linalg.norm(
        np.diff(diagnostics.actual_joint_positions[:, side], axis=0),
        axis=-1,
    )
    target_steps = _direction_steps(diagnostics.target_directions[:, side])
    discrepancy = actual_steps - target_steps
    return int(np.argmax(discrepancy)) + 1


def make_shoulder_direction_figure(
    diagnostics: ShoulderDirectionDiagnostics,
):
    """Build a dashboard exposing tracking errors and candidate branch geometry."""

    side_count = len(diagnostics.side_names)
    figure = plt.figure(figsize=(23, 5.8 * side_count), constrained_layout=True)
    grid = figure.add_gridspec(side_count, 4, width_ratios=(1.35, 1.15, 1.0, 1.15))
    frames = np.arange(diagnostics.target_directions.shape[0])
    colors = ("#2563eb", "#dc2626", "#16a34a")

    for side in range(side_count):
        worst_frame = _worst_transition(diagnostics, side)
        side_label = diagnostics.side_names[side]
        angle_axis = figure.add_subplot(grid[side, 0])
        actual_degrees = np.rad2deg(diagnostics.actual_joint_positions[:, side])
        reference_degrees = np.rad2deg(diagnostics.reference_joint_positions[:, side])
        for joint, color in enumerate(colors):
            short_name = diagnostics.joint_names[side][joint].replace("_joint", "")
            angle_axis.plot(frames, actual_degrees[:, joint], color=color, label=short_name)
            angle_axis.plot(
                frames,
                reference_degrees[:, joint],
                color=color,
                linestyle="--",
                alpha=0.58,
            )
        angle_axis.axvline(worst_frame, color="black", linewidth=1, linestyle=":")
        angle_axis.set_title(f"{side_label}: actual (solid) and planned branch (dashed)")
        angle_axis.set_xlabel("frame")
        angle_axis.set_ylabel("joint angle (deg)")
        angle_axis.legend(fontsize=7, loc="best")
        angle_axis.grid(alpha=0.2)

        step_axis = figure.add_subplot(grid[side, 1])
        actual_step = np.rad2deg(
            np.linalg.norm(
                np.diff(diagnostics.actual_joint_positions[:, side], axis=0),
                axis=-1,
            ),
        )
        reference_step = np.rad2deg(
            np.linalg.norm(
                np.diff(diagnostics.reference_joint_positions[:, side], axis=0),
                axis=-1,
            ),
        )
        target_step = np.rad2deg(
            _direction_steps(diagnostics.target_directions[:, side]),
        )
        step_frames = frames[1:]
        step_axis.plot(step_frames, actual_step, label="actual shoulder |Δq|", color="#111827")
        step_axis.plot(step_frames, reference_step, label="planned |Δq|", color="#7c3aed")
        step_axis.plot(step_frames, target_step, label="human direction Δ", color="#f59e0b")
        step_axis.axvline(worst_frame, color="black", linewidth=1, linestyle=":")
        step_axis.set_title(f"branch-change evidence; marked frame {worst_frame}")
        step_axis.set_xlabel("frame")
        step_axis.set_ylabel("change (deg/frame)")
        step_axis.legend(fontsize=7, loc="best")
        step_axis.grid(alpha=0.2)

        error_axis = figure.add_subplot(grid[side, 2])
        error_axis.plot(
            frames,
            np.rad2deg(diagnostics.direction_errors[:, side]),
            color="#dc2626",
            label="direction error",
        )
        error_axis.axvline(worst_frame, color="black", linewidth=1, linestyle=":")
        error_axis.set_title("direction tracking and conditioning")
        error_axis.set_xlabel("frame")
        error_axis.set_ylabel("direction error (deg)", color="#dc2626")
        conditioning_axis = error_axis.twinx()
        singular_values = diagnostics.singular_values[:, side]
        conditioning_axis.plot(
            frames,
            singular_values[:, -1],
            color="#0891b2",
            alpha=0.72,
            label="smallest singular value",
        )
        conditioning_axis.set_ylabel("smallest J singular value", color="#0891b2")
        error_axis.grid(alpha=0.2)

        manifold_axis = figure.add_subplot(grid[side, 3], projection="3d")
        mask = diagnostics.candidate_valid[worst_frame, side]
        candidates = np.rad2deg(
            diagnostics.candidate_joint_positions[worst_frame, side, mask],
        )
        costs = diagnostics.candidate_node_costs[worst_frame, side, mask]
        scatter = manifold_axis.scatter(
            candidates[:, 0],
            candidates[:, 1],
            candidates[:, 2],
            c=costs,
            cmap="viridis",
            s=42,
            alpha=0.88,
            label="feasible samples",
        )
        selected_index = int(
            diagnostics.selected_candidate_indices[worst_frame, side],
        )
        selected_q = np.rad2deg(
            diagnostics.candidate_joint_positions[
                worst_frame,
                side,
                selected_index,
            ],
        )
        actual_q = actual_degrees[worst_frame]
        manifold_axis.scatter(*selected_q, marker="*", s=180, color="#f97316", label="selected")
        manifold_axis.scatter(*actual_q, marker="x", s=110, color="#dc2626", label="actual")
        neighborhood = slice(max(0, worst_frame - 8), min(len(frames), worst_frame + 9))
        local_reference = reference_degrees[neighborhood]
        manifold_axis.plot(
            local_reference[:, 0],
            local_reference[:, 1],
            local_reference[:, 2],
            color="black",
            linewidth=1.4,
            label="selected temporal path",
        )
        manifold_axis.set_title(f"sampled solution manifold at frame {worst_frame}")
        manifold_axis.set_xlabel("pitch (deg)")
        manifold_axis.set_ylabel("roll (deg)")
        manifold_axis.set_zlabel("yaw (deg)")
        manifold_axis.legend(fontsize=7, loc="best")
        figure.colorbar(scatter, ax=manifold_axis, shrink=0.58, pad=0.08, label="node cost")

    figure.suptitle(
        "Shoulder direction branch diagnostics: large |Δq| with small human-direction Δ indicates an ambiguity switch",
        fontsize=14,
    )
    return figure


def save_shoulder_direction_diagnostics(
    result_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Render a diagnostic PNG next to a shoulder-enabled result by default."""

    result = Path(result_path)
    destination = (
        Path(output_path) if output_path is not None else result.with_name(f"{result.stem}_shoulder_direction.png")
    )
    diagnostics = load_shoulder_direction_diagnostics(result)
    figure = make_shoulder_direction_figure(diagnostics)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination


def main() -> None:
    """Render diagnostics from a saved retargeting result."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    print(save_shoulder_direction_diagnostics(arguments.result, arguments.output))


if __name__ == "__main__":
    main()
