# ruff: noqa: CPY001

"""Static diagnostics for direct upper-arm direction tracking."""

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
    """Saved upper-arm direction targets and solved shoulder motion."""

    side_names: tuple[str, ...]
    joint_names: tuple[tuple[str, ...], ...]
    target_directions: np.ndarray
    actual_directions: np.ndarray
    direction_errors: np.ndarray
    actual_joint_positions: np.ndarray
    singular_values: np.ndarray


def load_shoulder_direction_diagnostics(
    result_path: str | Path,
) -> ShoulderDirectionDiagnostics:
    """Load and validate the direction-only diagnostic arrays."""

    path = Path(result_path)
    required = {
        "shoulder_direction_tracking_enabled",
        "shoulder_direction_human_arm_names",
        "shoulder_direction_joint_names",
        "shoulder_direction_target_vectors",
        "shoulder_direction_actual_vectors",
        "shoulder_direction_errors_rad",
        "shoulder_direction_actual_joint_positions",
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
        target = np.asarray(
            result["shoulder_direction_target_vectors"],
            dtype=np.float64,
        )
        actual = np.asarray(
            result["shoulder_direction_actual_vectors"],
            dtype=np.float64,
        )
        errors = np.asarray(
            result["shoulder_direction_errors_rad"],
            dtype=np.float64,
        )
        actual_q = np.asarray(
            result["shoulder_direction_actual_joint_positions"],
            dtype=np.float64,
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
    if singular_values.shape != expected_vector_shape:
        raise ValueError("Shoulder singular values have inconsistent dimensions")
    if errors.shape != (frame_count, side_count):
        raise ValueError("Shoulder direction errors have inconsistent dimensions")
    if not all(np.isfinite(array).all() for array in (target, actual, errors, actual_q, singular_values)):
        raise ValueError("Shoulder direction diagnostics contain non-finite values")
    if len(joint_names) != side_count or any(len(names) != 3 for names in joint_names):
        raise ValueError("Every shoulder side must contain three joint names")

    return ShoulderDirectionDiagnostics(
        side_names=side_names,
        joint_names=joint_names,
        target_directions=target,
        actual_directions=actual,
        direction_errors=errors,
        actual_joint_positions=actual_q,
        singular_values=singular_values,
    )


def _direction_steps(directions: np.ndarray) -> np.ndarray:
    dots = np.sum(directions[1:] * directions[:-1], axis=-1)
    return np.arccos(np.clip(dots, -1.0, 1.0))


def make_shoulder_direction_figure(
    diagnostics: ShoulderDirectionDiagnostics,
):
    """Build a dashboard for direction error and solved shoulder continuity."""

    side_count = len(diagnostics.side_names)
    figure, axes = plt.subplots(
        side_count,
        3,
        figsize=(18, 4.8 * side_count),
        constrained_layout=True,
        squeeze=False,
    )
    frames = np.arange(diagnostics.target_directions.shape[0])
    colors = ("#2563eb", "#dc2626", "#16a34a")

    for side in range(side_count):
        side_label = diagnostics.side_names[side]
        angle_axis, step_axis, error_axis = axes[side]
        actual_degrees = np.rad2deg(
            diagnostics.actual_joint_positions[:, side],
        )
        for joint, color in enumerate(colors):
            short_name = diagnostics.joint_names[side][joint].replace(
                "_joint",
                "",
            )
            angle_axis.plot(
                frames,
                actual_degrees[:, joint],
                color=color,
                label=short_name,
            )
        angle_axis.set_title(f"{side_label}: solved shoulder angles")
        angle_axis.set_xlabel("frame")
        angle_axis.set_ylabel("joint angle (deg)")
        angle_axis.legend(fontsize=7, loc="best")
        angle_axis.grid(alpha=0.2)

        actual_step = np.rad2deg(
            np.linalg.norm(
                np.diff(diagnostics.actual_joint_positions[:, side], axis=0),
                axis=-1,
            ),
        )
        target_step = np.rad2deg(
            _direction_steps(diagnostics.target_directions[:, side]),
        )
        step_axis.plot(
            frames[1:],
            actual_step,
            label="solved shoulder |Δq|",
            color="#111827",
        )
        step_axis.plot(
            frames[1:],
            target_step,
            label="human upper-arm direction Δ",
            color="#f59e0b",
        )
        step_axis.set_title("joint continuity versus source direction")
        step_axis.set_xlabel("frame")
        step_axis.set_ylabel("change (deg/frame)")
        step_axis.legend(fontsize=7, loc="best")
        step_axis.grid(alpha=0.2)

        error_axis.plot(
            frames,
            np.rad2deg(diagnostics.direction_errors[:, side]),
            color="#dc2626",
            label="direction error",
        )
        error_axis.set_title("direction tracking and conditioning")
        error_axis.set_xlabel("frame")
        error_axis.set_ylabel("direction error (deg)", color="#dc2626")
        conditioning_axis = error_axis.twinx()
        conditioning_axis.plot(
            frames,
            diagnostics.singular_values[:, side, -2],
            color="#0891b2",
            alpha=0.72,
            label="smallest nonzero singular value",
        )
        conditioning_axis.set_ylabel(
            "smallest nonzero J singular value",
            color="#0891b2",
        )
        error_axis.grid(alpha=0.2)

    figure.suptitle(
        "Direct upper-arm direction tracking diagnostics",
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
