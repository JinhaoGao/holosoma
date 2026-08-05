# ruff: noqa: CPY001
"""Joint-angle trajectory diagnostics for saved robot motions."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import plotly.graph_objects as go


@dataclass(frozen=True)
class JointAngleDiagnostics:
    """Joint trajectories and limits aligned to the visualized URDF order."""

    joint_names: tuple[str, ...]
    angles: np.ndarray
    limits: tuple[tuple[float | None, float | None], ...]


def build_joint_angle_diagnostics(
    qpos: np.ndarray,
    joint_limits: Mapping[str, tuple[float | None, float | None]],
    qpos_to_viser_joint_indices: Sequence[int] | np.ndarray | None,
) -> JointAngleDiagnostics:
    """Extract saved joint trajectories and align them with URDF joint names."""

    qpos_array = np.asarray(qpos, dtype=np.float64)
    if qpos_array.ndim != 2 or qpos_array.shape[0] == 0:
        raise ValueError(f"qpos must have shape (frames, values) with at least one frame, got {qpos_array.shape}")

    joint_names = tuple(str(name) for name in joint_limits)
    robot_dof = len(joint_names)
    if robot_dof == 0:
        raise ValueError("The visualized robot has no actuated joints.")

    if qpos_to_viser_joint_indices is None:
        source_indices = np.arange(robot_dof, dtype=int)
    else:
        source_indices = np.asarray(qpos_to_viser_joint_indices, dtype=int)
        if source_indices.shape != (robot_dof,):
            raise ValueError(f"qpos_to_viser_joint_indices must have shape ({robot_dof},), got {source_indices.shape}")
        if np.any(source_indices < 0) or len(np.unique(source_indices)) != robot_dof:
            raise ValueError("qpos_to_viser_joint_indices must contain unique non-negative indices.")

    required_width = 7 + int(source_indices.max()) + 1
    if qpos_array.shape[1] < required_width:
        raise ValueError(
            f"qpos has {qpos_array.shape[1]} values, but {required_width} are required "
            "to extract the visualized robot joints."
        )

    angles = qpos_array[:, 7 + source_indices]
    if not np.all(np.isfinite(angles)):
        raise ValueError("Joint-angle trajectories must contain only finite values.")

    normalized_limits: list[tuple[float | None, float | None]] = []
    for name in joint_names:
        lower, upper = joint_limits[name]
        lower_value = None if lower is None else float(lower)
        upper_value = None if upper is None else float(upper)
        if lower_value is not None and not np.isfinite(lower_value):
            raise ValueError(f"Lower limit for {name} must be finite or None.")
        if upper_value is not None and not np.isfinite(upper_value):
            raise ValueError(f"Upper limit for {name} must be finite or None.")
        if lower_value is not None and upper_value is not None and lower_value > upper_value:
            raise ValueError(f"Lower limit exceeds upper limit for {name}.")
        normalized_limits.append((lower_value, upper_value))

    return JointAngleDiagnostics(
        joint_names=joint_names,
        angles=angles,
        limits=tuple(normalized_limits),
    )


def make_joint_angle_figure(
    diagnostics: JointAngleDiagnostics,
    joint_name: str,
    current_frame: int = 0,
) -> go.Figure:
    """Build one complete-sequence plot with a synchronized frame marker."""

    try:
        joint_index = diagnostics.joint_names.index(joint_name)
    except ValueError as exc:
        raise ValueError(f"Unknown joint {joint_name!r}.") from exc

    frames = np.arange(diagnostics.angles.shape[0])
    values = diagnostics.angles[:, joint_index]
    lower, upper = diagnostics.limits[joint_index]
    frame_index = int(np.clip(current_frame, 0, diagnostics.angles.shape[0] - 1))
    current_value = float(values[frame_index])
    range_values = [float(np.min(values)), float(np.max(values))]
    if lower is not None:
        range_values.append(lower)
    if upper is not None:
        range_values.append(upper)
    y_min = min(range_values)
    y_max = max(range_values)
    y_padding = max((y_max - y_min) * 0.06, 0.05)
    display_y_min = y_min - y_padding
    display_y_max = y_max + y_padding
    text_horizontal = "left" if frame_index >= diagnostics.angles.shape[0] * 0.8 else "right"
    text_vertical = "bottom" if current_value > (y_min + y_max) * 0.5 else "top"

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=frames,
            y=values,
            mode="lines",
            name="Joint angle",
            line={"color": "#06b6d4", "width": 2},
            hovertemplate="Frame %{x}<br>Angle %{y:.4f} rad<extra></extra>",
        )
    )
    if lower is not None:
        figure.add_trace(
            go.Scatter(
                x=frames,
                y=np.full(frames.shape, lower),
                mode="lines",
                name=f"Lower limit ({lower:.3f} rad)",
                line={"color": "#f59e0b", "dash": "dash", "width": 1.5},
                hovertemplate=f"Lower limit {lower:.4f} rad<extra></extra>",
            )
        )
    if upper is not None:
        figure.add_trace(
            go.Scatter(
                x=frames,
                y=np.full(frames.shape, upper),
                mode="lines",
                name=f"Upper limit ({upper:.3f} rad)",
                line={"color": "#ef4444", "dash": "dash", "width": 1.5},
                hovertemplate=f"Upper limit {upper:.4f} rad<extra></extra>",
            )
        )
    if lower is not None and upper is not None:
        figure.add_hrect(
            y0=lower,
            y1=upper,
            fillcolor="rgba(72, 229, 229, 0.06)",
            line_width=0,
            layer="below",
        )

    figure.add_trace(
        go.Scatter(
            x=(frame_index, frame_index),
            y=(display_y_min, display_y_max),
            mode="lines",
            name="Current frame",
            line={"color": "#8b5cf6", "dash": "dot", "width": 2},
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=(frame_index,),
            y=(current_value,),
            mode="markers+text",
            name="Current value",
            showlegend=False,
            marker={"color": "#8b5cf6", "size": 8, "line": {"color": "#ffffff", "width": 1}},
            text=(f"F{frame_index}: {current_value:.3f} rad",),
            textposition=f"{text_vertical} {text_horizontal}",
            textfont={"color": "#6d28d9", "size": 10},
            hovertemplate="Frame %{x}<br>Current angle %{y:.4f} rad<extra></extra>",
        )
    )

    figure.update_layout(
        hovermode="x unified",
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"color": "#1f2937", "size": 10},
        legend={
            "orientation": "h",
            "y": 1.03,
            "yanchor": "bottom",
            "x": 0.0,
            "xanchor": "left",
            "font": {"color": "#1f2937", "size": 9},
            "bgcolor": "rgba(255, 255, 255, 0.9)",
        },
        xaxis={
            "title": {"text": "Frame", "font": {"color": "#374151", "size": 10}},
            "tickfont": {"color": "#374151", "size": 9},
            "gridcolor": "#d1d5db",
            "linecolor": "#6b7280",
            "tickcolor": "#6b7280",
            "zerolinecolor": "#9ca3af",
        },
        yaxis={
            "title": {"text": "Angle (rad)", "font": {"color": "#374151", "size": 10}},
            "tickfont": {"color": "#374151", "size": 9},
            "tickformat": ".2f",
            "gridcolor": "#d1d5db",
            "linecolor": "#6b7280",
            "tickcolor": "#6b7280",
            "zerolinecolor": "#9ca3af",
            "range": (display_y_min, display_y_max),
        },
        margin={"l": 50, "r": 12, "t": 50, "b": 38},
        height=220,
    )
    return figure


def _format_current_value(
    diagnostics: JointAngleDiagnostics,
    joint_name: str,
    frame_index: int,
) -> str:
    joint_index = diagnostics.joint_names.index(joint_name)
    angle = float(diagnostics.angles[frame_index, joint_index])
    lower, upper = diagnostics.limits[joint_index]
    lower_text = "unbounded" if lower is None else f"{lower:.4f}"
    upper_text = "unbounded" if upper is None else f"{upper:.4f}"
    return (
        f"**Frame:** `{frame_index}` · **Angle:** `{angle:.4f} rad`  \n**Limits:** `[{lower_text}, {upper_text}] rad`"
    )


class JointAngleGuiController:
    """Synchronize the selected joint plot with the playback frame."""

    def __init__(self, diagnostics, joint_selector, status_handle, plot_handle) -> None:
        self.diagnostics = diagnostics
        self.joint_selector = joint_selector
        self.status_handle = status_handle
        self.plot_handle = plot_handle
        self.current_frame = 0
        self._lock = threading.Lock()

    def update_frame(self, frame_float: float) -> None:
        frame_index = int(
            np.clip(
                np.floor(float(frame_float) + 1e-9),
                0,
                self.diagnostics.angles.shape[0] - 1,
            )
        )
        with self._lock:
            if frame_index == self.current_frame:
                return
            self.current_frame = frame_index
            self._refresh_locked()

    def refresh_joint(self) -> None:
        with self._lock:
            self._refresh_locked()

    def _refresh_locked(self) -> None:
        joint_name = str(self.joint_selector.value)
        self.status_handle.content = _format_current_value(
            self.diagnostics,
            joint_name,
            self.current_frame,
        )
        self.plot_handle.figure = make_joint_angle_figure(
            self.diagnostics,
            joint_name,
            self.current_frame,
        )


def add_joint_angle_gui(gui, diagnostics: JointAngleDiagnostics) -> JointAngleGuiController:
    """Add a joint selector and an expandable Plotly trajectory to Viser."""

    initial_joint = diagnostics.joint_names[0]
    with gui.add_folder("Joint Angles", expand_by_default=True):
        joint_selector = gui.add_dropdown(
            "Joint",
            diagnostics.joint_names,
            initial_value=initial_joint,
            hint="Select one actuated joint to inspect over the complete sequence.",
        )
        status_handle = gui.add_markdown(
            _format_current_value(diagnostics, initial_joint, 0),
        )
        plot_handle = gui.add_plotly(
            make_joint_angle_figure(diagnostics, initial_joint),
            aspect=0.55,
        )

    controller = JointAngleGuiController(
        diagnostics,
        joint_selector,
        status_handle,
        plot_handle,
    )

    @joint_selector.on_update
    def _(_event) -> None:
        controller.refresh_joint()

    return controller
