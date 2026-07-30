"""Canonical visualization-layer names and shared Viser GUI controls."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Callable, Iterable

from holosoma_retargeting.src.viser_utils import register_keyboard_shortcut


class LayerId(StrEnum):
    """Stable identifiers shared by single- and multi-motion viewers."""

    ROBOT_MESH = "robot_mesh"
    OBJECT_MESH = "object_mesh"
    HUMAN_SKELETON = "human_skeleton"
    ROBOT_SKELETON = "robot_skeleton"
    HUMAN_HANDS = "human_hands"
    OBJECT_KEYPOINTS = "object_keypoints"
    INTERACTION_MESH = "interaction_mesh"
    FOOT_STICKING = "foot_sticking"
    SOURCE_ORIENTATION = "source_orientation"
    TARGET_ORIENTATION = "target_orientation"
    ROBOT_ORIENTATION = "robot_orientation"
    JOINT_LABELS = "joint_labels"
    BODY_COM = "body_com"
    BODY_VELOCITY = "body_velocity"


@dataclass(frozen=True)
class LayerSpec:
    """User-facing metadata for one independently controllable layer."""

    layer_id: LayerId
    group: str
    label: str
    hint: str
    hotkey: str | None = None


LAYER_SPECS: tuple[LayerSpec, ...] = (
    LayerSpec(
        LayerId.ROBOT_MESH,
        "Mesh",
        "Robot mesh",
        "Robot URDF visual geometry.",
        ",",
    ),
    LayerSpec(
        LayerId.OBJECT_MESH,
        "Mesh",
        "Object / scene mesh",
        "Dynamic object, terrain, or source-scene geometry.",
        "o",
    ),
    LayerSpec(
        LayerId.HUMAN_SKELETON,
        "Skeleton",
        "Human skeleton",
        "Source-human keypoints and connecting bones.",
        "h",
    ),
    LayerSpec(
        LayerId.ROBOT_SKELETON,
        "Skeleton",
        "Robot skeleton",
        "Mapped robot link keypoints and connecting bones.",
        ".",
    ),
    LayerSpec(
        LayerId.HUMAN_HANDS,
        "Skeleton",
        "Human hand details",
        "Visual-only finger keypoints and chains when saved source joints support them.",
    ),
    LayerSpec(
        LayerId.OBJECT_KEYPOINTS,
        "Interaction",
        "Object keypoints",
        "Saved demonstration-scale and target-scale object samples.",
        "k",
    ),
    LayerSpec(
        LayerId.INTERACTION_MESH,
        "Interaction",
        "Interaction mesh",
        "Saved source and target interaction-mesh edges.",
        "m",
    ),
    LayerSpec(
        LayerId.FOOT_STICKING,
        "Interaction",
        "Foot sticking",
        "Per-frame left/right sticking detector and constraint status.",
        "f",
    ),
    LayerSpec(
        LayerId.SOURCE_ORIENTATION,
        "Diagnostics",
        "Source orientation axes",
        "Global source-human joint coordinate frames.",
        "r",
    ),
    LayerSpec(
        LayerId.TARGET_ORIENTATION,
        "Diagnostics",
        "Target orientation axes",
        "Calibrated target link frames saved by orientation tracking.",
        "t",
    ),
    LayerSpec(
        LayerId.ROBOT_ORIENTATION,
        "Diagnostics",
        "Robot orientation axes",
        "Actual robot link frames saved by orientation tracking.",
        "a",
    ),
    LayerSpec(
        LayerId.JOINT_LABELS,
        "Diagnostics",
        "Joint names",
        "Source joint-name labels, when a complete source skeleton is available.",
        "l",
    ),
    LayerSpec(
        LayerId.BODY_COM,
        "Diagnostics",
        "Body centers",
        "Saved rigid-body center-of-mass positions.",
    ),
    LayerSpec(
        LayerId.BODY_VELOCITY,
        "Diagnostics",
        "Body velocity",
        "Saved rigid-body linear-velocity vectors.",
        "v",
    ),
)

_SPEC_BY_ID = {spec.layer_id: spec for spec in LAYER_SPECS}


@dataclass
class _LayerState:
    available: bool
    visible: bool
    callback: Callable[[bool], None]
    unavailable_reason: str | None
    checkbox: object | None = None
    updating: bool = False


class LayerController:
    """Own availability, visibility, callbacks, and GUI state for all layers."""

    def __init__(self) -> None:
        self._states: dict[LayerId, _LayerState] = {}

    def register(
        self,
        layer_id: LayerId,
        *,
        available: bool,
        visible: bool,
        callback: Callable[[bool], None],
        unavailable_reason: str | None = None,
    ) -> None:
        """Register a layer before building controls."""

        if layer_id in self._states:
            raise ValueError(f"Layer {layer_id.value!r} was registered twice")
        initial_visible = bool(visible) and bool(available)
        self._states[layer_id] = _LayerState(
            available=bool(available),
            visible=initial_visible,
            callback=callback,
            unavailable_reason=unavailable_reason,
        )
        if available:
            callback(initial_visible)

    def is_available(self, layer_id: LayerId) -> bool:
        state = self._states.get(layer_id)
        return bool(state and state.available)

    def is_visible(self, layer_id: LayerId) -> bool:
        state = self._states.get(layer_id)
        return bool(state and state.visible)

    def set_visible(
        self,
        layer_id: LayerId,
        visible: bool,
        *,
        sync_checkbox: bool = True,
    ) -> None:
        state = self._states[layer_id]
        if not state.available:
            return
        state.visible = bool(visible)
        state.callback(state.visible)
        if sync_checkbox and state.checkbox is not None:
            state.updating = True
            try:
                state.checkbox.value = state.visible
            finally:
                state.updating = False

    def toggle(self, layer_id: LayerId) -> None:
        if self.is_available(layer_id):
            self.set_visible(layer_id, not self.is_visible(layer_id))

    def add_gui(self, gui, *, layer_ids: Iterable[LayerId] | None = None) -> None:
        """Populate the current Viser container with canonical layer controls."""

        selected = set(layer_ids) if layer_ids is not None else set(self._states)
        for group in ("Mesh", "Skeleton", "Interaction", "Diagnostics"):
            specs = [
                spec
                for spec in LAYER_SPECS
                if spec.group == group and spec.layer_id in selected and spec.layer_id in self._states
            ]
            if not specs:
                continue
            with gui.add_folder(group, expand_by_default=True):
                for spec in specs:
                    state = self._states[spec.layer_id]
                    hint = spec.hint
                    if not state.available:
                        reason = state.unavailable_reason or "Required data is not present in this input."
                        hint = f"{hint} Unavailable: {reason}"
                    checkbox = gui.add_checkbox(
                        spec.label,
                        initial_value=state.visible,
                        disabled=not state.available,
                        hint=hint,
                    )
                    state.checkbox = checkbox

                    def _register(layer_id: LayerId, checkbox_ref) -> None:
                        @checkbox_ref.on_update
                        def _(_event) -> None:
                            current = self._states[layer_id]
                            if not current.updating:
                                self.set_visible(
                                    layer_id,
                                    bool(checkbox_ref.value),
                                    sync_checkbox=False,
                                )

                    _register(spec.layer_id, checkbox)

    def register_shortcuts(self, server) -> None:
        """Register canonical keyboard shortcuts for available layers."""

        for layer_id, state in self._states.items():
            spec = _SPEC_BY_ID[layer_id]
            if not state.available or spec.hotkey is None:
                continue
            register_keyboard_shortcut(
                server,
                f"Display: Toggle {spec.label}",
                hotkey=spec.hotkey,
                callback=lambda layer_id_ref=layer_id: self.toggle(layer_id_ref),
            )


@dataclass(frozen=True)
class VisualizationTabs:
    """Reusable tab handles shared by all public motion viewers."""

    playback: object
    layers: object
    motions: object | None
    style: object


def add_visualization_tabs(gui, *, include_motions: bool) -> VisualizationTabs:
    """Add the canonical top-level tab layout."""

    tab_group = gui.add_tab_group()
    playback = tab_group.add_tab("Playback")
    layers = tab_group.add_tab("Layers")
    motions = tab_group.add_tab("Motions") if include_motions else None
    style = tab_group.add_tab("Style")
    return VisualizationTabs(
        playback=playback,
        layers=layers,
        motions=motions,
        style=style,
    )
