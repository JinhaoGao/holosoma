#!/usr/bin/env python3
"""Synchronously compare multiple retargeting result files in Viser."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
import viser  # type: ignore[import-not-found]
import yourdfpy  # type: ignore[import-untyped]
from viser.extras import ViserUrdf  # type: ignore[import-not-found]

from holosoma_retargeting.augmentation_viser_player import (
    VariantResult,
    _resolve_object_urdf,
    _resolve_robot_urdf,
    _resolve_robot_xml,
    _rgba,
    _robot_joints_for_viser,
    _variant_color,
    interpolate_qpos,
    load_variant_result,
)
from holosoma_retargeting.src.viser_utils import (
    actuated_joint_names_from_mujoco_xml,
    build_joint_order_indices,
    create_motion_control_sliders,
)


@dataclass(frozen=True)
class AblationViserConfig:
    """Configuration for synchronized comparison of arbitrary retargeting results."""

    qpos_npzs: tuple[Path, ...]
    """Result NPZ paths in display order."""

    labels: tuple[str, ...] = ()
    """Optional display labels. Defaults to each result filename."""

    robot_urdf: Path | None = None
    """Optional robot URDF override. Otherwise resolved from result metadata."""

    robot_mujoco_xml: Path | None = None
    """Optional MuJoCo XML override used to recover qpos joint order."""

    object_urdf: Path | None = None
    """Optional object URDF override for results that contain object poses."""

    fps: float | None = None
    """Playback FPS override. Otherwise uses the FPS saved in the results."""

    loop: bool = False
    """Loop playback."""

    visual_fps_multiplier: int = 2
    """Interpolation multiplier for smooth playback."""

    x_offset: float = 0.6
    """Horizontal spacing between results. Set to zero for an exact overlay."""

    robot_mesh_opacity: float = 0.65
    """Opacity of the color-coded robot meshes."""

    object_mesh_opacity: float = 0.45
    """Opacity of object meshes when object poses are present."""

    grid_width: float = 8.0
    """Viewer grid width."""

    grid_height: float = 8.0
    """Viewer grid height."""


@dataclass
class ComparisonScene:
    """Viser handles and display state for one comparison result."""

    label: str
    result: VariantResult
    color: tuple[int, int, int]
    offset: np.ndarray
    robot: ViserUrdf
    robot_root: object
    object_visual: ViserUrdf | None
    object_root: object | None
    enabled: bool = True

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self.robot.show_visual = self.enabled
        if self.object_visual is not None:
            self.object_visual.show_visual = self.enabled


def _comparison_labels(config: AblationViserConfig) -> tuple[str, ...]:
    if len(config.qpos_npzs) < 2:
        raise ValueError("At least two result NPZ paths are required for comparison.")
    if config.labels and len(config.labels) != len(config.qpos_npzs):
        raise ValueError(f"labels has {len(config.labels)} entries for {len(config.qpos_npzs)} result paths")
    if len(set(config.qpos_npzs)) != len(config.qpos_npzs):
        raise ValueError("Comparison result paths must be unique.")
    return config.labels or tuple(path.stem for path in config.qpos_npzs)


def load_comparison_results(
    config: AblationViserConfig,
) -> tuple[tuple[str, ...], list[VariantResult]]:
    """Load arbitrary result paths and verify that their timelines are synchronized."""

    labels = _comparison_labels(config)
    results = [load_variant_result(label, path) for label, path in zip(labels, config.qpos_npzs, strict=True)]
    reference = results[0]
    incompatibilities: list[str] = []
    for result in results[1:]:
        if result.qpos.shape != reference.qpos.shape:
            incompatibilities.append(f"{result.path}: qpos shape {result.qpos.shape} != {reference.qpos.shape}")
        if not np.isclose(result.fps, reference.fps):
            incompatibilities.append(f"{result.path}: fps={result.fps} != {reference.fps}")
        if result.robot_type != reference.robot_type:
            incompatibilities.append(f"{result.path}: robot_type={result.robot_type!r} != {reference.robot_type!r}")
        if result.contains_object_in_qpos != reference.contains_object_in_qpos:
            incompatibilities.append(f"{result.path}: object qpos layout differs")
        if result.contains_object_in_qpos and result.object_name != reference.object_name:
            incompatibilities.append(f"{result.path}: object_name={result.object_name!r} != {reference.object_name!r}")
    if incompatibilities:
        formatted = "\n".join(f"  {message}" for message in incompatibilities)
        raise ValueError(f"Comparison results are not synchronized:\n{formatted}")
    return labels, results


def make_ablation_player(
    config: AblationViserConfig,
    labels: tuple[str, ...],
    results: list[VariantResult],
):
    """Build one shared-timeline Viser scene for all comparison results."""

    reference = results[0]
    robot_urdf = _resolve_robot_urdf(config, reference)
    robot_xml = _resolve_robot_xml(config, robot_urdf)
    contains_object = reference.contains_object_in_qpos
    object_urdf = _resolve_object_urdf(config, reference) if contains_object else None

    server = viser.ViserServer()
    server.scene.add_grid(
        "/grid",
        width=config.grid_width,
        height=config.grid_height,
        position=(0.0, 0.0, 0.0),
    )

    robot_urdf_model = yourdfpy.URDF.load(
        str(robot_urdf),
        load_meshes=True,
        build_scene_graph=True,
    )
    object_urdf_model = (
        yourdfpy.URDF.load(
            str(object_urdf),
            load_meshes=True,
            build_scene_graph=True,
        )
        if object_urdf is not None
        else None
    )

    scenes: list[ComparisonScene] = []
    robot_dof: int | None = None
    joint_order_indices: np.ndarray | None = None
    centered_indices = np.arange(len(results), dtype=float) - 0.5 * (len(results) - 1)

    for index, (label, result) in enumerate(zip(labels, results, strict=True)):
        color = _variant_color(label, index)
        namespace = f"/comparisons/{index:02d}_{label}"
        offset = np.asarray([centered_indices[index] * config.x_offset, 0.0, 0.0])
        robot_root = server.scene.add_frame(f"{namespace}/robot", show_axes=False)
        robot = ViserUrdf(
            server,
            urdf_or_path=robot_urdf_model,
            root_node_name=f"{namespace}/robot",
            mesh_color_override=_rgba(color, config.robot_mesh_opacity),
        )

        object_root = None
        object_visual = None
        if object_urdf_model is not None:
            object_root = server.scene.add_frame(f"{namespace}/object", show_axes=False)
            object_visual = ViserUrdf(
                server,
                urdf_or_path=object_urdf_model,
                root_node_name=f"{namespace}/object",
                mesh_color_override=_rgba(color, config.object_mesh_opacity),
            )

        current_robot_dof = len(robot.get_actuated_joint_limits())
        if robot_dof is None:
            robot_dof = current_robot_dof
            if robot_xml is not None:
                qpos_joint_names = actuated_joint_names_from_mujoco_xml(robot_xml)
                viser_joint_names = list(robot.get_actuated_joint_limits().keys())
                if qpos_joint_names != viser_joint_names:
                    joint_order_indices = build_joint_order_indices(
                        qpos_joint_names,
                        viser_joint_names,
                    )
        elif current_robot_dof != robot_dof:
            raise ValueError("Loaded comparison robots expose inconsistent actuated joint counts.")

        scenes.append(
            ComparisonScene(
                label=label,
                result=result,
                color=color,
                offset=offset,
                robot=robot,
                robot_root=robot_root,
                object_visual=object_visual,
                object_root=object_root,
            )
        )

    if robot_dof is None:
        raise RuntimeError("No comparison results were loaded.")

    last_frame = {"value": 0.0}

    def _render_scene(scene: ComparisonScene, q: np.ndarray) -> None:
        scene.robot.update_cfg(
            _robot_joints_for_viser(
                q,
                robot_dof,
                joint_order_indices,
            )
        )
        scene.robot_root.position = q[0:3] + scene.offset
        scene.robot_root.wxyz = q[3:7]
        if contains_object and scene.object_root is not None:
            scene.object_root.position = q[-7:-4] + scene.offset
            scene.object_root.wxyz = q[-4:]

    def _render_comparison(driver_q: np.ndarray, frame_float: float) -> None:
        last_frame["value"] = float(frame_float)
        for index, scene in enumerate(scenes):
            if not scene.enabled:
                continue
            q = (
                driver_q
                if index == 0
                else interpolate_qpos(
                    scene.result.qpos,
                    frame_float,
                    robot_dof,
                    contains_object=contains_object,
                )
            )
            _render_scene(scene, q)

    with server.gui.add_folder("Comparison results"):
        for scene in scenes:
            color_hex = "#" + "".join(f"{channel:02x}" for channel in scene.color)
            checkbox = server.gui.add_checkbox(
                f"{scene.label} ({color_hex})",
                initial_value=True,
            )

            def _register_visibility_callback(scene_ref: ComparisonScene, checkbox_ref) -> None:
                @checkbox_ref.on_update
                def _(_event) -> None:
                    scene_ref.set_enabled(bool(checkbox_ref.value))
                    if scene_ref.enabled:
                        q = interpolate_qpos(
                            scene_ref.result.qpos,
                            last_frame["value"],
                            robot_dof,
                            contains_object=contains_object,
                        )
                        _render_scene(scene_ref, q)

            _register_visibility_callback(scene, checkbox)

    driver = scenes[0]
    create_motion_control_sliders(
        server=server,
        viser_robot=driver.robot,
        robot_base_frame=driver.robot_root,
        motion_sequence=reference.qpos,
        robot_dof=robot_dof,
        viser_object=driver.object_visual,
        object_base_frame=driver.object_root,
        contains_object_in_qpos=contains_object,
        initial_fps=round(config.fps or reference.fps),
        initial_interp_mult=config.visual_fps_multiplier,
        loop=config.loop,
        qpos_to_viser_joint_indices=joint_order_indices,
        on_frame=_render_comparison,
    )

    print(
        f"[ablation_viser_player] Loaded {len(results)} synchronized results, "
        f"frames={reference.qpos.shape[0]}, fps={config.fps or reference.fps:.3f}"
    )
    for scene in scenes:
        print(f"  {scene.label}: rgb={scene.color}, path={scene.result.path}")
    print("Open the viewer URL printed above. Close the process (Ctrl+C) to exit.")
    return server


def main(config: AblationViserConfig) -> None:
    """Load the requested results and keep their synchronized player alive."""

    labels, results = load_comparison_results(config)
    make_ablation_player(config, labels, results)
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main(tyro.cli(AblationViserConfig))
