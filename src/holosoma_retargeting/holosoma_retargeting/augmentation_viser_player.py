#!/usr/bin/env python3
"""Overlay one retargeting result and its augmentation variants in Viser."""

from __future__ import annotations

import re
import sys
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
import viser  # type: ignore[import-not-found]
import yourdfpy  # type: ignore[import-untyped]
from viser.extras import ViserUrdf  # type: ignore[import-not-found]

src_root = Path(__file__).resolve().parent.parent
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from holosoma_retargeting.config_types.robot import RobotConfig  # noqa: E402
from holosoma_retargeting.data_utils.object_assets import get_omomo_object_asset  # noqa: E402
from holosoma_retargeting.src.viser_utils import (  # noqa: E402
    actuated_joint_names_from_mujoco_xml,
    build_joint_order_indices,
    create_motion_control_sliders,
    infer_mujoco_xml_path,
)
from holosoma_retargeting.viser_player import (  # noqa: E402
    _interpolate_sequence,
    _mapped_skeleton_edges,
)

DEFAULT_VARIANTS: tuple[str, ...] = (
    "original",
    "trans_0",
    "trans_1",
    "trans_2",
    "rot_0",
    "rot_1",
)

VARIANT_COLORS: dict[str, tuple[int, int, int]] = {
    "original": (230, 230, 230),
    "trans_0": (31, 119, 180),
    "trans_1": (255, 127, 14),
    "trans_2": (44, 160, 44),
    "rot_0": (214, 39, 40),
    "rot_1": (148, 103, 189),
}

_RESULT_SUFFIX_PATTERN = re.compile(
    r"^(?P<sequence>.+)_(?P<variant>original|augmented|trans_[0-9]+|rot_[0-9]+)$"
)


@dataclass(frozen=True)
class AugmentationViserConfig:
    """Configuration for synchronized original/augmentation comparison."""

    qpos_npz: Path
    """Any result in the augmentation family, normally ``*_original.npz``."""

    variants: tuple[str, ...] = DEFAULT_VARIANTS
    """Variant suffixes to load and display."""

    robot_urdf: Path | None = None
    """Optional robot URDF override. Otherwise resolved from result metadata."""

    robot_mujoco_xml: Path | None = None
    """Optional MuJoCo XML override used to recover qpos joint order."""

    object_urdf: Path | None = None
    """Optional object URDF override. Otherwise resolved from result metadata."""

    fps: float | None = None
    """Playback FPS override. Otherwise uses the FPS saved in the results."""

    loop: bool = False
    """Loop playback."""

    visual_fps_multiplier: int = 2
    """Interpolation multiplier for smooth playback."""

    robot_mesh_opacity: float = 0.45
    """Opacity shared by the color-coded robot meshes."""

    object_mesh_opacity: float = 0.65
    """Opacity shared by the color-coded object meshes."""

    skeleton_line_width: float = 2.0
    """Line width for human and robot skeletons."""

    grid_width: float = 8.0
    """Viewer grid width."""

    grid_height: float = 8.0
    """Viewer grid height."""


@dataclass(frozen=True)
class VariantResult:
    """Arrays and metadata required to render one augmentation variant."""

    variant: str
    path: Path
    qpos: np.ndarray
    fps: float
    human_points: np.ndarray
    robot_points: np.ndarray
    mapped_joint_names: tuple[str, ...]
    robot_type: str
    object_name: str
    object_urdf: str | None
    contains_object_in_qpos: bool


def split_result_family(path: str | Path) -> tuple[str, str | None]:
    """Return the shared sequence stem and optional augmentation suffix."""

    stem = Path(path).stem
    match = _RESULT_SUFFIX_PATTERN.fullmatch(stem)
    if match is None:
        return stem, None
    return match.group("sequence"), match.group("variant")


def discover_variant_paths(
    reference_path: str | Path,
    variants: tuple[str, ...] = DEFAULT_VARIANTS,
) -> dict[str, Path]:
    """Resolve every requested variant next to one reference result."""

    reference = Path(reference_path).expanduser()
    sequence, _ = split_result_family(reference)
    if not variants:
        raise ValueError("At least one augmentation variant must be requested.")
    if len(set(variants)) != len(variants):
        raise ValueError(f"Duplicate augmentation variants are not allowed: {variants}")

    paths = {variant: reference.parent / f"{sequence}_{variant}.npz" for variant in variants}
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(f"Missing augmentation result files:\n{formatted}")
    return paths


def _npz_scalar(data: np.lib.npyio.NpzFile, key: str, default=None):
    if key not in data:
        return default
    return np.asarray(data[key]).item()


def _npz_string_list(data: np.lib.npyio.NpzFile, key: str) -> tuple[str, ...]:
    if key not in data:
        raise KeyError(f"Result is missing required field {key!r}")
    value = np.asarray(data[key])
    if value.ndim == 0:
        return (str(value.item()),)
    return tuple(str(item) for item in value.tolist())


def load_variant_result(variant: str, path: str | Path) -> VariantResult:
    """Load one current-format retargeting result without Interaction Mesh data."""

    result_path = Path(path)
    with np.load(result_path, allow_pickle=False) as data:
        required_arrays = ("qpos", "human_joints", "mapped_robot_joints")
        missing = [key for key in required_arrays if key not in data]
        if missing:
            raise KeyError(f"{result_path} is missing required fields: {', '.join(missing)}")

        qpos = np.asarray(data["qpos"], dtype=np.float32)
        human_joints = np.asarray(data["human_joints"], dtype=np.float32)
        robot_points = np.asarray(data["mapped_robot_joints"], dtype=np.float32)
        human_joint_names = _npz_string_list(data, "human_joint_names")
        mapped_joint_names = _npz_string_list(data, "mapped_human_joint_names")
        fps = float(_npz_scalar(data, "fps", 30.0))
        robot_type = str(_npz_scalar(data, "robot_type", ""))
        object_name = str(_npz_scalar(data, "object_name", ""))
        object_urdf_value = str(_npz_scalar(data, "object_urdf", ""))
        contains_object = bool(_npz_scalar(data, "contains_object_in_qpos", False))

    if qpos.ndim != 2 or qpos.shape[0] == 0 or not np.isfinite(qpos).all():
        raise ValueError(f"{result_path} qpos must be a non-empty finite 2-D array, got {qpos.shape}")
    if human_joints.ndim != 3 or human_joints.shape[0] != qpos.shape[0] or human_joints.shape[-1] != 3:
        raise ValueError(
            f"{result_path} human_joints must have shape (frames, joints, 3) matching qpos; "
            f"got {human_joints.shape}"
        )
    if robot_points.shape != (qpos.shape[0], len(mapped_joint_names), 3):
        raise ValueError(
            f"{result_path} mapped_robot_joints must have shape "
            f"({qpos.shape[0]}, {len(mapped_joint_names)}, 3), got {robot_points.shape}"
        )
    if len(human_joint_names) != human_joints.shape[1]:
        raise ValueError(
            f"{result_path} human_joint_names has {len(human_joint_names)} entries "
            f"for {human_joints.shape[1]} joints"
        )
    if not robot_type:
        raise ValueError(f"{result_path} has no robot_type metadata")
    if not object_name:
        raise ValueError(f"{result_path} has no object_name metadata")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"{result_path} has invalid fps={fps}")

    human_index = {name: index for index, name in enumerate(human_joint_names)}
    unavailable = [name for name in mapped_joint_names if name not in human_index]
    if unavailable:
        raise ValueError(f"{result_path} mapped human joints are absent from human_joint_names: {unavailable}")
    human_points = human_joints[:, [human_index[name] for name in mapped_joint_names]]

    return VariantResult(
        variant=variant,
        path=result_path,
        qpos=qpos,
        fps=fps,
        human_points=human_points,
        robot_points=robot_points,
        mapped_joint_names=mapped_joint_names,
        robot_type=robot_type,
        object_name=object_name,
        object_urdf=object_urdf_value or None,
        contains_object_in_qpos=contains_object,
    )


def load_result_family(config: AugmentationViserConfig) -> list[VariantResult]:
    """Load and cross-check every requested augmentation result."""

    paths = discover_variant_paths(config.qpos_npz, config.variants)
    results = [load_variant_result(variant, path) for variant, path in paths.items()]
    reference = results[0]
    incompatibilities: list[str] = []
    for result in results[1:]:
        if result.qpos.shape != reference.qpos.shape:
            incompatibilities.append(
                f"{result.path.name}: qpos shape {result.qpos.shape} != {reference.qpos.shape}"
            )
        if result.mapped_joint_names != reference.mapped_joint_names:
            incompatibilities.append(f"{result.path.name}: mapped joint names differ")
        if result.robot_type != reference.robot_type:
            incompatibilities.append(
                f"{result.path.name}: robot_type={result.robot_type!r} != {reference.robot_type!r}"
            )
        if result.object_name != reference.object_name:
            incompatibilities.append(
                f"{result.path.name}: object_name={result.object_name!r} != {reference.object_name!r}"
            )
        if result.contains_object_in_qpos != reference.contains_object_in_qpos:
            incompatibilities.append(f"{result.path.name}: object qpos layout differs")
        if not np.isclose(result.fps, reference.fps):
            incompatibilities.append(f"{result.path.name}: fps={result.fps} != {reference.fps}")
    if incompatibilities:
        formatted = "\n".join(f"  {message}" for message in incompatibilities)
        raise ValueError(f"Augmentation results are not synchronized:\n{formatted}")
    if not reference.contains_object_in_qpos:
        raise ValueError("Augmentation comparison requires object poses in qpos.")
    return results


def _resolve_asset_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    package_root = Path(__file__).resolve().parent
    package_candidate = package_root / path
    return package_candidate if package_candidate.exists() else path


def _resolve_robot_urdf(config: AugmentationViserConfig, result: VariantResult) -> Path:
    value = config.robot_urdf or RobotConfig(robot_type=result.robot_type).ROBOT_URDF_FILE
    path = _resolve_asset_path(value)
    if not path.is_file():
        raise FileNotFoundError(f"Robot URDF not found: {path}")
    return path


def _resolve_object_urdf(config: AugmentationViserConfig, result: VariantResult) -> Path:
    value = config.object_urdf or result.object_urdf
    if value:
        path = _resolve_asset_path(value)
    else:
        path = get_omomo_object_asset(result.object_name).urdf_path
    if not path.is_file():
        raise FileNotFoundError(f"Object URDF not found: {path}")
    return path


def _resolve_robot_xml(config: AugmentationViserConfig, robot_urdf: Path) -> Path | None:
    if config.robot_mujoco_xml is not None:
        path = _resolve_asset_path(config.robot_mujoco_xml)
        if not path.is_file():
            raise FileNotFoundError(f"Robot MuJoCo XML not found: {path}")
        return path
    return infer_mujoco_xml_path(robot_urdf)


def _variant_color(variant: str, index: int) -> tuple[int, int, int]:
    if variant in VARIANT_COLORS:
        return VARIANT_COLORS[variant]
    fallback = (
        (23, 190, 207),
        (227, 119, 194),
        (188, 189, 34),
        (127, 127, 127),
    )
    return fallback[index % len(fallback)]


def _rgba(color: tuple[int, int, int], opacity: float) -> tuple[float, float, float, float]:
    alpha = float(np.clip(opacity, 0.0, 1.0))
    return tuple(channel / 255.0 for channel in color) + (alpha,)


def _slerp(q0: np.ndarray, q1: np.ndarray, fraction: float) -> np.ndarray:
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    q0 /= max(float(np.linalg.norm(q0)), 1e-12)
    q1 /= max(float(np.linalg.norm(q1)), 1e-12)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        result = q0 + fraction * (q1 - q0)
        return result / max(float(np.linalg.norm(result)), 1e-12)
    theta = float(np.arccos(np.clip(dot, -1.0, 1.0)))
    return (
        np.sin((1.0 - fraction) * theta) * q0
        + np.sin(fraction * theta) * q1
    ) / np.sin(theta)


def interpolate_qpos(
    qpos: np.ndarray,
    frame_float: float,
    robot_dof: int,
    *,
    contains_object: bool,
) -> np.ndarray:
    """Interpolate one MuJoCo-order qpos frame with quaternion SLERP."""

    frame_float = float(np.clip(frame_float, 0.0, qpos.shape[0] - 1))
    i0 = int(np.floor(frame_float))
    i1 = min(i0 + 1, qpos.shape[0] - 1)
    fraction = frame_float - i0
    if i0 == i1:
        return qpos[i0].copy()

    q0 = qpos[i0]
    q1 = qpos[i1]
    result = (1.0 - fraction) * q0 + fraction * q1
    result[3:7] = _slerp(q0[3:7], q1[3:7], fraction)
    if contains_object:
        result[-4:] = _slerp(q0[-4:], q1[-4:], fraction)
    expected_minimum = 7 + robot_dof + (7 if contains_object else 0)
    if result.shape[0] < expected_minimum:
        raise ValueError(
            f"qpos has {result.shape[0]} values, expected at least {expected_minimum} "
            f"for robot_dof={robot_dof}"
        )
    return result


class SkeletonOverlay:
    """Draw synchronized color-coded human and robot mapped skeletons."""

    def __init__(
        self,
        server: viser.ViserServer,
        namespace: str,
        result: VariantResult,
        color: tuple[int, int, int],
        line_width: float,
    ) -> None:
        self.server = server
        self.namespace = namespace
        self.result = result
        self.color = np.asarray(color, dtype=np.float32) / 255.0
        self.human_color = 0.45 * self.color + 0.55
        self.edges = _mapped_skeleton_edges(list(result.mapped_joint_names))
        self.line_width = float(line_width)
        self.visible = True
        self._lock = threading.Lock()
        self._handles: list[object] = []

    def set_visible(self, visible: bool) -> None:
        with self._lock:
            self.visible = bool(visible)
            if not self.visible:
                self._clear_locked()

    def draw(self, frame_float: float) -> None:
        with self._lock:
            if not self.visible:
                return
            self._clear_locked()
            human_points = _interpolate_sequence(self.result.human_points, frame_float)
            robot_points = _interpolate_sequence(self.result.robot_points, frame_float)
            human_handle = self._draw_skeleton(
                f"{self.namespace}/skeleton/human",
                human_points,
                self.human_color,
            )
            robot_handle = self._draw_skeleton(
                f"{self.namespace}/skeleton/robot",
                robot_points,
                self.color,
            )
            if human_handle is not None:
                self._handles.append(human_handle)
            if robot_handle is not None:
                self._handles.append(robot_handle)

    def _clear_locked(self) -> None:
        for handle in self._handles:
            with suppress(Exception):
                handle.remove()
        self._handles.clear()

    def _draw_skeleton(self, name: str, points: np.ndarray, color: np.ndarray):
        if not self.edges:
            return None
        segments = np.asarray([[points[i], points[j]] for i, j in self.edges], dtype=np.float32)
        colors = np.tile(color.reshape(1, 1, 3), (segments.shape[0], 2, 1))
        return self.server.scene.add_line_segments(
            name,
            points=segments,
            colors=colors,
            line_width=self.line_width,
        )


@dataclass
class VariantScene:
    """Viser scene handles for one result variant."""

    result: VariantResult
    color: tuple[int, int, int]
    robot: ViserUrdf
    robot_root: object
    object_visual: ViserUrdf
    object_root: object
    skeleton: SkeletonOverlay
    enabled: bool = True

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self.robot.show_visual = self.enabled
        self.object_visual.show_visual = self.enabled
        self.skeleton.set_visible(self.enabled)


def _robot_joints_for_viser(
    q: np.ndarray,
    robot_dof: int,
    joint_order_indices: np.ndarray | None,
) -> np.ndarray:
    if joint_order_indices is None:
        return np.asarray(q[7 : 7 + robot_dof])
    required_length = int(joint_order_indices.max()) + 1
    raw_joints = np.asarray(q[7 : 7 + required_length])
    if raw_joints.shape[0] < required_length:
        raise ValueError(
            f"qpos only has {raw_joints.shape[0]} robot joints; {required_length} are required"
        )
    return raw_joints[joint_order_indices]


def make_augmentation_player(config: AugmentationViserConfig, results: list[VariantResult]):
    """Build a synchronized color-coded Viser scene for a result family."""

    reference = results[0]
    robot_urdf = _resolve_robot_urdf(config, reference)
    object_urdf = _resolve_object_urdf(config, reference)
    robot_xml = _resolve_robot_xml(config, robot_urdf)

    server = viser.ViserServer()
    server.scene.add_grid(
        "/grid",
        width=config.grid_width,
        height=config.grid_height,
        position=(0.0, 0.0, 0.0),
    )

    robot_urdf_model = yourdfpy.URDF.load(str(robot_urdf), load_meshes=True, build_scene_graph=True)
    object_urdf_model = yourdfpy.URDF.load(str(object_urdf), load_meshes=True, build_scene_graph=True)
    scenes: list[VariantScene] = []
    robot_dof: int | None = None
    joint_order_indices: np.ndarray | None = None

    for index, result in enumerate(results):
        color = _variant_color(result.variant, index)
        namespace = f"/variants/{result.variant}"
        robot_root = server.scene.add_frame(f"{namespace}/robot", show_axes=False)
        object_root = server.scene.add_frame(f"{namespace}/object", show_axes=False)
        robot = ViserUrdf(
            server,
            urdf_or_path=robot_urdf_model,
            root_node_name=f"{namespace}/robot",
            mesh_color_override=_rgba(color, config.robot_mesh_opacity),
        )
        object_visual = ViserUrdf(
            server,
            urdf_or_path=object_urdf_model,
            root_node_name=f"{namespace}/object",
            mesh_color_override=_rgba(color, config.object_mesh_opacity),
        )
        current_joint_limits = robot.get_actuated_joint_limits()
        if robot_dof is None:
            robot_dof = len(current_joint_limits)
            if robot_xml is not None:
                qpos_joint_names = actuated_joint_names_from_mujoco_xml(robot_xml)
                viser_joint_names = list(current_joint_limits.keys())
                if qpos_joint_names != viser_joint_names:
                    joint_order_indices = build_joint_order_indices(
                        qpos_joint_names,
                        viser_joint_names,
                    )
        elif len(current_joint_limits) != robot_dof:
            raise ValueError("Loaded robot variants expose inconsistent actuated joint counts.")

        scenes.append(
            VariantScene(
                result=result,
                color=color,
                robot=robot,
                robot_root=robot_root,
                object_visual=object_visual,
                object_root=object_root,
                skeleton=SkeletonOverlay(
                    server,
                    namespace,
                    result,
                    color,
                    config.skeleton_line_width,
                ),
            )
        )

    if robot_dof is None:
        raise RuntimeError("No augmentation variants were loaded.")

    last_frame = {"value": 0.0}

    def _render_scene(scene: VariantScene, q: np.ndarray, frame_float: float) -> None:
        scene.robot.update_cfg(_robot_joints_for_viser(q, robot_dof, joint_order_indices))
        scene.robot_root.position = q[0:3]
        scene.robot_root.wxyz = q[3:7]
        scene.object_root.position = q[-7:-4]
        scene.object_root.wxyz = q[-4:]
        scene.skeleton.draw(frame_float)

    def _render_family(driver_q: np.ndarray, frame_float: float) -> None:
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
                    contains_object=True,
                )
            )
            if index == 0:
                scene.skeleton.draw(frame_float)
            else:
                _render_scene(scene, q, frame_float)

    with server.gui.add_folder("Augmentation series"):
        for scene in scenes:
            color_hex = "#" + "".join(f"{channel:02x}" for channel in scene.color)
            checkbox = server.gui.add_checkbox(
                f"{scene.result.variant} ({color_hex})",
                initial_value=True,
            )

            def _register_visibility_callback(scene_ref: VariantScene, checkbox_ref) -> None:
                @checkbox_ref.on_update
                def _(_event) -> None:
                    scene_ref.set_enabled(bool(checkbox_ref.value))
                    if scene_ref.enabled:
                        q = interpolate_qpos(
                            scene_ref.result.qpos,
                            last_frame["value"],
                            robot_dof,
                            contains_object=True,
                        )
                        _render_scene(scene_ref, q, last_frame["value"])

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
        contains_object_in_qpos=True,
        initial_fps=config.fps or reference.fps,
        initial_interp_mult=config.visual_fps_multiplier,
        loop=config.loop,
        qpos_to_viser_joint_indices=joint_order_indices,
        on_frame=_render_family,
    )

    sequence, _ = split_result_family(reference.path)
    print(
        f"[augmentation_viser_player] Loaded sequence={sequence}, "
        f"variants={len(results)}, frames={reference.qpos.shape[0]}"
    )
    for scene in scenes:
        print(
            f"  {scene.result.variant}: rgb={scene.color}, "
            f"path={scene.result.path}"
        )
    print("Open the viewer URL printed above. Close the process (Ctrl+C) to exit.")
    return server


def main(config: AugmentationViserConfig) -> None:
    """Load a result family and keep its synchronized Viser player alive."""

    results = load_result_family(config)
    make_augmentation_player(config, results)
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main(tyro.cli(AugmentationViserConfig))
