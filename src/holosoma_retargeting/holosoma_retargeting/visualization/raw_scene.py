# ruff: noqa: CPY001
"""Load and render source human motion without retargeting preprocessing.

This viewer goes through the repository's registered motion adapters so that
all supported source formats share one command, but deliberately does not call
``preprocess_motion_data``.  Consequently it applies no robot-height scale,
foot-height translation, object augmentation, or retargeting.

This internal adapter is selected by ``viser_player.py --input-kind raw`` or
by automatic input inspection.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import trimesh
import viser

from holosoma_retargeting.config_types.data_type import (
    DEMO_JOINTS_REGISTRY,
    normalize_data_format,
)
from holosoma_retargeting.data_utils.motion_data import (
    HumanMotion,
    get_motion_format_spec,
    load_human_motion,
    validate_motion_task,
)
from holosoma_retargeting.data_utils.object_assets import get_omomo_object_asset
from holosoma_retargeting.data_utils.omomo import parse_omomo_sequence_name
from holosoma_retargeting.src.viser_utils import register_keyboard_shortcut
from holosoma_retargeting.visualization.layers import (
    LayerController,
    LayerId,
    add_visualization_tabs,
)

TaskType = Literal["auto", "robot_only", "object_interaction", "climbing"]


@dataclass(frozen=True)
class Config:
    """Command-line options for the raw motion viewer."""

    motion_path: Path
    """Motion file, one climbing task directory, or a dataset directory."""

    data_format: str = "auto"
    """Source adapter, or ``auto`` to infer it from the selected file."""

    task_type: TaskType = "auto"
    """Task context. ``auto`` detects climbing folders and OMOMO objects."""

    sequence: str | None = None
    """Sequence stem when ``motion_path`` is a dataset directory."""

    human_height: float | None = None
    """Optional height override required by source datasets lacking metadata."""

    mesh_path: Path | None = None
    """Optional object/terrain mesh override. No scale or offset is applied."""

    object_name: str | None = None
    """Optional object category override used for mesh lookup."""

    port: int = 8080
    """Viser web-server port."""

    start_frame: int = 0
    """Initial frame index."""

    playing: bool = True
    """Start playback immediately."""

    loop: bool = True
    """Loop at the end of the sequence."""

    point_size: float = 0.03
    """Human keypoint size in scene units."""

    line_width: float = 5.0
    """Skeleton line width."""

    mesh_opacity: float = 0.65
    """Object or terrain mesh opacity."""

    show_joint_labels: bool = False
    """Show every source joint name beside its keypoint."""

    show_joint_orientations: bool = False
    """Show source global joint coordinate frames when orientations are available."""

    orientation_axis_length: float = 0.09
    """Length of source joint orientation axes in scene units."""

    dry_run: bool = False
    """Validate and summarize the raw scene without starting Viser."""


@dataclass(frozen=True)
class RawScene:
    """Everything needed to render one unprocessed source sequence."""

    motion: HumanMotion
    data_format: str
    task_type: str
    joint_names: tuple[str, ...]
    skeleton_edges: tuple[tuple[int, int], ...]
    mesh_path: Path | None
    mesh: trimesh.Trimesh | None
    object_poses_wxyz_xyz: np.ndarray | None


def _candidate_motion_files(directory: Path) -> list[Path]:
    direct = [
        path for suffix in ("*.pt", "*.npy", "*.npz") for path in sorted(directory.glob(suffix)) if path.is_file()
    ]
    nested = [
        path
        for suffix in ("*.pt", "*.npy", "*.npz")
        for path in sorted(directory.glob(f"*/{suffix}"))
        if path.is_file()
    ]
    return sorted(set(direct + nested))


def resolve_motion_file(motion_path: Path, sequence: str | None) -> Path:
    """Resolve a CLI path to exactly one registered source-motion file."""

    path = motion_path.expanduser().resolve()
    if path.is_file():
        if sequence is not None:
            raise ValueError("--sequence cannot be used when motion_path is already a file")
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Motion path does not exist: {path}")

    candidates = _candidate_motion_files(path)
    if sequence is not None:
        candidates = [candidate for candidate in candidates if sequence in {candidate.stem, candidate.parent.name}]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        suffix = f" for sequence {sequence!r}" if sequence is not None else ""
        raise FileNotFoundError(f"No supported motion file found under {path}{suffix}")

    preview = "\n".join(f"  {candidate}" for candidate in candidates[:12])
    remainder = len(candidates) - 12
    if remainder > 0:
        preview += f"\n  ... and {remainder} more"
    raise ValueError(f"Motion path is ambiguous; pass an exact file or --sequence. Candidates:\n{preview}")


def infer_data_format(source_path: Path) -> str:
    """Infer one canonical adapter without relying on the task name."""

    suffix = source_path.suffix.lower()
    if suffix == ".pt":
        return "omomo"
    if suffix == ".npy":
        array = np.load(source_path, mmap_mode="r", allow_pickle=False)
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError(f"Cannot infer data format from NPY shape {array.shape}: {source_path}")
        if array.shape[1] == len(DEMO_JOINTS_REGISTRY["mocap"]):
            return "mocap"
        if array.shape[1] == len(DEMO_JOINTS_REGISTRY["lafan"]):
            return "lafan"
        raise ValueError(f"No registered skeleton has {array.shape[1]} joints: {source_path}")
    if suffix != ".npz":
        raise ValueError(f"Unsupported motion suffix {suffix!r}: {source_path}")

    with np.load(source_path, allow_pickle=False) as data:
        fields = set(data.files)
        if "source_format" in fields and str(np.asarray(data["source_format"]).item()).lower() == "gvhmr":
            return "gvhmr"
        if {"raw_joint_names", "source_bvh"}.intersection(fields):
            return "noetix_mocap"
        if "joint_names" in fields:
            names = [str(name) for name in np.asarray(data["joint_names"]).tolist()]
            if names == DEMO_JOINTS_REGISTRY["noetix_mocap"]:
                return "noetix_mocap"
        if "root_quaternions_wxyz" in fields:
            return "gvhmr"
        if "global_joint_positions" in fields:
            return "amass"
    raise ValueError(f"Cannot infer a registered NPZ format from fields in {source_path}")


def _load_exact_source(
    source_path: Path,
    data_format: str,
    human_height: float | None,
) -> HumanMotion:
    """Load an exact source file through the public adapter interface."""

    spec = get_motion_format_spec(data_format)
    if spec.nested_files:
        data_dir = source_path.parent.parent
        task_name = source_path.parent.name
    else:
        data_dir = source_path.parent
        task_name = source_path.stem
    motion = load_human_motion(
        data_format,
        data_dir,
        task_name,
        human_height=human_height,
    )
    if motion.source_path.resolve() != source_path.resolve():
        raise RuntimeError(f"Motion adapter resolved a different source: {motion.source_path} != {source_path}")
    return motion


def infer_task_type(
    requested: TaskType,
    data_format: str,
    source_path: Path,
    motion: HumanMotion,
) -> str:
    """Resolve task context while keeping robot-only OMOMO selectable."""

    if requested != "auto":
        validate_motion_task(data_format, requested)
        return requested
    if data_format == "mocap" and any(
        (source_path.parent / filename).is_file()
        for filename in ("multi_boxes.obj", "multi_boxes.urdf", "box_assets.xml")
    ):
        return "climbing"
    if data_format == "omomo" and motion.object_poses_wxyz_xyz is not None:
        return "object_interaction"
    return "robot_only"


def _add_chain(
    pairs: list[tuple[str, str]],
    names: tuple[str, ...],
) -> None:
    pairs.extend(zip(names[:-1], names[1:]))


def build_skeleton_edges(joint_names: tuple[str, ...]) -> tuple[tuple[int, int], ...]:
    """Build the complete named topology for every registered skeleton."""

    pairs: list[tuple[str, str]] = []
    names = set(joint_names)

    if {"Pelvis", "L_Thorax", "R_Thorax"}.issubset(names):
        _add_chain(pairs, ("Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe"))
        _add_chain(pairs, ("Pelvis", "R_Hip", "R_Knee", "R_Ankle", "R_Toe"))
        _add_chain(pairs, ("Pelvis", "Torso", "Spine", "Chest", "Neck", "Head"))
        _add_chain(pairs, ("Chest", "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist"))
        _add_chain(pairs, ("Chest", "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist"))
        for side in ("L", "R"):
            for finger in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
                _add_chain(
                    pairs,
                    (
                        f"{side}_Wrist",
                        f"{side}_{finger}1",
                        f"{side}_{finger}2",
                        f"{side}_{finger}3",
                    ),
                )
    elif {"Pelvis", "L_Collar", "R_Collar"}.issubset(names):
        _add_chain(pairs, ("Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Foot"))
        _add_chain(pairs, ("Pelvis", "R_Hip", "R_Knee", "R_Ankle", "R_Foot"))
        _add_chain(pairs, ("Pelvis", "Spine1", "Spine2", "Spine3", "Neck", "Head"))
        _add_chain(pairs, ("Spine3", "L_Collar", "L_Shoulder", "L_Elbow", "L_Wrist"))
        _add_chain(pairs, ("Spine3", "R_Collar", "R_Shoulder", "R_Elbow", "R_Wrist"))
    elif "LeftHandThumb1" in names:
        _add_chain(pairs, ("Hips", "Spine", "Spine1", "Neck", "Head"))
        _add_chain(pairs, ("Spine1", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand"))
        _add_chain(pairs, ("Spine1", "RightShoulder", "RightArm", "RightForeArm", "RightHand"))
        _add_chain(pairs, ("Hips", "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase"))
        _add_chain(pairs, ("Hips", "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase"))
        pairs.extend((("LeftFoot", "LeftFootMod"), ("RightFoot", "RightFootMod")))
        for side in ("Left", "Right"):
            for finger in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
                _add_chain(
                    pairs,
                    (
                        f"{side}Hand",
                        f"{side}Hand{finger}1",
                        f"{side}Hand{finger}2",
                        f"{side}Hand{finger}3",
                    ),
                )
    elif {"Hips", "Spine2", "LeftToeBase", "RightToeBase"}.issubset(names):
        _add_chain(pairs, ("Hips", "Spine", "Spine1", "Spine2", "Neck", "Head"))
        _add_chain(pairs, ("Spine2", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand"))
        _add_chain(pairs, ("Spine2", "RightShoulder", "RightArm", "RightForeArm", "RightHand"))
        _add_chain(pairs, ("Hips", "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase"))
        _add_chain(pairs, ("Hips", "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase"))
    else:
        raise ValueError(f"No visualization topology registered for joints: {joint_names}")

    joint_index = {name: index for index, name in enumerate(joint_names)}
    edges = {
        (joint_index[parent], joint_index[child])
        for parent, child in pairs
        if parent in joint_index and child in joint_index
    }
    covered = {index for edge in edges for index in edge}
    missing = [joint_names[index] for index in range(len(joint_names)) if index not in covered]
    if missing:
        raise ValueError(f"Skeleton topology does not connect all source joints: {missing}")
    return tuple(sorted(edges))


def resolve_mesh_path(
    explicit_mesh: Path | None,
    object_name: str | None,
    task_type: str,
    data_format: str,
    source_path: Path,
) -> Path | None:
    """Find the source object's unscaled mesh, if this task has one."""

    if explicit_mesh is not None:
        path = explicit_mesh.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Mesh file not found: {path}")
        return path
    if task_type == "robot_only":
        return None
    if data_format == "omomo":
        sequence_object = parse_omomo_sequence_name(
            source_path,
            require_known_object=False,
        ).object_name
        resolved_object = object_name or sequence_object
        if object_name is not None and object_name != sequence_object:
            raise ValueError(f"Sequence contains object {sequence_object!r}, but --object-name is {object_name!r}")
        return get_omomo_object_asset(resolved_object).mesh_path.resolve()

    task_dir = source_path.parent
    preferred_names = [
        name
        for name in (
            f"{object_name}.obj" if object_name else None,
            "multi_boxes.obj",
        )
        if name is not None
    ]
    for filename in preferred_names:
        candidate = task_dir / filename
        if candidate.is_file():
            return candidate.resolve()

    top_level_meshes = sorted(task_dir.glob("*.obj"))
    if len(top_level_meshes) == 1:
        return top_level_meshes[0].resolve()
    return None


def load_mesh(mesh_path: Path | None) -> trimesh.Trimesh | None:
    """Load one OBJ/PLY/STL scene as a single visual mesh."""

    if mesh_path is None:
        return None
    loaded = trimesh.load(mesh_path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        mesh = loaded.to_mesh()
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise ValueError(f"Mesh loader returned unsupported type {type(loaded).__name__}: {mesh_path}")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"Mesh has no triangles: {mesh_path}")
    return mesh


def load_raw_scene(cfg: Config) -> RawScene:
    """Resolve and load one raw human/object scene without spatial preprocessing."""

    source_path = resolve_motion_file(cfg.motion_path, cfg.sequence)
    data_format = (
        infer_data_format(source_path) if cfg.data_format == "auto" else normalize_data_format(cfg.data_format)
    )
    motion = _load_exact_source(source_path, data_format, cfg.human_height)
    task_type = infer_task_type(cfg.task_type, data_format, source_path, motion)
    joint_names = tuple(DEMO_JOINTS_REGISTRY[data_format])
    edges = build_skeleton_edges(joint_names)
    mesh_path = resolve_mesh_path(
        cfg.mesh_path,
        cfg.object_name,
        task_type,
        data_format,
        source_path,
    )
    mesh = load_mesh(mesh_path)
    object_poses = motion.object_poses_wxyz_xyz if task_type == "object_interaction" else None
    return RawScene(
        motion=motion,
        data_format=data_format,
        task_type=task_type,
        joint_names=joint_names,
        skeleton_edges=edges,
        mesh_path=mesh_path,
        mesh=mesh,
        object_poses_wxyz_xyz=object_poses,
    )


def _joint_colors(
    joint_names: tuple[str, ...],
    *,
    for_lines: bool = False,
) -> np.ndarray:
    if for_lines:
        left_color = (205, 0, 20)
        right_color = (0, 65, 205)
        center_color = (120, 0, 175)
    else:
        left_color = (255, 35, 45)
        right_color = (0, 120, 255)
        center_color = (190, 30, 255)

    colors = np.empty((len(joint_names), 3), dtype=np.uint8)
    for index, name in enumerate(joint_names):
        if name.startswith(("L_", "Left")):
            colors[index] = left_color
        elif name.startswith(("R_", "Right")):
            colors[index] = right_color
        else:
            colors[index] = center_color
    return colors


def _edge_segments(points: np.ndarray, edges: tuple[tuple[int, int], ...]) -> np.ndarray:
    return np.asarray([[points[start], points[end]] for start, end in edges], dtype=np.float32)


def _edge_colors(
    point_colors: np.ndarray,
    edges: tuple[tuple[int, int], ...],
) -> np.ndarray:
    return np.asarray(
        [[point_colors[start], point_colors[end]] for start, end in edges],
        dtype=np.uint8,
    )


def print_summary(scene: RawScene) -> None:
    """Print enough provenance to make accidental preprocessing obvious."""

    joints = scene.motion.joints
    lower = joints.min(axis=(0, 1))
    upper = joints.max(axis=(0, 1))
    print("[viser_player:raw] Loaded source scene")
    print(f"  source: {scene.motion.source_path}")
    print(f"  data format: {scene.data_format}")
    print(f"  task type: {scene.task_type}")
    print(f"  frames/joints/fps: {joints.shape[0]}/{joints.shape[1]}/{scene.motion.fps:.3f}")
    print(f"  registered human height: {scene.motion.human_height:.4f} m")
    print(f"  raw joint bounds xyz: {lower.round(4).tolist()} -> {upper.round(4).tolist()}")
    print("  spatial processing: NONE (no foot-height shift, robot-height scale, augmentation, or retargeting)")
    if scene.motion.orientation_quaternions_wxyz is None:
        print("  source joint orientations: unavailable")
    else:
        print(
            "  source joint orientations: "
            f"{scene.motion.orientation_quaternions_wxyz.shape}, "
            f"direct subset={scene.motion.orientation_joint_names}, global Z-up wxyz"
        )
    if scene.mesh is None:
        print("  mesh: none; displaying the complete source skeleton only")
    else:
        print(
            f"  mesh: {scene.mesh_path} ({len(scene.mesh.vertices)} vertices, {len(scene.mesh.faces)} faces, unscaled)"
        )
        print(
            "  mesh motion: "
            + ("raw per-frame object pose" if scene.object_poses_wxyz_xyz is not None else "static identity pose")
        )


def make_viewer(cfg: Config, scene: RawScene) -> viser.ViserServer:
    """Create the interactive Viser player for one raw scene."""

    joints = scene.motion.joints
    num_frames = joints.shape[0]
    start_frame = int(np.clip(cfg.start_frame, 0, num_frames - 1))
    point_colors = _joint_colors(scene.joint_names)
    line_colors = _joint_colors(scene.joint_names, for_lines=True)
    edge_colors = _edge_colors(line_colors, scene.skeleton_edges)

    server = viser.ViserServer(port=cfg.port)
    tabs = add_visualization_tabs(server.gui, include_motions=False)
    server.scene.add_grid(
        "/raw/grid",
        width=4.0,
        height=4.0,
        position=(0.0, 0.0, 0.0),
    )
    points_handle = server.scene.add_point_cloud(
        "/raw/human/keypoints",
        points=joints[start_frame],
        colors=point_colors,
        point_size=cfg.point_size,
        point_shape="circle",
        precision="float32",
        point_shading="flat",
    )
    skeleton_handle = server.scene.add_line_segments(
        "/raw/human/skeleton",
        points=_edge_segments(joints[start_frame], scene.skeleton_edges),
        colors=edge_colors,
        line_width=cfg.line_width,
    )
    labels = [
        server.scene.add_label(
            f"/raw/human/labels/{index:02d}",
            name,
            position=joints[start_frame, index],
            visible=cfg.show_joint_labels,
            font_screen_scale=0.55,
            depth_test=True,
        )
        for index, name in enumerate(scene.joint_names)
    ]
    source_orientation_names = scene.motion.orientation_joint_names or ()
    source_orientations = scene.motion.orientation_quaternions_wxyz
    full_joint_index = {
        name: index
        for index, name in enumerate(scene.joint_names)
    }
    source_orientation_point_indices = np.asarray(
        [full_joint_index[name] for name in source_orientation_names],
        dtype=np.int32,
    )
    orientation_frames = (
        [
            server.scene.add_frame(
                f"/raw/human/orientations/{index:02d}_{name}",
                wxyz=source_orientations[start_frame, index],
                position=joints[
                    start_frame,
                    source_orientation_point_indices[index],
                ],
                show_axes=True,
                axes_length=cfg.orientation_axis_length,
                axes_radius=max(cfg.orientation_axis_length * 0.025, 0.001),
                visible=cfg.show_joint_orientations,
            )
            for index, name in enumerate(source_orientation_names)
        ]
        if source_orientations is not None
        else []
    )

    object_root = server.scene.add_frame("/raw/object", show_axes=False)
    mesh_handle = None
    if scene.mesh is not None:
        mesh_handle = server.scene.add_mesh_simple(
            "/raw/object/mesh",
            vertices=np.asarray(scene.mesh.vertices, dtype=np.float32),
            faces=np.asarray(scene.mesh.faces, dtype=np.int32),
            color=(120, 210, 135),
            opacity=cfg.mesh_opacity,
            flat_shading=False,
            side="double",
        )

    with tabs.playback, server.gui.add_folder("Playback"):
        playing_handle = server.gui.add_checkbox("Playing", initial_value=cfg.playing)
        frame_handle = server.gui.add_slider(
            "Frame",
            min=0,
            max=num_frames - 1,
            step=1,
            initial_value=start_frame,
        )
        fps_handle = server.gui.add_slider(
            "FPS",
            min=1.0,
            max=max(120.0, scene.motion.fps),
            step=1.0,
            initial_value=scene.motion.fps,
        )
        loop_handle = server.gui.add_checkbox("Loop", initial_value=cfg.loop)

    render_lock = threading.Lock()
    updating_frame_slider = {"flag": False}
    playback_clock = {"next": time.monotonic()}

    def render_frame(frame_index: int) -> None:
        frame_index = int(np.clip(frame_index, 0, num_frames - 1))
        with render_lock, server.atomic():
            frame_points = joints[frame_index]
            points_handle.points = frame_points
            skeleton_handle.points = _edge_segments(frame_points, scene.skeleton_edges)
            for index, label in enumerate(labels):
                label.position = frame_points[index]
            if source_orientations is not None:
                for index, orientation_frame in enumerate(orientation_frames):
                    orientation_frame.position = frame_points[
                        source_orientation_point_indices[index]
                    ]
                    orientation_frame.wxyz = source_orientations[
                        frame_index,
                        index,
                    ]
            if scene.object_poses_wxyz_xyz is not None:
                pose = scene.object_poses_wxyz_xyz[frame_index]
                object_root.wxyz = pose[:4]
                object_root.position = pose[4:7]

    def set_discrete_frame(frame_index: int) -> None:
        frame_index = int(np.clip(frame_index, 0, num_frames - 1))
        playing_handle.value = False
        playback_clock["next"] = time.monotonic()
        updating_frame_slider["flag"] = True
        try:
            frame_handle.value = frame_index
        finally:
            updating_frame_slider["flag"] = False
        render_frame(frame_index)

    def toggle_playback() -> None:
        playing_handle.value = not bool(playing_handle.value)
        playback_clock["next"] = time.monotonic()

    @frame_handle.on_update
    def _(_) -> None:
        if not updating_frame_slider["flag"]:
            set_discrete_frame(int(frame_handle.value))

    def set_human_skeleton_visibility(visible: bool) -> None:
        points_handle.visible = bool(visible)
        skeleton_handle.visible = bool(visible)

    def set_label_visibility(visible: bool) -> None:
        for label in labels:
            label.visible = bool(visible)

    def set_orientation_visibility(visible: bool) -> None:
        for orientation_frame in orientation_frames:
            orientation_frame.visible = bool(visible)

    def set_mesh_visibility(visible: bool) -> None:
        if mesh_handle is not None:
            mesh_handle.visible = bool(visible)

    layer_controller = LayerController()
    layer_controller.register(
        LayerId.ROBOT_MESH,
        available=False,
        visible=False,
        callback=lambda _visible: None,
        unavailable_reason="Raw source inputs do not contain a robot.",
    )
    layer_controller.register(
        LayerId.OBJECT_MESH,
        available=mesh_handle is not None,
        visible=True,
        callback=set_mesh_visibility,
        unavailable_reason="No source object or terrain mesh was resolved.",
    )
    layer_controller.register(
        LayerId.HUMAN_SKELETON,
        available=True,
        visible=True,
        callback=set_human_skeleton_visibility,
    )
    for layer_id, reason in (
        (LayerId.ROBOT_SKELETON, "Raw source inputs do not contain mapped robot links."),
        (LayerId.HUMAN_HANDS, "Hand joints are included in the complete source skeleton."),
        (LayerId.OBJECT_KEYPOINTS, "Raw inputs do not contain retargeting object samples."),
        (LayerId.INTERACTION_MESH, "Raw inputs do not contain saved interaction meshes."),
        (LayerId.FOOT_STICKING, "Raw inputs do not contain saved retargeting foot states."),
    ):
        layer_controller.register(
            layer_id,
            available=False,
            visible=False,
            callback=lambda _visible: None,
            unavailable_reason=reason,
        )
    layer_controller.register(
        LayerId.SOURCE_ORIENTATION,
        available=bool(orientation_frames),
        visible=cfg.show_joint_orientations,
        callback=set_orientation_visibility,
        unavailable_reason="The selected source adapter did not provide global joint frames.",
    )
    for layer_id, reason in (
        (LayerId.TARGET_ORIENTATION, "Target frames exist only in retargeting results."),
        (LayerId.ROBOT_ORIENTATION, "Robot frames exist only in retargeting results."),
    ):
        layer_controller.register(
            layer_id,
            available=False,
            visible=False,
            callback=lambda _visible: None,
            unavailable_reason=reason,
        )
    layer_controller.register(
        LayerId.JOINT_LABELS,
        available=True,
        visible=cfg.show_joint_labels,
        callback=set_label_visibility,
    )
    for layer_id, reason in (
        (LayerId.BODY_COM, "Body centers exist only in converted robot motions."),
        (LayerId.BODY_VELOCITY, "Body velocities exist only in converted robot motions."),
    ):
        layer_controller.register(
            layer_id,
            available=False,
            visible=False,
            callback=lambda _visible: None,
            unavailable_reason=reason,
        )

    with tabs.layers:
        layer_controller.add_gui(server.gui)
    with tabs.style:
        server.gui.add_number(
            "Human point size",
            initial_value=float(cfg.point_size),
            min=0.001,
            max=0.2,
            step=0.005,
            disabled=True,
        )
        server.gui.add_number(
            "Skeleton line width",
            initial_value=float(cfg.line_width),
            min=0.1,
            max=20.0,
            step=0.5,
            disabled=True,
        )
        server.gui.add_number(
            "Object mesh opacity",
            initial_value=float(cfg.mesh_opacity),
            min=0.0,
            max=1.0,
            step=0.05,
            disabled=True,
        )
    layer_controller.register_shortcuts(server)
    server._holosoma_layer_controller = layer_controller

    render_frame(start_frame)

    register_keyboard_shortcut(
        server,
        "Playback: Play / Pause",
        hotkey="space",
        callback=toggle_playback,
    )
    register_keyboard_shortcut(
        server,
        "Playback: Previous Frame",
        hotkey="[",
        callback=lambda: set_discrete_frame(int(frame_handle.value) - 1),
    )
    register_keyboard_shortcut(
        server,
        "Playback: Next Frame",
        hotkey="]",
        callback=lambda: set_discrete_frame(int(frame_handle.value) + 1),
    )
    register_keyboard_shortcut(
        server,
        "Playback: First Frame",
        hotkey="home",
        callback=lambda: set_discrete_frame(0),
    )
    register_keyboard_shortcut(
        server,
        "Playback: Last Frame",
        hotkey="end",
        callback=lambda: set_discrete_frame(num_frames - 1),
    )
    print("[viser_player:raw] Open the Viser URL printed above")
    print("[viser_player:raw] Ctrl+C stops the viewer")

    def playback_loop() -> None:
        while True:
            if not playing_handle.value:
                playback_clock["next"] = time.monotonic()
                time.sleep(0.01)
                continue
            fps = max(float(fps_handle.value), 1.0)
            playback_clock["next"] += 1.0 / fps
            current = int(frame_handle.value)
            following = current + 1
            if following >= num_frames:
                if loop_handle.value:
                    following = 0
                else:
                    playing_handle.value = False
                    continue
            updating_frame_slider["flag"] = True
            try:
                frame_handle.value = following
            finally:
                updating_frame_slider["flag"] = False
            render_frame(following)
            delay = playback_clock["next"] - time.monotonic()
            if delay > 0.0:
                time.sleep(min(delay, 0.05))
            else:
                playback_clock["next"] = time.monotonic()

    threading.Thread(target=playback_loop, daemon=True).start()
    return server


def main(cfg: Config) -> None:
    scene = load_raw_scene(cfg)
    print_summary(scene)
    if cfg.dry_run:
        return
    make_viewer(cfg, scene)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[viser_player:raw] Stopped")
