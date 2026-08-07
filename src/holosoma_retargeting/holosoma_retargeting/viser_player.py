#!/usr/bin/env python3
# viser_player.py
# ruff: noqa: CPY001, E402, PLR0917
from __future__ import annotations

import re
import sys
import threading
import time
from contextlib import suppress
from dataclasses import replace
from pathlib import Path

import mujoco  # type: ignore[import-not-found]
import numpy as np
import trimesh
import tyro
import viser  # type: ignore[import-not-found]  # pip install viser
import yourdfpy  # type: ignore[import-untyped]  # pip install yourdfpy
from viser.extras import ViserUrdf  # type: ignore[import-not-found]

src_root = Path(__file__).resolve().parent.parent
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))
from holosoma_retargeting.config_types.data_type import (
    DEMO_JOINTS_REGISTRY,
    JOINTS_MAPPINGS,
    MotionDataConfig,
    normalize_data_format,
)
from holosoma_retargeting.config_types.robot import RobotConfig
from holosoma_retargeting.config_types.viser import ViserConfig
from holosoma_retargeting.data_utils.hand_skeleton import (
    build_hand_visualization_spec,
)
from holosoma_retargeting.data_utils.object_assets import get_omomo_object_asset
from holosoma_retargeting.data_utils.omomo import (
    OMOMO_OBJECT_NAMES,
    resolve_omomo_result_object_name,
)
from holosoma_retargeting.src.utils import interaction_mesh_edges_from_tetrahedra
from holosoma_retargeting.src.viser_utils import (
    create_motion_control_sliders,
    format_foot_sticking_status,
    infer_mujoco_xml_path,
)
from holosoma_retargeting.visualization.joint_angles import (
    add_joint_angle_gui,
    build_joint_angle_diagnostics,
)
from holosoma_retargeting.visualization.layers import (
    LayerController,
    LayerId,
    add_visualization_tabs,
)
from holosoma_retargeting.visualization.orientation import (
    OrientationDiagnostics,
    OrientationPreview,
    interpolate_orientation_quaternions,
    orientation_axis_segments,
    orientation_joint_indices,
    orientation_skeleton_point_indices,
)
from holosoma_retargeting.visualization.result_loader import (
    _resolve_asset_path,
    load_variant_result,
    resolve_qpos_to_viser_joint_indices,
    variant_result_metadata,
)


def load_npz(npz_path: str):
    """Load one result through the shared visualization adapter."""

    result = load_variant_result(
        Path(npz_path).stem,
        npz_path,
    )
    return (
        result.qpos,
        result.fps,
        result.human_joints,
        variant_result_metadata(result),
        result.interaction_mesh,
    )


def _saved_foot_sticking_constraint_status(
    foot_sticking: dict[str, object],
    frame_idx: int,
) -> str:
    """Describe whether detected sticking was actually constrained in a saved frame."""
    if not bool(foot_sticking["enabled"]):
        return "disabled for saved trajectory"
    states = np.asarray(foot_sticking["states"], dtype=bool)
    if bool(np.any(states[frame_idx])):
        return "active"
    return "inactive (no sticking foot)"


def _resolve_runtime_config(config: ViserConfig, metadata: dict[str, object]) -> ViserConfig:
    robot_type = config.robot_type or metadata.get("robot_type")
    if robot_type is not None:
        robot_type = str(robot_type)

    robot_urdf = config.robot_urdf
    if robot_urdf is None:
        if robot_type is None:
            raise ValueError("Result has no robot_type metadata; pass --robot-urdf explicitly.")
        robot_urdf = RobotConfig(robot_type=robot_type).ROBOT_URDF_FILE
    robot_urdf = str(_resolve_asset_path(robot_urdf))

    robot_mujoco_xml = config.robot_mujoco_xml
    if robot_mujoco_xml is not None:
        robot_mujoco_xml = str(_resolve_asset_path(robot_mujoco_xml))

    object_urdf = config.object_urdf
    saved_object_urdf = metadata.get("object_urdf")
    if object_urdf is None and saved_object_urdf:
        object_urdf = str(saved_object_urdf)
    if object_urdf is None and metadata.get("contains_object_in_qpos") is not False:
        object_name = metadata.get("object_name")
        if object_name in OMOMO_OBJECT_NAMES:
            object_urdf = str(get_omomo_object_asset(str(object_name)).urdf_path)
        else:
            result_stem = Path(config.qpos_npz).stem
            dynamic_name_hint = re.search(
                r"_(?:original|augmented|trans_[0-9]+|rot_[0-9]+)$",
                result_stem,
            )
            if config.assume_object_in_qpos is True or dynamic_name_hint:
                try:
                    inferred_object = resolve_omomo_result_object_name(config.qpos_npz)
                except ValueError:
                    pass
                else:
                    object_urdf = str(get_omomo_object_asset(inferred_object).urdf_path)
    if object_urdf is not None:
        object_urdf = str(_resolve_asset_path(object_urdf))

    return replace(
        config,
        robot_type=robot_type,
        robot_urdf=robot_urdf,
        robot_mujoco_xml=robot_mujoco_xml,
        object_urdf=object_urdf,
    )


def _build_qpos_to_viser_joint_indices(
    config: ViserConfig,
    viser_joint_names: list[str],
    npz_metadata: dict[str, object],
) -> np.ndarray | None:
    if config.robot_urdf is None:
        raise ValueError("robot_urdf must be resolved before building the Viser player")
    saved_names_value = npz_metadata.get("robot_actuated_joint_names")
    saved_joint_names = (
        tuple(str(name) for name in saved_names_value) if isinstance(saved_names_value, (list, tuple)) else ()
    )
    fallback_xml = (
        Path(config.robot_mujoco_xml) if config.robot_mujoco_xml else infer_mujoco_xml_path(config.robot_urdf)
    )
    indices = resolve_qpos_to_viser_joint_indices(
        result_path=config.qpos_npz,
        saved_joint_names=saved_joint_names,
        viser_joint_names=tuple(viser_joint_names),
        fallback_mujoco_xml=fallback_xml,
    )
    if indices is not None:
        source = "saved robot_actuated_joint_names" if saved_joint_names else f"MuJoCo XML {fallback_xml}"
        print(f"[viser_player] Reordering qpos joints from {source}.")
    return indices


def _resolve_robot_mujoco_xml(config: ViserConfig) -> Path | None:
    if config.robot_urdf is None:
        return None
    return Path(config.robot_mujoco_xml) if config.robot_mujoco_xml else infer_mujoco_xml_path(config.robot_urdf)


def _mesh_color_override(opacity: float):
    opacity = float(np.clip(opacity, 0.0, 1.0))
    if opacity >= 0.999:
        return None
    return (0.7, 0.7, 0.7, opacity)


def _resolve_robot_type(config: ViserConfig, npz_metadata: dict[str, object]) -> str:
    if config.robot_type:
        return config.robot_type
    if npz_metadata.get("robot_type"):
        return str(npz_metadata["robot_type"])
    if config.robot_urdf is None:
        raise ValueError("Cannot resolve robot type; pass --robot-type or --robot-urdf.")
    return Path(config.robot_urdf).parent.name


def _resolve_data_format(
    config: ViserConfig,
    human_joints: np.ndarray,
    robot_type: str,
    npz_metadata: dict[str, object],
) -> str | None:
    if config.data_format:
        return normalize_data_format(config.data_format)
    if npz_metadata.get("source_data_format"):
        return normalize_data_format(str(npz_metadata["source_data_format"]))

    n_joints = int(human_joints.shape[1])
    candidates = [
        data_format
        for data_format, demo_joints in DEMO_JOINTS_REGISTRY.items()
        if len(demo_joints) == n_joints and (data_format, robot_type) in JOINTS_MAPPINGS
    ]

    path_hint = _data_format_from_path_hint(config.qpos_npz)
    if path_hint in candidates:
        print(f"[viser_player] Inferred data_format={path_hint} from qpos_npz path: {config.qpos_npz}")
        return path_hint

    if len(candidates) == 1:
        print(
            f"[viser_player] Inferred data_format={candidates[0]} from human_joints shape and robot_type={robot_type}"
        )
        return candidates[0]
    if len(candidates) > 1:
        print(
            "[viser_player] Cannot infer data_format uniquely for mapped skeletons. "
            f"Candidates: {candidates}. Pass --data_format explicitly."
        )
    else:
        print(
            "[viser_player] Cannot infer data_format for mapped skeletons from "
            f"human_joints.shape={human_joints.shape} and robot_type={robot_type}."
        )
    return None


def _data_format_from_path_hint(path: str) -> str | None:
    normalized = path.lower().replace("\\", "/")
    if "fbx_mocap" in normalized:
        return "fbx_mocap"
    if "amass_smplx" in normalized or "smplx" in normalized:
        return "amass"
    if "gvhmr" in normalized:
        return "gvhmr"
    if "noetix" in normalized:
        return "noetix_mocap"
    if "lafan" in normalized:
        return "lafan"
    if "omomo" in normalized or "smplh" in normalized:
        return "omomo"
    if "mocap" in normalized or "climb" in normalized:
        return "mocap"
    return None


def _mapped_skeleton_edges(joint_names: list[str]) -> list[tuple[int, int]]:
    joint_idx = {name: idx for idx, name in enumerate(joint_names)}
    candidate_edges = [
        ("Pelvis", "L_Hip"),
        ("L_Hip", "L_Knee"),
        ("L_Knee", "L_Ankle"),
        ("L_Ankle", "L_Toe"),
        ("L_Ankle", "L_Foot"),
        ("Pelvis", "R_Hip"),
        ("R_Hip", "R_Knee"),
        ("R_Knee", "R_Ankle"),
        ("R_Ankle", "R_Toe"),
        ("R_Ankle", "R_Foot"),
        ("Pelvis", "L_Shoulder"),
        ("L_Shoulder", "L_Elbow"),
        ("L_Elbow", "L_Wrist"),
        ("Pelvis", "R_Shoulder"),
        ("R_Shoulder", "R_Elbow"),
        ("R_Elbow", "R_Wrist"),
        ("L_Shoulder", "R_Shoulder"),
        ("L_Hip", "R_Hip"),
        ("Spine1", "LeftUpLeg"),
        ("LeftUpLeg", "LeftLeg"),
        ("LeftLeg", "LeftFoot"),
        ("LeftFoot", "LeftToeBase"),
        ("Spine1", "RightUpLeg"),
        ("RightUpLeg", "RightLeg"),
        ("RightLeg", "RightFoot"),
        ("RightFoot", "RightToeBase"),
        ("Spine1", "LeftArm"),
        ("LeftArm", "LeftForeArm"),
        ("LeftForeArm", "LeftHand"),
        ("LeftForeArm", "LeftHandMiddle3"),
        ("Spine1", "RightArm"),
        ("RightArm", "RightForeArm"),
        ("RightForeArm", "RightHand"),
        ("RightForeArm", "RightHandMiddle3"),
        ("LeftArm", "RightArm"),
        ("LeftUpLeg", "RightUpLeg"),
    ]
    edges: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for a, b in candidate_edges:
        if a not in joint_idx or b not in joint_idx:
            continue
        edge = (joint_idx[a], joint_idx[b])
        if edge in seen:
            continue
        edges.append(edge)
        seen.add(edge)
    return edges


def _interpolate_sequence(
    sequence: np.ndarray,
    frame_float: float,
    *,
    loop: bool = False,
) -> np.ndarray:
    n_frames = int(sequence.shape[0])
    if n_frames == 1:
        return sequence[0]

    if loop:
        sample_frame = float(frame_float) % n_frames
        i0 = int(np.floor(sample_frame))
        i1 = (i0 + 1) % n_frames
    else:
        sample_frame = float(np.clip(frame_float, 0.0, n_frames - 1))
        i0 = int(np.floor(sample_frame))
        i1 = min(i0 + 1, n_frames - 1)
    u = sample_frame - i0
    return (1.0 - u) * sequence[i0] + u * sequence[i1]


class MappedSkeletonOverlay:
    def __init__(
        self,
        server: viser.ViserServer,
        human_joints: np.ndarray,
        demo_joints: list[str],
        joints_mapping: dict[str, str],
        robot_xml_path: Path | None,
        point_radius: float,
        line_width: float,
        human_joint_parent_indices: np.ndarray | None = None,
        mapped_robot_joints: np.ndarray | None = None,
        robot_skeleton_joints: np.ndarray | None = None,
        robot_skeleton_parent_indices: np.ndarray | None = None,
        namespace: str = "/overlays/mapped",
        offset: np.ndarray | None = None,
        loop: bool = False,
    ) -> None:
        self.server = server
        self.human_joints = np.asarray(human_joints)
        self.demo_joints = demo_joints
        self.joints_mapping = joints_mapping
        self.human_joint_names = list(demo_joints)
        self.mapped_human_joint_names = list(joints_mapping.keys())
        self.robot_link_names = list(joints_mapping.values())
        self.human_joint_indices = list(range(len(demo_joints)))
        self.mapped_robot_joints = (
            np.asarray(mapped_robot_joints, dtype=np.float32) if mapped_robot_joints is not None else None
        )
        self.mapped_edges = _mapped_skeleton_edges(self.mapped_human_joint_names)
        if human_joint_parent_indices is not None:
            human_parents = np.asarray(human_joint_parent_indices, dtype=np.int32)
            if human_parents.shape != (len(demo_joints),):
                raise ValueError("Full human skeleton parent indices have an incompatible shape")
            self.human_edges = [
                (int(parent), index)
                for index, parent in enumerate(human_parents)
                if parent >= 0
            ]
        else:
            self.human_edges = _mapped_skeleton_edges(self.human_joint_names)
        self.robot_skeleton_joints = (
            np.asarray(robot_skeleton_joints, dtype=np.float32) if robot_skeleton_joints is not None else None
        )
        if self.robot_skeleton_joints is not None:
            parent_indices = np.asarray(
                robot_skeleton_parent_indices,
                dtype=np.int32,
            )
            if (
                self.robot_skeleton_joints.ndim != 3
                or self.robot_skeleton_joints.shape[0] != self.human_joints.shape[0]
                or self.robot_skeleton_joints.shape[-1] != 3
                or parent_indices.shape != (self.robot_skeleton_joints.shape[1],)
            ):
                raise ValueError("Full robot skeleton positions and parent indices have incompatible shapes")
            self.robot_edges = [(int(parent), index) for index, parent in enumerate(parent_indices) if parent >= 0]
        elif self.mapped_robot_joints is not None:
            self.robot_edges = self.mapped_edges
        else:
            self.robot_edges = self.mapped_edges
        self.hand_visualization_spec = build_hand_visualization_spec(
            self.demo_joints,
            self.mapped_human_joint_names,
        )
        self.line_width = float(line_width)
        self.namespace = namespace.rstrip("/")
        self.offset = np.zeros(3, dtype=np.float32) if offset is None else np.asarray(offset, dtype=np.float32)
        self.loop = bool(loop)
        if self.offset.shape != (3,):
            raise ValueError(f"Mapped skeleton offset must have shape (3,), got {self.offset.shape}")
        self.human_visible = True
        self.robot_visible = True
        self.hands_visible = True
        self.labels_visible = False
        self._lock = threading.Lock()
        self._handles: list[object] = []

        self._sphere = trimesh.primitives.Sphere(radius=float(point_radius))
        self._sphere_vertices = self._sphere.vertices.astype(np.float32)
        self._sphere_faces = self._sphere.faces.astype(np.int32)
        self._hand_sphere = trimesh.primitives.Sphere(radius=float(point_radius) * 0.6)
        self._hand_sphere_vertices = self._hand_sphere.vertices.astype(np.float32)
        self._hand_sphere_faces = self._hand_sphere.faces.astype(np.int32)

        self.robot_model = None
        self.robot_data = None
        self.robot_body_ids: list[int] = []
        needs_robot_fk = self.robot_skeleton_joints is None and self.mapped_robot_joints is None
        if needs_robot_fk:
            if robot_xml_path is None:
                raise ValueError("A robot MuJoCo XML is required when no saved robot skeleton positions are available")
            self.robot_model = mujoco.MjModel.from_xml_path(str(robot_xml_path))
            self.robot_data = mujoco.MjData(self.robot_model)
            missing_links: list[str] = []
            for link_name in self.robot_link_names:
                body_id = mujoco.mj_name2id(
                    self.robot_model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    link_name,
                )
                if body_id == -1:
                    missing_links.append(link_name)
                self.robot_body_ids.append(body_id)
            if missing_links:
                raise ValueError(
                    f"Mapped skeleton robot links are missing in MuJoCo model {robot_xml_path}: {missing_links}"
                )

    def set_visible(self, visible: bool) -> None:
        self.set_human_visible(visible)
        self.set_robot_visible(visible)
        self.set_hands_visible(visible)

    @property
    def visible(self) -> bool:
        return self.human_visible or self.robot_visible or self.hands_visible or self.labels_visible

    def set_human_visible(self, visible: bool) -> None:
        with self._lock:
            self.human_visible = bool(visible)
            if not self.visible:
                self._clear_locked()

    def set_robot_visible(self, visible: bool) -> None:
        with self._lock:
            self.robot_visible = bool(visible)
            if not self.visible:
                self._clear_locked()

    def set_hands_visible(self, visible: bool) -> None:
        with self._lock:
            self.hands_visible = bool(visible)
            if not self.visible:
                self._clear_locked()

    def set_labels_visible(self, visible: bool) -> None:
        with self._lock:
            self.labels_visible = bool(visible)
            if not self.visible:
                self._clear_locked()

    def draw(self, q: np.ndarray, frame_float: float) -> None:
        with self._lock:
            if not self.visible:
                return

            self._clear_locked()
            human_frame = self._human_frame(frame_float)
            human_points = (
                np.asarray(
                    human_frame[self.human_joint_indices],
                    dtype=np.float32,
                )
                + self.offset
            )
            if self.human_visible:
                self._handles.append(
                    self._draw_points(
                        f"{self.namespace}/human_kpts",
                        human_points,
                        color=(0, 0, 255),
                    )
                )
                human_skeleton = self._draw_skeleton(
                    f"{self.namespace}/human_skeleton",
                    human_points,
                    self.human_edges,
                    color=np.array([0.0, 0.0, 1.0]),
                    line_width=self.line_width,
                )
                if human_skeleton is not None:
                    self._handles.append(human_skeleton)
            if self.robot_visible:
                robot_points = self._robot_points(q, frame_float) + self.offset
                self._handles.append(
                    self._draw_points(
                        f"{self.namespace}/robot_kpts",
                        robot_points,
                        color=(0, 255, 0),
                    )
                )
                robot_skeleton = self._draw_skeleton(
                    f"{self.namespace}/robot_skeleton",
                    robot_points,
                    self.robot_edges,
                    color=np.array([0.0, 1.0, 0.0]),
                    line_width=self.line_width,
                )
                if robot_skeleton is not None:
                    self._handles.append(robot_skeleton)
            hand_spec = self.hand_visualization_spec
            if self.hands_visible and hand_spec.keypoint_indices and hand_spec.edge_indices:
                hand_points = (
                    np.asarray(
                        human_frame[np.asarray(hand_spec.keypoint_indices, dtype=int)],
                        dtype=np.float32,
                    )
                    + self.offset
                )
                self._handles.append(
                    self._draw_points(
                        f"{self.namespace}/human_hand_kpts",
                        hand_points,
                        color=(64, 64, 255),
                        vertices=self._hand_sphere_vertices,
                        faces=self._hand_sphere_faces,
                    )
                )
                hand_skeleton = self._draw_skeleton(
                    f"{self.namespace}/human_hand_skeleton",
                    human_frame + self.offset,
                    list(hand_spec.edge_indices),
                    color=np.array([0.25, 0.25, 1.0]),
                    line_width=self.line_width * 0.75,
                )
                if hand_skeleton is not None:
                    self._handles.append(hand_skeleton)
            if self.labels_visible:
                for index, name in enumerate(self.demo_joints):
                    self._handles.append(
                        self.server.scene.add_label(
                            f"{self.namespace}/joint_labels/{index:03d}",
                            name,
                            position=human_frame[index] + self.offset,
                            font_screen_scale=0.45,
                            depth_test=True,
                        )
                    )

    def _clear_locked(self) -> None:
        for handle in self._handles:
            with suppress(Exception):
                handle.remove()
        self._handles.clear()

    def _human_frame(self, frame_float: float) -> np.ndarray:
        return np.asarray(
            _interpolate_sequence(self.human_joints, frame_float, loop=self.loop),
            dtype=np.float32,
        )

    def _robot_points(self, q: np.ndarray, frame_float: float) -> np.ndarray:
        if self.robot_skeleton_joints is not None:
            return np.asarray(
                _interpolate_sequence(
                    self.robot_skeleton_joints,
                    frame_float,
                    loop=self.loop,
                ),
                dtype=np.float32,
            )
        if self.mapped_robot_joints is not None:
            return np.asarray(
                _interpolate_sequence(
                    self.mapped_robot_joints,
                    frame_float,
                    loop=self.loop,
                ),
                dtype=np.float32,
            )

        if self.robot_model is None or self.robot_data is None:
            raise RuntimeError("Robot FK is unavailable and no saved skeleton positions exist")
        q = np.asarray(q, dtype=float)
        model_q = np.zeros(self.robot_model.nq, dtype=float)
        if q.shape[0] >= self.robot_model.nq:
            model_q[:] = q[: self.robot_model.nq]
        else:
            model_q[: q.shape[0]] = q

        self.robot_data.qpos[:] = model_q
        mujoco.mj_forward(self.robot_model, self.robot_data)
        return self.robot_data.xpos[np.asarray(self.robot_body_ids, dtype=int)].copy().astype(np.float32)

    def _draw_points(
        self,
        name: str,
        points: np.ndarray,
        color: tuple[int, int, int],
        *,
        vertices: np.ndarray | None = None,
        faces: np.ndarray | None = None,
    ):
        return self.server.scene.add_batched_meshes_simple(
            name,
            vertices=self._sphere_vertices if vertices is None else vertices,
            faces=self._sphere_faces if faces is None else faces,
            batched_positions=points,
            batched_wxyzs=np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (points.shape[0], 1)),
            batched_colors=color,
            opacity=1.0,
        )

    def _draw_skeleton(
        self,
        name: str,
        points: np.ndarray,
        edges: list[tuple[int, int]],
        color: np.ndarray,
        line_width: float,
    ):
        if not edges:
            return None

        segments = np.asarray([[points[i], points[j]] for i, j in edges], dtype=np.float32)
        colors = np.tile(color.reshape(1, 1, 3), (segments.shape[0], 2, 1))
        return self.server.scene.add_line_segments(name, points=segments, colors=colors, line_width=line_width)


class ObjectKeypointOverlay:
    """Draw saved source/demo and target object samples with retargeting colors."""

    def __init__(
        self,
        server: viser.ViserServer,
        keypoint_data: dict[str, np.ndarray],
        point_radius: float,
        namespace: str = "/overlays/object_keypoints",
        offset: np.ndarray | None = None,
        loop: bool = False,
    ) -> None:
        self.server = server
        self.demo_world = np.asarray(keypoint_data["demo_world"], dtype=np.float32)
        self.target_world = np.asarray(keypoint_data["target_world"], dtype=np.float32)
        if (
            self.demo_world.ndim != 3
            or self.target_world.ndim != 3
            or self.demo_world.shape != self.target_world.shape
            or self.demo_world.shape[-1] != 3
        ):
            raise ValueError("Saved object keypoints must have matching (frames, points, 3) shapes.")
        self.visible = True
        self.namespace = namespace.rstrip("/")
        self.offset = np.zeros(3, dtype=np.float32) if offset is None else np.asarray(offset, dtype=np.float32)
        self.loop = bool(loop)
        if self.offset.shape != (3,):
            raise ValueError(f"Object-keypoint offset must have shape (3,), got {self.offset.shape}")
        self._lock = threading.Lock()
        self._handles: list[object] = []
        sphere = trimesh.primitives.Sphere(radius=float(point_radius))
        self._sphere_vertices = sphere.vertices.astype(np.float32)
        self._sphere_faces = sphere.faces.astype(np.int32)

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
            demo_points = _interpolate_sequence(self.demo_world, frame_float, loop=self.loop) + self.offset
            target_points = _interpolate_sequence(self.target_world, frame_float, loop=self.loop) + self.offset
            self._handles.extend(
                [
                    self._draw_points(
                        f"{self.namespace}/demo_scaled",
                        demo_points,
                        color=(255, 0, 0),
                    ),
                    self._draw_points(
                        f"{self.namespace}/target",
                        target_points,
                        color=(0, 255, 255),
                    ),
                ]
            )

    def _clear_locked(self) -> None:
        for handle in self._handles:
            with suppress(Exception):
                handle.remove()
        self._handles.clear()

    def _draw_points(
        self,
        name: str,
        points: np.ndarray,
        color: tuple[int, int, int],
    ):
        return self.server.scene.add_batched_meshes_simple(
            name,
            vertices=self._sphere_vertices,
            faces=self._sphere_faces,
            batched_positions=np.asarray(points, dtype=np.float32),
            batched_wxyzs=np.tile(
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                (points.shape[0], 1),
            ),
            batched_colors=color,
            opacity=1.0,
        )


class RetargetingPointCloudOverlay:
    """Draw the compact solver points retained for result inspection."""

    _COLORS = {
        "human": np.asarray((50, 100, 255), dtype=np.uint8),
        "robot": np.asarray((40, 220, 80), dtype=np.uint8),
        "terrain": np.asarray((135, 135, 135), dtype=np.uint8),
        "object_demo": np.asarray((255, 80, 40), dtype=np.uint8),
        "object_target": np.asarray((30, 220, 235), dtype=np.uint8),
    }

    def __init__(
        self,
        server: viser.ViserServer,
        trajectories: dict[str, np.ndarray],
        point_size: float,
        namespace: str = "/overlays/retargeting_points",
        loop: bool = False,
    ) -> None:
        self.trajectories = {
            name: np.asarray(points, dtype=np.float32)
            for name, points in trajectories.items()
            if np.asarray(points).ndim == 3 and np.asarray(points).shape[1] > 0
        }
        self.loop = bool(loop)
        self.visible = True
        self.handles: dict[str, object] = {}
        namespace = namespace.rstrip("/")
        for name, points in self.trajectories.items():
            colors = np.tile(self._COLORS[name], (points.shape[1], 1))
            self.handles[name] = server.scene.add_point_cloud(
                f"{namespace}/{name}",
                points=points[0],
                colors=colors,
                point_size=float(point_size),
                point_shape="circle",
                precision="float32",
                point_shading="flat",
                visible=True,
            )

    def set_visible(self, visible: bool) -> None:
        self.visible = bool(visible)
        for handle in self.handles.values():
            handle.visible = self.visible

    def draw(self, frame_float: float) -> None:
        if not self.visible:
            return
        for name, points in self.trajectories.items():
            self.handles[name].points = np.asarray(
                _interpolate_sequence(points, frame_float, loop=self.loop),
                dtype=np.float32,
            )


class InteractionMeshOverlay:
    def __init__(
        self,
        server: viser.ViserServer,
        mesh_data: dict[str, np.ndarray | int],
        mode: str,
        edge_mode: str,
        line_width: float,
        namespace: str = "/overlays/interaction_mesh",
        offset: np.ndarray | None = None,
        loop: bool = False,
    ) -> None:
        self.server = server
        self.source_vertices = np.asarray(mesh_data["source_vertices"], dtype=np.float32)
        self.target_vertices = np.asarray(mesh_data["target_vertices"], dtype=np.float32)
        self.tetrahedra = np.asarray(mesh_data["tetrahedra"], dtype=np.int32)
        self.tetrahedra_counts = np.asarray(mesh_data["tetrahedra_counts"], dtype=np.int32)
        self.num_human_vertices = int(mesh_data["num_human_vertices"])
        self.mode = mode
        self.edge_mode = edge_mode
        self.line_width = float(line_width)
        self.namespace = namespace.rstrip("/")
        self.offset = np.zeros(3, dtype=np.float32) if offset is None else np.asarray(offset, dtype=np.float32)
        self.loop = bool(loop)
        if self.offset.shape != (3,):
            raise ValueError(f"Interaction-mesh offset must have shape (3,), got {self.offset.shape}")
        self.visible = True
        self._lock = threading.Lock()
        self._handles: list[object] = []

        if self.source_vertices.ndim != 3 or self.target_vertices.ndim != 3:
            raise ValueError("Saved interaction mesh vertices must have shape (frames, vertices, 3).")
        if self.tetrahedra.ndim != 3 or self.tetrahedra.shape[-1] != 4:
            raise ValueError("Saved interaction tetrahedra must have shape (frames, tetrahedra, 4).")
        if self.source_vertices.shape[0] != self.tetrahedra.shape[0]:
            raise ValueError("Saved interaction mesh frame count does not match tetrahedra frame count.")

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
            if self.loop:
                frame_idx = round(float(frame_float)) % self.source_vertices.shape[0]
            else:
                frame_idx = int(
                    np.clip(
                        round(float(frame_float)),
                        0,
                        self.source_vertices.shape[0] - 1,
                    )
                )
            tet_count = int(self.tetrahedra_counts[frame_idx])
            frame_tetrahedra = self.tetrahedra[frame_idx, :tet_count]
            if self.mode in {"source", "both"}:
                self._handles.extend(
                    self._draw_mesh(
                        f"{self.namespace}/source",
                        self.source_vertices[frame_idx] + self.offset,
                        frame_tetrahedra,
                        color=np.array([1.0, 0.55, 0.0], dtype=np.float32),
                    )
                )
            if self.mode in {"target", "both"}:
                self._handles.extend(
                    self._draw_mesh(
                        f"{self.namespace}/target",
                        self.target_vertices[frame_idx] + self.offset,
                        frame_tetrahedra,
                        color=np.array([0.0, 0.9, 1.0], dtype=np.float32),
                    )
                )

    def _clear_locked(self) -> None:
        for handle in self._handles:
            with suppress(Exception):
                handle.remove()
        self._handles.clear()

    def _draw_mesh(self, name: str, vertices: np.ndarray, tetrahedra: np.ndarray, color: np.ndarray) -> list[object]:
        edges = interaction_mesh_edges_from_tetrahedra(
            tetrahedra,
            num_anchor_vertices=self.num_human_vertices,
            edge_mode=self.edge_mode,
        )
        if edges.size == 0:
            return []

        segments = np.asarray(vertices, dtype=np.float32)[edges]
        colors = np.tile(color.reshape(1, 1, 3), (segments.shape[0], 2, 1))
        handle = self.server.scene.add_line_segments(
            name,
            points=segments,
            colors=colors,
            line_width=self.line_width,
        )
        return [handle]


class SingleOrientationOverlay:
    """Saved target and robot orientation axes for one result."""

    _AXIS_COLORS = np.asarray(
        ((255, 0, 0), (0, 255, 0), (0, 0, 255)),
        dtype=np.uint8,
    )

    def __init__(
        self,
        *,
        server,
        diagnostics: OrientationDiagnostics,
        joint_indices: np.ndarray,
        point_indices: np.ndarray,
        human_points: np.ndarray,
        robot_points: np.ndarray,
        axis_length: float,
        shaft_radius: float,
        head_radius: float,
        head_length: float,
        target_visible: bool,
        robot_visible: bool,
        loop: bool,
        namespace: str = "/overlays/orientation",
    ) -> None:
        self.diagnostics = diagnostics
        self.joint_indices = np.asarray(joint_indices, dtype=np.int32)
        self.point_indices = np.asarray(point_indices, dtype=np.int32)
        self.human_points = np.asarray(human_points, dtype=np.float32)
        self.robot_points = np.asarray(robot_points, dtype=np.float32)
        self.axis_length = float(axis_length)
        self.loop = bool(loop)
        self.target_visible = bool(target_visible)
        self.robot_visible = bool(robot_visible)
        target_origins = self.human_points[0, self.point_indices]
        robot_origins = self.robot_points[0, self.point_indices]
        colors = np.tile(self._AXIS_COLORS, (len(self.joint_indices), 1))
        self.target_axes = server.scene.add_arrows(
            f"{namespace}/target_axes",
            points=orientation_axis_segments(
                target_origins,
                diagnostics.target_quaternions_wxyz[0, self.joint_indices],
                self.axis_length * 0.78,
            ),
            colors=colors,
            shaft_radius=shaft_radius,
            head_radius=head_radius,
            head_length=head_length,
            visible=self.target_visible,
        )
        self.robot_axes = server.scene.add_arrows(
            f"{namespace}/robot_axes",
            points=orientation_axis_segments(
                robot_origins,
                diagnostics.robot_quaternions_wxyz[0, self.joint_indices],
                self.axis_length,
            ),
            colors=colors,
            shaft_radius=shaft_radius,
            head_radius=head_radius,
            head_length=head_length,
            visible=self.robot_visible,
        )

    def set_target_visible(self, visible: bool) -> None:
        self.target_visible = bool(visible)
        self.target_axes.visible = self.target_visible

    def set_robot_visible(self, visible: bool) -> None:
        self.robot_visible = bool(visible)
        self.robot_axes.visible = self.robot_visible

    def draw(self, frame_float: float) -> None:
        human_points = _interpolate_sequence(
            self.human_points,
            frame_float,
            loop=self.loop,
        )
        robot_points = _interpolate_sequence(
            self.robot_points,
            frame_float,
            loop=self.loop,
        )
        if self.target_visible:
            target_quaternions = interpolate_orientation_quaternions(
                self.diagnostics.target_quaternions_wxyz[:, self.joint_indices],
                frame_float,
                loop=self.loop,
            )
            self.target_axes.points = orientation_axis_segments(
                human_points[self.point_indices],
                target_quaternions,
                self.axis_length * 0.78,
            )
        if self.robot_visible:
            robot_quaternions = interpolate_orientation_quaternions(
                self.diagnostics.robot_quaternions_wxyz[:, self.joint_indices],
                frame_float,
                loop=self.loop,
            )
            self.robot_axes.points = orientation_axis_segments(
                robot_points[self.point_indices],
                robot_quaternions,
                self.axis_length,
            )


class SavedOrientationAxesOverlay:
    """One independently toggled saved orientation trajectory."""

    _AXIS_COLORS = np.asarray(
        ((255, 0, 0), (0, 255, 0), (0, 0, 255)),
        dtype=np.uint8,
    )

    def __init__(
        self,
        *,
        server,
        namespace: str,
        names: tuple[str, ...],
        positions: np.ndarray,
        quaternions_wxyz: np.ndarray,
        axis_length: float,
        shaft_radius: float,
        head_radius: float,
        head_length: float,
        visible: bool,
        loop: bool,
    ) -> None:
        self.names = names
        self.positions = np.asarray(positions, dtype=np.float32)
        self.quaternions_wxyz = np.asarray(
            quaternions_wxyz,
            dtype=np.float32,
        )
        expected_quaternion_shape = (
            self.positions.shape[0],
            self.positions.shape[1],
            4,
        )
        if (
            self.positions.ndim != 3
            or self.positions.shape[-1] != 3
            or self.quaternions_wxyz.shape != expected_quaternion_shape
            or self.positions.shape[1] != len(names)
        ):
            raise ValueError(
                "Saved orientation axes require positions (frames, items, 3), "
                "quaternions (frames, items, 4), and matching names"
            )
        self.axis_length = float(axis_length)
        self.loop = bool(loop)
        self.visible = bool(visible)
        self.axes = server.scene.add_arrows(
            f"{namespace.rstrip('/')}/axes",
            points=orientation_axis_segments(
                self.positions[0],
                self.quaternions_wxyz[0],
                self.axis_length,
            ),
            colors=np.tile(self._AXIS_COLORS, (len(names), 1)),
            shaft_radius=shaft_radius,
            head_radius=head_radius,
            head_length=head_length,
            visible=self.visible,
        )

    def set_visible(self, visible: bool) -> None:
        self.visible = bool(visible)
        self.axes.visible = self.visible

    @property
    def enabled(self) -> bool:
        return self.visible

    def set_enabled(self, enabled: bool) -> None:
        self.set_visible(enabled)

    def update(self, _q: np.ndarray, frame_float: float) -> None:
        self.draw(frame_float)

    def draw(self, frame_float: float) -> None:
        if not self.visible:
            return
        positions = _interpolate_sequence(
            self.positions,
            frame_float,
            loop=self.loop,
        )
        quaternions = interpolate_orientation_quaternions(
            self.quaternions_wxyz,
            frame_float,
            loop=self.loop,
        )
        self.axes.points = orientation_axis_segments(
            positions,
            quaternions,
            self.axis_length,
        )


class ResultOrientationOverlays:
    """Saved axes with a runtime full/retargeting-link scope filter."""

    def __init__(
        self,
        *,
        source_all: SavedOrientationAxesOverlay | None,
        source_retargeting: SavedOrientationAxesOverlay | None,
        target: SavedOrientationAxesOverlay | None,
        robot_all: SavedOrientationAxesOverlay | None,
        robot_retargeting: SavedOrientationAxesOverlay | None,
        scope: str,
        source_visible: bool,
        robot_visible: bool,
    ) -> None:
        self.source_all = source_all
        self.source_retargeting = source_retargeting
        self.target = target
        self.robot_all = robot_all
        self.robot_retargeting = robot_retargeting
        self.scope = "retargeting"
        self.source_visible = bool(source_visible)
        self.robot_visible = bool(robot_visible)
        self.set_scope(scope)

    @property
    def source_available(self) -> bool:
        return self.source_all is not None or self.source_retargeting is not None

    @property
    def source(self) -> SavedOrientationAxesOverlay | None:
        """Return the overlay selected by the current scope for compatibility."""

        return self.source_all if self.scope == "all" else self.source_retargeting

    @property
    def robot_available(self) -> bool:
        return self.robot_all is not None or self.robot_retargeting is not None

    @property
    def robot(self) -> SavedOrientationAxesOverlay | None:
        """Return the overlay selected by the current scope for compatibility."""

        return self.robot_all if self.scope == "all" else self.robot_retargeting

    def set_scope(self, scope: str) -> None:
        if scope not in {"retargeting", "all"}:
            raise ValueError(f"Unknown orientation scope: {scope!r}")
        self.scope = scope
        self._sync_scope_visibility()

    def _sync_scope_visibility(self) -> None:
        if self.source_all is not None:
            self.source_all.set_visible(self.source_visible and self.scope == "all")
        if self.source_retargeting is not None:
            self.source_retargeting.set_visible(self.source_visible and self.scope == "retargeting")
        if self.robot_all is not None:
            self.robot_all.set_visible(self.robot_visible and self.scope == "all")
        if self.robot_retargeting is not None:
            self.robot_retargeting.set_visible(self.robot_visible and self.scope == "retargeting")

    def set_source_visible(self, visible: bool) -> None:
        self.source_visible = bool(visible)
        self._sync_scope_visibility()

    def set_target_visible(self, visible: bool) -> None:
        if self.target is not None:
            self.target.set_visible(visible)

    def set_robot_visible(self, visible: bool) -> None:
        self.robot_visible = bool(visible)
        self._sync_scope_visibility()

    def draw(self, frame_float: float) -> None:
        for overlay in (
            self.source_all,
            self.source_retargeting,
            self.target,
            self.robot_all,
            self.robot_retargeting,
        ):
            if overlay is not None:
                overlay.draw(frame_float)


def _requested_saved_orientation_indices(
    available_names: tuple[str, ...],
    requested_names: tuple[str, ...],
    *,
    aliases: dict[str, str] | None = None,
) -> np.ndarray:
    if not requested_names:
        return np.arange(len(available_names), dtype=np.int32)
    name_to_index = {name: index for index, name in enumerate(available_names)}
    aliases = aliases or {}
    selected: list[int] = []
    for requested_name in requested_names:
        resolved_name = requested_name if requested_name in name_to_index else aliases.get(requested_name)
        if resolved_name in name_to_index:
            index = name_to_index[resolved_name]
            if index not in selected:
                selected.append(index)
    return np.asarray(selected, dtype=np.int32)


def _build_orientation_overlay(
    config: ViserConfig,
    server,
    human_joints: np.ndarray | None,
    npz_metadata: dict[str, object],
    num_frames: int,
) -> ResultOrientationOverlays:
    source_all_overlay = None
    source_retargeting_overlay = None
    target_overlay = None
    robot_all_overlay = None
    robot_retargeting_overlay = None

    human_joint_names = npz_metadata.get("human_joint_names")
    mapped_joint_names = npz_metadata.get("mapped_human_joint_names")
    mapped_robot_link_names = npz_metadata.get("mapped_robot_link_names")
    mapped_robot_joints = npz_metadata.get("mapped_robot_joints")
    diagnostics = npz_metadata.get("orientation_diagnostics")
    human_index = (
        {name: index for index, name in enumerate(human_joint_names)} if isinstance(human_joint_names, list) else {}
    )

    source_names_value = npz_metadata.get("human_orientation_joint_names")
    source_quaternions_value = npz_metadata.get("human_orientation_quaternions_wxyz")
    preview = npz_metadata.get("orientation_preview")
    if human_joints is not None and isinstance(source_names_value, list) and source_quaternions_value is not None:
        source_names = tuple(source_names_value)
        source_point_indices = np.asarray(
            [human_index[name] for name in source_names],
            dtype=np.int32,
        )
        selected_all = _requested_saved_orientation_indices(
            source_names,
            config.orientation_joints,
        )
        retargeting_human_names = (
            preview.human_joint_names
            if isinstance(preview, OrientationPreview)
            else (
                diagnostics.human_joint_names
                if isinstance(diagnostics, OrientationDiagnostics)
                else tuple(mapped_joint_names or ())
            )
        )
        selected_retargeting = _requested_saved_orientation_indices(
            source_names,
            config.orientation_joints or retargeting_human_names,
        )
        if selected_all.size:
            source_all_overlay = SavedOrientationAxesOverlay(
                server=server,
                namespace="/overlays/orientation/source/all",
                names=tuple(source_names[int(index)] for index in selected_all),
                positions=np.asarray(human_joints)[
                    :,
                    source_point_indices[selected_all],
                ],
                quaternions_wxyz=np.asarray(source_quaternions_value)[
                    :,
                    selected_all,
                ],
                axis_length=config.orientation_axis_length,
                shaft_radius=config.orientation_axis_shaft_radius,
                head_radius=config.orientation_axis_head_radius,
                head_length=config.orientation_axis_head_length,
                visible=False,
                loop=config.loop,
            )
        if selected_retargeting.size:
            source_retargeting_overlay = SavedOrientationAxesOverlay(
                server=server,
                namespace="/overlays/orientation/source/retargeting",
                names=tuple(source_names[int(index)] for index in selected_retargeting),
                positions=np.asarray(human_joints)[:, source_point_indices[selected_retargeting]],
                quaternions_wxyz=np.asarray(source_quaternions_value)[:, selected_retargeting],
                axis_length=config.orientation_axis_length,
                shaft_radius=config.orientation_axis_shaft_radius,
                head_radius=config.orientation_axis_head_radius,
                head_length=config.orientation_axis_head_length,
                visible=False,
                loop=config.loop,
            )

    if isinstance(preview, OrientationPreview) and human_joints is not None:
        missing_preview_joints = [name for name in preview.human_joint_names if name not in human_index]
        if missing_preview_joints:
            raise ValueError(
                "Orientation preview joints are absent from human_joint_names: "
                f"{missing_preview_joints}",
            )
        selected = _requested_saved_orientation_indices(
            preview.human_joint_names,
            config.orientation_joints,
        )
        if selected.size:
            target_overlay = SavedOrientationAxesOverlay(
                server=server,
                namespace="/overlays/orientation/aligned_target",
                names=tuple(preview.human_joint_names[int(index)] for index in selected),
                positions=np.asarray(human_joints)[
                    :,
                    [human_index[preview.human_joint_names[int(index)]] for index in selected],
                ],
                quaternions_wxyz=preview.target_quaternions_wxyz[:, selected],
                axis_length=config.orientation_axis_length * 0.78,
                shaft_radius=config.orientation_axis_shaft_radius,
                head_radius=config.orientation_axis_head_radius,
                head_length=config.orientation_axis_head_length,
                visible=config.show_target_orientation_axes,
                loop=config.loop,
            )
            print(
                "[viser_player] Orientation preview uses independently calibrated "
                f"per-link frames ({preview.alignment_mode}); no SO(3) errors are loaded.",
            )

    if isinstance(diagnostics, OrientationDiagnostics) and human_joints is not None:
        missing = [name for name in diagnostics.human_joint_names if name not in human_index]
        if missing:
            raise ValueError(f"Orientation joints are absent from human_joint_names: {missing}")
        selected = orientation_joint_indices(
            diagnostics,
            config.orientation_joints,
        )
        target_overlay = SavedOrientationAxesOverlay(
            server=server,
            namespace="/overlays/orientation/target",
            names=tuple(diagnostics.human_joint_names[int(index)] for index in selected),
            positions=np.asarray(human_joints)[
                :,
                [human_index[diagnostics.human_joint_names[int(index)]] for index in selected],
            ],
            quaternions_wxyz=diagnostics.target_quaternions_wxyz[
                :,
                selected,
            ],
            axis_length=config.orientation_axis_length * 0.78,
            shaft_radius=config.orientation_axis_shaft_radius,
            head_radius=config.orientation_axis_head_radius,
            head_length=config.orientation_axis_head_length,
            visible=config.show_target_orientation_axes,
            loop=config.loop,
        )
        if npz_metadata.get("robot_link_quaternions_wxyz") is None:
            if (
                not isinstance(mapped_joint_names, list)
                or not isinstance(mapped_robot_link_names, list)
                or mapped_robot_joints is None
            ):
                raise ValueError(
                    "Orientation robot axes require either complete robot-link trajectories "
                    "or mapped robot skeleton trajectories",
                )
            robot_points = np.asarray(mapped_robot_joints, dtype=np.float32)
            expected_shape = (num_frames, len(mapped_joint_names), 3)
            if robot_points.shape != expected_shape:
                raise ValueError(
                    "Mapped robot skeleton trajectory must have shape "
                    f"{expected_shape}; got {robot_points.shape}",
                )
            point_indices = orientation_skeleton_point_indices(
                diagnostics,
                tuple(mapped_joint_names),
                tuple(mapped_robot_link_names),
            )[selected]
            robot_retargeting_overlay = SavedOrientationAxesOverlay(
                server=server,
                namespace="/overlays/orientation/robot/retargeting",
                names=tuple(diagnostics.robot_link_names[int(index)] for index in selected),
                positions=robot_points[:, point_indices],
                quaternions_wxyz=diagnostics.robot_quaternions_wxyz[
                    :,
                    selected,
                ],
                axis_length=config.orientation_axis_length,
                shaft_radius=config.orientation_axis_shaft_radius,
                head_radius=config.orientation_axis_head_radius,
                head_length=config.orientation_axis_head_length,
                visible=False,
                loop=config.loop,
            )

    robot_link_names_value = npz_metadata.get("robot_link_names")
    robot_link_positions = npz_metadata.get("robot_link_positions")
    robot_link_quaternions = npz_metadata.get("robot_link_quaternions_wxyz")
    if (
        isinstance(robot_link_names_value, list)
        and robot_link_positions is not None
        and robot_link_quaternions is not None
    ):
        robot_link_names = tuple(robot_link_names_value)
        aliases = (
            dict(zip(mapped_joint_names, mapped_robot_link_names))
            if isinstance(mapped_joint_names, list) and isinstance(mapped_robot_link_names, list)
            else {}
        )
        if isinstance(diagnostics, OrientationDiagnostics):
            aliases.update(
                zip(
                    diagnostics.human_joint_names,
                    diagnostics.robot_link_names,
                    strict=True,
                ),
            )
        selected_all = _requested_saved_orientation_indices(
            robot_link_names,
            config.orientation_joints,
            aliases=aliases,
        )
        retargeting_robot_names = (
            preview.robot_link_names
            if isinstance(preview, OrientationPreview)
            else (
                tuple(
                    dict.fromkeys(
                        (*tuple(mapped_robot_link_names or ()), *diagnostics.robot_link_names),
                    ),
                )
                if isinstance(diagnostics, OrientationDiagnostics)
                else tuple(mapped_robot_link_names or ())
            )
        )
        selected_retargeting = _requested_saved_orientation_indices(
            robot_link_names,
            config.orientation_joints or retargeting_robot_names,
            aliases=aliases,
        )
        if selected_all.size:
            robot_all_overlay = SavedOrientationAxesOverlay(
                server=server,
                namespace="/overlays/orientation/robot/all",
                names=tuple(robot_link_names[int(index)] for index in selected_all),
                positions=np.asarray(robot_link_positions)[:, selected_all],
                quaternions_wxyz=np.asarray(robot_link_quaternions)[
                    :,
                    selected_all,
                ],
                axis_length=config.orientation_axis_length,
                shaft_radius=config.orientation_axis_shaft_radius,
                head_radius=config.orientation_axis_head_radius,
                head_length=config.orientation_axis_head_length,
                visible=False,
                loop=config.loop,
            )
        if selected_retargeting.size:
            robot_retargeting_overlay = SavedOrientationAxesOverlay(
                server=server,
                namespace="/overlays/orientation/robot/retargeting",
                names=tuple(robot_link_names[int(index)] for index in selected_retargeting),
                positions=np.asarray(robot_link_positions)[:, selected_retargeting],
                quaternions_wxyz=np.asarray(robot_link_quaternions)[:, selected_retargeting],
                axis_length=config.orientation_axis_length,
                shaft_radius=config.orientation_axis_shaft_radius,
                head_radius=config.orientation_axis_head_radius,
                head_length=config.orientation_axis_head_length,
                visible=False,
                loop=config.loop,
            )

    overlays = ResultOrientationOverlays(
        source_all=source_all_overlay,
        source_retargeting=source_retargeting_overlay,
        target=target_overlay,
        robot_all=robot_all_overlay,
        robot_retargeting=robot_retargeting_overlay,
        scope=config.orientation_scope,
        source_visible=config.show_source_orientation_axes,
        robot_visible=config.show_robot_orientation_axes,
    )
    available_names = {
        "source_all": source_all_overlay.names if source_all_overlay is not None else (),
        "source_retargeting": (
            source_retargeting_overlay.names if source_retargeting_overlay is not None else ()
        ),
        "target": target_overlay.names if target_overlay is not None else (),
        "robot_all": robot_all_overlay.names if robot_all_overlay is not None else (),
        "robot_retargeting": (
            robot_retargeting_overlay.names if robot_retargeting_overlay is not None else ()
        ),
    }
    if any(available_names.values()):
        print(f"[viser_player] Saved orientation overlays enabled | {available_names}")
    return overlays


def _build_mapped_skeleton_overlay(
    config: ViserConfig,
    server: viser.ViserServer,
    human_joints: np.ndarray | None,
    npz_metadata: dict[str, object],
) -> MappedSkeletonOverlay | None:
    if human_joints is None:
        print("[viser_player] Mapped skeleton unavailable: result does not contain human_joints.")
        return None

    robot_xml_path = _resolve_robot_mujoco_xml(config)
    robot_type = _resolve_robot_type(config, npz_metadata)
    saved_demo_joints = npz_metadata.get("human_joint_names")
    saved_human_parent_indices = npz_metadata.get("human_joint_parent_indices")
    saved_mapped_human_joint_names = npz_metadata.get("mapped_human_joint_names")
    saved_mapped_robot_joints = npz_metadata.get("mapped_robot_joints")
    saved_mapped_robot_link_names = npz_metadata.get("mapped_robot_link_names")
    saved_robot_skeleton_joints = npz_metadata.get("robot_link_positions")
    saved_robot_parent_indices = npz_metadata.get("robot_link_parent_indices")

    if isinstance(saved_demo_joints, list) and len(saved_demo_joints) == human_joints.shape[1]:
        demo_joints = saved_demo_joints
        data_format = normalize_data_format(config.data_format) if config.data_format else "saved"
    else:
        data_format = _resolve_data_format(config, human_joints, robot_type, npz_metadata)
        if data_format is None:
            return None
        motion_data_config = MotionDataConfig(data_format=data_format, robot_type=robot_type)
        demo_joints = motion_data_config.resolved_demo_joints

    if (
        isinstance(saved_mapped_human_joint_names, list)
        and isinstance(saved_mapped_robot_link_names, list)
        and len(saved_mapped_human_joint_names) == len(saved_mapped_robot_link_names)
    ):
        joints_mapping = dict(zip(saved_mapped_human_joint_names, saved_mapped_robot_link_names))
    else:
        if data_format == "saved":
            data_format = _resolve_data_format(config, human_joints, robot_type, npz_metadata)
            if data_format is None:
                return None
        motion_data_config = MotionDataConfig(data_format=data_format, robot_type=robot_type)
        joints_mapping = motion_data_config.resolved_joints_mapping

    mapped_robot_trajectory = (
        np.asarray(saved_mapped_robot_joints)
        if saved_mapped_robot_joints is not None
        and np.asarray(saved_mapped_robot_joints).ndim == 3
        and np.asarray(saved_mapped_robot_joints).shape[1] == len(joints_mapping)
        else None
    )
    robot_skeleton_trajectory = (
        np.asarray(saved_robot_skeleton_joints)
        if saved_robot_skeleton_joints is not None and saved_robot_parent_indices is not None
        else None
    )
    if robot_xml_path is None and mapped_robot_trajectory is None and robot_skeleton_trajectory is None:
        print(
            "[viser_player] Mapped skeleton unavailable: result has no saved "
            "robot positions and a robot MuJoCo XML could not be inferred."
        )
        return None

    overlay = MappedSkeletonOverlay(
        server=server,
        human_joints=human_joints,
        demo_joints=demo_joints,
        human_joint_parent_indices=(
            np.asarray(saved_human_parent_indices)
            if saved_human_parent_indices is not None
            else None
        ),
        joints_mapping=joints_mapping,
        robot_xml_path=robot_xml_path,
        point_radius=config.skeleton_point_radius,
        line_width=config.skeleton_line_width,
        mapped_robot_joints=mapped_robot_trajectory,
        robot_skeleton_joints=robot_skeleton_trajectory,
        robot_skeleton_parent_indices=(
            np.asarray(saved_robot_parent_indices) if saved_robot_parent_indices is not None else None
        ),
        loop=config.loop,
    )
    print(
        "[viser_player] Mapped skeleton overlay enabled | "
        f"data_format={data_format}, robot_type={robot_type}, links={len(joints_mapping)}"
    )
    return overlay


def _build_interaction_mesh_overlay(
    config: ViserConfig,
    server: viser.ViserServer,
    interaction_mesh: dict[str, np.ndarray | int] | None,
) -> InteractionMeshOverlay | None:
    if interaction_mesh is None:
        print("[viser_player] Interaction mesh unavailable: result does not contain saved interaction mesh data.")
        return None

    overlay = InteractionMeshOverlay(
        server=server,
        mesh_data=interaction_mesh,
        mode=config.interaction_mesh_mode,
        edge_mode=config.interaction_mesh_edges,
        line_width=config.interaction_mesh_line_width,
        loop=config.loop,
    )
    print(
        "[viser_player] Interaction mesh overlay enabled | "
        f"mode={config.interaction_mesh_mode}, edges={config.interaction_mesh_edges}, "
        f"frames={overlay.source_vertices.shape[0]}"
    )
    return overlay


def _build_object_keypoint_overlay(
    config: ViserConfig,
    server: viser.ViserServer,
    npz_metadata: dict[str, object],
) -> ObjectKeypointOverlay | None:
    keypoint_data = npz_metadata.get("object_keypoints")
    if not isinstance(keypoint_data, dict):
        print(
            "[viser_player] Object keypoints unavailable: result does not contain saved demo/target object keypoints."
        )
        return None

    overlay = ObjectKeypointOverlay(
        server=server,
        keypoint_data=keypoint_data,
        point_radius=config.object_keypoint_radius,
        loop=config.loop,
    )
    print(
        "[viser_player] Object keypoint overlay enabled | "
        f"frames={overlay.demo_world.shape[0]}, points={overlay.demo_world.shape[1]}"
    )
    return overlay


def _build_retargeting_point_cloud_overlay(
    config: ViserConfig,
    server: viser.ViserServer,
    npz_metadata: dict[str, object],
) -> RetargetingPointCloudOverlay | None:
    trajectories: dict[str, np.ndarray] = {}
    for layer_name, field_name in (
        ("human", "human_points_world"),
        ("robot", "robot_points_world"),
        ("terrain", "terrain_points_world"),
    ):
        value = npz_metadata.get(field_name)
        if isinstance(value, np.ndarray):
            trajectories[layer_name] = value
    object_keypoints = npz_metadata.get("object_keypoints")
    if isinstance(object_keypoints, dict):
        for layer_name, field_name in (
            ("object_demo", "demo_world"),
            ("object_target", "target_world"),
        ):
            value = object_keypoints.get(field_name)
            if isinstance(value, np.ndarray):
                trajectories[layer_name] = value
    if not trajectories:
        print("[viser_player] Retargeting point clouds unavailable in this result.")
        return None
    overlay = RetargetingPointCloudOverlay(
        server,
        trajectories,
        config.point_cloud_point_size,
        loop=config.loop,
    )
    print(f"[viser_player] Retargeting point-cloud layer enabled | content={','.join(overlay.trajectories)}")
    return overlay


def make_player(
    config: ViserConfig,
    qpos: np.ndarray,
    human_joints: np.ndarray | None = None,
    fps: float | None = None,
    npz_metadata: dict[str, object] | None = None,
    interaction_mesh: dict[str, np.ndarray | int] | None = None,
):
    """
    qpos layout (MuJoCo order):
      [0:3]   robot base position (xyz)
      [3:7]   robot base quat (wxyz)
      [7:7+R] robot joint positions (R = actuated dof)
      [end-7:end-4] (optional) object position (xyz)
      [end-4:end]   (optional) object quat (wxyz)

    We'll infer R from the robot URDF's actuated joints in ViserUrdf.
    """
    server = viser.ViserServer()
    tabs = add_visualization_tabs(server.gui, include_motions=False)

    # Root frames
    robot_root = server.scene.add_frame("/robot", show_axes=False)
    object_root = server.scene.add_frame("/object", show_axes=False)

    # URDFs (using yourdfpy so meshes show up)
    robot_mesh_opacity = config.mesh_opacity if config.robot_mesh_opacity is None else config.robot_mesh_opacity
    object_mesh_opacity = config.mesh_opacity if config.object_mesh_opacity is None else config.object_mesh_opacity
    robot_mesh_color_override = _mesh_color_override(robot_mesh_opacity)
    object_mesh_color_override = _mesh_color_override(object_mesh_opacity)
    if config.robot_urdf is None:
        raise ValueError("robot_urdf must be resolved before building the Viser player")
    robot_urdf_y = yourdfpy.URDF.load(config.robot_urdf, load_meshes=True, build_scene_graph=True)
    vr = ViserUrdf(
        server,
        urdf_or_path=robot_urdf_y,
        root_node_name="/robot",
        mesh_color_override=robot_mesh_color_override,
    )

    vo = None
    if config.object_urdf:
        object_urdf_y = yourdfpy.URDF.load(config.object_urdf, load_meshes=True, build_scene_graph=True)
        vo = ViserUrdf(
            server,
            urdf_or_path=object_urdf_y,
            root_node_name="/object",
            mesh_color_override=object_mesh_color_override,
        )

    # A tiny grid
    server.scene.add_grid("/grid", width=config.grid_width, height=config.grid_height, position=(0.0, 0.0, 0.0))

    mapped_skeleton_overlay = _build_mapped_skeleton_overlay(config, server, human_joints, npz_metadata or {})
    object_keypoint_overlay = _build_object_keypoint_overlay(config, server, npz_metadata or {})
    point_cloud_overlay = _build_retargeting_point_cloud_overlay(
        config,
        server,
        npz_metadata or {},
    )
    interaction_mesh_overlay = _build_interaction_mesh_overlay(config, server, interaction_mesh)
    orientation_overlay = _build_orientation_overlay(
        config,
        server,
        human_joints,
        npz_metadata or {},
        int(qpos.shape[0]),
    )

    # Figure robot DOF from actuated limits in ViserUrdf
    joint_limits = vr.get_actuated_joint_limits()
    robot_dof = len(joint_limits)
    qpos_to_viser_joint_indices = _build_qpos_to_viser_joint_indices(
        config,
        list(joint_limits.keys()),
        npz_metadata or {},
    )
    joint_angle_diagnostics = build_joint_angle_diagnostics(
        qpos,
        joint_limits,
        qpos_to_viser_joint_indices,
    )
    contains_object_in_qpos = config.assume_object_in_qpos
    if contains_object_in_qpos is None:
        saved_contains_object = (npz_metadata or {}).get("contains_object_in_qpos")
        if saved_contains_object is not None:
            contains_object_in_qpos = bool(saved_contains_object)
        else:
            contains_object_in_qpos = qpos.shape[1] == 7 + robot_dof + 7
    contains_object_in_qpos = bool(contains_object_in_qpos) and vo is not None

    # Use fps from config if not provided, otherwise use the one from npz file
    actual_fps = fps if fps is not None else config.fps

    # Set initial visibility. Legacy flags remain aliases for the canonical layers.
    show_robot_mesh = config.show_meshes if config.show_robot_mesh is None else config.show_robot_mesh
    show_object_mesh = config.show_meshes if config.show_object_mesh is None else config.show_object_mesh
    show_human_skeleton = (
        config.show_mapped_skeletons if config.show_human_skeleton is None else config.show_human_skeleton
    )
    show_robot_skeleton = (
        config.show_mapped_skeletons if config.show_robot_skeleton is None else config.show_robot_skeleton
    )
    show_human_hands = bool(show_human_skeleton) if config.show_human_hands is None else config.show_human_hands

    foot_sticking = (npz_metadata or {}).get("foot_sticking")
    foot_sticking_status_handle = None
    if foot_sticking is not None:
        foot_sticking_states = np.asarray(foot_sticking["states"], dtype=bool)
    last_rendered_frame: dict[str, np.ndarray | float | None] = {"q": None, "frame": None}
    joint_angle_gui = None

    def _redraw_mapped_skeleton() -> None:
        if (
            mapped_skeleton_overlay is not None
            and last_rendered_frame["q"] is not None
            and last_rendered_frame["frame"] is not None
        ):
            mapped_skeleton_overlay.draw(
                np.asarray(last_rendered_frame["q"]),
                float(last_rendered_frame["frame"]),
            )

    def _set_human_skeleton_visibility(visible: bool) -> None:
        if mapped_skeleton_overlay is not None:
            mapped_skeleton_overlay.set_human_visible(visible)
            _redraw_mapped_skeleton()

    def _set_robot_skeleton_visibility(visible: bool) -> None:
        if mapped_skeleton_overlay is not None:
            mapped_skeleton_overlay.set_robot_visible(visible)
            _redraw_mapped_skeleton()

    def _set_human_hands_visibility(visible: bool) -> None:
        if mapped_skeleton_overlay is not None:
            mapped_skeleton_overlay.set_hands_visible(visible)
            _redraw_mapped_skeleton()

    def _set_joint_label_visibility(visible: bool) -> None:
        if mapped_skeleton_overlay is not None:
            mapped_skeleton_overlay.set_labels_visible(visible)
            _redraw_mapped_skeleton()

    def _set_object_keypoints_visibility(visible: bool) -> None:
        if object_keypoint_overlay is None:
            return
        object_keypoint_overlay.set_visible(visible)
        if visible and last_rendered_frame["frame"] is not None:
            object_keypoint_overlay.draw(float(last_rendered_frame["frame"]))

    def _set_interaction_mesh_visibility(visible: bool) -> None:
        if interaction_mesh_overlay is None:
            return
        interaction_mesh_overlay.set_visible(visible)
        if visible and last_rendered_frame["frame"] is not None:
            interaction_mesh_overlay.draw(float(last_rendered_frame["frame"]))

    def _set_point_cloud_visibility(visible: bool) -> None:
        if point_cloud_overlay is None:
            return
        point_cloud_overlay.set_visible(visible)
        if visible and last_rendered_frame["frame"] is not None:
            point_cloud_overlay.draw(float(last_rendered_frame["frame"]))

    def _set_foot_sticking_visibility(visible: bool) -> None:
        if foot_sticking_status_handle is not None:
            foot_sticking_status_handle.visible = bool(visible)

    layer_controller = LayerController()
    layer_controller.register(
        LayerId.ROBOT_MESH,
        available=True,
        visible=bool(show_robot_mesh),
        callback=lambda visible: setattr(vr, "show_visual", bool(visible)),
    )
    layer_controller.register(
        LayerId.OBJECT_MESH,
        available=vo is not None,
        visible=bool(show_object_mesh),
        callback=lambda visible: setattr(vo, "show_visual", bool(visible)) if vo is not None else None,
        unavailable_reason="No object or scene asset was resolved.",
    )
    layer_controller.register(
        LayerId.HUMAN_SKELETON,
        available=mapped_skeleton_overlay is not None,
        visible=bool(show_human_skeleton),
        callback=_set_human_skeleton_visibility,
        unavailable_reason="Mapped source-human joints are not available.",
    )
    layer_controller.register(
        LayerId.ROBOT_SKELETON,
        available=mapped_skeleton_overlay is not None,
        visible=bool(show_robot_skeleton),
        callback=_set_robot_skeleton_visibility,
        unavailable_reason="Mapped robot-link trajectories are not available.",
    )
    hands_available = bool(
        mapped_skeleton_overlay is not None
        and mapped_skeleton_overlay.hand_visualization_spec.keypoint_indices
        and mapped_skeleton_overlay.hand_visualization_spec.edge_indices
    )
    layer_controller.register(
        LayerId.HUMAN_HANDS,
        available=hands_available,
        visible=bool(show_human_hands),
        callback=_set_human_hands_visibility,
        unavailable_reason="The source joint set has no registered finger chains.",
    )
    layer_controller.register(
        LayerId.OBJECT_KEYPOINTS,
        available=object_keypoint_overlay is not None,
        visible=config.show_object_keypoints,
        callback=_set_object_keypoints_visibility,
        unavailable_reason="Saved demonstration/target object samples are absent.",
    )
    layer_controller.register(
        LayerId.RETARGETING_POINT_CLOUDS,
        available=point_cloud_overlay is not None,
        visible=config.show_point_clouds,
        callback=_set_point_cloud_visibility,
        unavailable_reason="Compact retargeting point-cloud fields are absent.",
    )
    layer_controller.register(
        LayerId.INTERACTION_MESH,
        available=interaction_mesh_overlay is not None,
        visible=config.show_interaction_mesh,
        callback=_set_interaction_mesh_visibility,
        unavailable_reason="Saved interaction vertices and tetrahedra are absent.",
    )
    layer_controller.register(
        LayerId.FOOT_STICKING,
        available=foot_sticking is not None,
        visible=config.show_foot_sticking,
        callback=_set_foot_sticking_visibility,
        unavailable_reason="Saved foot-sticking detector states are absent.",
    )
    layer_controller.register(
        LayerId.TARGET_ORIENTATION,
        available=orientation_overlay.target is not None,
        visible=config.show_target_orientation_axes,
        callback=orientation_overlay.set_target_visible,
        unavailable_reason="Saved target orientation frames are absent.",
    )
    layer_controller.register(
        LayerId.ROBOT_ORIENTATION,
        available=orientation_overlay.robot_available,
        visible=config.show_robot_orientation_axes,
        callback=orientation_overlay.set_robot_visible,
        unavailable_reason="Saved full robot link orientation frames are absent.",
    )
    layer_controller.register(
        LayerId.SOURCE_ORIENTATION,
        available=orientation_overlay.source_available,
        visible=config.show_source_orientation_axes,
        callback=orientation_overlay.set_source_visible,
        unavailable_reason=("The result does not contain a directly observed source-human orientation subset."),
    )
    layer_controller.register(
        LayerId.JOINT_LABELS,
        available=mapped_skeleton_overlay is not None,
        visible=config.show_joint_labels,
        callback=_set_joint_label_visibility,
        unavailable_reason="A complete named source-human skeleton is absent.",
    )
    for unavailable_layer, reason in (
        (LayerId.BODY_COM, "Body-center trajectories are available for converted motion inputs."),
        (LayerId.BODY_VELOCITY, "Body velocities are available for converted motion inputs."),
    ):
        layer_controller.register(
            unavailable_layer,
            available=False,
            visible=False,
            callback=lambda _visible: None,
            unavailable_reason=reason,
        )

    with tabs.layers:
        layer_controller.add_gui(server.gui)
        if orientation_overlay.source_available or orientation_overlay.robot_available:
            with server.gui.add_folder("Orientation debug settings", expand_by_default=True):
                orientation_scope = server.gui.add_dropdown(
                    "Displayed link scope",
                    ("retargeting", "all"),
                    initial_value=config.orientation_scope,
                    hint=(
                        "retargeting shows only source/robot links participating in the map; "
                        "all shows every saved FBX joint and robot link. This changes visualization only."
                    ),
                )

                @orientation_scope.on_update
                def _(_event) -> None:
                    orientation_overlay.set_scope(str(orientation_scope.value))
                    if last_rendered_frame["frame"] is not None:
                        orientation_overlay.draw(float(last_rendered_frame["frame"]))

        if interaction_mesh_overlay is not None:
            with server.gui.add_folder("Interaction mesh settings", expand_by_default=False):
                interaction_mode = server.gui.add_dropdown(
                    "Content",
                    ("source", "target", "both"),
                    initial_value=config.interaction_mesh_mode,
                    hint="Choose source human-object edges, target robot-object edges, or both.",
                )
                interaction_edges = server.gui.add_dropdown(
                    "Edges",
                    ("cross", "all"),
                    initial_value=config.interaction_mesh_edges,
                    hint="Cross keeps only human/robot-to-object edges.",
                )

                @interaction_mode.on_update
                def _(_event) -> None:
                    interaction_mesh_overlay.mode = str(interaction_mode.value)
                    if layer_controller.is_visible(LayerId.INTERACTION_MESH):
                        _set_interaction_mesh_visibility(True)

                @interaction_edges.on_update
                def _(_event) -> None:
                    interaction_mesh_overlay.edge_mode = str(interaction_edges.value)
                    if layer_controller.is_visible(LayerId.INTERACTION_MESH):
                        _set_interaction_mesh_visibility(True)

        if foot_sticking is not None:
            with server.gui.add_folder("Foot sticking status", expand_by_default=True):
                foot_sticking_status_handle = server.gui.add_markdown(
                    format_foot_sticking_status(
                        0,
                        foot_sticking_states[0],
                        constraint_status=_saved_foot_sticking_constraint_status(
                            foot_sticking,
                            0,
                        ),
                    ),
                    visible=config.show_foot_sticking,
                )

    with tabs.style:
        server.gui.add_number(
            "Robot mesh opacity",
            initial_value=float(robot_mesh_opacity),
            min=0.0,
            max=1.0,
            step=0.05,
            disabled=True,
            hint="Set with --robot-mesh-opacity before loading the URDF.",
        )
        server.gui.add_number(
            "Object mesh opacity",
            initial_value=float(object_mesh_opacity),
            min=0.0,
            max=1.0,
            step=0.05,
            disabled=True,
            hint="Set with --object-mesh-opacity before loading the URDF.",
        )
        server.gui.add_number(
            "Skeleton line width",
            initial_value=float(config.skeleton_line_width),
            min=0.1,
            max=20.0,
            step=0.5,
            disabled=True,
            hint="Set with --skeleton-line-width before creating the scene.",
        )

    layer_controller.register_shortcuts(server)
    server._holosoma_layer_controller = layer_controller

    def _draw_frame_overlay(q: np.ndarray, frame_float: float) -> None:
        last_rendered_frame["q"] = np.asarray(q).copy()
        last_rendered_frame["frame"] = float(frame_float)
        if foot_sticking_status_handle is not None:
            foot_frame_count = np.asarray(foot_sticking["states"]).shape[0]
            if config.loop:
                foot_frame_idx = round(float(frame_float)) % foot_frame_count
            else:
                foot_frame_idx = int(
                    np.clip(
                        round(float(frame_float)),
                        0,
                        foot_frame_count - 1,
                    )
                )
            foot_sticking_status_handle.content = format_foot_sticking_status(
                foot_frame_idx,
                np.asarray(foot_sticking["states"], dtype=bool)[foot_frame_idx],
                constraint_status=_saved_foot_sticking_constraint_status(
                    foot_sticking,
                    foot_frame_idx,
                ),
            )
        if mapped_skeleton_overlay is not None:
            mapped_skeleton_overlay.draw(q, frame_float)
        if object_keypoint_overlay is not None:
            object_keypoint_overlay.draw(frame_float)
        if point_cloud_overlay is not None:
            point_cloud_overlay.draw(frame_float)
        if interaction_mesh_overlay is not None:
            interaction_mesh_overlay.draw(frame_float)
        orientation_overlay.draw(frame_float)
        if joint_angle_gui is not None:
            joint_angle_gui.update_frame(frame_float)

    # ---------- Use reusable motion control sliders from viser_utils ----------
    with tabs.playback:
        create_motion_control_sliders(
            server=server,
            viser_robot=vr,
            robot_base_frame=robot_root,
            motion_sequence=qpos,
            robot_dof=robot_dof,
            viser_object=vo if contains_object_in_qpos else None,
            object_base_frame=object_root if contains_object_in_qpos else None,
            contains_object_in_qpos=contains_object_in_qpos,
            initial_fps=actual_fps,
            initial_interp_mult=config.visual_fps_multiplier,
            loop=config.loop,
            qpos_to_viser_joint_indices=qpos_to_viser_joint_indices,
            on_frame=_draw_frame_overlay,
        )
        joint_angle_gui = add_joint_angle_gui(server.gui, joint_angle_diagnostics)
    n_frames = int(qpos.shape[0])
    print(
        f"[viser_player] Loaded {n_frames} frames | robot_dof={robot_dof} | "
        f"object={'yes' if contains_object_in_qpos else 'no'}"
    )
    print("Open the viewer URL printed above. Close the process (Ctrl+C) to exit.")
    return server


def resolve_input_kind(path: str | Path, requested_kind: str = "auto") -> str:
    """Resolve the public single-motion input adapter."""

    if requested_kind not in {"auto", "result", "raw", "converted"}:
        raise ValueError("input_kind must be one of auto, result, raw, or converted")
    if requested_kind != "auto":
        return requested_kind
    input_path = Path(path).expanduser()
    if input_path.is_dir() or input_path.suffix.lower() in {".pt", ".npy"}:
        return "raw"
    if input_path.suffix.lower() != ".npz":
        raise ValueError(f"Cannot infer visualization input kind from {input_path}; pass --input-kind explicitly.")
    with np.load(input_path, allow_pickle=False) as data:
        keys = set(data.files)
    if "qpos" in keys:
        return "result"
    if {"joint_pos", "body_pos_w", "body_lin_vel_w"}.issubset(keys):
        return "converted"
    if "global_joint_positions" in keys:
        return "raw"
    raise ValueError(
        f"Cannot infer visualization input kind from NPZ fields in {input_path}. "
        "Expected qpos, converted body trajectories, or global_joint_positions."
    )


def main(cfg: ViserConfig) -> None:
    """Run the unified single-motion viewer for any supported input adapter."""

    if cfg.input_path is not None and cfg.qpos_npz is not None:
        raise ValueError(
            "Pass either --input-path or the legacy --qpos-npz option, not both.",
        )
    input_path = cfg.input_path or cfg.qpos_npz
    if input_path is None:
        raise ValueError(
            "Single-motion visualization requires --input-path (or the legacy --qpos-npz option).",
        )
    input_kind = resolve_input_kind(input_path, cfg.input_kind)
    if input_kind == "raw":
        from holosoma_retargeting.visualization.raw_scene import (  # noqa: I001, PLC0415
            Config as RawViewerConfig,
            main as raw_viewer_main,
        )

        raw_viewer_main(
            RawViewerConfig(
                motion_path=Path(input_path),
                data_format=cfg.data_format or "auto",
                task_type=cfg.task_type,
                sequence=cfg.sequence,
                human_height=cfg.human_height,
                mesh_path=Path(cfg.source_mesh) if cfg.source_mesh else None,
                object_name=cfg.object_name,
                port=cfg.port,
                start_frame=cfg.start_frame,
                playing=cfg.playing,
                loop=cfg.loop,
                point_size=cfg.skeleton_point_radius * 1.5,
                line_width=cfg.skeleton_line_width,
                mesh_opacity=(cfg.mesh_opacity if cfg.object_mesh_opacity is None else cfg.object_mesh_opacity),
                show_joint_labels=cfg.show_joint_labels,
                show_joint_orientations=cfg.show_source_orientation_axes,
                orientation_axis_length=cfg.orientation_axis_length,
                dry_run=cfg.dry_run,
            )
        )
        return
    if input_kind == "converted":
        from holosoma_retargeting.visualization.body_velocity_scene import (  # noqa: I001, PLC0415
            Config as BodyVelocityConfig,
            main as body_velocity_main,
        )

        robot_urdf = cfg.robot_urdf
        if robot_urdf is None and cfg.robot_type is not None:
            robot_urdf = RobotConfig(robot_type=cfg.robot_type).ROBOT_URDF_FILE
        if robot_urdf is None:
            raise ValueError("Converted motion visualization requires --robot-urdf or --robot-type.")
        robot_urdf = str(_resolve_asset_path(robot_urdf))
        body_velocity_main(
            BodyVelocityConfig(
                npz_path=str(input_path),
                robot_urdf=str(robot_urdf),
                grid_width=cfg.grid_width,
                grid_height=cfg.grid_height,
                show_meshes=cfg.show_meshes,
                show_body_com=cfg.show_body_com,
                show_body_velocity=cfg.show_body_velocity,
                playing=cfg.playing,
                loop=cfg.loop,
                vel_scale=cfg.velocity_scale,
            )
        )
        return

    cfg = replace(cfg, qpos_npz=str(input_path))
    qpos, fps, human_joints, npz_metadata, interaction_mesh = load_npz(cfg.qpos_npz)
    cfg = _resolve_runtime_config(cfg, npz_metadata)
    make_player(
        config=cfg,
        qpos=qpos,
        human_joints=human_joints,
        fps=fps,
        npz_metadata=npz_metadata,
        interaction_mesh=interaction_mesh,
    )

    # keep process alive
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    cfg = tyro.cli(ViserConfig)
    main(cfg)
