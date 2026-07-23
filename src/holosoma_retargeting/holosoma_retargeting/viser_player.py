#!/usr/bin/env python3
# viser_player.py
from __future__ import annotations

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
from holosoma_retargeting.config_types.data_type import (  # noqa: E402
    DEMO_JOINTS_REGISTRY,
    JOINTS_MAPPINGS,
    MotionDataConfig,
    normalize_data_format,
)
from holosoma_retargeting.config_types.robot import RobotConfig  # noqa: E402
from holosoma_retargeting.config_types.viser import ViserConfig  # noqa: E402
from holosoma_retargeting.src.utils import interaction_mesh_edges_from_tetrahedra  # noqa: E402
from holosoma_retargeting.src.viser_utils import (  # noqa: E402
    actuated_joint_names_from_mujoco_xml,
    build_joint_order_indices,
    create_motion_control_sliders,
    infer_mujoco_xml_path,
    register_keyboard_shortcut,
)


def load_npz(npz_path: str):
    with np.load(npz_path, allow_pickle=False) as data:
        # expected: qpos [T, ?], and optional fps
        qpos = data["qpos"]
        fps = float(data["fps"]) if "fps" in data else 30.0
        human_joints = data.get("human_joints")
        metadata = {
            "human_joint_names": _npz_string_list(data, "human_joint_names"),
            "mapped_human_joint_names": _npz_string_list(data, "mapped_human_joint_names"),
            "mapped_robot_joints": data.get("mapped_robot_joints"),
            "mapped_robot_link_names": _npz_string_list(data, "mapped_robot_link_names"),
            "source_data_format": _npz_scalar(data, "source_data_format"),
            "robot_type": _npz_scalar(data, "robot_type"),
            "object_name": _npz_scalar(data, "object_name"),
            "object_urdf": _npz_scalar(data, "object_urdf"),
            "contains_object_in_qpos": _npz_scalar(data, "contains_object_in_qpos"),
        }
        interaction_mesh = _load_interaction_mesh_npz(data)
    return qpos, fps, human_joints, metadata, interaction_mesh


def _load_interaction_mesh_npz(data) -> dict[str, np.ndarray | int] | None:
    required_keys = (
        "interaction_source_vertices_w",
        "interaction_target_vertices_w",
        "interaction_tetrahedra",
        "interaction_tetrahedra_counts",
        "interaction_num_human_vertices",
    )
    if not all(key in data for key in required_keys):
        return None

    return {
        "source_vertices": np.asarray(data["interaction_source_vertices_w"], dtype=np.float32),
        "target_vertices": np.asarray(data["interaction_target_vertices_w"], dtype=np.float32),
        "tetrahedra": np.asarray(data["interaction_tetrahedra"], dtype=np.int32),
        "tetrahedra_counts": np.asarray(data["interaction_tetrahedra_counts"], dtype=np.int32),
        "num_human_vertices": int(np.asarray(data["interaction_num_human_vertices"]).item()),
    }


def _npz_string_list(data, key: str) -> list[str] | None:
    if key not in data:
        return None
    value = np.asarray(data[key])
    if value.ndim == 0:
        return [str(value.item())]
    return [str(item) for item in value.tolist()]


def _npz_scalar(data, key: str):
    if key not in data:
        return None
    return np.asarray(data[key]).item()


def _resolve_runtime_config(config: ViserConfig, metadata: dict[str, object]) -> ViserConfig:
    robot_type = config.robot_type or metadata.get("robot_type")
    if robot_type is not None:
        robot_type = str(robot_type)

    robot_urdf = config.robot_urdf
    if robot_urdf is None:
        if robot_type is None:
            raise ValueError("Result has no robot_type metadata; pass --robot-urdf explicitly.")
        robot_urdf = RobotConfig(robot_type=robot_type).ROBOT_URDF_FILE

    object_urdf = config.object_urdf
    saved_object_urdf = metadata.get("object_urdf")
    if object_urdf is None and saved_object_urdf:
        object_urdf = str(saved_object_urdf)

    return replace(
        config,
        robot_type=robot_type,
        robot_urdf=robot_urdf,
        object_urdf=object_urdf,
    )


def _build_qpos_to_viser_joint_indices(config: ViserConfig, viser_joint_names: list[str]):
    if config.robot_urdf is None:
        raise ValueError("robot_urdf must be resolved before building the Viser player")
    xml_path = Path(config.robot_mujoco_xml) if config.robot_mujoco_xml else infer_mujoco_xml_path(config.robot_urdf)
    if xml_path is None:
        return None

    qpos_joint_names = actuated_joint_names_from_mujoco_xml(xml_path)
    if qpos_joint_names == viser_joint_names:
        return None

    print(f"[viser_player] Reordering qpos joints from MuJoCo XML order: {xml_path}")
    return build_joint_order_indices(qpos_joint_names, viser_joint_names)


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


def _interpolate_sequence(sequence: np.ndarray, frame_float: float) -> np.ndarray:
    n_frames = int(sequence.shape[0])
    if n_frames == 1:
        return sequence[0]

    frame_float = float(np.clip(frame_float, 0.0, n_frames - 1))
    i0 = int(np.floor(frame_float))
    i1 = min(i0 + 1, n_frames - 1)
    u = frame_float - i0
    return (1.0 - u) * sequence[i0] + u * sequence[i1]


class MappedSkeletonOverlay:
    def __init__(
        self,
        server: viser.ViserServer,
        human_joints: np.ndarray,
        demo_joints: list[str],
        joints_mapping: dict[str, str],
        robot_xml_path: Path,
        point_radius: float,
        line_width: float,
        mapped_robot_joints: np.ndarray | None = None,
    ) -> None:
        self.server = server
        self.human_joints = np.asarray(human_joints)
        self.demo_joints = demo_joints
        self.joints_mapping = joints_mapping
        self.human_joint_names = list(joints_mapping.keys())
        self.robot_link_names = list(joints_mapping.values())
        self.human_joint_indices = [demo_joints.index(name) for name in self.human_joint_names]
        self.mapped_robot_joints = (
            np.asarray(mapped_robot_joints, dtype=np.float32) if mapped_robot_joints is not None else None
        )
        self.mapped_edges = _mapped_skeleton_edges(self.human_joint_names)
        self.line_width = float(line_width)
        self.visible = True
        self._lock = threading.Lock()
        self._handles: list[object] = []

        self._sphere = trimesh.primitives.Sphere(radius=float(point_radius))
        self._sphere_vertices = self._sphere.vertices.astype(np.float32)
        self._sphere_faces = self._sphere.faces.astype(np.int32)

        self.robot_model = mujoco.MjModel.from_xml_path(str(robot_xml_path))
        self.robot_data = mujoco.MjData(self.robot_model)
        self.robot_body_ids = []
        missing_links: list[str] = []
        for link_name in self.robot_link_names:
            body_id = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, link_name)
            if body_id == -1:
                missing_links.append(link_name)
            self.robot_body_ids.append(body_id)
        if missing_links:
            raise ValueError(
                f"Mapped skeleton robot links are missing in MuJoCo model {robot_xml_path}: {missing_links}"
            )

    def set_visible(self, visible: bool) -> None:
        with self._lock:
            self.visible = bool(visible)
            if not self.visible:
                self._clear_locked()

    def draw(self, q: np.ndarray, frame_float: float) -> None:
        with self._lock:
            if not self.visible:
                return

            self._clear_locked()
            human_points = self._human_points(frame_float)
            robot_points = self._robot_points(q, frame_float)
            self._handles.extend(
                [
                    self._draw_points("/overlays/mapped/human_kpts", human_points, color=(0, 0, 255)),
                    self._draw_points("/overlays/mapped/robot_kpts", robot_points, color=(0, 255, 0)),
                ]
            )
            human_skeleton = self._draw_skeleton(
                "/overlays/mapped/human_skeleton",
                human_points,
                self.mapped_edges,
                color=np.array([0.0, 0.0, 1.0]),
                line_width=self.line_width,
            )
            robot_skeleton = self._draw_skeleton(
                "/overlays/mapped/robot_skeleton",
                robot_points,
                self.mapped_edges,
                color=np.array([0.0, 1.0, 0.0]),
                line_width=self.line_width,
            )
            if human_skeleton is not None:
                self._handles.append(human_skeleton)
            if robot_skeleton is not None:
                self._handles.append(robot_skeleton)

    def _clear_locked(self) -> None:
        for handle in self._handles:
            with suppress(Exception):
                handle.remove()
        self._handles.clear()

    def _human_points(self, frame_float: float) -> np.ndarray:
        human_frame = _interpolate_sequence(self.human_joints, frame_float)
        return np.asarray(human_frame[self.human_joint_indices], dtype=np.float32)

    def _robot_points(self, q: np.ndarray, frame_float: float) -> np.ndarray:
        if self.mapped_robot_joints is not None:
            return np.asarray(_interpolate_sequence(self.mapped_robot_joints, frame_float), dtype=np.float32)

        q = np.asarray(q, dtype=float)
        model_q = np.zeros(self.robot_model.nq, dtype=float)
        if q.shape[0] >= self.robot_model.nq:
            model_q[:] = q[: self.robot_model.nq]
        else:
            model_q[: q.shape[0]] = q

        self.robot_data.qpos[:] = model_q
        mujoco.mj_forward(self.robot_model, self.robot_data)
        return self.robot_data.xpos[np.asarray(self.robot_body_ids, dtype=int)].copy().astype(np.float32)

    def _draw_points(self, name: str, points: np.ndarray, color: tuple[int, int, int]):
        return self.server.scene.add_batched_meshes_simple(
            name,
            vertices=self._sphere_vertices,
            faces=self._sphere_faces,
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


class InteractionMeshOverlay:
    def __init__(
        self,
        server: viser.ViserServer,
        mesh_data: dict[str, np.ndarray | int],
        mode: str,
        edge_mode: str,
        line_width: float,
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
            frame_idx = int(np.clip(round(float(frame_float)), 0, self.source_vertices.shape[0] - 1))
            tet_count = int(self.tetrahedra_counts[frame_idx])
            frame_tetrahedra = self.tetrahedra[frame_idx, :tet_count]
            if self.mode in {"source", "both"}:
                self._handles.extend(
                    self._draw_mesh(
                        "/overlays/interaction_mesh/source",
                        self.source_vertices[frame_idx],
                        frame_tetrahedra,
                        color=np.array([1.0, 0.55, 0.0], dtype=np.float32),
                    )
                )
            if self.mode in {"target", "both"}:
                self._handles.extend(
                    self._draw_mesh(
                        "/overlays/interaction_mesh/target",
                        self.target_vertices[frame_idx],
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


def _build_mapped_skeleton_overlay(
    config: ViserConfig,
    server: viser.ViserServer,
    human_joints: np.ndarray | None,
    npz_metadata: dict[str, object],
) -> MappedSkeletonOverlay | None:
    if not config.show_mapped_skeletons:
        return None
    if human_joints is None:
        print("[viser_player] --show_mapped_skeletons requested, but qpos npz does not contain human_joints.")
        return None

    robot_xml_path = _resolve_robot_mujoco_xml(config)
    if robot_xml_path is None:
        print("[viser_player] --show_mapped_skeletons requested, but robot MuJoCo XML could not be inferred.")
        return None

    robot_type = _resolve_robot_type(config, npz_metadata)
    saved_demo_joints = npz_metadata.get("human_joint_names")
    saved_mapped_human_joint_names = npz_metadata.get("mapped_human_joint_names")
    saved_mapped_robot_joints = npz_metadata.get("mapped_robot_joints")
    saved_mapped_robot_link_names = npz_metadata.get("mapped_robot_link_names")

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

    overlay = MappedSkeletonOverlay(
        server=server,
        human_joints=human_joints,
        demo_joints=demo_joints,
        joints_mapping=joints_mapping,
        robot_xml_path=robot_xml_path,
        point_radius=config.skeleton_point_radius,
        line_width=config.skeleton_line_width,
        mapped_robot_joints=(
            np.asarray(saved_mapped_robot_joints)
            if saved_mapped_robot_joints is not None
            and np.asarray(saved_mapped_robot_joints).ndim == 3
            and np.asarray(saved_mapped_robot_joints).shape[1] == len(joints_mapping)
            else None
        ),
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
    if not config.show_interaction_mesh:
        return None
    if interaction_mesh is None:
        print(
            "[viser_player] --show-interaction-mesh requested, but qpos npz does not contain saved "
            "interaction mesh data. "
            "Re-run retargeting with --retargeter.save-interaction-mesh."
        )
        return None

    overlay = InteractionMeshOverlay(
        server=server,
        mesh_data=interaction_mesh,
        mode=config.interaction_mesh_mode,
        edge_mode=config.interaction_mesh_edges,
        line_width=config.interaction_mesh_line_width,
    )
    print(
        "[viser_player] Interaction mesh overlay enabled | "
        f"mode={config.interaction_mesh_mode}, edges={config.interaction_mesh_edges}, "
        f"frames={overlay.source_vertices.shape[0]}"
    )
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
    interaction_mesh_overlay = _build_interaction_mesh_overlay(config, server, interaction_mesh)

    # Figure robot DOF from actuated limits in ViserUrdf
    joint_limits = vr.get_actuated_joint_limits()
    robot_dof = len(joint_limits)
    qpos_to_viser_joint_indices = _build_qpos_to_viser_joint_indices(config, list(joint_limits.keys()))
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

    # Set initial mesh visibility. show_meshes remains a backward-compatible default.
    show_robot_mesh = config.show_meshes if config.show_robot_mesh is None else config.show_robot_mesh
    show_object_mesh = config.show_meshes if config.show_object_mesh is None else config.show_object_mesh
    vr.show_visual = bool(show_robot_mesh)
    if vo is not None:
        vo.show_visual = bool(show_object_mesh)

    # ---------- Additional GUI controls (mesh visibility) ----------
    with server.gui.add_folder("Display"):
        show_robot_mesh_cb = server.gui.add_checkbox("Show robot mesh", initial_value=bool(show_robot_mesh))
        show_object_mesh_cb = (
            server.gui.add_checkbox("Show object mesh", initial_value=bool(show_object_mesh))
            if vo is not None
            else None
        )
        show_mapped_skeletons_cb = (
            server.gui.add_checkbox("Show mapped skeletons", initial_value=True)
            if mapped_skeleton_overlay is not None
            else None
        )
        show_interaction_mesh_cb = (
            server.gui.add_checkbox("Show interaction mesh", initial_value=True)
            if interaction_mesh_overlay is not None
            else None
        )

    updating_robot_mesh_checkbox = {"flag": False}
    updating_object_mesh_checkbox = {"flag": False}
    updating_skeleton_checkbox = {"flag": False}
    updating_interaction_mesh_checkbox = {"flag": False}
    last_rendered_frame: dict[str, np.ndarray | float | None] = {"q": None, "frame": None}

    def _set_robot_mesh_visibility(visible: bool, *, sync_checkbox: bool = True) -> None:
        if sync_checkbox:
            updating_robot_mesh_checkbox["flag"] = True
            try:
                show_robot_mesh_cb.value = bool(visible)
            finally:
                updating_robot_mesh_checkbox["flag"] = False
        vr.show_visual = bool(visible)

    def _set_object_mesh_visibility(visible: bool, *, sync_checkbox: bool = True) -> None:
        if vo is None or show_object_mesh_cb is None:
            return
        if sync_checkbox:
            updating_object_mesh_checkbox["flag"] = True
            try:
                show_object_mesh_cb.value = bool(visible)
            finally:
                updating_object_mesh_checkbox["flag"] = False
            vo.show_visual = bool(visible)
        else:
            vo.show_visual = bool(visible)

    @show_robot_mesh_cb.on_update
    def _(_):
        if not updating_robot_mesh_checkbox["flag"]:
            _set_robot_mesh_visibility(bool(show_robot_mesh_cb.value), sync_checkbox=False)

    if show_object_mesh_cb is not None:

        @show_object_mesh_cb.on_update
        def _(_):
            if not updating_object_mesh_checkbox["flag"]:
                _set_object_mesh_visibility(bool(show_object_mesh_cb.value), sync_checkbox=False)

    register_keyboard_shortcut(
        server,
        "Display: Toggle Robot Mesh",
        hotkey=",",
        callback=lambda: _set_robot_mesh_visibility(not bool(show_robot_mesh_cb.value)),
    )
    if show_object_mesh_cb is not None:
        register_keyboard_shortcut(
            server,
            "Display: Toggle Object Mesh",
            hotkey="o",
            callback=lambda: _set_object_mesh_visibility(not bool(show_object_mesh_cb.value)),
        )

    if show_mapped_skeletons_cb is not None:

        def _set_mapped_skeleton_visibility(visible: bool, *, sync_checkbox: bool = True) -> None:
            if sync_checkbox:
                updating_skeleton_checkbox["flag"] = True
                try:
                    show_mapped_skeletons_cb.value = bool(visible)
                finally:
                    updating_skeleton_checkbox["flag"] = False
            mapped_skeleton_overlay.set_visible(bool(visible))
            if visible and last_rendered_frame["q"] is not None and last_rendered_frame["frame"] is not None:
                mapped_skeleton_overlay.draw(
                    np.asarray(last_rendered_frame["q"]),
                    float(last_rendered_frame["frame"]),
                )

        @show_mapped_skeletons_cb.on_update
        def _(_):
            if not updating_skeleton_checkbox["flag"]:
                _set_mapped_skeleton_visibility(bool(show_mapped_skeletons_cb.value), sync_checkbox=False)

        register_keyboard_shortcut(
            server,
            "Display: Toggle Mapped Skeletons",
            hotkey=".",
            callback=lambda: _set_mapped_skeleton_visibility(not bool(show_mapped_skeletons_cb.value)),
        )

    if show_interaction_mesh_cb is not None:

        def _set_interaction_mesh_visibility(visible: bool, *, sync_checkbox: bool = True) -> None:
            if sync_checkbox:
                updating_interaction_mesh_checkbox["flag"] = True
                try:
                    show_interaction_mesh_cb.value = bool(visible)
                finally:
                    updating_interaction_mesh_checkbox["flag"] = False
            interaction_mesh_overlay.set_visible(bool(visible))
            if visible and last_rendered_frame["frame"] is not None:
                interaction_mesh_overlay.draw(float(last_rendered_frame["frame"]))

        @show_interaction_mesh_cb.on_update
        def _(_):
            if not updating_interaction_mesh_checkbox["flag"]:
                _set_interaction_mesh_visibility(bool(show_interaction_mesh_cb.value), sync_checkbox=False)

        register_keyboard_shortcut(
            server,
            "Display: Toggle Interaction Mesh",
            hotkey="m",
            callback=lambda: _set_interaction_mesh_visibility(not bool(show_interaction_mesh_cb.value)),
        )

    def _draw_frame_overlay(q: np.ndarray, frame_float: float) -> None:
        last_rendered_frame["q"] = np.asarray(q).copy()
        last_rendered_frame["frame"] = float(frame_float)
        if mapped_skeleton_overlay is not None:
            mapped_skeleton_overlay.draw(q, frame_float)
        if interaction_mesh_overlay is not None:
            interaction_mesh_overlay.draw(frame_float)

    # ---------- Use reusable motion control sliders from viser_utils ----------
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
        on_frame=(
            _draw_frame_overlay
            if (mapped_skeleton_overlay is not None or interaction_mesh_overlay is not None)
            else None
        ),
    )
    n_frames = int(qpos.shape[0])
    print(
        f"[viser_player] Loaded {n_frames} frames | robot_dof={robot_dof} | "
        f"object={'yes' if contains_object_in_qpos else 'no'}"
    )
    print("Open the viewer URL printed above. Close the process (Ctrl+C) to exit.")
    return server


def main(cfg: ViserConfig) -> None:
    """Main function for viser player."""
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
