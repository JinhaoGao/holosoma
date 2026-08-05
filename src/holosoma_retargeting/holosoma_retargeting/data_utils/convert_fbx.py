# ruff: noqa: CPY001

"""Convert multi-actor FBX motion files into canonical mocap NPZ motions.

The converter uses Assimp only as an FBX transform-stack evaluator. Assimp
exports a temporary GLB whose animation contains local translations,
quaternions, and scales after FBX pivots, pre-rotations, post-rotations, and
Euler rotation order have been resolved. This module performs FK itself,
splits every Hips-rooted skeleton, converts centimetre Y-up data to the shared
metre Z-up scene, and writes one ``fbx_mocap`` NPZ per actor.
"""

from __future__ import annotations

import argparse
import json
import math
import mmap
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Self, Sequence

import numpy as np
from holosoma_retargeting.config_types.data_type import (
    DEMO_JOINT_PARENT_INDICES,
    FBX_MOCAP_DEMO_JOINTS,
)
from scipy.spatial.transform import Rotation

_GLB_MAGIC = b"glTF"
_JSON_CHUNK = b"JSON"
_BIN_CHUNK = b"BIN\x00"
_COMPONENT_DTYPES = {
    5120: np.dtype("i1"),
    5121: np.dtype("u1"),
    5122: np.dtype("<i2"),
    5123: np.dtype("<u2"),
    5125: np.dtype("<u4"),
    5126: np.dtype("<f4"),
}
_ACCESSOR_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}
_FBX_Y_UP_TO_RIGHT_HANDED_Z_UP = np.asarray(
    (
        (1.0, 0.0, 0.0),
        (0.0, 0.0, -1.0),
        (0.0, 1.0, 0.0),
    ),
    dtype=np.float64,
)
_FBX_CENTIMETRES_TO_METRES = 0.01
_DIRECT_ORIENTATION_SOURCE = "fbx_local_rotation_curves_fk"
_SOURCE_JOINT_BY_CANONICAL = {
    **{name: name for name in FBX_MOCAP_DEMO_JOINTS},
    "LeftFootMod": "LToeEnd",
    "RightFootMod": "RToeEnd",
}
_ORIENTATION_JOINT_NAMES = tuple(name for name in FBX_MOCAP_DEMO_JOINTS if name not in {"LeftFootMod", "RightFootMod"})


@dataclass(frozen=True)
class ConvertedActor:
    """One actor-level canonical motion written from a multi-actor FBX."""

    actor: str
    output_path: Path
    frame_count: int
    fps: float
    height: float


@dataclass(frozen=True)
class _NodeDefaults:
    translation: np.ndarray
    rotation_xyzw: np.ndarray
    scale: np.ndarray


@dataclass(frozen=True)
class _SampledScene:
    times: np.ndarray
    source_fps: float
    animated_rotation_nodes: frozenset[int]
    positions: tuple[np.ndarray, ...]
    rotations_xyzw: tuple[np.ndarray, ...]
    scales: tuple[np.ndarray, ...]
    rest_positions: np.ndarray
    rest_rotations_xyzw: np.ndarray


@dataclass(frozen=True)
class _FbxGlobalSettings:
    up_axis: int
    up_axis_sign: int
    front_axis: int
    front_axis_sign: int
    coord_axis: int
    coord_axis_sign: int
    unit_scale_factor: float


class _GlbDocument:
    """Memory-mapped GLB JSON and binary accessor reader."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._file: BinaryIO | None = None
        self._mapping: mmap.mmap | None = None
        self.document: dict[str, object] = {}
        self._binary_offset = 0
        self._binary_length = 0

    def __enter__(self) -> Self:
        self._file = self.path.open("rb")
        self._mapping = mmap.mmap(self._file.fileno(), length=0, access=mmap.ACCESS_READ)
        if len(self._mapping) < 20:
            raise ValueError(f"GLB is truncated: {self.path}")
        magic, version, total_length = struct.unpack_from("<4sII", self._mapping, 0)
        if magic != _GLB_MAGIC or version != 2 or total_length != len(self._mapping):
            raise ValueError(f"Invalid GLB 2 header: {self.path}")
        json_length, json_type = struct.unpack_from("<I4s", self._mapping, 12)
        if json_type != _JSON_CHUNK:
            raise ValueError(f"GLB does not start with a JSON chunk: {self.path}")
        json_start = 20
        json_stop = json_start + json_length
        self.document = json.loads(self._mapping[json_start:json_stop])
        if json_stop + 8 > len(self._mapping):
            raise ValueError(f"GLB has no binary animation chunk: {self.path}")
        binary_length, binary_type = struct.unpack_from("<I4s", self._mapping, json_stop)
        if binary_type != _BIN_CHUNK or json_stop + 8 + binary_length > len(self._mapping):
            raise ValueError(f"Invalid GLB binary chunk: {self.path}")
        self._binary_offset = json_stop + 8
        self._binary_length = binary_length
        return self

    def __exit__(self, *_: object) -> None:
        if self._mapping is not None:
            self._mapping.close()
        if self._file is not None:
            self._file.close()

    def accessor(self, index: int) -> np.ndarray:
        """Return one dense, non-sparse GLB accessor as a copied array."""
        if self._mapping is None:
            raise RuntimeError("GLB document is not open")
        accessors = self.document.get("accessors")
        buffer_views = self.document.get("bufferViews")
        if not isinstance(accessors, list) or not isinstance(buffer_views, list):
            raise ValueError("GLB is missing accessors or bufferViews")
        accessor = accessors[index]
        if not isinstance(accessor, dict) or "bufferView" not in accessor:
            raise ValueError(f"Animation accessor {index} is not dense")
        if "sparse" in accessor:
            raise ValueError(f"Sparse animation accessor {index} is unsupported")
        view = buffer_views[int(accessor["bufferView"])]
        if not isinstance(view, dict) or int(view.get("buffer", 0)) != 0:
            raise ValueError(f"Animation accessor {index} uses an unsupported buffer")
        component_type = int(accessor["componentType"])
        try:
            dtype = _COMPONENT_DTYPES[component_type]
            components = _ACCESSOR_COMPONENTS[str(accessor["type"])]
        except KeyError as exc:
            raise ValueError(f"Unsupported GLB animation accessor {index}") from exc
        count = int(accessor["count"])
        item_bytes = dtype.itemsize * components
        stride = int(view.get("byteStride", item_bytes))
        if stride < item_bytes:
            raise ValueError(f"Invalid byteStride for accessor {index}")
        byte_offset = self._binary_offset + int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
        required = 0 if count == 0 else stride * (count - 1) + item_bytes
        if byte_offset < self._binary_offset or byte_offset + required > self._binary_offset + self._binary_length:
            raise ValueError(f"Animation accessor {index} exceeds the GLB binary chunk")
        array = np.ndarray(
            shape=(count, components),
            dtype=dtype,
            buffer=self._mapping,
            offset=byte_offset,
            strides=(stride, dtype.itemsize),
        )
        result = np.asarray(array).copy()
        return result[:, 0] if components == 1 else result


def _resolve_assimp_executable(explicit: Path | str | None) -> Path:
    if explicit is not None:
        executable = Path(explicit).expanduser()
        if not executable.is_file():
            raise FileNotFoundError(f"Assimp executable not found: {executable}")
        return executable
    discovered = shutil.which("assimp")
    if discovered is not None:
        return Path(discovered)
    environment_candidate = Path(sys.prefix) / "bin" / "assimp"
    if environment_candidate.is_file():
        return environment_candidate
    raise FileNotFoundError(
        "FBX conversion requires Assimp 5.4.x. Install the 'fbx' project extra "
        "or pass --assimp-executable /path/to/assimp."
    )


def _export_fbx_to_glb(fbx_path: Path, glb_path: Path, assimp_executable: Path) -> None:
    command = [
        str(assimp_executable),
        "export",
        str(fbx_path),
        str(glb_path),
        "-f",
        "glb2",
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0 or not glb_path.is_file():
        details = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"Assimp failed to convert {fbx_path}: {details}")


def _node_defaults(node: dict[str, object]) -> _NodeDefaults:
    if "matrix" not in node:
        return _NodeDefaults(
            translation=np.asarray(node.get("translation", (0.0, 0.0, 0.0)), dtype=np.float64),
            rotation_xyzw=_normalize_quaternions_xyzw(
                np.asarray(node.get("rotation", (0.0, 0.0, 0.0, 1.0)), dtype=np.float64)
            ),
            scale=np.asarray(node.get("scale", (1.0, 1.0, 1.0)), dtype=np.float64),
        )
    matrix_values = np.asarray(node["matrix"], dtype=np.float64)
    if matrix_values.shape != (16,):
        raise ValueError("GLB node matrix must contain 16 values")
    matrix = matrix_values.reshape(4, 4).T
    linear = matrix[:3, :3]
    scale = np.linalg.norm(linear, axis=0)
    if not np.isfinite(matrix).all() or np.any(scale <= 1e-12):
        raise ValueError("GLB node matrix must be finite and non-singular")
    rotation = linear / scale[None, :]
    if np.linalg.det(rotation) < 0.0:
        scale[0] *= -1.0
        rotation[:, 0] *= -1.0
    return _NodeDefaults(
        translation=matrix[:3, 3],
        rotation_xyzw=Rotation.from_matrix(rotation).as_quat(),
        scale=scale,
    )


def _normalize_quaternions_xyzw(quaternions: np.ndarray) -> np.ndarray:
    values = np.asarray(quaternions, dtype=np.float64)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if not np.isfinite(values).all() or np.any(norms <= 1e-12):
        raise ValueError("Animation quaternions must be finite and non-zero")
    return values / norms


def _sample_vectors(times: np.ndarray, values: np.ndarray, target_times: np.ndarray, interpolation: str) -> np.ndarray:
    if times.ndim != 1 or values.ndim != 2 or values.shape[0] != times.shape[0]:
        raise ValueError("Invalid vector animation sampler")
    if not np.isfinite(times).all() or not np.isfinite(values).all() or np.any(np.diff(times) <= 0.0):
        raise ValueError("Animation sampler times and values must be finite with increasing times")
    if interpolation == "STEP":
        indices = np.searchsorted(times, target_times, side="right") - 1
        return values[np.clip(indices, 0, len(times) - 1)]
    if interpolation != "LINEAR":
        raise ValueError(f"Unsupported GLB animation interpolation: {interpolation}")
    return np.stack(
        [np.interp(target_times, times, values[:, axis]) for axis in range(values.shape[1])],
        axis=-1,
    )


def _sample_quaternions(
    times: np.ndarray,
    quaternions_xyzw: np.ndarray,
    target_times: np.ndarray,
    interpolation: str,
) -> np.ndarray:
    values = _normalize_quaternions_xyzw(quaternions_xyzw)
    if times.ndim != 1 or values.shape != (times.shape[0], 4):
        raise ValueError("Invalid quaternion animation sampler")
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0.0):
        raise ValueError("Quaternion animation times must be finite and increasing")
    if interpolation == "STEP":
        indices = np.searchsorted(times, target_times, side="right") - 1
        return values[np.clip(indices, 0, len(times) - 1)]
    if interpolation != "LINEAR":
        raise ValueError(f"Unsupported GLB quaternion interpolation: {interpolation}")
    upper = np.searchsorted(times, target_times, side="right")
    upper = np.clip(upper, 1, len(times) - 1)
    lower = upper - 1
    before_start = target_times <= times[0]
    after_stop = target_times >= times[-1]
    interval = times[upper] - times[lower]
    alpha = np.divide(
        target_times - times[lower],
        interval,
        out=np.zeros_like(target_times, dtype=np.float64),
        where=interval > 0.0,
    )
    alpha = np.clip(alpha, 0.0, 1.0)
    first = values[lower]
    second = values[upper].copy()
    dots = np.sum(first * second, axis=-1)
    negative = dots < 0.0
    second[negative] *= -1.0
    dots = np.clip(np.abs(dots), 0.0, 1.0)
    angles = np.arccos(dots)
    sin_angles = np.sin(angles)
    near = sin_angles <= 1e-8
    first_weight = np.empty_like(alpha)
    second_weight = np.empty_like(alpha)
    first_weight[near] = 1.0 - alpha[near]
    second_weight[near] = alpha[near]
    first_weight[~near] = np.sin((1.0 - alpha[~near]) * angles[~near]) / sin_angles[~near]
    second_weight[~near] = np.sin(alpha[~near] * angles[~near]) / sin_angles[~near]
    sampled = first_weight[:, None] * first + second_weight[:, None] * second
    sampled[before_start] = values[0]
    sampled[after_stop] = values[-1]
    return _normalize_quaternions_xyzw(sampled)


def _quat_multiply_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = np.moveaxis(left, -1, 0)
    rx, ry, rz, rw = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        axis=-1,
    )


def _quat_rotate_xyzw(quaternion: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    xyz = quaternion[..., :3]
    uv = np.cross(xyz, vectors)
    uuv = np.cross(xyz, uv)
    return vectors + 2.0 * (quaternion[..., 3:4] * uv + uuv)


def _topological_nodes(
    nodes: list[dict[str, object]],
    scenes: object,
    selected_scene: int,
) -> tuple[list[int], np.ndarray]:
    parents = np.full(len(nodes), -1, dtype=np.int32)
    for parent_index, node in enumerate(nodes):
        for child in node.get("children", []):
            child_index = int(child)
            if parents[child_index] != -1:
                raise ValueError(f"GLB node {child_index} has multiple parents")
            parents[child_index] = parent_index
    roots: list[int] = []
    if isinstance(scenes, list) and scenes:
        scene = scenes[selected_scene]
        if isinstance(scene, dict):
            roots.extend(int(index) for index in scene.get("nodes", []))
    roots.extend(index for index, parent in enumerate(parents) if parent == -1 and index not in roots)
    order: list[int] = []
    state = np.zeros(len(nodes), dtype=np.int8)

    def visit(index: int) -> None:
        if state[index] == 1:
            raise ValueError("GLB node hierarchy contains a cycle")
        if state[index] == 2:
            return
        state[index] = 1
        order.append(index)
        for child in nodes[index].get("children", []):
            visit(int(child))
        state[index] = 2

    for root in roots:
        visit(root)
    if len(order) != len(nodes):
        raise ValueError("GLB hierarchy traversal did not reach every node")
    return order, parents


def _animation_channels(
    document: _GlbDocument,
) -> tuple[dict[tuple[int, str], tuple[np.ndarray, np.ndarray, str]], float, float]:
    animations = document.document.get("animations")
    if not isinstance(animations, list) or len(animations) != 1 or not isinstance(animations[0], dict):
        count = len(animations) if isinstance(animations, list) else 0
        raise ValueError(f"FBX conversion requires exactly one animation stack, found {count}")
    animation = animations[0]
    channels = animation.get("channels")
    samplers = animation.get("samplers")
    if not isinstance(channels, list) or not isinstance(samplers, list) or not channels:
        raise ValueError("FBX animation stack contains no channels")
    result: dict[tuple[int, str], tuple[np.ndarray, np.ndarray, str]] = {}
    duration = 0.0
    sample_intervals: list[float] = []
    for channel in channels:
        if not isinstance(channel, dict) or not isinstance(channel.get("target"), dict):
            raise ValueError("Invalid GLB animation channel")
        target = channel["target"]
        node_index = int(target["node"])
        path = str(target["path"])
        if path not in {"translation", "rotation", "scale"}:
            continue
        key = (node_index, path)
        if key in result:
            raise ValueError(f"Duplicate animation channel for node={node_index}, path={path}")
        sampler = samplers[int(channel["sampler"])]
        if not isinstance(sampler, dict):
            raise ValueError("Invalid GLB animation sampler")
        times = np.asarray(document.accessor(int(sampler["input"])), dtype=np.float64)
        values = np.asarray(document.accessor(int(sampler["output"])), dtype=np.float64)
        if times.size == 0:
            raise ValueError("Animation sampler is empty")
        if times.size > 1:
            positive_intervals = np.diff(times)
            positive_intervals = positive_intervals[positive_intervals > 1e-9]
            if positive_intervals.size:
                sample_intervals.append(float(np.median(positive_intervals)))
        duration = max(duration, float(times[-1]))
        result[key] = (times, values, str(sampler.get("interpolation", "LINEAR")))
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("FBX animation duration must be positive and finite")
    if not sample_intervals:
        raise ValueError("FBX animation has no measurable sample interval")
    source_fps = 1.0 / float(np.median(np.asarray(sample_intervals)))
    if not math.isfinite(source_fps) or source_fps <= 0.0:
        raise ValueError("FBX animation source FPS must be positive and finite")
    return result, duration, source_fps


def _sample_scene(
    document: _GlbDocument,
    target_fps: float,
) -> tuple[_SampledScene, list[dict[str, object]], np.ndarray]:
    if not math.isfinite(target_fps) or target_fps <= 0.0:
        raise ValueError(f"target_fps must be positive and finite, got {target_fps}")
    raw_nodes = document.document.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes or not all(isinstance(node, dict) for node in raw_nodes):
        raise ValueError("GLB contains no valid nodes")
    nodes = list(raw_nodes)
    order, parents = _topological_nodes(
        nodes,
        document.document.get("scenes"),
        int(document.document.get("scene", 0)),
    )
    channels, duration, source_fps = _animation_channels(document)
    frame_count = math.floor(duration * target_fps + 1e-5) + 1
    target_times = np.arange(frame_count, dtype=np.float64) / target_fps
    defaults = [_node_defaults(node) for node in nodes]
    positions: list[np.ndarray | None] = [None] * len(nodes)
    rotations: list[np.ndarray | None] = [None] * len(nodes)
    scales: list[np.ndarray | None] = [None] * len(nodes)
    rest_positions = np.zeros((len(nodes), 3), dtype=np.float64)
    rest_rotations = np.zeros((len(nodes), 4), dtype=np.float64)
    rest_scales = np.ones((len(nodes), 3), dtype=np.float64)

    for node_index in order:
        default = defaults[node_index]
        local_translation = np.broadcast_to(default.translation, (frame_count, 3)).copy()
        local_rotation = np.broadcast_to(default.rotation_xyzw, (frame_count, 4)).copy()
        local_scale = np.broadcast_to(default.scale, (frame_count, 3)).copy()
        for path, fallback in (
            ("translation", local_translation),
            ("rotation", local_rotation),
            ("scale", local_scale),
        ):
            sampler = channels.get((node_index, path))
            if sampler is None:
                continue
            times, values, interpolation = sampler
            if path == "rotation":
                sampled = _sample_quaternions(times, values, target_times, interpolation)
            else:
                sampled = _sample_vectors(times, values, target_times, interpolation)
            fallback[:] = sampled

        parent = int(parents[node_index])
        if parent == -1:
            global_translation = local_translation
            global_rotation = local_rotation
            global_scale = local_scale
            rest_positions[node_index] = default.translation
            rest_rotations[node_index] = default.rotation_xyzw
            rest_scales[node_index] = default.scale
        else:
            parent_position = positions[parent]
            parent_rotation = rotations[parent]
            parent_scale = scales[parent]
            if parent_position is None or parent_rotation is None or parent_scale is None:
                raise RuntimeError("GLB nodes were not processed in parent-first order")
            global_translation = parent_position + _quat_rotate_xyzw(
                parent_rotation,
                parent_scale * local_translation,
            )
            global_rotation = _normalize_quaternions_xyzw(_quat_multiply_xyzw(parent_rotation, local_rotation))
            global_scale = parent_scale * local_scale
            rest_positions[node_index] = rest_positions[parent] + _quat_rotate_xyzw(
                rest_rotations[parent],
                rest_scales[parent] * default.translation,
            )
            rest_rotations[node_index] = _normalize_quaternions_xyzw(
                _quat_multiply_xyzw(rest_rotations[parent], default.rotation_xyzw)
            )
            rest_scales[node_index] = rest_scales[parent] * default.scale
        positions[node_index] = global_translation
        rotations[node_index] = global_rotation
        scales[node_index] = global_scale

    if any(value is None for value in positions + rotations + scales):
        raise RuntimeError("GLB scene evaluation left uninitialized nodes")
    return (
        _SampledScene(
            times=target_times,
            source_fps=source_fps,
            animated_rotation_nodes=frozenset(node_index for node_index, path in channels if path == "rotation"),
            positions=tuple(value for value in positions if value is not None),
            rotations_xyzw=tuple(value for value in rotations if value is not None),
            scales=tuple(value for value in scales if value is not None),
            rest_positions=rest_positions,
            rest_rotations_xyzw=rest_rotations,
        ),
        nodes,
        parents,
    )


def _split_actor_joint(name: str) -> tuple[str, str] | None:
    if ":" in name:
        actor, joint = name.rsplit(":", 1)
    elif "_" in name:
        actor, joint = name.split("_", 1)
    else:
        return None
    if not actor or not joint:
        return None
    return actor, joint


def _descendants(nodes: list[dict[str, object]], root: int) -> set[int]:
    found: set[int] = set()
    stack = [root]
    while stack:
        index = stack.pop()
        if index in found:
            continue
        found.add(index)
        stack.extend(int(child) for child in nodes[index].get("children", []))
    return found


def _discover_actors(nodes: list[dict[str, object]]) -> dict[str, dict[str, int]]:
    actors: dict[str, dict[str, int]] = {}
    for hips_index, node in enumerate(nodes):
        split = _split_actor_joint(str(node.get("name", "")))
        if split is None or split[1] != "Hips":
            continue
        actor, _ = split
        joint_indices: dict[str, int] = {}
        for node_index in _descendants(nodes, hips_index):
            candidate = _split_actor_joint(str(nodes[node_index].get("name", "")))
            if candidate is None or candidate[0] != actor:
                continue
            joint = candidate[1]
            if joint in joint_indices:
                raise ValueError(f"Actor {actor!r} contains duplicate joint {joint!r}")
            joint_indices[joint] = node_index
        missing = sorted(set(_SOURCE_JOINT_BY_CANONICAL.values()).difference(joint_indices))
        if missing:
            raise ValueError(f"Actor {actor!r} is missing canonical FBX joints: {missing}")
        actors[actor] = joint_indices
    if not actors:
        raise ValueError("FBX contains no supported Hips-rooted skeletons")
    return dict(sorted(actors.items()))


def _actor_joint_topology(
    nodes: list[dict[str, object]],
    parents: np.ndarray,
    actor: str,
    joint_indices: dict[str, int],
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    """Return every named actor joint in source hierarchy order.

    The canonical retargeting tensor deliberately stays stable across files,
    while this complete topology is preserved for inspection and rendering.
    A parent is resolved to the nearest named actor ancestor so harmless FBX
    transform nodes cannot break the displayed skeleton.
    """

    index_to_name = {node_index: name for name, node_index in joint_indices.items()}
    root_index = joint_indices["Hips"]
    ordered_indices: list[int] = []

    def visit(node_index: int) -> None:
        if node_index in index_to_name:
            ordered_indices.append(node_index)
        for child in nodes[node_index].get("children", []):
            visit(int(child))

    visit(root_index)
    if len(ordered_indices) != len(joint_indices) or set(ordered_indices) != set(joint_indices.values()):
        raise ValueError(f"Actor {actor!r} joint hierarchy does not form one Hips-rooted tree")

    local_index = {node_index: index for index, node_index in enumerate(ordered_indices)}
    parent_indices: list[int] = []
    for node_index in ordered_indices:
        parent = int(parents[node_index])
        while parent >= 0 and parent not in local_index:
            parent = int(parents[parent])
        parent_indices.append(-1 if parent < 0 else local_index[parent])
    return (
        tuple(index_to_name[node_index] for node_index in ordered_indices),
        np.asarray(ordered_indices, dtype=np.int32),
        np.asarray(parent_indices, dtype=np.int32),
    )


def _orientations_y_up_to_z_up_wxyz(quaternions_xyzw: np.ndarray) -> np.ndarray:
    shape = quaternions_xyzw.shape
    flat = _normalize_quaternions_xyzw(quaternions_xyzw).reshape(-1, 4)
    source_matrices = Rotation.from_quat(flat).as_matrix()
    target_matrices = np.einsum(
        "ab,nbc,cd->nad",
        _FBX_Y_UP_TO_RIGHT_HANDED_Z_UP,
        source_matrices,
        _FBX_Y_UP_TO_RIGHT_HANDED_Z_UP.T,
    )
    target_xyzw = Rotation.from_matrix(target_matrices).as_quat().reshape(shape)
    result = target_xyzw[..., [3, 0, 1, 2]]
    for frame in range(1, result.shape[0]):
        flip = np.sum(result[frame - 1] * result[frame], axis=-1) < 0.0
        result[frame, flip] *= -1.0
    return result / np.linalg.norm(result, axis=-1, keepdims=True)


def _rest_orientations_y_up_to_z_up_wxyz(quaternions_xyzw: np.ndarray) -> np.ndarray:
    transformed = _orientations_y_up_to_z_up_wxyz(quaternions_xyzw[None, ...])
    return transformed[0]


def _estimate_height_metres(scene: _SampledScene, joint_indices: dict[str, int]) -> float:
    head_index = joint_indices.get("HeadEnd", joint_indices["Head"])
    toe_indices = (joint_indices["LToeEnd"], joint_indices["RToeEnd"])
    rest_head = scene.rest_positions[head_index]
    rest_toe_y = min(scene.rest_positions[index, 1] for index in toe_indices)
    height = abs(float(rest_head[1] - rest_toe_y)) * _FBX_CENTIMETRES_TO_METRES
    if not math.isfinite(height) or not 1.0 <= height <= 2.5:
        dynamic_head = scene.positions[head_index][:, 1]
        dynamic_toes = np.minimum(
            scene.positions[toe_indices[0]][:, 1],
            scene.positions[toe_indices[1]][:, 1],
        )
        height = float(np.percentile(dynamic_head - dynamic_toes, 95.0)) * _FBX_CENTIMETRES_TO_METRES
    if not math.isfinite(height) or not 1.0 <= height <= 2.5:
        raise ValueError(f"Cannot estimate a plausible human height, got {height}")
    return height


def _validate_actor_bind_axes(scene: _SampledScene, joint_indices: dict[str, int], actor: str) -> None:
    rest = scene.rest_positions
    anatomical_left = rest[joint_indices["LeftUpLeg"]] - rest[joint_indices["RightUpLeg"]]
    anatomical_left[1] = 0.0
    left_norm = float(np.linalg.norm(anatomical_left))
    if left_norm <= 1e-8 or float(anatomical_left[0] / left_norm) < 0.95:
        raise ValueError(
            f"Actor {actor!r} bind pose does not establish anatomical left as raw FBX +X: {anatomical_left}",
        )
    anatomical_forward = 0.5 * (
        rest[joint_indices["LeftToeBase"]]
        - rest[joint_indices["LeftFoot"]]
        + rest[joint_indices["RightToeBase"]]
        - rest[joint_indices["RightFoot"]]
    )
    anatomical_forward[1] = 0.0
    forward_norm = float(np.linalg.norm(anatomical_forward))
    if forward_norm <= 1e-8 or float(anatomical_forward[2] / forward_norm) < 0.95:
        raise ValueError(
            f"Actor {actor!r} bind pose does not establish anatomical forward as raw FBX +Z: "
            f"{anatomical_forward}",
        )


def _read_fbx_property(
    mapping: mmap.mmap,
    offset: int,
    limit: int,
) -> tuple[object, int]:
    if offset >= limit:
        raise ValueError("FBX property list is truncated")
    property_type = chr(mapping[offset])
    offset += 1
    scalar_formats = {
        "Y": "<h",
        "C": "<?",
        "I": "<i",
        "F": "<f",
        "D": "<d",
        "L": "<q",
    }
    if property_type in scalar_formats:
        scalar_format = scalar_formats[property_type]
        size = struct.calcsize(scalar_format)
        if offset + size > limit:
            raise ValueError("FBX scalar property is truncated")
        return struct.unpack_from(scalar_format, mapping, offset)[0], offset + size
    if property_type in {"S", "R"}:
        if offset + 4 > limit:
            raise ValueError("FBX string property is truncated")
        size = int(struct.unpack_from("<I", mapping, offset)[0])
        offset += 4
        if offset + size > limit:
            raise ValueError("FBX string payload is truncated")
        payload = bytes(mapping[offset : offset + size])
        value: object = payload.decode("utf-8", errors="replace") if property_type == "S" else payload
        return value, offset + size
    if property_type in {"f", "d", "l", "i", "b", "c"}:
        if offset + 12 > limit:
            raise ValueError("FBX array property is truncated")
        _, _, compressed_size = struct.unpack_from("<III", mapping, offset)
        offset += 12
        if offset + compressed_size > limit:
            raise ValueError("FBX array payload is truncated")
        return None, offset + compressed_size
    raise ValueError(f"Unsupported FBX property type {property_type!r}")


def _iter_fbx_nodes(
    mapping: mmap.mmap,
    start: int,
    stop: int,
    *,
    wide_offsets: bool,
    parent_path: tuple[str, ...] = (),
) -> Iterator[tuple[tuple[str, ...], tuple[object, ...]]]:
    header_format = "<QQQB" if wide_offsets else "<IIIB"
    header_size = struct.calcsize(header_format)
    offset = start
    while offset + header_size <= stop:
        end_offset, property_count, property_list_size, name_size = struct.unpack_from(
            header_format,
            mapping,
            offset,
        )
        if end_offset == property_count == property_list_size == name_size == 0:
            return
        if end_offset <= offset or end_offset > len(mapping):
            raise ValueError("FBX node has an invalid end offset")
        name_start = offset + header_size
        name_stop = name_start + name_size
        property_start = name_stop
        property_stop = property_start + property_list_size
        if property_stop > end_offset:
            raise ValueError("FBX node property list exceeds its node")
        name = bytes(mapping[name_start:name_stop]).decode("utf-8", errors="replace")
        properties: list[object] = []
        property_offset = property_start
        for _ in range(property_count):
            value, property_offset = _read_fbx_property(mapping, property_offset, property_stop)
            properties.append(value)
        if property_offset != property_stop:
            raise ValueError("FBX node property list size does not match its header")
        path = (*parent_path, name)
        yield path, tuple(properties)
        child_stop = end_offset - header_size
        if property_stop < child_stop:
            yield from _iter_fbx_nodes(
                mapping,
                property_stop,
                child_stop,
                wide_offsets=wide_offsets,
                parent_path=path,
            )
        offset = end_offset


def _read_fbx_global_settings(path: Path) -> tuple[int, _FbxGlobalSettings]:
    with path.open("rb") as source, mmap.mmap(source.fileno(), length=0, access=mmap.ACCESS_READ) as mapping:
        if len(mapping) < 27 or not mapping[:23].startswith(b"Kaydara FBX Binary"):
            raise ValueError(f"Only binary FBX files are supported: {path}")
        version = int(struct.unpack_from("<I", mapping, 23)[0])
        values: dict[str, object] = {}
        selected = {
            "UpAxis",
            "UpAxisSign",
            "FrontAxis",
            "FrontAxisSign",
            "CoordAxis",
            "CoordAxisSign",
            "UnitScaleFactor",
        }
        for node_path, properties in _iter_fbx_nodes(
            mapping,
            27,
            len(mapping),
            wide_offsets=version >= 7500,
        ):
            if node_path[-3:] != ("GlobalSettings", "Properties70", "P") or not properties:
                continue
            key = str(properties[0])
            if key in selected:
                values[key] = properties[-1]
    missing = sorted(selected.difference(values))
    if missing:
        raise ValueError(f"FBX GlobalSettings is missing required properties: {missing}")
    settings = _FbxGlobalSettings(
        up_axis=int(values["UpAxis"]),
        up_axis_sign=int(values["UpAxisSign"]),
        front_axis=int(values["FrontAxis"]),
        front_axis_sign=int(values["FrontAxisSign"]),
        coord_axis=int(values["CoordAxis"]),
        coord_axis_sign=int(values["CoordAxisSign"]),
        unit_scale_factor=float(values["UnitScaleFactor"]),
    )
    expected = _FbxGlobalSettings(
        up_axis=1,
        up_axis_sign=1,
        front_axis=2,
        front_axis_sign=1,
        coord_axis=0,
        coord_axis_sign=1,
        unit_scale_factor=1.0,
    )
    if settings != expected:
        raise ValueError(
            "Unsupported raw FBX GlobalSettings; this converter requires the supplied "
            f"+Y-up centimetre convention {expected}, got {settings}",
        )
    return version, settings


def _atomic_savez(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".npz",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            np.savez_compressed(temporary, **payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.replace(path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _safe_actor_name(actor: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_-]+", "_", actor).strip("_")
    if not result:
        raise ValueError(f"Actor name cannot be represented in a filename: {actor!r}")
    return result


def convert_file(
    fbx_path: Path | str,
    output_dir: Path | str,
    *,
    target_fps: float = 30.0,
    assimp_executable: Path | str | None = None,
    overwrite: bool = False,
) -> tuple[ConvertedActor, ...]:
    """Split one FBX into actor-level canonical fbx_mocap NPZ files."""
    source_path = Path(fbx_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if not source_path.is_file() or source_path.suffix.lower() != ".fbx":
        raise FileNotFoundError(f"FBX source file not found: {source_path}")
    assimp = _resolve_assimp_executable(assimp_executable)
    version, global_settings = _read_fbx_global_settings(source_path)
    with tempfile.TemporaryDirectory(prefix="holosoma-fbx-") as temporary_directory:
        glb_path = Path(temporary_directory) / "scene.glb"
        _export_fbx_to_glb(source_path, glb_path, assimp)
        with _GlbDocument(glb_path) as document:
            sampled, nodes, parents = _sample_scene(document, float(target_fps))
            actors = _discover_actors(nodes)

    converted: list[ConvertedActor] = []
    for actor, source_indices in actors.items():
        _validate_actor_bind_axes(sampled, source_indices, actor)
        (
            source_skeleton_joint_names,
            source_skeleton_indices,
            source_skeleton_parent_indices,
        ) = _actor_joint_topology(nodes, parents, actor, source_indices)
        canonical_indices = np.asarray(
            [source_indices[_SOURCE_JOINT_BY_CANONICAL[name]] for name in FBX_MOCAP_DEMO_JOINTS],
            dtype=np.int32,
        )
        orientation_indices = np.asarray(
            [source_indices[_SOURCE_JOINT_BY_CANONICAL[name]] for name in _ORIENTATION_JOINT_NAMES],
            dtype=np.int32,
        )
        missing_rotation_channels = [
            name
            for name, node_index in zip(_ORIENTATION_JOINT_NAMES, orientation_indices, strict=True)
            if int(node_index) not in sampled.animated_rotation_nodes
        ]
        if missing_rotation_channels:
            raise ValueError(
                f"Actor {actor!r} is missing direct FBX rotation curves for canonical joints: "
                f"{missing_rotation_channels}"
            )
        positions_y_up_cm = np.stack([sampled.positions[index] for index in canonical_indices], axis=1)
        positions_z_up_m = (
            positions_y_up_cm @ _FBX_Y_UP_TO_RIGHT_HANDED_Z_UP.T
        ) * _FBX_CENTIMETRES_TO_METRES
        root_xy_origin = positions_z_up_m[0, 0, :2].copy()
        positions_z_up_m[..., :2] -= root_xy_origin
        orientations_y_up_xyzw = np.stack(
            [sampled.rotations_xyzw[index] for index in orientation_indices],
            axis=1,
        )
        orientations_z_up_wxyz = _orientations_y_up_to_z_up_wxyz(orientations_y_up_xyzw)
        t_pose_y_up_xyzw = sampled.rest_rotations_xyzw[orientation_indices]
        t_pose_z_up_wxyz = _rest_orientations_y_up_to_z_up_wxyz(t_pose_y_up_xyzw)
        source_skeleton_positions_y_up_cm = np.stack(
            [sampled.positions[index] for index in source_skeleton_indices],
            axis=1,
        )
        source_skeleton_positions_z_up_m = (
            source_skeleton_positions_y_up_cm @ _FBX_Y_UP_TO_RIGHT_HANDED_Z_UP.T
        ) * _FBX_CENTIMETRES_TO_METRES
        source_skeleton_positions_z_up_m[..., :2] -= root_xy_origin
        source_skeleton_orientations_y_up_xyzw = np.stack(
            [sampled.rotations_xyzw[index] for index in source_skeleton_indices],
            axis=1,
        )
        source_skeleton_orientations_z_up_wxyz = _orientations_y_up_to_z_up_wxyz(
            source_skeleton_orientations_y_up_xyzw,
        )
        source_skeleton_bind_orientations_z_up_wxyz = _rest_orientations_y_up_to_z_up_wxyz(
            sampled.rest_rotations_xyzw[source_skeleton_indices],
        )
        height = _estimate_height_metres(sampled, source_indices)
        actor_filename = _safe_actor_name(actor)
        output_path = destination / f"{source_path.stem}__{actor_filename}.npz"
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"Converted actor motion already exists: {output_path}")
        payload: dict[str, object] = {
            "global_joint_positions": positions_z_up_m.astype(np.float32),
            "joint_names": np.asarray(FBX_MOCAP_DEMO_JOINTS, dtype=str),
            "joint_parents": np.asarray(DEMO_JOINT_PARENT_INDICES["fbx_mocap"], dtype=np.int32),
            "orientation_joint_names": np.asarray(_ORIENTATION_JOINT_NAMES, dtype=str),
            "orientation_quaternions_wxyz": orientations_z_up_wxyz.astype(np.float32),
            "orientation_source": np.asarray(_DIRECT_ORIENTATION_SOURCE),
            "orientation_provenance": np.asarray("direct_fbx_local_rotation_curves_fk"),
            "orientation_coordinate_system": np.asarray("z_up"),
            "t_pose_orientation_joint_names": np.asarray(_ORIENTATION_JOINT_NAMES, dtype=str),
            "t_pose_orientation_quaternions_wxyz": t_pose_z_up_wxyz.astype(np.float32),
            "source_skeleton_joint_names": np.asarray(source_skeleton_joint_names, dtype=str),
            "source_skeleton_parent_indices": source_skeleton_parent_indices,
            "source_skeleton_positions": source_skeleton_positions_z_up_m.astype(np.float32),
            "source_skeleton_quaternions_wxyz": source_skeleton_orientations_z_up_wxyz.astype(np.float32),
            "source_skeleton_bind_quaternions_wxyz": (
                source_skeleton_bind_orientations_z_up_wxyz.astype(np.float32)
            ),
            "height": np.float32(height),
            "fps": np.float32(target_fps),
            "source_fps": np.float32(sampled.source_fps),
            "source_format": np.asarray("fbx_mocap"),
            "source_container_format": np.asarray("fbx"),
            "source_fbx": np.asarray(str(source_path)),
            "source_fbx_version": np.int32(version),
            "source_actor": np.asarray(actor),
            "raw_joint_names": np.asarray(source_skeleton_joint_names, dtype=str),
            "coordinate_system": np.asarray("z_up"),
            "source_coordinate_system": np.asarray("fbx_global_settings_y_up_centimetres"),
            "source_fbx_up_axis": np.int8(global_settings.up_axis),
            "source_fbx_up_axis_sign": np.int8(global_settings.up_axis_sign),
            "source_fbx_front_axis": np.int8(global_settings.front_axis),
            "source_fbx_front_axis_sign": np.int8(global_settings.front_axis_sign),
            "source_fbx_coord_axis": np.int8(global_settings.coord_axis),
            "source_fbx_coord_axis_sign": np.int8(global_settings.coord_axis_sign),
            "source_fbx_unit_scale_factor": np.float32(global_settings.unit_scale_factor),
            "source_anatomical_left_axis": np.asarray("positive_x_from_named_bind_joints"),
            "source_anatomical_forward_axis": np.asarray("positive_z_from_named_bind_toes"),
            "position_coordinate_transform": np.asarray(
                "metres_x_negative_z_y_and_initial_hips_xy_recenter",
            ),
            "orientation_coordinate_transform": np.asarray(
                "basis_conjugation_rx_plus_90_right_handed",
            ),
            "root_frame_to_robot_base_quaternion_wxyz": np.asarray(
                [2**-0.5, 0.0, 0.0, -(2**-0.5)],
                dtype=np.float32,
            ),
            "quaternion_convention": np.asarray("wxyz"),
            "source_xy_origin_m": root_xy_origin.astype(np.float32),
        }
        if not np.isfinite(positions_z_up_m).all() or not np.isfinite(orientations_z_up_wxyz).all():
            raise ValueError(f"Converted actor {actor!r} contains non-finite values")
        _atomic_savez(output_path, payload)
        converted.append(
            ConvertedActor(
                actor=actor,
                output_path=output_path,
                frame_count=int(positions_z_up_m.shape[0]),
                fps=float(target_fps),
                height=height,
            )
        )
    return tuple(converted)


def convert_directory(
    input_dir: Path | str,
    output_dir: Path | str,
    *,
    target_fps: float = 30.0,
    assimp_executable: Path | str | None = None,
    overwrite: bool = False,
) -> tuple[ConvertedActor, ...]:
    """Convert every top-level FBX file and return all actor artifacts."""
    source_directory = Path(input_dir).expanduser().resolve()
    if not source_directory.is_dir():
        raise FileNotFoundError(f"FBX input directory not found: {source_directory}")
    sources = sorted(path for path in source_directory.iterdir() if path.is_file() and path.suffix.lower() == ".fbx")
    if not sources:
        raise FileNotFoundError(f"No FBX files found in {source_directory}")
    converted: list[ConvertedActor] = []
    for source in sources:
        converted.extend(
            convert_file(
                source,
                output_dir,
                target_fps=target_fps,
                assimp_executable=assimp_executable,
                overwrite=overwrite,
            )
        )
    return tuple(converted)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--assimp-executable", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    converted = convert_directory(
        args.input_dir,
        args.output_dir,
        target_fps=args.target_fps,
        assimp_executable=args.assimp_executable,
        overwrite=args.overwrite,
    )
    for actor in converted:
        print(
            f"{actor.actor}: frames={actor.frame_count}, fps={actor.fps:.3f}, "
            f"height={actor.height:.3f} -> {actor.output_path}"
        )


if __name__ == "__main__":
    main()
