# ruff: noqa: CPY001

"""Convert centimetre Y-up Hips-rooted BVH to the canonical FBX mocap layout."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from holosoma_retargeting.config_types.data_type import DEMO_JOINT_PARENT_INDICES, FBX_MOCAP_DEMO_JOINTS
from holosoma_retargeting.data_utils.convert_fbx import (
    _FBX_Y_UP_TO_RIGHT_HANDED_Z_UP,
    _ORIENTATION_JOINT_NAMES,
    _atomic_savez,
    _orientations_y_up_to_z_up_wxyz,
)


def read_bvh(path: Path) -> tuple[list[str], np.ndarray, np.ndarray, list[list[str]], np.ndarray, float]:
    """Parse explicit per-joint BVH channel orders, including end sites."""
    hierarchy, motion = path.read_text(encoding="utf-8-sig").split("MOTION", 1)
    tokens = iter(hierarchy.replace("{", " { ").replace("}", " } ").split())
    names: list[str] = []
    parents: list[int] = []
    offsets: list[list[float]] = []
    channels: list[list[str]] = []

    def joint(parent: int, name: str) -> None:
        index = len(names)
        names.append(name)
        parents.append(parent)
        offsets.append([0.0, 0.0, 0.0])
        channels.append([])
        if next(tokens) != "{":
            raise ValueError("Expected BVH joint opening brace")
        for word in tokens:
            if word == "}":
                return
            if word == "OFFSET":
                offsets[index] = [float(next(tokens)) for _ in range(3)]
            elif word == "CHANNELS":
                channels[index] = [next(tokens) for _ in range(int(next(tokens)))]
            elif word == "JOINT":
                joint(index, next(tokens))
            elif word == "End":
                if next(tokens) != "Site":
                    raise ValueError("Expected BVH End Site")
                joint(index, name + "_End")
            else:
                raise ValueError(f"Unexpected BVH token: {word}")
        raise ValueError("Unterminated BVH joint")

    if next(tokens) != "HIERARCHY" or next(tokens) != "ROOT":
        raise ValueError("Expected BVH hierarchy and root")
    joint(-1, next(tokens))
    lines = motion.strip().splitlines()
    count = int(lines[0].split(":")[1])
    timestep = float(lines[1].split(":")[1])
    values = np.fromstring(" ".join(lines[2:]), sep=" ").reshape(count, -1)
    if values.shape[1] != sum(map(len, channels)) or not np.isfinite(values).all():
        raise ValueError("BVH motion does not match its channels")
    if count < 2 or not np.isfinite(timestep) or timestep <= 0:
        raise ValueError("BVH requires at least two frames and a positive frame time")
    return names, np.asarray(parents), np.asarray(offsets), channels, values, timestep


def convert_file(source: Path, output: Path, *, target_fps: float = 30.0) -> Path:
    """Retain source positions and direct Euler-channel FK orientations."""
    names, parents, offsets, channels, values, timestep = read_bvh(source)
    if not np.isfinite(target_fps) or target_fps <= 0:
        raise ValueError("target_fps must be positive and finite")
    name_index = {name: index for index, name in enumerate(names)}
    if len(name_index) != len(names) or names[0] != "Hips":
        raise ValueError("BVH requires unique joint names and a Hips root")
    required = {name: name for name in FBX_MOCAP_DEMO_JOINTS}
    required.update(LeftFootMod="LeftToeBase_End", RightFootMod="RightToeBase_End")
    missing = set(required.values()).difference(names)
    if missing:
        raise ValueError(f"BVH is missing canonical joints: {sorted(missing)}")
    frames = len(values)
    positions = np.empty((frames, len(names), 3))
    matrices = np.empty((frames, len(names), 3, 3))
    rest = offsets.copy()
    cursor = 0
    for index, joint_channels in enumerate(channels):
        translation = np.broadcast_to(offsets[index], (frames, 3)).copy()
        rotation_channels = []
        angles = []
        for channel in joint_channels:
            axis = "XYZ".index(channel[0])
            if channel.endswith("position"):
                translation[:, axis] += values[:, cursor]
            elif channel.endswith("rotation"):
                rotation_channels.append(channel[0])
                angles.append(values[:, cursor])
            else:
                raise ValueError(f"Unsupported BVH channel {channel}")
            cursor += 1
        local = (
            Rotation.from_euler("".join(rotation_channels), np.stack(angles, axis=-1), degrees=True).as_matrix()
            if angles
            else np.broadcast_to(np.eye(3), (frames, 3, 3))
        )
        parent = parents[index]
        if parent < 0:
            positions[:, index] = translation
            matrices[:, index] = local
        else:
            positions[:, index] = positions[:, parent] + np.einsum("fij,fj->fi", matrices[:, parent], translation)
            matrices[:, index] = matrices[:, parent] @ local
            rest[index] += rest[parent]
    left = rest[name_index["LeftUpLeg"]] - rest[name_index["RightUpLeg"]]
    up = rest[name_index["Head"]] - rest[0]
    if left[0] <= 0 or up[1] <= abs(up[2]) or not 100 < np.ptp(rest[:, 1]) < 250:
        raise ValueError("This converter requires centimetre Y-up BVH with anatomical left along +X")
    times = np.arange(frames) * timestep
    sample_times = np.arange(int(np.floor(times[-1] * target_fps + 1e-5)) + 1) / target_fps
    sample_times = np.minimum(sample_times, times[-1])
    sampled_positions = np.empty((len(sample_times), len(names), 3))
    sampled_quaternions = np.empty((len(sample_times), len(names), 4))
    for index in range(len(names)):
        for axis in range(3):
            sampled_positions[:, index, axis] = np.interp(sample_times, times, positions[:, index, axis])
        sampled_quaternions[:, index] = Slerp(times, Rotation.from_matrix(matrices[:, index]))(sample_times).as_quat()
    sampled_positions = sampled_positions @ _FBX_Y_UP_TO_RIGHT_HANDED_Z_UP.T / 100
    origin = sampled_positions[0, 0, :2].copy()
    sampled_positions[..., :2] -= origin
    quaternions = _orientations_y_up_to_z_up_wxyz(sampled_quaternions)
    canonical = [name_index[required[name]] for name in FBX_MOCAP_DEMO_JOINTS]
    orientation = [name_index[name] for name in _ORIENTATION_JOINT_NAMES]
    height = (
        rest[name_index["Head_End"], 1]
        - min(rest[name_index[name], 1] for name in ("LeftToeBase_End", "RightToeBase_End"))
    ) / 100
    payload = {
        "global_joint_positions": sampled_positions[:, canonical].astype(np.float32),
        "joint_names": np.asarray(FBX_MOCAP_DEMO_JOINTS),
        "joint_parents": np.asarray(DEMO_JOINT_PARENT_INDICES["fbx_mocap"]),
        "orientation_joint_names": np.asarray(_ORIENTATION_JOINT_NAMES),
        "orientation_quaternions_wxyz": quaternions[:, orientation].astype(np.float32),
        "orientation_source": np.asarray("bvh_rotation_channels_fk"),
        "orientation_provenance": np.asarray("direct_bvh_euler_channels_fk"),
        "orientation_coordinate_system": np.asarray("z_up"),
        "quaternion_convention": np.asarray("wxyz"),
        "t_pose_orientation_joint_names": np.asarray(_ORIENTATION_JOINT_NAMES),
        "t_pose_orientation_quaternions_wxyz": np.tile([1.0, 0.0, 0.0, 0.0], (len(orientation), 1)).astype(np.float32),
        "source_skeleton_joint_names": np.asarray(names),
        "source_skeleton_parent_indices": parents,
        "source_skeleton_positions": sampled_positions.astype(np.float32),
        "source_skeleton_quaternions_wxyz": quaternions.astype(np.float32),
        "source_skeleton_bind_quaternions_wxyz": np.tile([1.0, 0.0, 0.0, 0.0], (len(names), 1)).astype(np.float32),
        "height": np.float32(height),
        "fps": np.float32(target_fps),
        "source_fps": np.float32(1 / timestep),
        "source_format": np.asarray("fbx_mocap"),
        "source_container_format": np.asarray("bvh"),
        "source_coordinate_system": np.asarray("bvh_y_up_centimetres"),
        "orientation_coordinate_transform": np.asarray("basis_conjugation_rx_plus_90_right_handed"),
        "source_anatomical_left_axis": np.asarray("positive_x_from_named_bind_joints"),
        "source_anatomical_forward_axis": np.asarray("positive_z_from_named_bind_toes"),
        "source_bvh": np.asarray(str(source.resolve())),
        "source_actor": np.asarray("Skeleton"),
        "source_xy_origin_m": origin.astype(np.float32),
        "position_coordinate_transform": np.asarray("metres_x_negative_z_y_and_initial_hips_xy_recenter"),
        "coordinate_system": np.asarray("z_up"),
        "root_frame_to_robot_base_quaternion_wxyz": np.asarray([2**-0.5, 0, 0, -(2**-0.5)], dtype=np.float32),
    }
    _atomic_savez(output, payload)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--target-fps", type=float, default=30.0)
    args = parser.parse_args()
    print(convert_file(args.source, args.output, target_fps=args.target_fps))
