#!/usr/bin/env python3
# ruff: noqa: CPY001

"""Convert the mixed Noetix BVH samples into a canonical retargeting format."""

from __future__ import annotations

import os
import re
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from scipy.spatial.transform import Rotation

src_root = Path(__file__).resolve().parents[2]
data_utils_root = Path(__file__).resolve().parent
for path in (src_root, data_utils_root):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from holosoma_retargeting.config_types.data_type import NOETIX_MOCAP_DEMO_JOINTS  # noqa: E402
from lafan1 import extract, utils  # type: ignore[import-not-found] # noqa: E402

NOETIX_FULLBODY_CHEST_MAPPING = {
    "Hips": "Hips",
    # Normalize the company BVH exports into the Noetix 22-joint tensor order.
    # The source export's left/right labels are inverted relative to that order.
    "RightUpLeg": "LeftUpLeg",
    "RightLeg": "LeftLeg",
    "RightFoot": "LeftFoot",
    "RightToeBase": "LeftToe",
    "LeftUpLeg": "RightUpLeg",
    "LeftLeg": "RightLeg",
    "LeftFoot": "RightFoot",
    "LeftToeBase": "RightToe",
    "Spine": "Spine1",
    "Spine1": "Spine2",
    "Spine2": "Chest",
    "Neck": "Neck",
    "Head": "Head",
    "RightShoulder": "LeftShoulder",
    "RightArm": "LeftArm",
    "RightForeArm": "LeftForeArm",
    "RightHand": "LeftHand",
    "LeftShoulder": "RightShoulder",
    "LeftArm": "RightArm",
    "LeftForeArm": "RightForeArm",
    "LeftHand": "RightHand",
}

NOETIX_FULLBODY_SPINE_MAPPING = {
    **NOETIX_FULLBODY_CHEST_MAPPING,
    "Spine": "Spine",
    "Spine1": "Spine1",
    "Spine2": "Spine2",
}

NOETIX_22_JOINT_MAPPING = NOETIX_FULLBODY_SPINE_MAPPING.copy()

# Some Noetix exports use the same full-body layout but omit the third spine
# joint and call the toe joints ``*ToeBase``. ``Spine2`` is synthesized halfway
# between source ``Spine1`` and ``Neck`` after selecting the joints.
NOETIX_FULLBODY_REDUCED_SPINE_MAPPING = {
    **NOETIX_FULLBODY_CHEST_MAPPING,
    "RightToeBase": "LeftToeBase",
    "LeftToeBase": "RightToeBase",
    "Spine": "Spine",
    "Spine1": "Spine1",
    "Spine2": "Neck",
}

NOETIX_FULLBODY_REDUCED_SPINE_JOINTS = {
    "Hips",
    "Spine",
    "Spine1",
    "Neck",
    "Head",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToeBase",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToeBase",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
}

# Newer Noetix exports can collapse the torso to a single ``Spine1`` joint.
# Keep that source joint as the canonical ``Spine1`` retargeting anchor, then
# synthesize the two non-anchor spine landmarks from the adjacent chain.
NOETIX_FULLBODY_SINGLE_SPINE_MAPPING = {
    **NOETIX_FULLBODY_REDUCED_SPINE_MAPPING,
    "Spine": "Hips",
    "Spine1": "Spine1",
    "Spine2": "Neck",
}

NOETIX_FULLBODY_SINGLE_SPINE_JOINTS = NOETIX_FULLBODY_REDUCED_SPINE_JOINTS - {"Spine"}

SYNTHETIC_ORIENTATION_JOINTS_BY_SOURCE_TYPE = {
    "fullbody_reduced_spine_zyx": frozenset({"Spine2"}),
    "fullbody_single_spine_zyx": frozenset({"Spine", "Spine2"}),
}

NOETIX_CANONICAL_PARENT_NAMES = {
    "Hips": None,
    "RightUpLeg": "Hips",
    "RightLeg": "RightUpLeg",
    "RightFoot": "RightLeg",
    "RightToeBase": "RightFoot",
    "LeftUpLeg": "Hips",
    "LeftLeg": "LeftUpLeg",
    "LeftFoot": "LeftLeg",
    "LeftToeBase": "LeftFoot",
    "Spine": "Hips",
    "Spine1": "Spine",
    "Spine2": "Spine1",
    "Neck": "Spine2",
    "Head": "Neck",
    "RightShoulder": "Spine2",
    "RightArm": "RightShoulder",
    "RightForeArm": "RightArm",
    "RightHand": "RightForeArm",
    "LeftShoulder": "Spine2",
    "LeftArm": "LeftShoulder",
    "LeftForeArm": "LeftArm",
    "LeftHand": "LeftForeArm",
}


def canonical_parent_indices() -> np.ndarray:
    canonical_index = {name: index for index, name in enumerate(NOETIX_MOCAP_DEMO_JOINTS)}
    return np.asarray(
        [
            -1 if NOETIX_CANONICAL_PARENT_NAMES[name] is None else canonical_index[NOETIX_CANONICAL_PARENT_NAMES[name]]
            for name in NOETIX_MOCAP_DEMO_JOINTS
        ],
        dtype=np.int32,
    )


NOETIX_RUN23_MAPPING = {
    "Hips": "Hips",
    "RightUpLeg": "LeftHip",
    "RightLeg": "LeftKnee",
    "RightFoot": "LeftAnkle",
    "RightToeBase": "LeftToe",
    "LeftUpLeg": "RightHip",
    "LeftLeg": "RightKnee",
    "LeftFoot": "RightAnkle",
    "LeftToeBase": "RightToe",
    "Spine": "Chest",
    "Spine1": "Chest2",
    "Spine2": "Chest4",
    "Neck": "Neck",
    "Head": "Head",
    "RightShoulder": "LeftCollar",
    "RightArm": "LeftShoulder",
    "RightForeArm": "LeftElbow",
    "RightHand": "LeftWrist",
    "LeftShoulder": "RightCollar",
    "LeftArm": "RightShoulder",
    "LeftForeArm": "RightElbow",
    "LeftHand": "RightWrist",
}

Y_UP_TO_Z_UP_BASIS = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


def classify_bvh(joint_names: list[str]) -> tuple[str, dict[str, str]]:
    names = set(joint_names)

    if {"LeftHip", "RightHip", "Chest4", "LeftWrist", "RightWrist"} <= names:
        return "run23_yxz", NOETIX_RUN23_MAPPING

    if names >= NOETIX_FULLBODY_SINGLE_SPINE_JOINTS and "Spine" not in names:
        return "fullbody_single_spine_zyx", NOETIX_FULLBODY_SINGLE_SPINE_MAPPING

    if names >= NOETIX_FULLBODY_REDUCED_SPINE_JOINTS and "Spine2" not in names:
        return "fullbody_reduced_spine_zyx", NOETIX_FULLBODY_REDUCED_SPINE_MAPPING

    if "FZLeftThumb1" in names or "LeftHandPalm" in names:
        if "Spine" in names:
            return "fullbody57_spine_zyx", NOETIX_FULLBODY_SPINE_MAPPING
        return "fullbody57_chest_zyx", NOETIX_FULLBODY_CHEST_MAPPING

    if {"LeftUpLeg", "RightUpLeg", "Spine", "Spine1", "Spine2", "LeftToe", "RightToe"} <= names:
        return "noetix22_zyx", NOETIX_22_JOINT_MAPPING

    raise ValueError(f"Unsupported BVH skeleton. Joints: {joint_names}")


def select_canonical_joints(
    values: np.ndarray,
    source_joint_names: list[str],
    mapping: dict[str, str],
) -> np.ndarray:
    source_idx = {name: idx for idx, name in enumerate(source_joint_names)}
    missing = [
        source_name
        for canonical_name in NOETIX_MOCAP_DEMO_JOINTS
        for source_name in [mapping[canonical_name]]
        if source_name not in source_idx
    ]
    if missing:
        raise ValueError(f"Missing source joints for canonical conversion: {missing}")

    indices = [source_idx[mapping[name]] for name in NOETIX_MOCAP_DEMO_JOINTS]
    return values[:, indices, :]


def select_direct_canonical_orientations(
    values: np.ndarray,
    source_joint_names: list[str],
    mapping: dict[str, str],
    source_type: str,
) -> tuple[tuple[str, ...], np.ndarray]:
    """Select only canonical frames backed by a real source BVH bone."""
    omitted = SYNTHETIC_ORIENTATION_JOINTS_BY_SOURCE_TYPE.get(
        source_type,
        frozenset(),
    )
    orientation_names = tuple(name for name in NOETIX_MOCAP_DEMO_JOINTS if name not in omitted)
    source_idx = {name: idx for idx, name in enumerate(source_joint_names)}
    missing = [mapping[name] for name in orientation_names if mapping[name] not in source_idx]
    if missing:
        raise ValueError(f"Missing source joints for direct orientations: {missing}")
    indices = [source_idx[mapping[name]] for name in orientation_names]
    return orientation_names, values[:, indices, :]


def synthesize_reduced_spine2(canonical_positions_y_up: np.ndarray) -> np.ndarray:
    """Insert a stable third spine landmark for Noetix's two-spine layout."""
    canonical_positions_y_up = canonical_positions_y_up.copy()
    joint_idx = {name: idx for idx, name in enumerate(NOETIX_MOCAP_DEMO_JOINTS)}
    canonical_positions_y_up[:, joint_idx["Spine2"]] = 0.5 * (
        canonical_positions_y_up[:, joint_idx["Spine1"]] + canonical_positions_y_up[:, joint_idx["Neck"]]
    )
    return canonical_positions_y_up


def synthesize_single_spine_chain(
    canonical_positions_y_up: np.ndarray,
) -> np.ndarray:
    """Fill canonical Spine and Spine2 around a retained source Spine1."""
    result = canonical_positions_y_up.copy()
    joint_idx = {name: idx for idx, name in enumerate(NOETIX_MOCAP_DEMO_JOINTS)}
    hips = result[:, joint_idx["Hips"]]
    spine1 = result[:, joint_idx["Spine1"]]
    neck = result[:, joint_idx["Neck"]]
    result[:, joint_idx["Spine"]] = hips + (spine1 - hips) / 3.0
    result[:, joint_idx["Spine2"]] = 0.5 * (spine1 + neck)
    return result


def transform_y_up_to_z_up(points: np.ndarray) -> np.ndarray:
    return points[..., [0, 2, 1]]


def transform_orientations_y_up_to_z_up(
    quaternions_wxyz: np.ndarray,
) -> np.ndarray:
    """Change quaternion coordinates using the same reflected basis as positions."""
    quaternions_wxyz = np.asarray(quaternions_wxyz, dtype=np.float64)
    if quaternions_wxyz.ndim != 3 or quaternions_wxyz.shape[-1] != 4:
        raise ValueError(f"global joint quaternions must have shape (T, J, 4), got {quaternions_wxyz.shape}")
    flat_wxyz = quaternions_wxyz.reshape(-1, 4)
    norms = np.linalg.norm(flat_wxyz, axis=1)
    if not np.isfinite(flat_wxyz).all() or np.any(norms <= 1e-8):
        raise ValueError("global joint quaternions must be finite and non-zero")
    flat_wxyz = flat_wxyz / norms[:, None]
    matrices_y_up = Rotation.from_quat(flat_wxyz[:, [1, 2, 3, 0]]).as_matrix()
    matrices_z_up = np.einsum(
        "ab,nbc,cd->nad",
        Y_UP_TO_Z_UP_BASIS,
        matrices_y_up,
        Y_UP_TO_Z_UP_BASIS.T,
    )
    flat_xyzw = Rotation.from_matrix(matrices_z_up).as_quat()
    transformed = flat_xyzw[:, [3, 0, 1, 2]].reshape(quaternions_wxyz.shape)
    return enforce_quaternion_continuity_wxyz(transformed)


def enforce_quaternion_continuity_wxyz(quaternions_wxyz: np.ndarray) -> np.ndarray:
    """Normalize quaternions and choose temporally continuous signs per joint."""
    result = np.asarray(quaternions_wxyz, dtype=np.float64).copy()
    norms = np.linalg.norm(result, axis=-1)
    if not np.isfinite(result).all() or np.any(norms <= 1e-8):
        raise ValueError("global joint quaternions must be finite and non-zero")
    result /= norms[..., None]
    for frame_idx in range(1, result.shape[0]):
        flip = np.sum(result[frame_idx - 1] * result[frame_idx], axis=-1) < 0.0
        result[frame_idx, flip] *= -1.0
    return result


def normalize_xy(vector: np.ndarray) -> np.ndarray | None:
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector[:2])
    if norm <= 1e-6:
        return None
    return vector[:2] / norm


def infer_noetix_forward(positions_z_up: np.ndarray, frame_window: int = 30) -> np.ndarray:
    joint_idx = {name: idx for idx, name in enumerate(NOETIX_MOCAP_DEMO_JOINTS)}
    frame_count = min(frame_window, positions_z_up.shape[0])
    directions = []
    for frame_idx in range(frame_count):
        for foot_name, toe_name in (
            ("RightFoot", "RightToeBase"),
            ("LeftFoot", "LeftToeBase"),
        ):
            foot_to_toe = (
                positions_z_up[frame_idx, joint_idx[toe_name], :2] - positions_z_up[frame_idx, joint_idx[foot_name], :2]
            )
            direction = normalize_xy(foot_to_toe)
            if direction is not None:
                directions.append(direction)

    if directions:
        forward = normalize_xy(np.mean(directions, axis=0))
        if forward is not None:
            return forward

    hips_to_spine = positions_z_up[0, joint_idx["Hips"], :2] - positions_z_up[0, joint_idx["Spine"], :2]
    forward = normalize_xy(hips_to_spine)
    if forward is not None:
        return forward
    return np.array([0.0, 1.0])


def apply_noetix_root_orientation_hint(
    positions_z_up: np.ndarray,
    spine_horizontal_offset_m: float = 0.03,
) -> np.ndarray:
    """Stabilize the Noetix root-to-spine horizontal facing direction."""
    positions_z_up = positions_z_up.copy()
    joint_idx = {name: idx for idx, name in enumerate(NOETIX_MOCAP_DEMO_JOINTS)}
    forward = infer_noetix_forward(positions_z_up)
    hips_idx = joint_idx["Hips"]
    spine_idx = joint_idx["Spine"]
    positions_z_up[:, spine_idx, :2] = positions_z_up[:, hips_idx, :2] - spine_horizontal_offset_m * forward
    return positions_z_up


def source_height_from_filename(path: Path) -> float | None:
    # Height tokens occur as both ``_160_`` and ``_{168}_``.  Accept any
    # non-digit delimiter while rejecting dates and sequence ordinals by range.
    for match in re.finditer(r"(?<!\d)(\d{3})(?!\d)", path.stem):
        height_cm = int(match.group(1))
        if 120 <= height_cm <= 230:
            return height_cm / 100.0
    return None


def read_bvh_with_normalized_motion_rows(bvh_path: Path):
    """Read a BVH after normalizing its motion-row whitespace.

    The upstream LAFAN reader splits motion rows on a literal single space.
    Noetix BVHs commonly contain repeated spaces (and a trailing blank row),
    which otherwise produces empty numeric tokens.  Normalize only rows after
    ``Frame Time`` so the hierarchy and channel ordering remain untouched.
    """
    temp_path: Path | None = None
    try:
        with NamedTemporaryFile(mode="w", suffix=".bvh", encoding="utf-8", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            motion_rows = False
            for line in bvh_path.read_text(errors="replace").splitlines():
                if motion_rows:
                    fields = line.split()
                    if fields:
                        temp_file.write(" ".join(fields) + "\n")
                    continue

                temp_file.write(line + "\n")
                if line.lstrip().startswith("Frame Time:"):
                    motion_rows = True

        return extract.read_bvh(str(temp_path))
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def read_source_fps(path: Path) -> float:
    for line in path.read_text(errors="replace").splitlines():
        match = re.match(r"\s*Frame Time:\s+([\d.]+)", line)
        if match is not None:
            return 1.0 / float(match.group(1))
    raise ValueError(f"Frame Time not found in {path}")


def estimate_height(positions_z_up: np.ndarray) -> float:
    joint_idx = {name: idx for idx, name in enumerate(NOETIX_MOCAP_DEMO_JOINTS)}
    head_z = positions_z_up[:, joint_idx["Head"], 2]
    left_toe_z = positions_z_up[:, joint_idx["LeftToeBase"], 2]
    right_toe_z = positions_z_up[:, joint_idx["RightToeBase"], 2]
    toe_z = np.minimum(left_toe_z, right_toe_z)
    head_to_toe = np.percentile(head_z - toe_z, 95)
    return float(max(head_to_toe + 0.12, 1.2))


def drop_initial_jump(positions_z_up: np.ndarray, threshold_m: float) -> tuple[np.ndarray, bool]:
    if positions_z_up.shape[0] < 2:
        return positions_z_up, False

    root_xy_delta = np.linalg.norm(positions_z_up[1, 0, :2] - positions_z_up[0, 0, :2])
    if root_xy_delta <= threshold_m:
        return positions_z_up, False
    return positions_z_up[1:], True


def recenter_xy(positions_z_up: np.ndarray) -> np.ndarray:
    positions_z_up = positions_z_up.copy()
    root_origin_xy = positions_z_up[0, 0, :2].copy()
    positions_z_up[:, :, :2] -= root_origin_xy
    return positions_z_up


def downsample(positions: np.ndarray, source_fps: float, target_fps: float) -> tuple[np.ndarray, int, float]:
    stride = max(1, round(source_fps / target_fps))
    return positions[::stride], stride, source_fps / stride


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _atomic_savez_compressed(
    output_path: Path,
    payload: dict[str, object],
) -> None:
    """Publish a complete NPZ without exposing a partially written ZIP."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            np.savez_compressed(temporary_file, **payload)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, output_path)  # noqa: PTH105
        _fsync_directory(output_path.parent)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def convert_file(
    bvh_path: Path,
    output_dir: Path,
    target_fps: float,
    drop_jump_threshold_m: float,
    overwrite: bool = False,
) -> None:
    anim = read_bvh_with_normalized_motion_rows(bvh_path)
    global_quaternions_wxyz, global_positions_cm = utils.quat_fk(
        anim.quats,
        anim.pos,
        anim.parents,
    )
    source_type, mapping = classify_bvh(anim.bones)

    canonical_y_up_m = select_canonical_joints(global_positions_cm / 100.0, anim.bones, mapping)
    orientation_joint_names, direct_y_up_quaternions_wxyz = select_direct_canonical_orientations(
        global_quaternions_wxyz,
        anim.bones,
        mapping,
        source_type,
    )
    if source_type == "fullbody_single_spine_zyx":
        canonical_y_up_m = synthesize_single_spine_chain(canonical_y_up_m)
    elif source_type == "fullbody_reduced_spine_zyx":
        canonical_y_up_m = synthesize_reduced_spine2(canonical_y_up_m)
    canonical_z_up_m = transform_y_up_to_z_up(canonical_y_up_m)
    direct_z_up_quaternions_wxyz = transform_orientations_y_up_to_z_up(direct_y_up_quaternions_wxyz)
    canonical_z_up_m, dropped_initial_frame = drop_initial_jump(canonical_z_up_m, drop_jump_threshold_m)
    if dropped_initial_frame:
        direct_z_up_quaternions_wxyz = direct_z_up_quaternions_wxyz[1:]
    canonical_z_up_m = recenter_xy(canonical_z_up_m)
    canonical_z_up_m = apply_noetix_root_orientation_hint(canonical_z_up_m)

    source_fps = read_source_fps(bvh_path)
    canonical_z_up_m, stride, output_fps = downsample(canonical_z_up_m, source_fps, target_fps)
    direct_z_up_quaternions_wxyz = direct_z_up_quaternions_wxyz[::stride]
    if direct_z_up_quaternions_wxyz.shape[0] != canonical_z_up_m.shape[0]:
        raise RuntimeError("Position and orientation frame counts diverged during Noetix conversion")

    height = source_height_from_filename(bvh_path) or estimate_height(canonical_z_up_m)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{bvh_path.stem}.npz"
    if output_path.exists() and not overwrite:
        print(f"Skipping existing output: {output_path}")
        return
    _atomic_savez_compressed(
        output_path,
        {
            "global_joint_positions": canonical_z_up_m.astype(np.float32),
            "orientation_joint_names": np.asarray(
                orientation_joint_names,
                dtype=str,
            ),
            "orientation_quaternions_wxyz": direct_z_up_quaternions_wxyz.astype(np.float32),
            "height": np.float32(height),
            "joint_names": np.asarray(NOETIX_MOCAP_DEMO_JOINTS, dtype=str),
            "joint_parents": canonical_parent_indices(),
            "raw_joint_names": np.asarray(anim.bones, dtype=str),
            "source_bvh": str(bvh_path),
            "source_format": np.asarray("noetix_mocap"),
            "source_type": source_type,
            "coordinate_system": np.asarray("z_up"),
            "quaternion_convention": np.asarray("wxyz"),
            "orientation_provenance": np.asarray("direct_source_bones"),
            "orientation_source": np.asarray("bvh_rotation_channels_fk"),
            "orientation_coordinate_transform": np.asarray("basis_conjugation_xzy_reflection"),
            "position_root_orientation_hint_applied": np.bool_(True),
            "source_fps": np.float32(source_fps),
            "fps": np.float32(output_fps),
            "downsample_stride": np.int32(stride),
            "dropped_initial_frame": np.bool_(dropped_initial_frame),
        },
    )

    print(
        f"{bvh_path.name}: {source_type}, frames={canonical_z_up_m.shape[0]}, "
        f"fps={output_fps:.2f}, height={height:.3f} -> {output_path}"
    )


@dataclass
class Config:
    input_dir: Path = Path("demo_data/noetix")
    output_dir: Path = Path("demo_data/noetix_mocap")
    target_fps: float = 30.0
    drop_jump_threshold_m: float = 2.0
    overwrite: bool = False


def main(cfg: Config) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    bvh_files = sorted(cfg.input_dir.rglob("*.bvh"))
    if not bvh_files:
        raise FileNotFoundError(f"No BVH files found in {cfg.input_dir}")

    for bvh_path in bvh_files:
        relative_parent = bvh_path.parent.relative_to(cfg.input_dir)
        convert_file(
            bvh_path,
            cfg.output_dir / relative_parent,
            cfg.target_fps,
            cfg.drop_jump_threshold_m,
            overwrite=cfg.overwrite,
        )


def parse_args() -> Config:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Config.input_dir)
    parser.add_argument("--output-dir", type=Path, default=Config.output_dir)
    parser.add_argument("--target-fps", type=float, default=Config.target_fps)
    parser.add_argument("--drop-jump-threshold-m", type=float, default=Config.drop_jump_threshold_m)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    return Config(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_fps=args.target_fps,
        drop_jump_threshold_m=args.drop_jump_threshold_m,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main(parse_args())
