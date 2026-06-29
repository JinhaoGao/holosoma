#!/usr/bin/env python3
"""Convert the mixed Noetix BVH samples into a canonical retargeting format."""

from __future__ import annotations

import re
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import numpy as np

src_root = Path(__file__).resolve().parents[2]
data_utils_root = Path(__file__).resolve().parent
for path in (src_root, data_utils_root):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from holosoma_retargeting.config_types.data_type import NOETIX_LAFAN_DEMO_JOINTS  # noqa: E402
from lafan1 import extract, utils  # type: ignore[import-not-found] # noqa: E402


NOETIX_FULLBODY_CHEST_MAPPING = {
    "Hips": "Hips",
    # Match the existing LAFAN retargeting tensor convention. LAFAN BVH files
    # store left-side joints before right-side joints, while LAFAN_DEMO_JOINTS
    # names the right side first. This looks inverted, but it is the convention
    # used by the known-good LAFAN retargeting path.
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

NOETIX_LAFAN22_MAPPING = NOETIX_FULLBODY_SPINE_MAPPING.copy()

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


def classify_bvh(joint_names: list[str]) -> tuple[str, dict[str, str]]:
    names = set(joint_names)

    if {"LeftHip", "RightHip", "Chest4", "LeftWrist", "RightWrist"} <= names:
        return "run23_yxz", NOETIX_RUN23_MAPPING

    if "FZLeftThumb1" in names or "LeftHandPalm" in names:
        if "Spine" in names:
            return "fullbody57_spine_zyx", NOETIX_FULLBODY_SPINE_MAPPING
        return "fullbody57_chest_zyx", NOETIX_FULLBODY_CHEST_MAPPING

    if {"LeftUpLeg", "RightUpLeg", "Spine", "Spine1", "Spine2", "LeftToe", "RightToe"} <= names:
        return "lafan22_zyx", NOETIX_LAFAN22_MAPPING

    raise ValueError(f"Unsupported BVH skeleton. Joints: {joint_names}")


def select_canonical_joints(
    positions_y_up: np.ndarray,
    source_joint_names: list[str],
    mapping: dict[str, str],
) -> np.ndarray:
    source_idx = {name: idx for idx, name in enumerate(source_joint_names)}
    missing = [
        source_name
        for canonical_name in NOETIX_LAFAN_DEMO_JOINTS
        for source_name in [mapping[canonical_name]]
        if source_name not in source_idx
    ]
    if missing:
        raise ValueError(f"Missing source joints for canonical conversion: {missing}")

    indices = [source_idx[mapping[name]] for name in NOETIX_LAFAN_DEMO_JOINTS]
    return positions_y_up[:, indices, :]


def transform_y_up_to_z_up(points: np.ndarray) -> np.ndarray:
    return points[..., [0, 2, 1]]


def normalize_xy(vector: np.ndarray) -> np.ndarray | None:
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector[:2])
    if norm <= 1e-6:
        return None
    return vector[:2] / norm


def infer_lafan_forward(positions_z_up: np.ndarray, frame_window: int = 30) -> np.ndarray:
    joint_idx = {name: idx for idx, name in enumerate(NOETIX_LAFAN_DEMO_JOINTS)}
    frame_count = min(frame_window, positions_z_up.shape[0])
    directions = []
    for frame_idx in range(frame_count):
        for foot_name, toe_name in (
            ("RightFoot", "RightToeBase"),
            ("LeftFoot", "LeftToeBase"),
        ):
            foot_to_toe = (
                positions_z_up[frame_idx, joint_idx[toe_name], :2]
                - positions_z_up[frame_idx, joint_idx[foot_name], :2]
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


def apply_lafan_root_orientation_hint(
    positions_z_up: np.ndarray,
    spine_horizontal_offset_m: float = 0.03,
) -> np.ndarray:
    """Make the Hips->Spine horizontal component match LAFAN's facing convention."""
    positions_z_up = positions_z_up.copy()
    joint_idx = {name: idx for idx, name in enumerate(NOETIX_LAFAN_DEMO_JOINTS)}
    forward = infer_lafan_forward(positions_z_up)
    hips_idx = joint_idx["Hips"]
    spine_idx = joint_idx["Spine"]
    positions_z_up[:, spine_idx, :2] = positions_z_up[:, hips_idx, :2] - spine_horizontal_offset_m * forward
    return positions_z_up


def source_height_from_filename(path: Path) -> float | None:
    match = re.search(r"(?:^|_)(\d{3})__", path.name)
    if match is None:
        return None
    return float(match.group(1)) / 100.0


def read_source_fps(path: Path) -> float:
    for line in path.read_text(errors="replace").splitlines():
        match = re.match(r"\s*Frame Time:\s+([\d.]+)", line)
        if match is not None:
            return 1.0 / float(match.group(1))
    raise ValueError(f"Frame Time not found in {path}")


def estimate_height(positions_z_up: np.ndarray) -> float:
    joint_idx = {name: idx for idx, name in enumerate(NOETIX_LAFAN_DEMO_JOINTS)}
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
    stride = max(1, int(round(source_fps / target_fps)))
    return positions[::stride], stride, source_fps / stride


def convert_file(
    bvh_path: Path,
    output_dir: Path,
    target_fps: float,
    drop_jump_threshold_m: float,
) -> None:
    anim = extract.read_bvh(str(bvh_path))
    _, global_positions_cm = utils.quat_fk(anim.quats, anim.pos, anim.parents)
    source_type, mapping = classify_bvh(anim.bones)

    canonical_y_up_m = select_canonical_joints(global_positions_cm / 100.0, anim.bones, mapping)
    canonical_z_up_m = transform_y_up_to_z_up(canonical_y_up_m)
    canonical_z_up_m, dropped_initial_frame = drop_initial_jump(canonical_z_up_m, drop_jump_threshold_m)
    canonical_z_up_m = recenter_xy(canonical_z_up_m)
    canonical_z_up_m = apply_lafan_root_orientation_hint(canonical_z_up_m)

    source_fps = read_source_fps(bvh_path)
    canonical_z_up_m, stride, output_fps = downsample(canonical_z_up_m, source_fps, target_fps)

    height = source_height_from_filename(bvh_path) or estimate_height(canonical_z_up_m)
    output_path = output_dir / f"{bvh_path.stem}.npz"
    np.savez_compressed(
        output_path,
        global_joint_positions=canonical_z_up_m.astype(np.float32),
        height=np.float32(height),
        joint_names=np.asarray(NOETIX_LAFAN_DEMO_JOINTS, dtype=str),
        raw_joint_names=np.asarray(anim.bones, dtype=str),
        source_bvh=str(bvh_path),
        source_type=source_type,
        source_fps=np.float32(source_fps),
        fps=np.float32(output_fps),
        downsample_stride=np.int32(stride),
        dropped_initial_frame=np.bool_(dropped_initial_frame),
    )

    print(
        f"{bvh_path.name}: {source_type}, frames={canonical_z_up_m.shape[0]}, "
        f"fps={output_fps:.2f}, height={height:.3f} -> {output_path}"
    )


@dataclass
class Config:
    input_dir: Path = Path("demo_data/noetix")
    output_dir: Path = Path("demo_data/noetix_lafan")
    target_fps: float = 30.0
    drop_jump_threshold_m: float = 2.0


def main(cfg: Config) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    bvh_files = sorted(cfg.input_dir.glob("*.bvh"))
    if not bvh_files:
        raise FileNotFoundError(f"No BVH files found in {cfg.input_dir}")

    for bvh_path in bvh_files:
        convert_file(bvh_path, cfg.output_dir, cfg.target_fps, cfg.drop_jump_threshold_m)


def parse_args() -> Config:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Config.input_dir)
    parser.add_argument("--output-dir", type=Path, default=Config.output_dir)
    parser.add_argument("--target-fps", type=float, default=Config.target_fps)
    parser.add_argument("--drop-jump-threshold-m", type=float, default=Config.drop_jump_threshold_m)
    args = parser.parse_args()
    return Config(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_fps=args.target_fps,
        drop_jump_threshold_m=args.drop_jump_threshold_m,
    )


if __name__ == "__main__":
    main(parse_args())
