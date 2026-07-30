#!/usr/bin/env python3
# ruff: noqa: CPY001
"""Convert LAFAN BVH motions into the canonical retargeting NPZ schema."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from scipy.spatial.transform import Rotation

src_root = Path(__file__).resolve().parents[2]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from holosoma_retargeting.config_types.data_type import LAFAN_DEMO_JOINTS  # noqa: E402
from holosoma_retargeting.data_utils.lafan1 import extract, utils  # noqa: E402

LAFAN_CANONICAL_TO_SOURCE = {
    name: ("LeftToe" if name == "LeftToeBase" else "RightToe" if name == "RightToeBase" else name)
    for name in LAFAN_DEMO_JOINTS
}

Y_UP_TO_Z_UP_BASIS = np.asarray(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


def _read_source_fps(path: Path) -> float:
    for line in path.read_text(errors="replace").splitlines():
        match = re.match(r"\s*Frame Time:\s+([\d.]+)", line)
        if match is not None:
            frame_time = float(match.group(1))
            if frame_time <= 0.0:
                break
            return 1.0 / frame_time
    raise ValueError(f"Positive BVH Frame Time not found in {path}")


def _enforce_quaternion_continuity_wxyz(
    quaternions_wxyz: np.ndarray,
) -> np.ndarray:
    result = np.asarray(quaternions_wxyz, dtype=np.float64).copy()
    norms = np.linalg.norm(result, axis=-1)
    if not np.isfinite(result).all() or np.any(norms <= 1e-8):
        raise ValueError("LAFAN global quaternions must be finite and non-zero")
    result /= norms[..., None]
    for frame_index in range(1, result.shape[0]):
        flip = np.sum(result[frame_index - 1] * result[frame_index], axis=-1) < 0.0
        result[frame_index, flip] *= -1.0
    return result


def transform_orientations_y_up_to_z_up(
    quaternions_wxyz: np.ndarray,
) -> np.ndarray:
    """Express BVH global rotations in the legacy LAFAN Z-up basis."""
    quaternions_wxyz = np.asarray(quaternions_wxyz, dtype=np.float64)
    if quaternions_wxyz.ndim != 3 or quaternions_wxyz.shape[-1] != 4:
        raise ValueError(f"LAFAN global quaternions must have shape (T, J, 4), got {quaternions_wxyz.shape}")
    flat = _enforce_quaternion_continuity_wxyz(quaternions_wxyz).reshape(-1, 4)
    matrices_y_up = Rotation.from_quat(flat[:, [1, 2, 3, 0]]).as_matrix()
    matrices_z_up = np.einsum(
        "ab,nbc,cd->nad",
        Y_UP_TO_Z_UP_BASIS,
        matrices_y_up,
        Y_UP_TO_Z_UP_BASIS.T,
    )
    xyzw = Rotation.from_matrix(matrices_z_up).as_quat()
    wxyz = xyzw[:, [3, 0, 1, 2]].reshape(quaternions_wxyz.shape)
    return _enforce_quaternion_continuity_wxyz(wxyz)


def _canonical_parent_indices(
    source_joint_names: list[str],
    source_parents: np.ndarray,
) -> np.ndarray:
    source_to_canonical = {
        source_name: canonical_name for canonical_name, source_name in LAFAN_CANONICAL_TO_SOURCE.items()
    }
    canonical_index = {name: index for index, name in enumerate(LAFAN_DEMO_JOINTS)}
    source_index = {name: index for index, name in enumerate(source_joint_names)}
    parents: list[int] = []
    for canonical_name in LAFAN_DEMO_JOINTS:
        source_name = LAFAN_CANONICAL_TO_SOURCE[canonical_name]
        parent_index = int(source_parents[source_index[source_name]])
        while parent_index >= 0 and source_joint_names[parent_index] not in source_to_canonical:
            parent_index = int(source_parents[parent_index])
        parents.append(
            -1 if parent_index < 0 else canonical_index[source_to_canonical[source_joint_names[parent_index]]]
        )
    return np.asarray(parents, dtype=np.int32)


def extract_global_positions(bvh_file_path: str | Path) -> dict[str, object]:
    """Extract canonical positions and direct global rotations through BVH FK."""
    source_path = Path(bvh_file_path)
    anim = extract.read_bvh(str(source_path))
    source_index = {name: index for index, name in enumerate(anim.bones)}
    missing = set(LAFAN_CANONICAL_TO_SOURCE.values()).difference(source_index)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"LAFAN BVH is missing canonical source joints: {names}")

    global_quaternions, global_positions_cm = utils.quat_fk(
        anim.quats,
        anim.pos,
        anim.parents,
    )
    indices = [source_index[LAFAN_CANONICAL_TO_SOURCE[name]] for name in LAFAN_DEMO_JOINTS]
    positions_y_up_m = global_positions_cm[:, indices] / 100.0
    positions_z_up_m = np.einsum(
        "ab,tjb->tja",
        Y_UP_TO_Z_UP_BASIS,
        positions_y_up_m,
    )
    orientations_z_up_wxyz = transform_orientations_y_up_to_z_up(global_quaternions[:, indices])
    return {
        "positions": positions_z_up_m.astype(np.float32),
        "orientations_wxyz": orientations_z_up_wxyz.astype(np.float32),
        "joint_names": tuple(LAFAN_DEMO_JOINTS),
        "parents": _canonical_parent_indices(anim.bones, anim.parents),
        "fps": _read_source_fps(source_path),
        "raw_joint_names": tuple(anim.bones),
        "num_frames": positions_z_up_m.shape[0],
        "num_joints": positions_z_up_m.shape[1],
    }


@dataclass
class Config:
    """Configuration for extracting global positions from BVH files."""

    input_dir: str = "./lafan1/lafan"
    output_dir: str = "../demo_data/lafan"

    overwrite: bool = False
    """Replace converted .npz files that already exist."""


def convert_file(
    bvh_path: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Convert one LAFAN BVH without estimating any missing orientation."""
    result = extract_global_positions(bvh_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_npz = output_dir / f"{bvh_path.stem}.npz"
    if output_npz.exists() and not overwrite:
        print(f"Skipping existing output: {output_npz}")
        return output_npz
    np.savez_compressed(
        output_npz,
        global_joint_positions=result["positions"],
        joint_names=np.asarray(result["joint_names"], dtype=str),
        joint_parents=result["parents"],
        orientation_joint_names=np.asarray(
            result["joint_names"],
            dtype=str,
        ),
        orientation_quaternions_wxyz=result["orientations_wxyz"],
        height=np.float32(1.7),
        fps=np.float32(result["fps"]),
        source_fps=np.float32(result["fps"]),
        raw_joint_names=np.asarray(result["raw_joint_names"], dtype=str),
        source_bvh=np.asarray(str(bvh_path)),
        source_format=np.asarray("lafan"),
        coordinate_system=np.asarray("z_up"),
        quaternion_convention=np.asarray("wxyz"),
        orientation_provenance=np.asarray("direct_source"),
        orientation_source=np.asarray("bvh_rotation_channels_fk"),
        orientation_coordinate_transform=np.asarray("basis_conjugation_xzy_reflection"),
    )
    print(f"Saved converted LAFAN motion to: {output_npz}")
    return output_npz


def main(cfg: Config):
    """
    Main function to extract global positions from BVH files.
    """
    input_dir = Path(cfg.input_dir)
    output_dir = Path(cfg.output_dir)

    # Check if input directory exists
    if not input_dir.exists():
        raise FileNotFoundError(f"LAFAN input directory not found: {input_dir}")

    bvh_files = sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".bvh")
    if not bvh_files:
        raise FileNotFoundError(f"No BVH files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each BVH file
    for bvh_path in bvh_files:
        print(f"\nProcessing: {bvh_path.name}")
        output_npz = convert_file(
            bvh_path,
            output_dir,
            overwrite=cfg.overwrite,
        )
        with np.load(output_npz, allow_pickle=False) as data:
            print(f"  Frames: {data['global_joint_positions'].shape[0]}")
            print(f"  Joints: {data['global_joint_positions'].shape[1]}")


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
