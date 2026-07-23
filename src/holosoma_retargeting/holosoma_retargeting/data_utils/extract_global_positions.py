#!/usr/bin/env python3
"""
Simple script to extract global positions from LAFAN dataset BVH files.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from lafan1 import extract, utils  # type: ignore[import-not-found]


def extract_global_positions(bvh_file_path):
    """
    Extract global positions from a BVH file.

    Args:
        bvh_file_path (str): Path to the BVH file

    Returns:
        dict: Dictionary containing:
            - 'positions': numpy array of shape (frames, joints, 3) with global positions
            - 'joint_names': list of joint names
            - 'parents': list of parent indices
            - 'num_frames': number of frames
            - 'num_joints': number of joints
    """
    # Read BVH file
    anim = extract.read_bvh(bvh_file_path)

    # Compute global positions using Forward Kinematics
    _, global_positions = utils.quat_fk(anim.quats, anim.pos, anim.parents)
    return {
        "positions": (global_positions / 100).astype(np.float32),
        "joint_names": anim.bones,
        "parents": anim.parents,
        "num_frames": global_positions.shape[0],
        "num_joints": global_positions.shape[1],
    }


@dataclass
class Config:
    """Configuration for extracting global positions from BVH files."""

    input_dir: str = "./lafan1/lafan"
    output_dir: str = "../demo_data/lafan"

    overwrite: bool = False
    """Replace converted .npy files that already exist."""


def main(cfg: Config):
    """
    Main function to extract global positions from BVH files.
    """
    input_dir = Path(cfg.input_dir)
    output_dir = Path(cfg.output_dir)

    # Check if input directory exists
    if not input_dir.exists():
        raise FileNotFoundError(f"LAFAN input directory not found: {input_dir}")

    # Get list of BVH files
    bvh_files = [f.name for f in input_dir.iterdir() if f.is_file() and f.suffix == ".bvh"]
    if not bvh_files:
        raise FileNotFoundError(f"No BVH files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each BVH file
    for bvh_file in bvh_files:  # Process first 3 files to avoid memory issues
        print(f"\nProcessing: {bvh_file}")

        bvh_path = input_dir / bvh_file

        # Extract global positions
        result = extract_global_positions(str(bvh_path))

        print(f"  Frames: {result['num_frames']}")
        print(f"  Joints: {result['num_joints']}")
        print(f"  Joint names: {result['joint_names']}")

        output_npy = output_dir / f"{bvh_file[:-4]}.npy"
        if output_npy.exists() and not cfg.overwrite:
            print(f"Skipping existing output: {output_npy}")
            continue
        np.save(output_npy, result["positions"])
        print(f"Saved converted LAFAN motion to: {output_npy}")


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
