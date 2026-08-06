#!/usr/bin/env python3
# ruff: noqa: CPY001
"""Recursively convert a directory of retargeted NPZ files to CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

from holosoma_retargeting.data_utils.npz_to_csv import convert_npz_to_csv


def convert_npz_directory(
    input_dir: str | Path,
    urdf_path: str | Path,
    output_dir: str | Path,
) -> tuple[Path, ...]:
    """Convert every NPZ while preserving its relative path."""

    source_root = Path(input_dir).expanduser().resolve()
    destination_root = Path(output_dir).expanduser().resolve()
    sources = tuple(sorted(path for path in source_root.rglob("*.npz") if path.is_file()))
    if not sources:
        raise FileNotFoundError(f"No NPZ files found under {source_root}")

    outputs: list[Path] = []
    for source in sources:
        destination = destination_root / source.relative_to(source_root).with_suffix(".csv")
        outputs.append(convert_npz_to_csv(source, urdf_path, destination))
    return tuple(outputs)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="Directory containing NPZ files")
    parser.add_argument("urdf_path", type=Path, help="Robot URDF shared by the inputs")
    parser.add_argument("output_dir", type=Path, help="Destination CSV directory")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    outputs = convert_npz_directory(args.input_dir, args.urdf_path, args.output_dir)
    print(f"Converted {len(outputs)} NPZ files to: {args.output_dir.expanduser().resolve()}")


if __name__ == "__main__":
    main()
