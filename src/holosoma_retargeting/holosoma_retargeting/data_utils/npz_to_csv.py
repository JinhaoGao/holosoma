#!/usr/bin/env python3
# ruff: noqa: CPY001, S314
"""Convert one retargeted robot motion NPZ to a numeric CSV."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def actuated_joint_names_from_urdf(urdf_path: str | Path) -> tuple[str, ...]:
    """Return scalar actuated joints in URDF file order."""

    names: list[str] = []
    for joint in ET.parse(urdf_path).getroot().findall("joint"):
        joint_type = joint.attrib.get("type", "")
        if joint_type == "fixed":
            continue
        if joint_type not in {"revolute", "continuous", "prismatic"}:
            raise ValueError(
                f"URDF joint {joint.attrib.get('name', '<unnamed>')!r} has unsupported type {joint_type!r}"
            )
        name = joint.attrib.get("name")
        if not name:
            raise ValueError("Every actuated URDF joint must have a name")
        names.append(name)
    return tuple(names)


def _require_unique(names: tuple[str, ...], description: str) -> None:
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"{description} contains duplicate joint names: {duplicates}")


def convert_npz_to_csv(
    npz_path: str | Path,
    urdf_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Save root xyz, root xyzw, and URDF-ordered joints without a header."""

    source = Path(npz_path).expanduser().resolve()
    urdf = Path(urdf_path).expanduser().resolve()
    destination = source.with_suffix(".csv") if output_path is None else Path(output_path).expanduser().resolve()

    with np.load(source, allow_pickle=False) as data:
        if "qpos" not in data or "robot_actuated_joint_names" not in data:
            raise ValueError(f"{source} must contain qpos and robot_actuated_joint_names")
        qpos = np.asarray(data["qpos"], dtype=np.float64)
        saved_joint_names = tuple(str(name) for name in np.asarray(data["robot_actuated_joint_names"]).tolist())

    if qpos.ndim != 2 or qpos.shape[0] == 0 or not np.isfinite(qpos).all():
        raise ValueError(f"{source} qpos must be a non-empty finite 2-D array")
    _require_unique(saved_joint_names, "NPZ robot_actuated_joint_names")
    urdf_joint_names = actuated_joint_names_from_urdf(urdf)
    _require_unique(urdf_joint_names, "URDF")
    if set(saved_joint_names) != set(urdf_joint_names):
        missing_from_npz = tuple(name for name in urdf_joint_names if name not in saved_joint_names)
        missing_from_urdf = tuple(name for name in saved_joint_names if name not in urdf_joint_names)
        raise ValueError(
            "NPZ and URDF actuated joints do not match: "
            f"missing_from_npz={missing_from_npz}, missing_from_urdf={missing_from_urdf}"
        )

    robot_qpos_width = 7 + len(saved_joint_names)
    trailing_width = qpos.shape[1] - robot_qpos_width
    if trailing_width not in {0, 7}:
        raise ValueError(
            f"{source} qpos width {qpos.shape[1]} is incompatible with "
            f"7 root values and {len(saved_joint_names)} robot joints"
        )
    saved_index = {name: index for index, name in enumerate(saved_joint_names)}
    urdf_order = np.asarray(
        [saved_index[name] for name in urdf_joint_names],
        dtype=np.int64,
    )
    robot_joints = qpos[:, 7:robot_qpos_width][:, urdf_order]
    root_xyz_xyzw = qpos[:, (0, 1, 2, 4, 5, 6, 3)]
    csv_values = np.concatenate((root_xyz_xyzw, robot_joints), axis=1)

    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(destination, csv_values, delimiter=",", fmt="%.18g")
    return destination


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz_path", type=Path, help="Retargeted motion NPZ")
    parser.add_argument("urdf_path", type=Path, help="Robot URDF")
    parser.add_argument("--output-path", "-o", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output = convert_npz_to_csv(args.npz_path, args.urdf_path, args.output_path)
    print(f"Saved CSV to: {output}")


if __name__ == "__main__":
    main()
