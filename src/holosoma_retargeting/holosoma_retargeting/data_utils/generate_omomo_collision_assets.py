"""Generate reproducible convex collision decompositions for OMOMO objects.

This is an asset-maintenance utility, not a runtime dependency. Install
``coacd==1.0.11`` in the retargeting environment before running it.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from holosoma_retargeting.data_utils.object_assets import (
    OMOMO_MESH_SHA256,
    default_models_root,
)
from holosoma_retargeting.data_utils.omomo import OMOMO_OBJECT_NAMES
from trimesh.exchange.obj import export_obj

COACD_VERSION = "1.0.11"
COACD_PARAMETERS: dict[str, object] = {
    "threshold": 0.02,
    "max_convex_hull": 64,
    "preprocess_mode": "auto",
    "preprocess_resolution": 60,
    "resolution": 3000,
    "mcts_nodes": 20,
    "mcts_iterations": 150,
    "mcts_max_depth": 3,
    "pca": False,
    "merge": True,
    "decimate": False,
    "max_ch_vertex": 64,
    "extrude": False,
    "extrude_margin": 0.01,
    "apx_mode": "ch",
    "seed": 0,
    "real_metric": True,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_object_urdf(object_name: str, collision_paths: list[Path], destination: Path) -> None:
    robot = ET.Element("robot", {"name": object_name})
    ET.SubElement(robot, "dynamics", {"damping": "0.5", "friction": "0.9"})
    link = ET.SubElement(robot, "link", {"name": f"{object_name}_link"})

    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "mass", {"value": "0.1"})
    ET.SubElement(inertial, "origin", {"xyz": "0 0 0"})
    ET.SubElement(
        inertial,
        "inertia",
        {
            "ixx": "0.002",
            "ixy": "0",
            "ixz": "0",
            "iyy": "0.002",
            "iyz": "0",
            "izz": "0.002",
        },
    )

    contact = ET.SubElement(link, "contact")
    ET.SubElement(contact, "lateral_friction", {"value": "0.9"})
    ET.SubElement(contact, "rolling_friction", {"value": "0.5"})
    ET.SubElement(contact, "stiffness", {"value": "30000"})
    ET.SubElement(contact, "damping", {"value": "1000"})

    visual = ET.SubElement(link, "visual")
    ET.SubElement(visual, "origin", {"rpy": "0 0 0", "xyz": "0 0 0"})
    visual_geometry = ET.SubElement(visual, "geometry")
    ET.SubElement(
        visual_geometry,
        "mesh",
        {"filename": f"{object_name}.obj", "scale": "1 1 1"},
    )
    material = ET.SubElement(visual, "material", {"name": "mat"})
    ET.SubElement(material, "color", {"rgba": "0.7 0.8 0.9 0.7"})

    for index, collision_path in enumerate(collision_paths):
        collision = ET.SubElement(
            link,
            "collision",
            {"name": f"{object_name}_collision_{index:03d}"},
        )
        ET.SubElement(collision, "origin", {"rpy": "0 0 0", "xyz": "0 0 0"})
        geometry = ET.SubElement(collision, "geometry")
        ET.SubElement(
            geometry,
            "mesh",
            {
                "filename": collision_path.relative_to(destination.parent).as_posix(),
                "scale": "1 1 1",
            },
        )

    ET.indent(robot, space="  ")
    ET.ElementTree(robot).write(destination, encoding="utf-8", xml_declaration=True)


def _generate_object(object_name: str, models_root: Path, coacd_module) -> dict[str, object]:
    object_dir = models_root / object_name
    source_path = object_dir / f"{object_name}.obj"
    if _sha256(source_path) != OMOMO_MESH_SHA256[object_name]:
        raise ValueError(f"Source mesh checksum mismatch: {source_path}")

    source_mesh = trimesh.load(source_path, force="mesh", process=False)
    coacd_mesh = coacd_module.Mesh(
        np.asarray(source_mesh.vertices, dtype=np.float64),
        np.asarray(source_mesh.faces, dtype=np.int32),
    )
    decomposition = coacd_module.run_coacd(coacd_mesh, **COACD_PARAMETERS)
    if not decomposition:
        raise RuntimeError(f"CoACD produced no collision parts for {object_name}")

    collision_dir = object_dir / "collision"
    collision_dir.mkdir(parents=True, exist_ok=True)
    collision_paths: list[Path] = []
    part_records: list[dict[str, object]] = []
    collision_volume = 0.0
    for index, (vertices, faces) in enumerate(decomposition):
        part = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        if not part.is_watertight or not part.is_convex:
            raise ValueError(f"CoACD returned a non-convex part: {object_name}[{index}]")
        destination = collision_dir / f"{object_name}_collision_{index:03d}.obj"
        destination.write_text(
            export_obj(
                part,
                include_normals=False,
                include_color=False,
                include_texture=False,
                digits=8,
                header=None,
            ),
            encoding="utf-8",
        )
        collision_paths.append(destination)
        volume = float(abs(part.volume))
        collision_volume += volume
        part_records.append(
            {
                "file": destination.relative_to(models_root).as_posix(),
                "sha256": _sha256(destination),
                "vertices": int(len(part.vertices)),
                "faces": int(len(part.faces)),
                "volume_m3": volume,
            }
        )

    expected = set(collision_paths)
    for stale_path in collision_dir.glob(f"{object_name}_collision_*.obj"):
        if stale_path not in expected:
            stale_path.unlink()

    _write_object_urdf(object_name, collision_paths, object_dir / f"{object_name}.urdf")
    global_hull_volume = float(abs(source_mesh.convex_hull.volume))
    return {
        "source_mesh": source_path.relative_to(models_root).as_posix(),
        "source_sha256": OMOMO_MESH_SHA256[object_name],
        "collision_part_count": len(part_records),
        "collision_volume_sum_m3": collision_volume,
        "global_convex_hull_volume_m3": global_hull_volume,
        "collision_to_global_hull_volume_ratio": collision_volume / global_hull_volume,
        "parts": part_records,
    }


def generate_collision_assets(models_root: Path, object_names: tuple[str, ...]) -> Path:
    """Generate selected assets and update the shared provenance manifest."""

    try:
        import coacd  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            f"Generating OMOMO collision assets requires coacd=={COACD_VERSION}"
        ) from exc

    installed_version = importlib.metadata.version("coacd")
    if installed_version != COACD_VERSION:
        raise RuntimeError(
            f"Expected coacd=={COACD_VERSION}, found coacd=={installed_version}"
        )

    manifest_path = models_root / "omomo_collision_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"schema_version": 1, "objects": {}}
    manifest["generator"] = {
        "name": "CoACD",
        "version": COACD_VERSION,
        "parameters": COACD_PARAMETERS,
    }
    for object_name in object_names:
        print(f"Generating {object_name} ...", flush=True)
        manifest["objects"][object_name] = _generate_object(
            object_name,
            models_root,
            coacd,
        )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate convex OMOMO collision assets with CoACD."
    )
    parser.add_argument(
        "objects",
        nargs="*",
        choices=OMOMO_OBJECT_NAMES,
        default=OMOMO_OBJECT_NAMES,
    )
    parser.add_argument(
        "--models-root",
        type=Path,
        default=default_models_root(),
    )
    args = parser.parse_args()
    object_names = tuple(args.objects) if args.objects else OMOMO_OBJECT_NAMES
    manifest_path = generate_collision_assets(
        args.models_root.expanduser().resolve(),
        object_names,
    )
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
