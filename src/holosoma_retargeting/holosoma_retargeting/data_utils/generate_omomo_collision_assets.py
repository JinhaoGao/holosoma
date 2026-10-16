# ruff: noqa: CPY001
"""Generate OMOMO URDFs using one source-mesh collision body per object."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from holosoma_retargeting.data_utils.object_assets import (
    default_models_root,
    get_omomo_object_asset,
    validate_omomo_object_asset,
)
from holosoma_retargeting.data_utils.omomo import OMOMO_OBJECT_NAMES


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


def generate_collision_assets(models_root: Path, object_names: tuple[str, ...]) -> Path:
    """Regenerate source-mesh collision URDFs and their provenance manifest."""
    manifest_path = models_root / "omomo_collision_manifest.json"
    objects: dict[str, object] = {}
    manifest = {"schema_version": 1, "generator": {"name": "source_mesh", "version": 1}, "objects": objects}
    # Record the active catalog consistently, even when regenerating a subset.
    for object_name in OMOMO_OBJECT_NAMES:
        asset = get_omomo_object_asset(object_name, models_root=models_root)
        validation = validate_omomo_object_asset(asset)
        if object_name in object_names:
            _write_object_urdf(object_name, [asset.mesh_path], asset.urdf_path)
        objects[object_name] = {
            "source_mesh": asset.mesh_path.relative_to(models_root).as_posix(),
            "source_sha256": asset.mesh_sha256,
            "collision_part_count": 1,
            "parts": [
                {
                    "file": asset.mesh_path.relative_to(models_root).as_posix(),
                    "sha256": asset.mesh_sha256,
                    "vertices": validation.vertex_count,
                    "faces": validation.face_count,
                }
            ],
        }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate single-source-mesh OMOMO collision assets.")
    parser.add_argument("objects", nargs="*", metavar="OBJECT")
    parser.add_argument("--models-root", type=Path, default=default_models_root())
    args = parser.parse_args()
    if set(args.objects).difference(OMOMO_OBJECT_NAMES):
        parser.error("Unknown OMOMO object name")
    names = tuple(args.objects) if args.objects else OMOMO_OBJECT_NAMES
    print(f"Wrote {generate_collision_assets(args.models_root.expanduser().resolve(), names)}")


if __name__ == "__main__":
    main()
