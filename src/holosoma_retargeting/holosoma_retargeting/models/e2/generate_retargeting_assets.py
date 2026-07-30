# ruff: noqa: CPY001, S314

"""Generate retargeting-ready E2 URDF and MuJoCo assets.

The supplied E2 URDF is fixed-base and does not contain the virtual hand,
toe, and sole links used by the interaction-mesh retargeter.  This generator
keeps that file as the mechanical source of truth and derives the two assets
expected by :class:`RobotConfig`:

* ``e2_23dof.urdf`` for visualization;
* ``e2_23dof.xml`` for floating-base MuJoCo optimization.
"""

from __future__ import annotations

import copy
import math
import os
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

ASSET_DIR = Path(__file__).resolve().parent
SOURCE_URDF = ASSET_DIR / "simplified" / "E2_urdf_0729_02.urdf"
OUTPUT_URDF = ASSET_DIR / "e2_23dof.urdf"
OUTPUT_MJCF = ASSET_DIR / "e2_23dof.xml"

ROOT_HEIGHT = 0.7595
MARKER_RADIUS = 0.005
MARKER_MASS = 0.001
MARKER_INERTIA = 1e-7


def _origin(parent: ET.Element, *, xyz: tuple[float, float, float]) -> None:
    ET.SubElement(
        parent,
        "origin",
        xyz=" ".join(f"{value:.9g}" for value in xyz),
        rpy="0 0 0",
    )


def _add_marker_link(
    robot: ET.Element,
    *,
    name: str,
    parent_name: str,
    xyz: tuple[float, float, float],
    radius: float = MARKER_RADIUS,
) -> None:
    link = ET.SubElement(robot, "link", name=name)
    inertial = ET.SubElement(link, "inertial")
    _origin(inertial, xyz=(0.0, 0.0, 0.0))
    ET.SubElement(inertial, "mass", value=f"{MARKER_MASS:.9g}")
    ET.SubElement(
        inertial,
        "inertia",
        ixx=f"{MARKER_INERTIA:.9g}",
        ixy="0",
        ixz="0",
        iyy=f"{MARKER_INERTIA:.9g}",
        iyz="0",
        izz=f"{MARKER_INERTIA:.9g}",
    )
    visual = ET.SubElement(link, "visual")
    _origin(visual, xyz=(0.0, 0.0, 0.0))
    geometry = ET.SubElement(visual, "geometry")
    ET.SubElement(geometry, "sphere", radius=f"{radius:.9g}")
    material = ET.SubElement(visual, "material", name=f"{name}_material")
    ET.SubElement(material, "color", rgba="0.2 0.2 0.8 1")

    joint = ET.SubElement(robot, "joint", name=name.replace("_link", "_joint"), type="fixed")
    _origin(joint, xyz=xyz)
    ET.SubElement(joint, "parent", link=parent_name)
    ET.SubElement(joint, "child", link=name)


def _derived_urdf_root() -> ET.Element:
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    source_root = ET.parse(SOURCE_URDF, parser=parser).getroot()
    robot = copy.deepcopy(source_root)
    robot.set("name", "E2_23dof")

    for link in robot.findall("link"):
        for layer_name in ("visual", "collision"):
            layer = link.find(layer_name)
            mesh = None if layer is None else layer.find("geometry/mesh")
            if layer is None or mesh is None:
                continue
            filename = mesh.get("filename")
            if filename is None:
                continue
            source_mesh = Path(filename)
            if not source_mesh.is_absolute():
                source_mesh = SOURCE_URDF.parent / source_mesh
            if not source_mesh.is_file():
                link.remove(layer)
                continue
            mesh.set("filename", f"simplified/{Path(filename).name}")

    for side in ("l", "r"):
        lateral = 1.0 if side == "l" else -1.0
        _add_marker_link(
            robot,
            name=f"{side}_leg_ankle_intermediate_1_link",
            parent_name=f"{side}_leg_knee_link",
            xyz=(0.008, lateral * 0.0025, -0.355),
            radius=0.01,
        )
        foot_points = (
            (-0.055, 0.0275, -0.041),
            (-0.055, -0.0275, -0.041),
            (0.11, 0.0275, -0.041),
            (0.11, -0.0275, -0.041),
            (0.145, 0.0, -0.041),
        )
        for index, point in enumerate(foot_points, start=1):
            _add_marker_link(
                robot,
                name=f"{side}_foot_sphere_{index}_link",
                parent_name=f"{side}_leg_ankle_roll_link",
                xyz=point,
            )

    _add_marker_link(
        robot,
        name="l_hand_sphere_link",
        parent_name="l_arm_elbow_link",
        xyz=(0.249, -0.008, -0.159),
    )
    _add_marker_link(
        robot,
        name="r_hand_sphere_link",
        parent_name="r_arm_elbow_link",
        xyz=(0.25, 0.008, -0.159),
    )
    return robot


def _numbers(element: ET.Element | None, attribute: str, default: str) -> str:
    if element is None:
        return default
    return element.get(attribute, default)


def _add_mjcf_inertial(body: ET.Element, link: ET.Element) -> None:
    inertial = link.find("inertial")
    if inertial is None:
        return
    inertia = inertial.find("inertia")
    mass = inertial.find("mass")
    if inertia is None or mass is None:
        return
    origin = inertial.find("origin")
    ET.SubElement(
        body,
        "inertial",
        pos=_numbers(origin, "xyz", "0 0 0"),
        mass=mass.get("value", "0.001"),
        fullinertia=" ".join(inertia.get(name, "0") for name in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")),
    )


def _add_mjcf_geom(body: ET.Element, link: ET.Element) -> str | None:
    visual = link.find("visual")
    if visual is None:
        return None
    geometry = visual.find("geometry")
    if geometry is None:
        return None
    origin = visual.find("origin")
    common = {
        "name": f"{link.get('name')}_geom",
        "pos": _numbers(origin, "xyz", "0 0 0"),
        "euler": _numbers(origin, "rpy", "0 0 0"),
    }
    mesh = geometry.find("mesh")
    if mesh is not None:
        ET.SubElement(
            body,
            "geom",
            type="mesh",
            mesh=str(link.get("name")),
            contype="1",
            conaffinity="1",
            group="2",
            **common,
        )
        return mesh.get("filename")
    sphere = geometry.find("sphere")
    if sphere is not None:
        ET.SubElement(
            body,
            "geom",
            type="sphere",
            size=sphere.get("radius", f"{MARKER_RADIUS:.9g}"),
            rgba="0.2 0.2 0.8 1",
            contype="0",
            conaffinity="0",
            density="0",
            group="1",
            **common,
        )
    return None


def _mjcf_root(robot: ET.Element) -> ET.Element:
    links = {str(link.get("name")): link for link in robot.findall("link") if link.get("name") is not None}
    joints_by_child: dict[str, ET.Element] = {}
    children_by_parent: dict[str, list[ET.Element]] = defaultdict(list)
    for joint in robot.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        child_name = str(child.get("link"))
        joints_by_child[child_name] = joint
        children_by_parent[str(parent.get("link"))].append(joint)

    roots = [name for name in links if name not in joints_by_child]
    if roots != ["base_link"]:
        raise ValueError(f"Expected only base_link as the E2 root, got {roots}")

    mujoco_root = ET.Element("mujoco", model="E2_23dof")
    ET.SubElement(
        mujoco_root,
        "compiler",
        angle="radian",
        meshdir=".",
        autolimits="true",
        balanceinertia="true",
    )
    default = ET.SubElement(mujoco_root, "default")
    ET.SubElement(default, "joint", damping="0.001", armature="0.03", frictionloss="0.1")
    ET.SubElement(default, "geom", rgba="0.78 0.82 0.92 1")

    asset = ET.SubElement(mujoco_root, "asset")
    ET.SubElement(
        asset,
        "texture",
        name="ground_texture",
        type="2d",
        builtin="checker",
        rgb1=".2 .3 .4",
        rgb2=".1 .15 .2",
        width="512",
        height="512",
    )
    ET.SubElement(
        asset,
        "material",
        name="ground_material",
        texture="ground_texture",
        texrepeat="1 1",
        texuniform="true",
        reflectance="0",
    )

    mesh_files: dict[str, str] = {}
    for link_name, link in links.items():
        visual = link.find("visual")
        mesh = None if visual is None else visual.find("geometry/mesh")
        if mesh is not None and mesh.get("filename") is not None:
            mesh_files[link_name] = str(mesh.get("filename"))
    for mesh_name, filename in mesh_files.items():
        ET.SubElement(asset, "mesh", name=mesh_name, file=filename)

    worldbody = ET.SubElement(mujoco_root, "worldbody")
    ET.SubElement(
        worldbody,
        "light",
        directional="true",
        diffuse=".7 .7 .7",
        specular=".2 .2 .2",
        pos="0 0 4",
        dir="0 0 -1",
    )
    ET.SubElement(
        worldbody,
        "geom",
        name="ground",
        type="plane",
        size="0 0 1",
        pos="0 0 0",
        material="ground_material",
        condim="3",
        conaffinity="15",
    )

    def add_body(parent_element: ET.Element, link_name: str, joint: ET.Element | None) -> None:
        attributes = {"name": link_name}
        if joint is None:
            attributes["pos"] = f"0 0 {ROOT_HEIGHT:.9g}"
        else:
            origin = joint.find("origin")
            attributes["pos"] = _numbers(origin, "xyz", "0 0 0")
            rpy = _numbers(origin, "rpy", "0 0 0")
            if rpy != "0 0 0":
                attributes["euler"] = rpy
        body = ET.SubElement(parent_element, "body", **attributes)
        _add_mjcf_inertial(body, links[link_name])
        _add_mjcf_geom(body, links[link_name])
        if joint is None:
            ET.SubElement(body, "freejoint", name="root")
        elif joint.get("type") != "fixed":
            axis = joint.find("axis")
            limit = joint.find("limit")
            ET.SubElement(
                body,
                "joint",
                name=str(joint.get("name")),
                type="hinge",
                axis=_numbers(axis, "xyz", "0 0 1"),
                range=(
                    f"{limit.get('lower', '-3.141592653589793')} {limit.get('upper', '3.141592653589793')}"
                    if limit is not None
                    else f"{-math.pi:.17g} {math.pi:.17g}"
                ),
            )
        for child_joint in children_by_parent.get(link_name, ()):
            child = child_joint.find("child")
            if child is not None:
                add_body(body, str(child.get("link")), child_joint)

    add_body(worldbody, "base_link", None)
    return mujoco_root


def _atomic_write_xml(path: Path, root: ET.Element) -> None:
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as output:
            tree.write(output, encoding="utf-8", xml_declaration=True)
            output.write(b"\n")
            output.flush()
            os.fsync(output.fileno())
        Path(temporary_name).replace(path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def main() -> None:
    robot = _derived_urdf_root()
    _atomic_write_xml(OUTPUT_URDF, robot)
    _atomic_write_xml(OUTPUT_MJCF, _mjcf_root(robot))
    print(f"Wrote {OUTPUT_URDF}")
    print(f"Wrote {OUTPUT_MJCF}")


if __name__ == "__main__":
    main()
