# ruff: noqa: CPY001, S314

"""Derive floating-base M5 assets from the supplied updated mechanical URDF."""

import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import mujoco
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parent


def numbers(values):
    return " ".join(f"{value:.12g}" for value in values)


def marker(robot, name, parent, point):
    link = ET.SubElement(robot, "link", name=name)
    visual = ET.SubElement(link, "visual")
    ET.SubElement(ET.SubElement(visual, "geometry"), "sphere", radius="0.003")
    joint = ET.SubElement(robot, "joint", name=name.replace("_link", "_joint"), type="fixed")
    ET.SubElement(joint, "parent", link=parent)
    ET.SubElement(joint, "child", link=name)
    ET.SubElement(joint, "origin", xyz=numbers(point), rpy="0 0 0")


def write(path, root):
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def generate(robot_type, *, root=None, package=None, hand_points=None):
    root = Path(root) if root is not None else ROOT / robot_type
    package = package or f"__00-mia-v5-0{robot_type[-1].upper()}"
    source = root / package / "urdf" / f"update_{package}.urdf"
    robot = ET.parse(source).getroot()
    robot.set("name", f"{robot_type}_17dof")
    for mesh in robot.findall(".//mesh"):
        mesh.set("filename", f"{package}/meshes/" + Path(mesh.get("filename")).name)
    for side in ("l", "r"):
        foot = trimesh.load(root / f"{package}/meshes/{side}_leg_ankle_pitch_link.STL", force="mesh")
        low, high = foot.bounds
        # Surface landmarks from the actual sole: heel, forefoot, and toe.
        heel, toe = low[0] + 0.005, high[0] - 0.005
        lateral = min(abs(low[1]), abs(high[1])) - 0.003
        for i, point in enumerate(
            (
                (heel, lateral, low[2]),
                (heel, -lateral, low[2]),
                (toe - 0.025, lateral, low[2]),
                (toe - 0.025, -lateral, low[2]),
                (toe, 0, low[2]),
            ),
            1,
        ):
            marker(robot, f"{side}_foot_sphere_{i}_link", f"{side}_leg_ankle_pitch_link", point)
        marker(
            robot,
            f"{side}_hand_sphere_link",
            f"{side}_arm_elbow_link",
            (hand_points[side] if hand_points is not None else (0, -0.0035 if side == "l" else 0.0035, -0.2230001986)),
        )
    write(root / f"{robot_type}_17dof.urdf", robot)

    mj = ET.Element("mujoco", model=f"{robot_type}_17dof")
    ET.SubElement(mj, "compiler", angle="radian", autolimits="true", meshdir=".")
    defaults = ET.SubElement(mj, "default")
    ET.SubElement(defaults, "joint", damping=".001", armature=".03", frictionloss=".1")
    ET.SubElement(defaults, "geom", rgba=".78 .82 .92 1")
    assets = ET.SubElement(mj, "asset")
    world = ET.SubElement(mj, "worldbody")
    ET.SubElement(world, "light", pos="0 0 4", dir="0 0 -1", directional="true")
    ET.SubElement(
        world, "geom", name="ground", type="plane", size="0 0 1", rgba=".25 .28 .3 1", contype="1", conaffinity="1"
    )
    links = {link.get("name"): link for link in robot.findall("link")}
    children = defaultdict(list)
    for joint in robot.findall("joint"):
        children[joint.find("parent").get("link")].append(joint)
    actuators = ET.SubElement(mj, "actuator")

    def add_body(parent, name, joint=None):
        attrs = {"name": name}
        if joint is None:
            attrs["pos"] = "0 0 1.3335"
        else:
            origin = joint.find("origin")
            attrs["pos"] = origin.get("xyz", "0 0 0")
            attrs["euler"] = origin.get("rpy", "0 0 0")
        body = ET.SubElement(parent, "body", **attrs)
        link = links[name]
        inertial = link.find("inertial")
        if inertial is not None:
            inertia = inertial.find("inertia")
            assert np.allclose(np.fromstring(inertial.find("origin").get("rpy", "0 0 0"), sep=" "), 0)
            ET.SubElement(
                body,
                "inertial",
                pos=inertial.find("origin").get("xyz"),
                mass=inertial.find("mass").get("value"),
                fullinertia=" ".join(inertia.get(k) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")),
            )
        if joint is None:
            ET.SubElement(body, "freejoint", name="root")
        elif joint.get("type") != "fixed":
            limit = joint.find("limit")
            ET.SubElement(
                body,
                "joint",
                name=joint.get("name"),
                type="hinge",
                axis=joint.find("axis").get("xyz"),
                range=f"{limit.get('lower')} {limit.get('upper')}",
            )
            ET.SubElement(
                actuators,
                "motor",
                name=joint.get("name"),
                joint=joint.get("name"),
                ctrllimited="true",
                ctrlrange=f"-{limit.get('effort')} {limit.get('effort')}",
            )
        mesh = link.find("visual/geometry/mesh")
        if mesh is not None:
            ET.SubElement(assets, "mesh", name=name, file=mesh.get("filename"))
            origin = link.find("visual/origin")
            ET.SubElement(
                body,
                "geom",
                name=f"{name}_geom",
                type="mesh",
                mesh=name,
                pos=origin.get("xyz", "0 0 0"),
                euler=origin.get("rpy", "0 0 0"),
                contype="1",
                conaffinity="1",
                group="2",
            )
        else:
            is_sole = "_foot_sphere_" in name
            ET.SubElement(
                body,
                "geom",
                name=f"{name}_geom",
                type="sphere",
                size=".003",
                pos="0 0 .003" if is_sole else "0 0 0",
                contype="1" if is_sole else "0",
                conaffinity="1" if is_sole else "0",
                density="0",
                group="1",
            )
        for child in children[name]:
            add_body(body, child.find("child").get("link"), child)

    add_body(world, "base_link")
    destination = root / f"{robot_type}_17dof.xml"
    write(destination, mj)
    model = mujoco.MjModel.from_xml_path(str(destination))
    assert model.nq == 24 and model.nu == 17
    print(f"Wrote {destination}; nq={model.nq}, nu={model.nu}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=("m5c", "m5d", "m5e"), required=True)
    generate(parser.parse_args().robot)
