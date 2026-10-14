# ruff: noqa: CPY001, S314

"""Generate the reverse-knee MIA V5 assets from its updated mechanical URDF."""

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import trimesh

from holosoma_retargeting.models.m5.generate_retargeting_assets import generate, marker, numbers, write

ROOT = Path(__file__).resolve().parent
PACKAGE = "mia-v5-fanqu"


def main():
    hands = {}
    for side in ("l", "r"):
        mesh = trimesh.load(ROOT / PACKAGE / "meshes" / f"{side}_arm_elbow_link.STL", force="mesh")
        distal = mesh.vertices[mesh.vertices[:, 2] < mesh.bounds[0, 2] + 0.05]
        hands[side] = np.r_[(distal.min(axis=0)[:2] + distal.max(axis=0)[:2]) / 2, mesh.bounds[0, 2] + 0.025]
    generate("m5fanqu", root=ROOT, package=PACKAGE, hand_points=hands)
    path = ROOT / "m5fanqu_17dof.xml"
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    floor = min(data.body(f"{side}_foot_sphere_{i}_link").xpos[2] for side in ("l", "r") for i in range(1, 6))
    tree = ET.parse(path)
    body = tree.find("worldbody/body[@name='base_link']")
    pos = np.fromstring(body.get("pos"), sep=" ")
    pos[2] -= floor
    body.set("pos", numbers(pos))
    # A fixed, massless chest anchor at the shoulder midpoint supplies a
    # positional torso target without adding any mechanical degree of freedom.
    chest = (model.body("l_arm_shoulder_pitch_link").pos + model.body("r_arm_shoulder_pitch_link").pos) / 2
    waist = tree.find(".//body[@name='waist_yaw_link']")
    tracking = ET.SubElement(waist, "body", name="torso_tracking_link", pos=numbers(chest))
    ET.SubElement(
        tracking,
        "geom",
        name="torso_tracking_link_geom",
        type="sphere",
        size=".003",
        mass="0",
        contype="0",
        conaffinity="0",
        rgba="1 .6 0 1",
    )
    write(path, tree.getroot())
    urdf_path = ROOT / "m5fanqu_17dof.urdf"
    urdf = ET.parse(urdf_path).getroot()
    marker(urdf, "torso_tracking_link", "waist_yaw_link", chest)
    write(urdf_path, urdf)
    for asset in (path, ROOT / "m5fanqu_17dof.urdf"):
        asset.write_text(asset.read_text().rstrip() + "\n")


if __name__ == "__main__":
    main()
