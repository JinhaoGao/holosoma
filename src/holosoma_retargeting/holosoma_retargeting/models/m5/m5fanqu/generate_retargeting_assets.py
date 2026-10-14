# ruff: noqa: CPY001, S314

"""Generate the reverse-knee MIA V5 assets from its updated mechanical URDF."""

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import trimesh

from holosoma_retargeting.models.m5.generate_retargeting_assets import generate, numbers, write

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
    write(path, tree.getroot())
    for asset in (path, ROOT / "m5fanqu_17dof.urdf"):
        asset.write_text(asset.read_text().rstrip() + "\n")


if __name__ == "__main__":
    main()
