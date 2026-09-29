# ruff: noqa: CPY001, S314

"""Generate the supplied MIA v4.2-G model, preserving updated mechanical data."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import trimesh

from holosoma_retargeting.models.m5.generate_retargeting_assets import generate, numbers, write

ROOT = Path(__file__).resolve().parent
PACKAGE = "__00-mia-v4_2-G"


def main():
    hands = {}
    for side in ("l", "r"):
        mesh = trimesh.load(ROOT / PACKAGE / "meshes" / f"{side}_arm_elbow_link.STL", force="mesh")
        # Center of the distal 5 cm of the supplied hand mesh, 2.5 cm from its tip.
        distal = mesh.vertices[mesh.vertices[:, 2] < mesh.bounds[0, 2] + 0.05]
        hands[side] = np.r_[(distal.min(axis=0)[:2] + distal.max(axis=0)[:2]) / 2, mesh.bounds[0, 2] + 0.025]
    generate("m4g", root=ROOT, package=PACKAGE, hand_points=hands)
    path = ROOT / "m4g_17dof.xml"
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    floor = min(data.body(f"{s}_foot_sphere_{i}_link").xpos[2] for s in ("l", "r") for i in range(1, 6))
    tree = ET.parse(path)
    body = tree.find("worldbody/body[@name='base_link']")
    pos = np.fromstring(body.get("pos"), sep=" ")
    pos[2] -= floor
    body.set("pos", numbers(pos))
    write(path, tree.getroot())
    (ROOT / "landmarks.json").write_text(
        json.dumps(
            {
                "hand_centers_m": {s: v.tolist() for s, v in hands.items()},
                "standing_base_height_m": float(pos[2]),
                "source_urdf": str(Path(PACKAGE) / "urdf" / f"update_{PACKAGE}.urdf"),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
