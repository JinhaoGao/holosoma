# ruff: noqa: CPY001

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from holosoma_retargeting.config_types.data_type import FBX_MOCAP_DEMO_JOINTS
from holosoma_retargeting.data_utils.convert_bvh_mocap import convert_file, read_bvh
from holosoma_retargeting.data_utils.convert_fbx import _SOURCE_JOINT_BY_CANONICAL, _discover_actors
from holosoma_retargeting.data_utils.motion_data import load_human_motion


def fixture(path):
    names = [name for name in FBX_MOCAP_DEMO_JOINTS if name not in {"Hips", "LeftFootMod", "RightFootMod"}]
    special = {
        "Head": [0, 80, 0],
        "LeftUpLeg": [10, 0, 0],
        "RightUpLeg": [-10, 0, 0],
        "LeftToeBase": [10, -90, 10],
        "RightToeBase": [-10, -90, 10],
    }
    lines = [
        "HIERARCHY",
        "ROOT Hips",
        "{",
        "OFFSET 0 0 0",
        "CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation",
    ]
    for name in names:
        lines.extend(
            [
                f"JOINT {name}",
                "{",
                "OFFSET " + " ".join(map(str, special.get(name, [0, 0, 0]))),
                "CHANNELS 3 Zrotation Xrotation Yrotation",
            ]
        )
        if name in {"Head", "LeftToeBase", "RightToeBase"}:
            lines.extend(["End Site", "{", "OFFSET 0 10 0" if name == "Head" else "OFFSET 0 0 5", "}"])
        lines.append("}")
    lines += [
        "}",
        "MOTION",
        "Frames: 2",
        "Frame Time: 0.1",
        " ".join(map(str, [0, 100, 0, 0, 0, 0] + [0] * (3 * len(names)))),
        " ".join(map(str, [10, 100, 0, 90, 0, 0] + [0] * (3 * len(names)))),
    ]
    path.write_text("\n".join(lines))


def test_bvh_conversion_preserves_channel_fk_and_end_sites(tmp_path):
    source = tmp_path / "source.bvh"
    fixture(source)
    output = convert_file(source, tmp_path / "converted.npz", target_fps=10)
    motion = load_human_motion("fbx_mocap", output.parent, output.stem)
    assert motion.fps == 10
    with np.load(output) as data:
        names = list(data["joint_names"])
        positions = data["global_joint_positions"]
        np.testing.assert_allclose(positions[:, names.index("Hips")], [[0, 0, 1], [0.1, 0, 1]], atol=1e-6)
        np.testing.assert_allclose(positions[1, names.index("LeftUpLeg")], [0.1, 0, 1.1], atol=1e-6)
        np.testing.assert_allclose(positions[0, names.index("LeftFootMod")], [0.1, -0.15, 0.1], atol=1e-6)
        rotation = Rotation.from_quat(data["orientation_quaternions_wxyz"][1, 0][[1, 2, 3, 0]])
        np.testing.assert_allclose(rotation.apply([1, 0, 0]), [0, 0, 1], atol=1e-6)
        assert str(data["source_container_format"]) == "bvh"
        assert str(data["orientation_source"]) == "bvh_rotation_channels_fk"
        assert float(data["height"]) == pytest.approx(1.8)


def test_bvh_rejects_motion_rows_that_do_not_match_channels(tmp_path):
    source = tmp_path / "bad.bvh"
    fixture(source)
    source.write_text(source.read_text() + " 42")
    with pytest.raises(ValueError, match="reshape"):
        read_bvh(source)


def test_fbx_actor_prefix_can_contain_multiple_underscores():
    joints = sorted(set(_SOURCE_JOINT_BY_CANONICAL.values()) - {"Hips"})
    nodes = [{"name": "Chen_Zichong_Skeleton_Hips", "children": list(range(1, len(joints) + 1))}]
    nodes.extend({"name": "Chen_Zichong_Skeleton_" + name} for name in joints)
    actors = _discover_actors(nodes)
    assert set(actors) == {"Chen_Zichong_Skeleton"}
    assert set(actors["Chen_Zichong_Skeleton"]) == set(joints) | {"Hips"}
