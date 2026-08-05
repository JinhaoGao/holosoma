# ruff: noqa: CPY001, PT009, PT027

from __future__ import annotations

import json
import shutil
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from holosoma_retargeting.config_types.data_type import FBX_MOCAP_DEMO_JOINTS
from holosoma_retargeting.data_utils.convert_fbx import (
    _FBX_Y_UP_TO_RIGHT_HANDED_Z_UP,
    _ORIENTATION_JOINT_NAMES,
    _SOURCE_JOINT_BY_CANONICAL,
    convert_file,
)
from holosoma_retargeting.data_utils.motion_data import load_human_motion


def _append_float_accessor(
    payload: bytearray,
    buffer_views: list[dict[str, int]],
    accessors: list[dict[str, object]],
    values: np.ndarray,
    accessor_type: str,
) -> int:
    while len(payload) % 4:
        payload.append(0)
    offset = len(payload)
    array = np.asarray(values, dtype="<f4")
    raw = array.tobytes()
    payload.extend(raw)
    view_index = len(buffer_views)
    buffer_views.append(
        {
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": len(raw),
        }
    )
    accessor_index = len(accessors)
    accessors.append(
        {
            "bufferView": view_index,
            "componentType": 5126,
            "count": int(array.shape[0]),
            "type": accessor_type,
        }
    )
    return accessor_index


def _write_two_actor_glb(path: Path, *, omit_rotation_joint: str | None = None) -> None:
    source_joints = sorted(set(_SOURCE_JOINT_BY_CANONICAL.values()))
    nodes: list[dict[str, object]] = [{"name": "Root", "children": []}]
    actor_nodes: dict[str, dict[str, int]] = {}
    for actor_index, actor in enumerate(("Actor0", "Actor1")):
        hips_index = len(nodes)
        nodes.append(
            {
                "name": f"{actor}:Hips",
                "translation": [100.0 + 200.0 * actor_index, 100.0, 200.0],
                "children": [],
            }
        )
        nodes[0]["children"].append(hips_index)
        indices = {"Hips": hips_index}
        for joint in source_joints:
            if joint == "Hips":
                continue
            translation = [0.0, 0.0, 0.0]
            if joint == "Head":
                translation = [0.0, 70.0, 0.0]
            elif joint == "LeftUpLeg":
                translation = [10.0, 0.0, 0.0]
            elif joint == "RightUpLeg":
                translation = [-10.0, 0.0, 0.0]
            elif joint == "LeftFoot":
                translation = [10.0, -80.0, 0.0]
            elif joint == "RightFoot":
                translation = [-10.0, -80.0, 0.0]
            elif joint == "LeftToeBase":
                translation = [10.0, -85.0, 14.0]
            elif joint == "RightToeBase":
                translation = [-10.0, -85.0, 14.0]
            elif joint == "LToeEnd":
                translation = [10.0, -100.0, 17.0]
            elif joint == "RToeEnd":
                translation = [-10.0, -100.0, 17.0]
            node_index = len(nodes)
            nodes.append(
                {
                    "name": f"{actor}:{joint}",
                    "translation": translation,
                }
            )
            nodes[hips_index]["children"].append(node_index)
            indices[joint] = node_index
        actor_nodes[actor] = indices

    binary = bytearray()
    buffer_views: list[dict[str, int]] = []
    accessors: list[dict[str, object]] = []
    times_accessor = _append_float_accessor(
        binary,
        buffer_views,
        accessors,
        np.asarray([0.0, 1.0 / 30.0]),
        "SCALAR",
    )
    identity_accessor = _append_float_accessor(
        binary,
        buffer_views,
        accessors,
        np.asarray([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]),
        "VEC4",
    )
    channels: list[dict[str, object]] = []
    samplers: list[dict[str, object]] = []
    for actor_index, actor in enumerate(("Actor0", "Actor1")):
        translations = np.asarray(
            [
                [100.0 + 200.0 * actor_index, 100.0, 200.0],
                [110.0 + 200.0 * actor_index, 100.0, 200.0],
            ]
        )
        translation_accessor = _append_float_accessor(
            binary,
            buffer_views,
            accessors,
            translations,
            "VEC3",
        )
        sampler_index = len(samplers)
        samplers.append(
            {
                "input": times_accessor,
                "output": translation_accessor,
                "interpolation": "LINEAR",
            }
        )
        channels.append(
            {
                "sampler": sampler_index,
                "target": {
                    "node": actor_nodes[actor]["Hips"],
                    "path": "translation",
                },
            }
        )
        for canonical_name in _ORIENTATION_JOINT_NAMES:
            source_joint = _SOURCE_JOINT_BY_CANONICAL[canonical_name]
            if canonical_name == omit_rotation_joint:
                continue
            sampler_index = len(samplers)
            samplers.append(
                {
                    "input": times_accessor,
                    "output": identity_accessor,
                    "interpolation": "LINEAR",
                }
            )
            channels.append(
                {
                    "sampler": sampler_index,
                    "target": {
                        "node": actor_nodes[actor][source_joint],
                        "path": "rotation",
                    },
                }
            )

    document = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": nodes,
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": buffer_views,
        "accessors": accessors,
        "animations": [{"name": "Test", "channels": channels, "samplers": samplers}],
    }
    json_bytes = json.dumps(document, separators=(",", ":")).encode()
    json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)
    binary.extend(b"\x00" * ((4 - len(binary) % 4) % 4))
    total_length = 12 + 8 + len(json_bytes) + 8 + len(binary)
    path.write_bytes(
        struct.pack("<4sII", b"glTF", 2, total_length)
        + struct.pack("<I4s", len(json_bytes), b"JSON")
        + json_bytes
        + struct.pack("<I4s", len(binary), b"BIN\x00")
        + binary
    )


def _write_binary_fbx_header(path: Path) -> None:
    prefix = b"Kaydara FBX Binary  \x00\x1a\x00"
    assert len(prefix) == 23

    def string_property(value: str) -> bytes:
        payload = value.encode()
        return b"S" + struct.pack("<I", len(payload)) + payload

    def integer_property(value: int) -> bytes:
        return b"I" + struct.pack("<i", value)

    def double_property(value: float) -> bytes:
        return b"D" + struct.pack("<d", value)

    def p_node(key: str, value: float) -> tuple[str, tuple[bytes, ...], tuple[object, ...]]:
        value_type = "double" if isinstance(value, float) else "int"
        value_property = double_property(value) if isinstance(value, float) else integer_property(value)
        return (
            "P",
            (
                string_property(key),
                string_property(value_type),
                string_property("Number" if isinstance(value, float) else "Integer"),
                string_property(""),
                value_property,
            ),
            (),
        )

    properties = (
        p_node("UpAxis", 1),
        p_node("UpAxisSign", 1),
        p_node("FrontAxis", 2),
        p_node("FrontAxisSign", 1),
        p_node("CoordAxis", 0),
        p_node("CoordAxisSign", 1),
        p_node("UnitScaleFactor", 1.0),
    )
    global_settings = ("GlobalSettings", (), (("Properties70", (), properties),))

    def encode_node(node: tuple[str, tuple[bytes, ...], tuple[object, ...]], start: int) -> bytes:
        name, encoded_properties, children = node
        name_bytes = name.encode()
        property_bytes = b"".join(encoded_properties)
        child_start = start + 25 + len(name_bytes) + len(property_bytes)
        child_bytes = bytearray()
        for child in children:
            encoded_child = encode_node(child, child_start + len(child_bytes))
            child_bytes.extend(encoded_child)
        if children:
            child_bytes.extend(bytes(25))
        end = child_start + len(child_bytes)
        return (
            struct.pack("<QQQB", end, len(encoded_properties), len(property_bytes), len(name_bytes))
            + name_bytes
            + property_bytes
            + child_bytes
        )

    header = prefix + struct.pack("<I", 7500)
    root_node = encode_node(global_settings, len(header))
    path.write_bytes(header + root_node + bytes(25))


class FbxConversionTest(unittest.TestCase):
    def test_axis_conversion_is_a_proper_right_handed_rotation(self) -> None:
        self.assertAlmostEqual(np.linalg.det(_FBX_Y_UP_TO_RIGHT_HANDED_Z_UP), 1.0)
        np.testing.assert_allclose(
            _FBX_Y_UP_TO_RIGHT_HANDED_Z_UP @ np.asarray((0.0, 1.0, 0.0)),
            (0.0, 0.0, 1.0),
        )
        np.testing.assert_allclose(
            _FBX_Y_UP_TO_RIGHT_HANDED_Z_UP @ np.asarray((0.0, 0.0, 1.0)),
            (0.0, -1.0, 0.0),
        )

    def test_two_skeletons_are_split_into_loadable_canonical_motions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "two_people.fbx"
            fixture_glb = root / "fixture.glb"
            output = root / "converted"
            _write_binary_fbx_header(source)
            _write_two_actor_glb(fixture_glb)

            def fake_export(_: Path, destination: Path, __: Path) -> None:
                shutil.copyfile(fixture_glb, destination)

            with patch(
                "holosoma_retargeting.data_utils.convert_fbx._resolve_assimp_executable",
                return_value=Path("assimp"),
            ), patch(
                "holosoma_retargeting.data_utils.convert_fbx._export_fbx_to_glb",
                side_effect=fake_export,
            ):
                converted = convert_file(source, output, target_fps=30.0)

            self.assertEqual([item.actor for item in converted], ["Actor0", "Actor1"])
            self.assertEqual([item.frame_count for item in converted], [2, 2])
            for actor in converted:
                motion = load_human_motion("fbx_mocap", output, actor.output_path.stem)
                self.assertEqual(motion.joints.shape, (2, len(FBX_MOCAP_DEMO_JOINTS), 3))
                self.assertEqual(motion.orientation_quaternions_wxyz.shape, (2, 51, 4))
                self.assertEqual(motion.orientation_source, "fbx_local_rotation_curves_fk")
                self.assertEqual(motion.source_skeleton_joint_names[0], "Hips")
                self.assertEqual(len(motion.source_skeleton_joint_names), 53)
                self.assertEqual(motion.source_skeleton_positions.shape, (2, 53, 3))
                self.assertEqual(motion.source_skeleton_quaternions_wxyz.shape, (2, 53, 4))
                self.assertEqual(motion.source_skeleton_bind_quaternions_wxyz.shape, (53, 4))
                self.assertAlmostEqual(motion.human_height, 1.7, places=6)
                np.testing.assert_allclose(motion.joints[0, 0], (0.0, 0.0, 1.0), atol=1e-7)
                np.testing.assert_allclose(motion.joints[1, 0], (0.1, 0.0, 1.0), atol=1e-7)
                np.testing.assert_allclose(
                    np.linalg.norm(motion.orientation_quaternions_wxyz, axis=-1),
                    1.0,
                    atol=1e-7,
                )
                with np.load(actor.output_path, allow_pickle=False) as data:
                    self.assertEqual(data["source_actor"].item(), actor.actor)
                    self.assertEqual(data["source_format"].item(), "fbx_mocap")
                    self.assertEqual(data["source_container_format"].item(), "fbx")
                    self.assertEqual(data["source_fbx_up_axis"].item(), 1)
                    self.assertEqual(data["source_fbx_up_axis_sign"].item(), 1)
                    self.assertEqual(data["source_fbx_front_axis"].item(), 2)
                    self.assertEqual(data["source_fbx_front_axis_sign"].item(), 1)
                    self.assertEqual(data["source_fbx_coord_axis"].item(), 0)
                    self.assertEqual(data["source_fbx_coord_axis_sign"].item(), 1)
                    self.assertEqual(data["source_fbx_unit_scale_factor"].item(), 1.0)
                    self.assertEqual(
                        data["source_anatomical_left_axis"].item(),
                        "positive_x_from_named_bind_joints",
                    )
                    self.assertEqual(
                        data["source_anatomical_forward_axis"].item(),
                        "positive_z_from_named_bind_toes",
                    )
                    self.assertEqual(data["t_pose_orientation_quaternions_wxyz"].shape, (51, 4))
                    self.assertEqual(data["source_skeleton_positions"].shape, (2, 53, 3))
                    self.assertEqual(data["source_skeleton_quaternions_wxyz"].shape, (2, 53, 4))
                    self.assertEqual(data["source_skeleton_parent_indices"].shape, (53,))
                    np.testing.assert_allclose(
                        data["root_frame_to_robot_base_quaternion_wxyz"],
                        (2**-0.5, 0.0, 0.0, -(2**-0.5)),
                        atol=1e-7,
                    )

    def test_missing_direct_rotation_curve_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "missing_rotation.fbx"
            fixture_glb = root / "fixture.glb"
            _write_binary_fbx_header(source)
            _write_two_actor_glb(fixture_glb, omit_rotation_joint="LeftArm")

            def fake_export(_: Path, destination: Path, __: Path) -> None:
                shutil.copyfile(fixture_glb, destination)

            with patch(
                "holosoma_retargeting.data_utils.convert_fbx._resolve_assimp_executable",
                return_value=Path("assimp"),
            ), patch(
                "holosoma_retargeting.data_utils.convert_fbx._export_fbx_to_glb",
                side_effect=fake_export,
            ), self.assertRaisesRegex(ValueError, "missing direct FBX rotation curves.*LeftArm"):
                convert_file(source, root / "converted")


if __name__ == "__main__":
    unittest.main()
