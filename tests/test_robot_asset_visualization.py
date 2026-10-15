# ruff: noqa: CPY001, E402, S314

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest
import yourdfpy

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "holosoma_retargeting"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from holosoma_retargeting.config_types.viser import ViserConfig
from holosoma_retargeting.viser_player import _resolve_runtime_config, load_npz
from holosoma_retargeting.visualization.multi_scene import MultiResultViserConfig, make_multi_result_player
from holosoma_retargeting.visualization.result_loader import (
    AugmentationViserConfig,
    _resolve_asset_path,
    _resolve_robot_assets,
    load_variant_result,
    resolve_qpos_to_viser_joint_indices,
    variant_result_metadata,
)

DEFAULT_URDF = "models/e2/e2_23dof.urdf"
DEFAULT_XML = "models/e2/e2_23dof.xml"


@pytest.fixture(scope="module")
def alternate_e2_assets(tmp_path_factory: pytest.TempPathFactory) -> tuple[str, str]:
    """Reproduce same-name model variants using only versioned robot assets."""
    directory = tmp_path_factory.mktemp("alternate_e2")
    urdf_path = directory / "robot.urdf"
    xml_path = directory / "solver.xml"
    source_urdf = _resolve_asset_path(DEFAULT_URDF)
    source_xml = _resolve_asset_path(DEFAULT_XML)
    urdf = ET.parse(source_urdf)
    xml = ET.parse(source_xml)
    for mesh in urdf.findall(".//mesh"):
        mesh.set("filename", str(source_urdf.parent / mesh.attrib["filename"]))
    compiler = xml.find("compiler")
    assert compiler is not None
    compiler.set("meshdir", str(source_xml.parent))
    for side in ("l", "r"):
        origin = urdf.find(f"joint[@name='{side}_arm_elbow_joint']/origin")
        body = xml.find(f".//body[@name='{side}_arm_elbow_link']")
        assert origin is not None and body is not None
        origin.set("rpy", "0 0.35 0")
        body.set("euler", "0 0.35 0")
        origin.set("xyz", "0.04 0 -0.2")
        body.set("pos", "0.04 0 -0.2")
    urdf.write(urdf_path)
    xml.write(xml_path)
    return str(urdf_path), str(xml_path)


def _write_result(path: Path, config_json: str | None) -> Path:
    payload = {
        "qpos": np.asarray([[0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0] + [0.0] * 23]),
        "robot_type": np.asarray("e2"),
        "contains_object_in_qpos": np.asarray(False),
    }
    if config_json is not None:
        payload["config_json"] = np.asarray(config_json)
    np.savez(path, **payload)
    return path


def _snapshot(urdf: str | None, xml: str | None) -> str:
    return json.dumps({"config": {"robot_config": {"robot_urdf_file": urdf, "robot_mujoco_xml_file": xml}}})


@pytest.mark.parametrize("use_variant", [True, False])
def test_saved_e2_model_aligns_mesh_links_with_solver_skeleton(
    tmp_path: Path,
    alternate_e2_assets: tuple[str, str],
    use_variant: bool,
) -> None:
    urdf, xml = alternate_e2_assets if use_variant else (DEFAULT_URDF, DEFAULT_XML)
    model = mujoco.MjModel.from_xml_path(str(_resolve_asset_path(xml)))
    data = mujoco.MjData(model)
    joint_names = tuple(model.joint(i).name for i in range(1, model.njnt))
    link_names = tuple(model.body(i).name for i in range(1, model.nbody))
    frames, positions, quaternions = [], [], []
    for angle in (0.0, -0.25, -0.6):
        data.qpos[:] = model.qpos0
        data.qpos[:3] += (0.4, -0.2, 0.1)
        data.qpos[3:7] = (np.cos(0.3), 0.0, 0.0, np.sin(0.3))
        for side in ("l", "r"):
            data.qpos[model.joint(f"{side}_arm_elbow_joint").qposadr] = angle
            data.qpos[model.joint(f"{side}_arm_shoulder_yaw_joint").qposadr] = angle * 0.5
        mujoco.mj_forward(model, data)
        # Deliberately save a different articulation order, as real results may do.
        frames.append(np.concatenate((data.qpos[:7], data.qpos[7:][::-1])))
        positions.append(data.xpos[1:].copy())
        quaternions.append(data.xquat[1:].copy())
    path = tmp_path / "e2.npz"
    np.savez(
        path,
        qpos=np.asarray(frames),
        robot_type=np.asarray("e2"),
        config_json=np.asarray(_snapshot(urdf, xml)),
        contains_object_in_qpos=np.asarray(False),
        robot_actuated_joint_names=np.asarray(joint_names[::-1]),
        robot_link_names=np.asarray(link_names),
        robot_link_positions=np.asarray(positions, dtype=np.float32),
        robot_link_quaternions_wxyz=np.asarray(quaternions, dtype=np.float32),
        robot_link_parent_indices=model.body_parentid[1:] - 1,
    )

    qpos, _, _, metadata, _ = load_npz(str(path))
    config = _resolve_runtime_config(ViserConfig(qpos_npz=str(path)), metadata)
    assert Path(config.robot_urdf) == _resolve_asset_path(urdf)
    assert Path(config.robot_mujoco_xml) == _resolve_asset_path(xml)
    robot = yourdfpy.URDF.load(config.robot_urdf, load_meshes=False)
    indices = resolve_qpos_to_viser_joint_indices(
        result_path=path,
        saved_joint_names=tuple(metadata["robot_actuated_joint_names"]),
        viser_joint_names=tuple(robot.actuated_joint_names),
        fallback_mujoco_xml=Path(config.robot_mujoco_xml),
    )
    for frame_index, q in enumerate(qpos):
        robot.update_cfg(q[7:] if indices is None else q[7:][indices])
        root = np.eye(4)
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, q[3:7])
        root[:3, :3] = rotation.reshape(3, 3)
        root[:3, 3] = q[:3]
        mesh_link_positions = np.asarray([(root @ robot.get_transform(name))[:3, 3] for name in link_names])
        np.testing.assert_allclose(mesh_link_positions, metadata["robot_link_positions"][frame_index], atol=1e-6)

    result = load_variant_result("identity", path)
    assert _resolve_robot_assets(AugmentationViserConfig(qpos_npz=path), result) == (
        Path(config.robot_urdf),
        Path(config.robot_mujoco_xml),
    )


@pytest.mark.parametrize("snapshot", [None, "invalid json", "null", "{}", '{"config": null}', _snapshot(None, None)])
def test_results_without_saved_assets_keep_robot_type_defaults(tmp_path: Path, snapshot: str | None) -> None:
    path = _write_result(tmp_path / "legacy.npz", snapshot)
    _, _, _, metadata, _ = load_npz(str(path))
    config = _resolve_runtime_config(ViserConfig(qpos_npz=str(path)), metadata)
    assert Path(config.robot_urdf) == _resolve_asset_path(DEFAULT_URDF)
    assert Path(config.robot_mujoco_xml) == _resolve_asset_path(DEFAULT_XML)


def test_explicit_urdf_override_does_not_reuse_saved_xml(
    tmp_path: Path,
    alternate_e2_assets: tuple[str, str],
) -> None:
    path = _write_result(tmp_path / "e2.npz", _snapshot(*alternate_e2_assets))
    result = load_variant_result("identity", path)
    config = _resolve_runtime_config(
        ViserConfig(qpos_npz=str(path), robot_urdf=DEFAULT_URDF),
        variant_result_metadata(result),
    )
    expected = (_resolve_asset_path(DEFAULT_URDF), _resolve_asset_path(DEFAULT_XML))
    assert (Path(config.robot_urdf), Path(config.robot_mujoco_xml)) == expected
    comparison_config = AugmentationViserConfig(qpos_npz=path, robot_urdf=Path(DEFAULT_URDF))
    assert _resolve_robot_assets(comparison_config, result) == expected


def test_explicit_xml_override_wins_and_same_urdf_keeps_saved_xml(
    tmp_path: Path,
    alternate_e2_assets: tuple[str, str],
) -> None:
    urdf, xml = alternate_e2_assets
    path = _write_result(tmp_path / "e2.npz", _snapshot(urdf, xml))
    _, _, _, metadata, _ = load_npz(str(path))
    config = _resolve_runtime_config(ViserConfig(robot_urdf=urdf), metadata)
    assert Path(config.robot_mujoco_xml) == _resolve_asset_path(xml)
    config = _resolve_runtime_config(ViserConfig(robot_mujoco_xml=DEFAULT_XML), metadata)
    assert Path(config.robot_mujoco_xml) == _resolve_asset_path(DEFAULT_XML)


def test_missing_saved_asset_is_reported_instead_of_loading_another_model(tmp_path: Path) -> None:
    path = _write_result(tmp_path / "e2.npz", _snapshot(str(tmp_path / "missing.urdf"), None))
    _, _, _, metadata, _ = load_npz(str(path))
    with pytest.raises(FileNotFoundError, match=r"missing\.urdf"):
        _resolve_runtime_config(ViserConfig(), metadata)


def test_comparison_rejects_different_e2_models_before_starting_viewer(
    tmp_path: Path,
    alternate_e2_assets: tuple[str, str],
) -> None:
    path = _write_result(tmp_path / "e2.npz", _snapshot(*alternate_e2_assets))
    result = load_variant_result("identity", path)
    other = replace(result, robot_urdf=DEFAULT_URDF, robot_mujoco_xml=DEFAULT_XML)
    with pytest.raises(ValueError, match="different robot assets"):
        make_multi_result_player(MultiResultViserConfig(qpos_npzs=(path, path)), ("new", "old"), [result, other])
