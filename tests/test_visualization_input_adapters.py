# ruff: noqa: CPY001
from __future__ import annotations

import sys
from contextlib import nullcontext
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tyro
import yourdfpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "holosoma_retargeting"))
from holosoma_retargeting import viser_player
from holosoma_retargeting.config_types.viser import ViserConfig
from holosoma_retargeting.data_utils.npz_to_csv import convert_npz_to_csv
from holosoma_retargeting.multi_viser_player import MultiViserConfig, resolve_multi_config
from holosoma_retargeting.paired_retargeting.visualization import PairedViserConfig
from holosoma_retargeting.src import viser_utils
from holosoma_retargeting.visualization import multi_scene
from holosoma_retargeting.visualization.csv_motion import load_csv_motion
from holosoma_retargeting.visualization.multi_scene import (
    HumanSkeletonTrajectory,
    MultiResultViserConfig,
    _make_human_reference_overlay,
)
from holosoma_retargeting.visualization.orientation import OrientationDiagnostics, orientation_skeleton_point_indices
from holosoma_retargeting.visualization.result_loader import load_variant_result, resolve_robot_asset_paths
from holosoma_retargeting.visualization.validation import validate_playback_options


@pytest.fixture
def csv_pair(tmp_path):
    # XML joint order differs from tree order and from the saved solver order.
    urdf = tmp_path / "robot.urdf"
    urdf.write_text("""<robot name="test">
<link name="base"/><link name="arm"/><link name="tip"/>
<joint name="tip_joint" type="revolute"><parent link="arm"/><child link="tip"/>
<origin xyz="1 0 0"/><axis xyz="0 1 0"/><limit lower="-2" upper="2" effort="1" velocity="2"/></joint>
<joint name="root_joint" type="revolute"><parent link="base"/><child link="arm"/>
<origin xyz="0 0 1"/><axis xyz="0 0 1"/><limit lower="-2" upper="2" effort="1" velocity="2"/></joint>
</robot>""")
    npz = tmp_path / "motion.npz"
    qpos = np.asarray(
        [[1, 2, 3, np.cos(0.3), 0, 0, np.sin(0.3), 0.7, -0.2], [2, 3, 4, np.cos(0.4), 0, np.sin(0.4), 0, -0.3, 0.8]]
    )
    np.savez(npz, qpos=qpos, robot_actuated_joint_names=["root_joint", "tip_joint"], fps=29.9976)
    csv = convert_npz_to_csv(npz, urdf)
    return csv, npz, urdf, qpos


def test_csv_roundtrip_preserves_root_and_named_joint_kinematics(csv_pair):
    csv, _, urdf, original = csv_pair
    q, fps, human, metadata, mesh = load_csv_motion(ViserConfig(input_path=str(csv), robot_urdf=str(urdf)))
    assert fps == 30 and human is None and mesh is None
    assert metadata["contains_object_in_qpos"] is False
    assert metadata["robot_actuated_joint_names"] == ("tip_joint", "root_joint")
    np.testing.assert_allclose(q[:, :7], original[:, :7], atol=1e-15)
    np.testing.assert_array_equal(q[:, 7:], original[:, [8, 7]])
    expected = yourdfpy.URDF.load(urdf, load_meshes=False)
    actual = yourdfpy.URDF.load(urdf, load_meshes=False)
    for source, target in zip(original, q, strict=True):
        expected.update_cfg({"root_joint": source[7], "tip_joint": source[8]})
        actual.update_cfg(dict(zip(metadata["robot_actuated_joint_names"], target[7:], strict=True)))
        np.testing.assert_allclose(actual.get_transform("tip"), expected.get_transform("tip"))


def test_single_frame_csv_needs_no_npz_or_xml(csv_pair):
    csv, npz, urdf, _ = csv_pair
    values = np.loadtxt(csv, delimiter=",")[:1]
    np.savetxt(csv, values, delimiter=",")
    npz.unlink()
    q, fps, *_ = load_csv_motion(ViserConfig(input_path=str(csv), robot_urdf=str(urdf), fps=59.94))
    assert q.shape == (1, 9) and fps == 59.94


@pytest.mark.parametrize("invalid", ["empty", "header", "column", "nan", "quaternion"])
def test_csv_invalid_data_reports_a_clear_error(csv_pair, invalid):
    csv, _, urdf, _ = csv_pair
    values = np.loadtxt(csv, delimiter=",")
    if invalid == "empty":
        csv.write_text("")
    elif invalid == "header":
        csv.write_text("x,y,z\n1,2,3\n")
    else:
        if invalid == "column":
            values = values[:, :-1]
        elif invalid == "nan":
            values[0, 0] = np.nan
        else:
            values[0, 3:7] = 0
        np.savetxt(csv, values, delimiter=",")
    with pytest.raises(ValueError, match=r"empty|numeric|columns|non-finite|quaternion"):
        load_csv_motion(ViserConfig(input_path=str(csv), robot_urdf=str(urdf)))


def test_csv_requires_robot_identity(csv_pair):
    with pytest.raises(ValueError, match="--robot-type or --robot-urdf"):
        load_csv_motion(ViserConfig(input_path=str(csv_pair[0])))


@pytest.mark.parametrize("format_name", ["csv", "npz"])
def test_single_dry_run_starts_no_server_and_accepts_robot_only(csv_pair, monkeypatch, format_name):
    csv, npz, urdf, _ = csv_pair
    monkeypatch.setattr(viser_player, "make_player", lambda **_kw: pytest.fail("dry-run started a viewer"))
    path = csv if format_name == "csv" else npz
    viser_player.main(ViserConfig(input_path=str(path), robot_urdf=str(urdf), dry_run=True))


def test_public_cli_aliases_and_removed_start_frame():
    assert "start_frame" not in {f.name for f in fields(ViserConfig)}
    with pytest.raises(SystemExit):
        tyro.cli(ViserConfig, args=["--start-frame", "4"])
    config = tyro.cli(MultiViserConfig, args=["--input-paths", "a.npz", "b.npz", "--port", "8881", "--fps", "29.97"])
    resolved = resolve_multi_config(config)
    assert resolved.qpos_npzs == (Path("a.npz"), Path("b.npz"))
    assert resolved.port == 8881 and resolved.fps == 29.97
    assert resolve_multi_config(MultiViserConfig(qpos_npzs=(Path("a.npz"),))).qpos_npzs == (Path("a.npz"),)
    with pytest.raises(ValueError, match="not both"):
        resolve_multi_config(replace(config, qpos_npzs=(Path("c.npz"),)))
    assert tyro.cli(PairedViserConfig, args=["--input-path", "pair.npz"]).input_path == Path("pair.npz")


@pytest.mark.parametrize(
    ("robot", "old"),
    [
        ("e2", "models/noetix_e2/urdf/E2_V1.0_0805_02.urdf"),
        ("m5c", "models/m5c/m5c_17dof.urdf"),
        ("e1_23dof", "/old/noetix_e1_v2/models/e1_23dof.urdf"),
        ("e1_24dof", "/old/noetix_e1_v2/models/e1_24dof.urdf"),
    ],
)
def test_known_migrated_assets_resolve_without_changing_explicit_models(robot, old):
    urdf, xml = resolve_robot_asset_paths(robot_type=robot, saved_robot_urdf=old)
    assert urdf.is_file()
    if robot != "e2":
        assert xml.is_file()
    if robot.startswith("e1_"):
        assert "noetix_e1_v2" in urdf.parts
    with pytest.raises(FileNotFoundError):
        resolve_robot_asset_paths(robot_type=robot, robot_urdf=old)
    with pytest.raises(FileNotFoundError):
        resolve_robot_asset_paths(robot_type=robot, saved_robot_urdf="/missing/custom_variant.urdf")


@pytest.mark.parametrize(("port", "fps"), [(0, 30), (65536, 30), (8080, 0), (8080, float("nan"))])
def test_invalid_playback_options(port, fps):
    with pytest.raises(ValueError, match=r"port|FPS"):
        validate_playback_options(port, fps)


def test_playback_preserves_fractional_fps_and_stops_at_last_frame(monkeypatch):
    handles = {}

    class Handle:
        def __init__(self, **kw):
            self.value = kw.get("initial_value")

        def on_click(self, callback):
            self.click = callback
            return callback

        def on_update(self, callback):
            self.update = callback
            return callback

    def add(name, **kw):
        handles[name] = Handle(**kw)
        return handles[name]

    gui = SimpleNamespace(add_folder=lambda *_a, **_kw: nullcontext(), add_slider=add, add_button=add, add_number=add)
    server = SimpleNamespace(gui=gui, atomic=nullcontext)
    observed = []
    robot = SimpleNamespace(update_cfg=lambda q: observed.append(q.copy()))
    workers = []
    monkeypatch.setattr(viser_utils, "register_keyboard_shortcut", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        viser_utils.threading, "Thread", lambda target, **_kw: SimpleNamespace(start=lambda: workers.append(target))
    )
    clock = {"now": 0.0}

    def sleep(seconds):
        clock["now"] += max(seconds, 0.002)
        if clock["now"] > 0.5:
            raise StopIteration

    monkeypatch.setattr(viser_utils.time, "perf_counter", lambda: clock["now"])
    monkeypatch.setattr(viser_utils.time, "sleep", sleep)
    q = np.zeros((3, 8))
    q[:, 3] = 1
    q[:, 7] = [0, 1, 2]
    viser_utils.create_motion_control_sliders(
        server,
        robot,
        SimpleNamespace(),
        q,
        robot_dof=1,
        initial_fps=29.9976,
        initial_playing=True,
        initial_interp_mult=2,
        loop=False,
    )
    assert handles["FPS"].value == 29.9976
    with pytest.raises(StopIteration):
        workers[0]()
    assert len(observed) == 5  # initial, half-frame, frame 1, half-frame, final
    np.testing.assert_allclose(np.asarray(observed)[:, 0], [0, 0.5, 1, 1.5, 2])


def test_legacy_e2_keeps_the_saved_mechanical_variant():
    urdf, xml = resolve_robot_asset_paths(
        robot_type="e2",
        saved_robot_urdf="models/noetix_e2/urdf/E2_V1.0_0805_02.urdf",
        saved_robot_mujoco_xml="models/noetix_e2/mjcf/e2.xml",
    )
    repository = Path(__file__).resolve().parents[1]
    assets = repository / "src/holosoma/holosoma/data/robots/e2"
    assert urdf.resolve() == assets / "urdf/E2_V1.0_0805_02.urdf"
    assert xml.resolve() == assets / "mjcf/e2.xml"
    default, _ = resolve_robot_asset_paths(robot_type="e2")
    assert urdf.resolve() != default.resolve()


def test_multimotion_target_axes_use_full_human_joints():
    full = np.asarray([[[0.0, 0.0, 1.0], [1.0, 0.0, 2.0]], [[2.0, 0.0, 1.0], [3.0, 0.0, 2.0]]])
    trajectory = HumanSkeletonTrajectory(("Hand",), full[:, 1:], (), ("Hips", "Hand"), full)
    quat = np.tile([1.0, 0.0, 0.0, 0.0], (2, 1, 1))
    diagnostics = OrientationDiagnostics(("Hips",), ("base",), np.ones(1), quat, quat, np.zeros((2, 1)))
    indices = orientation_skeleton_point_indices(diagnostics, trajectory.full_joint_names, ("", ""))

    def handle(_name, **kw):
        return SimpleNamespace(**kw)

    scene = SimpleNamespace(add_point_cloud=handle, add_line_segments=handle, add_arrows=handle)
    overlay = _make_human_reference_overlay(
        server=SimpleNamespace(scene=scene),
        namespace="test",
        trajectory=trajectory,
        diagnostics=diagnostics,
        joint_indices=np.asarray([0]),
        axis_point_indices=indices,
        axis_points=full,
        offset=np.zeros(3),
        config=MultiResultViserConfig(qpos_npzs=()),
    )
    overlay.update(0.5)
    np.testing.assert_allclose(overlay.target_axes.points[:, 0], np.tile([1.0, 0.0, 1.0], (3, 1)))
    np.testing.assert_allclose(overlay.joints_handle.points, [[2.0, 0.0, 2.0]])


def test_missing_optional_orientation_origin_has_no_negative_index_alias():
    quat = np.tile([1.0, 0.0, 0.0, 0.0], (1, 2, 1))
    diagnostics = OrientationDiagnostics(
        ("Hips", "Missing"), ("base", "unknown"), np.ones(2), quat, quat, np.zeros((1, 2))
    )
    indices = orientation_skeleton_point_indices(diagnostics, ("Hips",), ("base",), allow_missing=True)
    np.testing.assert_array_equal(indices, [0, -1])
    with pytest.raises(ValueError, match="cannot be matched"):
        orientation_skeleton_point_indices(diagnostics, ("Hips",), ("base",))


def test_robot_only_motion_never_reads_human_arm_diagnostics(csv_pair, monkeypatch):
    def unexpected(*_args, **_kwargs):
        pytest.fail("Robot-only motion attempted to read human diagnostics")

    monkeypatch.setattr(viser_player, "load_optional_arm_direction_diagnostics", unexpected)
    overlay = viser_player._build_arm_direction_overlay(
        ViserConfig(qpos_npz=str(csv_pair[0])),
        None,
        None,
        {},
        2,
    )
    assert overlay is None


def test_orientation_selection_uses_diagnostic_names_without_position_mapping(csv_pair, monkeypatch):
    result = load_variant_result("identity", csv_pair[1])
    quat = np.tile([1.0, 0.0, 0.0, 0.0], (2, 1, 1))
    diagnostics = OrientationDiagnostics(("Hips",), ("base",), np.ones(1), quat, quat, np.zeros((2, 1)))
    result = replace(
        result,
        orientation_diagnostics=diagnostics,
        robot_link_names=("base",),
        robot_link_positions=np.zeros((2, 1, 3)),
        robot_link_quaternions_wxyz=quat,
    )
    monkeypatch.setattr(multi_scene, "SavedOrientationAxesOverlay", lambda **kwargs: SimpleNamespace(**kwargs))
    overlay = multi_scene._make_full_robot_orientation_overlay(
        server=None,
        namespace="test",
        result=result,
        offset=np.zeros(3),
        config=MultiResultViserConfig(qpos_npzs=(), orientation_joints=("Hips",)),
    )
    assert overlay.names == ("base",)
