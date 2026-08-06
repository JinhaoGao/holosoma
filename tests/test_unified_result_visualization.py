# ruff: noqa: CPY001, E402, PT009, PT027
from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "holosoma_retargeting"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from holosoma_retargeting.config_types.robot import RobotConfig
from holosoma_retargeting.config_types.viser import ViserConfig
from holosoma_retargeting.multi_viser_player import (
    MultiViserConfig,
    resolve_multi_config,
)
from holosoma_retargeting.src.viser_utils import (
    actuated_joint_names_from_mujoco_xml,
    actuated_joint_names_from_urdf,
)
from holosoma_retargeting.viser_player import (
    MappedSkeletonOverlay,
    _build_mapped_skeleton_overlay,
    _build_orientation_overlay,
    _build_qpos_to_viser_joint_indices,
    _build_retargeting_point_cloud_overlay,
    load_npz,
)
from holosoma_retargeting.viser_player import (
    main as single_viewer_main,
)
from holosoma_retargeting.visualization.multi_scene import (
    MultiResultViserConfig,
    load_comparison_results,
    resolve_comparison_object_urdfs,
)
from holosoma_retargeting.visualization.result_loader import (
    AugmentationViserConfig,
    _resolve_asset_path,
    discover_variant_paths,
    load_result_family,
    load_variant_result,
    resolve_qpos_to_viser_joint_indices,
    variant_result_metadata,
)


def _identity_quaternions(frames: int, items: int) -> np.ndarray:
    quaternions = np.zeros((frames, items, 4), dtype=np.float32)
    quaternions[..., 0] = 1.0
    return quaternions


def _refresh_config_identity(payload: dict[str, object]) -> None:
    decoded = json.loads(str(np.asarray(payload["config_json"]).item()))
    decoded["run_kind"] = str(np.asarray(payload["run_kind"]).item())
    decoded["experiment_name"] = str(np.asarray(payload["experiment_name"]).item()) or None
    decoded["dataset_partition"] = str(np.asarray(payload["dataset_partition"]).item())
    decoded["sequence_key"] = str(np.asarray(payload["sequence_key"]).item())
    decoded.setdefault("variant", {})["name"] = str(np.asarray(payload["variant"]).item())
    solver_config = decoded.setdefault("config", {})
    solver_config["data_format"] = str(np.asarray(payload["source_data_format"]).item())
    solver_config["task_type"] = str(np.asarray(payload["task_type"]).item())
    solver_config.setdefault("robot_config", {})["robot_type"] = str(np.asarray(payload["robot_type"]).item())
    solver_config.setdefault("task_config", {})["object_name"] = str(np.asarray(payload["object_name"]).item())
    config_json = json.dumps(
        decoded,
        separators=(",", ":"),
        sort_keys=True,
    )
    payload["config_json"] = np.asarray(config_json)


def _set_real_object_asset_manifest(
    payload: dict[str, object],
    object_urdf: Path,
) -> None:
    payload["object_urdf"] = np.asarray(str(object_urdf))


def write_loose_result(
    path: str | Path,
    payload: dict[str, object],
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **payload)
    return destination


def _result_payload(
    *,
    variant: str = "identity",
    include_source_orientations: bool = True,
    include_diagnostics: bool = False,
) -> dict[str, object]:
    human_joints = np.asarray(
        [
            [[0.0, 0.0, 1.0], [0.0, 0.2, 1.2], [0.2, 0.4, 1.4]],
            [[0.1, 0.0, 1.0], [0.1, 0.2, 1.2], [0.3, 0.4, 1.4]],
        ],
        dtype=np.float32,
    )
    robot_link_positions = np.asarray(
        [
            [[0.0, 0.0, 0.9], [0.0, 0.0, 1.1], [0.2, 0.3, 1.3]],
            [[0.1, 0.0, 0.9], [0.1, 0.0, 1.1], [0.3, 0.3, 1.3]],
        ],
        dtype=np.float32,
    )
    qpos = np.zeros((2, 7), dtype=np.float64)
    qpos[:, 3] = 1.0
    config_json = json.dumps(
        {
            "config": {
                "data_format": "gvhmr",
                "task_type": "robot_only",
                "robot_config": {"robot_type": "g1"},
                "task_config": {"object_name": "ground"},
            },
            "run_kind": "single",
            "variant": {"name": variant},
            "experiment_name": None,
            "dataset_partition": "gvhmr",
            "sequence_key": "sequence",
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    payload: dict[str, object] = {
            "variant": np.asarray(variant),
            "run_kind": np.asarray("single"),
            "dataset_partition": np.asarray("gvhmr"),
            "sequence_key": np.asarray("sequence"),
            "experiment_name": np.asarray(""),
            "source_path": np.asarray("/dataset/sequence.npz"),
            "config_json": np.asarray(config_json),
            "source_data_format": np.asarray("gvhmr"),
            "robot_type": np.asarray("g1"),
            "task_type": np.asarray("robot_only"),
            "object_name": np.asarray("ground"),
            "object_urdf": np.asarray(""),
            "qpos": qpos,
            "human_joints": human_joints[:, (0, 2)],
            "human_joint_names": np.asarray(("Hips", "LeftHand")),
            "human_joint_parent_indices": np.asarray(
                (-1, 0),
                dtype=np.int32,
            ),
            "mapped_human_joints": human_joints[:, (0, 2)],
            "mapped_human_joint_names": np.asarray(("Hips", "LeftHand")),
            "mapped_robot_joints": robot_link_positions[:, (0, 2)],
            "mapped_robot_link_names": np.asarray(("base", "left_hand")),
            "human_points_world": human_joints[:, (0, 2)],
            "robot_points_world": robot_link_positions[:, (0, 2)],
            "terrain_points_world": np.empty((2, 0, 3), dtype=np.float32),
            "robot_link_positions": robot_link_positions[:, (0, 2)],
            "robot_link_quaternions_wxyz": _identity_quaternions(2, 2),
            "robot_link_names": np.asarray(("base", "left_hand")),
            "robot_link_parent_indices": np.asarray(
                (-1, 0),
                dtype=np.int32,
            ),
            "object_poses_demo": np.zeros((2, 7), dtype=np.float32),
            "object_poses_target": np.zeros((2, 7), dtype=np.float32),
            "object_pose_layout": np.asarray("xyz_wxyz"),
            "foot_sticking_side_names": np.asarray(("left", "right")),
            "foot_sticking_states": np.zeros((2, 2), dtype=bool),
            "foot_sticking_tolerance": np.asarray(1e-3),
            "foot_sticking_enabled_for_saved_trajectory": np.asarray(False),
            "interaction_source_vertices_w": human_joints[:, (0, 2)],
            "interaction_target_vertices_w": (robot_link_positions[:, (0, 2)]),
            "interaction_tetrahedra": np.empty(
                (2, 0, 4),
                dtype=np.int32,
            ),
            "interaction_tetrahedra_counts": np.zeros(
                2,
                dtype=np.int32,
            ),
            "interaction_num_human_vertices": np.int32(2),
            "interaction_num_object_vertices": np.int32(0),
            "interaction_mesh_edges_default": np.asarray("cross"),
            "contains_object_in_qpos": np.asarray(False),
            "fps": np.asarray(30.0),
            "cost": np.asarray(0.5),
            "source_human_height": np.asarray(1.78),
            "human_position_scale": np.asarray(1.0),
        }
    if include_source_orientations:
        payload.update(
            {
                "human_orientation_joint_names": np.asarray(("Hips", "LeftHand")),
                "human_orientation_quaternions_wxyz": (_identity_quaternions(2, 2)),
                "orientation_source": np.asarray("direct_local_rotation_fk"),
            }
        )
    else:
        payload.pop("human_orientation_joint_names", None)
        payload.pop("human_orientation_quaternions_wxyz", None)
        payload["orientation_source"] = np.asarray("absent")
    if include_diagnostics:
        payload.update(
            {
                "orientation_human_joint_names": np.asarray(("Hips",)),
                "orientation_robot_link_names": np.asarray(("base",)),
                "orientation_weights": np.asarray((1.0,), dtype=np.float32),
                "orientation_tracking_enabled": np.asarray(True),
                "orientation_diagnostics_enabled": np.asarray(True),
                "orientation_alignment_quaternions_wxyz": (_identity_quaternions(1, 1).reshape(1, 4)),
                "orientation_reference_human_quaternions_wxyz": (_identity_quaternions(1, 1).reshape(1, 4)),
                "orientation_reference_robot_quaternions_wxyz": (_identity_quaternions(1, 1).reshape(1, 4)),
                "orientation_reference_robot_qpos": qpos[0],
                "orientation_target_quaternions_wxyz": (_identity_quaternions(2, 1)),
                "orientation_robot_quaternions_wxyz": (_identity_quaternions(2, 1)),
                "orientation_errors_rad": np.zeros((2, 1), dtype=np.float32),
                "orientation_frame_costs": np.zeros(2, dtype=np.float64),
            }
        )
    if variant != "identity":
        payload["run_kind"] = np.asarray("augmentation")
        decoded = json.loads(str(np.asarray(payload["config_json"]).item()))
        variant_config = decoded.setdefault("variant", {})
        if variant.startswith("z_scale_"):
            variant_config["object_scale"] = [1.0, 1.0, 1.2]
        elif variant.startswith("rot_"):
            variant_config["rotation"] = 0.1
        else:
            variant_config["translation"] = [0.1, 0.0, 0.0]
        payload["config_json"] = np.asarray(
            json.dumps(
                decoded,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    _refresh_config_identity(payload)
    return payload


class _FakeScene:
    def __init__(self) -> None:
        self.arrows: dict[str, SimpleNamespace] = {}
        self.point_clouds: dict[str, SimpleNamespace] = {}

    def add_arrows(self, name: str, **kwargs) -> SimpleNamespace:
        handle = SimpleNamespace(**kwargs)
        self.arrows[name] = handle
        return handle

    def add_point_cloud(self, name: str, **kwargs) -> SimpleNamespace:
        handle = SimpleNamespace(**kwargs)
        self.point_clouds[name] = handle
        return handle


class UnifiedResultVisualizationTests(unittest.TestCase):
    def test_standard_package_assets_resolve_from_visualization_loader(self):
        resolved = _resolve_asset_path("models/g1/g1_29dof.urdf")

        self.assertTrue(resolved.is_file())

    def test_compact_retargeting_point_clouds_are_directly_renderable(self):
        trajectories = {
            "human_points_world": np.asarray(
                [[[0.0, 0.0, 1.0]], [[0.2, 0.0, 1.0]]],
                dtype=np.float32,
            ),
            "robot_points_world": np.asarray(
                [[[0.0, 0.0, 0.9]], [[0.2, 0.0, 0.9]]],
                dtype=np.float32,
            ),
            "terrain_points_world": np.asarray(
                [[[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]],
                dtype=np.float32,
            ),
            "object_keypoints": {
                "demo_world": np.asarray(
                    [[[0.0, 0.4, 0.5]], [[0.1, 0.4, 0.5]]],
                    dtype=np.float32,
                ),
                "target_world": np.asarray(
                    [[[0.0, 0.5, 0.5]], [[0.1, 0.5, 0.5]]],
                    dtype=np.float32,
                ),
            },
        }
        server = SimpleNamespace(scene=_FakeScene())

        overlay = _build_retargeting_point_cloud_overlay(
            ViserConfig(show_point_clouds=True),
            server,
            trajectories,
        )

        self.assertIsNotNone(overlay)
        self.assertEqual(
            tuple(overlay.trajectories),
            ("human", "robot", "terrain", "object_demo", "object_target"),
        )
        overlay.draw(0.5)
        np.testing.assert_allclose(
            server.scene.point_clouds["/overlays/retargeting_points/human"].points,
            np.asarray([[0.1, 0.0, 1.0]], dtype=np.float32),
        )
        overlay.set_visible(False)
        self.assertTrue(
            all(not handle.visible for handle in server.scene.point_clouds.values()),
        )

    def test_flat_motion_family_is_discovered_from_directory_or_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "sequence.npz").touch()
            (root / "sequence_rot_0.npz").touch()

            from_directory = discover_variant_paths(
                root,
                variants=("identity", "rot_0"),
            )
            from_identity = discover_variant_paths(
                root / "sequence.npz",
                variants=("identity", "rot_0"),
            )
            from_sibling = discover_variant_paths(
                root / "sequence_rot_0.npz",
                variants=("identity", "rot_0"),
            )

        self.assertEqual(from_directory, from_identity)
        self.assertEqual(from_directory, from_sibling)
        self.assertEqual(from_directory["identity"].name, "sequence.npz")

    def test_family_without_variants_discovers_climbing_scale_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            object_urdf = root / "multi_boxes.urdf"
            object_urdf.write_text(
                "<robot name='multi_boxes'><link name='base'/></robot>",
                encoding="utf-8",
            )
            for variant in ("identity", "z_scale_1p2"):
                payload = _result_payload(variant=variant)
                payload["task_type"] = np.asarray("climbing")
                payload["object_name"] = np.asarray("multi_boxes")
                _set_real_object_asset_manifest(payload, object_urdf)
                _refresh_config_identity(payload)
                suffix = "" if variant == "identity" else f"_{variant}"
                write_loose_result(root / f"climb{suffix}.npz", payload)

            resolved = resolve_multi_config(MultiViserConfig(family=root))
            labels, results = load_comparison_results(resolved)

        self.assertEqual(
            tuple(path.name for path in resolved.qpos_npzs),
            ("climb.npz", "climb_z_scale_1p2.npz"),
        )
        self.assertEqual(labels, ("identity", "z_scale_1p2"))
        self.assertEqual(
            tuple(result.variant for result in results),
            labels,
        )

    def test_comparison_default_labels_use_saved_semantics_and_stay_unique(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = (
                root / "first" / "motion.npz",
                root / "second" / "motion.npz",
            )
            for path in paths:
                path.parent.mkdir()
                write_loose_result(path, _result_payload())

            labels, _results = load_comparison_results(
                MultiResultViserConfig(qpos_npzs=paths),
            )

        self.assertEqual(labels, ("single:identity", "single:identity [2]"))

    def test_comparison_rejects_symlink_aliases_of_the_same_result(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            result_path = root / "motion.npz"
            alias_path = root / "motion_alias.npz"
            write_loose_result(result_path, _result_payload())
            alias_path.symlink_to(result_path)

            with self.assertRaisesRegex(
                ValueError,
                "unique physical files",
            ):
                load_comparison_results(
                    MultiResultViserConfig(
                        qpos_npzs=(result_path, alias_path),
                    )
                )

    def test_single_viewer_rejects_dual_canonical_and_legacy_inputs(self):
        with self.assertRaisesRegex(
            ValueError,
            "either --input-path or the legacy --qpos-npz",
        ):
            single_viewer_main(
                ViserConfig(
                    input_path="canonical.npz",
                    qpos_npz="legacy.npz",
                )
            )

    def test_family_discovery_uses_filenames_without_a_metadata_audit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_loose_result(
                root / "motion.npz",
                _result_payload(variant="identity"),
            )
            write_loose_result(
                root / "motion_rot_0.npz",
                _result_payload(variant="trans_0"),
            )
            resolved = resolve_multi_config(MultiViserConfig(family=root))
            results = load_result_family(
                AugmentationViserConfig(
                    qpos_npz=root,
                    variants=("identity", "rot_0"),
                )
            )

        self.assertEqual(
            resolved.qpos_npzs,
            (root / "motion.npz", root / "motion_rot_0.npz"),
        )
        self.assertEqual([result.variant for result in results], ["identity", "rot_0"])

    def test_strict_family_accepts_original_as_identity_alias(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_loose_result(
                root / "motion.npz",
                _result_payload(variant="identity"),
            )

            results = load_result_family(
                AugmentationViserConfig(
                    qpos_npz=root,
                    variants=("original",),
                )
            )

        self.assertEqual([result.variant for result in results], ["identity"])

    def test_production_fields_flow_through_single_and_multi_viewers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "motion.npz"
            payload = _result_payload()
            foot_states = np.asarray(payload["foot_sticking_states"]).copy()
            foot_states[0, 0] = True
            payload["foot_sticking_states"] = foot_states
            payload["interaction_mesh_edges_default"] = np.asarray("all")
            write_loose_result(path, payload)

            result = load_variant_result("requested-label", path)
            qpos, fps, human_joints, metadata, interaction_mesh = load_npz(str(path))
            labels, comparison_results = load_comparison_results(
                MultiResultViserConfig(
                    qpos_npzs=(path,),
                    labels=("baseline",),
                )
            )

        self.assertEqual(result.variant, "requested-label")
        self.assertIsNotNone(result.config_json)
        self.assertEqual(result.dataset_partition, "gvhmr")
        self.assertEqual(result.sequence_key, "sequence")
        self.assertIsNone(result.experiment_name)
        self.assertEqual(result.source_data_format, "gvhmr")
        self.assertEqual(result.cost, 0.5)
        self.assertEqual(result.source_human_height, 1.78)
        self.assertEqual(result.human_position_scale, 1.0)
        self.assertEqual(
            result.orientation_source,
            "direct_local_rotation_fk",
        )
        self.assertEqual(result.robot_actuated_joint_names, ())
        self.assertEqual(
            result.human_orientation_joint_names,
            ("Hips", "LeftHand"),
        )
        np.testing.assert_array_equal(
            result.human_joint_parent_indices,
            np.asarray((-1, 0), dtype=np.int32),
        )
        self.assertEqual(result.robot_link_names, ("base", "left_hand"))
        np.testing.assert_array_equal(
            result.robot_link_parent_indices,
            np.asarray((-1, 0), dtype=np.int32),
        )
        self.assertEqual(result.robot_link_positions.shape, (2, 2, 3))
        self.assertEqual(result.robot_link_quaternions_wxyz.shape, (2, 2, 4))
        np.testing.assert_array_equal(
            result.human_points_world,
            result.human_points,
        )
        np.testing.assert_array_equal(
            result.robot_points_world,
            result.robot_points,
        )
        self.assertEqual(result.terrain_points_world.shape, (2, 0, 3))
        self.assertIsNotNone(result.interaction_mesh)
        self.assertEqual(result.interaction_mesh_edges_default, "all")
        self.assertEqual(qpos.shape, (2, 7))
        self.assertEqual(human_joints.shape, (2, 2, 3))
        self.assertEqual(fps, 30.0)
        self.assertEqual(metadata["dataset_partition"], "gvhmr")
        self.assertEqual(metadata["sequence_key"], "sequence")
        self.assertEqual(
            metadata["orientation_source"],
            "direct_local_rotation_fk",
        )
        self.assertIsNone(metadata["robot_actuated_joint_names"])
        np.testing.assert_array_equal(
            metadata["human_joint_parent_indices"],
            np.asarray((-1, 0), dtype=np.int32),
        )
        self.assertEqual(metadata["robot_link_names"], ["base", "left_hand"])
        np.testing.assert_array_equal(
            metadata["human_points_world"],
            result.human_points,
        )
        np.testing.assert_array_equal(
            metadata["robot_points_world"],
            result.robot_points,
        )
        self.assertEqual(metadata["terrain_points_world"].shape, (2, 0, 3))
        self.assertEqual(metadata["interaction_mesh_edges_default"], "all")
        self.assertEqual(metadata["foot_sticking"]["tolerance"], 1e-3)
        self.assertIsNotNone(interaction_mesh)
        self.assertEqual(labels, ("baseline",))
        np.testing.assert_array_equal(
            comparison_results[0].robot_link_quaternions_wxyz,
            result.robot_link_quaternions_wxyz,
        )

    def test_object_pose_and_asset_metadata_flow_through_variant_loader(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            object_urdf = root / "multi_boxes.urdf"
            object_urdf.write_text(
                "<robot name='object'><link name='object'/></robot>",
                encoding="utf-8",
            )
            payload = _result_payload()
            payload["task_type"] = np.asarray("climbing")
            payload["object_name"] = np.asarray("multi_boxes")
            _set_real_object_asset_manifest(payload, object_urdf)
            _refresh_config_identity(payload)
            result_path = root / "motion.npz"

            write_loose_result(result_path, payload)
            result = load_variant_result("identity", result_path)
            metadata = variant_result_metadata(result)

        np.testing.assert_array_equal(
            result.object_poses_demo,
            payload["object_poses_demo"],
        )
        np.testing.assert_array_equal(
            result.object_poses_target,
            payload["object_poses_target"],
        )
        self.assertEqual(result.object_pose_layout, "xyz_wxyz")
        for field in (
            "object_poses_demo",
            "object_poses_target",
            "object_pose_layout",
        ):
            if isinstance(metadata[field], np.ndarray):
                np.testing.assert_array_equal(
                    metadata[field],
                    getattr(result, field),
                )
            else:
                self.assertEqual(metadata[field], getattr(result, field))

    def test_multi_viewer_resolves_each_climbing_variant_scene_asset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            asset_paths = (
                root / "identity.urdf",
                root / "z_scale_1p2.urdf",
            )
            result_paths = (
                root / "motion.npz",
                root / "motion_z_scale_1.2.npz",
            )
            variants = ("identity", "z_scale_1.2")
            for variant, asset_path, result_path in zip(
                variants,
                asset_paths,
                result_paths,
                strict=True,
            ):
                asset_path.write_text(
                    "<robot name='object'><link name='object'/></robot>",
                    encoding="utf-8",
                )
                payload = _result_payload(variant=variant)
                payload.update(
                    {
                        "task_type": np.asarray("climbing"),
                        "object_name": np.asarray("multi_boxes"),
                        "contains_object_in_qpos": np.asarray(False),
                    }
                )
                _set_real_object_asset_manifest(payload, asset_path)
                decoded = json.loads(str(np.asarray(payload["config_json"]).item()))
                decoded["variant"]["scale"] = 1.0 if variant == "identity" else 1.2
                payload["config_json"] = np.asarray(
                    json.dumps(
                        decoded,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                )
                _refresh_config_identity(payload)
                write_loose_result(result_path, payload)
            results = [
                load_variant_result(label, path)
                for label, path in zip(
                    variants,
                    result_paths,
                    strict=True,
                )
            ]
            family = load_result_family(
                AugmentationViserConfig(
                    qpos_npz=result_paths[0],
                    variants=variants,
                )
            )
            _, comparison = load_comparison_results(MultiResultViserConfig(qpos_npzs=result_paths))

            resolved = resolve_comparison_object_urdfs(
                MultiResultViserConfig(qpos_npzs=result_paths),
                results,
            )
            override = root / "override.urdf"
            override.touch()
            overridden = resolve_comparison_object_urdfs(
                MultiResultViserConfig(
                    qpos_npzs=result_paths,
                    object_urdf=override,
                ),
                results,
            )

        self.assertEqual(resolved, asset_paths)
        self.assertEqual(overridden, (override, override))
        self.assertEqual(len(family), 2)
        self.assertEqual(len(comparison), 2)
        self.assertNotEqual(family[0].object_urdf, family[1].object_urdf)

    def test_visualization_loader_does_not_audit_saved_asset_closure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            mesh = root / "mesh.obj"
            mesh.write_bytes(b"saved mesh")
            object_urdf = root / "object.urdf"
            object_urdf.write_text(
                (
                    "<robot name='object'><link name='object'><visual><geometry>"
                    "<mesh filename='mesh.obj'/></geometry></visual></link></robot>"
                ),
                encoding="utf-8",
            )
            override = root / "override.urdf"
            override.write_text(
                "<robot name='override'><link name='override'/></robot>",
                encoding="utf-8",
            )
            result_path = root / "motion.npz"
            payload = _result_payload()
            payload["task_type"] = np.asarray("climbing")
            payload["object_name"] = np.asarray("multi_boxes")
            _set_real_object_asset_manifest(payload, object_urdf)
            _refresh_config_identity(payload)
            write_loose_result(result_path, payload)

            load_variant_result("identity", result_path)
            mesh.write_bytes(b"tampered mesh")

            result = load_variant_result("identity", result_path)
            family = load_result_family(
                AugmentationViserConfig(
                    qpos_npz=result_path,
                    variants=("identity",),
                    object_urdf=override,
                )
            )

            self.assertEqual(result.variant, "identity")
            self.assertEqual(len(family), 1)

    def test_family_and_comparison_ignore_result_identity_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            identity_path = root / "motion.npz"
            variant_path = root / "motion_rot_0.npz"
            identity_payload = _result_payload(variant="identity")
            variant_payload = _result_payload(variant="rot_0")
            variant_payload["source_path"] = np.asarray("/dataset/other-sequence.npz")
            variant_payload["dataset_partition"] = np.asarray("other-partition")
            variant_payload["sequence_key"] = np.asarray("other-sequence")
            _refresh_config_identity(variant_payload)
            write_loose_result(identity_path, identity_payload)
            write_loose_result(variant_path, variant_payload)

            family = load_result_family(
                AugmentationViserConfig(
                    qpos_npz=identity_path,
                    variants=("identity", "rot_0"),
                )
            )
            _, comparison = load_comparison_results(
                MultiResultViserConfig(
                    qpos_npzs=(identity_path, variant_path),
                )
            )

        self.assertEqual([result.variant for result in family], ["identity", "rot_0"])
        self.assertEqual(len(comparison), 2)

    def test_family_loads_each_saved_actuated_order_independently(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            identity_path = root / "motion.npz"
            variant_path = root / "motion_rot_0.npz"
            payloads = (
                _result_payload(variant="identity"),
                _result_payload(variant="rot_0"),
            )
            for payload, names, path in zip(
                payloads,
                (("joint_a", "joint_b"), ("joint_b", "joint_a")),
                (identity_path, variant_path),
                strict=True,
            ):
                qpos = np.zeros((2, 9), dtype=np.float64)
                qpos[:, 3] = 1.0
                payload["qpos"] = qpos
                payload["robot_actuated_joint_names"] = np.asarray(names)
                write_loose_result(path, payload)

            family = load_result_family(
                AugmentationViserConfig(
                    qpos_npz=identity_path,
                    variants=("identity", "rot_0"),
                )
            )
            _, comparison = load_comparison_results(MultiResultViserConfig(qpos_npzs=(identity_path, variant_path)))

        self.assertEqual(family[0].robot_actuated_joint_names, ("joint_a", "joint_b"))
        self.assertEqual(family[1].robot_actuated_joint_names, ("joint_b", "joint_a"))
        self.assertEqual(len(comparison), 2)

    def test_saved_joint_order_drives_single_and_multi_mapping(self):
        saved_indices = resolve_qpos_to_viser_joint_indices(
            result_path="result.npz",
            saved_joint_names=("joint_b", "joint_a"),
            viser_joint_names=("joint_a", "joint_b"),
            fallback_mujoco_xml=None,
        )
        np.testing.assert_array_equal(saved_indices, (1, 0))

        config = ViserConfig(
            qpos_npz="strict.npz",
            robot_urdf="robot.urdf",
        )
        single_indices = _build_qpos_to_viser_joint_indices(
            config,
            ["joint_a", "joint_b"],
            {
                "robot_actuated_joint_names": [
                    "joint_b",
                    "joint_a",
                ],
            },
        )
        np.testing.assert_array_equal(single_indices, (1, 0))

        with self.assertRaisesRegex(ValueError, "missing_from_result"):
            resolve_qpos_to_viser_joint_indices(
                result_path="result.npz",
                saved_joint_names=("joint_a",),
                viser_joint_names=("joint_a", "joint_b"),
                fallback_mujoco_xml=None,
            )

    def test_packaged_robot_joint_orders_map_by_saved_names(self):
        mappings: dict[str, np.ndarray | None] = {}
        for robot_type in ("g1", "e1"):
            robot_urdf = _resolve_asset_path(RobotConfig(robot_type=robot_type).ROBOT_URDF_FILE)
            robot_xml = robot_urdf.with_suffix(".xml")
            saved_names = tuple(actuated_joint_names_from_mujoco_xml(robot_xml))
            viser_names = tuple(actuated_joint_names_from_urdf(robot_urdf))
            mapping = resolve_qpos_to_viser_joint_indices(
                result_path=f"{robot_type}.npz",
                saved_joint_names=saved_names,
                viser_joint_names=viser_names,
                fallback_mujoco_xml=None,
            )
            mappings[robot_type] = mapping
            reordered = saved_names if mapping is None else tuple(saved_names[int(index)] for index in mapping)
            self.assertEqual(reordered, viser_names)

        self.assertIsNone(mappings["g1"])
        self.assertIsNotNone(mappings["e1"])

    def test_joint_order_falls_back_to_mujoco_xml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            xml_path = Path(tmpdir) / "robot.xml"
            xml_path.write_text(
                '<mujoco><worldbody><body><joint name="joint_b"/><joint name="joint_a"/></body></worldbody></mujoco>',
                encoding="utf-8",
            )
            recovered = resolve_qpos_to_viser_joint_indices(
                result_path="result.npz",
                saved_joint_names=(),
                viser_joint_names=("joint_a", "joint_b"),
                fallback_mujoco_xml=xml_path,
            )

        np.testing.assert_array_equal(recovered, (1, 0))
        with self.assertRaisesRegex(
            FileNotFoundError,
            "no robot_actuated_joint_names",
        ):
            resolve_qpos_to_viser_joint_indices(
                result_path="result.npz",
                saved_joint_names=(),
                viser_joint_names=("joint_a", "joint_b"),
                fallback_mujoco_xml=None,
            )

    def test_schema_metadata_does_not_gate_a_comparison(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            strict_path = root / "strict.npz"
            legacy_path = root / "legacy.npz"
            payload = _result_payload()
            payload["schema_version"] = np.asarray(999, dtype=np.int32)
            write_loose_result(strict_path, payload)
            legacy_payload = copy.deepcopy(payload)
            legacy_payload.pop("schema_version")
            np.savez_compressed(legacy_path, **legacy_payload)

            _, results = load_comparison_results(MultiResultViserConfig(qpos_npzs=(strict_path, legacy_path)))

        self.assertEqual(len(results), 2)

    def test_source_orientation_is_never_fabricated_from_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "motion.npz"
            payload = _result_payload(
                include_source_orientations=False,
                include_diagnostics=True,
            )
            payload.pop("schema_version", None)
            np.savez_compressed(path, **payload)
            result = load_variant_result("identity", path)

        self.assertEqual(result.human_orientation_joint_names, ())
        self.assertIsNone(result.human_orientation_quaternions_wxyz)
        self.assertIsNotNone(result.orientation_diagnostics)

        server = SimpleNamespace(scene=_FakeScene())
        overlays = _build_orientation_overlay(
            ViserConfig(
                show_source_orientation_axes=True,
                show_target_orientation_axes=True,
                show_robot_orientation_axes=True,
            ),
            server,
            result.human_joints,
            variant_result_metadata(result),
            result.qpos.shape[0],
        )

        self.assertIsNone(overlays.source)
        self.assertIsNotNone(overlays.target)
        self.assertIsNotNone(overlays.robot)
        self.assertEqual(overlays.robot.names, result.robot_link_names)

    def test_unprovenanced_legacy_dense_orientations_are_not_displayed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_payload = {
                "qpos": np.zeros((2, 7), dtype=np.float32),
                "human_joints": np.zeros((2, 2, 3), dtype=np.float32),
                "human_joint_names": np.asarray(("Hips", "Head")),
                "mapped_human_joint_names": np.asarray(("Hips",)),
                "mapped_robot_joints": np.zeros((2, 1, 3), dtype=np.float32),
                "mapped_robot_link_names": np.asarray(("base",)),
                "global_joint_quaternions_wxyz": _identity_quaternions(2, 2),
            }
            synthetic_path = root / "synthetic.npz"
            np.savez(
                synthetic_path,
                **base_payload,
                orientation_provenance=np.asarray("synthesized_from_positions"),
            )
            direct_path = root / "direct.npz"
            np.savez(
                direct_path,
                **base_payload,
                orientation_provenance=np.asarray("direct_source_bvh_fk"),
            )

            synthetic = load_variant_result("synthetic", synthetic_path)
            direct = load_variant_result("direct", direct_path)

        self.assertIsNone(synthetic.human_orientation_quaternions_wxyz)
        self.assertEqual(direct.human_orientation_joint_names, ("Hips", "Head"))
        self.assertEqual(
            direct.human_orientation_quaternions_wxyz.shape,
            (2, 2, 4),
        )

    def test_explicit_source_subset_and_compact_robot_axes_are_rendered(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "motion.npz"
            write_loose_result(path, _result_payload())
            result = load_variant_result("identity", path)

        server = SimpleNamespace(scene=_FakeScene())
        overlays = _build_orientation_overlay(
            ViserConfig(
                show_source_orientation_axes=True,
                show_robot_orientation_axes=True,
            ),
            server,
            result.human_joints,
            variant_result_metadata(result),
            result.qpos.shape[0],
        )

        self.assertEqual(overlays.source.names, ("Hips", "LeftHand"))
        self.assertEqual(overlays.robot.names, ("base", "left_hand"))
        np.testing.assert_array_equal(
            overlays.robot.positions,
            result.robot_link_positions,
        )

    def test_orientation_scope_switches_between_retargeting_and_all_saved_links(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "motion.npz"
            payload = _result_payload()
            payload["human_joints"] = np.zeros((2, 3, 3), dtype=np.float32)
            payload["human_joint_names"] = np.asarray(("Hips", "Spine", "LeftHand"))
            payload["human_joint_parent_indices"] = np.asarray((-1, 0, 1), dtype=np.int32)
            payload["human_orientation_joint_names"] = np.asarray(("Hips", "Spine", "LeftHand"))
            payload["human_orientation_quaternions_wxyz"] = _identity_quaternions(2, 3)
            payload["robot_link_positions"] = np.zeros((2, 3, 3), dtype=np.float32)
            payload["robot_link_quaternions_wxyz"] = _identity_quaternions(2, 3)
            payload["robot_link_names"] = np.asarray(("base", "waist", "left_hand"))
            payload["robot_link_parent_indices"] = np.asarray((-1, 0, 1), dtype=np.int32)
            write_loose_result(path, payload)
            result = load_variant_result("identity", path)

        overlays = _build_orientation_overlay(
            ViserConfig(
                show_source_orientation_axes=True,
                show_robot_orientation_axes=True,
                orientation_scope="retargeting",
            ),
            SimpleNamespace(scene=_FakeScene()),
            result.human_joints,
            variant_result_metadata(result),
            result.qpos.shape[0],
        )

        self.assertEqual(overlays.source.names, ("Hips", "LeftHand"))
        self.assertEqual(overlays.robot.names, ("base", "left_hand"))
        overlays.set_scope("all")
        self.assertEqual(overlays.source.names, ("Hips", "Spine", "LeftHand"))
        self.assertEqual(overlays.robot.names, ("base", "waist", "left_hand"))
        self.assertTrue(overlays.source.enabled)
        self.assertTrue(overlays.robot.enabled)

    def test_full_saved_robot_skeleton_does_not_require_mujoco_xml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "motion.npz"
            write_loose_result(path, _result_payload())
            result = load_variant_result("identity", path)

        overlay = _build_mapped_skeleton_overlay(
            ViserConfig(),
            SimpleNamespace(scene=_FakeScene()),
            result.human_joints,
            variant_result_metadata(result),
        )

        self.assertIsNotNone(overlay)
        self.assertIsNone(overlay.robot_model)
        np.testing.assert_array_equal(
            overlay.robot_skeleton_joints,
            result.robot_link_positions,
        )

    def test_mapped_robot_skeleton_uses_semantic_edges_before_compact_body_tree(self):
        joint_names = [
            "Pelvis",
            "L_Knee",
            "L_Ankle",
            "L_Foot",
            "L_Shoulder",
            "R_Shoulder",
        ]
        mapped_positions = np.arange(18, dtype=np.float32).reshape(1, 6, 3)
        compact_positions = np.arange(21, dtype=np.float32).reshape(1, 7, 3)
        overlay = MappedSkeletonOverlay(
            server=SimpleNamespace(scene=_FakeScene()),
            human_joints=np.zeros((1, 6, 3), dtype=np.float32),
            demo_joints=joint_names,
            joints_mapping={name: f"robot_{name.lower()}" for name in joint_names},
            robot_xml_path=None,
            point_radius=0.02,
            line_width=2.0,
            mapped_robot_joints=mapped_positions,
            robot_skeleton_joints=compact_positions,
            robot_skeleton_parent_indices=np.asarray(
                (-1, 0, 1, 1, 0, 0, 0),
                dtype=np.int32,
            ),
        )

        self.assertEqual(
            overlay.robot_edges,
            [(0, 1), (1, 2), (1, 3), (0, 4), (0, 5), (0, 6)],
        )
        np.testing.assert_array_equal(
            overlay._robot_points(np.empty(0), 0.0),
            compact_positions[0],
        )


if __name__ == "__main__":
    unittest.main()
