# ruff: noqa: CPY001, E402, PT009, PT027, SIM117
from __future__ import annotations

import copy
import hashlib
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
    make_multi_result_player,
    resolve_multi_config,
)
from holosoma_retargeting.result_artifact import (
    RESULT_SCHEMA_VERSION,
    compute_human_orientation_sha256,
    write_result_artifact,
)
from holosoma_retargeting.src.viser_utils import (
    actuated_joint_names_from_mujoco_xml,
    actuated_joint_names_from_urdf,
)
from holosoma_retargeting.viser_player import (
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
from test_result_artifact import (
    _set_real_object_asset_manifest,
    _valid_payload,
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
    retargeter_config = solver_config.setdefault("retargeter", {})
    payload["frame_zero_ground_retry_eligible"] = np.asarray(
        bool(
            retargeter_config.get(
                "retry_frame_zero_ground_on_infeasible",
                False,
            )
        )
        and solver_config["task_type"] == "robot_only"
        and decoded["run_kind"] in {"single", "ablation"}
        and 7 + int(retargeter_config.get("q_a_init_idx", 0)) <= 2
    )
    config_json = json.dumps(
        decoded,
        separators=(",", ":"),
        sort_keys=True,
    )
    payload["config_json"] = np.asarray(config_json)
    payload["config_sha256"] = np.asarray(hashlib.sha256(config_json.encode("utf-8")).hexdigest())


def _refresh_human_orientation_identity(
    payload: dict[str, object],
) -> None:
    if "human_orientation_joint_names" not in payload or "human_orientation_quaternions_wxyz" not in payload:
        payload.pop("human_orientation_sha256", None)
        return
    payload["human_orientation_sha256"] = np.asarray(
        compute_human_orientation_sha256(
            np.asarray(payload["human_orientation_joint_names"]),
            np.asarray(
                payload["human_orientation_quaternions_wxyz"],
                dtype=np.float32,
            ),
        )
    )


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
    payload = _valid_payload()
    payload.update(
        {
            "variant": np.asarray(variant),
            "dataset_partition": np.asarray("gvhmr"),
            "sequence_key": np.asarray("sequence"),
            "experiment_name": np.asarray(""),
            "source_path": np.asarray("/dataset/sequence.npz"),
            "source_data_format": np.asarray("gvhmr"),
            "robot_type": np.asarray("g1"),
            "task_type": np.asarray("robot_only"),
            "object_name": np.asarray("ground"),
            "object_urdf": np.asarray(""),
            "qpos": qpos,
            "human_joints": human_joints,
            "human_joint_names": np.asarray(("Hips", "Spine", "LeftHand")),
            "human_joint_parent_indices": np.asarray(
                (-1, 0, 1),
                dtype=np.int32,
            ),
            "mapped_human_joints": human_joints[:, (0, 2)],
            "mapped_human_joint_names": np.asarray(("Hips", "LeftHand")),
            "mapped_robot_joints": robot_link_positions[:, (0, 2)],
            "mapped_robot_link_names": np.asarray(("base", "left_hand")),
            "robot_link_positions": robot_link_positions,
            "robot_link_quaternions_wxyz": _identity_quaternions(2, 3),
            "robot_link_names": np.asarray(("base", "torso", "left_hand")),
            "robot_link_parent_indices": np.asarray(
                (-1, 0, 1),
                dtype=np.int32,
            ),
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
        }
    )
    if include_source_orientations:
        payload.update(
            {
                "human_orientation_joint_names": np.asarray(("Hips", "LeftHand")),
                "human_orientation_quaternions_wxyz": (_identity_quaternions(2, 2)),
                "orientation_source": np.asarray("direct_local_rotation_fk"),
            }
        )
        _refresh_human_orientation_identity(payload)
    else:
        payload.pop("human_orientation_joint_names", None)
        payload.pop("human_orientation_quaternions_wxyz", None)
        payload.pop("human_orientation_sha256", None)
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

    def test_canonical_directory_and_legacy_flat_families_are_discovered(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            canonical = root / "sequence"
            canonical.mkdir()
            (canonical / "identity.npz").touch()
            (canonical / "rot_0.npz").touch()

            from_directory = discover_variant_paths(
                canonical,
                variants=("identity", "rot_0"),
            )
            from_identity = discover_variant_paths(
                canonical / "identity.npz",
                variants=("identity", "rot_0"),
            )
            from_sibling = discover_variant_paths(
                canonical / "rot_0.npz",
                variants=("identity", "rot_0"),
            )

            (root / "clip_original.npz").touch()
            (root / "clip_rot_0.npz").touch()
            legacy = discover_variant_paths(
                root / "clip_rot_0.npz",
                variants=("identity", "rot_0"),
            )

        self.assertEqual(from_directory, from_identity)
        self.assertEqual(from_directory, from_sibling)
        self.assertEqual(from_directory["identity"].name, "identity.npz")
        self.assertEqual(legacy["identity"].name, "clip_original.npz")

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
                payload["object_non_penetration_eligible_for_saved_trajectory"] = np.asarray(True)
                _set_real_object_asset_manifest(payload, object_urdf)
                _refresh_config_identity(payload)
                write_result_artifact(root / f"{variant}.npz", payload)

            resolved = resolve_multi_config(MultiViserConfig(family=root))
            labels, results = load_comparison_results(resolved)

        self.assertEqual(
            tuple(path.name for path in resolved.qpos_npzs),
            ("identity.npz", "z_scale_1p2.npz"),
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
                root / "first" / "identity.npz",
                root / "second" / "identity.npz",
            )
            for path in paths:
                path.parent.mkdir()
                write_result_artifact(path, _result_payload())

            labels, _results = load_comparison_results(
                MultiResultViserConfig(qpos_npzs=paths),
            )

        self.assertEqual(labels, ("single:identity", "single:identity [2]"))

    def test_comparison_rejects_symlink_aliases_of_the_same_result(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            result_path = root / "identity.npz"
            alias_path = root / "identity_alias.npz"
            write_result_artifact(result_path, _result_payload())
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

    def test_strict_family_rejects_requested_and_saved_variant_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_result_artifact(
                root / "identity.npz",
                _result_payload(variant="identity"),
            )
            write_result_artifact(
                root / "rot_0.npz",
                _result_payload(variant="trans_0"),
            )
            family_config = AugmentationViserConfig(
                qpos_npz=root,
                variants=("identity", "rot_0"),
            )
            multi_config = MultiViserConfig(
                family=root,
                variants=("identity", "rot_0"),
            )

            with self.assertRaisesRegex(
                ValueError,
                "filename must be 'trans_0.npz'",
            ):
                resolve_multi_config(MultiViserConfig(family=root))
            with self.assertRaisesRegex(
                ValueError,
                "family variant identity mismatch",
            ):
                load_result_family(family_config)
            with self.assertRaisesRegex(
                ValueError,
                "family variant identity mismatch",
            ):
                make_multi_result_player(multi_config)

    def test_strict_family_accepts_original_as_identity_alias(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_result_artifact(
                root / "identity.npz",
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
            path = Path(tmpdir) / "identity.npz"
            payload = _result_payload()
            foot_states = np.asarray(payload["foot_sticking_states"]).copy()
            foot_states[0, 0] = True
            payload["foot_sticking_states"] = foot_states
            payload["constraint_mode_foot_sticking"] = np.asarray(
                ("normal", "inactive"),
            )
            payload["constraint_mode_trust_region_released"] = np.asarray(
                (True, False),
            )
            payload["foot_sticking_violation"] = np.asarray(
                (5e-7, 0.0),
                dtype=np.float64,
            )
            payload["interaction_mesh_edges_default"] = np.asarray("all")
            write_result_artifact(path, payload)

            result = load_variant_result("requested-label", path)
            qpos, fps, human_joints, metadata, interaction_mesh = load_npz(str(path))
            labels, comparison_results = load_comparison_results(
                MultiResultViserConfig(
                    qpos_npzs=(path,),
                    labels=("baseline",),
                )
            )

        self.assertEqual(result.schema_version, RESULT_SCHEMA_VERSION)
        self.assertEqual(result.variant, "identity")
        self.assertEqual(result.source_path, "/dataset/sequence.npz")
        self.assertEqual(result.source_sha256, "a" * 64)
        self.assertIsNotNone(result.config_json)
        self.assertRegex(result.config_sha256 or "", r"^[0-9a-f]{64}$")
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
        self.assertIsNotNone(result.human_orientation_sha256)
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
        np.testing.assert_array_equal(
            result.constraint_mode_foot_sticking,
            np.asarray(("normal", "inactive")),
        )
        np.testing.assert_array_equal(
            result.constraint_mode_object_non_penetration_released,
            np.asarray((False, False)),
        )
        np.testing.assert_array_equal(
            result.constraint_mode_trust_region_released,
            np.asarray((True, False)),
        )
        np.testing.assert_array_equal(
            result.foot_sticking_violation,
            np.asarray((5e-7, 0.0), dtype=np.float64),
        )
        for residual in (
            result.ground_non_penetration_violation,
            result.object_non_penetration_violation,
            result.foot_lock_violation,
            result.self_collision_violation,
            result.joint_limits_violation,
        ):
            np.testing.assert_array_equal(
                residual,
                np.zeros(2, dtype=np.float64),
            )
        np.testing.assert_array_equal(
            result.object_non_penetration_release_frames,
            np.empty(0, dtype=np.int32),
        )
        self.assertFalse(
            result.release_object_non_penetration_on_infeasible,
        )
        self.assertFalse(
            result.object_non_penetration_eligible_for_saved_trajectory,
        )
        self.assertEqual(
            result.frame_zero_ground_retry,
            {
                "policy": "robot_only_frame_zero_horizontal_ground_lift_v1",
                "eligible": True,
                "triggered": False,
                "initial_min_distance_m": 0.0,
                "corrected_min_distance_m": 0.0,
                "lift_m": 0.0,
                "interior_margin_m": 0.0,
                "initial_sqp_iterations": 0,
            },
        )
        self.assertEqual(qpos.shape, (2, 7))
        self.assertEqual(human_joints.shape, (2, 2, 3))
        self.assertEqual(fps, 30.0)
        self.assertEqual(metadata["schema_version"], RESULT_SCHEMA_VERSION)
        self.assertEqual(metadata["source_sha256"], "a" * 64)
        self.assertEqual(metadata["config_sha256"], result.config_sha256)
        self.assertEqual(metadata["dataset_partition"], "gvhmr")
        self.assertEqual(metadata["sequence_key"], "sequence")
        self.assertEqual(
            metadata["orientation_source"],
            "direct_local_rotation_fk",
        )
        self.assertEqual(metadata["robot_actuated_joint_names"], [])
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
        np.testing.assert_array_equal(
            metadata["constraint_mode_foot_sticking"],
            np.asarray(("normal", "inactive")),
        )
        np.testing.assert_array_equal(
            metadata["constraint_mode_trust_region_released"],
            np.asarray((True, False)),
        )
        np.testing.assert_array_equal(
            metadata["foot_sticking_violation"],
            np.asarray((5e-7, 0.0), dtype=np.float64),
        )
        np.testing.assert_array_equal(
            metadata["object_non_penetration_release_frames"],
            np.empty(0, dtype=np.int32),
        )
        self.assertEqual(metadata["foot_sticking"]["tolerance"], 1e-3)
        self.assertEqual(
            metadata["foot_sticking"]["fallback_tolerance"],
            0.02,
        )
        self.assertTrue(metadata["foot_sticking"]["release_on_infeasible"])
        self.assertEqual(
            metadata["foot_sticking"]["full_sequence_retry_frame"],
            -1,
        )
        self.assertEqual(
            metadata["frame_zero_ground_retry"],
            result.frame_zero_ground_retry,
        )
        self.assertIsNotNone(interaction_mesh)
        self.assertEqual(labels, ("baseline",))
        self.assertEqual(comparison_results[0].schema_version, RESULT_SCHEMA_VERSION)
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
            payload["object_non_penetration_eligible_for_saved_trajectory"] = np.asarray(
                True,
            )
            _set_real_object_asset_manifest(payload, object_urdf)
            _refresh_config_identity(payload)
            result_path = root / "identity.npz"

            write_result_artifact(result_path, payload)
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
        self.assertEqual(
            result.object_urdf_sha256,
            str(np.asarray(payload["object_urdf_sha256"]).item()),
        )
        self.assertEqual(
            result.object_asset_manifest_json,
            str(np.asarray(payload["object_asset_manifest_json"]).item()),
        )
        self.assertEqual(
            result.object_asset_manifest_sha256,
            str(np.asarray(payload["object_asset_manifest_sha256"]).item()),
        )
        for field in (
            "object_poses_demo",
            "object_poses_target",
            "object_pose_layout",
            "object_urdf_sha256",
            "object_asset_manifest_json",
            "object_asset_manifest_sha256",
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
                root / "identity.npz",
                root / "z_scale_1.2.npz",
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
                        "object_non_penetration_eligible_for_saved_trajectory": np.asarray(
                            True,
                        ),
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
                write_result_artifact(result_path, payload)
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
        self.assertNotEqual(
            family[0].config_sha256,
            family[1].config_sha256,
        )
        self.assertNotEqual(family[0].object_urdf, family[1].object_urdf)

    def test_strict_loader_validates_saved_asset_closure_before_honoring_override(self):
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
            result_path = root / "identity.npz"
            payload = _result_payload()
            payload["task_type"] = np.asarray("climbing")
            payload["object_name"] = np.asarray("multi_boxes")
            payload["object_non_penetration_eligible_for_saved_trajectory"] = np.asarray(
                True,
            )
            _set_real_object_asset_manifest(payload, object_urdf)
            _refresh_config_identity(payload)
            write_result_artifact(result_path, payload)

            load_variant_result("identity", result_path)
            mesh.write_bytes(b"tampered mesh")

            with self.assertRaisesRegex(
                ValueError,
                "dependency closure differs",
            ):
                load_variant_result("identity", result_path)
            with self.assertRaisesRegex(
                ValueError,
                "dependency closure differs",
            ):
                load_result_family(
                    AugmentationViserConfig(
                        qpos_npz=result_path,
                        variants=("identity",),
                        object_urdf=override,
                    )
                )

    def test_strict_family_and_comparison_reject_identity_mismatches(self):
        def change_human_names(payload: dict[str, object]) -> None:
            human_joints = np.asarray(payload["human_joints"]).copy()
            payload["human_joints"] = human_joints[:, (2, 0, 1)]
            payload["human_joint_names"] = np.asarray(("LeftHand", "Hips", "Spine"))
            payload["human_joint_parent_indices"] = np.asarray(
                (-1, 0, 1),
                dtype=np.int32,
            )

        def change_orientation_names(payload: dict[str, object]) -> None:
            payload["human_orientation_joint_names"] = np.asarray(("LeftHand", "Hips"))
            payload["human_orientation_quaternions_wxyz"] = np.asarray(payload["human_orientation_quaternions_wxyz"])[
                :, ::-1
            ].copy()
            _refresh_human_orientation_identity(payload)

        def change_orientation_tensor(payload: dict[str, object]) -> None:
            quaternions = np.asarray(payload["human_orientation_quaternions_wxyz"]).copy()
            quaternions[0, 0] = (0.0, 0.0, 0.0, 1.0)
            payload["human_orientation_quaternions_wxyz"] = quaternions
            _refresh_human_orientation_identity(payload)

        def change_robot_topology_names(payload: dict[str, object]) -> None:
            positions = np.asarray(payload["robot_link_positions"]).copy()
            quaternions = np.asarray(
                payload["robot_link_quaternions_wxyz"],
            ).copy()
            payload["robot_link_positions"] = positions[:, (2, 0, 1)]
            payload["robot_link_quaternions_wxyz"] = quaternions[:, (2, 0, 1)]
            payload["robot_link_names"] = np.asarray(("left_hand", "base", "torso"))
            payload["robot_link_parent_indices"] = np.asarray(
                (-1, 0, 1),
                dtype=np.int32,
            )

        def change_task_type(payload: dict[str, object]) -> None:
            payload["task_type"] = np.asarray("climbing")
            payload["object_name"] = np.asarray("multi_boxes")
            payload["object_non_penetration_eligible_for_saved_trajectory"] = np.asarray(
                True,
            )

        mutations = (
            (
                "source_path",
                lambda payload: payload.__setitem__(
                    "source_path",
                    np.asarray("/dataset/other-sequence.npz"),
                ),
            ),
            (
                "source_sha256",
                lambda payload: payload.__setitem__(
                    "source_sha256",
                    np.asarray("b" * 64),
                ),
            ),
            ("human_joint_names", change_human_names),
            (
                "human_orientation_joint_names",
                change_orientation_names,
            ),
            (
                "human_orientation_(sha256|quaternions_wxyz)",
                change_orientation_tensor,
            ),
            ("robot_link_names", change_robot_topology_names),
            (
                "source_data_format",
                lambda payload: payload.__setitem__(
                    "source_data_format",
                    np.asarray("amass"),
                ),
            ),
            ("task_type", change_task_type),
            (
                "dataset_partition",
                lambda payload: payload.__setitem__(
                    "dataset_partition",
                    np.asarray("other-partition"),
                ),
            ),
            (
                "sequence_key",
                lambda payload: payload.__setitem__(
                    "sequence_key",
                    np.asarray("other-sequence"),
                ),
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for index, (expected_field, mutate) in enumerate(mutations):
                case_root = root / f"case_{index}"
                case_root.mkdir()
                identity_path = case_root / "identity.npz"
                variant_path = case_root / "rot_0.npz"
                identity_payload = _result_payload(variant="identity")
                variant_payload = _result_payload(variant="rot_0")
                mutate(variant_payload)
                if expected_field == "task_type":
                    object_urdf = case_root / "multi_boxes.urdf"
                    object_urdf.write_text(
                        "<robot name='object'><link name='object'/></robot>",
                        encoding="utf-8",
                    )
                    _set_real_object_asset_manifest(
                        variant_payload,
                        object_urdf,
                    )
                _refresh_config_identity(variant_payload)
                write_result_artifact(identity_path, identity_payload)
                write_result_artifact(variant_path, variant_payload)

                with self.subTest(
                    field=expected_field,
                    entry_point="family",
                ):
                    with self.assertRaisesRegex(
                        ValueError,
                        expected_field,
                    ):
                        load_result_family(
                            AugmentationViserConfig(
                                qpos_npz=identity_path,
                                variants=("identity", "rot_0"),
                            )
                        )
                with self.subTest(
                    field=expected_field,
                    entry_point="comparison",
                ):
                    with self.assertRaisesRegex(
                        ValueError,
                        expected_field,
                    ):
                        load_comparison_results(
                            MultiResultViserConfig(
                                qpos_npzs=(
                                    identity_path,
                                    variant_path,
                                )
                            )
                        )

    def test_strict_family_rejects_saved_actuated_order_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            identity_path = root / "identity.npz"
            variant_path = root / "rot_0.npz"
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
                write_result_artifact(path, payload)

            with self.assertRaisesRegex(
                ValueError,
                "robot_actuated_joint_names",
            ):
                load_result_family(
                    AugmentationViserConfig(
                        qpos_npz=identity_path,
                        variants=("identity", "rot_0"),
                    )
                )
            with self.assertRaisesRegex(
                ValueError,
                "robot_actuated_joint_names",
            ):
                load_comparison_results(MultiResultViserConfig(qpos_npzs=(identity_path, variant_path)))

    def test_saved_joint_order_drives_strict_single_and_multi_mapping(self):
        strict_indices = resolve_qpos_to_viser_joint_indices(
            result_path="strict.npz",
            schema_version=RESULT_SCHEMA_VERSION,
            saved_joint_names=("joint_b", "joint_a"),
            viser_joint_names=("joint_a", "joint_b"),
            legacy_mujoco_xml=None,
        )
        np.testing.assert_array_equal(strict_indices, (1, 0))

        config = ViserConfig(
            qpos_npz="strict.npz",
            robot_urdf="robot.urdf",
        )
        single_indices = _build_qpos_to_viser_joint_indices(
            config,
            ["joint_a", "joint_b"],
            {
                "schema_version": RESULT_SCHEMA_VERSION,
                "robot_actuated_joint_names": [
                    "joint_b",
                    "joint_a",
                ],
            },
        )
        np.testing.assert_array_equal(single_indices, (1, 0))

        with self.assertRaisesRegex(ValueError, "missing_from_result"):
            resolve_qpos_to_viser_joint_indices(
                result_path="strict.npz",
                schema_version=RESULT_SCHEMA_VERSION,
                saved_joint_names=("joint_a",),
                viser_joint_names=("joint_a", "joint_b"),
                legacy_mujoco_xml=None,
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
                schema_version=RESULT_SCHEMA_VERSION,
                saved_joint_names=saved_names,
                viser_joint_names=viser_names,
                legacy_mujoco_xml=None,
            )
            mappings[robot_type] = mapping
            reordered = saved_names if mapping is None else tuple(saved_names[int(index)] for index in mapping)
            self.assertEqual(reordered, viser_names)

        self.assertIsNone(mappings["g1"])
        self.assertIsNotNone(mappings["e1"])

    def test_legacy_joint_order_recovery_is_explicit_and_not_strict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            xml_path = Path(tmpdir) / "robot.xml"
            xml_path.write_text(
                '<mujoco><worldbody><body><joint name="joint_b"/><joint name="joint_a"/></body></worldbody></mujoco>',
                encoding="utf-8",
            )
            recovered = resolve_qpos_to_viser_joint_indices(
                result_path="legacy.npz",
                schema_version=None,
                saved_joint_names=(),
                viser_joint_names=("joint_a", "joint_b"),
                legacy_mujoco_xml=xml_path,
            )

        np.testing.assert_array_equal(recovered, (1, 0))
        with self.assertRaisesRegex(
            FileNotFoundError,
            "legacy result.*robot-mujoco-xml",
        ):
            resolve_qpos_to_viser_joint_indices(
                result_path="legacy.npz",
                schema_version=None,
                saved_joint_names=(),
                viser_joint_names=("joint_a", "joint_b"),
                legacy_mujoco_xml=None,
            )

    def test_strict_and_legacy_results_cannot_share_a_comparison(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            strict_path = root / "strict.npz"
            legacy_path = root / "legacy.npz"
            payload = _result_payload()
            write_result_artifact(strict_path, payload)
            legacy_payload = copy.deepcopy(payload)
            legacy_payload.pop("schema_version")
            np.savez_compressed(legacy_path, **legacy_payload)

            with self.assertRaisesRegex(
                ValueError,
                "strict and legacy",
            ):
                load_comparison_results(MultiResultViserConfig(qpos_npzs=(strict_path, legacy_path)))

    def test_source_orientation_is_never_fabricated_from_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "identity.npz"
            payload = _result_payload(
                include_source_orientations=False,
                include_diagnostics=True,
            )
            payload.pop("schema_version")
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
            path = Path(tmpdir) / "identity.npz"
            write_result_artifact(path, _result_payload())
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

    def test_full_saved_robot_skeleton_does_not_require_mujoco_xml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "identity.npz"
            write_result_artifact(path, _result_payload())
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


if __name__ == "__main__":
    unittest.main()
