# ruff: noqa: CPY001, PT009, PT027

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "holosoma_retargeting"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from holosoma_retargeting.result_artifact import (  # noqa: E402
    FRAME_ZERO_GROUND_RETRY_POLICY,
    RESULT_SCHEMA_VERSION,
    ResultArtifactValidationError,
    build_object_asset_manifest,
    collision_interior_margin_m,
    compute_human_orientation_sha256,
    is_legacy_result_artifact,
    read_result_schema_version,
    validate_result_artifact,
    validate_result_external_assets,
    write_result_artifact,
)


def _identity_quaternions(frames: int, count: int) -> np.ndarray:
    quaternions = np.zeros((frames, count, 4), dtype=np.float32)
    quaternions[..., 0] = 1.0
    return quaternions


def _z_rotation_quaternions(angles_rad: np.ndarray) -> np.ndarray:
    angles = np.asarray(angles_rad, dtype=np.float32)
    quaternions = np.zeros((*angles.shape, 4), dtype=np.float32)
    quaternions[..., 0] = np.cos(angles / np.float32(2.0))
    quaternions[..., 3] = np.sin(angles / np.float32(2.0))
    return quaternions


def _valid_payload() -> dict[str, object]:
    frames = 2
    human_joints = np.asarray(
        [
            [[0.0, 0.0, 1.0], [0.0, 0.1, 0.5], [0.0, -0.1, 0.0]],
            [[0.1, 0.0, 1.0], [0.1, 0.1, 0.5], [0.1, -0.1, 0.0]],
        ],
        dtype=np.float32,
    )
    robot_link_positions = np.asarray(
        [
            [[0.0, 0.0, 0.9], [0.0, 0.1, 0.4], [0.0, -0.1, 0.0]],
            [[0.1, 0.0, 0.9], [0.1, 0.1, 0.4], [0.1, -0.1, 0.0]],
        ],
        dtype=np.float32,
    )
    config_json = json.dumps(
        {
            "config": {
                "data_format": "noetix_mocap",
                "robot_config": {"robot_type": "e1"},
                "task_config": {"object_name": "ground"},
                "task_type": "robot_only",
                "retargeter": {
                    "activate_obj_non_penetration": True,
                    "penetration_tolerance": 0.001,
                    "q_a_init_idx": -7,
                    "retry_frame_zero_ground_on_infeasible": True,
                    "sqp_max_iterations": 50,
                },
            },
            "dataset_partition": "noetix_mocap",
            "experiment_name": None,
            "run_kind": "single",
            "sequence_key": "walk",
            "variant": {
                "name": "identity",
                "translation": [0.0, 0.0, 0.0],
                "rotation": 0.0,
                "object_scale": [1.0, 1.0, 1.0],
            },
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    qpos = np.zeros((frames, 7), dtype=np.float64)
    qpos[:, 3] = 1.0
    object_poses = np.zeros((frames, 7), dtype=np.float32)
    object_poses[:, 3] = 1.0
    frame_costs = np.asarray([1.0, 0.5], dtype=np.float64)
    payload: dict[str, object] = {
        "schema_version": np.int32(RESULT_SCHEMA_VERSION),
        "run_kind": np.asarray("single"),
        "variant": np.asarray("identity"),
        "dataset_partition": np.asarray("noetix_mocap"),
        "sequence_key": np.asarray("walk"),
        "experiment_name": np.asarray(""),
        "source_path": np.asarray("demo_data/noetix_mocap/walk.npz"),
        "source_sha256": np.asarray("a" * 64),
        "config_json": np.asarray(config_json),
        "config_sha256": np.asarray(hashlib.sha256(config_json.encode("utf-8")).hexdigest()),
        "source_data_format": np.asarray("noetix_mocap"),
        "robot_type": np.asarray("e1"),
        "task_type": np.asarray("robot_only"),
        "object_name": np.asarray("ground"),
        "object_urdf": np.asarray(""),
        "object_urdf_sha256": np.asarray(""),
        "object_asset_manifest_json": np.asarray(""),
        "object_asset_manifest_sha256": np.asarray(""),
        "qpos": qpos,
        "qpos_layout": np.asarray("mujoco_free_root_xyz_wxyz_then_actuated_then_optional_object_free_joint"),
        "human_joints": human_joints,
        "human_joint_names": np.asarray(["Hips", "LeftFoot", "RightFoot"]),
        "human_joint_parent_indices": np.asarray([-1, 0, 0], dtype=np.int32),
        "mapped_human_joints": human_joints[:, [0, 2]],
        "mapped_human_joint_names": np.asarray(["Hips", "RightFoot"]),
        "mapped_robot_joints": robot_link_positions[:, [0, 2]],
        "mapped_robot_link_names": np.asarray(["pelvis_link", "right_foot_link"]),
        "robot_link_positions": robot_link_positions,
        "robot_link_quaternions_wxyz": _identity_quaternions(frames, 3),
        "robot_link_names": np.asarray(["pelvis_link", "right_knee_link", "right_foot_link"]),
        "robot_link_parent_indices": np.asarray([-1, 0, 1], dtype=np.int32),
        "robot_actuated_joint_names": np.asarray([], dtype=str),
        "contains_object_in_qpos": np.asarray(False),
        "object_poses_demo": object_poses,
        "object_poses_target": object_poses.copy(),
        "object_pose_layout": np.asarray("xyz_wxyz"),
        "quaternion_convention": np.asarray("wxyz"),
        "world_coordinate_system": np.asarray("right_handed_z_up"),
        "orientation_source": np.asarray("absent"),
        "source_human_height": np.float64(1.78),
        "human_position_scale": np.float64(1.0),
        "human_position_preprocessing": np.asarray("ground_align_and_uniform_scale"),
        "foot_sticking_side_names": np.asarray(["left", "right"]),
        "foot_sticking_states": np.zeros((frames, 2), dtype=bool),
        "foot_sticking_tolerance": np.float64(1e-3),
        "foot_sticking_fallback_tolerance": np.float64(0.02),
        "foot_sticking_fallback_frames": np.asarray([], dtype=np.int32),
        "release_foot_sticking_on_infeasible": np.asarray(True),
        "foot_sticking_release_frames": np.asarray([], dtype=np.int32),
        "release_object_non_penetration_on_infeasible": np.asarray(False),
        "object_non_penetration_release_frames": np.asarray([], dtype=np.int32),
        "object_non_penetration_eligible_for_saved_trajectory": np.asarray(
            False,
        ),
        "foot_sticking_enabled_for_saved_trajectory": np.asarray(True),
        "foot_sticking_full_sequence_retry_frame": np.int32(-1),
        "frame_zero_ground_retry_policy": np.asarray(
            FRAME_ZERO_GROUND_RETRY_POLICY,
        ),
        "frame_zero_ground_retry_eligible": np.asarray(True),
        "frame_zero_ground_retry_triggered": np.asarray(False),
        "frame_zero_ground_retry_initial_min_distance_m": np.float64(0.0),
        "frame_zero_ground_retry_corrected_min_distance_m": np.float64(0.0),
        "frame_zero_ground_retry_lift_m": np.float64(0.0),
        "frame_zero_ground_retry_interior_margin_m": np.float64(0.0),
        "frame_zero_ground_retry_initial_sqp_iterations": np.int32(0),
        "frame_costs": frame_costs,
        "sqp_iteration_counts": np.asarray([4, 5], dtype=np.int32),
        "sqp_stop_reasons": np.asarray(["converged", "converged"]),
        "ground_non_penetration_violation": np.zeros(
            frames,
            dtype=np.float64,
        ),
        "object_non_penetration_violation": np.zeros(
            frames,
            dtype=np.float64,
        ),
        "foot_sticking_violation": np.zeros(frames, dtype=np.float64),
        "foot_lock_violation": np.zeros(frames, dtype=np.float64),
        "self_collision_violation": np.zeros(frames, dtype=np.float64),
        "joint_limits_violation": np.zeros(frames, dtype=np.float64),
        "constraint_mode_foot_sticking": np.asarray(
            ["inactive"] * frames,
        ),
        "constraint_mode_object_non_penetration_released": np.zeros(
            frames,
            dtype=bool,
        ),
        "constraint_mode_trust_region_released": np.zeros(
            frames,
            dtype=bool,
        ),
        "cost": frame_costs[-1],
        "orientation_tracking_enabled": np.asarray(False),
        "orientation_diagnostics_enabled": np.asarray(False),
        "orientation_human_joint_names": np.asarray([], dtype=str),
        "orientation_robot_link_names": np.asarray([], dtype=str),
        "orientation_weights": np.empty((0,), dtype=np.float64),
        "orientation_alignment_mode": np.asarray("t_pose"),
        "orientation_alignment_quaternions_wxyz": np.empty(
            (0, 4),
            dtype=np.float32,
        ),
        "orientation_reference_human_quaternions_wxyz": np.empty(
            (0, 4),
            dtype=np.float32,
        ),
        "orientation_reference_robot_quaternions_wxyz": np.empty(
            (0, 4),
            dtype=np.float32,
        ),
        "orientation_reference_robot_qpos": np.empty((0,), dtype=np.float32),
        "orientation_target_quaternions_wxyz": np.empty(
            (frames, 0, 4),
            dtype=np.float32,
        ),
        "orientation_robot_quaternions_wxyz": np.empty(
            (frames, 0, 4),
            dtype=np.float32,
        ),
        "orientation_errors_rad": np.empty(
            (frames, 0),
            dtype=np.float32,
        ),
        "orientation_frame_costs": np.zeros(frames, dtype=np.float64),
        "fps": np.float64(30.0),
    }
    return _with_interaction_mesh(payload)


def _with_human_orientations(payload: dict[str, object]) -> dict[str, object]:
    result = copy.deepcopy(payload)
    names = np.asarray(["Hips", "RightFoot"])
    quaternions = _identity_quaternions(2, 2)
    result["human_orientation_joint_names"] = names
    result["human_orientation_quaternions_wxyz"] = quaternions
    result["human_orientation_sha256"] = np.asarray(compute_human_orientation_sha256(names, quaternions))
    result["orientation_source"] = np.asarray("bvh_rotation_channels_fk")
    return result


def _with_interaction_mesh(payload: dict[str, object]) -> dict[str, object]:
    result = copy.deepcopy(payload)
    mapped_human = np.asarray(result["mapped_human_joints"])
    mapped_robot = np.asarray(result["mapped_robot_joints"])
    human_count = mapped_human.shape[1]
    source_objects = np.asarray(
        [
            [
                [0.5, 0.0, 0.0],
                [0.5, 0.5, 0.0],
                [0.5, 0.0, 0.5],
            ],
            [
                [0.6, 0.0, 0.0],
                [0.6, 0.5, 0.0],
                [0.6, 0.0, 0.5],
            ],
        ],
        dtype=np.float32,
    )
    target_objects = source_objects + np.float32(0.2)
    result.update(
        {
            "interaction_source_vertices_w": np.concatenate(
                (mapped_human, source_objects),
                axis=1,
            ),
            "interaction_target_vertices_w": np.concatenate(
                (mapped_robot, target_objects),
                axis=1,
            ),
            "interaction_tetrahedra": np.asarray(
                [
                    [[0, 1, 2, 3], [-1, -1, -1, -1]],
                    [[0, 1, 2, 3], [0, 1, 2, 4]],
                ],
                dtype=np.int32,
            ),
            "interaction_tetrahedra_counts": np.asarray([1, 2], dtype=np.int32),
            "interaction_num_human_vertices": np.int32(human_count),
            "interaction_num_object_vertices": np.int32(source_objects.shape[1]),
            "interaction_mesh_edges_default": np.asarray("cross"),
        }
    )
    return result


def _replace_config(
    payload: dict[str, object],
    config: dict[str, object],
) -> None:
    config_json = json.dumps(
        config,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    payload["config_json"] = np.asarray(config_json)
    payload["config_sha256"] = np.asarray(hashlib.sha256(config_json.encode("utf-8")).hexdigest())


def _set_fake_object_asset_manifest(
    payload: dict[str, object],
    *,
    object_urdf: str,
    object_urdf_sha256: str,
) -> None:
    manifest_json = json.dumps(
        {
            "dependencies": [],
            "urdf": {
                "path": object_urdf,
                "sha256": object_urdf_sha256,
                "size": 1,
            },
            "version": 1,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    payload["object_urdf"] = np.asarray(object_urdf)
    payload["object_urdf_sha256"] = np.asarray(object_urdf_sha256)
    payload["object_asset_manifest_json"] = np.asarray(manifest_json)
    payload["object_asset_manifest_sha256"] = np.asarray(
        hashlib.sha256(manifest_json.encode("utf-8")).hexdigest(),
    )


def _set_real_object_asset_manifest(
    payload: dict[str, object],
    object_urdf: Path,
) -> None:
    manifest_json, manifest_sha256 = build_object_asset_manifest(object_urdf)
    manifest = json.loads(manifest_json)
    payload["object_urdf"] = np.asarray(str(object_urdf))
    payload["object_urdf_sha256"] = np.asarray(manifest["urdf"]["sha256"])
    payload["object_asset_manifest_json"] = np.asarray(manifest_json)
    payload["object_asset_manifest_sha256"] = np.asarray(manifest_sha256)


def _dynamic_object_payload() -> dict[str, object]:
    result = _valid_payload()
    result["task_type"] = np.asarray("object_interaction")
    result["object_name"] = np.asarray("suitcase")
    _set_fake_object_asset_manifest(
        result,
        object_urdf="/assets/suitcase.urdf",
        object_urdf_sha256="b" * 64,
    )
    result["contains_object_in_qpos"] = np.asarray(True)
    result["object_non_penetration_eligible_for_saved_trajectory"] = np.asarray(
        True,
    )
    result["frame_zero_ground_retry_eligible"] = np.asarray(False)

    object_poses_demo = np.asarray(result["object_poses_demo"]).copy()
    object_poses_demo[1, 0] = np.float32(0.1)
    object_poses_target = object_poses_demo.copy()
    object_poses_target[:, :3] += np.asarray(
        [0.2, 0.2, 0.2],
        dtype=np.float32,
    )
    result["object_poses_demo"] = object_poses_demo
    result["object_poses_target"] = object_poses_target

    qpos = np.zeros((2, 14), dtype=np.float64)
    qpos[:, 3] = 1.0
    qpos[:, -7:] = object_poses_target
    result["qpos"] = qpos

    demo_local = np.asarray(
        [
            [0.5, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
        ],
        dtype=np.float32,
    )
    target_local = demo_local.copy()
    result.update(
        {
            "object_points_demo_local": demo_local,
            "object_points_target_local": target_local,
            "object_points_demo_world": np.asarray(
                result["interaction_source_vertices_w"],
            )[:, 2:].copy(),
            "object_points_target_world": np.asarray(
                result["interaction_target_vertices_w"],
            )[:, 2:].copy(),
        }
    )

    config = json.loads(str(np.asarray(result["config_json"]).item()))
    config["config"]["task_type"] = "object_interaction"
    config["config"]["task_config"]["object_name"] = "suitcase"
    _replace_config(result, config)
    return result


def _climbing_payload() -> dict[str, object]:
    result = _valid_payload()
    result["task_type"] = np.asarray("climbing")
    result["object_name"] = np.asarray("multi_boxes")
    _set_fake_object_asset_manifest(
        result,
        object_urdf="/assets/multi_boxes.urdf",
        object_urdf_sha256="c" * 64,
    )
    result["object_non_penetration_eligible_for_saved_trajectory"] = np.asarray(
        True,
    )
    result["frame_zero_ground_retry_eligible"] = np.asarray(False)
    config = json.loads(str(np.asarray(result["config_json"]).item()))
    config["config"]["task_type"] = "climbing"
    config["config"]["task_config"]["object_name"] = "multi_boxes"
    _replace_config(result, config)
    return result


def _with_orientation_diagnostics(
    payload: dict[str, object],
) -> dict[str, object]:
    result = _with_human_orientations(payload)
    errors = np.asarray([0.25, 0.5], dtype=np.float32)
    robot_orientations = _z_rotation_quaternions(errors).reshape(2, 1, 4)
    full_robot_orientations = np.asarray(result["robot_link_quaternions_wxyz"]).copy()
    full_robot_orientations[:, 0] = robot_orientations[:, 0]
    result["robot_link_quaternions_wxyz"] = full_robot_orientations
    result.update(
        {
            "orientation_tracking_enabled": np.asarray(True),
            "orientation_diagnostics_enabled": np.asarray(True),
            "orientation_human_joint_names": np.asarray(["Hips"]),
            "orientation_robot_link_names": np.asarray(["pelvis_link"]),
            "orientation_weights": np.asarray([2.0], dtype=np.float64),
            "orientation_alignment_mode": np.asarray("t_pose"),
            "orientation_alignment_quaternions_wxyz": _identity_quaternions(
                1,
                1,
            ).reshape(1, 4),
            "orientation_reference_human_quaternions_wxyz": _identity_quaternions(
                1,
                1,
            ).reshape(1, 4),
            "orientation_reference_robot_quaternions_wxyz": _identity_quaternions(
                1,
                1,
            ).reshape(1, 4),
            "orientation_reference_robot_qpos": np.asarray(
                result["qpos"],
                dtype=np.float32,
            )[0].copy(),
            "orientation_target_quaternions_wxyz": _identity_quaternions(
                2,
                1,
            ),
            "orientation_robot_quaternions_wxyz": robot_orientations,
            "orientation_errors_rad": errors.reshape(2, 1),
            "orientation_frame_costs": np.asarray(
                [0.125, 0.5],
                dtype=np.float64,
            ),
        }
    )
    return result


class ResultArtifactTests(unittest.TestCase):
    def test_canonical_payload_requires_mesh_without_fabricating_orientations(self):
        self.assertEqual(RESULT_SCHEMA_VERSION, 2)
        payload = _valid_payload()

        validate_result_artifact(payload)

        self.assertNotIn("human_orientation_joint_names", payload)
        self.assertNotIn("human_orientation_quaternions_wxyz", payload)
        self.assertNotIn("human_orientation_sha256", payload)
        self.assertIn("interaction_source_vertices_w", payload)

    def test_frame_zero_ground_retry_is_fully_audited_and_cross_checked(self):
        payload = _valid_payload()
        initial_distance = -0.016500531
        margin = collision_interior_margin_m(0.001)
        lift = -initial_distance - 0.001 + margin
        payload.update(
            {
                "frame_zero_ground_retry_triggered": np.asarray(True),
                "frame_zero_ground_retry_initial_min_distance_m": np.float64(
                    initial_distance,
                ),
                "frame_zero_ground_retry_corrected_min_distance_m": np.float64(
                    initial_distance + lift,
                ),
                "frame_zero_ground_retry_lift_m": np.float64(lift),
                "frame_zero_ground_retry_interior_margin_m": np.float64(
                    margin,
                ),
                "frame_zero_ground_retry_initial_sqp_iterations": np.int32(50),
                "sqp_stop_reasons": np.asarray(
                    ["frame_zero_ground_retry:feasible_step_stable", "converged"],
                ),
            }
        )
        validate_result_artifact(payload)

        tampered = copy.deepcopy(payload)
        tampered["frame_zero_ground_retry_lift_m"] = np.float64(lift + 1e-4)
        with self.assertRaisesRegex(
            ResultArtifactValidationError,
            "one-step strict horizontal-ground correction",
        ):
            validate_result_artifact(tampered)

        contradictory = _valid_payload()
        contradictory["frame_zero_ground_retry_lift_m"] = np.float64(0.01)
        with self.assertRaisesRegex(
            ResultArtifactValidationError,
            "zero-valued numeric sentinels",
        ):
            validate_result_artifact(contradictory)

        ineligible = copy.deepcopy(payload)
        config = json.loads(str(np.asarray(ineligible["config_json"]).item()))
        config["config"]["retargeter"]["retry_frame_zero_ground_on_infeasible"] = False
        _replace_config(ineligible, config)
        with self.assertRaisesRegex(
            ResultArtifactValidationError,
            "eligible.*normalized",
        ):
            validate_result_artifact(ineligible)

        impossible_foot_mode = copy.deepcopy(payload)
        impossible_foot_mode["foot_sticking_states"][0, 0] = True
        impossible_foot_mode["constraint_mode_foot_sticking"][0] = "normal"
        with self.assertRaisesRegex(
            ResultArtifactValidationError,
            "inactive frame-zero foot-sticking mode",
        ):
            validate_result_artifact(impossible_foot_mode)

        impossible_iteration_count = copy.deepcopy(payload)
        impossible_iteration_count["frame_zero_ground_retry_initial_sqp_iterations"] = np.int32(2_147_483_647)
        with self.assertRaisesRegex(
            ResultArtifactValidationError,
            "must not exceed.*sqp_max_iterations",
        ):
            validate_result_artifact(impossible_iteration_count)

    def test_atomic_writer_uses_compression_and_round_trips_current_version(self):
        payload = _with_human_orientations(_with_interaction_mesh(_valid_payload()))
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nested" / "result.npz"
            returned_path = write_result_artifact(output_path, payload)

            self.assertEqual(returned_path, output_path)
            self.assertEqual(read_result_schema_version(output_path), RESULT_SCHEMA_VERSION)
            self.assertFalse(is_legacy_result_artifact(output_path))
            with np.load(output_path, allow_pickle=False) as saved:
                self.assertEqual(set(saved.files), set(payload))
                np.testing.assert_array_equal(saved["qpos"], payload["qpos"])
            with zipfile.ZipFile(output_path) as archive:
                self.assertTrue(archive.infolist())
                self.assertTrue(all(member.compress_type == zipfile.ZIP_DEFLATED for member in archive.infolist()))

    def test_writer_verifies_dynamic_asset_closure_and_round_trips(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            mesh = root / "mesh.obj"
            mesh.write_bytes(b"verified mesh")
            object_urdf = root / "object.urdf"
            object_urdf.write_text(
                (
                    "<robot name='object'><link name='object'><visual><geometry>"
                    "<mesh filename='mesh.obj'/></geometry></visual></link></robot>"
                ),
                encoding="utf-8",
            )
            payload = _dynamic_object_payload()
            _set_real_object_asset_manifest(payload, object_urdf)
            output_path = root / "result.npz"

            returned_path = write_result_artifact(output_path, payload)
            with np.load(output_path, allow_pickle=False) as saved:
                validate_result_artifact(saved)
                observed_path, observed_manifest_sha256 = validate_result_external_assets(saved)

        self.assertEqual(returned_path, output_path)
        self.assertEqual(observed_path, object_urdf)
        self.assertEqual(
            observed_manifest_sha256,
            str(np.asarray(payload["object_asset_manifest_sha256"]).item()),
        )

    def test_writer_rejects_forged_dynamic_dependency_manifest_before_publish(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            mesh = root / "mesh.obj"
            mesh.write_bytes(b"real mesh")
            object_urdf = root / "object.urdf"
            object_urdf.write_text(
                (
                    "<robot name='object'><link name='object'><visual><geometry>"
                    "<mesh filename='mesh.obj'/></geometry></visual></link></robot>"
                ),
                encoding="utf-8",
            )
            payload = _dynamic_object_payload()
            _set_real_object_asset_manifest(payload, object_urdf)
            manifest = json.loads(
                str(np.asarray(payload["object_asset_manifest_json"]).item()),
            )
            manifest["dependencies"][0]["sha256"] = "0" * 64
            forged_manifest_json = json.dumps(
                manifest,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            payload["object_asset_manifest_json"] = np.asarray(
                forged_manifest_json,
            )
            payload["object_asset_manifest_sha256"] = np.asarray(
                hashlib.sha256(forged_manifest_json.encode("utf-8")).hexdigest(),
            )
            validate_result_artifact(payload)

            output_path = root / "result.npz"
            original_contents = b"existing result"
            output_path.write_bytes(original_contents)
            with self.assertRaisesRegex(
                ResultArtifactValidationError,
                "dependency closure differs",
            ):
                write_result_artifact(output_path, payload)

            self.assertEqual(output_path.read_bytes(), original_contents)
            self.assertEqual(
                list(root.glob(f".{output_path.name}.*.tmp.npz")),
                [],
            )

    def test_writer_failure_keeps_existing_result_and_cleans_temporary_file(self):
        payload = _valid_payload()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "result.npz"
            original_contents = b"existing result"
            output_path.write_bytes(original_contents)

            with mock.patch(
                "holosoma_retargeting.result_artifact.np.savez_compressed",
                side_effect=RuntimeError("simulated write failure"),
            ), self.assertRaisesRegex(RuntimeError, "simulated write failure"):
                write_result_artifact(output_path, payload)

            self.assertEqual(output_path.read_bytes(), original_contents)
            self.assertEqual(
                list(output_path.parent.glob(f".{output_path.name}.*.tmp.npz")),
                [],
            )

    def test_validation_failure_does_not_touch_existing_destination(self):
        payload = _valid_payload()
        del payload["robot_type"]
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "result.npz"
            original_contents = b"existing result"
            output_path.write_bytes(original_contents)

            with self.assertRaisesRegex(
                ResultArtifactValidationError,
                "robot_type",
            ):
                write_result_artifact(output_path, payload)

            self.assertEqual(output_path.read_bytes(), original_contents)

    def test_job_identity_and_asset_fields_are_required_and_cross_checked(self):
        for field in (
            "dataset_partition",
            "sequence_key",
            "experiment_name",
            "object_name",
            "object_urdf",
            "object_urdf_sha256",
            "object_asset_manifest_json",
            "object_asset_manifest_sha256",
        ):
            payload = _valid_payload()
            del payload[field]
            with self.subTest(missing=field), self.assertRaisesRegex(
                ResultArtifactValidationError,
                field,
            ):
                validate_result_artifact(payload)

        sequence_mismatch = _valid_payload()
        sequence_mismatch["sequence_key"] = np.asarray("different")
        with self.assertRaisesRegex(
            ResultArtifactValidationError,
            "sequence_key.*config_json",
        ):
            validate_result_artifact(sequence_mismatch)

        variant_mismatch = _valid_payload()
        variant_mismatch["variant"] = np.asarray("different")
        with self.assertRaisesRegex(
            ResultArtifactValidationError,
            "variant.*config_json",
        ):
            validate_result_artifact(variant_mismatch)

        ablation_without_experiment = _valid_payload()
        ablation_without_experiment["run_kind"] = np.asarray("ablation")
        config = json.loads(str(np.asarray(ablation_without_experiment["config_json"]).item()))
        config["run_kind"] = "ablation"
        _replace_config(ablation_without_experiment, config)
        with self.assertRaisesRegex(
            ResultArtifactValidationError,
            "experiment_name.*ablation",
        ):
            validate_result_artifact(ablation_without_experiment)

        invalid_partition = _valid_payload()
        invalid_partition["dataset_partition"] = np.asarray("bad/path")
        config = json.loads(str(np.asarray(invalid_partition["config_json"]).item()))
        config["dataset_partition"] = "bad/path"
        _replace_config(invalid_partition, config)
        with self.assertRaisesRegex(
            ResultArtifactValidationError,
            "one path component",
        ):
            validate_result_artifact(invalid_partition)

        ground_with_asset = _valid_payload()
        ground_with_asset["object_urdf"] = np.asarray("ground.urdf")
        with self.assertRaisesRegex(
            ResultArtifactValidationError,
            "canonical ground",
        ):
            validate_result_artifact(ground_with_asset)

        validate_result_artifact(_dynamic_object_payload())
        validate_result_artifact(_climbing_payload())

        climbing_without_asset = _climbing_payload()
        climbing_without_asset["object_urdf"] = np.asarray("")
        with self.assertRaisesRegex(
            ResultArtifactValidationError,
            "terrain asset",
        ):
            validate_result_artifact(climbing_without_asset)

    def test_object_asset_manifest_closes_relative_and_absolute_dependencies(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            relative_mesh = root / "mesh.obj"
            absolute_mesh = root / "absolute.obj"
            texture = root / "texture.png"
            relative_mesh.write_bytes(b"relative mesh")
            absolute_mesh.write_bytes(b"absolute mesh")
            texture.write_bytes(b"texture")
            object_urdf = root / "object.urdf"
            object_urdf.write_text(
                (
                    "<robot name='object'><link name='object'>"
                    "<visual><geometry><mesh filename='mesh.obj'/></geometry>"
                    f"<material><texture filename='{texture}'/></material></visual>"
                    "<collision><geometry>"
                    f"<mesh filename='{absolute_mesh}'/>"
                    "</geometry></collision></link></robot>"
                ),
                encoding="utf-8",
            )

            manifest_json, manifest_sha256 = build_object_asset_manifest(
                object_urdf,
            )
            manifest = json.loads(manifest_json)
            payload = _dynamic_object_payload()
            _set_real_object_asset_manifest(payload, object_urdf)

            validate_result_artifact(payload)
            observed_path, observed_sha256 = validate_result_external_assets(
                payload,
            )

        self.assertEqual(observed_path, object_urdf)
        self.assertEqual(observed_sha256, manifest_sha256)
        self.assertEqual(
            {(entry["kind"], entry["path"]) for entry in manifest["dependencies"]},
            {
                ("mesh", str(relative_mesh)),
                ("mesh", str(absolute_mesh)),
                ("texture", str(texture)),
            },
        )

    def test_external_asset_validation_fails_closed_for_urdf_and_mesh_changes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            mesh = root / "mesh.obj"
            mesh.write_bytes(b"mesh v1")
            object_urdf = root / "object.urdf"
            original_urdf = (
                "<robot name='object'><link name='object'><visual><geometry>"
                "<mesh filename='mesh.obj'/></geometry></visual></link></robot>"
            )
            object_urdf.write_text(original_urdf, encoding="utf-8")
            payload = _dynamic_object_payload()
            _set_real_object_asset_manifest(payload, object_urdf)
            validate_result_external_assets(payload)

            mesh.write_bytes(b"mesh v2")
            with self.assertRaisesRegex(
                ResultArtifactValidationError,
                "dependency closure differs",
            ):
                validate_result_external_assets(payload)

            mesh.write_bytes(b"mesh v1")
            object_urdf.write_text(
                original_urdf.replace("object", "changed", 1),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ResultArtifactValidationError,
                "dependency closure differs",
            ):
                validate_result_external_assets(payload)

            object_urdf.write_text(original_urdf, encoding="utf-8")
            tampered_payload = copy.deepcopy(payload)
            manifest = json.loads(
                str(
                    np.asarray(
                        tampered_payload["object_asset_manifest_json"],
                    ).item()
                )
            )
            manifest["dependencies"][0]["sha256"] = "0" * 64
            tampered_json = json.dumps(
                manifest,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            tampered_payload["object_asset_manifest_json"] = np.asarray(
                tampered_json,
            )
            tampered_payload["object_asset_manifest_sha256"] = np.asarray(
                hashlib.sha256(tampered_json.encode("utf-8")).hexdigest(),
            )
            validate_result_artifact(tampered_payload)
            with self.assertRaisesRegex(
                ResultArtifactValidationError,
                "dependency closure differs",
            ):
                validate_result_external_assets(tampered_payload)

            mesh.unlink()
            with self.assertRaisesRegex(
                ResultArtifactValidationError,
                "Cannot verify external object asset dependency closure",
            ):
                validate_result_external_assets(payload)

    def test_object_asset_manifest_tampering_and_unsafe_references_are_rejected(self):
        payload = _dynamic_object_payload()
        manifest = json.loads(
            str(np.asarray(payload["object_asset_manifest_json"]).item()),
        )
        manifest["urdf"]["size"] = 2
        tampered_json = json.dumps(
            manifest,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        payload["object_asset_manifest_json"] = np.asarray(tampered_json)
        with self.assertRaisesRegex(
            ResultArtifactValidationError,
            "object_asset_manifest_sha256",
        ):
            validate_result_artifact(payload)

        payload["object_asset_manifest_sha256"] = np.asarray(
            hashlib.sha256(tampered_json.encode("utf-8")).hexdigest(),
        )
        validate_result_artifact(payload)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            mesh = root / "mesh.obj"
            mesh.write_bytes(b"mesh")
            symlink = root / "mesh-link.obj"
            symlink.symlink_to(mesh)
            for case, filename in (
                ("traversal", "../mesh.obj"),
                ("URI", "package://object/mesh.obj"),
                ("symlink", "mesh-link.obj"),
            ):
                object_urdf = root / f"{case}.urdf"
                object_urdf.write_text(
                    (
                        "<robot name='object'><link name='object'><visual>"
                        f"<geometry><mesh filename='{filename}'/></geometry>"
                        "</visual></link></robot>"
                    ),
                    encoding="utf-8",
                )
                with self.subTest(case=case), self.assertRaises(
                    (FileNotFoundError, ValueError),
                ):
                    build_object_asset_manifest(object_urdf)

    def test_schema_rejects_unknown_fields_and_noncanonical_dtypes(self):
        unknown = _valid_payload()
        unknown["future_metadata"] = np.float32(1.0)
        with self.assertRaisesRegex(
            ResultArtifactValidationError,
            "outside schema",
        ):
            validate_result_artifact(unknown)

        cases: dict[str, tuple[str, object]] = {
            "float tensor": (
                "qpos",
                np.asarray(_valid_payload()["qpos"], dtype=np.float32),
            ),
            "integer tensor": (
                "robot_link_parent_indices",
                np.asarray(
                    _valid_payload()["robot_link_parent_indices"],
                    dtype=np.int64,
                ),
            ),
            "integer scalar": (
                "interaction_num_human_vertices",
                np.int64(2),
            ),
            "real scalar": ("fps", np.float32(30.0)),
            "Unicode scalar": ("robot_type", np.asarray(b"e1")),
            "boolean scalar": ("contains_object_in_qpos", np.int8(0)),
        }
        for case, (field, value) in cases.items():
            payload = _valid_payload()
            payload[field] = value
            with self.subTest(case=case), self.assertRaises(ResultArtifactValidationError):
                validate_result_artifact(payload)

    def test_required_shapes_names_finiteness_and_topology_are_strict(self):
        invalid_payloads: dict[str, dict[str, object]] = {}

        missing = _valid_payload()
        del missing["variant"]
        invalid_payloads["missing required field"] = missing

        empty_qpos = _valid_payload()
        empty_qpos["qpos"] = np.zeros((0, 7), dtype=np.float64)
        invalid_payloads["empty qpos"] = empty_qpos

        nonfinite = _valid_payload()
        nonfinite_human = np.asarray(nonfinite["human_joints"]).copy()
        nonfinite_human[0, 0, 0] = np.nan
        nonfinite["human_joints"] = nonfinite_human
        invalid_payloads["nonfinite human"] = nonfinite

        duplicate_names = _valid_payload()
        duplicate_names["human_joint_names"] = np.asarray(["Hips", "LeftFoot", "LeftFoot"])
        invalid_payloads["duplicate human names"] = duplicate_names

        mapped_mismatch = _valid_payload()
        mapped_positions = np.asarray(mapped_mismatch["mapped_human_joints"]).copy()
        mapped_positions[0, 0, 0] += 0.1
        mapped_mismatch["mapped_human_joints"] = mapped_positions
        invalid_payloads["mapped human mismatch"] = mapped_mismatch

        nonunit_robot = _valid_payload()
        robot_quaternions = np.asarray(nonunit_robot["robot_link_quaternions_wxyz"]).copy()
        robot_quaternions[0, 0, 0] = 2.0
        nonunit_robot["robot_link_quaternions_wxyz"] = robot_quaternions
        invalid_payloads["nonunit robot quaternion"] = nonunit_robot

        parent_cycle = _valid_payload()
        parent_cycle["robot_link_parent_indices"] = np.asarray([-1, 2, 1], dtype=np.int32)
        invalid_payloads["parent cycle"] = parent_cycle

        multiple_roots = _valid_payload()
        multiple_roots["robot_link_parent_indices"] = np.asarray([-1, -1, 1], dtype=np.int32)
        invalid_payloads["multiple roots"] = multiple_roots

        invalid_human_tree = _valid_payload()
        invalid_human_tree["human_joint_parent_indices"] = np.asarray(
            [-1, 2, 1],
            dtype=np.int32,
        )
        invalid_payloads["invalid human topology"] = invalid_human_tree

        invalid_fps = _valid_payload()
        invalid_fps["fps"] = np.float64(0.0)
        invalid_payloads["invalid fps"] = invalid_fps

        object_array = _valid_payload()
        object_array["arbitrary_metadata"] = np.asarray([{"unsafe": True}], dtype=object)
        invalid_payloads["object array"] = object_array

        for case, payload in invalid_payloads.items():
            with self.subTest(case=case), self.assertRaises(ResultArtifactValidationError):
                validate_result_artifact(payload)

    def test_required_hash_layout_preprocessing_and_object_contracts_are_strict(self):
        invalid_payloads: dict[str, dict[str, object]] = {}

        bad_config_digest = _valid_payload()
        bad_config_digest["config_sha256"] = np.asarray("0" * 64)
        invalid_payloads["config digest"] = bad_config_digest

        bad_source_digest = _valid_payload()
        bad_source_digest["source_sha256"] = np.asarray("A" * 64)
        invalid_payloads["source digest"] = bad_source_digest

        bad_qpos_layout = _valid_payload()
        bad_qpos_layout["qpos_layout"] = np.asarray("unknown")
        invalid_payloads["qpos layout"] = bad_qpos_layout

        bad_coordinate_system = _valid_payload()
        bad_coordinate_system["world_coordinate_system"] = np.asarray("y_up")
        invalid_payloads["coordinate convention"] = bad_coordinate_system

        bad_height = _valid_payload()
        bad_height["source_human_height"] = np.float64(0.0)
        invalid_payloads["source height"] = bad_height

        bad_object_pose = _valid_payload()
        object_poses = np.asarray(bad_object_pose["object_poses_target"]).copy()
        object_poses[0, 3:7] = 0.0
        bad_object_pose["object_poses_target"] = object_poses
        invalid_payloads["object quaternion"] = bad_object_pose

        dynamic_object_mismatch = _dynamic_object_payload()
        dynamic_qpos = np.asarray(dynamic_object_mismatch["qpos"]).copy()
        dynamic_qpos[0, -7] += np.float32(0.5)
        dynamic_object_mismatch["qpos"] = dynamic_qpos
        invalid_payloads["dynamic object qpos mismatch"] = dynamic_object_mismatch

        object_pose_frame_mismatch = _valid_payload()
        object_pose_frame_mismatch["object_poses_demo"] = np.asarray(object_pose_frame_mismatch["object_poses_demo"])[
            :1
        ]
        invalid_payloads["object pose frame mismatch"] = object_pose_frame_mismatch

        nonfinite_object_pose = _valid_payload()
        nonfinite_poses = np.asarray(nonfinite_object_pose["object_poses_demo"]).copy()
        nonfinite_poses[0, 0] = np.nan
        nonfinite_object_pose["object_poses_demo"] = nonfinite_poses
        invalid_payloads["nonfinite object pose"] = nonfinite_object_pose

        for case, payload in invalid_payloads.items():
            with self.subTest(case=case), self.assertRaises(ResultArtifactValidationError):
                validate_result_artifact(payload)

    def test_human_orientation_subset_is_optional_but_must_be_a_complete_real_subset(self):
        validate_result_artifact(_with_human_orientations(_valid_payload()))

        missing_quaternions = _valid_payload()
        missing_quaternions["human_orientation_joint_names"] = np.asarray(["Hips"])

        missing_digest = _with_human_orientations(_valid_payload())
        del missing_digest["human_orientation_sha256"]

        wrong_digest = _with_human_orientations(_valid_payload())
        wrong_digest["human_orientation_sha256"] = np.asarray("0" * 64)

        changed_tensor_bytes = _with_human_orientations(_valid_payload())
        changed_quaternions = np.asarray(changed_tensor_bytes["human_orientation_quaternions_wxyz"]).copy()
        changed_quaternions[0, 0] *= np.float32(-1.0)
        changed_tensor_bytes["human_orientation_quaternions_wxyz"] = changed_quaternions

        changed_names = _with_human_orientations(_valid_payload())
        changed_names["human_orientation_joint_names"] = np.asarray(["RightFoot", "Hips"])

        digest_without_orientations = _valid_payload()
        digest_without_orientations["human_orientation_sha256"] = np.asarray("0" * 64)

        unknown_name = _with_human_orientations(_valid_payload())
        unknown_name["human_orientation_joint_names"] = np.asarray(["Hips", "Unknown"])

        nonunit = _with_human_orientations(_valid_payload())
        quaternions = np.asarray(nonunit["human_orientation_quaternions_wxyz"]).copy()
        quaternions[0, 1] = 0.0
        nonunit["human_orientation_quaternions_wxyz"] = quaternions

        legacy_name = _valid_payload()
        legacy_name["global_joint_quaternions_wxyz"] = _identity_quaternions(2, 3)

        for case, payload in {
            "missing paired field": missing_quaternions,
            "missing digest": missing_digest,
            "wrong digest": wrong_digest,
            "changed tensor bytes": changed_tensor_bytes,
            "changed joint-name bytes": changed_names,
            "digest without orientation tensor": digest_without_orientations,
            "unknown source joint": unknown_name,
            "nonunit source quaternion": nonunit,
            "legacy field in current schema": legacy_name,
        }.items():
            with self.subTest(case=case), self.assertRaises(ResultArtifactValidationError):
                validate_result_artifact(payload)

    def test_human_orientation_digest_binds_names_shape_and_float32_bytes(self):
        names = np.asarray(["Hips", "RightFoot"])
        quaternions = _identity_quaternions(2, 2)
        digest = compute_human_orientation_sha256(names, quaternions)
        self.assertEqual(len(digest), 64)
        self.assertEqual(
            digest,
            compute_human_orientation_sha256(
                names.tolist(),
                quaternions.astype(np.float64),
            ),
        )
        self.assertNotEqual(
            digest,
            compute_human_orientation_sha256(
                names[::-1],
                quaternions,
            ),
        )
        changed = quaternions.copy()
        changed[0, 0] *= np.float32(-1.0)
        self.assertNotEqual(
            digest,
            compute_human_orientation_sha256(names, changed),
        )

    def test_orientation_provenance_and_diagnostics_are_cross_validated(self):
        validate_result_artifact(_with_orientation_diagnostics(_valid_payload()))

        missing_provenance = _with_human_orientations(_valid_payload())
        missing_provenance["orientation_source"] = np.asarray("absent")

        estimated_provenance = _with_human_orientations(_valid_payload())
        estimated_provenance["orientation_source"] = np.asarray("estimated_from_positions")

        unavailable_diagnostic_joint = _with_orientation_diagnostics(_valid_payload())
        unavailable_diagnostic_joint["human_orientation_joint_names"] = np.asarray(["RightFoot"])
        unavailable_quaternions = np.asarray(unavailable_diagnostic_joint["human_orientation_quaternions_wxyz"])[
            :, 1:2
        ].copy()
        unavailable_diagnostic_joint["human_orientation_quaternions_wxyz"] = unavailable_quaternions
        unavailable_diagnostic_joint["human_orientation_sha256"] = np.asarray(
            compute_human_orientation_sha256(
                unavailable_diagnostic_joint["human_orientation_joint_names"],
                unavailable_quaternions,
            )
        )

        wrong_diagnostic_cost = _with_orientation_diagnostics(_valid_payload())
        wrong_diagnostic_cost["orientation_frame_costs"] = np.zeros(
            2,
            dtype=np.float64,
        )

        wrong_target = _with_orientation_diagnostics(_valid_payload())
        wrong_target["orientation_target_quaternions_wxyz"] = _z_rotation_quaternions(
            np.asarray([0.1, 0.1], dtype=np.float32)
        ).reshape(2, 1, 4)

        wrong_robot_subset = _with_orientation_diagnostics(_valid_payload())
        wrong_robot_subset["orientation_robot_quaternions_wxyz"] = _identity_quaternions(2, 1)

        wrong_geodesic_error = _with_orientation_diagnostics(_valid_payload())
        wrong_geodesic_error["orientation_errors_rad"] = np.zeros(
            (2, 1),
            dtype=np.float32,
        )
        wrong_geodesic_error["orientation_frame_costs"] = np.zeros(
            2,
            dtype=np.float64,
        )

        wrong_reference_alignment = _with_orientation_diagnostics(_valid_payload())
        wrong_reference_alignment["orientation_reference_robot_quaternions_wxyz"] = _z_rotation_quaternions(
            np.asarray([0.2], dtype=np.float32)
        ).reshape(1, 4)

        wrong_first_frame_reference = _with_orientation_diagnostics(_valid_payload())
        wrong_first_frame_reference["orientation_alignment_mode"] = np.asarray("first_frame")
        reference_rotation = _z_rotation_quaternions(np.asarray([0.2], dtype=np.float32)).reshape(1, 4)
        wrong_first_frame_reference["orientation_reference_human_quaternions_wxyz"] = reference_rotation
        wrong_first_frame_reference["orientation_reference_robot_quaternions_wxyz"] = reference_rotation.copy()

        nonfinite_frame_cost = _valid_payload()
        nonfinite_frame_cost["frame_costs"] = np.asarray(
            [1.0, np.nan],
            dtype=np.float64,
        )

        for case, payload in {
            "missing direct provenance": missing_provenance,
            "estimated provenance": estimated_provenance,
            "unavailable diagnostic source joint": unavailable_diagnostic_joint,
            "wrong source-to-target alignment": wrong_target,
            "wrong full robot subset": wrong_robot_subset,
            "wrong geodesic error": wrong_geodesic_error,
            "wrong reference alignment": wrong_reference_alignment,
            "wrong first-frame reference": wrong_first_frame_reference,
            "wrong orientation diagnostic cost": wrong_diagnostic_cost,
            "nonfinite solver cost": nonfinite_frame_cost,
        }.items():
            with self.subTest(case=case), self.assertRaises(ResultArtifactValidationError):
                validate_result_artifact(payload)

    def test_orientation_targets_follow_object_rotation_without_mutating_source(self):
        payload = _with_orientation_diagnostics(_valid_payload())
        source_tensor = np.asarray(
            payload["human_orientation_quaternions_wxyz"],
        )
        source_bytes = source_tensor.tobytes(order="C")
        source_sha256 = str(
            np.asarray(payload["human_orientation_sha256"]).item(),
        )
        delta_angles = np.asarray([0.35, -0.6], dtype=np.float32)
        delta_quaternions = _z_rotation_quaternions(delta_angles)
        target_object_poses = np.asarray(payload["object_poses_target"]).copy()
        target_object_poses[:, 3:7] = delta_quaternions
        payload["object_poses_target"] = target_object_poses
        payload["orientation_target_quaternions_wxyz"] = delta_quaternions[:, None, :]
        payload["orientation_robot_quaternions_wxyz"] = delta_quaternions[:, None, :]
        full_robot_orientations = np.asarray(
            payload["robot_link_quaternions_wxyz"],
        ).copy()
        full_robot_orientations[:, 0] = delta_quaternions
        payload["robot_link_quaternions_wxyz"] = full_robot_orientations
        payload["orientation_errors_rad"] = np.zeros(
            (2, 1),
            dtype=np.float32,
        )
        payload["orientation_frame_costs"] = np.zeros(
            2,
            dtype=np.float64,
        )

        validate_result_artifact(payload)
        self.assertEqual(
            np.asarray(
                payload["human_orientation_quaternions_wxyz"],
            ).tobytes(order="C"),
            source_bytes,
        )
        self.assertEqual(
            str(np.asarray(payload["human_orientation_sha256"]).item()),
            source_sha256,
        )

        unrotated_target = copy.deepcopy(payload)
        unrotated_target["orientation_target_quaternions_wxyz"] = _identity_quaternions(2, 1)
        with self.assertRaisesRegex(
            ResultArtifactValidationError,
            "demo-to-target object rotation",
        ):
            validate_result_artifact(unrotated_target)

    def test_translation_only_object_delta_preserves_orientation_target(self):
        payload = _with_orientation_diagnostics(_valid_payload())
        target_object_poses = np.asarray(payload["object_poses_target"]).copy()
        target_object_poses[:, :3] += np.asarray(
            [0.2, -0.3, 0.7],
            dtype=np.float32,
        )
        payload["object_poses_target"] = target_object_poses

        validate_result_artifact(payload)

        rotated_target = copy.deepcopy(payload)
        rotated_target["orientation_target_quaternions_wxyz"] = _z_rotation_quaternions(
            np.asarray([0.2, 0.2], dtype=np.float32),
        ).reshape(2, 1, 4)
        with self.assertRaisesRegex(
            ResultArtifactValidationError,
            "demo-to-target object rotation",
        ):
            validate_result_artifact(rotated_target)

    def test_first_frame_reference_uses_target_world_source(self):
        payload = _with_orientation_diagnostics(_valid_payload())
        delta_angles = np.asarray([0.4, -0.2], dtype=np.float32)
        delta_quaternions = _z_rotation_quaternions(delta_angles)
        target_object_poses = np.asarray(payload["object_poses_target"]).copy()
        target_object_poses[:, 3:7] = delta_quaternions
        payload["object_poses_target"] = target_object_poses
        payload["orientation_alignment_mode"] = np.asarray("first_frame")
        payload["orientation_reference_human_quaternions_wxyz"] = delta_quaternions[:1].copy()
        payload["orientation_reference_robot_quaternions_wxyz"] = delta_quaternions[:1].copy()
        payload["orientation_target_quaternions_wxyz"] = delta_quaternions[:, None, :]
        payload["orientation_robot_quaternions_wxyz"] = delta_quaternions[:, None, :]
        full_robot_orientations = np.asarray(
            payload["robot_link_quaternions_wxyz"],
        ).copy()
        full_robot_orientations[:, 0] = delta_quaternions
        payload["robot_link_quaternions_wxyz"] = full_robot_orientations
        payload["orientation_errors_rad"] = np.zeros(
            (2, 1),
            dtype=np.float32,
        )
        payload["orientation_frame_costs"] = np.zeros(
            2,
            dtype=np.float64,
        )

        validate_result_artifact(payload)

        raw_first_frame_reference = copy.deepcopy(payload)
        raw_first_frame_reference["orientation_reference_human_quaternions_wxyz"] = _identity_quaternions(1, 1).reshape(
            1, 4
        )
        raw_first_frame_reference["orientation_reference_robot_quaternions_wxyz"] = _identity_quaternions(1, 1).reshape(
            1, 4
        )
        with self.assertRaisesRegex(
            ResultArtifactValidationError,
            "target-world first source frame",
        ):
            validate_result_artifact(raw_first_frame_reference)

    def test_object_non_penetration_fallback_frames_require_enabled_recording(self):
        payload = _dynamic_object_payload()
        payload["release_object_non_penetration_on_infeasible"] = np.asarray(True)
        payload["object_non_penetration_release_frames"] = np.asarray(
            [0],
            dtype=np.int32,
        )
        payload["constraint_mode_object_non_penetration_released"] = np.asarray(
            [True, False],
        )
        validate_result_artifact(payload)

        disabled = copy.deepcopy(payload)
        disabled["release_object_non_penetration_on_infeasible"] = np.asarray(False)
        with self.assertRaisesRegex(
            ResultArtifactValidationError,
            "object_non_penetration_release_frames",
        ):
            validate_result_artifact(disabled)

        invalid_frame = copy.deepcopy(payload)
        invalid_frame["object_non_penetration_release_frames"] = np.asarray(
            [2],
            dtype=np.int32,
        )
        with self.assertRaisesRegex(
            ResultArtifactValidationError,
            "valid frame indices",
        ):
            validate_result_artifact(invalid_frame)

    def test_constraint_modes_are_complete_and_match_legacy_release_histories(self):
        for field in (
            "ground_non_penetration_violation",
            "object_non_penetration_violation",
            "foot_sticking_violation",
            "foot_lock_violation",
            "self_collision_violation",
            "joint_limits_violation",
            "constraint_mode_foot_sticking",
            "constraint_mode_object_non_penetration_released",
            "constraint_mode_trust_region_released",
        ):
            missing = _valid_payload()
            del missing[field]
            with self.subTest(missing=field), self.assertRaisesRegex(
                ResultArtifactValidationError,
                field,
            ):
                validate_result_artifact(missing)

        normal = _valid_payload()
        states = np.asarray(normal["foot_sticking_states"]).copy()
        states[0, 0] = True
        normal["foot_sticking_states"] = states
        normal["constraint_mode_foot_sticking"] = np.asarray(
            ["normal", "inactive"],
        )
        validate_result_artifact(normal)

        relaxed = copy.deepcopy(normal)
        relaxed["constraint_mode_foot_sticking"] = np.asarray(
            ["relaxed", "inactive"],
        )
        relaxed["foot_sticking_fallback_frames"] = np.asarray(
            [0],
            dtype=np.int32,
        )
        validate_result_artifact(relaxed)

        released = copy.deepcopy(relaxed)
        released["constraint_mode_foot_sticking"] = np.asarray(
            ["released", "inactive"],
        )
        released["foot_sticking_release_frames"] = np.asarray(
            [0],
            dtype=np.int32,
        )
        validate_result_artifact(released)

        full_retry = _valid_payload()
        full_retry["foot_sticking_enabled_for_saved_trajectory"] = np.asarray(
            False,
        )
        full_retry["foot_sticking_full_sequence_retry_frame"] = np.int32(0)
        validate_result_artifact(full_retry)

        contradictory_retry = copy.deepcopy(relaxed)
        contradictory_retry["foot_sticking_enabled_for_saved_trajectory"] = np.asarray(
            False,
        )
        contradictory_retry["foot_sticking_full_sequence_retry_frame"] = np.int32(
            0,
        )
        with self.assertRaisesRegex(
            ResultArtifactValidationError,
            "constraint_mode_foot_sticking.*inactive exactly",
        ):
            validate_result_artifact(contradictory_retry)

        mismatched_object_history = _dynamic_object_payload()
        mismatched_object_history["release_object_non_penetration_on_infeasible"] = np.asarray(
            True,
        )
        mismatched_object_history["constraint_mode_object_non_penetration_released"] = np.asarray(
            [True, False],
        )
        with self.assertRaisesRegex(
            ResultArtifactValidationError,
            "object_non_penetration_release_frames.*exactly match",
        ):
            validate_result_artifact(mismatched_object_history)

        invalid_trust_region_frame = _valid_payload()
        invalid_trust_region_frame["constraint_mode_trust_region_released"] = np.asarray(
            [False, True],
        )
        with self.assertRaisesRegex(
            ResultArtifactValidationError,
            "only be true on the initial frame",
        ):
            validate_result_artifact(invalid_trust_region_frame)

    def test_non_penetration_residuals_require_finite_shapes_and_release_consistency(
        self,
    ):
        validate_result_artifact(_valid_payload())

        ground_penetration = _valid_payload()
        ground_penetration["ground_non_penetration_violation"] = np.asarray(
            [0.007, 0.0],
            dtype=np.float64,
        )
        with self.assertRaisesRegex(
            ResultArtifactValidationError,
            "ground_non_penetration_violation.*acceptance tolerance",
        ):
            validate_result_artifact(ground_penetration)

        object_penetration = _dynamic_object_payload()
        object_penetration["object_non_penetration_violation"] = np.asarray(
            [0.007, 0.0],
            dtype=np.float64,
        )
        with self.assertRaisesRegex(
            ResultArtifactValidationError,
            "ConstraintMode did not release",
        ):
            validate_result_artifact(object_penetration)

        object_penetration["release_object_non_penetration_on_infeasible"] = np.asarray(
            True,
        )
        object_penetration["object_non_penetration_release_frames"] = np.asarray(
            [0],
            dtype=np.int32,
        )
        object_penetration["constraint_mode_object_non_penetration_released"] = np.asarray(
            [True, False],
        )
        validate_result_artifact(object_penetration)

        for case, field, value in (
            (
                "wrong shape",
                "ground_non_penetration_violation",
                np.zeros(1, dtype=np.float64),
            ),
            (
                "nonfinite",
                "object_non_penetration_violation",
                np.asarray([0.0, np.nan], dtype=np.float64),
            ),
            (
                "negative",
                "object_non_penetration_violation",
                np.asarray([0.0, -1e-3], dtype=np.float64),
            ),
        ):
            payload = _valid_payload()
            payload[field] = value
            with self.subTest(case=case), self.assertRaises(
                ResultArtifactValidationError,
            ):
                validate_result_artifact(payload)

    def test_all_hard_nonlinear_residuals_use_the_accepted_constraint_mode(self):
        for field in (
            "foot_lock_violation",
            "self_collision_violation",
            "joint_limits_violation",
        ):
            payload = _valid_payload()
            payload[field] = np.asarray([0.007, 0.0], dtype=np.float64)
            with self.subTest(field=field), self.assertRaisesRegex(
                ResultArtifactValidationError,
                f"{field}.*acceptance tolerance",
            ):
                validate_result_artifact(payload)

        active_foot = _valid_payload()
        states = np.asarray(active_foot["foot_sticking_states"]).copy()
        states[0, 0] = True
        active_foot["foot_sticking_states"] = states
        active_foot["constraint_mode_foot_sticking"] = np.asarray(
            ["normal", "inactive"],
        )
        active_foot["foot_sticking_violation"] = np.asarray(
            [0.007, 0.0],
            dtype=np.float64,
        )
        with self.assertRaisesRegex(
            ResultArtifactValidationError,
            "foot_sticking_violation.*remained active",
        ):
            validate_result_artifact(active_foot)

        released_foot = copy.deepcopy(active_foot)
        released_foot["constraint_mode_foot_sticking"] = np.asarray(
            ["released", "inactive"],
        )
        released_foot["foot_sticking_fallback_frames"] = np.asarray(
            [0],
            dtype=np.int32,
        )
        released_foot["foot_sticking_release_frames"] = np.asarray(
            [0],
            dtype=np.int32,
        )
        validate_result_artifact(released_foot)

        for field, value in (
            (
                "foot_sticking_violation",
                np.asarray([0.0, np.nan], dtype=np.float64),
            ),
            (
                "foot_lock_violation",
                np.asarray([0.0, -1e-3], dtype=np.float64),
            ),
            (
                "self_collision_violation",
                np.zeros(1, dtype=np.float64),
            ),
            (
                "joint_limits_violation",
                np.zeros(2, dtype=np.float32),
            ),
        ):
            payload = _valid_payload()
            payload[field] = value
            with self.subTest(invalid=field), self.assertRaises(
                ResultArtifactValidationError,
            ):
                validate_result_artifact(payload)

    def test_dynamic_object_points_match_poses_and_interaction_mesh_suffixes(self):
        validate_result_artifact(_dynamic_object_payload())

        invalid_payloads: dict[str, dict[str, object]] = {}
        for field in (
            "object_points_demo_local",
            "object_points_target_local",
            "object_points_demo_world",
            "object_points_target_world",
        ):
            missing = _dynamic_object_payload()
            del missing[field]
            invalid_payloads[f"missing {field}"] = missing

        points_without_dynamic_object = _valid_payload()
        dynamic = _dynamic_object_payload()
        for field in (
            "object_points_demo_local",
            "object_points_target_local",
            "object_points_demo_world",
            "object_points_target_world",
        ):
            points_without_dynamic_object[field] = copy.deepcopy(dynamic[field])
        invalid_payloads["points without dynamic object"] = points_without_dynamic_object

        bad_local_shape = _dynamic_object_payload()
        bad_local_shape["object_points_demo_local"] = np.zeros(
            (3, 2),
            dtype=np.float32,
        )
        invalid_payloads["bad local shape"] = bad_local_shape

        bad_world_shape = _dynamic_object_payload()
        bad_world_shape["object_points_target_world"] = np.asarray(bad_world_shape["object_points_target_world"])[:1]
        invalid_payloads["bad world frame count"] = bad_world_shape

        nonfinite_points = _dynamic_object_payload()
        world_points = np.asarray(nonfinite_points["object_points_demo_world"]).copy()
        world_points[0, 0, 0] = np.nan
        nonfinite_points["object_points_demo_world"] = world_points
        invalid_payloads["nonfinite object point"] = nonfinite_points

        bad_demo_transform = _dynamic_object_payload()
        world_points = np.asarray(bad_demo_transform["object_points_demo_world"]).copy()
        world_points[0, 0, 0] += np.float32(0.1)
        bad_demo_transform["object_points_demo_world"] = world_points
        invalid_payloads["bad demo local-to-world transform"] = bad_demo_transform

        bad_target_transform = _dynamic_object_payload()
        world_points = np.asarray(bad_target_transform["object_points_target_world"]).copy()
        world_points[0, 0, 0] += np.float32(0.1)
        bad_target_transform["object_points_target_world"] = world_points
        invalid_payloads["bad target local-to-world transform"] = bad_target_transform

        bad_source_suffix = _dynamic_object_payload()
        vertices = np.asarray(bad_source_suffix["interaction_source_vertices_w"]).copy()
        vertices[0, 2, 0] += np.float32(0.1)
        bad_source_suffix["interaction_source_vertices_w"] = vertices
        invalid_payloads["bad source mesh object suffix"] = bad_source_suffix

        bad_target_suffix = _dynamic_object_payload()
        vertices = np.asarray(bad_target_suffix["interaction_target_vertices_w"]).copy()
        vertices[0, 2, 0] += np.float32(0.1)
        bad_target_suffix["interaction_target_vertices_w"] = vertices
        invalid_payloads["bad target mesh object suffix"] = bad_target_suffix

        for case, payload in invalid_payloads.items():
            with self.subTest(case=case), self.assertRaises(ResultArtifactValidationError):
                validate_result_artifact(payload)

    def test_interaction_mesh_group_validates_counts_indices_padding_and_prefixes(self):
        validate_result_artifact(_with_interaction_mesh(_valid_payload()))

        zero_tetrahedra = _valid_payload()
        zero_tetrahedra["interaction_source_vertices_w"] = np.asarray(
            zero_tetrahedra["mapped_human_joints"],
            dtype=np.float32,
        ).copy()
        zero_tetrahedra["interaction_target_vertices_w"] = np.asarray(
            zero_tetrahedra["mapped_robot_joints"],
            dtype=np.float32,
        ).copy()
        zero_tetrahedra["interaction_tetrahedra"] = np.empty(
            (2, 0, 4),
            dtype=np.int32,
        )
        zero_tetrahedra["interaction_tetrahedra_counts"] = np.zeros(
            2,
            dtype=np.int32,
        )
        zero_tetrahedra["interaction_num_object_vertices"] = np.int32(0)
        validate_result_artifact(zero_tetrahedra)

        missing_mesh_fields = []
        for field in (
            "interaction_source_vertices_w",
            "interaction_target_vertices_w",
            "interaction_tetrahedra",
            "interaction_tetrahedra_counts",
            "interaction_num_human_vertices",
            "interaction_num_object_vertices",
            "interaction_mesh_edges_default",
        ):
            partial = _with_interaction_mesh(_valid_payload())
            del partial[field]
            missing_mesh_fields.append((f"missing {field}", partial))

        count_too_large = _with_interaction_mesh(_valid_payload())
        count_too_large["interaction_tetrahedra_counts"] = np.asarray([1, 3], dtype=np.int32)

        negative_count = _with_interaction_mesh(_valid_payload())
        negative_count["interaction_tetrahedra_counts"] = np.asarray(
            [-1, 2],
            dtype=np.int32,
        )

        loose_width = _with_interaction_mesh(_valid_payload())
        loose_width["interaction_tetrahedra_counts"] = np.asarray([1, 1], dtype=np.int32)

        invalid_padding = _with_interaction_mesh(_valid_payload())
        tetrahedra = np.asarray(invalid_padding["interaction_tetrahedra"]).copy()
        tetrahedra[0, 1] = np.asarray([0, 1, 2, 3])
        invalid_padding["interaction_tetrahedra"] = tetrahedra

        out_of_range = _with_interaction_mesh(_valid_payload())
        tetrahedra = np.asarray(out_of_range["interaction_tetrahedra"]).copy()
        tetrahedra[1, 0, 3] = 5
        out_of_range["interaction_tetrahedra"] = tetrahedra

        repeated_vertex = _with_interaction_mesh(_valid_payload())
        tetrahedra = np.asarray(repeated_vertex["interaction_tetrahedra"]).copy()
        tetrahedra[1, 0] = np.asarray([0, 0, 2, 3])
        repeated_vertex["interaction_tetrahedra"] = tetrahedra

        duplicate_tetrahedron = _with_interaction_mesh(_valid_payload())
        tetrahedra = np.asarray(duplicate_tetrahedron["interaction_tetrahedra"]).copy()
        tetrahedra[1, 1] = np.asarray([3, 2, 1, 0], dtype=np.int32)
        duplicate_tetrahedron["interaction_tetrahedra"] = tetrahedra

        bad_vertex_count = _with_interaction_mesh(_valid_payload())
        bad_vertex_count["interaction_num_object_vertices"] = np.int32(1)

        bad_prefix = _with_interaction_mesh(_valid_payload())
        source_vertices = np.asarray(bad_prefix["interaction_source_vertices_w"]).copy()
        source_vertices[0, 0, 0] += 0.1
        bad_prefix["interaction_source_vertices_w"] = source_vertices

        bad_target_prefix = _with_interaction_mesh(_valid_payload())
        target_vertices = np.asarray(bad_target_prefix["interaction_target_vertices_w"]).copy()
        target_vertices[0, 0, 0] += np.float32(0.1)
        bad_target_prefix["interaction_target_vertices_w"] = target_vertices

        nonfinite_vertex = _with_interaction_mesh(_valid_payload())
        source_vertices = np.asarray(nonfinite_vertex["interaction_source_vertices_w"]).copy()
        source_vertices[0, 2, 0] = np.nan
        nonfinite_vertex["interaction_source_vertices_w"] = source_vertices

        wrong_frame_count = _with_interaction_mesh(_valid_payload())
        wrong_frame_count["interaction_target_vertices_w"] = np.asarray(
            wrong_frame_count["interaction_target_vertices_w"]
        )[:1]

        wrong_index_dtype = _with_interaction_mesh(_valid_payload())
        wrong_index_dtype["interaction_tetrahedra"] = np.asarray(
            wrong_index_dtype["interaction_tetrahedra"],
            dtype=np.int64,
        )

        bad_edge_mode = _with_interaction_mesh(_valid_payload())
        bad_edge_mode["interaction_mesh_edges_default"] = np.asarray("faces")

        invalid_payloads = {
            "count above width": count_too_large,
            "negative count": negative_count,
            "noncanonical padded width": loose_width,
            "invalid padding": invalid_padding,
            "active index out of range": out_of_range,
            "repeated tetrahedron vertex": repeated_vertex,
            "duplicate tetrahedron": duplicate_tetrahedron,
            "vertex count mismatch": bad_vertex_count,
            "mapped source prefix mismatch": bad_prefix,
            "mapped target prefix mismatch": bad_target_prefix,
            "nonfinite vertex": nonfinite_vertex,
            "wrong frame count": wrong_frame_count,
            "wrong tetrahedron dtype": wrong_index_dtype,
            "bad edge mode": bad_edge_mode,
        }
        invalid_payloads.update(missing_mesh_fields)
        for case, payload in invalid_payloads.items():
            with self.subTest(case=case), self.assertRaises(ResultArtifactValidationError):
                validate_result_artifact(payload)

    def test_schema_version_and_legacy_checks_are_lightweight_and_unambiguous(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            legacy_path = directory / "legacy.npz"
            unrelated_path = directory / "converted.npz"
            future_path = directory / "future.npz"
            malformed_path = directory / "malformed.npz"
            np.savez(
                legacy_path,
                qpos=np.zeros((1, 7), dtype=np.float32),
                global_joint_quaternions_wxyz=_identity_quaternions(1, 1),
            )
            np.savez(
                unrelated_path,
                global_joint_positions=np.zeros((1, 1, 3), dtype=np.float32),
            )
            np.savez(
                future_path,
                schema_version=np.int32(RESULT_SCHEMA_VERSION + 1),
                qpos=np.zeros((1, 7), dtype=np.float32),
            )
            np.savez(
                malformed_path,
                schema_version=np.asarray([RESULT_SCHEMA_VERSION], dtype=np.int32),
                qpos=np.zeros((1, 7), dtype=np.float32),
            )

            self.assertIsNone(read_result_schema_version(legacy_path))
            self.assertTrue(is_legacy_result_artifact(legacy_path))
            self.assertFalse(is_legacy_result_artifact(unrelated_path))
            self.assertEqual(
                read_result_schema_version(future_path),
                RESULT_SCHEMA_VERSION + 1,
            )
            self.assertFalse(is_legacy_result_artifact(future_path))
            with self.assertRaises(ResultArtifactValidationError):
                read_result_schema_version(malformed_path)


if __name__ == "__main__":
    unittest.main()
