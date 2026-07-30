# ruff: noqa: CPY001

from __future__ import annotations

import fcntl
import hashlib
import json
import multiprocessing
import os
import pickle
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "holosoma_retargeting"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import holosoma_retargeting.examples.rebuild_demo_results as rebuild  # noqa: E402
from holosoma_retargeting.config_types.data_type import (  # noqa: E402
    DEMO_JOINTS_REGISTRY,
)
from holosoma_retargeting.examples.rebuild_demo_results import (  # noqa: E402
    DATASET_SPEC_BY_ID,
    ArtifactPlanCollisionError,
    BatchPlan,
    DemoRebuildError,
    ExpectedArtifact,
    RebuildDemoResultsConfig,
    _direct_orientation_tensor_sha256,
    _full_matrix_selected,
    _parallel_config,
    build_rebuild_matrix,
    plan_expected_artifacts,
    promote_staging,
    run_rebuild,
    run_report_root,
    staging_root,
    validate_rebuild_directory,
)
from holosoma_retargeting.result_artifact import (  # noqa: E402
    RESULT_SCHEMA_VERSION,
    build_object_asset_manifest,
    compute_file_sha256,
    write_result_artifact,
)


class SimulatedHardCrash(BaseException):
    """Model a SIGKILL-like interruption outside ``except Exception``."""


def _touch_dataset_roots(data_root: Path) -> None:
    for name in (
        "OMOMO_new",
        "amass_smplx_processed",
        "gvhmr",
        "lafan",
        "noetix_mocap",
        "climb",
        "noetix_csv_climb",
    ):
        (data_root / name).mkdir(parents=True)


def _audit_plan_and_expected(
    tmp_path: Path,
    *,
    robot: str,
    dataset_id: str,
    include_augmentation_variants: bool = True,
) -> tuple[
    RebuildDemoResultsConfig,
    BatchPlan,
    tuple[ExpectedArtifact, ...],
    Path,
    Path,
]:
    spec = DATASET_SPEC_BY_ID[dataset_id]
    data_root = tmp_path / "demo_data"
    data_dir = data_root / str(spec.relative_data_dir)
    data_dir.mkdir(parents=True)
    source_path = data_dir / "sub1_tripod_001.pt"
    source_path.touch()
    results_root = tmp_path / "demo_results"
    run_id = "attempt-audit"
    stage = staging_root(results_root, run_id)
    report_root = run_report_root(results_root, run_id)
    plan = BatchPlan(
        robot=robot,
        spec=spec,
        data_dir=data_dir.resolve(),
    )
    cfg = RebuildDemoResultsConfig(
        data_root=data_root,
        results_root=results_root,
        run_id=run_id,
        robots=(robot,),
        datasets=(dataset_id,),
        include_augmentation_variants=include_augmentation_variants,
        omomo_preflight=False,
    )
    expected = tuple(
        ExpectedArtifact(
            output_path=(
                stage
                / "canonical"
                / robot
                / spec.task_type
                / spec.data_format
                / data_dir.name
                / source_path.stem
                / f"{variant}.npz"
            ).resolve(),
            source_path=source_path.resolve(),
            source_sha256="a" * 64,
            config_sha256="b" * 64,
            robot=robot,
            task_type=spec.task_type,
            data_format=spec.data_format,
            variant=variant,
            run_kind=("single" if variant == "identity" else "augmentation"),
            dataset_id=dataset_id,
            frame_count=1,
            human_joint_names=("Pelvis",),
            human_joint_parent_indices=(-1,),
            human_orientation_joint_names=None,
            human_orientation_source="absent",
            human_orientation_sha256=None,
            human_orientation_tensor_shape=None,
            human_orientation_tensor_bytes=None,
            robot_link_names=("pelvis_link",),
            robot_link_parent_indices=(-1,),
        )
        for variant in rebuild._expected_variant_names(cfg, plan)
    )
    return cfg, plan, expected, stage, report_root


def _write_attempt_batch_report(
    plan: BatchPlan,
    expected: tuple[ExpectedArtifact, ...],
    *,
    stage: Path,
    report_root: Path,
    failure_error: str | None,
    include_augmentation_variants: bool = True,
) -> Path:
    source_path = expected[0].source_path
    task_name = source_path.parent.name if plan.spec.task_type == "climbing" else source_path.stem
    if failure_error is None:
        results = [
            {
                "task_name": task_name,
                "object_name": "tripod",
                "generated_files": [str(item.output_path) for item in expected],
                "skipped_files": [],
                "elapsed_seconds": 1.0,
                "status": "completed",
            }
        ]
        failures: list[dict[str, str | None]] = []
    else:
        results = []
        failures = [
            {
                "source_path": str(source_path),
                "task_name": task_name,
                "object_name": "tripod",
                "error": failure_error,
            }
        ]
    report = {
        "task_type": plan.spec.task_type,
        "robot": plan.robot,
        "data_format": plan.spec.data_format,
        "data_dir": str(plan.data_dir),
        "save_dir": str(stage),
        "object_names": None,
        "total_files": 1,
        "per_object": {},
        "manifest": [
            {
                "source_path": str(source_path),
                "task_name": task_name,
                "object_name": "tripod",
            }
        ],
        "preflight": None,
        "rebuild_contract": rebuild._batch_rebuild_contract(
            outer_run_id=report_root.name,
            plan=plan,
            expected_items=expected,
            include_augmentation_variants=include_augmentation_variants,
        ),
        "status": ("completed_with_failures" if failure_error is not None else "completed"),
        "completed_tasks": int(failure_error is None),
        "skipped_tasks": 0,
        "failed_tasks": int(failure_error is not None),
        "results": results,
        "failures": failures,
    }
    report_path = report_root / "batches" / f"{plan.batch_id}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, sort_keys=True),
        encoding="utf-8",
    )
    return report_path


def _hold_rebuild_lock_until_released(
    lock_path: str,
    ready: Any,
    release: Any,
) -> None:
    descriptor = os.open(
        lock_path,
        os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        ready.set()
        if not release.wait(timeout=30.0):
            raise TimeoutError("Parent did not release the test rebuild lock")
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _write_adapter_npz(
    source_path: Path,
    *,
    data_format: str,
    frame_count: int = 2,
) -> None:
    joint_names = DEMO_JOINTS_REGISTRY[data_format]
    payload: dict[str, np.ndarray] = {
        "global_joint_positions": np.zeros(
            (frame_count, len(joint_names), 3),
            dtype=np.float32,
        ),
        "joint_names": np.asarray(joint_names),
        "height": np.asarray(1.7),
        "fps": np.asarray(30.0),
    }
    if data_format == "noetix_mocap":
        payload["source_format"] = np.asarray("noetix_mocap")
    np.savez(source_path, **payload)


def _write_omomo_source(
    source_path: Path,
    *,
    frame_count: int = 2,
) -> None:
    values = torch.zeros((frame_count, 591), dtype=torch.float32)
    orientations = values[:, 383:591].reshape(frame_count, 52, 4)
    orientations[..., 3] = 1.0
    torch.save(values, source_path)
    (source_path.parent.parent / "height_dict.pkl").write_bytes(pickle.dumps({"sub1": 1.75}))


def _write_valid_artifact(
    artifact_path: Path,
    *,
    source_path: Path,
    source_sha256: str,
    config_json: str,
    orientation: bool = True,
) -> ExpectedArtifact:
    config_json = json.dumps(
        {
            "config": {
                "data_format": "amass",
                "robot_config": {"robot_type": "g1"},
                "task_config": {"object_name": "ground"},
                "task_type": "robot_only",
                "retargeter": {
                    "activate_obj_non_penetration": True,
                    "penetration_tolerance": 0.001,
                    "q_a_init_idx": -7,
                    "retry_frame_zero_ground_on_infeasible": True,
                    "sqp_max_iterations": 10,
                },
            },
            "dataset_partition": "amass_smplx_processed",
            "experiment_name": None,
            "run_kind": "single",
            "sequence_key": artifact_path.parent.name,
            "test_payload": json.loads(config_json),
            "variant": {
                "name": "identity",
                "translation": [0.0, 0.0, 0.0],
                "rotation": 0.0,
                "object_scale": [1.0, 1.0, 1.0],
            },
        },
        sort_keys=True,
    )
    config_sha256 = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
    human_joints = np.asarray([[[0.0, 0.0, 1.0]]], dtype=np.float32)
    robot_joints = np.asarray([[[0.0, 0.0, 0.9]]], dtype=np.float32)
    qpos = np.zeros((1, 7), dtype=np.float64)
    qpos[:, 3] = 1.0
    object_poses = np.zeros((1, 7), dtype=np.float32)
    object_poses[:, 3] = 1.0
    payload = {
        "schema_version": np.int32(RESULT_SCHEMA_VERSION),
        "run_kind": np.asarray("single"),
        "variant": np.asarray("identity"),
        "source_path": np.asarray(str(source_path.resolve())),
        "source_data_format": np.asarray("amass"),
        "robot_type": np.asarray("g1"),
        "task_type": np.asarray("robot_only"),
        "dataset_partition": np.asarray("amass_smplx_processed"),
        "sequence_key": np.asarray(artifact_path.parent.name),
        "experiment_name": np.asarray(""),
        "object_name": np.asarray("ground"),
        "object_urdf": np.asarray(""),
        "object_urdf_sha256": np.asarray(""),
        "object_asset_manifest_json": np.asarray(""),
        "object_asset_manifest_sha256": np.asarray(""),
        "qpos": qpos,
        "qpos_layout": np.asarray("mujoco_free_root_xyz_wxyz_then_actuated_then_optional_object_free_joint"),
        "human_joints": human_joints,
        "human_joint_names": np.asarray(["Pelvis"]),
        "human_joint_parent_indices": np.asarray([-1], dtype=np.int32),
        "mapped_human_joints": human_joints.copy(),
        "mapped_human_joint_names": np.asarray(["Pelvis"]),
        "mapped_robot_joints": robot_joints.copy(),
        "mapped_robot_link_names": np.asarray(["pelvis_link"]),
        "robot_link_positions": robot_joints.copy(),
        "robot_link_quaternions_wxyz": np.asarray(
            [[[1.0, 0.0, 0.0, 0.0]]],
            dtype=np.float32,
        ),
        "robot_link_names": np.asarray(["pelvis_link"]),
        "robot_link_parent_indices": np.asarray([-1], dtype=np.int32),
        "robot_actuated_joint_names": np.asarray([], dtype=str),
        "contains_object_in_qpos": np.asarray(False),
        "object_poses_demo": object_poses.copy(),
        "object_poses_target": object_poses.copy(),
        "object_pose_layout": np.asarray("xyz_wxyz"),
        "quaternion_convention": np.asarray("wxyz"),
        "world_coordinate_system": np.asarray("right_handed_z_up"),
        "fps": np.float64(30.0),
        "source_sha256": np.asarray(source_sha256),
        "config_json": np.asarray(config_json),
        "config_sha256": np.asarray(config_sha256),
        "orientation_source": np.asarray("direct_local_rotation_fk" if orientation else "absent"),
        "source_human_height": np.asarray(1.7),
        "human_position_scale": np.asarray(1.0),
        "human_position_preprocessing": np.asarray("ground_align_and_uniform_scale"),
        "foot_sticking_side_names": np.asarray(["left", "right"]),
        "foot_sticking_states": np.zeros((1, 2), dtype=bool),
        "foot_sticking_tolerance": np.asarray(0.0),
        "foot_sticking_fallback_tolerance": np.asarray(np.nan),
        "foot_sticking_fallback_frames": np.asarray([], dtype=np.int32),
        "release_foot_sticking_on_infeasible": np.asarray(False),
        "foot_sticking_release_frames": np.asarray([], dtype=np.int32),
        "release_object_non_penetration_on_infeasible": np.asarray(False),
        "object_non_penetration_release_frames": np.asarray(
            [],
            dtype=np.int32,
        ),
        "object_non_penetration_eligible_for_saved_trajectory": np.asarray(
            False,
        ),
        "foot_sticking_enabled_for_saved_trajectory": np.asarray(False),
        "foot_sticking_full_sequence_retry_frame": np.asarray(
            -1,
            dtype=np.int32,
        ),
        "frame_zero_ground_retry_policy": np.asarray(
            "robot_only_frame_zero_horizontal_ground_lift_v1",
        ),
        "frame_zero_ground_retry_eligible": np.asarray(True),
        "frame_zero_ground_retry_triggered": np.asarray(False),
        "frame_zero_ground_retry_initial_min_distance_m": np.float64(0.0),
        "frame_zero_ground_retry_corrected_min_distance_m": np.float64(0.0),
        "frame_zero_ground_retry_lift_m": np.float64(0.0),
        "frame_zero_ground_retry_interior_margin_m": np.float64(0.0),
        "frame_zero_ground_retry_initial_sqp_iterations": np.int32(0),
        "frame_costs": np.asarray([1.0], dtype=np.float64),
        "sqp_iteration_counts": np.asarray([1], dtype=np.int32),
        "sqp_stop_reasons": np.asarray(["converged"]),
        "ground_non_penetration_violation": np.zeros(
            1,
            dtype=np.float64,
        ),
        "object_non_penetration_violation": np.zeros(
            1,
            dtype=np.float64,
        ),
        "foot_sticking_violation": np.zeros(1, dtype=np.float64),
        "foot_lock_violation": np.zeros(1, dtype=np.float64),
        "self_collision_violation": np.zeros(1, dtype=np.float64),
        "joint_limits_violation": np.zeros(1, dtype=np.float64),
        "constraint_mode_foot_sticking": np.asarray(["inactive"]),
        "constraint_mode_object_non_penetration_released": np.zeros(
            1,
            dtype=bool,
        ),
        "constraint_mode_trust_region_released": np.zeros(
            1,
            dtype=bool,
        ),
        "cost": np.asarray(1.0),
        "orientation_tracking_enabled": np.asarray(False),
        "orientation_diagnostics_enabled": np.asarray(False),
        "orientation_human_joint_names": np.asarray([], dtype=str),
        "orientation_robot_link_names": np.asarray([], dtype=str),
        "orientation_weights": np.asarray([], dtype=np.float64),
        "orientation_alignment_mode": np.asarray("explicit"),
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
        "orientation_reference_robot_qpos": np.asarray(
            [],
            dtype=np.float32,
        ),
        "orientation_target_quaternions_wxyz": np.empty(
            (1, 0, 4),
            dtype=np.float32,
        ),
        "orientation_robot_quaternions_wxyz": np.empty(
            (1, 0, 4),
            dtype=np.float32,
        ),
        "orientation_errors_rad": np.empty(
            (1, 0),
            dtype=np.float32,
        ),
        "orientation_frame_costs": np.zeros(1, dtype=np.float64),
        "interaction_source_vertices_w": human_joints.copy(),
        "interaction_target_vertices_w": robot_joints.copy(),
        "interaction_tetrahedra": np.empty((1, 0, 4), dtype=np.int32),
        "interaction_tetrahedra_counts": np.zeros(1, dtype=np.int32),
        "interaction_num_human_vertices": np.int32(1),
        "interaction_num_object_vertices": np.int32(0),
        "interaction_mesh_edges_default": np.asarray("cross"),
    }
    orientation_tensor = np.asarray(
        [[[1.0, 0.0, 0.0, 0.0]]],
        dtype=np.float32,
    )
    if orientation:
        payload.update(
            {
                "human_orientation_joint_names": np.asarray(["Pelvis"]),
                "human_orientation_quaternions_wxyz": orientation_tensor,
                "human_orientation_sha256": np.asarray(
                    _direct_orientation_tensor_sha256(
                        ("Pelvis",),
                        orientation_tensor,
                    )
                ),
            }
        )
    write_result_artifact(artifact_path, payload)
    return ExpectedArtifact(
        output_path=artifact_path.resolve(),
        source_path=source_path.resolve(),
        source_sha256=source_sha256,
        config_sha256=config_sha256,
        robot="g1",
        task_type="robot_only",
        data_format="amass",
        variant="identity",
        run_kind="single",
        dataset_id="amass",
        frame_count=1,
        human_joint_names=("Pelvis",),
        human_joint_parent_indices=(-1,),
        human_orientation_joint_names=(("Pelvis",) if orientation else None),
        human_orientation_source=("direct_local_rotation_fk" if orientation else "absent"),
        human_orientation_sha256=(
            _direct_orientation_tensor_sha256(("Pelvis",), orientation_tensor) if orientation else None
        ),
        human_orientation_tensor_shape=(tuple(orientation_tensor.shape) if orientation else None),
        human_orientation_tensor_bytes=(orientation_tensor.tobytes(order="C") if orientation else None),
        robot_link_names=("pelvis_link",),
        robot_link_parent_indices=(-1,),
    )


def _validated_promotion_tree(
    tmp_path: Path,
    *,
    run_id: str,
) -> tuple[Path, Path, Path, Any]:
    results_root = tmp_path / "demo_results"
    stage = staging_root(results_root, run_id)
    source_path = tmp_path / f"{run_id}-source.npz"
    source_path.touch()
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    artifact_path = (
        stage / "canonical" / "g1" / "robot_only" / "amass" / "amass_smplx_processed" / "walk" / "identity.npz"
    )
    expected = _write_valid_artifact(
        artifact_path,
        source_path=source_path,
        source_sha256=source_hash,
        config_json=json.dumps({"run_id": run_id}, sort_keys=True),
    )
    formal = results_root / "v1"
    formal.mkdir(parents=True)
    (formal / "old-result.txt").write_text("old", encoding="utf-8")
    validation = validate_rebuild_directory(stage, (expected,))
    assert validation.ok
    return results_root, stage, formal, validation


def _promotion_report_payload(
    *,
    run_id: str,
    results_root: Path,
    stage: Path,
    validation: Any,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "status": "planning",
        "results_root": str(results_root),
        "staging_root": str(stage),
        "include_augmentation_variants": True,
        "batch_failures": [],
        "planning": {
            "ok": True,
            "output_collision_count": 0,
        },
        "validation": validation.to_dict(),
        "promotion": None,
    }


def _rewrite_quality_fields(
    artifact_path: Path,
    *,
    foot_enabled: bool | None = None,
    fallback_frames: tuple[int, ...] | None = None,
    foot_release_frames: tuple[int, ...] | None = None,
    object_release_frames: tuple[int, ...] | None = None,
    object_release_enabled: bool | None = None,
    full_sequence_retry_frame: int | None = None,
) -> None:
    with np.load(artifact_path, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    if foot_enabled is not None:
        payload["foot_sticking_enabled_for_saved_trajectory"] = np.asarray(foot_enabled)
    if fallback_frames is not None:
        payload["foot_sticking_fallback_tolerance"] = np.float64(0.1)
        payload["foot_sticking_fallback_frames"] = np.asarray(
            fallback_frames,
            dtype=np.int32,
        )
        foot_modes = np.asarray(
            payload["constraint_mode_foot_sticking"],
        ).copy()
        foot_states = np.asarray(payload["foot_sticking_states"]).copy()
        foot_modes[list(fallback_frames)] = "relaxed"
        foot_states[list(fallback_frames), 0] = True
        payload["constraint_mode_foot_sticking"] = foot_modes
        payload["foot_sticking_states"] = foot_states
    if foot_release_frames is not None:
        payload["release_foot_sticking_on_infeasible"] = np.asarray(True)
        payload["foot_sticking_release_frames"] = np.asarray(
            foot_release_frames,
            dtype=np.int32,
        )
        foot_modes = np.asarray(
            payload["constraint_mode_foot_sticking"],
        ).copy()
        foot_states = np.asarray(payload["foot_sticking_states"]).copy()
        foot_modes[list(foot_release_frames)] = "released"
        foot_states[list(foot_release_frames), 0] = True
        payload["constraint_mode_foot_sticking"] = foot_modes
        payload["foot_sticking_states"] = foot_states
    if object_release_frames is not None:
        payload["release_object_non_penetration_on_infeasible"] = np.asarray(True)
        payload["object_non_penetration_release_frames"] = np.asarray(
            object_release_frames,
            dtype=np.int32,
        )
        object_modes = np.zeros(
            np.asarray(payload["qpos"]).shape[0],
            dtype=bool,
        )
        object_modes[list(object_release_frames)] = True
        payload["constraint_mode_object_non_penetration_released"] = object_modes
    if object_release_enabled is not None:
        payload["release_object_non_penetration_on_infeasible"] = np.asarray(object_release_enabled)
    if full_sequence_retry_frame is not None:
        payload["foot_sticking_full_sequence_retry_frame"] = np.int32(full_sequence_retry_frame)
    write_result_artifact(artifact_path, payload)


def _write_object_quality_artifact(
    artifact_path: Path,
    *,
    source_path: Path,
    variant: str,
) -> ExpectedArtifact:
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    expected = _write_valid_artifact(
        artifact_path,
        source_path=source_path,
        source_sha256=source_sha256,
        config_json=json.dumps(
            {
                "quality": "object",
                "variant": variant,
            },
            sort_keys=True,
        ),
        orientation=False,
    )
    with np.load(artifact_path, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    run_kind = "single" if variant == "identity" else "augmentation"
    sequence_key = artifact_path.parent.name
    variant_translation = [0.0, 0.0, 0.0] if variant == "identity" else [0.2, 0.0, 0.0]
    config_json = json.dumps(
        {
            "config": {
                "data_format": "amass",
                "robot_config": {"robot_type": "g1"},
                "task_config": {"object_name": "test_object"},
                "task_type": "object_interaction",
                "retargeter": {
                    "activate_obj_non_penetration": True,
                    "penetration_tolerance": 0.001,
                    "q_a_init_idx": -7,
                    "retry_frame_zero_ground_on_infeasible": True,
                    "sqp_max_iterations": 10,
                },
            },
            "dataset_partition": "object_quality",
            "experiment_name": None,
            "run_kind": run_kind,
            "sequence_key": sequence_key,
            "variant": {
                "name": variant,
                "translation": variant_translation,
                "rotation": 0.0,
                "object_scale": [1.0, 1.0, 1.0],
            },
        },
        sort_keys=True,
    )
    object_urdf = source_path.parent / "test-object.urdf"
    object_urdf.write_text("<robot name='test-object'/>\n", encoding="utf-8")
    object_asset_manifest_json, object_asset_manifest_sha256 = build_object_asset_manifest(object_urdf)
    object_poses = np.zeros((1, 7), dtype=np.float32)
    object_poses[:, 3] = 1.0
    qpos = np.zeros((1, 14), dtype=np.float64)
    qpos[:, 3] = 1.0
    qpos[:, 7:] = object_poses
    object_points = np.zeros((1, 3), dtype=np.float32)
    object_points_world = np.zeros((1, 1, 3), dtype=np.float32)
    payload.update(
        {
            "run_kind": np.asarray(run_kind),
            "variant": np.asarray(variant),
            "dataset_partition": np.asarray("object_quality"),
            "sequence_key": np.asarray(sequence_key),
            "config_json": np.asarray(config_json),
            "config_sha256": np.asarray(hashlib.sha256(config_json.encode("utf-8")).hexdigest()),
            "task_type": np.asarray("object_interaction"),
            "object_name": np.asarray("test_object"),
            "object_urdf": np.asarray(str(object_urdf.resolve())),
            "object_urdf_sha256": np.asarray(
                compute_file_sha256(object_urdf),
            ),
            "object_asset_manifest_json": np.asarray(
                object_asset_manifest_json,
            ),
            "object_asset_manifest_sha256": np.asarray(
                object_asset_manifest_sha256,
            ),
            "qpos": qpos,
            "contains_object_in_qpos": np.asarray(True),
            "object_poses_demo": object_poses,
            "object_poses_target": object_poses.copy(),
            "foot_sticking_enabled_for_saved_trajectory": np.asarray(False),
            "release_object_non_penetration_on_infeasible": np.asarray(True),
            "object_non_penetration_eligible_for_saved_trajectory": np.asarray(
                True,
            ),
            "frame_zero_ground_retry_eligible": np.asarray(False),
            "object_points_demo_local": object_points,
            "object_points_target_local": object_points.copy(),
            "object_points_demo_world": object_points_world,
            "object_points_target_world": object_points_world.copy(),
            "interaction_source_vertices_w": np.concatenate(
                (
                    np.asarray(payload["mapped_human_joints"]),
                    object_points_world,
                ),
                axis=1,
            ),
            "interaction_target_vertices_w": np.concatenate(
                (
                    np.asarray(payload["mapped_robot_joints"]),
                    object_points_world,
                ),
                axis=1,
            ),
            "interaction_num_object_vertices": np.int32(1),
        }
    )
    write_result_artifact(artifact_path, payload)
    return replace(
        expected,
        config_sha256=hashlib.sha256(config_json.encode("utf-8")).hexdigest(),
        task_type="object_interaction",
        variant=variant,
        run_kind=run_kind,
        dataset_id="object_quality",
    )


def _write_climbing_quality_artifact(
    artifact_path: Path,
    *,
    source_path: Path,
    variant: str,
    object_release_frames: tuple[int, ...] = (),
) -> ExpectedArtifact:
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    expected = _write_valid_artifact(
        artifact_path,
        source_path=source_path,
        source_sha256=source_sha256,
        config_json=json.dumps({"quality": "climbing"}, sort_keys=True),
        orientation=False,
    )
    with np.load(artifact_path, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    sequence_key = artifact_path.parent.name
    scale_by_variant = {
        "identity": 1.0,
        "z_scale_0p8": 0.8,
        "z_scale_0p9": 0.9,
        "z_scale_1p1": 1.1,
        "z_scale_1p2": 1.2,
    }
    object_scale = scale_by_variant[variant]
    run_kind = "single" if variant == "identity" else "augmentation"
    config_json = json.dumps(
        {
            "config": {
                "data_format": "amass",
                "robot_config": {"robot_type": "g1"},
                "task_config": {"object_name": "multi_boxes"},
                "task_type": "climbing",
                "retargeter": {
                    "activate_obj_non_penetration": True,
                    "penetration_tolerance": 0.001,
                    "q_a_init_idx": -7,
                    "retry_frame_zero_ground_on_infeasible": True,
                    "sqp_max_iterations": 10,
                },
            },
            "dataset_partition": "climb_quality",
            "experiment_name": None,
            "run_kind": run_kind,
            "sequence_key": sequence_key,
            "variant": {
                "name": variant,
                "translation": [0.0, 0.0, 0.0],
                "rotation": 0.0,
                "object_scale": [1.0, 1.0, object_scale],
            },
        },
        sort_keys=True,
    )
    terrain_urdf = source_path.parent / "multi-boxes.urdf"
    terrain_urdf.write_text(
        "<robot name='multi-boxes'/>\n",
        encoding="utf-8",
    )
    object_asset_manifest_json, object_asset_manifest_sha256 = build_object_asset_manifest(terrain_urdf)
    payload.update(
        {
            "run_kind": np.asarray(run_kind),
            "variant": np.asarray(variant),
            "task_type": np.asarray("climbing"),
            "dataset_partition": np.asarray("climb_quality"),
            "sequence_key": np.asarray(sequence_key),
            "config_json": np.asarray(config_json),
            "config_sha256": np.asarray(
                hashlib.sha256(config_json.encode("utf-8")).hexdigest(),
            ),
            "object_name": np.asarray("multi_boxes"),
            "object_urdf": np.asarray(str(terrain_urdf.resolve())),
            "object_urdf_sha256": np.asarray(
                compute_file_sha256(terrain_urdf),
            ),
            "object_asset_manifest_json": np.asarray(
                object_asset_manifest_json,
            ),
            "object_asset_manifest_sha256": np.asarray(
                object_asset_manifest_sha256,
            ),
            "release_object_non_penetration_on_infeasible": np.asarray(
                True,
            ),
            "object_non_penetration_release_frames": np.asarray(
                object_release_frames,
                dtype=np.int32,
            ),
            "constraint_mode_object_non_penetration_released": np.asarray(
                [0 in object_release_frames],
                dtype=bool,
            ),
            "object_non_penetration_eligible_for_saved_trajectory": np.asarray(
                True,
            ),
            "frame_zero_ground_retry_eligible": np.asarray(False),
        }
    )
    write_result_artifact(artifact_path, payload)
    return replace(
        expected,
        config_sha256=hashlib.sha256(config_json.encode("utf-8")).hexdigest(),
        task_type="climbing",
        variant=variant,
        run_kind=run_kind,
        dataset_id="climb_quality",
    )


def test_matrix_covers_both_robots_and_all_task_rows(tmp_path: Path) -> None:
    data_root = tmp_path / "demo_data"
    _touch_dataset_roots(data_root)
    csv_root = tmp_path / "converted_noetix_csv"
    csv_root.mkdir()
    cfg = RebuildDemoResultsConfig(
        data_root=data_root,
        results_root=tmp_path / "demo_results",
        noetix_csv_root=csv_root,
    )

    plans = build_rebuild_matrix(cfg)

    assert len(plans) == 16
    assert {plan.robot for plan in plans} == {"g1", "e1"}
    assert {plan.spec.dataset_id for plan in plans} == set(DATASET_SPEC_BY_ID)
    object_batches = [plan for plan in plans if plan.spec.dataset_id == "omomo_object_interaction"]
    climb_batches = [plan for plan in plans if plan.spec.task_type == "climbing"]
    assert all(plan.spec.augmentation for plan in object_batches + climb_batches)
    assert staging_root(cfg.results_root, cfg.run_id) == (cfg.results_root.resolve() / ".staging" / cfg.run_id / "v1")


def test_noetix_csv_uses_required_default_root_without_override(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "demo_data"
    csv_root = data_root / "noetix_csv_climb"
    csv_root.mkdir(parents=True)
    cfg = RebuildDemoResultsConfig(
        data_root=data_root,
        results_root=tmp_path / "demo_results",
        robots=("g1",),
        datasets=("noetix_csv_climb",),
    )

    (plan,) = build_rebuild_matrix(cfg)

    assert plan.available
    assert plan.data_dir == csv_root.resolve()
    assert plan.skip_reason is None


def test_full_promotion_selection_requires_noetix_csv_dataset(
    tmp_path: Path,
) -> None:
    without_csv = tuple(dataset_id for dataset_id in DATASET_SPEC_BY_ID if dataset_id != "noetix_csv_climb")
    partial = RebuildDemoResultsConfig(
        data_root=tmp_path / "demo_data",
        results_root=tmp_path / "demo_results",
        datasets=without_csv,
    )
    complete = replace(partial, datasets=())

    assert not _full_matrix_selected(partial)
    assert _full_matrix_selected(complete)
    assert _full_matrix_selected(
        replace(
            complete,
            include_augmentation_variants=False,
        )
    )


@pytest.mark.parametrize("field_name", sorted(rebuild.FORMAL_QUALITY_LIMITS))
def test_formal_promotion_rejects_every_wider_quality_limit(
    tmp_path: Path,
    field_name: str,
) -> None:
    cfg = RebuildDemoResultsConfig(
        data_root=tmp_path / "demo_data",
        results_root=tmp_path / "demo_results",
    )
    assert rebuild._formal_quality_profile_relaxations(cfg) == {}

    relaxed = replace(
        cfg,
        **{
            field_name: min(
                1.0,
                rebuild.FORMAL_QUALITY_LIMITS[field_name] + 0.01,
            )
        },
    )
    relaxations = rebuild._formal_quality_profile_relaxations(relaxed)

    if rebuild.FORMAL_QUALITY_LIMITS[field_name] == 1.0:
        assert relaxations == {}
    else:
        assert set(relaxations) == {field_name}


def test_formal_promotion_never_calls_publisher_with_wider_quality_profile(
    tmp_path: Path,
) -> None:
    cfg = RebuildDemoResultsConfig(
        data_root=tmp_path / "demo_data",
        results_root=tmp_path / "demo_results",
        run_id="relaxed-quality",
        validate_only=True,
        promote=True,
        max_fallback_frame_fraction_per_artifact=0.11,
    )
    validation = rebuild.ValidationSummary(
        ok=True,
        root=str(staging_root(cfg.results_root, cfg.run_id)),
        artifact_count=1,
        expected_artifact_count=1,
        expected_source_count=1,
        unexpected_artifact_count=0,
        tree_sha256="a" * 64,
        tree_file_count=1,
        tree_byte_count=1,
        external_asset_sha256="b" * 64,
        external_asset_file_count=0,
        external_asset_byte_count=0,
        issue_count=0,
        issues=(),
        statistics={"valid_artifacts": 1},
    )
    audit = rebuild._ExecutionAudit(
        ok=True,
        omitted_artifact_paths=frozenset(),
        blocking_failures=(),
        report={
            "ok": True,
            "source_job_counts": {},
            "batches": [],
        },
    )
    data_dir = cfg.data_root / "OMOMO_new"
    data_dir.mkdir(parents=True)
    plans = tuple(
        BatchPlan(
            robot=robot,
            spec=DATASET_SPEC_BY_ID["omomo_robot_only"],
            data_dir=data_dir,
        )
        for robot in ("g1", "e1")
    )

    with mock.patch(
        "holosoma_retargeting.examples.rebuild_demo_results.build_rebuild_matrix",
        return_value=plans,
    ), mock.patch(
        "holosoma_retargeting.examples.rebuild_demo_results.plan_expected_artifacts",
        return_value=(),
    ), mock.patch(
        "holosoma_retargeting.examples.rebuild_demo_results._audit_persisted_batch_reports",
        return_value=audit,
    ), mock.patch(
        "holosoma_retargeting.examples.rebuild_demo_results.validate_rebuild_directory",
        return_value=validation,
    ), mock.patch(
        "holosoma_retargeting.examples.rebuild_demo_results.promote_staging",
    ) as promote, pytest.raises(DemoRebuildError, match="failed"):
        run_rebuild(cfg)

    promote.assert_not_called()
    report = json.loads(
        (run_report_root(cfg.results_root, cfg.run_id) / "report.json").read_text(
            encoding="utf-8",
        )
    )
    assert "wider than the formal profile" in report["batch_failures"][0]["error"]


def test_attempt_audit_accepts_complete_g1_success(
    tmp_path: Path,
) -> None:
    cfg, plan, expected, stage, report_root = _audit_plan_and_expected(
        tmp_path,
        robot="g1",
        dataset_id="omomo_object_interaction",
    )
    for item in expected:
        item.output_path.parent.mkdir(parents=True, exist_ok=True)
        item.output_path.touch()
    _write_attempt_batch_report(
        plan,
        expected,
        stage=stage,
        report_root=report_root,
        failure_error=None,
    )

    audit = rebuild._audit_persisted_batch_reports(
        cfg,
        (plan,),
        expected,
        stage=stage,
        report_root=report_root,
    )

    assert audit.ok
    assert audit.omitted_artifact_paths == frozenset()
    counts = audit.report["source_job_counts"]
    assert counts["attempted_source_count"] == 1
    assert counts["succeeded_source_count"] == 1
    assert counts["omitted_source_count"] == 0
    assert counts["missing_required_artifact_count"] == 0
    assert audit.report["by_robot"]["g1"]["succeeded_source_count"] == 1


def test_attempt_audit_never_accepts_g1_failure(
    tmp_path: Path,
) -> None:
    cfg, plan, expected, stage, report_root = _audit_plan_and_expected(
        tmp_path,
        robot="g1",
        dataset_id="omomo_object_interaction",
    )
    _write_attempt_batch_report(
        plan,
        expected,
        stage=stage,
        report_root=report_root,
        failure_error="SQPNonlinearFeasibilityError: unreachable",
    )

    audit = rebuild._audit_persisted_batch_reports(
        cfg,
        (plan,),
        expected,
        stage=stage,
        report_root=report_root,
        invocations={
            plan.batch_id: rebuild._BatchInvocation(
                report_refreshed=True,
                error=("RuntimeError: 1 of 1 retargeting tasks failed; see batch report"),
            )
        },
    )

    assert not audit.ok
    assert audit.omitted_artifact_paths == frozenset()
    assert audit.report["source_job_counts"]["failed_source_count"] == 1
    assert "never eligible" in audit.blocking_failures[0]["error"]


def test_attempt_audit_maps_all_missing_e1_augmentation_variants_to_failure(
    tmp_path: Path,
) -> None:
    cfg, plan, expected, stage, report_root = _audit_plan_and_expected(
        tmp_path,
        robot="e1",
        dataset_id="omomo_object_interaction",
    )
    failure_error = "SQPNonlinearFeasibilityError: E1 reach limit"
    _write_attempt_batch_report(
        plan,
        expected,
        stage=stage,
        report_root=report_root,
        failure_error=failure_error,
    )

    audit = rebuild._audit_persisted_batch_reports(
        cfg,
        (plan,),
        expected,
        stage=stage,
        report_root=report_root,
        invocations={
            plan.batch_id: rebuild._BatchInvocation(
                report_refreshed=True,
                error=("RuntimeError: 1 of 1 retargeting tasks failed; see batch report"),
            )
        },
    )

    assert audit.ok
    assert audit.omitted_artifact_paths == {item.output_path for item in expected}
    assert audit.report["source_job_counts"]["attempted_source_count"] == 1
    assert audit.report["source_job_counts"]["omitted_source_count"] == 1
    assert audit.report["source_job_counts"]["omitted_artifact_count"] == len(expected)
    (omission,) = audit.report["omissions"]
    assert omission["error"] == failure_error
    assert omission["omitted_variants"] == sorted(item.variant for item in expected)
    assert len({omission["batch_report_sha256"]}) == 1
    assert audit.report["by_robot"]["e1"]["by_task"]["object_interaction"]["omitted_source_count"] == 1


def test_attempt_audit_rejects_stale_report_as_current_attempt_evidence(
    tmp_path: Path,
) -> None:
    cfg, plan, expected, stage, report_root = _audit_plan_and_expected(
        tmp_path,
        robot="e1",
        dataset_id="omomo_object_interaction",
    )
    _write_attempt_batch_report(
        plan,
        expected,
        stage=stage,
        report_root=report_root,
        failure_error="SQPNonlinearFeasibilityError: stale failure",
    )

    audit = rebuild._audit_persisted_batch_reports(
        cfg,
        (plan,),
        expected,
        stage=stage,
        report_root=report_root,
        invocations={
            plan.batch_id: rebuild._BatchInvocation(
                report_refreshed=False,
                error="ValueError: preflight failed before workers",
            )
        },
    )

    assert not audit.ok
    assert audit.omitted_artifact_paths == frozenset()
    assert "did not create or refresh" in audit.blocking_failures[0]["error"]


def test_attempt_audit_rejects_missing_batch_report_evidence(
    tmp_path: Path,
) -> None:
    cfg, plan, expected, stage, report_root = _audit_plan_and_expected(
        tmp_path,
        robot="e1",
        dataset_id="omomo_object_interaction",
    )

    audit = rebuild._audit_persisted_batch_reports(
        cfg,
        (plan,),
        expected,
        stage=stage,
        report_root=report_root,
    )

    assert not audit.ok
    assert audit.omitted_artifact_paths == frozenset()
    assert audit.report["source_job_counts"]["attempted_source_count"] == 0
    assert audit.report["source_job_counts"]["unverified_source_count"] == 1
    assert "does not exist" in audit.blocking_failures[0]["error"]


@pytest.mark.parametrize(
    ("field", "tampered_value", "expected_error"),
    [
        (
            "task_type",
            "robot_only",
            "identity does not match",
        ),
        (
            "total_files",
            2,
            "total_files does not match",
        ),
    ],
)
def test_attempt_audit_rejects_tampered_or_task_mismatched_report(
    tmp_path: Path,
    field: str,
    tampered_value: object,
    expected_error: str,
) -> None:
    cfg, plan, expected, stage, report_root = _audit_plan_and_expected(
        tmp_path,
        robot="e1",
        dataset_id="omomo_object_interaction",
    )
    report_path = _write_attempt_batch_report(
        plan,
        expected,
        stage=stage,
        report_root=report_root,
        failure_error="SQPNonlinearFeasibilityError: unreachable",
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload[field] = tampered_value
    report_path.write_text(
        json.dumps(payload, sort_keys=True),
        encoding="utf-8",
    )

    audit = rebuild._audit_persisted_batch_reports(
        cfg,
        (plan,),
        expected,
        stage=stage,
        report_root=report_root,
    )

    assert not audit.ok
    assert audit.omitted_artifact_paths == frozenset()
    assert expected_error in audit.blocking_failures[0]["error"]


def test_validate_only_rejects_report_without_final_rebuild_contract(
    tmp_path: Path,
) -> None:
    cfg, plan, expected, stage, report_root = _audit_plan_and_expected(
        tmp_path,
        robot="e1",
        dataset_id="omomo_object_interaction",
    )
    report_path = _write_attempt_batch_report(
        plan,
        expected,
        stage=stage,
        report_root=report_root,
        failure_error="SQPNonlinearFeasibilityError: E1 limit",
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    del payload["rebuild_contract"]
    report_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    audit = rebuild._audit_persisted_batch_reports(
        replace(cfg, validate_only=True),
        (plan,),
        expected,
        stage=stage,
        report_root=report_root,
    )

    assert not audit.ok
    assert audit.omitted_artifact_paths == frozenset()
    assert "no rebuild_contract" in audit.blocking_failures[0]["error"]


def test_fresh_batch_report_is_atomically_enriched_with_final_plan_contract(
    tmp_path: Path,
) -> None:
    cfg, plan, expected, stage, report_root = _audit_plan_and_expected(
        tmp_path,
        robot="e1",
        dataset_id="omomo_object_interaction",
    )
    report_path = _write_attempt_batch_report(
        plan,
        expected,
        stage=stage,
        report_root=report_root,
        failure_error="SQPNonlinearFeasibilityError: E1 limit",
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    del payload["rebuild_contract"]
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    rebuild._attach_batch_rebuild_contract(
        report_path,
        outer_run_id=cfg.run_id,
        plan=plan,
        expected_items=expected,
        include_augmentation_variants=(cfg.include_augmentation_variants),
    )

    enriched, _sha256 = rebuild._load_trusted_batch_report(report_path)
    rebuild._validate_batch_rebuild_contract(
        enriched,
        outer_run_id=cfg.run_id,
        plan=plan,
        expected_items=expected,
        include_augmentation_variants=(cfg.include_augmentation_variants),
    )
    assert enriched["rebuild_contract"]["plan_sha256"] == rebuild._batch_plan_digest(
        expected,
        include_augmentation_variants=cfg.include_augmentation_variants,
    )


def test_validate_only_rejects_report_bound_to_old_source_and_config_plan(
    tmp_path: Path,
) -> None:
    cfg, plan, expected, stage, report_root = _audit_plan_and_expected(
        tmp_path,
        robot="e1",
        dataset_id="omomo_object_interaction",
    )
    _write_attempt_batch_report(
        plan,
        expected,
        stage=stage,
        report_root=report_root,
        failure_error="SQPNonlinearFeasibilityError: old attempt",
    )
    expected[0].source_path.write_bytes(b"changed after the attempt")
    changed_expected = tuple(
        replace(
            item,
            source_sha256="c" * 64,
            config_sha256="d" * 64,
        )
        for item in expected
    )

    audit = rebuild._audit_persisted_batch_reports(
        replace(cfg, validate_only=True),
        (plan,),
        changed_expected,
        stage=stage,
        report_root=report_root,
    )

    assert not audit.ok
    assert audit.omitted_artifact_paths == frozenset()
    assert "does not match the final source/config plan" in audit.blocking_failures[0]["error"]


def test_validate_only_rejects_report_from_another_augmentation_mode(
    tmp_path: Path,
) -> None:
    cfg, plan, expected, stage, report_root = _audit_plan_and_expected(
        tmp_path,
        robot="e1",
        dataset_id="omomo_object_interaction",
        include_augmentation_variants=False,
    )
    assert [item.variant for item in expected] == ["identity"]
    assert [item.run_kind for item in expected] == ["single"]
    _write_attempt_batch_report(
        plan,
        expected,
        stage=stage,
        report_root=report_root,
        failure_error="SQPNonlinearFeasibilityError: old full-family attempt",
        include_augmentation_variants=True,
    )

    audit = rebuild._audit_persisted_batch_reports(
        replace(cfg, validate_only=True),
        (plan,),
        expected,
        stage=stage,
        report_root=report_root,
    )

    assert not audit.ok
    assert audit.omitted_artifact_paths == frozenset()
    assert "does not match the final source/config plan" in audit.blocking_failures[0]["error"]


@pytest.mark.parametrize(
    "failure_error",
    [
        "KeyError: implementation bug",
        "BrokenProcessPool: worker terminated",
        "ValueError: corrupt source data",
    ],
)
def test_e1_omission_rejects_non_structural_worker_failures(
    tmp_path: Path,
    failure_error: str,
) -> None:
    cfg, plan, expected, stage, report_root = _audit_plan_and_expected(
        tmp_path,
        robot="e1",
        dataset_id="omomo_object_interaction",
    )
    _write_attempt_batch_report(
        plan,
        expected,
        stage=stage,
        report_root=report_root,
        failure_error=failure_error,
    )

    audit = rebuild._audit_persisted_batch_reports(
        cfg,
        (plan,),
        expected,
        stage=stage,
        report_root=report_root,
    )

    assert not audit.ok
    assert audit.omitted_artifact_paths == frozenset()
    assert "only allowlisted SQPNonlinearFeasibilityError" in audit.blocking_failures[0]["error"]


def test_e1_robot_only_omission_requires_explicit_option(
    tmp_path: Path,
) -> None:
    cfg, plan, expected, stage, report_root = _audit_plan_and_expected(
        tmp_path,
        robot="e1",
        dataset_id="omomo_robot_only",
    )
    _write_attempt_batch_report(
        plan,
        expected,
        stage=stage,
        report_root=report_root,
        failure_error="SQPNonlinearFeasibilityError: E1 limit",
    )

    default_audit = rebuild._audit_persisted_batch_reports(
        cfg,
        (plan,),
        expected,
        stage=stage,
        report_root=report_root,
    )
    explicit_audit = rebuild._audit_persisted_batch_reports(
        replace(cfg, allow_e1_robot_only_omissions=True),
        (plan,),
        expected,
        stage=stage,
        report_root=report_root,
    )

    assert not default_audit.ok
    assert default_audit.omitted_artifact_paths == frozenset()
    assert explicit_audit.ok
    assert explicit_audit.omitted_artifact_paths == {item.output_path for item in expected}


def test_failure_evidence_never_exempts_an_existing_invalid_artifact(
    tmp_path: Path,
) -> None:
    cfg, plan, expected, stage, report_root = _audit_plan_and_expected(
        tmp_path,
        robot="e1",
        dataset_id="omomo_object_interaction",
    )
    existing_invalid = expected[0].output_path
    existing_invalid.parent.mkdir(parents=True, exist_ok=True)
    existing_invalid.write_bytes(b"not-an-npz")
    _write_attempt_batch_report(
        plan,
        expected,
        stage=stage,
        report_root=report_root,
        failure_error="SQPNonlinearFeasibilityError: E1 limit",
    )
    audit = rebuild._audit_persisted_batch_reports(
        cfg,
        (plan,),
        expected,
        stage=stage,
        report_root=report_root,
    )

    validation = validate_rebuild_directory(
        stage,
        expected,
        omitted_artifact_paths=audit.omitted_artifact_paths,
    )

    assert audit.ok
    assert existing_invalid not in audit.omitted_artifact_paths
    assert not validation.ok
    assert any(issue.code == "invalid_artifact" and issue.path == str(existing_invalid) for issue in validation.issues)


def test_validate_only_rebuilds_identical_omissions_from_persisted_reports(
    tmp_path: Path,
) -> None:
    cfg, plan, expected, stage, report_root = _audit_plan_and_expected(
        tmp_path,
        robot="e1",
        dataset_id="omomo_object_interaction",
    )
    cfg = replace(cfg, validate_only=True)
    stage.mkdir(parents=True)
    _write_attempt_batch_report(
        plan,
        expected,
        stage=stage,
        report_root=report_root,
        failure_error="SQPNonlinearFeasibilityError: persistent E1 limit",
    )
    validation = rebuild.ValidationSummary(
        ok=True,
        root=str(stage),
        artifact_count=0,
        expected_artifact_count=len(expected),
        expected_source_count=1,
        unexpected_artifact_count=0,
        tree_sha256="",
        tree_file_count=0,
        tree_byte_count=0,
        external_asset_sha256="",
        external_asset_file_count=0,
        external_asset_byte_count=0,
        issue_count=0,
        issues=(),
        statistics={"valid_artifacts": 0},
    )

    with mock.patch(
        "holosoma_retargeting.examples.rebuild_demo_results.plan_expected_artifacts",
        return_value=expected,
    ), mock.patch(
        "holosoma_retargeting.examples.rebuild_demo_results.validate_rebuild_directory",
        return_value=validation,
    ) as validate:
        first = run_rebuild(cfg)
        second = run_rebuild(cfg)

    assert first["status"] == "validated"
    assert second["status"] == "validated"
    assert first["execution_audit"]["omission_set_sha256"] == second["execution_audit"]["omission_set_sha256"]
    assert first["execution_audit"]["omissions"] == second["execution_audit"]["omissions"]
    assert validate.call_count == 2
    for call in validate.call_args_list:
        assert call.kwargs["omitted_artifact_paths"] == {item.output_path for item in expected}
    promoted_report = rebuild._final_promotion_report(
        second,
        formal=cfg.results_root / "v1",
        archived=None,
        journal_path=report_root / "promotion.json",
    )
    assert promoted_report["execution_audit"] == second["execution_audit"]


def test_object_interaction_expected_plan_has_identity_and_five_variants(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "OMOMO_new"
    data_dir.mkdir()
    source_path = data_dir / "sub1_tripod_001.pt"
    _write_omomo_source(source_path)
    plan = BatchPlan(
        robot="g1",
        spec=DATASET_SPEC_BY_ID["omomo_object_interaction"],
        data_dir=data_dir,
    )
    stage = tmp_path / "results" / ".staging" / "run" / "v1"

    expected = plan_expected_artifacts((plan,), stage=stage)

    assert {item.variant for item in expected} == {
        "identity",
        "trans_0",
        "trans_1",
        "trans_2",
        "rot_0",
        "rot_1",
    }
    assert all(item.output_path.is_relative_to(stage) for item in expected)
    assert all(item.robot == "g1" for item in expected)
    assert all(item.frame_count == 2 for item in expected)
    assert all(len(item.human_joint_parent_indices) == len(item.human_joint_names) for item in expected)
    assert all(item.human_orientation_joint_names == tuple(DEMO_JOINTS_REGISTRY["omomo"]) for item in expected)
    assert all(item.human_orientation_source == "intermimic_global_orientation_tensor" for item in expected)
    expected_orientation_tensor = np.zeros(
        (2, len(DEMO_JOINTS_REGISTRY["omomo"]), 4),
        dtype=np.float32,
    )
    expected_orientation_tensor[..., 0] = 1.0
    expected_orientation_sha256 = _direct_orientation_tensor_sha256(
        tuple(DEMO_JOINTS_REGISTRY["omomo"]),
        expected_orientation_tensor,
    )
    assert {item.human_orientation_sha256 for item in expected} == {expected_orientation_sha256}
    assert {item.human_orientation_tensor_shape for item in expected} == {expected_orientation_tensor.shape}
    assert len({id(item.human_orientation_tensor_bytes) for item in expected}) == 1
    assert {len(item.robot_link_names) for item in expected} == {50}


@pytest.mark.parametrize(
    "dataset_id",
    [
        "omomo_object_interaction",
        "generic_climb",
    ],
)
def test_identity_only_plan_keeps_each_source_and_only_expects_single_identity(
    tmp_path: Path,
    dataset_id: str,
) -> None:
    spec = DATASET_SPEC_BY_ID[dataset_id]
    data_dir = tmp_path / str(spec.relative_data_dir)
    data_dir.mkdir()
    if spec.data_format == "omomo":
        source_path = data_dir / "sub1_tripod_001.pt"
        _write_omomo_source(source_path)
    else:
        source_path = data_dir / "box_climb" / "motion.npz"
        source_path.parent.mkdir()
        _write_adapter_npz(source_path, data_format=spec.data_format)
    plan = BatchPlan(
        robot="g1",
        spec=spec,
        data_dir=data_dir,
    )
    cfg = RebuildDemoResultsConfig(
        data_root=tmp_path,
        results_root=tmp_path / "results",
        include_augmentation_variants=False,
    )

    expected = plan_expected_artifacts(
        (plan,),
        stage=tmp_path / "stage",
        cfg=cfg,
    )

    assert len(expected) == 1
    assert expected[0].source_path == source_path.resolve()
    assert expected[0].variant == "identity"
    assert expected[0].run_kind == "single"
    assert rebuild._expected_variant_names(cfg, plan) == ("identity",)


@pytest.mark.parametrize(
    ("dataset_id", "include_augmentation_variants", "expected_augmentation"),
    [
        ("omomo_object_interaction", True, True),
        ("omomo_object_interaction", False, False),
        ("generic_climb", True, True),
        ("generic_climb", False, False),
        ("omomo_robot_only", True, False),
    ],
)
def test_parallel_config_uses_the_effective_augmentation_mode(
    tmp_path: Path,
    dataset_id: str,
    include_augmentation_variants: bool,
    expected_augmentation: bool,
) -> None:
    spec = DATASET_SPEC_BY_ID[dataset_id]
    data_dir = tmp_path / dataset_id
    data_dir.mkdir()
    plan = BatchPlan(
        robot="g1",
        spec=spec,
        data_dir=data_dir,
    )
    cfg = RebuildDemoResultsConfig(
        data_root=tmp_path,
        results_root=tmp_path / "results",
        include_augmentation_variants=include_augmentation_variants,
    )

    parallel_config = _parallel_config(
        cfg,
        plan,
        stage=tmp_path / "stage",
        batch_report_path=tmp_path / "report.json",
    )

    assert rebuild._effective_augmentation(cfg, plan) is expected_augmentation
    assert parallel_config.augmentation is expected_augmentation


def test_augmentation_mode_is_part_of_the_batch_rebuild_contract(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "OMOMO_new"
    data_dir.mkdir()
    _write_omomo_source(data_dir / "sub1_tripod_001.pt")
    plan = BatchPlan(
        robot="g1",
        spec=DATASET_SPEC_BY_ID["omomo_object_interaction"],
        data_dir=data_dir,
    )
    full_cfg = RebuildDemoResultsConfig(
        data_root=tmp_path,
        results_root=tmp_path / "results",
    )
    identity_cfg = replace(
        full_cfg,
        include_augmentation_variants=False,
    )
    stage = tmp_path / "stage"
    full_expected = plan_expected_artifacts(
        (plan,),
        stage=stage,
        cfg=full_cfg,
    )
    identity_expected = plan_expected_artifacts(
        (plan,),
        stage=stage,
        cfg=identity_cfg,
    )

    full_contract = rebuild._batch_rebuild_contract(
        outer_run_id="mode-contract",
        plan=plan,
        expected_items=full_expected,
        include_augmentation_variants=True,
    )
    identity_contract = rebuild._batch_rebuild_contract(
        outer_run_id="mode-contract",
        plan=plan,
        expected_items=identity_expected,
        include_augmentation_variants=False,
    )

    assert full_contract["effective_augmentation"] is True
    assert full_contract["planned_artifact_count"] == 6
    assert identity_contract["effective_augmentation"] is False
    assert identity_contract["planned_artifact_count"] == 1
    assert full_contract["plan_sha256"] != identity_contract["plan_sha256"]
    with pytest.raises(
        ValueError,
        match="does not match the final source/config plan",
    ):
        rebuild._validate_batch_rebuild_contract(
            {"rebuild_contract": full_contract},
            outer_run_id="mode-contract",
            plan=plan,
            expected_items=identity_expected,
            include_augmentation_variants=False,
        )


def test_promotion_recovery_rejects_another_augmentation_mode(
    tmp_path: Path,
) -> None:
    results_root = tmp_path / "results"
    run_id = "mode-recovery"
    journal_path = rebuild.promotion_journal_path(
        results_root,
        run_id,
    )
    journal_path.parent.mkdir(parents=True)
    journal_path.write_text(
        json.dumps(
            {
                "final_report": {
                    "include_augmentation_variants": True,
                }
            }
        ),
        encoding="utf-8",
    )

    with mock.patch(
        "holosoma_retargeting.examples.rebuild_demo_results._commit_promotion_journal",
    ) as commit, pytest.raises(
        DemoRebuildError,
        match="augmentation mode does not match",
    ):
        rebuild._recover_promotion_if_present(
            results_root=results_root,
            run_id=run_id,
            report_path=journal_path.parent / "report.json",
            include_augmentation_variants=False,
        )

    commit.assert_not_called()


def test_identity_only_object_family_validation_is_complete(
    tmp_path: Path,
) -> None:
    stage = tmp_path / "stage"
    source_path = tmp_path / "source.npz"
    source_path.touch()
    expected = _write_object_quality_artifact(
        stage
        / "canonical"
        / "g1"
        / "object_interaction"
        / "amass"
        / "object_quality"
        / "identity-only"
        / "identity.npz",
        source_path=source_path,
        variant="identity",
    )

    validation = validate_rebuild_directory(stage, (expected,))

    assert validation.ok
    assert validation.statistics["variants"] == {"identity": 1}
    assert not any(
        issue.code == "incomplete_variant_family"
        for issue in validation.issues
    )


def test_identity_only_validation_rejects_stale_augmentation_artifacts(
    tmp_path: Path,
) -> None:
    stage = tmp_path / "stage"
    source_path = tmp_path / "source.npz"
    source_path.touch()
    family_root = (
        stage
        / "canonical"
        / "g1"
        / "object_interaction"
        / "amass"
        / "object_quality"
        / "identity-only"
    )
    expected_identity = _write_object_quality_artifact(
        family_root / "identity.npz",
        source_path=source_path,
        variant="identity",
    )
    stale_augmentation = family_root / "trans_0.npz"
    _write_object_quality_artifact(
        stale_augmentation,
        source_path=source_path,
        variant="trans_0",
    )

    validation = validate_rebuild_directory(
        stage,
        (expected_identity,),
    )

    assert not validation.ok
    assert any(
        issue.code == "unexpected_artifact"
        and issue.path == str(stale_augmentation.resolve())
        for issue in validation.issues
    )


@pytest.mark.parametrize(
    ("robot", "expected_enabled"),
    [
        ("g1", True),
        ("e1", False),
    ],
)
def test_rebuild_object_collision_fallback_matches_runner_plan_and_report(
    tmp_path: Path,
    robot: str,
    expected_enabled: bool,
) -> None:
    data_dir = tmp_path / "OMOMO_new"
    data_dir.mkdir()
    _write_omomo_source(data_dir / "sub1_tripod_001.pt")
    plan = BatchPlan(
        robot=robot,
        spec=DATASET_SPEC_BY_ID["omomo_object_interaction"],
        data_dir=data_dir,
    )
    cfg = RebuildDemoResultsConfig(
        data_root=tmp_path,
        results_root=tmp_path / "results",
        release_object_non_penetration_on_infeasible=True,
    )
    stage = tmp_path / "results" / ".staging" / "run" / "v1"

    parallel_config = _parallel_config(
        cfg,
        plan,
        stage=stage,
        batch_report_path=tmp_path / "report.json",
    )
    planned = plan_expected_artifacts(
        (plan,),
        stage=stage,
        cfg=cfg,
    )
    disabled_cfg = replace(
        cfg,
        release_object_non_penetration_on_infeasible=False,
    )
    disabled = plan_expected_artifacts(
        (plan,),
        stage=stage,
        cfg=disabled_cfg,
    )

    assert parallel_config.retargeter.release_object_non_penetration_on_infeasible is expected_enabled
    assert rebuild._release_object_non_penetration_for_plan(cfg, plan) is expected_enabled
    if expected_enabled:
        assert [item.config_sha256 for item in planned] != [item.config_sha256 for item in disabled]
    else:
        assert [item.config_sha256 for item in planned] == [item.config_sha256 for item in disabled]


@pytest.mark.parametrize(
    ("dataset_id", "filename", "encoded_stem"),
    [
        (
            "amass",
            "walk_turn_right_(45)_stageii.npz",
            "walk_turn_right_%2845%29_stageii",
        ),
        (
            "noetix_bvh",
            "breaking+hippop_{F}.npz",
            "breaking%2Bhippop_%7BF%7D",
        ),
    ],
)
def test_expected_plan_encodes_real_source_punctuation_without_collisions(
    tmp_path: Path,
    dataset_id: str,
    filename: str,
    encoded_stem: str,
) -> None:
    data_dir = tmp_path / dataset_id
    data_dir.mkdir()
    source_path = data_dir / filename
    _write_adapter_npz(
        source_path,
        data_format=DATASET_SPEC_BY_ID[dataset_id].data_format,
    )
    plan = BatchPlan(
        robot="g1",
        spec=DATASET_SPEC_BY_ID[dataset_id],
        data_dir=data_dir,
    )

    expected = plan_expected_artifacts((plan,), stage=tmp_path / "stage")

    assert len(expected) == 1
    assert expected[0].output_path.parent.name == encoded_stem


def test_dry_run_plans_all_artifacts_without_calling_workers_or_creating_stage(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "demo_data"
    data_dir = data_root / "OMOMO_new"
    data_dir.mkdir(parents=True)
    _write_omomo_source(data_dir / "sub1_tripod_001.pt")
    results_root = tmp_path / "demo_results"
    cfg = RebuildDemoResultsConfig(
        data_root=data_root,
        results_root=results_root,
        robots=("e1",),
        datasets=("omomo_robot_only", "omomo_object_interaction"),
        dry_run=True,
        omomo_preflight=False,
    )

    with mock.patch("holosoma_retargeting.examples.rebuild_demo_results.parallel_robot_retarget.main") as parallel_main:
        report = run_rebuild(cfg)

    assert report["status"] == "dry_run"
    parallel_main.assert_not_called()
    assert report["include_augmentation_variants"] is True
    assert report["planning"]["ok"]
    assert report["planning"]["include_augmentation_variants"] is True
    assert report["planning"]["effective_augmentation_by_batch"] == {
        "e1-omomo_object_interaction": True,
        "e1-omomo_robot_only": False,
    }
    assert report["planning"]["expected_artifact_count"] == 7
    assert report["planning"]["expected_source_count"] == 1
    assert report["planning"]["expected_source_job_count"] == 2
    assert report["planning"]["output_collision_count"] == 0
    assert {item["status"] for item in report["batches"]} == {"planned"}
    assert not staging_root(results_root, cfg.run_id).exists()
    assert (results_root / "runs" / cfg.run_id / "report.json").is_file()


def test_identity_only_dry_run_preserves_source_jobs_and_records_mode(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "demo_data"
    data_dir = data_root / "OMOMO_new"
    data_dir.mkdir(parents=True)
    _write_omomo_source(data_dir / "sub1_tripod_001.pt")
    results_root = tmp_path / "demo_results"
    cfg = RebuildDemoResultsConfig(
        data_root=data_root,
        results_root=results_root,
        robots=("g1",),
        datasets=("omomo_robot_only", "omomo_object_interaction"),
        include_augmentation_variants=False,
        dry_run=True,
        omomo_preflight=False,
    )

    with mock.patch("holosoma_retargeting.examples.rebuild_demo_results.parallel_robot_retarget.main") as parallel_main:
        report = run_rebuild(cfg)

    assert report["status"] == "dry_run"
    parallel_main.assert_not_called()
    assert report["include_augmentation_variants"] is False
    assert report["planning"]["include_augmentation_variants"] is False
    assert report["planning"]["effective_augmentation_by_batch"] == {
        "g1-omomo_object_interaction": False,
        "g1-omomo_robot_only": False,
    }
    assert report["planning"]["expected_artifact_count"] == 2
    assert report["planning"]["expected_source_count"] == 1
    assert report["planning"]["expected_source_job_count"] == 2
    assert {item["effective_augmentation"] for item in report["batches"]} == {False}
    assert not staging_root(results_root, cfg.run_id).exists()


def test_missing_default_noetix_csv_root_fails_before_worker_execution(
    tmp_path: Path,
) -> None:
    cfg = RebuildDemoResultsConfig(
        data_root=tmp_path / "demo_data",
        results_root=tmp_path / "demo_results",
        robots=("g1",),
        datasets=("noetix_csv_climb",),
        dry_run=True,
    )

    with mock.patch(
        "holosoma_retargeting.examples.rebuild_demo_results.parallel_robot_retarget.main"
    ) as parallel_main, pytest.raises(
        DemoRebuildError,
        match="planning failed",
    ):
        run_rebuild(cfg)

    parallel_main.assert_not_called()
    report_path = cfg.results_root / "runs" / cfg.run_id / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "dry_run_failed"
    assert report["batches"][0]["status"] == "failed"
    assert "noetix_csv_climb" in report["batches"][0]["reason"]
    assert report["planning"]["ok"] is False


def test_planning_failure_prevents_any_worker_execution(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "demo_data"
    (data_root / "amass_smplx_processed").mkdir(parents=True)
    cfg = RebuildDemoResultsConfig(
        data_root=data_root,
        results_root=tmp_path / "demo_results",
        robots=("g1",),
        datasets=("amass",),
    )

    collision_path = tmp_path / "duplicate.npz"
    collision = ArtifactPlanCollisionError(
        (collision_path,),
        expected_artifact_count=2,
        expected_source_count=2,
    )
    planner_patcher = mock.patch(
        "holosoma_retargeting.examples.rebuild_demo_results.plan_expected_artifacts",
        side_effect=collision,
    )
    parallel_patcher = mock.patch("holosoma_retargeting.examples.rebuild_demo_results.parallel_robot_retarget.main")
    with planner_patcher as planner, parallel_patcher as parallel_main, pytest.raises(
        DemoRebuildError,
        match="planning failed",
    ):
        run_rebuild(cfg)

    planner.assert_called_once()
    parallel_main.assert_not_called()
    assert not staging_root(cfg.results_root, cfg.run_id).exists()
    report = json.loads((cfg.results_root / "runs" / cfg.run_id / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "planning_failed"
    assert "duplicate output" in report["planning"]["error"]
    assert report["planning"]["expected_artifact_count"] == 2
    assert report["planning"]["expected_source_count"] == 2
    assert report["planning"]["output_collision_count"] == 1
    assert report["planning"]["collision_paths"] == [str(collision_path)]


def test_validation_stats_and_successful_promotion_archive_existing_v1(
    tmp_path: Path,
) -> None:
    results_root = tmp_path / "demo_results"
    run_id = "test-run"
    stage = staging_root(results_root, run_id)
    source_path = tmp_path / "source.npz"
    source_path.touch()
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    artifact_path = (
        stage / "canonical" / "g1" / "robot_only" / "amass" / "amass_smplx_processed" / "walk" / "identity.npz"
    )
    expected = _write_valid_artifact(
        artifact_path,
        source_path=source_path,
        source_sha256=source_hash,
        config_json=json.dumps({"test": True}, sort_keys=True),
    )
    formal = results_root / "v1"
    formal.mkdir(parents=True)
    (formal / "old-result.txt").write_text("old", encoding="utf-8")
    legacy_sibling = results_root / "demo_results_ablation"
    legacy_sibling.mkdir(parents=True)
    (legacy_sibling / "keep.txt").write_text("keep", encoding="utf-8")

    validation = validate_rebuild_directory(stage, (expected,))
    promotion = promote_staging(results_root, run_id, validation)

    assert validation.ok
    assert validation.statistics["human_orientation"]["amass"]["present"] == 1
    assert validation.statistics["interaction_mesh"]["present"] == 1
    assert validation.statistics["robot_link_counts"] == {"1": 1}
    assert validation.statistics["constraint_relaxations"] == {
        "foot_sticking_fallback_frames": {
            "artifact_count": 0,
            "frame_count": 0,
        },
        "foot_sticking_release_frames": {
            "artifact_count": 0,
            "frame_count": 0,
        },
        "object_non_penetration_release_frames": {
            "artifact_count": 0,
            "frame_count": 0,
        },
        "foot_sticking_full_sequence_retry_frame": {
            "artifact_count": 0,
            "frame_count": 0,
        },
    }
    assert promotion.promoted_root == formal
    assert artifact_path.name == "identity.npz"
    assert (
        formal / "canonical" / "g1" / "robot_only" / "amass" / "amass_smplx_processed" / "walk" / "identity.npz"
    ).is_file()
    assert promotion.archived_root is not None
    assert (promotion.archived_root / "old-result.txt").read_text() == "old"
    assert (legacy_sibling / "keep.txt").read_text() == "keep"


@pytest.mark.parametrize(
    ("crash_status", "write_status_first", "interrupted_status"),
    [
        ("exchanged", False, "prepared"),
        ("exchanged", True, "exchanged"),
        ("filesystem_promoted", False, "exchanged"),
    ],
)
def test_promotion_recovers_hard_crash_across_exchange_and_archive_windows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crash_status: str,
    write_status_first: bool,
    interrupted_status: str,
) -> None:
    run_id = f"{crash_status}-{write_status_first}"
    results_root, stage, formal, validation = _validated_promotion_tree(
        tmp_path,
        run_id=run_id,
    )
    real_write_journal = rebuild._write_promotion_journal
    crashed = False

    def _crash_before_exchange_status(
        path: Path,
        payload: dict[str, Any],
        *,
        status: str,
    ) -> dict[str, Any]:
        nonlocal crashed
        if status == crash_status and not crashed:
            crashed = True
            if write_status_first:
                real_write_journal(path, payload, status=status)
            raise SimulatedHardCrash
        return real_write_journal(path, payload, status=status)

    monkeypatch.setattr(
        rebuild,
        "_write_promotion_journal",
        _crash_before_exchange_status,
    )
    with pytest.raises(SimulatedHardCrash):
        promote_staging(results_root, run_id, validation)

    journal_path = rebuild.promotion_journal_path(results_root, run_id)
    interrupted_journal = json.loads(
        journal_path.read_text(encoding="utf-8"),
    )
    archived_root = Path(interrupted_journal["archived_root"])
    assert interrupted_journal["status"] == interrupted_status
    assert not (formal / "old-result.txt").exists()
    if crash_status == "filesystem_promoted":
        assert not stage.exists()
        assert (archived_root / "old-result.txt").read_text(encoding="utf-8") == "old"
    else:
        assert (stage / "old-result.txt").read_text(encoding="utf-8") == "old"
        assert not archived_root.exists()

    promotion = promote_staging(results_root, run_id, validation)

    assert promotion.archived_root == archived_root
    assert not stage.exists()
    assert (archived_root / "old-result.txt").read_text(encoding="utf-8") == "old"
    recovered_journal = json.loads(
        journal_path.read_text(encoding="utf-8"),
    )
    assert recovered_journal["status"] in {
        "filesystem_promoted",
        "completed",
    }

    repeated = promote_staging(results_root, run_id, validation)

    assert repeated.archived_root == archived_root
    assert not stage.exists()
    assert len(tuple((results_root / "archive").glob("*/v1"))) == 1


def test_promotion_rejects_ambiguous_tree_after_interrupted_exchange(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "ambiguous-exchange"
    results_root, stage, formal, validation = _validated_promotion_tree(
        tmp_path,
        run_id=run_id,
    )
    real_write_journal = rebuild._write_promotion_journal

    def _crash_before_exchange_status(
        path: Path,
        payload: dict[str, Any],
        *,
        status: str,
    ) -> dict[str, Any]:
        if status == "exchanged":
            raise SimulatedHardCrash
        return real_write_journal(path, payload, status=status)

    monkeypatch.setattr(
        rebuild,
        "_write_promotion_journal",
        _crash_before_exchange_status,
    )
    with pytest.raises(SimulatedHardCrash):
        promote_staging(results_root, run_id, validation)
    (stage / "ambiguous.txt").write_text("tampered", encoding="utf-8")

    with pytest.raises(DemoRebuildError, match="ambiguous"):
        promote_staging(results_root, run_id, validation)

    assert formal.is_dir()
    assert stage.is_dir()
    assert (stage / "ambiguous.txt").read_text(encoding="utf-8") == "tampered"


def test_final_report_failure_is_journaled_and_next_run_converges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "report-write-crash"
    results_root, stage, formal, validation = _validated_promotion_tree(
        tmp_path,
        run_id=run_id,
    )
    report_path = run_report_root(results_root, run_id) / "report.json"
    report_payload = _promotion_report_payload(
        run_id=run_id,
        results_root=results_root,
        stage=stage,
        validation=validation,
    )
    real_atomic_write = rebuild._atomic_write_json

    def _fail_final_report(
        path: Path,
        payload: dict[str, Any],
    ) -> None:
        if path == report_path and payload.get("status") == "promoted":
            raise OSError("injected final-report failure")
        real_atomic_write(path, payload)

    monkeypatch.setattr(
        rebuild,
        "_atomic_write_json",
        _fail_final_report,
    )
    with pytest.raises(OSError, match="final-report failure"):
        promote_staging(
            results_root,
            run_id,
            validation,
            report_path=report_path,
            report_payload=report_payload,
        )

    journal_path = rebuild.promotion_journal_path(results_root, run_id)
    interrupted_journal = json.loads(
        journal_path.read_text(encoding="utf-8"),
    )
    archived_root = Path(interrupted_journal["archived_root"])
    assert interrupted_journal["status"] == "filesystem_promoted"
    assert interrupted_journal["final_report"]["status"] == "promoted"
    assert not stage.exists()
    assert formal.is_dir()
    assert (archived_root / "old-result.txt").read_text(encoding="utf-8") == "old"
    assert not report_path.exists()

    monkeypatch.setattr(rebuild, "_atomic_write_json", real_atomic_write)
    recovered = run_rebuild(
        RebuildDemoResultsConfig(
            data_root=tmp_path / "unused-data",
            results_root=results_root,
            run_id=run_id,
            promote=True,
        )
    )

    assert recovered["status"] == "promoted"
    persisted_report = json.loads(report_path.read_text(encoding="utf-8"))
    completed_journal = json.loads(
        journal_path.read_text(encoding="utf-8"),
    )
    assert persisted_report == interrupted_journal["final_report"]
    assert completed_journal["status"] == "completed"

    repeated = run_rebuild(
        RebuildDemoResultsConfig(
            data_root=tmp_path / "still-unused-data",
            results_root=results_root,
            run_id=run_id,
            promote=True,
        )
    )

    assert repeated == persisted_report
    assert len(tuple((results_root / "archive").glob("*/v1"))) == 1


def test_hard_crash_after_final_report_write_is_idempotently_recovered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "post-report-crash"
    results_root, stage, _formal, validation = _validated_promotion_tree(
        tmp_path,
        run_id=run_id,
    )
    report_path = run_report_root(results_root, run_id) / "report.json"
    report_payload = _promotion_report_payload(
        run_id=run_id,
        results_root=results_root,
        stage=stage,
        validation=validation,
    )
    real_write_journal = rebuild._write_promotion_journal
    crashed = False

    def _crash_before_completed_status(
        path: Path,
        payload: dict[str, Any],
        *,
        status: str,
    ) -> dict[str, Any]:
        nonlocal crashed
        if status == "completed" and not crashed:
            crashed = True
            raise SimulatedHardCrash
        return real_write_journal(path, payload, status=status)

    monkeypatch.setattr(
        rebuild,
        "_write_promotion_journal",
        _crash_before_completed_status,
    )
    with pytest.raises(SimulatedHardCrash):
        promote_staging(
            results_root,
            run_id,
            validation,
            report_path=report_path,
            report_payload=report_payload,
        )

    journal_path = rebuild.promotion_journal_path(results_root, run_id)
    interrupted_journal = json.loads(
        journal_path.read_text(encoding="utf-8"),
    )
    persisted_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert interrupted_journal["status"] == "filesystem_promoted"
    assert persisted_report == interrupted_journal["final_report"]

    recovered = run_rebuild(
        RebuildDemoResultsConfig(
            data_root=tmp_path / "unused-data",
            results_root=results_root,
            run_id=run_id,
            promote=True,
        )
    )

    assert recovered == persisted_report
    completed_journal = json.loads(
        journal_path.read_text(encoding="utf-8"),
    )
    assert completed_journal["status"] == "completed"


def test_failed_validation_never_moves_formal_or_staging(tmp_path: Path) -> None:
    results_root = tmp_path / "demo_results"
    run_id = "test-run"
    stage = staging_root(results_root, run_id)
    invalid_path = stage / "canonical" / "bad.npz"
    invalid_path.parent.mkdir(parents=True)
    np.savez(invalid_path, qpos=np.zeros((1, 7)))
    formal = results_root / "v1"
    formal.mkdir(parents=True)
    marker = formal / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    validation = validate_rebuild_directory(stage, ())

    assert not validation.ok
    with pytest.raises(DemoRebuildError, match="Refusing"):
        promote_staging(results_root, run_id, validation)
    assert marker.read_text() == "keep"
    assert invalid_path.is_file()


def test_promotion_rejects_staging_tree_changed_after_validation(
    tmp_path: Path,
) -> None:
    results_root = tmp_path / "demo_results"
    run_id = "test-run"
    stage = staging_root(results_root, run_id)
    source_path = tmp_path / "source.npz"
    source_path.touch()
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    artifact_path = (
        stage / "canonical" / "g1" / "robot_only" / "amass" / "amass_smplx_processed" / "walk" / "identity.npz"
    )
    expected = _write_valid_artifact(
        artifact_path,
        source_path=source_path,
        source_sha256=source_hash,
        config_json=json.dumps({"test": True}, sort_keys=True),
    )
    formal = results_root / "v1"
    formal.mkdir(parents=True)
    marker = formal / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    validation = validate_rebuild_directory(stage, (expected,))
    assert validation.ok

    (stage / "changed-after-validation.txt").write_text(
        "changed",
        encoding="utf-8",
    )

    with pytest.raises(
        DemoRebuildError,
        match="changed after validation",
    ):
        promote_staging(results_root, run_id, validation)
    assert marker.read_text(encoding="utf-8") == "keep"
    assert artifact_path.is_file()


def test_promotion_fails_closed_when_atomic_exchange_is_unavailable(
    tmp_path: Path,
) -> None:
    results_root = tmp_path / "demo_results"
    run_id = "no-atomic-exchange"
    stage = staging_root(results_root, run_id)
    source_path = tmp_path / "source.npz"
    source_path.touch()
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    artifact_path = (
        stage / "canonical" / "g1" / "robot_only" / "amass" / "amass_smplx_processed" / "walk" / "identity.npz"
    )
    expected = _write_valid_artifact(
        artifact_path,
        source_path=source_path,
        source_sha256=source_sha256,
        config_json=json.dumps({"promotion": "fail-closed"}, sort_keys=True),
    )
    formal = results_root / "v1"
    formal.mkdir(parents=True)
    formal_marker = formal / "formal-marker.txt"
    formal_marker.write_bytes(b"formal-before")
    stage_before = {path.relative_to(stage): path.read_bytes() for path in stage.rglob("*") if path.is_file()}
    formal_before = {path.relative_to(formal): path.read_bytes() for path in formal.rglob("*") if path.is_file()}
    validation = validate_rebuild_directory(stage, (expected,))
    assert validation.ok

    with mock.patch(  # noqa: SIM117 - Python 3.8 lacks parenthesized contexts.
        "holosoma_retargeting.examples.rebuild_demo_results._exchange_directories",
        return_value=False,
    ) as exchange_directories:
        with pytest.raises(
            DemoRebuildError,
            match="cannot atomically exchange",
        ):
            promote_staging(results_root, run_id, validation)

    exchange_directories.assert_called_once_with(stage, formal)
    assert {path.relative_to(stage): path.read_bytes() for path in stage.rglob("*") if path.is_file()} == stage_before
    assert {
        path.relative_to(formal): path.read_bytes() for path in formal.rglob("*") if path.is_file()
    } == formal_before


def test_promotion_rejects_journal_status_ahead_of_filesystem(
    tmp_path: Path,
) -> None:
    run_id = "status-regression"
    results_root, stage, formal, validation = _validated_promotion_tree(
        tmp_path,
        run_id=run_id,
    )
    with mock.patch(
        "holosoma_retargeting.examples.rebuild_demo_results._exchange_directories",
        return_value=False,
    ), pytest.raises(DemoRebuildError, match="cannot atomically exchange"):
        promote_staging(results_root, run_id, validation)

    journal_path = rebuild.promotion_journal_path(results_root, run_id)
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    journal["status"] = "completed"
    journal_path.write_text(
        f"{json.dumps(journal, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )

    with mock.patch(
        "holosoma_retargeting.examples.rebuild_demo_results._exchange_directories",
    ) as exchange_directories, pytest.raises(
        DemoRebuildError,
        match="regressed",
    ):
        promote_staging(results_root, run_id, validation)

    exchange_directories.assert_not_called()
    assert stage.is_dir()
    assert (formal / "old-result.txt").read_text(encoding="utf-8") == "old"


def test_non_artifact_file_makes_staging_tree_invalid(
    tmp_path: Path,
) -> None:
    results_root = tmp_path / "demo_results"
    stage = staging_root(results_root, "test-run")
    source_path = tmp_path / "source.npz"
    source_path.touch()
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    artifact_path = (
        stage / "canonical" / "g1" / "robot_only" / "amass" / "amass_smplx_processed" / "walk" / "identity.npz"
    )
    expected = _write_valid_artifact(
        artifact_path,
        source_path=source_path,
        source_sha256=source_hash,
        config_json=json.dumps({"test": True}, sort_keys=True),
    )
    (stage / "partial.tmp").write_text("partial", encoding="utf-8")

    validation = validate_rebuild_directory(stage, (expected,))

    assert not validation.ok
    assert validation.tree_file_count == 2
    assert any(issue.code == "unexpected_file" for issue in validation.issues)


def test_constraint_relaxations_are_reported_without_failing_validation(
    tmp_path: Path,
) -> None:
    stage = tmp_path / "stage"
    source_path = tmp_path / "source.npz"
    source_path.touch()
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    artifact_path = (
        stage / "canonical" / "g1" / "robot_only" / "amass" / "amass_smplx_processed" / "walk" / "identity.npz"
    )
    expected = _write_valid_artifact(
        artifact_path,
        source_path=source_path,
        source_sha256=source_hash,
        config_json=json.dumps({"fallback": True}, sort_keys=True),
    )
    with np.load(artifact_path, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    payload.update(
        {
            "foot_sticking_fallback_tolerance": np.asarray(0.1),
            "foot_sticking_fallback_frames": np.asarray([0], dtype=np.int32),
            "release_foot_sticking_on_infeasible": np.asarray(True),
            "foot_sticking_release_frames": np.asarray([0], dtype=np.int32),
            "foot_sticking_enabled_for_saved_trajectory": np.asarray(True),
            "foot_sticking_states": np.asarray([[True, False]]),
            "constraint_mode_foot_sticking": np.asarray(["released"]),
        }
    )
    write_result_artifact(artifact_path, payload)

    validation = validate_rebuild_directory(stage, (expected,))

    assert validation.ok
    assert validation.statistics["constraint_relaxations"] == {
        "foot_sticking_fallback_frames": {
            "artifact_count": 1,
            "frame_count": 1,
        },
        "foot_sticking_release_frames": {
            "artifact_count": 1,
            "frame_count": 1,
        },
        "object_non_penetration_release_frames": {
            "artifact_count": 0,
            "frame_count": 0,
        },
        "foot_sticking_full_sequence_retry_frame": {
            "artifact_count": 0,
            "frame_count": 0,
        },
    }

    gated_validation = validate_rebuild_directory(
        stage,
        (expected,),
        max_fallback_frame_fraction_per_artifact=0.10,
        max_release_union_frame_fraction_per_artifact=0.10,
        max_full_sequence_retry_fraction_per_artifact=0.10,
        max_fallback_artifact_fraction_per_group=0.10,
        max_foot_release_artifact_fraction_per_group=0.10,
        max_object_release_artifact_fraction_per_group=0.10,
        max_release_union_artifact_fraction_per_group=0.10,
        max_full_sequence_retry_artifact_fraction_per_group=0.10,
    )

    assert not gated_validation.ok
    assert {issue.code for issue in gated_validation.issues} == {
        "quality_group_fraction",
        "quality_per_artifact_fraction",
    }
    assert gated_validation.statistics["quality_gates"]["per_artifact"] == {
        "checked_artifact_count": {
            "foot_sticking_fallback": 1,
            "release_frame_union": 1,
            "foot_sticking_full_sequence_retry": 1,
        },
        "exceeded_artifact_count": {
            "foot_sticking_fallback": 1,
            "release_frame_union": 1,
            "foot_sticking_full_sequence_retry": 0,
        },
    }


def test_grouped_fallback_gate_uses_only_foot_enabled_denominator(
    tmp_path: Path,
) -> None:
    stage = tmp_path / "stage"
    expected: list[ExpectedArtifact] = []
    for index in range(5):
        source_path = tmp_path / f"source-{index}.npz"
        source_path.touch()
        source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        artifact_path = (
            stage
            / "canonical"
            / "g1"
            / "robot_only"
            / "amass"
            / "amass_smplx_processed"
            / f"walk-{index}"
            / "identity.npz"
        )
        expected.append(
            _write_valid_artifact(
                artifact_path,
                source_path=source_path,
                source_sha256=source_hash,
                config_json=json.dumps(
                    {"foot_eligible": index == 0},
                    sort_keys=True,
                ),
                orientation=False,
            )
        )
        _rewrite_quality_fields(
            artifact_path,
            foot_enabled=index == 0,
            fallback_frames=((0,) if index == 0 else ()),
        )

    validation = validate_rebuild_directory(
        stage,
        expected,
        max_fallback_artifact_fraction_per_group=0.5,
    )

    assert not validation.ok
    reports = validation.statistics["quality_gates"]["groups"]["robot_dataset_task"]
    assert len(reports) == 1
    fallback = reports[0]["metrics"]["foot_sticking_fallback"]
    assert fallback == {
        "affected_artifact_numerator": 1,
        "eligible_artifact_denominator": 1,
        "affected_artifact_fraction": 1.0,
        "affected_frame_numerator": 1,
        "eligible_frame_denominator": 1,
        "affected_frame_fraction": 1.0,
    }
    assert {
        item["level"]
        for item in validation.statistics["quality_gates"]["exceeded_groups"]
        if item["metric"] == "foot_sticking_fallback"
    } == {
        "robot_dataset_task",
        "robot_dataset_task_variant",
    }


def test_full_sequence_retry_remains_eligible_after_saved_foot_disable(
    tmp_path: Path,
) -> None:
    stage = tmp_path / "stage"
    source_path = tmp_path / "retry-source.npz"
    source_path.touch()
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    artifact_path = (
        stage / "canonical" / "g1" / "robot_only" / "amass" / "amass_smplx_processed" / "retry" / "identity.npz"
    )
    expected = _write_valid_artifact(
        artifact_path,
        source_path=source_path,
        source_sha256=source_hash,
        config_json=json.dumps({"retry": True}, sort_keys=True),
        orientation=False,
    )
    _rewrite_quality_fields(
        artifact_path,
        foot_enabled=False,
        full_sequence_retry_frame=0,
    )

    validation = validate_rebuild_directory(
        stage,
        (expected,),
        max_full_sequence_retry_artifact_fraction_per_group=0.5,
    )

    assert not validation.ok
    (report,) = validation.statistics["quality_gates"]["groups"]["robot_dataset_task"]
    retry = report["metrics"]["foot_sticking_full_sequence_retry"]
    assert retry["affected_artifact_numerator"] == 1
    assert retry["eligible_artifact_denominator"] == 1
    fallback = report["metrics"]["foot_sticking_fallback"]
    assert fallback["eligible_artifact_denominator"] == 0


def test_object_release_denominator_requires_actual_release_eligibility(
    tmp_path: Path,
) -> None:
    stage = tmp_path / "stage"
    source_path = tmp_path / "object-source.npz"
    source_path.touch()
    expected: list[ExpectedArtifact] = []
    variants = (
        "identity",
        "trans_0",
        "trans_1",
        "trans_2",
        "rot_0",
        "rot_1",
    )
    for index, variant in enumerate(variants):
        artifact_path = (
            stage
            / "canonical"
            / "g1"
            / "object_interaction"
            / "amass"
            / "object_quality"
            / "object-sequence"
            / f"{variant}.npz"
        )
        expected.append(
            _write_object_quality_artifact(
                artifact_path,
                source_path=source_path,
                variant=variant,
            )
        )
        if index:
            _rewrite_quality_fields(
                artifact_path,
                object_release_enabled=False,
            )

    validation = validate_rebuild_directory(stage, expected)

    assert validation.ok
    (report,) = validation.statistics["quality_gates"]["groups"]["robot_dataset_task"]
    object_release = report["metrics"]["object_non_penetration_release"]
    assert object_release["eligible_artifact_denominator"] == 1
    assert object_release["affected_artifact_numerator"] == 0


def test_climbing_static_terrain_release_is_quality_eligible(
    tmp_path: Path,
) -> None:
    stage = tmp_path / "stage"
    source_path = tmp_path / "climbing-source.npz"
    source_path.touch()
    artifact_path = (
        stage / "canonical" / "g1" / "climbing" / "amass" / "climb_quality" / "climb-sequence" / "identity.npz"
    )
    variants = (
        "identity",
        "z_scale_0p8",
        "z_scale_0p9",
        "z_scale_1p1",
        "z_scale_1p2",
    )
    expected = tuple(
        _write_climbing_quality_artifact(
            artifact_path.with_name(f"{variant}.npz"),
            source_path=source_path,
            variant=variant,
            object_release_frames=((0,) if variant == "identity" else ()),
        )
        for variant in variants
    )

    validation = validate_rebuild_directory(stage, expected)

    assert validation.ok
    (report,) = validation.statistics["quality_gates"]["groups"]["robot_dataset_task"]
    object_release = report["metrics"]["object_non_penetration_release"]
    assert object_release == {
        "affected_artifact_numerator": 1,
        "eligible_artifact_denominator": 5,
        "affected_artifact_fraction": 0.2,
        "affected_frame_numerator": 1,
        "eligible_frame_denominator": 5,
        "affected_frame_fraction": 0.2,
    }
    assert not any(issue.code == "quality_ineligible_event" for issue in validation.issues)


def test_robot_only_artifact_cannot_claim_object_release_frames(
    tmp_path: Path,
) -> None:
    stage = tmp_path / "stage"
    source_path = tmp_path / "robot-only-source.npz"
    source_path.touch()
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    artifact_path = (
        stage / "canonical" / "g1" / "robot_only" / "amass" / "amass_smplx_processed" / "walk" / "identity.npz"
    )
    expected = _write_valid_artifact(
        artifact_path,
        source_path=source_path,
        source_sha256=source_sha256,
        config_json=json.dumps({"quality": "robot-only"}, sort_keys=True),
        orientation=False,
    )
    with np.load(artifact_path, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    payload["release_object_non_penetration_on_infeasible"] = np.asarray(True)
    payload["object_non_penetration_release_frames"] = np.asarray(
        [0],
        dtype=np.int32,
    )
    np.savez_compressed(artifact_path, **payload)

    validation = validate_rebuild_directory(stage, (expected,))

    assert not validation.ok
    assert validation.statistics["valid_artifacts"] == 0
    assert any(
        issue.code == "invalid_artifact" and "object non-penetration was not eligible" in issue.message
        for issue in validation.issues
    )


def test_variant_group_gate_prevents_cross_variant_quality_dilution(
    tmp_path: Path,
) -> None:
    stage = tmp_path / "stage"
    expected: list[ExpectedArtifact] = []
    variants = (
        "identity",
        "trans_0",
        "trans_1",
        "trans_2",
        "rot_0",
        "rot_1",
    )
    for source_index in range(4):
        source_path = tmp_path / f"object-source-{source_index}.npz"
        source_path.touch()
        for variant in variants:
            artifact_path = (
                stage
                / "canonical"
                / "g1"
                / "object_interaction"
                / "amass"
                / "object_quality"
                / f"object-{source_index}"
                / f"{variant}.npz"
            )
            expected.append(
                _write_object_quality_artifact(
                    artifact_path,
                    source_path=source_path,
                    variant=variant,
                )
            )
            if source_index == 0 and variant == "identity":
                _rewrite_quality_fields(
                    artifact_path,
                    object_release_frames=(0,),
                )

    validation = validate_rebuild_directory(
        stage,
        expected,
        max_object_release_artifact_fraction_per_group=0.10,
        max_release_union_artifact_fraction_per_group=0.10,
    )

    assert not validation.ok
    batch_reports = validation.statistics["quality_gates"]["groups"]["robot_dataset_task"]
    (batch_report,) = batch_reports
    batch_object_release = batch_report["metrics"]["object_non_penetration_release"]
    assert batch_object_release["affected_artifact_numerator"] == 1
    assert batch_object_release["eligible_artifact_denominator"] == 24
    assert batch_report["exceeded_gates"] == []

    exceeded = validation.statistics["quality_gates"]["exceeded_groups"]
    assert {
        (
            item["level"],
            item["group"].get("variant"),
            item["metric"],
            item["affected_artifact_numerator"],
            item["eligible_artifact_denominator"],
        )
        for item in exceeded
    } == {
        (
            "robot_dataset_task_variant",
            "identity",
            "object_non_penetration_release",
            1,
            4,
        ),
        (
            "robot_dataset_task_variant",
            "identity",
            "release_frame_union",
            1,
            4,
        ),
    }


def test_validation_rejects_missing_direct_human_orientation(
    tmp_path: Path,
) -> None:
    stage = tmp_path / "stage"
    source_path = tmp_path / "source.npz"
    source_path.touch()
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    artifact_path = (
        stage / "canonical" / "g1" / "robot_only" / "amass" / "amass_smplx_processed" / "walk" / "identity.npz"
    )
    expected = _write_valid_artifact(
        artifact_path,
        source_path=source_path,
        source_sha256=source_hash,
        config_json=json.dumps({"orientation": "absent"}, sort_keys=True),
        orientation=False,
    )
    expected = replace(
        expected,
        human_orientation_joint_names=("Pelvis",),
        human_orientation_source="direct_local_rotation_fk",
        human_orientation_sha256=(
            _direct_orientation_tensor_sha256(
                ("Pelvis",),
                np.asarray(
                    [[[1.0, 0.0, 0.0, 0.0]]],
                    dtype=np.float32,
                ),
            )
        ),
        human_orientation_tensor_shape=(1, 1, 4),
        human_orientation_tensor_bytes=np.asarray(
            [[[1.0, 0.0, 0.0, 0.0]]],
            dtype=np.float32,
        ).tobytes(order="C"),
    )

    validation = validate_rebuild_directory(stage, (expected,))

    assert not validation.ok
    assert any(
        issue.code == "invalid_artifact" and "omits human orientation fields" in issue.message
        for issue in validation.issues
    )


def test_validation_rejects_self_consistent_but_wrong_orientation_tensor(
    tmp_path: Path,
) -> None:
    stage = tmp_path / "stage"
    source_path = tmp_path / "source.npz"
    source_path.touch()
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    artifact_path = (
        stage / "canonical" / "g1" / "robot_only" / "amass" / "amass_smplx_processed" / "walk" / "identity.npz"
    )
    expected = _write_valid_artifact(
        artifact_path,
        source_path=source_path,
        source_sha256=source_hash,
        config_json=json.dumps({"orientation": "wrong-value"}, sort_keys=True),
    )
    with np.load(artifact_path, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    wrong_tensor = np.asarray(
        [[[0.0, 1.0, 0.0, 0.0]]],
        dtype=np.float32,
    )
    payload["human_orientation_quaternions_wxyz"] = wrong_tensor
    payload["human_orientation_sha256"] = np.asarray(_direct_orientation_tensor_sha256(("Pelvis",), wrong_tensor))
    write_result_artifact(artifact_path, payload)

    validation = validate_rebuild_directory(stage, (expected,))

    assert not validation.ok
    assert any(
        issue.code == "invalid_artifact" and "production adapter's frozen named tensor digest" in issue.message
        for issue in validation.issues
    )


def test_validation_compares_orientation_values_and_bytes_even_when_sha_matches(
    tmp_path: Path,
) -> None:
    stage = tmp_path / "stage"
    source_path = tmp_path / "source.npz"
    source_path.touch()
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    artifact_path = (
        stage / "canonical" / "g1" / "robot_only" / "amass" / "amass_smplx_processed" / "walk" / "identity.npz"
    )
    expected = _write_valid_artifact(
        artifact_path,
        source_path=source_path,
        source_sha256=source_hash,
        config_json=json.dumps({"orientation": "bytes"}, sort_keys=True),
    )
    byte_distinct_equal_values = np.asarray(
        [[[1.0, -0.0, 0.0, 0.0]]],
        dtype=np.float32,
    ).tobytes(order="C")
    expected = replace(
        expected,
        human_orientation_tensor_bytes=byte_distinct_equal_values,
    )

    validation = validate_rebuild_directory(stage, (expected,))

    assert not validation.ok
    assert any(
        issue.code == "invalid_artifact" and "bytes do not exactly match" in issue.message
        for issue in validation.issues
    )

    value_distinct_expected = replace(
        expected,
        human_orientation_tensor_bytes=np.asarray(
            [[[0.0, 1.0, 0.0, 0.0]]],
            dtype=np.float32,
        ).tobytes(order="C"),
    )
    value_validation = validate_rebuild_directory(
        stage,
        (value_distinct_expected,),
    )

    assert not value_validation.ok
    assert any(
        issue.code == "invalid_artifact" and "values do not exactly match" in issue.message
        for issue in value_validation.issues
    )


def test_validation_rejects_orientation_group_for_adapter_without_direct_data(
    tmp_path: Path,
) -> None:
    stage = tmp_path / "stage"
    source_path = tmp_path / "source.npz"
    source_path.touch()
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    artifact_path = (
        stage / "canonical" / "g1" / "robot_only" / "amass" / "amass_smplx_processed" / "walk" / "identity.npz"
    )
    expected = _write_valid_artifact(
        artifact_path,
        source_path=source_path,
        source_sha256=source_hash,
        config_json=json.dumps({"orientation": "unexpected"}, sort_keys=True),
    )
    expected = replace(
        expected,
        human_orientation_joint_names=None,
        human_orientation_source="absent",
        human_orientation_sha256=None,
        human_orientation_tensor_shape=None,
        human_orientation_tensor_bytes=None,
    )

    validation = validate_rebuild_directory(stage, (expected,))

    assert not validation.ok
    assert any(
        issue.code == "invalid_artifact" and "adapter has no direct orientations" in issue.message
        for issue in validation.issues
    )


def test_validation_rejects_truncated_robot_link_tree(
    tmp_path: Path,
) -> None:
    stage = tmp_path / "stage"
    source_path = tmp_path / "source.npz"
    source_path.touch()
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    artifact_path = (
        stage / "canonical" / "g1" / "robot_only" / "amass" / "amass_smplx_processed" / "walk" / "identity.npz"
    )
    expected = _write_valid_artifact(
        artifact_path,
        source_path=source_path,
        source_sha256=source_hash,
        config_json=json.dumps({"robot": "truncated"}, sort_keys=True),
    )
    expected = replace(
        expected,
        robot_link_names=("pelvis_link", "child_link"),
        robot_link_parent_indices=(-1, 0),
    )

    validation = validate_rebuild_directory(stage, (expected,))

    assert not validation.ok
    assert any(
        issue.code == "invalid_artifact" and "complete G1 MuJoCo subtree" in issue.message
        for issue in validation.issues
    )


def test_validation_rejects_wrong_canonical_human_parent_tree(
    tmp_path: Path,
) -> None:
    stage = tmp_path / "stage"
    source_path = tmp_path / "source.npz"
    source_path.touch()
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    artifact_path = (
        stage / "canonical" / "g1" / "robot_only" / "amass" / "amass_smplx_processed" / "walk" / "identity.npz"
    )
    expected = _write_valid_artifact(
        artifact_path,
        source_path=source_path,
        source_sha256=source_hash,
        config_json=json.dumps({"human_tree": "wrong"}, sort_keys=True),
        orientation=False,
    )
    with np.load(artifact_path, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    human_joints = np.asarray(payload["human_joints"])
    payload["human_joints"] = np.concatenate(
        (human_joints, human_joints + np.float32(0.1)),
        axis=1,
    )
    payload["human_joint_names"] = np.asarray(["Pelvis", "Child"])
    payload["human_joint_parent_indices"] = np.asarray([1, -1], dtype=np.int32)
    write_result_artifact(artifact_path, payload)
    expected = replace(
        expected,
        human_joint_names=("Pelvis", "Child"),
        human_joint_parent_indices=(-1, 0),
    )

    validation = validate_rebuild_directory(stage, (expected,))

    assert not validation.ok
    assert any(
        issue.code == "invalid_artifact" and "canonical adapter topology" in issue.message
        for issue in validation.issues
    )


def test_valid_but_unexpected_artifact_blocks_promotion(tmp_path: Path) -> None:
    results_root = tmp_path / "demo_results"
    run_id = "test-run"
    stage = staging_root(results_root, run_id)
    source_path = tmp_path / "removed-source.npz"
    source_path.touch()
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    artifact_path = (
        stage / "canonical" / "g1" / "robot_only" / "amass" / "amass_smplx_processed" / "removed" / "identity.npz"
    )
    _write_valid_artifact(
        artifact_path,
        source_path=source_path,
        source_sha256=source_hash,
        config_json=json.dumps({"removed": True}, sort_keys=True),
    )

    validation = validate_rebuild_directory(stage, ())

    assert not validation.ok
    assert validation.unexpected_artifact_count == 1
    assert any(issue.code == "unexpected_artifact" for issue in validation.issues)
    with pytest.raises(DemoRebuildError, match="Refusing"):
        promote_staging(results_root, run_id, validation)


def test_same_run_id_is_rejected_before_report_or_staging_mutation(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "demo_data"
    _touch_dataset_roots(data_root)
    results_root = tmp_path / "demo_results"
    run_id = "locked-run"
    report_path = run_report_root(results_root, run_id) / "report.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_bytes(b"existing-success-report")
    lock_path = results_root / ".locks" / f"rebuild-{run_id}.lock"
    lock_path.parent.mkdir(parents=True)
    process_context = multiprocessing.get_context("fork")
    ready = process_context.Event()
    release = process_context.Event()
    lock_holder = process_context.Process(
        target=_hold_rebuild_lock_until_released,
        args=(str(lock_path), ready, release),
    )
    lock_holder.start()
    try:
        assert ready.wait(timeout=10.0)
        with pytest.raises(DemoRebuildError, match="already owns run_id"):
            run_rebuild(
                RebuildDemoResultsConfig(
                    data_root=data_root,
                    results_root=results_root,
                    run_id=run_id,
                    dry_run=True,
                )
            )
    finally:
        release.set()
        lock_holder.join(timeout=10.0)
        if lock_holder.is_alive():
            lock_holder.kill()
            lock_holder.join()

    assert lock_holder.exitcode == 0
    assert report_path.read_bytes() == b"existing-success-report"
    assert not staging_root(results_root, run_id).exists()


def test_promotion_rejects_object_urdf_changed_after_validation(
    tmp_path: Path,
) -> None:
    results_root = tmp_path / "demo_results"
    run_id = "asset-tamper"
    stage = staging_root(results_root, run_id)
    source_path = tmp_path / "object-source.npz"
    source_path.touch()
    expected = [
        _write_object_quality_artifact(
            stage
            / "canonical"
            / "g1"
            / "object_interaction"
            / "amass"
            / "object_quality"
            / "object-sequence"
            / f"{variant}.npz",
            source_path=source_path,
            variant=variant,
        )
        for variant in (
            "identity",
            "trans_0",
            "trans_1",
            "trans_2",
            "rot_0",
            "rot_1",
        )
    ]
    validation = validate_rebuild_directory(stage, expected)
    assert validation.ok
    object_urdf = source_path.parent / "test-object.urdf"
    object_urdf.write_text(
        "<robot name='tampered-object'/>\n",
        encoding="utf-8",
    )

    with pytest.raises(
        DemoRebuildError,
        match="External object assets",
    ):
        promote_staging(results_root, run_id, validation)

    assert stage.is_dir()
    assert not (results_root / "v1").exists()
