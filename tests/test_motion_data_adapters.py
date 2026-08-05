# ruff: noqa: CPY001, PT009, PT027

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "holosoma_retargeting"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from holosoma_retargeting.config_types.data_type import (  # noqa: E402
    DEMO_JOINT_PARENT_INDICES,
    DEMO_JOINTS_REGISTRY,
    MotionDataConfig,
    normalize_data_format,
)
from holosoma_retargeting.data_utils.motion_data import (  # noqa: E402
    discover_motion_files,
    load_human_motion,
    resolve_motion_path,
    validate_motion_skeleton_contract,
    validate_motion_task,
)
from holosoma_retargeting.src.utils import extract_foot_sticking_sequence_velocity  # noqa: E402


def _joints(data_format: str, frames: int = 3) -> np.ndarray:
    joint_count = len(DEMO_JOINTS_REGISTRY[data_format])
    return np.arange(frames * joint_count * 3, dtype=np.float32).reshape(frames, joint_count, 3)


def _save_standard_npz(
    path: Path,
    data_format: str,
    *,
    joints: np.ndarray | None = None,
    height: float = 1.75,
    fps: float = 30.0,
    root_quaternions: np.ndarray | None = None,
    global_joint_quaternions: np.ndarray | None = None,
    orientation_joint_names: tuple[str, ...] | None = None,
    orientation_quaternions: np.ndarray | None = None,
    orientation_provenance: str | None = None,
    orientation_source: str | None = None,
) -> None:
    joints = _joints(data_format) if joints is None else joints
    payload = {
        "global_joint_positions": joints,
        "joint_names": np.asarray(DEMO_JOINTS_REGISTRY[data_format]),
        "height": np.float32(height),
        "fps": np.float32(fps),
        "source_format": np.asarray(data_format),
    }
    if root_quaternions is not None:
        payload["root_quaternions_wxyz"] = root_quaternions
    if global_joint_quaternions is not None:
        payload["global_joint_quaternions_wxyz"] = global_joint_quaternions
    if orientation_joint_names is not None:
        payload["orientation_joint_names"] = np.asarray(
            orientation_joint_names,
            dtype=str,
        )
    if orientation_quaternions is not None:
        payload["orientation_quaternions_wxyz"] = orientation_quaternions
    if orientation_provenance is not None:
        payload["orientation_provenance"] = np.asarray(
            orientation_provenance,
        )
    if orientation_source is not None:
        payload["orientation_source"] = np.asarray(
            orientation_source,
        )
    if orientation_joint_names is not None or orientation_quaternions is not None:
        coordinate_system = "right_handed_z_up" if data_format in {"amass", "gvhmr"} else "z_up"
        payload["quaternion_convention"] = np.asarray("wxyz")
        payload["coordinate_system"] = np.asarray(coordinate_system)
        if data_format in {"amass", "fbx_mocap", "gvhmr"}:
            payload["orientation_coordinate_system"] = np.asarray(
                coordinate_system,
            )
        if data_format == "fbx_mocap":
            payload["source_coordinate_system"] = np.asarray(
                "fbx_global_settings_y_up_centimetres",
            )
            payload["source_fbx_up_axis"] = np.int8(1)
            payload["source_fbx_up_axis_sign"] = np.int8(1)
            payload["source_fbx_front_axis"] = np.int8(2)
            payload["source_fbx_front_axis_sign"] = np.int8(1)
            payload["source_fbx_coord_axis"] = np.int8(0)
            payload["source_fbx_coord_axis_sign"] = np.int8(1)
            payload["source_fbx_unit_scale_factor"] = np.float32(1.0)
            payload["source_anatomical_left_axis"] = np.asarray(
                "positive_x_from_named_bind_joints",
            )
            payload["source_anatomical_forward_axis"] = np.asarray(
                "positive_z_from_named_bind_toes",
            )
            payload["position_coordinate_transform"] = np.asarray(
                "metres_x_negative_z_y_and_initial_hips_xy_recenter",
            )
            payload["orientation_coordinate_transform"] = np.asarray(
                "basis_conjugation_rx_plus_90_right_handed",
            )
            payload["root_frame_to_robot_base_quaternion_wxyz"] = np.asarray(
                [2**-0.5, 0.0, 0.0, -(2**-0.5)],
                dtype=np.float32,
            )
            t_pose_names = tuple(orientation_joint_names or ())
            payload["t_pose_orientation_joint_names"] = np.asarray(t_pose_names, dtype=str)
            t_pose_quaternions = np.zeros((len(t_pose_names), 4), dtype=np.float32)
            t_pose_quaternions[:, 0] = 1.0
            payload["t_pose_orientation_quaternions_wxyz"] = t_pose_quaternions
            source_names = tuple(DEMO_JOINTS_REGISTRY[data_format])
            source_quaternions = np.zeros(
                (joints.shape[0], len(source_names), 4),
                dtype=np.float32,
            )
            source_quaternions[..., 0] = 1.0
            source_bind_quaternions = np.zeros(
                (len(source_names), 4),
                dtype=np.float32,
            )
            source_bind_quaternions[..., 0] = 1.0
            payload["source_skeleton_joint_names"] = np.asarray(source_names, dtype=str)
            payload["source_skeleton_parent_indices"] = np.asarray(
                DEMO_JOINT_PARENT_INDICES[data_format],
                dtype=np.int32,
            )
            payload["source_skeleton_positions"] = joints
            payload["source_skeleton_quaternions_wxyz"] = source_quaternions
            payload["source_skeleton_bind_quaternions_wxyz"] = source_bind_quaternions
    np.savez(path, **payload)


def _tamper_npz(
    path: Path,
    *,
    remove: tuple[str, ...] = (),
    replace: dict[str, str] | None = None,
) -> None:
    with np.load(path, allow_pickle=False) as data:
        payload = {key: np.array(data[key], copy=True) for key in data.files}
    for key in remove:
        payload.pop(key)
    for key, value in (replace or {}).items():
        payload[key] = np.asarray(value)
    np.savez(path, **payload)


class MotionFormatRegistrationTests(unittest.TestCase):
    def test_legacy_names_are_normalized(self):
        aliases = {
            "smplh": "omomo",
            "smplx": "amass",
            "noetix_lafan": "noetix_mocap",
            "noetix-mocap": "noetix_mocap",
        }
        for alias, canonical in aliases.items():
            with self.subTest(alias=alias):
                self.assertEqual(normalize_data_format(alias), canonical)
                self.assertEqual(MotionDataConfig(data_format=alias).data_format, canonical)

    def test_unknown_formats_and_unsupported_tasks_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "Invalid data_format"):
            normalize_data_format("unknown")
        with self.assertRaisesRegex(ValueError, "supports task types: robot_only"):
            validate_motion_task("gvhmr", "object_interaction")
        self.assertEqual(validate_motion_task("omomo", "object_interaction"), "omomo")

    def test_custom_human_topology_must_be_one_rooted_tree(self):
        valid = MotionDataConfig(
            data_format="amass",
            demo_joints=["Root", "Child", "Leaf"],
            joint_parent_indices=(-1, 0, 1),
        )
        self.assertEqual(
            valid.resolved_joint_parent_indices,
            (-1, 0, 1),
        )

        invalid_cases = {
            "rooted": (-1, -1, 1),
            "cycles": (-1, 2, 1),
            "self-parent": (-1, 1, 0),
            "valid joint indices": (-1, 0, 3),
        }
        for message, parents in invalid_cases.items():
            with self.subTest(parents=parents):
                invalid = MotionDataConfig(
                    data_format="amass",
                    demo_joints=["Root", "Child", "Leaf"],
                    joint_parent_indices=parents,
                )
                with self.assertRaisesRegex(ValueError, message):
                    _ = invalid.resolved_joint_parent_indices

    def test_all_five_formats_have_g1_mappings(self):
        for data_format in ("amass", "lafan", "omomo", "noetix_mocap", "gvhmr"):
            with self.subTest(data_format=data_format):
                config = MotionDataConfig(data_format=data_format, robot_type="g1")
                self.assertTrue(config.resolved_demo_joints)
                self.assertTrue(config.resolved_joints_mapping)
                self.assertEqual(len(config.toe_names), 2)

    def test_contact_keys_follow_each_formats_registered_toe_names(self):
        for data_format in ("amass", "lafan", "omomo", "noetix_mocap", "gvhmr"):
            with self.subTest(data_format=data_format):
                config = MotionDataConfig(data_format=data_format, robot_type="g1")
                contacts = extract_foot_sticking_sequence_velocity(
                    np.zeros((2, len(config.resolved_demo_joints), 3), dtype=np.float32),
                    config.resolved_demo_joints,
                    config.toe_names,
                )
                self.assertEqual(list(contacts[0]), config.toe_names)

    def test_every_registered_format_has_one_complete_rooted_topology(self):
        for data_format, joint_names in DEMO_JOINTS_REGISTRY.items():
            with self.subTest(data_format=data_format):
                parents = np.asarray(
                    DEMO_JOINT_PARENT_INDICES[data_format],
                    dtype=np.int32,
                )
                self.assertEqual(parents.shape, (len(joint_names),))
                self.assertEqual(np.flatnonzero(parents == -1).size, 1)
                for joint_index in range(len(joint_names)):
                    visited = set()
                    current = joint_index
                    while current != -1:
                        self.assertNotIn(current, visited)
                        visited.add(current)
                        current = int(parents[current])


class MotionAdapterTests(unittest.TestCase):
    def test_all_registered_direct_orientation_npz_contracts_load(self):
        orientation_sources = {
            "amass": "direct_local_rotation_fk",
            "fbx_mocap": "fbx_local_rotation_curves_fk",
            "gvhmr": "direct_local_rotation_fk",
            "lafan": "bvh_rotation_channels_fk",
            "mocap": "bone_rotation_channels_fk",
            "noetix_mocap": "bvh_rotation_channels_fk",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            for data_format, orientation_source in orientation_sources.items():
                with self.subTest(data_format=data_format):
                    orientation_name = DEMO_JOINTS_REGISTRY[data_format][0]
                    quaternions = np.zeros((3, 1, 4), dtype=np.float32)
                    quaternions[..., 0] = 1.0
                    _save_standard_npz(
                        directory / f"{data_format}.npz",
                        data_format,
                        orientation_joint_names=(orientation_name,),
                        orientation_quaternions=quaternions,
                        orientation_source=orientation_source,
                    )

                    motion = load_human_motion(
                        data_format,
                        directory,
                        data_format,
                    )

                    self.assertEqual(
                        motion.orientation_joint_names,
                        (orientation_name,),
                    )
                    np.testing.assert_array_equal(
                        motion.orientation_quaternions_wxyz,
                        quaternions,
                    )
                    if data_format == "fbx_mocap":
                        np.testing.assert_allclose(
                            motion.root_frame_to_robot_base_quaternion_wxyz,
                            (2**-0.5, 0.0, 0.0, -(2**-0.5)),
                            atol=1e-7,
                        )
                        self.assertEqual(
                            motion.t_pose_orientation_joint_names,
                            (orientation_name,),
                        )
                        np.testing.assert_allclose(
                            motion.t_pose_orientation_quaternions_wxyz,
                            ((1.0, 0.0, 0.0, 0.0),),
                        )

    def test_direct_orientation_npz_metadata_tampering_fails_loudly(self):
        cases = (
            (
                "missing_source_format",
                ("source_format",),
                {},
                "source_format",
            ),
            (
                "wrong_source_format",
                (),
                {"source_format": "gvhmr"},
                "source_format",
            ),
            (
                "missing_convention",
                ("quaternion_convention",),
                {},
                "quaternion_convention",
            ),
            (
                "xyzw_convention",
                (),
                {"quaternion_convention": "xyzw"},
                "quaternion_convention",
            ),
            (
                "missing_coordinate_system",
                ("coordinate_system",),
                {},
                "coordinate_system",
            ),
            (
                "wrong_coordinate_system",
                (),
                {"coordinate_system": "right_handed_y_up"},
                "coordinate_system",
            ),
            (
                "missing_orientation_coordinate_system",
                ("orientation_coordinate_system",),
                {},
                "orientation_coordinate_system",
            ),
            (
                "wrong_orientation_coordinate_system",
                (),
                {"orientation_coordinate_system": "right_handed_y_up"},
                "orientation_coordinate_system",
            ),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            for task_name, remove, replace, message in cases:
                with self.subTest(task_name=task_name):
                    quaternions = np.zeros((3, 1, 4), dtype=np.float32)
                    quaternions[..., 0] = 1.0
                    path = directory / f"{task_name}.npz"
                    _save_standard_npz(
                        path,
                        "amass",
                        orientation_joint_names=("Pelvis",),
                        orientation_quaternions=quaternions,
                        orientation_source="direct_local_rotation_fk",
                    )
                    _tamper_npz(path, remove=remove, replace=replace)

                    with self.assertRaisesRegex(
                        (KeyError, ValueError),
                        message,
                    ):
                        load_human_motion(
                            "amass",
                            directory,
                            task_name,
                        )

    def test_optional_orientation_coordinate_system_is_validated_when_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            quaternions = np.zeros((3, 1, 4), dtype=np.float32)
            quaternions[..., 0] = 1.0
            path = directory / "dance.npz"
            _save_standard_npz(
                path,
                "lafan",
                orientation_joint_names=("Hips",),
                orientation_quaternions=quaternions,
                orientation_source="bvh_rotation_channels_fk",
            )
            _tamper_npz(
                path,
                replace={"orientation_coordinate_system": "right_handed_y_up"},
            )

            with self.assertRaisesRegex(
                ValueError,
                "orientation_coordinate_system",
            ):
                load_human_motion("lafan", directory, "dance")

    def test_generic_positional_mocap_climb_has_no_inferred_orientations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            task_directory = directory / "climb"
            task_directory.mkdir()
            joints = _joints("mocap", frames=8)
            np.save(task_directory / "motion.npy", joints)

            motion = load_human_motion("mocap", directory, "climb")

            np.testing.assert_array_equal(motion.joints, joints[::4])
            self.assertIsNone(motion.orientation_joint_names)
            self.assertIsNone(motion.orientation_quaternions_wxyz)
            self.assertIsNone(motion.orientation_source)

    def test_amass_loads_legacy_and_standard_metadata_without_legacy_orientation_fallbacks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            joints = _joints("amass")
            np.savez(directory / "legacy.npz", global_joint_positions=joints, height=np.float32(1.8))

            legacy = load_human_motion("amass", directory, "legacy")
            self.assertEqual(legacy.joints.shape, (3, 22, 3))
            self.assertEqual(legacy.fps, 30.0)
            self.assertIsNone(legacy.root_quaternions_wxyz)

            quaternions = np.tile([2.0, 0.0, 0.0, 0.0], (3, 1))
            _save_standard_npz(
                directory / "standard.npz",
                "amass",
                joints=joints,
                height=1.9,
                fps=60.0,
                root_quaternions=quaternions,
            )
            standard = load_human_motion("smplx", directory, "standard")
            self.assertEqual(standard.fps, 60.0)
            self.assertAlmostEqual(standard.human_height, 1.9, places=5)
            self.assertIsNone(standard.root_quaternions_wxyz)
            self.assertIsNone(standard.orientation_source)

    def test_gvhmr_does_not_treat_legacy_root_tensor_as_direct_provenance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            quaternions = np.tile([0.0, 0.0, 0.0, 3.0], (3, 1))
            _save_standard_npz(
                directory / "tennis.npz",
                "gvhmr",
                root_quaternions=quaternions,
            )
            motion = load_human_motion("gvhmr", directory, "tennis")
            self.assertIsNone(motion.root_quaternions_wxyz)
            self.assertIsNone(motion.orientation_joint_names)
            self.assertIsNone(motion.global_joint_quaternions_wxyz)

            _save_standard_npz(directory / "missing_root.npz", "gvhmr")
            missing = load_human_motion("gvhmr", directory, "missing_root")
            self.assertIsNone(missing.orientation_joint_names)
            self.assertIsNone(missing.orientation_quaternions_wxyz)
            self.assertIsNone(missing.root_quaternions_wxyz)
            self.assertIsNone(missing.global_joint_quaternions_wxyz)

    def test_lafan_converts_official_y_up_npy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            joints_y_up = _joints("lafan")
            np.save(directory / "dance.npy", joints_y_up)
            motion = load_human_motion("lafan", directory, "dance")
            np.testing.assert_array_equal(motion.joints, joints_y_up[..., [0, 2, 1]])
            self.assertAlmostEqual(motion.human_height, 1.7)
            self.assertEqual(motion.fps, 30.0)
            self.assertIsNone(motion.orientation_joint_names)
            self.assertIsNone(motion.orientation_quaternions_wxyz)

    def test_lafan_prefers_canonical_npz_with_direct_orientations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            legacy_y_up = _joints("lafan")
            np.save(directory / "dance.npy", legacy_y_up)
            quaternions = np.zeros((3, 22, 4), dtype=np.float64)
            quaternions[..., 0] = 1.0
            _save_standard_npz(
                directory / "dance.npz",
                "lafan",
                joints=np.full((3, 22, 3), 7.0, dtype=np.float32),
                orientation_joint_names=tuple(DEMO_JOINTS_REGISTRY["lafan"]),
                orientation_quaternions=quaternions,
                orientation_source="bvh_rotation_channels_fk",
            )

            self.assertEqual(
                resolve_motion_path(directory, "dance", "lafan"),
                directory / "dance.npz",
            )
            motion = load_human_motion("lafan", directory, "dance")

            np.testing.assert_array_equal(motion.joints, 7.0)
            self.assertEqual(
                motion.orientation_joint_names,
                tuple(DEMO_JOINTS_REGISTRY["lafan"]),
            )
            self.assertEqual(
                motion.global_joint_quaternions_wxyz.shape,
                (3, 22, 4),
            )
            self.assertEqual(
                motion.orientation_source,
                "bvh_rotation_channels_fk",
            )
            np.testing.assert_array_equal(
                motion.orientation_quaternions_wxyz,
                quaternions.astype(np.float32),
            )

    def test_noetix_validates_joint_order_and_height(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            _save_standard_npz(directory / "fall.npz", "noetix_mocap", height=1.71)
            motion = load_human_motion("noetix-mocap", directory, "fall")
            self.assertEqual(motion.joints.shape[1], 22)
            self.assertAlmostEqual(motion.human_height, 1.71, places=5)

            np.savez(
                directory / "bad_names.npz",
                global_joint_positions=_joints("noetix_mocap"),
                joint_names=np.asarray(list(reversed(DEMO_JOINTS_REGISTRY["noetix_mocap"]))),
                height=np.float32(1.7),
                fps=np.float32(30.0),
            )
            with self.assertRaisesRegex(ValueError, "joint_names"):
                load_human_motion("noetix_mocap", directory, "bad_names")

    def test_noetix_does_not_expose_unprovenanced_legacy_dense_orientations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            joint_count = len(DEMO_JOINTS_REGISTRY["noetix_mocap"])
            quaternions = np.zeros((3, joint_count, 4), dtype=np.float64)
            quaternions[..., 0] = 2.0
            _save_standard_npz(
                directory / "dance.npz",
                "noetix_mocap",
                global_joint_quaternions=quaternions,
            )

            motion = load_human_motion(
                "noetix_mocap",
                directory,
                "dance",
            )

            self.assertIsNone(motion.orientation_joint_names)
            self.assertIsNone(motion.orientation_quaternions_wxyz)
            self.assertIsNone(motion.orientation_source)
            self.assertIsNone(motion.root_quaternions_wxyz)
            self.assertIsNone(motion.global_joint_quaternions_wxyz)

    def test_noetix_does_not_accept_legacy_dense_orientations_even_with_legacy_markers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            joint_count = len(DEMO_JOINTS_REGISTRY["noetix_mocap"])
            quaternions = np.zeros((3, joint_count, 4), dtype=np.float64)
            quaternions[..., 0] = 2.0
            _save_standard_npz(
                directory / "dance.npz",
                "noetix_mocap",
                global_joint_quaternions=quaternions,
                orientation_provenance="direct_source",
                orientation_source="legacy_direct_bvh_fk",
            )

            with self.assertRaisesRegex(
                KeyError,
                "orientation_joint_names",
            ):
                load_human_motion(
                    "noetix_mocap",
                    directory,
                    "dance",
                )

    def test_noetix_rejects_invalid_global_joint_orientations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            joint_count = len(DEMO_JOINTS_REGISTRY["noetix_mocap"])
            for name, quaternions, message in (
                (
                    "bad_shape",
                    np.ones((3, joint_count - 1, 4)),
                    "must have shape",
                ),
                (
                    "zero",
                    np.zeros((3, joint_count, 4)),
                    "finite and non-zero",
                ),
                (
                    "nan",
                    np.full((3, joint_count, 4), np.nan),
                    "finite and non-zero",
                ),
                (
                    "non_unit",
                    np.full((3, joint_count, 4), 0.6),
                    "must already be unit length",
                ),
            ):
                with self.subTest(name=name):
                    _save_standard_npz(
                        directory / f"{name}.npz",
                        "noetix_mocap",
                        orientation_joint_names=tuple(
                            DEMO_JOINTS_REGISTRY["noetix_mocap"],
                        ),
                        orientation_quaternions=quaternions,
                        orientation_source="bvh_rotation_channels_fk",
                    )
                    with self.assertRaisesRegex(ValueError, message):
                        load_human_motion(
                            "noetix_mocap",
                            directory,
                            name,
                        )

    def test_noetix_partial_orientation_subset_is_not_densified(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            names = ("Hips", "Spine1", "Neck")
            quaternions = np.zeros((3, len(names), 4), dtype=np.float64)
            quaternions[..., 0] = 1.0
            _save_standard_npz(
                directory / "partial.npz",
                "noetix_mocap",
                orientation_joint_names=names,
                orientation_quaternions=quaternions,
                orientation_source="bvh_rotation_channels_fk",
            )

            motion = load_human_motion(
                "noetix_mocap",
                directory,
                "partial",
            )

            self.assertEqual(motion.orientation_joint_names, names)
            self.assertEqual(
                motion.orientation_quaternions_wxyz.shape,
                (3, 3, 4),
            )
            self.assertEqual(motion.root_quaternions_wxyz.shape, (3, 4))
            self.assertIsNone(motion.global_joint_quaternions_wxyz)
            self.assertEqual(
                motion.orientation_source,
                "bvh_rotation_channels_fk",
            )

    def test_orientation_names_must_be_unique_canonical_subset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            quaternions = np.zeros((3, 2, 4), dtype=np.float64)
            quaternions[..., 0] = 1.0
            for task_name, names, message in (
                ("duplicate", ("Hips", "Hips"), "must be unique"),
                ("unknown", ("Hips", "SyntheticSpine"), "not canonical"),
            ):
                with self.subTest(task_name=task_name):
                    _save_standard_npz(
                        directory / f"{task_name}.npz",
                        "noetix_mocap",
                        orientation_joint_names=names,
                        orientation_quaternions=quaternions,
                        orientation_source="bvh_rotation_channels_fk",
                    )
                    with self.assertRaisesRegex(ValueError, message):
                        load_human_motion(
                            "noetix_mocap",
                            directory,
                            task_name,
                        )

    def test_explicit_orientation_fields_require_approved_direct_source_provenance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            quaternions = np.zeros((3, 1, 4), dtype=np.float64)
            quaternions[..., 0] = 1.0
            for task_name, provenance, message in (
                ("missing", None, "must contain"),
                ("estimated", "estimated_from_positions", "approved direct-source"),
                ("wrong_format", "direct_local_rotation_fk", "approved direct-source"),
            ):
                with self.subTest(task_name=task_name):
                    _save_standard_npz(
                        directory / f"{task_name}.npz",
                        "noetix_mocap",
                        orientation_joint_names=("Hips",),
                        orientation_quaternions=quaternions,
                        orientation_source=provenance,
                    )
                    with self.assertRaisesRegex(
                        (KeyError, ValueError),
                        message,
                    ):
                        load_human_motion(
                            "noetix_mocap",
                            directory,
                            task_name,
                        )

    def test_loaded_motion_exposes_complete_registered_topology(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            _save_standard_npz(directory / "walk.npz", "amass")

            motion = load_human_motion("amass", directory, "walk")

            self.assertEqual(
                motion.joint_parent_indices,
                DEMO_JOINT_PARENT_INDICES["amass"],
            )
            self.assertEqual(
                motion.canonical_joint_names,
                tuple(DEMO_JOINTS_REGISTRY["amass"]),
            )

    def test_pipeline_skeleton_contract_rejects_silent_relabeling(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            _save_standard_npz(directory / "walk.npz", "amass")
            motion = load_human_motion("amass", directory, "walk")
            canonical_names = list(DEMO_JOINTS_REGISTRY["amass"])
            canonical_parents = DEMO_JOINT_PARENT_INDICES["amass"]

            validate_motion_skeleton_contract(
                motion,
                canonical_names,
                canonical_parents,
                data_format="amass",
            )
            validate_motion_skeleton_contract(
                motion,
                list(canonical_names),
                tuple(canonical_parents),
                data_format="amass",
            )

            swapped_names = list(canonical_names)
            swapped_names[0], swapped_names[1] = (
                swapped_names[1],
                swapped_names[0],
            )
            renamed_names = list(canonical_names)
            renamed_names[-1] = "RenamedJoint"
            for invalid_names in (swapped_names, renamed_names):
                with self.subTest(invalid_names=invalid_names), self.assertRaisesRegex(
                    ValueError,
                    "joint names/order must match exactly",
                ):
                    validate_motion_skeleton_contract(
                        motion,
                        invalid_names,
                        canonical_parents,
                        data_format="amass",
                    )

    def test_pipeline_skeleton_contract_rejects_different_valid_topology(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            _save_standard_npz(directory / "walk.npz", "amass")
            motion = load_human_motion("amass", directory, "walk")
            canonical_names = DEMO_JOINTS_REGISTRY["amass"]
            alternative_parents = (-1, *(0 for _ in canonical_names[1:]))

            with self.assertRaisesRegex(
                ValueError,
                "parent topology must match exactly",
            ):
                validate_motion_skeleton_contract(
                    motion,
                    canonical_names,
                    alternative_parents,
                    data_format="amass",
                )

    def test_omomo_extracts_joints_and_wxyz_xyz_object_pose(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            values = torch.zeros(2, 325)
            joints = _joints("omomo", frames=2)
            values[:, 162:318] = torch.from_numpy(joints.reshape(2, -1))
            # Source layout is xyz + xyzw. The adapter exposes wxyz + xyz.
            values[:, 318:325] = torch.tensor([[1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14]])
            torch.save(values, directory / "sub1_box_001.pt")

            motion = load_human_motion("omomo", directory, "sub1_box_001", human_height=1.68)
            np.testing.assert_array_equal(motion.joints, joints)
            np.testing.assert_array_equal(motion.object_poses_wxyz_xyz[0], [7, 4, 5, 6, 1, 2, 3])
            self.assertAlmostEqual(motion.human_height, 1.68)
            self.assertIsNone(motion.orientation_joint_names)
            self.assertIsNone(motion.orientation_quaternions_wxyz)

    def test_omomo_reads_documented_inter_mimic_global_xyzw_block(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            values = torch.zeros(2, 591)
            joints = _joints("omomo", frames=2)
            values[:, 162:318] = torch.from_numpy(joints.reshape(2, -1))
            values[:, 318:325] = torch.tensor([[1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14]])
            orientations_xyzw = torch.zeros(2, 52, 4)
            orientations_xyzw[..., 3] = 1.0
            orientations_xyzw[0, 0] = torch.tensor([0.0, 0.6, 0.0, 0.8])
            values[:, 383:591] = orientations_xyzw.reshape(2, -1)
            torch.save(values, directory / "sub1_box_002.pt")

            motion = load_human_motion(
                "omomo",
                directory,
                "sub1_box_002",
                human_height=1.68,
            )

            self.assertEqual(
                motion.orientation_joint_names,
                tuple(DEMO_JOINTS_REGISTRY["omomo"]),
            )
            np.testing.assert_array_equal(
                motion.orientation_quaternions_wxyz[0, 0],
                np.asarray([0.8, 0.0, 0.6, 0.0], dtype=np.float32),
            )
            self.assertEqual(
                motion.global_joint_quaternions_wxyz.shape,
                (2, 52, 4),
            )
            self.assertEqual(
                motion.orientation_source,
                "intermimic_global_orientation_tensor",
            )

    def test_omomo_width_below_orientation_contract_stays_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            values = torch.zeros(1, 590)
            values[:, 318:325] = torch.tensor([[0, 0, 0, 0, 0, 0, 1]])
            torch.save(values, directory / "sub1_box_003.pt")

            motion = load_human_motion(
                "omomo",
                directory,
                "sub1_box_003",
                human_height=1.68,
            )

            self.assertIsNone(motion.orientation_joint_names)
            self.assertIsNone(motion.orientation_quaternions_wxyz)

    def test_invalid_shapes_do_not_fall_through_to_another_loader(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            np.savez(
                directory / "bad.npz",
                global_joint_positions=np.zeros((2, 21, 3), dtype=np.float32),
                height=np.float32(1.7),
            )
            with self.assertRaisesRegex(ValueError, r"shape \(T, 22, 3\)"):
                load_human_motion("amass", directory, "bad")


class MotionDiscoveryTests(unittest.TestCase):
    def test_lafan_and_noetix_discover_only_their_own_file_types(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            np.save(directory / "dance.npy", _joints("lafan"))
            _save_standard_npz(directory / "company_motion.npz", "noetix_mocap")

            self.assertEqual(discover_motion_files(directory, "lafan"), [directory / "dance.npy"])
            self.assertEqual(
                discover_motion_files(directory, "noetix_mocap"),
                [directory / "company_motion.npz"],
            )
            self.assertEqual(resolve_motion_path(directory, "dance", "lafan"), directory / "dance.npy")
            with self.assertRaises(FileNotFoundError):
                resolve_motion_path(directory, "dance", "noetix_mocap")
            with self.assertRaises(FileNotFoundError):
                resolve_motion_path(directory, "company_motion", "lafan")

    def test_omomo_object_filter_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            torch.save(torch.zeros(1, 325), directory / "sub1_box_001.pt")
            torch.save(torch.zeros(1, 325), directory / "sub1_chair_001.pt")
            self.assertEqual(
                discover_motion_files(directory, "omomo", object_name="box"),
                [directory / "sub1_box_001.pt"],
            )

    def test_noetix_discovery_is_recursive_and_excludes_scene_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            nested = directory / "session"
            nested.mkdir()
            _save_standard_npz(
                nested / "company_motion.npz",
                "noetix_mocap",
            )
            np.savez(
                nested / "scene_reconstruction.npz",
                mocap_joint_positions=np.zeros((1, 53, 3)),
            )

            self.assertEqual(
                discover_motion_files(directory, "noetix_mocap"),
                [nested / "company_motion.npz"],
            )
            self.assertEqual(
                resolve_motion_path(
                    directory,
                    "company_motion",
                    "noetix_mocap",
                ),
                nested / "company_motion.npz",
            )

    def test_recursive_noetix_preserves_same_stem_in_relative_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            for session in ("a", "b"):
                nested = directory / session
                nested.mkdir()
                _save_standard_npz(
                    nested / "same_name.npz",
                    "noetix_mocap",
                )

            self.assertEqual(
                discover_motion_files(directory, "noetix_mocap"),
                [
                    directory / "a" / "same_name.npz",
                    directory / "b" / "same_name.npz",
                ],
            )
            self.assertEqual(
                resolve_motion_path(
                    directory,
                    "a/same_name",
                    "noetix_mocap",
                ),
                directory / "a" / "same_name.npz",
            )
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                resolve_motion_path(
                    directory,
                    "same_name",
                    "noetix_mocap",
                )


if __name__ == "__main__":
    unittest.main()
