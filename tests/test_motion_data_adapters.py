# ruff: noqa: PT009, PT027

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
    DEMO_JOINTS_REGISTRY,
    MotionDataConfig,
    normalize_data_format,
)
from holosoma_retargeting.data_utils.motion_data import (  # noqa: E402
    discover_motion_files,
    load_human_motion,
    resolve_motion_path,
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
) -> None:
    joints = _joints(data_format) if joints is None else joints
    payload = {
        "global_joint_positions": joints,
        "joint_names": np.asarray(DEMO_JOINTS_REGISTRY[data_format]),
        "height": np.float32(height),
        "fps": np.float32(fps),
    }
    if root_quaternions is not None:
        payload["root_quaternions_wxyz"] = root_quaternions
    if global_joint_quaternions is not None:
        payload["global_joint_quaternions_wxyz"] = global_joint_quaternions
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


class MotionAdapterTests(unittest.TestCase):
    def test_amass_loads_legacy_and_standard_metadata(self):
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
            np.testing.assert_allclose(standard.root_quaternions_wxyz[:, 0], 1.0)

    def test_gvhmr_requires_and_normalizes_root_quaternions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            quaternions = np.tile([0.0, 0.0, 0.0, 3.0], (3, 1))
            _save_standard_npz(
                directory / "tennis.npz",
                "gvhmr",
                root_quaternions=quaternions,
            )
            motion = load_human_motion("gvhmr", directory, "tennis")
            np.testing.assert_allclose(np.linalg.norm(motion.root_quaternions_wxyz, axis=1), 1.0)

            _save_standard_npz(directory / "missing_root.npz", "gvhmr")
            with self.assertRaisesRegex(KeyError, "root_quaternions_wxyz"):
                load_human_motion("gvhmr", directory, "missing_root")

    def test_lafan_converts_official_y_up_npy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            joints_y_up = _joints("lafan")
            np.save(directory / "dance.npy", joints_y_up)
            motion = load_human_motion("lafan", directory, "dance")
            np.testing.assert_array_equal(motion.joints, joints_y_up[..., [0, 2, 1]])
            self.assertAlmostEqual(motion.human_height, 1.7)
            self.assertEqual(motion.fps, 30.0)

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

    def test_noetix_loads_and_normalizes_global_joint_orientations(self):
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

            self.assertIsNone(motion.root_quaternions_wxyz)
            self.assertEqual(
                motion.global_joint_quaternions_wxyz.shape,
                (3, joint_count, 4),
            )
            np.testing.assert_allclose(
                np.linalg.norm(
                    motion.global_joint_quaternions_wxyz,
                    axis=-1,
                ),
                1.0,
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
            ):
                with self.subTest(name=name):
                    _save_standard_npz(
                        directory / f"{name}.npz",
                        "noetix_mocap",
                        global_joint_quaternions=quaternions,
                    )
                    with self.assertRaisesRegex(ValueError, message):
                        load_human_motion(
                            "noetix_mocap",
                            directory,
                            name,
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


if __name__ == "__main__":
    unittest.main()
