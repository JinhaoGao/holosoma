# ruff: noqa: CPY001, PT009

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from typing import BinaryIO
from unittest.mock import patch

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "holosoma_retargeting"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from holosoma_retargeting.config_types.data_type import (  # noqa: E402
    NOETIX_MOCAP_DEMO_JOINTS,
)
from holosoma_retargeting.data_utils.convert_noetix_bvh import (  # noqa: E402
    NOETIX_FULLBODY_REDUCED_SPINE_JOINTS,
    NOETIX_FULLBODY_SINGLE_SPINE_JOINTS,
    Y_UP_TO_Z_UP_BASIS,
    Config,
    _atomic_savez_compressed,
    classify_bvh,
    convert_file,
    enforce_quaternion_continuity_wxyz,
    main,
    select_direct_canonical_orientations,
    source_height_from_filename,
    synthesize_single_spine_chain,
    transform_orientations_y_up_to_z_up,
)

EXPERIMENT_BVH = (
    PACKAGE_ROOT
    / "holosoma_retargeting"
    / "demo_data"
    / "noetix_ori"
    / "0724_BEITI"
    / "breaking+hippop.bvh_Skeleton1.bvh"
)


class NoetixOrientationTransformTests(unittest.TestCase):
    def test_single_spine_export_is_classified_and_synthesized(self):
        source_type, mapping = classify_bvh(sorted(NOETIX_FULLBODY_SINGLE_SPINE_JOINTS))

        self.assertEqual(source_type, "fullbody_single_spine_zyx")
        self.assertEqual(mapping["Spine1"], "Spine1")

        joint_idx = {name: idx for idx, name in enumerate(NOETIX_MOCAP_DEMO_JOINTS)}
        positions = np.zeros((1, len(joint_idx), 3))
        positions[0, joint_idx["Spine1"], 1] = 3.0
        positions[0, joint_idx["Neck"], 1] = 5.0

        synthesized = synthesize_single_spine_chain(positions)

        np.testing.assert_allclose(
            synthesized[0, joint_idx["Spine"]],
            [0.0, 1.0, 0.0],
        )
        np.testing.assert_allclose(
            synthesized[0, joint_idx["Spine2"]],
            [0.0, 4.0, 0.0],
        )
        np.testing.assert_allclose(
            synthesized[0, joint_idx["Spine1"]],
            positions[0, joint_idx["Spine1"]],
        )

    def test_braced_height_token_is_recognized(self):
        path = Path("gaotaitui{F}_{CKX}_{168}_000_Skeleton.bvh")

        self.assertEqual(source_height_from_filename(path), 1.68)

    def test_single_spine_omits_both_synthetic_orientation_frames(self):
        source_names = sorted(NOETIX_FULLBODY_SINGLE_SPINE_JOINTS)
        source_type, mapping = classify_bvh(source_names)
        source_quaternions = np.zeros((2, len(source_names), 4))
        source_quaternions[..., 0] = 1.0

        orientation_names, orientations = select_direct_canonical_orientations(
            source_quaternions,
            source_names,
            mapping,
            source_type,
        )

        self.assertNotIn("Spine", orientation_names)
        self.assertIn("Spine1", orientation_names)
        self.assertNotIn("Spine2", orientation_names)
        self.assertEqual(
            orientations.shape,
            (2, len(NOETIX_MOCAP_DEMO_JOINTS) - 2, 4),
        )

    def test_reduced_spine_omits_only_synthetic_spine2_orientation(self):
        source_names = sorted(NOETIX_FULLBODY_REDUCED_SPINE_JOINTS)
        source_type, mapping = classify_bvh(source_names)
        source_quaternions = np.zeros((2, len(source_names), 4))
        source_quaternions[..., 0] = 1.0

        orientation_names, orientations = select_direct_canonical_orientations(
            source_quaternions,
            source_names,
            mapping,
            source_type,
        )

        self.assertIn("Spine", orientation_names)
        self.assertIn("Spine1", orientation_names)
        self.assertNotIn("Spine2", orientation_names)
        self.assertEqual(
            orientations.shape,
            (2, len(NOETIX_MOCAP_DEMO_JOINTS) - 1, 4),
        )

    def test_recursive_main_preserves_source_relative_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            input_dir = base / "input"
            output_dir = base / "output"
            nested = input_dir / "session" / "actor"
            nested.mkdir(parents=True)
            source = nested / "motion.bvh"
            source.write_text("placeholder", encoding="utf-8")

            with patch("holosoma_retargeting.data_utils.convert_noetix_bvh.convert_file") as mocked:
                main(
                    Config(
                        input_dir=input_dir,
                        output_dir=output_dir,
                    )
                )

            self.assertEqual(mocked.call_count, 1)
            self.assertEqual(
                mocked.call_args.args[1],
                output_dir / "session" / "actor",
            )

    def test_reflected_basis_conjugation_preserves_rotated_vectors(self):
        rotation_y_up = Rotation.from_euler("xyz", [35.0, -20.0, 80.0], degrees=True)
        xyzw = rotation_y_up.as_quat()
        wxyz = xyzw[[3, 0, 1, 2]].reshape(1, 1, 4)

        transformed_wxyz = transform_orientations_y_up_to_z_up(wxyz)[0, 0]
        transformed_matrix = Rotation.from_quat(transformed_wxyz[[1, 2, 3, 0]]).as_matrix()
        vector_y_up = np.array([0.3, -0.7, 1.2])

        expected_z_up = Y_UP_TO_Z_UP_BASIS @ rotation_y_up.apply(vector_y_up)
        actual_z_up = transformed_matrix @ (Y_UP_TO_Z_UP_BASIS @ vector_y_up)

        np.testing.assert_allclose(actual_z_up, expected_z_up, atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(transformed_matrix), 1.0, places=12)

    def test_quaternion_signs_are_continuous_per_joint(self):
        quaternions = np.array(
            [
                [[1.0, 0.0, 0.0, 0.0]],
                [[-1.0, 0.0, 0.0, 0.0]],
                [[2.0, 0.0, 0.0, 0.0]],
            ]
        )

        continuous = enforce_quaternion_continuity_wxyz(quaternions)

        np.testing.assert_allclose(np.linalg.norm(continuous, axis=-1), 1.0)
        self.assertTrue(np.all(np.sum(continuous[:-1] * continuous[1:], axis=-1) >= 0.0))

    def test_atomic_npz_writer_preserves_existing_output_on_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "converted.npz"
            np.savez_compressed(output_path, generation=np.asarray("old"))

            def write_partial_then_fail(
                temporary_file: BinaryIO,
                **_payload: object,
            ) -> None:
                temporary_file.write(b"partial zip")
                temporary_file.flush()
                raise OSError("simulated conversion write failure")

            with patch(
                "holosoma_retargeting.data_utils.convert_noetix_bvh.np.savez_compressed",
                side_effect=write_partial_then_fail,
            ), pytest.raises(
                OSError,
                match="simulated conversion write failure",
            ):
                _atomic_savez_compressed(
                    output_path,
                    {"generation": np.asarray("new")},
                )

            with np.load(output_path, allow_pickle=False) as preserved:
                self.assertEqual(str(preserved["generation"]), "old")
            self.assertEqual(
                list(output_path.parent.glob(f".{output_path.name}.*.tmp")),
                [],
            )


@unittest.skipUnless(EXPERIMENT_BVH.is_file(), "local Noetix experiment BVH is unavailable")
class NoetixExperimentConversionTests(unittest.TestCase):
    def test_breaking_hippop_contains_aligned_global_orientations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            convert_file(
                EXPERIMENT_BVH,
                output_dir,
                target_fps=30.0,
                drop_jump_threshold_m=2.0,
            )
            output_path = output_dir / f"{EXPERIMENT_BVH.stem}.npz"

            with np.load(output_path, allow_pickle=False) as data:
                positions = data["global_joint_positions"]
                quaternions = data["orientation_quaternions_wxyz"]

                self.assertEqual(positions.shape, (2660, 22, 3))
                self.assertEqual(quaternions.shape, (2660, 22, 4))
                self.assertEqual(
                    data["joint_names"].tolist(),
                    NOETIX_MOCAP_DEMO_JOINTS,
                )
                self.assertEqual(
                    data["orientation_joint_names"].tolist(),
                    NOETIX_MOCAP_DEMO_JOINTS,
                )
                self.assertEqual(data["joint_parents"].shape, (22,))
                self.assertEqual(str(data["source_type"]), "fullbody57_chest_zyx")
                self.assertEqual(str(data["quaternion_convention"]), "wxyz")
                self.assertEqual(
                    str(data["orientation_source"]),
                    "bvh_rotation_channels_fk",
                )
                self.assertEqual(
                    str(data["orientation_coordinate_transform"]),
                    "basis_conjugation_xzy_reflection",
                )
                np.testing.assert_allclose(
                    np.linalg.norm(quaternions, axis=-1),
                    1.0,
                    atol=1e-6,
                )
                self.assertTrue(
                    np.all(
                        np.sum(
                            quaternions[:-1] * quaternions[1:],
                            axis=-1,
                        )
                        >= -1e-6
                    )
                )
