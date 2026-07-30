# ruff: noqa: CPY001, PT009, PT027

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "holosoma_retargeting"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from holosoma_retargeting.config_types.data_type import (  # noqa: E402
    GVHMR_DEMO_JOINTS,
    MotionDataConfig,
)
from holosoma_retargeting.data_utils.convert_gvhmr import (  # noqa: E402
    GVHMR_TO_Z_UP,
    ConvertedGVHMRMotion,
    convert_gvhmr_parameters,
    load_gvhmr_parameters,
    save_converted_motion,
    transform_gvhmr_points_to_z_up,
    transform_gvhmr_root_orientations,
)

SMPLX_22_PARENTS = torch.tensor(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19],
    dtype=torch.long,
)


def _valid_prediction(num_frames: int = 3, num_betas: int = 10):
    return {
        "smpl_params_global": {
            "body_pose": torch.zeros(num_frames, 63),
            "betas": torch.zeros(num_frames, num_betas),
            "global_orient": torch.zeros(num_frames, 3),
            "transl": torch.zeros(num_frames, 3),
        }
    }


class _FakeOutput:
    def __init__(self, joints: torch.Tensor, vertices: torch.Tensor):
        self.joints = joints
        self.vertices = vertices


class _FakeBodyModel:
    parents = SMPLX_22_PARENTS

    def __call__(self, *, betas, body_pose, transl, **_kwargs):
        batch_size = body_pose.shape[0]
        joints = torch.zeros(batch_size, len(GVHMR_DEMO_JOINTS), 3)
        joints[..., 1] = 1.0
        joints += transl[:, None]
        vertices = torch.tensor([[[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]]]).repeat(batch_size, 1, 1)
        return _FakeOutput(joints=joints, vertices=vertices)


class GVHMRParameterLoadingTests(unittest.TestCase):
    def test_loads_canonical_prediction_and_expands_single_beta_row(self):
        prediction = _valid_prediction(num_frames=4)
        prediction["smpl_params_global"]["betas"] = torch.zeros(1, 10)
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "hmr4d_results.pt"
            torch.save(prediction, input_path)
            parameters = load_gvhmr_parameters(input_path)

        self.assertEqual(parameters.body_pose.shape, (4, 63))
        self.assertEqual(parameters.betas.shape, (4, 10))
        self.assertEqual(parameters.num_frames, 4)
        self.assertEqual(parameters.num_betas, 10)

    def test_rejects_missing_world_parameters(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "hmr4d_results.pt"
            torch.save({"smpl_params_incam": {}}, input_path)
            with self.assertRaisesRegex(KeyError, "smpl_params_global"):
                load_gvhmr_parameters(input_path)

    def test_rejects_invalid_shape_and_non_finite_values(self):
        prediction = _valid_prediction()
        prediction["smpl_params_global"]["body_pose"] = torch.zeros(3, 62)
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "bad_shape.pt"
            torch.save(prediction, input_path)
            with self.assertRaisesRegex(ValueError, r"\(T, 63\)"):
                load_gvhmr_parameters(input_path)

            prediction = _valid_prediction()
            prediction["smpl_params_global"]["transl"][0, 0] = torch.nan
            input_path = Path(tmpdir) / "nan.pt"
            torch.save(prediction, input_path)
            with self.assertRaisesRegex(ValueError, "NaN or Inf"):
                load_gvhmr_parameters(input_path)


class GVHMRCoordinateTests(unittest.TestCase):
    def test_coordinate_rotation_is_right_handed_and_maps_y_to_z(self):
        self.assertAlmostEqual(float(np.linalg.det(GVHMR_TO_Z_UP)), 1.0, places=6)
        source = np.eye(3, dtype=np.float32)
        converted = transform_gvhmr_points_to_z_up(source)
        np.testing.assert_allclose(converted[0], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(converted[1], [0.0, 0.0, 1.0])
        np.testing.assert_allclose(converted[2], [0.0, -1.0, 0.0])

    def test_root_orientation_uses_the_same_frame_rotation(self):
        quaternion = transform_gvhmr_root_orientations(np.zeros((1, 3)))[0]
        matrix = Rotation.from_quat(quaternion, scalar_first=True).as_matrix()
        np.testing.assert_allclose(matrix, GVHMR_TO_Z_UP, atol=1e-6)


class GVHMRConversionTests(unittest.TestCase):
    def test_conversion_chunks_fk_and_computes_rest_height(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "hmr4d_results.pt"
            torch.save(_valid_prediction(num_frames=5), input_path)
            parameters = load_gvhmr_parameters(input_path)

        with mock.patch(
            "holosoma_retargeting.data_utils.convert_gvhmr._make_body_model",
            return_value=_FakeBodyModel(),
        ):
            motion = convert_gvhmr_parameters(parameters, model_path="unused", fps=25.0, batch_size=2)

        self.assertEqual(motion.global_joint_positions.shape, (5, 22, 3))
        self.assertEqual(motion.root_quaternions_wxyz.shape, (5, 4))
        self.assertEqual(motion.orientation_joint_names, tuple(GVHMR_DEMO_JOINTS))
        self.assertEqual(motion.orientation_quaternions_wxyz.shape, (5, 22, 4))
        self.assertAlmostEqual(motion.height, 2.0)
        self.assertAlmostEqual(motion.fps, 25.0)
        np.testing.assert_allclose(motion.global_joint_positions[..., 2], 1.0)
        np.testing.assert_allclose(
            np.linalg.norm(motion.orientation_quaternions_wxyz, axis=-1),
            1.0,
            atol=1e-6,
        )
        np.testing.assert_array_equal(
            motion.root_quaternions_wxyz,
            motion.orientation_quaternions_wxyz[:, 0],
        )
        matrices = (
            Rotation.from_quat(
                motion.orientation_quaternions_wxyz.reshape(-1, 4),
                scalar_first=True,
            )
            .as_matrix()
            .reshape(5, 22, 3, 3)
        )
        np.testing.assert_allclose(
            matrices,
            np.broadcast_to(GVHMR_TO_Z_UP, matrices.shape),
            atol=1e-6,
        )

    def test_orientation_fk_composes_parent_chain_before_world_transform(self):
        prediction = _valid_prediction(num_frames=2)
        prediction["smpl_params_global"]["global_orient"][:, 2] = np.pi / 2
        prediction["smpl_params_global"]["body_pose"][:, 0] = np.pi / 2
        prediction["smpl_params_global"]["body_pose"][:, 10] = np.pi / 2
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "hmr4d_results.pt"
            torch.save(prediction, input_path)
            parameters = load_gvhmr_parameters(input_path)

        with mock.patch(
            "holosoma_retargeting.data_utils.convert_gvhmr._make_body_model",
            return_value=_FakeBodyModel(),
        ):
            motion = convert_gvhmr_parameters(parameters, model_path="unused")

        matrices = (
            Rotation.from_quat(
                motion.orientation_quaternions_wxyz.reshape(-1, 4),
                scalar_first=True,
            )
            .as_matrix()
            .reshape(2, 22, 3, 3)
        )
        root = Rotation.from_rotvec([0.0, 0.0, np.pi / 2]).as_matrix()
        left_hip_local = Rotation.from_rotvec([np.pi / 2, 0.0, 0.0]).as_matrix()
        left_knee_local = Rotation.from_rotvec([0.0, np.pi / 2, 0.0]).as_matrix()
        np.testing.assert_allclose(
            matrices[:, 0],
            np.broadcast_to(GVHMR_TO_Z_UP @ root, matrices[:, 0].shape),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            matrices[:, 1],
            np.broadcast_to(
                GVHMR_TO_Z_UP @ root @ left_hip_local,
                matrices[:, 1].shape,
            ),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            matrices[:, 4],
            np.broadcast_to(
                GVHMR_TO_Z_UP @ root @ left_hip_local @ left_knee_local,
                matrices[:, 4].shape,
            ),
            atol=1e-6,
        )

    def test_standard_npz_contains_required_metadata(self):
        motion = ConvertedGVHMRMotion(
            global_joint_positions=np.zeros((2, 22, 3), dtype=np.float32),
            root_quaternions_wxyz=np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (2, 1)),
            orientation_joint_names=tuple(GVHMR_DEMO_JOINTS),
            orientation_quaternions_wxyz=np.tile(
                np.array([[[1.0, 0.0, 0.0, 0.0]]], dtype=np.float32),
                (2, 22, 1),
            ),
            orientation_source="direct_local_rotation_fk",
            height=1.8,
            fps=30.0,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "tennis.npz"
            save_converted_motion(motion, output_path, "hmr4d_results.pt")
            with np.load(output_path, allow_pickle=False) as data:
                self.assertEqual(data["global_joint_positions"].shape, (2, 22, 3))
                self.assertEqual(data["joint_names"].tolist(), GVHMR_DEMO_JOINTS)
                self.assertEqual(str(data["source_format"]), "gvhmr")
                self.assertAlmostEqual(float(data["height"]), 1.8, places=5)
                self.assertEqual(data["orientation_joint_names"].tolist(), GVHMR_DEMO_JOINTS)
                self.assertEqual(data["orientation_quaternions_wxyz"].shape, (2, 22, 4))
                self.assertEqual(str(data["orientation_source"]), "direct_local_rotation_fk")
                self.assertEqual(str(data["quaternion_convention"]), "wxyz")
                self.assertEqual(
                    str(data["orientation_coordinate_system"]),
                    "right_handed_z_up",
                )


class GVHMRFormatRegistrationTests(unittest.TestCase):
    def test_g1_and_e1_mappings_are_registered(self):
        for robot in ("g1", "e1"):
            config = MotionDataConfig(data_format="gvhmr", robot_type=robot)
            self.assertEqual(config.resolved_demo_joints, GVHMR_DEMO_JOINTS)
            self.assertEqual(config.toe_names, ["L_Foot", "R_Foot"])
            self.assertIn("Pelvis", config.resolved_joints_mapping)


if __name__ == "__main__":
    unittest.main()
