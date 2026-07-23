# ruff: noqa: PT009

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "holosoma_retargeting"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from holosoma_retargeting.config_types.data_type import AMASS_DEMO_JOINTS  # noqa: E402
from holosoma_retargeting.data_utils.prep_amass_smplx_for_rt import (  # noqa: E402
    convert_amass_parameters,
    load_amass_parameters,
    save_converted_amass,
)


class _FakeOutput:
    def __init__(self, joints: torch.Tensor):
        self.joints = joints


class AMASSConversionTests(unittest.TestCase):
    def test_loads_and_resamples_raw_stageii_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "walk_stageii.npz"
            frames = 8
            trans = np.arange(frames * 3, dtype=np.float32).reshape(frames, 3)
            poses = np.zeros((frames, 66), dtype=np.float32)
            poses[:, 2] = np.arange(frames, dtype=np.float32)
            np.savez(
                input_path,
                trans=trans,
                poses=poses,
                betas=np.arange(10, dtype=np.float32),
                mocap_frame_rate=np.float32(60.0),
            )
            parameters = load_amass_parameters(input_path, fps=30.0)

        self.assertEqual(parameters.num_frames, 4)
        self.assertEqual(parameters.body_pose.shape, (4, 63))
        self.assertEqual(parameters.betas.shape, (4, 10))
        np.testing.assert_array_equal(parameters.transl.numpy(), trans[[0, 2, 4, 6]])
        self.assertEqual(parameters.source_fps, 60.0)
        self.assertEqual(parameters.fps, 30.0)

    def test_batched_fk_height_and_root_orientation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "turn_stageii.npz"
            frames = 5
            poses = np.zeros((frames, 66), dtype=np.float32)
            poses[:, 2] = np.pi / 2
            np.savez(
                input_path,
                trans=np.zeros((frames, 3), dtype=np.float32),
                poses=poses,
                betas=np.zeros(10, dtype=np.float32),
                mocap_frame_rate=np.float32(30.0),
            )
            parameters = load_amass_parameters(input_path)

        batch_sizes: list[int] = []

        def fake_forward(_model, *, body_pose, **_kwargs):
            batch_sizes.append(body_pose.shape[0])
            joints = torch.ones(body_pose.shape[0], len(AMASS_DEMO_JOINTS), 3)
            return _FakeOutput(joints)

        with mock.patch(
            "holosoma_retargeting.data_utils.prep_amass_smplx_for_rt.create_smplx_model",
            return_value=object(),
        ), mock.patch(
            "holosoma_retargeting.data_utils.prep_amass_smplx_for_rt.forward_smplx_model",
            side_effect=fake_forward,
        ), mock.patch(
            "holosoma_retargeting.data_utils.prep_amass_smplx_for_rt.compute_smplx_height",
            return_value=1.83,
        ):
            motion = convert_amass_parameters(parameters, model_path="unused", batch_size=2)

        self.assertEqual(batch_sizes, [2, 2, 1])
        self.assertEqual(motion.global_joint_positions.shape, (5, 22, 3))
        self.assertAlmostEqual(motion.height, 1.83)
        expected = np.tile([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)], (5, 1))
        np.testing.assert_allclose(motion.root_quaternions_wxyz, expected, atol=1e-6)

    def test_saved_output_uses_unified_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            input_path = directory / "source_stageii.npz"
            input_path.touch()
            parameters = type(
                "Motion",
                (),
                {
                    "global_joint_positions": np.zeros((2, 22, 3), dtype=np.float32),
                    "root_quaternions_wxyz": np.tile([1.0, 0.0, 0.0, 0.0], (2, 1)).astype(np.float32),
                    "height": 1.8,
                    "source_fps": 60.0,
                    "fps": 30.0,
                },
            )()
            output_path = directory / "converted.npz"
            save_converted_amass(parameters, output_path, input_path)

            with np.load(output_path, allow_pickle=False) as data:
                self.assertEqual(data["joint_names"].tolist(), AMASS_DEMO_JOINTS)
                self.assertEqual(str(data["source_format"]), "amass")
                self.assertEqual(str(data["coordinate_system"]), "right_handed_z_up")
                self.assertEqual(float(data["fps"]), 30.0)


if __name__ == "__main__":
    unittest.main()
