# ruff: noqa: PT009

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

from holosoma_retargeting.config_types.viser import ViserConfig  # noqa: E402
from holosoma_retargeting.data_conversion.convert_data_format_mj import MotionLoader  # noqa: E402
from holosoma_retargeting.viser_player import (  # noqa: E402
    _mesh_color_override,
    _resolve_data_format,
    _resolve_runtime_config,
    load_npz,
)


class ResultVisualizationTests(unittest.TestCase):
    def test_result_metadata_and_interaction_mesh_load_without_pickle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = Path(tmpdir) / "result.npz"
            np.savez(
                result_path,
                qpos=np.zeros((2, 36), dtype=np.float32),
                fps=np.float32(29.97),
                human_joints=np.zeros((2, 22, 3), dtype=np.float32),
                human_joint_names=np.asarray([f"joint_{index}" for index in range(22)]),
                mapped_human_joint_names=np.asarray(["Pelvis"]),
                mapped_robot_joints=np.zeros((2, 1, 3), dtype=np.float32),
                mapped_robot_link_names=np.asarray(["pelvis_contour_link"]),
                source_data_format=np.asarray("gvhmr"),
                robot_type=np.asarray("g1"),
                object_name=np.asarray("ground"),
                object_urdf=np.asarray(""),
                contains_object_in_qpos=np.asarray(False),
                interaction_source_vertices_w=np.zeros((2, 3, 3), dtype=np.float32),
                interaction_target_vertices_w=np.ones((2, 3, 3), dtype=np.float32),
                interaction_tetrahedra=np.zeros((2, 1, 4), dtype=np.int32),
                interaction_tetrahedra_counts=np.ones(2, dtype=np.int32),
                interaction_num_human_vertices=np.int32(1),
            )

            qpos, fps, human_joints, metadata, interaction_mesh = load_npz(str(result_path))

        self.assertEqual(qpos.shape, (2, 36))
        self.assertAlmostEqual(fps, 29.97, places=2)
        self.assertEqual(human_joints.shape, (2, 22, 3))
        self.assertEqual(metadata["source_data_format"], "gvhmr")
        self.assertEqual(metadata["robot_type"], "g1")
        self.assertFalse(metadata["contains_object_in_qpos"])
        self.assertEqual(interaction_mesh["source_vertices"].shape, (2, 3, 3))

    def test_runtime_config_uses_saved_robot_object_and_format_metadata(self):
        metadata = {
            "robot_type": "g1",
            "object_urdf": "models/largebox/largebox.urdf",
            "source_data_format": "gvhmr",
        }
        config = _resolve_runtime_config(ViserConfig(qpos_npz="result.npz"), metadata)
        self.assertEqual(config.robot_urdf, "models/g1/g1_29dof.urdf")
        self.assertEqual(config.object_urdf, "models/largebox/largebox.urdf")
        self.assertEqual(
            _resolve_data_format(
                config,
                np.zeros((1, 22, 3), dtype=np.float32),
                "g1",
                metadata,
            ),
            "gvhmr",
        )

    def test_runtime_config_resolves_catalog_object_from_result_metadata(self):
        metadata = {
            "robot_type": "e1",
            "object_name": "tripod",
            "object_urdf": "",
            "contains_object_in_qpos": True,
        }
        config = _resolve_runtime_config(
            ViserConfig(qpos_npz="sub2_tripod_019_original.npz"),
            metadata,
        )
        self.assertEqual(config.robot_urdf, "models/e1/e1_23dof.urdf")
        self.assertTrue(config.object_urdf.endswith("models/tripod/tripod.urdf"))

    def test_runtime_config_infers_legacy_result_object_from_filename(self):
        config = _resolve_runtime_config(
            ViserConfig(
                qpos_npz="sub2_whitechair_019_original.npz",
                robot_type="g1",
            ),
            {},
        )
        self.assertTrue(config.object_urdf.endswith("models/whitechair/whitechair.urdf"))

    def test_mesh_opacity_preserves_opaque_materials_and_clamps_translucency(self):
        self.assertIsNone(_mesh_color_override(1.0))
        self.assertEqual(_mesh_color_override(0.25), (0.7, 0.7, 0.7, 0.25))
        self.assertEqual(_mesh_color_override(-1.0), (0.7, 0.7, 0.7, 0.0))

    def test_training_converter_uses_fps_saved_in_result(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = Path(tmpdir) / "result.npz"
            np.savez(
                result_path,
                qpos=np.zeros((3, 36), dtype=np.float32),
                fps=np.float32(29.97),
            )
            motion = MotionLoader(
                motion_file=str(result_path),
                input_fps=30,
                output_fps=50,
                device=torch.device("cpu"),
                line_range=None,
                has_dynamic_object=False,
                use_omniretarget_data=False,
                robot_dof=29,
            )

        self.assertAlmostEqual(motion.input_fps, 29.97, places=2)
        self.assertAlmostEqual(motion.input_dt, 1.0 / 29.97, places=5)


if __name__ == "__main__":
    unittest.main()
