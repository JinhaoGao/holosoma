# ruff: noqa: CPY001, E402, I001, PT009, PT027

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

from holosoma_retargeting.config_types.viser import ViserConfig
from holosoma_retargeting.data_conversion.convert_data_format_mj import MotionLoader
from holosoma_retargeting.src.viser_utils import format_foot_sticking_status
from holosoma_retargeting.viser_player import (
    ObjectKeypointOverlay,
    _mesh_color_override,
    _resolve_data_format,
    _resolve_robot_mujoco_xml,
    _resolve_runtime_config,
    _saved_foot_sticking_constraint_status,
    load_npz,
    main as single_viewer_main,
    resolve_input_kind,
)
from holosoma_retargeting.visualization.layers import (
    LAYER_SPECS,
    LayerController,
    LayerId,
)
from holosoma_retargeting.visualization.orientation import (
    load_orientation_diagnostics,
)


class ResultVisualizationTests(unittest.TestCase):
    def test_canonical_layer_ids_and_labels_are_unique(self):
        self.assertEqual(
            len({spec.layer_id for spec in LAYER_SPECS}),
            len(LAYER_SPECS),
        )
        self.assertEqual(
            len({spec.label for spec in LAYER_SPECS}),
            len(LAYER_SPECS),
        )
        self.assertIn(LayerId.INTERACTION_MESH, {spec.layer_id for spec in LAYER_SPECS})
        self.assertIn(LayerId.FOOT_STICKING, {spec.layer_id for spec in LAYER_SPECS})
        hotkeys = [spec.hotkey for spec in LAYER_SPECS if spec.hotkey is not None]
        self.assertEqual(len(hotkeys), len(set(hotkeys)))

    def test_layer_controller_keeps_availability_and_visibility_separate(self):
        changes = []
        controller = LayerController()
        controller.register(
            LayerId.ROBOT_MESH,
            available=True,
            visible=True,
            callback=changes.append,
        )
        controller.register(
            LayerId.OBJECT_MESH,
            available=False,
            visible=True,
            callback=lambda _visible: self.fail("unavailable callback must not run"),
        )

        controller.toggle(LayerId.ROBOT_MESH)
        controller.set_visible(LayerId.OBJECT_MESH, True)

        self.assertEqual(changes, [True, False])
        self.assertFalse(controller.is_visible(LayerId.ROBOT_MESH))
        self.assertFalse(controller.is_visible(LayerId.OBJECT_MESH))

    def test_single_viewer_auto_detects_all_supported_input_adapters(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            result_path = directory / "result.npz"
            converted_path = directory / "converted.npz"
            raw_path = directory / "raw.npz"
            np.savez(result_path, qpos=np.zeros((1, 8), dtype=np.float32))
            np.savez(
                converted_path,
                joint_pos=np.zeros((1, 8), dtype=np.float32),
                body_pos_w=np.zeros((1, 1, 3), dtype=np.float32),
                body_lin_vel_w=np.zeros((1, 1, 3), dtype=np.float32),
            )
            np.savez(
                raw_path,
                global_joint_positions=np.zeros((1, 1, 3), dtype=np.float32),
            )

            self.assertEqual(resolve_input_kind(result_path), "result")
            self.assertEqual(resolve_input_kind(converted_path), "converted")
            self.assertEqual(resolve_input_kind(raw_path), "raw")
            self.assertEqual(resolve_input_kind(directory), "raw")

    def test_single_viewer_requires_an_explicit_input(self):
        with self.assertRaisesRegex(
            ValueError,
            "requires --input-path",
        ):
            single_viewer_main(ViserConfig())

    def test_empty_saved_orientation_arrays_are_an_unavailable_layer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "empty_orientation.npz"
            np.savez(
                path,
                orientation_human_joint_names=np.asarray([], dtype=str),
                orientation_robot_link_names=np.asarray([], dtype=str),
                orientation_weights=np.empty((0,), dtype=np.float32),
                orientation_target_quaternions_wxyz=np.empty((2, 0, 4), dtype=np.float32),
                orientation_robot_quaternions_wxyz=np.empty((2, 0, 4), dtype=np.float32),
                orientation_errors_rad=np.empty((2, 0), dtype=np.float32),
            )

            diagnostics = load_orientation_diagnostics(path, expected_frames=2)

        self.assertIsNone(diagnostics)

    def test_only_two_public_visualization_scripts_remain(self):
        package = PACKAGE_ROOT / "holosoma_retargeting"
        public_scripts = {
            path.name
            for path in package.glob("*.py")
            if "__main__" in path.read_text(encoding="utf-8") and "Viser" in path.read_text(encoding="utf-8")
        }
        self.assertEqual(
            public_scripts,
            {"viser_player.py", "multi_viser_player.py"},
        )
        self.assertFalse((package / "augmentation_viser_player.py").exists())
        self.assertFalse((package / "examples" / "ablation_viser_player.py").exists())
        self.assertFalse((package / "examples" / "raw_human_motion_viewer.py").exists())
        self.assertFalse((package / "data_conversion" / "viser_body_vel_player.py").exists())

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
                foot_sticking_side_names=np.asarray(["left", "right"]),
                foot_sticking_states=np.asarray(
                    [
                        [True, False],
                        [False, True],
                    ],
                    dtype=bool,
                ),
                foot_sticking_enabled_for_saved_trajectory=np.asarray(True),
                foot_sticking_fallback_frames=np.asarray([1], dtype=np.int32),
                foot_sticking_release_frames=np.empty(0, dtype=np.int32),
                object_points_demo_local=np.zeros((4, 3), dtype=np.float32),
                object_points_target_local=np.ones((4, 3), dtype=np.float32),
                object_points_demo_world=np.zeros((2, 4, 3), dtype=np.float32),
                object_points_target_world=np.ones((2, 4, 3), dtype=np.float32),
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
        np.testing.assert_array_equal(
            metadata["foot_sticking"]["states"],
            np.asarray(
                [
                    [True, False],
                    [False, True],
                ],
                dtype=bool,
            ),
        )
        self.assertEqual(
            _saved_foot_sticking_constraint_status(metadata["foot_sticking"], 1),
            "active with relaxed tolerance",
        )
        self.assertEqual(metadata["object_keypoints"]["demo_local"].shape, (4, 3))
        self.assertEqual(metadata["object_keypoints"]["target_local"].shape, (4, 3))
        self.assertEqual(metadata["object_keypoints"]["demo_world"].shape, (2, 4, 3))
        self.assertEqual(metadata["object_keypoints"]["target_world"].shape, (2, 4, 3))
        self.assertEqual(interaction_mesh["source_vertices"].shape, (2, 3, 3))

    def test_foot_sticking_status_uses_green_and_red_lights(self):
        status = format_foot_sticking_status(
            12,
            (True, False),
            constraint_status="active",
        )

        self.assertIn("Frame:** `12`", status)
        self.assertIn("🟢 **Left:** `sticking=True`", status)
        self.assertIn("🔴 **Right:** `sticking=False`", status)
        self.assertIn("Hard constraint:** `active`", status)

    def test_result_loader_rejects_mismatched_foot_sticking_frames(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = Path(tmpdir) / "invalid_sticking_result.npz"
            np.savez(
                result_path,
                qpos=np.zeros((2, 36), dtype=np.float32),
                foot_sticking_states=np.zeros((1, 2), dtype=bool),
            )

            with self.assertRaisesRegex(ValueError, r"shape \(2, 2\)"):
                load_npz(str(result_path))

    def test_object_keypoint_overlay_rejects_mismatched_saved_sequences(self):
        with self.assertRaisesRegex(ValueError, r"matching \(frames, points, 3\)"):
            ObjectKeypointOverlay(
                server=None,
                keypoint_data={
                    "demo_world": np.zeros((2, 4, 3), dtype=np.float32),
                    "target_world": np.zeros((2, 5, 3), dtype=np.float32),
                },
                point_radius=0.02,
            )

    def test_legacy_interaction_mesh_supplies_object_keypoint_overlay(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = Path(tmpdir) / "legacy_result.npz"
            source_vertices = np.arange(2 * 5 * 3, dtype=np.float32).reshape(2, 5, 3)
            target_vertices = source_vertices + 1.0
            np.savez(
                result_path,
                qpos=np.zeros((2, 43), dtype=np.float32),
                interaction_source_vertices_w=source_vertices,
                interaction_target_vertices_w=target_vertices,
                interaction_tetrahedra=np.zeros((2, 1, 4), dtype=np.int32),
                interaction_tetrahedra_counts=np.ones(2, dtype=np.int32),
                interaction_num_human_vertices=np.int32(2),
            )

            _, _, _, metadata, _ = load_npz(str(result_path))

        np.testing.assert_array_equal(
            metadata["object_keypoints"]["demo_world"],
            source_vertices[:, 2:],
        )
        np.testing.assert_array_equal(
            metadata["object_keypoints"]["target_world"],
            target_vertices[:, 2:],
        )

    def test_runtime_config_uses_saved_robot_object_and_format_metadata(self):
        metadata = {
            "robot_type": "g1",
            "object_urdf": "models/largebox/largebox.urdf",
            "source_data_format": "gvhmr",
        }
        config = _resolve_runtime_config(ViserConfig(qpos_npz="result.npz"), metadata)
        self.assertTrue(Path(config.robot_urdf).is_absolute())
        self.assertTrue(Path(config.robot_urdf).is_file())
        self.assertTrue(config.robot_urdf.endswith("models/g1/g1_29dof.urdf"))
        self.assertTrue(Path(config.object_urdf).is_absolute())
        self.assertTrue(Path(config.object_urdf).is_file())
        self.assertTrue(config.object_urdf.endswith("models/largebox/largebox.urdf"))
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
        self.assertTrue(Path(config.robot_urdf).is_absolute())
        self.assertTrue(Path(config.robot_urdf).is_file())
        self.assertTrue(config.robot_urdf.endswith("models/e1/e1_23dof.urdf"))
        self.assertTrue(config.object_urdf.endswith("models/tripod/tripod.urdf"))

    def test_runtime_config_resolves_packaged_g1_and_e1_assets_outside_package_directory(self):
        for robot_type, model_name in (
            ("g1", "g1_29dof"),
            ("e1", "e1_23dof"),
        ):
            with self.subTest(robot_type=robot_type):
                config = _resolve_runtime_config(
                    ViserConfig(qpos_npz="result.npz"),
                    {
                        "robot_type": robot_type,
                        "contains_object_in_qpos": False,
                    },
                )
                robot_urdf = Path(config.robot_urdf)
                robot_xml = _resolve_robot_mujoco_xml(config)
                self.assertTrue(robot_urdf.is_absolute())
                self.assertTrue(robot_urdf.is_file())
                self.assertEqual(robot_urdf.name, f"{model_name}.urdf")
                self.assertIsNotNone(robot_xml)
                self.assertTrue(robot_xml.is_file())

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
