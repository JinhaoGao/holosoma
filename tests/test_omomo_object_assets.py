# ruff: noqa: PT009, PT027

from __future__ import annotations

import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import yourdfpy

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "holosoma_retargeting"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from holosoma_retargeting.data_utils.object_assets import (  # noqa: E402
    OMOMO_MESH_SHA256,
    create_omomo_object_scene,
    default_models_root,
    get_all_omomo_object_assets,
    validate_all_omomo_object_assets,
)
from holosoma_retargeting.data_utils.omomo import OMOMO_OBJECT_NAMES  # noqa: E402


class OmomoObjectCatalogTests(unittest.TestCase):
    def test_catalog_is_complete_and_assets_are_valid(self):
        assets = get_all_omomo_object_assets()
        self.assertEqual(tuple(asset.name for asset in assets), OMOMO_OBJECT_NAMES)
        self.assertEqual(set(OMOMO_MESH_SHA256), set(OMOMO_OBJECT_NAMES))

        validations = validate_all_omomo_object_assets()
        self.assertEqual(len(validations), len(OMOMO_OBJECT_NAMES))
        for result in validations:
            with self.subTest(object_name=result.object_name):
                self.assertGreater(result.vertex_count, 3)
                self.assertGreater(result.face_count, 3)
                self.assertTrue(result.sha256_matches)
                self.assertTrue(np.all(np.asarray(result.extents) > 1e-3))

    def test_every_object_urdf_loads_with_one_named_link(self):
        for asset in get_all_omomo_object_assets():
            with self.subTest(object_name=asset.name):
                model = yourdfpy.URDF.load(asset.urdf_path, load_meshes=False)
                self.assertIn(asset.body_name, model.link_map)


class OmomoObjectSceneTests(unittest.TestCase):
    ROBOTS = {
        "g1": ("g1/g1_29dof.xml", 29),
        "e1": ("e1/e1_23dof.xml", 23),
    }

    def test_all_g1_and_e1_object_scenes_compile(self):
        models_root = default_models_root()
        with tempfile.TemporaryDirectory() as tmpdir:
            for robot_name, (relative_xml, robot_dof) in self.ROBOTS.items():
                robot_xml = models_root / relative_xml
                for object_name in OMOMO_OBJECT_NAMES:
                    with self.subTest(robot=robot_name, object_name=object_name):
                        scene_path = create_omomo_object_scene(
                            robot_xml,
                            object_name,
                            models_root=models_root,
                            output_dir=tmpdir,
                        )
                        model = mujoco.MjModel.from_xml_path(str(scene_path))
                        self.assertEqual(model.nq, 7 + robot_dof + 7)
                        self.assertGreaterEqual(
                            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{object_name}_link"),
                            0,
                        )
                        self.assertGreaterEqual(
                            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, object_name),
                            0,
                        )
                        joint_id = mujoco.mj_name2id(
                            model,
                            mujoco.mjtObj.mjOBJ_JOINT,
                            f"{object_name}_freejoint",
                        )
                        self.assertGreaterEqual(joint_id, 0)
                        self.assertEqual(int(model.jnt_qposadr[joint_id]), 7 + robot_dof)

    def test_scene_records_anisotropic_object_scale(self):
        models_root = default_models_root()
        with tempfile.TemporaryDirectory() as tmpdir:
            scene_path = create_omomo_object_scene(
                models_root / "g1/g1_29dof.xml",
                "smallbox",
                scale=(0.5, 1.0, 1.5),
                models_root=models_root,
                output_dir=tmpdir,
            )
            root = ET.parse(scene_path).getroot()  # noqa: S314
            mesh = root.find("./asset/mesh[@name='smallbox_mesh']")
            self.assertIsNotNone(mesh)
            self.assertEqual(mesh.get("scale"), "0.5 1 1.5")


if __name__ == "__main__":
    unittest.main()
