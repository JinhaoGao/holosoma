# ruff: noqa: PT009

from __future__ import annotations

import json
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import mujoco
import numpy as np
import trimesh
import yourdfpy

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "holosoma_retargeting"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from holosoma_retargeting.data_utils.object_assets import (  # noqa: E402
    OMOMO_MESH_SHA256,
    _atomic_write_xml,
    create_omomo_object_scene,
    create_scaled_omomo_object_urdf,
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
                self.assertGreater(result.collision_part_count, 0)
                self.assertTrue(result.sha256_matches)
                self.assertTrue(np.all(np.asarray(result.extents) > 1e-3))

    def test_every_object_urdf_loads_with_visual_and_convex_collisions(self):
        for asset in get_all_omomo_object_assets():
            with self.subTest(object_name=asset.name):
                model = yourdfpy.URDF.load(asset.urdf_path, load_meshes=False)
                self.assertIn(asset.body_name, model.link_map)
                root = ET.parse(asset.urdf_path).getroot()  # noqa: S314
                self.assertEqual(len(root.findall(".//visual")), 1)
                self.assertEqual(
                    len(root.findall(".//collision")),
                    len(asset.collision_mesh_paths),
                )

    def test_clothesstand_collision_is_not_the_global_convex_hull(self):
        asset = next(
            asset
            for asset in get_all_omomo_object_assets()
            if asset.name == "clothesstand"
        )
        visual_mesh = trimesh.load(asset.mesh_path, force="mesh", process=False)
        collision_volume = sum(
            abs(trimesh.load(path, force="mesh", process=False).volume)
            for path in asset.collision_mesh_paths
        )
        global_hull_volume = abs(visual_mesh.convex_hull.volume)
        self.assertLess(collision_volume / global_hull_volume, 0.2)

    def test_collision_manifest_covers_the_catalog(self):
        models_root = default_models_root()
        manifest = json.loads(
            (models_root / "omomo_collision_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["generator"]["name"], "CoACD")
        self.assertEqual(set(manifest["objects"]), set(OMOMO_OBJECT_NAMES))
        for asset in get_all_omomo_object_assets(models_root=models_root):
            with self.subTest(object_name=asset.name):
                record = manifest["objects"][asset.name]
                self.assertEqual(
                    record["collision_part_count"],
                    len(asset.collision_mesh_paths),
                )
                self.assertEqual(
                    {models_root / item["file"] for item in record["parts"]},
                    set(asset.collision_mesh_paths),
                )


class OmomoObjectSceneTests(unittest.TestCase):
    ROBOTS = {
        "g1": ("g1/g1_29dof.xml", 29),
        "e1": ("e1/e1_23dof.xml", 23),
    }

    def test_generated_xml_is_published_atomically(self):
        started = Event()
        release = Event()

        class SlowTree:
            def write(self, target, *, encoding, xml_declaration):
                self.assertions = (encoding, xml_declaration)
                target.write(b"<new")
                target.flush()
                started.set()
                if not release.wait(timeout=5):
                    raise TimeoutError("Test did not release the delayed XML writer")
                target.write(b" />")

        with tempfile.TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir) / "scene.xml"
            destination.write_text("<old />", encoding="utf-8")
            tree = SlowTree()
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_atomic_write_xml, tree, destination)
                self.assertTrue(started.wait(timeout=5))
                self.assertEqual(destination.read_text(encoding="utf-8"), "<old />")
                release.set()
                future.result(timeout=5)

            self.assertEqual(ET.parse(destination).getroot().tag, "new")  # noqa: S314
            self.assertEqual(tree.assertions, ("utf-8", True))
            self.assertFalse(list(destination.parent.glob(".scene.xml.*.tmp")))

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
                            mujoco.mj_name2id(
                                model,
                                mujoco.mjtObj.mjOBJ_GEOM,
                                f"{object_name}_visual",
                            ),
                            0,
                        )
                        asset = next(
                            item
                            for item in get_all_omomo_object_assets(models_root=models_root)
                            if item.name == object_name
                        )
                        visual_id = mujoco.mj_name2id(
                            model,
                            mujoco.mjtObj.mjOBJ_GEOM,
                            asset.visual_geom_name,
                        )
                        self.assertEqual(int(model.geom_contype[visual_id]), 0)
                        self.assertEqual(int(model.geom_conaffinity[visual_id]), 0)
                        for index in range(len(asset.collision_mesh_paths)):
                            collision_id = mujoco.mj_name2id(
                                model,
                                mujoco.mjtObj.mjOBJ_GEOM,
                                asset.collision_geom_name(index),
                            )
                            self.assertGreaterEqual(collision_id, 0)
                            self.assertEqual(int(model.geom_contype[collision_id]), 1)
                            self.assertEqual(int(model.geom_conaffinity[collision_id]), 1)
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
            mesh = root.find("./asset/mesh[@name='smallbox_visual_mesh']")
            self.assertIsNotNone(mesh)
            self.assertEqual(mesh.get("scale"), "0.5 1 1.5")
            collision_meshes = [
                item
                for item in root.findall("./asset/mesh")
                if item.get("name", "").startswith("smallbox_collision_mesh_")
            ]
            self.assertTrue(collision_meshes)
            self.assertTrue(
                all(item.get("scale") == "0.5 1 1.5" for item in collision_meshes)
            )

            urdf_path = create_scaled_omomo_object_urdf(
                "smallbox",
                (0.5, 1.0, 1.5),
                models_root=models_root,
                output_dir=tmpdir,
            )
            urdf_root = ET.parse(urdf_path).getroot()  # noqa: S314
            urdf_meshes = urdf_root.findall(".//mesh")
            smallbox_asset = next(
                asset
                for asset in get_all_omomo_object_assets(models_root=models_root)
                if asset.name == "smallbox"
            )
            self.assertEqual(
                len(urdf_meshes),
                1 + len(smallbox_asset.collision_mesh_paths),
            )
            self.assertTrue(all(item.get("scale") == "0.5 1 1.5" for item in urdf_meshes))
            self.assertEqual(
                Path(urdf_meshes[0].get("filename", "")),
                smallbox_asset.mesh_path.resolve(),
            )
            self.assertEqual(
                {
                    Path(item.get("filename", ""))
                    for item in urdf_meshes[1:]
                },
                {path.resolve() for path in smallbox_asset.collision_mesh_paths},
            )
            yourdfpy.URDF.load(urdf_path)


if __name__ == "__main__":
    unittest.main()
