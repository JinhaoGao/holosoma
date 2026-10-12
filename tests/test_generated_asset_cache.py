# ruff: noqa: CPY001, PT009, PT027

from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mujoco
import trimesh
import yourdfpy

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "holosoma_retargeting"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from holosoma_retargeting.config_types.data_type import MotionDataConfig  # noqa: E402
from holosoma_retargeting.config_types.retargeting import RetargetingConfig  # noqa: E402
from holosoma_retargeting.config_types.robot import RobotConfig  # noqa: E402
from holosoma_retargeting.config_types.task import TaskConfig  # noqa: E402
from holosoma_retargeting.data_utils.object_assets import (  # noqa: E402
    create_omomo_object_scene,
    create_scaled_omomo_object_urdf,
    default_models_root,
)
from holosoma_retargeting.retargeting_pipeline import (  # noqa: E402
    RetargetVariant,
    build_retarget_job,
    create_task_constants,
    setup_object_data,
)


def _tree_snapshot(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class JobAssetPathTests(unittest.TestCase):
    def test_generated_assets_and_locks_stay_with_the_motion_family(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            source_dir = workspace / "motions"
            source_dir.mkdir()
            source = source_dir / "sub1_tripod_001.pt"
            source.write_bytes(b"stable source")
            config = RetargetingConfig(
                task_type="object_interaction",
                robot="g1",
                data_format="omomo",
                task_name=source.stem,
                data_path=source_dir,
                robot_config=RobotConfig(robot_type="g1"),
                motion_data_config=MotionDataConfig(
                    data_format="omomo",
                    robot_type="g1",
                    human_height=1.75,
                ),
                task_config=TaskConfig(),
            )
            results_root = workspace / "demo_results"
            identity_job = build_retarget_job(
                config,
                results_root=results_root,
                source_path=source,
            )
            self.assertEqual(
                identity_job.generated_assets_dir,
                identity_job.output_path.parent / ".assets" / "sub1_tripod_001",
            )
            self.assertTrue(identity_job.generated_assets_dir.is_relative_to(results_root))
            self.assertFalse(identity_job.generated_assets_dir.is_relative_to(source_dir))

            augmented_job = build_retarget_job(
                config,
                variant=RetargetVariant(
                    name="trans_0",
                    translation=(0.2, 0.0, 0.0),
                ),
                run_kind="augmentation",
                results_root=results_root,
                source_path=source,
            )
            self.assertEqual(
                augmented_job.generated_assets_dir,
                augmented_job.output_path.parent / ".assets" / "sub1_tripod_001_trans_0",
            )


class ClimbingGeneratedAssetTests(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace.cleanup)
        self.source_root = Path(self.workspace.name) / "climb"
        for sequence_index in range(5):
            source_dir = self.source_root / f"mocap_climb_seq_{sequence_index}"
            source_dir.mkdir(parents=True)
            trimesh.creation.box(extents=(1.0, 0.5, 0.1 * (sequence_index + 1))).export(source_dir / "multi_boxes.obj")
            (source_dir / "multi_boxes.urdf").write_text(
                '<robot name="terrain"><link name="terrain"><visual><geometry>'
                '<mesh filename="multi_boxes.obj"/></geometry></visual></link></robot>',
                encoding="utf-8",
            )
            (source_dir / "box_assets.xml").write_text(
                '<mujoco><asset><mesh name="terrain_mesh" file="multi_boxes.obj"/></asset></mujoco>',
                encoding="utf-8",
            )
            for robot, dof in (("g1", 29), ("e1", 23)):
                # Exercise scene relocation with self-contained models, independent
                # of private motion captures and hand-authored climbing scenes.
                root = ET.Element("mujoco", model=robot)
                ET.SubElement(root, "compiler", meshdir=".")
                ET.SubElement(root, "include", file="box_assets.xml")
                world = ET.SubElement(root, "worldbody")
                ET.SubElement(world, "geom", type="mesh", mesh="terrain_mesh")
                body = ET.SubElement(world, "body", name="robot", pos="0 0 2")
                ET.SubElement(body, "freejoint")
                ET.SubElement(body, "geom", type="sphere", size="0.1")
                for joint_index in range(dof):
                    child = ET.SubElement(body, "body", name=f"link_{joint_index}", pos="0 0 0.1")
                    ET.SubElement(child, "joint", type="hinge", axis="0 1 0")
                    ET.SubElement(child, "geom", type="sphere", size="0.01")
                scene_name = Path(RobotConfig(robot_type=robot).ROBOT_URDF_FILE).stem + "_w_multi_boxes.xml"
                ET.ElementTree(root).write(source_dir / scene_name)

    def test_g1_and_e1_assets_are_relocatable_and_do_not_modify_source(self):
        before = _tree_snapshot(self.source_root)
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_root = Path(tmpdir)
            for sequence_index in range(5):
                source_dir = self.source_root / f"mocap_climb_seq_{sequence_index}"
                for robot, expected_nq in (("g1", 36), ("e1", 30)):
                    with self.subTest(sequence=sequence_index, robot=robot):
                        task_config = TaskConfig(
                            object_name="multi_boxes",
                            object_dir=source_dir,
                        )
                        constants = create_task_constants(
                            RobotConfig(robot_type=robot),
                            MotionDataConfig(
                                data_format="mocap",
                                robot_type=robot,
                            ),
                            task_config,
                            "climbing",
                        )
                        _, _, object_urdf = setup_object_data(
                            "climbing",
                            constants,
                            source_dir,
                            0.73123456789,
                            task_config,
                            False,
                            generated_assets_dir=(cache_root / f"sequence-{sequence_index}" / robot),
                        )

                        object_urdf_path = Path(object_urdf)
                        scene_path = Path(constants.SCENE_XML_FILE)
                        self.assertTrue(object_urdf_path.is_relative_to(cache_root))
                        self.assertTrue(scene_path.is_relative_to(cache_root))
                        self.assertEqual(
                            mujoco.MjModel.from_xml_path(str(scene_path)).nq,
                            expected_nq,
                        )
                        yourdfpy.URDF.load(
                            object_urdf_path,
                            load_meshes=True,
                            build_scene_graph=True,
                        )
                        urdf_root = ET.parse(  # noqa: S314
                            object_urdf_path
                        ).getroot()
                        mesh_paths = tuple(Path(mesh.get("filename", "")) for mesh in urdf_root.findall(".//mesh"))
                        self.assertTrue(mesh_paths)
                        self.assertTrue(all(path.is_absolute() and path.is_file() for path in mesh_paths))

        self.assertEqual(before, _tree_snapshot(self.source_root))

    def test_source_directory_is_rejected_as_generated_output(self):
        source_dir = self.source_root / "mocap_climb_seq_0"
        forbidden = source_dir / ".generated-assets-test"
        self.assertFalse(forbidden.exists())
        task_config = TaskConfig(
            object_name="multi_boxes",
            object_dir=source_dir,
        )
        constants = create_task_constants(
            RobotConfig(robot_type="g1"),
            MotionDataConfig(data_format="mocap", robot_type="g1"),
            task_config,
            "climbing",
        )
        with self.assertRaisesRegex(ValueError, "outside source data"):
            setup_object_data(
                "climbing",
                constants,
                source_dir,
                0.75,
                task_config,
                False,
                generated_assets_dir=forbidden,
            )
        self.assertFalse(forbidden.exists())


class ConcurrentOmomoAssetTests(unittest.TestCase):
    def test_omomo_generators_require_an_explicit_output_directory(self):
        models_root = default_models_root()
        with self.assertRaisesRegex(TypeError, "output_dir"):
            create_omomo_object_scene(
                models_root / "g1" / "g1_29dof.xml",
                "smallbox",
                models_root=models_root,
            )
        with self.assertRaisesRegex(TypeError, "output_dir"):
            create_scaled_omomo_object_urdf(
                "smallbox",
                (0.5, 1.0, 1.5),
                models_root=models_root,
            )

    def test_shared_cache_generation_is_atomic_and_idempotent(self):
        models_root = default_models_root()
        robot_xml = models_root / "g1" / "g1_29dof.xml"
        scale = (0.5000000000000001, 1.0, 1.5)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            def generate() -> tuple[Path, Path]:
                return (
                    create_scaled_omomo_object_urdf(
                        "smallbox",
                        scale,
                        models_root=models_root,
                        output_dir=output_dir,
                    ),
                    create_omomo_object_scene(
                        robot_xml,
                        "smallbox",
                        scale=scale,
                        models_root=models_root,
                        output_dir=output_dir,
                    ),
                )

            with ThreadPoolExecutor(max_workers=8) as executor:
                generated = tuple(executor.map(lambda _: generate(), range(32)))

            self.assertEqual(len({pair[0] for pair in generated}), 1)
            self.assertEqual(len({pair[1] for pair in generated}), 1)
            object_urdf, scene = generated[0]
            yourdfpy.URDF.load(
                object_urdf,
                load_meshes=True,
                build_scene_graph=True,
            )
            self.assertEqual(
                mujoco.MjModel.from_xml_path(str(scene)).nq,
                43,
            )
            self.assertFalse(tuple(output_dir.glob(".*.tmp")))


if __name__ == "__main__":
    unittest.main()
