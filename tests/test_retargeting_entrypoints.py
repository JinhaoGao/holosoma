# ruff: noqa: CPY001, PT009, PT027

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "holosoma_retargeting"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from holosoma_retargeting.config_types.retargeting import (  # noqa: E402
    DATASET_DEFAULT_PATHS,
    RetargetingCommand,
    RetargetingConfig,
    internal_config_from_command,
)
from holosoma_retargeting.examples import (  # noqa: E402
    PUBLIC_RETARGETING_COMMANDS,
    parallel_robot_retarget,
    robot_retarget,
)
from holosoma_retargeting.retargeting_pipeline import (  # noqa: E402
    RetargetVariant,
    _canonical_result_path,
)


class RetargetingEntrypointTests(unittest.TestCase):
    def test_only_two_production_commands_are_registered(self):
        self.assertEqual(
            PUBLIC_RETARGETING_COMMANDS,
            ("robot_retarget", "parallel_robot_retarget"),
        )

    def test_single_entry_forces_identity_and_uses_single_result_root(self):
        cfg = RetargetingConfig(augmentation=True)
        sentinel = object()
        with mock.patch.object(
            robot_retarget,
            "run_retargeting_family",
            return_value=sentinel,
        ) as runner:
            result = robot_retarget.run_config(cfg)

        self.assertIs(result, sentinel)
        submitted = runner.call_args.args[0]
        self.assertFalse(submitted.augmentation)
        self.assertEqual(submitted.save_dir, robot_retarget.DEFAULT_RESULTS_ROOT)
        self.assertTrue(cfg.augmentation)
        self.assertIsNone(cfg.save_dir)

    def test_augmented_entry_forces_variants_and_uses_parallel_result_root(self):
        cfg = RetargetingConfig(augmentation=False)
        sentinel = object()
        with mock.patch.object(
            parallel_robot_retarget,
            "run_retargeting_family",
            return_value=sentinel,
        ) as runner:
            result = parallel_robot_retarget.run_config(cfg)

        self.assertIs(result, sentinel)
        submitted = runner.call_args.args[0]
        self.assertTrue(submitted.augmentation)
        self.assertEqual(
            submitted.save_dir,
            parallel_robot_retarget.DEFAULT_RESULTS_ROOT,
        )
        self.assertFalse(cfg.augmentation)
        self.assertIsNone(cfg.save_dir)

    def test_augmented_entry_rejects_robot_only(self):
        with self.assertRaisesRegex(
            ValueError,
            "only object_interaction and climbing",
        ):
            parallel_robot_retarget.run_config(
                RetargetingConfig(task_type="robot_only"),
            )

    def test_explicit_result_root_is_preserved(self):
        custom_root = Path("/tmp/retargeting-results")
        for module in (robot_retarget, parallel_robot_retarget):
            with self.subTest(module=module.__name__):
                cfg = RetargetingConfig(save_dir=custom_root)
                with mock.patch.object(module, "run_retargeting_family") as runner:
                    module.run_config(cfg)
                self.assertEqual(runner.call_args.args[0].save_dir, custom_root)

    def test_result_path_is_partitioned_by_robot_and_dataset(self):
        result = _canonical_result_path(
            results_root=Path("/results"),
            robot="e2",
            task_type="robot_only",
            dataset_partition="noetix_mocap",
            sequence_key="session/walk",
            variant=RetargetVariant(
                name="translated",
                translation=(0.2, 0.0, 0.0),
            ),
        )

        self.assertEqual(
            result,
            Path(
                "/results/e2/robot_only/noetix_mocap/session/walk/translated.npz",
            ),
        )
        self.assertNotIn("canonical", result.parts)

    def test_compact_command_resolves_dataset_defaults(self):
        command = RetargetingCommand(
            task="robot_only",
            robot="e2",
            dataset="noetix_mocap",
            motion="walk/clip",
        )
        config = internal_config_from_command(command)
        self.assertEqual(config.task_type, "robot_only")
        self.assertEqual(config.robot, "e2")
        self.assertEqual(config.data_format, "noetix_mocap")
        self.assertEqual(config.task_name, "walk/clip")
        self.assertEqual(config.data_path, DATASET_DEFAULT_PATHS["noetix_mocap"])

    def test_orientation_tracking_is_off_by_default(self):
        config = internal_config_from_command(
            RetargetingCommand(
                task="robot_only",
                robot="g1",
                dataset="noetix_mocap",
            ),
        )
        self.assertEqual(config.retargeter.orientation_weights, {})

    def test_orientation_switch_enables_all_mapped_links_with_equal_weights(self):
        config = internal_config_from_command(
            RetargetingCommand(
                task="robot_only",
                robot="e2",
                dataset="gvhmr",
                orientation=True,
            ),
        )
        self.assertEqual(len(config.retargeter.orientation_weights), 15)
        self.assertEqual(set(config.retargeter.orientation_weights.values()), {1.0})

    def test_orientation_profile_accepts_keypoint_and_link_weight_names(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            profile_path = Path(temporary_dir) / "shoulders.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "weights": {
                            "LeftArm": 2.5,
                            "r_arm_shoulder_yaw_link": 0.25,
                        },
                    },
                ),
                encoding="utf-8",
            )
            config = internal_config_from_command(
                RetargetingCommand(
                    task="robot_only",
                    robot="e1",
                    dataset="noetix_mocap",
                    orientation_config=profile_path,
                ),
            )

        self.assertEqual(
            config.retargeter.orientation_weights,
            {
                "LeftArm": 2.5,
                "RightArm": 0.25,
            },
        )

    def test_disabled_orientation_profile_preserves_position_only_path(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            profile_path = Path(temporary_dir) / "disabled.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "enabled": False,
                        "weights": {"LeftArm": 2.5},
                    },
                ),
                encoding="utf-8",
            )
            config = internal_config_from_command(
                RetargetingCommand(
                    task="robot_only",
                    robot="g1",
                    dataset="noetix_mocap",
                    orientation_config=profile_path,
                ),
            )

        self.assertEqual(config.retargeter.orientation_weights, {})

    def test_orientation_profile_rejects_unknown_or_duplicate_names(self):
        for payload in (
            {"weights": {"unknown_link": 1.0}},
            {
                "weights": {
                    "LeftArm": 1.0,
                    "left_shoulder_yaw_link": 1.0,
                },
            },
            {"weights": {"LeftArm": -1.0}},
            {"weights": {"LeftArm": 0.0}},
        ):
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as temporary_dir:
                profile_path = Path(temporary_dir) / "invalid.json"
                profile_path.write_text(
                    json.dumps(payload),
                    encoding="utf-8",
                )
                with self.assertRaises(ValueError):
                    internal_config_from_command(
                        RetargetingCommand(
                            task="robot_only",
                            robot="g1",
                            dataset="noetix_mocap",
                            orientation_config=profile_path,
                        ),
                    )

    def test_public_command_has_only_compact_production_options(self):
        self.assertEqual(
            tuple(field.name for field in fields(RetargetingCommand)),
            (
                "task",
                "robot",
                "dataset",
                "motion",
                "data_path",
                "save_dir",
                "overwrite",
                "orientation",
                "orientation_config",
            ),
        )

    def test_public_main_accepts_only_compact_command(self):
        command = RetargetingCommand(motion="sub3_tripod_001")
        with mock.patch.object(robot_retarget, "run_config") as runner:
            robot_retarget.main(command)
        submitted = runner.call_args.args[0]
        self.assertIsInstance(submitted, RetargetingConfig)
        self.assertEqual(submitted.task_name, "sub3_tripod_001")

    def test_production_task_matrix_accepts_required_combinations(self):
        datasets = (
            "climbing",
            "gvhmr",
            "lafan",
            "noetix_csv_climb",
            "noetix_mocap",
            "OMOMO_new",
        )
        for robot in ("g1", "e1", "e2"):
            for dataset in datasets:
                with self.subTest(task="robot_only", robot=robot, dataset=dataset):
                    config = internal_config_from_command(
                        RetargetingCommand(
                            task="robot_only",
                            robot=robot,
                            dataset=dataset,
                        ),
                    )
                    self.assertEqual(config.dataset, dataset)

        for dataset in ("climbing", "noetix_csv_climb"):
            with self.subTest(task="climbing", dataset=dataset):
                internal_config_from_command(
                    RetargetingCommand(
                        task="climbing",
                        robot="g1",
                        dataset=dataset,
                    ),
                )
        internal_config_from_command(
            RetargetingCommand(
                task="object_interaction",
                robot="g1",
                dataset="OMOMO_new",
            ),
        )

    def test_production_task_matrix_rejects_out_of_scope_combinations(self):
        for command in (
            RetargetingCommand(
                task="object_interaction",
                robot="e1",
                dataset="OMOMO_new",
            ),
            RetargetingCommand(
                task="climbing",
                robot="e2",
                dataset="climbing",
            ),
            RetargetingCommand(
                task="object_interaction",
                robot="g1",
                dataset="lafan",
            ),
        ):
            with self.subTest(command=command), self.assertRaises(ValueError):
                internal_config_from_command(command)


if __name__ == "__main__":
    unittest.main()
