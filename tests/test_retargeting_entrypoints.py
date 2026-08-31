# ruff: noqa: CPY001, PT009, PT027

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import tyro

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "holosoma_retargeting"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from holosoma_retargeting.config_types.retargeting import (  # noqa: E402
    DATASET_DEFAULT_PATHS,
    RetargeterRuntimeOptions,
    RetargetingCommand,
    RetargetingConfig,
    internal_config_from_command,
)
from holosoma_retargeting.config_types.robot_profiles import (  # noqa: E402
    ORIENTATION_PROFILE_DATASETS,
    load_robot_profile,
)
from holosoma_retargeting.examples import (  # noqa: E402
    PUBLIC_RETARGETING_COMMANDS,
    parallel_robot_retarget,
    robot_retarget,
)
from holosoma_retargeting.retargeting_pipeline import (  # noqa: E402
    RetargetVariant,
    _canonical_result_path,
    build_retargeter_kwargs_from_config,
)

ROBOT_PROFILE_DIR = PACKAGE_ROOT / "holosoma_retargeting" / "examples" / "robot_profiles"


class RetargetingEntrypointTests(unittest.TestCase):
    def test_all_production_commands_are_registered(self):
        self.assertEqual(
            PUBLIC_RETARGETING_COMMANDS,
            ("robot_retarget", "parallel_robot_retarget", "paired_retargeting.robot_refine"),
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
                "/results/e2/robot_only/noetix_mocap/session/walk_translated.npz",
            ),
        )
        self.assertNotIn("canonical", result.parts)

    def test_output_name_replaces_only_the_artifact_filename(self):
        identity = _canonical_result_path(
            results_root=Path("/results"),
            robot="e2",
            task_type="robot_only",
            dataset_partition="fbx_mocap",
            sequence_key="session/Take_38_003_R__nan",
            variant=RetargetVariant(),
            output_name="Take_38_003_R__nan_o0.npz",
        )
        augmented = _canonical_result_path(
            results_root=Path("/results"),
            robot="e2",
            task_type="robot_only",
            dataset_partition="fbx_mocap",
            sequence_key="session/Take_38_003_R__nan",
            variant=RetargetVariant(
                name="translated",
                translation=(0.2, 0.0, 0.0),
            ),
            output_name="Take_38_003_R__nan_o0",
        )

        self.assertEqual(
            identity,
            Path("/results/e2/robot_only/fbx_mocap/session/Take_38_003_R__nan_o0.npz"),
        )
        self.assertEqual(
            augmented,
            Path("/results/e2/robot_only/fbx_mocap/session/Take_38_003_R__nan_o0_translated.npz"),
        )

    def test_output_name_rejects_paths_and_non_npz_suffixes(self):
        for output_name in ("nested/result.npz", "/tmp/result.npz", "result.csv", ".npz"):
            with self.subTest(output_name=output_name), self.assertRaises(ValueError):
                _canonical_result_path(
                    results_root=Path("/results"),
                    robot="e2",
                    task_type="robot_only",
                    dataset_partition="fbx_mocap",
                    sequence_key="motion",
                    variant=RetargetVariant(),
                    output_name=output_name,
                )

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

    def test_fbx_mocap_preset_uses_an_independent_motion_format(self):
        command = RetargetingCommand(
            task="robot_only",
            robot="g1",
            dataset="fbx_mocap",
            motion="Take_38_003_R__nv",
        )
        config = internal_config_from_command(command)

        self.assertEqual(config.dataset, "fbx_mocap")
        self.assertEqual(config.data_format, "fbx_mocap")
        self.assertEqual(config.data_path, DATASET_DEFAULT_PATHS["fbx_mocap"])

    def test_fbx_orientation_preview_does_not_enable_orientation_weights(self):
        config = internal_config_from_command(
            RetargetingCommand(
                task="robot_only",
                robot="g1",
                dataset="fbx_mocap",
                motion="Take_38_003_R__nv",
                orientation_preview=True,
            ),
        )

        self.assertTrue(config.retargeter.orientation_preview)
        self.assertEqual(config.retargeter.orientation_weights, {})

    def test_optional_objectives_are_off_by_default(self):
        config = internal_config_from_command(
            RetargetingCommand(
                task="robot_only",
                robot="g1",
                dataset="noetix_mocap",
            ),
        )
        self.assertEqual(config.retargeter.orientation_weights, {})
        self.assertEqual(config.retargeter.natural_pose_joint_positions, {})
        self.assertEqual(config.retargeter.natural_pose_weights, {})

    def test_live_visualization_is_on_by_default(self):
        config = internal_config_from_command(
            RetargetingCommand(
                task="robot_only",
                robot="e2",
                dataset="lafan",
            ),
        )

        self.assertTrue(config.retargeter.visualize)
        self.assertFalse(config.retargeter.debug)

    def test_public_retargeter_runtime_options_map_to_internal_config(self):
        config = internal_config_from_command(
            RetargetingCommand(
                task="robot_only",
                robot="e2",
                dataset="lafan",
                retargeter=RetargeterRuntimeOptions(
                    visualize=False,
                    debug=True,
                ),
            ),
        )

        self.assertFalse(config.retargeter.visualize)
        self.assertTrue(config.retargeter.debug)

    def test_legacy_visualization_flags_are_accepted_by_compact_cli(self):
        command = tyro.cli(
            RetargetingCommand,
            args=[
                "--task",
                "robot_only",
                "--robot",
                "e2",
                "--dataset",
                "lafan",
                "--motion",
                "walk2_subject3",
                "--retargeter.visualize",
                "--retargeter.debug",
            ],
        )

        self.assertTrue(command.retargeter.visualize)
        self.assertTrue(command.retargeter.debug)

    def test_live_visualization_can_be_disabled_for_headless_runs(self):
        command = tyro.cli(
            RetargetingCommand,
            args=["--retargeter.no-visualize"],
        )

        self.assertFalse(command.retargeter.visualize)

    def test_output_name_cli_maps_to_internal_config(self):
        command = tyro.cli(
            RetargetingCommand,
            args=["--output-name", "Take_38_003_R__nan_o0.npz"],
        )
        config = internal_config_from_command(command)

        self.assertEqual(command.output_name, "Take_38_003_R__nan_o0.npz")
        self.assertEqual(config.output_name, "Take_38_003_R__nan_o0.npz")

    def test_foot_sticking_is_on_by_default(self):
        command = tyro.cli(RetargetingCommand, args=[])
        config = internal_config_from_command(command)

        self.assertIsNone(command.foot_sticking)
        self.assertTrue(config.retargeter.activate_foot_sticking)

    def test_foot_sticking_cli_accepts_explicit_true_and_false(self):
        for cli_value, expected in (("True", True), ("False", False)):
            with self.subTest(cli_value=cli_value):
                command = tyro.cli(
                    RetargetingCommand,
                    args=["--foot-sticking", cli_value],
                )
                config = internal_config_from_command(command)

                self.assertIs(command.foot_sticking, expected)
                self.assertIs(
                    config.retargeter.activate_foot_sticking,
                    expected,
                )

    def test_both_production_entries_preserve_disabled_foot_sticking(self):
        command = tyro.cli(
            RetargetingCommand,
            args=["--foot-sticking", "False"],
        )
        for module in (robot_retarget, parallel_robot_retarget):
            with self.subTest(module=module.__name__), mock.patch.object(
                module,
                "run_config",
            ) as runner:
                module.main(command)

                submitted = runner.call_args.args[0]
                self.assertFalse(
                    submitted.retargeter.activate_foot_sticking,
                )

    def test_disabled_foot_sticking_reaches_solver_constructor_kwargs(self):
        command = tyro.cli(
            RetargetingCommand,
            args=["--foot-sticking", "False"],
        )
        config = internal_config_from_command(command)
        constants = SimpleNamespace(
            ORIENTATION_JOINTS_MAPPING={},
            ORIENTATION_T_POSE_HUMAN_QUATERNIONS_WXYZ={},
            ORIENTATION_T_POSE_ROBOT_BASE_QUATERNION_WXYZ=(1.0, 0.0, 0.0, 0.0),
            ORIENTATION_T_POSE_ROBOT_JOINT_POSITIONS={},
        )

        kwargs = build_retargeter_kwargs_from_config(
            config.retargeter,
            constants,
            object_urdf_path=None,
            task_type="robot_only",
        )

        self.assertFalse(kwargs["activate_foot_sticking"])

    def test_root_and_mesh_tuning_reach_solver_constructor_kwargs(self):
        command = tyro.cli(
            RetargetingCommand,
            args=[
                "--interaction-mesh-weight",
                "7.5",
                "--arm-interaction-mesh-weight-scale",
                "0.4",
                "--root-position-weight",
                "250",
                "--root-orientation-weight",
                "80",
            ],
        )
        config = internal_config_from_command(command)
        constants = SimpleNamespace(
            ORIENTATION_JOINTS_MAPPING={},
            ORIENTATION_T_POSE_HUMAN_QUATERNIONS_WXYZ={},
            ORIENTATION_T_POSE_ROBOT_BASE_QUATERNION_WXYZ=(1.0, 0.0, 0.0, 0.0),
            ORIENTATION_T_POSE_ROBOT_JOINT_POSITIONS={},
        )

        kwargs = build_retargeter_kwargs_from_config(
            config.retargeter,
            constants,
            object_urdf_path=None,
            task_type="robot_only",
        )

        self.assertEqual(kwargs["interaction_mesh_weight"], 7.5)
        self.assertEqual(kwargs["arm_interaction_mesh_weight_scale"], 0.4)
        self.assertEqual(kwargs["root_stability"].position_weight, 250.0)
        self.assertEqual(kwargs["root_stability"].orientation_weight, 80.0)

    def test_cli_values_override_robot_profile_defaults(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            profile = json.loads(
                (ROBOT_PROFILE_DIR / "g1.json").read_text(encoding="utf-8"),
            )
            profile["retargeting"]["root_position_weight"] = 12.0
            profile["retargeting"]["foot_sticking"] = False
            profile_path = Path(temporary_dir) / "g1.json"
            profile_path.write_text(json.dumps(profile), encoding="utf-8")

            inherited = internal_config_from_command(
                RetargetingCommand(robot_profile=profile_path),
            )
            overridden = internal_config_from_command(
                RetargetingCommand(
                    robot_profile=profile_path,
                    root_position_weight=1.0,
                    foot_sticking=True,
                ),
            )

        self.assertEqual(inherited.retargeter.root_stability.position_weight, 12.0)
        self.assertFalse(inherited.retargeter.activate_foot_sticking)
        self.assertEqual(overridden.retargeter.root_stability.position_weight, 1.0)
        self.assertTrue(overridden.retargeter.activate_foot_sticking)

    def test_orientation_uniform_weight_enables_all_mapped_links(self):
        config = internal_config_from_command(
            RetargetingCommand(
                task="robot_only",
                robot="e2",
                dataset="gvhmr",
                orientation_weights=0.125,
            ),
        )
        self.assertEqual(len(config.retargeter.orientation_weights), 15)
        self.assertEqual(
            set(config.retargeter.orientation_weights.values()),
            {0.125},
        )

    def test_orientation_uniform_weight_cli_accepts_a_number(self):
        command = tyro.cli(
            RetargetingCommand,
            args=["--orientation_weights", "1"],
        )

        self.assertEqual(command.orientation_weights, 1.0)

    def test_fbx_orientation_uniform_weight_enables_all_calibrated_links(self):
        config = internal_config_from_command(
            RetargetingCommand(
                task="robot_only",
                robot="g1",
                dataset="fbx_mocap",
                orientation_weights=0.1,
            ),
        )

        self.assertEqual(len(config.retargeter.orientation_weights), 15)
        self.assertEqual(set(config.retargeter.orientation_weights.values()), {0.1})

    def test_orientation_uniform_weight_rejects_invalid_values_and_bool(self):
        for value in (-0.1, float("inf"), float("nan"), True):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError,
                "finite and non-negative",
            ):
                internal_config_from_command(
                    RetargetingCommand(
                        task="robot_only",
                        robot="g1",
                        dataset="gvhmr",
                        orientation_weights=value,
                    ),
                )

    def test_robot_profiles_cover_every_orientation_dataset_and_robot(self):
        for robot in ("g1", "e1_23dof", "e1_24dof", "e2"):
            profile = load_robot_profile(robot=robot)
            raw = json.loads(profile.path.read_text(encoding="utf-8"))
            self.assertEqual(raw["schema_version"], 1)
            self.assertEqual(profile.robot, robot)
            self.assertEqual(
                set(profile.orientation_weights_by_dataset),
                set(ORIENTATION_PROFILE_DATASETS),
            )
            for dataset in ORIENTATION_PROFILE_DATASETS:
                with self.subTest(robot=robot, dataset=dataset):
                    config = internal_config_from_command(
                        RetargetingCommand(
                            task="robot_only",
                            robot=robot,
                            dataset=dataset,
                            orientation_tracking=True,
                        ),
                    )
                    expected = profile.orientation_weights_by_dataset[dataset]
                    if set(expected.values()) == {0.0}:
                        self.assertEqual(config.retargeter.orientation_weights, {})
                    else:
                        self.assertEqual(
                            config.retargeter.orientation_weights,
                            expected,
                        )

    def test_fbx_e2_orientation_profile_tracks_hips_legs_and_forearms(self):
        weights = load_robot_profile(
            robot="e2",
        ).orientation_weights_by_dataset["fbx_mocap"]
        self.assertEqual(weights["Hips"], 20.0)
        self.assertEqual(weights["LeftUpLeg"], 1.0)
        self.assertEqual(weights["RightUpLeg"], 1.0)
        self.assertEqual(weights["LeftForeArm"], 0.3)
        self.assertEqual(weights["RightForeArm"], 0.3)
        self.assertEqual(weights["LeftArm"], 0.0)
        self.assertEqual(weights["RightArm"], 0.0)

    def test_robot_profile_accepts_orientation_link_alias_overrides(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            profile_path = Path(temporary_dir) / "e1_23dof.json"
            profile = json.loads(
                (ROBOT_PROFILE_DIR / "e1_23dof.json").read_text(
                    encoding="utf-8",
                ),
            )
            overrides = profile["orientation"]["datasets"]["noetix_mocap"]["overrides"]
            overrides["LeftArm"] = 0.0
            overrides["r_arm_shoulder_yaw_link"] = 0.25
            profile_path.write_text(json.dumps(profile), encoding="utf-8")
            config = internal_config_from_command(
                RetargetingCommand(
                    task="robot_only",
                    robot="e1_23dof",
                    dataset="noetix_mocap",
                    robot_profile=profile_path,
                    orientation_tracking=True,
                ),
            )

        self.assertEqual(config.retargeter.orientation_weights["LeftArm"], 0.0)
        self.assertEqual(config.retargeter.orientation_weights["RightArm"], 0.25)

    def test_robot_profile_rejects_robot_mismatch(self):
        with self.assertRaisesRegex(ValueError, "does not match command robot"):
            internal_config_from_command(
                RetargetingCommand(
                    task="robot_only",
                    robot="e2",
                    dataset="lafan",
                    robot_profile=ROBOT_PROFILE_DIR / "g1.json",
                ),
            )

    def test_climbing_rejects_enabled_orientation_tracking(self):
        with self.assertRaisesRegex(ValueError, "does not provide direct"):
            internal_config_from_command(
                RetargetingCommand(
                    task="climbing",
                    robot="g1",
                    dataset="climbing",
                    orientation_tracking=True,
                ),
            )

    def test_robot_profile_rejects_invalid_orientation_tables(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            profile = json.loads(
                (ROBOT_PROFILE_DIR / "g1.json").read_text(encoding="utf-8"),
            )
            profile["orientation"]["datasets"]["noetix_mocap"]["overrides"]["unknown_link"] = 1.0
            profile_path = Path(temporary_dir) / "g1.json"
            profile_path.write_text(json.dumps(profile), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "neither a mapped"):
                internal_config_from_command(
                    RetargetingCommand(
                        task="robot_only",
                        robot="g1",
                        dataset="noetix_mocap",
                        robot_profile=profile_path,
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
                "output_name",
                "overwrite",
                "robot_profile",
                "robot_dof",
                "robot_height",
                "robot_urdf_file",
                "foot_sticking",
                "dynamic_ground_window",
                "interaction_mesh_weight",
                "arm_interaction_mesh_weight_scale",
                "root_position_weight",
                "root_orientation_weight",
                "q_a_init_idx",
                "activate_joint_limits",
                "activate_obj_non_penetration",
                "penetration_tolerance",
                "foot_sticking_tolerance",
                "step_size",
                "w_nominal_tracking_init",
                "nominal_tracking_tau",
                "retargeter",
                "orientation_weights",
                "orientation_tracking",
                "orientation_preview",
                "shoulder_direction_tracking",
                "shoulder_direction_weight",
                "shoulder_wrist_axis_weight_scale",
                "natural_pose_tracking",
                "nature_weights",
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
            "amass",
            "climbing",
            "fbx_mocap",
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
