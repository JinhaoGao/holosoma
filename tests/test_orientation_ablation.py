# ruff: noqa: CPY001, E402, PT009, PT027

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "holosoma_retargeting"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from holosoma_retargeting.config_types.data_type import (
    DEMO_JOINTS_REGISTRY,
)
from holosoma_retargeting.examples.run_orientation_ablation import (
    EXPERIMENT_NAME as ABLATION_EXPERIMENT_NAME,
)
from holosoma_retargeting.examples.run_orientation_ablation import (
    ORIENTATION_JOINTS,
    ORIENTATION_ROLE_ALIASES,
    Config,
    _build_ablation_job,
    _prepare_input,
    _summary_path,
    ablation_run_specs,
    dataset_partition_identity,
    main,
    orientation_weights_for_variant,
    preparation_payload_sha256,
)
from holosoma_retargeting.examples.search_orientation_weights import (
    EXPERIMENT_NAME as SEARCH_EXPERIMENT_NAME,
)
from holosoma_retargeting.examples.search_orientation_weights import (
    _prepare_window_inputs,
    _run_candidate_window,
    balanced_score,
    expand_candidate_weights,
    pareto_front,
)
from holosoma_retargeting.retargeting_pipeline import RetargetJobResult


class OrientationAblationTests(unittest.TestCase):
    @staticmethod
    def _write_direct_npz(
        path: Path,
        *,
        data_format: str,
        frames: int = 4,
        position_value: float = 0.0,
        with_orientations: bool = True,
    ) -> None:
        joint_names = DEMO_JOINTS_REGISTRY[data_format]
        positions = np.full(
            (frames, len(joint_names), 3),
            position_value,
            dtype=np.float32,
        )
        payload = {
            "global_joint_positions": positions,
            "joint_names": np.asarray(joint_names),
            "height": np.asarray(1.7, dtype=np.float32),
            "fps": np.asarray(30.0, dtype=np.float32),
            "source_format": np.asarray(data_format),
        }
        if with_orientations:
            coordinate_system = "right_handed_z_up" if data_format == "amass" else "z_up"
            quaternions = np.zeros(
                (frames, len(joint_names), 4),
                dtype=np.float32,
            )
            quaternions[..., 0] = 1.0
            payload.update(
                {
                    "orientation_joint_names": np.asarray(joint_names),
                    "orientation_quaternions_wxyz": quaternions,
                    "orientation_source": np.asarray(
                        "bvh_rotation_channels_fk" if data_format == "noetix_mocap" else "direct_local_rotation_fk"
                    ),
                    "quaternion_convention": np.asarray("wxyz"),
                    "coordinate_system": np.asarray(coordinate_system),
                }
            )
            if data_format == "amass":
                payload["orientation_coordinate_system"] = np.asarray(
                    coordinate_system,
                )
        np.savez_compressed(path, **payload)

    def test_search_candidate_expansion_respects_group_and_joint_overrides(self):
        weights = expand_candidate_weights(
            {
                "all": 0.1,
                "ankles": 0.2,
                "LeftFoot": 0.3,
            }
        )

        self.assertEqual(tuple(weights), ORIENTATION_JOINTS)
        self.assertEqual(weights["Hips"], 0.1)
        self.assertEqual(weights["RightFoot"], 0.2)
        self.assertEqual(weights["LeftFoot"], 0.3)

    def test_balanced_score_is_dimensionless_and_penalizes_instability(self):
        baseline = {
            "position_mean_m": 0.05,
            "position_p95_m": 0.10,
            "orientation_mean_rad": 0.50,
            "orientation_p95_rad": 1.00,
            "joint_second_difference_mean_rad": 0.20,
            "sqp_max_iteration_fraction": 0.0,
            "constraint_release_count": 0,
        }
        baseline_score, ratios = balanced_score(baseline, baseline)
        unstable = dict(baseline)
        unstable["joint_second_difference_mean_rad"] = 0.40
        unstable_score, _ = balanced_score(unstable, baseline)

        self.assertAlmostEqual(baseline_score, 1.0)
        self.assertEqual(
            set(ratios), {"position_mean", "position_p95", "orientation_mean", "orientation_p95", "smoothness"}
        )
        self.assertGreater(unstable_score, baseline_score)

    def test_pareto_front_excludes_dominated_candidates(self):
        baseline_metrics = {
            "position_mean_m": 1.0,
            "position_p95_m": 1.0,
            "orientation_mean_rad": 1.0,
            "orientation_p95_rad": 1.0,
            "joint_second_difference_mean_rad": 1.0,
        }
        improved_metrics = dict(baseline_metrics)
        improved_metrics["orientation_mean_rad"] = 0.5
        dominated_metrics = {name: value * 2.0 for name, value in baseline_metrics.items()}

        self.assertEqual(
            pareto_front(
                {
                    "baseline": baseline_metrics,
                    "improved": improved_metrics,
                    "dominated": dominated_metrics,
                }
            ),
            ["improved"],
        )

    def test_profiles_share_diagnostic_links_and_only_enable_selected_groups(self):
        baseline = orientation_weights_for_variant("baseline", 1.0)
        feet = orientation_weights_for_variant("feet", 0.5)
        gmr_legs = orientation_weights_for_variant("gmr_legs", 1.0)
        shoulders = orientation_weights_for_variant("shoulders", 0.025)
        shoulders_feet = orientation_weights_for_variant("shoulders_feet", 10.0)
        full = orientation_weights_for_variant("full", 1.0)
        full_equal = orientation_weights_for_variant("full_equal", 100.0)
        balanced_optimal = orientation_weights_for_variant("balanced_optimal", 1.0)

        self.assertEqual(tuple(baseline), ORIENTATION_JOINTS)
        self.assertTrue(all(weight == 0.0 for weight in baseline.values()))
        self.assertEqual(feet["LeftFoot"], 1.0)
        self.assertEqual(feet["RightFoot"], 1.0)
        self.assertEqual(feet["LeftArm"], 0.0)
        self.assertEqual(gmr_legs["LeftUpLeg"], 1.0)
        self.assertEqual(gmr_legs["RightLeg"], 1.0)
        self.assertEqual(gmr_legs["LeftFoot"], 1.0)
        self.assertEqual(gmr_legs["Hips"], 0.0)
        self.assertEqual(gmr_legs["LeftArm"], 0.0)
        self.assertEqual(shoulders["LeftArm"], 0.025)
        self.assertEqual(shoulders["RightArm"], 0.025)
        self.assertTrue(all(weight == 0.0 for name, weight in shoulders.items() if name not in {"LeftArm", "RightArm"}))
        for name in ("LeftArm", "RightArm", "LeftFoot", "RightFoot"):
            self.assertEqual(shoulders_feet[name], 10.0)
        self.assertTrue(
            all(
                weight == 0.0
                for name, weight in shoulders_feet.items()
                if name not in {"LeftArm", "RightArm", "LeftFoot", "RightFoot"}
            )
        )
        self.assertTrue(all(full[name] > 0.0 for name in ORIENTATION_JOINTS))
        self.assertTrue(all(weight == 100.0 for weight in full_equal.values()))
        self.assertTrue(all(weight == 0.085 for weight in balanced_optimal.values()))
        amass_full = orientation_weights_for_variant(
            "full",
            1.0,
            data_format="amass",
        )
        self.assertEqual(
            tuple(amass_full),
            tuple(ORIENTATION_ROLE_ALIASES["amass"].values()),
        )
        self.assertEqual(amass_full["Pelvis"], 1.0)
        self.assertEqual(amass_full["L_Ankle"], 2.0)
        self.assertEqual(amass_full["R_Wrist"], 1.0)

    def test_baseline_is_not_duplicated_across_weight_scales(self):
        specs = ablation_run_specs(
            ("baseline", "feet"),
            (0.25, 0.5, 1.0),
        )
        self.assertEqual(
            specs,
            (
                ("baseline", "baseline", 1.0),
                ("feet_x0.25", "feet", 0.25),
                ("feet_x0.5", "feet", 0.5),
                ("feet", "feet", 1.0),
            ),
        )

    def test_only_named_baseline_may_have_zero_weights(self):
        self.assertTrue(
            all(
                weight == 0.0
                for weight in orientation_weights_for_variant(
                    "baseline",
                    0.0,
                ).values()
            )
        )
        with self.assertRaisesRegex(
            ValueError,
            "Only the named baseline variant",
        ):
            orientation_weights_for_variant("full", 0.0)

    def test_dry_run_creates_isolated_manifests_and_sliced_input(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "data"
            output_root = root / "outputs"
            data_dir.mkdir()
            task_name = "breaking+hippop.bvh_Skeleton1"
            self._write_direct_npz(
                data_dir / f"{task_name}.npz",
                data_format="noetix_mocap",
                frames=10,
            )

            config = Config(
                data_path=data_dir,
                task_name=task_name,
                output_root=output_root,
                variants=("baseline", "full"),
                frame_start=2,
                frame_count=3,
                dry_run=True,
            )
            main(config)

            subset_paths = list(
                (output_root / "ablations" / ABLATION_EXPERIMENT_NAME / "_inputs").glob(f"*/{task_name}.npz")
            )
            self.assertEqual(len(subset_paths), 1)
            with np.load(subset_paths[0], allow_pickle=False) as subset:
                self.assertEqual(
                    subset["global_joint_positions"].shape,
                    (3, 22, 3),
                )
                self.assertEqual(
                    subset["orientation_quaternions_wxyz"].shape,
                    (3, 22, 4),
                )
                self.assertEqual(
                    str(subset["preparation_data_format"]),
                    "noetix_mocap",
                )
                self.assertEqual(
                    str(subset["preparation_source_sha256"]),
                    hashlib.sha256((data_dir / f"{task_name}.npz").read_bytes()).hexdigest(),
                )
                self.assertEqual(int(subset["preparation_frame_start"]), 2)
                self.assertEqual(int(subset["preparation_frame_end"]), 5)

            for variant in ("baseline", "full"):
                manifest_path = (
                    output_root
                    / "ablations"
                    / ABLATION_EXPERIMENT_NAME
                    / variant
                    / "e1"
                    / "robot_only"
                    / "noetix_mocap"
                    / dataset_partition_identity(data_dir)
                    / "breaking%2Bhippop.bvh_Skeleton1"
                    / "frames_000002_000005"
                    / "manifest.json"
                )
                self.assertTrue(manifest_path.is_file())
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                self.assertEqual(manifest["status"], "planned")
                self.assertEqual(manifest["variant"], variant)
                self.assertEqual(manifest["run_kind"], "ablation")
                self.assertEqual(
                    manifest["experiment_name"],
                    ABLATION_EXPERIMENT_NAME,
                )
                self.assertEqual(
                    Path(manifest["result_path"]).name,
                    "identity.npz",
                )
                self.assertIn("/canonical/", manifest["canonical_baseline_path"])
                self.assertEqual(len(manifest["orientation_weights"]), 15)
            summary = json.loads(_summary_path(config).read_text(encoding="utf-8"))
            self.assertEqual(set(summary["runs"]), {"baseline", "full"})
            self.assertEqual(summary["comparisons_to_baseline"], {})
            self.assertEqual(summary["failures"], [])

    def test_sliced_input_cache_is_bound_to_source_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "data"
            data_dir.mkdir()
            task_name = "motion"
            source_path = data_dir / f"{task_name}.npz"
            self._write_direct_npz(
                source_path,
                data_format="noetix_mocap",
            )
            config = Config(
                data_path=data_dir,
                task_name=task_name,
                output_root=root / "results",
                frame_start=1,
                frame_count=2,
            )

            first_directory, first_path = _prepare_input(config)
            second_directory, second_path = _prepare_input(config)

            self.assertEqual(first_directory, second_directory)
            self.assertEqual(first_path, second_path)

            self._write_direct_npz(
                source_path,
                data_format="noetix_mocap",
                position_value=1.0,
            )
            third_directory, third_path = _prepare_input(config)

            self.assertNotEqual(third_directory, first_directory)
            self.assertNotEqual(third_path, first_path)
            with np.load(third_path, allow_pickle=False) as subset:
                np.testing.assert_array_equal(
                    subset["global_joint_positions"],
                    np.ones((2, 22, 3), dtype=np.float32),
                )

    def test_sliced_input_cache_rejects_self_consistent_metadata_tampering(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "data"
            data_dir.mkdir()
            task_name = "motion"
            self._write_direct_npz(
                data_dir / f"{task_name}.npz",
                data_format="noetix_mocap",
            )
            config = Config(
                data_path=data_dir,
                task_name=task_name,
                output_root=root / "results",
                frame_start=1,
                frame_count=2,
            )
            _, subset_path = _prepare_input(config)
            with np.load(subset_path, allow_pickle=False) as subset:
                tampered = {key: np.asarray(subset[key]) for key in subset.files}
            tampered["fps"] = np.asarray(1.0, dtype=np.float32)
            tampered["height"] = np.asarray(99.0, dtype=np.float32)
            tampered["preparation_payload_sha256"] = np.asarray(preparation_payload_sha256(tampered))
            np.savez_compressed(subset_path, **tampered)

            _, repaired_path = _prepare_input(config)

            self.assertEqual(repaired_path, subset_path)
            with np.load(repaired_path, allow_pickle=False) as repaired:
                self.assertEqual(float(repaired["fps"]), 30.0)
                self.assertAlmostEqual(float(repaired["height"]), 1.7)
                self.assertEqual(
                    str(repaired["preparation_payload_sha256"]),
                    preparation_payload_sha256({key: np.asarray(repaired[key]) for key in repaired.files}),
                )
                self.assertIn(
                    str(data_dir / f"{task_name}.npz"),
                    str(repaired["preparation_lineage_json"]),
                )

    def test_ablation_main_calls_shared_runner_with_explicit_job(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "data"
            output_root = root / "results"
            data_dir.mkdir()
            task_name = "breaking+hippop.bvh_Skeleton1"
            input_path = data_dir / f"{task_name}.npz"
            self._write_direct_npz(
                input_path,
                data_format="noetix_mocap",
                frames=2,
            )
            jobs = []

            def fake_runner(job):
                jobs.append(job)
                return RetargetJobResult(
                    output_path=job.output_path,
                    source_path=job.source_path,
                    sequence_key=job.sequence_key,
                    variant=job.variant.name,
                )

            with patch(
                "holosoma_retargeting.examples.run_orientation_ablation.run_retargeting_job",
                side_effect=fake_runner,
            ), patch(
                "holosoma_retargeting.examples.run_orientation_ablation._result_summary",
                return_value={},
            ):
                main(
                    Config(
                        data_path=data_dir,
                        task_name=task_name,
                        output_root=output_root,
                        variants=("baseline",),
                    )
                )

            self.assertEqual(len(jobs), 1)
            job = jobs[0]
            self.assertEqual(job.run_kind, "ablation")
            self.assertEqual(job.experiment_name, ABLATION_EXPERIMENT_NAME)
            self.assertEqual(job.variant.name, "baseline")
            self.assertEqual(job.source_path, input_path.resolve())
            self.assertEqual(job.output_path.name, "identity.npz")
            self.assertIn("ablations", job.output_path.parts)
            self.assertIn("canonical", job.baseline_path.parts)
            manifest = json.loads((job.output_path.parent / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["result_path"], str(job.output_path))
            self.assertEqual(
                manifest["canonical_baseline_path"],
                str(job.baseline_path),
            )

    def test_generic_dry_run_builds_isolated_amass_and_noetix_jobs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cases = (
                ("amass", "g1", "first_frame"),
                ("amass", "e1", "first_frame"),
                ("noetix_mocap", "e1", "t_pose"),
            )
            config_hashes = set()
            for data_format, robot, alignment_mode in cases:
                with self.subTest(
                    data_format=data_format,
                    robot=robot,
                    alignment_mode=alignment_mode,
                ):
                    case_root = root / f"{data_format}-{robot}"
                    data_dir = case_root / "data"
                    output_root = case_root / "results"
                    data_dir.mkdir(parents=True)
                    task_name = "motion"
                    self._write_direct_npz(
                        data_dir / f"{task_name}.npz",
                        data_format=data_format,
                    )

                    config = Config(
                        robot=robot,
                        task_type="robot_only",
                        data_format=data_format,
                        data_path=data_dir,
                        task_name=task_name,
                        output_root=output_root,
                        orientation_alignment_mode=alignment_mode,
                        variants=("baseline", "full"),
                        dry_run=True,
                    )
                    main(config)

                    expected_paths = {}
                    for variant in ("baseline", "full"):
                        result_path = (
                            output_root
                            / "ablations"
                            / ABLATION_EXPERIMENT_NAME
                            / variant
                            / robot
                            / "robot_only"
                            / data_format
                            / dataset_partition_identity(data_dir)
                            / task_name
                            / "identity.npz"
                        ).resolve()
                        expected_paths[variant] = result_path
                        manifest = json.loads((result_path.parent / "manifest.json").read_text(encoding="utf-8"))
                        self.assertEqual(manifest["status"], "planned")
                        self.assertEqual(manifest["run_kind"], "ablation")
                        self.assertEqual(
                            manifest["experiment_name"],
                            ABLATION_EXPERIMENT_NAME,
                        )
                        self.assertEqual(manifest["robot"], robot)
                        self.assertEqual(
                            manifest["data_format"],
                            data_format,
                        )
                        self.assertEqual(
                            manifest["orientation_alignment_mode"],
                            alignment_mode,
                        )
                        self.assertEqual(
                            Path(manifest["result_path"]),
                            result_path,
                        )
                        normalized = manifest["retargeting_config"]
                        self.assertEqual(normalized["robot"], robot)
                        self.assertEqual(
                            normalized["robot_config"]["robot_type"],
                            robot,
                        )
                        self.assertEqual(
                            normalized["motion_data_config"]["robot_type"],
                            robot,
                        )
                        self.assertEqual(
                            normalized["motion_data_config"]["data_format"],
                            data_format,
                        )
                        config_hashes.add(manifest["config_sha256"])
                        if data_format == "amass":
                            orientation_mapping = normalized["motion_data_config"]["orientation_joints_mapping"]
                            self.assertEqual(
                                set(orientation_mapping),
                                set(ORIENTATION_ROLE_ALIASES["amass"].values()),
                            )
                        else:
                            self.assertIsNone(normalized["motion_data_config"]["orientation_joints_mapping"])
                    self.assertNotEqual(
                        expected_paths["baseline"],
                        expected_paths["full"],
                    )
                    summary = json.loads(_summary_path(config).read_text(encoding="utf-8"))
                    self.assertEqual(summary["robot"], robot)
                    self.assertEqual(
                        summary["data_format"],
                        data_format,
                    )
                    self.assertEqual(
                        summary["orientation_alignment_mode"],
                        alignment_mode,
                    )
                    self.assertEqual(summary["failures"], [])
            self.assertEqual(len(config_hashes), 6)

    def test_summary_paths_are_isolated_by_robot_and_dataset(self):
        g1 = Config(
            robot="g1",
            data_format="amass",
            data_path=Path("/data/source_a"),
            task_name="motion",
            output_root=Path("/tmp/results"),
            orientation_alignment_mode="first_frame",
        )
        e1 = replace(g1, robot="e1")
        other_dataset = replace(
            g1,
            data_path=Path("/data/source_b"),
        )

        self.assertNotEqual(_summary_path(g1), _summary_path(e1))
        self.assertNotEqual(_summary_path(g1), _summary_path(other_dataset))
        self.assertEqual(_summary_path(g1).name, "summary.json")
        self.assertIn("_summaries", _summary_path(g1).parts)

    def test_special_names_and_same_dataset_basename_do_not_collide(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_root = root / "results"
            configs = []
            for parent, task_name in (
                ("source_a", "walk+a"),
                ("source_b", "walk a"),
            ):
                data_dir = root / parent / "motions"
                data_dir.mkdir(parents=True)
                self._write_direct_npz(
                    data_dir / f"{task_name}.npz",
                    data_format="noetix_mocap",
                )
                configs.append(
                    Config(
                        data_path=data_dir,
                        task_name=task_name,
                        output_root=output_root,
                        variants=("baseline",),
                        dry_run=True,
                    )
                )

            jobs = []
            for config in configs:
                input_dir, input_path = _prepare_input(config)
                jobs.append(
                    _build_ablation_job(
                        cfg=config,
                        input_dir=input_dir,
                        input_path=input_path,
                        run_name="baseline",
                        orientation_weights=orientation_weights_for_variant(
                            "baseline",
                            1.0,
                        ),
                    )
                )

            self.assertNotEqual(
                dataset_partition_identity(configs[0].data_path),
                dataset_partition_identity(configs[1].data_path),
            )
            self.assertEqual(jobs[0].sequence_key, "walk+a")
            self.assertEqual(jobs[1].sequence_key, "walk a")
            self.assertNotEqual(jobs[0].output_path, jobs[1].output_path)
            self.assertIn("walk%2Ba", jobs[0].output_path.parts)
            self.assertIn("walk%20a", jobs[1].output_path.parts)
            self.assertNotEqual(
                _summary_path(configs[0]),
                _summary_path(configs[1]),
            )

    def test_fail_fast_finalizes_summary_before_raising(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "data"
            data_dir.mkdir()
            self._write_direct_npz(
                data_dir / "motion.npz",
                data_format="noetix_mocap",
            )
            config = Config(
                data_path=data_dir,
                task_name="motion",
                output_root=root / "results",
                variants=("baseline", "full"),
                fail_fast=True,
            )
            attempted_jobs = []

            def fail(job):
                attempted_jobs.append(job)
                raise ValueError("intentional failure")

            with patch(
                "holosoma_retargeting.examples.run_orientation_ablation.run_retargeting_job",
                side_effect=fail,
            ), self.assertRaisesRegex(
                RuntimeError,
                "Orientation ablations failed",
            ):
                main(config)

            self.assertEqual(len(attempted_jobs), 1)
            manifest = json.loads((attempted_jobs[0].output_path.parent / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
            summary = json.loads(_summary_path(config).read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["failures"], ["baseline"])
            self.assertEqual(summary["not_run"], ["full"])
            self.assertEqual(
                summary["runs"]["baseline"]["status"],
                "failed",
            )
            self.assertEqual(
                summary["runs"]["full"]["status"],
                "not_run",
            )

    def test_scaled_profile_manifest_uses_artifact_variant_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "data"
            data_dir.mkdir()
            self._write_direct_npz(
                data_dir / "motion.npz",
                data_format="noetix_mocap",
            )
            main(
                Config(
                    data_path=data_dir,
                    task_name="motion",
                    output_root=root / "results",
                    variants=("feet",),
                    weight_scales=(0.5,),
                    dry_run=True,
                )
            )
            manifests = list((root / "results").rglob("manifest.json"))
            self.assertEqual(len(manifests), 1)
            manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
            self.assertEqual(manifest["run_name"], "feet_x0.5")
            self.assertEqual(manifest["variant"], "feet_x0.5")
            self.assertEqual(manifest["profile"], "feet")

    def test_amass_t_pose_ablation_requires_explicit_first_frame(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "data"
            data_dir.mkdir()
            task_name = "motion"
            self._write_direct_npz(
                data_dir / f"{task_name}.npz",
                data_format="amass",
            )
            cfg = Config(
                robot="g1",
                data_format="amass",
                data_path=data_dir,
                task_name=task_name,
                output_root=root / "results",
                variants=("full",),
            )
            input_dir, input_path = _prepare_input(cfg)

            with self.assertRaisesRegex(
                ValueError,
                "Pass --orientation-alignment-mode first_frame explicitly",
            ):
                _build_ablation_job(
                    cfg=cfg,
                    input_dir=input_dir,
                    input_path=input_path,
                    run_name="full",
                    orientation_weights=orientation_weights_for_variant(
                        "full",
                        1.0,
                        data_format="amass",
                    ),
                )

    def test_invalid_format_task_combination_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir, self.assertRaisesRegex(
            ValueError,
            "supports task types: robot_only",
        ):
            _prepare_input(
                Config(
                    task_type="object_interaction",
                    data_format="amass",
                    data_path=Path(tmpdir),
                    task_name="motion",
                )
            )

    def test_source_without_direct_orientation_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            task_name = "motion"
            self._write_direct_npz(
                root / f"{task_name}.npz",
                data_format="amass",
                with_orientations=False,
            )

            with self.assertRaisesRegex(
                ValueError,
                "adapter-proven direct source orientations",
            ):
                _prepare_input(
                    Config(
                        robot="g1",
                        data_format="amass",
                        data_path=root,
                        task_name=task_name,
                        orientation_alignment_mode="first_frame",
                    )
                )

    def test_complete_source_uses_registry_extension_before_orientation_check(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            task_name = "lafan_motion"
            np.save(
                root / f"{task_name}.npy",
                np.zeros(
                    (
                        3,
                        len(DEMO_JOINTS_REGISTRY["lafan"]),
                        3,
                    ),
                    dtype=np.float32,
                ),
            )

            with self.assertRaisesRegex(
                ValueError,
                r"lafan source .*lafan_motion\.npy has none",
            ):
                _prepare_input(
                    Config(
                        data_format="lafan",
                        data_path=root,
                        task_name=task_name,
                        orientation_alignment_mode="first_frame",
                    )
                )

    def test_frame_subset_rejects_non_npz_format_before_loading(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            task_name = "subject_box_motion"
            (root / f"{task_name}.pt").write_bytes(b"not-a-tensor")

            with self.assertRaisesRegex(
                ValueError,
                "Frame subsetting is only supported",
            ):
                _prepare_input(
                    Config(
                        task_type="robot_only",
                        data_format="omomo",
                        data_path=root,
                        task_name=task_name,
                        frame_start=1,
                        frame_count=2,
                        orientation_alignment_mode="first_frame",
                    )
                )

    def test_search_worker_calls_shared_runner_and_reports_artifact_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "inputs" / "window_000010"
            input_dir.mkdir(parents=True)
            task_name = "breaking+hippop.bvh_Skeleton1"
            input_path = input_dir / f"{task_name}.npz"
            np.savez_compressed(
                input_path,
                global_joint_positions=np.zeros((2, 22, 3), dtype=np.float32),
            )
            jobs = []

            def fake_runner(job):
                jobs.append(job)
                return RetargetJobResult(
                    output_path=job.output_path,
                    source_path=job.source_path,
                    sequence_key=job.sequence_key,
                    variant=job.variant.name,
                )

            with patch(
                "holosoma_retargeting.examples.search_orientation_weights.run_retargeting_job",
                side_effect=fake_runner,
            ), patch(
                "holosoma_retargeting.examples.search_orientation_weights._result_summary",
                return_value={"frames": 2},
            ):
                record = _run_candidate_window(
                    run_name="candidate_a",
                    orientation_weights=dict.fromkeys(
                        ORIENTATION_JOINTS,
                        0.1,
                    ),
                    frame_start=10,
                    frame_count=2,
                    input_path=input_path,
                    output_root=root / "results",
                    task_name=task_name,
                    source_hash="original-source-hash",
                    git_commit="deadbeef",
                    overwrite=False,
                )

            self.assertEqual(len(jobs), 1)
            job = jobs[0]
            self.assertEqual(job.run_kind, "ablation")
            self.assertEqual(job.experiment_name, SEARCH_EXPERIMENT_NAME)
            self.assertEqual(job.variant.name, "candidate_a")
            self.assertEqual(
                job.sequence_key.split("/")[-2:],
                ["frames_000010_000002", "source_original-source-hash"],
            )
            self.assertEqual(record["result_path"], str(job.output_path))
            manifest = json.loads((job.output_path.parent / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["result_path"], str(job.output_path))
            self.assertEqual(
                manifest["canonical_baseline_path"],
                str(job.baseline_path),
            )

    def test_window_cache_identity_includes_count_and_source_hash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / "motion.npz"
            destination = root / "windows"
            positions = np.zeros((8, 22, 3), dtype=np.float32)
            np.savez_compressed(
                source_path,
                global_joint_positions=positions,
            )

            first = _prepare_window_inputs(
                source_path=source_path,
                destination_root=destination,
                frame_starts=(1,),
                frame_count=2,
                overwrite=False,
            )[1]
            different_count = _prepare_window_inputs(
                source_path=source_path,
                destination_root=destination,
                frame_starts=(1,),
                frame_count=3,
                overwrite=False,
            )[1]
            positions[0, 0, 0] = 1.0
            np.savez_compressed(
                source_path,
                global_joint_positions=positions,
            )
            different_source = _prepare_window_inputs(
                source_path=source_path,
                destination_root=destination,
                frame_starts=(1,),
                frame_count=2,
                overwrite=False,
            )[1]
            with np.load(first, allow_pickle=False) as cached:
                self.assertEqual(int(cached["_window_schema_version"]), 2)
                self.assertEqual(
                    len(str(cached["_window_implementation_sha256"])),
                    64,
                )

        self.assertNotEqual(first.parent, different_count.parent)
        self.assertNotEqual(first.parent.parent, different_source.parent.parent)
        self.assertIn("frames_000001_000002", first.parent.name)
        self.assertIn("frames_000001_000003", different_count.parent.name)
