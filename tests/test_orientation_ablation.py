# ruff: noqa: PT009

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "holosoma_retargeting"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from holosoma_retargeting.examples.run_orientation_ablation import (  # noqa: E402
    ORIENTATION_JOINTS,
    Config,
    ablation_run_specs,
    main,
    orientation_weights_for_variant,
)
from holosoma_retargeting.examples.search_orientation_weights import (  # noqa: E402
    balanced_score,
    expand_candidate_weights,
    pareto_front,
)


class OrientationAblationTests(unittest.TestCase):
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

    def test_dry_run_creates_isolated_manifests_and_sliced_input(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "data"
            output_root = root / "outputs"
            data_dir.mkdir()
            task_name = "breaking+hippop.bvh_Skeleton1"
            positions = np.zeros((10, 22, 3), dtype=np.float32)
            quaternions = np.zeros((10, 22, 4), dtype=np.float32)
            quaternions[..., 0] = 1.0
            np.savez_compressed(
                data_dir / f"{task_name}.npz",
                global_joint_positions=positions,
                global_joint_quaternions_wxyz=quaternions,
                joint_names=np.asarray([f"joint_{idx}" for idx in range(22)]),
                height=np.float32(1.7),
                fps=np.float32(30.0),
            )

            main(
                Config(
                    data_path=data_dir,
                    task_name=task_name,
                    output_root=output_root,
                    variants=("baseline", "full"),
                    frame_start=2,
                    frame_count=3,
                    dry_run=True,
                )
            )

            with np.load(
                output_root / "_input" / f"{task_name}.npz",
                allow_pickle=False,
            ) as subset:
                self.assertEqual(
                    subset["global_joint_positions"].shape,
                    (3, 22, 3),
                )
                self.assertEqual(
                    subset["global_joint_quaternions_wxyz"].shape,
                    (3, 22, 4),
                )

            for variant in ("baseline", "full"):
                manifest_path = output_root / variant / "manifest.json"
                self.assertTrue(manifest_path.is_file())
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                self.assertEqual(manifest["status"], "planned")
                self.assertEqual(manifest["variant"], variant)
                self.assertEqual(len(manifest["orientation_weights"]), 15)
            summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(set(summary["runs"]), {"baseline", "full"})
            self.assertEqual(summary["comparisons_to_baseline"], {})
            self.assertEqual(summary["failures"], [])
