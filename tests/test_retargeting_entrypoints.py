# ruff: noqa: CPY001, PT009

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "holosoma_retargeting"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from holosoma_retargeting.config_types.retargeting import RetargetingConfig  # noqa: E402
from holosoma_retargeting.examples import PUBLIC_RETARGETING_COMMANDS  # noqa: E402
from holosoma_retargeting.examples import parallel_robot_retarget, robot_retarget  # noqa: E402


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
            result = robot_retarget.main(cfg)

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
            result = parallel_robot_retarget.main(cfg)

        self.assertIs(result, sentinel)
        submitted = runner.call_args.args[0]
        self.assertTrue(submitted.augmentation)
        self.assertEqual(
            submitted.save_dir,
            parallel_robot_retarget.DEFAULT_RESULTS_ROOT,
        )
        self.assertFalse(cfg.augmentation)
        self.assertIsNone(cfg.save_dir)

    def test_explicit_result_root_is_preserved(self):
        custom_root = Path("/tmp/retargeting-results")
        for module in (robot_retarget, parallel_robot_retarget):
            with self.subTest(module=module.__name__):
                cfg = RetargetingConfig(save_dir=custom_root)
                with mock.patch.object(module, "run_retargeting_family") as runner:
                    module.main(cfg)
                self.assertEqual(runner.call_args.args[0].save_dir, custom_root)


if __name__ == "__main__":
    unittest.main()
