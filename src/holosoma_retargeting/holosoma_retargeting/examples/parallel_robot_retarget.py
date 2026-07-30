#!/usr/bin/env python3
# ruff: noqa: CPY001

"""Retarget one explicitly selected motion and its augmentation variants.

The historical implementation discovered every motion below ``data_dir`` and
scheduled one process per source. That behavior made this command a dataset
production pipeline rather than a usable retargeting interface. The production
contract is deliberately narrower: source selection is identical to
``robot_retarget.py`` and parallelism, when added, may only be used to execute
the variants of that one selected source.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import tyro

from holosoma_retargeting.config_types.retargeting import (
    RetargetingCommand,
    RetargetingConfig,
    internal_config_from_command,
)
from holosoma_retargeting.retargeting_pipeline import (
    RetargetFamilyResult,
)
from holosoma_retargeting.retargeting_pipeline import (
    main as run_retargeting_family,
)

DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parents[1] / "demo_results_parallel"


def run_config(cfg: RetargetingConfig) -> RetargetFamilyResult:
    """Run an internal solver config through the augmentation interface."""

    if cfg.task_type not in {"object_interaction", "climbing"}:
        raise ValueError(
            "The augmentation interface supports only object_interaction and climbing tasks",
        )
    normalized = deepcopy(cfg)
    normalized.augmentation = True
    if normalized.save_dir is None:
        normalized.save_dir = DEFAULT_RESULTS_ROOT
    return run_retargeting_family(normalized)


def main(command: RetargetingCommand) -> RetargetFamilyResult:
    """Resolve one motion and run its identity and augmentation variants."""

    return run_config(internal_config_from_command(command))


if __name__ == "__main__":
    main(tyro.cli(RetargetingCommand))
