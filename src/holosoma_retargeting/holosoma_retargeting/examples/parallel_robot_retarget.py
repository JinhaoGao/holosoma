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

from holosoma_retargeting.config_types.retargeting import RetargetingConfig
from holosoma_retargeting.retargeting_pipeline import (
    RetargetFamilyResult,
    main as run_retargeting_family,
)

DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parents[1] / "demo_results_parallel"


def main(cfg: RetargetingConfig) -> RetargetFamilyResult:
    """Run identity plus configured augmentations for one selected motion."""

    normalized = deepcopy(cfg)
    normalized.augmentation = True
    if normalized.save_dir is None:
        normalized.save_dir = DEFAULT_RESULTS_ROOT
    return run_retargeting_family(normalized)


if __name__ == "__main__":
    main(tyro.cli(RetargetingConfig))
