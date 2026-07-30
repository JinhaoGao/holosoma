#!/usr/bin/env python3
# ruff: noqa: CPY001

"""Retarget one explicitly selected motion without augmentation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import tyro

from holosoma_retargeting.config_types.retargeting import RetargetingConfig
from holosoma_retargeting.retargeting_pipeline import (
    DEFAULT_DATA_FORMATS,
    IDENTITY_VARIANT,
    RetargetFamilyResult,
    RetargetJob,
    RetargetJobResult,
    RetargetVariant,
    build_retarget_job,
    build_retargeter_kwargs_from_config,
    create_ground_points,
    create_task_constants,
    initialize_robot_pose,
    main as run_retargeting_family,
    normalize_retargeting_config,
    planned_variants,
    resolve_task_object_name,
    result_artifact_matches_job,
    run_retargeting_job,
    setup_object_data,
    validate_config,
)

__all__ = [
    "DEFAULT_DATA_FORMATS",
    "IDENTITY_VARIANT",
    "RetargetFamilyResult",
    "RetargetJob",
    "RetargetJobResult",
    "RetargetVariant",
    "build_retarget_job",
    "build_retargeter_kwargs_from_config",
    "create_ground_points",
    "create_task_constants",
    "initialize_robot_pose",
    "main",
    "normalize_retargeting_config",
    "planned_variants",
    "resolve_task_object_name",
    "result_artifact_matches_job",
    "run_retargeting_job",
    "setup_object_data",
    "validate_config",
]

DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parents[1] / "demo_results"


def main(cfg: RetargetingConfig) -> RetargetFamilyResult:
    """Run the identity variant for exactly one explicitly selected motion."""

    normalized = deepcopy(cfg)
    normalized.augmentation = False
    if normalized.save_dir is None:
        normalized.save_dir = DEFAULT_RESULTS_ROOT
    return run_retargeting_family(normalized)


if __name__ == "__main__":
    main(tyro.cli(RetargetingConfig))
