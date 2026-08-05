#!/usr/bin/env python3
# ruff: noqa: CPY001

"""Retarget one explicitly selected motion without augmentation."""

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
    normalize_retargeting_config,
    planned_variants,
    resolve_task_object_name,
    run_retargeting_job,
    saved_result_has_qpos,
    setup_object_data,
    validate_config,
)
from holosoma_retargeting.retargeting_pipeline import (
    main as run_retargeting_family,
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
    "run_retargeting_job",
    "saved_result_has_qpos",
    "setup_object_data",
    "validate_config",
]

DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parents[1] / "demo_results"


def run_config(cfg: RetargetingConfig) -> RetargetFamilyResult:
    """Run an internal solver config through the single-motion interface."""

    normalized = deepcopy(cfg)
    normalized.augmentation = False
    if normalized.save_dir is None:
        normalized.save_dir = DEFAULT_RESULTS_ROOT
    return run_retargeting_family(normalized)


def main(command: RetargetingCommand) -> RetargetFamilyResult:
    """Resolve and retarget exactly one user-selected motion."""

    return run_config(internal_config_from_command(command))


if __name__ == "__main__":
    main(tyro.cli(RetargetingCommand))
