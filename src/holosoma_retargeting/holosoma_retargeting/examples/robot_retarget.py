#!/usr/bin/env python3
# ruff: noqa: CPY001

"""Compatibility entry point for one identity-first retargeting action family."""

from __future__ import annotations

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
    main,
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


if __name__ == "__main__":
    main(tyro.cli(RetargetingConfig))
