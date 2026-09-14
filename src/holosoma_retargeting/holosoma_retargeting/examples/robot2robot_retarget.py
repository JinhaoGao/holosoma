#!/usr/bin/env python3
# ruff: noqa: CPY001
"""Retarget one robot CSV through the shared Interaction Mesh solver, saving NPZ and CSV."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

import tyro

from holosoma_retargeting.config_types.retargeting import RetargetingOptions, config_from_robot_profile
from holosoma_retargeting.config_types.robot_profiles import load_robot_profile
from holosoma_retargeting.data_utils.npz_to_csv import convert_npz_to_csv
from holosoma_retargeting.data_utils.robot_motion import RobotKinematics, RobotMotionProfile, load_robot_motion
from holosoma_retargeting.data_utils.robot_proportions import ROBOT_POSITION_SCALING_VERSION, rebuild_robot_positions
from holosoma_retargeting.retargeting_pipeline import RetargetJobResult, build_retarget_job, run_retargeting_job


@dataclass(frozen=True)
class Robot2RobotCommand(RetargetingOptions):
    """Source and target profiles describe robots independently; shared controls tune the solver."""

    source_robot: str = tyro.MISSING
    """Source robot name; custom names use --source-robot-profile."""
    robot: str = tyro.MISSING
    """Target robot name; custom names use --robot-profile."""
    input_path: Path = tyro.MISSING
    """Numeric CSV exported in root xyz, root xyzw, then source URDF joint order."""
    fps: float = tyro.MISSING
    """Source frame rate. CSV contains no timing metadata."""
    position_scaling: Literal["segment", "uniform"] = "segment"
    """Rebuild target segment lengths, or select the legacy uniform height ratio."""
    source_robot_profile: Path | None = None
    """Source robot JSON or profile directory, including its robot_motion section."""

    task: tyro.conf.Suppress[str] = "robot_only"
    dataset: tyro.conf.Suppress[str] = "robot"
    motion: tyro.conf.Suppress[str] = ""
    data_path: tyro.conf.Suppress[Path | None] = None


@dataclass(frozen=True)
class Robot2RobotResult:
    job: RetargetJobResult
    csv_path: Path


def main(command: Robot2RobotCommand) -> Robot2RobotResult:
    """Adapt source FK, run the ordinary retargeting lifecycle, and export its result."""
    source_path = command.input_path.expanduser().resolve()
    if source_path.suffix.lower() != ".csv" or not source_path.is_file():
        raise ValueError(f"input_path must identify one existing CSV: {source_path}")
    source_profile = load_robot_profile(robot=command.source_robot, config_path=command.source_robot_profile)
    target_profile = load_robot_profile(robot=command.robot, config_path=command.robot_profile)
    # Apply the same explicit target overrides as the human entry point before resolving FK.
    target_profile = replace(
        target_profile,
        robot_dof=target_profile.robot_dof if command.robot_dof is None else command.robot_dof,
        robot_height=target_profile.robot_height if command.robot_height is None else command.robot_height,
        robot_urdf_file=(
            target_profile.robot_urdf_file
            if command.robot_urdf_file is None
            else str(command.robot_urdf_file.expanduser().resolve())
        ),
    )
    source = RobotKinematics(RobotMotionProfile.from_profile(source_profile))
    target = RobotMotionProfile.from_profile(target_profile)
    # Validate target model, joint order, and reference pose before beginning an expensive solve.
    target_kinematics = RobotKinematics(target)
    missing_positions = set(target.position_links) - set(source.profile.position_links)
    missing_orientations = set(target.orientation_links) - set(source.profile.orientation_links)
    if missing_positions or missing_orientations:
        raise ValueError(
            "Source robot lacks target semantic landmarks: "
            f"positions={sorted(missing_positions)}, orientations={sorted(missing_orientations)}",
        )
    if source.profile.root_joint != target.root_joint:
        raise ValueError("Source and target profiles must use the same semantic root name")
    motion = load_robot_motion(source_path, source=source, target=target, fps=command.fps)
    motion_config = replace(
        target.motion_config(),
        demo_joints=list(source.profile.position_links),
        joint_parent_indices=source.profile.parents,
        source_toe_names=source.profile.toe_names,
        orientation_t_pose_human_quaternions_wxyz={
            name: tuple(q)
            for name, q in zip(
                source.profile.orientation_links,
                source.reference_orientations,
                strict=True,
            )
        },
    )
    target_profile = replace(target_profile, orientation_weights_by_dataset={"robot": target.orientation_weights})
    resolved_command = replace(command, robot=target_profile.robot, motion=source_path.stem)
    config = config_from_robot_profile(
        resolved_command,
        target_profile,
        dataset=None,
        data_format="robot",
        default_data_path=source_path.parent,
        motion_data_config=motion_config,
        robot_config=target.robot,
    )
    config.source_motion_options = {
        "fps": command.fps,
        "source_profile": asdict(source.profile),
        "position_scaling": command.position_scaling,
        "position_scaling_version": ROBOT_POSITION_SCALING_VERSION
        if command.position_scaling == "segment"
        else "uniform",
        "target_torso_link": target.torso_link,
        "target_torso_anchor_names": target.torso_anchor_names,
    }
    if command.position_scaling == "segment":
        motion = rebuild_robot_positions(
            motion,
            source=source,
            target=target_kinematics,
            contact_config=config.retargeter.planar_foot_contact,
        )
    job = build_retarget_job(
        config,
        source_path=source_path,
        dataset_partition="robot2robot",
        sequence_key=f"{source_profile.robot}/{source_path.stem}",
    )
    csv_path = job.output_path.with_suffix(".csv")
    if csv_path.resolve() == source_path:
        raise ValueError("Output CSV must not overwrite the input motion; select a different save directory or name")
    if csv_path.exists() and not job.output_path.exists() and not command.overwrite:
        raise FileExistsError(f"CSV already exists without its NPZ: {csv_path}; pass --overwrite to replace it")
    result = run_retargeting_job(job, motion=motion)
    if not result.resumed or not csv_path.is_file():
        convert_npz_to_csv(result.output_path, target.robot.ROBOT_URDF_FILE, csv_path)
    print(f"Saved NPZ: {result.output_path}\nSaved CSV: {csv_path}")
    return Robot2RobotResult(job=result, csv_path=csv_path)


if __name__ == "__main__":
    main(tyro.cli(Robot2RobotCommand))
