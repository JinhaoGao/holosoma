#!/usr/bin/env python3
# ruff: noqa: CPY001
"""Jointly refine two synchronized robot-only retargeting results."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import tyro

from holosoma_retargeting.paired_retargeting.config import (
    PairedContactConfig,
    PairedFrameWindow,
    PairedRefinementCommand,
)
from holosoma_retargeting.paired_retargeting.result import (
    PairedRefinementResult,
    load_paired_trajectories,
    save_paired_result,
)
from holosoma_retargeting.paired_retargeting.retargeter import PairedTrajectoryRefiner
from holosoma_retargeting.paired_retargeting.visualization import (
    PairedViserConfig,
    run_paired_result_player,
)


def _contacts_from_json(path: Path) -> tuple[PairedContactConfig, ...]:
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("contacts")
    if not isinstance(payload, list):
        raise ValueError("Contact configuration must be a JSON list or an object with a 'contacts' list")
    contacts: list[PairedContactConfig] = []
    for contact_payload in payload:
        if not isinstance(contact_payload, dict):
            raise ValueError("Each contact configuration must be a JSON object")
        values = dict(contact_payload)
        if "target_offset" in values:
            values["target_offset"] = tuple(values["target_offset"])
        if "windows" in values:
            if not isinstance(values["windows"], list):
                raise ValueError("Contact windows must be a JSON list")
            values["windows"] = tuple(PairedFrameWindow(**window) for window in values["windows"])
        contacts.append(PairedContactConfig(**values))
    return tuple(contacts)


def run_config(command: PairedRefinementCommand) -> PairedRefinementResult:
    """Load two baselines, jointly refine them, and save one paired artifact."""
    actor_a, actor_b = load_paired_trajectories(
        command.actor_a.result_path,
        command.actor_b.result_path,
    )
    refinement_config = command.refinement
    if command.contacts_path is not None:
        refinement_config = replace(
            refinement_config,
            contacts=_contacts_from_json(command.contacts_path),
        )
    result = PairedTrajectoryRefiner(
        command.actor_a.name,
        actor_a,
        command.actor_b.name,
        actor_b,
        refinement_config,
    ).refine()
    save_paired_result(result, command.output_path, overwrite=command.overwrite)
    if command.visualize:
        run_paired_result_player(
            PairedViserConfig(
                qpos_npz=command.output_path,
                loop=command.loop_visualization,
            )
        )
    return result


def main(command: PairedRefinementCommand) -> PairedRefinementResult:
    """Run the paired refinement command."""
    return run_config(command)


if __name__ == "__main__":
    main(tyro.cli(PairedRefinementCommand))
