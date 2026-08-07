#!/usr/bin/env python3
# ruff: noqa: CPY001
"""Public Viser entry point for one saved paired refinement result."""

from __future__ import annotations

import tyro

from holosoma_retargeting.paired_retargeting.visualization import (
    PairedViserConfig,
    run_paired_result_player,
)


def main(config: PairedViserConfig) -> None:
    """Run the paired Viser player until interrupted."""
    run_paired_result_player(config)


if __name__ == "__main__":
    main(tyro.cli(PairedViserConfig))
