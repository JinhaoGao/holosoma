#!/usr/bin/env python3
"""Unified synchronized visualization for one or more motion results."""

# ruff: noqa: CPY001, E402
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, fields
from pathlib import Path

import tyro

package_root = Path(__file__).resolve().parent.parent
if str(package_root) not in sys.path:
    sys.path.insert(0, str(package_root))

from holosoma_retargeting.visualization.multi_scene import (
    MultiResultViserConfig,
    load_comparison_results,
)
from holosoma_retargeting.visualization.multi_scene import (
    make_multi_result_player as _make_multi_result_player,
)
from holosoma_retargeting.visualization.result_loader import (
    discover_variant_paths,
)


@dataclass(frozen=True)
class MultiViserConfig(MultiResultViserConfig):
    """Configuration for the public multi-motion visualization entry point."""

    qpos_npzs: tuple[Path, ...] = ()
    """Explicit result NPZ paths in display order."""

    input_paths: tuple[Path, ...] = ()
    """Canonical result paths. The legacy qpos_npzs option remains an alias."""

    family: Path | None = None
    """Reference result used to discover an augmentation family."""

    variants: tuple[str, ...] = ()
    """Optional variant names loaded with ``family``.

    Empty discovers every existing strict canonical variant.
    """


def resolve_multi_config(config: MultiViserConfig) -> MultiResultViserConfig:
    """Resolve explicit inputs or one augmentation family into the shared config."""

    if config.input_paths and config.qpos_npzs:
        raise ValueError("Pass either --input-paths or --qpos-npzs, not both.")
    paths = config.input_paths or config.qpos_npzs
    if bool(paths) == bool(config.family):
        raise ValueError("Pass either --input-paths (--qpos-npzs) or --family, but not both.")
    qpos_npzs = paths
    labels = config.labels
    if config.family is not None:
        requested_variants = config.variants or None
        discovered = discover_variant_paths(config.family, requested_variants)
        qpos_npzs = tuple(discovered.values())
    base_values = {item.name: getattr(config, item.name) for item in fields(MultiResultViserConfig)}
    base_values["qpos_npzs"] = qpos_npzs
    base_values["labels"] = labels
    return MultiResultViserConfig(**base_values)


def make_multi_result_player(config: MultiViserConfig):
    """Load motions and build their shared-timeline Viser scene."""

    resolved = resolve_multi_config(config)
    labels, results = load_comparison_results(resolved)
    return _make_multi_result_player(resolved, labels, results)


def main(config: MultiViserConfig) -> None:
    """Run the unified multi-motion viewer until interrupted."""

    make_multi_result_player(config)
    if config.dry_run:
        return
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main(tyro.cli(MultiViserConfig))
