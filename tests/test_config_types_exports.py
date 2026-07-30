# ruff: noqa: CPY001, F403

"""Public export-contract tests for configuration types."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "holosoma_retargeting"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from holosoma_retargeting.config_types import *  # noqa: E402

from holosoma_retargeting import config_types  # noqa: E402


def test_star_import_exports_every_declared_configuration_type() -> None:
    """Every public name must exist and participate in a star import."""

    assert "EvaluationConfig" not in config_types.__all__
    assert all(hasattr(config_types, name) for name in config_types.__all__)
    assert all(name in globals() for name in config_types.__all__)
