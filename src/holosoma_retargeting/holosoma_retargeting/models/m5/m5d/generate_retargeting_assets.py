# ruff: noqa: CPY001
"""Generate this M5 variant from its own updated mechanical URDF."""

import runpy
from pathlib import Path

if __name__ == "__main__":
    namespace = runpy.run_path(str(Path(__file__).resolve().parents[1] / "generate_retargeting_assets.py"))
    namespace["generate"](Path(__file__).resolve().parent.name)
