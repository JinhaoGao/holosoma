"""Load the optional LAFAN reader only when a BVH conversion is requested."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType


def load_lafan_modules() -> tuple[ModuleType, ModuleType]:
    package = "holosoma_retargeting.data_utils.lafan1"
    try:
        return import_module(f"{package}.extract"), import_module(f"{package}.utils")
    except ModuleNotFoundError as exc:
        if exc.name is None or not (exc.name == package or exc.name.startswith(f"{package}.")):
            raise
        raise ModuleNotFoundError(
            "BVH conversion requires the optional LAFAN reader. "
            "Install its extract.py and utils.py in holosoma_retargeting/data_utils/lafan1/."
        ) from exc
