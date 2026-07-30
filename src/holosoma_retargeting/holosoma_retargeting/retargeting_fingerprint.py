# ruff: noqa: CPY001

"""Deterministic solver and static-asset fingerprints for safe result resume."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Mapping

PACKAGE_ROOT = Path(__file__).resolve().parent
IMPLEMENTATION_PATHS = (
    PACKAGE_ROOT / "config_types",
    PACKAGE_ROOT / "data_utils" / "motion_data.py",
    PACKAGE_ROOT / "data_utils" / "object_assets.py",
    PACKAGE_ROOT / "data_utils" / "omomo.py",
    PACKAGE_ROOT / "result_artifact.py",
    PACKAGE_ROOT / "retargeting_fingerprint.py",
    PACKAGE_ROOT / "retargeting_pipeline.py",
    PACKAGE_ROOT / "src",
)
RUNTIME_DISTRIBUTIONS = (
    "clarabel",
    "cvxpy",
    "mujoco",
    "numpy",
    "osqp",
    "scipy",
    "smplx",
    "torch",
    "trimesh",
    "yourdfpy",
)
NUMERICAL_RUNTIME_ENVIRONMENT = (
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
)


@dataclass(frozen=True)
class TreeFingerprint:
    """Content identity and bounded inventory for one file or directory tree."""

    sha256: str
    file_count: int
    byte_count: int


_TREE_FINGERPRINT_CACHE: dict[
    str,
    tuple[
        tuple[tuple[str, int, int, int, int, int], ...],
        TreeFingerprint,
    ],
] = {}
_TREE_FINGERPRINT_CACHE_LOCK = Lock()


def _update_length_prefixed(
    digest: hashlib._Hash,
    value: bytes,
) -> None:
    digest.update(len(value).to_bytes(8, byteorder="big", signed=False))
    digest.update(value)


def _iter_tree_files(path: Path) -> tuple[tuple[str, Path], ...]:
    if path.is_symlink():
        raise ValueError(f"Fingerprint inputs must not be symlinks: {path}")
    if path.is_file():
        return ((path.name, path),)
    if not path.is_dir():
        raise FileNotFoundError(f"Fingerprint input does not exist: {path}")
    files: list[tuple[str, Path]] = []
    for child in path.rglob("*"):
        if child.is_symlink():
            raise ValueError(f"Fingerprint inputs must not contain symlinks: {child}")
        if child.is_file():
            files.append((child.relative_to(path).as_posix(), child))
    return tuple(sorted(files, key=lambda item: item[0]))


def fingerprint_tree(path_text: str) -> TreeFingerprint:
    """Hash a tree, reusing bytes only while its inventory is unchanged."""

    path = Path(path_text).expanduser().resolve(strict=True)
    files = _iter_tree_files(path)
    signature = tuple(
        (
            relative_path,
            file_stat.st_dev,
            file_stat.st_ino,
            file_stat.st_size,
            file_stat.st_mtime_ns,
            file_stat.st_ctime_ns,
        )
        for relative_path, file_path in files
        for file_stat in (file_path.stat(),)
    )
    with _TREE_FINGERPRINT_CACHE_LOCK:
        cached = _TREE_FINGERPRINT_CACHE.get(str(path))
    if cached is not None and cached[0] == signature:
        return cached[1]

    digest = hashlib.sha256()
    file_count = 0
    byte_count = 0
    for relative_path, file_path in files:
        _update_length_prefixed(digest, relative_path.encode("utf-8"))
        size = file_path.stat().st_size
        digest.update(size.to_bytes(8, byteorder="big", signed=False))
        with file_path.open("rb") as file:
            while chunk := file.read(1024 * 1024):
                digest.update(chunk)
        file_count += 1
        byte_count += size
    if file_count == 0:
        raise ValueError(f"Fingerprint input tree contains no files: {path}")
    fingerprint = TreeFingerprint(
        sha256=digest.hexdigest(),
        file_count=file_count,
        byte_count=byte_count,
    )
    final_signature = tuple(
        (
            relative_path,
            file_stat.st_dev,
            file_stat.st_ino,
            file_stat.st_size,
            file_stat.st_mtime_ns,
            file_stat.st_ctime_ns,
        )
        for relative_path, file_path in files
        for file_stat in (file_path.stat(),)
    )
    if final_signature != signature:
        return fingerprint_tree(str(path))
    with _TREE_FINGERPRINT_CACHE_LOCK:
        _TREE_FINGERPRINT_CACHE[str(path)] = (
            final_signature,
            fingerprint,
        )
    return fingerprint


def _combined_fingerprint(
    paths: Mapping[str, Path],
) -> dict[str, object]:
    digest = hashlib.sha256()
    entries: dict[str, dict[str, object]] = {}
    for label, raw_path in sorted(paths.items()):
        path = Path(raw_path).expanduser().resolve(strict=True)
        tree = fingerprint_tree(str(path))
        entry = {
            "path": str(path),
            "sha256": tree.sha256,
            "file_count": tree.file_count,
            "byte_count": tree.byte_count,
        }
        entries[label] = entry
        _update_length_prefixed(digest, label.encode("utf-8"))
        _update_length_prefixed(digest, tree.sha256.encode("ascii"))
    return {
        "sha256": digest.hexdigest(),
        "entries": entries,
    }


@lru_cache(maxsize=None)
def _distribution_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _implementation_source_paths() -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for path in IMPLEMENTATION_PATHS:
        candidates = tuple(sorted(path.rglob("*.py"))) if path.is_dir() else (path,)
        paths.update({candidate.relative_to(PACKAGE_ROOT).as_posix(): candidate for candidate in candidates})
    return paths


def solver_implementation_fingerprint() -> dict[str, object]:
    """Return the current retargeting implementation and runtime identity."""

    implementation = _combined_fingerprint(_implementation_source_paths())
    runtime_versions = {distribution: _distribution_version(distribution) for distribution in RUNTIME_DISTRIBUTIONS}
    runtime_versions.update(
        {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform_machine": platform.machine(),
            "platform_system": platform.system(),
        }
    )
    runtime_environment = {
        variable: os.environ.get(variable, "")
        for variable in NUMERICAL_RUNTIME_ENVIRONMENT
    }
    runtime_digest = hashlib.sha256()
    for distribution, version in sorted(runtime_versions.items()):
        _update_length_prefixed(runtime_digest, distribution.encode("utf-8"))
        _update_length_prefixed(runtime_digest, version.encode("utf-8"))
    for variable, value in sorted(runtime_environment.items()):
        _update_length_prefixed(runtime_digest, variable.encode("utf-8"))
        _update_length_prefixed(runtime_digest, value.encode("utf-8"))
    return {
        "implementation_sha256": implementation["sha256"],
        "runtime_sha256": runtime_digest.hexdigest(),
        "runtime_environment": runtime_environment,
        "runtime_versions": runtime_versions,
    }


def solver_dependency_fingerprint(
    paths: Mapping[str, Path],
) -> dict[str, object]:
    """Return the exact static robot, object, and preprocessing inputs."""

    if not paths:
        raise ValueError("At least one static solver dependency is required")
    return _combined_fingerprint(paths)
