# ruff: noqa: CPY001

"""Transactionally archive the explicitly known legacy demo-result layouts.

The command is a dry run unless ``--execute`` is supplied.  Its command-line
scope is intentionally fixed to this package's result directories; tests can
call :func:`plan_organization` with an isolated package root.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIRECTORY_NAME = "demo_results"
ALLOWED_RESULTS_ENTRIES = frozenset(
    {
        "v1",
        "runs",
        "archive",
        ".generated-assets",
        ".locks",
        ".staging",
    }
)
LEGACY_RESULTS_ENTRIES = frozenset({"g1", "e1", "h1", "t1"})
LEGACY_RESULTS_ENTRY_PATTERNS = (
    re.compile(r"\.bench_concurrency_\d{8}"),
    re.compile(r"\.smoke_\d{8}"),
)
LEGACY_SIBLING_DIRECTORIES = (
    "demo_results_parallel",
    "demo_results_ablation",
    "demo_results_orientation",
    "demo_results_foot_ablation",
    "demo_results_validation",
)
ARCHIVE_TIMESTAMP_PATTERN = re.compile(r"\d{8}T\d{12}Z")
MANIFEST_FILE_NAME = "manifest.json"
MANIFEST_SCHEMA_VERSION = 3
MANIFEST_TEMP_PREFIX = f".{MANIFEST_FILE_NAME}."
MANIFEST_TEMP_SUFFIX = ".tmp"
PROMOTED_REPORT_FILE_NAME = "report.json"
PROMOTION_JOURNAL_FILE_NAME = "promotion.json"
PROMOTION_JOURNAL_SCHEMA_VERSION = 1
PUBLISHED_TREE_HASH_PREFIX = b"holosoma-demo-results-tree-v1\0"
REQUIRED_ROBOTS = frozenset({"g1", "e1"})
REQUIRED_DATASETS = frozenset(
    {
        "omomo_robot_only",
        "omomo_object_interaction",
        "amass",
        "gvhmr",
        "lafan",
        "noetix_bvh",
        "generic_climb",
        "noetix_csv_climb",
    }
)
E1_DEFAULT_OMISSION_TASK_TYPES = frozenset(
    {
        "object_interaction",
        "climbing",
    }
)
FORMAL_REPORT_QUALITY_LIMITS = {
    "fallback_frame_fraction_per_artifact": 0.10,
    "release_union_frame_fraction_per_artifact": 0.10,
    "full_sequence_retry_fraction_per_artifact": 1.0,
    "fallback_artifact_fraction_per_group": 0.10,
    "foot_release_artifact_fraction_per_group": 0.10,
    "object_release_artifact_fraction_per_group": 0.10,
    "release_union_artifact_fraction_per_group": 0.10,
    "full_sequence_retry_artifact_fraction_per_group": 0.02,
}


class ResultsOrganizationError(RuntimeError):
    """Base error for result-organization failures."""


class UnsafeResultsLayoutError(ResultsOrganizationError):
    """Raised when a path cannot be handled without broadening the scope."""


class ResultsLayoutChangedError(ResultsOrganizationError):
    """Raised when the result tree changed after it was planned."""


class ResultsOrganizationTransactionError(ResultsOrganizationError):
    """Raised after an execution error and its attempted rollback."""

    def __init__(
        self,
        message: str,
        *,
        rollback_errors: Sequence[str] = (),
    ) -> None:
        super().__init__(message)
        self.rollback_errors = tuple(rollback_errors)


@dataclass(frozen=True)
class TreeSnapshot:
    """Read-only inventory of one directory tree."""

    file_count: int
    byte_count: int
    directory_count: int
    tree_digest: str

    def to_dict(self) -> dict[str, int | str]:
        """Return a JSON-compatible snapshot."""

        return {
            "file_count": self.file_count,
            "byte_count": self.byte_count,
            "directory_count": self.directory_count,
            "tree_digest": self.tree_digest,
        }


@dataclass(frozen=True)
class PublishedTreeFingerprint:
    """Content identity shared with the rebuild promotion validator."""

    sha256: str
    file_count: int
    byte_count: int

    def to_dict(self) -> dict[str, int | str]:
        """Return the exact report-compatible fingerprint fields."""

        return {
            "tree_sha256": self.sha256,
            "tree_file_count": self.file_count,
            "tree_byte_count": self.byte_count,
        }


@dataclass(frozen=True)
class PromotedV1Proof:
    """A promoted rebuild report bound to the currently published v1 bytes."""

    run_id: str
    report_path: Path
    report_sha256: str
    journal_path: Path
    journal_sha256: str
    promoted_root: Path
    fingerprint: PublishedTreeFingerprint

    def to_dict(self) -> dict[str, object]:
        """Return a durable proof that can be revalidated during recovery."""

        return {
            "run_id": self.run_id,
            "report_path": str(self.report_path),
            "report_sha256": self.report_sha256,
            "journal_path": str(self.journal_path),
            "journal_sha256": self.journal_sha256,
            "promoted_root": str(self.promoted_root),
            **self.fingerprint.to_dict(),
        }


@dataclass(frozen=True)
class ResultMove:
    """One exact source-to-archive rename."""

    source: Path
    target: Path
    source_kind: str
    snapshot: TreeSnapshot

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible move description."""

        return {
            "source": str(self.source),
            "target": str(self.target),
            "source_kind": self.source_kind,
            **self.snapshot.to_dict(),
        }


@dataclass(frozen=True)
class OrganizationPlan:
    """Complete immutable plan for one archive transaction."""

    package_root: Path
    results_root: Path
    archive_root: Path
    manifest_path: Path
    timestamp_utc: str
    planned_at_utc: str
    moves: tuple[ResultMove, ...]
    promotion_proof: PromotedV1Proof | None = None
    resuming: bool = False

    @property
    def file_count(self) -> int:
        """Return the total number of regular files to archive."""

        return sum(move.snapshot.file_count for move in self.moves)

    @property
    def byte_count(self) -> int:
        """Return the total regular-file bytes to archive."""

        return sum(move.snapshot.byte_count for move in self.moves)

    def to_dict(self, *, mode: str = "dry_run") -> dict[str, object]:
        """Return a JSON-compatible plan."""

        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "mode": mode,
            "status": "planned",
            "timestamp_utc": self.timestamp_utc,
            "planned_at_utc": self.planned_at_utc,
            "package_root": str(self.package_root),
            "results_root": str(self.results_root),
            "archive_root": str(self.archive_root),
            "manifest_path": str(self.manifest_path),
            "move_count": len(self.moves),
            "file_count": self.file_count,
            "byte_count": self.byte_count,
            "resuming": self.resuming,
            "promotion_proof": (self.promotion_proof.to_dict() if self.promotion_proof is not None else None),
            "moves": [move.to_dict() for move in self.moves],
        }


def _lexists(path: Path) -> bool:
    """Return whether a directory entry exists, including a broken symlink."""

    return os.path.lexists(path)


def _assert_real_directory(path: Path, *, label: str) -> None:
    """Require an existing non-symlink directory."""

    if not _lexists(path):
        raise UnsafeResultsLayoutError(f"{label} does not exist: {path}")
    path_stat = path.lstat()
    if stat.S_ISLNK(path_stat.st_mode):
        raise UnsafeResultsLayoutError(f"{label} must not be a symlink: {path}")
    if not stat.S_ISDIR(path_stat.st_mode):
        raise UnsafeResultsLayoutError(f"{label} must be a directory: {path}")


def _assert_real_directory_if_present(path: Path, *, label: str) -> None:
    """Validate an optional directory without following a symlink."""

    if _lexists(path):
        _assert_real_directory(path, label=label)


def _is_known_legacy_results_entry(name: str) -> bool:
    """Return whether a main-root entry is an explicitly known old layout."""

    return name in LEGACY_RESULTS_ENTRIES or any(pattern.fullmatch(name) for pattern in LEGACY_RESULTS_ENTRY_PATTERNS)


def _format_timestamp_utc(moment: datetime) -> str:
    """Format an aware moment as a collision-resistant UTC archive name."""

    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    return moment.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _normalize_archive_timestamp(timestamp_utc: str | None, *, now: datetime) -> str:
    """Validate an optional archive timestamp or create one from ``now``."""

    normalized = timestamp_utc or _format_timestamp_utc(now)
    if ARCHIVE_TIMESTAMP_PATTERN.fullmatch(normalized) is None:
        raise ValueError("timestamp_utc must use the UTC form YYYYMMDDTHHMMSSffffffZ")
    try:
        datetime.strptime(normalized, "%Y%m%dT%H%M%S%fZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise ValueError(f"timestamp_utc is not a valid UTC timestamp: {normalized}") from exc
    return normalized


def _snapshot_tree(root: Path) -> TreeSnapshot:
    """Inventory a real directory tree without following symlinks."""

    _assert_real_directory(root, label="archive source")
    digest = hashlib.sha256()
    file_count = 0
    byte_count = 0
    directory_count = 0
    stack: list[tuple[Path, str]] = [(root, ".")]

    while stack:
        directory, relative_directory = stack.pop()
        directory_stat = directory.lstat()
        if not stat.S_ISDIR(directory_stat.st_mode) or stat.S_ISLNK(directory_stat.st_mode):
            raise UnsafeResultsLayoutError(f"archive source directory changed type: {directory}")
        directory_count += 1
        digest.update(
            (
                f"D\0{relative_directory}\0{directory_stat.st_dev}\0"
                f"{directory_stat.st_ino}\0{stat.S_IMODE(directory_stat.st_mode)}\0"
                f"{directory_stat.st_mtime_ns}\n"
            ).encode()
        )

        with os.scandir(directory) as iterator:
            entries = sorted(iterator, key=lambda entry: entry.name)
        child_directories: list[tuple[Path, str]] = []
        for entry in entries:
            relative_path = entry.name if relative_directory == "." else f"{relative_directory}/{entry.name}"
            entry_stat = entry.stat(follow_symlinks=False)
            if stat.S_ISLNK(entry_stat.st_mode):
                raise UnsafeResultsLayoutError(f"symlinks are not allowed in archive sources: {entry.path}")
            if stat.S_ISDIR(entry_stat.st_mode):
                child_directories.append((Path(entry.path), relative_path))
                continue
            if not stat.S_ISREG(entry_stat.st_mode):
                raise UnsafeResultsLayoutError(f"unsupported filesystem entry in archive source: {entry.path}")
            file_count += 1
            byte_count += entry_stat.st_size
            digest.update(
                (
                    f"F\0{relative_path}\0{entry_stat.st_dev}\0{entry_stat.st_ino}\0"
                    f"{stat.S_IMODE(entry_stat.st_mode)}\0{entry_stat.st_size}\0"
                    f"{entry_stat.st_mtime_ns}\n"
                ).encode()
            )
            with Path(entry.path).open("rb") as file:
                while chunk := file.read(1024 * 1024):
                    digest.update(chunk)
            final_stat = entry.stat(follow_symlinks=False)
            if (
                final_stat.st_dev,
                final_stat.st_ino,
                final_stat.st_mode,
                final_stat.st_size,
                final_stat.st_mtime_ns,
                final_stat.st_ctime_ns,
            ) != (
                entry_stat.st_dev,
                entry_stat.st_ino,
                entry_stat.st_mode,
                entry_stat.st_size,
                entry_stat.st_mtime_ns,
                entry_stat.st_ctime_ns,
            ):
                raise ResultsLayoutChangedError(f"archive source changed while hashing: {entry.path}")
        stack.extend(reversed(child_directories))

    return TreeSnapshot(
        file_count=file_count,
        byte_count=byte_count,
        directory_count=directory_count,
        tree_digest=digest.hexdigest(),
    )


def _published_tree_inventory(
    root: Path,
) -> tuple[tuple[str, int, int, int, int], ...]:
    """Inventory regular files exactly as the rebuild validator does."""

    _assert_real_directory(root, label="published v1")
    entries: list[tuple[str, int, int, int, int]] = []
    for path in root.rglob("*"):
        path_stat = path.lstat()
        if stat.S_ISLNK(path_stat.st_mode):
            raise UnsafeResultsLayoutError(f"published v1 must not contain symlinks: {path}")
        if stat.S_ISDIR(path_stat.st_mode):
            continue
        if not stat.S_ISREG(path_stat.st_mode):
            raise UnsafeResultsLayoutError(f"published v1 contains an unsupported entry: {path}")
        entries.append(
            (
                path.relative_to(root).as_posix(),
                path_stat.st_size,
                path_stat.st_mtime_ns,
                path_stat.st_ctime_ns,
                path_stat.st_ino,
            )
        )
    return tuple(sorted(entries))


def _update_length_prefixed(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, byteorder="big", signed=False))
    digest.update(value)


def _fingerprint_published_tree(root: Path) -> PublishedTreeFingerprint:
    """Bind the current formal tree to a promoted rebuild report."""

    root = Path(root)
    before = _published_tree_inventory(root)
    digest = hashlib.sha256()
    digest.update(PUBLISHED_TREE_HASH_PREFIX)
    byte_count = 0
    for relative_path, size, _mtime_ns, _ctime_ns, _inode in before:
        path = root / relative_path
        _update_length_prefixed(digest, relative_path.encode("utf-8"))
        digest.update(size.to_bytes(8, byteorder="big", signed=False))
        with path.open("rb") as file:
            while chunk := file.read(1024 * 1024):
                digest.update(chunk)
        byte_count += size
    after = _published_tree_inventory(root)
    if after != before:
        raise ResultsLayoutChangedError(f"published v1 changed while fingerprinting: {root}")
    return PublishedTreeFingerprint(
        sha256=digest.hexdigest(),
        file_count=len(before),
        byte_count=byte_count,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not _lexists(path):
        raise UnsafeResultsLayoutError(f"{label} does not exist: {path}")
    path_stat = path.lstat()
    if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISREG(path_stat.st_mode):
        raise UnsafeResultsLayoutError(f"{label} must be a regular non-symlink file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UnsafeResultsLayoutError(f"{label} is not readable valid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise UnsafeResultsLayoutError(f"{label} must contain a JSON object: {path}")
    return payload


def _required_mapping(
    payload: Mapping[str, Any],
    key: str,
    *,
    label: str,
) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise UnsafeResultsLayoutError(f"{label}.{key} must be an object")
    return value


def _required_non_negative_int(
    payload: Mapping[str, Any],
    key: str,
    *,
    label: str,
) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise UnsafeResultsLayoutError(f"{label}.{key} must be a non-negative integer")
    return value


def _required_bool(
    payload: Mapping[str, Any],
    key: str,
    *,
    label: str,
) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise UnsafeResultsLayoutError(f"{label}.{key} must be boolean")
    return value


def _required_string_set(
    payload: Mapping[str, Any],
    key: str,
    *,
    label: str,
) -> frozenset[str]:
    value = payload.get(key)
    if (
        not isinstance(value, list)
        or not all(isinstance(item, str) and item for item in value)
        or len(set(value)) != len(value)
    ):
        raise UnsafeResultsLayoutError(
            f"{label}.{key} must be a duplicate-free string list",
        )
    return frozenset(value)


def _validate_formal_quality_policy(report: Mapping[str, Any]) -> None:
    quality_limits = _required_mapping(
        report,
        "quality_limits",
        label="report",
    )
    for key, formal_maximum in FORMAL_REPORT_QUALITY_LIMITS.items():
        value = quality_limits.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= float(value) <= formal_maximum:
            raise UnsafeResultsLayoutError(
                f"report.quality_limits.{key} is wider than the formal maximum {formal_maximum}",
            )


def _validate_execution_audit(
    report: Mapping[str, Any],
    validation: Mapping[str, Any],
    *,
    artifact_count: int,
    expected_artifact_count: int,
) -> int:
    """Validate omission-aware G1/E1 completeness and return omitted artifacts."""

    statistics = _required_mapping(
        validation,
        "statistics",
        label="report.validation",
    )
    audited_omissions = _required_mapping(
        statistics,
        "audited_omissions",
        label="report.validation.statistics",
    )
    omitted_artifact_count = _required_non_negative_int(
        audited_omissions,
        "artifact_count",
        label="report.validation.statistics.audited_omissions",
    )
    omitted_artifact_paths = audited_omissions.get("artifact_paths")
    if (
        not isinstance(omitted_artifact_paths, list)
        or not all(isinstance(path, str) and Path(path).is_absolute() for path in omitted_artifact_paths)
        or len(set(omitted_artifact_paths)) != len(omitted_artifact_paths)
        or len(omitted_artifact_paths) != omitted_artifact_count
    ):
        raise UnsafeResultsLayoutError(
            "report.validation audited omission paths are incomplete or duplicated",
        )
    if artifact_count + omitted_artifact_count != expected_artifact_count:
        raise UnsafeResultsLayoutError(
            "promoted rebuild report does not account for every planned artifact "
            "as either present or an audited E1 omission",
        )

    audit = _required_mapping(report, "execution_audit", label="report")
    if audit.get("ok") is not True or audit.get("issues") != []:
        raise UnsafeResultsLayoutError(
            "promoted rebuild execution audit contains blocking issues",
        )
    policy = _required_mapping(audit, "policy", label="report.execution_audit")
    if (
        policy.get("g1_requires_complete_success") is not True
        or policy.get("existing_artifacts_are_always_validated") is not True
    ):
        raise UnsafeResultsLayoutError(
            "promoted rebuild execution policy does not preserve strict G1/artifact validation",
        )
    allow_e1_best_effort = _required_bool(
        policy,
        "allow_e1_best_effort_omissions",
        label="report.execution_audit.policy",
    )
    allow_e1_robot_only = _required_bool(
        policy,
        "allow_e1_robot_only_omissions",
        label="report.execution_audit.policy",
    )
    if allow_e1_robot_only and not allow_e1_best_effort:
        raise UnsafeResultsLayoutError(
            "E1 robot-only omissions cannot be enabled without E1 best-effort omissions",
        )
    default_e1_task_types = _required_string_set(
        policy,
        "default_e1_task_types",
        label="report.execution_audit.policy",
    )
    if default_e1_task_types != E1_DEFAULT_OMISSION_TASK_TYPES:
        raise UnsafeResultsLayoutError(
            "promoted rebuild execution policy has an unexpected E1 omission task set",
        )

    aggregate = _required_mapping(
        audit,
        "source_job_counts",
        label="report.execution_audit",
    )
    aggregate_counts = {
        key: _required_non_negative_int(
            aggregate,
            key,
            label="report.execution_audit.source_job_counts",
        )
        for key in (
            "planned_source_count",
            "attempted_source_count",
            "succeeded_source_count",
            "omitted_source_count",
            "failed_source_count",
            "unverified_source_count",
            "planned_artifact_count",
            "present_artifact_count",
            "omitted_artifact_count",
            "missing_required_artifact_count",
            "validated_artifact_count",
        )
    }
    if (
        aggregate_counts["planned_artifact_count"] != expected_artifact_count
        or aggregate_counts["present_artifact_count"] != artifact_count
        or aggregate_counts["validated_artifact_count"] != artifact_count
        or aggregate_counts["omitted_artifact_count"] != omitted_artifact_count
        or aggregate_counts["missing_required_artifact_count"] != 0
        or aggregate_counts["failed_source_count"] != 0
        or aggregate_counts["unverified_source_count"] != 0
        or aggregate_counts["attempted_source_count"] != aggregate_counts["planned_source_count"]
        or aggregate_counts["succeeded_source_count"] + aggregate_counts["omitted_source_count"]
        != aggregate_counts["planned_source_count"]
    ):
        raise UnsafeResultsLayoutError(
            "promoted rebuild aggregate source/artifact counts are inconsistent",
        )

    by_robot = _required_mapping(
        audit,
        "by_robot",
        label="report.execution_audit",
    )
    if set(by_robot) != REQUIRED_ROBOTS:
        raise UnsafeResultsLayoutError(
            "promoted rebuild execution audit must contain exactly G1 and E1",
        )
    robot_counts: dict[str, dict[str, int]] = {}
    for robot in sorted(REQUIRED_ROBOTS):
        robot_payload = _required_mapping(
            by_robot,
            robot,
            label="report.execution_audit.by_robot",
        )
        robot_counts[robot] = {
            key: _required_non_negative_int(
                robot_payload,
                key,
                label=f"report.execution_audit.by_robot.{robot}",
            )
            for key in aggregate_counts
            if key != "validated_artifact_count"
        }
    g1_counts = robot_counts["g1"]
    if (
        g1_counts["planned_source_count"] == 0
        or g1_counts["attempted_source_count"] != g1_counts["planned_source_count"]
        or g1_counts["succeeded_source_count"] != g1_counts["planned_source_count"]
        or g1_counts["omitted_source_count"] != 0
        or g1_counts["failed_source_count"] != 0
        or g1_counts["unverified_source_count"] != 0
        or g1_counts["planned_artifact_count"] == 0
        or g1_counts["present_artifact_count"] != g1_counts["planned_artifact_count"]
        or g1_counts["omitted_artifact_count"] != 0
        or g1_counts["missing_required_artifact_count"] != 0
    ):
        raise UnsafeResultsLayoutError(
            "promoted rebuild does not prove complete strict G1 coverage",
        )
    e1_counts = robot_counts["e1"]
    if (
        e1_counts["planned_source_count"] == 0
        or e1_counts["attempted_source_count"] != e1_counts["planned_source_count"]
        or e1_counts["succeeded_source_count"] + e1_counts["omitted_source_count"] != e1_counts["planned_source_count"]
        or e1_counts["failed_source_count"] != 0
        or e1_counts["unverified_source_count"] != 0
        or e1_counts["planned_artifact_count"] == 0
        or e1_counts["present_artifact_count"] + e1_counts["omitted_artifact_count"]
        != e1_counts["planned_artifact_count"]
        or e1_counts["missing_required_artifact_count"] != 0
    ):
        raise UnsafeResultsLayoutError(
            "promoted rebuild E1 coverage/omission counts are inconsistent",
        )
    for key in robot_counts["g1"]:
        if robot_counts["g1"][key] + robot_counts["e1"][key] != aggregate_counts[key]:
            raise UnsafeResultsLayoutError(
                f"promoted rebuild per-robot {key} does not sum to its aggregate",
            )

    omissions = audit.get("omissions")
    if not isinstance(omissions, list):
        raise UnsafeResultsLayoutError(
            "report.execution_audit.omissions must be a list",
        )
    observed_omitted_paths: set[str] = set()
    for omission in omissions:
        if not isinstance(omission, Mapping):
            raise UnsafeResultsLayoutError(
                "report.execution_audit omission entries must be objects",
            )
        if omission.get("robot") != "e1":
            raise UnsafeResultsLayoutError("only E1 artifacts may be omitted")
        task_type = omission.get("task_type")
        if task_type not in E1_DEFAULT_OMISSION_TASK_TYPES and not (task_type == "robot_only" and allow_e1_robot_only):
            raise UnsafeResultsLayoutError(
                "E1 omission is outside the promoted policy",
            )
        if omission.get("failure_type") != "SQPNonlinearFeasibilityError":
            raise UnsafeResultsLayoutError(
                "E1 omission is not backed by an allowlisted structural solver failure",
            )
        error = omission.get("error")
        if not isinstance(error, str) or not error.startswith("SQPNonlinearFeasibilityError:"):
            raise UnsafeResultsLayoutError(
                "E1 omission has invalid structural failure evidence",
            )
        report_sha256 = omission.get("batch_report_sha256")
        if not isinstance(report_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", report_sha256) is None:
            raise UnsafeResultsLayoutError(
                "E1 omission has an invalid batch report digest",
            )
        paths = omission.get("omitted_artifact_paths")
        if (
            not isinstance(paths, list)
            or not paths
            or not all(isinstance(path, str) and Path(path).is_absolute() for path in paths)
        ):
            raise UnsafeResultsLayoutError(
                "E1 omission does not identify absolute omitted artifact paths",
            )
        for path in paths:
            if path in observed_omitted_paths:
                raise UnsafeResultsLayoutError(
                    "E1 omission evidence duplicates an artifact path",
                )
            observed_omitted_paths.add(path)
    if (
        len(omissions) != aggregate_counts["omitted_source_count"]
        or observed_omitted_paths != set(omitted_artifact_paths)
        or (omitted_artifact_count > 0 and not allow_e1_best_effort)
    ):
        raise UnsafeResultsLayoutError(
            "promoted rebuild omission evidence does not match validation counts",
        )
    omission_set_sha256 = audit.get("omission_set_sha256")
    if not isinstance(omission_set_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", omission_set_sha256) is None:
        raise UnsafeResultsLayoutError(
            "promoted rebuild omission-set digest is invalid",
        )
    return omitted_artifact_count


def _validate_promoted_v1(
    results_root: Path,
    run_id: str,
) -> PromotedV1Proof:
    """Require a successful full promotion report matching formal v1 bytes."""

    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id) is None:
        raise UnsafeResultsLayoutError(
            "promoted_run_id must start with an alphanumeric character and "
            "contain only letters, digits, '.', '_', or '-'"
        )
    formal = results_root / "v1"
    _assert_real_directory(formal, label="formal promoted v1")
    run_root = results_root / "runs" / run_id
    report_path = run_root / PROMOTED_REPORT_FILE_NAME
    journal_path = run_root / PROMOTION_JOURNAL_FILE_NAME
    report = _read_json_object(report_path, label="promoted rebuild report")
    journal = _read_json_object(
        journal_path,
        label="completed promotion journal",
    )
    if (
        journal.get("schema_version") != PROMOTION_JOURNAL_SCHEMA_VERSION
        or journal.get("status") != "completed"
        or journal.get("run_id") != run_id
        or journal.get("results_root") != str(results_root)
        or journal.get("formal_root") != str(formal)
        or journal.get("final_report") != report
    ):
        raise UnsafeResultsLayoutError(
            f"promotion journal is not a completed transaction matching its final report: {journal_path}",
        )
    if report.get("run_id") != run_id or report.get("status") != "promoted":
        raise UnsafeResultsLayoutError(
            f"rebuild report is not a successful promotion for run_id={run_id!r}: {report_path}"
        )
    if report.get("batch_failures") != []:
        raise UnsafeResultsLayoutError(f"promoted rebuild report contains batch failures: {report_path}")
    planning = _required_mapping(report, "planning", label="report")
    if planning.get("ok") is not True or planning.get("output_collision_count") != 0:
        raise UnsafeResultsLayoutError(f"promoted rebuild report has an invalid artifact plan: {report_path}")
    if (
        _required_string_set(report, "robots", label="report") != REQUIRED_ROBOTS
        or _required_string_set(report, "datasets", label="report") != REQUIRED_DATASETS
    ):
        raise UnsafeResultsLayoutError(
            f"promoted rebuild report does not select the complete G1/E1 dataset matrix: {report_path}",
        )
    _validate_formal_quality_policy(report)
    release_policy = _required_mapping(
        report,
        "release_object_non_penetration_on_infeasible_by_robot",
        label="report",
    )
    if (
        set(release_policy) != REQUIRED_ROBOTS
        or not isinstance(release_policy.get("g1"), bool)
        or release_policy.get("e1") is not False
    ):
        raise UnsafeResultsLayoutError(
            "promoted rebuild report has an invalid per-robot object-release policy",
        )
    validation = _required_mapping(report, "validation", label="report")
    if validation.get("ok") is not True:
        raise UnsafeResultsLayoutError(f"promoted rebuild report has no successful validation: {report_path}")
    artifact_count = _required_non_negative_int(
        validation,
        "artifact_count",
        label="report.validation",
    )
    expected_artifact_count = _required_non_negative_int(
        validation,
        "expected_artifact_count",
        label="report.validation",
    )
    if (
        _required_non_negative_int(
            planning,
            "expected_artifact_count",
            label="report.planning",
        )
        != expected_artifact_count
        or _required_non_negative_int(
            planning,
            "expected_source_count",
            label="report.planning",
        )
        == 0
        or _required_non_negative_int(
            planning,
            "expected_source_job_count",
            label="report.planning",
        )
        == 0
    ):
        raise UnsafeResultsLayoutError(
            "promoted rebuild planning counts do not match validation",
        )
    unexpected_artifact_count = _required_non_negative_int(
        validation,
        "unexpected_artifact_count",
        label="report.validation",
    )
    issue_count = _required_non_negative_int(
        validation,
        "issue_count",
        label="report.validation",
    )
    if artifact_count == 0 or unexpected_artifact_count != 0 or issue_count != 0:
        raise UnsafeResultsLayoutError(f"promoted rebuild report does not prove a complete exact matrix: {report_path}")
    _validate_execution_audit(
        report,
        validation,
        artifact_count=artifact_count,
        expected_artifact_count=expected_artifact_count,
    )
    promotion = _required_mapping(report, "promotion", label="report")
    promoted_root_value = promotion.get("promoted_root")
    if not isinstance(promoted_root_value, str):
        raise UnsafeResultsLayoutError(f"report.promotion.promoted_root must be a path string: {report_path}")
    promoted_root = Path(promoted_root_value).expanduser().resolve()
    if promoted_root != formal.resolve(strict=True):
        raise UnsafeResultsLayoutError(
            f"promoted rebuild report points at a different formal tree: {promoted_root} != {formal}"
        )
    expected_fingerprint = PublishedTreeFingerprint(
        sha256=str(validation.get("tree_sha256", "")),
        file_count=_required_non_negative_int(
            validation,
            "tree_file_count",
            label="report.validation",
        ),
        byte_count=_required_non_negative_int(
            validation,
            "tree_byte_count",
            label="report.validation",
        ),
    )
    if re.fullmatch(r"[0-9a-f]{64}", expected_fingerprint.sha256) is None:
        raise UnsafeResultsLayoutError(f"report.validation.tree_sha256 is invalid: {report_path}")
    actual_fingerprint = _fingerprint_published_tree(formal)
    if actual_fingerprint != expected_fingerprint:
        raise UnsafeResultsLayoutError(
            "formal v1 no longer matches its promoted rebuild report: "
            f"expected {expected_fingerprint}, found {actual_fingerprint}"
        )
    journal_new_tree = _required_mapping(
        journal,
        "new_tree",
        label="promotion journal",
    )
    journal_fingerprint = PublishedTreeFingerprint(
        sha256=str(journal_new_tree.get("sha256", "")),
        file_count=_required_non_negative_int(
            journal_new_tree,
            "file_count",
            label="promotion journal.new_tree",
        ),
        byte_count=_required_non_negative_int(
            journal_new_tree,
            "byte_count",
            label="promotion journal.new_tree",
        ),
    )
    if journal_fingerprint != actual_fingerprint:
        raise UnsafeResultsLayoutError(
            "completed promotion journal new_tree does not match formal v1",
        )
    return PromotedV1Proof(
        run_id=run_id,
        report_path=report_path,
        report_sha256=_sha256_file(report_path),
        journal_path=journal_path,
        journal_sha256=_sha256_file(journal_path),
        promoted_root=promoted_root,
        fingerprint=actual_fingerprint,
    )


def _discover_move_sources(package_root: Path, results_root: Path) -> tuple[tuple[Path, str], ...]:
    """Resolve only the fixed legacy names and reject every unknown main entry."""

    sources: list[tuple[Path, str]] = []
    unknown_entries: list[str] = []
    with os.scandir(results_root) as iterator:
        main_entries = sorted(iterator, key=lambda entry: entry.name)
    for entry in main_entries:
        entry_path = Path(entry.path)
        entry_stat = entry.stat(follow_symlinks=False)
        if stat.S_ISLNK(entry_stat.st_mode):
            raise UnsafeResultsLayoutError(f"symlinks are not allowed at the results root: {entry_path}")
        if entry.name in ALLOWED_RESULTS_ENTRIES:
            if not stat.S_ISDIR(entry_stat.st_mode):
                raise UnsafeResultsLayoutError(f"reserved results entry must be a directory: {entry_path}")
            continue
        if _is_known_legacy_results_entry(entry.name):
            if not stat.S_ISDIR(entry_stat.st_mode):
                raise UnsafeResultsLayoutError(f"legacy results entry must be a directory: {entry_path}")
            sources.append((entry_path, "legacy_main_layout"))
            continue
        unknown_entries.append(entry.name)

    if unknown_entries:
        names = ", ".join(unknown_entries)
        raise UnsafeResultsLayoutError(
            f"unknown demo_results top-level entries would violate the final layout: {names}"
        )

    for sibling_name in LEGACY_SIBLING_DIRECTORIES:
        sibling_path = package_root / sibling_name
        if not _lexists(sibling_path):
            continue
        _assert_real_directory(sibling_path, label=f"legacy sibling {sibling_name}")
        sources.append((sibling_path, "legacy_sibling_tree"))

    return tuple(sources)


def _validate_distinct_paths(paths: Iterable[Path], *, label: str) -> None:
    """Reject duplicate canonical paths."""

    values = tuple(paths)
    if len(set(values)) != len(values):
        raise UnsafeResultsLayoutError(f"{label} contains duplicate paths")


def _is_manifest_temporary_path(path: Path) -> bool:
    return path.name.startswith(MANIFEST_TEMP_PREFIX) and path.name.endswith(MANIFEST_TEMP_SUFFIX)


def _orphaned_manifest_temporary_paths(
    archive_root: Path,
) -> tuple[Path, ...]:
    temporary_paths = tuple(
        sorted(
            (path for path in archive_root.iterdir() if _is_manifest_temporary_path(path)),
            key=lambda path: path.name,
        )
    )
    for path in temporary_paths:
        path_stat = path.lstat()
        if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISREG(path_stat.st_mode):
            raise UnsafeResultsLayoutError(f"orphaned manifest temporary entry must be a regular file: {path}")
    return temporary_paths


def _proof_from_payload(payload: Mapping[str, Any]) -> PromotedV1Proof:
    label = "transaction promotion_proof"
    run_id = payload.get("run_id")
    report_path = payload.get("report_path")
    report_sha256 = payload.get("report_sha256")
    journal_path = payload.get("journal_path")
    journal_sha256 = payload.get("journal_sha256")
    promoted_root = payload.get("promoted_root")
    tree_sha256 = payload.get("tree_sha256")
    if not all(
        isinstance(value, str)
        for value in (
            run_id,
            report_path,
            report_sha256,
            journal_path,
            journal_sha256,
            promoted_root,
            tree_sha256,
        )
    ):
        raise UnsafeResultsLayoutError(f"{label} string fields are incomplete")
    assert isinstance(run_id, str)
    assert isinstance(report_path, str)
    assert isinstance(report_sha256, str)
    assert isinstance(journal_path, str)
    assert isinstance(journal_sha256, str)
    assert isinstance(promoted_root, str)
    assert isinstance(tree_sha256, str)
    if re.fullmatch(r"[0-9a-f]{64}", report_sha256) is None:
        raise UnsafeResultsLayoutError(f"{label}.report_sha256 is invalid")
    if re.fullmatch(r"[0-9a-f]{64}", journal_sha256) is None:
        raise UnsafeResultsLayoutError(f"{label}.journal_sha256 is invalid")
    if re.fullmatch(r"[0-9a-f]{64}", tree_sha256) is None:
        raise UnsafeResultsLayoutError(f"{label}.tree_sha256 is invalid")
    return PromotedV1Proof(
        run_id=run_id,
        report_path=Path(report_path),
        report_sha256=report_sha256,
        journal_path=Path(journal_path),
        journal_sha256=journal_sha256,
        promoted_root=Path(promoted_root),
        fingerprint=PublishedTreeFingerprint(
            sha256=tree_sha256,
            file_count=_required_non_negative_int(
                payload,
                "tree_file_count",
                label=label,
            ),
            byte_count=_required_non_negative_int(
                payload,
                "tree_byte_count",
                label=label,
            ),
        ),
    )


def _snapshot_from_payload(payload: Mapping[str, Any]) -> TreeSnapshot:
    tree_digest = payload.get("tree_digest")
    if (
        not isinstance(tree_digest, str)
        or re.fullmatch(
            r"[0-9a-f]{64}",
            tree_digest,
        )
        is None
    ):
        raise UnsafeResultsLayoutError("transaction move tree_digest is invalid")
    return TreeSnapshot(
        file_count=_required_non_negative_int(
            payload,
            "file_count",
            label="transaction move",
        ),
        byte_count=_required_non_negative_int(
            payload,
            "byte_count",
            label="transaction move",
        ),
        directory_count=_required_non_negative_int(
            payload,
            "directory_count",
            label="transaction move",
        ),
        tree_digest=tree_digest,
    )


def _validate_move_scope(
    *,
    source: Path,
    target: Path,
    source_kind: str,
    package_root: Path,
    results_root: Path,
    archive_root: Path,
) -> None:
    expected_kind: str | None = None
    if source.parent == results_root and _is_known_legacy_results_entry(source.name):
        expected_kind = "legacy_main_layout"
    elif source.parent == package_root and source.name in LEGACY_SIBLING_DIRECTORIES:
        expected_kind = "legacy_sibling_tree"
    if expected_kind is None or source_kind != expected_kind:
        raise UnsafeResultsLayoutError(f"transaction move escaped the fixed legacy scope: {source}")
    if target != archive_root / source.name:
        raise UnsafeResultsLayoutError(f"transaction target is not the fixed archive path: {target}")


def _load_resume_plan(
    *,
    package_root: Path,
    results_root: Path,
    archive_root: Path,
    manifest_path: Path,
    timestamp_utc: str,
    promoted_run_id: str | None,
) -> OrganizationPlan:
    manifest = _read_json_object(
        manifest_path,
        label="organization transaction manifest",
    )
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise UnsafeResultsLayoutError("organization transaction schema is not recoverable by this implementation")
    if manifest.get("status") not in {"in_progress", "completed"}:
        raise UnsafeResultsLayoutError("only in_progress or completed organization transactions can be resumed")
    exact_paths = {
        "package_root": package_root,
        "results_root": results_root,
        "archive_root": archive_root,
        "manifest_path": manifest_path,
    }
    for key, expected in exact_paths.items():
        raw_value = manifest.get(key)
        if not isinstance(raw_value, str) or Path(raw_value) != expected:
            raise UnsafeResultsLayoutError(f"organization transaction {key} does not match the selected path")
    if manifest.get("timestamp_utc") != timestamp_utc:
        raise UnsafeResultsLayoutError("organization transaction timestamp does not match the selected timestamp")
    proof_payload = manifest.get("promotion_proof")
    if not isinstance(proof_payload, Mapping):
        raise UnsafeResultsLayoutError("organization transaction has no durable promoted-v1 proof")
    stored_proof = _proof_from_payload(proof_payload)
    if promoted_run_id is not None and promoted_run_id != stored_proof.run_id:
        raise UnsafeResultsLayoutError("promoted_run_id does not match the fixed transaction proof")
    current_proof = _validate_promoted_v1(
        results_root,
        stored_proof.run_id,
    )
    if current_proof != stored_proof:
        raise ResultsLayoutChangedError("promoted v1 proof changed after the organization transaction started")
    raw_moves = manifest.get("moves")
    if not isinstance(raw_moves, list):
        raise UnsafeResultsLayoutError("organization transaction moves must be a list")
    moves: list[ResultMove] = []
    for raw_move in raw_moves:
        if not isinstance(raw_move, Mapping):
            raise UnsafeResultsLayoutError("organization transaction move must be an object")
        source_value = raw_move.get("source")
        target_value = raw_move.get("target")
        source_kind = raw_move.get("source_kind")
        if not all(isinstance(value, str) for value in (source_value, target_value, source_kind)):
            raise UnsafeResultsLayoutError("organization transaction move paths/kind are invalid")
        assert isinstance(source_value, str)
        assert isinstance(target_value, str)
        assert isinstance(source_kind, str)
        source = Path(source_value)
        target = Path(target_value)
        _validate_move_scope(
            source=source,
            target=target,
            source_kind=source_kind,
            package_root=package_root,
            results_root=results_root,
            archive_root=archive_root,
        )
        moves.append(
            ResultMove(
                source=source,
                target=target,
                source_kind=source_kind,
                snapshot=_snapshot_from_payload(raw_move),
            )
        )
    _validate_distinct_paths(
        (move.source for move in moves),
        label="transaction sources",
    )
    _validate_distinct_paths(
        (move.target for move in moves),
        label="transaction targets",
    )
    allowed_archive_entries = {
        MANIFEST_FILE_NAME,
        *(move.target.name for move in moves),
    }
    actual_archive_entries = tuple(archive_root.iterdir())
    _orphaned_manifest_temporary_paths(archive_root)
    unexpected_archive_entries = sorted(
        path.name
        for path in actual_archive_entries
        if path.name not in allowed_archive_entries and not _is_manifest_temporary_path(path)
    )
    if unexpected_archive_entries:
        raise UnsafeResultsLayoutError(
            f"organization transaction contains unexpected archive entries: {unexpected_archive_entries}"
        )
    discovered_sources = {
        source
        for source, _source_kind in _discover_move_sources(
            package_root,
            results_root,
        )
    }
    planned_sources = {move.source for move in moves}
    if not discovered_sources.issubset(planned_sources):
        raise UnsafeResultsLayoutError("new legacy result sources appeared after the transaction started")
    planned_at_utc = manifest.get("planned_at_utc")
    if not isinstance(planned_at_utc, str):
        raise UnsafeResultsLayoutError("organization transaction planned_at_utc is invalid")
    return OrganizationPlan(
        package_root=package_root,
        results_root=results_root,
        archive_root=archive_root,
        manifest_path=manifest_path,
        timestamp_utc=timestamp_utc,
        planned_at_utc=planned_at_utc,
        moves=tuple(moves),
        promotion_proof=stored_proof,
        resuming=True,
    )


def plan_organization(
    package_root: Path = PACKAGE_ROOT,
    *,
    timestamp_utc: str | None = None,
    now: datetime | None = None,
    promoted_run_id: str | None = None,
    resume: bool = False,
) -> OrganizationPlan:
    """Build a read-only, exact plan for all known legacy result trees."""

    package_root = Path(package_root).absolute()
    _assert_real_directory(package_root, label="package root")
    package_root = package_root.resolve(strict=True)
    results_root = package_root / RESULTS_DIRECTORY_NAME
    _assert_real_directory(results_root, label="demo results root")
    results_root = results_root.resolve(strict=True)
    if results_root.parent != package_root:
        raise UnsafeResultsLayoutError(f"demo results root escaped the package root: {results_root}")

    current_time = now or datetime.now(timezone.utc)
    archive_timestamp = _normalize_archive_timestamp(timestamp_utc, now=current_time)
    archive_parent = results_root / "archive"
    legacy_archive_parent = archive_parent / "legacy"
    archive_root = legacy_archive_parent / archive_timestamp
    manifest_path = archive_root / MANIFEST_FILE_NAME

    _assert_real_directory_if_present(archive_parent, label="archive directory")
    _assert_real_directory_if_present(
        legacy_archive_parent,
        label="legacy archive directory",
    )
    resuming_empty_transaction = False
    if _lexists(archive_root):
        if not resume:
            raise UnsafeResultsLayoutError(
                f"archive transaction already exists: {archive_root}; pass --resume to reconcile it"
            )
        _assert_real_directory(
            archive_root,
            label="archive transaction directory",
        )
        if _lexists(manifest_path):
            return _load_resume_plan(
                package_root=package_root,
                results_root=results_root,
                archive_root=archive_root,
                manifest_path=manifest_path,
                timestamp_utc=archive_timestamp,
                promoted_run_id=promoted_run_id,
            )
        temporary_paths = set(_orphaned_manifest_temporary_paths(archive_root))
        unexpected_entries = sorted(path.name for path in archive_root.iterdir() if path not in temporary_paths)
        if unexpected_entries:
            raise UnsafeResultsLayoutError(
                f"manifest-less archive transaction contains unexpected entries: {unexpected_entries}"
            )
        resuming_empty_transaction = True
    if resume and not resuming_empty_transaction:
        raise UnsafeResultsLayoutError(f"archive transaction does not exist for --resume: {archive_root}")

    discovered_sources = _discover_move_sources(package_root, results_root)
    source_paths = tuple(source for source, _ in discovered_sources)
    _validate_distinct_paths(source_paths, label="archive sources")
    target_names = tuple(source.name for source in source_paths)
    if len(set(target_names)) != len(target_names):
        raise UnsafeResultsLayoutError("legacy result names collide inside the archive transaction")

    moves: list[ResultMove] = []
    for source, source_kind in discovered_sources:
        if source.parent not in {results_root, package_root}:
            raise UnsafeResultsLayoutError(f"archive source is outside the fixed result-tree scope: {source}")
        target = archive_root / source.name
        if target.parent != archive_root:
            raise UnsafeResultsLayoutError(f"archive target escaped the transaction directory: {target}")
        if _lexists(target):
            raise UnsafeResultsLayoutError(f"archive target already exists: {target}")
        moves.append(
            ResultMove(
                source=source,
                target=target,
                source_kind=source_kind,
                snapshot=_snapshot_tree(source),
            )
        )

    promotion_proof = _validate_promoted_v1(results_root, promoted_run_id) if promoted_run_id is not None else None
    return OrganizationPlan(
        package_root=package_root,
        results_root=results_root,
        archive_root=archive_root,
        manifest_path=manifest_path,
        timestamp_utc=archive_timestamp,
        planned_at_utc=current_time.astimezone(timezone.utc).isoformat(),
        moves=tuple(moves),
        promotion_proof=promotion_proof,
        resuming=resuming_empty_transaction,
    )


def _plan_identity(plan: OrganizationPlan) -> tuple[object, ...]:
    """Return the immutable filesystem identity of a plan."""

    return (
        plan.package_root,
        plan.results_root,
        plan.archive_root,
        plan.promotion_proof,
        plan.resuming,
        tuple(
            (
                move.source,
                move.target,
                move.source_kind,
                move.snapshot,
            )
            for move in plan.moves
        ),
    )


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    """Atomically replace a JSON file within its existing parent."""

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        temporary_file = os.fdopen(
            file_descriptor,
            "w",
            encoding="utf-8",
        )
        file_descriptor = -1
        with temporary_file:
            temporary_file.write(f"{json.dumps(payload, indent=2, sort_keys=True)}\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        temporary_path.replace(path)
        _fsync_directory(path.parent)
    finally:
        if file_descriptor >= 0:
            os.close(file_descriptor)
        if _lexists(temporary_path):
            temporary_path.unlink()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _exclusive_results_state_lock(
    results_root: Path,
) -> Iterator[None]:
    lock_directory = results_root / ".locks"
    if _lexists(lock_directory):
        _assert_real_directory(
            lock_directory,
            label="results lock directory",
        )
    else:
        lock_directory.mkdir()
    lock_path = lock_directory / "results-state.lock"
    descriptor = os.open(
        lock_path,
        os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _manifest_payload(
    plan: OrganizationPlan,
    *,
    status: str,
    move_states: dict[Path, str],
    executed_at_utc: str | None = None,
    error: str | None = None,
    rollback_errors: Sequence[str] = (),
) -> dict[str, object]:
    """Build the durable transaction journal and final manifest."""

    payload = plan.to_dict(mode="execute")
    payload["status"] = status
    payload["executed_at_utc"] = executed_at_utc
    payload["error"] = error
    payload["rollback_errors"] = list(rollback_errors)
    payload["moves"] = [
        {
            **move.to_dict(),
            "state": move_states[move.source],
        }
        for move in plan.moves
    ]
    return payload


def _create_archive_root(plan: OrganizationPlan) -> None:
    """Create fixed archive parents one component at a time."""

    archive_parent = plan.results_root / "archive"
    legacy_parent = archive_parent / "legacy"
    for path in (archive_parent, legacy_parent):
        if _lexists(path):
            _assert_real_directory(path, label="archive parent")
        else:
            path.mkdir()
            _assert_real_directory(path, label="created archive parent")
            _fsync_directory(path.parent)
    if _lexists(plan.archive_root):
        raise UnsafeResultsLayoutError(f"archive transaction already exists: {plan.archive_root}")
    plan.archive_root.mkdir()
    _assert_real_directory(plan.archive_root, label="archive transaction directory")
    _fsync_directory(plan.archive_root.parent)


def _rename_path(source: Path, target: Path) -> None:
    """Rename one tree without a copy fallback."""

    source.rename(target)


def _assert_snapshot(path: Path, expected: TreeSnapshot, *, label: str) -> None:
    """Require the current tree to match its planned inventory."""

    actual = _snapshot_tree(path)
    if actual != expected:
        raise ResultsLayoutChangedError(f"{label} changed after planning: {path}; expected {expected}, found {actual}")


def _validate_plan_is_current(plan: OrganizationPlan) -> None:
    """Re-plan immediately before mutation and reject every difference."""

    fresh_plan = plan_organization(
        plan.package_root,
        timestamp_utc=plan.timestamp_utc,
        now=datetime.now(timezone.utc),
        promoted_run_id=(plan.promotion_proof.run_id if plan.promotion_proof is not None else None),
    )
    if _plan_identity(fresh_plan) != _plan_identity(plan):
        raise ResultsLayoutChangedError("demo result layout changed after planning")


def _validate_resume_plan_is_current(plan: OrganizationPlan) -> None:
    fresh_plan = plan_organization(
        plan.package_root,
        timestamp_utc=plan.timestamp_utc,
        promoted_run_id=(plan.promotion_proof.run_id if plan.promotion_proof is not None else None),
        resume=True,
    )
    if _plan_identity(fresh_plan) != _plan_identity(plan):
        raise ResultsLayoutChangedError("organization transaction changed after its resume plan was built")


def _validate_plan_promotion_proof(plan: OrganizationPlan) -> None:
    if plan.promotion_proof is None:
        raise ResultsOrganizationError("execution requires --promoted-run-id naming a successful rebuild promotion")
    current_proof = _validate_promoted_v1(
        plan.results_root,
        plan.promotion_proof.run_id,
    )
    if current_proof != plan.promotion_proof:
        raise ResultsLayoutChangedError("formal v1 or its promoted rebuild report changed after planning")


def _observed_move_location(move: ResultMove) -> str:
    source_exists = _lexists(move.source)
    target_exists = _lexists(move.target)
    if source_exists == target_exists:
        qualifier = "both exist" if source_exists else "both are missing"
        raise UnsafeResultsLayoutError(
            f"organization transaction is ambiguous for {move.source} <-> {move.target}: {qualifier}"
        )
    if source_exists:
        _assert_snapshot(
            move.source,
            move.snapshot,
            label="transaction source",
        )
        return "source"
    _assert_snapshot(
        move.target,
        move.snapshot,
        label="transaction target",
    )
    return "target"


def _remove_orphaned_manifest_temporary_paths(archive_root: Path) -> None:
    temporary_paths = _orphaned_manifest_temporary_paths(archive_root)
    for path in temporary_paths:
        path.unlink()
    if temporary_paths:
        _fsync_directory(archive_root)


def _execute_organization_locked(
    plan: OrganizationPlan,
) -> dict[str, object]:
    """Execute one exact archive plan while the result-state lock is held."""

    _validate_plan_promotion_proof(plan)
    if plan.resuming:
        _validate_resume_plan_is_current(plan)
    else:
        _validate_plan_is_current(plan)
    if not plan.moves:
        payload = plan.to_dict(mode="execute")
        payload["status"] = "no_changes"
        payload["executed_at_utc"] = datetime.now(timezone.utc).isoformat()
        return payload

    if plan.resuming:
        _assert_real_directory(
            plan.archive_root,
            label="archive transaction directory",
        )
    locations = {move.source: _observed_move_location(move) for move in plan.moves}
    if not plan.resuming:
        _create_archive_root(plan)
    else:
        _remove_orphaned_manifest_temporary_paths(plan.archive_root)
    move_states = {move.source: ("moved" if locations[move.source] == "target" else "pending") for move in plan.moves}
    moved = [move for move in plan.moves if locations[move.source] == "target"]
    try:
        _write_json_atomic(
            plan.manifest_path,
            _manifest_payload(plan, status="in_progress", move_states=move_states),
        )
        for move in plan.moves:
            if locations[move.source] == "target":
                continue
            _assert_snapshot(move.source, move.snapshot, label="archive source")
            if _lexists(move.target):
                raise UnsafeResultsLayoutError(f"archive target appeared during execution: {move.target}")
            _rename_path(move.source, move.target)
            _fsync_directory(move.source.parent)
            _fsync_directory(move.target.parent)
            moved.append(move)
            locations[move.source] = "target"
            move_states[move.source] = "moved"
            _assert_snapshot(move.target, move.snapshot, label="archived target")
            _write_json_atomic(
                plan.manifest_path,
                _manifest_payload(plan, status="in_progress", move_states=move_states),
            )
        completed_at = datetime.now(timezone.utc).isoformat()
        payload = _manifest_payload(
            plan,
            status="completed",
            move_states=move_states,
            executed_at_utc=completed_at,
        )
        _write_json_atomic(plan.manifest_path, payload)
    except Exception as exc:
        rollback_errors: list[str] = []
        for move in reversed(moved):
            try:
                if _lexists(move.source):
                    raise UnsafeResultsLayoutError(f"rollback source already exists: {move.source}")
                _assert_snapshot(move.target, move.snapshot, label="rollback target")
                _rename_path(move.target, move.source)
                _fsync_directory(move.target.parent)
                _fsync_directory(move.source.parent)
                move_states[move.source] = "rolled_back"
                locations[move.source] = "source"
                _assert_snapshot(move.source, move.snapshot, label="restored source")
            except Exception as rollback_exc:  # noqa: PERF203
                move_states[move.source] = "rollback_failed"
                rollback_errors.append(f"{move.target} -> {move.source}: {rollback_exc}")
        transaction_status = "rollback_failed" if rollback_errors else "rolled_back"
        try:
            _write_json_atomic(
                plan.manifest_path,
                _manifest_payload(
                    plan,
                    status=transaction_status,
                    move_states=move_states,
                    executed_at_utc=datetime.now(timezone.utc).isoformat(),
                    error=str(exc),
                    rollback_errors=rollback_errors,
                ),
            )
        except Exception as manifest_exc:
            rollback_errors.append(f"could not update transaction manifest: {manifest_exc}")
        raise ResultsOrganizationTransactionError(
            f"result organization failed and transaction status is {transaction_status}: {exc}",
            rollback_errors=rollback_errors,
        ) from exc

    return payload


def execute_organization(
    plan: OrganizationPlan,
    *,
    execute: bool = False,
) -> dict[str, object]:
    """Execute an exact archive plan and roll back every completed move on failure."""

    if not execute:
        raise ResultsOrganizationError("execution requires execute=True (CLI: --execute)")
    with _exclusive_results_state_lock(plan.results_root):
        return _execute_organization_locked(plan)


def _build_argument_parser() -> argparse.ArgumentParser:
    """Create the intentionally narrow command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Archive known legacy demo-result layouts. The default is a read-only "
            "dry run; pass --execute to perform the planned renames."
        )
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="perform the transaction; without this flag only print the exact plan",
    )
    parser.add_argument(
        "--timestamp-utc",
        help="optional archive timestamp in YYYYMMDDTHHMMSSffffffZ form",
    )
    parser.add_argument(
        "--promoted-run-id",
        help=("rebuild run whose successful promoted report and formal-v1 fingerprint authorize execution"),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=("reconcile and finish the fixed timestamp transaction from its durable manifest"),
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    package_root: Path = PACKAGE_ROOT,
) -> int:
    """Plan or explicitly execute organization for the package result trees."""

    args = _build_argument_parser().parse_args(argv)
    try:
        plan = plan_organization(
            package_root,
            timestamp_utc=args.timestamp_utc,
            promoted_run_id=args.promoted_run_id,
            resume=args.resume,
        )
        payload = execute_organization(plan, execute=True) if args.execute else plan.to_dict()
    except (ResultsOrganizationError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
