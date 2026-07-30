# ruff: noqa: CPY001

from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SOURCE_ROOT = REPO_ROOT / "src" / "holosoma_retargeting"
if str(PACKAGE_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SOURCE_ROOT))

import holosoma_retargeting.examples.organize_demo_results as organizer  # noqa: E402
import holosoma_retargeting.examples.rebuild_demo_results as rebuild  # noqa: E402
from holosoma_retargeting.examples.organize_demo_results import (  # noqa: E402
    ALLOWED_RESULTS_ENTRIES,
    ResultsLayoutChangedError,
    ResultsOrganizationError,
    ResultsOrganizationTransactionError,
    UnsafeResultsLayoutError,
    execute_organization,
    plan_organization,
)

FIXED_NOW = datetime(2026, 7, 29, 4, 5, 6, 123456, tzinfo=timezone.utc)
FIXED_TIMESTAMP = "20260729T040506123456Z"
PROMOTED_RUN_ID = "trusted-promotion"


class SimulatedHardCrash(BaseException):
    """Model a SIGKILL-like interruption outside ``except Exception``."""


def test_organizer_formal_contract_matches_rebuild_policy() -> None:
    assert set(rebuild.SUPPORTED_ROBOTS) == organizer.REQUIRED_ROBOTS
    assert organizer.REQUIRED_DATASETS == rebuild.CORE_DATASET_IDS
    assert organizer.E1_DEFAULT_OMISSION_TASK_TYPES == (rebuild.E1_DEFAULT_OMISSION_TASK_TYPES)
    assert {
        "fallback_frame_fraction_per_artifact": rebuild.FORMAL_QUALITY_LIMITS[
            "max_fallback_frame_fraction_per_artifact"
        ],
        "release_union_frame_fraction_per_artifact": rebuild.FORMAL_QUALITY_LIMITS[
            "max_release_union_frame_fraction_per_artifact"
        ],
        "full_sequence_retry_fraction_per_artifact": rebuild.FORMAL_QUALITY_LIMITS[
            "max_full_sequence_retry_fraction_per_artifact"
        ],
        "fallback_artifact_fraction_per_group": rebuild.FORMAL_QUALITY_LIMITS[
            "max_fallback_artifact_fraction_per_group"
        ],
        "foot_release_artifact_fraction_per_group": rebuild.FORMAL_QUALITY_LIMITS[
            "max_foot_release_artifact_fraction_per_group"
        ],
        "object_release_artifact_fraction_per_group": rebuild.FORMAL_QUALITY_LIMITS[
            "max_object_release_artifact_fraction_per_group"
        ],
        "release_union_artifact_fraction_per_group": rebuild.FORMAL_QUALITY_LIMITS[
            "max_release_union_artifact_fraction_per_group"
        ],
        "full_sequence_retry_artifact_fraction_per_group": rebuild.FORMAL_QUALITY_LIMITS[
            "max_full_sequence_retry_artifact_fraction_per_group"
        ],
    } == organizer.FORMAL_REPORT_QUALITY_LIMITS


def _package_tree(tmp_path: Path) -> tuple[Path, Path]:
    package_root = tmp_path / "holosoma_retargeting"
    results_root = package_root / "demo_results"
    results_root.mkdir(parents=True)
    return package_root, results_root


def _write_tree(root: Path, files: dict[str, bytes]) -> None:
    root.mkdir(parents=True)
    for relative_path, content in files.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def _write_promoted_proof(
    results_root: Path,
    *,
    run_id: str = PROMOTED_RUN_ID,
    e1_omitted_artifact_count: int = 0,
    write_journal: bool = True,
) -> str:
    formal = results_root / "v1"
    marker = formal / "canonical" / "artifact.npz"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_bytes(b"trusted-promoted-tree")
    e1_marker = formal / "canonical" / "e1" / "artifact.npz"
    e1_marker.parent.mkdir(parents=True)
    e1_marker.write_bytes(b"trusted-promoted-e1-tree")
    fingerprint = organizer._fingerprint_published_tree(formal)
    omitted_paths = [
        str(
            formal
            / "canonical"
            / "e1"
            / "object_interaction"
            / "omomo"
            / "OMOMO_new"
            / f"omitted-{index}"
            / "identity.npz"
        )
        for index in range(e1_omitted_artifact_count)
    ]
    omissions = [
        {
            "batch_id": "e1-omomo_object_interaction",
            "batch_report_path": str(results_root / "runs" / run_id / "batches" / "e1-omomo_object_interaction.json"),
            "batch_report_sha256": "a" * 64,
            "robot": "e1",
            "dataset": "omomo_object_interaction",
            "task_type": "object_interaction",
            "data_format": "omomo",
            "source_path": str(results_root / f"source-{index}.pt"),
            "task_name": f"omitted-{index}",
            "error": "SQPNonlinearFeasibilityError: E1 structural limit",
            "failure_type": "SQPNonlinearFeasibilityError",
            "omitted_variants": ["identity"],
            "omitted_artifact_paths": [path],
        }
        for index, path in enumerate(omitted_paths)
    ]
    g1_counts = {
        "planned_source_count": 1,
        "attempted_source_count": 1,
        "succeeded_source_count": 1,
        "omitted_source_count": 0,
        "failed_source_count": 0,
        "unverified_source_count": 0,
        "planned_artifact_count": 1,
        "present_artifact_count": 1,
        "omitted_artifact_count": 0,
        "missing_required_artifact_count": 0,
    }
    e1_counts = {
        "planned_source_count": 1 + e1_omitted_artifact_count,
        "attempted_source_count": 1 + e1_omitted_artifact_count,
        "succeeded_source_count": 1,
        "omitted_source_count": e1_omitted_artifact_count,
        "failed_source_count": 0,
        "unverified_source_count": 0,
        "planned_artifact_count": 1 + e1_omitted_artifact_count,
        "present_artifact_count": 1,
        "omitted_artifact_count": e1_omitted_artifact_count,
        "missing_required_artifact_count": 0,
    }
    aggregate_counts = {key: g1_counts[key] + e1_counts[key] for key in g1_counts}
    aggregate_counts["validated_artifact_count"] = 2
    expected_artifact_count = 2 + e1_omitted_artifact_count
    report = {
        "run_id": run_id,
        "status": "promoted",
        "batch_failures": [],
        "robots": ["g1", "e1"],
        "datasets": sorted(organizer.REQUIRED_DATASETS),
        "release_object_non_penetration_on_infeasible_by_robot": {
            "g1": True,
            "e1": False,
        },
        "quality_limits": dict(organizer.FORMAL_REPORT_QUALITY_LIMITS),
        "planning": {
            "ok": True,
            "output_collision_count": 0,
            "expected_artifact_count": expected_artifact_count,
            "expected_source_count": 2 + e1_omitted_artifact_count,
            "expected_source_job_count": 2 + e1_omitted_artifact_count,
        },
        "execution_audit": {
            "ok": True,
            "policy": {
                "allow_e1_best_effort_omissions": True,
                "default_e1_task_types": sorted(organizer.E1_DEFAULT_OMISSION_TASK_TYPES),
                "allow_e1_robot_only_omissions": False,
                "g1_requires_complete_success": True,
                "existing_artifacts_are_always_validated": True,
            },
            "source_job_counts": aggregate_counts,
            "by_robot": {
                "g1": {**g1_counts, "by_task": {}},
                "e1": {**e1_counts, "by_task": {}},
            },
            "batches": [],
            "sources": [],
            "omissions": omissions,
            "omission_set_sha256": "b" * 64,
            "issues": [],
        },
        "validation": {
            "ok": True,
            "artifact_count": 2,
            "expected_artifact_count": expected_artifact_count,
            "unexpected_artifact_count": 0,
            "issue_count": 0,
            "statistics": {
                "valid_artifacts": 2,
                "audited_omissions": {
                    "artifact_count": e1_omitted_artifact_count,
                    "artifact_paths": omitted_paths,
                },
            },
            **fingerprint.to_dict(),
        },
        "promotion": {
            "promoted_root": str(formal.resolve()),
            "archived_root": None,
        },
    }
    report_path = results_root / "runs" / run_id / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        f"{json.dumps(report, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    if write_journal:
        journal = {
            "schema_version": organizer.PROMOTION_JOURNAL_SCHEMA_VERSION,
            "status": "completed",
            "run_id": run_id,
            "results_root": str(results_root),
            "staging_root": str(results_root / ".staging" / run_id / "v1"),
            "formal_root": str(formal),
            "archived_root": None,
            "had_formal": False,
            "new_tree": {
                "sha256": fingerprint.sha256,
                "file_count": fingerprint.file_count,
                "byte_count": fingerprint.byte_count,
            },
            "old_tree": None,
            "external_assets": {
                "sha256": "0" * 64,
                "file_count": 0,
                "byte_count": 0,
            },
            "final_report": report,
        }
        journal_path = results_root / "runs" / run_id / "promotion.json"
        journal_path.write_text(
            f"{json.dumps(journal, indent=2, sort_keys=True)}\n",
            encoding="utf-8",
        )
    return run_id


def _rewrite_report_and_sync_journal(
    results_root: Path,
    run_id: str,
    report: dict[str, object],
) -> None:
    report_path = results_root / "runs" / run_id / "report.json"
    report_path.write_text(
        f"{json.dumps(report, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    journal_path = results_root / "runs" / run_id / "promotion.json"
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    journal["final_report"] = report
    journal_path.write_text(
        f"{json.dumps(journal, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )


def test_default_plan_is_read_only_and_execution_requires_explicit_flag(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    package_root, results_root = _package_tree(tmp_path)
    _write_tree(results_root / "g1", {"motion.npz": b"1234"})

    plan = plan_organization(package_root, now=FIXED_NOW)

    assert plan.timestamp_utc == FIXED_TIMESTAMP
    assert plan.file_count == 1
    assert plan.byte_count == 4
    assert (results_root / "g1" / "motion.npz").read_bytes() == b"1234"
    assert not (results_root / "archive").exists()
    with pytest.raises(ResultsOrganizationError, match="execute=True"):
        execute_organization(plan)

    exit_code = organizer.main(
        ["--timestamp-utc", FIXED_TIMESTAMP],
        package_root=package_root,
    )
    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert output["mode"] == "dry_run"
    assert output["status"] == "planned"
    assert (results_root / "g1").is_dir()
    assert not (results_root / "archive").exists()


def test_execute_requires_a_promoted_v1_proof_before_any_write(
    tmp_path: Path,
) -> None:
    package_root, results_root = _package_tree(tmp_path)
    _write_tree(results_root / "g1", {"result.npz": b"legacy"})
    plan = plan_organization(
        package_root,
        timestamp_utc=FIXED_TIMESTAMP,
        now=FIXED_NOW,
    )

    with pytest.raises(
        ResultsOrganizationError,
        match="promoted-run-id",
    ):
        execute_organization(plan, execute=True)

    assert (results_root / "g1" / "result.npz").read_bytes() == b"legacy"
    assert not (results_root / "archive").exists()


def test_execute_rechecks_promoted_v1_fingerprint_before_any_write(
    tmp_path: Path,
) -> None:
    package_root, results_root = _package_tree(tmp_path)
    _write_tree(results_root / "g1", {"result.npz": b"legacy"})
    promoted_run_id = _write_promoted_proof(results_root)
    plan = plan_organization(
        package_root,
        timestamp_utc=FIXED_TIMESTAMP,
        now=FIXED_NOW,
        promoted_run_id=promoted_run_id,
    )
    (results_root / "v1" / "canonical" / "artifact.npz").write_bytes(
        b"tampered-promoted-tree",
    )

    with pytest.raises(
        UnsafeResultsLayoutError,
        match="no longer matches",
    ):
        execute_organization(plan, execute=True)

    assert (results_root / "g1" / "result.npz").read_bytes() == b"legacy"
    assert not (results_root / "archive").exists()


def test_promoted_proof_accepts_audited_e1_structural_omission(
    tmp_path: Path,
) -> None:
    package_root, results_root = _package_tree(tmp_path)
    _write_tree(results_root / "g1", {"result.npz": b"legacy"})
    promoted_run_id = _write_promoted_proof(
        results_root,
        e1_omitted_artifact_count=1,
    )

    plan = plan_organization(
        package_root,
        timestamp_utc=FIXED_TIMESTAMP,
        promoted_run_id=promoted_run_id,
    )

    assert plan.promotion_proof is not None
    assert plan.promotion_proof.journal_path == (results_root / "runs" / promoted_run_id / "promotion.json")
    assert len(plan.promotion_proof.journal_sha256) == 64
    assert len(plan.moves) == 1


def test_promoted_proof_requires_completed_promotion_journal(
    tmp_path: Path,
) -> None:
    package_root, results_root = _package_tree(tmp_path)
    _write_tree(results_root / "g1", {"result.npz": b"legacy"})
    promoted_run_id = _write_promoted_proof(
        results_root,
        write_journal=False,
    )

    with pytest.raises(
        UnsafeResultsLayoutError,
        match="completed promotion journal",
    ):
        plan_organization(
            package_root,
            timestamp_utc=FIXED_TIMESTAMP,
            promoted_run_id=promoted_run_id,
        )


def test_promoted_proof_rejects_report_not_matching_completed_journal(
    tmp_path: Path,
) -> None:
    package_root, results_root = _package_tree(tmp_path)
    _write_tree(results_root / "g1", {"result.npz": b"legacy"})
    promoted_run_id = _write_promoted_proof(results_root)
    report_path = results_root / "runs" / promoted_run_id / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["quality_limits"]["fallback_frame_fraction_per_artifact"] = 0.09
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(
        UnsafeResultsLayoutError,
        match="matching its final report",
    ):
        plan_organization(
            package_root,
            timestamp_utc=FIXED_TIMESTAMP,
            promoted_run_id=promoted_run_id,
        )


def test_promoted_proof_rejects_tampered_journal_new_tree(
    tmp_path: Path,
) -> None:
    package_root, results_root = _package_tree(tmp_path)
    _write_tree(results_root / "g1", {"result.npz": b"legacy"})
    promoted_run_id = _write_promoted_proof(results_root)
    journal_path = results_root / "runs" / promoted_run_id / "promotion.json"
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    journal["new_tree"]["sha256"] = "f" * 64
    journal_path.write_text(json.dumps(journal), encoding="utf-8")

    with pytest.raises(
        UnsafeResultsLayoutError,
        match="new_tree does not match",
    ):
        plan_organization(
            package_root,
            timestamp_utc=FIXED_TIMESTAMP,
            promoted_run_id=promoted_run_id,
        )


def test_promoted_proof_rejects_wider_quality_profile(
    tmp_path: Path,
) -> None:
    package_root, results_root = _package_tree(tmp_path)
    _write_tree(results_root / "g1", {"result.npz": b"legacy"})
    promoted_run_id = _write_promoted_proof(results_root)
    report_path = results_root / "runs" / promoted_run_id / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["quality_limits"]["fallback_artifact_fraction_per_group"] = 0.11
    _rewrite_report_and_sync_journal(
        results_root,
        promoted_run_id,
        report,
    )

    with pytest.raises(
        UnsafeResultsLayoutError,
        match="wider than the formal maximum",
    ):
        plan_organization(
            package_root,
            timestamp_utc=FIXED_TIMESTAMP,
            promoted_run_id=promoted_run_id,
        )


def test_promoted_proof_rejects_incomplete_robot_matrix(
    tmp_path: Path,
) -> None:
    package_root, results_root = _package_tree(tmp_path)
    _write_tree(results_root / "g1", {"result.npz": b"legacy"})
    promoted_run_id = _write_promoted_proof(results_root)
    report_path = results_root / "runs" / promoted_run_id / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["robots"] = ["g1"]
    _rewrite_report_and_sync_journal(
        results_root,
        promoted_run_id,
        report,
    )

    with pytest.raises(
        UnsafeResultsLayoutError,
        match="complete G1/E1 dataset matrix",
    ):
        plan_organization(
            package_root,
            timestamp_utc=FIXED_TIMESTAMP,
            promoted_run_id=promoted_run_id,
        )


def test_promoted_proof_rejects_any_g1_omission(
    tmp_path: Path,
) -> None:
    package_root, results_root = _package_tree(tmp_path)
    _write_tree(results_root / "g1", {"result.npz": b"legacy"})
    promoted_run_id = _write_promoted_proof(results_root)
    report_path = results_root / "runs" / promoted_run_id / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    omitted_path = str(results_root / "v1" / "canonical" / "g1" / "robot_only" / "amass" / "omitted" / "identity.npz")
    validation = report["validation"]
    validation["expected_artifact_count"] = 3
    validation["statistics"]["audited_omissions"] = {
        "artifact_count": 1,
        "artifact_paths": [omitted_path],
    }
    report["planning"]["expected_artifact_count"] = 3
    counts = report["execution_audit"]["source_job_counts"]
    counts.update(
        {
            "planned_source_count": 3,
            "attempted_source_count": 3,
            "succeeded_source_count": 2,
            "omitted_source_count": 1,
            "planned_artifact_count": 3,
            "omitted_artifact_count": 1,
        }
    )
    g1_counts = report["execution_audit"]["by_robot"]["g1"]
    g1_counts.update(
        {
            "planned_source_count": 2,
            "attempted_source_count": 2,
            "succeeded_source_count": 1,
            "omitted_source_count": 1,
            "planned_artifact_count": 2,
            "omitted_artifact_count": 1,
        }
    )
    report["execution_audit"]["omissions"] = [
        {
            "robot": "g1",
            "task_type": "robot_only",
            "failure_type": "SQPNonlinearFeasibilityError",
            "error": "SQPNonlinearFeasibilityError: unreachable",
            "batch_report_sha256": "a" * 64,
            "omitted_artifact_paths": [omitted_path],
        }
    ]
    _rewrite_report_and_sync_journal(
        results_root,
        promoted_run_id,
        report,
    )

    with pytest.raises(
        UnsafeResultsLayoutError,
        match="strict G1 coverage",
    ):
        plan_organization(
            package_root,
            timestamp_utc=FIXED_TIMESTAMP,
            promoted_run_id=promoted_run_id,
        )


def test_execute_archives_known_layouts_and_writes_complete_manifest(
    tmp_path: Path,
) -> None:
    package_root, results_root = _package_tree(tmp_path)
    for allowed_name in ALLOWED_RESULTS_ENTRIES.difference({"archive"}):
        (results_root / allowed_name).mkdir()
    _write_tree(results_root / "g1", {"a.npz": b"abc", "nested/b.npz": b"12345"})
    _write_tree(results_root / ".smoke_20260729", {"smoke.npz": b"smoke"})
    _write_tree(package_root / "demo_results_parallel", {"e1/result.npz": b"parallel"})
    promoted_run_id = _write_promoted_proof(results_root)

    plan = plan_organization(
        package_root,
        timestamp_utc=FIXED_TIMESTAMP,
        now=FIXED_NOW,
        promoted_run_id=promoted_run_id,
    )
    payload = execute_organization(plan, execute=True)

    archive_root = results_root / "archive" / "legacy" / FIXED_TIMESTAMP
    assert payload["status"] == "completed"
    assert payload["move_count"] == 3
    assert payload["file_count"] == 4
    assert payload["byte_count"] == 21
    assert (archive_root / "g1" / "nested" / "b.npz").read_bytes() == b"12345"
    assert (archive_root / ".smoke_20260729" / "smoke.npz").read_bytes() == b"smoke"
    assert (archive_root / "demo_results_parallel" / "e1" / "result.npz").read_bytes() == b"parallel"
    assert not (package_root / "demo_results_parallel").exists()
    assert {path.name for path in results_root.iterdir()} == ALLOWED_RESULTS_ENTRIES

    manifest = json.loads((archive_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["timestamp_utc"] == FIXED_TIMESTAMP
    assert manifest["executed_at_utc"]
    assert {move["state"] for move in manifest["moves"]} == {"moved"}
    assert {Path(move["source"]).name for move in manifest["moves"]} == {
        "g1",
        ".smoke_20260729",
        "demo_results_parallel",
    }
    assert {Path(move["target"]).parent for move in manifest["moves"]} == {archive_root}
    assert manifest["promotion_proof"]["journal_path"] == str(
        results_root / "runs" / promoted_run_id / "promotion.json"
    )
    assert len(manifest["promotion_proof"]["journal_sha256"]) == 64


def test_runtime_lock_directory_is_retained_as_allowed_state(
    tmp_path: Path,
) -> None:
    package_root, results_root = _package_tree(tmp_path)
    lock_file = results_root / ".locks" / "retargeting" / "ab" / "job.lock"
    lock_file.parent.mkdir(parents=True)
    lock_file.write_text("", encoding="utf-8")
    _write_tree(results_root / "g1", {"result.npz": b"legacy"})
    promoted_run_id = _write_promoted_proof(results_root)

    plan = plan_organization(
        package_root,
        timestamp_utc=FIXED_TIMESTAMP,
        now=FIXED_NOW,
        promoted_run_id=promoted_run_id,
    )
    payload = execute_organization(plan, execute=True)

    assert payload["status"] == "completed"
    assert lock_file.is_file()
    assert not (results_root / "g1").exists()
    assert (results_root / "archive" / "legacy" / FIXED_TIMESTAMP / "g1" / "result.npz").read_bytes() == b"legacy"


def test_unknown_main_entry_is_rejected_instead_of_moved(tmp_path: Path) -> None:
    package_root, results_root = _package_tree(tmp_path)
    _write_tree(results_root / "unclassified_experiment", {"result.npz": b"x"})

    with pytest.raises(UnsafeResultsLayoutError, match="unknown demo_results"):
        plan_organization(package_root, now=FIXED_NOW)

    assert (results_root / "unclassified_experiment" / "result.npz").exists()
    assert not (results_root / "archive").exists()


@pytest.mark.parametrize("symlink_location", ["top_level", "nested"])
def test_symlinks_are_rejected_without_following_them(
    tmp_path: Path,
    symlink_location: str,
) -> None:
    package_root, results_root = _package_tree(tmp_path)
    outside = tmp_path / "outside"
    _write_tree(outside, {"secret.npz": b"secret"})
    if symlink_location == "top_level":
        (results_root / "g1").symlink_to(outside, target_is_directory=True)
    else:
        (results_root / "g1").mkdir()
        (results_root / "g1" / "outside").symlink_to(outside, target_is_directory=True)

    with pytest.raises(UnsafeResultsLayoutError, match=r"[Ss]ymlink"):
        plan_organization(package_root, now=FIXED_NOW)

    assert (outside / "secret.npz").read_bytes() == b"secret"
    assert not (results_root / "archive").exists()


def test_existing_archive_transaction_is_a_hard_conflict(tmp_path: Path) -> None:
    package_root, results_root = _package_tree(tmp_path)
    _write_tree(results_root / "g1", {"result.npz": b"x"})
    (results_root / "archive" / "legacy" / FIXED_TIMESTAMP).mkdir(parents=True)

    with pytest.raises(UnsafeResultsLayoutError, match="already exists"):
        plan_organization(
            package_root,
            timestamp_utc=FIXED_TIMESTAMP,
            now=FIXED_NOW,
        )

    assert (results_root / "g1" / "result.npz").exists()


def test_source_change_after_plan_blocks_execution_before_any_write(
    tmp_path: Path,
) -> None:
    package_root, results_root = _package_tree(tmp_path)
    _write_tree(results_root / "g1", {"result.npz": b"old"})
    promoted_run_id = _write_promoted_proof(results_root)
    plan = plan_organization(
        package_root,
        timestamp_utc=FIXED_TIMESTAMP,
        now=FIXED_NOW,
        promoted_run_id=promoted_run_id,
    )
    (results_root / "g1" / "new.npz").write_bytes(b"new")

    with pytest.raises(ResultsLayoutChangedError, match="changed after planning"):
        execute_organization(plan, execute=True)

    assert (results_root / "g1" / "result.npz").exists()
    assert (results_root / "g1" / "new.npz").exists()
    assert not (results_root / "archive").exists()


def test_same_size_source_byte_change_after_plan_is_detected(
    tmp_path: Path,
) -> None:
    package_root, results_root = _package_tree(tmp_path)
    source_path = results_root / "g1" / "result.npz"
    _write_tree(results_root / "g1", {"result.npz": b"old"})
    promoted_run_id = _write_promoted_proof(results_root)
    plan = plan_organization(
        package_root,
        timestamp_utc=FIXED_TIMESTAMP,
        now=FIXED_NOW,
        promoted_run_id=promoted_run_id,
    )
    original_stat = source_path.stat()
    source_path.write_bytes(b"new")
    os.utime(
        source_path,
        ns=(
            original_stat.st_atime_ns,
            original_stat.st_mtime_ns,
        ),
    )

    with pytest.raises(
        ResultsLayoutChangedError,
        match="changed after planning",
    ):
        execute_organization(plan, execute=True)

    assert source_path.read_bytes() == b"new"
    assert not (results_root / "archive").exists()


def test_tampered_out_of_scope_target_is_rejected_before_any_write(
    tmp_path: Path,
) -> None:
    package_root, results_root = _package_tree(tmp_path)
    _write_tree(results_root / "g1", {"result.npz": b"safe"})
    promoted_run_id = _write_promoted_proof(results_root)
    plan = plan_organization(
        package_root,
        timestamp_utc=FIXED_TIMESTAMP,
        now=FIXED_NOW,
        promoted_run_id=promoted_run_id,
    )
    tampered_move = replace(plan.moves[0], target=tmp_path / "outside" / "g1")
    tampered_plan = replace(plan, moves=(tampered_move,))

    with pytest.raises(ResultsLayoutChangedError, match="changed after planning"):
        execute_organization(tampered_plan, execute=True)

    assert (results_root / "g1" / "result.npz").read_bytes() == b"safe"
    assert not (tmp_path / "outside").exists()
    assert not (results_root / "archive").exists()


def test_failure_rolls_back_every_completed_rename_and_journals_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root, results_root = _package_tree(tmp_path)
    _write_tree(results_root / "g1", {"first.npz": b"first"})
    _write_tree(package_root / "demo_results_parallel", {"second.npz": b"second"})
    promoted_run_id = _write_promoted_proof(results_root)
    plan = plan_organization(
        package_root,
        timestamp_utc=FIXED_TIMESTAMP,
        now=FIXED_NOW,
        promoted_run_id=promoted_run_id,
    )
    real_rename = organizer._rename_path

    def _fail_second_forward_rename(source: Path, target: Path) -> None:
        if source == package_root / "demo_results_parallel":
            raise OSError("injected rename failure")
        real_rename(source, target)

    monkeypatch.setattr(organizer, "_rename_path", _fail_second_forward_rename)
    with pytest.raises(
        ResultsOrganizationTransactionError,
        match="rolled_back",
    ) as exc_info:
        execute_organization(plan, execute=True)

    assert exc_info.value.rollback_errors == ()
    assert (results_root / "g1" / "first.npz").read_bytes() == b"first"
    assert (package_root / "demo_results_parallel" / "second.npz").read_bytes() == b"second"
    archive_root = results_root / "archive" / "legacy" / FIXED_TIMESTAMP
    assert not (archive_root / "g1").exists()
    assert not (archive_root / "demo_results_parallel").exists()
    manifest = json.loads((archive_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "rolled_back"
    states = {Path(move["source"]).name: move["state"] for move in manifest["moves"]}
    assert states == {
        "g1": "rolled_back",
        "demo_results_parallel": "pending",
    }
    assert "injected rename failure" in manifest["error"]


def test_final_manifest_failure_also_rolls_back_the_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root, results_root = _package_tree(tmp_path)
    _write_tree(results_root / "g1", {"result.npz": b"result"})
    promoted_run_id = _write_promoted_proof(results_root)
    plan = plan_organization(
        package_root,
        timestamp_utc=FIXED_TIMESTAMP,
        now=FIXED_NOW,
        promoted_run_id=promoted_run_id,
    )
    real_write_json = organizer._write_json_atomic

    def _fail_completed_manifest(path: Path, payload: dict[str, object]) -> None:
        if payload["status"] == "completed":
            raise OSError("injected completed-manifest failure")
        real_write_json(path, payload)

    monkeypatch.setattr(organizer, "_write_json_atomic", _fail_completed_manifest)
    with pytest.raises(ResultsOrganizationTransactionError, match="rolled_back"):
        execute_organization(plan, execute=True)

    assert (results_root / "g1" / "result.npz").read_bytes() == b"result"
    manifest_path = results_root / "archive" / "legacy" / FIXED_TIMESTAMP / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "rolled_back"
    assert "completed-manifest failure" in manifest["error"]


def test_resume_converges_after_hard_crash_between_rename_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root, results_root = _package_tree(tmp_path)
    _write_tree(results_root / "g1", {"first.npz": b"first"})
    _write_tree(
        package_root / "demo_results_parallel",
        {"second.npz": b"second"},
    )
    promoted_run_id = _write_promoted_proof(results_root)
    plan = plan_organization(
        package_root,
        timestamp_utc=FIXED_TIMESTAMP,
        now=FIXED_NOW,
        promoted_run_id=promoted_run_id,
    )
    first_move = plan.moves[0]
    real_rename = organizer._rename_path
    crashed = False

    def _crash_after_first_rename(source: Path, target: Path) -> None:
        nonlocal crashed
        real_rename(source, target)
        if not crashed:
            crashed = True
            raise SimulatedHardCrash

    monkeypatch.setattr(
        organizer,
        "_rename_path",
        _crash_after_first_rename,
    )
    with pytest.raises(SimulatedHardCrash):
        execute_organization(plan, execute=True)

    assert not first_move.source.exists()
    assert first_move.target.is_dir()
    manifest = json.loads(
        plan.manifest_path.read_text(encoding="utf-8"),
    )
    assert manifest["status"] == "in_progress"
    assert manifest["moves"][0]["state"] == "pending"

    resume_plan = plan_organization(
        package_root,
        timestamp_utc=FIXED_TIMESTAMP,
        promoted_run_id=promoted_run_id,
        resume=True,
    )
    payload = execute_organization(resume_plan, execute=True)

    assert payload["status"] == "completed"
    assert all(not move.source.exists() for move in plan.moves)
    assert all(move.target.is_dir() for move in plan.moves)
    completed_manifest = json.loads(
        plan.manifest_path.read_text(encoding="utf-8"),
    )
    assert {move["state"] for move in completed_manifest["moves"]} == {
        "moved",
    }

    repeated_plan = plan_organization(
        package_root,
        timestamp_utc=FIXED_TIMESTAMP,
        promoted_run_id=promoted_run_id,
        resume=True,
    )
    repeated_payload = execute_organization(repeated_plan, execute=True)

    assert repeated_payload["status"] == "completed"
    assert all(not move.source.exists() for move in plan.moves)
    assert all(move.target.is_dir() for move in plan.moves)


def test_resume_recovers_manifest_creation_crash_and_orphaned_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root, results_root = _package_tree(tmp_path)
    _write_tree(results_root / "g1", {"result.npz": b"legacy"})
    promoted_run_id = _write_promoted_proof(results_root)
    plan = plan_organization(
        package_root,
        timestamp_utc=FIXED_TIMESTAMP,
        now=FIXED_NOW,
        promoted_run_id=promoted_run_id,
    )
    real_write_json = organizer._write_json_atomic
    crashed = False

    def _crash_during_initial_manifest(
        path: Path,
        payload: dict[str, object],
    ) -> None:
        nonlocal crashed
        if not crashed:
            crashed = True
            orphaned_temp = path.parent / ".manifest.json.interrupted.tmp"
            orphaned_temp.write_bytes(b'{"status": "in_progress"')
            raise SimulatedHardCrash
        real_write_json(path, payload)

    monkeypatch.setattr(
        organizer,
        "_write_json_atomic",
        _crash_during_initial_manifest,
    )
    with pytest.raises(SimulatedHardCrash):
        execute_organization(plan, execute=True)

    orphaned_temp = plan.archive_root / ".manifest.json.interrupted.tmp"
    assert orphaned_temp.is_file()
    assert not plan.manifest_path.exists()
    assert plan.moves[0].source.is_dir()
    assert not plan.moves[0].target.exists()

    resume_plan = plan_organization(
        package_root,
        timestamp_utc=FIXED_TIMESTAMP,
        promoted_run_id=promoted_run_id,
        resume=True,
    )
    payload = execute_organization(resume_plan, execute=True)

    assert payload["status"] == "completed"
    assert not orphaned_temp.exists()
    assert not plan.moves[0].source.exists()
    assert plan.moves[0].target.is_dir()


def test_resume_rejects_ambiguous_source_and_target_before_new_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root, results_root = _package_tree(tmp_path)
    _write_tree(results_root / "g1", {"first.npz": b"first"})
    _write_tree(
        package_root / "demo_results_parallel",
        {"second.npz": b"second"},
    )
    promoted_run_id = _write_promoted_proof(results_root)
    plan = plan_organization(
        package_root,
        timestamp_utc=FIXED_TIMESTAMP,
        now=FIXED_NOW,
        promoted_run_id=promoted_run_id,
    )
    first_move, second_move = plan.moves
    real_rename = organizer._rename_path
    crashed = False

    def _crash_after_first_rename(source: Path, target: Path) -> None:
        nonlocal crashed
        real_rename(source, target)
        if not crashed:
            crashed = True
            raise SimulatedHardCrash

    monkeypatch.setattr(
        organizer,
        "_rename_path",
        _crash_after_first_rename,
    )
    with pytest.raises(SimulatedHardCrash):
        execute_organization(plan, execute=True)
    _write_tree(first_move.source, {"first.npz": b"first"})

    resume_plan = plan_organization(
        package_root,
        timestamp_utc=FIXED_TIMESTAMP,
        promoted_run_id=promoted_run_id,
        resume=True,
    )
    with pytest.raises(
        UnsafeResultsLayoutError,
        match=r"ambiguous.*both exist",
    ):
        execute_organization(resume_plan, execute=True)

    assert first_move.source.is_dir()
    assert first_move.target.is_dir()
    assert second_move.source.is_dir()
    assert not second_move.target.exists()


def test_no_legacy_results_is_an_execute_no_op_without_creating_archive(
    tmp_path: Path,
) -> None:
    package_root, results_root = _package_tree(tmp_path)
    promoted_run_id = _write_promoted_proof(results_root)
    plan = plan_organization(
        package_root,
        timestamp_utc=FIXED_TIMESTAMP,
        now=FIXED_NOW,
        promoted_run_id=promoted_run_id,
    )

    payload = execute_organization(plan, execute=True)

    assert payload["status"] == "no_changes"
    assert payload["move_count"] == 0
    assert not (results_root / "archive").exists()
