# ruff: noqa: PT009, PT027

from __future__ import annotations

import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "holosoma_retargeting"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from holosoma_retargeting.data_utils.omomo import (  # noqa: E402
    OMOMO_OBJECT_NAMES,
    parse_omomo_result_name,
    parse_omomo_sequence_name,
    preflight_omomo_dataset,
    resolve_omomo_result_object_name,
    select_omomo_files,
)


def _write_heights(path: Path, heights: dict[str, float]) -> None:
    with path.open("wb") as file:
        pickle.dump(heights, file)


class OmomoSequenceNameTests(unittest.TestCase):
    def test_parse_round_trip(self):
        sequence = parse_omomo_sequence_name("sub17_woodchair_009.pt")
        self.assertEqual(sequence.subject, "sub17")
        self.assertEqual(sequence.object_name, "woodchair")
        self.assertEqual(sequence.sequence_index, 9)
        self.assertEqual(sequence.stem, "sub17_woodchair_009")

    def test_rejects_malformed_and_unknown_names(self):
        for invalid in ("sub1_largebox_1", "sub0_largebox_001", "sub1_unknown_001"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_omomo_sequence_name(invalid)

    def test_selection_uses_exact_object_category(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            torch.save(torch.zeros(1, 325), directory / "sub1_largebox_001.pt")
            torch.save(torch.zeros(1, 325), directory / "sub1_smallbox_001.pt")

            self.assertEqual(
                select_omomo_files(directory, ["largebox"]),
                [directory / "sub1_largebox_001.pt"],
            )

    def test_result_names_and_metadata_resolve_the_same_object(self):
        self.assertEqual(
            parse_omomo_result_name("sub2_woodchair_019_trans_2.npz").object_name,
            "woodchair",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = Path(tmpdir) / "sub2_woodchair_019_original.npz"
            np.savez(result_path, object_name=np.asarray("woodchair"))
            self.assertEqual(
                resolve_omomo_result_object_name(result_path),
                "woodchair",
            )

    def test_result_object_resolution_rejects_disagreement(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = Path(tmpdir) / "sub2_woodchair_019_original.npz"
            np.savez(result_path, object_name=np.asarray("tripod"))
            with self.assertRaisesRegex(ValueError, "conflicting"):
                resolve_omomo_result_object_name(result_path)


class OmomoPreflightTests(unittest.TestCase):
    def test_validates_tensors_heights_and_assets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "OMOMO_new"
            asset_root = root / "models"
            data_dir.mkdir()
            _write_heights(root / "height_dict.pkl", {"sub1": 1.7})
            torch.save(torch.zeros(2, 325), data_dir / "sub1_largebox_001.pt")

            largebox_dir = asset_root / "largebox"
            largebox_dir.mkdir(parents=True)
            (largebox_dir / "largebox.obj").write_text("v 0 0 0\n", encoding="utf-8")
            (largebox_dir / "largebox.urdf").write_text("<robot/>\n", encoding="utf-8")

            report = preflight_omomo_dataset(data_dir, asset_root=asset_root)

            self.assertEqual(report.total_files, 1)
            self.assertEqual(report.valid_sequences, 1)
            self.assertEqual(report.subjects, ("sub1",))
            counts = {item.object_name: item.sequence_count for item in report.objects}
            self.assertEqual(counts["largebox"], 1)
            self.assertEqual(len(counts), len(OMOMO_OBJECT_NAMES))
            issue_codes = {issue.code for issue in report.issues}
            self.assertNotIn("invalid_tensor", issue_codes)
            self.assertIn("missing_object_mesh", issue_codes)

    def test_reports_bad_tensor_and_missing_subject_height(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "OMOMO_new"
            data_dir.mkdir()
            _write_heights(root / "height_dict.pkl", {"sub2": 1.8})
            torch.save(torch.zeros(0, 324), data_dir / "sub1_tripod_001.pt")

            report = preflight_omomo_dataset(data_dir)

            self.assertFalse(report.ok)
            self.assertEqual(
                {issue.code for issue in report.issues},
                {"invalid_tensor", "missing_subject_height"},
            )

    def test_report_json_includes_status_and_inventory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "OMOMO_new"
            data_dir.mkdir()
            _write_heights(root / "height_dict.pkl", {"sub1": 1.7})
            torch.save(torch.zeros(1, 325), data_dir / "sub1_monitor_001.pt")

            payload = preflight_omomo_dataset(data_dir).to_json()

            self.assertIn('"ok": true', payload)
            self.assertIn('"object_name": "monitor"', payload)


if __name__ == "__main__":
    unittest.main()
