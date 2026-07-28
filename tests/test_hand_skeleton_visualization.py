# ruff: noqa: PT009

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "holosoma_retargeting"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from holosoma_retargeting.config_types.data_type import (  # noqa: E402
    JOINTS_MAPPINGS,
    OMOMO_DEMO_JOINTS,
)
from holosoma_retargeting.data_utils.hand_skeleton import (  # noqa: E402
    build_hand_visualization_spec,
)


class HandSkeletonVisualizationTests(unittest.TestCase):
    def test_omomo_full_finger_chains_are_visual_only(self):
        mapped_names = list(JOINTS_MAPPINGS[("omomo", "g1")])
        spec = build_hand_visualization_spec(OMOMO_DEMO_JOINTS, mapped_names)
        joint_index = {name: index for index, name in enumerate(OMOMO_DEMO_JOINTS)}

        self.assertEqual(len(spec.keypoint_indices), 30)
        self.assertEqual(len(spec.edge_indices), 30)
        self.assertTrue(set(spec.keypoint_indices).isdisjoint(joint_index[name] for name in mapped_names))
        self.assertIn(
            (joint_index["L_Wrist"], joint_index["L_Thumb1"]),
            spec.edge_indices,
        )
        self.assertIn(
            (joint_index["R_Index2"], joint_index["R_Index3"]),
            spec.edge_indices,
        )

    def test_missing_finger_names_yield_no_visual_overlay(self):
        spec = build_hand_visualization_spec(
            ["Pelvis", "L_Wrist", "R_Wrist"],
            ["Pelvis", "L_Wrist", "R_Wrist"],
        )
        self.assertEqual(spec.keypoint_indices, ())
        self.assertEqual(spec.edge_indices, ())


if __name__ == "__main__":
    unittest.main()
