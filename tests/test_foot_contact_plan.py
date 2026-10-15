# ruff: noqa: CPY001

"""Backward compatibility for previously exported contact mode metadata."""

import numpy as np
from holosoma_retargeting.visualization.result_loader import _load_foot_sticking_npz


def test_saved_contact_plan_metadata_round_trips(tmp_path) -> None:
    path = tmp_path / "contact_plan.npz"
    modes = np.asarray((("swing", "swing"), ("pivot", "flat")))
    phase_ids = np.asarray(((-1, -1), (0, 0)), dtype=np.int32)
    pivot_uv = np.zeros((2, 2, 2), dtype=np.float32)
    reference_displacements = np.zeros((2, 2, 2), dtype=np.float32)
    np.savez(
        path,
        foot_sticking_states=np.asarray(((False, False), (True, True))),
        foot_sticking_side_names=np.asarray(("left", "right")),
        foot_sticking_enabled_for_saved_trajectory=True,
        foot_sticking_tolerance=1e-3,
        foot_contact_modes=modes,
        foot_contact_phase_ids=phase_ids,
        foot_contact_confidences=np.ones((2, 2), dtype=np.float32),
        foot_contact_pivot_regions=np.asarray((("sole", "sole"), ("toe", "sole"))),
        foot_contact_pivot_uv=pivot_uv,
        foot_contact_reference_displacements_xy=reference_displacements,
        foot_contact_plan_version=1,
    )

    with np.load(path) as data:
        metadata = _load_foot_sticking_npz(data, num_frames=2)

    assert metadata is not None
    np.testing.assert_array_equal(metadata["modes"], modes)
    np.testing.assert_array_equal(metadata["phase_ids"], phase_ids)
    np.testing.assert_array_equal(metadata["pivot_uv"], pivot_uv)
    assert metadata["plan_version"] == 1
