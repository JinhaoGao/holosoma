# ruff: noqa: CPY001
"""Exercise independent human references through the complete comparison scene."""

import numpy as np
from holosoma_retargeting.visualization import multi_scene
from holosoma_retargeting.visualization.result_loader import load_variant_result


def test_each_result_renders_its_own_human_through_playback_and_visibility(tmp_path, monkeypatch):
    results, paths = [], []
    names = np.asarray(["Pelvis", "L_Hip", "R_Hip"])
    for index in range(2):
        path = tmp_path / f"{index}.npz"
        qpos = np.zeros((3, 36))
        qpos[:, 3] = 1
        points = np.zeros((3, 3, 3))
        points[:, :, 2] = (np.arange(3)[:, None] + index * 3) * 0.2
        points[:, 1, 1] = 0.1
        points[:, 2, 1] = -0.1
        np.savez(
            path,
            qpos=qpos,
            fps=30.0,
            robot_type="g1",
            contains_object_in_qpos=False,
            human_joints=points,
            human_joint_names=names,
            mapped_human_joint_names=names,
            mapped_robot_joints=np.zeros_like(points),
            mapped_robot_link_names=["pelvis_contour_link", "left_hip_pitch_link", "right_hip_pitch_link"],
        )
        paths.append(path)
        results.append(load_variant_result(str(index), path))
    scenes, playback = [], {}
    original = multi_scene.ComparisonScene

    def capture_scene(**kwargs):
        scene = original(**kwargs)
        scenes.append(scene)
        return scene

    monkeypatch.setattr(multi_scene, "ComparisonScene", capture_scene)
    monkeypatch.setattr(multi_scene, "create_motion_control_sliders", lambda **kwargs: playback.update(kwargs))
    server = multi_scene.make_multi_result_player(
        multi_scene.MultiResultViserConfig(qpos_npzs=tuple(paths)),
        ("first", "second"),
        results,
    )
    try:
        assert len(scenes) == 2
        for frame in (0.0, 1.5, 2.0):
            playback["on_frame"](results[0].qpos[0], frame)
            for scene in scenes:
                expected = multi_scene.interpolate_human_points(scene.result.human_joints, frame, loop=True)
                np.testing.assert_allclose(
                    scene.human_reference.joints_handle.points, expected + scene.offset, atol=5e-4
                )
        assert not np.array_equal(
            scenes[0].human_reference.trajectory.points, scenes[1].human_reference.trajectory.points
        )
        scenes[1].set_enabled(False)
        assert not scenes[1].group_root.visible
        assert scenes[0].group_root.visible
        scenes[1].set_enabled(True)
        assert scenes[1].group_root.visible
    finally:
        server.stop()
