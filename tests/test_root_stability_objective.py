# ruff: noqa: CPY001, PT009, PT027

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "holosoma_retargeting"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from holosoma_retargeting.config_types.data_type import MotionDataConfig  # noqa: E402
from holosoma_retargeting.config_types.retargeter import (  # noqa: E402
    RetargeterConfig,
    RootStabilityConfig,
)
from holosoma_retargeting.config_types.robot import RobotConfig  # noqa: E402
from holosoma_retargeting.config_types.task import TaskConfig  # noqa: E402
from holosoma_retargeting.retargeting_pipeline import (  # noqa: E402
    build_retargeter_kwargs_from_config,
    create_task_constants,
)
from holosoma_retargeting.src.interaction_mesh_retargeter import (  # noqa: E402
    InteractionMeshRetargeter,
)


def _build_retargeter(
    *,
    root_stability: RootStabilityConfig | None = None,
    interaction_mesh_weight: float = 10.0,
    arm_interaction_mesh_weight_scale: float = 1.0,
    q_a_init_idx: int = -7,
    orientation_weights: dict[str, float] | None = None,
) -> InteractionMeshRetargeter:
    robot_urdf = (
        PACKAGE_ROOT
        / "holosoma_retargeting"
        / "models"
        / "e1"
        / "e1_23dof.urdf"
    )
    constants = create_task_constants(
        RobotConfig(
            robot_type="e1",
            robot_urdf_file=str(robot_urdf),
        ),
        MotionDataConfig(data_format="noetix_mocap", robot_type="e1"),
        TaskConfig(object_name="ground"),
        "robot_only",
    )
    config = RetargeterConfig(
        q_a_init_idx=q_a_init_idx,
        activate_foot_sticking=False,
        root_stability=root_stability or RootStabilityConfig(),
        interaction_mesh_weight=interaction_mesh_weight,
        arm_interaction_mesh_weight_scale=arm_interaction_mesh_weight_scale,
        orientation_weights=orientation_weights or {},
    )
    return InteractionMeshRetargeter(
        **build_retargeter_kwargs_from_config(
            config,
            constants,
            None,
            "robot_only",
        ),
    )


class RootStabilityObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retargeter = _build_retargeter(
            root_stability=RootStabilityConfig(
                position_weight=250.0,
                orientation_weight=80.0,
            ),
            interaction_mesh_weight=7.5,
            arm_interaction_mesh_weight_scale=0.4,
        )

    def _human_motion(self) -> np.ndarray:
        motion = np.zeros(
            (2, len(self.retargeter.demo_joints), 3),
            dtype=np.float64,
        )
        root_index = self.retargeter.demo_joints.index("Hips")
        left_index = self.retargeter.demo_joints.index("LeftArm")
        right_index = self.retargeter.demo_joints.index("RightArm")
        motion[0, root_index] = (0.0, 0.0, 1.0)
        motion[0, left_index] = (0.0, 0.25, 1.5)
        motion[0, right_index] = (0.0, -0.25, 1.5)

        source_rotation = Rotation.from_euler("z", 0.2).as_matrix()
        source_translation = np.asarray((0.12, -0.03, 0.04))
        for joint_index in (root_index, left_index, right_index):
            relative = motion[0, joint_index] - motion[0, root_index]
            motion[1, joint_index] = (
                source_rotation @ relative
                + motion[0, root_index]
                + source_translation
            )
        return motion

    def test_targets_preserve_source_root_delta_and_torso_rotation(self):
        initial_q = self.retargeter.robot_model.qpos0.copy()
        target_positions, target_matrices = (
            self.retargeter._prepare_root_stability_targets(
                self._human_motion(),
                initial_q,
            )
        )
        robot_position, robot_matrix, _, _ = (
            self.retargeter._get_root_stability_data(
                initial_q,
                with_jacobians=False,
            )
        )

        np.testing.assert_allclose(target_positions[0], robot_position)
        np.testing.assert_allclose(
            target_positions[1] - target_positions[0],
            (0.12, -0.03, 0.04),
        )
        np.testing.assert_allclose(target_matrices[0], robot_matrix, atol=1e-12)
        relative_target = target_matrices[1] @ target_matrices[0].T
        expected_relative = Rotation.from_euler("z", 0.2).as_matrix()
        np.testing.assert_allclose(relative_target, expected_relative, atol=1e-12)

    def test_arm_scale_only_changes_upper_limb_mesh_rows(self):
        robot_link_keys = list(self.retargeter.laplacian_match_links)
        vertex_count = len(robot_link_keys) + 3
        weights = self.retargeter._interaction_mesh_vertex_weights(
            robot_link_keys,
            vertex_count,
        )

        for index, name in enumerate(robot_link_keys):
            expected = 3.0 if self.retargeter._is_upper_limb_anchor(name) else 7.5
            self.assertEqual(weights[index], expected)
        np.testing.assert_allclose(weights[len(robot_link_keys) :], 7.5)

    def test_direct_root_orientations_replace_the_shoulder_basis(self):
        initial_q = self.retargeter.robot_model.qpos0.copy()
        source_matrices = Rotation.from_euler(
            "xyz",
            ((0.0, 0.0, 0.0), (0.1, -0.05, 0.2)),
        ).as_matrix()

        _, target_matrices = self.retargeter._prepare_root_stability_targets(
            self._human_motion(),
            initial_q,
            root_orientation_reference_matrices=source_matrices,
        )
        _, initial_robot_matrix, _, _ = (
            self.retargeter._get_root_stability_data(
                initial_q,
                with_jacobians=False,
            )
        )

        np.testing.assert_allclose(
            target_matrices[0],
            initial_robot_matrix,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            target_matrices[1] @ target_matrices[0].T,
            source_matrices[1] @ source_matrices[0].T,
            atol=1e-12,
        )
        self.assertEqual(
            self.retargeter.root_stability_orientation_source,
            "direct_root_orientation",
        )

    def test_bootstrap_keeps_generic_hips_orientation_until_root_alignment(self):
        retargeter = _build_retargeter(
            root_stability=RootStabilityConfig(orientation_weight=80.0),
            orientation_weights={"Hips": 1.0},
        )
        root_index = retargeter.root_stability_orientation_index

        self.assertGreaterEqual(root_index, 0)
        self.assertNotIn(
            root_index,
            retargeter._reserved_orientation_objective_indices(
                root_stability_bootstrap=True,
            ),
        )
        self.assertIn(
            root_index,
            retargeter._reserved_orientation_objective_indices(
                root_stability_bootstrap=False,
            ),
        )

    def test_position_only_mode_does_not_build_unused_torso_targets(self):
        retargeter = _build_retargeter(
            root_stability=RootStabilityConfig(position_weight=1.0),
        )
        motion = np.zeros(
            (1, len(retargeter.demo_joints), 3),
            dtype=np.float64,
        )

        _, target_matrices = retargeter._prepare_root_stability_targets(
            motion,
            retargeter.robot_model.qpos0.copy(),
        )

        self.assertEqual(target_matrices.shape, (1, 0, 3, 3))

    def test_positive_root_weight_requires_the_full_floating_base(self):
        with self.assertRaisesRegex(ValueError, "q_a_init_idx=-7"):
            _build_retargeter(
                root_stability=RootStabilityConfig(position_weight=1.0),
                q_a_init_idx=0,
            )

    def test_negative_tuning_weights_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            _build_retargeter(interaction_mesh_weight=-1.0)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            _build_retargeter(
                root_stability=RootStabilityConfig(orientation_weight=-1.0),
            )


if __name__ == "__main__":
    unittest.main()
