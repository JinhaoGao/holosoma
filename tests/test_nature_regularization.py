# ruff: noqa: CPY001, PT009, PT027

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import mujoco
import numpy as np
import tyro

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "holosoma_retargeting"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from holosoma_retargeting.config_types.data_type import MotionDataConfig  # noqa: E402
from holosoma_retargeting.config_types.retargeting import (  # noqa: E402
    DatasetName,
    RetargeterRuntimeOptions,
    RetargetingCommand,
    RobotName,
    internal_config_from_command,
)
from holosoma_retargeting.config_types.robot_profiles import (  # noqa: E402
    load_robot_profile,
)
from holosoma_retargeting.config_types.task import TaskConfig  # noqa: E402
from holosoma_retargeting.retargeting_pipeline import (  # noqa: E402
    build_retargeter_kwargs_from_config,
    create_task_constants,
)
from holosoma_retargeting.src.interaction_mesh_retargeter import (  # noqa: E402
    InteractionMeshRetargeter,
)

ROBOT_PROFILE_DIR = PACKAGE_ROOT / "holosoma_retargeting" / "examples" / "robot_profiles"


def _command(
    robot: RobotName,
    *,
    dataset: DatasetName = "noetix_mocap",
    nature_weights: float | None = None,
    natural_pose_tracking: bool | None = None,
    robot_profile: Path | None = None,
) -> RetargetingCommand:
    return RetargetingCommand(
        task="robot_only",
        robot=robot,
        dataset=dataset,
        retargeter=RetargeterRuntimeOptions(visualize=False),
        nature_weights=nature_weights,
        natural_pose_tracking=natural_pose_tracking,
        robot_profile=robot_profile,
    )


def _build_retargeter(command: RetargetingCommand) -> InteractionMeshRetargeter:
    config = internal_config_from_command(command)
    robot = str(command.robot)
    constants = create_task_constants(
        config.robot_config,
        MotionDataConfig(data_format=str(config.data_format), robot_type=robot),
        TaskConfig(object_name="ground"),
        "robot_only",
    )
    return InteractionMeshRetargeter(
        **build_retargeter_kwargs_from_config(
            config.retargeter,
            constants,
            None,
            "robot_only",
        ),
    )


class RobotProfileTests(unittest.TestCase):
    def test_optional_objectives_are_disabled_by_profile_default(self) -> None:
        config = internal_config_from_command(_command("g1"))

        self.assertEqual(config.retargeter.orientation_weights, {})
        self.assertEqual(config.retargeter.natural_pose_joint_positions, {})
        self.assertEqual(config.retargeter.natural_pose_weights, {})

    def test_uniform_nature_weight_applies_to_every_actuated_joint(self) -> None:
        config = internal_config_from_command(
            _command("g1", nature_weights=0.25),
        )
        retargeter = _build_retargeter(_command("g1", nature_weights=0.25))

        self.assertEqual(len(config.retargeter.natural_pose_weights), 29)
        self.assertEqual(set(config.retargeter.natural_pose_weights.values()), {0.25})
        self.assertEqual(len(retargeter.natural_pose_joint_names), 29)

    def test_one_profile_covers_each_concrete_robot(self) -> None:
        expected_joint_counts: dict[RobotName, int] = {
            "g1": 29,
            "e1_23dof": 23,
            "e1_24dof": 24,
            "e2": 23,
        }
        for robot, expected_joint_count in expected_joint_counts.items():
            with self.subTest(robot=robot):
                profile = load_robot_profile(robot=robot)
                raw = json.loads(profile.path.read_text(encoding="utf-8"))
                retargeter = _build_retargeter(
                    _command(robot, natural_pose_tracking=True),
                )

                self.assertEqual(raw["schema_version"], 1)
                self.assertEqual(profile.robot, robot)
                self.assertEqual(profile.robot_dof, expected_joint_count)
                self.assertEqual(
                    len(profile.natural_pose.references),
                    expected_joint_count,
                )
                self.assertEqual(
                    set(profile.natural_pose.references),
                    set(profile.natural_pose.weights),
                )
                self.assertEqual(
                    set(profile.natural_pose.references),
                    set(retargeter.robot_actuated_joint_names),
                )

    def test_profile_natural_pose_weights_preserve_the_old_tables(self) -> None:
        expected = {
            "g1": ({"left_shoulder_pitch_joint", "right_shoulder_pitch_joint"}, 2.0),
            "e1_23dof": ({"l_arm_shoulder_pitch_joint", "r_arm_shoulder_pitch_joint"}, 0.5),
            "e1_24dof": (set(), None),
            "e2": ({"l_arm_shoulder_pitch_joint", "r_arm_shoulder_pitch_joint"}, 0.8),
        }
        for robot, (joint_names, weight) in expected.items():
            with self.subTest(robot=robot):
                profile = load_robot_profile(robot=robot)
                active = {name: value for name, value in profile.natural_pose.weights.items() if value > 0.0}
                self.assertEqual(set(active), joint_names)
                self.assertEqual(
                    set(active.values()),
                    set() if weight is None else {weight},
                )

    def test_natural_pose_switch_uses_the_profile_table(self) -> None:
        for robot in ("g1", "e1_23dof", "e1_24dof", "e2"):
            with self.subTest(robot=robot):
                profile = load_robot_profile(robot=robot)
                config = internal_config_from_command(
                    _command(robot, natural_pose_tracking=True),
                )
                self.assertEqual(
                    config.retargeter.natural_pose_joint_positions,
                    profile.natural_pose.references,
                )
                self.assertEqual(
                    config.retargeter.natural_pose_weights,
                    profile.natural_pose.weights,
                )

    def test_e1_alias_resolves_the_e1_23dof_profile(self) -> None:
        profile = load_robot_profile(robot="e1")
        config = internal_config_from_command(_command("e1"))

        self.assertEqual(profile.robot, "e1_23dof")
        self.assertEqual(config.robot_config.ROBOT_DOF, 23)
        self.assertEqual(config.robot_config.ROBOT_HEIGHT, 1.5)
        self.assertEqual(
            config.robot_config.ROBOT_URDF_FILE,
            "models/e1/e1_23dof.urdf",
        )

    def test_g1_natural_shoulder_yaw_references_are_mirrored(self) -> None:
        references = load_robot_profile(robot="g1").natural_pose.references
        self.assertAlmostEqual(
            references["left_shoulder_yaw_joint"],
            -references["right_shoulder_yaw_joint"],
        )

    def test_nature_weight_rejects_invalid_numbers(self) -> None:
        for value in (-0.1, float("inf"), float("nan"), True):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError,
                "finite and non-negative",
            ):
                internal_config_from_command(_command("g1", nature_weights=value))

    def test_robot_profile_rejects_natural_pose_joint_count_mismatch(self) -> None:
        raw = json.loads(
            (ROBOT_PROFILE_DIR / "e1_23dof.json").read_text(encoding="utf-8"),
        )
        raw["natural_pose"]["joint_names"].pop()
        with tempfile.TemporaryDirectory() as temporary_dir:
            profile_path = Path(temporary_dir) / "e1_23dof.json"
            profile_path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "every actuated joint"):
                internal_config_from_command(
                    _command("e1_23dof", robot_profile=profile_path),
                )

    def test_underscore_cli_aliases_are_supported(self) -> None:
        command = tyro.cli(
            RetargetingCommand,
            args=[
                "--robot_profile",
                str(ROBOT_PROFILE_DIR / "g1.json"),
                "--orientation_weights",
                "0.5",
                "--nature_weights",
                "0.25",
            ],
        )

        self.assertEqual(command.robot_profile, ROBOT_PROFILE_DIR / "g1.json")
        self.assertEqual(command.orientation_weights, 0.5)
        self.assertEqual(command.nature_weights, 0.25)


class NatureObjectiveTests(unittest.TestCase):
    def test_initialization_and_objective_pull_joint_toward_fixed_reference(
        self,
    ) -> None:
        command = _command("e1", nature_weights=10.0)
        config = internal_config_from_command(command)
        retargeter = _build_retargeter(command)
        joint_name = "l_arm_shoulder_yaw_joint"
        joint_id = mujoco.mj_name2id(
            retargeter.robot_model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )
        qpos_address = int(retargeter.robot_model.jnt_qposadr[joint_id])
        reference = config.retargeter.natural_pose_joint_positions[joint_name]

        initialized = retargeter.apply_natural_pose_to_initial_qpos(
            retargeter.robot_model.qpos0,
        )
        self.assertAlmostEqual(initialized[qpos_address], reference)
        self.assertFalse(np.shares_memory(initialized, retargeter.robot_model.qpos0))

        qpos = retargeter.robot_model.qpos0.copy()
        qpos[qpos_address] = reference + 0.15
        robot_keys = list(retargeter.laplacian_match_links)
        zero_jacobians = {key: np.zeros((3, retargeter.nq_a), dtype=np.float64) for key in robot_keys}
        zero_positions = {key: np.zeros(3, dtype=np.float64) for key in robot_keys}
        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(
                    retargeter,
                    "_calc_manipulator_jacobians",
                    return_value=(zero_jacobians, zero_positions, None),
                ),
            )
            stack.enter_context(
                mock.patch.object(
                    retargeter,
                    "_update_jacobians_and_phis_from_q",
                    return_value=({}, {}),
                ),
            )
            stack.enter_context(
                mock.patch.object(
                    retargeter,
                    "_compute_self_collision_constraints",
                    return_value=({}, {}),
                ),
            )
            candidate, _ = retargeter.solve_single_iteration(
                q_locked=qpos,
                q_a_n_last=qpos[retargeter.q_a_indices],
                q_t_last=qpos,
                target_laplacian=np.zeros(
                    (len(robot_keys), 3),
                    dtype=np.float64,
                ),
                adj_list=[[] for _ in robot_keys],
                obj_pts_local=np.empty((0, 3), dtype=np.float64),
                foot_sticking={"left": False, "right": False},
                init_t=True,
                frame_idx=0,
            )

        self.assertLess(
            abs(candidate[qpos_address] - reference),
            abs(qpos[qpos_address] - reference),
        )


if __name__ == "__main__":
    unittest.main()
