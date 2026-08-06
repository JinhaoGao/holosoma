# ruff: noqa: CPY001, PT009, PT027

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from typing import Literal
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
    RetargeterRuntimeOptions,
    RetargetingCommand,
    internal_config_from_command,
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

NATURE_PROFILE_DIR = PACKAGE_ROOT / "holosoma_retargeting" / "examples" / "nature_weights"
ORIENTATION_PROFILE_DIR = PACKAGE_ROOT / "holosoma_retargeting" / "examples" / "orientation_weights"


def _command(
    robot: Literal["g1", "e1", "e1_23dof", "e1_24dof", "e2"],
    *,
    dataset: str = "noetix_mocap",
    nature_weights: float | None = None,
    nature_config: Path | None = None,
) -> RetargetingCommand:
    return RetargetingCommand(
        task="robot_only",
        robot=robot,
        dataset=dataset,
        retargeter=RetargeterRuntimeOptions(visualize=False),
        nature_weights=nature_weights,
        nature_config=nature_config,
    )


def _build_retargeter(command: RetargetingCommand) -> InteractionMeshRetargeter:
    config = internal_config_from_command(command)
    robot = str(command.robot)
    constants = create_task_constants(
        RobotConfig(robot_type=robot),
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


class NatureProfileTests(unittest.TestCase):
    def test_optional_objectives_are_disabled_by_default(self) -> None:
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

    def test_profiles_correspond_to_every_explicit_robot_model(self) -> None:
        expected_joint_counts = {
            "g1": 29,
            "e1_23dof": 23,
            "e1_24dof": 24,
            "e2": 23,
        }
        expected_active_joint_counts = {
            "g1": 2,
            "e1_23dof": 2,
            "e1_24dof": 2,
            "e2": 2,
        }
        for robot, expected_joint_count in expected_joint_counts.items():
            with self.subTest(robot=robot):
                profile_path = NATURE_PROFILE_DIR / f"{robot}.json"
                profile = json.loads(profile_path.read_text(encoding="utf-8"))
                config = internal_config_from_command(
                    _command(robot, nature_config=NATURE_PROFILE_DIR),
                )
                retargeter = _build_retargeter(
                    _command(robot, nature_config=profile_path),
                )

                self.assertEqual(profile["schema_version"], 1)
                self.assertEqual(profile["robot"], robot)
                self.assertEqual(set(profile["references"]), set(profile["weights"]))
                self.assertEqual(len(profile["references"]), expected_joint_count)
                self.assertEqual(
                    set(profile["references"]),
                    set(retargeter.robot_actuated_joint_names),
                )
                self.assertEqual(
                    len(retargeter.natural_pose_joint_names),
                    expected_active_joint_counts[robot],
                )
                self.assertEqual(
                    config.retargeter.natural_pose_weights,
                    profile["weights"],
                )

    def test_profiles_regularize_only_both_shoulder_pitch_joints(self) -> None:
        expected_pitch_joints = {
            "g1": {
                "left_shoulder_pitch_joint",
                "right_shoulder_pitch_joint",
            },
            "e1_23dof": {
                "l_arm_shoulder_pitch_joint",
                "r_arm_shoulder_pitch_joint",
            },
            "e1_24dof": {
                "l_arm_shoulder_pitch_joint",
                "r_arm_shoulder_pitch_joint",
            },
            "e2": {
                "l_arm_shoulder_pitch_joint",
                "r_arm_shoulder_pitch_joint",
            },
        }
        expected_weights = {
            "g1": 2.0,
            "e1_23dof": 0.5,
            "e1_24dof": 0.5,
            "e2": 0.5,
        }
        for robot, pitch_joints in expected_pitch_joints.items():
            with self.subTest(robot=robot):
                profile = json.loads(
                    (NATURE_PROFILE_DIR / f"{robot}.json").read_text(
                        encoding="utf-8",
                    ),
                )
                active_weights = {
                    joint_name: weight
                    for joint_name, weight in profile["weights"].items()
                    if weight > 0
                }
                self.assertEqual(set(active_weights), pitch_joints)
                self.assertEqual(
                    set(active_weights.values()),
                    {expected_weights[robot]},
                )

    def test_fbx_mocap_selects_each_robot_natural_pose_table(self) -> None:
        for robot in ("g1", "e1_23dof", "e1_24dof", "e2"):
            with self.subTest(robot=robot):
                profile = json.loads(
                    (NATURE_PROFILE_DIR / f"{robot}.json").read_text(
                        encoding="utf-8",
                    ),
                )
                config = internal_config_from_command(
                    _command(
                        robot,
                        dataset="fbx_mocap",
                        nature_config=NATURE_PROFILE_DIR,
                    ),
                )
                self.assertEqual(
                    config.retargeter.natural_pose_joint_positions,
                    profile["references"],
                )
                self.assertEqual(
                    config.retargeter.natural_pose_weights,
                    profile["weights"],
                )

    def test_g1_natural_shoulder_yaw_references_are_mirrored(self) -> None:
        profile = json.loads(
            (NATURE_PROFILE_DIR / "g1.json").read_text(encoding="utf-8"),
        )
        references = profile["references"]
        self.assertAlmostEqual(
            references["left_shoulder_yaw_joint"],
            -references["right_shoulder_yaw_joint"],
        )

    def test_uniform_and_profile_nature_sources_are_mutually_exclusive(self) -> None:
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            internal_config_from_command(
                _command(
                    "e1",
                    nature_weights=1.0,
                    nature_config=NATURE_PROFILE_DIR / "e1_23dof.json",
                ),
            )

    def test_nature_weight_rejects_invalid_numbers(self) -> None:
        for value in (-0.1, float("inf"), float("nan"), True):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError,
                "finite and non-negative",
            ):
                internal_config_from_command(_command("g1", nature_weights=value))

    def test_nature_profile_rejects_mismatched_tables(self) -> None:
        profile = json.loads(
            (NATURE_PROFILE_DIR / "e1_23dof.json").read_text(encoding="utf-8"),
        )
        profile["weights"].pop("l_arm_shoulder_yaw_joint")
        with tempfile.TemporaryDirectory() as temporary_dir:
            profile_path = Path(temporary_dir) / "e1_23dof.json"
            profile_path.write_text(json.dumps(profile), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly the same joint names"):
                internal_config_from_command(
                    _command("e1_23dof", nature_config=profile_path),
                )

    def test_underscore_cli_aliases_are_supported(self) -> None:
        uniform = tyro.cli(
            RetargetingCommand,
            args=["--orientation_weights", "0.5", "--nature_weights", "0.25"],
        )
        configured = tyro.cli(
            RetargetingCommand,
            args=[
                "--orientation_config",
                str(ORIENTATION_PROFILE_DIR / "g1.json"),
                "--nature_config",
                str(NATURE_PROFILE_DIR / "g1.json"),
            ],
        )

        self.assertEqual(uniform.orientation_weights, 0.5)
        self.assertEqual(uniform.nature_weights, 0.25)
        self.assertEqual(
            configured.orientation_config,
            ORIENTATION_PROFILE_DIR / "g1.json",
        )
        self.assertEqual(configured.nature_config, NATURE_PROFILE_DIR / "g1.json")


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
                target_laplacian=np.zeros((len(robot_keys), 3), dtype=np.float64),
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
