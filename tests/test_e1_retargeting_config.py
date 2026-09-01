# ruff: noqa: CPY001

from __future__ import annotations

from pathlib import Path

import mujoco
import tyro

from holosoma_retargeting.config_types.data_type import (
    DEMO_JOINTS_REGISTRY,
    MotionDataConfig,
)
from holosoma_retargeting.config_types.retargeting import (
    RetargetingCommand,
    RobotName,
    internal_config_from_command,
)
from holosoma_retargeting.config_types.robot import RobotConfig
from holosoma_retargeting.config_types.robot_profiles import load_robot_profile
from holosoma_retargeting.config_types.task import TaskConfig
from holosoma_retargeting.retargeting_pipeline import (
    build_retargeter_kwargs_from_config,
    create_task_constants,
    normalize_retargeting_config,
)
from holosoma_retargeting.src.interaction_mesh_retargeter import (
    InteractionMeshRetargeter,
)
from holosoma_retargeting.src.viser_utils import (
    actuated_joint_names_from_mujoco_xml,
    actuated_joint_names_from_urdf,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "holosoma_retargeting" / "holosoma_retargeting"
ROBOT_PROFILE_DIR = PACKAGE_ROOT / "examples" / "robot_profiles"


def _model(robot: str) -> mujoco.MjModel:
    path = (PACKAGE_ROOT / RobotConfig(robot_type=robot).ROBOT_URDF_FILE).with_suffix(".xml")
    return mujoco.MjModel.from_xml_path(str(path))


def test_e1_variants_select_the_expected_assets_and_dofs() -> None:
    legacy = RobotConfig(robot_type="e1")
    e1_23 = RobotConfig(robot_type="e1_23dof")
    e1_24 = RobotConfig(robot_type="e1_24dof")

    assert legacy.ROBOT_URDF_FILE == e1_23.ROBOT_URDF_FILE
    assert legacy.ROBOT_HEIGHT == e1_23.ROBOT_HEIGHT == 1.5
    assert e1_23.ROBOT_DOF == 23
    assert e1_24.ROBOT_DOF == 24
    assert e1_24.ROBOT_URDF_FILE == "models/e1/e1_24dof.urdf"
    assert e1_23.NOMINAL_TRACKING_INDICES.tolist() == list(range(20))
    assert e1_24.NOMINAL_TRACKING_INDICES.tolist() == list(range(21))

    model_23 = _model("e1_23dof")
    model_24 = _model("e1_24dof")
    assert (model_23.nq, model_23.nv) == (30, 29)
    assert (model_24.nq, model_24.nv) == (31, 30)
    waist_roll_id = mujoco.mj_name2id(
        model_24,
        mujoco.mjtObj.mjOBJ_JOINT,
        "waist_roll_joint",
    )
    assert model_24.jnt_range[waist_roll_id].tolist() == [-0.52, 0.52]
    assert actuated_joint_names_from_urdf(
        PACKAGE_ROOT / e1_24.ROBOT_URDF_FILE,
    ) == actuated_joint_names_from_mujoco_xml(
        (PACKAGE_ROOT / e1_24.ROBOT_URDF_FILE).with_suffix(".xml"),
    )


def test_e1_variants_cover_every_registered_human_format() -> None:
    for robot in ("e1_23dof", "e1_24dof"):
        model = _model(robot)
        body_names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) for body_id in range(1, model.nbody)}
        config = RobotConfig(robot_type=robot)
        assert set(config.FOOT_STICKING_LINKS).issubset(body_names)

        for data_format in DEMO_JOINTS_REGISTRY:
            motion = MotionDataConfig(
                data_format=data_format,
                robot_type=robot,
            )
            assert set(motion.resolved_joints_mapping).issubset(
                motion.resolved_demo_joints,
            )
            assert set(motion.resolved_joints_mapping.values()).issubset(
                body_names,
            )
            assert set(
                motion.resolved_orientation_joints_mapping.values(),
            ).issubset(body_names)


def test_fbx_e1_variants_share_the_waist_roll_torso_mapping() -> None:
    e1_23 = MotionDataConfig(
        data_format="fbx_mocap",
        robot_type="e1_23dof",
    )
    e1_24 = MotionDataConfig(
        data_format="fbx_mocap",
        robot_type="e1_24dof",
    )

    assert e1_23.resolved_joints_mapping["Spine"] == "waist_roll_link"
    assert e1_23.resolved_orientation_joints_mapping["Hips"] == "waist_roll_link"
    assert e1_24.resolved_joints_mapping["Spine"] == "waist_roll_link"
    assert e1_24.resolved_orientation_joints_mapping["Hips"] == "waist_roll_link"


def test_e1_variants_have_independent_natural_pose_profiles() -> None:
    variants: tuple[tuple[RobotName, int], ...] = (("e1_23dof", 23), ("e1_24dof", 24))
    for robot, joint_count in variants:
        profile = load_robot_profile(robot=robot)
        config = internal_config_from_command(
            RetargetingCommand(
                task="robot_only",
                robot=robot,
                dataset="fbx_mocap",
                robot_profile=ROBOT_PROFILE_DIR / f"{robot}.json",
                natural_pose_tracking=True,
            ),
        )

        assert profile.robot == robot
        assert Path(profile.robot_urdf_file).name == f"{robot}.urdf"
        assert len(profile.natural_pose.references) == joint_count
        assert set(profile.natural_pose.references) == set(profile.natural_pose.weights)
        assert config.retargeter.natural_pose_joint_positions == profile.natural_pose.references


def test_e1_24dof_robot_only_command_reaches_the_24dof_mjcf_asset() -> None:
    command = tyro.cli(
        RetargetingCommand,
        args=[
            "--task",
            "robot_only",
            "--robot",
            "e1_24dof",
            "--dataset",
            "lafan",
            "--motion",
            "walk2_subject3",
            "--nature_weights",
            "0.1",
        ],
    )
    config = internal_config_from_command(command)
    normalized = normalize_retargeting_config(config)
    constants = create_task_constants(
        normalized.robot_config,
        normalized.motion_data_config,
        TaskConfig(object_name="ground"),
        "robot_only",
    )

    assert normalized.robot == "e1_24dof"
    assert normalized.robot_config.ROBOT_DOF == 24
    assert Path(constants.ROBOT_URDF_FILE).with_suffix(".xml") == (PACKAGE_ROOT / "models/e1/e1_24dof.xml")
    assert set(config.retargeter.natural_pose_weights.values()) == {0.1}
    assert config.retargeter.natural_pose_joint_positions["waist_roll_joint"] == 0.0
    assert len(config.retargeter.natural_pose_joint_positions) == 24

    retargeter = InteractionMeshRetargeter(
        **build_retargeter_kwargs_from_config(
            normalized.retargeter,
            constants,
            None,
            "robot_only",
        ),
    )
    assert retargeter.robot_model.nq == 31
    assert len(retargeter.robot_actuated_joint_names) == 24
    assert set(retargeter.natural_pose_configured_joint_names) == set(
        retargeter.robot_actuated_joint_names,
    )


def test_e1_23dof_is_available_from_the_public_cli() -> None:
    command = tyro.cli(
        RetargetingCommand,
        args=["--task", "robot_only", "--robot", "e1_23dof", "--dataset", "fbx_mocap"],
    )

    assert command.robot == "e1_23dof"
    config = internal_config_from_command(command)
    assert config.robot == "e1_23dof"
    assert config.robot_config.ROBOT_HEIGHT == 1.5
