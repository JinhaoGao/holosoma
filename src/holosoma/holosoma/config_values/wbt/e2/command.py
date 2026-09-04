"""Whole Body Tracking command preset for the Noetix E2 robot."""

from holosoma.config_types.command import CommandManagerCfg, CommandTermCfg, MotionConfig, NoiseToInitialPoseConfig

E2_CRAWL_MOTION_FILE = "holosoma/data/motions/e2_23dof/whole_body_tracking/crawl_7_cat_crawl_slow_002_fps50.npz"

E2_TRACKED_BODY_NAMES = [
    "base_link",
    "l_leg_hip_roll_link",
    "l_leg_knee_link",
    "l_leg_ankle_roll_link",
    "l_foot_sphere_5_link",
    "r_leg_hip_roll_link",
    "r_leg_knee_link",
    "r_leg_ankle_roll_link",
    "r_foot_sphere_5_link",
    "waist_pitch_link",
    "l_arm_shoulder_roll_link",
    "l_arm_elbow_link",
    "l_hand_sphere_link",
    "r_arm_shoulder_roll_link",
    "r_arm_elbow_link",
    "r_hand_sphere_link",
]

e2_init_pose_noise = NoiseToInitialPoseConfig(
    overall_noise_scale=1.0,
    dof_pos=0.1,
    root_pos=[0.05, 0.05, 0.01],
    root_rot=[0.1, 0.1, 0.2],
    root_lin_vel=[0.5, 0.5, 0.2],
    root_ang_vel=[0.52, 0.52, 0.78],
    object_pos=[0.05, 0.05, 0.0],
)

e2_motion_config = MotionConfig(
    motion_file=E2_CRAWL_MOTION_FILE,
    body_names_to_track=E2_TRACKED_BODY_NAMES,
    body_name_ref=["waist_pitch_link"],
    use_adaptive_timesteps_sampler=True,
    noise_to_initial_pose=e2_init_pose_noise,
)

e2_23dof_wbt_command = CommandManagerCfg(
    params={},
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
            params={"motion_config": e2_motion_config},
        ),
    },
    reset_terms={"motion_command": CommandTermCfg(func="holosoma.managers.command.terms.wbt:MotionCommand")},
    step_terms={"motion_command": CommandTermCfg(func="holosoma.managers.command.terms.wbt:MotionCommand")},
)

__all__ = ["E2_CRAWL_MOTION_FILE", "E2_TRACKED_BODY_NAMES", "e2_23dof_wbt_command"]
