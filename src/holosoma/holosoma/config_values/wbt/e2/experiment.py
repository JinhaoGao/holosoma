"""PPO Whole Body Tracking experiment for the Noetix E2 robot."""

from dataclasses import replace

from holosoma.config_types.experiment import ExperimentConfig, NightlyConfig, TrainingConfig
from holosoma.config_values import algo, robot, simulator, terrain
from holosoma.config_values.wbt.e2.action import e2_23dof_joint_pos
from holosoma.config_values.wbt.e2.command import e2_23dof_wbt_command
from holosoma.config_values.wbt.e2.curriculum import e2_23dof_wbt_curriculum
from holosoma.config_values.wbt.e2.observation import e2_23dof_wbt_observation
from holosoma.config_values.wbt.e2.randomization import e2_23dof_wbt_randomization
from holosoma.config_values.wbt.e2.reward import e2_23dof_wbt_reward
from holosoma.config_values.wbt.e2.termination import e2_23dof_wbt_termination

e2_23dof_wbt = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="e2_23dof_cat_crawl_wbt_manager",
        num_envs=4096,
    ),
    env_class="holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager",
    algo=replace(
        algo.ppo,
        config=replace(
            algo.ppo.config,
            num_learning_iterations=30000,
            num_learning_epochs=5,
            save_interval=4000,
            entropy_coef=0.005,
            init_noise_std=1.0,
            actor_learning_rate=1e-3,
            critic_learning_rate=1e-3,
            init_at_random_ep_len=True,
            empirical_normalization=True,
            use_symmetry=False,
            actor_optimizer=replace(algo.ppo.config.actor_optimizer, weight_decay=0.0),
            critic_optimizer=replace(algo.ppo.config.critic_optimizer, weight_decay=0.0),
        ),
    ),
    simulator=replace(
        simulator.isaacsim,
        config=replace(
            simulator.isaacsim.config,
            sim=replace(simulator.isaacsim.config.sim, max_episode_length_s=10.0),
        ),
    ),
    robot=robot.e2_23dof,
    terrain=terrain.terrain_locomotion_plane,
    observation=e2_23dof_wbt_observation,
    action=e2_23dof_joint_pos,
    termination=e2_23dof_wbt_termination,
    randomization=e2_23dof_wbt_randomization,
    command=e2_23dof_wbt_command,
    curriculum=e2_23dof_wbt_curriculum,
    reward=e2_23dof_wbt_reward,
    nightly=NightlyConfig(
        iterations=8000,
        metrics={
            "Episode/rew_motion_global_ref_position_error_exp": [0.3, "inf"],
            "Episode/rew_motion_global_ref_orientation_error_exp": [0.4, "inf"],
            "Episode/rew_motion_relative_body_position_error_exp": [0.85, "inf"],
            "Episode/rew_motion_relative_body_orientation_error_exp": [0.7, "inf"],
            "Episode/rew_motion_global_body_lin_vel": [0.6, "inf"],
            "Episode/rew_motion_global_body_ang_vel": [0.45, "inf"],
        },
    ),
)

__all__ = ["e2_23dof_wbt"]
