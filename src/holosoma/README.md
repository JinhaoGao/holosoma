# Holosoma Training Framework

Core training framework for humanoid robot reinforcement learning with support for locomotion (velocity tracking) and whole-body tracking tasks.

For the end-to-end Noetix E2 motion conversion, training, monitoring, resume, and playback workflow, see the [E2 Whole-Body Tracking training guide](../../docs/e2-whole-body-tracking-training.md).

| **Category** | **Supported Options** |
|-------------|----------------------|
| **Simulators** | IsaacGym, IsaacSim, MJWarp (training) \| Mujoco (evaluation) |
| **Algorithms** | PPO, FastSAC |
| **Robots** | Unitree G1, Booster T1, Noetix E2 (WBT) |

## Training

All training/eval scripts support `--help` for discovering available flags, e.g. `python src/holosoma/holosoma/train_agent.py --help`.

> **Note:** Video recording is enabled by default with `logger:wandb`. On headless servers, you may need to disable video or configure rendering. See [Video Recording](#video-recording) below.

### Locomotion (Velocity Tracking)

Train robots to track velocity commands.

```bash
# G1 with FastSAC on IsaacGym
source scripts/source_isaacgym_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-fast-sac \
    simulator:isaacgym \
    logger:wandb \
    --training.seed 1

# T1 with PPO on IsaacSim
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:t1-29dof \
    simulator:isaacsim \
    logger:wandb \
    --training.seed 1
```

Once checkpoints are saved, you can evaluate policies using [In-Training Evaluation](#in-training-evaluation) (same simulator as training) or cross-simulator evaluation in MuJoCo (see [holosoma_inference](../holosoma_inference/README.md)).

### MJWarp Training for Locomotion (Velocity Tracking)

Train using the MJWarp simulator (GPU-accelerated MuJoCo). **Note: MJWarp support is in beta.**

```bash
# G1 with FastSAC
source scripts/source_mujoco_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-fast-sac \
    simulator:mjwarp \
    logger:wandb

# G1 with PPO
source scripts/source_mujoco_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof \
    simulator:mjwarp \
    logger:wandb

# T1 with FastSAC
source scripts/source_mujoco_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:t1-29dof-fast-sac \
    simulator:mjwarp \
    logger:wandb

# T1 with PPO
source scripts/source_mujoco_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:t1-29dof \
    simulator:mjwarp \
    logger:wandb \
    --terrain.terrain-term.scale-factor=0.5  # required to avoid training instabilities

```

> **Note:**
> - MJWarp uses `nconmax=96` (maximum contacts per environment) by default. This can be adjusted via `--simulator.config.mujoco-warp.nconmax-per-env=96` if needed.
> - These examples use `--training.num-envs=4096`, but you may need to adjust this value based on your hardware.
> - When training T1 with PPO on mixed terrain, use `--terrain.terrain-term.scale-factor=0.5` to avoid training instabilities.


### Whole-Body Tracking

Train robots to track full-body motion sequences.

**Note**: Currently supported for Unitree G1 and Noetix E2 with IsaacSim.

```bash
# G1 with FastSAC
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt-fast-sac \
    logger:wandb

# G1 with PPO
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt \
    logger:wandb

# Noetix E2 cat-crawl with PPO
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:e2-23dof-wbt \
    logger:wandb \
    --training.seed 1

# Custom motion file
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt \
    logger:wandb \
    --command.setup_terms.motion_command.params.motion_config.motion_file="holosoma/data/motions/g1_29dof/whole_body_tracking/<your file>.npz"

# Visualize the motion file in isaacsim before training
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/replay.py \
    exp:g1-29dof-wbt \
    --training.headless=False \
    --training.num_envs=1

# Visualize the bundled Noetix E2 cat-crawl motion
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/replay.py \
    exp:e2-23dof-wbt \
    --training.headless False \
    --training.num-envs 1
```

#### Noetix E2 Training Workflow

The `exp:e2-23dof-wbt` preset uses PPO and IsaacSim to track the bundled 50 Hz E2 cat-crawl motion. It controls 23 joints and loads the E2 robot, motion, reward, termination, randomization, and curriculum settings automatically. Use the project `hssim` environment for replay, training, TensorBoard, resume, and policy evaluation. If the current shell is in `hsretargeting`, clear the inherited environment name before sourcing the IsaacSim setup script.

```bash
cd /home/jinhaogao/holosoma
conda deactivate 2>/dev/null || true
unset CONDA_ENV_NAME
export CONDA_ENV_NAME=hssim
source scripts/source_isaacsim_setup.sh

which python
pytest -q src/holosoma/tests/test_e2_wbt_config.py
```

Before a long run, use the replay command above to inspect the reference motion, then run a two-iteration smoke test. On a headless server, the following formal-training command uses offline W&B logging and disables video while retaining TensorBoard logs and checkpoints.

```bash
# Two-iteration smoke test
python src/holosoma/holosoma/train_agent.py \
    exp:e2-23dof-wbt \
    --training.num-envs 16 \
    --algo.config.num-learning-iterations 2

# Formal training
python src/holosoma/holosoma/train_agent.py \
    exp:e2-23dof-wbt \
    logger:wandb-offline \
    --training.seed 1 \
    --logger.video.enabled False
```

To train another converted E2 motion without changing the preset, override the motion path. The NPZ must follow the same 23-joint ordering and 50 Hz Whole-Body Tracking format as the bundled motion.

```bash
python src/holosoma/holosoma/train_agent.py \
    exp:e2-23dof-wbt \
    logger:wandb-offline \
    --training.seed 1 \
    --logger.video.enabled False \
    --command.setup-terms.motion-command.params.motion-config.motion-file \
    holosoma/data/motions/e2_23dof/whole_body_tracking/<your_e2_motion_fps50.npz>
```

Use a `.pt` checkpoint to resume PPO training or play the trained policy. Resuming restores the policy, critic, optimizers, and observation normalization; `num-learning-iterations` is the number of additional iterations to run after loading. `eval_agent.py` restores the saved experiment configuration automatically and keeps running until interrupted.

```bash
# Resume training
python src/holosoma/holosoma/train_agent.py \
    exp:e2-23dof-wbt \
    logger:wandb-offline \
    --training.checkpoint /absolute/path/to/model_04000.pt \
    --algo.config.num-learning-iterations 26000 \
    --logger.video.enabled False

# Play a trained policy
python src/holosoma/holosoma/eval_agent.py \
    --checkpoint /absolute/path/to/model_04000.pt

# Monitor all E2 WBT runs
tensorboard --logdir /home/jinhaogao/holosoma/logs/WholeBodyTracking --port 6006
```

For motion conversion, multi-motion training, checkpoint behavior, troubleshooting, and the complete tested workflow, see the [E2 Whole-Body Tracking training guide](../../docs/e2-whole-body-tracking-training.md).

Once checkpoints are saved, you can evaluate policies using [In-Training Evaluation](#in-training-evaluation) (same simulator as training) or cross-simulator evaluation in MuJoCo (see [holosoma_inference](../holosoma_inference/README.md)).

---

## Evaluation

### In-Training Evaluation

For evaluating policies with the exact same configuration used during training (same simulator, environment settings, etc.):

```bash
# Evaluate checkpoint from Wandb
python src/holosoma/holosoma/eval_agent.py \
    --checkpoint=wandb://<ENTITY>/<PROJECT>/<RUN_ID>/<CHECKPOINT_NAME>
# e.g., --checkpoint=wandb://username/fastsac-t1-locomotion/abcdefgh/model_0010000.pt

# Evaluate local checkpoint
python src/holosoma/holosoma/eval_agent.py \
    --checkpoint=<CHECKPOINT_PATH>
# e.g., --checkpoint=/home/username/checkpoints/fastsac-t1-locomotion/model_0010000.pt
```

This evaluation mode:
- Automatically loads the training configuration from the checkpoint
- Runs evaluation in the same simulator and environment as training
- Can export policies to ONNX format (via `--training.export_onnx=True`)
- For locomotion evaluation, supports interactive velocity commands via keyboard (when simulator window is active):
  - `w`/`a`/`s`/`d`: linear velocity commands
  - `q`/`e`: angular velocity commands
  - `z`: zero velocity command

### Cross-Simulator Evaluation (MuJoCo)

For testing trained policies in MuJoCo simulation or deploying to real robots, see the [holosoma_inference documentation](../holosoma_inference/README.md). This covers:
- Sim-to-sim evaluation (IsaacGym/IsaacSim → MuJoCo)
- Real robot deployment (both locomotion and WBT)

**Note**: ONNX policies are typically exported alongside `.pt` checkpoints during training, but can also be generated using the in-training evaluation script above.

## Advanced Configuration

The training system uses a hierarchical configuration system. The `exp` config serves as the main entry point with default configurations tuned for each algorithm and robot. You can customize training by overriding parameters on the command line.

> **Tip**: When composing Tyro configs, pass the `exp:<name>` preset before any other config fragments (e.g., `logger:wandb`). Tyro expects the base experiment to be declared first, and reversing the order can lead to confusing resolution errors.

### Logging with Weights & Biases

```bash
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof \
    simulator:isaacsim \
    --training.seed 1 \
    --algo.config.use-symmetry=False \
    logger:wandb \
    --logger.project locomotion-g1-29dof-ppo \
    --logger.name ppo-without-symmetry-seed1
```

### Video Recording

Video recording is **enabled by default** when using `logger:wandb`. Videos are recorded periodically and uploaded to Weights & Biases.

**Configuration:**
```bash
# Disable video recording
--logger.video.enabled False

# Adjust recording interval (episodes)
--logger.video.interval 10

# Change resolution
--logger.video.width 640 --logger.video.height 360
```

**Troubleshooting Headless Environments:**

If training fails on headless servers with display/rendering errors (e.g., `GLXBadFBConfig`, `eglInitialize failed`, `GLFW initialization failed`):

- **IsaacSim:** Disable video with `--logger.video.enabled False`, or force EGL with `DISPLAY= python ...`, or use virtual display with `xvfb-run -a python ...`
- **MJWarp/MuJoCo:** Set environment variable before training: `export MUJOCO_GL=egl`. See [MuJoCo docs](https://mujoco.readthedocs.io/en/stable/programming/index.html#using-opengl)
- **IsaacGym:** Usually works in headless environments. If issues occur, disable video with `--logger.video.enabled False`

### Terrain

```bash
# Use plane terrain instead of mixed terrain
source scripts/source_isaacgym_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-fast-sac \
    simulator:isaacgym \
    terrain:terrain-locomotion-plane
```

### Multi-GPU Training

```bash
source scripts/source_isaacgym_setup.sh
torchrun --nproc_per_node=4 src/holosoma/holosoma/train_agent.py \
    exp:t1-29dof-fast-sac \
    simulator:isaacgym \
    --training.num-envs 16384  # global/total number of environments
```

### Custom Reward Weights

```bash
source scripts/source_isaacgym_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-fast-sac \
    simulator:isaacgym \
    --reward.terms.tracking-lin-vel.weight=2.5 \
    --reward.terms.feet-phase.params.swing-height=0.12
```

### Observation Noise

```bash
# Disable observation noise
source scripts/source_isaacgym_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-fast-sac \
    simulator:isaacgym \
    --observation.groups.actor-obs.enable-noise=False
```

### Observation History Length

Some policies benefit from stacking multiple timesteps of observations. You can increase the history length used during training with:

```bash
source scripts/source_isaacgym_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-fast-sac \
    simulator:isaacgym \
    --observation.groups.actor_obs.history-length 4
```

Make sure to pass the same history length when running inference so the exported ONNX policy receives inputs with the correct shape.

### Curriculum Learning

```bash
# Disable curriculum
source scripts/source_isaacgym_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-fast-sac \
    simulator:isaacgym \
    --curriculum.setup-terms.penalty-curriculum.params.enabled=False

# Custom curriculum threshold (for shorter episodes)
source scripts/source_isaacgym_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-fast-sac \
    simulator:isaacgym \
    --simulator.config.sim.max-episode-length-s=10.0 \
    --curriculum.setup-terms.penalty-curriculum.params.level-up-threshold=350
```

### Domain Randomization

```bash
source scripts/source_isaacgym_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-fast-sac \
    simulator:isaacgym \
    --randomization.setup-terms.push-randomizer-state.params.enabled=False \
    --randomization.setup-terms.randomize-base-com-startup.params.enabled=True \
    --randomization.setup-terms.mass-randomizer.params.added-mass-range=[-1.0,3.0]
```
