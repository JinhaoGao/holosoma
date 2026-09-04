# Holosoma 训练框架

[English](README.md) | [简体中文](README_zh.md)

面向人形机器人强化学习的核心训练框架，支持运动控制（速度跟踪）与全身动作跟踪任务。

有关 Noetix E2 从动作转换、训练、监控、断点续训到策略播放的完整流程，请参阅 [E2 全身动作跟踪训练指南](../../docs/e2-whole-body-tracking-training.md)。

| **类别** | **支持选项** |
|-------------|----------------------|
| **仿真器** | IsaacGym、IsaacSim、MJWarp（训练）\| MuJoCo（评估） |
| **算法** | PPO、FastSAC |
| **机器人** | Unitree G1、Booster T1、Noetix E2（WBT） |

## 训练

所有训练和评估脚本都支持通过 `--help` 查看可用参数，例如 `python src/holosoma/holosoma/train_agent.py --help`。

> **注意：** 使用 `logger:wandb` 时默认开启视频录制。在无头服务器上，可能需要关闭视频或配置渲染环境，具体参见下文的[视频录制](#视频录制)。

### 运动控制（速度跟踪）

训练机器人跟踪速度指令。

```bash
# 在 IsaacGym 中使用 FastSAC 训练 G1
source scripts/source_isaacgym_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-fast-sac \
    simulator:isaacgym \
    logger:wandb \
    --training.seed 1

# 在 IsaacSim 中使用 PPO 训练 T1
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:t1-29dof \
    simulator:isaacsim \
    logger:wandb \
    --training.seed 1
```

保存 checkpoint 后，可以使用[训练内评估](#训练内评估)在与训练相同的仿真器和环境配置中评估策略，也可以在 MuJoCo 中进行跨仿真器评估，具体参见 [holosoma_inference 文档](../holosoma_inference/README.md)。

### 使用 MJWarp 进行运动控制（速度跟踪）训练

使用 MJWarp 仿真器，也就是 GPU 加速的 MuJoCo，执行训练。**注意：MJWarp 支持目前处于测试阶段。**

```bash
# 使用 FastSAC 训练 G1
source scripts/source_mujoco_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-fast-sac \
    simulator:mjwarp \
    logger:wandb

# 使用 PPO 训练 G1
source scripts/source_mujoco_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof \
    simulator:mjwarp \
    logger:wandb

# 使用 FastSAC 训练 T1
source scripts/source_mujoco_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:t1-29dof-fast-sac \
    simulator:mjwarp \
    logger:wandb

# 使用 PPO 训练 T1
source scripts/source_mujoco_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:t1-29dof \
    simulator:mjwarp \
    logger:wandb \
    --terrain.terrain-term.scale-factor=0.5  # 避免训练不稳定所必需

```

> **注意：** MJWarp 默认使用 `nconmax=96`，表示每个环境允许的最大接触数量；如有需要，可以通过 `--simulator.config.mujoco-warp.nconmax-per-env=96` 调整。这些示例使用 `--training.num-envs=4096`，实际使用时可能需要根据硬件资源调整。使用 PPO 在混合地形上训练 T1 时，应设置 `--terrain.terrain-term.scale-factor=0.5` 以避免训练不稳定。

### 全身动作跟踪

训练机器人跟踪全身动作序列。

**注意：** 当前支持在 IsaacSim 中训练 Unitree G1 和 Noetix E2。

```bash
# 使用 FastSAC 训练 G1
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt-fast-sac \
    logger:wandb

# 使用 PPO 训练 G1
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt \
    logger:wandb

# 使用 PPO 训练 Noetix E2 猫爬动作
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:e2-23dof-wbt \
    logger:wandb \
    --training.seed 1

# 使用自定义动作文件
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt \
    logger:wandb \
    --command.setup_terms.motion_command.params.motion_config.motion_file="holosoma/data/motions/g1_29dof/whole_body_tracking/<your file>.npz"

# 训练前在 IsaacSim 中查看动作文件
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/replay.py \
    exp:g1-29dof-wbt \
    --training.headless=False \
    --training.num_envs=1

# 查看项目内置的 Noetix E2 猫爬动作
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/replay.py \
    exp:e2-23dof-wbt \
    --training.headless False \
    --training.num-envs 1
```

#### Noetix E2 训练流程

`exp:e2-23dof-wbt` 预设使用 PPO 和 IsaacSim 跟踪项目内置的 50 Hz E2 猫爬动作。该预设控制 23 个关节，并自动加载 E2 机器人、动作、奖励、终止条件、随机化与课程学习配置。动作回放、训练、TensorBoard、断点续训和策略评估都应使用项目的 `hssim` 环境。如果当前终端处于 `hsretargeting` 环境，应先清除继承的环境名称，再加载 IsaacSim 环境脚本。

```bash
cd /home/jinhaogao/holosoma
conda deactivate 2>/dev/null || true
unset CONDA_ENV_NAME
export CONDA_ENV_NAME=hssim
source scripts/source_isaacsim_setup.sh

which python
pytest -q src/holosoma/tests/test_e2_wbt_config.py
```

开始长时间训练前，应先使用上面的 replay 命令检查参考动作，再运行两轮冒烟训练。在无头服务器上，下面的正式训练命令使用 W&B 离线日志并关闭视频，同时仍会保存 TensorBoard 日志和 checkpoint。

```bash
# 两轮冒烟训练
python src/holosoma/holosoma/train_agent.py \
    exp:e2-23dof-wbt \
    --training.num-envs 16 \
    --algo.config.num-learning-iterations 2

# 正式训练
python src/holosoma/holosoma/train_agent.py \
    exp:e2-23dof-wbt \
    logger:wandb-offline \
    --training.seed 1 \
    --logger.video.enabled False
```

如需训练其他已经转换的 E2 动作，可以在不修改预设的情况下覆盖动作路径。NPZ 必须与项目内置动作保持相同的 23 关节顺序，并遵循 50 Hz 全身动作跟踪数据格式。

```bash
python src/holosoma/holosoma/train_agent.py \
    exp:e2-23dof-wbt \
    logger:wandb-offline \
    --training.seed 1 \
    --logger.video.enabled False \
    --command.setup-terms.motion-command.params.motion-config.motion-file \
    holosoma/data/motions/e2_23dof/whole_body_tracking/<your_e2_motion_fps50.npz>
```

使用 `.pt` checkpoint 可以继续 PPO 训练或播放训练后的策略。续训会恢复策略、critic、优化器和观测归一化状态；`num-learning-iterations` 表示加载 checkpoint 后继续执行的迭代数量。`eval_agent.py` 会自动恢复保存的实验配置，并持续运行到用户主动终止。

```bash
# 断点续训
python src/holosoma/holosoma/train_agent.py \
    exp:e2-23dof-wbt \
    logger:wandb-offline \
    --training.checkpoint /absolute/path/to/model_04000.pt \
    --algo.config.num-learning-iterations 26000 \
    --logger.video.enabled False

# 播放训练后的策略
python src/holosoma/holosoma/eval_agent.py \
    --checkpoint /absolute/path/to/model_04000.pt

# 监控所有 E2 WBT 训练任务
tensorboard --logdir /home/jinhaogao/holosoma/logs/WholeBodyTracking --port 6006
```

有关动作转换、多动作训练、checkpoint 行为、故障排查和经过验证的完整流程，请参阅 [E2 全身动作跟踪训练指南](../../docs/e2-whole-body-tracking-training.md)。

保存 checkpoint 后，可以使用[训练内评估](#训练内评估)在与训练相同的仿真器和环境配置中评估策略，也可以在 MuJoCo 中进行跨仿真器评估，具体参见 [holosoma_inference 文档](../holosoma_inference/README.md)。

---

## 评估

### 训练内评估

使用与训练时完全相同的配置，包括相同的仿真器和环境设置，评估策略：

```bash
# 评估 W&B 中的 checkpoint
python src/holosoma/holosoma/eval_agent.py \
    --checkpoint=wandb://<ENTITY>/<PROJECT>/<RUN_ID>/<CHECKPOINT_NAME>
# 示例：--checkpoint=wandb://username/fastsac-t1-locomotion/abcdefgh/model_0010000.pt

# 评估本地 checkpoint
python src/holosoma/holosoma/eval_agent.py \
    --checkpoint=<CHECKPOINT_PATH>
# 示例：--checkpoint=/home/username/checkpoints/fastsac-t1-locomotion/model_0010000.pt
```

该评估模式会从 checkpoint 自动加载训练配置，在与训练相同的仿真器和环境中执行评估，并可通过 `--training.export_onnx=True` 导出 ONNX 策略。进行运动控制评估时，如果仿真器窗口处于活动状态，可以使用键盘交互式发送速度指令：`w`、`a`、`s`、`d` 控制线速度，`q`、`e` 控制角速度，`z` 将速度清零。

### 跨仿真器评估（MuJoCo）

如需在 MuJoCo 中测试训练后的策略或部署到真实机器人，请参阅 [holosoma_inference 文档](../holosoma_inference/README.md)。该文档介绍从 IsaacGym 或 IsaacSim 到 MuJoCo 的 Sim-to-Sim 评估，以及运动控制和 WBT 策略的真机部署。

**注意：** 训练过程中通常会在保存 `.pt` checkpoint 的同时导出 ONNX 策略，也可以使用上面的训练内评估脚本单独生成 ONNX。

## 高级配置

训练系统使用分层配置。`exp` 配置是主入口，其中包含针对不同算法和机器人的默认参数；可以在命令行中覆盖这些参数。

> **提示：** 组合 Tyro 配置时，应先传入 `exp:<name>` 预设，再传入其他配置片段，例如 `logger:wandb`。Tyro 要求先声明基础实验，颠倒顺序可能产生难以理解的配置解析错误。

### 使用 Weights & Biases 记录日志

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

### 视频录制

使用 `logger:wandb` 时，视频录制**默认开启**。系统会定期录制视频并上传到 Weights & Biases。

**配置方法：**

```bash
# 关闭视频录制
--logger.video.enabled False

# 调整录制间隔，以 episode 为单位
--logger.video.interval 10

# 修改分辨率
--logger.video.width 640 --logger.video.height 360
```

**无头环境故障排查：**

如果训练因显示或渲染错误而失败，例如出现 `GLXBadFBConfig`、`eglInitialize failed` 或 `GLFW initialization failed`，使用 IsaacSim 时可以通过 `--logger.video.enabled False` 关闭视频，也可以使用 `DISPLAY= python ...` 强制 EGL，或通过 `xvfb-run -a python ...` 使用虚拟显示。使用 MJWarp 或 MuJoCo 时，应在训练前设置 `export MUJOCO_GL=egl`，具体参见 [MuJoCo 文档](https://mujoco.readthedocs.io/en/stable/programming/index.html#using-opengl)。IsaacGym 通常可以直接在无头环境中运行；如果仍然出现问题，请关闭视频录制。

### 地形

```bash
# 使用平面地形代替混合地形
source scripts/source_isaacgym_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-fast-sac \
    simulator:isaacgym \
    terrain:terrain-locomotion-plane
```

### 多 GPU 训练

```bash
source scripts/source_isaacgym_setup.sh
torchrun --nproc_per_node=4 src/holosoma/holosoma/train_agent.py \
    exp:t1-29dof-fast-sac \
    simulator:isaacgym \
    --training.num-envs 16384  # 全局环境总数
```

### 自定义奖励权重

```bash
source scripts/source_isaacgym_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-fast-sac \
    simulator:isaacgym \
    --reward.terms.tracking-lin-vel.weight=2.5 \
    --reward.terms.feet-phase.params.swing-height=0.12
```

### 观测噪声

```bash
# 关闭观测噪声
source scripts/source_isaacgym_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-fast-sac \
    simulator:isaacgym \
    --observation.groups.actor-obs.enable-noise=False
```

### 观测历史长度

某些策略可以通过堆叠多个时间步的观测获得更好的效果。训练时可以通过以下参数增加观测历史长度：

```bash
source scripts/source_isaacgym_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-fast-sac \
    simulator:isaacgym \
    --observation.groups.actor_obs.history-length 4
```

运行推理时必须传入相同的历史长度，确保导出的 ONNX 策略接收到正确形状的输入。

### 课程学习

```bash
# 关闭课程学习
source scripts/source_isaacgym_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-fast-sac \
    simulator:isaacgym \
    --curriculum.setup-terms.penalty-curriculum.params.enabled=False

# 自定义课程学习阈值，适用于较短的 episode
source scripts/source_isaacgym_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-fast-sac \
    simulator:isaacgym \
    --simulator.config.sim.max-episode-length-s=10.0 \
    --curriculum.setup-terms.penalty-curriculum.params.level-up-threshold=350
```

### 域随机化

```bash
source scripts/source_isaacgym_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-fast-sac \
    simulator:isaacgym \
    --randomization.setup-terms.push-randomizer-state.params.enabled=False \
    --randomization.setup-terms.randomize-base-com-startup.params.enabled=True \
    --randomization.setup-terms.mass-randomizer.params.added-mass-range=[-1.0,3.0]
```
