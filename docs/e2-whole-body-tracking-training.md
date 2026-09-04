**Noetix E2 Whole-Body Tracking 训练使用说明**

本文说明如何在 Holosoma 中把 E2 重定向结果转换为强化学习动作数据，如何检查动作、启动 Whole-Body Tracking 训练、观察训练状态、停止或恢复训练，以及如何播放参考动作和训练策略。相关训练代码已经合并到 `dev` 分支，工作目录是 `/home/jinhaogao/holosoma`，训练端使用项目自带的 `hssim` 环境，不使用重定向阶段的 `hsretargeting` 环境。

**本次动作和运行结果。** 本次训练动作的原始重定向结果是 `src/holosoma_retargeting/holosoma_retargeting/demo_results/e2/robot_only/noetix_mocap/parkour/crawl_7_cat_crawl_slow_002_Skeleton.npz`。转换后的训练文件是 `src/holosoma/holosoma/data/motions/e2_23dof/whole_body_tracking/crawl_7_cat_crawl_slow_002_fps50.npz`，包含 1122 帧、23 个受控关节和 36 个 E2 刚体，采样频率为 50 Hz。本次正式训练的目录是 `logs/WholeBodyTracking/20260903_084518-e2_23dof_cat_crawl_wbt_manager-locomotion`。训练在约第 7241 轮被人工终止，训练进程已经退出。由于默认保存间隔是 4000 轮，最后一个完整可恢复 checkpoint 是 `model_04000.pt`；第 7241 轮的内存状态没有保存为 checkpoint。

**进入正确的代码和分支。** Git 切换分支时输出的 `M` 表示工作区内存在尚未提交的修改，不是错误，不应使用 `git reset --hard` 或 `git checkout --` 清理这些文件。日常训练应从仓库根目录启动。

```bash
cd /home/jinhaogao/holosoma
git switch dev
git branch --show-current
```

**首次安装训练环境。** 当前机器已经完成这一步，不需要重复安装。新机器首次部署时，可以在仓库根目录按照项目脚本创建 `hssim`。环境实际安装在 `/home/jinhaogao/.holosoma_deps/miniconda3/envs/hssim`，本次验证使用 Isaac Sim 5.1.0、IsaacLab v2.3.0、Python 3.11、PyTorch 2.7.0+cu128，并已在 RTX 4090 上确认 CUDA 可用。

```bash
cd /home/jinhaogao/holosoma
CONDA_ENV_NAME=hssim OMNI_KIT_ACCEPT_EULA=1 bash scripts/setup_isaacsim.sh
```

**每次训练前激活环境。** `source_isaacsim_setup.sh` 会沿用当前 shell 中已经存在的 `CONDA_ENV_NAME`。如果终端原来显示 `(hsretargeting)`，直接执行该脚本会错误地再次激活 `hsretargeting`。因此建议显式清除旧值并指定 `hssim`。激活完成后，提示符应显示 `(hssim)`，`which python` 应指向 `hssim/bin/python`，`tensorboard` 也应能被找到。

```bash
cd /home/jinhaogao/holosoma
conda deactivate 2>/dev/null || true
unset CONDA_ENV_NAME
export CONDA_ENV_NAME=hssim
source scripts/source_isaacsim_setup.sh

which python
which tensorboard
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

期望的 Python 路径是 `/home/jinhaogao/.holosoma_deps/miniconda3/envs/hssim/bin/python`。如果脚本打印 `conda environment name is set to: hsretargeting`，说明环境变量没有清理，应停止当前操作并重新执行上面的环境切换命令。

**E2 训练配置的位置。** 机器人关节、刚体、PD 参数、关节限位和资产入口定义在 `src/holosoma/holosoma/config_values/robot.py` 的 `e2_23dof` 配置中。E2 的 WBT 配置位于 `src/holosoma/holosoma/config_values/wbt/e2/`。其中 `experiment.py` 定义 PPO、Isaac Sim、环境数量和训练轮数，`command.py` 定义动作文件、参考刚体和自适应时间采样，`reward.py` 定义跟踪奖励和不期望接触惩罚，`termination.py` 定义跟踪失败终止条件。E2 URDF、MJCF 和网格资产位于 `src/holosoma/holosoma/data/robots/e2/`。四足爬行动作需要手掌接触地面，因此训练侧 URDF 为左右 `hand_sphere_link` 添加了半径 0.04 米的碰撞球；重定向侧原始模型没有被修改。

**时间频率和策略维度。** Isaac Sim 物理步长为 0.005 秒，即 200 Hz，控制 decimation 为 4，因此策略控制频率为 50 Hz，与转换后动作的 50 FPS 一致。当前策略输出 23 个 E2 关节位置目标，actor 观测维度为 124，critic 观测维度为 274。跟踪集合包含 base、左右髋部、膝部、踝部、脚端、腰部、左右肩部、肘部和手端，共 16 个关键刚体。手部、踝部和脚部接触被视为允许接触，其他不期望接触会产生惩罚。

**训练动作文件的数据约定。** RL 动作 NPZ 必须包含 `fps`、`joint_pos`、`joint_vel`、`body_pos_w`、`body_quat_w`、`body_lin_vel_w`、`body_ang_vel_w`、`body_names` 和 `joint_names`。对于 E2，`joint_pos` 的第二维应为 30，也就是自由根节点的 7 个坐标加 23 个关节；`joint_vel` 的第二维应为 29，也就是自由根节点的 6 个速度加 23 个关节速度。`joint_names` 必须与 E2 配置中的 23 个关节顺序完全一致，数组不能包含 NaN 或 Inf。当前仓库内的专项测试会验证这些条件。

```bash
cd /home/jinhaogao/holosoma
export CONDA_ENV_NAME=hssim
source scripts/source_isaacsim_setup.sh
pytest -q src/holosoma/tests/test_e2_wbt_config.py
```

**把新的 E2 重定向结果转换为 RL 数据。** 只有在加入新动作时才需要执行转换。转换阶段使用 `hsretargeting`，训练阶段再切回 `hssim`。下面的命令以本次 cat-crawl 为完整示例。转换器会通过 MuJoCo 计算重采样后的关节速度、刚体位姿和刚体速度；`xvfb-run` 使转换器在没有可见桌面的终端中也能创建其 MuJoCo viewer。只有当输出 NPZ 已生成并通过后续验证时，才能把转换视为成功。

```bash
cd /home/jinhaogao/holosoma
conda deactivate 2>/dev/null || true
unset CONDA_ENV_NAME
export CONDA_ENV_NAME=hsretargeting
source scripts/source_retargeting_setup.sh

xvfb-run -a python -m holosoma_retargeting.data_conversion.convert_data_format_mj \
  --input-file /home/jinhaogao/holosoma/src/holosoma_retargeting/holosoma_retargeting/demo_results/e2/robot_only/noetix_mocap/parkour/crawl_7_cat_crawl_slow_002_Skeleton.npz \
  --robot e2 \
  --data-format noetix_mocap \
  --output-fps 50 \
  --output-name /home/jinhaogao/holosoma/src/holosoma/holosoma/data/motions/e2_23dof/whole_body_tracking/crawl_7_cat_crawl_slow_002_fps50.npz \
  --once
```

转换后可以用下面的脚本检查基本结构。对于其他动作，只需要修改 `motion_path`。完整一致性仍应由 E2 专项测试和 Isaac Sim 冒烟训练确认。

```bash
conda deactivate 2>/dev/null || true
unset CONDA_ENV_NAME
export CONDA_ENV_NAME=hssim
source scripts/source_isaacsim_setup.sh

python - <<'PY'
from pathlib import Path
import numpy as np

motion_path = Path("src/holosoma/holosoma/data/motions/e2_23dof/whole_body_tracking/crawl_7_cat_crawl_slow_002_fps50.npz")
required = {
    "fps", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
    "body_lin_vel_w", "body_ang_vel_w", "body_names", "joint_names",
}
with np.load(motion_path) as motion:
    assert required.issubset(motion.files)
    assert float(motion["fps"].item()) == 50.0
    assert motion["joint_pos"].shape[1] == 30
    assert motion["joint_vel"].shape[1] == 29
    for key in required - {"body_names", "joint_names"}:
        assert np.isfinite(motion[key]).all(), key
    print("frames:", motion["joint_pos"].shape[0])
    print("joints:", len(motion["joint_names"]))
    print("bodies:", len(motion["body_names"]))
PY
```

**播放参考动作。** 在正式训练前，应先用 Isaac Sim 直接播放动作，确认朝向、根节点高度、手脚落点、关节顺序和播放速度合理。该命令不加载策略，只把机器人设置到参考轨迹。命令需要有效的图形显示；纯 SSH 环境中应使用远程桌面、X11 转发或本机可访问的显示服务。`xvfb-run` 可以完成无头验证，但不会提供可交互窗口。

```bash
cd /home/jinhaogao/holosoma
export CONDA_ENV_NAME=hssim
source scripts/source_isaacsim_setup.sh

python src/holosoma/holosoma/replay.py \
  exp:e2-23dof-wbt \
  --training.headless False \
  --training.num-envs 1
```

**运行短程冒烟训练。** 每次更换机器人资产、动作结构、跟踪刚体或奖励终止配置后，先运行小规模训练。下面的命令只创建 16 个环境并执行 2 个 PPO 迭代，目的是验证 Isaac Sim 能导入 E2、关节和刚体名称能解析、接触传感器能创建、动作能加载且 PPO 可以完成前向、反向和 checkpoint 保存。它不是正式训练结果。

```bash
python src/holosoma/holosoma/train_agent.py \
  exp:e2-23dof-wbt \
  --training.num-envs 16 \
  --algo.config.num-learning-iterations 2
```

**启动正式训练。** 当前 E2 预设保持项目 G1 WBT PPO 的主要训练参数，使用 4096 个并行环境、30000 个学习迭代、每次更新 5 个 learning epochs、24 个环境步、初始策略噪声 1.0、actor 与 critic 初始学习率均为 `1e-3`、熵系数 0.005、经验观测归一化和自适应时间片采样。每轮采样量是 `4096 × 24 = 98304` 个环境步。默认每 4000 轮保存一次 `.pt` checkpoint，并同时导出 `.onnx`。

没有配置 W&B 账号时，推荐使用离线日志。TensorBoard 无论是否启用 W&B 都会写入。关闭视频可以避免无头训练额外创建渲染管线，且不会改变 PPO、环境或奖励参数。

```bash
python src/holosoma/holosoma/train_agent.py \
  exp:e2-23dof-wbt \
  logger:wandb-offline \
  --training.seed 1 \
  --logger.video.enabled False
```

已经完成 `wandb login` 并希望在线同步时，可以把 `logger:wandb-offline` 改成 `logger:wandb`。无论在线还是离线，都建议继续关闭无头视频，除非已经验证服务器的 EGL 或显示配置。

```bash
python src/holosoma/holosoma/train_agent.py \
  exp:e2-23dof-wbt \
  logger:wandb \
  --training.seed 1 \
  --logger.video.enabled False
```

完全不使用 W&B 时，可以省略 logger 子命令。此时仍会保存 TensorBoard event、`train.log`、`holosoma_config.yaml`、checkpoint 和 ONNX。

```bash
python src/holosoma/holosoma/train_agent.py \
  exp:e2-23dof-wbt \
  --training.seed 1
```

**在 tmux 中保持训练。** 长时间训练建议使用 `tmux`，这样 SSH 断开不会终止 Isaac Sim。先创建会话，在会话内激活 `hssim` 并执行正式训练命令。按 `Ctrl+B` 后再按 `D` 可以脱离会话，使用 `tmux attach -t e2-wbt` 可以重新进入。不要在没有确认 checkpoint 的情况下直接杀死 tmux server。

```bash
tmux new -s e2-wbt
```

**训练其他单个 E2 动作。** 最安全的方法是在命令行覆盖动作文件，这样不需要修改默认预设。路径可以使用 `holosoma` 包内的相对数据路径。Tyro 参数名使用连字符，不要把 `setup-terms` 写成 `setup_terms`。

```bash
python src/holosoma/holosoma/train_agent.py \
  exp:e2-23dof-wbt \
  logger:wandb-offline \
  --training.seed 1 \
  --logger.video.enabled False \
  --command.setup-terms.motion-command.params.motion-config.motion-file \
  holosoma/data/motions/e2_23dof/whole_body_tracking/<new_e2_motion_fps50.npz>
```

如果希望把新动作固定为该分支的默认动作，可以修改 `src/holosoma/holosoma/config_values/wbt/e2/command.py` 中的 `E2_CRAWL_MOTION_FILE`。修改后应重新运行 E2 专项测试、参考动作回放和 2 轮冒烟训练。

**同时训练多个 E2 动作。** `MotionConfig.motion_dir` 非空时优先于单个 `motion_file`。建议为兼容的 E2 训练动作建立单独目录，避免把临时文件或其他机器人的 NPZ 混入。该目录内所有动作都必须遵守同一 E2 关节顺序和刚体名称约定。

```bash
python src/holosoma/holosoma/train_agent.py \
  exp:e2-23dof-wbt \
  logger:wandb-offline \
  --training.seed 1 \
  --logger.video.enabled False \
  --command.setup-terms.motion-command.params.motion-config.motion-dir \
  holosoma/data/motions/e2_23dof/whole_body_tracking/<motion_set_directory>
```

**观察训练状态。** 每次训练会创建 `logs/WholeBodyTracking/<UTC时间>-e2_23dof_cat_crawl_wbt_manager-locomotion`。目录内的 `holosoma_config.yaml` 是该次运行最终解析后的完整配置，应以它为准确认动作路径、环境数、seed 和 PPO 参数。`events.out.tfevents.*` 用于 TensorBoard，`train.log` 保存普通日志，`model_*.pt` 用于评估和续训，`model_*.onnx` 用于推理部署。训练终端中的 `Learning iteration N/30000`、`Computation`、`Mean reward`、`Mean episode length`、各类 motion error、KL 和噪声标准差用于判断训练是否正常。

```bash
tensorboard \
  --logdir /home/jinhaogao/holosoma/logs/WholeBodyTracking \
  --port 6006
```

在远程服务器上，TensorBoard 默认只监听服务器本机。可以通过 IDE 端口转发，或者从本地电脑建立 `ssh -L 6006:localhost:6006 <user>@<server>` 隧道，然后在本地浏览器访问 `http://localhost:6006`。GPU 和进程状态可以使用下面的命令检查。

```bash
nvidia-smi
pgrep -af '^python src/holosoma/holosoma/train_agent.py exp:e2-23dof-wbt'
```

**安全停止训练。** 如果训练在当前前台终端运行，使用一次 `Ctrl+C`。如果训练在 tmux 中运行，先重新进入 tmux，再使用一次 `Ctrl+C`。如果只能从另一个终端停止，应先用精确的 `pgrep` 命令确认 PID，再向训练 PID 发送 `SIGINT`。不要使用宽泛的 `pkill python`，因为它会同时终止动作回放、策略 Play、TensorBoard 或其他用户任务。

```bash
training_pid=$(pgrep -f '^python src/holosoma/holosoma/train_agent.py exp:e2-23dof-wbt' | head -1)
printf 'training PID: %s\n' "$training_pid"
kill -INT "$training_pid"
```

当前 PPO 训练循环收到 `Ctrl+C` 时不会额外保存一个“中断时 checkpoint”，因此停止前应关注最近一次定期保存。默认 `save_interval=4000` 时，即使终端已经显示第 7241 轮，仍只有 `model_04000.pt` 可恢复。需要更密集保存时，可在新训练命令中加入 `--algo.config.save-interval 500`，代价是更频繁的磁盘写入和更多 checkpoint 文件。

**从 checkpoint 继续训练。** `.pt` checkpoint 会恢复 actor、critic、观测归一化器、优化器和当前学习迭代。项目 PPO 实现把 `num_learning_iterations` 解释为“从加载点开始再执行的循环次数”，不是绝对终止轮数。本次 `model_04000.pt` 内记录的迭代号是 4000。希望训练编号继续运行到约 30000 时，应把额外轮数设为 26000；加载后的第一轮仍会显示 4000，这是当前 PPO 循环的既有行为。续训会创建新的日志目录，而不会写回原运行目录。

```bash
python src/holosoma/holosoma/train_agent.py \
  exp:e2-23dof-wbt \
  logger:wandb-offline \
  --training.seed 1 \
  --training.checkpoint \
  /home/jinhaogao/holosoma/logs/WholeBodyTracking/20260903_084518-e2_23dof_cat_crawl_wbt_manager-locomotion/model_04000.pt \
  --algo.config.num-learning-iterations 26000 \
  --logger.video.enabled False
```

如果目标是从 checkpoint 再训练完整的 30000 轮，则保留预设中的 `num_learning_iterations=30000`，但最终迭代编号会超过原先的 30000。续训时不要更换机器人自由度、策略观测结构或动作输出结构，否则 checkpoint 权重尺寸将无法匹配。调整奖励权重、随机化强度或动作文件虽然通常可以加载权重，但这会改变优化目标，应创建清晰的新实验名称并重新做冒烟验证。

**播放训练策略。** `eval_agent.py` 会从 `.pt` 内自动恢复训练配置，默认创建 1 个非 headless 环境，并默认导出 ONNX。下面播放本次最后一个完整 checkpoint。评估默认持续运行，使用 `Ctrl+C` 结束。若只想运行有限步数，可以追加 `--training.max-eval-steps 5000`。

```bash
python src/holosoma/holosoma/eval_agent.py \
  --checkpoint \
  /home/jinhaogao/holosoma/logs/WholeBodyTracking/20260903_084518-e2_23dof_cat_crawl_wbt_manager-locomotion/model_04000.pt
```

训练目录中的 `model_04000.onnx` 和 `exported/model_04000.onnx` 是推理格式，不能用于恢复 PPO 优化器；恢复训练必须使用 `.pt`。如果评估没有弹出窗口，先执行 `echo $DISPLAY`。输出为空通常说明当前 SSH 会话没有图形转发。此时应使用远程桌面、X11 转发或 IDE 的图形能力，而不是把训练端重新改成错误的 Conda 环境。

**常见故障判断。** 出现 `tensorboard: command not found`，并且脚本打印 `conda environment name is set to: hsretargeting`，原因是训练环境没有切到 `hssim`，应重新执行本文的环境激活段落。出现 CUDA 不可用时，应先确认 `which python` 和 `torch.__version__`，再检查 `nvidia-smi`，不要直接在 `hsretargeting` 中重新安装训练依赖。出现动作文件找不到时，应检查 `holosoma_config.yaml` 中最终保存的 `motion_file` 或 `motion_dir`。出现关节名、刚体名或张量形状错误时，应回到转换结果，运行 `test_e2_wbt_config.py` 并确认新 NPZ 使用 E2 的 23 关节顺序。出现训练很快终止、episode length 长期很低或手掌无法支撑时，应优先播放参考动作并检查根节点高度、手脚碰撞、手脚轨迹和 50 Hz 时间对齐，而不是立即增大奖励权重。

**推荐执行顺序。** 对每个新 E2 动作，先在 `hsretargeting` 中完成重定向和 50 FPS RL 格式转换，再切换到 `hssim` 验证 NPZ，随后播放参考动作、运行 16 环境 2 轮冒烟训练，最后才启动 4096 环境正式训练。正式训练开始后，应确认落盘的 `holosoma_config.yaml` 与预期一致、TensorBoard step 持续增长、GPU 进程存在且定期 checkpoint 正常生成，再让任务长期运行。
