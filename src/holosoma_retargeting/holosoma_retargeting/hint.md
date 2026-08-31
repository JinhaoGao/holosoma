# 重定向与可视化速查

以下命令均从重定向包目录执行：

```bash
conda activate robot_retargeter
cd /home/jinhaogao/holosoma/src/holosoma_retargeting/holosoma_retargeting
export PYTHONPATH=..
```

重定向默认启动 Viser 实时可视化，并在终端打印访问地址。机器人和
object 会随求解逐帧更新；`--retargeter.debug` 额外显示人体与机器人
映射关键点、骨架、手关键点和 object 点云，并在求解结束后保留页面，
按 Enter 才退出。无图形或批处理环境使用 `--retargeter.no-visualize`。
`--retargeter.visualize` 仍然兼容，但由于可视化已经默认开启，通常无需填写。

## 单任务重定向

`robot_retarget.py` 每次处理一个动作。`robot_only` 支持 G1、E1、E2，以及 `climbing`、`gvhmr`、`lafan`、`noetix_csv_climb`、`noetix_mocap`、`OMOMO_new` 六类人体数据。

```bash
python examples/robot_retarget.py \
  --task robot_only \
  --robot e2 \
  --dataset lafan \
  --motion walk2_subject3
```

G1 的 object-interaction 示例：

```bash
python examples/robot_retarget.py \
  --task object_interaction \
  --robot g1 \
  --dataset OMOMO_new \
  --motion sub3_largebox_003
```

G1 的 climbing 示例：

```bash
python examples/robot_retarget.py \
  --task climbing \
  --robot g1 \
  --dataset climbing \
  --motion mocap_climb_seq_0
```

常用可选参数：

```bash
# --data-path PATH               覆盖数据集的默认根目录
# --save-dir PATH                覆盖默认输出目录 demo_results
# --output-name NAME.npz         只修改最终 NPZ 文件名，不改变结果目录层级
# --overwrite                    覆盖已有结果
# --robot-profile FILE_OR_DIR    临时覆盖当前机器人统一 JSON 表
# --foot-sticking True|False     覆盖表内接触感知足底平面约束开关
# --retargeter.debug             显示诊断图层，并在完成后等待 Enter
# --retargeter.no-visualize      关闭默认实时可视化，适用于 headless/CI
# --orientation-tracking True    启用机器人表中当前数据集的朝向权重
# --orientation_weights WEIGHT   用同一非负权重覆盖全部映射 link
# --shoulder-direction-tracking True 直接跟踪躯干局部上臂方向
# --natural-pose-tracking True   启用机器人表中的逐关节自然姿态项
# --nature_weights WEIGHT        用同一权重覆盖全部 actuated joint
```

接触感知足底平面硬约束默认开启。它区分全脚掌、脚跟、脚尖、旋转支点、
主动滑步和摆动相；增强任务中的滑步轨迹会逐帧映射到目标世界坐标，
高台接触必须由攀爬场景的朝上支撑面确认。只在本次重定向中完全关闭该约束：

```bash
python examples/robot_retarget.py \
  --task robot_only \
  --robot e2 \
  --dataset lafan \
  --motion walk2_subject3 \
  --foot-sticking False
```

显式使用 `--foot-sticking True` 可保持开启。该开关不影响地面非穿透、
关节限位等其他约束。

如需完整查看实时诊断图层，直接使用：

```bash
python examples/robot_retarget.py \
  --task robot_only \
  --robot e2 \
  --dataset lafan \
  --motion walk2_subject3 \
  --retargeter.debug
```

结果已存在且命令显示为 resumed 时不会再次进入求解过程；此时增加
`--overwrite` 可重新求解并观看实时过程，或者直接使用下方的结果可视化脚本。

每个机器人只维护 `examples/robot_profiles` 中的一张表。表内包含身高、URDF、
足底开关、根节点与网格权重、直接上臂方向设置、各数据集朝向权重和自然姿态
权重。未写在命令行中的参数读取表内默认值，显式命令行值优先。例如只覆盖
根位置权重可使用：

```bash
python examples/robot_retarget.py \
  --task robot_only \
  --robot e2 \
  --dataset lafan \
  --motion walk2_subject3 \
  --root-position-weight 1.0
```

`--orientation-tracking True` 使用表内当前数据集的逐关键点权重，
`--orientation-weights 1.0` 用统一权重覆盖它们。`--natural-pose-tracking True`
使用表内逐关节自然姿态权重，`--nature-weights 0.1` 用统一权重覆盖。肩部模式
由 `--shoulder-direction-tracking True` 开启，当前实现直接跟踪上臂方向，不会
构造、评分或选择 pitch/roll/yaw 候选分支。

默认结果路径为：

```text
demo_results/<robot>/<task>/<dataset>/<motion>.npz
```

例如上述 E2/LAFAN 命令完成后，直接查看结果：

```bash
python viser_player.py \
  --input-path demo_results/e2/robot_only/lafan/walk2_subject3.npz
```

## 增强任务重定向

`parallel_robot_retarget.py` 用于 G1 的 object-interaction 和 climbing。它会自动生成任务所需的增强变体，无需在命令行逐个配置。

```bash
python examples/parallel_robot_retarget.py \
  --task object_interaction \
  --robot g1 \
  --dataset OMOMO_new \
  --motion sub3_largebox_003
```

climbing 只需替换任务、数据集和动作：

```bash
python examples/parallel_robot_retarget.py \
  --task climbing \
  --robot g1 \
  --dataset climbing \
  --motion mocap_climb_seq_0
```

该入口同样默认实时可视化，并支持 `--retargeter.debug`、
`--retargeter.no-visualize`、`--data-path`、`--save-dir`、`--overwrite`、
`--robot-profile FILE_OR_DIR`、`--foot-sticking True|False`、
`--orientation_weights WEIGHT` 和 `--nature_weights WEIGHT`。开启 debug 时，每个 identity/增强
变体完成后都会等待 Enter，便于逐个检查。默认结果路径为：

```text
demo_results_parallel/<robot>/<task>/<dataset>/<motion>.npz
demo_results_parallel/<robot>/<task>/<dataset>/<motion>_<variant>.npz
```

## NPZ 转 CSV

单文件转换会在 NPZ 旁生成同名 CSV。CSV 不包含表头或索引，每帧顺序为
根节点 `xyz`、根节点四元数 `xyzw`，以及按 URDF 顺序重排的机器人关节角：

```bash
python -m holosoma_retargeting.data_utils.npz_to_csv \
  demo_results/e2/robot_only/lafan/walk2_subject3.npz \
  models/e2/e2_23dof.urdf
```

批量转换递归读取输入目录，并在输出目录中保持相对路径：

```bash
python -m holosoma_retargeting.data_utils.batch_npz_to_csv \
  demo_results/e2/robot_only/lafan \
  models/e2/e2_23dof.urdf \
  demo_results_csv/e2/robot_only/lafan
```

## 查看单个结果

最简命令：

```bash
python viser_player.py \
  --input-path demo_results/e2/robot_only/lafan/walk2_subject3.npz
```

需要完整检查重定向辅助信息时：

```bash
python viser_player.py \
  --input-path demo_results/g1/object_interaction/OMOMO_new/sub3_largebox_003.npz \
  --show-point-clouds \
  --show-interaction-mesh \
  --show-source-orientation-axes \
  --show-target-orientation-axes \
  --show-robot-orientation-axes
```

常用可选参数：

```bash
# --show-mapped-skeletons          显示映射后的人体/机器人关键点骨架
# --show-object-keypoints          显示 object 关键点
# --interaction-mesh-mode both     显示 source 和 target Interaction Mesh
# --interaction-mesh-edges cross   只显示人/机器人与 object 之间的跨组连线
# --orientation-joints LeftArm RightArm
#                                  只显示指定关键点/link 的朝向坐标轴
# --loop                           循环播放
# --port 8081                      指定 Viser 服务端口
```

命令启动后，在终端输出的网址中查看结果；结果文件没有保存的图层会自动跳过。

## 对比增强结果

传入动作的 `<motion>.npz`，即可自动发现同目录下的增广文件并排显示：

```bash
python multi_viser_player.py \
  --family demo_results_parallel/g1/object_interaction/OMOMO_new/sub3_largebox_003.npz
```

常用可选参数：

```bash
# --variants identity trans_0 rot_0 只加载指定变体
# --show-interaction-mesh            显示 Interaction Mesh
# --show-object-keypoints            显示 object 关键点
# --orientation-joints LeftArm RightArm
#                                    只显示指定关键点/link 的朝向坐标轴
# --x-offset 0.8                     调整不同变体之间的横向间距
# --loop                             循环播放
```

也可以显式对比任意结果文件：

```bash
python multi_viser_player.py \
  --qpos-npzs result_a.npz result_b.npz \
  --labels baseline augmented
```
