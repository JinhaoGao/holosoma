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
# --overwrite                    覆盖已有结果
# --foot-sticking True|False     开启或完全关闭足底粘连 XY 硬约束，默认 True
# --retargeter.debug             显示诊断图层，并在完成后等待 Enter
# --retargeter.no-visualize      关闭默认实时可视化，适用于 headless/CI
# --orientation_weights WEIGHT     开启朝向损失，全部映射 link 使用同一非负权重
# --orientation_config FILE_OR_DIR 开启朝向损失，从 JSON 读取各 link 权重
# --nature_weights WEIGHT          开启自然姿态项，全部 actuated joint 使用同一非负权重
# --nature_config FILE_OR_DIR      开启自然姿态项，从 JSON 读取参考角与逐关节权重
```

足底粘连硬约束默认开启。只在本次重定向中完全关闭该 XY 硬约束：

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

统一使用权重 `1.0`：

```bash
python examples/robot_retarget.py \
  --task robot_only \
  --robot e2 \
  --dataset lafan \
  --motion walk2_subject3 \
  --orientation_weights 1
```

逐关键点配置使用
`examples/orientation_weights/g1.json`、`e1.json` 或 `e2.json`。每个文件
按 `gvhmr`、`lafan`、`noetix_csv_climb`、`noetix_mocap` 和 `OMOMO_new`
分别保存完整的 15 个关键点权重；程序会根据当前 `--dataset` 自动选表。
把某一项改成 `0` 即可取消该关键点的朝向损失。

```bash
python examples/robot_retarget.py \
  --task robot_only \
  --robot e2 \
  --dataset lafan \
  --motion walk2_subject3 \
  --orientation_config examples/orientation_weights/e2.json
```

`--orientation_weights` 与 `--orientation_config` 不能同时使用。旧 `climbing`
只有位置数据，不支持这两个朝向选项。

自然姿态正则项默认关闭。`--nature_weights 0.1` 会从
`examples/nature_weights/g1.json`、`e1.json` 或 `e2.json` 读取当前机器人的
自然参考角，并给全部 actuated joint 统一设置 `0.1` 权重。需要逐关节
设置不同权重时，使用 `--nature_config examples/nature_weights/e2.json`；
该参数也接受包含三张对应表的目录。两种自然姿态权重来源不能同时使用。

默认结果路径为：

```text
demo_results/<robot>/<task>/<dataset>/<motion>/identity.npz
```

例如上述 E2/LAFAN 命令完成后，直接查看结果：

```bash
python viser_player.py \
  --input-path demo_results/e2/robot_only/lafan/walk2_subject3/identity.npz
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
`--foot-sticking True|False`、`--orientation_weights WEIGHT`、
`--orientation_config FILE_OR_DIR`、`--nature_weights WEIGHT` 和
`--nature_config FILE_OR_DIR`。开启 debug 时，每个 identity/增强
变体完成后都会等待 Enter，便于逐个检查。默认结果路径为：

```text
demo_results_parallel/<robot>/<task>/<dataset>/<motion>/<variant>.npz
```

## 查看单个结果

最简命令：

```bash
python viser_player.py \
  --input-path demo_results/e2/robot_only/lafan/walk2_subject3/identity.npz
```

需要完整检查重定向辅助信息时：

```bash
python viser_player.py \
  --input-path demo_results/g1/object_interaction/OMOMO_new/sub3_largebox_003/identity.npz \
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

传入包含多个 `<variant>.npz` 的动作目录，即可自动加载并排显示：

```bash
python multi_viser_player.py \
  --family demo_results_parallel/g1/object_interaction/OMOMO_new/sub3_largebox_003
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
