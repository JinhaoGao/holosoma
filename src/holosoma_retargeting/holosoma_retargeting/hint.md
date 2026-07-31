# 重定向与可视化速查

以下命令均从重定向包目录执行：

```bash
conda activate robot_retargeter
cd /home/jinhaogao/holosoma/src/holosoma_retargeting/holosoma_retargeting
export PYTHONPATH=..
```

公开重定向入口不支持旧参数 `--retargeter.debug` 和
`--retargeter.visualize`。当前流程统一为先保存重定向结果，再用
`viser_player.py` 或 `multi_viser_player.py` 查看，不在求解过程中启动实时可视化。

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
# --orientation                  开启关键点朝向损失，所有已映射关键点使用等权重
# --orientation-config FILE.json 开启朝向损失，并从 JSON 读取各关键点/link 的权重
```

朝向权重文件的最小示例：

```json
{
  "enabled": true,
  "weights": {
    "Hips": 0.2,
    "LeftArm": 0.1,
    "RightArm": 0.1
  }
}
```

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

该入口同样支持 `--data-path`、`--save-dir`、`--overwrite`、`--orientation` 和 `--orientation-config`。默认结果路径为：

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
