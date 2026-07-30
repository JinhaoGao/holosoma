# Holosoma 人体到机器人重定向

当前生产设计只保留两个重定向入口。`examples/robot_retarget.py` 对一个明确指定的动作执行普通重定向，`examples/parallel_robot_retarget.py` 对一个明确指定的 object-interaction 或 climbing 动作执行 identity 与增强变体。第二个入口名称中的 `parallel` 是历史名称，不表示遍历数据集；两个入口都不会自动扫描并重定向全部动作。

命令默认从本目录执行：

```bash
cd src/holosoma_retargeting/holosoma_retargeting
```

## 最简用法

对 LAFAN 的一个动作重定向到 E2：

```bash
python examples/robot_retarget.py \
  --task robot_only \
  --robot e2 \
  --dataset lafan \
  --motion walk2_subject3
```

对 GVHMR 的一个动作重定向到 G1：

```bash
python examples/robot_retarget.py \
  --task robot_only \
  --robot g1 \
  --dataset gvhmr \
  --motion tennis
```

对 OMOMO 的一个 object-interaction 动作执行 G1 增强重定向：

```bash
python examples/parallel_robot_retarget.py \
  --task object_interaction \
  --robot g1 \
  --dataset OMOMO_new \
  --motion sub3_largebox_003
```

对已有 climbing 动作执行 G1 增强重定向：

```bash
python examples/parallel_robot_retarget.py \
  --task climbing \
  --robot g1 \
  --dataset climbing \
  --motion mocap_climb_seq_0
```

命令只要求选择任务、机器人、数据集和一个动作。数据不在仓库默认目录时增加 `--data-path /path/to/dataset`；需要覆盖结果根时增加 `--save-dir /path/to/results`；只有确定要替换同一路径中的旧结果时才使用 `--overwrite`。实时 debug、求解时 visualization 和全数据遍历都不属于公开参数。

## 数据与机器人矩阵

| 数据集 preset | 内部格式 | 原始关键点朝向 | `robot_only` |
| --- | --- | --- | --- |
| `climbing` | `mocap` | 否，仓库旧 climbing 源只有位置 NPY | G1、E1、E2 |
| `gvhmr` | `gvhmr` | 是，SMPL-X 直接旋转 FK | G1、E1、E2 |
| `lafan` | `lafan` | 是，BVH rotation channel FK | G1、E1、E2 |
| `noetix_csv_climb` | `mocap` | 是，标准转换 NPZ 中的 bone rotation FK | G1、E1、E2 |
| `noetix_mocap` | `noetix_mocap` | 是，BVH rotation channel FK | G1、E1、E2 |
| `OMOMO_new` | `omomo` | 是，InterMimic 全局朝向 tensor | G1、E1、E2 |

`object_interaction` 只接受 `OMOMO_new`，并且只支持 G1。`climbing` 只接受 `climbing` 和 `noetix_csv_climb`，并且只支持 G1。增强入口只接受这两类任务，因此 E1/E2 以及 `robot_only` 必须使用普通单动作入口。

## 可选的 link 朝向损失

朝向损失默认关闭。数据带有直接来源的朝向时，可以用一个开关对 15 个已映射关键点启用等权朝向损失：

```bash
python examples/robot_retarget.py \
  --task robot_only \
  --robot e1 \
  --dataset noetix_mocap \
  --motion run_to_the_right \
  --orientation
```

需要单独控制权重时，传入 JSON 配置。键既可以是人体关键点名，也可以是对应机器人的 link 名；未写入的映射权重为零：

```json
{
  "enabled": true,
  "weights": {
    "Hips": 0.2,
    "LeftArm": 0.1,
    "RightArm": 0.1,
    "l_leg_ankle_roll_link": 0.05,
    "r_leg_ankle_roll_link": 0.05
  }
}
```

```bash
python examples/robot_retarget.py \
  --task robot_only \
  --robot e1 \
  --dataset noetix_mocap \
  --motion run_to_the_right \
  --orientation-config ./orientation.json
```

配置使用人体 T-pose 与机器人 FK T-pose 计算固定 frame offset，再比较目标 frame 与实际机器人 link frame。G1、E1、E2 都有独立机器人 T-pose；GVHMR、LAFAN、mocap/Noetix CSV、Noetix mocap 和 OMOMO 均有数据格式对应的人体 frame 定义。实现不会从关键点位置或骨向量伪造朝向。若源动作没有直接朝向，例如旧 `climbing` NPY，显式开启朝向损失会清晰报错，位置重定向仍可正常使用。

## 增强变体

object-interaction 的增强族包含 `identity`、三个 object translation 变体 `trans_0` 至 `trans_2`，以及两个带平移的正负 45 度旋转变体 `rot_0`、`rot_1`。climbing 的增强族包含 `identity` 和 Z 方向尺度 `z_scale_0p8`、`z_scale_0p9`、`z_scale_1p1`、`z_scale_1p2`。identity 首先生成，后续变体复用其 nominal 轨迹。

## 结果目录

普通结果只写入：

```text
demo_results/<robot>/<task>/<dataset>/<motion>/identity.npz
```

增强结果只写入：

```text
demo_results_parallel/<robot>/<task>/<dataset>/<motion>/<variant>.npz
```

默认结果根已经按 `g1`、`e1`、`e2` 分层。仓库不再维护 ablation、orientation、search、comparison 或全数据 rebuild 结果树。

## 紧凑结果制品

结果只保留求解和后续可视化真正需要的数据。主要内容包括 float64 `qpos`、FPS、cost 与必要约束诊断；位置损失实际使用的人体关键点和机器人 link 点；源数据存在时的手部关键点；映射 link 的逐帧位置与 wxyz 朝向；源数据可证明为直接来源时的对应人体朝向；人体、机器人、地形和 object 的有效点云；source/target Interaction Mesh；object 位姿、点和外部资产身份；以及源文件和有效配置的 SHA-256。

未参与求解或手部可视化的完整人体骨架节点会被裁掉，未参与位置或朝向映射的完整机器人 link 也会被裁掉。这样既保留可解释性，也避免把全模型轨迹重复写入每个结果。

结果写入仍保留必要的局部安全边界：同一目标文件的锁、严格制品校验、临时文件原子替换、源文件 hash 与有效配置 hash。已经移除全局缓存、staging 树、代码/运行时/全部静态资产指纹、批量 manifest 和实验命名空间等与单动作生产接口无关的安全层。

## 可视化

单结果查看：

```bash
python viser_player.py \
  --input-path demo_results/e2/robot_only/lafan/walk2_subject3/identity.npz
```

查看保存的关键数据层：

```bash
python viser_player.py \
  --input-path demo_results/g1/object_interaction/OMOMO_new/sub3_largebox_003/identity.npz \
  --show-point-clouds \
  --show-interaction-mesh \
  --show-source-orientation-axes \
  --show-robot-orientation-axes
```

`--show-point-clouds` 用一个开关显示人体、机器人、地形、demo object 和 target object 点云。Interaction Mesh、人体/机器人骨架、object keypoints 与三维朝向轴在 Viser 的 Layers 面板中可独立开关。若某项源数据不存在，对应图层会显示为不可用，而不会生成虚假数据。

增强族也可以同步查看：

```bash
python multi_viser_player.py \
  --family demo_results_parallel/g1/object_interaction/OMOMO_new/sub3_largebox_003
```

## 验证

生产相关测试可以从仓库根执行：

```bash
PYTHONPATH=src/holosoma_retargeting \
  /home/jinhaogao/miniconda3/envs/robot_retargeter/bin/python \
  -m pytest --noconftest -q tests/test_*.py
```

更严格的接口、矩阵、制品和提交门禁见仓库根目录的 `docs/retargeting-production-contract.md`。新增人体格式的要求见 `ADD_MOTION_FORMAT_README_zh.md`。
