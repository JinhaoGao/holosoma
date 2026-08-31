# Holosoma 人体到机器人重定向

当前生产设计包含三个重定向入口。`examples/robot_retarget.py` 对一个明确指定的动作执行普通重定向，`examples/parallel_robot_retarget.py` 对一个明确指定的 object-interaction 或 climbing 动作执行 identity 与增强变体。第二个入口名称中的 `parallel` 是历史名称，不表示遍历数据集；两个入口都不会自动扫描并重定向全部动作。`paired_retargeting/robot_refine.py` 读取两份同步的 robot-only 结果，在保留两条 nominal 轨迹的同时执行跨 actor 联合 refinement；双人部分的配置、求解、结果、场景和可视化均集中在 `paired_retargeting/`，完整契约见 [paired_retargeting/README.md](paired_retargeting/README.md)。

对于来自同一个多 actor FBX 的结果，联合 refinement 会恢复转换阶段保存的 `source_xy_origin_m` 相对偏移，将恢复后的相对 root 位置和跨 actor Interaction Mesh 一起纳入优化。联合结果使用项目现有的 Viser 播放控件和图层系统显示两个机器人、源人体骨架、优化后的机器人映射骨架以及 Interaction Mesh，不使用 MuJoCo viewer，也不会为了排版给任一 actor 添加额外显示偏移。播放器还会从两份单机器人结果追溯到 actor 级 FBX 转换数据，并提供独立的 `Original FBX skeletons` 图层；该图层不进行任何尺度变换，只通过一个共同平移恢复原始 actor XY 和共享地面，因此保留 FBX 的原始人体尺寸、两人间距和高度差，两幅骨架统一显示为黑色。现有 `Human skeleton` 图层继续保留各 actor 经过机器人尺度预处理后的参考骨架。

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

命令只要求选择任务、机器人、数据集和一个动作。求解时的 Viser
可视化默认开启；`--retargeter.debug` 会增加人体/机器人映射关键点、
手骨架和 object 点云等诊断图层，并在完成后等待 Enter；
`--retargeter.no-visualize` 用于无界面或批处理环境。数据不在仓库默认
目录时增加 `--data-path /path/to/dataset`；需要覆盖结果根时增加
`--save-dir /path/to/results`；需要只修改最终文件名时增加
`--output-name custom.npz`，该参数只接受文件名并保留
`<robot>/<task>/<dataset>` 目录层级；只有确定要替换同一路径中的旧结果时才使用
`--overwrite`。已有结果被直接续用时不会重新进入实时求解，可增加
`--overwrite` 重跑，或使用结果可视化脚本。全数据遍历仍不属于公开参数。

接触感知的足底平面硬约束默认开启。它会区分全脚掌、脚跟、脚尖、旋转
支点、主动滑步和摆动相；旋转相只锁定实际支点，主动滑步会跟随逐帧映射
到目标世界坐标的人体足底轨迹。高台接触还必须匹配攀爬场景中明确的朝上
支撑面，单纯静止的悬空脚不会被锁定。`--foot-sticking True` 显式保持开启，
`--foot-sticking False` 会在本次单任务或增强任务重定向中完全关闭所有
足底平面约束，但不会关闭地面非穿透或关节限位。例如：

```bash
python examples/robot_retarget.py \
  --task robot_only \
  --robot e2 \
  --dataset lafan \
  --motion walk2_subject3 \
  --foot-sticking False
```

## 数据与机器人矩阵

| 数据集 preset | 内部格式 | 原始关键点朝向 | `robot_only` |
| --- | --- | --- | --- |
| `climbing` | `mocap` | 否，仓库旧 climbing 源只有位置 NPY | G1、E1、E2 |
| `fbx_mocap` | `fbx_mocap` | 是，FBX 局部旋转曲线直接 FK | G1、E1、E2 |
| `gvhmr` | `gvhmr` | 是，SMPL-X 直接旋转 FK | G1、E1、E2 |
| `lafan` | `lafan` | 是，BVH rotation channel FK | G1、E1、E2 |
| `noetix_csv_climb` | `mocap` | 是，标准转换 NPZ 中的 bone rotation FK | G1、E1、E2 |
| `noetix_mocap` | `noetix_mocap` | 是，BVH rotation channel FK | G1、E1、E2 |
| `OMOMO_new` | `omomo` | 是，InterMimic 全局朝向 tensor | G1、E1、E2 |

`object_interaction` 只接受 `OMOMO_new`，并且只支持 G1。`climbing` 只接受 `climbing` 和 `noetix_csv_climb`，并且只支持 G1。增强入口只接受这两类任务，因此 E1/E2 以及 `robot_only` 必须使用普通单动作入口。

## 每个机器人一份统一配置

每个具体机器人现在只维护 `examples/robot_profiles` 中的一份 JSON，即
`g1.json`、`e1_23dof.json`、`e1_24dof.json` 或 `e2.json`；兼容名称
`--robot e1` 会读取 E1 23DoF 的表。表内统一保存机器人身高、自由度、URDF、
根节点与 Interaction Mesh 权重、求解器和足底接触默认值、肩部方向参数、
各数据集的朝向权重，以及自然姿态参考角和权重。原先拆开的 `nature_weights`
与 `orientation_weights` 目录不再使用。

程序每次都会先读取机器人表。命令行显式给出的值优先于 JSON，未给出的值
则继承 JSON。例如下面的命令只把根位置权重覆盖为 `1.0`，其余参数仍取
E1 23DoF 表中的默认值：

```bash
python examples/robot_retarget.py \
  --task robot_only \
  --robot e1_23dof \
  --dataset noetix_mocap \
  --motion sequence/name \
  --root-position-weight 1.0
```

`--robot-profile FILE_OR_DIR` 可临时加载另一份完整表，而不必修改仓库内默认
文件。`--robot-height`、`--robot-dof` 和 `--robot-urdf-file` 可以直接覆盖
机器人参数；`--foot-sticking False` 这类显式布尔值同样会覆盖表内开关。

## 直接跟踪上臂方向

当机器人表中的 `shoulder_direction.enable` 为 `true`，或者命令行传入
`--shoulder-direction-tracking True` 时，求解器会关闭 Arm、ForeArm、Hand
的通用完整 SO(3) 目标，并直接在逐帧 SQP 中跟踪躯干局部坐标系下的上臂
单位方向。当前实现不再构造候选、不再评分候选、不再选择 pitch/roll/yaw
解分支，也不存在离线关节参考路径或分支参考项。上臂方向没有约束到的冗余
自由度由原有 Interaction Mesh、关节限位、可选自然姿态项和逐帧平滑项共同
决定。方向残差权重可用 `--shoulder-direction-weight` 临时覆盖。

G1 的 Hand 朝向只作用于三个腕关节，E1 的 Hand 朝向只投影到单个
elbow-yaw 轴，E2 不设置腕部朝向任务。开启朝向表时，下肢和躯干朝向仍正常
生效。结果会保存目标/实际上臂方向、方向误差、实际肩部关节角和方向雅可比
奇异值，可用下面的命令绘图：

```bash
python -m holosoma_retargeting.visualization.shoulder_direction result.npz
```

## 朝向、根稳定性与自然姿态

仓库内四张表保持原来的默认行为：朝向跟踪、肩部方向跟踪、自然姿态正则和
两个根稳定权重默认关闭，foot-sticking 默认开启。若希望长期改变某台机器人
的行为，只需修改它的一张 JSON。单次运行中，`--orientation-tracking True`
会启用当前数据集的逐关键点朝向表，`--orientation-weights WEIGHT` 会用一个
非负数覆盖全部映射权重；`--natural-pose-tracking True` 会启用表内逐关节
自然姿态权重，`--nature-weights WEIGHT` 则为全部 actuated joint 使用同一
权重。

根节点稳定仍是软目标。位置目标跟踪首帧对齐后的源人体 root 位移，朝向目标
优先使用源数据的直接 root 朝向，否则使用人体躯干坐标系；首帧求解结果用于
建立后续相对目标。根节点与网格权重可按下面方式联合调节：

```bash
python examples/robot_retarget.py \
  --task robot_only \
  --robot e1_23dof \
  --dataset noetix_mocap \
  --motion sequence/name \
  --orientation-tracking True \
  --shoulder-direction-tracking True \
  --root-position-weight 200 \
  --root-orientation-weight 50 \
  --interaction-mesh-weight 10 \
  --arm-interaction-mesh-weight-scale 0.5
```

自然姿态目标仍为 `sum_i w_i (q_i - q_i_natural)^2`。非零权重关节会在首帧
前从固定参考角初始化，后续帧继续使用上一帧解。

## 增强变体

object-interaction 的增强族包含 `identity`、三个 object translation 变体 `trans_0` 至 `trans_2`，以及两个带平移的正负 45 度旋转变体 `rot_0`、`rot_1`。climbing 的增强族包含 `identity` 和 Z 方向尺度 `z_scale_0p8`、`z_scale_0p9`、`z_scale_1p1`、`z_scale_1p2`。identity 首先生成，后续变体复用其 nominal 轨迹。

## 结果目录

普通结果只写入：

```text
demo_results/<robot>/<task>/<dataset>/<motion>.npz
```

传入 `--output-name custom.npz` 后，只会把末尾的 `<motion>.npz` 替换为
`custom.npz`。

增强结果只写入：

```text
demo_results_parallel/<robot>/<task>/<dataset>/<motion>.npz
demo_results_parallel/<robot>/<task>/<dataset>/<motion>_<variant>.npz
```

增强入口使用自定义名称时，identity 写为 `custom.npz`，各增强变体写为
`custom_<variant>.npz`。

默认结果根已经按 `g1`、`e1`、`e2` 分层。仓库不再维护 ablation、orientation、search、comparison 或全数据 rebuild 结果树。

## 紧凑结果

结果只保留求解和后续可视化真正需要的数据。主要内容包括 float64 `qpos`、FPS 与 cost；位置或朝向重定向实际使用的人体关键点和机器人 link 点；紧凑骨骼连接关系；映射 link 的逐帧位置与 wxyz 朝向；求解器实际使用的源人体朝向；人体、机器人、地形和 object 的有效点云；source/target Interaction Mesh；以及 object 位姿和关键点。这些数据默认保存，不需要额外命令行开关。

未参与位置或朝向重定向的完整人体骨架节点会被裁掉，未参与位置或朝向映射的完整机器人 link 也会被裁掉。这样既保留可解释性，也避免把全模型轨迹重复写入每个结果。

结果写入采用与 `main` 接近的直接 NPZ 保存方式，不再维护 schema、源文件 hash、配置 hash、外部资产 manifest、文件锁、临时制品发布或跨字段严格校验。

## 可视化

单结果查看：

```bash
python viser_player.py \
  --input-path demo_results/e2/robot_only/lafan/walk2_subject3.npz
```

可视化脚本默认加载结果中所有可用的辅助数据。Interaction Mesh、人体/机器人骨架、人体/机器人/地形/object 点云、object keypoints 与三维朝向轴都在 Viser 的 Layers 面板中独立开关，不需要在启动命令中逐项列出。若某项源数据不存在，对应图层会显示为不可用，而不会生成虚假数据。

增强族也可以同步查看：

```bash
python multi_viser_player.py \
  --family demo_results_parallel/g1/object_interaction/OMOMO_new/sub3_largebox_003.npz
```

单个结果转换为无表头、无索引的纯数值 CSV。每帧依次保存根节点 `xyz`、
根节点四元数 `xyzw`，以及按 URDF 顺序排列的机器人关节角：

```bash
python -m holosoma_retargeting.data_utils.npz_to_csv \
  demo_results/e2/robot_only/lafan/walk2_subject3.npz \
  models/e2/e2_23dof.urdf
```

批量转换并保持输入目录的相对结构：

```bash
python -m holosoma_retargeting.data_utils.batch_npz_to_csv \
  demo_results/e2/robot_only/lafan \
  models/e2/e2_23dof.urdf \
  demo_results_csv/e2/robot_only/lafan
```

## 验证

生产相关测试可以从仓库根执行：

```bash
PYTHONPATH=src/holosoma_retargeting \
  /home/jinhaogao/miniconda3/envs/robot_retargeter/bin/python \
  -m pytest --noconftest -q tests/test_*.py
```

更严格的接口、矩阵、制品和提交门禁见仓库根目录的 `docs/retargeting-production-contract.md`。新增人体格式的要求见 `ADD_MOTION_FORMAT_README_zh.md`。
