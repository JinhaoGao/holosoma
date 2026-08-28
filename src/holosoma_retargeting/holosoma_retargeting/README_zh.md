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

## 可选的 link 朝向损失

朝向损失默认关闭。数据带有直接来源的朝向时，可以为 15 个已映射关键点设置统一的非负权重：

```bash
python examples/robot_retarget.py \
  --task robot_only \
  --robot e1 \
  --dataset noetix_mocap \
  --motion run_to_the_right \
  --orientation_weights 1
```

`--orientation_weights 0.1` 会统一使用 `0.1`，`--orientation_weights 0` 等同于关闭。
需要单独控制权重时，使用机器人对应的完整 JSON 配置：

```bash
python examples/robot_retarget.py \
  --task robot_only \
  --robot e1 \
  --dataset noetix_mocap \
  --motion run_to_the_right \
  --orientation_config examples/orientation_weights/e1.json
```

仓库提供 `g1.json`、`e1.json` 和 `e2.json`。每个文件都包含
`gvhmr`、`lafan`、`noetix_csv_climb`、`noetix_mocap` 和 `OMOMO_new`
五张完整权重表，程序根据当前 `--dataset` 自动选择。权重键可以是人体
关键点名，也可以替换成对应机器人的 link 名；每个映射都必须出现，
将数值写为 `0` 即可取消该关键点的朝向损失。配置中的 `robot` 必须与
命令一致，`--orientation_weights` 与 `--orientation_config` 不能同时使用。

配置使用人体 T-pose 与机器人 FK T-pose 计算固定 frame offset，再比较目标 frame 与实际机器人 link frame。G1、E1、E2 都有独立机器人 T-pose；GVHMR、LAFAN、mocap/Noetix CSV、Noetix mocap 和 OMOMO 均有数据格式对应的人体 frame 定义。实现不会从关键点位置或骨向量伪造朝向。若源动作没有直接朝向，例如旧 `climbing` NPY，显式开启朝向损失会清晰报错，位置重定向仍可正常使用。

## 自然姿态正则项

自然姿态正则项同样默认关闭。使用 `--nature_weights WEIGHT` 时，全部 actuated joint 统一使用该非负权重；各关节的固定自然参考角按 `--robot` 从 `examples/nature_weights/g1.json`、`e1.json` 或 `e2.json` 读取。三张表与 `orientation_weights` 目录逐机器人对应，并以弧度保存自然参考角。

```bash
python examples/robot_retarget.py \
  --task robot_only \
  --robot e1 \
  --dataset noetix_mocap \
  --motion run_to_the_right \
  --nature_weights 0.1
```

需要逐关节设置不同权重时，使用 `--nature_config` 指定当前机器人的 JSON 文件，或指定包含三张机器人表的目录。配置表中的 `weights` 是直接参与目标函数的绝对权重，不再经过比例或全局缩放。

```bash
python examples/robot_retarget.py \
  --task robot_only \
  --robot e1 \
  --dataset noetix_mocap \
  --motion run_to_the_right \
  --nature_config examples/nature_weights/e1.json
```

求解器在每次 SQP 迭代中加入固定关节空间代价 `sum_i w_i (q_i - q_i_natural)^2`，因此时间平滑项不会把前一帧的歧义分支继续当作移动参考。带非零权重的关节还会在第一帧求解前初始化到自然参考角一次，后续帧仍从上一帧结果继续求解。

`--nature_weights` 与 `--nature_config` 不能同时使用。两者都不传时不会读取或加入自然姿态目标；统一权重或表内权重为 `0` 时，对应关节不参与该项代价。

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
