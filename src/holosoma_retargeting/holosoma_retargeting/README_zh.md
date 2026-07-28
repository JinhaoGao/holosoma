# Holosoma 动作重定向

简体中文 | [English](README.md)

本仓库提供将人体动作重定向到人形机器人的工具。官方项目原本支持 OMOMO、LAFAN、AMASS SMPL-X、通用 mocap，以及纯机器人动作、物体交互和攀爬任务。本分支在完整保留这些工作流的基础上，增加了 Noetix-mocap 与 GVHMR 的显式适配，并统一了结果保存和可视化接口。

**数据要求：**重定向流水线接收形状为 `(T, J, 3)` 的世界坐标系人体关节位置，其中 `T` 为帧数，`J` 为关节数。支持朝向跟踪的格式还可提供 `(T, J, 4)` 的世界坐标系 `wxyz` 四元数；朝向数据缺失时，默认位置求解路径不受影响。每种格式分别声明关节顺序、机器人映射、文件布局、坐标变换、人体身高、FPS 和支持的任务类型。如需添加其他格式，请参阅 [ADD_MOTION_FORMAT_README_zh.md](ADD_MOTION_FORMAT_README_zh.md)。

请在以下目录中执行本文命令：

```bash
cd src/holosoma_retargeting/holosoma_retargeting
```

## 支持的人体动作数据与任务

| `--data-format` | 来源/输入 | 重定向输入 | 任务类型 | 机器人 |
| --- | --- | --- | --- | --- |
| `omomo` | InterMimic 预处理 OMOMO | `.pt`，52 个 SMPL-H 关节 | `robot_only`、`object_interaction` | G1、T1、E1 |
| `lafan` | LAFAN BVH | `.npy`，22 个 LAFAN 关节 | `robot_only` | G1、T1、E1 |
| `amass` | AMASS SMPL-X | `.npz`，22 个 SMPL-X 关节 | `robot_only` | G1、E1 |
| `mocap` | 嵌套目录中的世界关节 mocap | `.npy`，53 个关节 | `robot_only`、`climbing` | G1、T1、E1 映射 |
| `noetix_mocap` | 公司自采 Noetix BVH | `.npz`，标准化 22 关节 Noetix 格式 | `robot_only` | G1、E1 |
| `gvhmr` | GVHMR `hmr4d_results.pt` | `.npz`，22 个 SMPL-X 关节 | `robot_only` | G1、E1 |

表中列出的是标准格式名称。命令行仍兼容旧名称：`smplh`（`omomo`）、`smplx`（`amass`）以及 `noetix_lafan`/`noetix-mocap`（`noetix_mocap`）。

> **LAFAN 与 Noetix-mocap 是两套相互独立的数据。**LAFAN 是 Ubisoft 公开数据集，遵循官方 `BVH → .npy` 流程；Noetix-mocap 是公司自采动捕数据，包含多种 Noetix BVH 导出骨架，使用独立的 `BVH → .npz` 流程。Noetix 标准化骨架只是在重定向层复用了兼容的 22 关节命名拓扑，这不代表 Noetix 源数据属于 LAFAN。

## 单序列动作重定向

下面是官方原有的任务工作流，仅将参数更新为当前 Tyro 使用的连字符形式，并使用标准数据格式名称：

```bash
# 纯机器人动作（OMOMO）
python examples/robot_retarget.py \
  --data-path demo_data/OMOMO_new \
  --task-type robot_only \
  --task-name sub3_largebox_003 \
  --data-format omomo \
  --retargeter.debug \
  --retargeter.visualize

# 物体交互（OMOMO）
python examples/robot_retarget.py \
  --data-path demo_data/OMOMO_new \
  --task-type object_interaction \
  --task-name sub10_tripod_000 \
  --data-format omomo \
  --robot e1 \
  --retargeter.debug \
  --retargeter.visualize

# 攀爬（通用 mocap）
python examples/robot_retarget.py \
  --data-path demo_data/climb \
  --task-type climbing \
  --task-name mocap_climb_seq_0 \
  --data-format mocap \
  --robot-config.robot-urdf-file models/g1/g1_29dof_spherehand.urdf \
  --retargeter.debug \
  --retargeter.visualize
```

其他纯机器人数据格式使用相同入口：

```bash
# LAFAN
python examples/robot_retarget.py \
  --data-path demo_data/lafan \
  --task-type robot_only \
  --task-name dance2_subject1 \
  --data-format lafan \
  --task-config.ground-range -10 10 \
  --save-dir demo_results/g1/robot_only/lafan \
  --retargeter.foot-sticking-tolerance 0.02

# AMASS SMPL-X
python examples/robot_retarget.py \
  --data-path demo_data/amass_smplx_processed \
  --task-type robot_only \
  --task-name ACCAD_Female1Running_c3d_C3_-_Run_stageii \
  --data-format amass \
  --task-config.ground-range -10 10 \
  --save-dir demo_results/g1/robot_only/amass

# Noetix-mocap
python examples/robot_retarget.py \
  --data-path demo_data/noetix_mocap \
  --task-type robot_only \
  --task-name run_to_the_right \
  --data-format noetix_mocap \
  --save-dir demo_results/g1/robot_only/noetix_mocap

# GVHMR
python examples/robot_retarget.py \
  --data-path demo_data/gvhmr \
  --task-type robot_only \
  --task-name tennis \
  --data-format gvhmr \
  --save-dir demo_results/g1/robot_only/gvhmr
```

使用 `--augmentation` 可运行物体交互或攀爬增强序列，但必须先生成相应的原始序列结果。
OMOMO 物体交互会从标准的 `subN_object_NNN` 任务名自动推断物体类别。
`--task-config.object-name` 不再是必需参数；若显式传入，它将用于检查任务名与配置是否一致。

## 批量动作重定向

官方原有的批处理工作流继续保留：

```bash
# 纯机器人动作（OMOMO）
python examples/parallel_robot_retarget.py \
  --data-dir demo_data/OMOMO_new \
  --task-type robot_only \
  --data-format omomo \
  --save-dir demo_results_parallel/g1/robot_only/omomo \
  --task-config.object-name ground

# 在 G1 上处理全部 OMOMO 物体交互序列
python examples/parallel_robot_retarget.py \
  --data-dir demo_data/OMOMO_new \
  --task-type object_interaction \
  --data-format omomo \
  --robot g1 \
  --save-dir demo_results_parallel/g1/object_interaction/omomo \
  --max-workers 4

# 攀爬
python examples/parallel_robot_retarget.py \
  --data-dir demo_data/climb \
  --task-type climbing \
  --data-format mocap \
  --robot-config.robot-urdf-file models/g1/g1_29dof_spherehand.urdf \
  --task-config.object-name multi_boxes \
  --save-dir demo_results_parallel/g1/climbing/mocap_climb
```

只需修改 `--data-format`、`--data-dir` 和 `--save-dir`，同一个批处理命令即可用于 `lafan`、`amass`、`noetix_mocap` 或 `gvhmr`。例如：

```bash
python examples/parallel_robot_retarget.py \
  --data-dir demo_data/gvhmr \
  --task-type robot_only \
  --data-format gvhmr \
  --save-dir demo_results_parallel/g1/robot_only/gvhmr \
  --max-workers 4 \
  --retargeter.save-interaction-mesh
```

为物体交互或攀爬批处理增加 `--augmentation` 后，会同时处理原始和增强序列。对 OMOMO 物体交互，每条动作都会生成 `original`、`trans_0`、`trans_1`、`trans_2`、`rot_0` 和 `rot_1` 六个结果；三种平移分别为初始人体朝向物体的局部坐标 `[0.2, 0, 0]`、`[0, 0.2, 0]` 和 `[0, -0.2, 0]` 米，随后会转换到世界坐标。两种旋转分别为绕 Z 轴 `+45°` 和 `-45°`，并分别附带 `[0, 0.2, 0]` 和 `[0, -0.2, 0]` 米局部平移。完整扰动在物体开始运动之前保持不变，之后按指数逐渐衰减。增强是在原始重定向结果的机器人初始/名义轨迹基础上重新求解，不是简单变换已经输出的 qpos。

增强配置不按动作语义做白名单筛选；只要序列能通过数据和物体资产预检，所选物体类别下的每条 `object_interaction` 动作都会尝试全部五种增强。比如覆盖重算全部 largebox，并同时保存缩放前后物体点云和 Interaction Mesh：

```bash
python examples/parallel_robot_retarget.py \
  --data-dir demo_data/OMOMO_new \
  --task-type object_interaction \
  --data-format omomo \
  --robot g1 \
  --object-names largebox \
  --save-dir demo_results_parallel/g1/object_interaction/omomo \
  --max-workers 4 \
  --augmentation \
  --overwrite-existing \
  --retargeter.save-interaction-mesh
```

已经存在的输出文件默认会被跳过，只有传入 `--overwrite-existing` 才会覆盖。OMOMO 批处理默认先检查数据、身高表和物体资产，并在结果目录写入 `batch_report.json`。

目前物体目录完整支持 `clothesstand`、`floorlamp`、`largebox`、`largetable`、`monitor`、`plasticbox`、`smallbox`、`smalltable`、`suitcase`、`trashcan`、`tripod`、`whitechair` 和 `woodchair`。省略 `--object-names` 会处理全部类别，也可以精确选择子集。建议先用 `--dry-run` 验证全部输入并生成清单，此时不会启动优化：

```bash
# 检查十三类物体并生成完整清单
python examples/parallel_robot_retarget.py \
  --data-dir demo_data/OMOMO_new \
  --task-type object_interaction \
  --data-format omomo \
  --robot g1 \
  --save-dir demo_results_parallel/g1/object_interaction/omomo \
  --dry-run

# 在 E1 上断点续跑指定类别
python examples/parallel_robot_retarget.py \
  --data-dir demo_data/OMOMO_new \
  --task-type object_interaction \
  --data-format omomo \
  --robot e1 \
  --object-names tripod suitcase whitechair \
  --save-dir demo_results_parallel/e1/object_interaction/omomo \
  --max-workers 4
```

## 数据准备

仓库提供 `demo_data/` 用于快速测试。若要准备更多序列，请按以下数据集说明操作。

### OMOMO

Holosoma 官方流水线使用 InterMimic 预处理后的数据，其格式与原始 OMOMO 发布格式不同。

1. 从[此链接](https://drive.google.com/file/d/141YoPOd2DlJ4jhU2cpZO5VU5GzV_lm5j/view)下载预处理 OMOMO 数据。
2. 解压至 `demo_data/OMOMO_new`。
3. 将 `height_dict.pkl` 放在 `OMOMO_new` 的上级目录。也可以传入 `--motion-data-config.human-height HEIGHT` 显式指定身高。

动作文件应为 `.pt` 张量。

长时间批处理前，可以单独检查完整数据目录、被试身高表和十三类内置资产：

```bash
python data_utils/preflight_omomo.py demo_data/OMOMO_new \
  --asset-root models \
  --output demo_results_parallel/omomo_preflight.json
```

发布前可以让 G1/E1 与每类物体各执行两帧真实优化。该命令会生成并校验 26 个结果 NPZ：

```bash
python data_utils/validate_omomo_retargeting.py demo_data/OMOMO_new \
  --robots g1 e1 \
  --frames 2 \
  --output-dir demo_results_validation/omomo_g1_e1
```

### LAFAN

#### 下载原始 LAFAN 数据

1. 打开 [lafan1.zip](https://github.com/ubisoft/ubisoft-laforge-animation-dataset/blob/master/lafan1/lafan1.zip)，点击 **View Raw** 下载。
2. 将 `lafan1.zip` 放到指定数据目录，并解压到 `DATA_FOLDER_PATH/lafan`。
3. 原始文件结构应为 `DATA_FOLDER_PATH/lafan/*.bvh`。

#### 转换为重定向所需的 LAFAN 格式

转换过程需要 [LAFAN GitHub 仓库](https://github.com/ubisoft/ubisoft-laforge-animation-dataset)中的文件：

```bash
cd data_utils
git clone https://github.com/ubisoft/ubisoft-laforge-animation-dataset.git
mv ubisoft-laforge-animation-dataset/lafan1 .
python extract_global_positions.py \
  --input-dir DATA_FOLDER_PATH/lafan \
  --output-dir ../demo_data/lafan
cd ..
```

该脚本会将每个 BVH 文件转换为包含全局关节位置的 `.npy` 文件。`.npy` 数据保持官方 LAFAN 的 Y-up 约定，重定向加载器会将其转换为 Z-up。LAFAN 数据通常需要使用 `--retargeter.foot-sticking-tolerance 0.02` 放宽足部固定约束，并可根据动作质量继续调整。

### AMASS SMPL-X

#### 下载原始 AMASS 数据

1. 按照 [AMASS](https://amass.is.tue.mpg.de/) 的说明下载原始 AMASS 数据。
2. 预期目录结构为 `/path/to/amass/dataset_name/subject_name/*_stageii.npz`。

#### 下载 SMPL-X 模型

1. 按照 [SMPL-X](https://smpl-x.is.tue.mpg.de/index.html) 的说明下载授权模型。
2. 官方流水线使用中性 SMPL-X 模型进行测试。
3. 预期模型路径为 `/path/to/models/smplx/SMPLX_NEUTRAL.npz`。

#### 转换为重定向所需的 AMASS SMPL-X 格式

官方文档曾使用 [human_body_prior](https://github.com/nghorbani/human_body_prior) 工具。当前转换器保留相同的 AMASS 输入/输出工作流，但直接使用环境中安装的 SMPL-X 模型：

```bash
python data_utils/prep_amass_smplx_for_rt.py \
  --amass-root-folder /path/to/amass \
  --output-folder /path/to/output \
  --model-root-folder /path/to/models
```

转换器递归写出包含全局关节位置、人体身高、FPS、关节名称和根节点四元数的 `.npz` 文件。使用 `--subdataset-folder HumanEva` 可只处理一个子数据集；省略该参数则处理全部子数据集。

### 攀爬 Mocap

官方攀爬任务使用嵌套的序列目录：

```text
demo_data/climb/
└── mocap_climb_seq_0/
    ├── <motion>.npy
    ├── multi_boxes.obj
    ├── multi_boxes.urdf
    ├── box_assets.xml
    └── <robot scene>.xml
```

`mocap` 适配器会将动作从 120 FPS 采样至 30 FPS，地形资源继续保存在对应序列目录中。请使用单序列和批处理攀爬命令中展示的球形手机器人 URDF。

### Noetix-mocap

将受支持的公司自采 Noetix BVH 骨架变体转换为标准化 22 关节 Noetix 重定向格式：

```bash
python data_utils/convert_noetix_bvh.py \
  --input-dir demo_data/noetix \
  --output-dir demo_data/noetix_mocap \
  --target-fps 30
```

这是 Noetix 专用转换器，与上面的官方 LAFAN BVH 转换流程无关。输出 `.npz` 同时包含 Z-up 全局关节位置 `global_joint_positions` 和全局关节朝向 `global_joint_quaternions_wxyz`，并保存关节名称、源/输出 FPS、人体身高、识别出的 Noetix 骨架类型和转换元数据。位置使用 `[x, z, y]` 基变换；朝向使用同一基变换的矩阵共轭 `R_z = S R_y S^T`，不能直接交换四元数分量。使用 `--overwrite` 可覆盖已有结果。

当前 `breaking+hippop` 实验的转换命令为：

```bash
python data_utils/convert_noetix_bvh.py \
  --input-dir demo_data/noetix_ori/0724_BEITI \
  --output-dir demo_data/noetix_mocap/0724_BEITI \
  --target-fps 30 \
  --overwrite
```

### Noetix E1 朝向跟踪与消融

朝向跟踪是独立于位置 Interaction Mesh 的软目标。它参考 GMR 的全局刚体 FrameTask 思路，为人体关节和机器人 link 建立固定帧对齐，但仍保留本项目原有的 SQP、碰撞和接触约束。每一帧使用 SO(3) 测地误差 `Log(R_target R_robot^T)`，并通过 MuJoCo 世界角速度雅可比线性化。默认 `orientation_weights={}`，因此旧配置和位置-only 求解路径保持不变；`first_frame` 对齐会在序列第一帧计算固定的人体关节到机器人 link 局部帧偏置。

可复现实验入口固定使用 `breaking+hippop.bvh_Skeleton1`，并生成 baseline、root、feet、gmr_legs、shoulders、upper 和 full 七组结果、每次运行的 manifest 以及统一 `summary.json`。其中 `gmr_legs` 对应 GMR E1 主任务实际启用旋转代价的髋、膝和足链，`shoulders` 只对 `LeftArm/RightArm → l/r_arm_shoulder_yaw_link` 加入朝向代价，其他十一个诊断 link 的权重严格为零；本项目不会直接照搬 GMR 的数值权重和固定四元数，因为两套 E1 MJCF、坐标变换和目标函数标度不同：

```bash
python examples/run_orientation_ablation.py \
  --data-path demo_data/noetix_mocap/0724_BEITI \
  --task-name 'breaking+hippop.bvh_Skeleton1' \
  --output-root demo_results_orientation/e1/robot_only/0724_BEITI/breaking+hippop \
  --variants baseline root feet gmr_legs shoulders upper full \
  --weight-scales 0.25 0.5 1.0 \
  --overwrite
```

baseline 会为相同的 13 个 link 保存朝向诊断，但所有朝向权重均为零，不向优化问题加入旋转项。结果文件保存目标/机器人 link 四元数、逐 link 测地误差、朝向代价、映射位置误差以及 SQP 诊断，可直接比较朝向改善是否以位置、收敛或约束退化为代价。可用 `--frame-start` 和 `--frame-count` 对相同输入切片做快速权重搜索，再对完整序列复核选中的配置。

转换后可直接检查源 BVH 的全局关节坐标轴：

```bash
python examples/raw_human_motion_viewer.py \
  --motion-path demo_data/noetix_mocap/0724_BEITI/breaking+hippop.bvh_Skeleton1.npz \
  --show-joint-orientations
```

也可同步对照两个重定向结果：

```bash
python examples/ablation_viser_player.py \
  --qpos-npzs \
    demo_results_orientation/e1/robot_only/0724_BEITI/breaking+hippop/baseline/breaking+hippop.bvh_Skeleton1.npz \
    demo_results_orientation/e1/robot_only/0724_BEITI/breaking+hippop/full/breaking+hippop.bvh_Skeleton1.npz \
  --labels baseline full \
  --x-offset 0.7 \
  --orientation-joints LeftArm RightArm \
  --show-orientation-error-labels
```

播放器中的短 RGB 实心箭头是标定后的目标 link frame，较长的 RGB 实心箭头是机器人实际 link frame，红、绿、蓝分别表示局部 X、Y、Z 轴；两套箭头越重合，朝向跟踪越准确。界面中的 mesh、每组结果的箭头、全局目标箭头、全局机器人箭头和 `SO(3) error labels` 均可独立开关。要只加强肩膀，可运行 `--variants baseline shoulders --weight-scales 0.01 0.025 0.05` 搜索权重，再用上面的 `--orientation-joints LeftArm RightArm` 对照显示候选结果。

### GVHMR

先运行 GVHMR 生成 `hmr4d_results.pt`。Holosoma 使用其中 `smpl_params_global` 保存的世界坐标系 SMPL-X 参数：

```bash
python data_utils/convert_gvhmr.py \
  --input-file /home/jinhaogao/GVHMR/outputs/demo/tennis/hmr4d_results.pt \
  --output-file demo_data/gvhmr/tennis.npz \
  --model-path models/smplx \
  --fps 30
```

转换器会分批执行 SMPL-X 正向运动学，将 GVHMR 的右手系 Y-up 世界坐标转换为右手系 Z-up，计算与体型对应的人体身高，并以 `wxyz` 顺序写入根节点四元数。

## 重定向前检查原始人体动作

统一的原始动作查看器可用于判断接触或姿态问题究竟已经存在于源数据中，还是由后续归一化与重定向引入：

```bash
# OMOMO 物体交互：完整 52 关节骨架和随原始位姿运动的物体 mesh
python examples/raw_human_motion_viewer.py \
  --motion-path demo_data/OMOMO_new/sub3_largebox_003.pt

# 攀爬：完整 53 关节骨架和静态原始 multi_boxes.obj
python examples/raw_human_motion_viewer.py \
  --motion-path demo_data/climb/mocap_climb_seq_0

# 没有场景 mesh 的纯人体动作
python examples/raw_human_motion_viewer.py \
  --motion-path demo_data/lafan/dance1_subject1.npy
```

查看器支持 `omomo`、`mocap`、`lafan`、`amass`、`gvhmr` 和 `noetix_mocap`，会根据所选文件自动识别数据格式，并根据数据内容及相邻资源推断任务类型。也可以同时传入数据集目录和 `--sequence`，例如 `--motion-path demo_data/OMOMO_new --sequence sub3_largebox_003`。如果希望忽略 OMOMO 文件中的物体并作为纯人体动作检查，请显式传入 `--task-type robot_only`；也可以通过 `--mesh-path PATH` 覆盖自动查找到的 mesh。

该工具会刻意绕过 `preprocess_motion_data`，不执行脚底高度平移、机器人身高缩放、数据增强或重定向。OMOMO mesh 使用原始逐帧物体位姿，攀爬 mesh 使用源文件中的静态坐标，没有对应 mesh 时只显示完整骨架。这里的“原始”指注册适配器刚输出的数据；格式所必需的解码仍会执行，例如 LAFAN 坐标转换和 mocap 声明的时间采样。加入 `--dry-run` 可以只检查并汇总场景，而不启动 Viser。

键盘操作与其他 Viser 查看器保持一致：

| 按键 | 操作 |
| --- | --- |
| `Space` | 播放或暂停 |
| `[` / `]` | 上一帧或下一帧 |
| `Home` / `End` | 第一帧或最后一帧 |
| `K` | 显示或隐藏全部人体关键点 |
| `.` | 显示或隐藏完整骨架 |
| `L` | 显示或隐藏关节名称 |
| `O` | 显示或隐藏原始物体或地形 mesh |

## 检查已保存重定向结果的可视化

新结果会保存机器人、数据格式、物体、FPS、骨架和 qpos 布局元数据，因此通常只需向查看器提供结果路径：

```bash
# OMOMO 物体交互
python viser_player.py \
  --qpos-npz demo_results/g1/object_interaction/omomo/sub3_largebox_003_original.npz

# 攀爬
python viser_player.py \
  --qpos-npz demo_results/g1/climbing/mocap/mocap_climb_seq_0_original.npz

# OMOMO 纯机器人动作
python viser_player.py \
  --qpos-npz demo_results/g1/robot_only/omomo/sub3_largebox_003.npz

# LAFAN
python viser_player.py \
  --qpos-npz demo_results/g1/robot_only/lafan/dance2_subject1.npz

# AMASS
python viser_player.py \
  --qpos-npz demo_results/g1/robot_only/amass/ACCAD_Female1Running_c3d_C3_-_Run_stageii.npz

# Noetix-mocap
python viser_player.py \
  --qpos-npz demo_results/g1/robot_only/noetix_mocap/run_to_the_right.npz

# GVHMR
python viser_player.py \
  --qpos-npz demo_results/g1/robot_only/gvhmr/tennis.npz
```

对于缺少新元数据的旧结果，继续使用官方原有的显式查看器形式：

```bash
# 物体交互
python viser_player.py \
  --robot-urdf models/g1/g1_29dof.urdf \
  --object-urdf models/largebox/largebox.urdf \
  --qpos-npz demo_results_parallel/g1/object_interaction/omomo/sub3_largebox_003_original.npz

# 原始攀爬结果
python viser_player.py \
  --robot-urdf models/g1/g1_29dof_spherehand.urdf \
  --object-urdf demo_data/climb/mocap_climb_seq_0/multi_boxes.urdf \
  --qpos-npz demo_results_parallel/g1/climbing/mocap_climb/mocap_climb_seq_0_original.npz

# 增强攀爬结果
python viser_player.py \
  --robot-urdf models/g1/g1_29dof_spherehand.urdf \
  --object-urdf demo_data/climb/mocap_climb_seq_0/multi_boxes_scaled_0.74_0.74_0.89.urdf \
  --qpos-npz demo_results_parallel/g1/climbing/mocap_climb_aug/mocap_climb_seq_0_z_scale_1.2.npz
```

### 骨架、透明度与 Interaction Mesh

在单序列或批处理重定向命令中加入 `--retargeter.save-interaction-mesh`，即可保存源和目标 Interaction Mesh。在重定向过程中实时查看：

```bash
python examples/robot_retarget.py \
  --data-path demo_data/OMOMO_new \
  --task-type object_interaction \
  --task-name sub3_largebox_003 \
  --data-format omomo \
  --retargeter.visualize \
  --retargeter.debug \
  --retargeter.mesh-opacity 0.4 \
  --retargeter.show-interaction-mesh \
  --retargeter.save-interaction-mesh \
  --retargeter.interaction-mesh-mode both \
  --retargeter.interaction-mesh-edges cross
```

回放全部叠加层：

```bash
python viser_player.py \
  --qpos-npz demo_results/g1/object_interaction/omomo/sub3_largebox_003_original.npz \
  --show-mapped-skeletons \
  --show-object-keypoints \
  --show-interaction-mesh \
  --interaction-mesh-mode both \
  --interaction-mesh-edges cross \
  --robot-mesh-opacity 0.45 \
  --object-mesh-opacity 0.25
```

同屏比较一条原始结果及其五种增强时，可以把六个结果中的任意一个传给增强查看器：

```bash
python augmentation_viser_player.py \
  --qpos-npz demo_results_parallel/g1/object_interaction/omomo/sub3_largebox_003_original.npz
```

该查看器会自动发现同目录下匹配的六个文件，用不同颜色同步显示每组机器人 mesh、人体映射骨架、机器人映射骨架和物体 mesh，并允许逐组开关。所有结果仍然保存 Interaction Mesh；为避免六组四面体边叠加遮挡场景，这个同屏查看器刻意不加载或绘制 Interaction Mesh。需要检查某一组的 Interaction Mesh 时，继续使用上面的 `viser_player.py --show-interaction-mesh` 单独回放对应 `.npz`。

OMOMO 的 `--show-mapped-skeletons` 会在原有 15 个蓝色映射关节之外，以较小的蓝色点和线补全 SMPL-H 两只手的五指关节链。这些手指关节只用于诊断显示，不会加入 Interaction Mesh，也不会改变优化结果。`--show-object-keypoints` 会显示重定向时保存的逐帧物体点：红色为随人体身高一起归一化的示范物体点，青色为目标资产尺度下的物体点。结果保存的是当次重定向实际使用的世界坐标，因此回放无需重新采样物体表面。

`--mesh-opacity` 提供共同透明度；`--robot-mesh-opacity` 和 `--object-mesh-opacity` 可分别覆盖。骨架点/线尺寸、物体点尺寸以及 Interaction Mesh 线宽分别由 `--skeleton-point-radius`、`--skeleton-line-width`、`--object-keypoint-radius` 和 `--interaction-mesh-line-width` 控制。

## 结果 NPZ 约定

新重定向结果包含 `qpos`、`fps`、`cost`、完整及映射后的人体骨架、映射机器人骨架位置，以及 `source_data_format`、`robot_type` 和物体元数据。物体交互结果还会始终保存 `object_points_demo_local`、`object_points_target_local`、`object_points_demo_world` 和 `object_points_target_world`，分别表示示范/目标尺度下的局部采样点及其逐帧世界坐标。启用 `--retargeter.save-interaction-mesh` 后，还会包含逐帧源/目标顶点和四面体。若默认 1 mm 脚部固定约束使某帧不可行，求解器会先仅对该帧使用 `foot_sticking_fallback_tolerance` 重试，再仅释放该帧的脚部固定约束；实际回退和释放帧分别保存在 `foot_sticking_fallback_frames` 与 `foot_sticking_release_frames`。若此前轨迹已经进入局部释放也无法恢复的不可行状态，则从第 0 帧开始关闭脚部固定并重跑完整序列，其他约束仍保持启用；结果通过 `foot_sticking_enabled_for_saved_trajectory` 和 `foot_sticking_full_sequence_retry_frame` 记录这一情况。机器人—物体非穿透约束默认不会被释放。只有显式启用 `--retargeter.release-object-non-penetration-on-infeasible` 时，最后一级局部回退才会释放失败帧的机器人—物体约束并继续保留地面防穿透；这类结果应视为降级结果而不是碰撞验收通过，并通过 `object_non_penetration_release_frames` 记录对应帧。

纯机器人 qpos 使用 `[root_xyz, root_wxyz, robot_dof]`；动态物体 qpos 会追加 `[object_xyz, object_wxyz]`。

## 定量评估

官方原有评估工作流继续保留：

```bash
# 机器人—物体交互
python evaluation/eval_retargeting.py \
  --res-dir demo_results_parallel/g1/object_interaction/omomo \
  --data-dir demo_data/OMOMO_new \
  --data-type robot_object

# 攀爬
python evaluation/eval_retargeting.py \
  --res-dir demo_results_parallel/g1/climbing/mocap_climb \
  --data-dir demo_data/climb \
  --data-type robot_terrain \
  --robot-config.robot-urdf-file models/g1/g1_29dof_spherehand.urdf

# OMOMO 纯机器人动作
python evaluation/eval_retargeting.py \
  --res-dir demo_results_parallel/g1/robot_only/omomo \
  --data-dir demo_data/OMOMO_new \
  --data-type robot_only
```

纯机器人评估会使用每个新结果中保存的预处理人体骨架。因此，修改 `--data-format` 和对应路径后，同一命令也可用于 LAFAN、AMASS、Noetix-mocap 和 GVHMR。
对于 OMOMO 机器人—物体结果，评估器会根据每个 NPZ 的元数据和标准文件名分别解析物体；当结果目录包含多个物体类别时，不要传入 `--object-name`。

## 为 RL 全身跟踪策略准备数据

官方工作流包含两个步骤：

1. 运行重定向，获得 `.npz` 机器人动作。
2. 将结果按目标帧率转换为全身跟踪策略所需格式。

在 macOS 上请使用 `mjpython`，而不是 `python`。

### macOS（`mjpython`）

```bash
mjpython data_conversion/convert_data_format_mj.py \
  --input-file ./demo_results/g1/robot_only/omomo/sub3_largebox_003.npz \
  --output-fps 50 \
  --output-name converted_res/robot_only/sub3_largebox_003_mj_fps50.npz \
  --data-format omomo \
  --object-name ground \
  --once

mjpython data_conversion/convert_data_format_mj.py \
  --input-file ./demo_results/g1/object_interaction/omomo/sub3_largebox_003_original.npz \
  --output-fps 50 \
  --output-name converted_res/object_interaction/sub3_largebox_003_mj_w_obj.npz \
  --data-format omomo \
  --has-dynamic-object \
  --once
```

### 纯机器人设置

```bash
python data_conversion/convert_data_format_mj.py \
  --input-file ./demo_results/g1/robot_only/omomo/sub3_largebox_003.npz \
  --output-fps 50 \
  --output-name converted_res/robot_only/sub3_largebox_003_mj_fps50.npz \
  --data-format omomo \
  --object-name ground \
  --once

python data_conversion/convert_data_format_mj.py \
  --input-file ./demo_results/g1/robot_only/lafan/dance2_subject1.npz \
  --output-fps 50 \
  --output-name converted_res/robot_only/dance2_subject1_mj_fps50.npz \
  --data-format lafan \
  --object-name ground \
  --once
```

### 机器人—物体设置

```bash
python data_conversion/convert_data_format_mj.py \
  --input-file ./demo_results/e1/object_interaction/omomo/sub10_tripod_000_original.npz \
  --robot e1 \
  --output-fps 50 \
  --output-name converted_res/object_interaction/sub10_tripod_000_mj_w_obj.npz \
  --data-format omomo \
  --has-dynamic-object \
  --once
```

OMOMO 动态物体转换会从结果元数据或标准文件名自动推断类别，并拒绝与之冲突的显式覆盖参数。

### OmniRetarget 数据

对于从 Hugging Face 下载的 OmniRetarget 数据，请增加 `--use-omniretarget-data`：

```bash
python data_conversion/convert_data_format_mj.py \
  --input-file OmniRetarget/robot-object/sub3_largebox_003_original.npz \
  --output-fps 50 \
  --output-name converted_res/object_interaction/sub3_largebox_003_mj_w_obj_omnirt.npz \
  --data-format omomo \
  --object-name largebox \
  --has-dynamic-object \
  --use-omniretarget-data \
  --once
```

## 自定义人体动作格式

参阅 [ADD_MOTION_FORMAT_README_zh.md](ADD_MOTION_FORMAT_README_zh.md)。

## 自定义机器人类型

参阅 [ADD_ROBOT_TYPE_README.md](ADD_ROBOT_TYPE_README.md)。
