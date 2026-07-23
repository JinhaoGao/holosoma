# Holosoma 动作重定向

简体中文 | [English](README.md)

本仓库提供将人体动作重定向到人形机器人的工具。官方项目原本支持 OMOMO、LAFAN、AMASS SMPL-X、通用 mocap，以及纯机器人动作、物体交互和攀爬任务。本分支在完整保留这些工作流的基础上，增加了 Noetix-mocap 与 GVHMR 的显式适配，并统一了结果保存和可视化接口。

**数据要求：**重定向流水线接收形状为 `(T, J, 3)` 的世界坐标系人体关节位置，其中 `T` 为帧数，`J` 为关节数。每种格式分别声明关节顺序、机器人映射、文件布局、坐标变换、人体身高、FPS 和支持的任务类型。如需添加其他格式，请参阅 [ADD_MOTION_FORMAT_README_zh.md](ADD_MOTION_FORMAT_README_zh.md)。

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
  --task-name sub3_largebox_003 \
  --data-format omomo \
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

# 物体交互（OMOMO）
python examples/parallel_robot_retarget.py \
  --data-dir demo_data/OMOMO_new \
  --task-type object_interaction \
  --data-format omomo \
  --save-dir demo_results_parallel/g1/object_interaction/omomo \
  --task-config.object-name largebox

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

为物体交互或攀爬批处理增加 `--augmentation` 后，会同时处理原始和增强序列。已经存在的输出文件会被跳过。

## 数据准备

仓库提供 `demo_data/` 用于快速测试。若要准备更多序列，请按以下数据集说明操作。

### OMOMO

Holosoma 官方流水线使用 InterMimic 预处理后的数据，其格式与原始 OMOMO 发布格式不同。

1. 从[此链接](https://drive.google.com/file/d/141YoPOd2DlJ4jhU2cpZO5VU5GzV_lm5j/view)下载预处理 OMOMO 数据。
2. 解压至 `demo_data/OMOMO_new`。
3. 将 `height_dict.pkl` 放在 `OMOMO_new` 的上级目录。也可以传入 `--motion-data-config.human-height HEIGHT` 显式指定身高。

动作文件应为 `.pt` 张量。

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

这是 Noetix 专用转换器，与上面的官方 LAFAN BVH 转换流程无关。输出 `.npz` 包含 Z-up 全局关节、关节名称、源/输出 FPS、人体身高、识别出的 Noetix 骨架类型和转换元数据。使用 `--overwrite` 可覆盖已有结果。

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
  --show-interaction-mesh \
  --interaction-mesh-mode both \
  --interaction-mesh-edges cross \
  --robot-mesh-opacity 0.45 \
  --object-mesh-opacity 0.25
```

`--mesh-opacity` 提供共同透明度；`--robot-mesh-opacity` 和 `--object-mesh-opacity` 可分别覆盖。骨架点/线尺寸以及 Interaction Mesh 线宽分别由 `--skeleton-point-radius`、`--skeleton-line-width` 和 `--interaction-mesh-line-width` 控制。

## 结果 NPZ 约定

新重定向结果包含 `qpos`、`fps`、`cost`、完整及映射后的人体骨架、映射机器人骨架位置，以及 `source_data_format`、`robot_type` 和物体元数据。启用 `--retargeter.save-interaction-mesh` 后，还会包含逐帧源/目标顶点和四面体。

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
  --object-name largebox \
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
  --input-file ./demo_results/g1/object_interaction/omomo/sub3_largebox_003_original.npz \
  --output-fps 50 \
  --output-name converted_res/object_interaction/sub3_largebox_003_mj_w_obj.npz \
  --data-format omomo \
  --object-name largebox \
  --has-dynamic-object \
  --once
```

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
