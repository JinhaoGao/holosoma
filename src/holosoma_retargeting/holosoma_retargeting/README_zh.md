# Holosoma 动作重定向

简体中文 | [English](README.md)

本包通过同一套加载、预处理、求解、保存、续跑和可视化契约，把 OMOMO、LAFAN、AMASS SMPL-X、GVHMR、Noetix BVH 与攀爬 mocap 重定向到人形机器人。公开重定向入口分为三类：`examples/robot_retarget.py` 处理单一动作，`examples/parallel_robot_retarget.py` 处理数据集批量任务或增强动作族，`examples/` 下名称含 `ablation`、`comparison` 或 `search` 的脚本处理受控消融实验。`examples/rebuild_demo_results.py` 只是经过完整验证的 G1/E1 演示数据矩阵编排器，不是第四种求解器或 artifact 类型。持久化的 `run_kind` 只有明确的语义：所有 identity 结果均为 `single`，只有非 identity 的动作变换为 `augmentation`，未做动作变换的实验 variant 为 `ablation`；批量执行本身只是编排方式，绝不会保存为单独的 `run_kind`。

流水线始终接收形状为 `(T, J, 3)` 的世界坐标人体关节位置。源人体朝向遵循更严格的规则：只有源表示直接包含旋转，而且适配器无需根据位置推测便可把这些旋转转换为世界坐标 `wxyz` 四元数时，结果才保存人体朝向；源数据也可以只为一部分具名关节提供朝向。若没有直接旋转，artifact 会记录 `orientation_source="absent"`，并省略包含 tensor digest 在内的整组源人体朝向字段。机器人初始化过程中内部估计的朝向绝不会伪装成源数据元数据。因此，`demo_data/climb` 中只有位置的通用攀爬数据不会保存人体朝向。

每种格式分别声明关节顺序与父树、机器人映射、文件布局、坐标变换、人体身高、FPS、支持的任务类型和允许的直接朝向来源。如需添加其他格式，请参阅 [ADD_MOTION_FORMAT_README_zh.md](ADD_MOTION_FORMAT_README_zh.md)。

请在以下目录中执行本文命令：

```bash
cd src/holosoma_retargeting/holosoma_retargeting
```

## 支持的人体动作数据与任务

| `--data-format` | 标准输入 | 任务类型 | 直接源人体朝向 |
| --- | --- | --- | --- |
| `omomo` | `.pt`，52 个 SMPL-H 关节 | `robot_only`、`object_interaction` | 第 383:591 列存在时为 `intermimic_global_orientation_tensor`，否则为 `absent` |
| `lafan` | `.npz`，22 个 LAFAN 关节 | `robot_only` | `bvh_rotation_channels_fk` |
| `amass` | `.npz`，22 个 SMPL-X 关节 | `robot_only` | `direct_local_rotation_fk` |
| `gvhmr` | `.npz`，22 个 SMPL-X 关节 | `robot_only` | `direct_local_rotation_fk` |
| `noetix_mocap` | `.npz`，标准化 22 关节 Noetix 格式 | `robot_only` | `bvh_rotation_channels_fk`；保存的关节子集可能不完整 |
| `mocap` | 嵌套 `.npz` 或 `.npy`，53 个关节 | `robot_only`、`climbing` | Noetix CSV 为 `bone_rotation_channels_fk`；通用攀爬为 `absent` |

表中只使用标准格式名称。加载器仍可能接受兼容别名，但新命令、路径、元数据和文档只使用标准名称。

> **LAFAN 与 Noetix-mocap 是两套相互独立的数据。**LAFAN 是 Ubisoft 公开数据集，会从原始 BVH 旋转通道转换为标准 `.npz` 契约；Noetix-mocap 是公司自采动捕数据，包含多种 Noetix BVH 导出骨架，使用独立的 `BVH → .npz` 流程。Noetix 标准化骨架只是在重定向层复用了兼容的 22 关节命名拓扑，这不代表 Noetix 源数据属于 LAFAN。

## 单序列动作重定向

`robot_retarget.py` 是唯一的单动作入口。它通过与批处理完全相同的适配器注册表解析源文件，统一嵌套的机器人、数据和任务配置，构造版本化 job，并且只有在 schema、源路径与 SHA-256、标准化配置、run kind 和 variant 全部相符时才续用已有 artifact；标准路径为空时才会执行共享求解生命周期。若同路径已有身份不匹配或无效的文件，命令会拒绝静默替换，只有明确传入 `--overwrite-existing` 才会覆盖。不要把 `--save-dir` 指向机器人或任务叶目录；该参数表示结果根目录。省略时使用 `demo_results/v1`。

```bash
# G1 上的 LAFAN 纯机器人动作
python examples/robot_retarget.py \
  --data-path demo_data/lafan \
  --task-type robot_only \
  --task-name dance2_subject1 \
  --data-format lafan \
  --robot g1

# E1 上的 OMOMO 物体交互
python examples/robot_retarget.py \
  --data-path demo_data/OMOMO_new \
  --task-type object_interaction \
  --task-name sub10_tripod_000 \
  --data-format omomo \
  --robot e1

# G1 上的通用攀爬
python examples/robot_retarget.py \
  --data-path demo_data/climb \
  --task-type climbing \
  --task-name mocap_climb_seq_0 \
  --data-format mocap \
  --robot g1
```

默认 artifact 路径为：

```text
demo_results/v1/canonical/<robot>/<task_type>/<data_format>/
  <dataset_partition>/<sequence_key>/<variant>.npz
```

例如第一条命令写入 `demo_results/v1/canonical/g1/robot_only/lafan/lafan/dance2_subject1/identity.npz`。批处理中嵌套的 sequence key 会继续保持嵌套，源名称中的不安全路径字符会做百分号编码。OMOMO 会从标准 `subN_object_NNN` 名称推断物体；显式传入 `--task-config.object-name` 时，它只作为一致性检查。目录名 `v1` 表示结果树的 release namespace，目录内 NPZ 使用 schema version 2，两者是相互独立的版本号。

每个 schema-v2 artifact 默认包含完整预处理人体骨架、完整机器人 link 位姿与父树、actuated joint 顺序、Interaction Mesh、求解诊断以及源/配置身份。只有适配器能证明原始数据存在获准的直接朝向字段时才保存人体朝向，并保留精确的具名子集和 tensor digest；仅位置数据完全省略该可选组。非 ground 结果还通过 `object_asset_manifest_json` 与 `object_asset_manifest_sha256` 固定对象 URDF 及其直接引用的全部 mesh/texture。精确续跑、staging 验证、提升和 strict 可视化都会重新计算该闭包，任一文件缺失或变化都会闭锁失败。canonical `qpos` 始终以 `float64` 保存，保留真实几何门禁最终接受的精确 SQP 状态，也保留增强任务复用 nominal warm start 所需的精度。查看器和下游转换器可以在载入后按需转换副本，但 canonical artifact 不能主动丢弃这部分精度。

为物体交互或攀爬命令加入 `--augmentation` 后，同一个入口会运行完整的 identity-first 动作族。OMOMO 生成 `identity`、`trans_0`、`trans_1`、`trans_2`、`rot_0` 和 `rot_1`；攀爬生成 `identity`、`z_scale_0p8`、`z_scale_0p9`、`z_scale_1p1` 和 `z_scale_1p2`；纯机器人任务仍只有 identity。任何增强 variant 在复用名义轨迹前，都会先创建或严格验证同一源与配置的 identity artifact。

## 批量动作重定向

`parallel_robot_retarget.py` 是唯一的多源与批量增强入口，与单动作命令共享 job builder、标准路径、schema writer 和严格续跑检查。`--data-dir` 选择数据集根，`--max-workers` 控制进程并行数，`--dry-run` 只写清单而不求解，`--overwrite-existing` 关闭严格续跑。默认结果根同样为 `demo_results/v1`；未显式覆盖时，批处理报告写入 `<save_dir>/runs/<run_id>/report.json`。

```bash
# G1 上全部 GVHMR
python examples/parallel_robot_retarget.py \
  --data-dir demo_data/gvhmr \
  --task-type robot_only \
  --data-format gvhmr \
  --robot g1 \
  --max-workers 4 \
  --run-id g1-gvhmr

# E1 上全部 OMOMO 物体交互及其增强
python examples/parallel_robot_retarget.py \
  --data-dir demo_data/OMOMO_new \
  --task-type object_interaction \
  --data-format omomo \
  --robot e1 \
  --augmentation \
  --max-workers 4 \
  --run-id e1-omomo-object

# 通用攀爬及四种缩放增强
python examples/parallel_robot_retarget.py \
  --data-dir demo_data/climb \
  --task-type climbing \
  --data-format mocap \
  --robot g1 \
  --augmentation \
  --max-workers 4 \
  --run-id g1-generic-climb
```

只需修改 `--data-format` 与 `--data-dir`，同一命令即可处理 `lafan`、`amass`、`noetix_mocap` 和 `gvhmr`。已有 artifact 只有在 schema 与源/配置身份完全匹配时才会续用；同名文件无效、源数据已变或配置已变时会显式拒绝，而不是静默重算，只有 `--overwrite-existing` 会授权替换。OMOMO 批处理默认先检查动作数据、受试者身高和物体资产。物体类别过滤是显式的：

```bash
python examples/parallel_robot_retarget.py \
  --data-dir demo_data/OMOMO_new \
  --task-type object_interaction \
  --data-format omomo \
  --robot g1 \
  --object-names largebox \
  --max-workers 4 \
  --augmentation \
  --overwrite-existing \
  --run-id g1-largebox
```

目前物体目录完整支持 `clothesstand`、`floorlamp`、`largebox`、`largetable`、`monitor`、`plasticbox`、`smallbox`、`smalltable`、`suitcase`、`trashcan`、`tripod`、`whitechair` 和 `woodchair`。省略 `--object-names` 会处理全部类别，也可以精确选择子集。建议先用 `--dry-run` 验证全部输入并生成清单，此时不会启动优化：

```bash
python examples/parallel_robot_retarget.py \
  --data-dir demo_data/OMOMO_new \
  --task-type object_interaction \
  --data-format omomo \
  --robot g1 \
  --dry-run \
  --run-id g1-omomo-plan
```

## 消融与权重搜索入口

受控实验同样构造 `RetargetJob`，并把 schema-v2 artifact 写入 `demo_results/v1/ablations/<experiment>/...`，不会维护第二套求解或保存实现。`run_orientation_ablation.py` 会把格式无关的语义身体角色映射到所选数据格式的直接朝向关节，并对单个动作和所选机器人运行具名 profile；`search_orientation_weights.py` 在多个时间窗口上评估 JSON 候选权重，`run_orientation_comparison_batch.py` 对整个 Noetix 集合比较零权重与已确认的 `balanced_optimal` profile。manifest 仍与对应消融 artifact 放在同一实验树中；单动作朝向消融的 summary 使用不会相互覆盖的作用域路径 `demo_results/v1/ablations/orientation_ablation/_summaries/<robot>/<task>/<format>/<dataset>/<sequence>/summary.json`，并发运行不同机器人、任务、格式、数据集或序列不会再争用一个全局 `summary.json`。

这里的 `<dataset>` 由可读的数据集目录名和其规范根路径的完整 SHA-256 组成，原始序列名的每个分量则交给共享的可逆百分号编码，因此 `walk+a` 与 `walk a` 不会重合，不同父目录下同名的数据集也不会重合。帧窗口缓存与朝向批量对比缓存会同时记录规范 lineage JSON 及其哈希、完整具名数组 payload 的哈希；每次复用前都会从正式源数据重新派生预期 payload，并逐项核对字段名、dtype、shape 与 C-order 字节，自洽修改元数据也会触发原子重建。启用 `--fail-fast` 时，两类入口都会先写出 terminal `failed` 汇总，明确记录已完成、已取消与尚未提交的工作，再抛出异常。批量对比报告另按数据集作用域写入 `demo_results/v1/ablations/orientation_comparison/_summaries/dataset-path-sha256-<digest>/batch_summary.json`。

```bash
python examples/run_orientation_ablation.py \
  --robot e1 \
  --task-type robot_only \
  --data-format noetix_mocap \
  --data-path demo_data/noetix_mocap/0724_BEITI \
  --task-name breaking+hippop.bvh_Skeleton1 \
  --output-root demo_results/v1 \
  --variants baseline balanced_optimal \
  --dry-run

python examples/search_orientation_weights.py \
  --candidate-file examples/orientation_weight_candidates_finalists.json \
  --output-root demo_results/v1
```

Noetix 的 G1/E1 默认使用共享且经过标定的 `t_pose` 对齐。AMASS、GVHMR 与 OMOMO 目前没有完整的共享 T-pose 标定，因此非零 profile 及其保留同组诊断的零权重 baseline 都必须显式传入 `--orientation-alignment-mode first_frame`。该模式只使用适配器证明为直接来源的人体四元数和机器人 FK，绝不会从位置、骨向量或脚尖方向估计朝向。profile 角色会转换为所选格式的源关节名，机器人 link 则沿用共享的格式/机器人 mapping。完整输入始终由运动格式注册表解析真实扩展名；帧窗口仅允许用于能够保留必要元数据的 NPZ 适配器，其他格式会明确要求运行完整源。

例如，可按下列方式只读规划 AMASS/G1 的同一组语义 profile：

```bash
python examples/run_orientation_ablation.py \
  --robot g1 \
  --task-type robot_only \
  --data-format amass \
  --data-path demo_data/amass_smplx_processed \
  --task-name ACCAD_Female1Running_c3d_C3_-_Run_stageii \
  --output-root demo_results/v1 \
  --orientation-alignment-mode first_frame \
  --variants baseline full \
  --dry-run
```

## 正式全量演示结果重建

`rebuild_demo_results.py` 是正式矩阵编排器。它会在启动求解器前规划 G1 与 E1 上的 OMOMO 纯机器人、OMOMO 物体交互及增强、AMASS、GVHMR、LAFAN、Noetix BVH、通用攀爬及增强、Noetix CSV 攀爬及增强。G1 是全量严格发布目标：每一个已规划源和 variant 都必须存在，并通过 schema、源/配置身份、完整人体与机器人父树、直接人体朝向应存在或应缺失、Interaction Mesh、最终接受的 float64 qpos、六项真实几何残差、三项最终 ConstraintMode 数组、资产闭包与释放质量门禁。

E1 会对相同矩阵真实发起任务，但其形态可能让部分物体交互和攀爬动作在结构上无法到达。只有实际执行过对应 E1 job，并把有界失败记录持久化到汇总报告后，才允许把该动作登记为结构性缺口。报告必须分别给出成功 artifact 与显式缺口，并保留源、job 身份、任务族和失败原因；不得伪造成功 artifact、静默跳过输入、估计缺失元数据或放宽 schema 与硬约束验收。E1 的 robot-only 类仍作为预期高覆盖子集。因此，提升要求 G1 全矩阵严格完整，同时要求 E1 矩阵的每项都有真实尝试和明确结论：成功项严格通过，获准的结构性缺口可以审计。

```bash
# 只读生成完整计划
python examples/rebuild_demo_results.py \
  --run-id demo-rebuild-v1 \
  --max-workers 24 \
  --dry-run

# 续跑或执行 staging 矩阵，验证后事务性提升
python examples/rebuild_demo_results.py \
  --run-id demo-rebuild-v1 \
  --max-workers 24 \
  --promote

# 不启动求解器，只重新验证现有 staging
python examples/rebuild_demo_results.py \
  --run-id demo-rebuild-v1 \
  --validate-only
```

稳定 staging 路径为 `demo_results/.staging/<run_id>/v1`，汇总报告为 `demo_results/runs/<run_id>/report.json`，提升成功后 staging 会重命名为 `demo_results/v1`。若正式 `v1` 已存在，会归档到 `demo_results/archive/<UTC timestamp>/v1`。提升阶段持有共享的 results-state 独占锁，在锁内重新核对 staging 内容 digest 和全部外部对象资产闭包，并通过原子目录交换保持正式树连续可见；若文件系统不支持原子交换，则不移动 staging 或正式树并直接拒绝提升。`demo_results/.locks/rebuild-<run_id>.lock` 从规划一直覆盖到最终报告，禁止两个编排器共用同一 staging/report 身份。生成的机器人与物体场景资产独立保存在 `demo_results/.generated-assets/jobs`，逐 job 并发锁保存在 `demo_results/.locks/retargeting`，二者都不会混入可提升结果树。

## 整理旧结果目录

完整验证并提升 `demo_results/v1` 后，使用专用整理器只归档仓库明确识别的旧布局。默认是只读 dry-run，会输出精确的目录清单、大小摘要与目标位置：

```bash
python examples/organize_demo_results.py \
  --promoted-run-id demo-rebuild-v1
```

核对计划后，再显式执行重命名事务：

```bash
python examples/organize_demo_results.py \
  --promoted-run-id demo-rebuild-v1 \
  --execute
```

该命令处理 `demo_results` 内旧的按机器人目录和固定的旧 sibling 结果目录。`--promoted-run-id` 会把事务绑定到成功的重建提升报告和当前正式 `v1` 的内容指纹。整理器把持久 manifest 写入 `demo_results/archive/legacy/<UTC timestamp>/manifest.json`，拒绝未知顶层条目与符号链接，并在修改前再次确认包含文件内容的树快照未变化。执行若被中断，应使用相同固定时间戳加 `--resume --execute` 续跑，manifest 会逐项核对 source/target 重命名而不猜测状态。执行阶段持有与 promotion 共用的 results-state 独占锁。它不会触碰 `v1`、`runs`、`.staging`、`.generated-assets`、`.locks` 或既有 archive。

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
  --output demo_results/runs/omomo_preflight.json
```

发布前可以让 G1/E1 与每类物体各执行两帧真实优化。该命令会生成并校验 26 个结果 NPZ：

```bash
python data_utils/validate_omomo_retargeting.py demo_data/OMOMO_new \
  --robots g1 e1 \
  --frames 2 \
  --output-dir demo_results/runs/omomo_g1_e1_validation
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

该脚本会把每个 BVH 转换为标准 Z-up `.npz`，其中包含全局关节位置、由 FK 将 22 个直接 BVH 旋转通道转换得到的全局 `wxyz` 四元数、关节名称与父索引、FPS、身高、源路径、坐标约定以及 `orientation_source="bvh_rotation_channels_fk"`。不会根据关节位置估计朝向。LAFAN 数据通常需要使用 `--retargeter.foot-sticking-tolerance 0.02` 放宽足部固定约束，并可根据动作质量继续调整。

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

转换器递归写出包含全局关节位置、人体身高、FPS、关节名称与父树，以及由源 SMPL-X 局部轴角旋转经 FK 得到的全部 22 个全局 `wxyz` 四元数的 `.npz` 文件，朝向来源为 `orientation_source="direct_local_rotation_fk"`。使用 `--subdataset-folder HumanEva` 可只处理一个子数据集；省略该参数则处理全部子数据集。

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

`mocap` 适配器会将动作从 120 FPS 采样至 30 FPS。这些通用 climb 文件只有位置，因此结果不保存源人体朝向数组；地形资源继续保存在对应序列目录中。转换后的 Noetix CSV 攀爬目录复用同一个 53 关节 `mocap` 任务接口，但会保留直接 `Bone Rotation` 通道，并记录 `orientation_source="bone_rotation_channels_fk"`。

### Noetix-mocap

将受支持的公司自采 Noetix BVH 骨架变体转换为标准化 22 关节 Noetix 重定向格式：

```bash
python data_utils/convert_noetix_bvh.py \
  --input-dir demo_data/noetix_ori \
  --output-dir demo_data/noetix_mocap \
  --target-fps 30 \
  --overwrite
```

这是独立于 LAFAN 的 Noetix 专用转换器。输出 `.npz` 包含 Z-up `global_joint_positions`、成对的 `orientation_joint_names`/`orientation_quaternions_wxyz`、关节名称与父树、源/输出 FPS、人体身高、识别出的骨架类型、源路径以及 `orientation_source="bvh_rotation_channels_fk"`。朝向子集只包含由源旋转通道直接支持的标准关节；仅由位置合成的缺失标准关节不会获得推测朝向。位置使用 `[x, z, y]` 基变换，旋转使用对应矩阵共轭 `R_z = S R_y S^T`。使用 `--overwrite` 可覆盖已有结果。

当前 `breaking+hippop` 实验的转换命令为：

```bash
python data_utils/convert_noetix_bvh.py \
  --input-dir demo_data/noetix_ori/0724_BEITI \
  --output-dir demo_data/noetix_mocap/0724_BEITI \
  --target-fps 30 \
  --overwrite
```

### Noetix E1 朝向跟踪与消融

朝向跟踪是独立于位置 Interaction Mesh 的软目标。它参考 GMR 的全局刚体 FrameTask 和逐 link 固定旋转偏置，为人体关节和机器人 link 建立 T-pose 帧对齐，同时保留本项目原有的 SQP、碰撞和接触约束。默认的 `t_pose` 模式使用 `R_align = R_human,T^T R_robot,T`，每帧目标为 `R_target(t) = R_human(t) R_align`，再最小化 SO(3) 测地误差 `Log(R_target R_robot^T)`。Noetix 零旋转 T-pose 的人体局部帧为单位阵，但骨架解剖前向是世界 `+Y`；E1 零位模型前向是 `+X` 且双臂自然下垂，所以机器人 T-pose 参考根节点先绕世界 Z 轴旋转 `+90°`，左、右肩 roll 再分别设为 `+π/2` 和 `−π/2`，由此形成真正的几何 T-pose，而不是直接把 E1 的 qpos0 当成 T-pose。随后通过实际 MJCF 正向运动学为全部 15 个 link 推导偏置，肘部和手部还会自动包含 E1 模型固定的 link frame 旋转。Noetix G1/E1 继续默认采用该共享 `t_pose`；缺少完整共享 T-pose 标定的格式必须显式选择 `first_frame`，并把该选择写入 manifest。该模式只使用直接源四元数和机器人 FK，不使用位置，但动作第一帧与几何标定姿态仍具有不同的实验语义。

`shoulders_feet` 消融配置会为 `LeftArm`、`RightArm`、`LeftFoot` 和 `RightFoot` 设置相同的指定权重。在 E1 上它们分别对应左右肩 yaw link 和左右脚 ankle-roll link，其他朝向权重保持为零。

`full_equal` 配置会用完全相同的指定权重跟踪全部 15 个 link。它与已有的 `full` 不同，后者为腿、脚、前臂和手保留了不相同的相对系数。

`balanced_optimal` 配置将全部 15 个朝向权重固定为 `0.085`。该值来自对 2660 帧 `breaking+hippop.bvh_Skeleton1` 的并行粗搜索、分组搜索、完整序列复核和局部精确搜索，在位置与朝向等权的归一化判据下取得最低综合分。完整数学目标、Pareto 分析和测量结果见 [orientation_weight_search_report_zh.md](examples/orientation_weight_search_report_zh.md)。

可复现实验入口固定使用 `breaking+hippop.bvh_Skeleton1`，并生成 baseline、root、feet、gmr_legs、shoulders、upper 和 full 七组结果、每次运行的 manifest，以及上述按机器人、任务、格式、数据集和序列划分的 summary。其中 `gmr_legs` 对应 GMR E1 主任务实际启用旋转代价的髋、膝和足链，`shoulders` 只对 `LeftArm/RightArm → l/r_arm_shoulder_yaw_link` 加入朝向代价，其他十一个诊断 link 的权重严格为零。本项目沿用 GMR 的逐 link 偏置设计，但从 Noetix T-pose 和本项目实际 E1 MJCF 自动推导偏置，不照搬另一套模型的四元数或数值权重：

```bash
python examples/run_orientation_ablation.py \
  --robot e1 \
  --task-type robot_only \
  --data-format noetix_mocap \
  --data-path demo_data/noetix_mocap/0724_BEITI \
  --task-name 'breaking+hippop.bvh_Skeleton1' \
  --output-root demo_results/v1 \
  --variants baseline root feet gmr_legs shoulders upper full \
  --weight-scales 0.25 0.5 1.0 \
  --overwrite
```

baseline 会为相同的 15 个 link 保存朝向诊断，但所有朝向权重均为零，不向优化问题加入旋转项。结果文件保存目标/机器人 link 四元数、逐 link 测地误差、朝向代价、映射位置误差以及 SQP 诊断，可直接比较朝向改善是否以位置、收敛或约束退化为代价。对明确支持且能保留元数据的 NPZ 格式，可用 `--frame-start` 和 `--frame-count` 对相同输入切片做快速权重搜索，再对完整序列复核选中的配置。

转换后可直接检查源 BVH 的全局关节坐标轴：

```bash
python viser_player.py \
  --input-path demo_data/noetix_mocap/0724_BEITI/breaking+hippop.bvh_Skeleton1.npz \
  --input-kind raw \
  --show-source-orientation-axes
```

也可同步对照两个重定向结果：

```bash
python multi_viser_player.py \
  --qpos-npzs \
    demo_results/v1/ablations/orientation_ablation/baseline/e1/robot_only/noetix_mocap/0724_BEITI--path-sha256-<dataset-root-digest>/breaking%2Bhippop.bvh_Skeleton1/identity.npz \
    demo_results/v1/ablations/orientation_ablation/full/e1/robot_only/noetix_mocap/0724_BEITI--path-sha256-<dataset-root-digest>/breaking%2Bhippop.bvh_Skeleton1/identity.npz \
  --labels baseline full \
  --x-offset 0.7 \
  --orientation-joints LeftArm RightArm \
  --show-target-orientation-axes \
  --show-robot-orientation-axes
```

请用生成汇总里 `dataset_partition` 字段的 digest 后缀替换 `<dataset-root-digest>`；每个 artifact 旁的 manifest 是最终路径的权威记录。

播放器将原始人体参考与各个机器人结果作为独立显示层同步播放。公共深灰色人体层读取 artifact 中保存的完整源人体轨迹与父树，标定后的目标 link 坐标轴仍锚定在对应的映射目标关节上；每个机器人结果层包含低饱和、低透明度的浅色 mesh、artifact 中保存的完整机器人 link 骨架，以及朝向和原点都来自实际机器人 link 位姿的坐标轴。结果采用色盲友好的蓝、朱红、蓝绿色配色，显示层可以只为减少遮挡而省略冗余的肩部和髋部横向连杆，层间由 `--x-offset` 分开。统一界面的 `Layers` 选项卡控制全局图层，`Motions` 选项卡控制每个结果的可见性、mesh 和 skeleton；红、绿、蓝箭头分别表示局部 X、Y、Z 轴。要只加强肩膀，可运行 `--variants baseline shoulders --weight-scales 0.01 0.025 0.05` 搜索权重，再用上面的 `--orientation-joints LeftArm RightArm` 对照显示候选结果。

### GVHMR

先运行 GVHMR 生成 `hmr4d_results.pt`。Holosoma 使用其中 `smpl_params_global` 保存的世界坐标系 SMPL-X 参数：

```bash
python data_utils/convert_gvhmr.py \
  --input-file /home/jinhaogao/GVHMR/outputs/demo/tennis/hmr4d_results.pt \
  --output-file demo_data/gvhmr/tennis.npz \
  --model-path models/smplx \
  --fps 30
```

转换器会分批执行 SMPL-X 正向运动学，将 GVHMR 的右手系 Y-up 世界坐标转换为右手系 Z-up，计算与体型对应的人体身高，并从源局部轴角旋转写入全部 22 个全局 `wxyz` 四元数，朝向来源为 `orientation_source="direct_local_rotation_fk"`。

## 重定向前检查原始人体动作

统一的原始动作查看器可用于判断接触或姿态问题究竟已经存在于源数据中，还是由后续归一化与重定向引入：

```bash
# OMOMO 物体交互：完整 52 关节骨架和随原始位姿运动的物体 mesh
python viser_player.py \
  --input-path demo_data/OMOMO_new/sub3_largebox_003.pt \
  --input-kind raw

# 攀爬：完整 53 关节骨架和静态原始 multi_boxes.obj
python viser_player.py \
  --input-path demo_data/climb/mocap_climb_seq_0 \
  --input-kind raw

# 没有场景 mesh 的纯人体动作
python viser_player.py \
  --input-path demo_data/lafan/dance1_subject1.npz \
  --input-kind raw
```

查看器支持 `omomo`、`mocap`、`lafan`、`amass`、`gvhmr` 和 `noetix_mocap`，会根据所选文件自动识别数据格式，并根据数据内容及相邻资源推断任务类型。也可以同时传入数据集目录和 `--sequence`，例如 `--input-path demo_data/OMOMO_new --sequence sub3_largebox_003`。如果希望忽略 OMOMO 文件中的物体并作为纯人体动作检查，请显式传入 `--task-type robot_only`；也可以通过 `--source-mesh PATH` 覆盖自动查找到的 mesh。

该工具会刻意绕过 `preprocess_motion_data`，不执行脚底高度平移、机器人身高缩放、数据增强或重定向。OMOMO mesh 使用原始逐帧物体位姿，攀爬 mesh 使用源文件中的静态坐标，没有对应 mesh 时只显示完整骨架。这里的“原始”指注册适配器刚输出的数据；格式所必需的解码仍会执行，例如 LAFAN 坐标转换和 mocap 声明的时间采样。加入 `--dry-run` 可以只检查并汇总场景，而不启动 Viser。

键盘操作与其他 Viser 查看器保持一致：

| 按键 | 操作 |
| --- | --- |
| `Space` | 播放或暂停 |
| `[` / `]` | 上一帧或下一帧 |
| `Home` / `End` | 第一帧或最后一帧 |
| `H` | 显示或隐藏完整人体骨架与关键点 |
| `R` | 显示或隐藏源关节姿态轴 |
| `L` | 显示或隐藏关节名称 |
| `O` | 显示或隐藏原始物体或地形 mesh |

所有单动作输入适配器都使用相同的 `Playback`、`Layers` 和 `Style` 选项卡。输入中不存在的图层会保留在界面中并显示为禁用状态，同时说明缺失原因，因此 result、raw-human 和 converted-motion 的控制名称保持一致。

## 检查已保存重定向结果的可视化

`viser_player.py` 是统一的单动作查看器，可载入 result、raw-human 与 converted 输入；`multi_viser_player.py` 是统一的同步多动作查看器，可接收显式结果集合或自动发现增强动作族。两者共享 schema-v2 loader、兼容性检查与图层语义，因此重定向始终保存完整契约，而可视化阶段再选择显示哪些人体/机器人骨架、mesh、关键点、Interaction Mesh、约束状态和朝向轴。

schema-v2 结果会保存足以恢复机器人、数据格式、物体、FPS、拓扑、位姿与 qpos 布局的元数据，因此查看器通常只需要结果路径：

```bash
# OMOMO 物体交互
python viser_player.py \
  --input-path demo_results/v1/canonical/g1/object_interaction/omomo/OMOMO_new/sub3_largebox_003/identity.npz

# 攀爬
python viser_player.py \
  --input-path demo_results/v1/canonical/g1/climbing/mocap/climb/mocap_climb_seq_0/identity.npz

# LAFAN
python viser_player.py \
  --input-path demo_results/v1/canonical/g1/robot_only/lafan/lafan/dance2_subject1/identity.npz

# Noetix-mocap
python viser_player.py \
  --input-path demo_results/v1/canonical/g1/robot_only/noetix_mocap/noetix_mocap/run_to_the_right/identity.npz
```

### 骨架、透明度与 Interaction Mesh

单动作、批量/增强和消融接口写出的每个 canonical schema-v2 artifact 都必须保存源与目标 Interaction Mesh；共享配置校验会拒绝关闭 `save_interaction_mesh`。在重定向过程中实时查看：

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
  --retargeter.interaction-mesh-mode both \
  --retargeter.interaction-mesh-edges cross
```

回放全部叠加层：

```bash
python viser_player.py \
  --input-path demo_results/v1/canonical/g1/object_interaction/omomo/OMOMO_new/sub3_largebox_003/identity.npz \
  --show-human-skeleton \
  --show-robot-skeleton \
  --show-object-keypoints \
  --show-interaction-mesh \
  --interaction-mesh-mode both \
  --interaction-mesh-edges cross \
  --robot-mesh-opacity 0.45 \
  --object-mesh-opacity 0.25
```

同屏比较 identity 及五种 OMOMO 增强时，可以把六个同级 variant 中的任意一个传给多动作查看器：

```bash
python multi_viser_player.py \
  --family demo_results/v1/canonical/g1/object_interaction/omomo/OMOMO_new/sub3_largebox_003/identity.npz
```

该查看器会自动发现同目录下匹配的六个文件，围绕同一份源人体参考，用不同颜色同步显示每组机器人 mesh、artifact 中保存的完整机器人 link 骨架和物体 mesh，并提供统一的全局图层开关与逐动作开关。物体关键点、Interaction Mesh、foot sticking 和姿态轴与单动作查看器使用相同的图层名称；为避免多组内容遮挡场景，这些诊断层在多动作模式下默认关闭，但可随时从 `Layers` 选项卡启用。同步之前，共享加载器会逐个验证 schema-v2，并要求源路径与 digest、dataset identity、机器人/格式/任务身份、完整人体与机器人拓扑、actuated joint 顺序及直接人体朝向 tensor identity 一致。

OMOMO 的 `--show-human-skeleton` 会读取已保存的完整 52 关节 SMPL-H 轨迹，而不会把参与位置目标的映射子集误当成完整人体骨架。手部细节开关只影响诊断显示，不会把关节加入 Interaction Mesh，也不会改变优化结果。`--show-object-keypoints` 会显示重定向时保存的逐帧物体点：红色为随人体身高一起归一化的示范物体点，青色为目标资产尺度下的物体点。结果保存的是当次重定向实际使用的世界坐标，因此回放无需重新采样物体表面。

`--mesh-opacity` 提供共同透明度；`--robot-mesh-opacity` 和 `--object-mesh-opacity` 可分别覆盖。骨架点/线尺寸、物体点尺寸以及 Interaction Mesh 线宽分别由 `--skeleton-point-radius`、`--skeleton-line-width`、`--object-keypoint-radius` 和 `--interaction-mesh-line-width` 控制。

## 结果 NPZ 约定

每个 schema-v2 artifact 都包含 `qpos`、FPS 与求解诊断；通过 `human_joints`、`human_joint_names` 和 `human_joint_parent_indices` 保存完整的预处理人体骨架；保存位置目标使用的人体与机器人映射关键点；并通过 `robot_link_positions`、`robot_link_quaternions_wxyz`、`robot_link_names` 和 `robot_link_parent_indices` 保存完整机器人子树。机器人位姿是逐帧精确 FK 结果，不需要查看器事后重算。canonical `qpos` 的 dtype 始终是 `float64`：每一行就是与已保存 cost、残差和 ConstraintMode 对应的最终接受 SQP 状态，增强 variant 也以相同精度把这些状态复用为 nominal warm start。可视化和下游转换可以按需转换已载入的副本，但 schema-v2 writer 不能把 canonical qpos 转成 float32。`robot_actuated_joint_names`、坐标与四元数约定、`qpos_layout`、`object_poses_demo`、`object_poses_target`、`object_pose_layout`、源/配置 hash、variant、run kind、人体缩放和预处理来源使结果具备完整自描述能力。标准化 `config_json` 身份还会指纹化求解实现与运行时、机器人模型树，以及本次任务实际使用的身高表、物体资产或攀爬场景资产，因此代码或静态资产变化不会被续跑逻辑误判为同一 job。

源人体朝向是可选但必须整体出现的字段组：`human_orientation_joint_names`、`human_orientation_quaternions_wxyz` 与 `human_orientation_sha256`。它只为源数据直接观测到的具名关节子集保存，并带有经过允许列表验证的 `orientation_source`；digest 同时绑定关节顺序和精确的 float32 四元数 tensor 字节。若没有直接旋转，则完全省略这组字段，同时令 `orientation_source` 为 `absent`。因此通用 climb 不含源人体朝向组，而标准 Noetix CSV 攀爬包含。完整机器人 link 朝向与朝向优化诊断相互独立：schema 始终保留诊断字段组，没有可直接支持的 joint/link 目标时使用零宽数组。

源人体朝向组记录的是来源事实，不是增强后的优化目标。identity 与所有 augmentation artifact 都原样保留适配器给出的具名 float32 源 tensor 及其 digest。若物体旋转增强改变了朝向目标，变换后的求解目标会单独保存在 `orientation_target_quaternions_wxyz`，绝不会覆盖 `human_orientation_quaternions_wxyz`。

物体交互结果还会保存 `object_points_demo_local`、`object_points_target_local`、`object_points_demo_world` 和 `object_points_target_world`，分别表示示范/目标尺度下的局部采样点及求解时实际使用的逐帧世界坐标。必需的 Interaction Mesh 字段组保存逐帧源/目标顶点、打包四面体与逐帧数量，以及人体和物体顶点分界。

求解器会在真实 MuJoCo 几何上检查每个 QP proposal。若完整 QP 步不可行而 incumbent 可行，求解器会沿两者之间二分并返回已知最大的真实几何可行步；若 incumbent 本身不可行，则仍可前进到不可行 proposal，让下一次 SQP 重新线性化并尝试恢复。约束回退是外层模式调度，不是一次性的 QP 重试。对启用了 foot sticking 的帧，求解器会依次让 `normal`、确实产生放宽效果时的 `relaxed`、以及配置允许时的 `released` 脚部模式各自从同一个 frame-entry 状态运行完整 SQP，此时机器人—物体非穿透仍然生效。只有这些模式均未产生可行候选且明确允许释放物体非穿透时，才会在物体非穿透已释放的条件下按相同脚部模式顺序重新运行。只要更严格的模式曾产生任何真实几何可行候选，就由最严格的这一级胜出，并保存该模式最后一个可行候选。

另有一项默认策略只处理一种狭窄的初始化失败，而且不会释放任何物理可行性约束。对于没有 nominal 轨迹、浮动基座 Z 参与优化的 identity 或 ablation `robot_only` 任务，只有结构化的第 0 帧失败表明地面非穿透是唯一违反的硬物理约束、对象确为水平地面平面，而且触发重试的这次失败没有使用物理约束 release 或 trust-region release 时，才允许严格重试一次。重试从原始 frame-entry 状态重新计算真实几何，只把 root Z 抬高“精确穿透修正量加共享严格内侧裕量”，重新检查全部硬约束后运行同一 SQP。在这次重跑的 SQP 内，原有的第 0 帧算法性 trust-region release 仍可能被选中，并会单独记录；它不释放地面或其他物理约束，每个候选仍必须通过真实几何门禁。该策略绝不会用于物体交互、攀爬、augmentation、非水平地面或混合失败；重试被拒绝或仍失败时继续作为硬失败，原本成功的路径不受影响。

每个 schema-v2 artifact 都保存八个审计字段；未触发时数值诊断使用零哨兵：`frame_zero_ground_retry_policy`、`frame_zero_ground_retry_eligible`、`frame_zero_ground_retry_triggered`、`frame_zero_ground_retry_initial_min_distance_m`、`frame_zero_ground_retry_corrected_min_distance_m`、`frame_zero_ground_retry_lift_m`、`frame_zero_ground_retry_interior_margin_m` 和 `frame_zero_ground_retry_initial_sqp_iterations`。触发后，第 0 帧的 `sqp_stop_reasons` 还必须带 `frame_zero_ground_retry:` 前缀。严格 artifact 验证会交叉核对资格、单轴距离等式、配置的穿透容差、共享裕量、迭代数与停止原因来源。

六个 float64 数组记录最终接受状态相对配置容差的非负超量：`ground_non_penetration_violation`、`object_non_penetration_violation`、`foot_sticking_violation`、`foot_lock_violation`、`self_collision_violation` 和 `joint_limits_violation`。地面非穿透、foot lock、自碰撞与关节限位始终是硬约束；物体非穿透在最终物体模式未释放时是硬约束；foot sticking 在 `normal` 和 `relaxed` 模式下是硬约束。被释放的约束仍然会测量和保存，而硬约束超量必须处于 schema 的非线性验收容差内。

每帧最终接受的精确模式分别保存在 `constraint_mode_foot_sticking`、`constraint_mode_object_non_penetration_released` 和 `constraint_mode_trust_region_released`。其中脚部模式只能是 `inactive`、`normal`、`relaxed` 或 `released`；trust-region release 只是初始线性化回退，只允许出现在第 0 帧。兼容摘要 `foot_sticking_fallback_frames`、`foot_sticking_release_frames` 与 `object_non_penetration_release_frames` 必须由最终接受模式精确派生，失败尝试不能进入这些帧列表。若真正与脚部有关的失败仍触发配置的完整序列重试，保存轨迹会全程关闭 foot sticking，并通过 `foot_sticking_enabled_for_saved_trajectory` 与 `foot_sticking_full_sequence_retry_frame` 记录原因。`sqp_iteration_counts` 是该帧所有已尝试模式的线性化迭代总数。任何最终接受的物体非穿透释放都仍是显式降级结果，不能视作通过物体碰撞验收。

纯机器人 qpos 使用 `[root_xyz, root_wxyz, robot_dof]`；动态物体 qpos 会追加 `[object_xyz, object_wxyz]`，两种布局都保留 canonical float64 精度。

## 定量评估

标准结果树的发布门禁是 `rebuild_demo_results.py` 内置验证器。它会检查精确的预期 artifact 集、schema 与 hash 身份、帧数、完整人体和机器人拓扑、直接朝向来源、Interaction Mesh、精确的最终 float64 qpos、全部六项非线性残差数组、全部三项最终 ConstraintMode 数组、由模式派生的帧列表摘要、外部资产闭包以及配置的释放质量阈值。可在不启动求解器的情况下运行：

```bash
python examples/rebuild_demo_results.py \
  --run-id demo-rebuild-v1 \
  --validate-only
```

汇总测量和有界的验证问题会写入 `demo_results/runs/demo-rebuild-v1/report.json`；该报告与 staging 验证器共同构成 `demo_results/v1` 的验收门禁。

## 为 RL 全身跟踪策略准备数据

官方工作流包含两个步骤：

1. 运行重定向，获得 `.npz` 机器人动作。
2. 将结果按目标帧率转换为全身跟踪策略所需格式。

在 macOS 上请使用 `mjpython`，而不是 `python`。

### macOS（`mjpython`）

```bash
mjpython data_conversion/convert_data_format_mj.py \
  --input-file ./demo_results/v1/canonical/g1/robot_only/omomo/OMOMO_new/sub3_largebox_003/identity.npz \
  --output-fps 50 \
  --output-name converted_res/robot_only/sub3_largebox_003_mj_fps50.npz \
  --data-format omomo \
  --object-name ground \
  --once

mjpython data_conversion/convert_data_format_mj.py \
  --input-file ./demo_results/v1/canonical/g1/object_interaction/omomo/OMOMO_new/sub3_largebox_003/identity.npz \
  --output-fps 50 \
  --output-name converted_res/object_interaction/sub3_largebox_003_mj_w_obj.npz \
  --data-format omomo \
  --has-dynamic-object \
  --once
```

### 纯机器人设置

```bash
python data_conversion/convert_data_format_mj.py \
  --input-file ./demo_results/v1/canonical/g1/robot_only/omomo/OMOMO_new/sub3_largebox_003/identity.npz \
  --output-fps 50 \
  --output-name converted_res/robot_only/sub3_largebox_003_mj_fps50.npz \
  --data-format omomo \
  --object-name ground \
  --once

python data_conversion/convert_data_format_mj.py \
  --input-file ./demo_results/v1/canonical/g1/robot_only/lafan/lafan/dance2_subject1/identity.npz \
  --output-fps 50 \
  --output-name converted_res/robot_only/dance2_subject1_mj_fps50.npz \
  --data-format lafan \
  --object-name ground \
  --once
```

### 机器人—物体设置

```bash
python data_conversion/convert_data_format_mj.py \
  --input-file ./demo_results/v1/canonical/e1/object_interaction/omomo/OMOMO_new/sub10_tripod_000/identity.npz \
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
