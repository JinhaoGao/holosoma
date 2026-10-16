# Holosoma 动作重定向

中文 | [English](README.md)

<!-- 维护约定：保持本文档稳定，仅在仓库维护者明确确认后修改，并同步更新中英文。 -->

将人体动作重定向到人形机器人，支持纯机器人动作、物体交互和攀爬任务。

支持的数据入口为 OMOMO（`OMOMO_new`）、LAFAN（`lafan`）、AMASS SMPL-X（`amass`）、GVHMR（`gvhmr`）、FBX（`fbx_mocap`）、Noetix BVH（`noetix_mocap`）、攀爬动捕（`climbing`）和 Noetix 攀爬 CSV（`noetix_csv_climb`）。输入文件与转换步骤见下方“数据准备”。

通过 `--robot` 选择机器人：G1（`g1`，29 自由度）、E1（`e1_23dof` / `e1_24dof`）、E2（`e2`，23 自由度），以及 M4G / M5C / M5D / M5E / M5 反屈（`m4g` / `m5c` / `m5d` / `m5e` / `m5fanqu`，17 自由度）。`e1` 是 `e1_23dof` 的兼容名称。这些机器人支持 `robot_only`；物体交互和攀爬目前仅支持 G1。

先完成[重定向环境安装](../../../README.md#setup)并激活对应环境，再从以下目录执行命令。示例中的 `/path/to/...` 和动作名需要替换为自己的路径与文件。

```bash
cd src/holosoma_retargeting/holosoma_retargeting
```

转换工具会生成以米为单位的世界坐标人体关节位置 `(T, J, 3)`，并保存对应格式所需的信息。原始数据请使用匹配的转换工具；仅数组形状相同的任意 NPZ 不能直接作为输入。

## 单动作重定向

将仓库自带的 OMOMO 示例重定向到 G1：

```bash
python examples/robot_retarget.py \
  --task robot_only --robot g1 \
  --dataset OMOMO_new --motion sub3_largebox_003
```

修改 `--robot` 即可选择其他支持的机器人。`--motion` 填写不含扩展名的文件名，有子目录时填写相对数据根目录的路径，名称含空格时加引号。结果保存到 `demo_results/<robot>/<task>/<dataset>/<motion>.npz`。

生成单个 G1 物体交互结果时，将任务改为 `object_interaction`；该任务只接受 `OMOMO_new`。攀爬任务接受 `climbing` 或 `noetix_csv_climb`，示例见“数据准备”。

### 可选参数

在主命令后追加 `--data-path /path/to/dataset` 可指定数据根目录，`--save-dir /path/to/results` 可指定结果根目录，`--output-name custom.npz` 只修改最终文件名。已有结果会被复用，需要重新生成时追加 `--overwrite`。

实时 Viser 可视化默认开启。无界面运行时追加 `--retargeter.no-visualize`；需要诊断图层时追加 `--retargeter.debug`，求解结束后按 Enter 退出。

机器人默认参数读取自 [robot_profiles](examples/robot_profiles/)，显式命令行参数优先于对应 JSON。使用 `--robot-profile /path/to/robot.json` 可选择另一份配置。

动作调节方面，`--foot-sticking False` 关闭足底平面接触约束，`--orientation-tracking True` 启用源朝向跟踪，`--shoulder-direction-tracking True` 跟踪上臂方向。源朝向跟踪需要朝向数据，旧版仅含位置的 NPY 不支持此项。

使用 `--natural-pose-tracking True` 可启用配置中的自然姿态参考；`--root-position-weight 数值` 和 `--root-orientation-weight 数值` 分别设置跟踪源根位置、根朝向的非负权重。

完整参数见 `python examples/robot_retarget.py --help`。这些公共参数同样适用于下方的增强重定向命令。

### 双角色联合优化

先分别通过 `robot_only` 生成两个角色的结果。两份结果需要帧数、帧率、源格式和坐标约定一致，并保留人体参考数据；FBX 角色需要来自同一个 FBX 文件。随后执行联合优化：

```bash
python -m holosoma_retargeting.paired_retargeting.robot_refine \
  --actor-a.name actor_a --actor-a.result-path /path/to/actor_a.npz \
  --actor-b.name actor_b --actor-b.result-path /path/to/actor_b.npz \
  --output-path demo_results/paired_refined.npz
```

可选参数：`--visualize` 在优化后打开结果；`--contacts-path /path/to/contacts.json` 指定接触关系。接触与碰撞配置详见[双角色说明](paired_retargeting/README.md)。

## 增强重定向

生成一个 G1 动作及其增强变体。此命令处理指定动作，不遍历整个数据集。

```bash
python examples/parallel_robot_retarget.py \
  --task object_interaction --robot g1 \
  --dataset OMOMO_new --motion sub3_largebox_003
```

物体交互会生成原始轨迹及平移、旋转变体。攀爬可使用 `--task climbing --dataset climbing --motion mocap_climb_seq_0`，也可选择已准备好的 `noetix_csv_climb` 动作，增强变体会缩放地形高度。原始轨迹会自动首先生成。

结果保存到 `demo_results_parallel/<robot>/<task>/<dataset>/<motion>.npz`，增强文件在同目录下命名为 `<motion>_<variant>.npz`。可追加上方公共参数；增强入口仅支持 G1 物体交互和攀爬任务。

## 数据准备

已有对应格式的转换后数据时，可跳过转换步骤。数据转换命令可以处理目录，重定向命令则选择其中一个动作。

### OMOMO

使用官方指南中的 [InterMimic 处理版 OMOMO](https://drive.google.com/file/d/141YoPOd2DlJ4jhU2cpZO5VU5GzV_lm5j/view)，将 `.pt` 文件解压到 `demo_data/OMOMO_new/`。原始 OMOMO 发布数据的格式不同。这里不需要额外的人体动作转换，使用上方单动作或增强示例中的 `--dataset OMOMO_new` 即可。

### LAFAN

下载并解压 [LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset/blob/master/lafan1/lafan1.zip)。转换器读取输入目录直属的 BVH 文件。若缺少 `data_utils/lafan1/`，先安装一次 BVH 读取工具：

```bash
git clone https://github.com/ubisoft/ubisoft-laforge-animation-dataset.git data_utils/lafan_source
cp -r data_utils/lafan_source/lafan1 data_utils/lafan1
```

将 BVH 转换为带关节名称的 NPZ，再选择一个动作重定向：

```bash
python -m holosoma_retargeting.data_utils.extract_global_positions \
  --input-dir /path/to/lafan --output-dir demo_data/lafan

python examples/robot_retarget.py \
  --task robot_only --robot e2 --dataset lafan --motion walk2_subject3
```

可选参数：转换时追加 `--overwrite` 可替换旧输出。需要源朝向时，请从 BVH 重新转换旧版仅含位置的 NPY 数据。

### AMASS SMPL-X

从 [AMASS](https://amass.is.tue.mpg.de/) 下载 SMPL-X 动作，从 [SMPL-X](https://smpl-x.is.tue.mpg.de/) 下载 neutral 模型并放到 `models/smplx/SMPLX_NEUTRAL.npz`。原始动作目录形如 `/path/to/amass/<数据集>/<受试者>/*_stageii.npz`。

```bash
python -m holosoma_retargeting.data_utils.prep_amass_smplx_for_rt \
  --amass-root-folder /path/to/amass \
  --output-folder demo_data/amass_smplx_processed

python examples/robot_retarget.py \
  --task robot_only --robot g1 --dataset amass \
  --motion HumanEva_S3_Jog_1_stageii
```

转换器递归搜索，并用下划线连接源文件的相对路径：`HumanEva/S3/Jog_1_stageii.npz` 会生成 `HumanEva_S3_Jog_1_stageii.npz`。转换命令的可选参数为 `--model-root-folder /path/to/models`、`--subdataset-folder HumanEva` 或 `--fps 30`，目标帧率不能高于源帧率。

### GVHMR

准备包含世界坐标 `smpl_params_global` 的 GVHMR `hmr4d_results.pt`，以及上文使用的 SMPL-X neutral 模型。先转换单个预测结果，再执行重定向：

```bash
python -m holosoma_retargeting.data_utils.convert_gvhmr \
  --input-file /path/to/hmr4d_results.pt --output-file demo_data/gvhmr/tennis.npz

python examples/robot_retarget.py \
  --task robot_only --robot g1 --dataset gvhmr --motion tennis
```

可选参数：`--model-path /path/to/models` 指定 SMPL-X 模型。预测帧率不是 30 时，通过 `--fps` 指定实际帧率；该选项记录时间信息，不对预测结果重采样。

### FBX 动捕

从当前目录安装 FBX 额外依赖，再转换当前支持的 Hips 根节点 FBX 骨架：

```bash
python -m pip install -e "..[fbx]"
python -m holosoma_retargeting.data_utils.convert_fbx \
  /path/to/fbx demo_data/fbx_mocap
```

每个角色会单独输出为 `<文件名>__<角色名>.npz`。使用转换器打印的实际输出名称，例如 `Take_38_003_R__nv.npz`：

```bash
python examples/robot_retarget.py \
  --task robot_only --robot e1_24dof --dataset fbx_mocap \
  --motion Take_38_003_R__nv
```

可选参数：转换时通过 `--target-fps 30` 设置采样帧率。使用 M5 反屈时，将 `--robot e1_24dof` 替换为 `--robot m5fanqu`，其 FBX 配置会自动保留源膝部方向并按机器人骨段缩放。双角色联合优化需要两个角色保持相同的采样时间轴。支持的骨架与坐标约定见 [FBX 详细说明](../../../docs/fbx-mocap-retargeting.md)。

### Noetix BVH

对于当前支持的 Noetix BVH 导出格式，使用专用转换器。它需要上方 LAFAN 小节安装的同一套 BVH 读取工具，输出会保留子目录结构。

```bash
python -m holosoma_retargeting.data_utils.convert_noetix_bvh \
  --input-dir /path/to/noetix_bvh --output-dir demo_data/noetix_mocap

python examples/robot_retarget.py \
  --task robot_only --robot m5c --dataset noetix_mocap --motion walk
```

这里的 `walk` 假设转换生成了 `demo_data/noetix_mocap/walk.npz`；子目录中的文件使用 `folder/walk`。可选参数：转换时设置 `--target-fps 30`，重定向时通过 `--robot` 选择其他支持的机器人。

### 攀爬动捕

仓库示例位于 `demo_data/climb/<动作名>/`。人体关节位置 NPY、场景 XML、URDF 和网格文件需要完整保留在对应目录中。单动作示例：

```bash
python examples/robot_retarget.py \
  --task climbing --robot g1 --dataset climbing --motion mocap_climb_seq_0
```

可选用法：将脚本替换为 `examples/parallel_robot_retarget.py` 可生成地形高度增强变体。将该人体动作重定向到其他机器人时，使用 `--task robot_only` 并选择对应 `--robot`。旧版 NPY 只包含位置；机器人配置已开启朝向跟踪时（包括 M4G 和 M5 系列），需追加 `--orientation-tracking False`。

### Noetix 攀爬 CSV

这里的输入是包含骨骼数据和攀爬场景信息的 Noetix 人体动捕 CSV，与重定向后导出的机器人关节 CSV 不同。先转换为完整的动作与场景目录：

```bash
python -m holosoma_retargeting.data_utils.convert_noetix_csv \
  /path/to/actor_170/climb.csv \
  --output-root demo_data/noetix_csv_climb --task-name climb

python examples/robot_retarget.py \
  --task climbing --robot g1 --dataset noetix_csv_climb --motion climb
```

请将生成的 NPZ/NPY 与场景资产一起保留在 `demo_data/noetix_csv_climb/climb/`。转换时可追加 `--human-height 1.70` 指定以米为单位的人体身高，`--target-fps 30` 降采样，或 `--scene-template-dir /path/to/template` 选择场景模板。默认模板为仓库自带的 `demo_data/climb/mocap_climb_seq_0`。需要增强时使用 parallel 重定向脚本；只重定向人体动作时使用 `--task robot_only`。

## 结果可视化

### 单动作

```bash
python viser_player.py \
  --input-path demo_results/g1/robot_only/OMOMO_new/sub3_largebox_003.npz
```

生成的纯机器人 NPZ 动作使用同一命令，无人体参考时会隐藏人体图层。若缺少机器人信息，追加 `--robot-type` 或 `--robot-urdf` 指定模型。播放机器人 CSV 时同样需要指定模型：

```bash
python viser_player.py --input-path /path/to/motion.csv --robot-type e2
```

CSV 每行依次为以米为单位的根位置 `xyz`、根四元数 `xyzw` 和按 URDF 文件顺序排列的关节值，不含表头与索引；旋转关节角使用弧度。特定型号使用 `--robot-urdf /path/to/robot.urdf`，模型必须与 CSV 导出时一致。CSV 默认以 30 FPS 播放，其他帧率通过 `--fps` 指定。

### 多动作

将使用同一机器人模型的多个 NPZ 结果放在共享时间轴上对比：

```bash
python multi_viser_player.py \
  --input-paths /path/to/motion_a.npz /path/to/motion_b.npz
```

查看完整增强族时，将 `--input-paths` 替换为 `--family`：

```bash
python multi_viser_player.py \
  --family demo_results_parallel/g1/object_interaction/OMOMO_new/sub3_largebox_003.npz
```

### 双角色

```bash
python paired_retargeting/viser_player.py --input-path demo_results/paired_refined.npz
```

三个播放器均可追加 `--port 8081` 指定端口、`--fps 30` 覆盖播放帧率、`--no-playing` 暂停启动、`--loop` 循环播放，或 `--dry-run` 仅检查输入而不启动服务。未指定帧率时读取结果中的 FPS；多动作结果帧率不一致时，需要显式指定 `--fps`。浏览器 Playback 控件用于跳转帧，Layers 面板用于选择图层。单动作入口没有 `--start-frame` 参数。

### 导出机器人动作 CSV

使用与结果匹配的 URDF 导出单个 NPZ。CSV 仅包含机器人轨迹，不保存人体参考、物体位姿和 FPS：

```bash
python -m holosoma_retargeting.data_utils.npz_to_csv \
  demo_results/g1/robot_only/OMOMO_new/sub3_largebox_003.npz models/g1/g1_29dof.urdf
```

默认在 NPZ 旁生成同名 CSV，可追加 `--output-path /path/to/motion.csv` 修改位置。批量转换使用同一机器人模型的结果目录：

```bash
python -m holosoma_retargeting.data_utils.batch_npz_to_csv \
  demo_results/g1/robot_only/OMOMO_new models/g1/g1_29dof.urdf \
  demo_results_csv/g1/robot_only/OMOMO_new
```

## 定量评估

对保留人体参考和求解信息的重定向 NPZ 结果执行评估：

```bash
python evaluation/eval_retargeting.py \
  --res-dir demo_results/g1/robot_only/OMOMO_new \
  --data-type robot_only --robot g1 --data-format omomo
```

物体交互使用对应结果目录及 `--data-type robot_object --data-format omomo`；攀爬使用 `--data-type robot_terrain --data-format mocap`。可追加 `--max-workers 4` 并行评估多个结果。这里的 `--data-format` 是内部格式名，因此 OMOMO 使用 `omomo`，重定向命令的 preset 则为 `OMOMO_new`。

## 准备全身跟踪策略的训练数据

将重定向 NPZ 转换为训练所需的刚体位姿与速度。下面的 G1 示例会打开 MuJoCo，需要图形环境；macOS 使用 `mjpython` 替代 `python`。

```bash
python data_conversion/convert_data_format_mj.py \
  --input-file demo_results/g1/robot_only/OMOMO_new/sub3_largebox_003.npz \
  --output-name converted_res/g1/sub3_largebox_003.npz \
  --robot g1 --data-format omomo --once
```

可选参数：`--output-fps 50` 设置训练采样帧率。转换含动态物体的结果时，选择相应输入文件，并追加 `--has-dynamic-object --object-name largebox`。`--use-omniretarget-data` 仅用于外部 OmniRetarget 的 qpos 排列。该转换工具内置的关节表为 `g1`、`e1` 和 `e2`；其他型号需要通过 `--joint-names` 提供关节表，并核对输入关节顺序与机器人资产是否一致。

## 自定义人体动作格式

数据加载与关节映射的接入要求见[新增人体动作格式](ADD_MOTION_FORMAT_README_zh.md)。

## 自定义机器人

模型准备与关节映射见[新增机器人](ADD_ROBOT_TYPE_README.md)，长期生效的重定向参数见[机器人配置](examples/robot_profiles/README.md)。
