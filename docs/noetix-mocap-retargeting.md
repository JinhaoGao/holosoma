# Noetix Mocap Retargeting 功能说明

本文梳理 `feature/noetix-mocap` 分支中与 Noetix 动捕数据重定向相关的改动。按当前代码理解，这次支持了两类 Noetix 数据链路：

1. **Noetix BVH / LAFAN-like 数据**：把不同 Noetix BVH 骨架统一转换成 LAFAN 22 关节格式，注册为 `data_format=noetix_lafan`，用于 `robot_only` 重定向。
2. **Noetix CSV box-climb 数据**：从 Noetix CSV 中恢复人体骨架、箱体 marker、箱体 mesh/URDF/XML 场景，导出为现有 `data_format=mocap` 的 `climbing` 任务输入。

当前工作区还有一个未提交新增文件：`src/holosoma_retargeting/holosoma_retargeting/data_utils/convert_noetix_csv.py`。如果这份能力要合入分支，需要把它一并提交。

## 总体实现路径

### BVH / noetix_lafan 路径

数据流：

```text
Noetix .bvh
  -> data_utils/convert_noetix_bvh.py
  -> demo_data/noetix_lafan/<sequence>.npz
  -> examples/robot_retarget.py --task-type robot_only --data_format noetix_lafan
  -> demo_results/<robot>/robot_only/noetix_lafan/<sequence>.npz
  -> viser_player.py 回放和检查
```

核心处理：

- 识别 BVH 骨架类型，兼容 `fullbody57_*`、`lafan22_zyx`、`run23_yxz` 等命名。
- 将源骨架关节映射到 LAFAN 22 关节顺序。
- 单位从 cm 转为 m，坐标从 y-up 转为 z-up。
- 根据根节点、脚尖方向修正朝向提示，避免 LAFAN 初始化朝向错误。
- 从文件名或关节高度估计人体身高，并写入 `.npz`，重定向时用于机器人身高缩放。

### CSV / mocap climbing 路径

数据流：

```text
Noetix .csv
  -> data_utils/convert_noetix_csv.py
  -> demo_data/<climb_root>/<sequence>/
       <sequence>_joint_positions.npy
       multi_boxes.obj
       multi_boxes.urdf
       box_assets.xml
       box_body.xml
       *_w_multi_boxes.xml
       scene_reconstruction.npz
  -> examples/robot_retarget.py --task-type climbing --data_format mocap
  -> demo_results/<robot>/climbing/<name>/<sequence>_original.npz
  -> viser_player.py 回放和检查
```

核心处理：

- 解析 Noetix CSV 的多行表头，把 Bone、SkeletonMarker、RigidBodyMarker、Marker、Rigid Body 分列索引化。
- 对 Bone 的局部 offset 和旋转做 forward kinematics，恢复全局骨架点。
- 从 Noetix y-up/mm 坐标转换到 MuJoCo z-up/m 坐标，使用 `[x, -z, y]`。
- 用箱体 marker 定义场景原点；如果 RigidBodyMarker 缺失，从普通 Marker 中找接近 1m 方形的箱体顶面 marker 作为 fallback。
- 将骨架导出为现有 `MOCAP_DEMO_JOINTS` 顺序的 `.npy`，并重建单箱平台的 OBJ、URDF、MuJoCo include 文件。
- 保存 `scene_reconstruction.npz`，用于调试原始骨架、marker、箱体网格和 fps/downsample 信息。

## 主要代码改动

### 1. 新增 Noetix BVH 转换器

文件：`src/holosoma_retargeting/holosoma_retargeting/data_utils/convert_noetix_bvh.py`

关键段落：

- `NOETIX_*_MAPPING`，第 24-85 行：定义不同源 BVH 骨架到 LAFAN 22 关节的映射。这里保留了 LAFAN 现有左右侧张量约定，所以看起来左右有反转，但与已有 LAFAN retarget 路径一致。
- `classify_bvh()`，第 88-102 行：根据关节名集合自动判断 BVH 类型，并选择对应 mapping。
- `select_canonical_joints()`，第 105-121 行：检查源关节是否完整，并按 `NOETIX_LAFAN_DEMO_JOINTS` 顺序抽取统一关节。
- `transform_y_up_to_z_up()`，第 124-125 行：做坐标轴转换。
- `infer_lafan_forward()` 和 `apply_lafan_root_orientation_hint()`，第 136-176 行：用脚到脚尖方向估计面朝方向，并调整 `Hips -> Spine` 的水平分量，让后续 `estimate_human_orientation()` 能稳定初始化机器人朝向。
- `estimate_height()`、`drop_initial_jump()`、`recenter_xy()`、`downsample()`，第 194-223 行：处理身高、首帧异常跳变、XY 归零和 fps。
- `convert_file()`，第 226-264 行：完整转换单个 BVH，并输出 `global_joint_positions`、`height`、`joint_names`、`raw_joint_names`、`source_type`、`fps` 等字段。

### 2. 新增 Noetix CSV 转 climbing 场景工具

文件：`src/holosoma_retargeting/holosoma_retargeting/data_utils/convert_noetix_csv.py`

关键段落：

- `NOETIX_BONE_PARENTS`，第 26-61 行：定义 Noetix skeleton 的父子关系，包含躯干、四肢、脚趾和手指。
- `MOCAP_SOURCE_BONES`，第 63-117 行：定义输出到 `MOCAP_DEMO_JOINTS` 时每个目标关节对应的 Noetix 源 Bone。
- `_build_column_index()`，第 177-201 行：解析 Noetix CSV 多层表头，建立 `(entity_type, name, id, prop, axis) -> column` 的索引。
- `_read_bone_channels()` 和 `_forward_kinematics()`，第 245-320 行：读取 Bone 的 offset/rotation，并通过 FK 恢复全局位置。
- `_y_up_m_to_scene_axes()`，第 323-327 行：将 Noetix y-up/m 转为 MuJoCo 场景坐标 `[x, -z, y]`。
- `_select_box_markers_from_unlabeled_markers()`，第 392-447 行：当箱体 marker 未出现在 RigidBodyMarker 中时，从普通 Marker 中按“近似 1m 方形、近似共面”的规则恢复箱顶四点。
- `load_noetix_csv()`，第 458-519 行：整合 CSV 解析、FK、marker 读取、箱体原点归一化，输出 `NoetixCsvMotion`。
- `map_to_mocap_joints()`，第 529-541 行：将 Noetix 骨架映射为现有 `mocap` 格式。
- `_write_single_platform_mesh()`、`_write_single_box_urdf()`、`_write_mujoco_box_includes()`，第 618-709 行：从箱体 marker 重建平台 OBJ，并生成 retargeting 所需的 URDF 和 MuJoCo include。
- `export_mocap_climb()`，第 729-786 行：导出 climbing 任务目录，包括 `.npy`、箱体资产和 `scene_reconstruction.npz`。
- `parse_args()` / `main()`，第 789-862 行：提供命令行入口。

### 3. 注册新数据格式和 E1 映射

文件：`src/holosoma_retargeting/holosoma_retargeting/config_types/data_type.py`

关键段落：

- 第 40 行：新增 `NOETIX_LAFAN_DEMO_JOINTS = LAFAN_DEMO_JOINTS.copy()`。Noetix BVH 转换后直接复用 LAFAN 22 关节拓扑。
- 第 197-213 行：新增 `("noetix_lafan", "g1")` 关节映射。
- 第 333-349 行：新增 `("noetix_lafan", "e1")` 关节映射。
- 第 384-400 行：新增 `("mocap", "e1")` 关节映射，支撑 CSV/climbing 路径重定向到 E1。
- 第 404-410 行：在 `TOE_NAMES_BY_FORMAT` 中加入 `noetix_lafan`。
- 第 431-437 行：在 `DEMO_JOINTS_REGISTRY` 中注册 `noetix_lafan`。
- 第 462-463 行和第 518-523 行：新增 `human_height` override，允许通过 `MotionDataConfig` 覆盖默认人体身高。这对不同 Noetix 受试者的尺度调整有用。

### 4. E1 机器人配置和资产

文件：

- `src/holosoma_retargeting/holosoma_retargeting/config_types/robot.py`
- `src/holosoma_retargeting/holosoma_retargeting/config_types/data_conversion.py`
- `src/holosoma_retargeting/holosoma_retargeting/models/e1/*`

关键段落：

- `robot.py` 第 18-22 行：在 `_ROBOT_DEFAULTS` 中注册 `e1`，设置 `robot_dof=23`、`robot_height=1.4`。
- `robot.py` 第 158-170 行：新增 E1 足底 sphere links，用于 foot sticking / foot lock。
- `robot.py` 第 245-250 行：新增 E1 的 nominal tracking indices。
- `data_conversion.py` 第 42-66 行：新增 E1 的 23 个关节名，用于 retarget 结果转换到仿真数据格式。
- `models/e1/`：新增 `e1_23dof.urdf`、`e1_23dof.xml`、`e1_23dof_w_largebox.xml` 和 meshes，为 E1 retarget、可视化和带 largebox 场景提供资产。

### 5. retarget 入口逻辑

文件：`src/holosoma_retargeting/holosoma_retargeting/examples/robot_retarget.py`

关键段落：

- 第 60-64 行：任务默认格式保持 `climbing -> mocap`，`robot_only -> smplh`。
- 第 82 行：新增 `BVH_LIKE_FORMATS = {"lafan", "noetix_lafan"}`，让 Noetix LAFAN 走和 LAFAN 相同的基座初始化逻辑。
- 第 141-164 行：配置校验中 `robot_only` 接受 registry 中任意格式；`climbing` 仍限定 `mocap`。
- 第 213-253 行：`load_motion_data()` 新增 `noetix_lafan` 读取逻辑。它从 `.npz` 读取 `global_joint_positions` 和 `height`，并用 `constants.ROBOT_HEIGHT / human_height` 得到缩放系数。
- 第 331-412 行：`setup_object_data()` 支持 `robot_only` 地面点、`object_interaction` 物体缩放、`climbing` 的 multi_boxes 资产缩放。
- 第 417-478 行：`_compute_q_init_base()` 中，`robot_only` 的 LAFAN 和 Noetix LAFAN 用人体朝向估计初始化浮动基；`climbing + mocap` 改用脚到脚尖方向估计初始 yaw，避免静态箱体 dummy pose 无法提供朝向。
- 第 672-678 行：使用 `dataclasses.replace()` 更新 nested config，只改 `data_format` 和 `robot_type`，保留调用者传进来的其他配置覆盖。

并行版本：

- `examples/parallel_robot_retarget.py` 第 60-94 行：批量查找时，未知格式默认按 `.npz` 查找，因此 `noetix_lafan` 可以批量处理。
- 第 323-329 行：同样使用 `replace()` 更新 nested config，避免丢失用户配置。

### 6. 朝向估计和 interaction mesh 工具

文件：`src/holosoma_retargeting/holosoma_retargeting/src/utils.py`

关键段落：

- 第 408-435 行：新增 `interaction_mesh_edges_from_tetrahedra()`，从 Delaunay 四面体中提取唯一边；`edge_mode="cross"` 只保留人体/机器人点到物体点的边，调试时更清晰。
- 第 643-672 行：新增 `create_scaled_object_scene_xml()`，支持 object interaction 时同步缩放 MuJoCo XML 里的 object mesh。
- 第 884-935 行：新增 `estimate_mocap_foot_orientation()`，从 `LeftFoot -> LeftToeBase` 和 `RightFoot -> RightToeBase` 的平均方向估计 mocap actor 初始 yaw。

### 7. retargeter 求解、保存和调试增强

文件：`src/holosoma_retargeting/holosoma_retargeting/config_types/retargeter.py`

关键段落：

- 第 9-25 行：新增 `FootLockConfig`，支持按帧窗口锁定左右脚 Z 高度。
- 第 27-45 行：新增 `SelfCollisionConfig`，支持指定 body pair、窗口和最小距离。
- 第 80-112 行：新增 mesh opacity、interaction mesh 显示/保存、自碰撞、nominal tracking 参数。

文件：`src/holosoma_retargeting/holosoma_retargeting/src/interaction_mesh_retargeter.py`

关键段落：

- 第 52-76 行：`InteractionMeshRetargeter.__init__()` 新增 foot lock、自碰撞、可视化、interaction mesh、nominal tracking 参数。
- 第 136-145 行：优先使用 `SCENE_XML_FILE`，使 climbing 中按任务目录生成的 scene XML 生效。
- 第 170-189 行：按 MuJoCo `qpos` 地址构建关节上下限，避免 free joint 或 object joint 导致 actuated joint limit 错位。
- 第 208-237 行：初始化 foot lock，并用命名规则识别 left/right。
- 第 239-278 行：初始化 self-collision，把用户传入的 body pair 展开成 MuJoCo collision geom pair。
- 第 432-490 行：新增 interaction mesh 的实时绘制。
- 第 492-719 行：`retarget_motion()` 中保存 `human_joint_names`、`mapped_human_joints`、`mapped_human_joint_names`、`mapped_robot_joints`、`mapped_robot_link_names`；如果启用 `save_interaction_mesh`，还保存 source/target vertices、tetrahedra、顶点数量和默认边模式。
- 第 914-961 行：在单帧 QP 中加入 foot sticking 和显式 foot lock Z 约束。
- 第 971-978 行：加入 self-collision 距离约束。
- 第 995-1000 行：加入 nominal tracking cost；增强 augmentation 或 climbing 时的姿态稳定性。
- 第 1173-1179 行：显示到 Viser 时按 qpos 到 Viser joint order 做重排。

### 8. Viser 回放和检查工具

文件：

- `src/holosoma_retargeting/holosoma_retargeting/config_types/viser.py`
- `src/holosoma_retargeting/holosoma_retargeting/viser_player.py`
- `src/holosoma_retargeting/holosoma_retargeting/src/viser_utils.py`

关键段落：

- `viser.py` 第 23-31 行：新增 `robot_mujoco_xml`、`robot_type`、`data_format`，用于推断 qpos 关节顺序和 mapped skeleton。
- `viser.py` 第 48-76 行：新增机器人/物体 mesh 可见性、opacity、mapped skeleton、interaction mesh 参数。
- `viser_player.py` 第 37-50 行：读取 retarget 输出中的 `human_joints`、`mapped_*` metadata 和 interaction mesh 数据。
- `viser_player.py` 第 225-362 行：新增 `MappedSkeletonOverlay`，叠加显示源人体 mapped joints 和目标机器人 mapped links；如果结果里保存了 `mapped_robot_joints`，直接使用保存值，避免重算时受 qpos/模型顺序影响。
- `viser_player.py` 第 365-453 行：新增 `InteractionMeshOverlay`，可回放保存的 source/target interaction mesh。
- `viser_player.py` 第 606-777 行：在播放器里接入 overlay 和 GUI/快捷键控制。
- `viser_utils.py` 第 28-60 行：提供 MuJoCo XML actuated joint name 读取和 joint order mapping。
- `viser_utils.py` 第 91-106 行：播放器支持 `qpos_to_viser_joint_indices` 和 `on_frame` 回调。

## 常用命令

以下命令建议在 `src/holosoma_retargeting/holosoma_retargeting` 目录下运行。

### 1. 转换 Noetix BVH 为 noetix_lafan

```bash
python data_utils/convert_noetix_bvh.py \
  --input-dir demo_data/noetix \
  --output-dir demo_data/noetix_lafan \
  --target-fps 30
```

输出：`demo_data/noetix_lafan/<sequence>.npz`。

### 2. 单条 Noetix BVH 重定向到 E1

```bash
python examples/robot_retarget.py \
  --robot e1 \
  --data_path demo_data/noetix_lafan \
  --task-type robot_only \
  --task-name <sequence> \
  --data_format noetix_lafan \
  --task-config.ground-range -10 10 \
  --save_dir demo_results/e1/robot_only/noetix_lafan \
  --retargeter.foot-sticking-tolerance 0.02 \
  --retargeter.save-interaction-mesh
```

如需实时调试，追加：

```bash
  --retargeter.visualize \
  --retargeter.debug \
  --retargeter.show-interaction-mesh \
  --retargeter.interaction-mesh-mode both \
  --retargeter.interaction-mesh-edges cross
```

### 3. 批量 Noetix BVH 重定向

```bash
python examples/parallel_robot_retarget.py \
  --robot e1 \
  --data-dir demo_data/noetix_lafan \
  --task-type robot_only \
  --data_format noetix_lafan \
  --task-config.object-name ground \
  --task-config.ground-range -10 10 \
  --save_dir demo_results_parallel/e1/robot_only/noetix_lafan \
  --retargeter.foot-sticking-tolerance 0.02 \
  --max-workers 8
```

### 4. 转换 Noetix CSV 为 climbing 场景

```bash
python data_utils/convert_noetix_csv.py /path/to/noetix_capture.csv \
  --output-root demo_data/noetix_climb \
  --task-name <sequence> \
  --target-fps 30
```

输出目录：`demo_data/noetix_climb/<sequence>/`。

注意：该目录中必须有目标机器人的 `*_w_multi_boxes.xml`。转换脚本默认从 `demo_data/climb/mocap_climb_seq_0` 复制模板；如果使用其他模板目录，传入：

```bash
  --scene-template-dir /path/to/template_dir
```

### 5. 单条 CSV/climbing 数据重定向到 E1

```bash
python examples/robot_retarget.py \
  --robot e1 \
  --data_path demo_data/noetix_climb \
  --task-type climbing \
  --task-name <sequence> \
  --data_format mocap \
  --task-config.object-name multi_boxes \
  --save_dir demo_results/e1/climbing/noetix_climb \
  --retargeter.save-interaction-mesh
```

### 6. 回放检查

Noetix BVH / robot-only：

```bash
python viser_player.py \
  --robot_urdf models/e1/e1_23dof.urdf \
  --qpos_npz demo_results/e1/robot_only/noetix_lafan/<sequence>.npz \
  --show_mapped_skeletons \
  --show_interaction_mesh \
  --interaction_mesh_mode both \
  --interaction_mesh_edges cross \
  --robot_mesh_opacity 0.8
```

CSV / climbing：

```bash
python viser_player.py \
  --robot_urdf models/e1/e1_23dof.urdf \
  --object_urdf demo_data/noetix_climb/<sequence>/multi_boxes_scaled_<scale>.urdf \
  --qpos_npz demo_results/e1/climbing/noetix_climb/<sequence>_original.npz \
  --show_mapped_skeletons \
  --show_interaction_mesh \
  --interaction_mesh_mode both \
  --interaction_mesh_edges cross \
  --robot_mesh_opacity 0.8 \
  --object_mesh_opacity 1
```

## 输出文件字段说明

### noetix_lafan 转换输出

`convert_noetix_bvh.py` 生成的 `.npz` 包含：

- `global_joint_positions`：shape 为 `[T, 22, 3]`，z-up、meter、XY 已归零。
- `height`：受试者身高，优先从文件名解析，否则估计。
- `joint_names`：`NOETIX_LAFAN_DEMO_JOINTS`。
- `raw_joint_names`：源 BVH 骨架关节名。
- `source_bvh`、`source_type`、`source_fps`、`fps`、`downsample_stride`、`dropped_initial_frame`：转换元数据。

### retarget 输出

`InteractionMeshRetargeter` 保存的 `.npz` 包含：

- `qpos`：最终机器人 qpos，MuJoCo 顺序。
- `human_joints`：预处理和缩放后的源人体关节。
- `human_joint_names`：源格式完整关节名。
- `mapped_human_joints` / `mapped_human_joint_names`：参与 Laplacian matching 的人体关节。
- `mapped_robot_joints` / `mapped_robot_link_names`：每帧对应的机器人 link 世界坐标。
- `interaction_source_vertices_w`、`interaction_target_vertices_w`、`interaction_tetrahedra` 等：仅在 `--retargeter.save-interaction-mesh` 时保存。

## 使用注意事项

- `noetix_lafan` 只用于 `robot_only`。`climbing` 任务仍使用现有 `mocap` 格式，这是 `robot_retarget.py` 的配置校验要求。
- Noetix BVH 输出 `.npz` 中的 `height` 会影响尺度。如果源文件名中有类似 `_170__` 的身高信息，会优先解析为 1.70m。
- CSV/climbing 数据的场景 XML 是按任务目录生成和读取的；如果缺少 `<robot>_w_multi_boxes.xml`，retarget 初始化 MuJoCo model 会失败。
- `mapped_robot_joints` 是当前工作区新增保存字段之一，便于回放时直接显示机器人 mapped skeleton；对应改动尚未全部提交。
- `demo_data/`、生成的 scaled XML/URDF 和 `models/*/generated/` 被 `.gitignore` 忽略，正式交付时需要明确哪些是数据资产，哪些只是本地生成物。
