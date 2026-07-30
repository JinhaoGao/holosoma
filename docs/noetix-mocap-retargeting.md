# Noetix Mocap 重定向运行手册

本文说明当前 Noetix 数据从原始文件、标准转换、统一重定向、消融实验到可视化的完整接口。命令默认从 `src/holosoma_retargeting/holosoma_retargeting` 执行。

公开重定向接口只有三类：`examples/robot_retarget.py` 处理单一动作，`examples/parallel_robot_retarget.py` 处理数据集批量任务或增强动作族，名称含 `ablation`、`comparison` 或 `search` 的实验脚本处理受控消融。三类入口共享 `RetargetJob`、求解生命周期、schema-v2 writer 和严格续跑契约；`rebuild_demo_results.py` 只是正式矩阵编排器，不是第四种求解接口。持久化 `run_kind` 与启动方式分开：所有 identity 结果均为 `single`，只有非 identity 动作变换为 `augmentation`，未做动作变换的实验 variant 为 `ablation`，批量执行本身不会保存为独立的 `run_kind`。

## 两类 Noetix 数据链路

Noetix BVH 用于 `robot_only`。原始 BVH 位于 `demo_data/noetix_ori`，转换后写入 `demo_data/noetix_mocap`，标准 `data_format` 为 `noetix_mocap`。标准人体骨架包含 22 个关节；转换器递归保留源目录层级，并识别多种 Noetix BVH 骨架变体。

Noetix CSV 箱体攀爬用于 `climbing`。原始 CSV 位于 `demo_data/noetix_ori/260625_box_csv`，每个 CSV 转换为 `demo_data/noetix_csv_climb/<actor>/<task>/` 下的完整场景目录，重定向时使用标准 `data_format=mocap`。场景目录同时包含 53 关节人体动作、平台 mesh、URDF、MuJoCo include 与重建诊断。

这两条链路都与公共 LAFAN 数据相互独立。Noetix BVH 只是在标准化后复用了兼容的 22 关节命名拓扑；Noetix CSV 则映射到现有 53 关节攀爬拓扑。

## 人体朝向保存原则

人体朝向只来自源文件直接提供的旋转。转换器不会根据关节位置、骨向量、脚尖方向或其他派生几何量估计并保存 link 朝向。机器人初始化可以在求解内部使用位置几何，但这种内部估计不会进入源人体朝向字段。

全部已注册格式的直接朝向来源如下。OMOMO 仅在 InterMimic `.pt` 明确包含第 383:591 列的 52 关节全局 `xyzw` 四元数时保存，并转换为 `wxyz`，来源记为 `intermimic_global_orientation_tensor`；较短 tensor 记为 `absent`。LAFAN 来自原始 BVH rotation channels，经层级 FK 后记为 `bvh_rotation_channels_fk`。AMASS 与 GVHMR 都来自源 SMPL-X 局部轴角旋转，经 FK 后记为 `direct_local_rotation_fk`。Noetix BVH 来自 BVH rotation channels，经 FK 后同样记为 `bvh_rotation_channels_fk`。转换后的 Noetix CSV 来自各 Bone 的 `Rotation` 通道，经 FK 后记为 `bone_rotation_channels_fk`。通用 climb 只有位置，记为 `absent`。这些来源之外的旋转不会作为源人体朝向写入 artifact。

Noetix BVH 的旋转来自 BVH rotation channels。转换器先按 BVH 层级做 FK，再通过基变换矩阵共轭把全局旋转从 Y-up 转到 Z-up，保存为 `wxyz` 四元数，并记录 `orientation_source="bvh_rotation_channels_fk"`。标准位置骨架始终有 22 个关节；当某种源骨架缺少一个或两个标准 spine 关节而必须合成其位置时，这些合成关节不保存朝向，因此 `orientation_joint_names` 可能只包含 20、21 或 22 个关节。

Noetix CSV 的旋转来自每个 Bone 的 `Rotation` 通道。转换器按 CSV 声明的父树做 FK，将全部 53 个标准 mocap 关节的直接旋转保存为世界坐标 `wxyz` 四元数，并记录 `orientation_source="bone_rotation_channels_fk"`。

`demo_data/climb` 中的通用攀爬文件只有人体位置，不含直接旋转。其重定向 artifact 必须记录 `orientation_source="absent"`，并完全省略 `human_orientation_joint_names`、`human_orientation_quaternions_wxyz` 与 `human_orientation_sha256`。这与完整机器人 link 朝向无关：所有结果仍保存逐帧机器人完整 link FK 位姿。

## 转换 Noetix BVH

下面的命令递归转换 `demo_data/noetix_ori` 中的全部 BVH，并在 `demo_data/noetix_mocap` 下保留相对目录：

```bash
python data_utils/convert_noetix_bvh.py \
  --input-dir demo_data/noetix_ori \
  --output-dir demo_data/noetix_mocap \
  --target-fps 30 \
  --overwrite
```

每个输出 `.npz` 包含 `global_joint_positions`、`joint_names`、`joint_parents`、`orientation_joint_names`、`orientation_quaternions_wxyz`、`orientation_source`、人体身高、源/输出 FPS、源 BVH 路径、识别出的骨架类型以及坐标和四元数约定。位置和旋转都转换为右手 Z-up 世界坐标；四元数顺序始终为 `wxyz`。

转换器可以丢弃一次超过 `--drop-jump-threshold-m` 的异常初始跳帧，并对位置做 XY 重心归零与既有 root-to-spine 位置提示。该提示只修改标准人体位置，不会生成、替换或补齐任何朝向。直接 BVH 旋转始终独立转换和保存。

## 转换 Noetix CSV 攀爬场景

转换器一次接收一个 CSV。例如：

```bash
python data_utils/convert_noetix_csv.py \
  demo_data/noetix_ori/260625_box_csv/actor_170/yuezhanggaotai-dongzuo1-kuai-box-170.csv \
  --output-root demo_data/noetix_csv_climb/actor_170 \
  --target-fps 30 \
  --scene-template-dir demo_data/climb/mocap_climb_seq_0
```

输出任务目录包含标准 `.npz` 与兼容 `.npy` 人体位置、`multi_boxes.obj`、`multi_boxes.urdf`、`box_assets.xml`、机器人场景 XML、`box_models/` 和 `scene_reconstruction.npz`。标准 `.npz` 是重定向适配器优先读取的文件，包含全部 53 个关节的位置、父树和直接 Bone Rotation 朝向。`scene_reconstruction.npz` 保留骨骼、marker、平台 mesh 与场景归一化信息，用于检查转换质量，不会被动作发现逻辑误当作输入序列。

## 单动作重定向

`examples/robot_retarget.py` 是唯一单动作入口。Noetix BVH 到 E1 的示例为：

```bash
python examples/robot_retarget.py \
  --data-path demo_data/noetix_mocap \
  --task-type robot_only \
  --task-name run_to_the_right \
  --data-format noetix_mocap \
  --robot e1
```

Noetix CSV 攀爬到 G1 的示例为：

```bash
python examples/robot_retarget.py \
  --data-path demo_data/noetix_csv_climb/actor_170 \
  --task-type climbing \
  --task-name yuezhanggaotai-dongzuo1-kuai-box-170 \
  --data-format mocap \
  --robot g1
```

省略 `--save-dir` 时，统一结果根为 `demo_results/v1`。标准 artifact 路径为：

```text
demo_results/v1/canonical/<robot>/<task_type>/<data_format>/
  <dataset_partition>/<sequence_key>/<variant>.npz
```

`--save-dir` 表示新的结果根，不应指向某个机器人、任务或格式叶目录。单动作、批处理/增强和消融都通过同一个 `RetargetJob`、同一个求解生命周期和同一个 schema writer 保存结果。这里目录名 `v1` 是结果树的 release namespace，目录内 artifact 使用 schema version 2；二者不是同一个版本号。

## 批量与增强重定向

Noetix BVH 批量重定向示例：

```bash
python examples/parallel_robot_retarget.py \
  --data-dir demo_data/noetix_mocap \
  --task-type robot_only \
  --data-format noetix_mocap \
  --robot e1 \
  --max-workers 8 \
  --run-id e1-noetix-bvh
```

Noetix CSV 攀爬及其增强示例：

```bash
python examples/parallel_robot_retarget.py \
  --data-dir demo_data/noetix_csv_climb \
  --task-type climbing \
  --data-format mocap \
  --robot e1 \
  --augmentation \
  --max-workers 8 \
  --run-id e1-noetix-csv-climb
```

攀爬增强族由 `identity`、`z_scale_0p8`、`z_scale_0p9`、`z_scale_1p1` 与 `z_scale_1p2` 组成。identity 会先生成或通过源 SHA-256、标准化配置 hash、schema、run kind 与 variant 的严格匹配检查，随后增强 variant 才能复用其名义轨迹。批处理同样只续用严格匹配的 artifact；同路径身份不匹配或 schema 无效时会显式拒绝，只有 `--overwrite-existing` 会授权替换。`--dry-run` 可只检查发现结果和写报告。未显式设置 `--report-path` 时，报告固定为 `<save_dir>/runs/<run_id>/report.json`。

## 朝向跟踪与消融

保存直接人体朝向不等于默认启用朝向优化。默认 `orientation_weights` 为空，求解仍按原有位置 Interaction Mesh 目标运行，但结果保留后续诊断所需的直接源朝向和完整机器人 link 朝向。统一消融入口会把格式无关的语义身体角色转换为所选格式的源关节名，并沿用共享格式/机器人 mapping 选择 link。Noetix G1/E1 使用完整共享 T-pose 标定；AMASS、GVHMR 与 OMOMO 尚无该标定，非零 profile 及其保留同组诊断的零权重 baseline 都必须显式传入 `--orientation-alignment-mode first_frame`。该模式只消费适配器证明为直接来源的人体四元数和机器人 FK，绝不从位置或骨向量估计朝向，其语义会写入 manifest 和标准化配置。完整源由格式注册表解析真实扩展名；帧窗口只支持能够完整保留元数据的 NPZ 适配器。

单动作 profile 消融入口为：

```bash
python examples/run_orientation_ablation.py \
  --robot e1 \
  --task-type robot_only \
  --data-format noetix_mocap \
  --data-path demo_data/noetix_mocap/0724_BEITI \
  --task-name breaking+hippop.bvh_Skeleton1 \
  --output-root demo_results/v1 \
  --variants baseline root feet gmr_legs shoulders upper full balanced_optimal \
  --dry-run
```

JSON 候选权重搜索入口为：

```bash
python examples/search_orientation_weights.py \
  --candidate-file examples/orientation_weight_candidates_finalists.json \
  --output-root demo_results/v1
```

整个 Noetix BVH 集的零权重与 `balanced_optimal` 对照入口为：

```bash
python examples/run_orientation_comparison_batch.py \
  --data-path demo_data/noetix_mocap/0724_BEITI \
  --bvh-path demo_data/noetix_ori/0724_BEITI \
  --output-root demo_results/v1 \
  --dry-run
```

`--bvh-path` 不是位置朝向估计器。只有对应的标准 NPZ 缺少直接朝向字段，而且原始 BVH 通过严格的关节顺序、FPS 与逐帧时间、以及位置轨迹兼容性检查时，对照入口才会从 BVH rotation channels 恢复 `bvh_rotation_channels_fk` 直接朝向；任何兼容性检查失败都会闭锁拒绝，绝不会根据 NPZ 位置、骨向量或脚尖方向补算朝向。若 NPZ 已包含合法的直接朝向组，则以 NPZ 内被 digest 绑定的 tensor 为准。

消融 artifact 统一保存在 `demo_results/v1/ablations/<experiment>/<variant>/.../identity.npz`，manifest 与消融结果放在同一实验树中。单动作朝向消融的 summary 使用不会相互覆盖的作用域路径 `demo_results/v1/ablations/orientation_ablation/_summaries/<robot>/<task>/<format>/<dataset>/<sequence>/summary.json`；并发机器人、任务、格式、数据集和序列不会再争用全局 `summary.json`。消融结果与正式 canonical artifact 分开，但保存 schema、人体和机器人拓扑、Interaction Mesh 与朝向来源的契约相同。

## 正式 G1/E1 全量重建

Noetix BVH 与 Noetix CSV 攀爬都属于正式全矩阵，不是可选数据集。先做只读规划：

```bash
python examples/rebuild_demo_results.py \
  --run-id demo-rebuild-v1 \
  --max-workers 24 \
  --dry-run
```

执行、续跑、严格验证并事务性提升：

```bash
python examples/rebuild_demo_results.py \
  --run-id demo-rebuild-v1 \
  --max-workers 24 \
  --promote
```

稳定 staging 为 `demo_results/.staging/demo-rebuild-v1/v1`，报告为 `demo_results/runs/demo-rebuild-v1/report.json`，正式结果为 `demo_results/v1`。G1 是全量严格目标：每个已规划源和 variant 都必须生成 artifact，并通过精确集合、源路径与 hash、配置 hash、帧数、人体完整关节名和父树、预期直接朝向 tensor 的名称顺序、精确值与 digest 或 absent、机器人完整 link 名和父树、Interaction Mesh、最终接受的 float64 qpos、六项真实几何残差、三项最终 ConstraintMode、由模式精确派生的帧列表、solver diagnostics、外部资产闭包及分组释放质量阈值。E1 会真实尝试相同矩阵，但其形态可能让部分物体交互和攀爬动作在结构上无法到达；只有实际执行过 job，并在汇总报告中保留源、job 身份、任务族和有界失败原因后，才可登记为显式结构性缺口。成功项必须严格验证，不得伪造 artifact、静默跳过或放宽硬约束；E1 robot-only 仍作为预期高覆盖子集。提升门禁因此要求完整严格的 G1，以及每项均有真实尝试和明确成功或结构性缺口结论的 E1。

每个非 ground artifact 还通过 `object_asset_manifest_json` 与 `object_asset_manifest_sha256` 固定 URDF 及其直接引用的 mesh/texture，strict 可视化、续跑、验证和提升都会重算闭包并在任一文件缺失或变化时闭锁失败。提升阶段持有 results-state 独占锁并重新核对 staging 内容 digest 和外部资产闭包；已有正式 `v1` 时必须使用原子目录交换，不支持该能力的文件系统会在不移动任一树的前提下拒绝提升。同一 `run_id` 的非阻塞独占锁覆盖规划、执行、验证、提升与最终报告，避免并发编排器混写。

提升完成后，可先只读规划旧结果整理：

```bash
python examples/organize_demo_results.py \
  --promoted-run-id demo-rebuild-v1
```

核对计划后以 `python examples/organize_demo_results.py --promoted-run-id demo-rebuild-v1 --execute` 显式执行。`--promoted-run-id` 会把事务绑定到成功的提升报告与当前正式 `v1` 内容指纹。整理器只处理代码中列出的旧 `demo_results` 按机器人目录与旧 sibling 结果树，写入 `demo_results/archive/legacy/<UTC timestamp>/manifest.json`，拒绝未知条目和符号链接，持有与 promotion 共用的 results-state 独占锁，并校验内容绑定的树快照；中断后使用相同固定时间戳加 `--resume --execute`，持久 manifest 会逐项核对 source/target 状态。`v1`、`runs`、`.staging`、`.generated-assets`、`.locks` 和既有 archive 都会保留。

## 默认结果字段

每个 schema-v2 结果保存完整预处理人体骨架 `human_joints`、`human_joint_names` 和 `human_joint_parent_indices`，位置目标采用的 `mapped_human_joints` 与 `mapped_robot_joints`，完整机器人子树 `robot_link_positions`、`robot_link_quaternions_wxyz`、`robot_link_names` 和 `robot_link_parent_indices`，以及 actuated joint 名称。机器人字段是逐帧精确 FK 结果，因此可视化显示完整机器人 link 骨架和 link 朝向时不需要重新推断。canonical `qpos` 必须是 float64；每一行都是与 cost、残差和最终模式对应的精确接受 SQP 状态，也以相同精度供增强任务复用为 nominal warm start。查看器或下游工具可以转换载入副本，但 writer 不能把 canonical qpos 降成 float32。

`qpos_layout`、`object_poses_demo`、`object_poses_target`、`object_pose_layout`、逐帧 cost、SQP 迭代与停止原因、源/配置 hash、dataset partition、sequence key、experiment、variant、run kind、缩放和预处理来源共同构成自描述身份。动态物体任务还会整组保存示范/目标的局部与逐帧世界坐标关键点。非 ground 结果必填 `object_urdf_sha256`、`object_asset_manifest_json` 与 `object_asset_manifest_sha256`，把 URDF 和直接引用的 mesh/texture 闭包固定到字节级身份。`config_json` 的身份还包含求解实现、Python/数值运行时与线程环境、机器人模型树和适用的 Noetix 攀爬场景资产指纹，避免代码、运行时或场景变化后误续跑旧结果。

Interaction Mesh 是 canonical schema-v2 的必需字段组，包括逐帧源/目标顶点、打包四面体、逐帧四面体数、人体与物体顶点数和默认边模式。单动作、批量/增强与消融接口共享的配置校验会拒绝关闭 `save_interaction_mesh`。

源人体朝向字段严格整组出现，包括 `human_orientation_joint_names`、`human_orientation_quaternions_wxyz` 与 `human_orientation_sha256`；digest 绑定具名关节顺序和精确 float32 tensor 字节。Noetix BVH 保存 `bvh_rotation_channels_fk` 直接 BVH 子集，Noetix CSV 保存 `bone_rotation_channels_fk` 的全部直接 Bone Rotation 映射，generic climb 完全不保存。identity 和 augmentation artifact 都保持完全相同的适配器源 tensor 与 digest；增强物体旋转产生的求解目标单独保存在 `orientation_target_quaternions_wxyz`，绝不能覆盖 `human_orientation_quaternions_wxyz`。所有机器人结果都保存完整 link 朝向；源朝向、目标朝向与机器人朝向不能混淆。朝向优化诊断字段组始终存在，没有可直接支持的 joint/link 目标时使用零宽数组。

每个 QP proposal 都会在真实 MuJoCo 几何上重新检查。完整 proposal 不可行而 incumbent 可行时，沿两者之间的二分回溯会返回最大的已知真实几何可行点；incumbent 本身不可行时仍允许前进并在下一次 SQP 重新线性化。约束回退是外层 ConstraintMode 调度：启用 foot sticking 的帧让 `normal`、确实产生放宽效果时的 `relaxed`、以及配置允许时的 `released` 脚部模式分别从同一个 frame-entry 状态运行完整 SQP；只有三者都没有可行候选且明确允许释放物体非穿透时，才在物体约束释放后按相同顺序重跑。只要更严格模式曾得到任何可行候选，就保存最严格模式最后一个真实几何可行候选。

默认的第 0 帧地面重试只处理一种狭窄的纯机器人初始化失败，不会释放任何物理可行性约束。对于没有 nominal 轨迹、浮动基座 Z 参与优化的 identity 或 ablation `robot_only` 任务，只有结构化失败证明地面非穿透是唯一违反的硬物理约束、对象确为水平地面，而且触发重试的这次失败没有使用物理约束 release 或 trust-region release 时，才从原始 frame-entry 状态严格重试一次。重试只把 root Z 抬高精确穿透修正量与共享严格内侧裕量，重新检查全部硬约束后运行同一 SQP。在这次重跑的 SQP 内，原有的第 0 帧算法性 trust-region release 仍可能被选中，并会单独记录；它不释放地面或其他物理约束，每个候选仍必须通过真实几何门禁。该策略不适用于物体交互、攀爬、augmentation、非水平地面或混合失败。被拒绝或仍失败的重试保持硬失败，原本成功的路径不变。

每个 schema-v2 artifact 都保存八个审计字段，未触发时数值诊断使用零哨兵：`frame_zero_ground_retry_policy`、`frame_zero_ground_retry_eligible`、`frame_zero_ground_retry_triggered`、`frame_zero_ground_retry_initial_min_distance_m`、`frame_zero_ground_retry_corrected_min_distance_m`、`frame_zero_ground_retry_lift_m`、`frame_zero_ground_retry_interior_margin_m` 和 `frame_zero_ground_retry_initial_sqp_iterations`。触发后第 0 帧的 `sqp_stop_reasons` 必须带 `frame_zero_ground_retry:` 前缀；严格验证会交叉核对资格、单轴距离等式、配置穿透容差、共享裕量、迭代数和停止原因。

六个 float64 数组保存最终接受状态相对配置容差的非负超量：`ground_non_penetration_violation`、`object_non_penetration_violation`、`foot_sticking_violation`、`foot_lock_violation`、`self_collision_violation` 和 `joint_limits_violation`。地面、foot lock、自碰撞和关节限位始终为硬约束；物体约束只有在最终物体模式已释放时才不再是硬约束；foot sticking 在 `normal` 与 `relaxed` 模式下为硬约束。释放不等于不记录，所有六项仍按真实几何测量。

每帧最终模式保存在 `constraint_mode_foot_sticking`、`constraint_mode_object_non_penetration_released` 与 `constraint_mode_trust_region_released`；脚部取值为 `inactive`、`normal`、`relaxed` 或 `released`，trust-region release 只允许第 0 帧。`foot_sticking_fallback_frames`、`foot_sticking_release_frames` 与 `object_non_penetration_release_frames` 必须由这些最终模式精确派生，失败尝试不得进入帧列表。完整序列 foot-sticking retry 的结果通过 `foot_sticking_enabled_for_saved_trajectory` 和 `foot_sticking_full_sequence_retry_frame` 固定，`sqp_iteration_counts` 则累计一帧中所有已尝试模式的线性化迭代。最终接受的物体约束释放仍是降级结果，不能算作碰撞验收通过。

## 可视化

检查转换后的 Noetix BVH 源人体与直接朝向轴：

```bash
python viser_player.py \
  --input-path demo_data/noetix_mocap/0724_BEITI/breaking+hippop.bvh_Skeleton1.npz \
  --input-kind raw \
  --show-source-orientation-axes
```

检查一个标准重定向结果及全部默认图层：

```bash
python viser_player.py \
  --input-path demo_results/v1/canonical/e1/robot_only/noetix_mocap/noetix_mocap/run_to_the_right/identity.npz \
  --show-human-skeleton \
  --show-robot-skeleton \
  --show-interaction-mesh \
  --show-source-orientation-axes \
  --show-robot-orientation-axes
```

同步比较多个消融结果时，使用唯一多动作入口：

```bash
python multi_viser_player.py \
  --qpos-npzs \
    demo_results/v1/ablations/orientation_ablation/baseline/e1/robot_only/noetix_mocap/0724_BEITI/breaking_hippop.bvh_Skeleton1/identity.npz \
    demo_results/v1/ablations/orientation_ablation/balanced_optimal/e1/robot_only/noetix_mocap/0724_BEITI/breaking_hippop.bvh_Skeleton1/identity.npz \
  --labels baseline balanced_optimal \
  --orientation-joints LeftArm RightArm \
  --show-target-orientation-axes \
  --show-robot-orientation-axes
```

`viser_player.py` 是唯一单动作可视化入口，自动适配 result、raw 与 converted 输入；`multi_viser_player.py` 是唯一多动作入口。两者共享版本化结果加载器和统一图层语义，因此可在完整元数据已经保存后自由选择映射人体骨架、完整机器人 link 骨架与 mesh、物体关键点、Interaction Mesh、foot sticking，以及直接源朝向、目标朝向和机器人 link 朝向轴。可视化可在内存中按需转换数值副本，但 artifact 中 canonical qpos 仍保持 float64。多动作同步前会强制核对源 digest、dataset/机器人/格式/任务身份、人体与机器人拓扑、actuated joint 顺序及直接人体朝向 tensor identity。完整人体关节轨迹与父树始终保留在 artifact 中；raw 输入模式可直接查看完整源骨架。
