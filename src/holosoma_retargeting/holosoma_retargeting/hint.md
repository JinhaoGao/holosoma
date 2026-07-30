# 统一重定向与可视化命令速查

本文档记录当前仓库统一后的重定向与可视化入口。公开重定向接口分为单动作、批量/增强和受控消融三类；正式数据集重建脚本只负责矩阵编排、验证和发布，不维护第四套求解或保存实现。`viser_player.py` 用于查看单个原始动作或结果，`multi_viser_player.py` 用于同步比较同一个源动作的多个重定向结果。

重定向命令统一从 Python 工程根目录以模块方式执行，避免依赖包是否已经 editable install：

```bash
cd /home/jinhaogao/holosoma/src/holosoma_retargeting
conda activate robot_retargeter
```

可视化命令仍从包含数据和模型的包目录执行：

```bash
cd /home/jinhaogao/holosoma/src/holosoma_retargeting/holosoma_retargeting
conda activate robot_retargeter
```

## 统一重定向接口

当前阶段已经停止数据遍历，下面只记录经过统一后的命令规范，不表示应立即恢复重定向。单动作与批处理共享 `RetargetJob`、运动适配器、预处理、SQP 求解、schema-v2 保存和严格续跑检查。identity artifact 的 `run_kind` 固定为 `single`，非 identity 动作变换固定为 `augmentation`，不改变动作而只改变实验参数的结果固定为 `ablation`。

默认 artifact 会保存完整预处理人体骨架、完整机器人 link 骨架与 `wxyz` 朝向、映射关键点、Interaction Mesh、物体与资产身份、最终 `float64` qpos、约束模式、残差和回退诊断，不需要额外保存开关。人体朝向只在源格式直接包含旋转并且适配器能够证明来源时保存；仅含位置的数据会明确写入 `orientation_source="absent"`，不会从位置或骨向量估计朝向。不要为正式结果传入 `--retargeter.no-save-interaction-mesh`。

### 单动作普通重定向

`robot_retarget` 是唯一的单动作入口。`--save-dir` 表示结果树根，不是机器人或任务的叶目录。下面使用被 Git 忽略的 `rt_results` scratch 根，避免手工命令直接污染正式 `demo_results/v1` 或 `rebuild_demo_results` 专属的 `.staging/<run-id>/v1`：

```bash
python -m holosoma_retargeting.examples.robot_retarget \
  --data-path holosoma_retargeting/demo_data/lafan \
  --task-type robot_only \
  --task-name dance2_subject1 \
  --data-format lafan \
  --robot g1 \
  --save-dir holosoma_retargeting/rt_results/manual-stage-v1
```

OMOMO 物体交互和通用攀爬仍使用同一个入口，只修改任务与数据参数：

```bash
python -m holosoma_retargeting.examples.robot_retarget \
  --data-path holosoma_retargeting/demo_data/OMOMO_new \
  --task-type object_interaction \
  --task-name sub10_tripod_000 \
  --data-format omomo \
  --robot e1 \
  --save-dir holosoma_retargeting/rt_results/manual-stage-v1

python -m holosoma_retargeting.examples.robot_retarget \
  --data-path holosoma_retargeting/demo_data/climb \
  --task-type climbing \
  --task-name mocap_climb_seq_0 \
  --data-format mocap \
  --robot g1 \
  --save-dir holosoma_retargeting/rt_results/manual-stage-v1
```

climbing 的输入目录必须同时提供动作、`multi_boxes` 资产和对应机器人的未缩放 MuJoCo 场景；E1 需要每个 sequence 下存在 `e1_23dof_w_multi_boxes.xml`。这些 E1 场景目前属于本机 demo dataset，并未随 Git checkout 分发，因此在新环境规划或执行 E1 climbing 前必须先恢复同一份外部数据资产，不能把 G1 场景当作 E1 fallback。

同一条命令在 artifact 的 schema、源路径与 SHA-256、标准化配置、solver identity、run kind 和 variant 全部匹配时会严格续用已有结果。正常恢复不要传 `--overwrite-existing`；只有明确决定替换同路径结果时才使用 `--overwrite-existing`。

### 单动作增强重定向

为物体交互或攀爬命令增加 `--augmentation` 即可运行 identity-first 动作族。OMOMO 会生成 identity、三个平移和两个旋转结果；climbing 会生成 identity 和四个高度缩放结果；robot-only 即使传入该开关也仍只有 identity。

```bash
python -m holosoma_retargeting.examples.robot_retarget \
  --data-path holosoma_retargeting/demo_data/OMOMO_new \
  --task-type object_interaction \
  --task-name sub10_tripod_000 \
  --data-format omomo \
  --robot g1 \
  --save-dir holosoma_retargeting/rt_results/manual-augmentation-v1 \
  --augmentation
```

### 普通批量与批量增强

`parallel_robot_retarget` 是唯一的多源入口。不启动求解器的预检与计划模式添加 `--dry-run`；该模式仍会创建结果目录并写入或覆盖 batch report，因此不是文件系统只读。正式执行或恢复时删除该开关并保持其余参数、`run-id` 与结果根不变。匹配的 artifact 会跳过，缺失项才会继续执行。

```bash
python -m holosoma_retargeting.examples.parallel_robot_retarget \
  --data-dir holosoma_retargeting/demo_data/gvhmr \
  --task-type robot_only \
  --data-format gvhmr \
  --robot g1 \
  --save-dir holosoma_retargeting/rt_results/manual-batch-v1 \
  --max-workers 4 \
  --run-id g1-gvhmr \
  --dry-run
```

批量增强仍使用这个入口，只增加 `--augmentation`。完整 OMOMO 输入较大，建议先通过 `--object-names` 限定类别验证命令，再决定是否移除过滤条件。

```bash
python -m holosoma_retargeting.examples.parallel_robot_retarget \
  --data-dir holosoma_retargeting/demo_data/OMOMO_new \
  --task-type object_interaction \
  --data-format omomo \
  --robot g1 \
  --save-dir holosoma_retargeting/rt_results/manual-augmentation-v1 \
  --object-names largebox \
  --augmentation \
  --max-workers 4 \
  --run-id g1-omomo-largebox \
  --dry-run
```

### 消融实验

消融脚本同样调用共享 job 与 artifact 管线，结果写入所选结果根的 `ablations/<experiment>/...`。下面的 `--dry-run` 不启动求解器或生成结果 NPZ，但仍会创建目录、逐 variant manifest 和汇总 JSON；删除该开关后才会实际求解。恢复时重复相同命令，不要传 `--overwrite`。

```bash
python -m holosoma_retargeting.examples.run_orientation_ablation \
  --robot e1 \
  --task-type robot_only \
  --data-format noetix_mocap \
  --data-path holosoma_retargeting/demo_data/noetix_mocap/0724_BEITI \
  --task-name "breaking+hippop.bvh_Skeleton1" \
  --output-root holosoma_retargeting/rt_results/manual-ablation-v1 \
  --variants baseline balanced_optimal \
  --dry-run
```

### 正式 G1/E1 数据矩阵

`rebuild_demo_results` 负责枚举、批量调用、严格验证和事务性发布。当前暂停的普通重定向采用 identity-only 模式，因此从规划到执行、验证和发布都必须保留同一个 `--no-include-augmentation-variants`，并使用同一个 `run-id`。下面第一条命令只写计划，不启动求解器：

```bash
python -m holosoma_retargeting.examples.rebuild_demo_results \
  --run-id demo-normal-identity-v1 \
  --no-include-augmentation-variants \
  --max-workers 16 \
  --dry-run
```

得到后续授权时，删除 `--dry-run` 即可执行或严格续跑。仅验证既有 staging 时使用：

```bash
python -m holosoma_retargeting.examples.rebuild_demo_results \
  --run-id demo-normal-identity-v1 \
  --no-include-augmentation-variants \
  --validate-only
```

只有完整矩阵、持久化 batch reports 与质量门禁全部通过后，才允许以相同模式事务性发布：

```bash
python -m holosoma_retargeting.examples.rebuild_demo_results \
  --run-id demo-normal-identity-v1 \
  --no-include-augmentation-variants \
  --validate-only \
  --promote
```

将来恢复全增强矩阵时必须使用新的 `run-id`，并移除 identity-only 开关。增强矩阵和 identity-only 矩阵不能共用 staging 或 `run-id`。

手工 scratch 根中的单动作、批处理报告、消融 manifest 和 summary 不构成正式矩阵，不能交给 `rebuild_demo_results --validate-only` 或 `--promote`。正式 staging 只能由 `rebuild_demo_results` 按完整计划创建和维护。

### 结果目录整理

正式结果只应出现在 `demo_results/v1`，运行中结果位于 `demo_results/.staging/<run-id>/v1`，报告位于 `demo_results/runs/<run-id>`，历史正式版本与旧布局位于 `demo_results/archive`。整理器默认只读，可在不求解的情况下查看精确移动计划：

```bash
python -m holosoma_retargeting.examples.organize_demo_results
```

旧目录只有在某次完整重建已经成功发布后才能执行归档。执行命令必须绑定该次发布报告；没有成功 promotion 时整理器会拒绝移动：

```bash
python -m holosoma_retargeting.examples.organize_demo_results \
  --promoted-run-id demo-normal-identity-v1 \
  --execute
```

若归档事务中断，必须使用原 manifest 中完全相同的 UTC 时间戳与 promotion 身份恢复，不能省略时间戳重新生成一笔事务：

```bash
python -m holosoma_retargeting.examples.organize_demo_results \
  --timestamp-utc 20260730T120000000000Z \
  --promoted-run-id demo-normal-identity-v1 \
  --resume \
  --execute
```

### 重定向帮助

```bash
python -m holosoma_retargeting.examples.robot_retarget --help
python -m holosoma_retargeting.examples.parallel_robot_retarget --help
python -m holosoma_retargeting.examples.run_orientation_ablation --help
python -m holosoma_retargeting.examples.rebuild_demo_results --help
python -m holosoma_retargeting.examples.organize_demo_results --help
```

## 统一可视化接口

以下可视化命令假设已经切换到 `/home/jinhaogao/holosoma/src/holosoma_retargeting/holosoma_retargeting`。Viser 启动后会在终端打印浏览器地址，通常是 `http://localhost:8080`；端口被占用时会自动选择下一个端口。

## 单动作可视化

对于带完整元数据的新结果，只需要提供 `--input-path`。`--input-kind auto` 是默认值，可以省略。

### G1 GVHMR robot-only

```bash
python viser_player.py \
  --input-path demo_results/g1/robot_only/gvhmr/tennis.npz \
  --show-mapped-skeletons \
  --show-joint-labels \
  --loop
```

### E1 Noetix robot-only

```bash
python viser_player.py \
  --input-path "demo_results/e1/robot_only/noetix/0724_BEITI/breaking+hippop.bvh_Skeleton1.npz" \
  --show-mapped-skeletons \
  --show-joint-labels \
  --robot-mesh-opacity 0.45 \
  --loop
```

### G1 object-interaction 完整图层

下面的结果包含物体关键点和 Interaction Mesh。红色关键点是 source/demo 物体，青色关键点是 target 物体。

```bash
python viser_player.py \
  --input-path demo_results_parallel/g1/object_interaction/omomo/sub8_largebox_042_original.npz \
  --show-mapped-skeletons \
  --show-object-keypoints \
  --show-interaction-mesh \
  --interaction-mesh-mode both \
  --interaction-mesh-edges cross \
  --robot-mesh-opacity 0.45 \
  --object-mesh-opacity 0.25 \
  --skeleton-point-radius 0.018 \
  --skeleton-line-width 2.5 \
  --interaction-mesh-line-width 1.0 \
  --loop
```

Interaction Mesh 支持 `source/cross`、`source/all`、`target/cross`、`target/all`、`both/cross` 和 `both/all` 六种组合。例如只看目标机器人到物体的全部边：

```bash
python viser_player.py \
  --input-path demo_results_parallel/g1/object_interaction/omomo/sub8_largebox_042_original.npz \
  --show-mapped-skeletons \
  --show-object-keypoints \
  --show-interaction-mesh \
  --interaction-mesh-mode target \
  --interaction-mesh-edges all
```

### E1 旧格式 object-interaction

当前 E1 object-interaction 文件属于较早格式，缺少机器人和物体元数据，因此需要显式提供资产。

```bash
python viser_player.py \
  --input-path demo_results/e1/object_interaction/omomo/sub3_largebox_003_original.npz \
  --robot-urdf models/e1/e1_23dof.urdf \
  --robot-mujoco-xml models/e1/e1_23dof.xml \
  --object-urdf models/largebox/largebox.urdf \
  --show-mapped-skeletons \
  --show-interaction-mesh \
  --interaction-mesh-mode both \
  --interaction-mesh-edges cross \
  --robot-mesh-opacity 0.45 \
  --object-mesh-opacity 0.25
```

### G1 climbing

带完整元数据的 G1 climbing 结果会自动读取保存的静态攀爬场景 URDF。

```bash
python viser_player.py \
  --input-path demo_results/g1/climbing/mocap/mocap_climb_seq_0_original.npz \
  --show-mapped-skeletons \
  --robot-mesh-opacity 0.45 \
  --object-mesh-opacity 0.35 \
  --loop
```

### E1 旧格式 climbing

旧的 E1 climbing 结果需要显式指定机器人和对应的静态攀爬场景。

```bash
python viser_player.py \
  --input-path demo_results/e1/climbing/mocap_climb/mocap_climb_seq_3_original.npz \
  --robot-urdf models/e1/e1_23dof.urdf \
  --robot-mujoco-xml models/e1/e1_23dof.xml \
  --object-urdf demo_data/climb/mocap_climb_seq_3/multi_boxes.urdf \
  --show-mapped-skeletons \
  --robot-mesh-opacity 0.45 \
  --object-mesh-opacity 0.35
```

### 单个 OMOMO Laplacian relation 消融结果

旧消融文件没有预存机器人骨架时，查看器会使用 MuJoCo FK 恢复。

```bash
python viser_player.py \
  --input-path demo_results_ablation/laplacian_relation/relation_neighbor_plan_full/omomo/O0_uniform/sub3_largebox_003_original.npz \
  --robot-urdf models/g1/g1_29dof.urdf \
  --robot-mujoco-xml models/g1/g1_29dof.xml \
  --object-urdf models/largebox/largebox.urdf \
  --show-mapped-skeletons \
  --show-object-keypoints \
  --show-interaction-mesh \
  --interaction-mesh-mode both \
  --interaction-mesh-edges cross
```

### 单个 climbing relation 消融结果

```bash
python viser_player.py \
  --input-path demo_results_ablation/laplacian_relation/relation_neighbor_plan_full/climbing/C3_foot_object_x5/mocap_climb_seq_0_original.npz \
  --robot-urdf models/g1/g1_29dof_spherehand.urdf \
  --robot-mujoco-xml models/g1/g1_29dof_spherehand.xml \
  --object-urdf demo_data/climb/mocap_climb_seq_0/multi_boxes.urdf \
  --show-mapped-skeletons \
  --show-interaction-mesh
```

### 单个 LAFAN relation 消融结果

```bash
python viser_player.py \
  --input-path demo_results_ablation/laplacian_relation/relation_neighbor_plan_full/lafan/L2_foot_ground_x20/dance2_subject1.npz \
  --robot-urdf models/g1/g1_29dof.urdf \
  --robot-mujoco-xml models/g1/g1_29dof.xml \
  --show-mapped-skeletons \
  --show-interaction-mesh
```

### 单个方向约束结果

下面的命令同时显示 target frame 和实际 robot frame，并只显示肩膀与脚部四组姿态轴。去掉 `--orientation-joints` 就会显示结果中保存的全部方向关节。

```bash
python viser_player.py \
  --input-path "demo_results_orientation/my_shoulder_feet_run/shoulders_feet_1/shoulders_feet/breaking+hippop.bvh_Skeleton1.npz" \
  --show-mapped-skeletons \
  --show-target-orientation-axes \
  --show-robot-orientation-axes \
  --orientation-joints LeftArm RightArm LeftFoot RightFoot \
  --orientation-axis-length 0.065 \
  --robot-mesh-opacity 0.35
```

### 单个 foot-sticking 结果

Foot-sticking 状态面板默认开启，也可以显式传入 `--show-foot-sticking`。

```bash
python viser_player.py \
  --input-path "demo_results_foot_ablation/e1/hard_on/breaking+hippop.bvh_Skeleton1.npz" \
  --show-mapped-skeletons \
  --show-foot-sticking \
  --loop
```

## 多动作与多结果同步比较

`multi_viser_player.py` 用于同一个源动作的多个同步结果。所有输入必须具有相同帧数、FPS、机器人类型和源人体轨迹。不能把 tennis、dance 和 climbing 等互不相关的动作放进同一次调用，也不能直接混合 G1 与 E1。

`--family` 和 `--qpos-npzs` 二选一，不能同时传入。使用 `--qpos-npzs` 时，`--labels` 的数量应与结果数量一致。

### Object-interaction 完整增强族

下面的命令会自动查找 `original`、三个平移结果和两个旋转结果。

```bash
python multi_viser_player.py \
  --family demo_results_parallel/g1/object_interaction/omomo/sub8_largebox_042_original.npz \
  --show-object-keypoints \
  --show-interaction-mesh \
  --interaction-mesh-mode both \
  --interaction-mesh-edges cross \
  --robot-mesh-opacity 0.45 \
  --object-mesh-opacity 0.22 \
  --x-offset 0.6 \
  --loop
```

只比较增强族中的部分结果，可以通过 `--variants` 选择。

```bash
python multi_viser_player.py \
  --family demo_results_parallel/g1/object_interaction/omomo/sub8_largebox_042_original.npz \
  --variants original trans_0 rot_0 rot_1 \
  --show-object-keypoints \
  --show-interaction-mesh
```

### OMOMO Laplacian relation 七组消融

Shell glob 会按 O0 到 O6 排序，后面的 labels 与这个顺序对应。

```bash
python multi_viser_player.py \
  --qpos-npzs demo_results_ablation/laplacian_relation/relation_neighbor_plan_full/omomo/O*/sub3_largebox_003_original.npz \
  --labels O0_uniform O1_hand_x5 O2_hand_x20 O3_hand_body_down O4_wrist_x20 O5_symmetric O6_body_x3 \
  --robot-urdf models/g1/g1_29dof.urdf \
  --robot-mujoco-xml models/g1/g1_29dof.xml \
  --object-urdf models/largebox/largebox.urdf \
  --show-object-keypoints \
  --show-interaction-mesh \
  --interaction-mesh-mode both \
  --interaction-mesh-edges cross \
  --robot-mesh-opacity 0.38 \
  --object-mesh-opacity 0.18 \
  --x-offset 0.55
```

### Climbing Laplacian relation 八组消融

静态攀爬地形也会在多动作模式中显示。

```bash
python multi_viser_player.py \
  --qpos-npzs demo_results_ablation/laplacian_relation/relation_neighbor_plan_full/climbing/C*/mocap_climb_seq_0_original.npz \
  --labels C0_uniform C1_hand_x5 C2_hand_x20 C3_foot_x5 C4_foot_x20 C5_root_ground C6_limb_focus C7_body_x3 \
  --robot-urdf models/g1/g1_29dof_spherehand.urdf \
  --robot-mujoco-xml models/g1/g1_29dof_spherehand.xml \
  --object-urdf demo_data/climb/mocap_climb_seq_0/multi_boxes.urdf \
  --show-interaction-mesh \
  --interaction-mesh-mode both \
  --interaction-mesh-edges cross \
  --robot-mesh-opacity 0.38 \
  --object-mesh-opacity 0.20 \
  --x-offset 0.55
```

### LAFAN Laplacian relation 七组消融

```bash
python multi_viser_player.py \
  --qpos-npzs demo_results_ablation/laplacian_relation/relation_neighbor_plan_full/lafan/L*/dance2_subject1.npz \
  --labels L0_uniform L1_foot_x5 L2_foot_x20 L3_root_x5 L4_body_x3 L5_body_x5 L6_foot_body_down \
  --robot-urdf models/g1/g1_29dof.urdf \
  --robot-mujoco-xml models/g1/g1_29dof.xml \
  --show-interaction-mesh \
  --robot-mesh-opacity 0.4 \
  --x-offset 0.55
```

### Object-interaction Laplacian residual weight 三组消融

```bash
python multi_viser_player.py \
  --qpos-npzs demo_results_ablation/laplacian_weight/residual_focus_20260708_root_effector_v1/object_interaction/sub3_largebox_003/*/sub3_largebox_003_original.npz \
  --labels end_effector_x20 end_effector_x5 uniform \
  --robot-urdf models/g1/g1_29dof.urdf \
  --robot-mujoco-xml models/g1/g1_29dof.xml \
  --object-urdf models/largebox/largebox.urdf \
  --show-object-keypoints \
  --show-interaction-mesh \
  --robot-mesh-opacity 0.4 \
  --object-mesh-opacity 0.2
```

### Climbing Laplacian residual weight 三组消融

```bash
python multi_viser_player.py \
  --qpos-npzs demo_results_ablation/laplacian_weight/residual_focus_20260708_root_effector_v1/climbing/mocap_climb_seq_0/*/mocap_climb_seq_0_original.npz \
  --labels end_effector_x20 end_effector_x5 uniform \
  --robot-urdf models/g1/g1_29dof_spherehand.urdf \
  --robot-mujoco-xml models/g1/g1_29dof_spherehand.xml \
  --object-urdf demo_data/climb/mocap_climb_seq_0/multi_boxes.urdf \
  --show-interaction-mesh \
  --robot-mesh-opacity 0.4 \
  --object-mesh-opacity 0.2
```

### Robot-only LAFAN residual weight 三组消融

```bash
python multi_viser_player.py \
  --qpos-npzs demo_results_ablation/laplacian_weight/residual_focus_20260708_root_effector_v1/robot_only/dance2_subject1/*/dance2_subject1.npz \
  --labels root_triangle_x20 root_triangle_x5 uniform \
  --robot-urdf models/g1/g1_29dof.urdf \
  --robot-mujoco-xml models/g1/g1_29dof.xml \
  --show-interaction-mesh \
  --robot-mesh-opacity 0.4
```

### Foot-sticking hard-off 与 hard-on

`Foot sticking` 面板会分别显示两条结果当前帧的左右脚状态及约束回退状态。

```bash
python multi_viser_player.py \
  --qpos-npzs \
    "demo_results_foot_ablation/e1/hard_off/breaking+hippop.bvh_Skeleton1.npz" \
    "demo_results_foot_ablation/e1/hard_on/breaking+hippop.bvh_Skeleton1.npz" \
  --labels hard_off hard_on \
  --show-foot-sticking \
  --robot-mesh-opacity 0.4 \
  --x-offset 0.7 \
  --loop
```

### Shoulder-and-feet 方向权重

```bash
python multi_viser_player.py \
  --qpos-npzs demo_results_orientation/my_shoulder_feet_run/shoulders_feet_*/*/breaking+hippop.bvh_Skeleton1.npz \
  --labels shoulder_feet_x1 shoulder_feet_x10 shoulder_feet_x100 \
  --show-target-orientation-axes \
  --show-robot-orientation-axes \
  --orientation-joints LeftArm RightArm LeftFoot RightFoot \
  --robot-mesh-opacity 0.35 \
  --x-offset 0.7
```

### Full-equal 方向权重

```bash
python multi_viser_player.py \
  --qpos-npzs demo_results_orientation/my_full_equal_run/full_equal_*/*/breaking+hippop.bvh_Skeleton1.npz \
  --labels full_equal_x1 full_equal_x10 full_equal_x100 \
  --show-target-orientation-axes \
  --show-robot-orientation-axes \
  --robot-mesh-opacity 0.35 \
  --x-offset 0.7
```

### Shoulder-only baseline 与多种权重

```bash
python multi_viser_player.py \
  --qpos-npzs \
    "demo_results_orientation/my_shoulder_run/baseline/breaking+hippop.bvh_Skeleton1.npz" \
    demo_results_orientation/my_shoulder_run/shoulders_x*/breaking+hippop.bvh_Skeleton1.npz \
  --labels baseline shoulder_x0.025 shoulder_x0.1 shoulder_x0.25 shoulder_x0.5 shoulder_x10 shoulder_x100 \
  --show-target-orientation-axes \
  --show-robot-orientation-axes \
  --orientation-joints LeftArm RightArm \
  --robot-mesh-opacity 0.35 \
  --x-offset 0.65
```

### Full-sequence 方向搜索方案

方向搜索的 coarse、fine 和 full-sequence 目录使用同一种写法，但必须选取相同 window。下面比较 full-sequence 中的几个代表性方案。

```bash
python multi_viser_player.py \
  --qpos-npzs \
    "demo_results_orientation/orientation_weight_search/full_sequence/runs/baseline/window_000000/breaking+hippop.bvh_Skeleton1.npz" \
    "demo_results_orientation/orientation_weight_search/full_sequence/runs/tapered/window_000000/breaking+hippop.bvh_Skeleton1.npz" \
    "demo_results_orientation/orientation_weight_search/full_sequence/runs/mixed_a/window_000000/breaking+hippop.bvh_Skeleton1.npz" \
    "demo_results_orientation/orientation_weight_search/full_sequence/runs/position_guard/window_000000/breaking+hippop.bvh_Skeleton1.npz" \
  --labels baseline tapered mixed_a position_guard \
  --show-target-orientation-axes \
  --show-robot-orientation-axes \
  --robot-mesh-opacity 0.35
```

## 显示图层组合

所有显示开关都可以正交组合，不需要为每一种布尔排列编写独立命令。

单动作中，`--show-mapped-skeletons` 会一起开启人体骨架、机器人骨架和可用的手部细节。需要拆开控制时，可以使用 `--show-human-skeleton True`、`--show-robot-skeleton False` 和 `--show-human-hands True`。

Mesh 可以通过 `--no-show-meshes` 全部关闭，也可以使用 `--show-robot-mesh True` 与 `--show-object-mesh False` 分别控制。其余可叠加项包括 `--show-object-keypoints`、`--show-interaction-mesh`、`--show-foot-sticking`、`--show-target-orientation-axes`、`--show-robot-orientation-axes` 和 `--show-joint-labels`。

多动作模式的开关采用普通正反参数，例如 `--show-robot-mesh`/`--no-show-robot-mesh`、`--show-object-mesh`/`--no-show-object-mesh`、`--show-human-skeleton`/`--no-show-human-skeleton` 和 `--show-robot-skeleton`/`--no-show-robot-skeleton`。启动后还可以在 `Layers` 中全局控制，在 `Motions` 中单独隐藏某个结果。

## 重定向前的原始数据

检查原始 OMOMO 物体交互数据：

```bash
python viser_player.py \
  --input-path demo_data/OMOMO_new/sub3_largebox_003.pt \
  --input-kind raw
```

检查原始 climbing 数据：

```bash
python viser_player.py \
  --input-path demo_data/climb/mocap_climb_seq_0 \
  --input-kind raw
```

检查原始 GVHMR 数据：

```bash
python viser_player.py \
  --input-path demo_data/gvhmr/tennis.npz \
  --input-kind raw \
  --data-format gvhmr
```

`--qpos-npz` 仍然是兼容旧命令的参数，但新命令统一推荐使用 `--input-path`。

## 查看帮助

```bash
python viser_player.py --help
python multi_viser_player.py --help
```
