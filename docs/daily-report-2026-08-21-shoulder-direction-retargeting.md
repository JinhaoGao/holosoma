# 工作日报：肩部方向约束与序列分支消歧重定向

日期：2026 年 8 月 21 日。工作分支为 `feature/shoulder-direction-sequence-retargeting`，工作树为 `/home/jinhaogao/holosoma-shoulder-direction`，功能提交为 `0280d2d`。

今日目标是解决伸展运动中 E1 与 E2 三自由度串联肩关节反复选择歧义解的问题，并将解决方案直接集成进现有 Interaction Mesh 重定向流程。测试动作来自 `20260806_TiCao_1st_ShenzhanMove_f_XJQ_T01_C502AF_QC_CLEAN.bvh`，分别完成了 E1 23DOF、E1 24DOF 和 E2 的整段 676 帧重定向。

问题根因是人体上臂单位方向只有两个独立自由度，而机器人 shoulder pitch、roll、yaw 一共有三个自由度。同一个上臂方向通常对应一条一维关节解流形，并可能因为关节限位与欧拉表示断成多个分支。原来的逐帧优化或多 link 完整 SO(3) 朝向跟踪会在这些等价分支之间跳转，表现为人体方向只发生小幅变化时，机器人肩关节突然变化约 90 至 107 度。ForeArm 完整 SO(3) 还可能要求上游肩关节补偿机器人肘腕无法实现的旋转，进一步诱发错误分支。

本次实现将肩部主任务改为躯干局部坐标系中的上臂单位方向。人体目标定义为 `normalize(ForeArm - Arm)`，机器人目标定义为 `normalize(elbow_link - shoulder_roll_link)`。机器人躯干基由左右 `shoulder_pitch_link` 的稳定关节中心和 torso 构造，方向段则从 `shoulder_roll_link` 指向 elbow，避免把 pitch 到 roll 的肩内结构偏置误算成上臂。方向解析雅可比使用 `J_u = (I - uu^T)(J_elbow - J_shoulder) / ||d||`，并通过有限差分测试验证。

为了处理方向任务留下的一维零空间，重定向开始后、逐帧 SQP 之前会直接基于原始人体序列生成肩部参考。每帧使用确定性的多初值有界最小二乘采样多组肩关节候选，候选节点代价综合上臂方向误差、低权重前臂平面兼容性和关节限位余量，时间边代价包含关节速度与加速度。二阶动态规划选择整段连续候选分支，随后使用全序列凸 QP 平滑离散路径并限制逐帧关节变化。最终 Interaction Mesh SQP 仍直接优化上臂方向、手部位置和其他原有任务，平滑路径只负责约束方向雅可比的零空间分支，并以较低权重帮助跨越离散候选之间的过渡。

启用该模式后，Arm、ForeArm 和 Hand 的通用完整 SO(3) 残差不会再参与肩臂优化。E1 的 Hand 朝向误差只投影到其唯一能够实现该旋转的 `elbow_yaw_joint` 轴，肩关节不会替腕部误差补偿；E2 没有腕部旋转自由度，因此对应权重为零。自然姿态正则化保持原状，本次工作没有加入或修改自然姿态配置。

这套方法不是对旧重定向结果执行第二轮优化。程序加载标准化人体动作后，先在内存中完成肩部分支规划，然后立即进入同一次最终重定向，最后只保存一份机器人结果。它消除了额外的机器人动作后处理环节，但当前实现属于离线序列重定向，因为动态规划和全序列平滑需要提前看到完整动作。如果未来用于实时流式重定向，需要把整段规划替换为有限窗口 beam search 或 receding-horizon 优化。

当前生产入口读取标准化 Noetix NPZ，而不是直接解析任意 BVH。已有测试动作的同名 NPZ 内保存的 `source_bvh` 已确认指向指定原始 BVH，因此最终机器人动作仍然是从原始人体动作生成的，而不是从旧机器人结果生成的。对于尚未转换的新 BVH，可以先执行一次格式转换：

```bash
cd /home/jinhaogao/holosoma-shoulder-direction
export PYTHONPATH=/home/jinhaogao/holosoma-shoulder-direction/src/holosoma_retargeting
/home/jinhaogao/miniconda3/envs/robot_retargeter/bin/python \
  -m holosoma_retargeting.data_utils.convert_noetix_bvh \
  --input-dir /path/to/bvh_directory \
  --output-dir /path/to/noetix_mocap_directory
```

格式转换完成后，肩部方向模式直接通过原重定向命令启用。以下命令会对本次动作依次生成三种机器人结果：

```bash
cd /home/jinhaogao/holosoma-shoulder-direction
export PYTHONPATH=/home/jinhaogao/holosoma-shoulder-direction/src/holosoma_retargeting
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

for robot in e1_23dof e1_24dof e2; do
  /home/jinhaogao/miniconda3/envs/robot_retargeter/bin/python \
    -m holosoma_retargeting.examples.robot_retarget \
    --task robot_only \
    --robot "${robot}" \
    --dataset noetix_mocap \
    --data-path /home/jinhaogao/holosoma/src/holosoma_retargeting/holosoma_retargeting/demo_data/noetix_mocap \
    --motion 20260806_guangboticao/20260806_TiCao_1st_ShenzhanMove_f_XJQ_T01_C502AF_QC_CLEAN \
    --save-dir /home/jinhaogao/holosoma-shoulder-direction/demo_results/shoulder_direction_validation \
    --output-name "20260806_TiCao_1st_ShenzhanMove_f_XJQ_T01_C502AF_QC_CLEAN_${robot}_direction" \
    --orientation-config /home/jinhaogao/holosoma-shoulder-direction/src/holosoma_retargeting/holosoma_retargeting/examples/orientation_weights \
    --shoulder-direction-tracking \
    --overwrite \
    --retargeter.no-visualize
done
```

关键开关是 `--shoulder-direction-tracking`。不传该参数时，默认行为与本地 `dev` 分支保持一致；传入后，肩臂完整 SO(3) 会被方向任务、序列分支参考和 E1 单轴腕部任务替代。结果仍是标准重定向 NPZ，可以直接进入现有播放器、训练数据处理或其他下游流程。

结果中的肩部诊断数据可以用以下命令生成静态图。图中包含实际与规划关节轨迹、人体方向变化与肩关节变化的对应关系、方向误差、雅可比奇异值，以及最坏帧附近的三维候选解流形和选中时间路径：

```bash
/home/jinhaogao/miniconda3/envs/robot_retargeter/bin/python \
  -m holosoma_retargeting.visualization.shoulder_direction \
  /path/to/retargeted_result.npz
```

定量结果显示，旧 E1 24DOF 左右肩最大单帧联合变化为 104.02 度和 106.86 度，旧 E2 为 92.69 度和 104.91 度。新 E1 23DOF 为 37.43 度和 43.27 度，新 E1 24DOF 为 37.40 度和 43.31 度，新 E2 为 40.93 度和 42.56 度；所有单个肩关节逐帧变化均不超过配置的 0.45 rad，即 25.78 度。候选层仍观测到约 100 至 164 度的等价分支跳变，证明歧义解客观存在，但这些跳变没有进入最终机器人轨迹。

新结果的平均上臂方向误差为 E1 左右约 2.66 度和 2.03 度，E2 左右约 2.03 度和 1.72 度。快速环绕帧仍存在约 21 至 25 度的瞬时误差峰值，主要来自人体目标本身单帧变化约 20 至 22.5 度，而最终机器人肩关节受连续性上限约束。这是避免换支和快速跟随之间的明确折中，后续可以结合动画观感继续调整 `max_frame_step_rad`、方向权重和参考权重。

验证方面，Ruff、Python 编译和 Git 差异格式检查均通过。肩部相关测试为 71 passed、74 subtests passed；排除仓库已有的可选可视化依赖、训练/Isaac E2E 环境和缺失 climbing 场景资产后，其余可运行测试为 238 passed、362 subtests passed。三种机器人均完成 676 帧整段重定向，结果中的 `qpos` 与帧代价全部为有限数值。

今日结论是肩部歧义已经从不可见的逐帧局部选择问题，转化为可枚举、可优化、可诊断的序列分支问题，并已直接并入生产重定向阶段。当前版本不需要读取或修复旧机器人结果，也不增加独立后处理步骤；下一阶段应以播放器中的肩、肘和手部整体观感为依据，在连续性上限与快速动作跟随之间做少量权重调优。
