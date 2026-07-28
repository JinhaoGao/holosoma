# Noetix E1 全身朝向权重搜索报告

本报告研究 `breaking+hippop.bvh_Skeleton1` 的 2660 帧完整动作。所有实验使用
Noetix T-pose 到 E1 几何 T-pose 的 15-link 固定坐标系标定，并在同一输入、初始化、
位置 Interaction Mesh、SQP 参数和约束设置下，仅改变逐 link 朝向权重。

## 无量纲平衡目标

零朝向权重的 15-link T-pose baseline 为归一化分母。设位置均值、位置 P95、SO(3)
测地朝向均值、朝向 P95 相对 baseline 的比值分别为
`r_p_mean`、`r_p_p95`、`r_R_mean` 和 `r_R_p95`，二阶关节差分比值为 `r_smooth`。
主评分为：

```text
J_core =
    0.35 r_p_mean + 0.15 r_p_p95
  + 0.35 r_R_mean + 0.15 r_R_p95
```

稳定性惩罚为：

```text
J_stability =
    0.10 max(0, r_smooth - 1)
  + 0.25 max_iteration_fraction
  + 1[存在约束释放]
```

最终最小化 `J = J_core + J_stability`。这一形式避免直接把米与弧度相加，并使位置
和朝向各占 50%，每一类内部均值占 70%、P95 尾部占 30%。同时保留五维
`position mean / position P95 / orientation mean / orientation P95 / smoothness`
Pareto 前沿，防止标量评分掩盖退化。

## 搜索过程

粗搜索在帧起点 `0/900/1800/2450` 的四个 120 帧窗口上测试 17 组等权和单组权重。
细搜索测试 21 组根、髋、膝、脚踝、脚尖、肩、前臂、手的非等权组合。随后在完整
2660 帧上并行复核 baseline、`0.1/0.2/0.3/0.5`、三组非等权候选，并在
`0.05–0.15` 和 `0.075–0.1` 区间进行两轮完整序列局部搜索。所有并行任务使用
独立输出目录，最终精确比较 `0.075/0.08/0.0825/0.085/0.0875/0.09/0.1`。

分组搜索发现根和髋单独增权的边际综合收益较低，但完整动作中的非等权降权方案仍
未超过等权方案，说明根/髋朝向与全身链条存在有益耦合。最终保留简单且可复现的
15-link 等权配置。

## 完整序列最优结果

选定配置为全部 15 个 link 的权重均为 `0.085`。

| 指标 | baseline | weight 0.085 | 相对变化 |
|---|---:|---:|---:|
| 位置均值 | 54.6533 mm | 52.6670 mm | -3.63% |
| 位置 P95 | 140.4703 mm | 143.2114 mm | +1.95% |
| 朝向均值 | 35.3719° | 19.7272° | -44.23% |
| 朝向 P95 | 124.2724° | 52.4370° | -57.80% |
| 关节二阶差分均值 | 0.139049 rad | 0.155982 rad | +12.18% |
| SQP 平均迭代 | 6.26 | 约 6.4 | 小幅增加 |
| 约束释放 | 0 | 0 | 无退化 |
| 综合分 | 1.000094 | 0.760970 | -23.91% |

`0.0875` 和 `0.09` 的综合分分别为 `0.761122` 和 `0.761125`；`0.075` 为
`0.761773`，`0.1` 为 `0.763483`。当位置/朝向各占 50% 时 `0.085` 最低；
若朝向占 60%，最优邻域移向 `0.09`，若位置占 60%，最优邻域移向 `0.075`。
因此 `0.085` 是等权平衡目标下的中心解，而不是依赖单一粗网格点的选择。

## 逐 link 朝向均值

| 人体关节 / 机器人 link | baseline | weight 0.085 | 变化 |
|---|---:|---:|---:|
| Hips / base_link | 21.147° | 19.638° | -7.14% |
| LeftUpLeg | 22.607° | 19.385° | -14.25% |
| RightUpLeg | 25.805° | 21.832° | -15.40% |
| LeftLeg | 20.445° | 15.717° | -23.12% |
| RightLeg | 20.407° | 14.146° | -30.68% |
| LeftFoot | 27.453° | 16.277° | -40.71% |
| RightFoot | 31.374° | 16.995° | -45.83% |
| LeftToeBase | 27.579° | 16.522° | -40.09% |
| RightToeBase | 36.042° | 20.840° | -42.18% |
| LeftArm | 55.354° | 19.318° | -65.10% |
| RightArm | 28.554° | 14.320° | -49.85% |
| LeftForeArm | 56.534° | 22.127° | -60.86% |
| RightForeArm | 52.792° | 32.261° | -38.89% |
| LeftHand | 60.691° | 25.675° | -57.69% |
| RightHand | 43.795° | 20.856° | -52.38% |

## 复现与结果位置

正式 profile 名为 `balanced_optimal`。使用
`run_orientation_ablation.py --variants baseline balanced_optimal --weight-scales 1`
可重新生成 baseline 与最优结果。完整搜索器是 `search_orientation_weights.py`，
候选表分别为 `orientation_weight_candidates_coarse.json`、
`orientation_weight_candidates_fine.json`、
`orientation_weight_candidates_finalists.json`、
`orientation_weight_candidates_refine.json` 和
`orientation_weight_candidates_exact.json`。

本次完整序列最优结果位于
`demo_results_orientation/orientation_weight_search/full_sequence/runs/uniform_0p085/window_000000/`，
归一化评分、排名、Pareto 前沿和逐 link 数据位于同级 `search_summary.json`。

该参数是针对当前 Noetix `breaking+hippop` 动作、E1 模型和现有 SQP/Interaction
Mesh 配置得到的最优平衡。迁移到不同机器人、坐标系标定或明显不同的动作类别时，
应复用同一搜索器重新验证，而不应假设 `0.085` 是跨任务常数。
