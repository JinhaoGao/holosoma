# 动作重定向阶段报告

本报告冻结 2026-07-30 的阶段状态。剩余人体数据、增强矩阵和消融矩阵均已停止遍历；核验时没有发现仍在运行的 `robot_retarget`、`parallel_robot_retarget` 或 `rebuild_demo_results` 进程。本阶段只做现状盘点、只读归档规划、接口文档整理和代码验证，没有重新启动任何求解任务。

## 阶段结论

重定向接口已经收敛为三类公开入口。单动作由 `holosoma_retargeting.examples.robot_retarget` 负责，多源普通批量和批量增强由 `holosoma_retargeting.examples.parallel_robot_retarget` 负责，受控消融由 `examples` 下相应的 ablation、comparison 和 search 入口负责。`rebuild_demo_results` 只编排完整 G1/E1 数据矩阵、验证 staging 并事务性发布，不是第四套求解器。三类入口共享同一个 `RetargetJob`、加载与预处理注册表、SQP 生命周期、schema-v2 writer、原子发布和严格续跑契约。

结果语义同样已经统一。identity artifact 的 `run_kind` 为 `single`，只有改变动作或物体几何的非 identity variant 才是 `augmentation`，保持源动作不变而改变实验参数的结果是 `ablation`。批量执行只是一种调度方式，不会形成新的 artifact 类型。

schema-v2 默认保存完整预处理人体骨架及父树、完整机器人 link 位置与 `wxyz` 四元数及父树、actuated joint 顺序、映射关键点、Interaction Mesh、物体位姿与资产闭包、最终 `float64` qpos、六类真实几何残差、三类最终约束模式、SQP 诊断和回退审计。可视化阶段通过统一 loader 选择显示层，不再依赖重新读取原始输入来补造缺失元数据。

## 直接人体朝向契约

人体朝向只在源表示直接包含旋转并且格式适配器能够无估计地转换为世界坐标四元数时保存。机器人完整 link 朝向始终来自最终 qpos 的 MuJoCo FK，与人体源朝向是否存在无关。

| 标准格式 | 直接人体朝向来源 | 保存策略 |
| --- | --- | --- |
| `omomo` | InterMimic tensor 的全局朝向区间存在时使用 `intermimic_global_orientation_tensor` | 有直接 tensor 才保存，否则标记 `absent` |
| `lafan` | 原始 BVH rotation channels 经 FK 得到 `bvh_rotation_channels_fk` | 保存直接旋转通道对应的朝向 |
| `amass` | SMPL-X local rotations 经 FK 得到 `direct_local_rotation_fk` | 保存 |
| `gvhmr` | SMPL-X local rotations 经 FK 得到 `direct_local_rotation_fk` | 保存 |
| `noetix_mocap` | Noetix BVH rotation channels 经 FK 得到 `bvh_rotation_channels_fk` | 只保存源文件真实提供的具名子集 |
| `mocap` 的 Noetix CSV | bone rotation channels 经 FK 得到 `bone_rotation_channels_fk` | 保存 |
| `mocap` 的通用 climbing | 只有关节位置 | 写入 `orientation_source="absent"` 并省略整组人体朝向字段 |

代表性验收结果再次确认了这一边界。当前候选中具有完整报告或严格验收依据的 artifact 均保存了 Interaction Mesh、人体父树、机器人完整 link 位置和机器人完整 link 四元数；通用 climbing 的两个样本没有人体朝向，这是源数据缺失导致的预期结果，不是保存失败。

## 为什么全量重定向很慢

瓶颈是逐帧非线性优化，不是元数据保存。一个动作的帧必须按时间顺序求解，以前一帧 qpos 作为下一帧 SQP 的初值和时序参考，并保持 foot sticking、碰撞和接触状态的语义；这不等于复用 Clarabel/CVXPY problem 的 solver warm start。在当前求解契约下只能直接在动作之间并行，不能把同一动作的帧任意拆散。每一个 constraint mode 默认至少执行 4 次、最多执行 50 次 SQP；活动足部可依次尝试 normal、relaxed 和 released，因此一帧可到 150 次，若同时允许 object release，理论上可到 300 次，之后还可能触发整序列重试。每次 SQP 都会重新建立 CVXPY 变量、约束与 `Problem`，调用 Clarabel，并重新计算 MuJoCo forward、link/contact Jacobian、碰撞距离和局部 Laplacian 项。在这八个 15+225 点 robot-only 样本中，每次 SQP 还会重复构造 240×240 dense Laplacian 及其 Kronecker 数据；Delaunay、邻接和目标 Laplacian 则是每帧构造一次。真实几何验收固定先评估 alpha=0 和 alpha=1；提议不可行而 incumbent 可行时最多再执行 20 次二分，并可能增加一次最终验收，因此一次 SQP 最多评估 23 个真实几何候选。

八个真实 robot-only 验收样本共 7,124 帧、78,179 次 SQP，进程耗时合计 12,751.497 秒。614 帧的记录迭代数恰好为 50，占全部帧的 8.62% 并消耗 39.27% 的全部迭代；其中 608 帧的保存 stop reason 明确是 `max_iterations`，其余 6 帧可能恰好在第 50 次满足其他停止条件。这八个成功样本没有 constraint fallback、release 或整序列重跑，因此这些数字描述的是成功路径，不能当作所有困难任务的整帧硬上限。完整 artifact 合计只有约 31.22 MiB，独立严格加载、校验和哈希约 8.99 秒；这不是求解过程的分阶段计时，也没有单独测量压缩写入与逐帧元数据采集，但量级上强烈表明它们不是小时级耗时的主因。

| 样本格式 | 帧数 | G1 耗时 | G1 秒/帧 | E1 耗时 | E1 秒/帧 |
| --- | ---: | ---: | ---: | ---: | ---: |
| AMASS | 51 | 46.958 秒 | 0.921 | 31.206 秒 | 0.612 |
| GVHMR | 312 | 710.771 秒 | 2.278 | 254.774 秒 | 0.817 |
| LAFAN | 3,066 | 7,344.726 秒 | 2.396 | 3,917.036 秒 | 1.278 |
| Noetix BVH | 133 | 293.170 秒 | 2.204 | 152.856 秒 | 1.149 |

G1 的四个样本合计耗时约为 E1 的 1.93 倍。G1 记录的迭代总数约为 E1 的 1.161 倍，总墙钟时间除以记录迭代数后的摊销值约为 E1 的 1.66 倍；该摊销仍包含帧级预处理和 I/O，不能解释成已隔离测量的 Clarabel 单次求解成本。G1 模型的 link、几何体和优化变量更多，是与代码结构一致的可能机制，但本阶段没有做逐 phase profiler 来证明各因素占比。开到 16 个 worker 只会并行 16 个独立动作，每个进程内部仍是串行逐帧 SQP；多进程还可能争用 CPU cache 和内存带宽，但本阶段没有并发 scaling benchmark，不能据此断言 16 worker 的精确最优点。

暂停前的 `demo-identity-v2` 从 17:33 左右启动，到最后一个原子 artifact 在 19:05 左右落盘，16 个 worker 发布了 77 个动作，包含启动和中断边界的观察吞吐约为 50 个 artifact/小时。identity-only 完整计划是 18,594 个 G1/E1 source-job；如果不严谨地假定后续所有批次都保持这一首批 G1 OMOMO robot-only 吞吐，情景外推约为 15.4 天。它不是完成时间预测或下界，因为不同帧数、任务、机器人、重试与提前失败都会改变吞吐。可以确定的是，“多开终端”只能减少动作级排队，无法消除单动作内部的 SQP 成本。

## 当前结果盘点

正式目录 `demo_results/v1` 当前没有可发布 artifact，只保留一个失败的早期批次报告，因此不能把任何 staging 或 legacy 目录称为正式 release。各 staging 必须按 solver implementation fingerprint 分开，不能仅凭 schema 相同而混用。本次阶段提交的 Git index 中、由指纹契约覆盖的 solver implementation 文件计算结果为 `f179f555…`；它不是整个提交树的 Git hash。现场工作树还叠加了未纳入本次提交的 E2 并行改动，因此盘点时完整工作树指纹为 `bd9d42a7…`，现有 `demo_results*` 与 `/tmp/holosoma*` 中没有与后者严格匹配的 artifact。下表中的 `f179f555…` 候选在 implementation 层与本次提交快照一致，但这一个条件不足以证明整批可严格续跑或发布，仍须同时核对 runtime、标准化配置、source identity、完整计划和 batch reports；`a9b4647d…` 与 `e032e66f…` 则属于其他历史源码快照。

| staging | 完整 artifact | 中断临时文件 | implementation fingerprint | 说明 |
| --- | ---: | ---: | --- | --- |
| `demo-identity-v2` | 77 | 16 | `a9b4647d…` | 本次旧运行的停止边界；全部是 G1 OMOMO robot-only identity |
| `demo-identity-v1` | 52 | 16 | `f179f555…` | 较早 identity-only 候选 |
| `demo-rebuild-v4` | 78 | 24 | `f179f555…` | 与 `demo-identity-v1` 有 52 个逐字节重复结果 |
| `demo-rebuild-v1` | 4,837 | 0 | `e032e66f…` | 4,498 个 single 与 339 个 augmentation，属于旧实现 |

所有列出的中断临时 NPZ 都是 0 字节求解候选，不是可恢复 artifact，也没有计入完成数。顶层 aggregate reports 目前是 dry-run 计划，部分 batch report 仍是 `running` tombstone；这能证明运行被中断，却不能证明 staging 完整或允许 promotion。

工作区中另外存在大量历史布局。只读整理器在固定时间戳 `20260730T120000000000Z` 下成功规划了 9 次精确移动，共 9,165 个文件、10,687,179,410 字节；没有执行任何 rename。

| 旧布局 | 文件数 | 字节数 | 未来归档类别 |
| --- | ---: | ---: | --- |
| `demo_results/.bench_concurrency_20260729` | 106 | 69,872,096 | benchmark |
| `demo_results/.smoke_20260729` | 6 | 3,213,281 | smoke |
| `demo_results/e1` | 197 | 472,274,908 | legacy robot layout |
| `demo_results/g1` | 139 | 490,423,220 | legacy robot layout |
| `demo_results/t1` | 1 | 221,826 | legacy robot layout |
| `demo_results_parallel` | 7,742 | 7,613,290,226 | legacy parallel/augmentation |
| `demo_results_ablation` | 61 | 1,218,531,054 | legacy ablation |
| `demo_results_orientation` | 762 | 380,590,516 | legacy orientation experiments |
| `demo_results_foot_ablation` | 151 | 438,762,283 | 混合 `f179f555…`、`a9b4647d…` 与 legacy 的 foot ablation |

结果整理器要求先存在一份完整验证并成功 promotion 的 `demo_results/v1`，再通过 `--promoted-run-id` 把归档事务绑定到该 release 的报告和内容指纹。当前不满足这道门禁，因此本阶段保留所有物理数据，只统一分类、忽略规则和命令文档。强行移动会让“历史结果”和“当前正式结果”之间失去可审计边界。

`/tmp` 下还保留了代表性测试产物和报告。它们包括 8 个 G1/E1 robot-only 严格样本、4 个 climbing 严格样本、方向消融样本，以及 OMOMO frame-zero、增强和 E1 结构性失败审计。`f179f555… / d3fcc033…` 组合在 `/tmp` 中共有 59 个路径、47 个 inode，其中 12 个路径是 OMOMO formal-quality 的硬链接视图；四个 final 根合计 51 个路径、39 个物理 artifact，另外 8 个属于 quality A/B 配置实验。`/tmp` 同时还有 106 个中断留下的 0 字节 solve 临时文件；加上 staging 的 56 个临时文件，全盘共 162 个 0 字节候选。这些文件是验收证据、实验候选或明确无效的中断残留，不是正式数据集 release；在没有复制、资产闭包校验和持久 manifest 前不会从 `/tmp` 混入 `demo_results/v1`。

## 已验证范围

统一结果契约、严格 loader、单/多动作可视化、批量规划、恢复、promotion 与归档保护均已有自动化覆盖。本阶段在不运行真实求解器的前提下重新执行了两组相关回归，结果合计为 297 项通过、1 项跳过和 192 个参数化 subtests；Ruff 的 `F`/`E9` 检查和 `git diff --check` 也均通过。更广的非 E2E 测试此前达到 436 项通过、1 项跳过和 370 个 subtests；E2E 余项依赖外部 Isaac、CUDA 或系统环境，不能用来证明本地算法失败。

真实求解验收覆盖 G1/E1 上的 AMASS、GVHMR、LAFAN 与 Noetix BVH robot-only，8/8 成功并通过 schema-v2、源 SHA、配置/solver identity、direct orientation 字节一致性、完整骨架、Interaction Mesh 和统一可视化 loader 检查。代表性 G1/E1 climbing 也完成 4/4；通用 climbing 按契约省略人体朝向。对本次 Git index 所代表的 solver implementation 快照做隔离测试时还发现一个资产边界：五个 `e1_23dof_w_multi_boxes.xml` 场景存在于本机 demo dataset，但受顶层忽略规则保护而没有进入 Git 快照。隔离环境补入这些本地只读夹具后，generated-asset cache 测试为 6 项通过和 10 个 subtests 通过；没有这些外部夹具的新 checkout 不能复现 E1 climbing 资产测试或任务。这是待单独解决的数据打包与环境前置条件，不是本阶段接口或 solver 回归，也不应通过悄悄把被忽略的 demo dataset 混入提交来掩盖。OMOMO 增强已证明入口、保存和严格恢复可工作，但部分 G1 变体超过正式释放质量比例，E1 largebox 出现可复现的结构性可行性失败，因此增强结果只保留为实验审计，不作为正式发布完成项。

## 当前停点与后续条件

本阶段完成的是接口、数据契约、可视化契约、命令速查、结果分类和阶段提交，不是全量数据发布。在已中断的 `demo-identity-v2` 历史运行内，原计划 18,594 个 identity-only source-job，`a9b4647d…` 旧指纹下完成 77 个，按该运行自身口径尚有 18,517 个。若将来直接以包含未提交 E2 改动的 `bd9d42a7…` 现场工作树建立严格 run，起点仍是完整的 18,594 个；若明确回到本阶段提交的 `f179f555…` 快照，则可另行审计 `demo-identity-v1` 与 `demo-rebuild-v4` 是否同时满足其余严格 identity 后再决定能否续用，不能把两者的路径数直接相加。增强与消融全矩阵也保持延期。现有 staging 未经过完整 exact-set、batch-report、质量门禁与外部资产闭包验证，不能执行 promotion；没有 promotion，旧结果整理器也不会执行物理归档。

后续若恢复普通遍历，应使用新的、固定的 identity-only `run-id` 或明确选择一个仍与代码指纹相符的 staging，并在执行、`--validate-only` 和 `--promote` 三个阶段始终保留 `--no-include-augmentation-variants`。E1 object-interaction 和 climbing 可继续采用真实尝试后记录结构性缺口的策略，E1 robot-only 仍要求高覆盖，G1 维持完整严格目标。若后续优先优化速度，最有价值的代码方向是先加入分阶段 profiler，再把单帧固定邻接下的 Laplacian/Kronecker 移出 SQP 内循环，减少重复 MuJoCo forward/Jacobian，并为接触数量动态变化的 CVXPY 问题设计固定槽位或多模板参数化复用，而不是假设整个 `Problem` 可以直接缓存；这应作为独立性能任务，不应在没有质量等价验证时直接降低迭代上限或放宽约束。

统一后的可复制命令、恢复规则、正式矩阵验证和归档命令已经写入 `src/holosoma_retargeting/holosoma_retargeting/hint.md`。本阶段不会继续运行这些命令，等待下一次明确授权。
