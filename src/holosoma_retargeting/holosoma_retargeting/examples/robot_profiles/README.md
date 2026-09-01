# 机器人重定向配置表使用说明

本目录为每个具体机器人维护一份权威 JSON 配置表。`g1.json`、`e1_23dof.json`、`e1_24dof.json` 和 `e2.json` 中的内容不是界面展示数据，而是重定向程序的运行配置。命令行中的 `--robot` 会选择对应配置表；兼容名称 `e1` 会选择 `e1_23dof.json`。

## 配置如何进入求解器

重定向入口首先在 `config_types/retargeting.py` 的 `internal_config_from_command()` 中调用 `config_types/robot_profiles.py` 的 `load_robot_profile()`。加载器会读取整份 JSON，严格检查字段、类型、数值范围、机器人名称、朝向映射和自然姿态关节数量，然后将数据转换为 `RobotConfig` 与 `RetargeterConfig`。`retargeting_pipeline.py` 再把这些配置传给 `InteractionMeshRetargeter`，求解器据此加载机器人模型、缩放人体数据、构造目标函数和启用约束。

因此，修改配置表会影响修改之后新启动的重定向任务。它不会改写已经生成的 NPZ 文件，也不会在正在运行的进程中热更新。若同名结果已经存在，需要更换输出名称或使用 `--overwrite` 才能重新计算并看到新参数的效果。

配置解析采用严格模式。顶层和各子表不能缺少字段，也不能加入当前 schema 不认识的字段。数值必须有限；权重和容差通常要求非负；`robot_dof` 必须为正整数；`robot_height`、`step_size` 和 `nominal_tracking_tau` 必须大于零。加载失败会直接停止任务，不会静默回退到代码默认值。

## 默认配置、定制配置与命令行覆盖

不传 `--robot-profile` 时，程序按照 `--robot` 从本目录选择 JSON。`--robot-profile` 可以指向一份 JSON 文件，也可以指向一个包含同名机器人 JSON 的目录。例如，`--robot e1_23dof --robot-profile /tmp/profiles` 会读取 `/tmp/profiles/e1_23dof.json`，而 `--robot-profile /tmp/my_e1.json` 会直接读取指定文件。配置文件中的 `robot` 仍必须与命令选择的具体机器人一致。

参数优先级是命令行显式值高于 JSON 值。命令行参数没有出现时，其内部值为 `None`，程序才继承 JSON。布尔覆盖项需要明确给出 `True` 或 `False`。例如，下面的命令继承 E1 表中的其他设置，但把根部位置权重改成 `1.0`，并显式开启肩部方向跟踪与关闭脚底粘连。

```bash
PYTHONPATH=src/holosoma_retargeting \
python -m holosoma_retargeting.examples.robot_retarget \
  --task robot_only \
  --robot e1_23dof \
  --dataset noetix_mocap \
  --motion 20260806_guangboticao/动作名称 \
  --root-position-weight 1.0 \
  --shoulder-direction-tracking True \
  --foot-sticking False \
  --overwrite \
  --retargeter.no-visualize
```

`robot_dof`、`robot_height`、`robot_urdf_file`、`retargeting` 中的全部字段、`shoulder_direction` 中的全部字段、`orientation.enable`、`orientation.preview` 和 `natural_pose.enable` 都有对应的命令行覆盖项。`--orientation-weights` 会把当前数据集所有已映射朝向统一覆盖为一个权重；`--nature-weights` 会把所有机器人关节统一覆盖为一个自然姿态权重。`planar_foot_contact` 的检测细节以及逐关节、逐数据集权重目前没有逐项命令行覆盖，应通过默认 JSON 或传给 `--robot-profile` 的实验 JSON 调整。

## 顶层字段

**`schema_version`** 表示配置结构版本。当前只接受整数 `1`。它只用于兼容性校验，不直接进入优化目标；改变为其他值会让加载器拒绝该配置。

**`robot`** 是配置表所属的具体机器人名称，用于防止把 G1、E1 或 E2 的表加载给错误模型。它主要承担身份校验；实际机器人选择仍由命令行 `--robot` 发起。`e1` 是公共兼容别名，因此它读取的配置表中仍应填写 `e1_23dof`。

**`robot_dof`** 是机器人的执行器自由度数量。它用于构造初始 `qpos`、确定机器人关节数据切片、校验输出宽度，并要求 `natural_pose.joint_names` 恰好覆盖同样数量的关节。修改它会实际改变数据布局；若与 URDF/MuJoCo 模型不一致，程序通常会在模型或关节元数据校验阶段报错，因此它不是用于随意裁剪自由度的调参项。命令行覆盖项是 `--robot-dof`。

**`robot_height`** 是目标机器人的参考身高，单位为米。人体位置会按照 `robot_height / source_human_height` 进行统一缩放，所以改变它会实际改变人体目标点、步幅、手脚位置和 Interaction Mesh 的空间尺度。命令行覆盖项是 `--robot-height`。

**`robot_urdf_file`** 指定机器人 URDF。相对路径以 `holosoma_retargeting` 包目录为基准解析，绝对路径也可以使用。管线会加载该 URDF，并使用对应的 MuJoCo XML 构造运动学、碰撞几何、关节限位和雅可比；因此修改它会实际更换求解模型。新模型必须与机器人映射、DoF 和关节名称一致。命令行覆盖项是 `--robot-urdf-file`。

## `retargeting` 求解默认值

**`dynamic_ground_window`** 控制 `robot_only` 任务的局部地面采样窗口是否随机器人根部移动。开启后，机器人移动时 Interaction Mesh 使用的地面区域会跟随根部，避免机器人走出固定局部地面点集。它会影响地面相关的 Interaction Mesh 几何，但不改变 URDF 中的真实地面碰撞约束。命令行覆盖项是 `--dynamic-ground-window True/False`。

**`interaction_mesh_weight`** 是 Interaction Mesh 拉普拉斯形变残差的全局权重。它直接进入每一帧的 SQP 目标函数。数值越大，求解越倾向保持人体、机器人锚点与环境之间的相对几何关系；数值越小，根部、朝向、肩部方向、自然姿态和平滑项等其他目标更容易主导结果。设为 `0` 会移除该形变残差对目标函数的贡献。命令行覆盖项是 `--interaction-mesh-weight`。

**`arm_interaction_mesh_weight_scale`** 是上肢锚点的 Interaction Mesh 权重倍率。求解器按锚点名称识别 shoulder、arm、elbow、wrist 和 hand，上肢的有效权重为 `interaction_mesh_weight * arm_interaction_mesh_weight_scale`，其他锚点仍使用全局权重。降低它可以减弱手臂局部误差向躯干和根部传播的程度，设为 `0` 会去掉上肢锚点的网格代价，但肩部方向或朝向目标若已开启仍可控制手臂。命令行覆盖项是 `--arm-interaction-mesh-weight-scale`。

**`root_position_weight`** 是机器人浮动根部跟踪已对齐人体根部世界位置的平方误差权重，直接进入 SQP 目标。值为 `0` 时不添加位置项；值大于 `0` 时会在首帧完成对齐标定，后续帧约束根部平移。增大它通常能减少肩部误差推动整个躯干平移的现象，但过大时可能牺牲脚底约束、整体形变匹配或可行性。命令行覆盖项是 `--root-position-weight`。

**`root_orientation_weight`** 是机器人浮动根部跟踪已对齐人体根部世界姿态的 SO(3) 误差权重，直接进入 SQP 目标。值为 `0` 时不添加姿态项；值大于 `0` 时会抑制躯干意外偏航、俯仰和横滚。它和位置权重彼此独立，只要任意一个大于零，根部稳定功能就会初始化。命令行覆盖项是 `--root-orientation-weight`。

**`q_a_init_idx`** 决定 SQP 从哪一个 `qpos` 地址开始优化，首个地址按 `7 + q_a_init_idx` 计算，末端为 `7 + robot_dof`。浮动根部占用 `qpos[0:7]`，因此 `-7` 表示根部与全部执行器都可优化，`0` 表示只优化执行器，正值会继续排除前面的执行器。该参数会从根本上改变可用自由度；根部稳定目标要实际移动根部通常应包含浮动根部变量。脚底粘连的当前实现还要求 `q_a_init_idx < 12`。命令行覆盖项是 `--q-a-init-idx`。

**`activate_joint_limits`** 控制是否把 URDF/MuJoCo 的关节上下限和机器人手工限位作为每帧硬约束加入 SQP。关闭后可能得到超出真实机构范围的关节角，一般只适合诊断。命令行覆盖项是 `--activate-joint-limits True/False`。

**`activate_obj_non_penetration`** 控制动态交互物体的非穿透硬约束。关闭后，机器人与桌、箱子等动态对象之间可以穿透；机器人与地面的非穿透约束仍保持开启。`robot_only` 的对象就是地面，因此该开关通常不会关闭地面约束。命令行覆盖项是 `--activate-obj-non-penetration True/False`。

**`foot_sticking`** 控制是否把自动检测的脚底接触计划作为平面硬约束加入 SQP。开启后，静止支撑、脚跟或脚尖支点、旋转支点和有意滑动会采用对应的平面约束。关闭后仍会计算接触计划和诊断信息，但这些计划不会约束最优解。当前求解器还要求 `q_a_init_idx < 12` 才实际应用脚底粘连。命令行覆盖项是 `--foot-sticking True/False`。

**`penetration_tolerance`** 是环境非穿透约束允许的穿透容差，单位为米。约束形式允许有符号距离最低达到 `-penetration_tolerance`，所以增大该值会放宽地面及已启用对象的穿透限制，减小则更严格。命令行覆盖项是 `--penetration-tolerance`。

**`foot_sticking_tolerance`** 是普通脚底固定约束允许的 XY 平面误差，单位为米。数值越小，支撑脚锁定越严格，也越可能与其他目标冲突；有意滑动使用独立的 `planar_foot_contact.slide_tracking_tolerance`。命令行覆盖项是 `--foot-sticking-tolerance`。

**`step_size`** 是每帧 SQP 更新量的二阶锥信赖域半径，限制单次关节配置增量的整体范数。较小值通常更平滑保守，但可能跟不上快速动作；较大值允许更快变化，也可能增加线性化误差或帧间跳动。它必须大于零。命令行覆盖项是 `--step-size`。

**`w_nominal_tracking_init`** 是已有名义姿态跟踪项在序列开头的权重。该项仅作用于机器人配置中预先指定的 `NOMINAL_TRACKING_INDICES`，与下面的逐关节 `natural_pose` 正则不是同一个功能。设为 `0` 会关闭这项代价。命令行覆盖项是 `--w-nominal-tracking-init`。

**`nominal_tracking_tau`** 是名义姿态权重按帧衰减的时间常数，当前公式为 `w_nominal_tracking_init * exp(-frame_index / nominal_tracking_tau)`。值越大，名义姿态影响保持得越久；值越小，影响衰减越快。它以帧索引为自变量，并不是秒。命令行覆盖项是 `--nominal-tracking-tau`。

## `planar_foot_contact` 接触检测与约束参数

这一节首先用于从人体脚跟和脚尖轨迹生成逐帧接触模式，然后由 `foot_sticking` 决定是否把计划应用到求解器。修改检测阈值会改变哪些帧被判为 swing、flat、heel、toe、pivot 或 slide。即使 `foot_sticking` 为 `false`，检测仍会运行并可进入结果诊断，但不会因此改变机器人轨迹。

**`smoothing_window_seconds`** 是计算脚部速度前，对人体脚跟和脚尖位置做居中平滑的时间窗口，单位为秒。程序会按源数据帧率换算为奇数帧窗口。更大可抑制噪声，但会模糊短暂接触和快速变化。

**`static_speed`** 是进入静止接触状态的脚部切向速度阈值，单位为米每秒。脚跟或脚尖的 XY 速度低于该值时才可能被判定为稳定接触。

**`release_speed`** 是已经进入静止状态后退出该状态的切向速度阈值，单位为米每秒。它与 `static_speed` 构成滞回区间以减少接触状态抖动，并且必须大于或等于 `static_speed`。

**`normal_speed`** 是接触候选允许的最大绝对竖直速度，单位为米每秒。超过该阈值说明脚部仍在明显抬起或落下，不应被判为稳定支撑。

**`pivot_angular_speed`** 是识别脚底旋转支点所需的最小平面角速度，单位为弧度每秒。超过阈值并满足支撑几何条件时，接触可从平脚分类为脚跟、脚尖或足底内部支点旋转。

**`slide_speed`** 是识别有意滑动所需的最小整体平面速度，单位为米每秒。脚部接近支撑面、竖直速度稳定、脚跟与脚尖运动一致且整体速度超过该值时，才会分类为 slide。该值必须严格大于 `static_speed`。

**`ground_clearance`** 是脚部被视为接近地面或高处支撑面的最大高度距离，单位为米。增大会更容易把悬空但接近地面的脚判为接触，减小则要求脚更贴近支撑面。

**`minimum_phase_seconds`** 是普通接触模式保留所需的最短持续时间，单位为秒。短于该时长的接触片段会被过滤或合并，用于消除逐帧闪烁。

**`elevated_support_seconds`** 是在攀爬等高处支撑面上确认稳定接触所需的持续时间，单位为秒。它会与普通最短阶段时长共同换算成帧数，并采用其中更严格的要求。

**`heading_tolerance`** 是平脚或支点状态下保持足部平面朝向的线性化容差。数值越小，足部 yaw 保持越严格；数值过小可能在快速转向或接触分类不准时降低可行性。

**`slide_tracking_tolerance`** 是 slide 模式跟踪目标世界 XY 轨迹时允许的误差，单位为米。它只用于有意滑动，普通固定接触使用 `foot_sticking_tolerance`。

## `shoulder_direction` 上臂方向跟踪

**`enable`** 控制是否启用直接上臂方向目标。开启后，求解器根据人体肩到肘方向构造左右上臂单位向量，并与机器人肩到肘方向的雅可比残差直接比较；它不会构造或评分 pitch、roll、yaw 候选分支。与此同时，普通朝向目标中的手臂、前臂和手部条目会从通用 SO(3) 代价中剔除，再按机器人腕部结构选择性地重新加入可实现的腕部目标，避免同一条上肢链受到互相冲突的完整姿态约束。命令行覆盖项是 `--shoulder-direction-tracking True/False`。

**`direction_weight`** 是左右上臂单位方向残差的 SQP 权重，直接影响最优解。增大它会更严格地复现人体上臂朝向，降低它会让 Interaction Mesh、根部稳定、自然姿态和平滑项拥有更大余量，设为 `0` 时虽然功能仍初始化，但方向残差不再贡献代价。命令行覆盖项是 `--shoulder-direction-weight`。

**`wrist_axis_weight_scale`** 是肩部方向模式下保留下来的腕部朝向权重倍率。它只有在 `shoulder_direction.enable` 已开启并且当前数据集的朝向跟踪或朝向权重提供了对应手部目标时才影响求解。G1 可在三个腕关节上保留手部朝向，E1 将手部朝向投影到单个 elbow-yaw 轴，E2 当前没有可保留的腕部自由度，因此该值对 E2 不改变轨迹。命令行覆盖项是 `--shoulder-wrist-axis-weight-scale`。

## `orientation` 人体与机器人链节朝向跟踪

**`enable`** 控制是否把当前数据集对应的非零 SO(3) 朝向权重加入求解目标。开启后，程序用数据格式提供的人体关节朝向、T-pose 标定和机器人链节映射构造旋转误差；关闭后，JSON 中的朝向权重仍会被读取和校验，但不会进入目标。命令行覆盖项是 `--orientation-tracking True/False`。显式传入 `--orientation-weights` 时，统一权重覆盖优先于该开关，因此正数可以直接启用朝向目标，`0` 可以直接关闭。

**`preview`** 只控制是否计算并保存独立标定的人体目标坐标系和机器人链节坐标系，便于可视化检查朝向映射。它本身不增加 SO(3) 目标，也不会改变机器人重定向轨迹，但会增加少量计算和输出诊断数据。命令行覆盖项是 `--orientation-preview True/False`。

**`datasets`** 保存各输入数据集自己的朝向权重表。当前 schema 要求完整包含 `amass`、`fbx_mocap`、`gvhmr`、`lafan`、`noetix_csv_climb`、`noetix_mocap` 和 `OMOMO_new`。程序每次加载配置时都会解析并校验这些表，但一次重定向只会选择与 `--dataset` 相同的那一张进入目标。旧的 `climbing` 数据集预设没有直接朝向配置，若强制为它开启朝向跟踪会报错。

**`datasets.<dataset>.default_weight`** 是该数据集所有已映射人体朝向的默认非负权重。值越大，相关机器人链节越严格跟随人体朝向；值为 `0` 的映射只可用于预览或诊断，不贡献求解代价。

**`datasets.<dataset>.overrides`** 用于覆盖个别映射的默认权重。键既可以写人体源关节名，也可以写该映射对应的机器人链节名；加载器会解析成唯一的人体关节权重。未知名称、同一个映射被人体名和机器人名重复配置、负数或非有限权重都会报错。这里的逐项值在 `orientation.enable` 为 `true` 时会实际改变对应链节的朝向目标强度。

## `natural_pose` 固定自然姿态正则

**`enable`** 控制是否启用逐关节自然姿态正则。开启后，所有权重大于零的关节会在序列初始化时设置到其参考角，并在每一帧通过固定参考代价抑制运动学冗余和零空间漂移；关闭后，整张自然姿态表仍会被读取并做结构校验，但不会传入求解器。命令行覆盖项是 `--natural-pose-tracking True/False`。显式传入 `--nature-weights` 时，统一权重覆盖优先于该开关。

**`joint_names`** 必须按配置维护机器人的全部执行器关节名，每个关节恰好出现一次，数量必须等于 `robot_dof`。这组名称用于展开默认参考角与默认权重，并在自然姿态启用后与 MuJoCo 模型的实际铰链关节核对。它不是关节求解顺序的重映射表，不应通过调换顺序来改变输出布局。

**`reference_default`** 是所有未在 `reference_overrides` 中单独指定关节的默认自然姿态角，单位为弧度。参考角可以为正或负，但在对应权重大于零时必须位于模型关节限位内。

**`reference_overrides`** 按关节名覆盖默认自然姿态角，单位为弧度。它只接受 `joint_names` 中已有的名称。某关节只有在其最终自然姿态权重大于零时才会使用这里的参考角并影响初始化与优化。

**`weight_default`** 是所有未在 `weight_overrides` 中单独指定关节的默认自然姿态代价权重。值必须非负；值为 `0` 的关节保留在完整配置表中，但不会被加入自然姿态目标。

**`weight_overrides`** 按关节名覆盖默认自然姿态权重。增大某个关节的值会更强地把它拉向对应参考角，常用于稳定肩部、腰部或其他冗余关节；过大会与动作跟踪目标竞争。它只接受 `joint_names` 中已有的名称。

## 哪些修改不会直接改变轨迹

只修改 `schema_version` 或 `robot` 不属于调参：正确值只用于校验，错误值会阻止运行。只开启 `orientation.preview` 会改变保存的诊断信息而不会改变 SQP 目标。修改某个已关闭功能内部的权重通常也不会改变轨迹，例如 `orientation.enable` 为 `false` 且没有 `--orientation-weights` 时，朝向数据集权重不参与求解；`natural_pose.enable` 为 `false` 且没有 `--nature-weights` 时，自然姿态参考和权重不参与求解；`foot_sticking` 为 `false` 时，接触检测参数只改变接触计划诊断；`shoulder_direction.enable` 为 `false` 时，两个肩部权重不参与求解。

相反，机器人模型与缩放字段、所有已经启用的目标权重、活动自由度范围、硬约束开关和容差都会实际进入重定向流程。比较参数效果时，应保持输入动作、输出版本和其他配置不变，并为不同实验使用不同 `--output-name`，以免误看旧结果。
