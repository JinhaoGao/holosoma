Robot-to-robot 重定向使用 `examples/robot2robot_retarget.py`。它读取一个机器人 CSV，通过源模型 FK 恢复语义关键点位置和朝向，再进入 human2robot 共用的预处理、Interaction Mesh 形变能、朝向目标、接触约束和 SQP 求解流程。FK 只还原源运动，目标关节角由优化器求解。输出同时包含目标 NPZ 和同名 CSV。

从 `src/holosoma_retargeting/holosoma_retargeting` 目录运行 `python examples/robot2robot_retarget.py --source-robot g1 --robot e2 --input-path /path/to/motion.csv --fps 50 --orientation-tracking True`。无界面运行增加 `--retargeter.no-visualize`。也可以从仓库根目录在配置好包路径的环境中运行 `python -m holosoma_retargeting.examples.robot2robot_retarget` 并传入相同参数。G1→E2 是首个测试用例；更换源、目标名称即可使用其他配置组合。维护的配置包含 G1、E1 23DoF、E1 24DoF 和 E2，`e1` 兼容名称解析为 `e1_23dof`。

CSV 与现有 `data_utils/npz_to_csv.py` 的输出一致：无表头，每行一帧，依次为根位置 xyz（米）、根四元数 xyzw、按源 URDF 文件中非固定关节顺序排列的关节值。旋转关节单位为弧度，移动关节单位为米，世界坐标采用右手 Z-up。必须提供正的有限 `--fps`，因为 CSV 不保存时间信息。程序不猜测外部列布局，也不根据自由度数量猜测关节名称。所有关节按名称映射到 MuJoCo 模型，根四元数转为内部 wxyz 并归一化；空文件、缺列、非有限数值和零四元数会在求解前报错。输入帧数和 FPS 保留。

每个机器人在 `examples/robot_profiles/<robot>.json` 的可选 `robot_motion` 部分描述自身，不维护源目标成对的配置。`position_links` 将语义关键点映射到模型 body 原点，`parents` 定义这些关键点的有根树，`root_joint` 指向实际浮动根 body，`toe_names` 按左、右顺序标记脚底前端点。`orientation_links` 为同一语义集合中的点指定朝向 body，因此位置锚点和朝向 body 可以不同。源配置必须提供目标映射所需的语义名称；无法对应时直接报错，不静默丢弃目标。可通过模型中的固定辅助 body 定义额外锚点。

`reference_base_quaternion_wxyz` 与 `reference_joint_positions` 定义双方具有相同语义姿态、相同世界朝向的参考姿态；未列出的关节取零。公共求解器根据源、目标参考姿态 FK 计算固定 link 坐标系偏移，再将该偏移应用到逐帧源朝向。不能把任意两个机器人的机械零位直接当成匹配参考姿态。内置配置使用已有的机器人参考姿态；朝向保存采用现有紧凑诊断格式。`robot_motion.orientation_weights` 为目标语义朝向声明权重，通过原有 `--orientation-tracking True` 开启，`--orientation-weights` 可以统一覆盖。默认开关继续继承目标机器人表。

`robot_motion.robot_config` 指定目标的脚底 link、手动关节边界、关节代价和 nominal tracking 索引。`foot_contact_links` 为每侧声明脚跟两点、前脚掌两点及脚尖。源 FK 使用真实脚跟和脚尖坐标进入公共接触检测，避免套用人体踝关节估计脚跟的偏移；检测结果进入同一套目标足底约束。尺度和地面对齐继续使用公共预处理，比例为目标身高除以源身高，不添加机器人专属肢段重建。`--foot-sticking True|False`、形变能、根部稳定、自然姿态及求解参数沿用现有接口。

新机器人通过 `--source-robot-profile FILE_OR_DIR` 或目标的 `--robot-profile FILE_OR_DIR` 接入，名称使用字符串，不需要修改脚本中的枚举或注册某个机器人对。提供完整机器人表、URDF 以及同目录同文件干名的 MuJoCo XML；两个模型必须描述同一组标量关节，MuJoCo 模型必须有一个位于 qpos 开头的浮动根。自定义模型路径建议写成绝对路径，相对路径按 retargeting 包目录解释。纯 robot2robot 自定义机器人可以把人体数据集的 `orientation.datasets` 设为空对象，朝向权重由 `robot_motion` 提供。当前公共求解器面向具有左右脚底定义的浮动基机器人，并不自动推断陌生机械结构的语义映射。

专用肩部方向项继续沿用现有实现。G1、E1、E2 使用已有的手臂链识别；其他模型可在 `robot_motion.robot_config.shoulder_direction_chains` 中按左、右顺序提供两份链定义，字段为 `shoulder_joint_names`（三个关节）、`torso_basis_link_name`、`shoulder_anchor_link_name`、`elbow_link_name`、`hand_link_name`、`elbow_joint_name` 和 `wrist_joint_names`（零至三个关节）。源语义名称需满足现有 Hips/Arm/ForeArm/Hand 或 Pelvis/Shoulder/Elbow/Wrist 骨架约定。通用位置和完整朝向重定向不要求开启该专用项。

默认结果为 `demo_results/<target>/robot_only/robot2robot/<source>/<motion>.npz` 和同目录的 `<motion>.csv`。`--save-dir` 修改结果根目录，`--output-name custom.npz` 同时确定两份输出的文件干名，`--overwrite` 重新求解并重新导出。已有兼容 NPZ 时沿用现有恢复机制，缺少 CSV 则直接补导出；配置不同会要求显式覆盖。NPZ 保留目标 qpos、FPS、实际映射点和朝向、Interaction Mesh、接触计划与诊断，并记录源机器人、源 URDF、输入关节顺序和目标 URDF。CSV 仅从最终 NPZ 导出，使用目标 URDF 关节顺序，因此可以作为下一次 robot2robot 的输入。

使用 `python viser_player.py --input-path <result.npz> --show-human-skeleton True --show-robot-skeleton True` 查看结果。参数名兼容现有播放器，机器人来源的参考图层显示为 Source robot skeleton；目标模型优先使用 NPZ 中记录的 URDF，支持未注册的自定义机器人名称。这里显示的是经过公共预处理的源参考骨架。

验证命令为 `PYTHONPATH=src/holosoma_retargeting:src/holosoma:src/holosoma_inference python -m pytest --confcutdir=tests tests/test_robot2robot_retargeting.py`，从仓库根目录、安装了 retargeting 依赖的环境运行。专用测试覆盖内置模型的 CSV/FK 往返、已知根旋转、非法输入、G1→E2 与反向及 E1→E2 的真实优化、自定义源目标名称、参考姿态校准、双格式一致性和恢复行为。完整重定向回归使用 `tests/test_*.py`。`--confcutdir=tests` 隔离仓库根目录面向训练环境的 conftest，不改变重定向测试内容。测试动作是确定性的合成短序列，用户的实际 CSV 仍需进行动作质量验收。
