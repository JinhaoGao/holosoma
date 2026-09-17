M5C 重定向使用 models/m5/m5c/__00-mia-v5-0C/urdf/update___00-mia-v5-0C.urdf 作为机械模型来源。模型共有 17 个执行器关节：左右腿各 5 个、腰部 1 个、左右手臂各 3 个。浮动基座结果 qpos 宽度为 24，排列为根位置 XYZ、根四元数 WXYZ、17 个标量关节。使用更新版 URDF 中的原始关节轴、相对变换、质量惯量和限位。

models/m5/m5c/generate_retargeting_assets.py 生成同目录下的 m5c_17dof.urdf 和 m5c_17dof.xml。原始机械文件保持不变。足底标记依据实际踝部网格的底面范围，手端标记位于前臂网格的远端；标记为固定链接，不增加自由度。由原始几何计算出的站立底面到顶部约为 1.80 米，重定向参考身高为 1.8 米。

左右脚底各有五个补充接触球，球体下表面与标记处的测量底面重合。保留完整脚掌网格碰撞的同时，这些球让脚跟与前脚掌同时进入地面非穿透约束，避免长脚掌最近接触点切换后另一端穿地。接触球质量为零，不改变原始机械质量惯量。

examples/robot_profiles/m5c.json 参考 E2 求解参数，开启朝向、自然姿态、直接肩部方向和关节限位；当前默认关闭 foot_sticking，保留地面非穿透。自然姿态仅对左右肩俯仰施加权重 10、参考角 0 的正则。源到目标的朝向标定使用 M5C 独立 T-pose FK：肩侧摆参考角由上臂段 (0, ±0.09612, -0.22644) 求出，在原始限位内将上臂对齐到左右横向。FBX 的人体参考关节朝向从各源文件的 bind pose 读取。

M5C 的位置匹配排除髋部附近的三个源点。对 FBX，它们是 Spine、LeftUpLeg、RightUpLeg；对 Noetix BVH，它们是 Hips、LeftUpLeg、RightUpLeg。根部稳定项保持 E2 的独立位置与朝向权重 1。朝向映射保留根部和髋部，脚踝映射到实际存在的 ankle_pitch，肩部映射到 shoulder_roll，不创建模型中不存在的 ankle_roll 或 shoulder_yaw 自由度。

两轴肩部复用直接上臂方向目标和解析雅可比。结果中的 shoulder_direction_joint_names 和 shoulder_direction_actual_joint_positions 最后一个维度为 2，播放器及静态诊断支持该形状。奇异值仍保存 3 个空间维度，缺少的第三个值为 0，不代表第三个执行器。

公开单动作入口可使用 --robot m5c --task robot_only --dataset fbx_mocap --data-path 数据目录 --motion 动作名 --retargeter.no-visualize，其他参数默认来自 m5c.json。Viser 读取结果中的机器人身份和模型路径，也可以直接播放 M5C 结果。原始 FBX 转换器同时支持带角色前缀的多人骨架和本批次不带前缀的单人 Hips 骨架。

tests/test_m5c_retargeting.py 验证生成模型相对原始 URDF 的逐链节 FK 与限位、实际自由度、两轴肩方向的有限差分雅可比、T-pose 与已知轴旋转的朝向标定，以及单人 FBX 骨架识别。批次结果的逐文件校验由交付目录的 validate_results.py 完成。运动学重定向不包含动力学跟踪验证。

M5C 的 robot_only + fbx_mocap 路径默认使用 body_proportions.py 按机械骨段尺寸重建位置目标，版本为 m5c_bone_lengths_v2。大腿、小腿、上臂和前臂分别保持每帧源骨段方向，长度取实际模型对应轴心或末端标记的距离。模型大腿约 0.568918 米、小腿 0.5665 米、上臂 0.245996 米、前臂到手端标记 0.223028 米。髋部偏移由模型读取并随源基座朝向旋转，肩宽和躯干偏移由模型读取并随源躯干几何坐标架旋转。手指和头部未加入优化目标，其显示位置继承所属区域的比例。

根部水平位移按左右腿骨链总长度比缩放。根部高度依据重建后两脚所有足底标记的最低点逐帧校正，使用接触检测中的脚跟代理和脚尖共同估计源离地高度。开启 foot_sticking 时，支撑帧的目标足底最低点设为地面，腾空区间按相邻支撑帧插值扣除地面估计漂移；关闭时仅去除整段最低离地量，不使用支撑相位压低高度。两种模式均按腿长比例换算离地量。足底标记使用与求解器朝向目标相同的源 bind pose 和机器人 T-pose FK 标定，包含 T-pose 基座的参考旋转。接触模式和相位仍从原始米制动作提取，滑动位移参考改用重建后足底中心的相对位移，足底粘连仅在 foot_sticking 开启时执行，非穿透由独立开关控制。重建目标是几何参考，M5C 的关节轴、限位及接触约束仍可能要求最终姿态与其存在差异。

结果的 config_json 和 human_position_preprocessing 均记录重建版本；不再声明适用于全身的 human_position_scale，而是保存 root_translation_scale、各源骨段及机器人骨段长度、逐帧高度修正和目标足底点。可视化骨架同步使用重建后的比例。tests/test_body_proportions.py 验证原始 URDF 骨段长度、方向保持、髋宽和肩宽、腾空高度、世界偏航等变性和足底朝向标定；tests/test_foot_contact_plan.py 验证替换滑动参考不会改变原始接触相位。

关闭 foot_sticking 时，求解器不再准备或复用足底接触目标。结果字段 foot_sticking_enabled_for_saved_trajectory 和 body_proportion_contact_phase_height_enabled 为 false，body_proportion_height_policy 为 heel_toe_clearance_without_contact_phases。源接触分类仍保存为诊断信息。旧交付目录 configs/m5c.json 保留开启接触的配置，新对照目录 m5c_no_foot_contact 使用独立的关闭接触配置。
