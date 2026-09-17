M5C、M5D 和 M5E 均已接入公开重定向入口，分别使用 `--robot m5c`、`--robot m5d` 和 `--robot m5e`。模型按型号保存在 `src/holosoma_retargeting/holosoma_retargeting/models/m5/`，配置位于包内 `examples/robot_profiles/`。

三者均为 17 自由度，关节语义映射共用 M5C 的五轴腿、两轴肩与单轴腰结构。机器人身份和模型路径仍按具体型号保存；连杆变换、质量、惯量与限位来自各自的 `update_` 机械 URDF。共享生成器使用 `python src/holosoma_retargeting/holosoma_retargeting/models/m5/generate_retargeting_assets.py --robot m5d` 这样的命令生成对应派生 URDF 与 MuJoCo XML，每个型号目录中的同名脚本也可直接运行。

M5D 的大腿竖直长度从 M5C 的 566.5 mm 改为 540 mm，小腿从 566.5 mm 改为 593 mm，左右肩侧摆轴各向外移动 30 mm。M5E 使用与 M5D 相同的上下腿竖直长度，肩部位置保持 M5C 的设置，髋俯仰轴到膝轴的横向偏移每侧增加 40 mm。FBX 位置目标从选定模型读取骨段长度、髋部和肩部位置；这些参数不会从 M5C 硬编码复制到新型号。

原有 `rebuild_m5c_body` 接口继续有效，共享实现为 `rebuild_m5_body`，可通过 `robot_type` 指定型号。保存结果分别记录 `m5c_bone_lengths_v2`、`m5d_bone_lengths_v2` 或 `m5e_bone_lengths_v2`。高度重建、足底接触与朝向标定的详细说明见 [M5C 重定向说明](m5c-retargeting.md)。

默认 M5D、M5E 配置沿用仓库当前 M5C 配置，关闭足底粘连并保留地面非穿透。具体批次若参照开启接触的 M5C 实验，应使用独立冻结配置，通过 `--robot-profile` 传入；每帧迭代次数和接触策略应在交付记录中明确保存。

`tests/test_m5c_retargeting.py` 对三个型号分别验证逐连杆 FK、限位、质量惯量、肩部方向雅可比和 T-pose 标定；`tests/test_body_proportions.py` 对三个型号分别验证原始机械骨段尺寸、肩宽、方向保持、根部比例、离地高度与足底朝向。
