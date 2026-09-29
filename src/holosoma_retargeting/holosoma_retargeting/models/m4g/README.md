MIA v4.2-G 使用本次随附的 `__00-mia-v4_2-G/urdf/update___00-mia-v4_2-G.urdf` 生成独立的 `m4g_17dof.urdf` 和 `m4g_17dof.xml`。原始包完整保留。生成脚本 `generate_retargeting_assets.py` 复用机械 URDF 转换器，保持质量、惯量、安装变换、关节轴与关节限位，足底参考点来自本型号脚网格，手部参考点来自手网格末端区域。`landmarks.json` 记录实际参考位置。

公开入口支持 `--robot m4g`，默认配置为 `examples/robot_profiles/m4g.json`。17 自由度语义映射复用 M5 家族，FBX 身体比例按本型号实际模型重建，结果标识为 `m4g_bone_lengths_v2`。右肩水平 T 姿态略超出限位，因此参考角使用 -1.17 弧度；左肩按自身肘部偏移计算。

在仓库根目录设置 `PYTHONPATH=src/holosoma_retargeting`，使用 `robot_retargeter` Python 运行本目录的生成脚本即可重新生成模型。模型和重定向结果仅用于运动学，不代表动力学或实机可执行性验证。
