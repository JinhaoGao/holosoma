M5 系列模型按型号归档到 `m5c/`、`m5d/` 和 `m5e/`，分别对应原始机械包 `__00-mia-v5-0C`、`__00-mia-v5-0D` 和 `__00-mia-v5-0E`。每个原始包完整保留 `urdf/`、`meshes/`、`config/`、`launch/`、`CMakeLists.txt` 和 `package.xml`，包名与文件内容保持原样，便于追溯来源和解析 `package://` 网格引用。

`m5c/m5c_17dof.urdf` 和 `m5c/m5c_17dof.xml` 是当前 M5C 重定向使用的派生模型，网格通过相对路径引用同目录下的原始机械包。生成脚本位于 `m5c/generate_retargeting_assets.py`，输入为 `m5c/__00-mia-v5-0C/urdf/update___00-mia-v5-0C.urdf`。在仓库根目录使用具备 MuJoCo、NumPy 和 trimesh 的 Python 环境运行 `python src/holosoma_retargeting/holosoma_retargeting/models/m5/m5c/generate_retargeting_assets.py` 即可重新生成。

M5C 的配置入口仍为 `examples/robot_profiles/m5c.json`，其中模型路径已更新为 `models/m5/m5c/`。详细说明见 [M5C 重定向文档](../../../../../docs/m5c-retargeting.md)。

`m5d/` 和 `m5e/` 同样包含各自的 `m5d_17dof.urdf/.xml`、`m5e_17dof.urdf/.xml` 和生成脚本。三个型号共用本目录的 `generate_retargeting_assets.py`，使用 `--robot m5c`、`--robot m5d` 或 `--robot m5e` 选择输入机械包。各型号目录中的脚本是固定型号入口，继续支持原有 M5C 生成命令。

三个型号均为 17 自由度，共用关节语义映射和两轴肩部处理；生成模型保留各自的连杆变换、质量、惯量与限位。M5D 的肩部更宽，M5D、M5E 的上下腿比例与 M5C 不同，M5E 的膝部横向偏移更大。FBX 位置目标按实际选定模型重建，结果分别记录 `m5c_bone_lengths_v2`、`m5d_bone_lengths_v2` 或 `m5e_bone_lengths_v2`。

公开重定向入口支持 `--robot m5d` 和 `--robot m5e`，默认配置位于 `examples/robot_profiles/m5d.json` 与 `m5e.json`，求解设置沿用当前 M5C 默认配置。批量交付可通过 `--robot-profile` 指定冻结的实验配置；具体接触开关与迭代次数以交付目录中的配置和运行脚本为准。
