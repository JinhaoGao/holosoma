# 新增人体动作格式

新增格式必须接入统一 motion adapter、生产数据 preset、机器人映射、紧凑结果制品和可视化加载器，不能新增第三个重定向脚本。

## 适配器契约

在 `data_utils/motion_data.py` 注册格式和文件发现规则，并让 loader 返回 `HumanMotion`。最少需要逐帧全局关键点位置、稳定的规范关节名、FPS、人体高度、源路径和父节点索引。坐标必须在进入求解器前转换到仓库的 Z-up 米制场景坐标。

若源数据提供直接关节朝向，额外返回 `orientation_joint_names`、`orientation_quaternions_wxyz` 和受批准的 `orientation_source`。四元数必须有限、单位化、使用 wxyz，并明确坐标系。不得从关键点位置或骨向量推断“直接朝向”。仅位置格式应完整省略朝向组。

在 `config_types/data_type.py` 注册规范关节列表、父树、G1/E1/E2 的位置映射，以及手部命名规则。只要生产 preset 允许 `robot_only`，三个机器人都必须有可加载的 URDF/MJCF、正确 DOF 顺序和位置映射测试。

## 朝向适配

有直接朝向时，为每个支持的机器人注册独立的人体关节到机器人 link 映射。映射应覆盖 15 个语义位置：root、左右 hip、knee、ankle、toe、shoulder、elbow 和 hand。肩膀等位置映射与朝向映射可以指向不同 link；朝向必须使用具有独立可解释 frame 的 link。

同时注册源数据 T-pose 中每个人体 frame 的 wxyz 朝向、机器人 T-pose base 朝向和机器人 T-pose 关节角。运行时固定偏置由人体 T-pose frame 与机器人 FK T-pose frame 对齐得到。新增格式必须通过 T-pose identity、已知轴旋转、G1/E1/E2 link 存在性和默认关闭朝向损失的测试。

## 生产 preset

在 `config_types/retargeting.py` 的 `DatasetName`、`DATASET_DATA_FORMATS`、`DATASET_DEFAULT_PATHS` 和 `SUPPORTED_TASK_DATASETS` 中显式加入 preset。生产命令始终要求一个 `motion`，不得加入全目录遍历。

普通动作使用：

```bash
python examples/robot_retarget.py \
  --task robot_only \
  --robot e2 \
  --dataset <preset> \
  --motion <one-motion>
```

只有确实提供 object-interaction 或 climbing 资产且通过 G1 验收时，才允许增强入口：

```bash
python examples/parallel_robot_retarget.py \
  --task <object_interaction-or-climbing> \
  --robot g1 \
  --dataset <preset> \
  --motion <one-motion>
```

## 结果与测试

writer 会自动将结果压缩为实际映射关键点、可用手关键点、实际映射 link 的位姿、有效点云和 Interaction Mesh。适配器必须提供足够信息生成这些字段，但不应自己建立新的结果 schema 或目录。

至少增加 loader 的成功与错误测试、单文件发现测试、坐标/单位测试、关节顺序与父树测试、G1/E1/E2 位置映射测试、直接朝向 provenance 测试、T-pose frame offset 测试、紧凑制品 round-trip 测试以及可视化 metadata 测试。完成后运行仓库根下的 `tests/test_*.py`、Ruff check、Ruff format check 和 `git diff --check`。
