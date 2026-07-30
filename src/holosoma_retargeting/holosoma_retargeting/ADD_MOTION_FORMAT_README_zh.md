# 添加人体动作格式

简体中文 | [English](ADD_MOTION_FORMAT_README.md)

每种人体格式需要在两个位置显式注册：

1. 在 `config_types/data_type.py` 中注册骨架和机器人映射；
2. 在 `data_utils/motion_data.py` 中注册文件约定和加载器。

系统特意不提供通用 NPZ fallback。每种新格式都必须声明并校验自己的数据约定。

## 1. 定义标准表示

所有适配器均返回 `HumanMotion`：

```python
HumanMotion(
    joints=world_joints,                       # 浮点数组 (T, J, 3)，右手系 Z-up
    fps=fps,                                   # 正数源帧率
    human_height=height_m,                     # 正数，单位为米
    source_path=input_path,
    object_poses_wxyz_xyz=object_poses,        # 可选，形状 (T, 7)
    orientation_joint_names=orientation_names, # 可选，源数据直接观测到的子集
    orientation_quaternions_wxyz=orientations, # 可选，形状 (T, O, 4)
    orientation_source=orientation_provenance, # 可选，已批准的来源声明
    joint_parent_indices=parent_indices,
)
```

关节位置必须全部为有限值，关节顺序必须与注册的关节名称列表完全一致。物体位姿使用 `[qw, qx, qy, qz, x, y, z]` 顺序。
不得根据位置伪造源人体 link 朝向；只能保存源数据直接观测到的关节子集，并明确声明其来源。

对于转换后的 NPZ 格式，建议使用以下字段：

```text
global_joint_positions  (T, J, 3)
joint_names             (J,)
height                  标量，单位为米
fps                     标量
root_quaternions_wxyz   (T, 4)，可选
source_format           标量字符串
coordinate_system       标量字符串
```

如果 NPZ 保存了源人体的直接朝向，则必须完整提供以下字段组：

```text
orientation_joint_names             (O,)，joint_names 的无重复子集
orientation_quaternions_wxyz         (T, O, 4)，有限值单位四元数
orientation_source                   标量，已批准的来源字符串
quaternion_convention                标量 "wxyz"
coordinate_system                    标量，已注册的标准坐标系
orientation_coordinate_system       标量；格式声明需要时为必填
```

约定元数据缺失或含糊时必须报错，加载器不得猜测或静默转换。

## 2. 注册骨架语义

在 `config_types/data_type.py` 中添加：

```python
MYFORMAT_DEMO_JOINTS = [
    "Pelvis",
    # 名称顺序必须与 joints[:, :, :] 完全一致。
]

DEMO_JOINTS_REGISTRY["myformat"] = MYFORMAT_DEMO_JOINTS
TOE_NAMES_BY_FORMAT["myformat"] = ["LeftToe", "RightToe"]

JOINTS_MAPPINGS[("myformat", "g1")] = {
    "Pelvis": "pelvis_contour_link",
    # 人体关节 -> 机器人 link。
}
```

每个受支持机器人都需要单独添加一份关节映射。只有在所有映射 link 都已对照机器人模型检查后，才能注册该机器人。

如果需要继续接受旧的公开名称，只添加边界别名：

```python
DATA_FORMAT_ALIASES["old_name"] = "myformat"
```

保存结果时应始终使用标准名称。

## 3. 注册文件与任务约定

在 `data_utils/motion_data.py` 中添加 `MotionFormatSpec`：

```python
MOTION_FORMATS["myformat"] = MotionFormatSpec(
    name="myformat",
    suffixes=(".npz",),
    task_types=frozenset({"robot_only"}),
    root_joint="Pelvis",
    orientation_mode="smpl",
    default_fps=30.0,
    direct_orientation_npz=DirectOrientationNPZSpec(
        source_formats=frozenset({"myformat"}),
        coordinate_systems=frozenset({"right_handed_z_up"}),
        require_orientation_coordinate_system=True,
    ),
)
```

当数据没有保存根节点四元数时，`orientation_mode` 决定如何初始化根节点方向：

- `smpl`：根据髋部和肩部估计方向；
- `bvh`：使用 BVH 风格的人体结构估计器；
- `mocap`：使用脚部朝向。

只有当适配器能够提供物体位姿时，才能在任务类型中加入 `object_interaction`。只有当一个序列由“目录内恰好一个动作文件”表示时，才设置 `nested_files=True`。

## 4. 实现并注册加载器

加载器只能读取其声明的文件类型，完成该格式特有的坐标转换，再将通用校验交给 `_validate_motion`：

```python
def _load_myformat(
    path: Path,
    spec: MotionFormatSpec,
    human_height: float | None,
) -> HumanMotion:
    with np.load(path, allow_pickle=False) as data:
        _validate_joint_names(data, spec)
        height = human_height if human_height is not None else _read_scalar(data, "height")
        return _validate_motion(
            joints=data["global_joint_positions"],
            spec=spec,
            source_path=path,
            human_height=height,
            fps=_read_scalar(data, "fps"),
            root_quaternions=data.get("root_quaternions_wxyz"),
        )


_LOADERS["myformat"] = _load_myformat
```

数值动作文件应避免使用 `allow_pickle=True`。不要捕获字段缺失错误后再把文件按另一种格式解释。

## 5. 保持统一结果约定

单动作、增强实验和消融实验入口都必须通过共享重定向流水线创建任务，并写出同一种严格 schema-v2 artifact。一个结果不只是 qpos；它还应保留源数据身份与配置哈希、FPS 与 cost、完整人体骨架及父节点树、完整机器人 link 骨架及朝向、映射关键点、源数据直接观测到的朝向、目标/机器人朝向诊断、物体位姿与外部资产闭包、物体关键点、Interaction Mesh、脚部 sticking 状态，以及逐帧约束审计。

这样设计是为了让可视化阶段在重定向完成后自由选择显示层。某个入口不能因为默认查看器没有展示某一层，就在重定向时丢弃该元数据。

单动作使用 `examples/robot_retarget.py`，数据集批处理使用 `examples/parallel_robot_retarget.py`。批处理以 `--data-dir` 为标准参数；继承而来的 `--data-path` 只作为兼容别名，若同时传入两个不同值必须报错。批处理会从源文件发现 task name，因此继承而来的非默认 `--task-name` 必须报错。顶层 robot/format 选择器绑定到嵌套配置时，必须保留 robot、motion、task 与 retargeter 的其他覆盖项。

标准增强结果族把 `identity.npz` 和所有已生成 variant 放在同一个序列目录。消融实验在实验命名空间下使用同一 artifact schema。评估只接受严格 single-run 的 `identity.npz`，直接使用其中保存的轨迹、FPS、接触状态、cost 和经过校验的外部资产；不会再把原始动作数据作为另一份事实来源重新加载。

## 6. 测试适配器与完整流水线

至少应添加以下测试：

- 标准名称及所有别名；
- 有效文件加载；
- 拒绝错误的关节数量和关节顺序；
- 拒绝无效 FPS、身高和 NaN；
- 坐标系及四元数约定；
- 单文件解析与批量文件发现；
- 支持和不支持的任务组合；
- 至少一个短序列端到端重定向结果。

端到端输出应包含有限的 qpos、正确的 FPS、标准 `source_data_format` 和映射骨架数组；启用时还应包含 Interaction Mesh 数组：

```bash
python examples/robot_retarget.py \
  --task-type robot_only \
  --robot g1 \
  --data-format myformat \
  --data-path /path/to/converted \
  --task-name example \
  --save-dir /tmp/myformat-result \
  --retargeter.save-interaction-mesh

python viser_player.py \
  --input-path /tmp/myformat-result/canonical/g1/robot_only/myformat/converted/example/identity.npz \
  --show-human-skeleton \
  --show-robot-skeleton \
  --show-interaction-mesh

python multi_viser_player.py \
  --family /tmp/myformat-result/canonical/g1/robot_only/myformat/converted/example
```

对严格结果族省略 `--variants` 时，会自动发现所有实际存在的 variant，包括 climbing 的 `z_scale_*`。显式多结果比较默认采用 artifact 中保存的语义 variant 作为标签；标签必须唯一，并拒绝指向同一物理文件的重复路径，包括符号链接别名。旧单动作查看器的 `--qpos-npz` 以及旧 Python 查看器别名只保留为 deprecated 兼容边界；新代码应使用 `--input-path`、`MultiViserConfig` 和 `make_multi_result_player`。

最后，在 `README_zh.md` 的支持表和数据准备章节中加入该格式，并同步更新英文 `README.md`。
