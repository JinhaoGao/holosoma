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
    joints=world_joints,                 # 浮点数组 (T, J, 3)，右手系 Z-up
    fps=fps,                             # 正数源帧率
    human_height=height_m,               # 正数，单位为米
    source_path=input_path,
    object_poses_wxyz_xyz=object_poses,  # 可选，形状 (T, 7)
    root_quaternions_wxyz=root_quats,    # 可选，形状 (T, 4)
)
```

关节位置必须全部为有限值，关节顺序必须与注册的关节名称列表完全一致。物体位姿使用 `[qw, qx, qy, qz, x, y, z]` 顺序。

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

## 5. 测试适配器与完整流水线

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
  --qpos-npz /tmp/myformat-result/example.npz \
  --show-mapped-skeletons \
  --show-interaction-mesh
```

最后，在 `README_zh.md` 的支持表和数据准备章节中加入该格式，并同步更新英文 `README.md`。
