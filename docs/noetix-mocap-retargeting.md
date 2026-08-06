# Noetix 数据重定向

Noetix 在当前生产接口中分为两个 preset。`noetix_mocap` 是已经转换为标准 LAFAN 风格骨架的 BVH/NPZ 动作，适合 `robot_only`。`noetix_csv_climb` 是 CSV 重建得到的攀爬动作和场景，既可以在 `robot_only` 中使用，也可以作为 G1 `climbing` 任务使用。

公开重定向命令只有 `examples/robot_retarget.py` 与 `examples/parallel_robot_retarget.py`。前者处理一个动作，后者只处理一个 G1 object-interaction/climbing 动作及其增强族。任何命令都不会遍历整个 Noetix 数据目录。

## Noetix mocap

对 E1 做位置重定向：

```bash
cd src/holosoma_retargeting/holosoma_retargeting
python examples/robot_retarget.py \
  --task robot_only \
  --robot e1 \
  --dataset noetix_mocap \
  --motion run_to_the_right
```

对同一动作启用统一的 `1.0` 朝向权重：

```bash
python examples/robot_retarget.py \
  --task robot_only \
  --robot e1 \
  --dataset noetix_mocap \
  --motion run_to_the_right \
  --orientation_weights 1
```

Noetix mocap 的人体关节朝向来自 BVH rotation channels 的 FK，保存为具名 wxyz 全局四元数。G1、E1、E2 都使用 15 个语义映射：root，左右 hip、knee、ankle、toe、shoulder、elbow、hand。T-pose 标定会将人体关节 frame 对齐到各机器人独立的 link frame。

## Noetix CSV climbing

标准转换后的动作 NPZ 同时保存位置、bone rotation FK 朝向和场景重建身份。对其中一个动作执行 G1 climbing 增强：

```bash
python examples/parallel_robot_retarget.py \
  --task climbing \
  --robot g1 \
  --dataset noetix_csv_climb \
  --motion actor_170/yuezhanggaotai-dongzuo2-man-box-170
```

需要为一个动作自定义朝向权重时，编辑 G1 的完整示例配置中
`noetix_csv_climb` 对应的权重表：

```bash
python examples/parallel_robot_retarget.py \
  --task climbing \
  --robot g1 \
  --dataset noetix_csv_climb \
  --motion actor_170/yuezhanggaotai-dongzuo2-man-box-170 \
  --orientation_config examples/orientation_weights/g1.json
```

表中的 15 个映射必须完整保留，将某个关键点或对应 link 的权重改为
`0` 即可取消该项朝向损失。

仓库旧 `climbing` preset 中的 NPY 只有人体位置，不带直接朝向；它可以完成位置与 Interaction Mesh 重定向，但不能开启朝向损失。`noetix_csv_climb` 的标准 NPZ 带直接朝向，因此能够开启。

## 结果与查看

普通 Noetix mocap 结果示例：

```text
demo_results/e1/robot_only/noetix_mocap/run_to_the_right.npz
```

Noetix CSV climbing 增强族示例：

```text
demo_results_parallel/g1/climbing/noetix_csv_climb/actor_170/yuezhanggaotai-dongzuo2-man-box-170.npz
```

结果保留实际映射的人体与机器人关键点、可用手关键点、实际映射 link 的朝向、人体/机器人/地形/object 点云、source/target Interaction Mesh 和攀爬场景资产信息。查看时可以同时打开点云、Interaction Mesh 和朝向轴：

```bash
python viser_player.py \
  --input-path demo_results/e1/robot_only/noetix_mocap/run_to_the_right.npz \
  --show-point-clouds \
  --show-interaction-mesh \
  --show-source-orientation-axes \
  --show-robot-orientation-axes
```

旧的朝向消融、权重搜索、整目录批处理和专用结果树已经移除。权重调整通过同一个生产命令的 JSON 配置逐动作完成。
