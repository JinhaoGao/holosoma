# Adding a human motion format

[简体中文](ADD_MOTION_FORMAT_README_zh.md) | English

Human formats have two explicit registrations:

1. the skeleton and robot mapping in `config_types/data_type.py`;
2. the file contract and loader in `data_utils/motion_data.py`.

There is intentionally no generic NPZ fallback. A new format must declare and
validate its own data contract.

## 1. Define the canonical representation

All adapters return `HumanMotion`:

```python
HumanMotion(
    joints=world_joints,                       # float array (T, J, 3), right-handed Z-up
    fps=fps,                                   # positive source FPS
    human_height=height_m,                     # positive height in metres
    source_path=input_path,
    object_poses_wxyz_xyz=object_poses,        # optional (T, 7)
    orientation_joint_names=orientation_names, # optional directly observed subset
    orientation_quaternions_wxyz=orientations, # optional (T, O, 4)
    orientation_source=orientation_provenance, # optional approved provenance
    joint_parent_indices=parent_indices,
)
```

Joint positions must be finite and their order must exactly match the
registered joint-name list. Object poses use `[qw, qx, qy, qz, x, y, z]`.
Do not synthesize source-human link orientations from positions. Save only the
directly observed joint subset and declare its provenance.

For a converted NPZ format, prefer these fields:

```text
global_joint_positions  (T, J, 3)
joint_names             (J,)
height                  scalar metres
fps                     scalar
root_quaternions_wxyz   (T, 4), optional
source_format           scalar string
coordinate_system       scalar string
```

If the NPZ stores direct source-human orientations, it must instead provide the
complete explicit group:

```text
orientation_joint_names             (O,), unique subset of joint_names
orientation_quaternions_wxyz         (T, O, 4), finite unit quaternions
orientation_source                   scalar approved provenance string
quaternion_convention                scalar "wxyz"
coordinate_system                    scalar registered canonical system
orientation_coordinate_system       scalar, required when the format registers it
```

Missing or ambiguous convention metadata is an error; the loader must not
guess or silently convert it.

## 2. Register skeleton semantics

In `config_types/data_type.py`:

```python
MYFORMAT_DEMO_JOINTS = [
    "Pelvis",
    # Names in the exact order of joints[:, :, :].
]

DEMO_JOINTS_REGISTRY["myformat"] = MYFORMAT_DEMO_JOINTS
TOE_NAMES_BY_FORMAT["myformat"] = ["LeftToe", "RightToe"]

JOINTS_MAPPINGS[("myformat", "g1")] = {
    "Pelvis": "pelvis_contour_link",
    # Human joint -> robot link.
}
```

Add one joint mapping per supported robot. Do not register a robot until every
mapped link has been checked against that robot model.

If an old public name must remain accepted, add only a boundary alias:

```python
DATA_FORMAT_ALIASES["old_name"] = "myformat"
```

Saved output should always use the canonical name.

## 3. Register the file and task contract

Add a `MotionFormatSpec` in `data_utils/motion_data.py`:

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

`orientation_mode` controls initial root orientation when the data does not
store root quaternions:

- `smpl`: estimate orientation from hips and shoulders;
- `bvh`: use the BVH-style anatomical estimator;
- `mocap`: use foot direction.

Only list `object_interaction` when the adapter supplies object poses. Set
`nested_files=True` only when one sequence is represented by a directory
containing exactly one motion file.

## 4. Implement and register the loader

The loader reads only its declared file type, performs format-specific
coordinate conversion, and delegates common validation to `_validate_motion`:

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

Avoid `allow_pickle=True` for numeric motion files. Do not catch missing fields
and reinterpret the file as another format.

## 5. Preserve the shared result contract

All single-action, augmentation, and ablation entry points must build jobs
through the shared retargeting pipeline and write the same strict schema-v2
artifact. A result is not only qpos. It preserves source identity and
configuration hashes, FPS and cost, the complete source-human skeleton and
parent tree, the complete robot-link skeleton and orientations, mapped
keypoints, directly observed source orientations, target/robot orientation
diagnostics, object poses and asset closure, object keypoints, Interaction Mesh,
foot-sticking state, and per-frame constraint audits.

This is deliberate: visualization decides which layers to show after
retargeting. An entry point must not discard metadata merely because its
default viewer does not display that layer.

Use `examples/robot_retarget.py` for one action and
`examples/parallel_robot_retarget.py` for a dataset batch. In the batch entry
point, `--data-dir` is canonical; the inherited `--data-path` remains a
compatibility alias, and passing two different values is rejected. Batch task
names are discovered from source files, so an inherited non-default
`--task-name` is rejected. Preserve nested robot, motion, task, and retargeter
overrides when binding top-level robot/format selectors.

Canonical augmentation families place `identity.npz` and every generated
variant in one sequence directory. Ablations use the same artifact schema under
an experiment namespace. Evaluation accepts strict single-run
`identity.npz` artifacts only and computes from saved trajectories, FPS,
contact state, cost, and validated external assets; it does not reload raw
motion as an alternate source of truth.

## 6. Test the adapter and full pipeline

At minimum, add tests for:

- canonical name and any aliases;
- valid file loading;
- joint count and joint-order rejection;
- invalid FPS/height and NaN rejection;
- coordinate and quaternion conventions;
- single-file resolution and batch discovery;
- supported and unsupported task combinations;
- at least one short end-to-end retargeting result.

The end-to-end output should have finite qpos, the correct FPS and canonical
`source_data_format`, mapped skeleton arrays, and—when enabled—Interaction Mesh
arrays:

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

For a strict family, omitting `--variants` discovers all existing saved
variants, including climbing `z_scale_*` variants. Explicit comparison paths
default to their saved semantic variant labels; labels must be unique, and
duplicate physical files (including symlink aliases) are rejected. The old
single-viewer `--qpos-npz` option and old Python viewer aliases remain
deprecated compatibility boundaries; new code should use `--input-path`,
`MultiViserConfig`, and `make_multi_result_player`.

Finally, add the format to the support table and data-preparation section in
`README.md`.
