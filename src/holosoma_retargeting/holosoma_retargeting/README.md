# Holosoma human-to-robot retargeting

The production surface has exactly two retargeting commands.
`examples/robot_retarget.py` retargets one selected motion without
augmentation. `examples/parallel_robot_retarget.py` retargets one selected
object-interaction or climbing motion and its augmentation variants. Despite
its historical filename, the second command never walks a dataset.

Run commands from this directory:

```bash
cd src/holosoma_retargeting/holosoma_retargeting
```

E1 robot-only runs accept `--robot e1_23dof` and `--robot e1_24dof`.
The historical `--robot e1` name remains an alias of `e1_23dof`. Both explicit
variants use the same human-data mappings; the 24DOF model adds the actuated
`waist_roll_joint` and loads its URDF and MuJoCo XML from `models/e1`.

## Quick start

Retarget one LAFAN motion to E2:

```bash
python examples/robot_retarget.py \
  --task robot_only \
  --robot e2 \
  --dataset lafan \
  --motion walk2_subject3
```

Retarget one OMOMO object-interaction motion and its G1 augmentations:

```bash
python examples/parallel_robot_retarget.py \
  --task object_interaction \
  --robot g1 \
  --dataset OMOMO_new \
  --motion sub3_largebox_003
```

Use `--data-path` only when the dataset is outside its repository default,
`--save-dir` to override the result root, and `--overwrite` to replace an
existing result at the same path. Live Viser visualization is enabled by
default. `--retargeter.debug` adds mapped human/robot keypoints, hand
skeletons, and object-point diagnostics and waits for Enter after solving.
Use `--retargeter.no-visualize` for headless runs. An existing result does not
rerun the solver, so use `--overwrite` to watch the live solve again or inspect
the saved result with the player. Dataset traversal, ablation, search, and
result rebuilding are not public command options.

Foot-sticking XY hard constraints are enabled by default. Pass
`--foot-sticking True` to state that choice explicitly or
`--foot-sticking False` to disable them completely for the selected single or
augmented run. This switch does not disable ground non-penetration or joint
limits:

```bash
python examples/robot_retarget.py \
  --task robot_only \
  --robot e2 \
  --dataset lafan \
  --motion walk2_subject3 \
  --foot-sticking False
```

## Supported matrix

| Dataset preset | Internal format | Direct source orientation | `robot_only` |
| --- | --- | --- | --- |
| `climbing` | `mocap` | No; the legacy source is position-only NPY | G1, E1 23/24DOF, E2 |
| `fbx_mocap` | `fbx_mocap` | Yes; direct FBX local-rotation FK | G1, E1 23/24DOF, E2 |
| `gvhmr` | `gvhmr` | Yes; direct SMPL-X rotation FK | G1, E1 23/24DOF, E2 |
| `lafan` | `lafan` | Yes; BVH rotation-channel FK | G1, E1 23/24DOF, E2 |
| `noetix_csv_climb` | `mocap` | Yes; converted bone-rotation FK | G1, E1 23/24DOF, E2 |
| `noetix_mocap` | `noetix_mocap` | Yes; BVH rotation-channel FK | G1, E1 23/24DOF, E2 |
| `OMOMO_new` | `omomo` | Yes; InterMimic global orientation tensor | G1, E1 23/24DOF, E2 |

`object_interaction` accepts only `OMOMO_new` on G1. `climbing` accepts only
`climbing` and `noetix_csv_climb` on G1. The augmentation command accepts only
those two interaction tasks.

## Optional orientation loss

Orientation loss is disabled by default. `--orientation_weights WEIGHT` assigns one
non-negative weight to all 15 calibrated mappings:

```bash
python examples/robot_retarget.py \
  --task robot_only \
  --robot g1 \
  --dataset gvhmr \
  --motion tennis \
  --orientation_weights 1
```

For per-keypoint or per-link weights, use the complete profile for the selected
robot:

```bash
python examples/robot_retarget.py \
  --task robot_only \
  --robot g1 \
  --dataset gvhmr \
  --motion tennis \
  --orientation_config examples/orientation_weights/g1.json
```

The orientation directory provides `g1.json`, `e1.json`, and `e2.json`. Both
explicit E1 variants use the E1 orientation table. Each profile contains
complete tables for `gvhmr`, `lafan`, `noetix_csv_climb`, `noetix_mocap`, and
`OMOMO_new`; the current `--dataset` selects the table. Every mapped key must
appear and may name either its human keypoint or robot link. Set a value to
zero to remove that keypoint's orientation loss. The profile robot must match
the command, and `--orientation_weights` and `--orientation_config` are mutually
exclusive. G1, E1, and E2 each use their own robot FK T-pose; each
direct-orientation format supplies its source T-pose frame. No orientation is
inferred from positions or bone vectors.

## Natural-posture regularization

Natural-posture regularization is also disabled by default. Use
`--nature_weights WEIGHT` to assign one non-negative weight to every actuated
joint. The fixed natural reference angles are read from the bundled
matching file in `examples/nature_weights`. The explicit E1 variants use
independent `e1_23dof.json` and `e1_24dof.json` tables; the latter defines all
24 joints, including `waist_roll_joint`:

```bash
python examples/robot_retarget.py \
  --task robot_only \
  --robot g1 \
  --dataset gvhmr \
  --motion tennis \
  --nature_weights 0.1
```

For independent joint weights, pass a robot-specific JSON file or a directory
containing the corresponding files through `--nature_config`. Each table
contains direct absolute weights and its natural reference angles in radians:

```bash
python examples/robot_retarget.py \
  --task robot_only \
  --robot g1 \
  --dataset gvhmr \
  --motion tennis \
  --nature_config examples/nature_weights/g1.json
```

Every SQP iteration adds the fixed joint-space cost
`sum_i w_i (q_i - q_i_natural)^2`. Active natural-pose joints are initialized from
the same references once before frame zero, while later frames continue from
the preceding solution. `--nature_weights` and `--nature_config` are mutually
exclusive. A zero uniform or table weight removes the corresponding cost.

## Outputs

Single-motion results:

```text
demo_results/<robot>/<task>/<dataset>/<motion>.npz
```

Augmented results:

```text
demo_results_parallel/<robot>/<task>/<dataset>/<motion>.npz
demo_results_parallel/<robot>/<task>/<dataset>/<motion>_<variant>.npz
```

Each NPZ retains float64 qpos, solver metadata, the human and robot points used
by position or orientation retargeting, compact skeleton connectivity, mapped
robot-link positions and wxyz orientations, direct source orientations used by
the solver, effective human/robot/terrain/object point clouds, object data, and
the source/target Interaction Mesh. These fields are saved by default without
schema, hash, manifest, or cross-field artifact validation. Unused full-body
and full-link trajectories are omitted.

## NPZ to CSV

The CSV contains numeric data only, without a header or index. Each row stores
root position `xyz`, root quaternion `xyzw`, and robot joint angles reordered
to match the URDF joint order.

Convert one motion and write `tennis.csv` beside `tennis.npz`:

```bash
python -m holosoma_retargeting.data_utils.npz_to_csv \
  demo_results/g1/robot_only/gvhmr/tennis.npz \
  models/g1/g1_29dof.urdf
```

Convert a directory recursively while preserving its relative structure:

```bash
python -m holosoma_retargeting.data_utils.batch_npz_to_csv \
  demo_results/g1/robot_only/gvhmr \
  models/g1/g1_29dof.urdf \
  demo_results_csv/g1/robot_only/gvhmr
```

## Visualization

```bash
python viser_player.py \
  --input-path demo_results/g1/robot_only/gvhmr/tennis.npz
```

The viewer loads every available saved layer by default. Interaction Mesh,
human and robot skeletons, point clouds, object keypoints, and orientation axes
are independently controlled from the Viser Layers tab; no layer list is
required on the command line. The Playback tab also provides a joint selector
with the complete angle trajectory and URDF lower/upper limits for each
actuated joint.
Use `multi_viser_player.py --family <motion.npz>` to inspect one saved
augmentation family on a synchronized timeline.

See `README_zh.md` for the full Chinese guide,
`ADD_MOTION_FORMAT_README.md` for adapter integration, and
`docs/retargeting-production-contract.md` at the repository root for release
gates.
