# Holosoma human-to-robot retargeting

The production surface has three retargeting commands.
`examples/robot_retarget.py` retargets one selected motion without
augmentation. `examples/parallel_robot_retarget.py` retargets one selected
object-interaction or climbing motion and its augmentation variants. Despite
its historical filename, the second command never walks a dataset.
`paired_retargeting/robot_refine.py` consumes two synchronized robot-only results
and performs cross-actor joint refinement while preserving both nominal
trajectories. Its full input, contact, collision, output, and visualization
contract is documented in [paired_retargeting/README.md](paired_retargeting/README.md).

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
`--save-dir` to override the result root, `--output-name NAME.npz` to replace
the final artifact filename, and `--overwrite` to replace an existing result
at the same path. `--output-name` accepts one filename rather than a path and
preserves the `<robot>/<task>/<dataset>` directory layout. Live Viser visualization is enabled by
default. `--retargeter.debug` adds mapped human/robot keypoints, hand
skeletons, and object-point diagnostics and waits for Enter after solving.
Use `--retargeter.no-visualize` for headless runs. An existing result does not
rerun the solver, so use `--overwrite` to watch the live solve again or inspect
the saved result with the player. Dataset traversal, ablation, search, and
result rebuilding are not public command options.

Contact-aware planar foot constraints are enabled by default. They distinguish
flat, heel, toe, pivot, slide, and swing phases; pivoting locks only its support
point and an intentional slide follows the source path mapped through each
frame's target transform. Elevated support requires a matching upward-facing
surface from the climbing scene, so a stationary airborne foot is not locked. Pass
`--foot-sticking True` to state that choice explicitly or
`--foot-sticking False` to disable all planar contact constraints for the
selected single or augmented run. This switch does not disable ground
non-penetration or joint limits:

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

## Per-robot retargeting profiles

Every concrete robot now has one authoritative JSON file in
`examples/robot_profiles`: `g1.json`, `e1_23dof.json`, `e1_24dof.json`, or
`e2.json`. The compatibility name `--robot e1` resolves to the E1 23DOF
profile. A profile contains the robot height, degree-of-freedom count, URDF,
root and Interaction Mesh weights, solver and foot-contact defaults, the
direct shoulder-direction settings, all dataset orientation weights, and the
natural-pose reference and weights. The former `nature_weights` and
`orientation_weights` profile directories are no longer used.

The selected profile is always loaded. An option supplied explicitly on the
command line takes precedence over its JSON value; an omitted option inherits
the profile value. For example, this command changes only the root-position
weight while retaining every other E1 23DOF profile default:

```bash
python examples/robot_retarget.py \
  --task robot_only \
  --robot e1_23dof \
  --dataset noetix_mocap \
  --motion sequence/name \
  --root-position-weight 1.0
```

Use `--robot-profile FILE_OR_DIR` to test another complete table without
replacing the bundled profile. Robot dimensions and assets can also be
overridden directly through `--robot-height`, `--robot-dof`, and
`--robot-urdf-file`. Boolean profile values such as foot sticking accept
explicit overrides such as `--foot-sticking False`.

## Direct upper-arm direction tracking

When `shoulder_direction.enable` is true in the robot profile, or
`--shoulder-direction-tracking True` is supplied, the solver replaces the
generic Arm, ForeArm, and Hand SO(3) costs with a direct torso-local upper-arm
unit-direction residual. There is no candidate construction, candidate
scoring, pitch/roll/yaw branch selection, offline reference path, or branch
reference in this mode. Redundant shoulder motion is resolved by the existing
Interaction Mesh, joint limits, natural-pose cost when enabled, and the
frame-to-frame smoothness objective. The direction weight can be overridden
with `--shoulder-direction-weight`.

G1 Hand orientation is retained only through its three wrist variables, E1
Hand orientation is projected onto its single elbow-yaw axis, and E2 has no
wrist orientation task. Lower-body and torso entries from an enabled
orientation table continue to work. Saved results contain target and actual
upper-arm directions, direction errors, solved shoulder angles, and direction
Jacobian singular values. Render them with:

```bash
python -m holosoma_retargeting.visualization.shoulder_direction result.npz
```

## Orientation, root stability, and natural pose

The bundled profiles preserve the previous default behavior: orientation,
direct shoulder-direction tracking, natural-pose regularization, and both root
stability weights start disabled, while foot sticking remains enabled. Edit a
robot JSON once to change its persistent defaults, or use command-line
overrides for one run. `--orientation-tracking True` enables the selected
dataset table, while `--orientation-weights WEIGHT` overrides every mapped
link with one non-negative weight. `--natural-pose-tracking True` enables the
stored per-joint table, while `--nature-weights WEIGHT` applies one weight to
all actuated joints.

Root stability remains a pair of soft objectives. The position target follows
the source-root displacement after frame-zero alignment, while the orientation
target uses a direct source-root orientation when available and otherwise the
source torso frame. The first solved robot frame anchors the relative targets.
The Interaction Mesh and root values can be tuned together as follows:

```bash
python examples/robot_retarget.py \
  --task robot_only \
  --robot e1_23dof \
  --dataset noetix_mocap \
  --motion sequence/name \
  --orientation-tracking True \
  --shoulder-direction-tracking True \
  --root-position-weight 200 \
  --root-orientation-weight 50 \
  --interaction-mesh-weight 10 \
  --arm-interaction-mesh-weight-scale 0.5
```

The natural-pose objective remains
`sum_i w_i (q_i - q_i_natural)^2`. Active joints are initialized from their
fixed references before frame zero, and later frames continue from the
preceding solution.

## Outputs

Single-motion results:

```text
demo_results/<robot>/<task>/<dataset>/<motion>.npz
```

With `--output-name custom.npz`, `<motion>.npz` is replaced by `custom.npz`.

Augmented results:

```text
demo_results_parallel/<robot>/<task>/<dataset>/<motion>.npz
demo_results_parallel/<robot>/<task>/<dataset>/<motion>_<variant>.npz
```

For augmented families, a custom identity name such as `custom.npz` produces
`custom.npz`, `custom_trans_0.npz`, and the remaining canonical variant names.

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
