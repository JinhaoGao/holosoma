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
existing result at the same path. Live solver debug, live visualization,
dataset traversal, ablation, search, and result rebuilding are not public
command options.

## Supported matrix

| Dataset preset | Internal format | Direct source orientation | `robot_only` |
| --- | --- | --- | --- |
| `climbing` | `mocap` | No; the legacy source is position-only NPY | G1, E1, E2 |
| `gvhmr` | `gvhmr` | Yes; direct SMPL-X rotation FK | G1, E1, E2 |
| `lafan` | `lafan` | Yes; BVH rotation-channel FK | G1, E1, E2 |
| `noetix_csv_climb` | `mocap` | Yes; converted bone-rotation FK | G1, E1, E2 |
| `noetix_mocap` | `noetix_mocap` | Yes; BVH rotation-channel FK | G1, E1, E2 |
| `OMOMO_new` | `omomo` | Yes; InterMimic global orientation tensor | G1, E1, E2 |

`object_interaction` accepts only `OMOMO_new` on G1. `climbing` accepts only
`climbing` and `noetix_csv_climb` on G1. The augmentation command accepts only
those two interaction tasks.

## Optional orientation loss

Orientation loss is disabled by default. `--orientation` enables equal weights
for the 15 calibrated mappings:

```bash
python examples/robot_retarget.py \
  --task robot_only \
  --robot g1 \
  --dataset gvhmr \
  --motion tennis \
  --orientation
```

For per-keypoint or per-link weights, pass a JSON file:

```json
{
  "enabled": true,
  "weights": {
    "Pelvis": 0.2,
    "L_Shoulder": 0.1,
    "right_shoulder_yaw_link": 0.1
  }
}
```

```bash
python examples/robot_retarget.py \
  --task robot_only \
  --robot g1 \
  --dataset gvhmr \
  --motion tennis \
  --orientation-config ./orientation.json
```

Keys may name either a mapped human keypoint or its robot link. Unspecified
mappings have zero weight. G1, E1, and E2 each use their own robot FK T-pose;
each direct-orientation format supplies its source T-pose frame. A fixed frame
offset aligns those frames before the loss is evaluated, following the same
principle as GMR. No orientation is inferred from positions or bone vectors.

## Outputs

Single-motion results:

```text
demo_results/<robot>/<task>/<dataset>/<motion>/identity.npz
```

Augmented results:

```text
demo_results_parallel/<robot>/<task>/<dataset>/<motion>/<variant>.npz
```

The artifact retains float64 qpos, solver metadata, mapped human and robot
points, available hand keypoints, mapped robot-link positions and wxyz
orientations, available direct source orientations, effective human/robot/
terrain/object point clouds, object data, and the source/target Interaction
Mesh. Unused full-body and full-link trajectories are pruned.

## Visualization

```bash
python viser_player.py \
  --input-path demo_results/g1/robot_only/gvhmr/tennis/identity.npz \
  --show-point-clouds \
  --show-source-orientation-axes \
  --show-robot-orientation-axes
```

The single point-cloud switch displays saved human, robot, terrain, demo-object,
and target-object points with distinct colors. Interaction Mesh, skeleton, key
point, and orientation-axis layers remain independently controllable in Viser.
Use `multi_viser_player.py --family <motion-directory>` to inspect one saved
augmentation family on a synchronized timeline.

See `README_zh.md` for the full Chinese guide,
`ADD_MOTION_FORMAT_README.md` for adapter integration, and
`docs/retargeting-production-contract.md` at the repository root for release
gates.
