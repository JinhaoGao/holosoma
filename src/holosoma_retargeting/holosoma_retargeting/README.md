# Holosoma Motion Retargeting

[简体中文](README_zh.md) | English

This repository provides tools for retargeting human motion data to humanoid
robots. The original project supports OMOMO, LAFAN, AMASS SMPL-X, generic mocap,
robot-only motion, object interaction, and climbing. This branch preserves those
workflows and adds explicit adapters for Noetix-mocap and GVHMR, together with a
unified result and visualization interface.

**Data requirements:** the retargeting pipeline consumes world-space human joint
positions with shape `(T, J, 3)`, where `T` is the number of frames and `J` is
the number of joints. Formats that support orientation tracking may additionally
provide world-space `wxyz` quaternions with shape `(T, J, 4)`. The default
position-only solver remains unchanged when orientations are absent. Each format
declares its joint order, robot mapping, file layout, coordinate conversion,
height, FPS, and supported task types. See
[ADD_MOTION_FORMAT_README.md](ADD_MOTION_FORMAT_README.md) when adding another
format.

Run the commands below from:

```bash
cd src/holosoma_retargeting/holosoma_retargeting
```

## Supported Human Motion Data and Tasks

| `--data-format` | Source/input | Retargeting input | Tasks | Robots |
| --- | --- | --- | --- | --- |
| `omomo` | InterMimic-processed OMOMO | `.pt`, 52 SMPL-H joints | `robot_only`, `object_interaction` | G1, T1, E1 |
| `lafan` | LAFAN BVH | `.npy`, 22 LAFAN joints | `robot_only` | G1, T1, E1 |
| `amass` | AMASS SMPL-X | `.npz`, 22 SMPL-X joints | `robot_only` | G1, E1 |
| `mocap` | nested world-joint mocap | `.npy`, 53 joints | `robot_only`, `climbing` | G1, T1, E1 mappings |
| `noetix_mocap` | company-collected Noetix BVH | `.npz`, normalized 22-joint Noetix schema | `robot_only` | G1, E1 |
| `gvhmr` | GVHMR `hmr4d_results.pt` | `.npz`, 22 SMPL-X joints | `robot_only` | G1, E1 |

The canonical format names are shown above. The command line remains compatible
with the legacy names `smplh` (`omomo`), `smplx` (`amass`), and
`noetix_lafan`/`noetix-mocap` (`noetix_mocap`).

> **LAFAN and Noetix-mocap are independent datasets.** LAFAN is the public
> Ubisoft dataset and follows the official `BVH → .npy` workflow.
> Noetix-mocap contains company-collected motion-capture data, accepts several
> Noetix BVH export variants, and follows a separate `BVH → .npz` workflow.
> The Noetix normalized skeleton reuses a compatible 22-joint naming topology
> for retargeting; this does not make the source data LAFAN data.

## Single Sequence Motion Retargeting

The following are the original task workflows, updated only to use the current
hyphenated CLI and canonical data-format names:

```bash
# Robot-only (OMOMO)
python examples/robot_retarget.py \
  --data-path demo_data/OMOMO_new \
  --task-type robot_only \
  --task-name sub3_largebox_003 \
  --data-format omomo \
  --retargeter.debug \
  --retargeter.visualize

# Object interaction (OMOMO)
python examples/robot_retarget.py \
  --data-path demo_data/OMOMO_new \
  --task-type object_interaction \
  --task-name sub10_tripod_000 \
  --data-format omomo \
  --robot e1 \
  --retargeter.debug \
  --retargeter.visualize

# Climbing (generic mocap)
python examples/robot_retarget.py \
  --data-path demo_data/climb \
  --task-type climbing \
  --task-name mocap_climb_seq_0 \
  --data-format mocap \
  --robot-config.robot-urdf-file models/g1/g1_29dof_spherehand.urdf \
  --retargeter.debug \
  --retargeter.visualize
```

Additional robot-only formats use the same entry point:

```bash
# LAFAN
python examples/robot_retarget.py \
  --data-path demo_data/lafan \
  --task-type robot_only \
  --task-name dance2_subject1 \
  --data-format lafan \
  --task-config.ground-range -10 10 \
  --save-dir demo_results/g1/robot_only/lafan \
  --retargeter.foot-sticking-tolerance 0.02

# AMASS SMPL-X
python examples/robot_retarget.py \
  --data-path demo_data/amass_smplx_processed \
  --task-type robot_only \
  --task-name ACCAD_Female1Running_c3d_C3_-_Run_stageii \
  --data-format amass \
  --task-config.ground-range -10 10 \
  --save-dir demo_results/g1/robot_only/amass

# Noetix-mocap
python examples/robot_retarget.py \
  --data-path demo_data/noetix_mocap \
  --task-type robot_only \
  --task-name run_to_the_right \
  --data-format noetix_mocap \
  --save-dir demo_results/g1/robot_only/noetix_mocap

# GVHMR
python examples/robot_retarget.py \
  --data-path demo_data/gvhmr \
  --task-type robot_only \
  --task-name tennis \
  --data-format gvhmr \
  --save-dir demo_results/g1/robot_only/gvhmr
```

Add `--augmentation` to run an augmented object-interaction or climbing
sequence. The corresponding original sequence must be generated first.
For OMOMO object interaction, the object category is inferred from the
canonical `subN_object_NNN` task name. An explicit
`--task-config.object-name` is optional and is treated as a consistency check.

## Batch Processing for Motion Retargeting

The original batch workflows remain available:

```bash
# Robot-only (OMOMO)
python examples/parallel_robot_retarget.py \
  --data-dir demo_data/OMOMO_new \
  --task-type robot_only \
  --data-format omomo \
  --save-dir demo_results_parallel/g1/robot_only/omomo \
  --task-config.object-name ground

# All OMOMO object-interaction sequences on G1
python examples/parallel_robot_retarget.py \
  --data-dir demo_data/OMOMO_new \
  --task-type object_interaction \
  --data-format omomo \
  --robot g1 \
  --save-dir demo_results_parallel/g1/object_interaction/omomo \
  --max-workers 4

# Climbing
python examples/parallel_robot_retarget.py \
  --data-dir demo_data/climb \
  --task-type climbing \
  --data-format mocap \
  --robot-config.robot-urdf-file models/g1/g1_29dof_spherehand.urdf \
  --task-config.object-name multi_boxes \
  --save-dir demo_results_parallel/g1/climbing/mocap_climb
```

The same batch command works with `lafan`, `amass`, `noetix_mocap`, or `gvhmr`
by changing `--data-format`, `--data-dir`, and `--save-dir`. For example:

```bash
python examples/parallel_robot_retarget.py \
  --data-dir demo_data/gvhmr \
  --task-type robot_only \
  --data-format gvhmr \
  --save-dir demo_results_parallel/g1/robot_only/gvhmr \
  --max-workers 4 \
  --retargeter.save-interaction-mesh
```

Add `--augmentation` to process both original and augmented
object-interaction/climbing sequences. For every OMOMO object-interaction
motion, the batch produces `original`, `trans_0`, `trans_1`, `trans_2`,
`rot_0`, and `rot_1`. The translations are `[0.2, 0, 0]`, `[0, 0.2, 0]`,
and `[0, -0.2, 0]` meters in the initial human-to-object local frame, then
converted to world coordinates. The rotations are `+45` and `-45` degrees
around Z, paired with `[0, 0.2, 0]` and `[0, -0.2, 0]` meter local
translations. The full perturbation is applied before the object begins
moving, then exponentially decays after that frame. Each variant is solved
again from the original robot initial/nominal trajectory; it is not a
post-transform of an output qpos.

There is no motion-semantic allowlist. Every selected object-interaction
sequence that passes the data/asset preflight attempts all five augmentations.
To overwrite and regenerate the complete largebox set while saving both
pre/post-scale object points and Interaction Mesh data:

```bash
python examples/parallel_robot_retarget.py \
  --data-dir demo_data/OMOMO_new \
  --task-type object_interaction \
  --data-format omomo \
  --robot g1 \
  --object-names largebox \
  --save-dir demo_results_parallel/g1/object_interaction/omomo \
  --max-workers 4 \
  --augmentation \
  --overwrite-existing \
  --retargeter.save-interaction-mesh
```

Existing output files are skipped unless `--overwrite-existing` is set. OMOMO
batches run a dataset/height/asset preflight by default and write
`batch_report.json` under the save directory.

The supported OMOMO catalog is `clothesstand`, `floorlamp`, `largebox`,
`largetable`, `monitor`, `plasticbox`, `smallbox`, `smalltable`, `suitcase`,
`trashcan`, `tripod`, `whitechair`, and `woodchair`. Omit `--object-names` to
process all categories, or select an exact subset. Use `--dry-run` first to
validate all inputs and write the manifest without starting optimization:

```bash
# Preflight and manifest for all thirteen categories
python examples/parallel_robot_retarget.py \
  --data-dir demo_data/OMOMO_new \
  --task-type object_interaction \
  --data-format omomo \
  --robot g1 \
  --save-dir demo_results_parallel/g1/object_interaction/omomo \
  --dry-run

# Resume a selected subset on E1
python examples/parallel_robot_retarget.py \
  --data-dir demo_data/OMOMO_new \
  --task-type object_interaction \
  --data-format omomo \
  --robot e1 \
  --object-names tripod suitcase whitechair \
  --save-dir demo_results_parallel/e1/object_interaction/omomo \
  --max-workers 4
```

## Data Preparation

We provide `demo_data/` for fast testing. Follow the dataset-specific
instructions below to prepare more sequences.

### OMOMO

The original Holosoma pipeline uses the dataset processed by InterMimic. Its
format differs from the original OMOMO release.

1. Download the processed OMOMO data from
   [this link](https://drive.google.com/file/d/141YoPOd2DlJ4jhU2cpZO5VU5GzV_lm5j/view).
2. Extract it to `demo_data/OMOMO_new`.
3. Place `height_dict.pkl` in the parent directory of `OMOMO_new`. Alternatively,
   pass `--motion-data-config.human-height HEIGHT`.

The motion files should be `.pt` tensors.

Validate the complete directory, the subject-height table, and all thirteen
package assets before a long run:

```bash
python data_utils/preflight_omomo.py demo_data/OMOMO_new \
  --asset-root models \
  --output demo_results_parallel/omomo_preflight.json
```

For a release-level smoke test, run two real optimizer frames for every
G1/E1-object pair. This creates and validates 26 result NPZ files:

```bash
python data_utils/validate_omomo_retargeting.py demo_data/OMOMO_new \
  --robots g1 e1 \
  --frames 2 \
  --output-dir demo_results_validation/omomo_g1_e1
```

### LAFAN

#### Download the Original LAFAN Data

1. Download
   [lafan1.zip](https://github.com/ubisoft/ubisoft-laforge-animation-dataset/blob/master/lafan1/lafan1.zip)
   by clicking **View Raw**.
2. Put `lafan1.zip` in the designated data folder and extract it to
   `DATA_FOLDER_PATH/lafan`.
3. The original files should have the layout
   `DATA_FOLDER_PATH/lafan/*.bvh`.

#### Convert LAFAN for Motion Retargeting

The conversion uses files from the
[LAFAN GitHub repository](https://github.com/ubisoft/ubisoft-laforge-animation-dataset):

```bash
cd data_utils
git clone https://github.com/ubisoft/ubisoft-laforge-animation-dataset.git
mv ubisoft-laforge-animation-dataset/lafan1 .
python extract_global_positions.py \
  --input-dir DATA_FOLDER_PATH/lafan \
  --output-dir ../demo_data/lafan
cd ..
```

This converts each BVH file to a `.npy` file containing global joint positions.
The `.npy` data remains in the official LAFAN Y-up convention; the retargeting
loader converts it to Z-up. For LAFAN, relax the foot-sticking constraint with
`--retargeter.foot-sticking-tolerance 0.02` and adjust it further if required by
the motion quality.

### AMASS SMPL-X

#### Download the Original AMASS Data

1. Follow the [AMASS](https://amass.is.tue.mpg.de/) instructions to download the
   original AMASS data.
2. The expected layout is
   `/path/to/amass/dataset_name/subject_name/*_stageii.npz`.

#### Download SMPL-X Models

1. Follow the [SMPL-X](https://smpl-x.is.tue.mpg.de/index.html) instructions to
   download the licensed SMPL-X models.
2. The original pipeline was tested with the neutral SMPL-X model.
3. The expected model path is
   `/path/to/models/smplx/SMPLX_NEUTRAL.npz`.

#### Convert AMASS SMPL-X for Motion Retargeting

The original documentation used the
[human_body_prior](https://github.com/nghorbani/human_body_prior) utilities. The
current converter keeps the same AMASS input/output workflow but directly uses
the installed SMPL-X model:

```bash
python data_utils/prep_amass_smplx_for_rt.py \
  --amass-root-folder /path/to/amass \
  --output-folder /path/to/output \
  --model-root-folder /path/to/models
```

The converter recursively writes `.npz` files containing global joint
positions, height, FPS, joint names, and root quaternions. Use
`--subdataset-folder HumanEva` to process one subset, or omit it to process all
subsets.

### Climbing Mocap

The official climbing task uses nested sequence directories:

```text
demo_data/climb/
└── mocap_climb_seq_0/
    ├── <motion>.npy
    ├── multi_boxes.obj
    ├── multi_boxes.urdf
    ├── box_assets.xml
    └── <robot scene>.xml
```

The motion is sampled from 120 FPS to 30 FPS by the `mocap` adapter. Terrain
assets stay inside the sequence directory. Use the sphere-hand robot URDF shown
in the single and batch climbing commands.

### Noetix-mocap

Convert the supported company-collected Noetix BVH variants into the normalized
22-joint Noetix retargeting schema:

```bash
python data_utils/convert_noetix_bvh.py \
  --input-dir demo_data/noetix \
  --output-dir demo_data/noetix_mocap \
  --target-fps 30
```

This is a Noetix-specific converter and is unrelated to the official LAFAN BVH
conversion above. The output `.npz` files include Z-up global positions in
`global_joint_positions`, global `wxyz` orientations in
`global_joint_quaternions_wxyz`, joint names, source/output FPS, human height,
the detected Noetix skeleton type, and conversion metadata. Positions use the
`[x, z, y]` basis change, while rotations use the corresponding matrix
conjugation `R_z = S R_y S^T`; quaternion components must not be swapped
directly. Use `--overwrite` to replace existing outputs.

### Noetix E1 Orientation Tracking and Ablation

Orientation tracking is an optional soft objective independent of the
position-based Interaction Mesh. Following GMR's global rigid-frame task
design, it keeps a fixed human-joint-to-robot-link frame alignment and minimizes
the SO(3) geodesic residual `Log(R_target R_robot^T)` through MuJoCo world
angular Jacobians. The default empty `orientation_weights` keeps the legacy
position-only path unchanged.

Orientation targets use a T-pose calibration by default. For every mapped
joint/link pair, `R_align = R_human,T^T R_robot,T` and
`R_target(t) = R_human(t) R_align`. The canonical Noetix zero-rotation
skeleton faces world +Y, while E1 qpos0 faces +X, so the E1 reference root is
rotated +90 degrees about world Z and its shoulder-roll joints are set to
`+pi/2` on the left and `-pi/2` on the right. This forms a geometric robot
T-pose instead of the E1 model's arms-down qpos0. MuJoCo FK then derives all 13
fixed link offsets and automatically captures the E1 elbow/hand links' fixed
frame rotations.
The legacy `first_frame` mode remains available only for reproducibility,
because an arbitrary first motion frame is not a valid calibration pose.

The `shoulders_feet` ablation profile assigns the same requested weight to
`LeftArm`, `RightArm`, `LeftFoot`, and `RightFoot`. On E1 these target the two
shoulder-yaw bodies and the two ankle-roll bodies; all other orientation
weights remain zero.

The `full_equal` profile tracks all 13 configured links with exactly the same
requested weight. This differs from `full`, which retains nonuniform relative
coefficients for the legs, feet, forearms, and hands.

Run the reproducible `breaking+hippop` baseline, grouped-link, and full-link
ablations with the `gmr_legs` profile mirroring the hip, knee, and foot chains
whose rotation costs are enabled by GMR's primary E1 task. The `shoulders`
profile only weights `LeftArm/RightArm → l/r_arm_shoulder_yaw_link`; all other
diagnostic links remain at exactly zero weight. GMR's numerical weights and
fixed quaternions are not copied because the MJCF frames, coordinate conversion,
and objective scales differ:

```bash
python examples/run_orientation_ablation.py \
  --data-path demo_data/noetix_mocap/0724_BEITI \
  --task-name 'breaking+hippop.bvh_Skeleton1' \
  --output-root demo_results_orientation/e1/robot_only/0724_BEITI/breaking+hippop \
  --variants baseline root feet gmr_legs shoulders upper full \
  --weight-scales 0.25 0.5 1.0 \
  --overwrite
```

Each run records a manifest and saves target/robot link quaternions, per-link
geodesic errors, mapped-position errors, and SQP diagnostics. The baseline
stores the same 13-link diagnostics with zero weights while adding no
orientation term to the optimization. Use `--frame-start` and `--frame-count`
for fast windowed tuning before confirming selected settings on the complete
sequence. The converted source frames can be inspected with:

```bash
python examples/raw_human_motion_viewer.py \
  --motion-path demo_data/noetix_mocap/0724_BEITI/breaking+hippop.bvh_Skeleton1.npz \
  --show-joint-orientations
```

Compare target and actual shoulder frames on a shared timeline with:

```bash
python examples/ablation_viser_player.py \
  --qpos-npzs \
    demo_results_orientation/e1/robot_only/0724_BEITI/breaking+hippop/baseline/breaking+hippop.bvh_Skeleton1.npz \
    demo_results_orientation/e1/robot_only/0724_BEITI/breaking+hippop/shoulders_x0.025/breaking+hippop.bvh_Skeleton1.npz \
  --labels baseline shoulders_x0.025 \
  --x-offset 0.7 \
  --orientation-joints LeftArm RightArm \
  --show-orientation-error-labels
```

The player uses one shared human-reference layer plus one layer per robot
result. The charcoal human skeleton contains only mapped keypoints used by the
retargeting position objective, with calibrated target frames located on the
human joints. Each robot layer combines a pale low-opacity mesh, darker
same-hue mapped skeleton, and actual link frames under one visibility control.
Robot layers use a colorblind-safe blue, vermilion, and bluish-green palette
in input order. Redundant shoulder and hip crossbars are omitted from the
display topology, while `--x-offset` separates all layers. RGB arrows are
local X, Y, and Z. Use
`--variants baseline shoulders --weight-scales 0.01 0.025 0.05` to tune only
the shoulder orientation objective.

### GVHMR

Run GVHMR first so that it produces `hmr4d_results.pt`. Holosoma uses the
world-frame SMPL-X parameters in `smpl_params_global`:

```bash
python data_utils/convert_gvhmr.py \
  --input-file /home/jinhaogao/GVHMR/outputs/demo/tennis/hmr4d_results.pt \
  --output-file demo_data/gvhmr/tennis.npz \
  --model-path models/smplx \
  --fps 30
```

The converter runs batched SMPL-X forward kinematics, converts GVHMR's
right-handed Y-up world frame to right-handed Z-up, computes shape-specific
height, and writes root quaternions in `wxyz` order.

## Inspect Raw Human Motion Before Retargeting

Use the unified raw-motion viewer to determine whether a contact or pose issue
already exists in the source data or is introduced later by normalization and
retargeting:

```bash
# OMOMO object interaction: complete 52-joint skeleton and moving source mesh
python examples/raw_human_motion_viewer.py \
  --motion-path demo_data/OMOMO_new/sub3_largebox_003.pt

# Climbing: complete 53-joint skeleton and static source multi_boxes.obj
python examples/raw_human_motion_viewer.py \
  --motion-path demo_data/climb/mocap_climb_seq_0

# Robot-only source without a scene mesh
python examples/raw_human_motion_viewer.py \
  --motion-path demo_data/lafan/dance1_subject1.npy
```

The viewer supports `omomo`, `mocap`, `lafan`, `amass`, `gvhmr`, and
`noetix_mocap`. It infers the data format from the selected file and infers the
task context from its contents and neighboring assets. A dataset directory can
also be passed together with `--sequence`, for example
`--motion-path demo_data/OMOMO_new --sequence sub3_largebox_003`. Use an
explicit `--task-type robot_only` when inspecting an OMOMO file without its
object, and use `--mesh-path PATH` to override automatic mesh lookup.

This tool intentionally bypasses `preprocess_motion_data`: it does not apply
the foot-height shift, robot-height scale, augmentation, or retargeting. OMOMO
meshes use the raw per-frame object pose, climbing meshes use the static source
coordinates, and data without a corresponding mesh is shown as a skeleton
only. “Raw” here means the registered adapter output; format-required decoding
still applies, including the LAFAN coordinate conversion and the declared
mocap temporal sampling. Add `--dry-run` to validate and summarize the scene
without starting Viser.

The keyboard controls follow the other Viser players:

| Key | Action |
| --- | --- |
| `Space` | Play or pause |
| `[` / `]` | Previous or next frame |
| `Home` / `End` | First or last frame |
| `K` | Toggle all human keypoints |
| `.` | Toggle the complete skeleton |
| `L` | Toggle joint-name labels |
| `O` | Toggle the source object or terrain mesh |

## Check Visualizations of Saved Retargeting Results

New results store robot, data-format, object, FPS, skeleton, and qpos-layout
metadata, so the viewer normally only needs the result path:

```bash
# OMOMO object interaction
python viser_player.py \
  --qpos-npz demo_results/g1/object_interaction/omomo/sub3_largebox_003_original.npz

# Climbing
python viser_player.py \
  --qpos-npz demo_results/g1/climbing/mocap/mocap_climb_seq_0_original.npz

# Robot-only OMOMO
python viser_player.py \
  --qpos-npz demo_results/g1/robot_only/omomo/sub3_largebox_003.npz

# LAFAN
python viser_player.py \
  --qpos-npz demo_results/g1/robot_only/lafan/dance2_subject1.npz

# AMASS
python viser_player.py \
  --qpos-npz demo_results/g1/robot_only/amass/ACCAD_Female1Running_c3d_C3_-_Run_stageii.npz

# Noetix-mocap
python viser_player.py \
  --qpos-npz demo_results/g1/robot_only/noetix_mocap/run_to_the_right.npz

# GVHMR
python viser_player.py \
  --qpos-npz demo_results/g1/robot_only/gvhmr/tennis.npz
```

For old results without metadata, preserve the original explicit viewer form:

```bash
# Object interaction
python viser_player.py \
  --robot-urdf models/g1/g1_29dof.urdf \
  --object-urdf models/largebox/largebox.urdf \
  --qpos-npz demo_results_parallel/g1/object_interaction/omomo/sub3_largebox_003_original.npz

# Original climbing result
python viser_player.py \
  --robot-urdf models/g1/g1_29dof_spherehand.urdf \
  --object-urdf demo_data/climb/mocap_climb_seq_0/multi_boxes.urdf \
  --qpos-npz demo_results_parallel/g1/climbing/mocap_climb/mocap_climb_seq_0_original.npz

# Augmented climbing result
python viser_player.py \
  --robot-urdf models/g1/g1_29dof_spherehand.urdf \
  --object-urdf demo_data/climb/mocap_climb_seq_0/multi_boxes_scaled_0.74_0.74_0.89.urdf \
  --qpos-npz demo_results_parallel/g1/climbing/mocap_climb_aug/mocap_climb_seq_0_z_scale_1.2.npz
```

### Skeleton, Opacity, and Interaction Mesh

Add `--retargeter.save-interaction-mesh` to a single or batch retargeting command
to save the source and target Interaction Mesh. During live retargeting, use:

```bash
python examples/robot_retarget.py \
  --data-path demo_data/OMOMO_new \
  --task-type object_interaction \
  --task-name sub3_largebox_003 \
  --data-format omomo \
  --retargeter.visualize \
  --retargeter.debug \
  --retargeter.mesh-opacity 0.4 \
  --retargeter.show-interaction-mesh \
  --retargeter.save-interaction-mesh \
  --retargeter.interaction-mesh-mode both \
  --retargeter.interaction-mesh-edges cross
```

Replay all overlays with:

```bash
python viser_player.py \
  --qpos-npz demo_results/g1/object_interaction/omomo/sub3_largebox_003_original.npz \
  --show-mapped-skeletons \
  --show-object-keypoints \
  --show-interaction-mesh \
  --interaction-mesh-mode both \
  --interaction-mesh-edges cross \
  --robot-mesh-opacity 0.45 \
  --object-mesh-opacity 0.25
```

To compare an original result and all five augmentations in one synchronized
scene, pass any one of the six matching files to:

```bash
python augmentation_viser_player.py \
  --qpos-npz demo_results_parallel/g1/object_interaction/omomo/sub3_largebox_003_original.npz
```

This viewer discovers the six siblings and color-codes each robot mesh, mapped
human skeleton, mapped robot skeleton, and object mesh, with a visibility
toggle per series. The result files still contain Interaction Mesh data, but
the grouped viewer intentionally does not load or draw it to avoid six
tetrahedral edge sets obscuring the scene. Use the single-result
`viser_player.py --show-interaction-mesh` command above to inspect the
Interaction Mesh for one variant.

For OMOMO, `--show-mapped-skeletons` augments the original 15 blue mapped
joints with smaller blue points and lines for both complete SMPL-H finger
chains. The finger joints are diagnostic only; they are not added to the
Interaction Mesh and do not change optimization. `--show-object-keypoints`
draws the exact per-frame object points saved during retargeting: red is the
demo object normalized with the human height, and cyan is the object at the
target asset scale. Because the result stores the world-space points actually
used by that run, replay does not resample the object surface.

`--mesh-opacity` provides a shared opacity. `--robot-mesh-opacity` and
`--object-mesh-opacity` override it independently. Skeleton point/line size,
object point size, and Interaction Mesh line width are controlled by
`--skeleton-point-radius`, `--skeleton-line-width`,
`--object-keypoint-radius`, and `--interaction-mesh-line-width`.

## Result NPZ Contract

New retargeting results contain `qpos`, `fps`, `cost`, the full and mapped human
skeletons, mapped robot skeleton positions, and
`source_data_format`/`robot_type`/object metadata. Object-interaction results
also always contain `object_points_demo_local`, `object_points_target_local`,
`object_points_demo_world`, and `object_points_target_world`, which preserve
the demo/target-scale local samples and their per-frame world coordinates. If
`--retargeter.save-interaction-mesh` is enabled, per-frame source/target
vertices and tetrahedra are included as well. If the normal 1 mm foot-sticking
constraints make one frame infeasible, that frame is first retried with
`foot_sticking_fallback_tolerance`, then with only its foot-sticking constraints
released. The affected frame indices are saved in
`foot_sticking_fallback_frames` and `foot_sticking_release_frames`. If the
trajectory has already entered a state that remains infeasible after this
local release, the complete sequence is restarted without foot sticking while
all other constraints remain enabled. The saved result records this in
`foot_sticking_enabled_for_saved_trajectory` and
`foot_sticking_full_sequence_retry_frame`. Robot-object non-penetration is not
released by default. The opt-in
`--retargeter.release-object-non-penetration-on-infeasible` fallback releases
only the failing frame's robot-object constraint while preserving ground
non-penetration. Such a result is degraded rather than collision-valid; the
affected frames are recorded in `object_non_penetration_release_frames`.

Robot-only qpos uses `[root_xyz, root_wxyz, robot_dof]`. Dynamic-object qpos
appends `[object_xyz, object_wxyz]`.

## Quantitative Evaluation

The original evaluation workflows remain available:

```bash
# Robot-object interaction
python evaluation/eval_retargeting.py \
  --res-dir demo_results_parallel/g1/object_interaction/omomo \
  --data-dir demo_data/OMOMO_new \
  --data-type robot_object

# Climbing
python evaluation/eval_retargeting.py \
  --res-dir demo_results_parallel/g1/climbing/mocap_climb \
  --data-dir demo_data/climb \
  --data-type robot_terrain \
  --robot-config.robot-urdf-file models/g1/g1_29dof_spherehand.urdf

# Robot-only OMOMO
python evaluation/eval_retargeting.py \
  --res-dir demo_results_parallel/g1/robot_only/omomo \
  --data-dir demo_data/OMOMO_new \
  --data-type robot_only
```

Robot-only evaluation uses the preprocessed human skeleton saved in each new
result. Therefore, the same command also supports LAFAN, AMASS, Noetix-mocap,
and GVHMR when `--data-format` and the paths are changed.
For OMOMO robot-object results, evaluation resolves the object independently
for every NPZ from saved metadata and the canonical filename. Do not pass
`--object-name` for a directory containing multiple object categories.

## Prepare Data for Training RL Whole-Body Tracking Policy

The official workflow has two steps:

1. Run retargeting to obtain a `.npz` robot motion.
2. Convert that result to the whole-body tracking policy format at the desired
   frame rate.

On macOS, use `mjpython` instead of `python`.

### macOS (`mjpython`)

```bash
mjpython data_conversion/convert_data_format_mj.py \
  --input-file ./demo_results/g1/robot_only/omomo/sub3_largebox_003.npz \
  --output-fps 50 \
  --output-name converted_res/robot_only/sub3_largebox_003_mj_fps50.npz \
  --data-format omomo \
  --object-name ground \
  --once

mjpython data_conversion/convert_data_format_mj.py \
  --input-file ./demo_results/g1/object_interaction/omomo/sub3_largebox_003_original.npz \
  --output-fps 50 \
  --output-name converted_res/object_interaction/sub3_largebox_003_mj_w_obj.npz \
  --data-format omomo \
  --has-dynamic-object \
  --once
```

### Robot-Only Setting

```bash
python data_conversion/convert_data_format_mj.py \
  --input-file ./demo_results/g1/robot_only/omomo/sub3_largebox_003.npz \
  --output-fps 50 \
  --output-name converted_res/robot_only/sub3_largebox_003_mj_fps50.npz \
  --data-format omomo \
  --object-name ground \
  --once

python data_conversion/convert_data_format_mj.py \
  --input-file ./demo_results/g1/robot_only/lafan/dance2_subject1.npz \
  --output-fps 50 \
  --output-name converted_res/robot_only/dance2_subject1_mj_fps50.npz \
  --data-format lafan \
  --object-name ground \
  --once
```

### Robot-Object Setting

```bash
python data_conversion/convert_data_format_mj.py \
  --input-file ./demo_results/e1/object_interaction/omomo/sub10_tripod_000_original.npz \
  --robot e1 \
  --output-fps 50 \
  --output-name converted_res/object_interaction/sub10_tripod_000_mj_w_obj.npz \
  --data-format omomo \
  --has-dynamic-object \
  --once
```

Dynamic OMOMO conversion infers the category from result metadata or the
canonical filename and rejects conflicting explicit overrides.

### OmniRetarget Data

For OmniRetarget data downloaded from Hugging Face, add
`--use-omniretarget-data`:

```bash
python data_conversion/convert_data_format_mj.py \
  --input-file OmniRetarget/robot-object/sub3_largebox_003_original.npz \
  --output-fps 50 \
  --output-name converted_res/object_interaction/sub3_largebox_003_mj_w_obj_omnirt.npz \
  --data-format omomo \
  --object-name largebox \
  --has-dynamic-object \
  --use-omniretarget-data \
  --once
```

## Custom Human Motion Data Format

See [ADD_MOTION_FORMAT_README.md](ADD_MOTION_FORMAT_README.md).

## Custom Robot Type

See [ADD_ROBOT_TYPE_README.md](ADD_ROBOT_TYPE_README.md).
