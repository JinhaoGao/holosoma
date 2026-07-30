# Holosoma Motion Retargeting

[简体中文](README_zh.md) | English

This package retargets OMOMO, LAFAN, AMASS SMPL-X, GVHMR, Noetix BVH, and
climbing mocap to humanoid robots through one shared load, preprocess, solve,
save, resume, and visualization contract. The public retargeting surface has
three families: `examples/robot_retarget.py` for one action,
`examples/parallel_robot_retarget.py` for a dataset batch or an augmented
family, and the scripts under `examples/` whose names contain `ablation`,
`comparison`, or `search` for controlled experiments.
`examples/rebuild_demo_results.py` orchestrates the validated G1/E1 demo
matrix; it is not a fourth solver or artifact kind. Persisted `run_kind` is
`single` for every identity result, `augmentation` only for a non-identity
motion transform, and `ablation` for an untransformed experiment variant.
Batch execution itself is orchestration and is never stored as a `run_kind`.

The pipeline always consumes world-space human joint positions with shape
`(T, J, 3)`. Source-human orientation is deliberately stricter: it is saved only
when the source representation directly contains rotations and the adapter can
convert those rotations to world-space `wxyz` quaternions without inferring
them from positions. A source may provide orientations for only a named subset
of its joints. If direct rotations are absent, the artifact records
`orientation_source="absent"` and omits the complete source-human orientation
group, including its tensor digest. Orientation inferred internally to initialize the robot is never presented as
source metadata. In particular, the position-only generic climbing data under
`demo_data/climb` does not save human orientations.

Each format declares its joint order and parent tree, robot mapping, file
layout, coordinate conversion, height, FPS, supported tasks, and approved
direct-orientation provenance. See
[ADD_MOTION_FORMAT_README.md](ADD_MOTION_FORMAT_README.md) when adding a format.

Run the commands below from:

```bash
cd src/holosoma_retargeting/holosoma_retargeting
```

## Supported Human Motion Data and Tasks

| `--data-format` | Canonical input | Tasks | Direct source-human orientation |
| --- | --- | --- | --- |
| `omomo` | `.pt`, 52 SMPL-H joints | `robot_only`, `object_interaction` | `intermimic_global_orientation_tensor` when columns 383:591 are present; otherwise `absent` |
| `lafan` | `.npz`, 22 LAFAN joints | `robot_only` | `bvh_rotation_channels_fk` |
| `amass` | `.npz`, 22 SMPL-X joints | `robot_only` | `direct_local_rotation_fk` |
| `gvhmr` | `.npz`, 22 SMPL-X joints | `robot_only` | `direct_local_rotation_fk` |
| `noetix_mocap` | `.npz`, normalized 22-joint Noetix schema | `robot_only` | `bvh_rotation_channels_fk`; the saved subset can be partial |
| `mocap` | nested `.npz` or `.npy`, 53 joints | `robot_only`, `climbing` | `bone_rotation_channels_fk` for converted Noetix CSV; `absent` for generic climb |

The table uses canonical format names. Compatibility aliases may still be
accepted by the loader, but new commands, paths, metadata, and documentation
use only the canonical names.

> **LAFAN and Noetix-mocap are independent datasets.** LAFAN is the public
> Ubisoft dataset and is converted from its BVH rotation channels into the
> canonical `.npz` contract.
> Noetix-mocap contains company-collected motion-capture data, accepts several
> Noetix BVH export variants, and follows a separate `BVH → .npz` workflow.
> The Noetix normalized skeleton reuses a compatible 22-joint naming topology
> for retargeting; this does not make the source data LAFAN data.

## Single Sequence Motion Retargeting

`robot_retarget.py` is the only single-action entry point. It resolves the
source through the same adapter registry used by batch processing, normalizes
the nested robot/data/task configuration, builds a versioned job, resumes only
an artifact whose schema, source path and SHA-256, normalized configuration,
run kind, and variant all match, and otherwise runs the shared solver lifecycle
when the canonical path is free. An existing path with a different or invalid
identity is rejected; pass `--overwrite-existing` only when replacing it is
intentional. Do not make a robot/task leaf directory with `--save-dir`; when
supplied, it is the result root. Omitting it uses `demo_results/v1`.

```bash
# Robot-only LAFAN on G1
python examples/robot_retarget.py \
  --data-path demo_data/lafan \
  --task-type robot_only \
  --task-name dance2_subject1 \
  --data-format lafan \
  --robot g1

# OMOMO object interaction on E1
python examples/robot_retarget.py \
  --data-path demo_data/OMOMO_new \
  --task-type object_interaction \
  --task-name sub10_tripod_000 \
  --data-format omomo \
  --robot e1

# Generic climbing on G1
python examples/robot_retarget.py \
  --data-path demo_data/climb \
  --task-type climbing \
  --task-name mocap_climb_seq_0 \
  --data-format mocap \
  --robot g1
```

The default artifact path is:

```text
demo_results/v1/canonical/<robot>/<task_type>/<data_format>/
  <dataset_partition>/<sequence_key>/<variant>.npz
```

For example, the first command writes
`demo_results/v1/canonical/g1/robot_only/lafan/lafan/dance2_subject1/identity.npz`.
Nested batch sequence keys remain nested, and unsafe source-derived path
characters are percent-encoded. OMOMO object names are inferred from canonical
`subN_object_NNN` names; an explicit `--task-config.object-name` acts as a
consistency check. The `v1` directory is the result-tree release namespace; the
NPZ contract inside it is schema version 2, so these two version numbers are
independent.

Every schema-v2 artifact includes the complete preprocessed human skeleton,
complete robot link poses and parent tree, robot actuated-joint order,
Interaction Mesh, solver diagnostics, and source/config identities. Direct
human orientations are saved only when the source adapter proves an approved
orientation field and then retain the exact named subset and tensor digest;
position-only sources omit that optional group. Non-ground results also bind
the object URDF and every directly referenced mesh/texture through
`object_asset_manifest_json` and `object_asset_manifest_sha256`. Exact resume,
staged validation, promotion, and strict visualization all recompute this
closure and fail closed if any referenced file is missing or changed.
Canonical `qpos` is always saved as `float64`, preserving the exact SQP state
accepted by the true-geometry gate and reused as an augmentation warm start.
Viewers and downstream converters may cast a loaded copy when lower precision
is sufficient, but the canonical artifact must not discard that precision.

Passing `--augmentation` to one object-interaction or climbing command runs the
complete identity-first family through the same interface. OMOMO produces
`identity`, `trans_0`, `trans_1`, `trans_2`, `rot_0`, and `rot_1`; climbing
produces `identity`, `z_scale_0p8`, `z_scale_0p9`, `z_scale_1p1`, and
`z_scale_1p2`. Robot-only tasks remain identity-only. The exact identity
artifact is created or validated before an augmented variant may reuse its
nominal trajectory.

## Batch Processing for Motion Retargeting

`parallel_robot_retarget.py` is the only multi-source and batch-augmentation
entry point. It uses the same job builder, canonical paths, schema writer, and
exact resume check as the single-action command. `--data-dir` selects the
dataset root, `--max-workers` controls process parallelism, `--dry-run` writes a
manifest without solving, and `--overwrite-existing` disables exact resume.
The default result root is again `demo_results/v1`. Unless explicitly
overridden, the batch report is
`<save_dir>/runs/<run_id>/report.json`.

```bash
# All GVHMR sequences on G1
python examples/parallel_robot_retarget.py \
  --data-dir demo_data/gvhmr \
  --task-type robot_only \
  --data-format gvhmr \
  --robot g1 \
  --max-workers 4 \
  --run-id g1-gvhmr

# All OMOMO object-interaction sequences and augmentations on E1
python examples/parallel_robot_retarget.py \
  --data-dir demo_data/OMOMO_new \
  --task-type object_interaction \
  --data-format omomo \
  --robot e1 \
  --augmentation \
  --max-workers 4 \
  --run-id e1-omomo-object

# Generic climbing and its four scale variants
python examples/parallel_robot_retarget.py \
  --data-dir demo_data/climb \
  --task-type climbing \
  --data-format mocap \
  --robot g1 \
  --augmentation \
  --max-workers 4 \
  --run-id g1-generic-climb
```

The same command covers `lafan`, `amass`, `noetix_mocap`, and `gvhmr` by
changing `--data-format` and `--data-dir`. Every valid matching artifact is
resumed. A file that merely exists but has stale source/config hashes or an
invalid schema is rejected instead of being silently replaced; use
`--overwrite-existing` to authorize replacement. OMOMO batches run data,
subject-height, and object-asset preflight by default. Object filtering is
explicit:

```bash
python examples/parallel_robot_retarget.py \
  --data-dir demo_data/OMOMO_new \
  --task-type object_interaction \
  --data-format omomo \
  --robot g1 \
  --object-names largebox \
  --max-workers 4 \
  --augmentation \
  --overwrite-existing \
  --run-id g1-largebox
```

The supported OMOMO catalog is `clothesstand`, `floorlamp`, `largebox`,
`largetable`, `monitor`, `plasticbox`, `smallbox`, `smalltable`, `suitcase`,
`trashcan`, `tripod`, `whitechair`, and `woodchair`. Omit `--object-names` to
process all categories. Use `--dry-run` to validate discovery and write the
batch report without starting optimization:

```bash
python examples/parallel_robot_retarget.py \
  --data-dir demo_data/OMOMO_new \
  --task-type object_interaction \
  --data-format omomo \
  --robot g1 \
  --dry-run \
  --run-id g1-omomo-plan
```

## Ablation and Search Entry Points

Controlled experiments also build `RetargetJob` objects and write schema-v2
artifacts below `demo_results/v1/ablations/<experiment>/...`; they do not have a
second solver/save implementation. `run_orientation_ablation.py` runs named
semantic orientation profiles for one directly oriented action on a selected
robot and registered motion format,
`search_orientation_weights.py` evaluates JSON-defined candidates across
windows, and `run_orientation_comparison_batch.py` compares the zero-weight and
checked `balanced_optimal` profiles across the Noetix collection. Manifests
remain beside their ablation artifacts. A single-action orientation-ablation
summary has a collision-free scoped path:
`demo_results/v1/ablations/orientation_ablation/_summaries/<robot>/<task>/<format>/<dataset>/<sequence>/summary.json`.
Consequently, concurrent robots, tasks, formats, datasets, and sequences never
overwrite one global `summary.json`.

Here `<dataset>` is the readable dataset basename followed by the complete
SHA-256 of its canonical root path, and each raw sequence component is passed
through the shared reversible percent encoder. Therefore `walk+a` and `walk a`
remain distinct, as do identically named dataset directories under different
parents. Frame-window and orientation-comparison preparation caches record
canonical lineage JSON and its hash together with a hash of the complete named
array payload. Before reuse, the expected payload is freshly derived from the
formal source and every key, dtype, shape, and C-order byte is compared; a
mismatch or self-consistent metadata tamper is rebuilt atomically. With
`--fail-fast`, both entry points write a terminal `failed` summary containing
completed, cancelled, and not-submitted work before raising. The batch
comparison report is independently scoped at
`demo_results/v1/ablations/orientation_comparison/_summaries/dataset-path-sha256-<digest>/batch_summary.json`.

```bash
python examples/run_orientation_ablation.py \
  --robot e1 \
  --task-type robot_only \
  --data-format noetix_mocap \
  --data-path demo_data/noetix_mocap/0724_BEITI \
  --task-name breaking+hippop.bvh_Skeleton1 \
  --output-root demo_results/v1 \
  --variants baseline balanced_optimal \
  --dry-run

python examples/search_orientation_weights.py \
  --candidate-file examples/orientation_weight_candidates_finalists.json \
  --output-root demo_results/v1
```

Noetix G1/E1 uses the shared calibrated `t_pose` alignment by default. AMASS,
GVHMR, and OMOMO currently have no complete shared T-pose calibration, so a
nonzero profile and its comparable zero-weight diagnostic baseline must
explicitly pass
`--orientation-alignment-mode first_frame`. That mode consumes only
adapter-proven direct source quaternions and robot FK; it never estimates
orientation from positions or bone directions. Profile roles are translated to
the selected format's source joint names, while robot links come from the
shared format/robot mapping. Full inputs are resolved through the motion-format
registry. Frame windows are accepted only for metadata-preserving NPZ adapters;
other formats fail with an explicit request to run the complete source.

For example, plan the same semantic profiles for AMASS/G1 with an explicit
direct-orientation first-frame alignment:

```bash
python examples/run_orientation_ablation.py \
  --robot g1 \
  --task-type robot_only \
  --data-format amass \
  --data-path demo_data/amass_smplx_processed \
  --task-name ACCAD_Female1Running_c3d_C3_-_Run_stageii \
  --output-root demo_results/v1 \
  --orientation-alignment-mode first_frame \
  --variants baseline full \
  --dry-run
```

## Complete Validated Demo Rebuild

`rebuild_demo_results.py` is the formal matrix orchestrator. It plans G1 and E1
for OMOMO robot-only, OMOMO object interaction plus augmentation, AMASS, GVHMR,
LAFAN, Noetix BVH, generic climbing plus augmentation, and Noetix CSV climbing
plus augmentation before starting a solver. G1 is a strict full-coverage
release target: every planned source and variant must exist and pass schema,
source/config identity, complete human and robot topology, direct-orientation
presence or absence, Interaction Mesh, float64 accepted qpos, all six
true-geometry residual arrays, all three accepted ConstraintMode arrays, asset
closure, and release-quality checks.

E1 is attempted over the same planned matrix, but its morphology can make some
object-interaction and climbing motions structurally unreachable. Such a case
may be omitted only after the real E1 job was attempted and its bounded failure
record was persisted in the aggregate report. The report separates successful
artifacts from explicit structural omissions and retains the source, job
identity, task family, and failure reason. It never fabricates a successful
artifact, silently skips an input, estimates missing metadata, or relaxes
schema and hard-constraint validation. E1 robot-only remains the expected
high-coverage subset. Promotion therefore requires the complete strict G1
matrix plus a fully accounted E1 matrix whose successes validate strictly and
whose allowed structural gaps are explicit and auditable.

```bash
# Read-only full plan
python examples/rebuild_demo_results.py \
  --run-id demo-rebuild-v1 \
  --max-workers 24 \
  --dry-run

# Resume or run the staging matrix, validate it, then transactionally promote it
python examples/rebuild_demo_results.py \
  --run-id demo-rebuild-v1 \
  --max-workers 24 \
  --promote

# Revalidate an existing staging tree without running solvers
python examples/rebuild_demo_results.py \
  --run-id demo-rebuild-v1 \
  --validate-only
```

The stable staging path is
`demo_results/.staging/<run_id>/v1`, the aggregate report is
`demo_results/runs/<run_id>/report.json`, and successful promotion renames the
staged tree to `demo_results/v1`. An existing formal `v1` is first moved below
`demo_results/archive/<UTC timestamp>/v1`. Promotion holds the shared
results-state lock, rechecks the validated staging content digest and every
external object-asset closure, and uses an atomic directory exchange. If the
filesystem cannot exchange the existing formal and staged trees atomically,
promotion is refused without moving either tree. A nonblocking lock under
`demo_results/.locks/rebuild-<run_id>.lock` covers planning through the final
report, so two orchestrators cannot share one staging/report identity.
Generated robot/object scene assets live outside the promoted tree under
`demo_results/.generated-assets/jobs`, while per-job concurrency locks live
under `demo_results/.locks/retargeting`.

## Organize Legacy Result Trees

After a validated `demo_results/v1` has been promoted, use the dedicated
organizer to archive only the legacy layouts known by the repository. Its
default mode is read-only and prints an exact tree inventory and destination:

```bash
python examples/organize_demo_results.py \
  --promoted-run-id demo-rebuild-v1
```

Review that plan before explicitly executing the rename transaction:

```bash
python examples/organize_demo_results.py \
  --promoted-run-id demo-rebuild-v1 \
  --execute
```

The command handles the old per-robot directories inside `demo_results` and
the fixed legacy sibling result directories. `--promoted-run-id` binds the
transaction to a successful rebuild report and to the current formal `v1`
fingerprint. The organizer writes a durable manifest under
`demo_results/archive/legacy/<UTC timestamp>/manifest.json`, rejects unknown
top-level entries or symlinks, and verifies that the plan is still current.
If execution is interrupted, rerun the same fixed timestamp with `--resume`
and `--execute`; the manifest reconciles each source/target rename without
guessing. Execution holds the same results-state lock used by promotion. It
does not touch `v1`, `runs`, `.staging`, `.generated-assets`, `.locks`, or
existing archives.

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
  --output demo_results/runs/omomo_preflight.json
```

For a release-level smoke test, run two real optimizer frames for every
G1/E1-object pair. This creates and validates 26 result NPZ files:

```bash
python data_utils/validate_omomo_retargeting.py demo_data/OMOMO_new \
  --robots g1 e1 \
  --frames 2 \
  --output-dir demo_results/runs/omomo_g1_e1_validation
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

This converts each BVH to a canonical Z-up `.npz` containing global joint
positions, the 22 direct BVH rotations converted to global `wxyz` quaternions
by FK, joint names and parent indices, FPS, height, source path, coordinate
conventions, and `orientation_source="bvh_rotation_channels_fk"`. No
orientation is estimated from joint positions. For LAFAN, relax the
foot-sticking constraint with `--retargeter.foot-sticking-tolerance 0.02` and
adjust it further if required by the motion quality.

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
positions, height, FPS, joint names and parent topology, and all 22 global
`wxyz` quaternions obtained by FK from the source SMPL-X local axis-angle
rotations. Their provenance is
`orientation_source="direct_local_rotation_fk"`. Use
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

The motion is sampled from 120 FPS to 30 FPS by the `mocap` adapter. These
generic climb files contain positions only, so no source-human orientation
arrays are saved. Terrain assets stay inside the sequence directory. Converted
Noetix CSV climbing directories use the same 53-joint `mocap` task interface
but retain direct `Bone Rotation` channels with
`orientation_source="bone_rotation_channels_fk"`.

### Noetix-mocap

Convert the supported company-collected Noetix BVH variants into the normalized
22-joint Noetix retargeting schema:

```bash
python data_utils/convert_noetix_bvh.py \
  --input-dir demo_data/noetix_ori \
  --output-dir demo_data/noetix_mocap \
  --target-fps 30 \
  --overwrite
```

This Noetix-specific converter is independent of LAFAN. The output `.npz`
contains Z-up `global_joint_positions`, paired
`orientation_joint_names`/`orientation_quaternions_wxyz`, joint names and
parents, source/output FPS, human height, detected skeleton type, source path,
and `orientation_source="bvh_rotation_channels_fk"`. The orientation subset
contains only canonical joints backed by direct source rotation channels;
positions synthesized for missing canonical joints do not receive inferred
orientations. Positions use the `[x, z, y]` basis change, while rotations use
the matching matrix conjugation `R_z = S R_y S^T`. Use `--overwrite` to replace
existing outputs.

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
T-pose instead of the E1 model's arms-down qpos0. MuJoCo FK then derives all 15
fixed link offsets and automatically captures the E1 elbow/hand links' fixed
frame rotations.
Noetix G1/E1 keeps this shared `t_pose` default. Formats without a complete
shared T-pose calibration must request `first_frame` explicitly; the manifest
records that choice. This mode uses only direct source quaternions and robot FK,
not positions, although an arbitrary first motion frame still has different
semantics from a geometric calibration pose.

The `shoulders_feet` ablation profile assigns the same requested weight to
`LeftArm`, `RightArm`, `LeftFoot`, and `RightFoot`. On E1 these target the two
shoulder-yaw bodies and the two ankle-roll bodies; all other orientation
weights remain zero.

The `full_equal` profile tracks all 15 configured links with exactly the same
requested weight. This differs from `full`, which retains nonuniform relative
coefficients for the legs, feet, forearms, and hands.

The `balanced_optimal` profile fixes every configured orientation weight at
`0.085`. It is the equal position-orientation balance selected by a parallel
coarse, grouped, full-sequence, and local search on the 2660-frame
`breaking+hippop.bvh_Skeleton1` sequence. See
[orientation_weight_search_report_zh.md](examples/orientation_weight_search_report_zh.md)
for the objective, Pareto analysis, and complete measurements.

Run the reproducible `breaking+hippop` baseline, grouped-link, and full-link
ablations with the `gmr_legs` profile mirroring the hip, knee, and foot chains
whose rotation costs are enabled by GMR's primary E1 task. The `shoulders`
profile only weights `LeftArm/RightArm → l/r_arm_shoulder_yaw_link`; all other
diagnostic links remain at exactly zero weight. GMR's numerical weights and
fixed quaternions are not copied because the MJCF frames, coordinate conversion,
and objective scales differ:

```bash
python examples/run_orientation_ablation.py \
  --robot e1 \
  --task-type robot_only \
  --data-format noetix_mocap \
  --data-path demo_data/noetix_mocap/0724_BEITI \
  --task-name 'breaking+hippop.bvh_Skeleton1' \
  --output-root demo_results/v1 \
  --variants baseline root feet gmr_legs shoulders upper full \
  --weight-scales 0.25 0.5 1.0 \
  --overwrite
```

Each run records a manifest and saves target/robot link quaternions, per-link
geodesic errors, mapped-position errors, and SQP diagnostics. The baseline
stores the same 15-link diagnostics with zero weights while adding no
orientation term to the optimization. Use `--frame-start` and `--frame-count`
for fast windowed tuning on the explicitly supported metadata-preserving NPZ
formats before confirming selected settings on the complete sequence. The
converted source frames can be inspected with:

```bash
python viser_player.py \
  --input-path demo_data/noetix_mocap/0724_BEITI/breaking+hippop.bvh_Skeleton1.npz \
  --input-kind raw \
  --show-source-orientation-axes
```

Compare target and actual shoulder frames on a shared timeline with:

```bash
python multi_viser_player.py \
  --qpos-npzs \
    demo_results/v1/ablations/orientation_ablation/baseline/e1/robot_only/noetix_mocap/0724_BEITI--path-sha256-<dataset-root-digest>/breaking%2Bhippop.bvh_Skeleton1/identity.npz \
    demo_results/v1/ablations/orientation_ablation/shoulders_x0.025/e1/robot_only/noetix_mocap/0724_BEITI--path-sha256-<dataset-root-digest>/breaking%2Bhippop.bvh_Skeleton1/identity.npz \
  --labels baseline shoulders_x0.025 \
  --x-offset 0.7 \
  --orientation-joints LeftArm RightArm \
  --show-target-orientation-axes \
  --show-robot-orientation-axes
```

Replace `<dataset-root-digest>` with the digest suffix in the generated
summary's `dataset_partition` field; the manifest beside each artifact is the
authoritative path record.

The player uses one shared human-reference layer plus one layer per robot
result. The shared charcoal layer reads the complete saved source-human
trajectory and topology; calibrated target frames remain anchored to the
corresponding mapped objective joints. Each robot result has separate mesh and
skeleton visibility controls; schema-v2 results draw the complete saved robot
link tree, and actual link-frame axes use their saved link positions as
origins. Robot layers use a colorblind-safe blue, vermilion, and bluish-green
palette in input order. Display-only redundant shoulder and hip crossbars may
be omitted for clarity, while `--x-offset` separates all layers. RGB arrows are
local X, Y, and Z. Use `--variants baseline shoulders --weight-scales 0.01
0.025 0.05` to tune only the shoulder orientation objective.

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
height, and writes all 22 global `wxyz` quaternions from the source local
axis-angle rotations with
`orientation_source="direct_local_rotation_fk"`.

## Inspect Raw Human Motion Before Retargeting

Use the unified raw-motion viewer to determine whether a contact or pose issue
already exists in the source data or is introduced later by normalization and
retargeting:

```bash
# OMOMO object interaction: complete 52-joint skeleton and moving source mesh
python viser_player.py \
  --input-path demo_data/OMOMO_new/sub3_largebox_003.pt \
  --input-kind raw

# Climbing: complete 53-joint skeleton and static source multi_boxes.obj
python viser_player.py \
  --input-path demo_data/climb/mocap_climb_seq_0 \
  --input-kind raw

# Robot-only source without a scene mesh
python viser_player.py \
  --input-path demo_data/lafan/dance1_subject1.npz \
  --input-kind raw
```

The viewer supports `omomo`, `mocap`, `lafan`, `amass`, `gvhmr`, and
`noetix_mocap`. It infers the data format from the selected file and infers the
task context from its contents and neighboring assets. A dataset directory can
also be passed together with `--sequence`, for example
`--input-path demo_data/OMOMO_new --sequence sub3_largebox_003`. Use an
explicit `--task-type robot_only` when inspecting an OMOMO file without its
object, and use `--source-mesh PATH` to override automatic mesh lookup.

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
| `H` | Toggle the complete human skeleton and keypoints |
| `R` | Toggle source joint orientation axes |
| `L` | Toggle joint-name labels |
| `O` | Toggle the source object or terrain mesh |

All single-motion input adapters use the same `Playback`, `Layers`, and
`Style` tabs. Unavailable layers remain visible but disabled with a reason,
which keeps result, raw-human, and converted-motion controls consistent.

## Check Visualizations of Saved Retargeting Results

`viser_player.py` is the unified single-motion viewer for result, raw-human,
and converted inputs. `multi_viser_player.py` is the unified synchronized
viewer for an explicit result set or a discovered augmentation family. Both
use the same schema-v2 loader, compatibility checks, and layer semantics, so
retargeting always saves the complete contract while visualization decides
which human/robot skeletons, meshes, keypoints, Interaction Mesh, constraint
state, and orientation axes to display.

Schema-v2 results store enough robot, data-format, object, FPS, topology, pose,
and qpos-layout metadata for the viewer to need only the result path:

```bash
# OMOMO object interaction
python viser_player.py \
  --input-path demo_results/v1/canonical/g1/object_interaction/omomo/OMOMO_new/sub3_largebox_003/identity.npz

# Climbing
python viser_player.py \
  --input-path demo_results/v1/canonical/g1/climbing/mocap/climb/mocap_climb_seq_0/identity.npz

# LAFAN
python viser_player.py \
  --input-path demo_results/v1/canonical/g1/robot_only/lafan/lafan/dance2_subject1/identity.npz

# Noetix-mocap
python viser_player.py \
  --input-path demo_results/v1/canonical/g1/robot_only/noetix_mocap/noetix_mocap/run_to_the_right/identity.npz
```

### Skeleton, Opacity, and Interaction Mesh

Source and target Interaction Mesh data are mandatory in every canonical
schema-v2 artifact written by the single, batch/enhancement, and ablation
interfaces. The shared configuration validator rejects attempts to disable
`save_interaction_mesh`. During live retargeting, use:

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
  --retargeter.interaction-mesh-mode both \
  --retargeter.interaction-mesh-edges cross
```

Replay all overlays with:

```bash
python viser_player.py \
  --input-path demo_results/v1/canonical/g1/object_interaction/omomo/OMOMO_new/sub3_largebox_003/identity.npz \
  --show-human-skeleton \
  --show-robot-skeleton \
  --show-object-keypoints \
  --show-interaction-mesh \
  --interaction-mesh-mode both \
  --interaction-mesh-edges cross \
  --robot-mesh-opacity 0.45 \
  --object-mesh-opacity 0.25
```

To compare an identity result and all five OMOMO augmentations in one
synchronized scene, pass any one of the six sibling variant files:

```bash
python multi_viser_player.py \
  --family demo_results/v1/canonical/g1/object_interaction/omomo/OMOMO_new/sub3_largebox_003/identity.npz
```

This viewer discovers the six siblings and color-codes each robot mesh,
complete saved robot link skeleton, and object mesh around one shared
source-human reference, with global layer controls and per-motion visibility
controls.
Object keypoints, Interaction Mesh, foot-sticking state, and orientation axes
use the same layer names as the single-motion viewer and remain off by default
when they would clutter a multi-motion scene. Before synchronization, the
shared loader validates every schema-v2 artifact and requires the same source
path and digest, dataset identity, robot/data/task identity, complete human and
robot topology, actuated joint order, and direct-human-orientation tensor
identity.

For OMOMO, `--show-human-skeleton` reads the saved complete 52-joint SMPL-H
trajectory rather than treating the mapped optimization subset as the complete
human skeleton. Optional hand-detail controls affect only diagnostic display;
they do not add joints to the Interaction Mesh or change optimization.
`--show-object-keypoints` draws the exact per-frame object points saved during
retargeting: red is the demo object normalized with the human height, and cyan
is the object at the target asset scale. Because the result stores the
world-space points actually used by that run, replay does not resample the
object surface.

`--mesh-opacity` provides a shared opacity. `--robot-mesh-opacity` and
`--object-mesh-opacity` override it independently. Skeleton point/line size,
object point size, and Interaction Mesh line width are controlled by
`--skeleton-point-radius`, `--skeleton-line-width`,
`--object-keypoint-radius`, and `--interaction-mesh-line-width`.

## Result NPZ Contract

Every schema-v2 artifact contains `qpos`, FPS, and solver diagnostics; the
complete preprocessed human skeleton in `human_joints`, `human_joint_names`,
and `human_joint_parent_indices`; the mapped human/robot objective keypoints;
and the complete robot subtree in `robot_link_positions`,
`robot_link_quaternions_wxyz`, `robot_link_names`, and
`robot_link_parent_indices`. Robot poses are exact per-frame FK values, not
reconstructed later by the viewer. Canonical `qpos` has dtype `float64`: its
rows are the exact accepted SQP states to which the saved cost, residuals, and
ConstraintMode refer, and augmented variants reuse those same-precision states
as nominal warm starts. Visualization and downstream conversion may cast a
loaded copy, but a schema-v2 writer may not cast canonical qpos to float32.
`robot_actuated_joint_names`, coordinate and quaternion conventions,
`qpos_layout`, `object_poses_demo`, `object_poses_target`,
`object_pose_layout`, source/config hashes, variant, run kind, and
preprocessing scale/provenance make the result self-describing. The normalized
`config_json` identity also fingerprints the solver implementation and runtime
plus the robot model tree and applicable height-table, object, or
climbing-scene assets, so resume does not treat a changed implementation or
static asset as the same job.

Source-human orientation is an optional all-or-none group:
`human_orientation_joint_names`,
`human_orientation_quaternions_wxyz`, and
`human_orientation_sha256`. It is present only for the directly observed named
subset, is accompanied by an approved `orientation_source`, and binds the
joint-name order plus exact float32 tensor bytes to its digest. The group is
absent and `orientation_source` is `absent` otherwise. Generic climb therefore
has no source-human orientation group, while canonical Noetix CSV climbing
does. Complete robot-link orientations and optional orientation-objective
diagnostic content remain independent: the schema always carries the
diagnostic field family, using zero-width arrays when no directly supported
joint/link targets are configured.

The source-human group is provenance, not an augmented optimization target.
Every identity and augmentation artifact preserves the adapter's exact named
float32 source tensor and digest. When an augmented object rotation changes an
orientation objective, the transformed solver targets are recorded separately
in `orientation_target_quaternions_wxyz`; they never overwrite
`human_orientation_quaternions_wxyz`.

Object-interaction results also contain `object_points_demo_local`,
`object_points_target_local`, `object_points_demo_world`, and
`object_points_target_world`, preserving the demo/target-scale samples and
their per-frame world coordinates. The mandatory Interaction Mesh group stores
per-frame source/target vertices, packed tetrahedra and counts, and the
human/object vertex split.

The solver evaluates every QP proposal on true MuJoCo geometry. If a full QP
step is infeasible while the incumbent is feasible, bisection along that
segment returns the largest known true-geometry-feasible step. An infeasible
incumbent may still advance to an infeasible proposal so the next SQP
linearization can recover. Constraint fallback is an outer mode schedule, not
a sequence of one-shot QP retries. For a frame with active foot sticking, the
solver gives `normal`, then an available `relaxed`, then an enabled `released`
foot mode a complete SQP run in that order while object non-penetration remains
active. Only if none has a feasible candidate and object release was explicitly
enabled does it repeat those foot modes with robot-object non-penetration
released. A less-strict mode restarts from the same frame-entry state. The
strictest mode that produces any true-geometry-feasible candidate wins; its
latest feasible candidate is the saved state.

An additional default policy handles one narrow initialization failure without
releasing any physical feasibility constraint. For an identity or ablation
`robot_only` run with no nominal trajectory and an optimized floating-base Z
coordinate, a structured frame-zero failure may be retried exactly once only
when ground non-penetration is the sole violated hard physical constraint and
the object is the horizontal ground plane; the structured triggering failure
must have neither a physical-constraint release nor trust-region release. The
retry recomputes true geometry from the original frame-entry state, raises only
root Z by the exact penetration correction plus the shared strict interior
margin, rechecks every hard constraint, and reruns the same SQP. During that
retried SQP, the existing frame-zero algorithmic trust-region release may still
be selected and is recorded separately; it does not release ground or another
physical constraint, and every candidate still passes the true-geometry gate.
The retry is never used for object interaction, climbing, augmentation,
non-horizontal terrain, or a mixed failure. A rejected or unsuccessful retry
remains a hard failure; an already successful path is unchanged.

Every schema-v2 artifact carries all eight audit fields, using zero-valued
numeric sentinels when no retry was triggered:
`frame_zero_ground_retry_policy`,
`frame_zero_ground_retry_eligible`,
`frame_zero_ground_retry_triggered`,
`frame_zero_ground_retry_initial_min_distance_m`,
`frame_zero_ground_retry_corrected_min_distance_m`,
`frame_zero_ground_retry_lift_m`,
`frame_zero_ground_retry_interior_margin_m`, and
`frame_zero_ground_retry_initial_sqp_iterations`. A triggered result also
prefixes frame zero's `sqp_stop_reasons` entry with
`frame_zero_ground_retry:`. Strict artifact validation cross-checks eligibility,
the one-axis distance equation, configured penetration tolerance, shared
margin, iteration count, and stop-reason provenance.

Six float64 arrays record non-negative excess beyond the configured tolerance
on the accepted state:
`ground_non_penetration_violation`,
`object_non_penetration_violation`, `foot_sticking_violation`,
`foot_lock_violation`, `self_collision_violation`, and
`joint_limits_violation`. Ground, foot-lock, self-collision, and joint-limit
excess are always hard; object excess is hard unless the accepted object mode
released it, and foot-sticking excess is hard in `normal` and `relaxed` modes.
Released constraints remain measured. The schema accepts hard excess only up
to the nonlinear acceptance tolerance.

The exact accepted mode is saved per frame in
`constraint_mode_foot_sticking` (`inactive`, `normal`, `relaxed`, or
`released`), `constraint_mode_object_non_penetration_released`, and
`constraint_mode_trust_region_released`. Trust-region release is an initial
linearization fallback and may be true only on frame zero. The compatibility
summaries `foot_sticking_fallback_frames`,
`foot_sticking_release_frames`, and
`object_non_penetration_release_frames` are derived exactly from these final
accepted modes; failed attempts never leak into them. If a genuinely
foot-related failure still triggers the configured full-sequence retry, the
saved trajectory has foot sticking disabled throughout and records the cause
in `foot_sticking_enabled_for_saved_trajectory` and
`foot_sticking_full_sequence_retry_frame`. `sqp_iteration_counts` is the total
number of linearized iterations across all modes attempted for that frame.
Any accepted object release remains an explicit degraded result rather than an
object-collision-valid frame.

Robot-only qpos uses `[root_xyz, root_wxyz, robot_dof]`. Dynamic-object qpos
appends `[object_xyz, object_wxyz]`; both layouts retain float64 canonical
precision.

## Quantitative Evaluation

The release gate for the canonical tree is the validator inside
`rebuild_demo_results.py`. It checks the exact planned artifact set, schema and
hash identity, frame counts, complete human and robot topology, direct
orientation provenance, Interaction Mesh, exact float64 accepted qpos, all six
nonlinear residual arrays, all three final ConstraintMode arrays, their derived
frame-list summaries, external asset closure, and configured release-quality
thresholds. Run it without solvers using:

```bash
python examples/rebuild_demo_results.py \
  --run-id demo-rebuild-v1 \
  --validate-only
```

The aggregate measurements and bounded validation issues are written to
`demo_results/runs/demo-rebuild-v1/report.json`; this report and the staged
validator are the acceptance gate for `demo_results/v1`.

## Prepare Data for Training RL Whole-Body Tracking Policy

The official workflow has two steps:

1. Run retargeting to obtain a `.npz` robot motion.
2. Convert that result to the whole-body tracking policy format at the desired
   frame rate.

On macOS, use `mjpython` instead of `python`.

### macOS (`mjpython`)

```bash
mjpython data_conversion/convert_data_format_mj.py \
  --input-file ./demo_results/v1/canonical/g1/robot_only/omomo/OMOMO_new/sub3_largebox_003/identity.npz \
  --output-fps 50 \
  --output-name converted_res/robot_only/sub3_largebox_003_mj_fps50.npz \
  --data-format omomo \
  --object-name ground \
  --once

mjpython data_conversion/convert_data_format_mj.py \
  --input-file ./demo_results/v1/canonical/g1/object_interaction/omomo/OMOMO_new/sub3_largebox_003/identity.npz \
  --output-fps 50 \
  --output-name converted_res/object_interaction/sub3_largebox_003_mj_w_obj.npz \
  --data-format omomo \
  --has-dynamic-object \
  --once
```

### Robot-Only Setting

```bash
python data_conversion/convert_data_format_mj.py \
  --input-file ./demo_results/v1/canonical/g1/robot_only/omomo/OMOMO_new/sub3_largebox_003/identity.npz \
  --output-fps 50 \
  --output-name converted_res/robot_only/sub3_largebox_003_mj_fps50.npz \
  --data-format omomo \
  --object-name ground \
  --once

python data_conversion/convert_data_format_mj.py \
  --input-file ./demo_results/v1/canonical/g1/robot_only/lafan/lafan/dance2_subject1/identity.npz \
  --output-fps 50 \
  --output-name converted_res/robot_only/dance2_subject1_mj_fps50.npz \
  --data-format lafan \
  --object-name ground \
  --once
```

### Robot-Object Setting

```bash
python data_conversion/convert_data_format_mj.py \
  --input-file ./demo_results/v1/canonical/e1/object_interaction/omomo/OMOMO_new/sub10_tripod_000/identity.npz \
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
