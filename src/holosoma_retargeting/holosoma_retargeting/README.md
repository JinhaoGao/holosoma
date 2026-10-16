# Holosoma Motion Retargeting

[中文](README_zh.md) | English

<!-- Maintenance: Keep this guide stable. Change it only with explicit owner approval and update both languages together. -->

Retarget human motion to humanoid robots for robot-only motion, object interaction, and climbing.

Supported data presets are OMOMO (`OMOMO_new`), LAFAN (`lafan`), AMASS SMPL-X (`amass`), GVHMR (`gvhmr`), FBX (`fbx_mocap`), Noetix BVH (`noetix_mocap`), climbing mocap (`climbing`), and Noetix climbing CSV (`noetix_csv_climb`). See Data Preparation below for the required inputs and conversions.

Select a robot with `--robot`: G1 (`g1`, 29 DoF), E1 (`e1_23dof` / `e1_24dof`), E2 (`e2`, 23 DoF), or M4G / M5C / M5D / M5E / M5 reverse-knee (`m4g` / `m5c` / `m5d` / `m5e` / `m5fanqu`, 17 DoF). `e1` is an alias for `e1_23dof`. These robots support `robot_only`; object interaction and climbing currently require G1.

Complete the [retargeting environment setup](../../../README.md#setup), activate that environment, then run the commands below from this directory. Replace `/path/to/...` and sample motion names with your own paths and files.

```bash
cd src/holosoma_retargeting/holosoma_retargeting
```

The converters prepare world-space human joint positions `(T, J, 3)` in metres, together with the format-specific metadata. Use the matching converter for raw data; an arbitrary NPZ with the same array shape is not a supported input.

## Single Sequence Motion Retargeting

Retarget the bundled OMOMO example to G1:

```bash
python examples/robot_retarget.py \
  --task robot_only --robot g1 \
  --dataset OMOMO_new --motion sub3_largebox_003
```

Change `--robot` to select another supported robot. `--motion` is the filename without its extension, including any subdirectory relative to the dataset root; quote names containing spaces. Results are written to `demo_results/<robot>/<task>/<dataset>/<motion>.npz`.

For one G1 object-interaction result, change the task to `object_interaction`; this task accepts only `OMOMO_new`. Climbing accepts `climbing` or `noetix_csv_climb`, as shown under Data Preparation.

### Optional settings

Append `--data-path /path/to/dataset` to use a different data root, `--save-dir /path/to/results` to change the result root, or `--output-name custom.npz` to change only the final filename. Existing results are reused; append `--overwrite` when you intend to regenerate them.

Live Viser visualization is enabled by default. Append `--retargeter.no-visualize` for a headless run, or `--retargeter.debug` for diagnostic layers and an Enter prompt after solving.

Robot defaults are loaded from [robot_profiles](examples/robot_profiles/). Explicit command-line options override the selected JSON. Use `--robot-profile /path/to/robot.json` to select another profile.

For motion tuning, `--foot-sticking False` disables planar foot-contact constraints; `--orientation-tracking True` enables source-orientation tracking; `--shoulder-direction-tracking True` tracks upper-arm directions. Source-orientation tracking requires orientation data and is unavailable for legacy position-only NPY files.

Use `--natural-pose-tracking True` to enable the profile’s natural-pose reference. `--root-position-weight VALUE` and `--root-orientation-weight VALUE` set non-negative weights for following the source root position and orientation.

Use `python examples/robot_retarget.py --help` for all options. These shared options also apply to the augmentation command below.

### Paired refinement

First retarget each actor independently with `robot_only`. The two results must have equal frame counts and FPS, matching source formats and coordinate conventions, and the saved human references. FBX actors must originate from the same FBX file. Then refine the pair:

```bash
python -m holosoma_retargeting.paired_retargeting.robot_refine \
  --actor-a.name actor_a --actor-a.result-path /path/to/actor_a.npz \
  --actor-b.name actor_b --actor-b.result-path /path/to/actor_b.npz \
  --output-path demo_results/paired_refined.npz
```

Optional: `--visualize` opens the result after refinement; `--contacts-path /path/to/contacts.json` adds specified contacts. See the [paired guide](paired_retargeting/README.md) for contact and collision settings.

## Motion Retargeting with Augmentation

Generate one G1 motion and its augmentation variants. The command processes the selected motion, not every motion in the dataset.

```bash
python examples/parallel_robot_retarget.py \
  --task object_interaction --robot g1 \
  --dataset OMOMO_new --motion sub3_largebox_003
```

Object interaction produces the identity trajectory plus translation and rotation variants. For climbing, use `--task climbing --dataset climbing --motion mocap_climb_seq_0`, or a prepared `noetix_csv_climb` motion; its variants scale the terrain height. The identity trajectory is generated first automatically.

Results are saved as `demo_results_parallel/<robot>/<task>/<dataset>/<motion>.npz` and `<motion>_<variant>.npz` in the same directory. The optional settings above apply here; augmentation is limited to G1 object-interaction and climbing tasks.

## Data Preparation

If the matching converted data is already available, skip its conversion step. Conversion commands may process a directory; retargeting commands select one motion.

### OMOMO

Use [InterMimic-processed OMOMO](https://drive.google.com/file/d/141YoPOd2DlJ4jhU2cpZO5VU5GzV_lm5j/view), as in the upstream guide. Extract the `.pt` files to `demo_data/OMOMO_new/`; the original OMOMO release has a different format. No additional human-motion conversion is required. Run the single-motion or augmentation example above with `--dataset OMOMO_new`.

### LAFAN

Download and extract [LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset/blob/master/lafan1/lafan1.zip). The converter reads the BVH files directly inside the input directory. Install the optional BVH reader once if `data_utils/lafan1/` is absent:

```bash
git clone https://github.com/ubisoft/ubisoft-laforge-animation-dataset.git data_utils/lafan_source
cp -r data_utils/lafan_source/lafan1 data_utils/lafan1
```

Convert the BVH files to named-joint NPZ files, then retarget one motion:

```bash
python -m holosoma_retargeting.data_utils.extract_global_positions \
  --input-dir /path/to/lafan --output-dir demo_data/lafan

python examples/robot_retarget.py \
  --task robot_only --robot e2 --dataset lafan --motion walk2_subject3
```

Optional: use `--overwrite` on conversion to replace old outputs. Reconvert position-only NPY data if you need source orientations.

### AMASS SMPL-X

Download SMPL-X sequences from [AMASS](https://amass.is.tue.mpg.de/) and the neutral model from [SMPL-X](https://smpl-x.is.tue.mpg.de/). Place the model at `models/smplx/SMPLX_NEUTRAL.npz`. Raw motions should follow `/path/to/amass/<dataset>/<subject>/*_stageii.npz`.

```bash
python -m holosoma_retargeting.data_utils.prep_amass_smplx_for_rt \
  --amass-root-folder /path/to/amass \
  --output-folder demo_data/amass_smplx_processed

python examples/robot_retarget.py \
  --task robot_only --robot g1 --dataset amass \
  --motion HumanEva_S3_Jog_1_stageii
```

Conversion searches recursively and joins the source relative path with underscores: `HumanEva/S3/Jog_1_stageii.npz` becomes `HumanEva_S3_Jog_1_stageii.npz`. Optional: `--model-root-folder /path/to/models`, `--subdataset-folder HumanEva`, or `--fps 30` on the conversion command; the requested FPS must not exceed the source FPS.

### GVHMR

Prepare a GVHMR `hmr4d_results.pt` containing world-space `smpl_params_global`, and the same SMPL-X neutral model used above. Convert one prediction before retargeting:

```bash
python -m holosoma_retargeting.data_utils.convert_gvhmr \
  --input-file /path/to/hmr4d_results.pt --output-file demo_data/gvhmr/tennis.npz

python examples/robot_retarget.py \
  --task robot_only --robot g1 --dataset gvhmr --motion tennis
```

Optional: `--model-path /path/to/models` selects the SMPL-X model. Set `--fps` to the prediction frame rate when it differs from 30; this option records timing and does not resample the prediction.

### FBX Motion Capture

Install the optional FBX dependency from this directory, then convert supported Hips-rooted FBX skeletons:

```bash
python -m pip install -e "..[fbx]"
python -m holosoma_retargeting.data_utils.convert_fbx \
  /path/to/fbx demo_data/fbx_mocap
```

Each actor is written as `<file>__<actor>.npz`. Use the exact output name printed by the converter; for example, `Take_38_003_R__nv.npz`:

```bash
python examples/robot_retarget.py \
  --task robot_only --robot e1_24dof --dataset fbx_mocap \
  --motion Take_38_003_R__nv
```

Optional: `--target-fps 30` sets the conversion sample rate. For M5 reverse-knee motion, replace `--robot e1_24dof` with `--robot m5fanqu`; its FBX profile preserves source knee directions and applies robot-segment scaling automatically. Keep both actors on the same sampling timeline for paired refinement. See [FBX details](../../../docs/fbx-mocap-retargeting.md) for the supported skeleton and coordinate conventions.

### Noetix BVH

Use the Noetix converter for supported company BVH exports. It requires the same BVH reader installed in the LAFAN section and preserves subdirectories in the output.

```bash
python -m holosoma_retargeting.data_utils.convert_noetix_bvh \
  --input-dir /path/to/noetix_bvh --output-dir demo_data/noetix_mocap

python examples/robot_retarget.py \
  --task robot_only --robot m5c --dataset noetix_mocap --motion walk
```

Here `walk` assumes the conversion produced `demo_data/noetix_mocap/walk.npz`; use `folder/walk` for a nested file. Optional: `--target-fps 30` on conversion, or another supported `--robot` on retargeting.

### Climbing Motion Capture

The bundled examples are under `demo_data/climb/<motion>/`. Keep the human joint-position NPY together with the scene XML, URDF and meshes in that directory. Run one example with:

```bash
python examples/robot_retarget.py \
  --task climbing --robot g1 --dataset climbing --motion mocap_climb_seq_0
```

Optional: replace the script with `examples/parallel_robot_retarget.py` to generate terrain-height variants. To use this human motion with another robot, select `--task robot_only` and the desired `--robot`. The legacy NPY contains positions only: add `--orientation-tracking False` when the robot profile enables orientation tracking, including the M4G and M5 profiles.

### Noetix Climbing CSV

This input is a Noetix human-motion capture CSV with bone data and climbing-scene information. It is different from the robot joint CSV exported after retargeting. Convert it into a complete motion and scene directory:

```bash
python -m holosoma_retargeting.data_utils.convert_noetix_csv \
  /path/to/actor_170/climb.csv \
  --output-root demo_data/noetix_csv_climb --task-name climb

python examples/robot_retarget.py \
  --task climbing --robot g1 --dataset noetix_csv_climb --motion climb
```

Keep the generated NPZ/NPY and scene assets together under `demo_data/noetix_csv_climb/climb/`. Optional conversion settings: `--human-height 1.70` supplies height in metres, `--target-fps 30` downsamples the capture, and `--scene-template-dir /path/to/template` selects scene templates. The default template is the bundled `demo_data/climb/mocap_climb_seq_0`. Use the parallel retargeting script for augmentation, or `--task robot_only` to retarget only the human motion.

## Check Visualizations of Saved Results

### Single motion

```bash
python viser_player.py \
  --input-path demo_results/g1/robot_only/OMOMO_new/sub3_largebox_003.npz
```

Robot-only generated NPZ motions use the same command; human layers are hidden when no human reference exists. If robot metadata is missing, add `--robot-type` or `--robot-urdf`. Robot CSV playback also requires a model:

```bash
python viser_player.py --input-path /path/to/motion.csv --robot-type e2
```

CSV rows must contain root `xyz` in metres, root quaternion `xyzw`, and joint values in URDF file order, with no header or index. Revolute joint angles use radians. Use `--robot-urdf /path/to/robot.urdf` for a specific model variant; it must match the CSV export model. CSV defaults to 30 FPS; set `--fps` for another rate.

### Multiple motions

Compare NPZ results using the same robot model on a shared timeline:

```bash
python multi_viser_player.py \
  --input-paths /path/to/motion_a.npz /path/to/motion_b.npz
```

To load one complete augmentation family, use `--family` instead of `--input-paths`:

```bash
python multi_viser_player.py \
  --family demo_results_parallel/g1/object_interaction/OMOMO_new/sub3_largebox_003.npz
```

### Paired result

```bash
python paired_retargeting/viser_player.py --input-path demo_results/paired_refined.npz
```

All three viewers accept `--port 8081` to choose a port, `--fps 30` to override playback rate, `--no-playing` to start paused, `--loop` to repeat, and `--dry-run` to validate without opening a server. FPS otherwise comes from the saved result; multi-motion inputs with different saved rates require an explicit `--fps`. Use the browser Playback controls to seek and the Layers tab to select overlays. The single-motion entry point has no `--start-frame` option.

### Export Robot Motion to CSV

Export one NPZ using its matching URDF. The CSV contains only the robot trajectory, without human references, object poses or FPS:

```bash
python -m holosoma_retargeting.data_utils.npz_to_csv \
  demo_results/g1/robot_only/OMOMO_new/sub3_largebox_003.npz models/g1/g1_29dof.urdf
```

By default, the CSV is written beside the NPZ; use `--output-path /path/to/motion.csv` to change it. To convert a directory of results sharing the same robot model:

```bash
python -m holosoma_retargeting.data_utils.batch_npz_to_csv \
  demo_results/g1/robot_only/OMOMO_new models/g1/g1_29dof.urdf \
  demo_results_csv/g1/robot_only/OMOMO_new
```

## Quantitative Evaluation

Evaluate saved retargeting NPZ results with the human reference and solver metadata retained:

```bash
python evaluation/eval_retargeting.py \
  --res-dir demo_results/g1/robot_only/OMOMO_new \
  --data-type robot_only --robot g1 --data-format omomo
```

For object interaction, select its result directory and `--data-type robot_object --data-format omomo`. For climbing, use `--data-type robot_terrain --data-format mocap`. Optional: `--max-workers 4` evaluates multiple results concurrently. Here `--data-format` names the internal format; OMOMO uses `omomo`, while the retargeting preset is `OMOMO_new`.

## Prepare Data for Training RL Whole-Body Tracking Policies

Convert a retargeted NPZ into body poses and velocities for training. The following G1 example opens MuJoCo and requires a graphical session; on macOS, use `mjpython` instead of `python`.

```bash
python data_conversion/convert_data_format_mj.py \
  --input-file demo_results/g1/robot_only/OMOMO_new/sub3_largebox_003.npz \
  --output-name converted_res/g1/sub3_largebox_003.npz \
  --robot g1 --data-format omomo --once
```

Optional: `--output-fps 50` sets the training sample rate. For a dynamic-object result, also pass `--has-dynamic-object --object-name largebox` with its corresponding input file. Use `--use-omniretarget-data` only for the external OmniRetarget qpos layout. This conversion tool has built-in joint tables for `g1`, `e1` and `e2`; other variants require `--joint-names` and a verified match between input joint order and robot assets.

## Custom Human Motion Data Format

See [Adding a human motion format](ADD_MOTION_FORMAT_README.md) for the loader and joint-mapping requirements.

## Custom Robot Type

See [Adding a robot](ADD_ROBOT_TYPE_README.md) for model preparation and mappings, and [robot profiles](examples/robot_profiles/README.md) for persistent retargeting settings.
