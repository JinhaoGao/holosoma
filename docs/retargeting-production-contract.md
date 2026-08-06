# Retargeting production contract

This document freezes the production scope for
`feature/noetix-orientation-tracking`. The only comparison base is the local
`dev` commit `9f237c10d7e6fa99dba59707b7e43a79488430ff`. Updating from an
upstream or remote `dev` branch is explicitly outside this work.

## Public interfaces

The package exposes exactly two retargeting commands.

`robot_retarget.py` retargets one explicitly selected motion without dataset
discovery. It writes the identity result under `demo_results`.

`parallel_robot_retarget.py` retargets one explicitly selected motion and its
configured augmentation variants. Parallelism is an implementation detail; the
command must not walk a dataset directory. It writes results under
`demo_results_parallel`.

Ablation runners, weight searches, comparison batches, full-dataset traversal,
result rebuilding, and result promotion are not production interfaces. Both
retargeting commands share the same explicit foot-sticking switch and small
runtime visualization group. Live Viser visualization is enabled by default,
debug overlays are opt-in, and headless runs can disable the viewer. These
display-only settings do not alter the saved trajectory.
Saved-result visualization remains a post-processing operation.

## Supported task matrix

The human dataset presets are `climbing`, `gvhmr`, `lafan`,
`noetix_csv_climb`, `noetix_mocap`, and `OMOMO_new`.

| Task | G1 | E1 | E2 |
| --- | --- | --- | --- |
| `robot_only` | all six presets | all six presets | all six presets |
| `object_interaction` | `OMOMO_new` | unsupported | unsupported |
| `climbing` | `climbing`, `noetix_csv_climb` | unsupported | unsupported |

AMASS remains outside the public acceptance matrix. E1 and E2 scene assets for
object interaction or climbing are not required.

## Result contract

Every result preserves the Interaction Mesh and the exact human and robot
anchors used by position or orientation retargeting. Unused full-body and
full-link payloads are not saved.

Mapped robot link orientations and available source joint orientations are
stored for post-process axis visualization. Orientation payloads use explicit
frame and quaternion conventions and contain only solver mappings; hand
orientations are included only when a hand joint participates in retargeting.

The NPZ stores the effective human, robot, terrain, and object point
clouds after preprocessing and augmentation. Dynamic objects retain the data
needed to reproduce their world-space points. Missing terrain or object inputs
are represented as an absent optional layer rather than fabricated points.
Saving follows the simple `main` model: all available visualization data is
written directly without schema, hash, manifest, or cross-field artifact
validation. The viewer loads every available layer and exposes visibility
through its Layers tab.

## Orientation contract

Orientation loss is disabled by default. `--orientation_weights WEIGHT` assigns one
finite non-negative weight to every reviewed mapping. A robot-specific JSON
profile passed with `--orientation_config` instead contains complete tables for every orientation-capable public
dataset and assigns weights by human keypoint or robot-link alias. The current
dataset selects exactly one table; a zero weight disables that mapping's loss.
The two options are mutually exclusive. Source data without reliable
orientations remains valid for position-only retargeting and fails clearly if
orientation loss is explicitly requested.

Each orientation-capable dataset and robot pair has a T-pose calibration. A
fixed frame offset aligns the source joint frame with the robot link frame
before the loss is evaluated. The implementation must be validated by an
identity T-pose and known-axis rotations instead of relying on visual
inspection.

## Natural-posture contract

Natural-posture regularization is disabled by default. `--nature_weights
WEIGHT` loads the selected robot's fixed reference angles from
`examples/nature_weights` and assigns the same finite non-negative weight to
every actuated joint. `--nature_config` instead accepts a robot-matching JSON
file, or a directory containing the three corresponding files, with direct
per-joint references and absolute weights. The two options are mutually
exclusive, and zero removes a joint from the loss.

The SQP objective measures candidate joint angles against fixed natural-pose
references rather than the preceding frame. Active joints are initialized to
those references once before frame zero so ambiguous branches start from the
same arms-down pose; later frames continue from the previous solution.

## Foot-sticking contract

Foot-sticking XY hard constraints are enabled by default. The public syntax is
exactly `--foot-sticking True|False`; `False` prevents those constraints from
being added for the selected single-motion or augmentation run. It does not
disable ground non-penetration, joint limits, or the separately configured
frame-window Z foot lock. Use `--overwrite` when changing this setting for an
output path that already exists.

## Output layout

Identity results use:

`demo_results/<robot>/<task>/<dataset>/<motion>.npz`

Augmented results use:

`demo_results_parallel/<robot>/<task>/<dataset>/<motion>.npz` for identity and
`demo_results_parallel/<robot>/<task>/<dataset>/<motion>_<variant>.npz` for augmentations.

No other `demo_results_*` tree belongs to the production result set.

## Commit gates

Each implementation stage is committed independently. A stage commit must have
a focused diff, relevant tests, no accidentally generated result files, and no
raw E2 mesh payloads. The final branch must pass the retargeting test suite,
format and lint checks, `git diff --check`, and the repository CI type check.
The final audit also verifies the exact public scripts, task matrix, result
directory layout, default-off orientation behavior, and that this work created
no untracked files. It also verifies default-on foot sticking and explicit
`True`/`False` CLI parsing.
