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
result rebuilding, and result promotion are not production interfaces.
Visualization is a post-processing operation over saved results and is not a
retargeting mode.

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
anchors used by the solver. Human hand keypoints are additionally preserved
when the source provides them. Unused full-body and full-link payloads are not
part of the production artifact.

Mapped robot link orientations and available source joint orientations are
stored for post-process axis visualization. Orientation payloads use explicit
frame and quaternion conventions and contain only solver mappings; hand
orientations are included only when a hand joint participates in retargeting.

The artifact stores the effective human, robot, terrain, and object point
clouds after preprocessing and augmentation. Dynamic objects retain the data
needed to reproduce their world-space points. Missing terrain or object inputs
are represented as an absent optional layer rather than fabricated points.

## Orientation contract

Orientation loss is disabled by default. A single command switch enables equal
weights for every reviewed mapping, while an optional JSON configuration
assigns weights by human keypoint or robot link. An omitted or zero weight
disables a mapping. Source data without reliable orientations remains valid for
position-only retargeting and fails clearly if orientation loss is explicitly
requested.

Each orientation-capable dataset and robot pair has a T-pose calibration. A
fixed frame offset aligns the source joint frame with the robot link frame
before the loss is evaluated. The implementation must be validated by an
identity T-pose and known-axis rotations instead of relying on visual
inspection.

## Output layout

Identity results use:

`demo_results/<robot>/<task>/<dataset>/<motion>/identity.npz`

Augmented results use:

`demo_results_parallel/<robot>/<task>/<dataset>/<motion>/<variant>.npz`

No other `demo_results_*` tree belongs to the production result set.

## Commit gates

Each implementation stage is committed independently. A stage commit must have
a focused diff, relevant tests, no accidentally generated result files, and no
raw E2 mesh payloads. The final branch must pass the retargeting test suite,
format and lint checks, `git diff --check`, and the repository CI type check.
The final audit also verifies the exact public scripts, task matrix, result
directory layout, default-off orientation behavior, and that this work created
no untracked files.
