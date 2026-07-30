# Adding a human motion format

A new format must integrate with the shared motion adapter, production dataset
preset, robot mappings, compact result artifact, and visualization loader. It
must not add another retargeting command.

## Adapter contract

Register discovery and loading in `data_utils/motion_data.py`. A `HumanMotion`
must provide finite frame-major global keypoint positions, stable canonical
joint names, FPS, human height, source path, and parent indices. Convert units
and axes to the repository's Z-up metric scene before the solver sees them.

If the source contains directly observed joint orientations, additionally
provide `orientation_joint_names`, unit wxyz
`orientation_quaternions_wxyz`, an approved `orientation_source`, and explicit
coordinate-system metadata. Never label an orientation inferred from positions
or bone vectors as direct source data. Position-only formats omit the entire
orientation group.

Register the canonical skeleton, topology, hand naming, and position mappings
for G1, E1, and E2 in `config_types/data_type.py`. Every format admitted to the
public `robot_only` matrix must load and map on all three robots.

## Orientation calibration

For direct-orientation data, define independent human-joint-to-robot-link
mappings for every supported robot. The reviewed semantic set contains root
and bilateral hip, knee, ankle, toe, shoulder, elbow, and hand frames. Position
and orientation mappings may target different links when their frame semantics
differ.

Register the source T-pose joint frames, robot T-pose base frame, and robot
T-pose joint positions. Runtime calibration derives a fixed frame offset from
the source T-pose and robot FK T-pose. Tests must cover T-pose identity, a
known-axis rotation, link existence on G1/E1/E2, and default-off loss behavior.

## Production preset

Add an explicit preset to `DatasetName`, `DATASET_DATA_FORMATS`,
`DATASET_DEFAULT_PATHS`, and `SUPPORTED_TASK_DATASETS` in
`config_types/retargeting.py`. Production commands always select one motion and
must not add dataset traversal.

```bash
python examples/robot_retarget.py \
  --task robot_only \
  --robot e2 \
  --dataset <preset> \
  --motion <one-motion>
```

Only a reviewed G1 object-interaction or climbing dataset may use the
augmentation command:

```bash
python examples/parallel_robot_retarget.py \
  --task <object_interaction-or-climbing> \
  --robot g1 \
  --dataset <preset> \
  --motion <one-motion>
```

The shared writer compacts each artifact to solver keypoints, available hand
points, mapped link poses, effective point clouds, and Interaction Mesh data.
Adapters must supply enough information for those fields but must not introduce
a separate schema or output tree.

Add tests for discovery, successful and invalid loads, units and coordinate
frames, joint order and topology, G1/E1/E2 position mappings, direct orientation
provenance, T-pose offsets, compact artifact round trips, and visualization
metadata. Run the production tests, Ruff check, Ruff format check, and
`git diff --check` before committing.
