# FBX multi-actor motion retargeting

The FBX ingestion path is an offline normalization step followed by the
independently registered `fbx_mocap` adapter and the ordinary production
retargeting command. It shares the `HumanMotion` interface and result schema,
but keeps its robot mappings and orientation calibration separate from legacy
`mocap` so their source-frame conventions cannot be mixed.

Install the optional Assimp converter dependency with
`pip install -e 'src/holosoma_retargeting[fbx]'`. The converter pins Assimp
5.4.3 because it is responsible for evaluating the FBX transform stack,
including Euler rotation order, pivots, pre-rotation, and post-rotation.

Convert every top-level FBX file with the following command. Each Hips-rooted
skeleton becomes a separate actor-level NPZ motion.

```bash
python -m holosoma_retargeting.data_utils.convert_fbx \
  src/holosoma_retargeting/holosoma_retargeting/demo_data/fbx \
  src/holosoma_retargeting/holosoma_retargeting/demo_data/fbx_mocap
```

For the supplied files this writes `Data 2026-07-24
10-58-15__Skeleton0.npz`, `Data 2026-07-24 10-58-15__Skeleton1.npz`,
`Take_38_003_R__nan.npz`, and `Take_38_003_R__nv.npz`. Existing outputs are
protected unless `--overwrite` is supplied. `--assimp-executable` can select a
reviewed external Assimp binary when the optional package is not installed.

The converter samples the evaluated local FBX transforms on one common time
axis, performs FK, and converts centimetre Y-up positions to metre Z-up
positions. The supplied raw FBX declares +Y up and its bind pose establishes
+X as anatomical left and +Z as anatomical forward. The converter therefore
uses the proper right-handed rotation `[x, y, z] -> [x, -z, y]`: source forward
+Z becomes scene forward -Y, source left +X remains scene left +X, and source
up +Y becomes scene up +Z. The previous `[x, z, y]` axis swap had determinant
-1 and mirrored the motion; artifacts written with that transform must be
reconverted from the original FBX. The converter stores global unit WXYZ
orientations in the same corrected basis. It uses the same 53 canonical joint
labels as `mocap`, maps `LToeEnd` and `RToeEnd` to the two foot modifier points,
and retains the 51 real canonical joint frames as direct orientations for the
retargeting contract. In parallel, it preserves every named joint in the
Hips-rooted FBX tree, including head, toe, and finger end sites, with its exact
parent topology, global position, global frame, and bind frame. This complete
tree is visualization-only and does not add optimization anchors. Marker
rotations and synthetic bone-vector orientations are not admitted. The
resulting NPZ also records the source FBX path, actor, FBX version, raw joint
names, source and output FPS, coordinate transforms, bind-pose joint frames,
and direct orientation provenance.

The converter reads `UpAxis`, `UpAxisSign`, `FrontAxis`, `FrontAxisSign`,
`CoordAxis`, `CoordAxisSign`, and `UnitScaleFactor` directly from the binary
FBX `GlobalSettings`. It also verifies the named left/right hip joints and the
foot-to-toe bind vectors before applying the transform. The current adapter
rejects any FBX whose raw settings or anatomical bind axes differ from the
reviewed files instead of inferring a replacement facing direction.

Run one normalized actor with the shared production command. A nonzero
orientation weight enables link orientation tracking; leaving it unset keeps
the orientation objective off while preserving the source orientations in the
result artifact.

```bash
python src/holosoma_retargeting/holosoma_retargeting/examples/robot_retarget.py \
  --task robot_only \
  --robot g1 \
  --dataset fbx_mocap \
  --motion Take_38_003_R__nv \
  --retargeter.no-visualize
```

By default, the FBX path uses position targets only for G1, E1, and E2. Robot
root orientation is read from the direct FBX Hips rotation and composed with
the root-frame offset declared by the FBX coordinate-conversion metadata. No
position-based facing estimate is used. Optional link orientation tracking uses
an independent bind-pose-to-robot-T-pose calibration for every mapped link.

The G1 FBX entry in `examples/robot_profiles/g1.json` remains zero so orientation
tracking stays opt-in. Pass `--orientation_weights WEIGHT` to apply one positive
weight to every independently calibrated mapped link, or edit the selected
robot profile and enable its orientation section. Without either option,
source orientations remain available for calibration and visual inspection but
do not affect retargeting.

Pass `--orientation-preview` to compute every mapped link's independent
`inverse(source FBX bind frame) * robot T-pose FK frame` calibration and save
the raw source, aligned target, and actual robot frame trajectories. Preview
mode leaves orientation tracking and diagnostics disabled: it adds no SO(3)
objective, Jacobian, error angle, or frame cost. Display the three RGB frame
sets with `--show-source-orientation-axes --show-target-orientation-axes
--show-robot-orientation-axes` in `viser_player.py`. The result stores the
complete source-human and MuJoCo robot link trees. In the Viser Layers tab,
`Displayed link scope` switches between `retargeting` and `all`; the former
shows only joints participating in the human-to-robot map, while the latter
shows every saved source joint and robot link. The switch is a display filter
and never enables orientation tracking.
