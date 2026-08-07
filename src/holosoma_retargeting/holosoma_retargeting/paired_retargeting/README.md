# Paired robot trajectory refinement

Paired refinement is a second-stage optimizer for two synchronized `robot_only` results. Each actor is first solved by the existing single-robot retargeter. The paired command then treats those two qpos trajectories as both its initialization and its nominal reference, attaches the original robot MJCF models to one namespaced MuJoCo world, and optimizes only the additional cross-actor relationships. It does not import the single-robot pipeline or the `InteractionMeshRetargeter`, so changes to either solver remain isolated.

The input NPZ files must have the same frame count, fps, source format, quaternion convention, and world coordinate system. FBX actors must come from the same `source_fbx` and have distinct `source_actor` values. The loader also checks that each result has a robot-only qpos layout, no object coordinates, the expected qpos width for its robot type, mapped human positions, mapped robot link names, and the exact saved actuated-joint order. Different robot variants can be paired; for example, an E1 23DOF result and an E1 24DOF result compile into a 61-coordinate joint scene without modifying either XML file.

The FBX converter recenters each actor independently but saves the removed common-scene origin as `source_xy_origin_m`. Before refinement, the paired solver restores `actor_b_origin - actor_a_origin` with one shared scene scale and applies the same translation to actor B's nominal robot root, mapped source points, and complete human skeleton. Actor A remains the world anchor, so the arbitrary absolute FBX origin does not move the whole scene. The default scene scale is the mean of both saved `human_position_scale` values and can be overridden with `--refinement.source-alignment.scene-scale`. Future single-robot results copy this source-scene metadata directly; older results recover it through their saved `source_path`.

The default refinement combines cross-actor Interaction Mesh Laplacian tracking with a per-frame source-human relative root-position target and a nominal relative root-yaw target. The source position anchor remains compatible with legacy mappings that include the robot root body and also accepts one explicit mapped torso joint named `Root`, `Hips`, `Pelvis`, or `Spine`, including `Spine -> waist_roll_link`. A strong nominal cost and a nominal-delta temporal cost protect the independently solved motion quality. With every cross-actor term disabled, the implementation bypasses the QP and returns the shared-world-aligned nominal trajectories bit-for-bit. Joint limits and separate trust-region bounds for root translation, root quaternion, and actuated joints apply to every solved frame.

The command can be run from the repository with the following form. Actor names become MuJoCo namespace prefixes and therefore must be unique Python identifiers.

```bash
PYTHONPATH=src/holosoma_retargeting \
python -m holosoma_retargeting.paired_retargeting.robot_refine \
  --actor-a.name actor_a \
  --actor-a.result-path /path/to/Skeleton0.npz \
  --actor-b.name actor_b \
  --actor-b.result-path /path/to/Skeleton1.npz \
  --output-path /path/to/paired_refined.npz
```

Explicit link contacts are supplied through `--contacts-path`. The file is a JSON list; every link pair preserves `actor_b_link - actor_a_link` in world coordinates, and every frame window is half-open. A finite `tolerance` adds hard component-wise bounds, while `weight` adds a soft quadratic objective.

```json
[
  {
    "actor_a_link": "l_hand_sphere_link",
    "actor_b_link": "r_hand_sphere_link",
    "target_offset": [0.0, 0.0, 0.0],
    "weight": 50.0,
    "tolerance": 0.01,
    "windows": [{"start": 120, "stop": 180}]
  }
]
```

Cross-actor collision is implemented as a signed-distance hard constraint and is intentionally opt-in because independently retargeted baselines can already overlap deeply. It can be enabled for all collidable cross-actor geometry with `--refinement.collision.enabled`, or restricted to unprefixed body-name pairs with `--refinement.collision.body-pairs`. Programmatic users can additionally restrict collision to half-open frame windows through `InterActorCollisionConfig.windows`. A baseline that penetrates farther than the root and joint trust regions can repair in one SQP step is infeasible by construction, so collision windows should begin before contact or be paired with a feasible initial placement.

The paired NPZ stores both refined qpos arrays, actor robot types and joint names, restored source-human skeletons, mapped robot trajectories, source-alignment evidence, fps and coordinate conventions, per-frame SQP diagnostics, contact and collision errors, source and target cross-actor vertices, and padded per-frame tetrahedra and edge topology. Passing `--visualize` starts the shared Viser playback interface after saving; `--loop-visualization` keeps that playback looping. The paired player uses the same playback controls, canonical layer panel, URDF renderer, skeleton styling, and Interaction Mesh colors as the existing visualization scripts, while deliberately applying no comparison-layout offset.

An already saved paired result can be inspected without rerunning refinement. The Viser URL is printed in the terminal, and the browser scene contains independently toggled robot meshes, both restored source-human skeletons, both refined mapped robot skeletons, and the paired source/target Interaction Mesh. For FBX results, the player also follows each saved single-robot result back to its converted actor motion and adds an `Original FBX skeletons` layer. Those two complete source trees remain at native metre scale and receive only one common translation based on a shared ground and the first actor's saved FBX origin. Their body sizes, separation, and height difference are therefore unscaled, while the existing `Human skeleton` layer retains the separately preprocessed robot-scale references. Both original skeletons are rendered in black. `--original-source-fbx` can explicitly validate the expected raw container, while `--original-start-frame` aligns a paired artifact created from a programmatically sliced source window.

```bash
PYTHONPATH=src/holosoma_retargeting \
python -m holosoma_retargeting.paired_retargeting.viser_player \
  --qpos-npz /path/to/paired_refined.npz \
  --original-source-fbx /path/to/source.fbx \
  --loop
```
