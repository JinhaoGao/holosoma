# ruff: noqa: CPY001, PLR0917

"""
Evaluation script for retargeting trajectories.
Evaluates:
1) Penetration depth & time duration
2) Contact precision (keypoints <=2cm from object/terrain surface)
3) Foot sliding
"""

from __future__ import annotations

import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Literal, Sequence

import igl  # type: ignore[import-not-found]
import mujoco  # type: ignore[import-not-found]
import numpy as np
import tyro

src_root = Path(__file__).resolve().parents[2]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))
from holosoma_retargeting.config_types.data_type import MotionDataConfig, normalize_data_format  # noqa: E402
from holosoma_retargeting.config_types.robot import RobotConfig  # noqa: E402
from holosoma_retargeting.data_utils.object_assets import (  # noqa: E402
    create_omomo_object_scene,
    default_generated_assets_root,
    get_omomo_object_asset,
)
from holosoma_retargeting.data_utils.omomo import OMOMO_OBJECT_NAMES  # noqa: E402
from holosoma_retargeting.src.mujoco_utils import _world_mesh_from_geom  # type: ignore[import-not-found]  # noqa: E402
from holosoma_retargeting.src.utils import (  # type: ignore[import-not-found]  # noqa: E402
    create_new_scene_xml_file,
    create_scaled_multi_boxes_xml,
    transform_points_world_to_local,
)
from holosoma_retargeting.visualization.result_loader import (  # noqa: E402
    VariantResult,
    load_variant_result,
)


def create_task_constants(
    robot_config: RobotConfig,
    motion_data_config: MotionDataConfig,
    *,
    object_name: str | None = None,
    object_dir: str | None = None,
    object_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    saved_object_urdf: str | None = None,
    saved_scene_xml: str | None = None,
    generated_assets_dir: str | Path | None = None,
) -> SimpleNamespace:
    """Create a mutable namespace that mimics the old constants modules."""
    namespace = SimpleNamespace()

    # Copy UPPER_CASE attributes from robot config
    for attr in dir(robot_config):
        if attr.isupper() and not attr.startswith("_"):
            setattr(namespace, attr, getattr(robot_config, attr))

    # Copy legacy constants from motion data config
    for attr, value in motion_data_config.legacy_constants().items():
        setattr(namespace, attr, value)
    namespace.ROBOT_TYPE = robot_config.robot_type
    namespace.SOURCE_DATA_FORMAT = motion_data_config.data_format

    # Override or supplement object information if requested
    if object_name is not None:
        namespace.OBJECT_NAME = object_name

    # Provide catalog-backed OMOMO assets and generated robot-object scenes.
    if namespace.OBJECT_NAME in OMOMO_OBJECT_NAMES:
        if generated_assets_dir is None:
            generated_assets_dir = (
                default_generated_assets_root() / "downstream" / robot_config.robot_type / namespace.OBJECT_NAME
            )
        object_asset = get_omomo_object_asset(namespace.OBJECT_NAME)
        robot_xml_path = Path(namespace.ROBOT_URDF_FILE).with_suffix(".xml")
        if not robot_xml_path.is_absolute():
            robot_xml_path = Path(__file__).resolve().parents[1] / robot_xml_path
        namespace.OBJECT_URDF_FILE = saved_object_urdf or str(object_asset.urdf_path)
        namespace.OBJECT_MESH_FILE = str(object_asset.mesh_path)
        namespace.OBJECT_URDF_TEMPLATE = str(
            object_asset.mesh_path.parent.parent / "templates" / "omomo_object.urdf.jinja"
        )
        namespace.SCENE_XML_FILE = saved_scene_xml or str(
            create_omomo_object_scene(
                robot_xml_path,
                namespace.OBJECT_NAME,
                scale=object_scale,
                output_dir=generated_assets_dir,
            )
        )
    elif namespace.OBJECT_NAME != "ground":
        namespace.OBJECT_URDF_FILE = f"models/{namespace.OBJECT_NAME}/{namespace.OBJECT_NAME}.urdf"
        namespace.OBJECT_MESH_FILE = f"models/{namespace.OBJECT_NAME}/{namespace.OBJECT_NAME}.obj"
        namespace.OBJECT_URDF_TEMPLATE = f"models/templates/{namespace.OBJECT_NAME}.urdf.jinja"
        namespace.SCENE_XML_FILE = (
            f"models/{robot_config.robot_type}/"
            f"{robot_config.robot_type}_{namespace.ROBOT_DOF}dof_w_{namespace.OBJECT_NAME}.xml"
        )
    else:
        namespace.SCENE_XML_FILE = namespace.ROBOT_URDF_FILE.replace(".urdf", ".xml")

    if object_dir is not None:
        namespace.OBJECT_DIR = object_dir
        namespace.OBJECT_URDF_FILE = saved_object_urdf or f"{object_dir}/{namespace.OBJECT_NAME}.urdf"
        namespace.OBJECT_MESH_FILE = f"{object_dir}/{namespace.OBJECT_NAME}.obj"
    if saved_scene_xml is not None:
        namespace.SCENE_XML_FILE = saved_scene_xml

    return namespace


class RetargetingEvaluator:
    """Evaluates retargeting trajectories against quality metrics."""

    def __init__(
        self,
        robot_model_path: str,
        object_name: str,
        demo_joints: List[str],
        joints_mapping: Dict[str, str],
        constants: SimpleNamespace | None = None,
    ):
        """Initialize evaluator with robot and object models."""
        if constants is None:
            raise ValueError("constants must be provided")

        self.object_name = object_name
        self.demo_joints = demo_joints
        self.joints_mapping = joints_mapping

        if self.object_name == "multi_boxes":
            self.collision_detection_threshold = 0.1
            self.penetration_tolerance = 0.01
            self.contact_threshold = 0.1
        else:
            self.collision_detection_threshold = 0.1
            self.penetration_tolerance = 0.01
            self.contact_threshold = 0.02

        # Foot sliding threshold (velocity in m/s)
        self.sliding_threshold = 0.01

        # Load Mujoco model
        if self.object_name == "ground":
            robot_xml_path = robot_model_path.replace(".urdf", ".xml")
        elif getattr(constants, "SCENE_XML_FILE", ""):
            robot_xml_path = constants.SCENE_XML_FILE  # type: ignore[attr-defined]
        else:
            robot_xml_path = robot_model_path.replace(".urdf", "_w_" + self.object_name + ".xml")

        self.robot_model = mujoco.MjModel.from_xml_path(robot_xml_path)
        print("Loading robot model from: ", robot_xml_path)

        self.robot_data = mujoco.MjData(self.robot_model)

        if self.robot_data.qpos.shape[0] > 7 + constants.ROBOT_DOF:
            self.has_dynamic_object = True
        else:
            self.has_dynamic_object = False

        # For climbing, bake the static terrain from the exact MuJoCo scene
        # selected from the artifact's saved configuration.
        self._obj_VW = np.zeros((0, 3), dtype=np.float64)
        self._obj_FW = np.zeros((0, 3), dtype=np.int32)
        self._have_terrain_mesh = False
        self._ground_z = 0.0  # ground z is always 0.0

        if self.object_name != "ground" and not self.has_dynamic_object and getattr(constants, "SCENE_XML_FILE", ""):
            self._have_terrain_mesh = True

        if self._have_terrain_mesh:
            self._bake_object_mesh_from_xml()
            self._have_terrain_mesh = bool(self._obj_VW.size and self._obj_FW.size)

        self.constants = constants

    def _bake_object_mesh_from_xml(self):
        """Bake the visual object mesh in world coordinates."""
        m, d = self.robot_model, self.robot_data
        mujoco.mj_forward(m, d)

        obj_Vs, obj_Fs, v_acc = [], [], 0
        visual_name = f"{self.object_name}_visual"
        geom_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid) or "" for gid in range(m.ngeom)]
        has_separate_visual = visual_name in geom_names
        for gid in range(m.ngeom):
            if m.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
                continue  # mesh-only
            name = geom_names[gid]
            if has_separate_visual:
                if name != visual_name:
                    continue
            elif self.object_name not in name:
                continue
            Vw, F = _world_mesh_from_geom(m, d, gid, name)  # your helper
            if Vw is None or F is None or Vw.size == 0 or F.size == 0:
                continue
            obj_Vs.append(Vw.astype(np.float64))
            obj_Fs.append(F.astype(np.int32) + v_acc)
            v_acc += Vw.shape[0]

        self._obj_VW = np.vstack(obj_Vs) if obj_Vs else np.zeros((0, 3), np.float64)
        self._obj_FW = np.vstack(obj_Fs) if obj_Fs else np.zeros((0, 3), np.int32)

    def _get_robot_link_positions(self, q, link_names):
        """Get robot link positions for given configuration using Mujoco.

        Assumes q is in MuJoCo order:
        - [0:3] robot base position (xyz)
        - [3:7] robot base quaternion (wxyz)
        - [7:7+R] robot joints
        - [-7:-4] object position (xyz) if has_dynamic_object
        - [-4:] object quaternion (wxyz) if has_dynamic_object
        """
        self.robot_data.qpos[:] = q
        # Forward kinematics to update all positions
        mujoco.mj_forward(self.robot_model, self.robot_data)

        robot_link_positions = []
        for link_name in link_names:
            # Get body ID from name
            body_id = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, link_name)
            if body_id == -1:
                raise ValueError(f"Body {link_name} not found in Mujoco model")

            # Get position in world frame
            # xpos gives us the position of the body's center of mass in world coordinates
            pos = self.robot_data.xpos[body_id].copy()
            robot_link_positions.append(pos)

        return np.array(robot_link_positions)

    def _prefilter_pairs_with_mj_collision(self, threshold: float):
        m, d = self.robot_model, self.robot_data
        ngeom = m.ngeom

        self._geom_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(ngeom)]

        if not hasattr(self, "_saved_margins"):
            self._saved_margins = np.empty_like(m.geom_margin)
        self._saved_margins[:] = m.geom_margin

        m.geom_margin[:] = threshold
        mujoco.mj_collision(m, d)

        candidates = set()
        for k in range(d.ncon):
            c = d.contact[k]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 < 0 or g2 < 0:
                continue
            candidates.add((min(g1, g2), max(g1, g2)))

        # Restore margins to keep physics untouched
        m.geom_margin[:] = self._saved_margins

        return candidates

    def evaluate_penetration(self, q_retarget: np.ndarray):
        """
        MuJoCo version of evaluate_penetration_old using your prefilter and distance calls.

        Returns:
            (fraction_with_penetration, penetration_max_depths)
            - fraction_with_penetration: float in [0,1]
            - penetration_max_depths: list[float], maximum penetration depth per penetrating frame
        """
        m, d = self.robot_model, self.robot_data

        penetration_max_depths = []
        penetration_frames = []

        # helper for name checks (populated by _prefilter_pairs_with_mj_collision)
        def _is_obj(g):
            return self.object_name in self._geom_names[g]

        def _is_ground(g):
            return "ground" in self._geom_names[g]

        def masks_ok(g1, g2):
            # skip geoms with both masks off
            if m.geom_contype[g1] == 0 and m.geom_conaffinity[g1] == 0:
                return False
            if m.geom_contype[g2] == 0 and m.geom_conaffinity[g2] == 0:
                return False
            # exclude object-ground specifically (either order)
            if (_is_obj(g1) and _is_ground(g2)) or (_is_obj(g2) and _is_ground(g1)):
                return False
            # keep only pairs that involve ground or object
            return _is_obj(g1) or _is_obj(g2) or _is_ground(g1) or _is_ground(g2)

        fromto = np.zeros(6, dtype=float)

        for i, q in enumerate(q_retarget):
            d.qpos[:] = q
            mujoco.mj_forward(m, d)  # compute kinematics, aabbs, etc.

            # 1) collect near pairs with temporary margins (also populates _geom_names)
            candidates = self._prefilter_pairs_with_mj_collision(self.collision_detection_threshold)

            # 2) precise distance on candidates; count only strict penetrations
            depths_this_frame = []
            for g1, g2 in candidates:
                if not masks_ok(g1, g2):
                    continue
                fromto[:] = 0.0
                dist = mujoco.mj_geomDistance(m, d, g1, g2, self.collision_detection_threshold, fromto)
                # penetration = negative signed distance
                if dist < -self.penetration_tolerance:
                    depths_this_frame.append(-float(dist))

            if depths_this_frame:
                penetration_frames.append(i)
                penetration_max_depths.append(float(np.max(depths_this_frame)))

        frac = len(penetration_frames) / max(len(q_retarget), 1)
        return frac, penetration_max_depths

    def detect_demo_contact(
        self,
        human_joints,
        joint_names: Sequence[str] | None = None,
    ):
        contact: dict[str, np.ndarray] = {}
        have_obj = self._obj_VW.shape[0] > 0
        if not have_obj:
            return contact  # no object mesh baked

        if joint_names is None:
            joint_names = (
                "LeftHandMiddle3",
                "RightHandMiddle3",
                "LeftFoot",
                "RightFoot",
                "LeftToeBase",
                "RightToeBase",
            )

        for jn in joint_names:
            if jn not in self.demo_joints:
                continue
            p = human_joints[self.demo_joints.index(jn)].reshape(1, 3).astype(np.float64)
            S, _, _, _ = igl.signed_distance(p, self._obj_VW, self._obj_FW)
            if S[0] <= self.contact_threshold:  # e.g., 0.02 for 2 cm
                contact[jn] = p.flatten()

        return contact

    def evaluate_contact_precision(
        self,
        human_joints_motion,
        object_poses,
        q_trajectory,
        joint_names: Sequence[str] | None = None,
    ):
        """
        Evaluate contact precision for keypoints within 2cm of surfaces.

        Args:
            q_trajectory: Robot joint configurations (N, DOF)
            object_poses: Object poses (N, 7)
            contact_sequences: Contact information per frame

        Returns:
            dict: Contact precision metrics
        """
        if joint_names is None:
            joint_names = ("L_Wrist", "R_Wrist")

        demo_local_points_list: list[np.ndarray] = []
        robot_local_points_list: list[np.ndarray] = []

        robot_joint_names = [self.joints_mapping[joint_name] for joint_name in joint_names]

        for q, human_joints, object_pose in zip(q_trajectory, human_joints_motion, object_poses):
            demo_points = np.array([human_joints[self.demo_joints.index(joint_name)] for joint_name in joint_names])
            demo_local_points_list.append(
                transform_points_world_to_local(
                    object_pose[3:7],
                    object_pose[:3],
                    demo_points,
                )
            )
            robot_joint_pos = self._get_robot_link_positions(q, robot_joint_names)
            # Object pose in MuJoCo order: [-7:-4] pos, [-4:] quat
            robot_local_points_list.append(transform_points_world_to_local(q[-4:], q[-7:-4], robot_joint_pos))
        demo_local_points = np.array(demo_local_points_list)
        robot_local_points = np.array(robot_local_points_list)

        demo_contact = np.linalg.norm(demo_local_points, axis=-1) <= 0.28
        robot_contact = np.linalg.norm(robot_local_points, axis=-1) <= 0.28

        miss_contact = demo_contact & (demo_contact != robot_contact)
        worst_miss_contact = np.logical_or.reduce(miss_contact, axis=1)

        return 1 - np.sum(worst_miss_contact) / max(len(q_trajectory), 1)

    def detect_foot_sliding(
        self,
        q_trajectory: np.ndarray,
        contact_states: np.ndarray,
        toe_names: Sequence[str],
        *,
        fps: float,
    ) -> tuple[float, np.ndarray]:
        """
        Detect foot sliding during contact phases.

        Args:
            q_trajectory: Robot joint configurations (N, DOF)
            contact_states: Saved left/right sticking state with shape (N, 2)
            toe_names: Source toe names in canonical left/right order
            fps: Saved trajectory rate used to convert displacement to m/s

        Returns:
            Sliding-frame fraction and maximum sliding velocity per sliding frame
        """

        q_trajectory = np.asarray(q_trajectory)
        contact_states = np.asarray(contact_states, dtype=bool)
        if q_trajectory.ndim != 2 or q_trajectory.shape[0] == 0:
            raise ValueError("q_trajectory must be a non-empty 2-D array")
        if contact_states.shape != (q_trajectory.shape[0], 2):
            raise ValueError(
                f"contact_states shape {contact_states.shape} != ({q_trajectory.shape[0]}, 2)",
            )
        if len(toe_names) != 2:
            raise ValueError("toe_names must contain canonical left/right names")
        if not np.isfinite(fps) or fps <= 0.0:
            raise ValueError(f"fps must be positive and finite, got {fps}")

        robot_toe_links = [self.joints_mapping[toe_name] for toe_name in toe_names]
        toe_positions_by_frame = [self._get_robot_link_positions(q, robot_toe_links) for q in q_trajectory]
        toe_positions = np.asarray(toe_positions_by_frame, dtype=np.float64)
        toe_xy_velocities = np.zeros((q_trajectory.shape[0], 2), dtype=np.float64)
        toe_xy_velocities[1:] = np.linalg.norm(
            np.diff(toe_positions[:, :, :2], axis=0),
            axis=2,
        ) * float(fps)
        sticking_frames = np.any(contact_states, axis=1)
        num_sticking_frames = int(np.count_nonzero(sticking_frames))
        if num_sticking_frames == 0:
            return 0.0, np.empty((0,), dtype=np.float64)

        sliding_by_foot = contact_states & (toe_xy_velocities > self.sliding_threshold)
        sliding_frames = np.any(sliding_by_foot, axis=1)
        max_sliding_velocities = np.max(
            np.where(sliding_by_foot, toe_xy_velocities, 0.0),
            axis=1,
        )[sliding_frames]
        return (
            float(np.count_nonzero(sliding_frames) / num_sticking_frames),
            max_sliding_velocities,
        )

    def _saved_foot_sliding(
        self,
        result: VariantResult,
    ) -> tuple[float, np.ndarray]:
        if result.foot_sticking is None:
            raise ValueError(
                f"{result.path} is missing saved foot-sticking metadata",
            )
        toe_names = MotionDataConfig(
            data_format=result.source_data_format,
            robot_type=result.robot_type,
        ).toe_names
        return self.detect_foot_sliding(
            result.qpos,
            np.asarray(result.foot_sticking["states"], dtype=bool),
            toe_names,
            fps=result.fps,
        )

    def evaluate_trajectory(self, result: VariantResult) -> dict[str, Any]:
        """Evaluate one strict dynamic-object artifact without raw-data reloads."""

        if result.human_joints is None or result.object_poses_demo is None:
            raise ValueError(
                f"{result.path} is missing saved human/object trajectories",
            )
        penetration_duration, penetration_max_depths = self.evaluate_penetration(
            result.qpos,
        )
        sliding_duration, max_toe_sliding_velocities = self._saved_foot_sliding(
            result,
        )
        contact_results = self.evaluate_contact_precision(
            result.human_joints,
            result.object_poses_demo,
            result.qpos,
        )
        return {
            "penetration_duration": penetration_duration,
            "penetration_max_depths": penetration_max_depths,
            "sliding_duration": sliding_duration,
            "max_toe_sliding_velocities": max_toe_sliding_velocities,
            "contact_preservation": contact_results,
            "opt_cost": float(result.cost),
        }

    def evaluate_terrain_contact_precision(
        self,
        human_joints_motion: np.ndarray,  # [T, J, 3] world
        q_trajectory: np.ndarray,  # [T, nq]
        joint_names=(
            "LeftHandMiddle3",
            "RightHandMiddle3",
            "LeftFoot",
            "RightFoot",
            "LeftToeBase",
            "RightToeBase",
        ),
    ) -> float:
        """
        For each frame:
        1) Detect demo contacts vs OBJECT.
        2) For each contacted joint, require mapped robot body to be within threshold to OBJECT.
        Returns preserved fraction over frames with any demo contact.
        """
        have_obj = self._obj_VW.shape[0] > 0
        if not have_obj:
            return 1.0  # nothing to check against

        preserved = []

        collision_prefix = f"{self.object_name}_collision_"
        collision_gids = [
            g
            for g in range(self.robot_model.ngeom)
            if (mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(collision_prefix)
        ]
        obj_gids = collision_gids or [
            g
            for g in range(self.robot_model.ngeom)
            if self.object_name in (mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_GEOM, g) or "")
        ]

        for q, demo_joints in zip(q_trajectory, human_joints_motion):
            self.robot_data.qpos[:] = q
            mujoco.mj_forward(self.robot_model, self.robot_data)
            # demo contacts (object only)
            dc = self.detect_demo_contact(demo_joints, joint_names)
            if not dc:
                continue

            ok = True
            for jn in dc:
                rb = self.joints_mapping.get(jn, "")
                if not rb:
                    continue
                bid = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, rb)
                if bid == -1:
                    continue
                dist_min = np.inf
                fromto = np.zeros(6)
                for g1 in range(self.robot_model.ngeom):
                    if self.robot_model.geom_bodyid[g1] != bid:
                        continue
                    for g2 in obj_gids:
                        dist = mujoco.mj_geomDistance(
                            self.robot_model, self.robot_data, g1, g2, self.collision_detection_threshold, fromto
                        )
                        dist_min = min(dist_min, dist)

                if dist_min > self.contact_threshold:
                    ok = False
                    break
            preserved.append(ok)
        return 1.0 if not preserved else float(np.mean(preserved))

    def evaluate_robot_terrain_trajectory(
        self,
        result: VariantResult,
    ) -> dict[str, Any]:
        """Evaluate one strict climbing artifact using its saved trajectories."""

        if result.human_joints is None:
            raise ValueError(f"{result.path} is missing saved human joints")
        penetration_duration, penetration_max_depths = self.evaluate_penetration(
            result.qpos,
        )
        sliding_duration, max_toe_sliding_velocities = self._saved_foot_sliding(
            result,
        )
        contact_results = self.evaluate_terrain_contact_precision(
            result.human_joints,
            result.qpos,
        )
        return {
            "penetration_duration": penetration_duration,
            "penetration_max_depths": penetration_max_depths,
            "sliding_duration": sliding_duration,
            "max_toe_sliding_velocities": max_toe_sliding_velocities,
            "contact_preservation": contact_results,
            "opt_cost": float(result.cost),
        }

    def evaluate_robot_only_trajectory(
        self,
        result: VariantResult,
    ) -> dict[str, Any]:
        """Evaluate one strict robot-only artifact using saved contact state."""

        penetration_duration, penetration_max_depths = self.evaluate_penetration(
            result.qpos,
        )
        sliding_duration, max_toe_sliding_velocities = self._saved_foot_sliding(
            result,
        )
        return {
            "penetration_duration": penetration_duration,
            "penetration_max_depths": penetration_max_depths,
            "sliding_duration": sliding_duration,
            "max_toe_sliding_velocities": max_toe_sliding_velocities,
            "opt_cost": float(result.cost),
        }


def _saved_job_payload(result: VariantResult) -> dict[str, Any]:
    if result.config_json is None:
        raise ValueError(f"{result.path} is missing saved config_json")
    try:
        payload = json.loads(result.config_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{result.path} has invalid config_json: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
        raise ValueError(f"{result.path} config_json has no normalized config")
    return payload


def _load_evaluation_result(
    path: str | Path,
    *,
    data_type: str,
    robot_type: str | None = None,
    data_format: str | None = None,
) -> VariantResult:
    """Load one saved trajectory and check fields required by evaluation."""

    result = load_variant_result(
        "identity",
        path,
    )
    return _validate_evaluation_result(
        result,
        data_type=data_type,
        robot_type=robot_type,
        data_format=data_format,
    )


def _validate_evaluation_result(
    result: VariantResult,
    *,
    data_type: str,
    robot_type: str | None = None,
    data_format: str | None = None,
) -> VariantResult:
    """Check fields that the requested evaluation actually consumes."""

    expected_task_type = _EVALUATION_TASK_TYPES[data_type]
    mismatches: list[str] = []
    if result.task_type != expected_task_type:
        mismatches.append(
            f"task_type={result.task_type!r}, expected {expected_task_type!r}",
        )
    if robot_type is not None and result.robot_type != robot_type:
        mismatches.append(
            f"robot_type={result.robot_type!r}, expected {robot_type!r}",
        )
    if data_format is not None and result.source_data_format != data_format:
        mismatches.append(
            f"source_data_format={result.source_data_format!r}, expected {data_format!r}",
        )
    if result.cost is None or not np.isfinite(result.cost):
        mismatches.append(f"cost={result.cost!r}")
    if result.human_joints is None or not result.human_joint_names:
        mismatches.append("saved human skeleton is absent")
    if result.foot_sticking is None:
        mismatches.append("saved foot-sticking state is absent")
    if data_type == "robot_object":
        if result.object_poses_demo is None:
            mismatches.append("saved demonstration object poses are absent")
        if result.object_urdf is None:
            mismatches.append("saved object asset is absent")
    if data_type == "robot_terrain":
        if result.human_position_scale is None:
            mismatches.append("saved human position scale is absent")
        if result.object_urdf is None:
            mismatches.append("saved terrain asset is absent")
    if mismatches:
        raise ValueError(
            f"Evaluation result {result.path} is missing or mismatches required data: {'; '.join(mismatches)}",
        )
    return result


def _climbing_source_scene(
    object_dir: Path,
    *,
    robot_type: str,
    object_name: str,
) -> Path:
    candidates = tuple(
        sorted(object_dir.glob(f"{robot_type}*_w_{object_name}.xml")),
    )
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one saved-config climbing scene for "
            f"{robot_type}/{object_name} in {object_dir}, found "
            f"{[path.name for path in candidates]}",
        )
    return candidates[0]


def _evaluate_single_task(
    task_name: str,
    data_path: str,
    robot_config_kwargs: Dict[str, Any],
    motion_data_config_kwargs: Dict[str, Any],
    object_name: str | None,
    data_type: str,
):
    robot_config = RobotConfig(**robot_config_kwargs)
    motion_data_config = MotionDataConfig(**motion_data_config_kwargs)
    result = _load_evaluation_result(
        data_path,
        data_type=data_type,
        robot_type=robot_config.robot_type,
        data_format=motion_data_config.data_format,
    )
    if result.sequence_key != task_name:
        raise ValueError(
            f"Discovered task name {task_name!r} does not match artifact sequence_key={result.sequence_key!r}",
        )
    saved_job = _saved_job_payload(result)
    saved_config = saved_job["config"]

    resolved_object_name = result.object_name
    if object_name is not None and object_name != resolved_object_name:
        raise ValueError(
            f"Explicit object_name={object_name!r} does not match saved object_name={resolved_object_name!r}",
        )

    asset_fingerprint = hashlib.sha256(
        (f"{robot_config.robot_type}:{data_type}:{task_name}:{resolved_object_name or 'ground'}").encode()
    ).hexdigest()
    generated_assets_dir = Path(data_path).expanduser().resolve().parent / ".generated-assets" / asset_fingerprint
    task_config = saved_config.get("task_config")
    if not isinstance(task_config, dict):
        raise ValueError(f"{result.path} saved config has no task_config")
    saved_object_scale = tuple(float(value) for value in task_config.get("object_scale", (1.0, 1.0, 1.0)))
    constants = create_task_constants(
        robot_config,
        motion_data_config,
        object_name=resolved_object_name,
        object_scale=saved_object_scale,
        saved_object_urdf=result.object_urdf,
        generated_assets_dir=generated_assets_dir,
    )

    if data_type == "robot_terrain":
        saved_object_dir = task_config.get("object_dir")
        if not isinstance(saved_object_dir, str) or not saved_object_dir:
            raise ValueError(
                f"{result.path} saved climbing config has no object_dir",
            )
        object_dir = Path(saved_object_dir).expanduser().resolve()
        source_scene = _climbing_source_scene(
            object_dir,
            robot_type=result.robot_type,
            object_name=result.object_name,
        )
        if result.human_position_scale is None:
            raise ValueError(
                f"{result.path} is missing saved human_position_scale",
            )
        variant = saved_job.get("variant")
        if not isinstance(variant, dict):
            raise ValueError(f"{result.path} saved config has no variant")
        variant_scale = np.asarray(
            variant.get("object_scale"),
            dtype=np.float64,
        )
        if variant_scale.shape != (3,):
            raise ValueError(
                f"{result.path} saved object_scale must have shape (3,)",
            )
        scene_scale = variant_scale * float(result.human_position_scale)
        constants.OBJECT_DIR = str(object_dir)
        constants.OBJECT_URDF_FILE = result.object_urdf
        constants.OBJECT_MESH_FILE = str(
            object_dir / f"{constants.OBJECT_NAME}.obj",
        )
        object_asset_xml_path = create_scaled_multi_boxes_xml(
            str(object_dir / "box_assets.xml"),
            tuple(scene_scale),
            output_dir=generated_assets_dir,
        )
        constants.SCENE_XML_FILE = create_new_scene_xml_file(
            str(source_scene),
            tuple(scene_scale),
            object_asset_xml_path,
            output_dir=generated_assets_dir,
        )

    evaluator = RetargetingEvaluator(
        robot_model_path=constants.ROBOT_URDF_FILE,
        object_name=constants.OBJECT_NAME,
        demo_joints=list(result.human_joint_names),
        joints_mapping=constants.JOINTS_MAPPING,
        constants=constants,
    )
    if data_type == "robot_object":
        return task_name, evaluator.evaluate_trajectory(result)
    if data_type == "robot_only":
        return task_name, evaluator.evaluate_robot_only_trajectory(result)
    if data_type == "robot_terrain":
        return task_name, evaluator.evaluate_robot_terrain_trajectory(result)
    raise ValueError(f"Invalid data type: {data_type}")


_EVALUATION_TASK_TYPES = {
    "robot_object": "object_interaction",
    "robot_only": "robot_only",
    "robot_terrain": "climbing",
}


def _canonical_evaluation_results(
    data_path: Path,
    data_type: str,
    *,
    robot_type: str | None = None,
    data_format: str | None = None,
) -> list[tuple[str, Path]]:
    """Discover identity results that contain the requested evaluation data."""

    identity_paths = sorted(path for path in data_path.rglob("identity.npz") if path.is_file())
    if not identity_paths:
        return []

    expected_task_type = _EVALUATION_TASK_TYPES[data_type]
    results: list[tuple[str, str, Path]] = []
    for path in identity_paths:
        try:
            result = load_variant_result(
                "identity",
                path,
            )
        except (KeyError, OSError, ValueError) as exc:
            raise ValueError(
                f"Cannot load evaluation result {path}: {exc}",
            ) from exc
        if result.task_type != expected_task_type:
            continue
        result = _validate_evaluation_result(
            result,
            data_type=data_type,
            robot_type=robot_type,
            data_format=data_format,
        )
        if result.dataset_partition is None or result.sequence_key is None:
            raise ValueError(
                f"Evaluation result {path} has no dataset identity",
            )
        results.append(
            (result.dataset_partition, result.sequence_key, path),
        )

    task_names = [sequence_key for _, sequence_key, _ in results]
    duplicates = sorted({task_name for task_name in task_names if task_names.count(task_name) > 1})
    if duplicates:
        conflicting = [
            f"{dataset_partition}/{sequence_key}: {path}"
            for dataset_partition, sequence_key, path in results
            if sequence_key in duplicates
        ]
        raise ValueError(
            "Canonical evaluation results contain duplicate sequence keys across dataset partitions. "
            "Select a narrower --res-dir before evaluation:\n" + "\n".join(conflicting)
        )
    return [(sequence_key, path) for _, sequence_key, path in results]


def get_task_names(
    data_dir: str | Path,
    data_type: str,
    *,
    robot_type: str | None = None,
    data_format: str | None = None,
) -> tuple[list[str], list[str]]:
    """Return identity results selected for evaluation."""

    if data_type not in {"robot_object", "robot_only", "robot_terrain"}:
        raise ValueError(f"Invalid data type: {data_type}")

    data_path = Path(data_dir).expanduser()
    canonical_results = _canonical_evaluation_results(
        data_path,
        data_type,
        robot_type=robot_type,
        data_format=data_format,
    )
    if canonical_results:
        return (
            [task_name for task_name, _ in canonical_results],
            [str(path) for _, path in canonical_results],
        )

    files = sorted(path for path in data_path.glob("*_original.npz") if path.is_file())
    if files:
        raise ValueError(
            "Evaluation no longer accepts legacy *_original.npz files because "
            "they do not provide the saved skeletons, contact state, FPS, and "
            "cost required by the evaluator. Regenerate them as identity.npz results first.",
        )
    return [], []


@dataclass
class Args:
    """Evaluation configuration."""

    res_dir: Path
    data_type: Literal["robot_object", "robot_only", "robot_terrain"] = "robot_object"
    robot: str = "g1"  # Use str to allow dynamic robot types
    data_format: str | None = None  # Use str to allow dynamic data formats
    object_name: str | None = None
    max_workers: int = 1

    # Nested configs for overrides
    robot_config: RobotConfig = field(default_factory=lambda: RobotConfig(robot_type="g1"))
    motion_data_config: MotionDataConfig = field(
        default_factory=lambda: MotionDataConfig(data_format="omomo", robot_type="g1")
    )


def main(cfg: Args) -> None:
    if cfg.max_workers <= 0:
        raise ValueError("max_workers must be greater than zero")

    default_data_formats = {
        "robot_object": "omomo",
        "robot_only": "omomo",
        "robot_terrain": "mocap",
    }

    data_format = normalize_data_format(cfg.data_format or default_data_formats[cfg.data_type])

    # Preserve nested overrides while binding the top-level selector fields.
    if cfg.robot_config.robot_type != cfg.robot:
        cfg.robot_config = replace(
            cfg.robot_config,
            robot_type=cfg.robot,
        )

    if cfg.motion_data_config.robot_type != cfg.robot or cfg.motion_data_config.data_format != data_format:
        cfg.motion_data_config = replace(
            cfg.motion_data_config,
            data_format=data_format,
            robot_type=cfg.robot,
        )

    # OMOMO robot-object workers infer their object independently from each result.
    if cfg.object_name is not None:
        object_name = cfg.object_name
    elif cfg.data_type == "robot_object":
        object_name = None
    elif cfg.data_type == "robot_terrain":
        object_name = "multi_boxes"
    else:
        # Default to "ground" for robot-only scenarios (matches robot defaults)
        object_name = "ground"

    task_names, files = get_task_names(
        cfg.res_dir,
        cfg.data_type,
        robot_type=cfg.robot,
        data_format=data_format,
    )
    print(f"Found {len(task_names)} tasks")

    robot_config_kwargs = asdict(cfg.robot_config)
    motion_data_config_kwargs = asdict(cfg.motion_data_config)

    results: Dict[str, Dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=cfg.max_workers) as executor:
        futures = {
            executor.submit(
                _evaluate_single_task,
                task_name,
                file_path,
                robot_config_kwargs,
                motion_data_config_kwargs,
                object_name,
                cfg.data_type,
            ): task_name
            for task_name, file_path in zip(task_names, files)
        }
        for fut in as_completed(futures):
            task_name, res = fut.result()
            if res is None:
                continue
            results[task_name] = res

    if not results:
        print("No evaluation results produced.")
        return

    # Aggregate metrics
    metrics: Dict[str, Any] = {}
    res_k_name = next(iter(results))
    for metric_k in results[res_k_name]:
        if "max" in metric_k:
            metrics[metric_k] = np.empty(0)
        else:
            metrics[metric_k] = []

    for res in results.values():
        for k, metric_vals in metrics.items():
            if "max" in k:
                metrics[k] = np.concatenate([metric_vals, res[k]])
            else:
                metric_vals.append(float(res[k]))

    for k, vals in metrics.items():
        if len(vals) == 0:
            mean_k, std_k = 0.0, 0.0
        else:
            mean_k, std_k = float(np.mean(vals)), float(np.std(vals))
        print(f"{k}: mean={mean_k:.6f}, std={std_k:.6f}")


if __name__ == "__main__":
    cfg = tyro.cli(Args)
    main(cfg)
