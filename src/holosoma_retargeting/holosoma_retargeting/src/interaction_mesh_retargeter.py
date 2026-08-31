# ruff: noqa: CPY001, PLR0917

from __future__ import annotations

import sys
import time
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import cvxpy as cp  # type: ignore[import-not-found]
import mujoco  # type: ignore[import-not-found]
import numpy as np
import trimesh
import viser  # type: ignore[import-not-found]
import yourdfpy  # type: ignore[import-untyped]
from scipy import sparse as sp  # type: ignore[import-untyped]
from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]
from tqdm import tqdm
from viser.extras import ViserUrdf  # type: ignore[import-not-found]

from holosoma_retargeting.config_types.retargeter import (
    FootLockConfig,
    PlanarFootContactConfig,
    RootStabilityConfig,
    SelfCollisionConfig,
    ShoulderDirectionConfig,
)
from holosoma_retargeting.data_utils.hand_skeleton import build_hand_visualization_spec
from holosoma_retargeting.foot_contact import FootContactPlan, FootContactState
from holosoma_retargeting.shoulder_direction import (
    ShoulderSideSpec,
    direction_error_angle,
    normalize_vector,
    unit_direction_jacobian,
)

# Add src to path for direct execution
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Import with type ignore for mypy compatibility
from mujoco_utils import (  # type: ignore[import-not-found,no-redef]  # noqa: E402
    _world_mesh_from_geom,
)
from utils import (  # type: ignore[import-not-found,no-redef]  # noqa: E402
    calculate_laplacian_coordinates,
    calculate_laplacian_matrix,
    create_interaction_mesh,
    get_adjacency_list,
    interaction_mesh_edges_from_tetrahedra,
    transform_points_local_to_world,
    transform_points_world_to_local,
)
from viser_utils import (  # type: ignore[import-not-found,no-redef]  # noqa: E402
    actuated_joint_names_from_mujoco_xml,
    build_joint_order_indices,
    create_motion_control_sliders,
    format_foot_sticking_status,
    register_keyboard_shortcut,
)


@dataclass(frozen=True)
class _ResolvedPlanarFootTarget:
    """Robot-space target resolved once at the start of a motion frame."""

    side: str
    mode: str
    phase_id: int
    pivot_uv: tuple[float, float]
    anchor_xy: np.ndarray
    heading_anchor_xy: np.ndarray | None = None


class InteractionMeshRetargeter:
    """
    A class to perform kinematic retargeting from human motion to a robot,
    preserving spatial relationships using an interaction mesh.
    """

    def __init__(
        self,
        task_constants: ModuleType,
        object_urdf_path: str,
        q_a_init_idx: int = -7,
        activate_foot_sticking: bool = True,
        activate_obj_non_penetration: bool = True,
        activate_joint_limits: bool = True,
        step_size: float = 0.2,
        collision_detection_threshold: float = 0.1,
        penetration_tolerance: float = 1e-3,
        foot_sticking_tolerance: float = 1e-3,
        planar_foot_contact: PlanarFootContactConfig | None = None,
        foot_lock: FootLockConfig | None = None,
        self_collision: SelfCollisionConfig | None = None,
        visualize: bool = False,
        mesh_opacity: float = 1.0,
        debug: bool = False,
        dynamic_ground_window: bool = True,
        interaction_mesh_weight: float = 10.0,
        arm_interaction_mesh_weight_scale: float = 1.0,
        root_stability: RootStabilityConfig | None = None,
        show_interaction_mesh: bool = False,
        save_interaction_mesh: bool = True,
        interaction_mesh_mode: str = "both",
        interaction_mesh_edges: str = "cross",
        interaction_mesh_line_width: float = 1.0,
        w_nominal_tracking_init: float = 5.0,
        nominal_tracking_tau: float = 10.0,
        orientation_joints_mapping: dict[str, str] | None = None,
        orientation_weights: dict[str, float] | None = None,
        orientation_preview: bool = False,
        orientation_alignment_mode: str = "t_pose",
        orientation_t_pose_human_quaternions_wxyz: dict[
            str,
            tuple[float, float, float, float],
        ]
        | None = None,
        orientation_t_pose_robot_base_quaternion_wxyz: tuple[
            float,
            float,
            float,
            float,
        ]
        | None = None,
        orientation_t_pose_robot_joint_positions: dict[str, float] | None = None,
        orientation_alignment_quaternions_wxyz: dict[
            str,
            tuple[float, float, float, float],
        ]
        | None = None,
        shoulder_direction: ShoulderDirectionConfig | None = None,
        natural_pose_joint_positions: dict[str, float] | None = None,
        natural_pose_weights: dict[str, float] | None = None,
    ):
        """This kinematic retargeter solves the diffIK problem with hard constraints in SQP style.
        During each SQP iteration, the problem is solved with the following constraints and costs:
            1. [Cost] Minimize the Laplacian deformation in the object frame.
            2. [Constraint] Enforce the non-penetration constraints w/ the ground and (if activated) the object.
            3. [Constraint] Enforce contact-aware planar foot constraints if activated.
            4. [Constraint] Enforce the joint limits if activated.
            5. [Constraint] Enforce trust region of dq.
        The constraints are linearized and the costs are quadratic with a trust region.

        Args:
            q_a_init_idx: offset used to select the first optimized qpos
                address as ``7 + q_a_init_idx``. For example, -7 starts at
                qpos[0] (the full floating base), -3 starts at qpos[4] (the
                final three floating-base quaternion components), and 0 starts
                at qpos[7] (the actuated DOFs). Positive offsets select later
                actuated joints according to the configured robot topology.
            step_size: trust region for each SQP iteration.
            collision_detection_threshold: only start to detect collision
            when the distance is smaller than this threshold.
            penetration_tolerance: tolerance for penetration when enforcing non-penetration constraints.
            foot_sticking_tolerance: XY tolerance for planted sole regions.
            planar_foot_contact: contact detection and mode-specific constraint settings.
            foot_lock: configuration for explicit frame-range based foot locking constraints.
            interaction_mesh_weight: global Interaction Mesh deformation-energy weight.
            arm_interaction_mesh_weight_scale: upper-limb anchor multiplier
                relative to the global Interaction Mesh weight.
            root_stability: optional source-aligned floating-root position and
                torso-orientation objective.
            nominal_tracking_tau: the time constant for the nominal tracking cost.
            natural_pose_joint_positions: fixed natural-pose references,
                keyed by scalar MuJoCo joint name.
            natural_pose_weights: non-negative absolute natural-pose cost
                weights,
                keyed by the same scalar joint names as the references.
        """

        self.task_constants = task_constants
        self.robot_model_path = task_constants.ROBOT_URDF_FILE
        if object_urdf_path:
            object_path = Path(object_urdf_path).expanduser()
            if not object_path.is_absolute():
                object_path = Path.cwd() / object_path
            self.object_model_path = str(object_path.resolve())
        else:
            self.object_model_path = None
        self.object_name = task_constants.OBJECT_NAME
        self.collision_detection_threshold = collision_detection_threshold
        self.activate_foot_sticking = activate_foot_sticking
        self.activate_obj_non_penetration = activate_obj_non_penetration
        self.activate_joint_limits = activate_joint_limits
        self._init_planar_foot_contact(planar_foot_contact)
        self.penetration_tolerance = float(penetration_tolerance)
        if not np.isfinite(self.penetration_tolerance) or self.penetration_tolerance < 0:
            raise ValueError("penetration_tolerance must be finite and non-negative")
        self.step_size = step_size
        self.visualize = visualize
        self.mesh_opacity = float(mesh_opacity)
        self.debug = debug
        self.dynamic_ground_window = dynamic_ground_window
        self.show_interaction_mesh = show_interaction_mesh
        if not save_interaction_mesh:
            raise ValueError("Canonical retargeting artifacts require Interaction Mesh data")
        self.save_interaction_mesh = True
        self.interaction_mesh_mode = interaction_mesh_mode
        self.interaction_mesh_edges = interaction_mesh_edges
        self.interaction_mesh_line_width = float(interaction_mesh_line_width)
        if self.interaction_mesh_mode not in {"source", "target", "both"}:
            raise ValueError(f"Unknown interaction_mesh_mode: {self.interaction_mesh_mode}")
        if self.interaction_mesh_edges not in {"all", "cross"}:
            raise ValueError(f"Unknown interaction_mesh_edges: {self.interaction_mesh_edges}")
        self.demo_joints = task_constants.DEMO_JOINTS
        self.human_joint_parent_indices = np.asarray(
            task_constants.DEMO_JOINT_PARENT_INDICES,
            dtype=np.int32,
        )
        if self.human_joint_parent_indices.shape != (len(self.demo_joints),):
            raise ValueError("DEMO_JOINT_PARENT_INDICES must contain one parent for every DEMO_JOINTS entry")
        self.laplacian_match_links = task_constants.JOINTS_MAPPING
        self.qpos_to_viser_joint_indices: np.ndarray | None = None

        self.mapped_joint_indices = [self.demo_joints.index(name) for name in self.laplacian_match_links]
        self._hand_visualization_spec = build_hand_visualization_spec(
            self.demo_joints,
            list(self.laplacian_match_links),
        )

        # Setup weights and parameters
        self.interaction_mesh_weight = float(interaction_mesh_weight)
        self.arm_interaction_mesh_weight_scale = float(
            arm_interaction_mesh_weight_scale,
        )
        if (
            not np.isfinite(self.interaction_mesh_weight)
            or self.interaction_mesh_weight < 0.0
        ):
            raise ValueError(
                "interaction_mesh_weight must be finite and non-negative",
            )
        if (
            not np.isfinite(self.arm_interaction_mesh_weight_scale)
            or self.arm_interaction_mesh_weight_scale < 0.0
        ):
            raise ValueError(
                "arm_interaction_mesh_weight_scale must be finite and non-negative",
            )
        # Retain the historical attribute for downstream code that inspects a
        # constructed retargeter directly.
        self.laplacian_weights = self.interaction_mesh_weight
        self.smooth_weight = 0.2
        # Tolerance for foot sticking constraints in x, y.
        self.foot_sticking_tolerance = float(foot_sticking_tolerance)
        if self.foot_sticking_tolerance < 0:
            raise ValueError("foot_sticking_tolerance must be non-negative")
        self._init_foot_lock(foot_lock)
        self._self_collision_config = self_collision

        scene_xml_file = getattr(self.task_constants, "SCENE_XML_FILE", "")
        if scene_xml_file:
            robot_xml_path = scene_xml_file
        elif self.object_name == "ground":
            robot_xml_path = self.robot_model_path.replace(".urdf", ".xml")
        elif self.object_name == "multi_boxes":
            robot_xml_path = self.task_constants.SCENE_XML_FILE
        else:
            robot_xml_path = self.robot_model_path.replace(".urdf", "_w_" + self.object_name + ".xml")
        self.robot_xml_path = robot_xml_path

        # Setup visualization if requested
        if self.visualize:
            self._setup_visualization()

        # Load Mujoco model
        self.robot_model = mujoco.MjModel.from_xml_path(self.robot_xml_path)
        print("Loading robot model from: ", self.robot_xml_path)

        self.robot_data = mujoco.MjData(self.robot_model)
        (
            self.robot_link_body_ids,
            self.robot_link_names,
            self.robot_link_parent_indices,
        ) = self._robot_link_topology()
        self.robot_actuated_joint_names = self._robot_actuated_joint_names()
        self.robot_link_name_to_index = {name: index for index, name in enumerate(self.robot_link_names)}
        missing_foot_links = sorted(set(self.foot_links).difference(self.robot_link_name_to_index))
        if missing_foot_links:
            raise ValueError(f"Planar foot-contact layout references unknown robot links: {missing_foot_links}")
        self._init_self_collision(self._self_collision_config)

        if self.robot_data.qpos.shape[0] > 7 + self.task_constants.ROBOT_DOF:
            self.has_dynamic_object = True
        else:
            self.has_dynamic_object = False

        self.nq = self.robot_model.nq

        self.q_a_init_idx = q_a_init_idx
        first_optimized_qpos = 7 + self.q_a_init_idx
        robot_qpos_stop = 7 + self.task_constants.ROBOT_DOF
        if not 0 <= first_optimized_qpos < robot_qpos_stop:
            raise ValueError(
                f"q_a_init_idx must select a non-empty suffix of the robot configuration; got {q_a_init_idx}"
            )
        self.q_a_indices = np.arange(first_optimized_qpos, robot_qpos_stop)

        self.nq_a = len(self.q_a_indices)

        # Align scalar joint limits by MuJoCo qpos address. Named free joints must not
        # shift the actuated joint limits.
        large_number = 1e6
        complete_lower_limits = -large_number * np.ones(self.nq)
        complete_upper_limits = large_number * np.ones(self.nq)
        scalar_joint_types = {
            int(mujoco.mjtJoint.mjJNT_SLIDE),
            int(mujoco.mjtJoint.mjJNT_HINGE),
        }
        for joint_idx in range(self.robot_model.njnt):
            if int(self.robot_model.jnt_type[joint_idx]) not in scalar_joint_types:
                continue
            if not self.robot_model.jnt_limited[joint_idx]:
                continue
            qpos_addr = int(self.robot_model.jnt_qposadr[joint_idx])
            complete_lower_limits[qpos_addr] = self.robot_model.jnt_range[joint_idx, 0]
            complete_upper_limits[qpos_addr] = self.robot_model.jnt_range[joint_idx, 1]

        self.q_a_lb = complete_lower_limits[self.q_a_indices]
        self.q_a_ub = complete_upper_limits[self.q_a_indices]

        self._apply_qpos_overrides(
            self.q_a_lb,
            self.task_constants.MANUAL_LB,
            metadata_name="MANUAL_LB",
        )
        self._apply_qpos_overrides(
            self.q_a_ub,
            self.task_constants.MANUAL_UB,
            metadata_name="MANUAL_UB",
        )

        # Prevent too much waist twist
        self.Q_diag = np.zeros(self.nq_a) * 1e-3
        self._apply_qpos_overrides(
            self.Q_diag,
            self.task_constants.MANUAL_COST,
            metadata_name="MANUAL_COST",
        )

        self.last_sqp_iteration_count = 0
        self.last_sqp_stop_reason = "not_started"

        self.w_nominal_tracking_init = w_nominal_tracking_init
        self.nominal_tracking_tau = nominal_tracking_tau
        self.track_nominal_indices = self._reduced_indices_for_qpos_addresses(
            task_constants.NOMINAL_TRACKING_INDICES,
            metadata_name="NOMINAL_TRACKING_INDICES",
        )
        self._init_orientation_tracking(
            orientation_joints_mapping=orientation_joints_mapping,
            orientation_weights=orientation_weights,
            orientation_preview=orientation_preview,
            orientation_alignment_mode=orientation_alignment_mode,
            orientation_t_pose_human_quaternions_wxyz=(orientation_t_pose_human_quaternions_wxyz),
            orientation_t_pose_robot_base_quaternion_wxyz=(orientation_t_pose_robot_base_quaternion_wxyz),
            orientation_t_pose_robot_joint_positions=(orientation_t_pose_robot_joint_positions),
            orientation_alignment_quaternions_wxyz=(orientation_alignment_quaternions_wxyz),
        )
        self._init_shoulder_direction_tracking(shoulder_direction)
        self._init_root_stability(root_stability)
        self._init_natural_pose_regularization(
            natural_pose_joint_positions=natural_pose_joint_positions,
            natural_pose_weights=natural_pose_weights,
        )

    def _init_shoulder_direction_tracking(
        self,
        config: ShoulderDirectionConfig | None,
    ) -> None:
        """Discover supported upper-limb chains and validate direction weights."""

        resolved = config or ShoulderDirectionConfig()
        numeric_weights = {
            "direction_weight": float(resolved.direction_weight),
            "wrist_axis_weight_scale": float(resolved.wrist_axis_weight_scale),
        }
        invalid_weights = {
            name: value
            for name, value in numeric_weights.items()
            if not np.isfinite(value) or value < 0.0
        }
        if invalid_weights:
            raise ValueError(
                "Shoulder direction weights must be finite and non-negative: "
                f"{invalid_weights}",
            )

        self.shoulder_direction_config = resolved
        self.shoulder_direction_enabled = bool(resolved.enable)
        self.shoulder_side_specs: tuple[ShoulderSideSpec, ...] = ()
        self.shoulder_joint_qpos_addresses = np.empty((0, 3), dtype=np.int32)
        self.shoulder_joint_reduced_indices = np.empty((0, 3), dtype=np.int32)
        self.shoulder_wrist_reduced_indices = np.empty((0, 3), dtype=np.int32)
        self.shoulder_wrist_dof_counts = np.empty((0,), dtype=np.int32)
        self.shoulder_wrist_orientation_indices = np.empty((0,), dtype=np.int32)
        self.shoulder_wrist_weights = np.empty((0,), dtype=np.float64)
        self.shoulder_reserved_orientation_indices = np.empty((0,), dtype=np.int32)
        self.shoulder_human_torso_origin_name = ""
        self.shoulder_human_torso_side_names: tuple[str, str] = ()
        self.shoulder_torso_link_name = ""
        if not self.shoulder_direction_enabled:
            return

        human_schemas = (
            (
                "Hips",
                (
                    ("LeftArm", "LeftForeArm", "LeftHand"),
                    ("RightArm", "RightForeArm", "RightHand"),
                ),
            ),
            (
                "Pelvis",
                (
                    ("L_Shoulder", "L_Elbow", "L_Wrist"),
                    ("R_Shoulder", "R_Elbow", "R_Wrist"),
                ),
            ),
        )
        human_schema = next(
            (
                (torso_origin, side_names)
                for torso_origin, side_names in human_schemas
                if {
                    torso_origin,
                    *(name for names in side_names for name in names),
                }.issubset(self.demo_joints)
            ),
            None,
        )
        if human_schema is None:
            raise ValueError(
                "Shoulder direction tracking requires either the BVH/FBX "
                "Hips/Arm/ForeArm/Hand schema or the SMPL-X "
                "Pelvis/Shoulder/Elbow/Wrist schema",
            )
        human_torso_origin, human_side_names = human_schema
        self.shoulder_human_torso_origin_name = human_torso_origin
        self.shoulder_human_torso_side_names = (
            human_side_names[0][0],
            human_side_names[1][0],
        )

        def _body_exists(name: str) -> bool:
            return (
                mujoco.mj_name2id(
                    self.robot_model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    name,
                )
                >= 0
            )

        def _joint_exists(name: str) -> bool:
            return (
                mujoco.mj_name2id(
                    self.robot_model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    name,
                )
                >= 0
            )

        is_e1 = _joint_exists("l_arm_elbow_pitch_joint")
        is_e2 = _joint_exists("l_arm_elbow_joint")
        is_g1 = _joint_exists("left_elbow_joint") and _joint_exists(
            "left_wrist_roll_joint",
        )
        if sum((is_e1, is_e2, is_g1)) != 1:
            raise ValueError(
                "Shoulder direction tracking supports the G1, E1, and E2 arm chains only",
            )
        specs: list[ShoulderSideSpec] = []
        robot_sides = (("l", "left"), ("r", "right"))
        for (short_prefix, long_prefix), human_names in zip(
            robot_sides,
            human_side_names,
            strict=True,
        ):
            if is_g1:
                shoulder_joint_names = (
                    f"{long_prefix}_shoulder_pitch_joint",
                    f"{long_prefix}_shoulder_roll_joint",
                    f"{long_prefix}_shoulder_yaw_joint",
                )
                torso_basis_link = f"{long_prefix}_shoulder_pitch_link"
                shoulder_anchor_link = f"{long_prefix}_shoulder_roll_link"
                elbow_link = f"{long_prefix}_elbow_link"
                elbow_joint = f"{long_prefix}_elbow_joint"
                hand_link = f"{long_prefix}_rubber_hand_link"
                wrist_joint_names = (
                    f"{long_prefix}_wrist_roll_joint",
                    f"{long_prefix}_wrist_pitch_joint",
                    f"{long_prefix}_wrist_yaw_joint",
                )
            else:
                shoulder_joint_names = (
                    f"{short_prefix}_arm_shoulder_pitch_joint",
                    f"{short_prefix}_arm_shoulder_roll_joint",
                    f"{short_prefix}_arm_shoulder_yaw_joint",
                )
                torso_basis_link = f"{short_prefix}_arm_shoulder_pitch_link"
                shoulder_anchor_link = f"{short_prefix}_arm_shoulder_roll_link"
                elbow_link = (
                    f"{short_prefix}_arm_elbow_pitch_link"
                    if is_e1
                    else f"{short_prefix}_arm_elbow_link"
                )
                elbow_joint = (
                    f"{short_prefix}_arm_elbow_pitch_joint"
                    if is_e1
                    else f"{short_prefix}_arm_elbow_joint"
                )
                hand_link = f"{short_prefix}_hand_sphere_link"
                wrist_joint_names = (
                    (f"{short_prefix}_arm_elbow_yaw_joint",)
                    if is_e1
                    else ()
                )
            spec = ShoulderSideSpec(
                human_arm_name=human_names[0],
                human_forearm_name=human_names[1],
                human_hand_name=human_names[2],
                shoulder_joint_names=shoulder_joint_names,
                torso_basis_link_name=torso_basis_link,
                shoulder_anchor_link_name=shoulder_anchor_link,
                elbow_link_name=elbow_link,
                hand_link_name=hand_link,
                elbow_joint_name=elbow_joint,
                wrist_joint_names=wrist_joint_names,
            )
            model_names = (
                *spec.shoulder_joint_names,
                spec.elbow_joint_name,
                *spec.wrist_joint_names,
            )
            missing_joints = [name for name in model_names if not _joint_exists(name)]
            body_names = (
                spec.torso_basis_link_name,
                spec.shoulder_anchor_link_name,
                spec.elbow_link_name,
                spec.hand_link_name,
            )
            missing_bodies = [name for name in body_names if not _body_exists(name)]
            if missing_joints or missing_bodies:
                raise ValueError(
                    "Incomplete shoulder direction chain: "
                    f"missing_joints={missing_joints}, missing_bodies={missing_bodies}",
                )
            specs.append(spec)

        left_basis_id = mujoco.mj_name2id(
            self.robot_model,
            mujoco.mjtObj.mjOBJ_BODY,
            specs[0].torso_basis_link_name,
        )
        torso_body_id = int(self.robot_model.body_parentid[left_basis_id])
        torso_name = mujoco.mj_id2name(
            self.robot_model,
            mujoco.mjtObj.mjOBJ_BODY,
            torso_body_id,
        )
        if not torso_name:
            raise ValueError("Could not resolve the shoulder parent torso body")
        self.shoulder_torso_link_name = str(torso_name)

        active_index_by_qpos = {
            int(address): index
            for index, address in enumerate(self.q_a_indices)
        }
        shoulder_qpos: list[list[int]] = []
        shoulder_reduced: list[list[int]] = []
        wrist_reduced: list[list[int]] = []
        wrist_dof_counts: list[int] = []
        wrist_orientation_indices: list[int] = []
        wrist_weights: list[float] = []
        orientation_index_by_human = {
            name: index
            for index, name in enumerate(self.orientation_human_joint_names)
        }
        for spec in specs:
            side_qpos: list[int] = []
            side_reduced: list[int] = []
            for joint_name in spec.shoulder_joint_names:
                joint_id = mujoco.mj_name2id(
                    self.robot_model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    joint_name,
                )
                qpos_address = int(self.robot_model.jnt_qposadr[joint_id])
                if qpos_address not in active_index_by_qpos:
                    raise ValueError(
                        f"Shoulder joint {joint_name!r} is outside active SQP variables",
                    )
                side_qpos.append(qpos_address)
                side_reduced.append(active_index_by_qpos[qpos_address])
            shoulder_qpos.append(side_qpos)
            shoulder_reduced.append(side_reduced)

            if not spec.wrist_joint_names:
                wrist_reduced.append([-1, -1, -1])
                wrist_dof_counts.append(0)
                wrist_orientation_indices.append(-1)
                wrist_weights.append(0.0)
                continue
            side_wrist_reduced: list[int] = []
            for wrist_joint_name in spec.wrist_joint_names:
                wrist_id = mujoco.mj_name2id(
                    self.robot_model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    wrist_joint_name,
                )
                wrist_address = int(self.robot_model.jnt_qposadr[wrist_id])
                if wrist_address not in active_index_by_qpos:
                    raise ValueError(
                        f"Wrist joint {wrist_joint_name!r} is outside active variables",
                    )
                side_wrist_reduced.append(active_index_by_qpos[wrist_address])
            orientation_index = orientation_index_by_human.get(
                spec.human_hand_name,
                -1,
            )
            wrist_dof_counts.append(len(side_wrist_reduced))
            wrist_reduced.append(
                side_wrist_reduced + [-1] * (3 - len(side_wrist_reduced)),
            )
            wrist_orientation_indices.append(orientation_index)
            wrist_weights.append(
                0.0
                if orientation_index < 0
                else float(self.orientation_weight_values[orientation_index])
                * float(resolved.wrist_axis_weight_scale)
            )

        reserved_names = {
            name
            for spec in specs
            for name in (
                spec.human_arm_name,
                spec.human_forearm_name,
                spec.human_hand_name,
            )
        }
        self.shoulder_side_specs = tuple(specs)
        self.shoulder_joint_qpos_addresses = np.asarray(
            shoulder_qpos,
            dtype=np.int32,
        )
        self.shoulder_joint_reduced_indices = np.asarray(
            shoulder_reduced,
            dtype=np.int32,
        )
        self.shoulder_wrist_reduced_indices = np.asarray(
            wrist_reduced,
            dtype=np.int32,
        )
        self.shoulder_wrist_dof_counts = np.asarray(
            wrist_dof_counts,
            dtype=np.int32,
        )
        self.shoulder_wrist_orientation_indices = np.asarray(
            wrist_orientation_indices,
            dtype=np.int32,
        )
        self.shoulder_wrist_weights = np.asarray(wrist_weights, dtype=np.float64)
        self.shoulder_reserved_orientation_indices = np.asarray(
            [
                index
                for index, name in enumerate(self.orientation_human_joint_names)
                if name in reserved_names
            ],
            dtype=np.int32,
        )

    def _init_root_stability(
        self,
        config: RootStabilityConfig | None,
    ) -> None:
        """Resolve and validate an optional source-aligned root objective."""

        resolved = config or RootStabilityConfig()
        position_weight = float(resolved.position_weight)
        orientation_weight = float(resolved.orientation_weight)
        invalid = {
            name: value
            for name, value in (
                ("position_weight", position_weight),
                ("orientation_weight", orientation_weight),
            )
            if not np.isfinite(value) or value < 0.0
        }
        if invalid:
            raise ValueError(
                "Root stability weights must be finite and non-negative: "
                f"{invalid}",
            )

        self.root_stability_config = resolved
        self.root_stability_enabled = bool(
            position_weight > 0.0 or orientation_weight > 0.0,
        )
        root_joint_candidates = [
            joint_id
            for joint_id in range(self.robot_model.njnt)
            if int(self.robot_model.jnt_qposadr[joint_id]) == 0
            and int(self.robot_model.jnt_type[joint_id])
            == int(mujoco.mjtJoint.mjJNT_FREE)
        ]
        if len(root_joint_candidates) != 1:
            raise ValueError(
                "Expected exactly one robot free-root joint at qpos address 0",
            )
        root_body_id = int(
            self.robot_model.jnt_bodyid[root_joint_candidates[0]],
        )
        root_body_name = mujoco.mj_id2name(
            self.robot_model,
            mujoco.mjtObj.mjOBJ_BODY,
            root_body_id,
        )
        if not root_body_name:
            raise ValueError("Could not resolve the robot free-root body name")
        self.root_stability_robot_link_name = str(root_body_name)
        self.root_stability_robot_link_index = self.robot_link_name_to_index[
            self.root_stability_robot_link_name
        ]

        human_root_indices = np.flatnonzero(
            self.human_joint_parent_indices == -1,
        )
        if len(human_root_indices) != 1:
            raise ValueError("Human skeleton must contain exactly one root joint")
        self.root_stability_human_root_index = int(human_root_indices[0])
        self.root_stability_human_root_name = self.demo_joints[
            self.root_stability_human_root_index
        ]
        self.root_stability_orientation_index = (
            self.orientation_human_joint_names.index(
                self.root_stability_human_root_name,
            )
            if self.root_stability_human_root_name
            in self.orientation_human_joint_names
            else -1
        )
        self.root_stability_orientation_source = "disabled"
        shoulder_name_pairs = (
            ("LeftArm", "RightArm"),
            ("L_Shoulder", "R_Shoulder"),
        )
        self.root_stability_human_shoulder_names = next(
            (
                pair
                for pair in shoulder_name_pairs
                if pair[0] in self.demo_joints and pair[1] in self.demo_joints
            ),
            (),
        )
        if not self.root_stability_enabled:
            return

        active_qpos = {int(address) for address in self.q_a_indices}
        if position_weight > 0.0 and not {0, 1, 2}.issubset(active_qpos):
            raise ValueError(
                "Root position stability requires the floating-base xyz qpos "
                "coordinates to be active; use q_a_init_idx=-7",
            )
        if orientation_weight > 0.0 and not {3, 4, 5, 6}.issubset(
            active_qpos,
        ):
            raise ValueError(
                "Root orientation stability requires the complete floating-base "
                "quaternion to be active; use q_a_init_idx=-7",
            )
        if (
            orientation_weight > 0.0
            and self.root_stability_orientation_index < 0
            and not self.root_stability_human_shoulder_names
        ):
            raise ValueError(
                "Root orientation stability requires direct root orientations "
                "or a supported left/right human shoulder pair",
            )

    def _prepare_root_stability_targets(
        self,
        human_joint_motions: np.ndarray,
        initial_q: np.ndarray,
        root_orientation_reference_matrices: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Align source-root motion to an already-retargeted robot frame zero."""

        frame_count = int(human_joint_motions.shape[0])
        if not self.root_stability_enabled:
            return (
                np.empty((frame_count, 0), dtype=np.float64),
                np.empty((frame_count, 0, 3, 3), dtype=np.float64),
            )
        motions = np.asarray(human_joint_motions, dtype=np.float64)
        expected_shape = (frame_count, len(self.demo_joints), 3)
        if motions.shape != expected_shape or not np.isfinite(motions).all():
            raise ValueError(
                "Root stability requires finite human joint positions with "
                f"shape {expected_shape}",
            )
        q = np.asarray(initial_q, dtype=np.float64)
        self.robot_data.qpos[:] = q
        mujoco.mj_forward(self.robot_model, self.robot_data)
        root_body_id = mujoco.mj_name2id(
            self.robot_model,
            mujoco.mjtObj.mjOBJ_BODY,
            self.root_stability_robot_link_name,
        )
        initial_robot_position = np.array(
            self.robot_data.xpos[root_body_id],
            dtype=np.float64,
            copy=True,
        )
        initial_robot_matrix = np.array(
            self.robot_data.xmat[root_body_id].reshape(3, 3),
            dtype=np.float64,
            copy=True,
        )
        human_root_positions = motions[
            :,
            self.root_stability_human_root_index,
        ]
        target_positions = (
            initial_robot_position[None, :]
            + human_root_positions
            - human_root_positions[0]
        )

        if (
            float(self.root_stability_config.orientation_weight) <= 0.0
        ):
            self.root_stability_orientation_source = "disabled"
            return target_positions, np.empty(
                (frame_count, 0, 3, 3),
                dtype=np.float64,
            )

        if root_orientation_reference_matrices is not None:
            human_matrices = np.asarray(
                root_orientation_reference_matrices,
                dtype=np.float64,
            )
            if human_matrices.shape != (frame_count, 3, 3):
                raise ValueError(
                    "Root orientation references must have shape "
                    f"{(frame_count, 3, 3)}, got {human_matrices.shape}",
                )
            if not np.isfinite(human_matrices).all():
                raise ValueError("Root orientation references must be finite")
            self.root_stability_orientation_source = "direct_root_orientation"
        else:
            if not self.root_stability_human_shoulder_names:
                raise ValueError(
                    "Root orientation stability requires either direct root "
                    "orientations or a supported left/right shoulder pair",
                )
            left_name, right_name = self.root_stability_human_shoulder_names
            left_index = self.demo_joints.index(left_name)
            right_index = self.demo_joints.index(right_name)
            human_matrices = np.asarray(
                [
                    self._anatomical_basis(
                        frame[left_index],
                        frame[right_index],
                        frame[self.root_stability_human_root_index],
                    )
                    for frame in motions
                ],
                dtype=np.float64,
            )
            self.root_stability_orientation_source = "shoulder_root_basis"
        alignment = human_matrices[0].T @ initial_robot_matrix
        target_matrices = human_matrices @ alignment[None, ...]
        return target_positions, target_matrices

    def _reserved_orientation_objective_indices(
        self,
        *,
        root_stability_bootstrap: bool,
    ) -> set[int]:
        """Return generic SO(3) rows replaced by specialized objectives."""

        reserved = set(self.shoulder_reserved_orientation_indices.tolist())
        if (
            not root_stability_bootstrap
            and float(self.root_stability_config.orientation_weight) > 0.0
            and self.root_stability_orientation_index >= 0
        ):
            reserved.add(self.root_stability_orientation_index)
        return reserved

    def _get_root_stability_data(
        self,
        q: np.ndarray,
        *,
        with_jacobians: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Return root-body pose and optional active position/angular Jacobians."""

        links = {"root": self.root_stability_robot_link_name}
        position_jacobians, positions, _ = self._calc_manipulator_jacobians(
            q,
            links=links,
            obj_frame=False,
        )
        matrices, orientation_jacobians = self._get_robot_link_orientation_data(
            q,
            [self.root_stability_robot_link_name],
            with_jacobians=with_jacobians,
        )
        return (
            positions["root"],
            matrices[0],
            position_jacobians["root"] if with_jacobians else None,
            (
                orientation_jacobians[0]
                if with_jacobians and orientation_jacobians is not None
                else None
            ),
        )

    @staticmethod
    def _is_upper_limb_anchor(name: str) -> bool:
        """Return whether a mapped human anchor belongs to either upper limb."""

        normalized = name.lower().replace("_", "")
        return any(
            token in normalized
            for token in ("shoulder", "arm", "elbow", "wrist", "hand")
        )

    def _interaction_mesh_vertex_weights(
        self,
        robot_link_keys: list[str],
        vertex_count: int,
    ) -> np.ndarray:
        """Build tunable per-row weights for the Interaction Mesh objective."""

        weights = np.full(
            vertex_count,
            self.interaction_mesh_weight,
            dtype=np.float64,
        )
        arm_scale = self.arm_interaction_mesh_weight_scale
        for index, name in enumerate(robot_link_keys):
            if self._is_upper_limb_anchor(name):
                weights[index] *= arm_scale
        return weights

    @staticmethod
    def _anatomical_basis(
        left_shoulder: np.ndarray,
        right_shoulder: np.ndarray,
        torso_origin: np.ndarray,
    ) -> np.ndarray:
        """Build a matching forward/lateral/up basis from three body points."""

        lateral = normalize_vector(
            np.asarray(left_shoulder) - np.asarray(right_shoulder),
            label="shoulder lateral axis",
        )
        shoulder_midpoint = 0.5 * (
            np.asarray(left_shoulder) + np.asarray(right_shoulder)
        )
        up_hint = normalize_vector(
            shoulder_midpoint - np.asarray(torso_origin),
            label="torso up axis",
        )
        forward = normalize_vector(
            np.cross(lateral, up_hint),
            label="torso forward axis",
        )
        up = normalize_vector(
            np.cross(forward, lateral),
            label="orthogonal torso up axis",
        )
        return np.column_stack((forward, lateral, up))

    def _prepare_shoulder_direction_targets(
        self,
        human_joint_motions: np.ndarray,
    ) -> np.ndarray:
        """Express human upper-arm directions in a torso-local basis."""

        frame_count = int(human_joint_motions.shape[0])
        if not self.shoulder_direction_enabled:
            return np.empty((frame_count, 0, 3), dtype=np.float64)
        torso_origin_index = self.demo_joints.index(
            self.shoulder_human_torso_origin_name,
        )
        left_arm_index = self.demo_joints.index(
            self.shoulder_human_torso_side_names[0],
        )
        right_arm_index = self.demo_joints.index(
            self.shoulder_human_torso_side_names[1],
        )
        upper_targets = np.empty((frame_count, 2, 3), dtype=np.float64)
        for frame in range(frame_count):
            positions = human_joint_motions[frame]
            basis = self._anatomical_basis(
                positions[left_arm_index],
                positions[right_arm_index],
                positions[torso_origin_index],
            )
            for side_index, spec in enumerate(self.shoulder_side_specs):
                arm = positions[self.demo_joints.index(spec.human_arm_name)]
                forearm = positions[
                    self.demo_joints.index(spec.human_forearm_name)
                ]
                upper_targets[frame, side_index] = basis.T @ normalize_vector(
                    forearm - arm,
                    label=f"{spec.human_arm_name} segment at frame {frame}",
                )
        return upper_targets

    def _robot_anatomical_basis_from_forward_data(self) -> np.ndarray:
        """Build the robot torso basis after ``mj_forward`` has been called."""

        left_id = mujoco.mj_name2id(
            self.robot_model,
            mujoco.mjtObj.mjOBJ_BODY,
            self.shoulder_side_specs[0].torso_basis_link_name,
        )
        right_id = mujoco.mj_name2id(
            self.robot_model,
            mujoco.mjtObj.mjOBJ_BODY,
            self.shoulder_side_specs[1].torso_basis_link_name,
        )
        torso_id = mujoco.mj_name2id(
            self.robot_model,
            mujoco.mjtObj.mjOBJ_BODY,
            self.shoulder_torso_link_name,
        )
        return self._anatomical_basis(
            self.robot_data.xpos[left_id],
            self.robot_data.xpos[right_id],
            self.robot_data.xpos[torso_id],
        )

    def _body_position_and_active_jacobian(
        self,
        body_name: str,
        *,
        transform_qdot_to_qvel: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return a body-origin position and qpos Jacobian after FK."""

        body_id = mujoco.mj_name2id(
            self.robot_model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_name,
        )
        jacobian_position = np.zeros(
            (3, self.robot_model.nv),
            dtype=np.float64,
        )
        jacobian_rotation = np.zeros_like(jacobian_position)
        mujoco.mj_jacBody(
            self.robot_model,
            self.robot_data,
            jacobian_position,
            jacobian_rotation,
            body_id,
        )
        jacobian_qpos = jacobian_position @ transform_qdot_to_qvel
        return (
            np.array(self.robot_data.xpos[body_id], dtype=np.float64, copy=True),
            np.array(
                jacobian_qpos[:, self.q_a_indices],
                dtype=np.float64,
                copy=True,
            ),
        )

    def _get_shoulder_direction_data(
        self,
        q: np.ndarray,
        *,
        with_jacobians: bool,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Return torso-local upper-arm directions and shoulder-only Jacobians."""

        if not self.shoulder_direction_enabled:
            empty_directions = np.empty((0, 3), dtype=np.float64)
            if with_jacobians:
                return empty_directions, np.empty(
                    (0, 3, self.nq_a),
                    dtype=np.float64,
                )
            return empty_directions, None
        self.robot_data.qpos[:] = np.asarray(q, dtype=np.float64)
        mujoco.mj_forward(self.robot_model, self.robot_data)
        basis = self._robot_anatomical_basis_from_forward_data()
        transform = (
            self._build_transform_qdot_to_qvel_fast()
            if with_jacobians
            else None
        )
        directions: list[np.ndarray] = []
        jacobians: list[np.ndarray] = []
        for side_index, spec in enumerate(self.shoulder_side_specs):
            anchor_id = mujoco.mj_name2id(
                self.robot_model,
                mujoco.mjtObj.mjOBJ_BODY,
                spec.shoulder_anchor_link_name,
            )
            elbow_id = mujoco.mj_name2id(
                self.robot_model,
                mujoco.mjtObj.mjOBJ_BODY,
                spec.elbow_link_name,
            )
            anchor_position = np.array(
                self.robot_data.xpos[anchor_id],
                dtype=np.float64,
                copy=True,
            )
            elbow_position = np.array(
                self.robot_data.xpos[elbow_id],
                dtype=np.float64,
                copy=True,
            )
            segment = elbow_position - anchor_position
            directions.append(
                basis.T @ normalize_vector(segment, label="robot upper arm"),
            )
            if not with_jacobians:
                continue
            if transform is None:
                raise RuntimeError("Expected the qpos-to-qvel transform")
            _, anchor_jacobian = self._body_position_and_active_jacobian(
                spec.shoulder_anchor_link_name,
                transform_qdot_to_qvel=transform,
            )
            _, elbow_jacobian = self._body_position_and_active_jacobian(
                spec.elbow_link_name,
                transform_qdot_to_qvel=transform,
            )
            direction_jacobian = basis.T @ unit_direction_jacobian(
                segment,
                elbow_jacobian - anchor_jacobian,
            )
            shoulder_only = np.zeros_like(direction_jacobian)
            reduced_indices = self.shoulder_joint_reduced_indices[side_index]
            shoulder_only[:, reduced_indices] = direction_jacobian[
                :,
                reduced_indices,
            ]
            jacobians.append(shoulder_only)
        direction_array = np.asarray(directions, dtype=np.float64).reshape(2, 3)
        if not with_jacobians:
            return direction_array, None
        return (
            direction_array,
            np.asarray(jacobians, dtype=np.float64).reshape(2, 3, self.nq_a),
        )

    def _init_natural_pose_regularization(
        self,
        *,
        natural_pose_joint_positions: dict[str, float] | None,
        natural_pose_weights: dict[str, float] | None,
    ) -> None:
        """Resolve fixed natural joint references into active SQP indices."""

        references = {str(name): float(value) for name, value in (natural_pose_joint_positions or {}).items()}
        weights = {str(name): float(value) for name, value in (natural_pose_weights or {}).items()}
        if set(references) != set(weights):
            missing_references = sorted(set(weights).difference(references))
            missing_weights = sorted(set(references).difference(weights))
            raise ValueError(
                "Natural-pose references and weights must contain exactly the same "
                f"joint names; missing_references={missing_references}, "
                f"missing_weights={missing_weights}",
            )
        if references:
            expected_joint_names = set(self.robot_actuated_joint_names)
            configured_joint_names = set(references)
            missing_joints = sorted(
                expected_joint_names.difference(configured_joint_names),
            )
            unknown_joints = sorted(
                configured_joint_names.difference(expected_joint_names),
            )
            if missing_joints or unknown_joints:
                raise ValueError(
                    "Natural-pose tables must contain every actuated robot joint; "
                    f"missing={missing_joints}, unknown={unknown_joints}",
                )
        invalid_references = {name: value for name, value in references.items() if not np.isfinite(value)}
        invalid_weights = {name: value for name, value in weights.items() if not np.isfinite(value) or value < 0.0}
        if invalid_references:
            raise ValueError(
                f"Natural-pose reference values must be finite: {invalid_references}",
            )
        if invalid_weights:
            raise ValueError(
                f"Natural-pose weights must be finite and non-negative: {invalid_weights}",
            )

        self.natural_pose_configured_joint_names = tuple(weights)
        self.natural_pose_configured_reference_values = np.asarray(
            [references[name] for name in self.natural_pose_configured_joint_names],
            dtype=np.float64,
        )
        self.natural_pose_configured_weight_values = np.asarray(
            [weights[name] for name in self.natural_pose_configured_joint_names],
            dtype=np.float64,
        )

        active_index_by_qpos = {
            int(qpos_address): reduced_index for reduced_index, qpos_address in enumerate(self.q_a_indices)
        }
        tracked_names: list[str] = []
        qpos_addresses: list[int] = []
        reduced_indices: list[int] = []
        reference_values: list[float] = []
        weight_values: list[float] = []
        for joint_name, weight in weights.items():
            if weight == 0.0:
                continue
            joint_id = mujoco.mj_name2id(
                self.robot_model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )
            if joint_id < 0:
                raise ValueError(
                    f"Natural-pose regularization references unknown joint {joint_name!r}",
                )
            joint_type = int(self.robot_model.jnt_type[joint_id])
            if joint_type != int(mujoco.mjtJoint.mjJNT_HINGE):
                raise ValueError(
                    "Natural-pose regularization supports hinge joints only; "
                    f"{joint_name!r} has MuJoCo joint type {joint_type}",
                )
            qpos_address = int(self.robot_model.jnt_qposadr[joint_id])
            if qpos_address not in active_index_by_qpos:
                raise ValueError(
                    f"Natural-pose joint {joint_name!r} at qpos[{qpos_address}] is not "
                    "included in the active optimization variables",
                )
            reference = references[joint_name]
            if bool(self.robot_model.jnt_limited[joint_id]):
                lower, upper = self.robot_model.jnt_range[joint_id]
                if reference < lower or reference > upper:
                    raise ValueError(
                        f"Natural-pose reference for {joint_name!r} ({reference}) lies "
                        f"outside its joint range [{lower}, {upper}]",
                    )
            tracked_names.append(joint_name)
            qpos_addresses.append(qpos_address)
            reduced_indices.append(active_index_by_qpos[qpos_address])
            reference_values.append(reference)
            weight_values.append(weight)

        self.natural_pose_joint_names = tuple(tracked_names)
        self.natural_pose_qpos_addresses = np.asarray(qpos_addresses, dtype=np.int32)
        self.natural_pose_reduced_indices = np.asarray(reduced_indices, dtype=np.int32)
        self.natural_pose_reference_values = np.asarray(
            reference_values,
            dtype=np.float64,
        )
        self.natural_pose_weight_values = np.asarray(weight_values, dtype=np.float64)
        self.natural_pose_tracking_enabled = bool(tracked_names)

    def apply_natural_pose_to_initial_qpos(self, qpos: np.ndarray) -> np.ndarray:
        """Return one sequence-initial qpos with active natural joints set."""

        initial_qpos = np.asarray(qpos, dtype=np.float64).copy()
        if initial_qpos.ndim != 1 or initial_qpos.shape[0] < self.robot_model.nq:
            raise ValueError(
                "Initial qpos must be a full one-dimensional MuJoCo configuration; "
                f"got {initial_qpos.shape}, expected at least ({self.robot_model.nq},)",
            )
        if self.natural_pose_tracking_enabled:
            initial_qpos[self.natural_pose_qpos_addresses] = self.natural_pose_reference_values
        return initial_qpos

    def _init_orientation_tracking(
        self,
        *,
        orientation_joints_mapping: dict[str, str] | None,
        orientation_weights: dict[str, float] | None,
        orientation_preview: bool,
        orientation_alignment_mode: str,
        orientation_t_pose_human_quaternions_wxyz: dict[
            str,
            tuple[float, float, float, float],
        ]
        | None,
        orientation_t_pose_robot_base_quaternion_wxyz: tuple[
            float,
            float,
            float,
            float,
        ]
        | None,
        orientation_t_pose_robot_joint_positions: dict[str, float] | None,
        orientation_alignment_quaternions_wxyz: dict[
            str,
            tuple[float, float, float, float],
        ]
        | None,
    ) -> None:
        """Validate and cache the independent SO(3) tracking configuration."""
        mapping = dict(orientation_joints_mapping or {})
        weights = {name: float(weight) for name, weight in (orientation_weights or {}).items()}
        if orientation_preview:
            for name in mapping:
                weights.setdefault(name, 0.0)
        unknown_weights = sorted(set(weights) - set(mapping))
        if unknown_weights:
            raise ValueError(f"Orientation weights reference joints outside the orientation mapping: {unknown_weights}")
        invalid_weights = {name: weight for name, weight in weights.items() if not np.isfinite(weight) or weight < 0.0}
        if invalid_weights:
            raise ValueError(f"Orientation weights must be finite and non-negative: {invalid_weights}")
        if orientation_alignment_mode not in {
            "t_pose",
            "first_frame",
            "explicit",
        }:
            raise ValueError("orientation_alignment_mode must be 't_pose', 'first_frame', or 'explicit'")

        tracked_human_joints = [name for name in mapping if name in weights]
        missing_human_joints = [name for name in tracked_human_joints if name not in self.demo_joints]
        if missing_human_joints:
            raise ValueError(f"Orientation mapping references unknown human joints: {missing_human_joints}")
        tracked_robot_links = [mapping[name] for name in tracked_human_joints]
        missing_robot_links = [
            link_name
            for link_name in tracked_robot_links
            if mujoco.mj_name2id(
                self.robot_model,
                mujoco.mjtObj.mjOBJ_BODY,
                link_name,
            )
            < 0
        ]
        if missing_robot_links:
            raise ValueError(f"Orientation mapping references unknown MuJoCo bodies: {missing_robot_links}")

        explicit_alignments = dict(orientation_alignment_quaternions_wxyz or {})
        t_pose_human_quaternions = dict(orientation_t_pose_human_quaternions_wxyz or {})
        t_pose_robot_joint_positions = {
            name: float(value) for name, value in (orientation_t_pose_robot_joint_positions or {}).items()
        }
        if orientation_alignment_mode == "t_pose" and tracked_human_joints:
            missing_t_pose_frames = [name for name in tracked_human_joints if name not in t_pose_human_quaternions]
            if missing_t_pose_frames:
                raise ValueError(
                    f"T-pose orientation alignment is missing source-human frames: {missing_t_pose_frames}"
                )
            for name in tracked_human_joints:
                quaternion = np.asarray(
                    t_pose_human_quaternions[name],
                    dtype=np.float64,
                )
                if quaternion.shape != (4,) or not np.isfinite(quaternion).all() or np.linalg.norm(quaternion) <= 1e-8:
                    raise ValueError(f"T-pose source-human quaternion for {name!r} must be finite, non-zero wxyz")
            robot_base_quaternion = np.asarray(
                orientation_t_pose_robot_base_quaternion_wxyz,
                dtype=np.float64,
            )
            if (
                robot_base_quaternion.shape != (4,)
                or not np.isfinite(robot_base_quaternion).all()
                or np.linalg.norm(robot_base_quaternion) <= 1e-8
            ):
                raise ValueError("T-pose robot base quaternion must be finite, non-zero wxyz")
            robot_base_quaternion /= np.linalg.norm(robot_base_quaternion)

            scalar_joint_types = {
                int(mujoco.mjtJoint.mjJNT_SLIDE),
                int(mujoco.mjtJoint.mjJNT_HINGE),
            }
            for joint_name, joint_position in t_pose_robot_joint_positions.items():
                joint_id = mujoco.mj_name2id(
                    self.robot_model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    joint_name,
                )
                if joint_id < 0:
                    raise ValueError(f"T-pose robot joint {joint_name!r} does not exist in the MuJoCo model")
                if int(self.robot_model.jnt_type[joint_id]) not in scalar_joint_types:
                    raise ValueError(f"T-pose robot joint {joint_name!r} must be a scalar hinge or slide joint")
                if not np.isfinite(joint_position):
                    raise ValueError(f"T-pose robot joint position for {joint_name!r} must be finite")
                if self.robot_model.jnt_limited[joint_id]:
                    lower, upper = self.robot_model.jnt_range[joint_id]
                    if joint_position < lower - 1e-9 or joint_position > upper + 1e-9:
                        raise ValueError(
                            f"T-pose robot joint position for {joint_name!r}={joint_position} "
                            f"is outside [{lower}, {upper}]"
                        )
        else:
            robot_base_quaternion = np.empty((0,), dtype=np.float64)
        if orientation_alignment_mode == "explicit":
            missing_alignments = [name for name in tracked_human_joints if name not in explicit_alignments]
            if missing_alignments:
                raise ValueError(f"Explicit orientation alignment is missing joints: {missing_alignments}")
            for name in tracked_human_joints:
                quaternion = np.asarray(
                    explicit_alignments[name],
                    dtype=np.float64,
                )
                if quaternion.shape != (4,) or not np.isfinite(quaternion).all() or np.linalg.norm(quaternion) <= 1e-8:
                    raise ValueError(
                        f"Explicit orientation alignment quaternion for {name!r} must be finite, non-zero wxyz"
                    )

        self.orientation_joints_mapping = mapping
        self.orientation_weights_by_human_joint = weights
        self.orientation_alignment_mode = orientation_alignment_mode
        self.orientation_t_pose_human_quaternions_wxyz = t_pose_human_quaternions
        self.orientation_t_pose_robot_base_quaternion_wxyz = robot_base_quaternion
        self.orientation_t_pose_robot_joint_positions = t_pose_robot_joint_positions
        self.orientation_alignment_quaternions_wxyz_config = explicit_alignments
        self.orientation_human_joint_names = tracked_human_joints
        self.orientation_robot_link_names = tracked_robot_links
        self.orientation_human_joint_indices = [self.demo_joints.index(name) for name in tracked_human_joints]
        self.orientation_weight_values = np.asarray(
            [weights[name] for name in tracked_human_joints],
            dtype=np.float64,
        )
        self.orientation_tracking_enabled = bool(np.any(self.orientation_weight_values > 0.0))
        self.orientation_preview_enabled = bool(orientation_preview and tracked_human_joints)
        self.orientation_diagnostics_enabled = bool(tracked_human_joints) and not (
            self.orientation_preview_enabled and not self.orientation_tracking_enabled
        )
        self.orientation_reference_human_matrices = np.empty(
            (0, 3, 3),
            dtype=np.float64,
        )
        self.orientation_reference_robot_matrices = np.empty(
            (0, 3, 3),
            dtype=np.float64,
        )
        self.orientation_reference_robot_qpos = np.empty(
            (0,),
            dtype=np.float64,
        )

    def _init_planar_foot_contact(
        self,
        planar_foot_contact: PlanarFootContactConfig | None,
    ) -> None:
        """Validate the semantic sole layout and initialize phase anchors."""
        self.planar_foot_contact = planar_foot_contact or PlanarFootContactConfig()
        constraint_values = {
            "heading_tolerance": self.planar_foot_contact.heading_tolerance,
            "slide_tracking_tolerance": self.planar_foot_contact.slide_tracking_tolerance,
        }
        invalid = {name: value for name, value in constraint_values.items() if not np.isfinite(value) or value < 0.0}
        if invalid:
            raise ValueError(f"Planar foot-contact constraint tolerances must be finite and non-negative: {invalid}")

        layout = getattr(self.task_constants, "FOOT_CONTACT_LINKS", None)
        required_regions = {
            "heel_positive_lateral",
            "heel_negative_lateral",
            "forefoot_positive_lateral",
            "forefoot_negative_lateral",
            "toe",
        }
        if layout is None:
            legacy_by_side = {
                side: [
                    str(name) for name in self.task_constants.FOOT_STICKING_LINKS if self._name_side(str(name)) == side
                ]
                for side in ("left", "right")
            }
            layout = {}
            for side, links in legacy_by_side.items():
                if len(links) < 4:
                    raise ValueError(
                        "Legacy FOOT_STICKING_LINKS must provide at least four links per foot",
                    )
                layout[side] = {
                    "heel_positive_lateral": links[0],
                    "heel_negative_lateral": links[1],
                    "forefoot_positive_lateral": links[2],
                    "forefoot_negative_lateral": links[3],
                    "toe": links[4] if len(links) >= 5 else links[2],
                }

        normalized_layout: dict[str, dict[str, str]] = {}
        for side in ("left", "right"):
            if side not in layout:
                raise ValueError(f"FOOT_CONTACT_LINKS is missing the {side!r} sole")
            regions = {str(region): str(link) for region, link in layout[side].items()}
            missing_regions = sorted(required_regions.difference(regions))
            if missing_regions:
                raise ValueError(
                    f"FOOT_CONTACT_LINKS[{side!r}] is missing semantic regions: {missing_regions}",
                )
            normalized_layout[side] = regions
        self.foot_contact_link_layout = normalized_layout
        link_names = tuple(
            dict.fromkeys(link for side in ("left", "right") for link in normalized_layout[side].values()),
        )
        self.foot_links = {name: name for name in link_names}
        self._foot_contact_phase_anchors: dict[tuple[str, int], _ResolvedPlanarFootTarget] = {}
        self._active_planar_foot_targets: dict[str, _ResolvedPlanarFootTarget] | None = None

    def _init_foot_lock(self, foot_lock: FootLockConfig | None) -> None:
        """Initialize foot lock configuration and normalize window mappings."""
        self.foot_lock = foot_lock or FootLockConfig()
        self._foot_lock_windows: dict[str, tuple[tuple[int, int], ...]] = {"left": (), "right": ()}
        if self.foot_lock.windows is None:
            return
        for key, windows in self.foot_lock.windows.items():
            side = self._name_side(key)
            if side is None:
                continue

            normalized_windows: list[tuple[int, int]] = []
            for window in windows:
                if len(window) != 2:
                    raise ValueError(f"Invalid foot lock window for {key}: {window}")
                start, end = int(window[0]), int(window[1])
                if end < start:
                    raise ValueError(f"Invalid foot lock window with end < start for {key}: {window}")
                normalized_windows.append((start, end))
            self._foot_lock_windows[side] = tuple(normalized_windows)

    @classmethod
    def _foot_sticking_states_array(
        cls,
        foot_sticking_sequences,
        num_frames: int,
    ) -> np.ndarray:
        """Pack per-frame source-joint dictionaries into canonical [left, right] order."""
        if len(foot_sticking_sequences) != num_frames:
            raise ValueError(
                "foot_sticking_sequences must contain one entry per motion frame; "
                f"got {len(foot_sticking_sequences)} entries for {num_frames} frames"
            )

        states = np.zeros((num_frames, 2), dtype=bool)
        for frame_idx, frame_state in enumerate(foot_sticking_sequences):
            side_values: dict[str, bool] = {}
            for key in frame_state:
                side = cls._name_side(key)
                if side is not None:
                    side_values[side] = bool(frame_state[key])
            missing = [side for side in ("left", "right") if side not in side_values]
            if missing:
                raise ValueError(f"foot_sticking_sequences[{frame_idx}] is missing side(s): " + ", ".join(missing))
            states[frame_idx] = (side_values["left"], side_values["right"])
        return states

    @staticmethod
    def _name_side(name: str) -> str | None:
        """Infer left/right side from common robot or mocap naming conventions."""
        normalized = name.lower().replace("-", "_")
        if normalized.startswith(("left", "l_")) or "_left" in normalized:
            return "left"
        if normalized.startswith(("right", "r_")) or "_right" in normalized:
            return "right"
        return None

    def _semantic_sole_point(
        self,
        side: str,
        region: str,
        jacobians: Mapping[str, np.ndarray],
        positions: Mapping[str, np.ndarray],
        *,
        pivot_uv: tuple[float, float] = (0.5, 0.0),
    ) -> tuple[np.ndarray, np.ndarray]:
        """Aggregate semantic robot sole links into one point and Jacobian."""
        layout = self.foot_contact_link_layout[side]

        def named(name: str) -> tuple[np.ndarray, np.ndarray]:
            link = layout[name]
            return jacobians[link], positions[link]

        J_hp, p_hp = named("heel_positive_lateral")
        J_hn, p_hn = named("heel_negative_lateral")
        J_fp, p_fp = named("forefoot_positive_lateral")
        J_fn, p_fn = named("forefoot_negative_lateral")
        J_toe, p_toe = named("toe")
        J_heel = 0.5 * (J_hp + J_hn)
        p_heel = 0.5 * (p_hp + p_hn)
        J_forefoot = 0.5 * (J_fp + J_fn)
        p_forefoot = 0.5 * (p_fp + p_fn)

        if region == "heel":
            return J_heel, p_heel
        if region == "forefoot":
            return J_forefoot, p_forefoot
        if region == "toe":
            return J_toe, p_toe
        if region == "center":
            return 0.5 * (J_heel + J_forefoot), 0.5 * (p_heel + p_forefoot)
        if region != "pivot":
            raise ValueError(f"Unknown semantic sole region: {region!r}")

        u = float(np.clip(pivot_uv[0], -0.2, 1.2))
        v = float(np.clip(pivot_uv[1], -1.5, 1.5))
        J_base = J_heel + u * (J_toe - J_heel)
        p_base = p_heel + u * (p_toe - p_heel)
        heel_half_width_J = 0.5 * (J_hp - J_hn)
        heel_half_width_p = 0.5 * (p_hp - p_hn)
        forefoot_half_width_J = 0.5 * (J_fp - J_fn)
        forefoot_half_width_p = 0.5 * (p_fp - p_fn)
        if u <= 0.8:
            blend = float(np.clip(u / 0.8, 0.0, 1.0))
            half_width_J = (1.0 - blend) * heel_half_width_J + blend * forefoot_half_width_J
            half_width_p = (1.0 - blend) * heel_half_width_p + blend * forefoot_half_width_p
        else:
            toe_width_scale = float(np.clip((1.0 - u) / 0.2, 0.0, 1.0))
            half_width_J = toe_width_scale * forefoot_half_width_J
            half_width_p = toe_width_scale * forefoot_half_width_p
        return J_base + v * half_width_J, p_base + v * half_width_p

    def _prepare_planar_foot_targets(
        self,
        q_reference: np.ndarray,
        frame_state: Mapping[str, FootContactState],
    ) -> None:
        """Resolve persistent phase anchors from the previous accepted robot pose."""
        unknown_sides = sorted(set(frame_state).difference({"left", "right"}))
        if unknown_sides:
            raise ValueError(f"Foot contact state contains unknown sides: {unknown_sides}")
        missing_sides = [side for side in ("left", "right") if side not in frame_state]
        if missing_sides:
            raise ValueError(f"Foot contact state is missing sides: {missing_sides}")

        active_states = {side: state for side, state in frame_state.items() if state.mode != "swing"}
        if not active_states:
            self._active_planar_foot_targets = {}
            return
        jacobians, positions, _ = self._calc_manipulator_jacobians(
            q_reference,
            links=self.foot_links,
            obj_frame=False,
        )
        active_targets: dict[str, _ResolvedPlanarFootTarget] = {}
        for side, state in active_states.items():
            if state.phase_id < 0:
                raise ValueError(f"Active {side} foot contact must have a non-negative phase_id")
            cache_key = (side, state.phase_id)
            cached = self._foot_contact_phase_anchors.get(cache_key)
            if cached is None:
                if state.mode == "flat":
                    _, p_center = self._semantic_sole_point(
                        side,
                        "center",
                        jacobians,
                        positions,
                    )
                    _, p_heel = self._semantic_sole_point(side, "heel", jacobians, positions)
                    _, p_forefoot = self._semantic_sole_point(side, "forefoot", jacobians, positions)
                    cached = _ResolvedPlanarFootTarget(
                        side=side,
                        mode=state.mode,
                        phase_id=state.phase_id,
                        pivot_uv=state.pivot_uv,
                        anchor_xy=p_center[:2].copy(),
                        heading_anchor_xy=(p_forefoot - p_heel)[:2].copy(),
                    )
                else:
                    if state.mode == "heel":
                        region = "heel"
                    elif state.mode == "toe":
                        region = "toe"
                    elif state.mode == "pivot":
                        region = "pivot"
                    elif state.mode == "slide":
                        region = "center"
                    else:
                        raise ValueError(f"Unknown planar foot contact mode: {state.mode!r}")
                    _, point = self._semantic_sole_point(
                        side,
                        region,
                        jacobians,
                        positions,
                        pivot_uv=state.pivot_uv,
                    )
                    cached = _ResolvedPlanarFootTarget(
                        side=side,
                        mode=state.mode,
                        phase_id=state.phase_id,
                        pivot_uv=state.pivot_uv,
                        anchor_xy=point[:2].copy(),
                    )
                self._foot_contact_phase_anchors[cache_key] = cached

            displacement = (
                np.asarray(state.reference_displacement_xy, dtype=np.float64)
                if state.mode == "slide"
                else np.zeros(2, dtype=np.float64)
            )
            active_targets[side] = _ResolvedPlanarFootTarget(
                side=cached.side,
                mode=cached.mode,
                phase_id=cached.phase_id,
                pivot_uv=cached.pivot_uv,
                anchor_xy=cached.anchor_xy + displacement,
                heading_anchor_xy=cached.heading_anchor_xy,
            )
        self._active_planar_foot_targets = active_targets

    def _append_contact_plan_constraints(
        self,
        constraints: list,
        dqa: cp.Variable,
        jacobians: Mapping[str, np.ndarray],
        positions: Mapping[str, np.ndarray],
    ) -> None:
        """Append the minimal independent planar constraints for active modes."""
        for side, target in (self._active_planar_foot_targets or {}).items():
            if target.mode == "flat":
                J_center, p_center = self._semantic_sole_point(
                    side,
                    "center",
                    jacobians,
                    positions,
                )
                delta = target.anchor_xy - p_center[:2]
                constraints.extend(
                    [
                        J_center[:2] @ dqa >= delta - self.foot_sticking_tolerance,
                        J_center[:2] @ dqa <= delta + self.foot_sticking_tolerance,
                    ],
                )
                J_heel, p_heel = self._semantic_sole_point(side, "heel", jacobians, positions)
                J_forefoot, p_forefoot = self._semantic_sole_point(
                    side,
                    "forefoot",
                    jacobians,
                    positions,
                )
                heading_anchor = np.asarray(target.heading_anchor_xy, dtype=np.float64)
                heading_norm = float(np.linalg.norm(heading_anchor))
                if heading_norm <= 1e-8:
                    raise RuntimeError(f"Degenerate flat-foot heading anchor for the {side} foot")
                heading_normal = np.asarray([-heading_anchor[1], heading_anchor[0]]) / heading_norm
                heading_now = (p_forefoot - p_heel)[:2]
                J_heading = (J_forefoot - J_heel)[:2]
                heading_delta = -float(heading_normal @ (heading_now - heading_anchor))
                heading_row = heading_normal @ J_heading
                constraints.extend(
                    [
                        heading_row @ dqa >= heading_delta - self.planar_foot_contact.heading_tolerance,
                        heading_row @ dqa <= heading_delta + self.planar_foot_contact.heading_tolerance,
                    ],
                )
                continue

            if target.mode == "heel":
                region = "heel"
            elif target.mode == "toe":
                region = "toe"
            elif target.mode == "pivot":
                region = "pivot"
            elif target.mode == "slide":
                region = "center"
            else:
                raise ValueError(f"Unknown planar foot contact mode: {target.mode!r}")
            J_point, p_point = self._semantic_sole_point(
                side,
                region,
                jacobians,
                positions,
                pivot_uv=target.pivot_uv,
            )
            tolerance = (
                self.planar_foot_contact.slide_tracking_tolerance
                if target.mode == "slide"
                else self.foot_sticking_tolerance
            )
            delta = target.anchor_xy - p_point[:2]
            constraints.extend(
                [
                    J_point[:2] @ dqa >= delta - tolerance,
                    J_point[:2] @ dqa <= delta + tolerance,
                ],
            )

    def _init_self_collision(self, self_collision: SelfCollisionConfig | None) -> None:
        """Initialize self-collision configuration and precompute geom pairs."""
        sc = self_collision or SelfCollisionConfig()
        self._self_collision_enabled = sc.enable and len(sc.pairs) > 0
        self._self_collision_tolerance = sc.tolerance
        self._self_collision_windows: list[tuple[int, int]] | None = sc.windows
        self._self_collision_geom_pairs: list[tuple[int, int]] = []

        self._sc_last_vis_frame = -1

        if not self._self_collision_enabled:
            return

        m = self.robot_model

        # Build body_name → [geom_ids] mapping (only geoms with collision enabled)
        body_to_geoms: dict[str, list[int]] = {}
        for g in range(m.ngeom):
            if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
                continue
            body_id = m.geom_bodyid[g]
            body_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            body_to_geoms.setdefault(body_name, []).append(g)

        # Build geom pairs from body name pairs
        for body_a, body_b in sc.pairs:
            geoms_a = body_to_geoms.get(body_a, [])
            geoms_b = body_to_geoms.get(body_b, [])
            if not geoms_a:
                print(f"[SelfCollision] Warning: no collision geoms found for body '{body_a}'")
            if not geoms_b:
                print(f"[SelfCollision] Warning: no collision geoms found for body '{body_b}'")
            for ga in geoms_a:
                for gb in geoms_b:
                    self._self_collision_geom_pairs.append((ga, gb))

        print(
            f"[SelfCollision] Initialized with {len(self._self_collision_geom_pairs)} geom pairs "
            f"from {len(sc.pairs)} body pairs, tolerance={sc.tolerance}m"
        )

    def _setup_visualization(self):
        """Setup Viser visualization components."""
        self.server = viser.ViserServer()
        mesh_color_override = self._mesh_color_override()

        # 1) Ensure a world frame exists (absolute path!)
        try:
            self.server.scene.add_frame("/world", show_axes=False)
        except Exception:
            print("Starting viser")

        # Create parent frames for robot and object
        self.robot_base = self.server.scene.add_frame("/world/robot", show_axes=False)

        print("robot_model_path: ", self.robot_model_path)

        # Load robot URDF
        self.robot_urdf = yourdfpy.URDF.load(
            self.robot_model_path,
            load_meshes=True,
            build_scene_graph=True,
        )

        print("Viser using robot URDF: ", self.robot_model_path)

        # Create ViserUrdf instance for robot, attaching it to the robot_base frame
        self.viser_robot = ViserUrdf(
            self.server,
            urdf_or_path=self.robot_urdf,
            root_node_name="/world/robot",  # This links to the robot_base frame we created
            mesh_color_override=mesh_color_override,
        )

        # Similarly for object
        if self.object_model_path:
            self.object_base = self.server.scene.add_frame("/world/object", show_axes=False)

            self.object_urdf = yourdfpy.URDF.load(
                self.object_model_path,
                load_meshes=True,
                build_scene_graph=True,
            )

            # Create ViserUrdf instance for object, attaching it to the object_base frame
            self.viser_object = ViserUrdf(
                self.server,
                urdf_or_path=self.object_urdf,
                root_node_name="/world/object",  # This links to the object_base frame we created
                mesh_color_override=mesh_color_override,
            )
            print("Viser using object URDF: ", self.object_model_path)

        else:
            self.viser_object = None

        # Check the number of actuated joints and their names
        robot_joint_limits = self.viser_robot.get_actuated_joint_limits()
        print("\nRobot joints:")
        print("Number of actuated joints:", len(robot_joint_limits))
        print("Joint names:", list(robot_joint_limits.keys()))
        mujoco_joint_names = actuated_joint_names_from_mujoco_xml(self.robot_xml_path)
        viser_joint_names = list(robot_joint_limits.keys())
        if mujoco_joint_names != viser_joint_names:
            self.qpos_to_viser_joint_indices = build_joint_order_indices(mujoco_joint_names, viser_joint_names)
            print("Qpos-to-viser joint order mapping:", self.qpos_to_viser_joint_indices.tolist())

        # Initialize robot with this configuration
        robot_initial_config = np.zeros(len(robot_joint_limits))
        self.viser_robot.update_cfg(robot_initial_config)

        # Add grid
        self.server.scene.add_grid(
            "/world/grid",
            width=8,
            height=8,
            position=(0.0, 0.0, 0.0),
        )
        with self.server.gui.add_folder("Foot sticking"):
            self._foot_sticking_status_handle = self.server.gui.add_markdown(
                format_foot_sticking_status(
                    0,
                    (False, False),
                    constraint_status="waiting for first frame",
                )
            )

    def _update_foot_sticking_status(
        self,
        frame_idx: int,
        states: np.ndarray,
        *,
        constraints_enabled: bool | None = None,
    ) -> None:
        """Update live/replay detector lights and actual constraint status."""
        if not self.visualize or not hasattr(self, "_foot_sticking_status_handle"):
            return

        enabled = self.activate_foot_sticking if constraints_enabled is None else constraints_enabled
        if not enabled:
            constraint_status = "disabled"
        elif self.q_a_init_idx >= 12:
            constraint_status = "not applied for current optimization variables"
        elif bool(np.any(states)):
            constraint_status = "active"
        else:
            constraint_status = "inactive (no sticking foot)"

        self._foot_sticking_status_handle.content = format_foot_sticking_status(
            frame_idx,
            states,
            constraint_status=constraint_status,
        )

    def _update_live_visualization_frame(
        self,
        q: np.ndarray,
        *,
        frame_idx: int,
        foot_sticking_state: np.ndarray,
    ) -> None:
        """Advance the live robot independently of optional debug overlays."""
        if not self.visualize:
            return
        self._update_foot_sticking_status(
            frame_idx,
            foot_sticking_state,
        )
        self.draw_q(q)

    def _mesh_color_override(self):
        opacity = float(np.clip(self.mesh_opacity, 0.0, 1.0))
        if opacity >= 0.999:
            return None
        return (0.7, 0.7, 0.7, opacity)

    def draw_mesh_from_geom(self, model, data, geom_id, geom_name, name="/mesh", color=(50, 150, 255), opacity=0.5):
        """
        Draw a single MuJoCo mesh geom (already baked to world coords) in viser.
        color is [0, 255] RGB ints; opacity is [0,1].
        """
        if not hasattr(self, "server"):
            return
        V, F = _world_mesh_from_geom(model, data, geom_id, geom_name)
        self.server.scene.add_mesh_simple(
            name,
            vertices=V.astype(np.float32),
            faces=F.astype(np.int32),
            position=(0.0, 0.0, 0.0),  # already world-frame
            color=tuple(int(c) for c in color),
            opacity=float(opacity),
        )

    def draw_mesh_pair_with_contact(
        self,
        model,
        data,
        geom_id1,
        geom_id2,
        geom1_name,
        geom2_name,
        fromto=None,
        group_name="pair",
        color1=(50, 150, 255),
        color2=(255, 120, 60),
        opacity=0.45,
        show_segment=True,
    ):
        """
        Draw two meshes and (optionally) a contact/query segment.
        Uses the existing self.draw_keypoints(...) to visualize points.
        """
        # Note: sometime geom does not have mesh, mesh_id will be -1
        if int(model.geom_dataid[geom_id1]) == -1 or int(model.geom_dataid[geom_id2]) == -1:
            return

        base = f"/{group_name}"
        # meshes
        self.draw_mesh_from_geom(model, data, geom_id1, geom1_name, name=f"{base}/mesh1", color=color1, opacity=opacity)
        self.draw_mesh_from_geom(model, data, geom_id2, geom2_name, name=f"{base}/mesh2", color=color2, opacity=opacity)

        # contact points (q: green, c: red) via your draw_keypoints
        if fromto is not None:
            q = np.asarray(fromto[:3], dtype=float)
            c = np.asarray(fromto[3:], dtype=float)

            # your existing helper (rgba expects floats 0..1)
            self.draw_keypoints(q, name=f"{group_name}_q", rgba=(0.0, 1.0, 0.0, 1.0))
            self.draw_keypoints(c, name=f"{group_name}_c", rgba=(1.0, 0.0, 0.0, 1.0))

    @staticmethod
    def _pack_interaction_tetrahedra(tetrahedra_list: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Pack variable-length per-frame tetrahedra into a padded int array."""
        counts = np.asarray([len(tets) for tets in tetrahedra_list], dtype=np.int32)
        if counts.size == 0:
            return np.empty((0, 0, 4), dtype=np.int32), counts

        max_count = int(np.max(counts))
        packed = np.full((len(tetrahedra_list), max_count, 4), -1, dtype=np.int32)
        for frame_idx, tets in enumerate(tetrahedra_list):
            frame_tets = np.asarray(tets, dtype=np.int32)
            packed[frame_idx, : frame_tets.shape[0], :] = frame_tets
        return packed, counts

    @staticmethod
    def _subset_parent_indices(
        parent_indices: np.ndarray,
        kept_indices: list[int],
    ) -> np.ndarray:
        """Reconnect a saved skeleton through the nearest retained ancestors."""

        parents = np.asarray(parent_indices, dtype=np.int32)
        old_to_new = {old: new for new, old in enumerate(kept_indices)}
        compact: list[int] = []
        for old_index in kept_indices:
            parent = int(parents[old_index])
            visited: set[int] = set()
            while parent >= 0 and parent not in old_to_new:
                if parent in visited:
                    parent = -1
                    break
                visited.add(parent)
                parent = int(parents[parent])
            compact.append(old_to_new.get(parent, -1))
        return np.asarray(compact, dtype=np.int32)

    def draw_interaction_mesh(
        self,
        vertices: np.ndarray,
        tetrahedra: np.ndarray,
        name: str,
        color: tuple[float, float, float] = (1.0, 0.55, 0.0),
        edge_mode: str | None = None,
    ) -> list[object]:
        """Draw interaction mesh edges as a Viser line-segment overlay."""
        if not hasattr(self, "server"):
            return []

        edges = interaction_mesh_edges_from_tetrahedra(
            tetrahedra,
            num_anchor_vertices=len(self.laplacian_match_links),
            edge_mode=edge_mode or self.interaction_mesh_edges,
        )
        if edges.size == 0:
            return []

        vertices = np.asarray(vertices, dtype=np.float32)
        segments = vertices[edges]
        colors = np.tile(np.asarray(color, dtype=np.float32).reshape(1, 1, 3), (segments.shape[0], 2, 1))
        handle = self.server.scene.add_line_segments(
            f"/{name}",
            points=segments,
            colors=colors,
            line_width=self.interaction_mesh_line_width,
        )
        return [handle]

    @staticmethod
    def _object_points_save_payload(
        *,
        has_dynamic_object: bool,
        object_points_local_demo: np.ndarray,
        object_points_local: np.ndarray,
        obj_pts_demo_list: list[np.ndarray],
        obj_pts_list: list[np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Return object-point trajectories only for genuine dynamic objects."""
        if not has_dynamic_object:
            return {}
        return {
            "object_points_demo_local": np.asarray(object_points_local_demo, dtype=np.float32),
            "object_points_target_local": np.asarray(object_points_local, dtype=np.float32),
            "object_points_demo_world": np.asarray(obj_pts_demo_list, dtype=np.float32),
            "object_points_target_world": np.asarray(obj_pts_list, dtype=np.float32),
        }

    @staticmethod
    def _ground_points_at_robot_root(
        ground_points: np.ndarray,
        q: np.ndarray,
    ) -> np.ndarray:
        """Translate a ground-point template to the robot root in XY."""
        return np.asarray(ground_points) + np.asarray([q[0], q[1], 0.0])

    def _apply_dynamic_object_poses(
        self,
        q_locked_list: np.ndarray,
        object_poses_augmented: np.ndarray,
    ) -> None:
        """Write object free-joint poses only when the model contains one."""

        if not self.has_dynamic_object:
            return
        object_poses_array = np.asarray(
            object_poses_augmented,
            dtype=np.float64,
        )
        expected_shape = (q_locked_list.shape[0], 7)
        if object_poses_array.shape != expected_shape:
            raise ValueError(f"Dynamic-object poses must have shape {expected_shape}, got {object_poses_array.shape}")
        q_locked_list[:, -7:] = object_poses_array

    @staticmethod
    def _wxyz_to_matrices(quaternions_wxyz: np.ndarray) -> np.ndarray:
        quaternions = np.asarray(quaternions_wxyz, dtype=np.float64)
        if quaternions.shape[-1] != 4:
            raise ValueError(f"Expected wxyz quaternions with final dimension 4, got {quaternions.shape}")
        if quaternions.size == 0:
            return np.empty((*quaternions.shape[:-1], 3, 3), dtype=np.float64)
        norms = np.linalg.norm(quaternions, axis=-1)
        if not np.isfinite(quaternions).all() or np.any(norms <= 1e-8):
            raise ValueError("Orientation quaternions must be finite and non-zero")
        normalized = quaternions / norms[..., None]
        flat = normalized.reshape(-1, 4)
        matrices = Rotation.from_quat(flat[:, [1, 2, 3, 0]]).as_matrix()
        return matrices.reshape(*quaternions.shape[:-1], 3, 3)

    @staticmethod
    def _matrices_to_wxyz(matrices: np.ndarray) -> np.ndarray:
        matrices = np.asarray(matrices, dtype=np.float64)
        if matrices.shape[-2:] != (3, 3):
            raise ValueError(f"Expected rotation matrices with shape (..., 3, 3), got {matrices.shape}")
        if matrices.size == 0:
            return np.empty((*matrices.shape[:-2], 4), dtype=np.float64)
        xyzw = Rotation.from_matrix(matrices.reshape(-1, 3, 3)).as_quat()
        wxyz = xyzw[:, [3, 0, 1, 2]]
        return wxyz.reshape(*matrices.shape[:-2], 4)

    @staticmethod
    def _so3_error_vectors(
        target_matrices: np.ndarray,
        current_matrices: np.ndarray,
    ) -> np.ndarray:
        """Return world-frame Log(R_target R_current^T) vectors."""
        target_matrices = np.asarray(target_matrices, dtype=np.float64)
        current_matrices = np.asarray(current_matrices, dtype=np.float64)
        if target_matrices.shape != current_matrices.shape:
            raise ValueError(
                "Target and current orientation matrices must have equal shape, "
                f"got {target_matrices.shape} and {current_matrices.shape}"
            )
        if target_matrices.size == 0:
            return np.empty((*target_matrices.shape[:-2], 3), dtype=np.float64)
        relative = target_matrices @ np.swapaxes(current_matrices, -1, -2)
        return Rotation.from_matrix(relative.reshape(-1, 3, 3)).as_rotvec().reshape(*relative.shape[:-2], 3)

    def _get_robot_link_orientation_data(
        self,
        q: np.ndarray,
        link_names: list[str],
        *,
        with_jacobians: bool,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Return world orientations and optional world angular Jacobians."""
        self.robot_data.qpos[:] = np.asarray(q, dtype=np.float64)
        mujoco.mj_forward(self.robot_model, self.robot_data)
        transform_qdot_to_qvel = self._build_transform_qdot_to_qvel_fast() if with_jacobians else None
        matrices: list[np.ndarray] = []
        jacobians: list[np.ndarray] = []
        for link_name in link_names:
            body_id = mujoco.mj_name2id(
                self.robot_model,
                mujoco.mjtObj.mjOBJ_BODY,
                link_name,
            )
            if body_id < 0:
                raise ValueError(f"Body {link_name!r} not found in MuJoCo model")
            matrices.append(
                np.array(
                    self.robot_data.xmat[body_id].reshape(3, 3),
                    dtype=np.float64,
                    copy=True,
                )
            )
            if with_jacobians:
                jacobian_position = np.zeros(
                    (3, self.robot_model.nv),
                    dtype=np.float64,
                )
                jacobian_rotation = np.zeros(
                    (3, self.robot_model.nv),
                    dtype=np.float64,
                )
                mujoco.mj_jacBody(
                    self.robot_model,
                    self.robot_data,
                    jacobian_position,
                    jacobian_rotation,
                    body_id,
                )
                jacobian_qpos = jacobian_rotation @ transform_qdot_to_qvel
                jacobians.append(
                    np.array(
                        jacobian_qpos[:, self.q_a_indices],
                        dtype=np.float64,
                        copy=True,
                    )
                )
        matrix_array = np.asarray(matrices, dtype=np.float64).reshape(-1, 3, 3)
        if not with_jacobians:
            return matrix_array, None
        return (
            matrix_array,
            np.asarray(jacobians, dtype=np.float64).reshape(
                -1,
                3,
                self.nq_a,
            ),
        )

    def _robot_link_topology(
        self,
    ) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
        """Return the robot root subtree, excluding world and scene objects."""

        root_joint_candidates = [
            joint_id for joint_id in range(self.robot_model.njnt) if int(self.robot_model.jnt_qposadr[joint_id]) == 0
        ]
        if len(root_joint_candidates) != 1:
            raise ValueError(
                f"Expected exactly one robot root joint at qpos address 0, found {len(root_joint_candidates)}"
            )
        root_body_id = int(self.robot_model.jnt_bodyid[root_joint_candidates[0]])

        body_ids: list[int] = []
        for body_id in range(1, self.robot_model.nbody):
            ancestor = body_id
            while ancestor not in {0, root_body_id}:
                ancestor = int(self.robot_model.body_parentid[ancestor])
            if ancestor == root_body_id:
                body_ids.append(body_id)

        if not body_ids or body_ids[0] != root_body_id:
            raise ValueError("Could not resolve the robot body subtree")
        names = tuple(
            name
            for body_id in body_ids
            for name in [
                mujoco.mj_id2name(
                    self.robot_model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    body_id,
                )
            ]
            if name is not None
        )
        if len(names) != len(body_ids) or len(set(names)) != len(names):
            raise ValueError("Robot bodies must have unique MuJoCo names")

        local_index = {body_id: index for index, body_id in enumerate(body_ids)}
        parent_indices = np.asarray(
            [
                -1 if body_id == root_body_id else local_index[int(self.robot_model.body_parentid[body_id])]
                for body_id in body_ids
            ],
            dtype=np.int32,
        )
        return np.asarray(body_ids, dtype=np.int32), names, parent_indices

    def _get_robot_link_transforms(
        self,
        q: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return world positions and rotation matrices for every robot link."""

        q_array = np.asarray(q, dtype=np.float64)
        if q_array.shape != self.robot_data.qpos.shape:
            raise ValueError(
                f"Robot qpos shape {q_array.shape} does not match model shape {self.robot_data.qpos.shape}"
            )
        self.robot_data.qpos[:] = q_array
        mujoco.mj_forward(self.robot_model, self.robot_data)
        positions = np.array(
            self.robot_data.xpos[self.robot_link_body_ids],
            dtype=np.float64,
            copy=True,
        )
        matrices = np.array(
            self.robot_data.xmat[self.robot_link_body_ids].reshape(-1, 3, 3),
            dtype=np.float64,
            copy=True,
        )
        return positions, matrices

    def _reduced_indices_for_qpos_addresses(
        self,
        qpos_addresses,
        *,
        metadata_name: str,
    ) -> np.ndarray:
        """Map full robot-qpos addresses into the active optimization suffix."""

        requested = np.asarray(qpos_addresses, dtype=np.int64).reshape(-1)
        robot_qpos_stop = 7 + self.task_constants.ROBOT_DOF
        if np.any(requested < 0) or np.any(requested >= robot_qpos_stop):
            raise ValueError(f"{metadata_name} contains addresses outside the robot qpos range [0, {robot_qpos_stop})")
        reduced_index = {int(qpos_address): index for index, qpos_address in enumerate(self.q_a_indices)}
        return np.asarray(
            [reduced_index[int(qpos_address)] for qpos_address in requested if int(qpos_address) in reduced_index],
            dtype=np.int32,
        )

    def _apply_qpos_overrides(
        self,
        target: np.ndarray,
        overrides,
        *,
        metadata_name: str,
    ) -> None:
        """Apply full-qpos keyed overrides only to active optimization values."""

        if not overrides:
            return
        qpos_addresses = np.asarray(
            [int(address) for address in overrides],
            dtype=np.int64,
        )
        reduced_indices = self._reduced_indices_for_qpos_addresses(
            qpos_addresses,
            metadata_name=metadata_name,
        )
        active_addresses = {int(address) for address in self.q_a_indices}
        values = np.asarray(
            [value for address, value in overrides.items() if int(address) in active_addresses],
            dtype=np.float64,
        )
        target[reduced_indices] = values

    def _initial_optimization_values(
        self,
        q_a_init,
    ) -> np.ndarray:
        """Accept either a reduced vector or the complete robot qpos vector."""

        if q_a_init is None:
            raise ValueError("q_a_init is required when q_nominal_list is not provided")
        initial = np.asarray(q_a_init, dtype=np.float64)
        if initial.shape == (self.nq_a,):
            return initial
        robot_qpos_count = 7 + self.task_constants.ROBOT_DOF
        if initial.shape in {
            (robot_qpos_count,),
            (self.nq,),
        }:
            return initial[self.q_a_indices]
        raise ValueError(
            "q_a_init must contain either the active optimization suffix "
            f"({self.nq_a} values) or a complete robot/model qpos vector "
            f"({robot_qpos_count} or {self.nq} values); got {initial.shape}"
        )

    def _initial_locked_qpos(self, q_a_init) -> np.ndarray:
        """Build a valid full model qpos while preserving locked coordinates."""

        if q_a_init is None:
            raise ValueError("q_a_init is required when q_nominal_list is not provided")
        initial = np.asarray(q_a_init, dtype=np.float64)
        full_qpos = self.robot_model.qpos0.copy()
        robot_qpos_count = 7 + self.task_constants.ROBOT_DOF
        if initial.shape == (self.nq,):
            full_qpos[:] = initial
        elif initial.shape == (robot_qpos_count,):
            full_qpos[:robot_qpos_count] = initial
        elif initial.shape == (self.nq_a,):
            full_qpos[self.q_a_indices] = initial
        else:
            raise ValueError(
                "q_a_init must contain either the active optimization suffix "
                f"({self.nq_a} values) or a complete robot/model qpos vector "
                f"({robot_qpos_count} or {self.nq} values); got "
                f"{initial.shape}"
            )
        if not np.isfinite(full_qpos).all():
            raise ValueError("q_a_init must contain only finite values")
        return full_qpos

    def _owned_nominal_qpos(
        self,
        q_nominal_list,
        *,
        num_frames: int,
    ) -> np.ndarray:
        """Validate and own the full-qpos nominal trajectory in solver precision."""

        try:
            nominal = np.asarray(q_nominal_list)
        except (TypeError, ValueError) as exc:
            raise ValueError("q_nominal_list must be a rectangular two-dimensional array") from exc
        if nominal.ndim != 2:
            raise ValueError(
                "q_nominal_list must be a two-dimensional array with shape "
                f"({num_frames}, {self.nq}); got {nominal.shape}"
            )
        if nominal.shape[0] != num_frames:
            raise ValueError(f"q_nominal_list must contain exactly {num_frames} frames; got {nominal.shape[0]}")
        if nominal.shape[1] != self.nq:
            raise ValueError(
                f"q_nominal_list must contain complete model qpos vectors with width {self.nq}; got {nominal.shape[1]}"
            )
        try:
            owned_nominal = np.array(
                nominal,
                dtype=np.float64,
                copy=True,
                order="C",
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("q_nominal_list must contain numeric values") from exc
        if not np.isfinite(owned_nominal).all():
            raise ValueError("q_nominal_list must contain only finite values")
        return owned_nominal

    def _robot_actuated_joint_names(self) -> tuple[str, ...]:
        """Return scalar robot joints in qpos order, excluding free joints."""

        robot_qpos_start = 7
        robot_qpos_stop = robot_qpos_start + self.task_constants.ROBOT_DOF
        joints: list[tuple[int, str]] = []
        for joint_id in range(self.robot_model.njnt):
            qpos_address = int(self.robot_model.jnt_qposadr[joint_id])
            if not robot_qpos_start <= qpos_address < robot_qpos_stop:
                continue
            joint_type = int(self.robot_model.jnt_type[joint_id])
            if joint_type not in {
                int(mujoco.mjtJoint.mjJNT_HINGE),
                int(mujoco.mjtJoint.mjJNT_SLIDE),
            }:
                raise ValueError(f"Robot actuated qpos range contains a non-scalar joint at address {qpos_address}")
            name = mujoco.mj_id2name(
                self.robot_model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_id,
            )
            if name is None:
                raise ValueError(f"Robot joint at qpos address {qpos_address} has no name")
            joints.append((qpos_address, name))
        joints.sort()
        names = tuple(name for _, name in joints)
        if len(names) != self.task_constants.ROBOT_DOF:
            raise ValueError(
                "Robot actuated joint metadata does not match ROBOT_DOF: "
                f"{len(names)} != {self.task_constants.ROBOT_DOF}"
            )
        return names

    def _validate_source_orientations(
        self,
        quaternions_wxyz: np.ndarray | None,
        joint_names: tuple[str, ...] | list[str] | np.ndarray | None,
        num_frames: int,
    ) -> tuple[np.ndarray | None, np.ndarray | None, tuple[str, ...]]:
        """Keep exact float32 source evidence separate from normalized math."""

        if quaternions_wxyz is None:
            if joint_names is not None and len(joint_names) != 0:
                raise ValueError("Source orientation names were provided without quaternions")
            return None, None, ()

        saved_quaternions = np.ascontiguousarray(
            quaternions_wxyz,
            dtype=np.float32,
        ).copy()
        if joint_names is None:
            if saved_quaternions.shape[1:] != (len(self.demo_joints), 4):
                raise ValueError("Source orientation names are required for a partial orientation tensor")
            names = tuple(self.demo_joints)
        else:
            names = tuple(str(name) for name in joint_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("Source orientation joint names must be non-empty and unique")
        unavailable = [name for name in names if name not in self.demo_joints]
        if unavailable:
            raise ValueError(f"Source orientation joints are absent from the human skeleton: {unavailable}")
        expected_shape = (num_frames, len(names), 4)
        if saved_quaternions.shape != expected_shape:
            raise ValueError(f"Source orientations must have shape {expected_shape}, got {saved_quaternions.shape}")
        math_quaternions = saved_quaternions.astype(np.float64)
        norms = np.linalg.norm(math_quaternions, axis=-1)
        if not np.isfinite(math_quaternions).all() or np.any(norms <= 1e-8):
            raise ValueError("Source orientation quaternions must be finite and non-zero")
        if np.any(np.abs(norms - 1.0) > 1e-3):
            raise ValueError("Source orientation quaternions must already be unit length")
        normalized_math_quaternions = math_quaternions / norms[..., None]
        return saved_quaternions, normalized_math_quaternions, names

    def _orientation_t_pose_robot_qpos(self) -> np.ndarray:
        """Build the robot configuration used for canonical T-pose alignment."""
        reference_q = self.robot_model.qpos0.copy()
        reference_q[3:7] = self.orientation_t_pose_robot_base_quaternion_wxyz
        for joint_name, joint_position in self.orientation_t_pose_robot_joint_positions.items():
            joint_id = mujoco.mj_name2id(
                self.robot_model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )
            qpos_address = int(self.robot_model.jnt_qposadr[joint_id])
            reference_q[qpos_address] = joint_position
        return reference_q

    def _prepare_orientation_targets(
        self,
        normalized_human_joint_quaternions_wxyz: np.ndarray | None,
        initial_q: np.ndarray,
        num_frames: int,
        human_orientation_joint_names: tuple[str, ...] | None = None,
        orientation_target_world_rotation_deltas_wxyz: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Construct target-world orientations from normalized source rotations."""

        if orientation_target_world_rotation_deltas_wxyz is None:
            world_rotation_deltas = np.zeros((num_frames, 4), dtype=np.float64)
            world_rotation_deltas[:, 0] = 1.0
        else:
            world_rotation_deltas = np.asarray(
                orientation_target_world_rotation_deltas_wxyz,
                dtype=np.float64,
            )
            if world_rotation_deltas.shape != (num_frames, 4):
                raise ValueError(
                    "orientation_target_world_rotation_deltas_wxyz must have "
                    f"shape {(num_frames, 4)}, got {world_rotation_deltas.shape}"
                )
            delta_norms = np.linalg.norm(
                world_rotation_deltas,
                axis=-1,
                keepdims=True,
            )
            if not np.isfinite(world_rotation_deltas).all() or np.any(delta_norms <= 1e-8):
                raise ValueError(
                    "orientation_target_world_rotation_deltas_wxyz must contain finite, non-zero quaternions"
                )
            world_rotation_deltas = world_rotation_deltas / delta_norms

        tracked_count = len(self.orientation_human_joint_names)
        if not (self.orientation_diagnostics_enabled or self.orientation_preview_enabled):
            return (
                np.empty((num_frames, 0, 3, 3), dtype=np.float64),
                np.empty((0, 3, 3), dtype=np.float64),
            )
        if normalized_human_joint_quaternions_wxyz is None:
            raise ValueError("Configured orientation tracking or diagnostics require source global joint orientations")
        quaternions = np.asarray(
            normalized_human_joint_quaternions_wxyz,
            dtype=np.float64,
        )
        available_orientation_names = (
            tuple(self.demo_joints) if human_orientation_joint_names is None else human_orientation_joint_names
        )
        orientation_index = {name: index for index, name in enumerate(available_orientation_names)}
        unavailable = [name for name in self.orientation_human_joint_names if name not in orientation_index]
        if unavailable:
            raise ValueError(
                "Configured orientation tracking or diagnostics require direct "
                f"source orientations for joints: {unavailable}"
            )
        selected_quaternions = quaternions[
            :,
            [orientation_index[name] for name in self.orientation_human_joint_names],
            :,
        ]
        human_matrices = self._wxyz_to_matrices(selected_quaternions)
        target_world_human_matrices = self._wxyz_to_matrices(world_rotation_deltas)[:, None, ...] @ human_matrices

        if self.orientation_alignment_mode == "t_pose":
            reference_human_quaternions = np.asarray(
                [self.orientation_t_pose_human_quaternions_wxyz[name] for name in self.orientation_human_joint_names],
                dtype=np.float64,
            )
            reference_human_matrices = self._wxyz_to_matrices(reference_human_quaternions)
            reference_robot_q = self._orientation_t_pose_robot_qpos()
            reference_robot_matrices, _ = self._get_robot_link_orientation_data(
                reference_robot_q,
                self.orientation_robot_link_names,
                with_jacobians=False,
            )
            alignment_matrices = np.swapaxes(reference_human_matrices, -1, -2) @ reference_robot_matrices
            if self.orientation_alignment_quaternions_wxyz_config:
                missing_calibrations = [
                    name
                    for name in self.orientation_human_joint_names
                    if name not in self.orientation_alignment_quaternions_wxyz_config
                ]
                if missing_calibrations:
                    raise ValueError(f"Saved T-pose orientation calibration is missing joints: {missing_calibrations}")
                saved_alignment_quaternions = np.asarray(
                    [
                        self.orientation_alignment_quaternions_wxyz_config[name]
                        for name in self.orientation_human_joint_names
                    ],
                    dtype=np.float64,
                )
                saved_alignment_matrices = self._wxyz_to_matrices(
                    saved_alignment_quaternions,
                )
                if not np.allclose(
                    saved_alignment_matrices,
                    alignment_matrices,
                    rtol=0.0,
                    atol=1e-10,
                ):
                    raise RuntimeError(
                        "Saved T-pose orientation calibration no longer matches "
                        "the configured human reference frames and robot FK; "
                        "regenerate the calibration tables"
                    )
                alignment_matrices = saved_alignment_matrices
        elif self.orientation_alignment_mode == "first_frame":
            initial_robot_matrices, _ = self._get_robot_link_orientation_data(
                initial_q,
                self.orientation_robot_link_names,
                with_jacobians=False,
            )
            reference_human_matrices = target_world_human_matrices[0].copy()
            reference_robot_matrices = initial_robot_matrices.copy()
            reference_robot_q = np.asarray(initial_q, dtype=np.float64).copy()
            alignment_matrices = np.swapaxes(target_world_human_matrices[0], -1, -2) @ initial_robot_matrices
        else:
            alignment_quaternions = np.asarray(
                [
                    self.orientation_alignment_quaternions_wxyz_config[name]
                    for name in self.orientation_human_joint_names
                ],
                dtype=np.float64,
            )
            alignment_matrices = self._wxyz_to_matrices(alignment_quaternions)
            reference_human_matrices = np.empty(
                (0, 3, 3),
                dtype=np.float64,
            )
            reference_robot_matrices = np.empty(
                (0, 3, 3),
                dtype=np.float64,
            )
            reference_robot_q = np.empty(
                (0,),
                dtype=np.float64,
            )

        if alignment_matrices.shape != (tracked_count, 3, 3):
            raise RuntimeError("Unexpected orientation alignment shape")
        self.orientation_reference_human_matrices = reference_human_matrices
        self.orientation_reference_robot_matrices = reference_robot_matrices
        self.orientation_reference_robot_qpos = reference_robot_q
        target_matrices = target_world_human_matrices @ alignment_matrices[None, ...]
        return target_matrices, alignment_matrices

    def draw_interaction_meshes(
        self,
        source_vertices: np.ndarray,
        target_vertices: np.ndarray,
        tetrahedra: np.ndarray,
        name_prefix: str = "interaction_mesh",
    ) -> list[object]:
        """Draw configured source/target interaction mesh overlays."""
        handles: list[object] = []
        if self.interaction_mesh_mode in {"source", "both"}:
            handles.extend(
                self.draw_interaction_mesh(
                    source_vertices,
                    tetrahedra,
                    name=f"{name_prefix}/source",
                    color=(1.0, 0.55, 0.0),
                )
            )
        if self.interaction_mesh_mode in {"target", "both"}:
            handles.extend(
                self.draw_interaction_mesh(
                    target_vertices,
                    tetrahedra,
                    name=f"{name_prefix}/target",
                    color=(0.0, 0.9, 1.0),
                )
            )
        return handles

    def retarget_motion(
        self,
        human_joint_motions,
        object_poses,
        object_poses_augmented,
        object_points_local_demo,
        object_points_local,
        foot_sticking_sequences,
        q_a_init=None,
        q_nominal_list=None,
        original=True,
        dest_res_path=None,
        fps=30.0,
        human_joint_quaternions_wxyz: np.ndarray | None = None,
        human_orientation_joint_names: (tuple[str, ...] | list[str] | np.ndarray | None) = None,
        visualization_human_joint_motions: np.ndarray | None = None,
        visualization_human_joint_names: (tuple[str, ...] | list[str] | np.ndarray | None) = None,
        visualization_human_joint_parent_indices: (tuple[int, ...] | list[int] | np.ndarray | None) = None,
        visualization_human_joint_quaternions_wxyz: np.ndarray | None = None,
        orientation_target_world_rotation_deltas_wxyz: np.ndarray | None = None,
        result_metadata: dict[str, object] | None = None,
        foot_contact_plan: FootContactPlan | None = None,
    ):
        """
        The main function to retarget an entire motion sequence frame by frame.

        Args:
            human_joint_motions (np.ndarray): (num_frames, num_joints, 3) array.
            human_joint_quaternions_wxyz (np.ndarray | None): Optional
                (num_frames, num_oriented_joints, 4) direct global human
                orientations in wxyz.
            human_orientation_joint_names: Joint names corresponding to the
                optional source orientation tensor. Missing joints remain
                absent rather than being estimated.
            visualization_human_joint_motions: Optional complete source FBX
                joint tree used only by the saved visualization artifact.
            visualization_human_joint_names: Names for the complete visual
                source tree.
            visualization_human_joint_parent_indices: Parent-first topology
                for the complete visual source tree.
            visualization_human_joint_quaternions_wxyz: Direct global source
                frames for every visual source joint. These never enter the
                optimizer.
            orientation_target_world_rotation_deltas_wxyz: Optional
                per-frame WXYZ world rotations applied only to orientation
                targets. The direct source tensor remains unchanged.
            object_poses (np.ndarray): source object poses in ``xyz_wxyz`` layout.
            object_poses_augmented (np.ndarray): target object poses in
                ``xyz_wxyz`` layout.
            object_points_local_demo (np.ndarray): Demo object points in local frame (rest pose).
            object_points_local (np.ndarray): Current object points in local frame (rest pose).
            foot_sticking_sequences (list): List of foot sticking sequences for each frame.
            q_a_init (np.ndarray, optional): Initial robot configuration.
            q_a_nominal (np.ndarray, optional): Nominal robot configuration.

        Returns:
            tuple: (retargeted_motions, obj_pts_demo_list, obj_pts_list, tetrahedra)
        """
        num_frames = human_joint_motions.shape[0]
        (
            saved_human_joint_quaternions_wxyz,
            normalized_math_human_joint_quaternions_wxyz,
            source_orientation_joint_names,
        ) = self._validate_source_orientations(
            human_joint_quaternions_wxyz,
            human_orientation_joint_names,
            num_frames,
        )
        visualization_fields_present = (
            visualization_human_joint_motions is not None,
            visualization_human_joint_names is not None,
            visualization_human_joint_parent_indices is not None,
            visualization_human_joint_quaternions_wxyz is not None,
        )
        if any(visualization_fields_present) and not all(visualization_fields_present):
            raise ValueError("Complete visualization-human fields must be provided together")
        visual_human_joints = None
        visual_human_names: tuple[str, ...] | None = None
        visual_human_parents = None
        visual_human_quaternions = None
        if all(visualization_fields_present):
            visual_human_joints = np.asarray(visualization_human_joint_motions, dtype=np.float32).copy()
            visual_human_names = tuple(str(name) for name in visualization_human_joint_names)
            visual_joint_count = len(visual_human_names)
            if visual_joint_count == 0 or len(set(visual_human_names)) != visual_joint_count:
                raise ValueError("Complete visualization-human joint names must be non-empty and unique")
            if visual_human_joints.shape != (num_frames, visual_joint_count, 3):
                raise ValueError(
                    "Complete visualization-human positions must have shape "
                    f"{(num_frames, visual_joint_count, 3)}, got {visual_human_joints.shape}"
                )
            visual_human_parents = np.asarray(
                visualization_human_joint_parent_indices,
                dtype=np.int32,
            ).copy()
            indices = np.arange(visual_joint_count, dtype=np.int32)
            if (
                visual_human_parents.shape != (visual_joint_count,)
                or np.any((visual_human_parents < -1) | (visual_human_parents >= indices))
                or np.count_nonzero(visual_human_parents == -1) != 1
            ):
                raise ValueError("Complete visualization-human parents must describe one parent-first rooted tree")
            visual_human_quaternions = np.asarray(
                visualization_human_joint_quaternions_wxyz,
                dtype=np.float32,
            ).copy()
            if visual_human_quaternions.shape != (num_frames, visual_joint_count, 4):
                raise ValueError(
                    "Complete visualization-human quaternions must have shape "
                    f"{(num_frames, visual_joint_count, 4)}, got {visual_human_quaternions.shape}"
                )
            visual_norms = np.linalg.norm(visual_human_quaternions.astype(np.float64), axis=-1)
            if (
                not np.isfinite(visual_human_joints).all()
                or not np.isfinite(visual_human_quaternions).all()
                or np.any(visual_norms <= 1e-8)
                or np.any(np.abs(visual_norms - 1.0) > 1e-3)
            ):
                raise ValueError("Complete visualization-human data must contain finite positions and unit quaternions")
            missing_mapped_names = sorted(set(self.laplacian_match_links).difference(visual_human_names))
            if missing_mapped_names:
                raise ValueError(f"Complete visualization-human tree is missing mapped joints: {missing_mapped_names}")
        legacy_foot_sticking_states = self._foot_sticking_states_array(
            foot_sticking_sequences,
            num_frames,
        )
        if foot_contact_plan is None:
            foot_contact_plan = FootContactPlan.from_sticking_sequences(
                [
                    {
                        "left": bool(frame[0]),
                        "right": bool(frame[1]),
                    }
                    for frame in legacy_foot_sticking_states
                ],
                num_frames=num_frames,
            )
        elif len(foot_contact_plan) != num_frames:
            raise ValueError(
                f"foot_contact_plan must contain {num_frames} frames, got {len(foot_contact_plan)}",
            )
        foot_sticking_states = foot_contact_plan.sticking_states
        if not hasattr(self, "_foot_contact_phase_anchors"):
            self._foot_contact_phase_anchors = {}
        else:
            self._foot_contact_phase_anchors.clear()
        self._active_planar_foot_targets = None
        if q_nominal_list is not None:
            q_nominal_list = self._owned_nominal_qpos(
                q_nominal_list,
                num_frames=num_frames,
            )
            q_locked_list = q_nominal_list
        else:
            q_locked_list = np.repeat(
                self._initial_locked_qpos(q_a_init)[None, :],
                num_frames,
                axis=0,
            )

        self._apply_dynamic_object_poses(
            q_locked_list,
            object_poses_augmented,
        )
        q = np.copy(q_locked_list[0])
        retargeted_motions = [q]
        (
            orientation_target_matrices,
            orientation_alignment_matrices,
        ) = self._prepare_orientation_targets(
            normalized_math_human_joint_quaternions_wxyz,
            q,
            num_frames,
            human_orientation_joint_names=source_orientation_joint_names,
            orientation_target_world_rotation_deltas_wxyz=(orientation_target_world_rotation_deltas_wxyz),
        )
        root_stability_target_positions = np.empty(
            (num_frames, 0),
            dtype=np.float64,
        )
        root_stability_target_matrices = np.empty(
            (num_frames, 0, 3, 3),
            dtype=np.float64,
        )
        root_orientation_reference_matrices = (
            orientation_target_matrices[
                :,
                self.root_stability_orientation_index,
            ]
            if self.root_stability_orientation_index >= 0
            and orientation_target_matrices.shape[1:] != (0, 3, 3)
            else None
        )
        shoulder_direction_targets = self._prepare_shoulder_direction_targets(
            human_joint_motions,
        )

        tetrahedra = []
        obj_pts_demo_list = []  # source/demo object pts after human-scale normalization
        obj_pts_list = []  # target object pts at the retargeting asset scale
        interaction_source_vertices_w_list = []
        interaction_target_vertices_w_list = []
        mapped_robot_joints_w_list = []
        robot_link_positions_w_list = []
        robot_link_quaternions_wxyz_list = []
        mapped_robot_link_indices = np.asarray(
            [self.robot_link_name_to_index[name] for name in self.laplacian_match_links.values()],
            dtype=np.int32,
        )
        orientation_robot_link_indices = np.asarray(
            [self.robot_link_name_to_index[name] for name in self.orientation_robot_link_names],
            dtype=np.int32,
        )
        frame_costs = []
        sqp_iteration_counts = []
        sqp_stop_reasons = []
        orientation_robot_quaternions_wxyz: list[np.ndarray] = []
        orientation_errors_rad: list[np.ndarray] = []
        orientation_frame_costs: list[float] = []
        shoulder_actual_directions: list[np.ndarray] = []
        shoulder_direction_errors_rad: list[np.ndarray] = []
        shoulder_actual_joint_positions: list[np.ndarray] = []
        shoulder_direction_singular_values: list[np.ndarray] = []
        root_stability_actual_positions: list[np.ndarray] = []
        root_stability_actual_matrices: list[np.ndarray] = []
        root_stability_position_errors: list[float] = []
        root_stability_orientation_errors: list[float] = []
        collect_interaction_mesh = self.save_interaction_mesh or (self.visualize and self.show_interaction_mesh)
        collect_object_point_trajectories = self.has_dynamic_object or self.visualize
        interaction_mesh_handle_list: list[object] = []

        def _clear_interaction_mesh_handles() -> None:
            for handle in interaction_mesh_handle_list:
                with suppress(Exception):
                    handle.remove()
            interaction_mesh_handle_list.clear()

        print(f"\nStarting motion retargeting for {num_frames} frames...")

        with tqdm(range(num_frames)) as pbar:
            for i in pbar:
                # Get object poses and transform points
                object_quat_demo = object_poses[i, 3:]
                object_trans_demo = object_poses[i, :3]

                object_points_local_demo_frame = object_points_local_demo
                object_points_local_frame = object_points_local
                if self.object_name == "ground" and self.dynamic_ground_window:
                    object_points_local_demo_frame = self._ground_points_at_robot_root(
                        object_points_local_demo,
                        q,
                    )
                    object_points_local_frame = self._ground_points_at_robot_root(
                        object_points_local,
                        q,
                    )

                # Get human joint positions and create interaction mesh in object frame
                human_mapped_joints = human_joint_motions[i, self.mapped_joint_indices]

                if self.object_name == "ground":
                    human_mapped_joints_in_object = human_mapped_joints
                else:
                    human_mapped_joints_in_object = transform_points_world_to_local(
                        object_quat_demo, object_trans_demo, human_mapped_joints
                    )

                source_vertices, source_tetrahedra = create_interaction_mesh(
                    np.vstack([human_mapped_joints_in_object, object_points_local_demo_frame])
                )
                tetrahedra.append(source_tetrahedra)

                object_quat = object_poses_augmented[i, 3:]
                object_trans = object_poses_augmented[i, :3]
                obj_pts_demo = transform_points_local_to_world(
                    object_quat_demo, object_trans_demo, object_points_local_demo_frame
                )
                obj_pts = transform_points_local_to_world(
                    object_quat,
                    object_trans,
                    object_points_local_frame,
                )
                if collect_object_point_trajectories:
                    obj_pts_demo_list.append(obj_pts_demo.astype(np.float32))
                    obj_pts_list.append(obj_pts.astype(np.float32))

                source_vertices_w = None
                target_vertices_w = None
                if collect_interaction_mesh:
                    source_vertices_w = np.vstack([human_mapped_joints, obj_pts_demo])

                if self.debug:
                    # Only for visualization
                    human_kpts_handle_list = self.draw_keypoints(human_mapped_joints, name="human_kpts")  # 15 X 3
                    human_hand_handle_list = self.draw_human_hand_skeleton(
                        human_joint_motions[i],
                        name="human_hands",
                    )
                    obj_kpts_demo_handle_list = self.draw_keypoints(
                        obj_pts_demo, name="object_demo_kpts", rgba=(1, 0, 0, 1)
                    )  # 100 X 3
                    obj_kpts_handle_list = self.draw_keypoints(
                        obj_pts, name="object_kpts", rgba=(0, 1, 1, 1)
                    )  # 100 X 3
                    human_skeleton_handle_list = self.draw_mapped_skeleton(
                        human_mapped_joints,
                        name="human_skeleton",
                        rgba=(0, 0, 1, 1),
                    )

                # Create adjacency list and calculate target Laplacian coordinates
                adj_list = get_adjacency_list(source_tetrahedra, len(source_vertices))
                target_laplacian = calculate_laplacian_coordinates(source_vertices, adj_list)

                # Run optimization
                if original:
                    w_nominal_tracking = self.w_nominal_tracking_init
                else:
                    w_nominal_tracking = self.w_nominal_tracking_init * np.exp(-i / self.nominal_tracking_tau)

                root_stability_bootstrap = self.root_stability_enabled and i == 0
                q, cost = self.iterate(
                    q_locked=q_locked_list[i],
                    q_n=q,
                    q_t_last=retargeted_motions[-1],
                    target_laplacian=target_laplacian,
                    adj_list=adj_list,
                    obj_pts_local=object_points_local_frame,
                    foot_sticking=foot_sticking_sequences[i],
                    w_nominal_tracking=w_nominal_tracking,
                    q_a_nominal=(q_nominal_list[i, self.q_a_indices] if q_nominal_list is not None else None),
                    init_t=i == 0,
                    n_iter=50 if i == 0 else 10,
                    frame_idx=i,
                    orientation_target_matrices=orientation_target_matrices[i],
                    foot_contact_state=foot_contact_plan.frames[i],
                    shoulder_direction_targets=shoulder_direction_targets[i],
                    root_stability_target_position=(
                        root_stability_target_positions[i]
                        if self.root_stability_enabled
                        and not root_stability_bootstrap
                        else None
                    ),
                    root_stability_target_matrix=(
                        root_stability_target_matrices[i]
                        if self.root_stability_enabled
                        and not root_stability_bootstrap
                        and root_stability_target_matrices.shape[1:] == (3, 3)
                        else None
                    ),
                    root_stability_bootstrap=root_stability_bootstrap,
                )
                frame_costs.append(float(cost))
                sqp_iteration_counts.append(self.last_sqp_iteration_count)
                sqp_stop_reasons.append(self.last_sqp_stop_reason)

                if root_stability_bootstrap:
                    (
                        root_stability_target_positions,
                        root_stability_target_matrices,
                    ) = self._prepare_root_stability_targets(
                        human_joint_motions,
                        q,
                        root_orientation_reference_matrices=(
                            root_orientation_reference_matrices
                        ),
                    )

                (
                    all_robot_link_positions,
                    all_robot_link_matrices,
                ) = self._get_robot_link_transforms(q)
                all_robot_link_quaternions = self._matrices_to_wxyz(all_robot_link_matrices)
                robot_link_positions_w_list.append(all_robot_link_positions.astype(np.float32))
                robot_link_quaternions_wxyz_list.append(all_robot_link_quaternions.astype(np.float32))
                robot_link_positions = all_robot_link_positions[mapped_robot_link_indices]
                mapped_robot_joints_w_list.append(robot_link_positions.astype(np.float32))
                if self.root_stability_enabled:
                    root_position = all_robot_link_positions[
                        self.root_stability_robot_link_index
                    ]
                    root_matrix = all_robot_link_matrices[
                        self.root_stability_robot_link_index
                    ]
                    root_stability_actual_positions.append(
                        root_position.astype(np.float32),
                    )
                    root_stability_actual_matrices.append(
                        root_matrix.astype(np.float32),
                    )
                    root_stability_position_errors.append(
                        float(
                            np.linalg.norm(
                                root_stability_target_positions[i]
                                - root_position,
                            ),
                        ),
                    )
                    if root_stability_target_matrices.shape[1:] == (3, 3):
                        root_stability_orientation_errors.append(
                            float(
                                np.linalg.norm(
                                    self._so3_error_vectors(
                                        root_stability_target_matrices[i][None],
                                        root_matrix[None],
                                    )[0],
                                ),
                            ),
                        )
                    else:
                        root_stability_orientation_errors.append(0.0)
                if self.orientation_diagnostics_enabled:
                    robot_orientation_matrices = all_robot_link_matrices[orientation_robot_link_indices]
                    error_vectors = self._so3_error_vectors(
                        orientation_target_matrices[i],
                        robot_orientation_matrices,
                    )
                    error_angles = np.linalg.norm(error_vectors, axis=-1)
                    orientation_robot_quaternions_wxyz.append(
                        self._matrices_to_wxyz(robot_orientation_matrices).astype(np.float32)
                    )
                    orientation_errors_rad.append(error_angles.astype(np.float32))
                    orientation_frame_costs.append(
                        float(np.sum(self.orientation_weight_values * np.square(error_angles)))
                    )
                if self.shoulder_direction_enabled:
                    actual_directions, direction_jacobians = self._get_shoulder_direction_data(
                        q,
                        with_jacobians=True,
                    )
                    if direction_jacobians is None:
                        raise RuntimeError("Expected final shoulder direction Jacobians")
                    actual_joint_positions = q[self.shoulder_joint_qpos_addresses]
                    singular_values = np.empty((2, 3), dtype=np.float64)
                    direction_errors = np.empty((2,), dtype=np.float64)
                    for side_index in range(2):
                        reduced = self.shoulder_joint_reduced_indices[side_index]
                        local_jacobian = direction_jacobians[side_index][:, reduced]
                        singular_values[side_index] = np.linalg.svd(
                            local_jacobian,
                            compute_uv=False,
                        )
                        direction_errors[side_index] = direction_error_angle(
                            actual_directions[side_index],
                            shoulder_direction_targets[i, side_index],
                        )
                    shoulder_actual_directions.append(actual_directions.astype(np.float32))
                    shoulder_direction_errors_rad.append(direction_errors.astype(np.float32))
                    shoulder_actual_joint_positions.append(
                        actual_joint_positions.astype(np.float32),
                    )
                    shoulder_direction_singular_values.append(
                        singular_values.astype(np.float32),
                    )
                if collect_interaction_mesh:
                    if source_vertices_w is None:
                        raise RuntimeError("Expected source interaction mesh vertices to be materialized")
                    target_vertices_w = np.vstack([robot_link_positions, obj_pts])
                    if self.save_interaction_mesh:
                        interaction_source_vertices_w_list.append(source_vertices_w)
                        interaction_target_vertices_w_list.append(target_vertices_w)
                    if self.visualize and self.show_interaction_mesh:
                        _clear_interaction_mesh_handles()
                        interaction_mesh_handle_list.extend(
                            self.draw_interaction_meshes(
                                source_vertices_w,
                                target_vertices_w,
                                source_tetrahedra,
                                name_prefix="world/interaction_mesh",
                            )
                        )

                if self.debug:
                    robot_kpts_handle_list = self.draw_keypoints(
                        robot_link_positions, name="robot_kpts", rgba=(0, 1, 0, 1)
                    )
                    robot_skeleton_handle_list = self.draw_mapped_skeleton(
                        robot_link_positions,
                        name="robot_skeleton",
                        rgba=(0, 1, 0, 1),
                    )

                retargeted_motions.append(q)
                self._update_live_visualization_frame(
                    q,
                    frame_idx=i,
                    foot_sticking_state=foot_sticking_states[i],
                )

                pbar.set_postfix(cost=cost)

        # Remove previous debug visualization
        if self.debug:
            for handle in human_kpts_handle_list:
                handle.remove()
            human_kpts_handle_list.clear()

            for handle in obj_kpts_demo_handle_list:
                handle.remove()
            obj_kpts_demo_handle_list.clear()

            for handle in obj_kpts_handle_list:
                handle.remove()
            obj_kpts_handle_list.clear()

            for handle in robot_kpts_handle_list:
                handle.remove()
            robot_kpts_handle_list.clear()

            for handle in human_skeleton_handle_list:
                handle.remove()
            human_skeleton_handle_list.clear()

            for handle in human_hand_handle_list:
                handle.remove()
            human_hand_handle_list.clear()

            for handle in robot_skeleton_handle_list:
                handle.remove()
            robot_skeleton_handle_list.clear()
        _clear_interaction_mesh_handles()

        # Save results
        mapped_human_joint_names = list(self.laplacian_match_links.keys())
        if visual_human_joints is not None and visual_human_names is not None and visual_human_parents is not None:
            saved_human_joint_names = list(visual_human_names)
            saved_human_joints = visual_human_joints
            saved_human_parent_indices = visual_human_parents
        else:
            saved_human_name_set = set(mapped_human_joint_names)
            saved_human_name_set.update(self.orientation_human_joint_names)
            saved_human_indices = [index for index, name in enumerate(self.demo_joints) if name in saved_human_name_set]
            saved_human_joint_names = [self.demo_joints[index] for index in saved_human_indices]
            saved_human_joints = np.asarray(
                human_joint_motions[:, saved_human_indices],
                dtype=np.float32,
            )
            saved_human_parent_indices = self._subset_parent_indices(
                self.human_joint_parent_indices,
                saved_human_indices,
            )
        saved_robot_indices = list(range(len(self.robot_link_names)))
        saved_robot_link_names = [self.robot_link_names[index] for index in saved_robot_indices]
        all_robot_link_positions = np.asarray(
            robot_link_positions_w_list,
            dtype=np.float32,
        )
        all_robot_link_quaternions = np.asarray(
            robot_link_quaternions_wxyz_list,
            dtype=np.float32,
        )
        tracked_count = len(self.orientation_human_joint_names)
        if self.orientation_diagnostics_enabled or self.orientation_preview_enabled:
            prepared_target_quaternions_wxyz = self._matrices_to_wxyz(
                orientation_target_matrices,
            ).astype(np.float32)
        else:
            prepared_target_quaternions_wxyz = np.empty(
                (num_frames, 0, 4),
                dtype=np.float32,
            )
        if self.orientation_diagnostics_enabled:
            target_quaternions_wxyz = prepared_target_quaternions_wxyz
            robot_quaternions_wxyz = np.asarray(
                orientation_robot_quaternions_wxyz,
                dtype=np.float32,
            ).reshape(num_frames, tracked_count, 4)
            orientation_error_array = np.asarray(
                orientation_errors_rad,
                dtype=np.float32,
            ).reshape(num_frames, tracked_count)
            orientation_frame_cost_array = np.asarray(
                orientation_frame_costs,
                dtype=np.float64,
            )
        else:
            target_quaternions_wxyz = np.empty(
                (num_frames, 0, 4),
                dtype=np.float32,
            )
            robot_quaternions_wxyz = np.empty(
                (num_frames, 0, 4),
                dtype=np.float32,
            )
            orientation_error_array = np.empty(
                (num_frames, 0),
                dtype=np.float32,
            )
            orientation_frame_cost_array = np.zeros(
                num_frames,
                dtype=np.float64,
            )
        preview_target_quaternions_wxyz = (
            prepared_target_quaternions_wxyz
            if self.orientation_preview_enabled
            else np.empty((num_frames, 0, 4), dtype=np.float32)
        )
        diagnostic_human_joint_names = (
            self.orientation_human_joint_names if self.orientation_diagnostics_enabled else ()
        )
        diagnostic_robot_link_names = self.orientation_robot_link_names if self.orientation_diagnostics_enabled else ()
        diagnostic_weights = (
            self.orientation_weight_values if self.orientation_diagnostics_enabled else np.empty((0,), dtype=np.float64)
        )
        save_payload = {
            "qpos": np.asarray(retargeted_motions[1:], dtype=np.float64),
            "qpos_layout": np.asarray("mujoco_free_root_xyz_wxyz_then_actuated_then_optional_object_free_joint"),
            "human_joints": saved_human_joints,
            "human_joint_names": np.asarray(saved_human_joint_names, dtype=str),
            "human_joint_parent_indices": saved_human_parent_indices,
            "mapped_human_joints": human_joint_motions[:, self.mapped_joint_indices],
            "mapped_human_joint_names": np.asarray(mapped_human_joint_names, dtype=str),
            "mapped_robot_joints": np.asarray(mapped_robot_joints_w_list, dtype=np.float32),
            "mapped_robot_link_names": np.asarray(list(self.laplacian_match_links.values()), dtype=str),
            "human_points_world": np.asarray(
                human_joint_motions[:, self.mapped_joint_indices],
                dtype=np.float32,
            ),
            "robot_points_world": np.asarray(
                mapped_robot_joints_w_list,
                dtype=np.float32,
            ),
            "robot_link_positions": np.asarray(
                all_robot_link_positions[:, saved_robot_indices],
                dtype=np.float32,
            ),
            "robot_link_quaternions_wxyz": np.asarray(
                all_robot_link_quaternions[:, saved_robot_indices],
                dtype=np.float32,
            ),
            "robot_link_names": np.asarray(saved_robot_link_names, dtype=str),
            "robot_link_parent_indices": self._subset_parent_indices(
                self.robot_link_parent_indices,
                saved_robot_indices,
            ),
            "robot_actuated_joint_names": np.asarray(
                self.robot_actuated_joint_names,
                dtype=str,
            ),
            "source_data_format": np.asarray(self.task_constants.SOURCE_DATA_FORMAT),
            "robot_type": np.asarray(self.task_constants.ROBOT_TYPE),
            "task_type": np.asarray(getattr(self.task_constants, "TASK_TYPE", "")),
            "object_name": np.asarray(self.object_name),
            "object_urdf": np.asarray(self.object_model_path or ""),
            "contains_object_in_qpos": np.asarray(bool(self.object_model_path) and bool(self.has_dynamic_object)),
            "object_poses_demo": np.asarray(object_poses, dtype=np.float32),
            "object_poses_target": np.asarray(
                object_poses_augmented,
                dtype=np.float32,
            ),
            "object_pose_layout": np.asarray("xyz_wxyz"),
            "quaternion_convention": np.asarray("wxyz"),
            "world_coordinate_system": np.asarray("right_handed_z_up"),
            "foot_sticking_side_names": np.asarray(("left", "right"), dtype=str),
            "foot_sticking_states": foot_sticking_states,
            "foot_sticking_tolerance": np.asarray(self.foot_sticking_tolerance),
            "foot_sticking_enabled_for_saved_trajectory": np.asarray(
                self.activate_foot_sticking and self.q_a_init_idx < 12,
            ),
            "foot_contact_modes": foot_contact_plan.modes,
            "foot_contact_phase_ids": foot_contact_plan.phase_ids,
            "foot_contact_confidences": foot_contact_plan.confidences,
            "foot_contact_pivot_regions": foot_contact_plan.pivot_regions,
            "foot_contact_pivot_uv": foot_contact_plan.pivot_uv,
            "foot_contact_reference_displacements_xy": (
                foot_contact_plan.reference_displacements_xy
            ),
            "foot_contact_plan_version": np.asarray(1, dtype=np.int32),
            "interaction_mesh_weight": np.asarray(
                self.interaction_mesh_weight,
            ),
            "arm_interaction_mesh_weight_scale": np.asarray(
                self.arm_interaction_mesh_weight_scale,
            ),
            "root_stability_enabled": np.asarray(
                self.root_stability_enabled,
            ),
            "root_stability_robot_link_name": np.asarray(
                self.root_stability_robot_link_name,
            ),
            "root_stability_human_root_name": np.asarray(
                self.root_stability_human_root_name,
            ),
            "root_stability_bootstrap_frame": np.asarray(
                0 if self.root_stability_enabled else -1,
                dtype=np.int32,
            ),
            "root_stability_orientation_source": np.asarray(
                self.root_stability_orientation_source,
            ),
            "root_stability_position_weight": np.asarray(
                float(self.root_stability_config.position_weight),
            ),
            "root_stability_orientation_weight": np.asarray(
                float(self.root_stability_config.orientation_weight),
            ),
            "root_stability_target_positions": (
                root_stability_target_positions.astype(np.float32)
            ),
            "root_stability_actual_positions": (
                np.asarray(root_stability_actual_positions, dtype=np.float32)
                if self.root_stability_enabled
                else np.empty((num_frames, 0), dtype=np.float32)
            ),
            "root_stability_position_errors_m": (
                np.asarray(root_stability_position_errors, dtype=np.float32)
                if self.root_stability_enabled
                else np.empty((num_frames, 0), dtype=np.float32)
            ),
            "root_stability_target_quaternions_wxyz": (
                self._matrices_to_wxyz(
                    root_stability_target_matrices,
                ).astype(np.float32)
            ),
            "root_stability_actual_quaternions_wxyz": (
                self._matrices_to_wxyz(
                    np.asarray(
                        root_stability_actual_matrices,
                        dtype=np.float64,
                    ),
                ).astype(np.float32)
                if self.root_stability_enabled
                else np.empty((num_frames, 0, 4), dtype=np.float32)
            ),
            "root_stability_orientation_errors_rad": (
                np.asarray(
                    root_stability_orientation_errors,
                    dtype=np.float32,
                )
                if self.root_stability_enabled
                else np.empty((num_frames, 0), dtype=np.float32)
            ),
            "frame_costs": np.asarray(frame_costs, dtype=np.float64),
            "sqp_iteration_counts": np.asarray(sqp_iteration_counts, dtype=np.int32),
            "sqp_stop_reasons": np.asarray(sqp_stop_reasons, dtype=str),
            "orientation_tracking_enabled": np.asarray(self.orientation_tracking_enabled),
            "orientation_diagnostics_enabled": np.asarray(self.orientation_diagnostics_enabled),
            "orientation_human_joint_names": np.asarray(
                diagnostic_human_joint_names,
                dtype=str,
            ),
            "orientation_robot_link_names": np.asarray(
                diagnostic_robot_link_names,
                dtype=str,
            ),
            "orientation_weights": diagnostic_weights.astype(np.float64),
            "orientation_alignment_mode": np.asarray(self.orientation_alignment_mode),
            "orientation_alignment_quaternions_wxyz": (
                self._matrices_to_wxyz(orientation_alignment_matrices).astype(np.float32)
                if self.orientation_diagnostics_enabled
                else np.empty((0, 4), dtype=np.float32)
            ),
            "orientation_reference_human_quaternions_wxyz": (
                self._matrices_to_wxyz(self.orientation_reference_human_matrices).astype(np.float32)
                if self.orientation_diagnostics_enabled
                else np.empty((0, 4), dtype=np.float32)
            ),
            "orientation_reference_robot_quaternions_wxyz": (
                self._matrices_to_wxyz(self.orientation_reference_robot_matrices).astype(np.float32)
                if self.orientation_diagnostics_enabled
                else np.empty((0, 4), dtype=np.float32)
            ),
            "orientation_reference_robot_qpos": (
                self.orientation_reference_robot_qpos.astype(np.float32)
                if self.orientation_diagnostics_enabled
                else np.empty((0,), dtype=np.float32)
            ),
            "orientation_target_quaternions_wxyz": target_quaternions_wxyz,
            "orientation_robot_quaternions_wxyz": robot_quaternions_wxyz,
            "orientation_errors_rad": orientation_error_array,
            "orientation_frame_costs": orientation_frame_cost_array,
            "orientation_preview_enabled": np.asarray(self.orientation_preview_enabled),
            "orientation_preview_human_joint_names": np.asarray(
                self.orientation_human_joint_names if self.orientation_preview_enabled else (),
                dtype=str,
            ),
            "orientation_preview_robot_link_names": np.asarray(
                self.orientation_robot_link_names if self.orientation_preview_enabled else (),
                dtype=str,
            ),
            "orientation_preview_alignment_mode": np.asarray(
                self.orientation_alignment_mode if self.orientation_preview_enabled else "",
            ),
            "orientation_preview_alignment_quaternions_wxyz": (
                self._matrices_to_wxyz(orientation_alignment_matrices).astype(np.float32)
                if self.orientation_preview_enabled
                else np.empty((0, 4), dtype=np.float32)
            ),
            "orientation_preview_target_quaternions_wxyz": preview_target_quaternions_wxyz,
            "shoulder_direction_tracking_enabled": np.asarray(
                self.shoulder_direction_enabled,
            ),
            "shoulder_direction_human_arm_names": np.asarray(
                [spec.human_arm_name for spec in self.shoulder_side_specs],
                dtype=str,
            ),
            "shoulder_direction_joint_names": np.asarray(
                [spec.shoulder_joint_names for spec in self.shoulder_side_specs],
                dtype=str,
            ).reshape(-1, 3),
            "shoulder_direction_target_vectors": (
                shoulder_direction_targets.astype(np.float32)
                if self.shoulder_direction_enabled
                else np.empty((num_frames, 0, 3), dtype=np.float32)
            ),
            "shoulder_direction_actual_vectors": (
                np.asarray(shoulder_actual_directions, dtype=np.float32)
                if self.shoulder_direction_enabled
                else np.empty((num_frames, 0, 3), dtype=np.float32)
            ),
            "shoulder_direction_errors_rad": (
                np.asarray(shoulder_direction_errors_rad, dtype=np.float32)
                if self.shoulder_direction_enabled
                else np.empty((num_frames, 0), dtype=np.float32)
            ),
            "shoulder_direction_actual_joint_positions": (
                np.asarray(shoulder_actual_joint_positions, dtype=np.float32)
                if self.shoulder_direction_enabled
                else np.empty((num_frames, 0, 3), dtype=np.float32)
            ),
            "shoulder_direction_singular_values": (
                np.asarray(shoulder_direction_singular_values, dtype=np.float32)
                if self.shoulder_direction_enabled
                else np.empty((num_frames, 0, 3), dtype=np.float32)
            ),
            "shoulder_direction_weight": np.asarray(
                float(self.shoulder_direction_config.direction_weight),
            ),
            "shoulder_wrist_axis_orientation_weights": (
                self.shoulder_wrist_weights.astype(np.float64)
            ),
            "shoulder_wrist_orientation_weights": (
                self.shoulder_wrist_weights.astype(np.float64)
            ),
            "shoulder_wrist_dof_counts": (
                self.shoulder_wrist_dof_counts.astype(np.int32)
            ),
            "shoulder_wrist_joint_names": np.asarray(
                [
                    list(spec.wrist_joint_names)
                    + [""] * (3 - len(spec.wrist_joint_names))
                    for spec in self.shoulder_side_specs
                ],
                dtype=str,
            ),
            "natural_pose_tracking_enabled": np.asarray(
                self.natural_pose_tracking_enabled,
            ),
            "natural_pose_configured_joint_names": np.asarray(
                self.natural_pose_configured_joint_names,
                dtype=str,
            ),
            "natural_pose_configured_joint_positions": (
                self.natural_pose_configured_reference_values.astype(np.float64)
            ),
            "natural_pose_configured_weights": (self.natural_pose_configured_weight_values.astype(np.float64)),
            "natural_pose_joint_names": np.asarray(
                self.natural_pose_joint_names,
                dtype=str,
            ),
            "natural_pose_qpos_addresses": self.natural_pose_qpos_addresses.astype(
                np.int32,
            ),
            "natural_pose_joint_positions": (self.natural_pose_reference_values.astype(np.float64)),
            "natural_pose_weights": self.natural_pose_weight_values.astype(np.float64),
            "fps": float(fps),
            "cost": cost,
        }
        if visual_human_quaternions is not None and visual_human_names is not None:
            save_payload.update(
                {
                    "human_orientation_joint_names": np.asarray(visual_human_names, dtype=str),
                    "human_orientation_quaternions_wxyz": visual_human_quaternions,
                }
            )
        elif saved_human_joint_quaternions_wxyz is not None and (
            self.orientation_diagnostics_enabled or self.orientation_preview_enabled
        ):
            source_orientation_index = {name: index for index, name in enumerate(source_orientation_joint_names)}
            saved_orientation_indices = [source_orientation_index[name] for name in self.orientation_human_joint_names]
            saved_human_orientation_names = np.asarray(
                self.orientation_human_joint_names,
                dtype=str,
            )
            saved_human_orientation_quaternions = np.asarray(
                saved_human_joint_quaternions_wxyz[
                    :,
                    saved_orientation_indices,
                ],
                dtype=np.float32,
            )
            save_payload.update(
                {
                    "human_orientation_joint_names": (saved_human_orientation_names),
                    "human_orientation_quaternions_wxyz": (saved_human_orientation_quaternions),
                }
            )
        if result_metadata:
            conflicts = sorted(set(result_metadata).intersection(save_payload))
            if conflicts:
                raise ValueError(f"Result metadata cannot replace solver fields: {conflicts}")
            save_payload.update({key: np.asarray(value) for key, value in result_metadata.items()})
        save_payload.update(
            self._object_points_save_payload(
                has_dynamic_object=self.has_dynamic_object,
                object_points_local_demo=object_points_local_demo,
                object_points_local=object_points_local,
                obj_pts_demo_list=obj_pts_demo_list,
                obj_pts_list=obj_pts_list,
            )
        )
        if self.save_interaction_mesh:
            packed_tetrahedra, tetrahedra_counts = self._pack_interaction_tetrahedra(tetrahedra)
            interaction_source_vertices = np.asarray(
                interaction_source_vertices_w_list,
                dtype=np.float32,
            )
            interaction_target_vertices = np.asarray(
                interaction_target_vertices_w_list,
                dtype=np.float32,
            )
            save_payload.update(
                {
                    "interaction_source_vertices_w": interaction_source_vertices,
                    "interaction_target_vertices_w": interaction_target_vertices,
                    "interaction_tetrahedra": packed_tetrahedra,
                    "interaction_tetrahedra_counts": tetrahedra_counts,
                    "interaction_num_human_vertices": np.asarray(
                        len(self.laplacian_match_links),
                        dtype=np.int32,
                    ),
                    "interaction_num_object_vertices": np.asarray(
                        len(object_points_local),
                        dtype=np.int32,
                    ),
                    "interaction_mesh_edges_default": self.interaction_mesh_edges,
                }
            )
            if getattr(self.task_constants, "TASK_TYPE", "") in {
                "robot_only",
                "climbing",
            }:
                save_payload["terrain_points_world"] = interaction_target_vertices[
                    :,
                    len(self.laplacian_match_links) :,
                ]
        destination = Path(dest_res_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, **save_payload)
        print("Saving results to path:", dest_res_path)

        if self.visualize:
            saved_foot_sticking_enabled = bool(
                self.activate_foot_sticking and self.q_a_init_idx < 12,
            )
            robot_dof = len(self.viser_robot.get_actuated_joint_limits())
            replay_overlay_handles = []

            def _clear_replay_overlay():
                nonlocal replay_overlay_handles
                for handle in replay_overlay_handles:
                    with suppress(Exception):
                        handle.remove()
                replay_overlay_handles = []

            def _human_joints_at_frame(frame_float: float) -> np.ndarray:
                if num_frames == 1:
                    return human_joint_motions[0]

                frame_float = float(np.clip(frame_float, 0.0, num_frames - 1))
                i0 = int(np.floor(frame_float))
                i1 = min(i0 + 1, num_frames - 1)
                u = frame_float - i0
                return (1.0 - u) * human_joint_motions[i0] + u * human_joint_motions[i1]

            def _object_points_at_frame(points: list[np.ndarray], frame_float: float) -> np.ndarray:
                if num_frames == 1:
                    return points[0]
                frame_float = float(np.clip(frame_float, 0.0, num_frames - 1))
                i0 = int(np.floor(frame_float))
                i1 = min(i0 + 1, num_frames - 1)
                u = frame_float - i0
                return (1.0 - u) * points[i0] + u * points[i1]

            def _interaction_mesh_frame_index(frame_float: float) -> int:
                return int(np.clip(round(float(frame_float)), 0, num_frames - 1))

            def _draw_replay_overlays(q: np.ndarray, frame_float: float) -> None:
                _clear_replay_overlay()
                replay_frame_idx = _interaction_mesh_frame_index(frame_float)
                self._update_foot_sticking_status(
                    replay_frame_idx,
                    foot_sticking_states[replay_frame_idx],
                    constraints_enabled=saved_foot_sticking_enabled,
                )

                if self.debug:
                    human_frame = _human_joints_at_frame(frame_float)
                    human_mapped_joints = human_frame[self.mapped_joint_indices]
                    robot_link_positions = self._get_robot_link_positions(q, self.laplacian_match_links.values())
                    replay_overlay_handles.extend(self.draw_keypoints(human_mapped_joints, name="replay_human_kpts"))
                    replay_overlay_handles.extend(
                        self.draw_mapped_skeleton(
                            human_mapped_joints,
                            name="replay_human_skeleton",
                            rgba=(0, 0, 1, 1),
                        )
                    )
                    replay_overlay_handles.extend(
                        self.draw_human_hand_skeleton(
                            human_frame,
                            name="replay_human_hands",
                        )
                    )
                    replay_overlay_handles.extend(
                        self.draw_keypoints(robot_link_positions, name="replay_robot_kpts", rgba=(0, 1, 0, 1))
                    )
                    replay_overlay_handles.extend(
                        self.draw_mapped_skeleton(
                            robot_link_positions,
                            name="replay_robot_skeleton",
                            rgba=(0, 1, 0, 1),
                        )
                    )
                    replay_overlay_handles.extend(
                        self.draw_keypoints(
                            _object_points_at_frame(obj_pts_demo_list, frame_float),
                            name="replay_object_demo_kpts",
                            rgba=(1, 0, 0, 1),
                        )
                    )
                    replay_overlay_handles.extend(
                        self.draw_keypoints(
                            _object_points_at_frame(obj_pts_list, frame_float),
                            name="replay_object_target_kpts",
                            rgba=(0, 1, 1, 1),
                        )
                    )

                if (
                    self.show_interaction_mesh
                    and interaction_source_vertices_w_list
                    and interaction_target_vertices_w_list
                ):
                    mesh_frame_idx = _interaction_mesh_frame_index(frame_float)
                    replay_overlay_handles.extend(
                        self.draw_interaction_meshes(
                            interaction_source_vertices_w_list[mesh_frame_idx],
                            interaction_target_vertices_w_list[mesh_frame_idx],
                            tetrahedra[mesh_frame_idx],
                            name_prefix="world/replay_interaction_mesh",
                        )
                    )

            create_motion_control_sliders(
                server=self.server,
                viser_robot=self.viser_robot,
                robot_base_frame=self.robot_base,
                motion_sequence=np.asarray(retargeted_motions)[1:],
                robot_dof=robot_dof,
                viser_object=self.viser_object,
                object_base_frame=getattr(self, "object_base", None) if self.viser_object else None,
                contains_object_in_qpos=bool(self.viser_object) and bool(self.has_dynamic_object),
                initial_fps=float(fps),
                initial_interp_mult=2,
                loop=False,
                qpos_to_viser_joint_indices=self.qpos_to_viser_joint_indices,
                on_frame=_draw_replay_overlays,
            )

            # 4) optional: visibility toggle
            with self.server.gui.add_folder("Visibility"):
                show_meshes_cb = self.server.gui.add_checkbox("Show meshes", self.viser_robot.show_visual)
                updating_mesh_checkbox = {"flag": False}

                def _set_mesh_visibility(visible: bool, *, sync_checkbox: bool = True) -> None:
                    if sync_checkbox:
                        updating_mesh_checkbox["flag"] = True
                        try:
                            show_meshes_cb.value = bool(visible)
                        finally:
                            updating_mesh_checkbox["flag"] = False
                    self.viser_robot.show_visual = bool(visible)
                    if self.viser_object is not None:
                        self.viser_object.show_visual = bool(visible)

                @show_meshes_cb.on_update
                def _(_):
                    if not updating_mesh_checkbox["flag"]:
                        _set_mesh_visibility(bool(show_meshes_cb.value), sync_checkbox=False)

                register_keyboard_shortcut(
                    self.server,
                    "Display: Toggle Meshes",
                    hotkey=",",
                    callback=lambda: _set_mesh_visibility(not bool(show_meshes_cb.value)),
                )

            self._update_foot_sticking_status(
                0,
                foot_sticking_states[0],
                constraints_enabled=saved_foot_sticking_enabled,
            )

        return (
            np.array(retargeted_motions)[1:],
            obj_pts_demo_list,
            obj_pts_list,
            tetrahedra,
        )

    def solve_single_iteration(
        self,
        q_locked: np.ndarray,
        q_a_n_last: np.ndarray,
        q_t_last: np.ndarray,
        target_laplacian: np.ndarray,
        adj_list: list[list[int]],
        obj_pts_local: np.ndarray,
        foot_sticking: Mapping[str, bool],
        w_nominal_tracking: float = 0.0,
        q_a_nominal: np.ndarray | None = None,
        verbose=False,
        init_t=False,
        frame_idx: int = 0,
        orientation_target_matrices: np.ndarray | None = None,
        shoulder_direction_targets: np.ndarray | None = None,
        root_stability_target_position: np.ndarray | None = None,
        root_stability_target_matrix: np.ndarray | None = None,
        root_stability_bootstrap: bool = False,
    ) -> tuple[np.ndarray, float]:
        """The main function to solve a single iteration of the DiffIK problem.
        Args:
            q_locked: the locked robot and object configuration.
            q_a_n_last: the last optimized robot configuration at current time step.
            q_t_last: the robot and object configuration at the last time step.
            foot_sticking: named left/right source-foot sticking states.
            smpl_joints: the (possibly scaled) SMPL joint positions to match for IK.
            q_ref: the reference robot configuration.
            smpl_joints_original: the original SMPL joint positions (used for contact matching).
            obj_original: the original object pose (used for contact matching).
            init_t: the current time step is the first time step.
            frame_idx: frame index used by explicit foot lock window constraints.
        """
        assert len(q_a_n_last) == self.nq_a

        # Lock the object pose and set the current robot slice to last accepted solution
        q = np.copy(q_locked)
        q[self.q_a_indices] = q_a_n_last

        # Compute Laplacian pieces
        J_OC_dict, p_OC_dict, _ = self._calc_manipulator_jacobians(
            q, links=self.laplacian_match_links, obj_frame=(self.object_name != "ground")
        )
        robot_link_keys = list(self.laplacian_match_links.keys())
        V_r = len(robot_link_keys)
        V_o = len(obj_pts_local)
        V = V_r + V_o

        # Stack Jacobians for robot points
        J_V = np.zeros((3 * V, self.nq_a))
        for i, key in enumerate(robot_link_keys):
            J_V[3 * i : 3 * (i + 1), :] = J_OC_dict[key]

        robot_pts_local = np.array([p_OC_dict[k] for k in robot_link_keys])
        vertices = np.vstack([robot_pts_local, obj_pts_local])  # (V x 3)

        L = calculate_laplacian_matrix(vertices, adj_list)  # (V x V), EXPECT SPARSE OR SMALL
        if not sp.issparse(L):
            L = sp.csr_matrix(L)

        Kron = sp.kron(L, sp.eye(3, format="csr"), format="csr")
        J_L = Kron @ J_V

        lap0 = L @ vertices
        lap0_vec = lap0.reshape(-1)  # (3V,)
        target_lap_vec = target_laplacian.reshape(-1)  # (3V,)

        w_v = self._interaction_mesh_vertex_weights(robot_link_keys, V)
        sqrt_w3 = np.sqrt(np.repeat(w_v, 3))

        # Decision variables
        dqa = cp.Variable(len(self.q_a_indices), name="dqa")
        lap_var = cp.Variable(3 * V, name="laplacian")

        constraints = []

        # Linear equality
        active_laplacian_jacobian = J_L
        constraints += [cp.Constant(active_laplacian_jacobian) @ dqa - lap_var == -lap0_vec]

        # Foot constraints (sticking + foot lock window Z pinning)
        apply_foot_sticking = (self.q_a_init_idx < 12) and self.activate_foot_sticking
        apply_foot_lock = (self.q_a_init_idx < 12) and self.foot_lock.enable
        use_contact_plan = apply_foot_sticking and self._active_planar_foot_targets is not None
        apply_legacy_sticking = apply_foot_sticking and self._active_planar_foot_targets is None
        has_contact_plan_targets = use_contact_plan and bool(self._active_planar_foot_targets)
        if apply_legacy_sticking or has_contact_plan_targets or apply_foot_lock:
            J_WF_dict, p_WF_dict, _ = self._calc_manipulator_jacobians(q, links=self.foot_links, obj_frame=False)

            if has_contact_plan_targets:
                self._append_contact_plan_constraints(
                    constraints,
                    dqa,
                    J_WF_dict,
                    p_WF_dict,
                )

            # Legacy foot sticking: constrain every sole point near its previous-frame position.
            if apply_legacy_sticking:
                _, p_WF_t_last_dict, _ = self._calc_manipulator_jacobians(
                    q_t_last, links=self.foot_links, obj_frame=False
                )
                left_key = right_key = None
                for key in foot_sticking:
                    side = self._name_side(key)
                    if side == "left":
                        left_key = key
                    elif side == "right":
                        right_key = key
                if left_key is None or right_key is None:
                    raise ValueError("foot_sticking must include one left* and one right* key")

                for key, J_WF in J_WF_dict.items():
                    side = self._name_side(key)
                    apply_left = (side == "left") and foot_sticking[left_key]
                    apply_right = (side == "right") and foot_sticking[right_key]
                    if apply_left or apply_right:
                        Jxy = J_WF[:2]
                        delta = p_WF_t_last_dict[key] - p_WF_dict[key]
                        p_lb = delta - self.foot_sticking_tolerance
                        p_ub = delta + self.foot_sticking_tolerance
                        constraints += [
                            Jxy @ dqa >= p_lb[:2],
                            Jxy @ dqa <= p_ub[:2],
                        ]

            # Foot lock windows: pin Z to floor within configured frame ranges
            if apply_foot_lock:
                for key, J_WF in J_WF_dict.items():
                    if not self._is_foot_locked_in_window(key, frame_idx):
                        continue

                    z_anchor = self.foot_lock.z_floor
                    z_delta = z_anchor - p_WF_dict[key][2]
                    Jz = J_WF[2]
                    constraints += [
                        Jz @ dqa >= z_delta - self.foot_lock.tolerance,
                        Jz @ dqa <= z_delta + self.foot_lock.tolerance,
                    ]

        # Non-penetration constraints
        Js, phis = self._update_jacobians_and_phis_from_q(q)
        for key, phi in phis.items():
            Ja_n_full = Js[key]
            Ja_n = Ja_n_full[self.q_a_indices]
            rhs = -phi - self.penetration_tolerance
            constraints += [Ja_n @ dqa >= rhs]

        # Self-collision constraints
        Js_sc, phis_sc = self._compute_self_collision_constraints(frame_idx)
        for key, phi in phis_sc.items():
            Ja_n_full = Js_sc[key]
            Ja_n = Ja_n_full[self.q_a_indices]
            # Enforce: new_distance >= tolerance  =>  phi + J @ dqa >= tol
            rhs = self._self_collision_tolerance - phi
            constraints += [Ja_n @ dqa >= rhs]

        # Joint limits constraints (actuated)
        if self.activate_joint_limits:
            constraints += [
                dqa >= (self.q_a_lb - q_a_n_last),
                dqa <= (self.q_a_ub - q_a_n_last),
            ]

        # Step size constraints (Lorentz cone)
        constraints += [cp.SOC(self.step_size, dqa)]

        # Objective
        obj_terms = []

        obj_terms.append(cp.sum_squares(cp.multiply(sqrt_w3, lap_var - target_lap_vec)))

        if self.root_stability_enabled and not root_stability_bootstrap:
            (
                current_root_position,
                current_root_matrix,
                root_position_jacobian,
                root_orientation_jacobian,
            ) = self._get_root_stability_data(
                q,
                with_jacobians=True,
            )
            position_weight = float(
                self.root_stability_config.position_weight,
            )
            if position_weight > 0.0:
                if root_stability_target_position is None:
                    raise ValueError(
                        "Root position stability requires a frame target",
                    )
                target_position = np.asarray(
                    root_stability_target_position,
                    dtype=np.float64,
                )
                if target_position.shape != (3,):
                    raise ValueError(
                        "Root position stability target must have shape (3,)",
                    )
                if root_position_jacobian is None:
                    raise RuntimeError("Expected a root position Jacobian")
                position_error = target_position - current_root_position
                obj_terms.append(
                    position_weight
                    * cp.sum_squares(
                        root_position_jacobian @ dqa - position_error,
                    ),
                )
            orientation_weight = float(
                self.root_stability_config.orientation_weight,
            )
            if orientation_weight > 0.0:
                if root_stability_target_matrix is None:
                    raise ValueError(
                        "Root orientation stability requires a frame target",
                    )
                target_matrix = np.asarray(
                    root_stability_target_matrix,
                    dtype=np.float64,
                )
                if target_matrix.shape != (3, 3):
                    raise ValueError(
                        "Root orientation stability target must have shape (3, 3)",
                    )
                if root_orientation_jacobian is None:
                    raise RuntimeError("Expected a root orientation Jacobian")
                orientation_error = self._so3_error_vectors(
                    target_matrix[None],
                    current_root_matrix[None],
                )[0]
                obj_terms.append(
                    orientation_weight
                    * cp.sum_squares(
                        root_orientation_jacobian @ dqa - orientation_error,
                    ),
                )

        if self.orientation_tracking_enabled:
            if orientation_target_matrices is None:
                raise ValueError("Orientation tracking requires per-frame target matrices")
            (
                current_orientation_matrices,
                orientation_jacobians,
            ) = self._get_robot_link_orientation_data(
                q,
                self.orientation_robot_link_names,
                with_jacobians=True,
            )
            if orientation_jacobians is None:
                raise RuntimeError("Expected orientation Jacobians")
            orientation_errors = self._so3_error_vectors(
                orientation_target_matrices,
                current_orientation_matrices,
            )
            reserved_orientation_indices = (
                self._reserved_orientation_objective_indices(
                    root_stability_bootstrap=root_stability_bootstrap,
                )
            )
            for link_idx, weight in enumerate(self.orientation_weight_values):
                if link_idx in reserved_orientation_indices:
                    continue
                obj_terms.append(
                    weight * cp.sum_squares(orientation_jacobians[link_idx] @ dqa - orientation_errors[link_idx])
                )
            if self.shoulder_direction_enabled:
                for side_index, weight in enumerate(self.shoulder_wrist_weights):
                    orientation_index = int(
                        self.shoulder_wrist_orientation_indices[side_index],
                    )
                    wrist_dof_count = int(
                        self.shoulder_wrist_dof_counts[side_index],
                    )
                    wrist_reduced_indices = self.shoulder_wrist_reduced_indices[
                        side_index,
                        :wrist_dof_count,
                    ]
                    if (
                        weight <= 0.0
                        or orientation_index < 0
                        or wrist_dof_count <= 0
                    ):
                        continue
                    if wrist_dof_count == 1:
                        wrist_reduced_index = int(wrist_reduced_indices[0])
                        wrist_axis_world = orientation_jacobians[
                            orientation_index,
                            :,
                            wrist_reduced_index,
                        ]
                        axis_norm = float(np.linalg.norm(wrist_axis_world))
                        if axis_norm <= 1e-8:
                            raise RuntimeError(
                                "Wrist orientation Jacobian axis is degenerate",
                            )
                        wrist_axis_world = wrist_axis_world / axis_norm
                        wrist_error = float(
                            wrist_axis_world
                            @ orientation_errors[orientation_index],
                        )
                        wrist_jacobian = np.zeros(self.nq_a, dtype=np.float64)
                        wrist_jacobian[wrist_reduced_index] = float(
                            wrist_axis_world
                            @ orientation_jacobians[
                                orientation_index,
                                :,
                                wrist_reduced_index,
                            ],
                        )
                        obj_terms.append(
                            float(weight)
                            * cp.square(wrist_jacobian @ dqa - wrist_error),
                        )
                    else:
                        wrist_orientation_jacobian = orientation_jacobians[
                            orientation_index,
                        ][:, wrist_reduced_indices]
                        obj_terms.append(
                            float(weight)
                            * cp.sum_squares(
                                wrist_orientation_jacobian
                                @ dqa[wrist_reduced_indices]
                                - orientation_errors[orientation_index],
                            ),
                        )

        if self.shoulder_direction_enabled:
            if shoulder_direction_targets is None:
                raise ValueError("Shoulder direction tracking requires frame targets")
            direction_targets = np.asarray(
                shoulder_direction_targets,
                dtype=np.float64,
            )
            if direction_targets.shape != (2, 3):
                raise ValueError("Shoulder frame targets must have shape (2, 3)")
            current_directions, direction_jacobians = self._get_shoulder_direction_data(
                q,
                with_jacobians=True,
            )
            if direction_jacobians is None:
                raise RuntimeError("Expected shoulder direction Jacobians")
            for side_index in range(2):
                direction_error = direction_targets[side_index] - current_directions[side_index]
                direction_jacobian = direction_jacobians[side_index]
                obj_terms.append(
                    float(self.shoulder_direction_config.direction_weight)
                    * cp.sum_squares(direction_jacobian @ dqa - direction_error),
                )

        # This fixed reference resolves kinematic ambiguity without inheriting
        # the preceding frame's null-space drift as a moving target.
        if self.natural_pose_tracking_enabled:
            natural_pose_candidate = (
                dqa[self.natural_pose_reduced_indices] + q_a_n_last[self.natural_pose_reduced_indices]
            )
            natural_pose_error = natural_pose_candidate - self.natural_pose_reference_values
            obj_terms.append(
                cp.sum_squares(
                    cp.multiply(
                        np.sqrt(self.natural_pose_weight_values),
                        natural_pose_error,
                    ),
                ),
            )

        # nominal tracking for selected indices
        if (w_nominal_tracking > 0) and (q_a_nominal is not None):
            idx = np.array(self.track_nominal_indices, dtype=int)
            if idx.size > 0:
                z = dqa[idx] - (q_a_nominal[idx] - q_a_n_last[idx])
                obj_terms.append(w_nominal_tracking * cp.sum_squares(z))

        # Q_diag cost
        Qd = np.asarray(self.Q_diag, dtype=float).reshape(-1)
        obj_terms.append(cp.sum_squares(cp.multiply(np.sqrt(Qd), dqa + q_a_n_last)))

        # Smoothness cost
        dqa_smooth = q_t_last[self.q_a_indices] - q_a_n_last
        if np.isscalar(self.smooth_weight):
            obj_terms.append(self.smooth_weight * cp.sum_squares(dqa - dqa_smooth))
        else:
            Wsmooth = np.asarray(self.smooth_weight, dtype=float)
            if Wsmooth.ndim == 1:
                obj_terms.append(cp.sum_squares(cp.multiply(np.sqrt(Wsmooth), dqa - dqa_smooth)))
            else:
                # if a full matrix was supplied, fall back to quad_form
                obj_terms.append(cp.quad_form(dqa - dqa_smooth, Wsmooth))

        problem = cp.Problem(cp.Minimize(cp.sum(obj_terms)), constraints)
        solver_kwargs = {"verbose": verbose}
        problem.solve(solver=cp.CLARABEL, **solver_kwargs)
        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and init_t:
            constraints = [
                constraint for constraint in constraints if not isinstance(constraint, cp.constraints.second_order.SOC)
            ]
            problem = cp.Problem(cp.Minimize(cp.sum(obj_terms)), constraints)
            problem.solve(solver=cp.CLARABEL, **solver_kwargs)

        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(f"CVXPY solve failed at frame {frame_idx}: {problem.status}")

        dqa_star = dqa.value
        if dqa_star is None:
            raise RuntimeError(
                f"CVXPY solve returned no candidate at frame {frame_idx}: {problem.status}",
            )
        proposal_step = np.asarray(dqa_star, dtype=np.float64).reshape(-1)
        if proposal_step.shape != (self.nq_a,) or not np.all(np.isfinite(proposal_step)):
            raise RuntimeError(
                f"CVXPY solve returned an invalid step at frame {frame_idx}",
            )

        candidate_q = np.copy(q)
        candidate_q[self.q_a_indices] = q_a_n_last + proposal_step
        candidate_q[3:7] /= np.linalg.norm(candidate_q[3:7]) + 1e-12
        return candidate_q, float(problem.value)

    def _is_foot_locked_in_window(self, foot_link_key: str, frame_idx: int) -> bool:
        """Check whether a foot link is locked by configured frame windows."""
        side = self._name_side(foot_link_key)
        if side is None:
            return False

        return any(start <= frame_idx <= end for start, end in self._foot_lock_windows.get(side, ()))

    def _compute_self_collision_constraints(self, frame_idx: int):
        """Compute Jacobians and distances for self-collision body pairs.

        Assumes ``mj_forward`` has already been called with the current q
        (done by ``_update_jacobians_and_phis_from_q`` which runs first).

        Returns:
            Js: dict mapping (geom_a, geom_b) -> relative Jacobian (1 x nq)
            phis: dict mapping (geom_a, geom_b) -> signed distance
        """
        if not self._self_collision_enabled:
            return {}, {}

        # Check frame windows
        if self._self_collision_windows is not None:
            if not any(start <= frame_idx <= end for start, end in self._self_collision_windows):
                return {}, {}

        m, d = self.robot_model, self.robot_data
        threshold = float(self.collision_detection_threshold)

        Js, phis = {}, {}
        fromto = np.zeros(6, dtype=float)

        if not hasattr(self, "_geom_names"):
            raise RuntimeError(
                "[SelfCollision] _geom_names not initialized. Please build environment collision candidates first."
            )

        _first_iter = self._sc_last_vis_frame != frame_idx
        if _first_iter:
            self._sc_last_vis_frame = frame_idx

        for geom_a, geom_b in self._self_collision_geom_pairs:
            fromto[:] = 0.0
            dist = mujoco.mj_geomDistance(m, d, geom_a, geom_b, threshold, fromto)
            if dist <= threshold:
                J_rel = self._compute_jacobian_for_contact_relative(
                    m.geom(geom_a),
                    m.geom(geom_b),
                    self._geom_names[geom_a],
                    self._geom_names[geom_b],
                    fromto,
                    dist,
                )
                key = ("self", geom_a, geom_b)
                Js[key] = J_rel
                phis[key] = float(dist)

        if _first_iter and self.visualize:
            self._draw_self_collision_geoms()

        return Js, phis

    def iterate(
        self,
        q_locked: np.ndarray,
        q_n: np.ndarray,
        q_t_last: np.ndarray,
        target_laplacian: np.ndarray,
        adj_list: list[list[int]],
        obj_pts_local: np.ndarray,
        foot_sticking: Mapping[str, bool],
        w_nominal_tracking: float = 0.0,
        q_a_nominal: np.ndarray | None = None,
        init_t: bool = False,
        n_iter: int = 10,
        frame_idx: int = 0,
        orientation_target_matrices: np.ndarray | None = None,
        foot_contact_state: Mapping[str, FootContactState] | None = None,
        shoulder_direction_targets: np.ndarray | None = None,
        root_stability_target_position: np.ndarray | None = None,
        root_stability_target_matrix: np.ndarray | None = None,
        root_stability_bootstrap: bool = False,
    ):
        """Apply each successful linearized QP step directly, matching main."""
        max_iterations = int(n_iter)
        if max_iterations <= 0:
            raise ValueError("n_iter must be positive")
        if foot_contact_state is None:
            self._active_planar_foot_targets = None
        else:
            self._prepare_planar_foot_targets(q_t_last, foot_contact_state)

        last_cost = np.inf
        stop_reason = "max_iterations"
        for iteration_idx in range(max_iterations):
            q_n, cost = self.solve_single_iteration(
                q_locked=q_locked,
                q_a_n_last=q_n[self.q_a_indices],
                q_t_last=q_t_last,
                target_laplacian=target_laplacian,
                adj_list=adj_list,
                obj_pts_local=obj_pts_local,
                foot_sticking=foot_sticking,
                q_a_nominal=q_a_nominal,
                w_nominal_tracking=w_nominal_tracking,
                init_t=init_t,
                frame_idx=frame_idx,
                orientation_target_matrices=orientation_target_matrices,
                shoulder_direction_targets=shoulder_direction_targets,
                root_stability_target_position=root_stability_target_position,
                root_stability_target_matrix=root_stability_target_matrix,
                root_stability_bootstrap=root_stability_bootstrap,
            )
            if not np.isfinite(cost):
                raise RuntimeError(
                    f"SQP returned a non-finite local cost at frame {frame_idx}, iteration {iteration_idx}: {cost}",
                )
            if np.isclose(cost, last_cost):
                stop_reason = "cost_stable"
                break
            last_cost = cost

        self.last_sqp_iteration_count = iteration_idx + 1
        self.last_sqp_stop_reason = stop_reason
        return q_n, float(cost)

    def _draw_self_collision_geoms(self):
        """Draw collision cylinders for self-collision geom pairs in viser."""
        if not hasattr(self, "server") or not self._self_collision_enabled:
            return
        m, d = self.robot_model, self.robot_data
        seen_geoms: set[int] = set()
        colors = [(255, 80, 80), (80, 80, 255)]  # red for first body, blue for second
        for geom_a, geom_b in self._self_collision_geom_pairs:
            for idx, gid in enumerate([geom_a, geom_b]):
                if gid in seen_geoms:
                    continue
                seen_geoms.add(gid)
                gtype = int(m.geom_type[gid])
                if gtype not in (3, 5):  # 3 = capsule, 5 = cylinder
                    continue
                radius = float(m.geom_size[gid][0])
                half_len = float(m.geom_size[gid][1])
                cyl = trimesh.creation.capsule(radius=radius, height=2 * half_len, count=[16, 16])
                # World transform from MuJoCo data
                pos = d.geom_xpos[gid]
                rot_mat = d.geom_xmat[gid].reshape(3, 3)
                transform = np.eye(4)
                transform[:3, :3] = rot_mat
                transform[:3, 3] = pos
                cyl.apply_transform(transform)
                body_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[gid]) or ""
                self.server.scene.add_mesh_simple(
                    f"/world/sc_geom/{body_name}_g{gid}",
                    vertices=cyl.vertices.astype(np.float32),
                    faces=cyl.faces.astype(np.int32),
                    color=colors[idx % 2],
                    opacity=0.35,
                )

    def draw_q(self, q: np.ndarray):
        """Draw a single robot configuration."""
        # Update robot joint configurations
        robot_joint_positions = q[7 : 7 + self.task_constants.ROBOT_DOF]
        if self.qpos_to_viser_joint_indices is not None:
            robot_joint_positions = robot_joint_positions[self.qpos_to_viser_joint_indices]
        self.viser_robot.update_cfg(robot_joint_positions)

        # Update robot base pose using set_transform
        robot_quat = q[3:7]  # Base orientation
        robot_pos = q[:3]  # Base position

        # Update robot base frame
        self.robot_base.position = robot_pos
        self.robot_base.wxyz = robot_quat  # Assuming quaternion is in wxyz order

        # Update object pose if it exists
        if hasattr(self, "viser_object") and self.viser_object is not None:
            if self.has_dynamic_object:
                object_quat = q[-4:]
                object_pos = q[-7:-4]
            else:
                object_quat = np.asarray([1, 0, 0, 0])
                object_pos = np.zeros(3)

            # Update object base frame
            self.object_base.position = object_pos
            self.object_base.wxyz = object_quat  # Assuming quaternion is in wxyz order

    def draw_keypoints(self, p, name="keypoint", rgba=(0, 0, 1, 1), radius=0.02):
        """Draw keypoints in visualization."""
        if not hasattr(self, "server"):
            return []

        # Create a sphere mesh using trimesh
        sphere = trimesh.primitives.Sphere(radius=float(radius))
        vertices = sphere.vertices
        faces = sphere.faces

        color = tuple(int(c * 255) for c in rgba[:3])
        opacity = float(rgba[3])

        kpts_handle_list = []

        # Draw keypoints
        if len(p.shape) == 1:
            # Single point
            kpts_handle = self.server.scene.add_mesh_simple(
                f"/{name}",
                vertices=vertices,
                faces=faces,
                position=p,
                color=color,
                opacity=opacity,
            )
            kpts_handle_list.append(kpts_handle)
        elif len(p.shape) == 2:
            # Multiple points
            kpts_handle = self.server.scene.add_batched_meshes_simple(
                f"/{name}",
                vertices=vertices,
                faces=faces,
                batched_positions=p,
                batched_wxyzs=np.tile(np.array([1, 0, 0, 0]), (p.shape[0], 1)),
                batched_colors=color,
                opacity=opacity,
            )
            kpts_handle_list.append(kpts_handle)

        return kpts_handle_list

    def draw_human_hand_skeleton(
        self,
        human_joints: np.ndarray,
        *,
        name: str,
        rgba=(0.25, 0.25, 1.0, 1.0),
    ) -> list[object]:
        """Draw visual-only SMPL-H finger joints without adding optimization anchors."""

        if not hasattr(self, "server"):
            return []
        points = np.asarray(human_joints, dtype=float)
        spec = self._hand_visualization_spec
        if points.ndim != 2 or not spec.keypoint_indices or not spec.edge_indices:
            return []

        handles = self.draw_keypoints(
            points[np.asarray(spec.keypoint_indices, dtype=int)],
            name=f"{name}_kpts",
            rgba=rgba,
            radius=0.012,
        )
        segments = np.asarray(
            [[points[start], points[end]] for start, end in spec.edge_indices],
            dtype=np.float32,
        )
        color = np.asarray(rgba[:3], dtype=float)
        colors = np.tile(color, (segments.shape[0], 2, 1))
        handles.append(
            self.server.scene.add_line_segments(
                f"/{name}_skeleton",
                points=segments,
                colors=colors,
                line_width=1.5,
            )
        )
        return handles

    def _mapped_skeleton_edges(self) -> list[tuple[int, int]]:
        """Build skeleton edges in the current JOINTS_MAPPING key order."""
        joint_names = list(self.laplacian_match_links.keys())
        joint_idx = {name: idx for idx, name in enumerate(joint_names)}
        candidate_edges = [
            ("Pelvis", "L_Hip"),
            ("L_Hip", "L_Knee"),
            ("L_Knee", "L_Ankle"),
            ("L_Ankle", "L_Toe"),
            ("L_Ankle", "L_Foot"),
            ("Pelvis", "R_Hip"),
            ("R_Hip", "R_Knee"),
            ("R_Knee", "R_Ankle"),
            ("R_Ankle", "R_Toe"),
            ("R_Ankle", "R_Foot"),
            ("Pelvis", "L_Shoulder"),
            ("L_Shoulder", "L_Elbow"),
            ("L_Elbow", "L_Wrist"),
            ("Pelvis", "R_Shoulder"),
            ("R_Shoulder", "R_Elbow"),
            ("R_Elbow", "R_Wrist"),
            ("L_Shoulder", "R_Shoulder"),
            ("L_Hip", "R_Hip"),
            ("Spine1", "LeftUpLeg"),
            ("LeftUpLeg", "LeftLeg"),
            ("LeftLeg", "LeftFoot"),
            ("LeftFoot", "LeftToeBase"),
            ("Spine1", "RightUpLeg"),
            ("RightUpLeg", "RightLeg"),
            ("RightLeg", "RightFoot"),
            ("RightFoot", "RightToeBase"),
            ("Spine1", "LeftArm"),
            ("LeftArm", "LeftForeArm"),
            ("LeftForeArm", "LeftHand"),
            ("LeftForeArm", "LeftHandMiddle3"),
            ("Spine1", "RightArm"),
            ("RightArm", "RightForeArm"),
            ("RightForeArm", "RightHand"),
            ("RightForeArm", "RightHandMiddle3"),
            ("LeftArm", "RightArm"),
            ("LeftUpLeg", "RightUpLeg"),
        ]
        return [(joint_idx[a], joint_idx[b]) for a, b in candidate_edges if a in joint_idx and b in joint_idx]

    def draw_mapped_skeleton(self, p, name="skeleton", rgba=(0, 0, 1, 1), line_width=2.0):
        """Draw line segments connecting mapped keypoints in skeleton order."""
        if not hasattr(self, "server"):
            return []

        points = np.asarray(p, dtype=float)
        edges = self._mapped_skeleton_edges()
        if points.ndim != 2 or not edges:
            return []

        segments = np.asarray([[points[i], points[j]] for i, j in edges], dtype=np.float32)
        color = np.asarray(rgba[:3], dtype=float)
        colors = np.tile(color, (segments.shape[0], 2, 1))
        handle = self.server.scene.add_line_segments(
            f"/{name}",
            points=segments,
            colors=colors,
            line_width=line_width,
        )
        return [handle]

    def visualize_motion(
        self,
        human_joint_motions,
        obj_pts_demo,
        obj_pts,
        retargeted_motions,
        tetrahedra,
        dt=1 / 30,
        visualize_tetrahedra=False,
    ):
        for i in range(len(human_joint_motions)):
            object_pts_demo = obj_pts_demo[i]
            object_pts = obj_pts[i]
            self.draw_keypoints(human_joint_motions[i, self.mapped_joint_indices], name="human")
            self.draw_keypoints(object_pts_demo, name="object_demo", rgba=(1, 0, 0, 1))
            self.draw_keypoints(object_pts, name="object", rgba=(0, 1, 0, 1))
            self.draw_q(retargeted_motions[i])
            robot_link_positions = self._get_robot_link_positions(
                retargeted_motions[i], self.laplacian_match_links.values()
            )
            self.draw_keypoints(robot_link_positions, name="robot", rgba=(0, 1, 0, 1))
            input()
            if visualize_tetrahedra:
                self.visualize_tetrahedra(
                    np.vstack(
                        [
                            human_joint_motions[i, self.mapped_joint_indices],
                            object_pts_demo,
                        ]
                    ),
                    tetrahedra[i],
                    name="human_tetrahedra",
                )
                self.visualize_tetrahedra(
                    np.vstack([robot_link_positions, object_pts]),
                    tetrahedra[i],
                    name="robot_tetrahedra",
                    rgba=(0, 1, 1, 1),
                )
            else:
                time.sleep(dt)

    def visualize_tetrahedra(self, vertices, tetrahedra, name="tetrahedra", color=(0, 0, 0, 1), rgba=None):
        """Compatibility wrapper for older manual tetrahedra visualization calls."""
        if rgba is not None:
            color = rgba
        return self.draw_interaction_mesh(
            vertices,
            tetrahedra,
            name=name,
            color=tuple(color[:3]),
            edge_mode="all",
        )

    @staticmethod
    def _as_scalar_id(value, name: str) -> int:
        """Normalize MuJoCo named-accessor IDs across NumPy/MuJoCo versions."""
        array = np.asarray(value)
        if array.size != 1:
            raise ValueError(f"{name} must contain exactly one ID, got shape {array.shape}.")
        return int(array.reshape(-1)[0])

    def _compute_jacobian_for_contact_relative(self, geom1, geom2, geom1_name, geom2_name, fromto, dist):
        # Get closest points from fromto buffer
        pos1 = fromto[:3]  # closest point on geom1
        pos2 = fromto[3:]  # closest point on geom2

        v = pos1 - pos2
        norm_v = np.linalg.norm(v)

        if norm_v > 1e-12:
            nhat_BA_W = np.sign(dist) * (v / norm_v)
        # Degenerate: points coincide. Heuristics fallback.
        # If one side is a plane/ground, use its known normal.
        elif "ground" in geom2_name.lower():
            nhat_BA_W = np.array([0.0, 0.0, 1.0]) * (1.0 if dist >= 0 else -1.0)
        elif "ground" in geom1_name.lower():
            nhat_BA_W = np.array([0.0, 0.0, -1.0]) * (1.0 if dist >= 0 else -1.0)
        else:
            nhat_BA_W = np.array([0.0, 0.0, 0.0])

        body1_id = self._as_scalar_id(geom1.bodyid, f"{geom1_name}.bodyid")
        body2_id = self._as_scalar_id(geom2.bodyid, f"{geom2_name}.bodyid")
        J_bodyA = self._calc_contact_jacobian_from_point(body1_id, pos1, input_world=True)
        J_bodyB = self._calc_contact_jacobian_from_point(body2_id, pos2, input_world=True)

        # Compute relative Jacobian
        Jc = J_bodyA - J_bodyB

        return nhat_BA_W @ Jc

    def _environment_collision_candidates(self):
        """Return collision-mask-compatible ground and object geom pairs.

        The candidate set depends only on the compiled model and retargeter
        configuration, so build it once instead of running MuJoCo's complete
        collision pipeline during every SQP iteration.  In particular, this
        avoids generating contact manifolds for robot self-collision pairs
        that are discarded by the environment constraints anyway.
        """

        cached_candidates = getattr(self, "_environment_collision_candidates_cache", None)
        if cached_candidates is not None:
            return cached_candidates

        m = self.robot_model
        ngeom = m.ngeom

        self._geom_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(ngeom)]
        candidates = set()
        contype, conaff = m.geom_contype, m.geom_conaffinity
        for g1 in range(ngeom):
            if contype[g1] == 0 and conaff[g1] == 0:
                continue
            for g2 in range(g1 + 1, ngeom):
                if contype[g2] == 0 and conaff[g2] == 0:
                    continue
                masks_match = (int(contype[g1]) & int(conaff[g2])) or (int(contype[g2]) & int(conaff[g1]))
                if not masks_match:
                    continue
                if self._environment_collision_pair_is_active(
                    self._geom_names[g1],
                    self._geom_names[g2],
                ):
                    candidates.add((g1, g2))

        self._environment_collision_candidates_cache = frozenset(candidates)
        return self._environment_collision_candidates_cache

    def _update_jacobians_and_phis_from_q(self, q: np.ndarray):
        self.robot_data.qpos[:] = q

        mujoco.mj_forward(self.robot_model, self.robot_data)  # kinematics & AABBs valid

        m, d = self.robot_model, self.robot_data
        threshold = float(self.collision_detection_threshold)

        # Candidate topology is static; precise distances are evaluated below.
        candidates = self._environment_collision_candidates()

        Js, phis = {}, {}
        fromto = np.zeros(6, dtype=float)

        # Compute precise distances only for relevant environment pairs.
        for g1, g2 in candidates:
            fromto[:] = 0.0
            dist = mujoco.mj_geomDistance(m, d, g1, g2, threshold, fromto)
            if dist <= threshold:
                J_rel = self._compute_jacobian_for_contact_relative(
                    m.geom(g1), m.geom(g2), self._geom_names[g1], self._geom_names[g2], fromto, dist
                )
                Js[(g1, g2)] = J_rel
                phis[(g1, g2)] = float(dist)

                # For debug
                # self.draw_mesh_pair_with_contact(self.robot_model, self.robot_data, g1, g2,   \
                #     self._geom_names[g1], self._geom_names[g2], fromto=fromto)

        return Js, phis

    def _environment_collision_pair_is_active(self, geom1_name: str, geom2_name: str) -> bool:
        """Select ground pairs and, when enabled, object pairs for hard constraints."""

        geom1_lower = geom1_name.lower()
        geom2_lower = geom2_name.lower()
        geom1_is_ground = "ground" in geom1_lower
        geom2_is_ground = "ground" in geom2_lower
        has_dynamic_object = self.object_name not in {"", "ground"}
        geom1_is_object = has_dynamic_object and self.object_name in geom1_name
        geom2_is_object = has_dynamic_object and self.object_name in geom2_name

        if (geom1_is_object and geom2_is_ground) or (geom1_is_ground and geom2_is_object):
            return False
        if geom1_is_ground or geom2_is_ground:
            return True
        if geom1_is_object or geom2_is_object:
            return self.activate_obj_non_penetration
        return False

    def _world_to_body_frame(self, p_w: np.ndarray, body_idx: int) -> np.ndarray:
        """Transform point from world frame to body frame."""
        p_w = np.asarray(p_w).reshape(3)
        body_pos = self.robot_data.xpos[body_idx].reshape(3)
        body_mat = self.robot_data.xmat[body_idx].reshape(3, 3)
        return body_mat.T @ (p_w - body_pos)

    def _get_geometry_name(self, geom_id: int) -> str:
        """Get geometry name from ID."""
        return mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)

    def _build_transform_qdot_to_qvel_fast(self, use_world_omega=True):
        """
        Return T(q) (nv x nq) such that v = T(q) @ qdot.
        - Free root: qpos=[x,y,z, qw,qx,qy,qz], qvel=[vx,vy,vz, ωx,ωy,ωz]
        where ω and v are WORLD-expressed in MuJoCo.
        - 23 hinge joints: v = qdot.

        If use_world_omega=False, uses BODY-omega mapping (for debugging).
        """
        nq, nv = self.robot_model.nq, self.robot_model.nv
        T = np.zeros((nv, nq), dtype=float)

        # ---- root free joint (assumed joint 0) ----
        j0 = 0
        assert self.robot_model.jnt_type[j0] == mujoco.mjtJoint.mjJNT_FREE
        qadr = self.robot_model.jnt_qposadr[j0]  # 0
        dadr = self.robot_model.jnt_dofadr[j0]  # 0

        # Linear block: v_lin = xyz_dot
        T[dadr : dadr + 3, qadr : qadr + 3] = np.eye(3)

        def get_e_world(qw, qx, qy, qz):
            return np.array(
                [
                    [-qx, qw, qz, -qy],
                    [-qy, -qz, qw, qx],
                    [-qz, qy, -qx, qw],
                ]
            )

        def get_e_body(qw, qx, qy, qz):
            return np.array(
                [
                    [-qx, qw, -qz, qy],
                    [-qy, qz, qw, -qx],
                    [-qz, -qy, qx, qw],
                ]
            )

        E_fn = get_e_world if use_world_omega else get_e_body

        # ---- FREE joint #1 (human/root): use model addresses, but this should be the first joint ----
        j_free1 = 0
        assert self.robot_model.jnt_type[j_free1] == mujoco.mjtJoint.mjJNT_FREE
        qadr1 = int(self.robot_model.jnt_qposadr[j_free1])  # expect 0
        dadr1 = int(self.robot_model.jnt_dofadr[j_free1])  # start of its 6 qvel dofs

        qw, qx, qy, qz = self.robot_data.qpos[qadr1 + 3 : qadr1 + 7]
        E1 = 2.0 * E_fn(qw, qx, qy, qz)
        # linear-first: v_W = rdot, ω_W = 2E(q) * quat_dot
        T[dadr1 + 0 : dadr1 + 3, qadr1 + 0 : qadr1 + 3] = np.eye(3)  # v block
        T[dadr1 + 3 : dadr1 + 6, qadr1 + 3 : qadr1 + 7] = E1  # ω block

        if self.has_dynamic_object:
            # ---- FREE joint #2 (object): assume it's the last FREE joint; fill its 6x7 block ----
            # Find it by type (safer than hardcoding tail indices)
            free_joints = [
                j for j in range(self.robot_model.njnt) if self.robot_model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
            ]
            assert len(free_joints) >= 2, "Expected two FREE joints (human + object)."
            j_free2 = free_joints[1]  # second FREE joint
            qadr2 = int(self.robot_model.jnt_qposadr[j_free2])  # expect nq-7
            dadr2 = int(self.robot_model.jnt_dofadr[j_free2])  # its 6 qvel dofs (often at nv-6)

            qw, qx, qy, qz = self.robot_data.qpos[qadr2 + 3 : qadr2 + 7]
            E2 = 2.0 * E_fn(qw, qx, qy, qz)
            T[dadr2 + 0 : dadr2 + 3, qadr2 + 0 : qadr2 + 3] = np.eye(3)  # v block
            T[dadr2 + 3 : dadr2 + 6, qadr2 + 3 : qadr2 + 7] = E2  # ω block

        # ---- remaining hinge/slide joints: v = qdot ----
        for j in range(1, self.robot_model.njnt):
            jt = int(self.robot_model.jnt_type[j])
            if jt in (
                int(mujoco.mjtJoint.mjJNT_HINGE),
                int(mujoco.mjtJoint.mjJNT_SLIDE),
            ):
                qa = self.robot_model.jnt_qposadr[j]
                da = self.robot_model.jnt_dofadr[j]
                T[da, qa] = 1.0
            elif jt == int(mujoco.mjtJoint.mjJNT_BALL):
                raise NotImplementedError("BALL joint block not implemented.")

        return T

    def _calc_contact_jacobian_from_point(self, body_idx: int, p_body: np.ndarray, input_world=False):
        """
        Translational Jacobian J(q) (3 x nq) such that
        v_point_world = J(q) @ qdot.

        Fast analytic version: J_qdot = J_v @ T(q)
        """

        p_body = np.asarray(p_body, dtype=float).reshape(3)

        # 1) Make sure kinematics are current once
        mujoco.mj_forward(self.robot_model, self.robot_data)

        # 2) World point (3,1) for mj_jac
        R_WB = self.robot_data.xmat[body_idx].reshape(3, 3)
        p_WB = self.robot_data.xpos[body_idx]

        if input_world:
            p_W = p_body.astype(np.float64).reshape(3, 1)
        else:
            p_W = (p_WB + R_WB @ p_body).astype(np.float64).reshape(3, 1)

        # 3) J_v: translational Jacobian wrt generalized velocities (3 x nv)
        Jp = np.zeros((3, self.robot_model.nv), dtype=np.float64, order="C")
        Jr = np.zeros((3, self.robot_model.nv), dtype=np.float64, order="C")
        mujoco.mj_jac(self.robot_model, self.robot_data, Jp, Jr, p_W, int(body_idx))  # Jp = J_v

        T = self._build_transform_qdot_to_qvel_fast()

        return Jp @ T

    def _calc_manipulator_jacobians(
        self,
        q: np.ndarray,
        links: dict[str, str],
        obj_frame: bool = False,
        point_offsets: np.ndarray | None = None,
    ):
        """Compute position-based Jacobians using MuJoCo."""
        J_XC_dict = {}
        p_XC_dict = {}

        if obj_frame:
            if self.has_dynamic_object:
                obj_quat = q[-4:]
                obj_pos = q[-7:-4]
                obj_rot = Rotation.from_quat([obj_quat[1], obj_quat[2], obj_quat[3], obj_quat[0]]).as_matrix()
                obj_rot_inv = obj_rot.T
            else:
                obj_rot = Rotation.from_quat([0, 0, 0, 1]).as_matrix()
                obj_rot_inv = obj_rot.T
                obj_pos = np.zeros(3)

        q_mujoco = q.copy()
        self.robot_data.qpos[:] = q_mujoco

        mujoco.mj_forward(self.robot_model, self.robot_data)

        for name, link_name in links.items():
            body_id = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, link_name)

            if point_offsets is not None:
                pC_B = point_offsets
            else:
                pC_B = np.zeros(3)

            J = self._calc_contact_jacobian_from_point(body_id, pC_B)
            pos_world = self.robot_data.xpos[body_id]

            if obj_frame:
                p_XC = obj_rot_inv @ (pos_world - obj_pos)
                J_XC = obj_rot_inv @ J
            else:
                p_XC = pos_world
                J_XC = J

            # Store reduced Jacobian and position with hard copies to avoid aliasing.
            J_XC_dict[name] = np.array(J_XC[:, self.q_a_indices], dtype=float, copy=True)
            p_XC_dict[name] = np.array(p_XC, dtype=float, copy=True)

        P_WO = {"position": obj_pos, "rotation": obj_rot} if obj_frame else None

        return J_XC_dict, p_XC_dict, P_WO

    def _get_robot_link_positions(self, q, link_names):
        """Get robot link positions for given configuration using Mujoco."""
        mujoco_q = q.copy()

        # Set the configuration
        if mujoco_q.shape != self.robot_data.qpos.shape:
            self.robot_data.qpos = mujoco_q[:-7]  # Exclude object information from q
        else:
            self.robot_data.qpos = mujoco_q
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
