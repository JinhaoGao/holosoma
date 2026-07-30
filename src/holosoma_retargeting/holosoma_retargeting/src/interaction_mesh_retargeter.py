# ruff: noqa: CPY001, PLR0917

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable, Mapping
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

from holosoma_retargeting.config_types.retargeter import FootLockConfig, SelfCollisionConfig
from holosoma_retargeting.data_utils.hand_skeleton import build_hand_visualization_spec
from holosoma_retargeting.result_artifact import (
    FRAME_ZERO_GROUND_RETRY_POLICY,
    RESULT_SCHEMA_VERSION,
    build_object_asset_manifest,
    collision_interior_margin_m,
    compute_human_orientation_sha256,
    write_result_artifact,
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

NONLINEAR_FEASIBILITY_ATOL = 1e-6
NONLINEAR_BACKTRACK_BISECTION_ITERATIONS = 20


@dataclass(frozen=True)
class ConstraintMode:
    """The exact hard-constraint mode used to produce one SQP candidate."""

    foot_sticking: str
    object_non_penetration_released: bool = False
    trust_region_released: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.foot_sticking, str) or self.foot_sticking not in {
            "inactive",
            "normal",
            "relaxed",
            "released",
        }:
            raise ValueError(f"Unknown foot-sticking constraint mode: {self.foot_sticking}")
        for field_name in (
            "object_non_penetration_released",
            "trust_region_released",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, (bool, np.bool_)):
                raise TypeError(f"{field_name} must be boolean, got {type(value).__name__}")


@dataclass(frozen=True)
class NonlinearConstraintResiduals:
    """True-geometry hard-constraint violations for one candidate."""

    ground_non_penetration: float
    object_non_penetration: float
    foot_sticking: float
    foot_lock: float
    self_collision: float
    joint_limits: float

    def __post_init__(self) -> None:
        for field_name in (
            "ground_non_penetration",
            "object_non_penetration",
            "foot_sticking",
            "foot_lock",
            "self_collision",
            "joint_limits",
        ):
            value = float(getattr(self, field_name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(
                    f"{field_name} must be finite and non-negative, got {value}",
                )

    def hard_max_violation(self, mode: ConstraintMode) -> float:
        """Return the maximum violation among constraints active in ``mode``."""

        values = [
            self.ground_non_penetration,
            self.foot_lock,
            self.self_collision,
            self.joint_limits,
        ]
        if not mode.object_non_penetration_released:
            values.append(self.object_non_penetration)
        if mode.foot_sticking not in {"inactive", "released"}:
            values.append(self.foot_sticking)
        return float(max(values, default=0.0))

    def is_feasible(
        self,
        mode: ConstraintMode,
        *,
        atol: float = NONLINEAR_FEASIBILITY_ATOL,
    ) -> bool:
        return self.hard_max_violation(mode) <= atol


@dataclass(frozen=True)
class SQPIterationResult:
    """One linearized solve and its nonlinear acceptance diagnostics."""

    q: np.ndarray
    linearized_cost: float
    constraint_mode: ConstraintMode
    residuals: NonlinearConstraintResiduals

    def __iter__(self):
        """Preserve the historical ``q, cost = solve_single_iteration(...)`` API."""

        yield self.q
        yield self.linearized_cost


@dataclass(frozen=True)
class FrameZeroGroundRetryState:
    """One top-level motion call's single, auditable ground-retry state."""

    eligible: bool
    attempted: bool = False
    triggered: bool = False
    initial_min_distance_m: float = 0.0
    corrected_min_distance_m: float = 0.0
    lift_m: float = 0.0
    interior_margin_m: float = 0.0
    initial_sqp_iterations: int = 0
    corrected_q: np.ndarray | None = None


class SQPNonlinearFeasibilityError(RuntimeError):
    """A full mode schedule failed, with structured foot-causality evidence."""

    def __init__(
        self,
        message: str,
        *,
        foot_related_failure: bool,
        frame_idx: int | None = None,
        total_iterations: int | None = None,
        closest_residuals: NonlinearConstraintResiduals | None = None,
        closest_constraint_mode: ConstraintMode | None = None,
        minimum_hard_constraint_violation: float | None = None,
    ) -> None:
        super().__init__(message)
        self.foot_related_failure = bool(foot_related_failure)
        self.frame_idx = frame_idx
        self.total_iterations = total_iterations
        self.closest_residuals = closest_residuals
        self.closest_constraint_mode = closest_constraint_mode
        self.minimum_hard_constraint_violation = minimum_hard_constraint_violation

    def __reduce__(self):
        """Serialize every structured diagnostic across spawned worker pools."""

        return (
            _reconstruct_sqp_nonlinear_feasibility_error,
            (
                str(self),
                self.foot_related_failure,
                self.frame_idx,
                self.total_iterations,
                self.closest_residuals,
                self.closest_constraint_mode,
                self.minimum_hard_constraint_violation,
            ),
        )


def _reconstruct_sqp_nonlinear_feasibility_error(
    message: str,
    foot_related_failure: bool,
    frame_idx: int | None,
    total_iterations: int | None,
    closest_residuals: NonlinearConstraintResiduals | None,
    closest_constraint_mode: ConstraintMode | None,
    minimum_hard_constraint_violation: float | None,
) -> SQPNonlinearFeasibilityError:
    """Rebuild a structured feasibility error after multiprocessing transport."""

    return SQPNonlinearFeasibilityError(
        message,
        foot_related_failure=foot_related_failure,
        frame_idx=frame_idx,
        total_iterations=total_iterations,
        closest_residuals=closest_residuals,
        closest_constraint_mode=closest_constraint_mode,
        minimum_hard_constraint_violation=minimum_hard_constraint_violation,
    )


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
        foot_sticking_fallback_tolerance: float | None = 0.02,
        release_foot_sticking_on_infeasible: bool = True,
        release_object_non_penetration_on_infeasible: bool = False,
        retry_without_foot_sticking_on_infeasible: bool = True,
        retry_frame_zero_ground_on_infeasible: bool = True,
        foot_lock: FootLockConfig | None = None,
        self_collision: SelfCollisionConfig | None = None,
        sqp_max_iterations: int = 50,
        sqp_min_iterations: int = 4,
        sqp_convergence_patience: int = 3,
        visualize: bool = False,
        mesh_opacity: float = 1.0,
        debug: bool = False,
        show_interaction_mesh: bool = False,
        save_interaction_mesh: bool = True,
        interaction_mesh_mode: str = "both",
        interaction_mesh_edges: str = "cross",
        interaction_mesh_line_width: float = 1.0,
        w_nominal_tracking_init: float = 5.0,
        nominal_tracking_tau: float = 10.0,
        orientation_joints_mapping: dict[str, str] | None = None,
        orientation_weights: dict[str, float] | None = None,
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
    ):
        """This kinematic retargeter solves the diffIK problem with hard constraints in SQP style.
        During each SQP iteration, the problem is solved with the following constraints and costs:
            1. [Cost] Minimize the Laplacian deformation in the object frame.
            2. [Constraint] Enforce the non-penetration constraints w/ the ground and (if activated) the object.
            3. [Constraint] Enforce the foot sticking constraints if activated.
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
            foot_sticking_tolerance: tolerance for foot sticking constraints in x, y.
            foot_sticking_fallback_tolerance: relaxed x/y tolerance used only when
                the normal foot-sticking problem is infeasible. None disables the fallback.
            release_foot_sticking_on_infeasible: release foot-sticking constraints
                only on a frame that remains infeasible after the relaxed retry.
            release_object_non_penetration_on_infeasible: release robot-object
                non-penetration constraints only on a frame that remains
                infeasible after foot-sticking fallbacks. Ground constraints
                remain enabled.
            retry_without_foot_sticking_on_infeasible: retry the full sequence
                without foot sticking if a local release remains infeasible.
            retry_frame_zero_ground_on_infeasible: permit one fail-closed,
                robot-only frame-zero retry that lifts only an optimized root Z
                coordinate after a pure horizontal-ground feasibility failure.
            foot_lock: configuration for explicit frame-range based foot locking constraints.
            sqp_max_iterations: safety cap for SQP iterations per frame.
            sqp_min_iterations: minimum iterations before convergence-based stopping.
            sqp_convergence_patience: consecutive stable feasible iterations
                required before stopping.
            nominal_tracking_tau: the time constant for the nominal tracking cost.
        """

        self.robot_model_path = task_constants.ROBOT_URDF_FILE
        if object_urdf_path:
            object_path = Path(object_urdf_path).expanduser()
            if not object_path.is_absolute():
                object_path = Path.cwd() / object_path
            (
                self.object_asset_manifest_json,
                self.object_asset_manifest_sha256,
            ) = build_object_asset_manifest(object_path)
            manifest = json.loads(self.object_asset_manifest_json)
            self.object_model_path = str(manifest["urdf"]["path"])
            self.object_urdf_sha256 = str(manifest["urdf"]["sha256"])
        else:
            self.object_model_path = None
            self.object_urdf_sha256 = ""
            self.object_asset_manifest_json = ""
            self.object_asset_manifest_sha256 = ""
        self.object_name = task_constants.OBJECT_NAME
        self.collision_detection_threshold = collision_detection_threshold
        self.activate_foot_sticking = activate_foot_sticking
        self.activate_obj_non_penetration = activate_obj_non_penetration
        self.activate_joint_limits = activate_joint_limits
        self.foot_links = dict(zip(task_constants.FOOT_STICKING_LINKS, task_constants.FOOT_STICKING_LINKS))
        self.penetration_tolerance = float(penetration_tolerance)
        if not np.isfinite(self.penetration_tolerance) or self.penetration_tolerance < 0:
            raise ValueError("penetration_tolerance must be finite and non-negative")
        self.step_size = step_size
        self.visualize = visualize
        self.mesh_opacity = float(mesh_opacity)
        self.debug = debug
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
        self.task_constants = task_constants
        self.qpos_to_viser_joint_indices: np.ndarray | None = None

        self.mapped_joint_indices = [self.demo_joints.index(name) for name in self.laplacian_match_links]
        self._hand_visualization_spec = build_hand_visualization_spec(
            self.demo_joints,
            list(self.laplacian_match_links),
        )

        # Setup weights and parameters
        self.laplacian_weights = 10
        self.smooth_weight = 0.2
        # Tolerance for foot sticking constraints in x, y.
        self.foot_sticking_tolerance = float(foot_sticking_tolerance)
        self.foot_sticking_fallback_tolerance = (
            None if foot_sticking_fallback_tolerance is None else float(foot_sticking_fallback_tolerance)
        )
        self.release_foot_sticking_on_infeasible = bool(release_foot_sticking_on_infeasible)
        self.release_object_non_penetration_on_infeasible = bool(release_object_non_penetration_on_infeasible)
        self.retry_without_foot_sticking_on_infeasible = bool(retry_without_foot_sticking_on_infeasible)
        self.retry_frame_zero_ground_on_infeasible = bool(
            retry_frame_zero_ground_on_infeasible,
        )
        if self.foot_sticking_tolerance < 0:
            raise ValueError("foot_sticking_tolerance must be non-negative")
        if (
            self.foot_sticking_fallback_tolerance is not None
            and self.foot_sticking_fallback_tolerance < self.foot_sticking_tolerance
        ):
            raise ValueError(
                "foot_sticking_fallback_tolerance must be greater than or equal to foot_sticking_tolerance"
            )
        self.foot_sticking_fallback_frames: set[int] = set()
        self.foot_sticking_release_frames: set[int] = set()
        self.object_non_penetration_release_frames: set[int] = set()
        self.foot_sticking_full_sequence_retry_frame: int | None = None
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

        self.sqp_max_iterations = int(sqp_max_iterations)
        self.sqp_min_iterations = int(sqp_min_iterations)
        self.sqp_convergence_patience = int(sqp_convergence_patience)
        if self.sqp_max_iterations <= 0:
            raise ValueError("sqp_max_iterations must be positive")
        if self.sqp_min_iterations <= 0:
            raise ValueError("sqp_min_iterations must be positive")
        if self.sqp_min_iterations > self.sqp_max_iterations:
            raise ValueError("sqp_min_iterations must not exceed sqp_max_iterations")
        if self.sqp_convergence_patience <= 0:
            raise ValueError("sqp_convergence_patience must be positive")
        self.last_sqp_iteration_count = 0
        self.last_sqp_stop_reason = "not_started"
        self.last_constraint_mode = ConstraintMode("inactive")
        self.last_nonlinear_constraint_residuals = NonlinearConstraintResiduals(
            ground_non_penetration=0.0,
            object_non_penetration=0.0,
            foot_sticking=0.0,
            foot_lock=0.0,
            self_collision=0.0,
            joint_limits=0.0,
        )

        self.w_nominal_tracking_init = w_nominal_tracking_init
        self.nominal_tracking_tau = nominal_tracking_tau
        self.track_nominal_indices = self._reduced_indices_for_qpos_addresses(
            task_constants.NOMINAL_TRACKING_INDICES,
            metadata_name="NOMINAL_TRACKING_INDICES",
        )
        self._init_orientation_tracking(
            orientation_joints_mapping=orientation_joints_mapping,
            orientation_weights=orientation_weights,
            orientation_alignment_mode=orientation_alignment_mode,
            orientation_t_pose_human_quaternions_wxyz=(orientation_t_pose_human_quaternions_wxyz),
            orientation_t_pose_robot_base_quaternion_wxyz=(orientation_t_pose_robot_base_quaternion_wxyz),
            orientation_t_pose_robot_joint_positions=(orientation_t_pose_robot_joint_positions),
            orientation_alignment_quaternions_wxyz=(orientation_alignment_quaternions_wxyz),
        )

    def _init_orientation_tracking(
        self,
        *,
        orientation_joints_mapping: dict[str, str] | None,
        orientation_weights: dict[str, float] | None,
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
        self.orientation_diagnostics_enabled = bool(tracked_human_joints)
        self.orientation_tracking_enabled = bool(np.any(self.orientation_weight_values > 0.0))
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
        elif frame_idx in self.foot_sticking_release_frames:
            constraint_status = "released after infeasible solve"
        elif frame_idx in self.foot_sticking_fallback_frames:
            constraint_status = "active with relaxed tolerance"
        elif bool(np.any(states)):
            constraint_status = "active"
        else:
            constraint_status = "inactive (no sticking foot)"

        self._foot_sticking_status_handle.content = format_foot_sticking_status(
            frame_idx,
            states,
            constraint_status=constraint_status,
        )

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
        if not self.orientation_diagnostics_enabled:
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
        orientation_target_world_rotation_deltas_wxyz: np.ndarray | None = None,
        result_metadata: dict[str, object] | None = None,
        _foot_sticking_retry: bool = False,
        _frame_zero_ground_retry_state: FrameZeroGroundRetryState | None = None,
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
        frame_zero_ground_retry_eligible = self._frame_zero_ground_retry_eligible(
            q_nominal_list=q_nominal_list,
            original=original,
        )
        if _frame_zero_ground_retry_state is None:
            frame_zero_ground_retry_state = FrameZeroGroundRetryState(
                eligible=frame_zero_ground_retry_eligible,
            )
        else:
            frame_zero_ground_retry_state = _frame_zero_ground_retry_state
            if frame_zero_ground_retry_state.eligible != frame_zero_ground_retry_eligible:
                raise ValueError(
                    "Inherited frame-zero ground retry state no longer matches the current motion invocation",
                )
        (
            saved_human_joint_quaternions_wxyz,
            normalized_math_human_joint_quaternions_wxyz,
            source_orientation_joint_names,
        ) = self._validate_source_orientations(
            human_joint_quaternions_wxyz,
            human_orientation_joint_names,
            num_frames,
        )
        foot_sticking_states = self._foot_sticking_states_array(
            foot_sticking_sequences,
            num_frames,
        )
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
        ground_non_penetration_violations = []
        object_non_penetration_violations = []
        foot_sticking_violations = []
        foot_lock_violations = []
        self_collision_violations = []
        joint_limits_violations = []
        constraint_mode_foot_sticking = []
        constraint_mode_object_non_penetration_released = []
        constraint_mode_trust_region_released = []
        orientation_robot_quaternions_wxyz: list[np.ndarray] = []
        orientation_errors_rad: list[np.ndarray] = []
        orientation_frame_costs: list[float] = []
        self.foot_sticking_fallback_frames.clear()
        self.foot_sticking_release_frames.clear()
        self.object_non_penetration_release_frames.clear()
        if not _foot_sticking_retry:
            self.foot_sticking_full_sequence_retry_frame = None
        collect_interaction_mesh = self.save_interaction_mesh or (self.visualize and self.show_interaction_mesh)
        collect_object_point_trajectories = self.has_dynamic_object or self.visualize
        interaction_mesh_handle_list: list[object] = []
        retry_without_foot_sticking_exception: RuntimeError | None = None
        retry_without_foot_sticking_frame: int | None = None

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

                # Get human joint positions and create interaction mesh in object frame
                human_mapped_joints = human_joint_motions[i, self.mapped_joint_indices]

                if self.object_name == "ground":
                    human_mapped_joints_in_object = human_mapped_joints
                else:
                    human_mapped_joints_in_object = transform_points_world_to_local(
                        object_quat_demo, object_trans_demo, human_mapped_joints
                    )

                source_vertices, source_tetrahedra = create_interaction_mesh(
                    np.vstack([human_mapped_joints_in_object, object_points_local_demo])
                )
                tetrahedra.append(source_tetrahedra)

                object_quat = object_poses_augmented[i, 3:]
                object_trans = object_poses_augmented[i, :3]
                obj_pts_demo = transform_points_local_to_world(
                    object_quat_demo, object_trans_demo, object_points_local_demo
                )
                obj_pts = transform_points_local_to_world(
                    object_quat,
                    object_trans,
                    object_points_local,
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

                frame_entry_q = np.copy(q)
                try:
                    q, cost = self.iterate(
                        q_locked=q_locked_list[i],
                        q_n=q,
                        q_t_last=retargeted_motions[-1],
                        target_laplacian=target_laplacian,
                        adj_list=adj_list,
                        obj_pts_local=object_points_local,
                        foot_sticking=foot_sticking_sequences[i],
                        w_nominal_tracking=w_nominal_tracking,
                        q_a_nominal=(q_nominal_list[i, self.q_a_indices] if q_nominal_list is not None else None),
                        init_t=i == 0,
                        frame_idx=i,
                        orientation_target_matrices=orientation_target_matrices[i],
                    )
                except RuntimeError as initial_exception:
                    try:
                        prepared_retry = (
                            self._prepare_frame_zero_ground_retry(
                                frame_entry_q=frame_entry_q,
                                foot_sticking=foot_sticking_sequences[i],
                                exception=initial_exception,
                                state=frame_zero_ground_retry_state,
                            )
                            if i == 0
                            else None
                        )
                    except RuntimeError as rejection:
                        if not isinstance(
                            initial_exception,
                            SQPNonlinearFeasibilityError,
                        ):
                            raise
                        raise SQPNonlinearFeasibilityError(
                            f"{initial_exception}; frame-zero ground retry was rejected: {rejection}",
                            foot_related_failure=False,
                            frame_idx=initial_exception.frame_idx,
                            total_iterations=initial_exception.total_iterations,
                            closest_residuals=initial_exception.closest_residuals,
                            closest_constraint_mode=(initial_exception.closest_constraint_mode),
                            minimum_hard_constraint_violation=(initial_exception.minimum_hard_constraint_violation),
                        ) from rejection

                    solve_exception: RuntimeError | None = initial_exception
                    if prepared_retry is not None:
                        frame_zero_ground_retry_state = prepared_retry
                        if prepared_retry.corrected_q is None:
                            raise RuntimeError(
                                "Triggered frame-zero ground retry has no corrected qpos",
                            )
                        corrected_q = np.array(
                            prepared_retry.corrected_q,
                            dtype=np.float64,
                            copy=True,
                        )
                        q = corrected_q
                        q_locked_list[i] = corrected_q
                        retargeted_motions[-1] = corrected_q.copy()
                        print(
                            "WARNING: Retrying robot-only frame zero after a "
                            "pure horizontal-ground feasibility failure by "
                            f"lifting root Z {prepared_retry.lift_m:.9g} m. "
                            "No physical feasibility constraint is released.",
                            flush=True,
                        )
                        try:
                            q, cost = self.iterate(
                                q_locked=q_locked_list[i],
                                q_n=q,
                                q_t_last=retargeted_motions[-1],
                                target_laplacian=target_laplacian,
                                adj_list=adj_list,
                                obj_pts_local=object_points_local,
                                foot_sticking=foot_sticking_sequences[i],
                                w_nominal_tracking=w_nominal_tracking,
                                q_a_nominal=None,
                                init_t=True,
                                frame_idx=0,
                                orientation_target_matrices=(orientation_target_matrices[i]),
                            )
                        except RuntimeError as retry_exception:
                            solve_exception = retry_exception
                        else:
                            solve_exception = None
                            self.last_sqp_stop_reason = self._frame_zero_ground_retry_stop_reason(
                                self.last_sqp_stop_reason,
                                triggered=True,
                            )

                    if solve_exception is not None:
                        should_retry_without_foot_sticking = self._should_retry_without_foot_sticking(
                            exception=solve_exception,
                            foot_sticking=foot_sticking_sequences[i],
                            already_retrying=_foot_sticking_retry,
                        )
                        if not should_retry_without_foot_sticking:
                            if solve_exception is initial_exception:
                                raise
                            raise solve_exception from initial_exception
                        retry_without_foot_sticking_exception = solve_exception
                        retry_without_foot_sticking_frame = i
                        break

                if i == 0:
                    self.last_sqp_stop_reason = self._frame_zero_ground_retry_stop_reason(
                        self.last_sqp_stop_reason,
                        triggered=frame_zero_ground_retry_state.triggered,
                    )
                frame_costs.append(float(cost))
                sqp_iteration_counts.append(self.last_sqp_iteration_count)
                sqp_stop_reasons.append(self.last_sqp_stop_reason)
                ground_non_penetration_violations.append(
                    self.last_nonlinear_constraint_residuals.ground_non_penetration,
                )
                object_non_penetration_violations.append(
                    self.last_nonlinear_constraint_residuals.object_non_penetration,
                )
                foot_sticking_violations.append(
                    self.last_nonlinear_constraint_residuals.foot_sticking,
                )
                foot_lock_violations.append(
                    self.last_nonlinear_constraint_residuals.foot_lock,
                )
                self_collision_violations.append(
                    self.last_nonlinear_constraint_residuals.self_collision,
                )
                joint_limits_violations.append(
                    self.last_nonlinear_constraint_residuals.joint_limits,
                )
                constraint_mode_foot_sticking.append(
                    self.last_constraint_mode.foot_sticking,
                )
                constraint_mode_object_non_penetration_released.append(
                    self.last_constraint_mode.object_non_penetration_released,
                )
                constraint_mode_trust_region_released.append(
                    self.last_constraint_mode.trust_region_released,
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
                if self.visualize:
                    self._update_foot_sticking_status(i, foot_sticking_states[i])
                    if self.debug or self.show_interaction_mesh:
                        self.draw_q(q)

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

        if retry_without_foot_sticking_exception is not None:
            if retry_without_foot_sticking_frame is None:
                raise RuntimeError("Missing frame index for foot-sticking retry")
            self.foot_sticking_full_sequence_retry_frame = retry_without_foot_sticking_frame
            print(
                "WARNING: Retrying the complete sequence without foot-sticking "
                f"constraints after frame {retry_without_foot_sticking_frame} "
                f"remained infeasible ({retry_without_foot_sticking_exception}). "
                "All other constraints are preserved.",
                flush=True,
            )
            activate_foot_sticking = self.activate_foot_sticking
            self.activate_foot_sticking = False
            try:
                inherited_q_a_init = (
                    frame_zero_ground_retry_state.corrected_q if frame_zero_ground_retry_state.triggered else q_a_init
                )
                return self.retarget_motion(
                    human_joint_motions=human_joint_motions,
                    human_joint_quaternions_wxyz=(saved_human_joint_quaternions_wxyz),
                    human_orientation_joint_names=source_orientation_joint_names,
                    orientation_target_world_rotation_deltas_wxyz=(orientation_target_world_rotation_deltas_wxyz),
                    object_poses=object_poses,
                    object_poses_augmented=object_poses_augmented,
                    object_points_local_demo=object_points_local_demo,
                    object_points_local=object_points_local,
                    foot_sticking_sequences=foot_sticking_sequences,
                    q_a_init=inherited_q_a_init,
                    q_nominal_list=q_nominal_list,
                    original=original,
                    dest_res_path=dest_res_path,
                    fps=fps,
                    result_metadata=result_metadata,
                    _foot_sticking_retry=True,
                    _frame_zero_ground_retry_state=(frame_zero_ground_retry_state),
                )
            finally:
                self.activate_foot_sticking = activate_foot_sticking

        # Save results
        mapped_human_joint_names = list(self.laplacian_match_links.keys())
        tracked_count = len(self.orientation_human_joint_names)
        if self.orientation_diagnostics_enabled:
            target_quaternions_wxyz = self._matrices_to_wxyz(orientation_target_matrices).astype(np.float32)
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
        constraint_mode_foot_sticking_array = np.asarray(
            constraint_mode_foot_sticking,
            dtype=str,
        )
        fallback_available = (
            self.foot_sticking_fallback_tolerance is not None
            and self.foot_sticking_fallback_tolerance > self.foot_sticking_tolerance
        )
        self.foot_sticking_fallback_frames = set(
            np.flatnonzero(
                (constraint_mode_foot_sticking_array == "relaxed")
                | ((constraint_mode_foot_sticking_array == "released") & fallback_available),
            ).tolist()
        )
        self.foot_sticking_release_frames = set(
            np.flatnonzero(
                constraint_mode_foot_sticking_array == "released",
            ).tolist()
        )
        constraint_mode_object_non_penetration_released_array = np.asarray(
            constraint_mode_object_non_penetration_released,
            dtype=bool,
        )
        self.object_non_penetration_release_frames = set(
            np.flatnonzero(
                constraint_mode_object_non_penetration_released_array,
            ).tolist()
        )
        if self.object_model_path:
            (
                observed_object_asset_manifest_json,
                observed_object_asset_manifest_sha256,
            ) = build_object_asset_manifest(self.object_model_path)
            if (
                observed_object_asset_manifest_json != self.object_asset_manifest_json
                or observed_object_asset_manifest_sha256 != self.object_asset_manifest_sha256
            ):
                raise RuntimeError(
                    f"Object URDF dependency closure changed during retargeting: {self.object_model_path}"
                )
        save_payload = {
            "schema_version": np.asarray(
                RESULT_SCHEMA_VERSION,
                dtype=np.int32,
            ),
            # Preserve the exact accepted SQP states. Casting to float32 here
            # can move a contact-constrained trajectory outside the nonlinear
            # feasibility tolerance and makes augmentation warm starts differ
            # from the states whose residuals and costs are saved below.
            "qpos": np.asarray(retargeted_motions[1:], dtype=np.float64),
            "qpos_layout": np.asarray("mujoco_free_root_xyz_wxyz_then_actuated_then_optional_object_free_joint"),
            "human_joints": np.asarray(
                human_joint_motions,
                dtype=np.float32,
            ),
            "human_joint_names": np.asarray(self.demo_joints, dtype=str),
            "human_joint_parent_indices": self.human_joint_parent_indices,
            "mapped_human_joints": human_joint_motions[:, self.mapped_joint_indices],
            "mapped_human_joint_names": np.asarray(mapped_human_joint_names, dtype=str),
            "mapped_robot_joints": np.asarray(mapped_robot_joints_w_list, dtype=np.float32),
            "mapped_robot_link_names": np.asarray(list(self.laplacian_match_links.values()), dtype=str),
            "robot_link_positions": np.asarray(
                robot_link_positions_w_list,
                dtype=np.float32,
            ),
            "robot_link_quaternions_wxyz": np.asarray(
                robot_link_quaternions_wxyz_list,
                dtype=np.float32,
            ),
            "robot_link_names": np.asarray(self.robot_link_names, dtype=str),
            "robot_link_parent_indices": self.robot_link_parent_indices,
            "robot_actuated_joint_names": np.asarray(
                self.robot_actuated_joint_names,
                dtype=str,
            ),
            "source_data_format": np.asarray(self.task_constants.SOURCE_DATA_FORMAT),
            "robot_type": np.asarray(self.task_constants.ROBOT_TYPE),
            "task_type": np.asarray(getattr(self.task_constants, "TASK_TYPE", "")),
            "object_name": np.asarray(self.object_name),
            "object_urdf": np.asarray(self.object_model_path or ""),
            "object_urdf_sha256": np.asarray(
                self.object_urdf_sha256,
            ),
            "object_asset_manifest_json": np.asarray(
                self.object_asset_manifest_json,
            ),
            "object_asset_manifest_sha256": np.asarray(
                self.object_asset_manifest_sha256,
            ),
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
            "foot_sticking_fallback_tolerance": np.asarray(
                np.nan if self.foot_sticking_fallback_tolerance is None else self.foot_sticking_fallback_tolerance
            ),
            "foot_sticking_fallback_frames": np.asarray(
                sorted(self.foot_sticking_fallback_frames),
                dtype=np.int32,
            ),
            "release_foot_sticking_on_infeasible": np.asarray(self.release_foot_sticking_on_infeasible),
            "foot_sticking_release_frames": np.asarray(
                sorted(self.foot_sticking_release_frames),
                dtype=np.int32,
            ),
            "release_object_non_penetration_on_infeasible": np.asarray(
                self.release_object_non_penetration_on_infeasible
            ),
            "object_non_penetration_release_frames": np.asarray(
                sorted(self.object_non_penetration_release_frames),
                dtype=np.int32,
            ),
            "object_non_penetration_eligible_for_saved_trajectory": np.asarray(
                bool(self.activate_obj_non_penetration)
                and self.object_name != "ground"
                and bool(self.object_model_path)
            ),
            "foot_sticking_enabled_for_saved_trajectory": np.asarray(
                self.activate_foot_sticking and self.q_a_init_idx < 12,
            ),
            "foot_sticking_full_sequence_retry_frame": np.asarray(
                -1
                if self.foot_sticking_full_sequence_retry_frame is None
                else self.foot_sticking_full_sequence_retry_frame,
                dtype=np.int32,
            ),
            "frame_zero_ground_retry_policy": np.asarray(
                FRAME_ZERO_GROUND_RETRY_POLICY,
            ),
            "frame_zero_ground_retry_eligible": np.asarray(
                frame_zero_ground_retry_state.eligible,
            ),
            "frame_zero_ground_retry_triggered": np.asarray(
                frame_zero_ground_retry_state.triggered,
            ),
            "frame_zero_ground_retry_initial_min_distance_m": np.asarray(
                frame_zero_ground_retry_state.initial_min_distance_m,
                dtype=np.float64,
            ),
            "frame_zero_ground_retry_corrected_min_distance_m": np.asarray(
                frame_zero_ground_retry_state.corrected_min_distance_m,
                dtype=np.float64,
            ),
            "frame_zero_ground_retry_lift_m": np.asarray(
                frame_zero_ground_retry_state.lift_m,
                dtype=np.float64,
            ),
            "frame_zero_ground_retry_interior_margin_m": np.asarray(
                frame_zero_ground_retry_state.interior_margin_m,
                dtype=np.float64,
            ),
            "frame_zero_ground_retry_initial_sqp_iterations": np.asarray(
                frame_zero_ground_retry_state.initial_sqp_iterations,
                dtype=np.int32,
            ),
            "frame_costs": np.asarray(frame_costs, dtype=np.float64),
            "sqp_iteration_counts": np.asarray(sqp_iteration_counts, dtype=np.int32),
            "sqp_stop_reasons": np.asarray(sqp_stop_reasons, dtype=str),
            "ground_non_penetration_violation": np.asarray(
                ground_non_penetration_violations,
                dtype=np.float64,
            ),
            "object_non_penetration_violation": np.asarray(
                object_non_penetration_violations,
                dtype=np.float64,
            ),
            "foot_sticking_violation": np.asarray(
                foot_sticking_violations,
                dtype=np.float64,
            ),
            "foot_lock_violation": np.asarray(
                foot_lock_violations,
                dtype=np.float64,
            ),
            "self_collision_violation": np.asarray(
                self_collision_violations,
                dtype=np.float64,
            ),
            "joint_limits_violation": np.asarray(
                joint_limits_violations,
                dtype=np.float64,
            ),
            "constraint_mode_foot_sticking": constraint_mode_foot_sticking_array,
            "constraint_mode_object_non_penetration_released": (constraint_mode_object_non_penetration_released_array),
            "constraint_mode_trust_region_released": np.asarray(
                constraint_mode_trust_region_released,
                dtype=bool,
            ),
            "orientation_tracking_enabled": np.asarray(self.orientation_tracking_enabled),
            "orientation_diagnostics_enabled": np.asarray(self.orientation_diagnostics_enabled),
            "orientation_human_joint_names": np.asarray(
                self.orientation_human_joint_names,
                dtype=str,
            ),
            "orientation_robot_link_names": np.asarray(
                self.orientation_robot_link_names,
                dtype=str,
            ),
            "orientation_weights": self.orientation_weight_values.astype(np.float64),
            "orientation_alignment_mode": np.asarray(self.orientation_alignment_mode),
            "orientation_alignment_quaternions_wxyz": (
                self._matrices_to_wxyz(orientation_alignment_matrices).astype(np.float32)
            ),
            "orientation_reference_human_quaternions_wxyz": (
                self._matrices_to_wxyz(self.orientation_reference_human_matrices).astype(np.float32)
            ),
            "orientation_reference_robot_quaternions_wxyz": (
                self._matrices_to_wxyz(self.orientation_reference_robot_matrices).astype(np.float32)
            ),
            "orientation_reference_robot_qpos": self.orientation_reference_robot_qpos.astype(np.float32),
            "orientation_target_quaternions_wxyz": target_quaternions_wxyz,
            "orientation_robot_quaternions_wxyz": robot_quaternions_wxyz,
            "orientation_errors_rad": orientation_error_array,
            "orientation_frame_costs": orientation_frame_cost_array,
            "fps": float(fps),
            "cost": cost,
        }
        if saved_human_joint_quaternions_wxyz is not None:
            saved_human_orientation_names = np.asarray(
                source_orientation_joint_names,
                dtype=str,
            )
            saved_human_orientation_quaternions = np.asarray(
                saved_human_joint_quaternions_wxyz,
                dtype=np.float32,
            )
            save_payload.update(
                {
                    "human_orientation_joint_names": (saved_human_orientation_names),
                    "human_orientation_quaternions_wxyz": (saved_human_orientation_quaternions),
                    "human_orientation_sha256": np.asarray(
                        compute_human_orientation_sha256(
                            saved_human_orientation_names,
                            saved_human_orientation_quaternions,
                        )
                    ),
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
            save_payload.update(
                {
                    "interaction_source_vertices_w": np.asarray(interaction_source_vertices_w_list, dtype=np.float32),
                    "interaction_target_vertices_w": np.asarray(interaction_target_vertices_w_list, dtype=np.float32),
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
        write_result_artifact(dest_res_path, save_payload)
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
        requested_mode: ConstraintMode | None = None,
    ) -> SQPIterationResult:
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

        w_v = (self.laplacian_weights * np.ones(V)).astype(float)  # (V,)
        sqrt_w3 = np.sqrt(np.repeat(w_v, 3))

        # Decision variables
        dqa = cp.Variable(len(self.q_a_indices), name="dqa")
        lap_var = cp.Variable(3 * V, name="laplacian")

        # Base constraints and replaceable foot-sticking constraints are kept
        # separate so an infeasible frame can be retried without weakening
        # collision, joint-limit, foot-lock, or trust-region constraints.
        constraints = []
        foot_sticking_constraints = []
        foot_sticking_fallback_constraints = []
        object_non_penetration_constraints = []

        # Linear equality
        active_laplacian_jacobian = J_L
        constraints += [cp.Constant(active_laplacian_jacobian) @ dqa - lap_var == -lap0_vec]

        # Foot constraints (sticking + foot lock window Z pinning)
        apply_foot_sticking = (self.q_a_init_idx < 12) and self.activate_foot_sticking
        apply_foot_lock = (self.q_a_init_idx < 12) and self.foot_lock.enable
        if apply_foot_sticking or apply_foot_lock:
            J_WF_dict, p_WF_dict, _ = self._calc_manipulator_jacobians(q, links=self.foot_links, obj_frame=False)

            # Foot sticking: constrain XY to stay near previous frame position
            if apply_foot_sticking:
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
                        foot_sticking_constraints += [
                            Jxy @ dqa >= p_lb[:2],
                            Jxy @ dqa <= p_ub[:2],
                        ]
                        if (
                            self.foot_sticking_fallback_tolerance is not None
                            and self.foot_sticking_fallback_tolerance > self.foot_sticking_tolerance
                        ):
                            fallback_lb = delta - self.foot_sticking_fallback_tolerance
                            fallback_ub = delta + self.foot_sticking_fallback_tolerance
                            foot_sticking_fallback_constraints += [
                                Jxy @ dqa >= fallback_lb[:2],
                                Jxy @ dqa <= fallback_ub[:2],
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
            # The qpos update is subsequently normalized on the quaternion
            # manifold, so a candidate that sits exactly on the linearized
            # penetration boundary can end up slightly outside the configured
            # tolerance in true MuJoCo geometry. Keep a small, bounded interior
            # margin instead of weakening the nonlinear acceptance criterion.
            rhs = self._non_penetration_linearization_rhs(
                phi,
                self.penetration_tolerance,
            )
            constraint = Ja_n @ dqa >= rhs
            if self._collision_pair_involves_dynamic_object(*key):
                object_non_penetration_constraints.append(constraint)
            else:
                constraints.append(constraint)

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
            for link_idx, weight in enumerate(self.orientation_weight_values):
                obj_terms.append(
                    weight * cp.sum_squares(orientation_jacobians[link_idx] @ dqa - orientation_errors[link_idx])
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

        objective = cp.Minimize(cp.sum(obj_terms))

        def _linearized_cost_at_step(active_step: np.ndarray) -> float:
            """Evaluate this iteration's objective at one actual active-q step."""

            step = np.asarray(active_step, dtype=np.float64)
            if step.shape != (self.nq_a,) or not np.all(np.isfinite(step)):
                raise ValueError(
                    "The linearized objective step must be a finite vector with "
                    f"shape ({self.nq_a},), got {step.shape}",
                )
            dqa.value = step
            laplacian_value = active_laplacian_jacobian @ step + lap0_vec
            lap_var.value = np.asarray(
                laplacian_value,
                dtype=np.float64,
            ).reshape(-1)
            value = objective.expr.value
            if value is None or not np.isfinite(value):
                raise RuntimeError(
                    f"Linearized objective evaluation failed at frame {frame_idx}",
                )
            return float(value)

        solver_kwargs = {"verbose": verbose}
        if requested_mode is None:
            problem, constraint_mode = self._solve_with_foot_sticking_fallback(
                objective=objective,
                base_constraints=constraints,
                object_non_penetration_constraints=object_non_penetration_constraints,
                foot_sticking_constraints=foot_sticking_constraints,
                foot_sticking_fallback_constraints=foot_sticking_fallback_constraints,
                solver_kwargs=solver_kwargs,
                remove_soc_on_failure=init_t,
                release_on_failure=self.release_foot_sticking_on_infeasible,
                release_object_non_penetration_on_failure=(
                    self.release_object_non_penetration_on_infeasible
                    and self.activate_obj_non_penetration
                    and self.object_name not in {"", "ground"}
                    and bool(self.object_model_path)
                ),
            )
        else:
            problem, constraint_mode = self._solve_requested_constraint_mode(
                objective=objective,
                base_constraints=constraints,
                object_non_penetration_constraints=object_non_penetration_constraints,
                foot_sticking_constraints=foot_sticking_constraints,
                foot_sticking_fallback_constraints=foot_sticking_fallback_constraints,
                solver_kwargs=solver_kwargs,
                requested_mode=requested_mode,
                remove_soc_on_failure=init_t,
            )

        active_root_quaternion = bool(set(range(3, 7)).intersection(int(index) for index in self.q_a_indices))

        def _candidate_at_alpha(
            alpha: float,
            proposal_step: np.ndarray,
        ) -> SQPIterationResult:
            """Evaluate one true-geometry point on the incumbent-to-QP segment."""

            alpha_value = float(alpha)
            if not np.isfinite(alpha_value) or not 0.0 <= alpha_value <= 1.0:
                raise ValueError(f"Backtracking alpha must be within [0, 1], got {alpha}")
            full_step = np.asarray(proposal_step, dtype=np.float64)
            if full_step.shape != (self.nq_a,) or not np.all(np.isfinite(full_step)):
                raise ValueError(
                    f"The QP proposal must be a finite vector with shape ({self.nq_a},), got {full_step.shape}",
                )

            candidate_q = np.copy(q)
            candidate_q[self.q_a_indices] = q_a_n_last + alpha_value * full_step
            if alpha_value != 0.0 and active_root_quaternion:
                quaternion_norm = float(np.linalg.norm(candidate_q[3:7]))
                if not np.isfinite(quaternion_norm) or quaternion_norm <= np.finfo(np.float64).eps:
                    raise RuntimeError(
                        f"QP proposal produced an invalid root quaternion at frame {frame_idx}",
                    )
                candidate_q[3:7] /= quaternion_norm

            actual_step = candidate_q[self.q_a_indices] - q_a_n_last
            residuals = self._evaluate_nonlinear_constraint_residuals(
                candidate_q,
                q_t_last=q_t_last,
                foot_sticking=foot_sticking,
                frame_idx=frame_idx,
                mode=constraint_mode,
            )
            return SQPIterationResult(
                q=candidate_q,
                linearized_cost=_linearized_cost_at_step(actual_step),
                constraint_mode=constraint_mode,
                residuals=residuals,
            )

        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            incumbent = _candidate_at_alpha(
                0.0,
                np.zeros(self.nq_a, dtype=np.float64),
            )
            if incumbent.residuals.is_feasible(incumbent.constraint_mode):
                return incumbent
            raise RuntimeError(
                f"CVXPY solve failed at frame {frame_idx}: {problem.status}; "
                f"foot tolerance={self.foot_sticking_tolerance}, "
                f"fallback tolerance={self.foot_sticking_fallback_tolerance}, "
                f"release foot on infeasible={self.release_foot_sticking_on_infeasible}, "
                "release object non-penetration on infeasible="
                f"{self.release_object_non_penetration_on_infeasible}"
            )

        dqa_star = dqa.value
        if dqa_star is None:
            raise RuntimeError(
                f"CVXPY solve returned no candidate at frame {frame_idx}: {problem.status}",
            )
        proposal_step = np.asarray(
            dqa_star,
            dtype=np.float64,
        ).reshape(-1)
        return self._accept_or_backtrack_candidate(
            lambda alpha: _candidate_at_alpha(alpha, proposal_step),
        )

    @staticmethod
    def _non_penetration_linearization_rhs(
        signed_distance: float,
        penetration_tolerance: float,
    ) -> float:
        """Return an interior collision boundary for one linearized solve."""

        distance = float(signed_distance)
        tolerance = float(penetration_tolerance)
        if not np.isfinite(distance):
            raise ValueError("signed_distance must be finite")
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError(
                "penetration_tolerance must be finite and non-negative",
            )
        buffer = collision_interior_margin_m(tolerance)
        return -distance - tolerance + buffer

    def _record_constraint_fallbacks(
        self,
        *,
        frame_idx: int,
        foot_sticking_resolution: str | None,
        foot_sticking_fallback_available: bool,
        object_non_penetration_released: bool,
    ) -> None:
        """Record only the constraint mode used by the successful solve."""
        if foot_sticking_resolution == "relaxed" and frame_idx not in self.foot_sticking_fallback_frames:
            self.foot_sticking_fallback_frames.add(frame_idx)
            print(
                "WARNING: Retried infeasible foot-sticking constraints at "
                f"frame {frame_idx} with tolerance "
                f"{self.foot_sticking_fallback_tolerance:.6g} m "
                f"(normal {self.foot_sticking_tolerance:.6g} m).",
                flush=True,
            )
        elif foot_sticking_resolution == "released" and frame_idx not in self.foot_sticking_release_frames:
            if foot_sticking_fallback_available:
                self.foot_sticking_fallback_frames.add(frame_idx)
            self.foot_sticking_release_frames.add(frame_idx)
            relaxed_attempt = " and relaxed" if foot_sticking_fallback_available else ""
            print(
                "WARNING: Released foot-sticking constraints at "
                f"frame {frame_idx} after the normal{relaxed_attempt} problems "
                "remained infeasible; all other constraints are preserved.",
                flush=True,
            )
        if object_non_penetration_released and frame_idx not in self.object_non_penetration_release_frames:
            self.object_non_penetration_release_frames.add(frame_idx)
            print(
                "WARNING: Released robot-object non-penetration constraints at "
                f"frame {frame_idx} after the foot-sticking fallbacks remained "
                "infeasible; ground non-penetration and all other constraints "
                "are preserved.",
                flush=True,
            )

    @staticmethod
    def _solve_requested_constraint_mode(
        *,
        objective,
        base_constraints: list,
        object_non_penetration_constraints: list,
        foot_sticking_constraints: list,
        foot_sticking_fallback_constraints: list,
        solver_kwargs: dict,
        requested_mode: ConstraintMode,
        remove_soc_on_failure: bool,
    ) -> tuple[cp.Problem, ConstraintMode]:
        """Solve one linearization without changing its foot/object mode."""

        if requested_mode.foot_sticking == "normal":
            if not foot_sticking_constraints:
                raise ValueError("Normal foot mode requires active foot-sticking constraints")
            active_foot_constraints = foot_sticking_constraints
        elif requested_mode.foot_sticking == "relaxed":
            if not foot_sticking_fallback_constraints:
                raise ValueError("Relaxed foot mode requires fallback foot-sticking constraints")
            active_foot_constraints = foot_sticking_fallback_constraints
        elif requested_mode.foot_sticking == "released":
            if not foot_sticking_constraints:
                raise ValueError("Released foot mode requires an active source foot constraint")
            active_foot_constraints = []
        else:
            if foot_sticking_constraints:
                raise ValueError("Inactive foot mode cannot omit active foot-sticking constraints")
            active_foot_constraints = []

        active_base_constraints = list(base_constraints)
        trust_region_released = bool(requested_mode.trust_region_released)
        if trust_region_released:
            active_base_constraints = [
                constraint
                for constraint in active_base_constraints
                if not isinstance(constraint, cp.constraints.second_order.SOC)
            ]

        def _solve() -> cp.Problem:
            object_constraints = (
                [] if requested_mode.object_non_penetration_released else object_non_penetration_constraints
            )
            problem = cp.Problem(
                objective,
                [
                    *active_base_constraints,
                    *object_constraints,
                    *active_foot_constraints,
                ],
            )
            problem.solve(solver=cp.CLARABEL, **solver_kwargs)
            return problem

        problem = _solve()
        if (
            problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
            and remove_soc_on_failure
            and not trust_region_released
        ):
            active_base_constraints = [
                constraint
                for constraint in active_base_constraints
                if not isinstance(constraint, cp.constraints.second_order.SOC)
            ]
            trust_region_released = True
            problem = _solve()

        return problem, ConstraintMode(
            foot_sticking=requested_mode.foot_sticking,
            object_non_penetration_released=(requested_mode.object_non_penetration_released),
            trust_region_released=trust_region_released,
        )

    @staticmethod
    def _solve_with_foot_sticking_fallback(
        objective,
        base_constraints: list,
        object_non_penetration_constraints: list,
        foot_sticking_constraints: list,
        foot_sticking_fallback_constraints: list,
        solver_kwargs: dict,
        remove_soc_on_failure: bool,
        release_on_failure: bool,
        release_object_non_penetration_on_failure: bool,
    ) -> tuple[cp.Problem, ConstraintMode]:
        """Preserve the historical linear-infeasibility fallback helper."""

        active_base_constraints = list(base_constraints)
        trust_region_released = False
        optimal_statuses = (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)

        def _solve(
            active_foot_constraints: list,
            *,
            include_object_non_penetration: bool = True,
        ) -> cp.Problem:
            object_constraints = object_non_penetration_constraints if include_object_non_penetration else []
            problem = cp.Problem(
                objective,
                [
                    *active_base_constraints,
                    *object_constraints,
                    *active_foot_constraints,
                ],
            )
            problem.solve(solver=cp.CLARABEL, **solver_kwargs)
            return problem

        foot_modes: list[tuple[str, list]] = [
            (
                "normal" if foot_sticking_constraints else "inactive",
                foot_sticking_constraints,
            )
        ]
        if foot_sticking_constraints and foot_sticking_fallback_constraints:
            foot_modes.append(("relaxed", foot_sticking_fallback_constraints))
        if foot_sticking_constraints and release_on_failure:
            foot_modes.append(("released", []))

        object_release_modes = [False]
        if release_object_non_penetration_on_failure:
            object_release_modes.append(True)
        attempts = [
            (object_released, foot_mode, active_foot_constraints)
            for object_released in object_release_modes
            for foot_mode, active_foot_constraints in foot_modes
        ]

        def _attempt(
            object_released: bool,
            foot_mode: str,
            active_foot_constraints: list,
        ) -> tuple[cp.Problem, ConstraintMode, bool]:
            problem = _solve(
                active_foot_constraints,
                include_object_non_penetration=not object_released,
            )
            mode = ConstraintMode(
                foot_sticking=foot_mode,
                object_non_penetration_released=object_released,
                trust_region_released=trust_region_released,
            )
            accepted = problem.status in optimal_statuses
            return problem, mode, accepted

        first_object_released, first_foot_mode, first_foot_constraints = attempts[0]
        problem, mode, accepted = _attempt(
            first_object_released,
            first_foot_mode,
            first_foot_constraints,
        )
        if accepted:
            return problem, mode

        if remove_soc_on_failure:
            active_base_constraints = [
                constraint
                for constraint in active_base_constraints
                if not isinstance(constraint, cp.constraints.second_order.SOC)
            ]
            trust_region_released = True
            problem, mode, accepted = _attempt(
                first_object_released,
                first_foot_mode,
                first_foot_constraints,
            )
            if accepted:
                return problem, mode

        for object_released, foot_mode, active_foot_constraints in attempts[1:]:
            problem, mode, accepted = _attempt(
                object_released,
                foot_mode,
                active_foot_constraints,
            )
            if accepted:
                return problem, mode
        return problem, ConstraintMode(
            foot_sticking=first_foot_mode,
            trust_region_released=trust_region_released,
        )

    def _is_foot_locked_in_window(self, foot_link_key: str, frame_idx: int) -> bool:
        """Check whether a foot link is locked by configured frame windows."""
        side = self._name_side(foot_link_key)
        if side is None:
            return False

        return any(start <= frame_idx <= end for start, end in self._foot_lock_windows.get(side, ()))

    def _foot_sticking_by_side(
        self,
        foot_sticking: Mapping[str, bool],
    ) -> dict[str, bool]:
        """Normalize one frame's source-foot flags without relying on key order."""

        values: dict[str, bool] = {}
        for key in foot_sticking:
            side = self._name_side(key)
            if side is not None:
                values[side] = bool(foot_sticking[key])
        missing = [side for side in ("left", "right") if side not in values]
        if missing:
            raise ValueError("foot_sticking must include one left* and one right* key")
        return values

    def _has_active_foot_sticking_constraints(
        self,
        foot_sticking: Mapping[str, bool],
    ) -> bool:
        """Return whether this frame actually contributes a foot-sticking constraint."""

        if not self.activate_foot_sticking or self.q_a_init_idx >= 12:
            return False
        return any(self._foot_sticking_by_side(foot_sticking).values())

    def _frame_zero_ground_retry_eligible(
        self,
        *,
        q_nominal_list: np.ndarray | None,
        original: bool,
    ) -> bool:
        """Return whether this motion invocation belongs to the narrow policy."""

        return (
            getattr(self, "retry_frame_zero_ground_on_infeasible", False)
            and bool(original)
            and q_nominal_list is None
            and getattr(self.task_constants, "TASK_TYPE", None) == "robot_only"
            and self.object_name == "ground"
            and self.object_model_path is None
            and 2 in self.q_a_indices
        )

    @staticmethod
    def _is_pure_frame_zero_ground_failure(
        exception: RuntimeError,
    ) -> bool:
        """Recognize only a structured, unreleased, pure-ground failure."""

        if not isinstance(exception, SQPNonlinearFeasibilityError):
            return False
        residuals = exception.closest_residuals
        mode = exception.closest_constraint_mode
        return (
            exception.frame_idx == 0
            and exception.total_iterations is not None
            and exception.total_iterations > 0
            and residuals is not None
            and mode is not None
            and mode.foot_sticking == "inactive"
            and not mode.object_non_penetration_released
            and not mode.trust_region_released
            and residuals.ground_non_penetration > NONLINEAR_FEASIBILITY_ATOL
            and residuals.object_non_penetration <= NONLINEAR_FEASIBILITY_ATOL
            and residuals.foot_sticking <= NONLINEAR_FEASIBILITY_ATOL
            and residuals.foot_lock <= NONLINEAR_FEASIBILITY_ATOL
            and residuals.self_collision <= NONLINEAR_FEASIBILITY_ATOL
            and residuals.joint_limits <= NONLINEAR_FEASIBILITY_ATOL
        )

    @staticmethod
    def _frame_zero_ground_retry_stop_reason(
        stop_reason: str,
        *,
        triggered: bool,
    ) -> str:
        """Carry the retry provenance through a later full-sequence foot retry."""

        prefix = "frame_zero_ground_retry:"
        if not triggered or stop_reason.startswith(prefix):
            return stop_reason
        return f"{prefix}{stop_reason}"

    def _minimum_active_horizontal_ground_distance(
        self,
        q: np.ndarray,
    ) -> float:
        """Measure active robot-ground pairs and reject non-horizontal terrain."""

        candidate = np.asarray(q, dtype=np.float64)
        if candidate.shape != self.robot_data.qpos.shape or not np.isfinite(candidate).all():
            raise ValueError(
                "Frame-zero ground retry requires one finite full model qpos "
                f"with shape {self.robot_data.qpos.shape}; got {candidate.shape}",
            )
        self.robot_data.qpos[:] = candidate
        mujoco.mj_forward(self.robot_model, self.robot_data)
        threshold = max(
            float(self.collision_detection_threshold),
            float(self.penetration_tolerance) + NONLINEAR_FEASIBILITY_ATOL,
        )
        candidates = self._prefilter_pairs_with_mj_collision(threshold)
        fromto = np.zeros(6, dtype=np.float64)
        distances: list[float] = []
        for geom1, geom2 in candidates:
            geom1_name = self._geom_names[geom1]
            geom2_name = self._geom_names[geom2]
            if not self._environment_collision_pair_is_active(
                geom1_name,
                geom2_name,
            ):
                continue
            geom1_is_ground = "ground" in geom1_name.lower()
            geom2_is_ground = "ground" in geom2_name.lower()
            if geom1_is_ground == geom2_is_ground:
                continue
            ground_geom = geom1 if geom1_is_ground else geom2
            robot_geom = geom2 if geom1_is_ground else geom1
            if int(self.robot_model.geom_type[ground_geom]) != int(
                mujoco.mjtGeom.mjGEOM_PLANE,
            ):
                raise RuntimeError(
                    "The active ground geometry is not a MuJoCo plane",
                )
            ground_rotation = self.robot_data.geom_xmat[ground_geom].reshape(
                3,
                3,
            )
            ground_normal = ground_rotation[:, 2]
            if not np.allclose(
                ground_normal,
                np.asarray([0.0, 0.0, 1.0]),
                atol=1e-10,
                rtol=0.0,
            ):
                raise RuntimeError(
                    "The active ground plane is not horizontal with world +Z normal",
                )
            if int(self.robot_model.geom_bodyid[robot_geom]) == 0:
                raise RuntimeError(
                    "The active ground pair does not contain a robot body geometry",
                )
            distance = float(
                mujoco.mj_geomDistance(
                    self.robot_model,
                    self.robot_data,
                    geom1,
                    geom2,
                    threshold,
                    fromto,
                )
            )
            if not np.isfinite(distance):
                raise RuntimeError(
                    "MuJoCo returned a non-finite robot-ground distance",
                )
            distances.append(distance)
        if not distances:
            raise RuntimeError(
                "No active robot-ground pair was available for frame-zero correction",
            )
        return min(distances)

    def _prepare_frame_zero_ground_retry(
        self,
        *,
        frame_entry_q: np.ndarray,
        foot_sticking: Mapping[str, bool],
        exception: RuntimeError,
        state: FrameZeroGroundRetryState,
    ) -> FrameZeroGroundRetryState | None:
        """Create one validated root-Z correction, or return no retry."""

        if not state.eligible or state.attempted or not self._is_pure_frame_zero_ground_failure(exception):
            return None
        original_q = np.array(frame_entry_q, dtype=np.float64, copy=True)
        strict_mode = ConstraintMode("inactive")
        original_residuals = self._evaluate_nonlinear_constraint_residuals(
            original_q,
            q_t_last=original_q,
            foot_sticking=foot_sticking,
            frame_idx=0,
            mode=strict_mode,
        )
        if not (
            original_residuals.ground_non_penetration > NONLINEAR_FEASIBILITY_ATOL
            and original_residuals.object_non_penetration <= NONLINEAR_FEASIBILITY_ATOL
            and original_residuals.foot_sticking <= NONLINEAR_FEASIBILITY_ATOL
            and original_residuals.foot_lock <= NONLINEAR_FEASIBILITY_ATOL
            and original_residuals.self_collision <= NONLINEAR_FEASIBILITY_ATOL
            and original_residuals.joint_limits <= NONLINEAR_FEASIBILITY_ATOL
        ):
            raise RuntimeError(
                "The original frame-entry state was not a pure ground violation",
            )
        initial_distance = self._minimum_active_horizontal_ground_distance(
            original_q,
        )
        margin = collision_interior_margin_m(self.penetration_tolerance)
        lift = max(
            0.0,
            -initial_distance - self.penetration_tolerance + margin,
        )
        if not np.isfinite(lift) or lift <= 0.0:
            raise RuntimeError(
                "The strict horizontal-ground correction was not finite and positive",
            )
        corrected_q = original_q.copy()
        corrected_q[2] += lift
        corrected_distance = self._minimum_active_horizontal_ground_distance(
            corrected_q,
        )
        if not np.isclose(
            corrected_distance,
            initial_distance + lift,
            atol=1e-8,
            rtol=0.0,
        ):
            raise RuntimeError(
                "The measured ground distance did not follow the one-axis horizontal lift",
            )
        corrected_residuals = self._evaluate_nonlinear_constraint_residuals(
            corrected_q,
            q_t_last=corrected_q,
            foot_sticking=foot_sticking,
            frame_idx=0,
            mode=strict_mode,
        )
        if not corrected_residuals.is_feasible(strict_mode):
            raise RuntimeError(
                "The one-axis horizontal-ground correction did not pass true-geometry validation",
            )
        corrected_q.setflags(write=False)
        return FrameZeroGroundRetryState(
            eligible=True,
            attempted=True,
            triggered=True,
            initial_min_distance_m=initial_distance,
            corrected_min_distance_m=corrected_distance,
            lift_m=lift,
            interior_margin_m=margin,
            initial_sqp_iterations=int(exception.total_iterations),
            corrected_q=corrected_q,
        )

    def _should_retry_without_foot_sticking(
        self,
        *,
        exception: RuntimeError,
        foot_sticking: Mapping[str, bool],
        already_retrying: bool,
    ) -> bool:
        """Return whether a failed frame justifies a full no-foot retry."""

        message = str(exception)
        if isinstance(exception, SQPNonlinearFeasibilityError):
            solve_failed = True
            foot_related_failure = exception.foot_related_failure
        else:
            solve_failed = "CVXPY solve failed" in message
            foot_related_failure = self._has_active_foot_sticking_constraints(
                foot_sticking,
            )
        return (
            self.retry_without_foot_sticking_on_infeasible
            and not already_retrying
            and solve_failed
            and foot_related_failure
            and self._has_active_foot_sticking_constraints(foot_sticking)
        )

    def _current_foot_positions(self) -> dict[str, np.ndarray]:
        """Read foot body origins from the current MuJoCo forward state."""

        positions: dict[str, np.ndarray] = {}
        for key, link_name in self.foot_links.items():
            body_id = mujoco.mj_name2id(
                self.robot_model,
                mujoco.mjtObj.mjOBJ_BODY,
                link_name,
            )
            if body_id < 0:
                raise ValueError(f"Foot body {link_name!r} not found in MuJoCo model")
            positions[key] = np.array(
                self.robot_data.xpos[body_id],
                dtype=np.float64,
                copy=True,
            )
        return positions

    def _evaluate_nonlinear_constraint_residuals(
        self,
        q: np.ndarray,
        *,
        q_t_last: np.ndarray,
        foot_sticking: Mapping[str, bool],
        frame_idx: int,
        mode: ConstraintMode,
    ) -> NonlinearConstraintResiduals:
        """Evaluate hard constraints on true MuJoCo geometry at ``q``.

        Values are non-negative excesses beyond the configured tolerance. A
        released constraint remains measured for artifact diagnostics but is
        omitted by :meth:`NonlinearConstraintResiduals.is_feasible`.
        """

        candidate = np.asarray(q, dtype=np.float64)
        if candidate.shape != self.robot_data.qpos.shape or not np.isfinite(candidate).all():
            raise ValueError(
                f"SQP candidate must be a finite full qpos vector with shape {self.robot_data.qpos.shape}, "
                f"got {candidate.shape}",
            )

        self.robot_data.qpos[:] = candidate
        mujoco.mj_forward(self.robot_model, self.robot_data)
        candidate_foot_positions = self._current_foot_positions()

        collision_threshold = max(
            float(self.collision_detection_threshold),
            float(self.penetration_tolerance) + NONLINEAR_FEASIBILITY_ATOL,
        )
        candidates = self._prefilter_pairs_with_mj_collision(
            collision_threshold,
        )
        ground_violation = 0.0
        object_violation = 0.0
        fromto = np.zeros(6, dtype=np.float64)
        for geom1, geom2 in candidates:
            geom1_name = self._geom_names[geom1]
            geom2_name = self._geom_names[geom2]
            if not self._environment_collision_pair_is_active(
                geom1_name,
                geom2_name,
            ):
                continue
            distance = float(
                mujoco.mj_geomDistance(
                    self.robot_model,
                    self.robot_data,
                    geom1,
                    geom2,
                    collision_threshold,
                    fromto,
                )
            )
            violation = max(
                0.0,
                -distance - float(self.penetration_tolerance),
            )
            if "ground" in geom1_name.lower() or "ground" in geom2_name.lower():
                ground_violation = max(ground_violation, violation)
            else:
                object_violation = max(object_violation, violation)

        self_collision_violation = 0.0
        self_collision_active = self._self_collision_enabled and (
            self._self_collision_windows is None
            or any(start <= frame_idx <= end for start, end in self._self_collision_windows)
        )
        if self_collision_active:
            self_collision_threshold = max(
                collision_threshold,
                float(self._self_collision_tolerance) + NONLINEAR_FEASIBILITY_ATOL,
            )
            for geom1, geom2 in self._self_collision_geom_pairs:
                distance = float(
                    mujoco.mj_geomDistance(
                        self.robot_model,
                        self.robot_data,
                        geom1,
                        geom2,
                        self_collision_threshold,
                        fromto,
                    )
                )
                self_collision_violation = max(
                    self_collision_violation,
                    0.0,
                    float(self._self_collision_tolerance) - distance,
                )

        foot_lock_violation = 0.0
        if self.q_a_init_idx < 12 and self.foot_lock.enable:
            for key, position in candidate_foot_positions.items():
                if self._is_foot_locked_in_window(key, frame_idx):
                    foot_lock_violation = max(
                        foot_lock_violation,
                        0.0,
                        abs(float(position[2]) - float(self.foot_lock.z_floor)) - float(self.foot_lock.tolerance),
                    )

        foot_sticking_violation = 0.0
        if self.q_a_init_idx < 12 and self.activate_foot_sticking and mode.foot_sticking != "inactive":
            side_values = self._foot_sticking_by_side(foot_sticking)
            previous = np.asarray(q_t_last, dtype=np.float64)
            self.robot_data.qpos[:] = previous
            mujoco.mj_forward(self.robot_model, self.robot_data)
            previous_foot_positions = self._current_foot_positions()
            if mode.foot_sticking == "normal":
                tolerance = self.foot_sticking_tolerance
            elif mode.foot_sticking == "relaxed":
                tolerance = self.foot_sticking_fallback_tolerance
            else:
                fallback_is_effective = (
                    self.foot_sticking_fallback_tolerance is not None
                    and self.foot_sticking_fallback_tolerance > self.foot_sticking_tolerance
                )
                tolerance = (
                    self.foot_sticking_fallback_tolerance if fallback_is_effective else self.foot_sticking_tolerance
                )
            if tolerance is None:
                raise RuntimeError("Relaxed foot-sticking mode requires a fallback tolerance")
            for key, position in candidate_foot_positions.items():
                side = self._name_side(key)
                if side is not None and side_values[side]:
                    xy_error = np.abs(
                        position[:2] - previous_foot_positions[key][:2],
                    )
                    foot_sticking_violation = max(
                        foot_sticking_violation,
                        float(np.max(np.maximum(0.0, xy_error - tolerance))),
                    )

        joint_limit_violation = 0.0
        if self.activate_joint_limits:
            active_q = candidate[self.q_a_indices]
            joint_limit_violation = float(
                max(
                    0.0,
                    float(np.max(self.q_a_lb - active_q)),
                    float(np.max(active_q - self.q_a_ub)),
                )
            )

        return NonlinearConstraintResiduals(
            ground_non_penetration=ground_violation,
            object_non_penetration=object_violation,
            foot_sticking=foot_sticking_violation,
            foot_lock=foot_lock_violation,
            self_collision=self_collision_violation,
            joint_limits=joint_limit_violation,
        )

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
                "[SelfCollision] _geom_names not initialized. Please run _prefilter_pairs_with_mj_collision first."
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

    @staticmethod
    def _accept_or_backtrack_candidate(
        evaluate_alpha: Callable[[float], SQPIterationResult],
        *,
        bisection_iterations: int = NONLINEAR_BACKTRACK_BISECTION_ITERATIONS,
    ) -> SQPIterationResult:
        """Keep a feasible incumbent or return the largest known feasible step."""

        if bisection_iterations <= 0:
            raise ValueError("bisection_iterations must be positive")

        incumbent = evaluate_alpha(0.0)
        proposal = evaluate_alpha(1.0)
        if proposal.residuals.is_feasible(proposal.constraint_mode):
            return proposal
        if not incumbent.residuals.is_feasible(incumbent.constraint_mode):
            return proposal

        feasible_alpha = 0.0
        infeasible_alpha = 1.0
        for _ in range(bisection_iterations):
            alpha = 0.5 * (feasible_alpha + infeasible_alpha)
            candidate = evaluate_alpha(alpha)
            if candidate.residuals.is_feasible(candidate.constraint_mode):
                feasible_alpha = alpha
            else:
                infeasible_alpha = alpha
        if feasible_alpha == 0.0:
            return incumbent
        return evaluate_alpha(feasible_alpha)

    @staticmethod
    def _select_feasible_candidate(
        candidates: list[SQPIterationResult],
        *,
        atol: float = NONLINEAR_FEASIBILITY_ATOL,
    ) -> SQPIterationResult | None:
        """Return the strictest feasible candidate without comparing local costs."""

        feasible = [
            candidate
            for candidate in candidates
            if candidate.residuals.is_feasible(
                candidate.constraint_mode,
                atol=atol,
            )
        ]
        if not feasible:
            return None
        foot_priority = {
            "inactive": 0,
            "normal": 0,
            "relaxed": 1,
            "released": 2,
        }

        def _mode_priority(candidate: SQPIterationResult) -> tuple[bool, int, bool]:
            mode = candidate.constraint_mode
            return (
                mode.object_non_penetration_released,
                foot_priority[mode.foot_sticking],
                mode.trust_region_released,
            )

        best_priority = min(_mode_priority(candidate) for candidate in feasible)
        return next(candidate for candidate in reversed(feasible) if _mode_priority(candidate) == best_priority)

    def _constraint_mode_schedule(
        self,
        foot_sticking: Mapping[str, bool],
    ) -> tuple[ConstraintMode, ...]:
        """Return full-SQP modes from strictest to most permissive."""

        if self._has_active_foot_sticking_constraints(foot_sticking):
            foot_modes = ["normal"]
            if (
                self.foot_sticking_fallback_tolerance is not None
                and self.foot_sticking_fallback_tolerance > self.foot_sticking_tolerance
            ):
                foot_modes.append("relaxed")
            if self.release_foot_sticking_on_infeasible:
                foot_modes.append("released")
        else:
            foot_modes = ["inactive"]

        object_release_modes = [False]
        if (
            self.release_object_non_penetration_on_infeasible
            and self.activate_obj_non_penetration
            and self.object_name not in {"", "ground"}
            and bool(self.object_model_path)
        ):
            object_release_modes.append(True)

        return tuple(
            ConstraintMode(
                foot_sticking=foot_mode,
                object_non_penetration_released=object_released,
            )
            for object_released in object_release_modes
            for foot_mode in foot_modes
        )

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
        n_iter: int | None = None,
        frame_idx: int = 0,
        orientation_target_matrices: np.ndarray | None = None,
    ):
        """Run a complete SQP solve for each constraint mode in priority order.

        ``n_iter`` is retained as an optional per-call safety-cap override for
        compatibility and applies independently to every attempted mode. Each
        less-strict mode restarts from the frame-entry state.
        """
        max_iterations = self.sqp_max_iterations if n_iter is None else int(n_iter)
        if max_iterations <= 0:
            raise ValueError("n_iter must be positive when provided")

        min_iterations = min(self.sqp_min_iterations, max_iterations)
        all_candidates: list[SQPIterationResult] = []
        accepted: SQPIterationResult | None = None
        accepted_stop_reason = "all_modes_exhausted"
        total_iterations = 0
        frame_entry_q = np.copy(q_n)
        step_tolerance = max(
            NONLINEAR_FEASIBILITY_ATOL,
            min(1e-3, float(self.step_size) * 1e-2),
        )
        mode_schedule = self._constraint_mode_schedule(foot_sticking)
        for requested_mode in mode_schedule:
            mode_q = np.copy(frame_entry_q)
            latest_feasible: SQPIterationResult | None = None
            previous_consecutive_feasible_q: np.ndarray | None = None
            stable_feasible_iterations = 0
            mode_stop_reason = "max_iterations"

            for mode_iteration_idx in range(max_iterations):
                total_iterations += 1
                try:
                    result = self.solve_single_iteration(
                        q_locked=q_locked,
                        q_a_n_last=mode_q[self.q_a_indices],
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
                        requested_mode=requested_mode,
                    )
                except RuntimeError as exc:
                    if "CVXPY solve failed" not in str(exc) and "CVXPY solve returned no candidate" not in str(exc):
                        raise
                    mode_stop_reason = (
                        "linear_solve_failed_after_feasible" if latest_feasible is not None else "linear_solve_failed"
                    )
                    break

                if not isinstance(result, SQPIterationResult):
                    raise TypeError(
                        "solve_single_iteration must return SQPIterationResult",
                    )
                if (
                    result.constraint_mode.foot_sticking != requested_mode.foot_sticking
                    or result.constraint_mode.object_non_penetration_released
                    != requested_mode.object_non_penetration_released
                ):
                    raise RuntimeError(
                        "solve_single_iteration changed the requested foot/object constraint mode",
                    )
                if not np.isfinite(result.linearized_cost):
                    raise RuntimeError(
                        "SQP returned a non-finite local cost at "
                        f"frame {frame_idx}, mode {requested_mode}, "
                        f"iteration {mode_iteration_idx}: "
                        f"{result.linearized_cost}"
                    )

                mode_q = result.q
                all_candidates.append(result)
                current_is_feasible = result.residuals.is_feasible(
                    result.constraint_mode,
                )
                if current_is_feasible:
                    feasible_step_norm = (
                        np.inf
                        if previous_consecutive_feasible_q is None
                        else float(
                            np.linalg.norm(
                                result.q[self.q_a_indices] - previous_consecutive_feasible_q[self.q_a_indices],
                            )
                        )
                    )
                    latest_feasible = result
                    if feasible_step_norm <= step_tolerance:
                        stable_feasible_iterations += 1
                    else:
                        stable_feasible_iterations = 0
                    previous_consecutive_feasible_q = np.copy(result.q)
                else:
                    stable_feasible_iterations = 0
                    previous_consecutive_feasible_q = None

                completed_in_mode = mode_iteration_idx + 1
                if completed_in_mode >= min_iterations and stable_feasible_iterations >= self.sqp_convergence_patience:
                    mode_stop_reason = "feasible_step_stable"
                    break

            if latest_feasible is not None:
                accepted = latest_feasible
                accepted_stop_reason = mode_stop_reason
                break

        if accepted is None:
            closest_candidate = min(
                all_candidates,
                key=lambda candidate: candidate.residuals.hard_max_violation(
                    candidate.constraint_mode,
                ),
                default=None,
            )
            maximum_violation = (
                np.inf
                if closest_candidate is None
                else closest_candidate.residuals.hard_max_violation(
                    closest_candidate.constraint_mode,
                )
            )
            residual_details = "unavailable" if closest_candidate is None else repr(closest_candidate.residuals)
            active_foot_failure_observed = any(
                candidate.constraint_mode.foot_sticking in {"normal", "relaxed"}
                and candidate.residuals.foot_sticking > NONLINEAR_FEASIBILITY_ATOL
                for candidate in all_candidates
            )
            footless_mode_attempted = any(mode.foot_sticking in {"inactive", "released"} for mode in mode_schedule)
            foot_related_failure = (
                self._has_active_foot_sticking_constraints(foot_sticking)
                and active_foot_failure_observed
                and not footless_mode_attempted
            )
            raise SQPNonlinearFeasibilityError(
                "SQP nonlinear feasibility failed at "
                f"frame {frame_idx} after {total_iterations} total iterations "
                f"across {len(mode_schedule)} constraint modes; "
                "no true-geometry feasible candidate was found; "
                f"minimum hard-constraint violation={maximum_violation:.9g}; "
                f"closest residuals={residual_details}",
                foot_related_failure=foot_related_failure,
                frame_idx=frame_idx,
                total_iterations=total_iterations,
                closest_residuals=(None if closest_candidate is None else closest_candidate.residuals),
                closest_constraint_mode=(None if closest_candidate is None else closest_candidate.constraint_mode),
                minimum_hard_constraint_violation=maximum_violation,
            )

        self.last_sqp_iteration_count = total_iterations
        self.last_sqp_stop_reason = accepted_stop_reason
        self.last_constraint_mode = accepted.constraint_mode
        self.last_nonlinear_constraint_residuals = accepted.residuals
        self._record_constraint_fallbacks(
            frame_idx=frame_idx,
            foot_sticking_resolution=accepted.constraint_mode.foot_sticking,
            foot_sticking_fallback_available=(
                self.foot_sticking_fallback_tolerance is not None
                and self.foot_sticking_fallback_tolerance > self.foot_sticking_tolerance
            ),
            object_non_penetration_released=(accepted.constraint_mode.object_non_penetration_released),
        )
        return np.copy(accepted.q), float(accepted.linearized_cost)

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

    def _prefilter_pairs_with_mj_collision(self, threshold: float):
        m, d = self.robot_model, self.robot_data
        ngeom = m.ngeom

        self._geom_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(ngeom)]

        if not hasattr(self, "_saved_margins"):
            self._saved_margins = np.empty_like(m.geom_margin)
        self._saved_margins[:] = m.geom_margin

        m.geom_margin[:] = threshold

        # Run collision. This runs broad→narrow and fills d.contact.
        mujoco.mj_collision(m, d)

        # Collect unique candidate pairs that involve at least one masked geom
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

    def _update_jacobians_and_phis_from_q(self, q: np.ndarray):
        self.robot_data.qpos[:] = q

        mujoco.mj_forward(self.robot_model, self.robot_data)  # kinematics & AABBs valid

        m, d = self.robot_model, self.robot_data
        threshold = float(self.collision_detection_threshold)

        # 1) Fast prefilter via mj_collision with temporary margins
        candidates = self._prefilter_pairs_with_mj_collision(threshold)

        Js, phis = {}, {}
        fromto = np.zeros(6, dtype=float)

        # 2) Precise distance only on candidates (early-exit at threshold)
        contype, conaff = m.geom_contype, m.geom_conaffinity

        def masks_ok(g1, g2):
            if contype[g1] == 0 and conaff[g1] == 0:
                return False
            if contype[g2] == 0 and conaff[g2] == 0:
                return False
            return self._environment_collision_pair_is_active(
                self._geom_names[g1],
                self._geom_names[g2],
            )

        for g1, g2 in candidates:
            # Optional: keep your own filters here (e.g., skip object-ground, only keep interaction with object/ground)
            if not masks_ok(g1, g2):
                continue

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

    def _collision_pair_involves_dynamic_object(
        self,
        geom1_id: int,
        geom2_id: int,
    ) -> bool:
        """Return whether an active environment pair is object rather than ground.

        The historical name is retained for compatibility, but static climbing
        terrain is also an object constraint and must participate in the same
        explicit release mode.
        """

        if self.object_name in {"", "ground"}:
            return False
        geom1_name = self._geom_names[geom1_id]
        geom2_name = self._geom_names[geom2_id]
        if "ground" in geom1_name.lower() or "ground" in geom2_name.lower():
            return False
        return self._environment_collision_pair_is_active(
            geom1_name,
            geom2_name,
        )

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
            jt = self.robot_model.jnt_type[j]
            if jt in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                qa = self.robot_model.jnt_qposadr[j]
                da = self.robot_model.jnt_dofadr[j]
                T[da, qa] = 1.0
            elif jt == mujoco.mjtJoint.mjJNT_BALL:
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
