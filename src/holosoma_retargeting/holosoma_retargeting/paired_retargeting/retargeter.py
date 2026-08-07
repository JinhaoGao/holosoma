# ruff: noqa: CPY001
"""Joint refinement of two independently retargeted robot trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cvxpy as cp
import numpy as np

from holosoma_retargeting.paired_retargeting.alignment import resolve_source_alignment
from holosoma_retargeting.paired_retargeting.config import PairedRefinementConfig
from holosoma_retargeting.paired_retargeting.interaction import (
    CrossActorInteractionMesh,
    create_cross_actor_interaction_mesh,
    laplacian_coordinates,
)
from holosoma_retargeting.paired_retargeting.result import (
    PairedActorTrajectory,
    PairedRefinementResult,
)
from holosoma_retargeting.paired_retargeting.scene import (
    CollisionLinearization,
    PairedRobotScene,
)


def _wrapped_angle(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


@dataclass(frozen=True)
class _FrameReference:
    frame_index: int
    initial_qpos: np.ndarray
    nominal_qpos: np.ndarray
    previous_refined: np.ndarray | None
    previous_nominal: np.ndarray | None
    mesh: CrossActorInteractionMesh
    source_laplacian: np.ndarray
    root_position_target: np.ndarray
    root_yaw_target: float


@dataclass(frozen=True)
class _FrameLinearization:
    laplacian_current: np.ndarray
    laplacian_jacobian: np.ndarray
    root_position: np.ndarray
    root_position_jacobian: np.ndarray
    root_yaw: float
    root_yaw_jacobian: np.ndarray
    collision_constraints: tuple[CollisionLinearization, ...]
    minimum_distance: float


class PairedTrajectoryRefiner:
    """Refine two nominal trajectories in one namespaced MuJoCo model."""

    def __init__(
        self,
        actor_a_name: str,
        actor_a: PairedActorTrajectory,
        actor_b_name: str,
        actor_b: PairedActorTrajectory,
        config: PairedRefinementConfig | None = None,
        *,
        package_root: Path | None = None,
    ) -> None:
        if actor_a.frame_count != actor_b.frame_count:
            raise ValueError("Paired trajectories must contain the same number of frames")
        if not np.isclose(actor_a.fps, actor_b.fps, rtol=0.0, atol=1e-9):
            raise ValueError("Paired trajectories must use the same frame rate")
        self.actor_a_name = actor_a_name
        self.actor_b_name = actor_b_name
        self.actor_a = actor_a
        self.actor_b = actor_b
        self.config = config or PairedRefinementConfig()
        self.scene = PairedRobotScene(
            actor_a_name,
            actor_a,
            actor_b_name,
            actor_b,
            package_root=package_root,
        )
        self.source_alignment = resolve_source_alignment(actor_a, actor_b, self.config.source_alignment)
        self.actor_a_source_anchor_index = self._mapped_source_anchor_index(self.scene.actor_a, self.actor_a)
        self.actor_b_source_anchor_index = self._mapped_source_anchor_index(self.scene.actor_b, self.actor_b)
        self._validate_contacts()

    @staticmethod
    def _mapped_source_anchor_index(actor_index, trajectory: PairedActorTrajectory) -> int:
        root_body_matches = np.flatnonzero(actor_index.mapped_body_ids == actor_index.root_body_id)
        if root_body_matches.shape == (1,):
            return int(root_body_matches[0])
        if root_body_matches.size > 1:
            raise ValueError(f"Actor {actor_index.name!r} mapped links contain its root body more than once")

        torso_names = {"root", "hips", "pelvis", "spine"}
        torso_matches = [
            index
            for index, joint_name in enumerate(trajectory.mapped_human_joint_names)
            if joint_name.casefold() in torso_names
        ]
        if len(torso_matches) != 1:
            raise ValueError(
                f"Actor {actor_index.name!r} must contain exactly one mapped torso anchor named "
                "Root, Hips, Pelvis, or Spine when its root body is not mapped; "
                f"got {trajectory.mapped_human_joint_names}"
            )
        return torso_matches[0]

    def _validate_contacts(self) -> None:
        for contact in self.config.contacts:
            if contact.actor_a_link not in self.scene.actor_a.body_ids_by_name:
                raise ValueError(f"Contact link {contact.actor_a_link!r} does not exist on actor {self.actor_a_name!r}")
            if contact.actor_b_link not in self.scene.actor_b.body_ids_by_name:
                raise ValueError(f"Contact link {contact.actor_b_link!r} does not exist on actor {self.actor_b_name!r}")
        for actor_a_body, actor_b_body in self.config.collision.body_pairs:
            if actor_a_body not in self.scene.actor_a.body_ids_by_name:
                raise ValueError(f"Collision body {actor_a_body!r} does not exist on actor {self.actor_a_name!r}")
            if actor_b_body not in self.scene.actor_b.body_ids_by_name:
                raise ValueError(f"Collision body {actor_b_body!r} does not exist on actor {self.actor_b_name!r}")

    def refine(self) -> PairedRefinementResult:
        """Run sequential SQP refinement and return two owned qpos arrays."""
        frame_count = self.actor_a.frame_count
        actor_a_nominal = self.actor_a.qpos.copy()
        actor_b_nominal = self.actor_b.qpos.copy()
        actor_a_nominal[:, :3] += self.source_alignment.actor_a_translation
        actor_b_nominal[:, :3] += self.source_alignment.actor_b_translation
        actor_a_source_points = self.source_alignment.translate_actor_a_points(self.actor_a.mapped_human_joints)
        actor_b_source_points = self.source_alignment.translate_actor_b_points(self.actor_b.mapped_human_joints)
        nominal = np.asarray(
            [
                self.scene.combine_qpos(actor_a_qpos, actor_b_qpos)
                for actor_a_qpos, actor_b_qpos in zip(actor_a_nominal, actor_b_nominal)
            ],
            dtype=np.float64,
        )
        refined = nominal.copy()
        frame_costs = np.zeros(frame_count, dtype=np.float64)
        iteration_counts = np.zeros(frame_count, dtype=np.int32)
        interaction_errors = np.zeros(frame_count, dtype=np.float64)
        root_position_errors = np.zeros(frame_count, dtype=np.float64)
        root_yaw_errors = np.zeros(frame_count, dtype=np.float64)
        minimum_distances = np.full(frame_count, np.nan, dtype=np.float64)
        contact_errors = np.full((frame_count, len(self.config.contacts)), np.nan, dtype=np.float64)
        source_vertex_frames: list[np.ndarray] = []
        target_vertex_frames: list[np.ndarray] = []
        tetrahedra_frames: list[np.ndarray] = []
        edge_frames: list[np.ndarray] = []

        for frame_index in range(frame_count):
            source_vertices = np.vstack(
                (
                    actor_a_source_points[frame_index],
                    actor_b_source_points[frame_index],
                )
            )
            mesh = create_cross_actor_interaction_mesh(
                actor_a_source_points[frame_index],
                actor_b_source_points[frame_index],
            )
            source_laplacian = laplacian_coordinates(mesh, source_vertices)

            self.scene.forward(nominal[frame_index])
            root_position_target = (
                actor_b_source_points[frame_index, self.actor_b_source_anchor_index]
                - actor_a_source_points[frame_index, self.actor_a_source_anchor_index]
            )
            root_yaw_target, _ = self.scene.root_relative_yaw_and_jacobian()
            previous_refined = refined[frame_index - 1] if frame_index else None
            previous_nominal = nominal[frame_index - 1] if frame_index else None
            (
                refined[frame_index],
                frame_costs[frame_index],
                iteration_counts[frame_index],
                minimum_distances[frame_index],
            ) = self._refine_frame(
                _FrameReference(
                    frame_index=frame_index,
                    initial_qpos=nominal[frame_index],
                    nominal_qpos=nominal[frame_index],
                    previous_refined=previous_refined,
                    previous_nominal=previous_nominal,
                    mesh=mesh,
                    source_laplacian=source_laplacian,
                    root_position_target=root_position_target,
                    root_yaw_target=root_yaw_target,
                )
            )

            self.scene.forward(refined[frame_index])
            if self.config.collision.is_active(frame_index):
                _, minimum_distances[frame_index] = self.scene.collision_linearizations(self.config.collision)
            target_vertices, _ = self.scene.mapped_points_and_jacobian()
            target_laplacian = laplacian_coordinates(mesh, target_vertices)
            interaction_errors[frame_index] = np.sqrt(np.mean(np.square(target_laplacian - source_laplacian)))
            root_position, _ = self.scene.root_relative_position_and_jacobian()
            root_yaw, _ = self.scene.root_relative_yaw_and_jacobian()
            root_position_errors[frame_index] = np.linalg.norm(root_position - root_position_target)
            root_yaw_errors[frame_index] = abs(_wrapped_angle(root_yaw - root_yaw_target))
            for contact_index, contact in enumerate(self.config.contacts):
                if contact.is_active(frame_index):
                    relative_position, _ = self.scene.relative_link_position_and_jacobian(
                        contact.actor_a_link,
                        contact.actor_b_link,
                    )
                    contact_errors[frame_index, contact_index] = np.linalg.norm(
                        relative_position - np.asarray(contact.target_offset, dtype=np.float64)
                    )
            source_vertex_frames.append(source_vertices)
            target_vertex_frames.append(target_vertices)
            tetrahedra_frames.append(mesh.tetrahedra)
            edge_frames.append(mesh.edges)

        actor_a_qpos = np.empty_like(self.actor_a.qpos, dtype=np.float64)
        actor_b_qpos = np.empty_like(self.actor_b.qpos, dtype=np.float64)
        for frame_index, qpos in enumerate(refined):
            actor_a_qpos[frame_index], actor_b_qpos[frame_index] = self.scene.split_qpos(qpos)
        return PairedRefinementResult(
            actor_a_name=self.actor_a_name,
            actor_b_name=self.actor_b_name,
            actor_a=self.actor_a,
            actor_b=self.actor_b,
            actor_a_qpos=actor_a_qpos,
            actor_b_qpos=actor_b_qpos,
            frame_costs=frame_costs,
            sqp_iteration_counts=iteration_counts,
            interaction_errors=interaction_errors,
            root_position_errors=root_position_errors,
            root_yaw_errors=root_yaw_errors,
            minimum_inter_actor_distances=minimum_distances,
            contact_errors=contact_errors,
            interaction_source_vertices=np.asarray(source_vertex_frames, dtype=np.float64),
            interaction_target_vertices=np.asarray(target_vertex_frames, dtype=np.float64),
            interaction_tetrahedra=tuple(tetrahedra_frames),
            interaction_edges=tuple(edge_frames),
            actor_a_source_translation=self.source_alignment.actor_a_translation.copy(),
            actor_b_source_translation=self.source_alignment.actor_b_translation.copy(),
            source_scene_scale=self.source_alignment.scene_scale,
            source_relative_xy_m=self.source_alignment.source_relative_xy_m.copy(),
            source_fbx=self.source_alignment.source_fbx,
            source_alignment_mode=self.source_alignment.mode,
            actor_a_human_joints=(
                None
                if self.actor_a.human_joints is None
                else self.source_alignment.translate_actor_a_points(self.actor_a.human_joints)
            ),
            actor_b_human_joints=(
                None
                if self.actor_b.human_joints is None
                else self.source_alignment.translate_actor_b_points(self.actor_b.human_joints)
            ),
            config=self.config,
        )

    def _has_active_coupling(self, frame_index: int) -> bool:
        has_active_contact = any(
            contact.is_active(frame_index) and (contact.weight > 0.0 or contact.tolerance is not None)
            for contact in self.config.contacts
        )
        return (
            self.config.interaction_weight > 0.0
            or self.config.root_position_weight > 0.0
            or self.config.root_yaw_weight > 0.0
            or has_active_contact
            or self.config.collision.is_active(frame_index)
        )

    def _linearize_frame(self, qpos: np.ndarray, reference: _FrameReference) -> _FrameLinearization:
        self.scene.forward(qpos)
        points, point_jacobian = self.scene.mapped_points_and_jacobian()
        laplacian_kron = np.kron(reference.mesh.laplacian_matrix, np.eye(3))
        root_position, root_position_jacobian = self.scene.root_relative_position_and_jacobian()
        root_yaw, root_yaw_jacobian = self.scene.root_relative_yaw_and_jacobian()
        collision_constraints: list[CollisionLinearization] = []
        minimum_distance = float("nan")
        if self.config.collision.is_active(reference.frame_index):
            collision_constraints, minimum_distance = self.scene.collision_linearizations(self.config.collision)
        return _FrameLinearization(
            laplacian_current=(reference.mesh.laplacian_matrix @ points).reshape(-1),
            laplacian_jacobian=laplacian_kron @ point_jacobian,
            root_position=root_position,
            root_position_jacobian=root_position_jacobian,
            root_yaw=root_yaw,
            root_yaw_jacobian=root_yaw_jacobian,
            collision_constraints=tuple(collision_constraints),
            minimum_distance=minimum_distance,
        )

    def _objective_terms(
        self,
        qpos: np.ndarray,
        reference: _FrameReference,
        linearization: _FrameLinearization,
        delta_qpos: cp.Variable,
    ) -> list[cp.Expression]:
        terms: list[cp.Expression] = [1e-10 * cp.sum_squares(delta_qpos)]
        if self.config.nominal_weight > 0.0:
            terms.append(self.config.nominal_weight * cp.sum_squares(qpos + delta_qpos - reference.nominal_qpos))
        if self.config.smoothness_weight > 0.0 and reference.previous_refined is not None:
            assert reference.previous_nominal is not None
            nominal_delta = reference.nominal_qpos - reference.previous_nominal
            smoothness_error = qpos - reference.previous_refined - nominal_delta
            terms.append(self.config.smoothness_weight * cp.sum_squares(smoothness_error + delta_qpos))
        if self.config.interaction_weight > 0.0:
            interaction_error = (
                linearization.laplacian_current
                + linearization.laplacian_jacobian @ delta_qpos
                - reference.source_laplacian.reshape(-1)
            )
            terms.append(self.config.interaction_weight * cp.sum_squares(interaction_error))
        if self.config.root_position_weight > 0.0:
            root_error = (
                linearization.root_position
                + linearization.root_position_jacobian @ delta_qpos
                - reference.root_position_target
            )
            terms.append(self.config.root_position_weight * cp.sum_squares(root_error))
        if self.config.root_yaw_weight > 0.0:
            yaw_error = _wrapped_angle(linearization.root_yaw - reference.root_yaw_target)
            terms.append(
                self.config.root_yaw_weight * cp.square(yaw_error + linearization.root_yaw_jacobian @ delta_qpos)
            )
        return terms

    def _contact_terms(
        self,
        frame_index: int,
        delta_qpos: cp.Variable,
    ) -> tuple[list[cp.Expression], list[cp.Constraint]]:
        objectives: list[cp.Expression] = []
        constraints: list[cp.Constraint] = []
        for contact in self.config.contacts:
            if not contact.is_active(frame_index):
                continue
            relative_position, relative_jacobian = self.scene.relative_link_position_and_jacobian(
                contact.actor_a_link,
                contact.actor_b_link,
            )
            contact_error = (
                relative_position + relative_jacobian @ delta_qpos - np.asarray(contact.target_offset, dtype=np.float64)
            )
            if contact.weight > 0.0:
                objectives.append(contact.weight * cp.sum_squares(contact_error))
            if contact.tolerance is not None:
                constraints.extend((contact_error >= -contact.tolerance, contact_error <= contact.tolerance))
        return objectives, constraints

    def _step_constraints(
        self,
        qpos: np.ndarray,
        linearization: _FrameLinearization,
        delta_qpos: cp.Variable,
    ) -> list[cp.Constraint]:
        constraints = [
            item.distance + item.jacobian @ delta_qpos >= self.config.collision.minimum_distance
            for item in linearization.collision_constraints
        ]
        lower, upper = self.scene.joint_limit_bounds(qpos)
        finite_lower = np.flatnonzero(np.isfinite(lower))
        finite_upper = np.flatnonzero(np.isfinite(upper))
        if finite_lower.size:
            constraints.append(delta_qpos[finite_lower] >= lower[finite_lower])
        if finite_upper.size:
            constraints.append(delta_qpos[finite_upper] <= upper[finite_upper])
        step_limits = self.scene.step_limits(
            self.config.root_translation_step_limit,
            self.config.root_quaternion_step_limit,
            self.config.joint_step_limit,
        )
        constraints.extend((delta_qpos >= -step_limits, delta_qpos <= step_limits))
        return constraints

    def _solve_iteration(
        self,
        qpos: np.ndarray,
        reference: _FrameReference,
        iteration_index: int,
    ) -> tuple[np.ndarray, float, float]:
        linearization = self._linearize_frame(qpos, reference)
        delta_qpos = cp.Variable(
            self.scene.model.nq,
            name=f"paired_dq_{reference.frame_index}_{iteration_index}",
        )
        objective_terms = self._objective_terms(qpos, reference, linearization, delta_qpos)
        contact_objectives, contact_constraints = self._contact_terms(reference.frame_index, delta_qpos)
        objective_terms.extend(contact_objectives)
        constraints = self._step_constraints(qpos, linearization, delta_qpos)
        constraints.extend(contact_constraints)
        problem = cp.Problem(cp.Minimize(cp.sum(objective_terms)), constraints)
        problem.solve(solver=cp.CLARABEL, verbose=False)
        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or delta_qpos.value is None:
            raise RuntimeError(
                f"Paired refinement failed at frame {reference.frame_index}, "
                f"iteration {iteration_index}: {problem.status}"
            )
        step = np.asarray(delta_qpos.value, dtype=np.float64).reshape(-1)
        if step.shape != (self.scene.model.nq,) or not np.all(np.isfinite(step)):
            raise RuntimeError(f"Paired refinement returned an invalid step at frame {reference.frame_index}")
        return step, float(problem.value), linearization.minimum_distance

    def _refine_frame(self, reference: _FrameReference) -> tuple[np.ndarray, float, int, float]:
        qpos = reference.initial_qpos.copy()
        if not self._has_active_coupling(reference.frame_index):
            return qpos, 0.0, 0, float("nan")
        cost = 0.0
        minimum_distance = float("nan")
        for iteration_index in range(self.config.max_sqp_iterations):
            step, cost, minimum_distance = self._solve_iteration(qpos, reference, iteration_index)
            qpos = self._normalize_and_align_quaternions(qpos + step, reference.nominal_qpos)
            if np.max(np.abs(step)) < self.config.convergence_tolerance:
                return qpos, cost, iteration_index + 1, minimum_distance
        return qpos, cost, self.config.max_sqp_iterations, minimum_distance

    def _normalize_and_align_quaternions(self, qpos: np.ndarray, nominal_qpos: np.ndarray) -> np.ndarray:
        normalized = self.scene.normalize_quaternions(qpos)
        for actor in (self.scene.actor_a, self.scene.actor_b):
            quaternion_slice = slice(actor.qpos_slice.start + 3, actor.qpos_slice.start + 7)
            if np.dot(normalized[quaternion_slice], nominal_qpos[quaternion_slice]) < 0.0:
                normalized[quaternion_slice] *= -1.0
        return normalized
