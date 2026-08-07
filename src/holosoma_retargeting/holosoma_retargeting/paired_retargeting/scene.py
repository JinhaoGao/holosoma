# ruff: noqa: CPY001
"""Namespaced two-actor MuJoCo scene used by paired refinement."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from holosoma_retargeting.config_types.robot import RobotConfig
from holosoma_retargeting.paired_retargeting.config import InterActorCollisionConfig
from holosoma_retargeting.paired_retargeting.result import PairedActorTrajectory


@dataclass(frozen=True)
class PairedActorIndex:
    """All model addresses owned by one actor in a combined model."""

    name: str
    prefix: str
    qpos_slice: slice
    qvel_slice: slice
    root_joint_id: int
    root_body_id: int
    mapped_body_ids: np.ndarray
    mapped_link_names: tuple[str, ...]
    body_ids_by_name: dict[str, int]
    body_names_by_id: dict[int, str]
    actuated_joint_ids: np.ndarray
    actuated_qpos_indices: np.ndarray
    collision_geom_ids: np.ndarray


@dataclass(frozen=True)
class CollisionLinearization:
    """Signed distance and its derivative with respect to combined qpos."""

    geom_a: int
    geom_b: int
    distance: float
    jacobian: np.ndarray


class PairedRobotScene:
    """Compile and index two robot models without rewriting their MJCF files."""

    def __init__(
        self,
        actor_a_name: str,
        actor_a: PairedActorTrajectory,
        actor_b_name: str,
        actor_b: PairedActorTrajectory,
        *,
        package_root: Path | None = None,
    ) -> None:
        if actor_a_name == actor_b_name:
            raise ValueError("Paired actor names must be unique")
        self.package_root = package_root or Path(__file__).resolve().parents[1]
        spec = mujoco.MjSpec()
        spec.copy_during_attach = True
        self._attach_actor(spec, actor_a_name, actor_a)
        self._attach_actor(spec, actor_b_name, actor_b)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.actor_a = self._build_actor_index(actor_a_name, actor_a)
        self.actor_b = self._build_actor_index(actor_b_name, actor_b)
        self._collision_pair_cache: dict[tuple[tuple[str, str], ...], tuple[tuple[int, int], ...]] = {}
        self._validate_disjoint_actor_slices()

    def _attach_actor(
        self,
        world: mujoco.MjSpec,
        name: str,
        trajectory: PairedActorTrajectory,
    ) -> None:
        robot_config = RobotConfig(robot_type=trajectory.robot_type)
        model_path = (self.package_root / robot_config.ROBOT_URDF_FILE).with_suffix(".xml")
        if not model_path.is_file():
            raise FileNotFoundError(f"MuJoCo model does not exist: {model_path}")
        child = mujoco.MjSpec.from_file(str(model_path))
        for geom in list(child.geoms):
            if geom.parent == child.worldbody:
                child.delete(geom)
        for light in list(child.lights):
            if light.parent == child.worldbody:
                child.delete(light)
        mount = world.worldbody.add_site(name=f"{name}__mount")
        world.attach(child, site=mount, prefix=f"{name}__")

    def _body_ids_by_name(self, prefix: str) -> dict[str, int]:
        body_ids_by_name: dict[str, int] = {}
        for body_id in range(1, self.model.nbody):
            namespaced_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            if namespaced_name.startswith(prefix):
                body_ids_by_name[namespaced_name.removeprefix(prefix)] = body_id
        return body_ids_by_name

    def _actuated_joint_index(
        self,
        name: str,
        prefix: str,
        trajectory: PairedActorTrajectory,
        qpos_start: int,
        qpos_stop: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        joint_addresses: list[int] = []
        actuated_joint_ids: list[int] = []
        model_actuated_names: list[tuple[int, str]] = []
        for joint_name in trajectory.robot_actuated_joint_names:
            joint_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                prefix + joint_name,
            )
            if joint_id < 0:
                raise ValueError(f"Actor {name!r} model is missing actuated joint {joint_name!r}")
            if int(self.model.jnt_type[joint_id]) not in {
                int(mujoco.mjtJoint.mjJNT_HINGE),
                int(mujoco.mjtJoint.mjJNT_SLIDE),
            }:
                raise ValueError(f"Actor {name!r} actuated joint {joint_name!r} is not scalar")
            qpos_address = int(self.model.jnt_qposadr[joint_id])
            actuated_joint_ids.append(joint_id)
            joint_addresses.append(qpos_address)
            model_actuated_names.append((qpos_address, joint_name))
        sorted_names = tuple(joint_name for _, joint_name in sorted(model_actuated_names))
        if sorted_names != trajectory.robot_actuated_joint_names:
            raise ValueError(f"Actor {name!r} saved joint order does not match its MuJoCo model")
        expected_addresses = np.arange(qpos_start + 7, qpos_stop, dtype=np.int32)
        actuated_qpos_indices = np.asarray(joint_addresses, dtype=np.int32)
        if not np.array_equal(actuated_qpos_indices, expected_addresses):
            raise ValueError(f"Actor {name!r} qpos coordinates are not contiguous")
        return np.asarray(actuated_joint_ids, dtype=np.int32), actuated_qpos_indices

    def _collision_geom_ids(self, body_ids_by_name: dict[str, int]) -> np.ndarray:
        return np.asarray(
            [
                geom_id
                for geom_id in range(self.model.ngeom)
                if int(self.model.geom_bodyid[geom_id]) in body_ids_by_name.values()
                and (self.model.geom_contype[geom_id] != 0 or self.model.geom_conaffinity[geom_id] != 0)
            ],
            dtype=np.int32,
        )

    def _build_actor_index(self, name: str, trajectory: PairedActorTrajectory) -> PairedActorIndex:
        prefix = f"{name}__"
        root_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}root")
        if root_joint_id < 0:
            raise ValueError(f"Robot {trajectory.robot_type!r} has no free joint named 'root'")
        if int(self.model.jnt_type[root_joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
            raise ValueError(f"Namespaced actor root joint {prefix + 'root'!r} is not free")
        qpos_start = int(self.model.jnt_qposadr[root_joint_id])
        qvel_start = int(self.model.jnt_dofadr[root_joint_id])
        qpos_stop = qpos_start + trajectory.qpos.shape[1]
        qvel_stop = qvel_start + 6 + len(trajectory.robot_actuated_joint_names)
        root_body_id = int(self.model.jnt_bodyid[root_joint_id])
        body_ids_by_name = self._body_ids_by_name(prefix)
        missing_links = sorted(set(trajectory.mapped_robot_link_names).difference(body_ids_by_name))
        if missing_links:
            raise ValueError(f"Actor {name!r} model is missing mapped links: {missing_links}")
        mapped_body_ids = np.asarray(
            [body_ids_by_name[link] for link in trajectory.mapped_robot_link_names],
            dtype=np.int32,
        )
        actuated_joint_ids, actuated_qpos_indices = self._actuated_joint_index(
            name,
            prefix,
            trajectory,
            qpos_start,
            qpos_stop,
        )
        return PairedActorIndex(
            name=name,
            prefix=prefix,
            qpos_slice=slice(qpos_start, qpos_stop),
            qvel_slice=slice(qvel_start, qvel_stop),
            root_joint_id=root_joint_id,
            root_body_id=root_body_id,
            mapped_body_ids=mapped_body_ids,
            mapped_link_names=trajectory.mapped_robot_link_names,
            body_ids_by_name=body_ids_by_name,
            body_names_by_id={body_id: body_name for body_name, body_id in body_ids_by_name.items()},
            actuated_joint_ids=actuated_joint_ids,
            actuated_qpos_indices=actuated_qpos_indices,
            collision_geom_ids=self._collision_geom_ids(body_ids_by_name),
        )

    def _validate_disjoint_actor_slices(self) -> None:
        slices = (self.actor_a.qpos_slice, self.actor_b.qpos_slice)
        occupied = np.concatenate([np.arange(value.start, value.stop) for value in slices])
        if len(np.unique(occupied)) != len(occupied) or len(occupied) != self.model.nq:
            raise ValueError("Attached actor qpos slices must be disjoint and cover the combined model")

    def combine_qpos(self, actor_a_qpos: np.ndarray, actor_b_qpos: np.ndarray) -> np.ndarray:
        """Pack two source-model qpos vectors into the combined model."""
        qpos = self.model.qpos0.copy()
        values = (
            (self.actor_a, np.asarray(actor_a_qpos, dtype=np.float64)),
            (self.actor_b, np.asarray(actor_b_qpos, dtype=np.float64)),
        )
        for actor, actor_qpos in values:
            expected_shape = (actor.qpos_slice.stop - actor.qpos_slice.start,)
            if actor_qpos.shape != expected_shape:
                raise ValueError(f"Actor {actor.name!r} qpos must have shape {expected_shape}, got {actor_qpos.shape}")
            qpos[actor.qpos_slice] = actor_qpos
        return qpos

    def split_qpos(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Unpack a combined qpos vector into owned actor vectors."""
        combined = np.asarray(qpos, dtype=np.float64)
        if combined.shape != (self.model.nq,):
            raise ValueError(f"Combined qpos must have shape {(self.model.nq,)}, got {combined.shape}")
        return combined[self.actor_a.qpos_slice].copy(), combined[self.actor_b.qpos_slice].copy()

    def normalize_quaternions(self, qpos: np.ndarray) -> np.ndarray:
        """Normalize every free- or ball-joint quaternion in a combined qpos."""
        normalized = np.asarray(qpos, dtype=np.float64).copy()
        for joint_id in range(self.model.njnt):
            joint_type = int(self.model.jnt_type[joint_id])
            qpos_address = int(self.model.jnt_qposadr[joint_id])
            if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
                quaternion_slice = slice(qpos_address + 3, qpos_address + 7)
            elif joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
                quaternion_slice = slice(qpos_address, qpos_address + 4)
            else:
                continue
            norm = np.linalg.norm(normalized[quaternion_slice])
            if norm <= 1e-12:
                raise ValueError("Combined qpos contains a zero-norm quaternion")
            normalized[quaternion_slice] /= norm
        return normalized

    def forward(self, qpos: np.ndarray) -> None:
        """Update scene kinematics for a combined qpos vector."""
        combined = np.asarray(qpos, dtype=np.float64)
        if combined.shape != (self.model.nq,):
            raise ValueError(f"Combined qpos must have shape {(self.model.nq,)}, got {combined.shape}")
        self.data.qpos[:] = combined
        mujoco.mj_forward(self.model, self.data)

    def qdot_to_qvel_matrix(self) -> np.ndarray:
        """Map direct qpos perturbations to MuJoCo tangent velocities."""
        transform = np.zeros((self.model.nv, self.model.nq), dtype=np.float64)
        for joint_id in range(self.model.njnt):
            joint_type = int(self.model.jnt_type[joint_id])
            qpos_address = int(self.model.jnt_qposadr[joint_id])
            dof_address = int(self.model.jnt_dofadr[joint_id])
            if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
                transform[dof_address : dof_address + 3, qpos_address : qpos_address + 3] = np.eye(3)
                quaternion = self.data.qpos[qpos_address + 3 : qpos_address + 7]
                transform[dof_address + 3 : dof_address + 6, qpos_address + 3 : qpos_address + 7] = (
                    2.0 * self._quaternion_velocity_matrix(quaternion)
                )
            elif joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
                quaternion = self.data.qpos[qpos_address : qpos_address + 4]
                transform[dof_address : dof_address + 3, qpos_address : qpos_address + 4] = (
                    2.0 * self._quaternion_velocity_matrix(quaternion)
                )
            else:
                transform[dof_address, qpos_address] = 1.0
        return transform

    @staticmethod
    def _quaternion_velocity_matrix(quaternion: np.ndarray) -> np.ndarray:
        qw, qx, qy, qz = quaternion
        return np.asarray(
            [
                [-qx, qw, qz, -qy],
                [-qy, -qz, qw, qx],
                [-qz, qy, -qx, qw],
            ],
            dtype=np.float64,
        )

    def body_point_jacobian(self, body_id: int, point_world: np.ndarray | None = None) -> np.ndarray:
        """Return a translational body-point Jacobian with qpos columns."""
        point = self.data.xpos[body_id] if point_world is None else np.asarray(point_world, dtype=np.float64)
        jacobian_position = np.zeros((3, self.model.nv), dtype=np.float64)
        jacobian_rotation = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jac(
            self.model,
            self.data,
            jacobian_position,
            jacobian_rotation,
            point,
            body_id,
        )
        return jacobian_position @ self.qdot_to_qvel_matrix()

    def body_rotation_jacobian(self, body_id: int) -> np.ndarray:
        """Return a rotational body Jacobian with qpos columns."""
        jacobian_position = np.zeros((3, self.model.nv), dtype=np.float64)
        jacobian_rotation = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jac(
            self.model,
            self.data,
            jacobian_position,
            jacobian_rotation,
            self.data.xpos[body_id],
            body_id,
        )
        return jacobian_rotation @ self.qdot_to_qvel_matrix()

    def mapped_points_and_jacobian(self) -> tuple[np.ndarray, np.ndarray]:
        """Stack both actors' mapped link positions and point Jacobians."""
        body_ids = np.concatenate((self.actor_a.mapped_body_ids, self.actor_b.mapped_body_ids))
        points = np.asarray([self.data.xpos[body_id] for body_id in body_ids], dtype=np.float64)
        jacobian = np.vstack([self.body_point_jacobian(int(body_id)) for body_id in body_ids])
        return points, jacobian

    def root_relative_position_and_jacobian(self) -> tuple[np.ndarray, np.ndarray]:
        """Return actor-B minus actor-A root position and Jacobian."""
        position = self.data.xpos[self.actor_b.root_body_id] - self.data.xpos[self.actor_a.root_body_id]
        jacobian = self.body_point_jacobian(self.actor_b.root_body_id) - self.body_point_jacobian(
            self.actor_a.root_body_id
        )
        return position.copy(), jacobian

    def relative_link_position_and_jacobian(
        self,
        actor_a_link: str,
        actor_b_link: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return actor-B minus actor-A link position and Jacobian."""
        try:
            body_a_id = self.actor_a.body_ids_by_name[actor_a_link]
        except KeyError as exc:
            raise ValueError(f"Actor {self.actor_a.name!r} has no body {actor_a_link!r}") from exc
        try:
            body_b_id = self.actor_b.body_ids_by_name[actor_b_link]
        except KeyError as exc:
            raise ValueError(f"Actor {self.actor_b.name!r} has no body {actor_b_link!r}") from exc
        position = self.data.xpos[body_b_id] - self.data.xpos[body_a_id]
        jacobian = self.body_point_jacobian(body_b_id) - self.body_point_jacobian(body_a_id)
        return position.copy(), jacobian

    def root_relative_yaw_and_jacobian(self) -> tuple[float, np.ndarray]:
        """Return actor-B minus actor-A root yaw and its local linearization."""
        yaw_a = self._body_yaw(self.actor_a.root_body_id)
        yaw_b = self._body_yaw(self.actor_b.root_body_id)
        relative_yaw = float(np.arctan2(np.sin(yaw_b - yaw_a), np.cos(yaw_b - yaw_a)))
        jacobian = (
            self.body_rotation_jacobian(self.actor_b.root_body_id)[2]
            - self.body_rotation_jacobian(self.actor_a.root_body_id)[2]
        )
        return relative_yaw, jacobian

    def _body_yaw(self, body_id: int) -> float:
        rotation = self.data.xmat[body_id].reshape(3, 3)
        return float(np.arctan2(rotation[1, 0], rotation[0, 0]))

    def joint_limit_bounds(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return lower and upper bounds on a direct qpos increment."""
        lower = np.full(self.model.nq, -np.inf, dtype=np.float64)
        upper = np.full(self.model.nq, np.inf, dtype=np.float64)
        for actor in (self.actor_a, self.actor_b):
            for joint_id, qpos_address in zip(actor.actuated_joint_ids, actor.actuated_qpos_indices):
                if self.model.jnt_limited[joint_id]:
                    lower[qpos_address] = self.model.jnt_range[joint_id, 0] - qpos[qpos_address]
                    upper[qpos_address] = self.model.jnt_range[joint_id, 1] - qpos[qpos_address]
        return lower, upper

    def step_limits(self, root_translation: float, root_quaternion: float, joint: float) -> np.ndarray:
        """Build per-coordinate trust-region limits for both actors."""
        limits = np.full(self.model.nq, joint, dtype=np.float64)
        for actor in (self.actor_a, self.actor_b):
            limits[actor.qpos_slice.start : actor.qpos_slice.start + 3] = root_translation
            limits[actor.qpos_slice.start + 3 : actor.qpos_slice.start + 7] = root_quaternion
        return limits

    def collision_linearizations(
        self,
        config: InterActorCollisionConfig,
    ) -> tuple[list[CollisionLinearization], float]:
        """Linearize active cross-actor signed-distance constraints."""
        if not config.enabled:
            return [], float("nan")
        candidates = self._active_collision_candidates(config)
        linearizations: list[CollisionLinearization] = []
        minimum_distance = config.activation_distance
        closest_points = np.zeros(6, dtype=np.float64)
        for geom_a, geom_b in candidates:
            closest_points[:] = 0.0
            distance = float(
                mujoco.mj_geomDistance(
                    self.model,
                    self.data,
                    geom_a,
                    geom_b,
                    config.activation_distance,
                    closest_points,
                )
            )
            minimum_distance = min(minimum_distance, distance)
            if distance > config.activation_distance:
                continue
            point_a = closest_points[:3].copy()
            point_b = closest_points[3:].copy()
            normal = self._collision_normal(distance, point_a, point_b, geom_a, geom_b)
            body_a = int(self.model.geom_bodyid[geom_a])
            body_b = int(self.model.geom_bodyid[geom_b])
            relative_jacobian = self.body_point_jacobian(body_a, point_a) - self.body_point_jacobian(
                body_b,
                point_b,
            )
            linearizations.append(
                CollisionLinearization(
                    geom_a=geom_a,
                    geom_b=geom_b,
                    distance=distance,
                    jacobian=normal @ relative_jacobian,
                )
            )
        return linearizations, minimum_distance

    def _collision_candidate_pairs(self, config: InterActorCollisionConfig) -> tuple[tuple[int, int], ...]:
        cache_key = tuple(config.body_pairs)
        if cache_key in self._collision_pair_cache:
            return self._collision_pair_cache[cache_key]
        allowed_body_pairs = set(config.body_pairs)
        candidates: list[tuple[int, int]] = []
        for geom_a in self.actor_a.collision_geom_ids:
            body_a_id = int(self.model.geom_bodyid[geom_a])
            body_a_name = self.actor_a.body_names_by_id[body_a_id]
            for geom_b in self.actor_b.collision_geom_ids:
                body_b_id = int(self.model.geom_bodyid[geom_b])
                body_b_name = self.actor_b.body_names_by_id[body_b_id]
                if allowed_body_pairs and (body_a_name, body_b_name) not in allowed_body_pairs:
                    continue
                collision_mask_active = (
                    int(self.model.geom_contype[geom_a]) & int(self.model.geom_conaffinity[geom_b])
                ) or (int(self.model.geom_contype[geom_b]) & int(self.model.geom_conaffinity[geom_a]))
                if not collision_mask_active:
                    continue
                candidates.append((int(geom_a), int(geom_b)))
        result = tuple(candidates)
        self._collision_pair_cache[cache_key] = result
        return result

    def _active_collision_candidates(self, config: InterActorCollisionConfig) -> tuple[tuple[int, int], ...]:
        active: list[tuple[int, int]] = []
        for geom_a, geom_b in self._collision_candidate_pairs(config):
            center_distance = np.linalg.norm(self.data.geom_xpos[geom_a] - self.data.geom_xpos[geom_b])
            broad_distance = center_distance - self.model.geom_rbound[geom_a] - self.model.geom_rbound[geom_b]
            if broad_distance <= config.activation_distance:
                active.append((geom_a, geom_b))
        return tuple(active)

    def _collision_normal(
        self,
        distance: float,
        point_a: np.ndarray,
        point_b: np.ndarray,
        geom_a: int,
        geom_b: int,
    ) -> np.ndarray:
        delta = point_a - point_b
        norm = np.linalg.norm(delta)
        if norm <= 1e-12:
            delta = self.data.geom_xpos[geom_a] - self.data.geom_xpos[geom_b]
            norm = np.linalg.norm(delta)
        if norm <= 1e-12:
            delta = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
            norm = 1.0
        direction = -1.0 if distance < 0.0 else 1.0
        return direction * delta / norm
