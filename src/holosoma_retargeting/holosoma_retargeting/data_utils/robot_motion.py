# ruff: noqa: CPY001
"""Robot CSV and FK adapter for the shared Interaction Mesh pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from holosoma_retargeting.config_types.data_type import MotionDataConfig
from holosoma_retargeting.config_types.robot import RobotConfig, _default_robot_defaults
from holosoma_retargeting.config_types.robot_profiles import RobotProfile
from holosoma_retargeting.data_utils.motion_data import SourceMotion
from holosoma_retargeting.data_utils.npz_to_csv import actuated_joint_names_from_urdf
from scipy.spatial.transform import Rotation

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def quaternion_wxyz(value, *, label: str) -> np.ndarray:
    quat = np.asarray(value, dtype=np.float64)
    if quat.shape != (4,) or not np.isfinite(quat).all() or np.linalg.norm(quat) < 1e-8:
        raise ValueError(f"{label} must be a finite nonzero wxyz quaternion")
    return quat / np.linalg.norm(quat)


@dataclass(frozen=True)
class RobotMotionProfile:
    """A robot's independent semantic skeleton, reference pose, and sole layout."""

    robot: RobotConfig
    position_links: dict[str, str]
    parents: tuple[int, ...]
    root_joint: str
    toe_names: tuple[str, str]
    orientation_links: dict[str, str]
    reference_base: tuple[float, ...]
    reference_joints: dict[str, float]
    orientation_weights: dict[str, float]
    torso_link: str | None = None
    torso_anchor_names: tuple[str, ...] = ()

    @classmethod
    def from_profile(cls, profile: RobotProfile) -> RobotMotionProfile:
        if profile.robot_dof <= 0 or not np.isfinite(profile.robot_height) or profile.robot_height <= 0:
            raise ValueError("Robot DOF and height must be positive and finite")
        table = profile.robot_motion
        required = {
            "position_links",
            "parents",
            "root_joint",
            "toe_names",
            "orientation_links",
            "reference_base_quaternion_wxyz",
            "reference_joint_positions",
            "orientation_weights",
            "robot_config",
        }
        if table is None or not required.issubset(table) or set(table) - required - {"position_scaling"}:
            raise ValueError(f"{profile.path}: robot_motion must define exactly {sorted(required)}")
        scaling = table.get("position_scaling", {})
        if not isinstance(scaling, dict) or (scaling and set(scaling) != {"torso_link", "torso_anchor_names"}):
            raise ValueError("position_scaling must declare torso_link and torso_anchor_names")
        torso_link = scaling.get("torso_link")
        torso_anchors = scaling.get("torso_anchor_names", [])
        if scaling and (not isinstance(torso_link, str) or not torso_link):
            raise ValueError("position_scaling/torso_link must be a body name")
        if not isinstance(torso_anchors, list) or any(not isinstance(n, str) for n in torso_anchors):
            raise ValueError("position_scaling/torso_anchor_names must be semantic landmark names")
        position_links = _string_mapping(table["position_links"], "position_links")
        orientation_links = _string_mapping(table["orientation_links"], "orientation_links")
        names = tuple(position_links)
        parents = table["parents"]
        root = table["root_joint"]
        toes = table["toe_names"]
        if not isinstance(parents, dict) or set(parents) != set(names):
            raise ValueError("robot_motion/parents must define every position landmark")
        if not isinstance(root, str) or root not in names or parents[root] is not None:
            raise ValueError("robot_motion/root_joint must identify the skeleton root")
        if any(parent is not None and parent not in names for parent in parents.values()):
            raise ValueError("robot_motion/parents references an unknown landmark")
        if not isinstance(toes, list) or len(toes) != 2 or len(set(toes)) != 2 or any(t not in names for t in toes):
            raise ValueError("robot_motion/toe_names must identify left and right sole landmarks")
        if not set(orientation_links).issubset(names) or root not in orientation_links:
            raise ValueError("Orientation landmarks must be position landmarks and include the root")
        weights = _numbers(table["orientation_weights"], "orientation_weights", nonnegative=True)
        if set(weights) != set(orientation_links):
            raise ValueError("robot_motion/orientation_weights must cover exactly orientation_links")
        reference_joints = _numbers(table["reference_joint_positions"], "reference_joint_positions")
        reference_base = quaternion_wxyz(table["reference_base_quaternion_wxyz"], label="reference base")
        defaults = _default_robot_defaults()
        defaults[profile.robot] = {
            "robot_dof": profile.robot_dof,
            "robot_height": profile.robot_height,
            "object_name": "ground",
        }
        overrides = table["robot_config"]
        allowed = {
            "foot_sticking_links",
            "foot_contact_links",
            "manual_lb",
            "manual_ub",
            "manual_cost",
            "nominal_tracking_indices",
        }
        if (
            not isinstance(overrides, dict)
            or not allowed.issubset(overrides)
            or set(overrides) - allowed - {"shoulder_direction_chains"}
        ):
            raise ValueError(f"robot_motion/robot_config must define exactly {sorted(allowed)}")
        urdf = Path(profile.robot_urdf_file).expanduser()
        # Maintained assets use package-relative paths; custom assets can use absolute paths.
        if not urdf.is_absolute():
            urdf = _PACKAGE_ROOT / urdf
        config = RobotConfig(
            robot_type=profile.robot,
            robot_defaults=defaults,
            robot_height=profile.robot_height,
            robot_dof=profile.robot_dof,
            robot_urdf_file=str(urdf.resolve()),
            **{**overrides, "nominal_tracking_indices": np.asarray(overrides["nominal_tracking_indices"], dtype=int)},
        )
        if any(n not in position_links or parents[n] != root for n in torso_anchors):
            raise ValueError("Torso anchors must be position landmarks directly below the semantic root")
        result = cls(
            robot=config,
            position_links=position_links,
            parents=tuple(-1 if parents[n] is None else names.index(parents[n]) for n in names),
            root_joint=root,
            toe_names=tuple(toes),
            orientation_links=orientation_links,
            reference_base=tuple(reference_base),
            reference_joints=reference_joints,
            orientation_weights=weights,
            torso_link=torso_link,
            torso_anchor_names=tuple(torso_anchors),
        )
        # Reuse the shared rooted-tree validation, including cycles.
        _ = result.motion_config().resolved_joint_parent_indices
        return result

    def motion_config(self) -> MotionDataConfig:
        return MotionDataConfig(
            data_format="robot",
            robot_type=self.robot.robot_type,
            robot_defaults=self.robot.robot_defaults,
            demo_joints=list(self.position_links),
            joint_parent_indices=self.parents,
            joints_mapping=self.position_links,
            orientation_joints_mapping=self.orientation_links,
            source_root_joint=self.root_joint,
            source_toe_names=self.toe_names,
            orientation_t_pose_robot_base_quaternion_wxyz=self.reference_base,
            orientation_t_pose_robot_joint_positions=self.reference_joints,
            orientation_alignment_quaternions_wxyz={},
        )


def _string_mapping(value, label: str) -> dict[str, str]:
    if (
        not isinstance(value, dict)
        or not value
        or not all(isinstance(k, str) and k and isinstance(v, str) and v for k, v in value.items())
    ):
        raise ValueError(f"robot_motion/{label} must be a nonempty string mapping")
    return dict(value)


def _numbers(value, label: str, *, nonnegative: bool = False) -> dict[str, float]:
    if not isinstance(value, dict) or not all(
        isinstance(k, str)
        and k
        and not isinstance(v, bool)
        and isinstance(v, (float, int))
        and np.isfinite(v)
        and (not nonnegative or v >= 0)
        for k, v in value.items()
    ):
        raise ValueError(f"robot_motion/{label} must contain finite {'nonnegative ' if nonnegative else ''}numbers")
    return {k: float(v) for k, v in value.items()}


class RobotKinematics:
    """Resolve joints by name and evaluate a floating-base robot without simulation."""

    def __init__(self, profile: RobotMotionProfile):
        self.profile = profile
        self.urdf = Path(profile.robot.ROBOT_URDF_FILE)
        self.model = mujoco.MjModel.from_xml_path(str(self.urdf.with_suffix(".xml")))
        self.data = mujoco.MjData(self.model)
        model = self.model
        if model.njnt < 1 or model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE or model.jnt_qposadr[0] != 0:
            raise ValueError("Robot model must begin with one floating root")
        if any(t not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE) for t in model.jnt_type[1:]):
            raise ValueError("Robot model must contain only scalar joints after its floating root")
        self.joint_addresses = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j): int(model.jnt_qposadr[j])
            for j in range(1, model.njnt)
        }
        self.csv_joint_names = actuated_joint_names_from_urdf(self.urdf)
        if (
            len(set(self.csv_joint_names)) != len(self.csv_joint_names)
            or set(self.csv_joint_names) != set(self.joint_addresses)
            or len(self.csv_joint_names) != profile.robot.ROBOT_DOF
        ):
            raise ValueError("CSV URDF, MuJoCo model, and robot_dof must describe the same scalar joints")
        self.csv_addresses = [self.joint_addresses[name] for name in self.csv_joint_names]
        self.position_ids = self.body_ids(profile.position_links.values())
        self.orientation_ids = self.body_ids(profile.orientation_links.values())
        root_id = self.body_ids([profile.position_links[profile.root_joint]])[0]
        if (
            root_id != model.jnt_bodyid[0]
            or profile.orientation_links[profile.root_joint] != profile.position_links[profile.root_joint]
        ):
            raise ValueError("The semantic root must use the floating-root body for position and orientation")
        self.sole_ids = {}
        self.sole_geometry_ids = {}
        for side in ("left", "right"):
            regions = profile.robot.FOOT_CONTACT_LINKS[side]
            self.sole_ids[side] = self.body_ids(
                [
                    regions["heel_positive_lateral"],
                    regions["heel_negative_lateral"],
                    regions["toe"],
                ]
            )
            self.sole_geometry_ids[side] = self.body_ids(
                [
                    regions["heel_positive_lateral"],
                    regions["heel_negative_lateral"],
                    regions["forefoot_positive_lateral"],
                    regions["forefoot_negative_lateral"],
                    regions["toe"],
                ]
            )
        self.torso_id = None if profile.torso_link is None else int(self.body_ids([profile.torso_link])[0])
        unknown = set(profile.reference_joints) - set(self.joint_addresses)
        if unknown:
            raise ValueError(f"Unknown reference-pose joints: {sorted(unknown)}")
        self.reference_qpos = model.qpos0.copy()
        self.reference_qpos[:3] = 0.0
        self.reference_qpos[3:7] = profile.reference_base
        # Match the shared solver's reference pose: undeclared joints are zero.
        self.reference_qpos[7:] = 0.0
        for name, value in profile.reference_joints.items():
            self.reference_qpos[self.joint_addresses[name]] = value
        self.forward(self.reference_qpos)
        self.reference_orientations = self.data.xquat[self.orientation_ids].copy()
        self.reference_positions = self.data.xpos[self.position_ids].copy()
        self.reference_sole_points = np.stack([self.data.xpos[self.sole_geometry_ids[s]] for s in ("left", "right")])
        self.reference_torso_position = None if self.torso_id is None else self.data.xpos[self.torso_id].copy()
        self.reference_torso_quaternion = None if self.torso_id is None else self.data.xquat[self.torso_id].copy()

    def body_ids(self, names) -> np.ndarray:
        names = tuple(names)
        ids = np.asarray([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) for name in names])
        if np.any(ids < 0):
            missing = [n for n, i in zip(names, ids, strict=True) if i < 0]
            raise ValueError(f"Robot profile references missing bodies: {missing}")
        return ids

    def forward(self, qpos: np.ndarray) -> None:
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)

    def read_csv(self, path: Path) -> np.ndarray:
        values = np.loadtxt(path, delimiter=",", ndmin=2)
        expected = 7 + len(self.csv_joint_names)
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != expected or not np.isfinite(values).all():
            raise ValueError(f"Robot CSV must be a nonempty finite numeric matrix with {expected} columns")
        qpos = np.zeros_like(values)
        qpos[:, :3] = values[:, :3]
        quats = values[:, [6, 3, 4, 5]]
        norms = np.linalg.norm(quats, axis=1)
        if np.any(norms < 1e-8):
            raise ValueError("Robot CSV contains zero root quaternions")
        qpos[:, 3:7] = quats / norms[:, None]
        qpos[:, self.csv_addresses] = values[:, 7:]
        return qpos


def load_robot_motion(path: Path, *, source: RobotKinematics, target: RobotMotionProfile, fps: float) -> SourceMotion:
    """Expose measured source FK in the same representation used by human adapters."""
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive; CSV does not store a frame rate")
    qpos = source.read_csv(path)
    profile = source.profile
    frame_count = len(qpos)
    positions = np.empty((frame_count, len(profile.position_links), 3))
    orientations = np.empty((frame_count, len(profile.orientation_links), 4))
    heels = np.empty((frame_count, 2, 3))
    toes = np.empty_like(heels)
    sole_points = np.empty((frame_count, 2, 5, 3))
    torso_quaternions = None if source.torso_id is None else np.empty((frame_count, 4))
    for index, pose in enumerate(qpos):
        source.forward(pose)
        positions[index] = source.data.xpos[source.position_ids]
        orientations[index] = source.data.xquat[source.orientation_ids]
        if torso_quaternions is not None:
            torso_quaternions[index] = source.data.xquat[source.torso_id]
        for side_index, side in enumerate(("left", "right")):
            points = source.data.xpos[source.sole_ids[side]]
            heels[index, side_index] = points[:2].mean(axis=0)
            toes[index, side_index] = points[2]
            sole_points[index, side_index] = source.data.xpos[source.sole_geometry_ids[side]]
    source_root = profile.reference_base
    root_alignment = Rotation.from_quat(np.asarray(source_root)[[1, 2, 3, 0]]).inv() * Rotation.from_quat(
        np.asarray(target.reference_base)[[1, 2, 3, 0]],
    )
    return SourceMotion(
        joints=positions,
        fps=float(fps),
        human_height=profile.robot.ROBOT_HEIGHT,
        source_path=path.resolve(),
        _canonical_joint_names=tuple(profile.position_links),
        joint_parent_indices=profile.parents,
        _root_joint_name=profile.root_joint,
        orientation_joint_names=tuple(profile.orientation_links),
        orientation_quaternions_wxyz=orientations,
        orientation_source="robot_fk",
        t_pose_orientation_joint_names=tuple(profile.orientation_links),
        t_pose_orientation_quaternions_wxyz=source.reference_orientations,
        root_frame_to_robot_base_quaternion_wxyz=root_alignment.as_quat()[[3, 0, 1, 2]],
        foot_heel_positions=heels,
        foot_toe_positions=toes,
        foot_polygon_positions=sole_points,
        position_frame_quaternions_wxyz={} if torso_quaternions is None else {"torso": torso_quaternions},
        metadata={
            "source_kind": "robot",
            "source_robot_type": profile.robot.robot_type,
            "source_robot_urdf": str(source.urdf),
            "source_robot_height": profile.robot.ROBOT_HEIGHT,
            "source_csv_joint_names": np.asarray(source.csv_joint_names, dtype=str),
        },
    )
