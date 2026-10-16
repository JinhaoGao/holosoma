# ruff: noqa: CPY001

"""Load one authoritative retargeting profile for each concrete robot."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from pathlib import Path

from holosoma_retargeting.config_types.data_type import MotionDataConfig
from holosoma_retargeting.config_types.retargeter import ShoulderDirectionConfig

ROBOT_PROFILE_SCHEMA_VERSION = 1
ORIENTATION_PROFILE_DATASETS = (
    "amass",
    "fbx_mocap",
    "gvhmr",
    "lafan",
    "noetix_csv_climb",
    "noetix_mocap",
    "OMOMO_new",
)

_ORIENTATION_DATA_FORMATS = {
    "amass": "amass",
    "fbx_mocap": "fbx_mocap",
    "gvhmr": "gvhmr",
    "lafan": "lafan",
    "noetix_csv_climb": "mocap",
    "noetix_mocap": "noetix_mocap",
    "OMOMO_new": "omomo",
}
_DEFAULT_PROFILE_DIR = Path(__file__).resolve().parents[1] / "examples" / "robot_profiles"
_CONCRETE_ROBOT = {"e1": "e1_23dof"}


@dataclass(frozen=True)
class NaturalPoseProfile:
    """Expanded fixed joint references and absolute regularization weights."""

    references: dict[str, float]
    weights: dict[str, float]


@dataclass(frozen=True)
class RetargetingProfileDefaults:
    """Robot-level solver defaults that may be overridden from the CLI."""

    dynamic_ground_window: bool
    interaction_mesh_weight: float
    arm_interaction_mesh_weight_scale: float
    root_position_weight: float
    root_orientation_weight: float
    q_a_init_idx: int
    activate_joint_limits: bool
    activate_obj_non_penetration: bool
    foot_sticking: bool
    penetration_tolerance: float
    foot_sticking_tolerance: float
    step_size: float
    w_nominal_tracking_init: float
    nominal_tracking_tau: float


@dataclass(frozen=True)
class RobotProfile:
    """Fully validated contents of one robot JSON profile."""

    path: Path
    robot: str
    robot_dof: int
    robot_height: float
    robot_urdf_file: str
    robot_mujoco_xml_file: str
    retargeting: RetargetingProfileDefaults
    shoulder_direction: ShoulderDirectionConfig
    orientation_enabled: bool
    orientation_preview: bool
    orientation_weights_by_dataset: dict[str, dict[str, float]]
    natural_pose_enabled: bool
    natural_pose: NaturalPoseProfile
    preserve_human_knees: bool
    body_scaling_mode: str
    leg_length_scale: float
    position_weights_by_dataset: dict[str, dict[str, float]]


def concrete_robot_name(robot: str) -> str:
    """Map a public compatibility alias to its maintained robot profile."""

    return _CONCRETE_ROBOT.get(robot, robot)


def _finite_number(
    value: object,
    *,
    label: str,
    non_negative: bool,
    positive: bool = False,
) -> float:
    qualifier = "finite"
    if positive:
        qualifier += " and positive"
    elif non_negative:
        qualifier += " and non-negative"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be {qualifier}")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0.0) or (non_negative and number < 0.0):
        raise ValueError(f"{label} must be {qualifier}")
    return number


def _boolean(value: object, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _integer(value: object, *, label: str, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if positive and value <= 0:
        raise ValueError(f"{label} must be positive")
    return value


def _object(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a JSON object with string keys")
    return value


def _exact_fields(
    value: dict[str, object],
    expected: set[str],
    *,
    label: str,
) -> None:
    missing = sorted(expected.difference(value))
    unknown = sorted(set(value).difference(expected))
    if missing or unknown:
        raise ValueError(
            f"{label} fields do not match the schema; missing={missing}, unknown={unknown}",
        )


def _profile_path(path: Path | None, *, robot: str) -> Path:
    selected = (path or _DEFAULT_PROFILE_DIR).expanduser()
    if selected.is_dir():
        selected = selected / f"{robot}.json"
    if selected.suffix.lower() != ".json":
        raise ValueError("robot_profile must be a JSON file or profile directory")
    return selected


def _read_profile(path: Path) -> dict[str, object]:
    try:
        with path.open(encoding="utf-8") as profile_file:
            profile = json.load(profile_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read robot_profile {path}: {exc}") from exc
    return _object(profile, label="robot_profile")


def _load_retargeting_defaults(
    value: object,
) -> RetargetingProfileDefaults:
    table = _object(value, label="retargeting")
    expected = {field.name for field in fields(RetargetingProfileDefaults)}
    _exact_fields(table, expected, label="retargeting")
    boolean_fields = {
        "dynamic_ground_window",
        "activate_joint_limits",
        "activate_obj_non_penetration",
        "foot_sticking",
    }
    positive_fields = {
        "step_size",
        "nominal_tracking_tau",
    }
    values: dict[str, object] = {}
    for name in expected:
        if name in boolean_fields:
            values[name] = _boolean(table[name], label=f"retargeting/{name}")
        elif name == "q_a_init_idx":
            values[name] = _integer(table[name], label=f"retargeting/{name}")
        else:
            values[name] = _finite_number(
                table[name],
                label=f"retargeting/{name}",
                non_negative=True,
                positive=name in positive_fields,
            )
    return RetargetingProfileDefaults(**values)  # type: ignore[arg-type]


def _load_shoulder_direction(value: object) -> ShoulderDirectionConfig:
    table = _object(value, label="shoulder_direction")
    expected = {field.name for field in fields(ShoulderDirectionConfig)}
    _exact_fields(table, expected, label="shoulder_direction")
    return ShoulderDirectionConfig(
        enable=_boolean(table["enable"], label="shoulder_direction/enable"),
        direction_weight=_finite_number(
            table["direction_weight"],
            label="shoulder_direction/direction_weight",
            non_negative=True,
        ),
        wrist_axis_weight_scale=_finite_number(
            table["wrist_axis_weight_scale"],
            label="shoulder_direction/wrist_axis_weight_scale",
            non_negative=True,
        ),
    )


def _resolve_orientation_table(
    value: object,
    *,
    robot: str,
    dataset: str,
) -> dict[str, float]:
    table = _object(value, label=f"orientation/datasets/{dataset}")
    _exact_fields(
        table,
        {"default_weight", "overrides"},
        label=f"orientation/datasets/{dataset}",
    )
    default_weight = _finite_number(
        table["default_weight"],
        label=f"orientation/datasets/{dataset}/default_weight",
        non_negative=True,
    )
    mapping = MotionDataConfig(
        data_format=_ORIENTATION_DATA_FORMATS[dataset],
        robot_type=robot,
    ).resolved_orientation_joints_mapping
    if not mapping:
        raise ValueError(
            f"dataset={dataset!r} has no orientation mapping for robot={robot!r}",
        )
    link_to_human = {robot_link: human_joint for human_joint, robot_link in mapping.items()}
    weights = dict.fromkeys(mapping, default_weight)
    seen: set[str] = set()
    for configured_name, raw_weight in _object(
        table["overrides"],
        label=f"orientation/datasets/{dataset}/overrides",
    ).items():
        if configured_name in mapping:
            human_joint = configured_name
        elif configured_name in link_to_human:
            human_joint = link_to_human[configured_name]
        else:
            raise ValueError(
                f"orientation dataset {dataset!r} override {configured_name!r} "
                "is neither a mapped human keypoint nor robot link",
            )
        if human_joint in seen:
            raise ValueError(
                f"orientation dataset {dataset!r} defines {human_joint!r} twice",
            )
        seen.add(human_joint)
        weights[human_joint] = _finite_number(
            raw_weight,
            label=f"orientation/datasets/{dataset}/overrides/{configured_name}",
            non_negative=True,
        )
    return weights


def _load_orientation(
    value: object,
    *,
    robot: str,
) -> tuple[bool, bool, dict[str, dict[str, float]]]:
    table = _object(value, label="orientation")
    _exact_fields(
        table,
        {"enable", "preview", "datasets"},
        label="orientation",
    )
    datasets = _object(table["datasets"], label="orientation/datasets")
    _exact_fields(
        datasets,
        set(ORIENTATION_PROFILE_DATASETS),
        label="orientation/datasets",
    )
    resolved = {
        dataset: _resolve_orientation_table(
            datasets[dataset],
            robot=robot,
            dataset=dataset,
        )
        for dataset in ORIENTATION_PROFILE_DATASETS
    }
    return (
        _boolean(table["enable"], label="orientation/enable"),
        _boolean(table["preview"], label="orientation/preview"),
        resolved,
    )


def _load_natural_pose(
    value: object,
    *,
    robot_dof: int,
) -> tuple[bool, NaturalPoseProfile]:
    table = _object(value, label="natural_pose")
    _exact_fields(
        table,
        {
            "enable",
            "joint_names",
            "reference_default",
            "reference_overrides",
            "weight_default",
            "weight_overrides",
        },
        label="natural_pose",
    )
    joint_names_raw = table["joint_names"]
    if not isinstance(joint_names_raw, list) or not all(isinstance(name, str) and name for name in joint_names_raw):
        raise ValueError("natural_pose/joint_names must be non-empty strings")
    joint_names = tuple(joint_names_raw)
    if len(joint_names) != robot_dof or len(set(joint_names)) != len(joint_names):
        raise ValueError(
            "natural_pose/joint_names must contain every actuated joint exactly once",
        )
    reference_default = _finite_number(
        table["reference_default"],
        label="natural_pose/reference_default",
        non_negative=False,
    )
    weight_default = _finite_number(
        table["weight_default"],
        label="natural_pose/weight_default",
        non_negative=True,
    )
    references = dict.fromkeys(joint_names, reference_default)
    weights = dict.fromkeys(joint_names, weight_default)
    for field_name, destination, non_negative in (
        ("reference_overrides", references, False),
        ("weight_overrides", weights, True),
    ):
        overrides = _object(table[field_name], label=f"natural_pose/{field_name}")
        unknown = sorted(set(overrides).difference(joint_names))
        if unknown:
            raise ValueError(
                f"natural_pose/{field_name} contains unknown joints: {unknown}",
            )
        for name, raw_value in overrides.items():
            destination[name] = _finite_number(
                raw_value,
                label=f"natural_pose/{field_name}/{name}",
                non_negative=non_negative,
            )
    return (
        _boolean(table["enable"], label="natural_pose/enable"),
        NaturalPoseProfile(references=references, weights=weights),
    )


def load_robot_profile(
    *,
    robot: str,
    config_path: Path | None = None,
) -> RobotProfile:
    """Read and validate the profile selected by one public robot name."""

    concrete_robot = concrete_robot_name(robot)
    path = _profile_path(config_path, robot=concrete_robot)
    profile = _read_profile(path)
    preserve_human_knees = _boolean(profile.pop("preserve_human_knees", False), label="preserve_human_knees")
    if preserve_human_knees and concrete_robot != "m5fanqu":
        raise ValueError("preserve_human_knees is supported only for m5fanqu")
    scaling = _object(profile.pop("body_scaling", {}), label="body_scaling")
    if set(scaling).difference({"mode", "leg_length_scale"}):
        raise ValueError("Unknown body_scaling fields")
    scaling_mode = scaling.get("mode", "uniform")
    if scaling_mode not in {"uniform", "robot_segments"}:
        raise ValueError("Unknown body_scaling mode")
    if scaling and concrete_robot != "m5fanqu":
        raise ValueError("body_scaling is specific to m5fanqu; other robots retain their own reconstruction")
    leg_length_scale = _finite_number(
        scaling.get("leg_length_scale", 1.0), label="body_scaling/leg_length_scale", non_negative=True, positive=True
    )
    if concrete_robot == "m5fanqu" and (scaling_mode != "robot_segments" or not preserve_human_knees):
        raise ValueError("m5fanqu requires source-preserving robot_segments scaling")
    position_weights: dict[str, dict[str, float]] = {}
    for dataset, raw_weights in _object(profile.pop("position_weights", {}), label="position_weights").items():
        if dataset not in _ORIENTATION_DATA_FORMATS:
            raise ValueError(f"Unknown position_weights dataset: {dataset}")
        mapping = MotionDataConfig(
            data_format=_ORIENTATION_DATA_FORMATS[dataset], robot_type=robot
        ).resolved_joints_mapping
        weights = _object(raw_weights, label=f"position_weights/{dataset}")
        if set(weights).difference(mapping):
            raise ValueError(f"position_weights/{dataset} contains unmapped joints")
        position_weights[dataset] = {
            name: _finite_number(value, label=f"position_weights/{dataset}/{name}", non_negative=True)
            for name, value in weights.items()
        }
    _exact_fields(
        profile,
        {
            "schema_version",
            "robot",
            "robot_dof",
            "robot_height",
            "robot_urdf_file",
            "robot_mujoco_xml_file",
            "retargeting",
            "shoulder_direction",
            "orientation",
            "natural_pose",
        },
        label="robot_profile",
    )
    if (
        isinstance(profile["schema_version"], bool)
        or not isinstance(profile["schema_version"], int)
        or profile["schema_version"] != ROBOT_PROFILE_SCHEMA_VERSION
    ):
        raise ValueError(
            f"robot_profile schema_version must be {ROBOT_PROFILE_SCHEMA_VERSION}",
        )
    if profile["robot"] != concrete_robot:
        raise ValueError(
            f"robot_profile robot {profile['robot']!r} does not match command robot {concrete_robot!r}",
        )
    robot_dof = _integer(
        profile["robot_dof"],
        label="robot_profile/robot_dof",
        positive=True,
    )
    robot_height = _finite_number(
        profile["robot_height"],
        label="robot_profile/robot_height",
        non_negative=False,
        positive=True,
    )
    robot_urdf_file = profile["robot_urdf_file"]
    if not isinstance(robot_urdf_file, str) or not robot_urdf_file or Path(robot_urdf_file).suffix.lower() != ".urdf":
        raise ValueError("robot_profile/robot_urdf_file must name one URDF file")
    robot_mujoco_xml_file = profile["robot_mujoco_xml_file"]
    if (
        not isinstance(robot_mujoco_xml_file, str)
        or not robot_mujoco_xml_file
        or Path(robot_mujoco_xml_file).suffix.lower() != ".xml"
    ):
        raise ValueError(
            "robot_profile/robot_mujoco_xml_file must name one MuJoCo XML file",
        )
    orientation_enabled, orientation_preview, orientation_weights = _load_orientation(
        profile["orientation"], robot=robot
    )
    natural_pose_enabled, natural_pose = _load_natural_pose(
        profile["natural_pose"],
        robot_dof=robot_dof,
    )
    return RobotProfile(
        path=path.resolve(),
        robot=concrete_robot,
        robot_dof=robot_dof,
        robot_height=robot_height,
        robot_urdf_file=robot_urdf_file,
        robot_mujoco_xml_file=robot_mujoco_xml_file,
        retargeting=_load_retargeting_defaults(profile["retargeting"]),
        shoulder_direction=_load_shoulder_direction(
            profile["shoulder_direction"],
        ),
        orientation_enabled=orientation_enabled,
        orientation_preview=orientation_preview,
        orientation_weights_by_dataset=orientation_weights,
        natural_pose_enabled=natural_pose_enabled,
        natural_pose=natural_pose,
        preserve_human_knees=preserve_human_knees,
        body_scaling_mode=str(scaling_mode),
        leg_length_scale=leg_length_scale,
        position_weights_by_dataset=position_weights,
    )


def resolve_orientation_weights(
    *,
    profile: RobotProfile,
    dataset: str,
    data_format: str,
    enabled: bool,
    uniform_weight: float | None,
    robot: str,
) -> dict[str, float]:
    """Resolve profile weights or an explicit uniform CLI override."""

    if uniform_weight is not None:
        weight = _finite_number(
            uniform_weight,
            label="orientation_weights",
            non_negative=True,
        )
        if weight == 0.0:
            return {}
        mapping = MotionDataConfig(
            data_format=data_format,
            robot_type=robot,
        ).resolved_orientation_joints_mapping
        if not mapping:
            raise ValueError(
                f"dataset={dataset!r} does not provide direct source orientations",
            )
        return dict.fromkeys(mapping, weight)
    if not enabled:
        return {}
    try:
        weights = profile.orientation_weights_by_dataset[dataset]
    except KeyError as exc:
        raise ValueError(
            f"dataset={dataset!r} does not provide direct source orientations",
        ) from exc
    return weights if any(weight > 0.0 for weight in weights.values()) else {}


def resolve_natural_pose(
    *,
    profile: RobotProfile,
    enabled: bool,
    uniform_weight: float | None,
) -> NaturalPoseProfile:
    """Resolve profile joint weights or an explicit uniform CLI override."""

    if uniform_weight is not None:
        weight = _finite_number(
            uniform_weight,
            label="nature_weights",
            non_negative=True,
        )
        return NaturalPoseProfile(
            references=profile.natural_pose.references,
            weights=dict.fromkeys(profile.natural_pose.references, weight),
        )
    if not enabled:
        return NaturalPoseProfile(references={}, weights={})
    return profile.natural_pose
