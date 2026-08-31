# ruff: noqa: CPY001

"""Load the optional orientation and natural-pose objective profiles."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from holosoma_retargeting.config_types.data_type import MotionDataConfig
from holosoma_retargeting.config_types.robot import RobotConfig, robot_family

ORIENTATION_PROFILE_DATASETS = (
    "amass",
    "fbx_mocap",
    "gvhmr",
    "lafan",
    "noetix_csv_climb",
    "noetix_mocap",
    "OMOMO_new",
)
ORIENTATION_PROFILE_SCHEMA_VERSION = 1
NATURE_PROFILE_SCHEMA_VERSION = 1

_ORIENTATION_DATA_FORMATS = {
    "amass": "amass",
    "fbx_mocap": "fbx_mocap",
    "gvhmr": "gvhmr",
    "lafan": "lafan",
    "noetix_csv_climb": "mocap",
    "noetix_mocap": "noetix_mocap",
    "OMOMO_new": "omomo",
}
_DEFAULT_NATURE_PROFILE_DIR = Path(__file__).resolve().parents[1] / "examples" / "nature_weights"


@dataclass(frozen=True)
class NaturalPoseProfile:
    """Resolved natural joint references and their absolute cost weights."""

    references: dict[str, float]
    weights: dict[str, float]


def _finite_number(
    value: object,
    *,
    label: str,
    non_negative: bool,
) -> float:
    qualifier = "finite and non-negative" if non_negative else "finite"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be {qualifier}")
    number = float(value)
    if not math.isfinite(number) or (non_negative and number < 0.0):
        raise ValueError(f"{label} must be {qualifier}")
    return number


def _profile_path(path: Path, *, robot: str, option: str) -> Path:
    profile_path = path.expanduser()
    if profile_path.is_dir():
        profile_path = profile_path / f"{robot}.json"
    if profile_path.suffix.lower() != ".json":
        raise ValueError(f"{option} must be a JSON file or a profile directory")
    return profile_path


def _read_profile(path: Path, *, option: str) -> dict[str, object]:
    try:
        with path.open(encoding="utf-8") as profile_file:
            profile = json.load(profile_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {option} {path}: {exc}") from exc
    if not isinstance(profile, dict):
        raise ValueError(f"{option} must contain one JSON object")
    return profile


def _validate_schema(
    profile: dict[str, object],
    *,
    option: str,
    robot: str,
    schema_version: int,
    fields: set[str],
) -> None:
    missing_fields = sorted(fields.difference(profile))
    unknown_fields = sorted(set(profile).difference(fields))
    if missing_fields or unknown_fields:
        raise ValueError(
            f"{option} fields do not match the profile schema; missing={missing_fields}, unknown={unknown_fields}",
        )
    configured_version = profile["schema_version"]
    if (
        isinstance(configured_version, bool)
        or not isinstance(configured_version, int)
        or configured_version != schema_version
    ):
        raise ValueError(f"{option} schema_version must be {schema_version}")
    configured_robot = profile["robot"]
    if configured_robot != robot:
        raise ValueError(
            f"{option} robot {configured_robot!r} does not match command robot {robot!r}",
        )


def _resolve_orientation_table(
    configured_weights: object,
    *,
    mapping: dict[str, str],
    dataset: str,
) -> dict[str, float]:
    if not isinstance(configured_weights, dict):
        raise ValueError(
            f"orientation_config dataset {dataset!r} field 'weights' must be an object",
        )

    link_to_human = {robot_link: human_joint for human_joint, robot_link in mapping.items()}
    weights: dict[str, float] = {}
    for configured_name, raw_weight in configured_weights.items():
        if not isinstance(configured_name, str):
            raise ValueError("orientation_config weight names must be strings")
        if configured_name in mapping:
            human_joint = configured_name
        elif configured_name in link_to_human:
            human_joint = link_to_human[configured_name]
        else:
            raise ValueError(
                f"orientation_config dataset {dataset!r} weight "
                f"{configured_name!r} is neither a mapped human keypoint nor robot link",
            )
        if human_joint in weights:
            raise ValueError(
                f"orientation_config dataset {dataset!r} defines "
                f"{human_joint!r} more than once through keypoint/link aliases",
            )
        weights[human_joint] = _finite_number(
            raw_weight,
            label=f"orientation_config weight {dataset}/{configured_name}",
            non_negative=True,
        )

    missing = sorted(set(mapping).difference(weights))
    if missing:
        raise ValueError(
            f"orientation_config dataset {dataset!r} must define every mapped human keypoint; missing: {missing}",
        )
    return weights


def _orientation_profile_tables(
    path: Path,
    *,
    robot: str,
) -> dict[str, dict[str, float]]:
    profile = _read_profile(path, option="orientation_config")
    _validate_schema(
        profile,
        option="orientation_config",
        robot=robot,
        schema_version=ORIENTATION_PROFILE_SCHEMA_VERSION,
        fields={"schema_version", "robot", "datasets"},
    )
    datasets = profile["datasets"]
    if not isinstance(datasets, dict):
        raise ValueError("orientation_config field 'datasets' must be an object")
    expected_datasets = set(ORIENTATION_PROFILE_DATASETS)
    missing_datasets = sorted(expected_datasets.difference(datasets))
    unknown_datasets = sorted(set(datasets).difference(expected_datasets))
    if missing_datasets or unknown_datasets:
        raise ValueError(
            "orientation_config must contain exactly the supported orientation "
            f"datasets; missing={missing_datasets}, unknown={unknown_datasets}",
        )

    tables: dict[str, dict[str, float]] = {}
    for dataset in ORIENTATION_PROFILE_DATASETS:
        table = datasets[dataset]
        if not isinstance(table, dict):
            raise ValueError(f"orientation_config dataset {dataset!r} must be an object")
        unknown_table_fields = sorted(set(table).difference({"weights"}))
        if unknown_table_fields:
            raise ValueError(
                f"orientation_config dataset {dataset!r} contains unknown fields: {unknown_table_fields}",
            )
        mapping = MotionDataConfig(
            data_format=_ORIENTATION_DATA_FORMATS[dataset],
            robot_type=robot,
        ).resolved_orientation_joints_mapping
        if not mapping:
            raise ValueError(
                f"dataset={dataset!r} has no orientation mapping for robot={robot!r}",
            )
        tables[dataset] = _resolve_orientation_table(
            table.get("weights"),
            mapping=mapping,
            dataset=dataset,
        )
    return tables


def resolve_orientation_weights(
    *,
    robot: str,
    dataset: str,
    data_format: str,
    uniform_weight: float | None,
    config_path: Path | None,
) -> dict[str, float]:
    """Resolve one optional orientation objective into per-mapping weights."""

    if uniform_weight is not None and config_path is not None:
        raise ValueError(
            "--orientation_weights and --orientation_config are mutually exclusive",
        )
    if uniform_weight is None and config_path is None:
        return {}
    if dataset not in ORIENTATION_PROFILE_DATASETS:
        raise ValueError(
            f"dataset={dataset!r} does not provide direct source orientations",
        )

    mapping = MotionDataConfig(
        data_format=data_format,
        robot_type=robot,
    ).resolved_orientation_joints_mapping
    if not mapping:
        raise ValueError(
            f"dataset={dataset!r} has no orientation mapping for robot={robot!r}",
        )
    if uniform_weight is not None:
        weight = _finite_number(
            uniform_weight,
            label="orientation_weights",
            non_negative=True,
        )
        return dict.fromkeys(mapping, weight) if weight > 0.0 else {}

    assert config_path is not None
    profile_robot = robot_family(robot)
    path = _profile_path(
        config_path,
        robot=profile_robot,
        option="orientation_config",
    )
    resolved = _orientation_profile_tables(path, robot=profile_robot)[dataset]
    return resolved if any(weight > 0.0 for weight in resolved.values()) else {}


def _load_nature_profile(path: Path, *, robot: str) -> NaturalPoseProfile:
    profile = _read_profile(path, option="nature_config")
    _validate_schema(
        profile,
        option="nature_config",
        robot=robot,
        schema_version=NATURE_PROFILE_SCHEMA_VERSION,
        fields={
            "schema_version",
            "robot",
            "robot_urdf_name",
            "references",
            "weights",
        },
    )
    expected_urdf_name = Path(RobotConfig(robot_type=robot).ROBOT_URDF_FILE).name
    if profile["robot_urdf_name"] != expected_urdf_name:
        raise ValueError(
            f"nature_config robot_urdf_name does not match the selected robot; expected {expected_urdf_name!r}",
        )

    references_raw = profile["references"]
    weights_raw = profile["weights"]
    if not isinstance(references_raw, dict) or not isinstance(weights_raw, dict):
        raise ValueError("nature_config references and weights must be objects")
    if set(references_raw) != set(weights_raw):
        missing_references = sorted(set(weights_raw).difference(references_raw))
        missing_weights = sorted(set(references_raw).difference(weights_raw))
        raise ValueError(
            "nature_config references and weights must contain exactly the same "
            f"joint names; missing_references={missing_references}, missing_weights={missing_weights}",
        )
    if not references_raw:
        raise ValueError("nature_config must define at least one robot joint")

    references: dict[str, float] = {}
    weights: dict[str, float] = {}
    for joint_name, reference in references_raw.items():
        if not isinstance(joint_name, str) or not joint_name:
            raise ValueError("nature_config joint names must be non-empty strings")
        references[joint_name] = _finite_number(
            reference,
            label=f"nature_config references/{joint_name}",
            non_negative=False,
        )
        weights[joint_name] = _finite_number(
            weights_raw[joint_name],
            label=f"nature_config weights/{joint_name}",
            non_negative=True,
        )
    return NaturalPoseProfile(references=references, weights=weights)


def resolve_natural_pose(
    *,
    robot: str,
    uniform_weight: float | None,
    config_path: Path | None,
) -> NaturalPoseProfile:
    """Resolve one optional natural-pose objective and its fixed references."""

    if uniform_weight is not None and config_path is not None:
        raise ValueError(
            "--nature_weights and --nature_config are mutually exclusive",
        )
    if uniform_weight is None and config_path is None:
        return NaturalPoseProfile(references={}, weights={})

    selected_path = config_path or _DEFAULT_NATURE_PROFILE_DIR
    profile_robot = "e1_23dof" if robot == "e1" else robot
    path = _profile_path(
        selected_path,
        robot=profile_robot,
        option="nature_config",
    )
    profile = _load_nature_profile(path, robot=profile_robot)
    if uniform_weight is None:
        return profile

    weight = _finite_number(
        uniform_weight,
        label="nature_weights",
        non_negative=True,
    )
    return NaturalPoseProfile(
        references=profile.references,
        weights=dict.fromkeys(profile.references, weight),
    )
