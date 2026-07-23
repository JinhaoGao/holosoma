"""OMOMO sequence naming, inventory, and preflight validation."""

from __future__ import annotations

import json
import pickle
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

OMOMO_OBJECT_NAMES: tuple[str, ...] = (
    "clothesstand",
    "floorlamp",
    "largebox",
    "largetable",
    "monitor",
    "plasticbox",
    "smallbox",
    "smalltable",
    "suitcase",
    "trashcan",
    "tripod",
    "whitechair",
    "woodchair",
)

_SEQUENCE_PATTERN = re.compile(r"^(?P<subject>sub[1-9][0-9]*)_(?P<object>[a-z][a-z0-9]*)_(?P<index>[0-9]{3})$")
_RESULT_PATTERN = re.compile(
    r"^(?P<sequence>sub[1-9][0-9]*_[a-z][a-z0-9]*_[0-9]{3})"
    r"(?:_(?:original|augmented|trans_[0-9]+|rot_[0-9]+))?$"
)


@dataclass(frozen=True)
class OmomoSequenceName:
    """Structured OMOMO sequence identifier."""

    subject: str
    object_name: str
    sequence_index: int

    @property
    def stem(self) -> str:
        """Return the canonical file stem."""

        return f"{self.subject}_{self.object_name}_{self.sequence_index:03d}"


@dataclass(frozen=True)
class OmomoPreflightIssue:
    """One actionable dataset or asset validation problem."""

    code: str
    path: str
    message: str


@dataclass(frozen=True)
class OmomoObjectInventory:
    """Per-object inventory and conventional asset availability."""

    object_name: str
    sequence_count: int
    mesh_path: str | None
    urdf_path: str | None
    mesh_exists: bool
    urdf_exists: bool


@dataclass(frozen=True)
class OmomoPreflightReport:
    """Serializable OMOMO dataset preflight result."""

    data_dir: str
    total_files: int
    valid_sequences: int
    subjects: tuple[str, ...]
    objects: tuple[OmomoObjectInventory, ...]
    issues: tuple[OmomoPreflightIssue, ...]

    @property
    def ok(self) -> bool:
        """Whether the dataset and requested assets passed every check."""

        return not self.issues

    def to_dict(self) -> dict:
        """Return a JSON-compatible representation."""

        payload = asdict(self)
        payload["ok"] = self.ok
        return payload

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the report as stable JSON."""

        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def parse_omomo_sequence_name(
    value: str | Path,
    *,
    require_known_object: bool = True,
) -> OmomoSequenceName:
    """Parse and validate ``subN_object_NNN`` OMOMO naming."""

    stem = Path(value).stem
    match = _SEQUENCE_PATTERN.fullmatch(stem)
    if match is None:
        raise ValueError(f"Invalid OMOMO sequence name {stem!r}; expected 'subN_object_NNN'")

    object_name = match.group("object")
    if require_known_object and object_name not in OMOMO_OBJECT_NAMES:
        supported = ", ".join(OMOMO_OBJECT_NAMES)
        raise ValueError(f"Unknown OMOMO object {object_name!r} in {stem!r}; supported objects: {supported}")

    return OmomoSequenceName(
        subject=match.group("subject"),
        object_name=object_name,
        sequence_index=int(match.group("index")),
    )


def parse_omomo_result_name(value: str | Path) -> OmomoSequenceName:
    """Parse an OMOMO source or retargeting-result filename."""

    stem = Path(value).stem
    match = _RESULT_PATTERN.fullmatch(stem)
    if match is None:
        raise ValueError(
            f"Invalid OMOMO result name {stem!r}; expected a canonical sequence "
            "with an optional retargeting augmentation suffix"
        )
    return parse_omomo_sequence_name(match.group("sequence"))


def resolve_omomo_result_object_name(
    result_path: str | Path,
    explicit_object_name: str | None = None,
) -> str:
    """Resolve an OMOMO result's object from metadata and its canonical filename."""

    path = Path(result_path)
    if explicit_object_name is not None and explicit_object_name not in OMOMO_OBJECT_NAMES:
        raise ValueError(f"Unknown OMOMO object override: {explicit_object_name!r}")

    metadata_object: str | None = None
    if path.is_file() and path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            if "object_name" in data:
                raw_object = np.asarray(data["object_name"])
                if raw_object.size != 1:
                    raise ValueError(f"Result object_name metadata must be scalar: {path}")
                value = str(raw_object.item())
                if value:
                    metadata_object = value
                    if metadata_object not in OMOMO_OBJECT_NAMES:
                        raise ValueError(
                            f"Result {path} declares non-OMOMO object {metadata_object!r}"
                        )

    filename_object: str | None = None
    try:
        filename_object = parse_omomo_result_name(path).object_name
    except ValueError:
        pass

    discovered = {
        value
        for value in (metadata_object, filename_object)
        if value is not None
    }
    if len(discovered) > 1:
        raise ValueError(
            f"OMOMO result {path} has conflicting object metadata and filename: "
            f"{', '.join(sorted(discovered))}"
        )
    inferred_object = next(iter(discovered), None)
    if (
        explicit_object_name is not None
        and inferred_object is not None
        and explicit_object_name != inferred_object
    ):
        raise ValueError(
            f"OMOMO result {path} contains object {inferred_object!r}, "
            f"but the explicit override is {explicit_object_name!r}"
        )
    resolved = explicit_object_name or inferred_object
    if resolved is None:
        raise ValueError(
            f"Could not infer the OMOMO object for {path}; preserve result metadata, "
            "use a canonical OMOMO filename, or pass an explicit object name"
        )
    return resolved


def select_omomo_files(
    data_dir: str | Path,
    object_names: Iterable[str] | None = None,
) -> list[Path]:
    """Discover OMOMO files and apply exact object-category filtering."""

    directory = Path(data_dir).expanduser()
    if not directory.is_dir():
        raise FileNotFoundError(f"OMOMO data directory not found: {directory}")

    selected_objects = set(object_names) if object_names is not None else None
    if selected_objects is not None:
        unknown = selected_objects.difference(OMOMO_OBJECT_NAMES)
        if unknown:
            raise ValueError(f"Unknown OMOMO object filters: {', '.join(sorted(unknown))}")

    selected: list[Path] = []
    for path in sorted(directory.glob("*.pt")):
        sequence = parse_omomo_sequence_name(path)
        if selected_objects is None or sequence.object_name in selected_objects:
            selected.append(path)
    return selected


def conventional_object_asset_paths(asset_root: str | Path, object_name: str) -> tuple[Path, Path]:
    """Return the conventional mesh and URDF paths for one OMOMO object."""

    if object_name not in OMOMO_OBJECT_NAMES:
        raise ValueError(f"Unknown OMOMO object: {object_name!r}")
    object_dir = Path(asset_root).expanduser() / object_name
    return object_dir / f"{object_name}.obj", object_dir / f"{object_name}.urdf"


def _load_subject_heights(height_file: Path, issues: list[OmomoPreflightIssue]) -> dict[str, float]:
    if not height_file.is_file():
        issues.append(
            OmomoPreflightIssue(
                code="missing_height_file",
                path=str(height_file),
                message="OMOMO subject height dictionary is missing",
            )
        )
        return {}

    try:
        with height_file.open("rb") as file:
            raw_heights = pickle.load(file)
    except Exception as exc:
        issues.append(
            OmomoPreflightIssue(
                code="invalid_height_file",
                path=str(height_file),
                message=f"Could not load OMOMO height dictionary: {exc}",
            )
        )
        return {}

    if not isinstance(raw_heights, dict):
        issues.append(
            OmomoPreflightIssue(
                code="invalid_height_file",
                path=str(height_file),
                message="OMOMO height dictionary must contain a mapping",
            )
        )
        return {}

    heights: dict[str, float] = {}
    for subject, value in raw_heights.items():
        try:
            height = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(height) and height > 0:
            heights[str(subject)] = height
    return heights


def _validate_tensor(path: Path) -> str | None:
    try:
        tensor = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        return f"Could not load tensor: {exc}"
    if not torch.is_tensor(tensor) or tensor.ndim != 2 or tensor.shape[0] == 0 or tensor.shape[1] < 325:
        shape = tuple(tensor.shape) if torch.is_tensor(tensor) else None
        return f"Expected a non-empty tensor with shape (T, >=325), got {shape}"
    required_values = tensor[:, 162:325]
    if not torch.isfinite(required_values).all():
        return "Required human-joint or object-pose values contain NaN or Inf"
    return None


def preflight_omomo_dataset(
    data_dir: str | Path,
    *,
    asset_root: str | Path | None = None,
    height_file: str | Path | None = None,
    validate_tensors: bool = True,
) -> OmomoPreflightReport:
    """Inventory an OMOMO directory and validate names, heights, tensors, and assets."""

    directory = Path(data_dir).expanduser()
    if not directory.is_dir():
        raise FileNotFoundError(f"OMOMO data directory not found: {directory}")

    issues: list[OmomoPreflightIssue] = []
    height_path = (
        Path(height_file).expanduser()
        if height_file is not None
        else directory.parent / "height_dict.pkl"
    )
    subject_heights = _load_subject_heights(height_path, issues)
    counts: Counter[str] = Counter()
    subjects: set[str] = set()
    valid_sequences = 0
    files = sorted(directory.glob("*.pt"))

    for path in files:
        try:
            sequence = parse_omomo_sequence_name(path)
        except ValueError as exc:
            issues.append(
                OmomoPreflightIssue(
                    code="invalid_sequence_name",
                    path=str(path),
                    message=str(exc),
                )
            )
            continue

        valid_sequences += 1
        counts[sequence.object_name] += 1
        subjects.add(sequence.subject)
        if sequence.subject not in subject_heights:
            issues.append(
                OmomoPreflightIssue(
                    code="missing_subject_height",
                    path=str(path),
                    message=f"No valid height is registered for {sequence.subject}",
                )
            )
        if validate_tensors:
            tensor_error = _validate_tensor(path)
            if tensor_error is not None:
                issues.append(
                    OmomoPreflightIssue(
                        code="invalid_tensor",
                        path=str(path),
                        message=tensor_error,
                    )
                )

    if not files:
        issues.append(
            OmomoPreflightIssue(
                code="empty_dataset",
                path=str(directory),
                message="No .pt files were found",
            )
        )

    inventories: list[OmomoObjectInventory] = []
    for object_name in OMOMO_OBJECT_NAMES:
        mesh_path: Path | None = None
        urdf_path: Path | None = None
        if asset_root is not None:
            mesh_path, urdf_path = conventional_object_asset_paths(asset_root, object_name)
            if not mesh_path.is_file():
                issues.append(
                    OmomoPreflightIssue(
                        code="missing_object_mesh",
                        path=str(mesh_path),
                        message=f"Missing mesh for OMOMO object {object_name}",
                    )
                )
            if not urdf_path.is_file():
                issues.append(
                    OmomoPreflightIssue(
                        code="missing_object_urdf",
                        path=str(urdf_path),
                        message=f"Missing URDF for OMOMO object {object_name}",
                    )
                )
        inventories.append(
            OmomoObjectInventory(
                object_name=object_name,
                sequence_count=counts[object_name],
                mesh_path=str(mesh_path) if mesh_path is not None else None,
                urdf_path=str(urdf_path) if urdf_path is not None else None,
                mesh_exists=mesh_path.is_file() if mesh_path is not None else False,
                urdf_exists=urdf_path.is_file() if urdf_path is not None else False,
            )
        )

    return OmomoPreflightReport(
        data_dir=str(directory),
        total_files=len(files),
        valid_sequences=valid_sequences,
        subjects=tuple(sorted(subjects)),
        objects=tuple(inventories),
        issues=tuple(issues),
    )
