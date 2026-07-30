"""Catalog and generated MuJoCo scenes for OMOMO object assets."""

from __future__ import annotations

import hashlib
import io
import os
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from holosoma_retargeting.data_utils.omomo import OMOMO_OBJECT_NAMES

OMOMO_MESH_SHA256: dict[str, str] = {
    "clothesstand": "60a20e223309f3f85bdcbb270a6eaaef657fbe90a83339795059395b7ee8a46d",
    "floorlamp": "b6886bfd29cbaf80af583d2e6b1574d7e842b35703016d55466e1463db08c029",
    "largebox": "dbcc11281f62e9226f49165252375080d2e490e5a3f0ab6ba917acbd8f7abc1c",
    "largetable": "b2375fe0d799a37d5ca1a0a8e15ae7cfb476bed0b0b5449a2698ed57d37e378c",
    "monitor": "2b49ad22e4b6b0d333db6ffdc5aad4409766f0c73fa6e24ee5076edf4f87c074",
    "plasticbox": "33a4df8ed2e52e4de59a602bbe10d50816b243bc668630715ccdcc59e6e93fd9",
    "smallbox": "e263f3ffce33f85e55e083a6eae19c69e0ae4a2d0545a3c256d7d17d4fbaf116",
    "smalltable": "ffd6ddd0e5b479fd2636ec2c318fa5a783a99b8d333c52ced84cd186cd544996",
    "suitcase": "4271f2e55ac85f2ebb2a4b80494c8392819476317d9bc5c95e25c20f9597e87b",
    "trashcan": "d60fe42102072518c3ec74ce910b0d6d640520b50b66dbd715e6713e79c84b62",
    "tripod": "435265929e6841c53d303196bca3a8cc13c9a2944d042b9b91c1a315a6206a24",
    "whitechair": "88b5da34924172bdcaa4ae8269332ff5ba2f965048f57dc09fa3c6bbe74ca63f",
    "woodchair": "734847bd49d0c015459b2b74e5256650e6660e077dfadfa3bfff622949f34403",
}


@dataclass(frozen=True)
class OmomoObjectAsset:
    """Paths and stable simulator names for one OMOMO object."""

    name: str
    mesh_path: Path
    collision_mesh_paths: tuple[Path, ...]
    urdf_path: Path
    mesh_sha256: str

    @property
    def body_name(self) -> str:
        """MuJoCo and URDF link name."""

        return f"{self.name}_link"

    @property
    def mesh_name(self) -> str:
        """MuJoCo visual-mesh asset name."""

        return f"{self.name}_visual_mesh"

    @property
    def visual_geom_name(self) -> str:
        """MuJoCo visual-only geom name."""

        return f"{self.name}_visual"

    def collision_mesh_name(self, index: int) -> str:
        """MuJoCo mesh asset name for one convex collision part."""

        return f"{self.name}_collision_mesh_{index:03d}"

    def collision_geom_name(self, index: int) -> str:
        """MuJoCo geom name for one convex collision part."""

        return f"{self.name}_collision_{index:03d}"

    @property
    def joint_name(self) -> str:
        """MuJoCo free-joint name."""

        return f"{self.name}_freejoint"


@dataclass(frozen=True)
class ObjectAssetValidation:
    """Geometry validation result for one catalog entry."""

    object_name: str
    vertex_count: int
    face_count: int
    extents: tuple[float, float, float]
    collision_part_count: int
    sha256_matches: bool


def default_models_root() -> Path:
    """Return the package-local model directory."""

    return Path(__file__).resolve().parents[1] / "models"


def default_generated_assets_root() -> Path:
    """Return a checkout-isolated temporary cache for library-level callers."""

    package_root = Path(__file__).resolve().parents[1]
    checkout_identity = hashlib.sha256(str(package_root).encode("utf-8")).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / "holosoma-retargeting" / checkout_identity / "generated-assets"


def get_omomo_object_asset(
    object_name: str,
    *,
    models_root: str | Path | None = None,
) -> OmomoObjectAsset:
    """Resolve one known OMOMO object from the package asset catalog."""

    if object_name not in OMOMO_OBJECT_NAMES:
        supported = ", ".join(OMOMO_OBJECT_NAMES)
        raise ValueError(f"Unknown OMOMO object {object_name!r}; supported objects: {supported}")
    root = Path(models_root).expanduser() if models_root is not None else default_models_root()
    object_dir = root / object_name
    collision_mesh_paths = tuple(sorted((object_dir / "collision").glob(f"{object_name}_collision_*.obj")))
    return OmomoObjectAsset(
        name=object_name,
        mesh_path=object_dir / f"{object_name}.obj",
        collision_mesh_paths=collision_mesh_paths,
        urdf_path=object_dir / f"{object_name}.urdf",
        mesh_sha256=OMOMO_MESH_SHA256[object_name],
    )


def get_all_omomo_object_assets(
    *,
    models_root: str | Path | None = None,
) -> tuple[OmomoObjectAsset, ...]:
    """Resolve all supported OMOMO objects in stable order."""

    return tuple(get_omomo_object_asset(name, models_root=models_root) for name in OMOMO_OBJECT_NAMES)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_omomo_object_asset(
    asset: OmomoObjectAsset,
    *,
    verify_hash: bool = True,
) -> ObjectAssetValidation:
    """Validate file presence, source hash, finite geometry, and meter-scale bounds."""

    if not asset.mesh_path.is_file():
        raise FileNotFoundError(f"OMOMO object mesh not found: {asset.mesh_path}")
    if not asset.urdf_path.is_file():
        raise FileNotFoundError(f"OMOMO object URDF not found: {asset.urdf_path}")
    if not asset.collision_mesh_paths:
        raise FileNotFoundError(f"OMOMO convex collision meshes not found: {asset.mesh_path.parent / 'collision'}")

    sha256_matches = _sha256(asset.mesh_path) == asset.mesh_sha256
    if verify_hash and not sha256_matches:
        raise ValueError(f"OMOMO object mesh checksum mismatch: {asset.mesh_path}")

    mesh = trimesh.load(asset.mesh_path, force="mesh", process=False)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4:
        raise ValueError(f"OMOMO object mesh has invalid vertices: {asset.mesh_path}")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) < 4:
        raise ValueError(f"OMOMO object mesh has invalid faces: {asset.mesh_path}")
    if not np.isfinite(vertices).all():
        raise ValueError(f"OMOMO object mesh contains NaN or Inf: {asset.mesh_path}")

    extents_array = np.asarray(mesh.bounds[1] - mesh.bounds[0], dtype=float)
    if np.any(extents_array <= 1e-3) or np.any(extents_array >= 5.0):
        raise ValueError(f"OMOMO object mesh extents are not plausible meters: {asset.name}={extents_array.tolist()}")

    for collision_path in asset.collision_mesh_paths:
        if not collision_path.is_file():
            raise FileNotFoundError(f"OMOMO collision mesh not found: {collision_path}")
        collision_mesh = trimesh.load(collision_path, force="mesh", process=False)
        collision_vertices = np.asarray(collision_mesh.vertices)
        collision_faces = np.asarray(collision_mesh.faces)
        if (
            collision_vertices.ndim != 2
            or collision_vertices.shape[1] != 3
            or len(collision_vertices) < 4
            or collision_faces.ndim != 2
            or collision_faces.shape[1] != 3
            or len(collision_faces) < 4
        ):
            raise ValueError(f"OMOMO collision mesh is invalid: {collision_path}")
        if not np.isfinite(collision_vertices).all():
            raise ValueError(f"OMOMO collision mesh contains NaN or Inf: {collision_path}")
        if not collision_mesh.is_watertight or not collision_mesh.is_convex:
            raise ValueError(f"OMOMO collision mesh must be a watertight convex hull: {collision_path}")

    return ObjectAssetValidation(
        object_name=asset.name,
        vertex_count=len(vertices),
        face_count=len(faces),
        extents=tuple(float(value) for value in extents_array),
        collision_part_count=len(asset.collision_mesh_paths),
        sha256_matches=sha256_matches,
    )


def validate_all_omomo_object_assets(
    *,
    models_root: str | Path | None = None,
    verify_hash: bool = True,
) -> tuple[ObjectAssetValidation, ...]:
    """Validate every OMOMO object asset."""

    return tuple(
        validate_omomo_object_asset(asset, verify_hash=verify_hash)
        for asset in get_all_omomo_object_assets(models_root=models_root)
    )


def _scale_text(scale: tuple[float, float, float]) -> str:
    values = np.asarray(scale, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError(f"Object scale must contain three positive finite values, got {scale}")
    encoded_values = []
    for value in values:
        text = repr(float(value))
        encoded_values.append(text[:-2] if text.endswith(".0") else text)
    return " ".join(encoded_values)


def _scale_filename_token(scale: tuple[float, float, float]) -> str:
    """Return a round-trip-safe filename token for one scale triple."""

    return "_".join(_scale_text(scale).split())


def _atomic_write_xml(tree: ET.ElementTree, destination: Path) -> None:
    """Idempotently publish XML without exposing a partial file to workers."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    tree.write(buffer, encoding="utf-8", xml_declaration=True)
    payload = buffer.getvalue()
    try:
        if destination.read_bytes() == payload:
            return
    except FileNotFoundError:
        pass
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(payload)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        try:
            if destination.read_bytes() == payload:
                return
        except FileNotFoundError:
            pass
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def create_omomo_object_scene(
    robot_xml_path: str | Path,
    object_name: str,
    *,
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    models_root: str | Path | None = None,
    output_dir: str | Path,
) -> Path:
    """Generate a robot-plus-free-object MuJoCo scene from a base robot XML."""

    robot_xml = Path(robot_xml_path).expanduser().resolve()
    if not robot_xml.is_file():
        raise FileNotFoundError(f"Robot MuJoCo XML not found: {robot_xml}")
    asset = get_omomo_object_asset(object_name, models_root=models_root)
    validate_omomo_object_asset(asset, verify_hash=False)
    scale_value = _scale_text(scale)
    destination_dir = Path(output_dir).expanduser().resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(robot_xml)  # noqa: S314
    root = tree.getroot()
    compiler = root.find("compiler")
    mesh_dir = robot_xml.parent
    if compiler is not None and compiler.get("meshdir"):
        mesh_dir = (robot_xml.parent / compiler.get("meshdir", "")).resolve()
        compiler.set("meshdir", Path(os.path.relpath(mesh_dir, destination_dir)).as_posix())

    asset_element = root.find("asset")
    worldbody = root.find("worldbody")
    if asset_element is None or worldbody is None:
        raise ValueError(f"Robot XML must contain top-level asset and worldbody elements: {robot_xml}")

    mesh_reference = Path(os.path.relpath(asset.mesh_path.resolve(), mesh_dir)).as_posix()
    ET.SubElement(
        asset_element,
        "mesh",
        {
            "name": asset.mesh_name,
            "file": mesh_reference,
            "scale": scale_value,
        },
    )
    for index, collision_mesh_path in enumerate(asset.collision_mesh_paths):
        collision_reference = Path(os.path.relpath(collision_mesh_path.resolve(), mesh_dir)).as_posix()
        ET.SubElement(
            asset_element,
            "mesh",
            {
                "name": asset.collision_mesh_name(index),
                "file": collision_reference,
                "scale": scale_value,
            },
        )

    body = ET.SubElement(worldbody, "body", {"name": asset.body_name})
    ET.SubElement(body, "freejoint", {"name": asset.joint_name})
    ET.SubElement(
        body,
        "inertial",
        {
            "pos": "0 0 0",
            "mass": "0.1",
            "diaginertia": "0.002 0.002 0.002",
        },
    )
    ET.SubElement(
        body,
        "geom",
        {
            "name": asset.visual_geom_name,
            "type": "mesh",
            "mesh": asset.mesh_name,
            "contype": "0",
            "conaffinity": "0",
            "pos": "0 0 0",
            "quat": "1 0 0 0",
            "rgba": "0.7 0.8 0.9 0.7",
        },
    )
    for index in range(len(asset.collision_mesh_paths)):
        ET.SubElement(
            body,
            "geom",
            {
                "name": asset.collision_geom_name(index),
                "type": "mesh",
                "mesh": asset.collision_mesh_name(index),
                "contype": "1",
                "conaffinity": "1",
                "pos": "0 0 0",
                "quat": "1 0 0 0",
                "rgba": "0 0 0 0",
                "friction": "0.9 0.5 0.5",
                "solref": "0.02 1",
                "solimp": "0.9 0.95 0.001",
            },
        )

    scale_suffix = ""
    if scale_value != "1 1 1":
        scale_suffix = f"_scaled_{_scale_filename_token(scale)}"
    destination = destination_dir / f"{robot_xml.stem}_w_{object_name}{scale_suffix}.xml"
    _atomic_write_xml(tree, destination)
    return destination


def create_scaled_omomo_object_urdf(
    object_name: str,
    scale: tuple[float, float, float],
    *,
    models_root: str | Path | None = None,
    output_dir: str | Path,
) -> Path:
    """Create a cached URDF whose visual and collision meshes use ``scale``."""

    asset = get_omomo_object_asset(object_name, models_root=models_root)
    validate_omomo_object_asset(asset, verify_hash=False)
    scale_value = _scale_text(scale)
    if scale_value == "1 1 1":
        return asset.urdf_path

    destination_dir = Path(output_dir).expanduser().resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / (f"{object_name}_scaled_{_scale_filename_token(scale)}.urdf")

    tree = ET.parse(asset.urdf_path)  # noqa: S314
    mesh_elements = tree.getroot().findall(".//mesh")
    if not mesh_elements:
        raise ValueError(f"OMOMO object URDF contains no mesh elements: {asset.urdf_path}")
    for mesh in mesh_elements:
        filename = mesh.get("filename")
        if not filename:
            raise ValueError(f"OMOMO object URDF mesh has no filename: {asset.urdf_path}")
        source_reference = Path(filename)
        if not source_reference.is_absolute():
            source_reference = (asset.urdf_path.parent / source_reference).resolve()
        mesh.set("filename", source_reference.as_posix())
        mesh.set("scale", scale_value)
    _atomic_write_xml(tree, destination)
    return destination
