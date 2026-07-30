# ruff: noqa: CPY001, PLR0917
"""Convert GVHMR world-frame SMPL-X predictions to Holosoma joint data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro
from holosoma_retargeting.config_types.data_type import SMPLX_DEMO_JOINTS
from holosoma_retargeting.data_utils.smplx_model import (
    compute_smplx_height,
    create_smplx_model,
    forward_smplx_model,
)
from scipy.spatial.transform import Rotation

GVHMR_TO_Z_UP = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)
DEFAULT_SMPLX_MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "smplx"


@dataclass(frozen=True)
class GVHMRParameters:
    """Validated world-frame parameters stored by GVHMR."""

    body_pose: torch.Tensor
    betas: torch.Tensor
    global_orient: torch.Tensor
    transl: torch.Tensor

    @property
    def num_frames(self) -> int:
        return int(self.body_pose.shape[0])

    @property
    def num_betas(self) -> int:
        return int(self.betas.shape[1])


@dataclass(frozen=True)
class ConvertedGVHMRMotion:
    """Holosoma-compatible representation produced from GVHMR."""

    global_joint_positions: np.ndarray
    root_quaternions_wxyz: np.ndarray
    orientation_joint_names: tuple[str, ...]
    orientation_quaternions_wxyz: np.ndarray
    orientation_source: str
    height: float
    fps: float


def _as_cpu_float_tensor(value: Any, field_name: str) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"GVHMR field '{field_name}' must be a torch.Tensor, got {type(value).__name__}")
    tensor = value.detach().to(device="cpu", dtype=torch.float32)
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"GVHMR field '{field_name}' contains NaN or Inf")
    return tensor


def load_gvhmr_parameters(input_file: Path | str) -> GVHMRParameters:
    """Safely load and validate ``smpl_params_global`` from a GVHMR result."""

    input_path = Path(input_file).expanduser()
    if not input_path.is_file():
        raise FileNotFoundError(f"GVHMR result file not found: {input_path}")

    prediction = torch.load(input_path, map_location="cpu", weights_only=True)
    if not isinstance(prediction, dict):
        raise TypeError(f"GVHMR result must contain a dictionary, got {type(prediction).__name__}")

    params = prediction.get("smpl_params_global")
    if not isinstance(params, dict):
        raise KeyError("GVHMR result is missing dictionary key 'smpl_params_global'")

    required = ("body_pose", "betas", "global_orient", "transl")
    missing = [key for key in required if key not in params]
    if missing:
        raise KeyError(f"GVHMR smpl_params_global is missing fields: {missing}")

    body_pose = _as_cpu_float_tensor(params["body_pose"], "body_pose")
    betas = _as_cpu_float_tensor(params["betas"], "betas")
    global_orient = _as_cpu_float_tensor(params["global_orient"], "global_orient")
    transl = _as_cpu_float_tensor(params["transl"], "transl")

    if body_pose.ndim != 2 or body_pose.shape[1] != 63:
        raise ValueError(f"GVHMR body_pose must have shape (T, 63), got {tuple(body_pose.shape)}")
    num_frames = int(body_pose.shape[0])
    if num_frames == 0:
        raise ValueError("GVHMR motion contains no frames")
    for name, tensor in (("global_orient", global_orient), ("transl", transl)):
        if tensor.shape != (num_frames, 3):
            raise ValueError(f"GVHMR {name} must have shape ({num_frames}, 3), got {tuple(tensor.shape)}")

    if betas.ndim == 1:
        betas = betas.unsqueeze(0)
    if betas.ndim != 2 or betas.shape[1] == 0:
        raise ValueError(f"GVHMR betas must have shape (B,) or (T, B), got {tuple(betas.shape)}")
    if betas.shape[0] == 1:
        betas = betas.expand(num_frames, -1).clone()
    elif betas.shape[0] != num_frames:
        raise ValueError(f"GVHMR betas must have one row or {num_frames} rows, got {tuple(betas.shape)}")

    return GVHMRParameters(
        body_pose=body_pose,
        betas=betas,
        global_orient=global_orient,
        transl=transl,
    )


def transform_gvhmr_points_to_z_up(points: np.ndarray) -> np.ndarray:
    """Rotate GVHMR Y-up points into Holosoma's right-handed Z-up frame."""

    points = np.asarray(points, dtype=np.float32)
    if points.shape[-1] != 3:
        raise ValueError(f"Expected points ending in dimension 3, got {points.shape}")
    return points @ GVHMR_TO_Z_UP.T


def _smplx_global_joint_rotation_matrices(
    global_orient: torch.Tensor,
    body_pose: torch.Tensor,
    parents: torch.Tensor | np.ndarray,
) -> np.ndarray:
    """Compose direct SMPL-X local rotations along the model's kinematic tree."""

    joint_count = len(SMPLX_DEMO_JOINTS)
    root_rotvec = global_orient.detach().cpu().numpy().astype(np.float64, copy=False)
    body_rotvec = body_pose.detach().cpu().numpy().astype(np.float64, copy=False)
    if root_rotvec.ndim != 2 or root_rotvec.shape[1] != 3:
        raise ValueError(f"SMPL-X global_orient must have shape (T, 3), got {root_rotvec.shape}")
    if body_rotvec.shape != (root_rotvec.shape[0], (joint_count - 1) * 3):
        raise ValueError(
            "SMPL-X body_pose must contain one local axis-angle rotation for each "
            f"non-root joint, got {body_rotvec.shape}"
        )

    parent_indices = np.asarray(
        parents.detach().cpu().numpy() if torch.is_tensor(parents) else parents,
        dtype=np.int64,
    ).reshape(-1)
    if parent_indices.size < joint_count:
        raise ValueError(
            f"SMPL-X parent hierarchy must contain at least {joint_count} joints, got {parent_indices.size}"
        )
    parent_indices = parent_indices[:joint_count]
    if parent_indices[0] != -1:
        raise ValueError(f"SMPL-X root parent must be -1, got {parent_indices[0]}")

    local_rotvec = np.concatenate(
        [root_rotvec[:, None, :], body_rotvec.reshape(root_rotvec.shape[0], joint_count - 1, 3)],
        axis=1,
    )
    local_matrices = (
        Rotation.from_rotvec(local_rotvec.reshape(-1, 3))
        .as_matrix()
        .reshape(
            root_rotvec.shape[0],
            joint_count,
            3,
            3,
        )
    )
    global_matrices = np.empty_like(local_matrices)
    global_matrices[:, 0] = local_matrices[:, 0]
    for joint_idx in range(1, joint_count):
        parent_idx = int(parent_indices[joint_idx])
        if parent_idx < 0 or parent_idx >= joint_idx:
            raise ValueError(
                f"SMPL-X parent hierarchy must be topologically ordered; joint {joint_idx} has parent {parent_idx}"
            )
        global_matrices[:, joint_idx] = global_matrices[:, parent_idx] @ local_matrices[:, joint_idx]
    return global_matrices


def _transform_gvhmr_global_orientations_to_z_up(
    source_global_matrices: np.ndarray,
) -> np.ndarray:
    """Rotate source-world global joint frames into Holosoma's Z-up world."""

    source_global_matrices = np.asarray(source_global_matrices, dtype=np.float64)
    if source_global_matrices.ndim != 4 or source_global_matrices.shape[-2:] != (3, 3):
        raise ValueError(
            f"Expected global joint rotation matrices with shape (T, J, 3, 3), got {source_global_matrices.shape}"
        )
    target_global_matrices = GVHMR_TO_Z_UP.astype(np.float64)[None, None] @ source_global_matrices
    return (
        Rotation.from_matrix(target_global_matrices.reshape(-1, 3, 3))
        .as_quat(scalar_first=True)
        .reshape(*target_global_matrices.shape[:2], 4)
        .astype(np.float32)
    )


def transform_gvhmr_root_orientations(global_orient: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert GVHMR axis-angle root rotations to Z-up ``wxyz`` quaternions."""

    orientations = np.asarray(
        global_orient.detach().cpu().numpy() if torch.is_tensor(global_orient) else global_orient,
        dtype=np.float64,
    )
    if orientations.ndim != 2 or orientations.shape[1] != 3:
        raise ValueError(f"Expected root orientations with shape (T, 3), got {orientations.shape}")
    source_rotation = Rotation.from_rotvec(orientations).as_matrix()[:, None]
    return _transform_gvhmr_global_orientations_to_z_up(source_rotation)[:, 0]


def _make_body_model(model_path: Path | str, num_betas: int):
    return create_smplx_model(model_path, num_betas=num_betas)


def _forward_body_model(body_model, parameters: GVHMRParameters, start: int, end: int):
    return forward_smplx_model(
        body_model,
        betas=parameters.betas[start:end],
        global_orient=parameters.global_orient[start:end],
        body_pose=parameters.body_pose[start:end],
        transl=parameters.transl[start:end],
    )


def convert_gvhmr_parameters(
    parameters: GVHMRParameters,
    model_path: Path | str,
    fps: float = 30.0,
    batch_size: int = 128,
) -> ConvertedGVHMRMotion:
    """Run SMPL-X FK and return Holosoma-compatible world-frame motion."""

    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"fps must be positive and finite, got {fps}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    body_model = _make_body_model(model_path, parameters.num_betas)
    joint_batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, parameters.num_frames, batch_size):
            end = min(parameters.num_frames, start + batch_size)
            output = _forward_body_model(body_model, parameters, start, end)
            joint_batches.append(output.joints[:, : len(SMPLX_DEMO_JOINTS)].cpu().numpy())

        height = compute_smplx_height(body_model, parameters.betas)

    source_joints = np.concatenate(joint_batches, axis=0).astype(np.float32, copy=False)
    global_joint_positions = transform_gvhmr_points_to_z_up(source_joints)
    source_global_orientations = _smplx_global_joint_rotation_matrices(
        parameters.global_orient,
        parameters.body_pose,
        body_model.parents,
    )
    orientation_quaternions = _transform_gvhmr_global_orientations_to_z_up(source_global_orientations)
    if not np.isfinite(height) or height <= 0:
        raise ValueError(f"Computed invalid SMPL-X height: {height}")

    return ConvertedGVHMRMotion(
        global_joint_positions=global_joint_positions,
        root_quaternions_wxyz=orientation_quaternions[:, 0].copy(),
        orientation_joint_names=tuple(SMPLX_DEMO_JOINTS),
        orientation_quaternions_wxyz=orientation_quaternions,
        orientation_source="direct_local_rotation_fk",
        height=height,
        fps=float(fps),
    )


def save_converted_motion(
    motion: ConvertedGVHMRMotion,
    output_file: Path | str,
    source_file: Path | str,
    overwrite: bool = False,
) -> Path:
    """Save converted motion using Holosoma's standard NPZ contract."""

    output_path = Path(output_file).expanduser()
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists (pass --overwrite to replace it): {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        global_joint_positions=motion.global_joint_positions,
        height=np.float32(motion.height),
        fps=np.float32(motion.fps),
        joint_names=np.asarray(SMPLX_DEMO_JOINTS, dtype=str),
        root_quaternions_wxyz=motion.root_quaternions_wxyz,
        orientation_joint_names=np.asarray(motion.orientation_joint_names, dtype=str),
        orientation_quaternions_wxyz=motion.orientation_quaternions_wxyz,
        orientation_source=np.asarray(motion.orientation_source),
        quaternion_convention=np.asarray("wxyz"),
        orientation_coordinate_system=np.asarray("right_handed_z_up"),
        source_format=np.asarray("gvhmr"),
        source_coordinate_system=np.asarray("right_handed_y_up"),
        coordinate_system=np.asarray("right_handed_z_up"),
        source_file=np.asarray(str(Path(source_file).expanduser().resolve())),
    )
    return output_path


def convert_gvhmr_file(
    input_file: Path | str,
    output_file: Path | str,
    model_path: Path | str = DEFAULT_SMPLX_MODEL_DIR,
    fps: float = 30.0,
    batch_size: int = 128,
    overwrite: bool = False,
) -> Path:
    """Convert one canonical ``hmr4d_results.pt`` file to NPZ."""

    parameters = load_gvhmr_parameters(input_file)
    motion = convert_gvhmr_parameters(parameters, model_path, fps=fps, batch_size=batch_size)
    return save_converted_motion(motion, output_file, input_file, overwrite=overwrite)


@dataclass
class Config:
    input_file: Path
    """Path to GVHMR's hmr4d_results.pt."""

    output_file: Path
    """Destination Holosoma NPZ file."""

    model_path: Path = DEFAULT_SMPLX_MODEL_DIR
    """SMPL-X model directory, its parent directory, or SMPLX_NEUTRAL.npz."""

    fps: float = 30.0
    """Source/output FPS. Canonical GVHMR demos run at 30 FPS."""

    batch_size: int = 128
    """Frames per SMPL-X forward pass."""

    overwrite: bool = False
    """Allow replacing an existing output file."""


def main(config: Config) -> None:
    output_path = convert_gvhmr_file(
        input_file=config.input_file,
        output_file=config.output_file,
        model_path=config.model_path,
        fps=config.fps,
        batch_size=config.batch_size,
        overwrite=config.overwrite,
    )
    print(f"Saved converted GVHMR motion to: {output_path}")


if __name__ == "__main__":
    main(tyro.cli(Config))
