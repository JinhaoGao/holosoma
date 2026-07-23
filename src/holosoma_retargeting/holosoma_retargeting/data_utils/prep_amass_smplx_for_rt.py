"""Convert raw AMASS SMPL-X sequences to the unified retargeting NPZ format."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from holosoma_retargeting.config_types.data_type import AMASS_DEMO_JOINTS
from holosoma_retargeting.data_utils.smplx_model import (
    compute_smplx_height,
    create_smplx_model,
    forward_smplx_model,
)
from scipy.spatial.transform import Rotation

DEFAULT_SMPLX_MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "smplx"


@dataclass(frozen=True)
class AMASSParameters:
    """Validated and resampled SMPL-X parameters from one AMASS sequence."""

    transl: torch.Tensor
    global_orient: torch.Tensor
    body_pose: torch.Tensor
    betas: torch.Tensor
    source_fps: float
    fps: float

    @property
    def num_frames(self) -> int:
        return int(self.body_pose.shape[0])


@dataclass(frozen=True)
class ConvertedAMASSMotion:
    global_joint_positions: np.ndarray
    root_quaternions_wxyz: np.ndarray
    height: float
    source_fps: float
    fps: float


def _finite_array(data: np.lib.npyio.NpzFile, key: str, dtype=np.float32) -> np.ndarray:
    if key not in data:
        raise KeyError(f"AMASS file is missing required field {key!r}")
    value = np.asarray(data[key], dtype=dtype)
    if not np.isfinite(value).all():
        raise ValueError(f"AMASS field {key!r} contains NaN or Inf")
    return value


def _resample_indices(num_frames: int, source_fps: float, target_fps: float) -> np.ndarray:
    if not np.isfinite(source_fps) or source_fps <= 0:
        raise ValueError(f"AMASS source FPS must be positive and finite, got {source_fps}")
    if not np.isfinite(target_fps) or target_fps <= 0 or target_fps > source_fps:
        raise ValueError(f"Target FPS must be in (0, {source_fps}], got {target_fps}")
    sample_times = np.arange(0.0, num_frames / source_fps, 1.0 / target_fps)
    return np.unique(np.minimum(np.rint(sample_times * source_fps).astype(np.int64), num_frames - 1))


def load_amass_parameters(input_file: Path | str, fps: float = 30.0) -> AMASSParameters:
    """Load one raw ``*_stageii.npz`` sequence and resample it."""

    input_path = Path(input_file).expanduser()
    if not input_path.is_file():
        raise FileNotFoundError(f"AMASS input file not found: {input_path}")

    with np.load(input_path, allow_pickle=False) as data:
        transl = _finite_array(data, "trans")
        poses = _finite_array(data, "poses")
        betas = _finite_array(data, "betas").reshape(-1)
        source_fps = float(np.asarray(data["mocap_frame_rate"]).item())

    if transl.ndim != 2 or transl.shape[1] != 3:
        raise ValueError(f"AMASS trans must have shape (T, 3), got {transl.shape}")
    if poses.ndim != 2 or poses.shape[0] != transl.shape[0] or poses.shape[1] < 66:
        raise ValueError(f"AMASS poses must have shape (T, >=66), got {poses.shape}")
    if transl.shape[0] == 0 or betas.size == 0:
        raise ValueError("AMASS sequence must contain frames and shape coefficients")

    indices = _resample_indices(transl.shape[0], source_fps, fps)
    frame_betas = np.broadcast_to(betas, (indices.size, betas.size)).copy()
    return AMASSParameters(
        transl=torch.from_numpy(transl[indices]),
        global_orient=torch.from_numpy(poses[indices, :3]),
        body_pose=torch.from_numpy(poses[indices, 3:66]),
        betas=torch.from_numpy(frame_betas),
        source_fps=source_fps,
        fps=float(fps),
    )


def convert_amass_parameters(
    parameters: AMASSParameters,
    model_path: Path | str,
    batch_size: int = 128,
) -> ConvertedAMASSMotion:
    """Run batched SMPL-X FK and preserve AMASS's world Z-up frame."""

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    body_model = create_smplx_model(model_path, num_betas=parameters.betas.shape[1])
    joint_batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, parameters.num_frames, batch_size):
            end = min(parameters.num_frames, start + batch_size)
            output = forward_smplx_model(
                body_model,
                betas=parameters.betas[start:end],
                global_orient=parameters.global_orient[start:end],
                body_pose=parameters.body_pose[start:end],
                transl=parameters.transl[start:end],
            )
            joint_batches.append(output.joints[:, : len(AMASS_DEMO_JOINTS)].cpu().numpy())
        height = compute_smplx_height(body_model, parameters.betas)

    joints = np.concatenate(joint_batches, axis=0).astype(np.float32, copy=False)
    root_quaternions = Rotation.from_rotvec(parameters.global_orient.numpy()).as_quat(scalar_first=True)
    if not np.isfinite(height) or height <= 0:
        raise ValueError(f"Computed invalid SMPL-X height: {height}")
    return ConvertedAMASSMotion(
        global_joint_positions=joints,
        root_quaternions_wxyz=root_quaternions.astype(np.float32),
        height=height,
        source_fps=parameters.source_fps,
        fps=parameters.fps,
    )


def save_converted_amass(
    motion: ConvertedAMASSMotion,
    output_file: Path | str,
    source_file: Path | str,
    overwrite: bool = False,
) -> Path:
    output_path = Path(output_file).expanduser()
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        global_joint_positions=motion.global_joint_positions,
        height=np.float32(motion.height),
        source_fps=np.float32(motion.source_fps),
        fps=np.float32(motion.fps),
        joint_names=np.asarray(AMASS_DEMO_JOINTS, dtype=str),
        root_quaternions_wxyz=motion.root_quaternions_wxyz,
        source_format=np.asarray("amass"),
        coordinate_system=np.asarray("right_handed_z_up"),
        source_file=np.asarray(str(Path(source_file).expanduser().resolve())),
    )
    return output_path


def convert_amass_file(
    input_file: Path | str,
    output_file: Path | str,
    model_path: Path | str = DEFAULT_SMPLX_MODEL_DIR,
    fps: float = 30.0,
    batch_size: int = 128,
    overwrite: bool = False,
) -> Path:
    parameters = load_amass_parameters(input_file, fps=fps)
    motion = convert_amass_parameters(parameters, model_path=model_path, batch_size=batch_size)
    return save_converted_amass(motion, output_file, input_file, overwrite=overwrite)


def get_amass_files(amass_root: Path, subdataset: str | None = None) -> list[Path]:
    search_root = amass_root / subdataset if subdataset else amass_root
    if not search_root.is_dir():
        raise FileNotFoundError(f"AMASS data directory not found: {search_root}")
    return sorted(search_root.rglob("*_stageii.npz"))


def output_name(input_file: Path, amass_root: Path) -> str:
    relative = input_file.relative_to(amass_root)
    return "_".join(relative.parts)


@dataclass
class Config:
    amass_root_folder: Path
    """Root containing raw AMASS ``*_stageii.npz`` files."""

    output_folder: Path
    """Directory for converted retargeting NPZ files."""

    model_root_folder: Path = DEFAULT_SMPLX_MODEL_DIR
    """SMPL-X model file, ``smplx`` directory, or its parent."""

    subdataset_folder: str | None = None
    """Optional AMASS subdataset directory."""

    fps: float = 30.0
    """Output FPS."""

    batch_size: int = 128
    """Frames per SMPL-X forward pass."""

    overwrite: bool = False
    """Replace converted files that already exist."""


def main(config: Config) -> None:
    files = get_amass_files(config.amass_root_folder, config.subdataset_folder)
    if not files:
        raise FileNotFoundError(f"No *_stageii.npz files found under {config.amass_root_folder}")

    converted = 0
    skipped = 0
    for input_file in files:
        destination = config.output_folder / output_name(input_file, config.amass_root_folder)
        if destination.exists() and not config.overwrite:
            skipped += 1
            continue
        convert_amass_file(
            input_file,
            destination,
            model_path=config.model_root_folder,
            fps=config.fps,
            batch_size=config.batch_size,
            overwrite=config.overwrite,
        )
        converted += 1
        print(f"Converted: {input_file} -> {destination}")
    print(f"AMASS conversion complete: converted={converted}, skipped={skipped}")


if __name__ == "__main__":
    main(tyro.cli(Config))
