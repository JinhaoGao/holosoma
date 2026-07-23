"""Shared SMPL-X model helpers for human-motion converters."""

from __future__ import annotations

from pathlib import Path

import smplx
import torch


def resolve_smplx_model_root(model_path: Path | str, gender: str = "neutral") -> Path:
    """Resolve a model file, ``smplx`` directory, or its parent for ``smplx.create``."""

    path = Path(model_path).expanduser().resolve()
    model_filename = f"SMPLX_{gender.upper()}.npz"
    if path.is_file():
        if path.name.upper() != model_filename.upper():
            raise ValueError(f"Expected {model_filename}, got: {path}")
        return path
    if (path / model_filename).is_file():
        return path.parent
    if (path / "smplx" / model_filename).is_file():
        return path
    raise FileNotFoundError(f"Could not find {model_filename} under {path} or {path / 'smplx'}")


def create_smplx_model(
    model_path: Path | str,
    num_betas: int,
    gender: str = "neutral",
):
    """Create a full-pose SMPL-X model with an explicit shape dimension."""

    model_root = resolve_smplx_model_root(model_path, gender)
    return smplx.create(
        str(model_root),
        model_type="smplx",
        gender=gender,
        use_pca=False,
        num_betas=num_betas,
    )


def forward_smplx_model(
    body_model,
    *,
    betas: torch.Tensor,
    global_orient: torch.Tensor,
    body_pose: torch.Tensor,
    transl: torch.Tensor,
):
    """Run SMPL-X FK with neutral face and hand poses."""

    batch_size = int(body_pose.shape[0])
    zeros_3 = torch.zeros(batch_size, 3, dtype=body_pose.dtype, device=body_pose.device)
    return body_model(
        betas=betas,
        global_orient=global_orient,
        body_pose=body_pose,
        transl=transl,
        left_hand_pose=torch.zeros(batch_size, 45, dtype=body_pose.dtype, device=body_pose.device),
        right_hand_pose=torch.zeros(batch_size, 45, dtype=body_pose.dtype, device=body_pose.device),
        jaw_pose=zeros_3,
        leye_pose=zeros_3,
        reye_pose=zeros_3,
        expression=torch.zeros(batch_size, 10, dtype=body_pose.dtype, device=body_pose.device),
    )


def compute_smplx_height(body_model, betas: torch.Tensor) -> float:
    """Measure subject height from a zero-pose SMPL-X mesh."""

    representative_betas = betas.median(dim=0).values[None]
    zeros_3 = torch.zeros(1, 3, dtype=betas.dtype, device=betas.device)
    output = forward_smplx_model(
        body_model,
        betas=representative_betas,
        global_orient=zeros_3,
        body_pose=torch.zeros(1, 63, dtype=betas.dtype, device=betas.device),
        transl=zeros_3,
    )
    vertical = output.vertices[0, :, 1]
    return float((vertical.max() - vertical.min()).item())
