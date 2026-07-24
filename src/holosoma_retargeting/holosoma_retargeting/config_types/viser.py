"""Configuration types for viser visualization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ViserConfig:
    """Configuration for viser player visualization.

    This follows the pattern from holosoma's config_types.
    Uses a flat structure with default values.
    """

    qpos_npz: str = "rt_results/OMOMO_new/box_parallel/sub8_largebox_051_original.npz"
    """Path to .npz file with qpos data."""

    robot_urdf: str | None = None
    """Robot URDF. If unset, resolved from metadata saved in the result NPZ."""

    robot_mujoco_xml: str | None = None
    """Optional MuJoCo XML path whose joint order matches qpos.
    If unset, the viewer tries the sibling .xml next to robot_urdf."""

    robot_type: str | None = None
    """Robot type used to resolve mapped skeleton links. If unset, inferred from robot_urdf parent directory."""

    data_format: str | None = None
    """Motion data format used to resolve mapped skeleton joints."""

    object_urdf: str | None = None
    """Path to object URDF file (optional)."""

    fps: int = 30
    """Frames per second for playback."""

    assume_object_in_qpos: bool | None = None
    """Whether qpos includes object pose. If unset, resolved from result metadata."""

    loop: bool = False
    """Whether to loop playback."""

    show_meshes: bool = True
    """Whether to show mesh visualizations."""

    show_robot_mesh: bool | None = None
    """Whether to show robot mesh visualization. If unset, uses show_meshes."""

    show_object_mesh: bool | None = None
    """Whether to show object mesh visualization. If unset, uses show_meshes."""

    mesh_opacity: float = 1.0
    """Default opacity for robot/object meshes. Values below 1.0 use a neutral translucent mesh color."""

    robot_mesh_opacity: float | None = None
    """Opacity for robot mesh. If unset, uses mesh_opacity."""

    object_mesh_opacity: float | None = None
    """Opacity for object mesh. If unset, uses mesh_opacity."""

    show_mapped_skeletons: bool = False
    """Whether to show saved human mapped skeleton and retargeted robot mapped skeleton."""

    show_object_keypoints: bool = False
    """Whether to show saved demo/target object keypoints."""

    show_interaction_mesh: bool = False
    """Whether to show saved source/target interaction mesh overlays."""

    interaction_mesh_mode: Literal["source", "target", "both"] = "both"
    """Which saved interaction mesh to show: source human-object, target robot-object, or both."""

    interaction_mesh_edges: Literal["all", "cross"] = "cross"
    """Which saved interaction mesh edges to draw. 'cross' keeps only human/robot-to-object edges."""

    interaction_mesh_line_width: float = 1.0
    """Line width for saved interaction mesh edges."""

    grid_width: float = 8.0
    """Grid width for visualization."""

    grid_height: float = 8.0
    """Grid height for visualization."""

    visual_fps_multiplier: int = 2
    """Visual FPS multiplier for interpolation."""

    min_fps: int = 1
    """Minimum FPS setting."""

    max_fps: int = 240
    """Maximum FPS setting."""

    min_interp_mult: int = 1
    """Minimum interpolation multiplier."""

    max_interp_mult: int = 8
    """Maximum interpolation multiplier."""

    skeleton_point_radius: float = 0.02
    """Radius for mapped skeleton keypoint spheres."""

    object_keypoint_radius: float = 0.02
    """Radius for saved demo/target object keypoint spheres."""

    skeleton_line_width: float = 2.0
    """Line width for mapped skeleton edges."""
