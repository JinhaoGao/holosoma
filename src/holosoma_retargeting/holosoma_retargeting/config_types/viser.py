# ruff: noqa: CPY001
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

    qpos_npz: str | None = None
    """Legacy result path. Prefer ``input_path`` for new commands."""

    input_path: str | None = None
    """Canonical motion input path for result, raw-human, or converted data."""

    input_kind: Literal["auto", "result", "raw", "converted"] = "auto"
    """Input adapter. Auto inspects the path and NPZ fields."""

    robot_urdf: str | None = None
    """Robot URDF. If unset, resolved from metadata saved in the result NPZ."""

    robot_mujoco_xml: str | None = None
    """Optional MuJoCo XML path whose joint order matches qpos.
    If unset, the viewer tries the sibling .xml next to robot_urdf."""

    robot_type: str | None = None
    """Robot type used to resolve robot assets for legacy results."""

    data_format: str | None = None
    """Motion data format used for raw inputs or legacy result recovery."""

    task_type: Literal["auto", "robot_only", "object_interaction", "climbing"] = "auto"
    """Task context used by raw-human inputs."""

    sequence: str | None = None
    """Sequence stem when a raw motion input is a dataset directory."""

    human_height: float | None = None
    """Raw-human height override for source datasets without metadata."""

    source_mesh: str | None = None
    """Optional raw-human object or terrain mesh override."""

    object_name: str | None = None
    """Optional raw-human object category override."""

    port: int = 8080
    """Viser web-server port for raw-human inputs."""

    start_frame: int = 0
    """Initial frame for raw-human inputs."""

    playing: bool = True
    """Start raw-human or converted playback immediately."""

    velocity_scale: float = 0.1
    """Length scale for converted rigid-body velocity vectors."""

    dry_run: bool = False
    """Validate and summarize a raw-human input without starting Viser."""

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
    """Backward-compatible default for both human and robot mapped skeletons."""

    show_human_skeleton: bool | None = None
    """Whether to show the complete saved source-human skeleton."""

    show_robot_skeleton: bool | None = None
    """Whether to show the complete saved retargeted robot link skeleton."""

    show_human_hands: bool | None = None
    """Whether to show visual-only source-human finger details."""

    show_object_keypoints: bool = False
    """Whether to show saved demo/target object keypoints."""

    show_point_clouds: bool = False
    """Whether to show the compact human, robot, terrain, and object point clouds."""

    show_interaction_mesh: bool = False
    """Whether to show saved source/target interaction mesh overlays."""

    show_foot_sticking: bool = True
    """Whether to show the saved per-frame foot-sticking status panel."""

    show_source_orientation_axes: bool = False
    """Whether to show raw source-human global joint frames when available."""

    show_target_orientation_axes: bool = False
    """Whether to show saved target orientation frames when available."""

    show_robot_orientation_axes: bool = False
    """Whether to show saved robot orientation frames when available."""

    orientation_joints: tuple[str, ...] = ()
    """Human joint names whose target and robot orientation frames are shown."""

    orientation_scope: Literal["retargeting", "all"] = "retargeting"
    """Show only retargeting-related frames or every saved human/robot link frame."""

    orientation_axis_length: float = 0.065
    """Length of each orientation axis in scene units."""

    orientation_axis_shaft_radius: float = 0.005
    """Radius of each solid orientation arrow shaft."""

    orientation_axis_head_radius: float = 0.01
    """Radius of each solid orientation arrow head."""

    orientation_axis_head_length: float = 0.015
    """Length of each solid orientation arrow head."""

    show_joint_labels: bool = False
    """Whether to show source joint names when a complete source skeleton is available."""

    show_body_com: bool = True
    """Whether to show saved rigid-body center positions in converted motion inputs."""

    show_body_velocity: bool = True
    """Whether to show saved rigid-body velocity vectors in converted motion inputs."""

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

    point_cloud_point_size: float = 0.012
    """Point size for the compact retargeting point-cloud layer."""

    skeleton_line_width: float = 2.0
    """Line width for mapped skeleton edges."""
