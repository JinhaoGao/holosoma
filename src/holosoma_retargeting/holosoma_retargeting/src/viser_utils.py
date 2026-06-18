# viser_utils.py
from __future__ import annotations

import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

import numpy as np
import viser  # type: ignore[import-not-found]
from viser.extras import ViserUrdf  # type: ignore[import-not-found]


def actuated_joint_names_from_urdf(urdf_path: str | Path) -> list[str]:
    """Return non-fixed URDF joint names in file order."""
    root = ET.parse(urdf_path).getroot()
    names: list[str] = []
    for joint in root.findall("joint"):
        if joint.attrib.get("type") == "fixed":
            continue
        name = joint.attrib.get("name")
        if name:
            names.append(name)
    return names


def actuated_joint_names_from_mujoco_xml(xml_path: str | Path) -> list[str]:
    """Return non-free MuJoCo joint names in XML traversal order."""
    root = ET.parse(xml_path).getroot()
    names: list[str] = []
    for joint in root.findall(".//joint"):
        if joint.attrib.get("type") == "free":
            continue
        name = joint.attrib.get("name")
        if name:
            names.append(name)
    return names


def infer_mujoco_xml_path(robot_urdf: str | Path) -> Path | None:
    """Infer the sibling MuJoCo XML path used by retargeted qpos files."""
    xml_path = Path(robot_urdf).with_suffix(".xml")
    return xml_path if xml_path.exists() else None


def build_joint_order_indices(source_joint_names: Sequence[str], target_joint_names: Sequence[str]) -> np.ndarray:
    """Map source joint values into target joint order.

    The returned array is indexed by target order. For example,
    ``source_values[indices]`` produces values aligned with ``target_joint_names``.
    """
    source_index = {name: idx for idx, name in enumerate(source_joint_names)}
    missing = [name for name in target_joint_names if name not in source_index]
    if missing:
        raise ValueError(
            "Cannot build joint order mapping; missing joints in source order: "
            + ", ".join(missing)
        )
    return np.asarray([source_index[name] for name in target_joint_names], dtype=int)


def register_keyboard_shortcut(
    server: viser.ViserServer,
    label: str,
    *,
    hotkey: str,
    callback: Callable[[], None],
    description: str | None = None,
    modifier: str | None = None,
):
    """Register a command-palette action with an optional keyboard shortcut."""
    add_command = getattr(server.gui, "add_command", None)
    if add_command is None:
        return None

    command = add_command(
        label,
        description=description,
        hotkey=hotkey,
        modifier=modifier,
    )

    @command.on_trigger
    def _(_evt):
        callback()

    return command


def create_motion_control_sliders(
    server: viser.ViserServer,
    viser_robot: ViserUrdf,
    robot_base_frame: viser.FrameHandle,
    motion_sequence: np.ndarray,
    *,
    robot_dof: int,
    viser_object: ViserUrdf | None = None,
    object_base_frame: viser.FrameHandle | None = None,
    contains_object_in_qpos: bool = True,
    initial_fps: int = 30,
    initial_interp_mult: int = 2,
    loop: bool = True,
    qpos_to_viser_joint_indices: Sequence[int] | np.ndarray | None = None,
    on_frame: Callable[[np.ndarray, float], None] | None = None,
) -> Tuple[List[viser.GuiInputHandle[int]], List[float]]:
    """
    Create a slider + play/pause controls and a background player thread with smooth, slerp-based interpolation.

    Assumed qpos layout per frame (MuJoCo order):
        [0:3]   robot base position   (xyz)
        [3:7]   robot base quaternion (wxyz)
        [7:7+R] robot joints          (R = robot_dof)
        [-7:-4] object position  (xyz)            # only if contains_object_in_qpos and viser_object provided
        [-4:]   object quaternion (wxyz)          # only if contains_object_in_qpos and viser_object provided

    Args:
        server: Viser server.
        viser_robot: ViserUrdf for the robot.
        robot_base_frame: server.scene.add_frame(...) return for the robot root frame (we set wxyz/position here).
        motion_sequence: np.ndarray with shape [T, D], sequence of qpos frames.
        robot_dof: number of actuated joints expected by viser_robot.
        viser_object: optional ViserUrdf for an object.
        object_base_frame: optional frame handle for the object root.
        contains_object_in_qpos: set True if motion_sequence includes the object 7D pose at the end.
        initial_fps: base FPS for playback.
        initial_interp_mult: visual upsampling multiplier.
        loop: whether to wrap around at the end.
        qpos_to_viser_joint_indices: Optional mapping from qpos joint order
            to the order expected by ``viser_robot.update_cfg``.
        on_frame: Optional callback invoked after applying each frame. Receives
            the qpos used for rendering and the fractional source frame index.

    Returns:
        (controls, initial_values) — currently returns the [frame_slider] and [0.0]
    """
    qpos = motion_sequence
    n_frames = int(qpos.shape[0])
    if n_frames == 0:
        raise ValueError("motion_sequence is empty.")

    joint_order_indices = None
    if qpos_to_viser_joint_indices is not None:
        joint_order_indices = np.asarray(qpos_to_viser_joint_indices, dtype=int)
        if joint_order_indices.shape != (robot_dof,):
            raise ValueError(
                "qpos_to_viser_joint_indices must have shape "
                f"({robot_dof},), got {joint_order_indices.shape}"
            )

    has_object_input = (
        viser_object is not None
        and object_base_frame is not None
        and contains_object_in_qpos
        and qpos.shape[1] >= (7 + robot_dof + 7)
    )

    # ---------------- GUI ----------------
    with server.gui.add_folder("Playback"):
        frame_slider = server.gui.add_slider("Frame", min=0, max=max(0, n_frames - 1), step=1, initial_value=0)
        play_btn = server.gui.add_button("Play / Pause")
        prev_frame_btn = server.gui.add_button("Previous Frame")
        next_frame_btn = server.gui.add_button("Next Frame")
        fps_in = server.gui.add_number("FPS", initial_value=int(initial_fps), min=1, max=240, step=1)
    with server.gui.add_folder("Smoothing"):
        interp_mult_in = server.gui.add_number(
            "Visual FPS multiplier", initial_value=int(initial_interp_mult), min=1, max=8, step=1
        )

    # ---------------- helpers ----------------
    def _quat_normalize(q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, float)
        n = float(np.linalg.norm(q))
        return q if n == 0.0 else q / n

    def _quat_continuous(prev_q: np.ndarray | None, curr_q: np.ndarray) -> np.ndarray:
        q = _quat_normalize(curr_q)
        if prev_q is None:
            return q
        return -q if float(np.dot(prev_q, q)) < 0.0 else q

    def _slerp(q0: np.ndarray, q1: np.ndarray, u: float) -> np.ndarray:
        q0 = _quat_normalize(q0)
        q1 = _quat_normalize(q1)
        dot = float(np.dot(q0, q1))
        if dot < 0.0:
            q1 = -q1
            dot = -dot
        if dot > 0.9995:
            q = q0 + u * (q1 - q0)
            return _quat_normalize(q)
        theta = np.arccos(np.clip(dot, -1.0, 1.0))
        s = np.sin(theta)
        return (np.sin((1.0 - u) * theta) * q0 + np.sin(u * theta) * q1) / s

    def _interp_frame(qpos_arr: np.ndarray, i0: int, i1: int, u: float) -> np.ndarray:
        """SLERP for base & (optional) object quats; linear for positions and joints."""
        q0 = qpos_arr[i0]
        q1 = qpos_arr[i1]
        out = q0.copy()

        # Robot base (MuJoCo order: pos first, then quat)
        out[0:3] = (1.0 - u) * q0[0:3] + u * q1[0:3]  # pos (xyz)
        out[3:7] = _slerp(q0[3:7], q1[3:7], u)  # quat (wxyz)

        # Joints
        j0 = q0[7 : 7 + robot_dof]
        j1 = q1[7 : 7 + robot_dof]
        out[7 : 7 + robot_dof] = (1.0 - u) * j0 + u * j1

        # Object (optional) (MuJoCo order: pos first, then quat)
        if has_object_input:
            out[-7:-4] = (1.0 - u) * q0[-7:-4] + u * q1[-7:-4]  # obj pos (xyz)
            out[-4:] = _slerp(q0[-4:], q1[-4:], u)  # obj quat (wxyz)
        return out

    def _robot_joints_for_viser(q: np.ndarray) -> np.ndarray:
        joints = q[7 : 7 + robot_dof]
        if joint_order_indices is not None:
            required_len = int(joint_order_indices.max()) + 1
            raw_joints = q[7 : 7 + required_len]
            if raw_joints.shape[0] < required_len:
                raise ValueError(
                    "qpos frame is too short for qpos_to_viser_joint_indices: "
                    f"need {required_len} joints, got {raw_joints.shape[0]}"
                )
            return raw_joints[joint_order_indices]

        if joints.shape[0] != robot_dof:
            joints = (
                joints[:robot_dof] if joints.shape[0] > robot_dof else np.pad(joints, (0, robot_dof - joints.shape[0]))
            )
        return joints

    # ---------------- state ----------------
    playing = {"flag": False}
    tick = {"next": time.perf_counter()}  # absolute time for next draw
    prev: dict[str, np.ndarray | None] = {"robot_q": None, "obj_q": None}  # for continuity
    nonlocal_f = {"f": float(frame_slider.value)}  # fractional frame cursor
    updating_programmatically = {"flag": False}  # flag to prevent callback from pausing during programmatic updates

    # ---------------- draw ----------------
    def _apply_frame_from_q(q: np.ndarray, frame_float: float) -> None:
        viser_robot.update_cfg(_robot_joints_for_viser(q))

        # robot base (MuJoCo order: pos first, then quat)
        robot_base_frame.position = q[0:3]  # pos (xyz)
        r_q = _quat_continuous(prev["robot_q"], q[3:7])
        prev["robot_q"] = r_q
        robot_base_frame.wxyz = r_q

        # object (optional) (MuJoCo order: pos first, then quat)
        if has_object_input and object_base_frame is not None:
            object_base_frame.position = q[-7:-4]  # obj pos (xyz)
            o_q = _quat_continuous(prev["obj_q"], q[-4:])
            prev["obj_q"] = o_q
            object_base_frame.wxyz = o_q
        elif object_base_frame is not None and viser_object is not None:
            # fallback static pose
            object_base_frame.position = np.zeros(3)
            object_base_frame.wxyz = np.array([1.0, 0.0, 0.0, 0.0])

        if on_frame is not None:
            on_frame(q, frame_float)

    def _apply_discrete_frame(i: int) -> None:
        i = int(np.clip(i, 0, n_frames - 1))
        _apply_frame_from_q(qpos[i], float(i))

    def _set_discrete_frame(i: int) -> None:
        frame_idx = int(np.clip(i, 0, n_frames - 1))
        playing["flag"] = False
        tick["next"] = time.perf_counter()
        prev["robot_q"] = None
        prev["obj_q"] = None
        nonlocal_f["f"] = float(frame_idx)

        updating_programmatically["flag"] = True
        try:
            frame_slider.value = frame_idx
        finally:
            updating_programmatically["flag"] = False
        _apply_discrete_frame(frame_idx)

    # ---------------- controls ----------------
    def _toggle_playback() -> None:
        playing["flag"] = not playing["flag"]
        # reset timing & continuity starting from the current slider frame
        tick["next"] = time.perf_counter()
        prev["robot_q"] = None
        prev["obj_q"] = None
        nonlocal_f["f"] = float(frame_slider.value)

    @play_btn.on_click
    def _(_evt) -> None:
        _toggle_playback()

    @prev_frame_btn.on_click
    def _(_evt) -> None:
        _set_discrete_frame(int(frame_slider.value) - 1)

    @next_frame_btn.on_click
    def _(_evt) -> None:
        _set_discrete_frame(int(frame_slider.value) + 1)

    @fps_in.on_update
    def _(_evt) -> None:
        tick["next"] = time.perf_counter()

    @interp_mult_in.on_update
    def _(_evt) -> None:
        tick["next"] = time.perf_counter()

    @frame_slider.on_update
    def _(_evt) -> None:
        # Only pause if this is a user interaction, not a programmatic update
        if not updating_programmatically["flag"]:
            # Pause when scrubbing so the background loop doesn't overwrite immediately
            _set_discrete_frame(int(frame_slider.value))

    register_keyboard_shortcut(
        server,
        "Playback: Play / Pause",
        hotkey="space",
        callback=_toggle_playback,
    )
    register_keyboard_shortcut(
        server,
        "Playback: Previous Frame",
        hotkey="[",
        callback=lambda: _set_discrete_frame(int(frame_slider.value) - 1),
    )
    register_keyboard_shortcut(
        server,
        "Playback: Next Frame",
        hotkey="]",
        callback=lambda: _set_discrete_frame(int(frame_slider.value) + 1),
    )
    register_keyboard_shortcut(
        server,
        "Playback: First Frame",
        hotkey="home",
        callback=lambda: _set_discrete_frame(0),
    )
    register_keyboard_shortcut(
        server,
        "Playback: Last Frame",
        hotkey="end",
        callback=lambda: _set_discrete_frame(n_frames - 1),
    )

    # ---------------- player loop ----------------
    def _player_loop() -> None:
        if n_frames <= 1:
            return
        while True:
            if playing["flag"]:
                now = time.perf_counter()
                fps_val = max(1, int(fps_in.value))
                mult = max(1, int(interp_mult_in.value))
                dt = 1.0 / (fps_val * mult)

                if now >= tick["next"]:
                    # advance by one visual step
                    f = nonlocal_f["f"] + 1.0 / mult
                    if loop:
                        f = f % max(1, n_frames)
                    else:
                        f = min(f, float(n_frames - 1))
                    nonlocal_f["f"] = f

                    k0 = int(np.floor(f))
                    k1 = (k0 + 1) % max(1, n_frames) if loop else min(k0 + 1, n_frames - 1)
                    u = float(f - k0)

                    q_interp = _interp_frame(qpos, k0, k1, u)
                    _apply_frame_from_q(q_interp, f)

                    # Update slider to show current frame number in real-time
                    # Use flag to prevent callback from pausing playback
                    updating_programmatically["flag"] = True
                    frame_slider.value = k0
                    updating_programmatically["flag"] = False

                    tick["next"] = now + dt
                else:
                    time.sleep(min(0.002, max(0.0, tick["next"] - now)))
            else:
                time.sleep(0.02)

    threading.Thread(target=_player_loop, daemon=True).start()

    # initial draw
    _apply_discrete_frame(0)

    # keep consistent with your previous return convention
    return [frame_slider], [0.0]
