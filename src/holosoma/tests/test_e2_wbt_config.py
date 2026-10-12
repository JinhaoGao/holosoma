from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from holosoma.config_values.robot import e2_23dof
from holosoma.config_values.wbt.e2.command import E2_CRAWL_MOTION_FILE, E2_TRACKED_BODY_NAMES
from holosoma.config_values.wbt.e2.experiment import e2_23dof_wbt
from holosoma.config_values.wbt.e2.termination import e2_23dof_wbt_termination
from holosoma.utils.path import resolve_data_file_path


def test_e2_wbt_motion_matches_robot_config() -> None:
    motion_path = Path(resolve_data_file_path(E2_CRAWL_MOTION_FILE))
    assert motion_path.is_file()

    required_keys = {
        "fps",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
        "body_names",
        "joint_names",
    }
    with np.load(motion_path) as motion:
        assert required_keys.issubset(motion.files)
        assert motion["fps"].item() == 50
        assert motion["joint_pos"].shape[1] == len(e2_23dof.dof_names) + 7
        assert motion["joint_vel"].shape[1] == len(e2_23dof.dof_names) + 6
        assert motion["joint_names"].tolist() == e2_23dof.dof_names
        assert set(e2_23dof.body_names).issubset(motion["body_names"].tolist())
        for key in required_keys - {"body_names", "joint_names"}:
            assert np.isfinite(motion[key]).all()


def test_e2_wbt_tracked_bodies_are_consistent() -> None:
    assert len(e2_23dof.dof_names) == e2_23dof.actions_dim == e2_23dof.dof_obs_size == 23
    assert len(e2_23dof.body_names) == e2_23dof.num_bodies == 36
    assert len(set(e2_23dof.body_names)) == len(e2_23dof.body_names)
    assert set(E2_TRACKED_BODY_NAMES).issubset(e2_23dof.body_names)

    bad_tracking = e2_23dof_wbt_termination.terms["bad_tracking"]
    assert bad_tracking.params["body_names_to_track"] == E2_TRACKED_BODY_NAMES
    assert set(bad_tracking.params["bad_motion_body_pos_body_names"]).issubset(E2_TRACKED_BODY_NAMES)
    assert e2_23dof_wbt.robot == e2_23dof
    assert e2_23dof_wbt.command is not None
    assert e2_23dof_wbt.command.setup_terms["motion_command"].params["motion_config"].motion_file == (
        E2_CRAWL_MOTION_FILE
    )


def test_e2_crawl_asset_has_hand_contacts() -> None:
    asset_root = Path(resolve_data_file_path("holosoma/data/robots"))
    urdf_path = asset_root / e2_23dof.asset.urdf_file
    assert urdf_path.is_file()

    root = ET.parse(urdf_path).getroot()  # noqa: S314 - parsing a trusted repository asset
    links = {link.attrib["name"]: link for link in root.findall("link")}
    for hand_name in ("l_hand_sphere_link", "r_hand_sphere_link"):
        collision_sphere = links[hand_name].find("collision/geometry/sphere")
        assert collision_sphere is not None
        assert float(collision_sphere.attrib["radius"]) == 0.04
