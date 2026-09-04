"""Whole Body Tracking termination preset for E2."""

from holosoma.config_types.termination import TerminationManagerCfg, TerminationTermCfg
from holosoma.config_values.wbt.e2.command import E2_TRACKED_BODY_NAMES

e2_23dof_wbt_termination = TerminationManagerCfg(
    terms={
        "timeout": TerminationTermCfg(
            func="holosoma.managers.termination.terms.common:timeout_exceeded",
            is_timeout=True,
        ),
        "bad_tracking": TerminationTermCfg(
            func="holosoma.managers.termination.terms.wbt:BadTrackingZOnly",
            params={
                "bad_ref_pos_threshold": 0.5,
                "bad_ref_ori_threshold": 0.8,
                "bad_motion_body_pos_threshold": 0.25,
                "body_names_to_track": E2_TRACKED_BODY_NAMES,
                "bad_motion_body_pos_body_names": [
                    "l_foot_sphere_5_link",
                    "r_foot_sphere_5_link",
                    "l_hand_sphere_link",
                    "r_hand_sphere_link",
                ],
                "bad_object_pos_threshold": 0.25,
                "bad_object_ori_threshold": 0.8,
            },
        ),
    }
)

__all__ = ["e2_23dof_wbt_termination"]
