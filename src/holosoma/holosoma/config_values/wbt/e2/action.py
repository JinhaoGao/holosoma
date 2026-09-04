"""Whole Body Tracking action preset for E2."""

from holosoma.config_types.action import ActionManagerCfg, ActionTermCfg

e2_23dof_joint_pos = ActionManagerCfg(
    terms={
        "joint_control": ActionTermCfg(
            func="holosoma.managers.action.terms.joint_control:JointPositionActionTerm",
            params={},
            scale=1.0,
            clip=None,
        ),
    }
)

__all__ = ["e2_23dof_joint_pos"]
