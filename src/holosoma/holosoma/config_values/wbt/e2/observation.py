"""Whole Body Tracking observations for E2.

The manager terms are robot agnostic; E2 dimensions are inferred from its
23-DoF robot and tracked-body configuration at environment setup.
"""

from holosoma.config_values.wbt.g1.observation import g1_29dof_wbt_observation

e2_23dof_wbt_observation = g1_29dof_wbt_observation

__all__ = ["e2_23dof_wbt_observation"]
