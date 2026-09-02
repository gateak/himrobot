"""Compat door: the published challenge entry's policy path string
``tt_sim.policies.param_residual_np:NumpyParamResidualPolicy`` -- see
himrobot.policies.param_residual_np. Torch-free (HIM's mujoco worker has no
torch): this module and everything it imports must stay importable with
torch entirely absent.
"""
from __future__ import annotations

from himrobot.policies.param_residual_np import *  # noqa: F401,F403
from himrobot.policies.param_residual_np import NumpyParamResidualPolicy, S3_WEIGHTS  # noqa: F401
