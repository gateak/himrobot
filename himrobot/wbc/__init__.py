"""himrobot.wbc: whole-body-controller lane (docs/design/whole_body_v1.md).

Runtime-side pieces only (library recording, observation/action layout, the
replay gate, export, diagnostics); the Isaac Lab task package lives under
himrobot.wbc.isaac and imports isaaclab only on a rented GPU box. Nothing
in this package is on the scored path today -- the shipped policy is still
himrobot.policies.param_residual_np.NumpyParamResidualPolicy.
"""
