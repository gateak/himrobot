"""himrobot.wbc.isaac -- the Isaac Lab side of the whole-body controller (WBC).

See docs/design/whole_body_v1.md sections 2-3 for what is grafted from where
and the exact observation/action/reward contract this package implements;
Isaac Lab task id ``Himrobot-WBC-G1-v0``.

Every submodule that needs isaaclab imports it directly at module level, with
a ``try/except ImportError`` fallback to plain-object bases where a class only
needs to be *importable* without isaaclab (the pure-torch math in ``mdp/*``,
exercised standalone by ``tests/test_wbc_mdp_torch.py`` with no isaaclab
anywhere). Real Isaac Lab wiring (``env_cfg.py``, ``ppo_cfg.py``, ``assets.py``,
and the isaaclab-cfg builders at the bottom of each ``mdp/*`` module) needs
isaaclab to be either really installed (the rented box) or stubbed into
``sys.modules`` first -- the same technique as
``work/pace/g1_port/stub_import_smoke.py``, done locally in
``tests/test_wbc_isaac_stub.py``.

This top-level package itself imports nothing beyond this docstring, so a
bare ``import himrobot.wbc.isaac`` never requires isaaclab.
"""
from __future__ import annotations
