"""Manager-based MDP terms for ``Himrobot-WBC-G1-v0`` -- the torch twins of
``himrobot/wbc/{obs,action}.py`` (docs/design/whole_body_v1.md sec 3).

Each submodule carries three tiers, in order:

1. Plain-data ``*Spec``/``*Block`` dataclasses -- the actual dims, weights,
   stds and thresholds from the design doc's tables, with no isaaclab (or
   even torch) dependency at all. ``env_cfg.py`` and the tests read these
   directly; they are the single source of truth the isaaclab wiring below
   is built *from*, not the other way round.
2. Pure ``torch`` functions implementing the real per-step math (rotations,
   exp kernels, library gathers, the joint-target formula). Import-safe with
   plain torch alone -- this is what ``tests/test_wbc_mdp_torch.py`` exercises
   with no isaaclab anywhere, real or stubbed.
Tier 3 -- the isaaclab wiring that turns tier-2 functions into real
``ObservationTermCfg``/``RewardTermCfg``/``ActionTerm``/``TerminationTermCfg``/
``CommandTerm`` objects -- lives entirely in ``env_cfg.py``, the only file in
this package that imports isaaclab unconditionally. Keeping it there (rather
than scattered across each ``mdp/*`` module) is what lets every module below
stay import-safe with plain torch alone.

Importing this package pulls in every submodule (so ``mdp.rewards``,
``mdp.commands`` etc. resolve as attributes), which is always safe: no
submodule requires isaaclab merely to be imported.
"""
from __future__ import annotations

from himrobot.wbc.isaac.mdp import actions, commands, observations, rewards, terminations

__all__ = ["actions", "commands", "observations", "rewards", "terminations"]
