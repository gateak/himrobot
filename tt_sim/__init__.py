"""Compatibility shim: ``tt_sim`` is the pre-restructure package name.

The real code lives in ``himrobot`` now (see himrobot/__init__.py's own
docstring). This package keeps three doors the HIM platform and a published
challenge entry load our code by, unchanged:

  * ``python -m tt_sim.evaluate_stance`` / ``python -m tt_sim.evaluate`` and
    the other ``python -m tt_sim.rl.*`` training entrypoints -- real, thin
    files under this package (tt_sim/evaluate.py, tt_sim/evaluate_stance.py,
    tt_sim/rl/ppo.py, tt_sim/rl/param_train.py, tt_sim/rl/eval_seeds.py,
    tt_sim/rl/balance_train.py) that star-import their himrobot module and
    forward to its ``main``. Real files, not aliases, because ``python -m``
    executes a module's own top-level code under ``__name__ == "__main__"``
    -- an aliased module object that is already fully imported (see below)
    never re-runs that code, so ``-m`` would silently do nothing.
  * the policy path string ``tt_sim.policies.param_residual_np:NumpyParamResidualPolicy``
    -- also a real, thin file (tt_sim/policies/param_residual_np.py), torch-free
    to import (HIM's mujoco worker has no torch).
  * every other old dotted import (``tt_sim.scene``, ``tt_sim.rl.env``, ...)
    -- resolved lazily by the ALIASES finder installed below, straight
    through to the real himrobot module. Nothing is imported eagerly: the
    finder only does anything when one of these names is actually imported.

The finder aliases rather than re-implements: ``sys.modules["tt_sim.scene"]``
ends up being the exact same module object as ``sys.modules["himrobot.sim.scene"]``
(create_module returns the already-imported target; exec_module is a no-op),
so there is exactly one copy of every module's state -- no drift, no double
registration of e.g. TableTennisEnv.
"""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys

import himrobot  # light: only runs the MuJoCo GL-backend probe, never imports himrobot.sim.scene

# Old dotted path -> new himrobot module. Deliberately does NOT include names
# that have a real thin file instead (tt_sim.evaluate, tt_sim.evaluate_stance,
# tt_sim.rl.ppo/param_train/eval_seeds/balance_train, tt_sim.policies.param_residual_np,
# and the tt_sim.rl/tt_sim.policies packages themselves) -- for those, this
# finder must return None so the normal filesystem finder loads the real file.
ALIASES: dict[str, str] = {
    "tt_sim.scene": "himrobot.sim.scene",
    "tt_sim.media": "himrobot.sim.media",
    "tt_sim.obs_perturb": "himrobot.sim.obs_perturb",
    "tt_sim.policies.base": "himrobot.policies.base",
    "tt_sim.policies.hold": "himrobot.policies.hold",
    "tt_sim.policies.ik_hitter": "himrobot.policies.ik_hitter",
    "tt_sim.policies.balance_np": "himrobot.policies.balance_np",
    "tt_sim.rl.env": "himrobot.rl.env",
    "tt_sim.rl.residual": "himrobot.rl.residual",
    "tt_sim.rl.balance_env": "himrobot.rl.balance_env",
    "tt_sim.rl.intercept": "himrobot.rl.intercept",
    "tt_sim.rl.nets": "himrobot.rl.nets",
    "tt_sim.rl.policy": "himrobot.rl.policy",
    "tt_sim.rl.demos": "himrobot.rl.demos",
    "tt_sim.rl.bc": "himrobot.rl.bc",
    "tt_sim.rl.param_residual": "himrobot.rl.param_residual",
}


class _AliasFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """sys.meta_path entry: for any name in ALIASES, resolve straight to the
    already-imported himrobot module. Returns None for anything else, so
    every other finder (in particular the normal filesystem one, for the
    real thin files above) is unaffected."""

    def find_spec(self, name, path, target=None):
        if name not in ALIASES:
            return None
        return importlib.util.spec_from_loader(name, self)

    def create_module(self, spec):
        return importlib.import_module(ALIASES[spec.name])

    def exec_module(self, module):
        pass  # create_module already returned the fully-imported target module; nothing left to do


sys.meta_path.insert(0, _AliasFinder())
