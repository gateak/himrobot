"""Compat door: ``python -m tt_sim.evaluate_stance`` -- see himrobot.eval.evaluate_stance.

Importing himrobot.eval.evaluate_stance itself sets TT_ROBOT_Y/TT_PARAM_NPZ
before it imports himrobot.eval.evaluate (which imports himrobot.sim.scene),
so that ordering is already correct by the time this shim's own import line
runs -- nothing to duplicate here.
"""
from __future__ import annotations

import sys

from himrobot.eval.evaluate_stance import *  # noqa: F401,F403
from himrobot.eval.evaluate_stance import main

if __name__ == "__main__":
    main(sys.argv[1:])
