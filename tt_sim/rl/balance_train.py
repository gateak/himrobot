"""Compat door: ``python -m tt_sim.rl.balance_train`` -- see himrobot.rl.balance_train.

Importing himrobot.rl.balance_train itself sets TT_ROBOT_Y/TT_PARAM_NPZ
before it imports himrobot.sim.scene, so that ordering is already correct
by the time this shim's own import line runs -- nothing to duplicate here.
"""
from __future__ import annotations

import sys

from himrobot.rl.balance_train import *  # noqa: F401,F403
from himrobot.rl.balance_train import main

if __name__ == "__main__":
    main(sys.argv[1:])
