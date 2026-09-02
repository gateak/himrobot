"""Compat door: ``python -m tt_sim.rl.param_train`` -- see himrobot.rl.param_train."""
from __future__ import annotations

import sys

from himrobot.rl.param_train import *  # noqa: F401,F403
from himrobot.rl.param_train import main

if __name__ == "__main__":
    main(sys.argv[1:])
