"""Compat door: ``python -m tt_sim.rl.eval_seeds`` -- see himrobot.eval.eval_seeds."""
from __future__ import annotations

import sys

from himrobot.eval.eval_seeds import *  # noqa: F401,F403
from himrobot.eval.eval_seeds import main

if __name__ == "__main__":
    main(sys.argv[1:])
