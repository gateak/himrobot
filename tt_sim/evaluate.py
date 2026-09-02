"""Compat door: ``python -m tt_sim.evaluate`` -- see himrobot.eval.evaluate."""
from __future__ import annotations

import sys

from himrobot.eval.evaluate import *  # noqa: F401,F403
from himrobot.eval.evaluate import main

if __name__ == "__main__":
    main(sys.argv[1:])
