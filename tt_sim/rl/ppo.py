"""Compat door: ``python -m tt_sim.rl.ppo`` -- see himrobot.rl.ppo."""
from __future__ import annotations

import sys

from himrobot.rl.ppo import *  # noqa: F401,F403
from himrobot.rl.ppo import main

if __name__ == "__main__":
    main(sys.argv[1:])
