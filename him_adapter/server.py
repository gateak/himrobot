"""Compat door: ``python -m him_adapter.server`` -- see himrobot.serve.server."""
from __future__ import annotations

from himrobot.serve.server import *  # noqa: F401,F403
from himrobot.serve.server import main

if __name__ == "__main__":
    raise SystemExit(main())
