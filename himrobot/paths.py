"""Repo-root-relative paths, usable from any himrobot module regardless of
how deep it lives in the package tree.

Every module that used to compute its own REPO/_REPO via a chain of
``os.path.dirname(os.path.dirname(...(__file__)))`` calls broke whenever the
file's depth from the repo root changed (exactly what a package restructure
does) -- this module is the single source of truth instead: one dirname hop
from this file (himrobot/paths.py, always one level under the repo root) up
to the repo root, however many packages import it.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = REPO_ROOT / "assets"
