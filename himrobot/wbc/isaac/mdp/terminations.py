"""Terminations for ``Himrobot-WBC-G1-v0``, docs/design/whole_body_v1.md
sec 3.3. Pure torch (tier 1 + 2); ``env_cfg.py`` wires these into real
isaaclab ``TerminationTermCfg`` objects (see the ``mdp`` package docstring).

Pelvis-height and episode-length thresholds are shared with the MuJoCo
evaluator on purpose (``scene.FALL_PELVIS_Z``, ``evaluate.EPISODE_CONTROL_STEPS``)
so Isaac and the Mac scoreboard agree on what a fall and an episode are.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from himrobot.sim import scene

FALL_PELVIS_Z = scene.FALL_PELVIS_Z  # 0.45 m
TILT_LIMIT_RAD = 0.7
EPISODE_LENGTH_STEPS = 150  # 3 s @ 50 Hz, matches himrobot.eval.evaluate.EPISODE_CONTROL_STEPS
TERMINATION_PENALTY = -10.0  # charged once by the reward manager, not by these functions


@dataclass(frozen=True)
class TerminationSpec:
    name: str
    note: str


TERMINATION_TERMS: list[TerminationSpec] = [
    TerminationSpec("fell", f"pelvis z < {FALL_PELVIS_Z} m (= scene.FALL_PELVIS_Z)"),
    TerminationSpec("tilted", f"tilt > {TILT_LIMIT_RAD} rad"),
    TerminationSpec("time_out", f"{EPISODE_LENGTH_STEPS} steps (3 s)"),
]
TERMINATION_TERM_NAMES = [t.name for t in TERMINATION_TERMS]


def fell(base_z: torch.Tensor, threshold: float = FALL_PELVIS_Z) -> torch.Tensor:
    """True once the pelvis drops below ``threshold`` -- same test and same
    number as scene.py's own free-base "fell" event latch."""
    return base_z < threshold


def tilted(tilt_rad: torch.Tensor, limit: float = TILT_LIMIT_RAD) -> torch.Tensor:
    return tilt_rad > limit


def time_out(episode_step: torch.Tensor, limit: int = EPISODE_LENGTH_STEPS) -> torch.Tensor:
    return episode_step >= limit
