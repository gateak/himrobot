from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

DEFAULT_CHECKPOINT = os.environ.get("HIM_POLICY_CHECKPOINT", "models/bc_policy.pt")

# Fallback values mirroring .him/contracts/runtime.json, used wherever the
# contract file can't be read (e.g. a packaged deployment without .him/).
CONTRACT_PATH = Path(__file__).resolve().parents[1] / ".him" / "contracts" / "runtime.json"
INTERFACE_ID = "himrobot-g1-v1"
JOINT_COUNT = 29
OBS_SIGNAL = "robot.joint_position"
ACT_SIGNAL = "robot.desired_joint_position"
ERROR_INVALID_OBSERVATION = "INVALID_OBSERVATION"


def load_contract() -> dict[str, object]:
    """Best-effort load of the runtime contract JSON. Returns {} if missing/unreadable."""
    try:
        return json.loads(CONTRACT_PATH.read_text())
    except (OSError, ValueError):
        return {}


def validate_observation(signals: object, expected_len: int = JOINT_COUNT) -> list[int | float] | None:
    """Extract and validate ``robot.joint_position`` from an observation's signals.

    Returns the list unchanged (same objects/types) on success. None if
    ``signals`` isn't a dict, the value is missing/not a list/wrong length, or
    any element isn't a finite (non-NaN, non-Infinity) real number.
    """
    if not isinstance(signals, dict):
        return None
    q = signals.get(OBS_SIGNAL)
    if not isinstance(q, list) or len(q) != expected_len:
        return None
    for v in q:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None
        if not math.isfinite(v):
            return None
    return list(q)


class HoldPolicy:
    """Safe template policy that returns the observed joint position."""

    def __init__(self) -> None:
        self._last_hold: list[int | float] | None = None

    def reset(self, episode: dict[str, object]) -> None:
        self._last_hold = None

    def infer(self, observation: dict[str, object]) -> dict[str, object]:
        signals = observation.get("signals", {}) if isinstance(observation, dict) else {}
        q = validate_observation(signals)
        if q is None:
            hold = self._last_hold if self._last_hold is not None else [0.0] * JOINT_COUNT
            return {
                "signals": {ACT_SIGNAL: hold},
                "valid_for_ms": 40,
                "error": ERROR_INVALID_OBSERVATION,
            }
        self._last_hold = q
        return {"signals": {ACT_SIGNAL: q}, "valid_for_ms": 40}


class BCPolicy:
    """Deployment wrapper around a trained behavior-cloning checkpoint.

    Loads torch lazily (keeps this module import-safe, and the smoke test
    green, without torch). Invalid observations degrade to the contract's
    "hold" fallback instead of raising; failures inside model inference
    still propagate.
    """

    def __init__(self, checkpoint_path: str = DEFAULT_CHECKPOINT):
        import torch  # lazy: only needed when a checkpoint is actually present
        from training.model import build_model_from_ckpt

        self._torch = torch
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.model = build_model_from_ckpt(ckpt)
        self.obs_signal = ckpt["deploy"]["obs_signal"]
        self.act_signal = ckpt["deploy"]["act_signal"]
        self.obs_dim = int(ckpt["obs_dim"])
        self.context = int(ckpt.get("context", 1))
        n = ckpt["norm"]
        self.obs_mean = torch.tensor(n["obs_mean"], dtype=torch.float32)
        self.obs_std = torch.tensor(n["obs_std"], dtype=torch.float32)
        self.act_mean = torch.tensor(n["act_mean"], dtype=torch.float32)
        self.act_std = torch.tensor(n["act_std"], dtype=torch.float32)
        self._history: list = []
        self._last_hold: list[int | float] | None = None

    def reset(self, episode: dict[str, object]) -> None:
        self._history = []
        self._last_hold = None

    def infer(self, observation: dict[str, object]) -> dict[str, object]:
        torch = self._torch
        signals = observation.get("signals", {}) if isinstance(observation, dict) else {}
        # Validate against the contract's joint-position signal, sized to this
        # checkpoint's own obs_dim (== JOINT_COUNT for every checkpoint trained
        # against the current contract).
        q = validate_observation(signals, self.obs_dim)
        if q is None:
            hold = self._last_hold if self._last_hold is not None else [0.0] * self.obs_dim
            return {
                "signals": {self.act_signal: hold},
                "valid_for_ms": 40,
                "error": ERROR_INVALID_OBSERVATION,
            }
        self._last_hold = q

        x = (torch.tensor(q, dtype=torch.float32) - self.obs_mean) / self.obs_std
        with torch.no_grad():
            if self.context > 1:
                self._history.append(x)
                self._history = self._history[-self.context :]
                window = self._history
                if len(window) < self.context:
                    window = [window[0]] * (self.context - len(window)) + window
                y = self.model(torch.stack(window).unsqueeze(0))
            else:
                y = self.model(x.unsqueeze(0))
            y = y.squeeze(0) * self.act_std + self.act_mean
        return {"signals": {self.act_signal: [float(v) for v in y.tolist()]}, "valid_for_ms": 40}


def load_policy(checkpoint_path: str | None = None):
    """Return a BCPolicy if a usable checkpoint exists, else HoldPolicy.

    Never raises: a missing/broken checkpoint or absent torch falls back to the
    safe hold policy, which is what the runtime contract expects.
    """
    path = checkpoint_path or DEFAULT_CHECKPOINT
    if not os.path.exists(path):
        return HoldPolicy()
    try:
        policy = BCPolicy(path)
        print(f"[him_adapter] loaded BC policy from {path}", file=sys.stderr)
        return policy
    except Exception as exc:  # torch missing, bad checkpoint, arch mismatch, ...
        print(
            f"[him_adapter] checkpoint {path!r} present but load failed ({exc!r}); "
            "using HoldPolicy fallback",
            file=sys.stderr,
        )
        return HoldPolicy()
