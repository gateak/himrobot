from __future__ import annotations

import os
import sys

DEFAULT_CHECKPOINT = os.environ.get("HIM_POLICY_CHECKPOINT", "models/bc_policy.pt")


class HoldPolicy:
    """Safe template policy that returns the observed joint position."""

    def reset(self, episode: dict[str, object]) -> None:
        return None

    def infer(self, observation: dict[str, object]) -> dict[str, object]:
        signals = observation.get("signals", {})
        q = list(signals.get("robot.joint_position", []))
        return {"signals": {"robot.desired_joint_position": q}, "valid_for_ms": 40}


class BCPolicy:
    """Deployment wrapper around a trained behavior-cloning checkpoint.

    Loads torch lazily so this module stays import-safe (and the smoke test
    stays green) in environments without torch. Any per-step problem degrades
    to the contract's "hold" fallback rather than raising.
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

    def reset(self, episode: dict[str, object]) -> None:
        self._history = []

    def infer(self, observation: dict[str, object]) -> dict[str, object]:
        torch = self._torch
        signals = observation.get("signals", {})
        q = list(signals.get(self.obs_signal, []))
        # Shape mismatch or missing input -> hold fallback (error_model.fallback_action).
        if len(q) != self.obs_dim:
            hold = list(signals.get("robot.joint_position", q))
            return {"signals": {self.act_signal: hold}, "valid_for_ms": 40}

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
