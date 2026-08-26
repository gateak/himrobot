from __future__ import annotations


class HoldPolicy:
    """Safe template policy that returns the observed joint position."""

    def reset(self, episode: dict[str, object]) -> None:
        return None

    def infer(self, observation: dict[str, object]) -> dict[str, object]:
        signals = observation.get("signals", {})
        q = list(signals.get("robot.joint_position", []))
        return {"signals": {"robot.desired_joint_position": q}, "valid_for_ms": 40}
