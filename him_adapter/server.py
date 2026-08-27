from __future__ import annotations

import json
import sys

from .policy import (
    ACT_SIGNAL,
    INTERFACE_ID,
    JOINT_COUNT,
    OBS_SIGNAL,
    load_contract,
    load_policy,
)


def _metadata_response(policy: object) -> dict[str, object]:
    """Build the 'metadata' lifecycle response from the runtime contract.

    Falls back to the policy module's constants (which mirror the contract)
    for any field the contract file doesn't provide, so this never fails just
    because .him/contracts/runtime.json isn't shipped alongside the package.
    """
    contract = load_contract()
    observations = contract.get("observations") or [{"name": OBS_SIGNAL, "shape": [JOINT_COUNT]}]
    actions = contract.get("actions") or [{"name": ACT_SIGNAL, "shape": [JOINT_COUNT]}]
    robot = (contract.get("extensions") or {}).get("robot") or {}
    joint_count = robot.get("dof") or JOINT_COUNT
    return {
        "interface_id": contract.get("interface_id", INTERFACE_ID),
        "observations": [{"name": o.get("name"), "shape": o.get("shape")} for o in observations],
        "actions": [{"name": a.get("name"), "shape": a.get("shape")} for a in actions],
        "policy_class": type(policy).__name__,
        "joint_count": joint_count,
    }


def main() -> int:
    policy = load_policy()
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue  # blank/whitespace-only lines are not requests; nothing to respond to

        try:
            request = json.loads(line)
            kind = request.get("type") if isinstance(request, dict) else None
            if kind is None:
                response = {"error": "invalid_request"}
            elif kind == "health":
                response = {"ok": True}
            elif kind == "metadata":
                response = _metadata_response(policy)
            elif kind == "reset":
                policy.reset(request.get("episode", {}))
                response = {"ok": True}
            elif kind == "infer":
                response = policy.infer(request.get("observation", {}))
            elif kind == "close":
                print(json.dumps({"ok": True}), flush=True)
                return 0
            else:
                response = {"error": "unsupported_request", "type": kind}
        except Exception:
            # Never let one malformed line kill the long-running stdio session.
            response = {"error": "invalid_request"}

        print(json.dumps(response), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
