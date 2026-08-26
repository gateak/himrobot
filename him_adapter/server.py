from __future__ import annotations

import json
import sys

from .policy import load_policy


def main() -> int:
    policy = load_policy()
    for raw in sys.stdin:
        request = json.loads(raw)
        kind = request.get("type")
        if kind == "health":
            response = {"ok": True}
        elif kind == "reset":
            policy.reset(request.get("episode", {}))
            response = {"ok": True}
        elif kind == "infer":
            response = policy.infer(request["observation"])
        elif kind == "close":
            print(json.dumps({"ok": True}), flush=True)
            return 0
        else:
            response = {"error": "unsupported_request", "type": kind}
        print(json.dumps(response), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
