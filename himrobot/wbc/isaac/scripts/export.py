#!/usr/bin/env python3
"""Export an rsl_rl checkpoint (``model_<it>.pt``) to the WB3 runtime npz
format consumed by ``himrobot.policies.wbc_np:NumpyWBCPolicy`` (torch-free,
docs/design/whole_body_v1.md sec 5.3) -- via ``himrobot.wbc.export_npz``
(WB3). Needs torch only (no Isaac Lab); runs on the box or the Mac.

    /workspace/venv/bin/python -m himrobot.wbc.isaac.scripts.export \\
        --checkpoint logs/rsl_rl/wbc_v1/wbc_v1_a/model_1500.pt \\
        --out work/wbc/v1/model_1500.npz
"""
from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="an rsl_rl model_<it>.pt")
    ap.add_argument("--out", required=True, help="output .npz, himrobot.policies.wbc_np's runtime format")
    ap.add_argument("--leg_default", choices=["stand", "mocap"], default="stand")
    args = ap.parse_args(argv)

    # himrobot.wbc.export_npz (WB3) owns the conversion: it reads the WHOLE
    # checkpoint -- actor Linear layers AND rsl_rl's EmpiricalNormalization
    # buffers (obs_norm_state_dict; dropping them would export a policy that
    # sees un-normalized inputs) -- and bakes leg_default / jnt_range /
    # scales into the npz schema himrobot.policies.wbc_np loads. Its own CLI
    # takes the same --leg-default {stand,mocap} flag, so delegate outright.
    from himrobot.wbc import export_npz

    export_npz.main([args.checkpoint, args.out, "--leg-default", args.leg_default])
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
