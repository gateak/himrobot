#!/usr/bin/env python3
"""Export an rsl_rl checkpoint (``model_<it>.pt``) to the WB3 runtime npz
format consumed by ``himrobot.policies.wbc_np:NumpyWBCPolicy`` (torch-free,
docs/design/whole_body_v1.md sec 5.3) -- via ``himrobot.wbc.export_npz``
(WB3, not yet landed). Isaac Lab only for loading the checkpoint's shapes;
run on the box.

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

    import torch

    from himrobot.wbc.isaac.mdp import actions

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    actor_state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    # himrobot.wbc.export_npz (WB3) turns an actor's Linear-layer weights
    # plus the observation normaliser's running mean/var into the same
    # W*/b*/rms_mean/rms_var/obs_version npz shape himrobot.policies.
    # balance_np.NumpyBalancePolicy already consumes -- see that module for
    # the exact schema this call must reproduce.
    from himrobot.wbc import export_npz  # noqa: F401 -- WB3, not yet landed

    leg_default = (
        actions.STAND_LEG_DEFAULT if args.leg_default == "stand"
        else tuple(_load_mocap_leg_default())
    )
    export_npz.export(actor_state=actor_state, out_path=args.out, leg_default=leg_default,
                      action_dim=actions.ACTION_DIM, leg_scale=actions.LEG_SCALE,
                      waist_scale=actions.WAIST_SCALE)
    print(f"wrote {args.out}")
    return 0


def _load_mocap_leg_default() -> "list[float]":
    import numpy as np

    from himrobot.paths import ASSETS_DIR
    z = np.load(str(ASSETS_DIR / "wbc" / "mocap_leg_prior.npz"), allow_pickle=False)
    return [float(v) for v in z["global_mean"]]


if __name__ == "__main__":
    raise SystemExit(main())
