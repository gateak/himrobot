"""Export a himrobot/rl/param_train.py checkpoint to the torch-free .npz that
himrobot.policies.param_residual_np.NumpyParamResidualPolicy loads.

    PYTHONPATH=. ./.venv/bin/python tools/export_param_policy.py work/tt/rl/param/s3/best.pt \\
        assets/policies/param_residual_s3.npz
"""
import sys

import numpy as np
import torch

from himrobot.rl import param_residual


def main(src: str, dst: str) -> None:
    model, feat_rms, specs = param_residual.load_checkpoint(src, torch.device("cpu"))
    linears = [m for m in model.actor if isinstance(m, torch.nn.Linear)]
    assert len(linears) == 3, len(linears)
    arrays = {}
    for i, lin in enumerate(linears):
        arrays[f"W{i}"] = lin.weight.detach().cpu().numpy().astype(np.float32)
        arrays[f"b{i}"] = lin.bias.detach().cpu().numpy().astype(np.float32)
    arrays["rms_mean"] = np.asarray(feat_rms.mean, dtype=np.float64)
    arrays["rms_var"] = np.asarray(feat_rms.var, dtype=np.float64)
    arrays["param_names"] = np.array([s[0] for s in specs])
    arrays["param_kinds"] = np.array([s[1] for s in specs])
    arrays["param_bounds"] = np.array([float(s[2]) for s in specs], dtype=np.float64)
    arrays["param_space"] = np.array(param_residual.space_name_for(specs))
    np.savez(dst, **arrays)
    print(f"wrote {dst}: {arrays['param_space']} / {len(specs)} params, "
          f"layers {[a.shape for k, a in arrays.items() if k.startswith('W')]}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
