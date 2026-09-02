"""PyTorch policy models + checkpoint (de)serialization.

Both `bc.train` (to build an architecture) and `him_adapter.policy` (to
reload one at deploy time) go through this module, so the checkpoint format
lives in exactly one place.
"""

from __future__ import annotations

import torch
import torch.nn as nn

CKPT_FORMAT = "him-bc-v1"


def _zero_init(linear: nn.Linear) -> nn.Linear:
    nn.init.zeros_(linear.weight)
    nn.init.zeros_(linear.bias)
    return linear


class MLPPolicy(nn.Module):
    """Feed-forward joint-state -> joint-target regressor.

    ``residual=True`` (default): output = x[:, :act_dim] + net(x) -- net predicts,
    in normalized space, a correction to the current joints (obs's first act_dim
    dims). A zero-initialised last layer starts the model exactly at the "hold"
    policy, which matters for next-pose data where consecutive frames are nearly
    identical. The skip lives inside the module, so the deploy-side loader
    (him_adapter.policy) needs no change.

    This "zero-init == hold" identity is exact only in *normalized* space, and only
    if obs/action norm stats agree on those first act_dim columns: same physical
    joints, but obs stats are fit over the windowed input while act stats are fit
    over the target, so they don't match for free. ``bc.dataset.compute_norm_stats``
    makes them match; see ``legacy/bc/tests/test_residual_hold.py``.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 256), dropout: float = 0.0,
                 residual: bool = True):
        super().__init__()
        self.act_dim = act_dim
        self.residual = residual
        dims = [obs_dim, *hidden]
        layers: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU(), nn.Dropout(dropout)]
        head = nn.Linear(dims[-1], act_dim)
        layers += [_zero_init(head) if residual else head]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, obs_dim]
        y = self.net(x)
        return y + x[:, : self.act_dim] if self.residual else y


class TinyTransformerPolicy(nn.Module):
    """Encodes a short window of joint states and predicts the next target."""

    def __init__(self, obs_dim, act_dim, context=8, d_model=128, nhead=4, layers=2, dropout=0.0,
                 residual: bool = True):
        super().__init__()
        self.context = context
        self.act_dim = act_dim
        self.residual = residual
        self.inp = nn.Linear(obs_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, context, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=4 * d_model, dropout=dropout, batch_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, layers)
        self.head = _zero_init(nn.Linear(d_model, act_dim)) if residual else nn.Linear(d_model, act_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, T, obs_dim]
        t = x.size(1)
        h = self.inp(x) + self.pos[:, :t]
        h = self.enc(h)
        y = self.head(h[:, -1])                          # predict from last step
        return y + x[:, -1, : self.act_dim] if self.residual else y


def build_model(model_cfg, obs_dim: int, act_dim: int) -> nn.Module:
    if model_cfg.kind == "mlp":
        return MLPPolicy(obs_dim, act_dim, tuple(model_cfg.hidden), model_cfg.dropout, model_cfg.residual)
    if model_cfg.kind == "transformer":
        return TinyTransformerPolicy(
            obs_dim, act_dim, model_cfg.context, model_cfg.d_model,
            model_cfg.nhead, model_cfg.layers, model_cfg.dropout, model_cfg.residual,
        )
    raise ValueError(f"unknown model kind: {model_cfg.kind!r}")


def save_checkpoint(
    path: str, *, model, cfg, norm: dict, val_loss: float, provenance: str, extra: dict | None = None
) -> None:
    """Write a self-contained checkpoint. ``extra`` (optional) holds informational
    training metadata (source, split, baseline); the loader ignores it.

    obs_dim is taken from the normalization stats so it matches the tensors the
    model was actually trained on (config and data can disagree, see train.py)."""
    ckpt = {
        "format": CKPT_FORMAT,
        "model_kind": cfg.model.kind,
        "obs_dim": len(norm["obs_mean"]),
        "act_dim": len(norm["act_mean"]),
        "context": cfg.context,
        "joint_order": list(cfg.data.joint_order),  # from .him/contracts/runtime.json (informational)
        "model": {
            "hidden": list(cfg.model.hidden),
            "dropout": cfg.model.dropout,
            "d_model": cfg.model.d_model,
            "nhead": cfg.model.nhead,
            "layers": cfg.model.layers,
            "residual": cfg.model.residual,
        },
        "deploy": {
            "obs_signal": cfg.data.deploy_obs_signal,
            "act_signal": cfg.data.deploy_act_signal,
        },
        "norm": norm,  # {"obs_mean","obs_std","act_mean","act_std"} as lists
        "val_loss": float(val_loss),
        "provenance": provenance,
        "extra": dict(extra or {}),
        "state_dict": model.state_dict(),
    }
    torch.save(ckpt, path)


def build_model_from_ckpt(ckpt: dict) -> nn.Module:
    """Reconstruct a model from a checkpoint dict and load its weights (eval mode)."""
    if ckpt.get("format") != CKPT_FORMAT:
        raise ValueError(f"unexpected checkpoint format: {ckpt.get('format')!r}")
    kind = ckpt["model_kind"]
    m = ckpt["model"]
    residual = m.get("residual", False)  # pre-residual checkpoints default to a plain head
    if kind == "mlp":
        model = MLPPolicy(ckpt["obs_dim"], ckpt["act_dim"], tuple(m["hidden"]), m.get("dropout", 0.0), residual)
    elif kind == "transformer":
        model = TinyTransformerPolicy(
            ckpt["obs_dim"], ckpt["act_dim"], ckpt.get("context", 8),
            m["d_model"], m["nhead"], m["layers"], m.get("dropout", 0.0), residual,
        )
    else:
        raise ValueError(f"unknown model kind in checkpoint: {kind!r}")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model
