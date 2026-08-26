"""PyTorch policy models + checkpoint (de)serialization.

Both `training.train` (to build an architecture) and `him_adapter.policy` (to
reload one at deploy time) go through this module, so the checkpoint format
lives in exactly one place.
"""

from __future__ import annotations

import torch
import torch.nn as nn

CKPT_FORMAT = "him-bc-v1"


class MLPPolicy(nn.Module):
    """Feed-forward joint-state -> joint-target regressor."""

    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 256), dropout: float = 0.0):
        super().__init__()
        dims = [obs_dim, *hidden]
        layers: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU(), nn.Dropout(dropout)]
        layers += [nn.Linear(dims[-1], act_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, obs_dim]
        return self.net(x)


class TinyTransformerPolicy(nn.Module):
    """Encodes a short window of joint states and predicts the next target."""

    def __init__(self, obs_dim, act_dim, context=8, d_model=128, nhead=4, layers=2, dropout=0.0):
        super().__init__()
        self.context = context
        self.inp = nn.Linear(obs_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, context, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=4 * d_model, dropout=dropout, batch_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, layers)
        self.head = nn.Linear(d_model, act_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, T, obs_dim]
        t = x.size(1)
        h = self.inp(x) + self.pos[:, :t]
        h = self.enc(h)
        return self.head(h[:, -1])                       # predict from last step


def build_model(model_cfg, obs_dim: int, act_dim: int) -> nn.Module:
    if model_cfg.kind == "mlp":
        return MLPPolicy(obs_dim, act_dim, tuple(model_cfg.hidden), model_cfg.dropout)
    if model_cfg.kind == "transformer":
        return TinyTransformerPolicy(
            obs_dim, act_dim, model_cfg.context, model_cfg.d_model,
            model_cfg.nhead, model_cfg.layers, model_cfg.dropout,
        )
    raise ValueError(f"unknown model kind: {model_cfg.kind!r}")


def save_checkpoint(path: str, *, model, cfg, norm: dict, val_loss: float, provenance: str) -> None:
    ckpt = {
        "format": CKPT_FORMAT,
        "model_kind": cfg.model.kind,
        "obs_dim": cfg.obs_dim,
        "act_dim": cfg.act_dim,
        "context": cfg.context,
        "model": {
            "hidden": list(cfg.model.hidden),
            "dropout": cfg.model.dropout,
            "d_model": cfg.model.d_model,
            "nhead": cfg.model.nhead,
            "layers": cfg.model.layers,
        },
        "deploy": {
            "obs_signal": cfg.data.deploy_obs_signal,
            "act_signal": cfg.data.deploy_act_signal,
        },
        "norm": norm,  # {"obs_mean","obs_std","act_mean","act_std"} as lists
        "val_loss": float(val_loss),
        "provenance": provenance,
        "state_dict": model.state_dict(),
    }
    torch.save(ckpt, path)


def _rebuild_from_meta(ckpt: dict) -> nn.Module:
    kind = ckpt["model_kind"]
    m = ckpt["model"]
    if kind == "mlp":
        return MLPPolicy(ckpt["obs_dim"], ckpt["act_dim"], tuple(m["hidden"]), m.get("dropout", 0.0))
    if kind == "transformer":
        return TinyTransformerPolicy(
            ckpt["obs_dim"], ckpt["act_dim"], ckpt.get("context", 8),
            m["d_model"], m["nhead"], m["layers"], m.get("dropout", 0.0),
        )
    raise ValueError(f"unknown model kind in checkpoint: {kind!r}")


def build_model_from_ckpt(ckpt: dict) -> nn.Module:
    """Reconstruct a model from a checkpoint dict and load its weights (eval mode)."""
    if ckpt.get("format") != CKPT_FORMAT:
        raise ValueError(f"unexpected checkpoint format: {ckpt.get('format')!r}")
    model = _rebuild_from_meta(ckpt)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model
