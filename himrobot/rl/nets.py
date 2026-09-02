"""Actor network shared by behavior cloning (himrobot/rl/bc.py) and PPO
(himrobot/rl/ppo.py -- not written by this task; this module is the shared
contract the sibling env/PPO task should import instead of re-declaring
its own actor class).

Architecture: 99 -> 256 -> 256 -> 29, tanh hidden activations + a tanh
output layer -- the task spec's "same architecture as PPO's actor". The
output tanh squashes to [-1, 1] to match himrobot.rl.env's action-space
convention: an action in [-1, 1]^29 is mapped to absolute joint-position
targets by himrobot.rl.env.map_action (STAND_POSE + action * ACTION_SCALE,
clipped to each joint's mujoco range; ACTION_SCALE is 0 for all 12 leg
joints, so those dims are always exactly 0 regardless of what the network
outputs there).

obs_dim/act_dim default to himrobot.rl.env.OBS_DIM/ACT_DIM (99/29) -- the
single source of truth for the flat-obs layout, so this module never
hardcodes a shape env.py doesn't already define.

State-dict keys are whatever nn.Sequential assigns positionally (net.0,
net.2, net.4 .weight/.bias -- Tanh has no parameters, so every other
index is a Linear). PPO's actor must instantiate this exact class (not a
re-declared lookalike with the same shape) so a himrobot/rl/bc.py checkpoint
loads into it unmodified via
``Actor().load_state_dict(torch.load(path, weights_only=False)["model_state_dict"])``
-- see bc.py's save_checkpoint and himrobot/rl/policy.py's BCPolicy for the
load side.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from himrobot.rl.env import ACT_DIM, OBS_DIM

HIDDEN = 256


class Actor(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM, hidden: int = HIDDEN):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden = hidden
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, act_dim),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)
