"""Configuration for behavior-cloning training.

PROVENANCE / ASSUMPTIONS  (read before trusting any number here)
----------------------------------------------------------------
Joint count and order come from the repo's own runtime contract,
``.him/contracts/runtime.json`` (``observations[0].shape == [29]``,
``extensions.robot.joint_order`` = the 29 Unitree G1 joints in MuJoCo-Menagerie
actuator order, identical to the Unitree SDK2 ``G1JointIndex`` enum). The
retargeted-motion CSVs under ``data/g1`` were written in that same order
(``gmr_g1_joint_order.txt`` alongside them; ``training.dataset`` checks it).

The deploy signal names ``robot.joint_position`` / ``robot.desired_joint_position``
and the 50 Hz clock are what OUR contract declares (taken from the HIM
single-policy template). They are UNCONFIRMED against HIM's server-side
evaluator -- HIM has not confirmed the wire format, joint order, or rate.
Everything is parameterised so corrected values drop in without touching the
training code.

Training data is NOT robot data: it is GMR-retargeted Rokoko human mocap
(kinematic poses, 30 fps). See training/README.md.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

# Path of the runtime contract, relative to the repo root (this file's grandparent).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTRACT_PATH = os.path.join(_REPO_ROOT, ".him", "contracts", "runtime.json")

# Fallback copy of extensions.robot.joint_order from .him/contracts/runtime.json,
# used only when the contract file is not readable (e.g. the training package
# imported from a checkout without the .him overlay). Keep in sync with the contract.
G1_JOINT_ORDER: tuple[str, ...] = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)


def load_contract_joint_order(path: str = CONTRACT_PATH) -> list[str]:
    """Joint names from the runtime contract's extensions.robot.joint_order.

    Falls back to the in-module copy if the contract is missing or malformed.
    """
    try:
        with open(path) as fh:
            contract = json.load(fh)
        order = contract["extensions"]["robot"]["joint_order"]
        if isinstance(order, list) and all(isinstance(n, str) for n in order):
            return list(order)
    except (OSError, KeyError, TypeError, ValueError):
        pass
    return list(G1_JOINT_ORDER)


@dataclass
class DataConfig:
    root: str = "data/g1"                           # dir of G1 CSVs (or LeRobot-v3 parquet episodes)
    # --- LeRobot-v3 parquet source ---
    obs_joint_key: str = "observation.state.joints"  # dataset column for joint state
    action_key: str = "action.joint_targets"         # dataset column for joint targets
    ball_key: str | None = None                     # e.g. "observation.state.ball" -> concat ball_state to obs
    ball_dim: int = 6
    # --- joint layout (from .him/contracts/runtime.json) ---
    joints: int = 29                                # G1 29-DoF; == len(joint_order)
    joint_order: list[str] = field(default_factory=load_contract_joint_order)
    # --- retargeted G1 motion CSV source (sim/gmr_bvh_to_g1_headless.py output) ---
    csv_root_cols: int = 7                          # root xyz (3, m) + root quat xyzw (4) precede the joints
    csv_fps: float = 30.0                           # frame rate the CSVs were written at (BVH/GMR --fps 30)
    csv_root_in_obs: bool = False                   # False: obs = joints only. True: append root z + quat xyzw (+5)
    csv_joint_order_file: str = "gmr_g1_joint_order.txt"  # checked against joint_order when present in root
    # --- deployment contract signal names (what him_adapter.server receives/emits) ---
    # UNCONFIRMED against HIM's evaluator; copied from .him/contracts/runtime.json.
    deploy_obs_signal: str = "robot.joint_position"
    deploy_act_signal: str = "robot.desired_joint_position"
    # --- synthetic-sample sizing (used only when `root` has neither parquet nor csv) ---
    synth_episodes: int = 20
    synth_steps: int = 300


@dataclass
class ModelConfig:
    kind: str = "mlp"                               # "mlp" | "transformer"
    hidden: tuple[int, ...] = (256, 256)
    dropout: float = 0.0
    residual: bool = True                           # predict a delta on top of the current joints (see model.py)
    # transformer-only (ignored for "mlp", which always trains/serves at context 1):
    context: int = 8                                 # history window; 1 works too (no crash), just
                                                      # degenerates to single-step attention -- see
                                                      # training/model.py:TinyTransformerPolicy
    d_model: int = 128
    nhead: int = 4
    layers: int = 2


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.0
    val_frac: float = 0.1
    val_split: str = "random"                       # "random" (per-step) | "episode" (hold out whole episodes)
    seed: int = 0
    device: str = "auto"                            # auto -> cuda|mps|cpu
    out: str = "models/bc_policy.pt"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @property
    def obs_dim(self) -> int:
        d = self.data.joints
        if self.data.ball_key:
            d += self.data.ball_dim
        if self.data.csv_root_in_obs:
            d += 5  # root height z + quat xyzw
        return d

    @property
    def act_dim(self) -> int:
        return self.data.joints

    @property
    def context(self) -> int:
        return self.model.context if self.model.kind == "transformer" else 1
