"""Configuration for behavior-cloning training.

PROVENANCE / ASSUMPTIONS  (read before trusting any number here)
----------------------------------------------------------------
The shapes below come from the *example* manifests shipped inside `him_sdk`,
NOT from the real downloaded dataset (the public table-tennis release was not
retrievable — the artifact endpoint returns SPA HTML instead of the tar; see
memory: him-env-and-download-gotchas). The relevant example manifests:

  dataset-adam-table-tennis-example.json
    observation.state.joints : float32 [T, 22] @100Hz  (rad)
    action.joint_targets     : float32 [T, 22] @100Hz  (rad)
    observation.images.front : uint8   [480,640,3] @60Hz  (unused by this
                               joint-only baseline)
  robot-adam-table-tennis-v0
    22 DOF (Booster K1), ball_state[6], control_hz 60

These manifests are `example_only` / `fixture_only` tier. Treat `joints = 22`,
the key names, and the control rate as UNCONFIRMED placeholders. Everything is
parameterised so the real values drop in once the dataset is available — do not
hard-code specs elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DataConfig:
    root: str = "data"                              # dir of LeRobot-v3 parquet episodes
    obs_joint_key: str = "observation.state.joints"  # dataset column for joint state
    action_key: str = "action.joint_targets"         # dataset column for joint targets
    ball_key: str | None = None                     # e.g. "observation.state.ball" -> concat ball_state to obs
    joints: int = 22                                # EXAMPLE value — reconfirm from real data
    ball_dim: int = 6
    # deployment contract signal names (what him_adapter.server receives/emits):
    deploy_obs_signal: str = "robot.joint_position"
    deploy_act_signal: str = "robot.desired_joint_position"
    # synthetic-sample sizing (used only when `root` has no parquet):
    synth_episodes: int = 20
    synth_steps: int = 300


@dataclass
class ModelConfig:
    kind: str = "mlp"                               # "mlp" | "transformer"
    hidden: tuple[int, ...] = (256, 256)
    dropout: float = 0.0
    # transformer-only:
    context: int = 1                                # history window (1 == feedforward)
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
        return d

    @property
    def act_dim(self) -> int:
        return self.data.joints

    @property
    def context(self) -> int:
        return self.model.context if self.model.kind == "transformer" else 1
