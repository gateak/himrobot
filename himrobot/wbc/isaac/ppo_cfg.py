"""rsl_rl PPO runner config for ``Himrobot-WBC-G1-v0``, docs/design/
whole_body_v1.md sec 2 ("From BeyondMimic ... the PPO hyperparameters of
config/g1/agents/rsl_rl_ppo_cfg.py") and sec 3.2 (zero-init final actor
layer -- the parity anchor). Requires isaaclab_rl/rsl_rl (real or stubbed,
see tests/test_wbc_isaac_stub.py); unverified against a real Isaac Lab
session, same caveat as env_cfg.py.
"""
from __future__ import annotations

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab.utils import configclass

EXPERIMENT_NAME = "wbc_v1"
SAVE_INTERVAL = 500
NUM_STEPS_PER_ENV = 24
MAX_ITERATIONS = 30000


@configclass
class HimrobotWBCPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env: int = NUM_STEPS_PER_ENV
    max_iterations: int = MAX_ITERATIONS
    save_interval: int = SAVE_INTERVAL
    experiment_name: str = EXPERIMENT_NAME
    empirical_normalization: bool = True
    policy: RslRlPpoActorCriticCfg = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    # Not a stock rsl_rl/isaaclab_rl field -- consumed by scripts/train.py's
    # own zero_init_final_actor_layer() after the runner is constructed (see
    # its docstring). Recorded here so the parity-anchor knob lives next to
    # the rest of the PPO config rather than as a bare script constant.
    zero_init_final_actor_layer: bool = True
