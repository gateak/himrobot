# Balance v1 — "stand still and swing" (free base, MuJoCo, local CPU)

Goal: recover the ~20 points lost by unwelding (73.5 % → 54.0 % legal, 19.3 % falls for
`param_residual_v2_wa0`; v3 not yet measured free-base) with a small per-step leg policy,
while the shipped v3 hitting stack runs unchanged on waist+arm. Target: stand rate > 95 %,
legal returns ≥ 70 % on the free-base scene, seeds 11/2000/4000.

## Architecture

`VecTTBalance(VecTT)` in `tt_sim/rl/balance_env.py`, modeled line-for-line on
`tt_sim/rl/residual.py`'s expert-rides-along pattern:

- Envs: `TableTennisEnv(seed=…, params=…, fixed_base=False)`. Stance: whatever
  `TT_ROBOT_Y` is at import (training runs set −0.10 via `tt_sim.evaluate_stance`'s
  mechanism or the env var; the env code itself reads nothing new).
- Expert: one **shipped v3 policy** per env slot — `NumpyParamResidualPolicy` (weights
  `assets/policies/param_residual_v3_s0final.npz`) — reset per episode like
  `VecTTResidual._hitters`, acting on the raw obs dict each step. Its 29-dim action
  `a_exp` (via `target_to_action`) supplies waist+arm.
- RL action `a_rl` ∈ [−1,1]^12 (legs only, JOINT_ORDER slots 0–11). Joint targets:
  `target = STAND_POSE + concat(a_rl * BALANCE_SCALE, a_exp[12:] * 3.0)`, clipped to
  joint ranges. `BALANCE_SCALE = 0.15` rad. **a_rl = 0 reproduces today's free-base
  probe behaviour exactly** (legs hold STAND_POSE; ACTION_SCALE's leg zeros made the
  expert's leg slots inert already) — that is the parity anchor.
- Obs (`BALANCE_OBS_DIM = 111`, `obs_version = 1`): `flatten_obs`'s 99, then a
  12-dim base block: gravity vector in the pelvis frame (3, from `base.orientation`),
  `base.position − scene.BASE_NOMINAL_POS` (3), `base.linear_velocity` (3),
  `base.angular_velocity` (3).
- Reward: VecTT's `_reward_and_done` on the combined action, plus
  `+0.05` alive per step, `−tilt_penalty * (tilt angle rad)²` per step
  (tilt_penalty 0.5), and on the first step `events["fell"]` latches: `−10`, done.
- Termination: VecTT's rules, plus done when `fell` latches.

## Trainer

`tt_sim/rl/balance_train.py`, mirroring `param_train`'s loop/logging/selection:
PPO (reuse `nets.py` MLP actor-critic sizes 111→128→128→12), log-std init −1.6
annealed to −2.5, `--det-eval-episodes 40 --det-eval-every 20` (mean action, fixed
serves, seed 1001) logging `det_legal_rate` and `det_stand_rate`; `best.pt` = highest
det legal rate. `train_log.csv` as in param_train. `--minutes` wall clock bound.

## Deployable policy

`tt_sim/policies/balance_np.py`: `NumpyBalancePolicy` wrapping the v3 stack + the
balance MLP (npz export like `param_residual_np`), for `tt_sim.evaluate --free-base`
and `eval_seeds`. Torch-free.

## Parity

- Welded scene untouched: no default changes anywhere; the 144-test suite must pass.
- New golden check: a_rl=0 rollout on one fixed serve equals the plain free-base
  rollout of the v3 policy (same TT_ROBOT_Y), bit-for-bit.

## Measurement (definition of done)

`tt_sim.rl.eval_seeds --free-base` (or evaluate --free-base × 3 seeds), 200 serves ×
seeds 11/2000/4000, mean action, at TT_ROBOT_Y=−0.10: legal %, target %, error, falls —
against the v3-free-base-no-balance baseline (to be measured first) and welded 77.0 %.
Result to `work/tt/rl/balance_v1/RESULT.md`.
