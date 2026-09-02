# Balance v1 — "stand still and swing" (free base, MuJoCo, Mac CPU) — 2026-09-01/02

Design: `docs/design/balance_v1.md`. Code: `himrobot/rl/balance_env.py`, `himrobot/rl/balance_train.py`,
`himrobot/policies/balance_np.py` (commit 2f77b5d, dedupe 71d1a4c).

## Setup

- Scene: `TableTennisEnv(fixed_base=False)`, stance `TT_ROBOT_Y=-0.10`, no physics jitter.
- Expert on waist+arm: shipped v3 `assets/policies/param_residual_v3_s0final.npz`, unchanged.
- RL: 12 leg joints, `BALANCE_SCALE = 0.15` rad, obs 99 + 12 base block (111), MLP 111→128→128→12,
  zero-init head (step 0 == no-balance baseline, verified 1e-8 on 10 serves).
- Reward: VecTT hitting reward (`--vel-penalty 1e-5 --legal-bonus 10`) + 0.05 alive − 0.5·tilt² − 10 on fall.
- Selection: `best.pt` by deterministic legal rate on 40 serves (seed 1001), stand rate as tiebreak.
- Run: 3 seeds × `--n-envs 16 --minutes 480`; scored here at 4.6 h (≈13 000 episodes/seed) from `best.pt`
  (s0 best @ep 9570, s1 @ep 4002, s2 @ep 3915 — s1/s2 had not improved for hours).

Command (per seed):

    TT_ROBOT_Y=-0.10 TT_BALANCE_NPZ=work/tt/rl/balance_v1/s<s>/best_now.npz \
      python -m himrobot.eval.eval_seeds --policy balance_s<s>=himrobot.policies.balance_np:NumpyBalancePolicy \
      --free-base --out work/tt/rl/balance_v1/score_s<s> --jobs 3

## Result (200 serves × seeds 11 / 2000 / 4000 = 600 trials per row, mean action)

| policy | seed 11 | seed 2000 | seed 4000 | legal | target | err | contacts | falls |
|---|---|---|---|---|---|---|---|---|
| free base, no balance (v3 expert alone) | 41.0 % | 49.0 % | 45.0 % | 45.0 % | 5.0 % | 0.61 m | 374/600 (62 %) | 131/600 (21.8 %) |
| balance s0 | 52.0 % | 55.5 % | 49.5 % | 52.3 % | 4.7 % | 0.67 m | 482 (80 %) | 71 (11.8 %) |
| balance s1 | 58.0 % | 61.0 % | 58.5 % | 59.2 % | 5.3 % | 0.66 m | 494 (82 %) | 119 (19.8 %) |
| balance s2 | 62.5 % | 64.0 % | 61.5 % | **62.7 %** | 10.0 % | 0.58 m | 507 (85 %) | 153 (25.5 %) |
| welded v3 (reference, `results/v3_stance`) | 77.0 % | — | — | 77.0 % | 49.5 % | 0.28 m | 88 % | — |

Baseline command: `python -m tt_sim.evaluate_stance --policy tt_sim.policies.param_residual_np:NumpyParamResidualPolicy
--trials 200 --seed <s> --free-base --no-trajectory` (2026-09-01).

## Read

- The leg policy steadies the swing: contacts 62 % → 80–85 %, legal returns +7 to +18 points, all three
  seeds up on all three disjoint serve sets.
- It trades falls for returns. s2 (best legal) falls more than the no-balance baseline; s0 halves falls but
  gains least. Cause: selection by legal rate alone, and a fall penalty (10) equal to the legal bonus (10) —
  the return pays before the fall is charged. `falls` latches anywhere in the 3 s episode, so post-return falls
  count (upper bound), but a fall is a fall on hardware.
- Placement is untouched by design (waist+arm unchanged); target hits stay at the free-base baseline.

## Next (v1.1)

Fall penalty ≥ 30, selection by legal × stand (or legal subject to stand ≥ 95 %), warm start from s2, and
per-trial fall timing in the evaluator so pre- and post-return falls can be told apart.
