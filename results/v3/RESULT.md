# v3 residual PPO — result (2026-08-28)

Machine: HIM SSH box `ssh_ee947b04aebe10238757` (L4, 128 cores, 51.2 GB cgroup cap), expired 03:18 UTC.
Recipe: `python -m tt_sim.rl.ppo --residual --residual-scale 0.2 --vel-penalty 1e-5 --legal-bonus 10 --n-envs 64 --minutes 90`
(commit ff83138). Seeds 30–33 launched 01:37 UTC; s30/s33 were OOM-killed at launch (4 × 10.8 GB + 4 intercept runs
> 51.2 GB), s31/s32 ran ~85 min (2.34 M steps) until the machine expired. Checkpoints pushed to HIM as
`himrobot-rl-checkpoints@0.1.29798103` (`dataset_d472ede26cc39ef603f2`), local copy `work/tt/rl/box/v3/`.

## Why v2 failed (resid_s0–3, icept_s20–23)

The per-step joint-velocity penalty (0.0005·Σv²) cost a fast swing ≈2.5 per episode, more than the +5 legal bonus.
PPO raised the return from −6.8 to −2.2 by swinging slower: contact rate rose 62 → 81 % while legal returns fell
47 → 3 %. Cold intercept-shaped PPO (icept_*) stayed at 0–1 % legal after 5.7 M steps.

## Deterministic evaluation (tt_sim.evaluate, seed 7, 40 serves, no exploration noise)

| policy | legal return | target hit | contacts | reaction |
|---|---|---|---|---|
| IK hitter (baseline) | 42.5 % | 0 % | 26/40 | 522 ms |
| resid_s31 best.pt | 40.0 % | 15 % | 29/40 | 35 ms |
| resid_s32 best.pt | 42.5 % | 5 % | 27/40 | 24 ms |

Train-time legal rate (with exploration noise) was still rising at cut-off: s32 22 %, s31 14.5 %; mean return 2.3 → 3.1.
200-serve evaluation (seed 11): see `eval200_*/evaluation/evaluator.json`.

## Read

- The residual matches the hitter on legality after 80 min and adds target hits the scripted hitter never gets.
- Exploration noise alone drops train-time legal rate to ~10 %: per-step joint deltas are a poor action space for a
  precision-contact task. Next recipe: residual over hitter *parameters* (swing timing, paddle angle, aim) chosen
  once per serve, or a much smaller log_std, plus longer runs on a 48 GB box (≤4 residual runs per 24 GB).
