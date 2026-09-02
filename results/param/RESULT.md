# Per-serve parameter residual — first result (2026-08-29)

Recipe (Mac, no GPU, no credits):
`python -m tt_sim.rl.param_train --n-envs 32 --minutes 45 --seed {0,1} --eval-every 200 --vel-penalty 1e-5 --legal-bonus 10`
≈29 k episodes per seed in 45 min. Policy: `tt_sim.rl.policy:ParamResidualPolicy` (K = 10 bounded offsets on the IK
hitter, chosen once per serve from the serve's initial ball state and the hitter's predicted intercept; u = 0 is the
plain hitter). Checkpoints: `work/tt/rl/param/s{0,1}/{best,final}.pt`.

## Deterministic evaluation, 200 serves per set

| serve set | IK hitter | per-step residual (best, v3 s32) | param residual s0 | param residual s1 |
|---|---|---|---|---|
| seed 11 | 46.5 % | 53.5 % | 67.5 % (best) / 67.0 % (final) | 68.0 % / 68.0 % |
| seed 7 | 46.0 % | – | – | 67.5 % |
| seed 3 | 46.5 % | – | – | 68.0 % |

Contacts 153–157 / 200 vs the hitter's 137–139. Target-hit rate is still ~0.5–2 % (hitter 3 %): the aim term
(+3·exp(−err²/0.5²)) is small next to the +10 legal bonus, so the bandit learns "get it over and in", not placement.
Landing error unchanged (0.72 m vs 0.71 m).

## Why it works where the per-step residual stalled

- One decision per serve: exploration noise perturbs 10 numbers once, not 29 joints 50×/s, so the plain hitter's
  47 % is not destroyed during training and every sample is informative.
- No temporal credit assignment; the return of the episode is the reward of the single action.
- `best.pt` ≈ `final.pt` on both seeds: no drift between the noisy training objective and the deterministic score.

## 120-minute seeds (same recipe, `--minutes 120`, ≈79 k episodes each)

| seed | legal (seed-11 set) | target hit | contacts |
|---|---|---|---|
| s2 best.pt | 69.0 % | 3.5 % | 156 |
| s3 best.pt | **70.5 %** | **10.0 %** | 154 |

Gain over 45 min is modest (+1–2.5 pts legal), but placement starts to appear (s3 10 % target hits vs hitter 3 %).
s3 confirmed on unseen sets: seed 7 → 70.5 % / 10.0 % / landing error 0.52 m; seed 3 → 71.0 % / 10.0 % / 0.52 m
(hitter 46.0–46.5 % / 3 % / 0.71 m). `work/tt/rl/param/s3/best.pt` is the current best policy.
Train-time (noisy) legal rate 65 % at the end for both.

## Next

1. Longer runs: 45 → 120 min gave +1–2.5 pts; the curve is flattening around 70 % under this reward. Diminishing
   returns from time alone — the next gains are reward (placement) and features, not hours.
2. Placement: raise the aim reward (add an `aim_bonus` knob like `legal_bonus`) or make the target part of the features
   so `target_hit_rate` moves.
3. Generalization: train with `serve_scale` 1.2–1.5 and `physics_jitter` 0.1, evaluate on nominal and widened sets.
4. Package: wire `ParamResidualPolicy` into `him_adapter` as the `inference` brain (needs ball position/velocity in the
   runtime contract — confirm with HIM what the official evaluator streams).
