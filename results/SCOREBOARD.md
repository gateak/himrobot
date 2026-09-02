# Scoreboard — policy lineage

One row per policy that was ever the "current best" (or is a named baseline it was measured
against). Every number below is copied from `docs/PROJECT_KNOWLEDGE.md` §2 or from a
`results/<run>/RESULT.md` file in this ledger — nothing here was computed for this table.
Numbers come only from the deterministic evaluator (`tt_sim.evaluate` / `tt_sim.rl.eval_seeds`,
mean action), per `CLAUDE.md`. Two serve sets appear below: seeds 11/7/3 (used through
2026-08-29; later found to share 192–196 of 200 serves, so three "independent" numbers on that
set are one set measured three times) and seeds 11/2000/4000 (disjoint, the scoreboard from
2026-08-29 onward).

| # | Policy | Scene | Serve set | Legal % | Target % | Landing err | Falls | Source |
|---|---|---|---|---|---|---|---|---|
| 1 | Hitter (scripted IK, no RL) — `ik_hitter.py` | welded (fixed pelvis) | 11/2000/4000, 600 trials | 45.3 % | 2.3 % | 0.74 m | — | `docs/PROJECT_KNOWLEDGE.md` §2 (free-base paragraph, "45.3 % welded"); `results/v3_stance/RESULT.md` (re-score table, row `plain ik_hitter @ y = 0.00 (scoreboard)`) |
| 2 | Param residual v1, seed 3 — `assets/policies/param_residual_s3.npz` | welded | 11/7/3 (overlapping set), 600 trials | 70.5 / 70.5 / 71.0 (per-seed) | 10.0 % | 0.52 m | — | `docs/PROJECT_KNOWLEDGE.md` §2 ("Best policy (2026-08-29)"); `results/param/RESULT.md` |
| 3 | Param residual v2, `wa0/final` — `assets/policies/param_residual_v2_wa0.npz` | welded | 11/2000/4000, 600 trials | 73.5 % | 28.5 % | 0.39 m | — | `results/v3_stance/RESULT.md` (re-score table, row `param_residual_v2_wa0 @ y = 0.00 (scoreboard)`); `docs/PROJECT_KNOWLEDGE.md` §2 |
| 4 | Param residual v3, `s0/final` (shipped) — `assets/policies/param_residual_v3_s0final.npz`, behind `tt_sim.evaluate_stance` (`TT_ROBOT_Y=-0.10`) | welded, stance shifted −0.10 m in y | 11/2000/4000, 600 trials | **77.0 %** | **49.5 %** | **0.28 m** | 0/0/0 | `results/v3_stance/RESULT.md` ("Retrain scoreboard" table, row `s0/final`); `docs/PROJECT_KNOWLEDGE.md` §2 ("Hitter v3 retrain + ship") |
| 5 | Param residual v3, `s0/final` — free base, no balance controller | free base (pelvis unwelded), stance shifted | 11/2000/4000, 600 trials | 45.0 % | 5.0 % | 0.61 m | 21.8 % | `results/balance_v1/RESULT.md` (baseline row, measured 2026-09-01) |
| 6 | Param residual v3, `s0/final` — HIM Arena entry (`him upload`) | welded, stance shifted | seed 2000, 1,000 serves | 78.6 % | 47.0 % | 0.288 m | — | `docs/PROJECT_KNOWLEDGE.md` §2 ("ENTERED (2026-09-01)" — also 88.3 % contact, 452 ms reaction) |
| 7 | Balance v1 s2 `best.pt` @4.6 h — leg-balance residual over the v3 expert (`work/tt/rl/balance_v1/s2/best_now.npz`) | free base, stance shifted | 11/2000/4000, 600 trials | **62.7 %** | 10.0 % | 0.58 m | 25.5 % | `results/balance_v1/RESULT.md` (s0: 52.3 % / 11.8 % falls, s1: 59.2 % / 19.8 %) |

## Reading this table

- Rows 1–4 are the same fixed-pelvis ("welded") scene the leaderboard cares about today; row 4 is
  the current best on it, and is what ships behind `evaluate`/`evaluate_stance`.
- Row 5 is the free-standing scene with **no balance controller** — position actuators just hold
  the stand pose while the hitter swings. It exists to size the cost of unwelding the pelvis
  before a balance controller is added (see `docs/design/balance_v1.md` §"why").
- Row 6 is the number HIM's own listing shows the public (the 1,000-serve entry), not a
  three-seed scoreboard — kept separate because it is a different trial count and serve seed.
- Row 7 is not a result yet — it is training now; this row exists so the ledger records that a
  run is in flight and unscored, per `CLAUDE.md`'s "numbers come only from the deterministic
  evaluator" rule (a training-time reward curve does not count, scored or not).
| 8 | Param residual v3 `s0/final` — free base, **honest scene** (`robot_table_collision=false`), no balance | free base, stance shifted | 11/2000/4000, 600 trials | 61.5 % | 4.8 % | 0.69 m | **75.7 %** | `results/balance_v1_2/RESULT.md` |
| 9 | Balance v1.2 c (0.40 authority, fresh) — best fall reduction of 4 runs | free base, honest scene | 11/2000/4000, 600 trials | 39.2 % | 4.0 % | 0.68 m | 60.3 % | `results/balance_v1_2/RESULT.md` (a: 56.7 % / 73.7 % falls) |
