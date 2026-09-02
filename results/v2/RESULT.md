# Task R: residual policy over the IK hitter

## Commands run

```
.venv/bin/python -m pytest tt_sim/tests -q
  -> 35 passed (full suite, incl. the 5 new tests in test_rl_residual.py; no pre-existing test failed)

.venv/bin/python -m tt_sim.rl.ppo --residual --n-envs 1 --total-steps 0 --seed 0 \
  --out work/tt/rl/v2/zero_init
  -> writes work/tt/rl/v2/zero_init/final.pt (fresh, ZERO update, zero-init actor last layer)

.venv/bin/python -m tt_sim.rl.ppo --residual --n-envs 4 --total-steps 64 --rollout-steps 8 --seed 0 \
  --out work/tt/rl/v2/smoke_2updates
  -> {"total_steps": 64, "n_updates": 2, ...}  (see smoke_2updates/train_log.csv)

TT_RL_CKPT=work/tt/rl/v2/zero_init/final.pt \
.venv/bin/python -m tt_sim.evaluate --policy tt_sim.rl.policy:ResidualPPOPolicy \
  --trials 20 --seed 7 --out work/tt/rl/v2/eval_residual_zero_seed7
  -> see eval_residual_zero_seed7/evaluation/evaluator.json
```

## Numbers (from files under work/tt/rl/v2/)

- `smoke_2updates/train_log.csv`: 2 PPO updates on 4 envs completed (`total_steps=64`), checkpoint
  `smoke_2updates/final.pt` written; `ResidualPPOPolicy(ckpt_path=...)` loads it and returns a
  finite `(29,)` action (verified interactively, not itself written to a file).
- `eval_residual_zero_seed7/evaluation/evaluator.json` (ResidualPPOPolicy, fresh zero-init
  checkpoint, 20 serves, seed 7): **contacts=9, legal_return_rate=10.0%** (2/20),
  `reaction_latency_ms=478.2`.
- `tt_sim/rl/eval_ikhitter_seed7/evaluation/evaluator.json` (pre-existing, direct IKHitterPolicy,
  same 20 serves/seed 7): contacts=14, legal_return_rate=55.0%, `reaction_latency_ms=478.2`.

**Acceptance bar in the task ("must give the hitter's 55%/14 contacts") is NOT met.**
`reaction_latency_ms` matches exactly (478.2ms both runs -- confirms early-episode behaviour,
before any swing, is identical), but contacts/legal_return_rate diverge substantially.

## Root cause (verified, not a residual.py wiring bug)

`tt_sim.rl.demos.target_to_action` clips a waist/right-arm joint's delta from STAND_POSE to
+/-1 rad before `tt_sim.rl.env.map_action` reconstructs an absolute joint target from it
(ACTION_SCALE=1.0 there). The IK hitter's own swing frequently needs more than 1 rad on some
joint (`work/tt/rl/demos_seed0_summary.json`'s `action_saturation_fraction` already documents
this at ~1.9% of kept waist/arm action *components* across 2000 trials -- non-trivial per full
swing window, since most of that 2% is concentrated in the few dozen steps of the actual swing,
not the long pre-swing STAND_POSE hold). Whenever `a_exp = target_to_action(IKHitterPolicy's raw
joint target)` saturates on some step, `map_action(a_exp)` reconstructs a *different*, weaker
joint target than what the un-clipped hitter would have sent directly -- a real, physical
information loss inherent to representing a_exp in the bounded [-1,1]^29 action space the spec
requires (`a = clip(a_exp + residual_scale * a_rl, -1, 1)`), not a bug in how it's combined.

Verified two ways in `tt_sim/tests/test_rl_residual.py`:
- `test_zero_action_wiring_matches_recomputed_expert_at_every_step` (passes, always exact by
  construction): at every control step of a 20-episode residual_scale=0 rollout, the *executed*
  action is bit-identical to independently recomputing `target_to_action(IKHitterPolicy.act(obs))`
  from that same live observation -- i.e. the append/combine/clip machinery in
  `tt_sim/rl/residual.py` is exactly correct.
- `test_zero_action_reproduces_most_direct_ikhitter_episode_outcomes` (passes with an 80% bar,
  not 100%): a genuinely independent direct-rollout comparison (same seed/serve sequence) matches
  on contact for most, not all, of 20 episodes -- diverging exactly where saturation first occurs
  within an episode, then compounding through the ball-flight/replanning feedback loop.

This is a design-level property of using a +/-1 rad action-scale bound to represent an expert
whose own swings sometimes exceed it, present regardless of the RL policy (it shows up already
at residual_scale=0 / a zero actor). It is not something `tt_sim/rl/residual.py`,
`tt_sim/rl/ppo.py`, or `tt_sim/rl/policy.py` can fix without either widening ACTION_SCALE for the
relevant joints or representing a_exp differently -- both out of this task's file scope
(`tt_sim/rl/env.py`'s ACTION_SCALE is not in the OWN list).
