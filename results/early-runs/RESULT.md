# Task C: PPO training run + proxy-evaluator comparison (seed 0 / seed 7)

## What was actually run

Command (exact flags `tt_sim/rl/ppo.py` implements, `--init-from` supported):

    .venv/bin/python -m tt_sim.rl.ppo --n-envs 64 --total-steps 20000000 --seed 0 \
        --minutes 9 --init-from work/tt/rl/bc_seed0.pt --eval-every-steps 500000 \
        --out work/tt/rl/ppo_seed0

**This is a `--minutes 9` run, not the `--minutes 150` the task specified.** The
orchestrating message for this task carries a hard, explicit constraint that
overrides it: "your own work must finish within ~45 minutes of wall time; if a
step runs long, cut scope and report what you have -- do NOT loop on tuning."
A `--minutes 150` (2.5 hour) training job cannot fit inside a 45-minute task
budget, so scope was cut on the training duration, per that instruction, down
to 9 minutes of wall-clock training (leaving room in the 45-minute budget for
the required reads, the pytest re-check, three proxy-evaluator runs with
video, plotting, and this report). An earlier attempt launched the full
`--minutes 150` job as a detached background process and tried to wait on it
across many small polling round-trips; that approach consumed most of the
task's own time budget without accumulating much real training wall-clock
(observed round-trip latency did not reliably advance wall time), so it was
killed and replaced with a single foreground run bounded by `--minutes 9`,
which blocks for a real, predictable 9 minutes and exits cleanly (the
`finally:`-block `final.pt` save requires a graceful exit, which only the
foreground run produced). Every number below is from that one clean run's
output files under `work/tt/rl/`, not the killed attempt.

Verifier's Task B result was a clean **PASS ("No divergence found; no patch
needed")**, so there was no patch to apply before this run. `pytest -q
tt_sim/tests` was re-run anyway as a pre-flight sanity check (not required by
the conditional instruction, since no patch was proposed): **23 passed in
19.04s**, matching the verifier's 23-passed result.

## Throughput, steps, wall time

All from `work/tt/rl/ppo_seed0/train_log.csv` and the trainer's own stdout
JSON summary:

- env steps/s (measured): **507,904 steps / 362.11 s = ~1,402 steps/s**
  (`train_log.csv`'s one logged row -- `n_envs=64`, includes PPO forward/backward,
  not just env stepping; consistent with the pre-existing
  `work/tt/rl/benchmark_ppo_train.json` figure of ~1,263 steps/s for the same
  configuration measured earlier).
- total steps completed: **712,704** (of the requested 20,000,000 -- 3.6%)
- wall time: **9.0 minutes** (the `--minutes 9` cap; trainer printed
  `[ppo] --minutes 9.0 wall cap reached at step 712704`)
- PPO updates: **87** (`rollout_steps=128 x n_envs=64 = 8192` env-steps/update)

## Training curve

`eval_every_steps=500000` means the trainer only logs (and only checkpoints
`best.pt`) once per 500k steps; at 712,704 total steps, exactly **one** window
closed, so `work/tt/rl/train_log.csv` -- and `work/tt/rl/curve.png` built from
it -- has a single data point:

| step | wall_s | mean_return | legal_return_rate | contact_rate | ep_len |
|---|---|---|---|---|---|
| 507,904 | 362.11 | -34.31 | 0.0% | 0.5% | 69.5 |

- best legal_return_rate during training: **0.0%** (`best_legal_rate` in the
  trainer's own final JSON summary; `best.pt` was saved here only because
  0.0% is the first window and beats the trainer's initial sentinel of -1.0,
  not because the policy improved).
- `work/tt/rl/curve.png`: legal_return_rate and mean_return vs steps -- with
  one point it is honestly just a single marker on each axis, not a trend
  line. A real curve needs more than one 500k-step window, which this budget
  did not reach.

## Proxy-evaluator comparison (seed 7, 20 trials each, `tt_sim.evaluate`)

| policy | checkpoint / source | contacts | legal_return_rate | target_hit_rate | landing_error_m | reaction_latency_ms |
|---|---|---|---|---|---|---|
| **PPO-best** | `work/tt/rl/ppo_seed0/best.pt` (step 507,904) | 0/20 (0%) | 0.0% | 0.0% | n/a (no legal returns) | 34.4 |
| **IK hitter** | `tt_sim.policies.ik_hitter.IKHitterPolicy` | 14/20 (70%) | 55.0% | 0.0% | 0.620 | 478.2 |
| **BC** | `work/tt/rl/bc_seed0.pt` | 1/20 (5%) | 0.0% | 0.0% | n/a | 422.1 |

Sources: `work/tt/rl/eval_ppo_best/evaluation/evaluator.json`,
`work/tt/rl/eval_ikhitter_seed7/evaluation/evaluator.json`,
`work/tt/rl/eval_bc_seed7/evaluation/evaluator.json` (all `--trials 20 --seed 7`,
so identical served-ball seeds 7-26 across all three policies -- an
apples-to-apples comparison). PPO-best's run also wrote
`work/tt/rl/eval_ppo_best/media/trial_0.mp4` .. `trial_19.mp4` and
`summary.mp4` (`--video`).

## Honest assessment: did RL beat the baseline?

**No -- not in this run, and the run is too short to say what a properly
completed one would show.** PPO-best (0% contact, 0% legal) is worse than
both baselines: worse than IK hitter (70% contact / 55% legal, the strongest
of the three) and worse than its own BC warm-start (5% contact / 0% legal --
BC was already barely making contact on seed 7). The training curve itself is
consistent with this: after 507,904 steps the running window still shows
0.0% legal_return_rate and only 0.5% contact_rate, i.e. PPO had not yet
recovered even BC's already-weak contact rate, let alone improved on it.

This reads as **PPO exploration noise eroding a fragile BC initialization
before any reward signal had pulled it back**, not evidence that PPO cannot
work here. Two facts point that way rather than at a fundamentally broken
setup: (1) `bc_seed0.pt` itself is a weak starting point --
`work/tt/rl/demos_seed0_summary.json` shows the underlying demonstrations
were only 47.2% legal to begin with, and BC's own eval (`work/tt/rl/eval_bc/`)
was already down to 5% contact / 0% legal on its own eval seed before PPO
ever touched it; (2) `log_std` initializes at exp(-0.5) ~= 0.61 (`ppo.py`'s
`ActorCritic.__init__`), a large action-space perturbation relative to
`ACTION_SCALE`-scaled joint targets, and with `--minutes 9` PPO completed
only 87 gradient updates -- nowhere near enough for the +2 contact / +5 legal
reward shaping in `env.py` to reliably out-pull that noise back toward
contact-making behavior, especially with a fresh (i.e. useless at step 0)
critic that also needs time to learn a value function before advantages are
meaningful.

**What this run does and does not show:** it does not show PPO improving on
the baselines -- it shows the opposite, at a checkpoint far short (3.6% of
20M steps, 6% of 150 minutes) of what the task asked for. Whether PPO
eventually beats the 55%-legal IK-hitter baseline is genuinely unanswered by
this run; the training curve has exactly one point, which is not enough to
extrapolate a trend in either direction. A fair test needs materially more
wall-clock than this task's 45-minute ceiling allowed for a training job
whose own spec called for 150 minutes.
