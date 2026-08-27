# tt_sim — G1 table-tennis development scene + local evaluator

`tt_sim` is a MuJoCo scene and evaluation harness for the Unitree G1
(29 DoF, fixed pelvis) hitting a served table-tennis ball, built to
develop and locally score policies before submitting them to HIM.

**This is a local proxy, not HIM's evaluator.** `tt_sim.evaluate` writes
`evaluator: "himrobot-local-proxy@0.1.0"` into every `evaluator.json` it
produces. It is a faithful reimplementation of the *published* challenge
scoring (legal-return rate, target-hit rate, landing error, reaction
latency) for fast local iteration — it is **not** HIM's real hidden
evaluator, does not use HIM's holdout seeds, and its authoritative=true
only means "this is the authoritative score for this local run," not an
official HIM Arena score. See `docs/him/llms-full.txt` for how HIM's own
isolated evaluator fits into the real submission pipeline.

## The scene

Built in `tt_sim/scene.py` by string-editing the vendored Menagerie G1
XML in memory (same technique as `sim/g1_model.py`'s paddle injection,
duplicated rather than imported so this package doesn't depend on
`sim/`):

- The pelvis `<freejoint>` is removed and the pelvis body repositioned to
  `x = -1.77 m` (0.4 m behind the robot's own end line), `z = 0.793 m`
  — a fixed-base manipulator that can't fall over. All 29 leg/waist/arm
  joints keep their own actuators and can still move.
- A paddle (welded disc, right-handed) is added to `right_wrist_yaw_link`.
- An ITTF table + net + floor + a free-joint ball are added.

**Table layout** (ITTF facts, as given): 2.74 m x 1.525 m, playing
surface at 0.76 m, net 0.1525 m tall spanning the width. The table is
centred at the world origin: **x** runs along the 2.74 m length (table
spans `[-1.37, 1.37]`, net at `x=0`, robot at the `x<0` end facing `+x`),
**y** along the 1.525 m width (table spans `[-0.7625, 0.7625]`). Ball:
40 mm diameter, 2.7 g. **Assumption** (not an ITTF fact given): ball/table
restitution ≈ 0.9, approximated via a calibrated MuJoCo contact solver
(`solref`/`solimp`) since MuJoCo has no literal restitution parameter —
see the comment block above `BOUNCE_SOLREF` in `tt_sim/scene.py` for the
drop-test numbers behind that choice.

See `tt_sim/scene.py`'s module docstring and `TableTennisEnv` for the
full env/observation/action contract (`reset`/`step`, the `obs` dict
shape, and the `events` dict `step()` returns as `info`).

## Legal-return definition

`legal_return(events)` (in `tt_sim/scene.py`) is true iff, in order:
the paddle hit the ball (`contact_paddle`), the ball then crossed the
net plane above net height heading toward the opponent
(`crossed_net` — the env only ever latches this for a crossing that
happens *after* a paddle hit), and its first subsequent table contact
was on the opponent's half (`bounce_opponent_side`). Equivalently:
`contact_paddle AND crossed_net AND bounce_opponent_side`.

Caveat inherited from `tt_sim/scene.py` (see its docstring): `events`
is a flat per-episode dict, so `events['landing_point']` holds the
*last* recorded table bounce, not necessarily the return's, if the ball
bounces again before the episode ends. `tt_sim/evaluate.py` mitigates
this in practice by keeping each trial's horizon short (`EPISODE_S`,
below) relative to how far a ~0.9-restitution ball travels between
bounces, but does not eliminate it.

## Metrics (`tt_sim/evaluate.py`, written to `evaluation/evaluator.json`)

Per the published challenge scoring (10 trials, one served ball each):

| metric | definition |
| --- | --- |
| `legal_return_rate` | % of trials where `legal_return(events)` is true. |
| `target_hit_rate` | % of trials that are both a legal return **and** land within `TARGET_HIT_RADIUS_M` (0.30 m — this proxy's own assumption, no ITTF/HIM fact gives a hit radius) of the requested `target_zone`/`target_depth` point. |
| `landing_error_m` | mean of `landing_error_m(events, target)` (Euclidean distance, metres) over **legal-return trials only**; `null` if there were none. |
| `reaction_latency_ms` | mean of each trial's `events['reaction_latency_ms']` (ms from serve to the paddle first moving >2 cm) over trials where the paddle moved at all; `null` if it never did. |
| `n_trials` | trials requested. |
| `contacts` | trials where `contact_paddle` was true at all (paddle touched the ball, whether or not the return was legal). |

**Caveat observed while validating with `HoldPolicy`:** a completely
static hold can still show a non-`null` `reaction_latency_ms` — the
G1's position actuators aren't in perfect static equilibrium in
`STAND_POSE` under gravity from a zero-velocity reset, so the paddle can
drift past the 2 cm threshold from settling dynamics alone, with no ball
contact (`contact_paddle` stays `false`, confirmed via
`work/tt/eval/evaluation/evaluator.json`'s `per_trial[*].events`). This
is a real property of the underlying G1 model/actuators in
`tt_sim/scene.py`, not a bug in the metric — treat a hold/near-hold
policy's `reaction_latency_ms` as noise, not signal.

`target_point(target_zone, target_depth)` (`tt_sim/scene.py`) maps the
`target_zone ∈ {line, center, crosscourt}` / `target_depth ∈ {short,
mid, deep}` parameters to an explicit `(x, y)` point on the opponent
half — `depth` 0.35/0.75/1.15 m from the net, `zone` -0.55/0.0/+0.55 m
laterally (`crosscourt`, the default, is the robot's forehand's long
diagonal, `y>0`). This mapping is this repo's own design (not given
numerically in the task); see `tt_sim/scene.py` for the exact table.

## Parameters

`--params` is a JSON object merged over `tt_sim.scene.DEFAULT_PARAMS`:

- `target_zone` ∈ `{center, line, crosscourt}`, default `crosscourt`.
- `target_depth` ∈ `{short, mid, deep}`, default `deep`.
- `risk_preference` ∈ `[0.1, 0.9]`, default `0.45` — recorded into
  `evaluator.json.params` and passed to policies via
  `Policy.reset(episode)`'s `episode["params"]`, but **not used by this
  proxy evaluator itself**: it's a policy-facing knob (how aggressively
  a policy should go for the target vs. play safe), not a scoring input.

## Running it

```bash
# 2-trial smoke check, no video (what tt_sim/tests/test_evaluate.py does headlessly):
.venv/bin/python -m tt_sim.evaluate --policy hold --trials 2 --out work/tt/eval

# Full 10-trial run with video, custom target:
.venv/bin/python -m tt_sim.evaluate --policy hold --trials 10 \
    --params '{"target_zone": "line", "target_depth": "short"}' --video
```

`--policy` is `hold` (the safe do-nothing baseline, `tt_sim.policies.hold.HoldPolicy`),
`ik_hitter` (a shortcut name reserved for `tt_sim.policies.ik_hitter.IKHitterPolicy`
— not implemented by this package; that's a separate task), or an explicit
`module.path:ClassName`.

Output layout under `--out DIR` (default `work/tt/eval`), each
subdirectory overridable independently by the matching `HIM_*_OUTPUT`
env var (the hosted HIM runtime sets these — see
`docs/him/llms-full.txt`):

```
<candidate_dir>/trajectory.jsonl   one JSON object per control step: {trial, step, time_s, obs, action, events}
<evaluation_dir>/evaluator.json    {authoritative: true, evaluator, note, params, metrics, per_trial, wall_time_s}
<media_dir>/trial_<k>.mp4          per-trial video, 640x480 @ 30 fps (only with --video)
<media_dir>/summary.mp4            all trials concatenated (only with --video)
```

Videos (`tt_sim/media.py`) show a broadcast-angle main view with a
diagnostic-camera picture-in-picture inset and a HUD (trial number,
live event flags, running `legal_return_rate`) — both cameras are
plain Python `mujoco.MjvCamera` free cameras, not XML `<camera>`
elements, so `tt_sim/media.py` doesn't need to touch `tt_sim/scene.py`.
Frames are captured on a sim-time schedule (not one-per-control-step),
so the video plays back at approximately real time despite the 50 Hz
control rate vs. 30 fps video rate mismatch.

Each trial runs for `EPISODE_S = 3.0` s (150 control steps at 50 Hz) —
long enough to cover one serve (flight time 0.45-0.75 s, see
`tt_sim.scene.SERVE_FLIGHT_TIME_RANGE`) plus a reaction, swing, and
return flight, with margin; not a cited fact, a chosen trial-horizon
assumption (see `tt_sim/evaluate.py`'s `EPISODE_S` comment, and the
`landing_point` caveat above).

## Tests

```bash
.venv/bin/python -m pytest tt_sim/tests -v
```

`tt_sim/tests/test_scene.py` (from the scene-building task) covers the
physics/env contract; `tt_sim/tests/test_evaluate.py` covers the CLI:
runs 2 trials with `HoldPolicy` headlessly (no `--video`, so the tests
stay fast) and checks `trajectory.jsonl` / `evaluator.json` exist and
are shaped right, `authoritative is True`, the proxy's `evaluator` id
is present, and — since `HoldPolicy` never swings and can never
register `contact_paddle` — `legal_return_rate == 0.0` and
`contacts == 0`.

## Runtime interface contract

`.him/contracts/tt-runtime.json` declares this task's observation/action
signals (`robot.joint_position[29]`, `robot.joint_velocity[29]`,
`ball.position[3]`, `ball.velocity[3]`, `paddle.position[3]`,
`paddle.quaternion[4]` in, `robot.desired_joint_position[29]` out, 50 Hz)
against the HIM SDK's `runtime-interface.schema.json`. It is a sibling
of `.him/contracts/runtime.json` (the inference-entrypoint contract),
not a replacement for it — `runtime.json` describes the free-floating
inference interface; `tt-runtime.json` describes this package's
fixed-base table-tennis evaluation scene, which adds the ball/paddle
signals and is `stateful: true` (`TableTennisEnv` carries per-episode
event state across steps, unlike the stateless inference server).

`work/tt/him.yaml.patch` is a proposed (**not applied**) unified diff for
`.him/him.yaml`, adding an `evaluate` entrypoint (`purpose: evaluation`,
pointing at `tt-runtime.json`), registering `tt-runtime.json` under
`contracts.runtime_interfaces`, and adding a `tt-sim` test entry running
`tt_sim/tests`. Before writing it out, a patched copy of `.him/him.yaml`
was validated end-to-end against the HIM SDK's `him-package.schema.json`
with `jsonschema` (in the `him` venv, which has it; the project `.venv`
does not). It is intentionally left unapplied — another task owns
`.him/him.yaml`.
