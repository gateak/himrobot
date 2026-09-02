# hitter v3 — reach

Design. No source file changed; everything below is measured today (2026-08-30) from scratch
scripts in `work/tt/analysis/reach/`, and every number carries the command that produced it.

Inputs: the three measurement reports in that directory — `ceiling.md` (A, the reach ceiling),
`solver.md` (B, the deep-plane self-collision), `taxonomy.md` (C, the failure taxonomy) — plus
nine new experiments run for this design, listed in §5.

Baseline being improved on: `assets/policies/param_residual_v2_wa0.npz`, mean action,
**73.5 % legal / 28.5 % target / 0.39 m** on seeds 11 / 2000 / 4000 (200 serves each). The plain
hitter is 46.5 / 47.0 / 42.5 % on the same sets.

---

## 0. The two numbers that shape this design

**One.** Inside the machine there are **7.3 points** and they are not reach. 79.3 % of the 600
scoreboard serves admit a collision-free blade placement somewhere on the flight (A); the shipped
policy returns 90.8 % of those and 7.3 % of the rest, which reproduces its 73.5 % exactly. The
ceiling is **80.8 %**. Every one of the 57 trials the reach clamp cuts is *also* geometrically
unreachable (0 clamped-but-reachable, measured below) — so the clamp costs nothing, and the
remaining 7.3 points are aim and timing on balls the arm can already touch.

**Two.** Nine uniform knob directions were flown through the real evaluator on the shipped policy.
**None of them buys legal returns.** The best is a 5 % slower swing at +1.25 points, for −10.25
points of target hits. The hitter's own parameter space is exhausted.

So the lever is not in `ik_hitter.py`'s parameters. It is **where the robot stands**:

> Sliding the welded pelvis **0.10 m toward −y** takes the **shipped, unretrained** policy from
> **73.50 % → 77.83 %** legal and **80.0 % → 87.8 %** paddle contact on the full 600-serve
> scoreboard (+4.33 points; 64 trials gained, 38 lost; McNemar exact two-sided **p = 0.013**).
> All three disjoint seeds improve. It is one line in `tt_sim/scene.py` plus a stance-relative
> reach frame in `tt_sim/policies/ik_hitter.py`.

That is more than the whole in-hitter headroom, and it arrives before any retraining.

---

## 1. The ceiling on the scoreboard sets, and what is actually available

### 1.1 Joining A's geometry to C's per-trial outcomes

A measured reachability per serve; C measured the failure class and the reach-clamp flag per
trial. They index the same trials (serve *k* of seed *S* is `default_rng(S + k)`), so they join.

```
OMP_NUM_THREADS=1 PYTHONPATH=. ./.venv/bin/python work/tt/analysis/reach/v3_join.py
```

400 trials (seeds 2000 / 4000), 4 cm criterion:

| population | n | legal | contact |
|---|---|---|---|
| geometrically reachable | 318 (79.5 %) | **90.9 %** | 97.5 % |
| geometrically unreachable | 82 (20.5 %) | 6.1 % | 15.9 % |
| **ceiling** (every reachable serve returned) | — | **80.8 %** | — |

matching A's 600-serve figure to the decimal.

**The load-bearing new fact: the reach clamp and geometric unreachability are the same 57 trials.**

| | n | legal |
|---|---|---|
| clamp-bound (C's definition: the committed plan's y was cut) | 57 | 0 |
| geometrically unreachable (A's definition: no pose within 4 cm anywhere on the flight) | 82 | 5 |
| both | 57 | 0 |
| **clamped but geometrically reachable** | **0** | — |
| unreachable but never clamped | 25 | 5 |

`Y_REACH`'s floor is not throwing away a single reachable ball. Widening it is worth exactly zero,
which §2.7 then confirms by flying it.

### 1.2 What the 7.3 points are made of

29 of the 400 trials are reachable and still missed — 7.2 points:

| | n | share |
|---|---|---|
| out_long (contact, ball past the baseline) | 14 | 48 % |
| no contact, never clamped ("in-reach whiff") | 8 | 28 % |
| net | 4 | 14 % |
| out_wide | 3 | 10 % |

21 of the 29 made clean contact. The 8 whiffs are late — committed contact time **0.995 s** against
**0.784 s** for the trials that do connect — which is C's "last 10 cm before the edge" bucket.

Two framings worth keeping:

* **Contact → legal is already 91.0 %** (294 legal of 323 contacts over the 400 trials). Once the
  paddle touches the ball, this policy converts. The lever is contacts, not returns.
* **Contacts (80.0 %) already sit at the 4 cm geometric ceiling (80.8 %)**, because the blade is
  8.5 cm wide and the swing sweeps. The 4 cm line is a proxy, not a wall.

### 1.3 A correction to the record: the plane the policy uses is not −1.20

`x_hit_eff` — the plane — has median **−1.185** (q10 −1.268, q90 −1.125). But `p_hit` is re-sampled
`pred_delay + t_hit_offset` further along the same predicted flight, and the ball keeps moving, so
the blade target's actual x has median **−1.308** (q10 −1.405, q90 −1.222). Every "at the plane the
policy already uses, x ≥ −1.20" statement in A understates the operative depth by ~12 cm. The
policy is already working at A's −1.28…−1.40 rows, not its −1.20 row.

---

## 2. Ranked changes

Gain is in **points of legal return on the 600-serve scoreboard** (seeds 11 / 2000 / 4000). "pts"
throughout = percentage points of that metric. 1 s.e. on 600 serves is ≈1.8 points unpaired; the
paired (same-serve) comparisons quote McNemar.

| # | change | measured gain | cost | risk |
|---|---|---|---|---|
| **1** | **welded pelvis −0.10 m in y** + stance-relative reach frame | **+4.33 pts** legal, **+7.8 pts** contact, +2.0 pts target — *unretrained*, p = 0.013 | 1 line `scene.py`, ~30 lines `ik_hitter.py`, both gated; ~1 h Mac | over-fits a forehand-only serve set; HIM may not let us choose the stance; a different evaluated system |
| **2** | retrain `wa0` at the new stance (warm start) | est. **+1 to +3 pts** on top (38 of the 102 discordant trials were *lost*, i.e. mis-aimed for the new geometry) | ~1 h Mac, no credits | none beyond ordinary training noise |
| **3** | reward the landing *depth* shortfall, retrain (C's recommendation) | **~0 pts legal**, est. **+5 to +10 pts target** | ~1 h Mac, same run as #2 | placement already saturated once (`param_v2_phase2`); the new lever is `swing_speed_scale`, not the aim knobs |
| **4** | `collision_guard` knob, default off | **0 pts** at both stances (466 vs 467 at −0.10; 293 vs 294 at today's stance) | ~25 lines behind a knob + 1 test | none if gated; costs 8.5 pts of *target* hits when forced on |
| **5** | prune the action space to the ~5 knobs that carry signal | 0 directly; buys exploration efficiency for #2 | spec edit + a v3 space | a v3 space is another decode path to keep parity on |
| 6 | deeper hitting plane (`x_hit_coarse`) — A's free runner-up | **−10 to −47 pts**: rejected | — | — |
| 7 | widen `Y_REACH` / `reach_expand_y` | **exactly 0** (bit-identical outcomes): rejected | — | — |
| 8 | more DLS iterations (20 → 40 → 60) | **0** (294 vs 294 legal on 400): rejected | — | — |
| 9 | lower face weight `w_normal` 8 → 0.5 | +1 trial of 200 (noise): rejected | — | — |
| 10 | swing window ×1.4 / ×0.8 | **−8 to −25 trials of 200**: rejected | — | — |
| 11 | `swing_speed_scale` ×0.95 | +1.25 pts legal for **−10.25 pts target**: a dial, not a change | — | — |
| 12 | waist roll/pitch limits ×2 | +3.2 pts *reachability*, 0 of 124 at the working plane (A) | — | a hardware claim: ±0.52 rad is the vendor G1 limit |
| 13 | yaw the welded pelvis ±0.35 rad | **+0.0 pts** (A): `waist_yaw` is never saturated | — | — |

### 2.1 #1 — where the robot stands

```
OMP_NUM_THREADS=1 PYTHONPATH=. ./.venv/bin/python \
    work/tt/analysis/reach/v3_stance_task.py <dy> <seed> policy
```

`v3_stance_task.py` wraps `scene.build_model` so both the env's model and the hitter's shadow model
come from an XML whose pelvis line is slid by `dy` (`reach_lib.variant_model`), and slides
`Y_REACH`, `REACH_MAX_Y` and every template's y box by the same `dy` — because those constants were
measured against a pelvis at y = 0 and only mean anything relative to where the robot stands.
`tt_sim/` is not modified. **`dy = 0` reproduces the scoreboard exactly** (policy 147 / 153 / 141
legal, hitter 93 / 94 / 85), which is the harness's own proof.

Shipped policy, **no retraining**, 200 serves per seed:

| pelvis dy | seed 11 | seed 2000 | seed 4000 | legal / 600 | legal % | contacts % | target % |
|---|---|---|---|---|---|---|---|
| **0.00** (today) | 147 | 153 | 141 | 441 | **73.50** | 80.0 | 28.5 |
| −0.05 | 157 | 159 | 150 | 466 | 77.67 | 84.5 | 28.8 |
| **−0.10** | **161** | 154 | 152 | **467** | **77.83** | **87.8** | **30.5** |
| −0.15 | 158 | 152 | 153 | 463 | 77.17 | 89.2 | 28.5 |
| −0.20 | 152 | 145 | 153 | 450 | 75.00 | 87.8 | 30.2 |
| −0.30 | — | 129 | 118 | — | — | — | — |

A broad plateau at −0.05…−0.15 and a clear fall-off past −0.20. Paired significance at −0.10
(`v3_st_policy_dy*.json`, McNemar over the same 600 serves):

| seed | legal | gained | lost | contacts |
|---|---|---|---|---|
| 11 | 147 → 161 | +23 | −9 | 157 → 176 |
| 2000 | 153 → 154 | +17 | −16 | 168 → 174 |
| 4000 | 141 → 152 | +24 | −13 | 155 → 177 |
| **all** | **441 → 467** | **+64** | **−38** | **480 → 527** |

z = 2.57, exact two-sided p = **0.013**.

**Read it honestly.** Contacts rise 47 (+7.8 pts) and legal returns only 26 (+4.3 pts): the
conversion on the *new* contacts is 26/47 ≈ 55 %, against 91 % on the ones the policy was trained
for. And 38 trials that used to come back no longer do. Both facts say the same thing — the swing,
aim and timing are tuned for the old geometry — and both are what a retrain (#2) is for.

Also honest: **A's static sweep does not predict this.** A ranked −0.30 m best (97.8 % reachable);
flown, −0.30 is the worst setting tested. Static reachability says a pose exists, not that the arm
gets there in time with a face angle that returns the ball. That gap is measured again in §2.6.

Why it works, mechanically: `DEFAULT_PARAMS` serves land at `target_y ∈ [−0.65, −0.10]` and the
misses cluster at far-forehand arrival. Moving the robot toward that corner converts the far
forehand into the middle of the workspace. Static confirmation, on the band the balls actually
occupy (y ∈ [−0.85, +0.05] × z ∈ [0.90, 1.35], 5 face normals, 950 poses, the test's own
`_ik_pose_at` recipe):

```
OMP_NUM_THREADS=1 PYTHONPATH=. ./.venv/bin/python work/tt/analysis/reach/v3_stance.py <x> <dy>
```

| plane | dy = 0 · frac < 4 cm | dy = −0.10 | dy = −0.20 |
|---|---|---|---|
| x = −1.20 | 0.512 | 0.609 | 0.647 |
| x = −1.31 (the blade's real depth) | 0.675 | 0.707 | 0.717 |

**Caveat that must travel with this change.** It is free *because every serve is on the forehand
side*. On a symmetric or backhand-inclusive distribution it would cost as much as it gains, and
there is no left-arm backhand (the paddle is welded to the right wrist). Whether HIM's evaluator
lets us choose the base pose at all is an open question (`docs/PROJECT_KNOWLEDGE.md` §2). Treat
−0.10 m as tuned to *this* serve distribution and re-tune it if the distribution changes; that is
exactly why §3 puts it behind a named constant rather than a new literal.

### 2.2 #2 — retrain at the new stance

The 38 lost trials and the 55 % conversion on new contacts are the measurable headroom. Warm-start
from `wa0` so the run starts from a policy that already works, and re-measure the ceiling first
(A's sweep at `--pelvis-dy -0.10` takes ~5 min and gives the number to be scored against). Recipe
in §4.

### 2.3 #3 — price the landing depth, not the aim point

C's finding stands and this design does not improve on it: every legal return lands 0.276 m short
and 0.214 m inboard, a bias 1.2–1.5× its own scatter, identical on two disjoint sets; the aim knobs
move it by 3 % of the requested amount because the swing-speed cap binds on 400/400 trials.
Confirmed here by the probe table in §2.6: `swing_speed_scale` ×0.95 moves the depth bias from
−0.276 m to −0.341 m over 400 serves (C measured ×1.05 → −0.236 / −0.228 m per seed, in the other
direction), while the two aim knobs move it by 3 cm of a requested 27.6 cm.

The one correction: `work/tt/rl/param_v2/RESULT.md`'s phase 2 already tried a bigger aim reward
with target-first selection and saturated at 28–30 %. So do not simply raise `--aim-bonus`. Add a
term that prices the *depth* shortfall specifically, in `VecTT._reward_and_done` next to the
existing `aim_bonus` exp-kernel, defaulting to 0.0 (v1 reward bit-for-bit):

```python
# next to:  reward += aim_bonus * exp(-err**2 / aim_sigma**2)
depth_w = getattr(self, "depth_bonus", 0.0)          # default 0.0 = today's reward
if depth_w > 0.0:
    dx = float(events["landing_point"][0]) - self._targets[i][0]
    reward -= depth_w * max(0.0, -dx)                # only shortfall is priced
```

That is the term whose gradient points at `swing_speed_scale`, which is the only knob with measured
authority over the bias (u = +0.215 today, 0.79 of its upward range unused).

### 2.4 #4 — the collision guard, and what it is really for

B's finding is right and its region is wrong. B scanned the two bands hitter v2's reach knobs
opened — the lateral strip y ∈ [−0.90, −0.75) and the high-z strip z ∈ [1.35, 1.45] — and concluded
the colliding branch lives only in the high-z band. **The committed intercepts of all 400
scoreboard trials are in neither band** (y ∈ [−0.75, +0.05], z ∈ [0.91, 1.35]). Scanning the band
the balls are actually in:

```
OMP_NUM_THREADS=1 PYTHONPATH=. ./.venv/bin/python work/tt/analysis/reach/v3_opband.py <x>
OMP_NUM_THREADS=1 PYTHONPATH=. ./.venv/bin/python work/tt/analysis/reach/v3_opband_pairs.py <x>
```

| plane | frac < 4 cm | (y ≥ −0.75) | median res | **right arm through torso** | idle left arm vs hip | face·normal |
|---|---|---|---|---|---|---|
| −1.20 | 0.512 | 0.572 | 0.0353 | **3** / 950 | 0 | 0.939 |
| −1.31 | 0.675 | 0.754 | 0.0000 | **86** | 109 | 0.969 |
| −1.41 | 0.759 | 0.848 | 0.0000 | **227** | 276 | 0.982 |
| −1.50 | 0.818 | 0.914 | 0.0000 | **306** | 276 | 0.990 |

Two corrections to the record:

* The deep planes **do** put the right arm through the torso in the operational band
  (`right_shoulder_roll_link | torso_link`, `right_shoulder_yaw_link | torso_link`), not only at
  z ≥ 1.35. B's recommended region (`x ≤ −1.24, z ≥ 1.34, y ≥ −0.45`) cannot see any of it: its
  z floor is above the whole band. Flown on the shipped policy it fires **0 / 6188** solves on seed
  2000 and 5 / 6145 on seed 4000, and scores 153 / 140 against a base of 153 / 141. It is free and
  it is worth nothing.
* A second, benign contact family dominates the same scan and is **not** covered by
  `test_hitter_v2._benign_contact`: the idle left arm against `left_hip_roll_link` and
  `left_elbow_link` against the hips (109–276 poses/plane). The filter excuses
  `left_wrist_* | left_hip_pitch_link` — one link over. Widen it, with the same justification: the
  left arm is held at `STAND_POSE`, is off the IK chain, and cannot touch the paddle.

A guard whose region covers the band does remove the real collisions:

```
OMP_NUM_THREADS=1 PYTHONPATH=. ./.venv/bin/python work/tt/analysis/reach/v3_guard.py scan <x> 0.85
```

| plane | right-arm poses, guard off → on | frac < 4 cm off → on |
|---|---|---|
| −1.20 | 3 → **0** | 0.512 → 0.514 |
| −1.31 | 86 → **0** | 0.675 → 0.683 |
| −1.41 | 227 → **0** | 0.759 → 0.781 |
| −1.50 | 306 → **5** | 0.818 → 0.813 |

(−1.20 and −1.31 measured with the region at `x ≤ −1.10` via `v3_stance.py <x> 0.0 guard -1.10`;
−1.41 and −1.50 with `v3_guard.py scan <x> 0.85`, whose `x ≤ −1.24` is the same test at those
depths.)

…and it costs legal returns when it fires often. Region `x ≤ −1.10, z ≥ 0.85, y ≥ −0.45`, shipped
policy, 400 serves: **287 legal vs 294** (−1.75 pts) and target 75 vs 107 (−8 pts), firing on ~35 %
of solves. At the recommended −0.10 m stance (where the working band moves inward and the count at
x = −1.31 rises 86 → 178 of 950) it is still a wash on legal returns and still costs placement:

| −0.10 m stance, 600 serves | legal | target |
|---|---|---|
| no guard | **467** | 183 |
| guard (`x ≤ −1.10, z ≥ 0.85, y ≥ −0.55`) | 466 | 132 |

**Verdict.** Ship `collision_guard` as a knob whose nominal is the identity, default **off**, and
do not turn it on for the current serve set. Its value is that it removes a real latent defect —
up to a third of the deep-band poses drive the upper arm into the chest — which is what a widened
serve set (`serve_scale` 1.2–1.5) or a `+y` crossbody box would immediately hit. Ship it now
because it is cheap and tested; enable it when there is a serve set that needs it.

### 2.5 #5 — prune the action space

C's decode over 400 serves: `gate_lift` and `gate_crossbody` are rectified to exactly zero on
400/400 serves; all ten v2 knobs sit within |u| ≤ 0.20 of nominal; `x_hit_coarse`'s decoded value
never exceeds 0.037 m of a ±0.30 m bound; `aim_x/y_offset` have gain ≈ 0.03 on the landing point.
The four knobs that genuinely vary per serve are `x_hit_offset`, `swing_speed_scale`,
`y_hit_offset`, `t_hit_offset` (u sd 0.32–0.49), plus `pitch_offset` and `flight_time_scale` as
near-constants. A v3 space of those six plus `collision_guard` is a 7-coordinate Gaussian instead
of 20, which is the single cheapest way to make #2's retrain find anything.

### 2.6 The nine rejected knobs, with the numbers that reject them

```
OMP_NUM_THREADS=1 PYTHONPATH=. ./.venv/bin/python work/tt/analysis/reach/v3_probe.py \
    --seed 2000 --trials 200 --tag <tag> [--dcoarse D] [--reach-y R] [--wnscale W] \
    [--iters N] [--swing-mult M] [--ft-mult F] [--win-mult W]
```

Seed 2000, 200 serves. Base row reproduces the scoreboard exactly.

| probe | legal | target | contacts | no contact | net | out | dx | dy |
|---|---|---|---|---|---|---|---|---|
| **base (shipped)** | **153** | **59** | **168** | 32 | 2 | 13 | −0.275 | −0.200 |
| `x_hit_coarse` −0.10 | 143 | 29 | 156 | 44 | 1 | 12 | −0.361 | −0.299 |
| `x_hit_coarse` −0.20 | 130 | 5 | 149 | 51 | 4 | 15 | −0.504 | −0.379 |
| `x_hit_coarse` −0.30 | 107 | 3 | 136 | 64 | 5 | 24 | −0.587 | −0.459 |
| `x_hit_coarse` +0.10 | 106 | 56 | 152 | 48 | 8 | 38 | −0.141 | −0.235 |
| `reach_expand_y` = 1.0 | 153 | 59 | 168 | 32 | 2 | 13 | −0.275 | −0.200 |
| `dls_iters` 40 | 155 | 60 | 167 | 33 | 2 | 10 | −0.263 | −0.202 |
| `dls_iters` 60 | 155 | 60 | 165 | — | — | — | −0.263 | −0.201 |
| `w_normal` 8 → 2 | 154 | 58 | 167 | — | — | — | −0.283 | −0.204 |
| `w_normal` 8 → 0.5 | 153 | 64 | 169 | 31 | 1 | 15 | −0.267 | −0.204 |
| `swing_speed_scale` ×0.95 | 157 | 34 | 168 | 32 | 2 | 9 | −0.352 | −0.225 |
| `swing_speed_scale` ×0.90 | 156 | 30 | 167 | 33 | 0 | 11 | −0.441 | −0.240 |
| `flight_time_scale` ×1.30 | 139 | 53 | 167 | 33 | 1 | 27 | −0.264 | −0.232 |
| `swing_window_scale` ×1.40 | 145 | 74 | 163 | — | — | — | −0.239 | −0.181 |
| `swing_window_scale` ×1.70 | 141 | 74 | 162 | — | — | — | −0.217 | −0.166 |
| `swing_window_scale` ×0.80 | 128 | 13 | 163 | — | — | — | −0.651 | −0.286 |

Confirmed on both seeds (400 serves) where it mattered: base 294 legal / 107 target;
`dls_iters` 40 → **294 / 107** (exactly zero); `swing_speed_scale` ×0.95 → 299 / 66;
`swing_window_scale` ×1.40 → 266 (−28).

Readings that matter for the design:

* **The hitting plane is at a sharp two-sided optimum.** Deeper loses *contacts* (168 → 156 → 149 →
  136) even though the static band gets steadily more reachable (§2.4's first table). Shallower
  loses to the net and off the end (net 2 → 8, out 13 → 38). A's "+18.5 points of reachability from
  a free plane, a training problem not a code problem" does not survive being flown: it is −10 to
  −47 legal returns. Kinematic reachability is a necessary condition for a contact, not a
  sufficient one, and the depth the policy chose is right.
* **`reach_expand_y = 1.0` is bit-identical in outcome.** It fires on exactly the 25 clamped trials
  of seed 2000, moves their commanded intercept by up to **0.15 m**, and produces **0 extra
  contacts and 0 extra legal returns**. With §1.1's "0 clamped-but-reachable", the reach clamp is
  now closed as a topic.
* **The late whiffs are not an IK-convergence problem.** Tripling the solver's iteration budget
  (well inside the 20 ms control step — B measured 2.48 ms/step stock) changes nothing at all.
  Whatever is late about those 8 balls, it is not that the DLS ran out of iterations.
* **Placement and legal returns trade along one axis** (paddle speed), in both directions, and the
  shipped policy sits within ~2 points of the legal-optimal point on it.

### 2.7 What this design explicitly does not do

* It does not widen `Y_REACH`. 0 of 57 clamped trials are reachable anywhere on their flight, and
  opening the floor to the full measured envelope changes no outcome.
* It does not open a `+y` crossbody box. Every commanded y ≥ +0.30 self-collides on this solver,
  and 3 % of intercepts land there.
* It does not relax the waist roll/pitch limits. ±0.52 rad is the vendored Menagerie G1's own
  `jnt_range`; ×2 buys +3.2 points of reachability and 0 of 124 serves at the working plane. That
  is a hardware claim, and the honest place to make it is a conversation with HIM, not a patch.
* It does not unweld the pelvis. Measured separately (`work/tt/rl/free_base/RESULT.md`): −20 points.

---

## 3. Implementation plan for #1, against `tt_sim/policies/ik_hitter.py`

The change has two halves: **where the pelvis is** (one constant in `scene.py`) and **the hitter's
reach frame following it** (`ik_hitter.py`). Everything is behind a value whose default is today's,
with an explicit early-out — the pattern `_rotate_normal`, `blend_shape`, `reach_box` and
`x_hit_coarse` already use in this file.

### 3.1 `tt_sim/scene.py` — one constant

```python
ROBOT_X = -(TABLE_HALF_LENGTH + ROBOT_STANDBACK_M)   # -1.77, unchanged
ROBOT_Y = 0.0        # lateral stance offset; -0.10 measured best on DEFAULT_PARAMS serves
ROBOT_Z = 0.793      # unchanged
BASE_NOMINAL_POS = np.array([ROBOT_X, ROBOT_Y, ROBOT_Z], dtype=np.float64)
```

and in `_build_xml`:

```python
f'<body name="pelvis" pos="{ROBOT_X} {ROBOT_Y:g} {ROBOT_Z}" childclass="g1">'
```

`:g` is deliberate: `f"{0.0:g}"` renders `0`, so at the default the XML string is **byte-identical**
to today's `pos="-1.77 0 0.793"`. Every model hash, every golden fixture and
`work/tt/analysis/reach/reach_lib.py`'s pelvis-line assert survive untouched.

`BASE_NOMINAL_POS` already flows to the free-base reset (`scene.py:561`) and to the hitter's
`to_shadow_frame`, so both follow the stance for free.

### 3.2 `tt_sim/policies/ik_hitter.py` — the reach frame follows the stance

`X_HIT` (x), `Z_REACH` (z) and every direction are unaffected. Exactly three things are lateral
constants measured against a pelvis at y = 0: `Y_REACH`, `REACH_MAX_Y`, and each template's
`y_lo`/`y_hi`.

```python
# Lateral reach is measured relative to where the pelvis is welded: Y_REACH and
# every template's y box came from an FK sweep with the pelvis at y = 0. If the
# stance moves, they move with it -- the arm did not change, the robot did.
STANCE_Y = float(scene.ROBOT_Y)


def _stance_y(pair):
    """Slide a (lo, hi) y box to the current stance. At the nominal stance it
    returns the argument untouched -- no float arithmetic at all -- so every
    template dict holds the identical float objects it holds today and
    blend_shape / reach_box stay bit-exact, including their identity early-outs
    (test_gates_at_zero_return_the_drive_template)."""
    if STANCE_Y == 0.0:
        return pair                      # WHY: bit-exact -- no float arithmetic at all
    return (pair[0] + STANCE_Y, pair[1] + STANCE_Y)
```

Applied in exactly four places, all module scope, all evaluated once at import:

```python
Y_REACH     = _stance_y((-0.75, 0.15))
REACH_MAX_Y = _stance_y((-0.90, 0.15))
# drive / block / crossbody already read y_lo, y_hi from Y_REACH, so they follow it
# for free.  The two templates with their own literals get the same treatment:
#   lift:   y_lo, y_hi = _stance_y((-0.78, 0.15))
#   punch:  y_lo, y_hi = _stance_y((-0.72, 0.15))
```

With `STANCE_Y == 0.0` every one of those returns its argument untouched, so
`SWING_TEMPLATES`, `blend_shape`'s early-out, `reach_box`'s early-out and `_drive_shape` are
unchanged object-for-object. **Nothing else in the file moves.** In particular the aim point stays
world-frame — the table did not move.

`_solve_ik` needs no change for the stance: it reads the shadow model, and the shadow model is
built by `scene.build_model`, which now welds the pelvis at `ROBOT_Y`.

### 3.3 `collision_guard` (item #4), in the same file

A **v3** parameter space, not v2: `test_v2_specs_extend_v1_without_touching_it` pins
`K_PARAMS_V2 == 20`.

```python
SERVE_PARAM_NOMINAL_V3 = {
    **SERVE_PARAM_NOMINAL_V2,
    "collision_guard": 0.0,     # >0.5 = keep the arm off the crossbody branch in the deep/inner region
}

# Measured (work/tt/analysis/reach/solver.md, v3_guard.py): the branch that drives the
# upper arm into the chest is one posture -- waist yaw strongly negative, waist roll on
# its -0.520 stop, shoulder roll adducted across the chest. Three per-iteration clip
# bounds remove it; no single bound and no pair does (0 vs 9-113 of 1450), and a
# half-strength box is worse than none (222 vs 216), so this is a SWITCH, not a lerp.
GUARD_BOX = (-0.20, -0.30, -0.15)          # right_shoulder_roll <=, waist_yaw >=, waist_roll >=
# Region, pelvis-relative in x and y (arm geometry) and world in z (the table):
GUARD_REGION = (0.67, 0.85, -0.45)         # x <= ROBOT_X + 0.67, z >= 0.85, y >= ROBOT_Y - 0.45
```

```python
def _solve_ik(self, q29_current, p_des, n_des, w_normal=None):
    if self.collision_guard <= 0.5:
        return self._solve_ik_stock(q29_current, p_des, n_des, w_normal)   # today's body, verbatim
    dx, z_min, dy = GUARD_REGION
    if not (p_des[0] <= scene.ROBOT_X + dx and p_des[2] >= z_min
            and p_des[1] >= scene.ROBOT_Y + dy):
        return self._solve_ik_stock(q29_current, p_des, n_des, w_normal)
    saved = self._ik_jnt_range
    rng = saved.copy()
    rng[IK_JOINT_NAMES.index("right_shoulder_roll_joint"), 1] = GUARD_BOX[0]
    rng[IK_JOINT_NAMES.index("waist_yaw_joint"), 0] = GUARD_BOX[1]
    rng[IK_JOINT_NAMES.index("waist_roll_joint"), 0] = GUARD_BOX[2]
    self._ik_jnt_range = rng
    try:
        return self._solve_ik_stock(q29_current, p_des, n_des, w_normal)
    finally:
        self._ik_jnt_range = saved
```

Split today's `_solve_ik` body into `_solve_ik_stock` unchanged and make `_solve_ik` the two-line
dispatcher above, so the off path is the *same code*, not a re-derivation. At
`collision_guard = 0.0` the first line returns before any arithmetic and before the region test —
the guard is unreachable at nominal, so `test_v2_nominal_matches_plain_hitter` and the committed
golden fixture stay green without being touched.

### 3.4 How the per-serve policy selects it

**The stance is not a knob.** It is a property of the scene the policy is trained against, like the
table height. Put it in `scene.py` and retrain; do not give the policy a coordinate it cannot
change mid-episode anyway.

**`collision_guard` ships as a hitter default, not a learned coordinate** — measured reasons, in
order:

1. It is a **switch**, not a lerp (B: half strength is 222 colliding poses vs 216 for none). A
   `tanh`-squashed Gaussian coordinate spends most of its mass in a region where the knob does
   nothing, so PPO would burn a coordinate to learn a threshold.
2. Its sign is **known from the stance**, not from the serve: 0 points at both stances tested,
   −8 points of target hits when on. There is nothing per-serve to learn.
3. Adding it to a v3 space is still cheap and reversible — put it in `SERVE_PARAM_NOMINAL_V3` so a
   future space *can* expose it, but leave it out of `PARAM_SPECS_V3` until a serve set exists on
   which it scores.

Default: `collision_guard = 0.0` in the constructor, overridable via `set_serve_params` and via
`IKHitterPolicy(serve_params={"collision_guard": 1.0})`, exactly like every other knob.

### 3.5 Tests to add (`tt_sim/tests/test_hitter_v3.py`)

1. **`test_nominal_stance_is_bit_identical`** — `scene.ROBOT_Y == 0.0`, `_stance_y(box) is box`,
   `blend_shape((0,0,0,0), h._drive_shape) is h._drive_shape`, and a 150-step rollout
   `np.array_equal` against `IKHitterPolicy()` — the v2 parity test's three routes, plus the
   existing golden fixture `tt_sim/tests/data/param_s3_actions.npz` re-run unchanged.
2. **`test_scene_xml_is_byte_identical_at_the_default_stance`** — assert the literal
   `'<body name="pelvis" pos="-1.77 0 0.793" childclass="g1">'` is in `scene._build_xml(True, True)`.
   Cheap, and it is the guard for the `:g` formatting above.
3. **`test_reach_boxes_follow_a_non_zero_stance`** — monkeypatch `scene.ROBOT_Y = -0.10`, reload
   the module, assert `Y_REACH == (-0.85, 0.05)`, `REACH_MAX_Y == (-1.00, 0.05)` and every
   template's `y_lo/y_hi` shifted by exactly −0.10; assert `X_HIT`, `Z_REACH`, `REACH_MAX_Z` and
   every `face_*`/`dir_*` field are untouched.
4. **`test_collision_guard_off_is_the_stock_solver`** — with `collision_guard = 0.0`, `_solve_ik`
   returns `np.array_equal` results to `_solve_ik_stock` on 200 random targets *inside* the region
   (proves the early-out precedes the region test), and a 150-step rollout is bit-exact.
5. **`test_the_operational_band_is_collision_free_under_the_guard`** — the new scan, i.e. the band
   the balls occupy rather than the band v2's knobs opened: y ∈ [−0.85, +0.05] step 0.05,
   z ∈ [0.90, 1.35] step 0.05, the five existing face normals, `_ik_pose_at`'s recipe. Pin
   **right-arm-vs-torso** poses (the filtered category, see 5b) at
   `{-1.20: 0, -1.31: 0, -1.41: 0, -1.50: 5}` with the guard on and
   `{-1.20: 3, -1.31: 86, -1.41: 227, -1.50: 306}` with it off, all measured today. As with the v2
   test, the guard-off counts are a regression ceiling, not an endorsement.
   *5b.* Widen `_benign_contact` to excuse the idle left arm against **any** hip/pelvis link
   (`left_wrist_*`/`left_elbow_link` vs `left_hip_roll_link`, `left_hip_pitch_link`,
   `right_hip_roll_link`), with the measured counts (109–276 poses/plane) and the reason in the
   docstring: the left arm is held at `STAND_POSE`, is off the IK chain, and cannot reach the
   paddle. Without this the new test is dominated by contacts nobody intends to fix.
6. **`test_the_guard_fits_the_50_hz_budget`** — mean wall time of one `_solve_ik` call over 200
   random in-region targets < 5 ms (measured: 1.48 ms in-region with the guard, 3.59 ms stock,
   2.46 ms out-of-region; the control step is 20 ms). Assert the *guard-on* mean is not worse than
   the *guard-off* mean on the same targets — it is a keep-out, so it produces fewer contacts for
   `mj_forward` to solve and is genuinely cheaper.
7. **`test_v3_does_not_widen_the_train_eval_pipeline_gap`** — the existing v2 test (R3), re-run at
   `ROBOT_Y = -0.10` and with `collision_guard = 1.0`: `VecTTParam` clips the hitter's target to
   ±3 rad from `STAND_POSE` while `tt_sim.evaluate` does not, and the stance shift moves every
   commanded pose.
8. **`test_v3_specs_extend_v2_without_touching_it`** — `PARAM_SPECS_V3[:20] == PARAM_SPECS_V2`,
   `K_PARAMS_V2 == 20` still, `PARAM_SPACES` gains `"v3"`, `DEFAULT_PARAM_SPACE` still `"v1"`.

Gate: `PYTHONPATH=. ./.venv/bin/python -m pytest -q tt_sim/tests`.

### 3.6 Order of work

`scene.py` constant + tests 1, 2 → `ik_hitter.py` stance frame + test 3 → re-score the shipped
`.npz` at `ROBOT_Y = -0.10` and confirm 467/600 → `collision_guard` + tests 4, 5, 5b, 6 →
v3 spec + test 8 → test 7 → retrain (§4).

The re-score in the middle is the checkpoint: it must reproduce **161 / 154 / 152** on seeds
11 / 2000 / 4000. If it does not, the in-package stance frame differs from the scratch harness and
nothing downstream is trustworthy.

---

## 4. What to train after it lands

### 4.1 Mac, not a box

A warm-started 20 000-episode run is ~65 min on the Mac at `--n-envs 32` and costs no credits;
`param_v2/RESULT.md` records nine of them. A box is only worth its 93 credits/h when the run is
*cold* in a new space and needs 100–200 k episodes — which is the pruned v3 space of §2.5, not
this. **Run #2 and #3 on the Mac. Say so out loud that the laptop must not sleep.**

### 4.2 The run

Both changes in one run, warm-started from the shipped policy so episode 0 is already 77.8 %:

```
PYTHONPATH=. ./.venv/bin/python -m tt_sim.rl.param_train \
    --out work/tt/rl/v3_stance/s0 \
    --n-envs 32 --episodes 20000 --eval-every 200 \
    --param-space v2 --resume-from work/tt/rl/param_v2/wa0/final.pt \
    --log-std-init-new -2.5 \
    --vel-penalty 1e-5 --legal-bonus 10 \
    --aim-bonus 8 --aim-sigma 0.30 --target-hit-bonus 4 --depth-bonus 3 \
    --det-eval-episodes 200 --det-eval-every 2000 --det-eval-seed 1001 --det-select legal
```

(`--depth-bonus` is the new flag from §2.3; drop it to reproduce `wa0`'s reward exactly.)

Two seeds (`--seed 0`, `--seed 1`), sequential, ~2.2 h total. Run a third with
`--det-select target` only if #3's placement gain is the thing being chased.

### 4.3 Deterministic selection recipe

* **Selection set: seed 1001, 200 serves, mean action** (`--det-eval-episodes 200
  --det-eval-every 2000 --det-eval-seed 1001`). 200, not 80: `param_v2/RESULT.md` §4 records that
  an 80-serve set picked the worse checkpoint in 5 of 9 runs (wa1: 65.3 % vs 71.2 %). Seed 1001
  draws serves 1001–1080… and with 200 trials, 1001–1200 — still disjoint from 11 / 2000 / 4000,
  which is why `eval_seeds` refuses seeds closer together than the trial count.
* **Scoreboard: seeds 11 / 2000 / 4000, 200 serves each, mean action**, never the selection set:

```
PYTHONPATH=. ./.venv/bin/python -m tt_sim.rl.eval_seeds \
    --ckpt work/tt/rl/v3_stance/s0/best.pt --ckpt work/tt/rl/v3_stance/s0/final.pt \
    --npz assets/policies/param_residual_v2_wa0.npz --hitter \
    --seeds 11 2000 4000 --trials 200 --jobs 4 --out work/tt/rl/v3_stance/scoreboard
```

* **Re-measure the ceiling first**, so the result is scored against the right number (~5 min):

```
PYTHONPATH=.:work/tt/analysis/reach ./.venv/bin/python work/tt/analysis/reach/sweep.py \
    --seeds 11 2000 4000 --trials 200 --procs 4 --pelvis-dy -0.10 --out work/tt/analysis/reach/sweep_v3.json
```

  A's graded curve puts reachability at 88.0 % for −0.10 m, so the expected new ceiling is
  ≈ 88 % × 0.91 + 12 % × 0.07 ≈ **81 %**, and the contact rate is already 87.8 %.

### 4.4 Pre-registered acceptance bar

| outcome on seeds 11 / 2000 / 4000 | verdict |
|---|---|
| < 77.8 % legal | the retrain lost what the stance gave; keep `wa0` at the new stance and stop |
| 77.8 – 79 % | the stance is the whole gain; ship it, close the reach lane |
| ≥ 80 % legal | the retrain converted the new contacts; continue with the pruned v3 space (§2.5) |
| target hits ≥ 35 % at ≥ 77 % legal | `--depth-bonus` worked; that is the placement result |

Record it in `work/tt/rl/v3_stance/RESULT.md` and update the scoreboard in
`docs/PROJECT_KNOWLEDGE.md` §2 either way.

---

## 5. Scripts and commands

All from the repo root, `OMP_NUM_THREADS=1 PYTHONPATH=. ./.venv/bin/python <script>`. Nothing under
`tt_sim/` was modified by any of them.

| script (in `work/tt/analysis/reach/`) | what it measures |
|---|---|
| `v3_join.py` | A's reachability × C's taxonomy on the same 400 trials (§1) |
| `v3_probe.py` | nine uniform knob directions on the shipped policy (§2.6) |
| `v3_opband.py`, `v3_opband_pairs.py` | the 950-pose scan of the band the balls occupy (§2.4) |
| `v3_guard.py` (`scan` / `task`) | the keep-out guard, geometrically and on the task (§2.4) |
| `v3_stance.py` | operational-band reachability with the pelvis slid, balls fixed (§2.1) |
| `v3_stance_task.py` | the shipped policy flown at a shifted stance, ± the guard (§2.1) |

Reference values every harness reproduces before its own numbers are read: shipped policy
147 / 153 / 141 legal on seeds 11 / 2000 / 4000; plain hitter 93 / 94 / 85.

---

## 6. Risks

**R1 — The stance shift is fitted to `DEFAULT_PARAMS`.** Serves land at `target_y ∈ [−0.65, −0.10]`,
forehand only. −0.10 m is a tuned constant for that distribution, and a symmetric distribution would
give it back. *Mitigation:* it is a named constant (`scene.ROBOT_Y`), the tuning curve is in §2.1,
and re-tuning is one sweep. Re-run that sweep before any `serve_scale` change ships.

**R2 — HIM may not let us choose the base pose.** The evaluator's scene and observation contract are
open questions (`docs/PROJECT_KNOWLEDGE.md` §2). *Mitigation:* ask, and note that the change is one
constant, so if the answer is "fixed at y = 0" nothing else in this design is invalidated —
items #3 and #5 stand on their own.

**R3 — Static reachability keeps over-promising.** Twice now (A's deep plane, A's −0.30 m stance)
the offline multi-restart reach number ranked an option first that the evaluator ranked last.
*Mitigation:* nothing in this design is recommended on a reach number alone; every ranked item in
§2 carries a flown result or an explicit "0 points, ship for insurance" label.

**R4 — 38 trials were lost at the new stance.** A retrain could plausibly lose more than it gains
if the policy over-fits the new geometry with the old, unpruned 20-knob space. *Mitigation:* §4.4's
pre-registered bar; `best.pt` selected on a 200-serve deterministic set; `wa0` at the new stance
(467/600) is the fallback and needs no training at all.

**R5 — The new collision test is expensive.** 950 poses per plane (190 grid points × 5 face
normals) × 8 solver calls ≈ **20 s per plane per guard state**, so the four planes × two states in
test 5 is ~2.5 min — more than the whole rest of the suite. *Mitigation:* pin two planes (−1.20,
the working plane, and −1.50, the worst) × two states, and subsample to three normals; that is
~30 s and still catches a regression. The existing v2 collision test is already the suite's slowest.

**R6 — `collision_guard` is dead code until a serve set needs it.** *Mitigation:* it is 15 lines and
one test, it fixes a defect that is real today (306 of 950 poses at x = −1.50), and it is the
prerequisite for the `+y` crossbody box and for `serve_scale > 1`. If none of those happens within
a lane or two, delete it rather than carry it.

---

## 7. Open questions for HIM (unchanged, restated because #1 depends on one)

* Does the evaluator fix the robot's base pose, or may a policy choose where it stands?
* What is the real serve distribution — forehand-biased like `DEFAULT_PARAMS`, or symmetric?
* Observation contract (ball state? base pose?), control rate, trial count, challenge version id.
