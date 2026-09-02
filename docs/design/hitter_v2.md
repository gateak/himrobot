# hitter v2 — widening the scripted hitter's action space

Design only. No source files changed. All numbers below are measured in this repo
today (scripts were throwaway; the commands are reproducible from the description
next to each number).

Baselines being improved on: `ik_hitter` 46.5% legal / 3.5% target (200 serves,
seed 11); `ParamResidualPolicy work/tt/rl/param/s3/best.pt` 70.5% / 10%.
The 300-serve runs in `work/tt/analysis/{hitter,param_s3}/evaluation/evaluator.json`
(46.7%/3.3% and 72.0%/8.3%) are the failure-mode dataset used throughout.

---

## 0. The number that shapes the whole design

I swept the *entire* predicted flight of all 300 evaluation serves (post-bounce,
x in [-1.90, -0.90], z > 0.55 — plus a separate pre-bounce pass) and solved a
position-only, multi-restart, joint-limit-respecting IK for the blade centre at
every sample, taking the minimum residual per serve:

| subset | n | median min-reach residual | frac with a residual < 4 cm | frac < 8 cm | current legal |
|---|---|---|---|---|---|
| intercept y >= -0.75 at x=-1.20 | 246 | 0.000 m | 0.93 | 0.96 | 87% |
| intercept y <  -0.75 at x=-1.20 |  54 | 0.244 m | 0.00 | 0.04 |  2% |
| all | 300 | — | 0.76 | 0.80 | 72% |

**20–24% of the serve distribution has no reachable intercept anywhere on the
ball's flight**, with the pelvis welded and the paddle welded to the right wrist.
The best achievable plane for those serves is x ≈ -1.40 (deeper helps reach more
than the ball's extra -y drift hurts) and they are still ~0.20–0.28 m short.
Pre-bounce interception does not rescue them: the serve's first bounce is at
x in [-1.05, -0.25] and forward blade reach stops at x ≈ -1.05, so the ball is
never both pre-bounce and in the workspace.

Consequences that the rest of this design is built around:

1. **The v1 policy at 70.5–72% is 5–8 points below its own kinematic ceiling.**
   A swing library cannot break 80% on this serve distribution. Anything claiming
   more requires unwelding the pelvis (a `tt_sim/scene.py` change, i.e. a different
   evaluated system) — explicitly out of scope here.
2. Therefore v2's honest legal-return target is **78–82%**, obtained by converting
   the ~8% of *reachable* serves that are still missed, plus robustness margin.
3. `target_hit_rate` is where the real headroom is: 10% today with a 30 cm radius,
   and the miss is **systematic, not random** (see §5).

---

## 1. Structural limits of the current hitter, and what each costs

### 1.1 The reach clamp `Y_REACH=(-0.75, 0.15)` / `Z_REACH=(0.65, 1.35)`

Measured collision-free blade-centre reach at x=-1.25, z=1.10 (16 random restarts
per point, `ncon == 0` required):

```
y = -0.55  err 0.000     y = +0.15  err 0.000
y = -0.65  err 0.011     y = +0.30  err 0.000
y = -0.70  err 0.051     y = +0.40  err 0.000
y = -0.75  err 0.098     y = +0.50  err 0.000
y = -0.80  err 0.148     y = +0.60  err 0.000
y = -0.85  err 0.207
```

and in z at x=-1.35, y=-0.50: reachable from 0.60 to 1.45, failing (6 cm) at 1.50.
Forward: x = -1.05 costs 4 cm, x = -0.95 costs 12 cm.

So the clamp is **wrong in both directions**:

* The **lower** y bound (-0.75) is already ~5 cm *beyond* the collision-free limit,
  and the *runtime* solver does far worse than the offline one (see 1.4). It is not
  the binding constraint on the hard 18% — geometry is — but it also buys nothing.
* The **upper** y bound (+0.15) throws away **45 cm** of genuinely reachable,
  collision-free crossbody workspace. On today's serve distribution only 3% of
  intercepts land at y > -0.15, so this costs ~0 now; it is what makes a widened
  serve set (`serve_scale` 1.2–1.5, item 3 of `work/tt/rl/param/RESULT.md`'s "Next")
  survivable at all.
* `Z_REACH` is conservative by 5 cm at the bottom and **10 cm at the top**. Only
  0.7% of intercepts exceed 1.35 today, so this is a ~0.5 pp item.

Cost today: near zero from the clamp itself. Its real cost is that it hides the
crossbody half of the workspace from the policy entirely.

### 1.2 `x_hit = -1.20`, a single fixed hitting plane

`x_hit_offset` exists but is bounded at ±8 cm. The measured best plane over the
free sweep is **x = -1.40 median (q10 -1.90, q90 -1.25)** for reachable serves,
i.e. the optimum sits 20 cm outside what the policy can currently ask for. Deeper
planes trade a little extra -y drift (about 0.045 m of y per 0.10 m of x) for a
lot of extra reach (about 0.08 m of reach per 0.25 m of x) and for ~25 ms more
time to complete the swing.

Cost: the policy is pinned to a plane that is 20 cm shallower than optimal for the
serves it currently fails on. This is the cheapest single fix in the whole design.

### 1.3 One swing profile shape

`_swing_target` is a cubic-Hermite ease-in onto a **single constant-velocity line**
through `p_hit`, cruising from `t_contact - speed_buffer` to `t_contact + t_follow`.
`swing_vel = risk_preference * (v_target - v_in)` capped at `max_swing_speed = 1.0`.
Every serve gets the same shape; only its direction and (capped) magnitude vary.
The window widths (`t_pre=0.30`, `speed_buffer=0.03`, `t_follow=0.20`), the
backswing (0.10 m), the speed cap and the 0.50 s return flight time are
constructor kwargs, invisible to the per-serve policy.

Costs, from the failure histogram:

| failure | ik_hitter (n=300) | param_s3 (n=300) |
|---|---|---|
| no contact | 85 (28.3%) | 59 (19.7%) — of which ~46 are the unreachable bucket |
| contact, net hit | 56 | 3 |
| contact, never crossed net | 15 | 13 |
| contact, crossed but no opponent bounce (long/out) | 4 | 9 |

The plain hitter's dominant *post-contact* failure is the net (56/75). The tuned
policy fixed that by pinning two knobs to their bounds (§1.6) and now fails
symmetrically short (13) and long (9). One profile shape cannot be both a
net-clearing lift for a low late intercept and a flat punch for a high early one;
the parameter policy has been forced to pick a compromise per serve out of knobs
that were designed as trim, not as shape.

### 1.4 The IK: chain, and the orientation/position tradeoff

The task brief says "arm-only IK". **That is not what the code does** —
`IK_JOINT_NAMES` is already waist yaw + waist roll + waist pitch + the 7 right-arm
joints (10 DOF). The real limits are different and worth stating precisely:

* `waist_roll` and `waist_pitch` **saturate at their ±0.52 rad limits in every
  deep-reach solve I ran**. The waist is already giving everything it has.
  `waist_yaw` (±2.618) is *not* saturated (-0.41 at the extreme) because yawing the
  torso moves the right shoulder mostly fore/aft, not laterally — it is nearly
  useless for -y reach. Adding "more waist" to the chain is a no-op.
* The 12 leg joints cannot help **at all**: the pelvis is the welded root body
  (`scene._build_xml` drops the freejoint and pins the pelvis at x=-1.77), so hip
  and knee motion moves the feet relative to a fixed torso. `ACTION_SCALE`'s
  "legs zero" is correct, not a limitation to lift.
* `w_normal = 8.0` weights the blade-normal residual 8× against position. Near the
  workspace boundary that is a large real cost: at x=-1.20, z=1.10 the warm-started
  runtime solver leaves 0.13 m of position error at y=-0.55, 0.24 m at y=-0.65 and
  0.35 m at y=-0.75, while a position-only solve at the same points leaves
  0.00 / 0.01 / 0.10 m. **Trading face accuracy for position buys ~0.13 m of
  effective lateral reach**, and near the edge the paddle is going to be
  face-inaccurate anyway.
* The solver is a local, warm-started DLS with `step_clamp = 0.35` — it finds
  whatever branch it is already on. `ns_gain = 0.0`, so there is no posture bias
  pulling it toward a branch that will still be good 100 ms later.

Cost: roughly 0.10–0.15 m of usable lateral workspace, unavailable to the policy
because `w_normal` is a constructor kwarg.

### 1.5 The timing model

One `PREDICTION_DELAY_S = 0.03` constant, calibrated (per its own comment) at
`x_hit = -1.15`, then used unchanged at -1.20 and — under v2 — would be used at
planes as deep as -1.55. The bias it corrects (compliant bounce vs instant
reflection) is upstream of the plane, but the *along-track* error it converts into
depends on how far past the bounce the plane sits. `t_hit_offset` (±40 ms) can
absorb the residual, but only if it isn't already spent on something else — and it
is: the trained policy's median `t_hit_offset` u is +0.34.

### 1.6 The paddle-normal model, and the bounds that are already saturating

`n_des = rotate(n_back, pitch_offset, yaw_offset)` — the commanded face is the
*swing direction* rotated. There is no way to ask for a face that is decoupled from
the swing path (e.g. a face-open block with a near-stationary paddle, where
`n_back` is a degenerate fallback `[2,0,0.5]`).

I decoded `assets/policies/param_residual_s3.npz` over all 300 evaluation serves
and looked at the distribution of the chosen `u`:

| param | bound | u median | frac \|u\| > 0.9 | reading |
|---|---|---|---|---|
| `flight_time_scale` | mul 1.43 | **-0.93** | **61%** | pinned at the fast/flat end; wants < 0.7× of 0.50 s |
| `pitch_offset` | add 0.35 | **+0.80** | 15% | pressed against +0.35 rad of face lift |
| `aim_y_offset` | add 0.35 | +0.68 | 0% | aiming at y = 0.79, i.e. *off the table* (half-width 0.7625) |
| `swing_speed_scale` | mul 2.0 | +0.40 | 4% | wants ~1.3 m/s cap |
| `y_hit_offset` | add 0.10 | +0.36 | 0% | |
| `t_hit_offset` | add 0.04 | +0.34 | 0% | |
| `x_hit_offset` | add 0.08 | +0.16 | 0% | |
| `aim_x_offset` | add 0.35 | -0.07 | 0% | |

Two knobs are at or near their bounds and the aim knobs are being spent as bias
correctors rather than as placement. That is the direct evidence that the v1 action
space is *shape-limited*, not just under-trained: give it more range on the same
axes and it will use it.

---

## 2. The swing library

### 2.1 What is deliberately *not* in it

**No topspin/brush template.** `scene.sample_serve` documents no Magnus force and
plain MuJoCo applies none; `<option>` sets no density or viscosity. Spin changes
nothing about the ball's flight in this sim, so a tangential brush has no
aerodynamic payoff. Templates here are justified only by effects that exist:
contact geometry, paddle speed at contact, face angle, dwell time around an
uncertain contact instant, and where on the flight the intercept is taken.

### 2.2 Shape vector

Every template is a preset over the same 17 continuous fields. Nothing structural
differs between templates, which is what makes blending well-defined.

```
SHAPE_FIELDS = ("x_hit", "pred_delay", "speed_cap", "flight_time", "risk_gain",
                "face_pitch", "face_yaw", "dir_pitch", "dir_yaw",
                "t_pre", "speed_buffer", "t_follow", "backswing",
                "y_lo", "y_hi", "z_lo", "z_hi")
```

`face_pitch/face_yaw` are template biases added to `pitch_offset`/`yaw_offset`
before `_rotate_normal`. `dir_pitch/dir_yaw` rotate `swing_vel` itself (and so the
backswing point and the cruise line) — a separate axis from the face.
`risk_gain` multiplies the episode's `risk_preference` inside the `swing_vel` solve.

### 2.3 The five templates

| field | `drive` (T0) | `block` (T1) | `lift` (T2) | `punch` (T3) | `crossbody` (T4) |
|---|---|---|---|---|---|
| `x_hit` | **-1.20** | -1.30 | -1.34 | -1.14 | -1.26 |
| `pred_delay` | **0.030** | 0.034 | 0.036 | 0.028 | 0.032 |
| `speed_cap` | **1.00** | 0.25 | 0.80 | 2.20 | 1.00 |
| `flight_time` | **0.50** | 0.66 | 0.80 | 0.33 | 0.55 |
| `risk_gain` | **1.00** | 0.40 | 0.90 | 1.40 | 1.00 |
| `face_pitch` | **0.00** | +0.18 | +0.30 | -0.05 | +0.10 |
| `face_yaw` | **0.00** | 0.00 | 0.00 | 0.00 | -0.28 |
| `dir_pitch` | **0.00** | +0.25 | +0.55 | -0.10 | +0.10 |
| `dir_yaw` | **0.00** | 0.00 | 0.00 | 0.00 | -0.15 |
| `t_pre` | **0.30** | 0.45 | 0.35 | 0.22 | 0.34 |
| `speed_buffer` | **0.03** | 0.08 | 0.05 | 0.03 | 0.04 |
| `t_follow` | **0.20** | 0.35 | 0.30 | 0.15 | 0.22 |
| `backswing` | **0.10** | 0.03 | 0.14 | 0.18 | 0.08 |
| `y_lo, y_hi` | **-0.75, 0.15** | -0.75, 0.15 | -0.78, 0.15 | -0.72, 0.15 | -0.50, **0.55** |
| `z_lo, z_hi` | **0.65, 1.35** | 0.62, 1.40 | 0.60, 1.30 | 0.68, 1.42 | 0.68, 1.40 |

Bold column = the current constructor defaults, unchanged, bit for bit.

Why each earns its place:

* **`block`** — near-static paddle, wide dwell (`speed_buffer` 0.08, `t_follow`
  0.35, `t_pre` 0.45), face opened +0.18 rad, `risk_gain` 0.4. The ball rebounds at
  e ≈ 0.9 off a correctly-angled 0.17 kg paddle without any swing, so this maximises
  *contact probability* by removing the moving-target component of the ~28 ms of
  prediction timing error. Targets the 13 "contact, never crossed net" and the
  residual no-contacts on marginal serves.
* **`lift`** — `dir_pitch` +0.55 rad and `flight_time` 0.80 s: a high, slow arc with
  large net-clearance margin, for low/late intercepts. Directly aimed at the failure
  the plain hitter dies of (56 net hits) and that the tuned policy currently buys
  off by pinning `pitch_offset`.
* **`punch`** — `speed_cap` 2.2, `flight_time` 0.33, `risk_gain` 1.4, longer
  backswing (0.18 m) and a shallower plane (-1.14). This is the range extension the
  saturated `flight_time_scale` is asking for, and it is the template that can
  actually reach a *deep* target (see §5).
* **`crossbody`** — right arm across the body, reach box opened to y ∈ [-0.50, 0.55]
  (measured collision-free), face yawed -0.28 rad so a ball taken on the backhand
  side still leaves crosscourt. **There is no left-arm backhand**: the paddle is
  welded to the right wrist only (`scene.PADDLE_XML` injected at
  `right_wrist_yaw_link`), so a left-arm mirror is not a possibility, and a mirrored
  right-arm posture is the only backhand available. Value on today's serve set is
  near zero (3% of intercepts at y > -0.15); its value is insurance for
  `serve_scale > 1` training and for the widened-serve generalization item.

### 2.4 Selector: rectified-gate barycentric blend (not argmax)

**Decision: continuous blend, gated so that u = 0 is exactly `drive`.**

```python
GATE_GAIN = 4.0
TEMPLATE_ORDER = ("drive", "block", "lift", "punch", "crossbody")

def blend_shape(gates):          # gates: the 4 raw u values, one per non-drive template
    w = [GATE_GAIN * g if g > 0.0 else 0.0 for g in gates]
    if not any(w):               # WHY: bit-exact v1 path -- no float arithmetic at all
        return SWING_TEMPLATES["drive"]
    weights = [1.0] + w          # drive always carries weight 1
    total = sum(weights)
    return {f: sum(wk * SWING_TEMPLATES[name][f]
                   for wk, name in zip(weights, TEMPLATE_ORDER)) / total
            for f in SHAPE_FIELDS}
```

Then, per serve:

```python
shape = blend_shape([u_gate_block, u_gate_lift, u_gate_punch, u_gate_crossbody])
# the 10 v1 offsets ride on top of the blended shape, exactly as they ride on the
# constructor defaults today:
x_hit_eff        = shape["x_hit"] + x_hit_offset + x_hit_coarse
flight_time_eff  = shape["flight_time"] * flight_time_scale
speed_cap_eff    = shape["speed_cap"] * swing_speed_scale
pitch_eff        = shape["face_pitch"] + pitch_offset
yaw_eff          = shape["face_yaw"]   + yaw_offset
```

Why blend rather than `argmax` over K logits, for a PPO-style Gaussian bandit:

1. **Nominal parity survives exploration.** With argmax over K zero-initialised
   logits, an untrained net picks a uniformly random template on every *sampled*
   episode — 80% of rollouts are non-nominal at step 0, so training no longer starts
   from the measured 46.5% baseline, only the deterministic evaluation does. With
   rectified gates, `u = 0` gives weights `(1,0,0,0,0)` and the early-out returns the
   `drive` tuple unmodified: step zero of training *is* the plain hitter, which is
   the property `test_untrained_param_policy_matches_plain_hitter` exists to protect.
2. **Gradient variance.** `param_train` is a clipped-ratio bandit on batches of 128
   episodes with the advantage renormalised each epoch. Under argmax the reward is a
   step function of the gate coordinates: a 0.01 change in one logit can swap the
   whole swing, so the advantage attributed to that coordinate is dominated by which
   side of the tie the sample fell on. The critic (a single scalar `V(features)`)
   must then average over K disjoint modes, and its residual — which *is* the
   advantage here — inflates for every serve where two templates are close. The
   blend makes the return a continuous function of `u`, so the same 128 samples
   estimate a smooth gradient.
3. **Blends are meaningful in this parameterisation.** Templates are presets over the
   same 17 scalars, so any convex combination is a valid, physically sensible swing
   (a half-block half-lift is a slow lifted push). This would not hold if templates
   differed structurally.
4. **No new machinery.** The gates are four more coordinates of the existing
   diagonal Gaussian over `z`, squashed by the existing `tanh`. No categorical head,
   no changed loss, no tanh-Jacobian correction (the trainer's action variable stays
   the pre-squash `z`).

Accepted cost: the policy cannot express a *pure* non-drive template. With
`GATE_GAIN = 4.0`, a single gate at `u = 1` gives weights `(1, 4)` → 80% of the
target template. That is deliberate — `drive` is the measured-good prior and keeping
20% of it is a regulariser, not a bug. The rectifier `max(u, 0)` has zero pathwise
gradient on half its domain, which is irrelevant here: PPO's gradient is
score-function (`∇ log π(z|s) · A`), so it flows through `z` regardless of the
mapping's smoothness.

---

## 3. Reach expansion, with parity

Three separable pieces, all identity at u = 0.

### 3.1 Relaxed Cartesian clamp

Interpolate each template's box toward the measured envelope:

```python
REACH_MAX_Y = (-0.80, 0.60)     # measured collision-free, +blade radius 0.085
REACH_MAX_Z = (0.58, 1.45)

def reach_box(shape, expand_y, expand_z):
    ey, ez = min(max(expand_y, 0.0), 1.0), min(max(expand_z, 0.0), 1.0)
    if ey == 0.0 and ez == 0.0:                     # WHY: bit-exact v1 clamp
        return (shape["y_lo"], shape["y_hi"]), (shape["z_lo"], shape["z_hi"])
    lerp = lambda a, b, t: a + t * (b - a)
    return ((lerp(shape["y_lo"], REACH_MAX_Y[0], ey), lerp(shape["y_hi"], REACH_MAX_Y[1], ey)),
            (lerp(shape["z_lo"], REACH_MAX_Z[0], ez), lerp(shape["z_hi"], REACH_MAX_Z[1], ez)))
```

**Joint-limit safety is already structural**: `_solve_ik` clips `newq` to
`self._ik_jnt_range` on every iteration, so no Cartesian box can ever produce an
out-of-range joint. What a relaxed box *can* produce is (a) a stretched, near-singular
pose with large joint excursions, and (b) self-collision. Mitigations:

* `REACH_MAX_*` is defined by *measurement with `ncon == 0` required*, not by
  optimism — the numbers above come from the collision-free sweep in §1.1.
* Keep `step_clamp = 0.35`; the reward's `vel_penalty` already prices thrash.
* Add a test asserting `ncon == 0` after IK at every template's box corner (see §6).
  This matters: one local IK branch for `y = +0.55` that I found had **28 contacts**
  (arm through torso) while a different branch reached the same point cleanly. The
  runtime solver is local and warm-started, so it can land on the bad branch.

### 3.2 A wider hitting plane, via a new coarse knob

`x_hit_coarse` (add, ±0.30 m) stacks with the existing `x_hit_offset` (±0.08).
Effective plane range: -1.58 … -0.82 around `drive`, which covers the measured
optimum (median -1.40, q10 -1.90). `x_hit_offset` keeps its exact v1 meaning as fine
trim so a resumed v1 checkpoint's first 10 coordinates still mean what they meant.

### 3.3 Orientation/position weight

`normal_weight_scale` (mul, bound 16.0). The hitter computes
`w_normal_eff = clip(8.0 * normal_weight_scale, 0.5, 16.0)`; at u = 0 the scale is
1.0 → exactly 8.0 → bit-identical. This is the cheapest 0.10–0.15 m of usable
lateral reach in the design (§1.4), and it lets the policy decide per serve whether
this ball is a "get the paddle there at all" ball or a "place it" ball.

### 3.4 What keeps u = 0 identical to today

Every one of the mechanisms above has an explicit early-out on the nominal value —
the same pattern `_rotate_normal` and `scene._scale_range` already use in this
codebase, and the reason it matters is that
`test_nominal_serve_params_match_plain_hitter` asserts `np.array_equal`, not
`allclose`. Concretely, at `SERVE_PARAM_NOMINAL_V2`:

* `blend_shape` returns the `drive` tuple object — no arithmetic, so `x_hit`,
  `t_pre`, `speed_cap` etc. are the identical float objects the constructor uses now.
* `reach_box` returns the drive box unchanged.
* `w_normal_eff` short-circuits to `self.w_normal` when the scale is exactly 1.0.
* `x_hit_coarse == 0.0` adds nothing (guard the add, don't rely on `x + 0.0`).

---

## 4. `PARAM_SPECS_V2`

The ten v1 entries come first, **byte-identical in name, kind and bound**, so
(a) a v1 checkpoint's stored `param_specs` still decodes exactly as it does today,
and (b) `--resume-from` a v1 checkpoint into a v2 run is meaningful coordinate for
coordinate. Where v1 evidence showed a bound saturating (`flight_time_scale`,
`pitch_offset`), the extra range is supplied by the *templates* rather than by
widening a v1 bound — the punch template's 0.33 s base times the unchanged 1.43×
multiplier reaches 0.23 s, and `lift`'s +0.30 rad face bias plus the unchanged
±0.35 reaches +0.65 rad.

```python
PARAM_SPECS_V2: tuple[tuple[str, str, float], ...] = (
    # --- v1, unchanged, in order (indices 0..9) ---------------------------
    ("x_hit_offset",        "add",  0.08),
    ("y_hit_offset",        "add",  0.10),
    ("z_hit_offset",        "add",  0.10),
    ("t_hit_offset",        "add",  0.04),
    ("swing_speed_scale",   "mul",  2.00),
    ("pitch_offset",        "add",  0.35),
    ("yaw_offset",          "add",  0.35),
    ("aim_x_offset",        "add",  0.35),
    ("aim_y_offset",        "add",  0.35),
    ("flight_time_scale",   "mul",  1.43),
    # --- v2 additions (indices 10..19) ------------------------------------
    ("gate_block",          "add",  1.00),
    ("gate_lift",           "add",  1.00),
    ("gate_punch",          "add",  1.00),
    ("gate_crossbody",      "add",  1.00),
    ("x_hit_coarse",        "add",  0.30),
    ("reach_expand_y",      "add",  1.00),
    ("reach_expand_z",      "add",  1.00),
    ("normal_weight_scale", "mul", 16.00),
    ("swing_window_scale",  "mul",  2.00),
    ("swing_dir_pitch",     "add",  0.60),
)
PARAM_SPACES = {"v1": PARAM_SPECS, "v2": PARAM_SPECS_V2}
```

| # | name | kind | bound | value at u=0 (nominal) | what it does |
|---|---|---|---|---|---|
| 0–9 | *(v1, unchanged)* | | | identity | as today |
| 10 | `gate_block` | add | 1.00 | 0.0 | blend weight `4·relu(u)` toward `block` |
| 11 | `gate_lift` | add | 1.00 | 0.0 | …toward `lift` |
| 12 | `gate_punch` | add | 1.00 | 0.0 | …toward `punch` |
| 13 | `gate_crossbody` | add | 1.00 | 0.0 | …toward `crossbody` |
| 14 | `x_hit_coarse` | add | 0.30 | 0.0 m | coarse hitting-plane depth, stacks on `x_hit_offset` |
| 15 | `reach_expand_y` | add | 1.00 | 0.0 | lerp the y clamp toward (-0.80, +0.60); negatives clip to 0 |
| 16 | `reach_expand_z` | add | 1.00 | 0.0 | lerp the z clamp toward (0.58, 1.45) |
| 17 | `normal_weight_scale` | mul | 16.00 | 1.0 | `w_normal = clip(8·s, 0.5, 16)`; face-vs-position trade |
| 18 | `swing_window_scale` | mul | 2.00 | 1.0 | scales `t_pre`, `speed_buffer`, `t_follow` together (dwell) |
| 19 | `swing_dir_pitch` | add | 0.60 | 0.0 rad | rotates `swing_vel` up/down, independent of the face |

`K_PARAMS_V2 = 20`. Notes:

* Every `add` bound is chosen so `u = ±1` is *usable*, not merely legal:
  `x_hit_coarse` ±0.30 covers the measured optimum; `swing_dir_pitch` ±0.60 rad
  spans flat drive to lifted push; the gates saturate the blend at 80%.
* `normal_weight_scale`'s `mul` is two-sided by construction (`exp(u·ln 16)` ∈
  [1/16, 16]); the useful direction is *down* (toward position priority), and the
  `clip(·, 0.5, 16)` in the hitter both keeps the useful range and stops
  `w_normal = 128` from making the DLS ill-conditioned.
* `swing_dir_yaw`, `backswing_scale` and `risk_scale` were considered and
  **dropped** into the templates: each is strongly correlated with a template's
  identity, and 20 coordinates is already double v1's for a bandit that needed
  ~80 k episodes to move 2.5 points at K = 10.

Back-compatibility, concretely: `load_checkpoint` already returns
`ckpt["param_specs"]`, `ParamActorCritic` is already sized from `ckpt["k_params"]`,
and `unsquash_to_params(u, specs)` already takes the specs explicitly. Loading a v1
checkpoint under v2 code therefore builds a 10-output net, decodes 10 names, and
`set_serve_params` leaves all ten v2 keys at their nominal — which by §3.4 is the
v1 hitter exactly. Add `ckpt["param_space"] = "v1"|"v2"` for legibility (with
`.get(..., "v1")` for old files) and mirror `param_names/kinds/bounds` into the
`.npz` export, which already carries them.

---

## 5. Reward

Today: `+2` contact, `+legal_bonus` legal, `+3·exp(-err²/0.5²)` aim, `-1` post-contact
fail, plus action-delta and velocity penalties.

The aim term is doing almost nothing. At `legal_bonus = 10` (the recipe in
`work/tt/rl/param/RESULT.md`), moving from a 0.60 m landing error to a perfect hit is
worth `3·(1 - exp(-0.36)) = 0.9` — 9% of the legal bonus. And the miss is systematic:
for param_s3's legal returns the landing median is **(0.78, 0.21)** against a target
of **(1.15, 0.55)** — 0.37 m short and 0.34 m inboard, i.e. the returns are
consistently under-hit. The policy already spends `aim_y_offset` (median +0.24 m,
aiming *off the table*) and pins `flight_time_scale` at its fast bound trying to
correct this. It is asking for more speed and more depth than the shape gives it.

Proposed, all defaulting to exactly today's behaviour:

```python
# tt_sim/rl/env.py -- VecTT.__init__ additions
aim_bonus: float = 3.0        # coefficient on the aim term  (v1 hard-coded 3.0)
aim_sigma: float = 0.5        # kernel width, metres         (v1 hard-coded 0.5)
target_hit_bonus: float = 0.0 # step reward at err <= TARGET_HIT_RADIUS_M (v1: absent)
```

```python
if legal_now and not self._legal_latched[i]:
    reward += getattr(self, "legal_bonus", 5.0)
    err = scene.landing_error_m(events, self._targets[i])
    if not np.isnan(err):
        sigma = getattr(self, "aim_sigma", 0.5)
        reward += getattr(self, "aim_bonus", 3.0) * float(np.exp(-err ** 2 / sigma ** 2))
        thb = getattr(self, "target_hit_bonus", 0.0)
        if thb > 0.0 and err <= TARGET_HIT_RADIUS_M:
            reward += thb
    self._legal_latched[i] = True
```

`getattr` with the v1 default is required, not stylistic: `VecTTResidual` builds its
own `__init__` without calling `super().__init__()` and inherits `_reward_and_done`
(the existing `intercept_reward`/`vel_penalty` lines already do this for the same
reason).

Recommended settings, and why the step bonus:

* `--aim-bonus 8 --aim-sigma 0.30 --target-hit-bonus 4` alongside `--legal-bonus 10`.
  `aim_sigma = 0.30` makes the kernel width equal to `evaluate.TARGET_HIT_RADIUS_M`,
  so the gradient is steepest exactly across the metric's boundary rather than being
  nearly flat there.
* The step `target_hit_bonus` rewards *the metric being reported*. A pure Gaussian
  kernel is scale-free and lets the policy trade 5 cm of error against a fraction of
  a legal return everywhere; a step at 0.30 m concentrates the pressure where the
  score changes. It is discontinuous, which is fine for a score-function bandit.
* Do this as **phase two**: raising aim pressure from scratch competes with getting
  the ball in at all. Sequence: (1) train v2 to convergence on legal at
  `--aim-bonus 3`; (2) `--resume-from` that checkpoint with the aim knobs raised.
  `--resume-from` already exists and already carries `feat_rms`.

Nothing else is needed for `target_hit_rate` — the `punch` template supplies the
depth the current shape cannot reach, and `aim_x/y_offset` become free to do
placement once they stop being bias correctors.

---

## 6. Implementation plan

### 6.1 `tt_sim/policies/ik_hitter.py`

1. Add `SHAPE_FIELDS`, `SWING_TEMPLATES` (5 dicts; `drive` holds the current
   constructor defaults verbatim), `TEMPLATE_ORDER`, `GATE_GAIN`, `REACH_MAX_Y/Z`.
2. Add `blend_shape(gates)` and `reach_box(shape, ey, ez)` exactly as in §2.4/§3.1,
   each with the explicit nominal early-out.
3. `SERVE_PARAM_NOMINAL_V2 = {**SERVE_PARAM_NOMINAL, "gate_block": 0.0, …}`.
   `SERVE_PARAM_NOMINAL` itself is left alone so v1 callers and the existing
   `set_serve_params` error message keep working; `set_serve_params` validates
   against the v2 superset and resets unnamed keys to their v2 nominal.
4. `_maybe_replan` computes `shape = blend_shape(...)` once, then:
   `x_hit_eff`, `p_hit` clamped by `reach_box`, `flight_time_eff`, `speed_cap_eff`,
   `risk_gain`, `n_back` rotated by `shape["dir_pitch"] + swing_dir_pitch` and
   `shape["dir_yaw"]`, `n_des = _rotate_normal(n_back, shape["face_pitch"] +
   pitch_offset, shape["face_yaw"] + yaw_offset)`, and `pred_delay` from the shape
   in place of the bare `PREDICTION_DELAY_S` (the module constant stays as
   `drive`'s value, so `tt_sim/rl/intercept.py` is untouched).
   Store the shape on `self._plan` so `_swing_target` and `act` read window widths
   from it.
5. `_swing_target` / `act` read `t_pre`, `speed_buffer`, `t_follow` (each times
   `swing_window_scale`) and `backswing` from the plan's shape instead of `self.*`.
   Keep the `self.*` constructor kwargs as the `drive` defaults so every existing
   caller and test is unaffected.
6. `_solve_ik` takes `w_normal` as an argument (default `self.w_normal`), the caller
   passes `clip(self.w_normal * normal_weight_scale, 0.5, 16.0)` with a short-circuit
   at scale == 1.0.

Only terse WHY comments — one on each nominal early-out saying it exists for
bit-exact v1 parity, one on `GATE_GAIN` saying 80% is deliberate, one on the
`w_normal` clip saying it bounds DLS conditioning.

### 6.2 `tt_sim/rl/param_residual.py`

7. Add `PARAM_SPECS_V2`, `PARAM_SPACES`, `K_PARAMS_V2`; keep `PARAM_SPECS`,
   `PARAM_NAMES`, `K_PARAMS` pointing at v1 so nothing importing them changes.
   Relax the module-level assertion to check against `SERVE_PARAM_NOMINAL_V2`.
8. `VecTTParam.__init__(..., param_specs=PARAM_SPECS)`; use
   `len(self.param_specs)` for the zero vector and pass the specs to
   `unsquash_to_params`.
9. `save_checkpoint(..., param_space="v1")` writes `param_space` and the specs it
   was actually trained with (currently it hard-codes `PARAM_SPECS` — that is a
   latent bug the moment a second space exists, and it must be fixed in the same
   change). `load_checkpoint` returns `(model, feat_rms, specs)` as it does now,
   defaulting `param_space` to `"v1"`.
10. `hitter_from_u(u, specs=...)` already threads specs; no change.

### 6.3 `tt_sim/rl/param_train.py`

11. `--param-space {v1,v2}` (default **`v1`**, so the existing recipe is
    reproducible), `--aim-bonus`, `--aim-sigma`, `--target-hit-bonus`.
12. `train()` resolves `specs = PARAM_SPACES[param_space]`, sizes
    `ParamActorCritic(FEATURE_DIM, len(specs))`, and passes both the specs and the
    reward knobs to `VecTTParam`.
13. `--resume-from` a v1 checkpoint into a v2 run: copy `model_state` for every
    tensor whose shape matches, and for the actor's last `Linear` copy the first 10
    rows of `weight`/`bias` into the 20-row head, leaving rows 10–19 at their
    zero init. Result: the v2 net's first 10 coordinates reproduce the v1 policy
    and the 10 new ones start at nominal, so **resuming starts from exactly the
    70.5% checkpoint's behaviour**. Same for `log_std` (pad with `log_std_init`).
    This is the single highest-value line in the plan — it turns v2 from a cold
    restart into a continuation.

### 6.4 `tt_sim/rl/env.py`, `tt_sim/rl/policy.py`, `tt_sim/policies/param_residual_np.py`

14. `env.py`: the three reward knobs from §5, read via `getattr` in
    `_reward_and_done`.
15. `policy.py` / `param_residual_np.py`: **no structural change** — both already
    decode from the checkpoint's stored specs and hand the resulting dict to
    `set_serve_params`. Add `param_space` to the `.npz` in
    `scripts/export_param_policy.py` for legibility.

### 6.5 Tests (`tt_sim/tests/`)

16. `test_v2_nominal_parity` — `IKHitterPolicy().set_serve_params(SERVE_PARAM_NOMINAL_V2)`
    and `hitter_from_u(zeros(20), PARAM_SPECS_V2)` both `np.array_equal` the plain
    hitter over 150 steps. Extends the existing
    `test_nominal_serve_params_match_plain_hitter`.
17. `test_v1_specs_unchanged` — `PARAM_SPECS_V2[:10] == PARAM_SPECS`, element for
    element including bounds.
18. `test_v1_checkpoint_reproduces_its_actions_bit_for_bit` — the real guard. Commit a
    small fixture `tt_sim/tests/data/param_s3_actions.npz` holding
    `(seeds, actions[150, 29])` generated from `assets/policies/param_residual_s3.npz`
    **before** the v2 change, then assert `np.array_equal` on a fresh rollout. A
    round-trip test (save v1 ckpt → load → roll) only proves self-consistency; a
    committed golden proves nothing regressed. `test_param_residual_np.py`'s existing
    2e-3 tolerance is for torch-vs-numpy matmul order and is *not* a substitute.
19. `test_gates_at_zero_return_the_drive_template` — `blend_shape([0,0,0,0]) is
    SWING_TEMPLATES["drive"]` (identity, not equality: proves the early-out fires).
20. `test_blend_is_within_the_template_hull` — every blended field lies in
    `[min, max]` over the templates, for random gate vectors.
21. `test_every_v2_param_changes_the_swing` — the existing loop, over
    `PARAM_SPECS_V2`. Note the gates need `u > 0` to have any effect, so drive the
    loop at `u = +0.9` (it already does).
22. `test_reach_extremes_are_collision_free` — solve IK at each template's box
    corners under `reach_expand_* = 1`, assert `mj_forward` gives `ncon == 0` and
    every IK joint inside `jnt_range`. This is the guard for the 28-contact branch
    I found at `y = +0.55`.
23. `test_v2_poses_survive_the_training_action_pipeline` — for every template at
    `u = ±1`, assert `map_action(target_to_action(q), jnt_range)` round-trips to
    within 1e-9 of `q`. See risk R3.
24. `test_reward_knobs_default_to_v1` — a `VecTT` built with no reward kwargs
    produces exactly the pre-change reward on a scripted event dict.

Gate: `PYTHONPATH=. ./.venv/bin/python -m pytest -q tt_sim/tests` (60 pass today;
expect 60 + ~9).

### 6.6 Suggested order

`ik_hitter.py` templates + parity tests (16, 19, 20) → `param_residual.py` specs +
tests (17, 18, 21) → reach expansion + safety tests (22, 23) → `env.py` reward +
test (24) → `param_train.py` flags + warm-start (13) → smoke run
`--param-space v2 --minutes 5 --n-envs 4` → real run.

---

## 7. Risks

**R1 — The kinematic ceiling (§0).** ~20–24% of serves are unreachable. If v2 lands
at 78–80% that is the ceiling, not a tuning failure; do not spend days chasing 90%.
*Mitigation:* measure the ceiling on the training serve set first (the sweep in §0 is
~10 s for 300 serves) and quote every result against it.

**R2 — K doubles from 10 to 20.** v1 needed ~80 k episodes (120 min) to gain 2.5
points at K = 10. A cold v2 run may look *worse* than v1 for a long time.
*Mitigation:* the warm-start in step 13. Always start v2 from `s3/best.pt`.
Secondary: consider `--log-std-init -1.5` for the gate coordinates only.

**R3 — Train/eval pipeline asymmetry.** `VecTTParam.step` sends the hitter's target
through `target_to_action` (clips the action to ±1, i.e. **±3 rad from
`STAND_POSE`**, and zeroes the legs) then `map_action`; `tt_sim/evaluate.py` feeds
the hitter's absolute targets straight to `TableTennisEnv.step`. Any v2 pose more
than 3 rad from stand therefore *trains* differently from how it *evaluates*.
Measured: the worst excursion at the reach extremes is 2.05 rad
(`right_shoulder_roll`), so there is margin — but `right_shoulder_pitch`
(stand 0.2, range −3.089) is the one joint that can exceed it.
*Mitigation:* test 23 makes this a hard gate.

**R4 — Self-collision on the crossbody branch.** Reaching `y = +0.55` has both a
clean and a 28-contact IK solution; the runtime solver is local and warm-started, so
it can commit to the bad one mid-swing. *Mitigation:* test 22, plus keep the
crossbody box's lower edge at −0.50 so the blend never sweeps the arm across the
torso within one swing.

**R5 — `PREDICTION_DELAY_S` was calibrated at `x_hit = -1.15`.** v2 planes reach
-1.58. *Mitigation:* per-template `pred_delay` (already in the shape vector), and
`t_hit_offset` as trim. If it bites, re-run the delay calibration at −1.40 rather
than widening `t_hit_offset`'s bound.

**R6 — Raising the aim reward costs legal returns.** *Mitigation:* the two-phase
schedule in §5; always report both metrics; `best.pt` is selected on legal rate
today, so phase two needs its selection criterion changed (or a second
`best_target.pt`) or it will never checkpoint the placement gains.

**R7 — Checkpoint-decoding regressions are silent.** A v1 `.npz`/`.pt` that decodes
with the wrong specs produces a *plausible* hitter, not a crash. *Mitigation:*
test 18's committed golden array, and fixing `save_checkpoint`'s hard-coded
`PARAM_SPECS` (step 9) in the same commit that adds a second space.

**R8 — Blend degeneracy.** The policy may sit at a mid-blend that is worse than
either endpoint. *Mitigation:* log the per-episode gate vector into
`train_log.csv` (or a sidecar `.npz`) so the blend distribution is inspectable;
if it collapses, raise `GATE_GAIN` toward a harder selector before reaching for
argmax.

---

## Implementation deviations (recorded 2026-08-29, after the adversarial review)

- **`REACH_MAX_Y` is `(-0.90, 0.15)`, not `(-0.80, 0.60)`.** The runtime solver (warm-started DLS from the current
  pose) self-collides for every commanded y ≥ +0.30 (23–45 torso/arm contacts, 0.14–0.37 m residual); the +y half of
  the workspace is not opened. Measured again by the review: 23–29 contacts, 0.28–0.42 m residual at y ≥ +0.30.
- **`crossbody` ships without its +y box and without its yaw biases.** With the box shut the template only applies
  on the forehand side, where a yawed face is an aim error the per-serve aim already solves: at full gate the shipped
  yaws scored 2–3/24 legal vs 7–9/24 nominal; zeroing both yaws scored best. What remains is a deeper, slightly
  lifted, slower-flight drive variant. Rename or re-purpose in a v3 space.
- **Face angles are relative to the (dir-rotated) swing direction**, as in v1 (`n_des = rotate(n_back, …)`), so a
  template's commanded face elevation is `dir_pitch + face_pitch` (lift: 0.85 rad, not 0.30). Documented, not changed.
- **Reach-expansion and face-weight knobs do nothing on this serve distribution** (noise probe, 96 serves per cell:
  legal rate identical at every noise level) — the clamped serves are physically out of reach regardless of the
  clamp. Candidates for removal in a v3 space.
- **Exploration on the new knobs must be small**: the trainer default std 0.37 collapses the warm-started policy to
  22 % during training; std 0.05–0.08 (`--log-std-init-new -3.0/-2.5`) is what the runs use.
- The collision guard (test 22) covers the two shallow planes and five face normals; deeper planes opened by
  `x_hit_coarse` self-collide on many points with this solver and are not gated.
