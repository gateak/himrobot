# Glossary — the words this project uses

Plain definitions, each tied to what it means *here*. Read top to bottom once; then use it as a lookup.

## The task

- **Legal return** — the robot's paddle hits the ball, the ball crosses the net above net height and lands on the
  opponent's half. The benchmark's primary score is the percentage of serves returned legally.
- **Target hit / landing error** — whether the return lands inside the target zone, and how far (metres) from the
  target point it lands. Secondary scores.
- **Reaction latency** — time from the serve appearing to the robot's first meaningful movement. Ours: ~25 ms
  (learned policy) vs ~520 ms (hand-coded hitter).
- **Serve distribution** — the range of serves the launcher produces (speed, height, where it lands). Ours is
  medium-speed, no spin, forehand side only; HIM's official one is unknown (open question).
- **Fixed pelvis / fixed base** — the robot's hips are bolted in place; the policy controls arms and waist only.
  HIM's development scene is fixed-pelvis. "Free-standing" or "whole-body" means the robot must also balance.

## The pieces

- **Policy** — the thing that answers "given what I see, what do I do?" — observation in, joint targets out, 50 times
  a second. Can be an algorithm, a neural network, or both. Ours is both.
- **Hitter (IK hitter)** — the hand-written policy: predict where the ball will be, choose a paddle pose, move the arm
  there with inverse kinematics. No learning. 46.5 % legal returns. `tt_sim/policies/ik_hitter.py`.
- **Inverse kinematics (IK)** — computing the joint angles that put the hand (paddle) at a desired position and angle.
- **Parameter residual** — our best policy: the hitter plus a small network that, once per serve, picks ten numbers
  (timing, intercept offsets, swing speed, paddle angles, aim) that adjust the hitter for that ball. 70.5 % legal.
- **Per-step residual** — the earlier attempt: a network that nudged all 29 joints every 20 ms on top of the hitter.
  Reached 53.5 %, then got worse with more training. Superseded.
- **Checkpoint** (`best.pt`, `final.pt`, `.npz`) — a saved copy of a policy's learned numbers. The deliverable.
- **Seed** — the random-number starting point. "Seed 11, 200 serves" means the same 200 serves every time, so two
  policies can be compared fairly. Training seeds (s0, s1, …) are independent training runs of the same recipe.
- **Observation / obs** — what the policy is given each step: joint angles and speeds, ball position and velocity,
  paddle pose. **Action** — what it returns: 29 desired joint angles.
- **Runtime contract** — the written list of observation and action signals, units and rate that HIM and the
  package agree on (`.him/contracts/*.json`).

## Learning

- **Reinforcement learning (RL)** — training by trial and score: the policy tries a serve, gets a reward, adjusts.
- **Reward** — the score signal during training (+10 legal return, +2 contact, small penalties). A wrong reward
  trains the wrong thing: a velocity penalty once taught the robot to swing slower instead of better.
- **PPO** — the specific RL algorithm we use (a standard, stable one). Not a model; a training method.
- **Exploration noise** — the random jitter added during training so the policy tries new things. Too much of it
  and the policy learns to hit *with a wobbly arm*, which scores worse with a steady one. The per-step residual
  died of this; the parameter residual doesn't have the problem (one decision per serve).
- **Deterministic evaluation** — scoring the policy with the noise off, on fixed serves. The only numbers that count.
- **Behaviour cloning / imitation learning** — training a policy to copy demonstrations (our day-one experiment on
  the motion-capture data). Not RL.
- **Domain randomization / physics jitter** — varying bounce, friction, ball mass each episode so the policy doesn't
  over-fit one exact simulator. Built (`physics_jitter`), not yet used in a scored run.
- **Observation noise / delay** — adding camera-like error and lag to what the policy sees, so it survives a real
  ball tracker. Built (`--obs-noise`, `--obs-delay-steps`), not yet used.
- **Sim-to-real** — the gap between the simulator and the real robot; closed with the two items above, a real
  perception front-end, and hardware tests.
- **Neural network / MLP** — layers of numbers multiplying inputs to outputs. Ours are tiny (5k–200k numbers).
  **Transformer** — a large network type for sequences and images; not used here yet.

## Simulation and compute

- **MuJoCo** — the physics simulator (CPU). HIM's dev scene and ours both use it. `tt_sim/` is our scene.
- **MJX** — MuJoCo rewritten to run thousands of simulations at once on a GPU. Requires porting the scene. Not done.
- **Isaac Lab / Isaac Sim** — NVIDIA's GPU simulator and RL framework; what whole-body (standing, stepping) policies are
  trained in. Needs an NVIDIA GPU. HIM runs its own Isaac validation lane on selected policies.
- **CPU vs GPU** — plain MuJoCo only runs on CPU cores, so more cores = more parallel training runs; a GPU helps
  only after the MJX/Isaac port or for vision networks.
- **Box / SSH machine** — a rented computer from HIM (RunPod), billed in credits per hour (93 for 24 GB GPU,
  186 for 48 GB), maximum 8 hours, its disk wiped when it stops. Our training runs there push results to HIM every
  30 minutes and stop the machine themselves.
- **Credits** — HIM's currency for compute. A failed HIM job is refunded; a box bills for time used.
- **cgroup memory cap** — the RAM limit on a rented box (51 or 94 GB here). Exceed it and processes are killed
  silently.

## HIM platform

- **HIM Arena** — the platform: data, compute, the benchmark leaderboard, the real-robot grant.
- **Package (`.him/`)** — the description of how HIM runs our code: entrypoints, contracts, tests, licences.
- **Entrypoint** — a named command in the package. `evaluate` runs our sim with our policy and writes a scorecard;
  `inference` is the door through which HIM's own evaluator talks to our policy; `train` runs training.
- **Guided job / `him run`** — HIM executing an entrypoint on its servers. Ours run, score and render, then stall at
  `publishing_outputs` (HIM-side; refunded).
- **Evaluator** — the scoring program. Ours (`tt_sim/evaluate.py`) is a proxy; only HIM's isolated evaluator produces
  official scores, on hidden serves.
- **Challenge version / `him arena submit`** — the official leaderboard entry. Requires an open version (open question).
- **Hardware review** — HIM's safety and compatibility gate before a policy touches the real G1.
- **Dataset push** — how we store results on HIM (`himrobot-rl-checkpoints`), so they survive a box being wiped.

## The robot

- **Unitree G1** — the 29-degree-of-freedom humanoid (12 leg joints, 3 waist, 14 arm). HIM's real one is "Adam".
- **Menagerie model** — the official MuJoCo description of the G1 we simulate (`assets/unitree_g1/`).
- **Motion capture (mocap) / retargeting** — recording a human's swings (Rokoko suit) and mapping them onto the G1's
  joints (`sim/gmr_bvh_to_g1_headless.py`). The motion prior for the future whole-body controller.
- **Orbbec / ball tracker** — the RGB-D cameras and our detector that find the ball in real footage
  (`sim/ball_detect.py`, `sim/ball_3d.py`). The perception front-end for the real robot.
