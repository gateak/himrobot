"""Roll himrobot.policies.ik_hitter.IKHitterPolicy over N random serves in
TableTennisEnv and record (obs, action) pairs for behavior cloning.

Coordinates with himrobot/rl/env.py (already present when this module was
written -- see the module docstring's "coordinate by reading env.py"
instruction): observations use env.flatten_obs's exact 99-dim layout
(joint_pos-STAND_POSE[29], joint_vel*0.1[29], ball.position[3],
ball.velocity*0.2[3], paddle.position[3], target[2], phase=time_s[1],
prev_action[29]), and the recorded "action" is target_to_action's inverse
of env.map_action -- the [-1,1]^29 action that, fed back through
env.map_action, reproduces the IK hitter's actual absolute joint target as
closely as ACTION_SCALE's 3.0 rad (waist/arms) / 0.0 rad (legs, frozen)
clip allows. This makes demos_seed0.npz directly usable by himrobot/rl/bc.py
(this package) and any future himrobot/rl/ppo.py `--init-from` warm start,
since both consume/produce actions in exactly VecTT's convention.

Only steps from trials where the IK hitter made contact_paddle are kept in
the obs/act arrays (a non-contact trial's steps are pure "swing and miss",
poor imitation targets for a paddle-swing behavior). Every trial -- kept or
not -- is still labelled in trial_contact/trial_legal so nothing about the
dropped trials is silently lost; see roll_demos's docstring for the exact
.npz layout.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from himrobot.policies.ik_hitter import IKHitterPolicy
from himrobot.rl.env import ACT_DIM, ACTION_SCALE, MAX_EPISODE_STEPS, OBS_DIM, flatten_obs
from himrobot.sim.scene import DEFAULT_PARAMS, STAND_POSE, TableTennisEnv, legal_return, sample_serve, target_point

N_TRIALS_DEFAULT = 2000
SEED_DEFAULT = 0

# Avoid a 0/0 on the 12 frozen-leg dims (ACTION_SCALE == 0 there); the IK
# hitter always targets STAND_POSE for legs anyway (delta == 0), so the
# divisor's value on those dims never actually matters -- see target_to_action.
_SCALE_SAFE = np.where(ACTION_SCALE > 0, ACTION_SCALE, 1.0)
_NONLEG = ACTION_SCALE > 0


def target_to_action(joint_target: np.ndarray) -> np.ndarray:
    """Inverse of himrobot.rl.env.map_action's forward direction (action in
    [-1,1]^29 -> STAND_POSE + action * ACTION_SCALE, clipped to jnt_range).

    ``joint_target`` is an absolute joint-position target in radians (what
    a Policy.act returns, e.g. IKHitterPolicy's output fed straight to
    TableTennisEnv.step). Frozen (ACTION_SCALE == 0) dims always map to
    action 0 ("legs zero", per the task spec). Clipped to [-1, 1]: the IK
    hitter's swings can carry a waist/arm joint's target beyond
    STAND_POSE +/- 1 rad (see demos_seed0_summary.json's
    action_saturation_fraction for how often), which a tanh-output network
    could never reproduce past +/-1 anyway.
    """
    delta = np.asarray(joint_target, dtype=np.float64) - STAND_POSE
    action = np.where(_NONLEG, delta / _SCALE_SAFE, 0.0)
    return np.clip(action, -1.0, 1.0)


def roll_demos(
    n_trials: int = N_TRIALS_DEFAULT,
    seed: int = SEED_DEFAULT,
    params: dict | None = None,
    episode_steps: int = MAX_EPISODE_STEPS,
) -> dict:
    """Roll IKHitterPolicy over n_trials independently-seeded random serves
    (trial k's serve is sampled from np.random.default_rng(seed + k), so
    this is reproducible and independent of n_trials/iteration order).

    Returns a dict:
      obs               float32 [n_kept_steps, 99]  -- env.flatten_obs, one row/control step
      act               float32 [n_kept_steps, 29]  -- target_to_action(IK hitter's output)
      episode_starts    int64   [n_kept_episodes]   -- row index into obs/act where each
                                                         kept (contact) episode begins
      legal             bool    [n_kept_episodes]   -- that kept episode's legal_return(events)
      trial_contact     bool    [n_trials]           -- contact_paddle, every trial (kept or not)
      trial_legal       bool    [n_trials]           -- legal_return(events), every trial
      action_saturation_fraction  float  -- fraction of kept (waist/arm) action components
                                             that hit the +/-1 clip in target_to_action
      target, n_trials, seed, params, episode_steps  -- run metadata
    """
    params = {**DEFAULT_PARAMS, **(params or {})}
    env = TableTennisEnv(seed=seed, params=params)
    policy = IKHitterPolicy()
    target = target_point(params["target_zone"], params["target_depth"])

    obs_rows: list[np.ndarray] = []
    act_rows: list[np.ndarray] = []
    episode_starts: list[int] = []
    kept_legal: list[bool] = []
    trial_contact = np.zeros(n_trials, dtype=bool)
    trial_legal = np.zeros(n_trials, dtype=bool)
    n_saturated = 0
    n_nonleg_total = 0

    for trial in range(n_trials):
        trial_seed = seed + trial
        rng = np.random.default_rng(trial_seed)
        serve = sample_serve(rng, env.params)
        obs = env.reset(serve=serve)
        policy.reset({"serve": serve, "params": env.params, "trial": trial, "seed": trial_seed})

        prev_action = np.zeros(ACT_DIM, dtype=np.float64)
        step_obs: list[np.ndarray] = []
        step_act: list[np.ndarray] = []
        info: dict = {}

        for _ in range(episode_steps):
            joint_target = policy.act(obs)  # absolute radians, straight to env.step
            action = target_to_action(joint_target)

            step_obs.append(flatten_obs(obs, prev_action, target, obs["time_s"]))
            step_act.append(action.astype(np.float32))

            raw = (np.asarray(joint_target, dtype=np.float64) - STAND_POSE) / _SCALE_SAFE
            n_saturated += int(np.sum(np.abs(raw[_NONLEG]) > 1.0))
            n_nonleg_total += int(np.sum(_NONLEG))

            obs, info = env.step(joint_target)
            prev_action = action

        contact = bool(info.get("contact_paddle"))
        legal = bool(legal_return(info))
        trial_contact[trial] = contact
        trial_legal[trial] = legal

        if contact:
            episode_starts.append(len(obs_rows))
            obs_rows.extend(step_obs)
            act_rows.extend(step_act)
            kept_legal.append(legal)

    obs_arr = np.asarray(obs_rows, dtype=np.float32).reshape(-1, OBS_DIM)
    act_arr = np.asarray(act_rows, dtype=np.float32).reshape(-1, ACT_DIM)

    return {
        "obs": obs_arr,
        "act": act_arr,
        "episode_starts": np.asarray(episode_starts, dtype=np.int64),
        "legal": np.asarray(kept_legal, dtype=bool),
        "trial_contact": trial_contact,
        "trial_legal": trial_legal,
        "action_saturation_fraction": (n_saturated / n_nonleg_total) if n_nonleg_total else 0.0,
        "target": np.asarray(target, dtype=np.float64),
        "n_trials": n_trials,
        "seed": seed,
        "params": params,
        "episode_steps": episode_steps,
    }


def save_demos(data: dict, npz_path: str) -> None:
    os.makedirs(os.path.dirname(npz_path) or ".", exist_ok=True)
    np.savez(
        npz_path,
        obs=data["obs"],
        act=data["act"],
        episode_starts=data["episode_starts"],
        legal=data["legal"],
        trial_contact=data["trial_contact"],
        trial_legal=data["trial_legal"],
        target=data["target"],
    )


def build_summary(data: dict, wall_time_s: float) -> dict:
    n_trials = data["n_trials"]
    n_contact = int(data["trial_contact"].sum())
    n_legal = int(data["trial_legal"].sum())
    return {
        "n_trials": n_trials,
        "seed": data["seed"],
        "params": data["params"],
        "contact_rate": n_contact / n_trials if n_trials else 0.0,
        "legal_rate": n_legal / n_trials if n_trials else 0.0,
        "n_contact_trials": n_contact,
        "n_legal_trials": n_legal,
        "n_kept_episodes": int(len(data["episode_starts"])),
        "n_kept_steps": int(data["obs"].shape[0]),
        "action_saturation_fraction": data["action_saturation_fraction"],
        "note": (
            "contact_rate/legal_rate are measured over ALL n_trials serves "
            "(not just the contact-only episodes kept in the .npz's obs/act "
            "arrays). IKHitterPolicy's own module docstring reports ~70% "
            "contact / ~35% legal_return_rate at n=20/seed=0/default params; "
            "these numbers are this exact run's measurement, from this file."
        ),
        "wall_time_s": wall_time_s,
    }


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=N_TRIALS_DEFAULT)
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--params", type=str, default="{}", help="JSON dict merged over himrobot.sim.scene.DEFAULT_PARAMS")
    parser.add_argument("--out", type=str, default=os.path.join("work", "tt", "rl"))
    parser.add_argument("--tag", type=str, default="seed0", help="output filename tag: demos_<tag>.npz")
    args = parser.parse_args(argv)

    params = json.loads(args.params)
    start = time.time()
    data = roll_demos(n_trials=args.n_trials, seed=args.seed, params=params)
    wall_time_s = time.time() - start

    npz_path = os.path.join(args.out, f"demos_{args.tag}.npz")
    summary_path = os.path.join(args.out, f"demos_{args.tag}_summary.json")
    save_demos(data, npz_path)
    summary = build_summary(data, wall_time_s)
    os.makedirs(args.out, exist_ok=True)
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
