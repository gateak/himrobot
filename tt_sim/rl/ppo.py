"""Compact PPO trainer for tt_sim.rl.env.VecTT.

    python -m tt_sim.rl.ppo --n-envs 64 --total-steps 200000000 --seed 0 \\
        --out work/tt/rl/run1 [--init-from CKPT.pt] [--minutes 180] \\
        [--eval-every-steps 20000]

MLP actor-critic (obs(99) -> 256 -> 256 -> 29, tanh hidden activations,
state-independent log-std; the actor is tt_sim.rl.nets.Actor, shared with
tt_sim/rl/bc.py so --init-from can warm-start from a behavior-cloning
checkpoint), GAE(lambda=0.95, gamma=0.99), PPO clip 0.2, 4 epochs/update,
minibatch 2048, entropy coef 0.0, Adam lr 3e-4, running obs normalization,
action clipping to [-1, 1] at the env boundary (the actor's own tanh
output is already bounded; the post-sample clip only guards the Normal
noise added around it). Runs on MPS if available, else CPU.

Writes ``train_log.csv`` to --out (step, wall_s, then mean_return,
legal_return_rate, contact_rate, ep_len over the last 200 completed
episodes, then det_legal_rate/det_contact_rate -- NaN unless
--det-eval-episodes is set), checkpoints every 10 minutes of wall time
(``checkpoint_<step>.pt``) and whenever the selection rate hits a new best
(``best.pt``; the deterministic rate when --det-eval-episodes is set, else
the last-200-episode legal_return_rate), and a ``final.pt`` at exit
(hard-capped by --minutes if given, else runs to --total-steps). A
checkpoint is loadable by tt_sim.evaluate via
``--policy tt_sim.rl.policy:PPOPolicy`` with ``TT_RL_CKPT=<path>`` (see
tt_sim/rl/policy.py).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from tt_sim.rl.env import ACT_DIM, OBS_DIM, VecTT
from tt_sim.rl.env import obs_dim as env_obs_dim
from tt_sim.rl.nets import Actor
from tt_sim.rl.residual import DEFAULT_RESIDUAL_SCALE, OBS_VERSION, RESIDUAL_OBS_DIM, VecTTResidual

HIDDEN = 256
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
EPOCHS = 4
MINIBATCH = 2048
ENTROPY_COEF = 0.0
VALUE_COEF = 0.5
LR = 3e-4
MAX_GRAD_NORM = 0.5
CHECKPOINT_EVERY_S = 600.0
EPISODE_WINDOW = 200


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ActorCritic(nn.Module):
    """actor: tt_sim.rl.nets.Actor (obs(99) -> 256 -> 256 -> 29, tanh
    hidden activations + tanh output squashing to [-1,1]) -- the shared
    architecture nets.py documents so a tt_sim/rl/bc.py behavior-cloning
    checkpoint's state_dict loads into it unmodified for --init-from (see
    load_bc_actor_into below). critic: a separate same-shaped MLP head
    (bc.py's Actor has no value output). log_std is a free (act_dim,)
    parameter, state-independent, not a network output."""

    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM, hidden: int = HIDDEN,
                 log_std_init: float = -0.5):
        super().__init__()
        self.actor = Actor(obs_dim, act_dim, hidden)
        self.log_std = nn.Parameter(torch.full((act_dim,), log_std_init))
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.actor(obs)
        value = self.critic(obs).squeeze(-1)
        return mean, value

    @torch.no_grad()
    def act(self, obs: torch.Tensor,
            deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, value = self.forward(obs)
        std = self.log_std.exp()
        if deterministic:
            return mean, torch.zeros(mean.shape[0], device=mean.device), value
        dist = Normal(mean, std)
        action = dist.sample()
        logp = dist.log_prob(action).sum(-1)
        return action, logp, value

    def evaluate_actions(self, obs: torch.Tensor,
                         action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, value = self.forward(obs)
        std = self.log_std.exp()
        dist = Normal(mean, std)
        logp = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return logp, entropy, value


class RunningMeanStd:
    """Welford/Chan-style running mean+var, used to normalize obs online
    (updated with every raw obs seen during rollout collection)."""

    def __init__(self, shape: tuple[int, ...], eps: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = eps

    def update(self, x: np.ndarray) -> None:
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        self.mean = new_mean
        self.var = m2 / tot_count
        self.count = tot_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -10.0, 10.0)


def compute_gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray,
                 last_values: np.ndarray, gamma: float = GAMMA,
                 lam: float = GAE_LAMBDA) -> tuple[np.ndarray, np.ndarray]:
    """rewards/values/dones: (T, N). last_values: (N,) bootstrap value for
    the step after the rollout. dones[t] gates the bootstrap at episode
    boundaries -- correct even under VecTT's auto-reset, since a done step's
    reward already includes any terminal reward and values[t+1]/last_values
    is multiplied by (1 - dones[t]) = 0 there."""
    T, N = rewards.shape
    advantages = np.zeros((T, N), dtype=np.float64)
    lastgaelam = np.zeros(N, dtype=np.float64)
    for t in reversed(range(T)):
        nextvalues = last_values if t == T - 1 else values[t + 1]
        nextnonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
        advantages[t] = lastgaelam
    returns = advantages + values
    return advantages, returns


def collect_rollout(vec: VecTT, model: ActorCritic, obs_rms: RunningMeanStd,
                     rollout_steps: int, device: torch.device, obs_dim: int = OBS_DIM):
    n = vec.n_envs
    obs_buf = np.zeros((rollout_steps, n, obs_dim), dtype=np.float32)
    act_buf = np.zeros((rollout_steps, n, ACT_DIM), dtype=np.float32)
    logp_buf = np.zeros((rollout_steps, n), dtype=np.float32)
    rew_buf = np.zeros((rollout_steps, n), dtype=np.float32)
    val_buf = np.zeros((rollout_steps, n), dtype=np.float32)
    done_buf = np.zeros((rollout_steps, n), dtype=np.float32)
    episode_stats: list[dict] = []

    obs = vec._last_obs.copy()
    for t in range(rollout_steps):
        obs_rms.update(obs.astype(np.float64))
        norm_obs = obs_rms.normalize(obs)
        obs_t = torch.as_tensor(norm_obs, dtype=torch.float32, device=device)
        action_t, logp_t, value_t = model.act(obs_t)
        action_np = action_t.cpu().numpy()
        clipped = np.clip(action_np, -1.0, 1.0)

        next_obs, reward, done, infos = vec.step(clipped)

        obs_buf[t] = obs
        act_buf[t] = action_np
        logp_buf[t] = logp_t.cpu().numpy()
        rew_buf[t] = reward
        val_buf[t] = value_t.cpu().numpy()
        done_buf[t] = done.astype(np.float32)
        episode_stats.extend(info["episode"] for info in infos if "episode" in info)

        obs = next_obs

    norm_obs = obs_rms.normalize(obs)
    obs_t = torch.as_tensor(norm_obs, dtype=torch.float32, device=device)
    with torch.no_grad():
        _, last_value_t = model.forward(obs_t)
    last_value = last_value_t.cpu().numpy()

    return obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf, last_value, episode_stats


def deterministic_eval(vec_eval: VecTT, model: ActorCritic, obs_rms: RunningMeanStd, device: torch.device,
                       max_steps: int = 600) -> tuple[float, float]:
    """Mean-action legal/contact rate (%) over one episode per env of a fixed-serve env.
    This is the number the evaluator scores; the noisy train-time rate is not (v4 box: it rose
    22->31% while the deterministic rate fell 53->35%). obs_rms is read, never updated."""
    n = vec_eval.n_envs
    done_once = np.zeros(n, dtype=bool)
    legal = np.zeros(n, dtype=bool)
    contact = np.zeros(n, dtype=bool)
    obs = vec_eval.reset()
    for _ in range(max_steps):
        obs_t = torch.as_tensor(obs_rms.normalize(obs), dtype=torch.float32, device=device)
        action_t, _, _ = model.act(obs_t, deterministic=True)
        obs, _, _, infos = vec_eval.step(np.clip(action_t.cpu().numpy(), -1.0, 1.0))
        for i, info in enumerate(infos):
            if "episode" in info and not done_once[i]:
                done_once[i] = True
                legal[i] = bool(info["episode"]["legal_return"])
                contact[i] = bool(info["episode"]["contact"])
        if done_once.all():
            break
    k = max(int(done_once.sum()), 1)
    return 100.0 * float(legal.sum()) / k, 100.0 * float(contact.sum()) / k


def ppo_update(model: ActorCritic, optimizer: torch.optim.Optimizer,
                obs_buf: np.ndarray, act_buf: np.ndarray, logp_buf: np.ndarray,
                adv_buf: np.ndarray, ret_buf: np.ndarray, obs_rms: RunningMeanStd,
                device: torch.device) -> dict[str, float]:
    obs_dim = obs_buf.shape[-1]
    b_obs = obs_rms.normalize(obs_buf.reshape(-1, obs_dim))
    b_act = act_buf.reshape(-1, act_buf.shape[-1])
    b_logp = logp_buf.reshape(-1)
    b_adv = adv_buf.reshape(-1)
    b_ret = ret_buf.reshape(-1)
    b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

    n = b_obs.shape[0]
    mb_size = min(MINIBATCH, n)
    idxs = np.arange(n)

    last = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    for _ in range(EPOCHS):
        np.random.shuffle(idxs)
        for start in range(0, n, mb_size):
            mb = idxs[start:start + mb_size]
            obs_t = torch.as_tensor(b_obs[mb], dtype=torch.float32, device=device)
            act_t = torch.as_tensor(b_act[mb], dtype=torch.float32, device=device)
            old_logp_t = torch.as_tensor(b_logp[mb], dtype=torch.float32, device=device)
            adv_t = torch.as_tensor(b_adv[mb], dtype=torch.float32, device=device)
            ret_t = torch.as_tensor(b_ret[mb], dtype=torch.float32, device=device)

            logp, entropy, value = model.evaluate_actions(obs_t, act_t)
            ratio = torch.exp(logp - old_logp_t)
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv_t
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * (ret_t - value).pow(2).mean()
            entropy_mean = entropy.mean()
            loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy_mean

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            last = {
                "policy_loss": float(policy_loss.item()),
                "value_loss": float(value_loss.item()),
                "entropy": float(entropy_mean.item()),
            }
    return last


def save_checkpoint(path: str, model: ActorCritic, obs_rms: RunningMeanStd, total_steps: int,
                     obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM, extra: dict | None = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "model_state": model.state_dict(),
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "obs_rms_mean": obs_rms.mean,
        "obs_rms_var": obs_rms.var,
        "obs_rms_count": obs_rms.count,
        "total_steps": total_steps,
    }
    ckpt.update(extra or {})
    torch.save(ckpt, path)


def load_checkpoint(path: str, device: torch.device) -> tuple[ActorCritic, RunningMeanStd]:
    # weights_only=False: our own checkpoints carry plain numpy arrays
    # (obs_rms mean/var) alongside the state_dict, which PyTorch >=2.6's
    # default weights_only=True unpickler rejects. Only ever load
    # checkpoints this trainer itself produced (see module docstring).
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = ActorCritic(ckpt["obs_dim"], ckpt["act_dim"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    obs_rms = RunningMeanStd((ckpt["obs_dim"],))
    obs_rms.mean = np.asarray(ckpt["obs_rms_mean"])
    obs_rms.var = np.asarray(ckpt["obs_rms_var"])
    obs_rms.count = ckpt["obs_rms_count"]
    return model, obs_rms


def load_bc_actor_into(model: ActorCritic, bc_path: str, device: torch.device) -> None:
    """--init-from bc.pt: warm-start the actor half of a fresh ActorCritic
    from a tt_sim/rl/bc.py behavior-cloning checkpoint (see nets.py/bc.py's
    docstrings for the checkpoint format -- a dict with a "model_state_dict"
    key holding a plain tt_sim.rl.nets.Actor state_dict). The critic and
    log_std stay at their fresh-init values: bc.py never trains a value
    function, so there is nothing to transfer for them."""
    ckpt = torch.load(bc_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)  # tolerate a bare Actor state_dict too
    model.actor.load_state_dict(state)


def train(n_envs: int, total_steps: int, seed: int, out: str, init_from: str | None = None,
          minutes: float | None = None, eval_every_steps: int = 20_000,
          rollout_steps: int = 128, params: dict | None = None,
          residual: bool = False, residual_scale: float = DEFAULT_RESIDUAL_SCALE,
          intercept_features: bool = False, intercept_reward: float = 0.0,
          vel_penalty: float = 0.0005, legal_bonus: float = 5.0, resume_from: str | None = None,
          obs_delay_steps: int = 0, obs_noise: float = 0.0, joint_noise: float = 0.0,
          det_eval_episodes: int = 0, log_std_init: float | None = None,
          log_std_final: float | None = None) -> dict:
    """residual=False reproduces the original (pre-residual) trainer
    exactly -- same VecTT, OBS_DIM, log_std init (-0.5) and actor init.
    residual=True instead builds a tt_sim.rl.residual.VecTTResidual (obs
    widened to RESIDUAL_OBS_DIM=128, see that module), initializes
    log_std to -1.6 (small noise -> step-zero behaviour close to the
    wrapped IK-hitter expert's own action) and zero-inits the actor's
    final Linear layer (weight+bias) so its tanh output -- the RL
    residual a_rl -- is exactly 0 at step zero, before any gradient
    update (VecTTResidual then sends exactly a_exp to the joints, per its
    own docstring's `a = clip(a_exp + residual_scale * a_rl, -1, 1)`)."""
    os.makedirs(out, exist_ok=True)
    device = pick_device()
    torch.manual_seed(seed)
    np.random.seed(seed)

    if residual and (intercept_features or intercept_reward > 0.0):
        raise NotImplementedError("--intercept-features/--intercept-reward do not compose with --residual yet")
    if residual and (obs_noise > 0.0 or joint_noise > 0.0):
        # VecTTResidual accepts these (constructor passthrough, see its
        # docstring) but its obs-construction path never applies them.
        raise NotImplementedError("--obs-noise/--joint-noise do not compose with --residual yet")
    obs_dim = RESIDUAL_OBS_DIM if residual else env_obs_dim(intercept_features=intercept_features)
    if residual:
        vec = VecTTResidual(n_envs=n_envs, seed=seed, params=params or {}, residual_scale=residual_scale,
                            vel_penalty=vel_penalty, legal_bonus=legal_bonus,
                            obs_delay_steps=obs_delay_steps, obs_noise=obs_noise, joint_noise=joint_noise)
    else:
        vec = VecTT(n_envs=n_envs, seed=seed, params=params or {},
                    intercept_features=intercept_features, intercept_reward=intercept_reward,
                    vel_penalty=vel_penalty, legal_bonus=legal_bonus,
                    obs_delay_steps=obs_delay_steps, obs_noise=obs_noise, joint_noise=joint_noise)

    if log_std_init is None:
        log_std_init = -1.6 if residual else -0.5
    model = ActorCritic(obs_dim, ACT_DIM, log_std_init=log_std_init).to(device)
    if residual:
        last_linear = model.actor.net[-2]  # net = [Linear, Tanh, Linear, Tanh, Linear, Tanh]
        nn.init.zeros_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)
    obs_rms = RunningMeanStd((obs_dim,))
    if resume_from:
        # Continue a PPO run: actor, critic, log_std and the obs-normalizer stats all come from the checkpoint.
        ck = torch.load(resume_from, map_location=device, weights_only=False)
        if ck['obs_dim'] != obs_dim:
            raise ValueError(f"--resume-from obs_dim {ck['obs_dim']} != {obs_dim}")
        model.load_state_dict(ck['model_state'])
        obs_rms.mean = np.asarray(ck['obs_rms_mean'], dtype=np.float64)
        obs_rms.var = np.asarray(ck['obs_rms_var'], dtype=np.float64)
        obs_rms.count = ck['obs_rms_count']
    ckpt_extra = ({"obs_version": OBS_VERSION, "residual": True, "residual_scale": residual_scale} if residual
                  else {"intercept_features": intercept_features, "intercept_reward": intercept_reward})
    # Recorded regardless of mode so eval can report what the policy was
    # trained under, e.g. how robust to expect it to be sim-to-real.
    ckpt_extra.update({"obs_delay_steps": obs_delay_steps, "obs_noise": obs_noise, "joint_noise": joint_noise})
    if init_from:
        load_bc_actor_into(model, init_from, device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    csv_path = os.path.join(out, "train_log.csv")
    csv_fh = open(csv_path, "w", newline="")
    writer = csv.writer(csv_fh)
    writer.writerow(["step", "wall_s", "mean_return", "legal_return_rate", "contact_rate", "ep_len",
                     "det_legal_rate", "det_contact_rate"])
    vec_eval = None
    if det_eval_episodes > 0:
        # Fixed serves, nominal physics, no obs noise: the same conditions the evaluator scores under.
        eval_params = {**(params or {}), "physics_jitter": 0.0}
        if residual:
            vec_eval = VecTTResidual(n_envs=det_eval_episodes, seed=seed + 12345, params=eval_params,
                                     randomize=False, residual_scale=residual_scale, obs_delay_steps=obs_delay_steps,
                                     vel_penalty=vel_penalty, legal_bonus=legal_bonus)
        else:
            vec_eval = VecTT(n_envs=det_eval_episodes, seed=seed + 12345, params=eval_params, randomize=False,
                             intercept_features=intercept_features, intercept_reward=intercept_reward,
                             vel_penalty=vel_penalty, legal_bonus=legal_bonus, obs_delay_steps=obs_delay_steps)
    csv_fh.flush()

    episode_window: deque[dict] = deque(maxlen=EPISODE_WINDOW)
    start = time.time()
    total = 0
    n_updates = 0
    last_ckpt_time = start
    last_log_steps = 0
    best_legal_rate = -1.0

    try:
        while total < total_steps:
            if minutes is not None and (time.time() - start) >= minutes * 60.0:
                print(f"[ppo] --minutes {minutes} wall cap reached at step {total}")
                break

            obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf, last_value, episode_stats = \
                collect_rollout(vec, model, obs_rms, rollout_steps, device, obs_dim=obs_dim)
            episode_window.extend(episode_stats)
            adv_buf, ret_buf = compute_gae(rew_buf, val_buf, done_buf, last_value)
            ppo_update(model, optimizer, obs_buf, act_buf, logp_buf, adv_buf, ret_buf, obs_rms, device)

            total += rollout_steps * n_envs
            n_updates += 1
            if log_std_final is not None:
                # Upper bound on exploration noise that shrinks linearly over the run; PPO may go lower on its own.
                progress = total / float(total_steps)
                if minutes is not None:
                    progress = max(progress, (time.time() - start) / (minutes * 60.0))
                cap = log_std_init + (log_std_final - log_std_init) * min(progress, 1.0)
                with torch.no_grad():
                    model.log_std.clamp_(max=cap)

            if total - last_log_steps >= eval_every_steps or total >= total_steps:
                last_log_steps = total
                if episode_window:
                    mean_return = float(np.mean([e["return"] for e in episode_window]))
                    legal_rate = 100.0 * float(np.mean([e["legal_return"] for e in episode_window]))
                    contact_rate = 100.0 * float(np.mean([e["contact"] for e in episode_window]))
                    mean_len = float(np.mean([e["length"] for e in episode_window]))
                else:
                    mean_return = legal_rate = contact_rate = mean_len = float("nan")
                det_legal = det_contact = float("nan")
                if vec_eval is not None:
                    det_legal, det_contact = deterministic_eval(vec_eval, model, obs_rms, device)
                wall_s = time.time() - start
                writer.writerow([total, f"{wall_s:.2f}", mean_return, legal_rate, contact_rate, mean_len,
                                 det_legal, det_contact])
                csv_fh.flush()
                print(f"[ppo] step={total} update={n_updates} wall_s={wall_s:.1f} "
                      f"mean_return={mean_return:.3f} legal_return_rate={legal_rate:.1f}% "
                      f"contact_rate={contact_rate:.1f}% ep_len={mean_len:.1f} det_legal_rate={det_legal:.1f}%")

                selection = det_legal if vec_eval is not None else legal_rate
                if (vec_eval is not None or len(episode_window) >= 20) and selection > best_legal_rate:
                    best_legal_rate = selection
                    save_checkpoint(os.path.join(out, "best.pt"), model, obs_rms, total,
                                     obs_dim=obs_dim, act_dim=ACT_DIM, extra=ckpt_extra)

            if time.time() - last_ckpt_time >= CHECKPOINT_EVERY_S:
                last_ckpt_time = time.time()
                save_checkpoint(os.path.join(out, f"checkpoint_{total}.pt"), model, obs_rms, total,
                                 obs_dim=obs_dim, act_dim=ACT_DIM, extra=ckpt_extra)
    finally:
        save_checkpoint(os.path.join(out, "final.pt"), model, obs_rms, total,
                         obs_dim=obs_dim, act_dim=ACT_DIM, extra=ckpt_extra)
        csv_fh.close()

    return {"total_steps": total, "n_updates": n_updates, "out": out, "csv": csv_path,
            "best_legal_rate": best_legal_rate}


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-envs", type=int, default=64)
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--init-from", type=str, default=None)
    parser.add_argument("--minutes", type=float, default=None, help="hard wall-clock cap, minutes")
    parser.add_argument("--eval-every-steps", type=int, default=20_000)
    parser.add_argument("--rollout-steps", type=int, default=128, help="per-env steps collected per PPO update")
    parser.add_argument("--params", type=str, default="{}", help="JSON dict merged over tt_sim.scene.DEFAULT_PARAMS")
    parser.add_argument("--residual", action="store_true",
                         help="train a residual policy over tt_sim.policies.ik_hitter.IKHitterPolicy "
                              "(tt_sim.rl.residual.VecTTResidual) instead of learning from scratch")
    parser.add_argument("--residual-scale", type=float, default=DEFAULT_RESIDUAL_SCALE,
                         help="a = clip(a_exp + residual_scale * a_rl, -1, 1); only used with --residual")
    parser.add_argument("--resume-from", type=str, default=None, help="continue from a tt_sim.rl.ppo checkpoint (weights + obs normalizer)")
    parser.add_argument("--vel-penalty", type=float, default=0.0005, help="per-step weight on sum(joint_velocity^2)")
    parser.add_argument("--legal-bonus", type=float, default=5.0, help="one-off reward for a legal return")
    parser.add_argument("--intercept-features", action="store_true", help="append the 8-dim predicted-intercept block to the observation (non-residual only)")
    parser.add_argument("--intercept-reward", type=float, default=0.0, help="dense per-step penalty weight on |paddle - predicted intercept| before contact (non-residual only)")
    parser.add_argument("--det-eval-episodes", type=int, default=0,
                        help="if >0, every --eval-every-steps run the mean action on this many fixed serves and select best.pt by that")
    parser.add_argument("--log-std-init", type=float, default=None, help="initial exploration log-std (default -1.6 residual / -0.5)")
    parser.add_argument("--log-std-final", type=float, default=None, help="anneal the log-std upper bound linearly to this value over the run")
    parser.add_argument("--obs-delay-steps", type=int, default=0, help="FIFO-delay the policy's observation by this many control steps (sim-to-real sensor/actuation lag)")
    parser.add_argument("--obs-noise", type=float, default=0.0, help="std, metres, of Gaussian noise added to the observed ball position (velocity noise is 10x this, see env.py); 0 = off (non-residual only)")
    parser.add_argument("--joint-noise", type=float, default=0.0, help="like --obs-noise but for the robot's own joint position/velocity readings; 0 = off (non-residual only)")
    args = parser.parse_args(argv)
    # --out left at its argparse default (None) means the flag wasn't passed explicitly,
    # so a HIM job's HIM_CANDIDATE_OUTPUT wins; an explicit --out always overrides it.
    out = args.out or os.environ.get("HIM_CANDIDATE_OUTPUT") or os.path.join("work", "tt", "rl", "run")

    result = train(
        n_envs=args.n_envs, total_steps=args.total_steps, seed=args.seed, out=out,
        init_from=args.init_from, minutes=args.minutes, eval_every_steps=args.eval_every_steps,
        rollout_steps=args.rollout_steps, params=json.loads(args.params),
        residual=args.residual, residual_scale=args.residual_scale, intercept_features=args.intercept_features, intercept_reward=args.intercept_reward, vel_penalty=args.vel_penalty, legal_bonus=args.legal_bonus, resume_from=args.resume_from,
        obs_delay_steps=args.obs_delay_steps, obs_noise=args.obs_noise, joint_noise=args.joint_noise,
        det_eval_episodes=args.det_eval_episodes, log_std_init=args.log_std_init, log_std_final=args.log_std_final,
    )
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
