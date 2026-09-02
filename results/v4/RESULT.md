# v4 residual PPO — result (2026-08-28, box #3 `ssh_a971a4ed67df61e3931a`)

Recipe: v3 flags (`--residual --residual-scale 0.2 --vel-penalty 1e-5 --legal-bonus 10 --n-envs 64`), 425-min caps,
six seeds: s40/s41 fresh, s42–s45 `--resume-from` the v3 s32 checkpoint (53.5 % deterministic). ~11.5 M steps each.
Box pushed to HIM as `himrobot-rl-checkpoints@0.1.29798675` (`dataset_19db501c770edb788385`) and stopped itself at
12:36 UTC. Local copy: `work/tt/rl/box/v4/resid_s4x/{best,final}.pt`. The box's own end-of-run evaluation crashed
(finish script called `python` instead of `/workspace/venv/bin/python`), so all scores below were produced on the Mac.

## Deterministic evaluation, 200 serves, seed 11 (hitter 46.5 % / 3.5 % target; v3 s32 53.5 % / 6.5 %)

| seed | start | best.pt legal / target | final.pt legal / target | train-time best legal (noisy) |
|---|---|---|---|---|
| s40 | fresh | 49.0 % / 5.5 % | **51.5 % / 4.5 %** | 27.5 % |
| s41 | fresh | 36.0 % / 3.5 % | – | 27 % |
| s42 | resumed | 48.5 % / 2.5 % | 32.0 % / 1.0 % | 27.5 % |
| s43 | resumed | 45.5 % / 3.5 % | – | 31 % |
| s44 | resumed | 35.0 % / 5.5 % | – | 31 % |
| s45 | resumed | 46.5 % / 6.5 % | 39.5 % / 8.5 % | 30 % |

## Read

- **Longer training did not help; resuming hurt.** The four seeds that started at 53.5 % deterministic ended at
  32–48.5 % while their *noisy* train-time legal rate rose 22 → 31 %. PPO optimizes returns under its own exploration
  noise (log_std −1.6, never annealed); the mean-action policy is shaped to hedge against that noise, and precision
  contact is exactly what hedging destroys. The fresh seed s40 is the only one still improving deterministically at the
  end (51.5 %).
- **Checkpoint selection is broken.** `best.pt` = max train-time legal rate over a 20-episode window
  (`tt_sim/rl/ppo.py:393`): a lucky streak, not a measurement. s40's best.pt (49.0 %) is worse than its final.pt (51.5 %).
- 200 serves ⇒ ±3.5 pt standard error per number; differences under ~7 pts between single runs are not conclusive.

## Fixes before the next box (no credits needed)

1. Deterministic periodic evaluation inside the trainer: every `eval_every_steps`, run the mean action on a fixed set of
   100 serves (fixed seed, `randomize=False` VecTT) and select `best.pt` by that; log it as `det_legal_rate`.
2. Expose `--log-std-init` (try −2.3) and anneal log_std toward a floor over the run so late training refines.
3. The per-serve parameter residual (workflow `tt-generalization-v1`) sidesteps the noise problem structurally: K
   parameters chosen once per serve, no per-step jitter.
