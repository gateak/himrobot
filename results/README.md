# Results ledger

This directory is the **versioned** record of scored training runs. Before this restructure
(2026-09-01) every run's write-up lived only under `work/tt/rl/<run>/RESULT.md` in the main
checkout — and `work/` is git-ignored, so the whole history of what was tried and what it scored
existed on one laptop and died with it. This ledger fixes that: one `RESULT.md` per run, copied
here verbatim the moment a run is scored.

## What lives here, what stays in `work/`

- **`results/<run>/RESULT.md`** — the write-up: what changed, the exact command, and the
  deterministic scoreboard. Copied byte-for-byte from `work/tt/rl/<run>/RESULT.md` (the
  project's oldest run, which predates the per-run subdirectory convention, lives at
  `results/early-runs/RESULT.md`, copied from the top-level `work/tt/rl/RESULT.md`). Copy, never
  move — `work/tt/rl/<run>/` stays the working copy a session writes to; this is the archive of
  what it said when the run was scored.
- **Checkpoints, videos, training logs, and raw per-seed `evaluator.json`/`trajectory.jsonl`
  output** stay in `work/tt/rl/<run>/` (git-ignored, local, disposable — regenerate them by
  re-running the command in the `RESULT.md`).
- **Shipped weights** — the `.npz` files a policy actually runs on (HIM's worker, or `he
  evaluate`/`evaluate_stance` entrypoints) — live in `assets/policies/`, which *is* checked in.
- **`results/SCOREBOARD.md`** — one table, the project's policy lineage: the numbers that made
  it into `docs/PROJECT_KNOWLEDGE.md` §2, with a source citation per row.

## The one rule that matters

Per `CLAUDE.md`'s engineering conventions: **numbers in this ledger come only from the
deterministic evaluator (`tt_sim.evaluate` / `himrobot.eval.evaluate`, or the seed-scoreboard
wrapper `tt_sim.rl.eval_seeds` / `himrobot.eval.eval_seeds`) run with the mean action**, on the
fixed serve seeds (11/2000/4000 today; older entries note where they used the earlier,
overlapping 11/7/3 set). A training run's own noisy episode-reward log is a diagnostic, never a
result — it never gets copied here as a score.
