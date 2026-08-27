<!-- Source: https://arena.himrobotics.com/docs/quickstart/raw — fetched 2026-08-27 with the CLI token; HIM Arena private-preview docs, kept here for reference. -->
# Quickstart

> Download the data, launch a machine, and share what you build.

Audience: Builders
Updated: 2026-08-17
Page: /docs/quickstart

## 1. Get the data

HIM collected and released real table-tennis data. Open Datasets to download the Rokoko full-body motion, Orbbec RGB-D sessions, and WitMotion paddle IMU recordings.

- [Open Datasets](/datasets) — HIM's data first, then datasets uploaded by builders.

## 2. Launch an SSH machine

1. Open Compute and choose CPU or GPU.
2. Set the maximum lifetime and hard credit limit.
3. Paste one ssh-ed25519 public key. Never paste a private key.
4. When provisioning is available, launch the machine and copy the SSH command.
5. Clone your repository and work with your normal tools.

> **Robotics tools included:** The fast workspace includes Python, Git, build tools, MuJoCo, MJX, PyTorch, and JAX; GPU machines add CUDA. Isaac Sim and Isaac Lab remain a separate guided Compute path until their heavier image passes its own launch canary.

- [Open Compute](/compute) — Launch the flexible SSH workspace.

## 3. Build your way

Use your existing repository, simulator, trainer, coding agent, and experiment tracker. HIM does not require one canonical simulator or training framework.

> **No .him setup required:** You do not need a .him directory for the default builder loop. Add it later only if you want guided jobs, immutable packages, CI validation, exact artifact capture, or benchmark submission.

## 4. Share what you built

1. Choose Unitree G1, Booster K1, or Booster T1.
2. Choose Policy + simulator, Policy only, or Simulator only.
3. Drop the project ZIP and the real MP4 it produced, then add a name and short caption.
4. Publish the video page to your builder profile. The ZIP stays private and non-downloadable.
5. Play or download the MP4, share the page on X or LinkedIn, or copy its permanent link.
6. Open Builders to meet people and explore their work across sports.

- [Upload your work](/build#upload) — Share a policy, simulator, or both together.
- [Meet builders](/leaderboard) — See people, projects, and real scored results.

## When to use advanced workflows

Use Run a guided job and the .him package workflow when your team needs reproducible execution, controlled datasets, checkpoints, evaluator output, CI, immutable versions, or an official benchmark submission.

- [Advanced .him packages](/docs/him-package) — Package and validate an existing repository without rewriting it.
- [Guided Compute](/compute?mode=job) — Run controlled jobs with packages, data, and saved outputs.
