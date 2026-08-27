<!-- Source: https://arena.himrobotics.com/docs/compute/raw — fetched 2026-08-27 with the CLI token (private preview). -->
# HIM Compute

> Launch a flexible SSH machine. Use guided jobs when you need tighter controls.

Audience: Builders
Updated: 2026-08-17
Page: /docs/compute

## Launch an SSH machine

1. Open Compute and choose CPU, a best-available 24 GB GPU, or a best-available 48 GB GPU. HIM automatically picks live Secure Cloud capacity within the displayed price tier.
2. Set a maximum lifetime and hard credit limit.
3. Paste one ssh-ed25519 public key. Never paste a private key.
4. Launch the machine and copy the SSH command when provisioning is available.
5. Clone your repository and use your normal shell, editor, simulator, and training commands.

> **The flexible default:** SSH gives you a full shell and the least process. The fast workspace includes Python, Git, build tools, MuJoCo, MJX, PyTorch, and JAX; GPU profiles add CUDA. Use guided Compute for the separately qualified Isaac path.

> **Runpod, billed in HIM credits:** Hosted SSH uses bounded Runpod Secure Cloud machines. HIM reserves the selected ceiling, settles actual elapsed time, stops idle machines, and never shows a connection command until the generated host key is verified.

## SSH or guided?

*Start with SSH unless you need a controlled run record.*

| Mode | Best for | What HIM controls |
| --- | --- | --- |
| SSH machine — default | Exploration, training, simulation, debugging, and agent-driven work | Machine allowlist, owner access, budget, and automatic lifetime |
| Guided job — advanced | Repeatable evaluation, CI, checkpoints, scoring, and exact artifacts | Package, inputs, runtime, outputs, evaluator, and execution record |

> **Guided jobs use .him:** The sections below describe the advanced guided path. You do not need the CLI, a .him directory, or a HIM Package to launch an SSH machine.

## Choose the job, not the infrastructure

*Workload mapping*

| Job | Typical entrypoint | Expected evidence | Common runtime |
| --- | --- | --- | --- |
| Inference | infer or inference | Predictions, logs, latency/usage | PyTorch, JAX, OpenPI, reviewed custom runtimes |
| Simulation + evaluation | evaluate, simulation, or sim | Trajectory, isolated evaluator metrics, videos, environment | MuJoCo/MJX, Brax, Isaac Lab |
| Training | train or training | Checkpoints, logs, evaluation outputs, resume identity | PyTorch/JAX with simulator rollouts |

> **MuJoCo is not the trainer:** A training job commonly uses PyTorch or JAX for optimization and MuJoCo/MJX, Brax, or Isaac Lab to generate rollouts. HIM records the complete declared runtime and entrypoint without making the core model simulator-shaped.

## What a completed job returns

- Immutable job spec digest, attempt number, provider identity, lifecycle events, and timestamps.
- Declared private outputs, logs, media, and checkpoints with owner-only delivery where hosted.
- Execution disclosure: provider reported, provider verified, or reproduced by HIM in simulation.
- For simulation, authoritative metrics only from the versioned evaluator attestation. Candidate logs, previews, and LLM text never become official scores.
- A private package may return its own metrics and artifacts for debugging. HIM labels those as package-reported evidence; they become an Arena score only through the challenge's versioned evaluator.
- For training, checkpoint identity and resume lineage across attempts.
- For failures, an owner-only technical record with the failed stage, code, retry guidance, and private provider log when one exists.

## Use HIM's table-tennis data

HIM's released table-tennis data includes Rokoko body motion, Orbbec RGB-D sessions, and WitMotion paddle IMU recordings. Datasets shows HIM's files first, followed by datasets shared by builders.

- [Open Datasets](/datasets) — See HIM's capture and builder datasets.

> **Advanced: register large or versioned data:** Use the dataset CLI when you need immutable versions, large multi-file uploads, private inputs, or an exact guided-job mount. This is not required to download HIM's data.

**Download from a terminal**

```bash
him data download table-tennis --output ./data --yes
```

**Advanced dataset workflow**

```bash
him dataset validate dataset-manifest.json
him dataset push dataset-manifest.json \
  --file r2://capture/session.mcap=./captures/session.mcap \
  --file r2://capture/episodes.parquet=./derived/episodes.parquet \
  --yes --json

# Mount the registered version into the repository's normal entrypoint:
him run --dataset <dataset-id> --yes --json
```

> **Private first:** Builders create private dataset versions. A HIM administrator can review and publish a ready version, then attach it to an exact Arena challenge. Large files stream directly to object storage; HIM does not buffer them through the web app.

## Choose the right path

*Start with the workflow that matches your goal.*

| Goal | Start with | You receive |
| --- | --- | --- |
| Check a policy | Inference | Predictions, logs, latency, and declared outputs |
| Measure behavior in simulation | Simulation + evaluation | Trajectory, evaluator metrics, artifacts, and media |
| Improve a model | Training | Checkpoints, logs, evaluation outputs, and resume history |
| Compete in a shared test | An Arena challenge | A verified score, run record, and leaderboard entry |

> **Works with your stack:** HIM does not replace your repository, simulator, trainer, model architecture, dataset, cluster, or experiment tracker. It adds a consistent way to package work, run it, and keep the result verifiable.

## Fix the common setup errors

*Start here before changing your project.*

| What you see | What it means | What to do |
| --- | --- | --- |
| Nothing appears while pasting a developer token | The login prompt hides every character by design | Paste the token, press Enter, then run him whoami --json |
| Cloudflare Error 1010 during him login | An outdated CLI sent a generic Python HTTP signature that Cloudflare rejected before HIM received the token | Update the HIM CLI and retry login; the token was not stored or consumed |
| No .him directory | The repository has not been connected to HIM yet | Run him init once with the template that matches your project, then edit only the generated entrypoint |
| Runtime family must be a short portable identifier | A display name, image tag, or command was used as the runtime | Use isaac_lab, mujoco_mjx, pytorch, jax, or brax; keep versions and commands in their own fields |
| A hosted job does not start | The package or account needs a specific correction | Open the job and read the first failed lifecycle step, then run him doctor . --json before retrying |

**Fast diagnostic**

```bash
him whoami --json
him doctor . --json
him validate --json
him test --local --json
```

> **Keep the native repository workflow:** Do not rewrite a working Isaac, MuJoCo, training, or inference command to fit HIM. Put that command behind the generated .him entrypoint and let HIM handle packaging, launch, and outputs.
