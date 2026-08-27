<!-- Source: https://arena.himrobotics.com/docs/cli-api/raw — fetched 2026-08-27 with the CLI token; HIM Arena private-preview docs, kept here for reference. -->
# CLI and Compute API

> Launch SSH machines and manage private guided jobs programmatically.

Audience: Developers and automation authors
Updated: 2026-08-24
Page: /docs/cli-api

## Connect your tools

1. Install the HIM CLI once on your workstation.
2. Sign in to HIM with GitHub or Discord.
3. Open Account and create a developer token under API Tokens.
4. Run him login. Paste the token into the hidden prompt; it will not enter shell history.
5. Run him whoami to verify the account and access before starting a job.

**Connect the CLI**

```bash
curl -fsSL https://arena.himrobotics.com/install.sh | sh
him login --api-url https://<arena-origin>
him whoami --json

# CI: store both values in your CI provider's secret settings
export HIM_API_URL=https://<arena-origin>
export HIM_TOKEN=him_pat_<secret>
```

> **The token prompt stays hidden:** Paste the token and press Enter even though the terminal displays nothing. This is intentional. For CI, skip the prompt and store HIM_API_URL and HIM_TOKEN as encrypted environment secrets.

> **Keep the token out of your repository:** HIM shows the token once. Never put it in source, .him, a package manifest, a command argument, or a log. Create a separate expiring token for each machine or CI environment and revoke it from Account when it is no longer needed.

## Launch an SSH machine

The CLI uses the same owner-scoped machine service as Compute. It reads one public-key file, chooses an allowlisted profile, calculates the minimum safe credit ceiling from live capabilities, waits for a verified SSH endpoint, and prints the connection and stop commands.

**SSH machine lifecycle**

```bash
him machine launch --key ~/.ssh/id_ed25519.pub --profile gpu --hours 2
him machine list --json
him machine status <machine-id> --json
him machine watch <machine-id> --json
him machine stop <machine-id>
```

> **Stop it when you finish:** The hard lifetime is a backstop. Copy out the work you need, then use him machine stop so HIM terminates the provider machine and settles the credit reservation immediately.

## Upload a policy or simulator

Create a metadata-only Builder listing from the CLI. The browser flow additionally accepts and publishes a result MP4. In both cases the ZIP stays private and non-downloadable.

**Headless Builder upload**

```bash
him upload ./project.zip \
+  --kind policy \
+  --run-command "python evaluate.py" \
+  --yes --json
```

## CLI job commands

**Common commands**

```bash
# From a repository with .him/him.yaml and a clean commit:
him run --max-credits 100 --yes --json
him job watch <job-id> --follow --json

# Upload a versioned private dataset:
him dataset push dataset-manifest.json --file URI=PATH --yes --json

# Create a Community challenge from that package:
him catalog create-challenge sha256:<package-lock-digest> \
  --name "Balance Recovery" \
  --description "Recover from a repeatable push." \
  --simulator-entrypoint evaluate \
  --score-metric success_rate \
  --yes --json

# Enter an exact challenge version with your policy package:
him arena submit <challenge-version-id> \
  --package sha256:<policy-package-digest> \
  --entrypoint evaluate --json
him arena leaderboard <challenge-version-id> --json

# Lower-level controls:
him job launch inference --runtime pytorch --json
him job launch simulation --runtime mujoco_mjx --json
him job launch simulation --runtime isaac_lab --resource gpu-memory --json
him job launch training --runtime jax --max-wall-seconds 86400 --json
him job list --json
him job cancel <job-id> --json
him job resume <training-job-id> --json
him job retry <failed-job-id> --json
```

**Approved package entrypoint**

```bash
him job launch simulation \
  --runtime mujoco_mjx \
  --package-digest sha256:<package-lock-digest> \
  --entrypoint evaluate \
  --idempotency-key team-eval-2026-07-21-001 \
  --json
```

## Versioned API

*Standalone Compute endpoints*

| Method | Path | Purpose |
| --- | --- | --- |
| GET | /api/health | Public deployment and provider readiness |
| GET | /api/v1/compute/capabilities | Provider, workloads, and whether code execution is enabled |
| GET | /api/v1/compute/ssh-machines/capabilities | Allowlisted machine profiles, environments, prices, and lifetime limits |
| GET / POST | /api/v1/compute/ssh-machines | List owner machines or launch one bounded SSH machine |
| GET | /api/v1/compute/ssh-machines/:workspaceId | Refresh owner-only status and verified connection details |
| POST | /api/v1/compute/ssh-machines/:workspaceId/stop | Idempotently terminate the machine and settle credits |
| GET | /api/v1/compute/jobs | List the current owner’s private jobs |
| POST | /api/v1/compute/jobs | Launch an immutable private job |
| PUT | /api/v1/packages/:packageDigest/archive | Store the exact private .him package archive |
| PUT | /api/v1/packages/:packageDigest/source | Store a clean private Git source snapshot by digest |
| GET | /api/v1/compute/jobs/:jobId | Read lifecycle, attempt, result, and evidence |
| POST | /api/v1/compute/jobs/:jobId/cancel | Cancel inside the allowed window |
| POST | /api/v1/compute/jobs/:jobId/retry | Retry a retryable failure as a new attempt |
| POST | /api/v1/compute/jobs/:jobId/resume | Resume cancelled training from a published checkpoint |
| GET | /api/v1/compute/jobs/:jobId/artifacts?uri=... | Owner-only verified artifact delivery |
| GET | /api/v1/compute/jobs/:jobId/failure-log | Owner-only provider log or sanitized failure timeline |
| POST | /api/v1/datasets | Register a private immutable dataset version |
| POST | /api/v1/datasets/:datasetId/artifacts | Issue one exact direct-upload grant |
| POST | /api/v1/datasets/:datasetId/finalize | Verify and finalize the Dataset Manifest |
| POST | /api/v1/packages/:packageDigest/promote-to-arena | Idempotently add package-declared immutable catalog versions |
| POST | /api/v1/packages/:packageDigest/community-challenges | Create a Community challenge from package entrypoints |
| POST | /api/v1/community-challenges/:challengeVersionId/submissions | Enter an approved policy package in an exact Community challenge |
| GET | /api/v1/community-challenges/:challengeVersionId/leaderboard | Read that version’s provider-verified Community rankings |

**Launch a simulation from a signed-in client**

```typescript
const response = await fetch(`${process.env.HIM_API_URL}/api/v1/compute/jobs`, {
  method: "POST",
  headers: {
    Authorization: `Bearer ${process.env.HIM_TOKEN}`,
    "Content-Type": "application/json",
    "Idempotency-Key": crypto.randomUUID(),
  },
  body: JSON.stringify({
    workload: "simulation",
    templateId: "mujoco-simulation-contract",
    runtimeFamily: "mujoco_mjx",
    source: {
      kind: "him_package",
      packageDigest: "sha256:<package-lock-digest>",
      entrypointId: "evaluate",
    },
    resourceClass: "gpu-standard",
    maxWallSeconds: 1800,
    maxCredits: 100,
  }),
});

if (!response.ok) throw new Error(`Job launch failed: ${response.status}`);
const job = await response.json();
```

> **Authentication:** Browser requests use your signed-in session. CLI, CI, and external API clients use an expiring developer token. Both resolve to the same builder account, so packages, jobs, credits, artifacts, and checkpoints keep the correct owner.

## API rules

- Every write requires a stable Idempotency-Key. Repeating the same action is safe; conflicting reuse fails.
- Send developer tokens only in the Authorization Bearer header. HIM rejects invalid, expired, or revoked tokens and does not fall back to another identity.
- Use packages:write to push repositories, datasets:write to register captures, and compute:write to launch or manage jobs. Read-only tokens cannot mutate those resources.
- Compute jobs are private and owner scoped. Super-admin management permission does not silently make results public.
- Clients select a portable resource class. HIM maps it to an available runtime profile and enforces the job’s limits.
- 401 means sign-in is required; 403 means the signed-in account lacks permission; 409 means the requested lifecycle action is not currently allowed.
- A result is complete only after HIM has recorded its final lifecycle state, artifacts, and evidence.
