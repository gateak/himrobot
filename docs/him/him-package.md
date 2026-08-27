<!-- Source: https://arena.himrobotics.com/docs/him-package/raw — fetched 2026-08-27 with the CLI token; HIM Arena private-preview docs, kept here for reference. -->
# Bring an existing robotics repository

> Use a .him overlay without replacing the repository, simulator, trainer, or agent.

Audience: Builders and research teams
Updated: 2026-08-02
Page: /docs/him-package

## The overlay is the integration

A HIM Package is an immutable envelope around work you already have. It declares identity, entrypoints, runtime/process contracts, artifacts, tests, compatibility, security, provenance, licenses, and publication policy. It does not force a HIM training framework or source layout.

**Typical repository shape**

```text
your-robotics-repo/
├── existing source, configs, trainer, simulator, datasets...
└── .him/
    ├── him.yaml
    ├── contracts/
    ├── adapters/
    ├── samples/
    └── tests/
```

## Package workflow

**Connect, validate, and run**

```bash
# Run once from the root of an existing robotics repository.
him init .
# Simulator repositories: him init . --simulator

# Point the generated entrypoint at the command the project already uses.
# Review the .him files, then commit the intended source.
him doctor . --json
him validate --json
him test --local --json

# Review the source upload and credit limit when prompted.
him run --max-credits 100 --json
him job watch <job-id> --json

# Upload an unlisted package without launching a job:
him push --yes --json
```

- Repeated builds produce identical lock and archive bytes.
- The CLI shows the exact package and committed-source upload before a run.
- Credential-like filenames, escaping symlinks, missing artifacts, invalid contracts, and publication-right gaps fail closed.
- Source is a separate private, digest-bound Git snapshot. Weights, datasets, endpoint addresses, and calibration stay in declared artifacts behind object-level authorization.
- Most projects upload all committed source outside .him. Large multi-project repositories can set extensions.source_paths to an exact list of committed files and directories so unrelated workspaces are not uploaded.
- Hosted commands receive HIM_CANDIDATE_OUTPUT, HIM_EVALUATION_OUTPUT, and HIM_MEDIA_OUTPUT. Existing artifacts/result.json outputs are preserved automatically for compatibility.

**Optional: narrow a multi-project repository**

```yaml
extensions:
  source_paths:
    - robotics/assets
    - robotics/train
    - motions/dance.csv
```

> **Keep .him out of source_paths:** List only committed files or directories outside .him. HIM packages the .him overlay automatically, so adding .him or one of its subdirectories fails before upload. Paths are relative to the repository root.

> **If the repository does not have .him yet:** That is normal. Run him init once; do not hand-write the entire manifest. Keep the repository’s native command and source layout, then let him doctor identify the remaining facts before a cloud run.

> **Runtime names are portable identifiers:** Use isaac_lab, mujoco_mjx, pytorch, jax, or brax. Names such as “Isaac Lab 2.2”, image tags, shell commands, and provider-specific GPU labels are not runtime names. Normally him run selects the runtime from your package, so you do not need to type one.

## Entrypoints preserve native work

*Common package entrypoints*

| Purpose | Examples | HIM expectation |
| --- | --- | --- |
| Inference | Python module, OCI command, policy service, private endpoint | Typed observations/actions or declared batch inputs/outputs |
| Evaluation | MuJoCo episode driver, Isaac play task, regression harness | Frozen candidate output and isolated evaluator contract |
| Training | torchrun, JAX/Orbax, native lab pipeline, multi-stage DAG | Checkpoint/resume, budgets, promoted candidate outputs |
| Render | Simulator camera or trajectory renderer | Factual media derived from recorded state |
| Deployment | Booster/Unitree adapter bundle | Separate hardware readiness and safety gate |

## Use the same contract with coding agents

Use Copy for AI at the top of any guide to generate a complete setup prompt for Codex, Claude Code, Cursor, or another coding agent. The prompt tells the agent to inspect first, preserve native commands, edit only .him by default, validate locally, and ask before it uploads source or spends credits.

- [AI handoff](/docs/ai-handoff) — Copy a bounded prompt and machine-readable docs context.
- [CLI and API](/docs/cli-api) — Use the same package and job contracts programmatically.
