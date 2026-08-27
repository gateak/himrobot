<!-- Source: https://arena.himrobotics.com/docs/ai-handoff/raw — fetched 2026-08-27 with the CLI token (private preview). -->
# Set up HIM with a coding agent

> Give Codex, Claude Code, Cursor, or another agent the exact setup task.

Audience: Developers using coding agents
Updated: 2026-08-02
Page: /docs/ai-handoff

## Copy the setup prompt

Choose Copy for AI at the top of this guide—or any HIM guide—then paste the prompt into your coding agent from the repository you want to connect. The generated prompt includes absolute links back to the current Arena docs.

- [llms.txt](/llms.txt) — Compact product, contract, and guide index.
- [Full docs as Markdown](/docs/llms-full.txt) — All in-product docs from the same typed source.
- [Public schemas](/docs/him-package) — Start with the package and contract boundary.

The agent should inspect the repository, identify its real headless command, create the smallest .him overlay, and run local checks. It must stop for your approval before uploading committed source or consuming compute credits.

## Repository integration prompt

**Use with Claude Code, Codex, Cursor, or another coding agent**

```text
Read the HIM Package guide linked in the copied prompt.

Goal: connect this existing robotics repository to HIM without replacing its native simulator, trainer, commands, or source layout.

Rules:
- Inspect the repository first. Do not read .env files, credentials, tokens, or large model/data files.
- Use him init only when .him is missing. Edit only .him by default.
- Show the proposed diff before writing.
- Run him doctor . --json, him validate --json, and him test --local --json.
- Stop and explain any failed check. Do not weaken tests or invent missing facts.
- Ask before running him run because it uploads committed source privately and may spend credits.
- Return the exact commands, results, job ID, and job URL.
```

## Simulator intake prompt

**Use before a MuJoCo or Isaac integration**

```text
Read the MuJoCo or Isaac guide linked in the copied prompt. Inspect this simulator project without reading secrets. Identify the smallest native headless command, robot assets, task launcher, evaluator, output and media paths, signal mapping gaps, licenses, and runtime requirements. Propose a private .him overlay, show every change, and use the smallest local smoke test before requesting approval for a cloud run.
```

## A good agent handoff returns

- Files inspected and the exact immutable source/package identity used.
- Facts learned versus assumptions/unknowns that still require a human.
- A minimal .him-only diff with contracts, adapters, samples, tests, and private publication policy.
- Commands proposed, which were actually executed, and their exact results.
- Security, license, privacy, simulator, timing, and hardware limitations.
- The next smallest safe action—not a broad rewrite or hidden provider deployment.
