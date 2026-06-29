---
name: upgrade-pipecat
description: Upgrade the Nemotron Voice Agent to a new Pipecat (pipecat-ai) version. Reads release notes for every release in range, diffs old vs new, discovers every example pipeline and Pipecat call site, implements changes, then runs multi-agent gap analysis until clean. Generic across Pipecat versions.
version: "2.0.0"
metadata:
  author: NVIDIA Voice Agent Team <nemotron-voice-agent@nvidia.com>
  tags: [upgrade, migration, pipecat, voice-agent, nemotron]
---

# Pipecat Version Upgrade — Nemotron Voice Agent

Autonomously migrate this repo to a new `pipecat-ai` version. Repo root is the working directory.

## Invocation

```text
/upgrade-pipecat new=<target> [old=<source>]
```

- **`new`** (required): target `pipecat-ai` version. PyPI version (`1.5.0`), local pipecat checkout path
  (full diff), git tag (`v1.5.0`), or docs URL (`https://docs.pipecat.ai/`, least complete).
- **`old`** (optional): current version. A version string enables CHANGELOG/git diff. Omit to auto-detect from
  `pyproject.toml` (`pipecat-ai[...]==X.Y.Z`) + `uv.lock`.

The only input is the `pipecat-ai` version. Latest version and release notes come from the canonical
**<https://github.com/pipecat-ai/pipecat/releases>**. The skill scans BOTH dependency surfaces for every Pipecat
package and reads each one's own release notes:

- **Server (Python)** — `pyproject.toml`/`uv.lock`: every `pipecat*` dependency (e.g. `pipecat-ai-subagents`,
  `pipecat-ai-flows`, …).
- **Client (npm)** — `client/package.json`: every `@pipecat-ai/*` dependency (e.g. `client-js`, `client-react`,
  `*-transport`, …).

Extras and every dependency change are derived from these notes + the lockfiles — nothing about specific
packages is hardcoded here.

## Phases (run in order)

1. **Explore & Discover** — [workflows/01-explore-discover.md](workflows/01-explore-discover.md). Read release
   notes for every release in range, analyze against the repo, confirm with source diff + per-example scan.
2. **Plan & Implement** — [workflows/02-plan-implement.md](workflows/02-plan-implement.md).
3. **Gap Analysis Loop** — [workflows/03-gap-analysis-loop.md](workflows/03-gap-analysis-loop.md).
4. **Deploy & Validate** — [workflows/04-deploy-validate.md](workflows/04-deploy-validate.md).

## Pipecat docs MCP — use for any doubt

For ANY Pipecat uncertainty (signature, moved module, intended usage, migration path), query the
`pipecat-docs` MCP tool `search_daily_knowledge_sources` (backed by <https://daily-docs.mcp.kapa.ai>) instead of
guessing. Pass one complete sentence as `query`; cite the returned `source_url` in the change log. For exact
signatures, trust the actual installed/new source; use the MCP for intent and migration guidance.

## Principles

- **Release-notes-driven (hard gate)**: read the `pipecat-ai` release notes + CHANGELOG for EVERY release in
  range — and each `pipecat*` subpackage's own notes — before doing anything else. Do not start the diff, the
  plan, or any edit until this is done and recorded. Most missed migrations come from skipping this. The source
  diff only confirms and completes the notes.
- **Dependencies follow the notes, not assumptions**: discover the repo's current Pipecat-related dependencies
  from `pyproject.toml`/`uv.lock`, then let the `pipecat-ai` notes dictate what happens to each — bumped,
  renamed, newly required, or folded into core (dependency removed + imports migrated). Never hardcode or assume
  a companion package stays separate, stays present, or co-versions.
- **Discovery-first**: never assume module paths, frame names, service constructors, or processor APIs — scan
  the installed package and new source. Pipecat reorganizes its module tree between versions.
- **Generic**: works for any transition; discover changes, hardcode nothing.
- **Examples are the unit of work**: 5 examples (`generic`, `multilingual`, `omni_assistant`,
  `omni_assistant_subagents`, `frontend_backend_agent`), each with its own `pipeline.py`. One agent per example + one
  cross-cutting agent for `src/examples/shared/` and `src/server.py`.
- **Server + client move together (RTVI contract)**: the RTVI wire protocol couples the Python server to the
  `@pipecat-ai/*` client packages, so they must be upgraded in lockstep. Bump `client/package.json` to versions
  compatible with the target `pipecat-ai`, migrate `client/src/` RTVI usage (renamed events/messages), and gate
  on `npm` lint+build. Discover the client packages — don't assume which exist.
- **Iterative convergence**: gap analysis loops until a pass finds zero gaps.
- **Validation gates**: `uv sync`, import smoke tests, `ruff`, `pytest`, Compose deploy. Human input only for
  plan confirmation and review.
