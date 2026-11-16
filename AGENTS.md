# Nemotron Voice Agent Repository Guidance

## Product Scope

Nemotron Voice Agent is an end-to-end voice-agent blueprint built on Pipecat,
with a LiveKit example.
It combines NVIDIA NIM services for automatic speech recognition (ASR), large
language model (LLM) inference, and text-to-speech (TTS) synthesis. Preserve
the repository's supported cloud, workstation, DGX Spark, and Jetson Thor
deployment profiles when changing shared behavior.

## Sources of Truth

- Use the root `pyproject.toml` and `uv.lock` for Python versions and dependencies. Each framework has its own
  dependency group (`pipecat`, `livekit`), and the `nva-pipecat` and `nva-livekit` entry points start them.
- Use `clients/pipecat/package.json` and `clients/pipecat/package-lock.json` for client
  dependencies and scripts.
- Use `examples_registry.yaml` for registered examples, transports, the
  services each example offers per slot, and per-example service overrides
  (`settings` and `categories`).
- Use `services.yaml` for built-in LLM, ASR, and TTS entries (`server` and
  `single-gpu` sections, `nvcf` for NVIDIA Cloud) and `settings.yaml` for
  editable service settings.
- The Python package is `src/nemotron_voice_agent/`. Each framework folder (`pipecat/`, `livekit/`) has
  `runtime/` for the server, gateways, and deployment helpers, and `examples/`. A framework adds `shared/` for
  code that several of its examples use, as `pipecat/` does. `core/` holds code that both frameworks use,
  such as the paths and loaders of `services.yaml` and `examples_registry.yaml`. `sidecars/` holds code that
  runs inside the model containers, such as the vLLM Realtime server. A runtime never imports an example, one framework never imports another,
  and `core/` and `sidecars/` import no framework. `tests/unit/core/test_layout_rules.py` checks these rules.
- Use each `src/nemotron_voice_agent/pipecat/examples/<example>/pipeline.py` and `prompts.yaml` for example
  behavior and configuration. Shared runtime behavior lives in `src/nemotron_voice_agent/pipecat/shared/` and
  `src/nemotron_voice_agent/pipecat/runtime/`, which holds `server.py`.
- Use `src/nemotron_voice_agent/livekit/examples/generic_assistant/` for the LiveKit example.
- For the Frontend/Backend Live example, also use the `config` block of its
  `examples_registry.yaml` entry, its `factory.py`, its `frontend.py`, which
  builds the pipeline for the UI bot and live sessions, the Live gateway in
  `src/nemotron_voice_agent/pipecat/runtime/gateways/live/`, and the delegation core in
  `src/nemotron_voice_agent/pipecat/examples/frontend_backend_live/delegation_core/`.
- Use `docker-compose.yml` for Compose profiles. It includes
  `docker/pipecat/docker-compose.yaml` and `docker/livekit/docker-compose.yaml`, and their model
  services extend the files in `docker/sidecars/`. These files define service names, container
  images, ports, and hardware-specific deployment behavior.
- Use `README.md`, the example READMEs, and `docs/` for user-facing behavior.
  When prose and implementation disagree, verify the implementation and update
  the affected documentation in the same change.

## Repository Workflows

- Run repository commands from the repository root.
- Preserve an existing `.env`. Never commit credentials, generated runtime
  state, benchmark results, caches, or local model data.
- Select exactly one recipe profile for a Docker Compose deployment. Cloud
  profiles use `<example>`; local profiles use `<example>/<hardware>`.
  Observability profiles such as `tracing` and `turn` are overlays.
- Load `.agents/skills/nemotron-voice-agent-deploy/SKILL.md` for deployment, startup troubleshooting, or
  changes to `.env`, `examples_registry.yaml`, prompts, service catalogs, settings, transports, tracing, or
  audio settings.
- Load `.agents/skills/nemotron-voice-agent-upgrade-pipecat/SKILL.md` before changing Pipecat server or
  client dependency versions or migrating Pipecat APIs.
- Preserve unrelated configuration keys, comments, examples, and deployment
  profiles. Do not infer hardware, credentials, or private-service access.

## Validation

Run checks that match the changed surfaces. The GitHub Actions workflow uses
these commands:

```bash
uvx ruff@0.15.6 check .
uvx ruff@0.15.6 format --check .
uv sync --locked
uv run pytest tests/unit/core tests/unit/sidecars tests/unit/pipecat tests/test_docs_review_receipt.py -v
uv run pytest tests/unit/livekit -v
uv run --group eval python tests/evals/pipecat/ci/smoke.py
npm --prefix clients/pipecat ci
npm --prefix clients/pipecat run lint
npm --prefix clients/pipecat run build
```

For documentation-only changes, run the configured pre-commit hooks on the
changed files:

```bash
uv run pre-commit run --files <changed-files>
```

Inspect relative links and referenced paths. This repository does not configure
a dedicated documentation renderer or Markdown link checker. Report any
validation that requires unavailable GPUs, services, credentials, or deployment
hardware instead of claiming it passed.

## Documentation

Before completing a change, determine whether it affects a user-visible
surface. These surfaces include public APIs, configuration, Compose profiles,
service catalogs, prompts, the browser client, workflows, defaults, errors,
deployment behavior, and other product behavior.

When a change affects users and the host supports subagents, start a
documentation subagent in parallel while the primary implementation continues.
Direct the subagent to read `docs/AGENTS.md`, provide the changed source files
and identified user impact, and require it to update the affected documentation
and run the documented validation. Reconcile the documentation changes and
validation evidence before completing the change.

If the host cannot run subagents, the primary task must read `docs/AGENTS.md`,
complete the documentation work, and run the same validation. Do not omit
required documentation because parallel execution is unavailable.

### Documentation Writer Review Receipt

Every pull request that changes code or documentation must include one
`## Documentation Writer Review` section from
`.github/PULL_REQUEST_TEMPLATE.md`. Complete the review after the changes and
applicable validation are finished.

- Check the review-completion box and keep exactly one result:
  `docs-updated`, `no-docs-needed`, or `blocked`.
- Name the changed documentation in **Evidence**, or explain why documentation
  is not needed or why the review is blocked.
- Record the agent product and surface that performed the review.
- After committing the reviewed changes, fill the hidden head and guidance
  fields with `git rev-parse --short HEAD` and
  `git rev-parse --short HEAD:AGENTS.md`.
- Any later commit makes the receipt stale. Rerun the documentation review and
  refresh both hidden fields.

The `CI / Documentation Writer Review` workflow checks the receipt in advisory
mode. Use the following command to measure adoption. The report also supports
`json` and `csv` formats.

```bash
python scripts/docs-review-receipt.py report --since <YYYY-MM-DD> --format summary
```
