# Nemotron Voice Agent Repository Guidance

## Product Scope

Nemotron Voice Agent is an end-to-end voice-agent blueprint built on Pipecat.
It combines NVIDIA NIM services for automatic speech recognition (ASR), large
language model (LLM) inference, and text-to-speech (TTS) synthesis. Preserve
the repository's supported cloud, workstation, DGX Spark, and Jetson Thor
deployment profiles when changing shared behavior.

## Sources of Truth

- Use `pyproject.toml` and `uv.lock` for Python versions and dependencies.
- Use `client/package.json` and `client/package-lock.json` for client
  dependencies and scripts.
- Use `examples_registry.yaml` for registered examples, transports, and
  per-example defaults.
- Use each `src/examples/<example>/pipeline.py`, `prompts.yaml`,
  `services.cloud.yaml`, and `services.local.yaml` for example behavior and
  configuration. Shared runtime behavior lives in `src/examples/shared/`,
  `src/server.py`, and the other root modules in `src/`.
- Use `docker-compose.yml` and the files in `docker/` for Compose profiles,
  service names, container images, ports, and hardware-specific deployment
  behavior.
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
- Load `skills/deploy/SKILL.md` for deployment or startup troubleshooting.
- Load `skills/configure-pipeline/SKILL.md` for changes to `.env`,
  `examples_registry.yaml`, prompts, service catalogs, transports, tracing, or
  audio settings.
- Load `skills/upgrade-pipecat/SKILL.md` before changing Pipecat server or
  client dependency versions or migrating Pipecat APIs.
- Preserve unrelated configuration keys, comments, examples, and deployment
  profiles. Do not infer hardware, credentials, or private-service access.

## Validation

Run checks that match the changed surfaces. The GitHub Actions workflow uses
these commands:

```bash
uvx ruff@0.15.6 check .
uvx ruff@0.15.6 format --check .
uv sync --dev
uv run pytest tests/ -v
npm --prefix client ci
npm --prefix client run lint
npm --prefix client run build
```

For documentation-only changes, run the configured pre-commit hooks on the
changed files:

```bash
uv run --project . --group dev pre-commit run --files <changed-files>
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

### NVIDIA DORI Routing

Select the documentation path from current host capabilities.
Do not ask the user to classify themselves or store repository-scoped identity
state during a normal documentation task.

1. Check whether the current agent exposes `dori_handle` or `dori_route` and
   `dori_collections`.
   If the user explicitly asks not to use DORI, use the
   [Writing Style Guide](docs/AGENTS.md#writing-style-guide) instead.
2. When those tools are available, list the installed collections.
   - If a collection source exactly matches `gitlab:tech-docs/skill-library`, use DORI for
     task routing.
   - If the collection is missing, inaccessible, or cannot be verified,
     continue with the
     [Writing Style Guide](docs/AGENTS.md#writing-style-guide).
3. When the DORI tools are unavailable, continue with the Writing Style Guide.
   Do not inspect a shell-visible CLI, install software, or configure the host
   during a normal documentation task.
4. Use [NVIDIA DORI Setup](docs/DORI_SETUP.md) only when the user explicitly
   asks to install or configure DORI.

Capability detection does not approve installation or host configuration.
DORI unavailability must not block documentation work.
