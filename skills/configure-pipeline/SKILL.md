---
name: configure-pipeline
description: Configure Nemotron Voice Agent runtime via `.env`, example-local `services.{cloud,local}.yaml`, and example-local `prompts.yaml`. Use when changing prompts, tracing, audio knobs, exposed pipelines or transports, or local NIM image overrides.
version: "2.0.0"
metadata:
  author: NVIDIA Voice Agent Team <nemotron-voice-agent@nvidia.com>
  tags: [configuration, pipeline, voice-agent, nemotron]
---

# Configure Nemotron Voice Agent Pipeline

## Purpose

Edit the runtime configuration of the voice agent (built-in catalogs, prompts, feature flags) and re-apply Compose without rebuilding images.

## Prerequisites

- An existing deployment created by `deploy` (root compose or one of its per-example references).

## Scope

- Run commands from the repository root.
- Limit repository-backed changes to `.env`, `examples_registry.yaml`, example-local `prompts.yaml`, and per-example service catalogs.
- UI-only prompt or service tests stay in browser localStorage. Redeployment is not required.
- Exposed UI examples and transports live in `examples_registry.yaml` (`selection` and `transports` fields). Use the `EXAMPLE_SELECTION` env var only to override the registry at runtime (e.g., for one-off benchmarks).
- Use `deploy` for initial deployment, profile selection, or auth troubleshooting.

## Instructions

1. Identify the active example by inspecting the running app container (`generic-assistant`, `multilingual-assistant`, `omni-assistant`, `omni-assistant-subagents`, or `frontend-backend-agent`). Each example has its own catalog under its package directory in `src/examples/` (the example id maps to a package dir: `generic-assistant` → `src/examples/generic`, `multilingual-assistant` → `src/examples/multilingual`, `omni-assistant` → `src/examples/omni_assistant`, `omni-assistant-subagents` → `src/examples/omni_assistant_subagents`, `frontend-backend-agent` → `src/examples/frontend_backend_agent`).

2. Edit the smallest configuration surface that satisfies the request:
   - `.env`: feature flags, tracing, chat history, audio debugging, and buffering.
   - `examples_registry.yaml`: visible examples (`selection`), allowed transports (`transports`), and per-example slot defaults (`defaults`).
   - `<example-package-dir>/services.cloud.yaml` (remote / NVCF) and `<example-package-dir>/services.local.yaml` (Compose-managed local NIMs nested under `workstation` / `dgxspark` / `jetson`, matching the example's supported `<example-id>/<hardware>` recipes): built-in LLM, ASR, TTS, and example-specific role catalogs.
   - `<example-package>/prompts.yaml`: built-in prompt presets and prompt content for the active example.

3. Validate:
   - Multilingual prompts must use multilingual-capable ASR and TTS from the active catalog (e.g. `parakeet-rnnt`, `nemotron-asr-streaming-multilingual`, `magpie-tts`). Verify the keys exist before referencing them.
   - Local catalog endpoints must use Compose service names (`nemotron-asr-streaming-english:50052`, `nemotron-asr-streaming-multilingual:50052`, `parakeet-ctc-asr:50052`, `parakeet-rnnt-asr:50052`, `tts-service:50051`, `nvidia-llm:8000`, `nvidia-llm-vllm:8000`, `nvidia-llm-vllm-omni:8002`, `nemotron-speech:50051`, `booking-server:8001`). Host-run backends auto-rewrite to the matching `localhost` ports.
   - ASR/TTS variants are selected via Compose profile (e.g. `parakeet-ctc-asr`, `parakeet-rnnt-asr`).
   - Workstation local Compose runs ASR/TTS and NIM LLM on GPU `0` by default. Single-GPU deployments are supported only when at least 80 GB of VRAM is available.

4. Apply and verify using `references/apply-changes.md`.

## Rules

- `.env` changes: compose re-apply.
- YAML catalog changes (`prompts.yaml`, `services.*.yaml`, `examples_registry.yaml`): compose restart of the example service. `./src` and `./examples_registry.yaml` are bind-mounted, so no rebuild needed.
- Preserve unrelated keys, comments, and entries while editing.
- ASR/TTS variants are selected by Compose profile. Parakeet variants use dedicated `parakeet-ctc-asr` / `parakeet-rnnt-asr` services; Magpie TTS uses `tts-service`.
- Per-example slot defaults live in `examples_registry.yaml` `defaults`. The catalog file ordering only affects UI listings. The actual default is whatever `defaults` declares.

## Examples

**Switch the default LLM to a different cloud model:**

1. Open `examples_registry.yaml` and update the relevant `defaults` entry for the active example (e.g. change `llm: [nemotron-nano]` to `llm: [nemotron-super]`). The catalog key must exist in the active example's `services.cloud.yaml` / `services.local.yaml`.
2. Compose restart of the example service and refresh browser.

**Add a multilingual persona prompt:**

1. Add the prompt to the active example's `prompts.yaml`.
2. To make it the per-example default, update `examples_registry.yaml` `defaults.prompt` for that example to the new prompt key.
3. Ensure the active example's catalog has multilingual-capable ASR (`parakeet-rnnt` or `nemotron-asr-streaming-multilingual`) and TTS (`magpie-tts`).
4. Compose restart of the example service and refresh browser.

## Limitations

- Does not deploy the stack or change profiles. Use `deploy` (and its per-example references) for that.
- Source code or `Dockerfile` changes require an image rebuild (`--build`). Out of scope here.
- UI-only ad-hoc service / prompt overrides (saved in `localStorage`) are intentionally not persisted. This skill writes only to repo files.

## Troubleshooting

- **YAML catalog change does not appear in the UI** -> compose re-apply and refresh browser.
- **`.env` change has no effect on a running container** -> environment is read at container start. Re-apply Compose so the container restarts.
- **Local LLM/ASR/TTS missing from the Services tab** -> the corresponding sidecar is not deployed or is unreachable. The catalog filters local entries by TCP reachability.
- **Local workstation LLM won't start or OOMs** -> match it to the GPU in `.env`: `NIM_KVCACHE_PERCENT` (**raise** on `No available memory for the cache blocks`, lower on an OOM kill), `NIM_TAGS_SELECTOR` (weight precision and tensor-parallel size), and `LLM_MAX_NUM_SEQS` (lower if CUDA-graph capture fails). On multi-GPU hosts, match the NIM profile `tp` to the exposed GPUs. See "VRAM & hardware support" in `docs/how-to/configure-llm.md`.
- **Multilingual responses do not use the right ASR/TTS** -> reorder catalog so a multilingual ASR/TTS sits first, or pick the entry from the UI Services tab.
- **ASR/TTS sidecar image fails to pull** -> log in to `nvcr.io` with a `NVIDIA_API_KEY` that has access to the image. The active image is set in `docker/docker-compose.<variant>.yaml`.
- **Local LLM 400 (`auto tool choice requires ...`), or reasoning spoken / `<think>` leaks** -> self-hosted Nemotron-3 2.x needs the parsers set (already in `docker/docker-compose.nemotron3-*.yaml`): NIM `NIM_PASSTHROUGH_ARGS=--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser nemotron_v3`, or the same flags on `vllm serve`. See `docs/06-troubleshooting.md`.
- **Raw vLLM `nemotron_v3` not found / Super (`MIXED_PRECISION`) won't load** -> image's vLLM too old; use NGC `vllm:26.05.post1-py3` (vLLM ≥ 0.20), not `vllm:25.12.post1-py3` (0.12.0).
