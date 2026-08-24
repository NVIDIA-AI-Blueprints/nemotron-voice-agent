---
name: configure-pipeline
description: Configure Nemotron Voice Agent runtime via `.env`, example-local `services.{cloud,local}.yaml`, and example-local `prompts.yaml`. Use when changing prompts, tracing, audio knobs, exposed pipelines or transports, or local NIM image overrides.
version: "2.1.1"
metadata:
  author: NVIDIA Voice Agent Team <nemotron-voice-agent@nvidia.com>
  tags: [configuration, pipeline, voice-agent, nemotron]
---

# Configure Nemotron Voice Agent Pipeline

## Purpose

Edit the runtime configuration of the voice agent (built-in catalogs, prompts, feature flags) and re-apply Compose without rebuilding images.

## Prerequisites

- An existing deployment created by `deploy`.

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
   - `<example-package-dir>/services.cloud.yaml` (remote / NVCF) and `<example-package-dir>/services.local.yaml` (Compose-managed local services nested under `server` / `singlegpu`, matching the example's supported `<example-id>/<hardware>` recipes): built-in LLM, ASR, TTS, and example-specific role catalogs.
   - `<example-package>/prompts.yaml`: built-in prompt presets and prompt content for the active example.

3. Validate:
   - Multilingual prompts must use multilingual-capable ASR and TTS from the active catalog (e.g. `parakeet-rnnt`, `nemotron-asr-streaming-multilingual`, `magpie-multilingual-tts`, `magpie-zeroshot-tts`, `chatterbox-multilingual-tts`). Verify the keys exist before referencing them.
   - Every TTS catalog entry must set `synthesis_mode` explicitly. Use `stitched` for Magpie Multilingual and Magpie Zeroshot, and `per_sentence` for Chatterbox.
   - Local catalog endpoints must use Compose service names (`nemotron-asr-streaming-english:50052`, `nemotron-asr-streaming-multilingual:50052`, `parakeet-ctc-asr:50052`, `parakeet-rnnt-asr:50052`, `magpie-multilingual-tts-service:50051`, `magpie-zeroshot-tts-service:50051`, `chatterbox-tts-service:50051`, `nvidia-llm:8000`, `nvidia-llm-vllm:8000`, `nvidia-llm-vllm-omni:8002`, `nemo-speech:50051`, `nemo-speech-multilingual:50051`, `nemo-speech-tts:50051`, `booking-server:8001`). Host-run backends auto-rewrite to the matching `localhost` ports.
   - Alternate local ASR/TTS use Compose profiles (`parakeet-ctc-asr`, `parakeet-rnnt-asr`, `chatterbox-tts`, `magpie-zeroshot-tts`) and share ports with the default. Scale the default off (e.g. `--scale magpie-multilingual-tts-service=0` or `--scale nemotron-asr-streaming-multilingual=0`). Stop `chatterbox-tts-service` or `magpie-zeroshot-tts-service` before returning to Magpie Multilingual.
   - A `*/server` recipe that colocates the default ASR, LLM, and TTS on one GPU needs about 80 GB of VRAM. This does not apply to `*/single-gpu` recipes, which use the memory-fit procedure in the `deploy` skill.

4. Apply and verify using `references/apply-changes.md`.

## Rules

- `.env` changes: compose re-apply.
- YAML catalog changes (`prompts.yaml`, `services.*.yaml`, `examples_registry.yaml`): compose restart of the example service. `./src` and `./examples_registry.yaml` are bind-mounted, so no rebuild needed.
- Preserve unrelated keys, comments, and entries while editing.
- Alternate local ASR/TTS use Compose profiles where needed (`parakeet-ctc-asr`, `parakeet-rnnt-asr`, `chatterbox-tts`, `magpie-zeroshot-tts`). Magpie Multilingual uses `magpie-multilingual-tts-service`. Magpie Zeroshot uses `magpie-zeroshot-tts-service`.
- Per-example slot defaults live in `examples_registry.yaml` `defaults`. The catalog file ordering only affects UI listings. The actual default is whatever `defaults` declares.

## Examples

**Switch the default LLM to a different cloud model:**

1. Open `examples_registry.yaml` and update the relevant `defaults` entry for the active example (e.g. change `llm: [nemotron-lightning]` to `llm: [nemotron-super]`). The catalog key must exist in the active example's `services.cloud.yaml` / `services.local.yaml`.
2. Compose restart of the example service and refresh browser.

**Add a multilingual persona prompt:**

1. Add the prompt to the active example's `prompts.yaml`.
2. To make it the per-example default, update `examples_registry.yaml` `defaults.prompt` for that example to the new prompt key.
3. Ensure the active example's catalog has multilingual-capable ASR (`parakeet-rnnt` or `nemotron-asr-streaming-multilingual`) and TTS (`magpie-multilingual-tts`, `magpie-zeroshot-tts`, or `chatterbox-multilingual-tts`).
4. Compose restart of the example service and refresh browser.

## Limitations

- Does not deploy the stack or change profiles. Use `deploy` for that.
- `NvidiaWordTTSService` is a source-level opt-in, not an `.env` or service-catalog setting. When word-level input streaming and timestamp-based context commits are requested, change only the service import and constructor in the example's `pipeline.py` as documented in `docs/how-to/configure-tts.md#word-level-input-streaming-and-timestamps`, then restart the example service.
- Other source customization is out of scope. Dependency or `Dockerfile` changes require an image rebuild (`--build`).
- UI-only ad-hoc service / prompt overrides (saved in `localStorage`) are intentionally not persisted. This skill writes only to repo files.

## Troubleshooting

- **YAML catalog change does not appear in the UI** -> compose re-apply and refresh browser.
- **`.env` change has no effect on a running container** -> environment is read at container start. Re-apply Compose so the container restarts.
- **Local LLM/ASR/TTS missing from the Services tab** -> the corresponding sidecar is not deployed or is unreachable. The catalog filters local entries by TCP reachability.
- **Local cascaded single-GPU LLM won't start or OOMs** -> inspect the capability-selected Lightning recipe in `docker/docker-compose.nemotron35-lightning.yaml` and verify the host meets its memory requirement. Keep vLLM tuning out of `.env`. Use cloud services when the supported recipe does not fit.
- **Multilingual responses do not use the right ASR/TTS** -> reorder catalog so a multilingual ASR/TTS sits first, or pick the entry from the UI Services tab.
- **ASR/TTS sidecar image fails to pull** on a `*/server` recipe -> log in to `nvcr.io` with a `NVIDIA_API_KEY` that has access to the image. Single-GPU recipes do not use `NVIDIA_API_KEY` or `nvcr.io` login. The active image is set in `docker/docker-compose.<variant>.yaml`.
- **Local LLM 400 (`auto tool choice requires ...`), or reasoning spoken / `<think>` leaks** -> self-hosted Nemotron-3 2.x needs the parsers set (already in `docker/docker-compose.nemotron3-*.yaml`): NIM `NIM_PASSTHROUGH_ARGS=--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser nemotron_v3`, or the same flags on `vllm serve`. See `docs/06-troubleshooting.md`.
- **Raw vLLM `nemotron_v3` not found / Super (`MIXED_PRECISION`) won't load** -> image's vLLM too old; use NGC `vllm:26.07-py3` (vLLM ≥ 0.20), not `vllm:25.12.post1-py3` (0.12.0).
