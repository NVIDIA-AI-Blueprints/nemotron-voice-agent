# Generic - cascaded pipeline example

Generic cascaded voice pipeline using Pipecat's built-in NVIDIA services (`NvidiaSTTService` -> `NvidiaLLMService` with function calling -> `NvidiaTTSService`). It is a minimal, production-shaped cascaded voice assistant that keeps ASR, LLM, tools, and TTS as separate services. To opt into Magpie word-level input streaming and timestamp-based LLM context commits, see [Configure TTS](../../../docs/how-to/configure-tts.md#word-level-input-streaming-and-timestamps).

![Architecture Diagram](../../../docs/images/arch.png)

## Running the example

This example runs with **Cloud**, **Server** (NIM), benchmark-only **Performance Server** (NIM), and universal **Single GPU** profiles. Server is workstation-only (not DGX Spark or Jetson Thor). The performance profile uses the dedicated four-GPU Blackwell layout and pinned NVFP4 TP2 LLM profile documented in the [scaling benchmark](../../../benchmarking_tools/scaling-perf/README.md#reproducing-the-recommended-scaling-setup). Older hardware requires a compatible TP2 profile. It runs 200 Uvicorn workers for load testing and is not intended for normal browser UI sessions. The single-gpu profile covers workstations, DGX Spark, and Jetson Thor. See the [Getting Started guide](../../../docs/01-getting-started.md) for prerequisites and hardware detail. Run commands from the repository root.

1. Preserve any existing `.env` file. Otherwise, copy the template, and then set `NVIDIA_API_KEY` in `.env` for the Cloud, Server, or Performance Server profile:

   ```bash
   test -f .env || cp .env.example .env
   ```

   > **Single-GPU profile:** set `HF_TOKEN` in `.env` only. Do not set `NVIDIA_API_KEY` or log in to `nvcr.io`. This profile serves the LLM with vLLM, which downloads model weights from Hugging Face. The Server and Performance Server profiles use NIMs from NGC (`NVIDIA_API_KEY`) and do not use `HF_TOKEN`.

2. Log in to the NVIDIA NGC container registry (Server and Performance Server only. Skip for Cloud and Single GPU):

   ```bash
   set -a; . ./.env; set +a
   printf '%s' "$NVIDIA_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin
   ```

3. Deploy the profile that matches your hardware:

   ```bash
   docker compose --profile generic-assistant up -d             # Cloud (no local GPU)
   docker compose --profile generic-assistant/server up -d      # Server (NIM)
   docker compose --profile generic-assistant/server-perf up -d # Four-GPU benchmark only

   # One GPU (incl. DGX Spark and Jetson Thor). Download speech weights once, as your user:
   bash scripts/download-nemo-speech-models.sh
   docker compose --profile generic-assistant/single-gpu up -d
   ```

   | Recipe profile | App service | Sidecars |
   | --- | --- | --- |
   | `generic-assistant` | `generic-assistant` | none (cloud NVCF) |
   | `generic-assistant/server` | `generic-assistant-server` | `nvidia-llm`, `nemotron-asr-streaming-english`, `magpie-multilingual-tts-service` |
   | `generic-assistant/server-perf` | `generic-assistant-server-perf` | `nvidia-llm-perf`, `nemotron-asr-streaming-english-perf`, `magpie-multilingual-tts-service-perf` |
   | `generic-assistant/single-gpu` | `generic-assistant-single-gpu` | `nvidia-llm-vllm-lightning`, `nemo-speech` |

   > The standard Lightning `server` deployment uses NIM's automatic compatible-profile selection. The `single-gpu` deployment uses its hardware-aware vLLM configuration. The four-GPU `server-perf` benchmark pins NVFP4 TP2 for Blackwell. On older hardware, change it to a compatible TP2 profile as described in [Configure LLM](../../../docs/how-to/configure-llm.md).

4. Open the UI at `https://localhost:7860/`. Keep TLS enabled for browser UI testing. `PIPELINE_TLS=false` serves plain HTTP for headless performance and API testing. For plain-HTTP browser testing, see [browser access](../../../docs/06-troubleshooting.md#browser-access).

5. Clean up when you are done by tearing down with the same profile you started with:

   ```bash
   docker compose --profile generic-assistant down             # Cloud (no local GPU)
   docker compose --profile generic-assistant/server down      # Server
   docker compose --profile generic-assistant/server-perf down # Four-GPU benchmark only
   docker compose --profile generic-assistant/single-gpu down  # One GPU (incl. DGX Spark and Jetson Thor)
   ```

To run host-native without Docker, set `selection: generic-assistant` in [`examples_registry.yaml`](../../../examples_registry.yaml), then run `uv run python3 src/server.py`.

### Running the TTM Assistant

The TTM Assistant uses an external TTM inference server to determine when user
turns start and stop. The Pipecat turn processor and Silero fallback live in
this repository; the model deployment remains independent.

1. Start Dockerized TTM, and keep its turn-events endpoint on `ws://127.0.0.1:7860/v1/audio/turn-events`.
2. Start the host voice-agent server on another port, such as `7862`, to avoid conflicting with TTM:

   ```bash
   EXAMPLE_SELECTION=ttm-assistant uv run python3 src/server.py --port 7862
   ```

3. Open the UI at `https://localhost:7862/`.

The default `TTM_TURN_EVENTS_URL` is `ws://127.0.0.1:7860/v1/audio/turn-events`. Set this environment variable only when TTM uses a different endpoint. The connection timeout defaults to 10 seconds; set `TTM_OPEN_TIMEOUT_SECS` if model load requires more time.

TTM owns the primary user turn boundaries for this assistant. If TTM emits SOU
but its matching EOU is delayed or missing, Silero ends the active turn after
five seconds of detected silence. Set `TTM_EOU_FALLBACK_SILENCE_SECS` to tune
this safety window. The generic `USE_SILERO_VAD_TURN_DETECTION`,
`SILERO_VAD_STOP_SECS`, and `SMART_TURN_STOP_SECS` settings do not apply.

## Customization

| Path | Role |
| --- | --- |
| `pipeline.py` | Pipecat entry point for the generic example |
| `ttm_pipeline.py` | TTM Assistant pipeline entry point |
| `ttm_user_turn_processor.py` | TTM SOU/EOU processor and Silero fallback |
| `prompts.yaml` | example-local prompt catalog. Each entry may list `tools_available` to gate function calling per prompt |
| `tools.yaml` | OpenAI function-calling schemas, keyed by tool name |
| `tool_handlers.py` | async handlers for each schema in `tools.yaml`, exposed through the `TOOL_HANDLERS` registry |
| `tools.py` | builds a filtered `ToolsSchema` from `tools.yaml` for the tool names a prompt requests, skipping entries without a matching handler |
| `services.cloud.yaml`, `services.local.yaml` | example-local service catalogs |

To change models, voices, prompts, or tool wiring, see [Configure Services](../../../docs/how-to/configure-services.md), [Configure LLM](../../../docs/how-to/configure-llm.md), [Configure ASR](../../../docs/how-to/configure-asr.md), [Configure TTS](../../../docs/how-to/configure-tts.md), and [Configure Prompts](../../../docs/how-to/configure-prompts.md).

## Tips & best practices

- **Start from this baseline.** The generic example is intentionally minimal. Add domain logic, custom tools, and deployment-specific service choices on top of it rather than starting from scratch.
- **Pick the model for the deployment.** Nemotron 3.5 Lightning suits latency-sensitive local profiles, and Nemotron 3 Super is the higher-capability cloud default. See [Configure LLM](../../../docs/how-to/configure-llm.md) for sizing and precision.
- **Tune turn-taking and latency** with the shared pipeline knobs in [Tune Pipeline Performance](../../../docs/how-to/tune-pipeline-performance.md).
- For deployment, ASR/LLM/TTS, and general failure modes, see the [Troubleshooting guide](../../../docs/06-troubleshooting.md).
