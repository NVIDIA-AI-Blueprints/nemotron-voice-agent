# Generic - cascaded pipeline example

Generic cascaded voice pipeline using Pipecat's built-in NVIDIA services (`NvidiaSTTService` -> `NvidiaLLMService` with function calling -> `NvidiaTTSService`). It is a minimal, production-shaped cascaded voice assistant that keeps ASR, LLM, tools, and TTS as separate services. When you use `NvidiaWordTTSService` with Magpie TTS Multilingual NIM 1.10.0 or newer, you can opt into word-level input streaming and timestamp-based LLM context commits. Other TTS models, older Magpie NIM versions, and the GGML/GGUF-based NeMo-Speech.cpp backend do not support this option. Refer to [Configure TTS](../../../docs/how-to/configure-tts.md#word-level-input-streaming-and-timestamps).

![Architecture Diagram](../../../docs/images/arch.png)

## Default Models

The defaults in [`examples_registry.yaml`](../../../examples_registry.yaml) resolve to the following models for each profile:

| Profile | ASR | LLM | TTS |
| --- | --- | --- | --- |
| Cloud | Nemotron ASR Streaming English | Nemotron 3.5 Lightning 30B A3B | Magpie TTS Multilingual |
| Server | Nemotron ASR Streaming English NIM | Nemotron 3.5 Lightning 30B A3B NIM | Magpie TTS Multilingual NIM |
| Server Performance | Nemotron ASR Streaming English NIM | Nemotron 3.5 Lightning 30B A3B NIM with a pinned NVFP4 TP2 profile | Magpie TTS Multilingual NIM with `batch_size=64` |
| Single GPU | Nemotron Speech Streaming English 0.6B through NeMo-Speech.cpp | Nemotron 3.5 Lightning 30B A3B through vLLM | Magpie TTS Multilingual through NeMo-Speech.cpp |

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

## Customization

| Path | Role |
| --- | --- |
| `pipeline.py` | Pipecat entry point for the generic example |
| `prompts.yaml` | example-local prompt catalog. Each entry may list `tools_available` to gate function calling per prompt |
| `tools.yaml` | OpenAI function-calling schemas, keyed by tool name |
| `tool_handlers.py` | async handlers for each schema in `tools.yaml`, exposed through the `TOOL_HANDLERS` registry |
| `tools.py` | builds a filtered `ToolsSchema` from `tools.yaml` for the tool names a prompt requests, skipping entries without a matching handler |
| `services.cloud.yaml`, `services.local.yaml` | example-local service catalogs |

To change models, voices, prompts, or tool wiring, see [Configure Services](../../../docs/how-to/configure-services.md), [Configure LLM](../../../docs/how-to/configure-llm.md), [Configure ASR](../../../docs/how-to/configure-asr.md), [Configure TTS](../../../docs/how-to/configure-tts.md), and [Configure Prompts](../../../docs/how-to/configure-prompts.md).

## Tips & best practices

- **Start from this baseline.** The generic example is intentionally minimal. Add domain logic, custom tools, and deployment-specific service choices on top of it rather than starting from scratch.
- **Pick the model for the deployment.** Nemotron 3.5 Lightning is the default across profiles. Nemotron 3 Super is a higher-capability alternative that you can deploy locally. Its build.nvidia.com endpoint is deprecated. See [Configure LLM](../../../docs/how-to/configure-llm.md) for sizing and precision.
- **Tune turn-taking and latency** with the shared pipeline knobs in [Tune Pipeline Performance](../../../docs/how-to/tune-pipeline-performance.md).
- For deployment, ASR/LLM/TTS, and general failure modes, see the [Troubleshooting guide](../../../docs/06-troubleshooting.md).
