# Nemotron Omni Assistant - cascaded pipeline example

Cascaded voice pipeline that uses Nemotron 3 Nano Omni as a single model for ASR and the LLM, then hands the text reply to Magpie TTS. Nemotron Omni consumes user audio directly and produces the assistant text that Magpie TTS speaks. This example enables only text and audio inputs. Uploaded media and webcam vision are covered by [`omni-assistant-subagents`](../omni_assistant_subagents/README.md).

The pattern replaces the separate ASR and text LLM stages with one audio-input LLM service while preserving the familiar Pipecat transport, TTS, prompt, and service-catalog flow. It showcases `NvidiaOmniLLMService`, audio-only turn finalization, and a user transcript taken from the Omni response rather than a separate ASR pipeline.

![Nemotron Omni Assistant architecture](images/omni-assistant-architecture.png)

## Running the example

This example runs with **Cloud**, **Server** (Omni NIM + NIM TTS, recommended for scaling), and universal **Single GPU** profiles. Server is workstation-only (not DGX Spark or Jetson Thor). The single-gpu profile covers workstations, DGX Spark, and Jetson Thor. See the [Getting Started guide](../../../docs/01-getting-started.md) for prerequisites and hardware detail. Run every command from the repository root.

1. Preserve any existing `.env` file. Otherwise, copy the template, and then set `NVIDIA_API_KEY` in `.env` for the Cloud or Server profile:

   ```bash
   test -f .env || cp .env.example .env
   ```

   > **Single-GPU profile:** set `HF_TOKEN` in `.env` only. Do not set `NVIDIA_API_KEY` or log in to `nvcr.io`. Omni is served with vLLM, which downloads the model weights from Hugging Face.

2. Log in to the NVIDIA NGC container registry (Server only. Skip for Cloud and Single GPU):

   ```bash
   set -a; . ./.env; set +a
   printf '%s' "$NVIDIA_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin
   ```

3. Deploy the profile that matches your hardware:

   ```bash
   docker compose --profile omni-assistant up -d              # Cloud (no local GPU)
   docker compose --profile omni-assistant/server up -d  # Server (Omni NIM + NIM TTS, recommended for scaling)

   # One GPU (incl. DGX Spark and Jetson Thor). Download speech weights once, as your user:
   bash scripts/download-nemo-speech-models.sh
   docker compose --profile omni-assistant/single-gpu up -d
   ```

   | Recipe profile | App service | Shared sidecars pulled from `docker/` |
   | --- | --- | --- |
   | `omni-assistant` | `omni-assistant` | none (cloud NVCF) |
   | `omni-assistant/server` | `omni-assistant-server` | `nvidia-llm-omni`, `magpie-multilingual-tts-service` |
   | `omni-assistant/single-gpu` | `omni-assistant-single-gpu` | `nvidia-llm-vllm-omni`, `nemo-speech-tts` |

   > The single-GPU recipe uses the Omni vLLM sidecar and takes TTS from the on-device NeMo-Speech.cpp `nemo-speech-tts` service instead of the NIM sidecars. Jetson Thor (128 GB unified memory) fits the 30B Omni NVFP4 model. Follow the [Jetson Thor guide](../../../docs/03-jetson-thor.md). Orin-class Jetson hardware is not supported because the models do not fit.

4. Open the UI at `https://localhost:7860/`. Keep TLS enabled for browser UI testing. `PIPELINE_TLS=false` serves plain HTTP for headless performance and API testing. For plain-HTTP browser testing, see [browser access](../../../docs/06-troubleshooting.md#browser-access).

5. Clean up when you are done by tearing down with the same profile you started with:

   ```bash
   docker compose --profile omni-assistant down              # Cloud (no local GPU)
   docker compose --profile omni-assistant/server down       # Server
   docker compose --profile omni-assistant/single-gpu down   # One GPU (incl. DGX Spark and Jetson Thor)
   ```

To run host-native without Docker, set `selection: omni-assistant` in [`examples_registry.yaml`](../../../examples_registry.yaml), then run `uv run python3 src/server.py`.

## Customization

| Path | Role |
| --- | --- |
| `pipeline.py` | pipecat entry point for the Omni Assistant example |
| `nvidia_omni_multimodal_service.py` | `NvidiaOmniLLMService` (upstream-shaped Pipecat `LLMService` for Nemotron Omni) |
| `audio_only_smart_turn_strategy.py` | smart-turn stop strategy that finalizes turns without an upstream `TranscriptionFrame` |
| `prompts.yaml` | example-local prompt catalog |
| `services.cloud.yaml`, `services.local.yaml` | example-local service catalogs for cloud and on-prem deployments |

Environment variables read by [`pipeline.py`](pipeline.py):

| Env var | Default | Purpose |
| --- | --- | --- |
| `OMNI_MAX_TOKENS` | `8192` | Max tokens for the Omni response |
| `OMNI_TEMPERATURE` | `0.6` | Sampling temperature |
| `OMNI_TOP_P` | `0.95` | Nucleus sampling top-p |
| `OMNI_MIN_USER_AUDIO_SECS` | `0.3` | Drop turns shorter than this |
| `OMNI_EMIT_TRANSCRIPTIONS` | `true` | Ask Omni for `<transcript>`/`<response>` sections so the user's words reach the UI and the conversation history |
| `TTS_STOP_FRAME_TIMEOUT_S` | `30` | TTS audio-context idle timeout |
| `AUDIO_OUT_10MS_CHUNKS` | `5` (WebRTC) / `10` (WebSocket) | Outbound audio framing |

For model selection, voices, and shared service-catalog mechanics, see [Configure LLM](../../../docs/how-to/configure-llm.md), [Configure TTS](../../../docs/how-to/configure-tts.md), and [Configure Services](../../../docs/how-to/configure-services.md).

## Tips & best practices

- **Omni model and hardware.** The Server recipe uses the `nvidia-llm-omni` NIM sidecar with automatic model-profile selection. The Single-GPU recipe uses `nvidia-llm-vllm-omni` and selects the model precision from the GPU at startup. Blackwell workstations, DGX Spark, and Jetson Thor use NVFP4. Hopper and Ada use FP8. See [Configure LLM](../../../docs/how-to/configure-llm.md#vram--hardware-support).
- **Tune Omni behavior** with the environment variables in the table above: keep user transcript on so the UI shows the user's words, raise the minimum-audio threshold if noise triggers spurious turns, and adjust max-tokens and sampling for your latency and verbosity targets.
- For deployment and general failure modes, see the [Troubleshooting guide](../../../docs/06-troubleshooting.md). VRAM sizing for the Omni vLLM sidecar is covered in [Configure LLM](../../../docs/how-to/configure-llm.md#vram--hardware-support), and edge deployment in the [Jetson Thor guide](../../../docs/03-jetson-thor.md).
