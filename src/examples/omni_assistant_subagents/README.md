# Nemotron 3 Omni Assistant Subagents - cascaded pipeline example

Multi-agent variant of [`omni-assistant`](../omni_assistant/README.md) built on Pipecat's built-in multi-agent framework (`pipecat.workers`). A transport agent owns I/O and TTS, a speaker agent owns spoken output, and worker agents handle uploaded media, live webcam vision, and deliberate reasoning. It keeps the voice conversation responsive while specialized agents analyze uploaded media and live webcam frames and escalate difficult turns to a reasoning pass.

The pattern splits responsibility across a transport agent, speaker agent, media analyzer, webcam agent, and thinker using `pipecat.workers`, with explicit dispatch and acknowledgement points. It showcases agent boundaries and prompt separation, visual barge-in, deferred media dispatch, rolling webcam scene summaries, on-demand high-resolution capture, proactive hand-gesture behavior, and UI capability declarations for attachments and webcam support.

![Omni Assistant Subagents architecture](images/omni-subagent-example.jpeg)

## Default Models

The defaults in [`examples_registry.yaml`](../../../examples_registry.yaml) resolve to the following models for each profile:

| Profile | Audio-input LLM | TTS |
| --- | --- | --- |
| Cloud | Nemotron 3 Nano Omni 30B A3B Reasoning | Magpie TTS Multilingual |
| Server | Nemotron 3 Nano Omni 30B A3B Reasoning NIM | Magpie TTS Multilingual NIM |
| Single GPU | Nemotron 3 Nano Omni 30B A3B Reasoning through vLLM | Magpie TTS Multilingual through NeMo-Speech.cpp |

Nemotron Omni handles both speech recognition and response generation, so this example does not configure a separate ASR model. The agents share this model and apply role-specific prompts and reasoning settings.

## Running the example

This example runs with **Cloud**, **Server** (Omni NIM + NIM TTS, recommended for scaling), and **Single GPU** profiles. Server is workstation-only (not DGX Spark or Jetson Thor). The single-gpu profile covers workstations and DGX Spark. It is **not supported on Jetson Thor**. See the [Getting Started guide](../../../docs/01-getting-started.md) for prerequisites and hardware detail. Run every command from the repository root.

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
   docker compose --profile omni-assistant-subagents up -d              # Cloud (no local GPU)
   docker compose --profile omni-assistant-subagents/server up -d  # Server (Omni NIM + NIM TTS, recommended for scaling)

   # One GPU (workstation or DGX Spark). Download speech weights once, as your user:
   bash scripts/download-nemo-speech-models.sh
   docker compose --profile omni-assistant-subagents/single-gpu up -d
   ```

   | Recipe profile | App service | Shared sidecars pulled from `docker/` |
   | --- | --- | --- |
   | `omni-assistant-subagents` | `omni-assistant-subagents` | none (cloud NVCF) |
   | `omni-assistant-subagents/server` | `omni-assistant-subagents-server` | `nvidia-llm-omni`, `magpie-multilingual-tts-service` |
   | `omni-assistant-subagents/single-gpu` | `omni-assistant-subagents-single-gpu` | `nvidia-llm-vllm-omni`, `nemo-speech-tts` |

4. Open the UI at `https://localhost:7860/`. Keep TLS enabled for browser UI testing. `PIPELINE_TLS=false` serves plain HTTP for headless performance and API testing. For plain-HTTP browser testing, see [browser access](../../../docs/06-troubleshooting.md#browser-access).

5. Clean up when you are done by tearing down with the same profile you started with:

   ```bash
   docker compose --profile omni-assistant-subagents down              # Cloud (no local GPU)
   docker compose --profile omni-assistant-subagents/server down       # Server
   docker compose --profile omni-assistant-subagents/single-gpu down   # One GPU (incl. DGX Spark)
   ```

To run host-native without Docker, set `selection: omni-assistant-subagents` in [`examples_registry.yaml`](../../../examples_registry.yaml), then run `uv run python3 src/server.py`.

## Customization

| Path | Role |
| --- | --- |
| `pipeline.py` | entry point that wires the five workers into a `WorkerRunner` over a shared `WorkerBus` |
| `subagents/speaker/agent.py` | `SpeakerOmniAgent` plus a structured-JSON wrapper around `NvidiaOmniLLMService` |
| `subagents/transport/agent.py` | `OmniTransportAgent` for transport I/O, TTS, visual barge-in, analyzer dispatch, and the pinned subagent state board |
| `subagents/media_analyzer/agent.py` | `MediaAnalyzerWorker` for uploaded image, audio, and video attachments |
| `subagents/webcam/agent.py` | `WebcamAgent` rolling scene summaries for live webcam context |
| `subagents/thinker/agent.py` | `ThinkerWorker` reruns difficult or low-confidence turns with reasoning enabled |
| `media_dispatch_processor.py` | frame-processor that defers analyzer dispatch until the speaker ack closes |
| `subagents.yaml` | source of truth for worker capabilities, routing rules, reasoning modes, and UI labels |
| `prompts.yaml` | example-local prompt catalog (top-level prompt plus `agent_prompts:` per agent) |
| `services.cloud.yaml`, `services.local.yaml` | example-local service catalogs for cloud and on-prem deployments |

The example declares `capabilities: [attachments, webcam]` in `examples_registry.yaml`, which gates these UI surfaces and backend endpoints:

| Endpoint | Purpose |
| --- | --- |
| `POST /api/sessions/{session_id}/attachments?kind={image,audio,video}` | Upload a media attachment for the media analyzer |
| `POST /api/sessions/{session_id}/webcam/frames` | Upload one webcam JPEG frame |
| `GET /api/webcam-config` | Browser webcam capture defaults |

## Tips & best practices

- **Keep the voice loop responsive.** Media, webcam, and reasoning analysis run as separate worker agents so the transport and speaker agents never block on vision or reasoning work. Preserve that split when adding new capabilities.
- **Reuse the deferred-dispatch pattern.** `media_dispatch_processor.py` holds analyzer dispatch until the current spoken turn finishes, which avoids cutting the user off. Reuse it for any new asynchronous worker.
- **Model selection and VRAM** follow the Omni sizing in [Configure LLM](../../../docs/how-to/configure-llm.md). For deployment and general failure modes, see the [Troubleshooting guide](../../../docs/06-troubleshooting.md).
