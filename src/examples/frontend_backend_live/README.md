# Frontend/Backend Live Cascaded Example

This example is a cascaded voice agent that puts a routing talker in front of backend work. On every caller turn, the talker, a fast large language model (LLM) call, speaks a reply, delegates the turn to the backend, or waits. The delegation core runs the delegated work on a backend adapter, and a writer words every ready result into one line, which the voice says when the caller is idle. The reference domain is the Bluebird Cafe ordering assistant: the backend looks up the menu, edits a cart, and places an order.

The example has two ways in:

- **Live session clients**, such as Pipecat apps and benchmark clients, connect through `POST /v1/live/sessions` or `WS /v1/live/sessions` and supply the prompts. For details, refer to [Live Sessions](#live-sessions).
- **The repository UI** (`/`) runs a bot whose backend runs the cafe tools on the server and shows each delegation as a task card. The Realtime gateway (`WS /v1/realtime`) drives the same bot.

Four parts serve a live session, and the [Frontend/Backend Live Architecture](../../../docs/design/architecture.md) page describes what each one owns and how they meet:

- The **Live gateway** ([`src/gateways/live/`](../../gateways/live/)) serves the live session protocol.
- The **frontend adapter** is the example's `adapter/`, `cascade/`, `media/`, `models/`, `config_manager/`, `prompts.py`, and `tables.py`. It turns each live session into a Pipecat cascade of automatic speech recognition (ASR), the talker, the writer, and text-to-speech (TTS).
- The **delegation core** ([`delegation_core/`](delegation_core/)), a package of its own with its own README, takes each delegation through the OpenAI Responses API. Its interaction manager (IM) owns the delegated work from then on.
- The **backend adapter** runs the work: the backend LLM, directly connected (the default); the backend LLM agent, a model conductor with LLM workers; or the harness, an external agent that you deploy, such as Hermes.

Compared to the [Frontend/Backend Agent](../frontend_backend_agent/README.md), the talker returns a structured decision on every turn instead of calling a `call_backend` tool, and the writer words the backend's results for speech.

## Default Models

The `services` lists in [`examples_registry.yaml`](../../../examples_registry.yaml) resolve to the following default models for each profile. Service entries live in the root [`services.yaml`](../../../services.yaml):

| Profile | ASR | Talker LLM | Backend LLM | TTS |
| --- | --- | --- | --- | --- |
| Cloud | Nemotron ASR Streaming English | Nemotron 3.5 Lightning 30B A3B | Nemotron 3 Ultra 550B A55B with reasoning enabled | Magpie TTS Multilingual |
| Server | Nemotron ASR Streaming English NIM | Nemotron 3.5 Lightning 30B A3B NIM | Nemotron 3 Ultra 550B A55B with reasoning enabled, on the public NVIDIA API | Magpie TTS Multilingual NIM |
| Single GPU | Nemotron Speech Streaming English 0.6B through NeMo-Speech.cpp | Nemotron 3.5 Lightning 30B A3B through vLLM | Nemotron 3.5 Lightning 30B A3B through vLLM with reasoning enabled | Magpie TTS Multilingual through NeMo-Speech.cpp |

The talker's `llm` slot lists Nemotron 3.5 Lightning. The backend's `thinker-llm` slot lists Nemotron 3 Ultra, then Lightning. A profile's default is the first entry that its catalog serves. Ultra has no local server: the Cloud and Server profiles call it on the public NVIDIA API (`integrate.api.nvidia.com`) with `NVIDIA_API_KEY`, and Single GPU keeps its local Lightning, with Ultra listed after it when `NVIDIA_API_KEY` is set. Each slot's `settings` in `examples_registry.yaml` set its sampling. The talker runs at temperature 0, with reasoning off and at most 512 output tokens. The backend on Ultra runs at temperature 0, with reasoning on, at most 4,096 output tokens, and no reasoning budget, which Ultra's endpoint refuses. On Nemotron 3.5 Lightning, the backend runs at temperature 0.7 with a 256-token reasoning budget.

## Running the Example

This example runs with the **Cloud**, **Server**, and **Single GPU** profiles. Server is workstation-only (not DGX Spark or Jetson Thor), and Single GPU covers workstations, DGX Spark, and Jetson Thor. For prerequisites and hardware detail, refer to the [Getting Started guide](../../../docs/01-getting-started.md). Run every command from the repository root.

1. Preserve any existing `.env` file. Otherwise, copy the template, and then set `NVIDIA_API_KEY` in `.env` for the Cloud or Server profile. For the Single GPU profile, set `HF_TOKEN` instead, which the profile uses to download model weights from Hugging Face:

   ```bash
   test -f .env || cp .env.example .env
   ```

2. Log in to the NVIDIA NGC container registry (Server only):

   ```bash
   set -a; . ./.env; set +a
   printf '%s' "$NVIDIA_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin
   ```

3. Deploy the profile that matches your hardware:

   ```bash
   docker compose --profile frontend-backend-live up -d              # Cloud ASR, LLM, TTS
   docker compose --profile frontend-backend-live/server up -d       # Local NIM ASR, TTS, LLM

   # One GPU (incl. DGX Spark and Jetson Thor). Download speech weights once, as your user:
   bash scripts/download-nemo-speech-models.sh
   docker compose --profile frontend-backend-live/single-gpu up -d   # Lightning + NeMo-Speech.cpp
   ```

   Each profile starts the following app service and sidecars:

   | Recipe profile | App service | Sidecars |
   | --- | --- | --- |
   | `frontend-backend-live` | `frontend-backend-live` | none (cloud NVCF) |
   | `frontend-backend-live/server` | `frontend-backend-live-server` | `nvidia-llm`, `nemotron-asr-streaming-english`, `magpie-multilingual-tts-service` |
   | `frontend-backend-live/single-gpu` | `frontend-backend-live-single-gpu` | `nvidia-llm-vllm-lightning-frontend-backend`, `nemo-speech` |

4. Open the UI at `https://localhost:7860/` and ask what teas the cafe has. Keep TLS enabled for browser testing, because browsers allow the microphone only on `localhost` or over HTTPS. `PIPELINE_TLS=false` serves plain HTTP. For plain-HTTP browser testing, refer to [browser access](../../../docs/06-troubleshooting.md#browser-access).

5. Clean up when you are done by tearing down with the same profile that you started, for example `docker compose --profile frontend-backend-live down`.

To run host-native without Docker, set `selection: frontend-backend-live` in [`examples_registry.yaml`](../../../examples_registry.yaml), then run `uv run python3 src/server.py`.

## Configuration

The following files configure the example. The server reads `config.yaml` and `core.yaml` once, on first use, so an edit takes effect at the next restart. Unknown keys are errors, and a key in the wrong file stops the load with a message that names where it belongs.

| File | What it sets |
| --- | --- |
| [`examples_registry.yaml`](../../../examples_registry.yaml) | The `services.yaml` entries that each slot offers (`llm` for the talker and the writer, `thinker-llm` for the backend), and their sampling and reasoning settings. |
| [`config.yaml`](config.yaml) | The frontend: the frontend adapter (`frontend.adapter`) and its block (`frontend.cascade`), and the UI bot's core profile and channel context (`ui_bot`). `LIVE_CONFIG_PATH` selects another file. |
| [`core.yaml`](core.yaml) | The delegation core: the IM's settings (`delegation`), the lines that the IM and the backend write (`texts`), and the backend adapter with its blocks (`backend`). `CORE_CONFIG_PATH` selects another file. |
| [`prompts.yaml`](prompts.yaml) | The talker prompt that the repository UI lists (`cafe`) and the other prompts (`agent_prompts`). Refer to [Prompts](#prompts). |

A profile is a file that names the file it builds on (`extends: core.yaml`) and sets only what differs. The example ships the following profiles:

| Profile | Use |
| --- | --- |
| [`core.agent.yaml`](core.agent.yaml) | The backend LLM agent (`backend.adapter: llm-agent`). |
| [`core.harness.yaml`](core.harness.yaml) | The harness (`backend.adapter: harness`), connected to an agent that you deploy. |
| [`core.ui-bot.yaml`](core.ui-bot.yaml) | The repository UI bot: a display part for the task cards, and the cafe's backend prompt and server tools (`backend.prompt` and `backend.tools`). `ui_bot.core_config` in `config.yaml` names it. |

In `config.yaml`, `frontend.cascade` sets the talker's and the writer's model and prompt, and the cascade's behavior. The following excerpt shows the keys that you are most likely to change:

```yaml
frontend:
  adapter: cascade
  cascade:
    provider: chat-completions     # chat-completions (any OpenAI-compatible chat endpoint) | openai-responses
    slot: llm                      # the examples_registry.yaml services slot of the model
    prompt: generic                # the talker prompt (agent_prompts.Talker) used, as written, when a session sets no instructions or blank ones
    delegation:
      app_wait_seconds: 50         # the app-wait line when the app's outputs take this long; null: none
      result_wording: writer       # writer | as_written
      progress_after_seconds: 8    # the writer can word a progress line after this long of silent work; null: none
      progress_max_per_delegation: 3  # progress lines offered for one delegation
    memory:
      context_tokens: 16000        # the talker's memory budget
    tts_preroll_ms: 300            # reply audio held before playback starts; 0: off
```

The same block also holds the retries and timeouts (`reliability`), the optional normalizer of spelled details (`normalizer`), and the speech settings (`speech`), each with a comment in the file. A model endpoint uses `NVIDIA_API_KEY` unless its `services.yaml` entry sets `api_key_env` to another variable. For what the speech and delegation settings do, refer to [How the Frontend Media Plane Meets the IM](../../../docs/design/architecture.md#how-the-frontend-media-plane-meets-the-im) and [Spoken Progress](../../../docs/design/architecture.md#spoken-progress).

In `core.yaml`, `backend.adapter` selects the backend adapter, and `delegation` sets the IM's limits, which [Build and Configure the Core](delegation_core/README.md#build-and-configure-the-core) describes.

To switch adapters, set `CORE_CONFIG_PATH` to its profile in `.env` (for Compose) or in your shell (on the host), for example `CORE_CONFIG_PATH=src/examples/frontend_backend_live/core.agent.yaml`, and restart the app service. For how the three adapters differ, refer to [The Three Backend Adapters](../../../docs/design/architecture.md#the-three-backend-adapters). The harness reaches an Agent Client Protocol (ACP) agent that you deploy, such as Hermes, through the `backend.harness` block of `core.yaml`. To deploy Hermes and connect it over stdio or WebSocket, refer to [Connect an Agent Harness](../../../docs/how-to/connect-an-agent-harness.md).

### Prompts

[`prompts.yaml`](prompts.yaml) holds the example's prompts and the texts of the cascade's own model calls. The file's top-level `cafe` entry is the Bluebird Cafe talker prompt, which the repository UI lists as the default (`defaults.prompt` in `examples_registry.yaml`). The **Prompts** tab shows the other texts read-only, in the `agent_prompts` groups that [Files and Profiles](../../../docs/design/architecture.md#files-and-profiles) describes.

Each session reads the file when it starts, so an edit takes effect at the next session. The Live core reads its default backend prompt when it is built, so restart the server after you edit that prompt. A prompt file that you start the server with (`PROMPT_FILE_PATH`) replaces the whole file, so it must also hold the `agent_prompts` entries.

## Live Sessions

A live session selects its delegation mode for the whole session: in Responses mode, the server runs the backend work, and in client mode, the client app does it. For the routes, the modes, the settings, and when the server serves the routes, refer to [Use the Live Gateway](../../../docs/how-to/use-live-gateway.md).

## Customization

The following list describes the common changes and where each one is documented:

- **Prompts.** Edit [`prompts.yaml`](prompts.yaml) as [Prompts](#prompts) describes, or send your own prompts in each session's `instructions` and `delegation.responses.instructions`.
- **Server tools.** A tool source runs tools on the server for the backend, as the cafe tools do under `backend.tools` in [`core.ui-bot.yaml`](core.ui-bot.yaml). Refer to [Add a Tool Source](../../../docs/design/architecture.md#add-a-tool-source).
- **Frontend or backend model provider.** Refer to [Add a Model Provider](../../../docs/design/architecture.md#add-a-model-provider).
- **Frontend adapter, backend adapter, or harness agent.** Refer to [Extension Recipes](../../../docs/design/architecture.md#extension-recipes).

For models, services, and prompts in general, refer to [Configure LLM](../../../docs/how-to/configure-llm.md), [Configure Services](../../../docs/how-to/configure-services.md), and [Configure Prompts](../../../docs/how-to/configure-prompts.md).

## Troubleshooting

For a Live route that answers 404, a refused session, `responses_validation_failed`, or a harness that does not connect, refer to [Frontend/Backend Live](../../../docs/06-troubleshooting.md#frontendbackend-live) in the Troubleshooting guide. For deployment and general failure modes, refer to the [Troubleshooting guide](../../../docs/06-troubleshooting.md).

## Tips and Best Practices

Follow these practices:

- **Repeat the talker's limits in the backend's instructions.** With `result_wording: writer`, the writer reads the talker's instructions and keeps any limits that they set on what the caller may be told. `as_written`, and every writer fallback for a result, says the backend's text as written, cleaned only for speech, so also put those limits in the backend's instructions: `delegation.responses.instructions`, or your own backend's prompt in client mode. For the writer's fallbacks, refer to [The Result Lifecycle](../../../docs/design/architecture.md#the-result-lifecycle).
- **Keep the app-wait line before the deadline.** When a delegation still waits for the app's function outputs after `frontend.cascade.delegation.app_wait_seconds` (50), the voice says the `app_wait_line` from `agent_prompts.Lines` in [`prompts.yaml`](prompts.yaml) once. Keep this value below the core's deadline for those outputs, `delegation.client_tool_timeout_seconds` (60) in `core.yaml`, so the line plays while the work still waits.
- **Change the UI bot's backend with its talker prompt.** A new top-level talker prompt runs with the cafe's channel context, backend prompt, and tools, and the greeting lines (`intro` and `greeting`) stay the cafe's. For another domain, also change `backend.prompt` and `backend.tools` in [`core.ui-bot.yaml`](core.ui-bot.yaml), `ui_bot.channel` in `config.yaml`, and the greeting lines in `agent_prompts.Lines`.
