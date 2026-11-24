# Frontend/Backend Live Cascaded Example

The Pipecat server runs the same pipeline for the Bluebird Cafe UI bot and Live API sessions. It uses automatic speech recognition (ASR), a talker large language model (LLM), and text-to-speech (TTS). The talker replies, delegates, or waits. The backend does the delegated work. The frontend uses recent conversation to word backend replies and speaks when the caller is idle.

Open the repository UI to ask about the cafe menu, update a cart, or place an order. Each task card in the **Conversation** tab tracks one delegated request and shows the status of any jobs that request starts. If routing fails without assigning any jobs, the card's **Stage** field shows **Couldn't start**. The **Backend tools** section of the **Tools** tab lists the functions configured for the browser example. To connect your own app over WebRTC or WebSocket and supply its prompts and tools, follow the [Live Gateway guide](../../../../../docs/how-to/use-live-gateway.md).

## Default Models

[`examples_registry.yaml`](../../../../../examples_registry.yaml) selects the models for each profile from [`services.yaml`](../../../../../services.yaml).

| Profile | ASR | Talker | Backend | TTS |
| --- | --- | --- | --- | --- |
| Cloud | Nemotron ASR Streaming English | Nemotron 3.5 Lightning 30B A3B | Nemotron 3.5 Super VL 120B A12B with reasoning | Magpie TTS Multilingual |
| Server | Nemotron ASR Streaming English NIM | Nemotron 3.5 Lightning 30B A3B NIM | Nemotron 3.5 Super VL 120B A12B NIM with reasoning | Magpie TTS Multilingual NIM |
| Single GPU | Nemotron Speech Streaming English 0.6B through NeMo-Speech.cpp | Nemotron 3.5 Lightning 30B A3B through vLLM | Nemotron 3.5 Lightning 30B A3B through vLLM with reasoning | Magpie TTS Multilingual through NeMo-Speech.cpp |

The talker uses the `llm` slot, and the backend uses `thinker-llm`. Each uses the first catalog entry available for the active recipe. The Server recipe can use Lightning for the backend when Super VL is unavailable. Restart the app to refresh the Live session catalog after changing models.

## Running the Example

The Cloud recipe uses hosted services. The Server recipe requires a workstation with 3 GPUs. It uses GPU 0 for Lightning, ASR, and TTS, and GPUs 1 and 2 for Super VL. Single GPU supports workstations, DGX Spark, and Jetson Thor. For prerequisites, refer to [Getting Started](../../../../../docs/01-getting-started.md).

Run these commands from the repository root.

1. Preserve your `.env` file, or copy the template. Set `NVIDIA_API_KEY` for Cloud or Server. For Single GPU, set `HF_TOKEN` for the model download.

   ```bash
   test -f .env || cp .env.example .env
   ```

2. For Server, sign in to the NVIDIA NGC container registry.

   ```bash
   set -a; . ./.env; set +a
   printf '%s' "$NVIDIA_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin
   ```

3. Start one recipe profile. Download the speech weights before starting Single GPU.

   ```bash
   docker compose --profile frontend-backend-live up -d
   docker compose --profile frontend-backend-live/server up -d

   bash scripts/download-nemo-speech-models.sh
   docker compose --profile frontend-backend-live/single-gpu up -d
   ```

4. Open `https://localhost:7860/` and ask what teas the cafe has. For browser connection issues, refer to [Browser Access](../../../../../docs/06-troubleshooting.md#browser-access).
5. When finished, run `docker compose --profile frontend-backend-live down`, using the same profile you started.

For local development without Docker, set `pipecat.selection: frontend-backend-live` in the registry, then run `uv run nva-pipecat`.

## Configuration

Edit the `config` block of the `pipecat.examples.frontend-backend-live` entry in [`examples_registry.yaml`](../../../../../examples_registry.yaml), then restart the app. Omitted optional settings use the defaults in the corresponding settings model.

| Block | What It Controls |
| --- | --- |
| `frontend.cascade` | Talker model slot, [routing format](#talker-style), and [turn and speech settings](#turns-and-interruptions). Refer to the [frontend settings](config_manager/schema.py). |
| `core` | Backend adapter, model slot, and delegation behavior. Refer to [Choose a Backend Adapter](#choose-a-backend-adapter) and the [Core settings](delegation_core/config.py). |
| `ui_bot` | The cafe's channel and overrides for its backend prompt and tools. The shipped `ui_bot.core` selects `llm` for the UI. |

For credentials and backend endpoint APIs, refer to [Configure Services](../../../../../docs/how-to/configure-services.md#frontendbackend-live-backend-endpoints).

### Choose a Backend Adapter

For Live API sessions, set `config.core.backend.adapter` to one of the following values. To change the browser bot, set `config.ui_bot.core.backend.adapter`. Adapter settings under `core.backend` also apply to the browser bot unless you override them in `ui_bot.core.backend`.

| Adapter | How It Runs Work | Workers per Call |
| --- | --- | --- |
| `llm` (default) | Calls the backend model directly. | 1 |
| `llm-agent` | Runs backend model workers. A conductor routes requests when more than one worker is enabled. | `core.backend.llm-agent.max_running` (3 by default) |
| `harness` | Runs workers and, for parallel work, a conductor through an Agent Client Protocol (ACP) agent such as Hermes. | `core.backend.harness.max_running` (3 by default) |

For `llm-agent` or `harness`, set `core.delegation.result_format: speak_display` to provide display and speech outputs. For harness setup, follow [Connect an Agent Harness](../../../../../docs/how-to/connect-an-agent-harness.md). For routing and interruption behavior, refer to the [Delegation Core](delegation_core/README.md).

Set `core.backend.llm-agent.conductor.strict_tool_arguments` to `true` if your Chat Completions endpoint supports strict function arguments and accepts the `submit_decision` schema. The default is `false`.

### Talker Style

Set `frontend.cascade.talker_style` by the model's service key, such as `nemotron-lightning`. The `json` default asks for a structured reply. For endpoints without JSON schema support, set `response_format: none` on the model's service entry. The experimental `tools` style requires an endpoint that streams tool calls with `tool_choice: "auto"`. Refer to [Configure LLM](../../../../../docs/how-to/configure-llm.md#reasoning-parser--tool-calling-self-hosted).

### Turns and Interruptions

While the assistant speaks, `frontend.cascade.speech.interrupt_min_words` recognized words start a caller turn. The default is 2. Smart Turn checks pauses to decide when the caller has finished. With per-sentence synthesis, the frontend tracks playback one sentence at a time.

The frontend can speak a progress update when the backend sends a new progress note. Set `frontend.cascade.delegation.progress_after_seconds` to change the initial delay of 8 seconds or to `null` to disable automatic progress updates.

### Prompts

[`prompts.yaml`](prompts.yaml) contains the `cafe` preset for the UI and the Live defaults under `agent_prompts`. To adapt the UI bot to another domain, update its talker prompt and greeting, `ui_bot.channel`, and the prompt and tools in `ui_bot.core.backend`. For Live prompt overrides, refer to [Frontend/Backend Live Prompt Catalog](../../../../../docs/how-to/configure-prompts.md#frontendbackend-live-prompt-catalog).

## Customization

| Path | Role |
| --- | --- |
| [`pipeline.py`](pipeline.py) and [`factory.py`](factory.py) | Entry points for the UI bot and Live sessions |
| [`frontend.py`](frontend.py) | Shared Pipecat pipeline builder |
| [`delegation_core/`](delegation_core/README.md) | Work routing, tool authorization, and backend adapters |

For deployment and connection issues, refer to [Frontend/Backend Live Troubleshooting](../../../../../docs/06-troubleshooting.md#frontendbackend-live).
