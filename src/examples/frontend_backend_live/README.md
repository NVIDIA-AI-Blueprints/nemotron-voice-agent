# Frontend/Backend Live Cascaded Example

This example is a cascaded voice agent that puts a routing talker in front of backend work. On every caller turn, the talker, a fast large language model (LLM) call, speaks a reply, delegates the turn to the backend, or waits. The delegation core runs the delegated work on a backend adapter, and the talker words every ready result into one line, which the voice says when the caller is idle. The reference domain is the Bluebird Cafe ordering assistant: the backend looks up the menu, edits a cart, and places an order.

The example has two ways in:

- **Live session clients**, such as Pipecat apps or your own app, connect through `POST /v1/live/sessions` or `WS /v1/live/sessions` and supply the prompts. The example's Compose profiles serve the Live routes when `.env` sets `LIVE_API_TOKEN`, or without a token when it sets `LIVE_ENABLED=true`. For the routes, the delegation modes, and the settings, refer to [Use the Live Gateway](../../../docs/how-to/use-live-gateway.md).
- **The repository UI** (`/`) runs a bot whose backend runs the cafe tools on the server and shows each delegation as a task card. The Realtime gateway (`WS /v1/realtime`) drives the same bot.

Four parts serve a live session. The following diagram shows a call from the caller to the backend and back:

```text
caller --- app (Live client) --- LIVE GATEWAY: sessions, media, command order, the wire schema
                                   |                                    |
                                   | frontend part                      | delegation part, by delegation.type
                                   v                                    v
      FRONTEND ADAPTER, a Pipecat pipeline                client: the app does the work
        ASR -> talker: speak | delegate | wait -> TTS     responses: POST /v1/responses to the core
        voice: says each result when the caller is idle                 |
                                   ^                                    v
                                   +----- official Responses events --- DELEGATION CORE: interaction manager
                                                                          | backend port
                                                                          v
                                                                        BACKEND ADAPTER (backend.adapter):
                                                                          llm (default) | llm-agent | harness
```

Each part owns its own decisions:

- The **Live gateway** ([`src/gateways/live/`](../../gateways/live/)) serves the live session protocol.
- The **frontend adapter** is the rest of the example's package, such as `adapter/`, `media/`, `frontend.py`, `talker.py`, and `voice.py`. It turns each live session into a Pipecat pipeline of automatic speech recognition (ASR), the talker on Pipecat's LLM service, and text-to-speech (TTS), in the order of the [Generic Assistant](../generic/README.md)'s pipeline. The repository UI bot runs the same pipeline.
- The **delegation core** ([`delegation_core/`](delegation_core/)), a package of its own with its own README, takes each delegation through the OpenAI Responses API. Its interaction manager (IM) owns the delegated work from then on.
- The **backend adapter** runs the work: the backend LLM, directly connected (the default); the backend LLM agent, a model conductor with LLM workers; or the harness, an external agent that you deploy, such as Hermes.

Compared to the [Frontend/Backend Agent](../frontend_backend_agent/README.md), the talker decides every turn, by default as a structured decision instead of a `call_backend` tool call, and words the backend's results for speech.

## Default Models

The `services` lists in [`examples_registry.yaml`](../../../examples_registry.yaml) resolve to the following default models for each profile. Service entries live in the root [`services.yaml`](../../../services.yaml):

| Profile | ASR | Talker LLM | Backend LLM | TTS |
| --- | --- | --- | --- | --- |
| Cloud | Nemotron ASR Streaming English | Nemotron 3.5 Lightning 30B A3B | Nemotron 3.5 Super VL 120B A12B with reasoning enabled | Magpie TTS Multilingual |
| Server | Nemotron ASR Streaming English NIM | Nemotron 3.5 Lightning 30B A3B NIM | Nemotron 3.5 Super VL 120B A12B NIM with reasoning enabled | Magpie TTS Multilingual NIM |
| Single GPU | Nemotron Speech Streaming English 0.6B through NeMo-Speech.cpp | Nemotron 3.5 Lightning 30B A3B through vLLM | Nemotron 3.5 Lightning 30B A3B through vLLM with reasoning enabled | Magpie TTS Multilingual through NeMo-Speech.cpp |

The talker's `llm` slot lists Nemotron 3.5 Lightning. The backend's `thinker-llm` slot lists Nemotron 3.5 Super VL (`nemotron-super`), then Lightning. A profile's default is the first listed entry whose service is up, as the repository UI picks it, and Live sessions pick it the same way. Live sessions read the catalog once, at the first use of the Live routes, so a model whose service starts later becomes their default at the next restart. The Cloud profile calls Super VL on the public NVIDIA API (`integrate.api.nvidia.com`) with `NVIDIA_API_KEY`, through the entry's `nvcf` variant. The Server profile starts the Super VL NIM on GPUs `1` and `2`, beside the Lightning NIM on GPU `0`. When the Super VL NIM is not running, the backend runs on the Lightning NIM. Single GPU does not start Super VL, so both roles run on its local Lightning, and Super VL on the public NVIDIA API follows it in the list when `NVIDIA_API_KEY` is set. Each slot's `settings` in `examples_registry.yaml` set its sampling. The talker runs at temperature 0, with reasoning off and at most 512 output tokens. The backend on Super VL runs at temperature 0, with reasoning on and at most 4,096 output tokens. On the public NVIDIA API, its settings profile adds a 1,024-token reasoning budget and medium reasoning effort, and the Super VL NIM's profile sets neither. On Nemotron 3.5 Lightning, the backend runs at temperature 0.7 with a 256-token reasoning budget.

## Running the Example

This example runs with the **Cloud**, **Server**, and **Single GPU** profiles. Server is workstation-only (not DGX Spark or Jetson Thor) and needs three GPUs: GPU `0` for the Lightning NIM, ASR, and TTS, and GPUs `1` and `2` for the Super VL NIM. Single GPU covers workstations, DGX Spark, and Jetson Thor. For prerequisites and hardware detail, refer to the [Getting Started guide](../../../docs/01-getting-started.md). Run every command from the repository root.

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
   | `frontend-backend-live/server` | `frontend-backend-live-server` | `nvidia-llm`, `nemotron-35-super-vl`, `nemotron-asr-streaming-english`, `magpie-multilingual-tts-service` |
   | `frontend-backend-live/single-gpu` | `frontend-backend-live-single-gpu` | `nvidia-llm-vllm-lightning-frontend-backend`, `nemo-speech` |

4. Open the UI at `https://localhost:7860/` and ask what teas the cafe has. Keep TLS enabled for browser testing, because browsers allow the microphone only on `localhost` or over HTTPS. `PIPELINE_TLS=false` serves plain HTTP. For plain-HTTP browser testing, refer to [browser access](../../../docs/06-troubleshooting.md#browser-access).

5. Clean up when you are done by tearing down with the same profile that you started, for example `docker compose --profile frontend-backend-live down`.

To run host-native without Docker, set `selection: frontend-backend-live` in [`examples_registry.yaml`](../../../examples_registry.yaml), then run `uv run python3 src/server.py`.

## Configuration

The following files configure the example:

| File | What it sets |
| --- | --- |
| [`examples_registry.yaml`](../../../examples_registry.yaml) | The `services.yaml` entries that each slot offers (`llm` for the talker, `thinker-llm` for the backend), and their sampling and reasoning settings. The `tts` settings make the Magpie entries speak per sentence, in the UI bot and in live sessions. The `live.settings` block sets the Live gateway's limits, as [Server Settings](../../../docs/how-to/use-live-gateway.md#server-settings) describes. The `config` block holds the example's own settings. |
| [`prompts.yaml`](prompts.yaml) | The talker prompt that the repository UI lists (`cafe`) and the other prompts and texts (`agent_prompts`), including the lines that the delegation core writes (`agent_prompts.Core`). Refer to [Prompts](#prompts). |

The `config` block of the `frontend-backend-live` entry must set `frontend.adapter` and its block, `frontend.cascade.slot`, `core.backend.slot`, `core.backend.prompt`, and `ui_bot.channel`. Every other key that it leaves out keeps its default. The settings models hold every default with a comment: [`config_manager/schema.py`](config_manager/schema.py) for `frontend`, [`delegation_core/config.py`](delegation_core/config.py) for `core.delegation`, and each backend adapter's `settings.py` for its block. The server reads the registry when it starts, so an edit takes effect at the next restart. An unknown key or an invalid value refuses every live session and the UI bot, and the server log names the key. A missing `prompts.yaml` entry that `frontend.cascade.prompt` or `ui_bot.channel` names, or an error in what `core.backend` names, in a backend adapter's block (`core.backend.<adapter>`), or in `ui_bot.core`, is found only when it is used. Such an error refuses only what uses it, such as Responses-mode sessions, `GET /v1/live/info`, and `POST /v1/live/delegate` for the Live core, or the UI bot for its own core. The block has three parts:

- `frontend`: the frontend adapter (`frontend.adapter`) and its block (`frontend.cascade`), which sets the talker's services slot, style, and prompt, the voice's delegation behavior, its memory, the talker's retries and its call timeout (`reliability`), the caller's turns (`speech`), and the reply audio's pre-roll.
- `core`: the delegation core of live sessions, with the IM's settings (`delegation`) and the backend adapter with its blocks (`backend`), as [Build and Configure the Core](delegation_core/README.md#build-and-configure-the-core) describes.
- `ui_bot`: the repository UI bot's channel context (`channel`) and its own core, which is `core` with `ui_bot.core` merged over it. A mapping merges key by key, and any other value replaces it, `null` included. `ui_bot.core` sets `backend.adapter: llm`, so the bot stays on the backend LLM whatever adapter live sessions use.

The following excerpt shows the frontend keys that you are most likely to change, with their shipped values:

```yaml
config:
  frontend:
    adapter: cascade
    cascade:
      slot: llm                          # the services slot of the talker's model
      talker_style:
        nemotron-lightning: json         # the talker's style by its model's services key: json | tools
      prompt: generic                    # the talker prompt (agent_prompts.Talker) of a session that sets no instructions
      delegation:
        app_wait_seconds: 50             # the app-wait line when the app's outputs take this long; null: none
        progress_after_seconds: 8        # a progress line after this long of silent work; null: none
        progress_max_per_delegation: 3   # progress lines offered for one delegation
      memory:
        context_tokens: 16000            # the talker's memory budget
      speech:
        interrupt_min_words: 2           # the words that start the caller's turn while the bot speaks
      tts_preroll_ms: 300                # reply audio held before playback starts; 0: off
```

A model endpoint uses `NVIDIA_API_KEY` unless its `services.yaml` entry sets `api_key_env` to another variable. The server reads `examples_registry.yaml` from the repository root, and Compose mounts that file into the app container, so a run that needs other values, such as a benchmark, sets them in this block of its own checkout.

### Choose a Backend Adapter

`core.backend.adapter` selects the backend adapter. The following table compares the three adapters:

| | `llm` (default) | `llm-agent` | `harness` |
| --- | --- | --- | --- |
| What runs the work | The backend LLM, directly connected | Backend LLM workers | An Agent Client Protocol (ACP) agent that you deploy, such as Hermes |
| Tasks at once | 1 | `max_running` (3) | `max_running` (3), at most the agent's turns per process (4 for Hermes) |
| What places each request | A rule, with no model call: the request joins the open work, or starts it | A model conductor, with one model call per decision. It can also answer a status question. | A conductor session of the agent, with one prompt per decision. It can also answer a status question. |
| Model | The request's model, which the catalog must list | The request's model for the workers, and `conductor.model`, or the default, for the conductor | The model that the agent's operator configures |
| What you deploy | Nothing more | Nothing more | The agent, as [Connect an Agent Harness](../../../docs/how-to/connect-an-agent-harness.md) describes |

`llm` adds no model call before work starts. With `llm-agent` or `harness`, `max_running` in the adapter's block sets how many tasks run at once. Above 1, the conductor places every request, the first one included, so a new request waits for a conductor decision, at most the conductor's 20-second timeout. At 1, no conductor runs: each request joins the open work, or starts it when none is open, as with `llm`. For how these rules differ from OpenAI's GPT-Live API, refer to [Server Delegation](../../../docs/how-to/use-live-gateway.md#server-delegation).

To switch adapters, set `core.backend.adapter` and restart the app service, for example `docker compose restart frontend-backend-live`. With `llm-agent` or `harness`, also set `core.delegation.result_format` to `speak_display`. Every worker's instructions then carry the envelope: a `show` item for the screen, then the answer to speak.

### Talker Style

`frontend.cascade.talker_style` sets the talker's style for each model, by the services key of the model's `services.yaml` entry, such as `nemotron-lightning`. A model that it does not list uses `json`. It takes the following values:

- `json`, the default, asks for a structured `{action, speech}` decision on every turn, in a strict JSON schema unless the model's `services.yaml` entry sets `response_format: none`.
- `tools` makes the routing decision with Pipecat's function calling, as the Frontend/Backend Agent's talker calls `call_backend`. Plain text is a spoken reply, a `delegate` call hands the turn over with its lead-in, and a `wait` call lets the turn pass. Relays and progress lines keep the JSON decision. The endpoint must stream tool calls with `tool_choice: "auto"`. For a self-hosted vLLM, refer to [Reasoning Parser and Tool Calling](../../../docs/how-to/configure-llm.md#reasoning-parser--tool-calling-self-hosted).

The shipped block sets `json` for Nemotron 3.5 Lightning. The `tools` style is experimental: the routing prompt's examples are JSON decisions, and Nemotron 3.5 Lightning can copy them as text in the `tools` style, which the voice then says aloud. Use it with a routing prompt without those examples, and test a model's routing decisions on your own calls before you set it to `tools`.

### Turns and Interruptions

`frontend.cascade.speech` sets the caller's turns. While the bot is silent, the caller's first sound, `vad_start_seconds` (0.12 seconds) of speech, starts their turn, as in the other examples. The bot then starts no new line until the turn ends: a ready result waits, a lead-in that is not said yet is dropped, and a decision under way is cut. While the bot speaks, the caller's turn starts, and cuts the bot off, after `interrupt_min_words` words (2), as Pipecat's `MinWordsUserTurnStartStrategy` counts them. A cough or a one-word backchannel, such as "okay," therefore does not cut the bot off. Smart Turn judges each pause of `vad_stop_seconds` (0.2 seconds), and a pause of `turn_silence_seconds` (1.2 seconds) ends any turn.

When the caller cuts a line off, the talker's context keeps the line as far as it played, followed by the `[cut off]` mark, and the talker decides what to say next. The voice never says the rest of a cut line word for word. A turn with no recognized words, such as a cough, counts as interrupting the bot when the bot is not waiting for the caller (Pipecat's rule): before its first line, after a decision to wait, or while the talker decides. Such a turn runs the talker once, through Pipecat's `EmptyUserTurnConfig`, with the example's own empty-turn note (`agent_prompts.Cascade.empty_turn_note`) in place of Pipecat's default note, so the talker continues or asks the caller to repeat. A cough while the talker decides cuts that decision, so the run with the note decides again on the caller's same words. After the bot has finished a line, a turn with no words gets no answer. Only one such turn in a row gets an answer. A line that the caller did not hear at all gives back its results, and the talker words them again.

### Prompts

[`prompts.yaml`](prompts.yaml) holds the example's prompts and texts. Its top-level `cafe` entry is the Bluebird Cafe talker prompt, which the repository UI lists as the default (`defaults.prompt` in `examples_registry.yaml`). The **Prompts** tab shows the other texts read-only, in the following `agent_prompts` groups:

| Group | What it holds |
| --- | --- |
| `Talker` | The talker prompt of a session that sets no instructions or blank ones, by the name that `frontend.cascade.prompt` gives. `generic` names no domain. |
| `Backend` | The backend prompt of a request that sets no instructions, by the name that `core.backend.prompt` gives: `generic` for live sessions, and `cafe` for the UI bot |
| `Channel` | The context that the UI bot adds after its talker and backend prompts, by the name that `ui_bot.channel` gives |
| `Cascade` | The text of the talker's calls: the decision rules (`json_rules`, or `tool_rules` with the `tools.delegate` and `tools.wait` schemas for the `tools` style), the writer's frame, the labels, the cut mark, the empty-turn note, and each delegation's input |
| `Lines` | The lines that the voice says word for word, such as `apology`, `holding`, and `app_wait_line`, and `intro`, from which the talker words the UI bot's greeting |
| `Core` | The delegation core's texts, such as the notes that the core adds to the backend's instructions and the conductor's prompt, each named by its place in the core's settings, for example `texts.conductor_prompt` |

Each session reads the file when it starts, so an edit takes effect at the next session. The Live core reads its default backend prompt when it is built, and the server reads the core's texts with the core's settings, once, so restart the server after you edit either of them. A prompt file that you start the server with (`PROMPT_FILE_PATH`) replaces the whole file, so it must also hold every `agent_prompts` entry.

## Customization

To change the prompts, edit [`prompts.yaml`](prompts.yaml) as [Prompts](#prompts) describes, or send your own prompts in each session's `instructions` and `delegation.responses.instructions`. A tool source runs tools on the server for the backend, as the cafe tools do under `ui_bot.core.backend.tools`. For models, services, and prompts in general, refer to [Configure LLM](../../../docs/how-to/configure-llm.md), [Configure Services](../../../docs/how-to/configure-services.md), and [Configure Prompts](../../../docs/how-to/configure-prompts.md).

Each part that you can add is one class and one entry in a name table, which the settings choose by name. For the delegation core's parts, such as a backend adapter, a model provider, or a tool source, refer to [Extend the Core](delegation_core/README.md#extend-the-core). The following table lists where to add each frontend part. The module of each table describes what its classes implement:

| Part | Table | Chosen by |
| --- | --- | --- |
| Frontend adapter | `FRONTEND_ADAPTERS` in [`adapter/registry.py`](adapter/registry.py), with the class contract in [`adapter/contract.py`](adapter/contract.py) | `frontend.adapter` |
| Talker service | `LLM_SERVICES` in [`talker.py`](talker.py) | `frontend.cascade.talker_style` |

## Troubleshooting

For a Live route that answers 404, a refused session, `responses_validation_failed`, or a harness that does not connect, refer to [Frontend/Backend Live](../../../docs/06-troubleshooting.md#frontendbackend-live) in the Troubleshooting guide. For a line or a sentence that gets no audio from the TTS service, in the UI bot or in a live session, refer to [Errors and Limits](../../../docs/how-to/use-live-gateway.md#errors-and-limits). For deployment and general failure modes, refer to the [Troubleshooting guide](../../../docs/06-troubleshooting.md).

For the UI bot and for live sessions, the server log has a `User-to-bot latency` line and a `Latency breakdown` line for each turn. Both measure from the moment the caller stops speaking to the bot's first audio. To record each turn's audio, refer to [Enable the Audio Recorder](../../../docs/how-to/enable-audio-recorder.md).

## Tips and Best Practices

Follow these practices:

- **Repeat the talker's limits in the backend's instructions.** The talker words each result with its instructions, so it keeps any limits that they set on what the caller may be told. A fallback of that wording says the backend's text as written, with its Markdown stripped for speech. The fallback follows a wording that fails, times out (`reliability.talker_attempt_timeout_seconds`), or returns unusable output, no words, the update unchanged, or a line that the voice already said. So also put those limits in the backend's instructions: `delegation.responses.instructions`, or your own backend's prompt in client mode.
- **Keep the app-wait line before the deadline.** When a delegation still waits for the app's function outputs after `frontend.cascade.delegation.app_wait_seconds` (50), the voice says the `app_wait_line` from `agent_prompts.Lines` in [`prompts.yaml`](prompts.yaml) once. Keep this value below the core's deadline for those outputs, `core.delegation.client_tool_timeout_seconds` (60), so the line plays while the work still waits.
- **Change the UI bot's backend with its talker prompt.** A new top-level talker prompt runs with the cafe's channel context, backend prompt, and tools, and the greeting lines (`intro` and `greeting`) stay the cafe's. For another domain, also change `backend.prompt` and `backend.tools` under `ui_bot.core`, `ui_bot.channel`, and the greeting lines in `agent_prompts.Lines`.
