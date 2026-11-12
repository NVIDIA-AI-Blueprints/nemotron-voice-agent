# Frontend/Backend Live Cascaded Example

This example is a cascaded voice agent that puts a routing talker in front of backend work. On every caller turn, the talker, a fast large language model (LLM) call, speaks a reply, delegates the turn, or waits. The delegation core runs the delegated work, and the talker words each result for the voice to say when the caller is idle. The reference domain is the Bluebird Cafe ordering assistant, whose backend looks up the menu, edits a cart, and places an order.

The example has two ways in:

- **Live session clients**, such as Pipecat apps or your own app, connect through `/v1/live/sessions` and supply the prompts. The Compose profiles serve the Live routes when `.env` sets `LIVE_API_TOKEN`, or `LIVE_ENABLED=true` to serve them without a token. For the routes and their settings, refer to [Use the Live Gateway](../../../docs/how-to/use-live-gateway.md).
- **The repository UI** (`/`) runs a bot whose backend runs the cafe tools on the server and shows each delegation as a task card. The Realtime gateway (`WS /v1/realtime`) drives the same bot.

A live session runs through four parts: the [Live gateway](../../gateways/live/), the frontend adapter (the rest of this package), the [delegation core](delegation_core/), and the backend adapter. The frontend adapter is a Pipecat pipeline of automatic speech recognition (ASR), the talker, and text-to-speech (TTS). The following diagram shows the four parts:

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

## Default Models

The `services` lists in [`examples_registry.yaml`](../../../examples_registry.yaml) resolve to the following default models for each profile. Service entries live in the root [`services.yaml`](../../../services.yaml):

| Profile | ASR | Talker LLM | Backend LLM | TTS |
| --- | --- | --- | --- | --- |
| Cloud | Nemotron ASR Streaming English | Nemotron 3.5 Lightning 30B A3B | Nemotron 3.5 Super VL 120B A12B with reasoning enabled | Magpie TTS Multilingual |
| Server | Nemotron ASR Streaming English NIM | Nemotron 3.5 Lightning 30B A3B NIM | Nemotron 3.5 Super VL 120B A12B NIM with reasoning enabled | Magpie TTS Multilingual NIM |
| Single GPU | Nemotron Speech Streaming English 0.6B through NeMo-Speech.cpp | Nemotron 3.5 Lightning 30B A3B through vLLM | Nemotron 3.5 Lightning 30B A3B through vLLM with reasoning enabled | Magpie TTS Multilingual through NeMo-Speech.cpp |

The talker uses the `llm` slot, and the backend uses the `thinker-llm` slot, which lists Super VL (`nemotron-super`) and then Lightning. Each slot's default is its first listed entry whose service is up. Live sessions read the catalog once, so a model that starts later becomes their default at the next restart. On Server, the backend runs on the Lightning NIM while the Super VL NIM is not running. With `NVIDIA_API_KEY` set, Single GPU lists Super VL on the public NVIDIA API after its local Lightning.

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
   | `frontend-backend-live/single-gpu` | `frontend-backend-live-single-gpu` | `nvidia-llm-vllm-lightning-frontend-backend`, `nemo-speech-frontend-backend` (network alias `nemo-speech`) |

4. Open the UI at `https://localhost:7860/` and ask what teas the cafe has. Keep TLS enabled for browser testing, because browsers allow the microphone only over HTTPS or on `localhost`. For plain-HTTP browser testing, refer to [browser access](../../../docs/06-troubleshooting.md#browser-access).

5. Clean up when you are done by tearing down with the same profile that you started, for example `docker compose --profile frontend-backend-live down`.

To run host-native without Docker, set `selection: frontend-backend-live` in [`examples_registry.yaml`](../../../examples_registry.yaml), then run `uv run python3 src/server.py`.

## Configuration

The example's settings are the `config` block of its entry in [`examples_registry.yaml`](../../../examples_registry.yaml), which must set `frontend.adapter` and its block, `frontend.cascade.slot`, `core.backend.slot`, `core.backend.prompt`, and `ui_bot.channel`. Every other key keeps its default, which the settings models document: [`config_manager/schema.py`](config_manager/schema.py) for `frontend`, [`delegation_core/config.py`](delegation_core/config.py) for `core.delegation`, and each backend adapter's `settings.py` for its block. The server reads the block once, so restart it after an edit. An unknown key or an invalid value refuses the sessions and routes that read it, and the server log names the key. The block has three parts:

- `frontend`: the frontend adapter (`frontend.adapter`) and its block (`frontend.cascade`), which sets the talker, the voice, how the backend's answers land (`delegation.answers`), and the caller's turns.
- `core`: the delegation core of live sessions, as [Build and Configure the Core](delegation_core/README.md#build-and-configure-the-core) describes.
- `ui_bot`: the UI bot's channel context (`channel`) and its own core: `core` with `ui_bot.core` merged over it. A mapping merges key by key, while any other value, `null` included, replaces it. The shipped `ui_bot.core` keeps the bot on the `llm` adapter.

The following excerpt shows the frontend keys that you are most likely to change, with the values that the example runs with:

```yaml
config:
  frontend:
    adapter: cascade
    cascade:
      slot: llm                          # the services slot of the talker's model
      talker_style:
        nemotron-lightning: json         # by the model's services key: json | tools
      prompt: generic                    # the Talker prompt of a session with no instructions
      delegation:
        app_wait_seconds: 50             # the app-wait line after the app takes this long; null: none
        progress_after_seconds: 8        # a progress line after this much silent work; null: none
        progress_max_per_delegation: 3   # progress lines for one delegation
        result_wording: writer           # writer | as_written (experimental): results said as the backend wrote them
        spelled_notes: true              # a [spelled: …] note on caller lines in the backend's transcript
        answers:
          max_tries: 3                   # misses before an unheard backend answer is dropped; null: no limit
      memory:
        context_tokens: 16000            # the talker's memory budget
      speech:
        interrupt_min_words: 2           # words that start the caller's turn while the bot speaks
        spelling_hold: true              # a turn that ends in a spelling ends only at turn_silence_seconds
      tts_preroll_ms: 300                # reply audio held before playback; 0: off
```

The `delegation.answers` block also holds five switches: `requeued_as_said`, `unheard_rest_once`, `lead_in_skip`, and `unheard_questions` are on by default. `clear_ended_on_cut` is off in the code, and the shipped `examples_registry.yaml` turns it on: a cut also settles a line still waiting for its words after a sentence with no audio. [Turns and Interruptions](#turns-and-interruptions) describes what they do, and [`config_manager/schema.py`](config_manager/schema.py) comments each one. `delegation.keep_questions` (`true`) says the backend's text as written when the talker's wording of a result no longer ends with the result's last question. `delegation.result_wording` is experimental: `writer`, the default, has the talker word each result, and `as_written` says results as the backend wrote them, with no talker call. `delegation.spelled_notes` (`true`) ends a caller line that holds single letters or digits in a row with a note that joins them, such as `[spelled: B4071]`, in the backend's transcript only.

A model endpoint uses `NVIDIA_API_KEY` unless its `services.yaml` entry sets `api_key_env` to another variable. The server reads `examples_registry.yaml` from the repository root, and Compose mounts that file into the app container.

The delegation core calls a backend model over chat completions unless the model's `services.yaml` entry sets `api: openai-responses`. Set it only for a server that serves the Responses API fully: one that honors `parallel_tool_calls: false` and ends a reply cut off at `max_output_tokens` with `response.incomplete`. The server must also stream each round's events as it writes them, or else end each round within the core's round time limits. The core waits 45 seconds for the first event and 30 seconds between events: the `first_event_timeout_seconds` and `idle_timeout_seconds` keys of `core.backend.llm`, or of `core.backend.llm-agent.worker` for the `llm-agent` adapter. The core does not support self-hosted NIM and vLLM servers on the Responses API. To set it for the NVIDIA Cloud variant of an entry only, put it in the entry's `nvcf` block, as [Configure Services](../../../docs/how-to/configure-services.md#adding-an-nvidia-cloud-service) describes. The Responses API has no field for the **Reasoning Budget** or the **Seed** of the model's settings profile. Keep a model whose settings turn **Reasoning** on with a budget, such as this example's Super VL backend on NVIDIA Cloud, on chat completions.

### Choose a Backend Adapter

`core.backend.adapter` selects the backend adapter. The following table compares the three adapters:

| | `llm` (default) | `llm-agent` | `harness` |
| --- | --- | --- | --- |
| What runs the work | The backend LLM, directly connected | Backend LLM workers | An Agent Client Protocol (ACP) agent that you deploy, such as Hermes, on its own model |
| Tasks at once | 1 | `max_running` (3) | `max_running` (3), at most 4 for Hermes |
| What places each request | A rule, with no model call | A model conductor, with one call per decision | A conductor session of the agent, with one prompt per decision |
| What you deploy | Nothing more | Nothing more | The agent, as [Connect an Agent Harness](../../../docs/how-to/connect-an-agent-harness.md) describes |

The rule joins each request to the open work, or starts the work when none is open. A conductor runs only while `max_running` is above 1. It places every request, so a new request waits up to its 20-second timeout for a decision, and it can answer a status question. For how this differs from OpenAI's GPT-Live API, refer to [How This Differs from GPT-Live](../../../docs/how-to/use-live-gateway.md#how-this-differs-from-gpt-live).

To switch adapters, set `core.backend.adapter`, and for `llm-agent` or `harness` also `core.delegation.result_format: speak_display`, which asks every worker for a `show` item for the screen before the answer. Then restart the app service, for example `docker compose restart frontend-backend-live`.

### Talker Style

`frontend.cascade.talker_style` sets the talker's style for each model by its services key, such as `nemotron-lightning`. A model that it does not list uses `json`:

- `json` asks for a structured `{action, speech}` decision on every turn, in a strict JSON schema unless the model's `services.yaml` entry sets `response_format: none`.
- `tools` makes the routing decision with Pipecat's function calling: the streamed text is the spoken reply, and a `delegate` or `wait` call routes the turn. Relays and progress lines keep the JSON decision. The endpoint must stream tool calls with `tool_choice: "auto"`, as [Reasoning Parser and Tool Calling](../../../docs/how-to/configure-llm.md#reasoning-parser--tool-calling-self-hosted) describes for a self-hosted vLLM.

The `tools` style is experimental, so test a model's routing decisions on your own calls first. To move a model to it, list the model under `services.llm`, set its style, for example `nemotron-super: tools`, and restart the server. Live sessions use the first listed model whose service is up, and the UI bot uses the model that you select in the **Services** tab.

### Turns and Interruptions

`frontend.cascade.speech` sets the caller's turns. While the bot is silent, `vad_start_seconds` (0.12 seconds) of the caller's speech starts their turn. While the bot speaks, the caller's turn starts, and cuts the bot off, after `interrupt_min_words` words (2), so a cough or a one-word backchannel does not. Smart Turn judges each pause of `vad_stop_seconds` (0.2 seconds), and a pause of `turn_silence_seconds` (1.2 seconds) ends any turn. With `spelling_hold` (`true`), a turn that ends in single letters or digits in a row, such as a spelled name or code, ends only at that `turn_silence_seconds` pause.

When the caller cuts off a backend answer, the voice says its unheard sentences once, as written, if the talker waits on the caller's turn; a request, an instruction, or a newer result drops them. A line that the caller did not hear at all is said again with the same words, unless something newer happened since or it came out silent. After `answers.max_tries` (3) misses, the voice drops a backend answer and the next delegation tells the backend that the caller never heard it. The next delegation's input also names a cut answer's questions that did not play.

### Prompts

[`prompts.yaml`](prompts.yaml) holds the talker prompt that the repository UI lists (`cafe`) and, in `agent_prompts`, the other texts, which the **Prompts** tab shows read-only. `Talker` and `Backend` hold the default talker and backend prompts, and `Channel` holds the UI bot's channel context. `Cascade` holds the text of the talker's calls, `Lines` holds the voice's fixed lines, and `Core` holds the delegation core's texts.

Each session reads the file when it starts, but the server reads the core's texts and the Live core's default backend prompt once, so restart it after you edit them. A file that `PROMPT_FILE_PATH` names replaces the whole file, so it must hold every `agent_prompts` entry. It can leave out the `tools` style's entries in `Cascade` (`tool_frame_before`, `tool_routing`, `tool_rules`, `tools.delegate`, and `tools.wait`) when no model uses that style. It can also leave out the texts added later: `transcript.not_said`, `transcript.now_heard`, `transcript.cut_question`, and `transcript.spelled` in `Cascade`, and `texts.questions_block` in `Core`. The feature that needs a missing text is off, and the server log has a `prompts.missing_text` warning that names the key. An unknown entry in `Cascade` or `Lines` refuses the live and UI bot sessions, and the server log names the entry or its group.

## Customization

To change the prompts, edit [`prompts.yaml`](prompts.yaml), or send them in each session's `instructions` and `delegation.responses.instructions`. For models, services, and prompts, refer to [Configure LLM](../../../docs/how-to/configure-llm.md), [Configure Services](../../../docs/how-to/configure-services.md), and [Configure Prompts](../../../docs/how-to/configure-prompts.md).

To add a part, add one class and one entry in its name table, which the settings, or a model's `services.yaml` entry, choose by name. The name tables are `FRONTEND_ADAPTERS` ([`adapter/registry.py`](adapter/registry.py)) for a frontend adapter, `LLM_SERVICES` ([`talker.py`](talker.py)) for a talker service, and the core's tables in [Extend the Core](delegation_core/README.md#extend-the-core).

## Troubleshooting

For a Live route that answers 404, a refused session, or a harness that does not connect, refer to [Frontend/Backend Live](../../../docs/06-troubleshooting.md#frontendbackend-live) in the Troubleshooting guide. For a line that gets no audio, refer to [Errors and Limits](../../../docs/how-to/use-live-gateway.md#errors-and-limits), and for general failure modes, to the [Troubleshooting guide](../../../docs/06-troubleshooting.md).

## Tips and Best Practices

Follow these practices:

- **Repeat the talker's limits in the backend's instructions.** When the talker cannot word a result, or, with `delegation.keep_questions`, its wording drops the result's last question, the voice says the backend's text as written. So also put any limits on what the caller may hear in `delegation.responses.instructions`, or in your backend's prompt in client mode.
- **Keep the app-wait line before the deadline.** `core.delegation.client_tool_timeout_seconds` (60) is the core's deadline for the app's function outputs. Keep `frontend.cascade.delegation.app_wait_seconds` (50) below it, so the `app_wait_line` plays while the work still waits.
- **Change the UI bot's backend with its talker prompt.** A new top-level talker prompt keeps the cafe's channel context, backend prompt, tools, and greeting lines. So also change `backend.prompt` and `backend.tools` under `ui_bot.core`, `ui_bot.channel`, and the `intro` and `greeting` lines in `agent_prompts.Lines`.
