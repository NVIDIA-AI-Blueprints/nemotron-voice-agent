# Frontend/Backend Live Cascaded Example

This example is a cascaded voice agent that puts a routing talker in front of backend work. On every caller turn, the talker, a fast large language model (LLM) call, speaks a reply, delegates the turn, or waits. The delegation core runs the delegated work, and the talker words each result for the voice to say when the caller is idle. The repository UI runs the Demo Air travel desk, a fictional airline whose backend searches flights, checks flight status, and books, changes, and cancels demo bookings. Refer to [Demo Air Travel Desk](#demo-air-travel-desk). The same pipeline can run the Bluebird Cafe ordering assistant, whose backend looks up the menu, edits a cart, and places an order, by a change to the registry entry.

The example has two ways in:

- **Live session clients**, such as Pipecat apps or your own app, connect through `/v1/live/sessions` and supply the prompts. The Compose profiles serve the Live routes when `.env` sets `LIVE_API_TOKEN`, or `LIVE_ENABLED=true` to serve them without a token. For the routes and their settings, refer to [Use the Live Gateway](../../../../../docs/how-to/use-live-gateway.md).
- **The repository UI** (`/`) runs a bot whose backend runs the airline tools on the server and shows each delegation as a task card. The Realtime gateway (`WS /v1/realtime`) drives the same bot.
  With the `event_log` capability, the bot relays its backend's OpenAI-format events to the browser as `live-event` messages, and the developer view's **EVENT LOG** tab prints them. Refer to [Event Relay](#event-relay) and the [Event log](../../../../../clients/pipecat/README.md#event-log) section of the client README.

A live session runs through four parts: the [Live gateway](../../runtime/gateways/live/), the frontend adapter (the rest of this package), the [delegation core](delegation_core/), and the backend adapter. The frontend adapter is a Pipecat pipeline of automatic speech recognition (ASR), the talker, and text-to-speech (TTS). The following diagram shows the four parts:

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

The `services` lists under `pipecat.examples` in [`examples_registry.yaml`](../../../../../examples_registry.yaml) resolve to the following default models for each profile. Service entries live in the root [`services.yaml`](../../../../../services.yaml):

| Profile | ASR | Talker LLM | Backend LLM | TTS |
| --- | --- | --- | --- | --- |
| Cloud | Nemotron ASR Streaming English | Nemotron 3.5 Lightning 30B A3B | Nemotron 3.5 Super VL 120B A12B with reasoning enabled | Magpie TTS Multilingual |
| Server | Nemotron ASR Streaming English NIM | Nemotron 3.5 Lightning 30B A3B NIM | Nemotron 3.5 Super VL 120B A12B NIM with reasoning enabled | Magpie TTS Multilingual NIM |
| Single GPU | Nemotron Speech Streaming English 0.6B through NeMo-Speech.cpp | Nemotron 3.5 Lightning 30B A3B through vLLM | Nemotron 3.5 Lightning 30B A3B through vLLM with reasoning enabled | Magpie TTS Multilingual through NeMo-Speech.cpp |

The talker uses the `llm` slot, and the backend uses the `thinker-llm` slot, which lists Super VL (`nemotron-super`) and then Lightning. Each slot's default is its first listed entry whose service is up. Live sessions read the catalog once, so a model that starts later becomes their default at the next restart. On Server, the backend runs on the Lightning NIM while the Super VL NIM is not running. With `NVIDIA_API_KEY` set, Single GPU lists Super VL on the public NVIDIA API after its local Lightning.

## Running the Example

This example runs with the **Cloud**, **Server**, and **Single GPU** profiles. Server is workstation-only (not DGX Spark or Jetson Thor) and needs three GPUs: GPU `0` for the Lightning NIM, ASR, and TTS, and GPUs `1` and `2` for the Super VL NIM. Single GPU covers workstations, DGX Spark, and Jetson Thor. For prerequisites and hardware detail, refer to the [Getting Started guide](../../../../../docs/01-getting-started.md). Run every command from the repository root.

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

4. Open the UI at `https://localhost:7860/` and ask what teas the cafe has. Keep TLS enabled for browser testing, because browsers allow the microphone only over HTTPS or on `localhost`. For plain-HTTP browser testing, refer to [browser access](../../../../../docs/06-troubleshooting.md#browser-access).

5. Clean up when you are done by tearing down with the same profile that you started, for example `docker compose --profile frontend-backend-live down`.

To run host-native without Docker, set `pipecat.selection` to `frontend-backend-live` in [`examples_registry.yaml`](../../../../../examples_registry.yaml), then run `uv run nva-pipecat`.

## Configuration

The example's settings are the `config` block of its `pipecat.examples.frontend-backend-live` entry in [`examples_registry.yaml`](../../../../../examples_registry.yaml), which must set `frontend.adapter` and its block, `frontend.cascade.slot`, `core.backend.slot`, `core.backend.prompt`, and `ui_bot.channel`. Every other key keeps its default, which the settings models document: [`config_manager/schema.py`](config_manager/schema.py) for `frontend`, [`delegation_core/config.py`](delegation_core/config.py) for `core.delegation`, and each backend adapter's `settings.py` for its block. The server reads the block once, so restart it after an edit. An unknown key or an invalid value refuses the sessions and routes that read it, and the server log names the key. The block has three parts:

- `frontend`: the frontend adapter (`frontend.adapter`) and its block (`frontend.cascade`), which sets the talker, the voice, and the caller's turns.
- `core`: the delegation core of live sessions, as [Build and Configure the Core](delegation_core/README.md#build-and-configure-the-core) describes.
- `ui_bot`: the UI bot's channel context (`channel`) and its own core: `core` with `ui_bot.core` merged over it. A mapping merges key by key, while any other value, `null` included, replaces it. The shipped `ui_bot.core` keeps the bot on the `llm` adapter. The shipped entry sets the channel, the backend prompt, and the server tools to the airline desk. Refer to [Demo Air Travel Desk](#demo-air-travel-desk).

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
        progress_after_seconds: 8        # a progress line after this much silent work (the fixed Lines.still_working once every tool call ended); null: none
        progress_max_per_delegation: 3   # progress lines for one delegation
      memory:
        context_tokens: 16000            # the talker's memory budget
      speech:
        interrupt_min_words: 2           # words that start the caller's turn while the bot speaks
      tts_preroll_ms: 300                # reply audio held before playback; 0: off
```

A model endpoint uses `NVIDIA_API_KEY` unless its `services.yaml` entry sets `api_key_env` to another variable. The server reads `examples_registry.yaml` from the repository root, and Compose mounts that file into the app container.

The delegation core calls a backend model over chat completions unless the model's `services.yaml` entry sets `api: openai-responses`. Set it only for a server that serves the Responses API fully: one that honors `parallel_tool_calls: false` and ends a reply cut off at `max_output_tokens` with `response.incomplete`. The server must also stream each round's events as it writes them, or else end each round within the core's round time limits. The core waits 45 seconds for the first event and 30 seconds between events: the `first_event_timeout_seconds` and `idle_timeout_seconds` keys of `core.backend.llm`, or of `core.backend.llm-agent.worker` for the `llm-agent` adapter. The core does not support self-hosted NIM and vLLM servers on the Responses API. To set it for the NVIDIA Cloud variant of an entry only, put it in the entry's `nvcf` block, as [Configure Services](../../../../../docs/how-to/configure-services.md#adding-an-nvidia-cloud-service) describes. The Responses API has no field for the **Reasoning Budget** or the **Seed** of the model's settings profile. Keep a model whose settings turn **Reasoning** on with a budget, such as this example's Super VL backend on NVIDIA Cloud, on chat completions.

### Choose a Backend Adapter

`core.backend.adapter` selects the backend adapter. The following table compares the three adapters:

| | `llm` (default) | `llm-agent` | `harness` |
| --- | --- | --- | --- |
| What runs the work | The backend LLM, directly connected | Backend LLM workers | An Agent Client Protocol (ACP) agent that you deploy, such as Hermes, on its own model |
| Tasks at once | 1 | `max_running` (3) | `max_running` (3), at most 4 for Hermes |
| What places each request | A rule, with no model call | A model conductor, with one call per decision | A conductor session of the agent, with one prompt per decision |
| What you deploy | Nothing more | Nothing more | The agent, as [Connect an Agent Harness](../../../../../docs/how-to/connect-an-agent-harness.md) describes |

The rule joins each request to the open work, else continues the work that ended last, else starts new work. A conductor runs only while `max_running` is above 1. It places every request, so a new request waits for the conductor's decision. That wait lasts up to 20 seconds with `llm-agent`, and with `harness` until the conductor's agent is free and has decided. The conductor can also answer a status question. For how this differs from OpenAI's GPT-Live API, refer to [How This Differs from GPT-Live](../../../../../docs/how-to/use-live-gateway.md#how-this-differs-from-gpt-live).

To switch adapters, set `core.backend.adapter`, and for `llm-agent` or `harness` also `core.delegation.result_format: speak_display`, which asks every worker for a `show` item for the screen before the answer. Then restart the app service, for example `docker compose restart frontend-backend-live`.

### Talker Style

`frontend.cascade.talker_style` sets the talker's style for each model by its services key, such as `nemotron-lightning`. A model that it does not list uses `json`:

- `json` asks for a structured `{action, speech}` decision on every turn, in a strict JSON schema unless the model's `services.yaml` entry sets `response_format: none`.
- `tools` makes the routing decision with Pipecat's function calling: the streamed text is the spoken reply, and a `delegate` or `wait` call routes the turn. Relays and progress lines keep the JSON decision. The endpoint must stream tool calls with `tool_choice: "auto"`, as [Reasoning Parser and Tool Calling](../../../../../docs/how-to/configure-llm.md#reasoning-parser--tool-calling-self-hosted) describes for a self-hosted vLLM.

The `tools` style is experimental, so test a model's routing decisions on your own calls first. To move a model to it, list the model under `services.llm`, set its style, for example `nemotron-super: tools`, and restart the server. Live sessions use the first listed model whose service is up, and the UI bot uses the model that you select in the **Services** tab.

### Turns and Interruptions

`frontend.cascade.speech` sets the caller's turns. While the bot is silent, `vad_start_seconds` (0.12 seconds) of the caller's speech starts their turn. While the bot speaks, the caller's turn starts, and cuts the bot off, after `interrupt_min_words` words (2), so a cough or a one-word backchannel does not. Smart Turn judges each pause of `vad_stop_seconds` (0.2 seconds), and a pause of `turn_silence_seconds` (1.2 seconds) ends any turn.

### Prompts

[`prompts.yaml`](prompts.yaml) holds the talker prompts `airline` and `cafe`, of which the repository UI lists the one that the registry's `agent_prompt_keys` does not hide, and, in `agent_prompts`, the other texts, which the **Prompts** tab shows read-only. `Talker` and `Backend` hold the default talker and backend prompts, and `Channel` holds the UI bot's channel context. `Cascade` holds the text of the talker's calls, `Lines` holds the voice's fixed lines, and `Core` holds the delegation core's texts.

Each session reads the file when it starts, but the server reads the core's texts and the Live core's default backend prompt once, so restart it after you edit them. A file that `PROMPT_FILE_PATH` names replaces the whole file, so it must hold every `agent_prompts` entry. It can leave out the `tools` style's entries in `Cascade` (`tool_frame_before`, `tool_routing`, `tool_rules`, `tools.delegate`, and `tools.wait`) when no model uses that style. An unknown entry in `Cascade` or `Lines` refuses the live and UI bot sessions, and the server log names the entry or its group.

## Demo Air Travel Desk

The repository UI runs the example as a travel desk. Ava, the desk's assistant at Demo Air (code `DA`), finds flights, checks flight status and on-time records, and looks up, books, changes, upgrades, and cancels demo bookings. Demo Air is a made-up airline. The data is a deterministic sample, and nothing is booked or charged. Bookings live in the session's memory, so nothing persists after the call ends.

The airline's code is in [`delegation_core/backend/tools/airline/`](delegation_core/backend/tools/airline/). It makes no network call and uses no paid or licensed data. The airport facts are public geographic data as published by OurAirports (public domain), and the time zones are IANA zones.

### Network and Flight Numbers

Demo Air flies a nonstop route between every pair of 12 U.S. cities:

| City | Code | City | Code |
| --- | --- | --- | --- |
| Atlanta | ATL | Los Angeles | LAX |
| Boston | BOS | Miami | MIA |
| Chicago | ORD | New York | JFK |
| Dallas | DFW | Phoenix | PHX |
| Denver | DEN | San Francisco | SFO |
| Las Vegas | LAS | Seattle | SEA |

The schedule has 528 daily flights: 3 to 5 a day each way for each pair, set by airport size. Callers can book dates up to 60 days ahead. The desk's day is the date in Central time, the zone of its Chicago hub. The flight numbers run from DA 100 to DA 755:

- Each pair of cities has its own block of 10 numbers, with the pairs ordered by airport code.
- A flight toward the east has an even number, and a flight toward the west has an odd number.
- A day's departures on a route are numbered in clock order. For example, Denver to Boston is DA 210, 212, 214, and 216, and Boston to Denver is DA 211, 213, 215, and 217.
- A flight keeps its number and its local time every day.
- Flights toward the east take a little less time than the flights back, and some flights arrive the next day.

A flight status is one of scheduled, boarding, delayed, en route, landed, or cancelled, with a cause for a delay or a cancellation.

### What the Desk Does Not Serve

The desk says what it does not serve. For any other place, the tools return a `not_served` answer. When a served city lies within 450 miles, the answer names it, for example "Demo Air doesn't fly to Boise, Idaho. The closest city we serve is Seattle, about 399 miles away." Otherwise, the answer lists the 12 cities. The `where_we_fly` tool lists the cities for a caller who asks.

The agent says once per call that the airline is made up and that nothing is charged.

### Sample Accounts

Two sample accounts tell two stories. A caller can also open an account in their own name.

| User ID | Spoken | Tier | Story |
| --- | --- | --- | --- |
| `jane_doe_001` | "jane doe zero zero one" | Gold | Two trips. One flight is delayed by 45 minutes (booking K7Q2ZP, Boston to Denver tomorrow). The other is a business trip from JFK to LAX in 24 days (M4X8RT). |
| `marcus_lee_002` | "marcus lee zero zero two" | Silver | One trip (R9T4WN, Chicago to Seattle) whose flight today Demo Air cancelled. The agent says so, finds another flight on the route, and moves the booking at no charge. The booking keeps the fare that was paid. Cancelling that booking refunds it in full. |

Try these things to say:

- "I need a flight from Boston to Denver tomorrow."
- "Anything leaving in the afternoon?"
- "Where do you fly?"
- "Can I fly from Boise to Denver?"
- "My user ID is jane doe zero zero one."
- "Is my flight today on time?" (as Marcus)

### Airline Tools

The tools are the `airline` tool source in `TOOL_SOURCES`, beside `cafe`. They run in the server process.

| Tool | What it does |
| --- | --- |
| `flights_search` | Lists up to five nonstop flights for a route and date. It accepts a city, airport name, or IATA code, and filters by time of day, departure window, cabin, and sort order. |
| `where_we_fly` | Lists the 12 cities that Demo Air serves. |
| `get_flight_status` | Returns the state, delay, times, terminal, and gate of one flight, with the cause of a delay or a cancellation. |
| `get_ontime_trend` | Returns how often a flight ran on time over about 90 days. |
| `sign_in` | Signs in with a user ID, or creates a demo account for a name. No password is needed. |
| `get_existing_bookings` | Lists the signed-in traveler's trips. |
| `new_booking`, `modify_existing_booking`, `cancel_booking`, `select_seat_and_meal`, `upgrade_class` | Read a change back, then commit it after the caller confirms. |

Each of the five change tools enforces the confirmation in the tool, not in the prompt. A call without `confirmed`, or a `confirmed: true` call with no matching read-back, returns `needs_confirmation` and changes nothing. A change commits only when the caller's yes follows the read-back of the same change. A repeated call ID returns the first result and never repeats a change. Fares are demo fares.

### Select the Domain

The registry entry selects the domain, and no code change is needed to switch. The `frontend-backend-live` entry in [`examples_registry.yaml`](../../../../../examples_registry.yaml) ships with these settings:

| Key | Airline value |
| --- | --- |
| `defaults.prompt` | `[airline]` |
| `agent_prompt_keys` | `[cafe]`, which hides the cafe talker prompt from the UI's prompt selector |
| `config.ui_bot.channel` | `airline` |
| `config.ui_bot.core.backend.prompt` | `airline` |
| `config.ui_bot.core.backend.tools` | `{airline: {}}` |

To run the Bluebird Cafe instead, make these changes in the same entry and restart the server:

- Set `config.ui_bot.channel` to `cafe`, `config.ui_bot.core.backend.prompt` to `cafe`, and `config.ui_bot.core.backend.tools` to `{cafe: {}}`.
- Set `defaults.prompt` to `[cafe]` and `agent_prompt_keys` to `[airline]`.

The entry's `event_log` capability, which turns on the [Event Relay](#event-relay), works for any domain, so it stays.

The Live gateway and its benchmarks use the `core` block, not `ui_bot`, so this choice does not affect them.

The `airline` entries of [`prompts.yaml`](prompts.yaml) are the top-level `airline` talker prompt (with its `tools_available`), `agent_prompts.Backend.airline`, and `agent_prompts.Channel.airline`. A `PROMPT_FILE_PATH` file that replaces `prompts.yaml` needs `Channel.airline` and `Backend.airline`. The shared `agent_prompts.Lines.intro` and `Lines.greeting` name no persona. The writer words the greeting from the talker prompt: "Greet the caller briefly as the person and business your instructions describe, and ask how you can help." The fixed fallback line is "Hi, thanks for calling. How can I help?"

### Event Relay

With the `event_log` capability in the registry entry, the UI bot relays its backend's OpenAI-format events to the browser as RTVI server messages, and the developer view's **EVENT LOG** tab prints them. Without the capability, the bot relays nothing. The relay is generic and works for any domain. Each message has this shape:

```json
{"type": "live-event", "v": 1, "seq": 0, "t": 0, "direction": "", "kind": "", "stream": "", "delegation": "", "event": {}}
```

The code is in [`media/live_events.py`](media/live_events.py), and [`pipeline.py`](pipeline.py) wires it with `relays_events()` and `tapped()`. A server tool call, which is an `mcp_call` item such as one with `server_label` `airline`, goes out whole: its `arguments` and `output` are not clipped, up to 64,000 characters for an event. A larger event has those fields cut until it fits. The relay cuts other long text at 2,000 characters. It shows the backend's prompt as its size and its tools as names.

For the tab, refer to the [client README](../../../../../clients/pipecat/README.md#event-log).

### Verify the Desk

Run the checks from the repository root:

```bash
uv run pytest tests/unit/pipecat/frontend_backend_live -q
uv run python scripts/airline_backend_smoke.py --script book manage scope disruption
```

The airline tests are `test_airline_tools.py`, `test_airline_network.py`, `test_live_events.py`, `test_ui_bot.py`, and `test_prompts.py`. The smoke script plays scripted callers against the real backend prompt and tools with a real model, so it needs `NVIDIA_API_KEY` in the environment. It prints the transcripts and flags long answers, repeated or missing demo notes, claims of a completed change in a read-back, and any airline other than Demo Air.

The client's Event log tests read a call recorded from these tools, which a unit test checks. Run this script after you change a tool or the sample accounts, and add `--check` to verify without writing:

```bash
uv run python scripts/export_airline_demo.py      # writes clients/pipecat/tests/airline-recorded-call.ts
```

### Balancing Conversation and Routing

The airline prompts add conversational behavior without replacing the routing rules that the cascade shares with every prompt:

- The talker says what the desk does, that the airline is made up, and who Ava is without a backend call. It never states a flight, price, code, or status before the backend returns it.
- The talker never asks for a user ID, a booking code, or a confirmation. The backend asks, so every yes, no, and correction reaches the tools.
- Lead-ins stay short and generic and name no detail, so the talker cannot promise an outcome.
- The backend words results for speech in short answers of about 35 words: two flights aloud and the rest on screen, trips by route and date, no recaps, 12-hour times, spoken dates, and one next step at most. It says once per call, or when the caller asks, that the airline is made up and that nothing is charged.

To change the desk's wording, edit the `airline` entries in [`prompts.yaml`](prompts.yaml). Keep these rules when you do, and repeat any limit on what the caller may hear in the backend prompt, as the tips below describe.

## Customization

To change the prompts, edit [`prompts.yaml`](prompts.yaml), or send them in each session's `instructions` and `delegation.responses.instructions`. For models, services, and prompts, refer to [Configure LLM](../../../../../docs/how-to/configure-llm.md), [Configure Services](../../../../../docs/how-to/configure-services.md), and [Configure Prompts](../../../../../docs/how-to/configure-prompts.md).

To add a part, add one class and one entry in its name table, which the settings, or a model's `services.yaml` entry, choose by name. The name tables are `FRONTEND_ADAPTERS` ([`adapter/registry.py`](adapter/registry.py)) for a frontend adapter, `LLM_SERVICES` ([`talker.py`](talker.py)) for a talker service, and the core's tables in [Extend the Core](delegation_core/README.md#extend-the-core).

## Troubleshooting

For a Live route that answers 404, a refused session, or a harness that does not connect, refer to [Frontend/Backend Live](../../../../../docs/06-troubleshooting.md#frontendbackend-live) in the Troubleshooting guide. For a line that gets no audio, refer to [Errors and Limits](../../../../../docs/how-to/use-live-gateway.md#errors-and-limits), and for general failure modes, to the [Troubleshooting guide](../../../../../docs/06-troubleshooting.md).

## Tips and Best Practices

Follow these practices:

- **Repeat the talker's limits in the backend's instructions.** When the talker cannot word a result, the voice says the backend's text as written. So also put any limits on what the caller may hear in `delegation.responses.instructions`, or in your backend's prompt in client mode.
- **Keep the app-wait line before the deadline.** `core.delegation.client_tool_timeout_seconds` (60) is the core's deadline for the app's function outputs. Keep `frontend.cascade.delegation.app_wait_seconds` (50) below it, so the `app_wait_line` plays while the work still waits.
- **Change the UI bot's domain in its registry entry.** A new top-level talker prompt does not change the UI bot's channel context, backend prompt, or tools. Set `ui_bot.channel`, `ui_bot.core.backend.prompt`, and `ui_bot.core.backend.tools` to the new domain, and set `defaults.prompt` and `agent_prompt_keys` to match, as [Select the Domain](#select-the-domain) describes for the airline and the cafe.
