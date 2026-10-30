# Frontend/Backend Live Cascaded Example

This example is a cascaded voice agent that puts a routing talker in front of backend work. On every caller turn, the talker, a fast large language model (LLM) call, speaks a reply, delegates the turn to the backend, or waits. The delegation core runs the delegated work on a backend adapter, and a writer words every ready result into one line, which the voice says when the caller is idle. The reference domain is the Bluebird Cafe ordering assistant: the backend looks up the menu, edits a cart, and places an order.

The example has two ways in:

- **Live session clients**, such as the **Live Console** (`/live.html`), Pipecat apps, and benchmark clients, connect through `POST /v1/live/sessions` or `WS /v1/live/sessions` and supply the prompts. For details, refer to [Live Sessions](#live-sessions).
- **The repository UI** (`/`) runs a bot whose backend runs the cafe tools on the server and shows each delegation as a task card. The Realtime gateway (`WS /v1/realtime`) drives the same bot.

Four parts serve a live session, and the [Frontend/Backend Live Architecture](../../../docs/design/architecture.md) page describes what each one owns and how they meet:

- The **Live gateway** ([`src/live/`](../../live/)) serves the live session protocol.
- The **frontend adapter** is this example. It turns each live session into a Pipecat cascade of automatic speech recognition (ASR), the talker, the writer, and text-to-speech (TTS).
- The **delegation core** ([`src/delegation_core/`](../../delegation_core/)) takes each delegation through the OpenAI Responses API. Its interaction manager (IM) owns the delegated work from then on.
- The **backend adapter** runs the work: the backend LLM, directly connected (the default); the backend LLM agent, a model conductor with LLM workers; or the harness, an external agent that you deploy, such as Hermes.

Compared to the [Frontend/Backend Agent](../frontend_backend_agent/README.md), the talker returns a structured decision on every turn instead of calling a `call_backend` tool, and the writer words the backend's results for speech. Each part is a class that you select by name in the configuration, so adding one does not change the pipeline.

## Default Models

The `services` lists in [`examples_registry.yaml`](../../../examples_registry.yaml) resolve to the following default models for each profile. Service entries live in the root [`services.yaml`](../../../services.yaml):

| Profile | ASR | Talker LLM | Backend LLM | TTS |
| --- | --- | --- | --- | --- |
| Cloud | Nemotron ASR Streaming English | Nemotron 3.5 Lightning 30B A3B | Nemotron 3 Ultra 550B A55B with reasoning enabled | Magpie TTS Multilingual |
| Server | Nemotron ASR Streaming English NIM | Nemotron 3.5 Lightning 30B A3B NIM | Nemotron 3 Ultra 550B A55B with reasoning enabled, on the public NVIDIA API | Magpie TTS Multilingual NIM |
| Single GPU | Nemotron Speech Streaming English 0.6B through NeMo-Speech.cpp | Nemotron 3.5 Lightning 30B A3B through vLLM | Nemotron 3.5 Lightning 30B A3B through vLLM with reasoning enabled | Magpie TTS Multilingual through NeMo-Speech.cpp |

The talker's `llm` slot lists Nemotron 3.5 Lightning, then Nemotron 3 Super. The backend's `thinker-llm` slot lists Nemotron 3 Ultra, then Super, then Lightning. A profile's default is the first entry that its catalog serves. Ultra has no local server: the Cloud and Server profiles call it on the public NVIDIA API (`integrate.api.nvidia.com`) with `NVIDIA_API_KEY`, and Single GPU keeps its local Lightning, with Ultra listed after it when `NVIDIA_API_KEY` is set. Each slot's `settings` in `examples_registry.yaml` set its sampling. The talker runs at temperature 0, with reasoning off and at most 512 output tokens. The backend on Ultra runs at temperature 0, with reasoning on, at most 4,096 output tokens, and no reasoning budget, which Ultra's endpoint refuses. On Nemotron 3.5 Lightning, the backend runs at temperature 0.7 with a 256-token reasoning budget.

The `frontend-backend-live/server` profile starts the Nemotron 3.5 Lightning NIM for the talker. To run the backend on a local Nemotron 3 Super NIM instead of Ultra, add its own `nemotron-3-super` profile, which runs it on two more GPUs, device IDs `1` and `2` (2 × 80 GB): `docker compose --profile frontend-backend-live/server --profile nemotron-3-super up -d`, and choose Super for the backend in the Services tab. Name both profiles again when you tear the deployment down.

## Running the Example

This example runs with the **Cloud**, **Server**, and **Single GPU** profiles. Server is workstation-only (not DGX Spark or Jetson Thor), and Single GPU covers workstations, DGX Spark, and Jetson Thor. For prerequisites and hardware detail, refer to the [Getting Started guide](../../../docs/01-getting-started.md). Run every command from the repository root.

1. Preserve any existing `.env` file. Otherwise, copy the template, and then set `NVIDIA_API_KEY` in `.env` for the Cloud or Server profile. The Server profile also uses it to run its backend, Nemotron 3 Ultra, on the public NVIDIA API. For the Single GPU profile, set `HF_TOKEN` instead, which the profile uses to download model weights from Hugging Face:

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

4. Open the **Live Console** at `https://localhost:7860/live.html`, press **Start conversation**, allow the microphone, and ask what teas the cafe has. The repository UI is at `https://localhost:7860/`. Keep TLS enabled for browser testing, because browsers allow the microphone only on `localhost` or over HTTPS. `PIPELINE_TLS=false` serves plain HTTP. For plain-HTTP browser testing, refer to [browser access](../../../docs/06-troubleshooting.md#browser-access).

5. Clean up when you are done by tearing down with the same profile that you started, for example `docker compose --profile frontend-backend-live down`.

To run the server on the host instead, you need [`uv`](https://docs.astral.sh/uv/), Node.js 20 or newer, and `NVIDIA_API_KEY` for the Cloud models. Build the client once, then start the server with this example selected:

```bash
uv sync
cd client && npm ci && npm run build && cd ..
EXAMPLE_SELECTION=frontend-backend-live uv run python src/server.py --host 0.0.0.0 --port 7860
```

The Live Console shows every protocol event and each delegation, and it can steer the session while the call runs. Its **Delegation** setting starts each session in Responses mode or in client mode, which [Live Sessions](#live-sessions) describes.

## Configuration

The following files configure the example. The server reads `config.yaml` and `core.yaml` once, on first use, so an edit takes effect at the next restart. Unknown keys are errors, and a key in the wrong file stops the load with a message that names where it belongs.

| File | What it sets |
| --- | --- |
| [`examples_registry.yaml`](../../../examples_registry.yaml) | The `services.yaml` entries that each slot offers (`llm` for the talker and the writer, `thinker-llm` for the backend), and their sampling and reasoning settings. |
| [`config.yaml`](config.yaml) | The frontend: the frontend adapter (`frontend.adapter`) and its block (`frontend.cascade`). `LIVE_CONFIG_PATH` selects another file. |
| [`core.yaml`](core.yaml) | The delegation core: the IM's settings (`delegation`), the lines that the IM and the backend write (`texts`), and the backend adapter with its blocks (`backend`). `CORE_CONFIG_PATH` selects another file. |
| [`prompts.yaml`](prompts.yaml) | The `voice` and `backend` prompts, used as written when a session or a request sets none (neither names a domain); the cafe's talker and backend prompts (`talker` and `thinker`) for the repository UI bot and the Live Console; the lines that the voice says word for word or the Live Console sends (`lines`); and the Live Console's other append presets (`console_presets`). |

A profile is a file that names the file it builds on (`extends: core.yaml`) and sets only what differs. The example ships the following profiles:

| Profile | Use |
| --- | --- |
| [`core.agent.yaml`](core.agent.yaml) | The backend LLM agent (`backend.adapter: llm-agent`). |
| [`core.harness.yaml`](core.harness.yaml) | The harness (`backend.adapter: harness`), connected to an agent that you deploy. |
| [`core.ui-bot.yaml`](core.ui-bot.yaml) | The repository UI bot: the cafe tools on the server, the cafe's backend prompt (`thinker`), and a display part for the task cards. `ui_bot.core_config` in `config.yaml` names it. |
| [`config.bench.yaml`](config.bench.yaml), [`core.bench.yaml`](core.bench.yaml) | A benchmark client: no app-wait or progress lines, no client-tool deadline, plain results, and one backend task and session at a time for each call. |

In `config.yaml`, `frontend.cascade` sets the talker's and the writer's model and prompt, and the cascade's behavior. The following excerpt shows the keys that you are most likely to change:

```yaml
frontend:
  adapter: cascade
  cascade:
    provider: chat-completions     # chat-completions (any OpenAI-compatible chat endpoint) | openai-responses
    slot: llm                      # the examples_registry.yaml services slot of the model
    prompt: voice                  # the prompts.yaml entry used, as written, when a session sets no instructions or blank ones
    prompt_version: default        # the prompt set (prompts/default.py, the one set)
    delegation:
      app_wait_seconds: 50         # the app-wait line when the app's outputs take this long; null: none
      result_wording: writer       # writer | as_written
      progress_after_seconds: 8    # the writer can word a progress line after this long of silent work; null: none
      progress_max_per_delegation: 3  # progress lines offered for one delegation
    memory:
      context_tokens: 16000        # the talker's memory budget
    tts_preroll_ms: 300            # reply audio held before playback starts; 0: off
```

The same block also holds the retries and timeouts (`reliability`), the optional normalizer of spelled details (`normalizer`), and the speech settings (`speech`), each with a comment in the file. A model endpoint uses `NVIDIA_API_KEY` unless its `services.yaml` entry sets `api_key_env` to another variable.

In `speech`, `turn_end` sets how live sessions and the repository UI bot decide that the caller has finished a turn, and when the speech-to-text finalizes what it heard. With `smart_turn`, the default, the speech-to-text finalizes only after Smart Turn judges the turn complete or after a pause of `turn_silence_seconds` (1.2), because a final forced between the letters of a spelled ID loses letters. `silence` ends each turn at that pause, without Smart Turn, and `shared` uses the [turn detection](../../../docs/how-to/tune-pipeline-performance.md#smart-turn-detection) of the other cascaded examples, with its environment variables. For the details, refer to [How the Frontend Media Plane Meets the IM](../../../docs/design/architecture.md#how-the-frontend-media-plane-meets-the-im).

With `result_wording: writer`, the writer reads the talker's instructions and keeps any limits that they set on what the caller may be told. `as_written`, and every writer fallback for a result, says the backend's text as written, cleaned only for speech, so also put those limits in the backend's instructions: `delegation.responses.instructions`, or your own backend's prompt in client mode. For the writer's fallbacks, refer to [The Result Lifecycle](../../../docs/design/architecture.md#the-result-lifecycle).

When a delegation still waits for the app's function outputs after `frontend.cascade.delegation.app_wait_seconds` (50), the voice says the `app_wait_line` from `lines:` in [`prompts.yaml`](prompts.yaml) once. Keep this value below the core's deadline for those outputs, `delegation.client_tool_timeout_seconds` (60) in `core.yaml`, so the line plays while the work still waits.

After `progress_after_seconds` (8) of quiet while work runs, the writer can word a short progress line, up to `progress_max_per_delegation` (3) for one delegation. The writer reads the session's `instructions`, so a client that asks for silence while the backend works, such as the cafe's "then wait," gets no line. With `as_written`, no progress line is said. For the details, refer to [Spoken Progress](../../../docs/design/architecture.md#spoken-progress).

In `core.yaml`, `backend.adapter` selects the backend adapter, and `delegation` sets the IM's limits, which [Build and Configure the Core](../../delegation_core/README.md#build-and-configure-the-core) describes. The following excerpt shows the keys that you are most likely to change:

```yaml
delegation:
  client_tool_timeout_seconds: 60   # how long a task waits for the app's function outputs; null: no deadline
  result_format: plain              # plain | speak_display (a show item for the screen, then the answer to speak)
  context_tokens: 24000             # the backend history's budget
backend:
  adapter: llm                      # llm (the default) | llm-agent | harness
  slot: thinker-llm                 # the services slot of the backend's default model
  prompt: backend                   # the prompts.yaml entry used when a request sets no instructions
  llm:
    provider: chat-completions      # chat-completions | openai-responses
    conductor: steer                # steer: a new request joins the newest open work | start_each: a queue
```

The three backend adapters differ in how they place each request and run the work:

| Adapter | Placing requests | Running the work |
| --- | --- | --- |
| `llm` (the default) | A rule with no model call (`backend.llm.conductor`) | One run of the backend LLM |
| `llm-agent` | A model conductor that starts, steers, or cancels work, or answers a status question; while no work is open, a request starts work with no model call (`conductor.when_idle: start`); with `max_running: 1`, the `steer` rule, with no model call | Up to `backend.llm-agent.max_running` (3) runs of the backend LLM at once |
| `harness` | A conductor session in the agent that you deploy, with the same choices; with `max_running: 1`, the `steer` rule, with no conductor session | Up to `backend.harness.max_running` (3) agent sessions at once |

To switch adapters, set `CORE_CONFIG_PATH` to its profile in `.env` (for Compose) or in your shell (on the host), for example `CORE_CONFIG_PATH=src/examples/frontend_backend_live/core.agent.yaml`, and restart the app service. The harness reaches an Agent Client Protocol (ACP) agent that you deploy, such as Hermes, through the `backend.harness` block of `core.yaml`. To deploy Hermes and connect it over stdio or WebSocket, refer to [Connect Your Own Harness](../../../docs/design/architecture.md#connect-your-own-harness).

## Live Sessions

A live session client sends its `instructions`, which are the talker's prompt (a session that sets none, or blank ones, gets the `voice` prompt of [`prompts.yaml`](prompts.yaml), which names no domain), and a `delegation` block that selects the delegation mode for the whole session:

- **Responses mode** (`delegation.type: responses`). The server runs the backend work with the settings in `delegation.responses`: the backend `model`, which is required, and optional `instructions` and function `tools`. The backend's function calls stream to the client as `response.event`. The client returns each result as a `function_call_output` item with `response.item.create`, and then sends `response.create` to continue. The results must arrive within `delegation.client_tool_timeout_seconds` in `core.yaml`, or the core stops the work. When a backend task repeats its newest call with the same function and arguments after the client answered it, the core answers the repeat with that output and a note (`texts.repeated_call_output` in `core.yaml`) instead of sending it again, unless a request was admitted since or the work has taken more than one request.
- **Client mode** (`delegation.type: client`, or no `delegation` block). The server runs no backend. It announces each delegation with `session.delegation.created`, and the client app does the work and sends the result back as a `session.commentary.append`.

When the talker delegates a turn, the lead-in that it words in the same decision is the delegation's spoken acknowledgement. The voice says it after the session's `delegation.acknowledgement_delay_ms` (`0`, at once, by default), unless a result, the delegation's end, the caller starting to speak after the turn was delegated, or commentary or thinking comes first. A delegation without a lead-in says nothing. The voice never starts a reply, a lead-in, or the `holding` or `refusal` line while the caller speaks.

During the call, the client can steer the cascade with three appends. Each one carries a `content` of at most 500 tokens and a `delegation_id`, which is `null` in Responses mode and can name an announced delegation in client mode:

- `session.instructions.append` cuts the line in progress, joins the talker's instructions, and has the talker decide again. Every later delegation also carries it, so each request repeats every instruction append.
- `session.thinking.append` adds silent context, which the talker sees and the next delegation carries.
- `session.commentary.append` adds a fact that the writer words and the voice says when the caller is idle. Under `as_written` or a writer fallback, the voice says the fact word for word.

For the events of each delegation and the ways it ends, refer to [Endings](../../../docs/design/architecture.md#endings) on the architecture page. For how the server passes requests and function results to the delegation core, refer to [Server Delegation](../../live/README.md#server-delegation) in the Live gateway's README.

A Pipecat app connects with `OpenAILiveLLMService`. `ResponsesDelegation` selects Responses mode and sends the tools in the pipeline's `LLMContext` as the backend's function tools, and `ClientDelegation` selects client mode with a backend of the app's own. The following example selects Responses mode on a server started with `PIPELINE_TLS=false`:

```python
from pipecat.services.openai.live.llm import OpenAILiveLLMService
from pipecat.services.openai.responses.llm import OpenAIResponsesLLMService

llm = OpenAILiveLLMService(
    api_key="<LIVE_API_TOKEN>",  # any value when the server sets no token
    base_url="ws://localhost:7860/v1/live/sessions",
    delegation=OpenAILiveLLMService.ResponsesDelegation(
        settings=OpenAIResponsesLLMService.Settings(model="<backend model>"),
    ),
)
```

### Routes and Settings

The server serves the Live routes when `LIVE_ENABLED` is `true`, when `LIVE_API_TOKEN` is set, or when the deployment runs this example alone (`EXAMPLE_SELECTION=frontend-backend-live`). `LIVE_ENABLED=false` turns them off, and this example's Compose profiles default it to `true`. A server with more than one worker (`UVICORN_WORKERS`) serves no Live route, because a session lives in one process. For the session limits and the other variables, refer to [Environment Variables](../../../docs/design/architecture.md#environment-variables) on the architecture page. The following list describes the routes and the settings that apply to every live session:

- **Authentication.** Set `LIVE_API_TOKEN` to require a bearer token on every Live route. Without one, anyone who can reach the server can use the routes. The Live Console asks for the token, or takes it from the page link as `live.html#token=<token>`. Browsers cannot send a header on a WebSocket, so use WebRTC from a browser when a token is set.
- **Server information.** `GET /v1/live/info` returns the voices, the backend's default model (`backend.model`) and the models it serves (`backend_models`), the names of the default prompts (`frontend.prompt` and `backend.prompt`), and a `warnings` list for configuration problems. For the Live Console, it also returns the catalog's default talker prompt, the cafe's `talker` (`instructions.frontend`), the backend prompt and tools of the repository UI bot (`instructions.backend` and `sample_tools`), and the lines and append presets that the console sends (`lines` and `presets`).
- **Client-mode rounds.** `POST /v1/live/delegate` runs one stateless backend round for a client app that does its own delegated work, such as the Live Console in client mode. At most `LIVE_MAX_CONSOLE_ROUNDS` (2) rounds run at once. With no token, the route answers only a `Host` that is an IP address, `localhost`, or a name in `LIVE_ALLOWED_HOSTS`.
- **Conversation history.** `GET /v1/live/sessions/{id}/history` returns what the talker and the backend hold now, with each system prompt split into parts named by source. The talker's last part is the runtime data its next call sends after the conversation: the open work and a one-line decision cue. Each talker message (`frontend.messages`) has a `kind`, and each line that the next delegation carries (`backend.pending`) is `{kind, text}`. In client mode, `backend` is `null`.
- **Voices.** The cascade speaks with the voice of its TTS service. Send `{"id": "<voice id>"}` to choose one of that service's voices.
- **Backend settings.** `delegation.responses.reasoning`, `text`, `service_tier`, and `parallel_tool_calls` reach a backend on the `openai-responses` provider as they are. A setting that the backend cannot honor, such as a hosted tool on a `chat-completions` backend, reaches the client as a nonfatal `responses_validation_failed` error.
- **Long calls.** The talker's memory drops its oldest whole turn groups past `compact_at` of `memory.context_tokens`, and the backend's history drops its oldest requests past `delegation.context_tokens`.
- **Recordings and forks.** Stored recordings and session forks are not available.

## Customization

The following list describes the common changes and where each one is documented:

- **Prompts.** Edit the `voice`, `backend`, `talker`, and `thinker` entries in [`prompts.yaml`](prompts.yaml), or send your own in each session's `instructions` and `delegation.responses.instructions`.
- **Server tools.** A tool source runs tools on the server for the backend, as the cafe tools do in [`core.ui-bot.yaml`](core.ui-bot.yaml). Refer to [Add a Tool Source](../../../docs/design/architecture.md#add-a-tool-source).
- **Frontend or backend model provider.** Refer to [Add a Model Provider](../../../docs/design/architecture.md#add-a-model-provider).
- **Prompt set.** Refer to [Add a Prompt Set](../../../docs/design/architecture.md#add-a-prompt-set).
- **Frontend adapter, backend adapter, or harness agent.** Refer to [Extension Recipes](../../../docs/design/architecture.md#extension-recipes).

The [`parked/realtime/`](parked/realtime/) package holds a Realtime frontend, which puts a remote model that speaks the OpenAI Realtime protocol in place of the cascade. No setting selects it. For models, services, and prompts in general, refer to [Configure LLM](../../../docs/how-to/configure-llm.md), [Configure Services](../../../docs/how-to/configure-services.md), and [Configure Prompts](../../../docs/how-to/configure-prompts.md). The Live tests run offline with scripted models. For their commands, refer to [Tests](../../../docs/design/architecture.md#tests) on the architecture page.

## Troubleshooting

For a Live route that answers 404, a refused session, `responses_validation_failed`, or a harness that does not connect, refer to [Frontend/Backend Live](../../../docs/06-troubleshooting.md#frontendbackend-live) in the Troubleshooting guide. For deployment and general failure modes, refer to the [Troubleshooting guide](../../../docs/06-troubleshooting.md).
