# Frontend/Backend Live Cascaded Example

This example is a cascaded voice agent that puts a routing talker in front of backend work. On every caller turn, the talker, a fast large language model (LLM) call, speaks a reply, delegates the turn to the backend, or waits. The delegation core runs the delegated work on a backend adapter, and a writer words every ready result into one line, which the voice says when the caller is idle. The reference domain is the Bluebird Cafe ordering assistant: the backend looks up the menu, edits a cart, and places an order.

The example has two ways in:

- **Live session clients**, such as the **Live Console** (`/live.html`), Pipecat apps, and benchmark clients, connect through `POST /v1/live/sessions` or `WS /v1/live/sessions` and supply the prompts. For details, refer to [Live Sessions](#live-sessions).
- **The repository UI** (`/`) runs a bot whose backend runs the cafe tools on the server and shows each delegation as a task card. The Realtime gateway (`WS /v1/realtime`) drives the same bot.

Four parts serve a live session:

- The **Live gateway** ([`src/live/`](../../live/)) serves the live session protocol.
- The **frontend adapter** is this example. It turns each live session into a Pipecat cascade of automatic speech recognition (ASR), the talker, the writer, and text-to-speech (TTS).
- The **delegation core** ([`src/delegation_core/`](../../delegation_core/)) takes each delegation through the OpenAI Responses API. Its interaction manager (IM) owns the delegated work from then on.
- The **backend adapter** runs the work: the backend LLM, directly connected (the default); the backend LLM agent, a model conductor with LLM workers; or the harness, an external agent that you deploy, such as Hermes.

For what each part owns and how the parts meet, refer to the [Frontend/Backend Live Architecture](../../../docs/design/architecture.md) page.

Compared to the [Frontend/Backend Agent](../frontend_backend_agent/README.md), the talker returns a structured decision on every turn instead of calling a `call_backend` tool, and the writer words the backend's results for speech. Each part is a class that you select by name in the configuration, so adding one does not change the pipeline.

## Default Models

The `services` lists in [`examples_registry.yaml`](../../../examples_registry.yaml) resolve to the following default models for each profile. Service entries live in the root [`services.yaml`](../../../services.yaml):

| Profile | ASR | Talker LLM | Backend LLM | TTS |
| --- | --- | --- | --- | --- |
| Cloud | Nemotron ASR Streaming English | Nemotron 3.5 Lightning 30B A3B | Nemotron 3.5 Lightning 30B A3B with reasoning enabled | Magpie TTS Multilingual |
| Server | Nemotron ASR Streaming English NIM | Nemotron 3.5 Lightning 30B A3B NIM | Nemotron 3.5 Lightning 30B A3B NIM with reasoning enabled | Magpie TTS Multilingual NIM |
| Single GPU | Nemotron Speech Streaming English 0.6B through NeMo-Speech.cpp | Nemotron 3.5 Lightning 30B A3B through vLLM | Nemotron 3.5 Lightning 30B A3B through vLLM with reasoning enabled | Magpie TTS Multilingual through NeMo-Speech.cpp |

The talker and the backend use the same model with different `settings` in `examples_registry.yaml`. The talker runs at temperature 0, with reasoning off and at most 512 output tokens. The backend runs at temperature 0.7, with reasoning on and at most 4,096 output tokens, and its reasoning budget on Nemotron 3.5 Lightning is 256 tokens. Both slots also list Nemotron 3 Super.

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
| [`core.yaml`](core.yaml) | The delegation core: the IM's settings (`delegation`), every line the core writes (`texts`), and the backend adapter with its blocks (`backend`). `CORE_CONFIG_PATH` selects another file. |
| [`prompts.yaml`](prompts.yaml) | The `talker` and `thinker` prompts, used when a session or a request sets none, and the lines that the voice says word for word (`lines`). |

A profile is a file that names the file it builds on (`extends: core.yaml`) and sets only what differs. The example ships the following profiles:

| Profile | Use |
| --- | --- |
| [`core.agent.yaml`](core.agent.yaml) | The backend LLM agent (`backend.adapter: llm-agent`). |
| [`core.harness.yaml`](core.harness.yaml) | The harness (`backend.adapter: harness`), connected to an agent that you deploy. |
| [`core.ui-bot.yaml`](core.ui-bot.yaml) | The repository UI bot: the cafe tools on the server and a display part for the task cards. `ui_bot.core_config` in `config.yaml` names it. |
| [`config.bench.yaml`](config.bench.yaml), [`core.bench.yaml`](core.bench.yaml) | A benchmark client: no app-wait or progress lines, no client-tool deadline, and plain results. |

In `config.yaml`, `frontend.cascade` sets the talker's and the writer's model and prompt, and the cascade's behavior. The following excerpt shows the keys that you are most likely to change:

```yaml
frontend:
  adapter: cascade
  cascade:
    provider: chat-completions     # chat-completions (any OpenAI-compatible chat endpoint) | openai-responses
    slot: llm                      # the examples_registry.yaml services slot of the model
    prompt: talker                 # the prompts.yaml entry used when a session sets no instructions
    prompt_version: v2             # the prompt set: v1 | v2 | v3
    delegation:
      result_wording: writer       # writer | as_written
      progress_after_seconds: 8    # a progress line after this long of silent work; null: none
    memory:
      context_tokens: 16000        # the talker's memory budget
    tts_preroll_ms: 300            # reply audio held before playback starts; 0: off
```

The same block also holds the guards (`guards`), the retries and timeouts (`reliability`), the optional normalizer of spelled details (`normalizer`), and the speech settings (`speech`), each with a comment in the file. A model endpoint uses `NVIDIA_API_KEY` unless its `services.yaml` entry sets `api_key_env` to another variable.

In `core.yaml`, `backend.adapter` selects the backend adapter, and `delegation` sets the IM's limits. The following excerpt shows the keys that you are most likely to change:

```yaml
delegation:
  client_tool_timeout_seconds: 60   # how long a task waits for the app's function outputs; null: no deadline
  result_format: plain              # plain | speak_display (a show item for the screen, then the answer to speak)
  context_tokens: 24000             # the backend history's budget
backend:
  adapter: llm                      # llm (the default) | llm-agent | harness
  slot: thinker-llm                 # the services slot of the backend's default model
  prompt: thinker                   # the prompts.yaml entry used when a request sets no instructions
  llm:
    provider: chat-completions      # chat-completions | openai-responses
    conductor: steer                # steer: a new request joins the newest open work | start_each: a queue
```

The three backend adapters differ in how they place each request and run the work:

| Adapter | Placing requests | Running the work |
| --- | --- | --- |
| `llm` (the default) | A rule with no model call (`backend.llm.conductor`) | One run of the backend LLM |
| `llm-agent` | A model conductor that starts, steers, or cancels work, or answers a status question | Up to `backend.llm-agent.max_running` (3) runs of the backend LLM at once |
| `harness` | A conductor session in the agent that you deploy, with the same choices | Up to `backend.harness.max_running` (3) agent sessions at once |

To switch adapters, set `CORE_CONFIG_PATH` to its profile in `.env` (for Compose) or in your shell (on the host), for example `CORE_CONFIG_PATH=src/examples/frontend_backend_live/core.agent.yaml`, and restart the app service. The harness reaches an Agent Client Protocol (ACP) agent, with Hermes as the first agent class, through the `backend.harness` block of `core.yaml`. To deploy Hermes and connect it over stdio or WebSocket, refer to [Connect Your Own Harness](../../../docs/design/architecture.md#connect-your-own-harness).

## Live Sessions

A live session client sends its `instructions`, which are the talker's prompt, and a `delegation` block that selects the delegation mode for the whole session:

- **Responses mode** (`delegation.type: responses`). The server runs the backend work with the settings in `delegation.responses`: the backend `model`, which is required, and optional `instructions` and function `tools`. The backend's function calls stream to the client as `response.event`. The client returns each result as a `function_call_output` item with `response.item.create`, and then sends `response.create` to continue. The results must arrive within `delegation.client_tool_timeout_seconds` in `core.yaml`, or the core stops the work.
- **Client mode** (`delegation.type: client`, or no `delegation` block). The server runs no backend. It announces each delegation with `session.delegation.created`, and the client app does the work and sends the result back as a `session.commentary.append`.

During the call, the client can steer the cascade with three appends. Each one carries `delegation_id`, which is `null` in Responses mode and can name an announced delegation in client mode:

- `session.instructions.append` cuts the line in progress, joins the talker's instructions, and has the talker decide again.
- `session.thinking.append` adds silent context, which the talker sees and the next delegation carries.
- `session.commentary.append` adds a fact that the writer words and the voice says when the caller is idle.

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

The server serves the Live routes when `LIVE_ENABLED` is `true`, when `LIVE_API_TOKEN` is set, or when the deployment runs this example alone (`EXAMPLE_SELECTION=frontend-backend-live`). `LIVE_ENABLED=false` turns them off, and this example's Compose profiles default it to `true`. A server with more than one worker (`UVICORN_WORKERS`) serves no Live route, because a session lives in one process. The following list describes the routes and the settings that apply to every live session:

- **Authentication.** Set `LIVE_API_TOKEN` to require a bearer token on every Live route. Without one, anyone who can reach the server can use the routes. The Live Console asks for the token, or takes it from the page link as `live.html#token=<token>`. Browsers cannot send a header on a WebSocket, so use WebRTC from a browser when a token is set.
- **Server information.** `GET /v1/live/info` returns the voices, the backend's default model (`backend.model`) and the models it serves (`backend_models`), the default prompts, and a `warnings` list for configuration problems.
- **Client-mode rounds.** `POST /v1/live/delegate` runs one stateless backend round for a client app that does its own delegated work, such as the Live Console in client mode. At most `LIVE_MAX_CONSOLE_ROUNDS` (2) rounds run at once. With no token, the route answers only a `Host` that is an IP address, `localhost`, or a name in `LIVE_ALLOWED_HOSTS`.
- **Conversation history.** `GET /v1/live/sessions/{id}/history` returns what the talker and the backend hold now, with each system prompt split into parts named by source. In client mode, `backend` is `null`.
- **Voices.** The cascade speaks with the voice of its TTS service. Send `{"id": "<voice id>"}` to choose one of that service's voices.
- **Backend settings.** `delegation.responses.reasoning`, `text`, `service_tier`, and `parallel_tool_calls` reach a backend on the `openai-responses` provider as they are. A setting that the backend cannot honor, such as a hosted tool on a `chat-completions` backend, reaches the client as a nonfatal `responses_validation_failed` error.
- **Long calls.** The talker's memory drops its oldest whole turn groups past `compact_at` of `memory.context_tokens`, and the backend's history drops its oldest requests past `delegation.context_tokens`.

Stored recordings and session forks are not available. For the session limits and the other variables, refer to [Environment Variables](../../../docs/design/architecture.md#environment-variables) on the architecture page.

## Customization

The following list describes the common changes and where each one is documented:

- **Prompts.** Edit the `talker` and `thinker` entries in [`prompts.yaml`](prompts.yaml), or send your own in each session's `instructions` and `delegation.responses.instructions`.
- **Server tools.** A tool source runs tools on the server for the backend, as the cafe tools do in [`core.ui-bot.yaml`](core.ui-bot.yaml). Refer to [Add a Tool Source](../../../docs/design/architecture.md#add-a-tool-source).
- **Frontend or backend model provider.** Refer to [Add a Model Provider](../../../docs/design/architecture.md#add-a-model-provider).
- **Prompt set.** Refer to [Add a Prompt Set](../../../docs/design/architecture.md#add-a-prompt-set).
- **Frontend adapter, backend adapter, or harness agent.** Refer to [Extension Recipes](../../../docs/design/architecture.md#extension-recipes).

A Realtime frontend, which puts a remote model that speaks the OpenAI Realtime protocol in place of the cascade, is parked in [`parked/realtime/`](parked/realtime/). No setting selects it. For models, services, and prompts in general, refer to [Configure LLM](../../../docs/how-to/configure-llm.md), [Configure Services](../../../docs/how-to/configure-services.md), and [Configure Prompts](../../../docs/how-to/configure-prompts.md).

The Live tests run offline with scripted models. For their commands, refer to [Tests](../../../docs/design/architecture.md#tests) on the architecture page.

## Troubleshooting

For a Live route that answers 404, a refused session, `responses_validation_failed`, or a harness that does not connect, refer to [Frontend/Backend Live](../../../docs/06-troubleshooting.md#frontendbackend-live) in the Troubleshooting guide. For deployment and general failure modes, refer to the [Troubleshooting guide](../../../docs/06-troubleshooting.md).
