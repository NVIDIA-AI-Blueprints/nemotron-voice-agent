# Frontend/Backend Live Cascaded Example

Cascaded voice agent that puts a routing talker in front of a tool-using backend. On every turn, the talker either speaks a reply or delegates to the backend, and a commentary step phrases the backend result for speech. The reference domain is the Bluebird Cafe ordering assistant: the backend looks up the menu, edits a cart, and places an order.

## Quick Start

Prerequisites: [`uv`](https://docs.astral.sh/uv/), Node.js 20 or newer, and an `NVIDIA_API_KEY` for the hosted models (the other recipes in the [Getting Started guide](../../../docs/01-getting-started.md) run local services instead).

```bash
uv sync                                                           # Python dependencies
cd client && npm ci && npm run build && cd ..                     # build the UIs once
export NVIDIA_API_KEY=...
EXAMPLE_SELECTION=frontend-backend-live uv run python src/server.py --host 0.0.0.0 --port 7860
```

Open `https://localhost:7860/live.html` (the **Live Console**), press **Start conversation**, allow the microphone, and ask what teas the cafe has. The console shows the frontend and backend models, every protocol event, each delegation with its tool calls, and controls to change the session, steer the frontend, and update the backend while the call runs. Set `PIPELINE_TLS=false` for plain `http://localhost:7860`. Browsers only allow the microphone on `localhost` or HTTPS.

The example has two ways in, both sharing one cascade:

- **The Live Console** (`/live.html`) and **live session clients** (`POST /v1/live/sessions` and `WS /v1/live/sessions`, served by [`src/live/`](../../live/)). The client supplies the prompts and the tools, and it executes the backend's tools itself.
- **The repository UI** (`/`). The backend runs the cafe tools on the server.

This example differs from the [Frontend/Backend Agent](../frontend_backend_agent/README.md) in three ways:

- The talker returns a structured decision (`speak` or `delegate`) on every turn instead of calling a `call_backend` tool.
- The backend answer passes through a commentary step that phrases it for speech, with guards that keep unverified claims out of the lead-in.
- The frontend and backend models are classes that you select by name in `config.yaml`. Adding a provider does not change the pipeline.

## How a Turn Works

```text
transport -> STT -> user aggregator -> LiveTalkerProcessor -> TTS -> transport -> assistant aggregator
                                              |
                          Talker --- speak ---+---> spoken reply
                             |
                          delegate ---> Thinker (backend model + tools) ---> CommentaryWriter ---> spoken answer
```

`LiveTalkerProcessor` sits where the LLM service normally sits. It receives the context frame for each user turn and pushes the same response frames an LLM service pushes, so TTS and the assistant aggregator need no changes.

1. The **talker** returns one JSON decision: `speak` with a reply, or `delegate` with an optional short lead-in. A turn always gets one of the two.
2. A **delegation** goes to a queue that runs one backend round at a time. The backend calls tools, the worker adds the results to the history, and the next round starts until the model answers.
3. The **commentary writer** phrases the answer for speech and does not repeat the lead-in. If phrasing fails, the verified text is spoken as written.
4. The **answer** is spoken after the caller finishes their sentence. It is dropped only when a newer delegation is queued.

## Configuration

| File | Decides |
| --- | --- |
| [`examples_registry.yaml`](../../../examples_registry.yaml) | The `services.yaml` entries each slot offers (`llm` for the talker, `thinker-llm` for the backend), and their default sampling and reasoning settings. |
| [`config.yaml`](config.yaml) | The registered implementation for each role, the prompt set, retries and timeouts, and the guards. Unknown keys are errors. |

```yaml
frontend:
  provider: chat-completions
  slot: llm
  prompt: talker
backend:
  provider: chat-completions
  slot: thinker-llm
  prompt: thinker
prompt_version: v1
```

To put a remote realtime model in front of the same backend instead of the cascade, set the frontend's `provider` to `realtime`. See [Using Your Own Realtime Frontend](#using-your-own-realtime-frontend). The cascade's routing guards do not apply to it; its prompt and the `delegate` tool do the routing.

`slot` picks the model, and `prompt` names the [`prompts.yaml`](prompts.yaml) entry a role uses when a live session sets no instructions of its own. A session's `instructions` and `delegation.responses.instructions` take precedence.

| Provider | Serves |
| --- | --- |
| `chat-completions` | Any OpenAI-compatible chat endpoint: NVIDIA NIM, vLLM, or a hosted API. |
| `openai-responses` | The OpenAI Responses API. |

`prompt_version` selects the routing, commentary, and answer text added to the prompts in [`prompts.yaml`](prompts.yaml). The guards (repeat guard, turn router, and failure recovery) are set under `guards` and `reliability`. A backend round that failed after emitting a function call is never retried.

## Using Your Own Realtime Frontend

Any server that speaks the OpenAI Realtime protocol and supports client function tools can be the frontend. The bridge connects to it, gives it a `delegate` function, and runs the backend when the model calls it. Three edits point the example at your server:

1. [`services.yaml`](../../../services.yaml), under `server.llm`, the server's entry:

   ```yaml
   my-realtime:
     name: "My realtime server"
     model_id: "my-realtime-model"
     base_url: "wss://realtime.example.com/v1/realtime"
     api_key_env: MY_REALTIME_API_KEY
     health_path: ""
   ```

2. [`examples_registry.yaml`](../../../examples_registry.yaml), in this example's block, offer it through its own slot: add `realtime-frontend: [my-realtime]` under `services` and `realtime-frontend: llm` under `categories`.
3. [`config.yaml`](config.yaml): `frontend: {provider: realtime, slot: realtime-frontend}`.

Export `MY_REALTIME_API_KEY` and start with `SERVICE_RECIPE=server`. A custom entry is only offered by the `server` recipe; the `cloud` recipe drops entries without an `nvcf` key. If the slot does not resolve to a `ws://` or `wss://` URL, sessions fail with the reason in the server log and `GET /v1/live/info` lists it under `warnings`.

For a server whose session settings differ, add `extra_params` to the entry as JSON: `headers` (extra request headers), `session` (merged into the `session.update` the bridge sends), and `transcription_model`. The bridge sends 24 kHz PCM16 audio and the current Realtime event names, and it accepts the older audio and transcript event names too. This is tested against a scripted server and OpenAI's hosted Realtime API.

## Adding a Frontend or Backend

Subclass `Frontend` or `Backend` from [`models/base.py`](models/base.py) and register it by name. Prompts, retries, and guards live outside the classes, so a new provider gets them unchanged.

```python
from examples.frontend_backend_live.models import FRONTENDS, Frontend
from examples.frontend_backend_live.models.decision import TalkerDecision


@FRONTENDS.register("acme")
class AcmeFrontend(Frontend):
    """Talker on the Acme API."""

    async def complete_once(self, instructions, history, actions, max_output_tokens):
        """Return one decision; raise on failure so transient errors are retried."""
        return TalkerDecision("speak", "Hello.")
```

Import the module in [`models/__init__.py`](models/__init__.py), then set `provider: acme` in `config.yaml`. A `Backend` implements `stream(payload)`, which yields Responses-style events.

## Live Sessions

A live session client sends its `instructions` (the talker prompt) and `delegation.responses` (backend model, instructions, and function tools). In `responses` mode the backend's function calls stream to the client as `response.event`. The client returns each result with `response.item.create`, and `response.create` continues the round ([`client_tools.py`](tool_calling/client_tools.py)). In `client` mode, the server announces each delegation, and `session.commentary.append` brings the result back for phrasing. `session.instructions.append` interrupts speech and decides again, and `session.thinking.append` adds silent context.

`GET /v1/live/info` returns the voices, models, default prompts, guards, and sample tools that the Live Console shows, plus a `warnings` list for configuration problems. Models, ASR, TTS, voice activity detection, and interruptions come from the same services as the other examples. Stored recordings and session forks are not available.

- **When the routes are served.** They are open to anyone who can reach the server, so they are added only when the deployment is pinned to this example (`EXAMPLE_SELECTION=frontend-backend-live`, as the Compose profiles do), when `LIVE_API_TOKEN` is set, or when `LIVE_ENABLED=true`. `LIVE_ENABLED=false` turns them off. Without a token the server logs a warning at startup.
- **Authentication.** Set `LIVE_API_TOKEN` to require a bearer token. The Live Console has a field for it. Browsers cannot send a header on a WebSocket, so use WebRTC from a browser when a token is set.
- **Client delegation.** `POST /v1/live/delegate` runs one backend round for a client that owns its delegation: it sends the conversation (`input`) and its function tools, runs the function calls in the reply, and sends the next round. The Live Console's `client` mode uses it, then sends the result back as `session.commentary.append`. It is served and authenticated like the session routes.
- **Voices.** Any voice name in the protocol is accepted, but the cascade speaks with the voice of its text-to-speech service. Send `{"id": "<voice id>"}` to choose one of that service's voices. A remote realtime frontend receives the name.
- **Backend settings.** `delegation.responses.reasoning`, `text`, `service_tier`, and `parallel_tool_calls` reach a backend on the `openai-responses` provider. A `chat-completions` backend has no equivalent and ignores them; it applies `model`, `instructions`, `tools`, `tool_choice`, and `max_output_tokens`.
- **Long calls.** The talker keeps the prompt and its latest `chat_history_recent_turns` messages, and the backend drops its oldest requests, whole, when its history outgrows `backend_context_tokens`.

## Running the Example

This example runs with the **Cloud**, **Server**, and **Single GPU** profiles. Refer to the [Getting Started guide](../../../docs/01-getting-started.md) for prerequisites. Run every command from the repository root.

```bash
docker compose --profile frontend-backend-live up -d              # Cloud ASR, LLM, TTS
docker compose --profile frontend-backend-live/server up -d       # Local NIM ASR, TTS, LLM
docker compose --profile frontend-backend-live/single-gpu up -d   # Lightning + NeMo-Speech.cpp
```

To run on the host:

```bash
EXAMPLE_SELECTION=frontend-backend-live uv run python src/server.py --host 0.0.0.0 --port 7860
```

## Source Layout

The talker, commentary, delegation, and model packages do not import Pipecat, so you can reuse them in another host.

| Package | Owns |
| --- | --- |
| [`engine/`](engine/) | What every live engine shares: session settings and updates, the pipeline host, and the factory the live server calls. |
| [`cascade/`](cascade/) | The core cascaded pipeline: the per-call session, the Pipecat processor, speech services, and the `talker/` and `commentary/` packages. |
| [`realtime/`](realtime/) | A remote realtime model as the frontend: the WebSocket client processor and its engine. |
| [`delegation/`](delegation/) | The delegation coordinator, the serialized backend worker, the thinker wrapper, and the events relayed to the client. |
| [`tool_calling/`](tool_calling/) | The client's tool gateway, the executor interface, and the reference cafe tools in [`cafe/`](tool_calling/cafe/). |
| [`models/`](models/) | The `Frontend` and `Backend` base classes, the registry, endpoint resolution, and the implementations. |
| [`prompts/`](prompts/) | Prompt sets `v1` and `v2`, the static prompts, and the realtime frontend's text. |
| [`config_manager/`](config_manager/) | Loading and validating `config.yaml`. |
| [`common/`](common/) | Ids, history conventions, word helpers, and retries with backoff. |

## Tests

```bash
uv run pytest tests/unit/test_frontend_backend_live*.py tests/unit/test_live_*.py -v
```

The tests run offline with scripted models. One test pins the SHA-256 digest of every prompt text.
