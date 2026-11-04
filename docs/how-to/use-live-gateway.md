# Use the Live Gateway

## Goals

`/v1/live` lets live session clients, such as Pipecat apps and benchmark clients, talk to the [Frontend/Backend Live example](../../src/examples/frontend_backend_live/README.md) over WebRTC or WebSocket, with the prompts that they supply.

The Live gateway serves the live session protocol at `/v1/live`. It owns each session: the order of its commands and of its close, the wire schema, and the relay of each delegation's events. It knows nothing about Pipecat or the cascade, and it reaches the frontend adapter and the delegation core only through the ports in [`ports.py`](../../src/gateways/live/ports.py). The Frontend/Backend Live example plugs its own parts into it. For the design, refer to the [Frontend/Backend Live Architecture](../design/architecture.md) page.

## Routes

The gateway serves the following routes:

| Route | What it serves |
| --- | --- |
| `POST /v1/live/sessions` | A WebRTC session: the client's SDP offer and settings in, the answer out |
| `WS /v1/live/sessions` | A primary WebSocket session: `session.start` first, then commands and base64 audio events |
| `WS /v1/live/sessions/{id}/attach` | A sideband connection that observes a session and sends trusted commands |
| `GET /v1/live/sessions/{id}/history` | What the session's models hold now: the frontend adapter's half and the delegation part's half |
| `GET /v1/live/info` | The server's voices and session limit, plus what the example's `describe` hook adds |
| `POST /v1/live/delegate` | One backend round for a client app that does its own delegated work |

Any other path under `/v1/live` answers 404.

### When the Routes Are Served

The server serves the Live routes when `LIVE_ENABLED` is `1`, `true`, `yes`, or `on`, and any other value turns them off. When `LIVE_ENABLED` is unset, the server serves them when `LIVE_API_TOKEN` is set or when the deployment runs the Frontend/Backend Live example alone. The example runs alone with `EXAMPLE_SELECTION=frontend-backend-live`, or with `selection: frontend-backend-live` in `examples_registry.yaml`. The example's Compose profiles default `LIVE_ENABLED` to `true`. A server with more than one worker (`UVICORN_WORKERS`) serves no Live route, because a session lives in one process.

### Authentication

Set `LIVE_API_TOKEN` to require a bearer token on every Live route. Without one, anyone who can reach the server can use the routes. The routes refuse a browser page from another origin with `403 Unexpected browser origin`. A browser cannot send the token on a WebSocket, so a page that this server serves uses WebRTC when a token is set.

### Client-Mode Rounds

`POST /v1/live/delegate` runs one stateless backend round for a client app that does its own delegated work. At most `LIVE_MAX_DELEGATE_ROUNDS` (2) rounds run at once. With no token, the route answers only a `Host` that is an IP address, `localhost`, or a name in `LIVE_ALLOWED_HOSTS`.

## Connect a Pipecat App

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

## Delegation Modes

A live session client sends its `instructions`, which are the talker's prompt (a session that sets none, or blank ones, gets the `generic` talker prompt of the example's [`prompts.yaml`](../../src/examples/frontend_backend_live/prompts.yaml), which names no domain), and a `delegation` block. The block's `type` selects the delegation mode for the whole session:

- **Responses mode** (`delegation.type: responses`). The server delegation ([`server_delegation/`](../../src/gateways/live/server_delegation/)) passes each delegation to the [delegation core](../../src/examples/frontend_backend_live/delegation_core/README.md), as [Server Delegation](#server-delegation) describes. The server runs the backend work with the settings in `delegation.responses`: the backend `model`, which is required, and optional `instructions` and function `tools`. The backend's function calls stream to the client as `response.event`. The client returns each result as a `function_call_output` item with `response.item.create`, and then sends `response.create` to continue. The results must arrive within `delegation.client_tool_timeout_seconds` in `core.yaml`, or the core stops the work. When a backend task repeats its newest call with the same function and arguments after the client answered it, the core answers the repeat with that output and a note (`texts.repeated_call_output` in `core.yaml`) instead of sending it again, unless a request was admitted since or the work has taken more than one request.
- **Client mode** (`delegation.type: client`, or no `delegation` block). The server runs no backend. The client delegation ([`client_delegation.py`](../../src/gateways/live/client_delegation.py)) announces each delegation with `session.delegation.created`. The client app does the work and answers with appends that name the delegation, and it sends the result back as a `session.commentary.append`. The server builds no delegation core for the session, which refuses `response.item.create` and `response.create`.

When the talker delegates a turn, the lead-in that it words in the same decision is the delegation's spoken acknowledgement. The voice says it after the session's `delegation.acknowledgement_delay_ms` (`0`, at once, by default), unless a result, the delegation's end, the caller starting to speak after the turn was delegated, or commentary or thinking comes first. A delegation without a lead-in says nothing. The voice never starts a reply, a lead-in, or the `holding` or `refusal` line while the caller speaks.

For the events of each delegation and the ways it ends, refer to [Endings](../design/architecture.md#endings) on the architecture page.

### Server Delegation

The server delegation holds no delegation logic: the delegation core decides what happens to each request, and the frontend adapter decides what the caller hears. For each session, the server delegation does the following:

- It opens the session's conversation in the core with `POST /v1/conversations`, and the session fails to start when the core cannot create it within 10 seconds. It deletes the conversation when the session closes.
- It sends each request, in order, to `POST /v1/responses` with the session's `delegation.responses` settings, read for each request, so a `session.update` applies to the requests after it.
- It relays each event that comes back to the frontend adapter and, in a `response.event` that names the `delegation_id`, to the client. When a stream ends early, the client gets a `response.failed` with `server_error`.
- It holds the client's call results and typed messages from `response.item.create`, at most 512 results and 16 messages, and sends them at `response.create`.
- It checks the `delegation.responses` settings in the background with `POST /v1/responses/input_tokens`, and it reports a setting that the backend cannot honor as a nonfatal `responses_validation_failed` error.

## Steer a Session

During the call, the client can steer the cascade with three appends. Each one carries a `content` of at most 500 tokens and a `delegation_id`, which is `null` in Responses mode and can name an announced delegation in client mode:

- `session.instructions.append` cuts the line in progress, joins the talker's instructions, and has the talker decide again. Every later delegation also carries it, so each request repeats every instruction append.
- `session.thinking.append` adds silent context, which the talker sees and the next delegation carries.
- `session.commentary.append` adds a fact that the writer words and the voice says when the caller is idle. Under `as_written` or a writer fallback, the voice says the fact word for word.

## Server Information and History

The following routes report on the server and on a session:

- **Server information.** `GET /v1/live/info` returns the voices, the backend's default model (`backend.model`) and the models it serves (`backend_models`), the names of the default prompts (`frontend.prompt` and `backend.prompt`), and a `warnings` list. A frontend adapter can report configuration problems there, and the cascade adapter reports none.
- **Conversation history.** `GET /v1/live/sessions/{id}/history` returns what the talker and the backend hold now, with each system prompt split into parts named by source. The talker's last part is the runtime data its next call sends after the conversation: the open work and a one-line decision cue. A call that decides caller turns also quotes their words before the cue, as the caller's newest turn, and the history leaves that line out. Each talker message (`frontend.messages`) has a `kind`, and each line that the next delegation carries (`backend.pending`) is `{kind, text}`. In client mode, `backend` is `null`.

## Settings

### Session Settings

The following settings apply to a live session:

- **Voices.** The cascade speaks with the voice of its TTS service. Send `{"id": "<voice id>"}` to choose one of that service's voices.
- **Backend settings.** `delegation.responses.reasoning`, `text`, `service_tier`, and `parallel_tool_calls` reach a backend on the `openai-responses` provider as they are. A setting that the backend cannot honor, such as a hosted tool on a `chat-completions` backend, reaches the client as a nonfatal `responses_validation_failed` error.

### Server Settings

The server reads the following environment variables when it starts:

| Variable | Default | What it sets |
| --- | --- | --- |
| `LIVE_ENABLED` | Unset | Whether the routes are served, as [When the Routes Are Served](#when-the-routes-are-served) describes. |
| `LIVE_API_TOKEN` | Unset | The bearer token that every route requires. Without one, the server logs a warning at startup, and anyone who can reach it can use the routes. |
| `LIVE_MAX_SESSIONS`, `LIVE_MAX_SESSION_SECONDS` | `8`, `1800` | The sessions at once and the length of one session, in seconds. |
| `LIVE_MAX_APPENDS_PER_MINUTE` | `64` | The appends that one session accepts in any 60 seconds, counting `session.instructions.append`, `session.thinking.append`, and `session.commentary.append` together. An append over the limit gets a `rate_limit_exceeded` error, and the gateway neither passes it to the frontend adapter nor acknowledges it. |
| `LIVE_ICE_URLS` | Unset | More ICE servers for the WebRTC peer, comma-separated, beside the TURN server that `TURN_USERNAME`, `TURN_PASSWORD`, and `TURN_URL` configure. |
| `LIVE_MAX_DELEGATE_ROUNDS`, `LIVE_ALLOWED_HOSTS` | `2`, unset | The `POST /v1/live/delegate` rounds at once, and, with no token, the host names besides IP addresses and `localhost` that the route answers to. |
| `UVICORN_WORKERS` | `1` | The server's workers. The Live routes need one, as [When the Routes Are Served](#when-the-routes-are-served) describes. |

When the Live routes are served, a value of `LIVE_MAX_SESSIONS`, `LIVE_MAX_SESSION_SECONDS`, `LIVE_MAX_APPENDS_PER_MINUTE`, or `LIVE_MAX_DELEGATE_ROUNDS` that is not a whole number of at least 1 stops the server at startup. `UVICORN_WORKERS` is read before that check: for a value that is not a whole number of at least 1, the server logs a warning and runs one worker.

For the example's own variables, refer to [Environment Variables](../design/architecture.md#environment-variables).

## Errors and Limits

When the frontend adapter reports a failure, the client gets an `error` with code `pipeline_error` if a client command failed or the session cannot go on, and an `info` notice with the same code if the session recovers and no command caused it. The gateway also logs each one as a warning.

The following limits apply to every live session:

- **Long calls.** The talker's memory drops its oldest whole turn groups past `compact_at` of `memory.context_tokens`, and the backend's history drops its oldest requests past `delegation.context_tokens`.
- **Recordings and forks.** Stored recordings and session forks are not available.
