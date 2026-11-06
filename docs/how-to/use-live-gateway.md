# Use the Live Gateway

## Goals

`/v1/live` lets live session clients, such as Pipecat apps or your own app, talk to the [Frontend/Backend Live example](../../src/examples/frontend_backend_live/README.md) over WebRTC or WebSocket, with the prompts that they supply.

The Live gateway serves the live session protocol at `/v1/live`. It owns each session: the order of its commands and of its close, the wire schema, and the relay of each delegation's events. It knows nothing about Pipecat or the cascade, and it reaches the frontend adapter and the delegation core only through the ports in [`ports.py`](../../src/gateways/live/ports.py). The Frontend/Backend Live example plugs its own parts into it.

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

An HTTP request to any other path under `/v1/live` answers 404, and a WebSocket connection there is refused with HTTP 403.

### When the Routes Are Served

The server serves the Live routes only when its selected example has a `live` block in [`examples_registry.yaml`](../../examples_registry.yaml), as `frontend-backend-live` does. The `frontend-backend-live` profiles select that example. A deployment of another example serves no Live route, whatever its settings. With `EXAMPLE_SELECTION=all`, the selected example is the first one listed, so that deployment serves no Live route either.

On a `frontend-backend-live` deployment, the server serves the Live routes when `LIVE_API_TOKEN` is set. Without a token, it serves them only when you set `LIVE_ENABLED=true`, for example on a trusted network. When `LIVE_ENABLED` is set, it decides: `1`, `true`, `yes`, or `on` serve the routes, and any other value turns them off. A server with more than one worker (`UVICORN_WORKERS`) serves no Live route, because a session lives in one process.

### Authentication

Set `LIVE_API_TOKEN` to require a bearer token on every Live route. The routes refuse a browser page from another origin with `403 Unexpected browser origin`.

Without a token, the routes answer only requests addressed to an IP address, `localhost`, or a host name in the `allowed_hosts` [server setting](#server-settings). This check stops a webpage on another website from using the routes through a host name that resolves to your server.

### Client-Mode Rounds

`POST /v1/live/delegate` runs one stateless backend round for a client app that does its own delegated work. The `max_delegate_rounds` [server setting](#server-settings) (2) caps the rounds that run at once. A round that names a model that the backend does not serve is refused with HTTP 400.

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

- **Responses mode** (`delegation.type: responses`). The server delegation ([`server_delegation/`](../../src/gateways/live/server_delegation/)) passes each delegation to the [delegation core](../../src/examples/frontend_backend_live/delegation_core/README.md), as [Server Delegation](#server-delegation) describes. The server runs the backend work with the settings in `delegation.responses`: the backend `model`, which is required, and optional `instructions` and function `tools`. The backend's function calls stream to the client as `response.event`. The client returns each result as a `function_call_output` item with `response.item.create`, and then sends `response.create` to continue. The results must arrive within the core's `delegation.client_tool_timeout_seconds` (60 seconds), or the core stops the work. When a backend task repeats its newest call with the same function and arguments after the client answered it, the core answers the repeat with that output and a note (`texts.repeated_call_output`, an `agent_prompts.Core` entry of the example's `prompts.yaml`) instead of sending it again, unless a request was admitted since or the work has taken more than one request.
- **Client mode** (`delegation.type: client`, or no `delegation` block). The server runs no backend. The client delegation ([`client_delegation.py`](../../src/gateways/live/client_delegation.py)) announces each delegation with `session.delegation.created`. The client app does the work and answers with appends that name the delegation, and it sends the result back as a `session.commentary.append`. The server builds no delegation core for the session, which refuses `response.item.create` and `response.create`.

When the talker delegates a turn, the lead-in that it words in the same decision is the delegation's spoken acknowledgement. The voice says it after the session's `delegation.acknowledgement_delay_ms`, unless a result, the delegation's end, the caller starting to speak after the turn was delegated, or commentary or thinking comes first. The delay is at most `800`, and `0`, the default, says it at once. A larger value at `session.start` or `session.update` is refused with `invalid_value`. A delegation without a lead-in says nothing. The voice never starts a reply, a lead-in, or the `holding` or `refusal` line while the caller speaks.

### Server Delegation

The server delegation holds no delegation logic: the delegation core decides what happens to each request, and the frontend adapter decides what the caller hears. For each session, the server delegation does the following:

- It opens the session's conversation in the core with `POST /v1/conversations`, and the session fails to start when the core cannot create it within 10 seconds. It deletes the conversation when the session closes.
- It sends each request, in order, to `POST /v1/responses` with the session's `delegation.responses` settings, read for each request, so a `session.update` applies to the requests after it. A delegation that joins open work keeps that work's settings until the client's next `response.create` brings the current ones. Otherwise, a change, such as new tools, applies from the next delegation that starts new work. OpenAI's GPT-Live API applies it to every later delegation.
- It relays each event that comes back to the frontend adapter and, in a `response.event` that names the `delegation_id`, to the client. When a stream ends early, the client gets a `response.failed` with `server_error`.
- It holds the client's call results and typed messages from `response.item.create`, at most 512 results and 16 messages, and sends them at `response.create`. A typed message steers open work as a spoken request does. OpenAI's GPT-Live API queues typed text behind the work already running.
- It checks the `delegation.responses` settings in the background with `POST /v1/responses/input_tokens`, and it reports a setting that the backend cannot honor as a nonfatal `responses_validation_failed` error. A model that the backend does not serve is one such setting: the session keeps it, and each delegation then fails, except on the harness, which runs it on its agent's own model.

The core places each request by its backend adapter's rule, as [Choose a Backend Adapter](../../src/examples/frontend_backend_live/README.md#choose-a-backend-adapter) describes. With the `llm` adapter, or at `max_running` 1, an unrelated request that arrives while work is open joins that work, so the caller hears both answers together. GPT-Live runs each delegation separately. With `max_running` above 1, the conductor starts unrelated work separately, up to `max_running` tasks at once.

### How a Delegation Ends

In Responses mode, the client sees each delegation's Responses events in `response.event`, and the voice reads each ending from those events, never from their text. The following table lists the endings and what the voice says for each one. The voice's lines are `agent_prompts.Lines` entries, and the `texts.*` lines are `agent_prompts.Core` entries, of the example's [`prompts.yaml`](../../src/examples/frontend_backend_live/prompts.yaml):

| Ending | Events | What the voice says |
| --- | --- | --- |
| Answered | Each task's `final_answer` message, then `response.completed` | The results, worded together and said when the caller is idle |
| Handed over | A newer request joined the work: its `response.in_progress` names this response in `previous_response_id`, and this response gets one `commentary` message (`texts.handover_line`, which names the newer delegation) and `response.completed` | Nothing of its own: the newer delegation carries the work |
| Stopped for a newer request | `texts.cancelled_line`, or `texts.unconfirmed_line` when the stop was not confirmed, as a `final_answer`, then `response.completed` | That line, as a result |
| Failed | `response.failed` with `rate_limit_exceeded`, `server_error`, or the failed attempt's own code, or `response.incomplete` with a reason such as `max_output_tokens` | `apology` when no tool ran, and `unclear` after a tool ran |
| Late | Function outputs for work that already ended get an empty response that ends with `response.completed` | `unclear` |
| Cancelled at close | `response.incomplete` with status `cancelled` and `incomplete_details: null` | Nothing: the session is closing |
| Lost | No terminal event: the stream ended, and the client gets a `response.failed` with `server_error` | `lost` |

The core writes no apology of its own and never retries work, so a client that stops on `response.failed` stops there. The following rules also apply in Responses mode:

- `response.create` continues each waiting response whose calls all have results. When it would continue none while results are still owed, it is refused with `function_call_outputs_required`.
- With the core's `delegation.result_format` set to `speak_display`, a result's screen-only detail comes before its answer, as an `mcp_call` item named `show`, with server label `delegation_core`, whose `output` is the display.

## Steer a Session

During the call, the client can steer the cascade with three appends. Each one carries a `content` of at most 500 tokens, which can be empty, and a `delegation_id`, which is `null` in Responses mode and can name an announced delegation in client mode:

- `session.instructions.append` cuts the line in progress, joins the talker's instructions, and has the talker decide again. Every later delegation also carries it, so each request repeats every instruction append. With empty content, it neither cuts the line nor adds an instruction.
- `session.thinking.append` adds silent context, which the talker sees and the next delegation carries. It also drops every lead-in that still waits for its delay. With empty content, it does only that, so a client that sets a delay can keep a delegation's acknowledgement unsaid.
- `session.commentary.append` adds a fact that the talker words and the voice says when the caller is idle. In a fallback of that wording, the voice says the fact word for word. With empty content, it only drops the lead-ins that still wait for their delay, as empty thinking does.

## Server Information and History

The following routes report on the server and on a session:

- **Server information.** `GET /v1/live/info` returns the voices, the talker's style (`frontend.talker_style`), the backend's default model (`backend.model`) and the models it serves (`backend_models`), the names of the default prompts (`frontend.prompt` and `backend.prompt`), and a `warnings` list. A frontend adapter can report configuration problems there, and the cascade adapter reports none.
- **Conversation history.** `GET /v1/live/sessions/{id}/history` returns what the talker and the backend hold now, with each system prompt split into parts named by source. The talker's last part is the runtime data its next call sends after the conversation: the open work and a short decision cue that restates the routing rules. A call that decides caller turns also quotes their words before the cue, as the caller's newest turn, and the history leaves that line out. Each talker message (`frontend.messages`) has a `kind`. With the `tools` talker style, a routing call (`call`) and its result (`call_result`) also keep their tool fields. Each line that the next delegation carries (`backend.pending`) is `{kind, text}`. In client mode, `backend` is `null`.

## Settings

### Session Settings

The following settings apply to a live session:

- **Voices.** The cascade speaks with the voice of its TTS service. Send `{"id": "<voice id>"}` to choose one of that service's voices.
- **Backend settings.** `delegation.responses.reasoning`, `text`, `service_tier`, and `parallel_tool_calls` reach a backend on the `openai-responses` provider as they are. A setting that the backend cannot honor, such as a hosted tool on a `chat-completions` backend, reaches the client as a nonfatal `responses_validation_failed` error.

### Server Settings

The `live.settings` block of the `frontend-backend-live` entry in [`examples_registry.yaml`](../../examples_registry.yaml) sets the following keys. The server reads them when it starts, and a key that the block leaves out keeps its default:

| Key | Default | What it sets |
| --- | --- | --- |
| `max_sessions`, `max_session_seconds` | `8`, `1800` | The sessions at once and the length of one session, in seconds. |
| `max_appends_per_minute` | `64` | The appends that one session accepts in any 60 seconds, counting `session.instructions.append`, `session.thinking.append`, and `session.commentary.append` together, empty ones included. An append over the limit gets a `rate_limit_exceeded` error, and the gateway neither passes it to the frontend adapter nor acknowledges it. |
| `ice_urls` | `[]` | More ICE servers for the WebRTC peer, for example `stun:stun.l.google.com:19302`, beside the TURN server that `TURN_USERNAME`, `TURN_PASSWORD`, and `TURN_URL` configure. |
| `max_delegate_rounds`, `allowed_hosts` | `2`, `[]` | The `POST /v1/live/delegate` rounds at once, and, with no token, the host names besides IP addresses and `localhost` that the routes answer to. An `allowed_hosts` entry is a host name only, without a port. An entry with a port never matches. |

An unknown key, a limit that is not a whole number of at least 1, or a list that is not a list of strings stops the server at startup.

The server reads the following environment variables when it starts:

| Variable | Default | What it sets |
| --- | --- | --- |
| `LIVE_ENABLED` | Unset | Whether the routes are served, as [When the Routes Are Served](#when-the-routes-are-served) describes. |
| `LIVE_API_TOKEN` | Unset | The bearer token that every route requires. When it is set, the server serves the routes unless `LIVE_ENABLED` turns them off. When `LIVE_ENABLED=true` serves them without a token, the server logs a warning at startup, and anyone who can reach it by an IP address, `localhost`, or a host name in `allowed_hosts` can use the routes. |
| `UVICORN_WORKERS` | `1` | The server's workers. The Live routes need one, as [When the Routes Are Served](#when-the-routes-are-served) describes. For a value that is not a whole number of at least 1, the server logs a warning and runs one worker. |

The example's own settings, such as its backend adapter, are the `config` block of the same entry, as [Configuration](../../src/examples/frontend_backend_live/README.md#configuration) in the example's README describes.

## Errors and Limits

When the frontend adapter reports a failure, the client gets an `error` with code `pipeline_error` if a client command failed or the session cannot go on, and an `info` notice with the same code if the session recovers and no command caused it. The gateway also logs each one as a warning.

When the text-to-speech returns no audio for a line, the client gets an `info` notice, and the voice says the line once more, except a progress line. When one sentence of a line gets no audio and the rest plays, the talker's context, `session.output_transcript.delta`, and the transcript that the next delegation carries leave that sentence out, and the voice does not say the line again. Three text-to-speech contexts in a row with no audio end the session, and the voice's single retry of a silent line counts as one of them. The client then gets an `error` with the message `3 consecutive TTS contexts completed with no audio`, followed by `session.closed` with reason `connection_lost`. In the bot of the repository UI, the voice handles a line with no audio the same way, and the same limit ends the call.

The following limits apply to every live session:

- **Long calls.** The talker's memory drops its oldest whole turn groups past `compact_at` of `memory.context_tokens`, and the backend's history drops its oldest requests past `delegation.context_tokens`.
- **Session start.** A command sent before `session.started` waits until the session is active, for up to 30 seconds. Caller audio that arrives before then can be dropped, so send audio after `session.started`.
- **WebSocket reply audio.** Reply audio arrives in `session.output_audio.delta` events of `AUDIO_OUT_10MS_CHUNKS` times 10 ms (100 ms by default). The last event of each reply is padded with silence, and 100 ms events of silence flow while no reply plays.
- **Captions.** Each sentence's `session.output_transcript.delta` arrives after that sentence's audio has played, so a caption follows its audio. GPT-Live's captions stream with its speech.
- **Recordings and forks.** Stored recordings and session forks are not available.
