# Use the Live Gateway

## Goals

`/v1/live` lets live session clients, such as Pipecat apps or your own app, talk to the [Frontend/Backend Live example](../../src/examples/frontend_backend_live/README.md) over WebRTC or WebSocket, with the prompts that they supply. The gateway serves the live session protocol of OpenAI's GPT-Live API on the example's cascade. It is not a drop-in replacement for the hosted API. The gateway imports no Pipecat: it reaches the example's frontend adapter and delegation core only through the ports in [`ports.py`](../../src/gateways/live/ports.py).

## How This Differs from GPT-Live

The following table compares OpenAI's hosted GPT-Live API with this gateway:

| | GPT-Live | This gateway |
|--|--|--|
| Runtime | A model that OpenAI hosts | The example's cascade of speech recognition, a talker model, and text-to-speech, with the delegation core for the delegated work |
| A `delegation.responses` change in `session.update` | Applies to every later delegation | Applies from the next delegation that starts new work. A delegation that joins open work keeps that work's settings until the next `response.create`. |
| A typed message in `response.item.create` | Queues behind the work already running | Steers open work, as a spoken request does |
| An unrelated request while work is open | Runs as its own delegation | Joins the open work with the `llm` adapter or at `max_running` 1, so the caller hears both answers together. Above 1, the conductor can start separate work. |
| Captions | Stream with the speech | Each sentence's `session.output_transcript.delta` arrives after its audio has played |
| Stored recordings and session forks | Available | Not available |
| `store: true` | Keeps the voice session for forks and recordings, and backend work ends with the session | Keeps backend work that outlives the call, and its records, for a later session, as [Work That Outlives the Call](#work-that-outlives-the-call) describes |

## Routes

The gateway serves the following routes:

| Route | What it serves |
| --- | --- |
| `POST /v1/live/sessions` | A WebRTC session: the client's SDP offer and settings in, the answer out |
| `WS /v1/live/sessions` | A primary WebSocket session: `session.start` first, then commands and base64 audio events |
| `WS /v1/live/sessions/{id}/attach` | A sideband connection that observes a session and sends trusted commands |
| `GET /v1/live/sessions/{id}/history` | What the session's talker and backend hold now |
| `GET /v1/live/info` | The server's voices and session limit, and the example's description |
| `POST /v1/live/delegate` | One backend round for a client app that does its own delegated work |

An HTTP request to any other path under `/v1/live` answers 404, and a WebSocket connection there is refused with HTTP 403.

### When the Routes Are Served

The server serves the Live routes only when its selected example has a `live` block in [`examples_registry.yaml`](../../examples_registry.yaml), as `frontend-backend-live` does. With `EXAMPLE_SELECTION=all`, the first listed example is selected, so no Live route is served.

On a `frontend-backend-live` deployment, the server serves the Live routes when `LIVE_API_TOKEN` is set. Without a token, it serves them only when you set `LIVE_ENABLED=true`, for example on a trusted network. When `LIVE_ENABLED` is set, it decides: `1`, `true`, `yes`, or `on` serve the routes, and any other value turns them off. A server with more than one worker (`UVICORN_WORKERS`) serves no Live route, because a session lives in one process.

### Authentication

Set `LIVE_API_TOKEN` to require a bearer token on every Live route. The routes refuse a browser page from another origin with `403 Unexpected browser origin`.

Without a token, the server logs a warning at startup. The routes then answer only requests addressed to an IP address, `localhost`, or a host name in the `allowed_hosts` [server setting](#server-settings). This check stops a webpage on another website from using the routes through a host name that resolves to your server.

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

Use `ws://` when `PIPELINE_TLS=false`; otherwise `wss://`.

## Delegation Modes

A live session client sends its `instructions`, which are the talker's prompt, and a `delegation` block, whose `type` selects the delegation mode for the whole session. A session that sets no instructions, or blank ones, gets the `generic` talker prompt of the example's [`prompts.yaml`](../../src/examples/frontend_backend_live/prompts.yaml), which names no domain. The modes work as follows:

- **Responses mode** (`delegation.type: responses`). The [server delegation](#server-delegation) passes each delegation to the [delegation core](../../src/examples/frontend_backend_live/delegation_core/README.md). The core runs the backend work with the settings in `delegation.responses`: the backend `model`, which is required, and optional `instructions` and function `tools`. The backend's function calls stream to the client as `response.event`. The client returns each result as a `function_call_output` item with `response.item.create`, and then sends `response.create` to continue. The results must arrive within the core's `delegation.client_tool_timeout_seconds` (60 seconds), counted from the end of the response with their calls, or the core stops the work. A backend task can repeat its newest call with the same function and arguments after the client answered it. The core then answers the repeat instead of sending it again, unless a request was admitted since or the work has taken more than one request. It answers with that output and a note (`texts.repeated_call_output` in `agent_prompts.Core`).
- **Client mode** (`delegation.type: client`, or no `delegation` block). The server runs no backend and announces each delegation with `session.delegation.created`. The client app does the work, answers with appends that name the delegation, and sends the result back as a `session.commentary.append`. The session refuses `response.item.create` and `response.create`.

When the talker delegates a turn, the lead-in that it words in the same decision is the delegation's spoken acknowledgement. The voice says it after the session's `delegation.acknowledgement_delay_ms`, unless a result, the delegation's end, the caller starting to speak again, or commentary or thinking comes first. The delay is at most `800`, and `0`, the default, says it at once.

### Server Delegation

For each session, the server delegation opens a conversation in the core and sends each request to it in order, with the session's current `delegation.responses` settings. The session fails to start when the core cannot create the conversation within 10 seconds. A start that continues a conversation waits up to 25 seconds by default instead (`delegation.close_grace_seconds` plus twice `delegation.stop_timeout_seconds`), while the core ends the earlier call. A start that comes while the earlier session still closes its frontend adapter is refused, so start it after the earlier session has closed. The server delegation also does the following:

- It holds the client's call results and typed messages from `response.item.create`, at most 512 results and 16 messages, and sends them at `response.create`. When the core refuses a request as invalid, such as one with two results for one call, the client gets `invalid_value`, and the server delegation drops what that request carried.
- It checks the `delegation.responses` settings in the background with `POST /v1/responses/input_tokens`, and reports a setting that the backend refuses as a nonfatal `responses_validation_failed` error. The session keeps the setting, so a model that the backend does not serve fails each delegation, except on the harness, which runs its agent's own model.

The core places each request by its backend adapter's rule, as [Choose a Backend Adapter](../../src/examples/frontend_backend_live/README.md#choose-a-backend-adapter) describes.

### How a Delegation Ends

In Responses mode, the client sees each delegation's events in `response.event`, and the voice reads each ending from those events, never from their text. The following table lists the endings. The voice's lines are `agent_prompts.Lines` entries, and the `texts.*` lines are `agent_prompts.Core` entries, of the example's [`prompts.yaml`](../../src/examples/frontend_backend_live/prompts.yaml):

| Ending | Events | What the voice says |
| --- | --- | --- |
| Answered | Each task's `final_answer` message, then `response.completed` | The results, worded together when the caller is idle |
| Handed over | Another request joined the work: its `response.in_progress` names this response in `previous_response_id`, and this response ends with a `commentary` message (`texts.handover_line`) and `response.completed` after its running backend calls end. A task's `final_answer` that comes first ends it answered instead | Nothing of its own: the other delegation carries the work |
| Stopped for a newer request | `texts.cancelled_line`, or `texts.unconfirmed_line` when the stop was not confirmed, as a `final_answer`, then `response.completed` | That line |
| Failed | `response.failed` with `rate_limit_exceeded`, `server_error`, or the attempt's own code, or `response.incomplete` with a reason such as `max_output_tokens` | `apology` when no tool ran, else `unclear` |
| Timed out | The function outputs did not arrive within `delegation.client_tool_timeout_seconds`, so the core stops the work and writes nothing more, unless another request that a task took in carries the work | `unclear`, or nothing when another request carries the work |
| Late | Function outputs for work that already ended get an empty response that ends with `response.completed` | `unclear`, or nothing more when the delegation already ended |
| Cancelled at close | `response.incomplete` with status `cancelled` and `incomplete_details: null` | Nothing: the session is closing |
| Lost | No terminal event: the stream ended, and the client gets a `response.failed` with `server_error` | `lost` |

The core writes no apology of its own and never retries work, so a client that stops on `response.failed` stops there. The following rules also apply in Responses mode:

- `response.create` continues each waiting response whose calls all have results. When it would continue none while results are still owed, it is refused with `function_call_outputs_required`.
- With the core's `delegation.result_format` set to `speak_display`, a result's screen-only detail comes before its answer, as an `mcp_call` item named `show`, with server label `delegation_core`, whose `output` is the display.
- A task can ask the caller a question while its work goes on. The question is a `final_answer` message while the response is still `in_progress`, and more items and the task's answer can follow. A `final_answer` is one result for the caller; only the response's terminal event ends the delegation. When the caller's reply joins the work while this response is still open, its `response.in_progress` names this response in `previous_response_id`, and this delegation still ends answered.
- The core's work tools use the names `cancel_work` and `ask_caller`, `show` with `speak_display`, and `continue_after_call` where [work can outlive the call](#work-that-outlives-the-call). A function in `delegation.responses.tools` with one of these names is refused, together with the work tool.

### Work That Outlives the Call

A session that sets `store: true` lets long backend work go on after the caller hangs up. The core offers each task the `continue_after_call` work tool when the backend runs more than one task at once: with `llm-agent` or `harness` above `max_running: 1`. A task calls it when its work will take longer than the caller should wait and needs nothing more from the caller or the app. The caller hears the task's message as a `final_answer` while the response stays `in_progress`.

When the session closes, the core keeps the conversation while its records list such work, open or among the newest ended work. The other work stops as at any close, and so does such work that waits for the app's function outputs. Every open response ends `cancelled`. With nobody on the call, the app's functions and `ask_caller` do not run, and the work's events reach no client. A request that the caller added to the work during the call does not run if the work ends before a task takes it in, so the caller asks for it again in a later call.

`session.started` names the core's conversation in `delegation.responses.conversation`. `session.closed` names it only when the core kept the conversation, so keep the id for a later call only then. A start that names the id and ends before `session.started`, by your own close or with an `error` other than `invalid_value`, leaves the conversation as it was. In that case, keep the id and try again. To continue, start a session with `store: true` and the id in `delegation.responses.conversation`. When the caller asks, the backend answers from the work's records. The following rules apply:

- `delegation.responses.conversation` is set at the start only, and it needs `store: true`.
- An id that is unknown, has expired, or belongs to a session that is still open fails the start with `invalid_value` on `session.delegation.responses.conversation`. Over WebRTC, the offer is answered, but the connection never opens and no event arrives. After a hang-up, retry after the earlier session has closed.
- The core keeps the conversation in this server process's memory for `delegation.after_call_seconds` (2 days) after its last call, and for that long again after its work ends. When that time runs out, the conversation closes, and work still running stops. A restart or deploy loses it and stops its work.
- `continue_after_call` is refused when such work would take every task slot of the backend, or when `delegation.max_after_call_conversations` (8) other conversations already have such work running. The task then finishes its work during the call.
- Whoever holds the id hears the work's results, so pass it only for a caller that your app has identified. Without `LIVE_API_TOKEN`, the id alone is enough.
- A Pipecat 1.15 app sets `store=True` in `OpenAILiveLLMService.Settings`, and its `on_session_started` handler gets the session that names the id. To continue, the app passes the id in the `extra` of its `ResponsesDelegation` settings, for example `OpenAIResponsesLLMService.Settings(model="<backend model>", extra={"conversation": "<id>"})`. Pipecat 1.15 does not pass `session.closed` to the app, so the app learns whether the core kept the conversation only from the next start with the id.

## Steer a Session

During the call, the client can steer the cascade with three appends. Each one carries a `content` of at most 500 tokens, which can be empty, and a `delegation_id`, which is `null` in Responses mode and can name an announced delegation in client mode:

- `session.instructions.append` cuts the line in progress, joins the talker's instructions, and has the talker decide again. Every later delegation also carries it. With empty content, it neither cuts the line nor adds an instruction.
- `session.thinking.append` adds silent context, which the talker sees and the next delegation carries. It also drops every lead-in that still waits for its delay, so empty thinking keeps a delayed acknowledgement unsaid.
- `session.commentary.append` adds a fact that the voice says when the caller is idle, in the talker's words, or word for word when the wording fails. With empty content, it only drops the lead-ins that still wait for their delay.

## Server Information and History

The following routes report on the server and on a session:

- **Server information.** `GET /v1/live/info` returns the Live API's voice names and the talker's style (`frontend.talker_style`). It also returns the backend's default model (`backend.model`) and the models it serves (`backend_models`), the names of the default prompts (`frontend.prompt` and `backend.prompt`), and a `warnings` list.
- **Conversation history.** `GET /v1/live/sessions/{id}/history` returns what the talker (`frontend.messages`, each with a `kind`) and the backend (`backend`, `null` in client mode) hold now. It returns each system prompt split into parts named by source. `backend.busy` says whether a delegation is open, and each line that the next delegation carries (`backend.pending`) is `{kind, text}`.

## Settings

### Session Settings

The following settings apply to a live session:

- **Voices.** The cascade speaks with its TTS service's voice, whatever built-in voice name a session sends. To choose another of the service's voices, send `{"id": "<voice id>"}`. As in the [Realtime gateway](use-realtime-gateway.md), the server checks the ID against the voices that the service lists. An ID that the list does not hold falls back to a default voice, and the server logs a WARNING that names the ID. When the list is empty or cannot be fetched, the cascade speaks the voice of the service's `services.yaml` entry, or the ID itself when the entry names no voice. The server then logs a WARNING that names the ID, unless the ID is the entry's voice.
- **Backend settings.** The `llm` and `llm-agent` adapters call a model over chat completions unless the model's `services.yaml` entry sets `api: openai-responses`. On either API, the model's settings profile turns **Reasoning** on or off, as [Reasoning (Thinking) On/Off](configure-llm.md#reasoning-thinking-onoff) describes. A `reasoning.effort` of `none` turns **Reasoning** off when the profile has it. On chat completions, the backend ignores any other effort and `reasoning.summary` without an error. A hosted tool, a `service_tier` other than `auto` or `default`, or `text.verbosity` gets a nonfatal `responses_validation_failed` error there. On the Responses API, `delegation.responses.reasoning`, `text`, `service_tier`, `parallel_tool_calls`, and the function and hosted `tools` reach the server as they are, except that `reasoning.effort` is `none` while **Reasoning** is off. The server decides what it honors, and it can ignore a hosted tool without an error. The core passes the client only function calls. On either API, any other tool whose calls the client runs or approves gets a nonfatal `responses_validation_failed` error, and the backend leaves it out of the model's request. The `custom`, `computer`, `apply_patch`, `namespace`, and `programmatic_tool_calling` tools are always such tools. So are a `shell` tool without a hosted container `environment` (`container_auto` or `container_reference`), a `tool_search` tool with `execution: client`, and an `mcp` tool whose `require_approval` is not `never`. A `tool_choice` that only such a tool matches becomes `auto`. With the `llm` and `llm-agent` adapters, an `allowed_tools` tool choice offers the backend model only the tools that it lists and the core's work tools, such as `cancel_work`. On every adapter, a call to a function that the list leaves out never reaches the client: the core answers it with `texts.unlisted_output`, an `agent_prompts.Core` entry.

### Server Settings

The `live.settings` block of the `frontend-backend-live` entry in [`examples_registry.yaml`](../../examples_registry.yaml) sets the following keys. The server reads them when it starts, and a key that the block leaves out keeps its default:

| Key | Default | What it sets |
| --- | --- | --- |
| `max_sessions`, `max_session_seconds` | `8`, `1800` | The sessions at once and the length of one session, in seconds. |
| `max_appends_per_minute` | `64` | The appends of all three kinds that one session accepts in any 60 seconds. An append over the limit gets a `rate_limit_exceeded` error and is dropped. |
| `ice_urls` | `[]` | More ICE servers for the WebRTC peer, for example `stun:stun.l.google.com:19302`, beside the TURN server that `TURN_USERNAME`, `TURN_PASSWORD`, and `TURN_URL` configure. |
| `max_delegate_rounds`, `allowed_hosts` | `2`, `[]` | The `POST /v1/live/delegate` rounds at once, and, with no token, the host names, without a port, that the routes answer to besides IP addresses and `localhost`. |

An unknown key or an invalid value stops the server at startup. The server also reads `LIVE_ENABLED`, `LIVE_API_TOKEN`, and `UVICORN_WORKERS` when it starts, as [When the Routes Are Served](#when-the-routes-are-served) and [Authentication](#authentication) describe.

The example's own settings, such as its backend adapter, are the `config` block of the same entry, as [Configuration](../../src/examples/frontend_backend_live/README.md#configuration) in the example's README describes.

## Errors and Limits

When the frontend adapter reports a failure, the client gets an `error` with code `pipeline_error` if a client command failed or the session cannot go on, and an `info` notice with the same code if the session recovers and no command caused it. The gateway also logs each one as a warning.

When the text-to-speech returns no audio for a line, the client gets an `info` notice, and the voice says the line once more, except a progress line. A sentence with no audio in a line that otherwise plays is left out of the transcripts. Three text-to-speech contexts in a row with no audio, the retry included, end the session with an `error`, `3 consecutive TTS contexts completed with no audio`. A `session.closed` with reason `connection_lost` follows. The repository UI bot handles silent lines the same way.

The following limits apply to every live session:

- **Long calls.** The talker's memory drops its oldest whole turn groups past `compact_at` of `memory.context_tokens`, and the backend's history drops its oldest requests past `delegation.context_tokens`.
- **Session start.** Before a session starts, the server checks each speech service and the talker's model that does not run on NVIDIA Cloud. A session started before one of them is ready is refused with `server_error` and a message such as "asr is still starting. Try again in a moment." A WebSocket gets that `error` event and a close with code 1008, and `POST /v1/live/sessions` answers HTTP 503. A command sent before `session.started` waits up to 30 seconds, and earlier audio can be dropped, so send audio after `session.started`.
- **Event and settings size.** A WebSocket event longer than 1,048,576 characters gets an `error` and ends the connection. `delegation.responses` settings longer than about 1,048,576 characters of JSON get `invalid_value`.
- **WebSocket reply audio.** Reply audio arrives in `session.output_audio.delta` events of `AUDIO_OUT_10MS_CHUNKS` times 10 ms (100 ms by default). The last event of each reply is padded with silence, and 100 ms events of silence flow while no reply plays.
