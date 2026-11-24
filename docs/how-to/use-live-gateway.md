# Use the Live Gateway

The [Frontend/Backend Live example](../../src/nemotron_voice_agent/pipecat/examples/frontend_backend_live/README.md) serves voice sessions at `/v1/live` over WebSocket and WebRTC using GPT-Live event shapes. [GPT-Live documentation](https://developers.openai.com/api/docs/guides/live)

## When the Routes Are Served

Run a `frontend-backend-live` [deployment profile](../01-getting-started.md) with its default `EXAMPLE_SELECTION` and one Uvicorn worker. Set `LIVE_API_TOKEN` in `.env` before starting the server. On a trusted network, you can serve the routes without a token by setting `LIVE_ENABLED=true`. Set `LIVE_ENABLED=false` to disable the routes.

### Authentication

If you set a token, send `Authorization: Bearer <LIVE_API_TOKEN>` on every request, including WebSocket connections. The gateway checks browser origins against the request host. A token-free deployment accepts an IP address, `localhost`, or a name in [`allowed_hosts`](#server-settings).

## Routes

Use the route that matches your transport or app operation.

| Route | Purpose |
| --- | --- |
| `WS /v1/live/sessions` | Start a call and exchange audio and events on the primary WebSocket. |
| `POST /v1/live/sessions` | Start a WebRTC call by sending a Session Description Protocol (SDP) offer. |
| `WS /v1/live/sessions/{id}/attach` | Send commands and observe new events from an active call on a sideband connection. |
| `GET /v1/live/sessions/{id}/history` | Inspect the active call's frontend and backend context. |
| `GET /v1/live/info` | Discover settings, limits, prompt names, and supported backend model IDs in `backend_models`. |
| `POST /v1/live/delegate` | Run a stateless backend round with the caller history and tools your app supplies. |

## Connect a Client

Use HTTPS and `wss://` with TLS, or HTTP and `ws://` when `PIPELINE_TLS=false`.

### WebSocket

Send `session.start` as the first JSON text message. The following example uses [Responses delegation](#delegation-modes). Replace the backend model with one from `GET /v1/live/info`.

```json
{
  "type": "session.start",
  "session": {
    "model": "live-1",
    "delegation": {
      "type": "responses",
      "responses": {"model": "<backend model>"}
    }
  }
}
```

Wait for `session.started`, and then send base64 audio in the `audio` field of `session.input_audio.append`. Read base64 reply audio in the `delta` field of `session.output_audio.delta`. Set `session.audio.format` at startup to choose PCM at 16 or 24 kHz, or G.711 `audio/pcmu` or `audio/pcma` at 8 kHz. The default is 24 kHz PCM. Send `session.close` to end the call.

### WebRTC

Create a peer connection with an audio track and an event data channel. Send its SDP offer to `POST /v1/live/sessions`. Include the same `session` settings as for WebSocket, and let SDP negotiate audio.

```json
{
  "transport": {"type": "webrtc", "sdp": "<SDP offer>"},
  "session": {
    "model": "live-1",
    "delegation": {
      "type": "responses",
      "responses": {"model": "<backend model>"}
    }
  }
}
```

Apply `transport.sdp` from the HTTP 201 response as the peer's answer. The response also includes `session.id`. Wait for `session.started` on the data channel before sending caller audio. Send JSON events on the data channel and audio on the media track. When the peers need a relay, configure [a TURN server](enable-turn-server.md#live-sessions).

### Connect a Pipecat App

`OpenAILiveLLMService` handles the WebSocket events. This example assumes a local server with TLS disabled. Register any function handlers and schemas in your Pipecat pipeline as usual.

```python
from pipecat.services.openai.live.llm import OpenAILiveLLMService
from pipecat.services.openai.responses.llm import OpenAIResponsesLLMService

llm = OpenAILiveLLMService(
    api_key="<LIVE_API_TOKEN>",
    base_url="ws://localhost:7860/v1/live/sessions",
    delegation=OpenAILiveLLMService.ResponsesDelegation(
        settings=OpenAIResponsesLLMService.Settings(model="<backend model>"),
    ),
)
```

For a token-free local server, the client accepts any nonempty value for `api_key`. To run a Pipecat worker as your app's backend, use `OpenAILiveLLMService.ClientDelegation`.

## Delegation Modes

Choose a mode at startup with `session.delegation.type`. Supply `session.instructions` to set the frontend prompt for the call. If you omit it or send blank instructions, the deployment uses its configured prompt, `generic` by default.

### Server Delegation

Set `type: responses` and provide `delegation.responses.model`. Put your backend instructions and the function tools your app authorizes for this session in `delegation.responses.instructions` and `delegation.responses.tools`. The server runs the delegated work and forwards its events inside `response.event`. Your app executes its registered functions.

The backend can provide more than one caller-facing message, including questions and status updates, while a response is still open. Use `response.completed`, `response.failed`, or `response.incomplete` to tell when that response ends. If it ends with app calls to process, follow the continuation procedure below. To choose the adapter or worker limit, refer to [Choose a Backend Adapter](../../src/nemotron_voice_agent/pipecat/examples/frontend_backend_live/README.md#choose-a-backend-adapter).

With `core.delegation.result_format: speak_display`, screen-only detail arrives before the spoken answer as an `mcp_call` item named `show`, with the default `server_label` of `delegation_core`. Read the display from its `output`.

### Client Delegation

Set `type: client`, or omit `delegation`. When your app receives `session.delegation.created`, run the work in your app. Send the answer as `session.commentary.append` with the announced `delegation.id` as `delegation_id`. You can call `POST /v1/live/delegate` explicitly for a stateless backend round. Send the whole history in `input`, then run any returned calls and include their results in the next round.

### Continue a Shared Response

For each pending function call, send its output with the same `call_id`. The output can be a string, a JSON data array, or an array of Responses-compatible text, image, or file parts.

```json
{
  "type": "response.item.create",
  "item": {
    "type": "function_call_output",
    "call_id": "<original call_id>",
    "output": "{\"available\": true}"
  }
}
```

Send `{"type": "response.create"}` after returning the complete batch. Shared responses send one tool batch at a time. If you receive `function_call_outputs_required`, return the outstanding results before continuing. The built-in Core uses `delegation.client_tool_timeout_seconds` to limit the wait after a response ends and queues its remaining calls. The default is 60 seconds.

You can also queue a typed user message with `response.item.create` and submit it with `response.create`. With an open tool batch, typed input accompanying the complete batch goes to the job that owns it. Send late tool results as a separate outputs-only batch with their original `call_id` values. The Core can record them as outcome evidence for ended work. Read the [reserved work tool names](../../src/nemotron_voice_agent/pipecat/examples/frontend_backend_live/delegation_core/README.md#build-and-configure-the-core) before naming your own tools.

### Resume a Dropped WebSocket Call

A custom client can resume the same Responses-mode WebSocket call in server memory. To enable version 1 reconnect, send `reconnect: {"version": 1}` next to `session` in `session.start` and set `session.delegation.reconnect_grace_seconds` to a positive value up to 120 seconds. The default is 0 seconds. With custom prompts, include the [reconnect attachment template](configure-prompts.md#frontendbackend-live-prompt-catalog).

1. Save the token and generation from `session.reconnect.ready` after `session.started`. Keep the token private.
2. Send `response.item.create`, `response.create`, and the instruction, thinking, and commentary appends on the primary socket with a stable `operation_id`. The `session.operation.receipt` reports gateway admission. Resend the same payload with the same ID to retrieve its recorded or unknown status.
3. If the connection drops, open a new primary WebSocket at the same endpoint with the same authentication, host, and origin before the grace period expires. Use the saved token and generation in the first message.

    ```json
    {
      "type": "session.start",
      "resume": {
        "version": 1,
        "token": "<protected token>",
        "generation": 1,
        "request_id": "<stable resume request identifier>"
      }
    }
    ```

4. Wait for `session.resumed` before sending audio. Save its new token and generation. The server discards audio read during restoration, keeps the call's audio clock, and retains mute until `session.input_audio.unmute`.

Use valid UTF-8 text of 1 through 128 characters for `operation_id` and `request_id`. An exact retry of a successful claim returns `session.resume.receipt` until that claim's original grace deadline. Only the first successful claim opens an attachment. An unavailable claim returns `resume_unavailable`.

Use `pending.results` and `pending.unclear_calls` in `session.resumed` to check the state of earlier work. Return known tool results with their original call IDs and stable operation IDs. Verify an uncertain action's outcome before issuing it again. A clean close, grace expiry, or server restart ends the reconnect lease. The gateway ends the call if the retained event journal exceeds 512 records or 2 MiB.

### Work That Outlives the Call

Set `store: true` in a Responses session to let eligible work continue after hang-up. This requires an `llm-agent` or `harness` backend with `max_running` above 1. The backend can keep work that can finish without the caller or the app.

Save `delegation.responses.conversation` from the session object in `session.started`. `session.closed` includes it if the server kept the conversation. To continue, start a new session with `store: true` and that ID in `delegation.responses.conversation` after the earlier call closes. The caller can then ask about the job. Treat the conversation ID as sensitive and associate it with the verified caller. Retention is in server memory and ends when the server restarts. The `delegation.after_call_seconds` default is two days.

## Steer a Session

Send an append with a `content` string of up to 500 tokens. Set `delegation_id` to `null` in Responses mode. In client mode, set it to the relevant announced ID or `null` for general session context.

| Event | Effect |
| --- | --- |
| `session.instructions.append` | Adds an instruction and interrupts active speech so the frontend can respond to the change. |
| `session.thinking.append` | Adds silent context and clears delayed acknowledgements. |
| `session.commentary.append` | Gives the frontend something to say when the caller is idle. |

## Settings

### Session Settings

The following settings apply to a live call.

- **Voice.** The deployment uses its configured text-to-speech (TTS) voice for built-in voice names. Set `audio.output.voice` to `{"id": "<voice id>"}` to select a voice from the deployed TTS service. An unknown ID falls back to the default.
- **Acknowledgement.** Set `delegation.acknowledgement_delay_ms` from 0 to 800 ms. The default is 0. A caller interruption or a result can suppress a pending acknowledgement.
- **Backend.** Send a `session.update` with a `delegation.responses` patch to change settings for later work. Open work keeps its settings until its next `response.create`. The `llm` and `llm-agent` adapters use Chat Completions or Responses according to the model's `api` in [`services.yaml`](../../services.yaml). Provider support determines which hosted tools and generation settings take effect. The harness runs its own model. Check `responses_validation_failed` for settings that the chosen backend rejects.

### Server Settings

Set operator limits under `live.settings` in the `pipecat.examples.frontend-backend-live` [registry entry](../../examples_registry.yaml). Restart the server to apply changes.

| Key | Default | Purpose |
| --- | --- | --- |
| `max_sessions` | `8` | Concurrent voice sessions. |
| `max_session_seconds` | `1800` | Maximum duration of a call in seconds. |
| `max_appends_per_minute` | `64` | Combined limit for instruction, thinking, and commentary appends per call. |
| `max_delegate_rounds` | `2` | Concurrent `POST /v1/live/delegate` requests. |
| `ice_urls` | `[]` | Additional ICE server URLs for WebRTC. |
| `allowed_hosts` | `[]` | Additional host names, without ports, for a token-free deployment. |

## Errors and Limits

- A speech service or frontend model that is still starting can return `server_error`. Retry after the service is ready.
- WebSocket JSON events and WebRTC setup requests are capped at 1 MiB. Bad session settings return a structured error.
- The frontend and backend keep bounded context. Configure the frontend with `frontend.cascade.memory.context_tokens` and the backend with `core.delegation.context_tokens`. [Tune Pipeline Performance](tune-pipeline-performance.md)
- Output transcripts follow audio. In per-sentence synthesis, the gateway records a sentence after synthesis completes and the transport accepts its audio. In stitched synthesis, it applies those checks to the full line.

For route refusals, backend validation errors, or missing audio, refer to [Frontend/Backend Live Troubleshooting](../06-troubleshooting.md#frontendbackend-live).
