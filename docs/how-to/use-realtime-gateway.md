# Use the OpenAI Realtime-Compatible Gateway

`WS /v1/realtime` connects OpenAI Realtime clients to the NVIDIA voice-agent
pipelines. The gateway implements an OpenAI Realtime-compatible WebSocket
subset. It is not an OpenAI-hosted Realtime model or a complete replacement
for the OpenAI Realtime service.

The gateway supports the following primary workflows:

- Send text or audio and receive text or audio.
- Update instructions and supported session settings before or during a
  connection.
- Select a trusted, server-configured pipeline and model profile.
- Run trusted server tools, client-owned function tools, and remote Model
  Context Protocol (MCP) tools.
- Run the shared application server or an API-only standalone server.
- Collect response usage and available NVIDIA pipeline metrics.

The current scope is WebSocket only. WebRTC, Session Initiation Protocol (SIP),
sideband control, OpenAI-managed MCP connectors, Secure MCP tunnels, and
deferred MCP loading are not implemented.

## Start the Server

The shared application exposes both the browser application and the Realtime
endpoint. Run it from the repository root:

```bash
uv run python src/server.py --host 0.0.0.0 --port 7860
```

To run only the Realtime API, start the standalone server:

```bash
uv run python src/realtime_server.py \
  --host 0.0.0.0 \
  --port 7860 \
  --workers 4
```

The standalone server exposes these routes:

| Route | Purpose |
| --- | --- |
| `GET /health` | Process-liveness check |
| `POST /v1/realtime/client_secrets` | Short-lived client-secret creation |
| `WS /v1/realtime` | OpenAI Realtime-compatible WebSocket |

Configure these deployment settings in `.env` when needed:

| Setting | Purpose |
| --- | --- |
| `REALTIME_API_KEY` | Protects the WebSocket and authorizes client-secret creation. Leave it unset only for unauthenticated local development. |
| `REALTIME_ALLOW_PROXY_BEARER` | Allows a trusted proxy's authenticated outer Bearer credential to coexist with a browser `ek_` client secret. The default is `false`. |
| `REALTIME_SERVICE_PLATFORM` | Selects the exact `cloud`, `server`, or `singlegpu` catalog section. Host-native runs default to `cloud`; Compose pins the value for each recipe. |
| `REALTIME_MCP_ALLOWED_SERVER_URLS` | JSON array of exact, trusted Streamable HTTP MCP URLs that the gateway can contact. |

For local development, `.env` supplies defaults for variables that are not
already present in the process environment. Existing process variables remain
authoritative. In a container deployment, set values such as
`REALTIME_SERVICE_PLATFORM` through Docker Compose, Helm, or NVCF instead of
relying on the image's `.env` file. Restart the application after changing a
deployment variable.

TLS is enabled by default. Set `PIPELINE_TLS=false` only for an isolated local
test. When TLS is enabled without explicit certificate paths, the server uses
a local self-signed certificate.

Check process liveness with:

```bash
curl -fk https://localhost:7860/health
```

The health response does not prove that the selected automatic speech
recognition (ASR), large language model (LLM), or text-to-speech (TTS) service
is ready. The first actionable Realtime event checks the services required by
the selected profile.

## Select a Realtime Model Profile

The standard `model` field selects a complete, server-owned profile from
`examples_registry.yaml`. A profile binds a pipeline, prompt, and catalog
selectors. Clients cannot pass provider URLs, credentials, filesystem paths,
or arbitrary provider model identifiers.

The repository defines these profiles:

| Realtime model ID | Pipeline |
| --- | --- |
| `nvidia/nemotron-realtime` | Generic cascaded pipeline with trusted server tools |
| `nvidia/nemotron-realtime-client-tools` | Generic pipeline intended for client-owned functions |
| `nvidia/nemotron-realtime-multilingual` | Multilingual cascaded pipeline |
| `nvidia/nemotron-realtime-omni` | Direct Omni pipeline with optional client-owned functions |
| `nvidia/nemotron-realtime-omni-subagents` | Omni multi-agent pipeline without client-defined functions |
| `nvidia/nemotron-realtime-frontend-backend` | Airline Frontend/Backend Agent with trusted delegation functions |
| `nvidia/nemotron-realtime-generic-frontend-backend` | Generic Frontend/Backend Agent with server and client tools planned by its hidden Thinker |

Select a model in the WebSocket URL:

```text
wss://voice.example.com/v1/realtime?model=nvidia%2Fnemotron-realtime-client-tools
```

To add a profile, declare a stable public identifier and trusted selectors:

```yaml
realtime_models:
  acme/realtime-support:
    label: Acme Realtime Support
    pipeline_mode: generic-assistant
    default: false
    selectors:
      prompt_key: generic_assistant_without_tools
      llm_id: registry-default
      asr_id: registry-default
      tts_id: registry-default
```

`registry-default` selects the first configured default for that service slot
within `REALTIME_SERVICE_PLATFORM`. It does not probe endpoints or switch to a
different platform. Use `platform_overrides` when a profile needs a different
catalog key on one platform.

A profile can also declare `session_defaults`. The registry loader validates
this field and accepts only `turn_detection.silence_duration_ms`, an integer
from 0 through 60,000. The value applies only when the client sets no
`silence_duration_ms` for `server_vad`; a client value is echoed unchanged. The
`nvidia/nemotron-realtime-generic-frontend-backend` profile sets 800 ms so
that a spelled identifier with pauses stays in one turn:

```yaml
realtime_models:
  nvidia/nemotron-realtime-generic-frontend-backend:
    # label, pipeline_mode, default, and selectors omitted
    session_defaults:
      turn_detection:
        silence_duration_ms: 800
```

Use standard `session.instructions` to customize agent behavior. The Generic
Frontend/Backend profile can update its instructions, tools, and tool choice
while connected. A live session update cannot change the selected model profile
or its service route.

## Authenticate and Connect

The shared server keeps RTVI and Realtime on separate paths:

| Path | Protocol |
| --- | --- |
| `WS /api/ws` | Pipecat Real-Time Voice/Video Inference (RTVI) |
| `WS /v1/realtime` | OpenAI Realtime-compatible events |

Use `ws://` only when `PIPELINE_TLS=false`. Use `wss://` for remote or
credential-bearing connections.

When `REALTIME_API_KEY` is set, send either the master key or a valid `ek_`
client secret as a Bearer credential. Do not expose the master key to a
browser. Browser clients can offer the short-lived secret through the OpenAI
Realtime subprotocol form:

```javascript
const socket = new WebSocket(
  "wss://voice.example.com/v1/realtime",
  ["realtime", `openai-insecure-api-key.${clientSecret}`],
);
```

### Use a Trusted Proxy Authorization Boundary

By default, the gateway rejects a WebSocket handshake when `Authorization`
and the browser subprotocol contain different credentials. This fail-closed
behavior prevents an untrusted intermediary from replacing either credential.

Set `REALTIME_ALLOW_PROXY_BEARER=true` only when NVIDIA Cloud Functions (NVCF)
or another trusted reverse proxy authenticates its own outer `Authorization`
Bearer credential and forwards that header to the application. The browser
must also send a valid `ek_` client secret through
`openai-insecure-api-key.<secret>` in `Sec-WebSocket-Protocol`.

In this opt-in mode, the application treats the outer Bearer credential as
proxy-owned and independently verifies the browser client secret against
`REALTIME_API_KEY`. An invalid, expired, or missing browser client secret still
fails authentication. When no browser secret is present, the existing Bearer
authentication rules still apply.

This setting affects only the WebSocket handshake. It does not change
`POST /v1/realtime/client_secrets`, which continues to require the exact
`REALTIME_API_KEY` Bearer credential. Keep the setting disabled when the
application receives client credentials directly.

The server emits both handshake events when a connection succeeds:

```text
session.created
conversation.created
```

The Realtime endpoint does not send an automatic welcome response. Send an
initial `session.update` before the first audio append, conversation item, or
`response.create`. If you omit it, the first actionable event starts the
profile defaults.

The following Python example connects through the OpenAI SDK:

```python
import asyncio
import os

from openai import AsyncOpenAI


async def main():
    client = AsyncOpenAI(
        api_key=os.environ["REALTIME_API_KEY"],
        websocket_base_url="wss://voice.example.com/v1",
    )
    async with client.realtime.connect(model="nvidia/nemotron-realtime-client-tools") as connection:
        assert (await connection.recv()).type == "session.created"
        assert (await connection.recv()).type == "conversation.created"

        await connection.send(
            {
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "instructions": "Answer briefly.",
                    "output_modalities": ["text"],
                },
            }
        )
        assert (await connection.recv()).type == "session.updated"


asyncio.run(main())
```

A Realtime connection lasts at most 60 minutes. Reconnect to create another
session.

### Create a Client Secret

Create short-lived credentials on a trusted application backend:

```bash
curl -sS https://voice.example.com/v1/realtime/client_secrets \
  -H "Authorization: Bearer $REALTIME_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "expires_after": {"anchor": "created_at", "seconds": 600},
    "session": {
      "type": "realtime",
      "model": "nvidia/nemotron-realtime-client-tools",
      "instructions": "Answer briefly.",
      "output_modalities": ["text"]
    }
  }'
```

`expires_after.seconds` accepts 10 through 7,200 and defaults to 600. The
response contains an `ek_` secret and its `expires_at` time. The expiry limits
when a connection can open; it does not shorten an established session. The
response is non-cacheable and omits private provider and MCP configuration.

### Test from a Browser

The repository includes a minimal local test client at `realtime_test_client/`.
Start it from the repository root while the server is running:

```bash
python realtime_test_client/proxy.py
```

Then open `http://localhost:8765`. The page connects to the Generic
Frontend/Backend profile by default and provides **Connect**, **Start mic**,
**Send text**, and **Clear** controls, a conversation view, the negotiated
session configuration, and a raw event log. It sends 24 kHz mono PCM16 audio
and plays streamed `response.output_audio.delta` events.

The client is a proxy rather than a page the server hosts, for two reasons:

- Browsers cannot set an `Authorization` header on a WebSocket.
- The browser fallback offers an `ek_` client secret as a WebSocket
  subprotocol. That secret encodes the whole bound session, and the Generic
  Frontend/Backend session exceeds the client-secret size cap. The gateway
  rejects it with "The bound Realtime session is too large for a WebSocket
  client secret." Refer to `_CLIENT_SECRET_MAX_CHARS` in
  `src/realtime/auth.py`.

The proxy therefore holds `REALTIME_API_KEY` server side and relays a plain
`ws://` browser connection to the gateway's TLS WebSocket. It also terminates
the gateway's self-signed certificate, so the browser raises no certificate
warning. The proxy reads `REALTIME_API_KEY` from the environment, or from the
repository `.env` file when the variable is not already set.

Configure the proxy with these options:

| Option | Default | Purpose |
| --- | --- | --- |
| `--port` | `8765` | Local port that serves the page and the browser WebSocket |
| `--upstream` | `wss://localhost:7860` | Realtime gateway to relay to |
| `--model` | `nvidia/nemotron-realtime-generic-frontend-backend` | Realtime model id to request |
| `--env-file` | repository `.env` | File to read `REALTIME_API_KEY` from |

This is a local development tool. It holds the master key in the proxy process
and serves the browser over plain HTTP on the loopback interface. Do not expose
it beyond localhost, and do not use it as a pattern for a production browser
application. A production application mints short-lived client secrets on its
own trusted backend so the master key never reaches browser code.

## Configure a Session

`session.update` is transactional. A rejected update leaves the previous
session unchanged. The following example configures text output, instructions,
a response limit, and the input format:

```json
{
  "type": "session.update",
  "session": {
    "type": "realtime",
    "instructions": "Answer in one short sentence.",
    "output_modalities": ["text"],
    "max_output_tokens": 256,
    "audio": {
      "input": {"format": {"type": "audio/pcm", "rate": 24000}}
    }
  }
}
```

The gateway supports these session fields:

| Field | Supported behavior |
| --- | --- |
| `model` | Reports the selected server-owned profile. A live update can only repeat it. |
| `instructions` | Replaces the public instructions for future responses while retaining pipeline controls and conversation history. |
| `output_modalities` | Exactly `["audio"]` or `["text"]`. Both accept audio input; text output bypasses TTS. |
| `max_output_tokens` | `"inf"` or an integer from 1 through 4,096. A trusted catalog can impose a lower provider ceiling. |
| `audio.input.format` / `audio.output.format` | Mono signed PCM16 at 8, 16, or 24 kHz; fixed 8 kHz PCMU; or fixed 8 kHz PCMA. The default is PCM16 at 24 kHz. Omit `rate` from PCMU and PCMA format objects. |
| `audio.input.noise_reduction` | Only `null`. Audio preprocessing remains pipeline-owned. |
| `audio.input.turn_detection` | Exactly the automatic type advertised by `session.created` (`server_vad` or `semantic_vad`), or `null` manual mode on eligible cascaded ASR pipelines. It becomes fixed after pipeline handoff. |
| `audio.input.transcription` | The profile's fixed model, an initial catalog-backed language selector, or `null` to suppress public transcript events on eligible cascaded pipelines. The model and language become fixed after pipeline handoff. |
| `audio.output.voice` | An exact voice from the selected TTS catalog. Unknown or custom voices are rejected. The first assistant audio locks the voice for the connection. |
| `audio.output.speed` | Only `1.0`. |
| `tools` | Trusted functions, client-owned functions, supported MCP definitions, or `[]` to disable tools. |
| `tool_choice` | `auto`, `none`, `required`, a named function, or a named active MCP operation. |
| `parallel_tool_calls` | Controls whether the model can request several tools in one response. |
| `truncation` | Available only for local cascaded profiles whose catalog enables exact tokenization. |
| `prompt` / `tracing` | `null` only. Stored prompt references and tracing modes are not supported. |
| `reasoning` | Recognized but unsupported. |

Before output audio begins, you can update the session voice while no audio
response owns a frozen output snapshot. If an active or pending audio response
has already captured its voice, a conflicting `session.update` returns
`immutable_field`; retry after `response.done`. Text-only responses do not
block voice updates. After the first `response.output_audio.delta`, the voice
remains locked for the connection.

For `server_vad`, the initial update accepts a numeric `threshold` from 0
through 1. `prefix_padding_ms` and `silence_duration_ms` accept integers from 0
through 60,000. `idle_timeout_ms` accepts an integer from 5,000 through 30,000
or `null`. The update also accepts Boolean `create_response` and
`interrupt_response` values. A profile that advertises `semantic_vad` accepts
only `eagerness: "auto"`. Eligible cascaded profiles
accept either Boolean value, while Omni Smart Turn profiles require both values
to remain `true`. Use `session.created` as the source of truth for the selected
profile and automatic type. When the client omits `silence_duration_ms`, the
session uses the profile's `session_defaults` value, if any, and otherwise the
gateway default.

Set `response.instructions`, `response.max_output_tokens`,
`response.output_modalities`, `response.audio`, `response.tools`,
`response.tool_choice`, `response.parallel_tool_calls`, `response.metadata`,
`response.conversation`, or `response.input` for one response. These values do
not change the session defaults.

Set `response.conversation` to `"auto"`; out-of-band responses with `"none"`
are not supported. `response.input` accepts item references, inline text
messages, and function-call context. It does not accept inline audio or image
input. The gateway runs one active response on the default conversation. A
second unrelated `response.create` returns `response_in_progress`. The Generic
Frontend/Backend profile defers a client request that meets a pipeline-started
response instead. For details, refer to
[Run the Generic Frontend/Backend Profile](#run-the-generic-frontendbackend-profile).
`response.prompt` and `response.reasoning` are recognized but unsupported.

OpenAI Realtime does not define a general session `temperature` field.
Configure sampling in the trusted LLM service catalog.

The [OpenAI Realtime API reference](https://developers.openai.com/api/reference/resources/realtime/client-events)
defines 24 kHz as the canonical `audio/pcm` rate. This gateway also accepts
8 kHz and 16 kHz PCM16 as NVIDIA extensions. PCMU and PCMA remain fixed at
8 kHz, and their format objects omit `rate`. Use 24 kHz PCM16 for strict OpenAI
client portability.

The wire rate is independent of the Pipecat pipeline rate. The gateway
resamples input to the pipeline's 16 kHz rate before voice activity detection
and automatic speech recognition. It resamples the pipeline's 22.05 kHz TTS
output to the selected wire rate. These conversions apply at the Realtime
boundary and do not change the underlying pipeline configuration.

## Send Text and Audio Turns

Create a text item, wait for its terminal acknowledgement, and request a
response:

```json
{
  "type": "conversation.item.create",
  "item": {
    "type": "message",
    "role": "user",
    "content": [{"type": "input_text", "text": "What is two plus two?"}]
  }
}
```

```json
{"type": "response.create"}
```

For audio, Base64-encode chunks in the session's selected input format and send
`input_audio_buffer.append`. In automatic mode, VAD commits the turn and can
create its response. The input lifecycle is:

```text
input_audio_buffer.speech_started
  -> input_audio_buffer.speech_stopped
  -> input_audio_buffer.committed
  -> conversation.item.added
  -> conversation.item.done
  -> conversation.item.input_audio_transcription.completed | .failed
```

Wait for `input_audio_buffer.committed`, not only `speech_stopped`, before
treating a turn as actionable. When public transcription is enabled, its
terminal `.completed` or `.failed` event precedes the response events for that
turn.

On eligible cascaded ASR profiles, `server_vad` commits at the detected speech
stop and includes the configured `prefix_padding_ms` in that turn's audio.
`semantic_vad` can keep consecutive speech segments in one turn until Smart
Turn classifies the utterance as complete. The gateway retains later input in
arrival order while the preceding ASR turn finishes, so audio and transcripts
cannot cross item boundaries. A response starts only after the matching ASR
turn reaches its terminal state.

An automatic acoustic segment can end successfully without recognized text.
This is not an ASR provider failure. With `semantic_vad`, the logical turn can
continue and use text recognized in earlier or later segments. If public
transcription is enabled and the complete automatic turn is empty, it completes
with `transcript: ""`. The gateway does not create a failed response for that
empty turn.

Automatic VAD turns stream into ASR and are not limited to 60 seconds of total
speech. The gateway independently bounds retained prefix audio and audio waiting
for the ASR provider, so long turns do not create an unbounded memory queue.

Setting `audio.input.transcription` to `null` suppresses the public
`conversation.item.input_audio_transcription.*` events. It does not disable the
private ASR result that supplies text to the cascaded model or the terminal
barrier that orders the response. The publication setting is captured when an
input item starts, so a live null/non-null update applies to the next item and
does not change an in-flight item's event contract.

Generic, Multilingual, and Frontend/Backend cascaded ASR profiles can select
manual mode in the initial session update:

```json
{
  "type": "session.update",
  "session": {"audio": {"input": {"turn_detection": null}}}
}
```

In manual mode, `append` buffers audio, `clear` discards uncommitted audio, and
`commit` submits exactly that accepted audio as one ASR turn without adding
endpoint audio. You can send `response.create` immediately after `commit`; the
gateway queues it until the matching transcription completes or fails. The
manual buffer holds at most 60 seconds. Direct Omni and Omni Assistant
Subagents do not support manual mode.

An explicit manual commit with no recognized text remains an `empty_transcript`
failure. With public transcription enabled,
`conversation.item.input_audio_transcription.failed` precedes `response.done`
for an accepted response. The response status is `failed` with error code
`input_audio_transcription_empty`.

The gateway supports tail-appended system and user text items. It can retrieve
non-audio items, delete completed items whose model-context owner is still
available, and truncate completed assistant audio at a published checkpoint.
It does not retain input or output PCM for item retrieval.

Each response has its own identifier and ends as `completed`, `cancelled`,
`failed`, or `incomplete`. Text responses emit
`response.output_text.delta` and `response.output_text.done`. Audio responses
emit audio and transcript deltas, then terminal item and response events.
`response.done` for audio follows transport drain.

`response.cancel` cancels the active response and discards queued media. VAD
interruption does the same. An accepted long-running backend task is separate;
cancel it through the application tool that owns that task.

## Configure Tools

Realtime uses the standard function type for direct client tools and
delegation. Function names do not have special routing semantics.

| Owner | Execution boundary |
| --- | --- |
| Client | The client declares and executes a function, returns `function_call_output`, and requests the final response. |
| Server | A trusted prompt catalog declares a registered Pipecat handler that the pipeline executes internally. |
| Delegate | A trusted server function sends work to another agent or backend and returns the result to the user-facing LLM, entirely inside the pipeline. |
| MCP | The client declares a remote server; the gateway performs discovery and execution, while the client handles approval. |

An effective tool set can mix trusted functions, uniquely named client
functions, and MCP servers. A client cannot redefine a trusted function name.
The application must bind each client-owned name to its handler by exact name,
not by a prefix, description, or regular expression.

A server or delegate call is the pipeline's own internal mechanics, and it
never reaches the client. No `function_call` item ever announces it, no
published field names it or its owner, and the response it runs inside closes
with an empty output array -- indistinguishable on the wire from a response
that legitimately produced nothing. The pipeline still executes the call,
applies its result to the model's own context, and can speak a short
acknowledgement from it ahead of the grounded final answer, but none of that
requires the client to do anything. A standard OpenAI Realtime client needs no
special handling to avoid answering a call it never sees, and no configuration
naming the trusted tools to filter them out.

For the Generic Frontend/Backend profile the trusted names are `call_backend`
and `cancel_backend`. That set is pinned by `DelegateToolNameContractTests`,
and the fact that neither the call nor its result ever reaches the wire is
pinned by `DelegateCallWireContractTests`, both in
`tests/unit/test_generic_fba_trusted_tool_routing.py`, and by
`ObserverServerToolLifecycleTests` in
`tests/unit/test_realtime_observer_lifecycle.py`. Treat a change to any of
them as a breaking wire-contract change.

### Run a Client-Owned Function

A client-owned call still uses the ordinary two-response shape: the response
that carries the call (call it Response A here), and a later response with a
different response ID that carries the answer (Response B). Declare a
client-owned function in an initial or live session update. The Generic
Frontend/Backend profile accepts instruction, tool, and tool-choice changes
while connected. It prepares the matching Thinker policy and Talker capability
prompt before it commits the session update:

```json
{
  "type": "session.update",
  "session": {
    "type": "realtime",
    "tools": [
      {
        "type": "function",
        "name": "browser_lookup",
        "description": "Look up current information.",
        "parameters": {
          "type": "object",
          "properties": {"query": {"type": "string"}},
          "required": ["query"],
          "additionalProperties": false
        }
      }
    ],
    "tool_choice": "auto"
  }
}
```

The server publishes the call but does not execute it. After Response A
finishes, execute the exact function name and return the original `call_id`:

```json
{
  "type": "conversation.item.create",
  "item": {
    "type": "function_call_output",
    "call_id": "call_from_response_a",
    "output": "{\"ok\":true,\"weather\":\"sunny\"}"
  }
}
```

Send `response.create` after the output item. You can send both events
back-to-back without waiting for the output acknowledgement; the gateway starts
Response B after Response A and the output are terminal. Response A and
Response B use different response IDs. Correlate the operation with `call_id`.

When the session forces a function with `required` or a named choice, set the
Response B override to `tool_choice: "none"`. This lets the model answer from
the result without forcing the same call again. The session choice still
applies to the next user turn.

The client-tool deadline defaults to 120 seconds. A timeout produces a
structured failure and keeps the connection available for a recovery response.
Tool-call state does not survive disconnection.

### Run the Generic Frontend/Backend Profile

Use `nvidia/nemotron-realtime-generic-frontend-backend` when the client owns
domain functions but the server must retain bounded planning and grounded
speech. This profile applies the following ownership boundary:

- The Lightning Talker sees only `call_backend` and `cancel_backend`. This
  remains true for its canonical context, live session updates, and each
  response-scoped context.
- `cancel_backend` withdraws only this assistant's own in-flight delegated
  request, for example when the user says stop, never mind, or ignore that
  while a `call_backend` call is still running. It never performs a
  cancellation the user asks for as a task, such as cancelling an order,
  booking, or reservation; the Talker sends requests like that to
  `call_backend` like any other domain request.
- The Talker receives a bounded capability digest with descriptions, but no
  client function identifiers or argument schemas. It is deterministic by
  default; model mode uses that same Lightning instance.
- The Super Thinker receives `session.instructions` verbatim as a distinct
  system message. Its trusted planner rules and complete server/client tool
  contract remain in a separate server-owned system message.
- Python validates the complete plan before the first side effect. Server tools
  run in process; client tools suspend the plan and use standard
  `function_call_output` events.
- Independent client calls from one planning round share one response. The
  backend waits up to `GENERIC_CLIENT_TOOL_TIMEOUT_SECONDS`, which defaults to
  25 seconds, before it returns one grounded failure.
- The backend retains one parked plan per session. Cancellation, a newer
  generation, a late result, or a repeated failed call cannot revive stale
  work.
- A session that declares its own client tools does not use the built-in
  server tools, such as weather or web search. The Talker capability digest
  and the Thinker tool contract list only the client tools.
- When the user only acknowledges or asks about progress while delegated work
  runs, a new `call_backend` call can continue that work instead of restarting
  it. The user hears one answer. `call_backend` and its `task` argument never
  reach the client. Set `FRONTEND_BACKEND_FRONTEND_VERDICT` to `false` to
  disable this.
- The Thinker receives a bounded history of earlier delegations in the session,
  including what was said, the tool calls it made, and how each result reached
  the user. Assistant text follows client truncation and deletion in the
  conversation. Set `FRONTEND_BACKEND_BACKEND_HISTORY` to `false` to disable
  this.
- The backend screens spoken identifiers in client read-tool calls, renders
  confirmation questions from a fixed template, and can issue a confirmed
  write after a bare "yes" without a new plan. For these safeguards and their
  switches, refer to
  [Generic Client-Tool Safeguards](../../src/examples/frontend_backend_agent/README.md#generic-client-tool-safeguards).

When `FRONTEND_BACKEND_REALTIME_TOOL_ROUNDS` is enabled, which is the default,
this profile applies the following protocol options. Other profiles keep the
standard behavior.

- User speech does not cancel a pipeline-owned client-tool round that the
  client already received. The round stays parked until its outputs arrive or
  it times out.
- A `function_call_output` for a call that was cancelled or timed out is
  acknowledged with `conversation.item.added` and `conversation.item.done`
  instead of an `error`. The output joins the conversation, but no handler
  acts on it. Only the first output for that call is accepted.
- A client `response.create` that arrives while a pipeline-started response is
  starting or active waits until that response finishes, instead of failing
  with `response_in_progress`.
- A second overlapping client request, a duplicate output, a function name
  mismatch, an unknown call, and a closed session keep their errors.

With the switch set to `false`, user speech during a parked client-tool round
cancels that round, and the next `call_backend` call starts new work.

`session.created` and `session.updated` for the Frontend/Backend profiles
include a read-only `session.nvidia.agent` object that describes what the
session runs with. Use it to tell whether two recorded conversations ran the
same build and settings. A client that echoes the object back in
`session.update` is not rejected; the server ignores the echoed values.

| Field | Content |
| --- | --- |
| `agent_git_sha` | The `AGENT_GIT_SHA` build argument of the image, or `unknown` |
| `domain` | The Frontend/Backend domain, such as `generic` or `airline` |
| `thinker` | `llm_id`, `model_id`, `max_tokens`, `reasoning_budget`, and, when the catalog sets it, `reasoning` |
| `flags` | The effective value of each `FRONTEND_BACKEND_*` feature switch |
| `session_tools_sha256` | A SHA-256 fingerprint of the session's function tools in declared order |

The Talker supplies a short, query-grounded `filler_text` in its original
`call_backend` arguments. `call_backend` itself never reaches the client, so
that response closes with an empty output array; only after it closes can the
pipeline emit the precomputed filler text as its own ordinary response,
without another model inference or a new user turn. The grounded final answer
follows as a later, ordinary response. A rejected or missing filler remains
silent; the runtime never substitutes a static phrase. From the client's point
of view this is nothing more than an assistant that sometimes speaks twice for
one user turn -- a brief acknowledgement, then the real answer -- with no
tool-call machinery visible in between.

The profile regenerates its capability digest whenever
`session.instructions`, `session.tools`, or `session.tool_choice` changes.
`REALTIME_CAPABILITY_MODE=static`, the default, renders it immediately from the
effective tool catalog and makes no additional model call. Set the mode to
`model` to ask the same Nemotron Lightning instance for schema-constrained
summaries with reasoning disabled, temperature zero, a 350-token output ceiling,
and a 30-second deadline. Model mode caches validated output in a bounded
process cache and, when configured, Redis for 24 hours. The cache key covers the
effective instructions, server and client tools, tool choice, profile, and
summary schema version.

Python uses exact names to validate completeness, but omits those identifiers
from the rendered Talker digest. It rejects omissions, renamed or invented
tools, policy text, and internal model details. A cache, timeout, provider,
schema, or validation failure
degrades to the deterministic digest; it never prevents session startup or a
live policy update. The Thinker instruction, Talker digest, and tool validators
still commit together.

Do not interpret this source implementation as a qualified deployment. The
32,768-token model context and reduced sequence limits in the dedicated Viking
evaluation values must pass model-startup, live Realtime, and benchmark gates
before promotion.

### Run Long-Running Delegation

Do not keep one Realtime tool call open for an unbounded backend job. Have a
submission function return a durable task identifier promptly, and expose
ordinary status, cancellation, and answer functions. The application owns the
task store, authorization, retries, and backend transport.

For example, an application can declare functions named `delegate_task`,
`check_status`, `cancel_task`, and `answer`, or use any other unique names. The
gateway treats all of them as ordinary functions. The backend can use HTTP, a
queue, Agent Communication Protocol (ACP), agent-to-agent (A2A), or another
transport without changing the Realtime contract.

Use the `nvidia/nemotron-realtime-frontend-backend` profile when the deployed
pipeline owns the included airline backend agent. Use
`nvidia/nemotron-realtime-client-tools` when the client application owns that
delegation boundary.

### Run an MCP Tool

The gateway supports remote Streamable HTTP MCP servers. Add the exact server
URL to `REALTIME_MCP_ALLOWED_SERVER_URLS`, and declare it as a session or
response tool:

```json
{
  "type": "mcp",
  "server_label": "company_tools",
  "server_url": "https://mcp.example.com/mcp",
  "allowed_tools": ["lookup_record"],
  "require_approval": "always"
}
```

When approval is required, answer the `mcp_approval_request` item by its ID:

```json
{
  "type": "conversation.item.create",
  "item": {
    "id": "approval_response_id",
    "type": "mcp_approval_response",
    "approval_request_id": "approval_item_id",
    "approve": true
  }
}
```

Wait for the response carrying the mcp_call items and every MCP call to reach
a terminal event, then send `response.create` for the final answer. Do not
send `function_call_output` for an MCP call. Unlike a server or delegate call,
an MCP call is always published; the client is the MCP host and owns approval.

The gateway accepts MCP Bearer credentials through `authorization` and other
string-valued headers through `headers`. It keeps them in private connection
state and redacts them from public session and response objects. Use
authenticated TLS whenever clients can supply remote endpoints or credentials.

### Validate Tool Arguments

The gateway preflights bounded JSON Schemas before changing an active tool set.
It validates trusted server and delegate arguments before their registered
handlers run. Invalid trusted arguments produce a correlated failure and
`invalid_tool_arguments`.

The gateway delivers model-generated client-function arguments to the owning
client without enforcing that schema. The client must validate arguments and
return one correlated success or failure output.

The model-serving endpoint owns model-specific tool parsing. The gateway uses
typed OpenAI-style calls and does not parse raw `<tool_call>` or XML text. The
`qwen3_coder` values in the checked-in vLLM configuration name a parser
implementation; they do not select or deploy a Qwen model.

## Handle Errors

Protocol failures use the standard Realtime `error` shape:

```json
{
  "type": "error",
  "event_id": "server_event_id",
  "error": {
    "type": "invalid_request_error",
    "code": "unsupported_capability",
    "message": "The requested capability is unavailable",
    "event_id": "rejected_client_event_id"
  }
}
```

Malformed or unsupported client events use `invalid_request_error`. Provider,
pipeline, trusted-tool, and gateway lifecycle failures use `server_error`.
Recoverable request errors keep the socket open. Fatal pipeline or correlation
failures close it with code `1011`.

For a client-owned tool, an exception, timeout, cancellation, or missing
result produces a correlated terminal result and a Realtime error. A server or
delegate tool's failure is logged server-side instead; it never reaches the
client, consistent with the rest of that call's lifecycle never reaching the
client. An LLM or TTS provider failure that owns an active response ends that
response instead of leaving the client waiting. MCP discovery and calls also
emit their native failed terminal events.

On cascaded profiles, an ASR failure with no usable transcript emits the
item-scoped `conversation.item.input_audio_transcription.failed` event when
public transcription is enabled for that item. If ASR already produced usable
text, the transcript instead completes with that text. When public transcription
is suppressed, the gateway emits a `server_error`. An accepted manual response
request, or automatic response creation with a free response slot, ends as
`failed` with empty output. An unrelated response that already occupies the slot
is not cancelled or replaced.

If TTS finishes an announced audio item without producing wire audio, the
gateway emits `tts_provider_error` and ends that response as `failed`.

When a required service fails its readiness check, the gateway emits a
`services_not_ready` error. The message names the failing service and a short
reason (`starting`, `timeout`, or `unavailable`), for example:

```text
One or more required Realtime services are not ready: TTS (timeout)
```

A Realtime session retries a readiness check that failed quickly, with
backoff, within `REALTIME_READINESS_RETRY_SECS`. The default is `6` seconds.

## Collect Metrics and Run Tests

When Pipecat supplies usage, `response.done` includes standard response token
counts. The gateway publishes available latency and throughput values through
`nvidia.metrics.updated`; it does not place RTVI messages on the Realtime
socket.

The scaling benchmark keeps RTVI and OpenAI Realtime as independent protocol
modes. Realtime runs can use audio or text input, audio or text output,
automatic or manual turn handling, authentication, and client-owned tools.
Refer to [Scaling Perf](../../benchmarking_tools/scaling-perf/README.md) for
commands, result fields, and concurrency testing.

Run the opt-in live compatibility suite against a configured gateway:

```bash
OPENAI_REALTIME_WS_BASE=ws://127.0.0.1:7860/v1 \
  RUN_REALTIME_COMPAT=1 \
  uv run pytest tests/integration/test_realtime_openai_sdk_compat.py -v -s
```

Refer to [Integration and Live
Tests](../../tests/integration/README.md#openai-realtime-compatibility) for the
required environment variables and test scope.

Use the upstream OpenAI documentation as the protocol reference:

- [Realtime API](https://developers.openai.com/api/docs/guides/realtime)
- [WebSocket transport](https://developers.openai.com/api/docs/guides/voice-websockets)
- [Realtime conversations](https://developers.openai.com/api/docs/guides/realtime-conversations)
- [Voice activity detection](https://developers.openai.com/api/docs/guides/realtime-vad)
- [Realtime MCP](https://developers.openai.com/api/docs/guides/realtime-mcp)
- [Realtime client events](https://developers.openai.com/api/reference/resources/realtime/client-events)
