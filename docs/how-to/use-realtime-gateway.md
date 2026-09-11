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
| `REALTIME_SERVICE_PLATFORM` | Selects the exact `cloud`, `server`, or `singlegpu` catalog section. Host-native runs default to `cloud`; Compose pins the value for each recipe. |
| `REALTIME_MCP_ALLOWED_SERVER_URLS` | JSON array of exact, trusted Streamable HTTP MCP URLs that the gateway can contact. |

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
| `nvidia/nemotron-realtime-frontend-backend` | Frontend/Backend Agent with trusted delegation functions |

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

Use standard `session.instructions` to customize agent behavior. A live
session update cannot change the selected model profile or its service route.

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
| `audio.input.format` / `audio.output.format` | 24 kHz signed PCM16, 8 kHz PCMU, or 8 kHz PCMA. |
| `audio.input.noise_reduction` | Only `null`. Audio preprocessing remains pipeline-owned. |
| `audio.input.turn_detection` | The advertised `server_vad` or `semantic_vad`, or `null` manual mode on eligible cascaded ASR pipelines. It becomes fixed after pipeline handoff. |
| `audio.input.transcription` | The profile's fixed producer or `null` to suppress public transcript events on eligible cascaded pipelines. |
| `audio.output.voice` | An exact voice from the selected TTS catalog. Unknown or custom voices are rejected. The first assistant audio locks the voice for the connection. |
| `audio.output.speed` | Only `1.0`. |
| `tools` | Trusted functions, client-owned functions, supported MCP definitions, or `[]` to disable tools. |
| `tool_choice` | `auto`, `none`, `required`, a named function, or a named active MCP operation. |
| `parallel_tool_calls` | Controls whether the model can request several tools in one response. |
| `truncation` | Available only for local cascaded profiles whose catalog enables exact tokenization. |
| `prompt` / `tracing` | `null` only. Stored prompt references and tracing modes are not supported. |
| `reasoning` | Recognized but unsupported. |

For `server_vad`, the initial update accepts `threshold`,
`prefix_padding_ms`, `silence_duration_ms`, `create_response`,
`interrupt_response`, and `idle_timeout_ms`. A Smart Turn pipeline advertises
`semantic_vad` and accepts its supported eagerness and Boolean controls. Use
the values advertised by `session.created` as the source of truth for the
selected profile.

Set `response.instructions`, `response.max_output_tokens`,
`response.output_modalities`, `response.audio`, `response.tools`,
`response.tool_choice`, or `response.parallel_tool_calls` for one response.
These values do not change the session defaults.

OpenAI Realtime does not define a general session `temperature` field.
Configure sampling in the trusted LLM service catalog.

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
  -> conversation.item.input_audio_transcription.completed
```

Wait for `input_audio_buffer.committed`, not only `speech_stopped`, before
treating a turn as actionable. When public transcription is enabled, its
terminal event precedes the response events for that turn.

Generic, Multilingual, and Frontend/Backend cascaded ASR profiles can select
manual mode in the initial session update:

```json
{
  "type": "session.update",
  "session": {"audio": {"input": {"turn_detection": null}}}
}
```

In manual mode, `append` buffers audio, `clear` discards uncommitted audio,
and `commit` submits one ASR turn. Wait for the transcript terminal, then send
`response.create`. The manual buffer holds at most 60 seconds. Direct Omni and
Omni Assistant Subagents do not support manual mode.

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
| Server | A trusted prompt catalog declares a registered Pipecat handler that the pipeline executes. |
| Delegate | A trusted server function sends work to another agent or backend and returns the result to the user-facing LLM. |
| MCP | The client declares a remote server; the gateway performs discovery and execution, while the client handles approval. |

An effective tool set can mix trusted functions, uniquely named client
functions, and MCP servers. A client cannot redefine a trusted function name.
The application must bind each client-owned name to its handler by exact name,
not by a prefix, description, or regular expression.

### Run a Client-Owned Function

Declare a client-owned function in an initial or live session update:

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
    "type": "mcp_approval_response",
    "approval_request_id": "approval_item_id",
    "approve": true
  }
}
```

Wait for Response A and every MCP call to reach a terminal event, then send
`response.create` for the final answer. Do not send `function_call_output` for
an MCP call.

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

Tool exceptions, timeouts, cancellations, and missing results produce a
correlated terminal result and a Realtime error. A provider failure ends the
active response instead of leaving the client waiting. MCP discovery and calls
also emit their native failed terminal events.

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
