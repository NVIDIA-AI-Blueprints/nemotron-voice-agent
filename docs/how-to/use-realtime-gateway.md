# Use the OpenAI Realtime–Compatible Gateway

## Goals

`WS /v1/realtime` lets OpenAI Realtime–shaped clients (SDK or raw WebSocket) talk to this blueprint without a separate Realtime model stack.

- **Wire compatibility** — Accept and emit OpenAI Realtime JSON events (GA names, plus pre-GA aliases where OpenAI renamed them).
- **Same pipelines as the UI** — Drive the cascaded ASR → LLM → TTS pipelines already used by `WS /api/ws` (Pipecat RTVI).
- **Nemotron config surface** — Map Realtime session fields that translate onto the cascade; expose NVIDIA-only knobs under `session.nvidia`.

It is a **protocol gateway**, not a drop-in clone of OpenAI’s hosted Realtime endpoint.

## How this differs from OpenAI Realtime

| | OpenAI Realtime | This gateway |
|--|-----------------|--------------|
| Runtime | Hosted multimodal Realtime models | Cascaded Nemotron ASR → LLM → TTS |
| URL | OpenAI Realtime WebSocket | `WS /v1/realtime` on this server |
| Voices | OpenAI voice names (`alloy`, …) | Magpie / catalog voice ids (unknown ids warn and fall back to catalog default) |
| `session.model` | Selects a Realtime model | Ignored (logged only) |
| Tools | Client-registered schemas + model-native calling | Client schemas are ignored; server catalog tools run internally |
| Welcome | Client-driven | Optional Nemotron welcome gate (RTVI parity): client text / `response.create` rejected until first assistant `response.done` |
| Turn detection | Several modes and tunable VAD | Server VAD only; no push-to-talk or VAD tuning |
| Live `session.update` | Broad session mutation | Voice and PCM updates only; turn mode is immutable |
| Transport | WebSocket, WebRTC, PCM16, and G.711 | WebSocket PCM16 only |

Invalid session config that would break audio (bad PCM format/rate, unsupported turn detection, or text-only output) returns a Realtime `error` and **keeps the socket open** for retry. Fields that do not translate to the cascade (including `session.tools`) are ignored.

| Path | Protocol |
|------|----------|
| `WS /api/ws` | Pipecat RTVI |
| `WS /v1/realtime` | OpenAI Realtime–shaped WebSocket |

## Connect with the OpenAI Python SDK

```python
import asyncio
from openai import AsyncOpenAI

async def main():
    client = AsyncOpenAI(
        api_key="unused",  # suitable only for an unsecured local deployment
        websocket_base_url="ws://127.0.0.1:7860/v1",
    )
    async with client.realtime.connect() as conn:
        assert (await conn.recv()).type == "session.created"
        await conn.send({"type": "session.update", "session": {"type": "realtime"}})
        print(await conn.recv())  # session.updated

asyncio.run(main())
```

Use `ws://` when `PIPELINE_TLS=false`; otherwise `wss://`.

Authentication, authorization, quotas, and rate limits are deployment-layer responsibilities; the gateway does not provide them.

### Optional session overrides

```json
{
  "type": "session.update",
  "session": {
    "type": "realtime",
    "instructions": "Be brief.",
    "voice": "Magpie-Multilingual.EN-US.Aria",
    "temperature": 0.8,
    "audio": {
      "input": {
        "format": { "type": "audio/pcm", "rate": 24000 },
        "turn_detection": { "type": "server_vad" },
        "transcription": { "model": "client-selector-is-ignored" }
      },
      "output": { "format": { "type": "audio/pcm", "rate": 24000 } }
    },
    "nvidia": { "pipeline_mode": "generic-assistant" }
  }
}
```

Client tools and `function_call_output` items are not supported. Catalog tools run inside the server and appear through `session.nvidia.server_tools` and the observational `nvidia.tool.started` / `nvidia.tool.completed` events.

## Field map

OpenAI top-level fields map onto the cascade when they have a Nemotron equivalent.

| OpenAI field | Maps to |
|--------------|---------|
| `instructions` | `prompt_content` (system / prompt text for the cascade) |
| `voice` / `audio.output.voice` | `tts_voice_id` (soft-validated against TTS catalog; unknown → default) |
| `tools` | ignored (catalog tools via `prompt.id` / `prompt_key` only) |
| `tool_choice` | `tool_choice` (applies when catalog tools are enabled) |
| `max_output_tokens` | `max_tokens` |
| `temperature` | LLM `temperature` |
| `prompt.id` | `prompt_key` |
| `model` | ignored |
| `audio.*.format` / rate | transport resample |
| `audio.input.turn_detection` | must be `{ "type": "server_vad" }`; `null` and other modes are rejected |
| `audio.input.transcription` | selector accepted; requested model ignored because pipeline ASR supplies transcription events |
| `output_modalities` / `modalities` | must include audio; a requested text modality is a compatibility no-op and does not enable multimodal output |

Defaults when omitted: `pipeline_mode=generic-assistant`, `prompt_key=generic_assistant_without_tools`.

### `session.nvidia` (no OpenAI equivalent)

Nemotron-only catalog / routing keys: `pipeline_mode`, `llm_id`, `asr_id`, `tts_id`, `asr_language_code`, `model_id`, `extra_params`, ASR/TTS `*_model`, `tts_synthesis_mode`.

The public session object also exposes the selected prompt's catalog tools as read-only `nvidia.server_tools`.

Prefer OpenAI fields for prompt and generation settings: use `prompt.id`, `instructions`, `temperature`, and `max_output_tokens`. Do not put `tts_voice_id`, `system_prompt`, `max_tokens`, or `temperature` under `nvidia`.

Do **not** send `base_url`, `asr_server`, `tts_server`, or `*_function_id` under `nvidia` — those are ignored. Endpoints resolve from the selected catalog service ids only (prevents client-controlled SSRF).

Public `session.updated` echoes catalog ids only (no internal ASR/TTS endpoints or function ids).

Changing `tts_id` / `tts_model` at connect time re-lists voices for that TTS selection only when that routing key is not already cached (same catalog path as RTVI `GET /api/tts-config`), then re-resolves `voice` against the cached list.

## Realtime API reference (v1)

### Client → server

| Event | Notes |
|-------|--------|
| `session.update` | Configure session; required before pipeline handoff. Invalid audio/modality config → `error`, socket stays open. |
| `input_audio_buffer.append` | Base64 PCM chunk |
| `input_audio_buffer.commit` | Accepted as a compatibility no-op in server-VAD mode; automatic commit follows `speech_stopped` |
| `input_audio_buffer.clear` | Reset client buffer accounting and emit `input_audio_buffer.cleared`; audio already streamed to VAD cannot be withdrawn |
| `conversation.item.create` | User text items; `function_call_output` is rejected |
| `conversation.item.truncate` | Barge-in while a response is in progress. Idle truncate is ignored. Does not emit `conversation.item.truncated`. |
| `response.create` | Start a model turn. Empty `response: {}` is accepted; non-empty per-response overrides are rejected. |
| `response.cancel` | Cancel the in-progress response. Idle cancel is a no-op. |

Conversation item delete/retrieve operations are not supported. There are no response-level instruction, tool, modality, audio, or generation overrides.

### Server → client

| Event | Notes |
|-------|--------|
| `session.created` / `session.updated` | Session object (OpenAI-shaped + optional `nvidia`) |
| `error` | Realtime-shaped error; common codes include `invalid_session`, `services_not_ready`, `unsupported_live_session_update`, `unsupported_response_override`, `item_rejected_pre_intro`, `response_create_rejected_pre_intro`, `invalid_item`, `invalid_truncate` |
| `input_audio_buffer.*` | Server-VAD speech / commit / clear events |
| `conversation.item.*` | Item created; cascaded ASR input transcription. `conversation.item.truncated` is not emitted. |
| `response.*` | Response lifecycle, audio, and audio-transcript events |
| `nvidia.tool.started` / `nvidia.tool.completed` | Observation of internally executed catalog tools |

GA clients receive `response.output_audio.*` and `response.output_audio_transcript.*`. Clients that negotiate the beta dialect receive the corresponding `response.audio.*` and `response.audio_transcript.*` names. Audio content parts do not emit `response.output_text.*`.

### Audio

- WebSocket PCM16 only (`audio/pcm` / `pcm16`); rates: 8 / 16 / 24 / 48 kHz. WebRTC and G.711 are not supported.
- Server VAD is required. `turn_detection: null` (push-to-talk) and non-`server_vad` modes are rejected; VAD threshold, silence duration, prefix padding, and similar tuning are not implemented.
- Default client rate is 24 kHz; resampled to/from pipeline 16 kHz.

### Session lifecycle notes

- Welcome enabled (RTVI parity): client text and `response.create` are rejected until the first assistant `response.done`. Audio append is unaffected.
- Spoken turns complete after the output transport drains all queued TTS audio; only then does the gateway emit `response.done`.
- Post-handoff `session.update` is narrow: voice and audio-format/rate updates apply live. Input-transcription selector updates are compatibility no-ops. The server-VAD turn mode is immutable; other changes return `unsupported_live_session_update`.
- Pipeline ASR emits `conversation.item.input_audio_transcription.*` regardless of the accepted transcription selector; its requested model is ignored.
- `response.output_audio.delta` events precede `response.output_audio.done`, `response.output_audio_transcript.done`, `response.content_part.done`, `response.output_item.done`, and finally `response.done`.
- Each server-VAD turn uses one item id and sample-based timestamps. The wire order is `speech_started`, `speech_stopped`, `committed`, `conversation.item.created`, and transcription completion.
- Pipeline barge-in (`InterruptionFrame`) cancels the active Realtime response even when `TTSStopped` never arrives.
- Non-empty `response.create.response` overrides return `unsupported_response_override`; omit the field or send `{}`.

Live check: `RUN_REALTIME_COMPAT=1 uv run pytest tests/integration/test_realtime_openai_sdk_compat.py -v` — see [`tests/integration/README.md`](../../tests/integration/README.md).
