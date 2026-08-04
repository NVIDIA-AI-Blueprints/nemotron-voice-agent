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
| Tools | Client-registered schemas + model-native calling | **Ignored** for client schemas; catalog tools follow `prompt.id` / `prompt_key` / `prompts.yaml` |
| Welcome | Client-driven | Optional Nemotron welcome gate (RTVI parity): client text / `response.create` rejected until first assistant `response.done` |
| Live `session.update` | Broad session mutation | Only voice, turn detection, and audio format/rate after handoff |
| Transport | WebSocket (and OpenAI WebRTC) | WebSocket only; WebRTC out of scope |

Invalid session config that would break audio (bad PCM format/rate, text-only modalities) returns a Realtime `error` and **keeps the socket open** for retry. Fields that do not translate to the cascade (including `session.tools`) are ignored.

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
        api_key="unused",  # gateway does not require OpenAI auth
        websocket_base_url="ws://127.0.0.1:7860/v1",
    )
    async with client.realtime.connect() as conn:
        assert (await conn.recv()).type == "session.created"
        await conn.send({"type": "session.update", "session": {"type": "realtime"}})
        print(await conn.recv())  # session.updated

asyncio.run(main())
```

Use `ws://` when `PIPELINE_TLS=false`; otherwise `wss://`.

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
      "input": { "format": { "type": "audio/pcm", "rate": 24000 } },
      "output": { "format": { "type": "audio/pcm", "rate": 24000 } }
    },
    "nvidia": { "pipeline_mode": "generic-assistant" }
  }
}
```

Catalog tools (when enabled for the selected prompt) still emit Realtime function-call events; return results with `conversation.item.create` / `function_call_output` if the client must supply tool output. Client-defined `session.tools` schemas are not registered.

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
| `audio.input.turn_detection` | transport commit policy |
| `audio.input.transcription` | rejected if set (separate OpenAI transcription model; cascaded ASR still emits `input_audio_transcription.*`) |
| `output_modalities` / `modalities` | must include `audio` when set |

Defaults when omitted: `pipeline_mode=generic-assistant`, `prompt_key=generic_assistant_without_tools`.

### `session.nvidia` (no OpenAI equivalent)

Nemotron-only catalog / routing keys: `pipeline_mode`, `llm_id`, `asr_id`, `tts_id`, `asr_language_code`, `model_id`, `base_url`, `extra_params`, ASR/TTS `*_server` / `*_model` / `*_function_id`, `tts_synthesis_mode`.

Prefer OpenAI fields for prompt and generation settings: use `prompt.id`, `instructions`, `temperature`, and `max_output_tokens`. Do not put `tts_voice_id`, `system_prompt`, `max_tokens`, or `temperature` under `nvidia`.

Public `session.updated` echoes a **redacted** `nvidia` view (pipeline / catalog ids only). Internal ASR/TTS endpoints and function ids are not returned to the client.

Changing `tts_id` / `tts_server` / `tts_model` / `tts_function_id` at connect time re-lists voices for that TTS selection only when that routing key is not already cached (same catalog path as RTVI `GET /api/tts-config`), then re-resolves `voice` against the cached list.

## Realtime API reference (v1)

### Client → server

| Event | Notes |
|-------|--------|
| `session.update` | Configure session; required before pipeline handoff. Invalid audio/modality config → `error`, socket stays open. |
| `input_audio_buffer.append` | Base64 PCM chunk |
| `input_audio_buffer.commit` | Commit buffered audio as a user turn |
| `input_audio_buffer.clear` | Drop uncommitted audio |
| `conversation.item.create` | User text / function_call_output items |
| `conversation.item.truncate` | Barge-in while a response is in progress. Idle truncate is ignored. Does not emit `conversation.item.truncated`. |
| `response.create` | Start a model turn. Empty `response: {}` is accepted; non-empty per-response overrides are rejected. |
| `response.cancel` | Cancel the in-progress response. Idle cancel is a no-op. |

### Server → client

| Event | Notes |
|-------|--------|
| `session.created` / `session.updated` | Session object (OpenAI-shaped + optional `nvidia`) |
| `error` | Realtime-shaped error; common codes include `invalid_session`, `services_not_ready`, `unsupported_live_session_update`, `unsupported_response_override`, `response_create_rejected_pre_intro` |
| `input_audio_buffer.*` | Speech / commit / clear acks |
| `conversation.item.*` | Item created; cascaded ASR input transcription. `conversation.item.truncated` is not emitted. |
| `response.*` | Lifecycle, audio, transcript, text, function-call events |

**Dual emit:** GA names are canonical (`response.output_audio.*`, `response.output_audio_transcript.*`, `response.output_text.*`). Matching pre-GA aliases (`response.audio.*`, `response.audio_transcript.*`, `response.text.*`) are also emitted so older SDKs work.

### Audio

- PCM only (`audio/pcm` / `pcm16`); rates: 8 / 16 / 24 / 48 kHz. Unsupported format/rate is rejected at `session.update`.
- Default client rate is 24 kHz; resampled to/from pipeline 16 kHz.

### Session lifecycle notes

- Welcome enabled (RTVI parity): client text and `response.create` are rejected until the first assistant `response.done`. Audio append is unaffected.
- Spoken turns complete when TTS finishes (`response.done`).
- Post-handoff `session.update`: only `voice`, `turn_detection`, and audio format/rate may change; other changes return `unsupported_live_session_update`. Live PCM rate changes recreate the stream resampler so audio keeps flowing.
- Pipeline barge-in (`InterruptionFrame`) cancels the active Realtime response even when `TTSStopped` never arrives.
- Non-empty `response.create.response` overrides return `unsupported_response_override`; omit the field or send `{}`.

Live check: `RUN_REALTIME_COMPAT=1 uv run pytest tests/integration/test_realtime_openai_sdk_compat.py -v` — see [`tests/integration/README.md`](../../tests/integration/README.md).
