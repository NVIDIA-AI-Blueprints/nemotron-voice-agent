# omni_service.py — implementation contract

**Reference implementation:** [assets/omni_service.py](omni_service.py) — **COPY to project root** and wire `bot.py` constants. Do not reimplement unless user explicitly asks; spec below is for debugging only.

Pipecat `LLMService` subclass for Nemotron Omni. **Not** in stock `pipecat.services.nvidia`.

## Class

```python
class NvidiaOmniMultimodalService(LLMService): ...
# alias NvidiaOmniService = NvidiaOmniMultimodalService
```

## Settings (extend LLMSettings)

| Field | Default | Purpose |
| --- | --- | --- |
| model | cloud omni id | chat model |
| input_modalities | ("text","audio") | voice assistant |
| emit_transcriptions | True | parse user transcript from JSON response |
| min_user_audio_secs | 0.3 | ignore tiny buffers |
| pre_speech_buffer_secs | 0.2 | optional lead-in audio |
| stream | True | streaming completions |
| response_format | json_object when emit_transcriptions + audio | structured transcript |

Voice instruction when `emit_transcriptions`: strict JSON `{"transcript":"...","response":"..."}` with speech-ready plain text in `response`.

## Client

`AsyncOpenAI(base_url=OMNI_BASE_URL, api_key=..., timeout=120)` — local may use `"not-needed"`.

## Frame handlers (process_frame)

| Frame | Action |
| --- | --- |
| InputAudioRawFrame | append to buffer if user speaking else pre-speech ring |
| UserStartedSpeaking / VADUserStartedSpeaking | start buffer; barge-in → cancel request + InterruptionFrame |
| UserStoppedSpeaking | `_maybe_run_audio_turn()` |
| LLMRunFrame / LLMContextFrame | text-only turn (e.g. manual retry — **not** connect greeting) |
| InterruptionFrame | cancel in-flight request; clear buffers if not speaking |
| BotStartedSpeaking / BotStoppedSpeaking | track `_bot_responding` |
| default | push_frame downstream |

## Audio turn (_maybe_run_audio_turn)

1. Guard: not bot responding; no pending task; buffer non-empty
2. Join chunks; drop if `< min_user_audio_secs` (16-bit PCM)
3. Build messages: `input_audio` wav base64 + text instruction; history from `LLMContext`
4. `chat.completions.create` stream
5. Parse JSON → `transcript`, `response` or plain text
6. Emit `TranscriptionFrame` (optional), `LLMFullResponseStartFrame`, `LLMTextFrame`, `LLMFullResponseEndFrame`

**TTS guard (in reference `omni_service.py`):** `_is_unusable_spoken()` drops short/numeric garbage (`"27"`, `"27\n30"`, `"27 30"`) on **both** text and audio JSON `response` paths before `LLMTextFrame` → TTS. Do not weaken to `len < 4 and isdigit()` only.

## Anti-patterns

Subclass `NvidiaLLMService` only; skip `UserStoppedSpeaking` path; require `TranscriptionFrame` before turn end; hardcode model without catalog/curl.
