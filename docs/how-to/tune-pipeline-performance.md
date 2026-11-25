# Tune Pipeline Performance

This section covers pipeline configurations for optimizing the performance and user experience of the Nemotron Voice Agent. It applies to the Pipecat examples (`nva-pipecat`).

- [Smart Turn Detection](#smart-turn-detection)
- [Chat History Limit](#chat-history-limit)
- [Audio Output Buffering](#audio-output-buffering)
- [Uvicorn Worker Scaling](#uvicorn-worker-scaling)
- [Transport Selection](#transport-selection)

## Smart Turn Detection

By default the cascaded pipeline uses Pipecat's ML-based [**Smart Turn**](https://docs.pipecat.ai/api-reference/server/utilities/turn-detection/smart-turn-overview) detection to decide when the user has finished speaking, so the agent replies promptly without cutting the user off. [Silero VAD](https://docs.pipecat.ai/server/utilities/audio/silero-vad-analyzer) (`stop_secs=0.2`) detects the pause, and the Smart Turn model then judges whether the turn is actually complete. If the model still has not finalized after the Smart Turn silence fallback (default **1.0 s**, `SMART_TURN_STOP_SECS`), the turn completes anyway (fallback).

> The [Frontend/Backend Live example](../../src/nemotron_voice_agent/pipecat/examples/frontend_backend_live/README.md#configuration) sets its own turn detection in `frontend.cascade.speech` of its settings, the `config` block of its `examples_registry.yaml` entry: `vad_start_seconds` (0.12 s of speech marks the caller as speaking and starts their turn while the bot is silent), `vad_stop_seconds`, `turn_silence_seconds`, and `interrupt_min_words` (the words that start the caller's turn while the bot speaks). The variables in this section do not apply to it. For how its turns start and end, refer to [Turns and Interruptions](../../src/nemotron_voice_agent/pipecat/examples/frontend_backend_live/README.md#turns-and-interruptions).

### How It Works

1. The user speaks, and ASR emits interim transcripts as audio streams in.
2. Silero VAD detects a pause in speech.
3. NVIDIA ASR independently finalizes after its configured 320 ms trailing-silence endpoint and emits a finalized transcript.
4. The Smart Turn model analyzes the recent audio and classifies the turn as **complete** or **incomplete**. If it is incomplete but silence continues past the Smart Turn stop threshold (default 1.0 s, `SMART_TURN_STOP_SECS`), the turn completes anyway (fallback).
5. Once the turn is complete, Pipecat's stock turn analyzer strategy closes it immediately if a finalized transcript is available. Otherwise, it closes the turn after the speech-to-text (STT) safety-net timeout, provided some transcript text has arrived. The transcript goes to the LLM, and TTS streams the reply back.

ASR endpointing is not triggered directly by the Silero VAD-stop frame. This
keeps ASR finalization independent from the acoustic pause signal used by
Smart Turn.

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_SILERO_VAD_TURN_DETECTION` | `false` | Keep `false` for Smart Turn. Set `true` to disable it and use pure Silero VAD end-of-utterance detection instead. |
| `SILERO_VAD_STOP_SECS` | `0.5`, or `0.8` with single-GPU Transcribe | Silence (seconds) before end-of-utterance in pure-VAD mode, including Transcribe. |
| `SMART_TURN_STOP_SECS` | `1.0` | Smart Turn silence fallback (seconds) before the turn completes without a `COMPLETE` classification. Applies **only** in Smart Turn mode (`USE_SILERO_VAD_TURN_DETECTION=false`). |

The Generic and Multilingual Assistants always use pure Silero VAD with Nemotron 3.5 Transcribe, regardless of `USE_SILERO_VAD_TURN_DETECTION`.

> On the Smart Turn path, a fixed `0.2 s` Silero VAD pause (`stop_secs=0.2`) first detects the silence. The Smart Turn model then gets up to the configured fallback period (default `1.0 s`, `SMART_TURN_STOP_SECS`) to finalize the turn. Only `SILERO_VAD_STOP_SECS` is ignored in Smart Turn mode. The `generic-assistant/server-perf` profile forces pure Silero VAD (`USE_SILERO_VAD_TURN_DETECTION=true`, `SILERO_VAD_STOP_SECS=0.5`) for lower-overhead load testing.

### Key Components

| Component | Purpose |
|-----------|---------|
| [`SileroVADAnalyzer`](https://docs.pipecat.ai/server/utilities/audio/silero-vad-analyzer) | Voice activity detection with a configurable silence threshold |
| [Smart Turn](https://docs.pipecat.ai/api-reference/server/utilities/turn-detection/smart-turn-overview) (default) | ML end-of-utterance detection for natural turn-taking (`LocalSmartTurnAnalyzerV3`, fallback `stop_secs` default `1.0`, `SMART_TURN_STOP_SECS`) |
| `SpeechTimeoutUserTurnStopStrategy` | End-of-turn strategy used **only** in pure-VAD mode. Ends the turn on a VAD silence timeout instead of the Smart Turn model |

The [Omni examples](../../src/nemotron_voice_agent/pipecat/examples/omni_assistant/README.md) run ASR inside the model, so no upstream `TranscriptionFrame` exists when Smart Turn completes. They use Pipecat's `TurnAnalyzerUserTurnStopStrategy` with `wait_for_transcript=False` and a `VADUserTurnStartStrategy`. This setting lets the same [Smart Turn](https://docs.pipecat.ai/api-reference/server/utilities/turn-detection/smart-turn-overview) model (`LocalSmartTurnAnalyzerV3`, fallback `stop_secs` default `1.0`, `SMART_TURN_STOP_SECS`) finalize the audio-only turn without waiting for a transcript.

## Chat History Limit

`CHAT_HISTORY_RECENT_TURNS` controls when the Pipecat Generic and Multilingual assistants attempt to summarize older conversation history. It does not set a hard context or token limit. The Omni examples do not use this setting.

For Frontend/Backend Live, set the talker's token budget with `frontend.cascade.memory.context_tokens` and the backend's budget with `core.delegation.context_tokens` in the example's `config` block. Refer to [Live Gateway Errors and Limits](use-live-gateway.md#errors-and-limits).

### Configuration

The default window is 10 user turns. To change it for a Generic or Multilingual deployment:

1. Set `CHAT_HISTORY_RECENT_TURNS` in `.env`.

   ```dotenv
   # Applies only to the Pipecat Generic and Multilingual assistants.
   CHAT_HISTORY_RECENT_TURNS=10
   ```

   Values below `1` disable this summarization.

2. Apply the change by restarting the host-native app or [recreating the Compose app container](../02-configuration-guide.md#where-to-make-changes).

### How It Works

After each assistant turn, the pipeline finds the recent window by counting backward through user-role messages, excluding the initial prompt. If non-prompt history precedes that window, the pipeline makes a separate LLM call to summarize it. On success, it replaces the older history with one summary message and keeps the initial prompt and recent messages unchanged. If summarization fails or the context changes during the call, the pipeline leaves the existing history intact.

### Recommendations

Start with the default. Use a larger window if the assistant needs more verbatim history, or a smaller one if older turns can be summarized sooner.

## Audio Output Buffering

`AUDIO_OUT_10MS_CHUNKS` sets the size of each server output batch in units of 10 ms of audio. Pipecat defaults to `5` (50 ms) for WebRTC and `10` (100 ms) for WebSocket. The `generic-assistant/server-perf` benchmark recipe sets `40` (400 ms); this is a benchmark setting, not a client latency target.

Larger batches send the same amount of audio in fewer chunks but can delay playback while the server fills each batch. The setting does not control the client playback queue. Audio already queued on the client can continue playing after a user interrupts unless the client discards it.

To tune a recipe that reads this setting from `.env`, use your client and expected concurrent load. Measure interruption delay from the start of user speech to the end of assistant playback.

1. Measure playback gaps and interruption delay with the transport default.
2. Set a different batch size in `.env`. For example, try 20 units (200 ms) for WebSocket.

   ```dotenv
   AUDIO_OUT_10MS_CHUNKS=20
   ```

3. Apply the change by restarting the host-native app or [recreating the Compose app container](../02-configuration-guide.md#where-to-make-changes).
4. Repeat the client measurements under the same load.
5. Choose the batch size that meets your playback and interruption targets.

## Uvicorn Worker Scaling

`UVICORN_WORKERS` controls how many `uvicorn` worker processes accept incoming sessions.

```bash
# .env or container env
UVICORN_WORKERS=<workers>
```

Keep `UVICORN_WORKERS=1` for local development or a personal assistant. For cloud deployments and scaling experiments, use a higher value. For the recommended scaling deployment shape and its companion tuning values, see [Reproducing the recommended scaling setup](../../benchmarking_tools/scaling-perf/README.md#reproducing-the-recommended-scaling-setup).

When `UVICORN_WORKERS > 1`, the stock app disables session-config-based WebRTC and WebSocket flows and does not serve `/v1/live`. Keep one worker for those flows. Sticky routing alone does not reenable the disabled routes. Use the [scaling benchmark](../../benchmarking_tools/scaling-perf/README.md) for the supported multi-worker test path.

## Transport Selection

When both transports are enabled, Pipecat exposes them on different endpoints:

| Transport | Endpoint | Best For |
|-----------|----------|----------|
| **WebRTC** | `POST /api/offer` | Direct browser sessions using WebRTC media transport |
| **WebSocket** | `WS /api/ws` | RTVI testing and integrations that handle audio framing and playback |

For a custom telephony bridge or server-to-server integration, the WebSocket endpoint uses the Pipecat client frame format. It is not a SIP endpoint. WebRTC provides media transport and jitter handling for browser clients. Compare playback gaps and interruption delay on your target client under load rather than assuming one transport always has lower latency.

### Choosing Which Transports Are Exposed

Pipecat exposes both transports by default. Set `pipecat.transports` in [`examples_registry.yaml`](../../examples_registry.yaml) to restrict the server to one transport. `TRANSPORT_SELECTION` overrides the registry value. The following values apply to both settings:

| Value | Exposes |
|-------|---------|
| `all` (default) | WebRTC and WebSocket |
| `webrtc` | WebRTC only |
| `websocket` | WebSocket only |

To expose only WebSocket using the environment override:

1. Set `TRANSPORT_SELECTION` in `.env`.

   ```dotenv
   TRANSPORT_SELECTION=websocket
   ```

2. Apply the change by restarting the host-native app or [recreating the Compose app container](../02-configuration-guide.md#where-to-make-changes).
