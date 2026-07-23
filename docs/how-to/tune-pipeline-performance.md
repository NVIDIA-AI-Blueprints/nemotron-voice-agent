# Tune Pipeline Performance

This section covers pipeline configurations for optimizing the performance and user experience of the Nemotron Voice Agent.

- [Smart Turn Detection](#smart-turn-detection)
- [Bot Introduction](#bot-introduction)
- [Chat History Limit](#chat-history-limit)
- [Audio Output Buffering](#audio-output-buffering)
- [Uvicorn Worker Scaling](#uvicorn-worker-scaling)
- [Transport Selection](#transport-selection)

## Smart Turn Detection

By default the cascaded pipeline uses Pipecat's ML-based [**Smart Turn**](https://docs.pipecat.ai/api-reference/server/utilities/turn-detection/smart-turn-overview) detection to decide when the user has finished speaking, so the agent replies promptly without cutting the user off. [Silero VAD](https://docs.pipecat.ai/server/utilities/audio/silero-vad-analyzer) (`stop_secs=0.2`) detects the pause, and the Smart Turn model then judges whether the turn is actually complete. If the model still has not finalized after **1.0 s** of silence, the turn completes anyway (fallback).

### How It Works

1. The user speaks, and ASR emits interim transcripts as audio streams in.
2. Silero VAD detects a pause in speech.
3. The Smart Turn model analyzes the recent audio and classifies the turn as **complete** or **incomplete**. If it's incomplete but silence continues past the 1.0 s stop threshold, the turn completes anyway (fallback).
4. On a completed turn, the transcript goes to the LLM and TTS streams the reply back.

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_SILERO_VAD_TURN_DETECTION` | `false` | Keep `false` for Smart Turn. Set `true` to disable it and use pure Silero VAD end-of-utterance detection instead. |
| `SILERO_VAD_STOP_SECS` | `0.5` | Silence (seconds) before end-of-utterance. Applies **only** in pure-VAD mode (`USE_SILERO_VAD_TURN_DETECTION=true`). |

> The default Smart Turn path uses a fixed `0.2 s` Silero floor (`stop_secs=0.2`) and a `1.0 s` Smart Turn silence fallback. `SILERO_VAD_STOP_SECS` does not apply on the Smart Turn path. The two thresholds are independent and never both active. The `generic-assistant/workstation-perf` profile forces pure Silero VAD (`USE_SILERO_VAD_TURN_DETECTION=true`, `SILERO_VAD_STOP_SECS=0.5`) for lower-overhead load testing.

### Key Components

| Component | Purpose |
|-----------|---------|
| [`SileroVADAnalyzer`](https://docs.pipecat.ai/server/utilities/audio/silero-vad-analyzer) | Voice activity detection with a configurable silence threshold |
| [Smart Turn](https://docs.pipecat.ai/api-reference/server/utilities/turn-detection/smart-turn-overview) (default) | ML end-of-utterance detection for natural turn-taking (`LocalSmartTurnAnalyzerV3`, fallback `stop_secs=1.0`) |
| `SpeechTimeoutUserTurnStopStrategy` | End-of-turn strategy used **only** in pure-VAD mode (`USE_SILERO_VAD_TURN_DETECTION=true`). Ends the turn on a VAD silence timeout instead of the Smart Turn model |
| `MuteUntilFirstBotCompleteUserMuteStrategy` | The user-mute strategy. Mutes user input until the first bot response completes |

The [Omni examples](../../src/examples/omni_assistant/README.md) run ASR inside the model, so there is no upstream `TranscriptionFrame` for Pipecat's stock Smart Turn stop strategy to wait on. They use a custom `AudioOnlySmartTurnStopStrategy` that wraps the same [Smart Turn](https://docs.pipecat.ai/api-reference/server/utilities/turn-detection/smart-turn-overview) model (`LocalSmartTurnAnalyzerV3`, fallback `stop_secs=1.0`) plus a `VADUserTurnStartStrategy`, and finalizes the turn as soon as the analyzer returns `COMPLETE`. The same `MuteUntilFirstBotCompleteUserMuteStrategy` applies.

## Bot Introduction

By default, when a client connects the bot introduces itself (greets the user) before the user says anything. This is driven by a synthetic first turn queued on `on_client_ready`. Set `ENABLE_BOT_INTRODUCTION=false` to skip that greeting so the bot stays silent until the user speaks first.

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_BOT_INTRODUCTION` | `true` | When `true`, the bot greets/introduces itself on connect. Set `false` to wait for the user to speak first. |

This applies to the Generic, Multilingual, Omni, and Frontend/Backend Agent examples. The `generic-assistant/workstation-perf` profile sets `ENABLE_BOT_INTRODUCTION=false` so benchmark runs start from a clean user turn.

```bash
# .env: keep the bot quiet until the user speaks
ENABLE_BOT_INTRODUCTION=false
```

## Chat History Limit

Every turn appends to the LLM's context. Left unbounded, that context keeps growing, which raises latency (more tokens to process means a higher time-to-first-token), increases cost, and eventually overflows the model's context window. For real-time voice, a small, bounded context is what keeps replies fast. Pipecat manages history through its [context aggregators](https://docs.pipecat.ai/guides/learn/context-management) and built-in [context summarization](https://docs.pipecat.ai/pipecat/fundamentals/context-summarization). The examples wire a **turn-count** window (`CHAT_HISTORY_RECENT_TURNS`) on top for a predictable per-turn budget.

We use context summarization logic for our examples to always **pin the initial prompt / system messages** loaded at session start. Nemotron's chat template carries the assistant instructions and tool definitions in the *user* section, so those must stay verbatim. Evicting them (as a token-based window might) would degrade tool-calling and persona. Only the older **conversational** turns are trimmed, and the handling differs by example:

- **Generic** and **Multilingual** assistants **summarize** older history: once the conversation grows past the recent window, the older turns are condensed into a single pinned summary message (an additional LLM call after the turn), and the most recent `CHAT_HISTORY_RECENT_TURNS` turns are kept verbatim.
- **Frontend/Backend Agent** uses a plain **sliding window**: older non-prompt messages are dropped, keeping only the most recent `CHAT_HISTORY_RECENT_TURNS` messages.

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `CHAT_HISTORY_RECENT_TURNS` | `10` (Generic / Multilingual), `20` (Frontend/Backend Agent) | Most recent non-prompt turns kept before older history is summarized (Generic/Multilingual) or dropped (Frontend/Backend Agent) |

```bash
# Override the per-example default (applies to whichever example is running)
CHAT_HISTORY_RECENT_TURNS=10
```

### How It Works

When the message count exceeds `CHAT_HISTORY_RECENT_TURNS`:

1. Initial prompt messages loaded at session start are always preserved.
2. The most recent `CHAT_HISTORY_RECENT_TURNS` turns are kept verbatim.
3. Older non-prompt turns are **summarized into one pinned summary message** (Generic / Multilingual) or **removed** (Frontend/Backend Agent).

### Recommendations

| Use Case | Value |
|----------|-------|
| Standard conversations | `10`–`20` |
| Multilingual mode | `5-10` |

## Audio Output Buffering

`AUDIO_OUT_10MS_CHUNKS` sets how many 10 ms audio frames the server batches per outbound send (the output buffer depth). Defaults are `5` (50 ms).

```bash
# In .env: override the transport default
AUDIO_OUT_10MS_CHUNKS=10
```

- **WebRTC, `5` (50 ms):** lowest latency.
- **WebSocket, `10` (100 ms):** smoother over a plain stream.
- **High concurrency, `10`–`40` (100–400 ms):** prevents glitches under load.

**Why buffer, and why more for WebSocket.** WebRTC ships its own jitter buffer and paces media for you, so a small buffer stays glitch-free at low latency. A raw WebSocket stream has no media-layer jitter buffer (frames ride a plain TCP connection), so under network jitter or many concurrent sessions, too-small chunks starve the client and you hear gaps or crackle. A larger buffer absorbs that variance, trading a little added latency for stable playback. Raise it further as concurrency grows.

**Telephony / server-side: send in bursts.** When the consumer is a telephony gateway (SIP/PSTN) or another server that does its own buffering and pacing, real-time chunking on our side only adds latency. Use a custom transport to send audio as soon as it is generated, and let the downstream handle playout timing.

**Barge-in trade-off.** Buffering works against fast barge-in: when the user interrupts, audio already queued downstream keeps playing until it drains, so the bot talks over the user for up to the buffer's duration. The bigger the buffer, the longer that tail. For low barge-in latency with a large buffer, the client must **flush its playback queue** on interruption (drop the buffered audio) rather than play it out. Custom and telephony clients (typically on the WebSocket path) should implement this flush.

## Uvicorn Worker Scaling

`UVICORN_WORKERS` controls how many `uvicorn` worker processes accept incoming sessions.

```bash
# .env or container env
UVICORN_WORKERS=<workers>
```

Keep `UVICORN_WORKERS=1` for local development or a personal assistant. For cloud deployments and scaling experiments, use a higher value. For the benchmarked best-scaling deployment shape and its companion tuning values, see [Reproducing the best scaling setup](../../benchmarking_tools/scaling-perf/README.md#reproducing-the-best-scaling-setup).

When `UVICORN_WORKERS > 1`, **session-config-based WebRTC and WebSocket flows are disabled** because that state is process-local. For multi-worker deployments, use one of these patterns:

- keep a **single worker** if you depend on per-process session config
- use **sticky routing** so a session stays on the same worker
- move session state into **shared storage**

## Transport Selection

The server supports both WebRTC and WebSocket transports simultaneously on different endpoints:

| Transport | Endpoint | Best For |
|-----------|----------|----------|
| **WebRTC** | `POST /api/offer` | Production voice interactions, lowest latency |
| **WebSocket** | `WS /api/ws` | Telephony / server-side integrations, testing, firewall-restricted environments, simpler deployments |

**For telephony (SIP/PSTN) and server-to-server use cases, use the WebSocket endpoint.** It streams raw audio frames you can bridge to a gateway or another service. WebRTC is best for direct browser clients, where its built-in jitter buffering and NAT traversal give the lowest latency.

### Choosing which transports are exposed

Both transports are exposed by default and the browser UI picks one. To restrict the server to a single transport, set `transports` in [`examples_registry.yaml`](../../examples_registry.yaml) or override it at runtime with the `TRANSPORT_SELECTION` environment variable (the env var wins):

| Value | Exposes |
|-------|---------|
| `all` (default) | WebRTC and WebSocket |
| `webrtc` | WebRTC only |
| `websocket` | WebSocket only |

```bash
# .env: e.g. expose only WebSocket for a telephony / server-side deployment
TRANSPORT_SELECTION=websocket
```
