# Configuration Guide

This is the index of everything you can configure in the Nemotron Voice Agent. Configuration lives in a small set of example-local files plus root `.env` settings, and the sections below index each area. For how the catalog files (`services.cloud.yaml` / `services.local.yaml`) work, see [Configure Services](how-to/configure-services.md).

## Model Service

What ASR / LLM / TTS models are available, their VRAM, precision, and known issues:

| Reference | Covers |
|-----------|--------|
| [Configure LLM](how-to/configure-llm.md) | Nemotron LLM models, reasoning on/off, GPU sizing & precision |
| [Configure ASR](how-to/configure-asr.md) | ASR models, VRAM, hardware support, Domain Adaptation & endpointing |
| [Configure TTS](how-to/configure-tts.md) | TTS models, VRAM, hardware support, voice selection, pronunciation (IPA), and text filters |

## Configuration how-to guides

| Guide | Description |
|-------|-------------|
| [Configure Services](how-to/configure-services.md) | How the catalog works: switch, add, and override LLM/ASR/TTS services via the UI or YAML |
| [Configure Prompts](how-to/configure-prompts.md) | Switch and add prompt presets via the UI or example-local prompt catalogs |
| [Configure Frontend/Backend Agent Domains](how-to/configure-frontend-backend-domains.md) | Select, configure, and extend the shared Talker/Thinker pipeline for airline, generic, or future domains |
| [Multilingual Voice Agent](../src/examples/multilingual/README.md) | Prompt-driven multilingual replies with a fixed language and TTS voice per session |
| [Enable OpenTelemetry Tracing](how-to/enable-opentelemetry-tracing.md) | Monitor latency and conversation flows with Phoenix or any OTLP backend |
| [Enable a TURN Server](how-to/enable-turn-server.md) | TURN server for remote / cross-network WebRTC access |
| [Enable the Audio Recorder](how-to/enable-audio-recorder.md) | Capture raw ASR/TTS audio per turn for debugging |
| [Use the Realtime Gateway](how-to/use-realtime-gateway.md) | OpenAI Realtime–compatible `WS /v1/realtime` for external clients |

## Welcome Message

When a client connects, the bot greets the user and introduces itself before the user speaks. Disable the welcome message to have the bot wait for the user instead. This behavior applies to the Generic, Multilingual, Omni, Omni Assistant Subagents, and Frontend/Backend Agent examples.

- **Per example (backend-only):** set `welcome_message: false` on an example in [`examples_registry.yaml`](../examples_registry.yaml). The default is `true` when the key is omitted.
- **Global override:** set `ENABLE_WELCOME_MESSAGE` in `.env`. When set, `ENABLE_WELCOME_MESSAGE` overrides every example's registry value. For example, the `generic-assistant/server-perf` profile uses it to disable the greeting.

```yaml
# examples_registry.yaml: keep one example quiet until the user speaks
examples:
  generic-assistant:
    welcome_message: false
```

While the welcome message is enabled, the user is muted until the bot finishes its opening turn (`MuteUntilFirstBotCompleteUserMuteStrategy`), which prevents interruption. When you disable the welcome message, the pipeline removes that mute strategy because there is no first bot turn. The user is unmuted immediately.

## Performance tuning

Pipeline tuning knobs (smart turn, chat-history window, audio buffering, transport) live in [Tune Pipeline Performance](how-to/tune-pipeline-performance.md). For benchmark results, see [Evaluation and Performance](04-evaluation-and-performance.md).
