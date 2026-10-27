# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

This release splits the Frontend/Backend Live example into the Live gateway, a frontend adapter that hosts the cascade, a delegation core, and a backend adapter. It adds two backend adapters beside the backend LLM and makes each choice a configuration setting.

### Added

- **Backend adapters** for the Frontend/Backend Live example, selected by `backend.adapter` in `core.yaml`: `llm`, the backend LLM (the default); `llm-agent`, a model conductor with LLM workers (`core.agent.yaml`); and `harness`, an agent that you deploy and connect over the Agent Client Protocol (ACP) (`core.harness.yaml`). The "Connect Your Own Harness" section of `docs/design/architecture.md` shows how to connect an agent.
- **Delegation core settings** in `src/examples/frontend_backend_live/core.yaml`, with the profiles `core.ui-bot.yaml`, `core.bench.yaml`, `core.agent.yaml`, and `core.harness.yaml`, selected by `CORE_CONFIG_PATH`. `LIVE_CONFIG_PATH` selects the frontend's benchmark profile, `config.bench.yaml`.
- **Parallel tasks.** A model conductor can split a request into tasks that run in parallel, and steer or cancel one task. Each task's answer is its own `final_answer` message.
- **Speak and display answers** (`delegation.result_format: speak_display`). Screen-only detail reaches the client as an `mcp_call` item, and the repository UI bot shows each delegation as a task card.
- **Official Live events.** A response opens at acceptance with `response.created` (status `queued`) and `response.queued`, every message carries `phase`, backend tools appear as `mcp_call` items, and a request that joins open work names the older response in `previous_response_id`.
- **Background settings checks.** A Live session reports a `delegation.responses` setting that the backend cannot honor as a nonfatal `responses_validation_failed` error.
- The Frontend/Backend Live architecture page, `docs/design/architecture.md`, and READMEs for the Live gateway (`src/live/README.md`) and the delegation core (`src/delegation_core/README.md`).
- `LIVE_MAX_CONSOLE_ROUNDS` and `LIVE_ALLOWED_HOSTS` to limit `POST /v1/live/delegate`, and `LIVE_ADAPTER_FACTORY` to name another factory for each session's frontend adapter.
- `ENABLE_TRACING` traces Live sessions, and each Live session log line names its session.
- Tests for the Live Console's JavaScript modules in `client/test/live/`, which CI runs with `node --test`.
- The `agent-client-protocol` 0.9.0 dependency for the harness adapter.

### Changed

- In the Frontend/Backend Live example, typed input (`response.item.create` with a user message, then `response.create`) keeps the turns spoken before it, and the next delegation carries them to the backend.
- The Frontend/Backend Live example's `config.yaml` holds only the frontend's settings (`frontend.adapter`, `frontend.cascade`, and `ui_bot.core_config`), and `core.yaml` holds the delegation core's settings. A key in its old place stops the load with a message that names its new place. A restart applies an edit to either file.
- In the Frontend/Backend Live example, a request that arrives while backend work runs joins that work at its next step. Ready results wait until the caller is idle and are spoken together, and a display part reaches the screen without being spoken.
- In the Frontend/Backend Live example, `frontend.cascade.memory` sets the talker's memory as a token budget instead of a count of recent turns.
- In the Frontend/Backend Live example, a delegation carries a labeled transcript (`Caller:`, `Agent:`, `App context:`) of what the caller said and heard since the last delegation.
- In the Frontend/Backend Live example, after `progress_after_seconds` of quiet, the voice says the backend's newest progress line or its `still_working` line.
- In the Frontend/Backend Live example, backend voice note `v2` (`delegation.voice_note` in `core.yaml`) is the default, and prompt set `v2` is the default when `config.yaml` names none.
- In the Frontend/Backend Live example, a failed backend task ends with `response.failed` and an official error code, or with `response.incomplete`, and the voice says its `apology`, `unclear`, or `lost` line from `lines:` in the example's `prompts.yaml`.
- In the Frontend/Backend Live example, `delegation.client_tool_timeout_seconds` (60 seconds) in `core.yaml` is the deadline for the app's function outputs. A `response.create` that continues nothing while outputs are owed is refused with `function_call_outputs_required` or `tool_outputs_required`.
- In the Frontend/Backend Live example, a backend model that the catalog does not list is refused (`unlisted_models: report`), except in `core.bench.yaml` (`fallback`). An `openai-responses` backend sends the whole history each round with `store: false`.
- The `frontend-backend-live/server` profile runs the talker and the backend on the Nemotron 3.5 Lightning NIM. A `services.yaml` entry with `api_key_env` never falls back to `NVIDIA_API_KEY`, and a slot with no model is an error that names the slot.
- The Live routes come from the `live:` block of the example's entry in `examples_registry.yaml`. A server with more than one Uvicorn worker serves no Live route, and a `LIVE_` limit that is not a whole number of at least 1 stops the server at startup.
- In client mode, `session.update` can change `delegation.acknowledgement_delay_ms`.
- When the repository UI starts a session, the server checks the readiness of every local LLM that the example uses, including the Thinker LLM of the Frontend/Backend Agent and Frontend/Backend Live examples.

### Deprecated

- `LIVE_ENGINE_FACTORY`. Use `LIVE_ADAPTER_FACTORY` instead. The old name still works and logs a deprecation warning.

### Removed

- The Frontend/Backend Live Realtime frontend (`provider: realtime` in the example's `config.yaml`). Its code is parked, unused, in `src/examples/frontend_backend_live/parked/realtime/`.
- The Frontend/Backend Live settings `merge_pending`, `fold_before_acting`, `status_from_ledger`, `progress_speech`, `progress_interval_seconds`, `recover_failures`, `chat_history_recent_turns`, `backend_context_tokens`, and the `backend` block and `voice_note` of `config.yaml`. `backend.llm.conductor` (`steer` or `start_each`) in `core.yaml` sets how the backend LLM places each request. Each removed key's error message names its replacement.
- The `examples.frontend_backend_live.engine` import path. Use `examples.frontend_backend_live.factory`.

### Fixed

- WebSocket caller audio of a Live session reaches speech recognition at 16 kHz, and local recipes send Live reply audio.
- G.711 audio (`audio/pcmu` or `audio/pcma`) on a Live session takes 8000 Hz when the format names no rate, and its samples follow the standard G.711 encoding.
- An error that a Live client command causes goes only to the connection that sent the command.
- A Live session that the server refuses because it is full gets a `rate_limit_exceeded` error (HTTP 429 over WebRTC) instead of a bare close.
- A Live session closes with `remote_hangup` when the client hangs up and `connection_lost` otherwise. A binary WebSocket frame gets an `error` event and leaves the session open, and a WebRTC session ends when only its data channel closes.
- The gateway paces delegation events to a Live client, so a burst does not close the session. A sideband connection that falls behind is closed with code 1013, and a delegation stream that ends before its terminal event gets one `response.failed` with `server_error`.
- A slow delegation core does not hold a Live caller's audio, and a frontend that hangs while a Live session closes does not keep backend work running.
- When a Live session's voice service fails, the client gets one `pipeline_error` error and then `session.closed` with `connection_lost`.
- The server accepts every documented voice and `service_tier` value and `null` for `parallel_tool_calls` in a Live session, refuses an unknown key with `unknown_parameter` and the key's dotted `param`, and leaves optional fields out of `error` events.
- The `frontend-backend-live` Compose profiles set `LIVE_ENABLED` to `true` by default, so they serve the Live routes with any `EXAMPLE_SELECTION`, such as `all`. In a deployment locked to another example, Live sessions use the Frontend/Backend Live example's own service slots.
- The Live gateway's WebRTC peer uses the TURN server that `TURN_URL`, `TURN_USERNAME`, and `TURN_PASSWORD` configure, followed by the servers in `LIVE_ICE_URLS`.
- The Live Console gets its ICE servers from `GET /api/ice-servers`, so it uses a configured TURN server. A TURN server that the browser cannot reach does not fail the **Start**.
- The Live Console escapes every value that it writes as HTML, begins a new call at each **Start**, keeps a `+` in a link token (`live.html#token=`), and keeps the newest 2,000 event log rows.
- The Live Console's delegation cards follow the official events, and the console shows the delegation mode and the backend badge from `session.started`. Its history panel polls only while it is open and the page is visible.
- In client mode, the Live Console runs one delegation at a time, each with its own history, and **End** stops every round and tool. Bad backend tools JSON stops the **Start** in both modes.
- The Live Console's browser cafe follows the server's cafe tool rules, and it places an order only when `confirmed` is exactly `true`.
- In the Frontend/Backend Live example's repository UI bot, the greeting comes from `intro` in `lines:` of the example's `prompts.yaml`, and no late result is worded after the call ends.

## [2.3.0] - TBD

This minor release upgrades Pipecat to 1.12.0, adopts its built-in transcript-independent Smart Turn handling for audio-only user turns, and adds streaming-input prefill for the Generic Assistant on single-GPU hosts.

### Added

- **Frontend/Backend Live** example (`frontend-backend-live`): a talker that speaks or delegates each turn, a tool-using backend, and a commentary step, with frontend and backend models selected in `config.yaml`. It also serves live sessions at `/v1/live/sessions`. Cloud, `/server`, and `/single-gpu` recipes are available.
- **Streaming-input prefill** for the Generic Assistant. The single-GPU Lightning vLLM sidecar also serves a text StreamingInput WebSocket at `/v1/streaming-session`, and the **Streaming Input** toggle on the Lightning service card streams each ASR update into it, so the prompt is prefilled while the user is still speaking. `NvidiaStreamingLLMService` and `StreamingLLMUserAggregator` are drop-in replacements for `NvidiaLLMService` and `LLMUserAggregator`.
- One root `services.yaml` for every example, selected by `SERVICE_RECIPE`, and editable LLM parameters such as reasoning and temperature in the Services tab.
- Services tab controls for TTS and ASR: **Synthesis Mode** on Magpie TTS entries and **Punctuation** on the single-GPU NeMo-Speech.cpp ASR entries.

### Changed

- Upgraded Pipecat to version 1.12.0.
- Replaced the custom Omni audio-only Smart Turn stop strategy with Pipecat's built-in `TurnAnalyzerUserTurnStopStrategy`, configured with `wait_for_transcript=False`.
- UI selections (example, transport, services, prompt) no longer persist in browser localStorage. They survive disconnect and reconnect, and a page refresh resets them to the server defaults. Custom services and prompts stay saved.

### Fixed

- Host-native runs reach Nemotron 3 Super on its published port `18001`.
- The Frontend/Backend Agent Thinker honors the Max Tokens setting.

### Removed

- `LOCAL_SERVICES_ENABLED`, `SERVICES_CLOUD_PATH`, `SERVICES_LOCAL_PATH`, and the `*-reasoning` catalog entries. Use `SERVICE_RECIPE` and the Reasoning setting instead.
- The per-example `services.cloud.yaml` and `services.local.yaml` files, and the `slots` field and `defaults.llm`, `defaults.asr`, and `defaults.tts` keys in `examples_registry.yaml`. List each example's services under `services` instead. The first available key is the default.
- The `nemotron-voice-agent-configure-pipeline` skill. `nemotron-voice-agent-deploy` now covers configuration changes.

## [2.2.0] - 2026-09-22

This minor release adds Nemotron 3.5 Lightning and Nemotron 3 Nano Omni support, universal `*/single-gpu` recipes with the NeMo-Speech.cpp stack, an OpenAI Realtime-compatible WebSocket gateway, and the `nemotron-voice-agent-builder` skill. It consolidates on-prem deployment under `*/server` and `*/single-gpu`, upgrades Pipecat to 1.7.0, and removes the platform-specific recipes and Nemotron 3 Nano cascaded LLM.

### Added

- **Nemotron 3.5 Lightning** NIM and vLLM sidecars. Single-GPU recipes load the NVFP4 checkpoint on supported GPUs. Blackwell, DGX Spark, and Jetson Thor serve native NVFP4, with DSpark speculative decoding on DGX Spark and DFlash on Blackwell workstations. Hopper and Ada serve the same NVFP4 checkpoint as W4A16 through Marlin.
- **Nemotron 3 Nano Omni NIM** for the Omni `*/server` recipes.
- **`*/single-gpu` recipes** running the NeMo-Speech.cpp speech stack next to vLLM on one GPU across all examples.
- **`scripts/download-nemo-speech-models.sh`** for one-time NeMo-Speech.cpp GGUF setup on single-GPU hosts.
- **OpenAI Realtime–compatible WebSocket gateway** (`WS /v1/realtime`) that drives the existing cascaded ASR → LLM → TTS pipelines for OpenAI Realtime–shaped clients, with NVIDIA-only knobs under `session.nvidia`.
- **`nemotron-voice-agent-builder` agent skill** for scaffolding and refining NVIDIA voice agents (cascaded or Omni) across Pipecat and LiveKit.
- Optional `NvidiaWordTTSService` for Magpie word streaming and timestamp-based context commits.

### Changed

- Serve Nemotron 3.5 Lightning NVFP4 as W4A16 through Marlin on Hopper and Ada single-GPU hosts, instead of a BF16 checkpoint with online FP8.
- Upgraded Pipecat to version 1.7.0.
- Set **Nemotron 3.5 Lightning** as the default LLM across cascaded examples. Nemotron 3 Super remains available in the service catalogs.
- Standardized the self-hosted Nemotron 3.5 Lightning served model ID with NVIDIA Cloud across NIM and single-GPU vLLM deployments.
- Consolidated on-prem deployment recipes under `<example>/server` for scaling-oriented stacks and universal `<example>/single-gpu` for supported one-GPU deployments on workstations, DGX Spark, and Jetson Thor. Renamed `generic-assistant/workstation-perf` to `generic-assistant/server-perf`.
- Replaced `PLATFORM`-based local service selection with endpoint reachability.
- Gated the NVIDIA Cloud (NVCF) service catalog on `NVIDIA_API_KEY`, so those entries appear in the UI and pipeline only when a cloud key is set.
- Updated Nemotron ASR Streaming NIM to version 1.3.1.
- Updated Parakeet CTC 1.1B ASR NIM to version 1.5.3.
- Updated Magpie TTS Multilingual NIM to version 1.10.0 and `nvidia-riva-client` to version 2.27.0.
- Renamed the Magpie Multilingual Compose services to `magpie-multilingual-tts-service` and `magpie-multilingual-tts-service-perf`.
- Updated Chatterbox TTS Multilingual NIM to version 1.1.0 and documented its available model profiles.
- Updated the React UI client to `@pipecat-ai/client-react` 1.8.2 and rendered the conversation transcript directly from Pipecat's `usePipecatConversation()` hook, removing the custom client-side turn reconstruction and timestamp re-anchoring.
- Relicensed the repository from BSD-2-Clause to a dual CC-BY-4.0 and Apache-2.0 license.

### Fixed

- Cap Omni NIM KV cache (`NIM_KVCACHE_PERCENT` default `0.6`, `NIM_MAX_MODEL_LEN` `32768`) so Magpie TTS can share GPU 0.
- Emit `user-turn-finalized` from Omni Assistant Subagents so the server signals user-turn boundaries.
- Keep chat transcripts in speaking order by rendering the Pipecat SDK message stream directly instead of merging server- and client-derived timestamps, which could reorder turns under clock skew.
- Render each bot turn as a single transcript bubble instead of splitting multi-sentence responses into separate messages.
- Show the negotiated RTVI protocol version in the Session panel instead of the client library version.
- Resolve shared TTS voice ids (`John`) against the catalog default language so the disabled language dropdown shows `en-US` instead of the alphabetically first locale.

### Removed

- Platform-specific **`*/workstation`, `*/dgx-spark`, and `*/jetson-thor` recipes** and the Jetson Riva sidecars (`nemotron-speech`, `nemotron-speech-tts`), along with the CUDA MPS helper scripts, CPU-pinning knobs, and `RIVA_*` environment knobs. Use `*/server`, `generic-assistant/server-perf`, or `*/single-gpu` instead.
- **Nemotron 3 Nano** cascaded LLM catalog entries and its NIM compose file. Use Nemotron 3.5 Lightning instead.
- **NVIDIA Cloud (NVCF) catalog entries** for Nemotron 3 Super (`nemotron-super`, `nemotron-super-reasoning`) and Parakeet CTC 1.1B (`parakeet-ctc`). Both models remain available self-hosted.

## [2.1.1] - 2026-08-21

Patch release replacing the deprecated Nemotron 3 Nano cloud endpoint in cascaded pipelines.

### Changed

- Replaced the deprecated Nemotron 3 Nano cloud endpoint with Nemotron 3.5 Lightning across cascaded NVIDIA Cloud service catalogs and defaults.

## [2.1.0] - 2026-08-18

Minor release with expanded TTS options, improved turn-taking, Omni Assistant Subagents refinements, multilingual model compatibility checks, and broader evaluation coverage.

### Added

- Chatterbox TTS Multilingual with 23 languages and Magpie TTS Zeroshot with optional voice cloning.
- Omni Assistant webcam steering, thinker reasoning, and subagent handoff improvements.
- Pipecat evaluation coverage for core examples, latency, and tool scenarios.

### Changed

- Updated Magpie TTS Multilingual NIM to version 1.9.0 and documented its 12 supported languages.
- Upgraded Pipecat to **1.5.0** and refreshed NIM, vLLM, and CVE-related dependency pins.
- Improved Smart Turn defaults and added inactivity checks for the Generic Assistant.
- Refactored `NvidiaOmniLLMService` on Pipecat's `NvidiaLLMService` base.
- Grouped sentence-level bot output into turn-level transcript messages across all examples.

### Fixed

- Improved Omni Assistant Subagents media handling.
- Restricted multilingual sessions to supported ASR, TTS, and LLM combinations.

## [2.0.0] - 2026-07-09

Re-architecture by upstreaming `nvidia-pipecat` changes to [Pipecat](https://github.com/pipecat-ai/pipecat) and a unified React UI client, new multimodal Nemotron Omni and Frontend Backend Agent examples and recipe-style deployment profiles.

### Added

- **Example pipelines**, each shipped as self-contained Compose recipes:
  - `generic-assistant`: baseline English cascaded pipeline (Nemotron ASR + LLM + Magpie TTS).
  - `multilingual-assistant`: showcases a multilingual pipeline using Multilingual ASR and TTS, with a fixed language per session for better reliability.
  - `omni-assistant`: Nemotron 3 Nano Omni as a single multimodal ASR + LLM service with Magpie TTS.
  - `omni-assistant-subagents`: the Omni service split across cooperating Pipecat Subagents with live webcam, vision and uploaded-media analysis.
  - `frontend-backend-agent`: a talker LLM front-ending a stateful backend agent (airline-booking reference).
- `NvidiaOmniLLMService`: an upstream-compatible Pipecat `LLMService` for Nemotron Omni
- **Unified UI client** (`client/`) with grouped LLM/ASR/TTS selectors (Self-hosted / NVIDIA Cloud / Custom), a prompt-preset picker, a Metrics panel, attachment upload, and a webcam panel.
- **New models**: Nemotron 3 Super 120B A12B and Nemotron 3 Nano Omni (LLMs), plus Nemotron ASR Streaming (English and Multilingual).
- **DGX Spark and single-GPU workstation** local NIM deployment, plus improved Jetson Thor support.
- Agent skills for example deployments and configurations.

### Fixed

- Improve TTS text filter to avoid removing commonly used special chars like $, %
- Summarize chat history for long sessions to limit LLM context growth.

### Changed

- Pipecat is used directly and upgraded **0.0.98 → 1.3.0**.
- Enabled Pipecat smart turn detection model as default turn taking solution replacing Silero VAD based silence EOU.
- **Recipe-style Compose profiles**: `<example>` (cloud-only) and `<example>/<hardware>` (on-prem, for example `generic-assistant/dgx-spark`, `omni-assistant/jetson-thor`). Each is a complete, self-contained stack. `tracing` (Phoenix OTel) and `turn` (Coturn) remain optional overlays.

### Removed

- `nvidia-pipecat` Git submodule (NVIDIA Services Upstreamed to Pipecat directly).
- The previous `frontend/` WebRTC and WebSocket UIs, replaced by the React `client/` in the Pipecat-based re-architecture.
- Speculative speech processing (`ENABLE_SPECULATIVE_SPEECH`), the latency optimization that used intermediate ASR transcripts.
- Legacy configuration environment variables (per-slot `DEFAULT_*`, `DEPLOYMENT_PLATFORM`, pipeline-shaping CLI flags, and example-specific service URLs). Configuration now lives in the example service catalogs and `examples_registry.yaml`.

## [1.0.0] - 2026-03-03

Initial release of Nemotron Voice Agent — an end-to-end voice agent blueprint powered by NVIDIA Nemotron ASR, LLM, and TTS, designed for scalable, production-ready deployments.

### Added

- End-to-end voice agent pipeline with NVIDIA Nemotron ASR, LLM, and TTS, supporting streaming audio and mid-conversation interruptions
- Built on the open source [Pipecat-ai](https://github.com/pipecat-ai/pipecat) and [nvidia-pipecat](https://github.com/NVIDIA/voice-agent-examples) frameworks
- NVIDIA Nemotron Speech models:
  - [Parakeet CTC 1.1B](https://build.nvidia.com/nvidia/parakeet-ctc-1_1b-asr/modelcard) (English ASR)
  - [Parakeet 1.1B RNNT](https://build.nvidia.com/nvidia/parakeet-1_1b-rnnt-multilingual-asr/modelcard) (Multilingual ASR)
  - [Magpie TTS Multilingual](https://build.nvidia.com/nvidia/magpie-tts-multilingual/modelcard)
- NVIDIA Nemotron LLMs using NVIDIA NIM:
  - [Nemotron 3 Nano 30B A3B](https://build.nvidia.com/nvidia/nemotron-3-nano-30b-a3b/modelcard)
  - [Llama 3.3 Nemotron Super 49B v1.5](https://build.nvidia.com/nvidia/llama-3_3-nemotron-super-49b-v1_5/modelcard)
- WebRTC transport for real-time, low-latency voice communication with a custom frontend UI
- Docker Compose deployment with optional TURN server support for remote access
- Multilingual support with automatic language detection and mid-conversation language switching
- Jetson Thor edge deployment support
- Pipeline customizations using environment variables and config files
  - ASR, LLM, TTS model change
  - Speculative speech processing enable/disable
  - Conversation history thresholds
  - output audio buffering
- Open telemetry tracing and monitoring support
- Documentation:
  - [Getting started guide](docs/01-getting-started.md) covering prerequisites, GPU configuration, and step-by-step setup
  - [Configuration guide](docs/02-configuration-guide.md) for pipeline customizations
  - [Jetson Thor deployment guide](docs/03-jetson-thor.md) for edge use cases
  - [Best practices guide](docs/05-best-practices.md) covering production deployment, latency optimization, and conversational UX
- AI agent deployment skill for Cursor and Claude Code to streamline deployment on workstations and Jetson Thor

### Known Issues

- ASR transcription can occasionally be inaccurate, though the LLM generally compensates by inferring meaning from context.
- The context aggregator limits chat history to 20 turns by default. Older turns are dropped when this limit is reached, rather than summarized.
