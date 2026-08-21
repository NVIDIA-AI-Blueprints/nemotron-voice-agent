# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Optional `NvidiaWordTTSService` for Magpie word streaming and timestamp-based context commits.
- **`*/single-gpu` recipes** running the NeMo-Speech.cpp speech stack next to vLLM on one GPU across all examples.
- **Nemotron 3.5 Lightning** NIM and vLLM sidecars. Single-GPU recipes automatically select NVFP4 or FP8 on supported GPUs, with DSpark speculative decoding on DGX Spark and DFlash on Blackwell workstations.
- **Nemotron 3 Nano Omni NIM** for the Omni `*/server` recipes.
- **`scripts/download-nemo-speech-models.sh`** for one-time NeMo-Speech.cpp GGUF setup on single-GPU hosts.

### Changed

- Updated Nemotron ASR Streaming NIM to version 1.3.1.
- Updated Parakeet CTC 1.1B ASR NIM to version 1.5.3.
- Updated Magpie TTS Multilingual NIM to version 1.10.0 and `nvidia-riva-client` to version 2.27.0.
- Renamed the Magpie Multilingual Compose services to `magpie-multilingual-tts-service` and `magpie-multilingual-tts-service-perf`.
- Updated Chatterbox TTS Multilingual NIM to version 1.1.0 and documented its available model profiles.
- Standardized the self-hosted Nemotron 3.5 Lightning served model ID with NVIDIA Cloud across NIM and single-GPU vLLM deployments.
- Consolidated on-prem deployment recipes under `<example>/server` for scaling-oriented stacks and universal `<example>/single-gpu` for supported one-GPU deployments on workstations, DGX Spark, and Jetson Thor. Renamed `generic-assistant/workstation-perf` to `generic-assistant/server-perf`.
- Replaced `PLATFORM`-based local service selection with endpoint reachability.
- Set **Nemotron 3.5 Lightning** as the default LLM across cascaded examples. Nemotron 3 Super remains available in the service catalogs.

### Removed

- Platform-specific **`*/workstation`, `*/dgx-spark`, and `*/jetson-thor` recipes** and the Jetson Riva sidecars (`nemotron-speech`, `nemotron-speech-tts`), along with the CUDA MPS helper scripts, CPU-pinning knobs, and `RIVA_*` environment knobs. Use `*/server`, `generic-assistant/server-perf`, or `*/single-gpu` instead.
- **Nemotron 3 Nano** cascaded LLM catalog entries and its NIM compose file. Use Nemotron 3.5 Lightning instead.

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
- **Recipe-style Compose profiles**: `<example>` (cloud-only) and `<example>/<hardware>` (on-prem, e.g. `generic-assistant/dgx-spark`, `omni-assistant/jetson-thor`). Each is a complete, self-contained stack. `tracing` (Phoenix OTel) and `turn` (Coturn) remain optional overlays.

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
- NVIDIA Nemotron LLMs via NVIDIA NIM:
  - [Nemotron 3 Nano 30B A3B](https://build.nvidia.com/nvidia/nemotron-3-nano-30b-a3b/modelcard)
  - [Llama 3.3 Nemotron Super 49B v1.5](https://build.nvidia.com/nvidia/llama-3_3-nemotron-super-49b-v1_5/modelcard)
- WebRTC transport for real-time, low-latency voice communication with a custom frontend UI
- Docker Compose deployment with optional TURN server support for remote access
- Multilingual support with automatic language detection and seamless mid-conversation language switching
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
