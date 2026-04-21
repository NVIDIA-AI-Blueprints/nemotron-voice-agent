# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
  - [Best practices guide](docs/04-best-practices.md) covering production deployment, latency optimization, and conversational UX
- AI agent deployment skill for Cursor and Claude Code to streamline deployment on workstations and Jetson Thor

### Known Issues

- ASR transcription can occasionally be inaccurate, though the LLM generally compensates by inferring meaning from context.
- The context aggregator limits chat history to 20 turns by default. Older turns are dropped when this limit is reached, rather than summarized.
