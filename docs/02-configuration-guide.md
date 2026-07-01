# Configuration Guide

Use the following guides to configure the Nemotron Voice Agent. Each guide is task-focused and includes the steps to enable or configure specific features.

## Choose Your Path

- First deployment: start with [Getting Started](./01-getting-started.md).
- Edge deployment: use [Deploying Voice Agent on Jetson Thor](./03-jetson-thor.md).
- Remote browser access: configure TURN in [Getting Started](./01-getting-started.md#optional-deploy-turn-server-for-remote-access).
- Benchmarking: switch to WebSocket in [Choose a Transport Method](./how-to/choose-transport-method.md), then use [Evaluation and Performance](./06-evaluation-and-performance.md).
- Production readiness: review [Best Practices](./04-best-practices.md).
- Deployment failures: see [Troubleshooting](./07-troubleshooting.md).

## How-To Guides

| Guide | Description |
|-------|-------------|
| [Switch LLM Models](./how-to/switch-llm-models.md) | Change the LLM backend between local NIM microservices and NVIDIA cloud endpoints. |
| [Customize System Prompts](./how-to/customize-system-prompts.md) | Select pre-built prompt samples or create custom prompts to define your agent's personality and behavior. |
| [Enable Multilingual Voice Agent](./how-to/enable-multilingual.md) | Deploy the agent with automatic language detection and multilingual responses. |
| [Configure TTS Settings](./how-to/configure-tts-settings.md) | Set up TTS voice, pronunciation correction (IPA), cloud TTS endpoints, and text filters. |
| [Enable Zero-Shot TTS](./how-to/enable-zero-shot-tts.md) | Clone any voice from a short audio sample using the Magpie Zero-shot model. |
| [Choose a Transport Method](./how-to/choose-transport-method.md) | Switch between WebRTC (default) and WebSocket transport. |
| [Enable OpenTelemetry Tracing](./how-to/enable-opentelemetry-tracing.md) | Add observability with Phoenix to monitor pipeline performance and conversation flows. |
| [Tune Pipeline Performance](./how-to/tune-pipeline-performance.md) | Adjust speculative speech, chat history, audio debugging, and output buffering settings. |

## Reference Guides

| Guide | Description |
|-------|-------------|
| [NVIDIA Pipecat Services](./05-nvidia-pipecat.md) | Reference for the NVIDIA Pipecat services used by the blueprint. |
| [Evaluation and Performance](./06-evaluation-and-performance.md) | Benchmarking entry point and reference performance results. |
| [Troubleshooting](./07-troubleshooting.md) | Common deployment, service health, WebRTC, and model startup issues. |
