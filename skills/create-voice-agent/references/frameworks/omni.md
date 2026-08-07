# Omni

Read with `frameworks/pipecat.md` when the approved pipeline is Omni. LiveKit is not
supported.

Omni replaces ASR and the text LLM with one audio-in model. TTS remains separate:

```text
transport → user context → Omni service → TTS → transport → assistant context
```

Do not add an ASR service, ASR NIM, or transcription-dependent turn strategy.

## Sources

Pipecat does not provide the required Omni service. Use NVIDIA's
[`nvidia_omni_multimodal_service.py`](https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent/blob/main/src/examples/omni_assistant/nvidia_omni_multimodal_service.py)
directly and build the pipeline around `NvidiaOmniMultimodalService`. Copy the current
upstream file into the generated project. Do not reimplement or simplify it.

Also copy NVIDIA's current
[`audio_only_smart_turn_strategy.py`](https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent/blob/main/src/examples/omni_assistant/audio_only_smart_turn_strategy.py).
Pipecat's stock Smart Turn stop strategy waits for an upstream `TranscriptionFrame`, which
does not exist before an Omni audio turn completes.

Generate the surrounding Pipecat pipeline, transport, context, turn handling, and TTS
wiring from the Pipecat docs MCP. Use NVIDIA's current
[`omni_assistant/pipeline.py`](https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent/blob/main/src/examples/omni_assistant/pipeline.py)
as the behavioral reference. Read the locked model's API reference or Hugging Face model
card for endpoint settings.

## Model and deployment

Resolve the exact model through `models/llm.md`. Keep reasoning off for voice.

| Host | Source |
| --- | --- |
| Cloud | model id from `/v1/models`, API shape from its build.nvidia.com reference |
| Workstation / DGX Spark | current self-host instructions from the locked model page or card |
| Jetson Thor | Omni Hugging Face card and vLLM path in `platforms/jetson-thor.md` |

Budget Omni + TTS only. `NIM_MODEL_PROFILE` applies only when the chosen self-hosted page
actually deploys a NIM. It never applies to raw vLLM.

A self-hosted Omni endpoint hits the same client traps as the cascaded LLM in
`frameworks/pipecat.md` §Local LLM wiring: an API-key argument may be required even
locally, reasoning-off has to arrive nested inside `extra_body`, and the model string is
the served id from the endpoint rather than the catalog slug.

For domain vocabulary, follow `domain/speech-customization.md`. Only TTS pronunciation
customization applies because Omni has no ASR slot.

## Turn handling

Wire transport audio and context into `NvidiaOmniMultimodalService`, then route its text
output to TTS.

- Start turns from VAD and stop them with `AudioOnlySmartTurnStopStrategy`.
- Use Pipecat's current `MuteUntilFirstBotCompleteUserMuteStrategy` so microphone input
  cannot race the first response.
- Follow the current upstream pipeline for the connect greeting and use one greeting path
  only. The current Omni service supports text turns.
- When user transcripts are enabled, request the structured response shape expected by
  the upstream service. It parses the JSON and sends only `response` text to TTS.
- Keep reasoning off through the request settings in `models/llm.md`.

## Verify

- `scripts/smoke.sh` passes, using the Omni audio request rather than a chat completion.
- No ASR service or ASR sidecar exists.
- The upstream Omni service and audio-only Smart Turn strategy were copied unchanged.
- The model endpoint serves the locked Omni model, and the agent uses its exact served id.
- User audio produces a coherent response and, when supported, a user transcription.
- Exactly one greeting plays and microphone input begins after it completes.
- TTS never speaks reasoning, JSON wrappers, or transcript metadata.
- Barge-in cancels the current Omni response.
- Complete the spoken exchange in `operations/run.md`.

## Anti-patterns

- LiveKit Omni.
- Stock text-only LLM service or separate ASR used in place of Omni.
- Rewriting `NvidiaOmniMultimodalService`.
- Stock Smart Turn stop strategy waiting for a transcription.
- Reasoning left at the model default.
