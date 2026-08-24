---
name: create-voice-agent
description: Create or refine an NVIDIA voice agent with Pipecat or LiveKit. Covers cascaded and omni pipelines, domain speech customization, NVIDIA cloud, GPU workstations, DGX Spark, and Jetson Thor. Use when asked to build, scaffold, start, or modify a voice agent, voice bot, speech assistant, real-time audio pipeline, ASR word boosting, or TTS pronunciation.
version: "2.1.1"
metadata:
  author: NVIDIA Voice Agent Team <nemotron-voice-agent@nvidia.com>
  tags: [voice-agent, nvidia, nemotron, magpie, pipecat, livekit, nim, omni]
---

# Create Voice Agent

Builds a working NVIDIA voice agent in an empty project:

- **Cascaded**: ASR transcribes, a text LLM answers, TTS speaks.
- **Omni**: one audio-in LLM replaces ASR and the text LLM. TTS still speaks.

## Workflow

Follow these phases in order:

| Phase | Read |
| --- | --- |
| 1. Intake | `references/intake.md` resolves pipeline and framework first, then routes preflight, models, language, and domain files |
| 2. Build | `references/platforms/readiness.md` if self-hosted → exact profile in the routed model file → `references/output-contract.md` → routed platform + `references/platforms/deployment.md` → `references/operations/observability.md` → selected framework files |
| 3. Hand over | `references/operations/run.md` gates the client on the generated `scripts/smoke.sh`, then `references/operations/troubleshoot.md` if the spoken exchange fails |
| 4. Iterate | `references/operations/iterate.md` for changes to a working agent |

Wait for approval before writing files.

During build, keep the approved language and behavior locks from intake. Read every routed
file before generating code. Cloud deployment skips readiness, `list-model-profiles`,
Compose, and the shared-GPU gate; wire endpoints per `references/platforms/deployment.md`
§Cloud. Use `references/networking/remote-webrtc.md` when Pipecat WebRTC crosses hosts or
networks.

During handover, require a successful spoken exchange. A running process alone is not a
working voice agent.
