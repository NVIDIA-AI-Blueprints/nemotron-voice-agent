# Multilingual - cascaded pipeline example

Multilingual cascaded voice pipeline using Pipecat's built-in NVIDIA services (`NvidiaSTTService` -> `NvidiaLLMService` -> `NvidiaTTSService`). The session is locked to a single language for the whole connection (selected in the UI, default `de-DE`): the ASR, the TTS voice, and the LLM all operate in that one language. The LLM replies with plain spoken text, kept on-language by the fixed-session prompt addon and a per-turn reminder.

The pattern uses dedicated ASR, LLM, and TTS services with a plain-text response, a per-turn language reminder injected at request time only, and a clean chat history that stores just the spoken reply.

![Architecture Diagram](../../../docs/images/arch.png)

## Running the example

This example runs on the **Cloud** (no local GPU, NVCF endpoints), **Workstation** (single GPU), and **DGX Spark** (Blackwell, 128 GB unified memory) profiles and can be extended to **Jetson Thor**, refer the [Jetson Thor guide](../../../docs/03-jetson-thor.md). See the [Getting Started guide](../../../docs/01-getting-started.md) for prerequisites and hardware detail. Run every command from the repository root.

1. Create your `.env` from the template and set your NVIDIA API key:

   ```bash
   cp .env.example .env
   export NVIDIA_API_KEY=<your-nvidia-api-key>
   ```

   > **DGX Spark:** also set `HF_TOKEN` in `.env`. The DGX Spark recipe serves the LLM with vLLM, which downloads the model weights from Hugging Face.

2. Log in to the NVIDIA NGC container registry:

   ```bash
   printf '%s' "$NVIDIA_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin
   ```

3. Deploy the profile that matches your hardware. The on-prem recipes start a Nemotron ASR Streaming Multilingual sidecar:

   ```bash
   docker compose --profile multilingual-assistant up -d              # Cloud (no local GPU)
   docker compose --profile multilingual-assistant/workstation up -d  # Workstation
   docker compose --profile multilingual-assistant/dgx-spark up -d    # DGX Spark
   ```

   | Recipe profile | App service | Sidecars |
   | --- | --- | --- |
   | `multilingual-assistant` | `multilingual-assistant` | none (cloud NVCF) |
   | `multilingual-assistant/workstation` | `multilingual-assistant` | `nvidia-llm`, `nemotron-asr-streaming-multilingual`, `tts-service` |
   | `multilingual-assistant/dgx-spark` | `multilingual-assistant` | `nvidia-llm-vllm`, `nemotron-asr-streaming-multilingual`, `tts-service` |

4. Open the UI at `https://localhost:7860/`. Keep TLS enabled for browser UI testing. `PIPELINE_TLS=false` serves plain HTTP for headless performance and API testing. For plain-HTTP browser testing, see [browser access](../../../docs/06-troubleshooting.md#browser-access).

5. Clean up when you are done by tearing down with the same profile you started with:

   ```bash
   docker compose --profile multilingual-assistant/workstation down
   ```

To run host-native without Docker, set `selection: multilingual-assistant` in [`examples_registry.yaml`](../../../examples_registry.yaml), then run `uv run python3 src/server.py`.

After deploying, validate the session language with the steps in [Testing](#testing).

## Customization

On-prem recipes default to **Nemotron ASR Streaming Multilingual** (`nemotron-asr-streaming-multilingual`) via `examples_registry.yaml` and `services.local.yaml`. The `multilingual-assistant/workstation` and `/dgx-spark` recipe profiles start that sidecar locally. There is no NVCF endpoint for it, so the cloud recipe falls back to **Parakeet 1.1B RNNT Multilingual** (`parakeet-rnnt`), the only multilingual ASR available on NVCF.

TTS voices and supported language codes are discovered at runtime by prewarming the configured TTS service, and the UI language selector is populated from the languages shared by the ASR and TTS services. The selected session language is injected into the prompt and pins the ASR and the TTS voice for the whole connection.

| Path | Role |
| --- | --- |
| `pipeline.py` | pipecat entry point, multilingual mode always on |
| `prompts.yaml` | multilingual prompt catalog (`multilingual_voice_assistant`) |
| `services.cloud.yaml` | cloud service endpoints and defaults |
| `services.local.yaml` | on-prem service endpoints (workstation / dgx-spark), registry default `nemotron-asr-streaming-multilingual` |

### How it works

1. The user selects the session language in the UI (default `de-DE`) before connecting.
2. The ASR and the TTS voice are pinned to that language when the connection starts. They do not change mid-session.
3. The fixed-session prompt addon instructs the LLM to reply only in that language, and the LLM returns plain spoken text (no JSON, labels, or metadata) that flows straight to TTS, the client transcript, and chat history.
4. `PerTurnReminderProcessor` re-states the "reply only in <language>" reminder on each user turn at request time only, so the reminder never pollutes stored history.

### Switching the multilingual ASR model

**Parakeet 1.1B RNNT Multilingual** offers stronger multilingual recognition quality at higher latency (see [Model Selection Notes](#model-selection-notes)). To run it instead of the default Nemotron ASR Streaming Multilingual on-prem:

1. In [`examples_registry.yaml`](../../../examples_registry.yaml), under `multilingual-assistant`, set `defaults.asr: [parakeet-rnnt]`.
2. Redeploy with the recipe profile plus the Parakeet profile, scaling the Nemotron sidecar off (only one local ASR may bind port `50152`):

   ```bash
   # Workstation
   docker compose --profile multilingual-assistant/workstation \
     --profile parakeet-rnnt-asr up -d --scale nemotron-asr-streaming-multilingual=0

   # DGX Spark
   docker compose --profile multilingual-assistant/dgx-spark \
     --profile parakeet-rnnt-asr up -d --scale nemotron-asr-streaming-multilingual-dgx-spark=0
   ```

3. Switch back to Nemotron by reversing the registry edit and redeploying the stock on-prem recipe.

## Tips & best practices

### Model Selection Notes

Multilingual behavior depends on the ASR model, the LLM, and the selected TTS voice. Use the notes below when choosing a deployment profile or setting expectations for demo and validation runs.

| Component | Recommendation and trade-offs |
| --- | --- |
| Nemotron ASR Streaming Multilingual | Prefer this model when latency and throughput are the main constraints. It is faster in this pipeline, but recognition quality is currently weaker for a few languages. Since the session language is fixed, its pinned-language recognition is a good fit here. In noisy environments, it can occasionally emit an empty transcript for turns, so the user may need to repeat themselves. A good microphone and reduced background noise help. |
| Parakeet 1.1B RNNT Multilingual | Prefer this model when multilingual recognition quality matters more than raw latency. Hindi and Chinese recognition are generally better than Nemotron ASR in this setup. The trade-off is slower latency and throughput. It can also miss the first word of an utterance in some cases and may produce occasional false transcripts when the microphone is muted or no user speech is intended, so validate turn-start and silence handling for production. |
| Nemotron 3 Super LLM | **Recommended for multilingual.** Stays more reliably in the fixed session language and delivers better conversation quality across languages, especially where Nemotron 3 Nano is weak (for example Hindi). Generally more concise as well. |
| Nemotron 3 Nano LLM | Useful for lower latency, lower resource usage, and faster local experiments. Its conversation quality is weaker in some languages (for example Hindi), and with reasoning disabled it can occasionally slip in foreign words on quantized builds, so the fixed-session prompt addon and the per-turn reminder both enforce a single language. Prefer Nemotron 3 Super when multilingual quality matters. |

### Testing

1. Start the app with the `multilingual-assistant` profile.
2. In Voice Settings, pick the session language (for example German, French, or Spanish), then connect.
3. Speak in the selected language and verify the bot responds in that same language.
4. Verify that:
   - the bot always responds in the selected session language, regardless of the language you speak
   - the transcript shows the clean spoken text
   - changing the language requires disconnecting, selecting a new language, and connecting again

### Troubleshooting

| Issue | Cause | What to check |
|-------|-------|---------------|
| Bot responds in the wrong language | LLM ignored the fixed session language | Confirm the fixed-session prompt addon and per-turn reminder name the selected language. Try the larger Nemotron 3 Super LLM |
| Bot slips in foreign words | Quantized small-model sampling artifacts | Lower the LLM `temperature` in `services.*.yaml`, or use a larger LLM |
| TTS uses the wrong voice or language | Selected session language is not supported by the active TTS service | Check the configured TTS service exposes that language code, or pick a supported language |
| No voices discovered at startup | TTS prewarm failed | Check TTS sidecar health (`docker compose ps`) and `NVIDIA_API_KEY` |
| Bot does not respond to a turn (no transcript) | Nemotron ASR Multilingual can drop a turn in noisy environments | Speak again, reduce background noise, and use a good microphone. See [Configure ASR](../../../docs/how-to/configure-asr.md#choosing-a-multilingual-asr-model) |
| Weak or awkward replies in some languages (for example Hindi) | Nemotron 3 Nano has weaker conversation quality in a few languages | Use Nemotron 3 Super for better multilingual quality. See [Configure LLM](../../../docs/how-to/configure-llm.md) |
| Port conflict on the ASR sidecar | Parakeet and Nemotron streaming both bind `50152` | Run only one local ASR. When opting into Parakeet, scale the Nemotron sidecar off (`--scale nemotron-asr-streaming-multilingual=0`) |
| Random ASR text while silent | Parakeet RNNT noise sensitivity | Expected with the Parakeet opt-in. The default Nemotron ASR is less prone to this. Otherwise reduce room noise and use a good mic |

For ASR, LLM, and TTS model details and general failure modes, see [Configure ASR](../../../docs/how-to/configure-asr.md), [Configure TTS](../../../docs/how-to/configure-tts.md), [Configure LLM](../../../docs/how-to/configure-llm.md), and the [Troubleshooting guide](../../../docs/06-troubleshooting.md).
