# ASR Models

The cascaded pipeline transcribes user speech with an **ASR** service. (The **Omni** examples handle ASR using the Omni model itself. See [LLM Models](configure-llm.md).) The ASR models are NVIDIA **Parakeet / Nemotron** speech models, served from the cloud (NVIDIA-hosted NVCF endpoints) or self-hosted next to the pipeline as an [**NVIDIA NIM for Speech**](https://docs.nvidia.com/nim/speech/latest/asr/index.html) sidecar.

ASR services are declared in the root [`services.yaml`](../../services.yaml), and each example lists the keys it offers in [`examples_registry.yaml`](../../examples_registry.yaml). This page is the **model reference**: what's available, how to size it, how to customize speech recognition, and known failure modes. For how the catalog is loaded, switched in the UI, and overridden, see [Configure Services](configure-services.md).

## Models

| Model | Self-hosted compose service | Modelcard |
|-------|-----------------------------|-----------|
| **Nemotron ASR Streaming (English)**: default, low-latency streaming ASR, English only | [`docker-compose.nemotron-asr.yaml`](../../docker/docker-compose.nemotron-asr.yaml) | [model card](https://build.nvidia.com/nvidia/nemotron-asr-streaming/modelcard) |
| **Nemotron ASR Streaming (Multilingual)**: cache-aware streaming multilingual ASR covering 40 language locales | [`docker-compose.nemotron-asr.yaml`](../../docker/docker-compose.nemotron-asr.yaml) | [model card](https://build.nvidia.com/nvidia/nemotron-asr-streaming/modelcard) |
| **Parakeet CTC 1.1B**: English-only ASR, self-hosted only | [`docker-compose.parakeet-asr.yaml`](../../docker/docker-compose.parakeet-asr.yaml) | [model card](https://build.nvidia.com/nvidia/parakeet-ctc-1_1b-asr/modelcard) |
| **Parakeet 1.1B RNNT Multilingual**: multilingual ASR (25+ languages)  | [`docker-compose.parakeet-asr.yaml`](../../docker/docker-compose.parakeet-asr.yaml) | [model card](https://build.nvidia.com/nvidia/parakeet-1_1b-rnnt-multilingual-asr/modelcard) |
| **Nemotron 3.5 Transcribe**: whole-turn ASR in Generic and Multilingual Assistants, self-hosted only | [`docker-compose.nemotron-transcribe.yaml`](../../docker/docker-compose.nemotron-transcribe.yaml) | [model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B) |

Each model is exposed as a **catalog key** in `services.yaml`:

| Model | Catalog key |
|-------|-------------|
| Nemotron ASR Streaming (English) | `nemotron-asr-streaming-english` |
| Nemotron ASR Streaming (Multilingual) | `nemotron-asr-streaming-multilingual` |
| Parakeet CTC 1.1B | `parakeet-ctc` (self-hosted only) |
| Parakeet 1.1B RNNT Multilingual | `parakeet-rnnt` |
| Nemotron 3.5 Transcribe | `nemotron-transcribe` (self-hosted only) |

> The active default per slot is the first available key under the example's `services` in [`examples_registry.yaml`](../../examples_registry.yaml).

### Choosing a multilingual ASR model

For multilingual deployments, select the ASR model based on your language requirements and quality expectations:

| Model | When to use |
| --- | --- |
| Nemotron ASR Streaming Multilingual | Best choice when latency is the priority. Recognition quality varies by language, and auto language detection is less reliable. Specify the session language explicitly for best results. In noisy environments, it can occasionally emit an empty transcript for turns, so the user may need to repeat themselves. |
| Nemotron 3.5 Transcribe | Whole-turn ASR in Generic and Multilingual Assistants. Refer to the [model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B) for supported languages. |
| Parakeet 1.1B RNNT Multilingual | Better recognition quality for many languages, with better auto language detection. Trade-off is higher latency. Note: may occasionally miss the first word of an utterance or produce spurious transcripts during silence. |

**Guidance:**
- For English, use the English-only model.
- For streaming ASR, do not rely on auto language detection. Explicitly set the target language when possible.
- Models behave differently across languages. Test your target language and choose based on observed recognition quality.
- For Nemotron ASR Streaming Multilingual, use only languages listed as transcription-ready in the [supported languages table](https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/asr.html#asr-nemotron-asr-streaming-supported-languages). Other locales may produce degraded results.
- For Parakeet 1.1B RNNT Multilingual, see the [supported languages table](https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/asr.html#asr-parakeet-11b-rnnt-multilingual-supported-languages).

**Other multilingual models to consider:**

The [ASR support matrix](https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/asr.html) lists additional multilingual NIM models that may better fit specific language targets, including:
- **Parakeet RNNT Indic**: optimized for Indic languages.
- **Code-switching models**: for mixed-language speech within a single utterance.

To use one of these, configure the NIM endpoint to point to the corresponding model and update the catalog entry in `services.yaml`.



## Hardware requirements and deployment configs

ASR runs one of three ways, and the repo wires the right one per profile:

- **Cloud (NVCF)**: no local GPU, and the catalog calls `grpc.nvcf.nvidia.com`. The simplest starting point.
- **NIM for Speech sidecar**: an ASR NIM microservice on the `*/server` profiles, on GPU `0` by default ([`docker-compose.nemotron-asr.yaml`](../../docker/docker-compose.nemotron-asr.yaml), [`docker-compose.parakeet-asr.yaml`](../../docker/docker-compose.parakeet-asr.yaml)).
- **Opt-in local ASR (Transcribe)**: enable the `transcribe-asr` Compose profile with the server recipe of the Generic or Multilingual Assistant ([`docker-compose.nemotron-transcribe.yaml`](../../docker/docker-compose.nemotron-transcribe.yaml), GPU `1`, HTTP port `9002`). Then select **Nemotron 3.5 Transcribe** in **Services > ASR**.

    Start Transcribe together with the server recipe of your example:

    ```bash
    docker compose --profile generic-assistant/server --profile transcribe-asr up -d
    docker compose --profile multilingual-assistant/server --profile transcribe-asr up -d
    ```

    Pass the same profiles to `docker compose down`. Transcribe is available only with the server recipe. The cloud and single-GPU recipes, including DGX Spark and Jetson Thor, do not include it.

    Transcribe writes the transcript in the spoken language. The session language does not change it.

    In the Multilingual Assistant, the language list shows only languages that Transcribe, the TTS, and the LLM all support.

    External NIMs must not require an API key because the pipeline does not pass one. Set `server.asr.nemotron-transcribe.server` to the HTTP API URL of the NIM, including `/v1`. Keep `asr_type: transcribe`. Transcribe produces final transcripts without speaker labels. In Generic Assistant, leave `transcription_prompt` unset for audio-only requests, or use an official NIM transcription template.

    In rare cases, Transcribe can follow an instruction spoken in the audio (for example, "answer in one word") instead of transcribing it.
- **NeMo-Speech.cpp (single GPU, including Jetson Thor)**: on `*/single-gpu`, one sidecar serves **ASR + TTS together** from local GGUF weights (`nemo-speech` for English, `nemo-speech-multilingual` for the multilingual example, both in `docker-compose.nemo-speech-cpp.yaml` and both reachable as `nemo-speech`). See [Jetson Thor](../03-jetson-thor.md).

> Check the **[ASR support matrix](https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/asr.html)** for supported GPUs and VRAM before choosing a model. ASR NIMs require compute capability **≥ 8.0** and **≥ 16 GB** VRAM.

### VRAM & hardware support

The streaming ASR sidecar uses roughly **~15 GB VRAM** and, on local profiles, runs alongside the LLM and TTS. Standard `*/server` deployments let NIM select the LLM profile automatically, so its memory footprint depends on the detected GPU and selected precision. Confirm the selected profile before assuming ASR (~15 GB), TTS (~14 GB), and the LLM fit on one ~80 GB GPU. If they don't fit, move the speech sidecars separately: update the ASR `device_ids` in [`docker-compose.nemotron-asr.yaml`](../../docker/docker-compose.nemotron-asr.yaml) and the selected TTS service's `device_ids` in its Compose file. See the [Server NIM device-placement guidance](../../.agents/skills/nemotron-voice-agent-deploy/references/server.md#precision) and [Configure LLM → VRAM & hardware support](configure-llm.md#vram--hardware-support) for the complete multi-GPU layout.

Transcribe runs on its own GPU (GPU 1 by default). Refer to the [Transcribe NIM support matrix](https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/asr.html#nemotron-transcribe) for supported GPUs and memory.

The Transcribe NIM support matrix does not list Jetson Thor. It lists DGX Spark only as a planned tier without measured settings, so this blueprint does not support Transcribe on DGX Spark.

### Performance

For ASR latency and throughput across GPUs and WER for different models, see the **[ASR performance benchmarks](https://docs.nvidia.com/nim/speech/latest/reference/performances/asr/performance.html)**. For end-to-end pipeline latency in this blueprint, see [Evaluation and Performance](../04-evaluation-and-performance.md).

## Customization

- **Word boosting (streaming ASR only)**: bias recognition toward domain terms (product names, acronyms, jargon) at request time, without retraining. NIM ASR also supports profanity filtering, automatic punctuation, inverse text normalization, custom vocabularies, and fine-tuned models. See [ASR customization](https://docs.nvidia.com/nim/speech/latest/asr/customization/customization.html).

  *Enable it in an example:* streaming ASR uses Pipecat's `NvidiaSTTService`, which exposes boosting through `NvidiaSTTSettings`. Add `boosted_lm_words` / `boosted_lm_score` where that settings object is constructed in the example's `pipeline.py` (e.g. [`src/examples/generic/pipeline.py`](../../src/examples/generic/pipeline.py)).

  ```python
  asr_kwargs["settings"] = NvidiaSTTSettings(
      language=asr_language_code or "en-US",
      boosted_lm_words=["Nemotron", "NVIDIA", "Pipecat"],
      boosted_lm_score=20.0,
  )
  ```

- **Endpointing (end-of-utterance, streaming ASR only)**: Riva/NIM ASR decides when the user has stopped speaking from trailing silence, via the endpoint parameters `start_history` / `start_threshold`, `stop_history` / `stop_threshold`. Silence windows are in ms, a multiple of 80, and `-1` keeps the model defaults. Shorter `stop_history` finalizes faster but can reduce recognition accuracy. The cascaded examples construct the stock `NvidiaSTTService` with `stop_history=320`. To customize this value, pass your chosen value in the example pipeline:

  ```python
  stt = NvidiaSTTService(
      **asr_kwargs,
      stop_history=320,  # ms trailing silence before ASR endpointing
  )
  ```

  This blueprint's turn-taking is driven mainly by pipeline-level [Smart Turn / Silero VAD](tune-pipeline-performance.md#smart-turn-detection). ASR endpointing independently returns an `is_final` transcript after its trailing-silence window. Stock `NvidiaSTTService` response handling emits `TranscriptionFrame(finalized=True)` for that result, and Pipecat's turn analyzer combines the finalized transcript with Smart Turn's decision before closing the semantic user turn. Refer to [ASR customization](https://docs.nvidia.com/nim/speech/latest/asr/customization/customization.html) for the endpoint parameter semantics and defaults.

  The [Frontend/Backend Live example](../../src/examples/frontend_backend_live/README.md#configuration) sets `stop_history` in `build_stt` in [`media/services.py`](../../src/examples/frontend_backend_live/media/services.py).

- **Transcribe**: in Generic Assistant, use `transcription_prompt` for expected words. Multilingual Assistant sends no prompt. Both assistants rely on [Silero VAD](tune-pipeline-performance.md#smart-turn-detection) for turn ends.

- **Language**: the multilingual example locks each session to a single locale, selectable per connection in the UI (any locale the ASR, TTS, and selected LLM support, default `de-DE`) and fixed for that session. Different sessions can use different languages. The streaming ASR also accepts `language_code: auto` for per-turn detection, but this blueprint does not use it, since a pinned locale is more reliable. For the full set a model supports, see its per-model supported-language table in the [ASR support matrix](https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/asr.html) (Nemotron ASR Streaming covers 40 locales, Parakeet RNNT Multilingual 25+).
- **Punctuation**: the single-GPU NeMo-Speech.cpp entries reference the `nemo-speech-asr` profile in [`settings.yaml`](../../settings.yaml), so their card in the Services tab has a **Punctuation** toggle that defaults to on. NeMo-Speech.cpp ignores `profanity_filter` and does not apply inverse text normalization, so the profile exposes only punctuation.
- **Catalog config**: a `server` entry points at the Compose sidecar `host:port`, and its `nvcf` key adds the NVIDIA Cloud `function_id` (and `model` when it differs). Host-run deployments rewrite sidecar endpoints to `localhost` automatically. See [Configure Services → Adding a self-hosted service](configure-services.md#adding-a-self-hosted-service).

```yaml
server:
  asr:
    my-custom-asr:
      name: "My Custom ASR"
      server: "my-custom-asr:50052"
      model: "my-asr-model"
      nvcf:
        function_id: "<NVCF_FUNCTION_ID>"
```

## Reference

- [Troubleshooting guide → ASR](../06-troubleshooting.md#asr-speech-to-text): Out of Memory, wrong language, model not found for language, no transcription.
- [Configure Services](configure-services.md): how the catalog is loaded, switched, and overridden.
- [Multilingual example](../../src/examples/multilingual/README.md): multilingual ASR/TTS behavior and example-specific troubleshooting.
- [NVIDIA NIM for Speech — ASR](https://docs.nvidia.com/nim/speech/latest/asr/index.html): [customization / word boosting](https://docs.nvidia.com/nim/speech/latest/asr/customization/customization.html), [support matrix](https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/asr.html), [performance benchmarks](https://docs.nvidia.com/nim/speech/latest/reference/performances/asr/performance.html), [ASR troubleshooting](https://docs.nvidia.com/nim/speech/latest/troubleshooting/asr.html).
- [Pipecat NVIDIA ASR service](https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/services/nvidia/stt.py): `NvidiaSTTService`.
