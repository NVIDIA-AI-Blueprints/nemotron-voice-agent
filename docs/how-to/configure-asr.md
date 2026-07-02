# ASR Models

The cascaded pipeline transcribes user speech with a streaming **ASR** service. (The **Omni** examples handle ASR using the Omni model itself. See [LLM Models](configure-llm.md).) The ASR models are NVIDIA **Parakeet / Nemotron** speech models, served from the cloud (NVIDIA-hosted NVCF endpoints) or self-hosted next to the pipeline as an [**NVIDIA NIM for Speech**](https://docs.nvidia.com/nim/speech/latest/asr/index.html) sidecar.

ASR services are declared per example in `services.cloud.yaml` (remote / NVCF) and `services.local.yaml` (Compose-managed sidecars). This page is the **model reference**: what's available, how to size it, how to customize speech recognition, and known failure modes. For how the catalog is loaded, switched in the UI, and overridden, see [Configure Services](configure-services.md).

## Models

| Model | Self-hosted compose service | Modelcard |
|-------|-----------------------------|-----------|
| **Nemotron ASR Streaming (English)**: default, low-latency streaming ASR, English only | [`docker-compose.nemotron-asr.yaml`](../../docker/docker-compose.nemotron-asr.yaml) | [model card](https://build.nvidia.com/nvidia/nemotron-asr-streaming/modelcard) |
| **Nemotron ASR Streaming (Multilingual)**: cache-aware streaming multilingual ASR covering 40 language locales | [`docker-compose.nemotron-asr.yaml`](../../docker/docker-compose.nemotron-asr.yaml) | [model card](https://build.nvidia.com/nvidia/nemotron-asr-streaming/modelcard) |
| **Parakeet CTC 1.1B**: English-only ASR | [`docker-compose.parakeet-asr.yaml`](../../docker/docker-compose.parakeet-asr.yaml) | [model card](https://build.nvidia.com/nvidia/parakeet-ctc-1_1b-asr/modelcard) |
| **Parakeet 1.1B RNNT Multilingual**: multilingual ASR (25+ languages)  | [`docker-compose.parakeet-asr.yaml`](../../docker/docker-compose.parakeet-asr.yaml) | [model card](https://build.nvidia.com/nvidia/parakeet-1_1b-rnnt-multilingual-asr/modelcard) |

Each model is exposed as a **catalog key** in `services.cloud.yaml` / `services.local.yaml`:

| Model | Catalog key |
|-------|-------------|
| Nemotron ASR Streaming (English) | `nemotron-asr-streaming-english` |
| Nemotron ASR Streaming (Multilingual) | `nemotron-asr-streaming-multilingual` |
| Parakeet CTC 1.1B | `parakeet-ctc` |
| Parakeet 1.1B RNNT Multilingual | `parakeet-rnnt` |

> The active default per slot is set in [`examples_registry.yaml`](../../examples_registry.yaml) (`defaults`).

### Choosing a multilingual ASR model

For multilingual deployments, choose between the two multilingual ASR models based on whether latency or recognition quality matters more:

| Model | Recommendation and trade-offs |
| --- | --- |
| Nemotron ASR Streaming Multilingual | Prefer this model when latency and throughput are the main constraints. It is faster in this pipeline, but recognition quality is currently weaker for a few languages, and language auto-detection is less reliable. For best results, preselect the session language instead of relying on auto-detection. |
| Parakeet 1.1B RNNT Multilingual | Prefer this model when multilingual recognition quality matters more than raw latency. Language auto-detection is relatively stronger, and Hindi and Chinese recognition are generally better than Nemotron ASR in this setup. The trade-off is slower latency and throughput. It can also miss the first word of an utterance in some cases and may produce occasional false transcripts when the microphone is muted or no user speech is intended, so validate turn-start and silence handling for production. |

## Hardware requirements and deployment configs

ASR runs one of three ways, and the repo wires the right one per profile:

- **Cloud (NVCF)**: no local GPU, and the catalog calls `grpc.nvcf.nvidia.com`. The simplest starting point.
- **NIM for Speech sidecar**: an ASR NIM microservice on the `*/workstation` and `*/dgx-spark` profiles, on GPU `0` by default ([`docker-compose.nemotron-asr.yaml`](../../docker/docker-compose.nemotron-asr.yaml), [`docker-compose.parakeet-asr.yaml`](../../docker/docker-compose.parakeet-asr.yaml)).
- **Riva embedded (Jetson Thor)**: on `*/jetson-thor`, Riva Embedded SDK (`nemotron-speech`, `docker-compose.speech-jetson.yaml`) serves **ASR + TTS together**. See [Jetson Thor](../03-jetson-thor.md) and [Riva embedded ASR](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/quick-start-guide/asr.html).

> Check the **[ASR support matrix](https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/asr.html)** for supported GPUs and VRAM before choosing a model. ASR NIMs run on compute capability **≥ 8.0** (Ampere and newer) with **≥ 16 GB** VRAM.

### VRAM & hardware support

The ASR sidecar uses roughly **~15 GB VRAM** and, on local profiles, runs alongside the LLM and TTS. On a single ~80 GB GPU, ASR (~15 GB) + TTS (~14 GB) + the LLM (~30 GB FP8) fit together. If they do not, move ASR/TTS to a second GPU via their `device_ids` in [`docker-compose.nemotron-asr.yaml`](../../docker/docker-compose.nemotron-asr.yaml). See [Configure LLM → VRAM & hardware support](configure-llm.md#vram--hardware-support) for the full multi-GPU layout.

### Performance

For ASR latency and throughput across GPUs and WER for different models, see the **[ASR performance benchmarks](https://docs.nvidia.com/nim/speech/latest/reference/performances/asr/performance.html)**. For end-to-end pipeline latency in this blueprint, see [Evaluation and Performance](../04-evaluation-and-performance.md).

## Customization

- **Word boosting**: bias recognition toward domain terms (product names, acronyms, jargon) at request time, without retraining. NIM ASR also supports profanity filtering, automatic punctuation, inverse text normalization, custom vocabularies, and fine-tuned models. See [ASR customization](https://docs.nvidia.com/nim/speech/latest/asr/customization/customization.html).

  *Enable it in an example:* the examples build ASR with Pipecat's `NvidiaSTTService`, which exposes boosting through `NvidiaSTTSettings`. Add `boosted_lm_words` / `boosted_lm_score` where that settings object is constructed in the example's `pipeline.py` (e.g. [`src/examples/generic/pipeline.py`](../../src/examples/generic/pipeline.py)).

  ```python
  asr_kwargs["settings"] = NvidiaSTTSettings(
      language=asr_language_code or "en-US",
      boosted_lm_words=["Nemotron", "NVIDIA", "Pipecat"],
      boosted_lm_score=20.0,
  )
  ```

- **Endpointing (end-of-utterance)**: Riva/NIM ASR decides when the user has stopped speaking from trailing silence, via the endpoint parameters `start_history` / `start_threshold`, `stop_history` / `stop_threshold`. Silence windows are in ms, a multiple of 80, and `-1` keeps the model defaults. Shorter `stop_history` finalizes faster (lower latency, but may clip trailing words). Each example already passes one, `NvidiaSTTService(**asr_kwargs, stop_history=400)` in its `pipeline.py`, so tune endpointing for your use case:

  ```python
  stt = NvidiaSTTService(
      **asr_kwargs,
      stop_history=400,       # ms trailing silence before finalizing (repo default; ≥560 favors accuracy)
  )
  ```

  This blueprint's turn-taking is driven mainly by pipeline-level [Smart Turn / Silero VAD](tune-pipeline-performance.md#smart-turn-detection). ASR endpointing is the lower-level, ASR-side signal. See [ASR customization](https://docs.nvidia.com/nim/speech/latest/asr/customization/customization.html) for exact semantics, defaults, and the `force_eou` runtime flag.

- **Language**: multilingual streaming uses `language_code: auto` for per-turn language detection. Pin a fixed locale (e.g. `es-US`) to force one language. For the full set a model supports, see its per-model supported-language table in the [ASR support matrix](https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/asr.html) (Nemotron ASR Streaming covers 40 locales, Parakeet RNNT Multilingual 25+).
- **Catalog config**: a cloud entry sets `server` / `model` / `function_id`, while a local entry points at the Compose sidecar `host:port`. Host-run deployments rewrite sidecar endpoints to `localhost` automatically. See [Configure Services → On-prem catalog](configure-services.md#on-prem-catalog).

```yaml
asr:
  my-custom-asr:
    name: "My Custom ASR"
    server: "grpc.nvcf.nvidia.com:443"   # cloud; local entries use the sidecar host:port
    model: "my-asr-model"
    function_id: ""
```

## Reference

- [Troubleshooting guide → ASR](../06-troubleshooting.md#asr-speech-to-text): Out of Memory, wrong language, model not found for language, no transcription.
- [Configure Services](configure-services.md): how the catalog is loaded, switched, and overridden.
- [Multilingual example](../../src/examples/multilingual/README.md): multilingual ASR/TTS behavior and example-specific troubleshooting.
- [NVIDIA NIM for Speech — ASR](https://docs.nvidia.com/nim/speech/latest/asr/index.html): [customization / word boosting](https://docs.nvidia.com/nim/speech/latest/asr/customization/customization.html), [support matrix](https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/asr.html), [performance benchmarks](https://docs.nvidia.com/nim/speech/latest/reference/performances/asr/performance.html), [ASR troubleshooting](https://docs.nvidia.com/nim/speech/latest/troubleshooting/asr.html).
- [Pipecat NVIDIA ASR service](https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/services/nvidia/stt.py): `NvidiaSTTService`.
- [Riva embedded (Jetson) ASR](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/quick-start-guide/asr.html): L4T / Jetson Thor ASR via the Riva quick-start.
