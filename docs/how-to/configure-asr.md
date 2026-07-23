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

For multilingual deployments, select the ASR model based on your language requirements and quality expectations:

| Model | When to use |
| --- | --- |
| Nemotron ASR Streaming Multilingual | Best choice when latency is the priority. Recognition quality varies by language, and auto language detection is less reliable. Specify the session language explicitly for best results. In noisy environments, it can occasionally emit an empty transcript for turns, so the user may need to repeat themselves. |
| Parakeet 1.1B RNNT Multilingual | Better recognition quality for many languages, with better auto language detection. Trade-off is higher latency. Note: may occasionally miss the first word of an utterance or produce spurious transcripts during silence. |

**Guidance:**
- For English, use the English-only model.
- Do not rely on auto language detection. Explicitly set the target language when possible.
- Both models behave differently across languages. Test both with your target language and choose based on observed recognition quality.
- For Nemotron ASR Streaming Multilingual, use only languages listed as transcription-ready in the [supported languages table](https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/asr.html#asr-nemotron-asr-streaming-supported-languages). Other locales may produce degraded results.
- For Parakeet 1.1B RNNT Multilingual, see the [supported languages table](https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/asr.html#asr-parakeet-11b-rnnt-multilingual-supported-languages).

**Other multilingual models to consider:**

The [ASR support matrix](https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/asr.html) lists additional multilingual NIM models that may better fit specific language targets, including:
- **Parakeet RNNT Indic**: optimized for Indic languages.
- **Code-switching models**: for mixed-language speech within a single utterance.

To use one of these, configure the NIM endpoint to point to the corresponding model and update the catalog key in `services.cloud.yaml` or `services.local.yaml`.



## Hardware requirements and deployment configs

ASR runs one of three ways, and the repo wires the right one per profile:

- **Cloud (NVCF)**: no local GPU, and the catalog calls `grpc.nvcf.nvidia.com`. The simplest starting point.
- **NIM for Speech sidecar**: an ASR NIM microservice on the `*/workstation` and `*/dgx-spark` profiles, on GPU `0` by default ([`docker-compose.nemotron-asr.yaml`](../../docker/docker-compose.nemotron-asr.yaml), [`docker-compose.parakeet-asr.yaml`](../../docker/docker-compose.parakeet-asr.yaml)).
- **Riva embedded (Jetson Thor)**: on `*/jetson-thor`, Riva Embedded SDK (`nemotron-speech`, `docker-compose.speech-jetson.yaml`) serves **ASR + TTS together**. See [Jetson Thor](../03-jetson-thor.md) and [Riva embedded ASR](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/quick-start-guide/asr.html).

> Check the **[ASR support matrix](https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/asr.html)** for supported GPUs and VRAM before choosing a model. ASR NIMs run on compute capability **≥ 8.0** (Ampere and newer) with **≥ 16 GB** VRAM.

### VRAM & hardware support

The ASR sidecar uses roughly **~15 GB VRAM** and, on local profiles, runs alongside the LLM and TTS. On a single ~80 GB GPU, ASR (~15 GB) + TTS (~14 GB) + the LLM (~30 GB FP8) fit together. If they don't, move ASR/TTS to a second GPU via their `device_ids` in [`docker-compose.nemotron-asr.yaml`](../../docker/docker-compose.nemotron-asr.yaml). See [Configure LLM → VRAM & hardware support](configure-llm.md#vram--hardware-support) for the full multi-GPU layout.

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
      stop_history=400,       # ms trailing silence before finalizing (repo default. ≥560 favors accuracy)
  )
  ```

  This blueprint's turn-taking is driven mainly by pipeline-level [Smart Turn / Silero VAD](tune-pipeline-performance.md#smart-turn-detection). ASR endpointing is the lower-level, ASR-side signal. See [ASR customization](https://docs.nvidia.com/nim/speech/latest/asr/customization/customization.html) for exact semantics, defaults, and the `force_eou` runtime flag.

- **Language**: the multilingual example locks each session to a single locale, selectable per connection in the UI (any locale the ASR and TTS both support, default `de-DE`) and fixed for that session. Different sessions can use different languages. The ASR also accepts `language_code: auto` for per-turn detection, but this blueprint does not use it, since a pinned locale is more reliable. For the full set a model supports, see its per-model supported-language table in the [ASR support matrix](https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/asr.html) (Nemotron ASR Streaming covers 40 locales, Parakeet RNNT Multilingual 25+).
- **Catalog config**: a cloud entry sets `server` / `model` / `function_id`, while a local entry points at the Compose sidecar `host:port`. Host-run deployments rewrite sidecar endpoints to `localhost` automatically. See [Configure Services → On-prem catalog](configure-services.md#on-prem-catalog).

```yaml
asr:
  my-custom-asr:
    name: "My Custom ASR"
    server: "grpc.nvcf.nvidia.com:443"   # cloud. Local entries use the sidecar host:port
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
