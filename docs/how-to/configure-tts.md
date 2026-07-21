# Configure TTS

The pipeline synthesizes the spoken reply with a streaming **TTS** service. The default is NVIDIA **Magpie TTS Multilingual**, served from the cloud (NVIDIA-hosted NVCF endpoints) or self-hosted next to the pipeline as an [**NVIDIA NIM for Speech**](https://docs.nvidia.com/nim/speech/latest/tts/index.html) sidecar.

TTS services are declared per example in `services.cloud.yaml` (remote / NVCF) and `services.local.yaml` (Compose-managed sidecars). This page is the **model reference and configuration guide**: available models, how to size them, and how to set voices, pronunciation, and text filtering. For catalog mechanics (switching, adding, and overriding services), see [Configure Services](configure-services.md).

## Models

| Model | Self-hosted compose service | Modelcard |
|-------|-----------------------------|-----------|
| **Magpie TTS Multilingual**: default, streaming multilingual TTS with per-language voices | [`docker-compose.magpie-tts.yaml`](../../docker/docker-compose.magpie-tts.yaml) | [model card](https://build.nvidia.com/nvidia/magpie-tts-multilingual/modelcard) |
| **Chatterbox TTS Multilingual**: optional streaming multilingual TTS (workstation local + cloud) | [`docker-compose.chatterbox-tts.yaml`](../../docker/docker-compose.chatterbox-tts.yaml) | [model card](https://build.nvidia.com/resembleai/chatterbox-multilingual-tts/modelcard) |

Magpie TTS Multilingual is exposed as the catalog key `magpie-multilingual-tts` in `services.cloud.yaml` / `services.local.yaml`. Chatterbox is `chatterbox-multilingual-tts` (cloud everywhere; local under `workstation` only). Voice IDs follow each model's naming (e.g. `Magpie-Multilingual.EN-US.Aria`, `Chatterbox-Multilingual.en-US.Male`). The available voices and emotions depend on the deployed NIM. See [available voices and emotions](https://docs.nvidia.com/nim/speech/latest/tts/voices.html).

> The active default per slot is set in [`examples_registry.yaml`](../../examples_registry.yaml) (`defaults`).

> **Streaming only.** The real-time pipeline needs a **streaming** TTS model. The streaming-capable TTS NIMs are **Magpie TTS Multilingual**, **Magpie TTS Zeroshot**, and **Chatterbox TTS Multilingual**. All three can be enabled with the latest Pipecat (**> 1.4.0**). Check the [Pipecat NVIDIA TTS service](https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/services/nvidia/tts.py) for details. This blueprint pins `pipecat-ai==1.3.0` and ships Magpie TTS Multilingual.

## Hardware requirements and deployment configs

TTS runs one of three ways, and the repo wires the right one per profile:

- **Cloud (NVCF)**: no local GPU, and the catalog calls `grpc.nvcf.nvidia.com`. The simplest starting point.
- **Magpie TTS NIM sidecar**: on the `*/workstation` and `*/dgx-spark` profiles (`tts-service` on GPU `0` by default, [`docker-compose.magpie-tts.yaml`](../../docker/docker-compose.magpie-tts.yaml)).
- **Riva embedded (Jetson Thor)**: on `*/jetson-thor`, on-device Riva serves TTS: `nemotron-speech` (ASR + TTS together) or `nemotron-speech-tts` (TTS only). See [Jetson Thor](../03-jetson-thor.md).

### VRAM & hardware support

The TTS sidecar uses roughly **~14 GB VRAM** and, on local profiles, runs alongside the LLM and ASR. On a single ~80 GB GPU, TTS (~14 GB) + ASR (~15 GB) + the LLM (~30 GB FP8) fit together. To split them across GPUs, set `device_ids` in [`docker-compose.magpie-tts.yaml`](../../docker/docker-compose.magpie-tts.yaml). See [Configure LLM → VRAM & hardware support](configure-llm.md#vram--hardware-support) for the full layout.

### Performance & scaling

`batch_size` (set on the Magpie service via `NIM_TAGS_SELECTOR=name=magpie-tts-multilingual,batch_size=8`) is the main throughput knob. Tune it per deployment shape, and benchmark before raising it on shared single-GPU profiles. For first-chunk / inter-chunk latency and throughput (RTFX) across GPUs, see the **[TTS performance benchmarks](https://docs.nvidia.com/nim/speech/latest/reference/performances/tts/performance.html)**. For end-to-end pipeline latency (TTS time-to-first-byte) in this blueprint, see [Evaluation and Performance](../04-evaluation-and-performance.md).

## Customization

### Voices & emotions

The active voice is the `voice_id` in the catalog entry. The client UI also has a voice selector that auto-discovers the connected service's available voices and languages, so you can switch mid-session. Voice IDs follow `Model.Language.VoiceName` (e.g. `Magpie-Multilingual.EN-US.Aria`), and Magpie also supports emotional styles. Available voices/emotions depend on your Magpie version (and can be discovered at runtime over gRPC/HTTP). See [available voices and emotions](https://docs.nvidia.com/nim/speech/latest/tts/voices.html).

To change the **default**, edit `voice_id` in the example's `services.cloud.yaml` / `services.local.yaml`. For a local Magpie NIM, point the entry at the sidecar (`tts-service:50051`) under the active platform block. See [Configure Services](configure-services.md).

```yaml
tts:
  magpie-multilingual-tts:
    name: "Magpie TTS Multilingual"
    server: "grpc.nvcf.nvidia.com:443"   # cloud; local entries use the sidecar host:port (e.g. tts-service:50051)
    voice_id: "Magpie-Multilingual.EN-US.Aria"
    model: "magpie-tts-multilingual"
    function_id: "877104f7-e885-42b9-8de8-f6e4c6303969"
    synthesis_mode: stitched             # Magpie; omit on other models to keep per_sentence

  chatterbox-multilingual-tts:
    name: "Chatterbox TTS Multilingual"
    server: "grpc.nvcf.nvidia.com:443"
    voice_id: "Chatterbox-Multilingual.en-US.Male"
    model: "chatterbox-tts-multilingual"
    function_id: "ddacc747-1269-4fab-bfd9-8f593dead106"
    # no synthesis_mode → Pipecat per_sentence
```

Catalog `model` + `function_id` are hydrated into the session and passed to Pipecat's `NvidiaTTSService` as `model_function_map` (same pattern as ASR). Magpie remains the registry default; pick Chatterbox in the Services tab to switch.

### Synthesis mode

Pipecat's `NvidiaTTSService` supports two synthesis modes via the catalog field `synthesis_mode`:

| Value | Behavior |
|-------|----------|
| `stitched` | Reuse one Magpie `SynthesizeOnline` stream across sentences in a reply (smoother multi-sentence audio). Use this for Magpie multilingual / zero-shot ≥ v1.7.0. |
| `per_sentence` | Open a fresh synthesis call per sentence. Safe for models without cross-sentence stitching. |

Set `synthesis_mode` on the catalog entry (hydrated as `tts_synthesis_mode`). Magpie ships with `stitched`. Omit the field on other models to leave Pipecat's default (`per_sentence`).

### Pronunciation (IPA)

Override Magpie's default pronunciation for specific words with an International Phonetic Alphabet (IPA) dictionary. Create a JSON or YAML dictionary file, then set `TTS_IPA_FILE_PATH` in `.env` to that path. Relative paths resolve from the repo root:

```bash
TTS_IPA_FILE_PATH=config/ipa.json
```

Example dictionary:

```json
{
  "NVIDIA": "ˈɛnˌvɪdiə",
  "GreenForce": "ɡriːn fɔrs",
  "API": "eɪ piː aɪ"
}
```

The dictionary loads at session start and applies to every TTS request. Restart the server (or re-apply the active Compose profile) after changing the file. For the dictionary format and the phonemes Magpie supports, see [TTS customization](https://docs.nvidia.com/nim/speech/latest/tts/customization.html) and [phoneme support](https://docs.nvidia.com/nim/speech/latest/tts/phoneme-support.html).

> **Check the wiring.** `TTS_IPA_FILE_PATH` only takes effect if the pipeline loads the dictionary and passes it to the `NvidiaTTSService`. The shipped examples do this with `custom_dictionary=load_ipa_dictionary()` where they construct the service (see the `NvidiaTTSService(...)` call in [`src/examples/generic/pipeline.py`](../../src/examples/generic/pipeline.py)). If you build a custom pipeline, confirm your `NvidiaTTSService(...)` is created with `custom_dictionary=load_ipa_dictionary()`, or the env var has no effect.

### TTS text filter

LLM output frequently contains Markdown emphasis and characters the Magpie preprocessor reserves for its own markup. Unfiltered, these are spoken literally, make synthesis fail, or produce odd audio. A text filter sits between the LLM and TTS and strips them before synthesis. The default filter removes:

- **`*`**: Markdown emphasis markers (for example `**bold**` and `*italic*`).
- **`{` and `}`**: ARPAbet phoneme tokens such as `{@AW1}`.
- **`<tag>`**: SSML tags parsed by the TTS engine.

These appear naturally in code, JSON, Markdown, or HTML output. The filter classes live in [`src/examples/shared/nemotron_speech_text_filter.py`](../../src/examples/shared/nemotron_speech_text_filter.py):

#### `NemotronSpeechTextFilter` (default)

A single regex pass that strips `*`, `{`, `}`, and tag-opening `<`. Everything else passes through unchanged: comparison operators (`5 < 7`), currency, emoji, and non-Latin scripts. Use it for plain or lightly formatted prose.

```python
# src/examples/generic/pipeline.py
from examples.shared.nemotron_speech_text_filter import NemotronSpeechTextFilter

tts = NvidiaTTSService(
    ...
    text_filters=[NemotronSpeechTextFilter()],  # default
)
```

#### `NemotronSpeechMarkdownTextFilter`

Extends Pipecat's `MarkdownTextFilter` with the same reserved-character strip. Use it when the LLM streams Markdown. All `MarkdownTextFilter` settings (`filter_code`, `filter_tables`) are inherited.

```python
# src/examples/generic/pipeline.py
from examples.shared.nemotron_speech_text_filter import NemotronSpeechMarkdownTextFilter

tts = NvidiaTTSService(
    ...
    text_filters=[NemotronSpeechMarkdownTextFilter()],
)
```

### Voice cloning / zero-shot

Magpie TTS Zeroshot clones a voice from a short reference clip. See [voice cloning](https://docs.nvidia.com/nim/speech/latest/tts/voice-cloning.html). Pipecat's `NvidiaTTSService` does not expose zero-shot voice cloning in releases **≤ 1.4.0** (this repo pins `pipecat-ai==1.3.0`). To use it, upgrade to the latest Pipecat release or run from its `main` branch.

## Reference

- [Troubleshooting guide](../06-troubleshooting.md#tts-text-to-speech): reserved-character synthesis failures, mispronunciations, and long-input limits.
- [Configure Services](configure-services.md): how the catalog is loaded, switched, and overridden.
- [NVIDIA NIM for Speech — TTS](https://docs.nvidia.com/nim/speech/latest/tts/index.html): [available voices & emotions](https://docs.nvidia.com/nim/speech/latest/tts/voices.html), [customization / pronunciation](https://docs.nvidia.com/nim/speech/latest/tts/customization.html), [phoneme support](https://docs.nvidia.com/nim/speech/latest/tts/phoneme-support.html), [voice cloning (zero-shot)](https://docs.nvidia.com/nim/speech/latest/tts/voice-cloning.html), [performance benchmarks](https://docs.nvidia.com/nim/speech/latest/reference/performances/tts/performance.html), [TTS troubleshooting](https://docs.nvidia.com/nim/speech/latest/troubleshooting/tts.html).
- [Pipecat NVIDIA TTS service](https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/services/nvidia/tts.py).
