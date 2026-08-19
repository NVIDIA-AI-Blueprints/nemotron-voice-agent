# Configure TTS

The pipeline synthesizes the spoken reply with a streaming **TTS** service. The default is NVIDIA **Magpie TTS Multilingual**, served from the cloud (NVIDIA-hosted NVCF endpoints) or self-hosted next to the pipeline as an [**NVIDIA NIM for Speech**](https://docs.nvidia.com/nim/speech/latest/tts/index.html) sidecar.

TTS services are declared per example in `services.cloud.yaml` (remote / NVCF) and `services.local.yaml` (Compose-managed sidecars). This page is the **model reference and configuration guide**: available models, how to size them, and how to set voices, pronunciation, and text filtering. For catalog mechanics (switching, adding, and overriding services), see [Configure Services](configure-services.md).

## Models

| Model | Catalog key | Self-hosted compose service | Modelcard |
|-------|-------------|-----------------------------|-----------|
| **Magpie TTS Multilingual**: default, streaming multilingual TTS with per-language voices | `magpie-multilingual-tts` | [`docker-compose.magpie-tts.yaml`](../../docker/docker-compose.magpie-tts.yaml) | [NGC container](https://catalog.ngc.nvidia.com/orgs/nim/nvidia/containers/magpie-tts-multilingual/1.10.0) · [model card](https://build.nvidia.com/nvidia/magpie-tts-multilingual/modelcard) |
| **Magpie TTS Zeroshot**: multilingual streaming TTS that supports zero-shot voice cloning and includes built-in female and male voices | `magpie-zeroshot-tts` | [`docker-compose.magpie-zeroshot-tts.yaml`](../../docker/docker-compose.magpie-zeroshot-tts.yaml) | [model card](https://build.nvidia.com/nvidia/magpie-tts-zeroshot/modelcard) |
| **Chatterbox TTS Multilingual**: alternate streaming multilingual TTS | `chatterbox-multilingual-tts` | [`docker-compose.chatterbox-tts.yaml`](../../docker/docker-compose.chatterbox-tts.yaml) | [model card](https://build.nvidia.com/resembleai/chatterbox-multilingual-tts/modelcard) |

> Magpie Multilingual is the registry default and the TTS sidecar started by local recipes. Chatterbox and Magpie Zeroshot are opt-in: select their catalog key in the Services tab (or `defaults.tts` in [`examples_registry.yaml`](../../examples_registry.yaml)). For local NIM, also enable the matching Compose profile (see [Hardware requirements](#hardware-requirements-and-deployment-configs)).

Voice IDs follow each model's naming. For example, use `Magpie-Multilingual.EN-US.Aria`, `Magpie-ZeroShot-Multilingual.Female`, or `Chatterbox-Multilingual.en-US.Male`. The available voices and emotions depend on the deployed NIM. Refer to [available voices and emotions](https://docs.nvidia.com/nim/speech/latest/tts/voices.html).

### Supported languages

The client discovers the active TTS service's available voices and language codes at runtime. Treat this table as model-level guidance, because exact availability can vary by endpoint, deployment profile, and selected NIM image.

For the multilingual assistant, this is **TTS-only** coverage, not the final session-language list. Voice Settings shows only the intersection of the selected ASR, TTS, and built-in LLM capabilities. For example, a Chatterbox deployment can advertise Arabic or Greek voices, but those locales are not available with the built-in Nemotron 3 Nano or Nemotron 3 Super LLMs. See [Configure LLM](configure-llm.md#multilingual-session-languages).

| Model | Supported languages |
| --- | --- |
| [Magpie TTS Multilingual](https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/tts.html#magpie-tts-multilingual) | English (`en-US`) · Spanish (`es-US`) · French (`fr-FR`) · German (`de-DE`) · Italian (`it-IT`) · Vietnamese (`vi-VN`) · Mandarin (`zh-CN`) · Hindi (`hi-IN`) · Japanese (`ja-JP`) · Modern Standard Arabic (`ar-AR`) · Korean (`ko-KR`) · Brazilian Portuguese (`pt-BR`) |
| [Magpie TTS Zeroshot](https://build.nvidia.com/nvidia/magpie-tts-zeroshot/modelcard) | English (`en-US`) · Spanish (`es-US`) · French (`fr-FR`) · German (`de-DE`) · Mandarin (`zh-CN`) · Vietnamese (`vi-VN`) · Italian (`it-IT`) · Hindi (`hi-IN`) · Japanese (`ja-JP`) · Modern Standard Arabic (`ar-AR`) · Brazilian Portuguese (`pt-BR`) · Korean (`ko-KR`) |
| [Chatterbox TTS Multilingual](https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/tts.html#chatterbox-tts-multilingual) | Arabic (`ar-SA`) · Danish (`da-DK`) · German (`de-DE`) · Greek (`el-GR`) · English (`en-US`) · Spanish (`es-ES`) · Finnish (`fi-FI`) · French (`fr-FR`) · Hebrew (`he-IL`) · Hindi (`hi-IN`) · Italian (`it-IT`) · Japanese (`ja-JP`) · Korean (`ko-KR`) · Malay (`ms-MY`) · Dutch (`nl-NL`) · Norwegian (`nb-NO`) · Polish (`pl-PL`) · Brazilian Portuguese (`pt-BR`) · Russian (`ru-RU`) · Swedish (`sv-SE`) · Swahili (`sw-KE`) · Turkish (`tr-TR`) · Mandarin (`zh-CN`) |

For NVIDIA's current model and deployment support details, see the [TTS support matrix](https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/tts.html).

> The active default per slot is set in [`examples_registry.yaml`](../../examples_registry.yaml) (`defaults`).
>
> **Streaming only.** The real-time pipeline needs a **streaming** TTS model. The streaming-capable TTS NIMs are **Magpie TTS Multilingual**, **Magpie TTS Zeroshot**, and **Chatterbox TTS Multilingual**. Check the [Pipecat NVIDIA TTS service](https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/services/nvidia/tts.py) for supported request fields and model-specific options.
>
> **Word-level streaming limitation.** `NvidiaWordTTSService` word-level input streaming and word timestamps are not available with Magpie TTS Multilingual 1.10.0 or newer. This limitation applies even with `nvidia-riva-client` 2.27.0. Do not use this integration for word-accurate spoken-context commits or interruption boundaries.

## Hardware requirements and deployment configs

TTS runs one of these ways, and the repo wires the right one per profile:

- **Cloud (NVCF)**: no local GPU. Magpie Multilingual and Chatterbox appear in the Services tab (no Compose change). Magpie Zeroshot has no cloud function.
- **Magpie TTS Multilingual 1.10.0 (default local)**: started by `*/workstation` and `*/dgx-spark` recipes as `tts-service` from the public [NGC container](https://catalog.ngc.nvidia.com/orgs/nim/nvidia/containers/magpie-tts-multilingual/1.10.0) ([`docker-compose.magpie-tts.yaml`](../../docker/docker-compose.magpie-tts.yaml)).
- **Opt-in local TTS (Chatterbox or Magpie Zeroshot)**: both are listed in Compose but do **not** start with the default recipe. They share Magpie Multilingual's host ports (`50151` / `9000`), so only one of Magpie Multilingual, Chatterbox, or Zeroshot can run at a time. Enable the opt-in profile and scale Magpie off:

  | Alternate | Compose profile | Catalog key | Compose file |
  |-----------|-----------------|-------------|--------------|
  | Chatterbox | `chatterbox-tts` | `chatterbox-multilingual-tts` | [`docker-compose.chatterbox-tts.yaml`](../../docker/docker-compose.chatterbox-tts.yaml) |
  | Magpie Zeroshot | `magpie-zeroshot-tts` | `magpie-zeroshot-tts` | [`docker-compose.magpie-zeroshot-tts.yaml`](../../docker/docker-compose.magpie-zeroshot-tts.yaml) |

  ```bash
  # Example: Magpie Zeroshot on workstation (same pattern for Chatterbox / dgx-spark)
  docker compose --profile generic-assistant/workstation --profile magpie-zeroshot-tts \
    up -d --scale tts-service=0
  ```

  Then select the matching catalog key in the Services tab (or `defaults.tts`). Omitting the opt-in profile leaves that sidecar running and holding the ports—stop it before Magpie Multilingual can bind again (`docker compose --profile <profile> stop <service>`, then recipe `up -d`).

  Magpie Zeroshot NGC access is restricted — apply at the [Magpie TTS Zeroshot NGC page](https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/magpie-tts-zeroshot). For audio-prompt cloning, see [Voice cloning / zero-shot](#voice-cloning--zero-shot).
- **Riva embedded (Jetson Thor)**: on `*/jetson-thor`, on-device Riva serves TTS: `nemotron-speech` (ASR + TTS together) or `nemotron-speech-tts` (TTS only). See [Jetson Thor](../03-jetson-thor.md).

### VRAM & hardware support

| Model | Typical VRAM | Notes |
|-------|--------------|-------|
| Magpie TTS Multilingual | **~14 GB** | Can share a single ~80 GB GPU with ASR (~15 GB) and the LLM (~30 GB FP8). Split across GPUs with `device_ids` in [`docker-compose.magpie-tts.yaml`](../../docker/docker-compose.magpie-tts.yaml). See [Configure LLM → VRAM & hardware support](configure-llm.md#vram--hardware-support). |
| Magpie TTS Zeroshot | **~13.06 GB** GPU / ~4.00 GB CPU at `batch_size=8` | Default `NIM_TAGS_SELECTOR=name=magpie-tts-zeroshot,batch_size=8` ([Speech NIM support matrix](https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/tts.html#magpie-tts-zeroshot)). Fits the shared H100 layout when Magpie Multilingual is scaled off. `batch_size=32` needs **~41.30 GB** GPU / ~7.08 GB CPU and should not share a single 80 GB GPU with ASR + LLM. |
| Chatterbox TTS | **~52.5 GB** GPU / ~6.4 GB CPU | `NIM_TAGS_SELECTOR=name=chatterbox-tts-multilingual` (GPU `0` by default). Does **not** fit the Magpie single-80-GB shared layout with LLM + ASR on a typical workstation GPU. |

### Performance & scaling

`batch_size` is the main Magpie throughput knob (`NIM_TAGS_SELECTOR`):

- Magpie Multilingual: `name=magpie-tts-multilingual,batch_size=8`
- Magpie Zeroshot: `batch_size=8` (default) or `batch_size=32` — keep `8` on shared single-GPU recipes
- Chatterbox: single profile `name=chatterbox-tts-multilingual` (no `batch_size` selector)

For first-chunk and inter-chunk latency and throughput (RTFX) across GPUs, refer to the **[TTS performance benchmarks](https://docs.nvidia.com/nim/speech/latest/reference/performances/tts/performance.html)**. For end-to-end pipeline latency (TTS time-to-first-byte) in this blueprint, refer to [Evaluation and Performance](../04-evaluation-and-performance.md).

## Customization

### Voices & emotions

The active voice is the `voice_id` in the catalog entry. The client UI includes a voice selector that discovers the connected service's available voices and languages, so you can switch mid-session. Voice IDs follow each model's naming. For example, use `Magpie-Multilingual.EN-US.Aria`, `Magpie-ZeroShot-Multilingual.Female`, or `Chatterbox-Multilingual.en-US.Male`. Available voices and emotions depend on the deployed NIM and can be discovered at runtime over gRPC or HTTP. Refer to [available voices and emotions](https://docs.nvidia.com/nim/speech/latest/tts/voices.html).

- **Magpie Multilingual**: multiple voices and emotional styles per locale.
- **Magpie Zeroshot**: languages listed in [Supported languages](#supported-languages); built-in voices across locales are `Magpie-ZeroShot-Multilingual.Female` (default) and `Magpie-ZeroShot-Multilingual.Male` ([model card](https://build.nvidia.com/nvidia/magpie-tts-zeroshot/modelcard)).
- **Chatterbox**: **one default speaker per locale**.

To change the **default**, edit `voice_id` in the example's `services.cloud.yaml` / `services.local.yaml`. For a local Magpie NIM, point the entry at the sidecar (`tts-service:50051` or `magpie-zeroshot-tts-service:50051`) under the active platform block. See [Configure Services](configure-services.md).

```yaml
tts:
  magpie-multilingual-tts:
    name: "Magpie TTS Multilingual"
    server: "grpc.nvcf.nvidia.com:443"   # cloud. Local entries use the sidecar host:port (e.g. tts-service:50051)
    voice_id: "Magpie-Multilingual.EN-US.Aria"
    model: "magpie-tts-multilingual"
    function_id: "877104f7-e885-42b9-8de8-f6e4c6303969"
    synthesis_mode: stitched

  chatterbox-multilingual-tts:
    name: "Chatterbox TTS Multilingual"
    server: "grpc.nvcf.nvidia.com:443"
    voice_id: "Chatterbox-Multilingual.en-US.Male"
    model: "chatterbox-tts-multilingual"
    function_id: "ddacc747-1269-4fab-bfd9-8f593dead106"
    synthesis_mode: per_sentence

  # Local only (services.local.yaml workstation / dgxspark). No cloud function_id.
  magpie-zeroshot-tts:
    name: "Magpie TTS Zeroshot"
    server: "magpie-zeroshot-tts-service:50051"
    voice_id: "Magpie-ZeroShot-Multilingual.Female"
    model: "magpie-tts-zeroshot"
    function_id: ""
    synthesis_mode: stitched
    language_code: en-US
    # optional voice cloning:
    # zero_shot_audio_prompt_file: "/path/to/prompt.wav"
```

The catalog hydrates the required `model` and `function_id` fields and the optional `zero_shot_audio_prompt_file` field into the session, then passes them to Pipecat's `NvidiaTTSService`.

### Synthesis mode

Pipecat's `NvidiaTTSService` supports two synthesis modes via the catalog field `synthesis_mode`:

| Value | Behavior |
|-------|----------|
| `stitched` | Reuse one Magpie `SynthesizeOnline` stream across sentences in a reply (smoother multi-sentence audio). Requires Pipecat `>=1.5.0`, plus Magpie TTS Multilingual `>=1.7.0` or Magpie TTS Zeroshot `>=1.2.0`. |
| `per_sentence` | Open a fresh synthesis call per sentence. Safe for models without cross-sentence stitching. |

Set `synthesis_mode` on the catalog entry (hydrated as `tts_synthesis_mode`). Magpie multilingual and Magpie zeroshot ship with `stitched`; Chatterbox ships with `per_sentence`. Always set the field explicitly so a UI/backend TTS switch cannot inherit another model's mode via the registry-default fallback in the pipeline.

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

Magpie TTS Zeroshot clones a voice from a short reference clip via Pipecat's `NvidiaTTSService(zero_shot_audio_prompt_file=...)`. Set the path only in catalog YAML (`services.local.yaml`); it is not accepted from the client session body. See also [voice cloning](https://docs.nvidia.com/nim/speech/latest/tts/voice-cloning.html).

1. Enable the Zeroshot sidecar and select `magpie-zeroshot-tts` (see [Hardware requirements](#hardware-requirements-and-deployment-configs)).
2. Prepare a 16-bit mono WAV (sample rate ≥ 22.05 kHz, about 3–10 seconds).
3. In the example's `services.local.yaml` (`workstation` or `dgxspark`), keep or set `voice_id` to a built-in such as `Magpie-ZeroShot-Multilingual.Female`, and add an **absolute path visible to the voice-agent process**:

   ```yaml
   magpie-zeroshot-tts:
     ...
     zero_shot_audio_prompt_file: "/data/prompts/clone.wav"
   ```

   - **Host-native** (`uv run` / local Python): use a host absolute path (for example `/home/you/prompts/clone.wav`).
   - **Compose / Docker**: mount the file into the app service for your Compose profile (for example `generic-assistant` with `--profile generic-assistant`, or `generic-assistant-workstation` with `--profile generic-assistant/workstation`). Use a Compose override, then set `zero_shot_audio_prompt_file` to that **container** absolute path. Relative paths are not resolved from the repo root.

     ```yaml
     # docker-compose.override.yaml (example for --profile generic-assistant)
     services:
       generic-assistant:
         volumes:
           - /home/you/prompts/clone.wav:/data/prompts/clone.wav:ro
     ```

   The catalog field is hydrated as `tts_zero_shot_audio_prompt_file` and passed into `NvidiaTTSService`.
4. Start a session.

Omit `zero_shot_audio_prompt_file` to use only built-in Zeroshot voices.

## Reference

- [Troubleshooting guide](../06-troubleshooting.md#tts-text-to-speech): reserved-character synthesis failures, mispronunciations, and long-input limits.
- [Configure Services](configure-services.md): how the catalog is loaded, switched, and overridden.
- [NVIDIA NIM for Speech — TTS](https://docs.nvidia.com/nim/speech/latest/tts/index.html): [available voices & emotions](https://docs.nvidia.com/nim/speech/latest/tts/voices.html), [customization / pronunciation](https://docs.nvidia.com/nim/speech/latest/tts/customization.html), [phoneme support](https://docs.nvidia.com/nim/speech/latest/tts/phoneme-support.html), [voice cloning (zero-shot)](https://docs.nvidia.com/nim/speech/latest/tts/voice-cloning.html), [performance benchmarks](https://docs.nvidia.com/nim/speech/latest/reference/performances/tts/performance.html), [TTS troubleshooting](https://docs.nvidia.com/nim/speech/latest/troubleshooting/tts.html).
- [Pipecat NVIDIA TTS service](https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/services/nvidia/tts.py).
