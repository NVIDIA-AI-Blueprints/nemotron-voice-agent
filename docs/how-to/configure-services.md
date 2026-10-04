# Configure Services

The Nemotron Voice Agent keeps every built-in LLM, ASR, and TTS service in one catalog, [`services.yaml`](../../services.yaml), at the repository root. Each example picks the services it offers in [`examples_registry.yaml`](../../examples_registry.yaml). Editable knobs, such as reasoning on/off and the reasoning budget, live in [`settings.yaml`](../../settings.yaml).

This guide covers the **mechanics**: how the catalog is laid out, how the active recipe selects entries, and how to extend it.

## Required credentials

Set one key per recipe family. Do not mix them.

| Recipe family | Required `.env` key |
| --- | --- |
| Cloud (`<example>`) | `NVIDIA_API_KEY` |
| Server (`<example>/server`) | `NVIDIA_API_KEY` |
| Performance Server (`generic-assistant/server-perf`) | `NVIDIA_API_KEY` |
| Single-GPU (`<example>/single-gpu`) | `HF_TOKEN` |

`NVIDIA_API_KEY` is required for cloud-only, Server, and Performance Server. Those recipes also expose NVIDIA Cloud catalog entries in the Services tab. Server and Performance Server additionally need `docker login nvcr.io` before `up`. Single-GPU requires `HF_TOKEN` for Hugging Face model downloads and does not use `NVIDIA_API_KEY`.

## Catalog layout

`services.yaml` has one section per self-hosted stack:

- `server`: NIM sidecars used by `*/server` and `generic-assistant/server-perf`.
- `single-gpu`: vLLM and NeMo-Speech.cpp used by `*/single-gpu`.

NVIDIA Cloud serves the same NIMs as `server`, so there is no separate cloud section. A `server` entry that is also hosted on NVIDIA Cloud carries an `nvcf` key with only the fields that differ on the cloud. `server` becomes `grpc.nvcf.nvidia.com:443` for speech and `base_url` becomes `https://integrate.api.nvidia.com/v1` for LLMs. Entries without `nvcf` are self-hosted only.

```yaml
server:
  asr:
    nemotron-asr-streaming-english:
      name: "Nemotron ASR Streaming English"
      server: "nemotron-asr-streaming-english:50052"
      model: "cache-aware-parakeet-rnnt-en-US-asr-streaming-sortformer"
      nvcf:
        function_id: "bb0837de-8c7b-481f-9ec8-ef5663e9c1fa"
        model: "nemotron-asr-streaming"
  llm:
    nemotron-lightning:
      name: "Nemotron 3.5 Lightning 30B A3B"
      model_id: "nvidia/nemotron-3.5-lightning-30b-a3b"
      base_url: "http://nvidia-llm:8000/v1"
      extra_params: '{"extra_body":{"repetition_penalty":1.05}}'
      settings: nemotron-nim
      nvcf: {}
```

## How the active recipe selects entries

Each Compose app service sets `SERVICE_RECIPE` to `cloud`, `server`, or `single-gpu`, so containers never probe endpoints. Host-native runs (`uv run python src/server.py`) leave it unset, which means `auto`: the backend probes the example's `server` and `single-gpu` endpoints in parallel (cached for 5 seconds), picks the section with more reachable entries, and lists only the reachable entries of each slot. When nothing is reachable, it uses NVIDIA Cloud if `NVIDIA_API_KEY` is set, and otherwise lists the `server` section. Start the sidecars, then start the app. Set `SERVICE_RECIPE` to pin a section.

| `SERVICE_RECIPE` | Self-hosted entries | NVIDIA Cloud entries |
| --- | --- | --- |
| `auto` (unset) | reachable entries of the detected section, or the whole `server` section when nothing is reachable and `NVIDIA_API_KEY` is not set | `server` entries with `nvcf`, when `NVIDIA_API_KEY` is set |
| `cloud` | none | `server` entries with `nvcf` |
| `server` | `server` section | `server` entries with `nvcf`, when `NVIDIA_API_KEY` is set |
| `single-gpu` | `single-gpu` section | `server` entries with `nvcf`, when `NVIDIA_API_KEY` is set |

`NVIDIA_API_KEY` counts as set only when it is a real key. Empty, unset, and the Compose placeholder `not-needed` hide NVIDIA Cloud entries. With an explicit recipe, the catalog lists every entry of the section. The backend checks the selected services when a session starts and reports a clear error when one is not running.

## Per-example services and defaults

Each example lists the catalog keys it offers per slot under `services` in `examples_registry.yaml`. The slot order is the UI order, and the list order is the order inside each slot. A slot reads the catalog category of the same name. To add a slot that reuses another category, map it under `categories`. The first key that is available for the active recipe becomes the default, with self-hosted entries preferred over NVIDIA Cloud.

```yaml
multilingual-assistant:
  services:
    llm: [nemotron-lightning]
    asr: [nemotron-asr-streaming-multilingual, parakeet-rnnt]
    tts: [magpie-multilingual-tts, chatterbox-multilingual-tts, magpie-zeroshot-tts]
```

On `server` and `single-gpu`, the multilingual ASR default is the self-hosted `nemotron-asr-streaming-multilingual`. On `cloud`, that key has no `nvcf`, so the default becomes `parakeet-rnnt`. To change a default, move the key to the front of its list.

The [Examples table](../../README.md#-examples) links to each example README. Each README lists the models selected for every supported profile.

## Switching services in the UI

The Services tab lists the active recipe's entries for the example. Click an entry to make it the active selection for that category. Selections and settings last for the browser tab: they survive disconnect and reconnect, and a page refresh resets them to the example's server defaults. Custom services added through the UI live in browser localStorage, so they survive a refresh.

## Editable settings

An entry with `settings: <profile>` shows the controls of that `settings.yaml` profile inside the service's card in the Services tab. Click the card to expand them. Values are kept per example, per slot, and per service, so they survive switching between Self-hosted and NVIDIA Cloud. Numbers with `min` and `max` render as a slider with a value box, other numbers as a value box, enums as a dropdown, and booleans as a toggle. The backend validates each value against its `type`, `min`/`max`, or `options`, falls back to the default for invalid or missing values, and writes it at `path`: inside `extra_params` for an LLM entry, or into the ASR or TTS entry field of that name, such as `synthesis_mode`. `path` defaults to the setting name, so only nested fields set it. A setting with `requires` is sent only while that setting is true.

```yaml
x-sampling: &sampling
  temperature: {label: "Temperature", type: float, min: 0.0, max: 1.0}
  top_p: {label: "Top P", type: float, min: 0.01, max: 1.0}
  max_tokens: {label: "Max Tokens", type: int, min: 1, max: 16384}
  seed: {label: "Seed", type: int, min: 0}

nemotron-nim:
  <<: *sampling
  enable_thinking:
    label: "Reasoning"
    type: bool
    default: false
    path: extra_body.chat_template_kwargs.enable_thinking
  reasoning_budget:
    label: "Reasoning Budget"
    type: int
    default: 16384
    min: 128
    max: 32768
    path: extra_body.reasoning_budget
    requires: enable_thinking
```

A setting's default comes from, in order: the example's `settings` for that slot and service in `examples_registry.yaml`, a value already at `path` in the entry's `extra_params`, then the profile's `default`. A setting without any default, such as `temperature`, shows `Auto` and is sent only when the user sets it, so the model or pipeline default applies otherwise. Defaults apply even when the client sends no settings, so direct API clients get the same request as the UI.

NIM and vLLM name the budget differently (`reasoning_budget` and `thinking_token_budget`), so each serving engine has its own profile. Keys that start with `x-` are shared YAML anchors, not profiles. Supported types are `bool`, `int`, `float`, and `enum`.

The Frontend/Backend thinker reads the same `llm` entries as the talker through `categories`, and the registry sets its defaults:

```yaml
frontend-backend-agent:
  services:
    llm: [nemotron-lightning]
    thinker-llm: [nemotron-lightning]
  categories:
    thinker-llm: llm
  settings:
    thinker-llm:
      nemotron-lightning:
        enable_thinking: true
        reasoning_budget: 1024
```

## Adding a self-hosted service

To configure a specific local model, check its Docker Compose file under [`docker/`](../../docker/) for the **service name**, **port**, and the **profile** that launches it, then point a catalog entry at that endpoint. For example, Nemotron ASR Streaming (English) is defined in [`docker/docker-compose.nemotron-asr.yaml`](../../docker/docker-compose.nemotron-asr.yaml):

```yaml
services:
  nemotron-asr-streaming-english:
    image: nvcr.io/nim/nvidia/nemotron-asr-streaming:1.4.0
    profiles:
      - generic-assistant/server
      - frontend-backend-agent/server
    ports:
      - "50152:50052"   # host:container (gRPC)
    environment:
      NIM_TAGS_SELECTOR: type=en-US,mode=str
```

The matching entry under `server` points at that Compose service name and **container** port (`50052`). For host-native runs, the backend rewrites it to the published host port automatically. Here `nemotron-asr-streaming-english:50052` becomes `localhost:50152`.

Every speech sidecar publishes its own host ports, so alternatives can run side by side and the Services tab switches between them. A new sidecar needs unused host ports in its Compose file and a matching row in `LOCAL_SPEECH_PORTS` in [`src/utils.py`](../../src/utils.py), which drives both the host rewrite and the readiness check.

| Sidecar | gRPC host port | Health host port |
| --- | --- | --- |
| `nemotron-asr-streaming-english` | `50152` | `9001` |
| `nemotron-asr-streaming-multilingual` | `50252` | `9101` |
| `parakeet-ctc-asr` | `50352` | `9201` |
| `parakeet-rnnt-asr` | `50452` | `9301` |
| `magpie-multilingual-tts-service` | `50151` | `9000` |
| `chatterbox-tts-service` | `50251` | `9100` |
| `magpie-zeroshot-tts-service` | `50351` | `9200` |

When two containers serve the same model for different profiles, give them the same Compose network alias so one catalog entry covers both. Every NeMo-Speech.cpp variant answers on `nemo-speech`, and both single-GPU Lightning containers answer on `nvidia-llm-vllm`.

To run models locally for a **new example**, add your example's profile (e.g. `my-example/server`) to the relevant service(s) under `docker/`, add the app service with `SERVICE_RECIPE` in `docker-compose.yml`, and list the catalog keys under the example's `services` in `examples_registry.yaml`. See [Deployment Profiles](../01-getting-started.md#docker-based-deployment) for the profile list.

## Adding an NVIDIA Cloud service

Add `nvcf` to the matching `server` entry. Refresh the browser for host-run development, or restart the app container. `services.yaml` is bind-mounted.

```yaml
server:
  tts:
    my-custom-tts:
      name: "My Custom TTS"
      server: "my-custom-tts-service:50051"
      voice_id: "Magpie-Multilingual.EN-US.Aria"
      model: "magpie-tts-multilingual"
      synthesis_mode: stitched
      nvcf:
        function_id: "<NVCF_FUNCTION_ID>"
```

```yaml
server:
  llm:
    my-custom-llm:
      name: "My Custom LLM"
      model_id: "org/model-name"
      base_url: "http://my-custom-llm:8000/v1"
      supported_languages: [en, de]
      nvcf: {}
```

`supported_languages` is optional LLM capability metadata for the multilingual assistant. When present, the UI offers only session locales whose base language appears in the list. Omit it for a custom LLM when its language capabilities are unknown; this preserves unrestricted, backward-compatible behavior. An explicitly empty list permits no session locales.

`health_path` is an optional self-hosted LLM field with the serving engine's readiness path, such as `/v1/health/ready` for NIM or `/health` for vLLM. Before a session starts, the backend checks it on the `base_url` host and port, and reports a stopped or loading LLM instead of connecting. Entries without it skip the check.

ASR entries can set `speaker_diarization_supported: true` to show the **Diarization** control in the UI, and `speaker_diarization_max_speakers` to cap the speaker count. For details, refer to [Enable Speaker Labels](enable-speaker-labels.md).

`streaming_url` is an optional self-hosted LLM field that points at a StreamingInput WebSocket. The Services tab shows a **Streaming Input** toggle for it when the example lists `streaming_input` under `capabilities`. For details, refer to [Configure LLM](configure-llm.md).
