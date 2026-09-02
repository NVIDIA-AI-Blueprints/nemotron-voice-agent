# Configure Services

The Nemotron Voice Agent uses example-local service catalogs to manage LLM, ASR, TTS, and example-specific services. Built-in entries come from each example's `services.cloud.yaml` (remote / NVCF) and `services.local.yaml` (Compose-managed sidecars). The active UI example determines which catalog is loaded.

This guide covers the **mechanics**: how catalogs are loaded, switched, and extended.

## How catalog selection works

- Each example owns its catalog at `<example-package>/services.cloud.yaml` (remote / NVCF) and optional `<example-package>/services.local.yaml` (Compose-managed sidecars).
- The cloud catalog is always loaded.
- The local catalog is merged on top, but only entries whose endpoint is reachable on TCP are exposed in the UI and used by the pipeline.
- `services.local.yaml` is grouped into recipe sections (`server` for NIM sidecars, `singlegpu` for vLLM + NeMo-Speech.cpp). The backend merges all sections and exposes only endpoints that are reachable on TCP. Host-native runs work the same way: start the sidecars, then start the app.
- The same `--profile` works whether you run cloud-only or with local sidecars. Nothing else needs to be set.

## Switching services in the UI

The Services tab lists all services exposed by the active catalog (cloud and reachable local entries). Click an entry to make it the active selection for that category. Selections persist in browser localStorage. Custom services added through the UI also live in localStorage.

## Changing built-in defaults

Each example declares its default service per slot via `defaults` in `examples_registry.yaml`. The pipeline resolves that default at startup, and the UI uses it as the initial selection. Edit `defaults` (and optionally reorder entries in the `services.cloud.yaml` / `services.local.yaml` for visual ordering in the UI) to change defaults.

When the same default key exists in both `services.cloud.yaml` and `services.local.yaml`, the resolver prefers the **self-hosted** variant so that deploying local NIM sidecars automatically promotes them to the active default. No UI click is needed. If the self-hosted endpoint is unreachable at session-start time, the runtime falls back to the cloud variant.

> **On-prem note:** self-hosted promotion only applies when the `defaults` key also exists in `services.local.yaml`. A default whose key exists **only** in `services.cloud.yaml` resolves to the cloud model even on an on-prem recipe. Point `defaults` at a local key or pick the model from the Services tab.

### Default Models by Example and Platform

The following models resolve from each example's `defaults` entries in [`examples_registry.yaml`](../../examples_registry.yaml). The profile determines which service catalog implementation supplies the model.

| Example | Cloud | Server | Single GPU |
| --- | --- | --- | --- |
| Generic Assistant | Nemotron ASR Streaming English; Nemotron 3.5 Lightning 30B A3B; Magpie TTS Multilingual | Nemotron ASR Streaming English NIM; Nemotron 3.5 Lightning 30B A3B NIM; Magpie TTS Multilingual NIM | Nemotron Speech Streaming English 0.6B; Nemotron 3.5 Lightning 30B A3B through vLLM; Magpie TTS Multilingual through NeMo-Speech.cpp |
| Multilingual Assistant | Parakeet 1.1B RNNT Multilingual ASR; Nemotron 3.5 Lightning 30B A3B; Magpie TTS Multilingual | Nemotron ASR Streaming Multilingual NIM; Nemotron 3.5 Lightning 30B A3B NIM; Magpie TTS Multilingual NIM | Nemotron 3.5 ASR Streaming Multilingual 0.6B; Nemotron 3.5 Lightning 30B A3B through vLLM; Magpie TTS Multilingual through NeMo-Speech.cpp |
| Nemotron Omni Assistant | Nemotron 3 Nano Omni 30B A3B Reasoning; Magpie TTS Multilingual | Nemotron 3 Nano Omni 30B A3B Reasoning NIM; Magpie TTS Multilingual NIM | Nemotron 3 Nano Omni 30B A3B Reasoning through vLLM; Magpie TTS Multilingual through NeMo-Speech.cpp |
| Nemotron Omni Assistant Subagents | Nemotron 3 Nano Omni 30B A3B Reasoning; Magpie TTS Multilingual | Nemotron 3 Nano Omni 30B A3B Reasoning NIM; Magpie TTS Multilingual NIM | Nemotron 3 Nano Omni 30B A3B Reasoning through vLLM; Magpie TTS Multilingual through NeMo-Speech.cpp |
| Frontend/Backend Agent | Nemotron ASR Streaming English; Nemotron 3.5 Lightning 30B A3B for the Talker and Thinker; Magpie TTS Multilingual | Nemotron ASR Streaming English NIM; Nemotron 3.5 Lightning 30B A3B NIM for the Talker and Thinker; Magpie TTS Multilingual NIM | Nemotron Speech Streaming English 0.6B; Nemotron 3.5 Lightning 30B A3B through vLLM for the Talker and Thinker; Magpie TTS Multilingual through NeMo-Speech.cpp |

The Generic Assistant's `server-perf` profile uses the same default model set as Server. It changes the GPU layout, LLM NIM profile, and Magpie batch size for benchmarking.

The Multilingual Assistant declares `nemotron-asr-streaming-multilingual` as its ASR default. This key resolves to Nemotron ASR for Server and Single GPU. The cloud catalog does not provide that key, so the resolver selects its first cloud ASR entry, Parakeet 1.1B RNNT Multilingual.

The Omni model handles both speech recognition and response generation, so the Omni examples do not have a separate ASR default. The Frontend/Backend Thinker enables reasoning on the same Nemotron 3.5 Lightning model used by the Talker.

## On-prem catalog

`services.local.yaml` groups entries under recipe sections (`server`, `singlegpu`).

To configure a specific local model, check its Docker Compose file under [`docker/`](../../docker/) for the **service name**, **port**, and the **profile** that launches it, then point a catalog entry at that endpoint. For example, Nemotron ASR Streaming (English) is defined in [`docker/docker-compose.nemotron-asr.yaml`](../../docker/docker-compose.nemotron-asr.yaml):

```yaml
services:
  nemotron-asr-streaming-english:
    image: nvcr.io/nim/nvidia/nemotron-asr-streaming:1.3.1
    profiles:
      - generic-assistant/server
      - frontend-backend-agent/server
    ports:
      - "50152:50052"   # host:container (gRPC)
    environment:
      NIM_TAGS_SELECTOR: type=en-US,mode=str
```

The matching `asr` entry in the example's `services.local.yaml` points at that Compose service name and **container** port (`50052`):

```yaml
server:
  asr:
    nemotron-asr-streaming-english:
      name: "Nemotron ASR Streaming English"
      server: "nemotron-asr-streaming-english:50052"
      model: "cache-aware-parakeet-rnnt-multi-asr-streaming-sortformer"
      function_id: ""
```

Use the Compose service name and container port in `server`. For host-native runs (outside Docker), the backend rewrites it to the published host port automatically. Here `nemotron-asr-streaming-english:50052` becomes `localhost:50152`.

Each ASR, LLM, and TTS model sidecar is defined in a `docker/docker-compose.*.yaml` file and gated by Compose `profiles`. Check those files for the service that serves a given model. To run models locally for a **new example**, add your example's profile (e.g. `my-example/server`) to the relevant service(s) there, and add the matching entries to that example's `services.local.yaml` (as shown above). The catalog then picks up the sidecars automatically once they are reachable. See [Deployment Profiles](../01-getting-started.md#docker-based-deployment) for the profile list.

## Adding built-in cloud services

Append entries to the relevant `services.cloud.yaml`. Refresh the browser for host-run development, or rebuild/redeploy Docker to package the change into the image.

```yaml
llm:
  my-custom-llm:
    name: "My Custom LLM"
    model_id: "org/model-name"
    base_url: "https://integrate.api.nvidia.com/v1"
    supported_languages: [en, de]
    system_prompt: ""
    extra_params: ""
```

`supported_languages` is optional LLM capability metadata for the multilingual assistant. When present, the UI offers only session locales whose base language appears in the list. Omit it for a custom LLM when its language capabilities are unknown; this preserves unrestricted, backward-compatible behavior. An explicitly empty list permits no session locales.

```yaml
asr:
  my-custom-asr:
    name: "My Custom ASR"
    server: "grpc.nvcf.nvidia.com:443"
    model: "my-asr-model"
    function_id: ""
```

```yaml
tts:
  my-custom-tts:
    name: "My Custom TTS"
    server: "grpc.nvcf.nvidia.com:443"
    voice_id: "Magpie-Multilingual.EN-US.Aria"
    model: "magpie-tts-multilingual"
    function_id: "<NVCF_FUNCTION_ID>"
    synthesis_mode: stitched
```
