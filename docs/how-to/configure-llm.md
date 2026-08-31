# LLM Models

The cascaded pipeline calls a text **LLM** for response generation. The **Omni** examples use a single audio-input model that performs ASR and the LLM together. All of them are **NVIDIA Nemotron** models, reasoning-capable open models with built-in tool calling, served either from the cloud (NVIDIA-hosted NVCF endpoints) or self-hosted next to the pipeline as a Compose sidecar.

Nemotron models are **transparent**: weights and training data are open on [Hugging Face](https://huggingface.co/nvidia) and the technical reports for reproducing them are public, so you can evaluate a model before putting it in production. The **Nemotron 3** family pairs a hybrid **Mamba-Transformer MoE** architecture for efficient, high-throughput, multimodal agentic AI, and deploys with open frameworks (vLLM, SGLang, Ollama, llama.cpp) on any NVIDIA GPU (edge, cloud, or data center) or as NVIDIA NIM microservices.

The reasoning family is tiered by capability. **Nemotron 3.5 Lightning** is the fast, efficient default for cascaded examples, while **Nemotron 3 Nano Omni** adds multimodal audio input. **Nemotron 3 Super** offers the highest efficiency with leading accuracy for reasoning and tool calling in multi-agent apps. **Ultra** gives the highest reasoning accuracy for the most complex agentic tasks. Learn more at [NVIDIA Nemotron](https://developer.nvidia.com/topics/ai/nemotron).

Models are declared per example in `services.cloud.yaml` (remote / NVCF) and `services.local.yaml` (Compose-managed sidecars). This page is the **model reference**. It covers what's available, how to deploy and size it, how to control reasoning and tool calling, and how to tune per-request sampling. For how the catalog is loaded, switched in the UI, and overridden, see [Configure Services](configure-services.md).

## Models

Three unique Nemotron models back the examples. Each is served by the self-hosted Compose service(s) below, or from the cloud catalog with no sidecar.

| Model | Self-hosted compose service | Modelcard |
|-------|-----------------------------|-----------|
| **Nemotron 3.5 Lightning 30B A3B**: fast, efficient text LLM | [`docker-compose.nemotron35-lightning-nim.yaml`](../../docker/docker-compose.nemotron35-lightning-nim.yaml) (NIM), [`docker-compose.nemotron35-lightning.yaml`](../../docker/docker-compose.nemotron35-lightning.yaml) (vLLM) | [modelcard](https://build.nvidia.com/nvidia/nemotron-3.5-lightning-30b-a3b/modelcard) |
| **Nemotron 3 Super 120B A12B**: recommended for cloud deployments, higher capability for complex tasks | [`docker-compose.nemotron3-super.yaml`](../../docker/docker-compose.nemotron3-super.yaml) | [modelcard](https://build.nvidia.com/nvidia/nemotron-3-super-120b-a12b/modelcard) |
| **Nemotron 3 Nano Omni 30B A3B**: audio-input model that does ASR and the LLM in one, used by the Omni examples | [`docker-compose.nemotron3-omni-nim.yaml`](../../docker/docker-compose.nemotron3-omni-nim.yaml) (NIM), [`docker-compose.nemotron3-omni.yaml`](../../docker/docker-compose.nemotron3-omni.yaml) (vLLM) | [modelcard](https://build.nvidia.com/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning) |

Each model is exposed as one or more **catalog keys** in `services.cloud.yaml` / `services.local.yaml`:

| Model | Catalog keys |
|-------|--------------|
| Nemotron 3.5 Lightning | `nemotron-lightning`, `nemotron-lightning-reasoning` |
| Nemotron 3 Super | `nemotron-super`, `nemotron-super-reasoning` |
| Nemotron 3 Nano Omni | `nemotron-omni-nvfp4` |

The `*-reasoning` keys are the **same weights** with thinking enabled (see [Reasoning, parser & tool calling](#reasoning-parser--tool-calling)). The active default per slot is set in [`examples_registry.yaml`](../../examples_registry.yaml) under `defaults`.

### Multilingual session languages

The multilingual assistant exposes only locales supported by the selected ASR, TTS, and built-in LLM. The LLM lists below are the model-level capability sets; locale variants match their base language (for example, `de-DE` matches `de`).

| Built-in LLM | Supported language bases |
| --- | --- |
| Nemotron 3.5 Lightning (`nemotron-lightning`, `nemotron-lightning-reasoning`) | English (`en`), German (`de`), Spanish (`es`), French (`fr`), Italian (`it`), Japanese (`ja`) |
| Nemotron 3 Super (`nemotron-super`, `nemotron-super-reasoning`) | English (`en`), German (`de`), Spanish (`es`), French (`fr`), Italian (`it`), Japanese (`ja`), Chinese (`zh`) |

The source of truth for the built-in capability metadata is the NVIDIA [Nemotron 3.5 Lightning model card](https://build.nvidia.com/nvidia/nemotron-3.5-lightning-30b-a3b/modelcard) and [Nemotron 3 Super model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8).

> **Multilingual conversation quality.** Nemotron 3.5 Lightning's conversation quality is weaker in some languages (for example Hindi). For multilingual deployments where language fidelity matters, prefer **Nemotron 3 Super** (`nemotron-super`). It stays more reliably in the target language and reads more naturally across languages.

## Hardware requirements and deployment configs

You can self-host the LLM two ways, and the repo wires the right one per profile:

- **NIM** (`nvidia-llm`, `nvidia-llm-omni`, `nemotron-3-super`): a prebuilt, optimized inference microservice with automatic, hardware-aware model-profile selection. It is used by `*/server` recipes.
- **vLLM** (`nvidia-llm-vllm*`, `nvidia-llm-vllm-omni`): serves weights directly. The `*/single-gpu` recipes select the serving precision automatically from the host and GPU.

Both expose the same OpenAI-compatible API, so the pipeline and the request tuning below behave identically against either.

> Check the [NIM for LLMs support matrix](https://docs.nvidia.com/nim/large-language-models/latest/reference/support-matrix.html) for cascaded models and the [NIM for VLMs support matrix](https://docs.nvidia.com/nim/vision-language-models/2.0.4-variant/support-matrix.html) for Omni before choosing a profile.

### VRAM & hardware support

The `*/single-gpu` vLLM services select the model checkpoint and precision from the GPU compute capability. They also calculate `--gpu-memory-utilization` at startup. The planner reserves `VLLM_VRAM_HEADROOM_MIB` from currently free memory, validates the usable amount, and caps the resulting utilization for the platform. The default headroom is 4096 MiB. Set `VLLM_GPU_MEMORY_UTILIZATION` only when an explicit override is required. These overrides apply to automatically sized workstation Lightning recipes and Omni recipes. Lightning on DGX Spark and Jetson Thor uses a fixed value of `0.35`.

| Service / hardware | Model selection | Automatic VRAM plan | Device IDs |
| --- | --- | --- | --- |
| Lightning on a Blackwell workstation | NVFP4 with DFlash | Free VRAM minus headroom, capped at `0.90`. Requires at least 28 GiB usable. | LLM + ASR + TTS -> `0` |
| Lightning on Ada or Hopper | BF16 checkpoint with online FP8 | Free VRAM minus headroom, capped at `0.90`. Requires at least 28 GiB usable. | LLM + ASR + TTS -> `0` |
| Lightning on DGX Spark | NVFP4 with DSpark | Fixed at `0.35` to preserve unified memory for speech and the system. | LLM + ASR + TTS -> `0` |
| Lightning on Jetson Thor | NVFP4 | Fixed at `0.35` to preserve unified memory for speech and the system. | LLM + ASR + TTS -> `0` |
| Omni on DGX Spark or Jetson Thor | NVFP4 | Free unified memory minus headroom, capped at `0.70`. Requires at least 24 GiB usable. | Omni + TTS -> `0` |
| Omni on a Blackwell workstation | NVFP4 | Free VRAM minus headroom, capped at `0.90`. Requires at least 24 GiB usable. | Omni + TTS -> `0` |
| Omni on Ada or Hopper | FP8 | Free VRAM minus headroom, capped at `0.90`. Requires at least 36 GiB usable. | Omni + TTS -> `0` |

These values are startup checks used by the planner, not guarantees that every workload will fit. Model weights, KV cache, speech services, context length, and concurrency must all fit within the selected budget.

Server recipes use model-specific NIM profiles and scaling controls instead of the single-GPU VRAM planner. Standard `*/server` services leave `NIM_MODEL_PROFILE` unset so NIM automatically chooses a compatible profile for the visible GPU. The dedicated `generic-assistant/server-perf` benchmark instead pins an NVFP4 TP2 profile for two Blackwell LLM GPUs.

| Server layout | Typical memory | Memory control | Device IDs |
| --- | --- | --- | --- |
| Lightning NIM on one GPU | 80 GB | `NIM_KVCACHE_PERCENT=0.6` (default) | LLM + ASR + TTS -> `0` |
| Lightning NIM split across two GPUs | 40 GB/GPU | `NIM_KVCACHE_PERCENT=0.9` | LLM (`nvidia-llm`) -> `0`, ASR + TTS -> `1` |
| Omni NIM | See the NIM for VLMs support matrix, plus TTS memory when sharing a GPU | Automatic NIM model profile | Omni + TTS -> `0` |
| Nemotron 3 Super | 2 × 80 GB (`tp=2`) | NIM defaults | LLM split across two GPUs |

Update each service's `device_ids` under `deploy.resources.reservations.devices` when splitting services across GPUs.

### Deployment tuning parameters

Single-GPU Compose services select precision and VRAM utilization automatically. Use `.env` only for the optional headroom or utilization override. Standard NIM Server deployments also use hardware-aware automatic profile selection. Only specialized recipes such as `server-perf` pin a profile.

| Controls | NIM (`.env`) | Single-GPU vLLM | Notes |
|----------|--------------|--------------------------|-------|
| **VRAM fit** | `NIM_KVCACHE_PERCENT` (default `0.6`) | `VLLM_VRAM_HEADROOM_MIB` (default `4096`) and optional `VLLM_GPU_MEMORY_UTILIZATION` override | vLLM calculates the utilization from free memory by default. |
| **Precision** | Automatic for standard `*/server`. `server-perf` pins `NIM_MODEL_PROFILE=vllm-nvfp4-tp2-pp1-18.0` | Selected automatically from GPU compute capability | NVFP4 needs Blackwell or later. On older hardware, choose a compatible profile listed by the NIM image. |
| **Hardware / scaling (TP)** | Automatic from the visible GPUs for standard `*/server`. Pinned to TP2 for `server-perf` | `--tensor-parallel-size N` | A pinned TP=N profile needs N visible `device_ids`. Merely exposing N GPUs does not guarantee automatic selection will use all of them. |
| **Context length** | `NIM_MAX_MODEL_LEN` (default `32768`) | `--max-model-len` | Larger context costs more KV-cache VRAM. |
| **Concurrency** | `LLM_MAX_NUM_SEQS` (default `256`) | `--max-num-seqs` | Max concurrent sequences. Nemotron models are a hybrid **Mamba** model, so each sequence draws one state block from the cache. If startup fails CUDA-graph capture, lower this (e.g. `64`–`128`). |
| **Explicit profile** | `NIM_MODEL_PROFILE=<id>` | n/a | Pin a specific NIM profile instead of auto-selection. |

**Cascaded NIM sizing (`nvidia-llm`).** Weight memory depends on the profile NIM selects. Confirm the selected precision and memory footprint in the startup logs and support matrix. The default `NIM_KVCACHE_PERCENT=0.6` targets one ~80 GB GPU shared with ASR (~15 GB) and TTS (~14 GB). On a smaller supported GPU, move ASR/TTS to a second card (their `device_ids`) and raise `NIM_KVCACHE_PERCENT` only after verifying that the selected LLM profile still fits.

**Omni vLLM sizing (`nvidia-llm-vllm-omni`).** The Single-GPU service selects NVFP4 or FP8 from the supported GPU compute capability. On DGX Spark and Jetson Thor, it also caps free memory using the host's `MemAvailable` value before calculating utilization. Increase `VLLM_VRAM_HEADROOM_MIB` when more memory must remain available for TTS or the system.

**Pick a NIM model profile.** Standard `*/server` leaves `NIM_MODEL_PROFILE` unset, and NIM chooses a compatible profile from the detected GPU and manifest. Use `NIM_MODEL_PROFILE` only for an explicitly pinned custom deployment. The `server-perf` recipe pins `vllm-nvfp4-tp2-pp1-18.0`, selected and benchmarked on two RTX PRO 6000 Blackwell GPUs. This is an RTX PRO 6000 benchmark baseline, not a portable recommendation. Before running the recipe on another target such as H100, list profiles using the deployed image and actual GPU assignment, benchmark compatible TP2 candidates for TTFT, inter-token latency, and throughput per GPU, then replace the pin with the winner's exact ID or full description. Leave the variable unset when portability is preferred.

For standard `*/server`, inspect the image on its visible LLM GPU:

```bash
docker run --rm --gpus '"device=0"' \
  -e NGC_API_KEY="$NVIDIA_API_KEY" \
  nvcr.io/nim/nvidia/nemotron-3.5-lightning-30b-a3b:2.0.9-variant \
  list-model-profiles
```

For `server-perf`, inspect the same image with its TP2 GPU assignment:

```bash
docker run --rm --gpus '"device=2,3"' \
  -e NGC_API_KEY="$NVIDIA_API_KEY" \
  nvcr.io/nim/nvidia/nemotron-3.5-lightning-30b-a3b:2.0.9-variant \
  list-model-profiles
```

For an Omni `*/server` deployment, use its checked-in image and visible GPU:

```bash
docker run --rm --gpus '"device=0"' \
  -e NGC_API_KEY="$NVIDIA_API_KEY" \
  nvcr.io/nim/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:2.0.4-variant \
  list-model-profiles
```

> Profile naming, the selection priority chain, and `NIM_MODEL_PROFILE` are documented in **[NIM model profiles and selection](https://docs.nvidia.com/nim/large-language-models/latest/deployment/model-profiles-and-selection.html)**.

## Reasoning, parser & tool calling

### Reasoning (thinking) on/off

Nemotron LLMs support a chain-of-thought "thinking" mode, controlled per catalog entry through `extra_params`, forwarded to the model as `extra_body`:

```yaml
llm:
  # Reasoning OFF: lowest latency (recommended default for spoken pipelines)
  nemotron-lightning:
    model_id: "nvidia/nemotron-3.5-lightning-30b-a3b"
    extra_params: '{"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}'

  # Reasoning ON: better on complex tasks, higher time-to-first-response
  nemotron-lightning-reasoning:
    model_id: "nvidia/nemotron-3.5-lightning-30b-a3b"
    extra_params: '{"extra_body":{"chat_template_kwargs":{"enable_thinking":true},"reasoning_budget":16384}}'
```

For spoken pipelines, prefer reasoning **OFF**, since thinking adds latency before the first spoken token. Turn it **ON** for complex tool/agent tasks where the quality gain outweighs the delay. Select a variant from the Services tab or set the default in [`examples_registry.yaml`](../../examples_registry.yaml).

### Reasoning parser & tool calling (self-hosted)

Cloud (NVCF) endpoints enable the parsers server-side. **Self-hosted NIM and vLLM do not enable them by default**, so the repo's `docker/docker-compose.nemotron3-*.yaml` set them for you:

| Capability | Flag | Why |
|------------|------|-----|
| Reasoning parser | `--reasoning-parser nemotron_v3` | Separates `<think>` reasoning from `content`, so TTS speaks only the answer and reasoning-OFF works. |
| Tool calling | `--enable-auto-tool-choice --tool-call-parser qwen3_coder` | Enables OpenAI-style function calling. Without it, `tool_choice:"auto"` returns `HTTP 400`. |

- **NIM** passes them via `NIM_PASSTHROUGH_ARGS` (already set in the Lightning / Super compose files).
- **Raw vLLM** (single-GPU / Omni) takes the same flags directly on `vllm serve`.

## Tuning LLM request parameters

LLM request parameters are set per catalog entry via `extra_params`, a JSON string merged into each chat-completion request. OpenAI-standard fields (`temperature`, `top_p`, `max_tokens`) go at the top level of `extra_params`. vLLM/NIM extensions (`repetition_penalty`, `chat_template_kwargs`) go under `extra_body`. This is how you default sampling in the `llm:` section of `services.cloud.yaml` / `services.local.yaml`:

```yaml
llm:
  nemotron-lightning:
    name: "Nemotron 3.5 Lightning 30B A3B"
    model_id: "nvidia/nemotron-3.5-lightning-30b-a3b"
    base_url: "https://integrate.api.nvidia.com/v1"
    extra_params: '{"temperature":0.6,"top_p":0.95,"max_tokens":1024,"extra_body":{"repetition_penalty":1.05,"chat_template_kwargs":{"enable_thinking":false}}}'
```

| Parameter | Where | Typical | Effect |
|-----------|-------|---------|--------|
| `temperature` | top level | `0.6` | Lower = more deterministic, higher = more varied. |
| `top_p` | top level | `0.95` | Nucleus-sampling cutoff. |
| `max_tokens` | top level | `512`–`1024` | Caps response length to keep spoken replies short and latency bounded. |
| `repetition_penalty` | `extra_body` | `1.05` | `> 1` discourages repeated phrasing. |
| `chat_template_kwargs.enable_thinking` | `extra_body` | `false` | Reasoning on/off. |

> The repo ships `repetition_penalty: 1.05` and the appropriate `enable_thinking` per entry. Add `temperature` / `top_p` / `max_tokens` to the same `extra_params` string to default them. Per session, you can override using the UI or session configurations.

## Reference

- [Troubleshooting guide](../06-troubleshooting.md): self-hosted startup/runtime failures (tool-parser `HTTP 400`, reasoning leaking into speech, `nemotron_v3` parser not found, CUDA-graph / precision aborts) and cloud rate limits (`HTTP 429`).
- [Configure Services](configure-services.md): how the catalog is loaded, switched, and overridden.
- [NIM for LLMs documentation](https://docs.nvidia.com/nim/large-language-models/latest/): [support matrix](https://docs.nvidia.com/nim/large-language-models/latest/reference/support-matrix.html), [model profiles and selection](https://docs.nvidia.com/nim/large-language-models/latest/deployment/model-profiles-and-selection.html), [GPU memory / OOM troubleshooting](https://docs.nvidia.com/nim/large-language-models/latest/troubleshooting/memory.html).
- [vLLM documentation](https://docs.vllm.ai/en/latest/): `vllm serve` flags, quantization, and the OpenAI-compatible server reference.
- [Pipecat NVIDIA LLM service](https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/services/nvidia/llm.py): `NvidiaLLMService`.
