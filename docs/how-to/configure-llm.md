# LLM Models

The cascaded pipeline calls a text **LLM** for response generation. The **Omni** examples use a single audio-input model that performs ASR and the LLM together. All of them are **NVIDIA Nemotron** models, reasoning-capable open models with built-in tool calling, served either from the cloud (NVIDIA-hosted NVCF endpoints) or self-hosted next to the pipeline as a Compose sidecar.

Nemotron models are **transparent**: weights and training data are open on [Hugging Face](https://huggingface.co/nvidia) and the technical reports for reproducing them are public, so you can evaluate a model before putting it in production. The **Nemotron 3** family pairs a hybrid **Mamba-Transformer MoE** architecture for efficient, high-throughput, multimodal agentic AI, and deploys with open frameworks (vLLM, SGLang, Ollama, llama.cpp) on any NVIDIA GPU (edge, cloud, or data center) or as NVIDIA NIM microservices.

The reasoning family is tiered by platform. **Nano** is cost-efficient with high accuracy for specialized sub-agents, and is multimodal via **Nano Omni**. **Super** offers the highest efficiency with leading accuracy for reasoning and tool calling in multi-agent apps. **Ultra** gives the highest reasoning accuracy for the most complex agentic tasks. Learn more at [NVIDIA Nemotron](https://developer.nvidia.com/topics/ai/nemotron).

Models are declared per example in `services.cloud.yaml` (remote / NVCF) and `services.local.yaml` (Compose-managed sidecars). This page is the **model reference**. It covers what's available, how to deploy and size it, how to control reasoning and tool calling, and how to tune per-request sampling. For how the catalog is loaded, switched in the UI, and overridden, see [Configure Services](configure-services.md).

## Models

Three unique Nemotron models back the examples. Each is served by the self-hosted Compose service(s) below, or from the cloud catalog with no sidecar.

| Model | Self-hosted compose service | Modelcard |
|-------|-----------------------------|-----------|
| **Nemotron 3 Nano 30B A3B**: fast, efficient text LLM | [`docker-compose.nemotron3-nano.yaml`](../../docker/docker-compose.nemotron3-nano.yaml) | [modelcard](https://build.nvidia.com/nvidia/nemotron-3-nano-30b-a3b/modelcard) |
| **Nemotron 3 Super 120B A12B**: recommended for cloud deployments, higher capability for complex tasks | [`docker-compose.nemotron3-super.yaml`](../../docker/docker-compose.nemotron3-super.yaml) | [modelcard](https://build.nvidia.com/nvidia/nemotron-3-super-120b-a12b/modelcard) |
| **Nemotron 3 Nano Omni 30B A3B**: audio-input model that does ASR and the LLM in one, used by the Omni examples | [`docker-compose.nemotron3-omni.yaml`](../../docker/docker-compose.nemotron3-omni.yaml) | [modelcard](https://build.nvidia.com/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning) |

Each model is exposed as one or more **catalog keys** in `services.cloud.yaml` / `services.local.yaml`:

| Model | Catalog keys |
|-------|--------------|
| Nemotron 3 Nano | `nemotron-nano`, `nemotron-nano-reasoning` |
| Nemotron 3 Super | `nemotron-super`, `nemotron-super-reasoning` |
| Nemotron 3 Nano Omni | `nemotron-omni-nvfp4` |

The `*-reasoning` keys are the **same weights** with thinking enabled (see [Reasoning, parser & tool calling](#reasoning-parser--tool-calling)). The active default per slot is set in [`examples_registry.yaml`](../../examples_registry.yaml) under `defaults`.

## Hardware requirements and deployment configs

You can self-host the LLM two ways, and the repo wires the right one per profile:

- **NIM** (`nvidia-llm`, `nemotron-3-super`): a prebuilt, optimized inference microservice with automatic, hardware-aware **model-profile** selection. Recommended for Nano and Super on supported data-center / workstation GPUs. Used by the `*/workstation` profiles.
- **vLLM** (`nvidia-llm-vllm*`, `nvidia-llm-vllm-omni`): serves the weights directly with `vllm serve`, giving explicit control over every flag. Used where a NIM profile isn't the right fit: the **Omni** NVFP4 model, and **DGX Spark** / **Jetson Thor** edge deployments. This is more manual, since you set precision, parsers, and memory flags yourself.

Both expose the same OpenAI-compatible API, so the pipeline and the request tuning below behave identically against either.

> Check the **[NIM for LLMs support matrix](https://docs.nvidia.com/nim/large-language-models/latest/reference/support-matrix.html)** for the GPUs, precisions, and tensor-parallel sizes each Nemotron NIM supports before choosing a profile.

### VRAM & hardware support

Single-GPU deployments need ≥ 80 GB VRAM. On a dual-GPU host you can drop to ~40 GB per GPU by placing the LLM on one GPU and the speech sidecars on the other. **Precision must match the GPU:** FP8 needs compute capability ≥ 8.9 (Ada / Hopper / Blackwell), A100 / Ampere needs BF16 (~2× the weight size), and NVFP4 needs Blackwell or later.

| Model / layout | Min VRAM | Memory knob | Device IDs |
| --- | --- | --- | --- |
| Nemotron 3 Nano (single GPU) | 80 GB | `NIM_KVCACHE_PERCENT=0.6` (default) | LLM + ASR + TTS -> `0` |
| Nemotron 3 Nano (dual GPU) | 40 GB/GPU | `NIM_KVCACHE_PERCENT=0.9` | LLM (`nvidia-llm`) -> `0`, ASR + TTS -> `1` |
| Omni (single GPU) | 80 GB | `--gpu-memory-utilization 0.3` | LLM (`nvidia-llm-vllm-omni`) + TTS -> `0` (ASR runs inside Omni) |
| Omni (dual GPU) | 40 GB/GPU | `--gpu-memory-utilization 0.9` | LLM (`nvidia-llm-vllm-omni`) -> `0`, TTS -> `1` |
| Nemotron 3 Super | 2 × 80 GB (`tp=2`) | NIM defaults (no `NIM_KVCACHE_PERCENT`) | LLM split across two GPUs |

Update each service's `device_ids` under `deploy.resources.reservations.devices` when splitting services across GPUs.

### Deployment tuning parameters

These control VRAM fit, precision, hardware mapping, and scaling.

| Controls | NIM (`.env`) | vLLM (`vllm serve` flag) | Notes |
|----------|--------------|--------------------------|-------|
| **VRAM fit** | `NIM_KVCACHE_PERCENT` (default `0.6`) | `--gpu-memory-utilization` (`0.3` shared / `0.9` dedicated) | Fraction of **total** GPU VRAM for weights + KV cache. Too low triggers `No available memory for the cache blocks`, so **raise** it. |
| **Precision** | `NIM_TAGS_SELECTOR=precision=fp8\|bf16,...` | quantization baked into the served checkpoint | FP8 (compute capability ≥ 8.9, Ada or later), BF16 (Ampere or later), NVFP4 (Blackwell or later). |
| **Hardware / scaling (TP)** | `NIM_TAGS_SELECTOR=...,tp=N` | `--tensor-parallel-size N` | Shard the model across `N` GPUs and give it `N` `device_ids`. |
| **Context length** | `NIM_MAX_MODEL_LEN` (default `32768`) | `--max-model-len` | Larger context costs more KV-cache VRAM. |
| **Concurrency** | `LLM_MAX_NUM_SEQS` (default `256`) | `--max-num-seqs` | Max concurrent sequences. Nemotron models are a hybrid **Mamba** model, so each sequence draws one state block from the cache. If startup fails CUDA-graph capture, lower this (e.g. `64`–`128`). |
| **Explicit profile** | `NIM_MODEL_PROFILE=<id>` | n/a | Pin a specific NIM profile instead of auto-selection. |

**Cascaded NIM sizing (`nvidia-llm`).** FP8 Nano weights are ~30 GB, so `NIM_KVCACHE_PERCENT × (GPU VRAM)` must stay above ~40 GB (weights + a usable KV cache). The default `0.6` suits one ~80 GB GPU shared with ASR (~15 GB) and TTS (~14 GB). On a smaller GPU, move ASR/TTS to a second card (their `device_ids`) and raise `NIM_KVCACHE_PERCENT` (e.g. `0.9` on a 48 GB L40). For A100/Ampere, switch to BF16. The weights are ~2× larger, so set `NIM_TAGS_SELECTOR=precision=bf16,tp=1` and `NIM_KVCACHE_PERCENT=0.9` on a dedicated 80 GB GPU, or split with `tp=N`.

**Omni vLLM sizing (`nvidia-llm-vllm-omni`).** NVFP4 weights are smaller (~15 GB) and ASR runs in-process, so it ships a lower `--gpu-memory-utilization 0.3` when sharing an 80 GB GPU with TTS. Increase it up to `0.9` on a dedicated GPU. NVFP4 requires a **Blackwell** GPU (DGX Spark, Jetson Thor, or a Blackwell workstation GPU). Only NVFP4 is supported in the compose profiles. To deploy other precisions, see the [Hugging Face documentation](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16).

**Pick a NIM model profile.** NIM auto-selects a profile from your GPU, precision, and TP size. List what's compatible, then optionally pin one with `NIM_MODEL_PROFILE`:

```bash
docker run --rm --gpus all \
  -e NGC_API_KEY="$NVIDIA_API_KEY" \
  nvcr.io/nim/nvidia/nemotron-3-nano:2.0.5 \
  list-model-profiles
```

> Profile naming, the selection priority chain, and `NIM_MODEL_PROFILE` are documented in **[NIM model profiles and selection](https://docs.nvidia.com/nim/large-language-models/latest/deployment/model-profiles-and-selection.html)**.

## Reasoning, parser & tool calling

### Reasoning (thinking) on/off

Nemotron LLMs support a chain-of-thought "thinking" mode, controlled per catalog entry through `extra_params`, forwarded to the model as `extra_body`:

```yaml
llm:
  # Reasoning OFF — lowest latency (recommended default for spoken pipelines)
  nemotron-nano:
    model_id: "nvidia/nemotron-3-nano-30b-a3b"
    extra_params: '{"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}'

  # Reasoning ON — better on complex tasks, higher time-to-first-response
  nemotron-nano-reasoning:
    model_id: "nvidia/nemotron-3-nano-30b-a3b"
    extra_params: '{"extra_body":{"chat_template_kwargs":{"enable_thinking":true}}}'
```

For spoken pipelines, prefer reasoning **OFF**, since thinking adds latency before the first spoken token. Turn it **ON** for complex tool/agent tasks where the quality gain outweighs the delay. Select a variant from the Services tab or set the default in [`examples_registry.yaml`](../../examples_registry.yaml).

### Reasoning parser & tool calling (self-hosted)

Cloud (NVCF) endpoints enable the parsers server-side. **Self-hosted NIM and vLLM do not enable them by default**, so the repo's `docker/docker-compose.nemotron3-*.yaml` set them for you:

| Capability | Flag | Why |
|------------|------|-----|
| Reasoning parser | `--reasoning-parser nemotron_v3` | Separates `<think>` reasoning from `content`, so TTS speaks only the answer and reasoning-OFF works. |
| Tool calling | `--enable-auto-tool-choice --tool-call-parser qwen3_coder` | Enables OpenAI-style function calling. Without it, `tool_choice:"auto"` returns `HTTP 400`. |

- **NIM** passes them via `NIM_PASSTHROUGH_ARGS` (already set in the Nano / Super compose files).
- **Raw vLLM** (DGX Spark / Jetson / Omni) takes the same flags directly on `vllm serve`.

## Tuning LLM request parameters

LLM request parameters are set per catalog entry via `extra_params`, a JSON string merged into each chat-completion request. OpenAI-standard fields (`temperature`, `top_p`, `max_tokens`) go at the top level of `extra_params`. vLLM/NIM extensions (`repetition_penalty`, `chat_template_kwargs`) go under `extra_body`. This is how you default sampling in the `llm:` section of `services.cloud.yaml` / `services.local.yaml`:

```yaml
llm:
  nemotron-nano:
    name: "Nemotron 3 Nano 30B A3B"
    model_id: "nvidia/nemotron-3-nano-30b-a3b"
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
