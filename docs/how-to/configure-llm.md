# LLM Models

The cascaded pipeline calls a text **LLM** for response generation. The **Omni** examples use a single audio-input model that performs ASR and the LLM together. All of them are **NVIDIA Nemotron** models, reasoning-capable open models with built-in tool calling, served either from the cloud (NVIDIA-hosted NVCF endpoints) or self-hosted next to the pipeline as a Compose sidecar.

Nemotron models are **transparent**: weights and training data are open on [Hugging Face](https://huggingface.co/nvidia) and the technical reports for reproducing them are public, so you can evaluate a model before putting it in production. The **Nemotron 3** family pairs a hybrid **Mamba-Transformer MoE** architecture for efficient, high-throughput, multimodal agentic AI, and deploys with open frameworks (vLLM, SGLang, Ollama, llama.cpp) on any NVIDIA GPU (edge, cloud, or data center) or as NVIDIA NIM microservices.

The reasoning family is tiered by capability. **Nemotron 3.5 Lightning** is the fast, efficient default for cascaded examples, while **Nemotron 3 Nano Omni** adds multimodal audio input. **Nemotron 3.5 Super VL** offers the highest efficiency with leading accuracy for reasoning and tool calling in multi-agent apps. **Ultra** gives the highest reasoning accuracy for the most complex agentic tasks. Learn more at [NVIDIA Nemotron](https://developer.nvidia.com/topics/ai/nemotron).

Models are declared in the root [`services.yaml`](../../services.yaml), and each example lists the keys it offers in [`examples_registry.yaml`](../../examples_registry.yaml). This page is the **model reference**. It covers the available models, deployment and sizing, reasoning and tool calling, and per-request sampling. For catalog loading, switching, and overrides, refer to [Configure Services](configure-services.md).

## Models

Five models back the examples. Each is served by the self-hosted Compose service(s) below. Nemotron 3.5 Lightning, Nemotron 3.5 Bolt E4B, and Nemotron 3 Nano Omni are also available from the cloud catalog with no sidecar. Nemotron 3.5 Super VL is self-hosted only, and the examples use it as a text LLM. Nemotron 3.5 Bolt 74B is self-hosted on single-GPU hosts only, and the examples use it as a text LLM.

| Model | Self-hosted compose service | Modelcard |
|-------|-----------------------------|-----------|
| **Nemotron 3.5 Lightning 30B A3B**: fast, efficient text LLM | [`docker-compose.nemotron35-lightning-nim.yaml`](../../docker/docker-compose.nemotron35-lightning-nim.yaml) (NIM), [`docker-compose.nemotron35-lightning.yaml`](../../docker/docker-compose.nemotron35-lightning.yaml) (vLLM) | [modelcard](https://build.nvidia.com/nvidia/nemotron-3.5-lightning-30b-a3b/modelcard) |
| **Nemotron 3.5 Super VL 120B A12B**: higher-capability alternative for complex tasks, available self-hosted | [`docker-compose.nemotron35-super-vl.yaml`](../../docker/docker-compose.nemotron35-super-vl.yaml) (`nemotron-35-super-vl` NIM, `nemotron-35-super-vl-vllm` vLLM) | [Hugging Face](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4) |
| **Nemotron 3.5 Bolt 74B A9B**: higher-capability alternative, available self-hosted on single-GPU hosts only | [`docker-compose.nemotron35-bolt.yaml`](../../docker/docker-compose.nemotron35-bolt.yaml) (`nemotron-35-bolt-vllm` vLLM) | [Hugging Face](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB) |
| **Nemotron 3.5 Bolt E4B**: compact conversational and tool-calling small language model (SLM) with a low time to first token (TTFT), 8B total and 4B effective | [`docker-compose.nemotron35-bolt-e4b.yaml`](../../docker/docker-compose.nemotron35-bolt-e4b.yaml) (`nemotron-35-bolt-e4b` NIM, `nemotron-35-bolt-e4b-vllm` vLLM) | [modelcard](https://build.nvidia.com/nvidia/nemotron-3.5-bolt-e4b) |
| **Nemotron 3 Nano Omni 30B A3B**: audio-input model that does ASR and the LLM in one, used by the Omni examples | [`docker-compose.nemotron3-omni-nim.yaml`](../../docker/docker-compose.nemotron3-omni-nim.yaml) (NIM), [`docker-compose.nemotron3-omni.yaml`](../../docker/docker-compose.nemotron3-omni.yaml) (vLLM) | [modelcard](https://build.nvidia.com/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning) |

Each model is exposed as one or more **catalog keys** in `services.yaml`:

| Model | Catalog keys |
|-------|--------------|
| Nemotron 3.5 Lightning | `nemotron-lightning` |
| Nemotron 3.5 Super VL | `nemotron-super` (self-hosted only) |
| Nemotron 3.5 Bolt 74B | `nemotron-bolt-74b` (self-hosted single-GPU only) |
| Nemotron 3.5 Bolt E4B | `nemotron-bolt-4b` |
| Nemotron 3 Nano Omni | `nemotron-omni-nvfp4` |

Reasoning is a per-session toggle on the same entry, not a separate key (see [Reasoning, parser & tool calling](#reasoning-parser--tool-calling)). The active default per slot is the first available key under the example's `services` in [`examples_registry.yaml`](../../examples_registry.yaml).

### Multilingual Session Languages

The multilingual assistant exposes only locales supported by the selected ASR, TTS, and built-in LLM. The LLM lists below are the model-level capability sets; locale variants match their base language (for example, `de-DE` matches `de`).

| Built-in LLM | Supported language bases |
| --- | --- |
| Nemotron 3.5 Lightning (`nemotron-lightning`) | English (`en`), German (`de`), Spanish (`es`), French (`fr`), Italian (`it`), Japanese (`ja`) |
| Nemotron 3.5 Super VL (`nemotron-super`, self-hosted) | English (`en`), German (`de`), Spanish (`es`), French (`fr`), Italian (`it`), Japanese (`ja`), Chinese (`zh`) |
| Nemotron 3.5 Bolt 74B (`nemotron-bolt-74b`, self-hosted single-GPU) | English (`en`), German (`de`), Spanish (`es`), French (`fr`), Italian (`it`), Japanese (`ja`) |
| Nemotron 3.5 Bolt E4B (`nemotron-bolt-4b`) | English (`en`), German (`de`), Spanish (`es`), French (`fr`), Italian (`it`), Japanese (`ja`) |

The source of truth for the built-in capability metadata is the NVIDIA [Nemotron 3.5 Lightning model card](https://build.nvidia.com/nvidia/nemotron-3.5-lightning-30b-a3b/modelcard), the [Nemotron 3.5 Super VL model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4), the [Nemotron 3.5 Bolt 74B model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB), and the [Nemotron 3.5 Bolt E4B model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16).

> **Multilingual conversation quality.** Nemotron 3.5 Lightning's conversation quality is weaker in some languages. For multilingual deployments where language fidelity matters, start the **Nemotron 3.5 Super VL** sidecar (see [Run Nemotron 3.5 Super VL](#run-nemotron-35-super-vl)) and select `nemotron-super` in the Services tab. The Generic and Multilingual assistants list it after Lightning. It stays more reliably in the target language and reads more naturally across languages.

## Hardware Requirements and Deployment Configs

You can self-host the LLM two ways, and the repo wires the right one per profile:

- **NIM** (`nvidia-llm`, `nvidia-llm-omni`, `nemotron-35-super-vl`, `nemotron-35-bolt-e4b`): a prebuilt, optimized inference microservice with automatic, hardware-aware model-profile selection. It is used by `*/server` recipes.
- **vLLM** (`nvidia-llm-vllm*`, `nvidia-llm-vllm-omni`, `nemotron-35-super-vl-vllm`, `nemotron-35-bolt-vllm`, `nemotron-35-bolt-e4b-vllm`): serves weights directly. The `*/single-gpu` recipes select the serving precision automatically from the host and GPU.

Both expose the same OpenAI-compatible API, so the pipeline and the request tuning below behave identically against either.

> Check the [NIM for LLMs support matrix](https://docs.nvidia.com/nim/large-language-models/latest/reference/support-matrix.html) for cascaded models and the [NIM for VLMs support matrix](https://docs.nvidia.com/nim/vision-language-models/2.0.4-variant/support-matrix.html) for Omni before choosing a profile.

### VRAM & Hardware Support

The `*/single-gpu` vLLM services select the model checkpoint and precision from the GPU compute capability. Bolt 74B always uses its IQ2 checkpoint and selects only the platform memory cap. Bolt E4B always uses its BF16 checkpoint and adds the NVFP4 DFlash drafter on Blackwell GPUs. They also calculate `--gpu-memory-utilization` at startup. The planner reserves `VLLM_VRAM_HEADROOM_MIB` from currently free memory, validates the usable amount, and caps the resulting utilization for the platform. The default headroom is 4096 MiB. Set `VLLM_GPU_MEMORY_UTILIZATION` only when an explicit override is required. These overrides apply to automatically sized workstation Lightning recipes, Omni recipes, Super VL, Bolt 74B, and Bolt E4B. Lightning on DGX Spark and Jetson Thor uses a fixed value of `0.35`.

| Service / hardware | Model selection | Automatic VRAM plan | Device IDs |
| --- | --- | --- | --- |
| Lightning on a Blackwell workstation | NVFP4 with DFlash | Free VRAM minus headroom, capped at `0.90`. Requires at least 28 GiB usable. | LLM + ASR + TTS -> `0` |
| Lightning on Ada or Hopper | NVFP4 W4A16 via Marlin | Free VRAM minus headroom, capped at `0.90`. Requires at least 28 GiB usable. | LLM + ASR + TTS -> `0` |
| Lightning on DGX Spark | NVFP4 | Fixed at `0.35` to preserve unified memory for speech and the system. | LLM + ASR + TTS -> `0` |
| Lightning on Jetson Thor | NVFP4 | Fixed at `0.35` to preserve unified memory for speech and the system. | LLM + ASR + TTS -> `0` |
| Omni on DGX Spark or Jetson Thor | NVFP4 | Free unified memory minus headroom, capped at `0.70`. Requires at least 24 GiB usable. | Omni + TTS -> `0` |
| Omni on a Blackwell workstation | NVFP4 | Free VRAM minus headroom, capped at `0.90`. Requires at least 24 GiB usable. | Omni + TTS -> `0` |
| Omni on Ada or Hopper | FP8 | Free VRAM minus headroom, capped at `0.90`. Requires at least 36 GiB usable. | Omni + TTS -> `0` |
| Omni on Ampere | BF16 | Free VRAM minus headroom, capped at `0.90`. Requires at least 66 GiB usable. | Omni + TTS -> `0` |
| Super VL on a Blackwell workstation (96 GB class) | Mixed NVFP4/FP8 with MTP, language model only | Free VRAM minus headroom, capped at `0.90`. Requires at least 78 GiB usable. | Super + ASR + TTS -> `0`, Lightning scaled to zero |
| Super VL on DGX Spark or Jetson Thor | Mixed NVFP4/FP8 with MTP, language model only | Free unified memory minus headroom, capped at `0.75`. Requires at least 78 GiB usable. On Thor, FlashInfer compiles one kernel at a time (`MAX_JOBS=1`) so the compiler does not run out of unified memory after the weights load. | Super + ASR + TTS -> `0`, Lightning scaled to zero |
| Super VL on single-GPU service on Hopper, Ada, or Ampere | This service requires Blackwell NVFP4. It exits with `Unsupported GPU for Super single-GPU (needs Blackwell NVFP4)` and restarts. | Depends on the supported NIM profile. | Use the Super VL server NIM (`nemotron-35-super-vl`) with a matching profile: 1× H200, 2× H100 80 GB, or 2× A100 80 GB. L40 / L40S (Ada) GPUs have no listed Super VL NIM profile and are not supported for Super VL server deployment. |
| Bolt 74B on a Blackwell workstation (SM120) | IQ2 with LiLiCorr DFlash, b12x kernels, language model only | Free VRAM minus headroom, capped at `0.90`. Requires at least 32 GiB usable. | Bolt 74B + ASR + TTS -> `0`, Lightning scaled to zero |
| Bolt 74B on DGX Spark (SM121) | IQ2 with LiLiCorr DFlash, b12x kernels, language model only | Free unified memory minus headroom, capped at `0.75`. Requires at least 32 GiB usable. | Bolt 74B + ASR + TTS -> `0`, Lightning scaled to zero |
| Bolt 74B on Jetson Thor, Hopper, Ada, Ampere, or Blackwell data center GPUs | The b12x kernels target only SM120 and SM121. It exits with `Unsupported GPU for Bolt single-GPU (needs SM120 or SM121): <name> <cc>` and restarts. | Not supported. | No NIM exists for Bolt 74B. Use Lightning, Super VL, or Bolt E4B instead. |
| Bolt E4B on a Blackwell GPU | BF16 with NVFP4 DFlash | Free VRAM minus headroom, capped at `0.90`. Requires at least 24 GiB usable. | Bolt E4B + ASR + TTS -> `0`, Lightning scaled to zero |
| Bolt E4B on DGX Spark or Jetson Thor | BF16 with NVFP4 DFlash | Free unified memory minus headroom, capped at `0.75`. Requires at least 24 GiB usable. On Thor, FlashInfer compiles one kernel at a time (`MAX_JOBS=1`). | Bolt E4B + ASR + TTS -> `0`, Lightning scaled to zero |
| Bolt E4B on Hopper, Ada, or Ampere (SM80+) | BF16 | Free VRAM minus headroom, capped at `0.90`. Requires at least 24 GiB usable. | Bolt E4B + ASR + TTS -> `0`, Lightning scaled to zero |

These values are startup checks used by the planner, not guarantees that every workload will fit. Model weights, KV cache, speech services, context length, and concurrency must all fit within the selected budget.

Server recipes use model-specific NIM profiles and scaling controls instead of the single-GPU VRAM planner. Standard `*/server` services leave `NIM_MODEL_PROFILE` unset so NIM automatically chooses a compatible profile for the visible GPU. The dedicated `generic-assistant/server-perf` benchmark instead pins an NVFP4 TP2 profile for two Blackwell LLM GPUs.

| Server layout | Typical memory | Memory control | Device IDs |
| --- | --- | --- | --- |
| Lightning NIM on one GPU | 80 GB | `NIM_KVCACHE_PERCENT=0.6` (default) | LLM + ASR + TTS -> `0` |
| Lightning NIM split across two GPUs | 40 GB/GPU | `NIM_KVCACHE_PERCENT=0.9` | LLM (`nvidia-llm`) -> `0`, ASR + TTS -> `1` |
| Omni NIM | See the NIM for VLMs support matrix, plus TTS memory when sharing a GPU | Automatic NIM model profile | Omni + TTS -> `0` |
| Nemotron 3.5 Super VL NIM | See the NIM support matrix for the selected profile (1× H200, 2× H100 80 GB, or 2× A100 80 GB; not supported on L40 / L40S) | Automatic NIM model profile | Super (`nemotron-35-super-vl`) -> `1`, `2` |
| Nemotron 3.5 Bolt 74B | No NIM exists for Bolt 74B. It runs on single-GPU hosts only. | n/a | n/a |
| Nemotron 3.5 Bolt E4B NIM | See the NIM support matrix for the selected profile (1× RTX PRO 6000 Blackwell Server Edition, 1× RTX PRO 4500 Blackwell Server Edition, or 1× H100 80 GB) | `NIM_KVCACHE_PERCENT=0.45` | Bolt E4B (`nemotron-35-bolt-e4b`) + ASR + TTS -> `0` |

Update each service's `device_ids` under `deploy.resources.reservations.devices` when splitting services across GPUs.

### Run Nemotron 3.5 Super VL

Super VL is an optional sidecar. It is not part of any example profile, so add its profile next to the example's. Lightning stays the default LLM; select `nemotron-super` in the Services tab to use Super.

```bash
# Server: NIM on GPUs 1 and 2. Lightning can keep running on GPU 0.
docker compose --profile generic-assistant/server --profile nemotron-35-super-vl \
  up -d --scale nvidia-llm=0

# Single-GPU: vLLM on GPU 0. Lightning must be scaled to zero to free the GPU.
docker compose --profile generic-assistant/single-gpu --profile nemotron-35-super-vl-single-gpu \
  up -d --scale nvidia-llm-vllm-lightning=0
```

Use `multilingual-assistant/*` in place of `generic-assistant/*` for the Multilingual Assistant. On the server recipe, `--scale nvidia-llm=0` is optional and only frees GPU `0` for ASR and TTS. Scaling only applies to that `up`; a later `up` without the flag starts Lightning again. To stop the stack, pass the same `--profile` flags to `down`, without `--scale`, which `down` does not accept.

The single-GPU service downloads the model checkpoint from Hugging Face on first start and requires an `HF_TOKEN`. It loads only the language model (`--language-model-only`), so image input is not served.

### Run Nemotron 3.5 Bolt 74B

Bolt 74B is an optional sidecar for single-GPU hosts. It is not part of any example profile, so add its profile next to the example's. No NIM exists for Bolt 74B, so there is no server command. Lightning stays the default LLM. Select `nemotron-bolt-74b` in the Services tab to use Bolt 74B.

```bash
# Single-GPU: vLLM on GPU 0. Lightning must be scaled to zero to free the GPU.
docker compose --profile generic-assistant/single-gpu --profile nemotron-35-bolt-single-gpu \
  up -d --scale nvidia-llm-vllm-lightning=0
```

Use `multilingual-assistant/single-gpu` in place of `generic-assistant/single-gpu` for the Multilingual Assistant. Scaling only applies to that `up`. A later `up` without the flag starts Lightning again. To stop the stack, pass the same `--profile` flags to `down`, without `--scale`, which `down` does not accept.

The service runs only on Blackwell workstation GPUs (SM120), such as RTX PRO 6000 and RTX PRO 5000, and on DGX Spark (SM121). It downloads the model checkpoint and the LiLiCorr DFlash drafter from Hugging Face on first start and requires an `HF_TOKEN`. It loads only the language model (`--language-model-only`), so image input is not served.

### Run Nemotron 3.5 Bolt E4B

Bolt E4B is available from NVIDIA Cloud (`nemotron-bolt-4b` with `NVIDIA_API_KEY` set) and as an optional self-hosted sidecar for `*/server` and `*/single-gpu` recipes. Select `nemotron-bolt-4b` in the Services tab to use Bolt E4B.

Both self-hosted services share the GPU with ASR and TTS in place of Lightning, so scale Lightning to zero. The server NIM limits its KV cache to 45% of GPU memory (`NIM_KVCACHE_PERCENT=0.45`) and its context to 32768 tokens.

```bash
# Server: NIM in place of the Lightning NIM, which must be scaled to zero.
docker compose --profile generic-assistant/server --profile nemotron-35-bolt-e4b \
  up -d --scale nvidia-llm=0

# Single-GPU: vLLM in place of Lightning, which must be scaled to zero.
docker compose --profile generic-assistant/single-gpu --profile nemotron-35-bolt-e4b-single-gpu \
  up -d --scale nvidia-llm-vllm-lightning=0
```

Use `multilingual-assistant/*` in place of `generic-assistant/*` for the Multilingual Assistant. Scaling and `down` work as for [Super VL](#run-nemotron-35-super-vl).

The single-GPU service downloads the BF16 checkpoint (and the NVFP4 DFlash drafter on Blackwell GPUs) from Hugging Face on first start and requires an `HF_TOKEN`. It uses the Triton attention backend (`--attention-backend TRITON_ATTN`) on every platform. The Gemma 4 full-attention layers of Bolt E4B have a head size of 512, which FlashAttention cannot run. vLLM treats Bolt E4B as an omni-capable model, so it also profiles the vision and audio encoders at startup.

The single-GPU vLLM service and the server NIM read the **Reasoning Budget** from `thinking_token_budget`, so the entry uses `nemotron-bolt-4b` (`nemotron-vllm` plus **Conversational Mode**). NVIDIA Cloud rejects that field with `HTTP 400` and reads `reasoning_budget`, so `nvcf: {settings: nemotron-bolt-4b-nvcf}` selects `nemotron-nim` plus **Conversational Mode** there.

**Conversational Mode** is a setting of the `nemotron-bolt-4b` settings profile, not of the service. It is on by default for every Bolt E4B endpoint, including NVIDIA Cloud, the server NIM, and the single-GPU service. Turn it off for a session in the Services tab. The model's own chat template defaults it to off, so a client that calls the endpoint directly gets it off unless the request sets `chat_template_kwargs.conversational_mode`.

### Deployment Tuning Parameters

Single-GPU Compose services select precision and VRAM utilization automatically. Use `.env` only for the optional headroom or utilization override. Standard NIM Server deployments also use hardware-aware automatic profile selection. Only specialized recipes such as `server-perf` pin a profile. For Server deployments, the stock Compose files expose only the NIM settings shown as environment variables below. Use a Compose override for controls that the stock files fix or omit.

| Control | Server NIM | Single-GPU vLLM | Notes |
|----------|--------------|--------------------------|-------|
| **VRAM fit** | `NIM_KVCACHE_PERCENT` (default `0.6`) | `VLLM_VRAM_HEADROOM_MIB` (default `4096`) and optional `VLLM_GPU_MEMORY_UTILIZATION` override | vLLM calculates the utilization from free memory by default. |
| **Precision** | Automatic for standard `*/server`. `server-perf` pins `NIM_MODEL_PROFILE=vllm-nvfp4-tp2-pp1-18.0` | Selected automatically from GPU compute capability | Lightning single-GPU loads the NVFP4 checkpoint on every supported GPU. Hopper and Ada serve it as W4A16 through Marlin. Native NVFP4 compute still needs Blackwell or later. For NIM on older hardware, choose a compatible profile listed by the image. |
| **Hardware / scaling (TP)** | Automatic from the visible GPUs for standard `*/server`. Pinned to TP2 for `server-perf` | `--tensor-parallel-size N` | A pinned TP=N profile needs N visible `device_ids`. Merely exposing N GPUs does not guarantee automatic selection will use all of them. |
| **Context length** | Fixed at `32768` by `NIM_MAX_MODEL_LEN` in the stock Compose files | `--max-model-len` | To change the NIM value, use a Compose override or edit the matching Compose service. Larger context costs more KV-cache VRAM. |
| **Concurrency** | `LLM_MAX_NUM_SEQS` (default `256`) | `--max-num-seqs` (`256` in the Lightning recipe) | Maximum concurrent sequences. Nemotron models are a hybrid **Mamba** model, so each sequence draws one state block from the cache. If startup fails CUDA-graph capture, lower this, for example to `64`–`128`. |
| **Explicit profile** | Automatic for standard `*/server`. Pinned for `server-perf` | n/a | Add `NIM_MODEL_PROFILE=<id-or-description>` to a Compose override to pin a custom profile. |

**Cascaded NIM sizing (`nvidia-llm`).** Weight memory depends on the profile NIM selects. Confirm the selected precision and memory footprint in the startup logs and support matrix. The default `NIM_KVCACHE_PERCENT=0.6` targets one ~80 GB GPU shared with ASR (~15 GB) and TTS (~14 GB). On a smaller supported GPU, move ASR/TTS to a second card (their `device_ids`) and raise `NIM_KVCACHE_PERCENT` only after verifying that the selected LLM profile still fits.

**Lightning vLLM sizing (`nvidia-llm-vllm-lightning`).** The Single-GPU service loads `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` on every supported GPU. Blackwell workstations, DGX Spark, and Jetson Thor serve native NVFP4. Hopper and Ada serve the same checkpoint as W4A16 through Marlin (`--quantization modelopt_fp4`).

**Streaming input.** The Lightning vLLM service also serves a text StreamingInput WebSocket at `ws://nvidia-llm-vllm:8000/v1/streaming-session`, next to chat completions. The single-GPU Super VL service (`nemotron-35-super-vl-vllm`), the Bolt 74B service (`nemotron-35-bolt-vllm`), and the Bolt E4B service (`nemotron-35-bolt-e4b-vllm`) serve the same endpoint on their own ports. The single-GPU `nemotron-lightning`, `nemotron-super`, `nemotron-bolt-74b`, and `nemotron-bolt-4b` entries declare it as `streaming_url`. When an example lists `streaming_input` under `capabilities` in `examples_registry.yaml`, each of those cards shows a **Streaming Input** toggle, and turning it on sends the session to that URL. The Generic Assistant then prefills every ASR update while the user speaks, so the answer starts from a warm KV cache. The toggle appears only for self-hosted vLLM entries with a `streaming_url`.

The streaming server holds the latest word until another word arrives or the turn commits. If an ASR correction changes previously queued prompt tokens, the engine request restarts with the updated prompt. Voice activity detection (VAD) stopping does not guarantee that prefill is complete: final ASR updates or pending chunks can still require processing.

**Omni vLLM sizing (`nvidia-llm-vllm-omni`).** The Single-GPU service selects NVFP4, FP8, or BF16 from the supported GPU compute capability. On DGX Spark and Jetson Thor, it also caps free memory using the host's `MemAvailable` value before calculating utilization. Increase `VLLM_VRAM_HEADROOM_MIB` when more memory must remain available for TTS or the system.

**Pick a NIM model profile.** Standard `*/server` leaves `NIM_MODEL_PROFILE` unset, and NIM chooses a compatible profile from the detected GPU and manifest. Use `NIM_MODEL_PROFILE` only for an explicitly pinned custom deployment. The `server-perf` recipe pins `vllm-nvfp4-tp2-pp1-18.0`, selected and benchmarked on two RTX PRO 6000 Blackwell GPUs. This is an RTX PRO 6000 benchmark baseline, not a portable recommendation. Before running the recipe on another target such as H100, list profiles using the deployed image and actual GPU assignment, benchmark compatible TP2 candidates for TTFT, inter-token latency, and throughput per GPU, then replace the pin with the winner's exact ID or full description. Leave the variable unset when portability is preferred.

For standard `*/server`, inspect the image on its visible LLM GPU:

```bash
docker run --rm --gpus '"device=0"' \
  -e NGC_API_KEY="$NVIDIA_API_KEY" \
  nvcr.io/nim/nvidia/nemotron-3.5-lightning-30b-a3b:2.1.4 \
  list-model-profiles
```

For `server-perf`, inspect the same image with its TP2 GPU assignment:

```bash
docker run --rm --gpus '"device=2,3"' \
  -e NGC_API_KEY="$NVIDIA_API_KEY" \
  nvcr.io/nim/nvidia/nemotron-3.5-lightning-30b-a3b:2.1.4 \
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

## Reasoning, Parser & Tool Calling

### Reasoning (Thinking) On/Off

Nemotron LLMs support a chain-of-thought "thinking" mode. Catalog entries reference a [`settings.yaml`](../../settings.yaml) profile that exposes **Reasoning**, **Reasoning Budget**, **Temperature**, **Top P**, **Max Tokens**, and **Seed** under the active LLM in the Services tab. Omni uses `nemotron-omni-nim` / `nemotron-omni-vllm`, whose temperature range goes up to 2.0; its `OMNI_TEMPERATURE`, `OMNI_TOP_P`, and `OMNI_MAX_TOKENS` env values still apply until a user sets the parameter.

For spoken pipelines, prefer reasoning **OFF**, since thinking adds latency before the first spoken token. Turn it **ON** for complex tool/agent tasks where the quality gain outweighs the delay. Each serving engine names the budget field differently: `nemotron-nim` sends `reasoning_budget` and `nemotron-vllm` sends `thinking_token_budget`. Lightning uses `nemotron-nim` on the server NIM and NVIDIA Cloud, and `nemotron-vllm` on the single-GPU vLLM service. Super VL and Bolt 74B use `nemotron-super`, which has no **Reasoning Budget**. Bolt E4B adds **Conversational Mode** and picks its budget field per endpoint (refer to [Run Nemotron 3.5 Bolt E4B](#run-nemotron-35-bolt-e4b)). When **Max Tokens** is set, every **Reasoning Budget** stays 64 tokens under it, and under 192 **Max Tokens** the backend turns **Reasoning** off for the session, so reasoning cannot use up the tokens for the spoken answer. The Frontend/Backend thinker reads the same `nemotron-lightning` entry as the talker, and its `settings` in `examples_registry.yaml` turn reasoning **ON** with a 1024-token budget. Its 4096 max tokens default lives in the example's `pipeline.py`, like the talker's 2048. Change a profile's `default`, the entry's `extra_params`, or the example's registry `settings` to change the default for every session. See [Configure Services](configure-services.md#editable-settings) for the schema.

> **Known limitation (Reasoning OFF with tool calling).** With reasoning **OFF** (`enable_thinking: false`), smaller models such as **Nemotron 3.5 Lightning 30B A3B** can occasionally emit tool-selection planning text (function names, argument math, or prompt rules) as normal assistant `content` before or instead of the structured tool call, which TTS then speaks aloud. Once this planning text enters the conversation history, subsequent turns in the same session can continue repeating that pattern. Start a new session to clear the conversation context, or enable **Reasoning** / switch to **Nemotron 3.5 Super VL** for tool-heavy sessions.

> **Known limitation (Bolt E4B tool calls as text).** **Nemotron 3.5 Bolt E4B** sometimes writes a tool call as plain text instead of a structured tool call, for example `get_weather(city="Impugn")` or `get_weather{city:<|"|>impugn<|"|>}`. This happens mostly when an argument is unclear, such as a misheard city name. The tool does not run, and TTS speaks the text. Later turns in the same session can repeat the pattern. It happens on the self-hosted services and on NVIDIA Cloud. Start a new session to clear the conversation context, or use **Nemotron 3.5 Lightning** or **Nemotron 3.5 Super VL** for tool-heavy sessions.

> **Known limitation (Bolt E4B on NVIDIA Cloud with Reasoning ON).** With **Reasoning** on, the NVIDIA Cloud endpoint for **Nemotron 3.5 Bolt E4B** returns the model's reasoning inside the reply after a tool result, instead of in the separate reasoning field. TTS then speaks the reasoning before the answer, for example "The user asked for the weather in Delhi. I used the get_weather tool...". Plain turns and the tool call itself are not affected, and the self-hosted services split the reasoning correctly. Keep **Reasoning** off for Bolt E4B on NVIDIA Cloud, or use the self-hosted Bolt E4B service.

> **Known limitation (Bolt E4B weekday of another date).** The `get_current_date_time` tool returns only the current date. When asked the weekday of a past or future date, **Nemotron 3.5 Bolt E4B** works it out from its own knowledge or says it cannot. The weekday it gives can be wrong. For example, it answered "Thursday" for 4 July 1976, which was a Sunday. Questions about the current date and time use the tool and are correct.

### Reasoning Parser & Tool Calling (Self-Hosted)

Cloud (NVCF) endpoints enable the parsers server-side. **Self-hosted NIM and vLLM do not enable them by default**, so the repo's `docker/docker-compose.nemotron3*.yaml` files set them for you. The parser names depend on the model:

| Capability | Flag | Why |
|------------|------|-----|
| Reasoning parser | `--reasoning-parser nemotron_v3` for Lightning, Super VL, and Omni. `ultra_v3` for Bolt 74B. `gemma4` for Bolt E4B. | Separates `<think>` reasoning from `content`, so TTS speaks only the answer and reasoning-OFF works. |
| Tool calling | `--enable-auto-tool-choice --tool-call-parser qwen3_coder` for Lightning, Super VL, Omni, and Bolt 74B. `gemma4` for Bolt E4B. | Enables OpenAI-style function calling. Without it, `tool_choice:"auto"` returns `HTTP 400`. |

- **NIM** receives them through `NIM_PASSTHROUGH_ARGS`, which is already fixed in the Lightning, Super VL, and Bolt E4B Compose files.
- **Raw vLLM** (single-GPU or Omni) takes the same flags directly on `vllm serve`.

## Tuning LLM Request Parameters

LLM request parameters are set per catalog entry using `extra_params`, a JSON string merged into each chat-completion request. OpenAI-standard fields (`temperature`, `top_p`, `max_tokens`) go at the top level of `extra_params`. vLLM/NIM extensions (`repetition_penalty`, `chat_template_kwargs`) go under `extra_body`. Use the following structure to set default sampling for an entry in `services.yaml`:

```yaml
llm:
  nemotron-lightning:
    name: "Nemotron 3.5 Lightning 30B A3B"
    model_id: "nvidia/nemotron-3.5-lightning-30b-a3b"
    base_url: "http://nvidia-llm:8000/v1"
    extra_params: '{"temperature":0.6,"top_p":0.95,"max_tokens":1024,"extra_body":{"repetition_penalty":1.05}}'
    settings: nemotron-nim
```

| Parameter | Where | Typical | Effect |
|-----------|-------|---------|--------|
| `temperature` | top level | `0.6` | Lower = more deterministic, higher = more varied. |
| `top_p` | top level | `0.95` | Nucleus-sampling cutoff. |
| `max_tokens` | top level | `512`–`1024` | Caps response length to keep spoken replies short and latency bounded. |
| `repetition_penalty` | `extra_body` | `1.05` | `> 1` discourages repeated phrasing. |
| `chat_template_kwargs.enable_thinking` | `extra_body` | `false` | Reasoning on/off, set by the entry's `settings` profile. |

> The repo ships `repetition_penalty: 1.05` per entry, and `enable_thinking` comes from the `settings` profile. Add `temperature` / `top_p` / `max_tokens` to the same `extra_params` string to default them. Per session, you can override using the UI or session configurations.

## Reference

- [Troubleshooting guide](../06-troubleshooting.md): self-hosted startup/runtime failures (tool-parser `HTTP 400`, reasoning leaking into speech, `nemotron_v3` parser not found, CUDA-graph / precision aborts) and cloud rate limits (`HTTP 429`).
- [Configure Services](configure-services.md): how the catalog is loaded, switched, and overridden.
- [NIM for LLMs documentation](https://docs.nvidia.com/nim/large-language-models/latest/): [support matrix](https://docs.nvidia.com/nim/large-language-models/latest/reference/support-matrix.html), [model profiles and selection](https://docs.nvidia.com/nim/large-language-models/latest/deployment/model-profiles-and-selection.html), [GPU memory / OOM troubleshooting](https://docs.nvidia.com/nim/large-language-models/latest/troubleshooting/memory.html).
- [vLLM documentation](https://docs.vllm.ai/en/latest/): `vllm serve` flags, quantization, and the OpenAI-compatible server reference.
- [Pipecat NVIDIA LLM service](https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/services/nvidia/llm.py): `NvidiaLLMService`.
