# Single-GPU Memory Fit

Use only for `*/single-gpu`. Skip for cloud and `*/server`.

Auth is in `../SKILL.md`. Use `HF_TOKEN` only. Do not require `NVIDIA_API_KEY` or `docker login nvcr.io`.

Generic, Multilingual, Omni, and Frontend/Backend cover compatible workstations, DGX Spark, and Jetson Thor. `omni-assistant-subagents/single-gpu` is **not supported on Jetson Thor** (workstation and DGX Spark only). On Thor, deploy `omni-assistant-subagents` (cloud). Orin-class Jetson is unsupported. The Lightning compose file (`docker/docker-compose.nemotron35-lightning.yaml`) and Omni compose file (`docker/docker-compose.nemotron3-omni.yaml`) detect product and compute capability. Speech sidecars live in `docker/docker-compose.nemo-speech-cpp.yaml`.

## Speech GGUFs

Once, from the repo root, as your user (not sudo):

```bash
bash scripts/download-nemo-speech-models.sh
```

The script reads `HF_TOKEN` from `.env`. If `docker compose` already created `models/nemo-speech` as root, the script reclaims ownership automatically.

## Memory Fit

Derive vLLM `--gpu-memory-utilization` from **this host's** measured memory. Never copy a value from another platform or budget against advertised unified-memory size (for example 128 GB).

### 1. Record Available Memory

Record CUDA-visible total and free memory plus host memory:

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
free -b
awk '/MemTotal|MemAvailable/ {print}' /proc/meminfo
```

On DGX Spark and Jetson Thor, treat `MemAvailable` as the host-side limit because GPU, OS, containers, and page cache share unified memory. Use the smaller of CUDA-visible free memory and `MemAvailable`.

### 2. Measure the Speech Reservation

Start **only** the speech sidecar from `recipes.md`. Wait until ASR/TTS are loaded and warmed. Measure again. The drop is the speech reservation. Do not estimate from GGUF file sizes.

```bash
docker compose --profile <recipe> up -d <speech-sidecar>
```

### 3. Calculate the vLLM Budget

Reserve extra memory for the OS, agent, CUDA context, request spikes, and model warm-up (`operational_headroom_bytes`). Calculate the remaining safe vLLM budget after normalizing values to bytes:

```text
safe_vllm_bytes =
  min(cuda_free_before_speech, host_mem_available_before_speech)
  - measured_speech_bytes
  - operational_headroom_bytes

gpu_memory_utilization =
  safe_vllm_bytes / cuda_total_bytes
```

`--gpu-memory-utilization` is a fraction of **total** CUDA-visible memory, not of memory left after speech. Round down, stay conservative, never exceed `0.90`. If any input is missing or there is no clear headroom, do not guess: use cloud or stop and ask.

### 4. Compare the Budget with the Recipe

Compare that budget with the vLLM recipe: `docker/docker-compose.nemotron35-lightning.yaml` for cascaded examples, `docker/docker-compose.nemotron3-omni.yaml` for Omni. If the automatic plan leaves the wrong reserve for this host, set only the supported `.env` controls: `VLLM_VRAM_HEADROOM_MIB` or, when an explicit fraction is required, `VLLM_GPU_MEMORY_UTILIZATION`. If the model does not fit with safe headroom, use cloud.

### 5. Verify the Warm Stack

After `up`, a healthy container is not proof of fit. Warm ASR, TTS, and vLLM, and then complete a spoken turn while watching memory. On OOM, allocation failure, or swap pressure, lower utilization in `0.05` steps and repeat.

The Lightning Compose matrix selects the following model and precision automatically:

- DGX Spark: NVFP4 with DSpark speculative decoding.
- Blackwell workstation: NVFP4 with DFlash speculative decoding.
- Jetson Thor: NVFP4 without a draft model.
- Ada/Hopper workstation: BF16 checkpoint with online FP8 quantization.
- Older compute capabilities: unsupported → cloud.

Omni examples use the hardware selection in `docker/docker-compose.nemotron3-omni.yaml`. Omni vLLM health: `curl -f http://localhost:8002/health`.

## Failures

- Hang or OOM → GGUFs are missing, or `--gpu-memory-utilization` was not sized with the speech sidecar loaded. Follow the memory-fit procedure and use the documented `.env` controls when an override is required.
- `Engine core initialization failed` in `nvidia-llm-vllm-lightning` or `nvidia-llm-vllm-omni` → reclaim page cache and retry:

```bash
free -h
sudo sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
docker compose --profile <recipe> up -d
```

- Omni `No available memory for the cache blocks` on vLLM startup → `--gpu-memory-utilization` is too **low** (no room for KV cache after weights). Raise it. Do not lower it. True CUDA OOM during load is the opposite: the fraction collides with another process on the same GPU. Lower `--gpu-memory-utilization` or `--max-model-len`, or move TTS off that GPU.
