---
name: deploy
description: Deploy Nemotron Voice Agent via root compose using recipe profiles. Use when deploying or troubleshooting auth/startup.
version: "2.0.0"
metadata:
  author: NVIDIA Voice Agent Team <nemotron-voice-agent@nvidia.com>
  tags: [deployment, docker-compose, voice-agent, nemotron]
---

# Nemotron Voice Agent Deployment

## Rules

- Run commands from the repository root containing `docker-compose.yml`.
- Use Docker Compose for deployment.
- Preserve existing `.env`. Create it only if missing.
- Use `configure-pipeline` for `.env`, catalog, or prompt changes.
- Every deployment specifies **exactly one recipe profile** (plus optional observability profiles). `docker compose up` with no profile is a no-op.
- Recipe profile names are `<example>` for cloud-only deployments, `<example>/server` for NIM deployments, and `<example>/single-gpu` for the NeMo-Speech.cpp and vLLM stack. `generic-assistant/server-perf` is a benchmark-only scaling profile, not a normal UI deployment. Each profile is a complete, self-contained recipe. Never combine two recipes.
- Generic, Multilingual, Omni, and Frontend/Backend single-GPU recipes support compatible workstations, DGX Spark, and Jetson Thor. Omni Assistant Subagents is documented for workstations and DGX Spark only. Do not infer that a recipe fits from the platform name. Complete the memory-fit procedure below first.
- Selector modes (`all`, or a single `<example>` such as `generic-assistant`) remain host-native only (`uv run`) and have no compose profile.
- Observability profiles (`tracing`, `turn`) compose orthogonally with any recipe.
- When adding the `turn` profile, complete the TURN preflight before `docker compose up`: confirm bundled coturn is supported on the host architecture, ensure `.env` has `TURN_USERNAME` and `TURN_PASSWORD`, set `TURN_URL` when TURN is hosted separately or the request host is not client-reachable, and remind the user to open UDP `3478` and `49160-49200`.

## Deploy

1. Check hardware:

```bash
cat /sys/class/dmi/id/product_name 2>/dev/null || true
cat /proc/device-tree/model 2>/dev/null || true
nvidia-smi --query-gpu=index,name,memory.total,memory.free,compute_cap --format=csv,noheader
free -h
```

1. Identify the hardware target:
- `single-gpu`: universal one-GPU path for workstations, DGX Spark, and Jetson Thor. The service detects the product and compute capability automatically.
- `server`: NIM ASR + LLM + TTS. Recommended for scaling on workstation / server GPUs.
- _(omit hardware)_: local platform requirements are not met, or remote/NVCF services are preferred (cloud-only).

### Single-GPU memory fit (mandatory)

Before starting a `*/single-gpu` recipe, derive vLLM's
`--gpu-memory-utilization` from measured memory. Never copy a value from another
platform or budget against the advertised 128 GB on a unified-memory system.

1. Record the CUDA-visible total and current free memory:

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
free -b
awk '/MemTotal|MemAvailable/ {print}' /proc/meminfo
```

For DGX Spark and Jetson Thor, treat `MemAvailable` as the host-side limit
because GPU, OS, containers, and page cache share unified memory. Use the
smaller of CUDA-visible free memory and `MemAvailable` when both are available.

1. Start only the NeMo-Speech.cpp sidecar, wait until ASR/TTS are loaded and
warmed, then measure again. The reduction is the measured speech reservation.
Do not estimate it only from GGUF file sizes.

```bash
docker compose --profile generic-assistant/single-gpu up -d nemo-speech
# Omni uses: ... up -d nemo-speech-tts
```

1. Reserve additional memory for the OS, agent, CUDA context, request spikes,
and model warm-up. The remaining safe vLLM budget is:

```text
safe_vllm_bytes =
  min(cuda_free_before_speech, host_mem_available_before_speech)
  - measured_speech_bytes
  - operational_headroom_bytes

gpu_memory_utilization =
  safe_vllm_bytes / cuda_total_bytes
```

`--gpu-memory-utilization` is a fraction of **total CUDA-visible memory**, not a
fraction of memory left after speech loads. Round the result down, start
conservatively, and never exceed `0.90`. Normalize all measurements to bytes
before calculating. If any input is unavailable or the
result leaves no clear headroom, do not guess: use cloud services or stop and
ask the user.

1. Compare the safe budget with the vLLM recipe for the selected example.
Use `docker/docker-compose.nemotron35-lightning.yaml` for cascaded examples and
`docker/docker-compose.nemotron3-omni.yaml` for Omni examples. Do not add
host-specific vLLM settings to `.env`. If the selected recipe does not fit, use
cloud services rather than silently overriding the support matrix.

1. Validate under concurrent load: warm ASR, TTS, and vLLM, then complete a
spoken exchange while monitoring memory. On OOM, allocation failure, or system
swap pressure, lower utilization in `0.05` steps and repeat. A successful
container startup alone is not proof of fit.

1. Prepare `.env`:

```bash
test -f .env || cp .env.example .env
```

Required keys: `NVIDIA_API_KEY` for cloud and server recipes. `HF_TOKEN` is
required for `*/single-gpu` recipes.
`TURN_USERNAME` and `TURN_PASSWORD` are required when adding `--profile turn`.

For `*/single-gpu`, download the NeMo-Speech.cpp GGUFs once before deployment:

```bash
bash scripts/download-nemo-speech-models.sh
```

Local catalogs merge by TCP reachability: NIM sidecars from `*/server` and
NeMo-Speech.cpp from `*/single-gpu` appear when those endpoints are up.
Host-native `uv run` uses the same rule.

The cascaded Lightning support matrix is fixed in its compose file:
- DGX Spark: NVFP4 with DSpark speculative decoding.
- Blackwell workstation: NVFP4 with DFlash speculative decoding.
- Jetson Thor: NVFP4 without a draft model.
- Ada/Hopper workstation: BF16 checkpoint with online FP8 quantization.
- Older compute capabilities are unsupported. Use cloud services.

Device placement is **not** an `.env` knob. Standard `*/server` sidecars default to GPU `0`. The four-GPU `generic-assistant/server-perf` recipe places ASR on GPU `0`, TTS on GPU `1`, and the tensor-parallel LLM on GPUs `2` and `3`. To move a service, edit `device_ids` under `deploy.resources.reservations.devices` in its Compose file. Server files include `docker/docker-compose.nemotron-asr.yaml`, `docker/docker-compose.magpie-tts.yaml`, `docker/docker-compose.magpie-zeroshot-tts.yaml`, `docker/docker-compose.chatterbox-tts.yaml`, `docker/docker-compose.nemotron35-lightning-nim.yaml`, and `docker/docker-compose.nemotron3-omni-nim.yaml`. Single-GPU files include `docker/docker-compose.nemo-speech-cpp.yaml`, `docker/docker-compose.nemotron35-lightning.yaml`, and `docker/docker-compose.nemotron3-omni.yaml`. A tensor-parallel LLM (`tp=N`) needs `N` GPUs. List every index it uses and keep those GPUs free of ASR and TTS. Each target index must appear in the step 1 readout. With only one GPU you cannot split. Keep everything on GPU 0, or run the cloud-only profile so ASR, TTS, and LLM use NVCF instead.

Apply only what step 1 indicates. Never silently change values. See `docs/how-to/configure-llm.md` (VRAM & hardware support) for the full reasoning.

1. Pick the recipe profile:

| Goal | Recipe profile |
| --- | --- |
| Cloud-only Generic Cascaded | `generic-assistant` |
| Cloud-only Multilingual Cascaded | `multilingual-assistant` |
| Cloud-only Omni Assistant | `omni-assistant` |
| Cloud-only Omni Assistant Subagents | `omni-assistant-subagents` |
| Cloud-only Frontend/Backend Agent Airline Assistant | `frontend-backend-agent` |
| Generic Cascaded, single GPU (workstation, DGX Spark, Jetson Thor) | `generic-assistant/single-gpu` |
| Multilingual Cascaded, single GPU (workstation, DGX Spark, Jetson Thor) | `multilingual-assistant/single-gpu` |
| Omni Assistant, single GPU (workstation, DGX Spark, Jetson Thor) | `omni-assistant/single-gpu` |
| Omni Assistant Subagents, single GPU (workstation, DGX Spark) | `omni-assistant-subagents/single-gpu` |
| Frontend/Backend Agent, single GPU (workstation, DGX Spark, Jetson Thor) | `frontend-backend-agent/single-gpu` |
| Generic Cascaded NIM stack | `generic-assistant/server` |
| Generic Cascaded NIM performance benchmark (benchmark-only) | `generic-assistant/server-perf` |
| Multilingual Cascaded NIM stack (recommended for scaling) | `multilingual-assistant/server` |
| Omni Assistant NIM stack (recommended for scaling) | `omni-assistant/server` |
| Omni Assistant Subagents NIM stack (recommended for scaling) | `omni-assistant-subagents/server` |
| Frontend/Backend Agent NIM stack (recommended for scaling) | `frontend-backend-agent/server` |

For any on-prem recipe, log in to `nvcr.io` first.

1. Start:

```bash
docker compose --profile <recipe> up -d
```

Add observability profiles freely: `--profile tracing` (Phoenix), `--profile turn` (coturn). Before adding `--profile turn`, follow `references/platform-deployment.md#turn` to populate TURN credentials and any required `TURN_URL`. Use `--build` only after source or `Dockerfile` changes.

After containers are healthy, remind the user that on local recipes (`*/server`, `*/single-gpu`) the first voice turn may take longer than later turns while on GPU LLM sidecars finish loading or warm up. This is more common right after a fresh deploy. If later turns are fast, the deploy is fine.

1. Verify:

```bash
docker compose ps
docker compose logs --tail 200 <service-name>
```

For TURN deployments, also verify `coturn` is running and the app publishes ICE config:

```bash
docker compose ps coturn
# HTTPS by default. If PIPELINE_TLS=false the HTTPS call fails and the HTTP one returns the config
curl -k https://localhost:${PIPELINE_APP_PORT:-7860}/api/ice-servers \
  || curl http://localhost:${PIPELINE_APP_PORT:-7860}/api/ice-servers
```

App service names follow the active example. Server recipes use `generic-assistant-server`, `generic-assistant-server-perf`, `multilingual-assistant-server`, `omni-assistant-server`, `omni-assistant-subagents-server`, and `frontend-backend-agent-server`. Single-GPU recipes use `generic-assistant-single-gpu`, `multilingual-assistant-single-gpu`, `omni-assistant-single-gpu`, `omni-assistant-subagents-single-gpu`, and `frontend-backend-agent-single-gpu`. Their speech sidecars are `nemo-speech`, `nemo-speech-multilingual`, and `nemo-speech-tts`.

## References

- Hardware details and TURN: `references/platform-deployment.md`
- Generic-only deploy: `references/generic-deploy.md`
- Omni Assistant deploy: `references/omni-assistant-deploy.md`
- Omni Assistant Subagents deploy: `references/omni-assistant-subagents-deploy.md`
- Frontend/Backend Agent deploy: `references/frontend-backend-agent-deploy.md`
