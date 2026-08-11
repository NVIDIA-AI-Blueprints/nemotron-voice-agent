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
- Recipe profile names are `<example>` for cloud-only deployments, `<example>/<hardware>` for existing platform-specific deployments, and `<example>/single-gpu` for the NeMo-Speech.cpp + vLLM stack. The profile is a complete, self-contained recipe — never combine two recipes.
- `generic-assistant/single-gpu`, `multilingual-assistant/single-gpu`, and `omni-assistant/single-gpu` use the same profile on a supported workstation, DGX Spark, or Jetson Thor. It is the only supported recipe on Jetson Thor. Do not infer that it fits from the platform name; complete the memory-fit procedure below first.
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

2. Identify the hardware target:
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

2. Start only the NeMo-Speech.cpp sidecar, wait until ASR/TTS are loaded and
warmed, then measure again. The reduction is the measured speech reservation.
Do not estimate it only from GGUF file sizes.

```bash
docker compose --profile generic-assistant/single-gpu up -d nemo-speech
# Omni uses: ... up -d nemo-speech-tts
```

3. Reserve additional memory for the OS, agent, CUDA context, request spikes,
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

4. Compare the safe budget with the fixed Lightning recipe selected in
`docker/docker-compose.nemotron35-lightning.yaml`. Do not add host-specific
vLLM settings to `.env`; if the selected recipe does not fit, use cloud
services rather than silently overriding the support matrix.

5. Validate under concurrent load: warm ASR, TTS, and vLLM, then complete a
spoken exchange while monitoring memory. On OOM, allocation failure, or system
swap pressure, lower utilization in `0.05` steps and repeat. A successful
container startup alone is not proof of fit.

3. Prepare `.env`:

```bash
test -f .env || cp .env.example .env
```

Required keys: `NVIDIA_API_KEY` for all recipes. `HF_TOKEN` for local vLLM
recipes, including `*/single-gpu` and Omni `/server` profiles.
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
- Ampere workstation: BF16.
- Older compute capabilities: unsupported; use cloud services.

Device placement (which GPU each sidecar uses) is **not** an `.env` knob — `device_ids` are hardcoded to `['0']`. To move a service to GPU `N`, edit `device_ids: ['N']` under `deploy.resources.reservations.devices` in that service's compose file: `docker/docker-compose.nemotron-asr.yaml` (ASR), `docker/docker-compose.magpie-tts.yaml` (TTS), `docker/docker-compose.nemotron35-lightning-nim.yaml` (NIM LLM), `docker/docker-compose.nemotron3-omni.yaml` (Omni LLM). A tensor-parallel LLM (`tp=N`) needs `N` GPUs — list every index it uses (e.g. `device_ids: ['0','1']` for `tp=2`) and keep those GPUs free of ASR/TTS. Each target index must appear in the step 1 readout. With only one GPU you cannot split — keep everything on GPU 0, or run the cloud-only profile (no `/server`) so ASR/TTS/LLM use NVCF instead.

Apply only what step 1 indicates; never silently change values. See `docs/how-to/configure-llm.md` (VRAM & hardware support) for the full reasoning.

4. Pick the recipe profile:

| Goal | Recipe profile |
| --- | --- |
| Cloud-only Generic Cascaded | `generic-assistant` |
| Cloud-only Multilingual Cascaded | `multilingual-assistant` |
| Cloud-only Omni Assistant | `omni-assistant` |
| Cloud-only Omni Assistant Subagents | `omni-assistant-subagents` |
| Cloud-only Frontend/Backend Agent Airline Assistant | `frontend-backend-agent` |
| Generic Cascaded, single GPU (workstation / DGX Spark / Jetson Thor) | `generic-assistant/single-gpu` |
| Multilingual Cascaded, single GPU (workstation / DGX Spark / Jetson Thor) | `multilingual-assistant/single-gpu` |
| Omni Assistant, single GPU (workstation / DGX Spark / Jetson Thor) | `omni-assistant/single-gpu` |
| Omni Assistant Subagents, single GPU (workstation / DGX Spark / Jetson Thor) | `omni-assistant-subagents/single-gpu` |
| Frontend/Backend Agent, single GPU (workstation / DGX Spark / Jetson Thor) | `frontend-backend-agent/single-gpu` |
| Generic Cascaded NIM stack (recommended for scaling) | `generic-assistant/server` |
| Multilingual Cascaded NIM stack (recommended for scaling) | `multilingual-assistant/server` |
| Omni Assistant NIM TTS (recommended for scaling) | `omni-assistant/server` |
| Omni Assistant Subagents NIM TTS (recommended for scaling) | `omni-assistant-subagents/server` |
| Frontend/Backend Agent NIM stack (recommended for scaling) | `frontend-backend-agent/server` |


For any on-prem recipe, log in to `nvcr.io` first.

5. Start:

```bash
docker compose --profile <recipe> up -d
```

Add observability profiles freely: `--profile tracing` (Phoenix), `--profile turn` (coturn). Before adding `--profile turn`, follow `references/platform-deployment.md#turn` to populate TURN credentials and any required `TURN_URL`. Use `--build` only after source or `Dockerfile` changes.

After containers are healthy, remind the user that on local recipes (`*/server`, `*/single-gpu`) the first voice turn may take longer than later turns while on GPU LLM sidecars finish loading or warm up. This is more common right after a fresh deploy. If later turns are fast, the deploy is fine.

6. Verify:

```bash
docker compose ps
docker compose logs --tail 200 <service-name>
```

For TURN deployments, also verify `coturn` is running and the app publishes ICE config:

```bash
docker compose ps coturn
# HTTPS by default; if PIPELINE_TLS=false the HTTPS call fails and the HTTP one returns the config
curl -k https://localhost:${PIPELINE_APP_PORT:-7860}/api/ice-servers \
  || curl http://localhost:${PIPELINE_APP_PORT:-7860}/api/ice-servers
```

App service names follow the active example. Server NIM recipes use `*-server` app services (`generic-assistant-server`, `omni-assistant-server`, …). Single-GPU app services are `generic-assistant-single-gpu`, `multilingual-assistant-single-gpu`, and `omni-assistant-single-gpu`; their speech sidecars are `nemo-speech`, `nemo-speech-multilingual`, and `nemo-speech-tts`.

## References

- Hardware details and TURN: `references/platform-deployment.md`
- Generic-only deploy: `references/generic-deploy.md`
- Omni Assistant deploy: `references/omni-assistant-deploy.md`
- Omni Assistant Subagents deploy: `references/omni-assistant-subagents-deploy.md`
- Frontend/Backend Agent deploy: `references/frontend-backend-agent-deploy.md`
