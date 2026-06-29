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
- Recipe profile names are `<example>` for cloud-only deployments and `<example>/<hardware>` for on-prem deployments (for example `generic-assistant` and `generic-assistant/workstation`). The profile is a complete, self-contained recipe — never combine two recipes.
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
- `jetson-thor`: `/proc/device-tree/model` identifies a Jetson platform, or the GPU name is `NVIDIA Thor`.
- `dgx-spark`: `/sys/class/dmi/id/product_name` contains `DGX Spark` or `DGX_Spark` case-insensitively.
- `workstation`: non-DGX Spark, non-Jetson host with enough GPU VRAM for the selected local NIM services. Single-GPU hosts are valid when capacity is sufficient.
- _(omit hardware)_: local platform requirements are not met, or remote/NVCF services are preferred (cloud-only).

3. Prepare `.env`:

```bash
test -f .env || cp .env.example .env
```

Required keys: `NVIDIA_API_KEY` for all recipes. `HF_TOKEN` for any recipe that ends in `/dgx-spark` or `/jetson-thor`, plus `omni-assistant/workstation` and `omni-assistant-subagents/workstation`. `TURN_USERNAME` and `TURN_PASSWORD` are required when adding `--profile turn`.

Compose recipe profiles set `PLATFORM` automatically for UI service filtering.
For host-native `uv run` on-prem testing, set `PLATFORM` in `.env` manually.

For on-prem (`workstation`) recipes, set GPU-aware overrides for the local LLM (`nvidia-llm`) in `.env` from the step 1 readout, and say what you changed and why. The defaults (`NIM_KVCACHE_PERCENT=0.6`, `NIM_TAGS_SELECTOR=precision=fp8,tp=1`) suit one ~80 GB Ada/Hopper GPU running ASR + TTS + LLM together.

- `compute_cap` < 8.9 (e.g. A100/Ampere): FP8 is unsupported (`modelopt ... Minimum capability: 89`) — set `NIM_TAGS_SELECTOR=precision=bf16,tp=1`. BF16 weights are ~60 GB and need an 80 GB GPU dedicated to the LLM (move ASR/TTS to a second GPU) with `NIM_KVCACHE_PERCENT=0.9`. On GPUs too small for the BF16 weights, set `precision=bf16,tp=N` and give the LLM `N` GPUs (`device_ids: ['0','1']` for `tp=2`, ASR/TTS in the cloud), or use a cloud LLM.
- LLM alone on a GPU below ~72 GB (ASR/TTS on a second GPU): raise `NIM_KVCACHE_PERCENT` so `value × VRAM` stays above ~40 GB (≈ `0.9` on a 48 GB L40), else it aborts with `No available memory for the cache blocks`.
- A single GPU below ~72 GB cannot host all three models at once — split ASR/TTS onto a second GPU.
- Startup fails CUDA-graph capture (not the memory error): the cache holds fewer Mamba blocks than sequences — lower `LLM_MAX_NUM_SEQS` (e.g. 64-128).
- Omni Assistant (`nvidia-llm-vllm-omni`) and the DGX Spark / Jetson cascaded vLLM are NVFP4 and need a Blackwell GPU (DGX Spark, Jetson Thor, or a Blackwell workstation); on Ampere (A100) only the cascaded NIM (BF16) or a cloud LLM is viable.

Device placement (which GPU each sidecar uses) is **not** an `.env` knob — `device_ids` are hardcoded to `['0']`. To move a service to GPU `N`, edit `device_ids: ['N']` under `deploy.resources.reservations.devices` in that service's compose file: `docker/docker-compose.nemotron-asr.yaml` (ASR), `docker/docker-compose.magpie-tts.yaml` (TTS), `docker/docker-compose.nemotron3-nano.yaml` (NIM LLM), `docker/docker-compose.nemotron3-omni.yaml` (Omni LLM). A tensor-parallel LLM (`tp=N`) needs `N` GPUs — list every index it uses (e.g. `device_ids: ['0','1']` for `tp=2`) and keep those GPUs free of ASR/TTS. Each target index must appear in the step 1 readout. With only one GPU you cannot split — keep everything on GPU 0, or run the cloud-only profile (no `/workstation`) so ASR/TTS/LLM use NVCF instead.

Apply only what step 1 indicates; never silently change values. See `docs/how-to/configure-llm.md` (VRAM & hardware support) for the full reasoning.

4. Pick the recipe profile:

| Goal | Recipe profile |
| --- | --- |
| Cloud-only Generic Cascaded | `generic-assistant` |
| Cloud-only Multilingual Cascaded | `multilingual-assistant` |
| Cloud-only Omni Assistant | `omni-assistant` |
| Cloud-only Omni Assistant Subagents | `omni-assistant-subagents` |
| Cloud-only Frontend/Backend Agent Airline Assistant | `frontend-backend-agent` |
| Generic Cascaded on a workstation | `generic-assistant/workstation` |
| Generic Cascaded on DGX Spark | `generic-assistant/dgx-spark` |
| Generic Cascaded on Jetson Thor | `generic-assistant/jetson-thor` |
| Multilingual Cascaded on a workstation | `multilingual-assistant/workstation` |
| Multilingual Cascaded on DGX Spark | `multilingual-assistant/dgx-spark` |
| Omni Assistant on a workstation | `omni-assistant/workstation` |
| Omni Assistant on DGX Spark | `omni-assistant/dgx-spark` |
| Omni Assistant on Jetson Thor | `omni-assistant/jetson-thor` |
| Omni Assistant Subagents on a workstation | `omni-assistant-subagents/workstation` |
| Omni Assistant Subagents on DGX Spark | `omni-assistant-subagents/dgx-spark` |
| Frontend/Backend Agent Airline Assistant on a workstation | `frontend-backend-agent/workstation` |


For any on-prem recipe, log in to `nvcr.io` first.

5. Start:

```bash
docker compose --profile <recipe> up -d
```

Add observability profiles freely: `--profile tracing` (Phoenix), `--profile turn` (coturn). Before adding `--profile turn`, follow `references/platform-deployment.md#turn` to populate TURN credentials and any required `TURN_URL`. Use `--build` only after source or `Dockerfile` changes.

After containers are healthy, remind the user that on local recipes (`*/workstation`, `*/dgx-spark`, `*/jetson-thor`) the first voice turn may take longer than later turns while on GPU LLM sidecars finish loading or warm up. This is more common right after a fresh deploy. If later turns are fast, the deploy is fine.

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

App service names follow the active example: `generic-assistant`, `multilingual-assistant`, `omni-assistant`, `omni-assistant-subagents`, or `frontend-backend-agent`. Sidecars keep their own names (`nvidia-llm`, `nvidia-llm-vllm`, `nvidia-llm-vllm-omni`, `nemotron-asr-streaming-english`, `nemotron-asr-streaming-multilingual`, `parakeet-ctc-asr`, `parakeet-rnnt-asr`, `tts-service`, `nemotron-speech`, `booking-server`).

## References

- Hardware details and TURN: `references/platform-deployment.md`
- Generic-only deploy: `references/generic-deploy.md`
- Omni Assistant deploy: `references/omni-assistant-deploy.md`
- Omni Assistant Subagents deploy: `references/omni-assistant-subagents-deploy.md`
- Frontend/Backend Agent deploy: `references/frontend-backend-agent-deploy.md`
