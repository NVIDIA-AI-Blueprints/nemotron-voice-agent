# Server NIM preflight

Use only for `*/server` and `generic-assistant/server-perf` on a **workstation**. Skip for cloud and `*/single-gpu`.

**Not supported on DGX Spark or Jetson Thor.** If the hardware readout is Spark or Thor, stop. Use `*/single-gpu` or cloud. Do not run NGC login or this file on those platforms.

Auth and NGC login are in `../SKILL.md`. Do not use `HF_TOKEN` here.

Needs enough workstation GPU VRAM for the selected NIM services. One GPU is valid when capacity is sufficient. Multi-GPU workstations may split speech sidecars and LLM across devices. For the VRAM, memory-knob, and device-placement matrix, see `docs/how-to/configure-llm.md` (VRAM & hardware support).

## Precision

`NIM_TAGS_SELECTOR` defaults to `precision=fp8,tp=1`. That default is not universal. The NIM restart-loops with `Could not match a profile in manifest` when the GPU has no profile at that precision. Blackwell (for example RTX PRO 5000, `compute_cap` 12.0) has **no** `fp8` profile.

1. List profiles on GPU 0. Read the image tag from the recipe compose file rather than assuming `:latest` (`docker/docker-compose.nemotron35-lightning-nim.yaml` for cascaded, `docker/docker-compose.nemotron3-omni-nim.yaml` for Omni):

```bash
docker run --rm --gpus '"device=0"' -e NGC_API_KEY="$NVIDIA_API_KEY" \
  <nim_llm_image> list-model-profiles
```

2. Pick the lightest **Compatible** precision. Prefer readable tags over a profile hash. `tp=1` on one GPU, `tp=N` on N GPUs.

| GPU compute capability | `NIM_TAGS_SELECTOR` precision |
| --- | --- |
| Blackwell (CC 10.0+) | `nvfp4` (no `fp8` profile) |
| Hopper or Ada (CC 8.9–9.0) | `fp8`, or `nvfp4` when listed |
| Ampere (CC 8.0–8.6) | `int4` or `bf16` |
| Below CC 8.0 | unsupported → cloud |

3. Set in `.env`, preserving other keys: `NIM_TAGS_SELECTOR=precision=<compatible-precision>,tp=1`.

This applies to `*/server` only. `generic-assistant/server-perf` pins `precision=bf16,tp=2` as a literal in its Compose service (`nvidia-llm-perf`) and ignores the `.env` `NIM_TAGS_SELECTOR`. It targets a Hopper-class four-GPU host. On Blackwell, prefer a compatible `nvfp4` profile. To retune it, edit the literal in the compose file, not `.env`.

Device placement is **not** an `.env` knob. Standard `*/server` sidecars default to GPU `0`. `generic-assistant/server-perf` places ASR on `0`, TTS on `1`, tensor-parallel LLM on `2` and `3`. To move a service, edit `device_ids` under `deploy.resources.reservations.devices` in its Compose file:

- Server: `docker/docker-compose.nemotron-asr.yaml`, `docker/docker-compose.magpie-tts.yaml`, `docker/docker-compose.magpie-zeroshot-tts.yaml`, `docker/docker-compose.chatterbox-tts.yaml`, `docker/docker-compose.nemotron35-lightning-nim.yaml`, `docker/docker-compose.nemotron3-omni-nim.yaml`
- A `tp=N` LLM needs N free GPUs. List every index it uses and keep those GPUs free of ASR and TTS. Each target index must appear in the hardware readout. One GPU: keep everything on `0`, or use cloud.

Apply only what the hardware readout indicates. Never silently change values.

NIM model-profile docs: https://docs.nvidia.com/nim/large-language-models/latest/deployment/model-profiles-and-selection.html

Cascaded Lightning NIM health: `curl -f http://localhost:18000/v1/health/ready`. Omni NIM health: `curl -f http://localhost:18002/v1/health/ready`.

## Failures

- **`pull access denied` / `unauthorized`** → NGC login missing or expired. Single-GPU does not use `nvcr.io`.
- **`Could not match a profile in manifest`** → default `fp8` has no profile on this GPU. Run `list-model-profiles`, set a Compatible `NIM_TAGS_SELECTOR`, recreate the LLM service.
- **`No available memory for the cache blocks`** → `NIM_KVCACHE_PERCENT` is too **low**. Raise it. CUDA OOM during load is the opposite: lower it, or move TTS off that GPU. `LLM_MAX_NUM_SEQS` can be lowered if CUDA-graph capture fails.
