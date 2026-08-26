# Server NIM preflight

Use only for `*/server` and `generic-assistant/server-perf` on a **workstation**. Skip for cloud and `*/single-gpu`.

**Not supported on DGX Spark or Jetson Thor.** If the hardware readout is Spark or Thor, stop. Use `*/single-gpu` or cloud. Do not run NGC login or this file on those platforms.

Auth and NGC login are in `../SKILL.md`. Do not use `HF_TOKEN` here.

Needs enough workstation GPU VRAM for the selected NIM services. One GPU is valid when capacity is sufficient. Multi-GPU workstations may split speech sidecars and LLM across devices. For the VRAM, memory-knob, and device-placement matrix, see `docs/how-to/configure-llm.md` (VRAM & hardware support).

## Precision

Standard `*/server` services leave `NIM_TAGS_SELECTOR` unset so NIM can select a hardware-compatible profile for the single visible LLM GPU. Confirm the selected profile in the startup logs; automatic selection does not guarantee that the remaining VRAM is enough to colocate ASR and TTS.

1. List profiles on GPU 0. Read the image tag from the recipe compose file rather than assuming `:latest` (`docker/docker-compose.nemotron35-lightning-nim.yaml` for cascaded, `docker/docker-compose.nemotron3-omni-nim.yaml` for Omni):

```bash
docker run --rm --gpus '"device=0"' -e NGC_API_KEY="$NVIDIA_API_KEY" \
  <nim_llm_image> list-model-profiles
```

1. Check that the manifest contains a **Compatible** profile for the target GPU. Prefer the lightest compatible precision when pinning a profile. `tp=1` uses one GPU; `tp=N` needs N visible GPUs.

| GPU compute capability | Preferred compatible precision |
| --- | --- |
| Blackwell (CC 10.0+) | `nvfp4` (no `fp8` profile) |
| Hopper or Ada (CC 8.9–9.0) | `bf16` or another compatible quantized profile listed by the image |
| Ampere (CC 8.0–8.6) | `bf16` or `int4` when listed |
| Below CC 8.0 | unsupported → cloud |

1. For standard `*/server`, keep `NIM_TAGS_SELECTOR` unset and let NIM choose from the compatible manifest profiles. Use `NIM_MODEL_PROFILE` when an exact LLM profile must be pinned.

`generic-assistant/server-perf` is the exception: its Compose service (`nvidia-llm-perf`) pins `precision=nvfp4,tp=2` and exposes GPUs `2` and `3` to the LLM. It targets a four-GPU Blackwell host. On older non-Blackwell hardware, run `list-model-profiles` and edit that literal to a compatible TP2 profile, such as BF16 when listed and when both GPUs have enough VRAM.

Device placement is **not** an `.env` knob. Standard `*/server` sidecars default to GPU `0`. `generic-assistant/server-perf` places ASR on `0`, TTS on `1`, tensor-parallel LLM on `2` and `3`. To move a service, edit `device_ids` under `deploy.resources.reservations.devices` in its Compose file:

- Server: `docker/docker-compose.nemotron-asr.yaml`, `docker/docker-compose.magpie-tts.yaml`, `docker/docker-compose.magpie-zeroshot-tts.yaml`, `docker/docker-compose.chatterbox-tts.yaml`, `docker/docker-compose.nemotron35-lightning-nim.yaml`, `docker/docker-compose.nemotron3-omni-nim.yaml`
- A `tp=N` LLM needs N free GPUs. List every index it uses and keep those GPUs free of ASR and TTS. Each target index must appear in the hardware readout. One GPU: keep everything on `0`, or use cloud.

Apply only what the hardware readout indicates. Never silently change values.

NIM model-profile docs: https://docs.nvidia.com/nim/large-language-models/latest/deployment/model-profiles-and-selection.html

Cascaded Lightning NIM health: `curl -f http://localhost:18000/v1/health/ready`. Omni NIM health: `curl -f http://localhost:18002/v1/health/ready`.

## Failures

- **`pull access denied` / `unauthorized`** → NGC login missing or expired. Single-GPU does not use `nvcr.io`.
- **`Could not match a profile in manifest`** → no profile matches the detected hardware or the pinned selector. Run `list-model-profiles`; for standard `*/server`, leave automatic selection enabled or pin a compatible `NIM_MODEL_PROFILE`. For `server-perf`, update its literal TP2 selector to a compatible precision, then recreate the LLM service.
- **`No available memory for the cache blocks`** → `NIM_KVCACHE_PERCENT` is too **low**. Raise it. CUDA OOM during load is the opposite: lower it, or move TTS off that GPU. `LLM_MAX_NUM_SEQS` can be lowered if CUDA-graph capture fails.
