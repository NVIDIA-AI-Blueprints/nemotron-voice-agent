# Omni Assistant Cascaded Example — Deployment Reference

Use this reference from the `deploy` skill when deploying the examples/omni_assistant example — Nemotron 3 Nano Omni handles ASR and LLM in a single multimodal chat-completions call, with Magpie TTS for the spoken reply.

## When to use

Pinning a Docker Compose deployment to the Omni Assistant example. Recipe profile names are `<example>` for cloud-only and `<example>/<hardware>` for on-prem. The companion `omni-assistant-subagents` example is a separate recipe (see its deploy reference). Selector modes (`all`, or a single `<example>`) are host-native only — they are not exposed as compose profiles.

Per-example catalogs at `src/examples/omni_assistant/services.{cloud,local}.yaml` are auto-selected on container startup because the registry resolves the example for the active recipe.

Hardware support: cloud-only, `server` (NIM TTS, recommended for scaling), and universal `single-gpu`. The single-GPU recipe covers workstations, DGX Spark, and Jetson Thor; it reuses the same Omni vLLM sidecar with TTS served by NeMo-Speech.cpp instead of the Magpie NIM. Orin-class Jetson hardware is still unsupported because the model does not fit.

## Compose deploy

```bash
# Cloud (NVCF)
docker compose --profile omni-assistant up -d

# Server (local Omni vLLM + NIM TTS, recommended for scaling)
docker compose --profile omni-assistant/server up -d
docker compose --profile omni-assistant/single-gpu up -d

# Single GPU, including Jetson Thor (local Omni vLLM + on-device NeMo-Speech.cpp TTS)
docker compose --profile omni-assistant/single-gpu up -d
```

| Recipe profile | App service | Sidecars from `docker/` |
| --- | --- | --- |
| `omni-assistant` | `omni-assistant` | none (cloud NVCF) |
| `omni-assistant/server` | `omni-assistant-server` | `nvidia-llm-vllm-omni`, `tts-service` |
| `omni-assistant/single-gpu` | `omni-assistant-single-gpu` | `nvidia-llm-vllm-omni`, `nemo-speech-tts` |

Tear down with the same recipe used at `up` time.

## Verify

- UI at `https://<host>:7860/` by default, or `http://<host>:7860/` when `PIPELINE_TLS=false`.
- App logs: `docker compose logs --tail 200 omni-assistant-server`.
- Omni vLLM logs (local recipes only): `docker compose logs --tail 200 nvidia-llm-vllm-omni`.
- Omni vLLM health: `curl -f http://localhost:8002/health` from the host or `curl -f http://nvidia-llm-vllm-omni:8002/health` from inside the compose network.

## GPU memory & device placement

Omni runs the LLM in `nvidia-llm-vllm-omni`; ASR is handled inside the Omni model, so there is no separate ASR NIM. The `server` recipe uses `tts-service` for TTS, while the universal single-GPU recipe uses `nemo-speech-tts`.

For the VRAM, `--gpu-memory-utilization`, and device-placement matrix, see [VRAM & hardware support](../../../docs/how-to/configure-llm.md#vram--hardware-support).

## Common failures

- **`pull access denied` / `unauthorized`** -> NGC login was not done or expired. See the root `deploy` skill.
- **Omni vLLM stuck on first-run model download** -> initial download of `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` from Hugging Face requires `HF_TOKEN` in `.env`. Allow up to 30 minutes on first start.
- **`No available memory for the cache blocks` on startup** -> `--gpu-memory-utilization` is too **low** for this GPU, leaving no room for the KV cache after the weights. Raise it and give the LLM a dedicated GPU. Do not lower it.
- **True out-of-memory (CUDA OOM) during model load** -> the fraction collides with another process on the same GPU. Lower `--gpu-memory-utilization` or `--max-model-len`, or move `tts-service` to a separate GPU.
- **Tear-down leaves orphan services after a service rename** -> rerun `up` or `down` with `--remove-orphans`.
