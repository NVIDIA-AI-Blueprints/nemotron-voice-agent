# Omni Assistant Cascaded Example — Deployment Reference

Use this reference from the `deploy` skill when deploying the examples/omni_assistant example — Nemotron 3 Nano Omni handles ASR and LLM in a single multimodal chat-completions call, with Magpie TTS for the spoken reply.

## When to use

Pinning a Docker Compose deployment to the Omni Assistant example. Use `omni-assistant` for cloud, `omni-assistant/server` for Omni NIM and NIM TTS, or `omni-assistant/single-gpu` for Omni vLLM and NeMo-Speech.cpp TTS. The companion `omni-assistant-subagents` example is a separate recipe. Selector modes are host-native only and are not exposed as Compose profiles.

Per-example catalogs at `src/examples/omni_assistant/services.{cloud,local}.yaml` are auto-selected on container startup because the registry resolves the example for the active recipe.

Hardware support: cloud-only, `server` (Omni NIM and NIM TTS, recommended for scaling), and universal `single-gpu`. The single-GPU recipe covers workstations, DGX Spark, and Jetson Thor. It uses Omni vLLM with TTS served by NeMo-Speech.cpp instead of the NIM sidecars. Orin-class Jetson hardware is still unsupported because the model does not fit.

## Compose deploy

```bash
# Cloud (NVCF)
docker compose --profile omni-assistant up -d

# Server (Omni NIM + NIM TTS, recommended for scaling)
docker compose --profile omni-assistant/server up -d

# Single GPU, including Jetson Thor (local Omni vLLM + on-device NeMo-Speech.cpp TTS)
docker compose --profile omni-assistant/single-gpu up -d
```

| Recipe profile | App service | Sidecars from `docker/` |
| --- | --- | --- |
| `omni-assistant` | `omni-assistant` | none (cloud NVCF) |
| `omni-assistant/server` | `omni-assistant-server` | `nvidia-llm-omni`, `magpie-multilingual-tts-service` |
| `omni-assistant/single-gpu` | `omni-assistant-single-gpu` | `nvidia-llm-vllm-omni`, `nemo-speech-tts` |

Tear down with the same recipe used at `up` time.

## Verify

- UI at `https://<host>:7860/` by default, or `http://<host>:7860/` when `PIPELINE_TLS=false`.
- Cloud app logs: `docker compose logs --tail 200 omni-assistant`.
- Server app and NIM logs: `docker compose logs --tail 200 omni-assistant-server nvidia-llm-omni magpie-multilingual-tts-service`.
- Single-GPU app and sidecar logs: `docker compose logs --tail 200 omni-assistant-single-gpu nvidia-llm-vllm-omni nemo-speech-tts`.
- Server NIM health: `curl -f http://localhost:18002/v1/health/ready`.
- Single-GPU vLLM health: `curl -f http://localhost:8002/health`.

## GPU memory & device placement

Omni handles ASR inside the model, so there is no separate ASR NIM. The `server` recipe uses `nvidia-llm-omni` and `magpie-multilingual-tts-service`. The single-GPU recipe uses `nvidia-llm-vllm-omni` and `nemo-speech-tts`.

For the VRAM, `--gpu-memory-utilization`, and device-placement matrix, see [VRAM & hardware support](../../../docs/how-to/configure-llm.md#vram--hardware-support).

## Common failures

- **`pull access denied` / `unauthorized`** -> NGC login was not done or expired. See the root `deploy` skill.
- **Omni vLLM stuck on first-run model download** -> initial download of `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` from Hugging Face requires `HF_TOKEN` in `.env`. Allow up to 30 minutes on first start.
- **`No available memory for the cache blocks` on startup** -> `--gpu-memory-utilization` is too **low** for this GPU, leaving no room for the KV cache after the weights. Raise it and give the LLM a dedicated GPU. Do not lower it.
- **True out-of-memory (CUDA OOM) during model load** -> the fraction collides with another process on the same GPU. Lower `--gpu-memory-utilization` or `--max-model-len`, or move `magpie-multilingual-tts-service` to a separate GPU.
- **Tear-down leaves orphan services after a service rename** -> rerun `up` or `down` with `--remove-orphans`.
