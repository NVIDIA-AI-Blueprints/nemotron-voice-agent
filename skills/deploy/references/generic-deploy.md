# Generic Cascaded Example — Deployment Reference

Use this reference from the `deploy` skill when deploying the generic voice pipeline example (NVIDIA STT, NIM LLM, NVIDIA TTS with function calling).

## When to use

Pinning a Docker Compose deployment to the Generic Cascaded example. Recipe profile names are `<example>` for cloud-only and `<example>/<hardware>` for on-prem. Selector modes (`all`, or a single `<example>`) are host-native only — they are not exposed as compose profiles.

Per-example catalogs at `src/examples/generic/services.{cloud,local}.yaml` are auto-selected on container startup because the registry resolves the example for the active recipe.

## Compose deploy

Pick one recipe profile:

```bash
docker compose --profile <recipe> up -d
```

| Recipe profile | App service | Sidecars from `docker/` |
| --- | --- | --- |
| `generic-assistant` | `generic-assistant` | none (cloud NVCF) |
| `generic-assistant/workstation` | `generic-assistant` | `nvidia-llm`, `nemotron-asr-streaming-english`, `tts-service` |
| `generic-assistant/dgx-spark` | `generic-assistant` | `nvidia-llm-vllm`, `nemotron-asr-streaming-english`, `tts-service` |
| `generic-assistant/jetson-thor` | `generic-assistant` | `nvidia-llm-vllm`, `nemotron-speech` |

Tear down with the same recipe used at `up` time:

```bash
docker compose --profile <recipe> down
```

## Verify

- UI at `https://<host>:7860/` by default, or `http://<host>:7860/` when `PIPELINE_TLS=false`.
- `docker compose ps` and `docker compose logs --tail 200 generic-assistant`.

## Local LLM NIM profiles

- List profiles before changing LLM precision or tensor parallelism:

```bash
docker run --rm --gpus all \
  -e NGC_API_KEY="$NVIDIA_API_KEY" \
  nvcr.io/nim/nvidia/nemotron-3-nano:2.0.9 \
  list-model-profiles
```

- For one GPU, use `tp=1`. Higher `tp` values require that many GPUs.
- Prefer readable tag selection over profile hashes: `NIM_TAGS_SELECTOR=precision=fp8,tp=1`.
- Match the local LLM to the GPU via `.env`: `NIM_KVCACHE_PERCENT` (VRAM budget — **raise** it on `No available memory for the cache blocks`, lower it on an OOM kill), `NIM_TAGS_SELECTOR` (weight precision and tensor-parallel size), and `LLM_MAX_NUM_SEQS` (lower it if startup fails CUDA-graph capture). On multi-GPU hosts, choose a NIM profile with matching `tp` and expose that many GPUs. See "VRAM & hardware support" in `docs/how-to/configure-llm.md`.
- More details: https://docs.nvidia.com/nim/large-language-models/latest/deployment/model-profiles-and-selection.html

## Common failures

- **`pull access denied` / `unauthorized`** -> NGC login was not done or expired. See the root `deploy` skill.
- **Jetson startup hangs or fails with kernel/CASK errors** -> `riva_init.sh` was not run, or MPS/cpuset is misconfigured. Follow `platform-deployment.md`.
- **Tear-down leaves orphan services after a service rename** -> rerun `up` or `down` with `--remove-orphans`.
