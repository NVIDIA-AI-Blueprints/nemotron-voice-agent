---
name: deploy
description: Deploy Nemotron Voice Agent via root compose using recipe profiles. Use when deploying or troubleshooting auth/startup.
version: "2.2.0"
metadata:
  author: NVIDIA Voice Agent Team <nemotron-voice-agent@nvidia.com>
  tags: [deployment, docker-compose, voice-agent, nemotron]
---

# Nemotron Voice Agent Deployment

`SKILL.md` is the decision tree. Open a reference only for the family you chose. Do not open the others.

## Rules

- Run commands from the repository root that contains `docker-compose.yml`. Use Docker Compose.
- Specify **exactly one recipe**. Each profile is a complete recipe. `docker compose up` with no profile is a no-op. Never combine two recipes. `tracing` and `turn` compose orthogonally with any recipe. They are not recipes.
- Preserve existing `.env`. Create it only if missing: `test -f .env || cp .env.example .env`.
- Use `configure-pipeline` for `.env`, catalog, or prompt changes. Do not write host-specific vLLM flags into `.env`.
- Recipe names: `<example>` = cloud NVCF, `<example>/server` = local NIM on a **workstation** (not DGX Spark, not Jetson Thor), `<example>/single-gpu` = local vLLM + NeMo-Speech.cpp. `generic-assistant/server-perf` is a 4-GPU workstation load benchmark, not a UI deploy. See `benchmarking_tools/scaling-perf/README.md`.
- Selector modes (`all`, or one `<example>`) are host-native (`uv run`) only. No compose profile.
- Generic, Multilingual, Omni, and Frontend/Backend `/single-gpu` support compatible workstations, DGX Spark, and Jetson Thor. `omni-assistant-subagents/single-gpu` is **not supported on Jetson Thor** (workstation and DGX Spark only). On Thor, that example is cloud-only (`omni-assistant-subagents`). Orin-class Jetson is unsupported (the model does not fit). Do not infer fit from the platform name. Complete memory-fit first. The single-GPU compose files detect product and compute capability. Do not set those by hand.

## Auth (do not mix)

| Recipe family | `.env` key | `docker login nvcr.io` |
| --- | --- | --- |
| Cloud (`<example>`) | `NVIDIA_API_KEY` (NVCF) | **No** |
| Server (`<example>/server`, including `server-perf`) | `NVIDIA_API_KEY` (NGC + NIM) | **Yes**, before `up` |
| Single-GPU (`<example>/single-gpu`) | `HF_TOKEN` (Hugging Face download speed / rate limits) | **No**. Do not use `NVIDIA_API_KEY`. |

`HF_TOKEN` is not a substitute for `NVIDIA_API_KEY`. The reverse is also false. `TURN_USERNAME` and `TURN_PASSWORD` are required only with `--profile turn`.

Server NGC login:

```bash
set -a; . ./.env; set +a
printf '%s' "$NVIDIA_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin
```

## Deploy

1. Hardware:

```bash
cat /sys/class/dmi/id/product_name 2>/dev/null || true
cat /proc/device-tree/model 2>/dev/null || true
nvidia-smi --query-gpu=index,name,memory.total,memory.free,compute_cap --format=csv,noheader
free -h
```

2. Choose **one** recipe from `references/recipes.md`. Hardware only tells you what *can* run:

- Cloud: needs `NVIDIA_API_KEY`. It is the fallback when local VRAM is not enough.
- `server`: workstation NIM only. **Not supported on DGX Spark or Jetson Thor.** If the hardware readout is Spark or Thor, do not open `references/server.md`. Use `single-gpu` or cloud. On a workstation: NGC login + `references/server.md`.
- `single-gpu`: one GPU (workstation, DGX Spark, Jetson Thor). Needs `HF_TOKEN` + `references/single-gpu.md`. Exception: `omni-assistant-subagents/single-gpu` is **not supported on Jetson Thor**. Use cloud.

If more than one family is viable, present the options and let the user pick. Auto-select only when a single option is viable (for example no usable GPU → cloud).

3. Apply **only** that family's preflight. Never run the other family's login or key checks.

4. Start:

```bash
docker compose --profile <recipe> up -d
```

Add `--profile tracing` (Phoenix) freely. Add `--profile turn` only after `references/turn.md`. Use `--build` only after source or `Dockerfile` changes.

Local recipes: the first voice turn may be slow while GPU sidecars finish loading. If later turns are fast, the deploy is fine.

5. Verify:

```bash
docker compose ps
docker compose logs --tail 200 <app-or-sidecar>
```

UI is `https://<host>:7860/` by default, or `http://<host>:7860/` when `PIPELINE_TLS=false`.

Tear down with the same recipe: `docker compose --profile <recipe> down`. After a service rename, add `--remove-orphans`.

## References

Open only what the chosen family needs:

- Recipe table and sidecar names: `references/recipes.md`
- Server NIM precision: `references/server.md`
- Single-GPU memory-fit: `references/single-gpu.md`
- External clients: `references/turn.md`
