# Apply Configuration Changes

Use this reference after editing `.env`, `examples_registry.yaml`, an example-local `prompts.yaml`, or an example-local `services.cloud.yaml` / `services.local.yaml`.

## Default Rule

- `.env` changes: compose re-apply (`up -d` with the same profile combination).
- YAML changes (`examples_registry.yaml`, `prompts.yaml`, `services.*.yaml`): compose restart of the example service and refresh browser. `./src` and `./examples_registry.yaml` are bind-mounted, so no rebuild needed.

## Endpoint Rules

The catalog stores Compose DNS endpoints. The backend rewrites them to `localhost` automatically when running outside Docker (`uv run`). Local entries are filtered by TCP reachability and only show in the UI when the corresponding sidecar is up.

| Compose endpoint | Host-run rewrite |
| --- | --- |
| `http://nvidia-llm:8000/v1` | `http://localhost:18000/v1` |
| `http://nvidia-llm-vllm:8000/v1` | `http://localhost:18000/v1` |
| `http://nvidia-llm-omni:8000/v1` | `http://localhost:18002/v1` |
| `http://nvidia-llm-vllm-omni:8002/v1` | `http://localhost:8002/v1` |
| `http://booking-server:8001` | `http://localhost:8001` |
| `magpie-multilingual-tts-service:50051` | `localhost:50151` |
| `magpie-zeroshot-tts-service:50051` | `localhost:50151` |
| `chatterbox-tts-service:50051` | `localhost:50151` |
| `nemotron-asr-streaming-english:50052` | `localhost:50152` |
| `nemotron-asr-streaming-multilingual:50052` | `localhost:50152` |
| `parakeet-ctc-asr:50052` | `localhost:50152` |
| `parakeet-rnnt-asr:50052` | `localhost:50152` |
| `nemo-speech:50051` | `localhost:50051` |
| `nemo-speech-multilingual:50051` | `localhost:50051` |
| `nemo-speech-tts:50051` | `localhost:50051` |

Use the exact Compose service name for each local endpoint. Model-specific TTS and ASR names let Compose define alternatives that share host ports.

Cloud catalog entries use NVCF endpoints (`grpc.nvcf.nvidia.com:443`, `https://integrate.api.nvidia.com/v1`, `wss://grpc.nvcf.nvidia.com/v1/realtime`) and are not rewritten.

## Apply Commands

Pick one recipe profile. Use `<example>` for cloud, `<example>/server` for NIM deployments, or `<example>/single-gpu` for one-GPU deployments. Each recipe is a complete deployment. Never combine two recipes.

```bash
# Cloud-only (NVCF)
docker compose --profile generic-assistant up -d
docker compose --profile multilingual-assistant up -d
docker compose --profile omni-assistant up -d
docker compose --profile omni-assistant-subagents up -d
docker compose --profile frontend-backend-agent up -d

# Server (local NIM ASR/TTS/LLM, recommended for scaling)
docker compose --profile generic-assistant/server up -d
docker compose --profile multilingual-assistant/server up -d
docker compose --profile omni-assistant/server up -d
docker compose --profile omni-assistant-subagents/server up -d
docker compose --profile frontend-backend-agent/server up -d

# Multi-GPU performance benchmark only
docker compose --profile generic-assistant/server-perf up -d

# Universal one-GPU path (workstation, DGX Spark, or Jetson Thor)
docker compose --profile generic-assistant/single-gpu up -d
docker compose --profile multilingual-assistant/single-gpu up -d
docker compose --profile omni-assistant/single-gpu up -d
docker compose --profile frontend-backend-agent/single-gpu up -d

# Resource-heavy single-GPU path (workstation or DGX Spark)
docker compose --profile omni-assistant-subagents/single-gpu up -d
```

For YAML-only edits that do not change environment variables or sidecar membership, `docker compose restart <service>` is enough. For example, run `docker compose restart generic-assistant`.

## Optional Profile Overlays

Tracing and TURN compose orthogonally with any recipe. Re-apply must include them again to keep those services running.

### Tracing (`--profile tracing`)

Add when:
- `.env` has `ENABLE_TRACING=true`
- `OTEL_EXPORTER_OTLP_ENDPOINT` points to `phoenix:4317` or another in-repo Phoenix endpoint

### Remote WebRTC (`--profile turn`)

Add when clients connect from outside the host's network. Set `TURN_USERNAME` and `TURN_PASSWORD` in `.env`; the app only publishes ICE config when both values are present. Set `TURN_URL=turn:<host>:3478` if TURN runs on a different host or the request host is not client-reachable. The client auto-fetches ICE config from `/api/ice-servers`.

```bash
docker compose --profile generic-assistant --profile tracing up -d
docker compose --profile generic-assistant/server --profile turn up -d
docker compose --profile generic-assistant/single-gpu --profile tracing --profile turn up -d
```

## Validation Checklist

- The selected recipe profile matches the example and hardware you want active.
- `examples_registry.yaml` `defaults` references catalog keys that actually exist for that example.
- Multilingual prompt selection is paired with multilingual-capable ASR (`parakeet-rnnt` by default, or `nemotron-asr-streaming-multilingual` when opted in) and TTS (`magpie-multilingual-tts`, `magpie-zeroshot-tts`, or `chatterbox-multilingual-tts`) in the active catalog.
- If `ENABLE_TRACING=true` with `phoenix:4317`, the `phoenix` service is started through the `tracing` profile.
- Compose-managed local entries use service DNS names, not `localhost`.
- Local catalog endpoints must match the exact Compose service name. Do not replace model-specific names with generic role names.

## Verify

```bash
docker compose ps
docker compose logs --tail 200 <service-name>
```

Refresh open browser tabs after the backend is healthy. The client caches deployment metadata, built-in services, prompts, and ICE config for the page lifetime.

Verify behavior relevant to the change:
- New built-in services or prompts appear in the UI after refresh.
- Local services only appear when their containers are reachable.
- Tracing data appears in Phoenix when tracing is enabled.
