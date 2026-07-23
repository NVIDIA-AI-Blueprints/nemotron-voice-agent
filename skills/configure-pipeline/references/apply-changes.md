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
| `tts-service:50051` | `localhost:50151` |
| `chatterbox-tts-service:50051` | `localhost:50151` |
| `nemotron-asr-streaming-english:50052` | `localhost:50152` |
| `nemotron-asr-streaming-multilingual:50052` | `localhost:50152` |
| `parakeet-ctc-asr:50052` | `localhost:50152` |
| `parakeet-rnnt-asr:50052` | `localhost:50152` |
| `nemotron-speech:50051` | `localhost:50051` |

Cloud catalog entries use NVCF endpoints (`grpc.nvcf.nvidia.com:443`, `https://integrate.api.nvidia.com/v1`, `wss://grpc.nvcf.nvidia.com/v1/realtime`) and are not rewritten.

## Apply Commands

Pick a single recipe profile (`<example>` for cloud or `<example>/<hardware>` for on-prem). Each recipe is a complete deployment — never combine two recipes.

```bash
# Cloud-only (NVCF)
docker compose --profile generic-assistant up -d
docker compose --profile multilingual-assistant up -d
docker compose --profile omni-assistant up -d
docker compose --profile frontend-backend-agent up -d

# Workstation (local NIM ASR/TTS/LLM)
docker compose --profile generic-assistant/workstation up -d
docker compose --profile multilingual-assistant/workstation up -d
docker compose --profile omni-assistant/workstation up -d
docker compose --profile frontend-backend-agent/workstation up -d

# DGX Spark
docker compose --profile generic-assistant/dgx-spark up -d
docker compose --profile multilingual-assistant/dgx-spark up -d
docker compose --profile omni-assistant/dgx-spark up -d

# Jetson (Generic Cascaded only. Omni does not fit on Orin today)
docker compose --profile generic-assistant/jetson-thor up -d
```

For YAML-only edits that don't change env or sidecar membership, `docker compose restart <service>` is enough (e.g. `docker compose restart generic-assistant`).

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
docker compose --profile generic-assistant/workstation --profile turn up -d
docker compose --profile generic-assistant/dgx-spark --profile tracing --profile turn up -d
```

## Validation Checklist

- The selected recipe profile matches the example and hardware you want active.
- `examples_registry.yaml` `defaults` references catalog keys that actually exist for that example.
- Multilingual prompt selection is paired with multilingual-capable ASR (`parakeet-rnnt` by default, or `nemotron-asr-streaming-multilingual` when opted in) and TTS (`magpie-multilingual-tts` or `chatterbox-multilingual-tts`) in the active catalog.
- If `ENABLE_TRACING=true` with `phoenix:4317`, the `phoenix` service is started through the `tracing` profile.
- Compose-managed local entries use service DNS names, not `localhost`.
- ASR/TTS image variants use their Compose service DNS names (for example `nemotron-asr-streaming-english:50052`, `nemotron-asr-streaming-multilingual:50052`, `tts-service:50051`).

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
