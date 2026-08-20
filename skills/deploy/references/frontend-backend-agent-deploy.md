# Frontend/Backend Agent Deploy Reference

Use this reference from the `deploy` skill when deploying the cascaded Frontend/Backend Agent airline assistant.

## Profiles

Pin Docker Compose to one Frontend/Backend Agent recipe. The cloud recipe uses NVIDIA cloud services, the `/server` recipe runs local NIMs (recommended for scaling), and the universal single-GPU recipe runs Lightning with NeMo-Speech.cpp. Every recipe includes the booking-server sidecar.

```bash
docker compose --profile frontend-backend-agent up -d
docker compose --profile frontend-backend-agent/server up -d
docker compose --profile frontend-backend-agent/single-gpu up -d
```

| Recipe profile | App service | Sidecars |
| --- | --- | --- |
| `frontend-backend-agent` | `frontend-backend-agent` | `booking-server` |
| `frontend-backend-agent/server` | `frontend-backend-agent-server` | `booking-server`, `nvidia-llm`, `nemotron-asr-streaming-english`, `tts-service` |
| `frontend-backend-agent/single-gpu` | `frontend-backend-agent-single-gpu` | `booking-server`, `nvidia-llm-vllm-lightning`, `nemo-speech` |

Tear down with the same recipe used at `up` time.

```bash
docker compose --profile <recipe> down
```

## Verify

- Cloud app logs: `docker compose logs --tail 200 frontend-backend-agent`.
- Server app logs: `docker compose logs --tail 200 frontend-backend-agent-server`.
- Single-GPU app logs: `docker compose logs --tail 200 frontend-backend-agent-single-gpu`.
- Booking server logs: `docker compose logs --tail 200 booking-server`.
- Server local service logs: `docker compose logs --tail 200 nvidia-llm nemotron-asr-streaming-english tts-service`.

## Limits

Frontend/Backend Agent supports cloud-only, `server`, and universal single-GPU recipes. Use `frontend-backend-agent/single-gpu` on DGX Spark and Jetson Thor.
