# Run | start+connect canonical

Canonical health/start/stop. Other refs REF here for curls.

## .env secrets only
NVIDIA_API_KEY=... | LiveKit: LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET (LiveKit Cloud default — see frameworks/livekit.md). FORBID model ids/function_id in .env. Missing keys→clarifying/env-secrets.md STOP WAIT before scaffold.

## Cloud
`cp .env.example .env && uv sync && uv run python bot.py` | LiveKit: `uv run python agent.py console`

## Local workstation/DGX
ORDER: NIMs→agent. WebRTC remote→first-try-workstation-webrtc.md

**Post model-confirm:** before any start, run **per-slot probe** in `platforms/deployment.md` §Local selective slot reuse — check LLM, ASR, and TTS individually; keep slots that already match; start or recreate **only** missing or mismatched slots. Example: wrong ASR on GPU → replace `asr-service` only; leave matching LLM and TTS running.

Prereq:
```bash
nvidia-smi; docker info|grep -i nvidia; set -a&&source .env&&set +a; [ -n "$NVIDIA_API_KEY" ]
echo "$NVIDIA_API_KEY"|docker login nvcr.io -u '$oauthtoken' --password-stdin
```

Start (after per-slot probe — skip services already matching):
```bash
# one slot only — example ASR mismatch:
docker compose -f docker-compose.nim.workstation.yml stop asr-service
docker compose -f docker-compose.nim.workstation.yml rm -f asr-service
docker compose -f docker-compose.nim.workstation.yml --profile full up -d asr-service
# all three slots missing — start separately (profile full only for speech services):
docker compose -f docker-compose.nim.workstation.yml up -d nvidia-llm
docker compose -f docker-compose.nim.workstation.yml --profile full up -d tts-service
docker compose -f docker-compose.nim.workstation.yml --profile full up -d asr-service
# sequential helper (optional):
cp assets/start_nim_stack.sh scripts/&&chmod +x scripts/start_nim_stack.sh&&./scripts/start_nim_stack.sh
docker compose -f docker-compose.nim.omni.yml up -d  # omni; probe LLM :8002 + TTS only
```

Verify before agent:
```bash
# cascaded
curl -sf http://127.0.0.1:18000/v1/health/ready; curl -sf http://127.0.0.1:9001/v1/health/ready; curl -sf http://127.0.0.1:9000/v1/health/ready
# omni
curl -sf http://127.0.0.1:8002/health; curl -sf http://127.0.0.1:9000/v1/health/ready; curl -sf http://127.0.0.1:8002/v1/models
```

Agent (per gate transport — see networking/transport-selection.md):
```bash
uv sync
# WebRTC
uv run python bot.py --host VM_IP -t webrtc  # LAN
uv run python bot.py --host 127.0.0.1 -t webrtc  # SSH tunnel
# WebSocket
uv run python bot.py --host 127.0.0.1 -t websocket
# Both — omit -t; switch in Pipecat /client UI
uv run python bot.py --host 127.0.0.1
uv run python agent.py console
```

WebRTC POST /start+TURN→first-try §Pre-handoff. Connect: localhost:7860/client or VM_IP:7860/client. Both: `curl /status` lists webrtc+websocket; UI or POST `/start` picks transport. Mac remote→ssh.md §Mac remote connect.
Stop: Ctrl+C; docker compose -f docker-compose.nim.workstation.yml down (or nim.omni.yml / nim.jetson.yml).

Jetson→jetson-thor.md ORDER riva_init→vLLM+Riva→agent.

Handoff: platform; start/stop; Mac TPL; models+endpoints from py; first-try if WebRTC; perf artifacts if requested.
