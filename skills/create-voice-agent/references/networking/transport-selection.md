# Transport selection | Pipecat WebRTC / WebSocket / Both

Gate row 3 choices. Pipecat runner only (LiveKit→frameworks/livekit.md).

## Gate choices
| choice | scaffold | run | handoff |
| --- | --- | --- | --- |
| **WebRTC** | `transport_params` webrtc only; pyproject `[webrtc]`; remote SSH→`turn_support.py`+coturn | `uv run python bot.py --host HOST -t webrtc` | `:PORT/client`; POST `/start` `{"transport":"webrtc"}` |
| **WebSocket** | `transport_params` websocket only; pyproject `[websocket]` | `uv run python bot.py --host HOST -t websocket` | `ws://HOST:PORT/ws-client`; POST `/start` `{"transport":"websocket"}` |
| **Both** (recommended when unsure) | `transport_params` **webrtc + websocket**; pyproject `[webrtc,websocket]`; remote SSH→turn for WebRTC leg | `uv run python bot.py --host HOST` **omit `-t`** | `:PORT/client` switch transport in UI; `/status` lists both; deploy either |

FORBID infer transport at scaffold. FORBID `-t webrtc` when user chose Both (blocks WebSocket in `/start`).

## `transport_params` patterns

WebRTC only:
```python
transport_params = {
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
}
```

WebSocket only:
```python
transport_params = {
    "websocket": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
}
```

Both (REQ both keys):
```python
transport_params = {
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
    "websocket": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
}
```

`create_transport(runner_args, transport_params)` unchanged — runner picks key from CLI `-t` or `/start` body.

## pyproject extras
| transport | dependencies fragment |
| --- | --- |
| WebRTC | `pipecat-ai[nvidia,runner,webrtc,silero]>=1.3.0` |
| WebSocket | `pipecat-ai[nvidia,runner,websocket,silero]>=1.3.0` |
| Both | `pipecat-ai[nvidia,runner,webrtc,websocket,silero]>=1.3.0` |

## Pipecat UI / `/start` switching (Both only)
- Omit `-t` on `main()` so runner registers webrtc **and** websocket routes; `GET /status` → `{"transports":["webrtc","websocket",...]}`.
- Client at `/client` can start with `transport: "webrtc"` or `transport: "websocket"` in POST `/start`.
- Same `bot.py` pipeline — no duplicate bot files.

## When to recommend
- **Both (recommended)** — default MCQ first option; Pipecat playground flexibility; corporate UDP issues can fall back to WebSocket without rescaffold.
- **WebRTC** — user explicitly wants playground only, LAN, or SSH+TURN; UDP OK.
- **WebSocket** — user explicitly says UDP/firewall blocked; WS-only deployment.

## Verify
```bash
curl -s http://127.0.0.1:7860/status | python3 -m json.tool
# Both: transports includes webrtc and websocket

# WebRTC
curl -s -X POST http://127.0.0.1:7860/start -H 'Content-Type: application/json' \
  -d '{"transport":"webrtc","enableDefaultIceServers":true}' | python3 -m json.tool

# WebSocket (Both or -t websocket)
curl -s -X POST http://127.0.0.1:7860/start -H 'Content-Type: application/json' \
  -d '{"transport":"websocket"}' | python3 -m json.tool
```

WebRTC remote checklist→first-try-workstation-webrtc.md. UDP blocked→WebSocket or Both.
