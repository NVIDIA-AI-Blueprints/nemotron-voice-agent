# Cisco Webex BYOVA WebSocket adapter

This co-located sidecar exposes Cisco's published BYOVA-over-WebSocket JSON
contract and bridges each call to a dedicated Nemotron RTVI protobuf WebSocket
at `/api/ws`. It implements schema
`a38a10b7-43e4-4676-a076-a7d6dce9387d`; it is independent of the older gRPC
adapter.

## Endpoints

- `GET /v1/va`: one authenticated persistent WebSocket per conversation. It
  carries `SESSION_START`, all media and event turns, heartbeat traffic, and
  `SESSION_END`. Only JSON text frames are accepted. The envelope and payload
  follow Cisco's official `VoiceVirtualAgent_WsSchema.json`.
- `GET /v1/listVirtualAgents`: authenticated, short-lived WebSocket discovery.
  Its request and response are unwrapped, as required by
  `VoiceVirtualAgent_WsListVASchema.json`.
- `GET /health`: local/operator process health only. This is not part of either
  Cisco AsyncAPI schema and should normally remain behind the operator ingress.

The 2026-09-09 production qualification observed Cisco close the socket after
the welcome `FINAL` and open continuation sockets. The adapter retains the
logical conversation for a bounded grace period and attaches those sockets by
`conversation_id`, preserving the Nemotron session while resetting
connection-scoped sequence counters. See the example README for qualification
status.

The adapter converts raw Base64 G.711 mu-law 8 kHz mono caller media into
PCM16 16 kHz for Nemotron. Return media is converted back to raw mu-law and
normally sent in Cisco's recommended 8-KiB chunks, with a legal 320-byte
minimum for the final short tail. Conversion uses each pipeline frame's
declared sample rate. An adapter-owned output transport omits browser playback
pacing because Webex buffers and paces partner audio. Prompts always enable
caller barge-in.

RTVI events map as follows:

- `user-started-speaking` → `START_OF_INPUT`
- `user-turn-finalized` → `END_OF_INPUT`
- output audio → `CHUNK`
- output serialized and generation stopped → exactly one `FINAL`

The reply lifecycle is isolated in `completion.py`; it cannot infer completion
from arbitrary timing or caller-media arrival.

## Production configuration

Python 3.12 is required. TLS and bearer-JWS validation are enabled by safe
startup defaults:

```bash
export BYOVA_TLS_CERT=/run/secrets/tls.crt
export BYOVA_TLS_KEY=/run/secrets/tls.key
export CISCO_JWT_ISSUERS=https://idbroker-b-us.webex.com/idb
export CISCO_JWT_AUDIENCE='<registered audience>'
export CISCO_JWT_SUBJECT='<registered subject>'
export CISCO_JWT_NONCE='<registered nonce/jti>'
export CISCO_DATASOURCE_URL='wss://voice.example.com'
export CISCO_DATASOURCE_SCHEMA_UUID=a38a10b7-43e4-4676-a076-a7d6dce9387d
export CISCO_ORG_UUID='<customer Webex org UUID>'
export NEMOTRON_VOICE_AGENT_WS='wss://nemotron:7860/api/ws'
uv run cisco-webex-byova-ws-adapter
```

Validation occurs before WebSocket upgrade. It pins RS256, selects the
allowlisted issuer's JWK by `kid`, verifies `exp`, `iat`, audience and subject,
and binds `jti`, datasource URL, schema UUID, and `com.cisco.org.uuid`.

Plaintext and disabled authentication are local-test modes only and each needs
an explicit switch:

```bash
BYOVA_ALLOW_PLAINTEXT_LOCAL=true \
BYOVA_AUTH_ENABLED=false \
BYOVA_ALLOW_AUTH_DISABLED_LOCAL=true \
NEMOTRON_VOICE_AGENT_WS=ws://127.0.0.1:7860/api/ws \
./run.sh
```

Other bounds are configurable with `BYOVA_QUEUE_SIZE`,
`BYOVA_MAX_MESSAGE_BYTES`, `BYOVA_SEND_TIMEOUT_SECONDS`,
`BYOVA_FIRST_AUDIO_TIMEOUT_SECONDS`, and
`BYOVA_COMPLETION_TIMEOUT_SECONDS`.

## Validation

```bash
uv sync --dev
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```
