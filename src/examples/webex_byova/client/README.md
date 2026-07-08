# Cisco Webex BYOVA Adapter

A Cisco Webex Contact Center BYOVA (Bring-Your-Own-Virtual-Agent) adapter that
fronts the `nemotron-voice-agent` backend on the same host. Webex Universal
Harness connects to this adapter over gRPC+TLS; the adapter forwards caller
audio to Nemotron's WebSocket API and streams Nemotron TTS audio back to Webex
as BYOVA `Prompt.audio_content`.

> **Note:** This adapter is intended for POCs and experimentation. It is not a
> full-featured production connector; production deployments must assess and
> extend it for their security, availability, scaling, and operational needs.

```text
Caller (PSTN) -> Webex Calling -> Webex Contact Center flow
               -> VirtualAgentV2 activity
               -> gRPC+TLS -> Cisco Webex BYOVA adapter
               -> wss://127.0.0.1:7860/api/ws -> nemotron-voice-agent
```

## What is here

- `cisco_webex_byova_adapter/` - runtime adapter code
- `proto/` - source protobuf definitions
- `cisco_webex_byova_adapter/generated/` - checked-in generated protobuf Python files
- `scripts/run_external_adapter.sh` - main launcher
- `scripts/test_byova_client.py` - local smoke client

The adapter uses the checked-in `generated/` protobuf Python files at runtime.
The `proto/` directory is kept as the source-of-truth for those generated
bindings, so both are intentionally present.

## Run the adapter

From this directory:

```bash
uv sync
export NEMOTRON_BYOVA_ADAPTER_TLS_CERT=/etc/letsencrypt/live/<public-host>/fullchain.pem
export NEMOTRON_BYOVA_ADAPTER_TLS_KEY=/etc/letsencrypt/live/<public-host>/privkey.pem
./scripts/run_external_adapter.sh
```

You do not need to manually activate a virtual environment. `uv sync` prepares
the environment and the launcher uses `uv run` internally.

Default backend targets:

- `NEMOTRON_VOICE_AGENT_WS=wss://127.0.0.1:7860`

The backend itself is the `webex-byova-assistant` example in this repository.
The adapter opens `/api/ws` directly without `/api/session-config` or a
`session_id`, allowing connections to be distributed across multiple Uvicorn
workers. The backend Compose profile selects the example at startup with
`EXAMPLE_SELECTION=webex-byova-assistant`.
For the full backend + adapter + Cisco sandbox flow, use:

- [`../run.md`](../run.md)

## Cisco JWS validation

Enable Cisco JWS validation and configure the expected claims before connecting
the adapter to Cisco:

```bash
export ENABLE_CISCO_JWS_VALIDATION=true
export CISCO_JWT_ISSUER="https://idbroker-b-us.webex.com/idb"
export CISCO_JWT_AUDIENCE="NemotronVoiceAgent"
export CISCO_JWT_SUBJECT="callAudioData"
```

These are example values from one Cisco setup. The issuer may vary by Webex
region. Obtain the exact `iss`, `aud`, and optional `sub` values from the
Cisco-issued JWS claims or Cisco data-source configuration. If validation is
enabled, `CISCO_JWT_ISSUER` and `CISCO_JWT_AUDIENCE` are required;
`CISCO_JWT_SUBJECT` is optional.

For local-only smoke testing without Cisco TLS, you can explicitly allow
plaintext:

```bash
ALLOW_PLAINTEXT_ADAPTER=true ./scripts/run_external_adapter.sh
```

## Actual behavior

- Exposes Cisco `VoiceVirtualAgent` on `0.0.0.0:50061`
- Exposes health on `:8081/voiceva/v1/ping`
- Keeps one persistent Nemotron WebSocket session per call
- Sends `client-ready` on `SESSION_START`, so the bot can greet first
- Converts caller audio from 8 kHz G.711 to 16 kHz PCM for Nemotron
- Sends caller audio to Nemotron in unpaced chunks of at most 32 ms
- Converts bot audio from 16 kHz PCM to 8 kHz G.711 for Webex
- Does not apply adapter-side VAD gating to caller audio
- Supports keyword-triggered `TRANSFER_TO_AGENT` and `SESSION_END` when
  transcript text matches the configured keywords
- Leaves Cisco JWS validation off by default unless
  `ENABLE_CISCO_JWS_VALIDATION=true`
- **WARNING:** In production you must enable Cisco JWS validation by setting
  `ENABLE_CISCO_JWS_VALIDATION=true`

## Smoke test

Use your own WAV file and point the client at the adapter:

```bash
uv run python scripts/test_byova_client.py \
  --grpc-target <host>:50061 \
  --tls \
  --audio-file /path/to/audio.wav \
  --response-wait-secs 25
```
