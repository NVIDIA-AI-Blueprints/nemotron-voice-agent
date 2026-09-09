# Cisco Webex BYOVA WebSocket Assistant

This example connects the Generic cascaded voice pipeline to Cisco Webex
Contact Center Bring Your Own Virtual Agent (BYOVA) over WebSocket. A separate
adapter translates Cisco JSON events and G.711 mu-law audio to the internal
Pipecat RTVI WebSocket protocol.

This example does not modify or replace the existing
[gRPC Webex BYOVA example](../webex_byova/README.md). The pipeline, prompts,
tools, and service catalogs intentionally match the
[Generic Assistant](../generic/README.md). The sole pipeline transport
deviation disables browser playback pacing because Webex buffers and paces
partner audio itself.

## Connection Boundaries

Keep the external Cisco endpoint separate from the internal pipeline endpoint:

- Cisco connects to the adapter at
  `wss://<public-adapter-host>:<BYOVA_WS_PUBLISH_PORT>/v1/va`.
- Cisco retrieves the virtual-agent catalog at
  `wss://<public-adapter-host>:<BYOVA_WS_PUBLISH_PORT>/v1/listVirtualAgents`.
- The adapter connects internally to
  `wss://<pipeline-host>:7860/api/ws`, configured with
  `NEMOTRON_VOICE_AGENT_WS`.

Compose publishes adapter port `18086` by default and connects internally to
`wss://webex-byova-ws-pipeline:7860/api/ws`. The container listens on port
`8086`. A host-native adapter also defaults to port `8086` and
`wss://127.0.0.1:7860/api/ws`. Cisco must reach the public adapter hostname and
published port. It does not connect directly to port `7860` or `/api/ws`.

## Deploy the Pipeline

Run commands from the repository root. Preserve an existing `.env`; otherwise,
copy the template:

```bash
test -f .env || cp .env.example .env
```

Select one pipeline profile:

```bash
docker compose --profile webex-byova-ws-assistant up -d
docker compose --profile webex-byova-ws-assistant/server up -d

# Download the speech weights once, as your user, before this profile:
bash scripts/download-nemo-speech-models.sh
docker compose --profile webex-byova-ws-assistant/single-gpu up -d
```

The cloud and server profiles require `NVIDIA_API_KEY`. The server profile also
requires an NGC registry login. The single-GPU profile requires `HF_TOKEN`
only. Refer to the [Getting Started guide](../../../docs/01-getting-started.md)
for credential and hardware prerequisites.

Compose pins `EXAMPLE_SELECTION=webex-byova-ws-assistant` and
`TRANSPORT_SELECTION=websocket`. The app service listens on pipeline port
`7860`; `PIPELINE_APP_PORT` changes only its published host port. Each profile
also starts `webex-byova-ws-adapter`. Local defaults bind adapter port `18086`
to host loopback with plaintext and authentication disabled. Configure the
production settings below before exposing that port to Cisco.

## Run the Adapter Without Compose

For host-native development, install the adapter environment from the
repository root with Python 3.12:

```bash
uv sync --project src/examples/webex_byova_ws/client
```

Add the adapter settings described below to `.env`. Then export that file and
start the listener:

```bash
set -a; . ./.env; set +a
uv run --project src/examples/webex_byova_ws/client \
  cisco-webex-byova-ws-adapter
```

The adapter reads its process environment; it does not load the repository
`.env` file itself. The default internal URL targets the pipeline's published
host port. If you change `PIPELINE_APP_PORT`, update
`NEMOTRON_VOICE_AGENT_WS` to match.

## Configure Cisco TLS and Authentication

For a Cisco-facing deployment, provide a trusted certificate and private key,
and keep authentication enabled. Configure these adapter variables:

| Variable | Purpose |
| --- | --- |
| `BYOVA_WS_HOST` | Adapter bind address. Compose sets `0.0.0.0`. |
| `BYOVA_WS_PUBLISH_ADDRESS` | Compose host bind address. Set `0.0.0.0` for a directly exposed production listener. Default: `127.0.0.1`. |
| `BYOVA_WS_PUBLISH_PORT` | Compose host port exposed to Cisco. Default: `18086`. |
| `BYOVA_WS_PORT` | Host-native listener port. The Compose container always uses `8086`. |
| `BYOVA_TLS_CERT_HOST` | Absolute host path to the Compose-mounted TLS certificate. |
| `BYOVA_TLS_KEY_HOST` | Absolute host path to the Compose-mounted private key. |
| `BYOVA_TLS_CERT` | Certificate path inside the adapter. Use `/run/secrets/byova-tls.crt` with Compose. |
| `BYOVA_TLS_KEY` | Private-key path inside the adapter. Use `/run/secrets/byova-tls.key` with Compose. |
| `BYOVA_AUTH_ENABLED` | Enables Cisco bearer JWS validation. Default: `true`. |
| `CISCO_JWT_ISSUERS` | Comma-separated allowlist of Cisco token issuers. |
| `CISCO_JWT_AUDIENCE` | Required JWS `aud` value. |
| `CISCO_JWT_SUBJECT` | Required JWS `sub` value. |
| `CISCO_JWT_NONCE` | Required JWS `jti` value supplied during Cisco setup. |
| `CISCO_DATASOURCE_URL` | Required `com.cisco.datasource.url` claim value. |
| `CISCO_DATASOURCE_SCHEMA_UUID` | Cisco WebSocket schema UUID. Must be `a38a10b7-43e4-4676-a076-a7d6dce9387d`. |
| `CISCO_ORG_UUID` | Required `com.cisco.org.uuid` claim value. |

The adapter accepts RS256 bearer JSON Web Signatures (JWS). It requires
`exp`, `iat`, `iss`, `aud`, `sub`, and `jti`, validates the signature by
issuer and `kid`, and verifies the configured Cisco data-source and
organization claims. It retrieves issuer keys from
`/oauth2/v2/keys/verificationjwk`.

Both `BYOVA_TLS_CERT` and `BYOVA_TLS_KEY` are required together.
Authentication-enabled startup also requires every Cisco value in the table.
Use `CISCO_JWKS_TTL_SECONDS` to change the 3,600-second key-cache lifetime and
`CISCO_AUTH_CLOCK_SKEW_SECONDS` to change the 30-second clock tolerance.

The adapter is not a substitute for Cisco sandbox qualification. Validate the
complete call flow with your Webex Contact Center sandbox before production.

Configure the internal connection separately:

| Variable | Purpose |
| --- | --- |
| `NEMOTRON_VOICE_AGENT_WS` | Internal RTVI endpoint. Default: `wss://127.0.0.1:7860/api/ws`. |
| `NEMOTRON_INSECURE_TLS` | Disables verification for the internal TLS connection only. Default: `false`. |

`NEMOTRON_INSECURE_TLS=true` does not weaken TLS or authentication on the
external Cisco listener.

## Configure the Virtual-Agent Catalog

The catalog values default to the Generic assistant identity:

| Variable | Default |
| --- | --- |
| `BYOVA_VIRTUAL_AGENT_ID` | `nemotron-generic` |
| `BYOVA_VIRTUAL_AGENT_NAME` | `Nemotron Generic` |
| `BYOVA_VIRTUAL_AGENT_DESCRIPTION` | `NVIDIA Nemotron voice agent` |

The implemented configuration also defines bounded-operation settings:
`BYOVA_QUEUE_SIZE` (`128`), `BYOVA_SEND_TIMEOUT_SECONDS` (`5`),
`BYOVA_FIRST_AUDIO_TIMEOUT_SECONDS` (`120`),
`BYOVA_COMPLETION_TIMEOUT_SECONDS` (`10`), and
`BYOVA_MAX_MESSAGE_BYTES` (`1048576`).

## Use Local Test Mode

Plaintext and disabled authentication require separate, explicit opt-ins:

```bash
BYOVA_ALLOW_PLAINTEXT_LOCAL=true
BYOVA_AUTH_ENABLED=false
BYOVA_ALLOW_AUTH_DISABLED_LOCAL=true
NEMOTRON_VOICE_AGENT_WS=ws://127.0.0.1:7860/api/ws
```

If the pipeline also uses plaintext, start it with `PIPELINE_TLS=false`.
Restrict this mode to a trusted local environment. Do not expose an
authentication-disabled or plaintext listener to Cisco or a public network.

## Event and Audio Semantics

The adapter maps conversation events as follows:

| Internal or Cisco event | Result |
| --- | --- |
| Cisco `SESSION_START` | Opens one internal RTVI connection and sends `client-ready`. |
| Pipecat `user-started-speaking` | Sends Cisco `START_OF_INPUT` immediately for barge-in. |
| Pipecat `user-turn-finalized` | Sends Cisco `END_OF_INPUT` at the canonical finalized-turn edge. |
| Pipecat output audio | Uses the frame's declared sample rate and converts mono PCM16 to raw Base64 G.711 mu-law 8 kHz Cisco `CHUNK` output. |
| Adapter output-complete edge | Sends exactly one Cisco `FINAL` after unpaced generated audio is serialized. |
| Cisco `SESSION_END` or socket closure | Closes the internal connection and releases conversation state. |

The documented contract uses one authenticated Cisco WebSocket for the entire
conversation. The adapter also supports the continuation sockets observed in
production: the logical conversation and Nemotron connection remain stable,
while physical Cisco sockets are attached by `conversation_id` with
connection-scoped sequence counters.

An adapter-owned Pipecat output transport serializes audio without browser
playback sleeps. Its stop event is therefore an output-written edge rather
than simulated playback completion.

> **Live qualification (2026-09-09):** the Cisco production connector closed
> the documented socket immediately after the welcome `FINAL` and opened four
> continuation sockets. Compatibility handling is enabled so these sockets can
> resume the existing logical conversation. Cisco should still confirm this
> behavior because it differs from the published single-socket contract.

Caller audio must be raw Base64 G.711 mu-law, 8 kHz, mono data with
`encoding=MULAW_FORMAT` and `sample_rate_hertz=8000`. Adapter output chunks
range from 320 bytes through 32 KB.

## Generic Baseline Fit

The adapter ignores raw `user-stopped-speaking` and uses Pipecat's finalized
turn for `END_OF_INPUT`. Cisco receives generated audio without
transport-side playback pacing.

## Customize the Assistant

The following files remain equivalent to Generic:

| Path | Role |
| --- | --- |
| `pipeline.py` | Pipecat cascaded-pipeline entry point |
| `prompts.yaml` | Prompt catalog and per-prompt tool selection |
| `tools.yaml` | OpenAI function-calling schemas |
| `tool_handlers.py` | Tool implementations |
| `tools.py` | Filtered Pipecat tool-schema builder |
| `services.cloud.yaml` | NVIDIA cloud service catalog |
| `services.local.yaml` | Server and single-GPU service catalogs |

Refer to [Configure Services](../../../docs/how-to/configure-services.md),
[Configure LLM](../../../docs/how-to/configure-llm.md),
[Configure ASR](../../../docs/how-to/configure-asr.md),
[Configure TTS](../../../docs/how-to/configure-tts.md), and
[Configure Prompts](../../../docs/how-to/configure-prompts.md) before changing
the Generic-equivalent pipeline behavior.
