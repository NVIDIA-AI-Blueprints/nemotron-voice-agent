# Connect Webex Contact Center to Nemotron Voice Agent

This POC connects a Cisco Webex Contact Center IVR flow to the
`webex-byova-assistant` server example. A `VirtualAgentV2` activity invokes the
adapter, which bridges Cisco's BYOVA gRPC interface to the Nemotron Voice Agent
HTTP and WebSocket APIs and returns generated speech to the caller.

> **Note:** This is an integration example for POCs and experimentation, not a
> full-featured production connector. Review and adapt its security,
> availability, scaling, and operations for your deployment.

In this setup:

- the backend example is `webex-byova-assistant`
- the client is the vendored adapter in [`client/`](./client/)
- Cisco Webex Contact Center connects to the adapter over gRPC+TLS
- the adapter connects directly to this server over `/api/ws`

## Architecture

```text
Caller (PSTN)
  -> Webex Calling
  -> Webex Contact Center flow
  -> VirtualAgentV2 activity
  -> gRPC+TLS -> Cisco Webex BYOVA adapter
  -> wss://<server>:7860/api/ws -> nemotron-voice-agent
```

## 1. Start the backend server

From the repo root:

```bash
docker compose --profile webex-byova-assistant/workstation up -d
```

This recipe uses the normal workstation hardware/service stack and pins the
server to `websocket` transport for the adapter path.

It also locks the backend deployment with
`EXAMPLE_SELECTION=webex-byova-assistant`, so the adapter connects directly to
`/api/ws` without request-level example routing. Multiple Uvicorn workers are
supported because the adapter does not use `/api/session-config` or a
`session_id`:

```bash
UVICORN_WORKERS=4 \
  docker compose --profile webex-byova-assistant/workstation up -d
```

`SILERO_VAD_STOP_SECS` controls how much trailing silence is required after
speech before the user turn ends; it does not control speech-start detection.
The BYOVA default is `0.8` seconds. Lower values respond faster, while higher
values better tolerate natural pauses:

```bash
SILERO_VAD_STOP_SECS=1.0 \
  docker compose --profile webex-byova-assistant/workstation up -d
```

Confirm the backend is healthy before starting the adapter.

## 2. Start the adapter client

The adapter code lives in:

- [`client/`](./client/)

The adapter requires a certificate for the public hostname that Cisco uses to
reach it. The adapter does not generate this certificate, and certificate or
private-key files must not be committed to the repository. You can use a
certificate provisioned by your organization or obtain one with
[Certbot](https://certbot.eff.org/instructions). For example:

```bash
sudo certbot certonly --standalone -d <public-host>
```

For the standalone HTTP-01 flow, `<public-host>` must resolve to the adapter
host, inbound TCP port 80 must be reachable, and no other process can occupy
that port while Certbot runs. See the
[Let's Encrypt challenge documentation](https://letsencrypt.org/docs/challenge-types/)
for HTTP-01 and DNS-01 alternatives.

Start the adapter with the full certificate chain and private key generated or
provisioned for that hostname. The user running the adapter must have permission
to read both files.

```bash
cd src/examples/webex_byova/client
uv sync
export NEMOTRON_BYOVA_ADAPTER_TLS_CERT=/etc/letsencrypt/live/<public-host>/fullchain.pem
export NEMOTRON_BYOVA_ADAPTER_TLS_KEY=/etc/letsencrypt/live/<public-host>/privkey.pem
./scripts/run_external_adapter.sh
```

Important adapter defaults:

- `NEMOTRON_VOICE_AGENT_WS=wss://127.0.0.1:7860`
- `NEMOTRON_BYOVA_ADAPTER_GRPC_PORT=50061`
- `NEMOTRON_BYOVA_ADAPTER_HEALTH_PORT=8081`

The adapter expects a publicly trusted TLS certificate for the Cisco-facing
endpoint. Set:

- `NEMOTRON_BYOVA_ADAPTER_TLS_CERT=/etc/letsencrypt/live/<public-host>/fullchain.pem`
- `NEMOTRON_BYOVA_ADAPTER_TLS_KEY=/etc/letsencrypt/live/<public-host>/privkey.pem`

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
Cisco-issued JWS claims or Cisco data-source configuration. If
`ENABLE_CISCO_JWS_VALIDATION=true`, both `CISCO_JWT_ISSUER` and
`CISCO_JWT_AUDIENCE` are required; `CISCO_JWT_SUBJECT` remains optional.

Self-signed certificates are not suitable for the Cisco-facing endpoint. For
local-only smoke testing without Cisco, you can explicitly allow plaintext:

```bash
ALLOW_PLAINTEXT_ADAPTER=true ./scripts/run_external_adapter.sh
```

## 3. Add the adapter to a Cisco sandbox and create a flow

Cisco provisions a BYOVA connector through Bring Your Own Data Source (BYODS).
First obtain a Webex Contact Center sandbox and request the Voice Virtual Agent
entitlement through Cisco developer support. Then complete these steps:

1. In the Webex Developer Portal, create a Service App with the
   `spark-admin:dataSource_read` and `spark-admin:dataSource_write` scopes.
   Select the Voice Virtual Agent data exchange schema and add the adapter's
   approved public domain without a scheme or port. Submit the app for sandbox
   administrator approval.
2. In Control Hub, go to **Apps**, authorize the Service App, and then generate
   an access token for the sandbox organization from the Developer Portal.
3. Use that token to register a data source with a `POST` request to
   `https://webexapis.com/v1/dataSources`. Use the Voice Virtual Agent schema ID
   `5397013b-7920-4ffc-807c-e8a3e0a18f43` and the public TLS URL that routes to
   the adapter's gRPC port, such as `https://<public-host>:50061` or a port 443
   gateway that forwards to port 50061.
4. In Control Hub, go to **Contact Center > Integrations > Features** and create
   a CCAI configuration for the Service App and registered data source. The
   adapter advertises `nemotron-generic` as its default virtual agent unless
   `DEFAULT_VIRTUAL_AGENT_ID` is overridden before startup.
5. In Flow Designer, create a flow, add a **Virtual Agent V2** (Virtual Agent
   Voice) activity, and select the CCAI configuration and virtual agent. Save
   and publish the flow.
6. Go to **Contact Center > Channels > Entry Point**, map the phone entry point
   to the new flow, and place a test call.

Follow Cisco's
[Bring Your Own Virtual Agent](https://developer.webex.com/webex-contact-center/docs/bring-your-own-virtual-agent)
and
[Bring Your Own Data Source](https://developer.webex.com/webex-contact-center/docs/bring-your-own-data-source-cc)
guides for the current portal fields and authorization flow. As part of the
deployment, handle Cisco Service App and data source onboarding for each
organization, including the required token and nonce rotation.

## 4. Smoke test before calling from Cisco

Before placing a phone call, validate the adapter locally:

```bash
cd src/examples/webex_byova/client
uv run python scripts/test_byova_client.py \
  --grpc-target <host>:50061 \
  --tls \
  --audio-file /path/to/audio.wav \
  --response-wait-secs 25
```

This simulates the Webex Universal Harness and helps confirm that:

- the adapter accepts gRPC requests
- the backend session starts successfully
- bot audio is returned end-to-end
- transcript keyword matching can produce transfer-to-agent or session-end

## Notes

- Calls enter through the Webex Contact Center IVR; the adapter is the client of
  the Nemotron Voice Agent example server.
- The BYOVA backend and adapter are meant to be co-located on the same VM in
  the default setup.
