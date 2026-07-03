# Cisco Webex Contact Center BYOVA

This example connects Nemotron Voice Agent to a Cisco Webex Contact Center IVR
through BYOVA (Bring Your Own Virtual Agent). It runs the server-side voice
assistant used by the Cisco-facing adapter in [`client/`](./client/).

> **Note:** This integration is intended for POCs and experimentation. It is
> not a full-featured production connector; production deployments must assess
> their own security, availability, scaling, and operational requirements.

## How the adapter works

The Webex Contact Center flow invokes the adapter from a `VirtualAgentV2`
activity over gRPC+TLS. For each call, the adapter creates a session with this
example through `/api/session-config`, opens `/api/ws`, and translates audio and
events between the Cisco BYOVA protocol and the Nemotron Voice Agent WebSocket
protocol. Caller audio is sent to Nemotron Voice Agent, and generated speech is
returned to the IVR as BYOVA audio responses.

The example pushes generated TTS audio to the adapter without browser-style
real-time pacing. The adapter also defers Cisco `END_OF_INPUT` until bot speech
starts, preserving the caller's full barge-in opportunity before the response.

## Running the backend

Start the backend from the repo root:

```bash
docker compose --profile webex-byova-assistant/workstation up -d
```

This example uses the normal workstation ASR, LLM, and TTS services and pins the
server to `websocket` transport for the adapter path.

## Where to look next

- Adapter package and local adapter commands:
  [`src/examples/webex_byova/client/README.md`](./client/README.md)
- End-to-end backend + adapter + Cisco sandbox flow:
  [`run.md`](./run.md)
