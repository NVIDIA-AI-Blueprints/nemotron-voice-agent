# Cisco Webex Contact Center BYOVA

This example connects Nemotron Voice Agent to a Cisco Webex Contact Center IVR
through BYOVA (Bring Your Own Virtual Agent). It runs the server-side voice
assistant used by the Cisco-facing adapter in [`client/`](./client/).

> **Note:** This integration is intended for POCs and experimentation. It is
> not a full-featured production connector; production deployments must assess
> their own security, availability, scaling, and operational requirements.

## How the adapter works

The Webex Contact Center flow invokes the adapter from a `VirtualAgentV2`
activity over gRPC+TLS. For each call, the adapter opens `/api/ws` directly and
translates audio and events between the Cisco BYOVA protocol and the Nemotron
Voice Agent WebSocket protocol. No process-local HTTP session configuration is
created, so calls can be distributed across multiple Uvicorn workers. Caller
audio is sent to Nemotron Voice Agent, and generated speech is returned to the
IVR as BYOVA audio responses.

The example pushes generated TTS audio to the adapter without browser-style
real-time pacing, and the adapter converts every frame from the sample rate it
declares to the 8 kHz mu-law format Webex expects. The adapter also defers
Cisco `END_OF_INPUT` until bot speech starts, preserving the caller's full
barge-in opportunity before the response. It uses 500 ms Silero VAD and
preserves the adapter's proven 350 ms output-idle turn finalization.

## Capabilities

The default `customer_care` prompt demonstrates:

- **LLM call control.** `transfer_to_human` and `end_call` are zero-argument
  tools the model invokes on explicit caller intent. The adapter turns them
  into the matching BYOVA transfer or `SESSION_END` output event, so no
  transcript keyword matching remains anywhere in the call path.
- **Secure keypad authentication.** `request_keypad_input` collects a phone
  number and a date of birth over Cisco's DTMF-only input mode before the
  support conversation starts.
- **Voice turns after authentication.** Once every field arrives, the input
  mode returns to voice and the full tool surface is restored.

## Authentication call flow

Authentication is tool-driven and deterministic. The prompt never asks for
keypad digits on its own, and the LLM never speaks the instructions.

1. On session start the pipeline exposes only `request_keypad_input` and runs
   one inference, so the call opens with that tool instead of the usual
   welcome message.
2. The tool arms the collection and speaks the instruction itself: "Hello,
   I'm Nova from Northstar Services. To authenticate, please enter your 10
   digit phone number on the keypad."
3. The adapter advertises `INPUT_EVENT_DTMF` with `input_sensitive=true` and
   `dtmf_input_length` set to the field's exact digit count, so the digit
   count alone completes an entry. No terminator key is configured; a stray
   `#` arrives as its own interaction and is ignored.
4. A completed entry enters the LLM context as sensitive input. While another
   field is still required, the pipeline re-exposes the keypad tool and queues
   one inference so the model asks for the next field. After the last field it
   speaks a fixed confirmation with no extra inference and hands the
   conversation back to the model.
5. A rejected entry keeps the same field armed and speaks a retry prompt that
   names the reason, such as "it was not exactly 10 digits" or "it was not a
   valid date in day month year format". A Cisco keypad timeout retries the
   same way, and a caller who speaks instead of typing simply hears the
   instruction again.

The demo checks entry format only. Any 10-digit phone number and any valid
8-digit `DDMMYYYY` date of birth are accepted, verification always succeeds, and
the assistant then moves straight into the support conversation. Keypad values
are passed to the LLM context as sensitive input but are not written to adapter
logs or repeated aloud by the prompt.

## Example layout

Every resource this integration needs lives under this example directory:

- `pipeline.py` — cascaded pipeline plus the Webex session and DTMF wiring
- `transport.py` — WebSocket transport that writes audio without pacing
- `input_state.py` — keypad field sequence, spoken prompts, and the frame
  processor that consumes adapter DTMF messages
- `tools.py`, `tool_handlers.py`, `tools.yaml` — call-control and keypad tools
- `prompts.yaml` — the `customer_care` persona
- `services.cloud.yaml`, `services.local.yaml` — ASR, LLM, and TTS catalogs
- `client/` — the Cisco-facing gRPC adapter package

## Running the backend

Start the backend from the repo root:

```bash
docker compose --profile webex-byova-assistant/server up -d
```

This example uses the normal server ASR, LLM, and TTS services and pins the
server to `websocket` transport for the adapter path.

The Compose profile locks the backend to this example with
`EXAMPLE_SELECTION=webex-byova-assistant`, so the adapter connects directly to
`/api/ws` without any request-level example selection. To run multiple backend
workers:

```bash
UVICORN_WORKERS=4 \
  docker compose --profile webex-byova-assistant/server up -d
```

## Where to look next

- Adapter package and local adapter commands:
  [`src/examples/webex_byova/client/README.md`](./client/README.md)
- End-to-end backend + adapter + Cisco sandbox flow:
  [`run.md`](./run.md)
- Tool and keypad-state tests:
  [`tests/unit/test_webex_byova_controls.py`](../../../tests/unit/test_webex_byova_controls.py)
- Adapter protocol tests:
  [`client/tests/test_adapter.py`](./client/tests/test_adapter.py)
