# Cisco Webex BYOVA WebSocket Integration Feedback

Date: September 9, 2026

## Summary

The documented Webex BYOVA WebSocket lifecycle differs from the behavior
observed in our live Webex Contact Center tenant. A strict implementation of
the published single-socket contract completes the welcome message but cannot
receive the caller's subsequent audio. Supporting socket continuation by
`conversation_id` allows the conversation to proceed.

## Documented WebSocket Flow

The Webex documentation states:

1. Webex opens one secure `/v1/va` WebSocket for one `conversation_id`.
2. `SESSION_START` is the first application message on that connection.
3. The same connection carries the welcome response, caller media, DTMF,
   heartbeat traffic, and all subsequent assistant turns.
4. The partner sends exactly one `FINAL` for each logical reply.
5. `FINAL` ends only that reply; it does not end the conversation or close the
   WebSocket.
6. The same connection continues with the next caller turn.
7. Webex does not reconnect, resume, or replay an existing session following a
   connection failure.

Reference:
[BYOVA over WebSocket](https://developer.webex.com/mcp/docs/bring-your-own-virtual-agent-over-websocket)

## Observed Live Behavior

In live calls, we observe:

1. Webex opens the first `/v1/va` socket and sends `SESSION_START`.
2. The partner opens the internal Nemotron session.
3. Nemotron generates the welcome response successfully.
4. The partner sends schema-valid G.711 mu-law audio chunks followed by one
   `FINAL`.
5. Webex closes the original socket after the welcome `FINAL`.
6. Webex opens additional `/v1/va` sockets for the same call.
7. Caller audio is not sent on the original conversation socket.
8. The strict documented implementation rejects the continuation sockets
   because their first message is not a new `SESSION_START`; caller audio
   consequently never reaches Nemotron.

This welcome-only behavior was observed for these live Cisco conversations:

- `fd85c442-17da-4997-b58a-be5e4e812285`
  - Welcome first-speech latency: approximately 0.94 seconds.
  - Output media: raw G.711 mu-law, 8 kHz mono, normally 8,000 bytes per chunk.
  - Result: welcome completed, followed by four new `/v1/va` connections and
    no caller audio on the original socket.
- `2c9f1a2a-6ca2-49b4-860a-147dba2b306a`
  - Welcome first-speech latency: approximately 0.52 seconds.
  - Result: welcome completed and Nemotron became ready for caller input, but
    no subsequent caller speech reached the Nemotron pipeline.

Both IDs came from live Webex calls and identify the affected logical
conversations, not synthetic local tests.

## Behavior That Works

Our earlier compatibility implementation allowed calls to continue by:

1. Owning the logical conversation independently from any physical WebSocket.
2. Keeping one Nemotron RTVI connection alive for the logical conversation.
3. Resolving each continuation socket by `conversation_id`.
4. Attaching the continuation socket to the existing conversation.
5. Treating inbound and outbound sequence counters as connection-scoped and
   resetting them for each physical socket.
6. Waiting for an active assistant reply, including all audio and its `FINAL`,
   to finish before promoting a continuation socket.
7. Preserving caller barge-in and the existing Nemotron context across socket
   changes.

This compatibility behavior enabled successful multi-turn live calls. It was
removed when we implemented the documented canonical single-socket lifecycle,
which made the welcome-only failure reproducible.

After restoring the compatibility layer, live conversation
`fcd803ec-009f-4dfd-8ff2-5867773cd3a8` completed successfully. The adapter
attached seven Cisco continuation sockets to the same Nemotron session,
processed multiple caller transcripts and assistant replies, and closed with
20 output audio chunks totaling 130,104 raw mu-law bytes.

## Service App Documentation Versus Guidance

The WebSocket documentation says to use the same authorized Service App while
registering the WebSocket data source and explicitly says not to create a
second Service App for the WebSocket transport.

During integration, the Webex team advised us that two data sources could not
be kept in the same Service App and recommended creating a separate Service
App for WebSocket. We therefore currently use:

- One Service App and data source for gRPC BYOVA.
- A different Service App and data source for WebSocket BYOVA.

This works operationally, but it conflicts with the published instruction that
a second Service App should not be created for WebSocket.

## Clarifications Requested

Please confirm:

1. Is `/v1/va` intended to remain one physical WebSocket for the complete
   conversation, including all turns after the welcome `FINAL`?
2. If continuation sockets are expected, how must they identify and resume the
   existing conversation, and are sequence counters scoped to each physical
   socket?
3. Should a continuation socket begin with another `SESSION_START`, or may its
   first message contain caller audio?
4. Does an audio-bearing welcome `FINAL` require any additional field or
   acknowledgement to keep the original socket open?
5. Can gRPC and WebSocket data sources coexist in one Service App, or is one
   Service App per transport the supported production configuration?
6. If separate Service Apps are required, please update the WebSocket
   documentation accordingly.

No credentials, bearer tokens, caller audio, DTMF values, or transcripts are
included in this report.
