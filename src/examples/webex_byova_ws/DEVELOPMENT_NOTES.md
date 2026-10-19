# Cisco Webex BYOVA WebSocket Integration Notes

A running record of building the adapter in [`client/`](./client/) against a
live Webex Contact Center tenant during September 2026. It exists because
several behaviors the adapter depends on were not derivable from the published
documentation, and because for those the route to the answer is more useful
than the answer by itself.

Entries are chronological. Where a later finding overturned an earlier one, the
earlier conclusion is kept and marked superseded rather than deleted: the wrong
turn is part of what makes the final rule credible, and anyone hitting the same
symptom will recognize the detour.

Reference:
[BYOVA over WebSocket](https://developer.webex.com/mcp/docs/bring-your-own-virtual-agent-over-websocket)

## The Documented Contract

The documentation describes the lifecycle the adapter now implements:

1. Webex opens one secure `/v1/va` WebSocket for one `conversation_id`.
2. `SESSION_START` is the first application message on that connection.
3. The same connection carries the welcome response, caller media, DTMF,
   heartbeat traffic, and all subsequent assistant turns.
4. The partner sends exactly one `FINAL` for each logical reply.
5. `FINAL` ends only that reply; it does not end the conversation or close the
   WebSocket.
6. The same connection continues with the next caller turn.
7. Webex does not reconnect, resume, or replay a session after a connection
   failure.

Each side owns one monotonic sequence counter for the connection, and a `PONG`
echoes the sequence number of the `PING` it answers.

## Welcome-Only Calls (September 9)

A strict implementation of the contract above completed the welcome message
and then went deaf. Caller audio never arrived.

What we saw on every live call: Webex opened the first socket and sent
`SESSION_START`; the welcome reply went out as mu-law chunks closed by one
`FINAL`; Webex then **closed that socket** and opened further `/v1/va` sockets
for the same call. Because a strict implementation requires `SESSION_START` as
the first message, it rejected those continuation sockets, so the caller's
speech never reached the pipeline. Two live conversations showed the welcome
completing in roughly 0.5–0.9 seconds and then stalling, one of them followed
by four additional connections.

We worked around it by decoupling the logical conversation from any physical
socket: one pipeline session per `conversation_id`, continuation sockets
resolved by that ID and attached to the existing conversation, sequence
counters reset per physical socket, and promotion of a new socket deferred
until any in-flight reply and its `FINAL` had finished. That produced working
multi-turn calls.

**Resolved September 16.** Cisco identified the multi-socket behavior as an
internal bug and deployed a fix. The tenant now keeps one `/v1/va` socket for
the whole conversation. The workaround was removed, and the adapter implements
the published single-socket contract with no continuation handling at all.

## Prompt Audio Accepted and Silently Discarded (September 17)

With the lifecycle fixed, every call connected and every turn traversed the
pipeline correctly, but no assistant audio was ever rendered to the caller.
Webex accepted the prompts without error and reported nothing unusual, so there
was nothing to debug from: every log on the partner side shows a healthy call.

We isolated it with throwaway probe connectors that replaced the adapter and
replied with fixed payloads, holding everything constant and varying one factor
at a time. That ruled out most of the envelope: rendering is unaffected by
`response_type` (single `FINAL` replies and streamed `CHUNK` sequences behave
alike), by `is_barge_in_enabled`, and by reply position (immediately after
`SESSION_START`, unsolicited mid-call, and in answer to a caller turn all
behave alike). A reply sent while the caller is still speaking is discarded as
a barge-in, which is correct.

What did change the outcome was the prompt media itself:

| Prompt encoding | Rendered to caller |
| --- | --- |
| Raw G.711 mu-law 8 kHz, no container | No |
| Raw PCM16 8 kHz, no container | No |
| PCM16 8 kHz in a WAV container | Yes |

**Superseded conclusion.** We read this as "the container, not the codec, is
the deciding factor" and shipped WAV-contained PCM16, which gave us working
multi-turn conversations. That conclusion was wrong, but it was not
unreasonable: it also explained why Cisco's own sample connector works.
`AudioFileUtil.audioContentFromResources` base64-encodes each prompt asset
verbatim with `readAllBytes()`, and those assets are WAV files —
`call-start.wav` is RIFF WAVE, Microsoft PCM, 16 bit, mono, 8 kHz. So the
sample sends WAV-contained PCM16 while the *Audio format and chunking* media
profile says `Container: None; do not include WAV, RIFF, or other container
headers` and the FAQ answers "Can the partner send WAV audio?" with "No."

## Prompt Media Is Coupled to `dtmf_config` (September 18)

Cisco suggested sending `input_handling_config` alongside `INPUT_VOICE_DTMF`.
With a `dtmf_config` present, raw mu-law rendered — the documented profile had
been correct all along. Running all four combinations on the live tenant gave
the actual rule:

| Prompt audio | `dtmf_config` | Result |
| --- | --- | --- |
| Raw mu-law 8 kHz | absent | Accepted on the socket, nothing rendered |
| Raw mu-law 8 kHz | present | Rendered |
| WAV-contained PCM16 8 kHz | absent | Rendered |
| WAV-contained PCM16 8 kHz | present | Rendered as noise |

So the earlier table was measured entirely in the `dtmf_config`-absent row, and
the container looked decisive only because that row has exactly one working
cell. The adapter always sends `dtmf_config`, so it always sends raw mu-law.
`BYOVA_PROMPT_AUDIO_PROFILE=wav-pcm16` still selects the other profile for
comparison testing.

None of this coupling is documented, and the failure mode is a silent discard.

## Clipped Prompt Starts (September 18)

Once raw mu-law rendered, callers heard the *start of prompts cut off* —
including the welcome message, with no interruption and no keypress, and most
noticeably on the reply following a barge-in. Switching the same adapter to
WAV-contained PCM16 without `dtmf_config` removed the clipping, which is why
this surfaced only after the media fix.

Everything we could measure said the audio left intact. Decoding the exact
bytes written to the socket, each reply began at the first word at full level,
framed as roughly 100 ms `CHUNK` prompts closed by exactly one `FINAL`, with
contiguous sequence numbers and no gaps. Asked to confirm the `FINAL` framing,
we matched our wire log against Cisco's to within 4 ms: a 49-chunk welcome
turn, 39,129 mu-law bytes, was 4.89 seconds of audio fully transmitted in
238 ms, with the `FINAL` emitted 2 ms after the last chunk write. The tight
`CHUNK`-to-`FINAL` interval is faster-than-realtime delivery as the guide
recommends, not a truncated prompt. Inbound caller frames arriving inside that
window go to ASR only and never touch reply framing; the only thing that ends a
turn early is our own VAD, which always emits `START_OF_INPUT` first.

Because the only two intelligible combinations in the matrix above differ in
both variables, we could not isolate the cause from outside.

**Pacing experiment.** Cisco asked us to insert 80–100 ms between chunks. We
added 90 ms (chunks carry about 100 ms each, so just under realtime) and
measured inter-chunk gaps of 90–93 ms, median 91, none above 150 ms. The
caller then heard broken, choppy audio — worse than streaming as fast as
generated, which at least played continuously apart from the clipped starts.
Read against Cisco's own log line for one call (first chunk received at
`11:34:01.655`, playback started at `11:34:01.704`), it suggested roughly 49 ms
of per-chunk playback setup that is hidden when chunks queue up and becomes
audible when they arrive at the rate they are consumed. The pacing was removed.

**Resolved September 30.** Cisco deployed a fix. We retested the normal
configuration — raw mu-law with `dtmf_config`, `INPUT_VOICE_DTMF`,
`is_barge_in_enabled: true`, chunks streamed as generated — across three calls,
29 assistant turns and about 255 seconds of prompt audio. The caller
interrupted 11 times, several times with a large unplayed backlog (up to
4.1 seconds, others at 3.2, 2.4 and 2.2 seconds). Every following reply played
from its first word, the welcome message played complete, and prompts stopped
promptly on barge-in. Those were precisely the conditions that previously
produced clipped starts.

## Latency (September 30)

Measured inside the pipeline over 30 turns, end of caller turn to first byte of
bot audio is a median of 0.88 s and a p90 of 1.09 s. Most of that is turn
detection (500 ms VAD hangover, about 0.53 s) and ASR finalization (about
0.45 s); LLM first token is about 0.12 s and TTS starts in under 0.1 s. What
the caller perceives adds roughly 300–500 ms on top, which we attribute to
network hops plus the per-chunk playback setup above.

## Still Open

**Service App registration.** The documentation says to reuse the same
authorized Service App when registering the WebSocket data source and
explicitly advises against creating a second one for the WebSocket transport.
During integration we were advised that two data sources cannot be kept in one
Service App and that a separate Service App should be created for WebSocket. We
run one Service App and data source per transport, which works, but it
contradicts the published instruction. Which is the supported production
configuration is unconfirmed.

**Diagnosability of rejected prompt audio.** Both media problems above
presented as silence with a healthy call on every log we own. An error or
warning frame when a prompt cannot be decoded or rendered — rather than a
silent discard — would have removed most of the work recorded here.

## If You Are Building One of These

- Send raw G.711 mu-law 8 kHz with no container, and send `dtmf_config` with
  `INPUT_VOICE_DTMF`. The two are coupled; neither works alone.
- Silence with clean logs on both sides means the media contract, not the
  protocol. Vary one factor at a time with a probe connector rather than
  instrumenting the real adapter.
- Stream chunks as soon as they are generated. Pacing them near realtime made
  playback worse, not better.
- Treat the sample connector's behavior as evidence about the live service, but
  not as proof of a rule — it happened to sit in the one working cell of a
  matrix neither it nor the documentation describes.
