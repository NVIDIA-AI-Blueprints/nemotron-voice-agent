# Offline Processing for Nemotron Voice Agent

## Aim and Background

Enable Nemotron Voice Agent to process recorded conversations faster than real
time, so accuracy and task-completion benchmarks do not wait for every recording
to play. Input contains speech and silence. Output contains bot responses placed
on the same audio timeline, with transcripts and server metrics for inspection.

Bulk upload removes real-time input pacing, but arrival time no longer indicates
where speech occurs. Removing output sleeps alone also loses turn placement,
silence, and interruption behavior. The proposal addresses this by scheduling
conversation events on audio sample offsets and measuring model-service delays
separately.

For example, a 300,000 ms input produces at least 300,000 ms of bot output.
Gaps contain silence, responses begin when their inputs and generated audio are
available, and a final response can extend beyond the input.

The first integration uses automatic speech recognition (ASR), a large language
model (LLM), text-to-speech (TTS), and voice activity detection (VAD) in the
Generic Assistant. The scheduling architecture supports other examples through
provider adapters and composition factories.

The scope is a custom finite-session WebSocket endpoint, `/api/ws/offline`.
The existing live WebSocket, OpenAI Realtime gateway, and WebRTC paths retain
their execution behavior. This proposal does not add WebRTC offline support.

The design baseline uses standard continuous NVIDIA speech-to-text (STT) with
`stop_history=320` (320 ms), without force-EOU injection or synthetic control
silence. Both evaluation modes must use that baseline. This STT setting is
separate from Silero's silence threshold and adds no separate scheduler delay.

## Architecture

The server validates and seals the complete source before processing. VAD,
turn detection, and one continuous ASR request analyze that immutable recording.
A scheduler replays their events in source order, requests responses, and commits
the audio and conversation history that survive interruptions.

The following diagram shows the processing boundaries.

```mermaid
flowchart LR
    Input[Finite upload and explicit EOF] --> Source[Immutable source PCM]
    Source --> Analysis[VAD, turn detection, and continuous ASR]
    Analysis --> Replay[Sample-offset scheduler]
    Replay --> Generation[LLM and overlapping TTS]
    Generation --> Replay
    Replay --> Commit[Committed audio and history]
    Commit --> Output[Aligned WAVs, events, and positioned output]
```

Source analysis can run ahead, but response generation receives only transcripts
whose source dependency and availability have been reached. Each response uses
a private history snapshot. The scheduler decides which generated audio and
history updates become committed. Future user speech cannot enter an earlier
response.

Two time domains keep conversation placement independent of processing speed.

| Domain | Controls |
| --- | --- |
| Audio timeline: integer sample offsets | Speech, silence, endpoints, playback, interruption, and recording extent |
| Execution time: monotonic durations | Provider availability, processing measurements, and failure watchdogs |

At 16 kHz, 512 analysis samples always represent 32 ms. Upload speed and network
pauses do not change that duration. Watchdogs can fail stalled work; they cannot
end user turns or advance the audio timeline.

### Timing Rules

Offline execution replaces the following pacing and idle dependencies.

| Pipeline Behavior | Offline Rule |
| --- | --- |
| VAD thresholds, state resets, and Smart Turn buffers | Count and select source samples |
| Receive-idle and user-speech timeouts | Use source silence, detector decisions, and explicit EOF |
| ASR keepalive and stream completion | Consume source PCM through one continuous request and drain at EOF |
| Output pacing, empty-queue silence, and bot-speaking state | Schedule audio spans, zero-fill gaps, and derive playback start/end |
| TTS completion and conversation history | Observe synthesis completion; commit only completed or justified played text |
| Recording timestamps and session completion | Use source offsets and explicit completion events |

Measured compute and service delays remain part of response placement. A turn
becomes ready after both endpoint computation and its owned final transcripts
are available. Transcript availability is tracked independently of source
position and upload progress.

LLM output streams into TTS while generation continues. Each audio chunk retains
its observed readiness relative to the response request. The scheduler places it
after both that readiness and the preceding chunk's playback end.

The core placement rules are:

```text
S(duration_ns) = ceil(duration_ns * 16000 / 1_000_000_000)
turn_ready = max(endpoint_sample + S(endpoint_compute_ns),
                 latest_owned_transcript_available_sample)
chunk_start = max(response_request_sample + S(chunk_ready_ns),
                  previous_chunk_end)
chunk_end = chunk_start + chunk_sample_count
```

Continuous ASR associates each final transcript with its source dependency.
The adapter projects availability from that dependency plus the observed delay
after the relevant audio was supplied. Bulk-input backlog contributes to this
projection; it does not establish paced-live ASR latency.
The [scheduling reference](offline-processing-reference.md#provider-availability-and-playback)
defines the calculation and recorded policy.

### Interruption and Recording Rules

The scheduler and recorder enforce these invariants.

- **Immutable source:** validate sample count, packet count, and SHA-256 at explicit
  EOF. A disconnect cancels the session.
- **Causal history:** assign transcripts from turn start through endpoint,
  including quiet trailing words. Missing or ambiguous ownership fails explicitly.
- **Confirmed interruption:** user speech takes priority at its confirmed sample.
  Clip current bot audio there and discard later spans; do not backdate the cut.
- **Played context:** commit completed assistant text or a played prefix supported
  by valid TTS word alignment. Omit interrupted assistant text when alignment is invalid.
- **Aligned output:** spans are ordered and nonoverlapping, uncovered samples are
  zero, and output covers the input plus any final response tail.

These rules let reviewers inspect turn placement without relying on packet
arrival times or manually shifting recordings.

## Implementation Overview

The first integration lives in `src/examples/generic/offline/`. Narrow
`Detector`, `Recognizer`, and `Responder` contracts keep scheduling independent
of model-service SDKs and transport classes.

The implementation divides responsibility as follows.

| Component | Responsibility |
| --- | --- |
| Source detection and recognition | Source-indexed VAD/Smart Turn decisions and continuous ASR availability |
| Response providers | Streaming HTTP LLM, overlapping stitched Magpie TTS, timers, and word alignment |
| Replay engine | Turn readiness, private history, interruption, and audio/history commitment |
| WebSocket protocol and recorder | Bounded upload, offset/hash validation, explicit EOF, positioned output, and WAVs |
| Integration factory | Example-specific prompts, tools, service settings, and resource lifecycles |

This separation gives each component one responsibility and allows providers,
transports, and recorders to be replaced through their contracts. Another example
supplies a factory and compatible adapters. A gateway integration additionally
needs codec conversion, completion, cancellation, and conversation truncation
tied to scheduled playback. No global Pipecat clock is patched.

The first adapters support continuous Riva ASR, HTTP chat completions, and
stitched Magpie TTS. They preserve selected prompts, tools, voice, generation
settings, and a configuration fingerprint. Automatic introductions are disabled.
Both modes request the same native TTS rate before converting to canonical
16 kHz PCM. Unsupported provider modes fail explicitly.

The server saves `source.wav`, `full_output.wav`, `events.jsonl`, and
`manifest.json`. The client also saves a source-horizon `output.wav` and
`comparison.wav`, with user input on the left and bot output on the right.
Failures produce explicit status instead of successful-looking recordings.

For packet layouts, configuration restrictions, module mappings, and executable
commands, refer to the
[implementation and evaluation reference](offline-processing-reference.md).

## Validation and Results

An unscored composite recording has completed through both execution paths on
Pipecat 1.12.0 with standard continuous STT, `stop_history=320`, and no force-EOU
injection. Recording checks verify identical source PCM, output extents,
committed spans, and stereo alignment. The exported evidence bundle retains
WAVs, server metrics, and runtime provenance; refer to the
[validation artifact reference](offline-processing-reference.md#completed-composite-inspection).

Behavioral and scoring consistency against real-time streaming is being
validated. Full-Duplex-Bench (FDB) score validation is in progress; official
scoring will follow sharing this design proposal.

The comparison uses identical canonical PCM, application revision, model
services, prompts, voice, generation settings, and detector settings. Each case
starts a fresh conversation. Effective STT settings and source hashes are recorded.

The two execution paths are:

- **Default online:** send input at its recorded rate through the normal live
  pipeline at `/v1/realtime`.
- **Offline POC:** bulk-upload the same samples through `/api/ws/offline`.

Pacing uploads to the offline endpoint does not exercise the default live
pipeline. Independent generation can vary even at temperature zero, so matching
WAV bytes is not the acceptance criterion.

### Prepared Recordings and Acceptance Checks

Four sources cover a natural turn, a pause, an interruption, and a clean
baseline. Each preserves its canonical prefix and adds the same 4,000 ms of
explicit silence in both modes for a natural final endpoint.

| Source | Prepared Duration, ms |
| --- | ---: |
| FDB v1 smooth turn 117 | 13,099.94 |
| FDB v1 synthetic pause 88 | 9,322.69 |
| FDB v1.5 user interruption 135 | 20,594.69 |
| FDB v1.5 clean baseline 135 | 20,595.00 |

These padded files test comparison behavior. Official scoring must follow the
benchmark's required inputs, source horizon, and scorer rules.

Validation compares recognized text, response intent, turn ownership, first bot
audio, silence, interruption cuts, played context, and final-tail preservation.
File checks verify hashes, formats, server/client PCM, and stereo alignment.
The benchmark scorer separately measures accuracy and task outcomes; WAV
integrity alone does not establish either.

### Server Measurements and Test Coverage

All displayed durations use milliseconds. Reports retain explicit timing scopes:

- **Evaluation turnaround:** upload, processing, publication, and total completion.
- **Turn placement:** speech end, endpoint, first committed bot sample, and turn gap.
- **Providers:** ASR first text/finalization, LLM first text/completion, and TTS
  first audio/completion.
- **Live observations:** server-reported turn metrics and separate client
  receipt/playback measurements.

Offline batch E2E and live turn E2E cover different boundaries. LLM/TTS stages
overlap, so their durations must not be summed. Provider timers include transport
and queueing. Offline first audio includes canonical resampling; live service
time to first byte (TTFB) ends before output resampling. Only responses with
committed audio count in valid-turn summaries; silent regions do not become turns.

An optional passive server trace (`NVA_EVAL_TRACE_DIR`) supports the comparison
without changing pacing or provider requests. The reporter verifies actual
source equality and session identity. Missing measurements remain null;
incomplete traces cannot establish response associations.
The [metric reference](offline-processing-reference.md#server-tracing-and-metric-definitions)
defines exact boundaries and report files.

Deterministic tests cover packetization independence, source resets, ASR draining,
transcript ownership, future-context suppression, overlapping generation,
interruption, history commitment, silence, final tails, malformed upload,
disconnect, and watchdog failures. Client and trace tests cover delayed replies,
cancellation ownership, source/session pairing, and partial observations.
The completed composite inspection validates recording and measurement
integrity. Paired benchmark cases and official scoring provide the next
validation layer.

## Limits and Extensions

Bulk-ASR availability measurements include backlog from rapid input and must
retain their recorded timing scope. Replay can generate work that later
interruptions discard, so compute totals do not reproduce live cancellation
savings.
Time-sensitive tool results and independent generation can differ between runs.

The proposed mode targets faster accuracy and task-completion evaluation.
Real-world latency and concurrency benchmarks still require live execution.
Further provider modes, examples, and transport gateways need adapters that
preserve the same causality, completion, interruption, and recording contracts.
