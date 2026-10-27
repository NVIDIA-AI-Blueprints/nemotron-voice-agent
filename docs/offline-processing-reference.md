# Offline Processing Implementation and Evaluation Reference

This reference describes the incremental runtime selected by the factory and
accompanies the [Offline Processing for Nemotron Voice Agent proposal](12-offline-processing.md).
The [dated progress log](offline-processing-progress.md) preserves earlier
implementations, experiments, and changes in direction.

The proposal baseline is standard continuous speech-to-text (STT) with `stop_history=320` (320 ms) and no force-EOU injection. Verify effective settings in both modes before paired evaluation.

Use the following sections for detailed implementation and evaluation work.

- [Timing and scheduling](#keep-two-explicit-time-domains)
- [Modules and extension contracts](#implementation-overview)
- [Configuration and recording protocol](#configuration-and-lifecycle)
- [Server traces and metric scopes](#server-tracing-and-metric-definitions)
- [Client commands and paired reports](#running-and-reproducing)
- [Verification and tests](#verification-and-tests)

## Keep Two Explicit Time Domains

The two time domains have distinct responsibilities.

| Domain | Representation | Responsibility |
| --- | --- | --- |
| Recording timeline | Absolute integer samples; zero is the first source sample | Speech, turn endpoints, response placement, interruption, silence, and recording extent |
| Observed execution time | Monotonic durations, displayed in milliseconds | Provider delays, upload, processing, publication, and failure watchdogs |

Upload progress does not advance conversation time. Provider execution remains
measured because a response cannot be placed before its input and generated
audio are available. Watchdogs retain execution-time deadlines, but they only
fail stalled work. They cannot end a user turn or insert source silence.

The design replaces the following timing dependencies.

| Existing Behavior | Offline Rule |
| --- | --- |
| Silero speech and silence thresholds | Preserve fixed-window audio counting |
| Periodic Silero state reset | Reset after the corresponding source samples |
| User-turn timeout waits and watchdogs | Run the shared Pipecat policy against a sample clock |
| Smart Turn retained audio buffers | Require a separate adapter; current runtime supports Silero only |
| ASR inactivity keepalive | Send only source PCM and explicitly drain the finite request |
| Output pacing and queue-empty silence | Schedule PCM spans and zero-fill uncovered positions |
| Bot-speaking and TTS idle notifications | Preserve native frame handling and advance output deadlines by sample offsets |
| Conversation commitment after output | Feed actually delivered text/control frames to the native assistant aggregator |
| Recorder arrival timestamps | Record ingress PCM and positioned committed bot spans |
| Session inactivity | Require explicit input completion and explicit processing completion |

## Provider Availability and Playback

The scheduler consumes VAD checkpoints and continuous-ASR callbacks while their
producers remain active. Actual Pipecat user aggregation decides when accumulated
transcripts complete a turn and produce an LLM request. ASR finalization alone
does not establish conversational end-of-turn. There is no word-to-VAD ownership
join, whole-file recognition barrier, or whole-response trace barrier.

Each actor publishes a complete callback batch and its future availability bound
atomically. The scheduler checks these bounds before advancing the sample clock.
Bounds describe the selected observed-processing schedule; they are not backend
completeness acknowledgements or recovered realtime arrival times.

The ASR adapter maps a cumulative processed prefix to an actually supplied
source chunk. It charges each observed processing interval once:

```text
interval_i = max(0, arrival_i - max(previous_arrival, required_chunk_supplied_at))
availability_i = max(required_source_prefix_i, availability_previous) + interval_i
```

The calculation uses one time unit internally, retains integer nanoseconds, and
rounds forward when exposing sample offsets. Empty and coalesced results also
advance the ordered processing clock. Before source supply completes, advancement
is conservative. After supply EOF, pending-work bounds advance within this model.
Response-stream EOF is the terminal completeness barrier; neither source
exhaustion nor an empty queue substitutes for it.

Native NVIDIA LLM and TTS services remain active across responses. Their frames
arrive incrementally, so ASR, LLM, and TTS work can overlap. The output adapter
retains native `MediaSender` buffering and FIFO control order. WebSocket writes
send before their pacing wait, so write completion and audible PCM completion
are separate positions. The recording placement rule is:

```text
chunk_start = max(socket_delivery_sample, previous_presentation_end)
chunk_end = chunk_start + canonical_chunk_sample_count
```

The shared turn policy triggers interruption at its detected sample. It cancels
native generation, waits for acknowledgement, fences late callbacks, and clears
queued output. Recording clips client-buffered PCM using the selected playback
cancellation policy. Already delivered text/control frames retain their native
effect on assistant history, including text delivered before audible PCM.

The manifest identifies `serial_chunk_progress_with_incremental_observed_processing`
for ASR availability and `incremental_source_chunks_plus_observed_bulk_compute`
for session timing. Observed intervals include transport, queueing, and batching.
They do not isolate GPU inference or reproduce an independent online run.

## Recording and Conversation Invariants

The scheduler and recorder enforce the following rules.

* Input becomes immutable after sample-count, packet-count, and SHA-256 checks.
  Source silence remains ordinary PCM; a disconnect cancels the session.
* Future source speech or transcripts cannot enter a response history early.
* One continuous ASR request consumes the whole recording and drains at EOF.
* Ordered interim/final transcript events enter the shared user aggregator;
  ASR word timestamps do not assign conversational ownership.
* The native aggregators own context commits. Response generation receives the
  current request snapshot, and stale generation callbacks cannot affect a successor.
* Bot spans are ordered and nonoverlapping. Every uncovered sample is zero.
* Output spans at least the input duration and extends for a final bot reply.

A recording with 300,000 ms of source audio therefore produces at least
300,000 ms of bot output, including silence. Evaluation adapters can additionally
crop a source-horizon view when a benchmark requires a fixed interval.

## Implementation Overview

Small contracts separate scheduling policy from transport, providers, and files.
The first integration currently lives in `src/examples/generic/offline/`.

| Module | Responsibility |
| --- | --- |
| `models.py` | Immutable source chunks, sample positions, and committed-span records |
| `detection.py` | Source-indexed Silero decisions, periodic state resets, and per-window checkpoints |
| `actors.py` and `recognition.py` | Atomic callback batches, availability bounds, continuous Riva ASR, and explicit result-stream drain |
| `aggregation.py` | Actual Pipecat user/assistant aggregation with sample-clock policy waits |
| `native_response.py` | Persistent native NVIDIA LLM/TTS processors, interruption acknowledgements, and generation fencing |
| `incremental.py` | Session scheduling across active source, provider, policy, and output actors |
| `output_stream.py` and `playback.py` | Native FIFO buffering, output deadlines, separate WebSocket write/playback clocks, and cumulative recording resampling |
| `engine.py`, `causal.py`, and legacy provider helpers | Earlier finite-tape implementations retained as references; not the factory-selected runtime |
| `recorder.py` | Span validation, zero-filled rendering, WAVs, and audit records |
| `protocol.py` and `websocket.py` | Bounded upload, offset validation, explicit EOF, cancellation, publication, and failure watchdogs |
| `factory.py` | Generic Assistant prompt/service composition, supported-scope guards, and session-owned resources |
| `src/examples/shared/evaluation_trace.py` | Optional passive timing observations for the normal live pipeline, separate from offline scheduling |

These boundaries apply the SOLID principles directly. Each module owns one
responsibility. Provider protocols are narrow and keep the scheduler independent
of Riva, OpenAI, and WebSocket classes. New implementations must preserve the
same source, availability, completion, and commitment invariants. An additional
transport or recorder consumes the same committed result. No global Pipecat
clock is patched.

To integrate another example, supply a composition factory and adapters that
satisfy those contracts. Keep example-specific configuration and provider
lifecycles outside the scheduler. An OpenAI Realtime gateway integration additionally
needs codec conversion, response completion, cancellation, and conversation
truncation tied to scheduled playback.

## Configuration and Lifecycle

The first integration supports a fixed recording, continuous Riva ASR,
Silero-only turn detection, an HTTP chat-completions LLM, stitched Magpie TTS,
and a prompt without tools. It resolves the Generic Assistant's model selections,
language, punctuation, voice, temperature, and token settings. A configuration
fingerprint records the resolved choices.
Automatic introduction is disabled for finite-session processing.

Offline ASR keeps `RecognitionConfig.model` empty, matching native
`NvidiaSTTService`. For NVIDIA-hosted ASR, `function-id` metadata selects the
deployment; the catalog model label is not an RPC selector.

Select `prompt_key=generic_assistant_without_tools`. History summarization is
explicitly rejected when its native eligibility condition is reached. The paired
subset disables summarization in both modes with `CHAT_HISTORY_RECENT_TURNS=0`;
this does not validate the default summarization policy.

Offline synthesis requests the same native 22,050 Hz TTS output as the live
pipeline, using `PIPELINE_AUDIO_OUT_SAMPLE_RATE`. Native buffered chunks are
converted to canonical 16 kHz recording PCM using cumulative sample counts. The
manifest records `tts_sample_rate`, `output_sample_rate`, and `tts_resampling`.
The paired live gateway requests 16 kHz client PCM from its existing output
conversion path. Matching the native synthesis rate avoids changing the TTS
request merely to simplify comparison. Recording conversion does not add a
synthetic streaming-resampler delay to provider readiness. First provider PCM,
first packed chunk, transport delivery, and presentation remain separate events.

Source detection uses the shared turn settings.
`USE_SILERO_VAD_TURN_DETECTION=true` selects the required Silero-only detection;
`SILERO_VAD_STOP_SECS` then defaults to 500 ms. These environment values accept
seconds even though reported measurements use milliseconds.

Tools, Smart Turn, streaming-input LLM URLs, per-sentence synthesis, zero-shot
TTS models, and zero-shot audio prompts are rejected explicitly. They need adapters
rather than silent changes in behavior.

## WebSocket and Recording Contract

The custom protocol is `nva.offline.v1`. The client starts with
`offline.session.start`, declaring 16 kHz mono PCM16, the input sample count,
and optional session configuration. The server acknowledges bounds before audio.

Binary packets contain a network-order `!4sQQI` header: magic, sequence,
absolute start sample, and sample count. `NVAI` identifies input and `NVAO`
identifies output. PCM is little-endian signed 16-bit mono. Input is contiguous
and ordered. Output can omit silence but must be ordered and nonoverlapping.

`offline.input.complete` supplies the sample count, packet count, and SHA-256.
The server validates and seals the source, processes it, publishes committed
audio and events, and sends `offline.session.complete`. Disconnect is not EOF.
Input after EOF is rejected. Failures send `offline.session.error`.
For an actual gRPC failure, this record and `failure.json` can include optional
`grpc_status`, a recognized `grpc.StatusCode` name such as `UNAVAILABLE`.
The field is omitted when no recognized status is available. Public error
records exclude provider details, exception strings, metadata, and headers.

`OFFLINE_AUDIO_DUMP_PATH` selects the server-owned recording directory; its
default is `audio_dumps/offline`. A UUID session directory contains these files.

| File | Contents |
| --- | --- |
| `source.wav` | Canonical ingress PCM, including silence |
| `full_output.wav` | Positioned bot PCM with zero-filled gaps and any final tail |
| `events.jsonl` | Source, recognition availability, interruption, playback, and context events |
| `manifest.json` | Extents, configuration fingerprint, messages, compute traces, and metrics |
| `failure.json` | Explicit failure status when processing does not succeed |

The sample client also writes source-horizon `output.wav` and a stereo
`comparison.wav`, with user input on the left and bot output on the right.

The `effective_configuration_v1` capability adds the resolved, redacted runtime
`configuration` to `offline.session.complete`, beside `configuration_fingerprint`.
Adapters can verify the fingerprint and required model/settings against this
public configuration, rather than trusting their requested values. This records
the application's resolved request settings, not an independent backend attestation.

### Raw LLM Text for Benchmark Adapters

The acceptance message advertises `raw_llm_text_events_v1`. Clients that score
native pre-TTS text must require this capability; older class-name-only traces
cannot reconstruct that candidate reliably.

Each `provider_observation` inside an `offline.event` records `source`,
`direction`, `frame`, `response_id`, and its logical `sample`. Only downstream
`LLMTextFrame` observations from `source=llm` include the exact `text` fragment.
Preserve fragment order, whitespace, and response identity. Upstream frames,
TTS-normalized text, and thought frames do not supply candidate text. Cancelled
generations cannot contribute stale callbacks to a successor.

VoiceBench adapters must apply the existing online collector's interruption,
epoch-reset, and first-eligible text-completion rules to these ordered events.
Input upload EOF does not mean that the logical conversation has consumed the
entire recording. Do not substitute played-TTS text, the last assistant message,
or the final context for the selected completion epoch.

This candidate differs from the October 23 AA-BigBench comparison, which judges
independent ASR of the complete committed output WAV. Aligned audio remains
diagnostic evidence when a benchmark scores native LLM text. The planned
VoiceBench CI run uses offline concurrency 1 and Lightning with thinking
disabled; no completed VoiceBench run is claimed here.

## Server Tracing and Metric Definitions

Set `NVA_EVAL_TRACE_DIR` on the server to retain passive live-pipeline traces.
Without that setting, the normal provider classes and frame flow are unchanged.
The observer never creates audio, changes pacing, or makes provider requests.
It counts PCM, buffers bounded events, and writes a session JSON file after the
worker stops. Trace-file failures do not fail the voice session.

The live trace records ASR request yields and existing response callbacks,
LLM frames, TTS text submission and generated PCM, stitched-worker completion,
VAD and semantic turn events, and output frames after their transport write.
Context IDs and request indices retain associations established by observed
frames. They do not prove transcript completeness or text ownership. Absent
events remain null. Every trace includes `timing_scopes` for its boundaries.
Observer callbacks can run after frame pushes. The report therefore calculates
LLM frame intervals from their original pipeline timestamps. Provider hooks use
a separate clock origin, so the report does not subtract a TTS-hook timestamp
from an LLM-frame timestamp without clock calibration. Compare actual native
request logs when exact dispatch contexts matter: an observer's later context
snapshot can already include another frame. Delivered TTS text also does not
prove that a client heard it.

All human-readable durations are milliseconds. Raw manifests retain sample
positions and nanosecond timers. Stages overlap and their durations must not be
summed into an end-to-end total. The following scopes distinguish measurements
that have similar names but different start or end events.

| Metric | Offline Scope | Normal Live Scope |
| --- | --- | --- |
| Server E2E | Sealed source to precompletion for the entire file, after processing, recording, and publication | Pipecat inferred user speech end to `BotStartedSpeakingFrame`, per reported turn |
| ASR first text | Request start to first nonempty transcript; continuous-stream firsts cover the recording | Existing nonempty-results callback or final-text frame; Pipecat ASR TTFB starts at inferred user speech end and ends at transcript finalization |
| LLM first text | Response submission to the native queue through first nonempty LLM text; includes queue delay | Service start frame to first nonempty text frame; Pipecat TTFT can instead end at a choices-bearing role-only delta |
| LLM completion | Response submission through native LLM response-end observation | Start to end at the service frame boundary, including local request work |
| TTS first audio | Response submission through first native PCM; first packed output chunk is recorded separately | Filtered text entering `run_tts` to native PCM entering the TTS context queue, before output resampling and paced output |
| TTS completion | Native serialized response end plus worker drain; retain native processing metrics separately | Stitched-worker completion, including later text waits; cancelled or returned workers do not by themselves establish successful RPC EOF |
| Output observation | Committed PCM at scheduled sample offsets | Output frame after transport write; client receipt and playback are separate observations |
| Rendered turn gap | Latest acoustic speech end to first committed bot PCM; VAD stop-confirmation offset is separate | Reported separately from server latency; client playback uses packet receipt and queued PCM |

Offline batch E2E and live turn E2E cover different scopes. Likewise, ASR,
LLM, and TTS observations must retain their boundary labels instead of being
presented as interchangeable timers. Provider measurements include transport and
backend queueing; none isolates GPU execution. Response-relative first PCM is
not a TTS-only inference timer. Preserve native provider TTFB labels separately,
and do not treat a cancellation-stopped timer as a successful first token.

Offline turn-gap breakdown separates endpoint wait, post-endpoint readiness and
scheduling wait, and response-request-to-first-PCM delay. Only responses with
committed PCM count in valid-turn summaries. Cancelled responses still contribute
to processing totals. EOF-flushed responses are identified separately from
natural endpoints.

The online comparison client retains the server's reported metrics in
milliseconds and writes its own receipt/playback measurements with separate
names. It models zero-prebuffer playback by placing each packet no earlier than
receipt or the preceding queued packet's end. Its WAV is a client playback
recording, not a server sample clock. It applies no network-delay correction or
manual time shift. A semantic turn waiting for a response or an active response
prevents idle completion. Collection waits until that work settles or the
absolute post-input deadline fails the run. It does not send extra audio, manual
commits, or `response.create`.

## Limits of the Comparison

Both modes use continuous ASR with the same STT baseline (`stop_history=320`,
320 ms) and no force-EOU injection. Source audio, including silence, determines
recognition progress. Offline EOF explicitly completes the finite source; the
online stream remains open during normal conversation.

Paired evaluation measures transcript availability and response timing using
each mode's recorded boundaries. It also verifies the effective 320 ms
configuration in both modes rather than inferring it from the proposal baseline.
Behavioral and scoring consistency against real-time streaming is being
validated.

Native interruption cancels work and fences late callbacks. Observed bulk
processing still differs from paced execution, and its schedule does not recover
the counterfactual online cancellation or response latency. The current source
recording is fixed independently of the bot; feedback-driven user simulation
and tools require additional adapters.

The [October 23 scored subset checkpoint](offline-processing-progress.md#october-23-2026-eight-question-scored-comparison)
records 32 successful captures and uniformly judged spoken answers. User-message
grouping matches all 16 pairs, while assistant context, wording, and some verdicts
differ. This small repeated subset does not establish behavioral equivalence.

This workflow targets benchmark turnaround for accuracy and task completion.
Offline throughput is not real-time concurrency capacity. A benchmark that scores
real-world response delay, such as latency-sensitive Full-Duplex-Bench criteria,
still needs its live evaluation. WAV integrity checks do not constitute an
official accuracy, Full-Duplex-Bench, or task-completion score.

## Running and Reproducing

Configure the normal Generic Assistant and start the application with its
catalogs and credentials.

```bash
EXAMPLE_SELECTION=generic-assistant PIPELINE_TLS=false \
  USE_SILERO_VAD_TURN_DETECTION=true CHAT_HISTORY_RECENT_TURNS=0 \
  uv run python src/server.py
```

Create a credential-free `config.json` selecting the supported prompt.

```json
{"prompt_key": "generic_assistant_without_tools"}
```

The following client sends an existing canonical recording without pacing.

```bash
uv sync --group benchmark
uv run python benchmarking_tools/offline-eval/client.py \
  --url ws://127.0.0.1:7860/api/ws/offline \
  --input session.wav --gap-seconds 0 --upload-speed 0 \
  --config config.json --output-dir offline_run
```

Multiple `--input` files can be separated by `--gap-seconds` and padded with
`--duration-seconds`. The client refuses to shorten source recordings.
`--upload-speed 1` tests paced input to this same offline endpoint; it does not
change the server's execution engine.

The batch runner starts an independent session for each sample.

```bash
uv run python benchmarking_tools/offline-eval/run_fdb_samples.py \
  --manifest cases.json --url ws://127.0.0.1:7860/api/ws/offline \
  --config config.json --output-dir fdb_offline_run
uv run python benchmarking_tools/offline-eval/extract_metrics.py \
  --input-dir fdb_offline_run --output-dir extracted_metrics --run-label offline
```

The manifest format is
`{"cases":[{"id":"case-1","input":"canonical.wav","provenance":{}}]}`.
Relative inputs resolve beside it. Supply 16 kHz mono PCM16 and record any
conversion in provenance. The optional configuration uses the existing Generic
Assistant session keys. A session created through `/api/session-config` can
instead be passed as `?session_id=...`; those process-local IDs are not supported
across workers. Direct handshake configuration avoids that lookup.

### Reproduce the Paired Comparison

Start the same current application with optional passive server tracing enabled.
The live gateway and offline endpoint use that application and its configured
model services. Configure the server catalogs so the flat configuration selects
the same models, prompt, voice, and generation settings in both modes.

```bash
NVA_EVAL_TRACE_DIR=audio_dumps/evaluation \
  EXAMPLE_SELECTION=generic-assistant PIPELINE_TLS=false \
  USE_SILERO_VAD_TURN_DETECTION=true CHAT_HISTORY_RECENT_TURNS=0 \
  ENABLE_WELCOME_MESSAGE=false AUDIO_OUT_10MS_CHUNKS=40 \
  uv run python src/server.py
```

Prepare shared input files, run offline inference, and then replay the same
prepared files through the normal OpenAI Realtime gateway at their recorded rate.
The preparation command makes no model requests. Use new or empty output
directories for each invocation.

```bash
uv run python benchmarking_tools/offline-eval/compare_live.py \
  --manifest cases.json --prepare-only --trailing-silence-ms 4000 \
  --output-dir paired/prepared
uv run python benchmarking_tools/offline-eval/run_fdb_samples.py \
  --manifest paired/prepared/cases.json \
  --url ws://127.0.0.1:7860/api/ws/offline --config config.json \
  --output-dir paired/offline
uv run python benchmarking_tools/offline-eval/compare_live.py \
  --manifest paired/prepared/cases.json \
  --live-url ws://127.0.0.1:7860/v1/realtime --config config.json \
  --offline-results-dir paired/offline --output-dir paired/online
```

The live client verifies paired source PCM equality before replay. It requests
16 kHz PCM for both client directions. It writes input/bot comparison WAVs,
`offline_left_online_right.wav`, transcripts, send/receive traces, accepted
session settings, and server-reported latency tables. The optional
`REALTIME_API_KEY` environment variable supplies a live gateway bearer token;
credentials are not saved in session configuration.

After the live session closes, use the passive server trace files to build the
paired report. If the client runs on another host, first copy the completed
trace files to the path passed as `--server-traces-dir`.

```bash
uv run python benchmarking_tools/offline-eval/report_comparison.py \
  --offline-results-dir paired/offline --online-results-dir paired/online \
  --server-traces-dir audio_dumps/evaluation --output-dir paired/report
```

The reporter makes no provider requests. It requires successful runs with the
same case IDs, compares the actual input PCM bytes and format, and matches each
accepted live session ID to exactly one server trace. Missing runs, unequal
sources, and ambiguous trace matches fail explicitly. Its outputs include these
files, with all time values in milliseconds.

| Report File | Contents |
| --- | --- |
| `comparison.json` | Complete report, metric definitions, source hashes, file inventory, and warnings |
| `cases_ms.csv` | Per-case completion times, offline batch processing, response counts, scoped server summaries, and input-pacing lateness |
| `offline_valid_responses_ms.csv` | Offline committed-response timings and turn-gap breakdown |
| `online_valid_responses_ms.csv` | Live committed responses, server-reported metrics, and client receipt/playback observations |
| `asr_workers_ms.csv` | Live ASR worker lifetime and callback timings from worker start or first supplied audio |
| `llm_requests_ms.csv` | Live service-frame first-text and completion timings, with explicit ownership information |
| `tts_contexts_ms.csv` | Live text submission, first generated PCM, worker lifetime, and cancellation/outcome scope |
| `server_messages_ms.csv` | Server latency observations in their original event scope |
| `wav_inventory_ms.csv` | Native WAV paths, durations, formats, and file/PCM hashes |
| `transcripts.json` | Offline committed conversation and live transcription events |

Only responses with committed PCM populate valid-response tables. Empty source
regions do not become turns. Request indices, synthesis context IDs, and wire
response IDs remain separate unless the trace proves their association. Equal
ordinal counts are insufficient. Missing observations remain null or blank.

The per-case online/offline client completion ratio includes setup, publication,
and the online idle drain. The report shows these components separately. This
ratio describes evaluation turnaround, not inference acceleration or equivalent
response latency. It does not infer task accuracy or acoustic scores.

For official live benchmark procedures, refer to the
[Full-Duplex-Bench client](../benchmarking_tools/Full-Duplex-Bench-Eval/README.md)
and [scaling benchmark](../benchmarking_tools/scaling-perf/README.md).

## Verification and Tests

The offline tests use deterministic provider traces to verify the architecture
without model or judge endpoints.

```bash
uv run pytest tests/unit/test_offline*.py
```

Coverage includes packetization-independent detection, source resets, partial
EOF, continuous ASR draining, atomic callback batches, overlapping native
generation, acknowledged interruption, stale callback fencing, silence,
final-tail extension, malformed uploads, disconnect, and cleanup failures.
Controlled tests compare every request and final context with unmodified native
aggregators receiving identical events. An integration test exercises the engine,
queued native LLM/TTS, and output chain with only model I/O replaced.

Production source `4220eb5` passes 888 tests and 178 subtests, with 3 tests
skipped. Ruff lint and formatting checks pass for 289 Python files. The
[progress log](offline-processing-progress.md) distinguishes local checks,
fresh real-provider captures, and the later scored subset.

With raw-text event instrumentation, the complete suite passes 890 tests and
178 subtests, with 3 tests skipped. The targeted provider/engine/WebSocket
checks pass 49 tests and 40 subtests; Ruff formatting covers 290 Python files.

### Completed Composite Inspection

An unscored composite inspection completed on Pipecat 1.12.0 through the normal
`/v1/realtime` gateway and `/api/ws/offline`. Both paths use the same source PCM
and local model services, standard continuous STT with `stop_history=320`, and
no force-EOU injection. Recording checks verify source and runtime provenance,
output extents, offline server/client PCM equality, silence outside committed
offline spans, and stereo channel alignment.

Behavioral and scoring consistency against real-time streaming is being
validated. Full-Duplex-Bench score validation is in progress and follows sharing
the design proposal.

The exported evidence bundle keeps these files together, with paths relative
to its root.

| Artifact | Contents |
| --- | --- |
| `run_report.md` | Measured results, timing scopes, limitations, and links to the recordings |
| `offline/composite_300s/` and `online/composite_300s/` | Input, aligned bot output, and user-left/bot-right comparison WAVs |
| `comparison/` | Paired server metrics, transcripts, source hashes, and WAV inventory |
| `runtime_versions.json` and `source-provenance.json` | Installed dependencies, effective STT settings, and deployed application revision |
| `model-runtime.json` | Model deployment configuration |
| `verification.json` | Recording-integrity checks for both completed paths |

Keep model configuration differences in the report when comparing older runs.
The historical composite deployment uses TTS batch size 8; the October 23 scored
subset uses 32. Differences between these runs cannot be attributed solely to
the Pipecat version. Official benchmark scorers separately evaluate semantic
accuracy and task completion.
