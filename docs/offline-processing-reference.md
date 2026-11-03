# Offline Processing Implementation and Evaluation Reference

This reference describes the incremental runtime selected by the factory and
accompanies the [Offline Processing for Nemotron Voice Agent proposal](12-offline-processing.md).
The [dated progress log](offline-processing-progress.md) preserves earlier
implementations, experiments, and changes in direction.

The default configuration uses standard continuous speech-to-text (STT) with
320 ms of stop history and no force-EOU injection. Set the server environment
variable `ASR_STOP_HISTORY_MS` to override stop history in both Generic online
and offline pipelines. Use an integer from 1 through 10,000 ms. This setting
does not change the Silero silence threshold or the sample clock. Verify
effective settings in both modes before paired evaluation.

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
audio are available. Execution watchdogs retain wall-clock deadlines and fail
stalled work; they cannot end a user turn or insert source silence. Native
conversation-policy waits use the sample clock. These include the user-turn
stop watchdog, which retains its upstream ability to end a nonspeaking turn.

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
| Finite-session completion | Require explicit input completion and explicit processing completion |
| Proactive live activity checks | Not implemented offline; require a shared adapter or an explicitly matched profile |

## Provider Availability and Playback

The scheduler consumes VAD checkpoints and continuous-ASR callbacks while their
producers remain active. Actual Pipecat user aggregation decides when accumulated
transcripts complete a turn and produce an LLM request. ASR finalization alone
does not establish conversational end-of-turn. There is no word-to-VAD ownership
join, whole-file recognition barrier, or whole-response trace barrier.

Matching the Silero silence threshold alone does not guarantee identical context
commits. Final ASR arrival, VAD decisions, queued processing, interruptions, and
candidate selection can change which text reaches a native context boundary.
Compare the delivered events and immutable request messages when investigating
a difference. Equal concatenated ASR words do not establish equal request
contexts, and these timing differences are not all proven fixed.

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
Processed-counter checks allow bounded float32 rounding at the counter's magnitude,
with a one-sample minimum tolerance. This does not establish transcript completeness.
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

For completed full-session recordings, the scheduler and recorder enforce the
following rules.

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
| `models.py` | Immutable sources, bounded upload-prefix readers, sample positions, and committed-span records |
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
| `pool_supervisor.py` | Optional front-door startup and owned private single-worker processes |
| `pool_proxy.py` | Exclusive stream admission, bounded forwarding, and authenticated lease draining |
| `worker.py` | Private request guard, one offline handler per process, and handler-return status |
| `diagnostics.py` and `pool_telemetry.py` | Optional bounded scalar observations for native execution and exclusive-pool lifecycle |
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
Temperature resolves from an explicit request, prompt metadata, then the catalog;
native extra parameters retain their overriding precedence.

TTS synthesis mode resolves from the request, then the selected catalog, then
the pinned native `per_sentence` default. Offline requires an explicit `stitched`
selection in the request or catalog. Missing settings that resolve to `per_sentence`
fail before detector or provider initialization.

Offline ASR keeps `RecognitionConfig.model` empty, matching native
`NvidiaSTTService`. For NVIDIA-hosted ASR, `function-id` metadata selects the
deployment; the catalog model label is not an RPC selector.

Select `prompt_key=generic_assistant_without_tools`. History summarization is
explicitly rejected when its native eligibility condition is reached. The paired
subset disables summarization in both modes with `CHAT_HISTORY_RECENT_TURNS=0`;
this does not validate the default summarization policy.

Offline does not instantiate Generic Assistant's proactive activity-check processor.
The live registry sets a first warning after 480,000 ms of inactivity, a second
warning interval of 30,000 ms, and a warning-completion timeout of 45,000 ms. These
checks can generate LLM/TTS warnings, mutate context, and disconnect the session.
Disabling welcome audio does not disable them. The recorded short-case comparisons
do not exercise this path. Long-session parity requires a shared sample-clock adapter
or an explicit profile with equivalent behavior in both modes; finite-source EOF does
not replace these actions.

Offline synthesis requests the same native 22,050 Hz TTS output as the live
pipeline, using `PIPELINE_AUDIO_OUT_SAMPLE_RATE`. Native buffered chunks are
converted to canonical 16 kHz recording PCM using cumulative sample counts. The
manifest records `tts_sample_rate`, `output_sample_rate`, and `tts_resampling`.
The paired live gateway requests 16 kHz client PCM from its existing output
conversion path. Matching the native synthesis rate avoids changing the TTS
request merely to simplify comparison. Recording conversion does not add a
synthetic streaming-resampler delay to provider readiness. First provider PCM,
first packed chunk, transport delivery, and presentation remain separate events.

Generic Assistant's stitched WebSocket TTS uses
`explicit_provider_eof_pipecat_1_12_compat` in both modes. The helper requires
exactly Pipecat 1.12.0 and supplies `stop_frame_timeout_s=None`, so an idle audio
gap does not complete a context successfully. This constructor behavior is undocumented;
a supported upstream API is still needed.

Full offline response completion waits for provider-worker EOF and native
audio-context drain, then rechecks errors. Its execution watchdog fails stalled
work while drain waits race error and interruption signals.

The shared `ExplicitEOFCompletionMixin` applies to ordinary live WebSocket,
traced WebSocket, and offline stitched synthesis. If native RPC failure omits context
closure, it closes only that response's owned context, unless native EOF already
requested closure. It preserves native error frames and recovery for later generations.
Native provider request loops and request serialization remain inherited.

A provider-inactivity failure bound resets only after a successful append of
nonempty owned provider PCM. Empty responses do not extend it. Offline sets this
bound to 60,000 ms; the shared live WebSocket default remains 600,000 ms. Expiry
uses native RPC cancellation and emits an error frame. It cannot end user speech
or report silent success. The existing offline total-generation watchdog remains
600,000 ms, with a separate scope. None of these limits is a benchmark's deadline
for receiving the first LLM token.

The live path retains native session/evaluator cancellation bounds without adding
a total-generation watchdog. WebRTC and live per-sentence synthesis bypass this
compatibility wrapper and retain native idle handling.

Offline configuration records `tts_context_completion_policy`,
`tts_provider_idle_timeout_seconds=60`, and `tts_config_timeout_seconds=60`.
The live startup log and optional evaluation trace identify the completion
policy. The earlier full-audio readiness gates exercise
explicit EOF with real services, but do not validate the later lifecycle corrections.
Their focused model-free checks and remaining gates are recorded in the
[merge-review checkpoint](offline-processing-progress.md#october-30-2026-merge-review-and-validation-gates).
Refer to the
[rebased validation checkpoint](offline-processing-progress.md#rebased-source-validation-and-native-policy-contract)
for the earlier real-service evidence; the compatibility API remains undocumented.

Source detection uses the shared turn settings.
`USE_SILERO_VAD_TURN_DETECTION=true` selects the required Silero-only detection;
`SILERO_VAD_STOP_SECS` then defaults to 500 ms. These environment values accept
seconds even though reported measurements use milliseconds.

Tools, Smart Turn, streaming-input LLM URLs, per-sentence synthesis, zero-shot
TTS models, and zero-shot audio prompts are rejected explicitly. Effective speaker
diarization and prompts marked for multiple speakers are also unsupported. These
features need adapters rather than silent changes in behavior.

Cleanup attempts every owned resource release even when an earlier release fails.
Repeated cleanup retries failed releases without repeating completed ones. If execution
and cleanup both fail, the execution error remains primary; cleanup does not turn a
failed run into success.

### Offline Native Service Startup

The offline adapter fetches the native TTS configuration without blocking the
application event loop. Only the owned `GetRivaSynthesisConfig` RPC and result
wait run on a worker. Its 60,000 ms deadline includes worker queueing. Cancellation
retires that setup and cancels its owned RPC; it does not claim to stop a Python
thread. Late results cannot install a configuration into a retired service.

The adapter then runs the native setup on its owning event loop using the fetched
configuration. Native queues, task management, synthesis requests, and response
loops retain their execution policy. This compatibility boundary requires exactly
Pipecat 1.12.0. The normal online service setup is unchanged.

Text-only collection still initializes the native LLM and TTS services before
starting the ASR and VAD actors. Returning at LLM completion avoids waiting for
TTS audio completion; it does not remove provider initialization from the server
path.

The default `input_processing=sealed` path constructs providers after the server
validates and seals input EOF and its source hash. The experimental
[`during_upload` text mode](#process-text-input-during-upload) starts owned setup
after the first validated nonempty packet. It can overlap subsequent upload,
but a fast upload can finish before native setup and source analysis are ready.
An acknowledgement-only capability probe does not construct providers in either
path. The normal online pipeline is unchanged.

`GenericOfflineRuntime.prepare()` is an optional owning-loop warmup hook.
Concurrent callers join one preparation task; the runtime owner retires it
during cleanup. The sealed default leaves native startup in its existing engine
path. A first preparation call after execution starts is rejected.

## WebSocket and Recording Contract

The custom protocol is `nva.offline.v1`. The client starts with
`offline.session.start`, declaring 16 kHz mono PCM16, the input sample count,
and optional session configuration. The server acknowledges bounds before audio.
The optional `completion` field selects `full_session` (default) or `text`.
The optional `input_processing` field selects `sealed` (default) or the
experimental, text-only `during_upload` path. The acknowledgement echoes the
selected `input_processing` value.
Configured offline clients send their prompt and configuration in this start
message. They do not need a preceding `POST /api/session-config` or a
process-local session ID.

Binary packets contain a network-order `!4sQQI` header: magic, sequence,
absolute start sample, and sample count. `NVAI` identifies input and `NVAO`
identifies output. PCM is little-endian signed 16-bit mono. Input is contiguous
and ordered. Output can omit silence but must be ordered and nonoverlapping.

`offline.input.complete` supplies the sample count, packet count, and SHA-256.
In default `full_session` mode, the server validates and seals the source,
processes it, publishes committed audio and events, and sends
`offline.session.complete`. Disconnect is not EOF.
Input after EOF is rejected. Failures send `offline.session.error`.
For an actual gRPC failure, this record and `failure.json` can include optional
`grpc_status`, a recognized `grpc.StatusCode` name such as `UNAVAILABLE`.
The field is omitted when no recognized status is available.

For native response failures, `provider_source` identifies an owned `llm` or
`tts` processor only when the error frame supplies that identity.
`observed_callback_boundary` records where the error was received (`llm` or
`tts`); it does not identify the originating provider. `provider_error_category`
contains a recognized `ErrorCategory` name. Fields without a recognized value
are omitted. Public error records exclude provider details, exception strings,
metadata, and headers.

Owned TTS operations can additionally report these observations.

| Field | Allowed Values |
| --- | --- |
| `observed_provider_operation` | `tts_audio_context`, `tts_context_audio_outcome_no_audio`, `tts_context_audio_outcome_with_audio`, or `tts_response_stream` |
| `observed_generation_state` | `active`, `interrupted`, `closed`, `missing`, or `unknown` |
| `observed_provider_rpc_state` | `active`, `complete`, or `unavailable` |
| `observed_provider_rpc_status` | A typed `grpc.StatusCode` name, only when the identified call is confirmed inactive |

RPC observations are separate from the original exception's `grpc_status`.
An active call does not have its status queried. Missing, cleared, ambiguous,
or untyped call context yields `unavailable`, which does not establish a backend
failure. A completed call can omit status when no typed value is available.
These fields describe observed work and context; they do not establish a cause.

`OFFLINE_AUDIO_DUMP_PATH` selects the server-owned recording directory; its
default is `audio_dumps/offline`. A UUID session directory contains these files.
The session directory uses permissions `0700`; recordings and JSON artifacts
use `0600`. Publication uses private temporary files and atomic replacement,
without changing the process umask. These files can contain conversation audio
and text.

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
The `asr_stop_history_ms` field records the resolved shared STT setting, including
an explicit `ASR_STOP_HISTORY_MS` override.

### Incremental Text Completion

Clients selecting `completion=text` must require `incremental_text_events_v1`.
They upload the same finite source without realtime pacing. With the default
sealed input path, processing starts after source validation. After sealing,
the server sends `offline.runtime.configuration`, including `protocol`,
`configuration`, and `configuration_fingerprint`, before incremental
`offline.event` records. Verify configuration before accepting candidate text.
This mode sends JSON events, not output PCM.
Its bounded writer fails with `event_queue_overflow` if the client cannot keep
up; it does not silently discard events. If runtime execution drains first,
`offline.events.complete` ends the stream without a full-session result.

A benchmark client can apply the native online collector's first-eligible text
epoch rules as events arrive, then return and disconnect without waiting for
TTS or full-session completion. On disconnection, the server cancels and awaits
owned tasks and native cleanup under existing session/evaluator cancellation
limits. A completed WebSocket close handshake does not attest provider cleanup
or EOF. Text mode adds no separate per-generation watchdog. A session error
received before text completion remains a failure.
A text-mode disconnect can produce `cancellation.json`; this records
cancellation, not candidate success.
Text success does not attest a complete output WAV, final recording alignment,
provider cleanup, or success of later provider work. The default full-session
path retains those separate completion and recording boundaries.

Keep a client's response-start deadline separate from its response-completion
deadline. The first measures elapsed wall time until nonempty raw LLM text
arrives. The second bounds an answer that has started streaming. ASR events,
configuration records, and an LLM start frame do not establish that answer text
has arrived. A client deadline does not change the server's turn policy.
When a client measures this deadline after upload, it starts after sending the
source and any finite tail. That boundary does not attest completed ASR work.
The VoiceBench evaluator exposes the opt-in offline product setting
`product.offline_response_start_timeout_s`. Its default is unset. A value of `10`
bounds the first eligible nonempty raw LLM text to 10,000 ms after source and
tail upload. In the sealed path, this budget includes subsequent provider
initialization, ASR, turn processing, and LLM work before that text arrives.
Work that overlaps upload in `during_upload` has a different measured scope.
Retain its actual timestamps. The option does not replace the existing receive
or total-operation budgets, and it does not change the online collector.
Record the selected limit with coverage and retry counts. This offline option
does not make the online and offline timeout policies identical.

Input silence and wall-clock deadlines also have different scopes. Any silence
appended by a benchmark client becomes part of the declared source PCM. For
example, a 1,500 ms tail contains 24,000 zero-valued samples at 16 kHz. It does
not instruct the server to wait 1,500 ms of execution time. The endpoint does
not continuously append silence while a client waits for an answer.
After explicit input completion, a client keeps the session open for the
response-start budget and the separate completion limits. Waiting 10,000 ms
for the first text does not require sending 10,000 ms of additional silence.

The [VoiceBench checkpoint](offline-processing-progress.md#october-30-2026-voicebench-repeatability-and-context-timing)
records completed real-service text cohorts. The
[rebased validation checkpoint](offline-processing-progress.md#rebased-source-validation-and-native-policy-contract)
records a later matched-source diagnostic. Neither establishes full-session
completion or benchmark-wide behavioral equivalence.

### Process Text Input During Upload

The experimental input mode overlaps native service setup and source analysis
with a finite upload. It is an unbenchmarked prototype; existing scored results
describe the sealed input path. Select it by adding these fields to an otherwise
valid session start:

```json
{
  "completion": "text",
  "input_processing": "during_upload"
}
```

These fields belong to the session start, outside `config`. Existing example
client commands retain the sealed default.

The declared input must be nonempty and finite. Before uploading PCM, require
`streaming_input_text_v1` in the server capabilities and an acknowledgement with
`input_processing: "during_upload"`. Reject a missing or different echo. Older
servers can ignore unknown start fields, so a requested field alone does not
establish that the mode is active. Omitting it preserves the sealed default.
The protocol remains `nva.offline.v1`; existing packet envelopes are unchanged.

The two input modes have these boundaries:

| Mode | Source Processing | Configuration and Text Publication |
| --- | --- | --- |
| `sealed` | Starts after explicit extent, packet-count, and hash validation | Follows validated input and source recording |
| `during_upload` | Starts setup after the first valid nonempty packet; ASR and VAD then await available canonical windows | Waits for the same final validation and source recording |

ASR and VAD use independent cursors over the same accepted PCM prefixes.
They retain fixed analysis windows across packet boundaries. An upload stall
does not supply silence or EOF; a final partial window becomes available only
after validated EOF. The declaration is a total extent, not received progress.

Native turn/context processing can submit LLM requests before seal. This work
and its generated events remain provisional. The server queues
events within its existing bounds and publishes configuration before events
after validation and source recording finish. A full gated buffer fails with
`provisional_event_overflow`; it does not discard events. After publication
starts, the existing `event_queue_overflow` bound applies. A bad hash, malformed
packet, upload timeout, or disconnect aborts the source and cancels owned work.
Blocked readers wake with an abort, not a successful EOF. Provider work already
performed on a rejected prefix cannot be undone and does not make the case
successful.

With optional trusted request tracing enabled, private capture can include
provisional LLM requests before seal. The source-identity snapshot waits for
validated EOF. An aborted upload can retain private request observations;
their presence does not establish input validation or case success.

The mode adds no continuous silence, TTS completion wait, or audio-gap completion
rule. A client still selects the first eligible native raw LLM response end.
Record the negotiated input mode and actual setup/upload/analysis timestamps
with any later comparison. No throughput gain, score equivalence, or latency
reduction has been established for this prototype.

Native readiness still precedes ASR and VAD startup. In this mode, engine-relative
physical timings begin after explicit preparation; WebSocket/session diagnostics
retain setup time. ASR callback intervals include observed transport and queueing;
they do not isolate model inference time.

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

The native `LLMFullResponseEndFrame` ends a raw LLM response epoch. A benchmark
collector applies its eligibility and interruption rules at that boundary.
It does not infer text completion from a 5,000 ms audio gap or TTS inactivity.

VoiceBench adapters must apply the existing online collector's interruption,
epoch-reset, and first-eligible text-completion rules to these ordered events.
Input upload EOF does not mean that the logical conversation has consumed the
entire recording. Do not substitute played-TTS text, the last assistant message,
or the final context for the selected completion epoch.

This candidate differs from the October 23 AA-BigBench comparison, which judges
independent ASR of the complete committed output WAV. Aligned audio remains
diagnostic evidence when a benchmark scores native LLM text. Full-session audio
evidence and early text completion have different completion scopes; neither
establishes benchmark-wide equivalence.

## Server Tracing and Metric Definitions

Successful full-session processing can include `metrics.native_provider_observations`,
collected after engine completion and provider cleanup. It contains integer
`total`, boolean `truncated`, and an `observations` list. The list retains the
first 64 captured `ErrorFrame` callbacks with one-based `observation_index`
values and the allowlisted native provider fields described above. `total`
counts all captured callbacks, including forwarded duplicates and closed
generations; `truncated` indicates omitted entries. These are observations,
not unique failure counts, and do not override completion or scoring.

### Offline Scalar Server Metrics

Set the trusted server environment variable `NVA_OFFLINE_SERVER_METRICS_DIR` to
a job-owned directory to retain bounded scalar measurements for offline sessions.
This observer is disabled by default and is separate from request tracing with
`NVA_EVAL_TRACE_DIR`. It does not change the normal online collector or add
provider requests.

Each private JSON file uses schema `nva.offline.scalar-metrics.v1`. Its random
`session_id` is also exposed as optional `server_metrics_session_id` in
`offline.runtime.configuration`, so a client can associate case evidence with
the correct server file. Verify the association rather than matching files only
by order. The source digest identifies the sealed PCM, including any client tail.

The observer retains server-monotonic event times, numeric duration summaries,
and allowlisted labels. It excludes transcripts, audio, request messages,
provider bodies, credentials, and exception messages. Inspect `truncated` and
`rejected_fields` before interpreting missing measurements. Persistence is
attempted inside the offline handler's finalizer, before its enclosing ASGI
handler returns. A missing file is an observability gap, not evidence that a
provider returned no events.

The measurements distinguish these boundaries.

| Boundary | Interpretation |
| --- | --- |
| ASR request, first supplied audio, callbacks, and stream EOF | Observed provider and transport lifetime; a first final segment does not establish complete utterance ownership. |
| ASR final callback and matching event send | Server scheduling and delivery observations, joined by sequence. A completed socket write does not establish client receipt. |
| `duration_summaries.vad_model` | Wall time inside the voice-confidence function, including conversion, model execution, and state reset. This is not isolated ONNX compute or thread CPU time. |
| `duration_summaries.vad_executor` | Wall time inside the full native analyzer body, including volume and state work. It excludes executor queueing and event-loop resumption. |
| `duration_summaries.vad_await` | Wall time around awaited audio analysis. Differences from the analyzer-body measurement include several scheduling and dispatch boundaries, not only executor wait. |
| TTS configuration queue, RPC, and native setup | Worker queueing, configuration fetch, and service setup remain separate from synthesis latency. |
| LLM context queue, first nonempty text, and native end | Generation-relative wall observations with response identity, separate from native TTFB metric semantics. |
| Native provider metrics | Typed `TTFBMetricsData`, `TTFATMetricsData`, and `ProcessingMetricsData` values retain their original processor and metric labels. They do not isolate GPU compute. |

Text completion can cancel the session before ASR EOF or later TTS observations.
Keep those fields absent or censored instead of completing them from client
timestamps. The scalar file does not attest full provider EOF or per-session
cleanup. Its execution-time observations do not advance the sample clock.
Recording adds observation overhead, so retain observer enablement with benchmark
provenance when comparing execution times.

Use the client case result and inner outcome events when checking success.
`session_failed` retains only the exception type. `session_cancelled` distinguishes
client disconnection from server cancellation. `session_finalizer_entered` and
the legacy `session_returned` observation mark entry to finalization before
persistence. The latter records `boundary=pre_persistence`; it does not attest
enclosing ASGI handler return. In text mode, a client
can already have accepted its answer before disconnecting and triggering cleanup.
None of these observations alone establishes full provider EOF or successful cleanup.

### Attribute Offline Client and Server Waits

The VoiceBench offline adapter adds scalar `offline_client_` timings to each
attempt. These use client monotonic wall time in seconds. Existing
`offline_upload_wall_s`, `offline_text_completion_wall_s`, and
`offline_total_wall_s` retain their earlier boundaries after preflight.
The new attempt origin begins before deployment and capability preflight, after
argument validation. These are overlapping observations, not values to sum
indiscriminately.

`offline_client_attempt_started_monotonic_s` records the configured client's
monotonic origin. Combine it with relative client points only after verifying
the client and server use the same host clock domain. Clock overrides used by
fixtures require their own scope. Host monotonic timestamps are neither UTC
timestamps nor provider TTFB clocks; logical source samples remain separate.

The client measurements distinguish these stages.

| Stage | Scope |
| --- | --- |
| Deployment and capability preflight | Deployment metadata request plus the separate ACK-only probe, with probe connect, start send, ACK wait, and close measured separately. |
| PCM preparation | Copy already decoded PCM16 and allocate its finite silence tail. Earlier file loading, decoding, and resampling are outside this client stage. |
| Actual WebSocket connect and native ACK | Connect to the selected endpoint, send the session start, and await the native acknowledgement. These intervals can include proxy admission and backend work. |
| PCM and EOF send | Send source and tail packets, construct their existing headers, compute the existing digest, and send explicit input completion. |
| First eligible raw token and native text end | Client receipt under the existing token/deadline and candidate policies. These intervals are not native provider TTFB. An earlier eligible token epoch can later be interrupted. |
| Close handshake | Await the existing bounded close operation. Socket close does not attest server cleanup. |

Unobserved stages remain absent. Failed, cancelled, or unsuccessful stages use
`_incomplete_wall_s` for their observed elapsed interval, without inventing a
completed stage or a zero. Existing failure partials retain these measurements.
The optional 10-second response-start budget still begins after source and tail
upload; instrumentation adds no silence, timeout change, or TTS wait.

Set trusted `NVA_OFFLINE_PROFILE=1` together with
`NVA_OFFLINE_SERVER_METRICS_DIR` to add bounded server profiling under
`nva.offline.profile.v1`. The profiler observes the existing native executor;
it does not select a different executor or modify Silero's algorithm.
`vad_executor_queue` separates native submission from actual callable entry,
including executor dispatch and OS scheduling. It does not prove that another
queued task caused the delay.
`vad_executor_finish_to_loop_resume` measures return to the owning loop.
`vad_model_thread_cpu` and `vad_executor_thread_cpu` measure calling-thread CPU
within their respective bodies. They do not isolate ONNX compute or include
CPU consumed by every internal native thread.
Use the paired `vad_model_profile_wall` and `vad_executor_profile_wall`
observations when comparing those CPU intervals. The older wall summaries
retain their original probe boundaries.

The profile retains at most 16 slow completed VAD windows and two inflight
windows, with source pointers and stage timestamps. A cancelled await can end
before its executor callable. Keep missing finish/resume fields unobserved.
`event_loop_lag` samples loop scheduling, while process CPU and wall intervals
cover the observed worker process, not VAD alone.
`asr_callback_to_actor_lock` runs from a callback observation before publication
to entry of its existing locked callback. ASR dispatch-to-start also includes
thread dispatch, context propagation, and scheduling. These measurements do not
isolate ASR inference, remote provider/GPU queues, internal gRPC flow control,
or source-input starvation.
ASR thread dispatch, entry, first RPC response, and completion remain separate
observations. Numeric progress records include blank results and supplied/source
dependency pointers, without adding transcript text.

For the exclusive worker pool, set trusted `NVA_OFFLINE_POOL_METRICS_DIR` to
retain bounded `nva.offline.pool-metrics.v1` worker and proxy sidecars. They
contain admission, identity-check, backend-connect, relay, drain, and handler
return observations. Join the case's `server_metrics_session_id` to its worker
sidecar, then use that worker's opaque `lease_sha256` to join the proxy sidecar.
The proxy does not inspect response bodies to discover the session ID.
They exclude raw lease credentials,
transcripts, audio, and provider/request bodies. Missing or dropped sidecars
remain an observability gap.

`diagnostics_persistence_started` precedes the metrics write. Its completion
cannot be recorded in the snapshot being written, so the trusted worker holder
records that later boundary separately. The outer worker's handler return and
authenticated completed-lease status remain distinct from the inner
`session_returned` marker. Use matching clock domains and these observed joins
to attribute waiting; client totals or CPU correlations alone do not identify
a server bottleneck. Profiling adds observation overhead and establishes no
throughput, accuracy, or production-readiness claim.

### Passive Session Traces

Set `NVA_EVAL_TRACE_DIR` on the server to retain passive session traces for
normal Generic Assistant streaming and offline processing. Without that setting,
the normal provider classes and frame flow are unchanged. The trace never creates
audio, changes pacing, or makes provider requests. It counts PCM, buffers bounded
events, and writes a session JSON file after cleanup. Trace-file failures do not
fail the voice session.

The live trace records ASR request yields and existing response callbacks,
LLM frames, TTS text submission and generated PCM, stitched-worker completion,
VAD and semantic turn events, and output frames after their transport write.
Context IDs and request indices retain associations established by observed
frames. They do not prove transcript completeness or text ownership. Absent
events remain null. Every trace includes `timing_scopes` for its boundaries.
Observer callbacks can run after frame pushes. The report therefore calculates
LLM frame intervals from their original pipeline timestamps. Provider hooks use
a separate clock origin, so the report does not subtract a TTS-hook timestamp
from an LLM-frame timestamp without clock calibration. Use the immutable request
snapshots below when exact dispatch contexts matter: an observer's later context
snapshot can already include another frame. Delivered TTS text also does not
prove that a client heard it.

### Trace Context Commitment and LLM Requests

The trace also captures synchronous processor boundaries and immutable request
snapshots. Use these observations to separate ASR output arrival, user-context
commitment, and the request that reaches the LLM client.

| Boundary | Captured Evidence | Limit |
| --- | --- | --- |
| ASR frame publication or delivery | Frame identity and text before the live ASR push or at offline scheduler delivery | Publication does not prove queue acceptance or aggregator consumption. |
| User-aggregator processing start | The frame identity at the native processing callback | Processing has started after dequeue and pause handling; it has not necessarily returned. |
| User-aggregator processing return | The frame identity after its normal handler returns | A missing return can indicate cancellation or an error. |
| Context push | A copied context at the synchronous handoff boundary | This is separate from a later observer snapshot and provider dispatch. |
| LLM request preparation | Copied messages and options after native parameter conversion | Preparation is not an HTTP attempt or provider-consumption acknowledgement. |
| LLM HTTP attempt | The serialized chat-completion request body and its request/attempt association | The capture observes client dispatch, not server receipt or hidden model state. |

Join `processor_boundary` events by `frame_id`. Join `llm_request_prepared` and
`llm_http_request` by `request_id`; `http_attempt` separates SDK retry attempts.
`context_frame_id` connects a request to the context frame entering the service.
Offline dispatch creates a new native context frame. Bridge the original user
commit through `offline_native_context_delivery` using its `generation` and
immutable `request_messages`. The scheduler's `response_requested` event uses
`response_id` for that generation. Do not equate the original commit frame ID
with the new native context frame ID; mark nonunique joins as ambiguous.
Legacy observer `llm_request_index` values remain separate from these dispatch
identities. Preserve the distinction between an emitted transcript and one the
aggregator processes.
An emitted frame without a processing event is unobserved at that boundary;
it is not proof that the native queue discarded it. Check `dropped_events` and
`observation_errors` before interpreting missing records.
The event cap also bounds retained synthesis-context associations. After capture
fills, later observations increase the drop count rather than growing that map.

Inline `processor_boundary` observations retain `TranscriptionFrame.finalized` and
both transcript classes' `includes_inter_frame_spaces` flags. Unsupported
nonboolean values are `null`. `STTMetadataFrame` observations include
`stt_ttfs_p99_latency_s`, `stt_ttfs_p99_latency_ms`, and
`stt_ttfs_p99_latency_finite`. These describe configured time-to-final-segment P99
policy metadata, not measured request latency or ASR completeness. When present,
these fields let conditional policy replays use observed inputs for those values.
Missing fields remain unobserved, not false or default values; the earlier replay
results retain their reconstructed-metadata qualification.

Offline `logical_sample` is the scheduler delivery position in 16,000 Hz samples.
Online ingestion and supplied-audio cursors record observed PCM prefixes. These
positions do not assign words to chunks or establish that no earlier ASR event
remains pending. The trace does not use ASR word timestamps to place conversation
events.

Prepared parameters retain native omitted-value markers. The HTTP capture
separately retains the serialized JSON body, its canonical digest, and the raw
serialized-body digest. Check `llm_request_capture_capability` before requiring
HTTP evidence. An unsupported client or unobserved request identity is a
diagnostic gap, not a provider failure. Capture covers requests in the owned
completion scope. Out-of-band `run_inference` calls, including history
summarization, are outside that scope; this trace is not a complete HTTP inventory
when those calls occur. The paired diagnostic procedure below disables history
summarization.

Use `started_at_utc` with source PCM digests to correlate cases. Online
`asr_ingested_audio` describes observed ingestion; `offline_source` describes the
offline upload. Compare format and trailing padding before asserting byte identity
across those boundaries.

Request snapshots can contain conversation text. They are server-owned,
opt-in diagnostic artifacts written with file permissions `0600`. They exclude
authentication headers and provider response bodies. `sdk_retry_count` retains
only a numeric SDK retry counter; raw headers are not retained. Unscoped requests
on shared HTTP clients are skipped before their bodies are read. The hooks
preserve native queues, timers, and request construction. Snapshot copying and hashing still add CPU and memory overhead,
so a traced run is diagnostic evidence rather than an uninstrumented latency
baseline.

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
| TTS completion | Native serialized response end, provider-worker EOF, and audio-context drain; retain native processing metrics separately | Stitched-worker completion, including later text waits; cancelled or returned workers do not by themselves establish successful RPC EOF |
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
These checked-in client and batch commands use `full_session` completion for
audio diagnostics. They do not implement the separate VoiceBench text collector.
Text-mode consumers must apply the [candidate epoch contract](#raw-llm-text-for-benchmark-adapters).

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

A Uvicorn worker is a separate process, not a single-conversation slot. One
worker can serve concurrent asynchronous sessions. Increasing the worker count
does not remove blocking work inside each process, and configured online
sessions need routing affinity or shared session storage across workers.
Record the worker count with concurrency results instead of assuming that a
benchmark concurrency of six requires more than six workers.
An offline-only multiworker profile can use the existing inline start-message
configuration without creating process-local session IDs. This does not change
or validate the configured online session path. Retain observed server process
IDs with scalar timings; a configured worker count does not prove how sessions
were distributed or how many provider RPCs ran concurrently.

### Use the Exclusive Offline Worker Pool Prototype

This separate startup mode assigns at most one active offline ASGI handler to
each backend process. It uses a front door and 10 single-worker backends by
default. It does not change the normal `server.py` startup, live pipeline,
dependencies, provider selection, or default deployment configuration.

From the repository root, use the same configured Generic Assistant service
environment as the ordinary offline endpoint and choose a free front-door port:

```bash
PYTHONPATH=src uv run python -m examples.generic.offline.pool_supervisor \
  --host 127.0.0.1 --port 17927 --pool-size 10 \
  --startup-timeout-seconds 60 --drain-timeout-seconds 60 \
  --admission-wait-seconds 1
```

The prototype requires POSIX inherited sockets. The supervisor prebinds private
ephemeral IPv4 TCP sockets and passes each descriptor to one owned child. Each
worker validates that private endpoint and ensures it is listening without
rebinding it. The front door remains a single process. Internal authentication
and lease headers belong to the pool; clients do not configure or receive those
credentials.
Worker children retire if their original supervisor disappears. Stop this
foreground process to close the pool and its owned children. Ordinary shutdown
joins owned child cleanup before propagating caller cancellation. Cleanup
failures remain errors.

The optional `--inherit-worker-process-group` flag is for a CI-owned process
session. It requires the supervisor to be its own process-group leader and
keeps backend children in that group. The default retains separate worker
sessions. Each backend still owns its private socket and one Uvicorn worker.
The CI launcher records the front door and ten backend identities before
relying on group cleanup. Its outer cleanup permits 15 seconds for termination
and 5 seconds for forced stop and verification; it retains the ownership marker
if the live group cannot be verified or reaped. These bounds do not extend a
sample's response deadline or provider guards. Child parent-loss monitoring
continues to stop only the child itself.

The public surface is limited to these routes:

| Route | Purpose |
| --- | --- |
| `GET /health` | Front-door and worker-pool state |
| `GET /api/deployment` | Native deployment metadata for client preflight |
| `/api/ws/offline` | One leased native offline WebSocket stream |

The front door does not serve the Web UI, live `/api/ws`, or
`POST /api/session-config`. Supply configuration inline in
`offline.session.start`. Query-based configuration and session IDs are
unsupported; a nonempty WebSocket query is rejected with close code `1008`
before worker admission. Source-offset packets and the selected native
completion policy pass through without in-flight replay to another backend.
Both sealed and negotiated `during_upload` input retain their existing rules.

A reserved connection rechecks the worker's authenticated identity and idle
state before opening its backend WebSocket. A connection owns its worker lease
through native handler cleanup. Closing a socket moves that lease to draining.
The pool releases it only after an authenticated status response matches the
worker PID and exact completed lease, with no active handler. Unverified
identity or lease state, drain expiry, and
crashed workers are quarantined. Quarantined slots stay unavailable until the
pool restarts. Handler errors remain failed attempts even after their lifecycle
permits lease release.
This evidence concerns ASGI handler return; it does not attest every provider
RPC's EOF or completion of noncooperative native executor work.

Admission does not classify audio or recognize capability probes. An ACK-only
probe also leases a worker. When every slot is active, a new stream receives
WebSocket close code `1013` immediately. If capacity is draining, admission can
wait up to 1 second for verified release, with pending admissions bounded by
pool size. Expiry still rejects with `1013`; it never grants early reuse.

The following bounds have separate responsibilities:

| Bound | Default | Scope |
| --- | --- | --- |
| Pool size | 10 | Exclusive backend slots; configurable from 1 through 64 |
| Startup watchdog | 60 seconds | Authenticate readiness of all owned backends |
| Admission wait | 1 second | Await already-draining capacity before native ACK |
| Drain watchdog | 60 seconds | Confirm handler return; expiry quarantines the worker |
| Private status/preflight HTTP call | 5 seconds | Bound one internal metadata or status request |
| Backend WebSocket connection | 10 seconds | Connect to the reserved worker; no replay |
| Backend close handshake | 5 seconds | Bound socket close within the drain budget |
| Child stop/join | 5 seconds per phase | Await TERM, then forced stop if needed |
| WebSocket message / receive queue | 8 MiB / 1 message | Bounded front-door and backend transport buffering |

The drain watchdog is a maximum failure bound, not a delay after each successful
case. It does not replace the evaluator's post-upload 10-second first-text
deadline, the separate response-completion limits, or native provider guards.
The pool adds no input silence, TTS completion wait, or audio-gap completion rule.
Any caller retry is a new attempt and must retain the original failure evidence.
Client wall time includes admission and forwarding overhead; keep it separate
from native provider TTFB measurements.

This is a local prototype with fake-provider validation only. Pool readiness
does not establish real model readiness, production suitability, score
equivalence, or throughput improvement. Preserve source pins, input hashes,
lease outcomes, failures, and timing scopes before interpreting later tests.

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

Use the commands in [repository guidance](../AGENTS.md#validation) for the full
required checks. The [dated progress log](offline-processing-progress.md)
records historical test counts with their source revisions and distinguishes
model-free checks, real-provider captures, and scored subsets. Those counts are
validation receipts for the named revisions, not the current suite size.

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
