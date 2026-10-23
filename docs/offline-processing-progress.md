# Offline Processing Progress

This log records implementation findings, decisions, and validation after the
[offline processing proposal](12-offline-processing.md). The proposal and its
[implementation reference](offline-processing-reference.md) retain their
previously shared content. Later entries record changes in direction and qualify
the status of earlier implementation claims.

Append dated findings and test results here. Distinguish a proposed decision,
implemented behavior, and verified behavior. Benchmark audio integrity does not
establish behavioral equivalence or accuracy.

## October 21, 2026: Transcript Ownership And Streaming Behavior

### Finding

The initial implementation analyzes the complete recording with separate voice
activity detection (VAD) and automatic speech recognition (ASR) passes. It then
reconstructs conversational turns by joining ASR word intervals to VAD intervals.

The implementation performs these steps.

1. `OfflineSessionEngine.run()` gathers the complete detection and recognition
   traces before starting replay.
2. `ContinuousASR` enables ASR word offsets and stores final transcript segments
   with source intervals and projected availability.
3. `_Replay._bind_recognition()` assigns each segment, or each word when needed,
   to a detected turn using interval overlap.
4. Missing ownership raises `Final ASR text has no detected source turn` before
   response generation starts.

The relevant sources are the [session engine](../src/examples/generic/offline/engine.py)
and [provider adapters](../src/examples/generic/offline/providers.py).

ASR and VAD independently estimate speech boundaries. A leading word can precede
VAD confirmation, and an ASR word can have identical start and end offsets.
Sharing the same sample coordinate system does not guarantee overlapping model
estimates. Requiring overlap adds a failure condition that the online transcript
aggregator does not require.

The existing scripted tests exercise causal replay and interval ownership. They
do not establish that this reconstruction matches the online aggregator. Widening
intervals would preserve that architectural difference.

### Decision

ASR word timestamps must not determine conversational ownership. Keep one
continuous ASR stream and feed transcript events into the same aggregation and
turn-control policy used online. Audio chunk pointers determine media progress;
provider arrival order determines transcript event order.

Uploading a recording quickly only makes its PCM available. It must not release
future conversation events into an earlier response. Keep these positions
separate.

| Position | Meaning |
| --- | --- |
| Uploaded samples | Source PCM available in the bounded session buffer |
| Released samples | Source PCM exposed to the conversational pipeline |
| Confirmed ASR progress | Source prefix for which the provider's progress contract is established |
| Playback samples | Bot PCM already presented on the rendered media timeline |

A submitted chunk or an empty client queue does not establish completed ASR
processing. The adapter must establish what processing metadata guarantees before
using it as a barrier. ASR finalization also does not establish conversational
end-of-turn: several final transcripts can accumulate within one user turn.

The target integration preserves the online pipeline's decisions for transcript
aggregation, turn completion, interruption, and conversation context. Execution
mode selects media progression and output rendering. It must not introduce a
separate word-to-turn ownership policy.

### Open Dependency

The ASR chunk-progress and event-ordering contract requires verification. The
adapter must identify when it can advance past an audio prefix without skipping
transcript events for that prefix. Local upload progress and request iteration
are insufficient evidence.

If the backend cannot provide that guarantee, document the limitation explicitly.
Do not describe a heuristic cursor or widened timestamp window as equivalent to
online execution.

Measured compute durations remain separate from media duration. Preserve them
when scheduling response availability, but do not present bulk processing timings
as hypothetical online latency.

### Validation Gates

The implementation must pass the following gates before claiming the revised
approach is validated.

| Gate | Required Evidence | Status |
| --- | --- | --- |
| Provider progress | Verified chunk-progress semantics, monotonicity, event ordering, and EOF drain | Pending |
| Shared controller | Same fixed event trace produces the same text commits, turn transitions, request context, and interruption decisions | Pending |
| Media boundaries | Deterministic chunk-size independence, silence, output spans, and final-tail tests | Pending |
| Real samples | Previously failing benchmark recordings complete with their words preserved and inspectable audio | Pending |
| Paired behavior | Same recordings run online and offline, with run variation and scoring reported separately | In progress |

The deterministic cases must cover leading or quiet speech, transcripts before
VAD confirmation, multiple finals per turn, delayed finals, resumed speech,
barge-in, output starvation, and input EOF during pending generation. Cancellation
must affect generation and context, rather than only clipping completed audio.

Online benchmark results remain separate from validation of the offline fix.
Successful recording generation alone does not establish matching reasoning,
accuracy, task completion, or latency.

### Implementation And Results

The diagnosis and direction above are recorded. Implementation and real-sample
validation are pending at this entry. Append the implemented contract, affected
modules, test commands, and measured results after each gate completes.

## October 21, 2026: Shared Pipeline Implementation Checkpoint

The revised runtime is implemented in the working tree. Real-service protocol
checks and benchmark sample validation remain pending at this checkpoint.

### Event Capture And Turn Control

The [runtime factory](../src/examples/generic/offline/factory.py) selects the
[causal session engine](../src/examples/generic/offline/causal.py). The engine
drains one continuous ASR request before replaying conversation events. It keeps
every nonempty interim and final result in provider response order.

The [recognition adapter](../src/examples/generic/offline/recognition.py) disables
word timestamps. It maps the cumulative `audio_processed` counter to a supplied
PCM chunk boundary and rejects invalid or decreasing progress. Normal response
stream EOF, after all source audio is consumed, establishes completeness. The
last reported counter does not need to include trailing silence. The counter's
behavior on the deployed ASR service still requires the real protocol probe.

The [aggregation adapter](../src/examples/generic/offline/aggregation.py) uses the
actual Pipecat 1.12.0 user and assistant aggregators with one shared context. It
replaces the speech-timeout and watchdog waits with an integer sample clock.
Pipecat retains transcript accumulation, turn transitions, transcript-based
starts, interruption decisions, and context commits. There is no word-to-VAD
ownership join. ASR retains the default 320 ms stop history without force-EOU.

### Output Sequencing And Recording

The [TTS adapter](../src/examples/generic/offline/tts_trace.py) drives the actual
`NvidiaTTSService` with streamed language-model text. It captures the service's
audio, text, and control frames in their emitted order. The
[playback adapter](../src/examples/generic/offline/playback.py) passes those
frames through Pipecat's `MediaSender` buffering, including the final partial
chunk flush. It replaces the paced sink with recording, preserving output queue
order and configured chunk boundaries.

The scheduler places output after its observed provider readiness and preceding
playback. It renders gaps as silence and retains a final response beyond input
EOF. Native-rate output is converted to recording PCM using cumulative sample
pointers. Assistant context receives the text frames actually released in output
order. Interruption cancels pending generation and discards future output frames.

Observed bulk processing, transport, and queueing durations remain part of
response placement. These measurements do not recover realtime streaming
latency, and source-prefix processing metadata is not an arrival timestamp.

### Supported Scope And Remaining Validation

The initial integration explicitly requires Silero turn detection, stitched
Magpie synthesis, an HTTP chat-completions language-model endpoint, and tools
disabled. Unsupported timer policies, realtime service handoffs, and enabled
tools fail explicitly rather than use an unverified fallback.

The validation gates above remain open until their evidence is recorded. Next,
verify the deployed ASR progress contract, run deterministic aggregation and
output-order tests, and rerun the previously failing benchmark samples. Behavioral
and scoring consistency against realtime streaming remains in progress. No
accuracy-equivalence or real-benchmark pass claim is made at this checkpoint.

## October 21, 2026: Protocol Evidence And Paired Sample Validation

### Verified Progress

The deployed ASR service accepts one continuous, unpaced request with word
timestamps disabled. On benchmark sample `2`, the adapter captures 179 ordered
nonempty transcript events, including 21 final results. Their cumulative
`audio_processed` counters pass the supplied-chunk boundary checks. The response
stream drains to EOF after all source audio is consumed. Its last counter omits
trailing silence, as the completeness contract allows. This verifies one
recording on this backend, rather than every possible provider response.

Replaying that captured trace through actual Silero detection and the shared
Pipecat aggregators preserves every final transcript token in order, including
the previously unassigned leading words. This diagnostic uses a synthetic empty
responder. It does not generate bot audio or establish end-to-end behavior.

Before the subsequent WebSocket sink-clock correction, the full local test suite
passes with 814 tests, 3 skipped tests, and 173 subtests. The deployed Linux image
also passes 81 focused aggregation, recognition, synthesis, and buffering tests.
These results do not validate the sink-clock correction or the paired benchmark
outputs described below.

### WebSocket Output Timing Finding

The native WebSocket transport sends PCM before its pacing wait. With two
400 ms chunks already queued, it sends both at the initial offset. The following
text and stop frames can advance at 400 ms, while buffered client audio continues
through 800 ms. Treating each write as a full playback-duration wait delays
control frames and can change interruption and assistant-context behavior.

The correction separates client PCM presentation pointers from sender write
completion and control-frame delivery. It must also preserve pacing reset on
interruption and start the output-starvation timeout after write completion.
Focused native-transport comparisons and interruption tests are in progress.
No ASR word timestamps are involved in either clock.

### Bounded Comparison Plan

One shared model deployment serves both execution modes. The large language
model runs on GPU 4; ASR and text-to-speech (TTS) run on GPU 7. Other GPUs are
outside this validation. Both modes use the same model endpoints, prompt,
continuous ASR settings, and 320 ms stop history without force-EOU. Welcome audio
is disabled, Silero turn detection is enabled, and output uses the configured
400 ms chunk setting.

The sample set is `2`, `3`, `250`, `503`, and `750` from AA-BigBench Audio. Run
the five recordings online, then the identical PCM files offline, with
concurrency 1 in both modes. Retain source audio, complete output WAV files,
transcripts, event traces, and latency breakdowns in milliseconds. A required
offline correction can reuse the saved online baseline.

Paired results and behavioral and scoring consistency remain in progress.
Append measured findings after the runs complete. These five samples debug the
architecture; they do not replace a full accuracy benchmark or establish
equivalent realtime latency.

## October 21, 2026: Incremental Processing And Transport Corrections

### ASR Processing Clock

The first paired runs expose a scheduling error before language-model response
variation. The initial availability projection adds accumulated ASR processing
delay to each later source prefix. This repeats earlier work on the media
timeline and can postpone transcript delivery across a user-turn boundary.

The [processing clock](../src/examples/generic/offline/recognition.py) now charges
each observed processing interval once. For checkpoint `i`, it computes
`availability_i = max(source_prefix_i, availability_previous) + interval_i`.
Each prefix comes from a supplied audio chunk. Each interval starts after both
the preceding provider result and that chunk's supply time. Empty results and
multiple results received together also advance the ordered clock. Integer
arithmetic accumulates durations before rounding availability to a sample.

This permits earlier processing to overlap later source progression while
retaining a backlog when processing is slower. It uses no ASR word timestamps.
Measured intervals include transport, queueing, and backend batching; they are
not isolated inference times or reconstructed realtime arrival times.

The [recorded-trace regression](../tests/unit/test_offline_asr_processing_clock.py)
reproduces sample `3`. The earlier projection releases a final transcript at
16,323.125 ms, after a VAD stop at 15,392 ms and resumed speech at 15,808 ms.
The corrected clock allows the shared controller to commit at 15,392 ms. This
tests the specific ordering failure with the captured service profile.

### Native State And Lifecycle Corrections

The WebSocket sink correction is implemented. The
[native-consumer comparison](../tests/unit/test_offline_websocket_sink.py) runs
the unmodified Pipecat sender with a deterministic clock. It confirms control
delivery at 400 ms while buffered PCM continues through 800 ms. Interruption
tests separately check delivered assistant text, remaining PCM, and pacing reset.

The Silero adapter also follows native recurrent-state reset phase: reset after
the first window, then 5,000 ms after each actual reset window. The previous
absolute reset grid differed from the native policy. Matching the reset phase
produces identical VAD decisions in the sample `3` comparison; that difference
does not explain the initial transcript scheduling failure.

An independent online gateway issue creates an empty response after a completed
response when the same frame crosses another pipeline edge. The
[lifecycle observer](../src/realtime/observer.py) now retains bounded frame-ID
deduplication across response completion and cancellation, and clears it at
session shutdown. Regression tests use actual frame-pushed notifications for
three response-announcing frame types.

### Validation Status

After these corrections, the full local suite passes with 830 tests, 3 skipped
tests, and 176 subtests. Lint passes for the changed Python surfaces. A deployed
Linux validation passes 140 focused tests and 15 subtests before the last two
recorded-trace tests are added. Those two additions change tests only.

Compare complete backend final transcripts and actual language-model request
contexts when assessing preservation and turn grouping. Gateway transcript
publications alone are not exhaustive: a later final can overwrite the pending
publication. A difference in those publications does not establish lost ASR
words.

The backend's cumulative processed-audio field describes processing progress.
It does not by itself guarantee that no future result depends on later input.
Finite response-stream EOF remains the completeness barrier. Behavioral and
scoring consistency against realtime streaming remains in progress; successful
recordings and matching request counts do not establish equivalent behavior.

## October 21, 2026: Five-Sample Comparison Results

The corrected offline deployment completes all five selected samples at
concurrency 1. Each produces complete output WAV files with source coverage,
retained response tails, and no overlapping output spans. Input PCM matches the
online baseline. The shared model services remain on GPUs 4 and 7.

The online baseline combines four original successful runs with a retry of
sample `250` after the observer correction. The failed run and an earlier
connection-refused attempt remain separate evidence. This is a bounded
architecture comparison, rather than a full benchmark or repeated-run accuracy
study.

The measured processing durations and actual language-model request counts are
shown below. Request counts come from server request traces, not gateway
response-event counts.

| Sample | Source Duration (ms) | Offline Processing (ms) | Online Requests | Offline Requests |
| --- | --- | --- | --- | --- |
| `2` | 37,888.00 | 4,881.36 | 6 | 6 |
| `3` | 49,437.50 | 5,094.00 | 9 | 9 |
| `250` | 22,528.00 | 2,383.22 | 5 | 5 |
| `503` | 14,128.00 | 1,412.61 | 2 | 2 |
| `750` | 17,704.00 | 1,743.27 | 3 | 3 |

Together, 141,685.50 ms of source audio takes 15,514.46 ms to process offline,
about 9.13 times faster than source duration in this run. This ratio describes
bulk execution on this deployment; it is not a realtime response-latency result.

The complete backend ASR final-segment sequence matches in all five pairs.
User-message grouping also matches at every actual language-model request.
These findings validate the corrected transcript scheduling on this sample set.

Complete request contexts match throughout samples `503` and `750`. In samples
`2`, `3`, and `250`, they first differ at request 2: offline has committed an
assistant message that online has not committed. User-message grouping still
matches. Further validation must examine assistant output and context lifecycle
under the differing observed bulk and streaming processing durations.

The final server-published assistant text matches in all five pairs. This is
server-text evidence, not transcription of the played WAV files or a judge
score. Behavioral and scoring consistency against realtime streaming remains in
progress. AA-BigBench and FDB scoring validation is not established by this
comparison.

## October 22, 2026: Stateful Scheduling And Remaining Timing Questions

The next architecture target is one conversation runtime with a shared session
timeline and independent component cursors. Online and offline execution should
share turn detection, context commits, cancellation, and service lifecycle.
Changing input pacing alone cannot guarantee identical events when provider
processing times also change.

### Findings From The Current Implementation

The [live pipeline](../src/examples/generic/pipeline.py) processes input, ASR,
context aggregation, LLM, TTS, and output asynchronously. Components do not
process the same audio chunk simultaneously. At session offset 10,000 ms, VAD
can have consumed 10,000 ms of input while ASR has processed 9,700 ms and output
plays a response to an earlier turn. Real input arrival and output presentation
anchor their different positions to a common timeline.

The current [offline engine](../src/examples/generic/offline/causal.py) drains
the complete continuous ASR request before conversation replay. It also awaits
each response trace before scheduling its output frames. LLM and TTS overlap
within that response, but ASR and TTS do not overlap. The
[ASR processing clock](../src/examples/generic/offline/recognition.py) projects
observed bulk processing intervals onto source offsets. That projection is a
timing model; it does not reconstruct the corresponding online execution.

The five-sample comparison preserves all backend final transcript sequences
and user-message grouping at every model request. Samples `2`, `3`, and `250`
first differ in assistant context at request 2. Offline has delivered assistant
text before interruption; online publishes no response content before that
interruption in each inspected trace. This difference in delivered context
requires separating content-dependent framing, provider readiness, and runtime
lifecycle ordering. The saved traces do not isolate those causes.

Assistant text can reach the native aggregator before synthesized PCM. Context
commits must follow the shared native frame lifecycle. Inferring committed text
from the audible WAV prefix would introduce a different policy. Also, an
interruption can stop a pending first-token timer: such a value represents a
cancelled measurement, rather than a successful token arrival.

### Proposed Session State And Chunk Provenance

Track each input chunk with `session_id`, `chunk_sequence`, `sample_start`,
`sample_end`, and `sample_rate`. Keep uploaded input separate from input
released to the conversation. Uploading the whole recording must not expose
future speech to context or interruption logic.

Each stage retains its own cursor: input release, VAD consumption, ASR
submission and reported progress, generated output PCM, transport delivery,
and client presentation. Use integer sample positions with explicit rates and
cumulative resampling conversions. Measure provider dispatch-to-readiness
intervals with a monotonic clock, retaining their queueing, transport, and
input-wait scope. Project those intervals onto the session timeline; audio
offsets cannot measure processing durations themselves.

Each event carries a sequence, causal predecessor, turn and response generation,
context version, input dependency prefix when known, and logical availability.
Record provider dispatch and readiness separately. A dependency identifies what
input enabled an event; availability identifies when the event can affect state.
Concurrent processing intervals overlap rather than being added as sequential
delays.

The following boundaries define what tags can establish.

| Stage | Supported Correlation | Remaining Contract |
| --- | --- | --- |
| VAD | Exact consumed PCM prefix and decision offset | Preserve detector state, reset policy, and decision delay. |
| Continuous ASR | Submitted chunks, result order, and reported processed prefix | Prove completeness through silence and delayed finals or revisions. |
| LLM | Context version, triggering turn, response generation, and token order | Tokens have no one-to-one mapping to source PCM chunks. |
| TTS | Response/context ID, submitted text, and generated output sample ranges | Ordinary stitched output has no exact source-chunk or word-to-PCM mapping. |
| WebSocket Output | Send completion and independent PCM presentation position | Control delivery and audible completion occur at different times. |

ASR consumes audio and produces text; TTS produces audio. LLM and TTS adapters
can carry correlation tags around model requests without requiring the models
to understand those tags. Such tags establish provenance, not missing timing
or completeness guarantees.

### Proposed Incremental Runtime

Keep one continuous ASR stream while LLM and TTS remain active. Replace the
full-file recognition barrier and full-response await with incremental event
delivery. Share the native pipeline actors and lifecycle in both modes, with
interfaces for the session clock, ingress, provider events, and output sink.
The recorder consumes delivered output events and sample ranges; it does not
decide conversation state.

The scheduler releases input chunks and ready events in causal order. It can
advance beyond idle periods only when active components guarantee that no
earlier relevant event remains unresolved. This guarantee is a safe progress
frontier. A submitted chunk or an empty local queue is not that guarantee.
Source analysis can run ahead only where it is independent of conversation
feedback, and its results remain gated before state changes.

The current ASR `audio_processed` field reports progress. It has not been
verified as a promise that all events for an earlier logical interval have
already arrived. Attaching the latest uploaded chunk index to a delayed final
would invent its dependency. Response-stream EOF is the current verified
completeness barrier.

An incremental ASR adapter therefore needs a verified acknowledgement or
watermark contract, including silence and delayed revisions. Provider adapters
must also distinguish actual completion from pending work. Without a safe
frontier, the scheduler waits instead of jumping over unresolved events. A
declared timing model or a recorded provider trace supports controlled replay,
but does not establish identical fresh online behavior. Strict waiting can
reduce acceleration; the previous measured speedup is not a guarantee for this
proposed runtime.

On interruption, apply the same native turn policy at its detection offset,
cancel the active response generation, discard server-queued output, and fence
late callbacks from that generation. Match the client playback-buffer
cancellation policy: the server cannot recall PCM already sent. Preserve text
and control events already delivered under the native lifecycle. Keep input
EOF, ASR drain, and final response completion separate so late finals and
response tails are retained.

### Why Identical Input Can Still Produce Different Events

Consider a turn released at 10,500 ms and resumed user speech detected at
10,900 ms. If bot PCM becomes ready at 10,650 ms, playback starts and the next
speech interrupts it. If PCM becomes ready at 11,000 ms, interruption occurs
before any bot audio plays. Both outcomes can follow the same policy with the
same input dependencies. Different compute readiness can legitimately change
the next context, even with identical generated wording.

The framework must produce identical decisions for identical provider events
and readiness times. Fresh model executions require repeated behavioral and
scoring comparisons because both content and processing times can vary.

### Next Validation Gates

First, capture provider dispatch, text, PCM, control delivery, interruption,
and context-commit events in both modes. Replay identical provider content and
readiness through both adapters, checking every context version, cancellation,
turn decision, and output sample range. Preserve causal ordering for events
with equal logical offsets, and test alternate concurrent orders explicitly.

Cover silence without ASR responses, late finals, revisions, burst delivery,
short pauses, interruption before text, interruption between text and PCM,
partial output chunks, stale callbacks, trailing silence, EOF flush, and final
response extension. Verify the incremental ASR frontier contract before
enabling unrestricted advancement.

After those checks, repeat the same five real samples using the shared model
stack on GPUs 4 and 7. Classify differences as content variation, readiness
variation, or framework ordering defects. Behavioral and scoring consistency
against realtime streaming remains in progress. This entry records the next
design and validation gates; the incremental runtime is not yet implemented.

## October 22, 2026: Architecture Review And Native Pacing Diagnostic

### Experiment And Scope

The target remains the shared native conversation graph with clock and
input/output adapters. The current ASR prepass, custom LLM adapter,
captured-response replay, and lookahead budgets do not establish streaming
equivalence.

An isolated CPU application reuses models on GPUs 4 and 7
without restarting existing services. The five files run through the native
pipeline at concurrency 1, first paced, then unpaced.

Both modes send identical 512-sample chunks, or 32 ms at 16,000 Hz. The
[comparison client](../benchmarking_tools/offline-eval/compare_live.py) retains
default `input_speed=1.0`; callable `input_speed=0` removes source-duration waits.
Native turn timers and output pacing remain unchanged. Unpaced WAV files use
client wall-clock offsets, not an aligned source timeline.

### Paired And Repeated Results

All five paced captures succeed. Three unpaced captures succeed; samples `503`
and `750` fail with a fatal TTS context-completed-with-no-audio error. These are
pipeline errors, not collection deadlines. Failed audio and traces are retained.

The complete observed backend ASR final-segment sequence matches in all five
pairs. Provider-supplied PCM hashes match source PCM in all ten initial captures.
Observed VAD-start counts match: 6, 9, 5, 2, and 2. These counts do not establish
matching event order or boundaries.

Five fresh unpaced sessions repeat the experiment against the same saved paced
baseline. The table uses actual native dispatch logs, excluding delayed observer
context copies. Prefix measurements describe the first paired run.

| Sample | Paced/First Unpaced LLM Requests | Repeat Unpaced LLM Requests | Paced STT Prefix At First Final (ms) | Unpaced STT Prefix At First Final (ms) |
| --- | --- | --- | --- | --- |
| `2` | 6 / 15 | 17 | 1,280 | 37,888 |
| `3` | 9 / 21 | 21 | 1,600 | 49,437.5 |
| `250` | 5 / 15 | 14 | 1,760 | 22,528 |
| `503` | 2 / 4 | 7 | 960 | 14,128 |
| `750` | 3 / 5 | 5 | 960 | 17,704 |

Requests total 25 paced, 60 initially unpaced, and 64 in the unpaced repeat.
Both unpaced runs preserve observed ASR finals and VAD-start counts but change
user-message grouping in all five pairs. Four pairs differ at the first request,
before assistant text exists. LLM wording variation cannot explain those initial
context differences.

Four repeat captures succeed. Sample `503` instead reaches the 60,000 ms guard
with the client awaiting a response, despite a completed response. This is a
timeout, unlike its initial TTS error. Native keepalive adds 100 ms of silence
after its intact source PCM. Whole-session provider hashes match source in
14 of 15 captures; this repeat has extra PCM, rather than missing source bytes.

In every unpaced capture, STT ingests the entire source before the first ASR
final. This does not mean that all VAD processing has finished: VAD notifications
and ASR results still interleave. Equal chunk sizes preserve packet boundaries,
but do not preserve the relative progress and readiness of asynchronous stages.

### Required Runtime Boundaries

The next implementation must share the native conversation graph and service
lifecycle, with clock, ingress, provider-event, and output adapters. A safe
delivery frontier establishes whether an earlier state-changing event remains
pending; it does not assign ASR words to chunks. A verified provider watermark
can permit faster advancement. Correctness can conservatively wait without one;
an uploaded cursor or reported processed prefix is insufficient by itself.

Finite-input completion also needs an explicit half-close and drain contract.
The native STT `EndFrame` path closes its iterator and cancels its worker; it is
not a guarantee that queued input and delayed results drain. A quiet collection
window followed by WebSocket disconnect does not establish ASR response EOF.
The diagnostic therefore reports observed captures, not verified stream drain.

The passive trace records independent ingestion, provider supply/progress, text,
and output edges without changing turn policy. The full local suite passes with
839 tests, 3 skipped tests, and 181 subtests; lint passes for 279 Python files.
Controlled identical-event replay against the native and offline paths remains
pending. The incremental runtime is not implemented. Behavioral and scoring
consistency remains in progress; this diagnostic establishes neither a fix for
the current assistant-history difference nor benchmark accuracy equivalence.

## October 23, 2026: Incremental Runtime Implementation Checkpoint

### Implemented Execution Path

The [runtime factory](../src/examples/generic/offline/factory.py) now selects the
[incremental session engine](../src/examples/generic/offline/incremental.py).
Continuous ASR callbacks and VAD analysis-window checkpoints feed the scheduler
while their producers remain active. Conversation replay no longer waits for
whole-file recognition or a complete response trace. The previous tape engine
remains a reference implementation, rather than the selected execution path.

The [event actors](../src/examples/generic/offline/actors.py) publish callback
batches and future availability bounds atomically. The scheduler advances the
sample clock only after checking pending actor events, policy deadlines, and
output work. Equal-offset events retain explicit ordering. No future user-speech
lookahead determines a response compute budget.

The [continuous ASR adapter](../src/examples/generic/offline/recognition.py)
retains one request and delivers interim and final results incrementally.
Finite source exhaustion and response-stream EOF remain separate. EOF is
recorded after the response iterator drains, not inferred from input exhaustion
or an empty queue. ASR word timestamps remain disabled, and the standard
320 ms stop history remains unchanged without force-EOU.

Actor bounds describe the selected observed-compute timing model. They are not
backend completeness acknowledgements. Before source supply completes, ASR
advancement is conservative; later pending-work bounds retain the modeled
processing delay. These intervals include transport and queueing. Incremental
execution does not reconstruct the corresponding online arrival schedule.

### Native Services, Interruption, And Recording

The [response session](../src/examples/generic/offline/native_response.py)
keeps native NVIDIA LLM and TTS processors alive across turns. Their queues,
text filtering, sentence aggregation, and synthesis lifecycle remain active.
Actual frames become available before provider completion. User and assistant
context still use Pipecat's aggregators with sample-clock policy waits.

Interruption reaches the native processor queues and waits for acknowledgement
before a new generation starts. Generation identifiers fence late callbacks.
Context receives delivered text and control frames; the recorder does not infer
committed text from the audible PCM prefix.

The [incremental output adapter](../src/examples/generic/offline/output_stream.py)
retains native `MediaSender` buffering, FIFO order, partial-chunk flush, and
starvation resets. `WebsocketSinkClock` separates transport write completion
from client PCM presentation. Recording uses cumulative sample conversions,
fills gaps with silence, and retains final output beyond input duration.

The current scope requires a fixed recording, source-only ASR/VAD observation,
Silero turn detection, stitched Magpie synthesis, and no tools. History
summarization is explicitly rejected when the native eligibility condition is
reached; an incremental summarization adapter remains future work.

### Verification And Remaining Validation

Ten controlled engine tests cover early response generation before ASR EOF,
burst delivery, interruption, tied events, EOF policy, failure cleanup, and
acoustic-end versus VAD-stop offsets. One compares every request context and
final context against unmodified native aggregators receiving the same events.
Another exercises the full engine, queued native LLM/TTS, and output chain with
model I/O replaced. Additional tests cover provider lifecycle, actor/ASR
contracts, and output handling.

The full local suite passes with 888 tests, 3 skipped tests, and 178 subtests.
Ruff lint passes, and formatting checks pass for 289 Python files. These are
local controlled checks, not fresh ASR/LLM/TTS benchmark inference results.

Fresh local Silero inference on all five saved inputs (141,685.5 ms) matches the
reference decisions, intervals, and reset offsets. Incremental callback events
match the final trace, with monotone progress bounds. This checks VAD only.

Fresh paired runs for samples `2`, `3`, `250`, `503`, and `750` are planned
against the shared GPU 4 LLM and GPU 7 ASR/TTS services. The isolated CPU app
configuration uses port 17925. SSH connections time out before remote commands
execute, so the new real-service runs have not occurred. No new response WAVs,
latency results, or benchmark scores are available for this implementation.

The October 21 and October 22 experiments remain evidence for their recorded
implementations. They do not validate this runtime. Behavioral and scoring
consistency against realtime streaming remains in progress, including the
previously observed assistant-history differences.
