# Pipecat 1.15 Upgrade

This upgrade moves Nemotron Voice Agent from `pipecat-ai` 1.12.0 to 1.15.0
and updates the browser client to RTVI protocol 2.3-compatible packages.

## Dependency changes

- `pipecat-ai[nvidia,silero,runner,webrtc,websocket,openai]`: 1.12.0 → 1.15.0
- `pipecat-ai[evals]`: 1.12.0 → 1.15.0
- `@pipecat-ai/client-js`: 1.13.1 → 1.16.0
- `@pipecat-ai/client-react`: 1.8.2 → 1.11.0
- `@pipecat-ai/small-webrtc-transport`: 1.10.8 → 1.11.2
- `@pipecat-ai/websocket-transport`: 1.7.2 → 1.8.2

The Python and npm lockfiles were regenerated from these direct pins. Expected
transitive updates include `aiohttp`, `pipecat-ai-prebuilt`, and
`pipecat-ai-context-hub`. No Pipecat companion package was directly pinned or
needed removal.

## Runtime migrations

- Pipecat 1.13 made `get_llm_invocation_params()` asynchronous. The custom
  streaming LLM and Omni request builders now await it:
  [PR #6059](https://github.com/pipecat-ai/pipecat/pull/6059).
- Pipecat 1.15 made NVIDIA TTS synthesis-configuration discovery asynchronous.
  TTS catalog prewarming now runs the coroutine inside its bounded worker
  thread, retaining the existing synchronous cache API and returning without
  waiting for a timed-out worker.
- Pipecat 1.13 can start pipeline frame flow while processors are still
  completing setup. Initial greetings now require both client readiness and
  `on_pipeline_started`:
  [PR #6167](https://github.com/pipecat-ai/pipecat/pull/6167).
- Pipecat 1.14 emits one NVIDIA transcription frame per diarized speaker run.
  The custom speaker processor consumes the native numeric `user_id` tag
  without re-splitting the shared raw result, while preserving legacy result
  handling and UI speaker numbering:
  [PR #6076](https://github.com/pipecat-ai/pipecat/pull/6076).
- Adapter imports moved from `pipecat.adapters.services` and
  `pipecat.adapters.schemas` to `pipecat.adapters.llm.*`:
  [PR #6331](https://github.com/pipecat-ai/pipecat/pull/6331).
- Deprecated TTS `BaseTextFilter` objects were replaced by asynchronous
  `text_transforms`; the NVIDIA reserved-character cleanup and airline
  pronunciation transform retain their previous order:
  [PR #6264](https://github.com/pipecat-ai/pipecat/pull/6264).
- Audio recording now uses the structured `on_user_turn_audio` and
  `on_bot_turn_audio` events instead of deprecated per-speech-run callbacks.
- The custom Omni service uses `on_user_turn_message_added` for complete
  realtime transcripts, reports completed transcript frames as finalized,
  closes one-shot inference streams on cancellation, and explicitly reports
  TTFAT while retaining realtime aggregation semantics.
- The custom word-timestamp TTS sequencer now discards Pipecat 1.15 buffered
  word state when force-completing a context.
- Deprecated `AggregationType` and `aggregated_by` test fixtures now use
  `TextType` and `text_type`:
  [PR #6072](https://github.com/pipecat-ai/pipecat/pull/6072).

## Evals and transport

Local evals now use the development runner's accepted `/ws` connection. The
shared transport factory delegates `WebSocketRunnerArguments` to Pipecat's
factory, which consumes the initial `eval-connect` message and creates
`EvalFastAPIWebsocketTransport`. Manifests no longer pass deprecated
`-t eval`:
[PR #6064](https://github.com/pipecat-ai/pipecat/pull/6064).

The deterministic CI scenario exercises the new `text_matches` assertion. A
service-backed audio scenario inserts a 300 ms mid-utterance pause and checks
that the full request remains one usable turn:
[PR #5991](https://github.com/pipecat-ai/pipecat/pull/5991),
[PR #6164](https://github.com/pipecat-ai/pipecat/pull/6164).

## Metrics and benchmarks

- The scaling RTVI client now sends Pipecat's current protocol constant rather
  than the obsolete hard-coded `0.1.0`.
- Response and scaling telemetry use canonical `llm_ttfb`, add `tts_ttfa` and
  retain `llm_ttfat`. Existing result artifacts containing `llm_ttft` remain
  readable.
- Pipecat 1.9 latency breakdowns are consumed directly: LLM thinking time,
  TTS leading silence, and application text-aggregation time now have
  dedicated scaling columns. Embedded TTFA/TTFAT TTFB values are not counted
  again as standalone TTFB samples.
- Scaling clients reject non-finite and zero-value startup artifacts. Token
  throughput is reported only for the single-LLM Generic pipeline because
  RTVI usage records do not identify their originating processor.
- Realtime response metrics are response-scoped across cancellation, so late
  frames from a cancelled generation cannot contaminate the next response.
- Turn-taking latency is strategy-neutral and measures the interval from the
  acoustic VAD-stop frame to the LLM request. It includes ASR finalization and
  fixed, Smart Turn, or other configured turn-strategy processing.
- OpenAI-compatible `speech_started` and `speech_stopped` events now reflect
  raw VAD boundaries when available. Audio commit, transcript completion, and
  turn lifecycle remain tied to finalized user-turn frames. Pipelines without
  raw VAD frames retain finalized-turn fallback behavior.

## Behavior inherited from 1.13–1.15

- Silero was upgraded to v6.2 and its recurrent state is retained for the
  stream, fixing periodic state-reset stops during speech:
  [PR #6202](https://github.com/pipecat-ai/pipecat/pull/6202).
- RTVI advances through protocol 2.2 and 2.3, including capability negotiation
  and newer output metadata.
- NVIDIA TTS startup configuration fetches no longer block the event loop.
- NVIDIA STT stream-drop cleanup and PipelineWorker lifecycle handling include
  upstream fixes from these releases.

Release notes:
[1.13.0](https://github.com/pipecat-ai/pipecat/releases/tag/v1.13.0),
[1.14.0](https://github.com/pipecat-ai/pipecat/releases/tag/v1.14.0), and
[1.15.0](https://github.com/pipecat-ai/pipecat/releases/tag/v1.15.0).

## Deferred opportunities

- Replacing the custom Omni service with Pipecat's native
  `NvidiaOmniLLMService` requires explicit parity tests for transcript
  ownership, preemption, media conversion, reasoning, and subagent behavior.
- Expressive conversation/perception, classifier turn completion, predictive
  backend delegation, and `AICTytoObserver` are intentionally deferred because
  they add product behavior or dependencies rather than upgrade compatibility.
- Reintroducing force EOU remains a separate latency/accuracy decision. This
  upgrade includes the Silero state fix but does not change ASR force-padding
  policy.

## Validation

Completed validation:

- Imports for the server, shared code, and all five examples.
- Ruff format and lint checks across 247 files.
- 667 unit tests, 3 skipped tests, and 134 subtests.
- Eval WebSocket transport smoke checks.
- The two-scenario service-free Pipecat eval suite, including `text_matches`.
- Parsing all 30 scripted eval scenarios, including the new paused-audio case.
- Client `npm ci`, ESLint, TypeScript, and Vite production build.
- Ten concurrent real-time replays of
  `silero_vad_today_pad_768ms.wav` through Pipecat 1.15 Silero v6.2. All ten
  produced identical VAD transitions and no mid-word reset stop.
- Direct hosted NVIDIA ASR and TTS endpoint smoke tests.
- The 23 non-Omni hosted service scenarios: 18 passed on the full sweep.
  Retrying the five failures with an explicitly cloud-only recipe cleared both
  model-dependent text scenarios. The three audio scenarios consistently
  reached the hosted Magpie TTS endpoint but received no audio followed by HTTP
  504 responses, including on retry; this is an upstream service failure rather
  than an upgrade regression.
- A 1/2/4-client hosted-service scaling sweep. The 1- and 2-client runs
  completed without failures; all four clients in the 4-client run timed out
  after 30 seconds while hosted LLM/TTS latency increased. The result artifact
  includes the Pipecat 1.9 breakdown columns described above.

The five Omni service scenarios require a local Omni NIM on port 18002 and
were not run in this environment.
