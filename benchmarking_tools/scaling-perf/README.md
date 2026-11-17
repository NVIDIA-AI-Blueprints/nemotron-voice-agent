# Scaling Perf

Load-test the Nemotron Voice Agent with synthetic clients. Each "client" is a
Python process that connects to the running server over WebSocket, plays a
WAV file as the user, listens for the bot's response, and records how long
each turn took. You can run a single client (smoke test) or fan out to N
parallel clients (concurrency / scaling test) and produce a sweep across
multiple concurrency levels.

The default RTVI benchmark connects directly to `WS /api/ws` and does not use
the server's session-config flow. That keeps it compatible with multi-worker
deployments such as `generic-assistant/server-perf`.

**RTVI** (Real-Time Voice/Video Inference) is the Pipecat-standard
protocol the server uses to push per-turn timing breakdowns (LLM / TTS /
ASR sub-latencies) to the client over the same WebSocket as the audio.
The benchmark parses those frames alongside the audio stream.

Use `--protocol realtime` or provide `--ws-url` to run the same client,
concurrency, pacing, latency, glitch, and aggregation logic against an
OpenAI Realtime-compatible WebSocket. Realtime audio uses base64 PCM in JSON
events instead of RTVI protobuf frames.

## Layout

| File | Job |
|------|-----|
| `benchmark.py` | Provides the stable command-line interface and dispatches single-client and aggregation modes. |
| `benchmark_core.py` | Runs the shared turn loop and handles result schema, audio timing, metric windows, and glitch detection. |
| `benchmark_aggregation.py` | Aggregates client runs and writes the TXT, TSV, and JSON reports. |
| `benchmark_diagnostics.py` | Collects session evidence, classifies captured buffer deficits after the test, and writes session quality and error reports. |
| `rtvi_transport.py` | Provides the low-level RTVI WebSocket, readiness handshake, and protobuf framing. |
| `rtvi_client.py` | Adapts RTVI audio, messages, and server metrics to the shared core. |
| `realtime_transport.py` | Provides the low-level Realtime WebSocket, authorization, JSON events, and base64 PCM operations. |
| `realtime_client.py` | Adapts Realtime audio, synchronization, events, and NVIDIA metric metadata to the shared core. |
| `simulate_concurrency.sh` | Spawns N parallel `benchmark.py` workers per concurrency level (with synchronized metric windows + cooldowns) and then calls `benchmark.py` in aggregate mode. |

## Setup

These scripts reuse the repo's root environment — no separate venv required.
Dependencies are managed via the
root `benchmark` dependency group (shared by every tool under
`benchmarking_tools/`).

1. From the repository root, sync the project (one-time):

   ```bash
   uv sync --group benchmark
   ```

   This host-side sync is still required even when the server runs under
   Docker Compose, because `benchmark.py` and `simulate_concurrency.sh` run
   on the host, not inside the app container.

2. Start the voice-agent server ([Docker compose](../../docs/01-getting-started.md) or `uv run python src/server.py`).
3. Add WAV files into `dataset/`. The benchmark cycles through them as the simulated user's utterances each turn. Prepare each file so the benchmark can time turns correctly:

   - **Record the query or reuse existing audio**, using generic queries or ones that match your specific use case.
   - **Use one continuous utterance per file, with no long internal pauses.** The server runs voice-activity detection and turn endpointing on the incoming audio, so a long silence in the middle of a file looks exactly like the end of a turn. The server then endpoints early and the bot starts answering a partial query, which is a false early response. That premature reply races the real end of your utterance, so the benchmark flags it as a *reverse barge-in* (see the [`--reverse-barge-in-threshold`](#run) flag) and discards it. The turn is mis-segmented and its latency becomes meaningless. Keeping each query a single clean utterance is what lets the benchmark produce correct, comparable numbers.
   - **Trim all trailing silence from the end** (for example in Audacity). This is critical for the client-side latency measurement. The benchmark times from the end of the WAV to when the bot's response arrives, so the end of the file must coincide with the end of the spoken query. The scripts insert silence *between* files automatically, so do not pad the files yourself.
   - **Save as 16 kHz, mono, linear PCM (`int16`) WAV.** This matches the pipeline's input format.

   If you do not trim the trailing silence, the client-side end-to-end latency is lower because the client keeps waiting through that silence before it starts measuring. The client-side E2E numbers are then wrong, but the **RTVI-based, server-reported metrics stay reliable** (`server_e2e`, `asr_ttfb`, `tts_ttfb`, `llm_ttfat`, and `llm_processing_time`) because they use server-side pipeline events rather than the end of the file. Refer to [What it measures](#what-it-measures).

`simulate_concurrency.sh` auto-dispatches `benchmark.py` through `uv run`
when the root `pyproject.toml` is detected, so the commands below work
straight from a fresh `uv sync --group benchmark`.

### Realtime Endpoint

Set the OpenAI Realtime-compatible WebSocket URL and, when required, its API
key:

```bash
export OPENAI_REALTIME_WS_URL=wss://realtime.example.com/v1/realtime
export OPENAI_REALTIME_API_KEY=<api-key>
```

The API key is sent as `Authorization: Bearer <api-key>`. Set
`OPENAI_REALTIME_AUTH_SCHEME` or use `--auth-scheme` for a different scheme,
such as `Api-Key`. The client omits the header when no key is provided. It
refuses to send a key over `ws://` or when `--insecure` disables TLS
verification. Use `--insecure` only for a local endpoint that does not require
a key.

The repository's gateway is available at
`wss://localhost:7860/v1/realtime`. It uses the development TLS certificate,
so local runs need `--insecure`. Leave `OPENAI_REALTIME_API_KEY` unset for
this command. The local gateway does not require a key.

```bash
uv run python3 benchmark.py \
  --protocol realtime \
  --ws-url wss://localhost:7860/v1/realtime \
  --drain-bot-intro \
  --insecure
```

The client drains one welcome response by default for both protocols, matching
the built-in examples. Use `--skip-bot-intro` when the target server has
welcome messages disabled.
For concurrency runs, use a server with the welcome message disabled so intro
draining does not overlap the synchronized measurement window. The
`generic-assistant/server-perf` Compose profile disables the welcome message.

For direct `benchmark.py` runs without explicit `--metrics-start-time` and
`--session-end-time` values, the metric window starts after connection setup
and optional intro draining. The configured `--test-duration` starts at that
point.

The client validates that every WAV is mono, uncompressed PCM16 with one common
sample rate, resamples input chunks to the OpenAI Realtime 24 kHz PCM format,
and sends a minimal `session.update`. It waits up to 60 seconds for the
corresponding `session.updated`, preserving the endpoint's voice activity
detection (VAD) defaults. It uses the output sample rate from that event, or
24 kHz when the event does not provide one.

The endpoint must use server-side VAD and create responses automatically. The
client does not send `input_audio_buffer.commit` or `response.create`. It keeps
sending PCM silence after each WAV and expects base64 mono PCM in
`response.output_audio.delta` events. A `response.done` event marks response
completion; any explicit status other than `completed` records a failed turn
and the next turn continues. In-turn `error` events are handled the same way,
while errors during session configuration remain fatal. Item-level failures do
not end the session; the current turn records their details in its `error`
metric while the response continues.

### Prompt override for perf runs

By default, `generic-assistant/server-perf` uses the same default prompt
as the normal Generic Assistant server profile.

If you want to experiment with custom prompts with different input-token sizes,
point the server at the prompt catalog in this directory and select the prompt
key you want:

```bash
PROMPT_FILE_PATH=/app/benchmarking_tools/scaling-perf/perf_prompts.yaml \
PROMPT_SELECTOR=prompt_200_tokens \
docker compose --profile generic-assistant/server-perf up -d
```

This catalog defaults to `prompt_1000_tokens`. Available prompt entries are
`prompt_200_tokens`, `prompt_1000_tokens`, and `prompt_5000_tokens`.

## Reproducing the recommended scaling setup

The recommended scaling setup uses four Blackwell GPUs with `1 GPU` for ASR,
`1 GPU` for TTS, and `2 GPUs` for the `Nemotron 3.5 Lightning 30B` LLM.

This setup is available as the dedicated Compose recipe
`generic-assistant/server-perf`. It automatically applies the published
scaling configuration:

- Generic Assistant inherits the existing `nemotron-lightning` default from
  [`examples_registry.yaml`](../../examples_registry.yaml)
- `nvidia-llm`: `NIM_MODEL_PROFILE=vllm-nvfp4-tp2-pp1-18.0`, GPUs `2,3`, alias
  `nvidia-llm`
- `nemotron-asr-streaming-english`:
  `NIM_TAGS_SELECTOR=type=en-US,mode=str,batch_size=128`, GPU `0`, alias
  `nemotron-asr-streaming-english`
- `magpie-multilingual-tts-service-perf`:
  `NIM_TAGS_SELECTOR=name=magpie-tts-multilingual,batch_size=64`, GPU `1`, alias
  `magpie-multilingual-tts-service`
- app env: `UVICORN_WORKERS=200`,
  `USE_SILERO_VAD_TURN_DETECTION=true`, `SILERO_VAD_STOP_SECS=0.5`,
  `AUDIO_OUT_10MS_CHUNKS=40`, `SCALING_PERF_DIAGNOSTICS=true` by default

> **Hardware-specific profile:** `vllm-nvfp4-tp2-pp1-18.0` was selected from
> `list-model-profiles` and benchmarked on two NVIDIA RTX PRO 6000 Blackwell
> GPUs. The checked-in pin is an RTX PRO 6000 benchmark baseline, not a portable
> recommendation. Before running this performance recipe on any other hardware,
> including H100, replace the pin by following these steps:
>
> 1. Run `list-model-profiles` with the deployed image on the actual LLM GPUs.
> 1. Benchmark the compatible TP2 profiles for time to first token, inter-token
>    latency, and total throughput per GPU.
> 1. Set `NIM_MODEL_PROFILE` to the winning profile's exact ID or full
>    description. H100 requires its own comparison of the listed FP8 and BF16
>    TP2 profiles.
>
> If portability matters more than predictable benchmark performance, remove
> the pin and let NIM select automatically. Do not use the deprecated NIM 1.x
> `NIM_TAGS_SELECTOR` for an LLM.
>
> See NVIDIA NIM's [model profile selection](https://docs.nvidia.com/nim/large-language-models/latest/deployment/model-profiles-and-selection.html)
> and [environment variable](https://docs.nvidia.com/nim/large-language-models/latest/reference/environment-variables.html)
> documentation.

Deploy it with:

```bash
docker compose --profile generic-assistant/server-perf up -d
```

After the stack is healthy, run the sweep from this directory:

```bash
./simulate_concurrency.sh --clients "1 2 4 8 16" --insecure
```

## Run

From this directory:

```bash
# Single RTVI client (1 process)
uv run python3 benchmark.py --insecure

# Concurrent RTVI run (4 parallel processes, single concurrency level)
./simulate_concurrency.sh --clients 4 --insecure

# RTVI scaling sweep (one run per concurrency level, cooldown between levels)
./simulate_concurrency.sh --clients "1 2 4 8 16" --insecure

# Single Realtime client
uv run python3 benchmark.py --protocol realtime --ws-url "$OPENAI_REALTIME_WS_URL"

# Realtime scaling sweep
./simulate_concurrency.sh \
  --protocol realtime \
  --ws-url "$OPENAI_REALTIME_WS_URL" \
  --clients "1 2 4 8 16"
```

The shell wrapper accepts `-h`/`--help`. These flags apply to both the shell
wrapper and `benchmark.py` unless the description states otherwise:

| Flag | Default | Description |
|------|---------|-------------|
| `--clients "N1 N2 …"` | `1` | Shell wrapper only. One run per concurrency level. Quote the list. |
| `--host` / `--port` | `localhost` / `7860` | RTVI server target. |
| `--protocol` | `rtvi` | Select `realtime` for an OpenAI Realtime WebSocket. Providing `--ws-url` or setting `OPENAI_REALTIME_WS_URL` also selects Realtime mode when you omit this flag. |
| `--ws-url` | unset; falls back to `OPENAI_REALTIME_WS_URL` | Full Realtime WebSocket URL. Required in Realtime mode. |
| `--auth-scheme` | `Bearer`; falls back to `OPENAI_REALTIME_AUTH_SCHEME` | Authorization scheme for the Realtime API key. |
| `--connect-timeout` | RTVI: `30`; Realtime: `60` | WebSocket handshake timeout in seconds. In Realtime mode, the same value bounds the `session.updated` readiness wait. |
| `--turn-response-timeout` | RTVI: `10`; Realtime: `45` | Seconds to wait for first response audio after the WAV ends. |
| `--insecure` | off | Disable TLS certificate verification for Realtime or RTVI. Intended only for local development certificates. Realtime mode rejects an API key when this flag is set. |
| `--skip-bot-intro` | off | Skip draining an initial assistant utterance. Use for targets with welcome messages disabled. Mutually exclusive with `--drain-bot-intro`. |
| `--drain-bot-intro` | on | Wait for and discard an initial assistant utterance before opening the metric window. Mutually exclusive with `--skip-bot-intro`. |
| `--bot-intro-timeout` | `5` | Seconds to wait for initial bot audio. Increase this for high-latency cloud deployments. |
| `--test-duration` | `300` | Seconds of metric collection per level. |
| `--client-start-delay` | `1` | Shell wrapper only. Stagger between client connection attempts in seconds. With N clients and delay D, the shared metric window opens at `now + (N-1)*D`. |
| `--cooldown` | `10` | Shell wrapper only. Pause between sweep levels (s) — lets the server settle between bursts. |
| `--reverse-barge-in-threshold` | `0.4` | Bot audio arriving within this many seconds of the user finishing speaking is discarded as a *reverse* barge-in (the server racing the end of the user's utterance) instead of being timed as the real response. Used internally. Not surfaced in summaries. |
| `--no-save-audio` | (audio saved) | Skip writing per-client output WAVs. |
| `--dataset-dir DIR` | `./dataset` | Override input WAV directory. |
| `--output-dir DIR` | this folder | Override result destination. |

`Ctrl-C` is graceful — workers stop, partial results stay on disk.

## Output layout

**Where to look first:** open `results.txt` for `simulate_concurrency.sh` runs
(single-level or sweeps). For direct `uv run python3 benchmark.py` runs, check
the client summary line and `result_<id>.json`. Per-client `.log` files are
mainly for debugging specific failures.

A direct run exits with a nonzero status when `result.error` is set or no valid
turn completes. It still writes the result JSON and log for diagnosis.

Single concurrency level (`--clients 1`, `--clients 4`, etc.):

```
results_<timestamp>/
├── benchmark_summary.json       # rolled-up summary across all clients
├── results.txt                  # one-row summary table (human-readable)
├── results.tsv                  # one-row summary table (tab-separated)
├── results.json                 # one-row summary object list
├── glitch_events.{json,csv}      # one row per captured buffer deficit, all phases
├── glitch_summary.txt            # classified measurement-window counts and durations
├── session_errors.{json,csv}     # reported errors, including completed sessions
├── session_quality.{json,csv}    # one row per client with status and evidence coverage
└── client_<i>_<unix_ms>/        # i = 1..N, unix_ms makes the dir unique
    ├── benchmark_<id>.log       # turn-by-turn log
    ├── result_<id>.json         # per-client metrics, parsed back by aggregation
    └── audio_output_<id>.wav    # bot audio captured by this client (unless --no-save-audio)
```

Multi-level sweep (`--clients "1 4 16"`):

```
perf_suite_<timestamp>/
├── results.txt                  # column-aligned, human-readable
├── results.tsv                  # tab-separated (spreadsheets / pandas)
├── results.json                 # one object per concurrency level
└── run_<N>_clients/             # one of these per --clients value
    ├── benchmark_summary.json
    ├── glitch_events.{json,csv}
    ├── glitch_summary.txt
    ├── session_errors.{json,csv}
    ├── session_quality.{json,csv}
    └── client_<i>_<unix_ms>/...
```

## Session Diagnostics

Enable server frame metadata when you need to investigate individual buffer deficits, tool-followup waits, or reported errors. The server uses Pipecat's native frame observer through the shared pipeline helper. It does not patch TTS providers or transport methods.

### Enable or Disable Capture

The `generic-assistant/server-perf` Compose recipe enables `SCALING_PERF_DIAGNOSTICS=true` by default. Other recipes default to `false`. You can override the value explicitly for a comparison:

```bash
# Disable server markers while keeping the same performance recipe.
SCALING_PERF_DIAGNOSTICS=false \
docker compose --profile generic-assistant/server-perf up -d

# Enable server markers again.
SCALING_PERF_DIAGNOSTICS=true \
docker compose --profile generic-assistant/server-perf up -d
```

For a host-native server, set the environment variable on the server process:

```bash
SCALING_PERF_DIAGNOSTICS=true uv run python src/server.py
```

The benchmark always records received audio timing and available protocol errors. Server markers are optional. The repository's RTVI endpoint and Realtime gateway can deliver them; other endpoints can lack this extension. Inspect coverage before interpreting a classification.

Diagnostic batches use type `nva.perf.diagnostics` and schema version `1`. They carry session identifiers, event sequence numbers, timestamps, audio sequence numbers, and cumulative PCM offsets. Lifecycle markers cover LLM and tool activity, audio epochs, sentence boundaries, interruptions, and errors. They omit spoken text, tool arguments, tool results, and audio payloads. Structured diagnostic error records use Pipecat's credential redaction. Existing raw protocol/transcript logging and optional WAV capture remain separate from these metadata markers.

Capture uses bounded event collections and batched delivery. Dropped events and sequence gaps are reported. The observer adds metadata, CPU work, and transport traffic; compare enabled and disabled runs with the same workload when evaluating overhead.

### Inspect Errors and Completion Separately

Run aggregation writes the following artifacts beside `benchmark_summary.json`. In a sweep, inspect each `run_<N>_clients/` directory.

| Artifact | Purpose |
|---|---|
| `session_quality.json` and `.csv` | One row per client: status, valid and failed turns, reported error count, raw glitch flag, captured event count, and diagnostic coverage |
| `session_errors.json` and `.csv` | Errors associated with each client, including errors from sessions that later return valid responses |
| `glitch_events.json` and `.csv` | Each captured deficit with client, phase, audio cursor, classification, confidence, and supporting evidence |
| `glitch_summary.txt` | Measurement-window event counts, summed deficits, and event/duration shares by classification |

Per-client result JSON retains the underlying `diagnostics` and `session_errors` collections. Error records distinguish the reporting source and component where available. Component labels use processor or protocol evidence; they do not establish the underlying cause. Records retain fatal/nonfatal information when supplied. Session errors include warmup and reply-drain errors, with the phase preserved.

With diagnostics enabled, the RTVI server can return an HTTP 503 admission response containing `code: service_not_ready`, a component, and a redacted reason. This requires ASGI WebSocket denial-response support. Otherwise, the server retains its close-based rejection. The RTVI client records a structured admission reason as `source: handshake` when available, so it can be distinguished from an error after admission.

The completion categories have these meanings:

| Status | Meaning |
|---|---|
| `successful` | Valid response without recorded session errors, failed turns, or a session deadline; this is error-free completion, not a clean-audio guarantee |
| `degraded` | Valid response with recorded errors, failed turns, or a hard deadline |
| `failed` | A qualifying runtime error, even if earlier responses completed |
| `no_response` | No valid response and no qualifying runtime failure |

`Successful` and `Degraded` are separate columns in `results.txt`; their sum counts completed clients with valid responses. The JSON field `completed_clients` retains that combined count. `Errors` counts reported error records, while `Glitches` retains the legacy client flag count. A nonfatal provider error can therefore reduce `Successful` and increase `Degraded` without ending the client session.

Glitch evidence is orthogonal to completion status. A client can be `successful` and still have buffer deficits. Any recorded session error, including one outside the measurement window, can make completion degraded. A direct client command or aggregation command returning zero is not a strict quality pass; inspect session errors and gap coverage alongside latency.

### Interpret Post-Test Gap Classifications

The detector estimates how much received PCM remains for playback. A deficit greater than 20 ms records an event; it is not a listening-confirmed audio defect. Classification runs after the test so delayed markers can be considered. It uses output audio offsets and lifecycle evidence rather than guessing sentence boundaries from arrival-time pauses. Offsets describe queued PCM and sentence units, not forced word-to-audio alignment.

| Classification | Evidence Used |
|---|---|
| `within_sentence` | The playback cursor falls inside an aligned sentence unit |
| `between_sentences` | An aligned sentence boundary matches the playback cursor, with preceding aligned sentence evidence |
| `tool_wait` | First audio of a tool-driven follow-up epoch; this does not measure tool execution time |
| `interruption` | An interruption marker is adjacent to the playback boundary |
| `unknown` | Capture, audio epoch, or sentence-alignment evidence is missing or incomplete |

Each event retains `confidence` and `evidence`. `audio_offset` confidence indicates aligned sentence/audio offset evidence. `frame_evidence` indicates lifecycle evidence. `unavailable` means the classifier cannot support a more specific label. These values are evidence categories, not calibrated probabilities.

Coverage reports server marker availability, dropped server/client events, and sequence gaps. Capture loss forces conservative `unknown` classification. The classifier also requires client/server audio-epoch duration and packet counts to agree; mismatches remain `unknown`. Initial audio alone does not establish a gap between two sentences. Token-streaming providers can lack reliable sentence-to-audio alignment; those gaps also remain `unknown`. A boundary label describes context, not whether TTS generation, CPU scheduling, network delivery, or intentional application behavior caused the wait.

Events retain `warmup`, `measurement`, `reply_drain`, or `unknown` phase. The JSON/CSV reports preserve all phases; `glitch_summary.txt` compares measurement-window events by both count and summed deficit duration. Neither zero captured events nor an `unknown` label establishes glitch-free playback when capture is unavailable. A raw-flagged client without captured gap events is counted in `flagged_clients_without_event_evidence`; the text summary states that its event count and duration are unavailable.

Older results remain useful for recorded protocol errors, but missing markers or a Boolean glitch flag cannot reconstruct a per-event sentence classification. Legacy error extraction preserves `recorded_timestamp` when available, leaves `received_unix_s` as `null`, and uses `phase: unknown`. It does not invent a new historical occurrence time. Keep those coverage limits visible when comparing runs.

To regenerate run-level reports from saved client results, run this command from this directory:

```bash
uv run python3 benchmark.py --aggregate-run-dir /path/to/results --num-clients 4
```

Use the original configured client count. Aggregation cannot recover missing diagnostic events or client files.

## What it measures

Per-client (the core target of this tool):

- end-to-end response latency per turn (avg / p95 / min / max), measured at
  the simulated user — the wall-clock from "user finished speaking" to
  "first audio frame of the real response".
- **audio glitch** detection — flagged when the output buffer underruns
  (i.e. the player would have to insert silence to keep up).

Pulled from the server over RTVI (each metric is weighted-averaged by sample
count across all clients in a run):

| Metric | Meaning |
|--------|---------|
| `llm_ttfb` | Time from the LLM request to the first LLM output, including reasoning output |
| `llm_thinking_time` | Time between the first LLM output and first answer token |
| `llm_ttfat` | Time from the LLM request to the first answer token, or to the start of a tool call (`llm_ttfb` plus thinking time) |
| `tts_ttfb` | Time-to-first-byte from the TTS |
| `tts_leading_silence` | Silence before the first audible sample in the TTS response |
| `tts_ttfa` | Time from the TTS request to the first playable audio |
| `asr_ttfb` | Time-to-first-byte from the ASR |
| `server_e2e` | Server-side end-to-end (user-stop → first bot speech) |
| `turn_taking_latency` | Time from the acoustic VAD-stop signal to the LLM request, including ASR finalization and whichever turn strategy is configured |
| `text_aggregation` | Time from the first LLM token to the first complete sentence sent to TTS |
| `llm_processing_time` | LLM end-to-end (request → final token) |
| `llm_tokens_per_sec` | Completion tokens / `llm_processing_time` |

The LLM metrics (`llm_ttfb`, `llm_ttfat`, `llm_processing_time`, and
`llm_tokens_per_sec`) have one sample per LLM request, not one per user turn.
A turn that calls a tool makes two LLM requests, so it adds two samples to each
LLM metric: one for the tool call and one for the follow-up answer.
`llm_ttfb` and `llm_ttfat` are both counted per LLM request. The
`llm_thinking_time` and `tts_leading_silence` columns use Pipecat's direct
breakdown fields rather than recomputing differences between independently
aggregated samples. Existing result files that use the former `llm_ttft` key
remain readable. To measure latency per user turn, use `server_e2e`.
Speech-to-speech LLM services do not report `llm_ttfat`.
Pipecat's RTVI token-usage entries do not include a processor identity, so
`llm_tokens_per_sec` pairs each usage record with the next LLM processing
record. It is reliable for the benchmark's single-LLM Generic Assistant path.
Do not treat it as per-model throughput in a pipeline with multiple LLMs.

The Nemotron Realtime gateway maps Pipecat token accounting to the standard
`response.done.response.usage` object. It also places response-scoped server
timings in the string-valued `response.metadata.nvidia_metrics` field. The
Realtime client decodes that compact JSON and uses it to populate the same
server-metric columns as RTVI. Other Realtime endpoints that do not provide
this metadata continue to report those columns as `N/A`.

Per-client Realtime result JSON also includes `realtime_turn_metrics`. Each
Realtime turn, including failed turns, records these values in seconds:

- `input_end_to_speech_stopped`: time from the last input WAV frame to the
  server's `input_audio_buffer.speech_stopped` event.
- `from_speech_stopped`: `asr_transcription_completed`, `response_created`,
  `first_output_transcript`, `first_output_audio`, and `response_done`, all
  measured from that server-confirmed speech-stop event.
- `stage_deltas`: `response_after_asr`,
  `first_text_after_response_created`, and `first_audio_after_first_text`.
- `response_status`, `usage`, and `server_metrics`: values extracted from
  `response.done`, when provided.
- `error`: a timeout, no-audio, or failed-response description when applicable.

The client records the first matching event of each type after a turn starts.
If the server does not send a speech-stop event, speech-relative values are
`null`. Failed records can also contain other `null` lifecycle fields. These
metrics use client-observed event arrival times, not server processing
timestamps, so they do not replace server-side RTVI metrics.
