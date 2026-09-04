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
| `benchmark.py` | Drives **one** client by default. Also produces summaries when invoked with `--aggregate-run-dir` / `--aggregate-suite-dir`. |
| `openai_realtime_ws.py` | Handles the Realtime session handshake, JSON events, authorization, and base64 PCM audio. |
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

   If you do not trim the trailing silence, the client-side end-to-end latency is lower because the client keeps waiting through that silence before it starts measuring. The client-side E2E numbers are then wrong, but the **RTVI-based, server-reported metrics stay reliable** (`server_e2e`, `asr_ttfb`, `tts_ttfb`, and `llm_processing_time`) because they are measured at the server from the actual end-of-speech and turn events rather than from the end of the file. See [What it measures](#what-it-measures).

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
such as `Api-Key`. The client omits the header when no key is provided.

The repository's gateway is available at
`wss://localhost:7860/v1/realtime`. It uses the development TLS certificate,
so local runs need `--insecure`:

```bash
uv run python3 benchmark.py \
  --protocol realtime \
  --ws-url wss://localhost:7860/v1/realtime \
  --drain-bot-intro \
  --insecure
```

The built-in examples enable a welcome response by default. When you use one
of these examples, drain the response before sending the first measured turn.
Use `--skip-bot-intro` when the target server has welcome messages disabled.
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
while errors during session configuration remain fatal. Item-level failures
remain in the event log while the response continues.

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
  `AUDIO_OUT_10MS_CHUNKS=40`

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
./simulate_concurrency.sh --clients "1 2 4 8 16"
```

## Run

From this directory:

```bash
# Single RTVI client (1 process)
uv run python3 benchmark.py

# Concurrent RTVI run (4 parallel processes, single concurrency level)
./simulate_concurrency.sh --clients 4

# RTVI scaling sweep (one run per concurrency level, cooldown between levels)
./simulate_concurrency.sh --clients "1 2 4 8 16"

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
| `--insecure` | off | Disable TLS certificate verification for Realtime. Intended for local development certificates. |
| `--skip-bot-intro` | RTVI: off; Realtime: on | Skip draining an initial assistant utterance. Realtime skips it by default. Mutually exclusive with `--drain-bot-intro`. |
| `--drain-bot-intro` | RTVI: on; Realtime: off | Wait for and discard an initial assistant utterance. Required for Realtime targets that emit a welcome response. Mutually exclusive with `--skip-bot-intro`. |
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
    └── client_<i>_<unix_ms>/...
```

## What it measures

Per-client (the core target of this tool):

- end-to-end response latency per turn (avg / p95 / min / max), measured at
  the simulated user — the wall-clock from "user finished speaking" to
  "first audio frame of the real response".
- **audio glitch** detection — flagged when the output buffer underruns
  (i.e. the player would have to insert silence to keep up).

Pulled from the server over RTVI (each value is reported per turn,
weighted-averaged across all turns/clients in a run):

| Metric | Meaning |
|--------|---------|
| `llm_ttft` | Time-to-first-token from the LLM |
| `tts_ttfb` | Time-to-first-byte from the TTS |
| `asr_ttfb` | Time-to-first-byte from the ASR |
| `server_e2e` | Server-side end-to-end (user-stop → first bot speech) |
| `vad_smart_turn` | VAD + smart-turn analyzer time |
| `llm_processing_time` | LLM end-to-end (request → final token) |
| `llm_tokens_per_sec` | Completion tokens / `llm_processing_time` |

The OpenAI Realtime protocol does not carry RTVI timing frames, so RTVI
server-metric columns are `N/A` for Realtime endpoints. Per-client Realtime
result JSON instead includes `realtime_turn_metrics`. Each Realtime turn,
including failed turns, records these values in seconds:

- `input_end_to_speech_stopped`: time from the last input WAV frame to the
  server's `input_audio_buffer.speech_stopped` event.
- `from_speech_stopped`: `asr_transcription_completed`, `response_created`,
  `first_output_transcript`, `first_output_audio`, and `response_done`, all
  measured from that server-confirmed speech-stop event.
- `stage_deltas`: `response_after_asr`,
  `first_text_after_response_created`, and `first_audio_after_first_text`.
- `response_status` and `usage`: values from `response.done`, when provided.
- `error`: a timeout, no-audio, or failed-response description when applicable.

The client records the first matching event of each type after a turn starts.
If the server does not send a speech-stop event, speech-relative values are
`null`. Failed records can also contain other `null` lifecycle fields. These
metrics use client-observed event arrival times, not server processing
timestamps, so they do not replace server-side RTVI metrics.
