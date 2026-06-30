# Scaling Perf

Load-test the Nemotron Voice Agent with synthetic clients. Each "client" is a
Python process that connects to the running server over WebSocket, plays a
WAV file as the user, listens for the bot's response, and records how long
each turn took. You can run a single client (smoke test) or fan out to N
parallel clients (concurrency / scaling test) and produce a sweep across
multiple concurrency levels.

The scaling benchmark connects directly to `WS /api/ws` and does not use the
server's session-config flow. That keeps it compatible with multi-worker
deployments such as `generic-assistant/workstation-perf`.

**RTVI** (Real-Time Voice/Video Inference) is the Pipecat-standard
protocol the server uses to push per-turn timing breakdowns (LLM / TTS /
ASR sub-latencies) to the client over the same WebSocket as the audio.
The benchmark parses those frames alongside the audio stream.

## Layout

| File | Job |
|------|-----|
| `benchmark.py` | Drives **one** client by default. Also produces summaries when invoked with `--aggregate-run-dir` / `--aggregate-suite-dir`. |
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

   If you do not trim the trailing silence, the client-side end-to-end latency is lower, because the clients keeps waiting through that silence before it start measuring. The client-side E2E numbers are then wrong, but the **RTVI-based, server-reported metrics stay reliable** (`server_e2e`, `asr_ttfb`, `tts_ttfb`, and `llm_processing_time`), because they are measured at the server from the actual end-of-speech and turn events rather than from the end of the file. See [What it measures](#what-it-measures).

`simulate_concurrency.sh` auto-dispatches `benchmark.py` through `uv run`
when the root `pyproject.toml` is detected, so the commands below work
straight from a fresh `uv sync --group benchmark`.

### Prompt override for perf runs

By default, `generic-assistant/workstation-perf` uses the same default prompt
as the normal Generic Assistant workstation profile.

If you want to experiment with custom prompts with different input-token sizes,
point the server at the prompt catalog in this directory and select the prompt
key you want:

```bash
PROMPT_FILE_PATH=/app/benchmarking_tools/scaling-perf/perf_prompts.yaml \
PROMPT_SELECTOR=prompt_200_tokens \
docker compose --profile generic-assistant/workstation-perf up -d
```

This catalog defaults to `prompt_1000_tokens`. Available prompt entries are
`prompt_200_tokens`, `prompt_1000_tokens`, and `prompt_5000_tokens`.

## Reproducing the best scaling setup

For the best scaling numbers, use a `4xH100` setup with `1 GPU` for ASR,
`1 GPU` for TTS, and `2 GPUs` for the `Nemotron Nano 30B` LLM.

This setup is available as the dedicated Compose recipe
`generic-assistant/workstation-perf`. It automatically applies the published
scaling configuration:

- Generic Assistant inherits the existing `nemotron-nano` default from
  [`examples_registry.yaml`](../../examples_registry.yaml)
- `nvidia-llm`: `NIM_TAGS_SELECTOR=precision=fp8,tp=2`, GPUs `2,3`, alias
  `nvidia-llm`
- `nemotron-asr-streaming-english`: GPU `0`, alias
  `nemotron-asr-streaming-english`
- `tts-service`: `NIM_TAGS_SELECTOR=name=magpie-tts-multilingual,batch_size=64`,
  GPU `1`, alias `tts-service`
- app env: `UVICORN_WORKERS=200`,
  `USE_SILERO_VAD_TURN_DETECTION=true`, `SILERO_VAD_STOP_SECS=0.5`,
  `AUDIO_OUT_10MS_CHUNKS=40`

Deploy it with:

```bash
docker compose --profile generic-assistant/workstation-perf up -d
```

After the stack is healthy, run the sweep from this directory:

```bash
./simulate_concurrency.sh --clients "1 2 4 8 16"
```

## Run

From this directory:

```bash
# Single-client (1 process)
uv run python3 benchmark.py

# Concurrent run (4 parallel processes, single concurrency level)
./simulate_concurrency.sh --clients 4

# Scaling sweep (one run per concurrency level; cooldown between levels)
./simulate_concurrency.sh --clients "1 2 4 8 16"
```

The shell wrapper accepts `-h`/`--help`. Common flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--clients "N1 N2 …"` | `1` | One run per concurrency level. Quote the list. |
| `--host` / `--port` | `localhost` / `7860` | Server target. |
| `--test-duration` | `300` | Seconds of metric collection per level. |
| `--client-start-delay` | `1` | Stagger between clients connecting (s). With N clients and delay D, the metric window opens at ``now + (N-1)*D`` so every worker is connected before measurement starts. |
| `--cooldown` | `10` | Pause between sweep levels (s) — lets the server settle between bursts. |
| `--reverse-barge-in-threshold` | `0.4` | Bot audio arriving within this many seconds of the user finishing speaking is discarded as a *reverse* barge-in (the server racing the end of the user's utterance) instead of being timed as the real response. Used internally; not surfaced in summaries. |
| `--no-save-audio` | (audio saved) | Skip writing per-client output WAVs. |
| `--dataset-dir DIR` | `./dataset` | Override input WAV directory. |
| `--output-dir DIR` | this folder | Override result destination. |

`Ctrl-C` is graceful — workers stop, partial results stay on disk.

## Output layout

**Where to look first:** open `results.txt` for `simulate_concurrency.sh` runs
(single-level or sweeps). For direct `uv run python3 benchmark.py` runs, check
the client summary line and `result_<id>.json`. Per-client `.log` files are
mainly for debugging specific failures.

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
