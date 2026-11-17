# Run Scaling & Performance Tests

Use [`benchmarking_tools/scaling-perf/`](../../benchmarking_tools/scaling-perf/) to run latency and scaling tests against a running voice-agent server.

`benchmark.py` drives a single client. `simulate_concurrency.sh` orchestrates multiple parallel clients and rolls up suite-level results.

> **Note:** Performance numbers may vary based on hardware configuration (both CPU and GPU). Occasionally, higher latency may be observed due to uneven load balancing across FastAPI workers. For production deployments, using a Kubernetes setup is recommended to ensure stable load distribution and scalability.

## Setup

1. Create `benchmarking_tools/scaling-perf/dataset/` and add WAV files as the simulated user's utterances. The benchmark exits if the directory is missing or empty.

    ```bash
    mkdir -p benchmarking_tools/scaling-perf/dataset
    ```

    Prepare each file so turns are timed correctly:

    - **One continuous utterance per file, with no long internal pauses.** A long mid-file silence reads as the end of a turn, so the bot answers early and the turn is mis-timed.
    - **Trim all trailing silence** (for example in Audacity) so the end of the file is the end of the query. The benchmark times from the end of the WAV to the bot's response, and the scripts add silence between files automatically.
    - **Save as 16 kHz, mono, linear PCM (`int16`) WAV.**

    See [Setup in the perf README](../../benchmarking_tools/scaling-perf/README.md#setup) for the full rationale (reverse barge-ins, and client-side versus RTVI latency).
2. From the repository root, install benchmark dependencies once:

    ```bash
    uv sync --group benchmark
    ```

3. Start the voice-agent server with TLS enabled. The scaling benchmark connects to `wss://<host>:<port>/api/ws` and disables certificate verification for the local self-signed certificate.

    ```bash
    uv run python src/server.py
    ```

    To run perf tests against the Generic Cascaded example with the benchmark prompt catalog, set `selection: generic-assistant` in [`examples_registry.yaml`](../../examples_registry.yaml) and then start the server with the perf prompt catalog:

    ```bash
    uv run python src/server.py \
      --prompt-file benchmarking_tools/scaling-perf/perf_prompts.yaml
    ```

    Keep TLS enabled for the scaling benchmark. `PIPELINE_TLS=false` is for HTTP-only API tests and browser debugging, not for this WebSocket benchmark.

    Or run it under Docker Compose with `--profile generic-assistant/server-perf`. See the [scaling-perf README](../../benchmarking_tools/scaling-perf/README.md#reproducing-the-recommended-scaling-setup) for the four-GPU Blackwell layout, its pinned NVFP4 TP2 LLM profile, and the compatible-profile change required on older hardware.

## Run

From `benchmarking_tools/scaling-perf/`:

Single client (smoke test):

```bash
uv run python3 benchmark.py
```

Concurrent run (N parallel clients, single concurrency level):

```bash
./simulate_concurrency.sh --clients 4
```

Scaling sweep (one run per concurrency level, with cooldown between levels):

```bash
./simulate_concurrency.sh --clients "1 2 4 8 16"
```

Useful flags (passed through to `simulate_concurrency.sh`):

```bash
./simulate_concurrency.sh --host localhost --port 7860 --clients "1 4"
./simulate_concurrency.sh --clients "1 4" --test-duration 30 --cooldown 5
./simulate_concurrency.sh --clients "4" --no-save-audio
```

## Diagnose Individual Sessions

The `generic-assistant/server-perf` recipe enables `SCALING_PERF_DIAGNOSTICS=true` by default. Other Compose recipes leave it disabled. The observer sends bounded frame metadata to scaling clients without changing synthesis or audio pacing policy.

Enable capture on a host-native server with this command:

```bash
SCALING_PERF_DIAGNOSTICS=true uv run python src/server.py
```

To compare the performance recipe without server markers, recreate it with an explicit override:

```bash
SCALING_PERF_DIAGNOSTICS=false \
docker compose --profile generic-assistant/server-perf up -d
```

After a `simulate_concurrency.sh` run, inspect the artifacts in each concurrency directory. Direct `benchmark.py` results retain the underlying diagnostics in client JSON; use run aggregation to generate the CSV and text reports:

1. Open `session_quality.csv` for every client's completion status, raw glitch flag, captured event count, and diagnostic coverage.
2. Open `session_errors.json` for reported errors, including nonfatal errors from clients that returned valid responses.
3. Open `glitch_events.csv` for individual deficit durations, phases, classifications, confidence, and evidence.
4. Open `glitch_summary.txt` for measurement-window event and duration shares.

`Successful` means error-free completion, not guaranteed clean audio. `Degraded` retains valid-response sessions with recorded errors or failed turns. A raw playback-buffer deficit does not itself identify a TTS or CPU bottleneck. Missing or unaligned markers produce `unknown` classifications; inspect coverage instead of treating missing evidence as a pass.

The metadata extension is supported by the repository's RTVI endpoint and Realtime gateway. Other endpoints can omit it. Lifecycle markers contain identifiers, timing, PCM offsets, and boundary information. They omit spoken text, tool arguments/results, and PCM payloads. Error messages are captured separately with credential redaction. Standard transcript logs and optional WAV capture remain separate.

When supported, diagnostics-enabled RTVI readiness rejections include an HTTP 503 `service_not_ready` reason in the handshake report. This distinguishes a rejected startup from a later provider error; other endpoints can supply less detail.

Refer to [Session Diagnostics](../../benchmarking_tools/scaling-perf/README.md#session-diagnostics) for enable/disable behavior, completion definitions, privacy scope, classification limits, and reaggregation commands. Compare diagnostics-enabled and diagnostics-disabled runs when measuring observer overhead.

## Outputs

- Single-level run: `results_<timestamp>/` with `benchmark_summary.json`, `results.txt`, `results.tsv`, `results.json`, plus per-client logs and audio dumps.
- Sweep: `perf_suite_<timestamp>/` with suite-level `results.txt`, `results.tsv`, `results.json`, and one `run_<N>_clients/` directory per concurrency level.
- Run aggregation also writes `session_quality`, `session_errors`, and `glitch_events` files in JSON/CSV formats, plus `glitch_summary.txt`. These remain available for completed sessions with reported errors.

See the perf README for the full output layout and per-flag reference:

- [`benchmarking_tools/scaling-perf/README.md`](../../benchmarking_tools/scaling-perf/README.md)
