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

    Or run it under Docker Compose with the matching example profile, for example `--profile generic-assistant`. See [Getting Started](../01-getting-started.md) for the full list of profile combinations.

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

## Outputs

- Single-level run: `results_<timestamp>/` with `benchmark_summary.json`, `results.txt`, `results.tsv`, `results.json`, plus per-client logs and audio dumps.
- Sweep: `perf_suite_<timestamp>/` with suite-level `results.txt`, `results.tsv`, `results.json`, and one `run_<N>_clients/` directory per concurrency level.

See the perf README for the full output layout and per-flag reference:

- [`benchmarking_tools/scaling-perf/README.md`](../../benchmarking_tools/scaling-perf/README.md)
