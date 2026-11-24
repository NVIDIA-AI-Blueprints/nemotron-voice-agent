# Run Scaling and Performance Tests

Use [`benchmarking_tools/scaling-perf/`](../../benchmarking_tools/scaling-perf/) to run latency and scaling tests against a running voice-agent server. This guide applies to the Pipecat examples (`nva-pipecat`).

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

3. Start the voice-agent server with TLS enabled. The benchmark connects to the Pipecat RTVI endpoint at `wss://<host>:<port>/api/ws` and verifies the certificate by default.

    ```bash
    uv run nva-pipecat
    ```

    To run perf tests against the Generic Assistant example with the benchmark prompt catalog, set `pipecat.selection` to `generic-assistant` in [`examples_registry.yaml`](../../examples_registry.yaml) and then start the server with the perf prompt catalog:

    ```bash
    uv run nva-pipecat \
      --prompt-file benchmarking_tools/scaling-perf/perf_prompts.yaml
    ```

    Keep TLS enabled for the scaling benchmark. `PIPELINE_TLS=false` is for HTTP-only API tests and browser debugging, not for this WebSocket benchmark.

    Or run it under Docker Compose with `--profile generic-assistant/server-perf`. See the [scaling-perf README](../../benchmarking_tools/scaling-perf/README.md#reproducing-the-recommended-scaling-setup) for the four-GPU Blackwell layout, its pinned NVFP4 TP2 LLM profile, and the compatible-profile change required on older hardware.

## Run

These commands target a local server with the built-in self-signed certificate. Omit `--insecure` for a server with a trusted certificate. To benchmark the OpenAI Realtime gateway instead of RTVI, follow [Realtime Endpoint](../../benchmarking_tools/scaling-perf/README.md#realtime-endpoint).

### Run Against the Server Performance Recipe

The `generic-assistant/server-perf` recipe disables the welcome message. Use `--skip-bot-intro` so the client does not wait for audio the server will not send.

1. Change to `benchmarking_tools/scaling-perf`.
2. Run a single-client smoke test.

    ```bash
    uv run python3 benchmark.py --skip-bot-intro --insecure
    ```

3. Run a scaling sweep.

    ```bash
    ./simulate_concurrency.sh --clients "1 2 4 8 16" --skip-bot-intro --insecure
    ```

### Run Against a Target That Sends a Welcome Message

Keep the default intro drain for these targets. Each client waits for and discards the greeting before sending test turns. The sweep wrapper sets a shared measurement window in advance, so a slow greeting can leave less time for measured turns.

1. Change to the benchmark directory.

    ```bash
    cd benchmarking_tools/scaling-perf
    ```

2. Run a single-client smoke test.

    ```bash
    uv run python3 benchmark.py --insecure
    ```

3. Choose one of these commands to run a load test:

    **One concurrency level:**

    ```bash
    ./simulate_concurrency.sh --clients 4 --insecure
    ```

    **Scaling sweep with cooldown between levels:**

    ```bash
    ./simulate_concurrency.sh --clients "1 2 4 8 16" --insecure
    ```

## Outputs

- Single-level run: `results_<timestamp>/` with `benchmark_summary.json`, `results.txt`, `results.tsv`, `results.json`, plus per-client logs and audio dumps.
- Sweep: `perf_suite_<timestamp>/` with suite-level `results.txt`, `results.tsv`, `results.json`, and one `run_<N>_clients/` directory per concurrency level.

See the perf README for the full output layout and per-flag reference:

- [`benchmarking_tools/scaling-perf/README.md`](../../benchmarking_tools/scaling-perf/README.md)
