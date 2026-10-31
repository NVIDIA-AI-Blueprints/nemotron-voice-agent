# Run Big Bench Audio on the Frontend Delegation Agent

## Overview

This runbook takes you through a Big Bench Audio (BBA) run on the Frontend Delegation Agent (FDA) over the OpenAI Realtime API. It is for operators who deploy FDA on a GPU host and score it with the tools in this directory.

The [Condition Card](#condition-card) lists every condition of the run. The BBA client changes only its transport for this run: its timing constants and stop rules are the same as for the `/api/ws` transport, so the scores stay comparable.

The run time grows with the number of samples and shrinks with `--batch_size`. Each sample streams its question in real time and then waits for the agent's reply. Time the warm-up sample and a short range of samples first to estimate a full run.

For shared details, this runbook links to the [Deploy the Frontend Delegation Agent](../../docs/how-to/deploy-frontend-delegation-agent.md) guide and the [BigBench Audio Eval README](README.md).

## Condition Card

The condition card states the exact configuration of the run that you are about to start. Fill one card for each run, and keep it with the run's results.

- The first rows name every model of the run and where it is served: the frontend and backend LLMs, the agent's ASR and TTS, and the scoring judge and transcription ASR.
- The **Required** column holds the BBA configuration. It is the same for every BBA run on FDA.
- For each run, fill a copy of the card as a two-column table, `| Dimension | This run |`, with one row for each dimension. Each **This run** value is what you read from the deployment, the client, and the commands of this run. [Fill the Condition Card](#fill-the-condition-card) lists where to read each value and gives the commands.
- End the filled copy with a **Differences from Required** row that lists each dimension whose **This run** value differs from **Required**, or `None`.
- Fill every row before you start the warm-up sample, except the Pipecat settings, the proactive status, and the `config hash`, which the agent logs after the first session. Fill those three rows after the warm-up sample, before the full run.
- Start the full run only when every **This run** value matches **Required**. A run that differs in any row is a different condition: name the differences in the card, and do not compare its score with runs that match the card.
- **No run starts without approval.** Before any run, including the warm-up sample, a smoke run, and the full run, show the filled card, with every difference from **Required**, to the owner of the run, and wait for their explicit approval. A request to start a run does not approve the card. An AI agent that runs this runbook must stop at this point and ask.

| Dimension | Required |
| --- | --- |
| Frontend (FE) LLM | Model `nvidia/nvidia/nemotron-3.5-lightning`. URL `https://inference-api.nvidia.com/v1` (Inference Hub). Reasoning off (`enable_thinking: false`), temperature 0.0, at most 256 tokens. With the `general_qa` prompts, it speaks the filler "One moment." and delegates the question. |
| Backend (BE) LLM | Model `nvidia/nvidia/nemotron-3-ultra`. URL `https://inference-api.nvidia.com/v1` (Inference Hub). Reasoning on (`enable_thinking: true`, `reasoning_budget` 1024, `reasoning_effort` `medium`). Prompts in `config/prompts/general_qa/backend/llm.yaml`. |
| Agent ASR | Model `nemotron-speech-streaming-en-0.6b`. URL `nemo-speech:50051` (gRPC, NeMo-Speech.cpp, published on the host as `localhost:50051`). |
| Agent TTS | Model `magpie-tts-multilingual`, voice `John`. URL `nemo-speech:50051` (gRPC, NeMo-Speech.cpp, published on the host as `localhost:50051`). |
| Judge LLM | Model `aws/anthropic/bedrock-claude-sonnet-5` (Claude Sonnet 5). URL `https://inference-api.nvidia.com/v1` (Inference Hub). Request format `--judge_api openai`, temperature 0.0. `eval.py` expects `CORRECT` or `INCORRECT`. Claude Sonnet 5.5 is not a valid judge here: it explains its verdict before it gives it, so the 8-token reply of `eval.py` ends before the label. |
| Transcription ASR | Model `nemotron-speech-streaming-en-0.6b`. URL `localhost:50051` (gRPC, the `nemo-speech` container of this deployment, the same server as the agent ASR). `nemo-speech` serves one ASR model and ignores the `--model` option of `transcribe.py`. |
| Run date | |
| Run operator | |
| Host and GPU | An x86_64 Linux host with one NVIDIA GPU |
| Repository commit | |
| Benchmark | Big Bench Audio, the `train` split of [`ArtificialAnalysis/big_bench_audio`](https://huggingface.co/datasets/ArtificialAnalysis/big_bench_audio) |
| Dataset directory | A new directory for each run |
| Samples | All samples, or the range that you name here |
| Compose files | `docker-compose.yml`, `docker/docker-compose.fda-eval.yaml`, `docker/docker-compose.fda-llm.yaml`, `docker/docker-compose.fda-bba.yaml`, in this order. No `docker/docker-compose.fda-silent-frontend.yaml`. |
| Recipe and service | `frontend-delegation-agent/single-gpu`, voice service `frontend-delegation-agent-single-gpu` |
| Endpoint | `ws://localhost:7860/v1/realtime?model=nvidia/nemotron-realtime-frontend-delegation`, without TLS, with the `REALTIME_API_KEY` Bearer key |
| `FDA_PROFILE` | `eval` |
| `PIPELINE_TLS` | `false` |
| `FDA_PROMPT_PROFILE` | `general_qa` |
| `FDA_BACKEND` | `llm` |
| `FDA_BACKEND_LINK` | `in_process` |
| `FDA_GATEWAY_CONFIG` | `gateway.eval.yaml` |
| `FDA_SILENT_FRONTEND` | `false` |
| `FDA_VAD_STOP_SECS` | `0.5` |
| `FDA_PROACTIVE_AFTER_S` | `1.0` |
| `FDA_PROACTIVE_MAX_PER_RUN` | `3` |
| `FDA_BACKEND_REASONING_BUDGET` | `1024` |
| `FDA_MAX_SESSIONS` | `4` |
| Pipecat settings | `settings_file` ending in `adapters/pipecat.eval.yaml`, `vad.stop_secs` 0.5, `vad.audio_idle_timeout` 0.0, `turn.user_speech_timeout` 0.0 |
| Proactive status | Enabled, after 1.0 s of silence, at most 3 per backend run (`general_qa` status lines) |
| `config hash` | |
| Client transport and model | `--transport realtime`, the default `--model` |
| Client constants | `SAMPLE_RATE` 16000, `CHUNK_MS` 32, `POST_SEND_INITIAL_WAIT_SEC` 5.0, `POST_SEND_GAP_TOLERANCE_SEC` 2.0, `RECV_TIMEOUT_SEC` 2.0, `OPEN_TIMEOUT_SEC` 5.0, `REALTIME_HANDSHAKE_TIMEOUT_SEC` 30.0. Never change them for a run. |
| `--batch_size` | `4`, one client session for each of FDA's 4 Realtime sessions |
| Differences from Required | None |

BBA uses the proactive status settings because of the client's stop rule: the client stops listening 2 seconds after the last audio chunk. The filler ends about 1 second after the backend starts, so without a status line the client closes before an answer that takes longer than about 3 seconds. With `FDA_PROACTIVE_AFTER_S` 1.0, a status line, such as "Still checking, one moment.", reaches the client about 1.05 to 1.3 seconds after the filler and restarts the client's 2-second wait. Up to 3 lines keep the client listening for up to about 10 seconds. The client and the scoring do not change. Every other deployment, such as a Tau run, keeps the defaults of `docker-compose.yml`: 15 seconds and 1 line for each backend run.

A run can use the Super VL overlay, `docker/docker-compose.fda-llm-super-vl.yaml`, as a named variant. Pass it after the LLM backend overlay and before the BBA overlay. The variant differs from **Required** in the **Backend (BE) LLM** row: model `nvidia/nvidia/nemotron-3.5-super-vl`, reasoning on with `enable_thinking: true` and `reasoning_effort` `medium`, and no `reasoning_budget`. It also differs in the **Compose files** row and in the `FDA_BACKEND_REASONING_BUDGET` row, which prints `0`. Record these differences in **Differences from Required**, and do not compare the variant's scores with runs of the default configuration.

Every other eval run, such as Tau, keeps `vad.stop_secs` 0.8. The Hermes backend is not part of this card, because its per-session worker start can exceed the client's 30-second handshake limit. The client constants and the scoring scripts are the same for both transports, so two BBA runs are comparable when their cards match and they use the same transcription and judge models.

## Before You Begin

The run needs the prerequisites in the following table.

| Prerequisite | Details |
| --- | --- |
| GPU host | An x86_64 Linux host with one NVIDIA GPU for the `nemo-speech` container, Docker Engine 28 or newer, Docker Compose 2.24.4 or newer, and the NVIDIA Container Toolkit. The full list is in [Prerequisites](../../docs/how-to/deploy-frontend-delegation-agent.md#prerequisites). The Hermes and OpenShell prerequisites do not apply. |
| Speech model weights | The NeMo-Speech.cpp weights, downloaded with `bash scripts/download-nemo-speech-models.sh`, as step 1 of [Start the Deployment](../../docs/how-to/deploy-frontend-delegation-agent.md#start-the-deployment) describes. |
| Benchmark dependencies | Run `uv sync --group benchmark` from the repository root. |
| `ffmpeg` | `speech-inference.py --preprocess` uses it to convert MP3 to 16 kHz mono WAV. |
| Hugging Face access | `download_dataset.py` reads `HF_TOKEN` from the shell or takes `--token`. It does not read `.env`. You can also log in with `huggingface-cli login`. |
| Judge endpoint | `eval.py` needs `EVAL_API_URL` and `EVAL_API_KEY`. For the required judge, set `EVAL_API_URL` to `https://inference-api.nvidia.com/v1` and `EVAL_API_KEY` to an Inference Hub key, such as your `NVIDIA_API_KEY`. Refer to [Score the Results](#score-the-results). |
| ASR endpoint | `transcribe.py` needs a Riva-compatible gRPC ASR endpoint. The `nemo-speech` container of the FDA deployment serves it on `localhost:50051`, so keep the deployment running until transcription is done. Refer to [Score the Results](#score-the-results). |

Run the Compose commands from the repository root. Run the benchmark scripts from `benchmarking_tools/AA-BigBenchAudio-Eval/`.

## Check the Credentials

Confirm that `.env` holds the three keys that the run needs. The following command, run from the repository root, prints only a count for each key and never prints a value:

```bash
for key in NVIDIA_API_KEY HF_TOKEN REALTIME_API_KEY; do
  printf '%s: %s\n' "$key" "$(grep -c "^$key=." .env)"
done
```

Each key must show `1`. A `0` means that the key is missing or empty. Add it to `.env`, as [Configure the Environment](../../docs/how-to/deploy-frontend-delegation-agent.md#configure-the-environment) describes. Do not overwrite the other keys.

The client reads `REALTIME_API_KEY` from the shell first and falls back to `.env` only when the shell has no value. The following command reports whether the current shell sets the key, without printing it:

```bash
[ -n "${REALTIME_API_KEY:-}" ] && echo "REALTIME_API_KEY is set in this shell"
```

If the command prints the message, make sure the shell value matches `.env`, or run `unset REALTIME_API_KEY` so the client uses `.env`.

## Deploy the Agent

Deploy FDA with four Compose files in the following order: `docker-compose.yml`, the eval overlay, the LLM backend overlay, and the BBA overlay. Set the files in a variable in every shell that you use for this deployment:

```bash
FDA_FILES="-f docker-compose.yml -f docker/docker-compose.fda-eval.yaml -f docker/docker-compose.fda-llm.yaml -f docker/docker-compose.fda-bba.yaml"
```

Complete the following steps from the repository root:

1. Create the host log directory, so that it belongs to your user:

    ```bash
    mkdir -p logs
    ```

2. Check that Compose accepts the files and your `.env`. The command prints nothing when the configuration is valid:

    ```bash
    docker compose $FDA_FILES --profile frontend-delegation-agent/single-gpu config -q
    ```

3. Start the recipe with the `general_qa` prompt profile:

    ```bash
    FDA_PROMPT_PROFILE=general_qa docker compose $FDA_FILES \
      --profile frontend-delegation-agent/single-gpu up -d --build
    ```

Pass the same `$FDA_FILES` in every later Compose command of this deployment, including `exec`, `logs`, and `down`. Every command that recreates the containers, such as `up`, also needs `FDA_PROMPT_PROFILE=general_qa`. Without it, eval mode falls back to the `customer_service` profile. Never add `docker/docker-compose.fda-silent-frontend.yaml` to a BBA deployment.

If `up` fails with an unknown `!override` or `!reset` tag, update Docker Compose to 2.24.4 or newer. For other startup failures, refer to [Troubleshooting](../../docs/how-to/deploy-frontend-delegation-agent.md#troubleshooting) in the deploy guide.

## Verify the Deployment

Run the following checks from the repository root before you stream any sample:

1. Download the NLTK `punkt_tab` data in the voice container. Pipecat loads it for sentence splitting, and the first sessions otherwise wait for the download. The command prints `True`:

    ```bash
    docker compose $FDA_FILES --profile frontend-delegation-agent/single-gpu \
      exec frontend-delegation-agent-single-gpu \
      .venv/bin/python -c "import nltk; print(nltk.download('punkt_tab', quiet=True))"
    ```

2. Check the liveness route over plain HTTP. The route returns `{"status":"ok"}`. It confirms that the server process is up, but it does not check speech, the LLMs, or the backend:

    ```bash
    curl -f http://localhost:7860/health
    ```

    If you set `PIPELINE_APP_PORT` in `.env`, use that port here and in every later URL.

3. Check the deployment settings in the voice container:

    ```bash
    docker compose $FDA_FILES --profile frontend-delegation-agent/single-gpu \
      exec frontend-delegation-agent-single-gpu \
      printenv FDA_PROFILE PIPELINE_TLS FDA_PROMPT_PROFILE FDA_VAD_STOP_SECS FDA_PROACTIVE_AFTER_S FDA_PROACTIVE_MAX_PER_RUN FDA_BACKEND
    ```

    The command prints the following values, one for each line:

    ```text
    eval
    false
    general_qa
    0.5
    1.0
    3
    llm
    ```

4. Check that the frontend is not silent. The command prints `false`:

    ```bash
    docker compose $FDA_FILES --profile frontend-delegation-agent/single-gpu \
      exec frontend-delegation-agent-single-gpu printenv FDA_SILENT_FRONTEND
    ```

If a value differs, refer to the following list:

- `FDA_PROMPT_PROFILE` prints `customer_service`: the `up` command ran without `FDA_PROMPT_PROFILE=general_qa`. Rerun step 3 of [Deploy the Agent](#deploy-the-agent).
- `FDA_VAD_STOP_SECS` prints `0.8`: the BBA overlay is missing. Rerun the `up` command with all four files.
- `FDA_PROACTIVE_AFTER_S` prints `15` or `FDA_PROACTIVE_MAX_PER_RUN` prints `1`: the BBA overlay is missing. Rerun the `up` command with all four files.
- `FDA_BACKEND` prints nothing: the LLM backend overlay is missing. Rerun the `up` command with all four files.
- `FDA_SILENT_FRONTEND` prints `true`: the silent-frontend overlay is in the command. Rerun the `up` command with the four files only.

The agent logs its effective Pipecat settings and proactive status settings after the first Realtime session. You check them after the warm-up sample in [Run a Warm-Up Sample](#run-a-warm-up-sample).

## Prepare the Dataset

Download and preprocess the dataset from `benchmarking_tools/AA-BigBenchAudio-Eval/`. Use a new dataset directory for each run, so that files from an earlier run do not mix with this run.

1. Download the dataset. Each sample gets its own folder with `input.mp3` and `meta.json`:

    ```bash
    uv run python download_dataset.py --input_dir ./datasets/bigbench_audio --split train
    ```

    If the download fails with an authentication or rate-limit error, export `HF_TOKEN` in the shell, pass `--token`, or run `huggingface-cli login`, and then rerun the command.

2. Convert each `input.mp3` to a 16 kHz mono `input.wav`:

    ```bash
    uv run python speech-inference.py --input_dir ./datasets/bigbench_audio --preprocess
    ```

    The script prints a warning for each sample that it cannot convert. If every sample fails, check that `ffmpeg` is installed and on the `PATH`.

## Fill the Condition Card

Fill a copy of the [Condition Card](#condition-card) as a two-column table, `| Dimension | This run |`, after you deploy the agent and prepare the dataset, and before you stream any sample. End it with the **Differences from Required** row.

The following table lists where to read the **This run** value of each dimension:

| Dimension | Where to read it |
| --- | --- |
| Frontend (FE) LLM | `printenv FRONTEND_LLM_MODEL FRONTEND_LLM_BASE_URL` in the voice container. Reasoning: `frontend.llm` in `src/examples/frontend_delegation_agent/config/delegation_agent.yaml`. |
| Backend (BE) LLM | `printenv BACKEND_LLM_MODEL BACKEND_LLM_BASE_URL` in the voice container. Reasoning: `backend.request_overrides` and `llm.reasoning_effort` in `src/examples/frontend_delegation_agent/config/gateway.yaml`. The `reasoning_budget` comes from the `FDA_BACKEND_REASONING_BUDGET` row. |
| Agent ASR | The `singlegpu.asr` entry of `src/examples/frontend_delegation_agent/services.local.yaml`. `printenv FBA_ASR_SERVER` in the voice container must print an empty line; a value replaces the URL. |
| Agent TTS | The `singlegpu.tts` entry of `src/examples/frontend_delegation_agent/services.local.yaml`. `printenv FBA_TTS_SERVER` in the voice container must print an empty line; a value replaces the URL. |
| Judge LLM | The `--judge_api` and `--judge_model` options of `eval.py` and `EVAL_API_URL`. Never record `EVAL_API_KEY`. |
| Transcription ASR | The `--host` and `--port` options of `transcribe.py`. The model is the one that `nemo-speech` serves (the **Agent ASR** row), whatever `--model` says. |
| Run date | The date when the run starts |
| Run operator | The person or agent who starts the run |
| Host and GPU | `nvidia-smi --query-gpu=name,driver_version --format=csv,noheader` |
| Repository commit | `git rev-parse --short HEAD` and `git status --short`. The voice service runs the mounted `src/`, so list any uncommitted change. |
| Benchmark | The `download_dataset.py` command |
| Dataset directory | The `--input_dir` option |
| Samples | The `--start`, `--end`, or `--retry_samples` options |
| Compose files | `echo "$FDA_FILES"` |
| Recipe and service | The `up` command |
| Endpoint | The `--server-url` option of the inference command |
| `FDA_PROFILE` | `printenv` in the voice container |
| `PIPELINE_TLS` | `printenv` in the voice container |
| `FDA_PROMPT_PROFILE` | `printenv` in the voice container |
| `FDA_BACKEND` | `printenv` in the voice container |
| `FDA_BACKEND_LINK` | `printenv` in the voice container |
| `FDA_GATEWAY_CONFIG` | `printenv` in the voice container |
| `FDA_SILENT_FRONTEND` | `printenv` in the voice container |
| `FDA_VAD_STOP_SECS` | `printenv` in the voice container |
| `FDA_PROACTIVE_AFTER_S` | `printenv` in the voice container |
| `FDA_PROACTIVE_MAX_PER_RUN` | `printenv` in the voice container |
| `FDA_BACKEND_REASONING_BUDGET` | `printenv` in the voice container. It sets `reasoning_budget` of the backend requests, and `0` omits the parameter. |
| `FDA_MAX_SESSIONS` | `printenv` in the voice container |
| Pipecat settings | The `Pipecat adapter settings` line of the app log, after the warm-up sample |
| Proactive status | The `proactive status` line of the app log, after the warm-up sample |
| `config hash` | The ready line of the app log, after the warm-up sample. The Pipecat settings are not part of the hash. |
| Client transport and model | The inference command |
| Client constants | The constants of `speech-inference.py` |
| `--batch_size` | The inference command |
| Differences from Required | Each dimension whose **This run** value differs from **Required** |

Run the following commands from the repository root. They print no secret values.

1. Read the host, the GPU, and the repository commit:

    ```bash
    date -u +%Y-%m-%dT%H:%MZ
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    git rev-parse --short HEAD
    git status --short
    ```

    If `git status --short` prints any file, list the changed files in the **Repository commit** row.

2. Read the Compose files of the deployment:

    ```bash
    echo "$FDA_FILES"
    ```

3. Read the agent settings in the voice container:

    ```bash
    docker compose $FDA_FILES --profile frontend-delegation-agent/single-gpu \
      exec frontend-delegation-agent-single-gpu sh -c 'for v in FRONTEND_LLM_MODEL FRONTEND_LLM_BASE_URL BACKEND_LLM_MODEL BACKEND_LLM_BASE_URL FBA_ASR_SERVER FBA_TTS_SERVER FDA_PROFILE PIPELINE_TLS FDA_PROMPT_PROFILE FDA_BACKEND FDA_BACKEND_LINK FDA_GATEWAY_CONFIG FDA_SILENT_FRONTEND FDA_VAD_STOP_SECS FDA_PROACTIVE_AFTER_S FDA_PROACTIVE_MAX_PER_RUN FDA_BACKEND_REASONING_BUDGET FDA_MAX_SESSIONS; do printf "%s=%s\n" "$v" "$(printenv "$v")"; done'
    ```

    The command prints one `NAME=value` line for each setting. Copy each value to its row. `FBA_ASR_SERVER` and `FBA_TTS_SERVER` print an empty value when the agent's ASR and TTS use the catalog URL.

4. Read the agent's ASR and TTS models and URLs, and the reasoning settings of both LLMs:

    ```bash
    sed -n '/^singlegpu:/,$p' src/examples/frontend_delegation_agent/services.local.yaml | grep -E '^  (tts|asr):| server:| model:| voice_id:'
    grep -E 'enable_thinking|reasoning_budget|reasoning_effort' \
      src/examples/frontend_delegation_agent/config/delegation_agent.yaml \
      src/examples/frontend_delegation_agent/config/gateway.yaml
    ```

5. Read the client constants:

    ```bash
    grep -E '^(SAMPLE_RATE, CHUNK_MS|POST_SEND_INITIAL_WAIT_SEC|POST_SEND_GAP_TOLERANCE_SEC|RECV_TIMEOUT_SEC|OPEN_TIMEOUT_SEC|DEFAULT_REALTIME_MODEL|REALTIME_HANDSHAKE_TIMEOUT_SEC) =' \
      benchmarking_tools/AA-BigBenchAudio-Eval/speech-inference.py
    ```

6. Fill the rows for the judge LLM, the transcription ASR, the dataset directory, the samples, and `--batch_size` from the commands that you plan to run.

Compare each **This run** value with **Required**. If a value differs, fix the deployment or the command before you continue, or name the difference in the **Differences from Required** row.

Show the filled card to the owner of the run, with the Pipecat settings, the proactive status, and the `config hash` marked as pending, and wait for their explicit approval before you start the warm-up sample.

## Run a Warm-Up Sample

The first Realtime session after a container start also runs FDA's one-time warm-up, and the warm-up counts against the client's 30-second handshake limit. Run one sample first, so that the warm-up does not fail a sample of the real run.

1. From `benchmarking_tools/AA-BigBenchAudio-Eval/`, start the run record. The command fixes the run ID, which names the run's folder in [Dump the Run](#dump-the-run). Replace `<label>` with a short name for the run, such as `smoke10` or `full`:

    ```bash
    [ ! -e datasets/logs/run.env ] || echo "datasets/logs holds an earlier run: dump it before you continue"
    mkdir -p datasets/logs
    RUN_START=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    printf 'RUN_START=%s\nRUN_ID=%s_fda-llm-bba-%s\n' "$RUN_START" \
      "$(date -u -d "$RUN_START" +%Y-%m-%d_%H-%M-%SZ)" "<label>" > datasets/logs/run.env
    cat datasets/logs/run.env
    ```

    If the first line prints a message, `datasets/logs/` and the sample folders still hold an earlier run. Complete [Dump the Run](#dump-the-run) for that run, remove `datasets/logs/` and `datasets/bigbench_audio/`, prepare the dataset again, as [Prepare the Dataset](#prepare-the-dataset) describes, and then rerun this step. Each run starts with clean sample folders, so the dump holds only the outputs of its own run.

    Every later client and scoring command of this runbook appends its output to a log in `datasets/logs/`.

2. Run sample `0`:

    ```bash
    uv run python speech-inference.py --input_dir ./datasets/bigbench_audio --inference \
      --transport realtime --server-url http://127.0.0.1:7860 --start 0 --end 0 2>&1 | tee -a datasets/logs/inference.log
    ```

    The client prints `successfully processed idx: 0` when the sample succeeds. If it prints `unsuccessful idx: 0` with a reason, rerun the same command. A repeated failure needs the fix in [Troubleshooting](#troubleshooting).

3. Play `datasets/bigbench_audio/0/output.wav` in an audio player. The recording holds the frontend's filler, "One moment.", and then the backend's answer. When the answer takes longer than about 1 second, one to three status lines, such as "Still checking, one moment.", come between the filler and the answer. A recording with only the filler is a valid result under the client's stop rules. Refer to [Troubleshooting](#troubleshooting).

4. From the repository root, read the ready line, the Pipecat settings line, and the proactive status line from the app log:

    ```bash
    docker compose $FDA_FILES --profile frontend-delegation-agent/single-gpu \
      logs frontend-delegation-agent-single-gpu | grep -E 'frontend-delegation-agent ready|Pipecat adapter settings|proactive status'
    ```

    The ready line has the form `frontend-delegation-agent ready (adapter=pipecat, profile=eval)`, followed by the loaded configuration files and the `config hash`. The `Pipecat adapter settings` line must show the following values:

    - `settings_file` ends in `adapters/pipecat.eval.yaml`.
    - `vad` has `stop_secs` 0.5 and `audio_idle_timeout` 0.0.
    - `turn` has `user_speech_timeout` 0.0.

    This is the BBA exception to the eval settings check of the FDA deploy skill, where every other eval run must show `stop_secs` 0.8. If a BBA deployment shows 0.8, the BBA overlay is missing. Stop, and redeploy with all four files.

    The proactive status line follows the `Pipecat adapter settings` line and must read as follows:

    ```text
    proactive status enabled=True after_s=1 max_per_run=3
    ```

    If it shows `after_s=15` or `max_per_run=1`, the BBA overlay is missing. Stop, and redeploy with all four files.

5. Copy the `Pipecat adapter settings` line, the proactive status line, and the `config hash` to the [Condition Card](#condition-card). Show the completed card to the owner of the run again, and start the full run only after their explicit approval.

## Run the Benchmark

Run the full dataset from `benchmarking_tools/AA-BigBenchAudio-Eval/` with `--batch_size 4`. FDA serves at most 4 Realtime sessions and frees a session when its client disconnects, and the client waits 1 second after each sample before it starts the next one. The client's own default is 1; always pass `--batch_size 4` for a BBA run on FDA, and never more than 4.

1. Start the run. The first command marks the start of the full run in `datasets/logs/inference.log`, so that the monitor in step 2 does not count the warm-up sample. `PYTHONUNBUFFERED=1` makes the client write each result line to the log at once:

    ```bash
    echo "=== full run $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> datasets/logs/inference.log
    PYTHONUNBUFFERED=1 uv run python speech-inference.py --input_dir ./datasets/bigbench_audio --inference \
      --transport realtime --server-url http://127.0.0.1:7860 --batch_size 4 2>&1 | tee -a datasets/logs/inference.log
    ```

    The run includes sample `0` again and replaces its warm-up recording.

2. Monitor the run from a second shell in `benchmarking_tools/AA-BigBenchAudio-Eval/`. Every 30 minutes, the monitor transcribes and judges the samples that finished since its last check, with the commands of [Score the Results](#score-the-results), and prints one progress row. Replace `1000` with the number of samples of the run. The second argument is the interval in seconds; use a shorter interval, such as `120`, for a smoke run:

    ```bash
    EVAL_API_URL=https://inference-api.nvidia.com/v1 EVAL_API_KEY="$NVIDIA_API_KEY" python3 - 1000 1800 <<'EOF'
    import re, subprocess, sys, time
    from datetime import datetime
    from pathlib import Path

    total, interval = int(sys.argv[1]), int(sys.argv[2])
    root, log = Path("datasets/bigbench_audio"), Path("datasets/logs/inference.log")
    # The general_qa filler and proactive status lines: a response that holds only these has no answer.
    no_answer = re.compile(
        r"one moment( please)?|still checking|still on it|thanks for waiting|almost there|still working on that for you"
    )
    scored = set()

    def append(name, *cmd):
        with open(f"datasets/logs/{name}", "a", encoding="utf-8") as out:
            subprocess.run(["uv", "run", "python", *cmd], stdout=out, stderr=subprocess.STDOUT, check=False)

    def read(path):
        return path.read_text(encoding="utf-8").strip() if path.exists() else ""

    print("| Done | Failed | Scored | Correct | Accuracy | No answer | Time left |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    while True:
        marker, _, run = log.read_text(encoding="utf-8").rpartition("=== full run ")
        started = datetime.fromisoformat(run.split()[0].replace("Z", "+00:00")).timestamp()
        done = set(re.findall(r"successfully processed idx: (\d+)", run))
        failed = set(re.findall(r"unsuccessful idx: (\d+)", run)) - done
        new = ",".join(sorted(done - scored, key=int))
        if new:
            append("transcribe.log", "transcribe.py", "--input_dir", str(root), "--host", "localhost", "--port", "50051", "--retry_samples", new)
            append("eval.log", "eval.py", "--input_dir", str(root), "--retry_samples", new,
                   "--judge_api", "openai", "--judge_model", "aws/anthropic/bedrock-claude-sonnet-5")
            scored |= set(new.split(","))
        labels = [read(root / i / "result.txt") for i in scored]
        graded = [label for label in labels if label in ("CORRECT", "INCORRECT")]
        correct = graded.count("CORRECT")
        empty = sum(1 for i in scored if not no_answer.sub(" ", re.sub(r"[^a-z ]", " ", read(root / i / "response.txt").lower())).split())
        finished = len(done) + len(failed)
        rate = finished / max(time.time() - started, 1.0)
        left = (total - finished) / rate if rate else 0
        accuracy = f"{100 * correct / len(graded):.1f}%" if graded else "n/a"
        eta = "done" if finished >= total else f"~{int(left // 3600)} h {int(left % 3600 // 60)} m"
        print(f"| {len(done)} / {total} | {len(failed)} | {len(graded)} | {correct} | {accuracy} | {empty} | {eta} |", flush=True)
        if finished >= total:
            break
        time.sleep(interval)
    EOF
    ```

    A row has the following form. The values are an example:

    ```text
    | Done | Failed | Scored | Correct | Accuracy | No answer | Time left |
    | --- | --- | --- | --- | --- | --- | --- |
    | 412 / 1000 | 3 | 405 | 318 | 78.5% | 6 | ~1 h 40 m |
    ```

    - **Done** counts the samples that the client finished, and **Failed** counts the samples with an `unsuccessful idx` line.
    - **Scored** counts the samples that have a `CORRECT` or `INCORRECT` label, and **Accuracy** is **Correct** divided by **Scored**.
    - **No answer** counts the responses that hold only the filler and status lines.
    - **Time left** extrapolates from the samples finished since the start of the full run.

    The monitor shares the `nemo-speech` container with the agent while it transcribes, and it calls the judge as samples finish. It stops after the last sample. The labels that it writes are the final labels of those samples: [Score the Results](#score-the-results) transcribes only the samples without a transcript, and its `eval.py` step judges every sample again with the same judge.

3. List the failed samples:


    ```bash
    grep 'unsuccessful idx' datasets/logs/inference.log
    ```

    Each line has the form `unsuccessful idx: <id> — <reason>`. The client does not retry a sample automatically.

4. Rerun the failed samples, including `server_busy` failures, with a comma-separated list of their IDs:

    ```bash
    uv run python speech-inference.py --input_dir ./datasets/bigbench_audio --inference \
      --transport realtime --server-url http://127.0.0.1:7860 --retry_samples <id>,<id> 2>&1 | tee -a datasets/logs/inference.log
    ```

Finish every rerun before you start [Score the Results](#score-the-results). `transcribe.py` skips a sample that already has a `response.txt`, so a sample that you rerun after transcription keeps its old transcript until you delete its `response.txt`.

## Score the Results

Score the recordings from `benchmarking_tools/AA-BigBenchAudio-Eval/`. The scoring scripts are the same for both transports.

1. Transcribe `input.wav` to `question.txt` and `output.wav` to `response.txt` with the `nemo-speech` container of the FDA deployment. Keep the deployment running for this step:

    ```bash
    uv run python transcribe.py --input_dir ./datasets/bigbench_audio --host localhost --port 50051 2>&1 | tee -a datasets/logs/transcribe.log
    ```

    The script prints `successfully processed idx: <id>` for each sample. `transcribe.py` works against `nemo-speech`: `nemo-speech` serves one ASR model, `nemotron-speech-streaming-en-0.6b`, and ignores `--model`, so the default `parakeet-1.1b-en-US-asr-offline` and `nemotron-speech-streaming-en-0.6b` give the same transcripts. Record `nemotron-speech-streaming-en-0.6b` in the **Transcription ASR** row of the [Condition Card](#condition-card). A sample without `output.wav` gets no `response.txt`, and `eval.py` then prints `evaluation failed for idx=<id>: response.txt missing`. Rerun that sample, as [Run the Benchmark](#run-the-benchmark) describes, and transcribe it again. For a separate ASR service, refer to the [BigBench Audio Eval README](README.md#experiment-1-speech-pipeline).

2. Score each response with the LLM judge. The following command uses the judge that the condition card requires, Claude Sonnet 5 on the Inference Hub, with your Inference Hub key:

    ```bash
    EVAL_API_URL=https://inference-api.nvidia.com/v1 EVAL_API_KEY="$NVIDIA_API_KEY" \
      uv run python eval.py --input_dir ./datasets/bigbench_audio \
      --judge_api openai --judge_model aws/anthropic/bedrock-claude-sonnet-5 2>&1 | tee -a datasets/logs/eval.log
    ```

    The script writes a `result.txt` for each sample.

3. Find results whose label is neither `CORRECT` nor `INCORRECT`:

    ```bash
    uv run python find_invalid_results.py --input_dir ./datasets/bigbench_audio 2>&1 | tee -a datasets/logs/find-invalid.log
    ```

    If the script lists invalid results, it prints an `eval.py` command with `--retry_samples`. The printed command does not include the judge options. Add `--judge_api openai --judge_model aws/anthropic/bedrock-claude-sonnet-5`, and run it with the same `EVAL_API_URL` and `EVAL_API_KEY` as step 2.

4. Compute the accuracy:

    ```bash
    uv run python analyze_results.py --input_dir ./datasets/bigbench_audio 2>&1 | tee datasets/logs/analyze.log
    ```

    The script prints the counts of `CORRECT`, `INCORRECT`, and other labels, and the accuracy.

## Record the Run

Keep the filled [Condition Card](#condition-card) with the results of each run, and add the results in the following table. [Dump the Run](#dump-the-run) writes the same values to the run's `README.md`.

| Item | Value | Source |
| --- | --- | --- |
| Sample count | | `analyze_results.py` |
| Failed samples | | The `unsuccessful idx` lines that remain after the reruns |
| Filler-only responses | | The samples whose `response.txt` holds only the filler, "One moment.", and status lines such as "Still checking, one moment." |
| Accuracy | | `analyze_results.py` |

## Dump the Run

Dump every run, including a smoke run, to the evaluation dump repository before you stop the deployment. The container logs are lost when the containers are removed. Each run gets its own folder, `big-bench-audio/<RUN_ID>/`, where `RUN_ID` is the UTC start time of the run and the run label from [Run a Warm-Up Sample](#run-a-warm-up-sample), for example `2026-10-31_09-15-02Z_fda-llm-bba-smoke10`.

The dump has the following layout:

| Path | Content |
| --- | --- |
| `README.md` | The run summary: run ID and window, the approved condition card, the results, and one row for each sample |
| `samples/<id>/` | `meta.json`, `output.wav`, `question.txt`, `response.txt`, and `result.txt` of each sample that has an `output.wav`. The input audio is not copied, because the public dataset holds it. |
| `logs/` | The client and scoring logs, the voice container and `nemo-speech` logs of the run window, the FDA voice and gateway events of the run window, and the backend traces of the run's sessions |
| `provenance/` | The approved condition card, the repository commit and uncommitted changes, the Compose files, the agent settings, the container images, the client constants, and this runbook |

Run the following steps from the repository root, in the shell that holds `FDA_FILES`:

1. Set the run variables. Replace `<path-to-voice-agent-evaluation-dump>` with the path of your clone of the dump repository:

    ```bash
    BBA=benchmarking_tools/AA-BigBenchAudio-Eval
    . "$BBA/datasets/logs/run.env"
    RUN_END=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    DUMP_REPO=<path-to-voice-agent-evaluation-dump>
    DUMP="$DUMP_REPO/big-bench-audio/$RUN_ID"
    mkdir -p "$DUMP/samples" "$DUMP/logs" "$DUMP/provenance"
    ```

    If `$DUMP` already exists from an earlier dump of the same run, the next steps replace its files.

2. Save the approved condition card, with every **This run** value and the **Differences from Required** row, as `$DUMP/provenance/condition-card.md`.

3. Copy the samples:

    ```bash
    for dir in "$BBA"/datasets/bigbench_audio/*/; do
      [ -f "$dir/output.wav" ] || continue
      id=$(basename "$dir")
      mkdir -p "$DUMP/samples/$id"
      for f in meta.json output.wav question.txt response.txt result.txt; do
        [ -f "$dir/$f" ] && cp -p "$dir/$f" "$DUMP/samples/$id/"
      done
    done
    ```

4. Copy the logs. The FDA event files and the container logs hold every session since the deployment started, so the commands keep only the run window:

    ```bash
    cp -p "$BBA"/datasets/logs/*.log "$DUMP/logs/"
    for svc in frontend-delegation-agent-single-gpu nemo-speech; do
      docker compose $FDA_FILES --profile frontend-delegation-agent/single-gpu logs --no-color --timestamps \
        --since "$RUN_START" --until "$RUN_END" "$svc" > "$DUMP/logs/$svc.log" 2>&1
    done
    python3 - "$RUN_START" "$RUN_END" "$DUMP/logs" <<'EOF'
    import json, sys
    from datetime import datetime

    start, end = (datetime.fromisoformat(a.replace("Z", "+00:00")).timestamp() for a in sys.argv[1:3])
    for name, key in (("fda_voice_events.pipecat.jsonl", "timestamp"), ("fda_gateway_events.pipecat.jsonl", "ts")):
        with open(f"logs/{name}", encoding="utf-8") as src, open(f"{sys.argv[3]}/{name}", "w", encoding="utf-8") as dst:
            for line in src:
                try:
                    t = float(json.loads(line).get(key) or 0)
                except (ValueError, AttributeError):
                    continue
                if start <= t <= end:
                    dst.write(line)
    EOF
    find logs/fda_workers_pipecat -name '*.trace.jsonl' -newermt "$RUN_START" -print0 \
      | tar -czf "$DUMP/logs/fda_backend_traces.tar.gz" --null -T -
    ```

5. Record the provenance:

    ```bash
    P="$DUMP/provenance"
    git rev-parse HEAD > "$P/git-head.txt"
    git status --short > "$P/git-status.txt"
    git diff HEAD > "$P/git-diff.patch"
    git ls-files --others --exclude-standard -z | tar -czf "$P/git-untracked.tar.gz" --null -T -
    echo "$FDA_FILES" > "$P/compose-files.txt"
    for f in $FDA_FILES; do [ "$f" = -f ] || cp -p "$f" "$P/"; done
    docker compose $FDA_FILES --profile frontend-delegation-agent/single-gpu images > "$P/images.txt"
    docker compose $FDA_FILES --profile frontend-delegation-agent/single-gpu \
      exec -T frontend-delegation-agent-single-gpu sh -c 'for v in FRONTEND_LLM_MODEL FRONTEND_LLM_BASE_URL BACKEND_LLM_MODEL BACKEND_LLM_BASE_URL FBA_ASR_SERVER FBA_TTS_SERVER FDA_PROFILE PIPELINE_TLS FDA_PROMPT_PROFILE FDA_BACKEND FDA_BACKEND_LINK FDA_GATEWAY_CONFIG FDA_SILENT_FRONTEND FDA_VAD_STOP_SECS FDA_PROACTIVE_AFTER_S FDA_PROACTIVE_MAX_PER_RUN FDA_BACKEND_REASONING_BUDGET FDA_MAX_SESSIONS; do printf "%s=%s\n" "$v" "$(printenv "$v")"; done' > "$P/fda-env.txt"
    grep -h -E 'frontend-delegation-agent ready|Pipecat adapter settings|proactive status' "$DUMP/logs/frontend-delegation-agent-single-gpu.log" > "$P/fda-startup.txt"
    grep -E '^(SAMPLE_RATE, CHUNK_MS|POST_SEND_INITIAL_WAIT_SEC|POST_SEND_GAP_TOLERANCE_SEC|RECV_TIMEOUT_SEC|OPEN_TIMEOUT_SEC|DEFAULT_REALTIME_MODEL|REALTIME_HANDSHAKE_TIMEOUT_SEC) =' \
      "$BBA/speech-inference.py" > "$P/client-constants.txt"
    cp -p "$BBA/RUNBOOK-FDA.md" "$P/"
    ```

    The `printenv` list names only settings. Never copy `.env` or a `docker compose config` output into the dump, because both hold the keys. If the ready line, the settings line, and the proactive status line are missing from `fda-startup.txt`, the voice container started before `RUN_START`. Copy the three lines from the full app log instead.

6. Write the run summary:

    ```bash
    python3 - "$DUMP" "$RUN_START" "$RUN_END" <<'EOF'
    import json, re, sys
    from pathlib import Path

    dump, start, end = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    # The general_qa filler and proactive status lines (config/prompts/general_qa/frontend.yaml): a response
    # that holds only these has no answer.
    NO_ANSWER = re.compile(
        r"one moment( please)?|still checking|still on it|thanks for waiting|almost there|still working on that for you"
    )

    def no_answer(text):
        words = re.sub(r"[^a-z ]", " ", text.lower())
        return not NO_ANSWER.sub(" ", words).split()

    def read(path):
        return path.read_text(encoding="utf-8").strip() if path.exists() else ""

    rows, counts, filler_only = [], {}, []
    for d in sorted((p for p in (dump / "samples").iterdir() if p.is_dir()), key=lambda p: int(p.name)):
        meta = json.loads(read(d / "meta.json") or "{}")
        result, response = read(d / "result.txt") or "not scored", read(d / "response.txt")
        counts[result] = counts.get(result, 0) + 1
        if no_answer(response):
            filler_only.append(d.name)
        cell = response.replace("|", "\\|").replace("\n", " ")
        rows.append(f"| {d.name} | {meta.get('category', '')} | {meta.get('official_answer', '')} | {result} | {cell[:200]} |")

    scored = counts.get("CORRECT", 0) + counts.get("INCORRECT", 0)
    accuracy = f"{100.0 * counts.get('CORRECT', 0) / scored:.2f}%" if scored else "n/a"
    lines = [
        f"# {dump.name}: Big Bench Audio on the Frontend Delegation Agent",
        "",
        f"- **Run window (UTC):** {start} .. {end}",
        f"- **Samples dumped:** {len(rows)}",
        f"- **Accuracy (CORRECT / scored):** {accuracy}",
        f"- **Labels:** " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())),
        f"- **No answer (filler or status lines only):** {len(filler_only)}" + (f" ({', '.join(filler_only)})" if filler_only else ""),
        "",
        "## Condition Card",
        "",
        read(dump / "provenance" / "condition-card.md") or "Missing: save the approved card as provenance/condition-card.md.",
        "",
        "## Per Sample",
        "",
        "| Sample | Category | Official answer | Result | Response transcript |",
        "| --- | --- | --- | --- | --- |",
        *rows,
        "",
        "## Layout",
        "",
        "- `samples/<id>/`: meta.json, output.wav, question.txt, response.txt, result.txt.",
        "- `logs/`: client and scoring logs, container logs and FDA events of the run window, backend traces.",
        "- `provenance/`: condition card, git state, Compose files, agent settings, images, client constants, runbook.",
        "",
    ]
    (dump / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {dump / 'README.md'}: {len(rows)} samples, accuracy {accuracy}")
    EOF
    ```

    Compare the accuracy and the label counts with the output of `analyze_results.py` in `logs/analyze.log`. They must match.

7. Check that the dump holds no key. The commands print only the names of files that hold a key value, and never the value:

    ```bash
    grep -rIlE 'sk-[A-Za-z0-9_-]{20,}|nvapi-[A-Za-z0-9_-]{20,}' "$DUMP"
    for key in NVIDIA_API_KEY REALTIME_API_KEY HF_TOKEN; do
      values=$({ printenv "$key"; grep "^$key=." .env | cut -d= -f2-; } | grep -v '^$')
      [ -z "$values" ] || printf '%s\n' "$values" | grep -rlF -f - "$DUMP"
    done
    for archive in "$DUMP"/logs/*.tar.gz "$DUMP"/provenance/*.tar.gz; do
      tar -xzOf "$archive" | grep -qE 'sk-[A-Za-z0-9_-]{20,}|nvapi-[A-Za-z0-9_-]{20,}' && echo "key pattern in $archive"
    done
    ```

    The check passes when all three commands print nothing. The loop skips a key that has no value, because an empty pattern list would match every file. If a command prints a file name, delete that file from the dump, find where the key came from, and dump that file again without it.

8. Hand the dump to the owner of the run. The dump repository stores `*.wav`, `*.log`, `*log*.txt`, `*.jsonl`, and `*.gz` files in Git LFS. Confirm that the audio is tracked by LFS before a commit:

    ```bash
    git -C "$DUMP_REPO" check-attr filter -- "big-bench-audio/$RUN_ID/samples/0/output.wav"
    ```

    The command prints `filter: lfs`. Commit and push the dump only with the owner's approval.

## Troubleshooting

The following table lists failures that are specific to a BBA run on FDA. For deployment failures, refer to [Troubleshooting](../../docs/how-to/deploy-frontend-delegation-agent.md#troubleshooting) in the deploy guide.

| Symptom | Cause | Fix |
| --- | --- | --- |
| Every sample fails with an HTTP 403 error. | The server rejected the credential. The client sent no key or a key that differs from the server's `REALTIME_API_KEY`. | Run the checks in [Check the Credentials](#check-the-credentials). If you changed `.env` after the deployment started, rerun the `up` command of [Deploy the Agent](#deploy-the-agent), so the server reads the new key. |
| Samples fail with `server_busy`. | All 4 Realtime sessions are taken, for example because `--batch_size` is above 4 or another client is connected. The message `server is starting` means that the warm-up failed. | Use `--batch_size 4` with no other client connected, and rerun the failed samples with `--retry_samples`. For a warm-up failure, read the `startup failed` line in the app log. |
| The first sample after a container start fails with a timeout. | The first session runs FDA's warm-up, which counts against the client's 30-second handshake limit. | Run the warm-up sample again, as [Run a Warm-Up Sample](#run-a-warm-up-sample) describes. If it keeps failing, read the app log for `startup failed`. |
| Some `output.wav` files hold only the filler, "One moment." | The backend's answer did not start in time for the client's stop rules. After audio arrives, the client stops at the first 2-second stretch with no message that ends at least 2 seconds after the last audio chunk arrived. The backend model reasons before it answers, so some answers arrive later. | No fix is needed. The sample scores as given, and the same rule applies to every agent. Do not change the client constants. Record how many samples hold only the filler. |
| The client receives `session.created` with `silence_duration_ms: 800`. | `session.created` reports the configured voice value. It does not drive the VAD. | No fix is needed. The `Pipecat adapter settings` line shows the effective `stop_secs`. |
| The `Pipecat adapter settings` line shows `stop_secs` 0.8, or `printenv FDA_VAD_STOP_SECS` prints `0.8`. | The deployment runs without the BBA overlay. | Redeploy with all four files, as [Deploy the Agent](#deploy-the-agent) describes. |
| `output.wav` files hold one or more status lines, such as "Still checking, one moment.", before the answer. | The BBA overlay turns on a status line after 1.0 second of silence while the backend works, at most 3 for each backend run. | No fix is needed. The status lines keep the client listening for a slower answer, and the judge assesses the answer. |
| The proactive status line shows `after_s=15` or `max_per_run=1`, or `printenv` prints `15` and `1` for `FDA_PROACTIVE_AFTER_S` and `FDA_PROACTIVE_MAX_PER_RUN`. | The deployment runs without the BBA overlay. | Redeploy with all four files, as [Deploy the Agent](#deploy-the-agent) describes. |
| `speech-inference.py` exits with an argument error that names `--model`. | `--model` is valid only with `--transport realtime`. | Add `--transport realtime`, or remove `--model`. |
| `eval.py` writes a label that is neither `CORRECT` nor `INCORRECT`, such as `THE` or `COUNTING`, and `--retry_samples` gives the same label again. | The judge model starts its reply with an explanation. `eval.py` allows the judge 8 tokens and keeps the first word when the reply has no label. Claude Sonnet 5.5 does this for answers that hold more than the filler. | Use the judge that the [Condition Card](#condition-card) requires, Claude Sonnet 5, and rerun `eval.py` with `--retry_samples` for the affected samples. Do not change the prompt or the token limit of `eval.py`. |
| `transcribe.py` fails with a gRPC connection error, such as `UNAVAILABLE`. | No ASR service listens at the host and port. The `nemo-speech` container is stopped or restarting, for example because its model files are missing. | Check the container with `docker compose $FDA_FILES --profile frontend-delegation-agent/single-gpu ps nemo-speech` and read its log. Rerun `transcribe.py` with `--host localhost --port 50051` when it is running. |

## Stop and Clean Up

Complete [Dump the Run](#dump-the-run) first: `down` removes the containers and their logs. Then stop the deployment from the repository root with the same files that started it:

```bash
docker compose $FDA_FILES --profile frontend-delegation-agent/single-gpu down
```

The command keeps `.env`, the built images, the logs in `logs/`, and the speech model weights.

To return to a Tau or other eval deployment with the LLM backend, rerun the eval `up` command without the BBA overlay and without `FDA_PROMPT_PROFILE`. Compose then recreates the voice service with `FDA_VAD_STOP_SECS` 0.8 and the `customer_service` prompt profile:

```bash
docker compose -f docker-compose.yml -f docker/docker-compose.fda-eval.yaml -f docker/docker-compose.fda-llm.yaml \
  --profile frontend-delegation-agent/single-gpu up -d --build
```

Make sure that `.env` does not set `FDA_PROMPT_PROFILE` before a Tau run.

## Related Topics

- [BigBench Audio Eval README](README.md)
- [Deploy the Frontend Delegation Agent](../../docs/how-to/deploy-frontend-delegation-agent.md)
- [Evaluation and Performance](../../docs/04-evaluation-and-performance.md)
