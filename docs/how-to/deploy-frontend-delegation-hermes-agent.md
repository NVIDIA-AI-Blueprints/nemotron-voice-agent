# Deploy the Frontend Delegation Hermes Agent

## Overview

This guide takes you from a clean host to a running and verified deployment of the [Frontend Delegation Hermes (FDH) Agent](../../src/examples/frontend_delegation_hermes/README.md). A fast frontend large language model (LLM) talks to the user and delegates work to a Hermes agent backend. A backend gateway starts one Hermes worker process per session.

Both LLMs run on the NVIDIA Inference Hub at `https://inference-api.nvidia.com/v1`:

- The frontend LLM is `nvidia/nvidia/nemotron-3.5-lightning`.
- The backend LLM is `nvidia/nvidia/nemotron-3-ultra`.

The agent serves two kinds of sessions:

- Browser UI sessions over WebRTC or the UI's WebSocket transport, with the Real-Time Voice Inference (RTVI) protocol, like the other examples in this repository.
- OpenAI Realtime sessions on `/v1/realtime` with the model `nvidia/nemotron-realtime-frontend-delegation-hermes`, for Realtime clients and evaluation runs such as tau-bench (Tau).

The example has one Docker Compose recipe, `frontend-delegation-hermes-agent/single-gpu`. It runs the following containers:

- `frontend-delegation-hermes-agent-single-gpu`: the voice side in the shared app image, `nemotron-voice-agent:latest`, on the app port.
- `fdh-hermes`: the FDH gateway and its Hermes workers in the Hermes image, `nemotron-voice-agent-fdh:local`, with Python 3.14. The gateway listens on port `8790` of the Compose network and is not published to the host.
- `nemo-speech`: automatic speech recognition (ASR) and text-to-speech (TTS) with NeMo-Speech.cpp on one local GPU.
- `fdh-hermes-image`: a build-only service. It builds the Hermes image and exits.

The `FDH_PROFILE` setting decides what a deployment serves. Refer to [Choose a Profile](#choose-a-profile).

> **Note:** The commands in this guide are checked against the repository's Compose files, `docker/Dockerfile.fdh`, `docker/fdh/gateway.sh`, `.env.example`, and source code. They were not run on GPU hardware for this revision of the guide.

## Instructions for AI Agents

If you are an AI agent that deploys this example for a user, follow these rules:

1. If the user's request does not say which profile to deploy, ask: "Which profile: `default` (for the browser UI) or `eval` (for evaluation runs such as Tau)?" Wait for the answer. Do not choose one silently, even though Compose uses `default` when no overlay is passed.
2. For `default`, use the plain recipe command. For `eval`, add the `docker/docker-compose.fdh-eval.yaml` overlay to every Compose command. Refer to [Choose a Profile](#choose-a-profile).
3. For Tau or any other benchmark, always deploy `eval`. A `default` deployment pairs Realtime sessions with the wrong gateway configuration.
4. After startup and the first session, read the profile and the configuration hash from the app log, and the gateway configuration and source hash from the `fdh-hermes` log. Report them to the user. Refer to [Verify the Deployment](#verify-the-deployment).
5. Check only that `NVIDIA_API_KEY` and `REALTIME_API_KEY` are present in `.env`. Do not print their values.

Do not overwrite other keys in an existing `.env`. For smoke tests, benchmark checks, and logs, load the deploy skill at `.agents/skills/nemotron-voice-agent-fdh-deploy/SKILL.md`.

## Prerequisites

The deployment needs the following:

- Docker Compose v2.20 or newer.
- An x86_64 Linux host with one NVIDIA GPU for the `nemo-speech` container (GPU 0). The LLMs do not use the local GPU. The Hermes image's lock, `docker/fdh/requirements.txt`, is resolved for x86_64 Linux, so DGX Spark and Jetson Thor are not supported.
- NVIDIA driver 580.x or newer and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- Python 3 and either the Hugging Face `hf` CLI or `uvx` for the speech model download script.
- Git, to clone the Hermes agent.
- Outbound HTTPS access from the host to the NVIDIA Inference Hub at `https://inference-api.nvidia.com/v1`.
- Free host ports `7860` for the app, or another port set with `PIPELINE_APP_PORT`, and `50051` for `nemo-speech`.
- A clone of this repository. Run every command in this guide from the repository root.

## Credentials

The recipe uses the `.env` keys in the following table.

| `.env` key | Required | Purpose |
| --- | --- | --- |
| `NVIDIA_API_KEY` | Yes | An NVIDIA Inference Hub key (`sk-...`). Both LLMs send it. It is the only `.env` key that reaches the `fdh-hermes` container, and the gateway refuses to start without it. |
| `HF_TOKEN` | Yes | Hugging Face token. The download script reads it to fetch the NeMo-Speech.cpp model weights. |
| `REALTIME_API_KEY` | Recommended | The Bearer credential that OpenAI Realtime clients send. Leave it unset only for unauthenticated local development. |
| `HERMES_AGENT_DIR` | No | Path of the Hermes checkout for the image build. The default is `../hermes-agent`. Refer to [Get Hermes](#get-hermes). |

The recipe does not use `NVIDIA_SPEECH_API_KEY` and does not need `docker login nvcr.io`.

## Choose a Profile

The agent has two profiles. The following table compares them.

| Profile | Use it for | What it serves | Delegation profile | Pipecat settings | Gateway configuration | TLS | Deploy command |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `default` | The browser UI and development | Browser UI sessions with the `default` settings, and OpenAI Realtime sessions with the `eval` settings | `profiles/pipecat_demo.yaml` | `adapters/pipecat.yaml` | `gateway.yaml` | On (HTTPS and `wss://`) | `docker compose --profile frontend-delegation-hermes-agent/single-gpu up -d --build` |
| `eval` | Evaluation runs, such as Tau | OpenAI Realtime sessions only. Browser UI sessions are refused. | `profiles/pipecat_realtime_eval.yaml` | `adapters/pipecat.eval.yaml` | `gateway.eval.yaml` | Off (HTTP and `ws://`) | `docker compose -f docker-compose.yml -f docker/docker-compose.fdh-eval.yaml --profile frontend-delegation-hermes-agent/single-gpu up -d --build` |

The file paths are relative to `src/examples/frontend_delegation_hermes/config/`.

OpenAI Realtime sessions always use the `eval` delegation profile and Pipecat settings, whatever `FDH_PROFILE` says. The gateway configuration is different: one `fdh-hermes` container serves every session, so the deployment sets it, not the session. Only the eval overlay switches the gateway to `gateway.eval.yaml`. A Realtime session on a `default` deployment therefore pairs the eval frontend with `gateway.yaml`. Benchmark runs must use the eval overlay.

The `eval` profile refuses browser UI sessions, so they cannot take resources from an evaluation run. The UI shows no error: the session ends, and the app log shows the error.

`docker-compose.yml` sets every FDH variable of the voice service as a literal value, so `.env` cannot change an evaluation run. Setting `FDH_PROFILE` or another `FDH_*` key in `.env` has no effect on the Compose recipe. Only the eval overlay changes these values. `FDH_PROFILE` accepts only `default` and `eval`. Any other value fails each session with an error that names both.

## Configure the Environment

Create `.env` from the template only when it does not exist. This command preserves an existing `.env`:

```bash
test -f .env || cp .env.example .env
```

Open `.env` in an editor and set the keys from [Credentials](#credentials). Use your own values in place of the placeholders:

```env
NVIDIA_API_KEY=<your-nvidia-inference-hub-key>
HF_TOKEN=<your-hugging-face-token>
REALTIME_API_KEY=<a-strong-random-secret>
```

The following optional keys change the deployment. Leave them unset to keep the defaults.

| Key | Default | Purpose |
| --- | --- | --- |
| `PIPELINE_APP_PORT` | `7860` | Host port of the app service. |
| `PIPELINE_TLS` | `true` | `true` serves HTTPS and `wss://` with a self-signed certificate. `false` serves plain HTTP and `ws://`. Browsers need HTTPS for the microphone, except on `localhost`. The eval overlay sets `false`. |
| `NEMO_SPEECH_MODEL_LOC` | `models/nemo-speech` | Host directory that holds the speech model weights. |
| `ENABLE_WELCOME_MESSAGE` | Unset (greets) | `false` turns off the browser UI greeting. Realtime sessions never greet. |
| `HERMES_AGENT_DIR` | `../hermes-agent` | Path of the Hermes checkout for the image build. |

Do not set `UVICORN_WS_PING_INTERVAL` for this recipe. Compose sets it to `0` on the voice service, and Realtime sessions refuse to start when the server pings WebSocket clients.

## Get Hermes

The Hermes image installs Hermes from a clean checkout of the official [Hermes Agent repository](https://github.com/nousresearch/hermes-agent) at a pinned commit. The header of `docker/fdh/requirements.txt` records the commit, `af26acab737db4382815aadcd73c06755471ab52`. The `fdh-hermes-image` service passes the checkout to the build as the named build context `hermes`.

The `HERMES_AGENT_DIR` key in `.env` sets the checkout path. A relative path resolves against the repository root, and the default, `../hermes-agent`, is a sibling directory of this repository. Set an absolute path to avoid ambiguity.

The following commands clone Hermes at the pinned commit, record its absolute path in `.env`, and check the checkout:

```bash
HERMES_COMMIT=$(grep -m1 'Hermes commit' docker/fdh/requirements.txt | awk '{print $4}')
test -d ../hermes-agent || git clone https://github.com/nousresearch/hermes-agent.git ../hermes-agent
git -C ../hermes-agent checkout "$HERMES_COMMIT"
grep -q '^HERMES_AGENT_DIR=' .env || echo "HERMES_AGENT_DIR=$(realpath ../hermes-agent)" >> .env
HERMES_AGENT_DIR=$(grep -m1 '^HERMES_AGENT_DIR=' .env | cut -d= -f2-)
git -C "$HERMES_AGENT_DIR" status --porcelain
git -C "$HERMES_AGENT_DIR" rev-parse HEAD
```

The `status --porcelain` command must print nothing, and `rev-parse HEAD` must print the pinned commit. Hermes runs only inside the `fdh-hermes` container, so you do not need a Hermes environment on the host.

## Start the Deployment

The following table lists the services of the recipe. The later commands use these names.

| Service | Image | Role |
| --- | --- | --- |
| `frontend-delegation-hermes-agent-single-gpu` | `nemotron-voice-agent:latest` | The voice side: the frontend, speech, and both session kinds. |
| `fdh-hermes` | `nemotron-voice-agent-fdh:local` | The FDH gateway and its Hermes workers. |
| `fdh-hermes-image` | `nemotron-voice-agent-fdh:local` | Builds the Hermes image, and then exits. |
| `nemo-speech` | NeMo-Speech.cpp | ASR and TTS on GPU 0. |

Complete the following steps:

1. Download the NeMo-Speech.cpp weights once, as your user. Do not use `sudo`. The script reads `HF_TOKEN` from `.env`:

    ```bash
    bash scripts/download-nemo-speech-models.sh
    ```

2. Create the host log directory, so that it belongs to your user:

    ```bash
    mkdir -p logs
    ```

3. Check that Compose accepts the recipe and your `.env`. The command prints nothing when the configuration is valid:

    ```bash
    docker compose --profile frontend-delegation-hermes-agent/single-gpu config -q
    ```

4. Start the recipe with the `default` profile:

    ```bash
    docker compose --profile frontend-delegation-hermes-agent/single-gpu up -d --build
    ```

    For the `eval` profile, follow [Deploy in Eval Mode](#deploy-in-eval-mode) instead.

The first start builds both images and pulls the `nemo-speech` image, so it can take several minutes. Compose starts the services in order: `fdh-hermes-image` builds the image and exits, `fdh-hermes` starts and turns healthy after its gateway answers `GET /health`, and then the voice service starts.

The voice service mounts `src/` read-only, but the `fdh-hermes` container runs the source that is built into its image. Keep `--build` on every `up` after a source change, so that both sides run the same code. Rebuild also after a change to `docker/Dockerfile.fdh`, `docker/fdh/`, the Hermes checkout, or `HERMES_AGENT_DIR`.

## Verify the Deployment

Run the following checks in order.

1. Check that the services are running and that the voice service becomes healthy. The `fdh-hermes-image` service shows as exited with code 0:

    ```bash
    docker compose --profile frontend-delegation-hermes-agent/single-gpu ps -a
    ```

2. Check the liveness route of the app. Use `http://` when `PIPELINE_TLS=false`:

    ```bash
    curl -fk https://localhost:7860/health
    ```

    The route returns `{"status":"ok"}`. It confirms that the server process is up. It does not check speech, the LLMs, or the gateway.

3. Check the gateway health route inside the `fdh-hermes` container:

    ```bash
    docker compose --profile frontend-delegation-hermes-agent/single-gpu exec fdh-hermes \
      curl -s http://localhost:8790/health; echo
    ```

    The response contains `"ok": true`, the open `"sessions"`, `"max_sessions"` (8), and the number of running Hermes `"workers"`.

4. Check the gateway configuration and the source hash of the Hermes image:

    ```bash
    docker compose --profile frontend-delegation-hermes-agent/single-gpu logs fdh-hermes | grep 'FDH gateway'
    ```

    The line has the form `FDH gateway: config gateway.yaml, source <hash>`. The eval overlay shows `gateway.eval.yaml`. To confirm that the image runs your checkout, compute the same hash on the host and compare:

    ```bash
    find src -type f ! -path '*/__pycache__/*' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1
    ```

    A different hash means that the image is older than your checkout. Rerun the `up` command with `--build`.

5. Check the effective profile in the voice container:

    ```bash
    docker compose --profile frontend-delegation-hermes-agent/single-gpu exec \
      frontend-delegation-hermes-agent-single-gpu printenv FDH_PROFILE
    ```

6. Start one session. Open the browser UI, as [Use the Browser UI](#use-the-browser-ui) describes, or connect one Realtime client, as [Connect a Realtime Client](#connect-a-realtime-client) shows. The first session of each kind waits for a one-time warm-up: one frontend LLM request and a check of the gateway. Then read the ready line from the app log:

    ```bash
    docker compose --profile frontend-delegation-hermes-agent/single-gpu logs \
      frontend-delegation-hermes-agent-single-gpu | grep -E 'frontend-delegation-hermes-agent ready|backend gateway'
    ```

    The ready line has the form `frontend-delegation-hermes-agent ready (adapter=pipecat, profile=<profile>)`, followed by the loaded configuration files and the `config hash`. A browser UI session logs `profile=default`. A Realtime session logs `profile=eval`, in both profiles.

If a check fails, refer to [Troubleshooting](#troubleshooting).

## Use the Browser UI

The browser UI works with the `default` profile only. Open `https://<host>:7860` in a browser. Replace `<host>` with the host name or IP address, or use `localhost` on the same machine. The server uses a self-signed certificate, so the browser shows a warning that you must accept once. For plain-HTTP browser access, refer to [Browser Access](../06-troubleshooting.md#browser-access).

The recipe pins `EXAMPLE_SELECTION=frontend-delegation-hermes-agent`, so the UI shows only this example. When the server exposes several examples, select **Frontend Delegation Hermes Agent**. Then start a session and speak. The agent greets you with "Hi! How can I help you today?", unless `ENABLE_WELCOME_MESSAGE=false`.

A browser session runs the demo profile: the demo tools run in the server, and the agent waits 5 seconds before each backend run, so you can ask status questions and steer the work while the backend runs. Turn taking uses the Pipecat defaults.

The UI offers the following controls and views for this example:

- **LLM**: chooses the frontend LLM from the example's catalogs. The backend LLM is always the gateway's, Nemotron 3 Ultra on the NVIDIA Inference Hub. The UI has no control for it.
- **ASR**, **TTS**, and the TTS voice: choose the speech endpoints from the example's catalogs. A voice change during a session applies from the next sentence on.
- **Prompt**: the panel shows the informational `frontend` prompt from `prompts.yaml`. The agent does not apply edits. The frontend prompt is `frontend_system` in `config/prompts.yaml`.
- **Tools**: the read-only tools panel lists the two demo tools, `get_order` and `cancel_order`. The server runs them.
- **Transcripts and latency**: the UI shows the transcripts of both sides and the user-to-bot latency.

The demo tools know order `5512`, which has shipped, and order `5513`, which is processing. The following table lists what to say and what to expect.

| You say | Expected |
| --- | --- |
| "Can you give me the status of order 5512?" | A filler now, and the answer after about 5 seconds plus the model time |
| While it works: "What's the update so far?" | "Let me check…", then a spoken progress sentence. The running task continues. |
| While it works: "And order 5513 as well" | A filler. The running task is steered and answers both. |
| "Okay sure" | No reply, because the agent treats it as an acknowledgement |
| After an answer: "Please cancel order 5513" | A filler. A new run continues the same Hermes session. |
| Talk over an answer | The answer stops, and the backend learns what you heard |

Browser UI sessions share one cap of 4 concurrent sessions. That cap is separate from the cap of the Realtime sessions, so browser sessions never take an evaluation slot. Both kinds of sessions still share `nemo-speech`, the gateway, and the LLM endpoints.

## Connect a Realtime Client

Connect a Realtime client to the following URL. Replace `<host>` with the host name or IP address, and use `ws://` when `PIPELINE_TLS=false`:

```text
wss://<host>:7860/v1/realtime?model=nvidia/nemotron-realtime-frontend-delegation-hermes
```

When `REALTIME_API_KEY` is set, send it as a Bearer credential in the `Authorization` header. Realtime sessions always use the `eval` settings: the client's `session.update` supplies the tools, which the client runs, and the instructions, which become the backend's policy. The agent does not greet.

For a smoke test, use the repository's Realtime test client. It reads `REALTIME_API_KEY` from `.env` and relays a browser page to the server:

```bash
uv run python realtime_test_client/proxy.py --model nvidia/nemotron-realtime-frontend-delegation-hermes
```

Open `http://localhost:8765`, select **Connect**, and speak or send text. For an eval-mode deployment, add `--upstream ws://localhost:7860`. For the client's controls, refer to [Test from a Browser](use-realtime-gateway.md#test-from-a-browser).

Realtime sessions have the following client-facing limitations:

- Turns end on the server voice activity detection (VAD). The agent rejects `input_audio_buffer.commit`.
- The agent ignores the voice that the client requests and uses the catalog voice.
- At most 4 Realtime sessions run at the same time. A fifth session receives an `error` event with code `server_busy`, and the socket closes with code 1013.

For the full event reference, refer to [Use the OpenAI Realtime-Compatible Gateway](use-realtime-gateway.md).

## Deploy in Eval Mode

Eval mode deploys the `eval` profile for evaluation runs, such as a Tau evaluation. It uses the same images and code as the standard deployment. The `docker/docker-compose.fdh-eval.yaml` overlay changes the settings in the following table.

| Service | Setting | Eval mode value | Reason |
| --- | --- | --- | --- |
| `frontend-delegation-hermes-agent-single-gpu` | `FDH_PROFILE` | `eval` | The container serves OpenAI Realtime sessions only and refuses browser UI sessions. |
| `frontend-delegation-hermes-agent-single-gpu` | `PIPELINE_TLS` | `false` | The server serves plain HTTP and `ws://`, so an evaluation client connects without a certificate. |
| `fdh-hermes` | `FDH_GATEWAY_CONFIG` | `gateway.eval.yaml` | The gateway uses the evaluation backend settings, which pin every backend prompt variant and record the backend decision trace. |

Pass the overlay after `docker-compose.yml` in every Compose command for the deployment. A command without it recreates the containers with the `default` profile and `gateway.yaml`.

The eval Pipecat settings, `config/adapters/pipecat.eval.yaml`, measure the end of a user turn in input audio only. An evaluation client can pause its input for seconds, so a wall-clock timer would end turns mid-utterance. They change three keys compared to `adapters/pipecat.yaml`, which keeps the Pipecat defaults:

| Key | Default | Eval | Reason |
| --- | --- | --- | --- |
| `vad.stop_secs` | `0.2` | `0.8` | Silero counts 800 ms of silence in the audio. |
| `vad.audio_idle_timeout` | `1.0` (wall clock) | `0` (off) | A wall-clock timer would end the turn when the client pauses its input. |
| `turn.user_speech_timeout` | `0.6` (wall clock) | `0` | No wall-clock wait follows the VAD stop. |

To deploy in eval mode, complete the following steps:

1. Complete [Configure the Environment](#configure-the-environment), [Get Hermes](#get-hermes), and steps 1 and 2 of [Start the Deployment](#start-the-deployment).

2. Start the recipe with the overlay:

    ```bash
    docker compose -f docker-compose.yml -f docker/docker-compose.fdh-eval.yaml \
      --profile frontend-delegation-hermes-agent/single-gpu up -d --build
    ```

3. Download the NLTK `punkt_tab` data in the voice container. Pipecat loads it on first use for sentence splitting. On a new container, the first sessions otherwise wait for the download. The command prints `True`:

    ```bash
    docker compose -f docker-compose.yml -f docker/docker-compose.fdh-eval.yaml \
      --profile frontend-delegation-hermes-agent/single-gpu exec frontend-delegation-hermes-agent-single-gpu \
      .venv/bin/python -c "import nltk; print(nltk.download('punkt_tab', quiet=True))"
    ```

4. Check the liveness route over plain HTTP:

    ```bash
    curl -f http://localhost:7860/health
    ```

5. Check the eval settings. The first command prints `eval` and `false`. The second prints a line that names `gateway.eval.yaml`:

    ```bash
    docker compose -f docker-compose.yml -f docker/docker-compose.fdh-eval.yaml \
      --profile frontend-delegation-hermes-agent/single-gpu exec frontend-delegation-hermes-agent-single-gpu \
      printenv FDH_PROFILE PIPELINE_TLS
    docker compose -f docker-compose.yml -f docker/docker-compose.fdh-eval.yaml \
      --profile frontend-delegation-hermes-agent/single-gpu logs fdh-hermes | grep 'FDH gateway'
    ```

6. Connect one client to `ws://localhost:7860/v1/realtime?model=nvidia/nemotron-realtime-frontend-delegation-hermes`, as [Connect a Realtime Client](#connect-a-realtime-client) shows. Then check that the voice event log grew on the host:

    ```bash
    tail -n 1 logs/fdh_voice_events.pipecat.jsonl
    ```

To switch back to the standard deployment, run the `up -d --build` command from [Start the Deployment](#start-the-deployment) without the overlay.

## Run a Tau Evaluation

The Tau evaluation harness is not part of this repository. Use the following steps to point it at this deployment:

1. Deploy in eval mode, as [Deploy in Eval Mode](#deploy-in-eval-mode) describes. Every Tau run must use the eval overlay, because only the overlay switches the gateway to `gateway.eval.yaml`.
2. Configure the harness with the WebSocket URL `ws://<host>:7860/v1/realtime`, the model `nvidia/nemotron-realtime-frontend-delegation-hermes`, and your `REALTIME_API_KEY` as the Bearer credential. The former `pine-*` model names and the former ports `8775` and `8776` no longer reach FDH.
3. Keep the session count within the cap of 4 Realtime sessions.
4. Record the gateway configuration and the source hash from the `fdh-hermes` log, and the profile and `config hash` from the app log, with the results.

The `config hash` includes the backend URL, which is now `ws://fdh-hermes:8790/v1/backend`. It therefore differs from runs on the former FDH containers. To compare runs, use the prompt hashes and `session_tools_sha256`. Refer to [Deployed Fingerprint](../../src/examples/frontend_delegation_hermes/README.md#deployed-fingerprint).

## Read the Logs

The containers write the logs in the following table. The `logs/` paths are on the host, in the repository's `logs/` directory.

| Log | Content |
| --- | --- |
| `docker compose --profile frontend-delegation-hermes-agent/single-gpu logs frontend-delegation-hermes-agent-single-gpu` | The app log: the ready line, the Pipecat settings, and per-session errors. |
| `docker compose --profile frontend-delegation-hermes-agent/single-gpu logs fdh-hermes` | The gateway log: its configuration, the source hash, and gateway errors. |
| `logs/fdh_voice_events.pipecat.jsonl` | The voice event log of both session kinds. |
| `logs/fdh_gateway_events.pipecat.jsonl` | The gateway event log. |
| `logs/fdh_workers_pipecat/` | One log per Hermes worker. With `gateway.eval.yaml`, also one `<session_id>.trace.jsonl` backend decision trace per session. |

The containers write the log files as `root`. The repository ignores the `logs/` directory.

## Stop and Clean Up

Stop the recipe with the same Compose profile that started it:

```bash
docker compose --profile frontend-delegation-hermes-agent/single-gpu down
```

For an eval-mode deployment, pass the overlay as well:

```bash
docker compose -f docker-compose.yml -f docker/docker-compose.fdh-eval.yaml \
  --profile frontend-delegation-hermes-agent/single-gpu down
```

The gateway closes open sessions one after another, so the stop can take up to 75 seconds. The command keeps `.env`, the built images, the logs in `logs/`, and the speech model weights in `models/nemo-speech`.

## Troubleshooting

The following table lists common failures for this recipe. For general startup issues, refer to [Troubleshooting](../06-troubleshooting.md).

| Symptom | Cause | Fix |
| --- | --- | --- |
| `up` fails while it builds `fdh-hermes-image`, and the error names the `hermes` build context. | The Hermes checkout is missing at `HERMES_AGENT_DIR`. | Complete [Get Hermes](#get-hermes), and set an absolute `HERMES_AGENT_DIR` in `.env`. |
| The image build fails in the `uv pip check` step. | The Hermes checkout does not match the lock in `docker/fdh/requirements.txt`. | Check out the commit from the lock header in a clean checkout, or regenerate the lock as its header describes. |
| `fdh-hermes` exits at start, its log says `NVIDIA_API_KEY is not set`, and the voice service never starts. | `.env` has no `NVIDIA_API_KEY`. | Set `NVIDIA_API_KEY` to an NVIDIA Inference Hub key, and then rerun the `up` command. |
| A browser UI session ends at once or never starts, and the app log shows `runs with FDH_PROFILE=eval`. | The deployment runs the `eval` profile, which refuses browser UI sessions. | Start the recipe without the eval overlay. |
| A browser UI session ends at once, and the app log shows `browser session refused`. | Either 4 browser UI sessions are already open, or the warm-up failed. | Close idle browser sessions. For a warm-up failure, read the `startup failed` line in the app log, and then start a new session. |
| A Realtime client receives an `error` event with code `server_busy`, and the socket closes with code 1013. | The message `server at capacity (server.max_sessions)` means that 4 Realtime sessions are open. The message `server is starting` means that the warm-up failed. | Close idle sessions, or wait and reconnect. For a warm-up failure, read the `startup failed` line in the app log. The next session retries the warm-up. |
| The app log shows `server.max_sessions (N) exceeds the gateway's max_sessions`. | The voice side allows more sessions than the gateway. | Use the Compose values, which set 4 sessions per session kind and 8 gateway sessions. |
| The app log shows `backend gateway not reachable`, and sessions fail. | The `fdh-hermes` gateway is not running or not healthy. | Run `docker compose --profile frontend-delegation-hermes-agent/single-gpu logs fdh-hermes`, fix the cause, and rerun the `up` command. |
| A Realtime client receives an `error` event with code `server_error`, the socket closes with code 1011, and the error asks for `UVICORN_WS_PING_INTERVAL=0`. | The server pings WebSocket clients, which the `eval` settings do not allow. This happens in a host-native run or in a modified Compose file. | Set `UVICORN_WS_PING_INTERVAL=0` for the server process. |
| The WebSocket handshake fails, and the server closes it with code 1008 and reason `realtime authentication failed`. | `REALTIME_API_KEY` is set, and the Realtime client sent no credential or a different one. | Send `Authorization: Bearer <your-realtime-api-key>` with the value from `.env`. |
| `session.update` fails with `worker_start_timeout` or `backend_unavailable`. | A Hermes worker did not start. | Read the worker logs in `logs/fdh_workers_pipecat/`. Rerun the `up` command with `--build` if the Hermes checkout changed. |
| A code change works on the voice side but not in Hermes, or the source hash in the `fdh-hermes` log differs from your checkout. | The Hermes image runs the source that was built into it. | Rerun the `up` command with `--build`. |
| A benchmark run shows other backend prompt hashes than expected, or the `fdh-hermes` log names `gateway.yaml`. | The deployment runs without the eval overlay. | Redeploy with `-f docker/docker-compose.fdh-eval.yaml`, and pass it in every Compose command. |
| The client fails with `ssl argument is incompatible with a ws:// URI`, or a `wss://` handshake fails with an SSL error. | The client and the server disagree on TLS. Eval mode serves plain `ws://`. | Use `ws://` for an eval-mode deployment and `wss://` for the standard deployment. |
| The browser does not ask for the microphone. | The page is served over plain HTTP from another machine. | Keep `PIPELINE_TLS=true` (the default) and open the `https://` URL. |
| The download script fails, or `nemo-speech` cannot find its models. | `HF_TOKEN` is missing or invalid, or the weights are in another directory. | Set `HF_TOKEN` in `.env` and rerun the script as your user, or point `NEMO_SPEECH_MODEL_LOC` at the weights. |
| `up -d` fails with `port is already allocated` or `address already in use`. | Another process uses the app port or port `50051`. | Set `PIPELINE_APP_PORT` to a free port for the app. For `50051`, stop the other recipe that runs a speech sidecar. |
| `mv` or `rm` in `logs/` fails with permission denied. | The containers write logs as `root`. | Run `sudo chown -R "$USER" logs`. |

## Related Topics

- [Frontend Delegation Hermes README](../../src/examples/frontend_delegation_hermes/README.md)
- [Deploy the Frontend/Backend Agent](deploy-frontend-backend-agent.md)
- [Use the OpenAI Realtime-Compatible Gateway](use-realtime-gateway.md)
- [WebSocket Keepalive](tune-pipeline-performance.md#websocket-keepalive)
- [Troubleshooting](../06-troubleshooting.md)
