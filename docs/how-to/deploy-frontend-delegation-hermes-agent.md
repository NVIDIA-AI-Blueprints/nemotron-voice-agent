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
- `fdh-hermes`: the controller of the Hermes backend. By default, it runs the FDH gateway and its Hermes workers in an NVIDIA OpenShell sandbox. The sandbox runs the Hermes image, `nemotron-voice-agent-fdh:local`, with Python 3.14.
- `hermes-openshell-gateway`: the OpenShell gateway. It creates the sandbox and routes the voice side's backend connection to it.
- `hermes-openshell-init`: a one-time service. It creates the OpenShell gateway's signing keys and exits.
- `nemo-speech`: automatic speech recognition (ASR) and text-to-speech (TTS) with NeMo-Speech.cpp on one local GPU.
- `fdh-hermes-image`: a build-only service. It builds the Hermes image and exits.

The `FDH_PROFILE` setting decides what a deployment serves. Refer to [Choose a Profile](#choose-a-profile). The Hermes backend runs in OpenShell by default, or in a plain container with an overlay. Refer to [Choose the Hermes Runtime](#choose-the-hermes-runtime).

> **Note:** The commands in this guide are checked against the repository's Compose files, `docker/Dockerfile.fdh`, `docker/fdh/gateway.sh`, `docker/fdh/openshell/`, `.env.example`, and source code. They were not run on GPU hardware for this revision of the guide. The OpenShell runtime has not yet been validated on a live host. If it does not start on your host, use the plain runtime.

## Instructions for AI Agents

If you are an AI agent that deploys this example for a user, follow these rules:

1. If the user's request does not say which profile to deploy, ask: "Which profile: `default` (for the browser UI) or `eval` (for evaluation runs such as Tau)?" Wait for the answer. Do not choose one silently, even though Compose uses `default` when no overlay is passed.
2. For `default`, use the recipe command without overlays. For `eval`, add the `docker/docker-compose.fdh-eval.yaml` overlay to every Compose command. Refer to [Choose a Profile](#choose-a-profile).
3. For Tau or any other benchmark, always deploy `eval`. A `default` deployment pairs Realtime sessions with the wrong gateway configuration.
4. Use the default OpenShell runtime. Add the plain runtime overlay, `docker/docker-compose.fdh-hermes-plain.yaml`, only when the user asks for it or the host cannot run OpenShell. Pass the Compose files in this order: `docker-compose.yml`, then the plain overlay, then the eval overlay. Refer to [Choose the Hermes Runtime](#choose-the-hermes-runtime).
5. After startup and the first session, read the profile and the configuration hash from the app log, and the gateway configuration and source hash from the Hermes logs. Report them to the user. Refer to [Verify the Deployment](#verify-the-deployment).
6. Check only that `NVIDIA_API_KEY` and `REALTIME_API_KEY` are present in `.env`. Do not print their values.

Do not overwrite other keys in an existing `.env`. For smoke tests, benchmark checks, and logs, load the deploy skill at `.agents/skills/nemotron-voice-agent-fdh-deploy/SKILL.md`.

## Prerequisites

The deployment needs the following:

- Docker Engine 28 or newer and Docker Compose 2.24.4 or newer. The plain runtime overlay uses the Compose `!override` and `!reset` tags, which need Compose 2.24.4.
- An x86_64 Linux host with one NVIDIA GPU for the `nemo-speech` container (GPU 0). The LLMs do not use the local GPU. The Hermes image's lock, `docker/fdh/requirements.txt`, is resolved for x86_64 Linux, so DGX Spark and Jetson Thor are not supported.
- NVIDIA driver 580.x or newer and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- Python 3 and either the Hugging Face `hf` CLI or `uvx` for the speech model download script.
- Git, to clone the Hermes agent.
- Outbound HTTPS access from the host to the NVIDIA Inference Hub at `https://inference-api.nvidia.com/v1`.
- Free host ports `7860` for the app, or another port set with `PIPELINE_APP_PORT`, and `50051` for `nemo-speech`.
- A clone of this repository. Run every command in this guide from the repository root.

The default OpenShell runtime also needs the following on the host:

- Linux kernel 6.2 or newer with Landlock turned on. The sandbox refuses to run without Landlock. Check the kernel with `uname -r`.
- A free host port `127.0.0.1:8080` for the OpenShell gateway.
- No other OpenShell gateway on the host. The port `8080` and the directory `/var/lib/openshell` are shared by the whole host, so only one OpenShell gateway can run per host. This includes a second checkout of this repository.
- Write access for Docker to `/var/lib/openshell`. Docker creates the directory as `root` on the first start. It holds the gateway's signing keys and database.

The plain runtime does not need these. Refer to [Choose the Hermes Runtime](#choose-the-hermes-runtime).

## Credentials

The recipe uses the `.env` keys in the following table.

| `.env` key | Required | Purpose |
| --- | --- | --- |
| `NVIDIA_API_KEY` | Yes | An NVIDIA Inference Hub key (`sk-...`). Both LLMs send it. It is the only `.env` key that reaches the Hermes side, and `fdh-hermes` refuses to start without it. With OpenShell, the key goes to the OpenShell provider, never into the sandbox. |
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

## Choose the Hermes Runtime

The Hermes backend, that is the FDH gateway and its Hermes workers, can run in one of two runtimes. Both run the same image, `nemotron-voice-agent-fdh:local`, with the same `docker/fdh/gateway.sh` start script and the same `docker/fdh/gateway.env` settings. Agent behavior, turn taking, and evaluation behavior are the same in both. The following table compares them.

| Runtime | When to use it | Backend URL of the voice side | Extra Compose file |
| --- | --- | --- | --- |
| OpenShell (default) | Normal use and evaluation runs | `ws://default--fdh-hermes.openshell.localhost:8080/v1/backend` | None |
| Plain | A/B checks against OpenShell, debugging, and hosts that cannot run OpenShell | `ws://fdh-hermes:8790/v1/backend` | `docker/docker-compose.fdh-hermes-plain.yaml` |

In the OpenShell runtime, the following rules apply to the sandbox:

- The FDH gateway and its workers run as user ID 1500. The code and packages under `/app` are read-only. Only `/app/logs`, `/app/.cache`, `/tmp`, and `/sandbox` are writable. The rules are in `docker/fdh/openshell/policy.yaml`.
- The sandbox can call only `inference-api.nvidia.com`, plus any host that you add to `policy.yaml`.
- The sandbox never sees the real `NVIDIA_API_KEY`. Its `NVIDIA_API_KEY` holds a placeholder. OpenShell adds the real key to requests to `inference-api.nvidia.com` only. The provider profile is `docker/fdh/openshell/nvidia-fdh.yaml`.
- The OpenShell release is pinned in `docker/fdh/openshell/VERSION`. Every OpenShell image and the OpenShell command-line interface (CLI) use that release.

Always pass the Compose files in this order: `docker-compose.yml`, then the plain overlay if you use it, then the eval overlay if you use it. The following commands start each combination:

```bash
# OpenShell runtime (default)
docker compose -f docker-compose.yml \
  --profile frontend-delegation-hermes-agent/single-gpu up -d --build

# OpenShell runtime, eval profile
docker compose -f docker-compose.yml -f docker/docker-compose.fdh-eval.yaml \
  --profile frontend-delegation-hermes-agent/single-gpu up -d --build

# Plain runtime
docker compose -f docker-compose.yml -f docker/docker-compose.fdh-hermes-plain.yaml \
  --profile frontend-delegation-hermes-agent/single-gpu up -d --build

# Plain runtime, eval profile
docker compose -f docker-compose.yml -f docker/docker-compose.fdh-hermes-plain.yaml \
  -f docker/docker-compose.fdh-eval.yaml \
  --profile frontend-delegation-hermes-agent/single-gpu up -d --build
```

Pass the same files in every later Compose command for the deployment, such as `ps`, `logs`, `exec`, and `down`. The plain overlay does not start `hermes-openshell-init` or `hermes-openshell-gateway`. In the plain runtime, `fdh-hermes` is the FDH gateway container itself, on port `8790` of the Compose network, and is not published to the host.

### Known Security Limitations

The OpenShell runtime has the following known limitations. Each one is a deployment or transport setting, not a design limit. All of them can be fixed without changing agent behavior or evaluation results, and the fixes are planned.

- The OpenShell gateway mounts the Docker socket, which gives it the same power as `root` on the host.
- The OpenShell gateway runs without TLS and without user authentication. Compose publishes it only on the host loopback address, `127.0.0.1:8080`.
- The `/v1/backend` endpoint has no authentication, in either runtime.
- OpenShell adds the real `NVIDIA_API_KEY` to any request to `inference-api.nvidia.com`, not only to the model calls that Hermes makes.
- The image is not pinned by digest, and the plain runtime runs as `root`.

Until these fixes are available, run the recipe on a single-user host. Do not expose the internal Docker Compose network to other hosts or to untrusted containers.

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
| `HERMES_AGENT_DIR` | `../hermes-agent` | Path of the Hermes checkout for the image build. Both runtimes use it. |

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

The `status --porcelain` command must print nothing, and `rev-parse HEAD` must print the pinned commit. Hermes runs only inside the Hermes image, in the OpenShell sandbox or in the plain `fdh-hermes` container, so you do not need a Hermes environment on the host.

## Start the Deployment

The following table lists the services of the recipe. The later commands use these names.

| Service | Image | Role |
| --- | --- | --- |
| `frontend-delegation-hermes-agent-single-gpu` | `nemotron-voice-agent:latest` | The voice side: the frontend, speech, and both session kinds. |
| `fdh-hermes` | `nemotron-voice-agent-fdh-openshell-cli:local` | The controller. It creates the OpenShell sandbox `fdh-hermes`, which runs the FDH gateway and its Hermes workers from `nemotron-voice-agent-fdh:local`. It saves the sandbox logs and deletes the sandbox when it stops. |
| `hermes-openshell-gateway` | `ghcr.io/nvidia/openshell/gateway` (pinned) | The OpenShell gateway. It starts the sandbox through the Docker socket and routes the backend connection to it. |
| `hermes-openshell-init` | `ghcr.io/nvidia/openshell/gateway` (pinned) | Creates the OpenShell gateway's signing keys in `/var/lib/openshell/tls`, and then exits. It keeps existing keys. |
| `fdh-hermes-image` | `nemotron-voice-agent-fdh:local` | Builds the Hermes image, and then exits. |
| `nemo-speech` | NeMo-Speech.cpp | ASR and TTS on GPU 0. |

With the plain runtime overlay, `fdh-hermes` runs `nemotron-voice-agent-fdh:local` directly, and the two OpenShell services do not start.

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

    For the `eval` profile, follow [Deploy in Eval Mode](#deploy-in-eval-mode) instead. For the plain runtime, add the plain overlay, as [Choose the Hermes Runtime](#choose-the-hermes-runtime) shows.

The first start builds the images and pulls the `nemo-speech` and OpenShell images, so it can take several minutes. OpenShell publishes no CLI download for its prerelease tags, so `docker/fdh/openshell/Dockerfile.cli` builds the OpenShell CLI from source at the pinned commit. The first build compiles Rust code and takes several more minutes. Later builds reuse the Docker build cache.

Compose starts the services in order:

1. `fdh-hermes-image` builds the Hermes image and exits.
2. `hermes-openshell-init` creates the signing keys and exits. Then `hermes-openshell-gateway` starts.
3. `fdh-hermes` checks the kernel version, waits for the OpenShell gateway, sets up the API key provider, and creates the sandbox. It turns healthy after the FDH gateway in the sandbox answers `GET /health` through the OpenShell gateway. The health check allows 180 seconds for startup.
4. The voice service starts.

In the plain runtime, `fdh-hermes` turns healthy after its gateway answers `GET /health` on port `8790`.

The voice service mounts `src/` read-only, but the Hermes runtime runs the source that is built into the Hermes image. Keep `--build` on every `up` after a source change, so that both sides run the same code. Rebuild also after a change to `docker/Dockerfile.fdh`, `docker/fdh/`, the Hermes checkout, or `HERMES_AGENT_DIR`.

## Verify the Deployment

Run the following checks in order.

1. Check that the services are running and that the voice service becomes healthy. The `fdh-hermes-image` and `hermes-openshell-init` services show as exited with code 0:

    ```bash
    docker compose --profile frontend-delegation-hermes-agent/single-gpu ps -a
    ```

2. Check the liveness route of the app. Use `http://` when `PIPELINE_TLS=false`:

    ```bash
    curl -fk https://localhost:7860/health
    ```

    The route returns `{"status":"ok"}`. It confirms that the server process is up. It does not check speech, the LLMs, or the gateway.

3. Check the FDH gateway health route. In the OpenShell runtime, the check goes through the OpenShell gateway, the same route that the voice side uses:

    ```bash
    docker compose --profile frontend-delegation-hermes-agent/single-gpu exec fdh-hermes \
      curl -s -H 'Host: default--fdh-hermes.openshell.localhost' http://hermes-openshell-gateway:8080/health; echo
    ```

    In the plain runtime, check the gateway inside the `fdh-hermes` container instead:

    ```bash
    docker compose -f docker-compose.yml -f docker/docker-compose.fdh-hermes-plain.yaml \
      --profile frontend-delegation-hermes-agent/single-gpu exec fdh-hermes \
      curl -s http://localhost:8790/health; echo
    ```

    The response contains `"ok": true`, the open `"sessions"`, `"max_sessions"` (8), and the number of running Hermes `"workers"`.

4. Check the gateway configuration and the source hash of the Hermes image. In the OpenShell runtime, the `fdh-hermes` log shows the controller. It prints the image, the gateway configuration, and the OpenShell CLI version:

    ```bash
    docker compose --profile frontend-delegation-hermes-agent/single-gpu logs fdh-hermes
    ```

    The line `fdh-hermes (OpenShell runtime): image nemotron-voice-agent-fdh:local, gateway config gateway.yaml` names the configuration. The FDH gateway's own output is in the sandbox workload container, which OpenShell starts from the Hermes image. The following commands find that container and read the gateway line:

    ```bash
    docker ps --filter ancestor=nemotron-voice-agent-fdh:local --format '{{.Names}}'
    docker logs <sandbox-workload-container> 2>&1 | grep 'FDH gateway'
    ```

    In the plain runtime, run `logs fdh-hermes | grep 'FDH gateway'` with the plain overlay instead. The line has the form `FDH gateway: config gateway.yaml, source <hash>`. The eval overlay shows `gateway.eval.yaml`. To confirm that the image runs your checkout, compute the same hash on the host and compare:

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

5. Check the eval settings. The first command prints `eval` and `false`. The second prints a controller line that names `gateway.eval.yaml`. In the plain runtime, it prints the `FDH gateway` line:

    ```bash
    docker compose -f docker-compose.yml -f docker/docker-compose.fdh-eval.yaml \
      --profile frontend-delegation-hermes-agent/single-gpu exec frontend-delegation-hermes-agent-single-gpu \
      printenv FDH_PROFILE PIPELINE_TLS
    docker compose -f docker-compose.yml -f docker/docker-compose.fdh-eval.yaml \
      --profile frontend-delegation-hermes-agent/single-gpu logs fdh-hermes | grep -E 'gateway config|FDH gateway'
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
4. Record the Hermes runtime, the gateway configuration and the source hash from the Hermes logs, and the profile and `config hash` from the app log, with the results.

The `config hash` includes the backend URL. The URL is `ws://default--fdh-hermes.openshell.localhost:8080/v1/backend` in the OpenShell runtime and `ws://fdh-hermes:8790/v1/backend` in the plain runtime. The hash therefore differs between the two runtimes and from runs on the former FDH containers. To compare runs, use the prompt hashes and `session_tools_sha256`. Refer to [Deployed Fingerprint](../../src/examples/frontend_delegation_hermes/README.md#deployed-fingerprint).

## Read the Logs

The containers write the logs in the following table. The `logs/` paths are on the host, in the repository's `logs/` directory.

| Log | Content |
| --- | --- |
| `docker compose --profile frontend-delegation-hermes-agent/single-gpu logs frontend-delegation-hermes-agent-single-gpu` | The app log: the ready line, the Pipecat settings, and per-session errors. |
| `docker compose --profile frontend-delegation-hermes-agent/single-gpu logs fdh-hermes` | OpenShell runtime: the controller log, with the image, the gateway configuration, the OpenShell CLI version, and setup errors. Plain runtime: the gateway log, with its configuration, the source hash, and gateway errors. |
| `logs/fdh_voice_events.pipecat.jsonl` | The voice event log of both session kinds. |
| `logs/fdh_gateway_events.pipecat.jsonl` | Plain runtime: the gateway event log. |
| `logs/fdh_workers_pipecat/` | Plain runtime: one log per Hermes worker. With `gateway.eval.yaml`, also one `<session_id>.trace.jsonl` backend decision trace per session. |
| `logs/openshell/<time>/` | OpenShell runtime: the gateway event log, the worker logs, and the backend decision traces, saved from the sandbox's `/app/logs` when `fdh-hermes` stops. |

In the OpenShell runtime, the gateway and worker log files stay inside the sandbox, in `/app/logs`, while it runs. The following commands read the other OpenShell logs:

- The FDH gateway and worker output, and Landlock denials: `docker logs <sandbox-workload-container>`. Find the container name with `docker ps --filter ancestor=nemotron-voice-agent-fdh:local --format '{{.Names}}'`.
- The policy decisions, that is the allowed and denied outbound calls of the sandbox:

    ```bash
    docker compose --profile frontend-delegation-hermes-agent/single-gpu exec fdh-hermes \
      openshell logs fdh-hermes
    ```

- A copy of `/app/logs` during a run. The copy lands in `logs/openshell/now/` on the host:

    ```bash
    docker compose --profile frontend-delegation-hermes-agent/single-gpu exec fdh-hermes \
      openshell sandbox download fdh-hermes /app/logs /out/now
    ```

Add the overlay files of your deployment to these commands. The containers write the log files as `root`. The repository ignores the `logs/` directory.

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

In the OpenShell runtime, `fdh-hermes` saves the sandbox's `/app/logs` to `logs/openshell/<time>/` and then deletes the sandbox. The stop can take up to 60 seconds. If the controller crashed and left a sandbox behind, the next start saves its logs and deletes it before it creates a new one.

In the plain runtime, the gateway closes open sessions one after another, so the stop can take up to 75 seconds. Pass the plain overlay to `down` as well.

The command keeps `.env`, the built images, the logs in `logs/`, the speech model weights in `models/nemo-speech`, and the OpenShell gateway's keys and database in `/var/lib/openshell`.

## Troubleshooting

The following table lists common failures for this recipe. For general startup issues, refer to [Troubleshooting](../06-troubleshooting.md).

| Symptom | Cause | Fix |
| --- | --- | --- |
| `up` fails while it builds `fdh-hermes-image`, and the error names the `hermes` build context. | The Hermes checkout is missing at `HERMES_AGENT_DIR`. | Complete [Get Hermes](#get-hermes), and set an absolute `HERMES_AGENT_DIR` in `.env`. |
| The image build fails in the `uv pip check` step. | The Hermes checkout does not match the lock in `docker/fdh/requirements.txt`. | Check out the commit from the lock header in a clean checkout, or regenerate the lock as its header describes. |
| `fdh-hermes` exits at start, its log says `NVIDIA_API_KEY is not set`, and the voice service never starts. | `.env` has no `NVIDIA_API_KEY`. | Set `NVIDIA_API_KEY` to an NVIDIA Inference Hub key, and then rerun the `up` command. |
| `fdh-hermes` exits, and its log says `OpenShell needs Linux 6.2+ (Landlock ABI 3)`. | The host kernel is older than 6.2. | Run the plain runtime, or use a host with Linux 6.2 or newer. |
| The sandbox fails to start, and the workload container log shows a Landlock error. | Landlock is turned off in the host kernel. The policy does not allow the sandbox to run without it. | Turn on Landlock in the kernel boot settings, or run the plain runtime. |
| `fdh-hermes` exits, and its log says `OpenShell gateway at http://hermes-openshell-gateway:8080 not connected after 120 s`. | The OpenShell gateway did not start. | Read `docker compose --profile frontend-delegation-hermes-agent/single-gpu logs hermes-openshell-gateway hermes-openshell-init`. Check the Docker socket and `/var/lib/openshell`. |
| `up -d` fails with `port is already allocated` for port `8080`. | Another process, or another OpenShell gateway, uses `127.0.0.1:8080`. | Stop the other process or OpenShell gateway. Only one OpenShell gateway can run per host. |
| `up` fails with an unknown `!override` or `!reset` tag. | Docker Compose is older than 2.24.4. | Update Docker Compose to 2.24.4 or newer. |
| `fdh-hermes` never turns healthy in the OpenShell runtime. | The sandbox did not start, or the FDH gateway in it failed. | Read the `fdh-hermes` log and the sandbox workload container log, as [Read the Logs](#read-the-logs) describes. Then rerun the `up` command. |
| The FDH gateway or a worker fails on a call to a host other than `inference-api.nvidia.com`. | The sandbox policy blocks the host. | Read the denied calls with `openshell logs fdh-hermes`, as [Read the Logs](#read-the-logs) shows. Add the host under `network_policies` in `docker/fdh/openshell/policy.yaml`, or compare with the plain runtime. |
| A browser UI session ends at once or never starts, and the app log shows `runs with FDH_PROFILE=eval`. | The deployment runs the `eval` profile, which refuses browser UI sessions. | Start the recipe without the eval overlay. |
| A browser UI session ends at once, and the app log shows `browser session refused`. | Either 4 browser UI sessions are already open, or the warm-up failed. | Close idle browser sessions. For a warm-up failure, read the `startup failed` line in the app log, and then start a new session. |
| A Realtime client receives an `error` event with code `server_busy`, and the socket closes with code 1013. | The message `server at capacity (server.max_sessions)` means that 4 Realtime sessions are open. The message `server is starting` means that the warm-up failed. | Close idle sessions, or wait and reconnect. For a warm-up failure, read the `startup failed` line in the app log. The next session retries the warm-up. |
| The app log shows `server.max_sessions (N) exceeds the gateway's max_sessions`. | The voice side allows more sessions than the gateway. | Use the Compose values, which set 4 sessions per session kind and 8 gateway sessions. |
| The app log shows `backend gateway not reachable`, and sessions fail. | The `fdh-hermes` gateway is not running or not healthy. | Run `docker compose --profile frontend-delegation-hermes-agent/single-gpu logs fdh-hermes`, fix the cause, and rerun the `up` command. |
| A Realtime client receives an `error` event with code `server_error`, the socket closes with code 1011, and the error asks for `UVICORN_WS_PING_INTERVAL=0`. | The server pings WebSocket clients, which the `eval` settings do not allow. This happens in a host-native run or in a modified Compose file. | Set `UVICORN_WS_PING_INTERVAL=0` for the server process. |
| The WebSocket handshake fails, and the server closes it with code 1008 and reason `realtime authentication failed`. | `REALTIME_API_KEY` is set, and the Realtime client sent no credential or a different one. | Send `Authorization: Bearer <your-realtime-api-key>` with the value from `.env`. |
| `session.update` fails with `worker_start_timeout` or `backend_unavailable`. | A Hermes worker did not start. | Read the worker logs: `logs/fdh_workers_pipecat/` in the plain runtime, or a copy of the sandbox's `/app/logs` in the OpenShell runtime. Rerun the `up` command with `--build` if the Hermes checkout changed. |
| A code change works on the voice side but not in Hermes, or the source hash in the `FDH gateway` line differs from your checkout. | The Hermes image runs the source that was built into it. | Rerun the `up` command with `--build`. |
| A benchmark run shows other backend prompt hashes than expected, or the Hermes logs name `gateway.yaml`. | The deployment runs without the eval overlay. | Redeploy with `-f docker/docker-compose.fdh-eval.yaml`, and pass it in every Compose command. |
| The client fails with `ssl argument is incompatible with a ws:// URI`, or a `wss://` handshake fails with an SSL error. | The client and the server disagree on TLS. Eval mode serves plain `ws://`. | Use `ws://` for an eval-mode deployment and `wss://` for the standard deployment. |
| The browser does not ask for the microphone. | The page is served over plain HTTP from another machine. | Keep `PIPELINE_TLS=true` (the default) and open the `https://` URL. |
| The download script fails, or `nemo-speech` cannot find its models. | `HF_TOKEN` is missing or invalid, or the weights are in another directory. | Set `HF_TOKEN` in `.env` and rerun the script as your user, or point `NEMO_SPEECH_MODEL_LOC` at the weights. |
| `up -d` fails with `port is already allocated` or `address already in use`. | Another process uses the app port or port `50051`. | Set `PIPELINE_APP_PORT` to a free port for the app. For `50051`, stop the other recipe that runs a speech sidecar. |
| `mv` or `rm` in `logs/` or `logs/openshell/` fails with permission denied. | The containers write logs as `root`. | Run `sudo chown -R "$USER" logs`. |

## Related Topics

- [Frontend Delegation Hermes README](../../src/examples/frontend_delegation_hermes/README.md)
- [Deploy the Frontend/Backend Agent](deploy-frontend-backend-agent.md)
- [Use the OpenAI Realtime-Compatible Gateway](use-realtime-gateway.md)
- [WebSocket Keepalive](tune-pipeline-performance.md#websocket-keepalive)
- [Troubleshooting](../06-troubleshooting.md)
