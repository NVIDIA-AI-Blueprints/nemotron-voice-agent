# Deploy the Frontend Delegation Agent

## Overview

This guide takes you from a clean host to a running and verified deployment of the [Frontend Delegation Agent (FDA) Agent](../../src/examples/frontend_delegation_agent/README.md). A fast frontend large language model (LLM) talks to the user and delegates work to a backend agent. A backend gateway starts one backend worker per session.

The agent has two backends, and you choose one for each deployment:

- `hermes` (default): the Hermes agent, in its own `fda-hermes` service. By default, it runs in an NVIDIA OpenShell sandbox.
- `llm`: a lightweight tool-calling backend built on the OpenAI SDK. It runs inside the voice container, so the deployment needs no Hermes checkout, Hermes image, or OpenShell.

Refer to [Choose a Backend](#choose-a-backend).

Both LLMs run on the NVIDIA Inference Hub at `https://inference-api.nvidia.com/v1`:

- The frontend LLM is `nvidia/nvidia/nemotron-3.5-lightning`.
- The backend LLM is `nvidia/nvidia/nemotron-3-ultra`.

The agent serves two kinds of sessions:

- Browser UI sessions over WebRTC or the UI's WebSocket transport, with the Real-Time Voice Inference (RTVI) protocol, like the other examples in this repository.
- OpenAI Realtime sessions on `/v1/realtime` with the model `nvidia/nemotron-realtime-frontend-delegation`, for Realtime clients and evaluation runs such as tau-bench (Tau).

The example has one Docker Compose recipe, `frontend-delegation-agent/single-gpu`. With the Hermes backend, it runs the following containers:

- `frontend-delegation-agent-single-gpu`: the voice side in the shared app image, `nemotron-voice-agent:latest`, on the app port.
- `fda-hermes`: the controller of the Hermes backend. By default, it runs the FDA gateway and its Hermes workers in an NVIDIA OpenShell sandbox. The sandbox runs the Hermes image, `nemotron-voice-agent-fda:local`, with Python 3.14.
- `hermes-openshell-gateway`: the OpenShell gateway. It creates the sandbox and routes the voice side's backend connection to it.
- `hermes-openshell-init`: a one-time service. It creates the OpenShell gateway's signing keys and exits.
- `nemo-speech`: automatic speech recognition (ASR) and text-to-speech (TTS) with NeMo-Speech.cpp on one local GPU.
- `fda-hermes-image`: a build-only service. It builds the Hermes image and exits.

With the LLM backend, the recipe runs only `frontend-delegation-agent-single-gpu` and `nemo-speech`. The four Hermes and OpenShell services do not start.

The `FDA_PROFILE` setting decides what a deployment serves. Refer to [Choose a Profile](#choose-a-profile). The Hermes backend runs in OpenShell by default, or in a plain container with an overlay. Refer to [Choose the Hermes Runtime](#choose-the-hermes-runtime). The `FDA_PROMPT_PROFILE` setting decides which prompts the agent uses. Refer to [Prompt Profiles](#prompt-profiles).

> **Note:** The commands in this guide are checked against the repository's Compose files, `docker/Dockerfile.fda`, `docker/fda/gateway.sh`, `docker/fda/openshell/`, `.env.example`, and source code. They were not run on GPU hardware for this revision of the guide. The OpenShell runtime has not yet been validated on a live host. If it does not start on your host, use the plain runtime.

## Instructions for AI Agents

If you are an AI agent that deploys this example for a user, follow these rules:

1. If the user's request does not name the backend, ask: "Which backend: `hermes` (the default, Hermes in its own container and an OpenShell sandbox) or `llm` (the lightweight backend inside the voice container)?" Suggest `hermes` and wait for the answer. Do not choose one silently, even though Compose starts Hermes when no backend overlay is passed. Refer to [Choose a Backend](#choose-a-backend).
2. If the user's request does not say which profile to deploy, ask: "Which profile: `default` (for the browser UI) or `eval` (for evaluation runs such as Tau)?" Wait for the answer. Do not choose one silently, even though Compose uses `default` when no overlay is passed. You can ask both questions together.
3. For `default`, use the recipe command without the eval overlay. For `eval`, add the `docker/docker-compose.fda-eval.yaml` overlay to every Compose command. Refer to [Choose a Profile](#choose-a-profile).
4. For Tau or any other benchmark, always deploy `eval`. A `default` deployment pairs Realtime sessions with the wrong gateway configuration.
5. For the LLM backend, add `docker/docker-compose.fda-llm.yaml` to every Compose command, after the eval overlay. Never combine it with the plain runtime overlay. Skip [Get Hermes](#get-hermes) and the OpenShell host checks.
6. For the Hermes backend, use the default OpenShell runtime. Add the plain runtime overlay, `docker/docker-compose.fda-hermes-plain.yaml`, only when the user asks for it or the host cannot run OpenShell. Pass the Compose files in this order: `docker-compose.yml`, then the plain overlay, then the eval overlay. Refer to [Choose the Hermes Runtime](#choose-the-hermes-runtime).
7. After startup and the first session, read the profile and the configuration hash from the app log. For Hermes, also read the gateway configuration and source hash from the Hermes logs. For the LLM backend, read `FDA_BACKEND`, `FDA_BACKEND_LINK`, and `FDA_GATEWAY_CONFIG` from the voice container. Report them to the user. Refer to [Verify the Deployment](#verify-the-deployment).
8. Check only that `NVIDIA_API_KEY` and `REALTIME_API_KEY` are present in `.env`. Do not print their values.
9. If, and only if, the user says that the run is a VoiceBench evaluation, ask two questions and wait for both answers:
    - "Keep the frontend silent? The filler is then logged only, and the backend's answer is the first response. Default: no."
    - "Which prompt profile? Suggested: `general_qa`."

    If the first answer is yes, add `docker/docker-compose.fda-silent-frontend.yaml` after the eval overlay. Pass the Compose files in this order: `docker-compose.yml`, the plain overlay if used, the eval overlay, the LLM backend overlay if used, and then the silent-frontend overlay. Set `FDA_PROMPT_PROFILE` to the chosen profile on the Compose command line, for example `FDA_PROMPT_PROFILE=general_qa docker compose ... up -d --build`. VoiceBench runs use the `eval` profile. Refer to [Run a VoiceBench Evaluation](#run-a-voicebench-evaluation).
10. Do not ask these two questions for Tau, the browser UI, or any other run. Those runs use the default prompt profile of their mode without asking: `generic` for `default` and `customer_service` for `eval`. Never add the silent-frontend overlay for them, and do not set `FDA_PROMPT_PROFILE`. If an existing `.env` sets `FDA_PROMPT_PROFILE`, show the value and ask the user to confirm it or remove it. Refer to [Prompt Profiles](#prompt-profiles).

Do not overwrite other keys in an existing `.env`. For smoke tests, benchmark checks, and logs, load the deploy skill at `.agents/skills/nemotron-voice-agent-fda-deploy/SKILL.md`.

## Choose a Backend

Choose the backend before you start. Every later step, from the prerequisites to the stop command, depends on the choice. Hermes is the default. The following table compares the two backends.

| | `hermes` (default) | `llm` |
| --- | --- | --- |
| What runs the backend | A Hermes `AIAgent` per session, in the `fda-hermes` service | A lightweight OpenAI SDK tool loop per session, inside the voice container |
| Containers besides `nemo-speech` | The voice service, `fda-hermes`, and, in the OpenShell runtime, `hermes-openshell-gateway` and `hermes-openshell-init`, plus the build-only `fda-hermes-image` | The voice service only |
| Extra prerequisites | A clean Hermes checkout in `HERMES_AGENT_DIR` and, for OpenShell, Linux 6.2 or newer with Landlock and a free port `127.0.0.1:8080` | None beyond the shared prerequisites |
| Isolation | By default, an OpenShell sandbox that can call only `inference-api.nvidia.com` and never sees the real `NVIDIA_API_KEY` | None. The backend runs in the voice process, which holds the real `NVIDIA_API_KEY` |
| Images | The shared app image, the Hermes image, and the OpenShell controller image | The shared app image only |
| Backend prompt file | `config/prompts/<profile>/backend/hermes.yaml` | `config/prompts/<profile>/backend/llm.yaml` |
| Use it for | The reference Hermes behavior, evaluation runs that must match earlier Hermes runs, and sandboxed deployments | Quick starts, hosts that cannot build Hermes or run OpenShell, and comparisons with Hermes |
| Compose files | `docker-compose.yml` | `docker-compose.yml` and `docker/docker-compose.fda-llm.yaml` |

The LLM backend reproduces the backend prompt text and the Hermes behaviors that matter for voice, such as steer, redirect, interrupt, and tool-name repair. It is not Hermes, so its results can differ from Hermes runs. Both backends use the same backend model, `BACKEND_LLM_MODEL` on `BACKEND_LLM_BASE_URL` (`nvidia/nvidia/nemotron-3-ultra` on `https://inference-api.nvidia.com/v1`), with `NVIDIA_API_KEY`.

The `FDA_BACKEND` variable selects the backend. It sets `backend.kind` in the gateway configuration and accepts `hermes`, which is the default, and `llm`. The value `emulator` selects a scripted test backend that emulates Hermes without a model and exists for tests only. The LLM backend overlay, `docker/docker-compose.fda-llm.yaml`, sets the following values on the voice service:

- `FDA_BACKEND=llm`.
- `FDA_BACKEND_LINK=in_process`: the gateway and the backend worker run in the voice process, in place of a WebSocket link to `fda-hermes`.
- The backend model and base URL, and the gateway event log and the backend trace directory at the same `logs/` paths that the Hermes runtime uses.

The overlay also removes the four Hermes and OpenShell services from the recipe and the voice service's wait for `fda-hermes`. The voice service loads the gateway configuration from `FDA_GATEWAY_CONFIG`: `gateway.yaml`, or `gateway.eval.yaml` with the eval overlay. Both backends therefore use the same gateway settings, prompt features, and domains. Do not set `FDA_BACKEND` or `FDA_BACKEND_LINK` in `.env`. Choose the backend with the overlay, which sets both values and removes the Hermes services.

The following rules apply to the Compose files:

- Pass the overlays in this order: `docker-compose.yml`, then the eval overlay if you use it, then the LLM backend overlay, then the silent-frontend overlay if you use it.
- Never combine the LLM backend overlay with the plain runtime overlay, `docker/docker-compose.fda-hermes-plain.yaml`. The plain overlay runs Hermes.
- Pass the same files in every Compose command of the deployment, such as `ps`, `logs`, `exec`, and `down`. An `up` command without the LLM backend overlay recreates the deployment with Hermes.

The `config hash` covers the backend link and the gateway configuration file, so it differs between the two backends.

## Prerequisites

Every deployment needs the following:

- Docker Engine 28 or newer and Docker Compose 2.24.4 or newer. The plain runtime overlay and the LLM backend overlay use the Compose `!override` and `!reset` tags, which need Compose 2.24.4.
- An x86_64 Linux host with one NVIDIA GPU for the `nemo-speech` container (GPU 0). The LLMs do not use the local GPU. The Hermes image's lock, `docker/fda/requirements.txt`, is resolved for x86_64 Linux, so DGX Spark and Jetson Thor are not supported.
- NVIDIA driver 580.x or newer and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- Python 3 and either the Hugging Face `hf` CLI or `uvx` for the speech model download script.
- Git, to clone the Hermes agent. The LLM backend does not need it.
- Outbound HTTPS access from the host to the NVIDIA Inference Hub at `https://inference-api.nvidia.com/v1`.
- Free host ports `7860` for the app, or another port set with `PIPELINE_APP_PORT`, and `50051` for `nemo-speech`.
- A clone of this repository. Run every command in this guide from the repository root.

The Hermes backend in the default OpenShell runtime also needs the following on the host:

- Linux kernel 6.2 or newer with Landlock turned on. The sandbox refuses to run without Landlock. Check the kernel with `uname -r`.
- A free host port `127.0.0.1:8080` for the OpenShell gateway.
- No other OpenShell gateway on the host. The port `8080` and the directory `/var/lib/openshell` are shared by the whole host, so only one OpenShell gateway can run per host. This includes a second checkout of this repository.
- Write access for Docker to `/var/lib/openshell`. Docker creates the directory as `root` on the first start. It holds the gateway's signing keys and database.

The plain runtime and the LLM backend do not need these. Refer to [Choose the Hermes Runtime](#choose-the-hermes-runtime) and [Choose a Backend](#choose-a-backend).

## Credentials

The recipe uses the `.env` keys in the following table.

| `.env` key | Required | Purpose |
| --- | --- | --- |
| `NVIDIA_API_KEY` | Yes | An NVIDIA Inference Hub key (`sk-...`). Both LLMs send it. Hermes backend: apart from the optional `FDA_PROMPT_PROFILE`, it is the only `.env` key that reaches the Hermes side, and `fda-hermes` refuses to start without it. With OpenShell, the key goes to the OpenShell provider, never into the sandbox. LLM backend: the voice container sends it for both LLMs. |
| `HF_TOKEN` | Yes | Hugging Face token. The download script reads it to fetch the NeMo-Speech.cpp model weights. |
| `REALTIME_API_KEY` | Recommended | The Bearer credential that OpenAI Realtime clients send. Leave it unset only for unauthenticated local development. |
| `HERMES_AGENT_DIR` | No | Hermes backend only. Path of the Hermes checkout for the image build. The default is `../hermes-agent`. Refer to [Get Hermes](#get-hermes). |
| `FDA_PROMPT_PROFILE` | No | The prompt profile of both the voice side and the backend. Refer to [Prompt Profiles](#prompt-profiles). |

The recipe does not use `NVIDIA_SPEECH_API_KEY` and does not need `docker login nvcr.io`.

## Choose a Profile

The agent has two profiles. The following table compares them.

| Profile | Use it for | What it serves | Delegation profile | Pipecat settings | Gateway configuration | TLS | Compose files |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `default` | The browser UI and development | Browser UI sessions with the `default` settings, and OpenAI Realtime sessions with the `eval` settings | `profiles/pipecat_demo.yaml` | `adapters/pipecat.yaml` | `gateway.yaml` | On (HTTPS and `wss://`) | `docker-compose.yml`, plus the LLM backend overlay for `llm` |
| `eval` | Evaluation runs, such as Tau | OpenAI Realtime sessions only. Browser UI sessions are refused. | `profiles/pipecat_realtime_eval.yaml` | `adapters/pipecat.eval.yaml` | `gateway.eval.yaml` | Off (HTTP and `ws://`) | `docker-compose.yml` and `docker/docker-compose.fda-eval.yaml`, plus the LLM backend overlay for `llm` |

The file paths are relative to `src/examples/frontend_delegation_agent/config/`. [Start the Deployment](#start-the-deployment) lists the command for each backend and profile.

OpenAI Realtime sessions always use the `eval` delegation profile and Pipecat settings, whatever `FDA_PROFILE` says. The gateway configuration is different: one backend gateway serves every session, so the deployment sets it, not the session. Only the eval overlay switches the gateway to `gateway.eval.yaml`, with either backend. A Realtime session on a `default` deployment therefore pairs the eval frontend with `gateway.yaml`. Benchmark runs must use the eval overlay.

The `eval` profile refuses browser UI sessions, so they cannot take resources from an evaluation run. The UI shows no error: the session ends, and the app log shows the error.

`docker-compose.yml` sets every FDA variable of the voice service as a literal value, so `.env` cannot change an evaluation run. Setting `FDA_PROFILE` or another `FDA_*` key in `.env` has no effect on the Compose recipe. Only the overlays change these values. The one exception is `FDA_PROMPT_PROFILE`, which the shell or `.env` can set. Refer to [Prompt Profiles](#prompt-profiles). `FDA_PROFILE` accepts only `default` and `eval`. Any other value fails each session with an error that names both.

## Prompt Profiles

A prompt profile is a named set of the agent's prompts. Each profile is a folder, `src/examples/frontend_delegation_agent/config/prompts/<profile>/`, with the following files:

- `frontend.yaml`: the frontend prompt catalog of the voice side, including `frontend_system`, which makes the delegation decisions.
- `backend/hermes.yaml`: the backend prompt catalog of the Hermes backend.
- `backend/llm.yaml`: the backend prompt catalog of the LLM backend. It is a full copy of the backend prompts, plus the `harness_*` keys. These keys hold the text that Hermes adds to the conversation itself, such as its steer and interrupt messages. The Hermes lines that describe the machine are left out on purpose.
- `voice.yaml`: the voice-side messages about tool calls and formats.

The gateway reads `prompts/<profile>/backend/{kind}.yaml`, where `{kind}` is the backend, so each backend reads its own file. The two backend files are independent. To change the backend prompts of both backends, edit both files.

The agent ships the profiles in the following table.

| Prompt profile | Purpose | Default for | Folder in `config/prompts/` |
| --- | --- | --- | --- |
| `customer_service` | Customer-service prompts for multi-turn conversations with client tools, such as Tau. The evaluation pins cover its prompt hashes. | Eval mode, through the eval overlay | `customer_service/` |
| `generic` | The browser demo prompts. The files currently have the same content as `customer_service/` and are planned to become more generic. | The `default` deployment, including its Realtime sessions | `generic/` |
| `general_qa` | A general question-answering assistant. The frontend delegates every request (`delegate=true`) with the filler "One moment." The backend answers from the LLM with no tools. It puts the answer first, starts a multiple-choice answer with "The answer is X.", follows formatting instructions literally, and refuses harmful requests with "I'm sorry, but I can't help with that." | No mode. Choose it for single-turn question-answering evaluations, such as VoiceBench. | `general_qa/` |

The `FDA_PROMPT_PROFILE` variable names the profile and replaces the default of the mode. It is the one FDA setting that the shell or `.env` can change. Compose passes the same value to the voice service and to `fda-hermes`, so both sides always use the same profile. With the LLM backend, the voice service reads every file of the profile itself. In the OpenShell runtime, the controller passes it into the sandbox. Compose passes `${FDA_PROMPT_PROFILE:-generic}`, and the eval overlay passes `${FDA_PROMPT_PROFILE:-customer_service}`. Realtime sessions follow the deployment: on a `default` deployment, they also use `generic`. Use the eval overlay for `customer_service`.

To choose a profile for one run, set the variable on the Compose command line:

```bash
FDA_PROMPT_PROFILE=general_qa docker compose -f docker-compose.yml -f docker/docker-compose.fda-eval.yaml \
  --profile frontend-delegation-agent/single-gpu up -d --build
```

You can also set `FDA_PROMPT_PROFILE` in `.env`. The value then applies to every later run, including Tau runs, until you remove it. Never set the variable to an empty value. An unknown profile name fails the session or the gateway startup with an error that names the missing prompt file.

The Hermes image bakes in `src/`, so with the Hermes backend, keep `--build` on every `up` after a prompt change or a profile change. With the LLM backend, the voice service reads the prompts from the mounted `src/`, so a restart of the voice service applies a prompt change. Rerunning the `up -d --build` command also works for both backends.

To add a profile, complete the following steps:

1. Copy `config/prompts/customer_service/` to a new folder, `config/prompts/<name>/`.
2. Edit the prompts in `frontend.yaml`, `backend/hermes.yaml`, `backend/llm.yaml`, and `voice.yaml`. Keep every key of the `customer_service` files. A `backend/llm.yaml` file without the `harness_*` keys fails to load.
3. Set `FDA_PROMPT_PROFILE=<name>` and rerun the `up -d --build` command.

The unit tests load every profile in `config/prompts/`, so a broken profile fails `uv run pytest tests/`.

## Choose the Hermes Runtime

This section applies to the Hermes backend only. The LLM backend runs in the voice container and has no separate runtime.

The Hermes backend, that is the FDA gateway and its Hermes workers, can run in one of two runtimes. Both run the same image, `nemotron-voice-agent-fda:local`, with the same `docker/fda/gateway.sh` start script and the same `docker/fda/gateway.env` settings. Agent behavior, turn taking, and evaluation behavior are the same in both. The following table compares them.

| Runtime | When to use it | Backend URL of the voice side | Extra Compose file |
| --- | --- | --- | --- |
| OpenShell (default) | Normal use and evaluation runs | `ws://default--fda-hermes.openshell.localhost:8080/v1/backend` | None |
| Plain | A/B checks against OpenShell, debugging, and hosts that cannot run OpenShell | `ws://fda-hermes:8790/v1/backend` | `docker/docker-compose.fda-hermes-plain.yaml` |

In the OpenShell runtime, the following rules apply to the sandbox:

- The FDA gateway and its workers run as user ID 1500. The code and packages under `/app` are read-only. Only `/app/logs`, `/app/.cache`, `/tmp`, and `/sandbox` are writable. The rules are in `docker/fda/openshell/policy.yaml`.
- The sandbox can call only `inference-api.nvidia.com`, plus any host that you add to `policy.yaml`.
- The sandbox never sees the real `NVIDIA_API_KEY`. Its `NVIDIA_API_KEY` holds a placeholder. OpenShell adds the real key to requests to `inference-api.nvidia.com` only. The provider profile is `docker/fda/openshell/nvidia-fda.yaml`.
- The OpenShell release is pinned in `docker/fda/openshell/VERSION`. Every OpenShell image and the OpenShell command-line interface (CLI) use that release.

Always pass the Compose files in this order: `docker-compose.yml`, then the plain overlay if you use it, then the eval overlay if you use it. The following commands start each Hermes combination:

```bash
# OpenShell runtime (default)
docker compose -f docker-compose.yml \
  --profile frontend-delegation-agent/single-gpu up -d --build

# OpenShell runtime, eval profile
docker compose -f docker-compose.yml -f docker/docker-compose.fda-eval.yaml \
  --profile frontend-delegation-agent/single-gpu up -d --build

# Plain runtime
docker compose -f docker-compose.yml -f docker/docker-compose.fda-hermes-plain.yaml \
  --profile frontend-delegation-agent/single-gpu up -d --build

# Plain runtime, eval profile
docker compose -f docker-compose.yml -f docker/docker-compose.fda-hermes-plain.yaml \
  -f docker/docker-compose.fda-eval.yaml \
  --profile frontend-delegation-agent/single-gpu up -d --build
```

Pass the same files in every later Compose command for the deployment, such as `ps`, `logs`, `exec`, and `down`. The plain overlay does not start `hermes-openshell-init` or `hermes-openshell-gateway`. In the plain runtime, `fda-hermes` is the FDA gateway container itself, on port `8790` of the Compose network, and is not published to the host.

### Known Security Limitations

The OpenShell runtime has the following known limitations. Each one is a deployment or transport setting, not a design limit. All of them can be fixed without changing agent behavior or evaluation results, and the fixes are planned.

- The OpenShell gateway mounts the Docker socket, which gives it the same power as `root` on the host.
- The OpenShell gateway runs without TLS and without user authentication. Compose publishes it only on the host loopback address, `127.0.0.1:8080`.
- The `/v1/backend` endpoint has no authentication, in either runtime.
- OpenShell adds the real `NVIDIA_API_KEY` to any request to `inference-api.nvidia.com`, not only to the model calls that Hermes makes.
- The image is not pinned by digest, and the plain runtime runs as `root`.

The LLM backend has no sandbox at all. It runs in the voice container with the real `NVIDIA_API_KEY` and the voice container's network access.

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
| `HERMES_AGENT_DIR` | `../hermes-agent` | Path of the Hermes checkout for the image build. Both Hermes runtimes use it. The LLM backend does not. |
| `FDA_PROMPT_PROFILE` | `generic`, or `customer_service` with the eval overlay | The prompt profile: `generic`, `customer_service`, `general_qa`, or a profile that you add. It is the only `FDA_*` key that `.env` can change. Never set it to an empty value. Refer to [Prompt Profiles](#prompt-profiles). |
| `FDA_SILENT_FRONTEND` | `false` | `true` keeps the frontend silent on delegated turns for single-turn evaluations, such as VoiceBench. Compose sets `false` as a literal value, and only the silent-frontend overlay sets `true`. Setting it in `.env` has no effect. Never set it to an empty value. Refer to [Run a VoiceBench Evaluation](#run-a-voicebench-evaluation). |

Do not set `UVICORN_WS_PING_INTERVAL` for this recipe. Compose sets it to `0` on the voice service, and Realtime sessions refuse to start when the server pings WebSocket clients.

## Get Hermes

Complete this section for the Hermes backend only. The LLM backend needs no Hermes checkout, so skip to [Start the Deployment](#start-the-deployment).

The Hermes image installs Hermes from a clean checkout of the official [Hermes Agent repository](https://github.com/nousresearch/hermes-agent) at a pinned commit. The header of `docker/fda/requirements.txt` records the commit, `af26acab737db4382815aadcd73c06755471ab52`. The `fda-hermes-image` service passes the checkout to the build as the named build context `hermes`.

The `HERMES_AGENT_DIR` key in `.env` sets the checkout path. A relative path resolves against the repository root, and the default, `../hermes-agent`, is a sibling directory of this repository. Set an absolute path to avoid ambiguity.

The following commands clone Hermes at the pinned commit, record its absolute path in `.env`, and check the checkout:

```bash
HERMES_COMMIT=$(grep -m1 'Hermes commit' docker/fda/requirements.txt | awk '{print $4}')
test -d ../hermes-agent || git clone https://github.com/nousresearch/hermes-agent.git ../hermes-agent
git -C ../hermes-agent checkout "$HERMES_COMMIT"
grep -q '^HERMES_AGENT_DIR=' .env || echo "HERMES_AGENT_DIR=$(realpath ../hermes-agent)" >> .env
HERMES_AGENT_DIR=$(grep -m1 '^HERMES_AGENT_DIR=' .env | cut -d= -f2-)
git -C "$HERMES_AGENT_DIR" status --porcelain
git -C "$HERMES_AGENT_DIR" rev-parse HEAD
```

The `status --porcelain` command must print nothing, and `rev-parse HEAD` must print the pinned commit. Hermes runs only inside the Hermes image, in the OpenShell sandbox or in the plain `fda-hermes` container, so you do not need a Hermes environment on the host.

## Start the Deployment

The following table lists the services of the recipe and the backend that runs each one. The later commands use these names.

| Service | Image | Backend | Role |
| --- | --- | --- | --- |
| `frontend-delegation-agent-single-gpu` | `nemotron-voice-agent:latest` | Both | The voice side: the frontend, speech, and both session kinds. With the LLM backend, it also runs the backend gateway and one LLM backend worker per session. |
| `fda-hermes` | `nemotron-voice-agent-fda-openshell-cli:local` | Hermes | The controller. It creates the OpenShell sandbox `fda-hermes`, which runs the FDA gateway and its Hermes workers from `nemotron-voice-agent-fda:local`. It saves the sandbox logs and deletes the sandbox when it stops. |
| `hermes-openshell-gateway` | `ghcr.io/nvidia/openshell/gateway` (pinned) | Hermes, OpenShell runtime | The OpenShell gateway. It starts the sandbox through the Docker socket and routes the backend connection to it. |
| `hermes-openshell-init` | `ghcr.io/nvidia/openshell/gateway` (pinned) | Hermes, OpenShell runtime | Creates the OpenShell gateway's signing keys in `/var/lib/openshell/tls`, and then exits. It keeps existing keys. |
| `fda-hermes-image` | `nemotron-voice-agent-fda:local` | Hermes | Builds the Hermes image, and then exits. |
| `nemo-speech` | NeMo-Speech.cpp | Both | ASR and TTS on GPU 0. |

With the plain runtime overlay, `fda-hermes` runs `nemotron-voice-agent-fda:local` directly, and the two OpenShell services do not start. With the LLM backend overlay, none of the four Hermes services start.

Complete the following steps:

1. Download the NeMo-Speech.cpp weights once, as your user. Do not use `sudo`. The script reads `HF_TOKEN` from `.env`:

    ```bash
    bash scripts/download-nemo-speech-models.sh
    ```

2. Create the host log directory, so that it belongs to your user:

    ```bash
    mkdir -p logs
    ```

3. Check that Compose accepts the recipe and your `.env`. Pass the files of the backend that you chose. The command prints nothing when the configuration is valid:

    ```bash
    # Hermes backend
    docker compose --profile frontend-delegation-agent/single-gpu config -q

    # LLM backend
    docker compose -f docker-compose.yml -f docker/docker-compose.fda-llm.yaml \
      --profile frontend-delegation-agent/single-gpu config -q
    ```

4. Start the recipe with the `default` profile and the backend that you chose:

    ```bash
    # Hermes backend (default), OpenShell runtime
    docker compose --profile frontend-delegation-agent/single-gpu up -d --build

    # LLM backend
    docker compose -f docker-compose.yml -f docker/docker-compose.fda-llm.yaml \
      --profile frontend-delegation-agent/single-gpu up -d --build
    ```

    For the `eval` profile, follow [Deploy in Eval Mode](#deploy-in-eval-mode) instead. For the Hermes plain runtime, add the plain overlay, as [Choose the Hermes Runtime](#choose-the-hermes-runtime) shows.

With the Hermes backend, the first start builds the images and pulls the `nemo-speech` and OpenShell images, so it can take several minutes. OpenShell publishes no CLI download for its prerelease tags, so `docker/fda/openshell/Dockerfile.cli` builds the OpenShell CLI from source at the pinned commit. The first build compiles Rust code and takes several more minutes. Later builds reuse the Docker build cache.

With the Hermes backend, Compose starts the services in order:

1. `fda-hermes-image` builds the Hermes image and exits.
2. `hermes-openshell-init` creates the signing keys and exits. Then `hermes-openshell-gateway` starts.
3. `fda-hermes` checks the kernel version, waits for the OpenShell gateway, sets up the API key provider, and creates the sandbox. It turns healthy after the FDA gateway in the sandbox answers `GET /health` through the OpenShell gateway. The health check allows 180 seconds for startup.
4. The voice service starts.

In the plain runtime, `fda-hermes` turns healthy after its gateway answers `GET /health` on port `8790`.

With the LLM backend, Compose builds only the shared app image and starts the voice service without waiting for a backend service. The voice service starts the backend gateway in its own process. It does not check a gateway at startup.

The voice service mounts `src/` read-only, but the Hermes runtime runs the source that is built into the Hermes image. With the Hermes backend, keep `--build` on every `up` after a source change, so that both sides run the same code. The voice side and the Hermes image must also use the same backend protocol version. A Hermes image built before the current protocol, version 2, rejects the voice side's messages with `protocol version 2 is not supported (expected 1)`. Rebuild also after a change to `docker/Dockerfile.fda`, `docker/fda/`, the Hermes checkout, or `HERMES_AGENT_DIR`. The LLM backend runs from the mounted `src/`, so a restart of the voice service applies a source change.

## Verify the Deployment

Run the following checks in order. Each step says which backend it applies to. The commands show the files of a `default` deployment. With the LLM backend, add `-f docker-compose.yml -f docker/docker-compose.fda-llm.yaml` to every Compose command, as the LLM commands show.

1. Both backends: check that the services are running and that the voice service becomes healthy.

    ```bash
    # Hermes backend
    docker compose --profile frontend-delegation-agent/single-gpu ps -a

    # LLM backend
    docker compose -f docker-compose.yml -f docker/docker-compose.fda-llm.yaml \
      --profile frontend-delegation-agent/single-gpu ps -a
    ```

    With Hermes, the `fda-hermes-image` and `hermes-openshell-init` services show as exited with code 0. With the LLM backend, the list shows only the voice service and `nemo-speech`, and no `fda-hermes`.

2. Both backends: check the liveness route of the app. Use `http://` when `PIPELINE_TLS=false`:

    ```bash
    curl -fk https://localhost:7860/health
    ```

    The route returns `{"status":"ok"}`. It confirms that the server process is up. It does not check speech, the LLMs, or the backend.

3. Hermes backend: check the FDA gateway health route. In the OpenShell runtime, the check goes through the OpenShell gateway, the same route that the voice side uses:

    ```bash
    docker compose --profile frontend-delegation-agent/single-gpu exec fda-hermes \
      curl -s -H 'Host: default--fda-hermes.openshell.localhost' http://hermes-openshell-gateway:8080/health; echo
    ```

    In the plain runtime, check the gateway inside the `fda-hermes` container instead:

    ```bash
    docker compose -f docker-compose.yml -f docker/docker-compose.fda-hermes-plain.yaml \
      --profile frontend-delegation-agent/single-gpu exec fda-hermes \
      curl -s http://localhost:8790/health; echo
    ```

    The response contains `"ok": true`, the open `"sessions"`, `"max_sessions"` (8), and the number of running Hermes `"workers"`.

    LLM backend: skip this step. The backend gateway runs inside the voice process and has no health route. Step 2 covers the voice process.

4. Hermes backend: check the gateway configuration and the source hash of the Hermes image. In the OpenShell runtime, the `fda-hermes` log shows the controller. It prints the image, the gateway configuration, and the OpenShell CLI version:

    ```bash
    docker compose --profile frontend-delegation-agent/single-gpu logs fda-hermes
    ```

    The line `fda-hermes (OpenShell runtime): image nemotron-voice-agent-fda:local, gateway config gateway.yaml` names the configuration. The FDA gateway's own output is in the sandbox workload container, which OpenShell starts from the Hermes image. The following commands find that container and read the gateway line:

    ```bash
    docker ps --filter ancestor=nemotron-voice-agent-fda:local --format '{{.Names}}'
    docker logs <sandbox-workload-container> 2>&1 | grep 'FDA gateway'
    ```

    In the plain runtime, run `logs fda-hermes | grep 'FDA gateway'` with the plain overlay instead. The line has the form `FDA gateway: config gateway.yaml, source <hash>`. The eval overlay shows `gateway.eval.yaml`. To confirm that the image runs your checkout, compute the same hash on the host and compare:

    ```bash
    find src -type f ! -path '*/__pycache__/*' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1
    ```

    A different hash means that the image is older than your checkout. Rerun the `up` command with `--build`.

    LLM backend: check the backend settings in the voice container instead. The command prints `llm`, `in_process`, and `gateway.yaml`. The eval overlay shows `gateway.eval.yaml`:

    ```bash
    docker compose -f docker-compose.yml -f docker/docker-compose.fda-llm.yaml \
      --profile frontend-delegation-agent/single-gpu exec \
      frontend-delegation-agent-single-gpu printenv FDA_BACKEND FDA_BACKEND_LINK FDA_GATEWAY_CONFIG
    ```

    The voice service runs the mounted `src/`, so it has no image source hash to compare.

5. Both backends: check the effective profile in the voice container. Add the LLM backend overlay for the LLM backend:

    ```bash
    docker compose --profile frontend-delegation-agent/single-gpu exec \
      frontend-delegation-agent-single-gpu printenv FDA_PROFILE
    ```

6. Both backends: start one session. Open the browser UI, as [Use the Browser UI](#use-the-browser-ui) describes, or connect one Realtime client, as [Connect a Realtime Client](#connect-a-realtime-client) shows. The first session of each kind waits for a one-time warm-up: one frontend LLM request and, with Hermes, a check of the gateway. Then read the ready line from the app log. Add the LLM backend overlay for the LLM backend:

    ```bash
    docker compose --profile frontend-delegation-agent/single-gpu logs \
      frontend-delegation-agent-single-gpu | grep -E 'frontend-delegation-agent ready|backend gateway'
    ```

    The ready line has the form `frontend-delegation-agent ready (adapter=pipecat, profile=<profile>)`, followed by the loaded configuration files and the `config hash`. A browser UI session logs `profile=default`. A Realtime session logs `profile=eval`, in both profiles. With Hermes, a `backend gateway` line shows the gateway health. The LLM backend logs no `backend gateway` line.

    With the LLM backend, the first session also starts the gateway event log on the host. Its `session_open` record names the backend in `workers.agent_kind`, which must be `llm`:

    ```bash
    grep -o '"agent_kind": *"[a-z]*"' logs/fda_gateway_events.pipecat.jsonl | tail -n 1
    ```

7. Check the prompt profile. With Hermes, both commands print the same profile, `generic` by default:

    ```bash
    docker compose --profile frontend-delegation-agent/single-gpu exec \
      frontend-delegation-agent-single-gpu printenv FDA_PROMPT_PROFILE
    docker compose --profile frontend-delegation-agent/single-gpu exec fda-hermes printenv FDA_PROMPT_PROFILE
    ```

    With the LLM backend, run only the first command, with the LLM backend overlay. The voice service is the only side.

If a check fails, refer to [Troubleshooting](#troubleshooting).

## Use the Browser UI

The browser UI works with the `default` profile only. Open `https://<host>:7860` in a browser. Replace `<host>` with the host name or IP address, or use `localhost` on the same machine. The server uses a self-signed certificate, so the browser shows a warning that you must accept once. For plain-HTTP browser access, refer to [Browser Access](../06-troubleshooting.md#browser-access).

The recipe pins `EXAMPLE_SELECTION=frontend-delegation-agent`, so the UI shows only this example. When the server exposes several examples, select **Frontend Delegation Agent**. Then start a session and speak. The agent greets you with "Hi! How can I help you today?", unless `ENABLE_WELCOME_MESSAGE=false`.

A browser session runs the demo profile: the demo tools run in the server, and the agent waits 5 seconds before each backend run, so you can ask status questions and steer the work while the backend runs. Turn taking uses the Pipecat defaults.

The UI offers the following controls and views for this example:

- **LLM**: chooses the frontend LLM from the example's catalogs. The backend LLM is always the gateway's, Nemotron 3 Ultra on the NVIDIA Inference Hub. The UI has no control for it.
- **ASR**, **TTS**, and the TTS voice: choose the speech endpoints from the example's catalogs. A voice change during a session applies from the next sentence on.
- **Prompt**: the panel shows the informational `frontend` prompt from `prompts.yaml`. The agent does not apply edits. The frontend prompt is `frontend_system` in `config/prompts/<profile>/frontend.yaml` of the selected [prompt profile](#prompt-profiles).
- **Tools**: the read-only tools panel lists the two demo tools, `get_order` and `cancel_order`. The server runs them.
- **Transcripts and latency**: the UI shows the transcripts of both sides and the user-to-bot latency.

The demo tools know order `5512`, which has shipped, and order `5513`, which is processing. The following table lists what to say and what to expect.

| You say | Expected |
| --- | --- |
| "Can you give me the status of order 5512?" | A filler now, and the answer after about 5 seconds plus the model time |
| While it works: "What's the update so far?" | "Let me check…", then a spoken progress sentence. The running task continues. |
| While it works: "And order 5513 as well" | A filler. The running task is steered and answers both. |
| "Okay sure" | No reply, because the agent treats it as an acknowledgement |
| After an answer: "Please cancel order 5513" | A filler. A new run continues the same backend session. |
| Talk over an answer | The answer stops, and the backend learns what you heard |

Browser UI sessions share one cap of 4 concurrent sessions. That cap is separate from the cap of the Realtime sessions, so browser sessions never take an evaluation slot. Both kinds of sessions still share `nemo-speech`, the gateway, and the LLM endpoints.

## Connect a Realtime Client

Connect a Realtime client to the following URL. Replace `<host>` with the host name or IP address, and use `ws://` when `PIPELINE_TLS=false`:

```text
wss://<host>:7860/v1/realtime?model=nvidia/nemotron-realtime-frontend-delegation
```

When `REALTIME_API_KEY` is set, send it as a Bearer credential in the `Authorization` header. Realtime sessions always use the `eval` settings: the client's `session.update` supplies the tools, which the client runs, and the instructions, which become the backend's policy. The agent does not greet.

For a smoke test, use the repository's Realtime test client. It reads `REALTIME_API_KEY` from `.env` and relays a browser page to the server:

```bash
uv run python realtime_test_client/proxy.py --model nvidia/nemotron-realtime-frontend-delegation
```

Open `http://localhost:8765`, select **Connect**, and speak or send text. For an eval-mode deployment, add `--upstream ws://localhost:7860`. For the client's controls, refer to [Test from a Browser](use-realtime-gateway.md#test-from-a-browser).

Realtime sessions have the following client-facing limitations:

- Turns end on the server voice activity detection (VAD). The agent rejects `input_audio_buffer.commit`.
- The agent ignores the voice that the client requests and uses the catalog voice.
- At most 4 Realtime sessions run at the same time. A fifth session receives an `error` event with code `server_busy`, and the socket closes with code 1013.

For the full event reference, refer to [Use the OpenAI Realtime-Compatible Gateway](use-realtime-gateway.md).

## Deploy in Eval Mode

Eval mode deploys the `eval` profile for evaluation runs, such as a Tau evaluation. It works with both backends and uses the same images and code as the standard deployment. The `docker/docker-compose.fda-eval.yaml` overlay changes the settings in the following table.

| Service | Setting | Eval mode value | Reason |
| --- | --- | --- | --- |
| `frontend-delegation-agent-single-gpu` | `FDA_PROFILE` | `eval` | The container serves OpenAI Realtime sessions only and refuses browser UI sessions. |
| `frontend-delegation-agent-single-gpu` | `PIPELINE_TLS` | `false` | The server serves plain HTTP and `ws://`, so an evaluation client connects without a certificate. |
| `fda-hermes` | `FDA_GATEWAY_CONFIG` | `gateway.eval.yaml` | Hermes backend: the gateway uses the evaluation backend settings, which pin every backend prompt variant and record the backend decision trace. |
| `frontend-delegation-agent-single-gpu` | `FDA_GATEWAY_CONFIG` | `gateway.eval.yaml` | LLM backend: the in-process gateway uses the same evaluation backend settings. With Hermes, the voice service does not read it. |
| Both | `FDA_PROMPT_PROFILE` | `customer_service`, unless the shell or `.env` sets it | Evaluation runs use the customer-service prompts by default. Refer to [Prompt Profiles](#prompt-profiles). |

Pass the overlay after `docker-compose.yml` in every Compose command for the deployment. With the LLM backend, pass the LLM backend overlay after the eval overlay. An `up` command without the eval overlay recreates the containers with the `default` profile and `gateway.yaml`.

The commands in this section use the variable `FDA_FILES` for the Compose files of the backend that you chose. Set it in every shell:

```bash
# Hermes backend, OpenShell runtime
FDA_FILES="-f docker-compose.yml -f docker/docker-compose.fda-eval.yaml"

# Hermes backend, plain runtime
FDA_FILES="-f docker-compose.yml -f docker/docker-compose.fda-hermes-plain.yaml -f docker/docker-compose.fda-eval.yaml"

# LLM backend
FDA_FILES="-f docker-compose.yml -f docker/docker-compose.fda-eval.yaml -f docker/docker-compose.fda-llm.yaml"
```

The eval Pipecat settings, `config/adapters/pipecat.eval.yaml`, measure the end of a user turn in input audio only. An evaluation client can pause its input for seconds, so a wall-clock timer would end turns mid-utterance. They change three keys compared to `adapters/pipecat.yaml`, which keeps the Pipecat defaults:

| Key | Default | Eval | Reason |
| --- | --- | --- | --- |
| `vad.stop_secs` | `0.2` | `0.8` | Silero counts 800 ms of silence in the audio. |
| `vad.audio_idle_timeout` | `1.0` (wall clock) | `0` (off) | A wall-clock timer would end the turn when the client pauses its input. |
| `turn.user_speech_timeout` | `0.6` (wall clock) | `0` | No wall-clock wait follows the VAD stop. |

To deploy in eval mode, complete the following steps:

1. Complete [Configure the Environment](#configure-the-environment) and steps 1 and 2 of [Start the Deployment](#start-the-deployment). For the Hermes backend, also complete [Get Hermes](#get-hermes).

2. Start the recipe with the files of your backend:

    ```bash
    docker compose $FDA_FILES --profile frontend-delegation-agent/single-gpu up -d --build
    ```

3. Download the NLTK `punkt_tab` data in the voice container. Pipecat loads it on first use for sentence splitting. On a new container, the first sessions otherwise wait for the download. The command prints `True`:

    ```bash
    docker compose $FDA_FILES --profile frontend-delegation-agent/single-gpu \
      exec frontend-delegation-agent-single-gpu \
      .venv/bin/python -c "import nltk; print(nltk.download('punkt_tab', quiet=True))"
    ```

4. Check the liveness route over plain HTTP:

    ```bash
    curl -f http://localhost:7860/health
    ```

5. Check the eval settings. The first command prints `eval`, `false`, and `customer_service`, for both backends:

    ```bash
    docker compose $FDA_FILES --profile frontend-delegation-agent/single-gpu \
      exec frontend-delegation-agent-single-gpu printenv FDA_PROFILE PIPELINE_TLS FDA_PROMPT_PROFILE
    ```

    For the Hermes backend, the following command prints a controller line that names `gateway.eval.yaml`. In the plain runtime, it prints the `FDA gateway` line:

    ```bash
    docker compose $FDA_FILES --profile frontend-delegation-agent/single-gpu \
      logs fda-hermes | grep -E 'gateway config|FDA gateway'
    ```

    For the LLM backend, the following command prints `llm`, `in_process`, and `gateway.eval.yaml`:

    ```bash
    docker compose $FDA_FILES --profile frontend-delegation-agent/single-gpu \
      exec frontend-delegation-agent-single-gpu printenv FDA_BACKEND FDA_BACKEND_LINK FDA_GATEWAY_CONFIG
    ```

6. Connect one client to `ws://localhost:7860/v1/realtime?model=nvidia/nemotron-realtime-frontend-delegation`, as [Connect a Realtime Client](#connect-a-realtime-client) shows. Then check that the voice event log grew on the host:

    ```bash
    tail -n 1 logs/fda_voice_events.pipecat.jsonl
    ```

To switch back to the standard deployment, run the `up -d --build` command of your backend from [Start the Deployment](#start-the-deployment), without the eval overlay.

## Run a Tau Evaluation

The Tau evaluation harness is not part of this repository. Use the following steps to point it at this deployment:

1. Choose the backend, and deploy in eval mode, as [Deploy in Eval Mode](#deploy-in-eval-mode) describes. Every Tau run must use the eval overlay, because only the overlay switches the gateway to `gateway.eval.yaml`. Do not set `FDA_PROMPT_PROFILE` in the shell or in `.env`, so that the run uses `customer_service`.
2. Configure the harness with the WebSocket URL `ws://<host>:7860/v1/realtime`, the model `nvidia/nemotron-realtime-frontend-delegation`, and your `REALTIME_API_KEY` as the Bearer credential. The former `pine-*` model names and the former ports `8775` and `8776` no longer reach FDA.
3. Keep the session count within the cap of 4 Realtime sessions.
4. Record the backend with the results. For Hermes, also record the runtime, and the gateway configuration and the source hash from the Hermes logs. For the LLM backend, record `FDA_BACKEND`, `FDA_BACKEND_LINK`, and `FDA_GATEWAY_CONFIG` from the voice container, and the commit of your checkout. For both, record the profile and `config hash` from the app log.

Do not compare LLM backend runs with Hermes runs as if they were the same agent. The LLM backend reproduces the Hermes prompts and voice behaviors, but its own model calls can differ.

The `config hash` includes the backend link, the gateway configuration file, and the backend URL. The URL is `ws://default--fda-hermes.openshell.localhost:8080/v1/backend` in the OpenShell runtime and `ws://fda-hermes:8790/v1/backend` in the plain runtime. The hash therefore differs between the two backends, between the two Hermes runtimes, and from runs on the former FDA containers. To compare runs, use the prompt hashes and `session_tools_sha256`. Refer to [Deployed Fingerprint](../../src/examples/frontend_delegation_agent/README.md#deployed-fingerprint).

## Run a VoiceBench Evaluation

VoiceBench in `voice-agent-evaluation` connects through OpenAI Realtime with the server VAD and never sends `response.create`. It ends each case on the first `response.done` and scores only the audio transcript of that response. This agent speaks each filler and each proactive status line as its own response, so on a delegated turn VoiceBench would score the filler instead of the backend's answer.

The silent-frontend overlay, `docker/docker-compose.fda-silent-frontend.yaml`, sets `FDA_SILENT_FRONTEND=true`, which turns on the `delegation.silent_frontend` switch. With the switch on, the agent behaves as follows:

- The filler of a delegated turn is never spoken. The voice event log records it as a `filler_silenced` event with the reason `silent_frontend`.
- The proactive status line is never spoken. The voice event log records a `proactive_status_silenced` event once per backend run.
- Direct answers, that is turns that the frontend does not delegate, are still spoken.
- A turn that the frontend LLM leaves silent (`delegate=false` with an empty filler) but that holds a real request is delegated anyway, as a task with an empty filler. The voice event log records a `delegation_guard` event with `guard=silent_delegated`. Empty turns, backchannels from `backchannel_words`, such as "okay" or "thanks", pauses, such as "hold on", liveness pings, and every turn while the backend is working stay silent.

The switch is off by default, and without the overlay none of these rules apply.

A VoiceBench run makes two choices:

- Silent frontend: add the silent-frontend overlay, or leave it out. The default is to leave it out.
- Prompt profile: set `FDA_PROMPT_PROFILE`. The suggested profile is `general_qa`, whose frontend delegates every request. Without the variable, eval mode uses `customer_service`. Refer to [Prompt Profiles](#prompt-profiles).

Use the silent-frontend overlay for single-turn evaluations, such as VoiceBench, only. Tau runs need the spoken filler and status lines.

To run a VoiceBench evaluation with a silent frontend and the `general_qa` profile, complete the following steps:

1. Choose the backend, complete steps 1 to 4 of [Deploy in Eval Mode](#deploy-in-eval-mode), and start the recipe with the silent-frontend overlay last. In the Hermes plain runtime, pass the plain overlay before the eval overlay. For a spoken filler, leave out the silent-frontend overlay in this and every later command:

    ```bash
    # Hermes backend, OpenShell runtime
    FDA_PROMPT_PROFILE=general_qa docker compose -f docker-compose.yml -f docker/docker-compose.fda-eval.yaml \
      -f docker/docker-compose.fda-silent-frontend.yaml \
      --profile frontend-delegation-agent/single-gpu up -d --build

    # LLM backend
    FDA_PROMPT_PROFILE=general_qa docker compose -f docker-compose.yml -f docker/docker-compose.fda-eval.yaml \
      -f docker/docker-compose.fda-llm.yaml -f docker/docker-compose.fda-silent-frontend.yaml \
      --profile frontend-delegation-agent/single-gpu up -d --build
    ```

    Pass the same `-f` files in every Compose command for this deployment, including `exec`, `logs`, and `down`. Commands that recreate the containers, such as `up`, also need the same `FDA_PROMPT_PROFILE` value.

2. Check the switch and the prompt profile in the voice container. The command shows the Hermes files. For the LLM backend, add the LLM backend overlay before the silent-frontend overlay. The command prints `true` and `general_qa`:

    ```bash
    docker compose -f docker-compose.yml -f docker/docker-compose.fda-eval.yaml \
      -f docker/docker-compose.fda-silent-frontend.yaml \
      --profile frontend-delegation-agent/single-gpu exec frontend-delegation-agent-single-gpu \
      printenv FDA_SILENT_FRONTEND FDA_PROMPT_PROFILE
    ```

3. Connect one session. In `logs/fda_voice_events.pipecat.jsonl`, the settings of the `fda_session_start` record carry `silent_frontend: true`, and each delegated turn has a `filler_silenced` record.

4. Run the harness with the profile `nemotron-fdh-local` and the run configuration `configs/runs/voicebench-fdh-realtime-smoke.yaml` in `voice-agent-evaluation`.

5. Record the backend, the prompt profile, and the `config hash` from the app log with the results. The switch is part of the effective settings only when it is on, so the `config hash` of a run with a silent frontend differs from that of a normal eval run. The prompt paths are part of the effective settings too, so another prompt profile also changes the hash. With the switch off, the hash is the same as before the switch existed.

To switch back to a Tau or other evaluation deployment, rerun the `up -d --build` command of your backend from [Deploy in Eval Mode](#deploy-in-eval-mode) without the silent-frontend overlay and without `FDA_PROMPT_PROFILE`. Compose then recreates the containers with `FDA_SILENT_FRONTEND=false` and the `customer_service` prompt profile.

## Read the Logs

The containers write the logs in the following table. The `logs/` paths are on the host, in the repository's `logs/` directory.

| Log | Content |
| --- | --- |
| `docker compose --profile frontend-delegation-agent/single-gpu logs frontend-delegation-agent-single-gpu` | The app log: the ready line, the Pipecat settings, and per-session errors. With the LLM backend, also the output of the backend gateway and its workers. |
| `docker compose --profile frontend-delegation-agent/single-gpu logs fda-hermes` | Hermes backend only. OpenShell runtime: the controller log, with the image, the gateway configuration, the OpenShell CLI version, and setup errors. Plain runtime: the gateway log, with its configuration, the source hash, and gateway errors. |
| `logs/fda_voice_events.pipecat.jsonl` | The voice event log of both session kinds. |
| `logs/fda_gateway_events.pipecat.jsonl` | Hermes plain runtime and LLM backend: the gateway event log. With the LLM backend, the voice container writes it. |
| `logs/fda_workers_pipecat/` | Hermes plain runtime: one log per Hermes worker. LLM backend: written by the voice container. With `gateway.eval.yaml`, both hold one `<session_id>.trace.jsonl` backend decision trace per session. |
| `logs/openshell/<time>/` | OpenShell runtime: the gateway event log, the worker logs, and the backend decision traces, saved from the sandbox's `/app/logs` when `fda-hermes` stops. |

In the OpenShell runtime, the gateway and worker log files stay inside the sandbox, in `/app/logs`, while it runs. The following commands read the other OpenShell logs:

- The FDA gateway and worker output, and Landlock denials: `docker logs <sandbox-workload-container>`. Find the container name with `docker ps --filter ancestor=nemotron-voice-agent-fda:local --format '{{.Names}}'`.
- The policy decisions, that is the allowed and denied outbound calls of the sandbox:

    ```bash
    docker compose --profile frontend-delegation-agent/single-gpu exec fda-hermes \
      openshell logs fda-hermes
    ```

- A copy of `/app/logs` during a run. The copy lands in `logs/openshell/now/` on the host:

    ```bash
    docker compose --profile frontend-delegation-agent/single-gpu exec fda-hermes \
      openshell sandbox download fda-hermes /app/logs /out/now
    ```

The LLM backend has no `fda-hermes` service and no sandbox, so the `fda-hermes` and OpenShell commands do not apply to it. Add the overlay files of your deployment to these commands. The containers write the log files as `root`. The repository ignores the `logs/` directory.

## Stop and Clean Up

Stop the recipe with the same Compose profile and files that started it. The following commands stop each backend in the `default` profile:

```bash
# Hermes backend
docker compose --profile frontend-delegation-agent/single-gpu down

# LLM backend
docker compose -f docker-compose.yml -f docker/docker-compose.fda-llm.yaml \
  --profile frontend-delegation-agent/single-gpu down
```

For an eval-mode deployment, pass the eval overlay as well, or reuse `FDA_FILES` from [Deploy in Eval Mode](#deploy-in-eval-mode):

```bash
docker compose $FDA_FILES --profile frontend-delegation-agent/single-gpu down
```

In the OpenShell runtime, `fda-hermes` saves the sandbox's `/app/logs` to `logs/openshell/<time>/` and then deletes the sandbox. The stop can take up to 60 seconds. If the controller crashed and left a sandbox behind, the next start saves its logs and deletes it before it creates a new one.

If you started the deployment with the silent-frontend overlay, pass it last as well.

In the plain runtime, the gateway closes open sessions one after another, so the stop can take up to 75 seconds. Pass the plain overlay to `down` as well.

With the LLM backend, `down` stops the voice service and `nemo-speech`. The backend stops with the voice service, so no backend container or sandbox remains.

The command keeps `.env`, the built images, the logs in `logs/`, the speech model weights in `models/nemo-speech`, and, for the OpenShell runtime, the OpenShell gateway's keys and database in `/var/lib/openshell`.

## Troubleshooting

The following table lists common failures for this recipe. For general startup issues, refer to [Troubleshooting](../06-troubleshooting.md).

| Symptom | Cause | Fix |
| --- | --- | --- |
| `up` fails while it builds `fda-hermes-image`, and the error names the `hermes` build context. | The Hermes checkout is missing at `HERMES_AGENT_DIR`. | Complete [Get Hermes](#get-hermes), and set an absolute `HERMES_AGENT_DIR` in `.env`. |
| The image build fails in the `uv pip check` step. | The Hermes checkout does not match the lock in `docker/fda/requirements.txt`. | Check out the commit from the lock header in a clean checkout, or regenerate the lock as its header describes. |
| `fda-hermes` exits at start, its log says `NVIDIA_API_KEY is not set`, and the voice service never starts. | `.env` has no `NVIDIA_API_KEY`. | Set `NVIDIA_API_KEY` to an NVIDIA Inference Hub key, and then rerun the `up` command. |
| `fda-hermes` exits, and its log says `OpenShell needs Linux 6.2+ (Landlock ABI 3)`. | The host kernel is older than 6.2. | Run the plain runtime, or use a host with Linux 6.2 or newer. |
| The sandbox fails to start, and the workload container log shows a Landlock error. | Landlock is turned off in the host kernel. The policy does not allow the sandbox to run without it. | Turn on Landlock in the kernel boot settings, or run the plain runtime. |
| `fda-hermes` exits, and its log says `OpenShell gateway at http://hermes-openshell-gateway:8080 not connected after 120 s`. | The OpenShell gateway did not start. | Read `docker compose --profile frontend-delegation-agent/single-gpu logs hermes-openshell-gateway hermes-openshell-init`. Check the Docker socket and `/var/lib/openshell`. |
| `up -d` fails with `port is already allocated` for port `8080`. | Another process, or another OpenShell gateway, uses `127.0.0.1:8080`. | Stop the other process or OpenShell gateway. Only one OpenShell gateway can run per host. |
| `up` fails with an unknown `!override` or `!reset` tag. | Docker Compose is older than 2.24.4. | Update Docker Compose to 2.24.4 or newer. |
| `fda-hermes` never turns healthy in the OpenShell runtime. | The sandbox did not start, or the FDA gateway in it failed. | Read the `fda-hermes` log and the sandbox workload container log, as [Read the Logs](#read-the-logs) describes. Then rerun the `up` command. |
| The FDA gateway or a worker fails on a call to a host other than `inference-api.nvidia.com`. | The sandbox policy blocks the host. | Read the denied calls with `openshell logs fda-hermes`, as [Read the Logs](#read-the-logs) shows. Add the host under `network_policies` in `docker/fda/openshell/policy.yaml`, or compare with the plain runtime. |
| A browser UI session ends at once or never starts, and the app log shows `runs with FDA_PROFILE=eval`. | The deployment runs the `eval` profile, which refuses browser UI sessions. | Start the recipe without the eval overlay. |
| A browser UI session ends at once, and the app log shows `browser session refused`. | Either 4 browser UI sessions are already open, or the warm-up failed. | Close idle browser sessions. For a warm-up failure, read the `startup failed` line in the app log, and then start a new session. |
| A Realtime client receives an `error` event with code `server_busy`, and the socket closes with code 1013. | The message `server at capacity (server.max_sessions)` means that 4 Realtime sessions are open. The message `server is starting` means that the warm-up failed. | Close idle sessions, or wait and reconnect. For a warm-up failure, read the `startup failed` line in the app log. The next session retries the warm-up. |
| The app log shows `server.max_sessions (N) exceeds the gateway's max_sessions`. | The voice side allows more sessions than the gateway. | Use the Compose values, which set 4 sessions per session kind and 8 gateway sessions. |
| The app log shows `backend gateway not reachable`, and sessions fail. | Hermes backend: the `fda-hermes` gateway is not running or not healthy. | Run `docker compose --profile frontend-delegation-agent/single-gpu logs fda-hermes`, fix the cause, and rerun the `up` command. |
| A Realtime client receives an `error` event with code `server_error`, the socket closes with code 1011, and the error asks for `UVICORN_WS_PING_INTERVAL=0`. | The server pings WebSocket clients, which the `eval` settings do not allow. This happens in a host-native run or in a modified Compose file. | Set `UVICORN_WS_PING_INTERVAL=0` for the server process. |
| The WebSocket handshake fails, and the server closes it with code 1008 and reason `realtime authentication failed`. | `REALTIME_API_KEY` is set, and the Realtime client sent no credential or a different one. | Send `Authorization: Bearer <your-realtime-api-key>` with the value from `.env`. |
| `session.update` fails with `worker_start_timeout` or `backend_unavailable`. | A backend worker did not start. | Hermes backend: read the worker logs, `logs/fda_workers_pipecat/` in the plain runtime, or a copy of the sandbox's `/app/logs` in the OpenShell runtime. Rerun the `up` command with `--build` if the Hermes checkout changed. LLM backend: read the app log. |
| The Hermes gateway or the voice side reports `protocol version 2 is not supported (expected 1)`, or the reverse. | The Hermes image was built from older source than the voice side runs. | Rerun the `up` command with `--build`. |
| `fda-hermes` or the OpenShell services start although you chose the LLM backend, or `printenv FDA_BACKEND` prints nothing. | The `up` command ran without `docker/docker-compose.fda-llm.yaml`. | Rerun the `up` command with the LLM backend overlay, and pass it in every Compose command. |
| A session fails with `backend.kind: hermes cannot run in the voice process (link: in_process)`. | The voice service runs with `FDA_BACKEND_LINK=in_process` but without `FDA_BACKEND=llm`, for example in a modified Compose file. | Use `docker/docker-compose.fda-llm.yaml` unchanged. It sets both values. |
| The gateway fails with `workers.agent_kind was replaced by backend.kind`, `workers.mode: in_process_fake was renamed to in_process`, or `hermes.<key> moved to backend.<key>`. | A modified gateway configuration still uses the former key names. | Rename the keys as the error says. `FDH_AGENT_KIND` is now `FDA_BACKEND`, and the shared backend settings, such as `model`, `prompts`, and `backend_trace`, are in the `backend:` section. Refer to [Backends](../../src/examples/frontend_delegation_agent/README.md#backends). |
| A session fails with an error about `unknown backend kind`. | `FDA_BACKEND` or `backend.kind` holds a value other than `hermes`, `llm`, or `emulator`. | Use the Compose files unchanged. Choose the backend with the overlay. |
| A code change works on the voice side but not in Hermes, or the source hash in the `FDA gateway` line differs from your checkout. | The Hermes image runs the source that was built into it. | Rerun the `up` command with `--build`. |
| A benchmark run shows other backend prompt hashes than expected, or the Hermes logs name `gateway.yaml`. | The deployment runs without the eval overlay, or `FDA_PROMPT_PROFILE` is set in the shell or in `.env`. | Redeploy with `-f docker/docker-compose.fda-eval.yaml`, and pass it in every Compose command. Remove `FDA_PROMPT_PROFILE` for a `customer_service` run. |
| A session, or the FDA gateway at startup, fails with an error that names a missing prompt file under `config/prompts/`. | `FDA_PROMPT_PROFILE` names a prompt profile that does not exist, or it is set to an empty value. A profile that you added can also miss `backend/llm.yaml` or `backend/hermes.yaml`. | Set `FDA_PROMPT_PROFILE` to an existing profile name, such as `generic`, `customer_service`, or `general_qa`, or remove it from the shell and `.env`. Then rerun the `up -d --build` command. Refer to [Prompt Profiles](#prompt-profiles). |
| The client fails with `ssl argument is incompatible with a ws:// URI`, or a `wss://` handshake fails with an SSL error. | The client and the server disagree on TLS. Eval mode serves plain `ws://`. | Use `ws://` for an eval-mode deployment and `wss://` for the standard deployment. |
| The browser does not ask for the microphone. | The page is served over plain HTTP from another machine. | Keep `PIPELINE_TLS=true` (the default) and open the `https://` URL. |
| The download script fails, or `nemo-speech` cannot find its models. | `HF_TOKEN` is missing or invalid, or the weights are in another directory. | Set `HF_TOKEN` in `.env` and rerun the script as your user, or point `NEMO_SPEECH_MODEL_LOC` at the weights. |
| `up -d` fails with `port is already allocated` or `address already in use`. | Another process uses the app port or port `50051`. | Set `PIPELINE_APP_PORT` to a free port for the app. For `50051`, stop the other recipe that runs a speech sidecar. |
| `mv` or `rm` in `logs/` or `logs/openshell/` fails with permission denied. | The containers write logs as `root`. | Run `sudo chown -R "$USER" logs`. |

## Related Topics

- [Frontend Delegation Agent README](../../src/examples/frontend_delegation_agent/README.md)
- [Deploy the Frontend/Backend Agent](deploy-frontend-backend-agent.md)
- [Use the OpenAI Realtime-Compatible Gateway](use-realtime-gateway.md)
- [WebSocket Keepalive](tune-pipeline-performance.md#websocket-keepalive)
- [Troubleshooting](../06-troubleshooting.md)
