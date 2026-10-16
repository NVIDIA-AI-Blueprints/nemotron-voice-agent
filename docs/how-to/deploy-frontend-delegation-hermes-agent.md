# Deploy the Frontend Delegation Hermes Agent

## Overview

This guide takes you from a clean host to a running and verified deployment of the [Frontend Delegation Hermes (FDH) example](../../src/examples/frontend_delegation_hermes/README.md). A fast frontend large language model (LLM) talks to the user and delegates work to a Hermes agent backend. A backend gateway inside the container starts one Hermes worker process per session.

Both LLMs run on the NVIDIA Inference Hub at `https://inference-api.nvidia.com/v1`:

- The frontend LLM is `nvidia/nvidia/nemotron-3.5-lightning`.
- The backend LLM is `nvidia/nvidia/nemotron-3-ultra`.

Automatic speech recognition (ASR) and text-to-speech (TTS) run in the `nemo-speech` NeMo-Speech.cpp container on one local GPU. The example reads its speech endpoints from the single-GPU section of the Frontend/Backend Verdict Agent catalog, `src/examples/frontend_backend_verdict/services.local.yaml`.

The example has two adapters, which run as two services of one image. The following table compares them.

| Adapter | Service | Host port | Use |
| --- | --- | --- | --- |
| Pipecat (default) | `fdh-voice` | `8776` | The browser demo, or an evaluation over the OpenAI Realtime protocol with Pipecat turn taking |
| τ³ | `fdh-tau`, Compose profile `tau` | `8775` | τ³ benchmarking only |

The example is not in `examples_registry.yaml` and has no recipe in the root `docker-compose.yml`. It has its own image, `docker/Dockerfile.fdh`, and its own Compose file, `docker/docker-compose.frontend-delegation-hermes.yaml`.

> **Note:** The commands in this guide are checked against the repository's Compose files, `Dockerfile.fdh`, the container entrypoint, `.env.example`, and source code. They were not run on GPU hardware for this revision of the guide.

## Instructions for AI Agents

If you are an AI agent that deploys this example for a user, follow these rules:

1. If the request does not name an adapter, ask: "Which adapter: Pipecat (the default) or τ³?" Wait for the answer.
2. For the Pipecat adapter, ask "Demo or eval?" unless the request already says. Use [Run the Browser Demo](#run-the-browser-demo) or [Run the Eval Mode](#run-the-eval-mode).
3. For the τ³ adapter, use [Run the τ³ Adapter](#run-the-τ³-adapter), and ask nothing more.
4. Always name the service in `up` commands, and pass `--profile tau` to every `down` command.
5. Do not overwrite other keys in an existing `.env`.

For smoke tests, benchmark procedures, and logs, load the deploy skill at `.agents/skills/nemotron-voice-agent-fdh-deploy/SKILL.md`.

## Prerequisites

The deployment needs the following:

- One NVIDIA GPU for the `nemo-speech` container (GPU 0), with the NVIDIA Container Toolkit. The LLMs do not use the local GPU.
- Docker Compose v2 with support for `additional_contexts` in `build`.
- Outbound HTTPS access from the host to the NVIDIA Inference Hub at `https://inference-api.nvidia.com/v1`.
- `NVIDIA_API_KEY` in `.env`, set to an NVIDIA Inference Hub key. Both LLMs send it.
- `HF_TOKEN` in `.env` for the speech model download script.
- Git, to clone the Hermes agent.
- Free host ports `50051` for `nemo-speech`, `8776` for `fdh-voice`, and `8775` for `fdh-tau`.
- A clone of this repository. Run every command in this guide from the repository root.

Create `.env` from the template only when it does not exist. This command preserves an existing `.env`:

```bash
test -f .env || cp .env.example .env
```

Then set `NVIDIA_API_KEY` and `HF_TOKEN` in `.env`.

## Get Hermes

The image installs Hermes from a clean checkout of the official [Hermes Agent repository](https://github.com/nousresearch/hermes-agent) at a pinned commit. The header of `docker/fdh/requirements.txt` records the commit, `af26acab737db4382815aadcd73c06755471ab52`. Compose passes the checkout to the build as the named build context `hermes`.

The `HERMES_AGENT_DIR` key in `.env` sets the checkout path. A relative path resolves against `docker/`, and the default places the checkout in `hermes-agent`, a sibling directory of this repository. Set an absolute path to avoid ambiguity.

The following commands clone Hermes at the pinned commit, record its absolute path in `.env`, and check the checkout:

```bash
HERMES_COMMIT=$(grep -m1 'Hermes commit' docker/fdh/requirements.txt | awk '{print $4}')
test -d ../hermes-agent || git clone https://github.com/nousresearch/hermes-agent.git ../hermes-agent
git -C ../hermes-agent checkout "$HERMES_COMMIT"
grep -q '^HERMES_AGENT_DIR=' .env || echo "HERMES_AGENT_DIR=$(realpath ../hermes-agent)" >> .env
set -a; . ./.env; set +a
git -C "$HERMES_AGENT_DIR" status --porcelain
git -C "$HERMES_AGENT_DIR" rev-parse HEAD
```

The `status --porcelain` command must print nothing, and `rev-parse HEAD` must print the pinned commit. Hermes always runs inside the container, so you do not need a Hermes environment on the host.

## Start Speech

Both adapters use the `nemo-speech` container on port `50051` for ASR and TTS. Download the NeMo-Speech.cpp weights once, as your user. Do not use `sudo`. The script reads `HF_TOKEN` from `.env` and fetches only what is missing:

```bash
bash scripts/download-nemo-speech-models.sh
```

Start `nemo-speech` through the Frontend/Backend Verdict Agent single-GPU profile of the root project. Name the service, so that only `nemo-speech` starts:

```bash
docker compose --profile frontend-backend-verdict-agent/single-gpu up -d nemo-speech
```

This command also creates the `nemotron-voice-agent_default` network. The FDH Compose file declares that network as external, so start `nemo-speech` before any FDH service.

## Build the Image

Both services use one image, `nemotron-voice-agent-fdh:latest`, built from `docker/Dockerfile.fdh` with Python 3.14. The following command builds it once for both services:

```bash
docker compose --env-file .env -f docker/docker-compose.frontend-delegation-hermes.yaml --profile tau build fdh-tau
```

The build runs import checks for both adapters and the gateway, imports Hermes, and checks the dependency set with `uv pip check`. The `up` commands in this guide also pass `--build`, so a separate build step is optional on a fresh host.

Compose mounts `src/` read-only into the container, so code and prompt changes need only a restart. Rebuild the image after a change to any of the following:

- `docker/fdh/requirements.txt` or `docker/fdh/overrides.txt`
- `docker/Dockerfile.fdh`
- `docker/fdh/entrypoint.sh`
- The Hermes checkout or `HERMES_AGENT_DIR`

## Run the Browser Demo

The demo runs the Pipecat adapter with the `pipecat_demo.yaml` profile, the `gateway.yaml` gateway config, and the `adapters/pipecat.yaml` settings, which keep the Pipecat defaults. It greets you, runs the demo tools in the server, and serves HTTPS with a self-signed certificate. It waits `FDH_PIPECAT_BACKEND_DELAY_S` seconds (default `5`) before each backend run, so you can ask status questions and steer the work while the backend runs. A value of `0` turns off the delay.

1. Start the demo:

    ```bash
    docker compose --env-file .env -f docker/docker-compose.frontend-delegation-hermes.yaml up -d --build fdh-voice
    ```

2. After about 30 seconds, check the health route:

    ```bash
    curl -sk https://localhost:8776/health; echo
    ```

    The response contains `"status": "ok"`, `"prototype": "frontend-delegation-hermes"`, and `"adapter": "pipecat"`. Its `"pipecat"` object names `adapters/pipecat.yaml` in `"settings_file"`.

3. Open `https://<host-ip>:8776/` in a browser. Replace `<host-ip>` with the IP address of the host, or use `localhost` on the same machine. Accept the self-signed certificate, and allow the microphone. Use the IP address rather than a corporate host name, which HSTS can force through a proxy. From a laptop, you can instead tunnel with `ssh -N -L 8776:localhost:8776 <user>@<host>` and open `https://localhost:8776/`.

The demo tools know order `5512`, which has shipped, and order `5513`, which is processing. The following table lists what to say and what to expect.

| You say | Expected |
| --- | --- |
| "Can you give me the status of order 5512?" | A filler now, and the answer after about 5 seconds plus the model time |
| While it works: "What's the update so far?" | "Let me check…", then a spoken progress sentence. The running task continues. |
| While it works: "And order 5513 as well" | A filler. The running task is steered and answers both. |
| "Okay sure" | No reply, because the agent treats it as an acknowledgement |
| After an answer: "Please cancel order 5513" | A filler. A new run continues the same Hermes session. |
| Talk over an answer | The answer stops, and the backend learns what you heard |

The `fdh-voice` service has `restart: "no"`, so it does not restart by itself after a crash or a reboot.

## Run the Eval Mode

The eval mode lets an OpenAI Realtime client, such as a τ² client, evaluate the Pipecat adapter. Set the following four variables on the `up` command:

| Variable | Eval value | Effect |
| --- | --- | --- |
| `FDH_PIPECAT_PROFILE` | `pipecat_realtime_eval.yaml` | The client executes the tools, the client's `session.update` instructions become the policy, and the agent does not greet. |
| `FDH_PIPECAT_GATEWAY_CONFIG` | `gateway.eval.yaml` | The gateway uses the evaluation backend settings. |
| `FDH_PIPECAT_SETTINGS` | `adapters/pipecat.eval.yaml` | The end of a user turn is measured in input audio only. |
| `FDH_PIPECAT_TLS` | `false` | The server serves plain HTTP and `ws://`. |

1. Start the eval mode:

    ```bash
    FDH_PIPECAT_PROFILE=pipecat_realtime_eval.yaml FDH_PIPECAT_GATEWAY_CONFIG=gateway.eval.yaml \
    FDH_PIPECAT_SETTINGS=adapters/pipecat.eval.yaml FDH_PIPECAT_TLS=false \
      docker compose --env-file .env -f docker/docker-compose.frontend-delegation-hermes.yaml up -d --build fdh-voice
    ```

2. After about 30 seconds, check the health route over plain HTTP:

    ```bash
    curl -s http://localhost:8776/health; echo
    ```

    The `"pipecat"` object names `adapters/pipecat.eval.yaml` in `"settings_file"`, and shows `"vad": {"stop_secs": 0.8, "audio_idle_timeout": 0.0}` and `"turn": {"user_speech_timeout": 0.0}`.

The eval settings change three keys compared to the demo, which keeps the Pipecat defaults:

| Key | Demo | Eval | Reason |
| --- | --- | --- | --- |
| `vad.stop_secs` | `0.2` | `0.8` | Silero counts 800 ms of silence in the audio, like the τ³ adapter. |
| `vad.audio_idle_timeout` | `1.0` (wall clock) | `0` (off) | An evaluation client can pause its input for seconds. A wall-clock timer then ends the turn mid-utterance. |
| `turn.user_speech_timeout` | `0.6` (wall clock) | `0` | No wall-clock wait follows the voice activity detection (VAD) stop. |

The four variables apply only to the command that they prefix. A later `up -d fdh-voice` without them recreates the demo. Export them in the shell when later commands must keep the eval mode.

Connect the Realtime client to `ws://<host>:8776/v1/realtime`. For the τ² client setup and the smoke tests, refer to the deploy skill's `references/pipecat.md`.

## Run the τ³ Adapter

The τ³ adapter serves τ³ benchmarking only. Its defaults are the `realtime_eval.yaml` profile and the `gateway.eval.yaml` gateway config. It serves plain HTTP and `ws://`.

1. Stop `fdh-voice` first if it runs, because both adapters share `nemo-speech` and the Inference Hub:

    ```bash
    docker ps --filter name='^fdh-voice$' --format '{{.Names}} {{.Status}}'
    docker compose -f docker/docker-compose.frontend-delegation-hermes.yaml stop fdh-voice
    ```

2. Start `fdh-tau`. Always name the service, because `--profile tau up` without a service name starts both services:

    ```bash
    docker compose --env-file .env -f docker/docker-compose.frontend-delegation-hermes.yaml --profile tau up -d --build fdh-tau
    ```

3. After about 30 seconds, check the health route:

    ```bash
    curl -s http://localhost:8775/health; echo
    ```

    The response contains `"status": "ok"` and `"prototype": "frontend-delegation-hermes"`.

Connect the τ³ client to `ws://<host>:8775/v1/realtime`. The `fdh-tau` service has `restart: unless-stopped`. For benchmark runs, logs, and reports, refer to the deploy skill's `references/tau.md`.

## Verify the Deployment

Each container starts its own backend gateway on `127.0.0.1:8790` inside the container, then the voice server on container port `7860`. The gateway is not reachable from the host. Run the following checks.

1. Check that the services run and become healthy:

    ```bash
    docker compose -f docker/docker-compose.frontend-delegation-hermes.yaml --profile tau ps
    ```

2. Check the voice server health route. Use the command for your mode:

    ```bash
    curl -sk https://localhost:8776/health; echo   # Pipecat demo
    curl -s http://localhost:8776/health; echo     # Pipecat eval mode
    curl -s http://localhost:8775/health; echo     # τ³ adapter
    ```

3. Check the gateway health route inside the container. Replace `fdh-voice` with `fdh-tau` for the τ³ adapter:

    ```bash
    docker exec fdh-voice curl -s 127.0.0.1:8790/health; echo
    ```

    The response contains `"ok": true`, the open `"sessions"`, `"max_sessions"`, and the number of running Hermes `"workers"`.

4. Check which adapter started:

    ```bash
    docker logs fdh-voice 2>&1 | grep 'starting adapter'
    ```

If a check fails, refer to [Troubleshooting](#troubleshooting).

## Stop

The following command stops both adapters, their gateways, and their Hermes workers. Pass `--profile tau`, because a plain `down` leaves `fdh-tau` running:

```bash
docker compose -f docker/docker-compose.frontend-delegation-hermes.yaml --profile tau down
```

The gateway closes open sessions one after another, so the stop can take up to 75 seconds. Stop the FDH services before you take the root project down, because the FDH containers stay attached to its network. Then stop `nemo-speech`:

```bash
docker compose --profile frontend-backend-verdict-agent/single-gpu down
```

The commands keep `.env`, the built image, the logs in `logs/`, and the speech model weights in `models/nemo-speech`.

## Troubleshooting

The following table lists common failures. For more symptoms, refer to the deploy skill's `references/troubleshooting.md`.

| Symptom | Cause | Fix |
| --- | --- | --- |
| `up` fails with `network nemotron-voice-agent_default … not found`. | The root project network does not exist yet. | Start `nemo-speech` first, as [Start Speech](#start-speech) describes. |
| `--profile tau up` started both services. | `up` without a service name starts every service of the enabled profiles. | Stop `fdh-voice` with `docker compose -f docker/docker-compose.frontend-delegation-hermes.yaml stop fdh-voice`, and always name `fdh-tau`. |
| `fdh-tau` still runs after `down`. | `down` without `--profile tau` ignores the profiled service. | Run the `down` command with `--profile tau`. |
| A client cannot connect, or reaches the wrong adapter. | The adapters use different ports and schemes. | Use port `8775` with `ws://` for τ³. Use port `8776` with `wss://` for the demo, or `ws://` in eval mode. |
| The eval settings are lost after a later `up`. | The `FDH_PIPECAT_*` variables prefixed only one command. | Export the four variables, or prefix them again, and then check `/health`. |
| In eval mode, users are cut off mid-utterance, or the log shows `no audio received while speaking, forcing speech stop`. | The container runs the demo settings. | Start the eval mode with `FDH_PIPECAT_SETTINGS=adapters/pipecat.eval.yaml`, and check `/health` for `"audio_idle_timeout": 0.0`. |
| A container exits at start with `gateway not healthy`. | The gateway did not answer within 30 seconds, usually because of a wrong config name. | Run `docker logs <container>`, and check `FDH_GATEWAY_CONFIG` or `FDH_PIPECAT_GATEWAY_CONFIG`. |
| A container exits with `unknown adapter` or a usage message. | `FDH_ADAPTER`, `FDH_PROFILE`, or `FDH_GATEWAY_CONFIG` is empty or invalid. | Use the Compose services, which set all three. |
| `fdh-tau` restarts again and again. | The speech warm-up fails because `nemo-speech` is not up. | Start `nemo-speech` first. |
| The image build fails in the `uv pip check` step. | The Hermes checkout does not match the lock in `docker/fdh/requirements.txt`. | Check out the commit from the lock header in a clean checkout, or regenerate the lock as its header describes. |
| `session.update` fails with `worker_start_timeout` or `backend_unavailable`. | A Hermes worker did not start. | Read the worker logs in `logs/fdh_workers_pipecat/` or `logs/fdh_workers/`. Rebuild the image if the Hermes checkout changed. |
| The voice server exits with `server.max_sessions (N) exceeds the gateway's max_sessions`. | The voice server allows more sessions than the gateway. | Lower `FDH_PIPECAT_MAX_SESSIONS` or `FDH_MAX_SESSIONS`, or raise the gateway limit. |
| The browser does not ask for the microphone. | The page is served over plain HTTP or through a proxied host name. | Open `https://<host-ip>:8776/` and accept the certificate. |
| A root `docker compose … down` fails with `active endpoints`. | An FDH container is still attached to the root project network. | Stop the FDH services first. |
| `mv` or `rm` in `logs/` fails with permission denied. | The containers write logs as `root`. | Run `sudo chown -R "$USER" logs`. |

## Related Topics

- [Frontend Delegation Hermes README](../../src/examples/frontend_delegation_hermes/README.md)
- [Deploy the Frontend/Backend Verdict Agent](deploy-frontend-backend-verdict-agent.md)
- [Use the OpenAI Realtime-Compatible Gateway](use-realtime-gateway.md)
- [Troubleshooting](../06-troubleshooting.md)
