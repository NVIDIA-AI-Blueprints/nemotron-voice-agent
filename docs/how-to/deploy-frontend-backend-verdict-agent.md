# Deploy the Frontend/Backend Verdict Agent

## Overview

This guide takes you from a clean single-GPU host to a running and verified deployment of the [Frontend/Backend Verdict Agent](../../src/examples/frontend_backend_verdict/README.md). The deployment uses the `frontend-backend-verdict-agent/single-gpu` Docker Compose recipe.

The recipe runs the following parts:

- The `frontend-backend-verdict-agent-single-gpu` app service, which serves the agent on the OpenAI Realtime WebSocket at `/v1/realtime`.
- The `nemo-speech` sidecar, which runs NeMo-Speech.cpp automatic speech recognition (ASR) and text-to-speech (TTS) on GPU 0.
- Two large language models (LLMs) on the NVIDIA Inference Hub: a frontend LLM that talks to the caller, and a backend LLM that does the delegated work.

The agent serves only the Realtime model `nvidia/nemotron-realtime-frontend-backend-verdict`. It does not serve the browser UI, WebRTC, or RTVI.

For evaluation runs, deploy the same recipe in eval mode. Eval mode serves plain `ws://` and writes the agent's per-turn logs to the host. Refer to [Deploy in Eval Mode](#deploy-in-eval-mode).

> **Note:** The commands in this guide are checked against the repository's Compose files, `.env.example`, and source code. They were not run on GPU hardware for this revision of the guide.

## Instructions for AI Agents

If you are an AI agent that deploys this recipe for a user, follow these rules:

1. Before you start the deployment container, determine the prompt variant: `generic` or `tau`.
2. If the user has not stated it in the current request, ask which prompt variant to use. Do not assume a default. Do not continue until the user answers.
3. If an existing `.env` already sets `FBV_PROFILE`, do not treat it as the user's choice. Show the value and ask the user to confirm it or choose another.
4. If the user chooses `tau`, ask which `tau3_eval*` profile to use, and suggest the default arm (`tau3_eval_frontend_verdict_speak_history`).
5. Set the profile value from the [Choose a Prompt Variant](#choose-a-prompt-variant) table. Do not overwrite other keys in an existing `.env`.
6. After startup, read the profile, catalog path, and SHA-256 from the log, and report them to the user.

## Prerequisites

Before you start, make sure the host meets the following requirements:

- A host with one NVIDIA GPU that can run the `nemo-speech` sidecar of the `*/single-gpu` recipes. The recipe places `nemo-speech` on GPU 0. The LLMs do not use the local GPU.
- NVIDIA driver 580.x or newer for NeMo-Speech.cpp.
- Docker Compose v2.20 or newer, and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- Python 3, and either the Hugging Face `hf` CLI or `uvx`, for the speech model download script.
- Outbound HTTPS access from the host to the NVIDIA Inference Hub at `https://inference-api.nvidia.com/v1`.
- Free host ports `7860` (app) and `50051` (`nemo-speech`).
- A clone of this repository. Run every command in this guide from the repository root.

## Credentials

The recipe uses the credentials in the following table. This recipe is the exception to the general single-GPU rule: other `*/single-gpu` recipes use only `HF_TOKEN`, but this recipe also needs `NVIDIA_API_KEY` because its LLMs run on the NVIDIA Inference Hub.

| `.env` key | Required | Purpose |
| --- | --- | --- |
| `HF_TOKEN` | Yes | Hugging Face token. The download script reads it to fetch the NeMo-Speech.cpp model weights. |
| `NVIDIA_API_KEY` | Yes | An NVIDIA Inference Hub key. Both LLMs send it. |
| `REALTIME_API_KEY` | Recommended | The Bearer credential that Realtime clients send. Leave it unset only for unauthenticated local development. |

The recipe does not use `NVIDIA_SPEECH_API_KEY`, because speech runs in the local `nemo-speech` sidecar. The recipe does not need `docker login nvcr.io`.

## Configure the Environment

Create `.env` from the template only when it does not exist. This command preserves an existing `.env`:

```bash
test -f .env || cp .env.example .env
```

Open `.env` in an editor and set the keys from [Credentials](#credentials). Use your own values in place of the placeholders:

```env
HF_TOKEN=<your-hugging-face-token>
NVIDIA_API_KEY=<your-nvidia-inference-hub-key>
REALTIME_API_KEY=<a-strong-random-secret>
```

The following optional keys change the deployment. Leave them unset to keep the defaults.

| Key | Default | Purpose |
| --- | --- | --- |
| `FBV_PROFILE` | `tau3_eval_frontend_verdict_speak_history` | Behavior profile and prompt variant. Refer to [Select a Profile](#select-a-profile). |
| `FBV_PIPECAT_SETTINGS` | `eval.yaml` | Pipecat voice settings file in `src/examples/frontend_backend_verdict/adapters/pipecat/settings/`. |
| `PIPELINE_APP_PORT` | `7860` | Host port of the app service. |
| `PIPELINE_TLS` | `true` | `true` serves HTTPS and `wss://` with a self-signed certificate. `false` serves plain HTTP and `ws://`. |
| `FBA_VOICE_EVENT_LOG` | Empty | Path of a JSONL event log inside the container, for example `/tmp/fbv-events.jsonl`. Eval mode sets it. |
| `FBA_FILLER_LOG` | Empty | Path of a JSONL filler timing log inside the container. Eval mode sets it. |
| `NEMO_SPEECH_MODEL_LOC` | `models/nemo-speech` | Host directory that holds the speech model weights. |

Do not set `UVICORN_WS_PING_INTERVAL` or `UVICORN_WS_PING_TIMEOUT` in `.env` for this recipe. Compose sets them to `0` and `20`, which every served profile requires.

## Select a Profile

The `FBV_PROFILE` key selects the behavior profile. The profile also selects the prompt catalog. The following table lists the served profiles.

| Profile | Prompt catalog | Purpose |
| --- | --- | --- |
| `default` | `generic.yaml` | The default Tau arm with the generic prompt catalog. Use it for general use and prompt tuning. |
| `tau3_eval_frontend_verdict_speak_history` | `tau.yaml` | The default Tau arm: frontend barge-in verdict, spoken filler, and backend conversation history. Compose uses it when `FBV_PROFILE` is unset. |
| `tau3_eval_frontend_verdict_speak` | `tau.yaml` | Frontend verdict and spoken filler, without backend history. |
| `tau3_eval_frontend_verdict` | `tau.yaml` | Frontend verdict with filler logged but not spoken. |
| `tau3_eval_backend_history` | `tau.yaml` | Base evaluation arm plus backend conversation history. |
| `tau3_eval_backend_history_noguide` | `tau.yaml` | Backend history without its behavioral guidance. |
| `tau3_eval_normalization` | `tau.yaml` | Same as `tau3_eval`. It remains for existing commands. |
| `tau3_eval` | `tau.yaml` | Base evaluation arm. |
| `tau3_eval_frontend_verdict_speak_history_write_gate` | `tau.yaml` | The default Tau arm plus the write gate, which holds consequential tool calls until the caller confirms them. |

Every profile uses 800 ms server-owned turn detection and requires the WebSocket keepalive to be off. For the exact differences, refer to [Behavior Profiles](../../src/examples/frontend_backend_verdict/README.md#behavior-profiles). The profile files are in `src/examples/frontend_backend_verdict/core/config/profiles/`.

## Choose a Prompt Variant

Choose the prompt variant first, and then set the matching profile. The following table maps each variant to its `FBV_PROFILE` value.

| Prompt variant | Catalog | `FBV_PROFILE` value |
| --- | --- | --- |
| `generic` | `src/examples/frontend_backend_verdict/core/config/prompts/generic.yaml` | `default` |
| `tau` | `src/examples/frontend_backend_verdict/core/config/prompts/tau.yaml` | A `tau3_eval*` profile, for example the default arm `tau3_eval_frontend_verdict_speak_history` |

The Tau catalog is pinned by SHA-256 in `tests/unit/frontend_backend_verdict/test_fbv_prompt_variants.py`. Do not edit `tau.yaml` to tune prompts. A change to it requires a new pin and an attached Tau A/B evaluation run. Tune `generic.yaml` instead.

Set the value in `.env` without changing other keys. The following command sets `FBV_PROFILE=default` and replaces an existing `FBV_PROFILE` line if there is one:

```bash
grep -q '^FBV_PROFILE=' .env \
  && sed -i 's/^FBV_PROFILE=.*/FBV_PROFILE=default/' .env \
  || echo 'FBV_PROFILE=default' >> .env
```

Replace `default` with a `tau3_eval*` profile name to select the Tau catalog.

## Start the Deployment Container

Download the NeMo-Speech.cpp weights once, as your user. Do not use `sudo`. The script reads `HF_TOKEN` from `.env`:

```bash
bash scripts/download-nemo-speech-models.sh
```

Check that Compose accepts the recipe and your `.env`. The command prints nothing when the configuration is valid:

```bash
docker compose --profile frontend-backend-verdict-agent/single-gpu config -q
```

Start the recipe:

```bash
docker compose --profile frontend-backend-verdict-agent/single-gpu up -d
```

The first start builds the `nemotron-voice-agent:latest` image and pulls the `nemo-speech` image, so it can take several minutes. Use `--build` on later starts only after source or `Dockerfile` changes.

## Verify the Deployment

Run the following checks in order.

1. Check that both services are running and that the app service becomes healthy:

    ```bash
    docker compose --profile frontend-backend-verdict-agent/single-gpu ps
    ```

2. Check the liveness route. Use `http://` when `PIPELINE_TLS=false`:

    ```bash
    curl -fk https://localhost:7860/health
    ```

    The route returns `{"status":"ok"}`. It confirms that the server process is up. It does not check speech or the LLMs.

3. Check the effective profile and Pipecat settings in the container:

    ```bash
    docker compose --profile frontend-backend-verdict-agent/single-gpu exec \
      frontend-backend-verdict-agent-single-gpu printenv FBV_PROFILE FBV_PIPECAT_SETTINGS
    ```

4. Connect one Realtime client, as [Connect a Client](#connect-a-client) shows. The agent loads its profile, builds its LLM clients, and warms up speech with one short ASR request and one short TTS request when the first session connects.

5. Read the ready line and the prompt catalog line from the app log:

    ```bash
    docker compose --profile frontend-backend-verdict-agent/single-gpu logs \
      frontend-backend-verdict-agent-single-gpu | grep -E 'verdict agent ready|prompts .*sha256'
    ```

    The ready line contains `adapter=pipecat` and `profile=<profile>`. The prompt line has the form `prompts <catalog path> (sha256 <hash>)`. The catalog path ends in `generic.yaml` for the `default` profile and in `tau.yaml` for every `tau3_eval*` profile.

If a check fails, refer to [Troubleshooting](#troubleshooting).

## Connect a Client

Connect a Realtime client to the following URL. Replace `<host>` with the host name or IP address, and use `ws://` when `PIPELINE_TLS=false`:

```text
wss://<host>:7860/v1/realtime?model=nvidia/nemotron-realtime-frontend-backend-verdict
```

When `REALTIME_API_KEY` is set, send it as a Bearer credential in the `Authorization` header. The following Python script connects, sends one text turn, and prints the type of each server event until the response is done. Export your Realtime key in the shell first:

```bash
export REALTIME_API_KEY=<your-realtime-api-key>
uv run python - <<'EOF'
import asyncio, json, os, ssl

import websockets

# Use ws:// when PIPELINE_TLS=false, for example in eval mode.
URL = "wss://localhost:7860/v1/realtime?model=nvidia/nemotron-realtime-frontend-backend-verdict"


async def main():
    tls = ssl.create_default_context()
    tls.check_hostname = False
    tls.verify_mode = ssl.CERT_NONE  # Self-signed development certificate only.
    key = os.environ.get("REALTIME_API_KEY", "")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    secure = URL.startswith("wss://")
    async with websockets.connect(URL, additional_headers=headers, ssl=tls if secure else None) as ws:
        await ws.send(json.dumps({"type": "session.update", "session": {"type": "realtime"}}))
        await ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {"type": "message", "role": "user",
                     "content": [{"type": "input_text", "text": "Hello."}]},
        }))
        await ws.send(json.dumps({"type": "response.create"}))
        while True:
            event = json.loads(await ws.recv())
            print(event["type"], event.get("error", ""))
            if event["type"] in ("response.done", "error"):
                break


asyncio.run(main())
EOF
```

The first event is `session.created`. A `response.done` event shows that the session, both LLMs, and TTS answered. An `error` event carries a code. Refer to [Troubleshooting](#troubleshooting).

The agent has the following client-facing limitations:

- Turns end on the server voice activity detection (VAD). The agent refuses `input_audio_buffer.commit` with error code `unsupported_event`.
- The agent ignores the voice that the client requests and uses the catalog voice.
- The agent refuses per-response `instructions`, `tools`, or `tool_choice` on `response.create` with error code `unsupported_response_override`.
- At most 8 sessions run at the same time.

For the full event reference, refer to [Use the OpenAI Realtime-Compatible Gateway](use-realtime-gateway.md).

## Deploy in Eval Mode

Eval mode is the deployment for evaluation runs, such as a Tau evaluation. It uses the same image, code, and profile as the standard deployment and changes three settings, which the following table lists.

| Setting | Eval mode value | Reason |
| --- | --- | --- |
| `PIPELINE_TLS` | `false` | The server serves plain HTTP and `ws://`, so an evaluation client connects without a certificate. |
| `FBA_VOICE_EVENT_LOG` and `FBA_FILLER_LOG` | `/app/logs/fbv_voice_events.jsonl` and `/app/logs/fbv_filler.jsonl` | The agent writes its per-turn event log and its filler timing log. Evaluation metrics read both. |
| Volume | `./logs` mounted at `/app/logs` | The two logs reach the `logs/` directory of the repository on the host. |

The `docker/docker-compose.fbv-eval.yaml` override file holds these settings. Pass it after `docker-compose.yml` in every Compose command for the deployment. A command without it recreates the container with the standard settings.

To deploy in eval mode, complete the following steps:

1. Complete [Configure the Environment](#configure-the-environment) and [Choose a Prompt Variant](#choose-a-prompt-variant). For a Tau evaluation, use a `tau3_eval*` profile.

2. Create the host log directory, so that it belongs to your user:

    ```bash
    mkdir -p logs
    ```

3. Start the recipe with the override file:

    ```bash
    docker compose -f docker-compose.yml -f docker/docker-compose.fbv-eval.yaml \
      --profile frontend-backend-verdict-agent/single-gpu up -d
    ```

4. Download the NLTK `punkt_tab` data in the app container. Pipecat loads it when it sets up each pipeline. On a new container, the first sessions otherwise wait for the download and fail if it takes longer than 120 seconds. The command prints `True`:

    ```bash
    docker compose -f docker-compose.yml -f docker/docker-compose.fbv-eval.yaml \
      --profile frontend-backend-verdict-agent/single-gpu exec frontend-backend-verdict-agent-single-gpu \
      .venv/bin/python -c "import nltk; print(nltk.download('punkt_tab', quiet=True))"
    ```

5. Check the liveness route over plain HTTP:

    ```bash
    curl -f http://localhost:7860/health
    ```

6. Check the eval settings in the container:

    ```bash
    docker compose -f docker-compose.yml -f docker/docker-compose.fbv-eval.yaml \
      --profile frontend-backend-verdict-agent/single-gpu exec frontend-backend-verdict-agent-single-gpu \
      printenv PIPELINE_TLS FBA_VOICE_EVENT_LOG FBA_FILLER_LOG FBV_PROFILE
    ```

7. Connect one client to `ws://localhost:7860/v1/realtime?model=nvidia/nemotron-realtime-frontend-backend-verdict`, as [Connect a Client](#connect-a-client) shows. Then check that the event log grew on the host:

    ```bash
    tail -n 1 logs/fbv_voice_events.jsonl
    ```

    The `session_start` record of each session carries `adapter`, the Pipecat settings, the backend history and normalization settings, and the prompt catalog name and SHA-256 (`prompts`).

The container writes the log files as `root`. The repository ignores the `logs/` directory. To switch back to the standard deployment, run the `up -d` command from [Start the Deployment Container](#start-the-deployment-container) without the override file. Compose then recreates the container with TLS and without the logs.

## Run a Tau Evaluation

The tau-bench (Tau) evaluation harness, `voice-agent-evaluation`, is not part of this repository. Use the following steps to point it at this deployment:

1. Deploy with a `tau3_eval*` profile, so the run uses the pinned Tau prompt catalog. The default arm is `tau3_eval_frontend_verdict_speak_history`.
2. Deploy in eval mode, as [Deploy in Eval Mode](#deploy-in-eval-mode) describes. The agent's event and filler logs then reach `logs/` on the host.
3. Keep the Compose keepalive values (`UVICORN_WS_PING_INTERVAL=0` and `UVICORN_WS_PING_TIMEOUT=20`).
4. Configure the harness with `execution.backend: evaluator_realtime` and `product.profile: openai`, the base URL `http://localhost:7860` or the WebSocket URL `ws://localhost:7860/v1/realtime`, and your `REALTIME_API_KEY`. The `tau_native` lane cannot select this model ID.
5. Run with a concurrency of 1. The `tau3_eval.yaml` profile asks for this until `server.max_sessions` is measured.
6. Record the profile and the prompt catalog SHA-256 from the app log with the results. The `session_start` record of the event log also carries the catalog name and SHA-256.

## Stop and Clean Up

Stop the recipe with the same profile that started it:

```bash
docker compose --profile frontend-backend-verdict-agent/single-gpu down
```

For an eval-mode deployment, pass the override file as well:

```bash
docker compose -f docker-compose.yml -f docker/docker-compose.fbv-eval.yaml \
  --profile frontend-backend-verdict-agent/single-gpu down
```

The command removes the containers. It keeps `.env`, the built image, the eval-mode logs in `logs/`, and the speech model weights in `models/nemo-speech`. Remove the weights only when you no longer need any single-GPU recipe that uses them.

## Troubleshooting

The following table lists common failures for this recipe. For general startup issues, refer to [Troubleshooting](../06-troubleshooting.md).

| Symptom | Cause | Fix |
| --- | --- | --- |
| The client receives an `error` event with code `server_error`, and the socket closes with code 1011. The app log shows `WebSocket keepalive differs from the profile`. | The server keepalive does not match the profile. Every served profile requires `UVICORN_WS_PING_INTERVAL=0` and `UVICORN_WS_PING_TIMEOUT=20`. | Remove the two keys from `.env` or set them to `0` and `20`, and then rerun the `up -d` command. |
| The client receives an `error` event with code `server_error`, and the socket closes with code 1011. The app log names `FBV_PROFILE`. | `FBV_PROFILE` names a profile that the agent does not serve. | Set a profile from [Select a Profile](#select-a-profile), and then rerun the `up -d` command. |
| The client receives an `error` event with code `server_busy`, and the socket closes with code 1013. | Either 8 sessions are already open (`server.max_sessions`), or the speech warm-up failed. The app log shows `speech warm-up failed` for a warm-up failure. | Close idle sessions, or wait and reconnect. For a warm-up failure, fix the speech service, and then reconnect. The next session retries the warm-up. |
| The app log shows `ASR warm-up failed against` or `TTS warm-up failed against`. | The `nemo-speech` sidecar is not running, not ready, or has no model weights. | Run `docker compose --profile frontend-backend-verdict-agent/single-gpu logs nemo-speech`. If the weights are missing, run `bash scripts/download-nemo-speech-models.sh` as your user or set `NEMO_SPEECH_MODEL_LOC`. |
| The WebSocket handshake fails, and the server closes it with code 1008 and reason `realtime authentication failed`. | `REALTIME_API_KEY` is set, and the client sent no credential or a different one. | Send `Authorization: Bearer <your-realtime-api-key>` with the value from `.env`. |
| The session starts, but each turn returns an `error` event with code `agent_error`. | `NVIDIA_API_KEY` is missing, is not an Inference Hub key, or the host cannot reach the Inference Hub. | Set `NVIDIA_API_KEY` to an NVIDIA Inference Hub key, check outbound HTTPS access, and then rerun the `up -d` command. |
| The download script fails, or `nemo-speech` cannot find its models. | `HF_TOKEN` is missing or invalid, or the weights are in another directory. | Set `HF_TOKEN` in `.env` and rerun the script as your user, or point `NEMO_SPEECH_MODEL_LOC` at the weights. |
| The session fails, and the app log shows an error that names `agent.overrides.prompts.path`. | `FBA_VOICE_PROMPTS` points at a missing or invalid prompt catalog. | Remove `FBA_VOICE_PROMPTS` from `.env` to use the profile's catalog, or point it at a valid catalog, and then rerun the `up -d` command. |
| The client fails with `ssl argument is incompatible with a ws:// URI`, or a `wss://` handshake fails with an SSL error. | The client and the server disagree on TLS. Eval mode serves plain `ws://`. | Use `ws://` and no TLS context for an eval-mode deployment. Use `wss://` for the standard deployment. |
| In eval mode, `logs/` stays empty or the container recreates without the eval settings. | A Compose command ran without `-f docker/docker-compose.fbv-eval.yaml`, or no session has connected yet. | Pass both `-f` files in every Compose command for the deployment, and connect one session. |
| `up -d` fails with `port is already allocated` or `address already in use`. | Another process uses port `7860` or `50051`. | Set `PIPELINE_APP_PORT` to a free port for the app. For `50051`, stop the other process or the other recipe that runs a speech sidecar. |

## Related Topics

- [Frontend/Backend Verdict Agent README](../../src/examples/frontend_backend_verdict/README.md)
- [Getting Started](../01-getting-started.md)
- [Use the OpenAI Realtime-Compatible Gateway](use-realtime-gateway.md)
- [WebSocket Keepalive](tune-pipeline-performance.md#websocket-keepalive)
- [Troubleshooting](../06-troubleshooting.md)
