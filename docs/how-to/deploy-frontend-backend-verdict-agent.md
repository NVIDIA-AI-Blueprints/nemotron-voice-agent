# Deploy the Frontend/Backend Verdict Agent

## Overview

This guide takes you from a clean host to a running and verified deployment of the [Frontend/Backend Verdict Agent](../../src/examples/frontend_backend_verdict/README.md). The agent pairs a frontend large language model (LLM) that talks to the user with a backend LLM that does the delegated work. Both LLMs run on the NVIDIA Inference Hub by default. An optional overlay also runs the frontend LLM on the host. Refer to [Run the Frontend LLM Locally](#run-the-frontend-llm-locally).

The agent serves two kinds of sessions:

- Browser UI sessions over WebRTC or the UI's WebSocket transport, with the Real-Time Voice Inference (RTVI) protocol, like the other examples in this repository.
- OpenAI Realtime sessions on `/v1/realtime` with the model `nvidia/nemotron-realtime-frontend-backend-verdict`, for Realtime clients and evaluation runs such as tau-bench (Tau).

The example has three Docker Compose recipes, which differ in where automatic speech recognition (ASR) and text-to-speech (TTS) run:

- `frontend-backend-verdict-agent` (cloud): ASR and TTS run on NVIDIA cloud functions (NVCF). The recipe has no sidecars and needs no local GPU.
- `frontend-backend-verdict-agent/server`: ASR and TTS run in the `nemotron-asr-streaming-english` and `magpie-multilingual-tts-service` NIM sidecars on a workstation GPU.
- `frontend-backend-verdict-agent/single-gpu`: ASR and TTS run in the `nemo-speech` NeMo-Speech.cpp sidecar on one GPU.

The `FBV_PROFILE` setting decides what a deployment serves. Refer to [Choose a Profile](#choose-a-profile).

> **Note:** The commands in this guide are checked against the repository's Compose files, `.env.example`, and source code. They were not run on GPU hardware for this revision of the guide.

## Instructions for AI Agents

If you are an AI agent that deploys this example for a user, follow these rules:

1. If the user's request does not say which profile to deploy, ask: "Which profile: `default` (for the browser UI) or `eval` (for evaluation runs such as Tau)?" Wait for the answer. Do not choose one silently, even though the agent uses `default` when nothing is set.
2. If an existing `.env` already sets `FBV_PROFILE`, do not treat it as the answer. Show the value, and ask the user to confirm it or change it.
3. For `default`, use the plain recipe command for the Compose profile that the user wants: cloud (`frontend-backend-verdict-agent`), `frontend-backend-verdict-agent/server`, or `frontend-backend-verdict-agent/single-gpu`. For `eval`, use `frontend-backend-verdict-agent/single-gpu` with the `docker/docker-compose.fbv-eval.yaml` override. Refer to [Choose a Profile](#choose-a-profile).
4. After startup and the first session, read the profile, the prompt catalog path, and its SHA-256 from the app log, and report them to the user. Refer to [Verify the Deployment](#verify-the-deployment).
5. For Tau or any other evaluation run, deploy `eval`. Do not add the local LLM overlay, and do not set `FBV_REALTIME_*` switches, such as `FBV_REALTIME_TRACING`, `FBV_REALTIME_TURN_MODES`, or `FBV_REALTIME_MCP`, for an evaluation. With `FBV_PROFILE=eval`, the agent ignores every `FBV_REALTIME_*` switch.

Do not overwrite other keys in an existing `.env`.

## Prerequisites

Every recipe needs the following:

- Docker Compose v2.20 or newer.
- Outbound HTTPS access from the host to the NVIDIA Inference Hub at `https://inference-api.nvidia.com/v1`.
- A free host port `7860` for the app, or another port set with `PIPELINE_APP_PORT`.
- A clone of this repository. Run every command in this guide from the repository root.

Each Compose profile adds the requirements in the following table.

| Compose profile | Hardware | Additional requirements |
| --- | --- | --- |
| `frontend-backend-verdict-agent` | No local GPU | Outbound access to `grpc.nvcf.nvidia.com:443` for ASR and TTS. |
| `frontend-backend-verdict-agent/server` | One workstation GPU for both NIM sidecars (GPU 0). Not supported on DGX Spark or Jetson Thor. | The [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), NGC access to pull the NIM images, and free host ports `50151`, `9000`, `50152`, and `9001`. |
| `frontend-backend-verdict-agent/single-gpu` | One NVIDIA GPU for the `nemo-speech` sidecar (GPU 0). The LLMs do not use the local GPU. | NVIDIA driver 580.x or newer, the NVIDIA Container Toolkit, Python 3 and either the Hugging Face `hf` CLI or `uvx` for the model download script, and a free host port `50051`. |

## Credentials

The credentials depend on the Compose profile. The following table lists the required keys.

| Compose profile | Required `.env` keys | `docker login nvcr.io` |
| --- | --- | --- |
| `frontend-backend-verdict-agent` | `NVIDIA_API_KEY` | No |
| `frontend-backend-verdict-agent/server` | `NVIDIA_API_KEY` | Yes, before `up` |
| `frontend-backend-verdict-agent/single-gpu` | `HF_TOKEN` and `NVIDIA_API_KEY` | No |

Every recipe uses the keys in the following table.

| `.env` key | Required | Purpose |
| --- | --- | --- |
| `NVIDIA_API_KEY` | Yes | An NVIDIA Inference Hub key (`sk-...`). Both LLMs send it. |
| `HF_TOKEN` | `/single-gpu` only | Hugging Face token. The download script reads it to fetch the NeMo-Speech.cpp model weights. |
| `REALTIME_API_KEY` | Recommended | The Bearer credential that OpenAI Realtime clients send. Leave it unset only for unauthenticated local development. |
| `WEATHERAPI_KEY` | No | Live results for the `get_weather` tool in browser UI sessions. |
| `FINNHUB_API_KEY` | No | Live results for the `get_stock_price` tool in browser UI sessions. |
| `PERPLEXITY_API_KEY` | No | Live results for the `web_search` tool in browser UI sessions. |

Without its key, a tool returns an "unavailable" result, and the agent tells the user that it cannot do that. The `calculate_bmi` and `generate_random_number` tools need no key. Evaluation runs do not use these tools, because Realtime clients send their own tools.

The same `NVIDIA_API_KEY` variable has other uses on the cloud and server recipes:

- **Cloud recipe:** the ASR and TTS NVCF functions also authenticate with `NVIDIA_API_KEY`, so the recipe sends the same Inference Hub key to both the LLMs and NVCF speech. It has not been verified that one Inference Hub key also works for NVCF speech. If your key does not work for NVCF speech, the speech warm-up fails. In that case, use the `/server` or `/single-gpu` recipe, whose speech runs locally.
- **Server recipe:** `docker login nvcr.io` and the NIM sidecars (`NGC_API_KEY`) also use `NVIDIA_API_KEY`. It has not been verified that one Inference Hub key also works for NGC. If the image pull or the NIM model download fails with an authorization error, use the `/single-gpu` recipe.

The `/single-gpu` recipe is the exception to the general single-GPU rule. Other `*/single-gpu` recipes use only `HF_TOKEN`, but this recipe also needs `NVIDIA_API_KEY` for its LLMs. It does not use `NVIDIA_SPEECH_API_KEY` and does not need `docker login nvcr.io`.

## Choose a Profile

The agent has two profiles. When `FBV_PROFILE` is unset, the agent uses `default`. The following table compares them.

| Profile | Use it for | What it serves | Compose profiles | Prompt catalog | Turn taking | TLS | Logs | Deploy command |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `default` | The browser UI, development, and prompt tuning | Browser UI sessions with the `default` settings, and OpenAI Realtime sessions with the `eval` settings | `frontend-backend-verdict-agent`, `frontend-backend-verdict-agent/server`, or `frontend-backend-verdict-agent/single-gpu` | `core/config/prompts/generic.yaml`, free to tune | Smart Turn v3: a 200 ms voice activity detection (VAD) stop hands the turn to Smart Turn, with a 1 s silence fallback | On (HTTPS and `wss://`) | App log only | `docker compose --profile <compose-profile> up -d` |
| `eval` | Evaluation runs, such as Tau | OpenAI Realtime sessions only, with the `eval` settings. Browser UI sessions are refused. | `frontend-backend-verdict-agent/single-gpu` only | `core/config/prompts/tau.yaml`, pinned by SHA-256 | Server-owned 800 ms silence | Off (HTTP and `ws://`) | App log, plus the event and filler logs in `logs/` on the host | `docker compose -f docker-compose.yml -f docker/docker-compose.fbv-eval.yaml --profile frontend-backend-verdict-agent/single-gpu up -d` |

OpenAI Realtime sessions use the `eval` settings in both profiles, so `FBV_PROFILE` never changes what a Realtime client sees. Those settings take the tools and instructions from the client's `session.update`, use the Tau prompt catalog, and end turns after 800 ms of silence.

The `eval` profile refuses browser UI sessions, so they cannot take resources from an evaluation run. The UI shows no error: on WebRTC the session ends, and the app log shows the error.

You do not need to set `FBV_PROFILE` in `.env`. Compose sets `default` when the key is unset, and the eval override file sets `eval`. As an advanced option, you can set `FBV_PROFILE=default` or `FBV_PROFILE=eval` in `.env`. The agent supports no other value. An old value, such as a `tau3_eval*` profile name, fails the first session with an error that names `default` and `eval`. The agent checks the value when the first session arrives, not when the container starts.

The profile files are `src/examples/frontend_backend_verdict/core/config/profiles/default.yaml` and `eval.yaml`. Each profile's Pipecat settings file in `adapters/pipecat/settings/` is fixed in code: `default.yaml` for `default` and `eval.yaml` for `eval`. For every difference between the profiles, refer to [Profiles](../../src/examples/frontend_backend_verdict/README.md#profiles).

The Tau catalog is pinned by SHA-256 in `tests/unit/frontend_backend_verdict/test_fbv_prompt_variants.py`. Do not edit `tau.yaml` to tune prompts. A change to it requires a new pin and an attached Tau A/B evaluation run. Tune `generic.yaml` instead.

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

Set `HF_TOKEN` only for the `/single-gpu` recipe.

The following optional keys change the deployment. Leave them unset to keep the defaults.

| Key | Default | Purpose |
| --- | --- | --- |
| `FBV_PROFILE` | `default` | The profile: `default` or `eval`. Refer to [Choose a Profile](#choose-a-profile). |
| `PIPELINE_APP_PORT` | `7860` | Host port of the app service. |
| `PIPELINE_TLS` | `true` | `true` serves HTTPS and `wss://` with a self-signed certificate. `false` serves plain HTTP and `ws://`. Browsers need HTTPS for the microphone, except on `localhost`. |
| `FBA_VOICE_EVENT_LOG` | Empty | Path of a JSONL event log inside the container for Realtime sessions, for example `/tmp/fbv-events.jsonl`. Eval mode sets it. Browser UI sessions never write it. |
| `FBA_FILLER_LOG` | Empty | Path of a JSONL filler timing log inside the container for Realtime sessions. Eval mode sets it. |
| `FBA_ASR_SERVER`, `FBA_TTS_SERVER` | Empty | Replace the ASR or TTS server of the selected catalog entry. |
| `NEMO_SPEECH_MODEL_LOC` | `models/nemo-speech` | Host directory that holds the speech model weights for `/single-gpu`. |
| `ENABLE_WELCOME_MESSAGE` | Unset (greets) | `false` turns off the browser UI greeting. Realtime sessions never greet. |
| `ENABLE_TRACING` | `false` | `true` traces browser UI sessions. Refer to [Enable Tracing](#enable-tracing). |
| `FBV_REALTIME_TRACING` | `false` | With `ENABLE_TRACING=true`, `true` also traces Realtime sessions. Ignored when `FBV_PROFILE=eval`. |
| `FBV_REALTIME_TURN_MODES` | Unset (`server_vad` only) | Comma-separated Realtime turn modes that clients can select: `server_vad`, `manual`, and `semantic_vad`. Ignored when `FBV_PROFILE=eval`. Refer to [Realtime Turn Modes](../../src/examples/frontend_backend_verdict/README.md#realtime-turn-modes). |
| `FBV_REALTIME_MCP` | `false` | `true` lets Realtime clients declare Model Context Protocol (MCP) tools that the server runs. It also needs `REALTIME_MCP_ALLOWED_SERVER_URLS`. Ignored when `FBV_PROFILE=eval`. Refer to [MCP Tools](../../src/examples/frontend_backend_verdict/README.md#mcp-tools). |

Do not set `UVICORN_WS_PING_INTERVAL` or `UVICORN_WS_PING_TIMEOUT` in `.env` for these recipes. Compose sets them to `0` and `20`. The keepalive applies to the whole server process, and Realtime sessions require it off.

## Start the Deployment Container

The following table lists the app service and the sidecars of each Compose profile. The later commands use these names in place of `<compose-profile>` and `<app-service>`.

| Compose profile | App service | Sidecars |
| --- | --- | --- |
| `frontend-backend-verdict-agent` | `frontend-backend-verdict-agent` | None |
| `frontend-backend-verdict-agent/server` | `frontend-backend-verdict-agent-server` | `nemotron-asr-streaming-english` (ASR) and `magpie-multilingual-tts-service` (TTS) |
| `frontend-backend-verdict-agent/single-gpu` | `frontend-backend-verdict-agent-single-gpu` | `nemo-speech` (ASR and TTS) |

Complete the preparation step for your Compose profile:

- For `/server`, log in to the NVIDIA NGC Docker Registry:

    ```bash
    set -a; . ./.env; set +a
    printf '%s' "$NVIDIA_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin
    ```

- For `/single-gpu`, download the NeMo-Speech.cpp weights once, as your user. Do not use `sudo`. The script reads `HF_TOKEN` from `.env`:

    ```bash
    bash scripts/download-nemo-speech-models.sh
    ```

- The cloud recipe needs no preparation.

Check that Compose accepts the recipe and your `.env`. The command prints nothing when the configuration is valid:

```bash
docker compose --profile <compose-profile> config -q
```

Start the recipe with the `default` profile:

```bash
docker compose --profile <compose-profile> up -d
```

For example, run `docker compose --profile frontend-backend-verdict-agent/single-gpu up -d`. For the `eval` profile, follow [Deploy in Eval Mode](#deploy-in-eval-mode) instead.

The first start builds the `nemotron-voice-agent:latest` image and pulls the sidecar images, so it can take several minutes. On `/server`, the first start of the NIM sidecars downloads their models and can take much longer. Use `--build` on later starts only after source or `Dockerfile` changes.

## Run the Frontend LLM Locally

Without extra files, the recipes start no LLM container, and both LLMs run on the NVIDIA Inference Hub. To also run Nemotron 3.5 Lightning on the host, pass the `docker/docker-compose.fbv-local-llm.yaml` overlay after `docker-compose.yml`:

```bash
docker compose -f docker-compose.yml -f docker/docker-compose.fbv-local-llm.yaml \
  --profile frontend-backend-verdict-agent/single-gpu up -d
```

The overlay works with the `/server` and `/single-gpu` recipes, as the following table shows.

| Compose profile | Added LLM service | Endpoint |
| --- | --- | --- |
| `frontend-backend-verdict-agent/server` | `nvidia-llm`, the Nemotron 3.5 Lightning NIM | `http://nvidia-llm:8000/v1` |
| `frontend-backend-verdict-agent/single-gpu` | `nvidia-llm-vllm-lightning`, Nemotron 3.5 Lightning on vLLM | `http://nvidia-llm-vllm:8000/v1` |

Note the following behavior:

- The browser UI's **LLM** menu lists the `nemotron-3.5-lightning-local` entry only while its endpoint is reachable. The Inference Hub entry stays the default.
- The local LLM can replace only the frontend LLM. The backend LLM stays the profile's Nemotron 3 Ultra on the Inference Hub, so `NVIDIA_API_KEY` is still required.
- On `/server`, the `nvidia-llm` NIM also runs on GPU 0, beside the speech NIM sidecars.
- On `/single-gpu`, the vLLM service shares GPU 0 with `nemo-speech`. It sizes its memory from the free GPU memory at startup and needs about 28 GiB after the `VLLM_VRAM_HEADROOM_MIB` headroom, which is 4 GiB by default.
- In an eval-mode container, a Realtime session that selects a non-default LLM, for example through an `ek_` client secret, receives an `error` event with code `server_error`, and the socket closes with code 1011. Do not use the overlay for evaluation runs.

Pass the overlay file in every Compose command for the deployment, including `ps`, `logs`, and `down`.

## Enable Tracing

The agent uses the repository's OpenTelemetry tracing. Set `ENABLE_TRACING=true` in `.env`, and add the `tracing` overlay profile to start the Phoenix collector:

```bash
docker compose --profile frontend-backend-verdict-agent/single-gpu --profile tracing up -d
```

Browser UI sessions are then traced. Realtime sessions are traced only when you also set `FBV_REALTIME_TRACING=true` and the container does not run the `eval` profile. An eval-mode container ignores every `FBV_REALTIME_*` switch, and the eval override file sets `FBV_REALTIME_TRACING=false`. For the collector settings and the Phoenix UI, refer to [Enable OpenTelemetry Tracing](enable-opentelemetry-tracing.md).

## Verify the Deployment

Run the following checks in order.

1. Check that the services are running and that the app service becomes healthy:

    ```bash
    docker compose --profile <compose-profile> ps
    ```

2. Check the liveness route. Use `http://` when `PIPELINE_TLS=false`:

    ```bash
    curl -fk https://localhost:7860/health
    ```

    The route returns `{"status":"ok"}`. It confirms that the server process is up. It does not check speech or the LLMs.

3. Check the effective profile in the container:

    ```bash
    docker compose --profile <compose-profile> exec <app-service> printenv FBV_PROFILE
    ```

4. Start one session. Open the browser UI, as [Use the Browser UI](#use-the-browser-ui) describes, or connect one Realtime client, as [Connect a Realtime Client](#connect-a-realtime-client) shows. When the first session of each kind arrives, the agent loads its profile, builds its LLM clients, and warms up speech with one short ASR request and one short TTS request.

5. Read the ready line and the prompt catalog line from the app log:

    ```bash
    docker compose --profile <compose-profile> logs <app-service> | grep -E 'verdict agent ready|prompts .*sha256'
    ```

    The ready line contains `adapter=pipecat` and `profile=<profile>`. The prompt line has the form `prompts <catalog path> (sha256 <hash>)`. A browser UI session logs `profile=default` and a catalog path that ends in `generic.yaml`. A Realtime session logs `profile=eval` and a catalog path that ends in `tau.yaml`, in both profiles. On a `default` container that serves both kinds of sessions, both pairs of lines appear.

If a check fails, refer to [Troubleshooting](#troubleshooting).

## Use the Browser UI

The browser UI works with the `default` profile only. Open `https://<host>:7860` in a browser. Replace `<host>` with the host name or IP address, or use `localhost` on the same machine. The server uses a self-signed certificate, so the browser shows a warning that you must accept once. For plain-HTTP browser access, refer to [Browser Access](../06-troubleshooting.md#browser-access).

The Compose recipes pin `EXAMPLE_SELECTION=frontend-backend-verdict-agent`, so the UI shows only this example. When the server exposes several examples, select **Frontend Backend Verdict Agent**. Then start a session and speak. The agent greets you first, unless `ENABLE_WELCOME_MESSAGE=false`.

The UI offers the following controls and views for this example:

- **LLM**, **ASR**, **TTS**, and the TTS voice: these choose the frontend LLM and the speech endpoints from the example's catalogs. The UI has no control for the backend LLM. The backend is always the profile's, Nemotron 3 Ultra on the NVIDIA Inference Hub. The **LLM** menu also lists the local LLM when you use the [local LLM overlay](#run-the-frontend-llm-locally).
- Voice picker: a voice change during a session applies from the next sentence on.
- **Prompt**: you can edit the `frontend` prompt for the session. The agent's other prompts stay hidden.
- **Tools**: the read-only tools panel lists the five tools: `get_weather`, `get_stock_price`, `web_search`, `calculate_bmi`, and `generate_random_number`. All five are always enabled, and the server runs them. Refer to [Credentials](#credentials) for their keys.
- **Transcripts and metrics**: the UI shows the transcripts of both sides and the user-to-bot latency. After each answered turn, the metrics panel also receives one `Turn N` group with voice-to-voice latency, ASR final, agent start, agent to first audio, and the latency and input and output tokens of the frontend and backend LLMs. The UI does not show streamed LLM text, because the agent is not a Pipecat LLM service.

The UI sessions share one cap of 8 concurrent sessions. That cap is separate from the cap of the Realtime sessions, so browser sessions never take an evaluation slot. Both kinds of sessions still share the speech sidecars and the LLM endpoints.

## Connect a Realtime Client

Connect a Realtime client to the following URL. Replace `<host>` with the host name or IP address, and use `ws://` when `PIPELINE_TLS=false`:

```text
wss://<host>:7860/v1/realtime?model=nvidia/nemotron-realtime-frontend-backend-verdict
```

When `REALTIME_API_KEY` is set, send it as a Bearer credential in the `Authorization` header. An `ek_` client secret from `POST /v1/realtime/client_secrets` also works. The session ignores the secret template's standard session fields, such as `instructions`, `tools`, and `turn_detection`. The following Python script connects, sends one text turn, and prints the type of each server event until the response is done. Export your Realtime key in the shell first:

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

Realtime sessions have the following client-facing limitations:

- By default, turns end on the server voice activity detection (VAD). The agent refuses `input_audio_buffer.commit` with error code `unsupported_event` and approximates `semantic_vad` with silence-based turn detection. Set `FBV_REALTIME_TURN_MODES` to allow manual commits or Smart Turn `semantic_vad`.
- By default, the agent accepts function tools only and refuses `type: "mcp"` tools. Set `FBV_REALTIME_MCP=true` and `REALTIME_MCP_ALLOWED_SERVER_URLS` to allow MCP tools, which the server runs and hides from the client.
- A session cannot select its own ASR or TTS service.
- The agent ignores the voice that the client requests and uses the catalog voice.
- The agent refuses per-response `instructions`, `tools`, or `tool_choice` on `response.create` with error code `unsupported_response_override`.
- At most 8 Realtime sessions run at the same time.

For the full event reference, refer to [Use the OpenAI Realtime-Compatible Gateway](use-realtime-gateway.md).

## Deploy in Eval Mode

Eval mode deploys the `eval` profile for evaluation runs, such as a Tau evaluation. It uses the `frontend-backend-verdict-agent/single-gpu` recipe only. It uses the same image and code as the standard deployment and changes the settings in the following table.

| Setting | Eval mode value | Reason |
| --- | --- | --- |
| `FBV_PROFILE` | `eval` | The container serves OpenAI Realtime sessions only and refuses browser UI sessions. |
| `PIPELINE_TLS` | `false` | The server serves plain HTTP and `ws://`, so an evaluation client connects without a certificate. |
| `FBA_VOICE_EVENT_LOG` and `FBA_FILLER_LOG` | `/app/logs/fbv_voice_events.jsonl` and `/app/logs/fbv_filler.jsonl` | The agent writes its per-turn event log and its filler timing log. Evaluation metrics read both. |
| Volume | `./logs` mounted at `/app/logs` | The two logs reach the `logs/` directory of the repository on the host. |
| `FBV_REALTIME_TRACING` | `false` | Realtime sessions stay untraced. The `eval` profile also ignores every `FBV_REALTIME_*` switch. |
| `FBV_REALTIME_TURN_MODES` | `server_vad` | Turns end on the server VAD, as in every evaluation run. Clients cannot select manual or Smart Turn modes. |
| `FBV_REALTIME_MCP` | `false` | Realtime sessions accept function tools only and refuse `type: "mcp"` tools. |

The `docker/docker-compose.fbv-eval.yaml` override file holds these settings. Pass it after `docker-compose.yml` in every Compose command for the deployment. A command without it recreates the container with the standard settings and the `default` profile.

To deploy in eval mode, complete the following steps:

1. Complete [Configure the Environment](#configure-the-environment) and the `/single-gpu` preparation in [Start the Deployment Container](#start-the-deployment-container). You do not need to set `FBV_PROFILE`.

2. Create the host log directory, so that it belongs to your user:

    ```bash
    mkdir -p logs
    ```

3. Start the recipe with the override file. The `--build` flag rebuilds the app image when its Python packages are older than the code. Compose mounts `src/` from the repository, but the packages come from the image:

    ```bash
    docker compose -f docker-compose.yml -f docker/docker-compose.fbv-eval.yaml \
      --profile frontend-backend-verdict-agent/single-gpu up -d --build
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

6. Check the eval settings in the container. The command prints `eval`, `false`, and the two log paths:

    ```bash
    docker compose -f docker-compose.yml -f docker/docker-compose.fbv-eval.yaml \
      --profile frontend-backend-verdict-agent/single-gpu exec frontend-backend-verdict-agent-single-gpu \
      printenv FBV_PROFILE PIPELINE_TLS FBA_VOICE_EVENT_LOG FBA_FILLER_LOG
    ```

7. Connect one client to `ws://localhost:7860/v1/realtime?model=nvidia/nemotron-realtime-frontend-backend-verdict`, as [Connect a Realtime Client](#connect-a-realtime-client) shows. Then check that the event log grew on the host:

    ```bash
    tail -n 1 logs/fbv_voice_events.jsonl
    ```

    The `session_start` record of each session carries `adapter`, the Pipecat settings, the backend history and normalization settings, and the prompt catalog name and SHA-256 (`prompts`).

The container writes the log files as `root`. The repository ignores the `logs/` directory. To switch back to the standard deployment, run the `up -d` command from [Start the Deployment Container](#start-the-deployment-container) without the override file. Compose then recreates the container with the `default` profile, TLS, and no evaluation logs.

## Run a Tau Evaluation

The Tau evaluation harness, `voice-agent-evaluation`, is not part of this repository. Use the following steps to point it at this deployment:

1. Deploy in eval mode, as [Deploy in Eval Mode](#deploy-in-eval-mode) describes. Every Tau run must use `FBV_PROFILE=eval`. The run then uses the `eval` profile and the pinned Tau prompt catalog, every `FBV_REALTIME_*` extra stays off, and the agent's event and filler logs reach `logs/` on the host.
2. Keep the Compose keepalive values (`UVICORN_WS_PING_INTERVAL=0` and `UVICORN_WS_PING_TIMEOUT=20`).
3. Configure the harness with `execution.backend: evaluator_realtime` and `product.profile: openai`, the base URL `http://localhost:7860` or the WebSocket URL `ws://localhost:7860/v1/realtime`, and your `REALTIME_API_KEY`. The `tau_native` lane cannot select this model ID.
4. Run with a concurrency of 1. The `eval.yaml` profile asks for this until `server.max_sessions` is measured.
5. Record the profile and the prompt catalog SHA-256 from the app log with the results. The ready line shows `profile=eval`. Earlier runs recorded the profile name `tau3_eval_frontend_verdict_speak_history`, which had the same settings, prompts, and catalog SHA-256. The `session_start` record of the event log also carries the catalog name and SHA-256.

## Stop and Clean Up

Stop the recipe with the same Compose profile that started it:

```bash
docker compose --profile <compose-profile> down
```

For an eval-mode deployment, pass the override file as well:

```bash
docker compose -f docker-compose.yml -f docker/docker-compose.fbv-eval.yaml \
  --profile frontend-backend-verdict-agent/single-gpu down
```

If you started the deployment with the local LLM overlay, pass `-f docker-compose.yml -f docker/docker-compose.fbv-local-llm.yaml` to `down` as well. The command removes the containers. It keeps `.env`, the built image, the eval-mode logs in `logs/`, the NIM model caches, and the speech model weights in `models/nemo-speech`. Remove the weights only when you no longer need any single-GPU recipe that uses them.

## Troubleshooting

The following table lists common failures for these recipes. For general startup issues, refer to [Troubleshooting](../06-troubleshooting.md).

| Symptom | Cause | Fix |
| --- | --- | --- |
| A browser UI session ends at once or never starts, and the UI shows no error. The app log shows `runs with FBV_PROFILE=eval`. | The container runs the `eval` profile, which refuses browser UI sessions. | Start the recipe without the eval override file, so that it uses the `default` profile. Remove `FBV_PROFILE=eval` from `.env` if you set it there. |
| A browser UI session ends at once, and the app log shows `browser session refused`. | Either 8 browser UI sessions are already open, or the speech warm-up failed. | Close idle browser sessions. For a warm-up failure, fix the speech service as the warm-up row describes, and then start a new session. |
| A Realtime client receives an `error` event with code `server_error`, and the socket closes with code 1011. The app log names `FBV_PROFILE`. A browser UI session ends, and the app log shows the same error. | `FBV_PROFILE` has a value other than `default` or `eval`, for example an old `tau3_eval*` profile name. | Remove `FBV_PROFILE` from `.env`, or set it to `default` or `eval`, and then rerun the `up -d` command. Refer to [Choose a Profile](#choose-a-profile). |
| In eval mode, a Realtime client receives an `error` event with code `server_error`, and the socket closes with code 1011. The error says that an `llm_id` or `thinker_llm_id` is not available with `FBV_PROFILE=eval`. | The client's `ek_` secret selects an LLM other than the registry default, such as the local LLM. | Mint the secret without an LLM selection, or use the `REALTIME_API_KEY` Bearer credential. Evaluation runs use the default LLMs. |
| A Realtime client receives an `error` event with code `server_error`, and the socket closes with code 1011. The app log shows `WebSocket keepalive differs from the profile`. | The server keepalive does not match the `eval` profile, which every Realtime session uses. It requires `UVICORN_WS_PING_INTERVAL=0` and `UVICORN_WS_PING_TIMEOUT=20`. | Remove the two keys from `.env` or set them to `0` and `20`, and then rerun the `up -d` command. |
| A Realtime client receives an `error` event with code `server_busy`, and the socket closes with code 1013. | Either 8 Realtime sessions are already open (`server.max_sessions`), or the speech warm-up failed. The app log shows `speech warm-up failed` for a warm-up failure. | Close idle sessions, or wait and reconnect. For a warm-up failure, fix the speech service, and then reconnect. The next session retries the warm-up. |
| The app log shows `ASR warm-up failed against` or `TTS warm-up failed against`. | The speech service is not reachable or not ready. On `/single-gpu`, the `nemo-speech` sidecar is not running or has no model weights. On `/server`, a NIM sidecar is still starting. On the cloud recipe, NVCF rejected `NVIDIA_API_KEY` or is unreachable. | Run `docker compose --profile <compose-profile> logs <sidecar>` for the sidecar. If the `nemo-speech` weights are missing, run `bash scripts/download-nemo-speech-models.sh` as your user or set `NEMO_SPEECH_MODEL_LOC`. On the cloud recipe, if your key does not work for NVCF speech, use the `/server` or `/single-gpu` recipe. |
| The WebSocket handshake fails, and the server closes it with code 1008 and reason `realtime authentication failed`. | `REALTIME_API_KEY` is set, and the Realtime client sent no credential or a different one. | Send `Authorization: Bearer <your-realtime-api-key>` with the value from `.env`. |
| The session starts, but the agent does not answer. A Realtime client receives an `error` event with code `agent_error` on each turn. | `NVIDIA_API_KEY` is missing, is not an Inference Hub key, or the host cannot reach the Inference Hub. | Set `NVIDIA_API_KEY` to an NVIDIA Inference Hub key, check outbound HTTPS access, and then rerun the `up -d` command. |
| A browser UI session says that it cannot get the weather, a stock price, or a web result. | The tool's key (`WEATHERAPI_KEY`, `FINNHUB_API_KEY`, or `PERPLEXITY_API_KEY`) is not set, or the upstream service failed. | Set the key in `.env`, and then rerun the `up -d` command. |
| The browser does not ask for the microphone, or the microphone is blocked. | The page is served over plain HTTP from another machine. | Keep `PIPELINE_TLS=true` (the default) and open the `https://` URL. Do not use the eval override for browser sessions. |
| The download script fails, or `nemo-speech` cannot find its models. | `HF_TOKEN` is missing or invalid, or the weights are in another directory. | Set `HF_TOKEN` in `.env` and rerun the script as your user, or point `NEMO_SPEECH_MODEL_LOC` at the weights. |
| The session fails, and the app log shows an error that names `agent.overrides.prompts.path`. | `FBA_VOICE_PROMPTS` points at a missing or invalid prompt catalog. | Remove `FBA_VOICE_PROMPTS` from `.env` to use the profile's catalog, or point it at a valid catalog, and then rerun the `up -d` command. |
| The client fails with `ssl argument is incompatible with a ws:// URI`, or a `wss://` handshake fails with an SSL error. | The client and the server disagree on TLS. Eval mode serves plain `ws://`. | Use `ws://` and no TLS context for an eval-mode deployment. Use `wss://` for the standard deployment. |
| The app container exits at startup, and its log says `pipecat-ai <version> is installed, but this source tree requires <version>`. | The app image was built before a Pipecat upgrade. Compose mounts the current `src/`, but the Python packages come from the image. | Rebuild the image: add `--build` to the `up -d` command. |
| In eval mode, `logs/` stays empty or the container recreates without the eval settings. | A Compose command ran without `-f docker/docker-compose.fbv-eval.yaml`, or no session has connected yet. | Pass both `-f` files in every Compose command for the deployment, and connect one session. |
| `up -d` fails with `port is already allocated` or `address already in use`. | Another process uses the app port or a sidecar port. | Set `PIPELINE_APP_PORT` to a free port for the app. For a sidecar port, stop the other process or the other recipe that runs a speech sidecar. |

## Related Topics

- [Frontend/Backend Verdict Agent README](../../src/examples/frontend_backend_verdict/README.md)
- [Getting Started](../01-getting-started.md)
- [Use the OpenAI Realtime-Compatible Gateway](use-realtime-gateway.md)
- [Enable OpenTelemetry Tracing](enable-opentelemetry-tracing.md)
- [WebSocket Keepalive](tune-pipeline-performance.md#websocket-keepalive)
- [Troubleshooting](../06-troubleshooting.md)
