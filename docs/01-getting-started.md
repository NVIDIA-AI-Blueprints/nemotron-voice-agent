# Getting Started

Deploy a voice agent and test a first voice turn. Start with a Cloud recipe if you do not have a supported local GPU. The Docker Compose procedure below is for Pipecat; LiveKit has its own browser workflow.

## Prerequisites

You need Docker Compose 2.24.4 or newer. Check your version with `docker compose version`. For a local GPU recipe, install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) and verify that containers can access the GPU.

Choose credentials for your recipe:

| Recipe Family | Key in `.env` | NGC Registry Login |
| --- | --- | --- |
| Cloud (`<example>`) | `NVIDIA_API_KEY` from [build.nvidia.com](https://build.nvidia.com/) | Not required |
| Server (`<example>/server`) | `NVIDIA_API_KEY` with [NGC access](https://docs.nvidia.com/ngc/ngc-overview/index.html#registering-activating-ngc-account) | Required |
| Single-GPU (`<example>/single-gpu`) | [Hugging Face token](https://huggingface.co/docs/hub/en/security-tokens) (`HF_TOKEN`) | Not required |
| Dynamo (`<example>/dynamo`) | `NVIDIA_API_KEY` with NGC access for the speech NIMs **and** `HF_TOKEN` for the Lightning weights | Required |

Single-GPU recipes also need NVIDIA driver 580.x or newer for NeMo-Speech.cpp. The speech download script needs Python 3 and either the Hugging Face `hf` CLI or `uvx`. The Generic Assistant also needs `uv`, or Python with `gguf==0.19.0` and `numpy==2.2.6`, to build the voice activity detection (VAD) model used by diarization.

## Docker Based Deployment

Choose exactly one recipe per deployment. Cloud uses NVIDIA-hosted endpoints without a local GPU. Server runs NVIDIA NIM microservices on a workstation, not DGX Spark or Jetson Thor. Single-GPU uses local models; check [model sizing](how-to/configure-llm.md#vram--hardware-support) before starting. Dynamo requires an Ada, Hopper, or Blackwell workstation GPU with at least 64 GB for one worker. Refer to [Run Nemotron 3.5 Lightning on NVIDIA Dynamo](how-to/configure-llm.md#run-nemotron-35-lightning-on-nvidia-dynamo) for sizing and extra workers.

Choose a profile from the following table. Each linked example lists its default models and sidecars.

| Example | Cloud | Server | Single-GPU | Dynamo |
| --- | --- | --- | --- | --- |
| [Generic Assistant](../src/nemotron_voice_agent/pipecat/examples/generic/README.md) | `generic-assistant` | `generic-assistant/server` | `generic-assistant/single-gpu` | `generic-assistant/dynamo` |
| [Multilingual Assistant](../src/nemotron_voice_agent/pipecat/examples/multilingual/README.md) | `multilingual-assistant` | `multilingual-assistant/server` | `multilingual-assistant/single-gpu` | `multilingual-assistant/dynamo` |
| [Omni Assistant](../src/nemotron_voice_agent/pipecat/examples/omni_assistant/README.md) | `omni-assistant` | `omni-assistant/server` | `omni-assistant/single-gpu` | — |
| [Omni Assistant Subagents](../src/nemotron_voice_agent/pipecat/examples/omni_assistant_subagents/README.md) | `omni-assistant-subagents` | `omni-assistant-subagents/server` | `omni-assistant-subagents/single-gpu` (workstation or DGX Spark only) | — |
| [Frontend/Backend Live](../src/nemotron_voice_agent/pipecat/examples/frontend_backend_live/README.md) | `frontend-backend-live` | `frontend-backend-live/server` (three GPUs for the stock recipe) | `frontend-backend-live/single-gpu` | — |

Single-GPU profiles support compatible workstations and DGX Spark. For Jetson Thor, follow the [Jetson Thor guide](03-jetson-thor.md); Omni Assistant Subagents is Cloud only on Thor. For the four-GPU `generic-assistant/server-perf` load test, follow the [scaling benchmark](../benchmarking_tools/scaling-perf/README.md#reproducing-the-recommended-scaling-setup) instead of the browser procedure.

### Deployment Steps

Run the following steps for your selected Pipecat profile.

1. Clone the repository.

    ```bash
    git clone https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent.git
    ```

2. Change to the repository root.

    ```bash
    cd nemotron-voice-agent
    ```

3. Create `.env` only if it does not exist.

    ```bash
    test -f .env || cp .env.example .env
    ```

4. Set the key or keys for your recipe in `.env`. An exported shell variable alone does not populate the app's `.env` file.
5. [Log in to the NGC registry](#log-in-to-the-ngc-registry) for Server or Dynamo only.

6. Download the speech weights for Single-GPU only. Run the script as your user, not with `sudo`, and rerun it after upgrading to refresh the assets. Omni recipes use the speech sidecar for text-to-speech (TTS) only.

    ```bash
    bash scripts/download-nemo-speech-models.sh
    ```

7. For `generic-assistant/single-gpu`, check the download output for VAD build warnings before starting. The ASR sidecar loads the diarization and VAD models even when speaker labels are off.
8. Start your selected recipe. Substitute one profile from the table for `<recipe>`.

    ```bash
    docker compose --profile <recipe> up -d
    ```

    The first start can take time to build images, download models, and load weights. Most Pipecat profiles default `EXAMPLE_SELECTION` to the matching example. The performance profile hard-codes `generic-assistant`. Setting `EXAMPLE_SELECTION=all` on other profiles exposes the selector but does not start other examples' sidecars or serve `/v1/live`.

9. Check container status.

    ```bash
    docker compose --profile <recipe> ps
    ```

    Wait for the app to become healthy. A healthy app does not guarantee that all model services are ready. For startup failures, refer to [Troubleshooting](06-troubleshooting.md).

10. Open `https://localhost:7860/` on the Docker host, or `https://<machine-ip>:7860/` from another machine. If you changed `PIPELINE_APP_PORT`, use that host port. For TLS warnings or microphone problems, refer to [Browser Access](06-troubleshooting.md#browser-access). Remote WebRTC clients can need [a TURN server](how-to/enable-turn-server.md).
11. Select **Connect**. The default local services must be ready before a session can start.
12. Allow microphone access if prompted.
13. Ask a question after the greeting finishes. You should hear a spoken reply. The first turn on a local recipe can take longer while the models warm up.
14. Select **Disconnect** to end the session.
15. Stop the deployment with the same profile. Include any overlays you started with it.

    ```bash
    docker compose --profile <recipe> down
    ```

For application clients, follow [Use the Live Gateway](how-to/use-live-gateway.md) or [Use the Realtime Gateway](how-to/use-realtime-gateway.md). Neither is required to use the repository browser UI.

### Log In to the NGC Registry

Run this command in Bash before starting a recipe that pulls NGC images. At the hidden prompt, enter your NGC API key. The command passes it to Docker through standard input without parsing `.env` or saving the key in shell history.

```bash
(
  IFS= read -rs -p 'NGC API key: ' ngc_key || exit 1
  printf '\n' >&2
  printf '%s\n' "$ngc_key" | docker login nvcr.io --username '$oauthtoken' --password-stdin
)
```

Expect `Login Succeeded`. Registry login does not replace the `NVIDIA_API_KEY` setting in `.env`.

## Deploy the Generic Assistant on LiveKit

The LiveKit example runs its own server, agent worker, and web UI. Its [deployment guide](../src/nemotron_voice_agent/livekit/examples/generic_assistant/README.md#run-the-example) covers choosing one recipe, waiting for agent readiness, connecting to the UI at `http://localhost:3000/`, and using an SSH tunnel for a remote browser. Stop any Pipecat recipe before starting LiveKit on the same host to avoid resource or port conflicts. SIP is an optional [Linux-only setup](../src/nemotron_voice_agent/livekit/examples/generic_assistant/SIP.md).

## Optional: Local Development Without Docker

Run the Pipecat app directly with NVIDIA Cloud endpoints. You need Python 3.12 or 3.13, [uv](https://docs.astral.sh/uv/), and an NVIDIA Cloud API key. For the Pipecat web UI, use Node.js 22.14 or a later 22.x version to satisfy the engine declared by the locked `@daily-co/daily-js` dependency. Pipecat CI uses Node.js 20; the 22.x guidance is a local recommendation, not a minimum established by CI. Run the commands from the repository root. Stop any Compose app using port `7860` first.

1. Install Python dependencies from the lockfile.

    ```bash
    uv sync --locked
    ```

2. Install the Pipecat client dependencies.

    ```bash
    npm --prefix clients/pipecat ci
    ```

3. Build the web UI.

    ```bash
    npm --prefix clients/pipecat run build
    ```

4. Create `.env` only if it does not exist.

    ```bash
    test -f .env || cp .env.example .env
    ```

5. Set `NVIDIA_API_KEY` and `SERVICE_RECIPE=cloud` in `.env`. This keeps the app on NVIDIA Cloud endpoints even when local model services are running.
6. Set `pipecat.selection` to `generic-assistant` in [`examples_registry.yaml`](../examples_registry.yaml) for this first run. The registry defaults to `all`, which exposes the example selector. An `EXAMPLE_SELECTION` environment override takes precedence. There is no example-selection command-line flag.
7. Start the Pipecat server.

    ```bash
    uv run nva-pipecat
    ```

    The default address is `https://localhost:7860/`. For a browser on another machine, start with `--host 0.0.0.0` and use the server address instead of `localhost`.

8. Open the UI in your browser.
9. Select **Connect**.
10. Allow microphone access if prompted.
11. Ask a question after the greeting. The assistant should answer aloud.
12. Select **Disconnect** to end the session.
13. Press Ctrl+C in the server terminal to stop the app.

When you switch to local models or a Dynamo recipe, remove `SERVICE_RECIPE=cloud` from `.env` to restore automatic detection. For a host-native app, start only the model services from your example's Compose recipe so the app port stays free. Refer to [How the Active Recipe Selects Entries](how-to/configure-services.md#how-the-active-recipe-selects-entries).
