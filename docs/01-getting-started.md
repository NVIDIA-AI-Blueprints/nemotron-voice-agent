# Getting Started

This guide walks you through the cloud-only, server, and single-GPU deployment options for the Nemotron Voice Agent. The single-GPU recipes cover supported workstation GPUs, DGX Spark, and Jetson Thor.

## Prerequisites

Before you begin, ensure you have the following:

- Docker Compose v2.20 or newer. Check the version with `docker compose version`.
- **Cloud (`<example>`), Server (`*/server`), and Performance Server (`generic-assistant/server-perf`):** an NVIDIA API key (`NVIDIA_API_KEY`) from [build.nvidia.com](https://build.nvidia.com/). Server and Performance Server also need NGC access to pull NIM images. Refer to the [NGC Getting Started Guide](https://docs.nvidia.com/ngc/ngc-overview/index.html#registering-activating-ngc-account).
- **Single-GPU (`*/single-gpu`):** Python 3, either the Hugging Face `hf` CLI or `uvx`, and a [Hugging Face token](https://huggingface.co/docs/hub/en/security-tokens) (`HF_TOKEN`) for model downloads. `generic-assistant/single-gpu` also builds its diarization VAD model locally, which needs `uv`, or Python 3 with `gguf==0.19.0` and `numpy==2.2.6`. `NVIDIA_API_KEY` and `docker login nvcr.io` are not required for this family.
- **Dynamo (`*/dynamo`, experimental):** both `NVIDIA_API_KEY` with NGC access for the ASR and TTS NIMs, and `HF_TOKEN` for the Nemotron 3.5 Lightning weights. Refer to [Run Nemotron 3.5 Lightning on NVIDIA Dynamo](how-to/configure-llm.md#run-nemotron-35-lightning-on-nvidia-dynamo-experimental).

For cloud-only profiles, Docker and Docker Compose are sufficient. For local GPU profiles, install Docker with NVIDIA GPU support and verify `nvidia-smi` works inside containers. Refer to the [NVIDIA Container Toolkit installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

## Docker based Deployment

Each example ships as Docker Compose **profiles**. Pick exactly one per deployment. The bare **`<example>`** profile runs cloud-only (no local GPU, using NVIDIA cloud API endpoints). **`<example>/server`** is the scaling-oriented NIM stack on a **workstation** (not DGX Spark, not Jetson Thor). **`<example>/single-gpu`** is the one-GPU path. Supported hardware varies by example and is listed below. `docker compose up` with no profile is intentionally a no-op so the deployment is always explicit.

> **Note:** For example-specific architecture, configuration, and tunables, see each example's README (linked in the table below).

| Example | Description | Supported profiles |
|---------|-------------|--------------------|
| [`generic-assistant`](../src/examples/generic/README.md) | Baseline English-only cascaded pipeline (Nemotron ASR + LLM + Magpie TTS) | `generic-assistant`, `generic-assistant/server`, `generic-assistant/single-gpu` (workstation, DGX Spark, Jetson Thor), `generic-assistant/dynamo` (workstation, experimental) |
| [`multilingual-assistant`](../src/examples/multilingual/README.md) | Multilingual cascaded pipeline with a fixed language per session | `multilingual-assistant`, `multilingual-assistant/server`, `multilingual-assistant/single-gpu` (workstation, DGX Spark, Jetson Thor), `multilingual-assistant/dynamo` (workstation, experimental) |
| [`omni-assistant`](../src/examples/omni_assistant/README.md) | Nemotron Omni model (ASR + LLM) + Magpie TTS cascaded pipeline | `omni-assistant`, `omni-assistant/server`, `omni-assistant/single-gpu` (workstation, DGX Spark, Jetson Thor) |
| [`omni-assistant-subagents`](../src/examples/omni_assistant_subagents/README.md) | Multi-agent Omni with media + live-webcam understanding | `omni-assistant-subagents`, `omni-assistant-subagents/server`, `omni-assistant-subagents/single-gpu` (workstation, DGX Spark) |
| [`frontend-backend-live`](../src/examples/frontend_backend_live/README.md) | Routing talker with the `/v1/live` session API and three backend adapters: the backend LLM, an LLM agent, or an agent runtime harness (Bluebird Cafe reference) | `frontend-backend-live`, `frontend-backend-live/server`, `frontend-backend-live/single-gpu` (workstation, DGX Spark, Jetson Thor) |

> Observability overlays `tracing` (Phoenix OTel) and Coturn Server `turn` can be added to any profile.

---

### Deployment Steps

1. Clone the repository and navigate to the root directory.

    ```bash
    git clone git@github.com:NVIDIA-AI-Blueprints/nemotron-voice-agent.git
    cd nemotron-voice-agent
    ```

2. Configure the environment. Copy the example environment file [.env.example](../.env.example) only if `.env` is missing. Set the key for the recipe family you will run. Do not mix these.

    | Recipe family | Required `.env` key | `docker login nvcr.io` |
    | --- | --- | --- |
    | Cloud (`<example>`) | `NVIDIA_API_KEY` | No |
    | Server (`<example>/server`) | `NVIDIA_API_KEY` | Yes, before `up` |
    | Performance Server (`generic-assistant/server-perf`) | `NVIDIA_API_KEY` | Yes, before `up` |
    | Single-GPU (`<example>/single-gpu`) | `HF_TOKEN` | No |
    | Dynamo (`<example>/dynamo`) | `NVIDIA_API_KEY` and `HF_TOKEN` | Yes, before `up` |

    `NVIDIA_API_KEY` is required for cloud-only, Server, and Performance Server. `HF_TOKEN` is required for Single-GPU. Do not mix these, except for Dynamo, which needs both.

    ```bash
    test -f .env || cp .env.example .env
    ```

    Docker Compose passes `.env` values into the app and model sidecars, so exporting a key in your shell is not enough for runtime.

3. For **`*/server`, `*/dynamo`, and `generic-assistant/server-perf` only**, log in to the NVIDIA NGC Docker Registry:

    ```bash
    set -a; . ./.env; set +a
    printf '%s' "$NVIDIA_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin
    ```

    Skip this step for cloud-only and `*/single-gpu`.

4. Deploy the example profile of your choice.

    **4.1 Cloud only** (no local GPU):

    ```bash
    docker compose --profile generic-assistant up -d            # Generic Cascaded
    docker compose --profile multilingual-assistant up -d       # Multilingual Cascaded
    docker compose --profile omni-assistant up -d               # Nemotron Omni Assistant
    docker compose --profile omni-assistant-subagents up -d     # Nemotron Omni Assistant Subagents
    docker compose --profile frontend-backend-live up -d        # Frontend/Backend Live Cafe Assistant
    ```

    **4.2 Server** (workstation NIM stack, not DGX Spark or Jetson Thor):

    ```bash
    docker compose --profile generic-assistant/server up -d         # Generic Cascaded
    docker compose --profile multilingual-assistant/server up -d    # Multilingual Cascaded
    docker compose --profile omni-assistant/server up -d            # Nemotron Omni Assistant
    docker compose --profile omni-assistant-subagents/server up -d  # Nemotron Omni Assistant Subagents
    docker compose --profile frontend-backend-live/server up -d     # Frontend/Backend Live Cafe Assistant
    ```

    The `frontend-backend-live/server` recipe runs its talker on the Nemotron 3.5 Lightning NIM on GPU `0` and its backend on the Nemotron 3.5 Super VL NIM with reasoning enabled, which it starts on GPUs `1` and `2`, so it needs three GPUs. When the Super VL NIM is not running, the backend runs on the Lightning NIM. Refer to the example's [Default Models](../src/examples/frontend_backend_live/README.md#default-models).

    **4.3 Single GPU** (one supported GPU). Hardware support varies by example as listed above. `omni-assistant-subagents/single-gpu` is not supported on Jetson Thor. Cascaded recipes run NeMo-Speech.cpp next to Nemotron 3.5 Lightning. Omni recipes retain the multimodal Omni model and use NeMo-Speech.cpp for TTS. The Lightning container loads the NVFP4 checkpoint on every supported GPU. Hopper and Ada serve those weights as W4A16 through Marlin. Blackwell workstations enable DFlash speculative decoding automatically. DGX Spark does not load a DSpark draft model due to a known issue with vLLM streaming prefill. Follow the [Jetson Thor guide](03-jetson-thor.md) when applicable.

    Download the NeMo-Speech.cpp weights **as your user** (do not use `sudo`), and rerun this step after upgrading. The script reads `HF_TOKEN` from `.env`, pulls the GGUFs from Hugging Face, and fetches Magpie TTS text-normalization grammars from the [NeMo-Speech.cpp GitHub release](https://github.com/NVIDIA/NeMo-Speech.cpp/releases) into `models/nemo-speech`. For `generic-assistant/single-gpu`, it also installs the diarization and Silero VAD models, which the speech container loads even when diarization is off. A failed VAD build only prints a warning, so check the output before you start that recipe:

    ```bash
    bash scripts/download-nemo-speech-models.sh
    ```

    Then start the stack:

    ```bash
    docker compose --profile generic-assistant/single-gpu up -d          # Generic Cascaded
    docker compose --profile multilingual-assistant/single-gpu up -d     # Multilingual Cascaded
    docker compose --profile omni-assistant/single-gpu up -d             # Nemotron Omni Assistant
    docker compose --profile omni-assistant-subagents/single-gpu up -d   # Omni Assistant Subagents (workstation / DGX Spark)
    docker compose --profile frontend-backend-live/single-gpu up -d      # Frontend/Backend Live
    ```

    To verify all services are healthy, run `docker compose ps`.

    > **Note:** Each Docker Compose profile pins `EXAMPLE_SELECTION=<example>`, so the container runs that single example. Set `EXAMPLE_SELECTION=all` to expose every example in the UI selector instead.
    >
    > **Note:** The `frontend-backend-live` profiles serve the Live session routes (`/v1/live`) when you set `LIVE_API_TOKEN` in `.env` and keep the profile's `EXAMPLE_SELECTION`, and every route then requires that token. With `EXAMPLE_SELECTION=all`, no Live route is served. To serve the routes without a token, for example on a trusted network, set `LIVE_ENABLED=true` instead. Anyone who can reach the server by an IP address, `localhost`, or a host name in `allowed_hosts` can then use them. For details, refer to [When the Routes Are Served](how-to/use-live-gateway.md#when-the-routes-are-served).
    >
    > **Note:** The `frontend-backend-live` profiles connect the backend LLM directly by default. To use another backend adapter, set `core.backend.adapter` in the `config` block of the `frontend-backend-live` entry in `examples_registry.yaml`, for example to `llm-agent` for the backend LLM agent, together with `core.delegation.result_format: speak_display`, and restart the app service. For the adapters, refer to [Choose a Backend Adapter](../src/examples/frontend_backend_live/README.md#choose-a-backend-adapter), and for connecting an agent such as Hermes, refer to [Connect an Agent Harness](how-to/connect-an-agent-harness.md).
    >
    > **Note:** First-run deployment can take 30–60 minutes. On local recipes, the **first voice interaction** may also lag while GPU sidecars warm up. Later turns are much faster.

5. Access the application at `https://<machine-ip>:7860` (HTTPS by default, which browser microphone and WebRTC require).

    > **Note:** `PIPELINE_TLS=false` serves plain HTTP for headless/API testing only. For plain-HTTP browser testing, see [plain-HTTP deployment and usage](06-troubleshooting.md#browser-access).
    >
    > **Tip:** For the best experience, we recommend using a headset (preferably wired) instead of your laptop's built-in microphone.
    >
    > **Note:** If connecting from a different network (NAT, cloud, restrictive firewall), see [Enable a TURN Server for Remote Access](how-to/enable-turn-server.md).

---

## Optional: Local Development (without Docker)

For development and debugging, you can run the server directly:

1. Install [uv](https://docs.astral.sh/uv/) and Node.js 20+.

2. Install dependencies and build the client:

    ```bash
    uv sync --group dev
    cd client && npm install && npm run build && cd ..
    ```

3. Install local commit hooks:

    ```bash
    uv run --project . --group dev pre-commit install
    ```

    The hooks run formatting and linting checks on staged files during `git commit`.

4. Configure the environment:

    ```bash
    cp .env.example .env
    # Cloud or */server (including server-perf): set NVIDIA_API_KEY
    # */single-gpu sidecars: set HF_TOKEN instead
    ```

5. Start the server:

    ```bash
    uv run python src/server.py --host 0.0.0.0 --port 7860
    ```

    To serve plain HTTP instead of HTTPS, set `PIPELINE_TLS=false` in `.env` or prefix the command:

    ```bash
    PIPELINE_TLS=false uv run python src/server.py --host 0.0.0.0 --port 7860
    ```

    Host-native runs read [`examples_registry.yaml`](../examples_registry.yaml) at the repository root. Edit the `selection` field to choose what the UI exposes, then start the server normally. The server has no example-selection CLI flag. Pipeline options such as `--prompt-file` remain available.

    Without reachable local sidecars, a host-native server uses the NVIDIA Cloud (NVCF) service endpoints, which need a real `NVIDIA_API_KEY` (not empty or `not-needed`). To run against **local on-prem services**, start the matching Compose sidecars first, then start the app. The backend detects whether the NIM (`/server`) or NeMo-Speech.cpp (`/single-gpu`) sidecars are reachable and rewrites Compose endpoints to `localhost` automatically. Set `SERVICE_RECIPE=server` or `SERVICE_RECIPE=single-gpu` to pin the section instead.

    | `selection` in `examples_registry.yaml` | UI behavior |
    |-----------------------------------------|-------------|
    | `all` | Show every registered example (default) |
    | `generic-assistant` | Lock to Generic Assistant |
    | `multilingual-assistant` | Lock to Multilingual Assistant |
    | `omni-assistant` | Lock to Nemotron Omni Assistant |
    | `omni-assistant-subagents` | Lock to Nemotron Omni Assistant Subagents |
    | `frontend-backend-live` | Lock to Frontend Backend Live |

    > **Note:** Docker Compose deployments pin `EXAMPLE_SELECTION=<example>` to a single example. You can set `EXAMPLE_SELECTION=all` to expose every example in the UI selector instead.

6. Access the application locally at `https://localhost:7860`, or from another machine at
   `https://<machine-ip>:7860` (replace `<machine-ip>` with the host IP).

   > **Tip:** For the best experience, we recommend using a headset (preferably wired) instead of your laptop's built-in microphone.
